//! Rust dispatch wrapper for the GPU soft-on-soft VT contact pipeline
//! — Stage 33.2.
//!
//! Five compute pipelines built from `shaders/soft_vt_contact.wgsl`:
//!   - `compute_tri_aabbs` — writes per-tri AABBs into a buffer the
//!     LBVH consumes (refit input).
//!   - `gather_vt_aabbs`   — writes per-vert inflated AABBs (book-
//!     keeping for v1; the gather pass walks the BVH directly).
//!   - `gather_vt`         — DFS-walks the LBVH per particle, emits
//!     `VtContactGpu` records into the output buffer.
//!   - `apply_vt_penalty`  — atomically accumulates per-particle
//!     position deltas into `particle_dq` (a scratch buffer of the
//!     same size as `particle_q`).
//!   - `commit_dq`         — adds `particle_dq` into `particle_q` and
//!     zeroes `particle_dq` for the next round.
//!
//! Caller responsibility (orchestrated from `solvers/vbd.rs::step`):
//!   1. `kernel.dispatch_compute_tri_aabbs(...)` — populates
//!      `soft_bvh.lbvh().primitive_aabb_buffer()`.
//!   2. `soft_bvh.refit(model)` — runs the LBVH refit on those AABBs.
//!   3. `kernel.dispatch_gather_and_apply(...)` — runs gather → apply
//!      → commit_dq end-to-end with no host bounce.
//!
//! All entry points share the same set of bind groups so a single
//! pipeline layout serves them all.

use std::sync::Arc;

use crate::core::{Error, Result};
use crate::sim::{Model, State};

use super::{LbvhBroadPhase, SoftBvhGpu};

/// Per-dispatch uniforms — must match the `Uniforms` struct in
/// `shaders/soft_vt_contact.wgsl`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    particle_count: u32,
    tri_count: u32,
    surface_count: u32,
    contact_count: u32,
    margin: f32,
    k_contact: f32,
    out_capacity: u32,
    primitive_count: u32,
}

/// One emitted VT contact (GPU mirror of [`crate::solvers::VtContact`]).
/// 48 bytes — `(u32, u32, u32, u32) + vec4<f32> + vec4<f32>`.
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VtContactGpu {
    pub vertex_global_id: u32,
    pub vertex_surface: u32,
    pub tri_global_id: u32,
    pub tri_surface: u32,
    /// `(b0, b1, b2, depth)` — barycentric weights + penetration depth.
    pub bary: [f32; 4],
    /// `(nx, ny, nz, _pad)` — outward contact normal.
    pub normal: [f32; 4],
}

/// Stats surfaced after a [`SoftVtContactKernel::dispatch_gather_and_apply`]
/// run. Mirrors the diagnostics the CPU `project_vt_contacts` returns.
#[derive(Copy, Clone, Debug, Default)]
pub struct SoftVtContactStats {
    /// Number of contacts emitted by `gather_vt` (clamped to
    /// `out_capacity` if the buffer overflowed).
    pub contact_count: u32,
    /// `true` if the contact buffer was too small. Caller should
    /// resize for the next dispatch.
    pub overflowed: bool,
}

const WORKGROUP_SIZE: u32 = 64;

/// GPU dispatch driver for the soft-VT contact pipeline.
pub struct SoftVtContactKernel {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,

    pipe_compute_tri_aabbs: wgpu::ComputePipeline,
    pipe_gather_vt_aabbs: wgpu::ComputePipeline,
    pipe_gather_vt: wgpu::ComputePipeline,
    pipe_apply_vt_penalty: wgpu::ComputePipeline,
    pipe_commit_dq: wgpu::ComputePipeline,

    layout_state: wgpu::BindGroupLayout,
    layout_topo: wgpu::BindGroupLayout,
    layout_aabbs: wgpu::BindGroupLayout,
    layout_bvh: wgpu::BindGroupLayout,
    layout_contacts: wgpu::BindGroupLayout,

    /// `particle_dq` scratch — atomic-add accumulator. Same layout as
    /// `particle_q` (`vec4<f32>` per particle, 16 bytes); bound as
    /// `array<atomic<u32>>` from the WGSL side.
    particle_dq: wgpu::Buffer,
    /// Per-vertex inflated AABBs (point ± margin). `particle_count`
    /// `Aabb` slots.
    vert_aabbs: wgpu::Buffer,
    /// VT contact output. Sized for `vertex_count * 8` worst-case.
    out_contacts: wgpu::Buffer,
    /// Atomic contact-emit counter. Slot 0 is the high-water mark.
    out_contact_count: wgpu::Buffer,

    /// Sized arrays for buffer-presence checks.
    particle_count: u32,
    out_capacity: u32,
    /// Cached uniform buffer — re-uploaded per dispatch.
    uniforms: wgpu::Buffer,
}

impl SoftVtContactKernel {
    /// Build the kernel for a fixed model topology. Allocates the
    /// per-particle scratch + per-particle AABB + contact output
    /// buffers; reuses across substeps.
    pub fn new(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        model: &Model,
    ) -> Result<Self> {
        let particle_count = model.particle_count as u32;
        // Worst-case contact density: 8 contacts per particle (the CPU
        // dedup keeps at most one per other-surface and v1 caps
        // surface_count at 8). Bumped to a hard floor of 1024 so even
        // tiny models have a meaningful output buffer.
        let out_capacity = (particle_count.saturating_mul(8)).max(1024);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::soft_vt_contact"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/soft_vt_contact.wgsl").into(),
            ),
        });

        // ── Bind-group layouts ─────────────────────────────────────
        // Group 0: particle_q (rw), particle_inv_mass (ro), particle_dq (rw atomic)
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::soft_vt_contact state"),
            entries: &[
                rw_storage(0),
                ro_storage(1),
                rw_storage(2),
            ],
        });
        // Group 1: soft_surface_tris (ro), soft_surfaces (ro)
        let layout_topo = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::soft_vt_contact topo"),
            entries: &[ro_storage(0), ro_storage(1)],
        });
        // Group 2: tri_aabbs (rw), vert_aabbs (rw)
        let layout_aabbs = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::soft_vt_contact aabbs"),
            entries: &[rw_storage(0), rw_storage(1)],
        });
        // Group 3: bvh_nodes (ro), bvh_sorted_idx (ro), bvh_pair_count (ro)
        let layout_bvh = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::soft_vt_contact bvh"),
            entries: &[ro_storage(0), ro_storage(1), ro_storage(2)],
        });
        // Group 4: out_contacts (rw), out_contact_count (rw atomic), uniforms
        let layout_contacts = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::soft_vt_contact out"),
            entries: &[
                rw_storage(0),
                rw_storage(1),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::soft_vt_contact layout"),
            bind_group_layouts: &[
                &layout_state,
                &layout_topo,
                &layout_aabbs,
                &layout_bvh,
                &layout_contacts,
            ],
            push_constant_ranges: &[],
        });

        let make = |entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::soft_vt_contact"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let pipe_compute_tri_aabbs = make("compute_tri_aabbs");
        let pipe_gather_vt_aabbs = make("gather_vt_aabbs");
        let pipe_gather_vt = make("gather_vt");
        let pipe_apply_vt_penalty = make("apply_vt_penalty");
        let pipe_commit_dq = make("commit_dq");

        // ── Owned scratch buffers ──────────────────────────────────
        let dq_bytes = (particle_count as u64).max(1) * 16;
        let particle_dq = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::soft_vt_contact particle_dq"),
            size: dq_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // Zero-init dq so the first commit_dq pass is a no-op.
        queue.write_buffer(&particle_dq, 0, &vec![0u8; dq_bytes as usize]);

        let vert_aabbs_bytes = (particle_count as u64).max(1) * 32;
        let vert_aabbs = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::soft_vt_contact vert_aabbs"),
            size: vert_aabbs_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let out_contacts_bytes = out_capacity as u64 * core::mem::size_of::<VtContactGpu>() as u64;
        let out_contacts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::soft_vt_contact out_contacts"),
            size: out_contacts_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let out_contact_count = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::soft_vt_contact out_contact_count"),
            size: 16,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::soft_vt_contact uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            device,
            queue,
            pipe_compute_tri_aabbs,
            pipe_gather_vt_aabbs,
            pipe_gather_vt,
            pipe_apply_vt_penalty,
            pipe_commit_dq,
            layout_state,
            layout_topo,
            layout_aabbs,
            layout_bvh,
            layout_contacts,
            particle_dq,
            vert_aabbs,
            out_contacts,
            out_contact_count,
            particle_count,
            out_capacity,
            uniforms,
        })
    }

    /// Borrow the per-particle delta scratch buffer. Useful for tests
    /// that want to read it back after `apply_vt_penalty`.
    pub fn particle_dq_buffer(&self) -> &wgpu::Buffer {
        &self.particle_dq
    }

    /// Borrow the GPU contact output buffer.
    pub fn out_contacts_buffer(&self) -> &wgpu::Buffer {
        &self.out_contacts
    }

    /// Borrow the atomic contact-emit counter buffer (slot 0 = count).
    pub fn out_contact_count_buffer(&self) -> &wgpu::Buffer {
        &self.out_contact_count
    }

    /// Worst-case output contact buffer capacity.
    pub fn out_capacity(&self) -> u32 {
        self.out_capacity
    }

    /// Run the per-substep `compute_tri_aabbs` pass. Writes per-tri
    /// AABBs into `soft_bvh.lbvh().primitive_aabb_buffer()` so the
    /// LBVH refit can consume them without a host bounce.
    pub fn dispatch_compute_tri_aabbs(
        &self,
        model: &Model,
        state: &State,
        soft_bvh: &SoftBvhGpu,
    ) -> Result<()> {
        if model.soft_surface_tris.is_empty() {
            return Ok(());
        }
        let tri_count = soft_bvh.tri_count() as u32;
        if tri_count == 0 {
            return Ok(());
        }
        // Uniforms — only the tri_count is used by this entry point.
        self.write_uniforms(model, 0.0, 0.0, tri_count)?;

        // The `compute_tri_aabbs` shader writes to bind-group-2 binding
        // 0 (`tri_aabbs`); bind that to the LBVH's primitive_aabb so
        // the next refit consumes it directly.
        let bg_state = self.bind_state(model, state)?;
        let bg_topo = self.bind_topo(model)?;
        let bg_aabbs = self.bind_aabbs_with_lbvh(soft_bvh.lbvh());
        let bg_bvh = self.bind_bvh(soft_bvh.lbvh());
        let bg_contacts = self.bind_contacts();

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::soft_vt_contact compute_tri_aabbs encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::soft_vt_contact::compute_tri_aabbs"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_compute_tri_aabbs);
            pass.set_bind_group(0, &bg_state, &[]);
            pass.set_bind_group(1, &bg_topo, &[]);
            pass.set_bind_group(2, &bg_aabbs, &[]);
            pass.set_bind_group(3, &bg_bvh, &[]);
            pass.set_bind_group(4, &bg_contacts, &[]);
            let groups = tri_count.div_ceil(WORKGROUP_SIZE).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Run the gather → apply → commit_dq pipeline end-to-end. Writes
    /// the contact response into `state.particle_q` via `particle_dq`
    /// (no host bounce on `particle_q`). Returns the emitted contact
    /// count.
    ///
    /// One small device→host sync between `gather_vt` and the
    /// `apply` dispatch is unavoidable: we need to know how many
    /// contacts to dispatch `apply_vt_penalty` over. The 4-byte
    /// readback is the only host bounce per substep — orders of
    /// magnitude cheaper than the Stage 20.4
    /// `particle_q.download → upload` we replaced.
    ///
    /// Caller must dispatch [`Self::dispatch_compute_tri_aabbs`] +
    /// [`SoftBvhGpu::refit`] FIRST so the LBVH is up to date.
    pub fn dispatch_gather_and_apply(
        &self,
        model: &Model,
        state: &State,
        soft_bvh: &SoftBvhGpu,
        margin: f64,
        k_contact: f64,
    ) -> Result<SoftVtContactStats> {
        if model.soft_surface_tris.is_empty() || model.soft_surfaces.is_empty() {
            return Ok(SoftVtContactStats::default());
        }
        let tri_count = soft_bvh.tri_count() as u32;
        if tri_count == 0 || self.particle_count == 0 {
            return Ok(SoftVtContactStats::default());
        }

        // Reset out_contact_count[0] to zero before gather.
        self.queue
            .write_buffer(&self.out_contact_count, 0, bytemuck::cast_slice(&[0u32; 4]));

        let bg_state = self.bind_state(model, state)?;
        let bg_topo = self.bind_topo(model)?;
        let bg_aabbs = self.bind_aabbs_with_lbvh(soft_bvh.lbvh());
        let bg_bvh = self.bind_bvh(soft_bvh.lbvh());
        let bg_contacts = self.bind_contacts();

        let particle_groups = self.particle_count.div_ceil(WORKGROUP_SIZE).max(1);

        // ── Pass 1: gather_vt → out_contacts ───────────────────────
        // Uniforms — gather only consults `particle_count`, `margin`,
        // `surface_count`, `out_capacity`, `primitive_count`.
        let u_gather = Uniforms {
            particle_count: self.particle_count,
            tri_count,
            surface_count: model.soft_surfaces.len() as u32,
            contact_count: 0,
            margin: margin as f32,
            k_contact: k_contact as f32,
            out_capacity: self.out_capacity,
            primitive_count: tri_count,
        };
        self.queue
            .write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u_gather));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::soft_vt_contact gather encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::soft_vt_contact::gather_vt"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_gather_vt);
            pass.set_bind_group(0, &bg_state, &[]);
            pass.set_bind_group(1, &bg_topo, &[]);
            pass.set_bind_group(2, &bg_aabbs, &[]);
            pass.set_bind_group(3, &bg_bvh, &[]);
            pass.set_bind_group(4, &bg_contacts, &[]);
            pass.dispatch_workgroups(particle_groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // 4-byte readback — the only per-substep host bounce.
        let count_raw = self.read_contact_count()?;
        let overflowed = count_raw > self.out_capacity;
        let count = count_raw.min(self.out_capacity);
        if count == 0 {
            return Ok(SoftVtContactStats {
                contact_count: 0,
                overflowed,
            });
        }

        // ── Pass 2 + 3: apply_vt_penalty + commit_dq ───────────────
        let u_apply = Uniforms {
            particle_count: self.particle_count,
            tri_count,
            surface_count: model.soft_surfaces.len() as u32,
            contact_count: count,
            margin: margin as f32,
            k_contact: k_contact as f32,
            out_capacity: self.out_capacity,
            primitive_count: tri_count,
        };
        self.queue
            .write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u_apply));
        let apply_groups = count.div_ceil(WORKGROUP_SIZE).max(1);

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::soft_vt_contact apply encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::soft_vt_contact::apply_vt_penalty"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_apply_vt_penalty);
            pass.set_bind_group(0, &bg_state, &[]);
            pass.set_bind_group(1, &bg_topo, &[]);
            pass.set_bind_group(2, &bg_aabbs, &[]);
            pass.set_bind_group(3, &bg_bvh, &[]);
            pass.set_bind_group(4, &bg_contacts, &[]);
            pass.dispatch_workgroups(apply_groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::soft_vt_contact::commit_dq"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_commit_dq);
            pass.set_bind_group(0, &bg_state, &[]);
            pass.set_bind_group(1, &bg_topo, &[]);
            pass.set_bind_group(2, &bg_aabbs, &[]);
            pass.set_bind_group(3, &bg_bvh, &[]);
            pass.set_bind_group(4, &bg_contacts, &[]);
            pass.dispatch_workgroups(particle_groups, 1, 1);
        }
        self.queue.submit(std::iter::once(encoder.finish()));

        // particle_q is now device-dirty (commit_dq mutated it). The
        // `state.particle_q` Buffer<Vec3> wrapper doesn't track this
        // here — caller must `mark_device_dirty` if any subsequent
        // host read is expected. (The vbd.rs caller does this.)

        Ok(SoftVtContactStats {
            contact_count: count,
            overflowed,
        })
    }

    fn write_uniforms(
        &self,
        model: &Model,
        margin: f32,
        k_contact: f32,
        tri_count: u32,
    ) -> Result<()> {
        let u = Uniforms {
            particle_count: self.particle_count,
            tri_count,
            surface_count: model.soft_surfaces.len() as u32,
            contact_count: 0,
            margin,
            k_contact,
            out_capacity: self.out_capacity,
            primitive_count: tri_count,
        };
        self.queue
            .write_buffer(&self.uniforms, 0, bytemuck::bytes_of(&u));
        Ok(())
    }

    fn bind_state(&self, model: &Model, state: &State) -> Result<wgpu::BindGroup> {
        let q = state
            .particle_q
            .device_buffer()
            .ok_or_else(|| Error::Gpu("soft_vt_contact: state.particle_q has no device mirror".into()))?;
        let inv_mass = model
            .particle_inv_mass
            .device_buffer()
            .ok_or_else(|| Error::Gpu(
                "soft_vt_contact: model.particle_inv_mass has no device mirror".into(),
            ))?;
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::soft_vt_contact state bg"),
            layout: &self.layout_state,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: q.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: inv_mass.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.particle_dq.as_entire_binding(),
                },
            ],
        }))
    }

    fn bind_topo(&self, model: &Model) -> Result<wgpu::BindGroup> {
        let tris = model.soft_surface_tris_gpu.as_ref().ok_or_else(|| {
            Error::Gpu(
                "soft_vt_contact: model.soft_surface_tris_gpu is missing — \
                 call Model::upload_to_device first"
                    .into(),
            )
        })?;
        let surfs = model.soft_surfaces_gpu.as_ref().ok_or_else(|| {
            Error::Gpu(
                "soft_vt_contact: model.soft_surfaces_gpu is missing — \
                 call Model::upload_to_device first"
                    .into(),
            )
        })?;
        Ok(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::soft_vt_contact topo bg"),
            layout: &self.layout_topo,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: tris.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: surfs.as_entire_binding() },
            ],
        }))
    }

    fn bind_aabbs_with_lbvh(&self, lbvh: &LbvhBroadPhase) -> wgpu::BindGroup {
        // tri_aabbs binding is wired to the LBVH's primitive_aabb so
        // `compute_tri_aabbs` writes directly into the buffer the
        // LBVH refit will read.
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::soft_vt_contact aabbs bg"),
            layout: &self.layout_aabbs,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lbvh.primitive_aabb_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.vert_aabbs.as_entire_binding(),
                },
            ],
        })
    }

    fn bind_bvh(&self, lbvh: &LbvhBroadPhase) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::soft_vt_contact bvh bg"),
            layout: &self.layout_bvh,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: lbvh.nodes_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lbvh.sorted_idx_buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: lbvh.pair_count_buffer().as_entire_binding(),
                },
            ],
        })
    }

    fn bind_contacts(&self) -> wgpu::BindGroup {
        self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::soft_vt_contact out bg"),
            layout: &self.layout_contacts,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: self.out_contacts.as_entire_binding() },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.out_contact_count.as_entire_binding(),
                },
                wgpu::BindGroupEntry { binding: 2, resource: self.uniforms.as_entire_binding() },
            ],
        })
    }

    fn read_contact_count(&self) -> Result<u32> {
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::soft_vt_contact count staging"),
            size: 4,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("molla::soft_vt_contact count readback"),
            });
        encoder.copy_buffer_to_buffer(&self.out_contact_count, 0, &staging, 0, 4);
        self.queue.submit(std::iter::once(encoder.finish()));
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| Error::Gpu(format!("device poll: {e:?}")))?;
        rx.recv()
            .map_err(|e| Error::Gpu(format!("recv: {e}")))?
            .map_err(|e| Error::Gpu(format!("map: {e:?}")))?;
        let mapped = slice.get_mapped_range();
        let words: &[u32] = bytemuck::cast_slice(&mapped);
        let count = words[0];
        drop(mapped);
        staging.unmap();
        Ok(count)
    }
}

fn ro_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn rw_storage(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}
