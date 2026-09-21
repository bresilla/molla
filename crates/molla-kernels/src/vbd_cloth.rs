//! Rust dispatch wrapper for the cloth Vertex Block Descent (VBD)
//! solver in `shaders/vbd_cloth.wgsl`. The kernel is split into four
//! entry points: `predict`, `solve_color`, `apply_shape_contacts`,
//! `velocity_recovery`. Mirrors `XpbdParticlesKernel` (graph-coloured
//! per-pass dispatch, scratch buffers for the per-substep state).
//!
//! See `solvers/vbd.rs` for the algorithm description.

use molla_core::{Error, Result};
use molla_math::{GpuMat3, Mat3, Real, Transform, Vec3};
use molla_sim::{Model, State};
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    particle_count: u32,
    spring_count: u32,
    color_count: u32,
    color_index: u32,
    work_count: u32,
    dt: f32,
    linear_damping: f32,
    contact_stiffness: f32,
    /// AVBD mode flag: 0 = soft (Stage 20), 1 = hard (Stage 20.1
    /// augmented-Lagrangian dual updates).
    is_hard: u32,
    _pad1: f32,
}

const WORKGROUP_SIZE: u32 = 64;

/// Owns the VBD pipelines + per-substep scratch buffers.
/// Sized for a specific `(particle_count, spring_count)` — rebuild
/// the kernel if either changes.
pub struct VbdClothKernel {
    pipe_predict: wgpu::ComputePipeline,
    pipe_solve_color: wgpu::ComputePipeline,
    pipe_shape_contacts: wgpu::ComputePipeline,
    pipe_update_dual: wgpu::ComputePipeline,
    pipe_velocity: wgpu::ComputePipeline,
    /// Stage 20.3 — von Mises return mapping. Dispatched once per
    /// substep, after the per-vertex block iterations converge, with
    /// `work_count = tet_count`. No-op when `tet_count == 0` (the
    /// host skips the dispatch).
    pipe_plasticity: wgpu::ComputePipeline,

    layout_state: wgpu::BindGroupLayout,
    layout_topo: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
    layout_shapes: wgpu::BindGroupLayout,
    /// Stage-20.2 tet bind-group layout (group 5). Always 7 bindings;
    /// bound to dummy 1-element buffers when `tet_count == 0`.
    layout_tets: wgpu::BindGroupLayout,

    /// Per-spring stiffness (N/m), derived from XPBD compliance at
    /// upload time as `k = 1 / max(α, ε)` and then capped to the
    /// `max_stiffness` solver knob. Owned by the kernel because the
    /// compliance lives in the `Model`; converting once on the host
    /// avoids a per-step conversion in WGSL.
    spring_stiffness: Buffer<Real>,
    /// Per-particle displacement from the input position.
    particle_delta: Buffer<Vec3>,
    /// Per-particle inertial displacement target.
    inertia_y: Buffer<Vec3>,
    /// Per-particle augmented-Lagrangian multiplier (Stage 20.1
    /// hard-mode AVBD). Zero in soft mode. Stored as `Real` (host)
    /// → `f32` (device) via the standard `BufferElement` projection.
    contact_lambda: Buffer<Real>,

    /// 1-element placeholder buffers for any bind group whose model
    /// arrays may be empty (no springs / no shapes / no tets).
    dummy_u32: Buffer<u32>,
    dummy_real: Buffer<Real>,
    dummy_vec3: Buffer<Vec3>,
    dummy_i32: Buffer<i32>,
    dummy_xform: Buffer<Transform>,
    dummy_mat3: Buffer<Mat3>,
    /// Stage 20.3 — separate `read_write` placeholders for the F_p /
    /// ε_p bindings on bind group 5. wgpu refuses to bind the same
    /// buffer with conflicting usages within a single dispatch
    /// (read-only vs read-write), so we pre-allocate dedicated 1-elem
    /// buffers used only when the model has no tets.
    dummy_mat3_rw: Buffer<Mat3>,
    dummy_real_rw: Buffer<Real>,

    particle_count: u32,
    spring_count: u32,
}

impl VbdClothKernel {
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        particle_count: u32,
        spring_count: u32,
        per_spring_stiffness: &[Real],
    ) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::vbd_cloth"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/vbd_cloth.wgsl").into()),
        });

        // State group: 7 read-only inputs.
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::vbd_cloth state"),
            entries: &storage_entries(7, true),
        });
        // Topology group: 9 read-only inputs.
        let layout_topo = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::vbd_cloth topo"),
            entries: &storage_entries(9, true),
        });
        // Output / scratch group: 5 read-write (added contact_lambda
        // at binding 4 for Stage 20.1 hard-mode AVBD).
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::vbd_cloth out"),
            entries: &storage_entries(5, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::vbd_cloth uniforms"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        // Shapes group: 5 read-only (kind, data, body, transform, body_q).
        let layout_shapes = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::vbd_cloth shapes"),
            entries: &storage_entries(5, true),
        });
        // Stage 20.2/20.3/20.4 tet + cable-bending group:
        //   - 10 read-only entries (7 tet + 3 cable bending) — Stage 20.2.
        //   - 2 read-write entries (F_p, ε_p) — Stage 20.3 plasticity.
        //   - 2 read-only entries (yield, hardening) — Stage 20.3 material.
        //   - 1 read-write entry (back-stress α) — Stage 20.4.
        //   - 1 read-only entry (kinematic-hardening modulus c) — Stage 20.4.
        // 16 total. All buffers are non-empty — when the model has no
        // tets / bending the host binds dummy 1-element placeholders.
        let mut tet_entries = storage_entries(10, true);
        // Bindings 10 + 11: F_p / ε_p as read-write (mutated by the
        // plasticity entry point).
        tet_entries.extend(storage_entries(2, false).into_iter().map(|mut e| {
            e.binding += 10;
            e
        }));
        // Bindings 12 + 13: yield + hardening (read-only material).
        tet_entries.extend(storage_entries(2, true).into_iter().map(|mut e| {
            e.binding += 12;
            e
        }));
        // Binding 14: back-stress α (read-write, Stage 20.4).
        tet_entries.extend(storage_entries(1, false).into_iter().map(|mut e| {
            e.binding += 14;
            e
        }));
        // Binding 15: kinematic-hardening modulus c (read-only, Stage 20.4).
        tet_entries.extend(storage_entries(1, true).into_iter().map(|mut e| {
            e.binding += 15;
            e
        }));
        let layout_tets = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::vbd_cloth tets"),
            entries: &tet_entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::vbd_cloth layout"),
            bind_group_layouts: &[
                &layout_state,
                &layout_topo,
                &layout_out,
                &layout_uniforms,
                &layout_shapes,
                &layout_tets,
            ],
            push_constant_ranges: &[],
        });

        let make_pipeline = |entry: &str| -> wgpu::ComputePipeline {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::vbd_cloth"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        let pipe_predict = make_pipeline("predict");
        let pipe_solve_color = make_pipeline("solve_color");
        let pipe_shape_contacts = make_pipeline("apply_shape_contacts");
        let pipe_update_dual = make_pipeline("update_dual");
        let pipe_velocity = make_pipeline("velocity_recovery");
        let pipe_plasticity = make_pipeline("plasticity_return_map");

        let pn = particle_count.max(1) as usize;
        let sn = spring_count.max(1) as usize;
        let stiffness_init = if per_spring_stiffness.len() == sn
            || per_spring_stiffness.len() == spring_count as usize
        {
            let mut v = vec![0.0_f64; sn];
            for (dst, src) in v.iter_mut().zip(per_spring_stiffness.iter()) {
                *dst = *src;
            }
            v
        } else {
            vec![0.0_f64; sn]
        };
        let mut spring_stiffness = Buffer::from_vec(stiffness_init, BufferUsage::Storage);
        let mut particle_delta =
            Buffer::from_vec(vec![Vec3::ZERO; pn], BufferUsage::StorageReadWrite);
        let mut inertia_y = Buffer::from_vec(vec![Vec3::ZERO; pn], BufferUsage::StorageReadWrite);
        let mut contact_lambda = Buffer::from_vec(vec![0.0_f64; pn], BufferUsage::StorageReadWrite);
        let _ = spring_stiffness.ensure_device(device);
        let _ = spring_stiffness.upload(queue);
        let _ = particle_delta.ensure_device(device);
        let _ = particle_delta.upload(queue);
        let _ = inertia_y.ensure_device(device);
        let _ = inertia_y.upload(queue);
        let _ = contact_lambda.ensure_device(device);
        let _ = contact_lambda.upload(queue);

        let mut dummy_u32 = Buffer::from_vec(vec![0u32; 2], BufferUsage::Storage);
        let mut dummy_real = Buffer::from_vec(vec![0.0_f64; 1], BufferUsage::Storage);
        let mut dummy_vec3 = Buffer::from_vec(vec![Vec3::ZERO; 1], BufferUsage::Storage);
        let mut dummy_i32 = Buffer::from_vec(vec![0i32; 1], BufferUsage::Storage);
        let mut dummy_xform = Buffer::from_vec(vec![Transform::IDENTITY; 1], BufferUsage::Storage);
        let mut dummy_mat3 = Buffer::from_vec(vec![Mat3::IDENTITY; 1], BufferUsage::Storage);
        // Stage 20.3 — read-write dummies for F_p / ε_p bindings.
        let mut dummy_mat3_rw =
            Buffer::from_vec(vec![Mat3::IDENTITY; 1], BufferUsage::StorageReadWrite);
        let mut dummy_real_rw = Buffer::from_vec(vec![0.0_f64; 1], BufferUsage::StorageReadWrite);
        let _ = dummy_u32.ensure_device(device);
        let _ = dummy_u32.upload(queue);
        let _ = dummy_real.ensure_device(device);
        let _ = dummy_real.upload(queue);
        let _ = dummy_vec3.ensure_device(device);
        let _ = dummy_vec3.upload(queue);
        let _ = dummy_i32.ensure_device(device);
        let _ = dummy_i32.upload(queue);
        let _ = dummy_xform.ensure_device(device);
        let _ = dummy_xform.upload(queue);
        let _ = dummy_mat3.ensure_device(device);
        let _ = dummy_mat3.upload(queue);
        let _ = dummy_mat3_rw.ensure_device(device);
        let _ = dummy_mat3_rw.upload(queue);
        let _ = dummy_real_rw.ensure_device(device);
        let _ = dummy_real_rw.upload(queue);

        Self {
            pipe_predict,
            pipe_solve_color,
            pipe_shape_contacts,
            pipe_update_dual,
            pipe_velocity,
            pipe_plasticity,
            layout_state,
            layout_topo,
            layout_out,
            layout_uniforms,
            layout_shapes,
            layout_tets,
            spring_stiffness,
            particle_delta,
            inertia_y,
            contact_lambda,
            dummy_u32,
            dummy_real,
            dummy_vec3,
            dummy_i32,
            dummy_xform,
            dummy_mat3,
            dummy_mat3_rw,
            dummy_real_rw,
            particle_count,
            spring_count,
        }
    }

    /// Push a fresh per-spring stiffness table (host-side) and re-upload
    /// to the device. Useful when the user tweaks stiffness sliders in
    /// the viewer without rebuilding the model.
    pub fn set_spring_stiffness(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        per_spring_stiffness: &[Real],
    ) -> Result<()> {
        let sn = self.spring_count.max(1) as usize;
        let host = self.spring_stiffness.host_mut()?;
        for (dst, src) in host
            .iter_mut()
            .zip(
                per_spring_stiffness
                    .iter()
                    .chain(std::iter::repeat(&0.0_f64)),
            )
            .take(sn)
        {
            *dst = *src;
        }
        self.spring_stiffness.ensure_device(device)?;
        self.spring_stiffness.upload(queue)?;
        Ok(())
    }

    /// One VBD substep: predict → (iters × colours × `solve_color`
    /// [+ optional hard-mode `update_dual`]) → `velocity_recovery`.
    ///
    /// `hard_mode = true` enables Stage 20.1 augmented-Lagrangian dual
    /// updates: after every outer iteration, an `update_dual` dispatch
    /// bumps the per-particle multiplier `λ` by the residual penetration
    /// so the next iteration's `solve_color` sees the cumulative force.
    /// Soft mode (`false`) skips that pass entirely and keeps the
    /// Stage-20 v1 quadratic-penalty behaviour.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        state_out: &mut State,
        dt: Real,
        iterations: u32,
        contact_stiffness: f32,
        hard_mode: bool,
    ) -> Result<()> {
        if model.particle_count == 0 {
            return Ok(());
        }
        let particle_count_u = model.particle_count as u32;
        let spring_count_u = model.spring_count as u32;
        let shape_count_u = model.shape_count as u32;
        let color_count = model.particle_color_count as u32;

        // ── Bind groups ───────────────────────────────────────────
        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::vbd_cloth state bg"),
            layout: &self.layout_state,
            entries: &[
                buffer_entry(0, state_in.particle_q.device_buffer())?,
                buffer_entry(1, state_in.particle_qd.device_buffer())?,
                buffer_entry(2, model.particle_inv_mass.device_buffer())?,
                buffer_entry(3, model.particle_radius.device_buffer())?,
                buffer_entry(4, model.particle_world.device_buffer())?,
                buffer_entry(5, model.gravity.device_buffer())?,
                buffer_entry(6, model.initial_particle_q.device_buffer())?,
            ],
        });

        let dummy_u32_buf = self
            .dummy_u32
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_u32 missing".into()))?;
        let dummy_real_buf = self
            .dummy_real
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_real missing".into()))?;
        let sa_buf = model.spring_a.device_buffer().unwrap_or(dummy_u32_buf);
        let sb_buf = model.spring_b.device_buffer().unwrap_or(dummy_u32_buf);
        let rest_buf = model
            .spring_rest_length
            .device_buffer()
            .unwrap_or(dummy_real_buf);
        let stiff_buf = self
            .spring_stiffness
            .device_buffer()
            .unwrap_or(dummy_real_buf);
        let vadj_off_buf = model
            .vertex_adjacent_springs_offsets
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let vadj_buf = model
            .vertex_adjacent_springs
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let pcol_off_buf = model
            .particle_color_offsets
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let pcol_idx_buf = model
            .particle_indices_by_color
            .device_buffer()
            .unwrap_or(dummy_u32_buf);

        let bg_topo = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::vbd_cloth topo bg"),
            layout: &self.layout_topo,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sa_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sb_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: rest_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: stiff_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: vadj_off_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: vadj_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: pcol_off_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: pcol_idx_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: model
                        .spring_tension_only
                        .device_buffer()
                        .unwrap_or(dummy_u32_buf)
                        .as_entire_binding(),
                },
            ],
        });

        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::vbd_cloth out bg"),
            layout: &self.layout_out,
            entries: &[
                buffer_entry(0, state_out.particle_q.device_buffer())?,
                buffer_entry(1, state_out.particle_qd.device_buffer())?,
                buffer_entry(2, self.particle_delta.device_buffer())?,
                buffer_entry(3, self.inertia_y.device_buffer())?,
                buffer_entry(4, self.contact_lambda.device_buffer())?,
            ],
        });

        // Shapes bind group.
        let dummy_vec3_buf = self
            .dummy_vec3
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_vec3 missing".into()))?;
        let dummy_i32_buf = self
            .dummy_i32
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_i32 missing".into()))?;
        let dummy_xform_buf = self
            .dummy_xform
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_xform missing".into()))?;
        let kind_buf = model
            .shape_geom_kind
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let data_buf = model
            .shape_geom_data
            .device_buffer()
            .unwrap_or(dummy_vec3_buf);
        let body_buf = model.shape_body.device_buffer().unwrap_or(dummy_i32_buf);
        let xform_buf = model
            .shape_transform
            .device_buffer()
            .unwrap_or(dummy_xform_buf);
        let body_q_buf = state_in.body_q.device_buffer().unwrap_or(dummy_xform_buf);
        let bg_shapes = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::vbd_cloth shapes bg"),
            layout: &self.layout_shapes,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: kind_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: data_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: body_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: xform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: body_q_buf.as_entire_binding(),
                },
            ],
        });

        // Stage 20.2 — tet bind group. All buffers may be empty when
        // the model has no tets; fall through to the dummy buffers in
        // that case (the per-vertex CSR is zero-length, so the inner
        // tet loop in WGSL bails out immediately).
        let tet_indices_buf = model.tet_indices.device_buffer().unwrap_or(dummy_u32_buf);
        let dummy_mat3_buf = self
            .dummy_mat3
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_mat3 missing".into()))?;
        let tet_dm_inv_buf = model.tet_dm_inv.device_buffer().unwrap_or(dummy_mat3_buf);
        let tet_rest_buf = model
            .tet_rest_volume
            .device_buffer()
            .unwrap_or(dummy_real_buf);
        let tet_mu_buf = model.tet_mu.device_buffer().unwrap_or(dummy_real_buf);
        let tet_lambda_buf = model.tet_lambda.device_buffer().unwrap_or(dummy_real_buf);
        let vat_off_buf = model
            .vertex_adjacent_tets_offsets
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let vat_buf = model
            .vertex_adjacent_tets
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        // Stage 20.2 — per-vertex CABLE bending CSR. Same dummy
        // fallback as the tet buffers; the kernel reads only when
        // `vertex_cable_bend_offsets[v + 1] - offsets[v] > 0`.
        let vcb_off_buf = model
            .vertex_cable_bending_offsets
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let vcb_entries_buf = model
            .vertex_cable_bending_entries
            .device_buffer()
            .unwrap_or(dummy_u32_buf);
        let vcb_k_buf = model
            .vertex_cable_bending_k
            .device_buffer()
            .unwrap_or(dummy_real_buf);
        // Stage 20.3 — plasticity buffers. F_p / ε_p live in `state_out`
        // so the return-mapping kernel can write them in place; yield +
        // hardening live on `model` (read-only material params). When a
        // model has no tets, fall through to dedicated read-write
        // dummies (separate from the read-only `dummy_mat3_buf` /
        // `dummy_real_buf` to avoid wgpu's "conflicting usages on the
        // same buffer in a single dispatch" validation error).
        let dummy_mat3_rw_buf = self
            .dummy_mat3_rw
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_mat3_rw missing".into()))?;
        let dummy_real_rw_buf = self
            .dummy_real_rw
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dummy_real_rw missing".into()))?;
        let fp_buf = state_out
            .tet_f_plastic
            .device_buffer()
            .unwrap_or(dummy_mat3_rw_buf);
        let eps_p_buf = state_out
            .tet_plastic_strain
            .device_buffer()
            .unwrap_or(dummy_real_rw_buf);
        let yield_buf = model
            .tet_yield_stress
            .device_buffer()
            .unwrap_or(dummy_real_buf);
        let hard_buf = model
            .tet_hardening
            .device_buffer()
            .unwrap_or(dummy_real_buf);
        // Stage 20.4 — back-stress α + kinematic-hardening modulus c.
        let alpha_buf = state_out
            .tet_back_stress
            .device_buffer()
            .unwrap_or(dummy_mat3_rw_buf);
        let kin_hard_buf = model
            .tet_kinematic_hardening
            .device_buffer()
            .unwrap_or(dummy_real_buf);

        let bg_tets = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::vbd_cloth tets bg"),
            layout: &self.layout_tets,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: tet_indices_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: tet_dm_inv_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: tet_rest_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: tet_mu_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: tet_lambda_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: vat_off_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: vat_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: vcb_off_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: vcb_entries_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: vcb_k_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: fp_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: eps_p_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: yield_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 13,
                    resource: hard_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: alpha_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 15,
                    resource: kin_hard_buf.as_entire_binding(),
                },
            ],
        });

        // ── Uniforms (one per dispatch) ──────────────────────────
        let color_offsets_host = model.particle_color_offsets.host()?.to_vec();

        let mut uniform_bufs: Vec<wgpu::Buffer> = Vec::new();
        let mut bg_uniforms: Vec<wgpu::BindGroup> = Vec::new();
        let mut push_uniforms = |u: &Uniforms| {
            let buf = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("molla::vbd_cloth uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            queue.write_buffer(&buf, 0, bytemuck::bytes_of(u));
            let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::vbd_cloth uniforms bg"),
                layout: &self.layout_uniforms,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf.as_entire_binding(),
                }],
            });
            uniform_bufs.push(buf);
            bg_uniforms.push(bg);
        };

        let is_hard_u: u32 = if hard_mode { 1 } else { 0 };
        let tet_count_u = model.tet_count as u32;
        let base = |work_count: u32, color_index: u32, contact_k: f32| Uniforms {
            particle_count: particle_count_u,
            spring_count: spring_count_u,
            color_count,
            color_index,
            work_count,
            dt: dt as f32,
            linear_damping: model.linear_damping as f32,
            contact_stiffness: contact_k,
            is_hard: is_hard_u,
            _pad1: 0.0,
        };

        // Predict.
        push_uniforms(&base(particle_count_u, 0, 0.0));
        // All elastic iterations share the per-colour and dual uniforms.
        for k in 0..color_count {
            let lo = color_offsets_host[k as usize];
            let hi = color_offsets_host[k as usize + 1];
            push_uniforms(&base(hi - lo, k, contact_stiffness));
        }
        if hard_mode {
            push_uniforms(&base(particle_count_u, 0, contact_stiffness));
        }
        let _ = shape_count_u; // referenced in the apply_shape_contacts
        // dispatch path; kept for future use.
        // Velocity recovery.
        push_uniforms(&base(particle_count_u, 0, 0.0));
        // Stage 20.3 — plasticity return mapping. work_count = tet_count.
        if tet_count_u > 0 {
            push_uniforms(&base(tet_count_u, 0, 0.0));
        }

        // ── Encode + dispatch ─────────────────────────────────────
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::vbd_cloth encoder"),
        });
        // Stage 20.3 — copy state_in's plasticity state into state_out
        // once at the top of the step. The plasticity kernel then
        // reads/writes from state_out's buffers so a substep ping-pong
        // (state_in → state_out → state_in) sees the cumulative effect
        // without the host having to read back F_p between substeps.
        // Skip the copy entirely when `tet_count == 0`.
        if tet_count_u > 0 {
            if let (Some(src_fp), Some(dst_fp)) = (
                state_in.tet_f_plastic.device_buffer(),
                state_out.tet_f_plastic.device_buffer(),
            ) {
                let bytes = (model.tet_count * core::mem::size_of::<GpuMat3>()) as u64;
                if bytes > 0 {
                    encoder.copy_buffer_to_buffer(src_fp, 0, dst_fp, 0, bytes);
                }
            }
            if let (Some(src_e), Some(dst_e)) = (
                state_in.tet_plastic_strain.device_buffer(),
                state_out.tet_plastic_strain.device_buffer(),
            ) {
                let bytes = (model.tet_count * core::mem::size_of::<f32>()) as u64;
                if bytes > 0 {
                    encoder.copy_buffer_to_buffer(src_e, 0, dst_e, 0, bytes);
                }
            }
            // Stage 20.4 — back-stress α: same in→out copy as F_p so the
            // kinematic-hardening kernel can mutate state_out's α in
            // place across substeps.
            if let (Some(src_a), Some(dst_a)) = (
                state_in.tet_back_stress.device_buffer(),
                state_out.tet_back_stress.device_buffer(),
            ) {
                let bytes = (model.tet_count * core::mem::size_of::<GpuMat3>()) as u64;
                if bytes > 0 {
                    encoder.copy_buffer_to_buffer(src_a, 0, dst_a, 0, bytes);
                }
            }
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::vbd_cloth"),
                timestamp_writes: None,
            });
            pass.set_bind_group(0, &bg_state, &[]);
            pass.set_bind_group(1, &bg_topo, &[]);
            pass.set_bind_group(2, &bg_out, &[]);
            pass.set_bind_group(4, &bg_shapes, &[]);
            pass.set_bind_group(5, &bg_tets, &[]);

            let dispatch = |pass: &mut wgpu::ComputePass<'_>,
                            pipeline: &wgpu::ComputePipeline,
                            uniform_idx: usize,
                            work: u32| {
                if work == 0 {
                    return;
                }
                pass.set_pipeline(pipeline);
                pass.set_bind_group(3, &bg_uniforms[uniform_idx], &[]);
                let groups = work.div_ceil(WORKGROUP_SIZE);
                pass.dispatch_workgroups(groups.max(1), 1, 1);
            };

            let mut idx = 0usize;
            // Predict.
            dispatch(&mut pass, &self.pipe_predict, idx, particle_count_u);
            // Outer iterations × colours. Soft rigid contacts are
            // baked into `solve_color` so each per-vertex Newton
            // block sees the contact force + Hessian directly. In
            // hard mode we follow each outer iteration with an
            // `update_dual` dispatch that bumps the per-particle λ
            // by the residual penetration.
            for _ in 0..iterations {
                idx = 1;
                for k in 0..color_count {
                    let lo = color_offsets_host[k as usize];
                    let hi = color_offsets_host[k as usize + 1];
                    let nc = hi - lo;
                    dispatch(&mut pass, &self.pipe_solve_color, idx, nc);
                    idx += 1;
                }
                if hard_mode {
                    dispatch(&mut pass, &self.pipe_update_dual, idx, particle_count_u);
                }
            }
            idx = 1 + color_count as usize + usize::from(hard_mode);
            // Velocity recovery.
            dispatch(&mut pass, &self.pipe_velocity, idx, particle_count_u);
            idx += 1;
            // Stage 20.3 — plasticity return mapping (one thread per
            // tet). The kernel mutates `tet_f_plastic` / `tet_plastic_strain`
            // in `state_out` directly (read_write storage in bind group 5).
            if tet_count_u > 0 {
                dispatch(&mut pass, &self.pipe_plasticity, idx, tet_count_u);
                idx += 1;
            }
            // `pipe_shape_contacts` kept compiled for future use
            // (warm-start initialiser, alternate AVBD safety pass);
            // silence the unused-field warning.
            let _ = &self.pipe_shape_contacts;
            debug_assert_eq!(idx, uniform_bufs.len());
        }
        queue.submit(std::iter::once(encoder.finish()));

        state_out.particle_q.mark_device_dirty();
        state_out.particle_qd.mark_device_dirty();
        if tet_count_u > 0 {
            state_out.tet_f_plastic.mark_device_dirty();
            state_out.tet_plastic_strain.mark_device_dirty();
            state_out.tet_back_stress.mark_device_dirty();
        }
        Ok(())
    }

    pub fn particle_count(&self) -> u32 {
        self.particle_count
    }
    pub fn spring_count(&self) -> u32 {
        self.spring_count
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf =
        buf.ok_or_else(|| Error::Gpu(format!("vbd_cloth binding {binding}: no device mirror")))?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    })
}

fn storage_entries(count: u32, read_only: bool) -> Vec<wgpu::BindGroupLayoutEntry> {
    (0..count)
        .map(|i| wgpu::BindGroupLayoutEntry {
            binding: i,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}
