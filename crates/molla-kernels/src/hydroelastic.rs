//! Stage 41 v2 sub-pipeline 4 — high-level GPU hydroelastic
//! contact + anchor-reduction dispatcher.
//!
//! This wraps two compute passes from `shaders/hydroelastic.wgsl`:
//!
//! 1. **`hydro_contact`** — per-sample pressure-field evaluation
//!    (Stage 22.1). Each rigid surface sample is transformed into the
//!    soft body's local frame, the SDF is trilinearly sampled, and
//!    a per-sample `(point, normal, pressure × area)` triple is
//!    written into persistent output buffers.
//! 2. **`reduce_to_anchor`** — Stage 39D anchor reduction on GPU.
//!    A single-workgroup tree reduction takes the per-sample arrays
//!    produced by the pressure pass and emits a 2-contact "anchor
//!    reduced" pair (centroid + center-of-pressure) that preserves
//!    both net force AND net moment of the distributed patch.
//!
//! The CPU equivalents live in `geometry::hydroelastic`:
//! [`compute_hydroelastic_contact`] +
//! [`reduce_with_anchor`]. The GPU dispatcher reads back the anchor
//! pair into an [`AnchorReducedContact`] for direct CPU consumption
//! (matching the v1 spec's "Reads back AnchorReducedContact").
//!
//! ## v1 scope
//!
//! - **Soft-on-rigid.** Soft body carries an `SdfGrid` pressure field;
//!   rigid body is sampled at discrete surface points. v3 will lift
//!   soft-on-soft (volumetric intersection).
//! - **Single shape pair per dispatch.** Multi-pair will follow the
//!   broad-phase batching pattern (see Stage 28.1 contact reduction);
//!   not needed for the v1 anchor scenario.
//!
//! ## How to use
//!
//! ```ignore
//! use molla::compute::Engine;
//! use molla::geometry::{bake_mesh_to_sdf, sample_sphere_surface, HydroelasticConfig};
//! use molla::kernels::HydroelasticKernel as GpuHydroelastic;
//! use molla::math::{Transform, Vec3};
//!
//! let engine = Engine::new_gpu().unwrap();
//! let device = engine.device().unwrap().clone();
//! let queue = engine.queue().unwrap().clone();
//!
//! let sdf = bake_mesh_to_sdf(&plane_mesh, 32, 0.05);
//! let samples = sample_sphere_surface(0.1, 128);
//! let mut k = GpuHydroelastic::new(&device, &queue, samples.len() as u32);
//! let anchor = k
//!     .dispatch(
//!         &device,
//!         &queue,
//!         Transform::IDENTITY,
//!         sphere_pose,
//!         &sdf,
//!         &samples,
//!         &HydroelasticConfig::default(),
//!     )
//!     .unwrap();
//! ```
//!
//! See `tests/hydroelastic_gpu.rs::sphere_on_plane_anchor_gpu_parity`
//! for the GPU-vs-CPU agreement test that pins down the v1 contract.

#![cfg(feature = "gpu")]

use molla_core::{Error, Result};
use molla_geometry::SdfGrid;
use molla_geometry::hydroelastic::{
    AnchorReducedContact, HydroelasticConfig, PatchSample, SurfaceSample,
};
use molla_math::{Mat3, Real, Transform, Vec3};
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct HydroUniforms {
    sample_count: u32,
    nx: u32,
    ny: u32,
    nz: u32,
    cell_size: f32,
    modulus: f32,
    max_pressure: f32,
    culling_distance: f32,
    origin: [f32; 4],
    soft_translation: [f32; 4],
    rot_col0: [f32; 4],
    rot_col1: [f32; 4],
    rot_col2: [f32; 4],
    rigid_translation: [f32; 4],
    rigid_rot_col0: [f32; 4],
    rigid_rot_col1: [f32; 4],
    rigid_rot_col2: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct AnchorUniforms {
    sample_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const PRESSURE_WG_SIZE: u32 = 64;
const ANCHOR_WG_SIZE: u32 = 256;

/// Stage-41-v2 GPU hydroelastic kernel: pressure-field evaluation +
/// anchor reduction in a single high-level dispatcher.
///
/// Owns the per-sample output buffers and the 3-entry anchor-output
/// buffer (centroid + anchor + summary). Reuse across substeps;
/// re-construction is cheap but re-running [`Self::dispatch`] reuses
/// all device allocations.
pub struct HydroelasticKernel {
    /// Pipeline 1: per-sample pressure evaluation (uses group 0 only).
    pipe_pressure: wgpu::ComputePipeline,
    layout_pressure: wgpu::BindGroupLayout,
    /// Pipeline 2: anchor reduction (uses group 1 only).
    pipe_anchor: wgpu::ComputePipeline,
    layout_anchor: wgpu::BindGroupLayout,

    /// Per-sample output: world-frame normal (zero if non-penetrating).
    pub contact_normal: Buffer<Vec3>,
    /// Per-sample output: world-frame point.
    pub contact_point: Buffer<Vec3>,
    /// Per-sample output: `pressure × area` (0.0 if non-penetrating).
    pub contact_normal_force: Buffer<Real>,
    /// Anchor-reduction output. 3 vec4 entries:
    ///   [0] = (centroid.xyz, force_magnitude)
    ///   [1] = (anchor.xyz,   force_magnitude)
    ///   [2] = (force_dir.xyz, total_force_magnitude)
    pub anchor_out: Buffer<Vec3>,
    /// Sample-area buffer (m²), kept resident so the second pass can
    /// be re-issued with the same geometry without re-upload.
    sample_area_resident: Buffer<Real>,
    /// Sample world-frame normal buffer (in rigid local frame, padded
    /// to vec4 by `BufferElement` impl).
    sample_normal_resident: Buffer<Vec3>,
    /// Sample position buffer (rigid local frame).
    sample_pos_resident: Buffer<Vec3>,
    /// SDF data buffer (flat, row-major nz × ny × nx).
    sdf_resident: Buffer<Real>,

    sample_count: u32,
}

impl HydroelasticKernel {
    /// Build the kernel sized for `sample_count` rigid surface
    /// samples. Allocates GPU mirrors for every per-sample buffer +
    /// the anchor output. Re-use across substeps; call
    /// [`Self::dispatch`] each frame.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, sample_count: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::kernels::hydroelastic"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/hydroelastic.wgsl").into()),
        });

        // ── Group 0: pressure-pass bindings (uniform + 4 ro + 3 rw). ──
        let mut pressure_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        for (i, read_only) in [
            (1u32, true),
            (2, true),
            (3, true),
            (4, true),
            (5, false),
            (6, false),
            (7, false),
        ] {
            pressure_entries.push(wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let layout_pressure = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::kernels::hydroelastic pressure layout"),
            entries: &pressure_entries,
        });
        let pipeline_layout_pressure =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::kernels::hydroelastic pressure pipeline layout"),
                bind_group_layouts: &[Some(&layout_pressure)],
                immediate_size: 0,
            });
        let pipe_pressure = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::kernels::hydroelastic hydro_contact"),
            layout: Some(&pipeline_layout_pressure),
            module: &shader,
            entry_point: Some("hydro_contact"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── Group 1: anchor-pass bindings (uniform + 3 ro + 1 rw). ────
        // Mapped to @group(1) in WGSL because the same shader module
        // hosts both entry points and the pressure pass already
        // claims @group(0).
        let mut anchor_entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        // All anchor storage bindings are declared `read_write` to
        // match the WGSL declaration (which itself uses `read_write`
        // so the usage flag agrees with the pressure pass's output
        // bindings — both groups bind the same physical buffers).
        for i in 1u32..=4 {
            anchor_entries.push(wgpu::BindGroupLayoutEntry {
                binding: i,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let layout_anchor = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::kernels::hydroelastic anchor layout"),
            entries: &anchor_entries,
        });
        // Anchor pipeline declares BOTH layouts (group 0 + group 1)
        // even though it only uses group 1 — wgpu requires the
        // pipeline-layout group count to cover the maximum group
        // index actually used by the entry point. WGSL puts the
        // anchor bindings at @group(1), so we need a 2-slot pipeline
        // layout with a dummy group 0. We reuse the pressure layout
        // for that slot (it's never bound at dispatch time).
        let pipeline_layout_anchor =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("molla::kernels::hydroelastic anchor pipeline layout"),
                bind_group_layouts: &[Some(&layout_pressure), Some(&layout_anchor)],
                immediate_size: 0,
            });
        let pipe_anchor = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::kernels::hydroelastic reduce_to_anchor"),
            layout: Some(&pipeline_layout_anchor),
            module: &shader,
            entry_point: Some("reduce_to_anchor"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        // ── Allocate persistent output buffers. ───────────────────────
        let cn = sample_count.max(1) as usize;
        let mut contact_normal =
            Buffer::from_vec(vec![Vec3::ZERO; cn], BufferUsage::StorageReadWrite);
        let mut contact_point =
            Buffer::from_vec(vec![Vec3::ZERO; cn], BufferUsage::StorageReadWrite);
        let mut contact_normal_force =
            Buffer::from_vec(vec![0.0_f64; cn], BufferUsage::StorageReadWrite);
        // Anchor-out: 3 vec4 entries (centroid, anchor, summary).
        let mut anchor_out = Buffer::from_vec(vec![Vec3::ZERO; 3], BufferUsage::StorageReadWrite);

        for b in [&mut contact_normal, &mut contact_point] {
            b.ensure_device(device)
                .expect("kernels::hydroelastic: ensure_device output Vec3");
            b.upload(queue)
                .expect("kernels::hydroelastic: upload output Vec3");
        }
        contact_normal_force
            .ensure_device(device)
            .expect("kernels::hydroelastic: ensure_device contact_normal_force");
        contact_normal_force
            .upload(queue)
            .expect("kernels::hydroelastic: upload contact_normal_force");
        anchor_out
            .ensure_device(device)
            .expect("kernels::hydroelastic: ensure_device anchor_out");
        anchor_out
            .upload(queue)
            .expect("kernels::hydroelastic: upload anchor_out");

        // Resident input buffers — uploaded lazily on first dispatch.
        let sample_pos_resident = Buffer::from_vec(vec![Vec3::ZERO; cn], BufferUsage::Storage);
        let sample_normal_resident = Buffer::from_vec(vec![Vec3::ZERO; cn], BufferUsage::Storage);
        let sample_area_resident = Buffer::from_vec(vec![0.0_f64; cn], BufferUsage::Storage);
        let sdf_resident = Buffer::from_vec(vec![0.0_f64; 1], BufferUsage::Storage);

        Self {
            pipe_pressure,
            layout_pressure,
            pipe_anchor,
            layout_anchor,
            contact_normal,
            contact_point,
            contact_normal_force,
            anchor_out,
            sample_area_resident,
            sample_normal_resident,
            sample_pos_resident,
            sdf_resident,
            sample_count,
        }
    }

    /// Number of rigid surface samples this kernel was sized for.
    #[must_use]
    pub fn sample_count(&self) -> u32 {
        self.sample_count
    }

    /// Dispatch both passes and read back the [`AnchorReducedContact`].
    ///
    /// On the first call, the rigid sample positions/normals/areas
    /// and the SDF data are uploaded to GPU-resident buffers; on
    /// subsequent calls only the uniforms change (assuming geometry
    /// is constant — re-call with `Self::reset_geometry` to update
    /// the sample/SDF buffers when those mutate).
    ///
    /// Returns `Ok(None)` when the kernel computed no non-zero contact
    /// (empty manifold or zero net force), matching the CPU
    /// `molla_geometry::reduce_with_anchor` contract.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        soft_pose: Transform,
        rigid_pose: Transform,
        soft_sdf: &SdfGrid,
        rigid_samples: &[SurfaceSample],
        cfg: &HydroelasticConfig,
    ) -> Result<Option<AnchorReducedContact>> {
        if rigid_samples.len() as u32 != self.sample_count {
            return Err(Error::Gpu(format!(
                "hydroelastic kernel sized for {} samples, got {}",
                self.sample_count,
                rigid_samples.len()
            )));
        }

        // ── 1. Upload sample arrays + SDF (re-uploads each call — the
        //       Stage 41 v1 slice assumes single-pair, single-frame use;
        //       a v2 slice will move this to one-shot upload). ─────────
        let pos_host: Vec<Vec3> = rigid_samples.iter().map(|s| s.position).collect();
        let normal_host: Vec<Vec3> = rigid_samples.iter().map(|s| s.normal).collect();
        let area_host: Vec<Real> = rigid_samples.iter().map(|s| s.area).collect();

        // Resize-or-overwrite host buffers + re-upload. `host_mut`
        // hands back `&mut [T]` (fixed-length view into the underlying
        // `Vec<T>`); we copy element-wise since the buffers were
        // pre-sized in `new()` to `sample_count`.
        self.sample_pos_resident
            .host_mut()?
            .copy_from_slice(&pos_host);
        self.sample_normal_resident
            .host_mut()?
            .copy_from_slice(&normal_host);
        self.sample_area_resident
            .host_mut()?
            .copy_from_slice(&area_host);
        self.sample_pos_resident.ensure_device(device)?;
        self.sample_pos_resident.upload(queue)?;
        self.sample_normal_resident.ensure_device(device)?;
        self.sample_normal_resident.upload(queue)?;
        self.sample_area_resident.ensure_device(device)?;
        self.sample_area_resident.upload(queue)?;

        // SDF upload: replace + re-allocate if needed.
        if self.sdf_resident.len() != soft_sdf.data.len() {
            self.sdf_resident = Buffer::from_vec(soft_sdf.data.clone(), BufferUsage::Storage);
        } else {
            self.sdf_resident
                .host_mut()?
                .copy_from_slice(&soft_sdf.data);
        }
        self.sdf_resident.ensure_device(device)?;
        self.sdf_resident.upload(queue)?;

        // ── 2. Pressure pass uniforms. ────────────────────────────────
        let r_mat = Mat3::from_quat(soft_pose.rotation);
        let rmat = Mat3::from_quat(rigid_pose.rotation);
        let u_pressure = HydroUniforms {
            sample_count: self.sample_count,
            nx: soft_sdf.nx,
            ny: soft_sdf.ny,
            nz: soft_sdf.nz,
            cell_size: soft_sdf.cell_size as f32,
            modulus: cfg.hydroelastic_modulus as f32,
            max_pressure: cfg.max_pressure as f32,
            culling_distance: cfg.culling_distance as f32,
            origin: [
                soft_sdf.origin.x as f32,
                soft_sdf.origin.y as f32,
                soft_sdf.origin.z as f32,
                0.0,
            ],
            soft_translation: [
                soft_pose.position.x as f32,
                soft_pose.position.y as f32,
                soft_pose.position.z as f32,
                0.0,
            ],
            rot_col0: [
                r_mat.x_axis.x as f32,
                r_mat.x_axis.y as f32,
                r_mat.x_axis.z as f32,
                0.0,
            ],
            rot_col1: [
                r_mat.y_axis.x as f32,
                r_mat.y_axis.y as f32,
                r_mat.y_axis.z as f32,
                0.0,
            ],
            rot_col2: [
                r_mat.z_axis.x as f32,
                r_mat.z_axis.y as f32,
                r_mat.z_axis.z as f32,
                0.0,
            ],
            rigid_translation: [
                rigid_pose.position.x as f32,
                rigid_pose.position.y as f32,
                rigid_pose.position.z as f32,
                0.0,
            ],
            rigid_rot_col0: [
                rmat.x_axis.x as f32,
                rmat.x_axis.y as f32,
                rmat.x_axis.z as f32,
                0.0,
            ],
            rigid_rot_col1: [
                rmat.y_axis.x as f32,
                rmat.y_axis.y as f32,
                rmat.y_axis.z as f32,
                0.0,
            ],
            rigid_rot_col2: [
                rmat.z_axis.x as f32,
                rmat.z_axis.y as f32,
                rmat.z_axis.z as f32,
                0.0,
            ],
        };
        let pressure_uniform_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::kernels::hydroelastic pressure uniforms"),
                size: core::mem::size_of::<HydroUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(
            queue,
            &pressure_uniform_buf,
            0,
            bytemuck::bytes_of(&u_pressure)
        );

        // ── 3. Anchor-pass uniforms. ──────────────────────────────────
        let u_anchor = AnchorUniforms {
            sample_count: self.sample_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let anchor_uniform_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::kernels::hydroelastic anchor uniforms"),
                size: core::mem::size_of::<AnchorUniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(queue, &anchor_uniform_buf, 0, bytemuck::bytes_of(&u_anchor));

        // ── 4. Bind groups. ───────────────────────────────────────────
        let pos_buf = self
            .sample_pos_resident
            .device_buffer()
            .ok_or_else(|| Error::Gpu("kernels::hydroelastic: pos missing".into()))?;
        let normal_buf = self
            .sample_normal_resident
            .device_buffer()
            .ok_or_else(|| Error::Gpu("kernels::hydroelastic: normal missing".into()))?;
        let area_buf = self
            .sample_area_resident
            .device_buffer()
            .ok_or_else(|| Error::Gpu("kernels::hydroelastic: area missing".into()))?;
        let sdf_buf = self
            .sdf_resident
            .device_buffer()
            .ok_or_else(|| Error::Gpu("kernels::hydroelastic: sdf missing".into()))?;
        let n_buf = self
            .contact_normal
            .device_buffer()
            .ok_or_else(|| Error::Gpu("kernels::hydroelastic: contact_normal missing".into()))?;
        let p_buf = self
            .contact_point
            .device_buffer()
            .ok_or_else(|| Error::Gpu("kernels::hydroelastic: contact_point missing".into()))?;
        let f_buf = self.contact_normal_force.device_buffer().ok_or_else(|| {
            Error::Gpu("kernels::hydroelastic: contact_normal_force missing".into())
        })?;
        let a_out_buf = self
            .anchor_out
            .device_buffer()
            .ok_or_else(|| Error::Gpu("kernels::hydroelastic: anchor_out missing".into()))?;

        let bg_pressure = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::kernels::hydroelastic pressure bg"),
            layout: &self.layout_pressure,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pressure_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pos_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: normal_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: area_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: sdf_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: n_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: p_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: f_buf.as_entire_binding(),
                },
            ],
        });

        let bg_anchor = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::kernels::hydroelastic anchor bg"),
            layout: &self.layout_anchor,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: anchor_uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: n_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: f_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: a_out_buf.as_entire_binding(),
                },
            ],
        });

        // ── 5. Encode + submit both passes back-to-back. ──────────────
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::kernels::hydroelastic encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::kernels::hydroelastic pressure"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_pressure);
            pass.set_bind_group(0, &bg_pressure, &[]);
            let groups = self.sample_count.div_ceil(PRESSURE_WG_SIZE).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::kernels::hydroelastic anchor"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipe_anchor);
            // Group 0 is unused by `reduce_to_anchor`, but the
            // pipeline layout still declares it (so the WGSL group
            // indices align). Re-bind the pressure group as a dummy.
            pass.set_bind_group(0, &bg_pressure, &[]);
            pass.set_bind_group(1, &bg_anchor, &[]);
            // Single workgroup — the reduction is strided across the
            // sample count internally. ANCHOR_WG_SIZE is a compile-time
            // constant in the WGSL.
            let _ = ANCHOR_WG_SIZE;
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        // ── 6. Readback. ──────────────────────────────────────────────
        // The `Buffer<Vec3>` mirror drops the .w channel on download
        // (it maps `vec4<f32>` → `glam::DVec3`), but the anchor pass
        // writes packed `(point.xyz, force_mag)` plus a summary row of
        // `(force_dir.xyz, total_force_mag)`. Read the 12 raw f32
        // words via a one-shot staging copy.
        let (centroid_point, f_c_mag, anchor_point, f_a_mag, force_dir, total_mag) =
            read_anchor_packed(device, queue, a_out_buf)?;

        self.contact_normal.mark_device_dirty();
        self.contact_point.mark_device_dirty();
        self.contact_normal_force.mark_device_dirty();
        self.anchor_out.mark_device_dirty();

        if total_mag <= 1.0e-20 {
            return Ok(None);
        }
        let normal = if force_dir.length_squared() > 1.0e-24 {
            force_dir.normalize()
        } else {
            // Empty / degenerate — empty manifold semantics.
            return Ok(None);
        };

        Ok(Some(AnchorReducedContact {
            centroid: PatchSample {
                point: centroid_point,
                normal,
                force_magnitude: f_c_mag,
            },
            anchor: PatchSample {
                point: anchor_point,
                normal,
                force_magnitude: f_a_mag,
            },
        }))
    }
}

/// Side-channel readback for the packed 3-vec4 anchor output (the
/// `Buffer<Vec3>` mirror drops `.w` on download). One staging copy +
/// blocking map_async; cost is dominated by `device.poll` latency,
/// ~100 µs on a modern desktop adapter. Returns:
/// `(centroid_point, f_c_mag, anchor_point, f_a_mag, force_dir, total_mag)`.
fn read_anchor_packed(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    src: &wgpu::Buffer,
) -> Result<(Vec3, Real, Vec3, Real, Vec3, Real)> {
    // 3 × vec4<f32> = 48 bytes.
    const N_BYTES: u64 = 48;
    let staging = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("molla::kernels::hydroelastic anchor readback"),
            size: N_BYTES,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("molla::kernels::hydroelastic anchor copy"),
    });
    encoder.copy_buffer_to_buffer(src, 0, &staging, 0, N_BYTES);
    queue.submit(std::iter::once(encoder.finish()));

    let slice = staging.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|e| Error::Gpu(format!("kernels::hydroelastic anchor readback poll: {e:?}")))?;
    match rx.recv() {
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            return Err(Error::Gpu(format!(
                "kernels::hydroelastic anchor readback map: {e:?}"
            )));
        }
        Err(e) => {
            return Err(Error::Gpu(format!(
                "kernels::hydroelastic anchor readback chan: {e:?}"
            )));
        }
    }
    let data = molla_core::gpu_readback!(slice);
    let words: &[f32] = bytemuck::cast_slice(&data);
    // Layout: [cx, cy, cz, fc, ax, ay, az, fa, dx, dy, dz, tot].
    let centroid = Vec3::new(words[0] as Real, words[1] as Real, words[2] as Real);
    let f_c = words[3] as Real;
    let anchor = Vec3::new(words[4] as Real, words[5] as Real, words[6] as Real);
    let f_a = words[7] as Real;
    let dir = Vec3::new(words[8] as Real, words[9] as Real, words[10] as Real);
    let tot = words[11] as Real;
    drop(data);
    staging.unmap();
    Ok((centroid, f_c, anchor, f_a, dir, tot))
}
