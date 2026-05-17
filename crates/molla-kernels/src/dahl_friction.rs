//! Dahl friction GPU kernel — Stage 20.1.
//!
//! Smooth stick→slip friction for particle-vs-rigid contacts. Designed
//! generically: takes opaque contact-side state (positions, normals,
//! tangent velocities, normal force, friction coefficient) so Stage
//! 22.1's hydroelastic path can reuse it for distributed-pressure
//! friction without a kernel rewrite.
//!
//! See `shaders/dahl_friction.wgsl` for the algorithm description.
//!
//! ## Constants picked for v1
//!
//! - `sigma0 = 1.0e5 N/m` — moderate "rubber-on-steel" stiffness;
//!   matches the magnitude of the AVBD `contact_stiffness` knob so a
//!   user who is comfortable with one is in the same regime for both.
//! - `sigma1 = 0.0` — pure Dahl, no visco-elastic damping. Adding
//!   `sigma1 > 0` introduces rate-dependence that's useful for cable
//!   damping but not in v1.
//! - `f_normal_eps = 1.0e-9 N` — below this, the bristle state is
//!   reset (no friction can be sustained on an inactive contact).
//!
//! These are surfaced as solver-knob defaults; tests + viewer scenes
//! that need different values pass them via [`DahlConfig`].

use crate::dahl_oracle::DahlConfig;
use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_storage::{Buffer, BufferUsage};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    contact_count: u32,
    dt: f32,
    sigma0: f32,
    sigma1: f32,
    f_normal_eps: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

const WORKGROUP_SIZE: u32 = 64;

/// One Dahl friction step over `contact_count` contacts. Owns the
/// pipeline + the per-contact persistent bristle-state buffer `z`
/// (so hysteresis survives across substeps).
pub struct DahlFrictionKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    /// Per-contact bristle-deflection state (m, world frame). Padded
    /// to `Vec3` (16-byte aligned) so it matches the WGSL struct.
    contact_z: Buffer<Vec3>,
    /// Per-contact output friction force (N, world frame).
    contact_friction_force: Buffer<Vec3>,
    contact_count: u32,
}

impl DahlFrictionKernel {
    /// Build a kernel sized for `contact_count` slots. The kernel
    /// dispatches `ceil(contact_count / 64)` workgroups per call.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, contact_count: u32) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::dahl_friction"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/dahl_friction.wgsl").into()),
        });

        let mut entries = vec![wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }];
        // bindings 1..=4: read-only inputs. binding 5: read-write
        // state. binding 6: read-write output.
        for i in 1u32..=6 {
            let read_only = i <= 4;
            entries.push(wgpu::BindGroupLayoutEntry {
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
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::dahl_friction layout"),
            entries: &entries,
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::dahl_friction pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::dahl_friction"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("dahl_step"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        let cn = contact_count.max(1) as usize;
        let mut contact_z = Buffer::from_vec(vec![Vec3::ZERO; cn], BufferUsage::StorageReadWrite);
        let mut contact_friction_force =
            Buffer::from_vec(vec![Vec3::ZERO; cn], BufferUsage::StorageReadWrite);
        // Allocations / uploads at construction time are unrecoverable
        // failures (wgpu OOM); panic with context instead of swallowing
        // errors that would later surface as "missing device mirror".
        contact_z
            .ensure_device(device)
            .expect("dahl_friction: contact_z ensure_device");
        contact_z
            .upload(queue)
            .expect("dahl_friction: contact_z upload");
        contact_friction_force
            .ensure_device(device)
            .expect("dahl_friction: contact_friction_force ensure_device");
        contact_friction_force
            .upload(queue)
            .expect("dahl_friction: contact_friction_force upload");

        Self {
            pipeline,
            layout,
            contact_z,
            contact_friction_force,
            contact_count,
        }
    }

    /// Reset the persistent bristle state to zero. Call after a scene
    /// reset / when contact ids would otherwise carry stale state
    /// across discontinuous changes.
    pub fn reset_state(&mut self, queue: &wgpu::Queue) -> Result<()> {
        let cn = self.contact_z.host()?.len();
        let host = self.contact_z.host_mut()?;
        for v in host.iter_mut() {
            *v = Vec3::ZERO;
        }
        let _ = cn;
        self.contact_z.upload(queue)?;
        Ok(())
    }

    /// Number of contact slots this kernel was built for.
    pub fn contact_count(&self) -> u32 {
        self.contact_count
    }

    /// Read back the latest friction force buffer to host. Used by
    /// tests that integrate the friction force on the CPU side; the
    /// production path consumes the device buffer in-place.
    pub fn download_friction(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<&[Vec3]> {
        self.contact_friction_force.download(device, queue)?;
        self.contact_friction_force.host()
    }

    /// Read back the latest bristle-state buffer. Tests can inspect
    /// this to confirm stick vs slip behaviour.
    pub fn download_state(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Result<&[Vec3]> {
        self.contact_z.download(device, queue)?;
        self.contact_z.host()
    }

    /// Execute one Dahl-step pass.
    ///
    /// * `dt` — substep length (s).
    /// * `cfg` — the Dahl knobs.
    /// * `contact_normal`, `contact_v_tan` — per-contact world-frame
    ///   unit normal + tangential velocity. Lengths must equal
    ///   `contact_count`.
    /// * `contact_f_n` — per-contact normal force magnitude (N).
    /// * `contact_mu` — per-contact friction coefficient.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dt: Real,
        cfg: DahlConfig,
        contact_normal: &Buffer<Vec3>,
        contact_v_tan: &Buffer<Vec3>,
        contact_f_n: &Buffer<Real>,
        contact_mu: &Buffer<Real>,
    ) -> Result<()> {
        let u = Uniforms {
            contact_count: self.contact_count,
            dt: dt as f32,
            sigma0: cfg.sigma0 as f32,
            sigma1: cfg.sigma1 as f32,
            f_normal_eps: cfg.f_normal_eps as f32,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::dahl_friction uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&u));

        let normal_buf = contact_normal
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dahl: contact_normal missing device mirror".into()))?;
        let vtan_buf = contact_v_tan
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dahl: contact_v_tan missing device mirror".into()))?;
        let fn_buf = contact_f_n
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dahl: contact_f_n missing device mirror".into()))?;
        let mu_buf = contact_mu
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dahl: contact_mu missing device mirror".into()))?;
        let z_buf = self
            .contact_z
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dahl: contact_z missing".into()))?;
        let f_buf = self
            .contact_friction_force
            .device_buffer()
            .ok_or_else(|| Error::Gpu("dahl: contact_friction_force missing".into()))?;

        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::dahl_friction bg"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: normal_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: vtan_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: fn_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: mu_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: z_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: f_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::dahl_friction encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::dahl_friction"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg, &[]);
            let groups = self.contact_count.div_ceil(WORKGROUP_SIZE).max(1);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));
        self.contact_z.mark_device_dirty();
        self.contact_friction_force.mark_device_dirty();
        Ok(())
    }
}
