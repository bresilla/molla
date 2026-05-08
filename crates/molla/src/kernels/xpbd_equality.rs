//! Rust dispatch wrapper for `shaders/xpbd_equality.wgsl`.
//!
//! Single-threaded GPU port of the equality-constraint projection
//! pass. Replaces the host-side `apply_equality_post_gpu` shim — no
//! more body_q/body_qd download/upload around each XPBD substep.
//!
//! Velocity recovery is intentionally **not** done here; the existing
//! `recover_velocity` pass in `XpbdKernel` already runs after this
//! kernel and reads the post-equality `body_q_out`, so the corrected
//! pose flows through naturally.

use crate::core::{Error, Result};
use crate::sim::{Model, State};

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    body_count: u32,
    equality_count: u32,
    iterations: u32,
    dt: f32,
}

const RECOVER_WORKGROUP_SIZE: u32 = 64;

pub struct XpbdEqualityKernel {
    pipeline_project: wgpu::ComputePipeline,
    pipeline_recover: wgpu::ComputePipeline,
    layout_body: wgpu::BindGroupLayout,
    layout_eq: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl XpbdEqualityKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::xpbd_equality"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/xpbd_equality.wgsl").into(),
            ),
        });

        // Body group: body_q (RW), inv_mass, inv_inertia, com, flags,
        // body_q_in (read-only, pre-step pose), body_qd (RW, written
        // by the recovery pass).
        let layout_body = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_equality body"),
            entries: &[
                rw_storage_entry(0),
                ro_storage_entry(1),
                ro_storage_entry(2),
                ro_storage_entry(3),
                ro_storage_entry(4),
                ro_storage_entry(5),
                rw_storage_entry(6),
            ],
        });
        // Equality group: 7 read-only arrays.
        let layout_eq = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_equality eq"),
            entries: &(0..7).map(ro_storage_entry).collect::<Vec<_>>(),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::xpbd_equality uniforms"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::xpbd_equality layout"),
            bind_group_layouts: &[&layout_body, &layout_eq, &layout_uniforms],
            push_constant_ranges: &[],
        });

        let pipeline_project = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::xpbd_equality::project_equality"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("project_equality"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        let pipeline_recover = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::xpbd_equality::recover_velocity_after_equality"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("recover_velocity_after_equality"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline_project,
            pipeline_recover,
            layout_body,
            layout_eq,
            layout_uniforms,
        }
    }

    /// Run the equality-projection pass on `state_out.body_q`
    /// followed by a velocity-recovery pass that recomputes
    /// `state_out.body_qd` from `(state_out.body_q − state_in.body_q) / dt`.
    /// Replaces the host-side `apply_equality_post_gpu` shim.
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        model: &Model,
        state_in: &State,
        state_out: &mut State,
        iterations: u32,
        dt: f32,
    ) -> Result<()> {
        if model.equality_count == 0 {
            return Ok(());
        }

        let body_q_buf = state_out
            .body_q
            .device_buffer()
            .ok_or_else(missing_buf("state_out.body_q"))?;
        let body_q_in_buf = state_in
            .body_q
            .device_buffer()
            .ok_or_else(missing_buf("state_in.body_q"))?;
        let body_qd_buf = state_out
            .body_qd
            .device_buffer()
            .ok_or_else(missing_buf("state_out.body_qd"))?;

        let bg_body = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_equality body bg"),
            layout: &self.layout_body,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: body_q_buf.as_entire_binding(),
                },
                buffer_entry(1, model.body_inv_mass.device_buffer())?,
                buffer_entry(2, model.body_inv_inertia.device_buffer())?,
                buffer_entry(3, model.body_com.device_buffer())?,
                buffer_entry(4, model.body_flags.device_buffer())?,
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: body_q_in_buf.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: body_qd_buf.as_entire_binding(),
                },
            ],
        });
        let bg_eq = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_equality eq bg"),
            layout: &self.layout_eq,
            entries: &[
                buffer_entry(0, model.equality_type.device_buffer())?,
                buffer_entry(1, model.equality_body_a.device_buffer())?,
                buffer_entry(2, model.equality_body_b.device_buffer())?,
                buffer_entry(3, model.equality_anchor_a.device_buffer())?,
                buffer_entry(4, model.equality_anchor_b.device_buffer())?,
                buffer_entry(5, model.equality_relative_rotation.device_buffer())?,
                buffer_entry(6, model.equality_enabled.device_buffer())?,
            ],
        });

        let uniforms = Uniforms {
            body_count: model.body_count as u32,
            equality_count: model.equality_count as u32,
            iterations,
            dt,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::xpbd_equality uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::xpbd_equality uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::xpbd_equality encoder"),
        });
        // 1. Project equality (single-threaded Gauss-Seidel).
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::xpbd_equality::project_equality"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_project);
            pass.set_bind_group(0, &bg_body, &[]);
            pass.set_bind_group(1, &bg_eq, &[]);
            pass.set_bind_group(2, &bg_uniforms, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        // 2. Velocity recovery — one thread per body.
        {
            let groups = (model.body_count as u32).div_ceil(RECOVER_WORKGROUP_SIZE);
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::xpbd_equality::recover_velocity"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline_recover);
            pass.set_bind_group(0, &bg_body, &[]);
            pass.set_bind_group(1, &bg_eq, &[]);
            pass.set_bind_group(2, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        state_out.body_q.mark_device_dirty();
        state_out.body_qd.mark_device_dirty();
        Ok(())
    }
}

fn missing_buf(name: &'static str) -> impl FnOnce() -> Error {
    move || Error::Gpu(format!("xpbd_equality: missing device buffer for {name}"))
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!("xpbd_equality binding {binding}: no device mirror"))
    })?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    })
}

fn ro_storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
fn rw_storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
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
