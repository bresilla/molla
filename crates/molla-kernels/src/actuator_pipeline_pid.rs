//! Rust dispatch wrapper for `shaders/actuator_pipeline_pid.wgsl`.
//!
//! Pipeline-variant of the PID kernel: reads target pos/vel from
//! per-actuator arrays (the Delay stage's output) and writes effort
//! straight into `Control.joint_force[dof]`. Unlike the standalone
//! [`super::ActuatorPidKernel`], this kernel does **not** apply an
//! effort cap — clamping is its own pipeline stage.

use molla_core::{Error, Result};
use molla_sim::{Control, State};
use molla_storage::Buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    actuator_count: u32,
    dt: f32,
    _pad0: u32,
    _pad1: u32,
}

const WORKGROUP_SIZE: u32 = 64;

pub struct ActuatorPipelinePidKernel {
    pipeline: wgpu::ComputePipeline,
    layout_state: wgpu::BindGroupLayout,
    layout_actuator: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl ActuatorPipelinePidKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::actuator_pipeline_pid"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/actuator_pipeline_pid.wgsl").into(),
            ),
        });

        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::actuator_pipeline_pid state"),
            entries: &storage_entries(4, true),
        });
        let layout_actuator = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::actuator_pipeline_pid actuator"),
            entries: &storage_entries(6, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::actuator_pipeline_pid out"),
            entries: &storage_entries(2, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::actuator_pipeline_pid uniforms"),
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
            label: Some("molla::actuator_pipeline_pid layout"),
            bind_group_layouts: &[
                &layout_state,
                &layout_actuator,
                &layout_out,
                &layout_uniforms,
            ],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::actuator_pipeline_pid::pipeline_pid_effort"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("pipeline_pid_effort"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            layout_state,
            layout_actuator,
            layout_out,
            layout_uniforms,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        state: &State,
        control: &mut Control,
        delayed_target_pos: &Buffer<molla_math::Real>,
        delayed_target_vel: &Buffer<molla_math::Real>,
        dof_indices: &Buffer<u32>,
        kp: &Buffer<molla_math::Real>,
        ki: &Buffer<molla_math::Real>,
        kd: &Buffer<molla_math::Real>,
        integral_max: &Buffer<molla_math::Real>,
        current_integral: &Buffer<molla_math::Real>,
        next_integral: &mut Buffer<molla_math::Real>,
        actuator_count: u32,
        dt: f32,
    ) -> Result<()> {
        if actuator_count == 0 {
            return Ok(());
        }

        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_pipeline_pid state bg"),
            layout: &self.layout_state,
            entries: &[
                buffer_entry(0, state.joint_q.device_buffer())?,
                buffer_entry(1, state.joint_qd.device_buffer())?,
                buffer_entry(2, delayed_target_pos.device_buffer())?,
                buffer_entry(3, delayed_target_vel.device_buffer())?,
            ],
        });
        let bg_actuator = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_pipeline_pid actuator bg"),
            layout: &self.layout_actuator,
            entries: &[
                buffer_entry(0, dof_indices.device_buffer())?,
                buffer_entry(1, kp.device_buffer())?,
                buffer_entry(2, ki.device_buffer())?,
                buffer_entry(3, kd.device_buffer())?,
                buffer_entry(4, integral_max.device_buffer())?,
                buffer_entry(5, current_integral.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_pipeline_pid out bg"),
            layout: &self.layout_out,
            entries: &[
                buffer_entry(0, control.joint_force.device_buffer())?,
                buffer_entry(1, next_integral.device_buffer())?,
            ],
        });

        let uniforms = Uniforms {
            actuator_count,
            dt,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::actuator_pipeline_pid uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::actuator_pipeline_pid uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = actuator_count.div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::actuator_pipeline_pid encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::actuator_pipeline_pid::pipeline_pid_effort"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_state, &[]);
            pass.set_bind_group(1, &bg_actuator, &[]);
            pass.set_bind_group(2, &bg_out, &[]);
            pass.set_bind_group(3, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        control.joint_force.mark_device_dirty();
        next_integral.mark_device_dirty();
        Ok(())
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "actuator_pipeline_pid binding {binding}: no device mirror"
        ))
    })?;
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
