//! Rust dispatch wrapper for `shaders/clamping_dc_motor.wgsl`.

use molla_core::{Error, Result};
use molla_sim::{Control, State};
use molla_storage::Buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    actuator_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64;

pub struct ClampingDcMotorKernel {
    pipeline: wgpu::ComputePipeline,
    layout_actuator: wgpu::BindGroupLayout,
    layout_state: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl ClampingDcMotorKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::clamping_dc_motor"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/clamping_dc_motor.wgsl").into(),
            ),
        });

        let layout_actuator = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::clamping_dc_motor actuator"),
            entries: &storage_entries_mixed(&[true, true, true, true, true]),
        });
        // joint_qd read-only, joint_force read-write.
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::clamping_dc_motor state"),
            entries: &storage_entries_mixed(&[true, false]),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::clamping_dc_motor uniforms"),
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
            label: Some("molla::clamping_dc_motor layout"),
            bind_group_layouts: &[
                Some(&layout_actuator),
                Some(&layout_state),
                Some(&layout_uniforms),
            ],
            immediate_size: 0,
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::clamping_dc_motor::clamp_dc_motor"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("clamp_dc_motor"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            layout_actuator,
            layout_state,
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
        dof_indices: &Buffer<u32>,
        stall_torque: &Buffer<molla_math::Real>,
        no_load_speed: &Buffer<molla_math::Real>,
        max_motor_effort: &Buffer<molla_math::Real>,
        corner_velocity: &Buffer<molla_math::Real>,
        actuator_count: u32,
    ) -> Result<()> {
        if actuator_count == 0 {
            return Ok(());
        }

        let bg_actuator = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::clamping_dc_motor actuator bg"),
            layout: &self.layout_actuator,
            entries: &[
                buffer_entry(0, dof_indices.device_buffer())?,
                buffer_entry(1, stall_torque.device_buffer())?,
                buffer_entry(2, no_load_speed.device_buffer())?,
                buffer_entry(3, max_motor_effort.device_buffer())?,
                buffer_entry(4, corner_velocity.device_buffer())?,
            ],
        });
        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::clamping_dc_motor state bg"),
            layout: &self.layout_state,
            entries: &[
                buffer_entry(0, state.joint_qd.device_buffer())?,
                buffer_entry(1, control.joint_force.device_buffer())?,
            ],
        });

        let uniforms = Uniforms {
            actuator_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::clamping_dc_motor uniforms"),
                size: core::mem::size_of::<Uniforms>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        molla_core::gpu_write_buffer!(queue, &uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::clamping_dc_motor uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = actuator_count.div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::clamping_dc_motor encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::clamping_dc_motor::clamp_dc_motor"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_actuator, &[]);
            pass.set_bind_group(1, &bg_state, &[]);
            pass.set_bind_group(2, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        control.joint_force.mark_device_dirty();
        Ok(())
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "clamping_dc_motor binding {binding}: no device mirror"
        ))
    })?;
    Ok(wgpu::BindGroupEntry {
        binding,
        resource: buf.as_entire_binding(),
    })
}

fn storage_entries_mixed(read_only: &[bool]) -> Vec<wgpu::BindGroupLayoutEntry> {
    read_only
        .iter()
        .enumerate()
        .map(|(i, ro)| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: *ro },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        })
        .collect()
}
