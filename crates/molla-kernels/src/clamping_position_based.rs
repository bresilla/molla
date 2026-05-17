//! Rust dispatch wrapper for `shaders/clamping_position_based.wgsl`.

use molla_core::{Error, Result};
use molla_sim::{Control, State};
use molla_storage::Buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    actuator_count: u32,
    lookup_size: u32,
    _pad0: u32,
    _pad1: u32,
}

const WORKGROUP_SIZE: u32 = 64;

pub struct ClampingPositionBasedKernel {
    pipeline: wgpu::ComputePipeline,
    layout_actuator: wgpu::BindGroupLayout,
    layout_state: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl ClampingPositionBasedKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::clamping_position_based"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/clamping_position_based.wgsl").into(),
            ),
        });

        let layout_actuator = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::clamping_position_based actuator"),
            entries: &storage_entries(3, true),
        });
        // joint_q read-only, joint_force read-write.
        let layout_state = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::clamping_position_based state"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::clamping_position_based uniforms"),
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
            label: Some("molla::clamping_position_based layout"),
            bind_group_layouts: &[&layout_actuator, &layout_state, &layout_uniforms],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::clamping_position_based::clamp_position_based"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("clamp_position_based"),
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
        lookup_positions: &Buffer<molla_math::Real>,
        lookup_efforts: &Buffer<molla_math::Real>,
        actuator_count: u32,
        lookup_size: u32,
    ) -> Result<()> {
        if actuator_count == 0 {
            return Ok(());
        }

        let bg_actuator = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::clamping_position_based actuator bg"),
            layout: &self.layout_actuator,
            entries: &[
                buffer_entry(0, dof_indices.device_buffer())?,
                buffer_entry(1, lookup_positions.device_buffer())?,
                buffer_entry(2, lookup_efforts.device_buffer())?,
            ],
        });
        let bg_state = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::clamping_position_based state bg"),
            layout: &self.layout_state,
            entries: &[
                buffer_entry(0, state.joint_q.device_buffer())?,
                buffer_entry(1, control.joint_force.device_buffer())?,
            ],
        });

        let uniforms = Uniforms {
            actuator_count,
            lookup_size,
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::clamping_position_based uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::clamping_position_based uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = actuator_count.div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::clamping_position_based encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::clamping_position_based::clamp_position_based"),
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
            "clamping_position_based binding {binding}: no device mirror"
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
