//! Rust dispatch wrapper for `shaders/clamping_max_effort.wgsl`.

use crate::core::{Error, Result};
use crate::sim::Control;
use crate::storage::Buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    actuator_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const WORKGROUP_SIZE: u32 = 64;

pub struct ClampingMaxEffortKernel {
    pipeline: wgpu::ComputePipeline,
    layout_actuator: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl ClampingMaxEffortKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::clamping_max_effort"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../shaders/clamping_max_effort.wgsl").into(),
            ),
        });

        let layout_actuator = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::clamping_max_effort actuator"),
            entries: &storage_entries(2, true),
        });
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::clamping_max_effort out"),
            entries: &storage_entries(1, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::clamping_max_effort uniforms"),
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
            label: Some("molla::clamping_max_effort layout"),
            bind_group_layouts: &[&layout_actuator, &layout_out, &layout_uniforms],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::clamping_max_effort::clamp_max_effort"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("clamp_max_effort"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            layout_actuator,
            layout_out,
            layout_uniforms,
        }
    }

    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        control: &mut Control,
        dof_indices: &Buffer<u32>,
        max_effort: &Buffer<crate::math::Real>,
        actuator_count: u32,
    ) -> Result<()> {
        if actuator_count == 0 {
            return Ok(());
        }

        let bg_actuator = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::clamping_max_effort actuator bg"),
            layout: &self.layout_actuator,
            entries: &[
                buffer_entry(0, dof_indices.device_buffer())?,
                buffer_entry(1, max_effort.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::clamping_max_effort out bg"),
            layout: &self.layout_out,
            entries: &[buffer_entry(0, control.joint_force.device_buffer())?],
        });

        let uniforms = Uniforms {
            actuator_count,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::clamping_max_effort uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::clamping_max_effort uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = actuator_count.div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::clamping_max_effort encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::clamping_max_effort::clamp_max_effort"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_actuator, &[]);
            pass.set_bind_group(1, &bg_out, &[]);
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
        Error::Gpu(format!("clamping_max_effort binding {binding}: no device mirror"))
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
