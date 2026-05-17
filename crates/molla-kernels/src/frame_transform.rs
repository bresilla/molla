//! Rust dispatch wrapper for `shaders/frame_transform.wgsl`.

use molla_core::{Error, Result};
use molla_sim::State;
use molla_storage::Buffer;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Uniforms {
    target_count: u32,
    has_reference: u32,
    _pad0: u32,
    _pad1: u32,
}

const WORKGROUP_SIZE: u32 = 64;

pub struct FrameTransformKernel {
    pipeline: wgpu::ComputePipeline,
    layout_in: wgpu::BindGroupLayout,
    layout_out: wgpu::BindGroupLayout,
    layout_uniforms: wgpu::BindGroupLayout,
}

impl FrameTransformKernel {
    pub fn new(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::frame_transform"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/frame_transform.wgsl").into(),
            ),
        });

        // 4 read-only inputs.
        let layout_in = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::frame_transform in"),
            entries: &storage_entries(4, true),
        });
        // 4 read-write outputs (world_pos, world_rot, rel_pos, rel_rot).
        let layout_out = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::frame_transform out"),
            entries: &storage_entries(4, false),
        });
        let layout_uniforms = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::frame_transform uniforms"),
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
            label: Some("molla::frame_transform layout"),
            bind_group_layouts: &[&layout_in, &layout_out, &layout_uniforms],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::frame_transform::compute_frame_transforms"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("compute_frame_transforms"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });

        Self {
            pipeline,
            layout_in,
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
        target_body: &Buffer<u32>,
        target_local_pos: &Buffer<molla_math::Vec3>,
        target_local_rot: &Buffer<molla_math::Quat>,
        world_pos: &mut Buffer<molla_math::Vec3>,
        world_rot: &mut Buffer<molla_math::Quat>,
        relative_pos: &mut Buffer<molla_math::Vec3>,
        relative_rot: &mut Buffer<molla_math::Quat>,
        target_count: u32,
        has_reference: bool,
    ) -> Result<()> {
        if target_count == 0 {
            return Ok(());
        }

        let bg_in = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::frame_transform in bg"),
            layout: &self.layout_in,
            entries: &[
                buffer_entry(0, state.body_q.device_buffer())?,
                buffer_entry(1, target_body.device_buffer())?,
                buffer_entry(2, target_local_pos.device_buffer())?,
                buffer_entry(3, target_local_rot.device_buffer())?,
            ],
        });
        let bg_out = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::frame_transform out bg"),
            layout: &self.layout_out,
            entries: &[
                buffer_entry(0, world_pos.device_buffer())?,
                buffer_entry(1, world_rot.device_buffer())?,
                buffer_entry(2, relative_pos.device_buffer())?,
                buffer_entry(3, relative_rot.device_buffer())?,
            ],
        });

        let uniforms = Uniforms {
            target_count,
            has_reference: u32::from(has_reference),
            _pad0: 0,
            _pad1: 0,
        };
        let uniform_buf = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::frame_transform uniforms"),
            size: core::mem::size_of::<Uniforms>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(&uniform_buf, 0, bytemuck::bytes_of(&uniforms));
        let bg_uniforms = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::frame_transform uniforms bg"),
            layout: &self.layout_uniforms,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buf.as_entire_binding(),
            }],
        });

        let groups = target_count.div_ceil(WORKGROUP_SIZE);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::frame_transform encoder"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::frame_transform::compute_frame_transforms"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bg_in, &[]);
            pass.set_bind_group(1, &bg_out, &[]);
            pass.set_bind_group(2, &bg_uniforms, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        queue.submit(std::iter::once(encoder.finish()));

        world_pos.mark_device_dirty();
        world_rot.mark_device_dirty();
        relative_pos.mark_device_dirty();
        relative_rot.mark_device_dirty();
        Ok(())
    }
}

fn buffer_entry<'a>(
    binding: u32,
    buf: Option<&'a wgpu::Buffer>,
) -> Result<wgpu::BindGroupEntry<'a>> {
    let buf = buf.ok_or_else(|| {
        Error::Gpu(format!(
            "frame_transform binding {binding}: no device mirror"
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
