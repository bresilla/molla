//! Rust dispatch wrapper for `shaders/frame_transform.wgsl`.

use crate::sensor_bindings::{SensorBindings, required_buffers};
use molla_core::Result;
use molla_sim::State;
use molla_storage::Buffer;

/// Retained input/output bindings and one parameter buffer for a frame query.
pub struct FrameTransformBindings(SensorBindings<8, 2>);

impl FrameTransformBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 8]) -> bool {
        self.0.matches(buffers)
    }
}

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
            bind_group_layouts: &[Some(&layout_in), Some(&layout_out), Some(&layout_uniforms)],
            immediate_size: 0,
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

    /// Buffer order: body poses, target bodies/positions/rotations, world
    /// positions/rotations, relative positions/rotations. All belong to `device`.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 8],
    ) -> FrameTransformBindings {
        FrameTransformBindings(SensorBindings::new(
            device,
            buffers,
            [(&self.layout_in, 0..4), (&self.layout_out, 4..8)],
            &self.layout_uniforms,
            "molla::frame_transform prepared",
        ))
    }

    /// Encode without submitting. Submit before rewriting these bindings' parameters;
    /// use separate bindings for differing parameters in the same pending batch.
    /// The caller validates active lengths/indices and marks outputs device-authoritative.
    pub fn encode_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &FrameTransformBindings,
        target_count: u32,
        has_reference: bool,
    ) {
        if target_count == 0 {
            return;
        }
        let uniforms = Uniforms {
            target_count,
            has_reference: u32::from(has_reference),
            _pad0: 0,
            _pad1: 0,
        };
        bindings.0.encode(
            queue,
            encoder,
            &self.pipeline,
            bytemuck::cast_ref(&uniforms),
            target_count.div_ceil(WORKGROUP_SIZE),
        );
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

        let buffers = [
            state.body_q.device_buffer(),
            target_body.device_buffer(),
            target_local_pos.device_buffer(),
            target_local_rot.device_buffer(),
            world_pos.device_buffer(),
            world_rot.device_buffer(),
            relative_pos.device_buffer(),
            relative_rot.device_buffer(),
        ];
        let bindings = self.prepare(device, required_buffers(buffers, "frame_transform")?);
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::frame_transform encoder"),
        });
        self.encode_prepared(queue, &mut encoder, &bindings, target_count, has_reference);
        queue.submit(std::iter::once(encoder.finish()));

        world_pos.mark_device_dirty();
        world_rot.mark_device_dirty();
        relative_pos.mark_device_dirty();
        relative_rot.mark_device_dirty();
        Ok(())
    }
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
