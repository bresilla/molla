//! Closest-hit traversal over a refitted instance hierarchy and shared mesh atlas.

use molla_core::{Error, Result};
use wgpu::util::DeviceExt;

pub struct SceneTraceKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    params: wgpu::BindGroupLayout,
}

pub struct SceneTraceBindings {
    buffers: [wgpu::Buffer; 12],
    groups: [wgpu::BindGroup; 2],
    parameters: [u32; 8],
}

impl SceneTraceBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 12], parameters: [u32; 8]) -> bool {
        self.parameters == parameters && self.buffers.iter().zip(buffers).all(|(a, b)| a == b)
    }
}

impl SceneTraceKernel {
    pub fn new(device: &wgpu::Device) -> Result<Self> {
        if device.limits().max_storage_buffers_per_shader_stage < 12 {
            return Err(Error::Gpu(
                "scene trace requires twelve storage bindings".into(),
            ));
        }
        let entries = std::array::from_fn::<_, 12, _>(|i| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: i < 8 },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::scene trace"),
            entries: &entries,
        });
        let params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::scene trace params"),
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
            label: Some("molla::scene trace"),
            bind_group_layouts: &[Some(&layout), Some(&params)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::scene trace"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "{}\n{}\n{}",
                    molla_geometry::MeshBvhRaycastKernel::TRIANGLE_SHADER,
                    include_str!("../shaders/ray_shapes.wgsl"),
                    include_str!("../shaders/scene_trace.wgsl")
                )
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::scene trace"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });
        Ok(Self {
            pipeline,
            layout,
            params,
        })
    }

    /// Parameters are immutable: ray count, range bits, root, plane offset/count and padding.
    /// Topology records are [left, right, shape, escape]; leaf children and terminal escapes are u32::MAX.
    /// Mesh node word seven contains its mesh-local escape index.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 12],
        parameters: [u32; 8],
    ) -> SceneTraceBindings {
        let entries = std::array::from_fn::<_, 12, _>(|i| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buffers[i].as_entire_binding(),
        });
        let uniform = molla_core::gpu_buffer_init!(
            device,
            &wgpu::util::BufferInitDescriptor {
                label: Some("molla::scene trace immutable params"),
                contents: bytemuck::cast_slice(&parameters),
                usage: wgpu::BufferUsages::UNIFORM,
            }
        );
        SceneTraceBindings {
            buffers: buffers.map(Clone::clone),
            parameters,
            groups: [
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("molla::scene trace buffers"),
                    layout: &self.layout,
                    entries: &entries,
                }),
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("molla::scene trace params"),
                    layout: &self.params,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: uniform.as_entire_binding(),
                    }],
                }),
            ],
        }
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, bindings: &SceneTraceBindings) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::scene trace"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings.groups[0], &[]);
        pass.set_bind_group(1, &bindings.groups[1], &[]);
        pass.dispatch_workgroups(bindings.parameters[0].div_ceil(64), 1, 1);
    }
}
