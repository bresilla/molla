//! GPU rigid-instance bounds and level-ordered hierarchy refitting.

use wgpu::util::DeviceExt;

pub struct SceneBoundsKernel {
    leaves: wgpu::ComputePipeline,
    parents: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    levels: Vec<(wgpu::BindGroup, u32)>,
}

pub struct SceneBoundsBindings {
    buffers: [wgpu::Buffer; 7],
    group: wgpu::BindGroup,
}

impl SceneBoundsBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 7]) -> bool {
        self.buffers.iter().zip(buffers).all(|(a, b)| a == b)
    }
}

impl SceneBoundsKernel {
    /// Levels contain leaves first, then contiguous parent ranges in increasing height.
    pub fn new(device: &wgpu::Device, levels: &[(u32, u32)], body_count: u32) -> Self {
        let entries = std::array::from_fn::<_, 7, _>(|i| wgpu::BindGroupLayoutEntry {
            binding: i as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: i < 3 },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::scene bounds"),
            entries: &entries,
        });
        let params = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::scene bounds parameters"),
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
            label: Some("molla::scene bounds"),
            bind_group_layouts: &[Some(&layout), Some(&params)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::scene bounds"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/scene_bounds.wgsl").into()),
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("molla::scene bounds"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let levels = levels
            .iter()
            .map(|&(start, count)| {
                let buffer = molla_core::gpu_buffer_init!(
                    device,
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("molla::scene bounds immutable parameters"),
                        contents: bytemuck::cast_slice(&[start, count, body_count, 0]),
                        usage: wgpu::BufferUsages::UNIFORM,
                    }
                );
                let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("molla::scene bounds level"),
                    layout: &params,
                    entries: &[wgpu::BindGroupEntry {
                        binding: 0,
                        resource: buffer.as_entire_binding(),
                    }],
                });
                (group, count.div_ceil(64))
            })
            .collect();
        Self {
            leaves: pipeline("leaves"),
            parents: pipeline("parents"),
            layout,
            levels,
        }
    }

    /// Bind instances, topology, live body poses, lower/upper bounds, world poses and status.
    pub fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 7],
    ) -> SceneBoundsBindings {
        let entries = std::array::from_fn::<_, 7, _>(|i| wgpu::BindGroupEntry {
            binding: i as u32,
            resource: buffers[i].as_entire_binding(),
        });
        SceneBoundsBindings {
            buffers: buffers.map(Clone::clone),
            group: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("molla::scene bounds bindings"),
                layout: &self.layout,
                entries: &entries,
            }),
        }
    }

    /// Encode one dependency-ordered dispatch per nonempty level without queue writes or submission.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &SceneBoundsBindings,
    ) -> u64 {
        for (i, (group, workgroups)) in self.levels.iter().enumerate() {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::scene bounds refit"),
                timestamp_writes: None,
            });
            pass.set_pipeline(if i == 0 { &self.leaves } else { &self.parents });
            pass.set_bind_group(0, &bindings.group, &[]);
            pass.set_bind_group(1, group, &[]);
            pass.dispatch_workgroups(*workgroups, 1, 1);
        }
        self.levels.len() as u64
    }
}
