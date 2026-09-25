//! Persistent bindings for initializing and merging closest ray hits on device.

/// Fixed ray, scratch and output bindings for ordered mesh-hit reduction.
pub struct RaycastMergeKernel {
    pipeline: wgpu::ComputePipeline,
    layout: wgpu::BindGroupLayout,
    buffers: [wgpu::Buffer; 9],
    initial: RaycastMergeBindings,
}

/// A separate parameter buffer for one ordered merge in a command batch.
pub struct RaycastMergeBindings {
    bindings: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    buffers: [wgpu::Buffer; 9],
}

impl RaycastMergeBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 9]) -> bool {
        self.buffers.iter().zip(buffers).all(|(a, b)| a == b)
    }
}

impl RaycastMergeKernel {
    /// Bind origins, normalized directions, mesh t/triangle/normal, then output t/shape/point/normal.
    pub fn new(device: &wgpu::Device, buffers: [&wgpu::Buffer; 9]) -> Self {
        let mut entries: Vec<_> = (0..9)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding < 5,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect();
        entries.push(wgpu::BindGroupLayoutEntry {
            binding: 9,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::raycast_merge layout"),
            entries: &entries,
        });
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::raycast_merge uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 9,
            resource: uniform.as_entire_binding(),
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::raycast_merge bindings"),
            layout: &layout,
            entries: &entries,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::raycast_merge"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/raycast_merge.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::raycast_merge pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("molla::raycast_merge"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("merge_hits"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            pipeline,
            layout,
            buffers: std::array::from_fn(|i| buffers[i].clone()),
            initial: RaycastMergeBindings {
                bindings,
                uniform,
                buffers: std::array::from_fn(|i| buffers[i].clone()),
            },
        }
    }

    pub fn prepare(&self, device: &wgpu::Device) -> RaycastMergeBindings {
        self.prepare_with_buffers(device, std::array::from_fn(|i| &self.buffers[i]))
    }

    /// Retained bindings for explicit ray, scratch and output buffers on this device.
    pub fn prepare_with_buffers(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 9],
    ) -> RaycastMergeBindings {
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::raycast merge instance uniform"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 9,
            resource: uniform.as_entire_binding(),
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::raycast merge instance"),
            layout: &self.layout,
            entries: &entries,
        });
        RaycastMergeBindings {
            bindings,
            uniform,
            buffers: std::array::from_fn(|i| buffers[i].clone()),
        }
    }

    pub fn encode_initial(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        ray_count: u32,
        max_distance: f32,
    ) {
        self.encode_prepared(queue, encoder, &self.initial, ray_count, None, max_distance);
    }

    /// Use a distinct binding set per merge; submit before rewriting its parameters.
    pub fn encode_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &RaycastMergeBindings,
        ray_count: u32,
        shape_id: Option<u32>,
        max_distance: f32,
    ) {
        if ray_count == 0 {
            return;
        }
        let params = [
            ray_count,
            shape_id.unwrap_or(0),
            max_distance.to_bits(),
            u32::from(shape_id.is_none()),
        ];
        queue.write_buffer(&bindings.uniform, 0, bytemuck::cast_slice(&params));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::raycast merge prepared"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &bindings.bindings, &[]);
        pass.dispatch_workgroups(ray_count.div_ceil(64), 1, 1);
    }

    /// Initialize misses when `shape_id` is `None`, otherwise merge a mesh candidate.
    /// Equal distances retain the earlier result. Calls submit in queue order.
    pub fn dispatch(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        ray_count: u32,
        shape_id: Option<u32>,
        max_distance: f32,
    ) {
        if ray_count == 0 {
            return;
        }
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("molla::raycast_merge"),
        });
        self.encode_prepared(
            queue,
            &mut encoder,
            &self.initial,
            ray_count,
            shape_id,
            max_distance,
        );
        queue.submit([encoder.finish()]);
    }
}
