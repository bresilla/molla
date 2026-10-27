use molla_core::{Error, Result};

/// Retained label lookup, independent camera bounds and ordered instance compaction.
pub struct AnnotationKernel {
    pipelines: [wgpu::ComputePipeline; 3],
    group: wgpu::BindGroup,
    layout: wgpu::BindGroupLayout,
    params: wgpu::Buffer,
    limits: [u32; 3],
}

pub struct AnnotationBindings {
    group: wgpu::BindGroup,
    buffers: [wgpu::Buffer; 7],
}

impl AnnotationBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 7]) -> bool {
        self.buffers.iter().zip(buffers).all(|(a, b)| a == b)
    }
}

impl AnnotationKernel {
    /// Buffers: shapes, classes, labels, image descriptors, scratch boxes, output boxes, counts.
    pub fn new(
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 7],
        limits: [u32; 3],
    ) -> Result<Self> {
        let [pixels, images, labels] = limits;
        let slots = images as u64 * labels as u64;
        if slots > u32::MAX as u64 {
            return Err(Error::Gpu("annotation: slot count overflow".into()));
        }
        let l = device.limits();
        let required = [
            pixels as u64 * 4,
            pixels as u64 * 4,
            labels as u64 * 8,
            images as u64 * 16,
            slots * 32,
            slots * 32,
            images as u64 * 16,
        ];
        if limits.contains(&0)
            || slots > u32::MAX as u64
            || slots.max(pixels as u64).div_ceil(256)
                > l.max_compute_workgroups_per_dimension as u64
            || images > l.max_compute_workgroups_per_dimension
            || l.max_compute_invocations_per_workgroup < 256
            || l.max_compute_workgroup_size_x < 256
            || l.max_compute_workgroup_storage_size < 1024
            || l.max_storage_buffers_per_shader_stage < 7
            || buffers
                .iter()
                .zip(required)
                .any(|(b, n)| b.size() < n || b.size() > l.max_storage_buffer_binding_size)
        {
            return Err(Error::Gpu(
                "annotation: invalid device or buffer capacity".into(),
            ));
        }
        let params = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::annotation parameters"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::annotation layout"),
            entries: &std::array::from_fn::<_, 8, _>(|i| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if i == 7 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: [0, 2, 3].contains(&i),
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let all = [
            buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5], buffers[6],
            &params,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::annotation bindings"),
            layout: &layout,
            entries: &std::array::from_fn::<_, 8, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: all[i].as_entire_binding(),
            }),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::annotation"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/annotation.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::annotation pipelines"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = ["initialize", "annotate", "compact"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        Ok(Self {
            pipelines,
            group,
            layout,
            params,
            limits,
        })
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 7],
    ) -> Result<AnnotationBindings> {
        let [pixels, images, labels] = self.limits;
        let slots = images as u64 * labels as u64;
        let required = [
            pixels as u64 * 4,
            pixels as u64 * 4,
            labels as u64 * 8,
            images as u64 * 16,
            slots * 32,
            slots * 32,
            images as u64 * 16,
        ];
        if buffers
            .iter()
            .zip(required)
            .any(|(buffer, bytes)| buffer.size() < bytes)
        {
            return Err(Error::Gpu("annotation: output capacity mismatch".into()));
        }
        let all = [
            buffers[0],
            buffers[1],
            buffers[2],
            buffers[3],
            buffers[4],
            buffers[5],
            buffers[6],
            &self.params,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::direct annotation bindings"),
            layout: &self.layout,
            entries: &std::array::from_fn::<_, 8, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: all[i].as_entire_binding(),
            }),
        });
        Ok(AnnotationBindings {
            group,
            buffers: buffers.map(Clone::clone),
        })
    }

    /// Submit before parameter reuse. Count clearing is caller-owned.
    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        active: [u32; 3],
    ) -> Result<u64> {
        self.encode_group(queue, encoder, &self.group, active)
    }

    pub fn encode_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &AnnotationBindings,
        active: [u32; 3],
    ) -> Result<u64> {
        self.encode_group(queue, encoder, &bindings.group, active)
    }

    fn encode_group(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        group: &wgpu::BindGroup,
        active: [u32; 3],
    ) -> Result<u64> {
        if active.iter().zip(self.limits).any(|(&n, cap)| n > cap)
            || (active[0] != 0 && active[1] == 0)
        {
            return Err(Error::Gpu(
                "annotation: active dispatch exceeds capacity".into(),
            ));
        }
        let [pixels, images, labels] = active;
        let slots = images * labels;
        molla_core::gpu_write_buffer!(
            queue,
            &self.params,
            0,
            bytemuck::cast_slice(&[pixels, images, labels, slots]),
        );
        let groups = [
            slots.div_ceil(256),
            pixels.div_ceil(256),
            if labels == 0 { 0 } else { images },
        ];
        let mut dispatched = 0;
        for (pipeline, groups) in self.pipelines.iter().zip(groups) {
            if groups == 0 {
                continue;
            }
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::annotation stage"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            dispatched += 1;
        }
        Ok(dispatched)
    }
}
