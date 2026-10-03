use molla_core::{Error, Result};

/// Retained floating-point aperture sums and packed RGBA8 resolve pipelines.
pub struct CameraAccumulateKernel {
    color: wgpu::Buffer,
    sums: wgpu::Buffer,
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    layout: wgpu::BindGroupLayout,
    accumulate: wgpu::ComputePipeline,
    resolve: wgpu::ComputePipeline,
    dimensions: [u32; 3],
}

pub struct CameraAccumulateBindings {
    color: wgpu::Buffer,
    group: wgpu::BindGroup,
}

impl CameraAccumulateBindings {
    pub fn matches(&self, color: &wgpu::Buffer) -> bool {
        &self.color == color
    }
}

impl CameraAccumulateKernel {
    /// Owned aperture-sum and parameter buffer payload, excluding the borrowed color buffer.
    pub fn storage_bytes(&self) -> u64 {
        self.sums.size() + self.uniform.size()
    }

    pub fn new(device: &wgpu::Device, color: &wgpu::Buffer, dimensions: [u32; 3]) -> Result<Self> {
        let count = dimensions
            .into_iter()
            .try_fold(1u32, |product, n| product.checked_mul(n))
            .ok_or_else(|| Error::Gpu("camera accumulation: pixel count overflow".into()))?;
        let bytes = count as u64 * 16;
        let limits = device.limits();
        if count == 0
            || bytes > limits.max_buffer_size
            || bytes > limits.max_storage_buffer_binding_size
            || color.size() < count as u64 * 4
            || color.size() > limits.max_storage_buffer_binding_size
            || dimensions[0].div_ceil(8) > limits.max_compute_workgroups_per_dimension
            || dimensions[1].div_ceil(8) > limits.max_compute_workgroups_per_dimension
            || dimensions[2] > limits.max_compute_workgroups_per_dimension
        {
            return Err(Error::Gpu(
                "camera accumulation: dimensions exceed buffer/device limits".into(),
            ));
        }
        let sums = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::camera aperture sums"),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let uniform = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::camera aperture parameters"),
                size: 16,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::camera aperture layout"),
            entries: &std::array::from_fn::<_, 3, _>(|binding| wgpu::BindGroupLayoutEntry {
                binding: binding as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 2 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::camera aperture bindings"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: color.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::camera aperture"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/camera_accumulate.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::camera aperture pipeline layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry_point| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            color: color.clone(),
            sums,
            uniform,
            group,
            layout,
            accumulate: pipeline("accumulate"),
            resolve: pipeline("resolve"),
            dimensions,
        })
    }

    pub fn matches(&self, color: &wgpu::Buffer) -> bool {
        &self.color == color
    }

    /// Replace the output binding while retaining sums, parameters and pipelines.
    pub fn rebind_color(&mut self, device: &wgpu::Device, color: &wgpu::Buffer) -> Result<()> {
        let bindings = self.prepare(device, color)?;
        self.group = bindings.group;
        self.color = bindings.color;
        Ok(())
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        color: &wgpu::Buffer,
    ) -> Result<CameraAccumulateBindings> {
        if self.matches(color) {
            return Ok(CameraAccumulateBindings {
                color: color.clone(),
                group: self.group.clone(),
            });
        }
        if color.size() < self.sums.size() / 4
            || color.size() > device.limits().max_storage_buffer_binding_size
        {
            return Err(Error::Gpu(
                "camera accumulation: incompatible color buffer size".into(),
            ));
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::camera aperture bindings"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: color.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.sums.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.uniform.as_entire_binding(),
                },
            ],
        });
        Ok(CameraAccumulateBindings {
            group,
            color: color.clone(),
        })
    }

    /// Clear sums and set the divisor. Submit before resetting another pending frame.
    pub fn encode_reset(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        samples: u32,
    ) -> Result<()> {
        if samples == 0 || samples > 1024 {
            return Err(Error::Gpu(
                "camera accumulation: samples must be 1..=1024".into(),
            ));
        }
        let params = [
            self.dimensions[0],
            self.dimensions[1],
            self.dimensions[2],
            (1.0 / samples as f32).to_bits(),
        ];
        molla_core::gpu_write_buffer!(queue, &self.uniform, 0, bytemuck::cast_slice(&params));
        encoder.clear_buffer(&self.sums, 0, None);
        Ok(())
    }

    pub fn encode_accumulate(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.accumulate);
    }
    pub fn encode_resolve(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, &self.resolve);
    }

    pub fn encode_accumulate_prepared(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &CameraAccumulateBindings,
    ) {
        self.encode_group(encoder, &self.accumulate, &bindings.group);
    }

    pub fn encode_resolve_prepared(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &CameraAccumulateBindings,
    ) {
        self.encode_group(encoder, &self.resolve, &bindings.group);
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder, pipeline: &wgpu::ComputePipeline) {
        self.encode_group(encoder, pipeline, &self.group);
    }

    fn encode_group(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        pipeline: &wgpu::ComputePipeline,
        group: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::camera aperture"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(
            self.dimensions[0].div_ceil(8),
            self.dimensions[1].div_ceil(8),
            self.dimensions[2],
        );
    }
}
