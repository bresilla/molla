use molla_core::{Error, Result};
use std::num::NonZeroU64;

/// Retained packed-image adjustments with per-stage uniforms and white-balance reductions.
pub struct ImageProcessKernel {
    uniform: wgpu::Buffer,
    group: wgpu::BindGroup,
    layout: wgpu::BindGroupLayout,
    scratch: wgpu::Buffer,
    apply: wgpu::ComputePipeline,
    reduce: wgpu::ComputePipeline,
    finish: wgpu::ComputePipeline,
    capacity: u32,
    slots: u32,
    stride: u64,
}

pub struct ImageProcessBindings {
    group: wgpu::BindGroup,
    buffers: [wgpu::Buffer; 2],
}

impl ImageProcessBindings {
    pub fn matches(&self, pixels: &wgpu::Buffer, gains: &wgpu::Buffer) -> bool {
        self.buffers[0] == *pixels && self.buffers[1] == *gains
    }
}

impl ImageProcessKernel {
    pub fn new(
        device: &wgpu::Device,
        pixels: &wgpu::Buffer,
        gains: &wgpu::Buffer,
        capacity: u32,
        slots: u32,
    ) -> Result<Self> {
        let limits = device.limits();
        let stride = 32u64.div_ceil(limits.min_uniform_buffer_offset_alignment as u64)
            * limits.min_uniform_buffer_offset_alignment as u64;
        let bytes = slots as u64 * stride;
        if capacity == 0
            || slots == 0
            || slots > 4096
            || capacity.div_ceil(256) > limits.max_compute_workgroups_per_dimension
            || pixels.size() < capacity as u64 * 4
            || gains.size() < slots as u64 * 16
            || [
                pixels.size(),
                gains.size(),
                capacity.div_ceil(256) as u64 * 16,
            ]
            .into_iter()
            .any(|size| {
                size > limits.max_storage_buffer_binding_size || size > limits.max_buffer_size
            })
            || bytes > limits.max_buffer_size
            || bytes > u32::MAX as u64
            || limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_workgroup_storage_size < 4096
        {
            return Err(Error::Gpu(
                "image processing: capacity exceeds device limits".into(),
            ));
        }
        let scratch = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::image reduction partials"),
                size: capacity.div_ceil(256) as u64 * 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }
        );
        let uniform = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::image processing parameters"),
                size: bytes,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::image processing layout"),
            entries: &std::array::from_fn::<_, 4, _>(|i| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if i == 3 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: false }
                    },
                    has_dynamic_offset: i == 3,
                    min_binding_size: if i == 3 { NonZeroU64::new(32) } else { None },
                },
                count: None,
            }),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::image processing bindings"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pixels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scratch.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gains.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &uniform,
                        offset: 0,
                        size: NonZeroU64::new(32),
                    }),
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::image processing"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/image_process.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::image processing pipelines"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Ok(Self {
            uniform,
            group,
            layout,
            scratch,
            apply: pipeline("apply"),
            reduce: pipeline("reduce"),
            finish: pipeline("finish_reduction"),
            capacity,
            slots,
            stride,
        })
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        pixels: &wgpu::Buffer,
        gains: &wgpu::Buffer,
    ) -> Result<ImageProcessBindings> {
        if pixels.size() < self.capacity as u64 * 4 || gains.size() < self.slots as u64 * 16 {
            return Err(Error::Gpu(
                "image processing: output capacity mismatch".into(),
            ));
        }
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::direct image processing bindings"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pixels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: self.scratch.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: gains.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &self.uniform,
                        offset: 0,
                        size: NonZeroU64::new(32),
                    }),
                },
            ],
        });
        Ok(ImageProcessBindings {
            group,
            buffers: [pixels.clone(), gains.clone()],
        })
    }

    /// Encode one stage. Distinct pending stages need distinct slots; submit before slot reuse.
    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        slot: u32,
        start: u32,
        count: u32,
        operation: u32,
        a: f32,
        b: f32,
    ) -> Result<u32> {
        self.encode_group(
            queue,
            encoder,
            &self.group,
            slot,
            start,
            count,
            operation,
            a,
            b,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn encode_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &ImageProcessBindings,
        slot: u32,
        start: u32,
        count: u32,
        operation: u32,
        a: f32,
        b: f32,
    ) -> Result<u32> {
        self.encode_group(
            queue,
            encoder,
            &bindings.group,
            slot,
            start,
            count,
            operation,
            a,
            b,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn encode_group(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        group: &wgpu::BindGroup,
        slot: u32,
        start: u32,
        count: u32,
        operation: u32,
        a: f32,
        b: f32,
    ) -> Result<u32> {
        if slot >= self.slots
            || start
                .checked_add(count)
                .is_none_or(|end| end > self.capacity)
            || operation > 9
            || !a.is_finite()
            || !b.is_finite()
        {
            return Err(Error::Gpu("image processing: invalid stage".into()));
        }
        let params = [
            start,
            count,
            operation,
            slot,
            a.to_bits(),
            b.to_bits(),
            count.div_ceil(256),
            0,
        ];
        molla_core::gpu_write_buffer!(
            queue,
            &self.uniform,
            slot as u64 * self.stride,
            bytemuck::cast_slice(&params),
        );
        let mut dispatches = 0;
        if operation >= 8 {
            if count != 0 {
                self.pass(encoder, group, &self.reduce, slot, count.div_ceil(256));
                dispatches += 1;
            }
            self.pass(encoder, group, &self.finish, slot, 1);
            dispatches += 1;
        }
        if count != 0 {
            self.pass(encoder, group, &self.apply, slot, count.div_ceil(256));
            dispatches += 1;
        }
        Ok(dispatches)
    }

    fn pass(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        group: &wgpu::BindGroup,
        pipeline: &wgpu::ComputePipeline,
        slot: u32,
        groups: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("molla::image processing stage"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, group, &[(slot as u64 * self.stride) as u32]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}
