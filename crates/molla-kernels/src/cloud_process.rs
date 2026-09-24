use molla_core::{Error, Result};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudDescriptor {
    pub counts: [u32; 4],
    pub grid: [u32; 4],
    pub ids: [u32; 4],
    pub translation: [f32; 4],
    pub rotation: [f32; 4],
    pub distance: [f32; 4],
    pub bounds_min: [f32; 4],
    pub bounds_max: [f32; 4],
    pub angles: [f32; 4],
    pub misc: [u32; 4],
}

/// Retained transform/filter, global stable prefix scan and scatter passes.
pub struct CloudProcessKernel {
    pipelines: [wgpu::ComputePipeline; 3],
    layout: wgpu::BindGroupLayout,
    ranks: wgpu::Buffer,
    blocks: wgpu::Buffer,
    group: wgpu::BindGroup,
    uniform: wgpu::Buffer,
    capacity: u32,
    max_jobs: u32,
}

pub struct CloudProcessBindings {
    group: wgpu::BindGroup,
    buffers: [wgpu::Buffer; 5],
}

impl CloudProcessBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 5]) -> bool {
        self.buffers.iter().zip(buffers).all(|(a, b)| a == b)
    }
}

impl CloudProcessKernel {
    /// Buffers are raw rays, working points, output points, descriptors and counts.
    pub fn new(
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 5],
        capacity: u32,
        max_jobs: u32,
    ) -> Result<Self> {
        let limits = device.limits();
        let sizes = [
            capacity as u64 * 56,
            capacity as u64 * 112,
            capacity as u64 * 112,
            max_jobs as u64 * 160,
            (1 + 5 * max_jobs as u64) * 4,
        ];
        if capacity == 0
            || max_jobs == 0
            || max_jobs > 1024
            || capacity.max(max_jobs).div_ceil(256) > limits.max_compute_workgroups_per_dimension
            || limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_storage_buffers_per_shader_stage < 7
            || limits.max_compute_workgroup_storage_size < 1024
            || buffers.iter().zip(sizes).any(|(b, size)| {
                b.size() < size || b.size() > limits.max_storage_buffer_binding_size as u64
            })
        {
            return Err(Error::Gpu(
                "cloud processing: invalid buffer/device capacity".into(),
            ));
        }
        let storage = |size, label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        };
        let ranks = storage(capacity as u64 * 8, "molla::cloud local ranks");
        let blocks = storage(
            capacity.div_ceil(256) as u64 * 8,
            "molla::cloud block offsets",
        );
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::cloud parameters"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let all = [
            buffers[0], buffers[1], buffers[2], &ranks, &blocks, buffers[3], buffers[4], &uniform,
        ];
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::cloud processing layout"),
            entries: &std::array::from_fn::<_, 8, _>(|i| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if i == 7 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: i == 0 || i == 5,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::cloud processing bindings"),
            layout: &layout,
            entries: &std::array::from_fn::<_, 8, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: all[i].as_entire_binding(),
            }),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::cloud pipelines"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::cloud processing"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cloud_process.wgsl").into()),
        });
        let pipelines = ["prepare", "scan_blocks", "scatter"].map(|entry| {
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
            layout,
            ranks,
            blocks,
            group,
            uniform,
            capacity,
            max_jobs,
        })
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 5],
    ) -> Result<CloudProcessBindings> {
        let sizes = [
            self.capacity as u64 * 56,
            self.capacity as u64 * 112,
            self.capacity as u64 * 112,
            self.max_jobs as u64 * 160,
            (1 + 5 * self.max_jobs as u64) * 4,
        ];
        if buffers.iter().zip(sizes).any(|(buffer, size)| {
            buffer.size() < size
                || buffer.size() > device.limits().max_storage_buffer_binding_size as u64
        }) {
            return Err(Error::Gpu(
                "cloud processing: invalid output capacity".into(),
            ));
        }
        let all = [
            buffers[0],
            buffers[1],
            buffers[2],
            &self.ranks,
            &self.blocks,
            buffers[3],
            buffers[4],
            &self.uniform,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::cloud processing direct bindings"),
            layout: &self.layout,
            entries: &std::array::from_fn::<_, 8, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: all[i].as_entire_binding(),
            }),
        });
        Ok(CloudProcessBindings {
            group,
            buffers: buffers.map(Clone::clone),
        })
    }

    pub fn encode_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &CloudProcessBindings,
        points: u32,
        jobs: u32,
    ) -> Result<()> {
        self.encode_group(queue, encoder, &bindings.group, points, jobs)
    }

    /// Encode one complete batch; submit before reusing the shared parameter buffer.
    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        points: u32,
        jobs: u32,
    ) -> Result<()> {
        self.encode_group(queue, encoder, &self.group, points, jobs)
    }

    fn encode_group(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        group: &wgpu::BindGroup,
        points: u32,
        jobs: u32,
    ) -> Result<()> {
        if points > self.capacity || jobs > self.max_jobs || (points != 0 && jobs == 0) {
            return Err(Error::Gpu(
                "cloud processing: dispatch exceeds capacity".into(),
            ));
        }
        if points == 0 {
            return Ok(());
        }
        queue.write_buffer(
            &self.uniform,
            0,
            bytemuck::cast_slice(&[points, jobs, points.div_ceil(256), 0]),
        );
        for (pipeline, groups) in
            self.pipelines
                .iter()
                .zip([points.div_ceil(256), 1, points.max(jobs).div_ceil(256)])
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::cloud processing pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        Ok(())
    }
}
