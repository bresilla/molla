use molla_core::{Error, Result};

#[repr(C)]
#[derive(Debug, Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CloudGridParams {
    pub minimum: [f32; 4],
    pub maximum: [f32; 4],
    pub size: [f32; 4],
    pub dimensions: [u32; 4],
    pub counts: [u32; 4],
}

/// Retained CSR membership, cell-prefix and geometric reduction passes.
pub struct CloudGridKernel {
    pipelines: [wgpu::ComputePipeline; 5],
    group: wgpu::BindGroup,
    layout: wgpu::BindGroupLayout,
    uniform: wgpu::Buffer,
    scratch: wgpu::Buffer,
    memberships: wgpu::Buffer,
    blocks: wgpu::Buffer,
    max_points: u32,
    max_cells: u32,
    max_groups: u32,
}

pub struct CloudGridBindings {
    group: wgpu::BindGroup,
    buffers: [wgpu::Buffer; 4],
}

impl CloudGridBindings {
    pub fn matches(&self, buffers: [&wgpu::Buffer; 4]) -> bool {
        self.buffers.iter().zip(buffers).all(|(a, b)| a == b)
    }
}

impl CloudGridKernel {
    /// Buffers: packed points, sorted-by-cell indices, cell summaries, diagnostics.
    pub fn new(
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 4],
        max_points: u32,
        max_cells: u32,
    ) -> Result<Self> {
        let limits = device.limits();
        let required = [
            max_points as u64 * 112,
            max_points as u64 * 4,
            max_cells as u64 * 64,
            24,
        ];
        if max_points == 0
            || max_cells == 0
            || max_points.max(max_cells).div_ceil(256) > limits.max_compute_workgroups_per_dimension
            || max_cells as u64 > (limits.max_compute_workgroups_per_dimension as u64).pow(2)
            || limits.max_compute_invocations_per_workgroup < 256
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_workgroup_storage_size < 13_312
            || limits.max_storage_buffers_per_shader_stage < 7
            || buffers.iter().zip(required).any(|(buffer, bytes)| {
                buffer.size() < bytes
                    || buffer.size() > limits.max_storage_buffer_binding_size as u64
            })
        {
            return Err(Error::Gpu(
                "cloud grid: invalid buffer or device capacity".into(),
            ));
        }
        let storage = |size, label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let memberships = storage(max_points as u64 * 4, "molla::cloud grid memberships");
        let scratch = storage(
            max_cells as u64 * 12,
            "molla::cloud grid counts/offsets/cursors",
        );
        let blocks = storage(
            max_cells.div_ceil(256) as u64 * 8,
            "molla::cloud grid block offsets",
        );
        let uniform = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("molla::cloud grid parameters"),
            size: 80,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let all = [
            buffers[0],
            &memberships,
            buffers[1],
            &scratch,
            &blocks,
            buffers[2],
            buffers[3],
            &uniform,
        ];
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("molla::cloud grid layout"),
            entries: &std::array::from_fn::<_, 8, _>(|i| wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if i == 7 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage { read_only: i == 0 }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }),
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::cloud grid bindings"),
            layout: &layout,
            entries: &std::array::from_fn::<_, 8, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: all[i].as_entire_binding(),
            }),
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("molla::cloud grid"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/cloud_grid.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("molla::cloud grid pipelines"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipelines = [
            "classify",
            "scan_cells",
            "scan_blocks",
            "scatter",
            "reduce_cells",
        ]
        .map(|entry| {
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
            uniform,
            scratch,
            memberships,
            blocks,
            max_points,
            max_cells,
            max_groups: limits.max_compute_workgroups_per_dimension,
        })
    }

    pub fn prepare(
        &self,
        device: &wgpu::Device,
        buffers: [&wgpu::Buffer; 4],
    ) -> Result<CloudGridBindings> {
        let required = [
            self.max_points as u64 * 112,
            self.max_points as u64 * 4,
            self.max_cells as u64 * 64,
            24,
        ];
        if buffers
            .iter()
            .zip(required)
            .any(|(buffer, bytes)| buffer.size() < bytes)
        {
            return Err(Error::Gpu("cloud grid: output capacity mismatch".into()));
        }
        let all = [
            buffers[0],
            &self.memberships,
            buffers[1],
            &self.scratch,
            &self.blocks,
            buffers[2],
            buffers[3],
            &self.uniform,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("molla::direct cloud grid bindings"),
            layout: &self.layout,
            entries: &std::array::from_fn::<_, 8, _>(|i| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: all[i].as_entire_binding(),
            }),
        });
        Ok(CloudGridBindings {
            group,
            buffers: buffers.map(Clone::clone),
        })
    }

    /// Submit before parameter reuse. Output/diagnostic clearing is caller-owned.
    pub fn encode(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        params: CloudGridParams,
    ) -> Result<()> {
        self.encode_group(queue, encoder, &self.group, params)
    }

    pub fn encode_prepared(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        bindings: &CloudGridBindings,
        params: CloudGridParams,
    ) -> Result<()> {
        self.encode_group(queue, encoder, &bindings.group, params)
    }

    fn encode_group(
        &self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        group: &wgpu::BindGroup,
        mut params: CloudGridParams,
    ) -> Result<()> {
        let points = params.counts[0];
        let cells = params.counts[1];
        if points > self.max_points || cells == 0 || cells > self.max_cells {
            return Err(Error::Gpu("cloud grid: dispatch exceeds capacity".into()));
        }
        if params.dimensions[..3]
            .iter()
            .try_fold(1u32, |n, &d| n.checked_mul(d))
            != Some(cells)
            || (0..3).any(|i| {
                !params.minimum[i].is_finite()
                    || !params.maximum[i].is_finite()
                    || params.maximum[i] <= params.minimum[i]
                    || !params.size[i].is_finite()
                    || params.size[i] < f32::MIN_POSITIVE
            })
        {
            return Err(Error::Gpu(
                "cloud grid: invalid dimensions or bounds".into(),
            ));
        }
        if points == 0 {
            return Ok(());
        }
        let columns = cells.min(self.max_groups);
        params.counts[2] = cells.div_ceil(256);
        params.counts[3] = columns;
        queue.write_buffer(&self.uniform, 0, bytemuck::bytes_of(&params));
        encoder.clear_buffer(&self.scratch, 0, Some(cells as u64 * 12));
        let groups = [
            [points.div_ceil(256), 1],
            [cells.div_ceil(256), 1],
            [1, 1],
            [points.div_ceil(256), 1],
            [columns, cells.div_ceil(columns)],
        ];
        for (pipeline, [x, y]) in self.pipelines.iter().zip(groups) {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("molla::cloud grid stage"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(x, y, 1);
        }
        Ok(())
    }
}
