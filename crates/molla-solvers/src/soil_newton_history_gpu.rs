use molla_compute::Engine;
use molla_core::{Error, Result};

pub(crate) struct NewtonHistory {
    pipelines: [wgpu::ComputePipeline; 2],
    group: wgpu::BindGroup,
    restore_group: Option<wgpu::BindGroup>,
    layout: wgpu::BindGroupLayout,
    buffers: [wgpu::Buffer; 11],
    dimensions: [u32; 2],
    header: wgpu::Buffer,
    seed: wgpu::Buffer,
    groups: u32,
}

fn invalid() -> Error {
    Error::Solver("Newton history: invalid capacity, buffers or device limits".into())
}

impl NewtonHistory {
    pub(crate) fn required_bytes(capacity: u32) -> Result<u64> {
        if !(1..=16384).contains(&capacity) {
            return Err(invalid());
        }
        Ok(176 * u64::from(capacity) + 32)
    }

    pub(crate) fn new(
        engine: &Engine,
        inputs: [&wgpu::Buffer; 7],
        capacity: u32,
        nodes: u32,
        budget: u64,
    ) -> Result<Self> {
        engine.check_device()?;
        let bytes = Self::required_bytes(capacity)?;
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        let c = u64::from(capacity);
        if nodes == 0
            || bytes > budget
            || limits.max_storage_buffers_per_shader_stage < 10
            || limits.max_bindings_per_bind_group < 11
            || limits.max_uniform_buffers_per_shader_stage < 1
            || limits.max_compute_invocations_per_workgroup < 64
            || limits.max_compute_workgroup_size_x < 64
            || capacity.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || 112 * c > limits.max_buffer_size
            || 112 * c > limits.max_storage_buffer_binding_size
            || inputs
                .iter()
                .zip([4 * c, 48 * c, 64 * c, 64 * c, 4, 4, 4])
                .any(|(b, n)| {
                    b.size() < n
                        || b.size() > limits.max_storage_buffer_binding_size
                        || !b.usage().contains(wgpu::BufferUsages::STORAGE)
                })
            || inputs
                .iter()
                .enumerate()
                .any(|(i, b)| inputs[..i].contains(b))
        {
            return Err(invalid());
        }
        let create = |label, size, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage,
                    mapped_at_creation: false
                }
            )
        };
        let usage = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_DST
            | wgpu::BufferUsages::COPY_SRC;
        let history = create("Newton matched history", 112 * c, usage);
        let seed = create("Newton initial guess", 64 * c, usage);
        let header = create("Newton history validity and counters", 16, usage);
        let params = create(
            "Newton history capacity",
            16,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[capacity, nodes, 0, 0])
        );
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Newton history"),
            entries: &(0..11)
                .map(|binding| wgpu::BindGroupLayoutEntry {
                    binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: if binding == 9 {
                            wgpu::BufferBindingType::Uniform
                        } else {
                            wgpu::BufferBindingType::Storage {
                                read_only: !matches!(binding, 3 | 7 | 8 | 10),
                            }
                        },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                })
                .collect::<Vec<_>>(),
        });
        let buffers = [
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6], &history,
            &header, &params, &seed,
        ];
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Newton history"),
            layout: &layout,
            entries: &buffers
                .iter()
                .enumerate()
                .map(|(i, b)| wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Newton history"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Newton history"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_newton_history.wgsl").into(),
            ),
        });
        let pipelines = ["restore", "save"].map(|entry| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        });
        let retained_buffers = buffers.map(Clone::clone);
        Ok(Self {
            pipelines,
            group,
            restore_group: None,
            layout,
            buffers: retained_buffers,
            dimensions: [capacity, nodes],
            header,
            seed,
            groups: capacity.div_ceil(64),
        })
    }

    pub(crate) fn invalidate(&self, queue: &wgpu::Queue) {
        molla_core::gpu_write_buffer!(queue, &self.header, 0, bytemuck::cast_slice(&[0u32; 4]));
    }
    pub(crate) fn use_previous(&mut self, device: &wgpu::Device, previous: &Self) -> Result<()> {
        if self.dimensions != previous.dimensions || self.header == previous.header {
            return Err(invalid());
        }
        let mut buffers = self.buffers.each_ref();
        buffers[7] = &previous.buffers[7];
        buffers[8] = &previous.header;
        self.restore_group = Some(
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Newton preceding-parity history"),
                layout: &self.layout,
                entries: &buffers
                    .iter()
                    .enumerate()
                    .map(|(i, b)| wgpu::BindGroupEntry {
                        binding: i as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            }),
        );
        Ok(())
    }
    pub(crate) fn restore(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode(encoder, 0);
    }
    pub(crate) fn seed_buffer(&self) -> &wgpu::Buffer {
        &self.seed
    }
    pub(crate) fn save(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.header, 0, Some(8));
        self.encode(encoder, 1);
    }
    fn encode(&self, encoder: &mut wgpu::CommandEncoder, stage: usize) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipelines[stage]);
        let group = if stage == 0 {
            self.restore_group.as_ref().unwrap_or(&self.group)
        } else {
            &self.group
        };
        pass.set_bind_group(0, group, &[]);
        pass.dispatch_workgroups(self.groups, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
    #[cfg(test)]
    pub(crate) fn counters(&self) -> &wgpu::Buffer {
        &self.header
    }
}

#[cfg(test)]
#[path = "soil_newton_history_tests.rs"]
mod tests;
