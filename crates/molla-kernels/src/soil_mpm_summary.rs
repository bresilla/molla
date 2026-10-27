//! Retained reduction of MPM particle compaction into a 16-byte summary.

use molla_core::{Error, Result};

pub(super) struct CompactionSummaryPass {
    pub output: wgpu::Buffer,
    pipeline: wgpu::ComputePipeline,
    bindings: wgpu::BindGroup,
    count: u32,
}

impl CompactionSummaryPass {
    pub fn validate(
        device: &wgpu::Device,
        source: &wgpu::Buffer,
        count: u32,
        threshold: f32,
    ) -> Result<()> {
        let limits = device.limits();
        if !threshold.is_finite()
            || threshold < 0.
            || count.div_ceil(64) > limits.max_compute_workgroups_per_dimension
            || source.size() < u64::from(count.max(1)) * 16
            || source.size() > limits.max_storage_buffer_binding_size
            || !source.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(Error::Solver(
                "MPM summary: invalid source, count or threshold".into(),
            ));
        }
        Ok(())
    }

    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        source: &wgpu::Buffer,
        count: u32,
        threshold: f32,
    ) -> Result<Self> {
        Self::validate(device, source, count, threshold)?;
        let allocate = |label, usage| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: 16,
                    usage,
                    mapped_at_creation: false,
                }
            )
        };
        let output = allocate(
            "MPM compact statistics",
            wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        );
        let uniform = allocate(
            "MPM summary parameters",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        );
        molla_core::gpu_write_buffer!(
            queue,
            &uniform,
            0,
            bytemuck::cast_slice(&[count, threshold.to_bits(), 0, 0])
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MPM compact summary"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_mpm_summary.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MPM compact summary"),
            layout: None,
            module: &shader,
            entry_point: Some("summarize"),
            compilation_options: Default::default(),
            cache: None,
        });
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MPM compact summary bindings"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: source.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform.as_entire_binding(),
                },
            ],
        });
        Ok(Self {
            output,
            pipeline,
            bindings,
            count,
        })
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.output, 0, None);
        if self.count != 0 {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MPM optional compaction summary"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.bindings, &[]);
            pass.dispatch_workgroups(self.count.div_ceil(64), 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
    }
}
