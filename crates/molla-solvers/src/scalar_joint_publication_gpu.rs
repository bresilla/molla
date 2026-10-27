use molla_compute::Engine;
use molla_core::{Error, Result};

pub(crate) struct ScalarJointPublication {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    forces: wgpu::Buffer,
    rows: u32,
}

impl ScalarJointPublication {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        engine: &Engine,
        rows: &wgpu::Buffer,
        directions: &wgpu::Buffer,
        solutions: &wgpu::Buffer,
        samples: &wgpu::Buffer,
        forces: &wgpu::Buffer,
        params: &wgpu::Buffer,
        status: &wgpu::Buffer,
        count: u32,
    ) -> Result<Self> {
        let device = engine.device().unwrap();
        if solutions.size() < u64::from(count) * 32
            || !solutions.usage().contains(wgpu::BufferUsages::STORAGE)
        {
            return Err(Error::Gpu("invalid coupled scalar solution buffer".into()));
        }
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("coupled scalar joint publication"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/scalar_joint_publication.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("coupled scalar joint publication"),
            layout: None,
            module: &shader,
            entry_point: Some("publish"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("coupled scalar joint publication"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[rows, directions, solutions, samples, forces, params, status]
                .iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        });
        Ok(Self {
            pipeline,
            group,
            forces: forces.clone(),
            rows: count,
        })
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.forces, 0, None);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(self.rows.div_ceil(64).max(1), 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}
