use molla_compute::Engine;
use molla_sim::{Model, State};

pub(crate) struct ContactReactionInputs<'a> {
    pub rows: &'a wgpu::Buffer,
    pub heads: &'a wgpu::Buffer,
    pub links: &'a wgpu::Buffer,
    pub world_impulses: &'a wgpu::Buffer,
    pub response_delta: &'a wgpu::Buffer,
}

pub(crate) struct ContactReactionGpu {
    pipeline: wgpu::ComputePipeline,
    groups: [wgpu::BindGroup; 2],
    pub loads: wgpu::Buffer,
    pub delta: wgpu::Buffer,
    work: u32,
}

impl ContactReactionGpu {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engine: &Engine,
        model: &Model,
        states: &[State; 2],
        response: ContactReactionInputs<'_>,
        external: &wgpu::Buffer,
        dt: &wgpu::Buffer,
        predictor_delta: &wgpu::Buffer,
        status: &wgpu::Buffer,
        rows: u32,
    ) -> Self {
        let device = engine.device().unwrap();
        let allocate = |label, size| {
            molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }
            )
        };
        let empty = allocate("contact reaction empty", 32);
        let loads = allocate(
            "contact reaction COM loads",
            model.body_count.max(1) as u64 * 32,
        );
        let delta = allocate(
            "contact reaction velocity increment",
            model.joint_dof_total.max(1) as u64 * 4,
        );
        let params = allocate("contact reaction dimensions", 16);
        molla_core::gpu_write_buffer!(
            engine.queue().unwrap(),
            &params,
            0,
            bytemuck::cast_slice(&[
                model.body_count as u32,
                model.joint_dof_total as u32,
                rows,
                0,
            ]),
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("contact reaction assembly"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/contact_reactions.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("contact reaction assembly"),
            layout: None,
            module: &shader,
            entry_point: Some("assemble"),
            compilation_options: Default::default(),
            cache: None,
        });
        let groups = states.each_ref().map(|state| {
            let buffers = [
                response.rows,
                response.heads,
                response.links,
                response.world_impulses,
                state.body_q.device_buffer().unwrap_or(&empty),
                model.body_com.device_buffer().unwrap_or(&empty),
                external,
                predictor_delta,
                response.response_delta,
                &loads,
                &delta,
                status,
                dt,
                &params,
            ];
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("contact reaction inputs"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &buffers
                    .into_iter()
                    .enumerate()
                    .map(|(binding, buffer)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: buffer.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        });
        Self {
            pipeline,
            groups,
            loads,
            delta,
            work: model
                .body_count
                .max(model.joint_dof_total)
                .max(1)
                .div_ceil(64) as u32,
        }
    }

    pub fn encode(&self, encoder: &mut wgpu::CommandEncoder, side: usize) {
        encoder.clear_buffer(&self.loads, 0, None);
        encoder.clear_buffer(&self.delta, 0, None);
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.groups[side], &[]);
        pass.dispatch_workgroups(self.work, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}
