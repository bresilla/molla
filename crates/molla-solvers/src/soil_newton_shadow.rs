use crate::soil_newton_gpu::{
    ExperimentalSoilNewtonGpu as Solver, SoilNewtonCapacity, SoilNewtonInputs,
};
use molla_compute::Engine;
use molla_core::Result;

pub(crate) struct Comparison {
    pipeline: wgpu::ComputePipeline,
    group: wgpu::BindGroup,
    pub(crate) summary: wgpu::Buffer,
}

impl Comparison {
    pub(crate) fn new(
        engine: &Engine,
        control: [&wgpu::Buffer; 5],
        candidate: [&wgpu::Buffer; 5],
    ) -> Self {
        for (a, b) in control.iter().zip(candidate.iter()) {
            assert_eq!(a.size(), b.size());
            assert!(a.size() > 0 && a.size() % 4 == 0 && a.size() / 4 < (1 << 28));
        }
        let device = engine.device().unwrap();
        let summary = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("Newton equal-input comparison"),
                size: 64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Newton equal-input comparison"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../shaders/soil_newton_shadow.wgsl").into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Newton equal-input comparison"),
            layout: None,
            module: &shader,
            entry_point: Some("compare"),
            compilation_options: Default::default(),
            cache: None,
        });
        let entries: Vec<_> = control
            .into_iter()
            .chain(candidate)
            .chain([&summary])
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect();
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Newton equal-input comparison"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        Self {
            pipeline,
            group,
            summary,
        }
    }
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &self.group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        molla_core::gpu_metrics::record_dispatch();
    }
}

pub(crate) struct Shadow {
    control: Solver,
    comparison: Comparison,
}

fn buffers(s: &Solver) -> [&wgpu::Buffer; 5] {
    [
        s.outputs_buffer(),
        s.rates_buffer(),
        s.publication_status_buffer(),
        s.state_buffer(),
        s.status_buffer(),
    ]
}

impl Shadow {
    pub(crate) fn new(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
        candidate: &Solver,
    ) -> Result<Self> {
        assert!(!cfg!(feature = "warm-newton"));
        if cfg!(feature = "retained-local-newton") {
            return Err(molla_core::Error::Solver(
                "exact publication shadow does not support retained-local-newton".into(),
            ));
        }
        use crate::soil_newton_failure::NewtonFailureMethod as Method;
        let methods = if cfg!(feature = "publication-products-newton") {
            (
                Method::FischerConditionedProtectedActivePublication256,
                Method::FischerConditionedProtectedPublicationProducts256,
            )
        } else {
            (
                Method::FischerConditionedProtectedNorm256,
                Method::FischerConditionedProtectedActivePublication256,
            )
        };
        assert_eq!(candidate.failure_method(), methods.1);
        let control = Solver::new_publication_shadow_control(engine, input, capacity)?;
        assert_eq!(control.failure_method(), methods.0);
        let comparison = Comparison::new(engine, buffers(&control), buffers(candidate));
        Ok(Self {
            control,
            comparison,
        })
    }
    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.control.encode_reset(encoder);
        self.control.encode(encoder);
        self.comparison.encode(encoder);
    }
    pub(crate) fn summary(&self) -> &wgpu::Buffer {
        &self.comparison.summary
    }
}

#[path = "soil_newton_shadow_tests.rs"]
mod tests;
