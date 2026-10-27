use super::*;

#[path = "soil_newton_distributed_cycle.rs"]
mod cycle;
pub(crate) use cycle::nonlinear::NonlinearProbe;

#[path = "soil_newton_linear_capture.rs"]
mod linear_capture;
pub(crate) use linear_capture::capture_linearization;

#[path = "soil_newton_mixed.rs"]
pub(crate) mod mixed;

const ENTRIES: [&str; 6] = [
    "row_probe",
    "distributed_project_forward",
    "distributed_forward",
    "distributed_project_direction",
    "distributed_direction",
    "distributed_finish",
];

fn source(input: String) -> String {
    let mut source = crate::soil_newton_parallel_release::row_products::probe_source(input, false);
    let a = source
        .rfind("    storageBarrier();apply_operator();")
        .unwrap();
    source.truncate(a);
    source.push_str("}\n");
    let old = "states[c]=State(Six(),Six());";
    assert_eq!(source.matches(old).count(), 1);
    source = source.replace(old, "states[c]=State(scratch[c],scratch[c]);");
    source.push_str(include_str!("../shaders/soil_newton_distributed.wgsl"));
    source
}

fn base(input: String) -> String {
    source(ExperimentalSoilNewtonGpu::cached_contact_source(
        ExperimentalSoilNewtonGpu::cached_delta_source(
            ExperimentalSoilNewtonGpu::stalled_release_source(input),
        ),
    ))
}

fn marked(input: String) -> String {
    format!(
        "{}{}{}256\n{input}",
        crate::soil_newton_protected::PREFIX,
        crate::soil_newton_protected::VOLATILE_MARKER,
        crate::soil_newton_width::MARKER
    )
}

fn plain(input: String) -> String {
    crate::soil_newton_width::specialize(format!(
        "{}256\n{input}",
        crate::soil_newton_width::MARKER
    ))
    .unwrap()
}

pub(crate) struct Probe {
    owner: ExperimentalSoilNewtonGpu,
    pipelines: Vec<wgpu::ComputePipeline>,
    groups: u32,
}

impl Probe {
    pub(crate) fn new(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        capacity: SoilNewtonCapacity,
    ) -> Result<Self> {
        let (lanes, storage) = execution_limits(engine);
        let device = engine.device().ok_or_else(invalid)?;
        let limits = device.limits();
        if lanes != 256
            || !protected_enabled(engine)
            || limits.max_compute_workgroup_storage_size < storage
            || limits.max_compute_workgroup_size_x < 256
            || limits.max_compute_workgroup_size_y < 8
            || limits.max_compute_invocations_per_workgroup < 256
            || capacity.contacts.max(1).div_ceil(10) > limits.max_compute_workgroups_per_dimension
        {
            return Err(invalid());
        }
        let owner = ExperimentalSoilNewtonGpu::new_with_storage(
            engine,
            input,
            capacity,
            NewtonVariant::Fischer,
            |s| plain(base(s)),
            ENTRIES[0],
            true,
        )?;
        let bindings = owner.pipeline.get_bind_group_layout(0);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bindings)],
            immediate_size: 0,
        });
        let source = marked(base(shader_source_for(NewtonVariant::Fischer)));
        let mut pipelines = vec![owner.pipeline.clone()];
        for entry in &ENTRIES[1..] {
            let shader = if *entry == "distributed_finish" {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(entry),
                    source: wgpu::ShaderSource::Wgsl(
                        plain(base(shader_source_for(NewtonVariant::Fischer))).into(),
                    ),
                })
            } else {
                crate::soil_newton_protected::shader(device, &source, entry)?
            };
            pipelines.push(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&layout),
                    module: &shader,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                }),
            );
        }
        Ok(Self {
            owner,
            pipelines,
            groups: capacity.contacts.max(1).div_ceil(10),
        })
    }

    pub(crate) fn encode_reset(&self, encoder: &mut wgpu::CommandEncoder) {
        self.owner.encode_reset(encoder);
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        for (i, pipeline) in self.pipelines.iter().enumerate() {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.owner.group, &[]);
            pass.dispatch_workgroups(if i == 2 || i == 4 { self.groups } else { 1 }, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        }
    }

    pub(crate) fn status_buffer(&self) -> &wgpu::Buffer {
        self.owner.status_buffer()
    }
    pub(crate) fn state_buffer(&self) -> &wgpu::Buffer {
        self.owner.state_buffer()
    }
}

#[test]
fn newton_distributed_source_contract() {
    let source = marked(base(shader_source_for(NewtonVariant::Fischer)));
    assert!(4132 * 8 <= 4160 * 8);
    assert_eq!(source.matches("@compute @workgroup_size(8,8,1)").count(), 2);
    assert!(!source.contains("row_product_points"));
    for entry in &ENTRIES[1..5] {
        crate::soil_newton_protected::compile(&source, entry)
            .unwrap_or_else(|e| panic!("entry {entry}: {e}"));
    }
    let plain = plain(base(shader_source_for(NewtonVariant::Fischer)));
    let module = naga::front::wgsl::parse_str(&plain).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}
