use super::*;

#[path = "soil_newton_frozen_cycle.rs"]
mod frozen;

#[path = "soil_newton_distributed_nonlinear.rs"]
pub(crate) mod nonlinear;

const WORDS: u64 = 2184 * 2;
const NUMERICAL: [&str; 6] = [
    "cycle_seed",
    "cycle_resident",
    "cycle_initialize",
    "cycle_precondition",
    "cycle_project",
    "cycle_advance",
];

fn source() -> String {
    source_from(base(shader_source_for(NewtonVariant::Fischer)))
}

fn source_from(mut s: String) -> String {
    let begin = s.find("fn krylov_cycle(id:u32,cycle_limit:u32) {").unwrap();
    let body_begin = begin + s[begin..].find('\n').unwrap() + 1;
    let split = s[body_begin..]
        .find("    var checkpoint_estimate=1.0;")
        .unwrap()
        + body_begin;
    let header = s[body_begin..split].to_owned();
    let pre_begin = split + s[split..].find("        for (var slot=lane;").unwrap();
    let pre_end = pre_begin
        + s[pre_begin..]
            .find("        storageBarrier();apply_operator();")
            .unwrap();
    let precondition = s[pre_begin..pre_end].to_owned();
    let orth_begin = pre_end
        + s[pre_end..]
            .find("        for (var orthogonal_pass=")
            .unwrap();
    let orth_end = orth_begin
        + s[orth_begin..]
            .find("\n    }\n    if lane==0u && status.relative>params.tolerance")
            .unwrap();
    let mut orthogonal = s[orth_begin..orth_end].to_owned();
    assert_eq!(orthogonal.matches("break;").count(), 2);
    orthogonal = orthogonal.replace("break;", "cycle_finish(checkpoint_estimate);return;");
    s.push_str(include_str!(
        "../shaders/soil_newton_distributed_cycle.wgsl"
    ));
    s.push_str(&format!(r#"
@compute @workgroup_size(256)
fn cycle_initialize(@builtin(local_invocation_index) id:u32) {{
    lane=id;rate_lane=id;
    if lane==0u {{refinement_norm=condensed[2180u].x;condensed[2182u]=vec2<f32>(bitcast<f32>(1u),0.0);}}
    workgroupBarrier();
{header}
    if lane==0u {{condensed[2181u]=vec2<f32>(bitcast<f32>(0u),bitcast<f32>(params.iterations));}}
    cycle_save(1.0);
    if lane==0u {{condensed[2182u].x=bitcast<f32>(0u);}}
}}
@compute @workgroup_size(256)
fn cycle_precondition(@builtin(local_invocation_index) id:u32) {{
    lane=id;rate_lane=id;if !cycle_running() {{return;}}
    let j=bitcast<u32>(condensed[2181u].x);
{precondition}
}}
@compute @workgroup_size(256)
fn cycle_advance(@builtin(local_invocation_index) id:u32) {{
    lane=id;rate_lane=id;if !cycle_running() {{return;}}
    cycle_load();let j=bitcast<u32>(condensed[2181u].x);
    let cycle_limit=bitcast<u32>(condensed[2181u].y);
    let beta=control.beta;var checkpoint_estimate=condensed[2180u].y;
{orthogonal}
    if j+1u>=cycle_limit {{cycle_finish(checkpoint_estimate);return;}}
    cycle_save(checkpoint_estimate);
    if lane==0u {{condensed[2181u].x=bitcast<f32>(j+1u);}}
}}
"#));
    let product = include_str!("../shaders/soil_newton_distributed.wgsl");
    let start = product.find("fn distributed_product(").unwrap();
    let end = product[start..].find("\n@compute").unwrap() + start;
    let product = product[start..end]
        .replace("fn distributed_product(", "fn cycle_product(")
        .replace("states[c].low", "scratch[c]")
        .replace("states[c].high", "scratch[c]");
    s.push_str(&product);
    s.push_str(
        r#"
@compute @workgroup_size(8,8,1)
fn cycle_forward(@builtin(local_invocation_index) id:u32,@builtin(workgroup_id) group:vec3<u32>) {
    lane=id;rate_lane=id;if !cycle_running() {return;}cycle_product(group.x,false);
}
"#,
    );
    s
}

fn refined_source(cycle_cap: u32) -> String {
    refined_source_from(source(), cycle_cap)
}

fn refined_source_from(mut s: String, cycle_cap: u32) -> String {
    assert!(cycle_cap == 0 || cycle_cap == 1);
    replace(
        &mut s,
        "@compute @workgroup_size(256)\nfn cycle_initialize(@builtin(local_invocation_index) id:u32)",
        "fn cycle_initialize(id:u32)",
    );
    replace(
        &mut s,
        "bitcast<f32>(params.iterations));}\n    cycle_save(1.0);",
        "bitcast<f32>(params.iterations-bitcast<u32>(condensed[2183u].x)));}\n    cycle_save(1.0);",
    );
    let a = s.find("fn cycle_finish(checkpoint:f32) {").unwrap();
    let b = a + s[a..].find("\n@compute").unwrap();
    s.replace_range(
        a..b,
        include_str!("../shaders/soil_newton_distributed_refinement.wgsl"),
    );
    replace(
        &mut s,
        "fn cycle_precondition(@builtin(local_invocation_index) id:u32) {\n    lane=id;rate_lane=id;if !cycle_running() {return;}",
        "fn cycle_precondition(@builtin(local_invocation_index) id:u32) {\n    lane=id;rate_lane=id;\n    if lane==0u {control.stop=bitcast<u32>(condensed[2182u].x);}\n    if workgroupUniformLoad(&control.stop)==2u {cycle_initialize(id);}\n    if !cycle_running() {return;}",
    );
    replace(
        &mut s,
        "workgroupBarrier();krylov_cycle(id,params.iterations);",
        "workgroupBarrier();solve_bounded(id,params.iterations);",
    );
    if cycle_cap == 1 {
        replace(
            &mut s,
            "krylov_cycle(id,limit-used);",
            "krylov_cycle(id,min(1u,limit-used));",
        );
        replace(
            &mut s,
            "bitcast<f32>(params.iterations-bitcast<u32>(condensed[2183u].x))",
            "bitcast<f32>(min(1u,params.iterations-bitcast<u32>(condensed[2183u].x)))",
        );
    }
    s.push_str("\n@compute @workgroup_size(256)\nfn refined_initialize(@builtin(local_invocation_index) id:u32) {cycle_initialize(id);}\n");
    s
}

#[test]
fn newton_distributed_refinement_source_contract() {
    for cap in [0, 1] {
        let s = refined_source(cap);
        assert!(s.contains("status.iterations=used;status.beta=refinement_norm;"));
        assert!(s.contains("used>=params.iterations"));
        for entry in NUMERICAL
            .into_iter()
            .map(|e| {
                if e == "cycle_initialize" {
                    "refined_initialize"
                } else {
                    e
                }
            })
            .chain(["cycle_forward"])
        {
            crate::soil_newton_protected::compile(&marked(s.clone()), entry)
                .unwrap_or_else(|e| panic!("cap={cap} entry={entry}: {e}"));
        }
    }
}

#[test]
fn newton_distributed_cycle_source_contract() {
    let s = source();
    assert!(WORDS * 4 <= 4100 * 8);
    assert_eq!(
        s.matches("cycle_finish(checkpoint_estimate);return;")
            .count(),
        3
    );
    for entry in NUMERICAL.into_iter().chain(["cycle_forward"]) {
        crate::soil_newton_protected::compile(&marked(s.clone()), entry)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
    }
    let plain = plain(s);
    let module = naga::front::wgsl::parse_str(&plain).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
}

struct CycleProbe {
    owner: ExperimentalSoilNewtonGpu,
    pipelines: Vec<wgpu::ComputePipeline>,
    distributed: bool,
    iterations: u32,
    groups: u32,
}

impl CycleProbe {
    fn new(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        cap: SoilNewtonCapacity,
        distributed: bool,
        refined: bool,
        cycle_cap: u32,
    ) -> Result<Self> {
        let s = if refined {
            refined_source(cycle_cap)
        } else {
            source()
        };
        Self::new_source(engine, input, cap, distributed, refined, s)
    }

    fn new_source(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        cap: SoilNewtonCapacity,
        distributed: bool,
        refined: bool,
        s: String,
    ) -> Result<Self> {
        let device = engine.device().ok_or_else(invalid)?;
        let (lanes, storage) = execution_limits(engine);
        if lanes != 256
            || !protected_enabled(engine)
            || device.limits().max_compute_workgroup_storage_size < storage
            || device.limits().max_compute_workgroup_size_y < 8
        {
            return Err(invalid());
        }
        let owner = ExperimentalSoilNewtonGpu::new_with_storage(
            engine,
            input,
            cap,
            NewtonVariant::Fischer,
            |_| marked(s.clone()),
            "cycle_seed",
            true,
        )?;
        let bindings = owner.pipeline.get_bind_group_layout(0);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bindings)],
            immediate_size: 0,
        });
        let entries: &[&str] = if distributed && refined {
            &[
                "refined_initialize",
                "cycle_precondition",
                "cycle_project",
                "cycle_forward",
                "cycle_advance",
                "cycle_publish",
            ]
        } else if distributed {
            &[
                "cycle_initialize",
                "cycle_precondition",
                "cycle_project",
                "cycle_forward",
                "cycle_advance",
                "cycle_publish",
            ]
        } else {
            &["cycle_resident", "cycle_publish"]
        };
        let mut pipelines = vec![owner.pipeline.clone()];
        for &entry in entries {
            let shader = if entry == "cycle_publish" {
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(entry),
                    source: wgpu::ShaderSource::Wgsl(plain(s.clone()).into()),
                })
            } else {
                crate::soil_newton_protected::shader(device, &marked(s.clone()), entry)?
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
            distributed,
            iterations: cap.krylov_iterations,
            groups: cap.contacts.max(1).div_ceil(10),
        })
    }

    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.owner.encode_reset(encoder);
        encoder.clear_buffer(&self.owner.scratch[15], 0, Some(WORDS * 4));
        let dispatch = |encoder: &mut wgpu::CommandEncoder, i: usize, groups: u32| {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&self.pipelines[i]);
            pass.set_bind_group(0, &self.owner.group, &[]);
            pass.dispatch_workgroups(groups, 1, 1);
            molla_core::gpu_metrics::record_dispatch();
        };
        dispatch(encoder, 0, 1);
        dispatch(encoder, 1, 1);
        if self.distributed {
            for _ in 0..self.iterations {
                dispatch(encoder, 2, 1);
                dispatch(encoder, 3, 1);
                dispatch(encoder, 4, self.groups);
                dispatch(encoder, 5, 1);
            }
            dispatch(encoder, 6, 1);
        } else {
            dispatch(encoder, 2, 1);
        }
    }
}

#[test]
#[ignore = "requires protected GPU; persisted Krylov cycle and f64 residual"]
fn newton_distributed_cycle_gpu_edges() {
    cycle_edges(false, 0);
}

#[test]
#[ignore = "requires protected GPU; refined distributed solve"]
fn newton_distributed_refinement_gpu_edges() {
    cycle_edges(true, 0);
}

#[test]
#[ignore = "requires protected GPU; forced restart with unchanged total budget"]
fn newton_distributed_refinement_restart_edges() {
    cycle_edges(true, 1);
}

fn cycle_edges(refined: bool, cycle_cap: u32) {
    use crate::soil_shared_wall_friction_gpu::capture;
    use wgpu::util::DeviceExt;
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let storage = |v: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(v),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    const C: usize = 65;
    let mut saw_multiple = false;
    let mut saw_limit = false;
    let mut saw_restart = false;
    for n in [1usize, 3, 32] {
        let mut rows = vec![0_f32; C * n * 4];
        let mut response = rows.clone();
        for c in 0..C {
            for i in 0..n {
                for a in 0..3 {
                    rows[4 * (c * n + i) + a] =
                        (((c * 3 + i * 5 + a) % 17) as i32 - 8) as f32 / 128.;
                    response[4 * (c * n + i) + a] =
                        (((c * 7 + i * 3 + a) % 13) as i32 - 6) as f32 / 256.;
                }
            }
        }
        let rb = storage(&rows);
        let pb = storage(&response);
        let mass = storage(&vec![0.; n * n]);
        let free = storage(&vec![0.; 2 * n]);
        let contacts = storage(&vec![0.; C * 16]);
        let upstream = storage(&[0.]);
        for limit in [1u32, 8, 64] {
            let probes = [false, true].map(|distributed| {
                CycleProbe::new(
                    &engine,
                    SoilNewtonInputs {
                        mass: &mass,
                        free_rates: &free,
                        point_rows: &rb,
                        point_response: &pb,
                        contacts: &contacts,
                        upstream_status: &upstream,
                    },
                    SoilNewtonCapacity {
                        dofs: n as u32,
                        contacts: C as u32,
                        newton_iterations: 64,
                        krylov_iterations: limit,
                        max_owned_bytes: u64::MAX,
                    },
                    distributed,
                    refined,
                    cycle_cap,
                )
                .unwrap()
            });
            for count in [0usize, 1, 9, 10, 11, C] {
                queue.write_buffer(&contacts, 48, &(count as u32).to_le_bytes());
                let active: Vec<_> = (0..count).map(|i| (2 * i + 7) % C).collect();
                let mut expected = None;
                let mut restarts = 0;
                for (variant, probe) in probes.iter().enumerate() {
                    for repeat in 0..2 {
                        let mut encoder = device.create_command_encoder(&Default::default());
                        #[cfg(feature = "resource-counters")]
                        let scopes = (
                            molla_core::gpu_metrics::DeviceScope::start().unwrap(),
                            molla_core::gpu_metrics::TransferScope::start().unwrap(),
                            molla_core::gpu_metrics::DispatchScope::start().unwrap(),
                        );
                        probe.encode(&mut encoder);
                        queue.submit([encoder.finish()]);
                        #[cfg(feature = "resource-counters")]
                        {
                            let a = scopes.0.finish();
                            let t = scopes.1.finish();
                            let d = scopes.2.finish();
                            assert_eq!(a.observations, 0);
                            assert!(!a.invalid && !t.overflowed && !d.overflowed);
                            assert_eq!((t.upload_bytes, t.readback_bytes), (0, 0));
                            assert_eq!(
                                d.encoded_dispatches,
                                if variant == 0 {
                                    3
                                } else {
                                    3 + 4 * u64::from(limit)
                                }
                            );
                        }
                        let status = capture::read(&engine, probe.owner.status_buffer());
                        let state = capture::read(&engine, probe.owner.state_buffer());
                        if refined && variant == 1 {
                            let workspace = capture::read(&engine, &probe.owner.scratch[15]);
                            let used = u32::from_le_bytes(
                                workspace[2183 * 8..2183 * 8 + 4].try_into().unwrap(),
                            );
                            let count = u32::from_le_bytes(
                                workspace[2183 * 8 + 4..2184 * 8].try_into().unwrap(),
                            );
                            assert!(used <= limit);
                            restarts = restarts.max(count);
                            saw_restart |= count > 0;
                        }
                        if let Some((old_status, old_state)) = &expected {
                            assert!(
                                status == *old_status && state == *old_state,
                                "n={n} limit={limit} active={count} variant={variant} repeat={repeat}"
                            );
                        } else {
                            expected = Some((status.clone(), state.clone()));
                        }
                        let words: Vec<_> = status
                            .chunks_exact(4)
                            .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
                            .collect();
                        assert!(words[0] == 0 || words[0] == 8);
                        assert!(words[2] <= limit);
                        saw_multiple |= words[2] > 1;
                        saw_limit |= words[0] == 8;
                        let values: Vec<_> = state
                            .chunks_exact(4)
                            .map(|x| f32::from_le_bytes(x.try_into().unwrap()) as f64)
                            .collect();
                        assert!(values.iter().all(|x| x.is_finite()));
                        let component = |c: usize, i: usize| {
                            let offset = 16 * c + if i < 3 { i } else { i + 1 };
                            values[offset] + values[offset + 8]
                        };
                        let rates: Vec<f64> = (0..n)
                            .map(|i| {
                                active
                                    .iter()
                                    .map(|&c| {
                                        (0..3)
                                            .map(|a| {
                                                -(response[4 * (c * n + i) + a] as f64)
                                                    * component(c, a)
                                            })
                                            .sum::<f64>()
                                    })
                                    .sum()
                            })
                            .collect();
                        let mut residual = 0.;
                        let mut rhs_norm = 0.;
                        for &c in &active {
                            let point: [f64; 3] = std::array::from_fn(|a| {
                                (0..n)
                                    .map(|i| rows[4 * (c * n + i) + a] as f64 * rates[i])
                                    .sum()
                            });
                            for i in 0..6 {
                                let rhs =
                                    (0.1_f32 * (i + 1) as f32 + 0.01_f32 * (c % 7) as f32) as f64;
                                let product = (2. + 0.125 * (c % 5) as f64) * component(c, i)
                                    + (0..3)
                                        .map(|j| {
                                            ((((c + i + j * 7) % 13) as i32 - 6) as f64 / 64.
                                                + (((i + j) % 3) as i32 - 1) as f64 / 134217728.)
                                                * point[j]
                                        })
                                        .sum::<f64>();
                                residual += (rhs + product).powi(2);
                                rhs_norm += rhs * rhs;
                            }
                        }
                        let relative = if rhs_norm == 0. {
                            0.
                        } else {
                            (residual / rhs_norm).sqrt()
                        };
                        assert!(relative.is_finite());
                        if words[0] == 0 {
                            assert!(relative <= 1e-5, "f64 residual {relative}");
                        }
                    }
                }
                let status = &expected.unwrap().0;
                let fault = u32::from_le_bytes(status[0..4].try_into().unwrap());
                let iterations = u32::from_le_bytes(status[8..12].try_into().unwrap());
                println!(
                    "distributed-cycle n={n} limit={limit} active={count} iterations={iterations} fault={fault}: exact outputs/repeats and independent residual pass"
                );
                if refined {
                    println!(
                        "distributed-refinement n={n} limit={limit} active={count} cycle_cap={cycle_cap} iterations={iterations} restarts={restarts} fault={fault}"
                    );
                }
            }
        }
    }
    assert!(
        saw_multiple && saw_limit,
        "exercise persisted iterations and capped failure"
    );
    if cycle_cap == 1 {
        assert!(saw_restart, "force the refinement/restart path");
    }
}
