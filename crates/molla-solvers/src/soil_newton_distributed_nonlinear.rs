use super::*;

const OUTER_OFFSET: u64 = 4132 * 8;
const OUTER_BYTES: u64 = 9 * 8;

fn source() -> String {
    let original =
        super::super::linear_capture::method20_source(shader_source_for(NewtonVariant::Fischer));
    let mut s = crate::soil_newton_parallel_release::sliced::source(original);
    for i in 0..8 {
        s = s.replace(
            &format!("condensed[{}u", 4100 + i),
            &format!("condensed[{}u", 4132 + i),
        );
    }
    replace(
        &mut s,
        "var alpha=1.0;var backtrack=0u;var damping=0.0;var attempt=0u;",
        "var alpha=1.0;var backtrack=0u;var damping=0.0;var attempt=0u;\n    if slice_mode==3u {phase=5u;merit=condensed[4140u].x;state_norm=condensed[4140u].y;}",
    );
    replace(
        &mut s,
        "direct_linearize=!DIRECT_LAZY_JACOBIAN || (phase!=1u && phase!=3u);let candidate=evaluate();",
        "direct_linearize=!DIRECT_LAZY_JACOBIAN || (phase!=1u && phase!=3u);var candidate=0.0;if phase!=5u {candidate=evaluate();}",
    );
    replace(
        &mut s,
        "state_norm=scratch_norm();prepare_linear();hybrid_direction(lane);",
        "state_norm=scratch_norm();prepare_linear();save_outer_slice(step);\n            if lane==0u {condensed[4132u].x=bitcast<f32>(3u);condensed[4140u]=vec2<f32>(merit,state_norm);}return;\n        }\n        if phase==5u {hybrid_tail(lane);",
    );
    let hybrid = include_str!("../shaders/soil_newton_hybrid.wgsl")
        .replace("fn hybrid_direction(", "fn hybrid_tail(")
        .replace("    solve_bounded(id,min(8u,params.iterations));\n", "");
    s.push_str(&hybrid);
    s = refined_source_from(source_from(s), 0);
    replace(
        &mut s,
        "params.iterations-bitcast<u32>(condensed[2183u].x)",
        "min(8u,params.iterations)-bitcast<u32>(condensed[2183u].x)",
    );
    replace(
        &mut s,
        "used>=params.iterations || count==0u",
        "used>=min(8u,params.iterations) || count==0u",
    );
    let begin = s.find("fn solve_bounded(id:u32,limit:u32) {").unwrap();
    let end = begin + s[begin..].find("    var used=0u;").unwrap();
    let seed = s[begin..end].replace(
        "fn solve_bounded(id:u32,limit:u32) {",
        r#"
@compute @workgroup_size(256)
fn prefix_seed(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {condensed[2182u].x=bitcast<f32>(1u);condensed[2183u]=vec2<f32>(0.0);}
    if bitcast<u32>(condensed[4132u].x)!=3u {return;}
"#,
    );
    s.push_str(&seed);
    s.push_str(
        r#"
    if lane==0u {condensed[2180u]=vec2<f32>(initial,0.0);}
    storageBarrier();cycle_initialize(id);
}
@compute @workgroup_size(256)
fn prefix_resident(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if bitcast<u32>(condensed[4132u].x)!=3u {return;}
    solve_bounded(id,min(8u,params.iterations));
}
"#,
    );
    s
}

fn indirect_source() -> String {
    let mut s = source();
    for name in ["prefix_seed", "cycle_advance"] {
        replace(
            &mut s,
            &format!(
                "@compute @workgroup_size(256)\nfn {name}(@builtin(local_invocation_index) id:u32)"
            ),
            &format!("fn {name}(id:u32)"),
        );
        s.push_str(&format!("\n@compute @workgroup_size(256)\nfn {name}_indirect(@builtin(local_invocation_index) id:u32) {{{name}(id);write_dispatch_args();}}\n"));
    }
    s.push_str(r#"
@group(1) @binding(0) var<storage,read_write> dispatch_args:array<u32,6>;
fn write_dispatch_args() {
    if lane==0u {
        let stage=bitcast<u32>(condensed[2182u].x);
        let running=bitcast<u32>(condensed[4132u].x)==3u && (stage==0u || stage==2u);
        dispatch_args[0]=select(0u,1u,running);dispatch_args[1]=1u;dispatch_args[2]=1u;
        dispatch_args[3]=select(0u,(status.active_count+9u)/10u,running);dispatch_args[4]=1u;dispatch_args[5]=1u;
    }
}

"#);
    s
}

fn hybrid_source() -> String {
    let mut s = source();
    replace(&mut s, "if phase==5u {hybrid_tail(lane);", "if phase==5u {");
    replace(
        &mut s,
        "    solve_bounded(id,min(8u,params.iterations));\n}\n",
        "    fischer_smoothing=condensed[4134u].x;\n    smoothing_stalled=bitcast<u32>(condensed[4134u].y);\n    conditioned_model=bitcast<u32>(condensed[4135u].x)!=0u;\n    hybrid_direction(id);\n}\n",
    );
    s
}

#[test]
fn newton_hybrid_boundary_source_contract() {
    let original = source();
    let candidate = hybrid_source();
    assert_eq!(
        original.matches("var<storage").count(),
        candidate.matches("var<storage").count()
    );
    assert_eq!(
        original.matches("var<workgroup>").count(),
        candidate.matches("var<workgroup>").count()
    );
    let function = |s: &str, name: &str| {
        let start = s.find(&format!("fn {name}(")).unwrap();
        let end = start + s[start..].find("\n}").unwrap() + 2;
        s[start..end].to_owned()
    };
    for name in [
        "hybrid_direction",
        "solve_bounded",
        "direct_direction",
        "prepare_linear",
        "certify_candidate",
    ] {
        assert_eq!(function(&original, name), function(&candidate, name));
    }
    assert!(!function(&candidate, "newton").contains("hybrid_tail(lane)"));
    for entry in ["newton", "prefix_resident"] {
        let binary =
            crate::soil_newton_protected::compile(&marked(candidate.clone()), entry).unwrap();
        if let Some(root) = std::env::var_os("MOLLA_NEWTON_HYBRID_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join("candidate.wgsl"), &candidate).unwrap();
            std::fs::write(
                root.join(format!("{entry}.spv")),
                binary
                    .iter()
                    .flat_map(|w| w.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        }
    }
}

#[test]
fn newton_distributed_indirect_source_contract() {
    let s = indirect_source();
    for entry in ["prefix_seed_indirect", "cycle_advance_indirect"] {
        crate::soil_newton_protected::compile(&marked(s.clone()), entry)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
    }
    assert!(s.contains("stage==0u || stage==2u"));
    assert!(s.contains("(status.active_count+9u)/10u"));
}

fn fused_source() -> String {
    let mut s = indirect_source();
    for name in ["cycle_precondition", "cycle_project"] {
        replace(
            &mut s,
            &format!(
                "@compute @workgroup_size(256)\nfn {name}(@builtin(local_invocation_index) id:u32)"
            ),
            &format!("fn {name}(id:u32)"),
        );
    }
    for name in ["prefix_seed", "cycle_advance"] {
        replace(
            &mut s,
            &format!("fn {name}_indirect("),
            &format!("fn {name}_fused("),
        );
        replace(
            &mut s,
            &format!("{{{name}(id);write_dispatch_args();}}"),
            &format!(
                "{{{name}(id);storageBarrier();cycle_precondition(id);storageBarrier();cycle_project(id);storageBarrier();write_dispatch_args();}}"
            ),
        );
    }
    s
}

#[test]
fn newton_distributed_fused_source_contract() {
    let s = fused_source();
    for entry in ["prefix_seed_fused", "cycle_advance_fused"] {
        crate::soil_newton_protected::compile(&marked(s.clone()), entry)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
    }
    assert_eq!(s.matches("storageBarrier();cycle_precondition(id);storageBarrier();cycle_project(id);storageBarrier();write_dispatch_args();").count(),2);
}

#[test]
fn newton_distributed_nonlinear_source_contract() {
    let s = source();
    assert!(OUTER_OFFSET >= (4100 + 32) * 8 && OUTER_OFFSET + OUTER_BYTES <= 4160 * 8);
    assert!(s.contains("used>=min(8u,params.iterations) || count==0u"));
    for entry in [
        "newton",
        "prefix_seed",
        "prefix_resident",
        "cycle_precondition",
        "cycle_project",
        "cycle_forward",
        "cycle_advance",
    ] {
        crate::soil_newton_protected::compile(&marked(s.clone()), entry)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
    }
}

pub(crate) struct NonlinearProbe {
    pub(crate) owner: ExperimentalSoilNewtonGpu,
    pipelines: Vec<wgpu::ComputePipeline>,
    mode: u32,
    outer: u32,
    prefix: u32,
    groups: u32,
    indirect: Option<(wgpu::Buffer, wgpu::BindGroup)>,
}

impl NonlinearProbe {
    pub(crate) fn new(
        engine: &Engine,
        input: SoilNewtonInputs<'_>,
        cap: SoilNewtonCapacity,
        mode: u32,
    ) -> Result<Self> {
        let device = engine.device().ok_or_else(invalid)?;
        let (lanes, storage) = execution_limits(engine);
        if mode > 5
            || lanes != 256
            || !protected_enabled(engine)
            || device.limits().max_compute_workgroup_storage_size < storage
            || device.limits().max_compute_workgroup_size_y < 8
            || (matches!(mode, 3 | 4)
                && (device.limits().max_storage_buffers_per_shader_stage < 22
                    || device.limits().max_bind_groups < 2))
        {
            return Err(invalid());
        }
        let s = if mode == 0 {
            super::super::linear_capture::method20_source(shader_source_for(NewtonVariant::Fischer))
        } else if mode == 5 {
            hybrid_source()
        } else {
            source()
        };
        let mut owner_capacity = cap;
        if matches!(mode, 3 | 4) {
            owner_capacity.max_owned_bytes =
                cap.max_owned_bytes.checked_sub(24).ok_or_else(invalid)?;
        }
        let owner = ExperimentalSoilNewtonGpu::new_with_storage(
            engine,
            input,
            owner_capacity,
            NewtonVariant::Fischer,
            |_| marked(s.clone()),
            "newton",
            true,
        )?;
        let mut pipelines = vec![owner.pipeline.clone()];
        let bindings = owner.pipeline.get_bind_group_layout(0);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bindings)],
            immediate_size: 0,
        });
        let indirect_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Newton indirect output"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let writer_layout = matches!(mode, 3 | 4).then(|| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Newton indirect writer"),
                bind_group_layouts: &[Some(&bindings), Some(&indirect_layout)],
                immediate_size: 0,
            })
        });
        let indirect = matches!(mode, 3 | 4).then(|| {
            let buffer = molla_core::gpu_create_buffer!(
                device,
                &wgpu::BufferDescriptor {
                    label: Some("Newton indirect arguments"),
                    size: 24,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::INDIRECT
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }
            );
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &indirect_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            (buffer, group)
        });
        let entries: &[&str] = match mode {
            0 => &[],
            1 | 5 => &["prefix_resident"],
            4 => &["prefix_seed_fused", "cycle_forward", "cycle_advance_fused"],
            3 => &[
                "prefix_seed_indirect",
                "cycle_precondition",
                "cycle_project",
                "cycle_forward",
                "cycle_advance_indirect",
            ],
            _ => &[
                "prefix_seed",
                "cycle_precondition",
                "cycle_project",
                "cycle_forward",
                "cycle_advance",
            ],
        };
        let indirect_source = matches!(mode, 3 | 4).then(|| {
            if mode == 4 {
                fused_source()
            } else {
                indirect_source()
            }
        });
        for &entry in entries {
            let writer = entry.ends_with("_indirect") || entry.ends_with("_fused");
            let shader = crate::soil_newton_protected::shader(
                device,
                &marked(if writer {
                    indirect_source.as_ref().unwrap().clone()
                } else {
                    s.clone()
                }),
                entry,
            )?;
            pipelines.push(
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(if writer {
                        writer_layout.as_ref().unwrap()
                    } else {
                        &layout
                    }),
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
            mode,
            outer: cap.newton_iterations,
            prefix: cap.krylov_iterations.min(8),
            groups: cap.contacts.max(1).div_ceil(10),
            indirect,
        })
    }

    pub(crate) fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        self.encode_inner(encoder, None);
    }

    pub(crate) fn phase_query_count(&self) -> u32 {
        assert!(matches!(self.mode, 0 | 1 | 5));
        if self.mode == 0 {
            2
        } else {
            6 * (self.outer + 1)
        }
    }

    pub(crate) fn encode_profiled(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        queries: &wgpu::QuerySet,
    ) {
        let _ = self.phase_query_count();
        self.encode_inner(encoder, Some(queries));
    }

    fn encode_inner(&self, encoder: &mut wgpu::CommandEncoder, queries: Option<&wgpu::QuerySet>) {
        self.owner.encode_reset(encoder);
        encoder.clear_buffer(self.owner.publication_status_buffer(), 0, None);
        if self.mode != 0 {
            encoder.clear_buffer(&self.owner.scratch[15], OUTER_OFFSET, Some(OUTER_BYTES));
        }
        let mut query = 0;
        let mut dispatch = |encoder: &mut wgpu::CommandEncoder, i: usize, groups: u32| {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Newton scheduled phase"),
                timestamp_writes: queries.map(|query_set| {
                    let beginning = query;
                    query += 2;
                    wgpu::ComputePassTimestampWrites {
                        query_set,
                        beginning_of_pass_write_index: Some(beginning),
                        end_of_pass_write_index: Some(beginning + 1),
                    }
                }),
            });
            pass.set_pipeline(&self.pipelines[i]);
            pass.set_bind_group(0, &self.owner.group, &[]);
            if let Some((args, group)) = &self.indirect {
                if i == 1 || i == if self.mode == 4 { 3 } else { 5 } {
                    pass.set_bind_group(1, group, &[]);
                }
                if (self.mode == 4 && i == 2) || (self.mode == 3 && (2..=4).contains(&i)) {
                    pass.dispatch_workgroups_indirect(
                        args,
                        if self.mode == 4 || i == 4 { 12 } else { 0 },
                    );
                } else {
                    pass.dispatch_workgroups(groups, 1, 1);
                }
            } else {
                pass.dispatch_workgroups(groups, 1, 1);
            }
            molla_core::gpu_metrics::record_dispatch();
        };
        if self.mode == 0 {
            dispatch(encoder, 0, 1);
            return;
        }
        for _ in 0..=self.outer {
            dispatch(encoder, 0, 1);
            dispatch(encoder, 1, 1);
            if matches!(self.mode, 2..=4) {
                for _ in 0..self.prefix {
                    dispatch(encoder, 2, 1);
                    dispatch(encoder, 3, 1);
                    if self.mode != 4 {
                        dispatch(encoder, 4, self.groups);
                        dispatch(encoder, 5, 1);
                    }
                }
            }
            dispatch(encoder, 0, 1);
        }
    }

    pub(crate) fn indirect_buffer(&self) -> Option<&wgpu::Buffer> {
        self.indirect.as_ref().map(|(buffer, _)| buffer)
    }

    pub(crate) fn mode(&self) -> u32 {
        self.mode
    }
}
