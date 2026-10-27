use super::*;

use super::super::soa_scratch;

#[path = "soil_newton_basis_tiles.rs"]
mod basis_tiles;
#[path = "soil_newton_block_initialization.rs"]
mod block_initialization;
#[path = "soil_newton_vector_soa.rs"]
mod vector_soa;

#[path = "soil_newton_adjugate.rs"]
pub(crate) mod adjugate;

#[path = "soil_newton_accepted_rates.rs"]
mod accepted_rates;

#[path = "soil_newton_complementarity.rs"]
pub(crate) mod complementarity;

#[path = "soil_newton_private_samples.rs"]
pub(crate) mod private_samples;

#[path = "soil_newton_certification_cost.rs"]
pub(crate) mod certification_cost;

pub(crate) use super::super::active_indices;

#[path = "soil_newton_grouped_norm.rs"]
pub(crate) mod grouped_norm;

#[path = "soil_newton_pivot_suffix.rs"]
pub(crate) mod pivot_suffix;

pub(crate) use super::super::grouped_sum;

#[path = "soil_newton_common_handoff.rs"]
mod common_handoff;

#[path = "soil_newton_exact_inverse_cache.rs"]
mod exact_inverse_cache;
#[path = "soil_newton_fast_continuation.rs"]
mod fast_continuation;
#[path = "soil_newton_lagged_preconditioner.rs"]
mod lagged_preconditioner;
#[path = "soil_newton_mobility_cache.rs"]
mod mobility_cache;
#[path = "soil_newton_residual_value.rs"]
mod residual_value;
#[path = "soil_newton_slot_constants.rs"]
mod slot_constants;
pub(crate) use super::super::stream_block;

pub(crate) use super::super::conditioned_projection_cache as conditioned_cache;
#[path = "soil_newton_fixed_width.rs"]
pub(crate) mod fixed_width;
pub(crate) use super::super::shared_conditioned_core as shared_core;
#[path = "soil_newton_unit_division_tests.rs"]
mod unit_division_tests;

pub(crate) fn new(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    cap: SoilNewtonCapacity,
    candidate: bool,
) -> Result<ExperimentalSoilNewtonGpu> {
    if kind() == "direct-quality-trace" {
        #[cfg(feature = "retained-local-newton")]
        return crate::soil_newton_gpu::direct_quality::new(engine, input, cap, candidate);
        #[cfg(not(feature = "retained-local-newton"))]
        return Err(invalid());
    }
    if matches!(
        kind().as_str(),
        "retained-direct" | "retained-direct-lazy" | "retained-direct-quality"
    ) {
        #[cfg(feature = "retained-local-newton")]
        return crate::soil_newton_gpu::retained_direct::new(
            engine,
            input,
            cap,
            candidate,
            kind() == "retained-direct-lazy",
        );
        #[cfg(not(feature = "retained-local-newton"))]
        return Err(invalid());
    }
    let (lanes, storage) = execution_limits(engine);
    if kind() == "native-division"
        && !engine
            .device()
            .ok_or_else(invalid)?
            .features()
            .contains(wgpu::Features::SHADER_F64)
    {
        return Err(invalid());
    }
    if lanes != 256
        || !protected_enabled(engine)
        || engine
            .device()
            .ok_or_else(invalid)?
            .limits()
            .max_compute_workgroup_storage_size
            < storage
    {
        return Err(invalid());
    }
    if kind() == "common-handoff" {
        assert!(diagnostic_inner_limit()?.is_none());
    }
    if matches!(
        kind().as_str(),
        "adjugate-shared"
            | "pivot-suffix-shared"
            | "grouped-sum-shared"
            | "grouped-eight-shared"
            | "accepted-rates-shared"
            | "complementarity-shared"
            | "private-samples-shared"
            | "certification-cost-shared"
            | "active-indices-shared"
            | "subgroup-reduction-shared"
            | "exact-inverse-cache-shared"
            | "float-controls-shared"
            | "float-controls-selective"
            | "grouped-norm-shared"
    ) {
        assert!(cfg!(feature = "stagnant-branch-newton"));
        if kind() == "subgroup-reduction-shared"
            && !engine
                .device()
                .unwrap()
                .features()
                .contains(wgpu::Features::SUBGROUP)
        {
            return Err(invalid());
        }
        let solver = ExperimentalSoilNewtonGpu::new_with_storage(
            engine,
            input,
            cap,
            NewtonVariant::Fischer,
            |source| {
                if matches!(
                    kind().as_str(),
                    "float-controls-shared" | "float-controls-selective"
                ) {
                    let source = float_controls_shared_program(source);
                    return if kind() == "float-controls-selective" {
                        format!(
                            "{source}\n{}",
                            crate::soil_newton_protected::SELECTIVE_FLOAT_CONTROLS_MARKER
                        )
                    } else {
                        source
                    };
                }
                if kind() == "exact-inverse-cache-shared" {
                    return exact_inverse_cache::source(source);
                }
                #[cfg(feature = "stagnant-branch-newton")]
                if kind() == "subgroup-reduction-shared" {
                    return crate::soil_newton_gpu::subgroup_reduce::source(source);
                }
                let source =
                    ExperimentalSoilNewtonGpu::conditioned_program(source, true, lanes, cap);
                if kind() == "grouped-norm-shared" {
                    grouped_norm::source(
                        active_indices::source(grouped_sum::source_eight(source, false), false),
                        true,
                    )
                } else if kind() == "active-indices-shared" {
                    active_indices::source(grouped_sum::source_eight(source, false), true)
                } else if kind() == "certification-cost-shared" {
                    certification_cost::source(grouped_sum::source_eight(source, false))
                } else if kind() == "private-samples-shared" {
                    private_samples::source(grouped_sum::source_eight(source, false), true)
                } else if kind() == "complementarity-shared" {
                    complementarity::source(grouped_sum::source_eight(source, false), true)
                } else if kind() == "accepted-rates-shared" {
                    accepted_rates::source(grouped_sum::source_eight(source, false), true)
                } else if kind() == "pivot-suffix-shared" {
                    pivot_suffix::source(source, true)
                } else if kind() == "grouped-sum-shared" {
                    grouped_sum::source(source, true)
                } else if kind() == "grouped-eight-shared" {
                    grouped_sum::source_eight(source, true)
                } else {
                    adjugate::shared_source(source)
                }
            },
            "newton",
            true,
        )?;
        engine.queue().unwrap().write_buffer(
            &solver.scratch[7],
            20,
            &u32::from(candidate).to_le_bytes(),
        );
        return Ok(solver);
    }
    if kind() == "adjugate" && !candidate {
        assert!(cfg!(feature = "stagnant-branch-newton"));
        return ExperimentalSoilNewtonGpu::new_fischer_conditioned(engine, input, cap);
    }
    let mut solver = ExperimentalSoilNewtonGpu::new_with_storage(
        engine,
        input,
        cap,
        NewtonVariant::Fischer,
        |s| {
            if kind() == "vector-soa" {
                return vector_soa::program(s, candidate);
            }
            if kind() == "block-initialization" {
                return block_initialization::program(s, candidate);
            }
            if kind() == "basis-tiles" {
                return basis_tiles::program(s, candidate);
            }
            if kind() == "soa-scratch" {
                return soa_scratch::program(s, candidate);
            }
            if kind() == "float-controls-specialized" {
                return float_controls_specialized_program(s, candidate);
            }
            if kind() == "float-controls" {
                return float_controls_program(s, candidate);
            }
            if kind() == "adjugate" {
                let s = adjugate::baseline(s);
                return if candidate { adjugate::source(s) } else { s };
            }
            if kind() == "native-division" {
                return native_division_program(s, candidate);
            }
            if kind() == "unit-division" {
                return unit_division_program(s, candidate);
            }
            if kind() == "fixed-width" {
                let s = native_division_program(s, false);
                return if candidate {
                    fixed_width::source(s, cap.dofs).expect("validated Newton DOF count")
                } else {
                    s
                };
            }
            if kind() == "conditioned-cache" {
                let s = native_division_program(s, false);
                return if candidate {
                    conditioned_cache::source(s)
                } else {
                    s
                };
            }
            if kind() == "shared-core" {
                let s = native_division_program(s, false);
                return if candidate {
                    shared_core::source(s)
                } else {
                    conditioned_cache::source(s)
                };
            }
            if kind() == "fast-continuation" {
                let s = shared_core::source(native_division_program(s, false));
                return if candidate {
                    fast_continuation::source(s)
                } else {
                    s
                };
            }
            if kind() == "residual-value" {
                let s = shared_core::source(native_division_program(s, false));
                return if candidate {
                    residual_value::source(s)
                } else {
                    s
                };
            }
            if kind() == "stream-block" {
                let s = shared_core::source(native_division_program(s, false));
                return if candidate {
                    stream_block::source(s)
                } else {
                    s
                };
            }
            if kind() == "mobility-cache" {
                let s =
                    stream_block::source(shared_core::source(native_division_program(s, false)));
                return if candidate {
                    mobility_cache::source(s)
                } else {
                    s
                };
            }
            if kind() == "slot-constants" {
                let s = slot_constants::baseline(s);
                return if candidate {
                    slot_constants::source(s)
                } else {
                    s
                };
            }
            if kind() == "lagged-preconditioner" {
                let s = slot_constants::baseline(s);
                return if candidate {
                    lagged_preconditioner::source(s)
                } else {
                    s
                };
            }
            if kind() == "common-handoff" {
                let s = common_handoff::baseline(s);
                return if candidate {
                    common_handoff::source(s, false)
                } else {
                    s
                };
            }
            let s = super::linear_capture::method20_source(s);
            marked(if candidate { selected_source(s) } else { s })
        },
        "newton",
        true,
    )?;
    if kind() == "common-handoff" && candidate {
        common_handoff::attach(engine, &mut solver)?;
    }
    Ok(solver)
}

pub(crate) fn native_division_program(s: String, candidate: bool) -> String {
    let s =
        super::super::retained_conditioned::source(super::super::retained_conditioned::baseline(s));
    marked(if candidate {
        format!(
            "{}{s}",
            crate::soil_newton_protected::NATIVE_DIVISION_MARKER
        )
    } else {
        s
    })
}

pub(crate) fn unit_division_program(s: String, candidate: bool) -> String {
    let s = native_division_program(s, false);
    if candidate {
        format!(
            "{s}\n{}",
            crate::soil_newton_protected::UNIT_DIVISION_MARKER
        )
    } else {
        s
    }
}

pub(crate) fn float_controls_program(s: String, candidate: bool) -> String {
    let s = active_indices::source(complementarity::baseline(s), false);
    if candidate {
        format!(
            "{s}\n{}",
            crate::soil_newton_protected::FLOAT_CONTROLS_MARKER
        )
    } else {
        s
    }
}

pub(crate) fn float_controls_specialized_program(s: String, candidate: bool) -> String {
    let s = float_controls_program(s, candidate);
    if candidate {
        format!(
            "{s}\n{}",
            crate::soil_newton_protected::SELECTIVE_FLOAT_CONTROLS_MARKER
        )
    } else {
        s
    }
}

pub(crate) fn float_controls_shared_program(s: String) -> String {
    format!(
        "{}\n{}",
        float_controls_program(s, true),
        crate::soil_newton_protected::SHARED_FLOAT_CONTROLS_MARKER
    )
}

pub(crate) fn kind() -> String {
    let kind = std::env::var("MOLLA_NEWTON_MIXED_KIND").unwrap_or_else(|_| "inner".into());
    assert!(
        matches!(
            kind.as_str(),
            "inner"
                | "adjugate"
                | "adjugate-shared"
                | "pivot-suffix-shared"
                | "grouped-sum-shared"
                | "grouped-eight-shared"
                | "accepted-rates-shared"
                | "complementarity-shared"
                | "private-samples-shared"
                | "certification-cost-shared"
                | "active-indices-shared"
                | "subgroup-reduction-shared"
                | "exact-inverse-cache-shared"
                | "grouped-norm-shared"
                | "native-division"
                | "float-controls"
                | "float-controls-shared"
                | "float-controls-selective"
                | "float-controls-specialized"
                | "soa-scratch"
                | "basis-tiles"
                | "block-initialization"
                | "vector-soa"
                | "unit-division"
                | "fixed-width"
                | "conditioned-cache"
                | "shared-core"
                | "fast-continuation"
                | "residual-value"
                | "stream-block"
                | "mobility-cache"
                | "slot-constants"
                | "lagged-preconditioner"
                | "common-handoff"
                | "precondition"
                | "retained-direct"
                | "retained-direct-lazy"
                | "retained-direct-quality"
                | "direct-quality-trace"
        ),
        "unknown mixed arithmetic policy"
    );
    assert!(
        !(kind.starts_with("retained-direct")
            || kind == "direct-quality-trace"
            || kind == "native-division")
            && kind != "unit-division"
            && kind != "float-controls"
            && kind != "float-controls-shared"
            && kind != "float-controls-selective"
            && kind != "float-controls-specialized"
            && kind != "soa-scratch"
            && kind != "basis-tiles"
            && kind != "block-initialization"
            && kind != "vector-soa"
            && kind != "fixed-width"
            && kind != "conditioned-cache"
            && kind != "shared-core"
            && kind != "fast-continuation"
            && kind != "residual-value"
            && kind != "stream-block"
            && kind != "mobility-cache"
            && kind != "slot-constants"
            && kind != "lagged-preconditioner"
            && kind != "common-handoff"
            && kind != "adjugate"
            && kind != "adjugate-shared"
            && kind != "pivot-suffix-shared"
            && kind != "grouped-sum-shared"
            && kind != "grouped-eight-shared"
            && kind != "complementarity-shared"
            && kind != "private-samples-shared"
            && kind != "certification-cost-shared"
            && kind != "active-indices-shared"
            && kind != "subgroup-reduction-shared"
            && kind != "exact-inverse-cache-shared"
            || cfg!(feature = "retained-local-newton")
    );
    kind
}

pub(crate) fn selected_source(s: String) -> String {
    match kind().as_str() {
        "inner" => source(s),
        "precondition" => precondition_source(s),
        "retained-direct"
        | "retained-direct-lazy"
        | "retained-direct-quality"
        | "direct-quality-trace" => {
            panic!("retained-direct uses its full method-23 factory, not mixed linear capture")
        }
        "native-division" => panic!("native division requires the complete method-23 source"),
        "unit-division" => panic!("unit division requires the complete method-23 source"),
        "fixed-width" => panic!("fixed width requires the complete method-23 source"),
        "conditioned-cache" => panic!("conditioned cache requires the complete method-23 source"),
        "shared-core" => panic!("shared core requires the complete method-23 source"),
        "fast-continuation" => panic!("fast continuation requires the complete method-25 source"),
        "residual-value" => panic!("residual value requires the complete method-25 source"),
        "stream-block" => panic!("stream block requires the complete method-25 source"),
        "mobility-cache" => panic!("mobility cache requires the complete method-26 source"),
        "common-handoff" => panic!("common handoff requires the complete method-26 source"),
        _ => unreachable!(),
    }
}

fn precondition_source(mut s: String) -> String {
    let a = s.find("fn precondition(c:u32,x:Six)->Six {").unwrap();
    let b = a + s[a..].find("\n}").unwrap() + 2;
    s.replace_range(
        a..b,
        include_str!("../shaders/soil_newton_precondition_f32.wgsl"),
    );
    s
}

#[test]
fn newton_mixed_precondition_source_contract() {
    let control = super::linear_capture::method20_source(shader_source_for(NewtonVariant::Fischer));
    let candidate = precondition_source(control.clone());
    for name in [
        "build_inverse",
        "apply_operator",
        "apply_direction",
        "true_direction_error",
        "krylov_cycle",
        "recover_direction",
        "hybrid_direction",
        "solve_bounded",
        "certify_candidate",
    ] {
        let body = |s: &str| {
            let a = s.find(&format!("fn {name}(")).unwrap();
            let b = a + s[a..].find("\n}").unwrap() + 2;
            s[a..b].to_owned()
        };
        assert_eq!(body(&control), body(&candidate), "unchanged {name}");
    }
    assert!(!candidate.contains("apply_inner_f32"));
    crate::soil_newton_protected::compile(&marked(candidate), "newton").unwrap();
}

pub(crate) fn source(mut s: String) -> String {
    let a = s.find("fn krylov_cycle(id:u32,cycle_limit:u32) {").unwrap();
    let b = a + s[a..].find("\n}").unwrap() + 2;
    let mut cycle = s[a..b].to_owned();
    replace(
        &mut cycle,
        "storageBarrier();apply_operator();",
        "storageBarrier();apply_inner_f32();",
    );
    s.replace_range(a..b, &cycle);
    s.push_str(include_str!("../shaders/soil_newton_inner_f32.wgsl"));
    s
}

#[test]
fn newton_mixed_source_contract() {
    let control = super::linear_capture::method20_source(shader_source_for(NewtonVariant::Fischer));
    let candidate = source(control.clone());
    for name in [
        "apply_operator",
        "apply_direction",
        "true_direction_error",
        "recover_direction",
        "scratch_norm",
        "hybrid_direction",
        "solve_bounded",
        "certify_candidate",
    ] {
        let body = |s: &str| {
            let a = s.find(&format!("fn {name}(")).unwrap();
            let b = a + s[a..].find("\n}").unwrap() + 2;
            s[a..b].to_owned()
        };
        assert_eq!(body(&control), body(&candidate), "unchanged {name}");
    }
    assert_eq!(
        candidate
            .matches("storageBarrier();apply_inner_f32();")
            .count(),
        1
    );
    crate::soil_newton_protected::compile(&marked(candidate), "newton").unwrap();
}

#[test]
#[ignore = "requires protected GPU; approximate zero residual must not hide paired error"]
fn newton_mixed_cancellation_guard() {
    use crate::soil_shared_wall_friction_gpu::capture;
    use wgpu::util::DeviceExt;
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let device = engine.device().unwrap();
    let buffer = |data: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let mass = buffer(&[1.0]);
    let free = buffer(&[0.0; 2]);
    let rows = buffer(&[0.0; 4]);
    let contacts = buffer(&[0.0; 16]);
    let upstream = buffer(&[0.0]);
    let solver=ExperimentalSoilNewtonGpu::new_with_storage(&engine,SoilNewtonInputs {
        mass:&mass,free_rates:&free,point_rows:&rows,point_response:&rows,contacts:&contacts,upstream_status:&upstream,
    },SoilNewtonCapacity {dofs:1,contacts:1,newton_iterations:64,krylov_iterations:64,max_owned_bytes:u64::MAX},NewtonVariant::Fischer,|s| {
        let mut s=source(super::linear_capture::method20_source(s));
        s.push_str(r#"
@compute @workgroup_size(256)
fn mixed_range_probe(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {
        atomicStore(&status.fault,0u);status.active_count=1u;active_indices[0]=0u;
        var b=Block();b.residual[1]=-1.0;
        for(var i=0u;i<6u;i++) {b.direct[7u*i]=vec2<f32>(1.0,0.0);}
        b.direct[0]=vec2<f32>(16777216.0,1.0);b.direct[1]=vec2<f32>(-16777216.0,0.0);blocks[0]=b;
        vector[0]=Six(vec4<f32>(1.0,1.0,0.0,0.0),vec4<f32>(0.0));direction[0]=vector[0];
        basis[params.iterations].z=Six();refinement_norm=1.0;
    }
    storageBarrier();workgroupBarrier();apply_inner_f32();
    if lane==0u {scratch[0].k.y-=1.0;}
    storageBarrier();let estimate=scratch_norm();let truth=true_direction_error();
    if lane==0u {report.physical=estimate;report.residual=truth;atomicStore(&report.fault,select(0u,8u,truth>params.tolerance));}
}
"#);marked(s)
    },"mixed_range_probe",true).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    solver.encode_reset(&mut encoder);
    solver.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    let report: [u32; 8] =
        bytemuck::pod_read_unaligned(&capture::read(&engine, solver.status_buffer()));
    assert_eq!(report[0], 8);
    assert_eq!(f32::from_bits(report[5]), 0.0);
    assert_eq!(f32::from_bits(report[4]), 1.0);
    println!("mixed cancellation: approximate relative=0 paired relative=1 rejected at 1e-5");
}
