use super::*;
use crate::soil_newton_gpu::{
    ExperimentalSoilNewtonGpu, NewtonVariant, SoilNewtonCapacity, SoilNewtonInputs,
};

#[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_numerical_guard_tests.rs"]
mod numerical_guard_tests;

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[path = "soil_admm_gpu_probe.rs"]
mod admm_gpu_probe;

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_pipeline_cache_tests.rs"]
mod pipeline_cache_tests;

#[path = "soil_newton_release_tests.rs"]
mod local_release;

#[path = "soil_newton_condensed_tests.rs"]
mod condensed_tests;

#[path = "soil_newton_conditioned_tests.rs"]
mod conditioned_tests;

#[path = "soil_newton_cost_tests.rs"]
mod cost_tests;

#[path = "soil_newton_precision_trace_tests.rs"]
mod precision_trace_tests;

#[path = "soil_newton_publication_trace_tests.rs"]
mod publication_trace_tests;

#[path = "soil_newton_convergence_trace_tests.rs"]
mod convergence_trace_tests;

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_distributed_nonlinear_tests.rs"]
mod distributed_nonlinear_tests;

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_mixed_tests.rs"]
mod mixed_tests;

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_seed_cost_tests.rs"]
mod seed_cost_tests;

#[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_first_release_cost_tests.rs"]
mod first_release_cost_tests;

#[cfg(all(feature = "stagnant-branch-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_certification_isolated_tests.rs"]
mod certification_isolated_tests;

fn capacity(n: usize, c: usize) -> SoilNewtonCapacity {
    SoilNewtonCapacity {
        dofs: n as u32,
        contacts: c as u32,
        newton_iterations: 64,
        krylov_iterations: 64,
        max_owned_bytes: u64::MAX,
    }
}

pub(super) fn create(
    engine: &Engine,
    op: &ExperimentalSoilNewtonOperatorGpu,
    p: &Reference,
    c: usize,
) -> ExperimentalSoilNewtonGpu {
    create_capacity(engine, op, p, capacity(p.free.len(), c))
}

fn create_capacity(
    engine: &Engine,
    op: &ExperimentalSoilNewtonOperatorGpu,
    p: &Reference,
    cap: SoilNewtonCapacity,
) -> ExperimentalSoilNewtonGpu {
    create_capacity_arithmetic(engine, op, p, cap, NewtonVariant::Paired)
}

pub(super) fn create_native(
    engine: &Engine,
    op: &ExperimentalSoilNewtonOperatorGpu,
    p: &Reference,
    c: usize,
) -> ExperimentalSoilNewtonGpu {
    create_capacity_arithmetic(
        engine,
        op,
        p,
        capacity(p.free.len(), c),
        NewtonVariant::NativeF64,
    )
}

pub(super) fn create_cooperative(
    engine: &Engine,
    op: &ExperimentalSoilNewtonOperatorGpu,
    p: &Reference,
    c: usize,
) -> ExperimentalSoilNewtonGpu {
    create_capacity_arithmetic(
        engine,
        op,
        p,
        capacity(p.free.len(), c),
        NewtonVariant::Cooperative,
    )
}

pub(super) fn create_sorted(
    engine: &Engine,
    op: &ExperimentalSoilNewtonOperatorGpu,
    p: &Reference,
    c: usize,
) -> ExperimentalSoilNewtonGpu {
    create_capacity_arithmetic(
        engine,
        op,
        p,
        capacity(p.free.len(), c),
        NewtonVariant::Sorted,
    )
}

pub(super) fn create_chunked(
    engine: &Engine,
    op: &ExperimentalSoilNewtonOperatorGpu,
    p: &Reference,
    c: usize,
) -> ExperimentalSoilNewtonGpu {
    create_capacity_arithmetic(
        engine,
        op,
        p,
        capacity(p.free.len(), c),
        NewtonVariant::Chunked,
    )
}

pub(super) fn create_stagnation(
    engine: &Engine,
    op: &ExperimentalSoilNewtonOperatorGpu,
    p: &Reference,
    c: usize,
) -> ExperimentalSoilNewtonGpu {
    create_capacity_arithmetic(
        engine,
        op,
        p,
        capacity(p.free.len(), c),
        NewtonVariant::Stagnation,
    )
}

fn create_capacity_arithmetic(
    engine: &Engine,
    op: &ExperimentalSoilNewtonOperatorGpu,
    p: &Reference,
    cap: SoilNewtonCapacity,
    arithmetic: NewtonVariant,
) -> ExperimentalSoilNewtonGpu {
    let pairs: Vec<_> = p
        .free
        .iter()
        .map(|q| [*q as f32, (*q - (*q as f32) as f64) as f32])
        .collect();
    let free = storage(engine, &pairs);
    let mass = storage(engine, &p.mass);
    let [rows, response, contacts] = op.point_inputs();
    let inputs = || SoilNewtonInputs {
        mass: &mass,
        free_rates: &free,
        point_rows: rows,
        point_response: response,
        contacts,
        upstream_status: op.status_buffer(),
    };
    let direct = arithmetic == NewtonVariant::Fischer
        && std::env::var_os("MOLLA_NEWTON_TEST_DIRECT").is_some();
    let hybrid = arithmetic == NewtonVariant::Fischer
        && std::env::var_os("MOLLA_NEWTON_TEST_HYBRID").is_some();
    let adaptive_hybrid = arithmetic == NewtonVariant::Fischer
        && std::env::var_os("MOLLA_NEWTON_TEST_ADAPTIVE_HYBRID").is_some();
    let conditioned = arithmetic == NewtonVariant::Fischer
        && std::env::var_os("MOLLA_NEWTON_TEST_CONDITIONED").is_some();
    let bytes = if direct || hybrid || adaptive_hybrid || conditioned {
        ExperimentalSoilNewtonGpu::required_condensed_bytes(cap)
    } else if arithmetic == NewtonVariant::Fischer {
        ExperimentalSoilNewtonGpu::required_fischer_bytes(cap)
    } else {
        ExperimentalSoilNewtonGpu::required_bytes(cap)
    }
    .unwrap();
    let constructor = match arithmetic {
        NewtonVariant::NativeF64 => ExperimentalSoilNewtonGpu::new_native_f64,
        NewtonVariant::Cooperative => ExperimentalSoilNewtonGpu::new_cooperative_pairs,
        NewtonVariant::Chunked => ExperimentalSoilNewtonGpu::new_chunked_pairs,
        NewtonVariant::Stagnation => ExperimentalSoilNewtonGpu::new_stagnation_pairs,
        NewtonVariant::Fischer => {
            if conditioned {
                ExperimentalSoilNewtonGpu::new_fischer_conditioned
            } else if adaptive_hybrid {
                ExperimentalSoilNewtonGpu::new_fischer_adaptive_hybrid
            } else if hybrid {
                ExperimentalSoilNewtonGpu::new_fischer_hybrid
            } else if direct {
                ExperimentalSoilNewtonGpu::new_fischer_direct
            } else if std::env::var_os("MOLLA_NEWTON_TEST_LOCAL_RELEASE").is_some() {
                ExperimentalSoilNewtonGpu::new_fischer_continuation_release
            } else if std::env::var_os("MOLLA_NEWTON_TEST_CONTINUATION").is_some() {
                ExperimentalSoilNewtonGpu::new_fischer_continuation
            } else {
                ExperimentalSoilNewtonGpu::new_fischer_pairs
            }
        }
        NewtonVariant::SortedSerial => ExperimentalSoilNewtonGpu::new_sorted_serial,
        NewtonVariant::Sorted => ExperimentalSoilNewtonGpu::new_sorted_pairs,
        NewtonVariant::Paired => ExperimentalSoilNewtonGpu::new,
    };
    let allocation = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        constructor(
            engine,
            inputs(),
            SoilNewtonCapacity {
                max_owned_bytes: bytes - 1,
                ..cap
            }
        )
        .is_err()
    );
    assert_eq!(allocation.finish().observations, 0);
    let result = constructor(
        engine,
        inputs(),
        SoilNewtonCapacity {
            max_owned_bytes: bytes,
            ..cap
        },
    )
    .unwrap();
    assert_eq!(result.owned_buffer_bytes(), bytes);
    use crate::soil_newton_failure::NewtonFailureMethod;
    let expected = match arithmetic {
        NewtonVariant::Stagnation => NewtonFailureMethod::StagnationPairs,
        NewtonVariant::Fischer if conditioned => {
            if crate::soil_newton_gpu::protected_enabled(engine) {
                if cfg!(feature = "soa-scratch-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedSoaScratch256
                } else if cfg!(feature = "parallel-indices-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedParallelIndices256
                } else if cfg!(feature = "grouped-eight-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedGroupedEight256
                } else if cfg!(feature = "grouped-sum-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedGroupedSum256
                } else if cfg!(feature = "stagnant-branch-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedStagnantBranch256
                } else if cfg!(feature = "stream-block-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedStreamBlock256
                } else if cfg!(feature = "shared-conditioned-core-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedSharedCore256
                } else if cfg!(feature = "conditioned-projection-cache-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedProjectionCache256
                } else if cfg!(feature = "retained-local-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedRetainedLocal256
                } else if cfg!(feature = "publication-products-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedPublicationProducts256
                } else if cfg!(feature = "active-publication-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedActivePublication256
                } else if cfg!(feature = "norm-tree-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedNorm256
                } else if cfg!(feature = "schur-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedSchur256
                } else if cfg!(feature = "volatile-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedVolatile256
                } else if cfg!(feature = "parallel-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedParallel256
                } else if cfg!(feature = "cached-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedCached256
                } else if cfg!(feature = "wide-newton") {
                    NewtonFailureMethod::FischerConditionedProtectedWide256
                } else {
                    NewtonFailureMethod::FischerConditionedProtectedStalled
                }
            } else {
                NewtonFailureMethod::FischerConditionedStalled
            }
        }
        NewtonVariant::Fischer if adaptive_hybrid => NewtonFailureMethod::FischerAdaptiveHybrid,
        NewtonVariant::Fischer if hybrid => NewtonFailureMethod::FischerHybrid,
        NewtonVariant::Fischer if direct => NewtonFailureMethod::FischerDirect,
        NewtonVariant::Fischer if std::env::var_os("MOLLA_NEWTON_TEST_LOCAL_RELEASE").is_some() => {
            NewtonFailureMethod::FischerContinuationRelease
        }
        NewtonVariant::Fischer if std::env::var_os("MOLLA_NEWTON_TEST_CONTINUATION").is_some() => {
            NewtonFailureMethod::FischerContinuation
        }
        NewtonVariant::Fischer => NewtonFailureMethod::FischerPairs,
        _ => NewtonFailureMethod::Unspecified,
    };
    assert_eq!(result.failure_method(), expected);
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_PUBLICATION_INPUT_CACHE").is_some() {
        assert!(conditioned);
        result.set_publication_input_cache_probe(engine, true);
    }
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_SCALAR_PRODUCT").is_some() {
        assert!(conditioned);
        result.set_scalar_product_probe(engine, true);
    }
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_OPERATOR_ROWS").is_some() {
        assert!(conditioned);
        result.set_operator_rows_probe(engine, true);
    }
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_RESIDUAL_FUSION").is_some() {
        assert!(conditioned);
        result.set_residual_fusion_probe(engine, true);
    }
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_NUMERICAL_GUARD_LIFECYCLE").is_some() {
        assert!(conditioned);
        result.set_numerical_guard_probe(engine, true);
    }
    result
}

fn run(engine: &Engine, solver: &ExperimentalSoilNewtonGpu, reset: bool) -> [u32; 8] {
    let device = engine.device().unwrap();
    let timestamp = device
        .features()
        .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
        .then(|| {
            (
                device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("Newton replay timestamps"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2,
                }),
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("Newton replay timestamps"),
                    size: 16,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                }),
            )
        });
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if reset {
        solver.encode_reset(&mut encoder);
    }
    if let Some((queries, _)) = &timestamp {
        encoder.write_timestamp(queries, 0);
    }
    solver.encode(&mut encoder);
    if let Some((queries, resolved)) = &timestamp {
        encoder.write_timestamp(queries, 1);
        encoder.resolve_query_set(queries, 0..2, resolved, 0);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(
        dispatch.finish().encoded_dispatches,
        solver.replay_dispatch_count()
    );
    if let Some((_, resolved)) = &timestamp {
        let data = capture::read(engine, resolved);
        let ticks: [u64; 2] = bytemuck::pod_read_unaligned(&data[..16]);
        assert!(ticks[1] >= ticks[0]);
        eprintln!(
            "Newton replay GPU timestamp_ms={}",
            (ticks[1] - ticks[0]) as f64
                * f64::from(engine.queue().unwrap().get_timestamp_period())
                / 1e6
        );
    }
    bytemuck::pod_read_unaligned(&capture::read(engine, solver.status_buffer())[..32])
}

fn check(
    engine: &Engine,
    solver: &ExperimentalSoilNewtonGpu,
    f: &Fixture,
    p: &Reference,
) -> Vec<f64> {
    let status = run(engine, solver, true);
    eprintln!(
        "resident Newton contacts={} status={:?} residual={:e} physical={:e} damping={:e}",
        p.contacts.len(),
        &status[..4],
        f32::from_bits(status[4]),
        f32::from_bits(status[5]),
        f32::from_bits(status[6])
    );
    let states = values(engine, solver.state_buffer());
    let x: Vec<_> = p
        .contacts
        .iter()
        .flat_map(|c| {
            (0..6).map(|i| {
                let k = 16 * c.index + if i < 3 { i } else { i + 1 };
                states[k] as f64 + states[k + 8] as f64
            })
        })
        .collect();
    eprintln!(
        "resident Newton independent physical={:e} residual={:e}",
        p.physical_error(&x),
        norm(&p.residual(&x))
    );
    assert_eq!(status[0], 0, "nonlinear status {status:?}");
    let publication: [u32; 8] = bytemuck::pod_read_unaligned(
        &capture::read(engine, solver.publication_status_buffer())[..32],
    );
    assert_eq!(
        publication[0], 0,
        "resident publication status {publication:?}"
    );
    super::publication::verify_outputs(
        engine,
        solver.outputs_buffer(),
        solver.rates_buffer(),
        f,
        p,
    );
    assert!(status[1] <= 64 && status[2] <= 64 * 64 * 9 && status[3] <= 64 * 24);
    assert!(p.physical_error(&x) <= 1e-6);
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_NUMERICAL_GUARD_LIFECYCLE").is_some() {
        numerical_guard_tests::check_published(engine, solver, p, &x);
    }
    verify_solution(f, p, &x);
    for c in 0..f.contacts.len().max(1) {
        for i in [3, 7, 11, 15] {
            assert_eq!(states[16 * c + i].to_bits(), 0);
        }
        if !p.contacts.iter().any(|x| x.index == c) {
            assert!(
                states[16 * c..16 * (c + 1)]
                    .iter()
                    .all(|v| v.to_bits() == 0)
            );
        }
    }
    x
}

#[test]
fn resident_newton_diagnostic_shader() {
    for (arithmetic, limit) in [NewtonVariant::Stagnation, NewtonVariant::Fischer]
        .into_iter()
        .flat_map(|v| [1, 512, 36864].map(|limit| (v, limit)))
    {
        let source = crate::soil_newton_gpu::diagnostic_source(
            crate::soil_newton_gpu::shader_source_for(arithmetic),
            limit,
        );
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    }
}

#[test]
#[ignore = "requires GPU; explicit diagnostic work limit"]
fn resident_newton_diagnostic_work_limit() {
    diagnostic_work_limit(NewtonVariant::Stagnation);
}

#[test]
#[ignore = "requires GPU; Fischer schedule diagnostic work limit"]
fn resident_newton_fischer_diagnostic_work_limit() {
    diagnostic_work_limit(NewtonVariant::Fischer);
}

fn diagnostic_work_limit(arithmetic: NewtonVariant) {
    let limit = crate::soil_newton_gpu::diagnostic_inner_limit()
        .unwrap()
        .unwrap();
    assert_eq!(limit, 1);
    let engine = Engine::new_gpu().unwrap();
    let f = articulated(65);
    let p = Reference::new(&f, f.free.iter().map(|&v| v as f64).collect());
    let zero = vec![0.; p.contacts.len() * 6];
    let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver = create_capacity_arithmetic(
        &engine,
        &op,
        &p,
        capacity(p.free.len(), f.contacts.len()),
        arithmetic,
    );
    let first = run(&engine, &solver, true);
    assert_eq!((first[0], first[7]), (32, limit));
    assert!((1..=64).contains(&first[2]));
    let status: [u32; 8] = bytemuck::pod_read_unaligned(
        &capture::read(&engine, solver.publication_status_buffer())[..32],
    );
    assert_ne!(status[0], 0);
    assert_eq!(run(&engine, &solver, false), first);
    assert_eq!(run(&engine, &solver, true), first);
}

#[test]
fn resident_newton_local_release_shader() {
    use crate::soil_newton_gpu::{diagnostic_source, shader_source_for};
    for (diagnostic, mode) in [
        (false, 0),
        (true, 0),
        (false, 1),
        (true, 1),
        (false, 2),
        (true, 2),
        (false, 3),
        (true, 3),
        (false, 4),
        (true, 4),
        (false, 5),
        (true, 5),
        (false, 6),
        (true, 6),
    ] {
        let source = shader_source_for(NewtonVariant::Fischer);
        let source = if diagnostic {
            diagnostic_source(source, 512)
        } else {
            source
        };
        let condensed = mode != 0;
        let source = if mode == 6 {
            ExperimentalSoilNewtonGpu::conditioned_source(source)
        } else if mode == 5 {
            ExperimentalSoilNewtonGpu::adaptive_hybrid_source(source)
        } else if mode == 4 {
            ExperimentalSoilNewtonGpu::adaptive_direct_source(source)
        } else if mode == 3 {
            ExperimentalSoilNewtonGpu::hybrid_source(source)
        } else if mode == 2 {
            ExperimentalSoilNewtonGpu::direct_source(source)
        } else if condensed {
            ExperimentalSoilNewtonGpu::condensed_source(source)
        } else {
            ExperimentalSoilNewtonGpu::local_release_source(source)
        };
        assert_eq!(source.matches("let candidate=evaluate();").count(), 1);
        assert!(source.contains("outer.releases)>=8u"));
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        let mut layout = wgpu::naga::proc::Layouter::default();
        layout.update(module.to_ctx()).unwrap();
        let shared: u32 = module
            .global_variables
            .iter()
            .filter(|(_, v)| v.space == wgpu::naga::AddressSpace::WorkGroup)
            .map(|(_, v)| layout[v.ty].to_stride())
            .sum();
        assert!(shared <= 24000, "workgroup bytes: {shared}");
        assert_eq!(
            module
                .global_variables
                .iter()
                .filter(|(_, v)| v.binding.is_some())
                .count(),
            21 + usize::from(condensed)
        );
    }
}

#[test]
fn resident_newton_shader_and_budget() {
    assert_eq!(
        ExperimentalSoilNewtonGpu::required_bytes(capacity(32, 4096)).unwrap(),
        19874176
    );
    for contacts in [0, 1, 65, 4096] {
        let c = capacity(32, contacts);
        assert_eq!(
            ExperimentalSoilNewtonGpu::required_fischer_bytes(c).unwrap(),
            ExperimentalSoilNewtonGpu::required_bytes(c).unwrap() + 280 * contacts.max(1) as u64
        );
        assert_eq!(
            ExperimentalSoilNewtonGpu::required_condensed_bytes(c).unwrap(),
            ExperimentalSoilNewtonGpu::required_fischer_bytes(c).unwrap()
                + 33280
                + 432 * contacts.max(1) as u64
        );
    }
    for c in [
        SoilNewtonCapacity {
            dofs: 33,
            ..capacity(3, 1)
        },
        SoilNewtonCapacity {
            newton_iterations: 0,
            ..capacity(3, 1)
        },
        SoilNewtonCapacity {
            krylov_iterations: 65,
            ..capacity(3, 1)
        },
    ] {
        assert!(ExperimentalSoilNewtonGpu::required_bytes(c).is_err());
        assert!(ExperimentalSoilNewtonGpu::required_fischer_bytes(c).is_err());
        assert!(ExperimentalSoilNewtonGpu::required_condensed_bytes(c).is_err());
    }
    for arithmetic in [
        NewtonVariant::Paired,
        NewtonVariant::Cooperative,
        NewtonVariant::Chunked,
        NewtonVariant::Stagnation,
        NewtonVariant::Fischer,
        NewtonVariant::SortedSerial,
        NewtonVariant::Sorted,
        NewtonVariant::NativeF64,
    ] {
        let source = crate::soil_newton_gpu::shader_source_for(arithmetic);
        if arithmetic == NewtonVariant::Fischer {
            assert_eq!(source.matches("evaluate();").count(), 1);
        }
        assert_eq!(
            source.contains("fn release_separating_wall()"),
            arithmetic == NewtonVariant::Stagnation
        );
        assert_eq!(
            source.contains("fn reduce_parts()"),
            matches!(
                arithmetic,
                NewtonVariant::Cooperative
                    | NewtonVariant::Chunked
                    | NewtonVariant::Stagnation
                    | NewtonVariant::Fischer
            )
        );
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        let block = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("Block"))
            .unwrap()
            .1;
        let wgpu::naga::TypeInner::Struct { span, .. } = &block.inner else {
            panic!("Block must be a struct")
        };
        assert_eq!(
            *span,
            if arithmetic == NewtonVariant::Fischer {
                456
            } else {
                240
            }
        );
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    }
}

#[test]
#[ignore = "requires native GPU FP64; numerical fixtures and physical publication"]
fn resident_newton_native_f64_fixtures() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_F64).unwrap();
    arithmetic_fixtures(&engine, NewtonVariant::NativeF64);
}

#[test]
#[ignore = "requires GPU; sorted-pair physical fixture checks"]
fn resident_newton_cooperative_pairs_fixtures() {
    let engine = Engine::new_gpu().unwrap();
    arithmetic_fixtures(&engine, NewtonVariant::Cooperative);
}

#[test]
#[ignore = "requires GPU; chunked projection physical fixtures"]
fn resident_newton_chunked_pairs_fixtures() {
    let engine = Engine::new_gpu().unwrap();
    arithmetic_fixtures(&engine, NewtonVariant::Chunked);
}

#[test]
#[ignore = "requires GPU; stagnation fallback physical fixtures"]
fn resident_newton_stagnation_pairs_fixtures() {
    let engine = Engine::new_gpu().unwrap();
    arithmetic_fixtures(&engine, NewtonVariant::Stagnation);
}

#[test]
#[ignore = "requires GPU; stagnation fallback lifecycle"]
fn resident_newton_stagnation_pairs_reuse_and_faults() {
    reuse_and_faults(NewtonVariant::Stagnation);
}

#[test]
#[ignore = "requires GPU; Fischer complementarity lifecycle and physical publication"]
fn resident_newton_fischer_pairs_reuse_and_faults() {
    reuse_and_faults(NewtonVariant::Fischer);
}

#[test]
#[ignore = "requires GPU; Fischer empty and invalid mass"]
fn resident_newton_fischer_pairs_empty() {
    empty_and_mass(NewtonVariant::Fischer);
}

#[cfg(feature = "soa-scratch-newton")]
#[test]
#[ignore = "requires protected GPU; conditioned SoA sparse and empty lifecycle on one engine"]
fn resident_newton_conditioned_soa_lifecycle() {
    assert!(std::env::var_os("MOLLA_NEWTON_TEST_CONDITIONED").is_some());
    let engine = replay_engine();
    reuse_and_faults_on(&engine, NewtonVariant::Fischer);
    empty_and_mass_on(&engine, NewtonVariant::Fischer);
}

#[test]
#[ignore = "requires GPU; paired direction precision and spare basis lifecycle"]
fn resident_newton_fischer_direction_storage() {
    let engine = Engine::new_gpu().unwrap();
    let f = articulated(65);
    let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let zero = vec![0.0; 6 * p.contacts.len()];
    let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
    assert_eq!(encode(&engine, &op, true), 0);
    for k in [1, 64] {
        let mut cap = capacity(p.free.len(), f.contacts.len());
        cap.krylov_iterations = k;
        let mut solver = create_capacity_arithmetic(&engine, &op, &p, cap, NewtonVariant::Fischer);
        solver.use_paired_direction_probe(&engine);
        for reset in [true, false] {
            assert_eq!(run(&engine, &solver, reset)[0], 0);
            let output = values(&engine, solver.transpose_probe_output());
            for c in 0..65 {
                assert!((output[8 * c] as f64 - 3e-8).abs() < 1e-14);
                assert!((output[8 * c + 1] as f64 - 6e-8).abs() < 1e-14);
                assert!(output[8 * c + 2..8 * c + 8].iter().all(|&v| v == 0.0));
            }
        }
    }
}

#[test]
#[ignore = "requires GPU; stagnation fallback empty and invalid mass"]
fn resident_newton_stagnation_pairs_empty() {
    empty_and_mass(NewtonVariant::Stagnation);
}

#[test]
#[ignore = "requires GPU; true-residual stopping and inconsistent linear system"]
fn resident_newton_stagnation_true_residual() {
    let engine = Engine::new_gpu().unwrap();
    let f = articulated(7);
    let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let zero = vec![0.0; 6 * p.contacts.len()];
    let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
    assert_eq!(encode(&engine, &op, true), 0);
    for singular in [false, true] {
        let mut solver = create_stagnation(&engine, &op, &p, f.contacts.len());
        solver.use_stopping_probe(&engine, singular);
        let status = run(&engine, &solver, true);
        let [blocks, direction] = solver.linear_probe_buffers();
        let blocks = values(&engine, blocks);
        let direction = values(&engine, direction);
        let mut rhs = Vec::new();
        let mut residual = Vec::new();
        for c in 0..7 {
            for i in 0..6 {
                let b = blocks[60 * c + i] as f64;
                let a = blocks[60 * c + 6 + 7 * i] as f64;
                let d = direction[8 * c + if i < 3 { i } else { i + 1 }] as f64;
                rhs.push(b);
                residual.push(a * d + b);
            }
        }
        let relative = norm(&residual) / norm(&rhs);
        let reported = f32::from_bits(status[5]) as f64;
        eprintln!(
            "stopping singular={singular} fault={} iterations={} estimate={:e} true_gpu={reported:e} true_f64={relative:e}",
            status[0],
            status[2],
            f32::from_bits(status[7])
        );
        assert!((relative - reported).abs() <= 1e-6);
        if singular {
            assert_eq!(status[0], 8);
            assert!((16..64).contains(&status[2]));
            assert_eq!(status[2] % 8, 0);
            assert_eq!(status[3], 1);
            assert!(relative > 0.9);
        } else {
            assert_eq!(status[0], 0);
            assert_eq!(status[3], 0);
            assert!(relative <= 1e-5);
            assert!(status[2] < 16);
        }
    }
}

#[test]
#[ignore = "requires GPU; shared Krylov budget across residual corrections"]
fn resident_newton_fischer_refinement_budget() {
    let engine = Engine::new_gpu().unwrap();
    let f = articulated(7);
    let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let zero = vec![0.0; 6 * p.contacts.len()];
    let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
    assert_eq!(encode(&engine, &op, true), 0);
    for (singular, limit) in [(false, 64), (true, 1), (true, 17), (true, 64)] {
        let cap = SoilNewtonCapacity {
            krylov_iterations: limit,
            ..capacity(p.free.len(), f.contacts.len())
        };
        let mut solver = create_capacity_arithmetic(&engine, &op, &p, cap, NewtonVariant::Fischer);
        solver.use_stopping_probe(&engine, singular);
        let status = run(&engine, &solver, true);
        let [blocks, direction] = solver.linear_probe_buffers();
        let blocks = values(&engine, blocks);
        let direction = values(&engine, direction);
        let basis = values(&engine, solver.basis_buffer());
        let mut rhs = Vec::new();
        let mut residual = Vec::new();
        for c in 0..7 {
            for i in 0..6 {
                let component = if i < 3 { i } else { i + 1 };
                let b = blocks[114 * c + i] as f64;
                let a = blocks[114 * c + 6 + 14 * i] as f64 + blocks[114 * c + 7 + 14 * i] as f64;
                let d = direction[8 * c + component] as f64
                    + basis[16 * limit as usize * 7 + 16 * c + 8 + component] as f64;
                rhs.push(b);
                residual.push(a * d + b);
            }
        }
        let relative = norm(&residual) / norm(&rhs);
        eprintln!(
            "refinement budget singular={singular} limit={limit} iterations={} cycles={} relative={relative:e}",
            status[2], status[3]
        );
        assert!((1..=limit).contains(&status[2]));
        assert!((relative - f32::from_bits(status[5]) as f64).abs() <= 1e-6);
        if singular {
            assert_eq!(status[0], 8);
            assert!(relative > 0.9);
            if limit == 64 {
                assert!(status[3] > 1);
            }
        } else {
            assert_eq!(status[0], 0);
            assert!(relative <= 1e-5);
        }
    }
}

fn arithmetic_fixtures(engine: &Engine, arithmetic: NewtonVariant) {
    for f in [
        pair(false, 0.0),
        pair(false, 0.3),
        pair(true, 0.3),
        articulated(7),
        articulated(65),
        wide_fixture(),
    ] {
        let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
        let zero = vec![0.0; 6 * p.contacts.len()];
        let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
        assert_eq!(encode(&engine, &op, true), 0);
        let solver = create_capacity_arithmetic(
            engine,
            &op,
            &p,
            capacity(p.free.len(), f.contacts.len()),
            arithmetic,
        );
        let first = check(&engine, &solver, &f, &p);
        assert_eq!(first, check(&engine, &solver, &f, &p));
    }
}

#[test]
#[ignore = "requires GPU without enabled FP64; rejection before allocations"]
fn resident_newton_native_f64_requires_feature() {
    let engine = Engine::new_gpu().unwrap();
    assert!(
        !engine
            .device()
            .unwrap()
            .features()
            .contains(wgpu::Features::SHADER_F64)
    );
    let f = pair(false, 0.0);
    let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let zero = vec![0.0; 12];
    let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
    let mass = storage(&engine, &p.mass);
    let free = storage(
        &engine,
        &f.free.iter().map(|&q| [q, 0.0]).collect::<Vec<_>>(),
    );
    let [rows, response, contacts] = op.point_inputs();
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        ExperimentalSoilNewtonGpu::new_native_f64(
            &engine,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: rows,
                point_response: response,
                contacts,
                upstream_status: op.status_buffer(),
            },
            capacity(f.free.len(), f.contacts.len())
        )
        .is_err()
    );
    assert_eq!(scope.finish().observations, 0);
}

#[test]
#[ignore = "requires GPU; resident nonlinear solve and independent physical certificate"]
fn resident_newton_gpu_fixtures() {
    let engine = Engine::new_gpu().unwrap();
    let mut empty = pair(false, 0.0);
    empty.contacts.clear();
    empty.rows.clear();
    for f in [
        pair(false, 0.0),
        pair(false, 0.3),
        pair(true, 0.3),
        articulated(7),
        articulated(65),
        wide_fixture(),
        empty,
    ] {
        let p = Reference::new(&f, f.free.iter().map(|&x| x as f64).collect());
        let x = vec![0.0; 6 * p.contacts.len()];
        let (op, _, _) = create_operator(&engine, &f, &p, &x, &x);
        assert_eq!(encode(&engine, &op, true), 0);
        let solver = create(&engine, &op, &p, f.contacts.len());
        let first = check(&engine, &solver, &f, &p);
        assert_eq!(first, check(&engine, &solver, &f, &p));
    }
}

#[test]
#[ignore = "requires GPU; cached transpose parity and independent f64 products"]
fn resident_newton_cached_transpose_products() {
    let engine = Engine::new_gpu().unwrap();
    for f in [
        spread_contacts(&articulated(7), 512, 64),
        articulated(7),
        articulated(65),
        wide_fixture(),
        spread_contacts(&articulated(65), 4096, 63),
    ] {
        let (p, x, v) = sample(&f);
        let (op, states, _) = create_operator(&engine, &f, &p, &x, &v);
        assert_eq!(encode(&engine, &op, true), 0);
        let mut input = vec![SoilNewtonVector::default(); f.contacts.len().max(1)];
        for (c, v) in p.contacts.iter().zip(v.chunks_exact(6)) {
            input[c.index] = vector(v);
        }
        let mut results = Vec::new();
        for arithmetic in [NewtonVariant::SortedSerial, NewtonVariant::Cooperative] {
            let mut solver = create_capacity_arithmetic(
                &engine,
                &op,
                &p,
                capacity(p.free.len(), f.contacts.len()),
                arithmetic,
            );
            solver.use_transpose_probe(&engine);
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(&states, 0, solver.state_buffer(), 0, states.size());
            engine.queue().unwrap().submit([encoder.finish()]);
            let [blocks, direction] = solver.linear_probe_buffers();
            engine
                .queue()
                .unwrap()
                .write_buffer(direction, 0, bytemuck::cast_slice(&input));
            assert_eq!(run(&engine, &solver, false)[0], 0);
            let out = values(&engine, solver.transpose_probe_output());
            let blocks = values(&engine, blocks);
            let linear = Linearization {
                problem: &p,
                direct: p
                    .contacts
                    .iter()
                    .map(|c| {
                        std::array::from_fn(|i| {
                            std::array::from_fn(|j| blocks[60 * c.index + 6 + 6 * i + j] as f64)
                        })
                    })
                    .collect(),
                machine: p
                    .contacts
                    .iter()
                    .map(|c| {
                        std::array::from_fn(|i| {
                            std::array::from_fn(|j| blocks[60 * c.index + 42 + 3 * i + j] as f64)
                        })
                    })
                    .collect(),
            };
            let represented: Vec<_> = p
                .contacts
                .iter()
                .flat_map(|c| {
                    (0..6).map(|i| {
                        if i < 3 {
                            input[c.index].contact[i] as f64
                        } else {
                            input[c.index].wall[i - 3] as f64
                        }
                    })
                })
                .collect();
            let reference = linear.transpose(&represented);
            let result: Vec<_> = p
                .contacts
                .iter()
                .flat_map(|c| {
                    (0..6).map(|i| out[8 * c.index + if i < 3 { i } else { i + 1 }] as f64)
                })
                .collect();
            let difference: Vec<_> = result.iter().zip(&reference).map(|(a, b)| a - b).collect();
            let relative = norm(&difference) / (1.0 + norm(&reference));
            eprintln!(
                "cached transpose contacts={} slots={} width={} optimized={} f64_error={relative:e}",
                p.contacts.len(),
                f.contacts.len(),
                p.free.len(),
                arithmetic == NewtonVariant::Cooperative
            );
            assert!(relative <= 1e-6);
            results.push(result);
        }
        assert_eq!(results[0], results[1]);
    }
}

#[test]
#[ignore = "requires GPU; sparse indices with colliding contact-modulo lanes"]
fn resident_newton_sparse_lane_alias() {
    let engine = Engine::new_gpu().unwrap();
    let f = spread_contacts(&articulated(7), 512, 64);
    let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    assert_eq!(p.contacts.len(), 7);
    assert!(
        p.contacts
            .iter()
            .all(|c| c.index % 64 == p.contacts[0].index % 64)
    );
    let zero = vec![0.0; 6 * p.contacts.len()];
    let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
    assert_eq!(encode(&engine, &op, true), 0);
    for arithmetic in [NewtonVariant::Paired, NewtonVariant::Cooperative] {
        let solver = create_capacity_arithmetic(
            &engine,
            &op,
            &p,
            capacity(p.free.len(), f.contacts.len()),
            arithmetic,
        );
        let first = check(&engine, &solver, &f, &p);
        assert_eq!(first, check(&engine, &solver, &f, &p));
    }
}

#[test]
#[ignore = "requires GPU; damped normal equations and dense f64 direction"]
fn resident_newton_gpu_damped_stationarity() {
    damped_stationarity(NewtonVariant::Paired);
}

#[test]
#[ignore = "requires GPU; cooperative-reduction damped normal equations"]
fn resident_newton_cooperative_damped_stationarity() {
    damped_stationarity(NewtonVariant::Cooperative);
}

#[test]
#[ignore = "requires GPU; chunked projection normal equations"]
fn resident_newton_chunked_damped_stationarity() {
    damped_stationarity(NewtonVariant::Chunked);
}

fn cropped_wide(n: usize) -> Fixture {
    let mut f = wide_fixture();
    let width = f.free.len();
    let mut mass = Vec::new();
    for i in 0..n {
        mass.extend_from_slice(&f.mass[i * width..i * width + n]);
    }
    f.mass = mass;
    f.rows = f
        .rows
        .chunks_exact(width)
        .flat_map(|row| row[..n].iter().copied())
        .collect();
    f.free.truncate(n);
    f
}

#[test]
#[ignore = "requires GPU; chunked forward, transpose and residual f64 parity"]
fn resident_newton_chunked_products() {
    let engine = Engine::new_gpu().unwrap();
    for f in [
        cropped_wide(1),
        cropped_wide(3),
        spread_contacts(&cropped_wide(10), 512, 64),
        wide_fixture(),
        spread_contacts(&articulated(65), 4096, 63),
    ] {
        let (p, x, v) = sample(&f);
        let (op, states, _) = create_operator(&engine, &f, &p, &x, &v);
        assert_eq!(encode(&engine, &op, true), 0);
        let mut solver = create_chunked(&engine, &op, &p, f.contacts.len());
        solver.use_transpose_probe(&engine);
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&states, 0, solver.state_buffer(), 0, states.size());
        engine.queue().unwrap().submit([encoder.finish()]);
        let [blocks, direction] = solver.linear_probe_buffers();
        let mut input = vec![SoilNewtonVector::default(); f.contacts.len().max(1)];
        for (c, v) in p.contacts.iter().zip(v.chunks_exact(6)) {
            input[c.index] = vector(v);
        }
        engine
            .queue()
            .unwrap()
            .write_buffer(direction, 0, bytemuck::cast_slice(&input));
        assert_eq!(run(&engine, &solver, false)[0], 0);
        let blocks = values(&engine, blocks);
        let linear = Linearization {
            problem: &p,
            direct: p
                .contacts
                .iter()
                .map(|c| {
                    std::array::from_fn(|i| {
                        std::array::from_fn(|j| blocks[60 * c.index + 6 + 6 * i + j] as f64)
                    })
                })
                .collect(),
            machine: p
                .contacts
                .iter()
                .map(|c| {
                    std::array::from_fn(|i| {
                        std::array::from_fn(|j| blocks[60 * c.index + 42 + 3 * i + j] as f64)
                    })
                })
                .collect(),
        };
        let represented: Vec<_> = v.iter().map(|&v| v as f32 as f64).collect();
        let forward = values(&engine, solver.forward_probe_output());
        let transpose = values(&engine, solver.transpose_probe_output());
        let unpack = |raw: &[f32], stride: usize| {
            p.contacts
                .iter()
                .flat_map(|c| {
                    (0..6)
                        .map(move |i| raw[stride * c.index + if i < 3 { i } else { i + 1 }] as f64)
                })
                .collect::<Vec<_>>()
        };
        let residual: Vec<_> = p
            .contacts
            .iter()
            .flat_map(|c| (0..6).map(|i| blocks[60 * c.index + i] as f64))
            .collect();
        for (label, actual, expected) in [
            ("residual", residual, p.residual(&x)),
            (
                "forward",
                unpack(&forward, 16),
                linear.product(&represented),
            ),
            (
                "transpose",
                unpack(&transpose, 8),
                linear.transpose(&represented),
            ),
        ] {
            let error = norm(
                &actual
                    .iter()
                    .zip(&expected)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>(),
            ) / (1.0 + norm(&expected));
            eprintln!(
                "chunked {label} dofs={} contacts={} slots={} error={error:e}",
                p.free.len(),
                p.contacts.len(),
                f.contacts.len()
            );
            assert!(error <= 1e-6);
        }
    }
}

fn damped_stationarity(arithmetic: NewtonVariant) {
    let engine = Engine::new_gpu().unwrap();
    for (f, damping) in [
        (pair(true, 0.3), 2.0_f32),
        (articulated(7), 0.25),
        (wide_fixture(), 0.01),
        (articulated(65), 0.25),
    ] {
        let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
        let zero = vec![0.0; 6 * p.contacts.len()];
        let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
        assert_eq!(encode(&engine, &op, true), 0);
        let mut solver = create_capacity_arithmetic(
            &engine,
            &op,
            &p,
            capacity(p.free.len(), f.contacts.len()),
            arithmetic,
        );
        solver.use_damped_probe(&engine, damping);
        let status = run(&engine, &solver, true);
        let [blocks, direction] = solver.linear_probe_buffers();
        let blocks = values(&engine, blocks);
        let direction = values(&engine, direction);
        let mut d = Vec::new();
        let mut residual = Vec::new();
        let mut direct = Vec::new();
        let mut machine = Vec::new();
        for c in &p.contacts {
            d.extend((0..6).map(|i| direction[8 * c.index + if i < 3 { i } else { i + 1 }] as f64));
            residual.extend((0..6).map(|i| blocks[60 * c.index + i] as f64));
            direct.push(std::array::from_fn(|i| {
                std::array::from_fn(|j| blocks[60 * c.index + 6 + 6 * i + j] as f64)
            }));
            machine.push(std::array::from_fn(|i| {
                std::array::from_fn(|j| blocks[60 * c.index + 42 + 3 * i + j] as f64)
            }));
        }
        let linear = Linearization {
            problem: &p,
            direct,
            machine,
        };
        let lambda = (damping as f64).powi(2);
        let rhs: Vec<_> = linear.transpose(&residual).iter().map(|v| -v).collect();
        let jd_f: Vec<_> = linear
            .product(&d)
            .iter()
            .zip(&residual)
            .map(|(a, b)| a + b)
            .collect();
        let gradient: Vec<_> = linear
            .transpose(&jd_f)
            .iter()
            .zip(&d)
            .map(|(g, d)| g + lambda * d)
            .collect();
        let relative = norm(&gradient) / norm(&rhs);
        let unregularized = norm(&jd_f) / norm(&residual);
        let n = d.len();
        let mut normal = vec![0.0; n * n];
        for j in 0..n {
            let mut unit = vec![0.0; n];
            unit[j] = 1.0;
            let column = linear.transpose(&linear.product(&unit));
            for i in 0..n {
                normal[i * n + j] = column[i];
            }
            normal[j * n + j] += lambda;
        }
        let exact = solve_dense(normal, rhs).unwrap();
        let difference: Vec<_> = d.iter().zip(&exact).map(|(a, b)| a - b).collect();
        let direction_error = norm(&difference) / (1.0 + norm(&exact));
        eprintln!(
            "damped stationarity contacts={} damping={damping:e} iterations={} status={} GPU={:e} f64={relative:e} unregularized={unregularized:e} direction={direction_error:e}",
            p.contacts.len(),
            status[2],
            status[0],
            f32::from_bits(status[5])
        );
        assert_eq!(status[0], 0);
        assert!(status[2] < 64);
        assert!(relative <= 1e-5);
        let exact_residual: Vec<_> = linear
            .product(&exact)
            .iter()
            .zip(&residual)
            .map(|(a, b)| a + b)
            .collect();
        let exact_gradient: Vec<_> = linear
            .transpose(&exact_residual)
            .iter()
            .zip(&exact)
            .map(|(g, d)| g + lambda * d)
            .collect();
        assert!(norm(&exact_gradient) <= 1e-10 * (1.0 + norm(&linear.transpose(&residual))));
        assert!(
            norm(&difference)
                <= (norm(&gradient) + norm(&exact_gradient)) / lambda * (1.0 + 1e-7) + 1e-10
        );
        assert!(unregularized > 1e-5);
    }
}

#[test]
#[ignore = "requires GPU and saved shared input snapshot"]
fn resident_newton_gpu_captured_input() {
    let root = std::path::PathBuf::from(std::env::var("MOLLA_SHARED_FAILURE_REPLAY").unwrap());
    let (f, pairs) = failure::load_paired(&root).unwrap();
    let p = Reference::new(&f, combined(&pairs));
    let x = vec![0.0; 6 * p.contacts.len()];
    let engine = Engine::new_gpu().unwrap();
    let (op, _, _) = create_operator(&engine, &f, &p, &x, &x);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver = create(&engine, &op, &p, f.contacts.len());
    check(&engine, &solver, &f, &p);
}

#[test]
#[ignore = "requires GPU; opposed machine/floor contacts with unequal gaps"]
fn resident_newton_wall_release_unequal_gaps() {
    let engine = Engine::new_gpu().unwrap();
    wall_release_unequal_gaps(&engine, |engine, op, p| create_stagnation(engine, op, p, 2));
}

fn wall_release_unequal_gaps(
    engine: &Engine,
    create: impl Fn(
        &Engine,
        &ExperimentalSoilNewtonOperatorGpu,
        &Reference,
    ) -> ExperimentalSoilNewtonGpu,
) {
    for sign in [-1.0f32, 1.0] {
        for axis in 0..3 {
            let mut row = [0.; 4];
            row[axis] = 1.;
            let mut f = Fixture {
                mass: vec![10.],
                free: vec![-sign],
                rows: vec![row; 2],
                contacts: Vec::new(),
            };
            for gap in [0., 1e-4] {
                let mut soil_mass = [0., 0., 0., 1.];
                soil_mass[axis] = sign;
                let mut normal_gap = [0., 0., 0., gap];
                normal_gap[axis] = -sign;
                f.contacts.push(SoilSharedWallContact {
                    soil_mass,
                    normal_gap,
                    friction: [0.; 4],
                    boundary: [(if sign > 0. { 1 } else { 2 }) << (2 * axis), 0, 0, 0],
                });
            }
            let p = Reference::new(&f, vec![-f64::from(sign)]);
            let zero = vec![0.; 12];
            let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
            assert_eq!(encode(&engine, &op, true), 0);
            let solver = create(&engine, &op, &p);
            let x = check(&engine, &solver, &f, &p);
            assert!(x[9 + axis].abs() <= 1e-6);
            assert!(f64::from(sign) * x[3 + axis] > 7.);
            assert_eq!(x, check(&engine, &solver, &f, &p));
        }
    }
}

#[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires protected GPU; guarded first release on six unequal-gap wall directions"]
fn resident_newton_first_release_unequal_gaps() {
    first_release_unequal_gaps(crate::soil_newton_gpu::first_release::new_probe);
}

#[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires protected GPU; smoothed stall recovery on six unequal-gap wall directions"]
fn resident_newton_smoothed_release_unequal_gaps() {
    first_release_unequal_gaps(crate::soil_newton_gpu::first_release::new_smoothed_probe);
}

#[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires protected GPU; smoothed branch settling on six unequal-gap wall directions"]
fn resident_newton_smoothed_settle_unequal_gaps() {
    first_release_unequal_gaps(crate::soil_newton_gpu::first_release::smoothed_settle::new_probe);
}

#[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires protected GPU; bounded smoothing-stage recovery on six unequal-gap wall directions"]
fn resident_newton_smoothed_stages_unequal_gaps() {
    first_release_unequal_gaps(crate::soil_newton_gpu::first_release::smoothed_stages::new_probe);
}

#[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires protected GPU; stage-recovery continuation completion on six wall directions"]
fn resident_newton_stage_completion_unequal_gaps() {
    first_release_unequal_gaps(crate::soil_newton_gpu::first_release::smoothed_stages::finish_pending::new_completion_probe);
}

#[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
fn first_release_unequal_gaps(
    constructor: fn(
        &Engine,
        SoilNewtonInputs<'_>,
        SoilNewtonCapacity,
    ) -> molla_core::Result<ExperimentalSoilNewtonGpu>,
) {
    let engine = replay_engine();
    wall_release_unequal_gaps(&engine, |engine, op, p| {
        let pairs: Vec<_> = p
            .free
            .iter()
            .map(|q| [*q as f32, (*q - (*q as f32) as f64) as f32])
            .collect();
        let free = storage(engine, &pairs);
        let mass = storage(engine, &p.mass);
        let [rows, response, contacts] = op.point_inputs();
        constructor(
            engine,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: rows,
                point_response: response,
                contacts,
                upstream_status: op.status_buffer(),
            },
            capacity(p.free.len(), 2),
        )
        .unwrap()
    });
}

#[test]
#[ignore = "requires GPU and first-failure prepared Newton packet"]
fn resident_newton_gpu_failure_packet_replay() {
    let root = std::path::PathBuf::from(std::env::var("MOLLA_NEWTON_FAILURE_REPLAY").unwrap());
    replay_packet(&root, None);
}

fn replay_engine() -> Engine {
    let mut features = wgpu::Features::empty();
    if std::env::var("MOLLA_NEWTON_MIXED_KIND").as_deref() == Ok("subgroup-reduction-shared") {
        features |= wgpu::Features::SUBGROUP;
    }
    if std::env::var("MOLLA_NEWTON_MIXED_KIND").as_deref() == Ok("native-division") {
        features |= wgpu::Features::SHADER_F64;
    }
    if std::env::var_os("MOLLA_NEWTON_FAILURE_TIMESTAMPS").is_some() {
        features |=
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
    }
    if std::env::var_os("MOLLA_NEWTON_FAILURE_GPU_PRECISE").is_some()
        || cfg!(all(
            feature = "protected-newton",
            not(target_arch = "wasm32")
        ))
    {
        features |= wgpu::Features::PASSTHROUGH_SHADERS;
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    if matches!(
        std::env::var("MOLLA_NEWTON_MIXED_KIND").as_deref(),
        Ok("float-controls"
            | "float-controls-shared"
            | "float-controls-selective"
            | "float-controls-specialized")
    ) {
        return crate::soil_newton_precise_tests::float_controls::engine(features);
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_PHASE_CLOCK").is_some() {
        return crate::soil_newton_precise_tests::float_controls::clock_engine(features);
    }
    Engine::new_gpu_with_features(features).unwrap()
}

#[test]
#[ignore = "requires GPU and MOLLA_NEWTON_FAILURE_REPLAY_SET; retained-program cold-state replays"]
fn resident_newton_gpu_replay_batch() {
    let paths = std::env::var_os("MOLLA_NEWTON_FAILURE_REPLAY_SET").unwrap();
    let paths: Vec<_> = std::env::split_paths(&paths).collect();
    assert!((1..=8).contains(&paths.len()));
    assert!(std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_ONLY_SMOOTHING").is_none());
    for root in &paths {
        crate::soil_newton_failure::certificate::preflight(
            root,
            crate::soil_newton_failure::certificate::exact_requested(),
        )
        .unwrap();
    }
    let engine = replay_engine();
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_SCALAR_PRODUCT").is_some() {
        crate::soil_newton_gpu::scalar_product::probe(&engine);
    }
    for root in paths {
        eprintln!("Newton batch packet={}", root.display());
        replay_packet(&root, Some(&engine));
    }
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_INPUT_CACHE_LIFECYCLE").is_some() {
        assert!(std::env::var_os("MOLLA_NEWTON_PUBLICATION_INPUT_CACHE").is_some());
        assert!(std::env::var_os("MOLLA_NEWTON_TEST_CONDITIONED").is_some());
        reuse_and_faults_on(&engine, NewtonVariant::Fischer);
        empty_and_mass_on(&engine, NewtonVariant::Fischer);
        eprintln!("publication input cache lifecycle passed on replay engine");
    }
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_SCALAR_LIFECYCLE").is_some() {
        assert!(std::env::var_os("MOLLA_NEWTON_SCALAR_PRODUCT").is_some());
        assert!(std::env::var_os("MOLLA_NEWTON_TEST_CONDITIONED").is_some());
        reuse_and_faults_on(&engine, NewtonVariant::Fischer);
        empty_and_mass_on(&engine, NewtonVariant::Fischer);
        eprintln!("scalar product lifecycle passed on replay engine");
    }
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_OPERATOR_ROWS_LIFECYCLE").is_some() {
        assert!(std::env::var_os("MOLLA_NEWTON_OPERATOR_ROWS").is_some());
        assert!(std::env::var_os("MOLLA_NEWTON_TEST_CONDITIONED").is_some());
        reuse_and_faults_on(&engine, NewtonVariant::Fischer);
        empty_and_mass_on(&engine, NewtonVariant::Fischer);
        eprintln!("operator rows lifecycle passed on replay engine");
    }
}

fn profile_replay_method(profile: &str) -> crate::soil_newton_failure::NewtonFailureMethod {
    use crate::soil_newton_failure::NewtonFailureMethod;
    match profile {
        "volatile" => NewtonFailureMethod::FischerConditionedProtectedVolatile256,
        "soa" => NewtonFailureMethod::FischerConditionedProtectedSoaScratch256,
        _ => panic!("unknown fixed-input replay profile: {profile}"),
    }
}

#[test]
fn newton_profile_replay_identity_contract() {
    assert_ne!(
        profile_replay_method("volatile"),
        profile_replay_method("soa")
    );
    assert!(
        profile_replay_method("volatile")
            .name()
            .contains("volatile-256")
    );
    assert!(
        profile_replay_method("soa")
            .name()
            .contains("soa-scratch-256")
    );
    assert!(std::panic::catch_unwind(|| profile_replay_method("unknown")).is_err());
}

fn profile_replay_guess(
    success: u32,
    retained: u32,
    supplied: bool,
    ignored: bool,
) -> &'static str {
    assert_eq!(success, 1, "profile replay requires a successful capture");
    assert!(
        !ignored,
        "profile replay must preserve the captured initial guess"
    );
    assert_eq!(
        retained,
        u32::from(supplied),
        "profile replay seed provenance"
    );
    if supplied {
        "retained-seed"
    } else {
        "cold-zero"
    }
}

#[test]
fn newton_profile_replay_guess_contract() {
    assert_eq!(profile_replay_guess(1, 0, false, false), "cold-zero");
    assert_eq!(profile_replay_guess(1, 1, true, false), "retained-seed");
    for (success, retained, supplied, ignored) in [
        (0, 0, false, false),
        (1, 1, false, false),
        (1, 0, true, false),
        (1, 0, false, true),
        (1, 1, true, true),
        (1, 2, true, false),
    ] {
        assert!(
            std::panic::catch_unwind(|| profile_replay_guess(success, retained, supplied, ignored))
                .is_err()
        );
    }
}

fn replay_packet(root: &std::path::Path, shared_engine: Option<&Engine>) {
    let bytes = std::fs::read(root.join("newton.bin")).unwrap();
    let meta: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("capture.json")).unwrap()).unwrap();
    assert_eq!(
        meta["blake3"].as_str().unwrap(),
        blake3::hash(&bytes).to_hex().as_str()
    );
    crate::soil_newton_failure::certificate::validate_replay(
        &bytes,
        &meta,
        crate::soil_newton_failure::certificate::exact_requested(),
    )
    .unwrap();
    let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    assert_eq!(word(0), 1);
    assert_eq!((word(3), word(4)), (0, 0));
    let n = word(1) as usize;
    let c = word(2) as usize;
    assert!((1..=32).contains(&n) && c <= 16384);
    assert!((1..=64).contains(&word(5)) && (1..=64).contains(&word(6)));
    assert_eq!(
        bytes.len(),
        128 + 4 * n * n + 8 * n + (32 * n + 128) * c.max(1)
    );
    let archived_report: [u32; 8] = std::array::from_fn(|i| word(16 + i));
    let mut captured_seed = crate::soil_newton_failure::replay_seed(root, &meta, c).unwrap();
    if word(8) == 1 {
        assert_eq!(
            word(9),
            u32::from(captured_seed.is_some()),
            "success seed provenance"
        );
    }
    if std::env::var_os("MOLLA_NEWTON_FAILURE_IGNORE_SEED").is_some() {
        eprintln!("explicit cold comparison: ignoring captured initial guess");
        captured_seed = None;
    } else if captured_seed.is_some() {
        eprintln!("replaying hashed retained initial guess");
        for flag in [
            "CPU_ONLY_SMOOTHING",
            "CPU_SEED_GPU",
            "WARM",
            "REFERENCE_STEP",
            "PRECISION_TRACE",
            "RELEASE_COST",
        ] {
            assert!(
                std::env::var_os(format!("MOLLA_NEWTON_FAILURE_{flag}")).is_none(),
                "seeded replay does not support {flag}"
            );
        }
        assert!(std::env::var_os("MOLLA_NEWTON_PRECISION_TRACE").is_none());
    }
    let mass: Vec<_> = bytes[128..128 + 4 * n * n]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let pairs: Vec<[f32; 2]> = bytes[128 + 4 * n * n..128 + 4 * n * n + 8 * n]
        .chunks_exact(8)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    let mut f = Fixture {
        mass,
        free: combined(&pairs).iter().map(|v| *v as f32).collect(),
        rows: Vec::new(),
        contacts: Vec::new(),
    };
    let mut response: Vec<[f32; 4]> = Vec::new();
    let mut archived_state = Vec::new();
    let mut offset = 128 + 4 * n * n + 8 * n;
    for _ in 0..c.max(1) {
        f.rows.extend(
            bytes[offset..offset + 16 * n]
                .chunks_exact(16)
                .map(bytemuck::pod_read_unaligned::<[f32; 4]>),
        );
        offset += 16 * n;
        response.extend(
            bytes[offset..offset + 16 * n]
                .chunks_exact(16)
                .map(bytemuck::pod_read_unaligned::<[f32; 4]>),
        );
        offset += 16 * n;
        f.contacts
            .push(bytemuck::pod_read_unaligned(&bytes[offset..offset + 64]));
        offset += 64;
        archived_state.extend_from_slice(&bytes[offset..offset + 64]);
        offset += 64;
    }
    f.contacts.truncate(c);
    let p = Reference::new(&f, combined(&pairs));
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_ADMM_CPU") {
        assert!(captured_seed.is_none(), "ADMM diagnostic starts from zero");
        assert!(
            shared_engine.is_none(),
            "ADMM diagnostic must not construct a GPU engine"
        );
        admm::diagnose(
            &p,
            &response,
            &archived_state,
            meta["blake3"].as_str().unwrap(),
            &std::path::PathBuf::from(output),
        );
        return;
    }
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_PIVOT_TRACE") {
        assert!(captured_seed.is_none());
        let output = std::path::PathBuf::from(output);
        assert!(!output.exists(), "refusing to overwrite pivot evidence");
        let mut reference = p.clone();
        reference.use_prepared_response(&response);
        for contact in &mut reference.contacts {
            contact.fischer = true;
        }
        if let Some(prefixes) = std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_PREFIX_PIVOT") {
            convergence_trace_tests::pivot_prefixes(
                &reference,
                c,
                &std::path::PathBuf::from(prefixes),
                &output,
            );
            return;
        }
        let word = |i: usize| {
            f32::from_le_bytes(archived_state[4 * i..4 * i + 4].try_into().unwrap()) as f64
        };
        let x: Vec<_> = reference
            .contacts
            .iter()
            .flat_map(|contact| {
                (0..6).map(|i| {
                    let offset = 16 * contact.index + if i < 3 { i } else { i + 1 };
                    word(offset) + word(offset + 8)
                })
            })
            .collect();
        let visited_ids: Vec<usize> = std::env::var("MOLLA_NEWTON_FAILURE_CPU_PIVOT_VISITED")
            .unwrap_or_default()
            .split(',')
            .filter(|s| !s.is_empty())
            .map(|s| s.parse().unwrap())
            .collect();
        assert!(visited_ids.len() <= 8);
        assert!(
            visited_ids
                .iter()
                .all(|id| reference.contacts.iter().any(|c| c.index == *id))
        );
        let mut results = Vec::new();
        let base = norm(&reference.residual(&x));
        for conditioned in [false, true] {
            for all_branches in [false, true] {
                reference.conditioned_pivot = conditioned;
                let mut visited: Vec<_> = reference
                    .contacts
                    .iter()
                    .map(|c| visited_ids.contains(&c.index))
                    .collect();
                let pivot = if all_branches {
                    local_pivot::pivot_all_branches
                } else {
                    local_pivot::pivot
                };
                let trial = pivot(&reference, &x, base, &mut visited);
                results.push(serde_json::json!({
                    "conditioned": conditioned, "all_branches": all_branches,
                    "residual": trial.as_ref().map(|x| norm(&reference.residual(x))),
                    "physical": trial.as_ref().map(|x| reference.physical_error(x)),
                    "visited": reference.contacts.iter().zip(&visited).filter_map(|(c,v)| v.then_some(c.index)).collect::<Vec<_>>()
                }));
            }
        }
        let result = serde_json::json!({"acceptance": false, "packet": meta["blake3"],
            "base_residual": base, "base_physical": reference.physical_error(&x),
            "initial_visited": visited_ids, "cases": results});
        std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
        return;
    }
    let cpu_seed_gpu = std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_SEED_GPU").is_some();
    assert!(!cpu_seed_gpu || std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_ONLY_SMOOTHING").is_some());
    if let Ok(smoothing) = std::env::var("MOLLA_NEWTON_FAILURE_CPU_ONLY_SMOOTHING") {
        let mut reference = p.clone();
        reference.local_pivot = std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_LOCAL_PIVOT").is_some();
        reference.condensed_damping =
            std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_CONDENSED_DAMPING").is_some();
        reference.condensed_newton =
            std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_CONDENSED_NEWTON").is_some();
        reference.stagnation_smoothing =
            std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_STAGNATION_SMOOTHING").is_some();
        reference.conditioned_pivot =
            std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_CONDITIONED_PIVOT").is_some();
        if std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_EXACT_RESPONSE").is_none() {
            reference.use_prepared_response(&response);
        }
        for contact in &mut reference.contacts {
            contact.fischer = true;
            contact.conical = std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_CONICAL").is_some();
        }
        let seed = vec![0.; 6 * reference.contacts.len()];
        let now = std::time::Instant::now();
        let (x, steps, products) =
            solve_continuation(&reference, seed, smoothing.parse().unwrap()).unwrap();
        verify_solution(&f, &reference, &x);
        eprintln!(
            "CPU-only cold continuation steps={steps} products={products} physical={:e} residual={:e} wall_ms={}",
            reference.physical_error(&x),
            norm(&reference.residual(&x)),
            now.elapsed().as_secs_f64() * 1000.
        );
        if !cpu_seed_gpu {
            return;
        }
        let mut seeds = vec![crate::soil_newton_operator_gpu::SoilNewtonState::default(); c.max(1)];
        for (contact, local) in reference.contacts.iter().zip(x.chunks_exact(6)) {
            seeds[contact.index] = state(local);
        }
        archived_state.copy_from_slice(bytemuck::cast_slice(&seeds));
    }
    let owned_engine = shared_engine.is_none().then(replay_engine);
    let engine = shared_engine.unwrap_or_else(|| owned_engine.as_ref().unwrap());
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    if let Some(output) = std::env::var_os("MOLLA_ADMM_GPU_PROBE") {
        assert!(
            captured_seed.is_none(),
            "ADMM probe uses explicit cold start"
        );
        admm_gpu_probe::replay(
            engine,
            &f,
            &p,
            &pairs,
            &response,
            meta["blake3"].as_str().unwrap(),
            &std::path::PathBuf::from(output),
        );
        return;
    }
    let mass = storage(&engine, &f.mass);
    let free = storage(&engine, &pairs);
    let rows = storage(&engine, &f.rows);
    let response_values = response.clone();
    let response = storage(&engine, &response);
    let contacts = buffer(&engine, &f.contacts, 64);
    let upstream = storage(&engine, &[0u32]);
    #[cfg(all(feature = "stagnant-branch-newton", not(target_arch = "wasm32")))]
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_CERTIFICATION_ISOLATED") {
        assert!(
            captured_seed.is_none(),
            "isolated publication uses its explicit archived state"
        );
        assert!(std::env::var_os("MOLLA_NEWTON_MIXED_REPLAY").is_none());
        certification_isolated_tests::compare(
            engine,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                point_response: &response,
                contacts: &contacts,
                upstream_status: &upstream,
            },
            &f,
            &p,
            root,
            &std::path::PathBuf::from(output),
        );
        return;
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_MIXED_REPLAY") {
        assert!(captured_seed.is_none(), "mixed comparison starts from zero");
        assert_eq!((word(5), word(6)), (64, 64));
        mixed_tests::compare(
            engine,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                point_response: &response,
                contacts: &contacts,
                upstream_status: &upstream,
            },
            &f,
            &p,
            root,
            &std::path::PathBuf::from(output),
        );
        return;
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_DISTRIBUTED_NONLINEAR") {
        assert!(captured_seed.is_none(), "cold nonlinear comparison");
        assert_eq!((word(5), word(6)), (64, 64));
        distributed_nonlinear_tests::compare(
            engine,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                point_response: &response,
                contacts: &contacts,
                upstream_status: &upstream,
            },
            &f,
            &p,
            root,
            &std::path::PathBuf::from(output),
        );
        return;
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_DISTRIBUTED_CAPTURE") {
        assert!(
            captured_seed.is_none(),
            "linear capture currently starts from zero"
        );
        assert_eq!((word(5), word(6)), (64, 64));
        crate::soil_newton_gpu::distributed::capture_linearization(
            engine,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                point_response: &response,
                contacts: &contacts,
                upstream_status: &upstream,
            },
            capacity(n, c),
            root,
            &std::path::PathBuf::from(output),
        );
        return;
    }
    if std::env::var_os("MOLLA_NEWTON_PUBLICATION_TRACE").is_some() {
        let state = storage(&engine, &archived_state);
        let status = storage(&engine, &archived_report);
        publication_trace_tests::trace(
            engine,
            crate::soil_newton_publish_gpu::SoilNewtonPublishInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                point_response: &response,
                contacts: &contacts,
                state: &state,
                upstream_status: &upstream,
                solver_status: &status,
            },
            &f,
            &p,
        );
        return;
    }
    let advance = std::env::var("MOLLA_NEWTON_FAILURE_ADVANCE")
        .map(|v| v.parse::<u32>().unwrap())
        .unwrap_or(0);
    assert!(advance < 64);
    let linear_probe = std::env::var_os("MOLLA_NEWTON_FAILURE_LINEAR_PROBE").is_some();
    let fischer = std::env::var_os("MOLLA_NEWTON_FAILURE_GPU_FISCHER").is_some();
    assert!(!linear_probe || fischer);
    let continuation = std::env::var_os("MOLLA_NEWTON_FAILURE_GPU_CONTINUATION").is_some();
    assert!(!continuation || (fischer && !linear_probe && advance == 0));
    let local_release = std::env::var_os("MOLLA_NEWTON_FAILURE_GPU_LOCAL_RELEASE").is_some();
    assert!(!local_release || continuation);
    let condensed = std::env::var_os("MOLLA_NEWTON_FAILURE_GPU_CONDENSED").is_some();
    assert!(!condensed || local_release);
    let direct = std::env::var_os("MOLLA_NEWTON_FAILURE_GPU_DIRECT").is_some();
    assert!(!direct || condensed);
    let hybrid = std::env::var_os("MOLLA_NEWTON_FAILURE_GPU_HYBRID").is_some();
    assert!(!hybrid || condensed);
    let full_jacobian = std::env::var_os("MOLLA_NEWTON_FAILURE_DIRECT_FULL_JACOBIAN").is_some();
    assert!(!full_jacobian || direct);
    assert!(!hybrid || !full_jacobian);
    let adaptive = std::env::var_os("MOLLA_NEWTON_FAILURE_GPU_ADAPTIVE_DIRECT").is_some();
    assert!(!adaptive || (direct && !hybrid && !full_jacobian));
    let adaptive_hybrid = std::env::var_os("MOLLA_NEWTON_FAILURE_GPU_ADAPTIVE_HYBRID").is_some();
    assert!(!adaptive_hybrid || (hybrid && !adaptive && !full_jacobian));
    let conditioned = std::env::var_os("MOLLA_NEWTON_FAILURE_GPU_CONDITIONED").is_some();
    assert!(!conditioned || adaptive_hybrid);
    let precise = std::env::var("MOLLA_NEWTON_FAILURE_GPU_PRECISE").ok();
    assert!(precise.is_none() || conditioned);
    assert!(
        precise
            .as_deref()
            .is_none_or(|v| matches!(v, "control" | "candidate"))
    );
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_PRECISION_TRACE") {
        assert!(conditioned && advance == 0 && !linear_probe && !cpu_seed_gpu);
        assert!(precise.is_some());
        precision_trace_tests::compare_prefixes(
            engine,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                point_response: &response,
                contacts: &contacts,
                upstream_status: &upstream,
            },
            SoilNewtonCapacity {
                newton_iterations: word(5),
                krylov_iterations: word(6),
                ..capacity(n, c)
            },
            &f,
            &p,
            &std::path::PathBuf::from(output).join(meta["blake3"].as_str().unwrap()),
        );
        return;
    }
    let constructor = if precise.as_deref() == Some("control") {
        ExperimentalSoilNewtonGpu::new_precise_control_probe
    } else if precise.as_deref() == Some("candidate") {
        ExperimentalSoilNewtonGpu::new_precise_candidate_probe
    } else if conditioned {
        ExperimentalSoilNewtonGpu::new_fischer_conditioned
    } else if adaptive_hybrid {
        ExperimentalSoilNewtonGpu::new_fischer_adaptive_hybrid
    } else if adaptive {
        ExperimentalSoilNewtonGpu::new_fischer_adaptive_direct
    } else if hybrid {
        ExperimentalSoilNewtonGpu::new_fischer_hybrid
    } else if full_jacobian {
        ExperimentalSoilNewtonGpu::new_fischer_direct_full
    } else if direct {
        ExperimentalSoilNewtonGpu::new_fischer_direct
    } else if condensed {
        ExperimentalSoilNewtonGpu::new_fischer_condensed
    } else if local_release {
        ExperimentalSoilNewtonGpu::new_fischer_continuation_release
    } else if continuation {
        ExperimentalSoilNewtonGpu::new_fischer_continuation
    } else if linear_probe && advance == 0 {
        ExperimentalSoilNewtonGpu::new_fischer_linear_probe
    } else if fischer {
        ExperimentalSoilNewtonGpu::new_fischer_pairs
    } else {
        ExperimentalSoilNewtonGpu::new_stagnation_pairs
    };
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    let constructor = if std::env::var_os("MOLLA_NEWTON_STALLED_RELEASE").is_some() {
        assert!(conditioned && precise.is_none());
        ExperimentalSoilNewtonGpu::new_stalled_release_probe
    } else {
        constructor
    };
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    let constructor = if let Ok(lanes) = std::env::var("MOLLA_NEWTON_WIDE_LANES") {
        assert!(conditioned && precise.is_none() && !linear_probe && advance == 0);
        assert!(matches!(lanes.as_str(), "128" | "256"));
        eprintln!(
            "protected resident Newton test workgroup lanes={lanes}; mathematical budgets unchanged"
        );
        if std::env::var_os("MOLLA_NEWTON_WIDE_COMPARE").is_some() {
            constructor
        } else {
            ExperimentalSoilNewtonGpu::new_wide_probe
        }
    } else {
        constructor
    };
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    let constructor = if std::env::var_os("MOLLA_NEWTON_DELTA_CACHE").is_some() {
        assert!(
            conditioned
                && precise.is_none()
                && !linear_probe
                && advance == 0
                && std::env::var_os("MOLLA_NEWTON_WIDE_LANES").is_none()
        );
        eprintln!(
            "conditioned correction cache prototype; lanes={}",
            crate::soil_newton_gpu::execution_lanes(true)
        );
        ExperimentalSoilNewtonGpu::new_cached_delta_probe
    } else {
        constructor
    };
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    let constructor = if let Ok(policy) = std::env::var("MOLLA_NEWTON_SMOOTHED_RELEASE") {
        assert!(matches!(
            policy.as_str(),
            "1" | "trace" | "settle" | "stages" | "finish-trace" | "finish"
        ));
        assert!(conditioned && advance == 0 && !linear_probe && !cpu_seed_gpu && precise.is_none());
        assert!(captured_seed.is_none());
        for conflict in [
            "MOLLA_NEWTON_SUCCESS_EXACT",
            "MOLLA_NEWTON_FIRST_RELEASE",
            "MOLLA_NEWTON_FIRST_RELEASE_COMPARE",
        ] {
            assert!(std::env::var_os(conflict).is_none());
        }
        if policy == "trace" {
            crate::soil_newton_gpu::first_release::smoothed_trace::new_probe
        } else if policy == "settle" {
            crate::soil_newton_gpu::first_release::smoothed_settle::new_probe
        } else if policy == "stages" {
            crate::soil_newton_gpu::first_release::smoothed_stages::new_probe
        } else if policy == "finish-trace" {
            crate::soil_newton_gpu::first_release::smoothed_stages::finish_pending::new_trace_probe
        } else if policy == "finish" {
            crate::soil_newton_gpu::first_release::smoothed_stages::finish_pending::new_completion_probe
        } else {
            crate::soil_newton_gpu::first_release::new_smoothed_probe
        }
    } else {
        constructor
    };
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    let constructor = if let Ok(policy) = std::env::var("MOLLA_NEWTON_FIRST_RELEASE") {
        assert_eq!(policy, "1");
        assert!(conditioned && advance == 0 && !linear_probe && !cpu_seed_gpu && precise.is_none());
        assert!(std::env::var_os("MOLLA_NEWTON_SUCCESS_EXACT").is_none());
        crate::soil_newton_gpu::first_release::new_probe
    } else {
        constructor
    };
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    let constructor = if std::env::var_os("MOLLA_NEWTON_FIRST_RELEASE_COMPARE").is_some() {
        assert!(std::env::var_os("MOLLA_NEWTON_FIRST_RELEASE").is_none());
        assert!(std::env::var_os("MOLLA_NEWTON_SUCCESS_EXACT").is_none());
        assert!(
            conditioned
                && advance == 0
                && !linear_probe
                && !cpu_seed_gpu
                && precise.is_none()
                && captured_seed.is_none()
        );
        crate::soil_newton_gpu::first_release::new_shared_probe
    } else {
        constructor
    };
    let mut solver = constructor(
        &engine,
        SoilNewtonInputs {
            mass: &mass,
            free_rates: &free,
            point_rows: &rows,
            point_response: &response,
            contacts: &contacts,
            upstream_status: &upstream,
        },
        SoilNewtonCapacity {
            dofs: n as u32,
            contacts: c as u32,
            newton_iterations: if advance > 0 { 1 } else { word(5) },
            krylov_iterations: word(6),
            max_owned_bytes: u64::MAX,
        },
    )
    .unwrap();
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_NUMERICAL_GUARD") {
        assert!(conditioned && advance == 0 && !linear_probe && !cpu_seed_gpu && precise.is_none());
        numerical_guard_tests::compare(
            engine,
            &solver,
            &p,
            captured_seed.as_deref().expect("same-solve seed"),
            &archived_state,
            root,
            &std::path::PathBuf::from(output),
        );
        return;
    }
    let replay_profile = std::env::var("MOLLA_NEWTON_PROFILE_REPLAY").ok();
    if let Some(profile) = &replay_profile {
        assert!(conditioned && advance == 0 && !linear_probe && !cpu_seed_gpu);
        assert!(precise.is_none());
        profile_replay_guess(
            word(8),
            word(9),
            captured_seed.is_some(),
            std::env::var_os("MOLLA_NEWTON_FAILURE_IGNORE_SEED").is_some(),
        );
        assert_eq!(solver.failure_method(), profile_replay_method(profile));
        assert!(std::env::var_os("MOLLA_NEWTON_FAILURE_OUTPUT").is_some());
        assert_eq!(
            std::env::var("MOLLA_NEWTON_FAILURE_REPEATS").as_deref(),
            Ok("3")
        );
    }
    #[cfg(all(feature = "soa-scratch-newton", not(target_arch = "wasm32")))]
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_FIRST_RELEASE_COMPARE") {
        first_release_cost_tests::compare(
            engine,
            &solver,
            &f,
            &p,
            root,
            &std::path::PathBuf::from(output),
        );
        return;
    }
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_CONVERGENCE_TRACE") {
        assert!(conditioned && advance == 0 && !linear_probe && !cpu_seed_gpu);
        assert_eq!(word(5), 64);
        convergence_trace_tests::trace(
            engine,
            &solver,
            &p,
            &std::path::PathBuf::from(output),
            captured_seed.as_deref(),
        );
        return;
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_SEED_COST") {
        assert!(conditioned && advance == 0 && !linear_probe && !cpu_seed_gpu);
        assert_eq!(meta["method"], solver.failure_method().name());
        let zero_seed;
        let seed = if let Some(seed) = captured_seed.as_deref() {
            seed
        } else {
            assert!(
                std::env::var_os("MOLLA_NEWTON_PUBLICATION_ENERGY").is_some(),
                "same-solve seed required"
            );
            zero_seed = vec![0u8; solver.state_buffer().size() as usize];
            eprintln!(
                "energy comparison: explicit zero seed; archived state/report equality still required"
            );
            &zero_seed
        };
        seed_cost_tests::compare(
            engine,
            &solver,
            &f,
            &p,
            seed,
            root,
            &std::path::PathBuf::from(output),
        );
        return;
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_WIDE_COMPARE").is_some() {
        assert!(
            conditioned
                && precise.is_none()
                && captured_seed.is_none()
                && !cpu_seed_gpu
                && !linear_probe
                && advance == 0
        );
        cost_tests::compare_width_cost(
            engine,
            &solver,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                point_response: &response,
                contacts: &contacts,
                upstream_status: &upstream,
            },
            SoilNewtonCapacity {
                newton_iterations: word(5),
                krylov_iterations: word(6),
                ..capacity(n, c)
            },
            &f,
            &p,
        );
        return;
    }
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    if std::env::var_os("MOLLA_NEWTON_DELTA_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_PARALLEL_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_VOLATILE_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_TILED_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_CANDIDATE_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_NONLINEAR_CACHE_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_PHASE_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_SPARSE_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_INVERSE_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_SHARED_INVERSE_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_NOINLINE_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_SCHEDULE_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_SCHUR_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_SCHUR_INVERSE_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_ROW_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_CGS2_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_COUNT_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_OPERATION_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_NORM_COMPARE").is_some()
        || std::env::var_os("MOLLA_NEWTON_SLICED_COMPARE").is_some()
    {
        assert!(
            conditioned
                && precise.is_none()
                && captured_seed.is_none()
                && !cpu_seed_gpu
                && !linear_probe
                && advance == 0
        );
        assert!(
            std::env::var_os("MOLLA_NEWTON_WIDE_LANES").is_none()
                && std::env::var_os("MOLLA_NEWTON_DELTA_CACHE").is_none()
        );
        cost_tests::compare_delta_cost(
            engine,
            &solver,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                point_response: &response,
                contacts: &contacts,
                upstream_status: &upstream,
            },
            SoilNewtonCapacity {
                newton_iterations: word(5),
                krylov_iterations: word(6),
                ..capacity(n, c)
            },
            &f,
            &p,
        );
        return;
    }
    if std::env::var_os("MOLLA_NEWTON_FAILURE_RELEASE_COST").is_some() {
        assert!(conditioned && advance == 0 && !linear_probe && !cpu_seed_gpu);
        assert_eq!(archived_report[0], 0);
        assert!(std::env::var_os("MOLLA_NEWTON_FAILURE_TIMESTAMPS").is_some());
        cost_tests::compare_release_cost(
            &engine,
            &solver,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                point_response: &response,
                contacts: &contacts,
                upstream_status: &upstream,
            },
            SoilNewtonCapacity {
                newton_iterations: word(5),
                krylov_iterations: word(6),
                ..capacity(n, c)
            },
            &f,
            &p,
        );
    }
    if std::env::var_os("MOLLA_NEWTON_FAILURE_NO_PRECONDITIONER").is_some() {
        solver.use_identity_preconditioner(&engine);
    }
    let warm = cpu_seed_gpu || std::env::var_os("MOLLA_NEWTON_FAILURE_WARM").is_some();
    assert!(
        !cpu_seed_gpu
            || (!linear_probe
                && advance == 0
                && std::env::var_os("MOLLA_NEWTON_FAILURE_REFERENCE_STEP").is_none())
    );
    if std::env::var_os("MOLLA_NEWTON_FAILURE_REFERENCE_STEP").is_some() {
        assert!(linear_probe && warm && advance == 0);
        assert!(std::env::var_os("MOLLA_NEWTON_FAILURE_PREPARED_REFERENCE").is_some());
        let mut reference = Reference::new(&f, p.free.clone());
        reference.use_prepared_response(&response_values);
        for contact in &mut reference.contacts {
            contact.fischer = true;
        }
        let word = |i: usize| {
            f32::from_le_bytes(archived_state[4 * i..4 * i + 4].try_into().unwrap()) as f64
        };
        let x: Vec<_> = reference
            .contacts
            .iter()
            .flat_map(|contact| {
                (0..6).map(|i| {
                    let offset = 16 * contact.index + if i < 3 { i } else { i + 1 };
                    word(offset) + word(offset + 8)
                })
            })
            .collect();
        let residual = reference.residual(&x);
        let linear = Linearization::new(&reference, &x);
        let pre = reference.preconditioner(&x);
        let rhs: Vec<_> = residual.iter().map(|r| -*r).collect();
        let (y, iterations, error) = gmres(|v| linear.product(&apply_blocks(&pre, v)), &rhs, 64);
        assert!(error <= 1e-5);
        let direction = apply_blocks(&pre, &y);
        let mut accepted = None;
        for attempt in 0..16 {
            let alpha = 2.0_f64.powi(-attempt);
            let candidate: Vec<_> = x
                .iter()
                .zip(&direction)
                .map(|(x, d)| x + alpha * d)
                .collect();
            if norm(&reference.residual(&candidate)) < norm(&residual) * (1.0 - 1e-4 * alpha) {
                eprintln!(
                    "FB reference-only seed step alpha={alpha:e} iterations={iterations} relative={error:e}"
                );
                accepted = Some(candidate);
                break;
            }
        }
        let next = accepted.expect("bounded reference seed step must decrease residual");
        for (contact, local) in reference.contacts.iter().zip(next.chunks_exact(6)) {
            for (i, &value) in local.iter().enumerate() {
                let offset = 16 * contact.index + if i < 3 { i } else { i + 1 };
                let high = value as f32;
                let low = (value - high as f64) as f32;
                archived_state[4 * offset..4 * offset + 4].copy_from_slice(&high.to_le_bytes());
                archived_state[4 * (offset + 8)..4 * (offset + 8) + 4]
                    .copy_from_slice(&low.to_le_bytes());
            }
        }
    }
    if let Some(path) = std::env::var_os("MOLLA_NEWTON_PIVOT_INSPECT_STATE") {
        assert!(warm && conditioned && advance == 0 && !linear_probe && !cpu_seed_gpu);
        assert!(std::env::var_os("MOLLA_NEWTON_PIVOT_INSPECT_CONTACT").is_some());
        assert!(std::env::var_os("MOLLA_NEWTON_PIVOT_INSPECT_OUTPUT").is_some());
        let path = std::path::PathBuf::from(path);
        assert_eq!(path.metadata().unwrap().len(), archived_state.len() as u64);
        archived_state = std::fs::read(path).unwrap();
    }
    if warm {
        engine
            .queue()
            .unwrap()
            .write_buffer(solver.state_buffer(), 0, &archived_state);
    }
    if advance > 0 {
        assert!(linear_probe && warm);
        for step in 0..advance {
            let advanced = run(&engine, &solver, false);
            eprintln!("FB advance step={step} report={advanced:?}");
            assert_eq!(advanced[0], 8);
            engine.queue().unwrap().write_buffer(
                solver.status_buffer(),
                0,
                bytemuck::cast_slice(&[0u32; 8]),
            );
        }
    }
    if linear_probe && advance > 0 {
        solver.use_linear_probe(&engine);
    }
    assert!(captured_seed.is_none() || (!warm && !linear_probe && advance == 0));
    let execute = || {
        if let Some(seed) = &captured_seed {
            engine
                .queue()
                .unwrap()
                .write_buffer(solver.state_buffer(), 0, seed);
            engine.queue().unwrap().write_buffer(
                solver.status_buffer(),
                0,
                bytemuck::cast_slice(&[0u32; 8]),
            );
        }
        run(&engine, &solver, !warm && captured_seed.is_none())
    };
    let status = execute();
    let repeats = std::env::var("MOLLA_NEWTON_FAILURE_REPEATS")
        .map(|v| v.parse::<u32>().unwrap())
        .unwrap_or(1);
    assert!((1..=3).contains(&repeats));
    assert!(repeats == 1 || (!warm && !linear_probe && advance == 0));
    let repeated_buffers = [
        ("states", solver.state_buffer()),
        ("rates", solver.rates_buffer()),
        ("outputs", solver.outputs_buffer()),
        ("publication", solver.publication_status_buffer()),
    ];
    let expected_buffers: Vec<_> = if repeats > 1 {
        repeated_buffers
            .iter()
            .map(|(_, buffer)| capture::read(&engine, buffer))
            .collect()
    } else {
        Vec::new()
    };
    for _ in 1..repeats {
        assert_eq!(execute(), status);
        for ((name, buffer), expected) in repeated_buffers.iter().zip(&expected_buffers) {
            assert_eq!(
                &capture::read(&engine, buffer),
                expected,
                "cold repeat {name}"
            );
        }
    }
    if repeats > 1 {
        eprintln!("cold replay exact report/state/rates/output/publication repeats={repeats}");
    }
    if cpu_seed_gpu {
        assert_eq!(
            status[1], 0,
            "CPU seed must certify without a GPU Newton step"
        );
    }
    let publication: [u32; 8] = bytemuck::pod_read_unaligned(
        &capture::read(&engine, solver.publication_status_buffer())[..32],
    );
    let states = values(&engine, solver.state_buffer());
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_BARRIER_FOLD_OUTPUT") {
        let folded = std::env::var_os("MOLLA_NEWTON_BARRIER_FOLD").is_some();
        let control = std::env::var_os("MOLLA_NEWTON_BARRIER_FOLD_CONTROL").is_some();
        assert_ne!(folded, control);
        let output = std::path::PathBuf::from(output).join(meta["blake3"].as_str().unwrap());
        assert!(!output.exists());
        std::fs::create_dir_all(&output).unwrap();
        for (name, buffer) in [
            ("states", solver.state_buffer()),
            ("rates", solver.rates_buffer()),
            ("outputs", solver.outputs_buffer()),
            ("status", solver.status_buffer()),
            ("publication", solver.publication_status_buffer()),
        ] {
            std::fs::write(
                output.join(format!("{name}.bin")),
                capture::read(engine, buffer),
            )
            .unwrap();
        }
        let x: Vec<_> = p
            .contacts
            .iter()
            .flat_map(|contact| {
                (0..6).map(|i| {
                    let k = 16 * contact.index + if i < 3 { i } else { i + 1 };
                    states[k] as f64 + states[k + 8] as f64
                })
            })
            .collect();
        let physical = p.physical_error(&x);
        std::fs::write(output.join("audit.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "compiler_variant":if folded {"test-only-redundant-barrier-fold"} else {"unmodified-current-control"}, "input_blake3":meta["blake3"],
            "archived_state_exact":capture::read(engine, solver.state_buffer()) == archived_state,
            "archived_report_exact":status == archived_report,
            "archived_publication_exact":publication == std::array::from_fn(|i| word(24+i)),
            "physical_error":physical,"physical_bound":1e-6,"exact_repeats":repeats,"acceptance":false
        })).unwrap()).unwrap();
        assert!(physical.is_finite() && physical <= 1e-6);
    }
    if std::env::var_os("MOLLA_NEWTON_SUCCESS_EXACT").is_some() {
        assert_eq!(word(8), 1);
        profile_replay_guess(
            word(8),
            word(9),
            captured_seed.is_some(),
            std::env::var_os("MOLLA_NEWTON_FAILURE_IGNORE_SEED").is_some(),
        );
        assert_eq!(meta["method"], solver.failure_method().name());
        assert_eq!(status, archived_report);
        assert_eq!(publication, std::array::from_fn(|i| word(24 + i)));
        assert_eq!(
            capture::read(&engine, solver.state_buffer()),
            archived_state
        );
        eprintln!("same-solve initial guess: exact archived report/publication/state");
    }
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_PIVOT_INSPECT_OUTPUT") {
        assert!(
            warm && conditioned && std::env::var_os("MOLLA_NEWTON_PIVOT_INSPECT_CONTACT").is_some()
        );
        assert_eq!(status[0], 8);
        assert_eq!(
            capture::read(&engine, solver.state_buffer()),
            archived_state
        );
        let scratch = values(&engine, solver.precision_local_buffer());
        let blocks = solver.logical_scratch_values(&engine, 0);
        let output = std::path::PathBuf::from(output);
        assert!(!output.exists(), "refusing to overwrite pivot inspection");
        let result = serde_json::json!({"acceptance": false, "report": status,
            "state_blake3": blake3::hash(&archived_state).to_hex().as_str(),
            "header": &scratch[8000..8014], "machine_and_other": &scratch[8040..8064],
            "invalid_contacts": p.contacts.iter().filter_map(|c| (scratch[2*(4160+54*c.index+53)] != 1.0).then_some(c.index)).collect::<Vec<_>>(),
            "invalid_local_blocks": p.contacts.iter().filter(|c| scratch[2*(4160+54*c.index+53)] != 1.0).map(|c| serde_json::json!({
                "contact": c.index, "block": &blocks[114*c.index..114*(c.index+1)],
                "cache": &scratch[2*(4160+54*c.index)..2*(4160+54*(c.index+1))]
            })).collect::<Vec<_>>()});
        std::fs::write(output, serde_json::to_vec_pretty(&result).unwrap()).unwrap();
        return;
    }
    let x: Vec<_> = p
        .contacts
        .iter()
        .flat_map(|contact| {
            (0..6).map(|i| {
                let k = 16 * contact.index + if i < 3 { i } else { i + 1 };
                states[k] as f64 + states[k + 8] as f64
            })
        })
        .collect();
    if linear_probe {
        let mut reference = Reference::new(&f, p.free.clone());
        if std::env::var_os("MOLLA_NEWTON_FAILURE_PREPARED_REFERENCE").is_some() {
            reference.use_prepared_response(&response_values);
        }
        for contact in &mut reference.contacts {
            contact.fischer = true;
        }
        let linear = Linearization::new(&reference, &x);
        let cpu_f = reference.residual(&x);
        let directions = solver.direction_buffer();
        let paired_blocks = solver.logical_scratch_values(&engine, 0);
        let blocks: Vec<f64> = paired_blocks
            .chunks_exact(114)
            .flat_map(|block| {
                block[..6].iter().map(|&v| v as f64).chain(
                    block[6..]
                        .chunks_exact(2)
                        .map(|pair| pair[0] as f64 + pair[1] as f64),
                )
            })
            .collect();
        let directions = values(&engine, directions);
        let basis = values(&engine, solver.basis_buffer());
        let d: Vec<_> = p
            .contacts
            .iter()
            .flat_map(|c| {
                (0..6).map(|i| {
                    directions[8 * c.index + if i < 3 { i } else { i + 1 }] as f64
                        + basis[16 * word(6) as usize * f.contacts.len().max(1)
                            + 16 * c.index
                            + 8
                            + if i < 3 { i } else { i + 1 }] as f64
                })
            })
            .collect();
        let gpu_f: Vec<_> = p
            .contacts
            .iter()
            .flat_map(|c| (0..6).map(|i| blocks[60 * c.index + i] as f64))
            .collect();
        let mut differences: Vec<_> = gpu_f
            .iter()
            .zip(&cpu_f)
            .enumerate()
            .map(|(i, (a, b))| (i, (a - b).abs()))
            .collect();
        differences.sort_by(|a, b| b.1.total_cmp(&a.1));
        let q = reference.rates(&x, true);
        for &(i, error) in differences.iter().take(5) {
            let contact = &reference.contacts[i / 6];
            let local = &x[6 * (i / 6)..6 * (i / 6) + 6];
            let (soil, relative) = Reference::velocities(contact, local, &q, true);
            eprintln!(
                "FB residual difference slot={} component={} error={error:e} gpu={:e} cpu={:e} state={local:?} normal={:?} soil={soil:?} relative={relative:?} rho={:e}",
                contact.index,
                i % 6,
                gpu_f[i],
                cpu_f[i],
                contact.normal,
                contact.rho
            );
        }
        let mut dq = vec![0.; n];
        for (contact, local) in p.contacts.iter().zip(d.chunks_exact(6)) {
            for i in 0..n {
                for axis in 0..3 {
                    dq[i] -= response_values[contact.index * n + i][axis] as f64 * local[axis];
                }
            }
        }
        let mut gpu_jd = Vec::new();
        let mut matrix_error = 0f64;
        for (slot, (contact, local)) in p.contacts.iter().zip(d.chunks_exact(6)).enumerate() {
            let base = contact.index * 60;
            let point = contact
                .rows
                .iter()
                .zip(&dq)
                .map(|(row, q)| *row * *q)
                .sum::<DVec3>();
            for i in 0..6 {
                let mut value = 0.;
                for j in 0..6 {
                    let actual = blocks[base + 6 + 6 * i + j] as f64;
                    matrix_error = matrix_error.max(
                        (actual - linear.direct[slot][i][j]).abs()
                            / (1. + linear.direct[slot][i][j].abs()),
                    );
                    value += actual * local[j];
                }
                for j in 0..3 {
                    value += blocks[base + 42 + 3 * i + j] as f64 * point[j];
                }
                gpu_jd.push(value);
            }
        }
        let relative = |j: &[f64], f: &[f64]| {
            norm(&j.iter().zip(f).map(|(j, f)| j + f).collect::<Vec<_>>()) / norm(f)
        };
        let pre = reference.preconditioner(&x);
        let rhs: Vec<_> = cpu_f.iter().map(|v| -*v).collect();
        let (cpu_y, used, error) = gmres(|v| linear.product(&apply_blocks(&pre, v)), &rhs, 64);
        let cpu_d = apply_blocks(&pre, &cpu_y);
        let mut rounded = Linearization::new(&reference, &x);
        for block in &mut rounded.direct {
            for row in block {
                for value in row {
                    *value = (*value as f32) as f64;
                }
            }
        }
        for block in &mut rounded.machine {
            for row in block {
                for value in row {
                    *value = (*value as f32) as f64;
                }
            }
        }
        let (rounded_y, rounded_used, rounded_error) =
            gmres(|v| rounded.product(&apply_blocks(&pre, v)), &rhs, 64);
        let rounded_solution = apply_blocks(&pre, &rounded_y);
        let reference_product = linear.product(&d);
        let rounded_product = rounded.product(&d);
        let action_difference = |a: &[f64], b: &[f64]| {
            norm(&a.iter().zip(b).map(|(a, b)| a - b).collect::<Vec<_>>()) / norm(&cpu_f)
        };
        eprintln!(
            "FB coefficient rounding control iterations={rounded_used} rounded_relative={rounded_error:e} reference_relative={:e} direction_rounding_action={:e} evaluation_action={:e}",
            relative(&linear.product(&rounded_solution), &cpu_f),
            action_difference(&rounded_product, &reference_product),
            action_difference(&gpu_jd, &rounded_product)
        );
        let mut prepared_reference = Reference::new(&f, p.free.clone());
        prepared_reference.use_prepared_response(&response_values);
        let mut assembled = Linearization::new(&prepared_reference, &x);
        let mut assembled_pre = Vec::new();
        for (slot, contact) in reference.contacts.iter().enumerate() {
            let base = contact.index * 60;
            for i in 0..6 {
                for j in 0..6 {
                    assembled.direct[slot][i][j] = blocks[base + 6 + 6 * i + j] as f64;
                }
                for j in 0..3 {
                    assembled.machine[slot][i][j] = blocks[base + 42 + 3 * i + j] as f64;
                }
            }
            let mut matrix = assembled.direct[slot];
            for j in 0..3 {
                let point = contact
                    .rows
                    .iter()
                    .enumerate()
                    .map(|(dof, row)| *row * response_values[contact.index * n + dof][j] as f64)
                    .sum::<DVec3>();
                for i in 0..6 {
                    for axis in 0..3 {
                        matrix[i][j] -= assembled.machine[slot][i][axis] * point[axis];
                    }
                }
            }
            let mut inverse = [[0.; 6]; 6];
            for j in 0..6 {
                let mut rhs = vec![0.; 6];
                rhs[j] = 1.;
                if let Some(column) = solve_dense(matrix.into_iter().flatten().collect(), rhs) {
                    for i in 0..6 {
                        inverse[i][j] = column[i];
                    }
                } else {
                    inverse = std::array::from_fn(|i| {
                        std::array::from_fn(|j| if i == j { 1. } else { 0. })
                    });
                    break;
                }
            }
            assembled_pre.push(inverse);
        }
        let rhs: Vec<_> = gpu_f.iter().map(|v| -*v).collect();
        let (assembled_y, assembled_used, assembled_error) = gmres(
            |v| assembled.product(&apply_blocks(&assembled_pre, v)),
            &rhs,
            64,
        );
        let assembled_d = apply_blocks(&assembled_pre, &assembled_y);
        let inverse = solver.logical_scratch_values(&engine, 1);
        let gpu_pre: Vec<[[f64; 6]; 6]> = reference
            .contacts
            .iter()
            .map(|contact| {
                std::array::from_fn(|i| {
                    std::array::from_fn(|j| inverse[36 * contact.index + 6 * i + j] as f64)
                })
            })
            .collect();
        let (gpu_pre_y, gpu_pre_used, gpu_pre_error) =
            gmres(|v| assembled.product(&apply_blocks(&gpu_pre, v)), &rhs, 64);
        let gpu_pre_d = apply_blocks(&gpu_pre, &gpu_pre_y);
        eprintln!(
            "FB GPU-preconditioner f64 control iterations={gpu_pre_used} relative={gpu_pre_error:e} reference_relative={:e} direction={:e}",
            relative(&linear.product(&gpu_pre_d), &cpu_f),
            norm(&gpu_pre_d)
        );
        let rounded_d: Vec<_> = assembled_d.iter().map(|v| (*v as f32) as f64).collect();
        eprintln!(
            "FB assembled-coefficient f64 control iterations={assembled_used} relative={assembled_error:e} direction={:e} rounded_direction_relative={:e}",
            norm(&assembled_d),
            relative(&assembled.product(&rounded_d), &gpu_f)
        );
        eprintln!(
            "FB linear probe status={status:?} reported_relative={:e} assembled_relative={:e} reference_relative={:e} residual_difference={:e} direct_difference={matrix_error:e} GPU_direction={:e} CPU_iterations={used} CPU_relative={error:e} CPU_direction={:e} direction_difference={:e}",
            f32::from_bits(status[5]),
            relative(&gpu_jd, &gpu_f),
            relative(&linear.product(&d), &cpu_f),
            norm(
                &gpu_f
                    .iter()
                    .zip(&cpu_f)
                    .map(|(a, b)| a - b)
                    .collect::<Vec<_>>()
            ) / norm(&cpu_f),
            norm(&d),
            norm(&cpu_d),
            norm(&d.iter().zip(&cpu_d).map(|(a, b)| a - b).collect::<Vec<_>>())
                / (1. + norm(&cpu_d))
        );
        for (label, direction) in [
            ("gpu", &d),
            ("reference", &cpu_d),
            ("assembled", &assembled_d),
        ] {
            let base = norm(&cpu_f);
            let mut best = (base, 0.0, reference.physical_error(&x));
            for attempt in 0..16 {
                let alpha = 2.0_f64.powi(-attempt);
                let trial: Vec<_> = x
                    .iter()
                    .zip(direction)
                    .map(|(x, d)| x + alpha * d)
                    .collect();
                let merit = norm(&reference.residual(&trial));
                if merit < best.0 {
                    best = (merit, alpha, reference.physical_error(&trial));
                }
            }
            eprintln!(
                "FB bounded direction descent label={label} base={base:e} best={:e} alpha={:e} physical={:e}",
                best.0, best.1, best.2
            );
        }
        assert_eq!(status[0], 0);
        assert!(relative(&gpu_jd, &gpu_f) <= 1e-5);
        assert!(relative(&linear.product(&d), &cpu_f) <= 1e-5);
        return;
    }
    eprintln!(
        "first-failure Newton packet dofs={n} slots={c} active={} original={archived_report:?} replay={status:?} publication={publication:?} physical={:e} residual={:e}",
        p.contacts.len(),
        p.physical_error(&x),
        norm(&p.residual(&x))
    );
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    if let Some(output) = std::env::var_os("MOLLA_NEWTON_PHASE_CLOCK") {
        use crate::soil_newton_precise_tests::shader_clock::{TRACE_BYTES, phase_names};
        assert!(std::env::var_os("MOLLA_NEWTON_SUCCESS_EXACT").is_some());
        assert!(p.physical_error(&x) <= 1e-6);
        let bytes = capture::read(&engine, solver.clock_buffer());
        assert_eq!(bytes.len() as u64, TRACE_BYTES);
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        let phases: Vec<_> = phase_names()
            .iter()
            .zip(words.chunks_exact(4))
            .map(|(name, v)| {
                assert_eq!(v[3], 0);
                let ticks = u64::from(v[0]) | (u64::from(v[1]) << 32);
                assert!(v[2] == 0 || ticks > 0);
                serde_json::json!({"phase": name, "calls": v[2], "inclusive_subgroup_ticks": ticks})
            })
            .collect();
        assert!(words[2] > 0 && words[6] > 0 && words[10] > 0 && words[14] > 0);
        let output = std::path::PathBuf::from(output);
        assert!(!output.exists());
        std::fs::write(output, serde_json::to_vec_pretty(&serde_json::json!({
            "acceptance": false, "scope": "inclusive lane-zero subgroup ticks; no additive attribution or elapsed-time conversion",
            "report": status, "physical_error": p.physical_error(&x), "trace_bytes": TRACE_BYTES, "phases": phases
        })).unwrap()).unwrap();
    }
    let expected = std::env::var("MOLLA_NEWTON_FAILURE_EXPECT_FAULT")
        .map(|v| v.parse::<u32>().unwrap())
        .unwrap_or(0);
    assert_eq!(status[0], expected);
    if expected == 0 {
        assert_eq!(publication[0], 0);
        super::publication::verify_outputs(
            &engine,
            solver.outputs_buffer(),
            solver.rates_buffer(),
            &f,
            &p,
        );
        if let Some(output) = std::env::var_os("MOLLA_NEWTON_FAILURE_OUTPUT") {
            let output = std::path::PathBuf::from(output).join(meta["blake3"].as_str().unwrap());
            assert!(!output.exists(), "refusing to overwrite replay output");
            std::fs::create_dir_all(&output).unwrap();
            for (name, buffer) in [
                ("states", solver.state_buffer()),
                ("rates", solver.rates_buffer()),
                ("outputs", solver.outputs_buffer()),
                ("status", solver.status_buffer()),
                ("publication", solver.publication_status_buffer()),
            ] {
                std::fs::write(
                    output.join(format!("{name}.bin")),
                    capture::read(&engine, buffer),
                )
                .unwrap();
            }
            std::fs::write(
                output.join("mode.txt"),
                if std::env::var_os("MOLLA_NEWTON_DELTA_CACHE").is_some() {format!("protected-stalled-delta-cache-test-width-{}-contact-constants-{}",crate::soil_newton_gpu::execution_lanes(true),std::env::var_os("MOLLA_NEWTON_STATIC_CONTACT_CACHE").is_some())} else {std::env::var("MOLLA_NEWTON_WIDE_LANES").ok().map(|lanes|format!("protected-stalled-test-width-{lanes}")).unwrap_or_else(||precise.as_deref().unwrap_or(
                    if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedSoaScratch256 {
                        "protected-stalled-stream-block-stagnant-branch-grouped-eight-parallel-indices-soa-scratch-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedParallelIndices256 {
                        "protected-stalled-stream-block-stagnant-branch-grouped-eight-parallel-indices-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedGroupedEight256 {
                        "protected-stalled-stream-block-stagnant-branch-grouped-eight-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedGroupedSum256 {
                        "protected-stalled-stream-block-stagnant-branch-grouped-sum-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedStagnantBranch256 {
                        "protected-stalled-stream-block-stagnant-branch-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedStreamBlock256 {
                        "protected-stalled-stream-block-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedSharedCore256 {
                        "protected-stalled-shared-core-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedProjectionCache256 {
                        "protected-stalled-projection-cache-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedRetainedLocal256 {
                        "protected-stalled-retained-local-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedPublicationProducts256 {
                        "protected-stalled-schur-norm-publication-products-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedActivePublication256 {
                        "protected-stalled-schur-norm-active-publication-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedNorm256 {
                        "protected-stalled-schur-norm-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedSchur256 {
                        "protected-stalled-schur-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedVolatile256 {
                        "protected-stalled-volatile-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedParallel256 {
                        "protected-stalled-parallel-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedCached256 {
                        "protected-stalled-cached-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedWide256 {
                        "protected-stalled-runtime-256"
                    } else if solver.failure_method() == crate::soil_newton_failure::NewtonFailureMethod::FischerConditionedProtectedStalled {
                        "protected-stalled-runtime"
                    } else {
                        "wgsl"
                    },
                ).to_owned())},
            )
            .unwrap();
            if let Some(profile) = &replay_profile {
                let physical_error = p.physical_error(&x);
                assert!(physical_error.is_finite() && physical_error <= 1e-6);
                let initial_guess =
                    profile_replay_guess(word(8), word(9), captured_seed.is_some(), false);
                let manifest = serde_json::json!({
                    "acceptance": false,
                    "scope": "fixed captured problem and original initial guess; not an evolving trajectory or performance comparison",
                    "profile": profile,
                    "experimental_policy": match std::env::var("MOLLA_NEWTON_SMOOTHED_RELEASE").as_deref() { Ok("finish") => "smoothed-stage-completion", Ok("finish-trace") => "smoothed-stage-finish-trace", Ok("trace") => "smoothed-stall-release-trace", Ok("settle") => "smoothed-stall-settle", Ok("stages") => "smoothed-stage-release", Ok("1") => "smoothed-stall-release", _ if std::env::var_os("MOLLA_NEWTON_FIRST_RELEASE").is_some() => "guarded-first-release", _ => "none" },
                    "backend": solver.failure_method().name(),
                    "input_blake3": blake3::hash(&bytes).to_hex().to_string(),
                    "initial_guess": initial_guess,
                    "seed_blake3": captured_seed.as_ref().map(|seed| blake3::hash(seed).to_hex().to_string()),
                    "dofs": n, "contacts": c, "active_contacts": p.contacts.len(),
                    "active_slots": p.contacts.iter().map(|contact| contact.index).collect::<Vec<_>>(),
                    "newton_limit": word(5), "krylov_limit": word(6),
                    "physical_error": physical_error, "physical_limit": 1e-6,
                    "exact_repeats": repeats, "report": status, "publication": publication,
                    "matches_archived_state": capture::read(&engine, solver.state_buffer()) == archived_state,
                });
                std::fs::write(
                    output.join("profile.json"),
                    serde_json::to_vec_pretty(&manifest).unwrap(),
                )
                .unwrap();
            }
        }
    } else {
        assert_eq!(&status[..4], &archived_report[..4]);
        assert_eq!(
            capture::read(&engine, solver.state_buffer()),
            archived_state
        );
    }
    if std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_CONTROL").is_some() {
        let mut alternate = Reference::new(&f, p.free.clone());
        if std::env::var_os("MOLLA_NEWTON_FAILURE_PREPARED_REFERENCE").is_some() {
            alternate.use_prepared_response(&response_values);
        }
        if std::env::var_os("MOLLA_NEWTON_FAILURE_FISCHER").is_some() {
            for contact in &mut alternate.contacts {
                contact.fischer = true;
            }
        }
        let p = &alternate;
        let q = p.rates(&x, true);
        for (contact, local) in p.contacts.iter().zip(x.chunks_exact(6)) {
            let (soil, relative) = Reference::velocities(contact, local, &q, true);
            let k = xyz(local);
            let w = xyz(&local[3..]);
            let machine = soil - relative;
            let lambda = contact.normal.dot(k);
            let separation = contact.normal.dot(relative) + contact.gap;
            let ps = 1. + k.length() + w.length();
            let vs = 1. + soil.length() + machine.length();
            let ns = 1f64
                .max(contact.normal.dot(soil).abs())
                .max(contact.normal.dot(machine).abs())
                .max(contact.gap);
            if separation < -1e-6 * ns
                || (lambda * separation).abs() > 1e-6 * ps * vs
                || (0..3).any(|axis| {
                    contact.wall[axis] != 0.
                        && (contact.wall[axis] * soil[axis] < -1e-6 * vs
                            || (w[axis] * soil[axis]).abs() > 1e-6 * ps * vs)
                })
            {
                eprintln!(
                    "failed physical slot={} mass={} normal={:?} wall={:?} gap={} k={k:?} w={w:?} soil={soil:?} relative={relative:?} separation={separation:e}",
                    contact.index, contact.mass, contact.normal, contact.wall, contact.gap
                );
            }
        }
        let mut control_seed = x.clone();
        if std::env::var_os("MOLLA_NEWTON_FAILURE_CPU_COLD").is_some() {
            control_seed.fill(0.);
        }
        if std::env::var_os("MOLLA_NEWTON_FAILURE_WALL_RELEASE").is_some() {
            let mut selected = None;
            let mut worst = 0.;
            for (slot, (contact, local)) in p.contacts.iter().zip(x.chunks_exact(6)).enumerate() {
                let (soil, _) = Reference::velocities(contact, local, &q, true);
                for axis in 0..3 {
                    let w = local[3 + axis];
                    let mut k = xyz(local);
                    k[axis] += w;
                    let lambda = contact.normal.dot(k);
                    let tangent = k - contact.normal * lambda;
                    let score = w * soil[axis];
                    if contact.wall[axis] * soil[axis] > 0.
                        && contact.wall[axis] * w > 0.
                        && lambda >= 0.
                        && tangent.length() <= contact.mu * lambda
                        && score > worst
                    {
                        selected = Some((slot, axis));
                        worst = score;
                    }
                }
            }
            let (slot, axis) = selected.expect("releasable wall constraint");
            control_seed[6 * slot + axis] += control_seed[6 * slot + 3 + axis];
            control_seed[6 * slot + 3 + axis] = 0.;
            eprintln!(
                "wall-release diagnostic slot={} axis={axis} score={worst:e}",
                p.contacts[slot].index
            );
            let mut seeds =
                vec![crate::soil_newton_operator_gpu::SoilNewtonState::default(); c.max(1)];
            for (contact, local) in p.contacts.iter().zip(control_seed.chunks_exact(6)) {
                seeds[contact.index] = state(local);
            }
            engine.queue().unwrap().write_buffer(
                solver.state_buffer(),
                0,
                bytemuck::cast_slice(&seeds),
            );
            engine.queue().unwrap().write_buffer(
                solver.status_buffer(),
                0,
                bytemuck::cast_slice(&[0u32; 8]),
            );
            let released = run(&engine, &solver, false);
            eprintln!("wall-release GPU control={released:?}");
            assert_eq!(released[0], 0);
            super::publication::verify_outputs(
                &engine,
                solver.outputs_buffer(),
                solver.rates_buffer(),
                &f,
                &p,
            );
        }
        let (cpu, steps, inner) = solve(&p, control_seed).unwrap();
        eprintln!(
            "CPU control from failed candidate steps={steps} inner={inner} physical={:e} residual={:e}",
            p.physical_error(&cpu),
            norm(&p.residual(&cpu))
        );
        verify_solution(&f, &p, &cpu);
        let mut seeds = vec![crate::soil_newton_operator_gpu::SoilNewtonState::default(); c.max(1)];
        for (contact, local) in p.contacts.iter().zip(cpu.chunks_exact(6)) {
            seeds[contact.index] = state(local);
        }
        engine.queue().unwrap().write_buffer(
            solver.state_buffer(),
            0,
            bytemuck::cast_slice(&seeds),
        );
        engine.queue().unwrap().write_buffer(
            solver.status_buffer(),
            0,
            bytemuck::cast_slice(&[0u32; 8]),
        );
        let seeded = run(&engine, &solver, false);
        eprintln!("CPU-seeded GPU control={seeded:?}");
        assert_eq!(seeded[0], 0);
        super::publication::verify_outputs(
            &engine,
            solver.outputs_buffer(),
            solver.rates_buffer(),
            &f,
            &p,
        );
    }
}

#[test]
#[ignore = "requires GPU; bounded one-step wide diagnostic, not resident acceptance"]
fn resident_newton_gpu_wide_trace() {
    let f = wide_fixture();
    let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let engine = Engine::new_gpu().unwrap();
    let x = vec![0.0; 6 * p.contacts.len()];
    let (op, _, _) = create_operator(&engine, &f, &p, &x, &x);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver = create_capacity(
        &engine,
        &op,
        &p,
        SoilNewtonCapacity {
            newton_iterations: 1,
            ..capacity(f.free.len(), f.contacts.len())
        },
    );
    for step in 0..12 {
        let start = std::time::Instant::now();
        engine
            .queue()
            .unwrap()
            .write_buffer(solver.status_buffer(), 0, bytemuck::bytes_of(&0u32));
        let status = run(&engine, &solver, step == 0);
        eprintln!(
            "wide diagnostic step={step} ms={} status={:?} residual={:e} physical={:e}",
            start.elapsed().as_secs_f64() * 1000.0,
            &status[..4],
            f32::from_bits(status[4]),
            f32::from_bits(status[5])
        );
        assert!(status[0] == 0 || status[0] == 8);
        if status[0] == 0 {
            return;
        }
    }
    panic!("wide diagnostic did not converge in 12 single-step calls");
}

#[test]
#[ignore = "requires GPU; sparse resident state, warm starts, limits and fault recovery"]
fn resident_newton_gpu_reuse_and_faults() {
    reuse_and_faults(NewtonVariant::Paired);
}

#[test]
#[ignore = "requires GPU; sorted-pair sparse reuse, poisoning and fault recovery"]
fn resident_newton_cooperative_pairs_reuse_and_faults() {
    reuse_and_faults(NewtonVariant::Cooperative);
}

fn reuse_and_faults(arithmetic: NewtonVariant) {
    let engine = replay_engine();
    reuse_and_faults_on(&engine, arithmetic);
}

fn reuse_and_faults_on(engine: &Engine, arithmetic: NewtonVariant) {
    let f = spread_contacts(&articulated(65), 4096, 63);
    let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let x = vec![0.0; 6 * p.contacts.len()];
    let (op, _, _) = create_operator(&engine, &f, &p, &x, &x);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver = create_capacity_arithmetic(
        &engine,
        &op,
        &p,
        capacity(p.free.len(), f.contacts.len()),
        arithmetic,
    );
    let first = check(&engine, &solver, &f, &p);
    let status = run(&engine, &solver, false);
    assert_eq!(&status[..2], &[0, 0]);
    for buffer in solver.scratch_buffers() {
        engine.queue().unwrap().write_buffer(
            buffer,
            0,
            bytemuck::cast_slice(&vec![f32::NAN; buffer.size() as usize / 4]),
        );
    }
    assert_eq!(first, check(&engine, &solver, &f, &p));
    engine
        .queue()
        .unwrap()
        .write_buffer(op.status_buffer(), 0, bytemuck::bytes_of(&1u32));
    assert_eq!(run(&engine, &solver, true)[0], 1);
    assert!(
        capture::read(&engine, solver.outputs_buffer())
            .iter()
            .all(|&b| b == 0)
    );
    assert!(
        capture::read(&engine, solver.rates_buffer())
            .iter()
            .all(|&b| b == 0)
    );
    assert_eq!(encode(&engine, &op, true), 0);
    assert_eq!(run(&engine, &solver, false)[0], 1);
    assert_eq!(first, check(&engine, &solver, &f, &p));
    let contacts = op.point_inputs()[2];
    let original = capture::read(&engine, contacts);
    engine
        .queue()
        .unwrap()
        .write_buffer(contacts, 0, bytemuck::bytes_of(&f32::NAN));
    assert_eq!(run(&engine, &solver, true)[0], 1);
    engine.queue().unwrap().write_buffer(contacts, 0, &original);
    assert_eq!(first, check(&engine, &solver, &f, &p));
    let bounded = create_capacity_arithmetic(
        &engine,
        &op,
        &p,
        SoilNewtonCapacity {
            newton_iterations: 1,
            ..capacity(p.free.len(), f.contacts.len())
        },
        arithmetic,
    );
    let status = run(&engine, &bounded, true);
    assert_eq!(&status[..2], &[8, 1]);
    assert_eq!(run(&engine, &bounded, false), status);
    assert!(solver.shares_program_with(&bounded));
    assert_ne!(solver.state_buffer(), bounded.state_buffer());
    assert_ne!(solver.status_buffer(), bounded.status_buffer());
    assert_eq!(first, check(&engine, &solver, &f, &p));
}

#[test]
#[ignore = "requires GPU; frozen later wide-state operator and linear diagnosis"]
fn resident_newton_gpu_wide_linearization() {
    let f = wide_fixture();
    let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let engine = Engine::new_gpu().unwrap();
    let zero = vec![0.0; 6 * p.contacts.len()];
    let v: Vec<_> = (0..zero.len())
        .map(|i| ((i as f64).sin() as f32) as f64)
        .collect();
    let (op, states, _) = create_operator(&engine, &f, &p, &zero, &v);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver = create_capacity(
        &engine,
        &op,
        &p,
        SoilNewtonCapacity {
            newton_iterations: 1,
            ..capacity(f.free.len(), f.contacts.len())
        },
    );
    for step in 0..4 {
        engine
            .queue()
            .unwrap()
            .write_buffer(solver.status_buffer(), 0, bytemuck::bytes_of(&0u32));
        assert_eq!(run(&engine, &solver, step == 0)[0], 8);
    }
    let bytes = capture::read(&engine, solver.state_buffer());
    if let Some(path) = std::env::var_os("MOLLA_NEWTON_STATE_DUMP") {
        use std::io::Write;
        std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(path)
            .unwrap()
            .write_all(&bytes)
            .unwrap();
    }
    engine.queue().unwrap().write_buffer(&states, 0, &bytes);
    let states = values(&engine, solver.state_buffer());
    let x: Vec<_> = p
        .contacts
        .iter()
        .flat_map(|c| {
            (0..6).map(|i| {
                let k = 16 * c.index + if i < 3 { i } else { i + 1 };
                states[k] as f64 + states[k + 8] as f64
            })
        })
        .collect();
    assert_eq!(encode(&engine, &op, true), 0);
    compare_operator(&engine, &op, &p, &x, &v, f.contacts.len());
    let linear = Linearization::new(&p, &x);
    let pre = p.preconditioner(&x);
    let rhs: Vec<_> = p.residual(&x).iter().map(|v| -v).collect();
    let (y, used, error) = gmres(|v| linear.product(&apply_blocks(&pre, v)), &rhs, 64);
    eprintln!(
        "later wide reference linear: iterations={used} relative={error:e} direction_norm={:e}",
        norm(&apply_blocks(&pre, &y))
    );
    let inner = super::krylov::create_krylov(
        &engine,
        &op,
        crate::soil_newton_krylov_gpu::SoilNewtonKrylovCapacity {
            dofs: 32,
            contacts: 7,
            iterations: 64,
            relative_tolerance: 1e-5,
            max_owned_bytes: u64::MAX,
        },
    );
    let status = super::krylov::run(&engine, &inner, true);
    eprintln!(
        "later wide standalone GPU linear: status={:?} relative={:e}",
        &status[..4],
        f32::from_bits(status[4])
    );
    let (answer, steps, products) = solve(&p, x).unwrap();
    verify_solution(&f, &p, &answer);
    eprintln!("later wide f64 nonlinear solved: steps={steps} products={products}");
}

#[test]
#[ignore = "requires GPU; first-step reference and GPU branch diagnostics"]
fn resident_newton_gpu_initial_direction() {
    let engine = Engine::new_gpu().unwrap();
    for f in [articulated(7), wide_fixture()] {
        let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
        let zero = vec![0.0; 6 * p.contacts.len()];
        let pre = p.preconditioner(&zero);
        let linear = Linearization::new(&p, &zero);
        let rhs: Vec<_> = p.residual(&zero).iter().map(|v| -v).collect();
        let (y, used, error) = gmres(|v| linear.product(&apply_blocks(&pre, v)), &rhs, 64);
        let cpu = apply_blocks(&pre, &y);
        let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
        assert_eq!(encode(&engine, &op, true), 0);
        let solver = create_capacity(
            &engine,
            &op,
            &p,
            SoilNewtonCapacity {
                newton_iterations: 1,
                ..capacity(f.free.len(), f.contacts.len())
            },
        );
        let status = run(&engine, &solver, true);
        assert_eq!(status[0], 8);
        let raw = values(&engine, solver.state_buffer());
        let gpu: Vec<_> = p
            .contacts
            .iter()
            .flat_map(|c| {
                (0..6).map(|i| {
                    let k = 16 * c.index + if i < 3 { i } else { i + 1 };
                    raw[k] as f64 + raw[k + 8] as f64
                })
            })
            .collect();
        eprintln!(
            "initial direction dofs={} CPU iterations={used} relative={error:e} GPU iterations={} error_norm={:e}",
            p.free.len(),
            status[2],
            norm(&cpu.iter().zip(&gpu).map(|(a, b)| a - b).collect::<Vec<_>>())
        );
        for (slot, c) in p.contacts.iter().enumerate() {
            for (label, x) in [("CPU", &cpu), ("GPU", &gpu)] {
                let q = p.rates(x, true);
                let state = &x[6 * slot..6 * (slot + 1)];
                let (soil, relative) = Reference::velocities(c, state, &q, true);
                let lambda = c.normal.dot(xyz(state));
                let tangent = xyz(state) - lambda * c.normal;
                let slip = relative - c.normal * c.normal.dot(relative);
                let projection = (tangent - c.rho * slip).length() - c.mu * lambda.max(0.0);
                eprintln!(
                    "dofs={} contact={slot} {label}: lambda={lambda:e} disk_excess={projection:e} normal_branch={:e} wall_y={:e}",
                    p.free.len(),
                    lambda - c.rho * (c.normal.dot(relative) + c.gap),
                    state[4] - c.mass * soil.y
                );
            }
        }
    }
}

#[test]
#[ignore = "requires GPU; empty certified solve and invalid publication mass"]
fn resident_newton_gpu_empty() {
    empty_and_mass(NewtonVariant::Paired);
}

#[test]
#[ignore = "requires GPU; sorted-pair empty solve and invalid mass rejection"]
fn resident_newton_cooperative_pairs_empty() {
    empty_and_mass(NewtonVariant::Cooperative);
}

fn empty_and_mass(arithmetic: NewtonVariant) {
    let engine = replay_engine();
    empty_and_mass_on(&engine, arithmetic);
}

fn empty_and_mass_on(engine: &Engine, arithmetic: NewtonVariant) {
    let mut f = pair(false, 0.0);
    f.contacts.clear();
    f.rows.clear();
    let mut p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let (op, _, _) = create_operator(&engine, &f, &p, &[], &[]);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver =
        create_capacity_arithmetic(&engine, &op, &p, capacity(p.free.len(), 0), arithmetic);
    assert!(check(&engine, &solver, &f, &p).is_empty());
    assert_eq!(&run(&engine, &solver, false)[..2], &[0, 0]);
    p.mass[0] = f32::NAN;
    let invalid =
        create_capacity_arithmetic(&engine, &op, &p, capacity(p.free.len(), 0), arithmetic);
    assert_eq!(run(&engine, &invalid, true)[0], 1);
    assert!(
        capture::read(&engine, invalid.outputs_buffer())
            .iter()
            .all(|&b| b == 0)
    );
    assert!(
        capture::read(&engine, invalid.rates_buffer())
            .iter()
            .all(|&b| b == 0)
    );
}
