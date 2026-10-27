use super::*;
use crate::soil_newton_gpu::{ExperimentalSoilNewtonGpu, SoilNewtonCapacity, SoilNewtonInputs};
use crate::soil_newton_prepare_gpu::{ExperimentalSoilNewtonPrepareGpu, SoilNewtonPrepareInputs};

#[path = "soil_newton_prepare_capture_tests.rs"]
mod captured;

fn preparation_engine(features: wgpu::Features) -> Engine {
    Engine::new_gpu_with_features(
        features
            | if cfg!(feature = "strict-rounding-newton-preparation") {
                wgpu::Features::PASSTHROUGH_SHADERS
            } else {
                wgpu::Features::empty()
            },
    )
    .unwrap()
}

struct Prepared {
    driver: ExperimentalSoilNewtonPrepareGpu,
    mass: wgpu::Buffer,
    rows: wgpu::Buffer,
    contacts: wgpu::Buffer,
    free: wgpu::Buffer,
    upstream: wgpu::Buffer,
}

fn create(engine: &Engine, f: &Fixture, p: &Reference) -> Prepared {
    create_backend(engine, f, p, None)
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
#[ignore = "requires GPU; preparation program reuse with isolated mutable owners"]
fn newton_preparation_program_cache_ownership() {
    let engine = preparation_engine(wgpu::Features::empty());
    let f = pair(false, 0.0);
    let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let a = create(&engine, &f, &p);
    let b = create(&engine, &f, &p);
    assert!(a.driver.shares_program(&b.driver));
    let variant = create_backend(&engine, &f, &p, Some(9));
    assert!(!a.driver.shares_program(&variant.driver));
    for owner in [&a, &b, &variant] {
        assert_eq!(run(&engine, owner, None, true).0[0], 0);
        verify_response(&engine, owner, &f, &p);
    }
    let response = capture::read(&engine, b.driver.response_buffer());
    engine
        .queue()
        .unwrap()
        .write_buffer(&a.mass, 0, bytemuck::bytes_of(&-1.0f32));
    assert_ne!(run(&engine, &a, None, true).0[0], 0);
    assert_eq!(run(&engine, &b, None, true).0[0], 0);
    assert_eq!(response, capture::read(&engine, b.driver.response_buffer()));
    engine.clear_gpu_program_cache();
    let c = create(&engine, &f, &p);
    assert!(!a.driver.shares_program(&c.driver));
    engine
        .queue()
        .unwrap()
        .write_buffer(&a.mass, 0, bytemuck::cast_slice(&f.mass));
    for owner in [&a, &b, &c] {
        assert_eq!(run(&engine, owner, None, true).0[0], 0);
        assert_eq!(
            response,
            capture::read(&engine, owner.driver.response_buffer())
        );
    }
    let other = preparation_engine(wgpu::Features::empty());
    let d = create(&other, &f, &p);
    assert!(!c.driver.shares_program(&d.driver));
    assert_eq!(run(&other, &d, None, true).0[0], 0);
    assert_eq!(response, capture::read(&other, d.driver.response_buffer()));
}

fn create_backend(engine: &Engine, f: &Fixture, p: &Reference, reference: Option<u32>) -> Prepared {
    let mass = storage(engine, &f.mass);
    let rows = buffer(engine, &f.rows, 16 * f.free.len() as u64);
    let contacts = buffer(engine, &f.contacts, 64);
    let pairs: Vec<_> = p
        .free
        .iter()
        .map(|q| [*q as f32, (*q - (*q as f32) as f64) as f32])
        .collect();
    let free = storage(engine, &pairs);
    let upstream = storage(engine, &[0u32]);
    let inputs = || SoilNewtonPrepareInputs {
        mass: &mass,
        point_rows: &rows,
        contacts: &contacts,
        upstream_status: &upstream,
    };
    let n = f.free.len() as u32;
    let c = f.contacts.len() as u32;
    let bytes = ExperimentalSoilNewtonPrepareGpu::required_bytes(n, c).unwrap();
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(ExperimentalSoilNewtonPrepareGpu::new(engine, inputs(), n, c, bytes - 1).is_err());
    assert!(
        ExperimentalSoilNewtonPrepareGpu::new(
            engine,
            SoilNewtonPrepareInputs {
                point_rows: &upstream,
                ..inputs()
            },
            n,
            c,
            bytes
        )
        .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    let driver = match reference {
        Some(0) => ExperimentalSoilNewtonPrepareGpu::new_reference(engine, inputs(), n, c, bytes),
        Some(2) => ExperimentalSoilNewtonPrepareGpu::new_zero_terms(engine, inputs(), n, c, bytes),
        Some(3) => {
            ExperimentalSoilNewtonPrepareGpu::new_sparse_residual(engine, inputs(), n, c, bytes)
        }
        Some(7) => {
            ExperimentalSoilNewtonPrepareGpu::new_certificate_mask(engine, inputs(), n, c, bytes)
        }
        Some(8) => {
            ExperimentalSoilNewtonPrepareGpu::new_axis_parallel(engine, inputs(), n, c, bytes)
        }
        Some(9) => ExperimentalSoilNewtonPrepareGpu::new_axis_shared(engine, inputs(), n, c, bytes),
        Some(15) => {
            ExperimentalSoilNewtonPrepareGpu::new_row_parallel(engine, inputs(), n, c, bytes)
        }
        Some(17) => ExperimentalSoilNewtonPrepareGpu::new_row_factor(engine, inputs(), n, c, bytes),
        Some(19) => ExperimentalSoilNewtonPrepareGpu::new_row_sized(engine, inputs(), n, c, bytes),
        #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
        Some(11) => {
            ExperimentalSoilNewtonPrepareGpu::new_axis_rounding::<0>(engine, inputs(), n, c, bytes)
        }
        #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
        Some(12) => {
            ExperimentalSoilNewtonPrepareGpu::new_axis_rounding::<1>(engine, inputs(), n, c, bytes)
        }
        #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
        Some(13) => {
            ExperimentalSoilNewtonPrepareGpu::new_axis_rounding::<2>(engine, inputs(), n, c, bytes)
        }
        None => ExperimentalSoilNewtonPrepareGpu::new(engine, inputs(), n, c, bytes),
        _ => panic!("unknown preparation reference"),
    }
    .unwrap();
    assert_eq!(driver.owned_buffer_bytes(), bytes);
    Prepared {
        driver,
        mass,
        rows,
        contacts,
        free,
        upstream,
    }
}

fn solver(engine: &Engine, a: &Prepared, f: &Fixture) -> ExperimentalSoilNewtonGpu {
    ExperimentalSoilNewtonGpu::new_stagnation_pairs(
        engine,
        SoilNewtonInputs {
            mass: &a.mass,
            free_rates: &a.free,
            point_rows: &a.rows,
            point_response: a.driver.response_buffer(),
            contacts: &a.contacts,
            upstream_status: a.driver.status_buffer(),
        },
        SoilNewtonCapacity {
            dofs: f.free.len() as u32,
            contacts: f.contacts.len() as u32,
            newton_iterations: 64,
            krylov_iterations: 64,
            max_owned_bytes: u64::MAX,
        },
    )
    .unwrap()
}

fn run(
    engine: &Engine,
    a: &Prepared,
    solver: Option<&ExperimentalSoilNewtonGpu>,
    reset: bool,
) -> ([u32; 8], [u32; 8]) {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if reset {
        a.driver.encode_reset(&mut encoder);
        if let Some(solver) = solver {
            solver.encode_reset(&mut encoder);
        }
    }
    a.driver.encode(&mut encoder);
    if let Some(solver) = solver {
        solver.encode(&mut encoder);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(
        dispatch.finish().encoded_dispatches,
        3 + u64::from(solver.is_some())
    );
    let prep = bytemuck::pod_read_unaligned(&capture::read(engine, a.driver.status_buffer())[..32]);
    let solve = solver
        .map(|s| bytemuck::pod_read_unaligned(&capture::read(engine, s.status_buffer())[..32]))
        .unwrap_or([0; 8]);
    (prep, solve)
}

fn verify_response(engine: &Engine, a: &Prepared, f: &Fixture, p: &Reference) {
    let response = values(engine, a.driver.response_buffer());
    let n = f.free.len();
    let mut worst = 0.0_f64;
    let mut oracle_error = 0.0_f64;
    for c in 0..f.contacts.len().max(1) {
        if c >= f.contacts.len() || f.contacts[c].soil_mass[3] == 0.0 {
            assert!(
                response[4 * c * n..4 * (c + 1) * n]
                    .iter()
                    .all(|v| *v == 0.0)
            );
            continue;
        }
        for i in 0..n {
            assert_eq!(response[4 * (c * n + i) + 3], 0.0);
            for axis in 0..3 {
                let actual = response[4 * (c * n + i) + axis] as f64;
                let expected: f64 = (0..n)
                    .map(|j| p.inverse[i * n + j] * f.rows[c * n + j][axis] as f64)
                    .sum();
                oracle_error = oracle_error.max((actual - expected).abs() / (1.0 + expected.abs()));
                let row = f.rows[c * n + i][axis] as f64;
                let products: Vec<_> = (0..n)
                    .map(|j| f.mass[i * n + j] as f64 * response[4 * (c * n + j) + axis] as f64)
                    .collect();
                let scale = row.abs() + products.iter().map(|v| v.abs()).sum::<f64>();
                let error = (products.iter().sum::<f64>() - row).abs()
                    / if scale == 0.0 { 1.0 } else { scale };
                worst = worst.max(error);
            }
        }
    }
    eprintln!(
        "device preparation dofs={n} slots={} active={} backward={worst:e} oracle={oracle_error:e}",
        f.contacts.len(),
        p.contacts.len()
    );
    assert!(worst <= 1e-6);
    assert!(oracle_error <= 1e-6);
}

#[test]
fn newton_preparation_shader_and_budget() {
    assert_eq!(
        ExperimentalSoilNewtonPrepareGpu::backend_label(),
        if cfg!(feature = "strict-rounding-newton-preparation") {
            "compensated-axis-parallel-strict-volatile-rounding"
        } else if cfg!(feature = "row-parallel-newton-preparation") {
            "compensated-row-parallel-cooperative-factor"
        } else if cfg!(feature = "axis-parallel-newton-preparation") {
            "compensated-axis-parallel-zero-term-and-residual-elision"
        } else if cfg!(feature = "zero-residual-newton-preparation") {
            "compensated-zero-term-and-residual-elision"
        } else if cfg!(feature = "zero-term-newton-preparation") {
            "compensated-zero-term-elision"
        } else {
            "compensated-dense-terms"
        }
    );
    assert_eq!(
        ExperimentalSoilNewtonPrepareGpu::required_bytes(32, 4096).unwrap(),
        4_203_184
    );
    for n in [0, 33] {
        assert!(ExperimentalSoilNewtonPrepareGpu::required_bytes(n, 0).is_err());
    }
    let source = crate::soil_newton_prepare_gpu::shader_source();
    let module = wgpu::naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
}

#[test]
#[ignore = "requires GPU; shared axis response exact oracle and paired cost"]
fn newton_preparation_axis_shared() {
    preparation_variant(9);
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires GPU; SPIR-V atomic preparation control"]
fn newton_preparation_axis_atomic() {
    preparation_variant(11);
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires GPU; volatile preparation exact oracle and paired cost"]
fn newton_preparation_axis_volatile() {
    preparation_variant(12);
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires GPU; volatile preparation faults and recovery"]
fn newton_preparation_axis_volatile_faults() {
    preparation_faults(Some(12));
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires GPU; strict volatile preparation exact oracle and paired cost"]
fn newton_preparation_axis_strict() {
    preparation_variant(13);
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires GPU; strict volatile preparation faults and recovery"]
fn newton_preparation_axis_strict_faults() {
    preparation_faults(Some(13));
}

#[cfg(all(
    feature = "strict-rounding-newton-preparation",
    not(target_arch = "wasm32")
))]
#[test]
#[ignore = "requires GPU; runtime strict rounding versus ordinary axis preparation"]
fn newton_preparation_runtime_rounding() {
    preparation_variant(14);
}

#[cfg(all(
    feature = "strict-rounding-newton-preparation",
    not(target_arch = "wasm32")
))]
#[test]
#[ignore = "requires GPU; runtime strict rounding faults and recovery"]
fn newton_preparation_runtime_rounding_faults() {
    preparation_faults(None);
}

#[cfg(all(
    feature = "strict-rounding-newton-preparation",
    not(target_arch = "wasm32")
))]
#[test]
#[ignore = "requires GPU; missing passthrough fails before owned allocations"]
fn newton_preparation_runtime_rounding_device() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!crate::soil_newton_gpu::protected_enabled(&engine));
    let f = pair(false, 0.0);
    let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let a = create_backend(&engine, &f, &p, Some(8));
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let result = ExperimentalSoilNewtonPrepareGpu::new(
        &engine,
        SoilNewtonPrepareInputs {
            mass: &a.mass,
            point_rows: &a.rows,
            contacts: &a.contacts,
            upstream_status: &a.upstream,
        },
        f.free.len() as u32,
        f.contacts.len() as u32,
        a.driver.owned_buffer_bytes(),
    );
    assert!(result.is_err());
    assert_eq!(scope.finish().observations, 0);
}

#[test]
#[ignore = "requires GPU; shared axis response fault/reset/recovery"]
fn newton_preparation_axis_shared_faults() {
    preparation_faults(Some(9));
}

#[test]
fn newton_preparation_axis_shared_source() {
    use crate::soil_newton_prepare_gpu::{axis_shared_bytes, axis_shared_source};
    for n in [0, 33, u32::MAX] {
        assert!(axis_shared_bytes(n).is_err());
    }
    for n in [1, 3, 10, 32] {
        assert_eq!(axis_shared_bytes(n).unwrap(), 384 + 512 * n);
        let source = axis_shared_source(n);
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        assert!(source.contains("refinement>=2u"));
        assert_eq!(source.matches("refinement<8u").count(), 1);
        assert!(source.contains("atomicLoad(&prepare_certificate[slot])==7u"));
        assert_eq!(
            source
                .matches("response[c*n+i][axis]=axis_response")
                .count(),
            1
        );
        assert_eq!(
            source
                .matches("correction[c*n+i][axis]=axis_correction")
                .count(),
            1
        );
    }
}

#[test]
#[ignore = "requires GPU; axis-parallel preparation exact oracle and paired cost"]
fn newton_preparation_axis_parallel() {
    preparation_variant(8);
}

#[test]
fn newton_preparation_axis_parallel_source() {
    let source = crate::soil_newton_prepare_gpu::axis_parallel_source();
    let module = wgpu::naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    assert!(source.contains("let c=group.x*16u+slot"));
    assert!(source.contains("atomicLoad(&prepare_certificate[slot])==7u"));
    assert!(source.contains("refinement>=2u"));
    assert_eq!(source.matches("refinement<8u").count(), 1);
    assert!(source.contains("@workgroup_size(16)\nfn finish("));
    let original = crate::soil_newton_prepare_gpu::sparse_residual_source();
    assert_eq!(
        source
            .split("fn factor()")
            .nth(1)
            .unwrap()
            .split("var<workgroup> prepare_state")
            .next()
            .unwrap()
            .trim(),
        original
            .split("fn factor()")
            .nth(1)
            .unwrap()
            .split("@compute @workgroup_size(64)\nfn prepare(")
            .next()
            .unwrap()
            .trim()
    );
    assert_eq!(
        source.split("fn finish(").nth(1),
        original.split("fn finish(").nth(1)
    );
}

#[test]
#[ignore = "requires GPU; cached-certificate preparation exact oracle and paired cost"]
fn newton_preparation_certificate_mask() {
    preparation_variant(7);
}

#[test]
fn newton_preparation_certificate_mask_source() {
    let source = crate::soil_newton_prepare_gpu::certificate_mask_source();
    let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(source.matches("certificate_masks[i]=mask").count(), 1);
    assert_eq!(
        source
            .matches("let error=component_error(c,i,axis)")
            .count(),
        2
    );
    assert_eq!(source.matches("refinement<8u").count(), 1);
    assert!(source.contains("if refinement>=2u && certified(c) {break;}"));
    let original = crate::soil_newton_prepare_gpu::sparse_residual_source();
    for marker in ["fn factor()", "fn finish("] {
        let end = if marker == "fn factor()" {
            "fn prepare("
        } else {
            ""
        };
        let section = |s: &str| {
            let start = s.find(marker).unwrap();
            let end = if end.is_empty() {
                s.len()
            } else {
                s.find(end).unwrap()
            };
            s[start..end].to_string()
        };
        assert_eq!(section(&source), section(&original));
    }
}

#[test]
#[ignore = "requires GPU timestamps; fixed-width preparation exact oracle and paired cost"]
fn newton_preparation_fixed_width() {
    preparation_variant(0);
}

#[test]
#[ignore = "requires GPU timestamps; private-response preparation oracle and paired cost"]
fn newton_preparation_private_response() {
    preparation_variant(1);
}

#[test]
#[ignore = "requires GPU timestamps; zero-term preparation exact oracle and paired cost"]
fn newton_preparation_zero_terms() {
    preparation_variant(2);
}

#[test]
#[ignore = "requires GPU timestamps; sparse-residual preparation exact oracle and paired cost"]
fn newton_preparation_sparse_residual() {
    preparation_variant(3);
}

#[test]
#[ignore = "requires f64 GPU; native preparation accumulator physical, exact and cost comparison"]
fn newton_preparation_native_accumulator() {
    preparation_variant(4);
}

#[test]
#[ignore = "requires GPU; eight-lane preparation physical, exact and paired cost comparison"]
fn newton_preparation_narrow_workgroup() {
    preparation_variant(5);
}

#[test]
#[ignore = "requires f64 GPU; native-pair preparation physical, exact and paired cost comparison"]
fn newton_preparation_native_pairs() {
    preparation_variant(6);
}

#[test]
fn newton_preparation_native_pairs_source() {
    let source = crate::soil_newton_prepare_gpu::native_pair_source();
    let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(
        source.replace(
            &crate::soil_rate_encoding::native_arithmetic(64),
            &crate::soil_rate_encoding::sorted_arithmetic(64)
        ),
        crate::soil_newton_prepare_gpu::sparse_residual_source()
    );
}

#[test]
fn newton_preparation_narrow_workgroup_source() {
    let source = crate::soil_newton_prepare_gpu::narrow_workgroup_source();
    let module = wgpu::naga::front::wgsl::parse_str(&source).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(source.matches("@workgroup_size(8)").count(), 2);
    assert!(source.contains("rate_scratch:array<atomic<u32>,8>"));
    assert_eq!(
        source
            .replace("@workgroup_size(8)", "@workgroup_size(64)")
            .replace(
                "rate_scratch:array<atomic<u32>,8>",
                "rate_scratch:array<atomic<u32>,64>"
            ),
        crate::soil_newton_prepare_gpu::sparse_residual_source()
    );
}

#[test]
fn newton_preparation_native_accumulator_source() {
    let source = crate::soil_newton_prepare_gpu::native_accumulator_source();
    let module = wgpu::naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(!source.contains("rate_scratch"));
    assert!(source.contains("refinement<8u") && source.contains("refinement>=2u && certified(c)"));
    assert!(source.contains("error>1e-6") && source.contains("scale+=abs(m)*abs(r)"));
}

#[test]
fn newton_preparation_sparse_residual_source() {
    let source = crate::soil_newton_prepare_gpu::sparse_residual_source();
    let module = wgpu::naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(source.matches("residual=add_nonzero_term").count(), 2);
    assert!(source.contains("scale+=abs(m)*abs(r)"));
    assert!(source.contains("coefficient==0.0 && finite(term)"));
    assert_eq!(source.matches("refinement<8u").count(), 1);
    assert!(source.contains("error>1e-6"));
}

#[test]
fn newton_preparation_zero_terms_source() {
    let source = crate::soil_newton_prepare_gpu::zero_terms_source();
    let module = wgpu::naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(source.matches("sum=add_nonzero_term").count(), 5);
    assert!(source.contains("coefficient==0.0 && finite(term)"));
    assert_eq!(source.matches("refinement<8u").count(), 1);
    assert!(source.contains("error>1e-6"));
}

#[test]
fn newton_preparation_private_response_source() {
    let source = crate::soil_newton_prepare_gpu::private_response_source();
    let module = wgpu::naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert!(source.contains("let r=local_response[j][axis]"));
    assert!(source.contains("response[c*n+i]=local_response[i]"));
    assert_eq!(source.matches("refinement<8u").count(), 1);
}

fn preparation_variant(variant: u32) {
    let mut tractor = tractor_cancellation_fixture();
    let contact = tractor.contacts[0];
    let rows = tractor.rows[..10].to_vec();
    tractor.contacts = vec![contact; 251];
    tractor.rows = rows.repeat(251);
    let mut empty = pair(false, 0.0);
    empty.contacts.clear();
    empty.rows.clear();
    preparation_fixtures(
        variant,
        vec![
            pair(false, 0.0),
            wide_fixture(),
            tractor_cancellation_fixture(),
            moved_tractor_cancellation_fixture(),
            spread_contacts(&tractor, 4096, 13),
            empty,
        ],
    );
}

fn preparation_fixtures(variant: u32, fixtures: Vec<Fixture>) {
    let label = match variant {
        0 => "fixed-width",
        1 => "private-response",
        2 => "zero-terms",
        3 => "sparse-residual",
        4 => "native-accumulator",
        5 => "narrow-workgroup",
        6 => "native-pairs",
        7 => "certificate-mask",
        8 => "axis-parallel",
        9 => "axis-shared",
        10 => "axis-phase-control",
        11 => "axis-atomic-control",
        12 => "axis-volatile",
        13 => "axis-strict-volatile",
        14 => "runtime-strict-volatile",
        15 => "row-parallel",
        16 => "runtime-row-parallel",
        17 => "row-factor",
        18 => "runtime-row-factor",
        19 => "row-sized",
        _ => panic!("unknown preparation variant"),
    };
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | if matches!(variant, 11 | 12 | 13 | 14) {
                wgpu::Features::PASSTHROUGH_SHADERS
            } else {
                wgpu::Features::empty()
            }
            | if matches!(variant, 4 | 6) {
                wgpu::Features::SHADER_F64
            } else {
                wgpu::Features::empty()
            },
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let capture_root = (variant >= 4).then(|| {
        let path =
            std::path::PathBuf::from(std::env::var_os("MOLLA_NATIVE_PREPARE_CAPTURE").unwrap());
        std::fs::create_dir(&path).unwrap();
        path
    });
    for (fixture, f) in fixtures.into_iter().enumerate() {
        eprintln!("preparation fixture={fixture}");
        let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
        let reference = create_backend(
            &engine,
            &f,
            &p,
            Some(if variant == 19 {
                17
            } else if matches!(variant, 17 | 18) {
                15
            } else if matches!(variant, 12 | 13) {
                11
            } else if matches!(variant, 9 | 10 | 11 | 14 | 15 | 16) {
                8
            } else if variant >= 4 {
                3
            } else if variant == 3 {
                2
            } else {
                0
            }),
        );
        let build = match variant {
            0 => ExperimentalSoilNewtonPrepareGpu::new_fixed_width,
            1 => ExperimentalSoilNewtonPrepareGpu::new_private_response,
            2 => ExperimentalSoilNewtonPrepareGpu::new_zero_terms,
            3 => ExperimentalSoilNewtonPrepareGpu::new_sparse_residual,
            4 => ExperimentalSoilNewtonPrepareGpu::new_native_accumulator,
            5 => ExperimentalSoilNewtonPrepareGpu::new_narrow_workgroup,
            6 => ExperimentalSoilNewtonPrepareGpu::new_native_pairs,
            7 => ExperimentalSoilNewtonPrepareGpu::new_certificate_mask,
            8 => ExperimentalSoilNewtonPrepareGpu::new_axis_parallel,
            9 => ExperimentalSoilNewtonPrepareGpu::new_axis_shared,
            15 => ExperimentalSoilNewtonPrepareGpu::new_row_parallel,
            17 => ExperimentalSoilNewtonPrepareGpu::new_row_factor,
            19 => ExperimentalSoilNewtonPrepareGpu::new_row_sized,
            #[cfg(all(
                feature = "row-parallel-newton-preparation",
                not(feature = "strict-rounding-newton-preparation")
            ))]
            18 => ExperimentalSoilNewtonPrepareGpu::new,
            #[cfg(all(
                feature = "row-parallel-newton-preparation",
                not(feature = "strict-rounding-newton-preparation")
            ))]
            16 => ExperimentalSoilNewtonPrepareGpu::new,
            10 => ExperimentalSoilNewtonPrepareGpu::new_axis_parallel,
            #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
            11 => ExperimentalSoilNewtonPrepareGpu::new_axis_rounding::<0>,
            #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
            12 => ExperimentalSoilNewtonPrepareGpu::new_axis_rounding::<1>,
            #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
            13 => ExperimentalSoilNewtonPrepareGpu::new_axis_rounding::<2>,
            #[cfg(all(
                feature = "strict-rounding-newton-preparation",
                not(target_arch = "wasm32")
            ))]
            14 => ExperimentalSoilNewtonPrepareGpu::new,
            _ => unreachable!(),
        };
        let driver = build(
            &engine,
            SoilNewtonPrepareInputs {
                mass: &reference.mass,
                point_rows: &reference.rows,
                contacts: &reference.contacts,
                upstream_status: &reference.upstream,
            },
            f.free.len() as u32,
            f.contacts.len() as u32,
            reference.driver.owned_buffer_bytes(),
        )
        .unwrap();
        let candidate = Prepared {
            driver,
            mass: reference.mass.clone(),
            rows: reference.rows.clone(),
            contacts: reference.contacts.clone(),
            free: reference.free.clone(),
            upstream: reference.upstream.clone(),
        };
        for pass in 0..2 {
            let reference_status = run(&engine, &reference, None, true);
            let candidate_status = run(&engine, &candidate, None, true);
            let reference_response = capture::read(&engine, reference.driver.response_buffer());
            let candidate_response = capture::read(&engine, candidate.driver.response_buffer());
            let factors = if matches!(variant, 17 | 18 | 19) {
                Some([
                    capture::read(&engine, reference.driver.factor_buffer()),
                    capture::read(&engine, candidate.driver.factor_buffer()),
                ])
            } else {
                None
            };
            if let Some(root) = &capture_root {
                if let Some(factors) = &factors {
                    for (arm, bytes) in ["reference", "candidate"].into_iter().zip(factors) {
                        std::fs::write(
                            root.join(format!("fixture-{fixture}-pass-{pass}-{arm}-factor.bin")),
                            bytes,
                        )
                        .unwrap();
                    }
                }
                for (name, bytes) in [
                    ("reference-response", reference_response.as_slice()),
                    ("candidate-response", candidate_response.as_slice()),
                    ("reference-status", bytemuck::bytes_of(&reference_status.0)),
                    ("candidate-status", bytemuck::bytes_of(&candidate_status.0)),
                ] {
                    std::fs::write(
                        root.join(format!("fixture-{fixture}-pass-{pass}-{name}.bin")),
                        bytes,
                    )
                    .unwrap();
                }
            }
            verify_response(&engine, &reference, &f, &p);
            verify_response(&engine, &candidate, &f, &p);
            assert_eq!(reference_status, candidate_status);
            assert_eq!(reference_response, candidate_response);
            if let Some(factors) = factors {
                assert_eq!(factors[0], factors[1]);
            }
        }
        let query_stride = if variant >= 10 { 8u32 } else { 2u32 };
        if let Some(root) = &capture_root {
            std::fs::write(root.join("query-layout.json"), serde_json::to_vec_pretty(&serde_json::json!({
                "query_stride":query_stride,"samples":18,"timestamp_period_ns":queue.get_timestamp_period(),
                "phases":if variant>=10 {vec!["factor","response","cleanup"]} else {Vec::<&str>::new()},
            })).unwrap()).unwrap();
        }
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("fixed-width preparation cost"),
            ty: wgpu::QueryType::Timestamp,
            count: 18 * query_stride,
        });
        let ticks = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fixed-width preparation ticks"),
            size: u64::from(18 * query_stride) * 8,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let statuses = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fixed-width preparation statuses"),
            size: 18 * 32,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        for i in 0..18u32 {
            let arm = ((i / 2 + i % 2) % 2) as usize;
            let owner = [&reference, &candidate][arm];
            encoder.write_timestamp(&queries, query_stride * i);
            owner.driver.encode_reset(&mut encoder);
            if variant >= 10 {
                use crate::soil_newton_prepare_gpu::PreparationPhase;
                owner
                    .driver
                    .encode_traced(&mut encoder, &mut |encoder, phase, begin| {
                        let stage = match phase {
                            PreparationPhase::Factor => 0,
                            PreparationPhase::Response => 1,
                            PreparationPhase::Cleanup => 2,
                            _ => panic!("unexpected preparation phase"),
                        };
                        encoder.write_timestamp(
                            &queries,
                            query_stride * i + 1 + 2 * stage + u32::from(!begin),
                        );
                    });
            } else {
                owner.driver.encode(&mut encoder);
            }
            encoder.write_timestamp(&queries, query_stride * i + query_stride - 1);
            encoder.copy_buffer_to_buffer(
                owner.driver.status_buffer(),
                0,
                &statuses,
                u64::from(i) * 32,
                32,
            );
        }
        encoder.resolve_query_set(&queries, 0..18 * query_stride, &ticks, 0);
        queue.submit([encoder.finish()]);
        assert_eq!(scope.finish().observations, 0);
        let traffic = traffic.finish();
        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
        let raw = capture::read(&engine, &ticks);
        let states = capture::read(&engine, &statuses);
        if let Some(root) = &capture_root {
            std::fs::write(root.join(format!("fixture-{fixture}-ticks.bin")), &raw).unwrap();
            std::fs::write(
                root.join(format!("fixture-{fixture}-statuses.bin")),
                &states,
            )
            .unwrap();
        }
        let mut samples = [Vec::new(), Vec::new()];
        for (i, tick) in raw.chunks_exact(query_stride as usize * 8).enumerate() {
            let times: Vec<u64> = tick
                .chunks_exact(8)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            assert!(times.last().unwrap() > &times[0]);
            let status: [u32; 8] = bytemuck::pod_read_unaligned(&states[i * 32..(i + 1) * 32]);
            assert_eq!(status[0], 0);
            assert_eq!(status[2] as usize, p.contacts.len());
            let arm = (i / 2 + i % 2) % 2;
            let ms = (times.last().unwrap() - times[0]) as f64
                * f64::from(queue.get_timestamp_period())
                / 1e6;
            if variant >= 10 {
                assert!(times.windows(2).all(|t| t[0] <= t[1]));
                for (stage, name) in ["factor", "response", "cleanup"].into_iter().enumerate() {
                    let phase_ms = (times[2 + 2 * stage] - times[1 + 2 * stage]) as f64
                        * f64::from(queue.get_timestamp_period())
                        / 1e6;
                    eprintln!(
                        "preparation-phase fixture={fixture} arm={arm} index={i} phase={name} gpu_ms={phase_ms:.9}"
                    );
                }
            }
            samples[arm].push(ms);
            eprintln!(
                "{label} sample dofs={} slots={} active={} arm={arm} index={i} gpu_ms={ms:.9}",
                f.free.len(),
                f.contacts.len(),
                p.contacts.len()
            );
        }
        for arm in 0..2 {
            samples[arm].sort_by(f64::total_cmp);
        }
        eprintln!(
            "{label} median dofs={} slots={} active={} reference_ms={:.9} candidate_ms={:.9}",
            f.free.len(),
            f.contacts.len(),
            p.contacts.len(),
            samples[0][4],
            samples[1][4]
        );
        assert_eq!(
            capture::read(&engine, reference.driver.response_buffer()),
            capture::read(&engine, candidate.driver.response_buffer())
        );
    }
}

#[test]
#[ignore = "requires GPU; device Cholesky and response versus f64 oracle"]
fn newton_preparation_gpu_reference() {
    let engine = preparation_engine(wgpu::Features::empty());
    let mut empty = pair(false, 0.0);
    empty.contacts.clear();
    empty.rows.clear();
    let mut one = pair(false, 0.0);
    one.mass = vec![1.0];
    one.free = vec![0.0];
    one.rows = one.rows.chunks_exact(3).map(|r| r[0]).collect();
    for f in [
        one,
        pair(false, 0.0),
        articulated(7),
        wide_fixture(),
        spread_contacts(&articulated(65), 4096, 63),
        empty,
    ] {
        let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
        let a = create(&engine, &f, &p);
        for _ in 0..2 {
            let (status, _) = run(&engine, &a, None, true);
            assert_eq!(status[0], 0, "preparation {status:?}");
            assert_eq!(status[2] as usize, p.contacts.len());
            verify_response(&engine, &a, &f, &p);
            engine.queue().unwrap().write_buffer(
                a.driver.response_buffer(),
                0,
                bytemuck::cast_slice(&vec![
                    f32::NAN;
                    a.driver.response_buffer().size() as usize / 4
                ]),
            );
        }
    }
}

#[test]
#[ignore = "requires GPU; preparation and resident solve in one submission"]
fn newton_preparation_gpu_coupled() {
    let engine = preparation_engine(wgpu::Features::empty());
    for f in [
        pair(true, 0.3),
        articulated(7),
        spread_contacts(&articulated(65), 4096, 63),
    ] {
        let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
        let a = create(&engine, &f, &p);
        let solver = solver(&engine, &a, &f);
        for reset in [true, false] {
            let (prepared, solved) = run(&engine, &a, Some(&solver), reset);
            eprintln!(
                "device prepared solve prep={:?} solve={:?}",
                &prepared[..4],
                &solved[..4]
            );
            assert_eq!(prepared[0], 0);
            assert_eq!(solved[0], 0);
            verify_response(&engine, &a, &f, &p);
            publication::verify_outputs(
                &engine,
                solver.outputs_buffer(),
                solver.rates_buffer(),
                &f,
                &p,
            );
        }
    }
}

fn tractor_cancellation_fixture() -> Fixture {
    let mut mass = vec![0.; 100];
    for (i, bits) in [
        0x44570000, 0x44570000, 0x44570000, 0x432d34f1, 0x43dbea16, 0x43cd39fd, 0x4062cccc,
        0x4062cccc, 0x4062cccc, 0x4062cccc,
    ]
    .into_iter()
    .enumerate()
    {
        mass[11 * i] = f32::from_bits(bits);
    }
    for (i, j, bits) in [
        (0, 5, 0x42b6cccc),
        (1, 5, 0xc1e80000),
        (2, 3, 0xc2b6cccc),
        (2, 4, 0x41e80000),
        (3, 4, 0xc1a70a3e),
    ] {
        mass[10 * i + j] = f32::from_bits(bits);
        mass[10 * j + i] = f32::from_bits(bits);
    }
    for j in 6..10 {
        for (i, bits) in [
            (0, 0xb6000000),
            (4, if j % 2 == 0 { 0x35800000 } else { 0xb5800000 }),
            (5, 0x4062ccc8),
        ] {
            mass[10 * i + j] = f32::from_bits(bits);
            mass[10 * j + i] = f32::from_bits(bits);
        }
    }
    let rows = [
        [0x3f800000, 0, 0, 0],
        [0, 0x3f800000, 0, 0],
        [0, 0, 0x3f800000, 0],
        [0, 0xbf4cccce, 0xbf599999, 0],
        [0x3f4cccce, 0, 0xbf8ccccd, 0],
        [0x3f599999, 0x3f8ccccd, 0x80000000, 0],
        [0; 4],
        [0; 4],
        [0; 4],
        [0x3e999999, 0x3eb33334, 0, 0],
    ]
    .map(|row| row.map(f32::from_bits))
    .to_vec();
    let mut contact = pair(false, 0.).contacts[0];
    contact.soil_mass[3] = 1.;
    Fixture {
        mass,
        rows,
        contacts: vec![contact],
        free: vec![0.; 10],
    }
}

#[test]
#[ignore = "requires GPU; captured tractor cancellation and sparse-slot regression"]
fn newton_preparation_gpu_tractor_cancellation() {
    let engine = preparation_engine(wgpu::Features::empty());
    let mut fixtures = Vec::new();
    for mut f in [
        tractor_cancellation_fixture(),
        moved_tractor_cancellation_fixture(),
    ] {
        f.contacts = vec![f.contacts[0]; 65];
        f.rows = f.rows.repeat(65);
        fixtures.push(spread_contacts(&f, 4096, 63));
        f.contacts.truncate(1);
        f.rows.truncate(10);
        fixtures.push(f);
    }
    for f in fixtures {
        let p = Reference::new(&f, vec![0.; 10]);
        let a = create(&engine, &f, &p);
        for _ in 0..2 {
            let (status, _) = run(&engine, &a, None, true);
            assert_eq!(status[0], 0, "tractor cancellation {status:?}");
            verify_response(&engine, &a, &f, &p);
        }
    }
}

fn moved_tractor_cancellation_fixture() -> Fixture {
    let mut f = tractor_cancellation_fixture();
    f.mass.fill(0.);
    for (i, bits) in [
        0x44570000, 0x44570000, 0x44570000, 0x432d3529, 0x43dbe9fc, 0x43cd39fd, 0x4062ccc8,
        0x4062ccc8, 0x4062ccd0, 0x4062ccd0,
    ]
    .into_iter()
    .enumerate()
    {
        f.mass[11 * i] = f32::from_bits(bits);
    }
    for (i, j, bits) in [
        (0, 5, 0x42b6cd18),
        (1, 5, 0xc1e7fc4a),
        (2, 3, 0xc2b6cd18),
        (2, 4, 0x41e7fc4a),
        (3, 4, 0xc1a7150d),
        (4, 6, 0x35800000),
        (4, 7, 0xb5800000),
    ] {
        f.mass[10 * i + j] = f32::from_bits(bits);
        f.mass[10 * j + i] = f32::from_bits(bits);
    }
    for j in 6..10 {
        for (i, bits) in [
            (2, if j < 8 { 0xa1aa3d00 } else { 0x202e1800 }),
            (5, if j < 8 { 0x4062ccc8 } else { 0x4062ccd0 }),
        ] {
            f.mass[10 * i + j] = f32::from_bits(bits);
            f.mass[10 * j + i] = f32::from_bits(bits);
        }
    }
    f.rows[3] = [0, 0xbdccccd3, 0xbf59955a, 0].map(f32::from_bits);
    f.rows[4] = [0x3dccccd3, 0, 0x3fa66654, 0].map(f32::from_bits);
    f.rows[5] = [0x3f59955a, 0xbfa66654, 0, 0].map(f32::from_bits);
    f.rows[9] = [0.; 4];
    f
}

#[test]
#[ignore = "requires GPU; preparation faults, clearing and solver fault propagation"]
fn newton_preparation_gpu_faults() {
    preparation_faults(None);
}

#[test]
#[ignore = "requires GPU; cached-certificate faults, clearing and recovery"]
fn newton_preparation_cached_faults() {
    preparation_faults(Some(7));
}

#[test]
#[ignore = "requires GPU; axis-parallel preparation faults, clearing and recovery"]
fn newton_preparation_axis_faults() {
    preparation_faults(Some(8));
}

#[test]
#[ignore = "requires GPU; cooperative-row preparation exact fixtures and cost"]
fn newton_preparation_rows_gpu() {
    preparation_variant(15);
}

#[test]
#[ignore = "requires GPU; cooperative-row preparation faults and recovery"]
fn newton_preparation_rows_faults() {
    preparation_faults(Some(15));
}

#[test]
#[ignore = "requires GPU; cooperative Cholesky exact factors and preparation cost"]
fn newton_preparation_row_factor_gpu() {
    preparation_variant(17);
}

#[test]
#[ignore = "requires GPU; cooperative Cholesky invalid input and recovery"]
fn newton_preparation_row_factor_faults() {
    preparation_faults(Some(17));
}

#[test]
#[ignore = "requires GPU; width-sized workgroups, exact outputs and phase cost"]
fn newton_preparation_row_sized_gpu() {
    let mut fixtures = Vec::new();
    for n in [1usize, 3, 10, 11, 21, 22, 32] {
        let wide = wide_fixture();
        let mut f = Fixture {
            mass: (0..n)
                .flat_map(|i| wide.mass[i * 32..i * 32 + n].iter().copied())
                .collect(),
            free: wide.free[..n].to_vec(),
            rows: wide
                .rows
                .chunks_exact(32)
                .flat_map(|r| r[..n].iter().copied())
                .collect(),
            contacts: wide.contacts,
        };
        if n == 10 {
            f = spread_contacts(&f, 4096, 63);
        }
        fixtures.push(f);
    }
    let mut empty = pair(false, 0.0);
    empty.contacts.clear();
    empty.rows.clear();
    fixtures.push(empty);
    preparation_fixtures(19, fixtures);
}

#[test]
#[ignore = "requires GPU; width-sized preparation faults and recovery"]
fn newton_preparation_row_sized_faults() {
    preparation_faults(Some(19));
}

#[cfg(feature = "row-parallel-newton-preparation")]
#[test]
fn newton_preparation_rows_runtime_label() {
    assert_eq!(
        ExperimentalSoilNewtonPrepareGpu::backend_label(),
        if cfg!(feature = "strict-rounding-newton-preparation") {
            "compensated-axis-parallel-strict-volatile-rounding"
        } else {
            "compensated-row-parallel-cooperative-factor"
        }
    );
}

fn preparation_faults(backend: Option<u32>) {
    let engine = preparation_engine(if matches!(backend, Some(11 | 12 | 13)) {
        wgpu::Features::PASSTHROUGH_SHADERS
    } else {
        wgpu::Features::empty()
    });
    let f = pair(false, 0.0);
    let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let a = create_backend(&engine, &f, &p, backend);
    let solver = solver(&engine, &a, &f);
    assert_eq!(run(&engine, &a, Some(&solver), true).1[0], 0);
    for (target, offset, bad, expected) in [
        (&a.upstream, 0, 1u32, 1u32),
        (&a.mass, 0, (-1.0f32).to_bits(), 2),
        (&a.rows, 0, f32::NAN.to_bits(), 1),
        (&a.contacts, 12, (-1.0f32).to_bits(), 1),
    ] {
        let original = capture::read(&engine, target);
        engine
            .queue()
            .unwrap()
            .write_buffer(target, offset, bytemuck::bytes_of(&bad));
        let (prep, solve) = run(&engine, &a, Some(&solver), true);
        assert_eq!(prep[0], expected);
        let packet = capture::read(&engine, a.driver.failure_buffer());
        let word = |i: usize| u32::from_le_bytes(packet[4 * i..4 * i + 4].try_into().unwrap());
        assert_eq!((word(0), word(1), word(2)), (1, 3, f.contacts.len() as u32));
        assert_eq!(word(4), expected);
        assert_eq!(packet[64..100], capture::read(&engine, &a.mass));
        if word(3) > 0 {
            let slot = word(3) as usize - 1;
            let rows = capture::read(&engine, &a.rows);
            assert_eq!(packet[100..148], rows[48 * slot..48 * (slot + 1)]);
            let contacts = capture::read(&engine, &a.contacts);
            assert_eq!(packet[148..212], contacts[64 * slot..64 * (slot + 1)]);
        }
        assert_ne!(solve[0], 0);
        for buffer in [
            a.driver.response_buffer(),
            solver.outputs_buffer(),
            solver.rates_buffer(),
        ] {
            assert!(capture::read(&engine, buffer).iter().all(|&b| b == 0));
        }
        engine.queue().unwrap().write_buffer(target, 0, &original);
        assert_eq!(run(&engine, &a, Some(&solver), false).0, prep);
        assert_eq!(capture::read(&engine, a.driver.failure_buffer()), packet);
        assert_eq!(run(&engine, &a, Some(&solver), true).1[0], 0);
        assert_eq!(
            &capture::read(&engine, a.driver.failure_buffer())[..4],
            &[0; 4]
        );
    }
    for bad in [
        vec![0.0f32; 9],
        vec![1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        vec![1.0, 0.1, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    ] {
        engine
            .queue()
            .unwrap()
            .write_buffer(&a.mass, 0, bytemuck::cast_slice(&bad));
        let (prep, solve) = run(&engine, &a, Some(&solver), true);
        assert_eq!(prep[0], 2);
        assert_ne!(solve[0], 0);
        assert!(
            capture::read(&engine, a.driver.response_buffer())
                .iter()
                .all(|&b| b == 0)
        );
    }
    let tiny: Vec<f32> = f.mass.iter().map(|v| v * 1e-30).collect();
    engine
        .queue()
        .unwrap()
        .write_buffer(&a.mass, 0, bytemuck::cast_slice(&tiny));
    engine
        .queue()
        .unwrap()
        .write_buffer(&a.rows, 0, bytemuck::bytes_of(&1e20f32));
    let (prep, solve) = run(&engine, &a, Some(&solver), true);
    assert_eq!(prep[0], 4);
    assert_ne!(solve[0], 0);
    for b in [
        a.driver.response_buffer(),
        solver.outputs_buffer(),
        solver.rates_buffer(),
    ] {
        assert!(capture::read(&engine, b).iter().all(|&v| v == 0));
    }
    engine
        .queue()
        .unwrap()
        .write_buffer(&a.mass, 0, bytemuck::cast_slice(&f.mass));
    engine
        .queue()
        .unwrap()
        .write_buffer(&a.rows, 0, bytemuck::cast_slice(&f.rows));
    assert_eq!(run(&engine, &a, Some(&solver), true).1[0], 0);
}

#[test]
#[ignore = "requires GPU and saved input; fully device-prepared captured solve"]
fn newton_preparation_gpu_captured_input() {
    let root = std::path::PathBuf::from(std::env::var("MOLLA_SHARED_FAILURE_REPLAY").unwrap());
    let (f, pairs) = failure::load_paired(&root).unwrap();
    let p = Reference::new(&f, combined(&pairs));
    let engine = preparation_engine(wgpu::Features::empty());
    let a = create(&engine, &f, &p);
    let solver = solver(&engine, &a, &f);
    let (prep, solve) = run(&engine, &a, Some(&solver), true);
    eprintln!(
        "captured device preparation={prep:?} solve={:?}",
        &solve[..4]
    );
    assert_eq!(prep[0], 0);
    verify_response(&engine, &a, &f, &p);
    assert_eq!(solve[0], 0);
    publication::verify_outputs(
        &engine,
        solver.outputs_buffer(),
        solver.rates_buffer(),
        &f,
        &p,
    );
}

#[test]
#[ignore = "requires GPU and a latched Newton preparation packet"]
fn newton_preparation_gpu_failure_replay() {
    let root = std::path::PathBuf::from(std::env::var("MOLLA_NEWTON_PREPARATION_REPLAY").unwrap());
    let bytes = std::fs::read(root.join("preparation.bin")).unwrap();
    let metadata: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("capture.json")).unwrap()).unwrap();
    assert_eq!(
        metadata["blake3"].as_str().unwrap(),
        blake3::hash(&bytes).to_hex().as_str()
    );
    let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap());
    assert_eq!(word(0), 1);
    let n = word(1) as usize;
    assert!((1..=32).contains(&n));
    assert!(word(3) > 0 && word(3) <= word(2));
    assert_eq!(
        word(12),
        0,
        "replay requires an originating preparation failure"
    );
    assert_eq!(bytes.len(), 128 + 4 * n * n + 16 * n);
    let rows = 64 + 4 * n * n;
    let contact = rows + 16 * n;
    let f = Fixture {
        mass: bytes[64..rows]
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect(),
        free: vec![0.; n],
        rows: bytes[rows..contact]
            .chunks_exact(16)
            .map(bytemuck::pod_read_unaligned)
            .collect(),
        contacts: vec![bytemuck::pod_read_unaligned(&bytes[contact..])],
    };
    let p = Reference::new(&f, vec![0.; n]);
    let engine = preparation_engine(wgpu::Features::empty());
    let a = create(&engine, &f, &p);
    let (status, _) = run(&engine, &a, None, true);
    eprintln!(
        "preparation packet source_slots={} source_contact={} original_fault={} original_backward={:e}; replay={status:?} backward={:e}",
        word(2),
        word(3),
        word(4),
        f32::from_bits(word(7)),
        f32::from_bits(status[3])
    );
    let expected = std::env::var("MOLLA_NEWTON_PREP_EXPECT_FAULT")
        .map(|v| v.parse::<u32>().unwrap())
        .unwrap_or(0);
    if std::env::var_os("MOLLA_NEWTON_PREP_DIAGNOSTIC").is_some() {
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        a.driver.encode_reset(&mut encoder);
        a.driver.encode_uncleared_diagnostic(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        let response = values(&engine, a.driver.response_buffer());
        for axis in 0..3 {
            let q: Vec<_> = (0..n).map(|i| response[4 * i + axis] as f64).collect();
            let oracle: Vec<_> = (0..n)
                .map(|i| {
                    (0..n)
                        .map(|j| p.inverse[i * n + j] * f.rows[j][axis] as f64)
                        .sum::<f64>()
                })
                .collect();
            eprintln!("axis={axis} response={q:?} oracle={oracle:?}");
            for i in 0..n {
                let products: Vec<_> = (0..n).map(|j| f.mass[i * n + j] as f64 * q[j]).collect();
                let row = f.rows[i][axis] as f64;
                let scale = row.abs() + products.iter().map(|v| v.abs()).sum::<f64>();
                let error = (products.iter().sum::<f64>() - row).abs()
                    / if scale == 0. { 1. } else { scale };
                if error > 1e-6 {
                    eprintln!(
                        "bad component row={i} axis={axis} error={error:e} products={products:?} rhs={row:e}"
                    );
                }
            }
        }
    }
    assert_eq!(status[0], expected);
    if expected == 0 {
        verify_response(&engine, &a, &f, &p);
    }
}
