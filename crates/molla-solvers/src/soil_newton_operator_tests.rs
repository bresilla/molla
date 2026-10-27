use super::*;

#[path = "soil_newton_krylov_tests.rs"]
mod krylov;
#[path = "soil_native_pair_tests.rs"]
mod native_pairs;
#[path = "soil_newton_tests.rs"]
mod nonlinear;
#[path = "soil_newton_prepare_tests.rs"]
mod preparation;
#[path = "soil_newton_publish_tests.rs"]
mod publication;
use crate::soil_newton_operator_gpu::{
    ExperimentalSoilNewtonOperatorGpu, SoilNewtonOperatorInputs, SoilNewtonState, SoilNewtonVector,
};

fn vector(x: &[f64]) -> SoilNewtonVector {
    SoilNewtonVector {
        contact: pack(xyz(x), 0.0),
        wall: pack(xyz(&x[3..]), 0.0),
    }
}

fn state(x: &[f64]) -> SoilNewtonState {
    let high = vector(x);
    let low: Vec<_> = x
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v - if i < 3 {
                high.contact[i] as f64
            } else {
                high.wall[i - 3] as f64
            }
        })
        .collect();
    SoilNewtonState {
        high,
        low: vector(&low),
    }
}

fn create_operator(
    engine: &Engine,
    f: &Fixture,
    p: &Reference,
    x: &[f64],
    v: &[f64],
) -> (
    ExperimentalSoilNewtonOperatorGpu,
    wgpu::Buffer,
    wgpu::Buffer,
) {
    let n = f.free.len();
    let c = f.contacts.len();
    let mut states = vec![SoilNewtonState::default(); c.max(1)];
    let mut vectors = vec![SoilNewtonVector::default(); c.max(1)];
    for ((contact, x), v) in p
        .contacts
        .iter()
        .zip(x.chunks_exact(6))
        .zip(v.chunks_exact(6))
    {
        states[contact.index] = state(x);
        vectors[contact.index] = vector(v);
    }
    let states = storage(engine, &states);
    let vectors = storage(engine, &vectors);
    let pairs: Vec<_> = p
        .free
        .iter()
        .map(|q| [*q as f32, (*q - (*q as f32) as f64) as f32])
        .collect();
    let free = storage(engine, &pairs);
    let rows = buffer(engine, &f.rows, 16 * n as u64);
    let mut response = vec![[0f32; 4]; c.max(1) * n];
    for contact in &p.contacts {
        for i in 0..n {
            response[contact.index * n + i] = pack(
                (0..n).map(|j| p.inverse[i * n + j] * contact.rows[j]).sum(),
                0.0,
            );
        }
    }
    let response = storage(engine, &response);
    let contacts = buffer(engine, &f.contacts, 64);
    let inputs = || SoilNewtonOperatorInputs {
        free_rates: &free,
        point_rows: &rows,
        point_response: &response,
        contacts: &contacts,
        state: &states,
        vector: &vectors,
    };
    let required = ExperimentalSoilNewtonOperatorGpu::required_bytes(n as u32, c as u32).unwrap();
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        ExperimentalSoilNewtonOperatorGpu::new(engine, inputs(), n as u32, c as u32, required - 1)
            .is_err()
    );
    assert!(
        ExperimentalSoilNewtonOperatorGpu::new(
            engine,
            SoilNewtonOperatorInputs {
                state: &vectors,
                ..inputs()
            },
            n as u32,
            c as u32,
            required
        )
        .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    let solver =
        ExperimentalSoilNewtonOperatorGpu::new(engine, inputs(), n as u32, c as u32, required)
            .unwrap();
    assert_eq!(solver.owned_buffer_bytes(), required);
    (solver, states, vectors)
}

fn encode(engine: &Engine, solver: &ExperimentalSoilNewtonOperatorGpu, reset: bool) -> u32 {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if reset {
        solver.encode_reset(&mut encoder);
    }
    solver.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(dispatches.finish().encoded_dispatches, 6);
    u32::from_le_bytes(
        capture::read(engine, solver.status_buffer())[..4]
            .try_into()
            .unwrap(),
    )
}

fn values(engine: &Engine, buffer: &wgpu::Buffer) -> Vec<f32> {
    capture::read(engine, buffer)
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn check_operator(
    engine: &Engine,
    solver: &ExperimentalSoilNewtonOperatorGpu,
    p: &Reference,
    x: &[f64],
    v: &[f64],
    capacity: usize,
) {
    assert_eq!(encode(engine, solver, true), 0);
    compare_operator(engine, solver, p, x, v, capacity);
}

fn compare_operator(
    engine: &Engine,
    solver: &ExperimentalSoilNewtonOperatorGpu,
    p: &Reference,
    x: &[f64],
    v: &[f64],
    capacity: usize,
) {
    let residual = p.residual(x);
    let linear = Linearization::new(p, x);
    let jv = linear.product(v);
    let jtv = linear.transpose(v);
    let blocks = values(engine, solver.blocks_buffer());
    let products = values(engine, solver.products_buffer());
    let close = |actual: f32, expected: f64, label: &str| {
        assert!(
            actual.is_finite() && (actual as f64 - expected).abs() <= 8e-6 * (1.0 + expected.abs()),
            "{label}: {actual:e} != {expected:e}"
        )
    };
    for (slot, c) in p.contacts.iter().enumerate() {
        let b = &blocks[60 * c.index..60 * (c.index + 1)];
        let out = &products[16 * c.index..16 * (c.index + 1)];
        for i in 0..6 {
            close(b[i], residual[6 * slot + i], "residual");
            for j in 0..6 {
                close(
                    b[6 + 6 * i + j],
                    linear.direct[slot][i][j],
                    "direct derivative",
                );
            }
            for j in 0..3 {
                close(
                    b[42 + 3 * i + j],
                    linear.machine[slot][i][j],
                    "machine derivative",
                );
            }
            let packed = if i < 3 { i } else { i + 1 };
            close(out[packed], jv[6 * slot + i], "Jv");
            close(out[8 + packed], jtv[6 * slot + i], "transpose Jv");
        }
        for i in [3, 7, 11, 15] {
            assert_eq!(out[i].to_bits(), 0);
        }
    }
    for c in 0..capacity.max(1) {
        if p.contacts.iter().any(|contact| contact.index == c) {
            continue;
        }
        assert!(
            blocks[60 * c..60 * (c + 1)]
                .iter()
                .all(|v| v.to_bits() == 0)
        );
        assert!(
            products[16 * c..16 * (c + 1)]
                .iter()
                .all(|v| v.to_bits() == 0)
        );
    }
}

fn sample(f: &Fixture) -> (Reference, Vec<f64>, Vec<f64>) {
    let p = Reference::new(
        f,
        f.free
            .iter()
            .enumerate()
            .map(|(i, &q)| q as f64 + (i + 1) as f64 * 2f64.powi(-30))
            .collect(),
    );
    let x = (0..6 * p.contacts.len())
        .map(|i| 0.31 * ((i * 17) as f64 + 0.2).sin())
        .collect();
    let v = (0..6 * p.contacts.len())
        .map(|i| (((i * 13) as f64).cos() as f32) as f64)
        .collect();
    (p, x, v)
}

#[test]
fn newton_operator_layout_and_shader() {
    assert_eq!(std::mem::size_of::<SoilNewtonVector>(), 32);
    assert_eq!(std::mem::size_of::<SoilNewtonState>(), 64);
    assert_eq!(
        ExperimentalSoilNewtonOperatorGpu::required_bytes(32, 65).unwrap(),
        20304
    );
    for n in [0, 33] {
        assert!(ExperimentalSoilNewtonOperatorGpu::required_bytes(n, 1).is_err());
    }
    let source = format!(
        "{}\n{}",
        crate::soil_rate_encoding::arithmetic(64),
        include_str!("../shaders/soil_newton_operator.wgsl")
    );
    let module = wgpu::naga::front::wgsl::parse_str(&source)
        .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
#[ignore = "requires GPU; matrix-free contact operator and f64 oracle"]
fn newton_operator_gpu_oracle() {
    let engine = Engine::new_gpu().unwrap();
    let mut empty = pair(false, 0.0);
    empty.contacts.clear();
    empty.rows.clear();
    for f in [
        pair(false, 0.0),
        pair(true, 0.3),
        articulated(7),
        articulated(65),
        wide_fixture(),
        empty,
    ] {
        let (p, x, v) = sample(&f);
        let (solver, _, _) = create_operator(&engine, &f, &p, &x, &v);
        check_operator(&engine, &solver, &p, &x, &v, f.contacts.len());
        let first = capture::read(&engine, solver.products_buffer());
        assert_eq!(encode(&engine, &solver, false), 0);
        assert_eq!(first, capture::read(&engine, solver.products_buffer()));
        println!(
            "Newton GPU operator dofs={} contacts={} bytes={}",
            f.free.len(),
            f.contacts.len(),
            solver.owned_buffer_bytes()
        );
    }
}

#[test]
#[ignore = "requires GPU; sparse inactive slots, poisoned outputs and sticky faults"]
fn newton_operator_gpu_reuse_and_faults() {
    let engine = Engine::new_gpu().unwrap();
    let f = spread_contacts(&articulated(65), 4096, 63);
    let (p, x, v) = sample(&f);
    let (solver, states, vectors) = create_operator(&engine, &f, &p, &x, &v);
    for b in [solver.blocks_buffer(), solver.products_buffer()] {
        engine.queue().unwrap().write_buffer(
            b,
            0,
            bytemuck::cast_slice(&vec![f32::NAN; b.size() as usize / 4]),
        );
    }
    check_operator(&engine, &solver, &p, &x, &v, f.contacts.len());
    let baseline = capture::read(&engine, solver.products_buffer());
    let nan = SoilNewtonVector {
        contact: [f32::NAN; 4],
        wall: [f32::NAN; 4],
    };
    engine
        .queue()
        .unwrap()
        .write_buffer(&vectors, 32, bytemuck::bytes_of(&nan));
    engine.queue().unwrap().write_buffer(
        &states,
        64,
        bytemuck::bytes_of(&SoilNewtonState {
            high: nan,
            low: nan,
        }),
    );
    check_operator(&engine, &solver, &p, &x, &v, f.contacts.len());
    assert_eq!(baseline, capture::read(&engine, solver.products_buffer()));
    engine
        .queue()
        .unwrap()
        .write_buffer(&vectors, 0, bytemuck::bytes_of(&nan));
    assert_ne!(encode(&engine, &solver, true) & 1, 0);
    engine
        .queue()
        .unwrap()
        .write_buffer(&vectors, 0, bytemuck::bytes_of(&vector(&v[..6])));
    assert_ne!(encode(&engine, &solver, false) & 1, 0);
    check_operator(&engine, &solver, &p, &x, &v, f.contacts.len());
    engine.queue().unwrap().write_buffer(
        &states,
        0,
        bytemuck::bytes_of(&SoilNewtonState {
            high: nan,
            low: nan,
        }),
    );
    assert_ne!(encode(&engine, &solver, true) & 1, 0);
    engine
        .queue()
        .unwrap()
        .write_buffer(&states, 0, bytemuck::bytes_of(&state(&x[..6])));
    check_operator(&engine, &solver, &p, &x, &v, f.contacts.len());
    assert_eq!(baseline, capture::read(&engine, solver.products_buffer()));
}

#[test]
#[ignore = "requires GPU; paired cancellation and retained linearization products"]
fn newton_operator_gpu_paired_and_retained() {
    let engine = Engine::new_gpu().unwrap();
    let f = Fixture {
        mass: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        free: vec![1e8, 0.0, 0.0],
        rows: vec![
            pack(DVec3::X, 0.0),
            pack(DVec3::Y, 0.0),
            pack(DVec3::Z, 0.0),
        ],
        contacts: vec![SoilSharedWallContact {
            soil_mass: pack(-DVec3::X * 1e8, 1.0),
            normal_gap: pack(DVec3::X, 0.0),
            friction: [0.0; 4],
            boundary: [0; 4],
        }],
    };
    let p = Reference::new(&f, vec![1e8 + 0.25, 0.0, 0.0]);
    let x = vec![1e8 + 0.0625, 0.0, 0.0, 0.0, 0.0, 0.0];
    let v = vec![0.5, -0.75, 0.25, 0.0, 0.0, 0.0];
    let (solver, _, vectors) = create_operator(&engine, &f, &p, &x, &v);
    check_operator(&engine, &solver, &p, &x, &v, 1);
    let work = values(&engine, solver.status_buffer());
    assert_eq!(work[4] as f64 + work[5] as f64, 0.1875);
    assert_eq!(values(&engine, solver.blocks_buffer())[0], -0.03125);
    let mut v2 = v.clone();
    for x in &mut v2 {
        *x *= 2.0;
    }
    engine
        .queue()
        .unwrap()
        .write_buffer(&vectors, 0, bytemuck::bytes_of(&vector(&v2)));
    let before = capture::read(&engine, solver.blocks_buffer());
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    solver.encode_forward(&mut encoder);
    solver.encode_transpose(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(dispatches.finish().encoded_dispatches, 4);
    assert_eq!(capture::read(&engine, solver.status_buffer())[..4], [0; 4]);
    assert_eq!(before, capture::read(&engine, solver.blocks_buffer()));
    compare_operator(&engine, &solver, &p, &x, &v2, 1);
}

#[test]
#[ignore = "requires GPU and a saved actual shared-input snapshot"]
fn newton_operator_captured_input() {
    let root = std::path::PathBuf::from(std::env::var("MOLLA_SHARED_FAILURE_REPLAY").unwrap());
    let (f, pairs) = failure::load_paired(&root).unwrap();
    let p = Reference::new(&f, combined(&pairs));
    let x: Vec<_> = (0..6 * p.contacts.len())
        .map(|i| 0.31 * ((i * 17) as f64 + 0.2).sin())
        .collect();
    let v: Vec<_> = (0..6 * p.contacts.len())
        .map(|i| (((i * 13) as f64).cos() as f32) as f64)
        .collect();
    let engine = Engine::new_gpu().unwrap();
    let (solver, states, _) = create_operator(&engine, &f, &p, &x, &v);
    check_operator(&engine, &solver, &p, &x, &v, f.contacts.len());
    let outputs: Vec<SoilWallFrictionOutput> = std::fs::read(root.join("outputs.bin"))
        .unwrap()
        .chunks_exact(80)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    let actual: Vec<_> = p
        .contacts
        .iter()
        .flat_map(|c| {
            six(
                -vec(outputs[c.index].machine_impulse_normal),
                vec(outputs[c.index].wall_impulse),
            )
        })
        .collect();
    let mut input = vec![SoilNewtonState::default(); f.contacts.len().max(1)];
    for (c, x) in p.contacts.iter().zip(actual.chunks_exact(6)) {
        input[c.index] = state(x);
    }
    engine
        .queue()
        .unwrap()
        .write_buffer(&states, 0, bytemuck::cast_slice(&input));
    assert_eq!(encode(&engine, &solver, true), 0);
    let expected = p.residual(&actual);
    let blocks = values(&engine, solver.blocks_buffer());
    let mut worst = 0.0_f64;
    for (slot, c) in p.contacts.iter().enumerate() {
        for i in 0..6 {
            let error = (blocks[60 * c.index + i] as f64 - expected[6 * slot + i]).abs()
                / (1.0 + expected[6 * slot + i].abs());
            worst = worst.max(error);
            assert!(
                error <= 1e-6,
                "captured residual contact={} axis={i} error={error:e}",
                c.index
            );
        }
    }
    println!(
        "captured Newton GPU operator active={} slots={} dofs={} terminal_residual_error={worst:e}",
        p.contacts.len(),
        f.contacts.len(),
        p.free.len()
    );
}
