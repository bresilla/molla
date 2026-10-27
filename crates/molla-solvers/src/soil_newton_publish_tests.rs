use super::*;
use crate::soil_newton_publish_gpu::{ExperimentalSoilNewtonPublishGpu, SoilNewtonPublishInputs};

fn create(
    engine: &Engine,
    op: &ExperimentalSoilNewtonOperatorGpu,
    f: &Fixture,
    p: &Reference,
    state: &wgpu::Buffer,
    solver_status: &wgpu::Buffer,
) -> (ExperimentalSoilNewtonPublishGpu, wgpu::Buffer) {
    let mass = storage(engine, &f.mass);
    let pairs: Vec<_> = p
        .free
        .iter()
        .map(|q| [*q as f32, (*q - (*q as f32) as f64) as f32])
        .collect();
    let free = storage(engine, &pairs);
    let [rows, response, contacts] = op.point_inputs();
    let inputs = || SoilNewtonPublishInputs {
        mass: &mass,
        free_rates: &free,
        point_rows: rows,
        point_response: response,
        contacts,
        state,
        upstream_status: op.status_buffer(),
        solver_status,
    };
    let n = f.free.len() as u32;
    let c = f.contacts.len() as u32;
    let bytes = ExperimentalSoilNewtonPublishGpu::required_bytes(n, c).unwrap();
    let allocation = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(ExperimentalSoilNewtonPublishGpu::new(engine, inputs(), n, c, bytes - 1).is_err());
    assert_eq!(allocation.finish().observations, 0);
    let result = ExperimentalSoilNewtonPublishGpu::new(engine, inputs(), n, c, bytes).unwrap();
    assert_eq!(result.owned_buffer_bytes(), bytes);
    (result, mass)
}

fn run(engine: &Engine, publisher: &ExperimentalSoilNewtonPublishGpu, reset: bool) -> [u32; 8] {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if reset {
        publisher.encode_reset(&mut encoder);
    }
    publisher.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(dispatch.finish().encoded_dispatches, 1);
    bytemuck::pod_read_unaligned(&capture::read(engine, publisher.status_buffer())[..32])
}

fn verify(
    engine: &Engine,
    publisher: &ExperimentalSoilNewtonPublishGpu,
    f: &Fixture,
    p: &Reference,
) {
    verify_outputs(
        engine,
        publisher.outputs_buffer(),
        publisher.rates_buffer(),
        f,
        p,
    );
}

pub(super) fn verify_outputs(
    engine: &Engine,
    output_buffer: &wgpu::Buffer,
    rate_buffer: &wgpu::Buffer,
    f: &Fixture,
    p: &Reference,
) {
    let raw = capture::read(engine, output_buffer);
    let outputs: Vec<SoilWallFrictionOutput> = raw
        .chunks_exact(80)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    let pairs = values(engine, rate_buffer);
    let q: Vec<_> = pairs
        .chunks_exact(2)
        .map(|v| v[0] as f64 + v[1] as f64)
        .collect();
    verify_precise_free_tolerance(f, &p.free, &q, &outputs[..f.contacts.len()], 1e-6);
    for (c, node) in f.contacts.iter().enumerate() {
        if node.soil_mass[3] == 0.0 {
            continue;
        }
        let normal = vec(node.normal_gap);
        let soil = normal.dot(vec(outputs[c].soil_velocity_energy));
        let machine: f64 = (0..q.len())
            .map(|i| normal.dot(vec(f.rows[c * q.len() + i])) * q[i])
            .sum();
        let gap = node.normal_gap[3] as f64;
        let scale = 1.0_f64.max(soil.abs()).max(machine.abs()).max(gap);
        assert!(
            soil - machine + gap >= -1e-6 * scale,
            "publication strict normal at {c}"
        );
        for axis in 0..3 {
            let flag = (node.boundary[0] >> (2 * axis)) & 3;
            let velocity = outputs[c].soil_velocity_energy[axis];
            assert!(
                (flag != 1 || velocity >= 0.) && (flag != 2 || velocity <= 0.),
                "publication strict wall at {c} axis {axis}: {velocity}"
            );
        }
    }
}

fn rejected(engine: &Engine, publisher: &ExperimentalSoilNewtonPublishGpu) {
    assert!(
        capture::read(engine, publisher.outputs_buffer())
            .iter()
            .all(|&b| b == 0)
    );
    assert!(
        capture::read(engine, publisher.rates_buffer())
            .iter()
            .all(|&b| b == 0)
    );
}

#[test]
fn newton_publication_shader_and_budget() {
    assert_eq!(
        ExperimentalSoilNewtonPublishGpu::required_bytes(32, 4096).unwrap(),
        327984
    );
    for n in [0, 33] {
        assert!(ExperimentalSoilNewtonPublishGpu::required_bytes(n, 1).is_err());
    }
    let source = crate::soil_newton_publish_gpu::shader_source();
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
#[ignore = "requires GPU; independent publication of f64 reference solutions"]
fn newton_publication_gpu_reference() {
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
        spread_contacts(&articulated(65), 4096, 63),
        empty,
    ] {
        let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
        let (x, _, _) = solve(&p, vec![0.0; 6 * p.contacts.len()]).unwrap();
        let (op, states, _) = create_operator(&engine, &f, &p, &x, &x);
        assert_eq!(encode(&engine, &op, true), 0);
        let solver_status = storage(&engine, &[0u32]);
        let (publisher, _) = create(&engine, &op, &f, &p, &states, &solver_status);
        let status = run(&engine, &publisher, true);
        eprintln!(
            "publication reference contacts={} status={:?} energy={:e}",
            p.contacts.len(),
            &status[..4],
            f32::from_bits(status[4])
        );
        assert_eq!(status[0], 0);
        assert_eq!(status[2] as usize, p.contacts.len());
        verify(&engine, &publisher, &f, &p);
        let first = capture::read(&engine, publisher.outputs_buffer());
        for fault in [8u32, 16, 24] {
            engine
                .queue()
                .unwrap()
                .write_buffer(&solver_status, 0, bytemuck::bytes_of(&fault));
            assert_eq!(run(&engine, &publisher, true)[0], 0);
            assert_eq!(first, capture::read(&engine, publisher.outputs_buffer()));
        }
    }
}

#[test]
#[ignore = "requires GPU; publication rejection, clearing and reset"]
fn newton_publication_gpu_rejections() {
    let engine = Engine::new_gpu().unwrap();
    let f = pair(false, 0.0);
    let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
    let (x, _, _) = solve(&p, vec![0.0; 12]).unwrap();
    let (op, states, _) = create_operator(&engine, &f, &p, &x, &x);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver_status = storage(&engine, &[0u32]);
    let (publisher, mass) = create(&engine, &op, &f, &p, &states, &solver_status);
    assert_eq!(run(&engine, &publisher, true)[0], 0);
    let original = capture::read(&engine, &states);
    engine
        .queue()
        .unwrap()
        .write_buffer(&states, 0, bytemuck::bytes_of(&f32::NAN));
    assert_eq!(run(&engine, &publisher, true)[0], 1);
    rejected(&engine, &publisher);
    engine.queue().unwrap().write_buffer(&states, 0, &original);
    assert_eq!(run(&engine, &publisher, false)[0], 1);
    rejected(&engine, &publisher);
    assert_eq!(run(&engine, &publisher, true)[0], 0);
    for fault in [1u32, 2, 32, 9, 18] {
        engine
            .queue()
            .unwrap()
            .write_buffer(&solver_status, 0, bytemuck::bytes_of(&fault));
        assert_eq!(run(&engine, &publisher, true)[0], 1);
        rejected(&engine, &publisher);
    }
    engine
        .queue()
        .unwrap()
        .write_buffer(&solver_status, 0, bytemuck::bytes_of(&0u32));
    let wrong: Vec<_> = f.mass.iter().map(|v| 2.0 * v).collect();
    engine
        .queue()
        .unwrap()
        .write_buffer(&mass, 0, bytemuck::cast_slice(&wrong));
    assert_ne!(run(&engine, &publisher, true)[0] & 4, 0);
    rejected(&engine, &publisher);
    engine
        .queue()
        .unwrap()
        .write_buffer(&mass, 0, bytemuck::cast_slice(&f.mass));
    assert_eq!(run(&engine, &publisher, true)[0], 0);
    engine
        .queue()
        .unwrap()
        .write_buffer(&states, 16, bytemuck::bytes_of(&0.5f32));
    assert_ne!(run(&engine, &publisher, true)[0] & 16, 0);
    rejected(&engine, &publisher);
    engine.queue().unwrap().write_buffer(&states, 0, &original);
    assert_eq!(run(&engine, &publisher, true)[0], 0);
    verify(&engine, &publisher, &f, &p);
}

fn single(free: [f32; 3], soil: [f32; 3]) -> Fixture {
    Fixture {
        mass: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        free: free.to_vec(),
        rows: vec![
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        contacts: vec![SoilSharedWallContact {
            soil_mass: [soil[0], soil[1], soil[2], 1.0],
            normal_gap: [0.0, 1.0, 0.0, 0.0],
            friction: [0.0; 4],
            boundary: [0; 4],
        }],
    }
}

#[test]
fn newton_publication_gap_cannot_dilute_complementarity() {
    let mut f = single([0.; 3], [0.; 3]);
    let x = [0., 1e-7, 0., 0., 0., 0.];
    let zero_gap = Reference::new(&f, vec![0.; 3]);
    assert!(zero_gap.physical_error(&x) < 1e-6);
    f.contacts[0].normal_gap[3] = 100.;
    let large_gap = Reference::new(&f, vec![0.; 3]);
    let error = large_gap.physical_error(&x);
    assert!(error > 9e-6 && error < 1.1e-5);
    assert_eq!(large_gap.physical_error(&[0.; 6]), 0.);
}

#[test]
#[ignore = "requires GPU; gap-independent publication certificate and clearing"]
fn newton_publication_gpu_large_gap() {
    let engine = Engine::new_gpu().unwrap();
    for gap in [0., 100.] {
        let mut f = single([0.; 3], [0.; 3]);
        f.contacts[0].normal_gap[3] = gap;
        let p = Reference::new(&f, vec![0.; 3]);
        let x = [0., 1e-7, 0., 0., 0., 0.];
        let (op, states, _) = create_operator(&engine, &f, &p, &x, &x);
        assert_eq!(encode(&engine, &op, true), 0);
        let solver_status = storage(&engine, &[0u32]);
        let (publisher, _) = create(&engine, &op, &f, &p, &states, &solver_status);
        let report = run(&engine, &publisher, true);
        if gap == 0. {
            assert_eq!(report[0], 0);
            verify(&engine, &publisher, &f, &p);
        } else {
            assert_eq!(&report[..2], &[8, 1]);
            rejected(&engine, &publisher);
        }
        engine
            .queue()
            .unwrap()
            .write_buffer(&states, 0, bytemuck::bytes_of(&state(&[0.; 6])));
        assert_eq!(run(&engine, &publisher, true)[0], 0);
        verify(&engine, &publisher, &f, &p);
    }
}

#[test]
#[ignore = "requires GPU; six wall faces, bounded repair and recertification"]
fn newton_publication_gpu_strict_walls() {
    let engine = Engine::new_gpu().unwrap();
    for axis in 0..3 {
        for sign in [-1.0f64, 1.0] {
            let mut f = single([0.; 3], [0.; 3]);
            f.free[axis] = -sign as f32;
            f.contacts[0].normal_gap = [0.; 4];
            f.contacts[0].normal_gap[axis] = -sign as f32;
            f.contacts[0].boundary[0] = (if sign > 0. { 1 } else { 2 }) << (2 * axis);
            let p = Reference::new(&f, f.free.iter().map(|&v| v as f64).collect());
            let mut x = vec![0.; 6];
            x[axis] = -sign;
            x[3 + axis] = sign * (1. - 4.240913e-10);
            let (op, states, _) = create_operator(&engine, &f, &p, &x, &x);
            assert_eq!(encode(&engine, &op, true), 0);
            let solver_status = storage(&engine, &[0u32]);
            let (publisher, mass) = create(&engine, &op, &f, &p, &states, &solver_status);
            assert_eq!(run(&engine, &publisher, true)[0], 0);
            verify(&engine, &publisher, &f, &p);
            let output: SoilWallFrictionOutput = bytemuck::pod_read_unaligned(
                &capture::read(&engine, publisher.outputs_buffer())[..80],
            );
            assert_eq!(output.soil_velocity_energy[axis], 0.);
            assert_eq!(output.wall_impulse[axis], sign as f32);
            if axis == 1 && sign > 0. {
                let resident = super::nonlinear::create_stagnation(&engine, &op, &p, 1);
                engine.queue().unwrap().write_buffer(
                    resident.state_buffer(),
                    0,
                    bytemuck::bytes_of(&state(&x)),
                );
                let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
                let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
                let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
                let mut encoder = engine
                    .device()
                    .unwrap()
                    .create_command_encoder(&Default::default());
                resident.encode(&mut encoder);
                engine.queue().unwrap().submit([encoder.finish()]);
                assert_eq!(allocations.finish().observations, 0);
                let traffic = traffic.finish();
                assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
                assert_eq!(dispatch.finish().encoded_dispatches, 1);
                let report: [u32; 8] = bytemuck::pod_read_unaligned(
                    &capture::read(&engine, resident.status_buffer())[..32],
                );
                assert_eq!((report[0], report[1]), (0, 0));
                assert_eq!(
                    capture::read(&engine, resident.outputs_buffer()),
                    capture::read(&engine, publisher.outputs_buffer())
                );
                assert_eq!(
                    capture::read(&engine, resident.rates_buffer()),
                    capture::read(&engine, publisher.rates_buffer())
                );
                verify_outputs(
                    &engine,
                    resident.outputs_buffer(),
                    resident.rates_buffer(),
                    &f,
                    &p,
                );
            }
            let exact = capture::read(&engine, &states);
            x[3 + axis] = sign * (1. + 4.240913e-10);
            engine
                .queue()
                .unwrap()
                .write_buffer(&states, 0, bytemuck::bytes_of(&state(&x)));
            assert_eq!(run(&engine, &publisher, true)[0], 0);
            verify(&engine, &publisher, &f, &p);
            let output: SoilWallFrictionOutput = bytemuck::pod_read_unaligned(
                &capture::read(&engine, publisher.outputs_buffer())[..80],
            );
            assert!(sign * output.soil_velocity_energy[axis] as f64 > 0.);
            x[3 + axis] = sign * 0.99;
            engine
                .queue()
                .unwrap()
                .write_buffer(&states, 0, bytemuck::bytes_of(&state(&x)));
            assert_ne!(run(&engine, &publisher, true)[0] & 16, 0);
            rejected(&engine, &publisher);
            engine.queue().unwrap().write_buffer(&states, 0, &exact);
            assert_ne!(run(&engine, &publisher, false)[0], 0);
            rejected(&engine, &publisher);
            engine.queue().unwrap().write_buffer(
                &mass,
                4 * (4 * axis) as u64,
                bytemuck::bytes_of(&2.0f32),
            );
            assert_ne!(run(&engine, &publisher, true)[0] & 4, 0);
            rejected(&engine, &publisher);
            engine.queue().unwrap().write_buffer(
                &mass,
                4 * (4 * axis) as u64,
                bytemuck::bytes_of(&1.0f32),
            );
            assert_eq!(run(&engine, &publisher, true)[0], 0);
            verify(&engine, &publisher, &f, &p);
        }
    }
}

#[test]
#[ignore = "requires GPU; independent rejection on every machine momentum lane"]
fn newton_publication_gpu_momentum_lanes() {
    let engine = Engine::new_gpu().unwrap();
    let mut f = single([0.0; 3], [0.0; 3]);
    f.free = vec![1.0; 32];
    f.mass = vec![0.0; 32 * 32];
    for i in 0..32 {
        f.mass[32 * i + i] = 1.0;
    }
    f.rows = vec![[0.0, 1.0 / 32.0, 0.0, 0.0]; 32];
    let p = Reference::new(&f, vec![1.0; 32]);
    let x = vec![0.0, 32.0 / 33.0, 0.0, 0.0, 0.0, 0.0];
    let (op, states, _) = create_operator(&engine, &f, &p, &x, &x);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver_status = storage(&engine, &[0u32]);
    let (publisher, mass) = create(&engine, &op, &f, &p, &states, &solver_status);
    assert_eq!(run(&engine, &publisher, true)[0], 0);
    verify(&engine, &publisher, &f, &p);
    for lane in 0..32u64 {
        let offset = 4 * (32 * lane + lane);
        engine
            .queue()
            .unwrap()
            .write_buffer(&mass, offset, bytemuck::bytes_of(&2.0f32));
        let status = run(&engine, &publisher, true);
        assert_eq!(status[0], 4, "momentum lane {lane}");
        rejected(&engine, &publisher);
        engine
            .queue()
            .unwrap()
            .write_buffer(&mass, offset, bytemuck::bytes_of(&1.0f32));
    }
    assert_eq!(run(&engine, &publisher, true)[0], 0);
    verify(&engine, &publisher, &f, &p);
}

#[test]
#[ignore = "requires GPU; strict normal scale and paired energy rejection"]
fn newton_publication_gpu_strict_and_energy() {
    let engine = Engine::new_gpu().unwrap();
    for (f, x, expected) in [
        (
            single([1e8, 0.0, 0.0], [0.0, -2e-6, 0.0]),
            vec![0.0; 6],
            8u32,
        ),
        (
            single([0.0, -1e8, 0.0], [0.0; 3]),
            vec![0.0, 1e-7, 0.0, 0.0, 0.0, 0.0],
            32,
        ),
    ] {
        let p = Reference::new(&f, f.free.iter().map(|&q| q as f64).collect());
        let (op, states, _) = create_operator(&engine, &f, &p, &x, &x);
        assert_eq!(encode(&engine, &op, true), 0);
        let solver_status = storage(&engine, &[0u32]);
        let (publisher, _) = create(&engine, &op, &f, &p, &states, &solver_status);
        let status = run(&engine, &publisher, true);
        eprintln!(
            "publication adversarial status={:?} energy={:e}",
            &status[..4],
            f32::from_bits(status[4])
        );
        assert_eq!(status[0], expected);
        rejected(&engine, &publisher);
    }
}

#[test]
#[ignore = "requires GPU; opposing represented impulses and retained low-word velocity"]
fn newton_publication_gpu_cancellation() {
    let engine = Engine::new_gpu().unwrap();
    let mut f = single([0.0; 3], [0.0, -0.25, 0.0]);
    f.rows.fill([0.0; 4]);
    f.contacts[0].normal_gap = [0.0, -1.0, 0.0, 0.0];
    f.contacts[0].boundary = [4, 0, 0, 0];
    let p = Reference::new(&f, vec![0.0; 3]);
    let x = vec![0.0, -1e8 + 0.25, 0.0, 0.0, 1e8, 0.0];
    let (op, states, _) = create_operator(&engine, &f, &p, &x, &x);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver_status = storage(&engine, &[16u32]);
    let (publisher, _) = create(&engine, &op, &f, &p, &states, &solver_status);
    let status = run(&engine, &publisher, true);
    assert_eq!(status[0], 0);
    verify(&engine, &publisher, &f, &p);
    let output: SoilWallFrictionOutput =
        bytemuck::pod_read_unaligned(&capture::read(&engine, publisher.outputs_buffer())[..80]);
    assert_eq!(output.soil_velocity_energy[1], 0.0);
    assert_eq!(output.soil_velocity_energy[3], -0.03125);
    engine
        .queue()
        .unwrap()
        .write_buffer(&states, 36, bytemuck::bytes_of(&0f32));
    assert_ne!(run(&engine, &publisher, true)[0] & 16, 0);
    rejected(&engine, &publisher);
}

#[test]
#[ignore = "requires GPU and saved shared input snapshot"]
fn newton_publication_gpu_captured_input() {
    let root = std::path::PathBuf::from(std::env::var("MOLLA_SHARED_FAILURE_REPLAY").unwrap());
    let (f, pairs) = failure::load_paired(&root).unwrap();
    let p = Reference::new(&f, combined(&pairs));
    let engine = Engine::new_gpu().unwrap();
    let zero = vec![0.0; 6 * p.contacts.len()];
    let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver = super::nonlinear::create(&engine, &op, &p, f.contacts.len());
    let (publisher, _) = create(
        &engine,
        &op,
        &f,
        &p,
        solver.state_buffer(),
        solver.status_buffer(),
    );
    let allocation = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    solver.encode_reset(&mut encoder);
    solver.encode(&mut encoder);
    publisher.encode_reset(&mut encoder);
    publisher.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocation.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(dispatch.finish().encoded_dispatches, 2);
    let nonlinear: [u32; 8] =
        bytemuck::pod_read_unaligned(&capture::read(&engine, solver.status_buffer())[..32]);
    eprintln!(
        "captured certified Newton status={:?} residual={:e} physical_estimate={:e}",
        &nonlinear[..4],
        f32::from_bits(nonlinear[4]),
        f32::from_bits(nonlinear[5])
    );
    assert_eq!(nonlinear[0], 0);
    let status: [u32; 8] =
        bytemuck::pod_read_unaligned(&capture::read(&engine, publisher.status_buffer())[..32]);
    eprintln!(
        "captured publication status={:?} energy={:e} scale={:e}",
        &status[..4],
        f32::from_bits(status[4]),
        f32::from_bits(status[5])
    );
    assert_eq!(status[0], 0);
    verify(&engine, &publisher, &f, &p);
    assert_eq!(
        capture::read(&engine, solver.outputs_buffer()),
        capture::read(&engine, publisher.outputs_buffer())
    );
    assert_eq!(
        capture::read(&engine, solver.rates_buffer()),
        capture::read(&engine, publisher.rates_buffer())
    );
}

#[test]
#[ignore = "requires GPU timestamps and saved input; frozen-solve timing, not full-frame acceptance"]
fn newton_publication_gpu_capture_timing() {
    let root = std::path::PathBuf::from(std::env::var("MOLLA_SHARED_FAILURE_REPLAY").unwrap());
    let (f, pairs) = failure::load_paired(&root).unwrap();
    let p = Reference::new(&f, combined(&pairs));
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )
    .unwrap();
    let zero = vec![0.0; 6 * p.contacts.len()];
    let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
    assert_eq!(encode(&engine, &op, true), 0);
    let solver = super::nonlinear::create(&engine, &op, &p, f.contacts.len());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let count = 8u32;
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("frozen Newton solve timing"),
        ty: wgpu::QueryType::Timestamp,
        count: 2 * count,
    });
    let ticks = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frozen Newton timestamps"),
        size: 16 * u64::from(count),
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let statuses = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("frozen Newton statuses"),
        size: 32 * u64::from(count),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    for _ in 0..2 {
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_reset(&mut encoder);
        solver.encode(&mut encoder);
        queue.submit([encoder.finish()]);
        assert_eq!(
            u32::from_le_bytes(
                capture::read(&engine, solver.status_buffer())[..4]
                    .try_into()
                    .unwrap()
            ),
            0
        );
    }
    let allocation = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    for i in 0..count {
        encoder.write_timestamp(&queries, 2 * i);
        solver.encode_reset(&mut encoder);
        solver.encode(&mut encoder);
        encoder.write_timestamp(&queries, 2 * i + 1);
        encoder.copy_buffer_to_buffer(solver.status_buffer(), 0, &statuses, 32 * u64::from(i), 32);
    }
    encoder.resolve_query_set(&queries, 0..2 * count, &ticks, 0);
    queue.submit([encoder.finish()]);
    assert_eq!(allocation.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(dispatch.finish().encoded_dispatches, u64::from(count));
    let raw = capture::read(&engine, &ticks);
    let raw_status = capture::read(&engine, &statuses);
    let mut samples = Vec::new();
    for (i, pair) in raw.chunks_exact(16).enumerate() {
        let start = u64::from_le_bytes(pair[..8].try_into().unwrap());
        let end = u64::from_le_bytes(pair[8..].try_into().unwrap());
        assert!(end > start);
        let ms = (end - start) as f64 * f64::from(queue.get_timestamp_period()) / 1e6;
        samples.push(ms);
        let status: [u32; 8] = bytemuck::pod_read_unaligned(&raw_status[32 * i..32 * (i + 1)]);
        assert_eq!(status[0], 0);
        eprintln!(
            "frozen Newton sample={i} gpu_ms={ms:.6} steps={} linear={} trials={}",
            status[1], status[2], status[3]
        );
    }
    samples.sort_by(f64::total_cmp);
    eprintln!(
        "frozen Newton contacts={} median_gpu_ms={:.6} max_gpu_ms={:.6}; excludes preparation, particles, rendering",
        p.contacts.len(),
        0.5 * (samples[3] + samples[4]),
        samples[7]
    );
    verify_outputs(
        &engine,
        solver.outputs_buffer(),
        solver.rates_buffer(),
        &f,
        &p,
    );
}

#[test]
#[ignore = "requires native FP64, timestamps and saved input; paired frozen solve timing"]
fn newton_native_f64_gpu_paired_timing() {
    paired_arithmetic_timing(true, false, false, false);
}

#[test]
#[ignore = "requires GPU timestamps and saved input; sorted-pair frozen solve timing"]
fn newton_cooperative_pairs_gpu_paired_timing() {
    paired_arithmetic_timing(false, false, false, false);
}

#[test]
#[ignore = "requires GPU timestamps and saved input; chunked product paired timing"]
fn newton_chunked_pairs_gpu_paired_timing() {
    paired_arithmetic_timing(false, true, false, false);
}

#[test]
#[ignore = "requires GPU timestamps and saved input; stagnation fallback paired timing"]
fn newton_stagnation_pairs_gpu_paired_timing() {
    paired_arithmetic_timing(false, false, true, false);
}

#[test]
#[ignore = "requires GPU timestamps and saved input; parallel publication paired timing"]
fn newton_parallel_publication_gpu_paired_timing() {
    paired_arithmetic_timing(false, false, true, true);
}

fn paired_arithmetic_timing(native: bool, chunked: bool, stagnation: bool, publication: bool) {
    eprintln!(
        "paired arithmetic candidate={}",
        if publication {
            "parallel versus serial publication momentum"
        } else if native {
            "native-f64 versus paired"
        } else if stagnation {
            "stagnation fallback versus chunked projections"
        } else if chunked {
            "chunked projections versus cooperative serial projections"
        } else {
            "sorted cooperative reduction versus sorted cached serial reduction"
        }
    );
    let root = std::path::PathBuf::from(std::env::var("MOLLA_SHARED_FAILURE_REPLAY").unwrap());
    let (f, pairs) = failure::load_paired(&root).unwrap();
    let p = Reference::new(&f, combined(&pairs));
    let engine = Engine::new_gpu_with_features(
        (if native {
            wgpu::Features::SHADER_F64
        } else {
            wgpu::Features::empty()
        }) | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )
    .unwrap();
    let zero = vec![0.0; 6 * p.contacts.len()];
    let (op, _, _) = create_operator(&engine, &f, &p, &zero, &zero);
    assert_eq!(encode(&engine, &op, true), 0);
    let mut solvers = [
        if publication {
            super::nonlinear::create_stagnation(&engine, &op, &p, f.contacts.len())
        } else if native {
            super::nonlinear::create(&engine, &op, &p, f.contacts.len())
        } else if stagnation {
            super::nonlinear::create_chunked(&engine, &op, &p, f.contacts.len())
        } else if chunked {
            super::nonlinear::create_cooperative(&engine, &op, &p, f.contacts.len())
        } else {
            super::nonlinear::create_sorted(&engine, &op, &p, f.contacts.len())
        },
        if native {
            super::nonlinear::create_native(&engine, &op, &p, f.contacts.len())
        } else if stagnation {
            super::nonlinear::create_stagnation(&engine, &op, &p, f.contacts.len())
        } else if chunked {
            super::nonlinear::create_chunked(&engine, &op, &p, f.contacts.len())
        } else {
            super::nonlinear::create_cooperative(&engine, &op, &p, f.contacts.len())
        },
    ];
    if publication {
        solvers[0].use_serial_publication(&engine);
    }
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for _ in 0..2 {
        for solver in &solvers {
            let mut encoder = device.create_command_encoder(&Default::default());
            solver.encode_reset(&mut encoder);
            solver.encode(&mut encoder);
            queue.submit([encoder.finish()]);
            let status: [u32; 8] =
                bytemuck::pod_read_unaligned(&capture::read(&engine, solver.status_buffer())[..32]);
            eprintln!("paired arithmetic warmup status={:?}", &status[..4]);
            assert_eq!(status[0], 0);
            assert_eq!(
                u32::from_le_bytes(
                    capture::read(&engine, solver.publication_status_buffer())[..4]
                        .try_into()
                        .unwrap()
                ),
                0
            );
            verify_outputs(
                &engine,
                solver.outputs_buffer(),
                solver.rates_buffer(),
                &f,
                &p,
            );
        }
    }
    if publication {
        for (a, b) in [
            (solvers[0].outputs_buffer(), solvers[1].outputs_buffer()),
            (solvers[0].rates_buffer(), solvers[1].rates_buffer()),
            (solvers[0].status_buffer(), solvers[1].status_buffer()),
            (
                solvers[0].publication_status_buffer(),
                solvers[1].publication_status_buffer(),
            ),
        ] {
            assert_eq!(capture::read(&engine, a), capture::read(&engine, b));
        }
    }
    let count = 16u32;
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("paired native Newton timing"),
        ty: wgpu::QueryType::Timestamp,
        count: 2 * count,
    });
    let ticks = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paired Newton timestamps"),
        size: 16 * u64::from(count),
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let statuses = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paired Newton certificates"),
        size: 64 * u64::from(count),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    for pair in 0..count / 2 {
        for order in 0..2u32 {
            let i = 2 * pair + order;
            let solver = &solvers[((pair + order) % 2) as usize];
            encoder.write_timestamp(&queries, 2 * i);
            solver.encode_reset(&mut encoder);
            solver.encode(&mut encoder);
            encoder.write_timestamp(&queries, 2 * i + 1);
            encoder.copy_buffer_to_buffer(
                solver.status_buffer(),
                0,
                &statuses,
                64 * u64::from(i),
                32,
            );
            encoder.copy_buffer_to_buffer(
                solver.publication_status_buffer(),
                0,
                &statuses,
                64 * u64::from(i) + 32,
                32,
            );
        }
    }
    encoder.resolve_query_set(&queries, 0..2 * count, &ticks, 0);
    queue.submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(dispatch.finish().encoded_dispatches, u64::from(count));
    let ticks = capture::read(&engine, &ticks);
    let statuses = capture::read(&engine, &statuses);
    let mut samples = [Vec::new(), Vec::new()];
    for (i, tick) in ticks.chunks_exact(16).enumerate() {
        let start = u64::from_le_bytes(tick[..8].try_into().unwrap());
        let end = u64::from_le_bytes(tick[8..].try_into().unwrap());
        assert!(end > start);
        let ms = (end - start) as f64 * f64::from(queue.get_timestamp_period()) / 1e6;
        let variant = (i / 2 + i % 2) % 2;
        let status: [u32; 16] = bytemuck::pod_read_unaligned(&statuses[64 * i..64 * (i + 1)]);
        assert_eq!((status[0], status[8]), (0, 0));
        samples[variant].push(ms);
        eprintln!(
            "paired arithmetic sample={i} variant={variant} ms={ms:.6} steps={} linear={} trials={}",
            status[1], status[2], status[3]
        );
    }
    for (variant, samples) in samples.iter_mut().enumerate() {
        samples.sort_by(f64::total_cmp);
        eprintln!(
            "paired arithmetic contacts={} variant={variant} median_ms={:.6} max_ms={:.6}",
            p.contacts.len(),
            0.5 * (samples[3] + samples[4]),
            samples[7]
        );
    }
    for solver in &solvers {
        verify_outputs(
            &engine,
            solver.outputs_buffer(),
            solver.rates_buffer(),
            &f,
            &p,
        );
    }
}
