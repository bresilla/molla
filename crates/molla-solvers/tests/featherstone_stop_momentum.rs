use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder};
use molla_solvers::{FeatherstoneSolver, Solver};

#[cfg(feature = "gpu")]
#[path = "common/gpu_readback.rs"]
mod gpu_readback;

fn model(rotational: bool, sign: f64) -> Model {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.begin_articulation();
    let a = builder.add_body(
        BodyParams::new()
            .with_mass(2.0)
            .with_inertia(Mat3::IDENTITY * 4.0),
    );
    let b = builder.add_body(
        BodyParams::new()
            .with_mass(3.0)
            .with_inertia(Mat3::IDENTITY * 7.0),
    );
    builder
        .add_joint_free(BodyId::NONE, a, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let joint = if rotational {
        builder.add_joint_revolute(a, b, Transform::IDENTITY, Transform::IDENTITY, Vec3::X)
    } else {
        builder.add_joint_prismatic(a, b, Transform::IDENTITY, Transform::IDENTITY, Vec3::X)
    }
    .unwrap();
    let (lo, hi) = if sign > 0.0 { (0.0, 1.0) } else { (-1.0, 0.0) };
    builder.set_joint_limits(joint, &[lo], &[hi]).unwrap();
    builder.build().unwrap()
}

fn check(
    rotational: bool,
    sign: f64,
    q: f64,
    dt: f64,
    gpu: bool,
    output: &[f64],
    position: f64,
) -> bool {
    let (a, b, root) = if rotational {
        (4.0, 7.0, 3)
    } else {
        (2.0, 3.0, 0)
    };
    let initial_velocity = if q == 0.0 { 0.0 } else { -8.0 * sign };
    let expected = b * initial_velocity;
    let momentum = (a + b) * output[root] + b * output[6];
    let relative = initial_velocity - sign * 8.0 * (1.0 / a + 1.0 / b) * dt;
    let bounded = (sign * relative).max(-q / dt) * sign;
    let tolerance = if gpu { 3e-5 } else { 1e-10 };
    let valid = (momentum - expected).abs() < tolerance
        && (output[6] - bounded).abs() < tolerance
        && (position - (q * sign + bounded * dt)).abs() < tolerance;
    eprintln!(
        "stop rotational={rotational} sign={sign} q={q} dt={dt} gpu={gpu}: P={momentum} expected={expected}, v={} expected_v={bounded}, position={position}, valid={valid}",
        output[6]
    );
    valid
}

#[test]
fn cpu_scalar_stops_preserve_internal_momentum() {
    let mut failed = 0;
    for rotational in [false, true] {
        for sign in [-1.0, 1.0] {
            let model = model(rotational, sign);
            for q in [0.0, 0.1, 0.13, 0.07] {
                for dt in [0.03, 0.1] {
                    let mut initial = model.state().unwrap();
                    initial.joint_q.host_mut().unwrap()[7] = q * sign;
                    initial.joint_qd.host_mut().unwrap()[6] =
                        if q == 0.0 { 0.0 } else { -8.0 * sign };
                    molla_sim::eval_fk(&model, &mut initial).unwrap();
                    let mut control = model.control();
                    control.joint_force.host_mut().unwrap()[6] = -8.0 * sign;
                    let mut output = model.state().unwrap();
                    FeatherstoneSolver::new_cpu()
                        .with_rigorous_d_solve(true)
                        .with_substeps(1)
                        .step(
                            &model,
                            &initial,
                            &control,
                            &Contacts::with_capacity(0),
                            dt,
                            &mut output,
                        )
                        .unwrap();
                    failed += usize::from(!check(
                        rotational,
                        sign,
                        q,
                        dt,
                        false,
                        output.joint_qd.host().unwrap(),
                        output.joint_q.host().unwrap()[7],
                    ));
                }
            }
        }
    }
    assert_eq!(failed, 0, "scalar stops changed internal momentum");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_scalar_stops_preserve_internal_momentum() {
    let engine = molla_compute::Engine::new(molla_compute::Backend::AutoGpu).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut failed = 0;
    for rotational in [false, true] {
        for sign in [-1.0, 1.0] {
            let mut model = model(rotational, sign);
            model.upload_to_device(device, queue).unwrap();
            let mut solver = FeatherstoneSolver::new_gpu(&engine, &model)
                .unwrap()
                .with_rigorous_d_solve(true)
                .with_substeps(1);
            for q in [0.0, 0.1, 0.13, 0.07] {
                for dt in [0.03, 0.1] {
                    let mut initial = model.state().unwrap();
                    initial.joint_q.host_mut().unwrap()[7] = q * sign;
                    initial.joint_qd.host_mut().unwrap()[6] =
                        if q == 0.0 { 0.0 } else { -8.0 * sign };
                    molla_sim::eval_fk(&model, &mut initial).unwrap();
                    let mut control = model.control();
                    control.joint_force.host_mut().unwrap()[6] = -8.0 * sign;
                    let mut output = model.state().unwrap();
                    initial.upload_to_device(device, queue).unwrap();
                    output.upload_to_device(device, queue).unwrap();
                    control.upload_to_device(device, queue).unwrap();
                    solver
                        .step(
                            &model,
                            &initial,
                            &control,
                            &Contacts::with_capacity(0),
                            dt,
                            &mut output,
                        )
                        .unwrap();
                    assert_eq!(
                        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1),
                        [0]
                    );
                    let velocities = gpu_readback::read::<f32>(
                        &engine,
                        output.joint_qd.device_buffer().unwrap(),
                        7,
                    )
                    .into_iter()
                    .map(f64::from)
                    .collect::<Vec<_>>();
                    let position = gpu_readback::read::<f32>(
                        &engine,
                        output.joint_q.device_buffer().unwrap(),
                        8,
                    )[7] as f64;
                    failed +=
                        usize::from(!check(rotational, sign, q, dt, true, &velocities, position));
                    assert!(output.joint_qd.host().is_err());
                }
            }
        }
    }
    assert_eq!(failed, 0, "GPU scalar stops changed internal momentum");
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_coupled_stops_transfer_impacts_and_release() {
    let engine = molla_compute::Engine::new(molla_compute::Backend::AutoGpu).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for masses in [[2.0, 3.0, 5.0], [1000.0, 1.0, 1000.0]] {
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        builder.begin_articulation();
        let mut parent = BodyId::NONE;
        for (index, mass) in masses.into_iter().enumerate() {
            let body = builder.add_body(
                BodyParams::new()
                    .with_mass(mass)
                    .with_inertia(Mat3::IDENTITY),
            );
            if index == 0 {
                builder
                    .add_joint_free(parent, body, Transform::IDENTITY, Transform::IDENTITY)
                    .unwrap();
            } else {
                let joint = builder
                    .add_joint_prismatic(
                        parent,
                        body,
                        Transform::IDENTITY,
                        Transform::IDENTITY,
                        Vec3::X,
                    )
                    .unwrap();
                builder.set_joint_limits(joint, &[0.0], &[1.0]).unwrap();
            }
            parent = body;
        }
        let mut model = builder.build().unwrap();
        let mut initial = model.state().unwrap();
        initial.joint_qd.host_mut().unwrap()[0] = 1.0;
        initial.joint_qd.host_mut().unwrap()[6] = -3.0;
        initial.joint_qd.host_mut().unwrap()[7] = -2.0;
        let mut output = model.state().unwrap();
        let mut control = model.control();
        model.upload_to_device(device, queue).unwrap();
        initial.upload_to_device(device, queue).unwrap();
        output.upload_to_device(device, queue).unwrap();
        control.upload_to_device(device, queue).unwrap();
        let mut solver = FeatherstoneSolver::new_gpu(&engine, &model)
            .unwrap()
            .with_substeps(1);
        let dt = 1.0 / 120.0;
        solver
            .step(
                &model,
                &initial,
                &control,
                &Contacts::with_capacity(0),
                dt,
                &mut output,
            )
            .unwrap();
        let v = gpu_readback::read::<f32>(&engine, output.joint_qd.device_buffer().unwrap(), 8);
        assert_eq!(
            gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1),
            [0],
            "{v:?}"
        );
        let body_v = [
            v[0] as f64,
            (v[0] + v[6]) as f64,
            (v[0] + v[6] + v[7]) as f64,
        ];
        let total = masses.iter().sum::<f64>();
        let before = masses[0] - 2.0 * masses[1] - 4.0 * masses[2];
        let after = body_v.iter().zip(masses).map(|(v, m)| v * m).sum::<f64>();
        let energy = body_v
            .iter()
            .zip(masses)
            .map(|(v, m)| 0.5 * v * v * m)
            .sum::<f64>();
        let initial_energy = 0.5 * (masses[0] + 4.0 * masses[1] + 16.0 * masses[2]);
        eprintln!(
            "coupled stops masses={masses:?}, velocities={body_v:?}, momentum before={before}, after={after}"
        );
        assert!((after - before).abs() < 1e-5 * total);
        assert!(body_v.iter().all(|v| (v - before / total).abs() < 1e-4));
        assert!(energy < initial_energy - 1.0);
        control.joint_force.host_mut().unwrap()[6] = 10.0;
        control.joint_force.host_mut().unwrap()[7] = 10.0;
        control.upload_to_device(device, queue).unwrap();
        solver
            .step(
                &model,
                &output,
                &control,
                &Contacts::with_capacity(0),
                dt,
                &mut initial,
            )
            .unwrap();
        assert_eq!(
            gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1),
            [0]
        );
        let q = gpu_readback::read::<f32>(&engine, initial.joint_q.device_buffer().unwrap(), 9);
        assert!(q[7] > 0.0 && q[8] > 0.0, "stops failed to release: {q:?}");
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_stop_drift_uses_corrected_velocity_and_latches_faults() {
    let engine = molla_compute::Engine::new(molla_compute::Backend::AutoGpu).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = model(false, 1.0);
    let mut initial = model.state().unwrap();
    initial.joint_q.host_mut().unwrap()[7] = 0.1;
    initial.joint_qd.host_mut().unwrap().fill(99.0);
    let mut output = model.state().unwrap();
    output.joint_qd.host_mut().unwrap()[0] = 2.0;
    output.joint_qd.host_mut().unwrap()[6] = -8.0;
    let mut control = model.control();
    control.joint_force.host_mut().unwrap().fill(10000.0);
    model.upload_to_device(device, queue).unwrap();
    initial.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    solver
        .drift_gpu(&model, &initial, &control, 0.1, &mut output)
        .unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1),
        [0]
    );
    let v = gpu_readback::read::<f32>(&engine, output.joint_qd.device_buffer().unwrap(), 7);
    assert!(
        (v[0] + 2.2).abs() < 3e-5 && (v[6] + 1.0).abs() < 3e-5,
        "{v:?}"
    );
    assert!((5.0 * v[0] + 3.0 * v[6] + 14.0).abs() < 3e-5);
    queue.write_buffer(
        output.joint_qd.device_buffer().unwrap(),
        24,
        bytemuck::bytes_of(&f32::NAN),
    );
    solver
        .drift_gpu(&model, &initial, &control, 0.1, &mut output)
        .unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1),
        [32]
    );
    queue.write_buffer(
        output.joint_qd.device_buffer().unwrap(),
        24,
        bytemuck::bytes_of(&0.0_f32),
    );
    solver
        .drift_gpu(&model, &initial, &control, 0.1, &mut output)
        .unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1),
        [32]
    );
    assert!(output.joint_qd.host().is_err());
}

#[test]
fn cpu_stop_drift_preserves_contact_corrected_momentum() {
    let model = model(false, 1.0);
    let mut initial = model.state().unwrap();
    initial.joint_q.host_mut().unwrap()[7] = 0.1;
    initial.joint_qd.host_mut().unwrap().fill(99.0);
    let mut output = model.state().unwrap();
    output.joint_qd.host_mut().unwrap()[0] = 2.0;
    output.joint_qd.host_mut().unwrap()[6] = -8.0;
    FeatherstoneSolver::new_cpu()
        .drift_cpu(&model, &initial, 0.1, &mut output)
        .unwrap();
    let v = output.joint_qd.host().unwrap();
    assert!((v[0] + 2.2).abs() < 1e-10 && (v[6] + 1.0).abs() < 1e-10);
    assert!((5.0 * v[0] + 3.0 * v[6] + 14.0).abs() < 1e-10);
    assert!(output.joint_q.host().unwrap()[7].abs() < 1e-10);
}
