#![cfg(feature = "gpu")]

use molla_compute::{Backend, Engine};
use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, JointTargetMode, Model, ModelBuilder};
use molla_solvers::{FeatherstoneSolver, Solver};

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod gpu_readback;

fn model(rotational: bool, stops: bool) -> Model {
    let mut b = ModelBuilder::new();
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    b.begin_articulation();
    let root = b.add_body(
        BodyParams::new()
            .with_mass(2.0)
            .with_inertia(Mat3::IDENTITY * 4.0),
    );
    let child = b.add_body(
        BodyParams::new()
            .with_mass(3.0)
            .with_inertia(Mat3::IDENTITY * 0.168460975),
    );
    b.add_joint_free(BodyId::NONE, root, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let joint = if rotational {
        b.add_joint_revolute(
            root,
            child,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::X,
        )
    } else {
        b.add_joint_prismatic(
            root,
            child,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::X,
        )
    }
    .unwrap();
    if stops {
        b.set_joint_limits(joint, &[-0.02], &[0.02]).unwrap();
    }
    b.build().unwrap()
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn stable_feedback_matches_cpu_caps_and_coupled_momentum() {
    let engine = Engine::new(Backend::AutoGpu).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for rotational in [false, true] {
        for stops in [false, true] {
            let mut model = model(rotational, stops);
            model.upload_to_device(device, queue).unwrap();
            let mut cpu = FeatherstoneSolver::new_cpu()
                .with_rigorous_d_solve(true)
                .with_stable_motor_feedback(true);
            let mut gpu = FeatherstoneSolver::new_gpu(&engine, &model)
                .unwrap()
                .with_rigorous_d_solve(true)
                .with_stable_motor_feedback(true);
            let (a, b, root) = if rotational {
                (4.0, 0.168460975, 3)
            } else {
                (2.0, 3.0, 0)
            };
            for mode in [
                JointTargetMode::Position,
                JointTargetMode::Velocity,
                JointTargetMode::PositionVelocity,
                JointTargetMode::Effort,
            ] {
                for (cap, feedforward, total) in [
                    (20000.0, 0.0, f64::INFINITY),
                    (10.0, 25.0, 30.0),
                    (0.0, -25.0, 30.0),
                ] {
                    for h in [1.0 / 19200.0, 0.003] {
                        model.joint_effort_limit.host_mut().unwrap()[6] = total;
                        model.joint_effort_limit.upload(queue).unwrap();
                        let mut control = model.control();
                        control.joint_target_mode.host_mut().unwrap()[6] = mode as u32;
                        control.joint_target_pos.host_mut().unwrap()[6] = 0.2;
                        control.joint_target_vel.host_mut().unwrap()[6] = -0.1;
                        control.joint_target_kp.host_mut().unwrap()[6] = 572957.795;
                        control.joint_target_kd.host_mut().unwrap()[6] = 8594.367;
                        control.joint_motor_max_force.host_mut().unwrap()[6] = cap;
                        control.joint_force.host_mut().unwrap()[6] = feedforward;
                        let mut initial = model.state().unwrap();
                        initial.joint_q.host_mut().unwrap()[7] = if stops { 0.019 } else { 0.0 };
                        initial.joint_qd.host_mut().unwrap()[6] = 0.4;
                        initial.joint_qd.host_mut().unwrap()[root] = -b * 0.4 / (a + b);
                        molla_sim::eval_fk(&model, &mut initial).unwrap();
                        molla_sim::eval_body_velocities(&model, &mut initial).unwrap();
                        let mut expected = model.state().unwrap();
                        cpu.step(
                            &model,
                            &initial,
                            &control,
                            &Contacts::with_capacity(0),
                            h,
                            &mut expected,
                        )
                        .unwrap();
                        let mut output = model.state().unwrap();
                        initial.upload_to_device(device, queue).unwrap();
                        output.upload_to_device(device, queue).unwrap();
                        control.upload_to_device(device, queue).unwrap();
                        gpu.step(
                            &model,
                            &initial,
                            &control,
                            &Contacts::with_capacity(0),
                            h,
                            &mut output,
                        )
                        .unwrap();
                        let actual = gpu_readback::read::<f32>(
                            &engine,
                            output.joint_qd.device_buffer().unwrap(),
                            7,
                        );
                        let q = gpu_readback::read::<f32>(
                            &engine,
                            output.joint_q.device_buffer().unwrap(),
                            8,
                        );
                        assert_eq!(
                            gpu_readback::read::<u32>(&engine, gpu.gpu_status_buffer().unwrap(), 1)
                                [0],
                            0
                        );
                        for (index, (&actual, &expected)) in actual
                            .iter()
                            .zip(expected.joint_qd.host().unwrap())
                            .enumerate()
                        {
                            assert!(
                                (actual as f64 - expected).abs() < 2e-4,
                                "rot={rotational} stops={stops} mode={mode:?} cap={cap} h={h} dof={index}: GPU={actual} CPU={expected}"
                            );
                        }
                        assert!((q[7] as f64 - expected.joint_q.host().unwrap()[7]).abs() < 2e-6);
                        let momentum = (a + b) * actual[root] as f64 + b * actual[6] as f64;
                        assert!(momentum.abs() < 2e-4, "coupled momentum={momentum}");
                        assert!(output.joint_qd.host().is_err());
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn stiff_coupled_feedback_trajectory_stays_bounded_on_gpu() {
    let engine = Engine::new(Backend::AutoGpu).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.begin_articulation();
    let root = builder.add_body(
        BodyParams::new()
            .with_mass(2.0)
            .with_inertia(Mat3::IDENTITY * 4.0),
    );
    builder
        .add_joint_free(BodyId::NONE, root, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    for _ in 0..2 {
        let child = builder.add_body(
            BodyParams::new()
                .with_mass(3.0)
                .with_inertia(Mat3::IDENTITY * 0.168460975),
        );
        builder
            .add_joint_revolute(
                root,
                child,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::X,
            )
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut control = model.control();
    for (dof, target) in [(6, 1e-4), (7, 4e-5)] {
        control.joint_target_mode.host_mut().unwrap()[dof] =
            JointTargetMode::PositionVelocity as u32;
        control.joint_target_pos.host_mut().unwrap()[dof] = target;
        control.joint_target_kp.host_mut().unwrap()[dof] = 572957.795;
        control.joint_target_kd.host_mut().unwrap()[dof] = 8594.367;
        control.joint_motor_max_force.host_mut().unwrap()[dof] = 20000.0;
    }
    control.upload_to_device(device, queue).unwrap();
    let mut cpu_input = model.state().unwrap();
    let mut cpu_output = model.state().unwrap();
    let mut gpu_input = model.state().unwrap();
    let mut gpu_output = model.state().unwrap();
    gpu_input.upload_to_device(device, queue).unwrap();
    gpu_output.upload_to_device(device, queue).unwrap();
    let mut cpu = FeatherstoneSolver::new_cpu()
        .with_rigorous_d_solve(true)
        .with_stable_motor_feedback(true);
    let mut gpu = FeatherstoneSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_rigorous_d_solve(true)
        .with_stable_motor_feedback(true);
    let contacts = Contacts::with_capacity(0);
    let mut peak = 0.0_f64;
    let h = 1.0 / 19200.0;
    let inertia = 0.168460975;
    let kp = 572957.795;
    let kd = 8594.367;
    let denominator = 1.0 + (kd + kp * h) * h * (1.0 / inertia + 0.25);
    let mut angles = [0.0; 2];
    let mut rates = [0.0; 2];
    let mut root_rate = 0.0;
    for step in 1..=512 {
        let torques: [f64; 2] = std::array::from_fn(|i| {
            (kp * ([1e-4, 4e-5][i] - angles[i]) - (kd + kp * h) * rates[i]) / denominator
        });
        let reaction = (torques[0] + torques[1]) * 0.25;
        root_rate -= h * reaction;
        for i in 0..2 {
            rates[i] += h * (torques[i] / inertia + reaction);
            angles[i] += h * rates[i];
        }
        cpu.step(
            &model,
            &cpu_input,
            &control,
            &contacts,
            1.0 / 19200.0,
            &mut cpu_output,
        )
        .unwrap();
        gpu.step(
            &model,
            &gpu_input,
            &control,
            &contacts,
            1.0 / 19200.0,
            &mut gpu_output,
        )
        .unwrap();
        let velocity =
            gpu_readback::read::<f32>(&engine, gpu_output.joint_qd.device_buffer().unwrap(), 8);
        let position =
            gpu_readback::read::<f32>(&engine, gpu_output.joint_q.device_buffer().unwrap(), 9);
        for (dof, expected) in [(3, root_rate), (6, rates[0]), (7, rates[1])] {
            assert!((cpu_output.joint_qd.host().unwrap()[dof] - expected).abs() < 1e-12);
            assert!(
                (velocity[dof] as f64 - expected).abs() < 2e-7,
                "analytic response step={step} dof={dof}: GPU={}, expected={expected}",
                velocity[dof]
            );
        }
        for i in 0..2 {
            assert!((cpu_output.joint_q.host().unwrap()[7 + i] - angles[i]).abs() < 1e-12);
            assert!((position[7 + i] as f64 - angles[i]).abs() < 2e-9);
        }
        for (&actual, &expected) in velocity.iter().zip(cpu_output.joint_qd.host().unwrap()) {
            assert!(
                (actual as f64 - expected).abs() < 2e-5,
                "step={step}: GPU={actual}, CPU={expected}"
            );
        }
        peak = peak.max(velocity[6].abs().max(velocity[7].abs()) as f64);
        let momentum = (4.0 + 2.0 * 0.168460975) * velocity[3] as f64
            + 0.168460975 * (velocity[6] as f64 + velocity[7] as f64);
        assert!(momentum.abs() < 2e-6, "step={step}: momentum={momentum}");
        std::mem::swap(&mut cpu_input, &mut cpu_output);
        std::mem::swap(&mut gpu_input, &mut gpu_output);
    }
    assert_eq!(
        gpu_readback::read::<u32>(&engine, gpu.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    assert!(peak < 0.01, "stiff servo oscillation: peak={peak}");
    assert!(gpu_input.joint_qd.host().is_err());
    eprintln!(
        "512-step coupled GPU feedback trajectory: peak={peak}rad/s; CPU parity and momentum passed"
    );
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn stable_feedback_rejects_unsupported_modes_and_faults_nonfinite_gains() {
    let engine = Engine::new(Backend::AutoGpu).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = model(true, false);
    model.upload_to_device(device, queue).unwrap();
    let mut input = model.state().unwrap();
    let mut output = model.state().unwrap();
    input.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    let mut control = model.control();
    control.joint_target_mode.host_mut().unwrap()[6] = JointTargetMode::Position as u32;
    control.joint_target_kp.host_mut().unwrap()[6] = 100.0;
    control.upload_to_device(device, queue).unwrap();
    let mut gpu = FeatherstoneSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_stable_motor_feedback(true);
    assert!(
        gpu.step(
            &model,
            &input,
            &control,
            &Contacts::with_capacity(0),
            0.001,
            &mut output
        )
        .unwrap_err()
        .to_string()
        .contains("rigorous")
    );
    gpu.rigorous_d_solve = true;
    control.joint_target_mode.host_mut().unwrap()[0] = JointTargetMode::Position as u32;
    control.joint_target_mode.upload(queue).unwrap();
    assert!(
        gpu.step(
            &model,
            &input,
            &control,
            &Contacts::with_capacity(0),
            0.001,
            &mut output
        )
        .unwrap_err()
        .to_string()
        .contains("scalar")
    );
    control.joint_target_mode.host_mut().unwrap()[0] = JointTargetMode::None as u32;
    control.joint_target_mode.upload(queue).unwrap();
    control.joint_target_kp.host_mut().unwrap()[6] = f64::NAN;
    control.joint_target_kp.upload(queue).unwrap();
    gpu.step(
        &model,
        &input,
        &control,
        &Contacts::with_capacity(0),
        0.001,
        &mut output,
    )
    .unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, gpu.gpu_status_buffer().unwrap(), 1)[0],
        32
    );
    let velocity = gpu_readback::read::<f32>(&engine, output.joint_qd.device_buffer().unwrap(), 7);
    assert_eq!(velocity, vec![0.0; 7]);
}
