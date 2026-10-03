#![cfg(feature = "gpu")]

use molla_compute::{Backend, Engine};
use molla_core::{BodyId, WorldId};
use molla_math::{Transform, Vec3};
use molla_sim::{BodyParams, Contacts, JointTargetMode, ModelBuilder};
use molla_solvers::fem_rigid::FemRigidConfig;
use molla_solvers::fem_rigid_gpu::FemRigidGpuSystem;
use molla_solvers::{FeatherstoneSolver, Solver};

#[path = "common/fem_rotor.rs"]
#[allow(dead_code)]
mod fem_rotor;
#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod gpu_readback;

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_motor_caps_preserve_feedforward_and_total_limits() {
    let engine = Engine::new(Backend::AutoGpu).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.begin_articulation();
    let body = builder.add_body(BodyParams::new());
    builder
        .add_joint_revolute(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::X,
        )
        .unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut cpu = FeatherstoneSolver::new_cpu().with_substeps(1);
    let mut gpu = FeatherstoneSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_substeps(1);
    let contacts = Contacts::with_capacity(0);
    let dt = 0.01;
    for mode in [
        JointTargetMode::Position,
        JointTargetMode::Velocity,
        JointTargetMode::PositionVelocity,
        JointTargetMode::Effort,
    ] {
        for (sign, cap, feedforward, total) in [
            (1.0, 10.0, 25.0, f64::INFINITY),
            (-1.0, 10.0, 25.0, f64::INFINITY),
            (1.0, 0.0, 25.0, f64::INFINITY),
            (1.0, f64::INFINITY, 25.0, f64::INFINITY),
            (1.0, 10.0, 25.0, 30.0),
            (-1.0, 10.0, -25.0, 30.0),
        ] {
            model.joint_effort_limit.host_mut().unwrap()[0] = total;
            model.joint_effort_limit.upload(queue).unwrap();
            let mut control = model.control();
            control.joint_target_mode.host_mut().unwrap()[0] = mode as u32;
            control.joint_target_pos.host_mut().unwrap()[0] = if mode == JointTargetMode::Effort {
                100.0 * sign
            } else {
                sign
            };
            control.joint_target_vel.host_mut().unwrap()[0] = sign;
            let gain = if mode == JointTargetMode::PositionVelocity {
                50.0
            } else {
                100.0
            };
            control.joint_target_kp.host_mut().unwrap()[0] = gain;
            control.joint_target_kd.host_mut().unwrap()[0] = gain;
            control.joint_motor_max_force.host_mut().unwrap()[0] = cap;
            control.joint_force.host_mut().unwrap()[0] = feedforward;
            let initial = model.state().unwrap();
            let mut cpu_output = model.state().unwrap();
            cpu.step(&model, &initial, &control, &contacts, dt, &mut cpu_output)
                .unwrap();
            let expected =
                ((100.0_f64 * sign).clamp(-cap, cap) + feedforward).clamp(-total, total) * dt;
            let cpu_velocity = cpu_output.joint_qd.host().unwrap()[0];
            assert!((cpu_velocity - expected).abs() < 1e-10);
            let mut input = model.state().unwrap();
            let mut output = model.state().unwrap();
            input.upload_to_device(device, queue).unwrap();
            output.upload_to_device(device, queue).unwrap();
            control.upload_to_device(device, queue).unwrap();
            gpu.step(&model, &input, &control, &contacts, dt, &mut output)
                .unwrap();
            let velocity =
                gpu_readback::read::<f32>(&engine, output.joint_qd.device_buffer().unwrap(), 1)[0]
                    as f64;
            assert!(
                (velocity - expected).abs() < 2e-6,
                "{mode:?}, cap={cap}, ff={feedforward}, total={total}: GPU={velocity}, expected={expected}"
            );
            assert!(output.joint_qd.host().is_err());
        }
    }
    let mut control = model.control();
    let mut input = model.state().unwrap();
    let mut output = model.state().unwrap();
    input.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    for cap in [f64::NAN, -1.0, f64::NEG_INFINITY] {
        control.joint_motor_max_force.host_mut().unwrap()[0] = cap;
        control.upload_to_device(device, queue).unwrap();
        assert!(
            gpu.step(&model, &input, &control, &contacts, dt, &mut output)
                .is_err()
        );
    }
    control.joint_motor_max_force.host_mut().unwrap()[0] = 10.0;
    control.joint_motor_model.host_mut().unwrap()[0] = 1;
    control.joint_target_pos.host_mut().unwrap()[0] = 1.0;
    control.joint_target_vel.host_mut().unwrap()[0] = 1.0;
    control.joint_target_kp.host_mut().unwrap()[0] = 100.0;
    control.joint_target_kd.host_mut().unwrap()[0] = 100.0;
    for mode in [
        JointTargetMode::Position,
        JointTargetMode::Velocity,
        JointTargetMode::PositionVelocity,
    ] {
        control.joint_target_mode.host_mut().unwrap()[0] = mode as u32;
        control.upload_to_device(device, queue).unwrap();
        gpu.set_rigorous_d_solve(false);
        assert!(
            gpu.step(&model, &input, &control, &contacts, dt, &mut output)
                .unwrap_err()
                .to_string()
                .contains("acceleration motors require rigorous dynamics")
        );
        gpu.set_rigorous_d_solve(true);
        gpu.step(&model, &input, &control, &contacts, dt, &mut output)
            .unwrap();
        let velocity =
            gpu_readback::read::<f32>(&engine, output.joint_qd.device_buffer().unwrap(), 1)[0];
        assert!(
            (velocity as f64 - 10.0 * dt).abs() < 2e-6,
            "{mode:?}: {velocity}"
        );
    }
    for rigorous in [false, true] {
        gpu.set_rigorous_d_solve(rigorous);
        control.joint_target_mode.host_mut().unwrap()[0] = JointTargetMode::Effort as u32;
        control.joint_target_pos.host_mut().unwrap()[0] = 100.0;
        control.upload_to_device(device, queue).unwrap();
        gpu.step(&model, &input, &control, &contacts, dt, &mut output)
            .unwrap();
        let velocity =
            gpu_readback::read::<f32>(&engine, output.joint_qd.device_buffer().unwrap(), 1)[0];
        assert!((velocity as f64 - 10.0 * dt).abs() < 2e-6);
    }
    control.joint_motor_model.host_mut().unwrap()[0] = 2;
    control.upload_to_device(device, queue).unwrap();
    assert!(
        gpu.step(&model, &input, &control, &contacts, dt, &mut output)
            .unwrap_err()
            .to_string()
            .contains("motor model")
    );
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn fem_island_accepts_caps_and_applies_them_on_gpu() {
    let engine = std::sync::Arc::new(Engine::new(Backend::AutoGpu).unwrap());
    let mut scene = fem_rotor::scene();
    scene.rigid_model.joint_effort_limit.host_mut().unwrap()[0] = f64::INFINITY;
    scene.control.joint_target_mode.host_mut().unwrap()[0] = JointTargetMode::Effort as u32;
    scene.control.joint_target_pos.host_mut().unwrap()[0] = 100.0;
    scene.control.joint_motor_max_force.host_mut().unwrap()[0] = 0.1;
    scene.control.joint_force.host_mut().unwrap()[0] = 0.2;
    let h = 1.0 / 4096.0;
    let mut system = FemRigidGpuSystem::new(
        engine.clone(),
        scene,
        FemRigidConfig {
            max_substep: h,
            elastic_iterations: 4,
            ..Default::default()
        },
    )
    .unwrap();
    system.step(h).unwrap();
    let velocity = gpu_readback::read::<f32>(
        &engine,
        system.rigid_state().joint_qd.device_buffer().unwrap(),
        1,
    )[0] as f64;
    let expected = (0.1 + 0.2) / 0.25 * h;
    assert!(
        (velocity - expected).abs() < 1e-7,
        "{velocity} != {expected}"
    );
    assert_eq!(
        gpu_readback::read::<u32>(&engine, system.status_buffer(), 1),
        [0]
    );
    assert!(system.rigid_state().joint_qd.host().is_err());
    for cap in [f64::NAN, -1.0, f64::NEG_INFINITY] {
        let mut scene = fem_rotor::scene();
        scene.control.joint_motor_max_force.host_mut().unwrap()[0] = cap;
        assert!(FemRigidGpuSystem::new(engine.clone(), scene, FemRigidConfig::default()).is_err());
    }
    let mut scene = fem_rotor::scene();
    scene.control.joint_motor_model.host_mut().unwrap()[0] = 1;
    assert!(FemRigidGpuSystem::new(engine, scene, FemRigidConfig::default()).is_err());
}
