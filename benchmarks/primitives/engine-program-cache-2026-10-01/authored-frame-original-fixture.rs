#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Mat3, Quat, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder, eval_fk};
use molla_solvers::FeatherstoneSolver;

#[path = "common/gpu_readback.rs"]
mod gpu_readback;

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn compensated_free_quaternions_retain_euler_integration_tails() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let initial = [
        Quat::from_scaled_axis(Vec3::new(0.4, -0.3, 0.2)),
        Quat::from_scaled_axis(Vec3::new(1.2, 0.8, -2.3)),
        Quat::from_scaled_axis(Vec3::new(-0.13, 0.27, 0.09)),
    ];
    for _ in initial {
        builder.begin_articulation();
        let body = builder.add_body(BodyParams::new());
        builder
            .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    let mut source = model.state().unwrap();
    let omega = Vec3::new(0.031, -0.027, 0.013);
    for (i, rotation) in initial.into_iter().enumerate() {
        source.joint_q.host_mut().unwrap()[i * 7 + 3..i * 7 + 7]
            .copy_from_slice(&rotation.to_array());
        source.joint_qd.host_mut().unwrap()[i * 6 + 3..i * 6 + 6]
            .copy_from_slice(&omega.to_array());
    }
    eval_fk(&model, &mut source).unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(device, queue).unwrap();
    let mut states = [source.clone_host().unwrap(), source.clone_host().unwrap()];
    for state in &mut states {
        state.upload_to_device(device, queue).unwrap();
    }
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&states.each_ref())
        .unwrap();
    let lows: Vec<_> = source
        .joint_q
        .host()
        .unwrap()
        .iter()
        .map(|&value| (value - f64::from(value as f32)) as f32)
        .collect();
    for state in &states {
        queue.write_buffer(
            solver.gpu_coordinate_compensation_buffer(state).unwrap(),
            0,
            bytemuck::cast_slice(&lows),
        );
    }
    let dt = 0.004_f32;
    let omega = omega.to_array().map(|value| f64::from(value as f32));
    let dq = Quat::from_xyzw(0.5 * omega[0], 0.5 * omega[1], 0.5 * omega[2], 0.0);
    let mut expected = initial;
    for step in 0..64 {
        let [a, b] = &mut states;
        let (old, next) = if step % 2 == 0 { (a, b) } else { (b, a) };
        solver
            .drift_gpu(&model, old, &control, f64::from(dt), next)
            .unwrap();
        for rotation in &mut expected {
            *rotation = (*rotation + (dq * *rotation) * f64::from(dt)).normalize();
        }
    }
    let high = gpu_readback::read::<f32>(&engine, states[0].joint_q.device_buffer().unwrap(), 21);
    let low = gpu_readback::read::<f32>(
        &engine,
        solver
            .gpu_coordinate_compensation_buffer(&states[0])
            .unwrap(),
        21,
    );
    for (i, rotation) in expected.into_iter().enumerate() {
        for (lane, value) in rotation.to_array().into_iter().enumerate() {
            let coord = i * 7 + 3 + lane;
            let actual = f64::from(high[coord]) + f64::from(low[coord]);
            assert!(
                (actual - value).abs() < 1e-11,
                "body={i} lane={lane} actual={actual} expected={value}"
            );
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn compensated_fk_preserves_sub_ulp_translation_through_fixed_frames() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(molla_core::WorldId(0), Vec3::ZERO);
    let scales = [1e-6, 0.5, 1024.0];
    for scale in scales {
        builder.begin_articulation();
        let root = builder.add_body(BodyParams::new());
        let child = builder.add_body(BodyParams::new());
        builder
            .add_joint_free(BodyId::NONE, root, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        builder
            .add_joint_fixed(
                root,
                child,
                Transform::from_translation(Vec3::splat(scale * 0.125)),
                Transform::from_translation(Vec3::splat(scale * 0.0625)),
            )
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    for (i, scale) in scales.into_iter().enumerate() {
        initial.joint_q.host_mut().unwrap()[i * 7..i * 7 + 3].fill(scale);
        initial.joint_qd.host_mut().unwrap()[i * 6..i * 6 + 3].fill(scale * 1e-6);
    }
    eval_fk(&model, &mut initial).unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(device, queue).unwrap();
    let mut states = [initial.clone_host().unwrap(), initial.clone_host().unwrap()];
    for state in &mut states {
        state.upload_to_device(device, queue).unwrap();
    }
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&states.each_ref())
        .unwrap();
    for step in 0..10 {
        let [a, b] = &mut states;
        let (old, next) = if step % 2 == 0 { (a, b) } else { (b, a) };
        for _ in 0..2 {
            solver
                .drift_gpu(&model, old, &control, 0.001, next)
                .unwrap();
        }
    }
    let poses =
        gpu_readback::read::<[f32; 8]>(&engine, states[0].body_q.device_buffer().unwrap(), 6);
    let errors = gpu_readback::read::<[f32; 4]>(
        &engine,
        solver.gpu_position_compensation_buffer(&states[0]).unwrap(),
        6,
    );
    for (i, scale) in scales.into_iter().enumerate() {
        let travel = (scale * 1e-6) as f32 as f64 * 0.001_f32 as f64 * 10.0;
        for child in 0..2 {
            let expected = scale as f32 as f64
                + travel
                + child as f64 * ((scale * 0.125) as f32 as f64 - (scale * 0.0625) as f32 as f64);
            for axis in 0..3 {
                let body = 2 * i + child;
                let actual = poses[body][axis] as f64 + errors[body][axis] as f64;
                assert!(
                    (actual - expected).abs() < scale * 2e-13,
                    "scale={scale} child={child} axis={axis} actual={actual:e} expected={expected:e}"
                );
            }
        }
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    solver
        .encode_reset_gpu_coordinate_compensation(&mut encoder)
        .unwrap();
    queue.submit([encoder.finish()]);
    let errors = gpu_readback::read::<[f32; 4]>(
        &engine,
        solver.gpu_position_compensation_buffer(&states[0]).unwrap(),
        6,
    );
    assert_eq!(errors, [[0.0; 4]; 6]);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn compensated_drift_preserves_coordinates_across_large_reversals() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(molla_core::WorldId(0), Vec3::ZERO);
    let body = builder.add_body(BodyParams::new());
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    let origin = [1e-6, 0.5, 1024.0];
    initial.joint_q.host_mut().unwrap()[..3].copy_from_slice(&origin);
    eval_fk(&model, &mut initial).unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(device, queue).unwrap();
    let mut states = [initial.clone_host().unwrap(), initial.clone_host().unwrap()];
    for state in &mut states {
        state.upload_to_device(device, queue).unwrap();
    }
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&states.each_ref())
        .unwrap();
    for step in 0..6 {
        let [a, b] = &mut states;
        let (old, next) = if step % 2 == 0 { (a, b) } else { (b, a) };
        let jump = if step % 2 == 0 { 1e12_f32 } else { -1e12_f32 };
        let rates = [jump, -jump, jump, 0.0, 0.0, 0.0];
        queue.write_buffer(
            next.joint_qd.device_buffer().unwrap(),
            0,
            bytemuck::cast_slice(&rates),
        );
        solver.drift_gpu(&model, old, &control, 1.0, next).unwrap();
        if step % 2 == 1 {
            next.joint_q.download(device, queue).unwrap();
            for (axis, expected) in origin.into_iter().enumerate() {
                let actual = next.joint_q.host().unwrap()[axis];
                assert_eq!(actual, expected as f32 as f64, "step={step} axis={axis}");
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn compensated_drift_accumulates_sub_ulp_motion_without_double_redrift() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(molla_core::WorldId(0), Vec3::ZERO);
    let scales = [1e-6, 0.5, 1024.0];
    for scale in scales {
        builder.begin_articulation();
        let body = builder.add_body(BodyParams::new().with_position(Vec3::splat(scale)));
        builder
            .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    for (i, scale) in scales.into_iter().enumerate() {
        initial.joint_q.host_mut().unwrap()[i * 7..i * 7 + 3].fill(scale);
        initial.joint_qd.host_mut().unwrap()[i * 6..i * 6 + 3].copy_from_slice(&[
            scale * 1e-6,
            -scale * 1e-6,
            scale * 2e-6,
        ]);
    }
    eval_fk(&model, &mut initial).unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(device, queue).unwrap();
    for repeats in [1, 2] {
        let mut states = [initial.clone_host().unwrap(), initial.clone_host().unwrap()];
        for state in &mut states {
            state.upload_to_device(device, queue).unwrap();
        }
        let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
        solver
            .prepare_gpu_coordinate_compensation(&states.each_ref())
            .unwrap();
        assert!(
            solver
                .prepare_gpu_coordinate_compensation(&[&states[0], &states[0]])
                .is_err()
        );
        let mut unknown = initial.clone_host().unwrap();
        unknown.upload_to_device(device, queue).unwrap();
        assert!(
            solver
                .drift_gpu(&model, &states[0], &control, 0.001, &mut unknown)
                .is_err()
        );
        for phase in 0..2 {
            if phase == 1 {
                for state in &mut states {
                    state.joint_q.download(device, queue).unwrap();
                    state
                        .joint_q
                        .host_mut()
                        .unwrap()
                        .copy_from_slice(initial.joint_q.host().unwrap());
                    state.joint_q.upload(queue).unwrap();
                }
                let mut encoder = device.create_command_encoder(&Default::default());
                solver
                    .encode_reset_gpu_coordinate_compensation(&mut encoder)
                    .unwrap();
                queue.submit([encoder.finish()]);
            }
            for step in 0..1000 {
                let [a, b] = &mut states;
                let (old, next) = if step % 2 == 0 { (a, b) } else { (b, a) };
                for _ in 0..repeats {
                    solver
                        .drift_gpu(&model, old, &control, 0.001, next)
                        .unwrap();
                }
            }
            states[0].joint_q.download(device, queue).unwrap();
            let coordinates = states[0].joint_q.host().unwrap();
            for (i, scale) in scales.into_iter().enumerate() {
                for (axis, speed) in [scale * 1e-6, -scale * 1e-6, scale * 2e-6]
                    .into_iter()
                    .enumerate()
                {
                    let expected =
                        scale as f32 as f64 + speed as f32 as f64 * 0.001_f32 as f64 * 1000.0;
                    let actual = coordinates[i * 7 + axis];
                    assert!(
                        (actual - expected).abs() < scale * 1.2e-7,
                        "phase={phase} repeats={repeats} scale={scale} axis={axis} actual={actual:e} expected={expected:e}"
                    );
                    assert!((actual - scale).abs() > scale * 0.8e-6);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn corrected_velocity_drift_matches_cpu_without_reapplying_forces() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder.begin_articulation();
    let mut parent = BodyId::NONE;
    for kind in 0..8 {
        let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 2.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }));
        let xp = Transform::from_translation(Vec3::new(0.03, 0.04, 0.05));
        let xc = Transform::IDENTITY;
        let joint = match kind {
            0 => builder.add_joint_prismatic(parent, body, xp, xc, Vec3::Y),
            1 => builder.add_joint_revolute(parent, body, xp, xc, Vec3::X),
            2 => builder.add_joint_ball(parent, body, xp, xc),
            3 => builder.add_joint_fixed(parent, body, xp, xc),
            4 => builder.add_joint_free(parent, body, xp, xc),
            5 => builder.add_joint_distance(parent, body, xp, xc),
            6 => builder.add_joint_d6(parent, body, xp, xc),
            _ => builder.add_joint_cable(parent, body, xp, xc, 50.0, 20.0, 2.0, 2.0),
        }
        .unwrap();
        if kind == 6 {
            builder
                .set_joint_limits(joint, &[0.0; 6], &[0.0; 6])
                .unwrap();
        }
        parent = body;
    }
    let mut model = builder.build().unwrap();
    let mut start = model.state().unwrap();
    start.joint_qd.host_mut().unwrap().fill(-9.0);
    eval_fk(&model, &mut start).unwrap();
    let mut cpu = start.clone_host().unwrap();
    for (i, v) in cpu.joint_qd.host_mut().unwrap().iter_mut().enumerate() {
        *v = (i as f64 + 1.0) * 0.03;
    }
    let mut gpu = cpu.clone_host().unwrap();
    let mut control = model.control();
    control.joint_force.host_mut().unwrap().fill(1000.0);
    FeatherstoneSolver::new_cpu()
        .drift_cpu(&model, &start, 0.01, &mut cpu)
        .unwrap();
    model.upload_to_device(device, queue).unwrap();
    start.upload_to_device(device, queue).unwrap();
    gpu.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    FeatherstoneSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_damping(20.0)
        .drift_gpu(&model, &start, &control, 0.01, &mut gpu)
        .unwrap();
    assert!(gpu.joint_q.host().is_err());
    gpu.joint_q.download(device, queue).unwrap();
    gpu.joint_qd.download(device, queue).unwrap();
    gpu.body_q.download(device, queue).unwrap();
    for (a, b) in gpu
        .joint_q
        .host()
        .unwrap()
        .iter()
        .zip(cpu.joint_q.host().unwrap())
    {
        assert!((a - b).abs() < 2e-6, "coordinate {a} vs {b}");
    }
    for (a, b) in gpu
        .joint_qd
        .host()
        .unwrap()
        .iter()
        .zip(cpu.joint_qd.host().unwrap())
    {
        assert!((a - b).abs() < 2e-6, "velocity {a} vs {b}");
    }
    for (a, b) in gpu
        .body_q
        .host()
        .unwrap()
        .iter()
        .zip(cpu.body_q.host().unwrap())
    {
        assert!((a.position - b.position).length() < 2e-6);
        assert!(a.rotation.dot(b.rotation).abs() > 1.0 - 2e-6);
    }
}
