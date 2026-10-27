#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::{BodyId, WorldId};
use molla_math::{GpuSpatialVector, Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::{
    BodyFlags, BodyParams, Contacts, JointType, ModelBuilder, eval_body_velocities, eval_fk,
};
use molla_solvers::{FeatherstoneSolver, Solver};

#[path = "common/gpu_readback.rs"]
mod gpu_readback;

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn zero_coordinate_compensation_retains_redrift_residues_and_resets_explicitly() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder.begin_articulation();
    let body = builder.add_body(BodyParams {
        flags: BodyFlags::KINEMATIC,
        pose: Transform::from_translation(Vec3::X * 1024.),
        ..Default::default()
    });
    builder
        .add_joint_fixed(
            BodyId::NONE,
            body,
            Transform::from_translation(Vec3::X * 1024.),
            Transform::IDENTITY,
        )
        .unwrap();
    builder
        .set_kinematic_acceleration(body, Some(SpatialVector::ZERO))
        .unwrap();
    let mut model = builder.build().unwrap();
    assert_eq!(model.joint_coord_total, 0);
    let mut input = model.state().unwrap();
    input.body_qd.host_mut().unwrap()[0] = SpatialVector::new(Vec3::X * 1e-5, Vec3::ZERO);
    let mut output = input.clone_host().unwrap();
    let mut control = model.control();
    model.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    input.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&[&input, &output])
        .unwrap();
    assert!(
        solver
            .prepare_gpu_coordinate_compensation(&[&input, &input])
            .is_err()
    );
    let low = |solver: &FeatherstoneSolver, state: &molla_sim::State| {
        gpu_readback::read::<[f32; 4]>(
            &engine,
            solver.gpu_position_compensation_buffer(state).unwrap(),
            1,
        )[0]
    };
    for frame in 0..31 {
        solver
            .step(
                &model,
                &input,
                &control,
                &Contacts::with_capacity(0),
                0.001,
                &mut output,
            )
            .unwrap();
        let once = low(&solver, &output);
        for _ in 0..2 {
            solver
                .drift_gpu(&model, &input, &control, 0.001, &mut output)
                .unwrap();
            assert_eq!(low(&solver, &output), once);
        }
        let expected = (frame + 1) as f64 * 1e-5_f32 as f64 * 0.001_f32 as f64;
        assert!(
            (once[0] as f64 - expected).abs() < 2e-13,
            "frame={frame}, low={once:?}, expected={expected:e}"
        );
        std::mem::swap(&mut input, &mut output);
    }
    let poses = gpu_readback::read::<[f32; 8]>(&engine, input.body_q.device_buffer().unwrap(), 1);
    assert_eq!(poses[0][0], 1024.);
    eprintln!(
        "zero-coordinate GPU position residue={:?}",
        low(&solver, &input)
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    solver
        .encode_reset_gpu_coordinate_compensation(&mut encoder)
        .unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(low(&solver, &input), [0.; 4]);
    assert_eq!(low(&solver, &output), [0.; 4]);
    solver
        .step(
            &model,
            &input,
            &control,
            &Contacts::with_capacity(0),
            0.001,
            &mut output,
        )
        .unwrap();
    assert!((low(&solver, &output)[0] as f64 - 1e-8).abs() < 1e-14);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_driven_prediction_and_repeated_redrift_match_cpu_across_tree_kinds() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), -Vec3::Y * 9.81);
    let mut roots = Vec::new();
    for kind in [JointType::Fixed, JointType::Revolute, JointType::Ball] {
        for shift in [Vec3::ZERO, Vec3::new(32., -16., 8.)] {
            for rotation in [
                Quat::IDENTITY,
                Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.4),
            ] {
                builder.begin_articulation();
                let root = builder.add_body(BodyParams {
                    flags: BodyFlags::KINEMATIC,
                    com: Vec3::Y * 0.25,
                    pose: Transform {
                        position: shift,
                        rotation,
                    },
                    ..Default::default()
                });
                builder
                    .set_kinematic_acceleration(
                        root,
                        Some(SpatialVector::new(
                            rotation * Vec3::new(3., 6., -1.),
                            rotation * Vec3::new(0.2, -0.3, 2.),
                        )),
                    )
                    .unwrap();
                builder
                    .add_joint_fixed(
                        BodyId::NONE,
                        root,
                        Transform {
                            position: shift,
                            rotation,
                        },
                        Transform::IDENTITY,
                    )
                    .unwrap();
                let child = builder.add_body(BodyParams {
                    mass: 2.,
                    inertia: Mat3::IDENTITY * 4.,
                    com: Vec3::X,
                    pose: Transform {
                        position: shift + rotation * Vec3::Y * 0.25,
                        rotation,
                    },
                    ..Default::default()
                });
                let anchor = Transform::from_translation(Vec3::Y * 0.25);
                match kind {
                    JointType::Fixed => {
                        builder.add_joint_fixed(root, child, anchor, Transform::IDENTITY)
                    }
                    JointType::Revolute => builder.add_joint_revolute(
                        root,
                        child,
                        anchor,
                        Transform::IDENTITY,
                        Vec3::Z,
                    ),
                    _ => builder.add_joint_ball(root, child, anchor, Transform::IDENTITY),
                }
                .unwrap();
                roots.push((root, rotation));
            }
        }
    }
    builder.begin_articulation();
    let tiny = builder.add_body(BodyParams {
        flags: BodyFlags::KINEMATIC,
        com: Vec3::Y * 0.25,
        pose: Transform::from_translation(Vec3::X * 1024.),
        ..Default::default()
    });
    builder
        .set_kinematic_acceleration(tiny, Some(SpatialVector::new(Vec3::X * 1e-7, Vec3::ZERO)))
        .unwrap();
    builder
        .add_joint_fixed(
            BodyId::NONE,
            tiny,
            Transform::from_translation(Vec3::X * 1024.),
            Transform::IDENTITY,
        )
        .unwrap();
    let mut model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    initial.body_qd.host_mut().unwrap()[tiny.index()] =
        SpatialVector::new(Vec3::X * 1e-5, Vec3::ZERO);
    for (root, rotation) in roots {
        initial.body_qd.host_mut().unwrap()[root.index()] = SpatialVector::new(
            rotation * Vec3::new(1., -2., 3.),
            rotation * Vec3::new(0.4, -0.5, 1.2),
        );
    }
    eval_fk(&model, &mut initial).unwrap();
    eval_body_velocities(&model, &mut initial).unwrap();
    let mut control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut reference = initial.clone_host().unwrap();
    let mut cpu = FeatherstoneSolver::new_cpu();
    cpu.rigorous_d_solve = true;
    cpu.step(&model, &initial, &control, &contacts, 0.001, &mut reference)
        .unwrap();
    model.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    let mut output = initial.clone_host().unwrap();
    initial.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    let mut gpu = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    gpu.rigorous_d_solve = true;
    gpu.prepare_gpu_coordinate_compensation(&[&initial, &output])
        .unwrap();
    gpu.step(&model, &initial, &control, &contacts, 0.001, &mut output)
        .unwrap();
    let mut peak = 0.0_f64;
    for pass in 0..3 {
        if pass > 0 {
            gpu.drift_gpu(&model, &initial, &control, 0.001, &mut output)
                .unwrap();
            cpu.drift_cpu(&model, &initial, 0.001, &mut reference)
                .unwrap();
        }
        let poses = gpu_readback::read::<[f32; 8]>(
            &engine,
            output.body_q.device_buffer().unwrap(),
            model.body_count,
        );
        let velocities = gpu_readback::read::<GpuSpatialVector>(
            &engine,
            output.body_qd.device_buffer().unwrap(),
            model.body_count,
        );
        let rates = gpu_readback::read::<f32>(
            &engine,
            output.joint_qd.device_buffer().unwrap(),
            model.joint_dof_total,
        );
        for (actual, expected) in rates.iter().zip(reference.joint_qd.host().unwrap()) {
            peak = peak.max((*actual as f64 - expected).abs());
        }
        for body in 0..model.body_count {
            let pose = reference.body_q.host().unwrap()[body];
            let position = Vec3::new(
                poses[body][0] as f64,
                poses[body][1] as f64,
                poses[body][2] as f64,
            );
            peak = peak.max((position - pose.position).length());
            let actual: SpatialVector = velocities[body].into();
            let expected = reference.body_qd.host().unwrap()[body];
            peak = peak
                .max((actual.linear - expected.linear).length())
                .max((actual.angular - expected.angular).length());
            let rotation = Quat::from_xyzw(
                poses[body][4] as f64,
                poses[body][5] as f64,
                poses[body][6] as f64,
                poses[body][7] as f64,
            )
            .normalize();
            assert!(rotation.dot(pose.rotation).abs() > 1. - 1e-10);
        }
        assert!(peak < 2e-4, "pass={pass}, peak={peak:e}");
    }
    eprintln!("GPU prescribed tree prediction/redrift peak={peak:e}");
    #[cfg(feature = "resource-counters")]
    let counters = || {
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let c = device.get_internal_counters().hal;
        [
            c.buffers.read(),
            c.buffer_memory.read(),
            c.bind_groups.read(),
            c.compute_pipelines.read(),
            c.shader_modules.read(),
        ]
    };
    #[cfg(feature = "resource-counters")]
    let mut baseline = None;
    for step in 0..42 {
        std::mem::swap(&mut initial, &mut output);
        gpu.step(&model, &initial, &control, &contacts, 0.001, &mut output)
            .unwrap();
        #[cfg(feature = "resource-counters")]
        {
            let current = counters();
            if step == 1 {
                baseline = Some(current);
            }
            if step > 1 {
                assert_eq!(Some(current), baseline);
            }
        }
        let _ = step;
    }
    #[cfg(feature = "resource-counters")]
    eprintln!("GPU prescribed prediction warm resources={baseline:?}");
    let poses = gpu_readback::read::<[f32; 8]>(
        &engine,
        output.body_q.device_buffer().unwrap(),
        model.body_count,
    );
    let errors = gpu_readback::read::<[f32; 4]>(
        &engine,
        gpu.gpu_position_compensation_buffer(&output).unwrap(),
        model.body_count,
    );
    let travel = poses[tiny.index()][0] as f64 - 1024. + errors[tiny.index()][0] as f64;
    let time = 43. * 0.001_f32 as f64;
    let expected = 1e-5_f32 as f64 * time + 0.5 * 1e-7_f32 as f64 * time * time;
    assert!(
        (travel - expected).abs() < 1e-10,
        "travel={travel:e}, expected={expected:e}"
    );
    eprintln!("GPU prescribed compensated travel={travel:e}, expected={expected:e}");
}
