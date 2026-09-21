#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Quat, SpatialInertia, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, ModelBuilder, eval_fk};
use molla_solvers::{FeatherstoneSolver, Solver};

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn floating_carrier_and_fixed_payload_match_cpu_dynamics() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut errors = Vec::new();
    for (offset, payload, rotor, moving) in [
        (Vec3::ZERO, false, false, false),
        (Vec3::new(0.3, -0.2, 0.4), false, false, false),
        (Vec3::new(0.3, -0.2, 0.4), true, false, false),
        (Vec3::new(0.3, -0.2, 0.4), true, true, false),
        (Vec3::new(0.3, -0.2, 0.4), true, true, true),
    ] {
        let mut builder = ModelBuilder::new();
        builder.begin_articulation();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        let carrier = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 5.0,
            com: Vec3::new(0.1, 0.2, -0.05),
            inertia: Mat3::from_diagonal(Vec3::new(0.2, 0.3, 0.4)),
        }));
        builder
            .add_joint_free(
                BodyId::NONE,
                carrier,
                Transform::from_translation(offset),
                Transform::IDENTITY,
            )
            .unwrap();
        if payload {
            let child = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 3.0,
                com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(Vec3::splat(0.07)),
            }));
            builder
                .add_joint_fixed(
                    carrier,
                    child,
                    Transform::from_translation(Vec3::new(0.5, 0.4, 0.2)),
                    Transform::IDENTITY,
                )
                .unwrap();
        }
        if rotor {
            let wheel = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 0.8,
                com: Vec3::new(0.02, 0.01, 0.0),
                inertia: Mat3::from_diagonal(Vec3::new(0.03, 0.04, 0.05)),
            }));
            builder
                .add_joint_revolute(
                    carrier,
                    wheel,
                    Transform::from_translation(Vec3::new(-0.2, 0.3, 0.1)),
                    Transform::IDENTITY,
                    Vec3::X,
                )
                .unwrap();
        }
        if moving {
            builder.set_world_gravity(WorldId(0), Vec3::new(-0.7, -9.81, 0.3));
        }
        let mut model = builder.build().unwrap();
        let mut start = model.state().unwrap();
        if moving {
            start.joint_q.host_mut().unwrap()[..3].copy_from_slice(&[0.12, 0.03, -0.08]);
            start.joint_q.host_mut().unwrap()[3..7]
                .copy_from_slice(&Quat::from_rotation_y(0.3).to_array());
            start
                .joint_qd
                .host_mut()
                .unwrap()
                .copy_from_slice(&[0.3, -0.1, 0.2, 0.4, -0.2, 0.1, 0.7]);
        }
        eval_fk(&model, &mut start).unwrap();
        start.body_f.host_mut().unwrap()[0] = SpatialVector {
            linear: Vec3::new(2.0, -3.0, 1.0),
            angular: Vec3::new(0.7, 0.3, -0.5),
        };
        let mut cpu = model.state().unwrap();
        let mut control = model.control();
        if rotor {
            control.joint_force.host_mut().unwrap()[6] = 2.0;
        }
        FeatherstoneSolver::new_cpu()
            .with_rigorous_d_solve(true)
            .step(
                &model,
                &start,
                &control,
                &Contacts::with_capacity(0),
                0.005,
                &mut cpu,
            )
            .unwrap();
        model.upload_to_device(device, queue).unwrap();
        start.upload_to_device(device, queue).unwrap();
        control.upload_to_device(device, queue).unwrap();
        let mut gpu = model.state().unwrap();
        gpu.upload_to_device(device, queue).unwrap();
        FeatherstoneSolver::new_gpu(&engine, &model)
            .unwrap()
            .with_rigorous_d_solve(true)
            .step(
                &model,
                &start,
                &control,
                &Contacts::with_capacity(0),
                0.005,
                &mut gpu,
            )
            .unwrap();
        gpu.joint_qd.download(device, queue).unwrap();
        let error = gpu
            .joint_qd
            .host()
            .unwrap()
            .iter()
            .zip(cpu.joint_qd.host().unwrap())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        eprintln!(
            "carrier offset={offset:?}, fixed payload={payload}, rotor={rotor}, moving={moving}, maximum velocity error={error}, cpu={:?}, gpu={:?}",
            cpu.joint_qd.host().unwrap(),
            gpu.joint_qd.host().unwrap()
        );
        errors.push(error);
        if moving {
            let mut cpu_solver = FeatherstoneSolver::new_cpu().with_rigorous_d_solve(true);
            let mut gpu_solver = FeatherstoneSolver::new_gpu(&engine, &model)
                .unwrap()
                .with_rigorous_d_solve(true);
            let mut cpu_next = model.state().unwrap();
            let mut gpu_next = model.state().unwrap();
            gpu_next.upload_to_device(device, queue).unwrap();
            for _ in 0..20 {
                cpu_solver
                    .step(
                        &model,
                        &cpu,
                        &control,
                        &Contacts::with_capacity(0),
                        0.005,
                        &mut cpu_next,
                    )
                    .unwrap();
                gpu_solver
                    .step(
                        &model,
                        &gpu,
                        &control,
                        &Contacts::with_capacity(0),
                        0.005,
                        &mut gpu_next,
                    )
                    .unwrap();
                std::mem::swap(&mut cpu, &mut cpu_next);
                std::mem::swap(&mut gpu, &mut gpu_next);
            }
            gpu.joint_qd.download(device, queue).unwrap();
            gpu.joint_q.download(device, queue).unwrap();
            let velocity_error = gpu
                .joint_qd
                .host()
                .unwrap()
                .iter()
                .zip(cpu.joint_qd.host().unwrap())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            let position_error = gpu
                .joint_q
                .host()
                .unwrap()
                .iter()
                .zip(cpu.joint_q.host().unwrap())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0_f64, f64::max);
            eprintln!(
                "moving carrier over 0.105 s: velocity error={velocity_error}, coordinate error={position_error}"
            );
            assert!(velocity_error < 1e-5 && position_error < 1e-5);
        }
    }
    assert!(
        errors.iter().all(|e| *e < 1e-6),
        "coupled carrier dynamics mismatch: {errors:?}"
    );
}
