#![cfg(feature = "gpu")]

use molla_compute::{Backend, Engine};
use molla_core::{BodyId, WorldId};
use molla_math::{SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, JointType, ModelBuilder};
use molla_solvers::{FeatherstoneSolver, Solver};

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod gpu_readback;

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn predictor_increment_is_independent_of_fused_rate_rounding_gpu() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let dt = 0.0001_f32;
    let cases: Vec<_> = (0..65536)
        .filter_map(|i| {
            let velocity = 0.5_f32 + i as f32 / 4096.0;
            let force = (31.123_f32 + i as f32 * 0.0137) * if i % 2 == 0 { 1.0 } else { -1.0 };
            let fused = force.mul_add(dt, velocity);
            let separate = force * dt + velocity;
            (fused != separate).then_some((velocity, force, fused, separate))
        })
        .take(64)
        .collect();
    assert_eq!(cases.len(), 64);
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    for _ in &cases {
        builder.begin_articulation();
        let body = builder.add_body(BodyParams::new());
        builder
            .add_joint_prismatic(
                BodyId::NONE,
                body,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::X,
            )
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut input = model.state().unwrap();
    let mut control = model.control();
    for (i, &(velocity, force, _, _)) in cases.iter().enumerate() {
        input.joint_qd.host_mut().unwrap()[i] = velocity as f64;
        control.joint_force.host_mut().unwrap()[i] = force as f64;
    }
    input.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    let mut output = model.state().unwrap();
    output.upload_to_device(device, queue).unwrap();
    let mut kernel = molla_kernels::FeatherstoneKernel::new(device, queue, &model);
    kernel
        .dispatch(
            device,
            queue,
            &model,
            &input,
            &control,
            &Contacts::with_capacity(0),
            0.0,
            dt as f64,
            true,
            false,
            &mut output,
            None,
        )
        .unwrap();
    let deltas = gpu_readback::read::<f32>(&engine, kernel.velocity_delta_buffer(), cases.len());
    let velocities = gpu_readback::read::<f32>(
        &engine,
        output.joint_qd.device_buffer().unwrap(),
        cases.len(),
    );
    for (i, &(_, force, fused, separate)) in cases.iter().enumerate() {
        assert_eq!(
            deltas[i].to_bits(),
            (force * dt).to_bits(),
            "case={i}: {:?}",
            cases[i]
        );
        assert!(velocities[i] == fused || velocities[i] == separate);
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn predictor_retains_small_increments_and_velocity_overrides_gpu() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (locked, cap, damping) in [
        (false, f64::INFINITY, 0.0),
        (false, 0.5, 0.0),
        (true, f64::INFINITY, 0.0),
        (false, f64::INFINITY, 0.25),
    ] {
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        builder.begin_articulation();
        let body = builder.add_body(BodyParams::new());
        let joint = builder
            .add_joint_prismatic(
                BodyId::NONE,
                body,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::X,
            )
            .unwrap();
        if locked {
            builder.set_joint_limits(joint, &[0.0], &[0.0]).unwrap();
        }
        builder.set_joint_velocity_limit(joint, &[cap]).unwrap();
        let mut model = builder.build().unwrap();
        model.upload_to_device(device, queue).unwrap();
        let mut input = model.state().unwrap();
        input.joint_qd.host_mut().unwrap()[0] = 1.0;
        input.upload_to_device(device, queue).unwrap();
        let mut output = model.state().unwrap();
        output.upload_to_device(device, queue).unwrap();
        let mut control = model.control();
        control.joint_force.host_mut().unwrap()[0] = 1e-6;
        control.upload_to_device(device, queue).unwrap();
        let mut kernel = molla_kernels::FeatherstoneKernel::new(device, queue, &model);
        kernel
            .dispatch(
                device,
                queue,
                &model,
                &input,
                &control,
                &Contacts::with_capacity(0),
                damping,
                0.01,
                true,
                false,
                &mut output,
                None,
            )
            .unwrap();
        let delta = gpu_readback::read::<f32>(&engine, kernel.velocity_delta_buffer(), 1)[0];
        let velocity =
            gpu_readback::read::<f32>(&engine, output.joint_qd.device_buffer().unwrap(), 1)[0];
        if locked {
            assert_eq!(delta, -1.0);
            assert_eq!(velocity, 0.0);
        } else if cap.is_finite() {
            assert_eq!(delta, -0.5);
            assert_eq!(velocity, 0.5);
        } else if damping == 0.0 {
            assert_eq!(velocity, 1.0);
            assert!((delta - 1e-8).abs() < 1e-13, "delta={delta}");
        } else {
            assert!((delta + 0.0025).abs() < 1e-7, "delta={delta}");
            assert!((velocity - 0.9975).abs() < 1e-7);
        }
        kernel
            .drift(
                device,
                queue,
                &model,
                &input,
                &control,
                0.01,
                false,
                &mut output,
            )
            .unwrap();
        let after_drift = gpu_readback::read::<f32>(&engine, kernel.velocity_delta_buffer(), 1)[0];
        assert_eq!(after_drift.to_bits(), delta.to_bits());
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn scalar_equality_transmits_child_load_to_floating_parent_gpu() {
    let engine = Engine::new(Backend::AutoGpu).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for kind in [JointType::Revolute, JointType::Prismatic] {
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        builder.begin_articulation();
        let parent = builder.add_body(BodyParams::new());
        let child = builder.add_body(BodyParams::new());
        builder
            .add_joint_free(
                BodyId::NONE,
                parent,
                Transform::IDENTITY,
                Transform::IDENTITY,
            )
            .unwrap();
        let joint = if kind == JointType::Revolute {
            builder.add_joint_revolute(
                parent,
                child,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::X,
            )
        } else {
            builder.add_joint_prismatic(
                parent,
                child,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::X,
            )
        }
        .unwrap();
        builder.set_joint_limits(joint, &[0.0], &[0.0]).unwrap();
        let mut model = builder.build().unwrap();
        model.upload_to_device(device, queue).unwrap();
        let mut input = model.state().unwrap();
        let mut control = model.control();
        let force = if kind == JointType::Prismatic {
            SpatialVector::new(Vec3::X * 10.0, Vec3::ZERO)
        } else {
            SpatialVector::new(Vec3::ZERO, Vec3::X * 10.0)
        };
        input.body_f.host_mut().unwrap()[child.index()] = force;
        let mut reference = model.state().unwrap();
        FeatherstoneSolver::new_cpu()
            .step(
                &model,
                &input,
                &control,
                &Contacts::with_capacity(0),
                0.01,
                &mut reference,
            )
            .unwrap();
        input.upload_to_device(device, queue).unwrap();
        control.upload_to_device(device, queue).unwrap();
        let mut output = model.state().unwrap();
        output.upload_to_device(device, queue).unwrap();
        FeatherstoneSolver::new_gpu(&engine, &model)
            .unwrap()
            .step(
                &model,
                &input,
                &control,
                &Contacts::with_capacity(0),
                0.01,
                &mut output,
            )
            .unwrap();
        let velocities =
            gpu_readback::read::<f32>(&engine, output.joint_qd.device_buffer().unwrap(), 7);
        for (&gpu, &cpu) in velocities.iter().zip(reference.joint_qd.host().unwrap()) {
            assert!(
                (gpu as f64 - cpu).abs() < 1e-6,
                "{kind:?}: GPU={gpu}, CPU={cpu}"
            );
        }
        let response = velocities[if kind == JointType::Prismatic { 0 } else { 3 }];
        assert!(
            (response - 0.05).abs() < 1e-6,
            "{kind:?}: response={response}"
        );
        assert_eq!(velocities[6], 0.0);
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn scalar_equality_clears_velocity_and_holds_nonzero_coordinate_gpu() {
    let engine = Engine::new(Backend::AutoGpu).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for kind in [JointType::Revolute, JointType::Prismatic] {
        for velocity in [-3.0, 0.0, 3.0] {
            let mut builder = ModelBuilder::new();
            builder.set_world_gravity(WorldId(0), Vec3::ZERO);
            builder.begin_articulation();
            let body = builder.add_body(BodyParams::new());
            let joint = if kind == JointType::Revolute {
                builder.add_joint_revolute(
                    BodyId::NONE,
                    body,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                    Vec3::X,
                )
            } else {
                builder.add_joint_prismatic(
                    BodyId::NONE,
                    body,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                    Vec3::X,
                )
            }
            .unwrap();
            builder.set_joint_limits(joint, &[0.2], &[0.2]).unwrap();
            let mut model = builder.build().unwrap();
            model.upload_to_device(device, queue).unwrap();
            let mut input = model.state().unwrap();
            input.joint_q.host_mut().unwrap()[0] = -0.4;
            input.joint_qd.host_mut().unwrap()[0] = velocity;
            input.upload_to_device(device, queue).unwrap();
            let mut output = model.state().unwrap();
            output.upload_to_device(device, queue).unwrap();
            let mut control = model.control();
            control.joint_force.host_mut().unwrap()[0] = 10.0;
            control.upload_to_device(device, queue).unwrap();
            let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
            solver
                .step(
                    &model,
                    &input,
                    &control,
                    &Contacts::with_capacity(0),
                    0.01,
                    &mut output,
                )
                .unwrap();
            let q =
                gpu_readback::read::<f32>(&engine, output.joint_q.device_buffer().unwrap(), 1)[0];
            let qd =
                gpu_readback::read::<f32>(&engine, output.joint_qd.device_buffer().unwrap(), 1)[0];
            assert!((q - 0.2).abs() < 1e-7, "{kind:?}: q={q}");
            assert_eq!(qd, 0.0, "{kind:?}: incoming velocity={velocity}");
            assert!(output.joint_q.host().is_err());
            let mut contacts = Contacts::with_capacity(1);
            contacts.count = 1;
            let error = solver
                .step(&model, &input, &control, &contacts, 0.01, &mut output)
                .unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("contact response is not implemented")
            );
            let unchanged =
                gpu_readback::read::<f32>(&engine, output.joint_q.device_buffer().unwrap(), 1)[0];
            assert_eq!(unchanged, q);
        }
    }
}
