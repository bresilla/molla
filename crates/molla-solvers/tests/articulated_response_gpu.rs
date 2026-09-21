#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_kernels::{JacobianKernel, JacobianRequest, MassMatrixKernel};
use molla_math::{Mat3, Quat, Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Model, ModelBuilder, State, eval_fk};

fn fixture(reversed: bool, offset: Vec3) -> (Model, State) {
    let mut b = ModelBuilder::new();
    b.begin_articulation();
    let bodies: Vec<_> = (0..8)
        .map(|i| {
            b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 2.0 + i as Real,
                com: Vec3::new(0.03, -0.05, 0.02),
                inertia: Mat3::from_diagonal(Vec3::new(0.2, 0.3, 0.4)),
            }))
        })
        .collect();
    let frame = Transform {
        position: Vec3::new(0.25, 0.13, -0.31),
        rotation: Quat::from_rotation_y(0.37),
    };
    let child = Transform {
        position: Vec3::new(-0.04, 0.07, 0.03),
        rotation: Quat::from_rotation_z(-0.2),
    };
    b.add_joint_free(
        BodyId::NONE,
        bodies[0],
        Transform::IDENTITY,
        Transform::IDENTITY,
    )
    .unwrap();
    b.add_joint_revolute(bodies[0], bodies[1], frame, child, Vec3::X)
        .unwrap();
    let slide = b
        .add_joint_prismatic(bodies[0], bodies[2], frame, child, Vec3::Z)
        .unwrap();
    b.add_joint_ball(bodies[1], bodies[3], frame, child)
        .unwrap();
    b.add_joint_fixed(bodies[0], bodies[4], frame, child)
        .unwrap();
    let d6 = b.add_joint_d6(bodies[3], bodies[5], frame, child).unwrap();
    b.set_joint_limits(
        d6,
        &[0.0, -1.0, -1.0, -1.0, 0.0, -1.0],
        &[0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
    )
    .unwrap();
    b.add_joint_cable(bodies[5], bodies[6], frame, child, 100.0, 100.0, 0.0, 0.0)
        .unwrap();
    b.add_joint_distance(bodies[4], bodies[7], frame, child)
        .unwrap();
    b.set_joint_enabled(slide, false).unwrap();
    let mut model = b.build().unwrap();
    if reversed {
        for j in [0, 1, 3, 5, 7] {
            model.joint_reversed.host_mut().unwrap()[j] = 1;
        }
    }
    let mut state = model.state().unwrap();
    for j in 0..model.joint_count {
        let c = model.joint_coord_offset.host().unwrap()[j] as usize;
        let kind = model.joint_type.host().unwrap()[j];
        let q = state.joint_q.host_mut().unwrap();
        match kind {
            0 | 1 => q[c] = 0.31,
            2 => q[c..c + 4].copy_from_slice(&Quat::from_rotation_x(0.4).to_array()),
            4..=6 => {
                q[c..c + 3].copy_from_slice(
                    &(Vec3::new(0.2, 0.12, -0.1) + if j == 0 { offset } else { Vec3::ZERO })
                        .to_array(),
                );
                q[c + 3..c + 7].copy_from_slice(
                    &(Quat::from_rotation_x(0.3) * Quat::from_rotation_z(-0.25)).to_array(),
                );
            }
            7 => {
                q[c] = 0.05;
                q[c + 1] = 0.15;
            }
            _ => {}
        }
    }
    eval_fk(&model, &mut state).unwrap();
    (model, state)
}

fn compare(a: &[Real], b: &[f32], tolerance: Real, label: &str) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        assert!(
            b.is_finite() && (a - b as Real).abs() <= tolerance * (1.0 + a.abs()),
            "{label}[{i}]: CPU={a} GPU={b}"
        );
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_mass_and_jacobian_cover_floating_locked_and_reversed_joints() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for reversed in [false, true] {
        for offset in [Vec3::ZERO, Vec3::new(8.0, -3.0, 5.0)] {
            let (mut model, mut state) = fixture(reversed, offset);
            let expected_mass = molla_solvers::mass_matrix(&model, &state, 0).unwrap();
            let point = Vec3::new(0.1, -0.04, 0.13);
            let expected_j: Vec<_> = (0..model.body_count)
                .map(|body| molla_solvers::jacobian(&model, &state, 0, body, point).unwrap())
                .collect();
            model.upload_to_device(device, queue).unwrap();
            state.upload_to_device(device, queue).unwrap();
            let mass = MassMatrixKernel::new(&engine, &model).unwrap();
            let mass_buffer = mass.create_output_buffer(device);
            mass.dispatch(&engine, &model, &state, &mass_buffer)
                .unwrap();
            let gpu_mass = mass.read_back(&engine, &mass_buffer).unwrap();
            compare(&expected_mass, &gpu_mass, 3e-5, "mass");
            let jacobian = JacobianKernel::new(&engine, &model).unwrap();
            let width = 6 * model.joint_dof_total;
            let jacobian_buffer = jacobian.create_output_buffer(device, width * model.body_count);
            let requests: Vec<_> = (0..model.body_count)
                .map(|body| JacobianRequest {
                    articulation: 0,
                    target_body: body as u32,
                    output_offset: (body * width) as u32,
                    body_q_offset: 0,
                    point_local: [point.x as f32, point.y as f32, point.z as f32, 0.0],
                })
                .collect();
            jacobian
                .dispatch(&engine, &model, &state, &requests, &jacobian_buffer)
                .unwrap();
            let gpu_j = jacobian
                .read_back(&engine, &jacobian_buffer, width * model.body_count)
                .unwrap();
            for (body, expected) in expected_j.iter().enumerate() {
                compare(
                    expected,
                    &gpu_j[body * width..(body + 1) * width],
                    5e-6,
                    "jacobian",
                );
            }
            eprintln!(
                "GPU response parity: {} DOFs, {} bodies, reversed={reversed}, offset={offset:?}",
                model.joint_dof_total, model.body_count
            );
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_free_body_mass_matches_analytic_inertia() {
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 750.0,
        com: Vec3::ZERO,
        inertia: Mat3::from_diagonal(Vec3::new(70.0, 110.0, 140.0)),
    }));
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state.joint_q.host_mut().unwrap()[0..3].copy_from_slice(&[1000.0, -700.0, 400.0]);
    eval_fk(&model, &mut state).unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let kernel = MassMatrixKernel::new(&engine, &model).unwrap();
    let output = kernel.create_output_buffer(engine.device().unwrap());
    kernel.dispatch(&engine, &model, &state, &output).unwrap();
    let result = kernel.read_back(&engine, &output).unwrap();
    let diagonal = [750.0, 750.0, 750.0, 70.0, 110.0, 140.0];
    for r in 0..6 {
        for c in 0..6 {
            let expected = if r == c { diagonal[r] } else { 0.0 };
            assert!((result[r * 6 + c] - expected).abs() < 1e-3, "{result:?}");
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_response_rejects_oversized_trees_and_invalid_buffers() {
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    builder.begin_articulation();
    let mut parent = BodyId::NONE;
    for _ in 0..65 {
        let child = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }));
        builder
            .add_joint_revolute(
                parent,
                child,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::X,
            )
            .unwrap();
        parent = child;
    }
    let oversized = builder.build().unwrap();
    assert!(MassMatrixKernel::new(&engine, &oversized).is_err());
    assert!(JacobianKernel::new(&engine, &oversized).is_err());
    let (mut model, mut state) = fixture(false, Vec3::ZERO);
    let device = engine.device().unwrap();
    model
        .upload_to_device(device, engine.queue().unwrap())
        .unwrap();
    state
        .upload_to_device(device, engine.queue().unwrap())
        .unwrap();
    let kernel = JacobianKernel::new(&engine, &model).unwrap();
    let output = kernel.create_output_buffer(device, 6 * model.joint_dof_total);
    let valid = JacobianRequest {
        articulation: 0,
        target_body: 0,
        body_q_offset: 0,
        output_offset: 0,
        point_local: [0.0; 4],
    };
    for invalid in [
        JacobianRequest {
            articulation: 99,
            ..valid
        },
        JacobianRequest {
            target_body: 99,
            ..valid
        },
        JacobianRequest {
            body_q_offset: u32::MAX,
            ..valid
        },
        JacobianRequest {
            output_offset: u32::MAX,
            ..valid
        },
        JacobianRequest {
            point_local: [f32::NAN; 4],
            ..valid
        },
    ] {
        assert!(
            kernel
                .dispatch(&engine, &model, &state, &[invalid], &output)
                .is_err()
        );
    }
    assert!(
        kernel
            .dispatch(&engine, &model, &state, &[valid, valid], &output)
            .is_err()
    );
    let small = kernel.create_output_buffer(device, 1);
    assert!(
        kernel
            .dispatch(&engine, &model, &state, &[valid], &small)
            .is_err()
    );
    let mass = MassMatrixKernel::new(&engine, &model).unwrap();
    assert!(mass.dispatch(&engine, &model, &state, &small).is_err());
}
