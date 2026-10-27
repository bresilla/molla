#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_contact::{
    SoftRigidBallJoint, SoftRigidContactGpu, SoftRigidContactKernel,
};
use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use molla_math::{Mat3, Vec3};
use wgpu::util::DeviceExt;

#[path = "common/fem_readback.rs"]
mod fem_readback;
#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod gpu_readback;

#[path = "common/fem_rotor.rs"]
#[allow(dead_code)]
mod fem_rotor;

fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, values: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("ball joint test data"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

struct ResultData {
    qd: Vec<f32>,
    soft: Vec<f32>,
    body: Vec<f32>,
    status: u32,
    rows: Vec<f32>,
}

fn solve(
    engine: &Engine,
    offsets: [Vec3; 2],
    error: Vec3,
    initial: [f32; 12],
    planar: bool,
    coupled: bool,
    fault: u32,
) -> ResultData {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let com = [Vec3::new(0.125, -0.25, 0.5), Vec3::new(-0.25, 0.125, 0.25)];
    let local_a = com[0] + Vec3::new(offsets[0].y, -offsets[0].x, offsets[0].z);
    let local_b = com[1] + offsets[1];
    let joint = SoftRigidBallJoint {
        body_a: 0,
        body_b: 1,
        anchor_a: local_a.to_array().map(|v| v as f32),
        anchor_b: local_b.to_array().map(|v| v as f32),
    };
    let gather =
        SoftRigidContactKernel::new_sampled_with_ball_joints(device, queue, 0, 0, 2, &[joint])
            .unwrap();
    assert_eq!(gather.row_count(), 1);
    let stored = gpu_readback::read::<SoftRigidContactGpu>(engine, gather.rows(), 1);
    assert_eq!(stored[0].ids, [0, 1, 0, 3]);
    let mut rows = stored;
    if coupled {
        rows.insert(
            0,
            SoftRigidContactGpu {
                normal_gap: [0.0, 1.0, 0.0, 0.0],
                material: [0.0; 4],
                ids: [0, 0, 0, 1],
                support: [0, 0, 1, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            },
        );
    }
    let last = rows.len() - 1;
    if fault == 1 {
        rows[last].ids[0] = 2;
    }
    if fault == 2 {
        rows[last].material[0] = f32::NAN;
    }
    let contacts = buffer(device, &rows);
    let ca = Vec3::new(-com[0].y, com[0].x, com[0].z);
    let origin_a = -ca - offsets[0] + error;
    let origin_b = -com[1] - offsets[1];
    let half = 0.5_f32.sqrt();
    let mut poses = [
        origin_a.x as f32,
        origin_a.y as f32,
        origin_a.z as f32,
        0.0,
        0.0,
        0.0,
        half,
        half,
        origin_b.x as f32,
        origin_b.y as f32,
        origin_b.z as f32,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ];
    if fault == 3 {
        poses[0] = f32::NAN;
    }
    let poses = buffer(device, &poses);
    let com_buffer = buffer(
        device,
        &[
            com[0].x as f32,
            com[0].y as f32,
            com[0].z as f32,
            0.0,
            com[1].x as f32,
            com[1].y as f32,
            com[1].z as f32,
            0.0,
        ],
    );
    let masses = [2.0, 2.0, 2.0, 0.7, 0.9, 1.1, 3.0, 3.0, 3.0, 1.2, 1.4, 1.6];
    let mut mass = [0.0_f32; 144];
    let mut jac = [0.0_f32; 144];
    for d in 0..12 {
        mass[d * 12 + d] = masses[d];
        if !planar || d % 6 == 1 {
            jac[d * 12 + d] = 1.0;
        }
    }
    let jac = buffer(device, &jac);
    let mass = buffer(device, &mass);
    let qd = buffer(device, &initial);
    let body_vel = buffer(device, &[0.0_f32; 16]);
    let point = if coupled {
        -offsets[0] + error
    } else {
        Vec3::ZERO
    };
    let positions = buffer(
        device,
        &[point.x as f32, point.y as f32, point.z as f32, 0.0],
    );
    let soft = buffer(
        device,
        &[0.0_f32, if coupled { -3.0 } else { 0.0 }, 0.0, 0.0],
    );
    let inv = buffer(device, &[1.0_f32]);
    let mut results = Vec::new();
    for constructor in [
        SoftRigidSolveKernel::new_serial,
        SoftRigidSolveKernel::new_cooperative,
    ] {
        queue.write_buffer(&qd, 0, bytemuck::cast_slice(&initial));
        queue.write_buffer(
            &soft,
            0,
            bytemuck::cast_slice(&[0.0_f32, if coupled { -3.0 } else { 0.0 }, 0.0, 0.0]),
        );
        queue.write_buffer(&body_vel, 0, bytemuck::cast_slice(&[0.0_f32; 16]));
        let kernel = constructor(device, 12, rows.len() as u32).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        kernel
            .encode(
                device,
                &mut encoder,
                SoftRigidSolveBindings {
                    positions: &positions,
                    velocities: &soft,
                    inverse_mass: &inv,
                    contacts: &contacts,
                    body_poses: &poses,
                    body_com: &com_buffer,
                    jacobians: &jac,
                    mass_matrix: &mass,
                    rigid_velocities: &qd,
                    body_velocities: &body_vel,
                    particle_count: 1,
                    body_count: 2,
                },
                SoftRigidSolveSettings {
                    iterations: 100,
                    dt: 0.01,
                    penetration_recovery: 0.2,
                    max_recovery_speed: 0.25,
                },
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        results.push(ResultData {
            qd: gpu_readback::read(engine, &qd, 12),
            soft: gpu_readback::read(engine, &soft, 4),
            body: gpu_readback::read(engine, &body_vel, 16),
            status: gpu_readback::read::<u32>(engine, kernel.status(), 1)[0],
            rows: gpu_readback::read(engine, kernel.row_state(), rows.len() * 8),
        });
    }
    let cooperative = results.pop().unwrap();
    let serial = results.pop().unwrap();
    assert_eq!(cooperative.status, serial.status);
    for (label, actual, expected) in [
        ("generalized", &cooperative.qd, &serial.qd),
        ("particle", &cooperative.soft, &serial.soft),
        ("body", &cooperative.body, &serial.body),
        ("rows", &cooperative.rows, &serial.rows),
    ] {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, b)) in actual.iter().zip(expected).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "{label}[{i}]: cooperative={a}, serial={b}"
            );
        }
    }
    cooperative
}

fn oracle(offsets: [Vec3; 2], initial: [f32; 12], target: Vec3) -> Vec<f64> {
    let masses = [2.0, 2.0, 2.0, 0.7, 0.9, 1.1, 3.0, 3.0, 3.0, 1.2, 1.4, 1.6];
    let mut j = [Vec3::ZERO; 12];
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    for body in 0..2 {
        let sign = if body == 0 { 1.0 } else { -1.0 };
        for axis in 0..3 {
            j[body * 6 + axis] = axes[axis] * sign;
            j[body * 6 + 3 + axis] = axes[axis].cross(offsets[body]) * sign;
        }
    }
    let mut effective = Mat3::ZERO;
    let mut relative = Vec3::ZERO;
    for d in 0..12 {
        effective += Mat3::from_cols(j[d] * j[d].x, j[d] * j[d].y, j[d] * j[d].z) / masses[d];
        relative += j[d] * initial[d] as f64;
    }
    let impulse = effective.inverse() * (target - relative);
    (0..12)
        .map(|d| initial[d] as f64 + j[d].dot(impulse) / masses[d])
        .collect()
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_ball_joint_matches_dense_impulse_and_conserves_momentum() {
    let engine = Engine::new_gpu().unwrap();
    let initial = [
        0.7, -0.4, 0.3, 0.6, -0.8, 0.1, -0.2, 0.8, -0.7, -0.3, 0.2, 0.4,
    ];
    for offsets in [
        [Vec3::ZERO; 2],
        [Vec3::new(0.5, 0.25, -0.125), Vec3::new(-0.25, 0.125, 0.5)],
    ] {
        for error in [Vec3::ZERO, Vec3::new(0.02, -0.003, 0.001)] {
            let result = solve(&engine, offsets, error, initial, false, false, 0);
            assert_eq!(result.status, 0);
            let target = (-error * 20.0).clamp(Vec3::splat(-0.25), Vec3::splat(0.25));
            let expected = oracle(offsets, initial, target);
            for (i, (actual, expected)) in result.qd.iter().zip(expected).enumerate() {
                assert!(
                    (*actual as f64 - expected).abs() < 3e-5,
                    "DOF{i}: {actual} != {expected}"
                );
            }
            for axis in 0..3 {
                let before = 2.0 * initial[axis] + 3.0 * initial[6 + axis];
                let after = 2.0 * result.qd[axis] + 3.0 * result.qd[6 + axis];
                assert!((before - after).abs() < 2e-5);
            }
            let point_velocity = |body: usize| {
                let q = &result.qd[body * 6..];
                Vec3::new(q[0] as f64, q[1] as f64, q[2] as f64)
                    + Vec3::new(q[3] as f64, q[4] as f64, q[5] as f64).cross(offsets[body])
            };
            assert!((point_velocity(0) - point_velocity(1) - target).length() < 3e-5);
            if error == Vec3::ZERO {
                let momentum = |v: &[f32]| {
                    let mut total = Vec3::ZERO;
                    for (body, inertia) in [[0.7, 0.9, 1.1], [1.2, 1.4, 1.6]].iter().enumerate() {
                        let v = &v[body * 6..];
                        let linear = Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64);
                        let spin = Vec3::new(
                            v[3] as f64 * inertia[0],
                            v[4] as f64 * inertia[1],
                            v[5] as f64 * inertia[2],
                        );
                        total += spin
                            + (-offsets[body]).cross(linear * if body == 0 { 2.0 } else { 3.0 });
                    }
                    total
                };
                assert!((momentum(&initial) - momentum(&result.qd)).length() < 3e-5);
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_ball_joint_couples_contact_and_rejects_invalid_constraints() {
    let engine = Engine::new_gpu().unwrap();
    let result = solve(
        &engine,
        [Vec3::ZERO; 2],
        Vec3::ZERO,
        [0.0; 12],
        false,
        true,
        0,
    );
    assert_eq!(result.status, 0);
    for v in [result.qd[1], result.qd[7], result.soft[1]] {
        assert!((v + 0.5).abs() < 2e-5);
    }
    assert!((result.rows[4] - 2.5).abs() < 2e-5);
    let mut initial = [0.0; 12];
    initial[1] = 1.0;
    initial[7] = -1.0;
    let planar = solve(
        &engine,
        [Vec3::ZERO; 2],
        Vec3::ZERO,
        initial,
        true,
        false,
        0,
    );
    assert_eq!(planar.status, 0);
    assert!((planar.qd[1] + 0.2).abs() < 2e-5 && (planar.qd[7] + 0.2).abs() < 2e-5);
    let incompatible = solve(
        &engine,
        [Vec3::ZERO; 2],
        Vec3::X * 0.01,
        initial,
        true,
        false,
        0,
    );
    assert_ne!(incompatible.status & 2, 0);
    let partial = solve(
        &engine,
        [Vec3::ZERO; 2],
        Vec3::Z * 0.01,
        initial,
        true,
        false,
        0,
    );
    assert_eq!(partial.status, 2);
    assert!((partial.qd[1] + 0.2).abs() < 2e-5 && (partial.qd[7] + 0.2).abs() < 2e-5);
    assert_eq!(&partial.rows[4..8], &[0.0; 4]);
    let mut extreme = [0.0; 12];
    extreme[1] = 1e20;
    extreme[7] = -1e20;
    let overflow = solve(
        &engine,
        [Vec3::ZERO; 2],
        Vec3::ZERO,
        extreme,
        false,
        false,
        0,
    );
    assert_eq!(overflow.status, 4);
    assert!(overflow.qd.iter().all(|v| v.is_finite()));
    assert_ne!(overflow.qd, extreme);
    assert_eq!(&overflow.rows[4..8], &[0.0; 4]);
    assert_eq!(overflow.body, [0.0; 16]);
    for fault in 1..=3 {
        let invalid = solve(
            &engine,
            [Vec3::ZERO; 2],
            Vec3::ZERO,
            initial,
            false,
            false,
            fault,
        );
        assert_ne!(invalid.status & 2, 0, "fault{fault}");
        assert_eq!(invalid.qd, initial);
    }
}

#[test]
fn ball_joint_validates_body_ids_and_anchors() {
    let valid = SoftRigidBallJoint {
        body_a: 0,
        body_b: 1,
        anchor_a: [0.0; 3],
        anchor_b: [0.0; 3],
    };
    assert!(valid.row(2).is_ok());
    for bad in [
        SoftRigidBallJoint { body_a: 1, ..valid },
        SoftRigidBallJoint { body_b: 2, ..valid },
        SoftRigidBallJoint {
            anchor_a: [f32::NAN; 3],
            ..valid
        },
        SoftRigidBallJoint {
            anchor_b: [f32::INFINITY; 3],
            ..valid
        },
    ] {
        assert!(bad.row(2).is_err());
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn fem_four_bar_closure_survives_gpu_substeps_and_contact_gather() {
    use molla_core::{BodyId, WorldId};
    use molla_kernels::soft_rigid_contact::SoftRigidShapeGpu;
    use molla_math::{SpatialInertia, Transform};
    use molla_sim::{BodyParams, ModelBuilder};
    use molla_solvers::fem_rigid_gpu::FemRigidGpuSystem;

    let engine = Engine::new_gpu().unwrap();
    let mut gaps = Vec::new();
    for closed in [false, true] {
        let mut builder = ModelBuilder::new();
        builder.begin_articulation();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        let mut bodies = Vec::new();
        for (mass, com) in [
            (10.0, Vec3::ZERO),
            (1.0, Vec3::new(0.0, -0.25, 0.0)),
            (1.0, Vec3::new(0.0, -0.25, 0.0)),
            (1.0, Vec3::new(0.5, 0.0, 0.0)),
        ] {
            bodies.push(
                builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
                    mass,
                    com,
                    inertia: Mat3::from_diagonal(Vec3::splat(0.1 * mass)),
                })),
            );
        }
        builder
            .add_joint_free(
                BodyId::NONE,
                bodies[0],
                Transform::IDENTITY,
                Transform::IDENTITY,
            )
            .unwrap();
        for (parent, child, position) in [
            (0, 1, Vec3::new(-0.5, 0.0, 0.0)),
            (0, 2, Vec3::new(0.5, 0.0, 0.0)),
            (1, 3, Vec3::new(0.0, -0.5, 0.0)),
        ] {
            builder
                .add_joint_revolute(
                    bodies[parent],
                    bodies[child],
                    Transform::from_translation(position),
                    Transform::IDENTITY,
                    Vec3::Z,
                )
                .unwrap();
        }
        let mut scene = fem_rotor::scene();
        scene.rigid_model = builder.build().unwrap();
        scene.rigid_state = scene.rigid_model.state().unwrap();
        scene.control = scene.rigid_model.control();
        scene.control.joint_force.host_mut().unwrap()[6] = 10.0;
        scene.shapes.push(SoftRigidShapeGpu {
            position: [100.0, 0.0, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            data: [0.1, 0.0, 0.0, 0.0],
            ids: [0, 0, 0, 0],
        });
        let loops = [SoftRigidBallJoint {
            body_a: 2,
            body_b: 3,
            anchor_a: [0.0, -0.5, 0.0],
            anchor_b: [1.0, 0.0, 0.0],
        }];
        let mut config = fem_rotor::config();
        config.contact.iterations = 64;
        let mut system = FemRigidGpuSystem::new_surface_sampled_with_ball_joints_on_device(
            engine.device().unwrap().clone(),
            engine.queue().unwrap().clone(),
            scene,
            config,
            if closed { &loops } else { &[] },
        )
        .unwrap();
        for _ in 0..128 {
            system.step(config.max_substep).unwrap();
        }
        assert_eq!(
            gpu_readback::read::<u32>(&engine, system.status_buffer(), 1)[0],
            0
        );
        assert!(system.rigid_state().body_q.host().is_err());
        let poses = fem_readback::body_poses(&engine, &system, 4);
        let a = poses[2].position + poses[2].rotation * Vec3::new(0.0, -0.5, 0.0);
        let b = poses[3].position + poses[3].rotation * Vec3::X;
        gaps.push((a - b).length());
        let velocities = gpu_readback::read::<[f32; 8]>(
            &engine,
            system.rigid_state().body_qd.device_buffer().unwrap(),
            4,
        );
        let mut momentum = Vec3::ZERO;
        for (body, v) in velocities.iter().enumerate() {
            momentum += Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
                * if body == 0 { 10.0 } else { 1.0 };
        }
        assert!(momentum.length() < 2e-4, "momentum {momentum:?}");
        println!(
            "closed={closed} anchor gap={} momentum={}",
            gaps.last().unwrap(),
            momentum.length()
        );
    }
    assert!(gaps[0] > 0.002, "open linkage did not separate: {gaps:?}");
    assert!(gaps[1] < 1e-4, "closed linkage separated: {gaps:?}");
}
