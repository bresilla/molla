use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyKind, JointDesc, RigidScene};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rnea::{
    inverse_dynamics_wrenches, inverse_dynamics_wrenches_with_boundary_accelerations,
};

struct Fixture {
    scene: RigidScene,
    boundary: usize,
    child: usize,
    joint: usize,
    qdd: Vec<f64>,
    acceleration: Vec<SpatialVector>,
    external: Vec<SpatialVector>,
    expected: SpatialVector,
}

fn fixture(kind: JointType, shift: Vec3, rotation: Quat) -> Fixture {
    let mut scene = RigidScene::new().unwrap();
    let gravity = Vec3::new(0.3, -9.81, -0.7);
    scene.set_gravity(gravity).unwrap();
    let parent_com = Vec3::Y * 0.25;
    let parent = scene
        .insert_body(BodyDesc {
            kind: BodyKind::Kinematic,
            params: BodyParams {
                pose: Transform {
                    position: shift,
                    rotation,
                },
                ..Default::default()
            },
            additional_mass: Some(MassProperties {
                mass: 3.,
                local_com: parent_com,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let inertia = Mat3::from_cols(
        Vec3::new(2., 0.1, 0.2),
        Vec3::new(0.1, 3., -0.3),
        Vec3::new(0.2, -0.3, 4.),
    );
    let com = Vec3::new(0.4, -0.8, 0.1);
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams {
                pose: Transform {
                    position: shift + rotation * parent_com,
                    rotation,
                },
                ..Default::default()
            },
            additional_mass: Some(MassProperties {
                mass: 2.,
                local_com: com,
                inertia,
            }),
            ..Default::default()
        })
        .unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(parent),
            child: body,
            kind,
            axis: Vec3::Z,
            frame_parent: Transform::from_translation(parent_com),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let omega_base = rotation * Vec3::new(0.4, -0.5, 1.2);
    scene
        .set_body_velocity(
            parent,
            SpatialVector::new(Vec3::new(1., -2., 3.), omega_base),
        )
        .unwrap();
    let joint = scene.joint_index(joint).unwrap().index();
    let base = scene.model().joint_dof_offset.host().unwrap()[joint] as usize;
    let mut qd = vec![0.; scene.model().joint_dof_total];
    let mut qdd = qd.clone();
    let (relative_omega, relative_alpha) = match kind {
        JointType::Revolute => {
            qd[base] = 0.6;
            qdd[base] = -0.4;
            (Vec3::Z * 0.6, Vec3::Z * (-0.4))
        }
        JointType::Ball => {
            qd[base..base + 3].copy_from_slice(&[0.6, -0.3, 0.9]);
            qdd[base..base + 3].copy_from_slice(&[-0.4, 0.7, 0.2]);
            (Vec3::new(0.6, -0.3, 0.9), Vec3::new(-0.4, 0.7, 0.2))
        }
        _ => (Vec3::ZERO, Vec3::ZERO),
    };
    scene
        .set_generalized_velocities_with_wake(&qd, true)
        .unwrap();
    let boundary = scene.body_index(parent).unwrap().index();
    let child = scene.body_index(body).unwrap().index();
    let mut acceleration = vec![SpatialVector::ZERO; scene.model().body_count];
    acceleration[boundary] = SpatialVector::new(
        rotation * Vec3::new(1.3, -0.7, 0.4),
        rotation * Vec3::new(-0.2, 0.8, 0.5),
    );
    let mut external = vec![SpatialVector::ZERO; scene.model().body_count];
    external[child] = SpatialVector::new(Vec3::new(2., -3., 1.), Vec3::new(0.4, 0.1, -0.7));
    let arm = rotation * com;
    let omega = omega_base + rotation * relative_omega;
    let alpha = acceleration[boundary].angular
        + rotation * relative_alpha
        + omega_base.cross(rotation * relative_omega);
    let a = acceleration[boundary].linear + alpha.cross(arm) + omega.cross(omega.cross(arm));
    let r = Mat3::from_quat(rotation);
    let iw = r * inertia * r.transpose();
    let force = (a - gravity) * 2. - external[child].linear;
    let torque = iw * alpha + omega.cross(iw * omega) - external[child].angular + arm.cross(force);
    Fixture {
        scene,
        boundary,
        child,
        joint,
        qdd,
        acceleration,
        external,
        expected: SpatialVector::new(force, torque),
    }
}

fn near(actual: SpatialVector, expected: SpatialVector, tolerance: f64) {
    assert!(
        (actual.linear - expected.linear).length() < tolerance,
        "actual={actual:?} expected={expected:?}"
    );
    assert!(
        (actual.angular - expected.angular).length() < tolerance,
        "actual={actual:?} expected={expected:?}"
    );
}

#[test]
fn prescribed_com_and_angular_acceleration_match_newton_euler() {
    for kind in [JointType::Fixed, JointType::Revolute, JointType::Ball] {
        for shift in [Vec3::ZERO, Vec3::new(1024., -2048., 4096.)] {
            for rotation in [
                Quat::IDENTITY,
                Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.4),
            ] {
                let f = fixture(kind, shift, rotation);
                let result = inverse_dynamics_wrenches_with_boundary_accelerations(
                    f.scene.model(),
                    f.scene.state(),
                    0,
                    &f.qdd,
                    &f.external,
                    &f.acceleration,
                )
                .unwrap();
                near(result.joints[f.joint], f.expected, 1e-8);
                let legacy = inverse_dynamics_wrenches(
                    f.scene.model(),
                    f.scene.state(),
                    0,
                    &f.qdd,
                    &f.external,
                )
                .unwrap();
                assert!((legacy.joints[f.joint].linear - f.expected.linear).length() > 0.1);
            }
        }
    }
}

#[test]
fn zero_boundary_acceleration_preserves_legacy_result_and_invalid_inputs_fail() {
    let mut f = fixture(JointType::Ball, Vec3::ZERO, Quat::IDENTITY);
    let zeros = vec![SpatialVector::ZERO; f.acceleration.len()];
    let legacy =
        inverse_dynamics_wrenches(f.scene.model(), f.scene.state(), 0, &f.qdd, &f.external)
            .unwrap();
    let zero = inverse_dynamics_wrenches_with_boundary_accelerations(
        f.scene.model(),
        f.scene.state(),
        0,
        &f.qdd,
        &f.external,
        &zeros,
    )
    .unwrap();
    assert_eq!(zero.efforts, legacy.efforts);
    assert_eq!(zero.joints, legacy.joints);
    for case in 0..3 {
        let mut invalid = f.acceleration.clone();
        match case {
            0 => {
                invalid.pop();
            }
            1 => invalid[f.boundary].angular.x = f64::NAN,
            _ => invalid[f.child].linear.x = 1.,
        }
        assert!(
            inverse_dynamics_wrenches_with_boundary_accelerations(
                f.scene.model(),
                f.scene.state(),
                0,
                &f.qdd,
                &f.external,
                &invalid
            )
            .is_err()
        );
    }
    let parent = f
        .scene
        .bodies()
        .find(|(_, body)| body.kind == BodyKind::Kinematic)
        .unwrap()
        .0;
    f.scene.set_body_kind(parent, BodyKind::Fixed).unwrap();
    let mut invalid = vec![SpatialVector::ZERO; f.scene.model().body_count];
    invalid[f.scene.body_index(parent).unwrap().index()].angular = Vec3::X;
    assert!(
        inverse_dynamics_wrenches_with_boundary_accelerations(
            f.scene.model(),
            f.scene.state(),
            0,
            &f.qdd,
            &f.external,
            &invalid
        )
        .is_err()
    );
}

#[cfg(feature = "gpu")]
mod gpu {
    use super::*;
    use molla_compute::Engine;
    use molla_kernels::{ReactionWrenchBuffers, ReactionWrenchDispatch, ReactionWrenchKernel};
    use molla_math::GpuSpatialVector;

    fn read<T: bytemuck::Pod>(engine: &Engine, source: &wgpu::Buffer, count: usize) -> Vec<T> {
        if count == 0 {
            return Vec::new();
        }
        let device = engine.device().unwrap();
        let size = (count * std::mem::size_of::<T>()) as u64;
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("boundary reaction readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
        engine.queue().unwrap().submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = staging.slice(..).get_mapped_range();
        let result = bytes
            .chunks_exact(std::mem::size_of::<T>())
            .map(bytemuck::pod_read_unaligned)
            .collect();
        drop(bytes);
        staging.unmap();
        result
    }

    fn upload_vectors(engine: &Engine, target: &wgpu::Buffer, values: &[SpatialVector]) {
        let data: Vec<_> = values.iter().copied().map(GpuSpatialVector::from).collect();
        engine
            .queue()
            .unwrap()
            .write_buffer(target, 0, bytemuck::cast_slice(&data));
    }

    fn prepare(
        engine: &Engine,
        fixture: &mut Fixture,
    ) -> (
        ReactionWrenchKernel,
        ReactionWrenchBuffers,
        ReactionWrenchDispatch,
    ) {
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        fixture.scene.upload_model_to_device(device, queue).unwrap();
        fixture
            .scene
            .state_mut()
            .upload_to_device(device, queue)
            .unwrap();
        let kernel = ReactionWrenchKernel::new(engine, fixture.scene.model()).unwrap();
        let buffers = kernel.create_buffers(engine).unwrap();
        queue.write_buffer(
            &buffers.acceleration,
            0,
            bytemuck::cast_slice(&fixture.qdd.iter().map(|&x| x as f32).collect::<Vec<_>>()),
        );
        upload_vectors(engine, &buffers.external, &fixture.external);
        upload_vectors(
            engine,
            &buffers.boundary_acceleration,
            &fixture.acceleration,
        );
        let dispatch = kernel
            .prepare(
                engine,
                fixture.scene.model(),
                fixture.scene.state(),
                &buffers,
            )
            .unwrap();
        (kernel, buffers, dispatch)
    }

    fn run(
        engine: &Engine,
        dispatch: &ReactionWrenchDispatch,
        buffers: &ReactionWrenchBuffers,
    ) -> u32 {
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        dispatch.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        read::<u32>(engine, &buffers.status, 1)[0]
    }

    #[test]
    #[ignore = "requires GPU; run via make test-primitives-boundary-reactions-gpu"]
    fn prescribed_derivatives_match_analytic_and_cpu_reactions() {
        let engine = Engine::new_gpu().unwrap();
        let mut peak = 0.0_f64;
        for kind in [JointType::Fixed, JointType::Revolute, JointType::Ball] {
            for shift in [Vec3::ZERO, Vec3::new(1024., -2048., 4096.)] {
                for rotation in [
                    Quat::IDENTITY,
                    Quat::from_rotation_y(0.5) * Quat::from_rotation_x(-0.4),
                ] {
                    let mut f = fixture(kind, shift, rotation);
                    let cpu = inverse_dynamics_wrenches_with_boundary_accelerations(
                        f.scene.model(),
                        f.scene.state(),
                        0,
                        &f.qdd,
                        &f.external,
                        &f.acceleration,
                    )
                    .unwrap();
                    let (_, buffers, dispatch) = prepare(&engine, &mut f);
                    for _ in 0..2 {
                        assert_eq!(run(&engine, &dispatch, &buffers), 0);
                        let gpu = read::<GpuSpatialVector>(
                            &engine,
                            &buffers.joints,
                            f.scene.model().joint_count,
                        );
                        near(
                            SpatialVector::from(gpu[f.joint]),
                            f.expected,
                            3e-4 * (1.
                                + f.expected.linear.length().max(f.expected.angular.length())),
                        );
                        for (gpu, cpu) in gpu.into_iter().zip(&cpu.joints) {
                            let gpu = SpatialVector::from(gpu);
                            for (a, b) in [(gpu.linear, cpu.linear), (gpu.angular, cpu.angular)] {
                                let error = (a - b).length() / (1. + b.length());
                                peak = peak.max(error);
                                assert!(
                                    error < 3e-4,
                                    "kind={kind:?} shift={shift:?} gpu={gpu:?} cpu={cpu:?} error={error}"
                                );
                            }
                        }
                        let efforts = read::<f32>(&engine, &buffers.efforts, f.qdd.len());
                        for (gpu, cpu) in efforts.into_iter().zip(&cpu.efforts) {
                            assert!((gpu as f64 - cpu).abs() < 3e-4 * (1. + cpu.abs()));
                        }
                    }
                }
            }
        }
        eprintln!("prescribed boundary reactions: 12 fixtures, peak scaled error={peak:e}");
    }

    #[test]
    #[ignore = "requires GPU; run via make test-primitives-boundary-reactions-gpu"]
    fn retained_boundary_updates_fault_retirement_and_recovery() {
        let engine = Engine::new_gpu().unwrap();
        let mut f = fixture(JointType::Ball, Vec3::ZERO, Quat::IDENTITY);
        let (kernel, mut buffers, dispatch) = prepare(&engine, &mut f);
        for case in 0..5 {
            let mut acceleration = f.acceleration.clone();
            let expected_status = match case {
                0 => {
                    acceleration.fill(SpatialVector::ZERO);
                    0
                }
                1 => 0,
                2 => {
                    acceleration[f.child].linear.x = 1.;
                    4
                }
                3 => {
                    acceleration[f.boundary].angular.y = f64::NAN;
                    1
                }
                _ => 0,
            };
            upload_vectors(&engine, &buffers.boundary_acceleration, &acceleration);
            assert_eq!(run(&engine, &dispatch, &buffers), expected_status);
            let gpu =
                read::<GpuSpatialVector>(&engine, &buffers.joints, f.scene.model().joint_count);
            if expected_status == 0 {
                let cpu = inverse_dynamics_wrenches_with_boundary_accelerations(
                    f.scene.model(),
                    f.scene.state(),
                    0,
                    &f.qdd,
                    &f.external,
                    &acceleration,
                )
                .unwrap();
                for (gpu, cpu) in gpu.into_iter().zip(cpu.joints) {
                    near(gpu.into(), cpu, 3e-4);
                }
            } else {
                assert!(
                    gpu.iter()
                        .all(|w| SpatialVector::from(*w) == SpatialVector::ZERO)
                );
                assert!(
                    read::<f32>(&engine, &buffers.efforts, f.qdd.len())
                        .iter()
                        .all(|&v| v == 0.)
                );
            }
        }
        #[cfg(feature = "resource-counters")]
        {
            let device = engine.device().unwrap();
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
            let baseline = counters();
            for _ in 0..5 {
                let mut encoder = device.create_command_encoder(&Default::default());
                for _ in 0..8 {
                    dispatch.encode(&mut encoder);
                }
                engine.queue().unwrap().submit([encoder.finish()]);
                assert_eq!(counters(), baseline);
            }
            eprintln!("retained nonzero boundary reaction resources={baseline:?}");
        }
        buffers.boundary_acceleration =
            engine
                .device()
                .unwrap()
                .create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 4,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                });
        assert!(
            kernel
                .prepare(&engine, f.scene.model(), f.scene.state(), &buffers)
                .is_err()
        );
        let parent = f
            .scene
            .bodies()
            .find(|(_, body)| body.kind == BodyKind::Kinematic)
            .unwrap()
            .0;
        f.scene.set_body_kind(parent, BodyKind::Fixed).unwrap();
        f.acceleration.fill(SpatialVector::ZERO);
        f.acceleration[f.scene.body_index(parent).unwrap().index()].angular = Vec3::X;
        let (_, buffers, dispatch) = prepare(&engine, &mut f);
        assert_eq!(run(&engine, &dispatch, &buffers), 4);
        assert!(
            read::<GpuSpatialVector>(&engine, &buffers.joints, f.scene.model().joint_count)
                .iter()
                .all(|&w| SpatialVector::from(w) == SpatialVector::ZERO)
        );
    }
}
