use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, RigidScene};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;
use molla_solvers::{FeatherstoneSolver, Solver};

#[cfg(feature = "gpu")]
#[path = "common/gpu_readback.rs"]
mod gpu_readback;

const DT: f64 = 0.01;

struct Case {
    body: BodyHandle,
    initial: Quat,
    expected_rate: Vec3,
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    for frame in [
        Quat::IDENTITY,
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.3),
    ] {
        for sign in [-1., 1.] {
            for pole in [false, true] {
                for two_locks in [false, true] {
                    for forbidden in [false, true] {
                        let pitch = sign
                            * if pole {
                                std::f64::consts::FRAC_PI_2
                            } else {
                                0.6
                            };
                        let relative = Quat::from_rotation_y(pitch);
                        let initial = frame * relative;
                        let tangent = if pole { Vec3::X } else { relative * Vec3::Z };
                        let normal = if pole { Vec3::Y } else { relative * Vec3::X };
                        let torque = frame * if forbidden { normal } else { tangent };
                        let body = scene
                            .insert_body(BodyDesc {
                                params: BodyParams::new()
                                    .with_pose(Transform::new(Vec3::ZERO, initial)),
                                additional_mass: Some(MassProperties {
                                    mass: 1.,
                                    local_com: Vec3::ZERO,
                                    inertia: Mat3::IDENTITY,
                                }),
                                ..Default::default()
                            })
                            .unwrap();
                        let joint = scene
                            .insert_joint(JointDesc {
                                parent: None,
                                child: body,
                                kind: JointType::D6,
                                axis: Vec3::X,
                                frame_parent: Transform::new(Vec3::ZERO, frame),
                                frame_child: Transform::IDENTITY,
                            })
                            .unwrap();
                        let locks = if !pole || two_locks {
                            JointAxes::LIN.with(JointAxis::AngX)
                        } else {
                            JointAxes::LIN
                        };
                        scene.set_joint_locked_axes(joint, locks).unwrap();
                        if pole || two_locks {
                            scene
                                .set_joint_limits(joint, JointAxis::AngY, Some([pitch, pitch]))
                                .unwrap();
                        }
                        scene
                            .add_body_wrench(body, SpatialVector::new(Vec3::ZERO, torque))
                            .unwrap();
                        cases.push(Case {
                            body,
                            initial,
                            expected_rate: if forbidden { Vec3::ZERO } else { torque * DT },
                        });
                    }
                }
            }
        }
    }
    for frame in [
        Quat::IDENTITY,
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.3),
    ] {
        for sign in [-1., 1.] {
            for locks in [
                JointAxes::LIN.with(JointAxis::AngX),
                JointAxes::LIN.with(JointAxis::AngZ),
                JointAxes::LIN.with(JointAxis::AngX).with(JointAxis::AngZ),
            ] {
                let both = locks.contains(JointAxis::AngX) && locks.contains(JointAxis::AngZ);
                for direction in 0..if both { 3 } else { 2 } {
                    let forbidden = direction != 0;
                    let x = 0.3;
                    let z = 0.2;
                    let relative = Quat::from_euler(
                        glam::EulerRot::XYZ,
                        x,
                        sign * std::f64::consts::FRAC_PI_2,
                        z,
                    );
                    let initial = frame * relative;
                    let tangent = Quat::from_rotation_x(x) * Vec3::Y;
                    let normal = Quat::from_rotation_x(x) * Vec3::Z;
                    let torque = frame
                        * if direction == 2 {
                            Vec3::X
                        } else if forbidden {
                            normal
                        } else {
                            tangent
                        };
                    let body = scene
                        .insert_body(BodyDesc {
                            params: BodyParams::new()
                                .with_pose(Transform::new(Vec3::ZERO, initial)),
                            additional_mass: Some(MassProperties {
                                mass: 1.,
                                local_com: Vec3::ZERO,
                                inertia: Mat3::IDENTITY,
                            }),
                            ..Default::default()
                        })
                        .unwrap();
                    let joint = scene
                        .insert_joint(JointDesc {
                            parent: None,
                            child: body,
                            kind: JointType::D6,
                            axis: Vec3::X,
                            frame_parent: Transform::new(Vec3::ZERO, frame),
                            frame_child: Transform::IDENTITY,
                        })
                        .unwrap();
                    scene.set_joint_locked_axes(joint, JointAxes::LIN).unwrap();
                    for (axis, value) in [(JointAxis::AngX, x), (JointAxis::AngZ, z)] {
                        if locks.contains(axis) {
                            scene
                                .set_joint_limits(joint, axis, Some([value, value]))
                                .unwrap();
                        }
                    }
                    scene
                        .add_body_wrench(body, SpatialVector::new(Vec3::ZERO, torque))
                        .unwrap();
                    cases.push(Case {
                        body,
                        initial,
                        expected_rate: if forbidden { Vec3::ZERO } else { torque * DT },
                    });
                }
            }
        }
    }
    for frame in [
        Quat::IDENTITY,
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.3),
    ] {
        for sign in [-1., 1.] {
            let angles = Vec3::new(0.3, sign * std::f64::consts::FRAC_PI_2, 0.2);
            for free in 0..3 {
                let left = match free {
                    0 => Quat::IDENTITY,
                    1 => Quat::from_rotation_x(angles.x),
                    _ => Quat::from_rotation_x(angles.x) * Quat::from_rotation_y(angles.y),
                };
                let initial =
                    frame * Quat::from_euler(glam::EulerRot::XYZ, angles.x, angles.y, angles.z);
                for forbidden in [false, true] {
                    let torque = frame
                        * left
                        * [Vec3::X, Vec3::Y, Vec3::Z][free]
                        * if forbidden { 1. } else { -1. };
                    let body = scene
                        .insert_body(BodyDesc {
                            params: BodyParams::new()
                                .with_pose(Transform::new(Vec3::ZERO, initial)),
                            additional_mass: Some(MassProperties {
                                mass: 1.,
                                local_com: Vec3::ZERO,
                                inertia: Mat3::IDENTITY,
                            }),
                            ..Default::default()
                        })
                        .unwrap();
                    let joint = scene
                        .insert_joint(JointDesc {
                            parent: None,
                            child: body,
                            kind: JointType::D6,
                            axis: Vec3::X,
                            frame_parent: Transform::new(Vec3::ZERO, frame),
                            frame_child: Transform::IDENTITY,
                        })
                        .unwrap();
                    scene.set_joint_locked_axes(joint, JointAxes::LIN).unwrap();
                    for (axis, control) in [JointAxis::AngX, JointAxis::AngY, JointAxis::AngZ]
                        .into_iter()
                        .enumerate()
                    {
                        let lower = angles[axis] - if axis == free { 0.2 } else { 0. };
                        scene
                            .set_joint_limits(joint, control, Some([lower, angles[axis]]))
                            .unwrap();
                    }
                    scene
                        .add_body_wrench(body, SpatialVector::new(Vec3::ZERO, torque))
                        .unwrap();
                    cases.push(Case {
                        body,
                        initial,
                        expected_rate: if forbidden { Vec3::ZERO } else { torque * DT },
                    });
                }
            }
        }
    }
    (scene, cases)
}

fn verify_standalone(
    scene: &RigidScene,
    state: &molla_sim::State,
    cases: &[Case],
    step: usize,
    tolerance: f64,
) {
    for (i, case) in cases.iter().enumerate() {
        let body = scene.body_index(case.body).unwrap().index();
        let pose = state.body_q.host().unwrap()[body];
        let velocity = state.body_qd.host().unwrap()[body];
        let expected =
            Quat::from_scaled_axis(case.expected_rate * (DT * step as f64)) * case.initial;
        let rate_error = (velocity.angular - case.expected_rate).length();
        let rotation_error = 2.
            * (pose.rotation.normalize() * expected.conjugate())
                .xyz()
                .length();
        println!(
            "standalone_d6_lock case={i} step={step} rate_error={rate_error:.9e} rotation_error={rotation_error:.9e}"
        );
        assert!(
            rate_error.is_finite() && rate_error <= tolerance,
            "case={i}: {velocity:?}, expected={:?}",
            case.expected_rate
        );
        assert!(rotation_error.is_finite() && rotation_error <= tolerance);
        assert!(pose.position.length() <= tolerance && velocity.linear.length() <= tolerance);
    }
}

fn inject_forbidden_rates(scene: &RigidScene, cases: &[Case], state: &mut molla_sim::State) {
    let model = scene.model();
    for case in cases.iter().filter(|case| case.expected_rate == Vec3::ZERO) {
        let body = scene.body_index(case.body).unwrap().index();
        let joint = model
            .joint_child
            .host()
            .unwrap()
            .iter()
            .position(|&child| child as usize == body)
            .unwrap();
        let rate = model.joint_x_p.host().unwrap()[joint].rotation.conjugate()
            * scene.state().body_f.host().unwrap()[body].angular
            * DT;
        let start = model.joint_dof_offset.host().unwrap()[joint] as usize + 3;
        state.joint_qd.host_mut().unwrap()[start..start + 3].copy_from_slice(&rate.to_array());
    }
}

#[test]
fn cpu_standalone_mixed_locks_and_drift_preserve_physical_tangents() {
    let (scene, cases) = fixture();
    let mut a = scene.state().clone_host().unwrap();
    let mut b = scene.model().state().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu();
    solver
        .step(
            scene.model(),
            &a,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            DT,
            &mut b,
        )
        .unwrap();
    verify_standalone(&scene, &b, &cases, 1, 1e-8);
    std::mem::swap(&mut a, &mut b);
    a.body_f.host_mut().unwrap().fill(SpatialVector::ZERO);
    b = a.clone_host().unwrap();
    solver.drift_cpu(scene.model(), &a, DT, &mut b).unwrap();
    verify_standalone(&scene, &b, &cases, 2, 1e-8);
    a = b.clone_host().unwrap();
    b = a.clone_host().unwrap();
    inject_forbidden_rates(&scene, &cases, &mut b);
    solver.drift_cpu(scene.model(), &a, DT, &mut b).unwrap();
    verify_standalone(&scene, &b, &cases, 3, 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_standalone_mixed_locks_preserve_physical_tangents() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let (mut scene, cases) = fixture();
    scene.upload_model_to_device(&device, &queue).unwrap();
    scene.upload_controls_to_device(&device, &queue).unwrap();
    let mut a = scene.state().clone_host().unwrap();
    let mut b = scene.model().state().unwrap();
    a.upload_to_device(&device, &queue).unwrap();
    b.upload_to_device(&device, &queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, scene.model()).unwrap();
    for step in 1..=2 {
        solver
            .step(
                scene.model(),
                &a,
                scene.control(),
                &molla_sim::Contacts::with_capacity(0),
                DT,
                &mut b,
            )
            .unwrap();
        b.body_q.download(&device, &queue).unwrap();
        b.body_qd.download(&device, &queue).unwrap();
        let status = gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0];
        assert_eq!(status, 0, "standalone angular-lock fault at step {step}");
        verify_standalone(&scene, &b, &cases, step, 1e-3);
        std::mem::swap(&mut a, &mut b);
        a.body_f.host_mut().unwrap().fill(SpatialVector::ZERO);
        a.body_f.upload(&queue).unwrap();
    }
    a.joint_q.download(&device, &queue).unwrap();
    a.joint_qd.download(&device, &queue).unwrap();
    b = a.clone_host().unwrap();
    inject_forbidden_rates(&scene, &cases, &mut b);
    b.upload_to_device(&device, &queue).unwrap();
    solver
        .drift_gpu(scene.model(), &a, scene.control(), DT, &mut b)
        .unwrap();
    b.body_q.download(&device, &queue).unwrap();
    b.body_qd.download(&device, &queue).unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    verify_standalone(&scene, &b, &cases, 3, 1e-3);
}

fn verify(world: &mut RigidWorld, cases: &[Case], tolerance: f64) {
    world.solver.substeps = 1;
    for step in 1..=2 {
        world.step(DT).unwrap();
        assert!(world.quarantined_bodies().is_empty());
        assert!(world.joint_break_events().is_empty());
        for (i, case) in cases.iter().enumerate() {
            let velocity = world.scene.body_velocity(case.body).unwrap();
            let pose = world.scene.body_pose(case.body).unwrap();
            let expected =
                Quat::from_scaled_axis(case.expected_rate * (DT * step as f64)) * case.initial;
            let rate_error = (velocity.angular - case.expected_rate).length();
            let rotation_error = 2.
                * (pose.rotation.normalize() * expected.conjugate())
                    .xyz()
                    .length();
            println!(
                "mixed_d6_lock case={i} step={step} rate_error={rate_error:.9e} rotation_error={rotation_error:.9e}"
            );
            assert!(
                rate_error.is_finite() && rate_error <= tolerance,
                "case={i}: {velocity:?}, expected={:?}",
                case.expected_rate
            );
            assert!(rotation_error.is_finite() && rotation_error <= tolerance);
            assert!(pose.position.length() <= tolerance && velocity.linear.length() <= tolerance);
            world.scene.reset_body_forces(case.body).unwrap();
        }
    }
}

#[test]
fn cpu_mixed_euler_locks_preserve_physical_tangent_response() {
    let (scene, cases) = fixture();
    verify(&mut RigidWorld::from_scene(scene).unwrap(), &cases, 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_mixed_euler_locks_preserve_physical_tangent_response() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let (scene, cases) = fixture();
    verify(
        &mut RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &cases,
        1e-3,
    );
}
