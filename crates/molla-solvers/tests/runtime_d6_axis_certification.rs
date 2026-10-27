use molla_math::{Mat3, Quat, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, RigidScene};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

struct Case {
    body: BodyHandle,
    parent: Transform,
    child: Transform,
    axis: usize,
    mode: usize,
    sign: f64,
    local_com: Vec3,
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    for properties in [
        MassProperties {
            mass: 1.,
            local_com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        },
        MassProperties {
            mass: 1.,
            local_com: Vec3::new(0.4, -0.3, 0.2),
            inertia: Mat3::from_cols(
                Vec3::new(3., 0.2, 0.3),
                Vec3::new(0.2, 4., 0.4),
                Vec3::new(0.3, 0.4, 5.),
            ),
        },
        MassProperties {
            mass: 17.,
            local_com: Vec3::new(0.4, -0.3, 0.2),
            inertia: Mat3::from_cols(
                Vec3::new(3., 0.2, 0.3),
                Vec3::new(0.2, 4., 0.4),
                Vec3::new(0.3, 0.4, 5.),
            ) * 17.,
        },
        MassProperties {
            mass: 83.,
            local_com: Vec3::new(0.4, -0.3, 0.2),
            inertia: Mat3::from_cols(
                Vec3::new(3., 0.2, 0.3),
                Vec3::new(0.2, 4., 0.4),
                Vec3::new(0.3, 0.4, 5.),
            ) * 83.,
        },
    ] {
        for axis in 0..6 {
            for mode in 0..3 {
                for sign in [-1., 1.] {
                    let parent = Transform::new(
                        Vec3::new(cases.len() as f64 * 2., 3., -1.),
                        Quat::from_rotation_y(0.2) * Quat::from_rotation_z(-0.15),
                    );
                    let child = Transform::new(
                        if properties.local_com == Vec3::ZERO {
                            Vec3::ZERO
                        } else {
                            Vec3::new(-0.3, 0.5, -0.9)
                        },
                        Quat::from_rotation_x(0.1),
                    );
                    let initial = parent.compose(&child.inverse());
                    let mut local_axis = Vec3::ZERO;
                    local_axis[axis % 3] = 1.;
                    let normal = parent.rotation * local_axis;
                    let arm = initial.rotation * (properties.local_com - child.position);
                    let rotation = Mat3::from_quat(initial.rotation);
                    let world_inertia = rotation * properties.inertia * rotation.transpose();
                    let effective = if axis < 3 {
                        properties.mass
                    } else {
                        normal.dot(world_inertia * normal)
                            + properties.mass * arm.cross(normal).length_squared()
                    };
                    let body = scene
                        .insert_body(BodyDesc {
                            params: BodyParams {
                                pose: initial,
                                ..Default::default()
                            },
                            additional_mass: Some(properties),
                            ..Default::default()
                        })
                        .unwrap();
                    let joint = scene
                        .insert_joint(JointDesc {
                            parent: None,
                            child: body,
                            kind: JointType::D6,
                            axis: Vec3::X,
                            frame_parent: parent,
                            frame_child: child,
                        })
                        .unwrap();
                    let locks = if mode == 2 {
                        JointAxes::ALL
                    } else {
                        JointAxes(JointAxes::ALL.0 & !(1 << axis))
                    };
                    scene.set_joint_locked_axes(joint, locks).unwrap();
                    if mode == 1 {
                        scene
                            .set_joint_limits(joint, JointAxis::ALL[axis], Some([-0.1, 0.1]))
                            .unwrap();
                    }
                    scene
                        .set_joint_motor(
                            joint,
                            JointAxis::ALL[axis],
                            JointMotor {
                                mode: JointTargetMode::Effort,
                                target_position: sign * effective,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                    cases.push(Case {
                        body,
                        parent,
                        child,
                        axis,
                        mode,
                        sign,
                        local_com: properties.local_com,
                    });
                }
            }
        }
    }
    assert_eq!(cases.len(), 144);
    (scene, cases)
}

fn verify(world: &mut RigidWorld, cases: &[Case], tolerance: f64) {
    world.solver.substeps = 2;
    let dt = 1. / 240.;
    let mut peak_pose: f64 = 0.;
    let mut peak_rate: f64 = 0.;
    for frame in 1..=240 {
        world.step(dt).unwrap();
        let time = frame as f64 * dt;
        let free_coordinate = 0.5 * time * (time + dt / 2.);
        for case in cases {
            let coordinate = case.sign
                * match case.mode {
                    0 => free_coordinate,
                    1 => free_coordinate.min(0.1),
                    _ => 0.,
                };
            let mut direction = Vec3::ZERO;
            direction[case.axis % 3] = 1.;
            let relative = if case.axis < 3 {
                Transform::from_translation(direction * coordinate)
            } else {
                Transform::from_rotation(Quat::from_axis_angle(direction, coordinate))
            };
            let expected = case
                .parent
                .compose(&relative)
                .compose(&case.child.inverse());
            let actual = world.scene.body_pose(case.body).unwrap();
            let error = (actual.position - expected.position).length()
                + actual
                    .rotation
                    .normalize()
                    .angle_between(expected.rotation.normalize())
                    .abs();
            peak_pose = peak_pose.max(error);
            assert!(
                error < tolerance,
                "frame={frame} axis={} mode={} sign={} pose_error={error}",
                case.axis,
                case.mode,
                case.sign
            );
            let rate = if case.mode == 2 || (case.mode == 1 && free_coordinate >= 0.1) {
                0.
            } else {
                case.sign * time
            };
            let expected_rate = case.parent.rotation * direction * rate;
            let actual = world.scene.body_velocity(case.body).unwrap();
            let arm = expected.rotation * (case.local_com - case.child.position);
            let expected_linear = if case.axis < 3 {
                expected_rate
            } else {
                expected_rate.cross(arm)
            };
            let expected_angular = if case.axis < 3 {
                Vec3::ZERO
            } else {
                expected_rate
            };
            let error = (actual.linear - expected_linear).length()
                + (actual.angular - expected_angular).length();
            peak_rate = peak_rate.max(error);
            assert!(
                error < tolerance,
                "frame={frame} axis={} mode={} sign={} rate_error={error}",
                case.axis,
                case.mode,
                case.sign
            );
        }
    }
    println!(
        "d6_axis_cases={} frames=240 substeps=2 masses=1,17,83 full_com_tensor=true peak_pose={peak_pose} peak_rate={peak_rate}",
        cases.len()
    );
}

#[test]
fn d6_each_axis_free_bounded_locked_matches_analytic_motion() {
    let (scene, cases) = fixture();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    verify(&mut world, &cases, 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-primitives-multidof-feedback-gpu"]
fn gpu_d6_each_axis_free_bounded_locked_matches_analytic_motion() {
    let (scene, cases) = fixture();
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software);
    println!("adapter={adapter:?}");
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    verify(&mut world, &cases, 1e-3);
}
