use molla_math::{Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene, SphericalConeLimits};
use molla_sim::{
    JointType,
    joint_control::{JointAxis, JointMotor, JointSoftness},
};

#[test]
fn rigid_fixed_loop_insertion_and_failed_controls_preserve_authored_state() {
    let mut scene = RigidScene::new().unwrap();
    let child = scene.insert_body(BodyDesc::default()).unwrap();
    let pose = scene.body_pose(child).unwrap();
    let frame = Transform::new(Vec3::new(0.1, -0.2, 0.3), Quat::from_rotation_y(0.4));
    let joint = scene
        .insert_rigid_loop_joint(JointDesc {
            parent: None,
            child,
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: frame,
            frame_child: frame,
        })
        .unwrap();
    assert!(scene.joint_index(joint).is_none());
    assert_eq!(scene.rigid_loop_joints().count(), 1);
    assert_eq!(scene.rigid_ball_joints().count(), 0);
    assert_eq!(scene.joint(joint).unwrap().frame_parent, frame);
    assert_eq!(scene.body_pose(child).unwrap(), pose);
    let stamp = scene.stamp();
    assert!(
        scene
            .set_joint_motor(joint, JointAxis::LinX, JointMotor::default())
            .is_err()
    );
    assert!(
        scene
            .set_joint_limits(joint, JointAxis::AngX, Some([-0.1, 0.1]))
            .is_err()
    );
    assert!(
        scene
            .set_cone_limits(joint, SphericalConeLimits::default())
            .is_err()
    );
    assert_eq!(scene.stamp(), stamp);
    scene
        .set_joint_softness(
            joint,
            Some(JointSoftness {
                natural_frequency: 4.0,
                damping_ratio: 1.0,
            }),
        )
        .unwrap();
    assert_eq!(scene.rigid_loop_joints().count(), 0);
    scene.set_joint_softness(joint, None).unwrap();
    assert_eq!(scene.rigid_loop_joints().count(), 1);
    assert_eq!(scene.body_pose(child).unwrap(), pose);
}

#[test]
fn rigid_axis_loops_author_limits_and_both_motor_models_atomically() {
    for (kind, axis) in [
        (JointType::Revolute, JointAxis::AngX),
        (JointType::Prismatic, JointAxis::LinX),
        (JointType::D6, JointAxis::AngZ),
    ] {
        let mut scene = RigidScene::new().unwrap();
        let child = scene.insert_body(BodyDesc::default()).unwrap();
        let joint = scene
            .insert_rigid_loop_joint(JointDesc {
                parent: None,
                child,
                kind,
                axis: Vec3::new(1.0, 2.0, -3.0),
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        let pose = scene.body_pose(child).unwrap();
        scene
            .set_joint_limits(joint, axis, Some([-0.1, 0.2]))
            .unwrap();
        assert_eq!(
            scene.joint_axis_settings(joint, axis).unwrap().limits,
            Some([-0.1, 0.2])
        );
        if kind == JointType::D6 {
            scene
                .set_joint_locked_axes(joint, molla_sim::joint_control::JointAxes::LIN)
                .unwrap();
            assert_eq!(
                scene.joint_locked_axes(joint).unwrap(),
                molla_sim::joint_control::JointAxes::LIN
            );
        }
        let stamp = scene.stamp();
        let motor = JointMotor {
            mode: molla_sim::JointTargetMode::Effort,
            target_position: 1.0,
            model: molla_sim::joint_control::MotorModel::Acceleration,
            ..Default::default()
        };
        assert!(
            scene
                .set_joint_motor(
                    joint,
                    axis,
                    JointMotor {
                        max_force: f64::NAN,
                        ..motor
                    }
                )
                .is_err()
        );
        assert_eq!(scene.stamp(), stamp);
        scene.set_joint_motor(joint, axis, motor).unwrap();
        assert_eq!(
            scene.joint_axis_settings(joint, axis).unwrap().motor,
            Some(motor)
        );
        scene.set_joint_brake(joint, axis, 1.0).unwrap();
        scene
            .set_joint_softness(
                joint,
                Some(JointSoftness {
                    natural_frequency: 4.0,
                    damping_ratio: 1.0,
                }),
            )
            .unwrap();
        scene.set_joint_motor(joint, axis, motor).unwrap();
        scene.set_joint_softness(joint, None).unwrap();
        assert!(scene.joint_softness(joint).unwrap().is_none());
        assert_eq!(
            scene.joint_axis_settings(joint, axis).unwrap().motor,
            Some(motor)
        );
        scene
            .set_joint_motor(joint, axis, JointMotor::default())
            .unwrap();
        scene.set_joint_softness(joint, None).unwrap();
        assert_eq!(scene.body_pose(child).unwrap(), pose);
        assert!(scene.joint_index(joint).is_none());
    }
}

#[test]
fn rigid_ball_loop_configuration_is_transactional() {
    let mut scene = RigidScene::new().unwrap();
    let child = scene.insert_body(BodyDesc::default()).unwrap();
    let desc = JointDesc {
        parent: None,
        child,
        kind: JointType::Ball,
        axis: Vec3::Y,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
    };
    let joint = scene.insert_rigid_loop_joint(desc.clone()).unwrap();
    let stamp = scene.stamp();
    let revision = scene.revision();
    assert!(
        scene
            .insert_rigid_loop_joint(JointDesc {
                kind: JointType::Free,
                ..desc
            })
            .is_err()
    );
    assert!(
        scene
            .set_joint_limits(joint, JointAxis::AngX, Some([-0.2, 0.2]))
            .is_err()
    );
    assert!(
        scene
            .set_joint_motor(
                joint,
                JointAxis::LinX,
                JointMotor {
                    mode: molla_sim::JointTargetMode::Effort,
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert!(scene.set_joint_brake(joint, JointAxis::LinX, 1.0).is_err());
    assert_eq!(scene.stamp(), stamp);
    assert_eq!(scene.revision(), revision);
    scene.set_joint_softness(joint, None).unwrap();
    assert_eq!(scene.stamp(), stamp);
    let pose = scene.body_pose(child).unwrap();
    let velocity = SpatialVector::new(Vec3::X * 0.2, Vec3::Y * 0.3);
    scene.set_body_velocity(child, velocity).unwrap();
    scene
        .set_joint_softness(
            joint,
            Some(JointSoftness {
                natural_frequency: 30.0,
                damping_ratio: 1.0,
            }),
        )
        .unwrap();
    assert_eq!(scene.rigid_ball_joints().count(), 0);
    scene
        .set_joint_limits(joint, JointAxis::AngX, Some([-0.2, 0.2]))
        .unwrap();
    let stamp = scene.stamp();
    assert!(scene.set_joint_softness(joint, None).is_err());
    assert_eq!(scene.stamp(), stamp);
    scene
        .set_joint_limits(joint, JointAxis::AngX, None)
        .unwrap();
    scene
        .set_joint_motor(
            joint,
            JointAxis::AngX,
            JointMotor {
                mode: molla_sim::JointTargetMode::Effort,
                ..Default::default()
            },
        )
        .unwrap();
    scene.set_joint_brake(joint, JointAxis::AngX, 1.0).unwrap();
    scene.set_joint_softness(joint, None).unwrap();
    assert_eq!(
        scene
            .joint_axis_settings(joint, JointAxis::AngX)
            .unwrap()
            .brake,
        1.0
    );
    assert_eq!(
        scene
            .joint_axis_settings(joint, JointAxis::AngX)
            .unwrap()
            .motor
            .unwrap()
            .mode,
        molla_sim::JointTargetMode::Effort
    );
    scene
        .set_joint_motor(joint, JointAxis::AngX, JointMotor::default())
        .unwrap();
    scene.set_joint_softness(joint, None).unwrap();
    assert_eq!(scene.rigid_ball_joints().count(), 1);
    assert_eq!(scene.body_pose(child).unwrap(), pose);
    assert_eq!(scene.body_velocity(child).unwrap(), velocity);
}

#[test]
fn cone_edits_preserve_tree_and_loop_handles_through_compaction() {
    for external in [false, true] {
        let mut scene = RigidScene::new().unwrap();
        let unrelated = scene.insert_body(BodyDesc::default()).unwrap();
        let parent = scene.insert_body(BodyDesc::default()).unwrap();
        let child = scene.insert_body(BodyDesc::default()).unwrap();
        let desc = JointDesc {
            parent: Some(parent),
            child,
            kind: JointType::Ball,
            axis: Vec3::Y,
            frame_parent: Transform::from_rotation(Quat::from_rotation_z(0.7)),
            frame_child: Transform::from_rotation(Quat::from_rotation_x(-0.3)),
        };
        let joint = if external {
            scene
                .insert_loop_joint(
                    desc.clone(),
                    JointSoftness {
                        natural_frequency: 10.0,
                        damping_ratio: 1.0,
                    },
                )
                .unwrap()
        } else {
            scene.insert_joint(desc.clone()).unwrap()
        };
        assert_eq!(
            scene.cone_limits(joint).unwrap(),
            SphericalConeLimits::default()
        );
        let limits = SphericalConeLimits {
            angle0: None,
            angle1: Some(0.7),
        };
        let activation = [
            scene.activation_stamp(parent),
            scene.activation_stamp(child),
            scene.activation_stamp(unrelated),
        ];
        let model = scene.model_stamp();
        scene.set_cone_limits(joint, limits).unwrap();
        assert_ne!(scene.model_stamp(), model);
        assert_ne!(scene.activation_stamp(parent), activation[0]);
        assert_ne!(scene.activation_stamp(child), activation[1]);
        assert_eq!(scene.activation_stamp(unrelated), activation[2]);
        let stamp = scene.stamp();
        let revision = scene.revision();
        let activation = scene.activation_stamp(child);
        scene.set_cone_limits(joint, limits).unwrap();
        assert_eq!(scene.stamp(), stamp);
        assert_eq!(scene.revision(), revision);
        assert_eq!(scene.activation_stamp(child), activation);
        scene.remove_body(unrelated).unwrap();
        assert_eq!(scene.cone_limits(joint).unwrap(), limits);
        assert_eq!(scene.joint(joint), Some(&desc));
        assert_eq!(scene.cone_joints().count(), 1);
        scene
            .set_cone_limits(joint, SphericalConeLimits::default())
            .unwrap();
        assert_eq!(scene.cone_joints().count(), 0);
        scene.set_cone_limits(joint, limits).unwrap();
        if external {
            scene.remove_joint(joint).unwrap();
        } else {
            scene.remove_body(parent).unwrap();
        }
        assert_eq!(scene.cone_joints().count(), 0);
        assert!(scene.cone_limits(joint).is_err());
        assert!(scene.set_cone_limits(joint, limits).is_err());
    }
}

#[test]
fn invalid_cone_mutations_are_atomic() {
    let mut scene = RigidScene::new().unwrap();
    let child = scene.insert_body(BodyDesc::default()).unwrap();
    let desc = JointDesc {
        parent: None,
        child,
        kind: JointType::Ball,
        axis: Vec3::X,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
    };
    let joint = scene.insert_joint(desc.clone()).unwrap();
    let stamp = scene.stamp();
    let activation = scene.activation_stamp(child);
    for angle in [f64::NAN, f64::INFINITY, -0.1, std::f64::consts::PI + 1e-10] {
        assert!(
            scene
                .set_cone_limits(
                    joint,
                    SphericalConeLimits {
                        angle0: Some(angle),
                        angle1: None
                    }
                )
                .is_err()
        );
        assert_eq!(scene.stamp(), stamp);
        assert_eq!(scene.activation_stamp(child), activation);
        assert_eq!(
            scene.cone_limits(joint).unwrap(),
            SphericalConeLimits::default()
        );
    }
    scene.remove_joint(joint).unwrap();
    let fixed = scene
        .insert_joint(JointDesc {
            kind: JointType::Fixed,
            ..desc
        })
        .unwrap();
    let stamp = scene.stamp();
    assert!(scene.cone_limits(fixed).is_err());
    assert!(
        scene
            .set_cone_limits(fixed, SphericalConeLimits::default())
            .is_err()
    );
    assert_eq!(scene.stamp(), stamp);
}
