use super::*;

#[test]
fn sleeping_kind_roundtrip_preserves_sleep_and_unrelated_bodies() {
    for intermediate in [BodyKind::Fixed, BodyKind::Kinematic] {
        for wake in [false, true] {
            let (mut world, a, other) = two();
            world.scene.set_gravity(Vec3::ZERO).unwrap();
            world.sleep_bodies(&[a, other]).unwrap();
            let pose = world.scene.body_pose(a).unwrap();
            world
                .scene
                .set_body_kind_with_wake(a, intermediate, false)
                .unwrap();
            for _ in 0..3 {
                world.step(0.001).unwrap();
                assert!(world.is_sleeping(a) && world.is_sleeping(other));
                assert_eq!(world.scene.body_pose(a).unwrap(), pose);
            }
            world
                .scene
                .set_body_kind_with_wake(a, BodyKind::Dynamic, wake)
                .unwrap();
            assert_eq!(world.is_sleeping(a), !wake);
            assert!(world.is_sleeping(other));
            world.step(0.001).unwrap();
            assert_eq!(world.is_sleeping(a), !wake);
            assert!(world.is_sleeping(other));
            assert_eq!(world.scene.body_pose(a).unwrap(), pose);
        }
    }
}

#[test]
fn kind_wake_policy_covers_rerooted_trees_and_loop_dependents() {
    for soft in [false, true] {
        for wake in [false, true] {
            let (mut world, a, b) = two();
            world.scene.set_gravity(Vec3::ZERO).unwrap();
            let other = body(&mut world, 10.0, BodyKind::Dynamic);
            let joint = JointDesc {
                parent: Some(a),
                child: b,
                kind: JointType::Fixed,
                frame_parent: Transform::from_translation(Vec3::X * 3.0),
                frame_child: Transform::IDENTITY,
                axis: Vec3::Z,
            };
            if soft {
                world
                    .scene
                    .insert_loop_joint(
                        joint,
                        JointSoftness {
                            natural_frequency: 20.0,
                            damping_ratio: 1.0,
                        },
                    )
                    .unwrap();
            } else {
                world.scene.insert_joint(joint).unwrap();
            }
            world.sleep_bodies(&[a, other]).unwrap();
            let poses = [
                world.scene.body_pose(a).unwrap(),
                world.scene.body_pose(b).unwrap(),
            ];
            world
                .scene
                .set_body_kind_with_wake(b, BodyKind::Fixed, true)
                .unwrap();
            assert!(world.is_sleeping(a) && world.is_sleeping(other));
            world.step(0.001).unwrap();
            assert!(world.is_sleeping(a) && world.is_sleeping(other));
            world
                .scene
                .set_body_kind_with_wake(b, BodyKind::Dynamic, wake)
                .unwrap();
            assert_eq!(world.is_sleeping(a), !wake);
            assert_eq!(world.is_sleeping(b), !wake);
            assert!(world.is_sleeping(other));
            world.step(0.001).unwrap();
            assert_eq!(world.is_sleeping(a), !wake);
            assert_eq!(world.is_sleeping(b), !wake);
            assert!(world.is_sleeping(other));
            assert_eq!(
                [
                    world.scene.body_pose(a).unwrap(),
                    world.scene.body_pose(b).unwrap()
                ],
                poses
            );
        }
    }
}

#[test]
fn changing_watched_boundary_to_moving_kinematic_wakes_only_dependents() {
    let (mut world, a, other) = two();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let anchor = body(&mut world, -3.0, BodyKind::Fixed);
    world
        .scene
        .insert_joint(JointDesc {
            parent: Some(anchor),
            child: a,
            kind: JointType::Revolute,
            frame_parent: Transform::from_translation(Vec3::X * 3.0),
            frame_child: Transform::IDENTITY,
            axis: Vec3::Z,
        })
        .unwrap();
    world.sleep_bodies(&[a, other]).unwrap();
    world
        .scene
        .set_body_kind_with_wake(anchor, BodyKind::Kinematic, false)
        .unwrap();
    assert!(world.is_sleeping(a) && world.is_sleeping(other));
    world
        .scene
        .set_body_velocity_with_wake(anchor, SpatialVector::new(Vec3::X, Vec3::ZERO), false)
        .unwrap();
    world.step(0.001).unwrap();
    assert!(!world.is_sleeping(a));
    assert!(world.is_sleeping(other));
    assert!(world.scene.body_velocity(a).unwrap().linear.x > 0.0);
}

#[test]
fn newly_dynamic_boundary_joins_awake_dependencies_even_without_explicit_wake() {
    let (mut world, a, other) = two();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let anchor = body(&mut world, -3.0, BodyKind::Fixed);
    world
        .scene
        .insert_joint(JointDesc {
            parent: Some(anchor),
            child: a,
            kind: JointType::Revolute,
            frame_parent: Transform::from_translation(Vec3::X * 3.0),
            frame_child: Transform::IDENTITY,
            axis: Vec3::Z,
        })
        .unwrap();
    world.sleep_bodies(&[a, other]).unwrap();
    world
        .scene
        .set_body_kind_with_wake(anchor, BodyKind::Dynamic, false)
        .unwrap();
    assert!(!world.is_sleeping(anchor));
    world.step(0.001).unwrap();
    assert!(!world.is_sleeping(a));
    assert!(world.is_sleeping(other));
}

#[test]
fn nonwaking_kind_edit_does_not_erase_an_earlier_wake_or_reset_other_quiet_windows() {
    let (mut world, a, other) = two();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world
        .set_sleep_settings(Some(molla_solvers::rigid_world::SleepSettings {
            quiet_time: 0.025,
            ..Default::default()
        }))
        .unwrap();
    world.step(0.01).unwrap();
    world.step(0.01).unwrap();
    world
        .scene
        .set_body_kind_with_wake(a, BodyKind::Kinematic, true)
        .unwrap();
    world.step(0.01).unwrap();
    assert!(world.is_sleeping(other));
    world
        .scene
        .set_body_kind_with_wake(a, BodyKind::Dynamic, false)
        .unwrap();
    world.sleep_body(a).unwrap();
    world
        .scene
        .add_body_wrench(a, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    assert!(!world.is_sleeping(a));
    world
        .scene
        .set_body_kind_with_wake(a, BodyKind::Kinematic, false)
        .unwrap();
    world
        .scene
        .set_body_kind_with_wake(a, BodyKind::Dynamic, false)
        .unwrap();
    assert!(!world.is_sleeping(a));
    world.step(0.001).unwrap();
    assert!(!world.is_sleeping(a));
    assert!(world.is_sleeping(other));
}
