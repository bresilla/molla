use super::*;
use molla_math::Quat;

#[test]
fn nonwaking_teleport_updates_subtree_and_queries_without_moving_sleepers() {
    for wake in [false, true] {
        let (mut world, a, b) = two();
        world.scene.set_gravity(Vec3::ZERO).unwrap();
        let other = body(&mut world, 20.0, BodyKind::Dynamic);
        sphere(&mut world, b, false);
        world
            .scene
            .insert_joint(JointDesc {
                parent: Some(a),
                child: b,
                kind: JointType::Fixed,
                frame_parent: Transform::from_translation(Vec3::X * 3.0),
                frame_child: Transform::IDENTITY,
                axis: Vec3::X,
            })
            .unwrap();
        world.sleep_bodies(&[a, other]).unwrap();
        let pose = Transform::new(Vec3::new(4.0, 2.0, 0.0), Quat::from_rotation_y(0.5));
        world.scene.set_body_pose_with_wake(a, pose, wake).unwrap();
        assert_eq!(world.is_sleeping(a), !wake);
        assert_eq!(world.is_sleeping(b), !wake);
        assert!(world.is_sleeping(other));
        let child = pose.transform_point(Vec3::X * 3.0);
        assert!((world.scene.body_pose(b).unwrap().position - child).length() < 1e-10);
        assert!(
            world
                .cast_ray(child + Vec3::Y * 2.0, -Vec3::Y, 4.0)
                .unwrap()
                .is_some()
        );
        assert!(
            world
                .cast_ray(Vec3::new(3.0, 4.0, 0.0), -Vec3::Y, 4.0)
                .unwrap()
                .is_none()
        );
        for _ in 0..5 {
            world.step(0.01).unwrap();
            assert_eq!(world.is_sleeping(a), !wake);
            assert_eq!(world.is_sleeping(b), !wake);
            assert!(world.is_sleeping(other));
            assert!((world.scene.body_pose(a).unwrap().position - pose.position).length() < 1e-10);
        }
    }
}

#[test]
fn invalid_and_unchanged_poses_do_not_activate_or_invalidate_state() {
    let (mut world, a, b) = two();
    world.sleep_bodies(&[a, b]).unwrap();
    let pose = world.scene.body_pose(a).unwrap();
    let stamp = world.scene.stamp();
    let revision = world.scene.revision();
    for wake in [false, true] {
        for bad in [
            Transform::from_translation(Vec3::splat(f64::NAN)),
            Transform::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)),
        ] {
            assert!(world.scene.set_body_pose_with_wake(a, bad, wake).is_err());
        }
        world.scene.set_body_pose_with_wake(a, pose, wake).unwrap();
        assert_eq!(world.scene.stamp(), stamp);
        assert_eq!(world.scene.revision(), revision);
        assert_eq!(world.scene.body_pose(a).unwrap(), pose);
        assert!(world.is_sleeping(a) && world.is_sleeping(b));
    }
}

#[test]
fn a_nonwaking_teleport_does_not_acknowledge_an_earlier_force_edit() {
    let (mut world, a, b) = two();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world.sleep_bodies(&[a, b]).unwrap();
    world
        .scene
        .add_body_wrench(a, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    world
        .scene
        .set_body_pose_with_wake(a, Transform::from_translation(Vec3::X * 5.0), false)
        .unwrap();
    assert!(!world.is_sleeping(a) && world.is_sleeping(b));
    world.step(0.01).unwrap();
    assert!(world.scene.body_pose(a).unwrap().position.x > 5.0);
    assert!(world.is_sleeping(b));
}
