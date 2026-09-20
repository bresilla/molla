use super::*;

#[test]
fn latent_velocity_and_wrench_are_retained_until_explicit_wake() {
    let (mut world, a, b) = two();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world.sleep_bodies(&[a, b]).unwrap();
    let pose = world.scene.body_pose(a).unwrap();
    let velocity = SpatialVector::new(Vec3::X * 2.0, Vec3::ZERO);
    let force = SpatialVector::new(Vec3::X * 3.0, Vec3::Z * 4.0);
    world
        .scene
        .set_body_velocity_with_wake(a, velocity, false)
        .unwrap();
    world
        .scene
        .add_body_wrench_with_wake(a, force, false)
        .unwrap();
    for _ in 0..20 {
        assert!(world.is_sleeping(a) && world.is_sleeping(b));
        world.step(0.01).unwrap();
        assert_eq!(world.scene.body_pose(a).unwrap(), pose);
        assert_eq!(world.scene.body_velocity(a).unwrap(), velocity);
        assert_eq!(world.scene.state().body_f.host().unwrap()[0], force);
    }
    world.wake_body(a).unwrap();
    world.step(0.01).unwrap();
    let moved = world.scene.body_velocity(a).unwrap();
    assert!((moved.linear.x - 2.03).abs() < 1e-10);
    assert!((moved.angular.z - 0.04).abs() < 1e-10);
    assert!(world.scene.body_pose(a).unwrap().position.x > pose.position.x);
    assert!(world.is_sleeping(b));
}

#[test]
fn reset_wrench_can_preserve_sleep_or_wake_the_dependency_group() {
    for wake in [false, true] {
        let (mut world, a, b) = two();
        let unrelated = body(&mut world, 10.0, BodyKind::Dynamic);
        world.scene.set_gravity(Vec3::ZERO).unwrap();
        world
            .scene
            .insert_joint(JointDesc {
                parent: Some(a),
                child: b,
                kind: JointType::Fixed,
                frame_parent: Transform::from_translation(Vec3::X * 3.0),
                frame_child: Transform::IDENTITY,
                axis: Vec3::Z,
            })
            .unwrap();
        world.sleep_bodies(&[a, unrelated]).unwrap();
        world
            .scene
            .add_body_wrench_with_wake(b, SpatialVector::new(Vec3::X, Vec3::Z), false)
            .unwrap();
        assert!(world.is_sleeping(a) && world.is_sleeping(b));
        world.scene.reset_body_forces_with_wake(b, wake).unwrap();
        assert_eq!(world.is_sleeping(a), !wake);
        assert_eq!(world.is_sleeping(b), !wake);
        world.step(0.01).unwrap();
        assert_eq!(world.is_sleeping(a), !wake);
        assert_eq!(world.is_sleeping(b), !wake);
        assert!(world.is_sleeping(unrelated));
        assert_eq!(world.scene.body_velocity(a).unwrap(), SpatialVector::ZERO);
        assert_eq!(world.scene.body_velocity(b).unwrap(), SpatialVector::ZERO);
    }
}

#[test]
fn sleeping_tree_preserves_projected_joint_rates_and_resumes_them() {
    let (mut world, a, b) = two();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world
        .scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Revolute,
            frame_parent: Transform::from_translation(Vec3::X * 3.0),
            frame_child: Transform::IDENTITY,
            axis: Vec3::Z,
        })
        .unwrap();
    world.sleep_body(a).unwrap();
    world
        .scene
        .set_body_velocity_with_wake(b, SpatialVector::new(Vec3::ZERO, Vec3::Z), false)
        .unwrap();
    let poses = world.scene.state().body_q.host().unwrap().to_vec();
    let rates = world.scene.state().joint_qd.host().unwrap().to_vec();
    assert!(rates.iter().any(|&rate| rate != 0.0));
    for _ in 0..5 {
        world.step(0.01).unwrap();
        assert!(world.is_sleeping(a) && world.is_sleeping(b));
        assert_eq!(world.scene.state().body_q.host().unwrap(), poses);
        assert_eq!(world.scene.state().joint_qd.host().unwrap(), rates);
    }
    world.wake_body(b).unwrap();
    world.step(0.01).unwrap();
    assert!(!world.is_sleeping(a) && !world.is_sleeping(b));
    assert_ne!(world.scene.state().body_q.host().unwrap(), poses);
}

#[test]
fn nonwaking_force_edits_do_not_disable_future_contact_wakes() {
    let (mut world, a, b) = two();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world
        .scene
        .set_body_pose(b, Transform::from_translation(Vec3::new(1.4, 2.0, 0.0)))
        .unwrap();
    sphere(&mut world, a, false);
    sphere(&mut world, b, false);
    world
        .scene
        .set_body_velocity(b, SpatialVector::new(-Vec3::X * 20.0, Vec3::ZERO))
        .unwrap();
    world.sleep_body(a).unwrap();
    world
        .scene
        .add_body_wrench_with_wake(a, SpatialVector::new(Vec3::X, Vec3::ZERO), false)
        .unwrap();
    assert!(world.is_sleeping(a));
    world.step(0.04).unwrap();
    assert!(!world.is_sleeping(a));
    assert!(world.scene.body_velocity(a).unwrap().linear.x < -0.1);
}

#[test]
fn nonwaking_edits_never_erase_an_existing_wake() {
    let (mut world, a, b) = two();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world.sleep_bodies(&[a, b]).unwrap();
    world
        .scene
        .add_body_wrench(a, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    world.scene.reset_body_forces_with_wake(a, false).unwrap();
    world
        .scene
        .set_body_velocity_with_wake(a, SpatialVector::new(Vec3::X, Vec3::ZERO), false)
        .unwrap();
    assert!(!world.is_sleeping(a) && world.is_sleeping(b));
    world.step(0.01).unwrap();
    assert!(world.scene.body_pose(a).unwrap().position.x > 0.0);
    assert!(world.is_sleeping(b));
}

#[test]
fn nonwaking_velocity_edit_preserves_an_awake_quiet_window() {
    for wake in [false, true] {
        let (mut world, a, b) = two();
        world.scene.set_gravity(Vec3::ZERO).unwrap();
        world
            .set_sleep_settings(Some(molla_solvers::rigid_world::SleepSettings {
                quiet_time: 0.025,
                ..Default::default()
            }))
            .unwrap();
        for _ in 0..2 {
            world.step(0.01).unwrap();
        }
        world
            .scene
            .set_body_velocity_with_wake(a, SpatialVector::new(Vec3::X * 1e-8, Vec3::ZERO), wake)
            .unwrap();
        world.step(0.01).unwrap();
        assert_eq!(world.is_sleeping(a), !wake);
        assert!(world.is_sleeping(b));
    }
}
