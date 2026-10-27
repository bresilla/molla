use super::*;

fn desc() -> BodyDesc {
    BodyDesc {
        additional_mass: Some(MassProperties {
            mass: 2.0,
            local_com: Vec3::ZERO,
            inertia: Mat3::IDENTITY * 3.0,
        }),
        ..Default::default()
    }
}

#[test]
fn impulse_wake_policy_preserves_unrelated_sleep_and_retains_latent_response() {
    for wake in [false, true] {
        for mode in 0..3 {
            let mut world = world();
            let a = world.scene.insert_body(desc()).unwrap();
            let b = world.scene.insert_body(desc()).unwrap();
            world.sleep_bodies(&[a, b]).unwrap();
            let pose = world.scene.body_pose(a).unwrap();
            let expected = match mode {
                0 => {
                    world
                        .apply_body_impulse_with_wake(
                            a,
                            SpatialVector::new(Vec3::X * 4.0, Vec3::ZERO),
                            wake,
                        )
                        .unwrap();
                    SpatialVector::new(Vec3::X * 2.0, Vec3::ZERO)
                }
                1 => {
                    world
                        .apply_body_impulse_with_wake(
                            a,
                            SpatialVector::new(Vec3::ZERO, Vec3::Z * 6.0),
                            wake,
                        )
                        .unwrap();
                    SpatialVector::new(Vec3::ZERO, Vec3::Z * 2.0)
                }
                _ => {
                    let point = world.scene.body_world_com(a).unwrap() + Vec3::Y;
                    world
                        .apply_body_impulse_at_point_with_wake(a, Vec3::X * 4.0, point, wake)
                        .unwrap();
                    SpatialVector::new(Vec3::X * 2.0, -Vec3::Z * (4.0 / 3.0))
                }
            };
            assert_eq!(world.is_sleeping(a), !wake);
            assert!(world.is_sleeping(b));
            near(
                world.scene.body_velocity(a).unwrap().linear,
                expected.linear,
            );
            near(
                world.scene.body_velocity(a).unwrap().angular,
                expected.angular,
            );
            if !wake {
                for _ in 0..5 {
                    world.step(0.01).unwrap();
                    assert!(world.is_sleeping(a) && world.is_sleeping(b));
                    assert_eq!(world.scene.body_pose(a).unwrap(), pose);
                    near(
                        world.scene.body_velocity(a).unwrap().linear,
                        expected.linear,
                    );
                    near(
                        world.scene.body_velocity(a).unwrap().angular,
                        expected.angular,
                    );
                }
                world.wake_body(a).unwrap();
            }
            world.step(0.01).unwrap();
            assert_ne!(world.scene.body_pose(a).unwrap(), pose);
            assert!(world.is_sleeping(b));
        }
    }
}

#[test]
fn child_impulse_wakes_the_whole_tree_only_when_requested() {
    for wake in [false, true] {
        let mut world = world();
        let a = world.scene.insert_body(desc()).unwrap();
        let mut child = desc();
        child.params.pose.position = Vec3::X;
        let b = world.scene.insert_body(child).unwrap();
        let other = world.scene.insert_body(desc()).unwrap();
        world
            .scene
            .insert_joint(JointDesc {
                parent: Some(a),
                child: b,
                kind: JointType::Fixed,
                frame_parent: Transform::from_translation(Vec3::X),
                frame_child: Transform::IDENTITY,
                axis: Vec3::X,
            })
            .unwrap();
        world.sleep_bodies(&[a, other]).unwrap();
        world
            .apply_body_impulse_with_wake(b, SpatialVector::new(Vec3::X * 8.0, Vec3::ZERO), wake)
            .unwrap();
        for body in [a, b] {
            assert_eq!(world.is_sleeping(body), !wake);
            near(
                world.scene.body_velocity(body).unwrap().linear,
                Vec3::X * 2.0,
            );
        }
        world.step(0.01).unwrap();
        assert!(world.is_sleeping(other));
        assert_eq!(world.is_sleeping(a), !wake);
        assert_eq!(world.is_sleeping(b), !wake);
    }
}

#[test]
fn zero_and_invalid_impulses_preserve_sleep_state_and_failed_overflow_is_atomic() {
    let mut world = world();
    let mut tiny = desc();
    tiny.additional_mass.as_mut().unwrap().mass = 1e-8;
    let a = world.scene.insert_body(tiny).unwrap();
    let removed = world.scene.insert_body(desc()).unwrap();
    world.scene.remove_body(removed).unwrap();
    world.sleep_body(a).unwrap();
    let stamp = world.scene.activation_stamp(a);
    for wake in [false, true] {
        world
            .apply_body_impulse_with_wake(a, SpatialVector::ZERO, wake)
            .unwrap();
        assert!(
            world
                .apply_body_impulse_with_wake(removed, SpatialVector::ZERO, wake)
                .is_err()
        );
        for bad in [f64::NAN, f64::INFINITY, 1e308] {
            assert!(
                world
                    .apply_body_impulse_with_wake(
                        a,
                        SpatialVector::new(Vec3::X * bad, Vec3::ZERO),
                        wake
                    )
                    .is_err()
            );
        }
        assert!(
            world
                .apply_body_impulse_at_point_with_wake(a, Vec3::X, Vec3::splat(f64::NAN), wake)
                .is_err()
        );
        assert!(world.is_sleeping(a));
        assert_eq!(world.scene.body_velocity(a).unwrap(), SpatialVector::ZERO);
        assert_eq!(world.scene.activation_stamp(a), stamp);
    }
}

#[test]
fn a_nonzero_fully_constrained_impulse_still_honors_wake() {
    for wake in [false, true] {
        let mut world = world();
        let a = world.scene.insert_body(desc()).unwrap();
        let other = world.scene.insert_body(desc()).unwrap();
        world
            .scene
            .insert_joint(JointDesc {
                parent: None,
                child: a,
                kind: JointType::Fixed,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
                axis: Vec3::X,
            })
            .unwrap();
        world.sleep_bodies(&[a, other]).unwrap();
        world
            .apply_body_impulse_with_wake(a, SpatialVector::new(Vec3::X, Vec3::ZERO), wake)
            .unwrap();
        assert_eq!(world.is_sleeping(a), !wake);
        assert!(world.is_sleeping(other));
        assert_eq!(world.scene.body_velocity(a).unwrap(), SpatialVector::ZERO);
    }
}

#[test]
fn impulses_preserve_unrelated_quiet_time_and_do_not_acknowledge_prior_edits() {
    let mut world = world();
    let a = world.scene.insert_body(desc()).unwrap();
    let b = world.scene.insert_body(desc()).unwrap();
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
        .apply_body_impulse(a, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    world.step(0.01).unwrap();
    assert!(!world.is_sleeping(a) && world.is_sleeping(b));
    world.sleep_body(a).unwrap();
    world
        .scene
        .add_body_wrench(b, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    world
        .apply_body_impulse_with_wake(a, SpatialVector::new(Vec3::X, Vec3::ZERO), false)
        .unwrap();
    assert!(world.is_sleeping(a) && !world.is_sleeping(b));
    world.step(0.01).unwrap();
    assert!(world.is_sleeping(a) && !world.is_sleeping(b));
    assert!(world.scene.body_velocity(b).unwrap().linear.x > 0.0);
}
