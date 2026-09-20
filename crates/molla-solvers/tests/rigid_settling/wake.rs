use super::*;

#[test]
fn weak_wake_preserves_awake_quiet_time_but_strong_wake_restarts_it() {
    let (mut world, a, b) = configured();
    for _ in 0..2 {
        world.step(0.01).unwrap();
    }
    world.wake_body_with_strength(a, false).unwrap();
    world.wake_body_with_strength(b, true).unwrap();
    world.step(0.01).unwrap();
    assert!(world.is_sleeping(a));
    assert!(!world.is_sleeping(b));
    world.step(0.01).unwrap();
    assert!(!world.is_sleeping(b));
    world.step(0.01).unwrap();
    assert!(world.is_sleeping(b));
}

#[test]
fn weak_wake_of_explicit_or_automatic_sleeper_can_resleep_next_step() {
    for explicit in [false, true] {
        let (mut world, a, b) = configured();
        if explicit {
            world.sleep_bodies(&[a, b]).unwrap();
        } else {
            for _ in 0..3 {
                world.step(0.01).unwrap();
            }
        }
        assert!(world.is_sleeping(a) && world.is_sleeping(b));
        world.wake_body_with_strength(a, false).unwrap();
        assert!(!world.is_sleeping(a) && world.is_sleeping(b));
        world.step(0.01).unwrap();
        assert!(world.is_sleeping(a) && world.is_sleeping(b));
        world.wake_body_with_strength(a, true).unwrap();
        for _ in 0..2 {
            world.step(0.01).unwrap();
            assert!(!world.is_sleeping(a) && world.is_sleeping(b));
        }
        world.step(0.01).unwrap();
        assert!(world.is_sleeping(a) && world.is_sleeping(b));
    }
}

#[test]
fn weak_wake_propagates_through_trees_loops_and_boundaries_only() {
    for kind in 0..3 {
        let (mut world, a, b) = configured();
        let unrelated = body(&mut world, 10.0, BodyKind::Dynamic);
        if kind == 2 {
            world.scene.set_body_kind(a, BodyKind::Fixed).unwrap();
        }
        let desc = JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Revolute,
            frame_parent: Transform::from_translation(Vec3::X * 3.0),
            frame_child: Transform::IDENTITY,
            axis: Vec3::Z,
        };
        if kind == 1 {
            world
                .scene
                .insert_loop_joint(
                    desc,
                    JointSoftness {
                        natural_frequency: 20.0,
                        damping_ratio: 1.0,
                    },
                )
                .unwrap();
        } else {
            world.scene.insert_joint(desc).unwrap();
        }
        world.sleep_bodies(&[b, unrelated]).unwrap();
        world.wake_body_with_strength(a, false).unwrap();
        assert!(!world.is_sleeping(b));
        assert!(world.is_sleeping(unrelated));
        if kind != 2 {
            assert!(!world.is_sleeping(a));
        }
        world.step(0.01).unwrap();
        assert!(world.is_sleeping(b) && world.is_sleeping(unrelated));
        if kind != 2 {
            assert!(world.is_sleeping(a));
        }
    }
}

#[test]
fn weak_wake_does_not_restore_eligibility_after_input_edits() {
    for edit in 0..3 {
        let (mut world, a, b) = configured();
        world.sleep_bodies(&[a, b]).unwrap();
        match edit {
            0 => world.scene.set_body_damping(a, 0.1, 0.0).unwrap(),
            1 => world
                .scene
                .set_body_velocity(a, SpatialVector::new(Vec3::X, Vec3::ZERO))
                .unwrap(),
            _ => world
                .scene
                .add_body_wrench(a, SpatialVector::new(Vec3::X * 1e-9, Vec3::ZERO))
                .unwrap(),
        }
        world.wake_body_with_strength(a, false).unwrap();
        world.step(0.01).unwrap();
        assert!(!world.is_sleeping(a) && world.is_sleeping(b), "edit={edit}");
    }
}

#[test]
fn weak_wake_does_not_enable_automatic_sleep() {
    let (mut world, a, b) = configured();
    world.sleep_bodies(&[a, b]).unwrap();
    world.set_sleep_settings(None).unwrap();
    world.wake_body_with_strength(a, false).unwrap();
    for _ in 0..10 {
        world.step(0.01).unwrap();
        assert!(!world.is_sleeping(a) && world.is_sleeping(b));
    }
}

#[test]
fn retained_eligibility_still_requires_quiet_inputs_and_motion() {
    for mode in 0..3 {
        let (mut world, a, _) = configured();
        if mode == 0 {
            world.scene.set_gravity(-Vec3::Y * 9.81).unwrap();
        } else if mode == 1 {
            world
                .scene
                .add_body_wrench(a, SpatialVector::new(Vec3::X * 1e-9, Vec3::ZERO))
                .unwrap();
        }
        world.sleep_body(a).unwrap();
        world.wake_body_with_strength(a, false).unwrap();
        if mode == 2 {
            world.scene.set_body_damping(a, 0.1, 0.0).unwrap();
        }
        world.step(0.01).unwrap();
        assert!(!world.is_sleeping(a), "mode={mode}");
    }
}

#[test]
fn invalid_wakes_and_failed_steps_preserve_weak_wake_eligibility() {
    let (mut world, a, b) = configured();
    world.scene.remove_body(b).unwrap();
    world.sleep_body(a).unwrap();
    for strong in [false, true] {
        assert!(world.wake_body_with_strength(b, strong).is_err());
        assert!(world.is_sleeping(a));
    }
    world.wake_body_with_strength(a, false).unwrap();
    assert!(world.step(f64::NAN).is_err());
    assert!(
        world
            .step_with_forces(
                0.01,
                |_, _| false,
                |_, _, _, _| Err(molla_core::Error::Solver("test failure".into()))
            )
            .is_err()
    );
    assert!(!world.is_sleeping(a));
    world.step(0.01).unwrap();
    assert!(world.is_sleeping(a));
}
