use super::*;
use molla_solvers::rigid_world::SleepSettings;

fn configured() -> (RigidWorld, BodyHandle, BodyHandle) {
    let (mut world, a, b) = two();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world
        .set_sleep_settings(Some(SleepSettings {
            quiet_time: 0.025,
            ..Default::default()
        }))
        .unwrap();
    (world, a, b)
}

#[test]
fn automatic_sleep_is_opt_in_and_disabling_preserves_sleepers() {
    let (mut world, a, _) = two();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    for _ in 0..200 {
        world.step(0.01).unwrap();
    }
    assert!(!world.is_sleeping(a));
    world
        .set_sleep_settings(Some(SleepSettings {
            quiet_time: 0.025,
            ..Default::default()
        }))
        .unwrap();
    for _ in 0..2 {
        world.step(0.01).unwrap();
        assert!(!world.is_sleeping(a));
    }
    world.step(0.01).unwrap();
    assert!(world.is_sleeping(a));
    world.set_sleep_settings(None).unwrap();
    world.step(0.01).unwrap();
    assert!(world.is_sleeping(a));
    world.wake_body(a).unwrap();
    for _ in 0..5 {
        world.step(0.01).unwrap();
    }
    assert!(!world.is_sleeping(a));
}

#[test]
fn failed_steps_and_settings_do_not_consume_or_reset_quiet_time() {
    let (mut world, a, _) = configured();
    world.step(0.01).unwrap();
    let settings = world.sleep_settings().unwrap();
    for bad in [
        SleepSettings {
            linear_speed: -1.0,
            ..settings
        },
        SleepSettings {
            angular_speed: f64::NAN,
            ..settings
        },
        SleepSettings {
            quiet_time: 0.0,
            ..settings
        },
    ] {
        assert!(world.set_sleep_settings(Some(bad)).is_err());
        assert_eq!(world.sleep_settings(), Some(settings));
    }
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
    world.step(0.01).unwrap();
    assert!(!world.is_sleeping(a));
    world.step(0.01).unwrap();
    assert!(world.is_sleeping(a));
}

#[test]
fn input_edits_and_explicit_wakes_restart_only_affected_quiet_windows() {
    for explicit in [false, true] {
        let (mut world, a, b) = configured();
        for _ in 0..2 {
            world.step(0.01).unwrap();
        }
        if explicit {
            world.wake_body(a).unwrap();
        } else {
            world.scene.set_body_damping(a, 0.1, 0.0).unwrap();
        }
        world.step(0.01).unwrap();
        assert!(!world.is_sleeping(a));
        assert!(world.is_sleeping(b));
        world.step(0.01).unwrap();
        assert!(!world.is_sleeping(a));
        world.step(0.01).unwrap();
        assert!(world.is_sleeping(a));
    }
}

#[test]
fn arbitrary_force_steps_clear_quiet_history() {
    let (mut world, a, _) = configured();
    for _ in 0..2 {
        world.step(0.01).unwrap();
    }
    world
        .step_with_forces(0.01, |_, _| false, |_, _, _, _| Ok(()))
        .unwrap();
    for _ in 0..2 {
        world.step(0.01).unwrap();
        assert!(!world.is_sleeping(a));
    }
    world.step(0.01).unwrap();
    assert!(world.is_sleeping(a));
}

#[test]
fn freefall_and_moving_groups_do_not_sleep() {
    let (mut world, a, b) = configured();
    world
        .scene
        .set_body_velocity(b, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    for _ in 0..10 {
        world.step(0.01).unwrap();
    }
    assert!(world.is_sleeping(a));
    assert!(!world.is_sleeping(b));
    world.scene.set_gravity(-Vec3::Y * 9.81).unwrap();
    for _ in 0..10 {
        world.step(0.01).unwrap();
    }
    assert!(!world.is_sleeping(a) && !world.is_sleeping(b));
}

#[test]
fn persistent_wrenches_and_joint_feedforward_keep_quiet_bodies_awake() {
    for force_source in 0..3 {
        let (mut world, a, b) = configured();
        let index = world.scene.body_index(a).unwrap().index();
        match force_source {
            0 => world
                .scene
                .add_body_wrench(a, SpatialVector::new(Vec3::X * 1e-9, Vec3::ZERO))
                .unwrap(),
            1 => {
                world
                    .scene
                    .control_mut()
                    .body_external_f
                    .host_mut()
                    .unwrap()[index] = SpatialVector::new(Vec3::X * 1e-9, Vec3::ZERO)
            }
            _ => world.scene.control_mut().joint_force.host_mut().unwrap()[0] = 1e-9,
        }
        for _ in 0..10 {
            world.step(0.01).unwrap();
        }
        assert!(!world.is_sleeping(a), "source {force_source}");
        assert!(world.is_sleeping(b));
    }
}

#[test]
fn tiny_motor_commands_keep_tree_and_loop_groups_awake() {
    for loop_joint in [false, true] {
        let (mut world, a, b) = configured();
        let desc = JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Revolute,
            frame_parent: Transform::from_translation(Vec3::X * 3.0),
            frame_child: Transform::IDENTITY,
            axis: Vec3::Z,
        };
        let joint = if loop_joint {
            world
                .scene
                .insert_loop_joint(
                    desc,
                    JointSoftness {
                        natural_frequency: 20.0,
                        damping_ratio: 1.0,
                    },
                )
                .unwrap()
        } else {
            world.scene.insert_joint(desc).unwrap()
        };
        world
            .scene
            .set_joint_motor(
                joint,
                JointAxis::AngX,
                JointMotor {
                    mode: JointTargetMode::Velocity,
                    target_velocity: 1e-8,
                    damping: 2.0,
                    ..Default::default()
                },
            )
            .unwrap();
        for _ in 0..10 {
            world.step(0.01).unwrap();
        }
        assert!(!world.is_sleeping(a) && !world.is_sleeping(b));
        world
            .scene
            .set_joint_motor(joint, JointAxis::AngX, JointMotor::default())
            .unwrap();
        for _ in 0..10 {
            world.step(0.01).unwrap();
        }
        assert!(world.is_sleeping(a) && world.is_sleeping(b));
    }
}

#[test]
fn moving_boundary_blocks_automatic_sleep() {
    let (mut world, a, b) = configured();
    let anchor = body(&mut world, 0.0, BodyKind::Kinematic);
    world
        .scene
        .insert_joint(JointDesc {
            parent: Some(anchor),
            child: a,
            kind: JointType::Revolute,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
            axis: Vec3::Z,
        })
        .unwrap();
    world
        .scene
        .set_body_velocity(anchor, SpatialVector::new(Vec3::X * 1e-9, Vec3::ZERO))
        .unwrap();
    for _ in 0..10 {
        world.step(0.01).unwrap();
    }
    assert!(!world.is_sleeping(a));
    assert!(world.is_sleeping(b));
    world
        .scene
        .set_body_velocity(anchor, SpatialVector::ZERO)
        .unwrap();
    for _ in 0..3 {
        world.step(0.01).unwrap();
    }
    assert!(world.is_sleeping(a));
}

#[test]
fn topology_change_restarts_quiet_time_for_merged_groups() {
    let (mut world, a, b) = configured();
    for _ in 0..2 {
        world.step(0.01).unwrap();
    }
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
    for _ in 0..2 {
        world.step(0.01).unwrap();
        assert!(!world.is_sleeping(a) && !world.is_sleeping(b));
    }
    world.step(0.01).unwrap();
    assert!(world.is_sleeping(a) && world.is_sleeping(b));
}

#[test]
fn batch_sleep_rejects_unknown_handles_without_partial_sleep() {
    let (mut world, a, b) = two();
    world.scene.remove_body(b).unwrap();
    assert!(world.sleep_bodies(&[a, b]).is_err());
    assert!(!world.is_sleeping(a));
}
