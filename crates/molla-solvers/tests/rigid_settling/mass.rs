use super::*;

fn replacement() -> MassProperties {
    MassProperties {
        mass: 2.0,
        local_com: Vec3::ZERO,
        inertia: Mat3::IDENTITY * 3.0,
    }
}

#[test]
fn additional_mass_wake_policy_covers_trees_and_loops_without_waking_other_groups() {
    for wake in [false, true] {
        for soft in [false, true] {
            let (mut world, a, b) = two();
            world.scene.set_gravity(Vec3::ZERO).unwrap();
            let other = body(&mut world, 10.0, BodyKind::Dynamic);
            let desc = JointDesc {
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
            world.sleep_bodies(&[a, other]).unwrap();
            let poses = world.scene.state().body_q.host().unwrap().to_vec();
            world
                .set_body_additional_mass_with_wake(b, replacement(), wake)
                .unwrap();
            assert_eq!(world.is_sleeping(a), !wake);
            assert_eq!(world.is_sleeping(b), !wake);
            assert!(world.is_sleeping(other));
            world.step(0.001).unwrap();
            assert_eq!(world.is_sleeping(a), !wake);
            assert_eq!(world.is_sleeping(b), !wake);
            assert!(world.is_sleeping(other));
            assert_eq!(world.scene.state().body_q.host().unwrap(), poses);
            assert_eq!(world.scene.body_mass_properties(b).unwrap(), replacement());
        }
    }
}

#[test]
fn nonwaking_mass_is_used_by_the_next_awake_impulse_and_recomputes_without_spurious_wake() {
    let (mut world, a, b) = two();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    world.sleep_bodies(&[a, b]).unwrap();
    world
        .set_body_additional_mass_with_wake(a, replacement(), false)
        .unwrap();
    world.recompute_body_mass(a).unwrap();
    assert!(world.is_sleeping(a) && world.is_sleeping(b));
    world
        .apply_body_impulse(a, SpatialVector::new(Vec3::X * 4.0, Vec3::Z * 6.0))
        .unwrap();
    assert!(!world.is_sleeping(a) && world.is_sleeping(b));
    let velocity = world.scene.body_velocity(a).unwrap();
    assert!((velocity.linear.x - 2.0).abs() < 1e-10);
    assert!((velocity.angular.z - 2.0).abs() < 1e-10);
}

#[test]
fn mass_edit_preserves_prior_wakes_and_unrelated_quiet_windows() {
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
    world.set_body_additional_mass(a, replacement()).unwrap();
    world.step(0.01).unwrap();
    assert!(!world.is_sleeping(a) && world.is_sleeping(b));
    world.sleep_body(a).unwrap();
    world
        .scene
        .add_body_wrench(b, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    world
        .set_body_additional_mass_with_wake(
            a,
            MassProperties {
                mass: 4.0,
                ..replacement()
            },
            false,
        )
        .unwrap();
    assert!(world.is_sleeping(a) && !world.is_sleeping(b));
    world.step(0.01).unwrap();
    assert!(world.is_sleeping(a) && !world.is_sleeping(b));
}
