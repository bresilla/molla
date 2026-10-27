use super::*;

#[test]
fn excluded_support_wakes_before_observing_a_teleported_tyres_degenerate_heading() {
    let (mut world, chassis, wheel, _, _) = fixture(false);
    world
        .wheels
        .configure_pressure_tire(wheel, pressure_config())
        .unwrap();
    world.step(0.001).unwrap();
    world.sleep_body(chassis).unwrap();
    let mut pose = world.scene.body_pose(chassis).unwrap();
    pose.rotation = molla_math::Quat::from_rotation_z(std::f64::consts::FRAC_PI_2);
    world
        .scene
        .set_body_pose_with_wake(chassis, pose, false)
        .unwrap();
    world.step_filtered(0.001, |a, _| a == Some(wheel)).unwrap();
    assert!(!world.is_sleeping(chassis));
    assert!(world.wheels.samples(&world.scene).is_empty());
}

#[test]
fn sleeping_teleport_refreshes_ground_friction_and_footprint_without_integration() {
    for pressure in [false, true] {
        let (mut world, chassis, wheel, _, ground) = fixture(false);
        if pressure {
            world
                .wheels
                .configure_pressure_tire(wheel, pressure_config())
                .unwrap();
        }
        world
            .wheels
            .register_ground(
                &world.scene,
                ground,
                Some(FrictionGrid {
                    origin: [-10.0, -5.0],
                    cell_size: [10.0, 10.0],
                    cols: 2,
                    rows: 1,
                    values: vec![0.2, 1.0],
                }),
            )
            .unwrap();
        world.step(0.001).unwrap();
        world.sleep_body(chassis).unwrap();
        let old = world.wheels.samples(&world.scene).last().unwrap().clone();
        assert_eq!(old.friction, 1.0);
        let mut pose = world.scene.body_pose(chassis).unwrap();
        pose.position.x = -2.0;
        pose.position.y = 0.48;
        world
            .scene
            .set_body_pose_with_wake(chassis, pose, false)
            .unwrap();
        assert!(world.is_sleeping(chassis) && world.is_sleeping(wheel));
        assert!(world.wheels.samples(&world.scene).is_empty());
        for _ in 0..3 {
            world.step(0.001).unwrap();
            assert!(world.is_sleeping(chassis) && world.is_sleeping(wheel));
            assert_eq!(world.scene.body_pose(chassis).unwrap(), pose);
            let sample = world.wheels.samples(&world.scene).last().unwrap();
            assert_eq!(sample.dt, 0.0);
            assert!((sample.ground_point.x + 2.0).abs() < 1e-10);
            assert_eq!(sample.friction, 0.2);
            assert_eq!(sample.output.slip_ratio, 0.0);
            assert_eq!(sample.output.slip_angle, 0.0);
            assert!(sample.output.fz > 0.0);
            if pressure {
                let state = sample.pressure.as_ref().unwrap();
                assert!((state.loaded_radius - 0.48).abs() < 1e-10);
                assert_eq!(
                    state.pressure_pa,
                    old.pressure.as_ref().unwrap().pressure_pa
                );
                assert!(state.patch_area > old.pressure.as_ref().unwrap().patch_area);
            }
        }
        world.wake_body(chassis).unwrap();
        world.step(0.001).unwrap();
        assert!(!world.is_sleeping(chassis));
        assert!(
            world
                .wheels
                .samples(&world.scene)
                .iter()
                .all(|s| s.dt > 0.0)
        );
    }
}

#[test]
fn sleeping_wheels_can_leave_and_return_to_terrain_without_stale_samples() {
    let (mut world, chassis, wheel, _, _) = fixture(false);
    world
        .wheels
        .configure_pressure_tire(wheel, pressure_config())
        .unwrap();
    world.step(0.001).unwrap();
    world.sleep_body(chassis).unwrap();
    let original = world.scene.body_pose(chassis).unwrap();
    let mut away = original;
    away.position.x = 100.0;
    world
        .scene
        .set_body_pose_with_wake(chassis, away, false)
        .unwrap();
    world.step(0.001).unwrap();
    assert!(world.is_sleeping(chassis));
    assert!(world.wheels.samples(&world.scene).is_empty());
    world
        .scene
        .set_body_pose_with_wake(chassis, original, false)
        .unwrap();
    let pressure = world
        .wheels
        .pressure_tire(&world.scene, wheel)
        .unwrap()
        .pressure_pa;
    assert!(
        world
            .step_with_forces(
                0.001,
                |_, _| false,
                |_, _, _, _| Err(molla_core::Error::Solver("injected".into()))
            )
            .is_err()
    );
    assert!(world.is_sleeping(chassis));
    assert_eq!(
        world
            .wheels
            .pressure_tire(&world.scene, wheel)
            .unwrap()
            .pressure_pa,
        pressure
    );
    world.step(0.001).unwrap();
    assert!(world.is_sleeping(chassis));
    assert!(
        world
            .wheels
            .samples(&world.scene)
            .last()
            .unwrap()
            .output
            .in_contact
    );
    assert_eq!(world.scene.body_pose(chassis).unwrap(), original);
}

#[test]
fn moving_static_terrain_without_wake_refreshes_frozen_tyre_support() {
    let (mut world, chassis, wheel, _, ground) = fixture(false);
    world
        .wheels
        .configure_pressure_tire(wheel, pressure_config())
        .unwrap();
    let floor = world
        .scene
        .insert_body(BodyDesc {
            kind: molla_sim::runtime::BodyKind::Fixed,
            ..Default::default()
        })
        .unwrap();
    let mut collider = world.scene.collider(ground).unwrap().clone();
    collider.parent = Some(floor);
    world.scene.replace_collider(ground, collider).unwrap();
    world.step(0.001).unwrap();
    world.sleep_body(chassis).unwrap();
    let original = world.scene.body_pose(chassis).unwrap();
    for y in [-0.1, 0.005] {
        world
            .scene
            .set_body_pose_with_wake(floor, Transform::from_translation(Vec3::Y * y), false)
            .unwrap();
        world.step(0.001).unwrap();
        assert!(world.is_sleeping(chassis));
        assert_eq!(world.scene.body_pose(chassis).unwrap(), original);
        let sample = world.wheels.samples(&world.scene).last().unwrap();
        assert!((sample.ground_point.y - y).abs() < 1e-10);
        assert_eq!(sample.output.in_contact, y > 0.0);
        assert_eq!(sample.dt, 0.0);
    }
}
