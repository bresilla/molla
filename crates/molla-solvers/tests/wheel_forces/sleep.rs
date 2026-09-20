use super::*;

#[test]
fn replacing_wheel_force_storage_invalidates_sleep_even_with_matching_revisions() {
    let (mut world, chassis, _, _, _) = fixture(true);
    let (replacement, _, _, _, _) = fixture(true);
    world.sleep_body(chassis).unwrap();
    assert!(world.is_sleeping(chassis));
    world.wheels = replacement.wheels;
    assert!(!world.is_sleeping(chassis));
}

#[test]
fn excluding_a_sleeping_tyres_support_wakes_its_group() {
    let (mut world, chassis, wheel, _, _) = shear_fixture();
    world.step(0.001).unwrap();
    world.sleep_body(chassis).unwrap();
    world.step_filtered(0.001, |a, _| a == Some(wheel)).unwrap();
    assert!(!world.is_sleeping(chassis));
    assert!(world.wheels.samples(&world.scene).is_empty());
}

#[test]
fn sleeping_tyres_retain_support_without_integrating_pressure_or_shear() {
    let (mut world, chassis, wheel, _, _) = shear_fixture();
    world.step(0.001).unwrap();
    world.sleep_body(chassis).unwrap();
    let pose = world.scene.body_pose(chassis).unwrap();
    let pressure = *world.wheels.pressure_tire(&world.scene, wheel).unwrap();
    let force = world
        .wheels
        .samples(&world.scene)
        .last()
        .unwrap()
        .output
        .force;
    for _ in 0..100 {
        world.step(0.001).unwrap();
    }
    assert!(world.is_sleeping(chassis) && world.is_sleeping(wheel));
    assert_eq!(world.scene.body_pose(chassis).unwrap(), pose);
    assert_eq!(
        world.scene.body_velocity(chassis).unwrap(),
        SpatialVector::ZERO
    );
    assert_eq!(
        world
            .wheels
            .pressure_tire(&world.scene, wheel)
            .unwrap()
            .pressure_pa,
        pressure.pressure_pa
    );
    assert!(
        world
            .wheels
            .samples(&world.scene)
            .iter()
            .all(|sample| sample.dt == 0.0 && sample.output.force == force)
    );
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
    world.step(0.001).unwrap();
    assert!(!world.wheels.samples(&world.scene).is_empty());
    assert_eq!(world.wheels.samples(&world.scene)[0].output.force, force);
}

#[test]
fn pressure_target_and_surface_changes_wake_sleeping_wheel_groups() {
    let (mut world, chassis, wheel, _, ground) = shear_fixture();
    world.step(0.001).unwrap();
    world.sleep_body(wheel).unwrap();
    let current = world
        .wheels
        .pressure_tire(&world.scene, wheel)
        .unwrap()
        .target_pressure_pa;
    world
        .wheels
        .set_pressures(&world.scene, &[(wheel, current)])
        .unwrap();
    assert!(world.is_sleeping(chassis));
    assert!(
        world
            .wheels
            .set_pressures(&world.scene, &[(wheel, f64::NAN)])
            .is_err()
    );
    assert!(world.is_sleeping(chassis));
    world
        .wheels
        .set_pressures(&world.scene, &[(wheel, current + 1000.0)])
        .unwrap();
    assert!(!world.is_sleeping(chassis) && !world.is_sleeping(wheel));
    world.sleep_body(chassis).unwrap();
    assert!(!world.is_sleeping(chassis));
    world.step(0.1).unwrap();
    assert!(
        world
            .wheels
            .pressure_tire(&world.scene, wheel)
            .unwrap()
            .pressure_pa
            > current
    );
    world
        .wheels
        .set_pressures(&world.scene, &[(wheel, current)])
        .unwrap();
    world.step(0.1).unwrap();
    world.sleep_body(chassis).unwrap();
    assert!(world.is_sleeping(chassis));
    world
        .wheels
        .register_ground(
            &world.scene,
            ground,
            Some(FrictionGrid {
                origin: [-10.0, -5.0],
                cell_size: [20.0, 10.0],
                cols: 1,
                rows: 1,
                values: vec![0.1],
            }),
        )
        .unwrap();
    assert!(!world.is_sleeping(chassis));
}

#[test]
fn tyre_shear_resumes_identically_after_a_sleep_interval() {
    let (mut a, chassis_a, wheel_a, _, _) = shear_fixture();
    let (mut b, chassis_b, wheel_b, _, _) = shear_fixture();
    a.step(0.001).unwrap();
    b.step(0.001).unwrap();
    a.sleep_body(chassis_a).unwrap();
    b.sleep_body(chassis_b).unwrap();
    for _ in 0..10 {
        a.step(0.001).unwrap();
    }
    a.wake_body(chassis_a).unwrap();
    b.wake_body(chassis_b).unwrap();
    a.step(0.001).unwrap();
    b.step(0.001).unwrap();
    assert_eq!(
        a.scene.state().joint_q.host().unwrap(),
        b.scene.state().joint_q.host().unwrap()
    );
    assert_eq!(
        a.scene.state().joint_qd.host().unwrap(),
        b.scene.state().joint_qd.host().unwrap()
    );
    assert_eq!(
        a.wheels.samples(&a.scene).last().unwrap().output.force,
        b.wheels.samples(&b.scene).last().unwrap().output.force
    );
    assert_eq!(
        a.wheels
            .pressure_tire(&a.scene, wheel_a)
            .unwrap()
            .pressure_pa,
        b.wheels
            .pressure_tire(&b.scene, wheel_b)
            .unwrap()
            .pressure_pa
    );
}
