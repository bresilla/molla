use super::*;

fn add_machine(world: &mut RigidWorld) -> (BodyHandle, BodyHandle, JointHandle) {
    let mut add = |mass| {
        world
            .scene
            .insert_body(BodyDesc {
                params: BodyParams::new()
                    .with_pose(Transform::from_translation(Vec3::new(5.0, 0.49, 0.0))),
                additional_mass: Some(MassProperties {
                    mass,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY,
                }),
                ..Default::default()
            })
            .unwrap()
    };
    let chassis = add(100.0);
    let wheel = add(10.0);
    let spin = world
        .scene
        .insert_joint(JointDesc {
            parent: Some(chassis),
            child: wheel,
            kind: JointType::Revolute,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
            axis: -Vec3::Z,
        })
        .unwrap();
    (chassis, wheel, spin)
}

fn sample_text(world: &RigidWorld, wheel: BodyHandle) -> String {
    let samples = world.wheels.samples(&world.scene);
    let samples: Vec<_> = samples.iter().filter(|s| s.wheel == wheel).collect();
    assert!(!samples.is_empty(), "missing samples for {wheel:?}");
    format!("{samples:?}")
}

#[test]
fn independent_wheel_registration_configuration_and_removal_preserve_readout() {
    let (mut world, _, first, _, _) = fixture(false);
    world
        .wheels
        .configure_pressure_tire(first, pressure_config())
        .unwrap();
    world
        .wheels
        .set_pressures(&world.scene, &[(first, 300_000.0)])
        .unwrap();
    world.step(0.001).unwrap();
    let before = sample_text(&world, first);
    let (chassis, second, spin) = add_machine(&mut world);
    assert_eq!(sample_text(&world, first), before);
    world
        .wheels
        .insert(&world.scene, descriptor(chassis, second, spin))
        .unwrap();
    assert_eq!(sample_text(&world, first), before);
    world
        .wheels
        .configure_pressure_tire(second, pressure_config())
        .unwrap();
    assert_eq!(sample_text(&world, first), before);
    world
        .wheels
        .set_pressures(&world.scene, &[(second, 100_000.0)])
        .unwrap();
    assert_eq!(sample_text(&world, first), before);
    world.step(0.001).unwrap();
    let stepped = sample_text(&world, first);
    assert!(
        world
            .wheels
            .samples(&world.scene)
            .iter()
            .any(|s| s.wheel == second)
    );
    world.wheels.remove_pressure_tire(second).unwrap();
    assert_eq!(sample_text(&world, first), stepped);
    assert!(
        world
            .wheels
            .samples(&world.scene)
            .iter()
            .all(|s| s.wheel == first)
    );
    world
        .wheels
        .configure_pressure_tire(second, pressure_config())
        .unwrap();
    assert_eq!(sample_text(&world, first), stepped);
    world.step(0.001).unwrap();
    let stepped = sample_text(&world, first);
    world.wheels.remove(second).unwrap();
    assert_eq!(sample_text(&world, first), stepped);
    world.scene.remove_body(second).unwrap();
    world.scene.remove_body(chassis).unwrap();
    assert_eq!(sample_text(&world, first), stepped);
    assert!(world.wheels.remove(second).is_none());
    assert_eq!(sample_text(&world, first), stepped);
    assert_eq!(
        world
            .wheels
            .pressure_tire(&world.scene, first)
            .unwrap()
            .target_pressure_pa,
        300_000.0
    );
}

#[test]
fn independent_machine_mass_edit_only_invalidates_its_own_samples() {
    let (mut world, _, first, _, _) = fixture(false);
    let (chassis, second, spin) = add_machine(&mut world);
    world
        .wheels
        .insert(&world.scene, descriptor(chassis, second, spin))
        .unwrap();
    world.step(0.001).unwrap();
    let before = sample_text(&world, first);
    let mut mass = world.scene.body_mass_properties(chassis).unwrap();
    mass.mass *= 2.0;
    world.scene.set_body_mass_properties(chassis, mass).unwrap();
    assert_eq!(sample_text(&world, first), before);
    assert!(
        world
            .wheels
            .samples(&world.scene)
            .iter()
            .all(|s| s.wheel == first)
    );
}

#[test]
fn independent_machine_state_edits_and_removal_preserve_other_samples() {
    for velocity in [false, true] {
        let (mut world, _, first, _, _) = fixture(false);
        let (chassis, second, spin) = add_machine(&mut world);
        world
            .wheels
            .insert(&world.scene, descriptor(chassis, second, spin))
            .unwrap();
        world.step(0.001).unwrap();
        let before = sample_text(&world, first);
        if velocity {
            world
                .scene
                .set_body_velocity(chassis, SpatialVector::new(Vec3::X, Vec3::ZERO))
                .unwrap();
        } else {
            world
                .scene
                .set_body_pose(
                    chassis,
                    Transform::from_translation(Vec3::new(6.0, 0.49, 0.0)),
                )
                .unwrap();
        }
        assert_eq!(sample_text(&world, first), before);
        assert!(
            world
                .wheels
                .samples(&world.scene)
                .iter()
                .all(|s| s.wheel == first)
        );
        world.step(0.001).unwrap();
        let before = sample_text(&world, first);
        assert!(
            world
                .wheels
                .samples(&world.scene)
                .iter()
                .any(|s| s.wheel == second)
        );
        world.scene.remove_body(second).unwrap();
        assert_eq!(sample_text(&world, first), before);
        assert!(
            world
                .wheels
                .samples(&world.scene)
                .iter()
                .all(|s| s.wheel == first)
        );
        let _ = world.scene.state_mut();
        assert!(world.wheels.samples(&world.scene).is_empty());
    }
}

#[test]
fn filtered_sample_view_indexes_and_iterates_only_valid_components() {
    let (mut world, first_chassis, _, _, _) = fixture(false);
    let (chassis, second, spin) = add_machine(&mut world);
    world
        .wheels
        .insert(&world.scene, descriptor(chassis, second, spin))
        .unwrap();
    world.step(0.001).unwrap();
    let expected = sample_text(&world, second);
    world
        .scene
        .set_body_pose(first_chassis, Transform::from_translation(Vec3::Y))
        .unwrap();
    let samples = world.wheels.samples(&world.scene);
    assert!(!samples.is_empty());
    assert_eq!(format!("{samples:?}"), expected);
    assert_eq!(samples.first().unwrap().wheel, second);
    assert_eq!(samples.last().unwrap().wheel, second);
    assert!(samples.get(samples.len()).is_none());
    for i in 0..samples.len() {
        assert_eq!(samples[i].wheel, second);
        assert_eq!(samples.get(i).unwrap().wheel, second);
    }
    assert_eq!(samples.iter().count(), samples.len());
    assert_eq!(samples.iter().rev().count(), samples.len());
    assert_eq!(samples.into_iter().count(), samples.len());
    assert_eq!((&samples).into_iter().count(), samples.len());
}

#[test]
fn independent_machine_lifecycle_preserves_pressure_transition_and_dynamics() {
    let (mut reference, _, first, _, _) = fixture(false);
    let (mut actual, _, actual_first, _, _) = fixture(false);
    assert_eq!(first, actual_first);
    for world in [&mut reference, &mut actual] {
        world
            .wheels
            .configure_pressure_tire(first, pressure_config())
            .unwrap();
        world
            .wheels
            .set_pressures(&world.scene, &[(first, 300_000.0)])
            .unwrap();
    }
    let mut other = None;
    for tick in 0..80 {
        if tick == 10 {
            let (chassis, wheel, spin) = add_machine(&mut actual);
            actual
                .wheels
                .insert(&actual.scene, descriptor(chassis, wheel, spin))
                .unwrap();
            actual
                .wheels
                .configure_pressure_tire(wheel, pressure_config())
                .unwrap();
            actual
                .wheels
                .set_pressures(&actual.scene, &[(wheel, 100_000.0)])
                .unwrap();
            other = Some((chassis, wheel));
        }
        if tick == 30 {
            let (chassis, _) = other.unwrap();
            actual
                .scene
                .set_body_velocity(chassis, SpatialVector::new(Vec3::X, Vec3::ZERO))
                .unwrap();
        }
        if tick == 60 {
            let (chassis, wheel) = other.take().unwrap();
            actual.wheels.remove(wheel).unwrap();
            actual.scene.remove_body(wheel).unwrap();
            actual.scene.remove_body(chassis).unwrap();
        }
        if tick > 0 {
            assert_eq!(
                sample_text(&actual, first),
                sample_text(&reference, first),
                "before tick {tick}"
            );
        }
        reference.step(0.001).unwrap();
        actual.step(0.001).unwrap();
        assert_eq!(
            actual.scene.body_pose(first).unwrap(),
            reference.scene.body_pose(first).unwrap()
        );
        assert_eq!(
            actual.scene.body_velocity(first).unwrap(),
            reference.scene.body_velocity(first).unwrap()
        );
        assert_eq!(
            sample_text(&actual, first),
            sample_text(&reference, first),
            "after tick {tick}"
        );
        assert_eq!(
            actual
                .wheels
                .pressure_tire(&actual.scene, first)
                .unwrap()
                .pressure_pa,
            reference
                .wheels
                .pressure_tire(&reference.scene, first)
                .unwrap()
                .pressure_pa
        );
    }
}
