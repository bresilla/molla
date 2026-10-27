use super::*;
use molla_sim::ColliderMass;
use molla_sim::runtime::BodyKind;

#[test]
fn unrelated_props_preserve_complete_last_step_tyre_samples() {
    let (mut world, _, wheel, _, _) = fixture(false);
    world
        .wheels
        .configure_pressure_tire(wheel, pressure_config())
        .unwrap();
    world
        .wheels
        .set_pressures(&world.scene, &[(wheel, 300_000.0)])
        .unwrap();
    world.step(0.001).unwrap();
    let before = format!("{:?}", world.wheels.samples(&world.scene));
    assert!(!world.wheels.samples(&world.scene).is_empty());
    for kind in [BodyKind::Fixed, BodyKind::Dynamic] {
        let body = world
            .scene
            .insert_body(BodyDesc {
                kind,
                params: BodyParams::new()
                    .with_pose(Transform::from_translation(Vec3::splat(100.0))),
                ..Default::default()
            })
            .unwrap();
        assert_eq!(format!("{:?}", world.wheels.samples(&world.scene)), before);
        let mut desc = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.2 });
        desc.parent = Some(body);
        let collider = world.scene.insert_collider(desc).unwrap();
        assert_eq!(format!("{:?}", world.wheels.samples(&world.scene)), before);
        world
            .scene
            .set_collider_mass(collider, ColliderMass::Density(1200.0))
            .unwrap();
        assert_eq!(format!("{:?}", world.wheels.samples(&world.scene)), before);
        world.scene.remove_body(body).unwrap();
        assert_eq!(format!("{:?}", world.wheels.samples(&world.scene)), before);
    }
}

type Edit = fn(&mut RigidWorld, BodyHandle, BodyHandle, JointHandle, ColliderHandle);

#[test]
fn dependent_edits_clear_last_step_tyre_samples() {
    let edits: &[(&str, Edit)] = &[
        ("heading disabled", |w, chassis, _, _, _| {
            w.scene.set_body_enabled(chassis, false).unwrap();
        }),
        ("wheel kind", |w, _, wheel, _, _| {
            w.scene.set_body_kind(wheel, BodyKind::Fixed).unwrap();
        }),
        ("heading mass", |w, chassis, _, _, _| {
            let mut mass = w.scene.body_mass_properties(chassis).unwrap();
            mass.mass *= 2.0;
            w.scene.set_body_mass_properties(chassis, mass).unwrap();
        }),
        ("wheel collider density", |w, _, wheel, _, _| {
            let collider = w.scene.body_colliders(wheel).next().unwrap();
            w.scene
                .set_collider_mass(collider, ColliderMass::Density(1500.0))
                .unwrap();
        }),
        ("spin removed", |w, _, _, spin, _| {
            w.scene.remove_joint(spin).unwrap();
        }),
        ("spin limits", |w, _, _, spin, _| {
            w.scene
                .set_joint_limits(spin, JointAxis::AngX, Some([-1.0, 1.0]))
                .unwrap();
        }),
        ("heading teleported", |w, chassis, _, _, _| {
            w.scene
                .set_body_pose(chassis, Transform::from_translation(Vec3::Y))
                .unwrap();
        }),
        ("heading velocity", |w, chassis, _, _, _| {
            w.scene
                .set_body_velocity(chassis, SpatialVector::new(Vec3::X, Vec3::ZERO))
                .unwrap();
        }),
        ("raw state access", |w, _, _, _, _| {
            let _ = w.scene.state_mut();
        }),
        ("wheel removed", |w, _, wheel, _, _| {
            w.scene.remove_body(wheel).unwrap();
        }),
        ("ground sensor", |w, _, _, _, ground| {
            let mut desc = w.scene.collider(ground).unwrap().clone();
            desc.sensor = true;
            w.scene.replace_collider(ground, desc).unwrap();
        }),
        ("ground removed", |w, _, _, _, ground| {
            w.scene.remove_collider(ground).unwrap();
        }),
    ];
    for (label, edit) in edits {
        let (mut world, chassis, wheel, spin, ground) = fixture(false);
        world.step(0.001).unwrap();
        assert!(!world.wheels.samples(&world.scene).is_empty());
        edit(&mut world, chassis, wheel, spin, ground);
        assert!(world.wheels.samples(&world.scene).is_empty(), "{label}");
        body(&mut world, 1.0);
        assert!(
            world.wheels.samples(&world.scene).is_empty(),
            "{label} after rebuild"
        );
    }
}

#[test]
fn attachment_and_detachment_invalidate_both_sides_of_component() {
    let (mut world, chassis, _, _, _) = fixture(false);
    let trailer = body(&mut world, 50.0);
    world.step(0.001).unwrap();
    let hitch = world
        .scene
        .insert_joint(JointDesc {
            parent: Some(chassis),
            child: trailer,
            kind: JointType::Fixed,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
            axis: Vec3::X,
        })
        .unwrap();
    assert!(world.wheels.samples(&world.scene).is_empty());
    world.step(0.001).unwrap();
    assert!(!world.wheels.samples(&world.scene).is_empty());
    let mut mass = world.scene.body_mass_properties(trailer).unwrap();
    mass.mass *= 2.0;
    world.scene.set_body_mass_properties(trailer, mass).unwrap();
    assert!(world.wheels.samples(&world.scene).is_empty());
    world.step(0.001).unwrap();
    assert!(!world.wheels.samples(&world.scene).is_empty());
    world.scene.remove_joint(hitch).unwrap();
    assert!(world.wheels.samples(&world.scene).is_empty());
}

#[test]
fn untouched_hit_does_not_hide_other_registered_ground_changes() {
    let (mut world, _, _, _, ground) = fixture(false);
    let parent = world
        .scene
        .insert_body(BodyDesc {
            kind: BodyKind::Fixed,
            params: BodyParams::new().with_pose(Transform::from_translation(Vec3::Y * -10.0)),
            ..Default::default()
        })
        .unwrap();
    let mut desc = world.scene.collider(ground).unwrap().clone();
    desc.parent = Some(parent);
    let other = world.scene.insert_collider(desc).unwrap();
    world
        .wheels
        .register_ground(&world.scene, other, None)
        .unwrap();
    world.step(0.001).unwrap();
    assert!(
        world
            .wheels
            .samples(&world.scene)
            .iter()
            .all(|s| s.ground == ground)
    );
    assert!(!world.wheels.samples(&world.scene).is_empty());
    world.scene.set_body_enabled(parent, false).unwrap();
    assert!(world.wheels.samples(&world.scene).is_empty());
    world.step(0.001).unwrap();
    assert!(!world.wheels.samples(&world.scene).is_empty());
    let mut desc = world.scene.collider(other).unwrap().clone();
    desc.friction = 0.2;
    world.scene.replace_collider(other, desc).unwrap();
    assert!(world.wheels.samples(&world.scene).is_empty());
}

#[test]
fn rejected_topology_edits_retain_last_step_samples() {
    let (mut world, _, wheel, _, _) = fixture(false);
    world.step(0.001).unwrap();
    let before = format!("{:?}", world.wheels.samples(&world.scene));
    let mut desc = ColliderDesc::new(ColliderGeometry::Ball { radius: -1.0 });
    desc.parent = Some(wheel);
    assert!(world.scene.insert_collider(desc).is_err());
    assert_eq!(format!("{:?}", world.wheels.samples(&world.scene)), before);
}
