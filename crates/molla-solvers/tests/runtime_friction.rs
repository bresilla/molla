use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::{
    BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, MassProperties,
    runtime::{BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, ColliderHandle, RigidScene},
};
use molla_solvers::{penalty_contact::FrictionRegime, rigid_world::RigidWorld};

fn block(
    slope: f64,
    mu_static: f64,
    mu_dynamic: f64,
    rotation: Quat,
    offset: Vec3,
) -> (RigidWorld, BodyHandle, ColliderHandle, Vec3) {
    let frame = rotation * Quat::from_rotation_z(slope);
    let normal = frame * Vec3::Y;
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(rotation * -Vec3::Y * 10.0).unwrap();
    let mass = 2.0;
    let stiffness = 1e5;
    let penetration = mass * 10.0 * slope.cos() / (4.0 * stiffness);
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform {
                position: offset + normal * (0.5 - penetration),
                rotation: frame,
            }),
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * (mass / 6.0),
            }),
            ..Default::default()
        })
        .unwrap();
    let material = ContactMaterial {
        stiffness,
        damping: 300.0,
        friction: Some(FrictionCoefficients {
            mu_static,
            mu_dynamic,
        }),
        ..Default::default()
    };
    for x in [-0.4, 0.4] {
        for z in [-0.4, 0.4] {
            let mut pad = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 });
            pad.parent = Some(body);
            pad.mass = ColliderMass::Mass(0.0);
            pad.pose = Transform::from_translation(Vec3::new(x, -0.4, z));
            pad.contact_material = material;
            scene.insert_collider(pad).unwrap();
        }
    }
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(100.0, 0.1, 100.0),
    });
    ground.pose = Transform {
        position: offset - normal * 0.1,
        rotation: frame,
    };
    ground.contact_material = material;
    let ground = scene.insert_collider(ground).unwrap();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.solver.substeps = 4;
    (world, body, ground, frame * Vec3::X)
}

#[test]
fn inclined_loaded_block_holds_without_tangential_creep_for_five_seconds() {
    for rotation in [
        Quat::IDENTITY,
        Quat::from_rotation_y(0.7) * Quat::from_rotation_x(-0.4),
    ] {
        let (mut world, body, _, tangent) =
            block(0.3, 0.6, 0.2, rotation, Vec3::new(100.0, -30.0, 20.0));
        let start = world.scene.body_pose(body).unwrap().position;
        let mut peak: f64 = 0.0;
        let mut settled = start;
        let mut creep: f64 = 0.0;
        for i in 0..600 {
            world.step(1.0 / 120.0).unwrap();
            if i == 119 {
                settled = world.scene.body_pose(body).unwrap().position;
            }
            if i >= 120 {
                creep = creep.max(
                    (world.scene.body_pose(body).unwrap().position - settled)
                        .dot(tangent)
                        .abs(),
                );
            }
            peak = peak.max(
                (world.scene.body_pose(body).unwrap().position - start)
                    .dot(tangent)
                    .abs(),
            );
        }
        eprintln!("loaded static block tangent displacement={peak:e}, post-settle creep={creep:e}");
        assert!(peak < 2e-4, "creep={peak:e}");
        assert!(creep < 1e-6, "post-settle creep={creep:e}");
        assert!(!world.is_sleeping(body));
        assert!(
            world
                .contact_pairs()
                .iter()
                .flat_map(|p| &p.points)
                .any(|p| p.friction_regime == FrictionRegime::Sticking)
        );
        assert!(
            world
                .scene
                .body_velocity(body)
                .unwrap()
                .linear
                .dot(tangent)
                .abs()
                < 1e-5
        );
    }
}

#[test]
fn supercritical_slope_slides_at_the_dynamic_coefficient_acceleration() {
    let slope: f64 = 0.5;
    let (mut world, body, _, tangent) = block(slope, 0.4, 0.2, Quat::IDENTITY, Vec3::ZERO);
    let mut begin = 0.0;
    for i in 0..240 {
        world.step(1.0 / 240.0).unwrap();
        if i == 119 {
            begin = world
                .scene
                .body_velocity(body)
                .unwrap()
                .linear
                .dot(-tangent);
        }
    }
    let end = world
        .scene
        .body_velocity(body)
        .unwrap()
        .linear
        .dot(-tangent);
    let measured = (end - begin) / 0.5;
    let expected = 10.0 * (slope.sin() - 0.2 * slope.cos());
    eprintln!("kinetic acceleration={measured} expected={expected}");
    assert!((measured - expected).abs() < 0.02);
    assert!(
        world
            .contact_pairs()
            .iter()
            .flat_map(|p| &p.points)
            .any(|p| p.friction_regime == FrictionRegime::Sliding)
    );
}

#[test]
fn sliding_block_stops_resticks_and_reverses_under_a_new_load() {
    let (mut world, body, _, tangent) = block(0.0, 0.6, 0.3, Quat::IDENTITY, Vec3::ZERO);
    world
        .scene
        .set_body_velocity(body, SpatialVector::new(tangent, Vec3::ZERO))
        .unwrap();
    for _ in 0..240 {
        world.step(1.0 / 240.0).unwrap();
    }
    assert!(
        world
            .scene
            .body_velocity(body)
            .unwrap()
            .linear
            .dot(tangent)
            .abs()
            < 1e-6
    );
    let stopped = world.scene.body_pose(body).unwrap().position;
    world
        .scene
        .add_body_wrench(body, SpatialVector::new(tangent * 4.0, Vec3::ZERO))
        .unwrap();
    for _ in 0..240 {
        world.step(1.0 / 240.0).unwrap();
    }
    assert!(
        (world.scene.body_pose(body).unwrap().position - stopped)
            .dot(tangent)
            .abs()
            < 2e-4
    );
    world.scene.reset_body_forces(body).unwrap();
    world
        .scene
        .add_body_wrench(body, SpatialVector::new(-tangent * 16.0, Vec3::ZERO))
        .unwrap();
    for _ in 0..120 {
        world.step(1.0 / 240.0).unwrap();
    }
    assert!(world.scene.body_velocity(body).unwrap().linear.dot(tangent) < -2.0);
}

#[test]
fn conveyor_relative_slip_material_edits_and_separation_refresh_friction() {
    let (mut world, body, ground, tangent) = block(0.0, 0.6, 0.3, Quat::IDENTITY, Vec3::ZERO);
    world
        .scene
        .set_collider_surface_velocity(ground, Vec3::X * 0.5)
        .unwrap();
    for _ in 0..240 {
        world.step(1.0 / 240.0).unwrap();
    }
    assert!((world.scene.body_velocity(body).unwrap().linear.dot(tangent) - 0.5).abs() < 1e-5);
    let before = world.scene.model_stamp();
    let mut desc = world.scene.collider(ground).unwrap().clone();
    desc.contact_material.friction = Some(FrictionCoefficients {
        mu_static: 0.1,
        mu_dynamic: 0.2,
    });
    assert!(world.scene.replace_collider(ground, desc).is_err());
    assert_eq!(world.scene.model_stamp(), before);
    for (id, mut desc) in world
        .scene
        .colliders()
        .map(|(id, d)| (id, d.clone()))
        .collect::<Vec<_>>()
    {
        desc.contact_material.friction = Some(FrictionCoefficients::uniform(0.0));
        world.scene.replace_collider(id, desc).unwrap();
    }
    world
        .scene
        .set_collider_surface_velocity(ground, -Vec3::X)
        .unwrap();
    world.step(1.0 / 240.0).unwrap();
    assert!((world.scene.body_velocity(body).unwrap().linear.dot(tangent) - 0.5).abs() < 1e-5);
    assert!(
        world
            .contact_pairs()
            .iter()
            .flat_map(|p| &p.points)
            .all(|p| p.tangent_impulse == Vec3::ZERO
                && p.friction_regime == FrictionRegime::Inactive)
    );
    let mut pose = world.scene.body_pose(body).unwrap();
    pose.position.y += 2.0;
    world.scene.set_body_pose(body, pose).unwrap();
    world.step(1.0 / 240.0).unwrap();
    assert!(world.contact_pairs().is_empty());
    world.scene.remove_collider(ground).unwrap();
    world.step(1.0 / 240.0).unwrap();
    assert!(world.contact_pairs().is_empty());
}

#[test]
fn equal_legacy_coefficients_and_zero_friction_keep_their_authored_meaning() {
    for coefficient in [0.0, 0.4] {
        let (mut paired, body, _, tangent) =
            block(0.3, coefficient, coefficient, Quat::IDENTITY, Vec3::ZERO);
        let mut legacy = RigidWorld::from_scene(paired.scene.fork().unwrap()).unwrap();
        legacy.solver.substeps = paired.solver.substeps;
        for (id, mut desc) in legacy
            .scene
            .colliders()
            .map(|(id, d)| (id, d.clone()))
            .collect::<Vec<_>>()
        {
            desc.contact_material.friction = None;
            desc.friction = coefficient;
            legacy.scene.replace_collider(id, desc).unwrap();
        }
        for _ in 0..120 {
            paired.step(1.0 / 240.0).unwrap();
            legacy.step(1.0 / 240.0).unwrap();
            assert!(
                (paired.scene.body_pose(body).unwrap().position
                    - legacy.scene.body_pose(body).unwrap().position)
                    .length()
                    < 1e-10
            );
        }
        let speed = paired
            .scene
            .body_velocity(body)
            .unwrap()
            .linear
            .dot(-tangent);
        if coefficient == 0.0 {
            assert!((speed - 10.0 * 0.3_f64.sin() * 0.5).abs() < 1e-5);
        } else {
            assert!(speed.abs() < 1e-5);
        }
    }
}

#[test]
fn collider_endpoint_exchange_and_dense_compaction_preserve_motion() {
    let (mut original, body, ground, _) = block(
        0.5,
        0.4,
        0.2,
        Quat::from_rotation_y(0.3),
        Vec3::new(10.0, 20.0, -30.0),
    );
    let mut exchanged = RigidWorld::from_scene(original.scene.fork().unwrap()).unwrap();
    exchanged.solver.substeps = original.solver.substeps;
    let pads: Vec<_> = exchanged
        .scene
        .colliders()
        .filter(|(id, _)| *id != ground)
        .map(|(id, d)| (id, d.clone()))
        .collect();
    for (id, _) in &pads {
        exchanged.scene.remove_collider(*id).unwrap();
    }
    for (_, desc) in pads {
        exchanged.scene.insert_collider(desc).unwrap();
    }
    for _ in 0..120 {
        original.step(1.0 / 240.0).unwrap();
        exchanged.step(1.0 / 240.0).unwrap();
        assert!(
            (original.scene.body_velocity(body).unwrap().linear
                - exchanged.scene.body_velocity(body).unwrap().linear)
                .length()
                < 1e-6
        );
    }
    assert!(
        (original.scene.body_pose(body).unwrap().position
            - exchanged.scene.body_pose(body).unwrap().position)
            .length()
            < 1e-6
    );
}

#[test]
fn cuboid_manifold_holds_a_loaded_incline_without_sleeping() {
    let (mut world, body, ground, tangent) = block(0.3, 0.6, 0.2, Quat::IDENTITY, Vec3::ZERO);
    let material = world.scene.collider(ground).unwrap().contact_material;
    let pads: Vec<_> = world
        .scene
        .colliders()
        .filter(|(id, _)| *id != ground)
        .map(|(id, _)| id)
        .collect();
    for pad in pads {
        world.scene.remove_collider(pad).unwrap();
    }
    let mut cube = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::splat(0.5),
    });
    cube.parent = Some(body);
    cube.mass = ColliderMass::Mass(0.0);
    cube.contact_material = material;
    world.scene.insert_collider(cube).unwrap();
    let start = world.scene.body_pose(body).unwrap().position;
    let mut peak: f64 = 0.0;
    for _ in 0..600 {
        world.step(1.0 / 120.0).unwrap();
        peak = peak.max(
            (world.scene.body_pose(body).unwrap().position - start)
                .dot(tangent)
                .abs(),
        );
    }
    eprintln!("cuboid incline displacement={peak:e}");
    assert!(peak < 2e-4);
    assert!(!world.is_sleeping(body));
    assert!(
        world
            .scene
            .body_velocity(body)
            .unwrap()
            .linear
            .dot(tangent)
            .abs()
            < 1e-5
    );
}

#[test]
fn a_sleeping_contact_does_not_leak_impulses_and_surface_edits_wake_it() {
    let (mut world, body, ground, tangent) = block(0.0, 0.6, 0.3, Quat::IDENTITY, Vec3::ZERO);
    for _ in 0..120 {
        world.step(1.0 / 240.0).unwrap();
    }
    world.sleep_body(body).unwrap();
    world.step(1.0 / 240.0).unwrap();
    assert!(world.is_sleeping(body));
    assert!(
        world
            .solver
            .contact_forces()
            .iter()
            .all(|f| f.tangent_impulse == Vec3::ZERO)
    );
    world
        .scene
        .set_collider_surface_velocity(ground, Vec3::X * 0.5)
        .unwrap();
    world.step(1.0 / 240.0).unwrap();
    assert!(!world.is_sleeping(body));
    assert!(world.scene.body_velocity(body).unwrap().linear.dot(tangent) > 0.001);
}

#[test]
fn regularized_friction_slides_a_block_at_the_dynamic_coefficient() {
    let slope: f64 = 0.5;
    let (mut world, body, _, tangent) = block(slope, 0.4, 0.2, Quat::IDENTITY, Vec3::ZERO);
    world.solver.regularized_friction = true;
    let mut begin = 0.0;
    for i in 0..240 {
        world.step(1.0 / 240.0).unwrap();
        if i == 119 {
            begin = world
                .scene
                .body_velocity(body)
                .unwrap()
                .linear
                .dot(-tangent);
        }
    }
    let end = world
        .scene
        .body_velocity(body)
        .unwrap()
        .linear
        .dot(-tangent);
    let measured = (end - begin) / 0.5;
    let expected = 10.0 * (slope.sin() - 0.2 * slope.cos());
    assert!(
        (measured - expected).abs() < 0.05,
        "{measured} != {expected}"
    );
    assert!(
        world
            .contact_pairs()
            .iter()
            .flat_map(|p| &p.points)
            .all(|p| p.friction_regime == FrictionRegime::Regularized)
    );
}
