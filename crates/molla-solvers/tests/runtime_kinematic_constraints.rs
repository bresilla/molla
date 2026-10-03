use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, BodyKind, ColliderDesc, ColliderGeometry, DistanceLimits, JointDesc,
    JointSettings, RigidScene,
};
use molla_sim::{
    BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, JointType, MassProperties,
};
use molla_solvers::rigid_world::RigidWorld;

fn body(scene: &mut RigidScene, kind: BodyKind, position: Vec3) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            kind,
            params: BodyParams::new().with_position(position),
            additional_mass: Some(MassProperties {
                mass: 2.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 4.,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn near(actual: Vec3, expected: Vec3) {
    assert!(
        (actual - expected).length() < 1e-8,
        "actual={actual:?} expected={expected:?}"
    );
}

#[test]
fn accelerating_distance_boundary_uses_free_velocity_and_reports_support() {
    distance_case(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

fn distance_case(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    let near = |actual: Vec3, expected: Vec3| {
        assert!(
            (actual - expected).length() < tolerance,
            "actual={actual:?} expected={expected:?}"
        )
    };
    for sleep in [false, true] {
        eprintln!("prescribed distance sleep={sleep}");
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let parent = body(&mut scene, BodyKind::Kinematic, Vec3::ZERO);
        let child = body(&mut scene, BodyKind::Dynamic, Vec3::X);
        let joint = scene
            .insert_joint(JointDesc {
                parent: Some(parent),
                child,
                kind: JointType::Distance,
                axis: Vec3::X,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        scene
            .set_distance_limits(
                joint,
                DistanceLimits {
                    min: Some(1.),
                    max: Some(1.),
                },
            )
            .unwrap();
        scene
            .set_kinematic_acceleration(parent, Some(SpatialVector::new(Vec3::X * 3., Vec3::ZERO)))
            .unwrap();
        let mut world = make(scene);
        world.solver.substeps = 1;
        world.set_reaction_readout(true).unwrap();
        if sleep {
            world.sleep_body(child).unwrap();
        }
        world.step(0.001).unwrap();
        near(
            world.scene.body_velocity(child).unwrap().linear,
            Vec3::X * 0.003,
        );
        near(world.distance_reading(joint).unwrap().force, Vec3::X * 6.);
        near(
            world.joint_reaction(joint).unwrap().child.linear,
            Vec3::X * 6.,
        );
        assert!(!world.is_sleeping(child));
    }
}

#[test]
fn accelerating_contact_boundary_captures_from_rest_in_contact_only_and_common_paths() {
    contact_case(
        |scene| RigidWorld::from_scene(scene).unwrap(),
        1e-8,
        &CONTACT_CASES,
        1.,
    );
}

const CONTACT_CASES: [(bool, bool, bool); 5] = [
    (false, false, false),
    (true, false, false),
    (false, true, false),
    (true, true, false),
    (false, true, true),
];

fn contact_case(
    make: impl Fn(RigidScene) -> RigidWorld,
    tolerance: f64,
    cases: &[(bool, bool, bool)],
    linear_acceleration: f64,
) {
    let near = |actual: Vec3, expected: Vec3| {
        assert!(
            (actual - expected).length() < tolerance,
            "actual={actual:?} expected={expected:?}"
        )
    };
    for &(common, inherited, angular) in cases {
        eprintln!("prescribed contact common={common} inherited={inherited} angular={angular}");
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let boundary = body(&mut scene, BodyKind::Kinematic, -Vec3::Z * 0.1);
        let mut support = None;
        let ground = if inherited {
            let ground = body(&mut scene, BodyKind::Dynamic, -Vec3::Z * 0.1);
            support = Some(
                scene
                    .insert_joint(JointDesc {
                        parent: Some(boundary),
                        child: ground,
                        kind: JointType::Fixed,
                        axis: Vec3::X,
                        frame_parent: Transform::IDENTITY,
                        frame_child: Transform::IDENTITY,
                    })
                    .unwrap(),
            );
            ground
        } else {
            boundary
        };
        let slider = body(&mut scene, BodyKind::Dynamic, Vec3::Z * 0.499);
        scene
            .insert_joint(JointDesc {
                parent: None,
                child: slider,
                kind: JointType::Prismatic,
                axis: Vec3::X,
                frame_parent: Transform::from_translation(Vec3::Z * 0.499),
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        let material = ContactMaterial {
            stiffness: 1e5,
            damping: 0.,
            friction: Some(FrictionCoefficients {
                mu_static: 0.5,
                mu_dynamic: 0.25,
            }),
            ..Default::default()
        };
        for (parent, geometry) in [
            (
                ground,
                ColliderGeometry::Cuboid {
                    half_extents: Vec3::new(5., 5., 0.1),
                },
            ),
            (slider, ColliderGeometry::Ball { radius: 0.5 }),
        ] {
            let mut collider = ColliderDesc::new(geometry);
            collider.parent = Some(parent);
            collider.mass = ColliderMass::Mass(0.);
            collider.contact_material = material;
            scene.insert_collider(collider).unwrap();
        }
        let radial = if common {
            let joint = scene
                .insert_joint(JointDesc {
                    parent: Some(ground),
                    child: slider,
                    kind: JointType::Distance,
                    axis: Vec3::X,
                    frame_parent: Transform::from_translation(Vec3::new(-1., 0., 0.599)),
                    frame_child: Transform::IDENTITY,
                })
                .unwrap();
            scene
                .set_distance_limits(
                    joint,
                    DistanceLimits {
                        min: Some(1.),
                        max: Some(1.),
                    },
                )
                .unwrap();
            scene
                .set_joint_settings(
                    joint,
                    JointSettings {
                        enabled: true,
                        contacts_enabled: true,
                    },
                )
                .unwrap();
            Some(joint)
        } else {
            None
        };
        let acceleration = if angular {
            SpatialVector::new(Vec3::ZERO, Vec3::Y)
        } else {
            SpatialVector::new(Vec3::X * linear_acceleration, Vec3::ZERO)
        };
        scene
            .set_kinematic_acceleration(boundary, Some(acceleration))
            .unwrap();
        let mut world = make(scene);
        world.solver.substeps = 1;
        world.set_reaction_readout(true).unwrap();
        world.step(0.001).unwrap();
        let pairs = world.contact_pairs();
        assert_eq!(pairs.len(), 1);
        let speed = if angular {
            (pairs[0].points[0].point.z + 0.1) * 0.001
        } else {
            linear_acceleration * 0.001
        };
        let actual = world.scene.body_velocity(slider).unwrap().linear;
        assert!(
            (actual - Vec3::X * speed).length() < tolerance * 0.001,
            "velocity={actual:?} expected_x={speed}"
        );
        assert_eq!(
            pairs[0].points[0].friction_regime,
            molla_solvers::penalty_contact::FrictionRegime::Sticking
        );
        assert!(pairs[0].points[0].tangent_impulse.length() > 0.);
        if linear_acceleration > 1. && !angular && !common {
            let contact = &pairs[0].points[0];
            assert!(2. * linear_acceleration > contact.friction * contact.normal_force);
            assert!(2. * linear_acceleration < contact.static_friction * contact.normal_force);
        }
        if !common {
            assert!(
                (pairs[0].points[0].tangent_impulse.length() - 2. * speed).abs() < tolerance * 0.01
            );
        }
        if let Some(support) = support {
            let sign = if pairs[0].bodies[1] == Some(ground) {
                1.
            } else {
                -1.
            };
            let contact = &pairs[0].points[0];
            let external = contact.force_on_b * sign;
            let mut expected_force = acceleration.linear * 2. - external;
            let mut expected_torque =
                acceleration.angular * 4. - (contact.point + Vec3::Z * 0.1).cross(external);
            if let Some(radial) = radial {
                let child_force = world.distance_reading(radial).unwrap().force;
                expected_force += child_force;
                expected_torque += Vec3::new(-1., 0., 0.599).cross(child_force);
            }
            let reading = world.joint_reaction(support).unwrap();
            near(reading.child.linear, expected_force);
            near(reading.child.angular, expected_torque);
        }
    }
}

#[test]
fn angular_boundary_acceleration_transmits_through_locked_swing() {
    cone_case(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

fn cone_case(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    let near = |actual: Vec3, expected: Vec3| {
        assert!(
            (actual - expected).length() < tolerance,
            "actual={actual:?} expected={expected:?}"
        )
    };
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let parent = body(&mut scene, BodyKind::Kinematic, Vec3::ZERO);
    let child = body(&mut scene, BodyKind::Dynamic, Vec3::ZERO);
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(parent),
            child,
            kind: JointType::Ball,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_cone_limits(
            joint,
            molla_sim::runtime::SphericalConeLimits {
                angle0: Some(0.),
                angle1: Some(0.),
            },
        )
        .unwrap();
    scene
        .set_kinematic_acceleration(parent, Some(SpatialVector::new(Vec3::ZERO, Vec3::Y)))
        .unwrap();
    let mut world = make(scene);
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    world.step(0.001).unwrap();
    near(
        world.scene.body_velocity(child).unwrap().angular,
        Vec3::Y * 0.001,
    );
    near(world.cone_reading(joint).unwrap().torque, Vec3::Y * 4.);
    near(
        world.joint_reaction(joint).unwrap().child.angular,
        Vec3::Y * 4.,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_accelerating_boundaries_drive_distance_cone_and_common_contacts() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let make = |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
    distance_case(make, 2e-4);
    cone_case(make, 2e-4);
    contact_case(make, 2e-4, &CONTACT_CASES, 1.);
}

#[test]
fn accelerating_boundary_captures_above_dynamic_but_below_static_capacity() {
    contact_case(
        |scene| RigidWorld::from_scene(scene).unwrap(),
        1e-8,
        &[(false, true, false)],
        20.,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_accelerating_static_capture_uses_input_slip_not_predicted_slip() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    contact_case(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-4,
        &[(false, true, false)],
        20.,
    );
}

#[test]
fn sustained_accelerating_distance_has_bounded_step_refining_error() {
    for steps in [50, 100, 200] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let parent = body(&mut scene, BodyKind::Kinematic, Vec3::ZERO);
        let child = body(&mut scene, BodyKind::Dynamic, Vec3::X);
        let joint = scene
            .insert_joint(JointDesc {
                parent: Some(parent),
                child,
                kind: JointType::Distance,
                axis: Vec3::X,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        scene
            .set_distance_limits(
                joint,
                DistanceLimits {
                    min: Some(1.),
                    max: Some(1.),
                },
            )
            .unwrap();
        scene
            .set_kinematic_acceleration(parent, Some(SpatialVector::new(Vec3::X * 3., Vec3::ZERO)))
            .unwrap();
        let mut world = RigidWorld::from_scene(scene).unwrap();
        world.solver.substeps = 1;
        let dt = 1. / steps as f64;
        for _ in 0..steps {
            world.step(dt).unwrap();
        }
        near(world.scene.body_world_com(parent).unwrap(), Vec3::X * 1.5);
        near(
            world.scene.body_world_com(child).unwrap(),
            Vec3::X * (2.5 + 1.5 * dt * dt),
        );
        near(
            world.scene.body_velocity(child).unwrap().linear,
            Vec3::X * (3. - 1.5 * dt),
        );
        near(world.distance_reading(joint).unwrap().force, Vec3::X * 6.);
    }
}

#[test]
fn builder_rejects_nonkinematic_driven_flags_and_invalid_acceleration_atomically() {
    use molla_sim::{BodyFlags, ModelBuilder};
    let mut builder = ModelBuilder::new();
    builder.add_body(BodyParams::new().with_flags(BodyFlags::KINEMATIC_DRIVEN));
    assert!(builder.build().is_err());
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_flags(BodyFlags::KINEMATIC));
    let acceleration = SpatialVector::new(Vec3::X, Vec3::Y);
    builder
        .set_kinematic_acceleration(body, Some(acceleration))
        .unwrap();
    assert!(
        builder
            .set_kinematic_acceleration(
                body,
                Some(SpatialVector::new(Vec3::ZERO, Vec3::splat(f64::INFINITY)))
            )
            .is_err()
    );
    let model = builder.build().unwrap();
    assert_eq!(
        model.body_kinematic_acceleration.host().unwrap()[body.index()],
        acceleration
    );
    assert_ne!(
        model.body_flags.host().unwrap()[body.index()] & BodyFlags::KINEMATIC_DRIVEN,
        0
    );
}
