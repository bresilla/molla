use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, ColliderHandle, JointBreakLimits,
    JointDesc, RigidScene,
};
use molla_sim::{
    BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, JointType, MassProperties,
};
use molla_solvers::rigid_world::RigidWorld;

fn fixture(coefficient: f64) -> (RigidWorld, BodyHandle, ColliderHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::Y * 0.52),
            velocity: SpatialVector::new(-Vec3::Y, Vec3::ZERO),
            additional_mass: Some(MassProperties {
                mass: 1.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 0.1,
            }),
            ..Default::default()
        })
        .unwrap();
    let material = ContactMaterial {
        stiffness: 1e4,
        damping: 0.,
        friction: Some(FrictionCoefficients::uniform(0.)),
        ..Default::default()
    };
    let mut sphere = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    sphere.parent = Some(body);
    sphere.mass = ColliderMass::Mass(0.);
    sphere.restitution = coefficient;
    sphere.contact_material = material;
    scene.insert_collider(sphere).unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(5., 0.1, 5.),
    });
    ground.pose = Transform::from_translation(-Vec3::Y * 0.1);
    ground.restitution = coefficient;
    ground.contact_material = material;
    let ground = scene.insert_collider(ground).unwrap();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.solver.substeps = 16;
    (world, body, ground)
}

fn rebound(coefficient: f64, dt: f64) -> f64 {
    let (mut world, body, _) = fixture(coefficient);
    for _ in 0..(0.25 / dt).ceil() as usize {
        world.step(dt).unwrap();
    }
    world.scene.body_velocity(body).unwrap().linear.y.max(0.)
}

#[test]
fn resting_exact_touch_retains_the_episode_for_penalty_support() {
    let (mut world, body, _) = fixture(0.);
    world.solver.substeps = 1;
    world.scene.set_gravity(-Vec3::Y * 10.).unwrap();
    world
        .scene
        .set_body_pose(body, Transform::from_translation(Vec3::Y * 0.5))
        .unwrap();
    world
        .scene
        .set_body_velocity(body, SpatialVector::ZERO)
        .unwrap();
    world.step(0.001).unwrap();
    assert_eq!(world.contact_pairs().len(), 1);
    assert!((world.scene.body_velocity(body).unwrap().linear.y + 0.01).abs() < 1e-12);
    assert!((world.scene.body_pose(body).unwrap().position.y - 0.49999).abs() < 1e-12);
    world.step(0.001).unwrap();
    assert!((world.scene.body_velocity(body).unwrap().linear.y + 0.0199).abs() < 1e-12);
    assert!((world.scene.body_pose(body).unwrap().position.y - 0.4999701).abs() < 1e-12);
    let impulse: f64 = world
        .contact_pairs()
        .iter()
        .flat_map(|pair| &pair.points)
        .map(|point| point.normal_impulse)
        .sum();
    assert!((impulse - 0.0001).abs() < 1e-12);
}

#[test]
fn cpu_normal_rebound_consumes_restitution_across_step_sizes() {
    let measurements: Vec<_> = [1. / 60., 1. / 120., 1. / 240.]
        .into_iter()
        .flat_map(|dt| {
            [0., 0.5, 1.]
                .into_iter()
                .map(move |e| (dt, e, rebound(e, dt)))
        })
        .collect();
    for &(dt, e, measured) in &measurements {
        eprintln!("normal rebound dt={dt:e}, substeps=16, restitution={e}, measured={measured:e}");
    }
    for (dt, e, measured) in measurements {
        assert!(
            (measured - e).abs() <= 0.02,
            "dt={dt:e}, restitution={e}, measured={measured:e}"
        );
    }
}

/// The fixture's spring has no damping, so without the impact it returns the landing.
#[test]
fn without_contact_impacts_the_penalty_spring_takes_the_landing() {
    let (mut world, body, _) = fixture(0.);
    world.solver.contact_impacts = false;
    for _ in 0..60 {
        world.step(1. / 240.).unwrap();
    }
    let returned = world.scene.body_velocity(body).unwrap().linear.y;
    assert!(returned > 0.9, "returned {returned}");
    assert!(rebound(0., 1. / 240.) < 0.02);
}

#[test]
fn rebound_is_independent_of_onset_penalty_stiffness_damping_and_substeps() {
    for substeps in [1, 4, 16] {
        for stiffness in [1e2, 1e4, 1e6] {
            for damping in [0., 100.] {
                for coefficient in [0., 0.5, 1.] {
                    let (mut world, body, _) = fixture(coefficient);
                    world.solver.substeps = substeps;
                    let shapes: Vec<_> = world.scene.colliders().map(|(id, _)| id).collect();
                    for shape in shapes {
                        let mut desc = world.scene.collider(shape).unwrap().clone();
                        desc.contact_material.stiffness = stiffness;
                        desc.contact_material.damping = damping;
                        world.scene.replace_collider(shape, desc).unwrap();
                    }
                    for _ in 0..16 {
                        world.step(1. / 120.).unwrap();
                    }
                    let measured = world.scene.body_velocity(body).unwrap().linear.y;
                    assert!(
                        (measured - coefficient).abs() < 1e-9,
                        "substeps={substeps}, stiffness={stiffness}, damping={damping}, e={coefficient}, measured={measured}"
                    );
                }
            }
        }
    }
}

#[test]
fn passive_oblique_runtime_impacts_do_not_increase_kinetic_energy() {
    for substeps in [1, 4, 16] {
        for coefficient in [0., 0.5, 1.] {
            for friction in [
                FrictionCoefficients::uniform(0.),
                FrictionCoefficients {
                    mu_static: 0.8,
                    mu_dynamic: 0.4,
                },
                FrictionCoefficients::uniform(2.),
            ] {
                let (mut world, body, _) = fixture(coefficient);
                world.solver.substeps = substeps;
                world
                    .scene
                    .set_body_velocity(
                        body,
                        SpatialVector::new(Vec3::new(1., -1., 0.4), Vec3::new(0.2, 0.3, -0.1)),
                    )
                    .unwrap();
                let shapes: Vec<_> = world.scene.colliders().map(|(id, _)| id).collect();
                for shape in shapes {
                    let mut desc = world.scene.collider(shape).unwrap().clone();
                    desc.contact_material.friction = Some(friction);
                    world.scene.replace_collider(shape, desc).unwrap();
                }
                let energy = |world: &RigidWorld| {
                    let v = world.scene.body_velocity(body).unwrap();
                    0.5 * (v.linear.length_squared() + 0.1 * v.angular.length_squared())
                };
                let mut previous = energy(&world);
                let mut impulse = 0.;
                for tick in 0..32 {
                    world.step(1. / 120.).unwrap();
                    let current = energy(&world);
                    assert!(
                        current <= previous + 1e-9,
                        "tick={tick}, substeps={substeps}, e={coefficient}, friction={friction:?}, energy={previous} -> {current}"
                    );
                    previous = current;
                    for pair in world.contact_pairs() {
                        for point in pair.points {
                            impulse += point.normal_impulse;
                            assert!(
                                point.tangent_impulse.length()
                                    <= friction.mu_static * point.normal_impulse + 1e-9
                            );
                        }
                    }
                }
                assert!(
                    (impulse - 1. - coefficient).abs() < 1e-8,
                    "impulse={impulse}"
                );
            }
        }
    }
}

#[test]
fn tilted_box_impacts_keep_kinetic_energy_below_the_initial_budget() {
    for coefficient in [0., 0.5, 1.] {
        for friction in [0., 0.4] {
            let (mut world, body, _) = fixture(coefficient);
            let shape = world.scene.body_colliders(body).next().unwrap();
            let mut desc = world.scene.collider(shape).unwrap().clone();
            desc.geometry = ColliderGeometry::Cuboid {
                half_extents: Vec3::new(0.3, 0.5, 0.2),
            };
            world.scene.replace_collider(shape, desc).unwrap();
            world
                .scene
                .set_body_pose(
                    body,
                    Transform::new(
                        Vec3::Y,
                        Quat::from_rotation_z(0.4) * Quat::from_rotation_x(0.2),
                    ),
                )
                .unwrap();
            world
                .scene
                .set_body_velocity(
                    body,
                    SpatialVector::new(Vec3::new(0.3, -1., 0.2), Vec3::new(0.2, 0.3, -0.1)),
                )
                .unwrap();
            let shapes: Vec<_> = world.scene.colliders().map(|(id, _)| id).collect();
            for shape in shapes {
                let mut desc = world.scene.collider(shape).unwrap().clone();
                desc.contact_material.friction = Some(FrictionCoefficients::uniform(friction));
                world.scene.replace_collider(shape, desc).unwrap();
            }
            let initial = world.scene.body_velocity(body).unwrap();
            let initial_energy =
                0.5 * (initial.linear.length_squared() + 0.1 * initial.angular.length_squared());
            let mut touched = false;
            for tick in 0..240 {
                world.step(1. / 240.).unwrap_or_else(|e| {
                    panic!("tick={tick}, e={coefficient}, friction={friction}: {e}")
                });
                let v = world.scene.body_velocity(body).unwrap();
                let kinetic = 0.5 * (v.linear.length_squared() + 0.1 * v.angular.length_squared());
                touched |= world
                    .contact_pairs()
                    .iter()
                    .any(|p| p.points.iter().any(|p| p.normal_impulse > 0.));
                assert!(
                    kinetic <= initial_energy + 1e-8,
                    "tick={tick}, e={coefficient}, friction={friction}, initial={initial_energy}, kinetic={kinetic}"
                );
            }
            assert!(touched);
        }
    }
}

#[test]
fn impact_impulse_is_published_once_and_matches_the_velocity_jump() {
    for restitution in [0., 0.5, 1.] {
        let (mut world, body, _) = fixture(restitution);
        let mut impulse = 0.;
        for _ in 0..24 {
            world.step(1. / 240.).unwrap();
            for pair in world.contact_pairs() {
                for point in pair.points {
                    impulse += point.normal_impulse;
                    assert!(
                        (point.normal_impulse - point.normal_force * point.sample_dt).abs() < 1e-12
                    );
                    assert!(point.tangent_impulse.length() < 1e-12);
                }
            }
        }
        assert!((impulse - 1. - restitution).abs() < 1e-10, "{impulse}");
        assert!((world.scene.body_velocity(body).unwrap().linear.y - restitution).abs() < 1e-10);
    }
}

#[test]
fn failed_frame_after_impact_restores_the_episode_and_physical_state() {
    let (mut world, body, _) = fixture(0.5);
    let (mut reference, reference_body, _) = fixture(0.5);
    let initial = world.scene.body_pose(body).unwrap();
    let mut calls = 0;
    let error = world
        .step_with_forces(
            0.05,
            |_, _| false,
            |_, _, _, _| {
                calls += 1;
                if calls == 13 {
                    return Err(molla_core::Error::Solver(
                        "injected post-impact failure".into(),
                    ));
                }
                Ok(())
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("injected post-impact failure"));
    assert_eq!(world.scene.body_pose(body).unwrap(), initial);
    assert_eq!(world.scene.body_velocity(body).unwrap().linear, -Vec3::Y);
    assert_eq!(world.completed_steps(), 0);
    assert!(world.contact_pairs().is_empty());
    world.step(0.05).unwrap();
    reference.step(0.05).unwrap();
    assert_eq!(
        world.scene.body_pose(body).unwrap(),
        reference.scene.body_pose(reference_body).unwrap()
    );
    assert_eq!(
        world.scene.body_velocity(body).unwrap(),
        reference.scene.body_velocity(reference_body).unwrap()
    );
}

#[test]
fn explicit_velocity_and_material_edits_start_a_fresh_impact_episode() {
    let (mut world, body, ground) = fixture(0.5);
    for _ in 0..40 {
        world.step(1. / 240.).unwrap();
        if world.scene.body_velocity(body).unwrap().linear.y > 0. {
            break;
        }
    }
    assert!((world.scene.body_velocity(body).unwrap().linear.y - 0.5).abs() < 1e-10);
    world
        .scene
        .set_body_pose(body, Transform::from_translation(Vec3::Y * 0.499))
        .unwrap();
    world
        .scene
        .set_body_velocity(body, SpatialVector::new(-Vec3::Y, Vec3::ZERO))
        .unwrap();
    let mut material = world.scene.collider(ground).unwrap().clone();
    material.restitution = 1.;
    world.scene.replace_collider(ground, material).unwrap();
    world.step(1. / 240.).unwrap();
    assert!((world.scene.body_velocity(body).unwrap().linear.y - 1.).abs() < 1e-10);
}

#[test]
fn restoring_published_state_after_a_failed_edit_preserves_contact_history() {
    let (mut world, body, _) = fixture(0.5);
    world.scene.set_gravity(-Vec3::Y * 10.).unwrap();
    world
        .scene
        .set_body_pose(body, Transform::from_translation(Vec3::Y * 0.4998))
        .unwrap();
    world
        .scene
        .set_body_velocity(body, SpatialVector::ZERO)
        .unwrap();
    let mut reference = RigidWorld::from_scene(world.scene.fork().unwrap()).unwrap();
    world.solver.substeps = 1;
    reference.solver.substeps = 1;
    world.step(0.001).unwrap();
    reference.step(0.001).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    assert!(velocity.linear.y < -1e-4);
    world
        .scene
        .set_body_velocity(body, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    let error = world
        .step_with_forces(
            0.001,
            |_, _| false,
            |_, _, _, _| {
                Err(molla_core::Error::Solver(
                    "injected edited-input failure".into(),
                ))
            },
        )
        .unwrap_err();
    assert!(error.to_string().contains("edited-input failure"));
    world.scene.set_body_velocity(body, velocity).unwrap();
    world.step(0.001).unwrap();
    reference.step(0.001).unwrap();
    assert_eq!(
        world.scene.body_pose(body).unwrap(),
        reference.scene.body_pose(body).unwrap()
    );
    assert_eq!(
        world.scene.body_velocity(body).unwrap(),
        reference.scene.body_velocity(body).unwrap()
    );
}

#[test]
fn breaking_a_support_preserves_an_existing_contact_episode() {
    for restitution in [0., 0.8] {
        let (mut world, falling, ground_shape) = fixture(restitution);
        world.solver.substeps = 4;
        world.scene.set_gravity(-Vec3::Y * 10.).unwrap();
        world
            .scene
            .set_body_pose(falling, Transform::from_translation(Vec3::Y * 0.4998))
            .unwrap();
        world
            .scene
            .set_body_velocity(falling, SpatialVector::ZERO)
            .unwrap();
        let ground = world
            .scene
            .insert_body(BodyDesc {
                params: BodyParams::new().with_position(-Vec3::Y * 0.1),
                additional_mass: Some(MassProperties {
                    mass: 3.,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY,
                }),
                ..Default::default()
            })
            .unwrap();
        let support = world
            .scene
            .insert_joint(JointDesc {
                parent: None,
                child: ground,
                kind: JointType::Fixed,
                axis: Vec3::X,
                frame_parent: Transform::from_translation(-Vec3::Y * 0.1),
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        let mut collider = world.scene.collider(ground_shape).unwrap().clone();
        collider.parent = Some(ground);
        collider.pose = Transform::IDENTITY;
        collider.mass = ColliderMass::Mass(0.);
        world
            .scene
            .replace_collider(ground_shape, collider)
            .unwrap();
        world
            .scene
            .set_joint_break_limits(
                support,
                JointBreakLimits {
                    force: Some(31.),
                    torque: None,
                },
            )
            .unwrap();
        world.step(0.004).unwrap();
        assert_eq!(world.joint_break_events().len(), 1);
        assert_eq!(world.joint_break_events()[0].sample.substep, 0);
        let position = world.scene.body_pose(falling).unwrap().position.y;
        assert!(
            position < 0.4998,
            "persistent contact was projected as a new impact: e={restitution}, y={position}"
        );
        let mut positions = [-0.1, 0.4998];
        let mut velocities = [0., 0.];
        for step in 0..4 {
            let depth = f64::max(positions[0] + 0.6 - positions[1], 0.);
            let force = 1e4 * depth;
            velocities[1] += (-10. + force) * 0.001;
            if step != 0 {
                velocities[0] += (-10. - force / 3.) * 0.001;
            }
            for side in 0..2 {
                positions[side] += velocities[side] * 0.001;
            }
        }
        for (side, body) in [ground, falling].into_iter().enumerate() {
            let position = world.scene.body_pose(body).unwrap().position.y;
            let velocity = world.scene.body_velocity(body).unwrap().linear.y;
            assert!((position - positions[side]).abs() < 1e-10);
            assert!((velocity - velocities[side]).abs() < 1e-10);
        }
    }
}

#[test]
fn a_remote_impact_does_not_change_an_unconnected_contact_trajectory() {
    let (mut world, falling, _) = fixture(0.5);
    world.scene.set_gravity(-Vec3::Y * 9.81).unwrap();
    let other = world
        .scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::new(3., 0.5, 0.)),
            additional_mass: Some(MassProperties {
                mass: 1.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 0.1,
            }),
            ..Default::default()
        })
        .unwrap();
    let mut shape = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    shape.parent = Some(other);
    shape.mass = ColliderMass::Mass(0.);
    shape.contact_material.stiffness = 1e4;
    shape.contact_material.damping = 0.;
    world.scene.insert_collider(shape).unwrap();
    let mut reference_scene = world.scene.fork().unwrap();
    reference_scene.remove_body(falling).unwrap();
    let mut reference = RigidWorld::from_scene(reference_scene).unwrap();
    reference.solver.substeps = world.solver.substeps;
    let mut bounced = false;
    for _ in 0..20 {
        world.step(1. / 240.).unwrap();
        reference.step(1. / 240.).unwrap();
        bounced |= world.scene.body_velocity(falling).unwrap().linear.y > 0.;
        assert_eq!(
            world.scene.body_pose(other).unwrap(),
            reference.scene.body_pose(other).unwrap()
        );
        assert_eq!(
            world.scene.body_velocity(other).unwrap(),
            reference.scene.body_velocity(other).unwrap()
        );
    }
    assert!(bounced);
}

#[test]
fn restitution_runs_in_the_sleeping_selection_path() {
    let (mut world, body, _) = fixture(0.5);
    let asleep = world
        .scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::X * 20.),
            ..Default::default()
        })
        .unwrap();
    world.sleep_body(asleep).unwrap();
    let pose = world.scene.body_pose(asleep).unwrap();
    for _ in 0..24 {
        world.step(1. / 240.).unwrap();
    }
    assert!((world.scene.body_velocity(body).unwrap().linear.y - 0.5).abs() < 1e-10);
    assert_eq!(world.scene.body_pose(asleep).unwrap(), pose);
    assert!(world.is_sleeping(asleep));
}

#[test]
fn impact_joint_reactions_include_the_impulse_and_can_trigger_breakage() {
    for breaking in [false, true] {
        let (mut world, ball, _) = fixture(0.5);
        let child = world
            .scene
            .insert_body(BodyDesc {
                params: BodyParams::new().with_position(Vec3::Y * 1.02),
                additional_mass: Some(MassProperties {
                    mass: 1.,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY * 0.1,
                }),
                ..Default::default()
            })
            .unwrap();
        let joint = world
            .scene
            .insert_joint(JointDesc {
                parent: Some(ball),
                child,
                kind: JointType::Fixed,
                axis: Vec3::Y,
                frame_parent: Transform::from_translation(Vec3::Y * 0.5),
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        world.set_reaction_readout(true).unwrap();
        if breaking {
            world
                .scene
                .set_joint_break_limits(
                    joint,
                    JointBreakLimits {
                        force: Some(100.),
                        torque: None,
                    },
                )
                .unwrap();
            let stamp = world.scene.model_stamp();
            let initial = [ball, child].map(|body| {
                (
                    world.scene.body_pose(body).unwrap(),
                    world.scene.body_velocity(body).unwrap(),
                )
            });
            let error = world
                .step_with_forces(
                    0.05,
                    |_, _| false,
                    |scene, _, _, _| {
                        if scene.joint_broken(joint)? {
                            Err(molla_core::Error::Solver(
                                "injected post-impact detachment failure".into(),
                            ))
                        } else {
                            Ok(())
                        }
                    },
                )
                .unwrap_err();
            assert!(error.to_string().contains("post-impact detachment failure"));
            assert_eq!(world.scene.model_stamp(), stamp);
            assert!(!world.scene.joint_broken(joint).unwrap());
            assert!(world.joint_break_events().is_empty());
            assert!(world.contact_pairs().is_empty());
            assert_eq!(world.completed_steps(), 0);
            for (body, (pose, velocity)) in [ball, child].into_iter().zip(initial) {
                assert_eq!(world.scene.body_pose(body).unwrap(), pose);
                assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
            }
            world.step(0.05).unwrap();
            assert_eq!(world.joint_break_events().len(), 1);
        } else {
            let mut peak: f64 = 0.;
            for _ in 0..24 {
                world.step(1. / 240.).unwrap();
                for sample in world.joint_reaction_samples().unwrap() {
                    peak = peak.max(sample.child.linear.length());
                }
            }
            assert!((peak / (240. * 16.) - 1.5).abs() < 1e-8, "peak={peak}");
        }
        assert!((world.scene.body_velocity(ball).unwrap().linear.y - 0.5).abs() < 1e-8);
        assert!((world.scene.body_velocity(child).unwrap().linear.y - 0.5).abs() < 1e-8);
    }
}
