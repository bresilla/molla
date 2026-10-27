#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, RigidScene};
use molla_sim::{BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, MassProperties};
use molla_solvers::rigid_world::{GpuContactCapacity, RigidWorld};
use std::sync::Arc;

fn scene(restitution: f64, velocity: Vec3) -> (RigidScene, BodyHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::Y * 0.499),
            velocity: SpatialVector::new(velocity, Vec3::ZERO),
            additional_mass: Some(MassProperties {
                mass: 2.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 0.2,
            }),
            ..Default::default()
        })
        .unwrap();
    let material = ContactMaterial {
        stiffness: 1e5,
        damping: 100.,
        friction: Some(FrictionCoefficients::uniform(0.)),
        ..Default::default()
    };
    let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    ball.parent = Some(body);
    ball.mass = ColliderMass::Mass(0.);
    ball.restitution = restitution;
    ball.contact_material = material;
    scene.insert_collider(ball).unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(5., 0.1, 5.),
    });
    ground.pose = Transform::from_translation(-Vec3::Y * 0.1);
    ground.restitution = restitution;
    ground.contact_material = material;
    scene.insert_collider(ground).unwrap();
    (scene, body)
}

fn gpu(scene: RigidScene, engine: &Arc<Engine>) -> RigidWorld {
    RigidWorld::from_scene_gpu_with_contact_capacity(
        scene,
        engine.clone(),
        GpuContactCapacity {
            pairs: 8,
            contacts_per_island: 8,
        },
    )
    .unwrap()
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-restitution-gpu"]
fn reaction_capture_does_not_change_contact_momentum_transport() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    for substeps in [1, 4] {
        let (mut scene, body) = scene(0.5, Vec3::new(1., -1., 0.2));
        let handles: Vec<_> = scene.colliders().map(|(id, _)| id).collect();
        for handle in handles {
            let mut desc = scene.collider(handle).unwrap().clone();
            desc.contact_material.friction = Some(FrictionCoefficients {
                mu_static: 0.4,
                mu_dynamic: 0.2,
            });
            scene.replace_collider(handle, desc).unwrap();
        }
        let mut plain = gpu(scene.fork().unwrap(), &engine);
        let mut captured = gpu(scene, &engine);
        for world in [&mut plain, &mut captured] {
            world.solver.substeps = substeps;
        }
        captured.set_reaction_readout(true).unwrap();
        for frame in 0..12 {
            if frame == 4 || frame == 8 {
                captured
                    .set_reaction_readout(!captured.reaction_readout_enabled())
                    .unwrap();
            }
            plain.step(0.004).unwrap();
            captured.step(0.004).unwrap();
            assert_eq!(
                plain.scene.body_pose(body).unwrap(),
                captured.scene.body_pose(body).unwrap()
            );
            assert_eq!(
                plain.scene.body_velocity(body).unwrap(),
                captured.scene.body_velocity(body).unwrap()
            );
            let impulse = |world: &RigidWorld| {
                world
                    .contact_pairs()
                    .iter()
                    .flat_map(|pair| &pair.points)
                    .fold((0.0, Vec3::ZERO), |(normal, tangent), point| {
                        (
                            normal + point.normal_impulse,
                            tangent + point.tangent_impulse,
                        )
                    })
            };
            assert_eq!(
                impulse(&plain),
                impulse(&captured),
                "frame={frame} substeps={substeps}"
            );
        }
        assert!(captured.scene.body_velocity(body).unwrap().linear.y > 0.49);
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-restitution-gpu"]
fn live_zero_depth_contact_retains_penalty_support_without_repeated_impacts() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (mut scene, body) = scene(0., Vec3::ZERO);
    scene.set_gravity(-Vec3::Y * 10.).unwrap();
    scene
        .set_body_pose(body, Transform::from_translation(Vec3::Y * 0.5))
        .unwrap();
    let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
    let mut gpu = gpu(scene, &engine);
    cpu.solver.substeps = 4;
    gpu.solver.substeps = 4;
    for frame in 0..250 {
        cpu.step(0.004).unwrap();
        gpu.step(0.004).unwrap();
        let position = gpu.scene.body_pose(body).unwrap().position;
        let velocity = gpu.scene.body_velocity(body).unwrap().linear;
        assert!(
            (position - cpu.scene.body_pose(body).unwrap().position).length() < 3e-6,
            "frame={frame} GPU={position:?} CPU={:?}",
            cpu.scene.body_pose(body).unwrap().position
        );
        assert!(
            (velocity - cpu.scene.body_velocity(body).unwrap().linear).length() < 1e-4,
            "frame={frame} GPU={velocity:?} CPU={:?}",
            cpu.scene.body_velocity(body).unwrap().linear
        );
    }
    for world in [&cpu, &gpu] {
        assert!((world.scene.body_pose(body).unwrap().position.y - 0.4998).abs() < 1e-6);
        let impulse: f64 = world
            .contact_pairs()
            .iter()
            .flat_map(|pair| &pair.points)
            .map(|point| point.normal_impulse)
            .sum();
        assert!((impulse / 0.004 - 20.).abs() < 0.05);
    }
    assert_eq!(gpu.gpu_diagnostics().unwrap().model_compilations, 1);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-restitution-gpu"]
fn live_normal_impacts_publish_one_impulse_and_match_cpu_rebound() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    for restitution in [0., 0.5, 1.] {
        let (scene, body) = scene(restitution, -Vec3::Y);
        let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
        let mut gpu = gpu(scene, &engine);
        cpu.solver.substeps = 4;
        gpu.solver.substeps = 4;
        let mut impulse = 0.;
        for frame in 0..8 {
            cpu.step(0.004).unwrap();
            gpu.step(0.004)
                .unwrap_or_else(|e| panic!("e={restitution}, frame={frame}: {e}"));
            let position = gpu.scene.body_pose(body).unwrap().position;
            let velocity = gpu.scene.body_velocity(body).unwrap().linear;
            assert!(
                (velocity.y - restitution).abs() < 2e-5,
                "e={restitution}, frame={frame}, v={velocity:?}"
            );
            assert!((position - cpu.scene.body_pose(body).unwrap().position).length() < 3e-6);
            for pair in gpu.contact_pairs() {
                for point in pair.points {
                    impulse += point.normal_impulse;
                    assert!(point.tangent_impulse.length() < 1e-7);
                    assert!((point.force_on_b - point.normal * point.normal_force).length() < 1e-7);
                    assert!(
                        (point.normal_impulse - point.normal_force * point.sample_dt).abs() < 1e-12
                    );
                }
            }
        }
        assert!(
            (impulse - 2. * (1. + restitution)).abs() < 2e-5,
            "e={restitution}, impulse={impulse}"
        );
        assert_eq!(gpu.gpu_diagnostics().unwrap().model_compilations, 1);
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-restitution-gpu"]
fn failed_impact_frame_restores_history_and_recovers_after_compaction() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (mut scene, body) = scene(0.5, -Vec3::Y);
    let mut extra = scene
        .colliders()
        .find(|(_, c)| c.parent == Some(body))
        .unwrap()
        .1
        .clone();
    extra.pose = Transform::from_translation(Vec3::X * 0.2);
    let extra = scene.insert_collider(extra).unwrap();
    let mut world = RigidWorld::from_scene_gpu_with_contact_capacity(
        scene,
        engine,
        GpuContactCapacity {
            pairs: 8,
            contacts_per_island: 1,
        },
    )
    .unwrap();
    world.solver.substeps = 4;
    let stamp = world.scene.model_stamp();
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    let error = world.step(0.004).unwrap_err();
    assert!(
        error.to_string().contains("solver fault 2097152"),
        "{error}"
    );
    assert_eq!(world.scene.model_stamp(), stamp);
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
    assert!(world.contact_pairs().is_empty());
    assert_eq!(world.completed_steps(), 0);
    world.scene.remove_collider(extra).unwrap();
    world.step(0.004).unwrap();
    assert!((world.scene.body_velocity(body).unwrap().linear.y - 0.5).abs() < 2e-5);
    let impulse: f64 = world
        .contact_pairs()
        .iter()
        .flat_map(|p| &p.points)
        .map(|p| p.normal_impulse)
        .sum();
    assert!((impulse - 3.).abs() < 2e-5);
    let compilations = world.gpu_diagnostics().unwrap().model_compilations;
    world.scene.set_body_pose(body, pose).unwrap();
    world.scene.set_body_velocity(body, velocity).unwrap();
    world.step(0.004).unwrap();
    assert!((world.scene.body_velocity(body).unwrap().linear.y - 0.5).abs() < 2e-5);
    assert_eq!(
        world.gpu_diagnostics().unwrap().model_compilations,
        compilations
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-restitution-gpu"]
fn oblique_impact_reports_final_slip_and_matches_cpu_energy_and_impulse() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    for friction in [0.2, 2.] {
        let (mut scene, body) = scene(0.5, Vec3::new(1., -1., 0.));
        let handles: Vec<_> = scene.colliders().map(|(id, _)| id).collect();
        for handle in handles {
            let mut desc = scene.collider(handle).unwrap().clone();
            desc.contact_material.friction = Some(FrictionCoefficients {
                mu_static: friction,
                mu_dynamic: friction * 0.5,
            });
            scene.replace_collider(handle, desc).unwrap();
        }
        let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
        let mut gpu = gpu(scene, &engine);
        cpu.solver.substeps = 1;
        gpu.solver.substeps = 1;
        cpu.step(0.001).unwrap();
        gpu.step(0.001).unwrap();
        let actual = gpu.scene.body_velocity(body).unwrap();
        let expected = cpu.scene.body_velocity(body).unwrap();
        assert!((actual.linear - expected.linear).length() < 3e-5);
        assert!(
            (actual.angular - expected.angular).length() < 3e-5,
            "mu={friction}, GPU={actual:?}, CPU={expected:?}, GPU contacts={:?}, CPU contacts={:?}",
            gpu.contact_pairs(),
            cpu.contact_pairs()
        );
        assert!(
            actual.linear.length_squared() + 0.1 * actual.angular.length_squared() <= 2. + 1e-6
        );
        let actual = &gpu.contact_pairs()[0].points[0];
        let expected = &cpu.contact_pairs()[0].points[0];
        assert_eq!(actual.friction_regime, expected.friction_regime);
        assert!((actual.applied_friction - expected.applied_friction).abs() < 1e-6);
        assert!((actual.normal_impulse - expected.normal_impulse).abs() < 3e-5);
        assert!((actual.tangent_impulse - expected.tangent_impulse).length() < 3e-5);
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-restitution-gpu"]
fn impact_reactions_break_a_joint_and_continue_without_repeating_the_episode() {
    use molla_sim::JointType;
    use molla_sim::runtime::{JointBreakLimits, JointDesc};
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (mut scene, ball) = scene(0.5, -Vec3::Y);
    let child = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::Y * 0.999),
            velocity: SpatialVector::new(-Vec3::Y, Vec3::ZERO),
            additional_mass: Some(MassProperties {
                mass: 2.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 0.2,
            }),
            ..Default::default()
        })
        .unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(ball),
            child,
            kind: JointType::Fixed,
            axis: Vec3::Y,
            frame_parent: Transform::from_translation(Vec3::Y * 0.5),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_joint_break_limits(
            joint,
            JointBreakLimits {
                force: Some(1000.),
                torque: None,
            },
        )
        .unwrap();
    let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
    let mut gpu = gpu(scene, &engine);
    for world in [&mut cpu, &mut gpu] {
        world.solver.substeps = 4;
        world.set_reaction_readout(true).unwrap();
        world.step(0.004).unwrap();
        assert!(world.scene.joint_broken(joint).unwrap());
        assert_eq!(world.joint_break_events().len(), 1);
        let event = &world.joint_break_events()[0];
        assert_eq!(event.sample.substep, 0);
        assert!(
            (event.sample.child.linear.y - 3000.).abs() < 0.05,
            "{event:?}"
        );
        for body in [ball, child] {
            assert!((world.scene.body_velocity(body).unwrap().linear.y - 0.5).abs() < 2e-5);
        }
        let impulse: f64 = world
            .contact_pairs()
            .iter()
            .flat_map(|p| &p.points)
            .map(|p| p.normal_impulse)
            .sum();
        assert!((impulse - 6.).abs() < 3e-5, "impulse={impulse}");
        world.step(0.004).unwrap();
        assert_eq!(world.joint_break_events().len(), 1);
    }
    for body in [ball, child] {
        assert!(
            (gpu.scene.body_pose(body).unwrap().position
                - cpu.scene.body_pose(body).unwrap().position)
                .length()
                < 4e-6
        );
    }
}
