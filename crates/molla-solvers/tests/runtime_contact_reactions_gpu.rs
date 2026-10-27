#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, BodyKind, ColliderDesc, ColliderGeometry, JointDesc, JointHandle,
    RigidScene,
};
use molla_sim::{
    BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, JointType, MassProperties,
};
use molla_solvers::rigid_world::{GpuContactCapacity, RigidWorld};
use std::sync::Arc;

fn joint(
    scene: &mut RigidScene,
    child: BodyHandle,
    kind: JointType,
    frame: Transform,
) -> JointHandle {
    scene
        .insert_joint(JointDesc {
            parent: None,
            child,
            kind,
            axis: Vec3::X,
            frame_parent: frame,
            frame_child: Transform::IDENTITY,
        })
        .unwrap()
}

fn loaded_scene(kind: BodyKind) -> (RigidScene, BodyHandle, JointHandle, BodyHandle, JointHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::new(2.0, -10.0, 0.0)).unwrap();
    let ground = scene
        .insert_body(BodyDesc {
            kind,
            params: BodyParams::new().with_position(-Vec3::Y * 0.1),
            additional_mass: Some(MassProperties {
                mass: 3.0,
                local_com: Vec3::new(0.2, 0.0, 0.1),
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let support = joint(
        &mut scene,
        ground,
        JointType::Fixed,
        Transform::from_translation(-Vec3::Y * 0.1),
    );
    let block = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::Y * 0.49995),
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY / 3.0,
            }),
            ..Default::default()
        })
        .unwrap();
    let free = joint(&mut scene, block, JointType::Free, Transform::IDENTITY);
    let material = ContactMaterial {
        stiffness: 1e5,
        damping: 300.0,
        friction: Some(FrictionCoefficients {
            mu_static: 0.6,
            mu_dynamic: 0.2,
        }),
        ..Default::default()
    };
    for x in [-0.4, 0.4] {
        for z in [-0.4, 0.4] {
            let mut pad = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 });
            pad.parent = Some(block);
            pad.mass = ColliderMass::Mass(0.0);
            pad.pose = Transform::from_translation(Vec3::new(x, -0.4, z));
            pad.contact_material = material;
            scene.insert_collider(pad).unwrap();
        }
    }
    let mut slab = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(5.0, 0.1, 5.0),
    });
    slab.parent = Some(ground);
    slab.mass = ColliderMass::Mass(0.0);
    slab.contact_material = material;
    scene.insert_collider(slab).unwrap();
    (scene, ground, support, block, free)
}

fn near(a: SpatialVector, b: SpatialVector, tolerance: f64) {
    assert!(
        (a.linear - b.linear).length() < tolerance,
        "force {a:?} != {b:?}"
    );
    assert!(
        (a.angular - b.angular).length() < tolerance,
        "torque {a:?} != {b:?}"
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-reactions-gpu"]
fn supported_contact_reactions_include_normal_tangent_and_com_moments() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    for horizontal in [2.0, -2.0] {
        let (mut scene, ground, support, block, free) = loaded_scene(BodyKind::Dynamic);
        scene
            .set_gravity(Vec3::new(horizontal, -10.0, 0.0))
            .unwrap();
        let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
        let mut gpu = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        cpu.set_reaction_readout(true).unwrap();
        gpu.set_reaction_readout(true).unwrap();
        for substeps in [1, 4] {
            cpu.solver.substeps = substeps;
            gpu.solver.substeps = substeps;
            for _ in 0..20 {
                cpu.step(0.001 * substeps as f64).unwrap();
                gpu.step(0.001 * substeps as f64).unwrap();
                let pairs = gpu.contact_pairs();
                assert_eq!(pairs.len(), 4);
                for sample in gpu.joint_reaction_samples().unwrap() {
                    if sample.joint == free {
                        near(sample.child, SpatialVector::ZERO, 1e-3);
                        continue;
                    }
                    assert_eq!(sample.joint, support);
                    let gravity = Vec3::new(horizontal, -10.0, 0.0) * 3.0;
                    let arm = gpu.scene.body_world_com(ground).unwrap() - sample.anchors[1];
                    let mut external = SpatialVector::new(gravity, arm.cross(gravity));
                    for pair in &pairs {
                        let point = &pair.points[sample.substep];
                        let force = point.force_on_b
                            * if pair.bodies[1] == Some(ground) {
                                1.0
                            } else {
                                -1.0
                            };
                        external = external
                            + SpatialVector::new(
                                force,
                                (point.point - sample.anchors[1]).cross(force),
                            );
                    }
                    near(sample.child, external.scale(-1.0), 1e-4);
                    near(sample.parent, sample.child.scale(-1.0), 1e-4);
                    let reference = cpu
                        .joint_reaction_samples()
                        .unwrap()
                        .iter()
                        .find(|r| r.joint == sample.joint && r.substep == sample.substep)
                        .unwrap();
                    near(sample.child, reference.child, 0.01);
                }
                assert!(
                    (gpu.scene.body_pose(block).unwrap().position
                        - cpu.scene.body_pose(block).unwrap().position)
                        .length()
                        < 3e-5
                );
            }
        }
        let support = gpu.joint_reaction(support).unwrap();
        assert!((support.child.linear.y - 50.0).abs() < 0.02);
        assert!((support.child.linear.x + 5.0 * horizontal).abs() < 0.02);
        assert!(
            gpu.scene
                .state()
                .body_f
                .host()
                .unwrap()
                .iter()
                .all(|&f| f == SpatialVector::ZERO)
        );
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-reactions-gpu"]
fn contact_reaction_fault_discards_readouts_and_recovers_the_warm_pipeline() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (scene, _, support, body, _) = loaded_scene(BodyKind::Dynamic);
    let mut reference = RigidWorld::from_scene_gpu(scene.fork().unwrap(), engine.clone()).unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    world.set_reaction_readout(true).unwrap();
    reference.set_reaction_readout(true).unwrap();
    world.solver.substeps = 3;
    reference.solver.substeps = 3;
    for _ in 0..10 {
        world.step(0.003).unwrap();
        reference.step(0.003).unwrap();
    }
    assert!(world.joint_reaction(support).is_some());
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    let diagnostics = world.gpu_diagnostics().unwrap();
    world
        .scene
        .set_body_velocity(body, SpatialVector::new(Vec3::X * 1e100, Vec3::ZERO))
        .unwrap();
    let stamp = world.scene.dynamics_stamp();
    let error = world.step(0.003).unwrap_err();
    assert!(error.to_string().contains("solver fault"), "{error}");
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(world.scene.dynamics_stamp(), stamp);
    assert_eq!(world.gpu_diagnostics().unwrap(), diagnostics);
    assert_eq!(world.completed_steps(), 10);
    assert!(world.contact_pairs().is_empty());
    assert!(world.joint_reaction(support).is_none());
    assert!(
        world
            .joint_reaction_samples()
            .is_none_or(|samples| samples.is_empty())
    );
    world.scene.set_body_velocity(body, velocity).unwrap();
    world.step(0.003).unwrap();
    reference.step(0.003).unwrap();
    assert_eq!(world.completed_steps(), 11);
    assert_eq!(
        world.gpu_diagnostics().unwrap().model_compilations,
        diagnostics.model_compilations
    );
    assert_eq!(world.contact_pairs().len(), 4);
    assert_eq!(world.joint_reaction_samples().unwrap().len(), 6);
    near(
        world.joint_reaction(support).unwrap().child,
        reference.joint_reaction(support).unwrap().child,
        1e-4,
    );
    assert!(
        (world.scene.body_pose(body).unwrap().position
            - reference.scene.body_pose(body).unwrap().position)
            .length()
            < 1e-7
    );
    near(
        world.scene.body_velocity(body).unwrap(),
        reference.scene.body_velocity(body).unwrap(),
        1e-6,
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-reactions-gpu"]
fn reaction_capture_recompilation_preserves_contact_coordinate_residues() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::new(1.0, 3.0, 2.0)),
            ..Default::default()
        })
        .unwrap();
    joint(&mut scene, body, JointType::Free, Transform::IDENTITY);
    scene
        .set_body_velocity(body, SpatialVector::new(Vec3::X * 1e-5, Vec3::ZERO))
        .unwrap();
    let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 });
    collider.parent = Some(body);
    collider.mass = ColliderMass::Mass(0.0);
    scene.insert_collider(collider).unwrap();
    let mut plain = RigidWorld::from_scene_gpu(scene.fork().unwrap(), engine.clone()).unwrap();
    let mut captured = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    for step in 0..60 {
        plain.solver.substeps = [1, 3, 8][step / 20];
        captured.solver.substeps = plain.solver.substeps;
        if step % 10 == 5 {
            captured
                .set_reaction_readout(!captured.reaction_readout_enabled())
                .unwrap();
        }
        plain.step(0.01).unwrap();
        captured.step(0.01).unwrap();
        assert_eq!(
            captured.scene.body_pose(body).unwrap(),
            plain.scene.body_pose(body).unwrap(),
            "step={step}"
        );
        assert_eq!(
            captured.scene.body_velocity(body).unwrap(),
            plain.scene.body_velocity(body).unwrap()
        );
    }
    assert_eq!(plain.gpu_diagnostics().unwrap().model_compilations, 1);
    assert!(captured.gpu_diagnostics().unwrap().model_compilations > 5);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-reactions-gpu"]
fn fixed_joint_boundary_keeps_independent_contact_islands_separate() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (mut scene, ground, _, block, _) = loaded_scene(BodyKind::Dynamic);
    scene
        .set_body_pose(
            block,
            Transform::from_translation(Vec3::new(-1.0, 0.49995, 0.0)),
        )
        .unwrap();
    let mut desc = scene.body(block).unwrap().clone();
    desc.params.pose.position.x = 1.0;
    let other = scene.insert_body(desc).unwrap();
    let pads: Vec<_> = scene
        .colliders()
        .filter(|(_, c)| c.parent == Some(block))
        .map(|(_, c)| c.clone())
        .collect();
    for mut pad in pads {
        pad.parent = Some(other);
        scene.insert_collider(pad).unwrap();
    }
    let mut gpu = RigidWorld::from_scene_gpu_with_contact_capacity(
        scene,
        engine,
        GpuContactCapacity {
            pairs: 64,
            contacts_per_island: 4,
        },
    )
    .unwrap();
    gpu.step(0.001).unwrap();
    assert_eq!(gpu.contact_pairs().len(), 8);
    let probe = gpu
        .scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::Y * 2.9),
            ..Default::default()
        })
        .unwrap();
    gpu.scene
        .insert_joint(JointDesc {
            parent: Some(ground),
            child: probe,
            kind: JointType::Revolute,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(Vec3::Y * 3.0),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    gpu.step(0.001).unwrap();
    assert_eq!(gpu.contact_pairs().len(), 8);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-reactions-gpu"]
fn contact_axis_bounds_fail_before_advancing_until_coupled_solving_is_available() {
    use molla_sim::joint_control::JointAxis;
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (scene, _, _, block, free) = loaded_scene(BodyKind::Dynamic);
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    gpu.scene
        .set_joint_limits(free, JointAxis::LinX, Some([-1.0, 1.0]))
        .unwrap();
    let pose = gpu.scene.body_pose(block).unwrap();
    let error = gpu.step(0.001).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("coupled contacts and joint axis bounds"),
        "{error}"
    );
    assert_eq!(gpu.scene.body_pose(block).unwrap(), pose);
    assert_eq!(gpu.completed_steps(), 0);
    gpu.scene
        .set_joint_limits(free, JointAxis::LinX, None)
        .unwrap();
    gpu.step(0.001).unwrap();
    assert_eq!(gpu.contact_pairs().len(), 4);
}
