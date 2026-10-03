#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, JointBreakLimits, JointBreakTrigger,
    JointDesc, JointHandle, RigidScene,
};
use molla_sim::{BodyParams, ColliderMass, ContactMaterial, JointType, MassProperties};
use molla_solvers::rigid_world::{GpuContactCapacity, RigidWorld};
use std::sync::Arc;

fn body(scene: &mut RigidScene, position: Vec3, mass: f64) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(position),
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn ball(scene: &mut RigidScene, parent: BodyHandle, offset: Vec3, radius: f64) {
    let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius });
    collider.parent = Some(parent);
    collider.pose = Transform::from_translation(offset);
    collider.mass = ColliderMass::Mass(0.0);
    collider.contact_material = ContactMaterial {
        stiffness: 1e4,
        damping: 100.0,
        ..Default::default()
    };
    scene.insert_collider(collider).unwrap();
}

fn supported() -> (RigidScene, BodyHandle, JointHandle, BodyHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(-Vec3::Y * 10.0).unwrap();
    let ground = body(&mut scene, -Vec3::Y * 0.1, 3.0);
    let joint = scene
        .insert_joint(JointDesc {
            parent: None,
            child: ground,
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(-Vec3::Y * 0.1),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let falling = body(&mut scene, Vec3::new(-1.0, 0.4998, 0.0), 2.0);
    ball(&mut scene, falling, Vec3::ZERO, 0.5);
    let mut slab = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(5.0, 0.1, 5.0),
    });
    slab.parent = Some(ground);
    slab.mass = ColliderMass::Mass(0.0);
    slab.contact_material = ContactMaterial {
        stiffness: 1e4,
        damping: 100.0,
        ..Default::default()
    };
    scene.insert_collider(slab).unwrap();
    scene
        .set_joint_break_limits(
            joint,
            JointBreakLimits {
                force: Some(31.0),
                torque: Some(1.0),
            },
        )
        .unwrap();
    (scene, ground, joint, falling)
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-breakage-gpu"]
fn contact_load_breaks_support_and_continues_with_cpu_event_and_state_agreement() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    for (substeps, capture) in [(1, false), (4, true)] {
        let (scene, ground, support, falling) = supported();
        let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
        let mut gpu = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        cpu.solver.substeps = substeps;
        gpu.solver.substeps = substeps;
        gpu.set_reaction_readout(capture).unwrap();
        let dt = substeps as f64 * 0.001;
        cpu.step(dt).unwrap();
        gpu.step(dt).unwrap();
        assert!(gpu.scene.joint_broken(support).unwrap());
        assert_eq!(gpu.joint_break_events().len(), 1);
        let actual = &gpu.joint_break_events()[0];
        let expected = &cpu.joint_break_events()[0];
        assert_eq!(actual.joint, support);
        assert_eq!(actual.trigger, JointBreakTrigger::Both);
        assert_eq!(actual.trigger, expected.trigger);
        assert_eq!(actual.sample.substep, 0);
        assert_eq!(actual.sample.substep, expected.sample.substep);
        assert_eq!(actual.sample.sample_dt, 0.001);
        assert!(
            (actual.sample.child.linear - expected.sample.child.linear).length() < 0.002,
            "substeps={substeps} GPU={:?} CPU={:?}",
            actual.sample.child.linear,
            expected.sample.child.linear
        );
        assert!((actual.sample.child.angular - expected.sample.child.angular).length() < 0.002);
        assert!((actual.sample.child.linear.y - 32.0).abs() < 0.002);
        assert!((actual.sample.child.angular.z + 2.0).abs() < 0.002);
        for b in [ground, falling] {
            let cpu_pose = cpu.scene.body_pose(b).unwrap();
            let gpu_pose = gpu.scene.body_pose(b).unwrap();
            let cpu_velocity = cpu.scene.body_velocity(b).unwrap();
            let gpu_velocity = gpu.scene.body_velocity(b).unwrap();
            assert!(
                (gpu_pose.position - cpu_pose.position).length() < 3e-5,
                "substeps={substeps}, body={b:?}, CPU pose={cpu_pose:?}, GPU pose={gpu_pose:?}, CPU velocity={cpu_velocity:?}, GPU velocity={gpu_velocity:?}"
            );
            assert!(
                (gpu_velocity.linear - cpu_velocity.linear).length() < 3e-4,
                "substeps={substeps}, body={b:?}, CPU velocity={cpu_velocity:?}, GPU velocity={gpu_velocity:?}"
            );
        }
        assert!(!gpu.contact_pairs().is_empty());
        if capture {
            let mean = gpu.joint_reaction(support).unwrap();
            assert!((mean.child.linear.y - actual.sample.child.linear.y / 4.0).abs() < 1e-4);
            assert_eq!(gpu.joint_reaction_samples().unwrap().len(), 1);
        }
        gpu.step(dt).unwrap();
        assert_eq!(gpu.joint_break_events().len(), 1);
    }
}

fn filtered_pair() -> (RigidScene, BodyHandle, BodyHandle, JointHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let a = body(&mut scene, Vec3::ZERO, 1.0);
    let b = body(&mut scene, Vec3::ZERO, 1.0);
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let mut settings = scene.joint_settings(joint).unwrap();
    settings.contacts_enabled = false;
    scene.set_joint_settings(joint, settings).unwrap();
    ball(&mut scene, a, Vec3::ZERO, 0.5);
    ball(&mut scene, b, Vec3::X * 0.9, 0.5);
    (scene, a, b, joint)
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-breakage-gpu"]
fn detachment_refreshes_contacts_without_removing_explicit_pair_exclusions() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    for excluded in [false, true] {
        let (mut scene, a, b, joint) = filtered_pair();
        if excluded {
            scene.set_body_pair_excluded(a, b, true).unwrap();
        }
        scene
            .set_joint_break_limits(
                joint,
                JointBreakLimits {
                    force: Some(0.0),
                    torque: None,
                },
            )
            .unwrap();
        let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        world.solver.substeps = 4;
        world.step(0.004).unwrap();
        assert!(world.scene.joint_broken(joint).unwrap());
        assert_eq!(world.joint_break_events().len(), 1);
        assert_eq!(world.contact_pairs().is_empty(), excluded);
        if !excluded {
            assert_eq!(world.contact_pairs()[0].points.len(), 3);
            assert!(world.scene.body_velocity(a).unwrap().linear.x < 0.0);
            assert!(world.scene.body_velocity(b).unwrap().linear.x > 0.0);
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-breakage-gpu"]
fn post_break_capacity_failure_restores_original_topology_and_precision() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (mut scene, _, b, joint) = filtered_pair();
    ball(&mut scene, b, -Vec3::X * 0.9, 0.5);
    let spectator = body(&mut scene, Vec3::new(10.0, 3.0, 2.0), 1.0);
    scene
        .set_body_velocity(spectator, SpatialVector::new(Vec3::X * 1e-5, Vec3::ZERO))
        .unwrap();
    let capacity = GpuContactCapacity {
        pairs: 8,
        contacts_per_island: 1,
    };
    let mut reference = RigidWorld::from_scene_gpu_with_contact_capacity(
        scene.fork().unwrap(),
        engine.clone(),
        capacity,
    )
    .unwrap();
    let mut world =
        RigidWorld::from_scene_gpu_with_contact_capacity(scene, engine, capacity).unwrap();
    world.set_reaction_readout(true).unwrap();
    reference.set_reaction_readout(true).unwrap();
    world.solver.substeps = 4;
    reference.solver.substeps = 4;
    for _ in 0..10 {
        world.step(0.004).unwrap();
        reference.step(0.004).unwrap();
    }
    world
        .scene
        .set_joint_break_limits(
            joint,
            JointBreakLimits {
                force: Some(0.0),
                torque: None,
            },
        )
        .unwrap();
    let stamp = world.scene.model_stamp();
    let dynamics = world.scene.dynamics_stamp();
    let pose = world.scene.body_pose(spectator).unwrap();
    let compilations = world.gpu_diagnostics().unwrap().model_compilations;
    let error = world.step(0.004).unwrap_err();
    assert!(
        error.to_string().ends_with("solver fault 2097152"),
        "{error}"
    );
    assert!(world.gpu_diagnostics().unwrap().model_compilations > compilations);
    assert_eq!(world.scene.model_stamp(), stamp);
    assert_eq!(world.scene.dynamics_stamp(), dynamics);
    assert_eq!(world.scene.body_pose(spectator).unwrap(), pose);
    assert!(!world.scene.joint_broken(joint).unwrap());
    assert!(world.joint_break_events().is_empty());
    assert!(world.contact_pairs().is_empty());
    assert_eq!(world.completed_steps(), 10);
    world
        .scene
        .set_joint_break_limits(joint, JointBreakLimits::default())
        .unwrap();
    let compilations = world.gpu_diagnostics().unwrap().model_compilations;
    for _ in 0..60 {
        world.step(0.004).unwrap();
        reference.step(0.004).unwrap();
        assert_eq!(
            world.scene.body_pose(spectator).unwrap(),
            reference.scene.body_pose(spectator).unwrap()
        );
    }
    assert_eq!(
        world.gpu_diagnostics().unwrap().model_compilations,
        compilations
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-breakage-gpu"]
fn successful_detachment_transfers_translation_residues_to_new_free_roots() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let a = body(&mut scene, Vec3::new(1.0, 3.0, 2.0), 1.0);
    let b = body(&mut scene, Vec3::new(2.0, 3.0, 2.0), 1.0);
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(Vec3::X),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    ball(&mut scene, a, Vec3::ZERO, 0.1);
    scene
        .set_body_velocity(a, SpatialVector::new(Vec3::X * 1e-5, Vec3::ZERO))
        .unwrap();
    let mut reference = RigidWorld::from_scene_gpu(scene.fork().unwrap(), engine.clone()).unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    world.set_reaction_readout(true).unwrap();
    reference.set_reaction_readout(true).unwrap();
    world.solver.substeps = 4;
    reference.solver.substeps = 4;
    for frame in 0..80 {
        if frame == 10 {
            world
                .scene
                .set_joint_break_limits(
                    joint,
                    JointBreakLimits {
                        force: Some(0.0),
                        torque: None,
                    },
                )
                .unwrap();
        }
        world.step(0.004).unwrap();
        reference.step(0.004).unwrap();
        for body in [a, b] {
            assert_eq!(
                world.scene.body_pose(body).unwrap(),
                reference.scene.body_pose(body).unwrap(),
                "frame={frame} body={body:?}"
            );
            assert_eq!(
                world.scene.body_velocity(body).unwrap(),
                reference.scene.body_velocity(body).unwrap()
            );
        }
    }
    assert!(world.scene.joint_broken(joint).unwrap());
    assert_eq!(world.joint_break_events().len(), 1);
    assert_eq!(world.joint_break_events()[0].frame, 10);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-breakage-gpu"]
fn sequential_contact_breaks_preserve_global_sample_times_and_frame_means() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (mut scene, _, first, _) = supported();
    let position = Vec3::new(20.0, -0.1, 0.0);
    let ground = body(&mut scene, position, 3.0);
    let second = scene
        .insert_joint(JointDesc {
            parent: None,
            child: ground,
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(position),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let falling = body(&mut scene, Vec3::new(19.0, 0.5, 0.0), 2.0);
    ball(&mut scene, falling, Vec3::ZERO, 0.5);
    let mut slab = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(5.0, 0.1, 5.0),
    });
    slab.parent = Some(ground);
    slab.mass = ColliderMass::Mass(0.0);
    slab.contact_material = ContactMaterial {
        stiffness: 1e4,
        damping: 100.0,
        ..Default::default()
    };
    scene.insert_collider(slab).unwrap();
    scene
        .set_joint_break_limits(
            second,
            JointBreakLimits {
                force: Some(30.5),
                torque: None,
            },
        )
        .unwrap();
    let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    cpu.solver.substeps = 4;
    gpu.solver.substeps = 4;
    cpu.set_reaction_readout(true).unwrap();
    gpu.set_reaction_readout(true).unwrap();
    cpu.step(0.004).unwrap();
    gpu.step(0.004).unwrap();
    assert_eq!(cpu.joint_break_events().len(), 2);
    assert_eq!(gpu.joint_break_events().len(), 2);
    for ((actual, expected), (step, joint)) in gpu
        .joint_break_events()
        .iter()
        .zip(cpu.joint_break_events())
        .zip([(0, first), (1, second)])
    {
        assert_eq!(actual.joint, joint);
        assert_eq!(actual.joint, expected.joint);
        assert_eq!(actual.sample.substep, step);
        assert_eq!(actual.sample.substep, expected.sample.substep);
        assert_eq!(actual.sample.time_offset, (step + 1) as f64 * 0.001);
        let analytic_force = if step == 0 { 32.0 } else { 31.1 };
        assert!((expected.sample.child.linear.y - analytic_force).abs() < 1e-9);
        assert!(
            (actual.sample.child.linear - expected.sample.child.linear).length() < 0.002,
            "substep={step} joint={joint:?} GPU={:?} CPU={:?}",
            actual.sample.child.linear,
            expected.sample.child.linear
        );
        assert!(
            (gpu.joint_reaction(joint).unwrap().child.linear
                - cpu.joint_reaction(joint).unwrap().child.linear)
                .length()
                < 0.002
        );
    }
    assert_eq!(gpu.joint_reaction_samples().unwrap().len(), 3);
    assert_eq!(gpu.gpu_diagnostics().unwrap().last_substeps, 4);
    assert_eq!(gpu.gpu_diagnostics().unwrap().completed_frames, 1);
}
