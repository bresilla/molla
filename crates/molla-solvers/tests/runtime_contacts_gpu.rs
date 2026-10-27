#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, ColliderHandle, RigidScene,
};
use molla_sim::{BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, MassProperties};
use molla_solvers::penalty_contact::FrictionRegime;
use molla_solvers::rigid_world::{GpuContactCapacity, RigidWorld};
use std::sync::Arc;

fn scene() -> (RigidScene, BodyHandle, Vec<ColliderHandle>, ColliderHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::new(2.0, -10.0, 0.0)).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::from_translation(Vec3::Y * 0.49995)),
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY / 3.0,
            }),
            ..Default::default()
        })
        .unwrap();
    let material = ContactMaterial {
        stiffness: 1e5,
        damping: 300.0,
        friction: Some(FrictionCoefficients {
            mu_static: 0.6,
            mu_dynamic: 0.2,
        }),
        ..Default::default()
    };
    let mut pads = Vec::new();
    for x in [-0.4, 0.4] {
        for z in [-0.4, 0.4] {
            let mut pad = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 });
            pad.parent = Some(body);
            pad.mass = ColliderMass::Mass(0.0);
            pad.pose = Transform::from_translation(Vec3::new(x, -0.4, z));
            pad.contact_material = material;
            pads.push(scene.insert_collider(pad).unwrap());
        }
    }
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(5.0, 0.1, 5.0),
    });
    ground.pose = Transform::from_translation(-Vec3::Y * 0.1);
    ground.contact_material = material;
    let ground = scene.insert_collider(ground).unwrap();
    (scene, body, pads, ground)
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-runtime-contacts-gpu"]
fn public_runtime_contacts_hold_and_publish_substep_forces() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (scene, body, pads, ground) = scene();
    let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    cpu.solver.substeps = 4;
    gpu.solver.substeps = 4;
    let mut error = 0.0_f64;
    for step in 0..1250 {
        cpu.step(0.004).unwrap();
        gpu.step(0.004).unwrap();
        error = error.max(
            (gpu.scene.body_pose(body).unwrap().position
                - cpu.scene.body_pose(body).unwrap().position)
                .length(),
        );
        let pairs = gpu.contact_pairs();
        assert_eq!(pairs.len(), 4, "step={step}");
        for pair in &pairs {
            assert!(pair.colliders.contains(&ground));
            assert!(pair.colliders.iter().any(|id| pads.contains(id)));
            assert_eq!(pair.points.len(), 4);
            for point in &pair.points {
                assert_eq!(point.friction_regime, FrictionRegime::Sticking);
                assert!((point.sample_dt - 0.001).abs() < 1e-12);
                assert!(
                    (point.normal_impulse - point.normal_force * point.sample_dt).abs() < 1e-12
                );
                assert!(
                    (point.force_on_b
                        - point.normal * point.normal_force
                        - point.tangent_impulse / point.sample_dt)
                        .length()
                        < 1e-10
                );
                assert_eq!(point.applied_friction, 0.6_f32 as f64);
            }
        }
        assert_eq!(
            gpu.scene.state().body_f.host().unwrap()[0],
            SpatialVector::ZERO
        );
    }
    eprintln!("public GPU contact hold CPU position error={error:e}");
    assert!(error < 3e-5);
    assert!(gpu.scene.body_pose(body).unwrap().position.x.abs() < 2e-4);
    assert!(gpu.scene.body_velocity(body).unwrap().linear.x.abs() < 1e-5);
    let diagnostics = gpu.gpu_diagnostics().unwrap();
    assert_eq!(diagnostics.model_compilations, 1);
    assert_eq!(diagnostics.completed_frames, 1250);
    assert_eq!(diagnostics.last_substeps, 4);
    assert!(diagnostics.last_state_readback_bytes > 128);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-runtime-contacts-gpu"]
fn contact_capacity_fault_rolls_back_and_topology_recovery_preserves_handles() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (scene, body, mut pads, _) = scene();
    let mut gpu = RigidWorld::from_scene_gpu_with_contact_capacity(
        scene,
        engine,
        GpuContactCapacity {
            pairs: 64,
            contacts_per_island: 3,
        },
    )
    .unwrap();
    let pose = gpu.scene.body_pose(body).unwrap();
    let velocity = gpu.scene.body_velocity(body).unwrap();
    let stamp = gpu.scene.dynamics_stamp();
    let error = gpu.step(0.008).unwrap_err();
    assert!(error.to_string().contains("solver fault"), "{error}");
    assert_eq!(gpu.scene.dynamics_stamp(), stamp);
    assert_eq!(gpu.scene.body_pose(body).unwrap(), pose);
    assert_eq!(gpu.scene.body_velocity(body).unwrap(), velocity);
    assert_eq!(gpu.completed_steps(), 0);
    assert_eq!(gpu.gpu_diagnostics().unwrap().completed_frames, 0);
    assert!(gpu.contact_pairs().is_empty());
    gpu.scene.remove_collider(pads.pop().unwrap()).unwrap();
    gpu.step(0.001).unwrap();
    assert_eq!(gpu.contact_pairs().len(), 3);
    assert!(gpu.scene.body(body).is_some());
    for pad in pads {
        assert!(gpu.scene.collider(pad).is_some());
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-runtime-contacts-gpu"]
fn contact_material_permissions_and_topology_mutations_recompile_without_stale_readouts() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (scene, body, pads, ground) = scene();
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    gpu.solver.substeps = 2;
    gpu.step(0.002).unwrap();
    let mut ground_desc = gpu.scene.collider(ground).unwrap().clone();
    ground_desc.enabled = false;
    gpu.scene
        .replace_collider(ground, ground_desc.clone())
        .unwrap();
    assert!(gpu.contact_pairs().is_empty());
    gpu.step(0.002).unwrap();
    assert!(gpu.contact_pairs().is_empty());
    ground_desc.enabled = true;
    gpu.scene.replace_collider(ground, ground_desc).unwrap();
    for id in pads.iter().copied().chain([ground]) {
        let mut desc = gpu.scene.collider(id).unwrap().clone();
        desc.contact_material.friction = Some(FrictionCoefficients {
            mu_static: 0.8,
            mu_dynamic: 0.4,
        });
        gpu.scene.replace_collider(id, desc).unwrap();
    }
    gpu.step(0.002).unwrap();
    assert_eq!(gpu.contact_pairs().len(), 4);
    for pair in gpu.contact_pairs() {
        for point in pair.points {
            assert_eq!(point.static_friction, 0.8_f32 as f64);
            assert_eq!(point.friction, 0.4_f32 as f64);
        }
    }
    for &pad in &pads {
        gpu.scene
            .set_collider_pair_excluded(pad, ground, true)
            .unwrap();
    }
    gpu.step(0.002).unwrap();
    assert!(gpu.contact_pairs().is_empty());
    for &pad in &pads {
        gpu.scene
            .set_collider_pair_excluded(pad, ground, false)
            .unwrap();
    }
    let saved = gpu.scene.collider(ground).unwrap().clone();
    gpu.scene.remove_collider(ground).unwrap();
    gpu.step(0.002).unwrap();
    assert!(gpu.contact_pairs().is_empty());
    let new_ground = gpu.scene.insert_collider(saved).unwrap();
    assert_ne!(ground, new_ground);
    gpu.step(0.002).unwrap();
    assert_eq!(gpu.contact_pairs().len(), 4);
    assert!(
        gpu.contact_pairs()
            .iter()
            .all(|p| p.colliders.contains(&new_ground))
    );
    assert!(gpu.scene.body(body).is_some());
    assert_eq!(gpu.gpu_diagnostics().unwrap().model_compilations, 6);
    let pose = gpu.scene.body_pose(body).unwrap();
    let mut called = false;
    let error = gpu
        .step_filtered(0.001, |_, _| {
            called = true;
            false
        })
        .unwrap_err();
    assert!(error.to_string().contains("host contact filter callbacks"));
    assert!(!called);
    assert_eq!(gpu.scene.body_pose(body).unwrap(), pose);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-runtime-contacts-gpu"]
fn warm_contact_fault_keeps_scene_and_recovers_without_recompilation() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let (scene, body, _, _) = scene();
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    gpu.solver.substeps = 3;
    for _ in 0..10 {
        gpu.step(0.003).unwrap();
    }
    let pose = gpu.scene.body_pose(body).unwrap();
    let velocity = gpu.scene.body_velocity(body).unwrap();
    let diagnostics = gpu.gpu_diagnostics().unwrap();
    gpu.scene
        .set_body_velocity(body, SpatialVector::new(Vec3::X * 1e100, Vec3::ZERO))
        .unwrap();
    let stamp = gpu.scene.dynamics_stamp();
    let error = gpu.step(0.003).unwrap_err();
    assert!(error.to_string().contains("solver fault"), "{error}");
    assert_eq!(gpu.scene.body_pose(body).unwrap(), pose);
    assert_eq!(gpu.scene.dynamics_stamp(), stamp);
    assert_eq!(gpu.gpu_diagnostics().unwrap(), diagnostics);
    assert_eq!(gpu.completed_steps(), 10);
    assert!(gpu.contact_pairs().is_empty());
    gpu.scene.set_body_velocity(body, velocity).unwrap();
    gpu.step(0.003).unwrap();
    assert_eq!(gpu.completed_steps(), 11);
    assert_eq!(gpu.gpu_diagnostics().unwrap().model_compilations, 1);
    assert_eq!(gpu.contact_pairs().len(), 4);
    assert!((gpu.scene.body_pose(body).unwrap().position - pose.position).length() < 1e-5);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-runtime-contacts-gpu"]
fn public_runtime_kinetic_contact_matches_cpu_and_analytic_acceleration() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(
        !adapter.is_software,
        "kinetic contact requires a hardware GPU"
    );
    println!("adapter={adapter:?}");
    let (mut scene, body, _, _) = scene();
    scene.set_gravity(Vec3::new(6.0, -10.0, 0.0)).unwrap();
    scene
        .set_body_velocity(body, SpatialVector::new(Vec3::X * 0.1, Vec3::ZERO))
        .unwrap();
    let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    gpu.solver.substeps = 4;
    cpu.solver.substeps = 4;
    let mut speed = 0.0;
    let mut peak = 0.0_f64;
    for step in 0..250 {
        cpu.step(0.004).unwrap();
        gpu.step(0.004).unwrap();
        for world in [&cpu, &gpu] {
            let pose = world.scene.body_pose(body).unwrap();
            let velocity = world.scene.body_velocity(body).unwrap();
            assert!(pose.position.is_finite() && pose.rotation.is_finite());
            assert!(velocity.linear.is_finite() && velocity.angular.is_finite());
            assert!(world.quarantined_bodies().is_empty());
            assert!(world.joint_break_events().is_empty());
            assert_eq!(world.completed_steps(), step + 1);
        }
        peak = peak.max(
            (cpu.scene.body_pose(body).unwrap().position
                - gpu.scene.body_pose(body).unwrap().position)
                .length(),
        );
        if step == 49 {
            speed = gpu.scene.body_velocity(body).unwrap().linear.x;
        }
    }
    let acceleration = (gpu.scene.body_velocity(body).unwrap().linear.x - speed) / 0.8;
    eprintln!(
        "public GPU kinetic contact CPU position error={peak:e}, acceleration={acceleration:e}"
    );
    assert!(peak < 3e-5);
    assert!(
        (gpu.scene.body_velocity(body).unwrap().linear
            - cpu.scene.body_velocity(body).unwrap().linear)
            .length()
            < 2e-3
    );
    assert!((acceleration - 4.0).abs() < 0.03);
    let pairs = gpu.contact_pairs();
    assert_eq!(pairs.len(), 4);
    assert!(
        pairs
            .iter()
            .flat_map(|p| &p.points)
            .all(|p| p.friction_regime == FrictionRegime::Sliding
                && p.applied_friction == 0.2_f32 as f64)
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-runtime-contacts-gpu"]
fn static_only_contact_scene_and_capacity_validation_are_transactional() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let mut scene = RigidScene::new().unwrap();
    let collider = scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Ball { radius: 1.0 }))
        .unwrap();
    for capacity in [
        GpuContactCapacity {
            pairs: 0,
            contacts_per_island: 1,
        },
        GpuContactCapacity {
            pairs: 1,
            contacts_per_island: 0,
        },
        GpuContactCapacity {
            pairs: u32::MAX,
            contacts_per_island: 1,
        },
        GpuContactCapacity {
            pairs: 1,
            contacts_per_island: u32::MAX,
        },
    ] {
        assert!(
            RigidWorld::from_scene_gpu_with_contact_capacity(
                scene.fork().unwrap(),
                engine.clone(),
                capacity
            )
            .is_err()
        );
    }
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    assert_eq!(
        gpu.gpu_contact_capacity(),
        Some(GpuContactCapacity::default())
    );
    gpu.step(0.01).unwrap();
    assert!(gpu.contact_pairs().is_empty());
    assert!(gpu.scene.collider(collider).is_some());
    gpu.solver.substeps = 16;
    gpu.step(0.016).unwrap();
    gpu.solver.substeps = 2;
    gpu.step(0.002).unwrap();
    assert_eq!(gpu.gpu_diagnostics().unwrap().model_compilations, 1);
    let body = gpu
        .scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::Y * 5.0),
            ..Default::default()
        })
        .unwrap();
    gpu.solver.substeps = 16;
    gpu.step(0.016).unwrap();
    assert!(gpu.scene.body_velocity(body).unwrap().linear.y < 0.0);
    gpu.scene.set_body_enabled(body, false).unwrap();
    let pose = gpu.scene.body_pose(body).unwrap();
    gpu.step(0.016).unwrap();
    assert_eq!(gpu.scene.body_pose(body).unwrap(), pose);
    assert!(gpu.contact_pairs().is_empty());
}
