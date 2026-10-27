#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyKind, ColliderDesc, ColliderGeometry, JointBreakLimits, JointDesc, RigidScene,
};
use molla_sim::{BodyParams, ColliderMass, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;
use std::sync::Arc;

fn retained_motion(kind: BodyKind, with_collider: bool) {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            kind,
            params: BodyParams::new().with_position(Vec3::X * 1024.),
            velocity: SpatialVector::new(Vec3::X * 1e-3, Vec3::ZERO),
            ..Default::default()
        })
        .unwrap();
    if kind == BodyKind::Kinematic {
        scene
            .set_kinematic_acceleration(body, Some(SpatialVector::ZERO))
            .unwrap();
        assert_eq!(scene.model().joint_coord_total, 0);
    } else {
        assert_eq!(scene.model().joint_coord_total, 7);
    }
    if with_collider {
        let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 });
        collider.parent = Some(body);
        collider.mass = ColliderMass::Mass(0.0);
        scene.insert_collider(collider).unwrap();
    }
    let mut reference = RigidWorld::from_scene_gpu(scene.fork().unwrap(), engine.clone()).unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    for frame in 0..60 {
        world.solver.substeps = [1, 3, 4][frame / 20];
        reference.solver.substeps = world.solver.substeps;
        if frame == 17 || frame == 37 {
            world
                .set_reaction_readout(!world.reaction_readout_enabled())
                .unwrap();
        }
        if frame == 31 {
            let pose = world.scene.body_pose(body).unwrap();
            let velocity = world.scene.body_velocity(body).unwrap();
            let diagnostics = world.gpu_diagnostics().unwrap();
            world
                .scene
                .set_body_velocity(body, SpatialVector::new(Vec3::X * 1e100, Vec3::ZERO))
                .unwrap();
            let error = world.step(0.01).unwrap_err();
            assert!(error.to_string().contains("solver fault"), "{error}");
            assert_eq!(world.scene.body_pose(body).unwrap(), pose);
            assert_eq!(world.gpu_diagnostics().unwrap(), diagnostics);
            world.scene.set_body_velocity(body, velocity).unwrap();
        }
        reference.step(0.01).unwrap();
        world.step(0.01).unwrap();
        assert_eq!(
            world.scene.body_pose(body).unwrap(),
            reference.scene.body_pose(body).unwrap(),
            "frame={frame}, kind={kind:?}, collider={with_collider}"
        );
        assert_eq!(
            world.scene.body_velocity(body).unwrap(),
            reference.scene.body_velocity(body).unwrap()
        );
        let travel = world.scene.body_pose(body).unwrap().position.x - 1024.;
        assert!(
            (travel - (frame + 1) as f64 * 1e-5).abs() <= 2_f64.powi(-14),
            "frame={frame}, travel={travel:e}"
        );
    }
    assert_eq!(reference.gpu_diagnostics().unwrap().model_compilations, 1);
    assert!(world.gpu_diagnostics().unwrap().model_compilations >= 3);
    let travel = world.scene.body_pose(body).unwrap().position.x - 1024.;
    eprintln!(
        "retained GPU motion kind={kind:?}, collider={with_collider}, travel={travel:e}, expected=6e-4"
    );
    world
        .scene
        .set_body_pose(body, Transform::from_translation(Vec3::X * 2048.))
        .unwrap();
    for _ in 0..13 {
        world.step(0.01).unwrap();
    }
    let travel = world.scene.body_pose(body).unwrap().position.x - 2048.;
    assert!(
        (travel - 13e-5).abs() <= 2_f64.powi(-13),
        "edited travel={travel:e}"
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-pose-precision-gpu"]
fn zero_coordinate_motion_survives_frames_capture_rebuilds_faults_and_pose_edits() {
    for collider in [false, true] {
        retained_motion(BodyKind::Kinematic, collider);
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-pose-precision-gpu"]
fn free_root_motion_survives_frames_capture_rebuilds_faults_and_pose_edits() {
    for collider in [false, true] {
        retained_motion(BodyKind::Dynamic, collider);
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-pose-precision-gpu"]
fn zero_coordinate_break_transfers_child_position_residue_to_free_coordinates() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let parent = scene
        .insert_body(BodyDesc {
            kind: BodyKind::Kinematic,
            params: BodyParams::new().with_position(Vec3::X * 1024.),
            velocity: SpatialVector::new(Vec3::X * 1e-3, Vec3::ZERO),
            ..Default::default()
        })
        .unwrap();
    scene
        .set_kinematic_acceleration(parent, Some(SpatialVector::ZERO))
        .unwrap();
    let child = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::X * 1025.),
            ..Default::default()
        })
        .unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(parent),
            child,
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(Vec3::X),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_joint_break_limits(
            joint,
            JointBreakLimits {
                force: Some(1.),
                torque: None,
            },
        )
        .unwrap();
    assert_eq!(scene.model().joint_coord_total, 0);
    let mut reference = RigidWorld::from_scene_gpu(scene.fork().unwrap(), engine.clone()).unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    world.solver.substeps = 3;
    reference.solver.substeps = 3;
    for frame in 0..60 {
        if frame == 17 {
            world
                .scene
                .add_body_wrench(child, SpatialVector::new(Vec3::Y * 2., Vec3::ZERO))
                .unwrap();
            let poses = [parent, child].map(|body| world.scene.body_pose(body).unwrap());
            let velocities = [parent, child].map(|body| world.scene.body_velocity(body).unwrap());
            let stamp = world.scene.model_stamp();
            let dynamics = world.scene.dynamics_stamp();
            let compilations = world.gpu_diagnostics().unwrap().model_compilations;
            let error = world.step(1e20).unwrap_err();
            eprintln!("post-break finite-state rejection: {error}");
            assert!(world.gpu_diagnostics().unwrap().model_compilations > compilations);
            assert_eq!(world.completed_steps(), 17);
            assert_eq!(world.scene.model_stamp(), stamp);
            assert_eq!(world.scene.dynamics_stamp(), dynamics);
            assert_eq!(world.scene.model().joint_coord_total, 0);
            assert!(!world.scene.joint_broken(joint).unwrap());
            assert!(world.joint_break_events().is_empty());
            assert_eq!(
                [parent, child].map(|body| world.scene.body_pose(body).unwrap()),
                poses
            );
            assert_eq!(
                [parent, child].map(|body| world.scene.body_velocity(body).unwrap()),
                velocities
            );
        }
        world.step(0.01).unwrap();
        reference.step(0.01).unwrap();
        assert_eq!(world.scene.joint_broken(joint).unwrap(), frame >= 17);
        for body in [parent, child] {
            assert_eq!(
                world.scene.body_pose(body).unwrap().position.x,
                reference.scene.body_pose(body).unwrap().position.x,
                "frame={frame}, body={body:?}"
            );
        }
    }
    assert_eq!(world.scene.model().joint_coord_total, 7);
    assert_eq!(world.joint_break_events().len(), 1);
    assert_eq!(world.joint_break_events()[0].sample.substep, 0);
    let travel = world.scene.body_pose(child).unwrap().position.x - 1025.;
    assert!((travel - 6e-4).abs() <= 2_f64.powi(-14));
    eprintln!("zero-coordinate break child travel={travel:e}, expected=6e-4");
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-pose-precision-gpu"]
fn mutation_preserves_unedited_motion_across_compaction_contact_paths_and_com_changes() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    for initial_contact in [false, true] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let retired = scene
            .insert_body(BodyDesc {
                params: BodyParams::new().with_position(-Vec3::X * 100.),
                ..Default::default()
            })
            .unwrap();
        let mut survivors = Vec::new();
        for (kind, x) in [(BodyKind::Kinematic, 1024.), (BodyKind::Dynamic, 1025.)] {
            let id = scene
                .insert_body(BodyDesc {
                    kind,
                    params: BodyParams::new().with_position(Vec3::X * x),
                    velocity: SpatialVector::new(Vec3::X * 1e-3, Vec3::ZERO),
                    ..Default::default()
                })
                .unwrap();
            if kind == BodyKind::Kinematic {
                scene
                    .set_kinematic_acceleration(id, Some(SpatialVector::ZERO))
                    .unwrap();
            }
            survivors.push(id);
        }
        let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 });
        collider.parent = Some(survivors[0]);
        collider.mass = ColliderMass::Mass(0.0);
        let initial_collider =
            initial_contact.then(|| scene.insert_collider(collider.clone()).unwrap());
        let mut reference =
            RigidWorld::from_scene_gpu(scene.fork().unwrap(), engine.clone()).unwrap();
        let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        world.solver.substeps = 3;
        reference.solver.substeps = 3;
        let mut added = None;
        for frame in 0..60 {
            match frame {
                17 => {
                    world.scene.remove_body(retired).unwrap();
                    assert_eq!(world.scene.body_index(survivors[0]).unwrap().index(), 0);
                }
                23 => {
                    added = Some(
                        world
                            .scene
                            .insert_body(BodyDesc {
                                params: BodyParams::new().with_position(Vec3::new(1024., 10., 0.)),
                                ..Default::default()
                            })
                            .unwrap(),
                    );
                }
                29 => {
                    if let Some(id) = initial_collider {
                        world.scene.remove_collider(id).unwrap();
                    } else {
                        world.scene.insert_collider(collider.clone()).unwrap();
                    }
                }
                35 => world
                    .scene
                    .set_kinematic_acceleration(
                        survivors[0],
                        Some(SpatialVector::new(Vec3::Y * 1e-3, Vec3::ZERO)),
                    )
                    .unwrap(),
                41 | 47 => {
                    world
                        .scene
                        .set_body_pose(
                            added.unwrap(),
                            Transform::from_translation(Vec3::new(1024., frame as f64, 0.)),
                        )
                        .unwrap();
                    if frame == 41 {
                        world.set_reaction_readout(true).unwrap();
                    } else {
                        let velocity = world.scene.body_velocity(survivors[0]).unwrap();
                        let diagnostics = world.gpu_diagnostics().unwrap();
                        world
                            .scene
                            .set_body_velocity(
                                survivors[0],
                                SpatialVector::new(Vec3::X * 1e100, Vec3::ZERO),
                            )
                            .unwrap();
                        let error = world.step(0.01).unwrap_err();
                        eprintln!("edited-frame precision rejection: {error}");
                        assert_eq!(world.gpu_diagnostics().unwrap(), diagnostics);
                        assert_eq!(
                            world.scene.body_pose(added.unwrap()).unwrap().position.y,
                            47.
                        );
                        world
                            .scene
                            .set_body_velocity(survivors[0], velocity)
                            .unwrap();
                    }
                }
                53 => world
                    .scene
                    .set_body_mass_properties(
                        survivors[1],
                        MassProperties {
                            mass: 2.,
                            local_com: Vec3::Y * 0.25,
                            inertia: Mat3::IDENTITY,
                        },
                    )
                    .unwrap(),
                _ => {}
            }
            reference.step(0.01).unwrap();
            world.step(0.01).unwrap();
            for &body in &survivors {
                assert_eq!(
                    world.scene.body_pose(body).unwrap().position.x,
                    reference.scene.body_pose(body).unwrap().position.x,
                    "frame={frame}, initial_contact={initial_contact}, body={body:?}"
                );
            }
            if let Some(body) = added {
                assert_eq!(
                    world.scene.body_pose(body).unwrap().position.x,
                    1024.,
                    "new body inherited stale residue"
                );
            }
        }
        assert!(world.gpu_diagnostics().unwrap().model_compilations >= 7);
        for (index, body) in survivors.iter().enumerate() {
            let travel = world.scene.body_pose(*body).unwrap().position.x - (1024. + index as f64);
            assert!((travel - 6e-4).abs() <= 2_f64.powi(-14));
        }
        eprintln!(
            "mutation precision initial_contact={initial_contact}, compilations={}",
            world.gpu_diagnostics().unwrap().model_compilations
        );
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-pose-precision-gpu"]
fn replacing_scene_does_not_reuse_residues_from_equal_numeric_handles() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let make = |position| {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let body = scene
            .insert_body(BodyDesc {
                kind: BodyKind::Kinematic,
                params: BodyParams::new().with_position(position),
                velocity: SpatialVector::new(Vec3::X * 1e-3, Vec3::ZERO),
                ..Default::default()
            })
            .unwrap();
        scene
            .set_kinematic_acceleration(body, Some(SpatialVector::ZERO))
            .unwrap();
        (scene, body)
    };
    let (scene, old_body) = make(Vec3::X * 1024.);
    let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
    world.solver.substeps = 3;
    for _ in 0..17 {
        world.step(0.01).unwrap();
    }
    let (scene, body) = make(world.scene.body_pose(old_body).unwrap().position);
    assert_eq!(body.to_bits(), old_body.to_bits());
    assert_ne!(
        scene.body_identity_stamp(body),
        world.scene.body_identity_stamp(old_body)
    );
    let mut reference = RigidWorld::from_scene_gpu(scene.fork().unwrap(), engine).unwrap();
    reference.solver.substeps = 3;
    world.scene = scene;
    for frame in 0..30 {
        reference.step(0.01).unwrap();
        world.step(0.01).unwrap();
        assert_eq!(
            world.scene.body_pose(body).unwrap(),
            reference.scene.body_pose(body).unwrap(),
            "replacement frame={frame}"
        );
    }
}
