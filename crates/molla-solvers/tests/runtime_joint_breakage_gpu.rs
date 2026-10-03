#![cfg(feature = "gpu")]

#[path = "support/hardware_engine.rs"]
mod hardware_engine;

use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis};
use molla_sim::{
    JointType, MassProperties,
    runtime::{
        BodyDesc, BodyHandle, JointBreakLimits, JointBreakTrigger, JointDesc, JointHandle,
        RigidScene,
    },
};
use molla_solvers::rigid_world::RigidWorld;

fn fixed(scene: &mut RigidScene) -> (BodyHandle, JointHandle) {
    let child = scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * 4.0,
            }),
            ..Default::default()
        })
        .unwrap();
    let joint = scene
        .insert_joint(JointDesc {
            parent: None,
            child,
            kind: JointType::Fixed,
            axis: Vec3::Z,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    (child, joint)
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gravity_break_preserves_first_boundary_across_queued_substeps_and_continues_free_motion() {
    let engine = hardware_engine::engine();
    for n in [1, 4, 8] {
        for capture in [false, true] {
            let mut scene = RigidScene::new().unwrap();
            scene.set_gravity(-Vec3::Y * 10.0).unwrap();
            let (body, joint) = fixed(&mut scene);
            scene
                .set_joint_break_limits(
                    joint,
                    JointBreakLimits {
                        force: Some(20.0),
                        torque: None,
                    },
                )
                .unwrap();
            let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
            world.solver.substeps = n;
            world.set_reaction_readout(capture).unwrap();
            let dt = 0.04;
            world.step(dt).unwrap();
            assert!(world.scene.joint_broken(joint).unwrap());
            let event = &world.joint_break_events()[0];
            assert_eq!(event.frame, 0);
            assert_eq!(event.trigger, JointBreakTrigger::Force);
            assert_eq!(event.sample.substep, 0);
            assert!((event.sample.child.linear.y - 20.0).abs() < 1e-5);
            assert!((event.sample.sample_dt - dt / n as f64).abs() < 1e-12);
            let free = (n - 1) as f64;
            let h = dt / n as f64;
            assert!(
                (world.scene.body_velocity(body).unwrap().linear.y + free * h * 10.0).abs() < 1e-6
            );
            assert!(
                (world.scene.body_pose(body).unwrap().position.y
                    + free * (free + 1.0) * 0.5 * h * h * 10.0)
                    .abs()
                    < 1e-6
            );
            assert_eq!(world.gpu_diagnostics().unwrap().completed_frames, 1);
            assert_eq!(world.gpu_diagnostics().unwrap().last_substeps, n);
            if capture {
                assert!(
                    (world.joint_reaction(joint).unwrap().child.linear.y - 20.0 / n as f64).abs()
                        < 1e-5
                );
                assert_eq!(world.joint_reaction_samples().unwrap().len(), 1);
            } else {
                assert!(world.joint_reaction_samples().is_none());
            }
            world.step(dt).unwrap();
            assert_eq!(world.joint_break_events().len(), 1);
            assert_eq!(world.completed_steps(), 2);
            assert_eq!(world.drain_joint_break_events().len(), 1);
            assert!(world.joint_break_events().is_empty());
        }
    }
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn simultaneous_gpu_channels_are_collected_before_halt_and_match_cpu() {
    let engine = hardware_engine::engine();
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let cases = [
        (Some(4.0), None),
        (None, Some(8.0)),
        (Some(4.0), Some(8.0)),
        (Some(5.0), Some(9.0)),
        (Some(f64::INFINITY), Some(f64::INFINITY)),
        (None, None),
    ];
    let mut ids = Vec::new();
    for (force, torque) in cases {
        let (body, joint) = fixed(&mut scene);
        scene
            .add_body_wrench(body, SpatialVector::new(Vec3::X * 4.0, Vec3::Z * 8.0))
            .unwrap();
        scene
            .set_joint_break_limits(joint, JointBreakLimits { force, torque })
            .unwrap();
        ids.push(joint);
    }
    let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    cpu.solver.substeps = 4;
    gpu.solver.substeps = 4;
    cpu.step(0.04).unwrap();
    gpu.step(0.04).unwrap();
    let describe = |world: &RigidWorld| {
        world
            .joint_break_events()
            .iter()
            .map(|event| (event.joint, event.trigger, event.sample.substep))
            .collect::<Vec<_>>()
    };
    assert_eq!(describe(&cpu), describe(&gpu));
    assert_eq!(gpu.joint_break_events().len(), 3);
    for (index, joint) in ids.iter().enumerate() {
        assert_eq!(gpu.scene.joint_broken(*joint).unwrap(), index < 3);
    }
    gpu.scene.repair_joint(ids[0]).unwrap();
    gpu.step(0.04).unwrap();
    assert_eq!(gpu.joint_break_events().len(), 4);
    assert_eq!(gpu.joint_break_events()[3].frame, 1);
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn subthreshold_and_sleeping_frames_keep_compact_transfer_and_zero_waits_for_a_sample() {
    let engine = hardware_engine::engine();
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let (body, joint) = fixed(&mut scene);
    scene
        .set_joint_break_limits(
            joint,
            JointBreakLimits {
                force: Some(1.0),
                torque: None,
            },
        )
        .unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
    world.solver.substeps = 8;
    world.step(0.04).unwrap();
    let compilations = world.gpu_diagnostics().unwrap().model_compilations;
    let stamp = world.scene.model_stamp();
    #[cfg(feature = "resource-counters")]
    let counters = || {
        let device = engine.device().unwrap();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let c = device.get_internal_counters().hal;
        [
            c.buffers.read(),
            c.buffer_memory.read(),
            c.bind_groups.read(),
            c.compute_pipelines.read(),
            c.shader_modules.read(),
        ]
    };
    #[cfg(feature = "resource-counters")]
    let baseline = counters();
    for _ in 0..3 {
        world.step(0.04).unwrap();
        assert_eq!(world.scene.model_stamp(), stamp);
        let d = world.gpu_diagnostics().unwrap();
        assert_eq!(d.model_compilations, compilations);
        assert_eq!(
            d.last_state_readback_bytes,
            12 + 4
                + 64
                + world.scene.model().body_count as u64 * 16
                + world.scene.model().joint_coord_total as u64 * 4
        );
        assert_eq!(d.last_state_upload_bytes, 96);
        assert!(world.joint_break_events().is_empty());
        #[cfg(feature = "resource-counters")]
        assert_eq!(counters(), baseline);
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
    world.sleep_body(body).unwrap();
    world.step(0.04).unwrap();
    assert!(!world.scene.joint_broken(joint).unwrap());
    world.wake_body(body).unwrap();
    world.step(0.04).unwrap();
    assert!(world.scene.joint_broken(joint).unwrap());
    assert_eq!(world.joint_break_events().len(), 1);
    assert_eq!(
        world.joint_break_events()[0].sample.child,
        SpatialVector::ZERO
    );
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn later_gpu_failure_rolls_back_the_original_model_and_discards_provisional_events() {
    let engine = hardware_engine::engine();
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(-Vec3::Y * 10.0).unwrap();
    let (body, joint) = fixed(&mut scene);
    scene
        .set_joint_break_limits(
            joint,
            JointBreakLimits {
                force: Some(1.0),
                torque: None,
            },
        )
        .unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    world.solver.substeps = 4;
    let model = world.scene.model_stamp();
    let dynamics = world.scene.dynamics_stamp();
    let activation = world.scene.activation_stamp(body).unwrap();
    let pose = world.scene.body_pose(body).unwrap();
    let compiled = world.gpu_diagnostics().unwrap().model_compilations;
    let error = world.step(1e20).unwrap_err();
    eprintln!("post-break overflow: {error}");
    assert!(world.gpu_diagnostics().unwrap().model_compilations >= compiled + 2);
    assert_eq!(world.scene.model_stamp(), model);
    assert_eq!(world.scene.dynamics_stamp(), dynamics);
    assert_eq!(world.scene.activation_stamp(body).unwrap(), activation);
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(
        world.scene.body_velocity(body).unwrap(),
        SpatialVector::ZERO
    );
    assert!(!world.scene.joint_broken(joint).unwrap());
    assert!(world.joint_break_events().is_empty());
    assert_eq!(world.completed_steps(), 0);
    assert_eq!(world.gpu_diagnostics().unwrap().completed_frames, 0);
    world.step(0.04).unwrap();
    assert_eq!(world.joint_break_events().len(), 1);
    assert_eq!(world.joint_break_events()[0].frame, 0);
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn simultaneous_breaks_cross_workgroups_without_losing_triggers() {
    let engine = hardware_engine::engine();
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut joints = Vec::new();
    for _ in 0..130 {
        let (_, joint) = fixed(&mut scene);
        scene
            .set_joint_break_limits(
                joint,
                JointBreakLimits {
                    force: Some(0.0),
                    torque: Some(0.0),
                },
            )
            .unwrap();
        joints.push(joint);
    }
    let (unrelated, _) = fixed(&mut scene);
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    world.sleep_body(unrelated).unwrap();
    world.solver.substeps = 4;
    world.step(0.04).unwrap();
    assert_eq!(world.joint_break_events().len(), joints.len());
    assert!(world.is_sleeping(unrelated));
    for (&id, event) in joints.iter().zip(world.joint_break_events()) {
        assert_eq!(event.joint, id);
        assert_eq!(event.trigger, JointBreakTrigger::Both);
        assert_eq!(event.sample.substep, 0);
        assert!(world.scene.joint_broken(id).unwrap());
    }
    world.scene.remove_body(unrelated).unwrap();
    world.step(0.04).unwrap();
    assert_eq!(world.joint_break_events().len(), joints.len());
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn delayed_stop_peaks_produce_sequential_break_boundaries_not_frame_mean_decisions() {
    let engine = hardware_engine::engine();
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut ids = Vec::new();
    for upper in [0.025, 0.045] {
        let (body, old) = fixed(&mut scene);
        scene.remove_joint(old).unwrap();
        let joint = scene
            .insert_joint(JointDesc {
                parent: None,
                child: body,
                kind: JointType::Revolute,
                axis: Vec3::Z,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        scene
            .set_joint_limits(joint, JointAxis::AngX, Some([-1.0, upper]))
            .unwrap();
        scene
            .set_body_velocity(body, SpatialVector::new(Vec3::ZERO, Vec3::Z * 2.0))
            .unwrap();
        scene
            .set_joint_break_limits(
                joint,
                JointBreakLimits {
                    force: None,
                    torque: Some(400.0),
                },
            )
            .unwrap();
        ids.push((body, joint));
    }
    let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
    let mut gpu = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    for world in [&mut cpu, &mut gpu] {
        world.solver.substeps = 4;
        world.set_reaction_readout(true).unwrap();
        world.step(0.04).unwrap();
        assert_eq!(
            world
                .joint_break_events()
                .iter()
                .map(|e| e.sample.substep)
                .collect::<Vec<_>>(),
            vec![1, 2]
        );
        for &(_, joint) in &ids {
            let mean = world.joint_reaction(joint).unwrap();
            assert!(mean.child.angular.length() < 400.0);
            assert!(mean.peak_torque > 590.0);
        }
    }
    for &(body, _) in &ids {
        assert!(
            (cpu.scene.body_velocity(body).unwrap().angular
                - gpu.scene.body_velocity(body).unwrap().angular)
                .length()
                < 1e-4
        );
        let a = cpu.scene.body_pose(body).unwrap();
        let b = gpu.scene.body_pose(body).unwrap();
        assert!((a.rotation.dot(b.rotation).abs() - 1.0).abs() < 1e-5);
    }
    assert_eq!(gpu.gpu_diagnostics().unwrap().completed_frames, 1);
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn tree_and_loop_joint_kinds_detach_at_the_same_support_load() {
    let engine = hardware_engine::engine();
    for (kind, external) in [
        (JointType::Fixed, false),
        (JointType::Revolute, false),
        (JointType::Prismatic, false),
        (JointType::Ball, false),
        (JointType::D6, false),
        (JointType::Distance, true),
        (JointType::Ball, true),
    ] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(-Vec3::Y * 10.0).unwrap();
        let (body, old) = fixed(&mut scene);
        scene.remove_joint(old).unwrap();
        let desc = JointDesc {
            parent: None,
            child: body,
            kind,
            axis: Vec3::Z,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        };
        let joint = if kind == JointType::Ball && external {
            scene.insert_rigid_loop_joint(desc).unwrap()
        } else {
            scene.insert_joint(desc).unwrap()
        };
        if kind == JointType::D6 {
            scene
                .set_joint_locked_axes(joint, JointAxes::NONE.with(JointAxis::LinY))
                .unwrap();
        }
        if kind == JointType::Distance {
            scene
                .set_distance_limits(
                    joint,
                    molla_sim::runtime::DistanceLimits {
                        min: None,
                        max: Some(0.0),
                    },
                )
                .unwrap();
        }
        scene
            .set_joint_break_limits(
                joint,
                JointBreakLimits {
                    force: Some(19.0),
                    torque: None,
                },
            )
            .unwrap();
        let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        world.solver.substeps = 4;
        world.step(0.04).unwrap();
        assert!(
            world.scene.joint_broken(joint).unwrap(),
            "{kind:?} external={external}"
        );
        assert_eq!(world.joint_break_events().len(), 1);
        assert!((world.joint_break_events()[0].sample.child.linear.y - 20.0).abs() < 1e-3);
        assert!((world.scene.body_velocity(body).unwrap().linear.y + 0.3).abs() < 1e-4);
    }
}
