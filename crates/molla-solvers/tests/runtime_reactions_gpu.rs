#![cfg(feature = "gpu")]

#[path = "support/hardware_engine.rs"]
mod hardware_engine;
use molla_compute::Engine;
use std::sync::Arc;

use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor};
use molla_sim::{
    BodyParams, JointTargetMode, JointType, MassProperties,
    runtime::{BodyDesc, BodyHandle, DistanceLimits, JointDesc, RigidScene, SphericalConeLimits},
};
use molla_solvers::rigid_world::RigidWorld;

fn body(scene: &mut RigidScene, com: Vec3) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: com,
                inertia: Mat3::IDENTITY * 4.0,
            }),
            ..Default::default()
        })
        .unwrap()
}
fn joint(parent: Option<BodyHandle>, child: BodyHandle, kind: JointType) -> JointDesc {
    JointDesc {
        parent,
        child,
        kind,
        axis: Vec3::Z,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
    }
}
fn gpu(scene: RigidScene, engine: &Arc<Engine>) -> RigidWorld {
    let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
    world.set_reaction_readout(true).unwrap();
    world
}
fn near(actual: SpatialVector, expected: SpatialVector, tolerance: f64) {
    assert!(
        (actual.linear - expected.linear).length() < tolerance,
        "{actual:?} != {expected:?}"
    );
    assert!(
        (actual.angular - expected.angular).length() < tolerance,
        "{actual:?} != {expected:?}"
    );
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn fixed_support_publishes_substeps_means_peaks_and_force_edit_invalidation() {
    let engine = hardware_engine::engine();
    for substeps in [1, 4, 8] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(-Vec3::Y * 10.0).unwrap();
        let b = body(&mut scene, Vec3::X);
        let id = scene
            .insert_joint(joint(None, b, JointType::Fixed))
            .unwrap();
        scene
            .add_body_wrench(b, SpatialVector::new(Vec3::X * 6.0, Vec3::Z * 4.0))
            .unwrap();
        let mut world = gpu(scene, &engine);
        world.solver.substeps = substeps;
        world.step(0.04).unwrap();
        let reading = world.joint_reaction(id).unwrap();
        let expected = SpatialVector::new(Vec3::new(-6.0, 20.0, 0.0), Vec3::Z * 16.0);
        near(reading.child, expected, 1e-4);
        near(reading.parent, expected.scale(-1.0), 1e-4);
        assert!((reading.peak_force - expected.linear.length()).abs() < 1e-4);
        assert!((reading.peak_torque - 16.0).abs() < 1e-4);
        assert_eq!(reading.sample_dt, 0.04);
        let samples = world.joint_reaction_samples().unwrap();
        assert_eq!(samples.len(), substeps);
        for (i, sample) in samples.iter().enumerate() {
            assert_eq!(sample.substep, i);
            assert_eq!(sample.time_offset, (i + 1) as f64 * 0.04 / substeps as f64);
            near(sample.child, expected, 1e-4);
        }
        world.scene.reset_body_forces_with_wake(b, false).unwrap();
        assert!(world.joint_reaction(id).is_none());
        world.step(0.04).unwrap();
        near(
            world.joint_reaction(id).unwrap().child,
            SpatialVector::new(Vec3::Y * 20.0, Vec3::Z * 20.0),
            1e-4,
        );
        world.set_reaction_readout(false).unwrap();
        world.step(0.04).unwrap();
        assert!(world.joint_reaction_samples().is_none());
    }
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn stop_impulse_peak_is_not_the_frame_average_or_motor_command() {
    let engine = hardware_engine::engine();
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let b = body(&mut scene, Vec3::ZERO);
    let id = scene
        .insert_joint(joint(None, b, JointType::Revolute))
        .unwrap();
    scene
        .set_joint_limits(id, JointAxis::AngX, Some([-1.0, 0.0]))
        .unwrap();
    scene
        .set_body_velocity(b, SpatialVector::new(Vec3::ZERO, Vec3::Z * 2.0))
        .unwrap();
    let mut world = gpu(scene, &engine);
    world.solver.substeps = 4;
    world.step(0.04).unwrap();
    let reading = world.joint_reaction(id).unwrap();
    near(
        reading.child,
        SpatialVector::new(Vec3::ZERO, -Vec3::Z * 200.0),
        0.02,
    );
    assert!((reading.peak_torque - 800.0).abs() < 0.05, "{reading:?}");
    let samples = world.joint_reaction_samples().unwrap();
    assert_eq!(samples.len(), 4);
    assert!((samples[0].child.angular.z + 800.0).abs() < 0.05);
    for sample in &samples[1..] {
        assert!(sample.child.angular.length() < 0.05);
    }
    world
        .scene
        .set_joint_motor(
            id,
            JointAxis::AngX,
            JointMotor {
                mode: JointTargetMode::Effort,
                target_position: 100.0,
                max_force: 7.0,
                ..Default::default()
            },
        )
        .unwrap();
    world
        .scene
        .add_body_wrench(b, SpatialVector::new(Vec3::ZERO, -Vec3::Z * 5.0))
        .unwrap();
    world.step(0.04).unwrap();
    near(
        world.joint_reaction(id).unwrap().child,
        SpatialVector::new(Vec3::ZERO, Vec3::Z * 5.0),
        1e-4,
    );
    world
        .scene
        .set_joint_limits(id, JointAxis::AngX, None)
        .unwrap();
    world.step(0.04).unwrap();
    near(
        world.joint_reaction(id).unwrap().child,
        SpatialVector::new(Vec3::ZERO, Vec3::Z * 7.0),
        1e-4,
    );
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn moving_ball_and_d6_transmit_only_constrained_components() {
    let engine = hardware_engine::engine();
    for kind in [JointType::Ball, JointType::D6] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let b = body(&mut scene, Vec3::ZERO);
        let id = scene.insert_joint(joint(None, b, kind)).unwrap();
        let force = Vec3::new(3.0, -4.0, 5.0);
        scene
            .add_body_wrench(b, SpatialVector::new(force, Vec3::X * 6.0))
            .unwrap();
        let mut world = gpu(scene, &engine);
        for substeps in [1, 4, 8, 4] {
            world.solver.substeps = substeps;
            world.step(0.01).unwrap();
            let expected = SpatialVector::new(
                if kind == JointType::Ball {
                    -force
                } else {
                    Vec3::ZERO
                },
                Vec3::ZERO,
            );
            near(world.joint_reaction(id).unwrap().child, expected, 3e-4);
            let samples = world.joint_reaction_samples().unwrap();
            assert_eq!(samples.len(), substeps);
            for sample in samples {
                near(sample.child, expected, 3e-4);
            }
        }
        if kind == JointType::Ball {
            world.scene.set_body_pose(b, Transform::IDENTITY).unwrap();
            world
                .scene
                .set_body_velocity(b, SpatialVector::ZERO)
                .unwrap();
            world
                .scene
                .set_cone_limits(
                    id,
                    SphericalConeLimits {
                        angle0: Some(0.0),
                        angle1: Some(0.0),
                    },
                )
                .unwrap();
            world.step(0.01).unwrap();
            near(
                world.joint_reaction(id).unwrap().child,
                SpatialVector::new(-force, -Vec3::X * 6.0),
                3e-4,
            );
        }
    }
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn empty_scene_reaction_capture_can_be_toggled() {
    let engine = hardware_engine::engine();
    let mut world = gpu(RigidScene::new().unwrap(), &engine);
    for enabled in [true, false, true] {
        world.set_reaction_readout(enabled).unwrap();
        world.step(0.01).unwrap();
        assert_eq!(
            world.joint_reaction_samples().map(|samples| samples.len()),
            enabled.then_some(0)
        );
        assert_eq!(
            world.gpu_diagnostics().unwrap().last_state_readback_bytes,
            4
        );
    }
    world.solver.substeps = usize::MAX;
    assert!(world.step(0.01).is_err());
    assert!(world.joint_reaction_samples().is_none());
    world.solver.substeps = 1;
    world.step(0.01).unwrap();
    assert!(world.joint_reaction_samples().unwrap().is_empty());
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn moving_floating_chain_matches_cpu_substep_wrenches() {
    let engine = hardware_engine::engine();
    let fixture = || {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(-Vec3::Y * 9.81).unwrap();
        let rotation = Quat::from_rotation_x(0.3) * Quat::from_rotation_y(-0.5);
        let pose = Transform {
            position: Vec3::new(16.0, -8.0, 4.0),
            rotation,
        };
        let mut bodies = Vec::new();
        for i in 0..2 {
            bodies.push(
                scene
                    .insert_body(BodyDesc {
                        params: BodyParams {
                            pose,
                            ..Default::default()
                        },
                        additional_mass: Some(MassProperties {
                            mass: 2.0 + i as f64,
                            local_com: Vec3::new(0.2, -0.3, 0.1) * (i + 1) as f64,
                            inertia: Mat3::from_cols(
                                Vec3::new(2.0, 0.1, -0.2),
                                Vec3::new(0.1, 3.0, 0.3),
                                Vec3::new(-0.2, 0.3, 4.0),
                            ),
                        }),
                        ..Default::default()
                    })
                    .unwrap(),
            );
        }
        let mut desc = joint(Some(bodies[0]), bodies[1], JointType::Ball);
        desc.frame_parent.position = Vec3::new(0.4, 0.2, -0.1);
        desc.frame_child = desc.frame_parent;
        desc.frame_parent.rotation = Quat::from_rotation_z(0.4);
        desc.frame_child.rotation = desc.frame_parent.rotation;
        let id = scene.insert_joint(desc).unwrap();
        scene
            .set_body_velocity(
                bodies[0],
                SpatialVector::new(Vec3::new(1.0, -0.5, 0.2), Vec3::new(0.3, -0.4, 0.5)),
            )
            .unwrap();
        scene
            .set_body_velocity(
                bodies[1],
                SpatialVector::new(Vec3::new(0.6, -0.2, 0.1), Vec3::new(-0.4, 0.6, 0.2)),
            )
            .unwrap();
        scene
            .add_body_wrench(
                bodies[1],
                SpatialVector::new(Vec3::new(3.0, -4.0, 5.0), Vec3::new(0.4, 1.2, -0.7)),
            )
            .unwrap();
        (scene, id)
    };
    let (scene, cpu_id) = fixture();
    let mut cpu = RigidWorld::from_scene(scene).unwrap();
    cpu.set_reaction_readout(true).unwrap();
    let (scene, gpu_id) = fixture();
    let mut device = gpu(scene, &engine);
    cpu.solver.substeps = 4;
    device.solver.substeps = 4;
    let mut peak = 0.0_f64;
    for _ in 0..20 {
        cpu.step(0.002).unwrap();
        device.step(0.002).unwrap();
        near(
            device.joint_reaction(gpu_id).unwrap().child,
            cpu.joint_reaction(cpu_id).unwrap().child,
            3e-3,
        );
        for (a, b) in cpu
            .joint_reaction_samples()
            .unwrap()
            .iter()
            .zip(device.joint_reaction_samples().unwrap())
        {
            peak = peak
                .max((a.child.linear - b.child.linear).length())
                .max((a.child.angular - b.child.angular).length());
            near(b.child, a.child, 3e-3);
            near(b.parent, a.parent, 3e-3);
            assert!((a.anchors[0] - b.anchors[0]).length() < 1e-4);
            assert!((a.anchors[1] - b.anchors[1]).length() < 1e-4);
        }
    }
    eprintln!("moving floating GPU reaction peak absolute wrench error={peak:e}");
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn external_loop_force_is_not_double_counted_as_tree_load() {
    let engine = hardware_engine::engine();
    for (kind, world_anchor) in [
        (JointType::Distance, true),
        (JointType::Ball, true),
        (JointType::Distance, false),
        (JointType::Ball, false),
    ] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let a = body(&mut scene, Vec3::ZERO);
        let b = body(&mut scene, Vec3::ZERO);
        let tree = scene
            .insert_joint(joint(Some(a), b, JointType::Fixed))
            .unwrap();
        let support = if world_anchor {
            None
        } else {
            Some(body(&mut scene, Vec3::ZERO))
        };
        let closure = if kind == JointType::Ball {
            scene
                .insert_rigid_loop_joint(joint(support, b, kind))
                .unwrap()
        } else {
            scene.insert_joint(joint(support, b, kind)).unwrap()
        };
        if kind == JointType::Distance {
            scene
                .set_distance_limits(
                    closure,
                    DistanceLimits {
                        min: None,
                        max: Some(0.0),
                    },
                )
                .unwrap();
        } else {
            scene
                .set_cone_limits(
                    closure,
                    SphericalConeLimits {
                        angle0: Some(0.0),
                        angle1: Some(0.0),
                    },
                )
                .unwrap();
        }
        let torque = if kind == JointType::Ball {
            Vec3::Y * 4.0
        } else {
            Vec3::ZERO
        };
        scene
            .add_body_wrench(b, SpatialVector::new(Vec3::X * 8.0, torque))
            .unwrap();
        let mut world = gpu(scene, &engine);
        world.step(0.01).unwrap();
        let load = SpatialVector::new(Vec3::X * 8.0, torque);
        let loop_load = load.scale(if world_anchor { -1.0 } else { -1.0 / 3.0 });
        let tree_load = load.scale(if world_anchor { 0.0 } else { -1.0 / 3.0 });
        near(
            world.joint_reaction(closure).unwrap().child,
            loop_load,
            2e-3,
        );
        near(
            world.joint_reaction(closure).unwrap().parent,
            loop_load.scale(-1.0),
            2e-3,
        );
        near(world.joint_reaction(tree).unwrap().child, tree_load, 2e-3);
        assert_eq!(world.joint_reaction_samples().unwrap().len(), 16);
        #[cfg(feature = "resource-counters")]
        {
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
            let baseline = counters();
            let diagnostics = world.gpu_diagnostics().unwrap();
            for _ in 0..10 {
                world.step(0.01).unwrap();
                assert_eq!(counters(), baseline);
                assert_eq!(
                    world.gpu_diagnostics().unwrap().model_compilations,
                    diagnostics.model_compilations
                );
                near(world.joint_reaction(tree).unwrap().child, tree_load, 2e-3);
            }
            eprintln!("{kind:?} world={world_anchor} loop GPU reaction resources={baseline:?}");
        }
    }
}

#[test]
#[ignore = "requires GPU through make test-gpu"]
fn sleeping_compaction_and_reaction_faults_do_not_publish_stale_samples() {
    let engine = hardware_engine::engine();
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let removed = body(&mut scene, Vec3::ZERO);
    let b = body(&mut scene, Vec3::ZERO);
    let id = scene.insert_joint(joint(None, b, JointType::Ball)).unwrap();
    let mut world = gpu(scene, &engine);
    world.step(0.01).unwrap();
    world.sleep_body(b).unwrap();
    world.step(0.01).unwrap();
    assert!(world.joint_reaction(id).is_none());
    assert!(world.joint_reaction_samples().unwrap().is_empty());
    world.wake_body(b).unwrap();
    world.step(0.01).unwrap();
    assert!(world.joint_reaction(id).is_some());
    world.scene.remove_body(removed).unwrap();
    assert!(world.joint_reaction(id).is_none());
    world.step(0.01).unwrap();
    assert!(world.joint_reaction(id).is_some());
    for scale in [1e-30, 1e30, 1e38] {
        world.scene.reset_body_forces(b).unwrap();
        world
            .scene
            .add_body_wrench(b, SpatialVector::new(Vec3::splat(scale), Vec3::ZERO))
            .unwrap();
        world.step(0.01).unwrap();
        let reading = world.joint_reaction(id).unwrap();
        assert!(
            (reading.peak_force / (3.0_f64.sqrt() * scale) - 1.0).abs() < 2e-6,
            "scale={scale:e} reading={reading:?}"
        );
    }
    world.scene.reset_body_forces(b).unwrap();
    world
        .scene
        .add_body_wrench(b, SpatialVector::new(Vec3::splat(2e38), Vec3::ZERO))
        .unwrap();
    let pose = world.scene.body_pose(b).unwrap();
    let velocity = world.scene.body_velocity(b).unwrap();
    let result = world.step(0.01);
    assert!(
        result.is_err(),
        "overflow reaction was published: {:?}",
        world.joint_reaction(id)
    );
    let error = result.unwrap_err();
    assert!(error.to_string().contains("16384"), "{error}");
    assert_eq!(world.scene.body_pose(b).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(b).unwrap(), velocity);
    assert!(world.joint_reaction_samples().is_none());
    world.scene.reset_body_forces(b).unwrap();
    world.step(0.01).unwrap();
    assert!(world.joint_reaction(id).is_some());
    world.scene.remove_joint(id).unwrap();
    assert!(world.joint_reaction(id).is_none());
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "requires GPU resource counters through make test-gpu"]
fn runtime_reaction_resources_and_frame_transfers_are_retained() {
    let engine = hardware_engine::engine();
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let b = body(&mut scene, Vec3::ZERO);
    scene
        .insert_joint(joint(None, b, JointType::Fixed))
        .unwrap();
    let mut world = gpu(scene, &engine);
    world.step(0.01).unwrap();
    let counters = || {
        let d = engine.device().unwrap();
        d.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let c = d.get_internal_counters().hal;
        [
            c.buffers.read(),
            c.buffer_memory.read(),
            c.bind_groups.read(),
            c.compute_pipelines.read(),
            c.shader_modules.read(),
        ]
    };
    let baseline = counters();
    let diagnostics = world.gpu_diagnostics().unwrap();
    for _ in 0..10 {
        world.step(0.01).unwrap();
        assert_eq!(counters(), baseline);
        let now = world.gpu_diagnostics().unwrap();
        assert_eq!(now.model_compilations, diagnostics.model_compilations);
        assert_eq!(
            now.current_model_bind_groups_created,
            diagnostics.current_model_bind_groups_created
        );
        assert_eq!(
            now.last_state_readback_bytes,
            diagnostics.last_state_readback_bytes
        );
    }
    assert_eq!(
        diagnostics.last_state_readback_bytes,
        4 + 64
            + 80
            + 8 * 112
            + world.scene.model().body_count as u64 * 16
            + world.scene.model().joint_coord_total as u64 * 4
    );
    eprintln!("runtime GPU reaction resources={baseline:?}, diagnostics={diagnostics:?}");
}
