#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor};
use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;
use std::sync::Arc;

#[test]
#[ignore = "requires GPU; run via make test-primitives-multidof-feedback-gpu"]
fn multidof_drives_use_nonzero_coordinates_and_match_cpu_with_stable_feedback() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    for kind in [JointType::D6, JointType::Free, JointType::Ball] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let parent = Transform::new(Vec3::new(0.7, 1.2, -0.4), Quat::from_rotation_y(0.2));
        let child = Transform::new(Vec3::new(0.1, -0.03, 0.07), Quat::from_rotation_x(0.1));
        let translation = if kind == JointType::Ball {
            Vec3::ZERO
        } else {
            Vec3::new(0.2, -0.1, 0.3)
        };
        let angles = Vec3::new(0.25, -0.35, 0.2);
        let rotation = Quat::from_rotation_x(angles.x)
            * Quat::from_rotation_y(angles.y)
            * Quat::from_rotation_z(angles.z);
        let pose = parent
            .compose(&Transform::new(translation, rotation))
            .compose(&child.inverse());
        let body = scene
            .insert_body(BodyDesc {
                params: BodyParams {
                    pose,
                    ..Default::default()
                },
                additional_mass: Some(MassProperties {
                    mass: 2.,
                    local_com: Vec3::new(0.15, -0.1, 0.05),
                    inertia: Mat3::from_cols(
                        Vec3::new(1.2, 0.1, 0.03),
                        Vec3::new(0.1, 1.5, 0.07),
                        Vec3::new(0.03, 0.07, 1.8),
                    ),
                }),
                ..Default::default()
            })
            .unwrap();
        let joint = scene
            .insert_joint(JointDesc {
                parent: None,
                child: body,
                kind,
                axis: Vec3::X,
                frame_parent: parent,
                frame_child: child,
            })
            .unwrap();
        let axes: Vec<_> = [
            JointAxis::LinX,
            JointAxis::LinY,
            JointAxis::LinZ,
            JointAxis::AngX,
            JointAxis::AngY,
            JointAxis::AngZ,
        ]
        .into_iter()
        .enumerate()
        .filter(|(i, _)| kind != JointType::Ball || *i >= 3)
        .collect();
        let coordinates = [
            translation.x,
            translation.y,
            translation.z,
            angles.x,
            angles.y,
            angles.z,
        ];
        let mut cpu = RigidWorld::from_scene(scene.fork().unwrap()).unwrap();
        let mut gpu = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        for world in [&mut cpu, &mut gpu] {
            world.solver.substeps = 2;
            world.set_reaction_readout(true).unwrap();
        }
        let mut compilations = None;
        for stable in [false, true] {
            for phase in 0..4 {
                let mode = [
                    JointTargetMode::Position,
                    JointTargetMode::Position,
                    JointTargetMode::Velocity,
                    JointTargetMode::PositionVelocity,
                ][phase];
                for world in [&mut cpu, &mut gpu] {
                    world.solver.stable_motor_feedback = stable;
                    world.scene.set_body_pose(body, pose).unwrap();
                    world
                        .scene
                        .set_body_velocity(body, SpatialVector::ZERO)
                        .unwrap();
                    for &(i, axis) in &axes {
                        world
                            .scene
                            .set_joint_motor(
                                joint,
                                axis,
                                JointMotor {
                                    mode,
                                    target_position: coordinates[i]
                                        + if phase == 0 {
                                            0.
                                        } else {
                                            0.1 * if i % 2 == 0 { 1. } else { -1. }
                                        },
                                    target_velocity: if phase < 2 {
                                        0.
                                    } else {
                                        0.07 * if i % 2 == 0 { 1. } else { -1. }
                                    },
                                    stiffness: 30.,
                                    damping: 5.,
                                    max_force: 2.,
                                    ..Default::default()
                                },
                            )
                            .unwrap();
                    }
                }
                let mut peak_pose: f64 = 0.;
                let mut peak_velocity: f64 = 0.;
                let mut peak_reaction: f64 = 0.;
                for step in 0..if phase == 0 { 1 } else { 240 } {
                    cpu.step(1. / 240.).unwrap();
                    gpu.step(1. / 240.).unwrap_or_else(|e| {
                        panic!("kind={kind:?} stable={stable} phase={phase} step={step}: {e}")
                    });
                    let a = cpu.scene.body_pose(body).unwrap();
                    let b = gpu.scene.body_pose(body).unwrap();
                    let va = cpu.scene.body_velocity(body).unwrap();
                    let vb = gpu.scene.body_velocity(body).unwrap();
                    let compiled = gpu.gpu_diagnostics().unwrap().model_compilations;
                    assert_eq!(*compilations.get_or_insert(compiled), compiled);
                    let ra = cpu.joint_reaction(joint).unwrap().child;
                    let rb = gpu.joint_reaction(joint).unwrap().child;
                    peak_pose = peak_pose.max(
                        (a.position - b.position)
                            .length()
                            .max(a.rotation.angle_between(b.rotation).abs()),
                    );
                    peak_velocity = peak_velocity
                        .max((va.linear - vb.linear).length() + (va.angular - vb.angular).length());
                    peak_reaction = peak_reaction
                        .max((ra.linear - rb.linear).length() + (ra.angular - rb.angular).length());
                    if phase == 0 {
                        assert!(
                            va.linear.length() + va.angular.length() < 1e-10,
                            "CPU nonzero-coordinate equilibrium: {va:?}"
                        );
                        assert!(
                            vb.linear.length() + vb.angular.length() < 1e-5,
                            "kind={kind:?} stable={stable} GPU nonzero-coordinate equilibrium: {vb:?}"
                        );
                    }
                    assert!(
                        (a.position - b.position).length() < 1e-3
                            && a.rotation.angle_between(b.rotation).abs() < 1e-3,
                        "kind={kind:?} stable={stable} phase={phase} step={step}: CPU={a:?} GPU={b:?}"
                    );
                    assert!(
                        (va.linear - vb.linear).length() + (va.angular - vb.angular).length()
                            < 1e-3,
                        "kind={kind:?} stable={stable} phase={phase} step={step}: CPU={va:?} GPU={vb:?}"
                    );
                    assert!(
                        peak_reaction < 0.02,
                        "kind={kind:?} stable={stable} phase={phase} step={step}: CPU reaction={ra:?} GPU reaction={rb:?}"
                    );
                }
                eprintln!(
                    "multidof kind={kind:?} stable={stable} phase={phase} peak_pose={peak_pose:e} peak_velocity={peak_velocity:e} peak_reaction={peak_reaction:e}"
                );
            }
        }
    }
}
