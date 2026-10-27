use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointSoftness};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, JointBreakLimits, JointBreakTrigger, JointDesc, JointHandle,
    JointSettings, RigidScene,
};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

#[derive(Debug)]
struct Case {
    parent: Option<BodyHandle>,
    child: BodyHandle,
    joint: JointHandle,
    kind: JointType,
    origin: Vec3,
    direction: Vec3,
    rotation: Quat,
    mass: [f64; 2],
    position: [f64; 2],
    velocity: [f64; 2],
    offset: f64,
    softness: JointSoftness,
    enabled: bool,
}

fn body(scene: &mut RigidScene, pose: Transform, mass: f64, velocity: Vec3) -> BodyHandle {
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(pose),
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * mass,
            }),
            ..Default::default()
        })
        .unwrap();
    scene
        .set_body_velocity(body, SpatialVector::new(velocity, Vec3::ZERO))
        .unwrap();
    body
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    for kind in [
        JointType::Fixed,
        JointType::Revolute,
        JointType::Prismatic,
        JointType::Ball,
        JointType::D6,
    ] {
        for mass in [0.01, 1.0, 100.0] {
            for floating in [false, true] {
                for rotated in [false, true] {
                    let rotation = if rotated {
                        Quat::from_rotation_z(0.7) * Quat::from_rotation_x(-0.4)
                    } else {
                        Quat::IDENTITY
                    };
                    let direction = rotation * Vec3::Y;
                    let origin = Vec3::new(8.0 * cases.len() as f64, 2.0, -3.0);
                    let parent = floating.then(|| {
                        body(
                            &mut scene,
                            Transform::new(origin, rotation),
                            mass,
                            direction * -0.03,
                        )
                    });
                    let child = body(
                        &mut scene,
                        Transform::new(origin + direction * 0.02, rotation),
                        3.0,
                        direction * 0.07,
                    );
                    let softness = JointSoftness {
                        natural_frequency: 4.0,
                        damping_ratio: 0.6,
                    };
                    let joint = scene
                        .insert_loop_joint(
                            JointDesc {
                                parent,
                                child,
                                kind,
                                axis: Vec3::X,
                                frame_parent: if floating {
                                    Transform::IDENTITY
                                } else {
                                    Transform::new(origin, rotation)
                                },
                                frame_child: Transform::IDENTITY,
                            },
                            softness,
                        )
                        .unwrap();
                    if kind == JointType::D6 {
                        scene
                            .set_joint_locked_axes(joint, JointAxes::NONE.with(JointAxis::LinY))
                            .unwrap();
                    }
                    scene
                        .add_body_wrench(
                            child,
                            SpatialVector::new(direction * (3.0 * 0.3), Vec3::ZERO),
                        )
                        .unwrap();
                    if let Some(parent) = parent {
                        scene
                            .add_body_wrench(
                                parent,
                                SpatialVector::new(direction * (mass * -0.2), Vec3::ZERO),
                            )
                            .unwrap();
                    }
                    cases.push(Case {
                        parent,
                        child,
                        joint,
                        kind,
                        origin,
                        direction,
                        rotation,
                        mass: [mass, 3.0],
                        position: [0.0, 0.02],
                        velocity: [if floating { -0.03 } else { 0.0 }, 0.07],
                        offset: 0.0,
                        softness,
                        enabled: true,
                    });
                }
            }
        }
    }
    (scene, cases)
}

fn advance(case: &mut Case, dt: f64) -> f64 {
    let inverse_mass = if case.parent.is_some() {
        1.0 / case.mass[0]
    } else {
        0.0
    } + 1.0 / case.mass[1];
    let reduced_mass = 1.0 / inverse_mass;
    let omega = std::f64::consts::TAU * case.softness.natural_frequency;
    let stiffness = reduced_mass * omega * omega;
    let damping = reduced_mass * 2.0 * case.softness.damping_ratio * omega;
    let error = case.position[1] - case.position[0] + case.offset;
    let rate = case.velocity[1] - case.velocity[0];
    let impulse = if case.enabled {
        (-rate - stiffness * error / (damping + dt * stiffness))
            / (inverse_mass + 1.0 / (dt * (damping + dt * stiffness)))
    } else {
        0.0
    };
    if case.parent.is_some() {
        case.velocity[0] += -impulse / case.mass[0] - 0.2 * dt;
        case.position[0] += case.velocity[0] * dt;
    }
    case.velocity[1] += impulse / case.mass[1] + 0.3 * dt;
    case.position[1] += case.velocity[1] * dt;
    impulse / dt
}

fn certify(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for hz in [60, 120, 240] {
        let (scene, mut cases) = fixture();
        let mut world = make_world(scene);
        world.solver.substeps = 1;
        world.set_reaction_readout(true).unwrap();
        let dt = 1.0 / hz as f64;
        for frame in 0..20 {
            for case in &mut cases {
                if frame == 5 {
                    case.softness = JointSoftness {
                        natural_frequency: 7.0,
                        damping_ratio: 1.2,
                    };
                    world
                        .scene
                        .set_joint_softness(case.joint, Some(case.softness))
                        .unwrap();
                }
                if frame == 8 {
                    case.offset = 0.004;
                    let desc = world.scene.joint(case.joint).unwrap().clone();
                    world
                        .scene
                        .set_joint_frames(
                            case.joint,
                            desc.frame_parent,
                            Transform::from_translation(Vec3::Y * case.offset),
                        )
                        .unwrap();
                }
                if frame == 12 || frame == 14 {
                    case.enabled = frame == 14;
                    world
                        .scene
                        .set_joint_settings(
                            case.joint,
                            JointSettings {
                                enabled: case.enabled,
                                contacts_enabled: false,
                            },
                        )
                        .unwrap();
                }
            }
            let expected: Vec<_> = cases.iter_mut().map(|case| advance(case, dt)).collect();
            world
                .step(dt)
                .unwrap_or_else(|error| panic!("hz={hz} frame={frame}: {error}"));
            assert!(world.quarantined_bodies().is_empty());
            for (case, force) in cases.iter().zip(expected) {
                for (index, body) in [case.parent, Some(case.child)].into_iter().enumerate() {
                    let Some(body) = body else { continue };
                    let pose = world.scene.body_pose(body).unwrap();
                    let velocity = world.scene.body_velocity(body).unwrap();
                    let position_error =
                        (pose.position - case.origin - case.direction * case.position[index])
                            .length();
                    let velocity_error =
                        (velocity.linear - case.direction * case.velocity[index]).length();
                    assert!(
                        position_error < tolerance && velocity_error < tolerance,
                        "hz={hz} frame={frame} end={index} kind={:?} position={position_error:e} velocity={velocity_error:e} {case:?}",
                        case.kind
                    );
                    assert!(velocity.angular.length() < tolerance);
                    assert!(
                        (case.rotation.conjugate() * pose.rotation)
                            .to_scaled_axis()
                            .length()
                            < tolerance
                    );
                }
                if case.enabled {
                    let reading = world.joint_reaction(case.joint).unwrap();
                    assert!(
                        (reading.child.linear - case.direction * force).length()
                            < tolerance * (1.0 + force.abs()),
                        "hz={hz} frame={frame} expected={force:e} reading={reading:?} {case:?}"
                    );
                    assert!((reading.parent.linear + reading.child.linear).length() < tolerance);
                    assert!(reading.child.angular.length() < tolerance);
                    assert!(reading.parent.angular.length() < tolerance);
                    assert!((reading.sample_dt - dt).abs() < 1e-12);
                } else {
                    assert!(world.joint_reaction(case.joint).is_none());
                }
            }
        }
        assert_eq!(cases.len(), 60);
        eprintln!(
            "compliant loop scalar oracle: hz={hz}, cases={}",
            cases.len()
        );
    }
}

#[test]
fn cpu_general_compliant_loops_match_independent_loaded_recurrence_and_edits() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_general_compliant_loops_match_independent_loaded_recurrence_and_edits() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-4,
    );
}

struct CoupledCase {
    bodies: [BodyHandle; 2],
    joints: [JointHandle; 2],
    mass: [f64; 2],
    origin: Vec3,
    direction: Vec3,
    position: [f64; 2],
    velocity: [f64; 2],
    softness: [JointSoftness; 2],
}

fn coupled_fixture() -> (RigidScene, Vec<CoupledCase>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    for mass in [0.01, 1.0, 100.0] {
        for angle in [0.0, 0.7] {
            let rotation = Quat::from_rotation_z(angle);
            let direction = rotation * Vec3::Y;
            let origin = Vec3::new(8.0 * cases.len() as f64, 2.0, -3.0);
            let a = body(
                &mut scene,
                Transform::new(origin, rotation),
                mass,
                Vec3::ZERO,
            );
            let b = body(
                &mut scene,
                Transform::new(origin, rotation),
                3.0,
                Vec3::ZERO,
            );
            for (parent, child) in [(None, a), (Some(a), b)] {
                scene
                    .insert_joint(JointDesc {
                        parent,
                        child,
                        kind: JointType::Prismatic,
                        axis: Vec3::Y,
                        frame_parent: if parent.is_none() {
                            Transform::new(origin, rotation)
                        } else {
                            Transform::IDENTITY
                        },
                        frame_child: Transform::IDENTITY,
                    })
                    .unwrap();
            }
            let softness = [
                JointSoftness {
                    natural_frequency: 4.0,
                    damping_ratio: 0.6,
                },
                JointSoftness {
                    natural_frequency: 7.0,
                    damping_ratio: 0.9,
                },
            ];
            let mut joints = Vec::new();
            for (index, (parent, child)) in [(None, a), (Some(a), b)].into_iter().enumerate() {
                let joint = scene
                    .insert_loop_joint(
                        JointDesc {
                            parent,
                            child,
                            kind: JointType::D6,
                            axis: Vec3::X,
                            frame_parent: if parent.is_none() {
                                Transform::new(origin - direction * 0.03, rotation)
                            } else {
                                Transform::IDENTITY
                            },
                            frame_child: Transform::from_translation(
                                Vec3::Y * if index == 0 { 0.0 } else { 0.02 },
                            ),
                        },
                        softness[index],
                    )
                    .unwrap();
                scene
                    .set_joint_locked_axes(joint, JointAxes::NONE.with(JointAxis::LinY))
                    .unwrap();
                joints.push(joint);
            }
            for (body, force) in [(a, mass * 0.3), (b, -3.0 * 0.2)] {
                scene
                    .add_body_wrench(body, SpatialVector::new(direction * force, Vec3::ZERO))
                    .unwrap();
            }
            cases.push(CoupledCase {
                bodies: [a, b],
                joints: [joints[0], joints[1]],
                mass: [mass, 3.0],
                origin,
                direction,
                position: [0.0; 2],
                velocity: [0.0; 2],
                softness,
            });
        }
    }
    (scene, cases)
}

fn coupled_advance(case: &mut CoupledCase, dt: f64) -> [f64; 2] {
    let g = [
        1.0 / case.mass[0],
        -1.0 / case.mass[0],
        1.0 / case.mass[0] + 1.0 / case.mass[1],
    ];
    let errors = [
        case.position[0] + 0.03,
        case.position[1] - case.position[0] + 0.02,
    ];
    let rates = [case.velocity[0], case.velocity[1] - case.velocity[0]];
    let mut diagonal = [g[0], g[2]];
    let mut rhs = [0.0; 2];
    for index in 0..2 {
        let omega = std::f64::consts::TAU * case.softness[index].natural_frequency;
        let stiffness = omega * omega / diagonal[index];
        let damping = 2.0 * case.softness[index].damping_ratio * omega / diagonal[index];
        diagonal[index] += 1.0 / (dt * (damping + dt * stiffness));
        rhs[index] = -rates[index] - stiffness * errors[index] / (damping + dt * stiffness);
    }
    let determinant = diagonal[0] * diagonal[1] - g[1] * g[1];
    let impulses = [
        (rhs[0] * diagonal[1] - rhs[1] * g[1]) / determinant,
        (rhs[1] * diagonal[0] - rhs[0] * g[1]) / determinant,
    ];
    case.velocity[0] += (impulses[0] - impulses[1]) / case.mass[0] + 0.3 * dt;
    case.velocity[1] += impulses[1] / case.mass[1] - 0.2 * dt;
    for index in 0..2 {
        case.position[index] += case.velocity[index] * dt;
    }
    impulses.map(|impulse| impulse / dt)
}

fn certify_coupled(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for hz in [60, 120, 240] {
        for substeps in [1, 4] {
            let (scene, mut cases) = coupled_fixture();
            let mut world = make_world(scene);
            world.solver.substeps = substeps;
            world.set_reaction_readout(true).unwrap();
            let mut initial_compilations = None;
            for frame in 0..20 {
                if frame == 5 {
                    for case in &mut cases {
                        case.softness[0].natural_frequency = 9.0;
                        for (joint, softness) in case.joints.into_iter().zip(case.softness) {
                            world
                                .scene
                                .set_joint_softness(joint, Some(softness))
                                .unwrap();
                        }
                    }
                }
                let mut expected = vec![[0.0; 2]; cases.len()];
                for _ in 0..substeps {
                    for (case, forces) in cases.iter_mut().zip(&mut expected) {
                        let sample = coupled_advance(case, 1.0 / (hz as f64 * substeps as f64));
                        for index in 0..2 {
                            forces[index] += sample[index] / substeps as f64;
                        }
                    }
                }
                world.step(1.0 / hz as f64).unwrap_or_else(|error| {
                    panic!("hz={hz} substeps={substeps} frame={frame}: {error}")
                });
                assert!(world.quarantined_bodies().is_empty());
                #[cfg(feature = "gpu")]
                if let Some(diagnostics) = world.gpu_diagnostics() {
                    let initial =
                        *initial_compilations.get_or_insert(diagnostics.model_compilations);
                    assert_eq!(
                        diagnostics.model_compilations, initial,
                        "softness edit must not rebuild retained loop resources"
                    );
                }
                #[cfg(not(feature = "gpu"))]
                let _: &mut Option<u64> = &mut initial_compilations;
                for (case, forces) in cases.iter().zip(expected) {
                    for index in 0..2 {
                        let pose = world.scene.body_pose(case.bodies[index]).unwrap();
                        let velocity = world.scene.body_velocity(case.bodies[index]).unwrap();
                        assert!(
                            (pose.position - case.origin - case.direction * case.position[index])
                                .length()
                                < tolerance,
                            "coupled position hz={hz} substeps={substeps} frame={frame} end={index}"
                        );
                        assert!(
                            (velocity.linear - case.direction * case.velocity[index]).length()
                                < tolerance,
                            "coupled velocity hz={hz} substeps={substeps} frame={frame} end={index}"
                        );
                        assert!(velocity.angular.length() < tolerance);
                        let reaction = world.joint_reaction(case.joints[index]).unwrap();
                        assert!(
                            (reaction.child.linear - case.direction * forces[index]).length()
                                < tolerance * (1.0 + forces[index].abs()),
                            "coupled reaction hz={hz} substeps={substeps} frame={frame} row={index} expected={} actual={:?}",
                            forces[index],
                            reaction.child
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn cpu_coupled_compliant_loops_match_two_by_two_loaded_recurrence_and_substeps() {
    certify_coupled(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_coupled_compliant_loops_match_two_by_two_loaded_recurrence_and_substeps() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify_coupled(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-4,
    );
}

fn certify_breakage(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for substeps in [1, 4] {
        for capture in [false, true] {
            let (mut scene, mut cases) = fixture();
            let dt = 1.0 / 60.0;
            let h = dt / substeps as f64;
            let forces: Vec<_> = cases.iter_mut().map(|case| advance(case, h)).collect();
            for (case, force) in cases.iter_mut().zip(&forces) {
                scene
                    .set_joint_break_limits(
                        case.joint,
                        JointBreakLimits {
                            force: Some(force.abs() * 0.5),
                            torque: None,
                        },
                    )
                    .unwrap();
                case.enabled = false;
            }
            for _ in 1..substeps {
                for case in &mut cases {
                    advance(case, h);
                }
            }
            let mut world = make_world(scene);
            world.solver.substeps = substeps;
            world.set_reaction_readout(capture).unwrap();
            world.step(dt).unwrap();
            assert_eq!(world.joint_break_events().len(), cases.len());
            for (case, force) in cases.iter().zip(&forces) {
                assert!(world.scene.joint_broken(case.joint).unwrap());
                let event = world
                    .joint_break_events()
                    .iter()
                    .find(|event| event.joint == case.joint)
                    .unwrap();
                assert_eq!(event.trigger, JointBreakTrigger::Force);
                assert_eq!(event.sample.substep, 0);
                assert!((event.sample.sample_dt - h).abs() < 1e-12);
                assert!((event.sample.time_offset - h).abs() < 1e-12);
                assert!(
                    (event.sample.child.linear - case.direction * force).length()
                        < tolerance * (1.0 + force.abs())
                );
                if capture {
                    let reaction = world.joint_reaction(case.joint).unwrap();
                    assert!(
                        (reaction.child.linear - case.direction * force / substeps as f64).length()
                            < tolerance * (1.0 + force.abs())
                    );
                } else {
                    assert!(world.joint_reaction_samples().is_none());
                }
            }
            for _ in 0..substeps {
                for case in &mut cases {
                    advance(case, h);
                }
            }
            world.step(dt).unwrap();
            assert_eq!(world.joint_break_events().len(), cases.len());
            for case in &cases {
                for (index, body) in [case.parent, Some(case.child)].into_iter().enumerate() {
                    let Some(body) = body else { continue };
                    let pose = world.scene.body_pose(body).unwrap();
                    let velocity = world.scene.body_velocity(body).unwrap();
                    assert!(
                        (pose.position - case.origin - case.direction * case.position[index])
                            .length()
                            < tolerance
                    );
                    assert!(
                        (velocity.linear - case.direction * case.velocity[index]).length()
                            < tolerance
                    );
                }
            }
            for case in &mut cases {
                world
                    .scene
                    .set_joint_break_limits(case.joint, JointBreakLimits::default())
                    .unwrap();
                world.scene.repair_joint(case.joint).unwrap();
                case.enabled = true;
            }
            for _ in 0..substeps {
                for case in &mut cases {
                    advance(case, h);
                }
            }
            world.step(dt).unwrap();
            assert_eq!(world.joint_break_events().len(), cases.len());
            for case in &cases {
                assert!(!world.scene.joint_broken(case.joint).unwrap());
                assert!(
                    (world.scene.body_velocity(case.child).unwrap().linear
                        - case.direction * case.velocity[1])
                        .length()
                        < tolerance
                );
            }
            assert_eq!(world.drain_joint_break_events().len(), cases.len());
            assert!(world.joint_break_events().is_empty());
        }
    }
}

#[test]
fn cpu_compliant_loop_breakage_preserves_substep_motion_and_explicit_repair() {
    certify_breakage(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_compliant_loop_breakage_preserves_substep_motion_and_explicit_repair() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify_breakage(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-4,
    );
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "requires GPU resource counters through make test-gpu"]
fn gpu_compliant_loop_numeric_edits_keep_device_resources_retained() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let (scene, mut cases) = coupled_fixture();
    let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
    world.solver.substeps = 4;
    world.set_reaction_readout(true).unwrap();
    world.step(1.0 / 120.0).unwrap();
    let counters = || {
        let device = engine.device().unwrap();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let counts = device.get_internal_counters().hal;
        [
            counts.buffers.read(),
            counts.buffer_memory.read(),
            counts.bind_groups.read(),
            counts.compute_pipelines.read(),
            counts.shader_modules.read(),
        ]
    };
    let baseline = counters();
    let diagnostics = world.gpu_diagnostics().unwrap();
    for frame in 0..12 {
        if frame % 3 == 0 {
            for case in &mut cases {
                case.softness[0].natural_frequency += 0.5;
                world
                    .scene
                    .set_joint_softness(case.joints[0], Some(case.softness[0]))
                    .unwrap();
            }
        }
        world.step(1.0 / 120.0).unwrap();
        assert!(world.quarantined_bodies().is_empty());
        assert_eq!(
            counters(),
            baseline,
            "retained device resources frame={frame}"
        );
        let current = world.gpu_diagnostics().unwrap();
        assert_eq!(current.model_compilations, diagnostics.model_compilations);
        assert_eq!(
            current.current_model_bind_groups_created,
            diagnostics.current_model_bind_groups_created
        );
        assert_eq!(
            current.last_state_upload_bytes,
            diagnostics.last_state_upload_bytes
        );
        assert_eq!(
            current.last_state_readback_bytes,
            diagnostics.last_state_readback_bytes
        );
    }
    eprintln!(
        "GPU compliant loop standing resources={baseline:?}; state transfer diagnostics={diagnostics:?}"
    );
}
