use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, JointSoftness};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, JointBreakLimits, JointBreakTrigger,
    JointDesc, JointHandle, JointSettings, RigidScene,
};
use molla_sim::{
    BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, JointTargetMode, JointType,
    MassProperties,
};
use molla_solvers::rigid_world::RigidWorld;

const AXIS: Vec3 = Vec3::new(0.8, 0.6, 0.0);
const LEVER: Vec3 = Vec3::new(0.8, 0.0, 0.5);
const ARM: Vec3 = Vec3::new(0.0, 0.0, 0.12);
const RADIUS: f64 = 0.2;

struct Case {
    bodies: [BodyHandle; 2],
    joints: [JointHandle; 3],
    origin: Vec3,
    mass: [f64; 2],
    materials: [ContactMaterial; 2],
    rotation: Quat,
    direction: Vec3,
    effort_direction: Vec3,
    softness: JointSoftness,
    motor: JointMotor,
    offset: f64,
    position: f64,
    velocity: f64,
    enabled: bool,
}

#[derive(Clone, Copy)]
struct Reading {
    child: SpatialVector,
    parent: SpatialVector,
    anchors: [Vec3; 2],
}

struct Step {
    reactions: [Reading; 3],
    pressure: [f64; 2],
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(-Vec3::Y * 9.81).unwrap();
    let mut cases = Vec::new();
    for scale in [0.01, 1.0, 100.0] {
        for cap in [0.0, 0.25, f64::INFINITY] {
            let origin = Vec3::new(8.0 * cases.len() as f64, RADIUS - 9.81 / 1.2e4, 0.0);
            let mass = [scale, 3.0 * scale];
            let materials = mass.map(|mass| ContactMaterial {
                stiffness: mass * 1.2e4,
                damping: mass * 10.0,
                friction: Some(FrictionCoefficients::uniform(0.0)),
                ..Default::default()
            });
            let mut bodies = Vec::new();
            for end in 0..2 {
                let position = origin + if end == 0 { Vec3::ZERO } else { LEVER };
                let body = scene
                    .insert_body(BodyDesc {
                        params: BodyParams::new().with_position(position),
                        additional_mass: Some(MassProperties {
                            mass: mass[end],
                            local_com: Vec3::ZERO,
                            inertia: Mat3::IDENTITY * mass[end],
                        }),
                        ..Default::default()
                    })
                    .unwrap();
                let mut floor = ColliderDesc::new(ColliderGeometry::Cuboid {
                    half_extents: Vec3::new(0.28, 0.1, 0.28),
                });
                floor.pose = Transform::from_translation(Vec3::new(position.x, -0.1, position.z));
                floor.contact_material = materials[end];
                scene.insert_collider(floor).unwrap();
                let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: RADIUS });
                ball.parent = Some(body);
                ball.mass = ColliderMass::Mass(0.0);
                ball.contact_material = materials[end];
                scene.insert_collider(ball).unwrap();
                scene
                    .add_body_wrench(
                        body,
                        SpatialVector::new(
                            AXIS * mass[end] * if end == 0 { 0.3 } else { -0.1 },
                            Vec3::ZERO,
                        ),
                    )
                    .unwrap();
                bodies.push(body);
            }
            let root = scene
                .insert_joint(JointDesc {
                    parent: None,
                    child: bodies[0],
                    kind: JointType::Prismatic,
                    axis: AXIS,
                    frame_parent: Transform::from_translation(origin),
                    frame_child: Transform::IDENTITY,
                })
                .unwrap();
            let fixed = scene
                .insert_joint(JointDesc {
                    parent: Some(bodies[0]),
                    child: bodies[1],
                    kind: JointType::Fixed,
                    axis: Vec3::X,
                    frame_parent: Transform::from_translation(LEVER),
                    frame_child: Transform::IDENTITY,
                })
                .unwrap();
            let rotation = Quat::from_rotation_z(-0.4);
            let softness = JointSoftness {
                natural_frequency: 4.0,
                damping_ratio: 0.6,
            };
            let joint = scene
                .insert_loop_joint(
                    JointDesc {
                        parent: None,
                        child: bodies[1],
                        kind: JointType::D6,
                        axis: Vec3::X,
                        frame_parent: Transform::new(origin + LEVER + ARM, rotation),
                        frame_child: Transform::new(ARM, rotation),
                    },
                    softness,
                )
                .unwrap();
            scene
                .set_joint_locked_axes(joint, JointAxes::NONE.with(JointAxis::LinX))
                .unwrap();
            let motor = JointMotor {
                mode: JointTargetMode::Effort,
                target_position: 0.0,
                max_force: mass.iter().sum::<f64>() * cap,
                ..Default::default()
            };
            scene
                .set_joint_motor(joint, JointAxis::LinY, motor)
                .unwrap();
            cases.push(Case {
                bodies: [bodies[0], bodies[1]],
                joints: [root, fixed, joint],
                origin,
                mass,
                materials,
                rotation,
                direction: rotation * Vec3::X,
                effort_direction: rotation * Vec3::Y,
                softness,
                motor,
                offset: 0.0,
                position: 0.0,
                velocity: 0.0,
                enabled: true,
            });
        }
    }
    (scene, cases)
}

fn advance(case: &mut Case, dt: f64) -> Step {
    let total_mass: f64 = case.mass.iter().sum();
    let alpha = AXIS.dot(case.direction);
    let beta = AXIS.dot(case.effort_direction);
    let omega = std::f64::consts::TAU * case.softness.natural_frequency;
    let stiffness = omega * omega * total_mass / (alpha * alpha);
    let damping = 2.0 * case.softness.damping_ratio * omega * total_mass / (alpha * alpha);
    let effort = if case.enabled {
        case.motor
            .target_position
            .clamp(-case.motor.max_force, case.motor.max_force)
    } else {
        0.0
    };
    let feedback_impulse = if case.enabled {
        let rhs = -alpha * case.velocity
            - stiffness * (case.offset + alpha * case.position) / (damping + dt * stiffness)
            - alpha * beta / total_mass * dt * effort;
        rhs / (alpha * alpha / total_mass + 1.0 / (dt * (damping + dt * stiffness)))
    } else {
        0.0
    };
    let loop_force = case.direction * (feedback_impulse / dt) + case.effort_direction * effort;
    let depth = RADIUS - case.origin.y - AXIS.y * case.position;
    assert!(
        depth > 0.0 && depth < 0.002,
        "articulated contact depth={depth}"
    );
    let pressure = case.materials.map(|material| {
        material.stiffness * depth + material.damping * (-AXIS.y * case.velocity).max(0.0)
    });
    let force_a = Vec3::Y * (pressure[0] - 9.81 * case.mass[0]) + AXIS * (0.3 * case.mass[0]);
    let force_b =
        Vec3::Y * (pressure[1] - 9.81 * case.mass[1]) - AXIS * (0.1 * case.mass[1]) + loop_force;
    let acceleration = AXIS.dot(force_a + force_b) / total_mass;
    let fixed_force = AXIS * (case.mass[1] * acceleration) - force_b;
    let fixed_torque = -ARM.cross(loop_force);
    let root_force = AXIS * (total_mass * acceleration) - force_a - force_b;
    let root_torque = LEVER.cross(fixed_force) + fixed_torque;
    let a = case.origin + AXIS * case.position;
    let b = a + LEVER;
    let loop_delta = AXIS * case.position + case.direction * case.offset;
    let reactions = [
        Reading {
            child: SpatialVector::new(root_force, root_torque),
            parent: SpatialVector::new(
                -root_force,
                -root_torque - (a - case.origin).cross(root_force),
            ),
            anchors: [case.origin, a],
        },
        Reading {
            child: SpatialVector::new(fixed_force, fixed_torque),
            parent: SpatialVector::new(-fixed_force, -fixed_torque),
            anchors: [b, b],
        },
        Reading {
            child: SpatialVector::new(loop_force, Vec3::ZERO),
            parent: SpatialVector::new(-loop_force, -loop_delta.cross(loop_force)),
            anchors: [
                case.origin + LEVER + ARM - case.direction * case.offset,
                b + ARM,
            ],
        },
    ];
    case.velocity += dt * acceleration;
    case.position += dt * case.velocity;
    Step {
        reactions,
        pressure,
    }
}

fn edit(scene: &mut RigidScene, case: &mut Case, frame: usize) {
    let joint = case.joints[2];
    let total_mass: f64 = case.mass.iter().sum();
    if frame == 2 {
        case.motor.target_position = 0.4 * total_mass;
    }
    if frame == 4 {
        case.offset = 0.0006;
        scene
            .set_joint_frames(
                joint,
                Transform::new(
                    case.origin + LEVER + ARM - case.direction * case.offset,
                    case.rotation,
                ),
                Transform::new(ARM, case.rotation),
            )
            .unwrap();
    }
    if frame == 6 {
        case.softness.natural_frequency = 7.0;
        scene
            .set_joint_softness(joint, Some(case.softness))
            .unwrap();
    }
    if frame == 8 {
        case.motor.max_force = 0.0;
    }
    if frame == 10 {
        case.motor.target_position = -0.2 * total_mass;
        case.motor.max_force = 0.3 * total_mass;
    }
    if matches!(frame, 2 | 8 | 10) {
        scene
            .set_joint_motor(joint, JointAxis::LinY, case.motor)
            .unwrap();
    }
    if matches!(frame, 12 | 14) {
        case.enabled = frame == 14;
        scene
            .set_joint_settings(
                joint,
                JointSettings {
                    enabled: case.enabled,
                    ..Default::default()
                },
            )
            .unwrap();
    }
}

fn near(actual: SpatialVector, expected: SpatialVector, tolerance: f64, mass: f64, context: &str) {
    let bound = tolerance * (1.0 + mass + expected.linear.length() + expected.angular.length());
    assert!(
        (actual.linear - expected.linear).length() < bound,
        "{context} force expected={expected:?} actual={actual:?} bound={bound}"
    );
    assert!(
        (actual.angular - expected.angular).length() < bound,
        "{context} torque expected={expected:?} actual={actual:?} bound={bound}"
    );
}

fn certify(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for hz in [60, 120, 240] {
        for substeps in [1, 4] {
            let (scene, mut cases) = fixture();
            let mut world = make_world(scene);
            world.solver.substeps = substeps;
            world.set_reaction_readout(true).unwrap();
            let dt = 1.0 / hz as f64;
            let sub_dt = dt / substeps as f64;
            for frame in 0..16 {
                for case in &mut cases {
                    edit(&mut world.scene, case, frame);
                }
                let expected: Vec<_> = (0..substeps)
                    .map(|_| {
                        cases
                            .iter_mut()
                            .map(|case| advance(case, sub_dt))
                            .collect::<Vec<_>>()
                    })
                    .collect();
                world.step(dt).unwrap_or_else(|error| {
                    panic!("articulated contact hz={hz} substeps={substeps} frame={frame}: {error}")
                });
                let samples = world.joint_reaction_samples().unwrap();
                let contacts = world.contact_pairs();
                assert_eq!(contacts.len(), cases.len() * 2);
                let expected_sample_count = cases
                    .iter()
                    .map(|case| if case.enabled { 3 } else { 2 })
                    .sum::<usize>()
                    * substeps;
                assert_eq!(samples.len(), expected_sample_count);
                for (index, case) in cases.iter().enumerate() {
                    let total_mass: f64 = case.mass.iter().sum();
                    let context =
                        format!("hz={hz} substeps={substeps} frame={frame} mass={total_mass}");
                    for end in 0..2 {
                        let pose = world.scene.body_pose(case.bodies[end]).unwrap();
                        let position = case.origin
                            + AXIS * case.position
                            + if end == 0 { Vec3::ZERO } else { LEVER };
                        assert!(
                            (pose.position - position).length() < tolerance,
                            "{context} end={end} position={position:?} actual={pose:?}"
                        );
                        assert!(pose.rotation.to_scaled_axis().length() < tolerance);
                        let velocity = world.scene.body_velocity(case.bodies[end]).unwrap();
                        assert!(
                            (velocity.linear - AXIS * case.velocity).length() < tolerance,
                            "{context} end={end} rate={} actual={velocity:?}",
                            case.velocity
                        );
                        assert!(velocity.angular.length() < tolerance);
                        let pair = contacts
                            .iter()
                            .find(|pair| pair.bodies.contains(&Some(case.bodies[end])))
                            .unwrap();
                        let impulse: f64 =
                            pair.points.iter().map(|point| point.normal_impulse).sum();
                        let reference: f64 = expected
                            .iter()
                            .map(|step| step[index].pressure[end] * sub_dt)
                            .sum();
                        assert!(
                            (impulse - reference).abs()
                                < tolerance * (1.0 + reference + total_mass * dt),
                            "{context} end={end} contact impulse expected={reference} actual={impulse}"
                        );
                    }
                    for (joint_index, &joint) in case.joints.iter().enumerate() {
                        if joint_index == 2 && !case.enabled {
                            assert!(world.joint_reaction(joint).is_none());
                            continue;
                        }
                        let mut mean_child = SpatialVector::ZERO;
                        let mut mean_parent = SpatialVector::ZERO;
                        let mut peak_force = 0.0_f64;
                        let mut peak_torque = 0.0_f64;
                        for (substep, expected) in expected.iter().enumerate() {
                            let reference = expected[index].reactions[joint_index];
                            let sample = samples
                                .iter()
                                .find(|sample| sample.joint == joint && sample.substep == substep)
                                .unwrap();
                            near(
                                sample.child,
                                reference.child,
                                tolerance,
                                total_mass,
                                &format!("{context} joint={joint_index} step={substep} child"),
                            );
                            near(
                                sample.parent,
                                reference.parent,
                                tolerance,
                                total_mass,
                                &format!("{context} joint={joint_index} step={substep} parent"),
                            );
                            for anchor in 0..2 {
                                assert!(
                                    (sample.anchors[anchor] - reference.anchors[anchor]).length()
                                        < tolerance,
                                    "{context} joint={joint_index} anchor={anchor} expected={:?} actual={:?}",
                                    reference.anchors,
                                    sample.anchors
                                );
                            }
                            assert!((sample.sample_dt - sub_dt).abs() < 1e-12);
                            assert!(
                                (sample.time_offset - (substep + 1) as f64 * sub_dt).abs() < 1e-12
                            );
                            mean_child = mean_child + reference.child.scale(1.0 / substeps as f64);
                            mean_parent =
                                mean_parent + reference.parent.scale(1.0 / substeps as f64);
                            peak_force = peak_force.max(reference.child.linear.length());
                            peak_torque = peak_torque.max(reference.child.angular.length());
                        }
                        let reading = world.joint_reaction(joint).unwrap();
                        near(
                            reading.child,
                            mean_child,
                            tolerance,
                            total_mass,
                            &format!("{context} joint={joint_index} mean child"),
                        );
                        near(
                            reading.parent,
                            mean_parent,
                            tolerance,
                            total_mass,
                            &format!("{context} joint={joint_index} mean parent"),
                        );
                        assert!(
                            (reading.peak_force - peak_force).abs()
                                < tolerance * (1.0 + total_mass + peak_force)
                        );
                        assert!(
                            (reading.peak_torque - peak_torque).abs()
                                < tolerance * (1.0 + total_mass + peak_torque)
                        );
                        assert!((reading.sample_dt - dt).abs() < 1e-12);
                    }
                }
                assert!(world.quarantined_bodies().is_empty());
            }
            eprintln!(
                "articulated rotated loop/contact oracle hz={hz} substeps={substeps}: 9 assemblies passed"
            );
        }
    }
}

#[test]
fn cpu_rotated_loops_and_articulated_contacts_match_motion_and_anchor_wrench_oracle() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_rotated_loops_and_articulated_contacts_match_motion_and_anchor_wrench_oracle() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify(
        |scene| {
            RigidWorld::from_scene_gpu_with_contact_capacity(
                scene,
                engine.clone(),
                molla_solvers::rigid_world::GpuContactCapacity {
                    pairs: 64,
                    contacts_per_island: 8,
                },
            )
            .unwrap()
        },
        2e-4,
    );
}

fn check_motion(world: &RigidWorld, cases: &[Case], tolerance: f64, stage: &str) {
    for case in cases {
        for end in 0..2 {
            let position =
                case.origin + AXIS * case.position + if end == 0 { Vec3::ZERO } else { LEVER };
            let actual = world.scene.body_pose(case.bodies[end]).unwrap();
            assert!(
                (actual.position - position).length() < tolerance,
                "{stage} break boundary expected={position:?} actual={actual:?}"
            );
            assert!(actual.rotation.to_scaled_axis().length() < tolerance);
            let velocity = world.scene.body_velocity(case.bodies[end]).unwrap();
            assert!(
                (velocity.linear - AXIS * case.velocity).length() < tolerance,
                "{stage} break boundary expected rate={} actual={velocity:?}",
                case.velocity
            );
            assert!(velocity.angular.length() < tolerance);
        }
    }
    assert!(world.quarantined_bodies().is_empty());
}

fn certify_breaks(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for hz in [60, 240] {
        for capture in [false, true] {
            let (scene, mut cases) = fixture();
            let mut world = make_world(scene);
            world.set_reaction_readout(capture).unwrap();
            world.solver.substeps = 4;
            let dt = 1.0 / hz as f64;
            for frame in 0..6 {
                for case in &mut cases {
                    edit(&mut world.scene, case, frame);
                }
                for _ in 0..4 {
                    for case in &mut cases {
                        advance(case, dt / 4.0);
                    }
                }
                world.step(dt).unwrap();
                check_motion(
                    &world,
                    &cases,
                    tolerance,
                    &format!("prefix hz={hz} capture={capture} frame={frame}"),
                );
            }
            world.solver.substeps = 1;
            for case in &mut cases {
                edit(&mut world.scene, case, 6);
            }
            let expected: Vec<_> = cases.iter_mut().map(|case| advance(case, dt)).collect();
            let triggers = [
                JointBreakTrigger::Torque,
                JointBreakTrigger::Force,
                JointBreakTrigger::Both,
            ];
            for (index, (case, step)) in cases.iter().zip(&expected).enumerate() {
                let root = step.reactions[0].child;
                let mass: f64 = case.mass.iter().sum();
                assert!(
                    root.angular.length() > 1e-3 * mass,
                    "fixture torque margin {root:?}"
                );
                if index % 3 != 0 {
                    assert!(
                        root.linear.length() > 1e-3 * mass,
                        "fixture force margin {root:?}"
                    );
                }
                world
                    .scene
                    .set_joint_break_limits(
                        case.joints[0],
                        JointBreakLimits {
                            force: (index % 3 != 0).then_some(0.6 * root.linear.length()),
                            torque: (index % 3 != 1).then_some(0.6 * root.angular.length()),
                        },
                    )
                    .unwrap();
            }
            world.step(dt).unwrap();
            check_motion(
                &world,
                &cases,
                tolerance,
                &format!("detachment hz={hz} capture={capture}"),
            );
            assert_eq!(world.completed_steps(), 7);
            assert_eq!(world.joint_break_events().len(), cases.len());
            for (index, (case, expected)) in cases.iter().zip(&expected).enumerate() {
                let root = case.joints[0];
                assert!(world.scene.joint_broken(root).unwrap());
                assert!(!world.scene.joint_broken(case.joints[1]).unwrap());
                assert!(!world.scene.joint_broken(case.joints[2]).unwrap());
                let event = world
                    .joint_break_events()
                    .iter()
                    .find(|event| event.joint == root)
                    .unwrap();
                assert_eq!(event.frame, 6);
                assert_eq!(event.trigger, triggers[index % 3]);
                assert_eq!(event.sample.substep, 0);
                assert!((event.sample.sample_dt - dt).abs() < 1e-12);
                assert!((event.sample.time_offset - dt).abs() < 1e-12);
                let mass: f64 = case.mass.iter().sum();
                near(
                    event.sample.child,
                    expected.reactions[0].child,
                    tolerance,
                    mass,
                    "articulated break child",
                );
                near(
                    event.sample.parent,
                    expected.reactions[0].parent,
                    tolerance,
                    mass,
                    "articulated break parent",
                );
                for anchor in 0..2 {
                    assert!(
                        (event.sample.anchors[anchor] - expected.reactions[0].anchors[anchor])
                            .length()
                            < tolerance
                    );
                }
                if capture {
                    near(
                        world.joint_reaction(root).unwrap().child,
                        expected.reactions[0].child,
                        tolerance,
                        mass,
                        "articulated break reading",
                    );
                } else {
                    assert!(world.joint_reaction_samples().is_none());
                }
            }
            for case in &cases {
                let before: Vec<_> = case
                    .bodies
                    .iter()
                    .map(|&body| {
                        (
                            world.scene.body_pose(body).unwrap(),
                            world.scene.body_velocity(body).unwrap(),
                        )
                    })
                    .collect();
                world
                    .scene
                    .set_joint_break_limits(case.joints[0], JointBreakLimits::default())
                    .unwrap();
                assert!(world.scene.joint_broken(case.joints[0]).unwrap());
                world.scene.repair_joint(case.joints[0]).unwrap();
                assert!(!world.scene.joint_broken(case.joints[0]).unwrap());
                for (&body, (pose, velocity)) in case.bodies.iter().zip(before) {
                    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
                    assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
                }
            }
            for case in &mut cases {
                advance(case, dt);
            }
            world.step(dt).unwrap();
            check_motion(
                &world,
                &cases,
                tolerance,
                &format!("repair first frame hz={hz} capture={capture}"),
            );
            assert_eq!(world.joint_break_events().len(), cases.len());
            assert_eq!(world.drain_joint_break_events().len(), cases.len());
            assert!(world.joint_break_events().is_empty());
            for case in &mut cases {
                advance(case, dt);
            }
            world.step(dt).unwrap();
            check_motion(
                &world,
                &cases,
                tolerance,
                &format!("repair second frame hz={hz} capture={capture}"),
            );
            assert!(world.joint_break_events().is_empty());
            eprintln!(
                "articulated contact torque/force/both break hz={hz} capture={capture}: 9 assemblies passed"
            );
        }
    }
}

#[test]
fn cpu_articulated_contact_break_channels_preserve_boundary_state_and_explicit_repair() {
    certify_breaks(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_articulated_contact_break_channels_preserve_boundary_state_and_explicit_repair() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify_breaks(
        |scene| {
            RigidWorld::from_scene_gpu_with_contact_capacity(
                scene,
                engine.clone(),
                molla_solvers::rigid_world::GpuContactCapacity {
                    pairs: 64,
                    contacts_per_island: 8,
                },
            )
            .unwrap()
        },
        2e-4,
    );
}
