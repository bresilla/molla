use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, JointHandle, RigidScene};
use molla_sim::{JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

#[derive(Debug)]
struct Oracle {
    parent: Option<BodyHandle>,
    child: BodyHandle,
    joint: JointHandle,
    kind: JointType,
    inertia: f64,
    motor: JointMotor,
    position: f64,
    velocity: f64,
    load_acceleration: f64,
}

fn body(scene: &mut RigidScene, mass: f64) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * (2.0 * mass),
            }),
            ..Default::default()
        })
        .unwrap()
}

fn fixture(floating: bool, loaded: bool) -> (RigidScene, Vec<Oracle>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    for kind in [JointType::Prismatic, JointType::Revolute] {
        for mass in [0.001, 1.0, 1000.0] {
            for model in [MotorModel::Force, MotorModel::Acceleration] {
                for mode in [
                    JointTargetMode::Position,
                    JointTargetMode::Velocity,
                    JointTargetMode::PositionVelocity,
                    JointTargetMode::Effort,
                ] {
                    for max_force in [0.25, f64::INFINITY] {
                        let parent = floating.then(|| body(&mut scene, mass * 0.25));
                        let child = body(&mut scene, mass);
                        let joint = scene
                            .insert_joint(JointDesc {
                                parent,
                                child,
                                kind,
                                axis: Vec3::Z,
                                frame_parent: Transform::IDENTITY,
                                frame_child: Transform::IDENTITY,
                            })
                            .unwrap();
                        let motor = JointMotor {
                            mode,
                            model,
                            target_position: 0.4,
                            target_velocity: 0.7,
                            stiffness: 12.0,
                            damping: 3.0,
                            max_force,
                        };
                        let axis = if kind == JointType::Prismatic {
                            JointAxis::LinX
                        } else {
                            JointAxis::AngX
                        };
                        scene.set_joint_motor(joint, axis, motor).unwrap();
                        let inertia =
                            mass * if kind == JointType::Prismatic {
                                1.0
                            } else {
                                2.0
                            } * if floating { 0.2 } else { 1.0 };
                        let load_acceleration = if loaded { 0.3 } else { 0.0 };
                        let load = Vec3::Z * (load_acceleration * mass);
                        scene
                            .add_body_wrench(
                                child,
                                if kind == JointType::Prismatic {
                                    SpatialVector::new(load, Vec3::ZERO)
                                } else {
                                    SpatialVector::new(Vec3::ZERO, load * 2.0)
                                },
                            )
                            .unwrap();
                        cases.push(Oracle {
                            parent,
                            child,
                            joint,
                            kind,
                            inertia,
                            motor,
                            position: 0.0,
                            velocity: 0.0,
                            load_acceleration,
                        });
                    }
                }
            }
        }
    }
    (scene, cases)
}

fn advance(case: &mut Oracle, dt: f64, stable: bool) {
    let motor = case.motor;
    let drive = match motor.mode {
        JointTargetMode::Position => {
            motor.stiffness * (motor.target_position - case.position)
                - motor.damping * case.velocity
        }
        JointTargetMode::Velocity => motor.damping * (motor.target_velocity - case.velocity),
        JointTargetMode::PositionVelocity => {
            motor.stiffness * (motor.target_position - case.position)
                + motor.damping * (motor.target_velocity - case.velocity)
        }
        JointTargetMode::Effort => motor.target_position,
        JointTargetMode::None => 0.0,
    };
    let feedback = matches!(
        motor.mode,
        JointTargetMode::Position | JointTargetMode::Velocity | JointTargetMode::PositionVelocity
    );
    let scale = if feedback && motor.model == MotorModel::Acceleration {
        case.inertia
    } else {
        1.0
    };
    let torque =
        if stable && motor.model == MotorModel::Force && motor.mode == JointTargetMode::Velocity {
            (drive - motor.damping * dt * case.load_acceleration)
                / (1.0 + motor.damping * dt / case.inertia)
        } else if stable && feedback {
            let stiffness = if motor.mode == JointTargetMode::Velocity {
                0.0
            } else {
                motor.stiffness
            };
            (drive - stiffness * dt * case.velocity) * scale
                / (1.0 + (motor.damping + stiffness * dt) * scale * dt / case.inertia)
        } else {
            drive * scale
        };
    case.velocity += (torque.clamp(-motor.max_force, motor.max_force) / case.inertia
        + case.load_acceleration)
        * dt;
    case.position += case.velocity * dt;
}

fn stiff_fixture(floating: bool, loaded: bool) -> (RigidScene, Vec<Oracle>) {
    let (mut scene, mut cases) = fixture(floating, loaded);
    for case in &mut cases {
        case.motor.stiffness = 1e8;
        case.motor.damping = 1e8;
        let axis = if case.kind == JointType::Prismatic {
            JointAxis::LinX
        } else {
            JointAxis::AngX
        };
        scene.set_joint_motor(case.joint, axis, case.motor).unwrap();
    }
    (scene, cases)
}

fn certify(mut world: RigidWorld, mut cases: Vec<Oracle>, stable: bool, tolerance: f64) {
    certify_controls(&mut world, &mut cases, stable, tolerance, false);
}

fn certify_controls(
    world: &mut RigidWorld,
    cases: &mut [Oracle],
    stable: bool,
    tolerance: f64,
    edits: bool,
) {
    world.solver.substeps = 4;
    world.solver.stable_motor_feedback = stable;
    world.set_reaction_readout(edits).unwrap();
    let dt = if stable { 0.04 } else { 0.0004 };
    let modes: Vec<_> = cases.iter().map(|case| case.motor.mode).collect();
    let mut inserted = None;
    for frame in 0..if edits { 40 } else { 10 } {
        if edits && [10, 20, 30].contains(&frame) {
            let stamp = world.scene.model_stamp();
            for (case, &mode) in cases.iter_mut().zip(&modes) {
                if frame == 10 {
                    case.motor.model = match case.motor.model {
                        MotorModel::Force => MotorModel::Acceleration,
                        MotorModel::Acceleration => MotorModel::Force,
                    };
                    case.motor.target_position = -0.3;
                    case.motor.target_velocity = -0.5;
                    case.motor.max_force = 0.07;
                }
                case.motor.mode = if frame == 20 {
                    JointTargetMode::None
                } else {
                    mode
                };
                let axis = if case.kind == JointType::Prismatic {
                    JointAxis::LinX
                } else {
                    JointAxis::AngX
                };
                world
                    .scene
                    .set_joint_motor(case.joint, axis, case.motor)
                    .unwrap();
            }
            assert_eq!(world.scene.model_stamp(), stamp);
        }
        if edits && frame == 30 {
            inserted = Some(body(&mut world.scene, 2.0));
        }
        if edits && frame == 35 {
            world.scene.remove_body(inserted.take().unwrap()).unwrap();
        }
        let mut reactions = Vec::with_capacity(cases.len());
        for case in cases.iter_mut() {
            let previous = case.velocity;
            for _ in 0..4 {
                advance(case, dt / 4.0, stable);
            }
            reactions
                .push(case.inertia * ((case.velocity - previous) / dt - case.load_acceleration));
        }
        world.step(dt).unwrap();
        for (case, reaction) in cases.iter().zip(reactions) {
            let j = world.scene.joint_index(case.joint).unwrap().index();
            let model = world.scene.model();
            let c = model.joint_coord_offset.host().unwrap()[j] as usize;
            let d = model.joint_dof_offset.host().unwrap()[j] as usize;
            let q = world.scene.state().joint_q.host().unwrap()[c];
            let qd = world.scene.state().joint_qd.host().unwrap()[d];
            assert!(
                (q - case.position).abs() < tolerance * (1.0 + case.position.abs())
                    && (qd - case.velocity).abs() < tolerance * (1.0 + case.velocity.abs()),
                "frame={frame} stable={stable} q={q} qd={qd}: {case:?}"
            );
            let rate = |body| {
                let velocity = world.scene.body_velocity(body).unwrap();
                if case.kind == JointType::Prismatic {
                    velocity.linear
                } else {
                    velocity.angular
                }
            };
            let child = rate(case.child);
            let parent = case.parent.map_or(Vec3::ZERO, rate);
            assert!(
                (child - parent - Vec3::Z * case.velocity).length()
                    < tolerance * (1.0 + case.velocity.abs()),
                "physical relative rate at frame={frame}: {case:?}"
            );
            if case.parent.is_some() {
                assert!(
                    (parent * 0.25 + child
                        - Vec3::Z * (case.load_acceleration * dt * (frame + 1) as f64))
                        .length()
                        < tolerance * (1.0 + case.velocity.abs()),
                    "floating momentum at frame={frame}: {case:?}"
                );
            }
            if edits {
                let reading = world.joint_reaction(case.joint).unwrap();
                let expected = Vec3::Z * reaction;
                let (actual, other) = if case.kind == JointType::Prismatic {
                    (reading.child.linear, reading.child.angular)
                } else {
                    (reading.child.angular, reading.child.linear)
                };
                assert!(
                    (actual - expected).length() + other.length()
                        < tolerance * (1.0 + expected.length()),
                    "reaction at frame={frame}: actual={actual:?}, expected={expected:?}: {case:?}"
                );
                assert!(
                    (reading.child.linear + reading.parent.linear).length()
                        + (reading.child.angular + reading.parent.angular).length()
                        < tolerance * (1.0 + expected.length())
                );
            }
        }
    }
}

#[test]
fn cpu_motor_edits_rebuilds_and_reactions_match_scalar_dynamics() {
    for floating in [false, true] {
        let (scene, mut cases) = fixture(floating, true);
        let mut world = RigidWorld::from_scene(scene).unwrap();
        certify_controls(&mut world, &mut cases, true, 1e-8, true);
    }
}

fn certify_d6(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for stable in [false, true] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let mut cases = Vec::new();
        for axis in 0..6 {
            for model in [MotorModel::Force, MotorModel::Acceleration] {
                for mode in [
                    JointTargetMode::Position,
                    JointTargetMode::Velocity,
                    JointTargetMode::PositionVelocity,
                    JointTargetMode::Effort,
                ] {
                    for max_force in [0.25, f64::INFINITY] {
                        let parent_frame = Transform::new(
                            Vec3::new(cases.len() as f64 * 2.0, 3.0, -1.0),
                            Quat::from_rotation_y(0.2) * Quat::from_rotation_z(-0.15),
                        );
                        let child_frame = Transform::from_rotation(Quat::from_rotation_x(0.1));
                        let child = body(&mut scene, 7.0);
                        scene
                            .set_body_pose(child, parent_frame.compose(&child_frame.inverse()))
                            .unwrap();
                        let joint = scene
                            .insert_joint(JointDesc {
                                parent: None,
                                child,
                                kind: JointType::D6,
                                axis: Vec3::X,
                                frame_parent: parent_frame,
                                frame_child: child_frame,
                            })
                            .unwrap();
                        scene
                            .set_joint_locked_axes(
                                joint,
                                JointAxes(JointAxes::ALL.0 & !(1 << axis)),
                            )
                            .unwrap();
                        let motor = JointMotor {
                            mode,
                            model,
                            target_position: 0.4,
                            target_velocity: 0.7,
                            stiffness: 12.0,
                            damping: 3.0,
                            max_force,
                        };
                        scene
                            .set_joint_motor(joint, JointAxis::ALL[axis], motor)
                            .unwrap();
                        cases.push((
                            Oracle {
                                parent: None,
                                child,
                                joint,
                                kind: JointType::D6,
                                inertia: if axis < 3 { 7.0 } else { 14.0 },
                                motor,
                                position: 0.0,
                                velocity: 0.0,
                                load_acceleration: 0.0,
                            },
                            axis,
                            parent_frame,
                            child_frame,
                        ));
                    }
                }
            }
        }
        let mut world = make_world(scene);
        world.solver.substeps = 4;
        world.solver.stable_motor_feedback = stable;
        let dt = 0.04;
        for frame in 0..10 {
            for (case, axis, _, _) in &mut cases {
                for _ in 0..4 {
                    let previous = case.position;
                    advance(case, dt / 4.0, stable);
                    if *axis >= 3 {
                        case.position = previous + 2.0 * (case.velocity * dt / 8.0).atan();
                    }
                }
            }
            world.step(dt).unwrap();
            for (case, axis, parent_frame, child_frame) in &cases {
                let mut direction = Vec3::ZERO;
                direction[axis % 3] = 1.0;
                let relative = if *axis < 3 {
                    Transform::from_translation(direction * case.position)
                } else {
                    Transform::from_rotation(Quat::from_axis_angle(direction, case.position))
                };
                let expected = parent_frame
                    .compose(&relative)
                    .compose(&child_frame.inverse());
                let actual = world.scene.body_pose(case.child).unwrap();
                let error = (actual.position - expected.position).length()
                    + actual
                        .rotation
                        .normalize()
                        .angle_between(expected.rotation.normalize())
                        .abs();
                assert!(
                    error < tolerance,
                    "pose frame={frame} axis={axis} error={error}: {case:?}"
                );
                let actual = world.scene.body_velocity(case.child).unwrap();
                let expected = parent_frame.rotation * direction * case.velocity;
                let error = if *axis < 3 {
                    (actual.linear - expected).length() + actual.angular.length()
                } else {
                    (actual.angular - expected).length() + actual.linear.length()
                };
                assert!(
                    error < tolerance,
                    "rate frame={frame} axis={axis} error={error}: {case:?}"
                );
            }
        }
    }
}

#[test]
fn cpu_d6_motor_models_match_each_authored_axis() {
    certify_d6(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-7);
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_d6_motor_models_match_each_authored_axis() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify_d6(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-4,
    );
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_motor_edits_rebuilds_and_reactions_match_scalar_dynamics() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    for floating in [false, true] {
        let (scene, mut cases) = fixture(floating, true);
        let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
        certify_controls(&mut world, &mut cases, true, 2e-4, true);
    }
}

#[test]
fn cpu_motor_models_match_independent_scalar_dynamics() {
    for floating in [false, true] {
        for stable in [false, true] {
            for loaded in [false, true] {
                let (scene, cases) = fixture(floating, loaded);
                certify(RigidWorld::from_scene(scene).unwrap(), cases, stable, 1e-8);
            }
        }
    }
}

#[test]
fn cpu_stiff_motor_models_match_independent_scalar_dynamics() {
    for floating in [false, true] {
        for loaded in [false, true] {
            let (scene, cases) = stiff_fixture(floating, loaded);
            certify(RigidWorld::from_scene(scene).unwrap(), cases, true, 1e-8);
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_stiff_motor_models_match_independent_scalar_dynamics() {
    use molla_compute::{Backend, Engine};
    use std::sync::Arc;
    let engine = Arc::new(Engine::new(Backend::AutoGpu).unwrap());
    for floating in [false, true] {
        for loaded in [false, true] {
            let (scene, cases) = stiff_fixture(floating, loaded);
            certify(
                RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
                cases,
                true,
                2e-4,
            );
        }
    }
}

#[test]
#[cfg(feature = "gpu")]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_motor_models_match_independent_scalar_dynamics() {
    use molla_compute::{Backend, Engine};
    use std::sync::Arc;
    let engine = Arc::new(Engine::new(Backend::AutoGpu).unwrap());
    for floating in [false, true] {
        for stable in [false, true] {
            for loaded in [false, true] {
                let (scene, cases) = fixture(floating, loaded);
                certify(
                    RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
                    cases,
                    stable,
                    2e-4,
                );
            }
        }
    }
}
