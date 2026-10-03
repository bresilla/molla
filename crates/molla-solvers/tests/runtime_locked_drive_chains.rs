use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, JointHandle, RigidScene};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

const H: f64 = 0.005;
const FRAMES: usize = 100;

struct Case {
    kind: JointType,
    models: [MotorModel; 2],
    axis: usize,
    frame: Quat,
    child_frame: Quat,
    inertias: [f64; 2],
    bodies: [BodyHandle; 2],
    joints: [JointHandle; 2],
    rates: [f64; 2],
    angles: [f64; 2],
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    for kind in [JointType::Ball, JointType::D6] {
        for axis in 0..3 {
            for (frame, child_frame) in [
                (Quat::IDENTITY, Quat::IDENTITY),
                (
                    -Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.7),
                    Quat::from_rotation_z(0.3),
                ),
            ] {
                for models in [
                    [MotorModel::Force, MotorModel::Force],
                    [MotorModel::Force, MotorModel::Acceleration],
                    [MotorModel::Acceleration, MotorModel::Force],
                    [MotorModel::Acceleration, MotorModel::Acceleration],
                ] {
                    let inertia = Mat3::from_cols(
                        Vec3::new(2., 0.1, 0.2),
                        Vec3::new(0.1, 3., 0.3),
                        Vec3::new(0.2, 0.3, 4.),
                    );
                    let direction = [Vec3::X, Vec3::Y, Vec3::Z][axis];
                    let local_axis = child_frame * direction;
                    let effective = local_axis.dot(inertia * local_axis)
                        + (Vec3::Z * 0.2).cross(local_axis).length_squared();
                    let bodies = [2., 17.].map(|mass| {
                        scene
                            .insert_body(BodyDesc {
                                params: BodyParams::new().with_pose(Transform::from_rotation(
                                    frame * child_frame.conjugate(),
                                )),
                                additional_mass: Some(MassProperties {
                                    mass,
                                    local_com: Vec3::Z * 0.2,
                                    inertia: inertia * mass,
                                }),
                                ..Default::default()
                            })
                            .unwrap()
                    });
                    let joints = [0, 1].map(|i| {
                        scene
                            .insert_joint(JointDesc {
                                parent: (i == 1).then_some(bodies[0]),
                                child: bodies[i],
                                kind,
                                axis: direction,
                                frame_parent: Transform::from_rotation(if i == 0 {
                                    frame
                                } else {
                                    child_frame
                                }),
                                frame_child: Transform::from_rotation(child_frame),
                            })
                            .unwrap()
                    });
                    for i in 0..2 {
                        if kind == JointType::D6 {
                            scene
                                .set_joint_locked_axes(
                                    joints[i],
                                    JointAxes(JointAxes::ALL.0 & !(1 << (axis + 3))),
                                )
                                .unwrap();
                        } else {
                            for locked in 0..3 {
                                if locked != axis {
                                    scene
                                        .set_joint_limits(
                                            joints[i],
                                            JointAxis::ALL[locked + 3],
                                            Some([0., 0.]),
                                        )
                                        .unwrap();
                                }
                            }
                        }
                        scene
                            .set_joint_motor(
                                joints[i],
                                JointAxis::ALL[axis + 3],
                                JointMotor {
                                    mode: JointTargetMode::Velocity,
                                    model: models[i],
                                    target_velocity: [0.7, -0.5][i],
                                    damping: [3., 7.][i],
                                    ..Default::default()
                                },
                            )
                            .unwrap();
                    }
                    cases.push(Case {
                        kind,
                        models,
                        axis,
                        frame,
                        child_frame,
                        inertias: [2. * effective, 17. * effective],
                        bodies,
                        joints,
                        rates: [0.; 2],
                        angles: [0.; 2],
                    });
                }
            }
        }
    }
    assert_eq!(cases.len(), 48);
    assert_eq!(scene.model().articulation_count, 48);
    (scene, cases)
}

fn advance(case: &mut Case) -> [f64; 2] {
    let inverse = [
        [1. / case.inertias[0], -1. / case.inertias[0]],
        [
            -1. / case.inertias[0],
            1. / case.inertias[0] + 1. / case.inertias[1],
        ],
    ];
    let gains = [3., 7.];
    let targets = [0.7, -0.5];
    let mut mean = [0.; 2];
    for _ in 0..2 {
        let mut torque = [0.; 2];
        for i in 0..2 {
            if case.models[i] == MotorModel::Acceleration {
                torque[i] =
                    gains[i] * (targets[i] - case.rates[i]) / ((1. + H * gains[i]) * inverse[i][i]);
            }
        }
        if case.models == [MotorModel::Force, MotorModel::Force] {
            let matrix = [
                [1. / gains[0] + H * inverse[0][0], H * inverse[0][1]],
                [H * inverse[1][0], 1. / gains[1] + H * inverse[1][1]],
            ];
            let rhs = [targets[0] - case.rates[0], targets[1] - case.rates[1]];
            let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
            torque = [
                (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
                (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
            ];
        } else {
            for i in 0..2 {
                if case.models[i] == MotorModel::Force {
                    let other = 1 - i;
                    torque[i] = gains[i]
                        * (targets[i] - case.rates[i] - H * inverse[i][other] * torque[other])
                        / (1. + H * gains[i] * inverse[i][i]);
                }
            }
        }
        for i in 0..2 {
            case.rates[i] += H * (inverse[i][0] * torque[0] + inverse[i][1] * torque[1]);
            case.angles[i] += 2. * (0.5 * H * case.rates[i]).atan();
            mean[i] += torque[i] / 2.;
        }
    }
    mean
}

type Trace = Vec<Vec<[(Transform, SpatialVector); 2]>>;

fn rotation_error(actual: Quat, expected: Quat) -> f64 {
    let relative = actual.normalize() * expected.normalize().conjugate();
    2. * Vec3::new(relative.x, relative.y, relative.z)
        .length()
        .atan2(relative.w.abs())
}

fn certify(
    world: &mut RigidWorld,
    cases: &mut [Case],
    tolerance: f64,
    reference: Option<&Trace>,
) -> Trace {
    world.solver.substeps = 2;
    world.solver.stable_motor_feedback = true;
    world.set_reaction_readout(true).unwrap();
    let mut peaks = vec![0_f64; cases.len()];
    let mut peak_parity: f64 = 0.;
    let mut trace = Vec::new();
    #[cfg(feature = "gpu")]
    let mut compilations = None;
    for frame in 0..FRAMES {
        let means: Vec<_> = cases.iter_mut().map(advance).collect();
        world
            .step(0.01)
            .unwrap_or_else(|error| panic!("locked chain frame={frame}: {error}"));
        assert_eq!(world.completed_steps(), (frame + 1) as u64);
        assert!(world.quarantined_bodies().is_empty() && world.joint_break_events().is_empty());
        let state = world.scene.state();
        assert!(
            state
                .body_f
                .host()
                .unwrap()
                .iter()
                .all(|f| f.linear.is_finite() && f.angular.is_finite())
        );
        assert!(state.joint_q.host().unwrap().iter().all(|q| q.is_finite()));
        assert!(state.joint_qd.host().unwrap().iter().all(|v| v.is_finite()));
        assert_eq!(
            world.joint_reaction_samples().unwrap().len(),
            cases.len() * 4
        );
        let mut row = Vec::new();
        for (c, case) in cases.iter().enumerate() {
            let mut entries = [(Transform::IDENTITY, SpatialVector::ZERO); 2];
            for (i, entry) in entries.iter_mut().enumerate() {
                let rate = case.rates[..=i].iter().sum::<f64>();
                let angle = case.angles[..=i].iter().sum::<f64>();
                let direction = [Vec3::X, Vec3::Y, Vec3::Z][case.axis];
                let world_axis = case.frame * direction;
                let expected_rotation = case.frame
                    * Quat::from_axis_angle(direction, angle)
                    * case.child_frame.conjugate();
                let expected_angular = world_axis * rate;
                let expected_linear = expected_angular.cross(expected_rotation * Vec3::Z * 0.2);
                let velocity = world.scene.body_velocity(case.bodies[i]).unwrap();
                let pose = world.scene.body_pose(case.bodies[i]).unwrap();
                assert!(pose.position.is_finite() && pose.rotation.is_finite());
                assert!(velocity.linear.is_finite() && velocity.angular.is_finite());
                let reaction = world.joint_reaction(case.joints[i]).unwrap();
                for wrench in [reaction.child, reaction.parent] {
                    assert!(wrench.linear.is_finite() && wrench.angular.is_finite());
                }
                let error = (velocity.angular - expected_angular)
                    .length()
                    .max((velocity.linear - expected_linear).length())
                    .max(pose.position.length())
                    .max(rotation_error(pose.rotation, expected_rotation))
                    .max(
                        (reaction.child.angular.dot(world_axis) - means[c][i]).abs()
                            / case.inertias[i],
                    );
                peaks[c] = peaks[c].max(error);
                assert!(
                    error.is_finite() && error <= tolerance,
                    "kind={:?} models={:?} axis={} endpoint_frame={:?} frame={frame} body={i} error={error:e} velocity={velocity:?}",
                    case.kind,
                    case.models,
                    case.axis,
                    case.frame
                );
                if let Some(reference) = reference {
                    let (cpu_pose, cpu_velocity) = reference[frame][c][i];
                    let parity = (pose.position - cpu_pose.position)
                        .length()
                        .max(rotation_error(pose.rotation, cpu_pose.rotation))
                        .max((velocity.linear - cpu_velocity.linear).length())
                        .max((velocity.angular - cpu_velocity.angular).length());
                    peak_parity = peak_parity.max(parity);
                    assert!(
                        parity <= 1e-3,
                        "frame={frame} case={c} body={i} parity={parity:e}"
                    );
                }
                *entry = (pose, velocity);
            }
            row.push(entries);
        }
        #[cfg(feature = "gpu")]
        if let Some(diagnostics) = world.gpu_diagnostics() {
            assert_eq!(
                *compilations.get_or_insert(diagnostics.model_compilations),
                diagnostics.model_compilations
            );
        }
        trace.push(row);
    }
    for (case, peak) in cases.iter().zip(peaks) {
        println!(
            "locked_drive_chain kind={:?} models={:?} axis={} endpoint_frame={:?} frames={FRAMES} dt=.01 substeps=2 peak={peak:e}",
            case.kind, case.models, case.axis, case.frame
        );
    }
    println!("locked_drive_chain_parity peak={peak_parity:e}");
    trace
}

#[test]
fn cpu_partial_angular_locks_preserve_coupled_force_acceleration_feedback() {
    let (scene, mut cases) = fixture();
    certify(
        &mut RigidWorld::from_scene(scene).unwrap(),
        &mut cases,
        1e-8,
        None,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_partial_angular_locks_preserve_coupled_feedback_and_match_cpu() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let (cpu_scene, mut cpu_cases) = fixture();
    let reference = certify(
        &mut RigidWorld::from_scene(cpu_scene).unwrap(),
        &mut cpu_cases,
        1e-8,
        None,
    );
    let (scene, mut cases) = fixture();
    println!("locked_drive_chain_prepare_begin cases={}", cases.len());
    let preparation = std::time::Instant::now();
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    println!(
        "locked_drive_chain_prepare_complete seconds={:.6}",
        preparation.elapsed().as_secs_f64()
    );
    certify(&mut world, &mut cases, 1e-3, Some(&reference));
}
