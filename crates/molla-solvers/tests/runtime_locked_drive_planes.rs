use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, JointHandle, RigidScene};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

const INITIAL_DT: f64 = 0.005;
const FRAMES: usize = 100;
const TARGETS: [f64; 2] = [0.2, -0.17];
const GAINS: [f64; 2] = [3., 7.];
type AngularPair = [Vec3; 2];
type Trace = Vec<Vec<[(Transform, SpatialVector); 2]>>;

struct Case {
    kind: JointType,
    models: [MotorModel; 2],
    locked: [usize; 2],
    driven: [usize; 2],
    bodies: [BodyHandle; 2],
    joints: [JointHandle; 2],
    parent_frames: [Quat; 2],
    child_frames: [Quat; 2],
    pivot_inertia: [Mat3; 2],
    lock_rows: [AngularPair; 2],
    drive_rows: [AngularPair; 2],
}

fn dot(a: AngularPair, b: AngularPair) -> f64 {
    a[0].dot(b[0]) + a[1].dot(b[1])
}

fn solve(matrix: [[f64; 2]; 2], rhs: [f64; 2]) -> [f64; 2] {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    assert!(determinant.is_finite() && determinant > 0.);
    [
        (rhs[0] * matrix[1][1] - matrix[0][1] * rhs[1]) / determinant,
        (matrix[0][0] * rhs[1] - rhs[0] * matrix[1][0]) / determinant,
    ]
}

fn project(case: &Case, torque: AngularPair) -> AngularPair {
    let inverse = case.pivot_inertia.map(|inertia| inertia.inverse());
    let response = |row: AngularPair| [inverse[0] * row[0], inverse[1] * row[1]];
    let free = response(torque);
    let locked = case.lock_rows.map(response);
    let gram = case
        .lock_rows
        .map(|row| locked.map(|column| dot(row, column)));
    let multiplier = solve(gram, case.lock_rows.map(|row| dot(row, free)));
    [0, 1].map(|i| free[i] - locked[0][i] * multiplier[0] - locked[1][i] * multiplier[1])
}

fn initial_oracle(case: &Case) -> AngularPair {
    let responses = case.drive_rows.map(|row| project(case, row));
    let inverse = case
        .drive_rows
        .map(|row| responses.map(|column| dot(row, column)));
    let mut torque = [0.; 2];
    for i in 0..2 {
        if case.models[i] == MotorModel::Acceleration {
            torque[i] = GAINS[i] * TARGETS[i] / ((1. + INITIAL_DT * GAINS[i]) * inverse[i][i]);
        }
    }
    if case.models == [MotorModel::Force; 2] {
        torque = solve(
            [
                [
                    1. / GAINS[0] + INITIAL_DT * inverse[0][0],
                    INITIAL_DT * inverse[0][1],
                ],
                [
                    INITIAL_DT * inverse[1][0],
                    1. / GAINS[1] + INITIAL_DT * inverse[1][1],
                ],
            ],
            TARGETS,
        );
    } else {
        for i in 0..2 {
            if case.models[i] == MotorModel::Force {
                let other = 1 - i;
                torque[i] = GAINS[i]
                    * (TARGETS[i] - INITIAL_DT * inverse[i][other] * torque[other])
                    / (1. + INITIAL_DT * GAINS[i] * inverse[i][i]);
            }
        }
    }
    [0, 1].map(|i| INITIAL_DT * (responses[0][i] * torque[0] + responses[1][i] * torque[1]))
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    let axes = [Vec3::X, Vec3::Y, Vec3::Z];
    for kind in [JointType::Ball, JointType::D6] {
        for axis in 0..3 {
            for models in [
                [MotorModel::Force, MotorModel::Force],
                [MotorModel::Force, MotorModel::Acceleration],
                [MotorModel::Acceleration, MotorModel::Force],
                [MotorModel::Acceleration, MotorModel::Acceleration],
            ] {
                let parent_frames = [
                    -Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.7),
                    Quat::from_euler(glam::EulerRot::XYZ, -0.3, 0.6, -0.2),
                ];
                let child_frames = [Quat::from_rotation_z(0.3), Quat::from_rotation_y(-0.2)];
                let world_frames = [
                    parent_frames[0],
                    parent_frames[0] * child_frames[0].conjugate() * parent_frames[1],
                ];
                let rotations = [0, 1].map(|i| world_frames[i] * child_frames[i].conjugate());
                let locked = [axis, (axis + 1) % 3];
                let driven = [(axis + 1) % 3, (axis + 2) % 3];
                let mut pivot_inertia = [Mat3::ZERO; 2];
                let bodies = [0, 1].map(|i| {
                    let mass = [2., 17.][i];
                    let inertia = Mat3::from_cols(
                        Vec3::new(2., 0.1, 0.2),
                        Vec3::new(0.1, 3., 0.3),
                        Vec3::new(0.2, 0.3, 4.),
                    ) * mass;
                    let com = Vec3::new(0.1, 0.2, -0.1);
                    let rotation = Mat3::from_quat(rotations[i]);
                    let world_com = rotation * com;
                    let outer = Mat3::from_cols(
                        world_com * world_com.x,
                        world_com * world_com.y,
                        world_com * world_com.z,
                    );
                    pivot_inertia[i] = rotation * inertia * rotation.transpose()
                        + mass * (Mat3::IDENTITY * world_com.length_squared() - outer);
                    scene
                        .insert_body(BodyDesc {
                            params: BodyParams::new()
                                .with_pose(Transform::from_rotation(rotations[i])),
                            additional_mass: Some(MassProperties {
                                mass,
                                local_com: com,
                                inertia,
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
                            axis: Vec3::X,
                            frame_parent: Transform::from_rotation(parent_frames[i]),
                            frame_child: Transform::from_rotation(child_frames[i]),
                        })
                        .unwrap()
                });
                for i in 0..2 {
                    let lock = JointAxis::ALL[locked[i] + 3];
                    if kind == JointType::D6 {
                        scene
                            .set_joint_locked_axes(joints[i], JointAxes::LIN.with(lock))
                            .unwrap();
                    } else {
                        scene
                            .set_joint_limits(joints[i], lock, Some([0., 0.]))
                            .unwrap();
                    }
                    scene
                        .set_joint_motor(
                            joints[i],
                            JointAxis::ALL[driven[i] + 3],
                            JointMotor {
                                mode: JointTargetMode::Velocity,
                                model: models[i],
                                target_velocity: TARGETS[i],
                                damping: GAINS[i],
                                ..Default::default()
                            },
                        )
                        .unwrap();
                }
                let rows = |indices: [usize; 2]| {
                    let root = world_frames[0] * axes[indices[0]];
                    let child = world_frames[1] * axes[indices[1]];
                    [[root, Vec3::ZERO], [-child, child]]
                };
                cases.push(Case {
                    kind,
                    models,
                    locked,
                    driven,
                    bodies,
                    joints,
                    parent_frames,
                    child_frames,
                    pivot_inertia,
                    lock_rows: rows(locked),
                    drive_rows: rows(driven),
                });
            }
        }
    }
    assert_eq!(cases.len(), 24);
    assert_eq!(scene.model().articulation_count, 24);
    (scene, cases)
}

fn rotation_error(a: Quat, b: Quat) -> f64 {
    let q = a.normalize() * b.normalize().conjugate();
    2. * Vec3::new(q.x, q.y, q.z).length().atan2(q.w.abs())
}

fn certify(
    world: &mut RigidWorld,
    cases: &[Case],
    oracle_tolerance: f64,
    reference: Option<&Trace>,
) -> Trace {
    world.solver.substeps = 1;
    world.solver.stable_motor_feedback = true;
    world.set_reaction_readout(true).unwrap();
    let mut trace = Vec::new();
    let mut oracle_peak: f64 = 0.;
    let mut anchor_peak: f64 = 0.;
    let mut lock_peak: f64 = 0.;
    let mut parity_peak: f64 = 0.;
    let mut passive_peak = vec![0_f64; cases.len()];
    #[cfg(feature = "gpu")]
    let mut compilations = None;
    for frame in 0..=FRAMES {
        if matches!(frame, 50 | 75 | 90) {
            for case in cases {
                for i in 0..2 {
                    world
                        .scene
                        .set_joint_motor(
                            case.joints[i],
                            JointAxis::ALL[case.driven[i] + 3],
                            JointMotor {
                                mode: if frame == 75 {
                                    JointTargetMode::None
                                } else {
                                    JointTargetMode::Velocity
                                },
                                model: case.models[i],
                                target_velocity: [-0.15, 0.13][i],
                                damping: GAINS[i],
                                ..Default::default()
                            },
                        )
                        .unwrap();
                }
            }
        }
        let dt = if frame == 0 { INITIAL_DT } else { 0.01 };
        world
            .step(dt)
            .unwrap_or_else(|error| panic!("plane frame={frame}: {error}"));
        assert_eq!(world.completed_steps(), frame as u64 + 1);
        assert!(world.quarantined_bodies().is_empty() && world.joint_break_events().is_empty());
        let state = world.scene.state();
        assert!(
            state
                .body_f
                .host()
                .unwrap()
                .iter()
                .all(|v| v.linear.is_finite() && v.angular.is_finite())
        );
        assert!(state.joint_q.host().unwrap().iter().all(|v| v.is_finite()));
        assert!(state.joint_qd.host().unwrap().iter().all(|v| v.is_finite()));
        assert_eq!(
            world.joint_reaction_samples().unwrap().len(),
            cases.len() * 2
        );
        let mut row = Vec::new();
        for (c, case) in cases.iter().enumerate() {
            let expected = (frame == 0).then(|| initial_oracle(case));
            if let Some(expected) = expected {
                let momentum_rate = (case.pivot_inertia[0] * expected[0]
                    + case.pivot_inertia[1] * expected[1])
                    / INITIAL_DT;
                let measured = world.joint_reaction(case.joints[0]).unwrap().child.angular;
                let error = (measured - momentum_rate).length() / momentum_rate.length().max(1e-6);
                assert!(
                    error <= 0.01,
                    "case={c} initial support torque oracle={error:e} measured={measured:?} expected={momentum_rate:?}"
                );
            }
            let root_rotation = world.scene.body_pose(case.bodies[0]).unwrap().rotation;
            let mut entries = [(Transform::IDENTITY, SpatialVector::ZERO); 2];
            for (i, entry) in entries.iter_mut().enumerate() {
                let pose = world.scene.body_pose(case.bodies[i]).unwrap();
                let velocity = world.scene.body_velocity(case.bodies[i]).unwrap();
                assert!(pose.position.is_finite() && pose.rotation.is_finite());
                assert!(velocity.linear.is_finite() && velocity.angular.is_finite());
                for wrench in [
                    world.joint_reaction(case.joints[i]).unwrap().parent,
                    world.joint_reaction(case.joints[i]).unwrap().child,
                ] {
                    assert!(wrench.linear.is_finite() && wrench.angular.is_finite());
                }
                let parent = if i == 0 {
                    case.parent_frames[0]
                } else {
                    root_rotation * case.parent_frames[1]
                };
                let relative = parent.conjugate() * pose.rotation * case.child_frames[i];
                let (x, y, z) = relative.to_euler(glam::EulerRot::XYZ);
                let error = [x, y, z][case.locked[i]].abs();
                lock_peak = lock_peak.max(error);
                anchor_peak = anchor_peak.max(pose.position.length());
                assert!(
                    error <= 1e-3 && pose.position.length() <= 1e-3,
                    "kind={:?} models={:?} case={c} frame={frame} body={i} lock={error:e} position={:?}",
                    case.kind,
                    case.models,
                    pose.position
                );
                if let Some(expected) = expected {
                    let error = (velocity.angular - expected[i]).length();
                    oracle_peak = oracle_peak.max(error);
                    assert!(
                        error <= oracle_tolerance,
                        "kind={:?} models={:?} case={c} body={i} oracle={error:e} actual={:?} expected={:?}",
                        case.kind,
                        case.models,
                        velocity.angular,
                        expected[i]
                    );
                }
                if let Some(reference) = reference {
                    let (cpu_pose, cpu_velocity) = reference[frame][c][i];
                    let parity = rotation_error(pose.rotation, cpu_pose.rotation)
                        .max((pose.position - cpu_pose.position).length())
                        .max((velocity.angular - cpu_velocity.angular).length())
                        .max((velocity.linear - cpu_velocity.linear).length());
                    parity_peak = parity_peak.max(parity);
                    assert!(
                        parity <= 1e-3,
                        "case={c} frame={frame} body={i} parity={parity:e}"
                    );
                }
                let dense = world.scene.joint_index(case.joints[i]).unwrap().index();
                let start = world.scene.model().joint_dof_offset.host().unwrap()[dense] as usize;
                let angular = if case.kind == JointType::D6 { 3 } else { 0 };
                let passive = (0..3)
                    .find(|&axis| axis != case.locked[i] && axis != case.driven[i])
                    .unwrap();
                passive_peak[c] = passive_peak[c]
                    .max(state.joint_qd.host().unwrap()[start + angular + passive].abs());
                *entry = (pose, velocity);
            }
            row.push(entries);
        }
        #[cfg(feature = "gpu")]
        if let Some(d) = world.gpu_diagnostics() {
            assert_eq!(
                *compilations.get_or_insert(d.model_compilations),
                d.model_compilations
            );
        }
        trace.push(row);
    }
    assert!(
        passive_peak.iter().all(|&v| v > 1e-7),
        "two-free-axis motion not exercised: {passive_peak:?}"
    );
    println!(
        "locked_drive_planes cases={} frames=101 seconds=1.005 oracle_peak={oracle_peak:e} anchor_peak={anchor_peak:e} lock_peak={lock_peak:e} parity_peak={parity_peak:e}",
        cases.len()
    );
    trace
}

#[test]
fn cpu_noncoaxial_two_free_axis_chains_match_projected_initial_mobility() {
    let (scene, cases) = fixture();
    certify(
        &mut RigidWorld::from_scene(scene).unwrap(),
        &cases,
        1e-6,
        None,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_noncoaxial_two_free_axis_chains_match_initial_oracle_and_cpu() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let (scene, cases) = fixture();
    let reference = certify(
        &mut RigidWorld::from_scene(scene).unwrap(),
        &cases,
        1e-6,
        None,
    );
    let (scene, cases) = fixture();
    let begin = std::time::Instant::now();
    println!("locked_drive_planes_prepare_begin");
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    println!(
        "locked_drive_planes_prepare_complete seconds={:.6}",
        begin.elapsed().as_secs_f64()
    );
    certify(&mut world, &cases, 1e-3, Some(&reference));
}
