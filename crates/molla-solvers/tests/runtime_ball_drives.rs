use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene, SphericalConeLimits};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

#[cfg(feature = "gpu")]
fn hardware_engine() -> std::sync::Arc<molla_compute::Engine> {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    engine
}

fn pivot_response(mut make: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let rotation = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(-0.4);
    let local_frame = Quat::from_rotation_y(0.7);
    let com = Vec3::new(0.12, 0.07, -0.05);
    let anchor = Vec3::new(0.3, -0.2, 0.1);
    let arm = rotation * (com - anchor);
    let tensor = Mat3::from_cols(
        Vec3::new(1.9, 0.23, -0.17),
        Vec3::new(0.23, 1.6, 0.11),
        Vec3::new(-0.17, 0.11, 1.4),
    );
    let outer = Mat3::from_cols(arm * arm.x, arm * arm.y, arm * arm.z);
    let h = 1. / 240.;
    let mut cases = Vec::new();
    for axis in [JointAxis::AngX, JointAxis::AngY, JointAxis::AngZ] {
        for mass in [0.01, 1., 100.] {
            for model in [MotorModel::Force, MotorModel::Acceleration] {
                for mode in [
                    JointTargetMode::None,
                    JointTargetMode::Position,
                    JointTargetMode::Velocity,
                    JointTargetMode::PositionVelocity,
                    JointTargetMode::Effort,
                ] {
                    for cap in [0., 0.2 * mass, f64::INFINITY] {
                        for brake in [0., 0.7 * mass] {
                            let position = Vec3::new(3. * cases.len() as f64, 0.5, -0.2);
                            let pose = Transform::new(position, rotation);
                            let body = scene
                                .insert_body(BodyDesc {
                                    params: BodyParams::new().with_pose(pose),
                                    additional_mass: Some(MassProperties {
                                        mass,
                                        local_com: com,
                                        inertia: tensor * mass,
                                    }),
                                    ..Default::default()
                                })
                                .unwrap();
                            let child_frame = Transform::new(anchor, local_frame);
                            let joint = scene
                                .insert_rigid_loop_joint(JointDesc {
                                    parent: None,
                                    child: body,
                                    kind: JointType::Ball,
                                    axis: Vec3::Y,
                                    frame_parent: pose.compose(&child_frame),
                                    frame_child: child_frame,
                                })
                                .unwrap();
                            let motor = JointMotor {
                                mode,
                                model,
                                target_position: 0.003,
                                target_velocity: 0.1,
                                stiffness: 30.,
                                damping: 3.,
                                max_force: cap,
                            };
                            scene.set_joint_motor(joint, axis, motor).unwrap();
                            scene.set_joint_brake(joint, axis, brake).unwrap();
                            let direction = rotation
                                * local_frame
                                * [Vec3::X, Vec3::Y, Vec3::Z][axis as usize % 3];
                            let inertia = Mat3::from_quat(rotation)
                                * tensor
                                * mass
                                * Mat3::from_quat(rotation).transpose();
                            let pivot =
                                inertia + (Mat3::IDENTITY * arm.length_squared() - outer) * mass;
                            let response = pivot.inverse() * direction;
                            let mobility = direction.dot(response);
                            let scale = if model == MotorModel::Acceleration {
                                1. / mobility
                            } else {
                                1.
                            };
                            let kp = if mode == JointTargetMode::Velocity {
                                0.
                            } else {
                                motor.stiffness * scale
                            };
                            let kd = motor.damping * scale;
                            let target_rate = if matches!(
                                mode,
                                JointTargetMode::Velocity | JointTargetMode::PositionVelocity
                            ) {
                                motor.target_velocity
                            } else {
                                0.
                            };
                            let beta = 1. + h * brake * mobility;
                            let impulse = match mode {
                                JointTargetMode::None => 0.,
                                JointTargetMode::Effort => {
                                    h * motor.target_position.clamp(-cap, cap)
                                }
                                _ => (h * (kp * motor.target_position + kd * target_rate)
                                    / (1. + h * (kd + h * kp) * mobility / beta))
                                    .clamp(-h * cap, h * cap),
                            } / beta;
                            let spin = response * impulse;
                            let linear = spin.cross(arm);
                            let next_rotation = (rotation
                                + Quat::from_xyzw(spin.x * 0.5, spin.y * 0.5, spin.z * 0.5, 0.)
                                    * rotation
                                    * h)
                                .normalize();
                            let next_inertia = Mat3::from_quat(next_rotation)
                                * tensor
                                * mass
                                * Mat3::from_quat(next_rotation).transpose();
                            let velocity = SpatialVector::new(
                                linear,
                                next_inertia.inverse() * (inertia * spin),
                            );
                            let pose = Transform::new(
                                position + rotation * com + linear * h - next_rotation * com,
                                next_rotation,
                            );
                            let force = linear * (mass / h);
                            let reaction = SpatialVector::new(force, direction * (impulse / h));
                            cases.push((
                                body, joint, mass, axis, model, mode, cap, brake, velocity, pose,
                                reaction,
                            ));
                        }
                    }
                }
            }
        }
    }
    let mut world = make(scene);
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    world.step(h).unwrap();
    assert!(world.quarantined_bodies().is_empty());
    assert!(world.joint_break_events().is_empty());
    assert_eq!(world.completed_steps(), 1);
    println!(
        "ball_pivot_response cases={} dt={h} substeps=1 tolerance={tolerance:e}",
        cases.len()
    );
    for (body, joint, mass, axis, model, mode, cap, brake, velocity, pose, reaction) in &cases {
        let label = format!("{mass} {axis:?} {model:?} {mode:?} cap={cap} brake={brake}");
        let actual = world.scene.body_velocity(*body).unwrap();
        assert!(
            (actual.linear - velocity.linear).length() < tolerance,
            "linear {label}: {actual:?} != {velocity:?}"
        );
        assert!(
            (actual.angular - velocity.angular).length() < tolerance,
            "angular {label}: {actual:?} != {velocity:?}"
        );
        let actual_pose = world.scene.body_pose(*body).unwrap();
        assert!(
            (actual_pose.position - pose.position).length() < tolerance,
            "position {label}"
        );
        assert!(
            (actual_pose.rotation - pose.rotation).length() < tolerance,
            "rotation {label}"
        );
        let reading = world.joint_reaction(*joint).unwrap();
        let bound = tolerance * (1. + mass + reaction.linear.length() + reaction.angular.length());
        assert!(
            (reading.child.linear - reaction.linear).length() < bound,
            "child force {label}: {:?} != {reaction:?}",
            reading.child
        );
        assert!(
            (reading.child.angular - reaction.angular).length() < bound,
            "child torque {label}: {:?} != {reaction:?}",
            reading.child
        );
        assert!(
            (reading.parent.linear + reaction.linear).length() < bound,
            "parent force {label}"
        );
        assert!(
            (reading.parent.angular + reaction.angular).length() < bound,
            "parent torque {label}"
        );
    }
    eprintln!(
        "{} full-tensor off-centre Ball closures: three authored-frame axes, all modes, Force/Acceleration, three masses/caps, brakes, independent inverse pivot tensor and anchor reactions",
        cases.len()
    );
}

fn cone_controls(mut make: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let rotation = Quat::from_rotation_y(0.6) * Quat::from_rotation_z(-0.4);
    let mut cases = Vec::new();
    for mass in [0.01, 1., 100.] {
        for model in [MotorModel::Force, MotorModel::Acceleration] {
            let position = Vec3::new(3. * cases.len() as f64, 0., 0.);
            let pose = Transform::new(position, rotation);
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
            let joint = scene
                .insert_rigid_loop_joint(JointDesc {
                    parent: None,
                    child: body,
                    kind: JointType::Ball,
                    axis: Vec3::X,
                    frame_parent: pose,
                    frame_child: Transform::IDENTITY,
                })
                .unwrap();
            scene
                .set_cone_limits(
                    joint,
                    SphericalConeLimits {
                        angle0: Some(0.),
                        angle1: Some(0.),
                    },
                )
                .unwrap();
            cases.push((body, joint, mass, model, pose, 0.));
        }
    }
    let mut world = make(scene);
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    let h = 1. / 240.;
    #[cfg(feature = "gpu")]
    let mut cached = None;
    for frame in 0..32 {
        for &(_, joint, mass, model, _, _) in &cases {
            let stamp = world.scene.model_stamp();
            let effort = frame < 16;
            world
                .scene
                .set_joint_motor(
                    joint,
                    JointAxis::AngX,
                    JointMotor {
                        mode: if effort {
                            JointTargetMode::Effort
                        } else {
                            JointTargetMode::Velocity
                        },
                        target_position: 0.2 * mass,
                        target_velocity: -0.1,
                        damping: if model == MotorModel::Acceleration {
                            3.
                        } else {
                            3. * mass
                        },
                        model,
                        ..Default::default()
                    },
                )
                .unwrap();
            world
                .scene
                .set_joint_motor(
                    joint,
                    JointAxis::AngY,
                    JointMotor {
                        mode: if effort {
                            JointTargetMode::Effort
                        } else {
                            JointTargetMode::Velocity
                        },
                        target_position: 0.3 * mass,
                        target_velocity: 0.1,
                        damping: if model == MotorModel::Acceleration {
                            3.
                        } else {
                            3. * mass
                        },
                        model,
                        ..Default::default()
                    },
                )
                .unwrap();
            world
                .scene
                .set_joint_brake(
                    joint,
                    JointAxis::AngX,
                    if frame < 8 { 0. } else { 0.7 * mass },
                )
                .unwrap();
            assert_eq!(world.scene.model_stamp(), stamp);
        }
        world.step(h).unwrap();
        #[cfg(feature = "gpu")]
        if let Some(diagnostics) = world.gpu_diagnostics() {
            let counts = (
                diagnostics.model_compilations,
                diagnostics.current_model_bind_groups_created,
            );
            assert_eq!(
                *cached.get_or_insert(counts),
                counts,
                "Ball control edits rebuilt GPU state"
            );
        }
        for (body, joint, mass, model, pose, rate) in &mut cases {
            let brake = if frame < 8 { 0. } else { 0.7 };
            let previous = *rate;
            let relative = rotation.conjugate() * pose.rotation;
            let angle = 2. * relative.x.atan2(relative.w);
            let half = 0.5 * angle;
            let transverse = if half == 0. { 1. } else { half / half.tan() };
            *rate = if frame < 16 {
                (previous + h * 0.2) / (1. + h * brake)
            } else {
                (previous - h * 3. * 0.1) / (1. + h * (3. + brake))
            };
            let spin = rotation * Vec3::X * *rate;
            pose.rotation = (pose.rotation
                + Quat::from_xyzw(spin.x * 0.5, spin.y * 0.5, spin.z * 0.5, 0.)
                    * pose.rotation
                    * h)
                .normalize();
            let label = format!("frame={frame} mass={mass} model={model:?}");
            let actual = world.scene.body_velocity(*body).unwrap();
            assert!(
                (actual.angular - spin).length() < tolerance,
                "rate {label}: {actual:?} != {spin:?}"
            );
            assert!(actual.linear.length() < tolerance, "anchor {label}");
            let actual_pose = world.scene.body_pose(*body).unwrap();
            assert!(
                (actual_pose.rotation - pose.rotation).length() < tolerance,
                "rotation {label}"
            );
            assert!(
                (actual_pose.position - pose.position).length() < tolerance,
                "position {label}"
            );
            let reaction = world.joint_reaction(*joint).unwrap();
            let torque = rotation * Vec3::X * (*mass * (*rate - previous) / h);
            let bound = tolerance * (1. + *mass);
            assert!(
                (reaction.child.angular - torque).length() < bound,
                "total reaction {label}: {:?} != {torque:?}",
                reaction.child
            );
            assert!(
                (reaction.parent.angular + torque).length() < bound,
                "parent reaction {label}"
            );
            let cone = world.cone_reading(*joint).unwrap();
            let cancelled = if frame < 16 || *model == MotorModel::Force {
                rotation * (Vec3::Y * transverse + Vec3::Z * half) * (-0.3 * *mass)
            } else {
                Vec3::ZERO
            };
            assert!(
                (cone.torque - cancelled).length() < bound,
                "cone reaction {label}: {:?} != {cancelled:?}",
                cone.torque
            );
        }
    }
    eprintln!(
        "32 live Ball cone/control frames, six rotated closures: blocked swing cancels Effort/Force PD and retires null-mobility Acceleration PD, free twist follows independent implicit brake/model law, setters retain layout and separate cone/total reactions"
    );
}

#[test]
fn cpu_ball_drives_use_full_pivot_response_and_physical_caps() {
    pivot_response(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

fn box_reference(matrix: Mat3, rhs: Vec3, caps: Vec3) -> Vec3 {
    for code in 0..27 {
        let mut status = [0; 3];
        let mut remaining = code;
        let mut fixed = Vec3::ZERO;
        let mut valid = true;
        for i in 0..3 {
            status[i] = remaining % 3;
            remaining /= 3;
            if status[i] != 0 {
                if !caps[i].is_finite() {
                    valid = false;
                    break;
                }
                fixed[i] = if status[i] == 1 { -caps[i] } else { caps[i] };
            }
        }
        if !valid {
            continue;
        }
        let mut coefficients = Mat3::IDENTITY;
        let mut target = fixed;
        for i in 0..3 {
            if status[i] == 0 {
                target[i] = rhs[i] - (matrix * fixed)[i];
                for j in 0..3 {
                    coefficients.col_mut(j)[i] = if status[j] == 0 { matrix.col(j)[i] } else { 0. };
                }
            }
        }
        let solution = coefficients.inverse() * target;
        let residual = matrix * solution - rhs;
        if (0..3).all(|i| {
            solution[i].abs() <= caps[i] + 1e-10
                && match status[i] {
                    0 => residual[i].abs() < 1e-9,
                    1 => residual[i] >= -1e-9,
                    _ => residual[i] <= 1e-9,
                }
        }) {
            return solution;
        }
    }
    panic!("independent three-axis box enumeration found no solution");
}

fn floating_controls(mut make: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let rotations = [
        Quat::from_rotation_x(0.4) * Quat::from_rotation_y(-0.7),
        Quat::from_rotation_z(-0.6) * Quat::from_rotation_y(0.3),
    ];
    let frame = Quat::from_rotation_y(0.8) * Quat::from_rotation_z(0.2);
    let basis = Mat3::from_quat(frame);
    let tensors = [
        Mat3::from_cols(
            Vec3::new(1.9, 0.23, -0.17),
            Vec3::new(0.23, 1.6, 0.11),
            Vec3::new(-0.17, 0.11, 1.4),
        ),
        Mat3::from_cols(
            Vec3::new(1.4, -0.16, 0.21),
            Vec3::new(-0.16, 1.8, -0.13),
            Vec3::new(0.21, -0.13, 1.7),
        ),
    ];
    let h = 1. / 240.;
    let mut cases = Vec::new();
    for mass in [0.01, 1., 100.] {
        for model in [MotorModel::Force, MotorModel::Acceleration] {
            for variant in 0..3 {
                let position = Vec3::new(3. * cases.len() as f64, 0.5, -0.2);
                let bodies: Vec<_> = (0..2)
                    .map(|i| {
                        scene
                            .insert_body(BodyDesc {
                                params: BodyParams::new()
                                    .with_pose(Transform::new(position, rotations[i])),
                                additional_mass: Some(MassProperties {
                                    mass: mass * (1. + 2. * i as f64),
                                    local_com: Vec3::ZERO,
                                    inertia: tensors[i] * mass * (1. + 2. * i as f64),
                                }),
                                ..Default::default()
                            })
                            .unwrap()
                    })
                    .collect();
                let inertias = [0, 1].map(|i| {
                    Mat3::from_quat(rotations[i])
                        * tensors[i]
                        * mass
                        * (1. + 2. * i as f64)
                        * Mat3::from_quat(rotations[i]).transpose()
                });
                let mobility =
                    basis.transpose() * (inertias[0].inverse() + inertias[1].inverse()) * basis;
                let brake = Vec3::new(0.7, 1.4, 2.1) * mass;
                let response = (mobility.inverse() + Mat3::from_diagonal(brake * h)).inverse();
                let joint = scene
                    .insert_rigid_loop_joint(JointDesc {
                        parent: Some(bodies[0]),
                        child: bodies[1],
                        kind: JointType::Ball,
                        axis: Vec3::Y,
                        frame_parent: Transform::from_rotation(rotations[0].conjugate() * frame),
                        frame_child: Transform::from_rotation(rotations[1].conjugate() * frame),
                    })
                    .unwrap();
                let mut diagonal = Vec3::ZERO;
                let mut desired = Vec3::ZERO;
                let mut caps = Vec3::ZERO;
                for (i, axis) in [JointAxis::AngX, JointAxis::AngY, JointAxis::AngZ]
                    .into_iter()
                    .enumerate()
                {
                    let kp = 30. + 10. * i as f64;
                    let kd = 3. + 0.5 * i as f64;
                    let target = [0.003, -0.002, 0.004][i];
                    let velocity = [0.1, -0.08, 0.12][i];
                    let cap = [0., 0.2 * mass, f64::INFINITY][(i + variant) % 3];
                    scene
                        .set_joint_motor(
                            joint,
                            axis,
                            JointMotor {
                                mode: JointTargetMode::PositionVelocity,
                                model,
                                stiffness: kp,
                                damping: kd,
                                target_position: target,
                                target_velocity: velocity,
                                max_force: cap,
                            },
                        )
                        .unwrap();
                    scene.set_joint_brake(joint, axis, brake[i]).unwrap();
                    let scale = if model == MotorModel::Acceleration {
                        1. / mobility.col(i)[i]
                    } else {
                        1.
                    };
                    diagonal[i] = 1. / (h * (kd + h * kp) * scale);
                    desired[i] = (kp * target + kd * velocity) / (kd + h * kp);
                    caps[i] = cap * h;
                }
                let motor = box_reference(response + Mat3::from_diagonal(diagonal), desired, caps);
                let rate = response * motor;
                let torque = basis * (motor - h * brake * rate) / h;
                let expected = [0, 1].map(|i| {
                    let spin = inertias[i].inverse() * torque * h * if i == 0 { -1. } else { 1. };
                    let rotation = (rotations[i]
                        + Quat::from_xyzw(spin.x * 0.5, spin.y * 0.5, spin.z * 0.5, 0.)
                            * rotations[i]
                            * h)
                        .normalize();
                    let next_inertia = Mat3::from_quat(rotation)
                        * tensors[i]
                        * mass
                        * (1. + 2. * i as f64)
                        * Mat3::from_quat(rotation).transpose();
                    (
                        Transform::new(position, rotation),
                        next_inertia.inverse() * (inertias[i] * spin),
                    )
                });
                cases.push((bodies, joint, mass, model, variant, expected, torque));
            }
        }
    }
    let mut world = make(scene);
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    world.step(h).unwrap();
    for (bodies, joint, mass, model, variant, expected, torque) in cases {
        let label = format!("mass={mass} model={model:?} cap rotation={variant}");
        for i in 0..2 {
            let actual = world.scene.body_velocity(bodies[i]).unwrap();
            let pose = world.scene.body_pose(bodies[i]).unwrap();
            assert!(
                actual.linear.length() < tolerance,
                "linear {label} side={i}"
            );
            assert!(
                (actual.angular - expected[i].1).length() < tolerance,
                "angular {label} side={i}: {actual:?} != {:?}",
                expected[i].1
            );
            assert!(
                (pose.position - expected[i].0.position).length() < tolerance,
                "position {label} side={i}"
            );
            assert!(
                (pose.rotation - expected[i].0.rotation).length() < tolerance,
                "rotation {label} side={i}"
            );
        }
        let reading = world.joint_reaction(joint).unwrap();
        let bound = tolerance * (1. + mass + torque.length());
        assert!(
            reading.child.linear.length() < bound && reading.parent.linear.length() < bound,
            "force {label}"
        );
        assert!(
            (reading.child.angular - torque).length() < bound,
            "child torque {label}: {:?} != {torque:?}",
            reading.child
        );
        assert!(
            (reading.parent.angular + torque).length() < bound,
            "parent torque {label}"
        );
    }
    eprintln!(
        "18 floating two-body full-inertia Ball pairs, simultaneous three-axis Force/Acceleration PD and brakes, mixed zero/finite/unbounded caps: independent box active-set enumeration, angular momentum and both-anchor reactions"
    );
}

#[test]
fn cpu_floating_ball_drives_match_independent_multiaxis_box_response() {
    floating_controls(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_floating_ball_drives_match_independent_multiaxis_box_response() {
    let engine = hardware_engine();
    floating_controls(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}

#[test]
fn cpu_ball_drives_and_brakes_share_cone_constraints_and_reactions() {
    cone_controls(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_ball_drives_use_full_pivot_response_and_physical_caps() {
    let engine = hardware_engine();
    pivot_response(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_ball_drives_and_brakes_share_cone_constraints_and_reactions() {
    let engine = hardware_engine();
    cone_controls(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}
