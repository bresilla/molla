use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, JointHandle, RigidScene};
use molla_sim::{JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

#[derive(Debug)]
struct Case {
    kind: JointType,
    bodies: [BodyHandle; 2],
    joints: [JointHandle; 2],
    masses: [f64; 2],
    models: [MotorModel; 2],
    cap: f64,
    loads: [f64; 2],
    q: [f64; 2],
    v: [f64; 2],
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    for kind in [JointType::Prismatic, JointType::Revolute] {
        for parent_mass in [0.01, 1.0, 100.0] {
            for models in [
                [MotorModel::Force, MotorModel::Acceleration],
                [MotorModel::Acceleration, MotorModel::Force],
                [MotorModel::Acceleration, MotorModel::Acceleration],
            ] {
                for cap in [0.2, f64::INFINITY] {
                    for loaded in [false, true] {
                        let masses = [parent_mass, 3.0];
                        let bodies = masses.map(|mass| {
                            scene
                                .insert_body(BodyDesc {
                                    additional_mass: Some(MassProperties {
                                        mass,
                                        local_com: Vec3::ZERO,
                                        inertia: Mat3::IDENTITY * mass,
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
                                    frame_parent: Transform::IDENTITY,
                                    frame_child: Transform::IDENTITY,
                                })
                                .unwrap()
                        });
                        let loads = if loaded {
                            [0.3 * masses[0], -0.4 * masses[1]]
                        } else {
                            [0.0; 2]
                        };
                        for i in 0..2 {
                            scene
                                .set_joint_motor(
                                    joints[i],
                                    if kind == JointType::Prismatic {
                                        JointAxis::LinX
                                    } else {
                                        JointAxis::AngX
                                    },
                                    JointMotor {
                                        mode: JointTargetMode::Velocity,
                                        model: models[i],
                                        target_velocity: [0.7, -0.5][i],
                                        damping: [3.0, 7.0][i],
                                        max_force: cap,
                                        ..Default::default()
                                    },
                                )
                                .unwrap();
                            scene
                                .add_body_wrench(
                                    bodies[i],
                                    if kind == JointType::Prismatic {
                                        SpatialVector::new(Vec3::X * loads[i], Vec3::ZERO)
                                    } else {
                                        SpatialVector::new(Vec3::ZERO, Vec3::X * loads[i])
                                    },
                                )
                                .unwrap();
                        }
                        cases.push(Case {
                            kind,
                            bodies,
                            joints,
                            masses,
                            models,
                            cap,
                            loads,
                            q: [0.0; 2],
                            v: [0.0; 2],
                        });
                    }
                }
            }
        }
    }
    (scene, cases)
}

fn advance(case: &mut Case, h: f64, stable: bool) -> [f64; 2] {
    let inv_parent = 1.0 / case.masses[0];
    let inverse = [
        [inv_parent, -inv_parent],
        [-inv_parent, inv_parent + 1.0 / case.masses[1]],
    ];
    let load_acceleration = [
        case.loads[0] / case.masses[0],
        case.loads[1] / case.masses[1] - case.loads[0] / case.masses[0],
    ];
    let gains = [3.0, 7.0];
    let targets = [0.7, -0.5];
    let mut torques = [0.0; 2];
    for i in 0..2 {
        let acceleration = case.models[i] == MotorModel::Acceleration;
        if stable && !acceleration {
            continue;
        }
        let scale = if acceleration {
            1.0 / inverse[i][i]
        } else {
            1.0
        };
        let denominator = if stable { 1.0 + gains[i] * h } else { 1.0 };
        torques[i] =
            (scale * gains[i] * (targets[i] - case.v[i]) / denominator).clamp(-case.cap, case.cap);
    }
    if stable {
        for i in 0..2 {
            if case.models[i] != MotorModel::Force {
                continue;
            }
            let other = 1 - i;
            let unactuated = load_acceleration[i] + inverse[i][other] * torques[other];
            torques[i] = (gains[i] * (targets[i] - case.v[i] - h * unactuated)
                / (1.0 + h * gains[i] * inverse[i][i]))
                .clamp(-case.cap, case.cap);
        }
    }
    for i in 0..2 {
        case.v[i] +=
            h * (load_acceleration[i] + inverse[i][0] * torques[0] + inverse[i][1] * torques[1]);
        case.q[i] += h * case.v[i];
    }
    torques
}

fn feedback_modes() -> Vec<bool> {
    let modes = match std::env::var("PRIMITIVE_MOTOR_FEEDBACK") {
        Err(std::env::VarError::NotPresent) => vec![false, true],
        Ok(value) if value == "both" => vec![false, true],
        Ok(value) if value == "explicit" => vec![false],
        Ok(value) if value == "stable" => vec![true],
        value => panic!("PRIMITIVE_MOTOR_FEEDBACK must be both, explicit or stable: {value:?}"),
    };
    println!("coupled_motor_feedback_modes={modes:?}");
    modes
}

fn certify(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for stable in feedback_modes() {
        let (scene, mut cases) = fixture();
        println!(
            "coupled_motor_prepare_begin stable={stable} cases={}",
            cases.len()
        );
        let preparation = std::time::Instant::now();
        let mut world = make_world(scene);
        println!(
            "coupled_motor_prepare_complete stable={stable} seconds={:.6}",
            preparation.elapsed().as_secs_f64()
        );
        world.solver.substeps = 4;
        world.solver.stable_motor_feedback = stable;
        world.set_reaction_readout(true).unwrap();
        let dt = if stable { 0.02 } else { 0.0004 };
        let mut peak_coordinate: f64 = 0.0;
        let mut peak_rate: f64 = 0.0;
        let mut peak_reaction: f64 = 0.0;
        #[cfg(feature = "gpu")]
        let mut compilations = None;
        for frame in 0..40 {
            let mean_torques: Vec<_> = cases
                .iter_mut()
                .map(|case| {
                    let mut mean = [0.0; 2];
                    for _ in 0..4 {
                        let torque = advance(case, dt / 4.0, stable);
                        for i in 0..2 {
                            mean[i] += torque[i] / 4.0;
                        }
                    }
                    mean
                })
                .collect();
            world.step(dt).unwrap();
            assert!(world.quarantined_bodies().is_empty());
            assert!(world.joint_break_events().is_empty());
            assert_eq!(world.completed_steps(), frame + 1);
            #[cfg(feature = "gpu")]
            {
                let current = world.gpu_diagnostics().map(|d| d.model_compilations);
                assert_eq!(*compilations.get_or_insert(current), current);
            }
            for (case, mean) in cases.iter().zip(mean_torques) {
                for (i, torque) in mean.into_iter().enumerate() {
                    let joint = world.scene.joint_index(case.joints[i]).unwrap().index();
                    let c = world.scene.model().joint_coord_offset.host().unwrap()[joint] as usize;
                    let d = world.scene.model().joint_dof_offset.host().unwrap()[joint] as usize;
                    let q = world.scene.state().joint_q.host().unwrap()[c];
                    let v = world.scene.state().joint_qd.host().unwrap()[d];
                    peak_coordinate = peak_coordinate.max((q - case.q[i]).abs());
                    peak_rate = peak_rate.max((v - case.v[i]).abs());
                    assert!(
                        (q - case.q[i]).abs() < tolerance * (1.0 + case.q[i].abs())
                            && (v - case.v[i]).abs() < tolerance * (1.0 + case.v[i].abs()),
                        "frame={frame} stable={stable} axis={i} q={q} v={v}: {case:?}"
                    );
                    let pose = world.scene.body_pose(case.bodies[i]).unwrap();
                    let velocity = world.scene.body_velocity(case.bodies[i]).unwrap();
                    let q = case.q[..=i].iter().sum::<f64>();
                    let v = case.v[..=i].iter().sum::<f64>();
                    assert!(pose.position.is_finite() && pose.rotation.is_finite());
                    assert!(velocity.linear.is_finite() && velocity.angular.is_finite());
                    let (rate, other) = if case.kind == JointType::Prismatic {
                        assert!(
                            (pose.position - Vec3::X * q).length() < tolerance * (1.0 + q.abs())
                        );
                        (velocity.linear, velocity.angular)
                    } else {
                        let relative =
                            pose.rotation.normalize() * Quat::from_rotation_x(q).conjugate();
                        let error = 2.0 * relative.xyz().length().atan2(relative.w.abs());
                        assert!(
                            pose.position.length() < tolerance
                                && error < tolerance * (1.0 + q.abs())
                        );
                        (velocity.angular, velocity.linear)
                    };
                    assert!(
                        (rate - Vec3::X * v).length() + other.length()
                            < tolerance * (1.0 + v.abs())
                    );
                    let reaction = world.joint_reaction(case.joints[i]).unwrap();
                    assert!(
                        reaction.child.linear.is_finite() && reaction.child.angular.is_finite()
                    );
                    assert!(
                        reaction.parent.linear.is_finite() && reaction.parent.angular.is_finite()
                    );
                    let (actual, other) = if case.kind == JointType::Prismatic {
                        (reaction.child.linear, reaction.child.angular)
                    } else {
                        (reaction.child.angular, reaction.child.linear)
                    };
                    peak_reaction =
                        peak_reaction.max((actual - Vec3::X * torque).length() + other.length());
                    assert!(
                        (actual - Vec3::X * torque).length() + other.length()
                            < tolerance * (1.0 + torque.abs()),
                        "reaction frame={frame} axis={i} expected={torque} actual={reaction:?}: {case:?}"
                    );
                }
            }
        }
        println!(
            "coupled_motor_complete stable={stable} cases={} frames=40 dt={dt} substeps=4 peak_coordinate={peak_coordinate:e} peak_rate={peak_rate:e} peak_reaction={peak_reaction:e}",
            cases.len()
        );
    }
}

#[test]
fn coupled_motor_models_match_two_body_inverse_mass() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_coupled_motor_models_match_two_body_inverse_mass() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    certify(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-4,
    );
}
