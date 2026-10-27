use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, JointSoftness, MotorModel};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, JointHandle, RigidScene};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn body(scene: &mut RigidScene, pose: Transform, mass: f64) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(pose),
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * mass,
            }),
            ..Default::default()
        })
        .unwrap()
}

#[derive(Debug)]
struct Case {
    body: BodyHandle,
    joint: JointHandle,
    kind: JointType,
    origin: Vec3,
    rotation: Quat,
    inertia: f64,
    motor: JointMotor,
    brake: f64,
    position: f64,
    velocity: f64,
}

impl Case {
    fn axis(&self) -> JointAxis {
        if self.kind == JointType::Prismatic {
            JointAxis::LinX
        } else {
            JointAxis::AngX
        }
    }

    fn advance(&mut self, h: f64) -> f64 {
        let mobility = 1.0 / self.inertia;
        let brake_factor = 1.0 + h * self.brake * mobility;
        let g = mobility / brake_factor;
        let free_rate = self.velocity / brake_factor;
        let omega = std::f64::consts::TAU * 4.0;
        let k = self.inertia * omega * omega;
        let c = self.inertia * 1.2 * omega;
        let coordinate = self.position + 0.03;
        let a = g + 1.0 / (h * (c + h * k));
        let b = -free_rate - k * coordinate / (c + h * k);
        let motor = self.motor;
        let motor_impulse = match motor.mode {
            JointTargetMode::None => 0.0,
            JointTargetMode::Effort => {
                h * motor
                    .target_position
                    .clamp(-motor.max_force, motor.max_force)
            }
            _ => {
                let scale = if motor.model == MotorModel::Acceleration {
                    self.inertia
                } else {
                    1.0
                };
                let stiffness = if motor.mode == JointTargetMode::Velocity {
                    0.0
                } else {
                    motor.stiffness * scale
                };
                let damping = motor.damping * scale;
                let target_rate = if matches!(
                    motor.mode,
                    JointTargetMode::Velocity | JointTargetMode::PositionVelocity
                ) {
                    motor.target_velocity
                } else {
                    0.0
                };
                let error = if motor.mode == JointTargetMode::Velocity {
                    0.0
                } else {
                    coordinate - motor.target_position
                };
                let diagonal = g + 1.0 / (h * (damping + h * stiffness));
                let rhs = -free_rate
                    + (damping * target_rate - stiffness * error) / (damping + h * stiffness);
                ((rhs * a - g * b) / (a * diagonal - g * g))
                    .clamp(-h * motor.max_force, h * motor.max_force)
            }
        };
        let spring_impulse = (b - g * motor_impulse) / a;
        let total =
            (spring_impulse + motor_impulse - h * self.brake * self.velocity) / brake_factor;
        self.velocity += total * mobility + 0.2 * h;
        self.position += self.velocity * h;
        total / h
    }
}

fn fixture(focused: bool) -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    for kind in [JointType::Prismatic, JointType::Revolute] {
        for mass in [0.01, 1.0, 100.0] {
            for model in [MotorModel::Force, MotorModel::Acceleration] {
                for mode in [
                    JointTargetMode::Position,
                    JointTargetMode::Velocity,
                    JointTargetMode::PositionVelocity,
                    JointTargetMode::Effort,
                ] {
                    for max_force in [0.0, 0.25, f64::INFINITY] {
                        for rotated in [false, true] {
                            if focused && (kind != JointType::Revolute || mass != 100.0 || !rotated)
                            {
                                continue;
                            }
                            let rotation = if rotated {
                                Quat::from_rotation_z(0.7) * Quat::from_rotation_y(-0.4)
                            } else {
                                Quat::IDENTITY
                            };
                            let origin = Vec3::new(8.0 * cases.len() as f64, 2.0, -3.0);
                            let body = body(&mut scene, Transform::new(origin, rotation), mass);
                            scene
                                .insert_joint(JointDesc {
                                    parent: None,
                                    child: body,
                                    kind,
                                    axis: Vec3::X,
                                    frame_parent: Transform::new(origin, rotation),
                                    frame_child: Transform::IDENTITY,
                                })
                                .unwrap();
                            let frame = if kind == JointType::Prismatic {
                                Transform::new(origin - rotation * Vec3::X * 0.03, rotation)
                            } else {
                                Transform::new(origin, rotation * Quat::from_rotation_x(-0.03))
                            };
                            let joint = scene
                                .insert_loop_joint(
                                    JointDesc {
                                        parent: None,
                                        child: body,
                                        kind: JointType::D6,
                                        axis: Vec3::X,
                                        frame_parent: frame,
                                        frame_child: Transform::IDENTITY,
                                    },
                                    JointSoftness {
                                        natural_frequency: 4.0,
                                        damping_ratio: 0.6,
                                    },
                                )
                                .unwrap();
                            let axis = if kind == JointType::Prismatic {
                                JointAxis::LinX
                            } else {
                                JointAxis::AngX
                            };
                            scene
                                .set_joint_locked_axes(joint, JointAxes::NONE.with(axis))
                                .unwrap();
                            let motor = JointMotor {
                                mode,
                                model,
                                target_position: 0.4,
                                target_velocity: 0.7,
                                stiffness: if mode == JointTargetMode::Effort {
                                    0.0
                                } else {
                                    12.0
                                },
                                damping: if mode == JointTargetMode::Effort {
                                    0.0
                                } else {
                                    3.0
                                },
                                max_force,
                            };
                            scene.set_joint_motor(joint, axis, motor).unwrap();
                            let load = rotation * Vec3::X * (mass * 0.2);
                            scene
                                .add_body_wrench(
                                    body,
                                    if kind == JointType::Prismatic {
                                        SpatialVector::new(load, Vec3::ZERO)
                                    } else {
                                        SpatialVector::new(Vec3::ZERO, load)
                                    },
                                )
                                .unwrap();
                            cases.push(Case {
                                body,
                                joint,
                                kind,
                                origin,
                                rotation,
                                inertia: mass,
                                motor,
                                brake: 0.0,
                                position: 0.0,
                                velocity: 0.0,
                            });
                        }
                    }
                }
            }
        }
    }
    (scene, cases)
}

fn certify(make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    certify_selected(make_world, tolerance, false, false);
}

fn certify_selected(
    mut make_world: impl FnMut(RigidScene) -> RigidWorld,
    tolerance: f64,
    focused: bool,
    brakes: bool,
) {
    let rates: &[i32] = if focused { &[240] } else { &[60, 120, 240] };
    let steps: &[usize] = if focused { &[1] } else { &[1, 4] };
    for &hz in rates {
        for &substeps in steps {
            let (scene, mut cases) = fixture(focused);
            let mut world = make_world(scene);
            world.solver.substeps = substeps;
            world.set_reaction_readout(true).unwrap();
            #[cfg(feature = "gpu")]
            let mut cached = None;
            for frame in 0..12 {
                if focused && frame >= 10 {
                    for case in cases.iter().take(3) {
                        let (tree, _) = world
                            .scene
                            .joints()
                            .find(|(_, joint)| joint.child == case.body && joint.kind == case.kind)
                            .unwrap();
                        let tree = world.scene.joint_index(tree).unwrap().index();
                        let coord =
                            world.scene.model().joint_coord_offset.host().unwrap()[tree] as usize;
                        let dof =
                            world.scene.model().joint_dof_offset.host().unwrap()[tree] as usize;
                        eprintln!(
                            "angular predictor frame={frame} cap={} expected_q={} actual_q={} expected_rate={} actual_rate={} pose={:?} velocity={:?}",
                            case.motor.max_force,
                            case.position,
                            world.scene.state().joint_q.host().unwrap()[coord],
                            case.velocity,
                            world.scene.state().joint_qd.host().unwrap()[dof],
                            world.scene.body_pose(case.body).unwrap(),
                            world.scene.body_velocity(case.body).unwrap()
                        );
                    }
                }
                for case in &mut cases {
                    if brakes {
                        case.brake = case.inertia
                            * match frame {
                                0..4 => 0.0,
                                4..8 => 12.0,
                                8..10 => 0.7,
                                _ => 0.0,
                            };
                        let model = world.scene.model_stamp();
                        let pose = world.scene.body_pose(case.body).unwrap();
                        world
                            .scene
                            .set_joint_brake(case.joint, case.axis(), case.brake)
                            .unwrap();
                        assert_eq!(world.scene.model_stamp(), model);
                        assert_eq!(world.scene.body_pose(case.body).unwrap(), pose);
                    }
                    if frame == 4 {
                        case.motor.target_position = -0.1;
                        case.motor.target_velocity = -0.2;
                        case.motor.stiffness *= 1.5;
                        case.motor.damping *= 1.5;
                    }
                    if frame == 8 {
                        case.motor.mode = JointTargetMode::None;
                    }
                    if frame == 4 || frame == 8 {
                        world
                            .scene
                            .set_joint_motor(case.joint, case.axis(), case.motor)
                            .unwrap();
                    }
                }
                let mut forces = vec![0.0; cases.len()];
                for _ in 0..substeps {
                    for (case, force) in cases.iter_mut().zip(&mut forces) {
                        *force +=
                            case.advance(1.0 / (hz as f64 * substeps as f64)) / substeps as f64;
                    }
                }
                world.step(1.0 / hz as f64).unwrap_or_else(|error| {
                    panic!("hz={hz} substeps={substeps} frame={frame}: {error}")
                });
                assert!(world.quarantined_bodies().is_empty());
                #[cfg(feature = "gpu")]
                if let Some(diagnostics) = world.gpu_diagnostics() {
                    let counts = (
                        diagnostics.model_compilations,
                        diagnostics.current_model_bind_groups_created,
                    );
                    assert_eq!(
                        *cached.get_or_insert(counts),
                        counts,
                        "live compliant controls rebuilt GPU state"
                    );
                }
                for (case, force) in cases.iter().zip(forces) {
                    let pose = world.scene.body_pose(case.body).unwrap();
                    let velocity = world.scene.body_velocity(case.body).unwrap();
                    let direction = case.rotation * Vec3::X;
                    let reaction = world.joint_reaction(case.joint).unwrap();
                    let (rate, load, unwanted) = if case.kind == JointType::Prismatic {
                        assert!(
                            (pose.position - case.origin - direction * case.position).length()
                                < tolerance,
                            "position hz={hz} substeps={substeps} frame={frame} case={case:?} actual={pose:?}"
                        );
                        assert!(
                            (case.rotation.conjugate() * pose.rotation)
                                .normalize()
                                .to_scaled_axis()
                                .length()
                                < tolerance
                        );
                        (
                            velocity.linear,
                            reaction.child.linear,
                            reaction.child.angular,
                        )
                    } else {
                        assert!((pose.position - case.origin).length() < tolerance);
                        let expected = case.rotation * Quat::from_rotation_x(case.position);
                        let rotation_error = (expected.conjugate() * pose.rotation)
                            .normalize()
                            .to_scaled_axis()
                            .length();
                        assert!(
                            rotation_error < tolerance,
                            "rotation hz={hz} substeps={substeps} frame={frame} case={case:?} actual={pose:?} error={rotation_error}"
                        );
                        (
                            velocity.angular,
                            reaction.child.angular,
                            reaction.child.linear,
                        )
                    };
                    assert!(
                        (rate - direction * case.velocity).length() < tolerance,
                        "rate hz={hz} substeps={substeps} frame={frame} case={case:?} actual={rate:?}"
                    );
                    assert!(
                        (load - direction * force).length() < tolerance * (1.0 + force.abs()),
                        "load hz={hz} substeps={substeps} frame={frame} case={case:?} expected={force} actual={load:?} pose={pose:?} velocity={velocity:?}"
                    );
                    assert!(unwanted.length() < tolerance);
                    assert!((reaction.parent.linear + reaction.child.linear).length() < tolerance);
                    assert!(
                        (reaction.parent.angular + reaction.child.angular).length() < tolerance
                    );
                }
            }
            eprintln!(
                "loop controls hz={hz} substeps={substeps} brakes={brakes}: {} cases passed",
                cases.len()
            );
        }
    }
}

#[test]
fn cpu_compliant_loop_controls_match_coupled_linear_angular_oracle() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[test]
fn cpu_compliant_loop_brakes_are_independent_of_motor_caps_and_springs() {
    certify_selected(
        |scene| RigidWorld::from_scene(scene).unwrap(),
        1e-8,
        false,
        true,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_compliant_loop_brakes_are_independent_of_motor_caps_and_springs() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify_selected(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-4,
        false,
        true,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_compliant_loop_controls_match_coupled_linear_angular_oracle() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-4,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_rotated_high_inertia_compliant_loop_matches_near_rest_oracle() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify_selected(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-4,
        true,
        false,
    );
}

fn certify_singular(
    mut make_world: impl FnMut(RigidScene) -> RigidWorld,
    tolerance: f64,
    capture: bool,
) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    for scale in [1e-6, 1.0, 1e6] {
        for weak in [0.0, 1e-10, 1e-8, 1.0] {
            let origin = Vec3::new(8.0 * cases.len() as f64, 2.0, -3.0);
            let inertia = scale * weak;
            let body = scene
                .insert_body(BodyDesc {
                    params: BodyParams::new().with_pose(Transform::from_translation(origin)),
                    additional_mass: Some(MassProperties {
                        mass: scale,
                        local_com: Vec3::ZERO,
                        inertia: Mat3::from_diagonal(Vec3::new(inertia, scale, scale)),
                    }),
                    ..Default::default()
                })
                .unwrap();
            let tree = scene
                .insert_joint(JointDesc {
                    parent: None,
                    child: body,
                    kind: JointType::D6,
                    axis: Vec3::X,
                    frame_parent: Transform::from_translation(origin),
                    frame_child: Transform::IDENTITY,
                })
                .unwrap();
            scene
                .set_joint_locked_axes(tree, JointAxes::NONE.with(JointAxis::AngZ))
                .unwrap();
            let joint = scene
                .insert_loop_joint(
                    JointDesc {
                        parent: None,
                        child: body,
                        kind: JointType::D6,
                        axis: Vec3::X,
                        frame_parent: Transform::new(origin, Quat::from_rotation_x(-0.03)),
                        frame_child: Transform::IDENTITY,
                    },
                    JointSoftness {
                        natural_frequency: 4.0,
                        damping_ratio: 0.6,
                    },
                )
                .unwrap();
            scene
                .set_joint_locked_axes(joint, JointAxes::NONE.with(JointAxis::AngX))
                .unwrap();
            scene
                .add_body_wrench(
                    body,
                    SpatialVector::new(Vec3::ZERO, Vec3::X * (0.2 * inertia)),
                )
                .unwrap();
            cases.push(Case {
                body,
                joint,
                kind: JointType::Revolute,
                origin,
                rotation: Quat::IDENTITY,
                inertia,
                motor: JointMotor::default(),
                brake: 0.0,
                position: 0.0,
                velocity: 0.0,
            });
        }
    }
    let mut world = make_world(scene);
    world.solver.substeps = 4;
    world.set_reaction_readout(capture).unwrap();
    for frame in 0..12 {
        let mut forces = vec![0.0; cases.len()];
        for _ in 0..4 {
            for (case, force) in cases.iter_mut().zip(&mut forces) {
                if case.inertia != 0.0 {
                    let previous = case.position;
                    *force += case.advance(1.0 / 240.0) / 4.0;
                    case.position = previous + 2.0 * (case.velocity / 480.0).atan();
                }
            }
        }
        world.step(1.0 / 60.0).unwrap();
        for (case, force) in cases.iter().zip(forces) {
            let pose = world.scene.body_pose(case.body).unwrap();
            let velocity = world.scene.body_velocity(case.body).unwrap();
            assert!((pose.position - case.origin).length() < tolerance);
            let relative = Quat::from_rotation_x(-case.position) * pose.rotation.normalize();
            assert!(
                Vec3::new(relative.x, relative.y, relative.z).length() * 2.0 < tolerance,
                "singular rotation frame={frame} case={case:?} pose={pose:?}"
            );
            assert!(
                (velocity.angular - Vec3::X * case.velocity).length() < tolerance,
                "singular rate frame={frame} case={case:?} velocity={velocity:?}"
            );
            assert!(velocity.linear.length() < tolerance);
            if !capture {
                continue;
            }
            let reaction = world.joint_reaction(case.joint).unwrap();
            let scale = if case.inertia == 0.0 {
                1.0
            } else {
                case.inertia
            };
            assert!(
                (reaction.child.angular - Vec3::X * force).length() / scale
                    < tolerance * (1.0 + (force / scale).abs()),
                "singular reaction frame={frame} case={case:?} force={force} reaction={reaction:?}"
            );
            assert!(reaction.child.linear.length() < tolerance);
            assert!(
                (reaction.parent.angular + reaction.child.angular).length() / scale < tolerance
            );
        }
        assert!(world.quarantined_bodies().is_empty());
    }
}

#[test]
fn cpu_compliant_loops_preserve_weak_modes_beside_null_modes() {
    for capture in [false, true] {
        certify_singular(
            |scene| RigidWorld::from_scene(scene).unwrap(),
            1e-8,
            capture,
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn gpu_compliant_loops_preserve_weak_modes_beside_null_modes() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    for capture in [false, true] {
        certify_singular(
            |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
            2e-4,
            capture,
        );
    }
}
