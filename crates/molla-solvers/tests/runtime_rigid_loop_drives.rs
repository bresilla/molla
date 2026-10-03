use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, MotorModel};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, JointHandle, RigidScene};
use molla_sim::{BodyParams, JointTargetMode, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

struct Case {
    body: BodyHandle,
    joint: JointHandle,
    axis: JointAxis,
    mass: f64,
    origin: Vec3,
    rotation: Quat,
    direction: Vec3,
    position: f64,
    velocity: f64,
    world_position: Vec3,
    world_velocity: Vec3,
    frame: Quat,
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    let rotation = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(-0.4);
    let authored = Vec3::new(1., 2., -3.).normalize();
    let alignment = Quat::from_rotation_arc(Vec3::X, authored);
    for (kind, axis) in [
        (JointType::Revolute, JointAxis::AngX),
        (JointType::Prismatic, JointAxis::LinX),
    ]
    .into_iter()
    .chain(JointAxis::ALL.map(|axis| (JointType::D6, axis)))
    {
        for mass in [0.01, 1., 100.] {
            let origin = Vec3::new(3. * cases.len() as f64, 0.5, -0.2);
            let direction = rotation * alignment * [Vec3::X, Vec3::Y, Vec3::Z][axis as usize % 3];
            let body = scene
                .insert_body(BodyDesc {
                    params: BodyParams::new().with_pose(Transform::new(origin, rotation)),
                    additional_mass: Some(MassProperties {
                        mass,
                        local_com: Vec3::ZERO,
                        inertia: Mat3::IDENTITY * mass,
                    }),
                    ..Default::default()
                })
                .unwrap();
            let velocity = if axis as usize >= 3 {
                SpatialVector::new(Vec3::ZERO, direction * 0.07)
            } else {
                SpatialVector::new(direction * 0.07, Vec3::ZERO)
            };
            scene.set_body_velocity(body, velocity).unwrap();
            let load = if axis as usize >= 3 {
                SpatialVector::new(Vec3::ZERO, direction * (mass * 0.1))
            } else {
                SpatialVector::new(direction * (mass * 0.1), Vec3::ZERO)
            };
            scene.add_body_wrench(body, load).unwrap();
            let joint = scene
                .insert_rigid_loop_joint(JointDesc {
                    parent: None,
                    child: body,
                    kind,
                    axis: authored,
                    frame_parent: Transform::new(
                        origin,
                        if kind == JointType::D6 {
                            rotation * alignment
                        } else {
                            rotation
                        },
                    ),
                    frame_child: Transform::new(
                        Vec3::ZERO,
                        if kind == JointType::D6 {
                            alignment
                        } else {
                            Quat::IDENTITY
                        },
                    ),
                })
                .unwrap();
            if kind == JointType::D6 {
                scene
                    .set_joint_locked_axes(joint, JointAxes(63 & !(1 << axis as u32)))
                    .unwrap();
            }
            scene
                .set_joint_limits(joint, axis, Some([-0.005, 0.007]))
                .unwrap();
            cases.push(Case {
                body,
                joint,
                axis,
                mass,
                origin,
                rotation,
                direction,
                position: 0.,
                velocity: 0.07,
                world_position: origin,
                world_velocity: velocity.linear,
                frame: if kind == JointType::D6 {
                    rotation * alignment
                } else {
                    (rotation * alignment).normalize()
                },
            });
        }
    }
    (scene, cases)
}

fn controls(frame: usize, mass: f64) -> (JointMotor, f64) {
    let mode = match frame {
        0..=15 => JointTargetMode::PositionVelocity,
        16..=23 => JointTargetMode::Velocity,
        24..=31 => JointTargetMode::Effort,
        32..=39 => JointTargetMode::None,
        40..=47 => JointTargetMode::Position,
        _ => JointTargetMode::None,
    };
    (
        JointMotor {
            mode,
            target_position: if mode == JointTargetMode::Effort {
                -0.4 * mass
            } else {
                -0.003
            },
            target_velocity: -0.02,
            stiffness: 30. * mass,
            damping: 3. * mass,
            max_force: 0.2 * mass,
            ..Default::default()
        },
        if (32..48).contains(&frame) {
            2. * mass
        } else {
            0.
        },
    )
}

fn certify(make: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    certify_model(make, tolerance, MotorModel::Force);
}

fn certify_model(
    mut make: impl FnMut(RigidScene) -> RigidWorld,
    tolerance: f64,
    model: MotorModel,
) {
    for (hz, substeps) in [(60, 1), (240, 4)] {
        let (scene, mut cases) = fixture();
        let mut world = make(scene);
        world.solver.substeps = substeps;
        world.set_reaction_readout(true).unwrap();
        let dt = 1. / hz as f64;
        let h = dt / substeps as f64;
        #[cfg(feature = "gpu")]
        let mut cached = None;
        for frame in 0..64 {
            let mut expected = Vec::new();
            for c in &mut cases {
                let (mut motor, brake) = controls(frame, c.mass);
                motor.model = if model == MotorModel::Acceleration && (16..24).contains(&frame) {
                    MotorModel::Force
                } else {
                    model
                };
                if motor.model == MotorModel::Acceleration {
                    motor.stiffness /= c.mass;
                    motor.damping /= c.mass;
                }
                let pose = world.scene.body_pose(c.body).unwrap();
                let layout = world.scene.model_stamp();
                world.scene.set_joint_motor(c.joint, c.axis, motor).unwrap();
                world.scene.set_joint_brake(c.joint, c.axis, brake).unwrap();
                assert_eq!(world.scene.body_pose(c.body).unwrap(), pose);
                assert_eq!(world.scene.model_stamp(), layout);
                let mut samples = Vec::new();
                for _ in 0..substeps {
                    let angular = c.axis as usize >= 3;
                    let (free, mobility, base, parallel, predicted) = if angular {
                        (
                            c.velocity + 0.1 * h,
                            1.0 / c.mass,
                            Vec3::ZERO,
                            c.direction,
                            Vec3::ZERO,
                        )
                    } else {
                        let basis = Mat3::from_cols(
                            c.frame * Vec3::X,
                            c.frame * Vec3::Y,
                            c.frame * Vec3::Z,
                        );
                        let inverse = basis.transpose().inverse();
                        let i = c.axis as usize;
                        let local = basis.transpose() * (c.world_position - c.origin);
                        c.position = local[i];
                        let mut locked = -local / h;
                        locked[i] = 0.0;
                        let base = inverse * locked;
                        let parallel = inverse.col(i);
                        let predicted = c.world_velocity + c.direction * (0.1 * h);
                        let square = parallel.length_squared();
                        (
                            parallel.dot(predicted - base) / square,
                            1.0 / (c.mass * square),
                            base,
                            parallel,
                            predicted,
                        )
                    };
                    let scale = if motor.model == MotorModel::Acceleration {
                        1. / mobility
                    } else {
                        1.
                    };
                    let kp = if matches!(
                        motor.mode,
                        JointTargetMode::Position | JointTargetMode::PositionVelocity
                    ) {
                        motor.stiffness * scale
                    } else {
                        0.
                    };
                    let kd = if matches!(
                        motor.mode,
                        JointTargetMode::Position
                            | JointTargetMode::Velocity
                            | JointTargetMode::PositionVelocity
                    ) {
                        motor.damping * scale
                    } else {
                        0.
                    };
                    let target = if matches!(
                        motor.mode,
                        JointTargetMode::Velocity | JointTargetMode::PositionVelocity
                    ) {
                        motor.target_velocity
                    } else {
                        0.
                    };
                    let gain = kd + h * kp;
                    let brake_factor = 1. + h * brake * mobility;
                    let effort = if motor.mode == JointTargetMode::Effort {
                        h * motor
                            .target_position
                            .clamp(-motor.max_force, motor.max_force)
                    } else {
                        (h * (kp * (motor.target_position - c.position) + kd * target
                            - gain * free / brake_factor)
                            / (1. + h * gain * mobility / brake_factor))
                            .clamp(-h * motor.max_force, h * motor.max_force)
                    };
                    let next = ((free + effort * mobility) / brake_factor)
                        .clamp((-0.005 - c.position) / h, (0.007 - c.position) / h);
                    samples.push(if angular {
                        c.direction * (c.mass * (next - free) / h)
                    } else {
                        c.world_velocity = base + parallel * next;
                        (c.world_velocity - predicted) * (c.mass / h)
                    });
                    c.velocity = next;
                    c.position += if c.axis as usize >= 3 {
                        2. * (0.5 * h * next).atan()
                    } else {
                        h * next
                    };
                    if !angular {
                        c.world_position += c.world_velocity * h;
                    }
                }
                expected.push(samples);
            }
            world
                .step(dt)
                .unwrap_or_else(|e| panic!("hz={hz} substeps={substeps} frame={frame}: {e}"));
            #[cfg(feature = "gpu")]
            if let Some(diagnostics) = world.gpu_diagnostics() {
                let counts = (
                    diagnostics.model_compilations,
                    diagnostics.current_model_bind_groups_created,
                );
                assert_eq!(
                    *cached.get_or_insert(counts),
                    counts,
                    "live model edits rebuilt GPU state"
                );
            }
            for (c, expected) in cases.iter().zip(&expected) {
                let angular = c.axis as usize >= 3;
                let pose = world.scene.body_pose(c.body).unwrap();
                let desired = if angular {
                    Transform::new(
                        c.origin,
                        Quat::from_axis_angle(c.direction, c.position) * c.rotation,
                    )
                } else {
                    Transform::new(c.world_position, c.rotation)
                };
                let label = format!(
                    "hz={hz} steps={substeps} frame={frame} axis={:?} mass={}",
                    c.axis, c.mass
                );
                assert!(
                    (pose.position - desired.position).length() < tolerance,
                    "position {label}: {pose:?} != {desired:?}"
                );
                let relative = (pose.rotation * desired.rotation.conjugate()).normalize();
                assert!(
                    Vec3::new(relative.x, relative.y, relative.z).length() < 0.5 * tolerance,
                    "rotation {label}"
                );
                let velocity = world.scene.body_velocity(c.body).unwrap();
                let expected_velocity = if angular {
                    SpatialVector::new(Vec3::ZERO, c.direction * c.velocity)
                } else {
                    SpatialVector::new(c.world_velocity, Vec3::ZERO)
                };
                assert!(
                    (velocity.linear - expected_velocity.linear).length() < tolerance,
                    "linear rate {label}"
                );
                assert!(
                    (velocity.angular - expected_velocity.angular).length() < tolerance,
                    "angular rate {label}"
                );
                let samples: Vec<_> = world
                    .joint_reaction_samples()
                    .unwrap()
                    .iter()
                    .filter(|s| s.joint == c.joint)
                    .collect();
                assert_eq!(samples.len(), substeps);
                for (sample, &force) in samples.iter().zip(expected) {
                    let child = if angular {
                        SpatialVector::new(Vec3::ZERO, force)
                    } else {
                        SpatialVector::new(force, Vec3::ZERO)
                    };
                    let bound = tolerance * (1. + c.mass + force.length());
                    assert!(
                        (sample.child.linear - child.linear).length() < bound,
                        "force {label}: {:?} != {child:?}",
                        sample.child
                    );
                    assert!(
                        (sample.child.angular - child.angular).length() < bound,
                        "torque {label}: {:?} != {child:?}",
                        sample.child
                    );
                    assert!(
                        (sample.parent.linear + child.linear).length() < bound,
                        "parent force {label}"
                    );
                    assert!(
                        (sample.parent.angular + child.angular).length() < bound,
                        "parent torque {label}"
                    );
                }
                let mean = expected.iter().sum::<Vec3>() / substeps as f64;
                let reading = world.joint_reaction(c.joint).unwrap();
                let actual = if angular {
                    reading.child.angular
                } else {
                    reading.child.linear
                };
                assert!(
                    (actual - mean).length() < tolerance * (1. + c.mass + mean.length()),
                    "mean {label}"
                );
            }
        }
        eprintln!(
            "rigid-loop {model:?} controls: 24 rotated hinge/slider/D6 closures, masses .01/1/100, all modes/brakes/caps/edits; {hz}Hz/{substeps} steps"
        );
    }
}

#[test]
fn cpu_rigid_loop_force_drives_match_independent_implicit_oracle() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[test]
fn cpu_rigid_loop_acceleration_drives_match_independent_implicit_oracle() {
    certify_model(
        |scene| RigidWorld::from_scene(scene).unwrap(),
        1e-8,
        MotorModel::Acceleration,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_rigid_loop_acceleration_drives_match_independent_implicit_oracle() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify_model(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
        MotorModel::Acceleration,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_rigid_loop_force_drives_match_independent_implicit_oracle() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}

fn contact_controls(frame: usize, mass: f64) -> (JointMotor, f64) {
    let motor = JointMotor {
        mode: match frame {
            0..=31 => JointTargetMode::Velocity,
            48..=63 => JointTargetMode::Effort,
            _ => JointTargetMode::None,
        },
        target_position: -20. * mass,
        target_velocity: 0.5,
        damping: 100. * mass,
        max_force: if frame < 16 { 2. * mass } else { 20. * mass },
        ..Default::default()
    };
    (
        motor,
        if (32..48).contains(&frame) || frame >= 64 {
            10. * mass
        } else {
            0.
        },
    )
}

fn contacts(make: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    contacts_model(make, tolerance, MotorModel::Force);
}

fn contacts_model(
    mut make: impl FnMut(RigidScene) -> RigidWorld,
    tolerance: f64,
    model: MotorModel,
) {
    use molla_sim::runtime::{ColliderDesc, ColliderGeometry};
    use molla_sim::{ColliderMass, ContactMaterial, FrictionCoefficients};
    use molla_solvers::penalty_contact::FrictionRegime;
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(-Vec3::Y * 10.).unwrap();
    let mut cases = Vec::new();
    for (i, mass) in [0.01, 1., 100.].into_iter().enumerate() {
        let origin = Vec3::new(i as f64 * 4., 0.5, 0.);
        let body = scene
            .insert_body(BodyDesc {
                params: BodyParams::new().with_position(origin),
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
                kind: JointType::D6,
                axis: Vec3::X,
                frame_parent: Transform::from_translation(origin),
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        scene
            .set_joint_locked_axes(joint, JointAxes::ANG.with(JointAxis::LinZ))
            .unwrap();
        let mut sphere = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
        sphere.parent = Some(body);
        sphere.mass = ColliderMass::Mass(0.);
        sphere.contact_material = ContactMaterial {
            stiffness: 2e4 * mass,
            damping: 200. * mass,
            friction: Some(FrictionCoefficients {
                mu_static: 0.6,
                mu_dynamic: 0.3,
            }),
            ..Default::default()
        };
        scene.insert_collider(sphere).unwrap();
        cases.push((body, joint, mass, origin, origin, Vec3::ZERO));
    }
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(100., 0.1, 10.),
    });
    ground.pose = Transform::from_translation(-Vec3::Y * 0.1);
    ground.contact_material = ContactMaterial {
        stiffness: 0.,
        damping: 0.,
        friction: Some(FrictionCoefficients {
            mu_static: 0.6,
            mu_dynamic: 0.3,
        }),
        ..Default::default()
    };
    scene.insert_collider(ground).unwrap();
    let mut world = make(scene);
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    let h = 1. / 240.;
    let mut regimes = [false; 2];
    for frame in 0..96 {
        let mut expected = Vec::new();
        for (body, joint, mass, _, position, velocity) in &mut cases {
            let (mut motor, brake) = contact_controls(frame, *mass);
            motor.model = model;
            if model == MotorModel::Acceleration {
                motor.damping /= *mass;
            }
            let pose = world.scene.body_pose(*body).unwrap();
            world
                .scene
                .set_joint_motor(*joint, JointAxis::LinX, motor)
                .unwrap();
            world
                .scene
                .set_joint_brake(*joint, JointAxis::LinX, brake)
                .unwrap();
            assert_eq!(world.scene.body_pose(*body).unwrap(), pose);
            let normal = (1e4 * *mass * (0.5 - position.y).max(0.)
                + 100. * *mass * (-velocity.y).max(0.))
                * h;
            let fixed = if motor.mode == JointTargetMode::Effort {
                h * motor
                    .target_position
                    .clamp(-motor.max_force, motor.max_force)
            } else {
                0.
            };
            let kd = if motor.mode == JointTargetMode::Velocity {
                motor.damping
                    * if model == MotorModel::Acceleration {
                        *mass
                    } else {
                        1.
                    }
            } else {
                0.
            };
            let at_rest = (h * kd * motor.target_velocity)
                .clamp(-h * motor.max_force, h * motor.max_force)
                + fixed;
            let stick = -(*mass * velocity.x + at_rest);
            let holding = if velocity.x.abs() > 1e-4 { 0.3 } else { 0.6 };
            let (next, control, friction) = if stick.abs() <= holding * normal {
                (0., at_rest, stick)
            } else {
                let sign = (*mass * velocity.x + at_rest).signum();
                let friction = -sign * 0.3 * normal;
                let free = velocity.x + friction / *mass;
                let b = 1. + h * brake / *mass;
                let motor_impulse = if motor.mode == JointTargetMode::Effort {
                    fixed
                } else {
                    (h * kd * (motor.target_velocity - free / b) / (1. + h * kd / (*mass * b)))
                        .clamp(-h * motor.max_force, h * motor.max_force)
                };
                let next = (free + motor_impulse / *mass) / b;
                (next, motor_impulse - h * brake * next, friction)
            };
            velocity.x = next;
            velocity.y += normal / *mass - 10. * h;
            *position += *velocity * h;
            expected.push((normal, control, friction));
        }
        world
            .step(h)
            .unwrap_or_else(|e| panic!("contact frame={frame}: {e}"));
        for ((body, joint, mass, _, position, velocity), &(normal, control, friction)) in
            cases.iter().zip(&expected)
        {
            let label = format!("contact frame={frame} mass={mass}");
            let pose = world.scene.body_pose(*body).unwrap();
            let actual = world.scene.body_velocity(*body).unwrap();
            assert!(
                (pose.position - *position).length() < tolerance,
                "position {label}: {:?} != {position:?}",
                pose.position
            );
            assert!(
                (actual.linear - *velocity).length() < tolerance,
                "rate {label}: {:?} != {velocity:?}",
                actual.linear
            );
            assert!(actual.angular.length() < tolerance, "angular {label}");
            let mut normal_sum = 0.;
            let mut tangent_sum = Vec3::ZERO;
            for pair in world
                .contact_pairs()
                .iter()
                .filter(|p| p.bodies.contains(&Some(*body)))
            {
                let sign = if pair.bodies[1] == Some(*body) {
                    1.
                } else {
                    -1.
                };
                for point in &pair.points {
                    normal_sum += point.normal_impulse;
                    tangent_sum += point.tangent_impulse * sign;
                    regimes[0] |= point.friction_regime == FrictionRegime::Sticking;
                    regimes[1] |= point.friction_regime == FrictionRegime::Sliding;
                }
            }
            let bound = tolerance * (1. + mass);
            assert!(
                (normal_sum - normal).abs() < bound,
                "normal {label}: {normal_sum} != {normal}"
            );
            assert!(
                (tangent_sum - Vec3::X * friction).length() < bound,
                "friction {label}: {tangent_sum:?} != {friction}"
            );
            let reaction = world.joint_reaction(*joint).unwrap();
            assert!(
                (reaction.child.linear.x - control / h).abs()
                    < tolerance * (1. + mass + (control / h).abs()),
                "motor reaction {label}: {} != {}",
                reaction.child.linear.x,
                control / h
            );
            assert!(
                (reaction.parent.linear + reaction.child.linear).length() < bound,
                "endpoint {label}"
            );
        }
    }
    assert_eq!(regimes, [true, true]);
    eprintln!(
        "D6 Force controls with live friction: 96 frames, masses .01/1/100, static hold/capped slide/brake/reversal, independent impulse/rate/reaction oracle"
    );
}

#[test]
fn cpu_rigid_loop_force_drives_share_live_friction_response() {
    contacts(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[test]
fn cpu_rigid_loop_acceleration_drives_share_live_friction_response() {
    contacts_model(
        |scene| RigidWorld::from_scene(scene).unwrap(),
        1e-8,
        MotorModel::Acceleration,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_rigid_loop_acceleration_drives_share_live_friction_response() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    contacts_model(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
        MotorModel::Acceleration,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_rigid_loop_force_drives_share_live_friction_response() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    contacts(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}

fn control_breaks(mut make: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    use molla_sim::joint_control::JointBreakTrigger;
    use molla_sim::runtime::JointBreakLimits;
    for capture in [false, true] {
        let (scene, cases) = fixture();
        let mut world = make(scene);
        world.solver.substeps = 4;
        world.set_reaction_readout(capture).unwrap();
        let dt = 1. / 240.;
        let h = dt / 4.;
        let next = (0.07 + h * (0.1 - 0.2)) / (1. + 2. * h);
        for c in &cases {
            world
                .scene
                .set_joint_motor(c.joint, c.axis, controls(24, c.mass).0)
                .unwrap();
            world
                .scene
                .set_joint_brake(c.joint, c.axis, 2. * c.mass)
                .unwrap();
            world
                .scene
                .set_joint_break_limits(
                    c.joint,
                    JointBreakLimits {
                        force: ((c.axis as usize) < 3).then_some(0.1 * c.mass),
                        torque: (c.axis as usize >= 3).then_some(0.1 * c.mass),
                    },
                )
                .unwrap();
        }
        world.step(dt).unwrap();
        assert_eq!(world.joint_break_events().len(), cases.len());
        for c in &cases {
            let angular = c.axis as usize >= 3;
            let event = world
                .joint_break_events()
                .iter()
                .find(|e| e.joint == c.joint)
                .unwrap();
            assert_eq!(event.sample.substep, 0);
            assert_eq!(
                event.trigger,
                if angular {
                    JointBreakTrigger::Torque
                } else {
                    JointBreakTrigger::Force
                }
            );
            let reaction = c.direction * (-c.mass * (0.2 + 2. * next));
            let bound = tolerance * (1. + c.mass);
            let actual = if angular {
                event.sample.child.angular
            } else {
                event.sample.child.linear
            };
            assert!(
                (actual - reaction).length() < bound,
                "control break axis={:?} mass={} {actual:?} != {reaction:?}",
                c.axis,
                c.mass
            );
            assert!((event.sample.child.linear + event.sample.parent.linear).length() < bound);
            assert!((event.sample.child.angular + event.sample.parent.angular).length() < bound);
            assert!(world.scene.joint_broken(c.joint).unwrap());
            let rate = world.scene.body_velocity(c.body).unwrap();
            let expected = c.direction * (next + 0.1 * (dt - h));
            let actual = if angular { rate.angular } else { rate.linear };
            assert!(
                (actual - expected).length() < tolerance,
                "broken control remains applied axis={:?} mass={} {actual:?} != {expected:?}",
                c.axis,
                c.mass
            );
            world
                .scene
                .set_joint_break_limits(c.joint, JointBreakLimits::default())
                .unwrap();
            let pose = world.scene.body_pose(c.body).unwrap();
            world.scene.repair_joint(c.joint).unwrap();
            assert_eq!(world.scene.body_pose(c.body).unwrap(), pose);
            assert_eq!(world.scene.body_velocity(c.body).unwrap(), rate);
            assert_eq!(
                world
                    .scene
                    .joint_axis_settings(c.joint, c.axis)
                    .unwrap()
                    .motor,
                Some(controls(24, c.mass).0)
            );
            assert_eq!(
                world
                    .scene
                    .joint_axis_settings(c.joint, c.axis)
                    .unwrap()
                    .brake,
                2. * c.mass
            );
        }
        world.step(dt).unwrap();
        for c in &cases {
            assert!(!world.scene.joint_broken(c.joint).unwrap());
            let mut expected = next + 0.1 * (dt - h);
            for _ in 0..4 {
                expected = (expected + h * (0.1 - 0.2)) / (1. + 2. * h);
            }
            let rate = world.scene.body_velocity(c.body).unwrap();
            let actual = if c.axis as usize >= 3 {
                rate.angular
            } else {
                rate.linear
            };
            assert!(
                (actual - c.direction * expected).length() < tolerance,
                "repaired control axis={:?} mass={}",
                c.axis,
                c.mass
            );
        }
        assert_eq!(world.drain_joint_break_events().len(), cases.len());
        assert!(world.quarantined_bodies().is_empty());
        eprintln!(
            "Force motor + brake breaking/repair: 24 rotated axis closures, capture={capture}, event oracle, broken-row retirement and live-control repair verified"
        );
    }
}

#[test]
fn cpu_rigid_loop_motor_brake_breaking_and_repair_retire_and_restore_controls() {
    control_breaks(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_rigid_loop_motor_brake_breaking_and_repair_retire_and_restore_controls() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    control_breaks(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_rigid_loop_finite_force_cap_overflow_rejects_without_publication_and_recovers() {
    let (scene, cases) = fixture();
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    let c = &cases[0];
    let mut motor = controls(24, c.mass).0;
    motor.max_force = 1e40;
    world.scene.set_joint_motor(c.joint, c.axis, motor).unwrap();
    let snapshot: Vec<_> = cases
        .iter()
        .map(|c| {
            (
                world.scene.body_pose(c.body).unwrap(),
                world.scene.body_velocity(c.body).unwrap(),
            )
        })
        .collect();
    let error = world.step(1. / 240.).unwrap_err();
    assert!(
        error.to_string().contains("control exceeds f32 range"),
        "{error}"
    );
    for (c, (pose, velocity)) in cases.iter().zip(snapshot) {
        assert_eq!(world.scene.body_pose(c.body).unwrap(), pose);
        assert_eq!(world.scene.body_velocity(c.body).unwrap(), velocity);
        assert!(world.joint_reaction(c.joint).is_none());
    }
    assert!(world.contact_pairs().is_empty());
    assert!(world.joint_break_events().is_empty());
    world
        .scene
        .set_joint_motor(c.joint, c.axis, controls(24, c.mass).0)
        .unwrap();
    world.step(1. / 240.).unwrap();
    let rate = world.scene.body_velocity(c.body).unwrap().angular;
    assert!((rate - c.direction * (0.07 - 0.1 / 240.)).length() < 2e-3);
    let force = world.joint_reaction(c.joint).unwrap().child.angular;
    assert!((force + c.direction * (0.2 * c.mass)).length() < 2e-3 * (1. + c.mass));
}

fn impact_control(mut make: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for mode in [
        JointTargetMode::Effort,
        JointTargetMode::Position,
        JointTargetMode::Velocity,
        JointTargetMode::PositionVelocity,
        JointTargetMode::None,
    ] {
        for sleep_remote in [false, true] {
            impact_control_case(
                &mut make,
                tolerance,
                JointMotor {
                    mode,
                    target_position: if mode == JointTargetMode::Effort {
                        0.2
                    } else {
                        0.003
                    },
                    target_velocity: 0.1,
                    stiffness: 30.,
                    damping: 3.,
                    max_force: 0.2,
                    ..Default::default()
                },
                if mode == JointTargetMode::Effort {
                    0.
                } else {
                    0.4
                },
                sleep_remote,
                1.,
            );
        }
    }
}

fn impact_control_case(
    make: &mut impl FnMut(RigidScene) -> RigidWorld,
    tolerance: f64,
    motor: JointMotor,
    brake: f64,
    sleep_remote: bool,
    mass: f64,
) {
    use molla_sim::runtime::{ColliderDesc, ColliderGeometry};
    use molla_sim::{ColliderMass, ContactMaterial, FrictionCoefficients};
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let origin = Vec3::Y * 0.501;
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(origin),
            velocity: SpatialVector::new(Vec3::new(0.2, -0.8, 0.), Vec3::ZERO),
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
            kind: JointType::D6,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(origin),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_joint_locked_axes(joint, JointAxes::ANG.with(JointAxis::LinZ))
        .unwrap();
    scene
        .set_joint_motor(joint, JointAxis::LinX, motor)
        .unwrap();
    scene
        .set_joint_brake(joint, JointAxis::LinX, brake)
        .unwrap();
    let material = ContactMaterial {
        stiffness: 1e4,
        damping: 0.,
        friction: Some(FrictionCoefficients::uniform(0.1)),
        ..Default::default()
    };
    let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    ball.parent = Some(body);
    ball.mass = ColliderMass::Mass(0.);
    ball.restitution = 0.5;
    ball.contact_material = material;
    scene.insert_collider(ball).unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(10., 0.1, 10.),
    });
    ground.pose = Transform::from_translation(-Vec3::Y * 0.1);
    ground.restitution = 0.5;
    ground.contact_material = material;
    scene.insert_collider(ground).unwrap();
    let remote = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::splat(100.)),
            additional_mass: Some(MassProperties {
                mass: 1.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let mut world = make(scene);
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    if sleep_remote {
        world.sleep_body(remote).unwrap();
    }
    let h = 1. / 240.;
    let mut total_normal = 0.;
    for frame in 0..4 {
        let before = world.scene.body_velocity(body).unwrap();
        let position = world.scene.body_pose(body).unwrap().position.x - origin.x;
        world.step(h).unwrap();
        assert_eq!(world.is_sleeping(remote), sleep_remote);
        assert_eq!(
            world.scene.body_pose(remote).unwrap().position,
            Vec3::splat(100.)
        );
        let after = world.scene.body_velocity(body).unwrap();
        let mut friction = Vec3::ZERO;
        for pair in world.contact_pairs() {
            let sign = if pair.bodies[1] == Some(body) {
                1.
            } else {
                -1.
            };
            for point in pair.points {
                total_normal += point.normal_impulse;
                friction += point.tangent_impulse * sign;
            }
        }
        let reaction = world.joint_reaction(joint).unwrap().child.linear.x;
        let kp = if matches!(
            motor.mode,
            JointTargetMode::Position | JointTargetMode::PositionVelocity
        ) {
            motor.stiffness
                * if motor.model == MotorModel::Acceleration {
                    mass
                } else {
                    1.
                }
        } else {
            0.
        };
        let kd = if matches!(motor.mode, JointTargetMode::None | JointTargetMode::Effort) {
            0.
        } else {
            motor.damping
                * if motor.model == MotorModel::Acceleration {
                    mass
                } else {
                    1.
                }
        };
        let target = if matches!(
            motor.mode,
            JointTargetMode::Velocity | JointTargetMode::PositionVelocity
        ) {
            motor.target_velocity
        } else {
            0.
        };
        let force = if motor.mode == JointTargetMode::Effort {
            motor.target_position
        } else {
            kp * (motor.target_position - position - h * after.linear.x)
                + kd * (target - after.linear.x)
        }
        .clamp(-motor.max_force, motor.max_force)
            - brake * after.linear.x;
        assert!(
            (reaction - force).abs() < tolerance * (1. + force.abs()),
            "impact frame={frame} mode={:?}: motor reaction {reaction} != {force}",
            motor.mode
        );
        assert!(
            (after.linear.x - before.linear.x - (force * h + friction.x) / mass).abs() < tolerance,
            "impact frame={frame} mode={:?}: duplicate/missing motor impulse",
            motor.mode
        );
    }
    assert!(
        (total_normal - 1.2 * mass).abs() < tolerance * (1. + mass),
        "normal impulse {total_normal} != {}",
        1.2 * mass
    );
    assert!((world.scene.body_velocity(body).unwrap().linear.y - 0.4).abs() < tolerance);
}

#[test]
fn cpu_rigid_loop_controls_are_applied_once_during_live_oblique_impacts() {
    impact_control(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_rigid_loop_controls_are_applied_once_during_live_oblique_impacts() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    impact_control(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}

fn acceleration_impacts(mut make: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for mass in [0.01, 1., 100.] {
        impact_control_case(
            &mut make,
            tolerance,
            JointMotor {
                mode: JointTargetMode::PositionVelocity,
                target_position: 0.003,
                target_velocity: 0.1,
                stiffness: 30.,
                damping: 3.,
                max_force: 0.2 * mass,
                model: MotorModel::Acceleration,
            },
            0.4 * mass,
            false,
            mass,
        );
    }
}

#[test]
fn cpu_rigid_loop_acceleration_feedback_is_applied_once_during_mass_scaled_impacts() {
    acceleration_impacts(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_rigid_loop_acceleration_feedback_is_applied_once_during_mass_scaled_impacts() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    acceleration_impacts(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_contact_bearing_sleep_preserves_state_and_recovers_on_wake() {
    use molla_sim::ColliderMass;
    use molla_sim::runtime::{ColliderDesc, ColliderGeometry};
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(Vec3::Y),
            additional_mass: Some(MassProperties {
                mass: 1.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let mut sphere = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    sphere.parent = Some(body);
    sphere.mass = ColliderMass::Mass(0.);
    scene.insert_collider(sphere).unwrap();
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    world.sleep_body(body).unwrap();
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    world.step(1. / 240.).unwrap();
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
    assert!(world.is_sleeping(body));
    world.wake_body(body).unwrap();
    world.step(1. / 240.).unwrap();
    assert!(!world.is_sleeping(body));
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
}
