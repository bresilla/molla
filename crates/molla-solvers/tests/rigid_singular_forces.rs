use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::BodyDesc;
use molla_sim::{BodyParams, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn certify_locked_singular_root(
    mut make_world: impl FnMut(molla_sim::runtime::RigidScene) -> RigidWorld,
    tolerance: f64,
) {
    use molla_sim::JointType;
    use molla_sim::joint_control::{JointAxes, JointAxis};
    use molla_sim::runtime::{JointDesc, RigidScene};

    for substeps in [1, 4] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let mut cases = Vec::new();
        for mass in [1e-6, 1.0, 1e6] {
            for rotated in [false, true] {
                let rotation = if rotated {
                    Quat::from_rotation_z(0.4) * Quat::from_rotation_x(-0.2)
                } else {
                    Quat::IDENTITY
                };
                let origin = Vec3::new(8.0 * cases.len() as f64, 2.0, -3.0);
                let body = scene
                    .insert_body(BodyDesc {
                        params: BodyParams::new().with_pose(Transform::new(origin, rotation)),
                        additional_mass: Some(MassProperties {
                            mass,
                            local_com: Vec3::ZERO,
                            inertia: Mat3::from_diagonal(Vec3::new(0.0, mass, mass)),
                        }),
                        ..Default::default()
                    })
                    .unwrap();
                let joint = scene
                    .insert_joint(JointDesc {
                        parent: None,
                        child: body,
                        kind: JointType::D6,
                        axis: Vec3::X,
                        frame_parent: Transform::new(origin, rotation),
                        frame_child: Transform::IDENTITY,
                    })
                    .unwrap();
                scene
                    .set_joint_locked_axes(joint, JointAxes::NONE.with(JointAxis::AngZ))
                    .unwrap();
                scene
                    .add_body_wrench(
                        body,
                        SpatialVector::new(Vec3::ZERO, rotation * Vec3::new(0.2, 0.1, 0.0) * mass),
                    )
                    .unwrap();
                cases.push((body, origin, rotation));
            }
        }
        let mut world = make_world(scene);
        world.solver.substeps = substeps;
        world.set_reaction_readout(true).unwrap();
        let mut rate = 0.0;
        let mut angle = 0.0;
        for frame in 0..24 {
            for _ in 0..substeps {
                let h = 1.0 / (120.0 * substeps as f64);
                rate += 0.1 * h;
                angle += 2.0 * (0.5 * h * rate).atan();
            }
            world.step(1.0 / 120.0).unwrap();
            for &(body, origin, rotation) in &cases {
                let velocity = world.scene.body_velocity(body).unwrap();
                let pose = world.scene.body_pose(body).unwrap();
                let relative = (rotation * Quat::from_rotation_y(angle)).conjugate()
                    * pose.rotation.normalize();
                assert!(
                    (velocity.angular - rotation * Vec3::Y * rate).length() < tolerance,
                    "locked singular root frame={frame} substeps={substeps} body={body:?} rate={rate} velocity={velocity:?}"
                );
                assert!(velocity.linear.length() < tolerance);
                assert!((pose.position - origin).length() < tolerance);
                assert!(2.0 * Vec3::new(relative.x, relative.y, relative.z).length() < tolerance);
            }
            assert!(world.quarantined_bodies().is_empty());
        }
    }
}

#[test]
fn locked_singular_d6_roots_keep_supported_spin_under_null_axis_loads() {
    certify_locked_singular_root(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_locked_singular_d6_roots_keep_supported_spin_under_null_axis_loads() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify_locked_singular_root(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-5,
    );
}

fn certify_null_mode_motors(
    mut make_world: impl FnMut(molla_sim::runtime::RigidScene) -> RigidWorld,
    tolerance: f64,
) {
    use molla_sim::joint_control::{JointAxis, JointMotor, MotorModel};
    use molla_sim::runtime::{JointDesc, RigidScene};
    use molla_sim::{JointTargetMode, JointType};

    for stable in [false, true] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let mut bodies = Vec::new();
        for mass in [0.001, 1.0, 1000.0] {
            for model in [MotorModel::Force, MotorModel::Acceleration] {
                for mode in [
                    JointTargetMode::Position,
                    JointTargetMode::Velocity,
                    JointTargetMode::PositionVelocity,
                ] {
                    let body = scene
                        .insert_body(BodyDesc {
                            additional_mass: Some(MassProperties {
                                mass,
                                local_com: Vec3::ZERO,
                                inertia: Mat3::from_diagonal(Vec3::new(mass, mass, 0.0)),
                            }),
                            ..Default::default()
                        })
                        .unwrap();
                    let joint = scene
                        .insert_joint(JointDesc {
                            parent: None,
                            child: body,
                            kind: JointType::D6,
                            axis: Vec3::X,
                            frame_parent: Transform::IDENTITY,
                            frame_child: Transform::IDENTITY,
                        })
                        .unwrap();
                    scene
                        .set_joint_motor(
                            joint,
                            JointAxis::AngZ,
                            JointMotor {
                                mode,
                                model,
                                target_position: 0.4,
                                target_velocity: 0.7,
                                stiffness: 12.0,
                                damping: 3.0,
                                max_force: f64::INFINITY,
                            },
                        )
                        .unwrap();
                    scene
                        .set_joint_motor(
                            joint,
                            JointAxis::LinX,
                            JointMotor {
                                mode: JointTargetMode::Velocity,
                                model: MotorModel::Acceleration,
                                target_velocity: 0.7,
                                damping: 3.0,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                    bodies.push((body, mass, model, mode));
                }
            }
        }
        let mut world = make_world(scene);
        world.solver.substeps = 2;
        world.solver.stable_motor_feedback = stable;
        let mut rate = 0.0;
        let mut position = 0.0;
        let h = 0.005;
        for frame in 0..12 {
            for _ in 0..2 {
                rate += 3.0 * (0.7 - rate) * h / if stable { 1.0 + 3.0 * h } else { 1.0 };
                position += rate * h;
            }
            world.step(2.0 * h).unwrap();
            for &(body, mass, model, mode) in &bodies {
                let pose = world.scene.body_pose(body).unwrap();
                let velocity = world.scene.body_velocity(body).unwrap();
                assert!(
                    (pose.position - Vec3::X * position).length() < tolerance
                        && pose
                            .rotation
                            .normalize()
                            .angle_between(Quat::IDENTITY)
                            .abs()
                            < tolerance
                        && (velocity.linear - Vec3::X * rate).length() < tolerance
                        && velocity.angular.length() < tolerance,
                    "frame={frame} stable={stable} mass={mass} model={model:?} mode={mode:?}: pose={pose:?} velocity={velocity:?}"
                );
                assert!(world.quarantined_bodies().is_empty());
            }
        }
    }
}

#[test]
fn null_mode_motors_preserve_supported_linear_drive() {
    certify_null_mode_motors(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_null_mode_motors_preserve_supported_linear_drive() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify_null_mode_motors(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        1e-5,
    );
}

#[test]
fn continuous_force_respects_singular_angular_modes_at_offset_com() {
    for (gravity, substeps) in [(Vec3::ZERO, 1), (Vec3::ZERO, 8), (Vec3::Y * -9.81, 8)] {
        for scale in [1e-6, 1.0, 1e6] {
            for inertia in [Mat3::ZERO, Mat3::from_diagonal(Vec3::new(0.0, 0.2, 0.2))] {
                for offset in [Vec3::ZERO, Vec3::new(0.03, 0.1, -0.02)] {
                    let mut world = RigidWorld::new().unwrap();
                    world.solver = world.solver.with_damping(0.0).with_substeps(substeps);
                    world.scene.set_gravity(gravity).unwrap();
                    let rotation = Quat::from_rotation_y(0.3);
                    let body = world
                        .scene
                        .insert_body(BodyDesc {
                            params: BodyParams::new()
                                .with_pose(Transform::new(Vec3::new(0.0, 0.4, -0.2), rotation)),
                            additional_mass: Some(MassProperties {
                                mass: 2.0 * scale,
                                local_com: offset,
                                inertia: inertia * scale,
                            }),
                            ..Default::default()
                        })
                        .unwrap();
                    let force = Vec3::new(0.3, 0.1, -0.2);
                    world
                        .scene
                        .add_body_wrench(
                            body,
                            SpatialVector::new(force * scale, rotation * Vec3::X * (0.1 * scale)),
                        )
                        .unwrap();
                    for step in 1..=120 {
                        world.step(1.0 / 120.0).unwrap();
                        let velocity = world.scene.body_velocity(body).unwrap();
                        let expected = (force / 2.0 + gravity) * (step as f64 / 120.0);
                        assert!(
                            (velocity.linear - expected).length() < 1e-8,
                            "step={step}, offset={offset:?}, inertia={inertia:?}, velocity={velocity:?}"
                        );
                        assert!(
                            velocity.angular.length() < 1e-8,
                            "step={step}, offset={offset:?}, inertia={inertia:?}, velocity={velocity:?}"
                        );
                        assert!(world.quarantined_bodies().is_empty());
                    }
                }
            }
        }
    }
}

#[test]
fn singular_body_keeps_unlocked_spin_and_com_acceleration() {
    for offset in [Vec3::ZERO, Vec3::new(0.03, 0.1, -0.02)] {
        let mut world = RigidWorld::new().unwrap();
        world.solver = world.solver.with_damping(0.0);
        world.scene.set_gravity(Vec3::ZERO).unwrap();
        let rotation = Quat::from_rotation_z(0.4);
        let axis = rotation * Vec3::Y;
        let body = world
            .scene
            .insert_body(BodyDesc {
                params: BodyParams::new()
                    .with_pose(Transform::new(Vec3::new(20.0, 3.0, -7.0), rotation)),
                additional_mass: Some(MassProperties {
                    mass: 2.0,
                    local_com: offset,
                    inertia: Mat3::from_diagonal(Vec3::new(0.0, 0.2, 0.2)),
                }),
                ..Default::default()
            })
            .unwrap();
        let initial_linear = Vec3::new(0.2, -0.1, 0.3);
        world
            .scene
            .set_body_velocity(body, SpatialVector::new(initial_linear, axis * 0.03))
            .unwrap();
        let force = Vec3::new(0.3, 0.1, -0.2);
        world
            .scene
            .add_body_wrench(body, SpatialVector::new(force, axis * 0.06))
            .unwrap();
        for step in 1..=120 {
            world
                .step(1.0 / 120.0)
                .unwrap_or_else(|error| panic!("step={step}, offset={offset:?}: {error}"));
            let velocity = world.scene.body_velocity(body).unwrap();
            let time = step as f64 / 120.0;
            assert!(
                (velocity.linear - initial_linear - force * (time / 2.0)).length() < 2e-5,
                "offset={offset:?}, step={step}, velocity={velocity:?}"
            );
            assert!(
                (velocity.angular - axis * (0.03 + time * 0.3)).length() < 1e-8,
                "offset={offset:?}, step={step}, velocity={velocity:?}"
            );
        }
    }
}

#[test]
fn raw_free_root_with_offset_frames_skips_zero_inertia_rotation() {
    use molla_core::BodyId;
    use molla_math::SpatialInertia;
    use molla_sim::{Contacts, ModelBuilder};
    use molla_solvers::{FeatherstoneSolver, Solver};

    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 2.0,
        com: Vec3::ZERO,
        inertia: Mat3::ZERO,
    }));
    builder
        .add_joint_free(
            BodyId::NONE,
            body,
            Transform::new(Vec3::new(0.0, 0.4, -0.2), Quat::from_rotation_x(0.2)),
            Transform::from_translation(Vec3::new(0.03, 0.1, -0.02)),
        )
        .unwrap();
    let mut model = builder.build().unwrap();
    model.gravity.host_mut().unwrap().fill(Vec3::ZERO);
    let mut state = model.state().unwrap();
    let mut next = model.state().unwrap();
    let force = Vec3::new(0.3, 0.1, -0.2);
    state.body_f.host_mut().unwrap()[body.index()] =
        SpatialVector::new(force, Vec3::new(0.01, -0.03, 0.02));
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu()
        .with_rigorous_d_solve(true)
        .with_damping(0.0);
    for step in 1..=120 {
        solver
            .step(&model, &state, &control, &contacts, 1.0 / 960.0, &mut next)
            .unwrap();
        let velocity = next.body_qd.host().unwrap()[body.index()];
        assert!((velocity.linear - force * (step as f64 / 960.0 / 2.0)).length() < 1e-8);
        assert!(velocity.angular.length() < 1e-8);
        std::mem::swap(&mut state, &mut next);
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_zero_inertia_roots_keep_com_acceleration_without_artificial_spin() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let gravity = -Vec3::Y * 9.81;
    let force = Vec3::new(0.3, 0.1, -0.2);
    for scale in [1e-6, 1., 1e6] {
        for offset in [Vec3::ZERO, Vec3::new(0.03, 0.1, -0.02)] {
            let mut scene = molla_sim::runtime::RigidScene::new().unwrap();
            scene.set_gravity(gravity).unwrap();
            let rotation = Quat::from_rotation_y(0.3);
            let body = scene
                .insert_body(BodyDesc {
                    params: BodyParams::new()
                        .with_pose(Transform::new(Vec3::new(20., 0.4, -7.), rotation)),
                    additional_mass: Some(MassProperties {
                        mass: 2. * scale,
                        local_com: offset,
                        inertia: Mat3::ZERO,
                    }),
                    ..Default::default()
                })
                .unwrap();
            scene
                .add_body_wrench(
                    body,
                    SpatialVector::new(force * scale, rotation * Vec3::X * (0.1 * scale)),
                )
                .unwrap();
            let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
            world.solver.substeps = 4;
            for step in 1..=120 {
                world.step(1. / 120.).unwrap_or_else(|error| {
                    panic!("step={step}, scale={scale}, offset={offset:?}: {error}")
                });
                let velocity = world.scene.body_velocity(body).unwrap();
                let expected = (force / 2. + gravity) * (step as f64 / 120.);
                assert!(
                    (velocity.linear - expected).length() < 1e-4,
                    "step={step}, scale={scale}, offset={offset:?}: {velocity:?}"
                );
                assert!(
                    velocity.angular.length() < 1e-5,
                    "step={step}, scale={scale}, offset={offset:?}: {velocity:?}"
                );
            }
        }
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_singular_and_weak_roots_preserve_supported_spin() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    for weak in [0., 1e-10] {
        for offset in [Vec3::ZERO, Vec3::new(0.03, 0.1, -0.02)] {
            if weak != 0. && offset != Vec3::ZERO {
                continue;
            }
            let rotation = if weak == 0. {
                Quat::from_rotation_z(0.4)
            } else {
                Quat::IDENTITY
            };
            let axis = rotation * if weak == 0. { Vec3::Y } else { Vec3::X };
            let moment = if weak == 0. { 0.2 } else { weak };
            let mut scene = molla_sim::runtime::RigidScene::new().unwrap();
            scene.set_gravity(Vec3::ZERO).unwrap();
            let body = scene
                .insert_body(BodyDesc {
                    params: BodyParams::new()
                        .with_pose(Transform::new(Vec3::new(20., 3., -7.), rotation)),
                    additional_mass: Some(MassProperties {
                        mass: 2.,
                        local_com: offset,
                        inertia: Mat3::from_diagonal(Vec3::new(weak, 0.2, 0.2)),
                    }),
                    ..Default::default()
                })
                .unwrap();
            let initial = Vec3::new(0.2, -0.1, 0.3);
            let force = Vec3::new(0.3, 0.1, -0.2);
            scene
                .set_body_velocity(body, SpatialVector::new(initial, axis * 0.03))
                .unwrap();
            scene
                .add_body_wrench(body, SpatialVector::new(force, axis * (moment * 0.3)))
                .unwrap();
            let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
            world.solver.substeps = 4;
            for step in 1..=120 {
                world.step(1. / 120.).unwrap_or_else(|error| {
                    panic!("step={step}, weak={weak}, offset={offset:?}: {error}")
                });
                let velocity = world.scene.body_velocity(body).unwrap();
                let time = step as f64 / 120.;
                assert!(
                    (velocity.linear - initial - force * (time / 2.)).length() < 1e-4,
                    "step={step}, weak={weak}, offset={offset:?}: {velocity:?}"
                );
                assert!(
                    (velocity.angular - axis * (0.03 + time * 0.3)).length() < 1e-5,
                    "step={step}, weak={weak}, offset={offset:?}: {velocity:?}"
                );
            }
        }
    }
}
