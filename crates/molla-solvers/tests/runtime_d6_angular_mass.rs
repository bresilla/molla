use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, BodyKind, JointDesc, JointHandle, RigidScene};
use molla_sim::{BodyParams, JointType, MassProperties, State};
use molla_solvers::rigid_world::RigidWorld;
use molla_solvers::{FeatherstoneSolver, Solver};

#[cfg(feature = "gpu")]
#[path = "common/gpu_readback.rs"]
mod gpu_readback;

const DT: f64 = 0.01;

struct Case {
    body: BodyHandle,
    joint: JointHandle,
    pose: Transform,
    anchor: Vec3,
    local_anchor: Vec3,
    com: Vec3,
    rate: Vec3,
    reaction: SpatialVector,
    reversed: bool,
}

fn fixture(dt: f64) -> (RigidScene, Vec<Case>) {
    fixture_with_bounds(dt, false)
}

fn fixture_regular_intervals(dt: f64, caps: bool) -> (RigidScene, Vec<Case>) {
    fixture_intervals_at(
        dt,
        caps,
        &[
            -0.6,
            -(std::f64::consts::PI - 0.6),
            0.6,
            std::f64::consts::PI - 0.6,
        ],
        &[0, 1, 2],
    )
}

fn fixture_near_pole_intervals(dt: f64) -> (RigidScene, Vec<Case>) {
    let pole = std::f64::consts::FRAC_PI_2;
    fixture_intervals_at(
        dt,
        false,
        &[-pole - 1e-7, -pole + 1e-7, pole - 1e-7, pole + 1e-7],
        &[1],
    )
}

fn fixture_intervals_at(
    dt: f64,
    caps: bool,
    pitches: &[f64],
    axes: &[usize],
) -> (RigidScene, Vec<Case>) {
    use molla_sim::joint_control::{JointAxes, JointAxis};
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let properties = MassProperties {
        mass: 2.,
        local_com: Vec3::new(0.4, -0.3, 0.2),
        inertia: Mat3::from_cols(
            Vec3::new(3., 0.2, 0.3),
            Vec3::new(0.2, 4., 0.4),
            Vec3::new(0.3, 0.4, 5.),
        ),
    };
    let frame_parent = Transform::new(
        Vec3::new(-0.3, 0.5, -0.9),
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.3, 0.4),
    );
    let frame_child = Transform::new(
        Vec3::new(-0.3, 0.5, -0.9),
        Quat::from_euler(glam::EulerRot::XYZ, -0.4, 0.3, -0.2),
    );
    let mut cases = Vec::new();
    for frame in [
        Quat::IDENTITY,
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.3),
    ] {
        for reversed in [false, true] {
            for &pitch in pitches {
                let angles = Vec3::new(0.3, pitch, 0.2);
                let gradients = [
                    Vec3::new(
                        1.,
                        angles.x.sin() * pitch.tan(),
                        -angles.x.cos() * pitch.tan(),
                    ),
                    Vec3::new(0., angles.x.cos(), angles.x.sin()),
                    Vec3::new(
                        0.,
                        -angles.x.sin() / pitch.cos(),
                        angles.x.cos() / pitch.cos(),
                    ),
                ];
                let tangents = [
                    Vec3::X,
                    Quat::from_rotation_x(angles.x) * Vec3::Y,
                    Quat::from_rotation_x(angles.x) * Quat::from_rotation_y(pitch) * Vec3::Z,
                ];
                for &axis in axes {
                    for upper_stop in [false, true] {
                        let relative =
                            Quat::from_euler(glam::EulerRot::XYZ, angles.x, angles.y, angles.z)
                                * if upper_stop { -1. } else { 1. };
                        let anchor = Vec3::new(1.2, -0.4, 0.8);
                        let parent_rotation = if reversed {
                            frame * relative.conjugate()
                        } else {
                            frame
                        };
                        let child_rotation = if reversed { frame } else { frame * relative };
                        let parent_pose = Transform::new(
                            anchor
                                - parent_rotation
                                    * frame_parent.rotation.conjugate()
                                    * frame_parent.position,
                            parent_rotation * frame_parent.rotation.conjugate(),
                        );
                        let child_pose = Transform::new(
                            anchor
                                - child_rotation
                                    * frame_child.rotation.conjugate()
                                    * frame_child.position,
                            child_rotation * frame_child.rotation.conjugate(),
                        );
                        for outward in [false, true] {
                            let parent = scene
                                .insert_body(BodyDesc {
                                    params: BodyParams::new().with_pose(parent_pose),
                                    kind: if reversed {
                                        BodyKind::Dynamic
                                    } else {
                                        BodyKind::Fixed
                                    },
                                    additional_mass: Some(properties),
                                    ..Default::default()
                                })
                                .unwrap();
                            let child = scene
                                .insert_body(BodyDesc {
                                    params: BodyParams::new().with_pose(child_pose),
                                    kind: if reversed {
                                        BodyKind::Fixed
                                    } else {
                                        BodyKind::Dynamic
                                    },
                                    additional_mass: Some(properties),
                                    ..Default::default()
                                })
                                .unwrap();
                            let joint = scene
                                .insert_joint(JointDesc {
                                    parent: Some(parent),
                                    child,
                                    kind: JointType::D6,
                                    axis: Vec3::X,
                                    frame_parent,
                                    frame_child,
                                })
                                .unwrap();
                            scene.set_joint_locked_axes(joint, JointAxes::LIN).unwrap();
                            if caps {
                                scene
                                    .set_joint_velocity_limit(
                                        joint,
                                        JointAxis::ALL[axis + 3],
                                        Some(0.),
                                    )
                                    .unwrap();
                            } else {
                                for k in 0..3 {
                                    let target = angles[k];
                                    let bounds = if k != axis {
                                        [target - 0.25, target + 0.25]
                                    } else if upper_stop {
                                        [target - 1e-9, target]
                                    } else {
                                        [target, target + 1e-9]
                                    };
                                    if k == axis {
                                        assert_eq!(bounds[0] as f32, bounds[1] as f32);
                                    }
                                    scene
                                        .set_joint_limits(
                                            joint,
                                            JointAxis::ALL[k + 3],
                                            Some(bounds),
                                        )
                                        .unwrap();
                                }
                            }
                            let body = if reversed { parent } else { child };
                            let pose = if reversed { parent_pose } else { child_pose };
                            let local_anchor = if reversed {
                                frame_parent.position
                            } else {
                                frame_child.position
                            };
                            let arm = pose.rotation * (properties.local_com - local_anchor);
                            let rotation = Mat3::from_quat(pose.rotation);
                            let inertia = rotation * properties.inertia * rotation.transpose();
                            let anchored = inertia
                                + (Mat3::IDENTITY * arm.length_squared()
                                    - Mat3::from_cols(arm * arm.x, arm * arm.y, arm * arm.z))
                                    * properties.mass;
                            let sign = if reversed { -1. } else { 1. };
                            let tangent = parent_rotation
                                * if caps {
                                    [Vec3::X, Vec3::Y, Vec3::Z][axis]
                                } else {
                                    tangents[axis]
                                }
                                * sign;
                            let normal = parent_rotation
                                * if caps {
                                    [Vec3::X, Vec3::Y, Vec3::Z][axis]
                                } else {
                                    gradients[axis]
                                }
                                * sign;
                            let direction =
                                if upper_stop { 1. } else { -1. } * if outward { 1. } else { -1. };
                            let free = tangent * (2. * direction);
                            let torque = anchored * free;
                            let inverse = anchored.inverse();
                            let acceleration = if caps || outward {
                                free - inverse
                                    * normal
                                    * (normal.dot(free) / normal.dot(inverse * normal))
                            } else {
                                free
                            };
                            if !caps {
                                assert!(normal.dot(acceleration).abs() * dt * dt < 1e-9);
                            }
                            let reaction_force = acceleration.cross(arm) * properties.mass;
                            let reaction_torque =
                                inertia * acceleration - torque + arm.cross(reaction_force);
                            scene
                                .add_body_wrench(body, SpatialVector::new(Vec3::ZERO, torque))
                                .unwrap();
                            cases.push(Case {
                                body,
                                joint,
                                pose,
                                anchor,
                                local_anchor,
                                com: properties.local_com,
                                rate: acceleration * dt,
                                reaction: SpatialVector::new(reaction_force, reaction_torque)
                                    .scale(sign),
                                reversed,
                            });
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases.len(), 2 * 2 * pitches.len() * axes.len() * 2 * 2);
    (scene, cases)
}

fn fixture_with_bounds(dt: f64, bounded: bool) -> (RigidScene, Vec<Case>) {
    fixture_at(dt, bounded, false, 0.2)
}

fn fixture_at(dt: f64, bounded: bool, regular: bool, width: f64) -> (RigidScene, Vec<Case>) {
    fixture_phase(dt, bounded, regular, width, false)
}

fn fixture_phase(
    dt: f64,
    bounded: bool,
    regular: bool,
    width: f64,
    phase: bool,
) -> (RigidScene, Vec<Case>) {
    use molla_sim::joint_control::{JointAxes, JointAxis};
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    let tensor = Mat3::from_cols(
        Vec3::new(3., 0.2, 0.3),
        Vec3::new(0.2, 4., 0.4),
        Vec3::new(0.3, 0.4, 5.),
    );
    let properties = MassProperties {
        mass: 2.,
        local_com: Vec3::new(0.4, -0.3, 0.2),
        inertia: tensor,
    };
    let frame_parent = Transform::new(
        Vec3::new(-0.3, 0.5, -0.9),
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.3, 0.4),
    );
    let frame_child = Transform::new(
        Vec3::new(-0.3, 0.5, -0.9),
        Quat::from_euler(glam::EulerRot::XYZ, -0.4, 0.3, -0.2),
    );
    for frame in [
        Quat::IDENTITY,
        Quat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.3),
    ] {
        for reversed in [false, true] {
            let pitches = if regular {
                let alternate = std::f64::consts::PI - 0.6;
                [
                    (-0.6, false),
                    (-alternate, false),
                    (0.6, false),
                    (alternate, false),
                ]
            } else {
                let pole = std::f64::consts::FRAC_PI_2;
                [(-pole, false), (-pole, true), (pole, false), (pole, true)]
            };
            let wrapped = if phase {
                let pole = 3. * std::f64::consts::FRAC_PI_2;
                vec![(-pole, false), (-pole, true), (pole, false), (pole, true)]
            } else {
                Vec::new()
            };
            for (pitch, upper_stop) in pitches.into_iter().chain(wrapped) {
                if !bounded && upper_stop {
                    continue;
                }
                let relative = Quat::from_euler(glam::EulerRot::XYZ, 0.3, pitch, 0.2);
                let relative = if bounded && upper_stop {
                    -relative
                } else {
                    relative
                };
                let anchor = Vec3::new(1.2, -0.4, 0.8);
                let parent_rotation = if reversed {
                    frame * relative.conjugate()
                } else {
                    frame
                };
                let child_rotation = if reversed { frame } else { frame * relative };
                let parent_pose = Transform::new(
                    anchor
                        - parent_rotation
                            * frame_parent.rotation.conjugate()
                            * frame_parent.position,
                    parent_rotation * frame_parent.rotation.conjugate(),
                );
                let child_pose = Transform::new(
                    anchor
                        - child_rotation * frame_child.rotation.conjugate() * frame_child.position,
                    child_rotation * frame_child.rotation.conjugate(),
                );
                for lock in if phase {
                    2..3
                } else {
                    (if bounded { 3 } else { 0 })..4
                } {
                    let hinge = lock == 3 || (lock == 2 && !regular);
                    for forbidden in [false, true] {
                        let parent = scene
                            .insert_body(BodyDesc {
                                params: BodyParams::new().with_pose(parent_pose),
                                kind: if reversed {
                                    BodyKind::Dynamic
                                } else {
                                    BodyKind::Fixed
                                },
                                additional_mass: Some(properties),
                                ..Default::default()
                            })
                            .unwrap();
                        let child = scene
                            .insert_body(BodyDesc {
                                params: BodyParams::new().with_pose(child_pose),
                                kind: if reversed {
                                    BodyKind::Fixed
                                } else {
                                    BodyKind::Dynamic
                                },
                                additional_mass: Some(properties),
                                ..Default::default()
                            })
                            .unwrap();
                        let joint = scene
                            .insert_joint(JointDesc {
                                parent: Some(parent),
                                child,
                                kind: JointType::D6,
                                axis: Vec3::X,
                                frame_parent,
                                frame_child,
                            })
                            .unwrap();
                        scene.set_joint_locked_axes(joint, JointAxes::LIN).unwrap();
                        if lock == 0 || lock == 3 {
                            scene
                                .set_joint_limits(joint, JointAxis::AngX, Some([0.3, 0.3]))
                                .unwrap();
                        }
                        if lock == 1 || lock == 3 {
                            scene
                                .set_joint_limits(joint, JointAxis::AngZ, Some([0.2, 0.2]))
                                .unwrap();
                        }
                        if lock == 2 {
                            scene
                                .set_joint_limits(joint, JointAxis::AngY, Some([pitch, pitch]))
                                .unwrap();
                        }
                        if bounded {
                            let limits = if phase {
                                [
                                    Some((JointAxis::AngX, 0.3, upper_stop)),
                                    Some((JointAxis::AngZ, 0.2, upper_stop == (pitch.sin() > 0.))),
                                ]
                            } else {
                                [Some((JointAxis::AngY, pitch, upper_stop)), None]
                            };
                            for (axis, target, upper) in limits.into_iter().flatten() {
                                let bounds = if upper {
                                    [target - width, target]
                                } else {
                                    [target, target + width]
                                };
                                if width < 1e-7 {
                                    assert_eq!(bounds[0] as f32, bounds[1] as f32);
                                    assert!(bounds[1] - bounds[0] > 1e-12);
                                }
                                scene.set_joint_limits(joint, axis, Some(bounds)).unwrap();
                            }
                            if phase {
                                let value = 0.3 + if pitch.sin() > 0. { 0.2 } else { -0.2 };
                                let other =
                                    value + if upper_stop { -2. * width } else { 2. * width };
                                assert_eq!(value as f32, other as f32);
                            }
                        }
                        let body = if reversed { parent } else { child };
                        let pose = if reversed { parent_pose } else { child_pose };
                        let local_anchor = if reversed {
                            frame_parent.position
                        } else {
                            frame_child.position
                        };
                        let arm = pose.rotation * (properties.local_com - local_anchor);
                        let rotation = Mat3::from_quat(pose.rotation);
                        let inertia = rotation * tensor * rotation.transpose();
                        let anchored = inertia
                            + (Mat3::IDENTITY * arm.length_squared()
                                - Mat3::from_cols(arm * arm.x, arm * arm.y, arm * arm.z))
                                * properties.mass;
                        let tangent = parent_rotation
                            * if lock == 2 {
                                Vec3::X
                            } else {
                                Quat::from_rotation_x(0.3) * Vec3::Y
                            };
                        let normal = parent_rotation
                            * if regular && lock == 0 {
                                Vec3::new(
                                    pitch.cos(),
                                    0.3_f64.sin() * pitch.sin(),
                                    -0.3_f64.cos() * pitch.sin(),
                                )
                            } else {
                                Quat::from_rotation_x(0.3)
                                    * if lock == 2 { Vec3::Y } else { Vec3::Z }
                            };
                        let force = if forbidden || bounded {
                            Vec3::ZERO
                        } else {
                            Vec3::new(1.5, -2., 0.8)
                        };
                        let torque = if bounded {
                            let direction = if upper_stop { 1. } else { -1. }
                                * if reversed { -1. } else { 1. }
                                * if forbidden { 1. } else { -1. };
                            tangent * (8. * direction) + normal * 3.
                        } else if forbidden {
                            normal * 8.
                        } else if hinge {
                            tangent * 8. + normal * 3.
                        } else {
                            Vec3::new(8., -6., 4.)
                        };
                        let load = torque + arm.cross(force);
                        let acceleration = if bounded && forbidden {
                            Vec3::ZERO
                        } else if hinge {
                            tangent * load.dot(tangent) / tangent.dot(anchored * tangent)
                        } else {
                            let inverse = anchored.inverse();
                            let free = inverse * load;
                            free - inverse
                                * normal
                                * (normal.dot(free) / normal.dot(inverse * normal))
                        };
                        if bounded && !forbidden {
                            assert!(acceleration.length() * dt * dt < width);
                        }
                        let reaction_force = acceleration.cross(arm) * properties.mass - force;
                        let reaction_torque =
                            inertia * acceleration - torque + arm.cross(reaction_force);
                        let reaction = SpatialVector::new(reaction_force, reaction_torque)
                            .scale(if reversed { -1. } else { 1. });
                        scene
                            .add_body_wrench(body, SpatialVector::new(force, torque))
                            .unwrap();
                        let index = scene.joint_index(joint).unwrap().index();
                        assert_eq!(
                            scene.model().joint_reversed.host().unwrap()[index] != 0,
                            reversed
                        );
                        if bounded {
                            let offset = scene.model().joint_coord_offset.host().unwrap()[index];
                            let scalar = scene.state().joint_q.host().unwrap()[offset as usize + 6];
                            assert_eq!(scalar < 0., relative.w < 0.);
                        }
                        cases.push(Case {
                            body,
                            joint,
                            pose,
                            anchor,
                            local_anchor,
                            com: properties.local_com,
                            rate: acceleration * dt,
                            reaction,
                            reversed,
                        });
                    }
                }
            }
        }
    }
    (scene, cases)
}

fn verify(scene: &RigidScene, state: &State, cases: &[Case], dt: f64, tolerance: f64) {
    for (i, case) in cases.iter().enumerate() {
        let index = scene.body_index(case.body).unwrap().index();
        let pose = state.body_q.host().unwrap()[index];
        let velocity = state.body_qd.host().unwrap()[index];
        let rotation = Quat::from_scaled_axis(case.rate * dt) * case.pose.rotation;
        let expected_linear = case
            .rate
            .cross(pose.rotation * (case.com - case.local_anchor));
        let rate_error = (velocity.angular - case.rate).length();
        let linear_error = (velocity.linear - expected_linear).length();
        let pose_error = (pose.position - (case.anchor - rotation * case.local_anchor)).length();
        let rotation_error = 2.
            * (pose.rotation.normalize() * rotation.conjugate())
                .xyz()
                .length();
        println!(
            "angular_mass case={i} reversed={} rate_error={rate_error:.9e} linear_error={linear_error:.9e} pose_error={pose_error:.9e} rotation_error={rotation_error:.9e}",
            case.reversed
        );
        assert!(
            rate_error.is_finite() && rate_error <= tolerance,
            "case={i} velocity={velocity:?} expected={:?}",
            case.rate
        );
        assert!(linear_error.is_finite() && linear_error <= tolerance);
        assert!(pose_error.is_finite() && pose_error <= tolerance);
        assert!(rotation_error.is_finite() && rotation_error <= tolerance);
    }
}

fn runtime(mut world: RigidWorld, cases: &[Case], dt: f64, tolerance: f64) {
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    world.step(dt).unwrap();
    assert!(world.quarantined_bodies().is_empty());
    assert!(world.joint_break_events().is_empty());
    verify(&world.scene, world.scene.state(), cases, dt, tolerance);
    for (i, case) in cases.iter().enumerate() {
        let actual = world.joint_reaction(case.joint).unwrap();
        let force_error = (actual.child.linear - case.reaction.linear).length();
        let torque_error = (actual.child.angular - case.reaction.angular).length();
        println!(
            "angular_mass_reaction case={i} force_error={force_error:.9e} torque_error={torque_error:.9e}"
        );
        assert!(
            force_error <= 0.01 * case.reaction.linear.length().max(1.),
            "case={i}: {actual:?} expected={:?}",
            case.reaction
        );
        assert!(
            torque_error <= 0.01 * case.reaction.angular.length().max(1.),
            "case={i}: {actual:?} expected={:?}",
            case.reaction
        );
        assert!((actual.child.linear + actual.parent.linear).length() <= tolerance);
        assert!((actual.child.angular + actual.parent.angular).length() <= tolerance);
    }
}

#[test]
fn cpu_mixed_locks_use_full_anchored_inertia_and_authored_reactions() {
    for dt in [DT, 0.0001] {
        let (scene, cases) = fixture(dt);
        runtime(RigidWorld::from_scene(scene).unwrap(), &cases, dt, 1e-8);
    }
}

#[test]
fn cpu_bounded_hinges_use_full_inertia_and_authored_reactions() {
    let dt = 0.0001;
    let (scene, cases) = fixture_with_bounds(dt, true);
    assert_eq!(cases.len(), 32);
    runtime(RigidWorld::from_scene(scene).unwrap(), &cases, dt, 1e-8);
}

#[test]
fn cpu_regular_points_use_full_inertia_and_authored_reactions() {
    let dt = 0.0001;
    let (scene, cases) = fixture_at(dt, false, true, 0.2);
    assert_eq!(cases.len(), 128);
    runtime(RigidWorld::from_scene(scene).unwrap(), &cases, dt, 1e-8);
}

#[test]
fn cpu_narrow_intervals_remain_bounded_hinges() {
    let dt = 0.00005;
    let (scene, cases) = fixture_at(dt, true, false, 1e-8);
    assert_eq!(cases.len(), 32);
    runtime(RigidWorld::from_scene(scene).unwrap(), &cases, dt, 1e-8);
}

#[test]
fn cpu_phase_intervals_preserve_motion_and_reactions() {
    let dt = 0.00001;
    let (scene, cases) = fixture_phase(dt, true, false, 1e-9, true);
    assert_eq!(cases.len(), 64);
    runtime(RigidWorld::from_scene(scene).unwrap(), &cases, dt, 1e-8);
}

#[test]
fn cpu_standalone_mixed_locks_use_full_anchored_inertia() {
    let (scene, cases) = fixture(DT);
    let mut next = scene.model().state().unwrap();
    FeatherstoneSolver::new_cpu()
        .step(
            scene.model(),
            scene.state(),
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            DT,
            &mut next,
        )
        .unwrap();
    verify(&scene, &next, &cases, DT, 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_mixed_locks_use_full_anchored_inertia_and_authored_reactions() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let (scene, cases) = fixture(DT);
    runtime(
        RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &cases,
        DT,
        1e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_small_steps_do_not_create_angular_point_reactions() {
    let dt = 0.0001;
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt}");
    let (scene, cases) = fixture(dt);
    runtime(
        RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &cases,
        dt,
        1e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_bounded_hinges_use_full_inertia_and_authored_reactions() {
    let dt = 0.0001;
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt}");
    let (scene, cases) = fixture_with_bounds(dt, true);
    assert_eq!(cases.len(), 32);
    runtime(
        RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &cases,
        dt,
        1e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_regular_points_use_full_inertia_and_authored_reactions() {
    let dt = 0.0001;
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt}");
    let (scene, cases) = fixture_at(dt, false, true, 0.2);
    assert_eq!(cases.len(), 128);
    runtime(
        RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &cases,
        dt,
        1e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_narrow_intervals_remain_bounded_hinges() {
    let dt = 0.00005;
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt} width=1e-8");
    let (scene, cases) = fixture_at(dt, true, false, 1e-8);
    assert_eq!(cases.len(), 32);
    runtime(
        RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &cases,
        dt,
        1e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_phase_intervals_preserve_motion_and_reactions() {
    let dt = 0.00001;
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt} width=1e-9");
    let (scene, cases) = fixture_phase(dt, true, false, 1e-9, true);
    assert_eq!(cases.len(), 64);
    runtime(
        RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &cases,
        dt,
        1e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_standalone_mixed_locks_use_full_anchored_inertia() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut scene, cases) = fixture(DT);
    scene.upload_model_to_device(device, queue).unwrap();
    scene.upload_controls_to_device(device, queue).unwrap();
    let mut initial = scene.state().clone_host().unwrap();
    let mut next = scene.model().state().unwrap();
    initial.upload_to_device(device, queue).unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, scene.model()).unwrap();
    solver
        .step(
            scene.model(),
            &initial,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            DT,
            &mut next,
        )
        .unwrap();
    next.download_from_device(device, queue).unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    verify(&scene, &next, &cases, DT, 1e-3);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_standalone_narrow_hinges_preserve_authored_stop_response() {
    let dt = 0.00005;
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt} width=1e-8");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut scene, cases) = fixture_at(dt, true, false, 1e-8);
    assert_eq!(cases.len(), 32);
    scene.upload_model_to_device(device, queue).unwrap();
    let references = scene
        .model()
        .joint_limit_references
        .device_buffer_current()
        .unwrap()
        .clone();
    scene.upload_model_to_device(device, queue).unwrap();
    assert_eq!(
        &references,
        scene
            .model()
            .joint_limit_references
            .device_buffer_current()
            .unwrap()
    );
    scene.upload_controls_to_device(device, queue).unwrap();
    let mut initial = scene.state().clone_host().unwrap();
    let mut next = scene.model().state().unwrap();
    initial.upload_to_device(device, queue).unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, scene.model()).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&[&initial, &next])
        .unwrap();
    solver.seed_gpu_coordinate_compensation(&initial).unwrap();
    solver
        .step(
            scene.model(),
            &initial,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            dt,
            &mut next,
        )
        .unwrap();
    next.download_from_device(device, queue).unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    verify(&scene, &next, &cases, dt, 1e-3);
    verify_narrow_acceleration(&scene, &next, &cases, dt);
    assert!(solver.seed_gpu_coordinate_compensation(&next).is_err());
    next.joint_q.download(device, queue).unwrap();
    let residues = gpu_readback::read::<f32>(
        &engine,
        solver.gpu_coordinate_compensation_buffer(&next).unwrap(),
        scene.model().joint_coord_total,
    );
    verify_retained_rotation_advance(&scene, &next, &residues, &cases, dt);
    for (i, case) in cases.iter().enumerate() {
        let joint = scene.joint_index(case.joint).unwrap().index();
        let coordinate = scene.model().joint_coord_offset.host().unwrap()[joint] as usize;
        let dof = scene.model().joint_dof_offset.host().unwrap()[joint] as usize;
        let lower = scene.model().joint_limit_lower.host().unwrap();
        let upper = scene.model().joint_limit_upper.host().unwrap();
        let q = next.joint_q.host().unwrap();
        let values: [f64; 4] = std::array::from_fn(|axis| {
            q[coordinate + 3 + axis] + residues[coordinate + 3 + axis] as f64
        });
        let relative = Quat::from_rotation_x(-lower[dof + 3])
            * Quat::from_array(values).normalize()
            * Quat::from_rotation_z(-lower[dof + 5]);
        let angle = 2. * relative.y.atan2(relative.w);
        let center = 0.5 * lower[dof + 4] + 0.5 * upper[dof + 4];
        let angle = angle
            + std::f64::consts::TAU * (((center - angle) / std::f64::consts::TAU + 0.5).floor());
        let violation = (lower[dof + 4] - angle).max(angle - upper[dof + 4]).max(0.);
        println!("standalone_narrow_retained case={i} violation={violation:.9e}");
        assert!(
            violation <= 1e-12,
            "case={i}: angle={angle} bounds={:?}",
            [lower[dof + 4], upper[dof + 4]]
        );
        assert!(relative.x.abs() <= 1e-10 && relative.z.abs() <= 1e-10);
    }
    initial.joint_q.host_mut().unwrap()[0] += 0.;
    assert!(solver.seed_gpu_coordinate_compensation(&initial).is_err());
}

fn verify_narrow_acceleration(scene: &RigidScene, next: &State, cases: &[Case], dt: f64) {
    for (i, case) in cases.iter().enumerate() {
        let index = scene.body_index(case.body).unwrap().index();
        let actual = next.body_qd.host().unwrap()[index].angular / dt;
        let expected = case.rate / dt;
        let error = (actual - expected).length();
        println!("standalone_narrow_acceleration case={i} error={error:.9e}");
        assert!(
            error <= 0.01 * expected.length().max(1.),
            "case={i}: {actual:?} expected={expected:?}"
        );
    }
}

fn verify_retained_rotation_advance(
    scene: &RigidScene,
    next: &State,
    residues: &[f32],
    cases: &[Case],
    dt: f64,
) {
    for (i, case) in cases.iter().enumerate() {
        let joint = scene.joint_index(case.joint).unwrap().index();
        let c = scene.model().joint_coord_offset.host().unwrap()[joint] as usize;
        let desc = scene.joint(case.joint).unwrap();
        let parent = desc
            .parent
            .map(|body| {
                scene.state().body_q.host().unwrap()[scene.body_index(body).unwrap().index()]
            })
            .unwrap_or(Transform::IDENTITY)
            .compose(&desc.frame_parent);
        let omega = parent.rotation.conjugate() * case.rate * if case.reversed { -1. } else { 1. };
        let initial = &scene.state().joint_q.host().unwrap()[c + 3..c + 7];
        let initial = Quat::from_xyzw(initial[0], initial[1], initial[2], initial[3]).normalize();
        let expected = Quat::from_scaled_axis(omega * dt) * initial;
        let retained: [f64; 4] = std::array::from_fn(|axis| {
            next.joint_q.host().unwrap()[c + 3 + axis]
                + residues.get(c + 3 + axis).copied().unwrap_or(0.) as f64
        });
        let actual = Quat::from_array(retained).normalize();
        let error = (actual * expected.conjugate()).xyz().length() * 2.;
        println!("retained_rotation_advance case={i} error={error:.9e}");
        assert!(error.is_finite() && error <= 1e-12);
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_standalone_regular_points_preserve_authored_stop_response() {
    let dt = 0.0001;
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt}");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut scene, cases) = fixture_at(dt, false, true, 0.2);
    assert_eq!(cases.len(), 128);
    scene.upload_model_to_device(device, queue).unwrap();
    scene.upload_controls_to_device(device, queue).unwrap();
    let mut initial = scene.state().clone_host().unwrap();
    let mut next = scene.model().state().unwrap();
    initial.upload_to_device(device, queue).unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, scene.model()).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&[&initial, &next])
        .unwrap();
    solver.seed_gpu_coordinate_compensation(&initial).unwrap();
    solver
        .step(
            scene.model(),
            &initial,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            dt,
            &mut next,
        )
        .unwrap();
    next.download_from_device(device, queue).unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    verify(&scene, &next, &cases, dt, 1e-3);
    verify_narrow_acceleration(&scene, &next, &cases, dt);
}

#[test]
fn cpu_standalone_regular_points_preserve_authored_stop_response() {
    let dt = 0.0001;
    let (scene, cases) = fixture_at(dt, false, true, 0.2);
    assert_eq!(cases.len(), 128);
    let initial = scene.state().clone_host().unwrap();
    let mut next = scene.model().state().unwrap();
    FeatherstoneSolver::new_cpu()
        .step(
            scene.model(),
            &initial,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            dt,
            &mut next,
        )
        .unwrap();
    verify(&scene, &next, &cases, dt, 1e-8);
    verify_narrow_acceleration(&scene, &next, &cases, dt);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_standalone_phase_intervals_preserve_authored_stop_response() {
    let dt = 0.00001;
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt} width=1e-9");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut scene, cases) = fixture_phase(dt, true, false, 1e-9, true);
    assert_eq!(cases.len(), 64);
    scene.upload_model_to_device(device, queue).unwrap();
    scene.upload_controls_to_device(device, queue).unwrap();
    let mut initial = scene.state().clone_host().unwrap();
    let mut next = scene.model().state().unwrap();
    initial.upload_to_device(device, queue).unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, scene.model()).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&[&initial, &next])
        .unwrap();
    solver.seed_gpu_coordinate_compensation(&initial).unwrap();
    solver
        .step(
            scene.model(),
            &initial,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            dt,
            &mut next,
        )
        .unwrap();
    next.download_from_device(device, queue).unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    verify(&scene, &next, &cases, dt, 1e-3);
    verify_narrow_acceleration(&scene, &next, &cases, dt);
    next.joint_q.download(device, queue).unwrap();
    let residues = gpu_readback::read::<f32>(
        &engine,
        solver.gpu_coordinate_compensation_buffer(&next).unwrap(),
        scene.model().joint_coord_total,
    );
    for (i, case) in cases.iter().enumerate() {
        let joint = scene.joint_index(case.joint).unwrap().index();
        let coordinate = scene.model().joint_coord_offset.host().unwrap()[joint] as usize;
        let dof = scene.model().joint_dof_offset.host().unwrap()[joint] as usize;
        let lower = scene.model().joint_limit_lower.host().unwrap();
        let upper = scene.model().joint_limit_upper.host().unwrap();
        let values: [f64; 4] = std::array::from_fn(|axis| {
            next.joint_q.host().unwrap()[coordinate + 3 + axis]
                + residues[coordinate + 3 + axis] as f64
        });
        let q = Quat::from_array(values).normalize();
        let phase = (2. * (q.y * q.z + q.w * q.x)).atan2(1. - 2. * (q.x * q.x + q.z * q.z));
        let positive = lower[dof + 4].sin() > 0.;
        let (lo, hi) = if positive {
            (
                lower[dof + 3] + lower[dof + 5],
                upper[dof + 3] + upper[dof + 5],
            )
        } else {
            (
                lower[dof + 3] - upper[dof + 5],
                upper[dof + 3] - lower[dof + 5],
            )
        };
        let center = 0.5 * lo + 0.5 * hi;
        let phase = phase
            + std::f64::consts::TAU * (((center - phase) / std::f64::consts::TAU + 0.5).floor());
        let violation = (lo - phase).max(phase - hi).max(0.);
        println!("standalone_phase_retained case={i} violation={violation:.9e}");
        assert!(
            violation <= 1e-12,
            "case={i}: phase={phase} bounds={:?}",
            [lo, hi]
        );
        let relative = Quat::from_rotation_y(-lower[dof + 4]) * q;
        assert!(relative.x.abs() <= 1e-10 && relative.y.abs() <= 1e-10);
    }
}

#[test]
fn cpu_standalone_phase_intervals_preserve_authored_stop_response() {
    let dt = 0.00001;
    let (scene, cases) = fixture_phase(dt, true, false, 1e-9, true);
    assert_eq!(cases.len(), 64);
    let initial = scene.state().clone_host().unwrap();
    let mut next = scene.model().state().unwrap();
    FeatherstoneSolver::new_cpu()
        .step(
            scene.model(),
            &initial,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            dt,
            &mut next,
        )
        .unwrap();
    verify(&scene, &next, &cases, dt, 1e-8);
    verify_narrow_acceleration(&scene, &next, &cases, dt);
}

#[test]
fn cpu_standalone_narrow_hinges_preserve_authored_stop_response() {
    let dt = 0.00005;
    let (scene, cases) = fixture_at(dt, true, false, 1e-8);
    assert_eq!(cases.len(), 32);
    let initial = scene.state().clone_host().unwrap();
    let mut next = scene.model().state().unwrap();
    FeatherstoneSolver::new_cpu()
        .step(
            scene.model(),
            &initial,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            dt,
            &mut next,
        )
        .unwrap();
    verify(&scene, &next, &cases, dt, 1e-8);
    verify_narrow_acceleration(&scene, &next, &cases, dt);
    verify_retained_rotation_advance(&scene, &next, &[], &cases, dt);
}

fn verify_narrow_stop_hold(scene: &RigidScene, next: &State, residues: &[f32], cases: &[Case]) {
    for (i, case) in cases.iter().enumerate() {
        let joint = scene.joint_index(case.joint).unwrap().index();
        let c = scene.model().joint_coord_offset.host().unwrap()[joint] as usize;
        let d = scene.model().joint_dof_offset.host().unwrap()[joint] as usize;
        let lo = scene.model().joint_limit_lower.host().unwrap();
        let hi = scene.model().joint_limit_upper.host().unwrap();
        let left = Quat::from_rotation_x(-lo[d + 3]);
        let right = Quat::from_rotation_z(-lo[d + 5]);
        let center = 0.5 * lo[d + 4] + 0.5 * hi[d + 4];
        let angle = |q: Quat| {
            let relative = left * q.normalize() * right;
            let value = 2. * relative.y.atan2(relative.w);
            value
                + std::f64::consts::TAU * (((center - value) / std::f64::consts::TAU + 0.5).floor())
        };
        let initial = &scene.state().joint_q.host().unwrap()[c + 3..c + 7];
        let initial_angle = angle(Quat::from_xyzw(
            initial[0], initial[1], initial[2], initial[3],
        ));
        let start_lower = (initial_angle - lo[d + 4]).abs() < (initial_angle - hi[d + 4]).abs();
        let finish_lower = if case.rate.length_squared() > 0. {
            !start_lower
        } else {
            start_lower
        };
        let expected = if finish_lower { lo[d + 4] } else { hi[d + 4] };
        let retained = Quat::from_array(std::array::from_fn(|axis| {
            next.joint_q.host().unwrap()[c + 3 + axis]
                + residues.get(c + 3 + axis).copied().unwrap_or(0.) as f64
        }))
        .normalize();
        let error = (angle(retained) - expected).abs();
        let relative = left * retained * right;
        let body = scene.body_index(case.body).unwrap().index();
        let rate = next.body_qd.host().unwrap()[body];
        println!(
            "narrow_stop_hold case={i} endpoint_error={error:.9e} angular_rate={:.9e}",
            rate.angular.length()
        );
        assert!(error <= 1e-12);
        assert!(relative.x.abs() <= 1e-10 && relative.z.abs() <= 1e-10);
        assert!(
            next.joint_qd.host().unwrap()[d..d + 6]
                .iter()
                .all(|rate| rate.abs() <= 1e-7)
        );
        assert!(rate.angular.length() <= 1e-7 && rate.linear.length() <= 1e-7);
    }
}

#[test]
fn cpu_standalone_narrow_hinges_advance_to_opposite_stop_and_hold() {
    let dt = 0.00005;
    let (scene, cases) = fixture_at(dt, true, false, 1e-8);
    let mut current = scene.state().clone_host().unwrap();
    let mut next = scene.model().state().unwrap();
    let mut solver = FeatherstoneSolver::new_cpu();
    for step in 0..8 {
        solver
            .step(
                scene.model(),
                &current,
                scene.control(),
                &molla_sim::Contacts::with_capacity(0),
                dt,
                &mut next,
            )
            .unwrap();
        if step == 0 {
            verify_narrow_acceleration(&scene, &next, &cases, dt);
            verify_retained_rotation_advance(&scene, &next, &[], &cases, dt);
        }
        std::mem::swap(&mut current, &mut next);
    }
    verify_narrow_stop_hold(&scene, &current, &[], &cases);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_standalone_narrow_hinges_advance_to_opposite_stop_and_hold() {
    let dt = 0.00005;
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt} width=1e-8 steps=8");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut scene, cases) = fixture_at(dt, true, false, 1e-8);
    scene.upload_model_to_device(device, queue).unwrap();
    scene.upload_controls_to_device(device, queue).unwrap();
    let mut current = scene.state().clone_host().unwrap();
    let mut next = scene.model().state().unwrap();
    current.upload_to_device(device, queue).unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, scene.model()).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&[&current, &next])
        .unwrap();
    solver.seed_gpu_coordinate_compensation(&current).unwrap();
    for step in 0..8 {
        solver
            .step(
                scene.model(),
                &current,
                scene.control(),
                &molla_sim::Contacts::with_capacity(0),
                dt,
                &mut next,
            )
            .unwrap();
        assert_eq!(
            gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
            0
        );
        if step == 0 {
            next.download_from_device(device, queue).unwrap();
            next.joint_q.download(device, queue).unwrap();
            let residues = gpu_readback::read::<f32>(
                &engine,
                solver.gpu_coordinate_compensation_buffer(&next).unwrap(),
                scene.model().joint_coord_total,
            );
            verify_narrow_acceleration(&scene, &next, &cases, dt);
            verify_retained_rotation_advance(&scene, &next, &residues, &cases, dt);
        }
        std::mem::swap(&mut current, &mut next);
    }
    current.download_from_device(device, queue).unwrap();
    current.joint_q.download(device, queue).unwrap();
    current.joint_qd.download(device, queue).unwrap();
    let residues = gpu_readback::read::<f32>(
        &engine,
        solver.gpu_coordinate_compensation_buffer(&current).unwrap(),
        scene.model().joint_coord_total,
    );
    verify_narrow_stop_hold(&scene, &current, &residues, &cases);
}

#[test]
fn cpu_standalone_regular_intervals_and_caps_use_coupled_response() {
    let dt = 0.00001;
    for caps in [false, true] {
        let (scene, cases) = fixture_regular_intervals(dt, caps);
        let initial = scene.state().clone_host().unwrap();
        let mut next = scene.model().state().unwrap();
        FeatherstoneSolver::new_cpu()
            .step(
                scene.model(),
                &initial,
                scene.control(),
                &molla_sim::Contacts::with_capacity(0),
                dt,
                &mut next,
            )
            .unwrap();
        verify(&scene, &next, &cases, dt, 1e-8);
        verify_narrow_acceleration(&scene, &next, &cases, dt);
        verify_regular_retained_bounds(&scene, &next, &[], &cases, caps);
    }
}

#[test]
fn cpu_runtime_regular_intervals_and_caps_preserve_reactions() {
    let dt = 0.00001;
    for caps in [false, true] {
        let (scene, cases) = fixture_regular_intervals(dt, caps);
        runtime(RigidWorld::from_scene(scene).unwrap(), &cases, dt, 1e-8);
    }
}

fn verify_near_pole_bounds(scene: &RigidScene, next: &State, residues: &[f32], cases: &[Case]) {
    for (i, case) in cases.iter().enumerate() {
        let joint = scene.joint_index(case.joint).unwrap().index();
        let c = scene.model().joint_coord_offset.host().unwrap()[joint] as usize;
        let d = scene.model().joint_dof_offset.host().unwrap()[joint] as usize;
        let retained = Quat::from_array(std::array::from_fn(|axis| {
            next.joint_q.host().unwrap()[c + 3 + axis]
                + residues.get(c + 3 + axis).copied().unwrap_or(0.) as f64
        }))
        .normalize();
        let matrix = Mat3::from_quat(retained);
        let pitch = matrix
            .z_axis
            .x
            .atan2(matrix.x_axis.x.hypot(matrix.y_axis.x));
        let lo = scene.model().joint_limit_lower.host().unwrap()[d + 4];
        let hi = scene.model().joint_limit_upper.host().unwrap()[d + 4];
        let center = 0.5 * lo + 0.5 * hi;
        let violation = [pitch, std::f64::consts::PI - pitch]
            .into_iter()
            .map(|value| {
                let angle = value
                    + std::f64::consts::TAU
                        * (((center - value) / std::f64::consts::TAU + 0.5).floor());
                (lo - angle).max(angle - hi).max(0.)
            })
            .fold(f64::INFINITY, f64::min);
        println!("near_pole_pitch_retained case={i} violation={violation:.9e}");
        assert!(violation.is_finite() && violation <= 1e-12);
    }
}

#[test]
fn cpu_standalone_near_pole_pitch_intervals_use_authored_references() {
    let dt = 0.00001;
    let (scene, cases) = fixture_near_pole_intervals(dt);
    let initial = scene.state().clone_host().unwrap();
    let mut next = scene.model().state().unwrap();
    FeatherstoneSolver::new_cpu()
        .step(
            scene.model(),
            &initial,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            dt,
            &mut next,
        )
        .unwrap();
    verify(&scene, &next, &cases, dt, 1e-8);
    verify_narrow_acceleration(&scene, &next, &cases, dt);
    verify_retained_rotation_advance(&scene, &next, &[], &cases, dt);
    verify_near_pole_bounds(&scene, &next, &[], &cases);
}

#[test]
fn cpu_runtime_near_pole_pitch_intervals_preserve_reactions() {
    let dt = 0.00001;
    let (scene, cases) = fixture_near_pole_intervals(dt);
    runtime(RigidWorld::from_scene(scene).unwrap(), &cases, dt, 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_standalone_near_pole_pitch_intervals_use_authored_references() {
    let dt = 0.00001;
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt} width=1e-9 pole_offset=1e-7");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut scene, cases) = fixture_near_pole_intervals(dt);
    scene.upload_model_to_device(device, queue).unwrap();
    scene.upload_controls_to_device(device, queue).unwrap();
    let mut initial = scene.state().clone_host().unwrap();
    let mut next = scene.model().state().unwrap();
    initial.upload_to_device(device, queue).unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut solver = FeatherstoneSolver::new_gpu(&engine, scene.model()).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&[&initial, &next])
        .unwrap();
    solver.seed_gpu_coordinate_compensation(&initial).unwrap();
    solver
        .step(
            scene.model(),
            &initial,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            dt,
            &mut next,
        )
        .unwrap();
    next.download_from_device(device, queue).unwrap();
    assert_eq!(
        gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
        0
    );
    next.joint_q.download(device, queue).unwrap();
    let residues = gpu_readback::read::<f32>(
        &engine,
        solver.gpu_coordinate_compensation_buffer(&next).unwrap(),
        scene.model().joint_coord_total,
    );
    verify(&scene, &next, &cases, dt, 1e-3);
    verify_narrow_acceleration(&scene, &next, &cases, dt);
    verify_retained_rotation_advance(&scene, &next, &residues, &cases, dt);
    verify_near_pole_bounds(&scene, &next, &residues, &cases);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_runtime_near_pole_pitch_intervals_preserve_reactions() {
    let dt = 0.00001;
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt} width=1e-9 pole_offset=1e-7");
    let (scene, cases) = fixture_near_pole_intervals(dt);
    runtime(
        RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        &cases,
        dt,
        1e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_standalone_regular_intervals_and_caps_use_coupled_response() {
    let dt = 0.00001;
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt} width=1e-9");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for caps in [false, true] {
        let (mut scene, cases) = fixture_regular_intervals(dt, caps);
        scene.upload_model_to_device(device, queue).unwrap();
        scene.upload_controls_to_device(device, queue).unwrap();
        let mut initial = scene.state().clone_host().unwrap();
        let mut next = scene.model().state().unwrap();
        initial.upload_to_device(device, queue).unwrap();
        next.upload_to_device(device, queue).unwrap();
        let mut solver = FeatherstoneSolver::new_gpu(&engine, scene.model()).unwrap();
        solver
            .prepare_gpu_coordinate_compensation(&[&initial, &next])
            .unwrap();
        solver.seed_gpu_coordinate_compensation(&initial).unwrap();
        solver
            .step(
                scene.model(),
                &initial,
                scene.control(),
                &molla_sim::Contacts::with_capacity(0),
                dt,
                &mut next,
            )
            .unwrap();
        next.download_from_device(device, queue).unwrap();
        assert_eq!(
            gpu_readback::read::<u32>(&engine, solver.gpu_status_buffer().unwrap(), 1)[0],
            0
        );
        verify(&scene, &next, &cases, dt, 1e-3);
        verify_narrow_acceleration(&scene, &next, &cases, dt);
        next.joint_q.download(device, queue).unwrap();
        next.joint_qd.download(device, queue).unwrap();
        let residues = gpu_readback::read::<f32>(
            &engine,
            solver.gpu_coordinate_compensation_buffer(&next).unwrap(),
            scene.model().joint_coord_total,
        );
        verify_retained_rotation_advance(&scene, &next, &residues, &cases, dt);
        verify_regular_retained_bounds(&scene, &next, &residues, &cases, caps);
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_runtime_regular_intervals_and_caps_preserve_reactions() {
    let dt = 0.00001;
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?} dt={dt} width=1e-9");
    for caps in [false, true] {
        let (scene, cases) = fixture_regular_intervals(dt, caps);
        runtime(
            RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
            &cases,
            dt,
            1e-3,
        );
    }
}

fn verify_regular_retained_bounds(
    scene: &RigidScene,
    next: &State,
    residues: &[f32],
    cases: &[Case],
    caps: bool,
) {
    for (i, case) in cases.iter().enumerate() {
        let joint = scene.joint_index(case.joint).unwrap().index();
        let dof = scene.model().joint_dof_offset.host().unwrap()[joint] as usize;
        if caps {
            for axis in 0..3 {
                if scene.model().joint_velocity_limit.host().unwrap()[dof + 3 + axis].is_finite() {
                    assert!(next.joint_qd.host().unwrap()[dof + 3 + axis].abs() <= 1e-9);
                }
            }
            continue;
        }
        let coordinate = scene.model().joint_coord_offset.host().unwrap()[joint] as usize;
        let values: [f64; 4] = std::array::from_fn(|axis| {
            let index = coordinate + 3 + axis;
            next.joint_q.host().unwrap()[index] + residues.get(index).copied().unwrap_or(0.) as f64
        });
        let (x, y, z) = Quat::from_array(values)
            .normalize()
            .to_euler(glam::EulerRot::XYZ);
        let primary = [x, y, z];
        let alternate = [
            x + std::f64::consts::PI,
            std::f64::consts::PI - y,
            z + std::f64::consts::PI,
        ];
        let violation = [primary, alternate]
            .into_iter()
            .map(|candidate| {
                (0..3)
                    .map(|axis| {
                        let lo = scene.model().joint_limit_lower.host().unwrap()[dof + 3 + axis];
                        let hi = scene.model().joint_limit_upper.host().unwrap()[dof + 3 + axis];
                        assert!(hi - lo > 1e-12);
                        let center = 0.5 * lo + 0.5 * hi;
                        let angle = candidate[axis]
                            + std::f64::consts::TAU
                                * (((center - candidate[axis]) / std::f64::consts::TAU + 0.5)
                                    .floor());
                        (lo - angle).max(angle - hi).max(0.)
                    })
                    .fold(0., f64::max)
            })
            .fold(f64::INFINITY, f64::min);
        println!("regular_interval_retained case={i} violation={violation:.9e}");
        assert!(violation.is_finite() && violation <= 1e-12);
    }
}
