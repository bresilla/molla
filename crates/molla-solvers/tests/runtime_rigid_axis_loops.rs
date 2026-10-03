use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxes, JointAxis};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, JointBreakLimits, JointBreakTrigger, JointDesc, JointHandle, RigidScene,
};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

struct Case {
    body: BodyHandle,
    joint: JointHandle,
    axis: JointAxis,
    origin: Vec3,
    rotation: Quat,
    direction: Vec3,
    force: Vec3,
    torque: Vec3,
    mass: f64,
    position: f64,
    velocity: f64,
    acceleration: f64,
    bounded: bool,
    seen: [bool; 2],
}

const LIMITS: [f64; 2] = [-0.004, 0.006];

fn fixture(limits: [f64; 2]) -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    let rotation = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(-0.4);
    let local_axis = Vec3::new(1.0, 2.0, -3.0).normalize();
    let alignment = Quat::from_rotation_arc(Vec3::X, local_axis);
    for (kind, axis) in [
        (JointType::Revolute, JointAxis::AngX),
        (JointType::Prismatic, JointAxis::LinX),
    ]
    .into_iter()
    .chain(JointAxis::ALL.into_iter().map(|axis| (JointType::D6, axis)))
    {
        for mass in [0.01, 1.0, 100.0] {
            let origin = Vec3::new(3.0 * cases.len() as f64, 0.5, -0.2);
            let angular = axis as usize >= 3;
            let index = axis as usize % 3;
            let frame_rotation = rotation * alignment;
            let direction = frame_rotation * [Vec3::X, Vec3::Y, Vec3::Z][index];
            let transverse = frame_rotation * [Vec3::Y, Vec3::Z, Vec3::X][index];
            let position = if limits[0] == limits[1] {
                limits[0]
            } else {
                0.0
            };
            let velocity = if limits[0] == limits[1] { 0.0 } else { 0.2 };
            let pose = if angular {
                Transform::new(
                    origin,
                    Quat::from_axis_angle(direction, position) * rotation,
                )
            } else {
                Transform::new(origin + direction * position, rotation)
            };
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
            scene
                .set_body_velocity(
                    body,
                    if angular {
                        SpatialVector::new(Vec3::ZERO, direction * velocity)
                    } else {
                        SpatialVector::new(direction * velocity, Vec3::ZERO)
                    },
                )
                .unwrap();
            let force = transverse * (0.3 * mass)
                + if angular {
                    Vec3::ZERO
                } else {
                    direction * (0.4 * mass)
                };
            let torque = transverse * (0.2 * mass)
                + if angular {
                    direction * (0.4 * mass)
                } else {
                    Vec3::ZERO
                };
            scene
                .add_body_wrench(body, SpatialVector::new(force, torque))
                .unwrap();
            let joint = scene
                .insert_rigid_loop_joint(JointDesc {
                    parent: None,
                    child: body,
                    kind,
                    axis: local_axis,
                    frame_parent: Transform::new(
                        origin,
                        if kind == JointType::D6 {
                            frame_rotation
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
            scene.set_joint_limits(joint, axis, Some(limits)).unwrap();
            assert!(scene.joint_index(joint).is_none());
            cases.push(Case {
                body,
                joint,
                axis,
                origin,
                rotation,
                direction,
                force,
                torque,
                mass,
                position,
                velocity,
                acceleration: 0.4,
                bounded: true,
                seen: [false; 2],
            });
        }
    }
    (scene, cases)
}

fn close(actual: Vec3, expected: Vec3, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).length() <= tolerance,
        "{label}: {actual:?} != {expected:?}"
    );
}

fn certify(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64, limits: [f64; 2]) {
    for hz in [60, 240] {
        for substeps in [1, 4] {
            let (scene, mut cases) = fixture(limits);
            let mut world = make_world(scene);
            world.solver.substeps = substeps;
            world.set_reaction_readout(true).unwrap();
            let dt = 1.0 / hz as f64;
            let h = dt / substeps as f64;
            for frame in 0..64 {
                for case in &mut cases {
                    let angular = case.axis as usize >= 3;
                    if frame == 24 {
                        case.acceleration = -1.2;
                        let delta = case.direction * (-1.6 * case.mass);
                        if angular {
                            case.torque += delta;
                        } else {
                            case.force += delta;
                        }
                        world
                            .scene
                            .add_body_wrench(
                                case.body,
                                if angular {
                                    SpatialVector::new(Vec3::ZERO, delta)
                                } else {
                                    SpatialVector::new(delta, Vec3::ZERO)
                                },
                            )
                            .unwrap();
                    }
                    if frame == 48 || frame == 56 {
                        case.bounded = frame == 56;
                        let before = world.scene.body_pose(case.body).unwrap();
                        world
                            .scene
                            .set_joint_limits(case.joint, case.axis, case.bounded.then_some(limits))
                            .unwrap();
                        assert_eq!(world.scene.body_pose(case.body).unwrap(), before);
                    }
                }
                let mut expected = Vec::new();
                for case in &mut cases {
                    let angular = case.axis as usize >= 3;
                    let mut samples = Vec::new();
                    for _ in 0..substeps {
                        let old = case.velocity;
                        let position = case.position;
                        let predicted = old + case.acceleration * h;
                        let bounds = [(limits[0] - position) / h, (limits[1] - position) / h];
                        case.velocity = if case.bounded {
                            predicted.clamp(bounds[0], bounds[1])
                        } else {
                            predicted
                        };
                        if case.bounded {
                            case.seen[0] |= predicted < bounds[0];
                            case.seen[1] |= predicted > bounds[1];
                        }
                        let support = case.direction * (case.mass * (case.velocity - old) / h);
                        let child = if angular {
                            SpatialVector::new(-case.force, support - case.torque)
                        } else {
                            SpatialVector::new(support - case.force, -case.torque)
                        };
                        let delta = if angular {
                            Vec3::ZERO
                        } else {
                            case.direction * position
                        };
                        let parent = SpatialVector::new(
                            -child.linear,
                            -child.angular - delta.cross(child.linear),
                        );
                        samples.push((child, parent));
                        case.position += if angular {
                            2.0 * (0.5 * case.velocity * h).atan()
                        } else {
                            case.velocity * h
                        };
                    }
                    expected.push(samples);
                }
                world.step(dt).unwrap_or_else(|error| {
                    panic!("hz={hz} substeps={substeps} frame={frame}: {error}")
                });
                assert!(world.quarantined_bodies().is_empty());
                for (case, samples) in cases.iter().zip(&expected) {
                    let angular = case.axis as usize >= 3;
                    let pose = world.scene.body_pose(case.body).unwrap();
                    close(
                        pose.position,
                        case.origin
                            + if angular {
                                Vec3::ZERO
                            } else {
                                case.direction * case.position
                            },
                        tolerance,
                        "position",
                    );
                    let reference = if angular {
                        Quat::from_axis_angle(case.direction, case.position) * case.rotation
                    } else {
                        case.rotation
                    };
                    let actual = if pose.rotation.dot(reference) < 0.0 {
                        -pose.rotation
                    } else {
                        pose.rotation
                    };
                    assert!(
                        (actual - reference).length() < tolerance,
                        "orientation hz={hz} substeps={substeps} frame={frame} axis={:?}",
                        case.axis
                    );
                    let velocity = world.scene.body_velocity(case.body).unwrap();
                    close(
                        velocity.linear,
                        if angular {
                            Vec3::ZERO
                        } else {
                            case.direction * case.velocity
                        },
                        tolerance,
                        "linear rate",
                    );
                    close(
                        velocity.angular,
                        if angular {
                            case.direction * case.velocity
                        } else {
                            Vec3::ZERO
                        },
                        tolerance,
                        "angular rate",
                    );
                    let actual: Vec<_> = world
                        .joint_reaction_samples()
                        .unwrap()
                        .iter()
                        .filter(|sample| sample.joint == case.joint)
                        .collect();
                    assert_eq!(actual.len(), substeps);
                    let mut mean = SpatialVector::ZERO;
                    for (sample, &(child, parent)) in actual.iter().zip(samples) {
                        let label = format!(
                            "hz={hz} substeps={substeps} frame={frame} axis={:?} mass={} origin={:?} sample={}",
                            case.axis, case.mass, case.origin, sample.substep
                        );
                        let bound = tolerance
                            * (1.0 + case.mass + child.linear.length() + child.angular.length());
                        close(sample.child.linear, child.linear, bound, &label);
                        close(sample.child.angular, child.angular, bound, "sample torque");
                        close(sample.parent.linear, parent.linear, bound, "parent force");
                        close(
                            sample.parent.angular,
                            parent.angular,
                            bound,
                            "parent torque",
                        );
                        mean = mean + child.scale(1.0 / substeps as f64);
                    }
                    let reading = world.joint_reaction(case.joint).unwrap();
                    let bound = tolerance
                        * (1.0 + case.mass + mean.linear.length() + mean.angular.length());
                    close(reading.child.linear, mean.linear, bound, "mean force");
                    close(reading.child.angular, mean.angular, bound, "mean torque");
                }
            }
            assert!(cases.iter().all(|case| case.seen == [true, true]));
            eprintln!(
                "rigid-axis loops hz={hz} substeps={substeps} limits={limits:?}: 24 world closures, both bounds and live limit edits passed"
            );
        }
    }
}

#[test]
fn cpu_rigid_axis_loops_match_independent_hard_bound_motion_and_wrenches() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8, LIMITS);
}

#[test]
fn cpu_rigid_axis_loops_match_nonzero_equal_bound_motion_and_wrenches() {
    certify(
        |scene| RigidWorld::from_scene(scene).unwrap(),
        1e-8,
        [0.003; 2],
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan GPU; use make test-gpu"]
fn gpu_rigid_axis_loops_match_independent_hard_bound_motion_and_wrenches() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
        LIMITS,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan GPU; use make test-gpu"]
fn gpu_rigid_axis_loops_match_nonzero_equal_bound_motion_and_wrenches() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
        [0.003; 2],
    );
}

fn solve_projection(mut matrix: [[f64; 6]; 6], mut rhs: [f64; 6]) -> [f64; 6] {
    for column in 0..6 {
        let pivot = (column..6)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        matrix.swap(column, pivot);
        rhs.swap(column, pivot);
        let diagonal = matrix[column][column];
        assert!(diagonal.abs() > 1e-12);
        for value in &mut matrix[column][column..] {
            *value /= diagonal;
        }
        rhs[column] /= diagonal;
        let row = matrix[column];
        let value = rhs[column];
        for i in 0..6 {
            if i == column {
                continue;
            }
            let factor = matrix[i][column];
            for j in column..6 {
                matrix[i][j] -= factor * row[j];
            }
            rhs[i] -= factor * value;
        }
    }
    rhs
}

struct ProjectionCase {
    bodies: [BodyHandle; 2],
    joint: JointHandle,
    poses: [Transform; 2],
    rates: [SpatialVector; 2],
    impulse: SpatialVector,
    scale: f64,
}

fn projection(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    let dt = 1.0 / 240.0;
    let frame_rotation = Quat::from_rotation_z(0.35) * Quat::from_rotation_y(-0.2);
    let axes = [Vec3::X, Vec3::Y, Vec3::Z].map(|axis| frame_rotation * axis);
    let rotations = [Quat::from_rotation_z(0.6), Quat::from_rotation_x(-0.4)];
    let arms = [Vec3::new(0.3, -0.2, 0.1), Vec3::new(-0.1, 0.25, -0.2)];
    let coms = [Vec3::new(0.04, -0.03, 0.02), Vec3::new(-0.02, 0.05, -0.01)];
    let velocities = [Vec3::new(0.3, -0.2, 0.1), Vec3::new(-0.2, 0.1, -0.3)];
    let inertia_local = Mat3::from_cols(
        Vec3::new(2.0, 0.2, -0.1),
        Vec3::new(0.2, 1.5, 0.15),
        Vec3::new(-0.1, 0.15, 1.0),
    );
    for (kind, free) in [
        (JointType::Revolute, JointAxis::AngX),
        (JointType::Prismatic, JointAxis::LinX),
    ]
    .into_iter()
    .chain(JointAxis::ALL.into_iter().map(|axis| (JointType::D6, axis)))
    {
        for equality in [false, true] {
            for scale in [0.01, 1.0, 100.0] {
                let anchor = Vec3::new(cases.len() as f64 * 2.0, 0.1, -0.15);
                let masses = [scale, 3.0 * scale];
                let poses = [0, 1].map(|i| Transform::new(anchor - arms[i], rotations[i]));
                let bodies = [0, 1].map(|i| {
                    let body = scene
                        .insert_body(BodyDesc {
                            params: BodyParams::new().with_pose(poses[i]),
                            additional_mass: Some(MassProperties {
                                mass: masses[i],
                                local_com: coms[i],
                                inertia: inertia_local * masses[i],
                            }),
                            ..Default::default()
                        })
                        .unwrap();
                    scene
                        .set_body_velocity(body, SpatialVector::new(velocities[i], Vec3::ZERO))
                        .unwrap();
                    body
                });
                let joint = scene
                    .insert_rigid_loop_joint(JointDesc {
                        parent: Some(bodies[0]),
                        child: bodies[1],
                        kind,
                        axis: Vec3::X,
                        frame_parent: Transform::new(
                            rotations[0].conjugate() * arms[0],
                            rotations[0].conjugate() * frame_rotation,
                        ),
                        frame_child: Transform::new(
                            rotations[1].conjugate() * arms[1],
                            rotations[1].conjugate() * frame_rotation,
                        ),
                    })
                    .unwrap();
                if kind == JointType::D6 {
                    scene
                        .set_joint_locked_axes(joint, JointAxes(63 & !(1 << free as u32)))
                        .unwrap();
                }
                if equality {
                    scene.set_joint_limits(joint, free, Some([0.0; 2])).unwrap();
                }
                let basis: Vec<_> = JointAxis::ALL
                    .into_iter()
                    .filter(|&axis| equality || axis != free)
                    .map(|axis| {
                        if (axis as usize) < 3 {
                            SpatialVector::new(axes[axis as usize], Vec3::ZERO)
                        } else {
                            SpatialVector::new(Vec3::ZERO, axes[axis as usize % 3])
                        }
                    })
                    .collect();
                let centers = [0, 1].map(|i| poses[i].position + rotations[i] * coms[i]);
                let offsets = [0, 1].map(|i| anchor - centers[i]);
                let inertias = [0, 1].map(|i| {
                    let rotation = Mat3::from_quat(rotations[i]);
                    rotation * inertia_local * masses[i] * rotation.transpose()
                });
                let inverse = inertias.map(|inertia| inertia.inverse());
                let mut matrix = [[0.0; 6]; 6];
                let mut rhs = [0.0; 6];
                for (column, wrench) in basis.iter().enumerate() {
                    let mut response = SpatialVector::ZERO;
                    for i in 0..2 {
                        let angular =
                            inverse[i] * (offsets[i].cross(wrench.linear) + wrench.angular);
                        response = response
                            + SpatialVector::new(
                                wrench.linear / masses[i] + angular.cross(offsets[i]),
                                angular,
                            );
                    }
                    for (row, direction) in basis.iter().enumerate() {
                        matrix[row][column] = direction.linear.dot(response.linear)
                            + direction.angular.dot(response.angular);
                    }
                    rhs[column] = wrench.linear.dot(velocities[0] - velocities[1]);
                }
                if basis.len() == 5 {
                    matrix[5][5] = 1.0;
                }
                let lambda = solve_projection(matrix, rhs);
                let impulse = basis
                    .iter()
                    .zip(lambda)
                    .fold(SpatialVector::ZERO, |sum, (wrench, value)| {
                        sum + wrench.scale(value)
                    });
                let mut final_poses = poses;
                let rates = [0, 1].map(|i| {
                    let sign = if i == 0 { -1.0 } else { 1.0 };
                    let linear = velocities[i] + impulse.linear * (sign / masses[i]);
                    let angular =
                        inverse[i] * (offsets[i].cross(impulse.linear) + impulse.angular) * sign;
                    let qdot =
                        Quat::from_xyzw(angular.x * 0.5, angular.y * 0.5, angular.z * 0.5, 0.0)
                            * rotations[i];
                    let rotation = (rotations[i] + qdot * dt).normalize();
                    final_poses[i] =
                        Transform::new(centers[i] + linear * dt - rotation * coms[i], rotation);
                    let r = Mat3::from_quat(rotation);
                    let final_inertia = r * inertia_local * masses[i] * r.transpose();
                    SpatialVector::new(linear, final_inertia.inverse() * (inertias[i] * angular))
                });
                cases.push(ProjectionCase {
                    bodies,
                    joint,
                    poses: final_poses,
                    rates,
                    impulse,
                    scale,
                });
            }
        }
    }
    let mut world = make_world(scene);
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    world.step(dt).unwrap();
    assert!(world.quarantined_bodies().is_empty());
    for case in &cases {
        let bound = tolerance * (1.0 + case.scale);
        let reading = world.joint_reaction(case.joint).unwrap();
        close(
            reading.child.linear,
            case.impulse.linear / dt,
            bound / dt,
            "floating force",
        );
        close(
            reading.child.angular,
            case.impulse.angular / dt,
            bound / dt,
            "floating torque",
        );
        close(
            reading.parent.linear,
            -case.impulse.linear / dt,
            bound / dt,
            "floating parent force",
        );
        close(
            reading.parent.angular,
            -case.impulse.angular / dt,
            bound / dt,
            "floating parent torque",
        );
        for i in 0..2 {
            let velocity = world.scene.body_velocity(case.bodies[i]).unwrap();
            close(
                velocity.linear,
                case.rates[i].linear,
                tolerance,
                "floating linear rate",
            );
            close(
                velocity.angular,
                case.rates[i].angular,
                tolerance,
                "floating angular rate",
            );
            let pose = world.scene.body_pose(case.bodies[i]).unwrap();
            close(
                pose.position,
                case.poses[i].position,
                tolerance,
                "floating pose",
            );
            let expected = case.poses[i].rotation;
            let actual = if pose.rotation.dot(expected) < 0.0 {
                -pose.rotation
            } else {
                pose.rotation
            };
            assert!((actual - expected).length() <= tolerance);
        }
    }
    eprintln!(
        "48 floating hinge/slider/D6 pairs: independent five/six-row impulse, COM, full inertia and both-anchor wrenches passed"
    );
}

#[test]
fn cpu_rigid_axis_loops_match_independent_floating_impulse_projection() {
    projection(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan GPU; use make test-gpu"]
fn gpu_rigid_axis_loops_match_independent_floating_impulse_projection() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    projection(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-4,
    );
}

fn breaks(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for capture in [false, true] {
        let (scene, cases) = fixture(LIMITS);
        let mut world = make_world(scene);
        world.solver.substeps = 1;
        world.set_reaction_readout(capture).unwrap();
        let expected: Vec<_> = cases
            .iter()
            .map(|case| {
                let support = case.direction * (0.4 * case.mass);
                if case.axis as usize >= 3 {
                    SpatialVector::new(-case.force, support - case.torque)
                } else {
                    SpatialVector::new(support - case.force, -case.torque)
                }
            })
            .collect();
        for (i, (case, wrench)) in cases.iter().zip(&expected).enumerate() {
            world
                .scene
                .set_joint_break_limits(
                    case.joint,
                    JointBreakLimits {
                        force: (i % 3 != 1).then_some(wrench.linear.length() * 0.6),
                        torque: (i % 3 != 0).then_some(wrench.angular.length() * 0.6),
                    },
                )
                .unwrap();
        }
        world.step(1.0 / 240.0).unwrap();
        assert_eq!(world.joint_break_events().len(), cases.len());
        for (i, (case, wrench)) in cases.iter().zip(&expected).enumerate() {
            let event = world
                .joint_break_events()
                .iter()
                .find(|event| event.joint == case.joint)
                .unwrap();
            assert_eq!(
                event.trigger,
                [
                    JointBreakTrigger::Force,
                    JointBreakTrigger::Torque,
                    JointBreakTrigger::Both
                ][i % 3]
            );
            assert_eq!(event.sample.substep, 0);
            let bound =
                tolerance * (1.0 + case.mass + wrench.linear.length() + wrench.angular.length());
            close(
                event.sample.child.linear,
                wrench.linear,
                bound,
                "axis break force",
            );
            close(
                event.sample.child.angular,
                wrench.angular,
                bound,
                "axis break torque",
            );
            assert!(world.scene.joint_broken(case.joint).unwrap());
            world
                .scene
                .set_joint_break_limits(case.joint, JointBreakLimits::default())
                .unwrap();
            assert!(world.scene.joint_broken(case.joint).unwrap());
            let pose = world.scene.body_pose(case.body).unwrap();
            let velocity = world.scene.body_velocity(case.body).unwrap();
            world.scene.repair_joint(case.joint).unwrap();
            assert_eq!(world.scene.body_pose(case.body).unwrap(), pose);
            assert_eq!(world.scene.body_velocity(case.body).unwrap(), velocity);
        }
        world.step(1.0 / 240.0).unwrap();
        assert!(world.quarantined_bodies().is_empty());
        assert_eq!(world.joint_break_events().len(), cases.len());
        for case in &cases {
            assert!(!world.scene.joint_broken(case.joint).unwrap());
        }
        assert_eq!(world.drain_joint_break_events().len(), cases.len());
        eprintln!(
            "24 rigid axis closures: force/torque/both breaks, capture={capture}, exact repair state and follow-up step passed"
        );
    }
}

#[test]
fn cpu_rigid_axis_loop_break_channels_and_explicit_repair_preserve_state() {
    breaks(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan GPU; use make test-gpu"]
fn gpu_rigid_axis_loop_break_channels_and_explicit_repair_preserve_state() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    breaks(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}
