use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, JointBreakLimits, JointBreakTrigger, JointDesc, JointHandle, RigidScene,
};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

struct Case {
    bodies: Vec<BodyHandle>,
    joint: JointHandle,
    poses: Vec<Transform>,
    force: Vec3,
    torque: Vec3,
    anchor: Vec3,
}

fn body(scene: &mut RigidScene, pose: Transform, mass: f64, com: Vec3) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(pose),
            additional_mass: Some(MassProperties {
                mass,
                local_com: com,
                inertia: Mat3::from_cols(
                    Vec3::new(2.0, 0.2, -0.1),
                    Vec3::new(0.2, 1.5, 0.15),
                    Vec3::new(-0.1, 0.15, 1.0),
                ) * mass,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn fixture() -> (RigidScene, Vec<Case>) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut cases = Vec::new();
    for scale in [0.01, 1.0, 100.0] {
        for floating in [false, true] {
            let anchor = Vec3::new(cases.len() as f64 * 3.0, 0.3, -0.2);
            let frame_rotation = Quat::from_rotation_z(0.6) * Quat::from_rotation_x(-0.4);
            let rotations = [Quat::from_rotation_y(0.3), Quat::from_rotation_x(-0.5)];
            let arms = [Vec3::new(0.1, 0.2, -0.15), Vec3::new(-0.2, 0.1, 0.3)];
            let coms = [Vec3::new(0.04, -0.03, 0.02), Vec3::new(-0.03, 0.02, 0.04)];
            let poses =
                [0, 1].map(|i| Transform::new(anchor - rotations[i] * arms[i], rotations[i]));
            let parent = floating.then(|| body(&mut scene, poses[0], scale, coms[0]));
            let child = body(&mut scene, poses[1], 3.0 * scale, coms[1]);
            let force = Vec3::new(0.3, -0.2, 0.15) * scale;
            let torque = Vec3::new(-0.1, 0.25, 0.18) * scale;
            let child_arm = rotations[1] * (arms[1] - coms[1]);
            scene
                .add_body_wrench(
                    child,
                    SpatialVector::new(-force, -torque - child_arm.cross(force)),
                )
                .unwrap();
            if let Some(parent) = parent {
                let parent_arm = rotations[0] * (arms[0] - coms[0]);
                scene
                    .add_body_wrench(
                        parent,
                        SpatialVector::new(force, torque + parent_arm.cross(force)),
                    )
                    .unwrap();
            }
            let joint = scene
                .insert_rigid_loop_joint(JointDesc {
                    parent,
                    child,
                    kind: JointType::Fixed,
                    axis: Vec3::X,
                    frame_parent: parent.map_or(Transform::new(anchor, frame_rotation), |_| {
                        Transform::new(arms[0], rotations[0].conjugate() * frame_rotation)
                    }),
                    frame_child: Transform::new(arms[1], rotations[1].conjugate() * frame_rotation),
                })
                .unwrap();
            assert!(scene.joint_index(joint).is_none());
            cases.push(Case {
                bodies: parent.into_iter().chain([child]).collect(),
                joint,
                poses: if floating {
                    poses.to_vec()
                } else {
                    vec![poses[1]]
                },
                force,
                torque,
                anchor,
            });
        }
    }
    (scene, cases)
}

fn vector(actual: Vec3, expected: Vec3, tolerance: f64, label: &str) {
    assert!(
        (actual - expected).length() <= tolerance,
        "{label}: {actual:?} != {expected:?}"
    );
}

fn certify(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for hz in [60, 120, 240] {
        for substeps in [1, 4] {
            let (scene, cases) = fixture();
            let mut world = make_world(scene);
            world.solver.substeps = substeps;
            world.set_reaction_readout(true).unwrap();
            for frame in 0..16 {
                world.step(1.0 / hz as f64).unwrap();
                assert_eq!(
                    world.joint_reaction_samples().unwrap().len(),
                    cases.len() * substeps
                );
                for case in &cases {
                    let bound = tolerance * (1.0 + case.force.length() + case.torque.length());
                    let reaction = world.joint_reaction(case.joint).unwrap();
                    vector(reaction.child.linear, case.force, bound, "mean force");
                    vector(reaction.child.angular, case.torque, bound, "mean torque");
                    vector(reaction.parent.linear, -case.force, bound, "parent force");
                    vector(
                        reaction.parent.angular,
                        -case.torque,
                        bound,
                        "parent torque",
                    );
                    for sample in world
                        .joint_reaction_samples()
                        .unwrap()
                        .iter()
                        .filter(|s| s.joint == case.joint)
                    {
                        let label = format!(
                            "hz={hz} substeps={substeps} frame={frame} joint={:?} sample={}",
                            case.joint, sample.substep
                        );
                        vector(
                            sample.child.linear,
                            case.force,
                            bound,
                            &format!("{label} sample force"),
                        );
                        vector(sample.child.angular, case.torque, bound, &label);
                        vector(sample.anchors[0], case.anchor, tolerance, "parent anchor");
                        vector(sample.anchors[1], case.anchor, tolerance, "child anchor");
                    }
                    for (&body, expected) in case.bodies.iter().zip(&case.poses) {
                        let pose = world.scene.body_pose(body).unwrap();
                        vector(pose.position, expected.position, tolerance, "position");
                        assert!(pose.rotation.dot(expected.rotation).abs() > 1.0 - tolerance);
                        let velocity = world.scene.body_velocity(body).unwrap();
                        vector(velocity.linear, Vec3::ZERO, tolerance, "linear rate");
                        vector(velocity.angular, Vec3::ZERO, tolerance, "angular rate");
                    }
                }
            }
        }
    }
}

#[test]
fn cpu_rigid_fixed_loops_preserve_rotated_frames_and_balance_anchor_wrenches() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[test]
fn cpu_rigid_fixed_loops_correct_displacements_above_coordinate_roundoff() {
    let (mut scene, cases) = fixture();
    let displacement = Vec3::new(1e-5, -2e-5, 3e-5);
    for case in cases.iter().filter(|case| case.bodies.len() == 1) {
        let body = case.bodies[0];
        let mut pose = scene.body_pose(body).unwrap();
        pose.position += displacement;
        scene.set_body_pose(body, pose).unwrap();
    }
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.solver.substeps = 1;
    world.step(0.001).unwrap();
    for case in cases.iter().filter(|case| case.bodies.len() == 1) {
        let body = case.bodies[0];
        vector(
            world.scene.body_pose(body).unwrap().position,
            case.poses[0].position,
            1e-8,
            "recovered position",
        );
        vector(
            world.scene.body_velocity(body).unwrap().linear,
            -displacement / 0.001,
            1e-8,
            "physical correction rate",
        );
    }
    assert!(world.quarantined_bodies().is_empty());
    assert_eq!(world.completed_steps(), 1);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan GPU"]
fn gpu_rigid_fixed_loops_preserve_rotated_frames_and_balance_anchor_wrenches() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    certify(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}

fn solve(mut matrix: [[f64; 7]; 6]) -> [f64; 6] {
    for column in 0..6 {
        let pivot = (column..6)
            .max_by(|&a, &b| matrix[a][column].abs().total_cmp(&matrix[b][column].abs()))
            .unwrap();
        matrix.swap(column, pivot);
        let diagonal = matrix[column][column];
        assert!(diagonal.abs() > 1e-12);
        for value in &mut matrix[column][column..] {
            *value /= diagonal;
        }
        let row = matrix[column];
        for (index, target) in matrix.iter_mut().enumerate() {
            if index == column {
                continue;
            }
            let multiplier = target[column];
            for entry in column..7 {
                target[entry] -= multiplier * row[entry];
            }
        }
    }
    matrix.map(|row| row[6])
}

fn projection(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for scale in [0.01, 1.0, 100.0] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let masses = [scale, 3.0 * scale];
        let rotations = [Quat::from_rotation_z(0.6), Quat::from_rotation_x(-0.4)];
        let arms = [Vec3::new(0.3, -0.2, 0.1), Vec3::new(-0.1, 0.25, -0.2)];
        let local_arms = [0, 1].map(|i| rotations[i].conjugate() * arms[i]);
        let poses = [0, 1].map(|i| Transform::new(-arms[i], rotations[i]));
        let velocities = [Vec3::new(0.3, -0.2, 0.1), Vec3::new(-0.2, 0.1, -0.3)];
        let bodies = [0, 1].map(|i| body(&mut scene, poses[i], masses[i], Vec3::ZERO));
        for i in 0..2 {
            scene
                .set_body_velocity(bodies[i], SpatialVector::new(velocities[i], Vec3::ZERO))
                .unwrap();
        }
        let joint = scene
            .insert_rigid_loop_joint(JointDesc {
                parent: Some(bodies[0]),
                child: bodies[1],
                kind: JointType::Fixed,
                axis: Vec3::X,
                frame_parent: Transform::new(local_arms[0], rotations[0].conjugate()),
                frame_child: Transform::new(local_arms[1], rotations[1].conjugate()),
            })
            .unwrap();
        let inertia_local = Mat3::from_cols(
            Vec3::new(2.0, 0.2, -0.1),
            Vec3::new(0.2, 1.5, 0.15),
            Vec3::new(-0.1, 0.15, 1.0),
        );
        let inertias = [0, 1].map(|i| {
            let rotation = Mat3::from_quat(rotations[i]);
            rotation * inertia_local * masses[i] * rotation.transpose()
        });
        let inverse = inertias.map(|inertia| inertia.inverse());
        let mut gram = [[0.0; 7]; 6];
        for column in 0..6 {
            let mut force = Vec3::ZERO;
            let mut torque = Vec3::ZERO;
            if column < 3 {
                force[column] = 1.0;
            } else {
                torque[column - 3] = 1.0;
            }
            let mut linear_response = Vec3::ZERO;
            let mut angular_response = Vec3::ZERO;
            for i in 0..2 {
                let angular = inverse[i] * (arms[i].cross(force) + torque);
                linear_response += force / masses[i] + angular.cross(arms[i]);
                angular_response += angular;
            }
            for row in 0..3 {
                gram[row][column] = linear_response[row];
                gram[row + 3][column] = angular_response[row];
            }
        }
        for row in 0..3 {
            gram[row][6] = velocities[0][row] - velocities[1][row];
        }
        let lambda = solve(gram);
        let force_impulse = Vec3::new(lambda[0], lambda[1], lambda[2]);
        let torque_impulse = Vec3::new(lambda[3], lambda[4], lambda[5]);
        let linear = [0, 1]
            .map(|i| velocities[i] + force_impulse * if i == 0 { -1.0 } else { 1.0 } / masses[i]);
        let angular = [0, 1].map(|i| {
            inverse[i]
                * (arms[i].cross(force_impulse) + torque_impulse)
                * if i == 0 { -1.0 } else { 1.0 }
        });
        let dt = 1.0 / 240.0;
        let mut world = make_world(scene);
        world.set_reaction_readout(true).unwrap();
        world.solver.substeps = 1;
        world.step(dt).unwrap();
        let bound = tolerance * (1.0 + scale);
        let reaction = world.joint_reaction(joint).unwrap();
        vector(
            reaction.child.linear,
            force_impulse / dt,
            bound * 240.0,
            "projected force",
        );
        vector(
            reaction.child.angular,
            torque_impulse / dt,
            bound * 240.0,
            "projected torque",
        );
        for i in 0..2 {
            let velocity = world.scene.body_velocity(bodies[i]).unwrap();
            vector(
                velocity.linear,
                linear[i],
                tolerance,
                "projected linear rate",
            );
            let q = rotations[i];
            let qdot = Quat::from_xyzw(
                angular[i].x * 0.5,
                angular[i].y * 0.5,
                angular[i].z * 0.5,
                0.0,
            ) * q;
            let expected = (q + qdot * dt).normalize();
            let rotation = Mat3::from_quat(expected);
            let final_inertia = rotation * inertia_local * masses[i] * rotation.transpose();
            let transported = final_inertia.inverse() * (inertias[i] * angular[i]);
            vector(
                velocity.angular,
                transported,
                tolerance,
                "transported angular rate",
            );
            let pose = world.scene.body_pose(bodies[i]).unwrap();
            vector(
                pose.position,
                poses[i].position + linear[i] * dt,
                tolerance,
                "projected pose",
            );
            assert!(pose.rotation.dot(expected).abs() > 1.0 - tolerance);
        }
        vector(
            linear[0] * masses[0] + linear[1] * masses[1],
            velocities[0] * masses[0] + velocities[1] * masses[1],
            bound,
            "linear momentum",
        );
        let before = poses[0].position.cross(velocities[0] * masses[0])
            + poses[1].position.cross(velocities[1] * masses[1]);
        let after = poses[0].position.cross(linear[0] * masses[0])
            + poses[1].position.cross(linear[1] * masses[1])
            + inertias[0] * angular[0]
            + inertias[1] * angular[1];
        vector(after, before, bound, "angular momentum");
    }
}

#[test]
fn cpu_rigid_fixed_loop_projection_matches_independent_six_dimensional_impulse() {
    projection(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan GPU"]
fn gpu_rigid_fixed_loop_projection_matches_independent_six_dimensional_impulse() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    projection(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-4,
    );
}

fn breaks(mut make_world: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for capture in [false, true] {
        let (scene, cases) = fixture();
        let mut world = make_world(scene);
        world.solver.substeps = 1;
        world.set_reaction_readout(capture).unwrap();
        for (i, case) in cases.iter().enumerate() {
            world
                .scene
                .set_joint_break_limits(
                    case.joint,
                    JointBreakLimits {
                        force: (i % 3 != 1).then_some(case.force.length() * 0.6),
                        torque: (i % 3 != 0).then_some(case.torque.length() * 0.6),
                    },
                )
                .unwrap();
        }
        world.step(1.0 / 240.0).unwrap();
        assert_eq!(world.joint_break_events().len(), cases.len());
        for (i, case) in cases.iter().enumerate() {
            let event = world
                .joint_break_events()
                .iter()
                .find(|event| event.joint == case.joint)
                .unwrap();
            let expected = [
                JointBreakTrigger::Force,
                JointBreakTrigger::Torque,
                JointBreakTrigger::Both,
            ][i % 3];
            assert_eq!(event.trigger, expected);
            assert_eq!(event.sample.substep, 0);
            let bound = tolerance * (1.0 + case.force.length() + case.torque.length());
            vector(event.sample.child.linear, case.force, bound, "break force");
            vector(
                event.sample.child.angular,
                case.torque,
                bound,
                "break torque",
            );
            assert!(world.scene.joint_broken(case.joint).unwrap());
            world
                .scene
                .set_joint_break_limits(case.joint, JointBreakLimits::default())
                .unwrap();
            assert!(world.scene.joint_broken(case.joint).unwrap());
            let before: Vec<_> = case
                .bodies
                .iter()
                .map(|&body| {
                    (
                        world.scene.body_pose(body).unwrap(),
                        world.scene.body_velocity(body).unwrap(),
                    )
                })
                .collect();
            world.scene.repair_joint(case.joint).unwrap();
            for (&body, &(pose, velocity)) in case.bodies.iter().zip(&before) {
                assert_eq!(world.scene.body_pose(body).unwrap(), pose);
                assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
            }
        }
        world.step(1.0 / 240.0).unwrap();
        assert_eq!(world.joint_break_events().len(), cases.len());
        assert!(world.quarantined_bodies().is_empty());
        for case in &cases {
            assert!(!world.scene.joint_broken(case.joint).unwrap());
            for (&body, pose) in case.bodies.iter().zip(&case.poses) {
                vector(
                    world.scene.body_pose(body).unwrap().position,
                    pose.position,
                    tolerance,
                    "repaired pose",
                );
            }
        }
        assert_eq!(world.drain_joint_break_events().len(), cases.len());
    }
}

#[test]
fn cpu_rigid_fixed_loop_break_channels_and_explicit_repair_preserve_state() {
    breaks(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires Vulkan GPU"]
fn gpu_rigid_fixed_loop_break_channels_and_explicit_repair_preserve_state() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    breaks(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-3,
    );
}
