use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc, RigidScene};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn mass(com: Vec3, value: f64) -> MassProperties {
    MassProperties {
        mass: value,
        local_com: com,
        inertia: Mat3::from_cols(
            Vec3::new(3.0, 0.2, 0.3),
            Vec3::new(0.2, 4.0, 0.4),
            Vec3::new(0.3, 0.4, 5.0),
        ) * value,
    }
}

fn edits(create: impl FnOnce(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let removed = scene.insert_body(BodyDesc::default()).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams {
                pose: Transform {
                    position: Vec3::new(1.0, 0.4, -0.2),
                    rotation: Quat::from_rotation_y(0.7),
                },
                ..Default::default()
            },
            additional_mass: Some(mass(Vec3::new(0.2, -0.1, 0.3), 2.0)),
            velocity: SpatialVector::new(Vec3::new(0.1, 0.2, -0.3), Vec3::new(0.3, -0.2, 0.1)),
            ..Default::default()
        })
        .unwrap();
    let force = Vec3::new(2.0, -1.0, 3.0);
    let point = Vec3::new(0.7, 0.2, -0.1);
    scene.add_body_force_at_point(body, force, point).unwrap();
    let mut world = create(scene);
    world.solver.substeps = 4;
    for (stage, com) in [
        Vec3::ZERO,
        Vec3::new(-0.3, 0.2, 0.1),
        Vec3::new(0.4, -0.1, -0.2),
    ]
    .into_iter()
    .enumerate()
    {
        world.step(0.01).unwrap();
        let pose = world.scene.body_pose(body).unwrap();
        let velocity_at_point = world.scene.body_velocity_at_point(body, point).unwrap();
        let old_com = world.scene.body_world_com(body).unwrap();
        let old_force = world.scene.state().body_f.host().unwrap()
            [world.scene.body_index(body).unwrap().index()];
        let torque = old_force.angular + old_com.cross(old_force.linear);
        let properties = mass(com, 3.0 + stage as f64);
        world
            .scene
            .set_body_mass_properties(body, properties)
            .unwrap();
        if stage == 0 {
            world.scene.remove_body(removed).unwrap();
        }
        assert_eq!(world.scene.body_pose(body).unwrap(), pose);
        assert!(
            (world.scene.body_velocity_at_point(body, point).unwrap() - velocity_at_point).length()
                < 1e-10
        );
        let index = world.scene.body_index(body).unwrap().index();
        let root = world
            .scene
            .model()
            .joint_child
            .host()
            .unwrap()
            .iter()
            .position(|&child| child as usize == index)
            .unwrap();
        assert_eq!(
            world.scene.model().joint_x_c.host().unwrap()[root].position,
            com
        );
        let coordinate = world.scene.model().joint_coord_offset.host().unwrap()[root] as usize;
        let dof = world.scene.model().joint_dof_offset.host().unwrap()[root] as usize;
        let q = world.scene.state().joint_q.host().unwrap();
        let qd = world.scene.state().joint_qd.host().unwrap();
        let center = world.scene.body_world_com(body).unwrap();
        let velocity = world.scene.body_velocity(body).unwrap().linear;
        assert!(
            (Vec3::new(q[coordinate], q[coordinate + 1], q[coordinate + 2]) - center).length()
                < 1e-10
        );
        assert!((Vec3::new(qd[dof], qd[dof + 1], qd[dof + 2]) - velocity).length() < 1e-10);
        let applied = world.scene.state().body_f.host().unwrap()[index];
        assert_eq!(applied.linear, force);
        assert!((applied.angular + center.cross(force) - torque).length() < 1e-10);
        for frame in 1..=120 {
            world.step(1.0 / 120.0).unwrap();
            let momentum = world.scene.body_velocity(body).unwrap().linear * properties.mass;
            let expected = velocity * properties.mass + force * (frame as f64 / 120.0);
            assert!(
                (momentum - expected).length() < tolerance,
                "stage={stage} frame={frame} momentum={momentum:?} expected={expected:?}"
            );
        }
    }
}

#[test]
fn cpu_com_edits_reseed_implicit_roots_and_preserve_physical_state() {
    edits(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_com_edits_reseed_implicit_roots_and_preserve_physical_state() {
    use molla_compute::{Backend, Engine};
    let engine = std::sync::Arc::new(Engine::new(Backend::AutoGpu).unwrap());
    edits(
        |scene| RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        1e-3,
    );
}

fn authored_roots(mut create: impl FnMut(RigidScene) -> RigidWorld, relative_tolerance: f64) {
    let mut failures = Vec::new();
    for kind in [JointType::Free, JointType::D6] {
        for frames in [60, 120, 240] {
            let mut scene = RigidScene::new().unwrap();
            scene.set_gravity(Vec3::ZERO).unwrap();
            let properties = mass(Vec3::new(0.2, -0.1, 0.3), 2.0);
            let pose = Transform {
                position: Vec3::new(1.0, -0.4, 0.2),
                rotation: Quat::from_rotation_y(0.7) * Quat::from_rotation_x(-0.2),
            };
            let frame_child = Transform {
                position: Vec3::new(-0.3, 0.2, -0.1),
                rotation: Quat::from_rotation_z(0.4) * Quat::from_rotation_x(0.3),
            };
            let body = scene
                .insert_body(BodyDesc {
                    params: BodyParams::new().with_pose(pose),
                    additional_mass: Some(properties),
                    ..Default::default()
                })
                .unwrap();
            let joint = scene
                .insert_joint(JointDesc {
                    parent: None,
                    child: body,
                    kind,
                    axis: Vec3::X,
                    frame_parent: pose.compose(&frame_child),
                    frame_child,
                })
                .unwrap();
            let velocity = SpatialVector::new(Vec3::new(0.1, 0.2, -0.3), Vec3::new(0.3, -0.2, 0.1));
            scene.set_body_velocity(body, velocity).unwrap();
            let mut world = create(scene);
            world.solver.substeps = 4;
            world.set_reaction_readout(true).unwrap();
            let momentum = |scene: &RigidScene| {
                let rotation = scene.body_pose(body).unwrap().rotation;
                let velocity = scene.body_velocity(body).unwrap();
                SpatialVector::new(
                    velocity.linear * properties.mass,
                    rotation * (properties.inertia * (rotation.conjugate() * velocity.angular)),
                )
            };
            let initial = momentum(&world.scene);
            let scale = [
                initial.linear.length().max(1e-3),
                initial.angular.length().max(1e-3),
            ];
            let mut peak = [0.0_f64; 2];
            for _ in 0..frames {
                world.step(1.0 / frames as f64).unwrap();
                let current = momentum(&world.scene);
                assert!(current.linear.is_finite() && current.angular.is_finite());
                let reaction = world.joint_reaction(joint).unwrap();
                assert!(reaction.child.linear.length() < relative_tolerance);
                assert!(reaction.child.angular.length() < relative_tolerance);
                peak[0] = peak[0].max((current.linear - initial.linear).length() / scale[0]);
                peak[1] = peak[1].max((current.angular - initial.angular).length() / scale[1]);
            }
            let desc = world.scene.joint(joint).unwrap();
            assert_eq!(desc.frame_child, frame_child);
            assert_eq!(desc.frame_parent, pose.compose(&frame_child));
            println!(
                "authored root {kind:?} frames={frames} substeps=4 relative momentum errors={peak:?}"
            );
            if peak.iter().any(|&error| error > relative_tolerance) {
                failures.push((kind, frames, peak));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "authored-root momentum failures: {failures:?}"
    );
}

#[test]
fn cpu_authored_free_and_d6_frames_preserve_free_system_momentum() {
    authored_roots(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_authored_free_and_d6_frames_preserve_free_system_momentum() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    authored_roots(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        1e-3,
    );
}

fn momentum(scene: &RigidScene, bodies: &[BodyHandle], origin: Vec3) -> SpatialVector {
    bodies.iter().fold(SpatialVector::ZERO, |sum, &body| {
        let properties = scene.body_mass_properties(body).unwrap();
        let pose = scene.body_pose(body).unwrap();
        let velocity = scene.body_velocity(body).unwrap();
        let p = velocity.linear * properties.mass;
        let h =
            pose.rotation * (properties.inertia * (pose.rotation.conjugate() * velocity.angular));
        sum + SpatialVector::new(
            p,
            h + (scene.body_world_com(body).unwrap() - origin).cross(p),
        )
    })
}

fn centroid(scene: &RigidScene, bodies: &[BodyHandle]) -> Vec3 {
    let total: f64 = bodies
        .iter()
        .map(|&b| scene.body_mass_properties(b).unwrap().mass)
        .sum();
    bodies
        .iter()
        .map(|&b| {
            scene.body_world_com(b).unwrap() * (scene.body_mass_properties(b).unwrap().mass / total)
        })
        .sum()
}

fn check_fk(scene: &RigidScene, tolerance: f64) {
    let mut rebuilt = scene.state().clone_host().unwrap();
    molla_sim::eval_fk(scene.model(), &mut rebuilt).unwrap();
    for (actual, expected) in scene
        .state()
        .body_q
        .host()
        .unwrap()
        .iter()
        .zip(rebuilt.body_q.host().unwrap())
    {
        assert!(
            (actual.position - expected.position).length() < tolerance,
            "FK mismatch {actual:?} != {expected:?}"
        );
        assert!(actual.rotation.dot(expected.rotation).abs() > 1.0 - tolerance);
    }
}

fn loaded_root(mut create: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for kind in [JointType::Free, JointType::D6] {
        for frames in [60, 120, 240] {
            let mut scene = RigidScene::new().unwrap();
            let gravity = Vec3::new(0.2, -0.4, 0.1);
            scene.set_gravity(gravity).unwrap();
            let properties = mass(Vec3::new(0.3, -0.2, 0.1), 2.0);
            let pose = Transform::new(Vec3::new(1.0, -0.5, 0.4), Quat::from_rotation_y(0.7));
            let body = scene
                .insert_body(BodyDesc {
                    params: BodyParams::new().with_pose(pose),
                    additional_mass: Some(properties),
                    ..Default::default()
                })
                .unwrap();
            let frame = Transform::new(Vec3::new(-0.2, 0.3, -0.1), Quat::from_rotation_z(0.3));
            let joint = scene
                .insert_joint(JointDesc {
                    parent: None,
                    child: body,
                    kind,
                    axis: Vec3::X,
                    frame_parent: pose.compose(&frame),
                    frame_child: frame,
                })
                .unwrap();
            let velocity = Vec3::new(0.1, 0.2, -0.3);
            scene
                .set_body_velocity(
                    body,
                    SpatialVector::new(velocity, Vec3::new(0.3, -0.2, 0.1)),
                )
                .unwrap();
            let persistent =
                SpatialVector::new(Vec3::new(0.4, -0.2, 0.6), Vec3::new(0.2, -0.1, 0.3));
            let control =
                SpatialVector::new(Vec3::new(-0.1, 0.3, -0.2), Vec3::new(-0.05, 0.1, -0.15));
            scene.add_body_wrench(body, persistent).unwrap();
            let index = scene.body_index(body).unwrap().index();
            scene.control_mut().body_external_f.host_mut().unwrap()[index] = control;
            let mut world = create(scene);
            world.solver.substeps = 4;
            world.set_reaction_readout(true).unwrap();
            let center = world.scene.body_world_com(body).unwrap();
            let initial = momentum(&world.scene, &[body], center);
            let acceleration = (persistent.linear + control.linear) / properties.mass + gravity;
            let torque = persistent.angular + control.angular;
            let mut peaks = [0.0_f64; 3];
            for frame in 1..=frames {
                world.step(1.0 / frames as f64).unwrap();
                let time = frame as f64 / frames as f64;
                let h = 1.0 / (frames * 4) as f64;
                let expected_center =
                    center + velocity * time + acceleration * (0.5 * time * (time + h));
                let actual = momentum(
                    &world.scene,
                    &[body],
                    world.scene.body_world_com(body).unwrap(),
                );
                peaks[0] = peaks[0].max(
                    (actual.linear - initial.linear - acceleration * (properties.mass * time))
                        .length(),
                );
                peaks[1] =
                    peaks[1].max((actual.angular - initial.angular - torque * time).length());
                peaks[2] = peaks[2]
                    .max((world.scene.body_world_com(body).unwrap() - expected_center).length());
                let reaction = world.joint_reaction(joint).unwrap();
                assert!(reaction.child.linear.length() < tolerance);
                assert!(reaction.child.angular.length() < tolerance);
                check_fk(&world.scene, tolerance);
            }
            println!(
                "loaded root {kind:?} frames={frames} substeps=4 absolute P/L/centroid errors={peaks:?}"
            );
            assert!(peaks.iter().all(|&error| error < tolerance));
        }
    }
}

fn moving_tree(mut create: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for kind in [JointType::Free, JointType::D6] {
        for frames in [60, 120, 240] {
            let mut scene = RigidScene::new().unwrap();
            scene.set_gravity(Vec3::ZERO).unwrap();
            let origin = Vec3::new(30.0, -10.0, 5.0);
            let bodies: Vec<_> = (0..4)
                .map(|i| {
                    scene
                        .insert_body(BodyDesc {
                            params: BodyParams::new().with_pose(Transform::new(
                                origin + Vec3::new(0.7, 0.3, -0.2) * i as f64,
                                Quat::from_rotation_y(0.2 * i as f64) * Quat::from_rotation_x(-0.2),
                            )),
                            additional_mass: Some(mass(Vec3::new(0.2, -0.1, 0.3), 2.0 + i as f64)),
                            ..Default::default()
                        })
                        .unwrap()
                })
                .collect();
            let child_frame =
                Transform::new(Vec3::new(-0.3, 0.2, -0.1), Quat::from_rotation_z(0.4));
            let root = scene
                .insert_joint(JointDesc {
                    parent: None,
                    child: bodies[0],
                    kind,
                    axis: Vec3::X,
                    frame_parent: scene.body_pose(bodies[0]).unwrap().compose(&child_frame),
                    frame_child: child_frame,
                })
                .unwrap();
            let mut joints = Vec::new();
            for (parent, child, kind) in [
                (0, 1, JointType::Revolute),
                (1, 2, JointType::Prismatic),
                (0, 3, JointType::Ball),
            ] {
                let anchor = Transform::new(
                    origin + Vec3::new(0.4, 0.2, -0.1) * child as f64,
                    Quat::from_rotation_z(0.2 * child as f64),
                );
                joints.push(
                    scene
                        .insert_joint(JointDesc {
                            parent: Some(bodies[parent]),
                            child: bodies[child],
                            kind,
                            axis: Vec3::Z,
                            frame_parent: scene
                                .body_pose(bodies[parent])
                                .unwrap()
                                .inverse()
                                .compose(&anchor),
                            frame_child: scene
                                .body_pose(bodies[child])
                                .unwrap()
                                .inverse()
                                .compose(&anchor),
                        })
                        .unwrap(),
                );
            }
            let mut rates = scene.state().joint_qd.host().unwrap().to_vec();
            let mut forces = vec![0.0; rates.len()];
            for (i, joint) in std::iter::once(root).chain(joints).enumerate() {
                let joint = scene.joint_index(joint).unwrap().index();
                let offsets = scene.model().joint_dof_offset.host().unwrap();
                for (lane, dof) in
                    (offsets[joint] as usize..offsets[joint + 1] as usize).enumerate()
                {
                    rates[dof] = [0.1, -0.2, 0.3, -0.15, 0.25, -0.1][lane] * (i + 1) as f64;
                    if i != 0 {
                        forces[dof] = 0.3 * (lane + 1) as f64;
                    }
                }
            }
            scene
                .set_generalized_velocities_with_wake(&rates, true)
                .unwrap();
            scene
                .control_mut()
                .joint_force
                .host_mut()
                .unwrap()
                .copy_from_slice(&forces);
            let mut world = create(scene);
            world.solver.substeps = 4;
            world.set_reaction_readout(true).unwrap();
            let initial = momentum(&world.scene, &bodies, origin);
            let center = centroid(&world.scene, &bodies);
            let total_mass: f64 = bodies
                .iter()
                .map(|&b| world.scene.body_mass_properties(b).unwrap().mass)
                .sum();
            let mut peaks = [0.0_f64; 3];
            for frame in 1..=frames {
                world.step(1.0 / frames as f64).unwrap();
                let current = momentum(&world.scene, &bodies, origin);
                peaks[0] = peaks[0].max(
                    (current.linear - initial.linear).length() / initial.linear.length().max(1e-3),
                );
                peaks[1] = peaks[1].max(
                    (current.angular - initial.angular).length()
                        / initial.angular.length().max(1e-3),
                );
                let expected =
                    center + initial.linear * (frame as f64 / frames as f64 / total_mass);
                peaks[2] = peaks[2].max((centroid(&world.scene, &bodies) - expected).length());
                let reaction = world.joint_reaction(root).unwrap();
                assert!(reaction.child.linear.length() < tolerance);
                assert!(reaction.child.angular.length() < tolerance);
                check_fk(&world.scene, tolerance);
            }
            println!(
                "moving tree {kind:?} frames={frames} substeps=4 relative P/L and absolute centroid errors={peaks:?}"
            );
            assert!(peaks.iter().all(|&error| error < tolerance));
        }
    }
}

#[test]
fn cpu_floating_roots_integrate_applied_wrenches_and_gravity() {
    loaded_root(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[test]
fn cpu_floating_branched_trees_preserve_momentum_under_internal_efforts() {
    moving_tree(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_floating_roots_integrate_applied_wrenches_and_gravity() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    loaded_root(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        1e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_floating_branched_trees_preserve_momentum_under_internal_efforts() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    moving_tree(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        1e-3,
    );
}
