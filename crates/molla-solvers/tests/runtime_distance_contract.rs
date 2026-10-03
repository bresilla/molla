use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, DistanceLimits, JointDesc, JointHandle, RigidScene,
};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::{DistanceBound, RigidWorld};

#[cfg(feature = "gpu")]
fn hardware_engine() -> std::sync::Arc<molla_compute::Engine> {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(
        !adapter.is_software,
        "distance contract requires a hardware GPU"
    );
    println!("adapter={adapter:?}");
    engine
}

fn body(scene: &mut RigidScene, x: f64, mass: f64) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            params: BodyParams {
                pose: Transform::from_translation(Vec3::X * x),
                ..Default::default()
            },
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * mass,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn link(
    scene: &mut RigidScene,
    a: BodyHandle,
    b: BodyHandle,
    limits: DistanceLimits,
) -> JointHandle {
    let id = scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Distance,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene.set_distance_limits(id, limits).unwrap();
    id
}

fn exchange(mut create: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    for (case, (length, min, max, velocity)) in [
        (0.0, Some(1.0), None, Vec3::ZERO),
        (0.0, Some(1.0), Some(1.0), Vec3::ZERO),
        (0.0, None, Some(0.0), Vec3::new(0.2, -0.1, 0.3)),
        (0.0, Some(0.0), None, Vec3::new(-0.2, 0.1, -0.3)),
        (0.5, Some(1.0), None, -Vec3::X),
        (1.5, None, Some(1.0), Vec3::X),
        (1.0, Some(1.0), Some(1.0), Vec3::Y * 0.2),
        (1.0, Some(0.5), Some(1.5), Vec3::X * 0.8),
        (1.0, None, None, Vec3::new(0.1, 0.2, -0.3)),
        (1.0, Some(1.0), Some(1.0), Vec3::new(0.1, 0.2, -0.3)),
    ]
    .into_iter()
    .enumerate()
    {
        let mut worlds = Vec::new();
        for reverse in [false, true] {
            let mut scene = RigidScene::new().unwrap();
            scene.set_gravity(Vec3::ZERO).unwrap();
            let a = body(&mut scene, 0.0, 2.0);
            let b = body(&mut scene, length, 3.0);
            scene
                .set_body_velocity(b, SpatialVector::new(velocity, Vec3::ZERO))
                .unwrap();
            let ends = if reverse { [b, a] } else { [a, b] };
            let joint = link(&mut scene, ends[0], ends[1], DistanceLimits { min, max });
            if case == 9 {
                let frames = [
                    Transform {
                        position: Vec3::new(0.0, 0.3, 0.1),
                        rotation: Quat::from_rotation_z(0.7),
                    },
                    Transform {
                        position: Vec3::new(0.0, -0.2, 0.1),
                        rotation: Quat::from_rotation_x(-0.4),
                    },
                ];
                for (i, id) in [a, b].into_iter().enumerate() {
                    scene
                        .set_body_mass_properties(
                            id,
                            MassProperties {
                                mass: [2.0, 3.0][i],
                                local_com: Vec3::new(0.1, -0.2, 0.05),
                                inertia: Mat3::from_cols(
                                    Vec3::new(3.0, 0.2, 0.3),
                                    Vec3::new(0.2, 4.0, 0.4),
                                    Vec3::new(0.3, 0.4, 5.0),
                                ),
                            },
                        )
                        .unwrap();
                    let mut pose = scene.body_pose(id).unwrap();
                    pose.rotation = Quat::from_rotation_y([0.3, -0.2][i]);
                    scene.set_body_pose(id, pose).unwrap();
                    scene
                        .set_body_velocity(
                            id,
                            SpatialVector::new(
                                if i == 0 { Vec3::ZERO } else { velocity },
                                Vec3::new(0.1, -0.2, 0.3),
                            ),
                        )
                        .unwrap();
                }
                let separation = (scene.body_pose(b).unwrap().compose(&frames[1]).position
                    - scene.body_pose(a).unwrap().compose(&frames[0]).position)
                    .length();
                scene
                    .set_joint_frames(
                        joint,
                        frames[usize::from(reverse)],
                        frames[usize::from(!reverse)],
                    )
                    .unwrap();
                scene
                    .set_distance_limits(
                        joint,
                        DistanceLimits {
                            min: Some(separation),
                            max: Some(separation),
                        },
                    )
                    .unwrap();
            }
            let mut world = create(scene);
            world.solver.substeps = 4;
            worlds.push((world, [a, b], joint));
        }
        let mut peak: f64 = 0.0;
        for frame in 0..120 {
            for (world, _, _) in &mut worlds {
                world.step(1.0 / 120.0).unwrap();
            }
            for body in 0..2 {
                let (w0, b0, _) = &worlds[0];
                let (w1, b1, _) = &worlds[1];
                let p0 = w0.scene.body_pose(b0[body]).unwrap();
                let p1 = w1.scene.body_pose(b1[body]).unwrap();
                let v0 = w0.scene.body_velocity(b0[body]).unwrap();
                let v1 = w1.scene.body_velocity(b1[body]).unwrap();
                let dp = (p0.position - p1.position).length();
                peak = peak.max(dp);
                assert!(
                    dp < tolerance && (v0.linear - v1.linear).length() < tolerance,
                    "case={case} frame={frame} body={body} dp={dp}, v0={v0:?}, v1={v1:?}"
                );
                assert!((v0.angular - v1.angular).length() < tolerance);
                assert!(p0.rotation.dot(p1.rotation).abs() > 1.0 - tolerance);
            }
            let readings: Vec<_> = worlds
                .iter()
                .map(|(w, _, j)| *w.distance_reading(*j).unwrap())
                .collect();
            assert_eq!(
                readings[0].active, readings[1].active,
                "case={case} frame={frame}"
            );
            let impulse_error = (readings[0].force + readings[1].force).length() / 120.0;
            assert!(
                impulse_error < tolerance,
                "case={case} frame={frame} impulse error={impulse_error}"
            );
            for (world, bodies, joint) in &worlds {
                let momentum = world.scene.body_velocity(bodies[0]).unwrap().linear * 2.0
                    + world.scene.body_velocity(bodies[1]).unwrap().linear * 3.0;
                assert!(
                    (momentum - velocity * 3.0).length() < tolerance,
                    "case={case} frame={frame} momentum={momentum:?} expected={:?} error={}",
                    velocity * 3.0,
                    (momentum - velocity * 3.0).length()
                );
                assert!(world.distance_reading(*joint).unwrap().violation < 1e-3);
            }
        }
        println!("distance endpoint exchange case={case} duration=1s peak position={peak}");
    }
}

#[test]
fn cpu_distance_endpoint_exchange_including_coincident_positive_bounds() {
    exchange(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-6);
}

fn rotating_free_body(create: impl FnOnce(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let id = body(&mut scene, 0.0, 2.0);
    scene
        .set_body_mass_properties(
            id,
            MassProperties {
                mass: 2.0,
                local_com: Vec3::new(0.1, -0.2, 0.05),
                inertia: Mat3::from_cols(
                    Vec3::new(3.0, 0.2, 0.3),
                    Vec3::new(0.2, 4.0, 0.4),
                    Vec3::new(0.3, 0.4, 5.0),
                ),
            },
        )
        .unwrap();
    scene
        .set_body_pose(
            id,
            Transform {
                position: Vec3::ZERO,
                rotation: Quat::from_rotation_y(0.3),
            },
        )
        .unwrap();
    let velocity = Vec3::new(0.1, 0.2, -0.3);
    scene
        .set_body_velocity(id, SpatialVector::new(velocity, Vec3::new(0.1, -0.2, 0.3)))
        .unwrap();
    let mut world = create(scene);
    world.solver.substeps = 4;
    let mut peak: f64 = 0.0;
    for _ in 0..120 {
        world.step(1.0 / 120.0).unwrap();
        peak = peak.max((world.scene.body_velocity(id).unwrap().linear - velocity).length() * 2.0);
    }
    println!("unconstrained rotating offset COM: one-second peak momentum error={peak}");
    assert!(peak < tolerance);
}

#[test]
fn cpu_rotating_offset_com_free_body_preserves_linear_momentum() {
    rotating_free_body(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_rotating_offset_com_free_body_preserves_linear_momentum() {
    let engine = hardware_engine();
    rotating_free_body(
        |scene| RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        1e-3,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_distance_endpoint_exchange_including_coincident_positive_bounds() {
    let engine = hardware_engine();
    exchange(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        1e-3,
    );
}

fn mutation(mut create: impl FnMut(RigidScene) -> RigidWorld, tolerance: f64) {
    use molla_sim::joint_control::{JointAxis, JointMotor};
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let removed = body(&mut scene, -10.0, 1.0);
    let a = body(&mut scene, 0.0, 2.0);
    let b = body(&mut scene, 1.0, 3.0);
    let rotor = body(&mut scene, 10.0, 1.0);
    let motor_joint = scene
        .insert_joint(JointDesc {
            parent: None,
            child: rotor,
            kind: JointType::Revolute,
            axis: Vec3::Z,
            frame_parent: Transform::from_translation(Vec3::X * 10.0),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let motor = JointMotor {
        mode: molla_sim::JointTargetMode::Effort,
        target_position: 0.5,
        max_force: 1.0,
        ..Default::default()
    };
    scene
        .set_joint_motor(motor_joint, JointAxis::AngX, motor)
        .unwrap();
    let equality = DistanceLimits {
        min: Some(1.0),
        max: Some(1.0),
    };
    let joint = link(&mut scene, a, b, equality);
    let mut world = create(scene);
    world.solver.substeps = 4;
    for stage in 0..10 {
        let poses = [a, b, rotor].map(|id| world.scene.body_pose(id).unwrap());
        let velocities = [a, b, rotor].map(|id| world.scene.body_velocity(id).unwrap());
        match stage {
            0 => {}
            1 => {
                world.scene.remove_body(removed).unwrap();
            }
            2 => {
                let mut settings = world.scene.joint_settings(joint).unwrap();
                settings.enabled = false;
                world.scene.set_joint_settings(joint, settings).unwrap();
            }
            3 => {
                let mut settings = world.scene.joint_settings(joint).unwrap();
                settings.enabled = true;
                world.scene.set_joint_settings(joint, settings).unwrap();
            }
            4 => {
                let frame = Transform {
                    position: Vec3::X * 0.25,
                    rotation: Quat::from_rotation_y(0.7),
                };
                world.scene.set_joint_frames(joint, frame, frame).unwrap();
            }
            5 => {
                world
                    .scene
                    .set_distance_limits(joint, DistanceLimits::default())
                    .unwrap();
            }
            6 => {
                world.scene.set_distance_limits(joint, equality).unwrap();
            }
            7 => {
                world.scene.set_body_enabled(a, false).unwrap();
            }
            8 => {
                world.scene.set_body_enabled(a, true).unwrap();
            }
            9 => {
                world.scene.remove_joint(joint).unwrap();
            }
            _ => unreachable!(),
        }
        for (i, id) in [a, b, rotor].into_iter().enumerate() {
            let p = world.scene.body_pose(id).unwrap();
            assert!(
                (p.position - poses[i].position).length() < 1e-10,
                "stage={stage}"
            );
            assert!(
                p.rotation
                    .normalize()
                    .dot(poses[i].rotation.normalize())
                    .abs()
                    > 1.0 - 1e-12,
                "stage={stage} rotation before={:?} after={:?}",
                poses[i].rotation,
                p.rotation
            );
            if stage != 7 || id != a {
                let v = world.scene.body_velocity(id).unwrap();
                assert!(
                    (v.linear - velocities[i].linear).length() < 1e-10,
                    "stage={stage}"
                );
                assert!(
                    (v.angular - velocities[i].angular).length() < 1e-10,
                    "stage={stage}"
                );
            }
        }
        assert_eq!(
            world
                .scene
                .joint_axis_settings(motor_joint, JointAxis::AngX)
                .unwrap()
                .motor,
            Some(motor)
        );
        if stage > 0 {
            assert!(world.distance_reading(joint).is_none());
        }
        let before = [a, b].map(|id| world.scene.body_velocity(id).unwrap().linear);
        world.scene.reset_body_forces(b).unwrap();
        world
            .scene
            .add_body_wrench(b, SpatialVector::new(Vec3::X * 10.0, Vec3::ZERO))
            .unwrap();
        world.step(0.01).unwrap();
        let constrained = ![2, 5, 7, 9].contains(&stage);
        if constrained {
            let reading = world.distance_reading(joint).unwrap();
            assert!(reading.violation < tolerance, "stage={stage} {reading:?}");
            let after = world.scene.body_velocity(a).unwrap().linear * 2.0
                + world.scene.body_velocity(b).unwrap().linear * 3.0;
            assert!(
                (after - before[0] * 2.0 - before[1] * 3.0 - Vec3::X * 0.1).length() < tolerance,
                "stage={stage} momentum before={before:?} after={after:?}"
            );
        } else {
            assert!(
                (world.scene.body_velocity(b).unwrap().linear - before[1] - Vec3::X / 30.0)
                    .length()
                    < tolerance,
                "stage={stage} free response"
            );
            if stage == 5 {
                let r = world.distance_reading(joint).unwrap();
                assert_eq!(r.active, DistanceBound::Inactive);
                assert_eq!(r.force, Vec3::ZERO);
            } else {
                assert!(world.distance_reading(joint).is_none());
            }
        }
        assert!(
            (world.scene.body_velocity(rotor).unwrap().angular.z - 0.005 * (stage + 1) as f64)
                .abs()
                < tolerance
        );
        let stamp = world.scene.stamp();
        let reading = world.distance_reading(joint).copied();
        assert!(
            world
                .scene
                .set_distance_limits(
                    joint,
                    DistanceLimits {
                        min: Some(f64::NAN),
                        max: None
                    }
                )
                .is_err()
        );
        assert!(
            world
                .scene
                .set_distance_limits(motor_joint, equality)
                .is_err()
        );
        assert!(
            world
                .scene
                .set_joint_frames(
                    joint,
                    Transform::from_translation(Vec3::splat(f64::NAN)),
                    Transform::IDENTITY
                )
                .is_err()
        );
        if stage != 0 {
            assert!(world.scene.remove_body(removed).is_err());
        }
        assert_eq!(world.scene.stamp(), stamp);
        assert_eq!(world.distance_reading(joint).copied(), reading);
    }
    let replacement = link(&mut world.scene, a, b, equality);
    assert_ne!(joint, replacement);
    assert!(world.scene.set_distance_limits(joint, equality).is_err());
    world.step(0.01).unwrap();
    assert!(world.distance_reading(replacement).unwrap().violation < tolerance);
    let pose = world.scene.body_pose(b).unwrap();
    world.scene.remove_body(a).unwrap();
    assert!(world.scene.joint(replacement).is_none());
    assert!(world.scene.distance_limits(replacement).is_err());
    assert!(world.scene.distance_joints().next().is_none());
    assert!((world.scene.body_pose(b).unwrap().position - pose.position).length() < 1e-10);
    world.step(0.01).unwrap();
    assert!(world.distance_reading(replacement).is_none());
}

#[test]
fn cpu_distance_runtime_mutation_and_unrelated_compaction() {
    mutation(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_distance_runtime_mutation_and_unrelated_compaction() {
    let engine = hardware_engine();
    mutation(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        1e-3,
    );
}

fn scales(mut create: impl FnMut(RigidScene) -> RigidWorld, relative_momentum: f64) {
    let rates = match std::env::var("PRIMITIVE_DISTANCE_HZ") {
        Err(std::env::VarError::NotPresent) => vec![60, 120, 240],
        Ok(value) => match value.parse::<usize>().unwrap() {
            hz @ (60 | 120 | 240) => vec![hz],
            _ => panic!("PRIMITIVE_DISTANCE_HZ must be 60, 120 or 240"),
        },
        Err(error) => panic!("invalid PRIMITIVE_DISTANCE_HZ: {error}"),
    };
    println!("distance scale rates={rates:?} cases_per_rate=40 duration=10s substeps=8");
    for hz in rates {
        let dt = 1.0 / hz as f64;
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let mut cases = Vec::new();
        for length in [0.01, 0.1, 1.0, 10.0] {
            for masses in [
                [0.1, 0.1],
                [1.0, 1.0],
                [100.0, 100.0],
                [1.0, 1000.0],
                [1000.0, 1.0],
            ] {
                for fixed in [false, true] {
                    let a = body(&mut scene, 0.0, masses[0]);
                    let b = body(&mut scene, length, masses[1]);
                    for (i, id) in [a, b].into_iter().enumerate() {
                        scene
                            .set_body_mass_properties(
                                id,
                                MassProperties {
                                    mass: masses[i],
                                    local_com: Vec3::ZERO,
                                    inertia: Mat3::IDENTITY * (masses[i] * length * length),
                                },
                            )
                            .unwrap();
                        scene
                            .set_body_velocity(id, SpatialVector::new(Vec3::ZERO, Vec3::X * 0.2))
                            .unwrap();
                    }
                    if fixed {
                        scene
                            .set_body_kind(a, molla_sim::runtime::BodyKind::Fixed)
                            .unwrap();
                    }
                    let joint = link(
                        &mut scene,
                        a,
                        b,
                        DistanceLimits {
                            min: Some(length),
                            max: Some(length),
                        },
                    );
                    let force = (masses[0] + masses[1]) * length * 0.5;
                    scene
                        .add_body_wrench(b, SpatialVector::new(Vec3::X * force, Vec3::ZERO))
                        .unwrap();
                    cases.push(([a, b], joint, length, masses, force, fixed));
                }
            }
        }
        let mut world = create(scene);
        world.solver.substeps = 8;
        let mut peak_length: f64 = 0.0;
        let mut peak_position: f64 = 0.0;
        let mut peak_reaction: f64 = 0.0;
        let mut peak_momentum: f64 = 0.0;
        for frame in 1..=10 * hz {
            world.step(dt).unwrap();
            assert!(world.quarantined_bodies().is_empty());
            assert!(world.joint_break_events().is_empty());
            assert_eq!(world.completed_steps(), frame as u64);
            let time = frame as f64 * dt;
            for &(bodies, joint, length, masses, force, fixed) in &cases {
                for body in bodies {
                    let pose = world.scene.body_pose(body).unwrap();
                    let velocity = world.scene.body_velocity(body).unwrap();
                    assert!(pose.position.is_finite() && pose.rotation.is_finite());
                    assert!(velocity.linear.is_finite() && velocity.angular.is_finite());
                }
                let r = world.distance_reading(joint).unwrap();
                peak_length = peak_length.max(r.violation / length);
                assert!(
                    r.violation / length < 1e-3,
                    "dt={dt} frame={frame} length={length} masses={masses:?} fixed={fixed} reading={r:?}"
                );
                if fixed {
                    let error = (r.force + Vec3::X * force).length() / force;
                    peak_reaction = peak_reaction.max(error);
                    assert!(
                        error < 0.01,
                        "dt={dt} frame={frame} length={length} masses={masses:?} quasi-static reaction error={error}"
                    );
                } else {
                    let momentum = world.scene.body_velocity(bodies[0]).unwrap().linear * masses[0]
                        + world.scene.body_velocity(bodies[1]).unwrap().linear * masses[1];
                    let expected = Vec3::X * (force * time);
                    let error = (momentum - expected).length() / expected.length().max(1e-3);
                    peak_momentum = peak_momentum.max(error);
                    assert!(
                        error <= relative_momentum,
                        "dt={dt} frame={frame} length={length} masses={masses:?} momentum={momentum:?} expected={expected:?}"
                    );
                }
                if time <= 1.0 {
                    for (i, body) in bodies.into_iter().enumerate() {
                        let pose = world.scene.body_pose(body).unwrap();
                        let travel = if fixed {
                            0.0
                        } else {
                            0.25 * length * time * (time + dt / 8.0)
                        };
                        let expected = Vec3::X * (i as f64 * length + travel);
                        let dp = (pose.position - expected).length();
                        peak_position = peak_position.max(dp);
                        assert!(
                            dp < 1e-3,
                            "dt={dt} frame={frame} length={length} masses={masses:?} position error={dp}"
                        );
                        let angle = if fixed && i == 0 { 0.0 } else { time * 0.2 };
                        let reference = Quat::from_rotation_x(angle);
                        let error = 2.0 * pose.rotation.dot(reference).abs().min(1.0).acos();
                        assert!(
                            error < 1e-3,
                            "dt={dt} frame={frame} length={length} rotation error={error}"
                        );
                    }
                }
            }
            if frame % hz == 0 {
                println!(
                    "distance scale progress: hz={hz} elapsed={time}s relative_length={peak_length:e} relative_momentum={peak_momentum:e} relative_reaction={peak_reaction:e}"
                );
            }
        }
        println!(
            "distance scale sweep: cases={} dt={dt} substeps=8 duration=10s peak relative length={peak_length}, 1s analytic position={peak_position}, quasi-static relative reaction={peak_reaction}, relative momentum={peak_momentum}",
            cases.len()
        );
    }
}

#[test]
fn cpu_distance_length_mass_scales_and_analytic_reactions() {
    scales(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-6);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_distance_length_mass_scales_and_analytic_reactions() {
    let engine = hardware_engine();
    scales(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        1e-3,
    );
}
