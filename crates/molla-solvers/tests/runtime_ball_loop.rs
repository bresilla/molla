use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene, SphericalConeLimits};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn loaded_anchor(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    let frame = Quat::from_rotation_z(0.7) * Quat::from_rotation_x(-0.4);
    let local_rotation = Quat::from_rotation_y(0.5);
    let rotation = frame * local_rotation.conjugate();
    let anchor = Vec3::new(1.5, -0.7, 0.3);
    let local_anchor = Vec3::new(0.3, -0.1, 0.2);
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(frame * (-Vec3::Y * 9.81)).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams {
                pose: Transform {
                    position: anchor - rotation * local_anchor,
                    rotation,
                },
                ..Default::default()
            },
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: local_anchor - local_rotation * Vec3::Y,
                inertia: Mat3::from_cols(
                    Vec3::new(0.8, 0.1, 0.0),
                    Vec3::new(0.1, 1.0, 0.05),
                    Vec3::new(0.0, 0.05, 0.9),
                ),
            }),
            ..Default::default()
        })
        .unwrap();
    let joint = scene
        .insert_rigid_loop_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::Ball,
            axis: Vec3::Y,
            frame_parent: Transform {
                position: anchor,
                rotation: frame,
            },
            frame_child: Transform {
                position: local_anchor,
                rotation: local_rotation,
            },
        })
        .unwrap();
    scene
        .set_cone_limits(
            joint,
            SphericalConeLimits {
                angle0: Some(0.0),
                angle1: Some(0.0),
            },
        )
        .unwrap();
    let torque = frame * Vec3::Z * 3.0;
    scene
        .add_body_wrench(body, SpatialVector::new(Vec3::ZERO, torque))
        .unwrap();
    let mut world = make(scene);
    let mut peak = 0.0_f64;
    for _ in 0..120 {
        world.step(1.0 / 120.0).unwrap();
        let reading = world.ball_anchor_reading(joint).unwrap();
        peak = peak.max(reading.separation);
        assert!(reading.separation < tolerance, "{reading:?}");
        assert!(world.distance_reading(joint).is_none());
    }
    let anchor = world.ball_anchor_reading(joint).unwrap();
    let cone = world.cone_reading(joint).unwrap();
    let force_error = (anchor.force - frame * Vec3::Y * 19.62).length();
    let torque_error = (cone.torque + torque).length();
    assert!(force_error < 0.1962, "{anchor:?}");
    assert!(torque_error < 0.03, "{cone:?}");
    assert!(cone.violation < tolerance);
    eprintln!(
        "rigid ball loaded: peak_anchor={peak:e}, force_error={force_error:e}, torque_error={torque_error:e}"
    );
    world.scene.remove_joint(joint).unwrap();
    assert!(world.ball_anchor_reading(joint).is_none());
    world.step(1.0 / 120.0).unwrap();
    assert!(world.scene.body_velocity(body).unwrap().linear.length() > 0.05);
}

fn pendulum(make: impl Fn(RigidScene) -> RigidWorld) {
    for dt in [1.0 / 60.0, 1.0 / 120.0, 1.0 / 240.0] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(-Vec3::Y * 9.81).unwrap();
        let body = scene
            .insert_body(BodyDesc {
                params: BodyParams {
                    pose: Transform::from_rotation(Quat::from_rotation_z(0.7)),
                    ..Default::default()
                },
                additional_mass: Some(MassProperties {
                    mass: 1.0,
                    local_com: -Vec3::Y,
                    inertia: Mat3::IDENTITY * 0.2,
                }),
                ..Default::default()
            })
            .unwrap();
        let joint = scene
            .insert_rigid_loop_joint(JointDesc {
                parent: None,
                child: body,
                kind: JointType::Ball,
                axis: Vec3::Y,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        scene
            .set_cone_limits(
                joint,
                SphericalConeLimits {
                    angle0: Some(0.8),
                    angle1: Some(0.8),
                },
            )
            .unwrap();
        let center = scene.body_pose(body).unwrap().rotation * (-Vec3::Y);
        let omega = Vec3::Z * 3.0;
        scene
            .set_body_velocity(body, SpatialVector::new(omega.cross(center), omega))
            .unwrap();
        let initial_energy = 0.5 * omega.cross(center).length_squared()
            + 0.1 * omega.length_squared()
            + 9.81 * (center.y + 1.0);
        let mut world = make(scene);
        let mut peak_anchor = 0.0_f64;
        let mut peak_cone = 0.0_f64;
        let mut peak_energy = initial_energy;
        let mut cone_active = false;
        for _ in 0..(10.0 / dt) as usize {
            world.step(dt).unwrap();
            peak_anchor = peak_anchor.max(world.ball_anchor_reading(joint).unwrap().separation);
            let cone = world.cone_reading(joint).unwrap();
            peak_cone = peak_cone.max(cone.violation);
            cone_active |= cone.active;
            let pose = world.scene.body_pose(body).unwrap();
            let center = pose.position - pose.rotation * Vec3::Y;
            let velocity = world.scene.body_velocity(body).unwrap();
            let energy = 0.5 * velocity.linear.length_squared()
                + 0.1 * velocity.angular.length_squared()
                + 9.81 * (center.y + 1.0);
            peak_energy = peak_energy.max(energy);
            assert!(peak_anchor < 1e-3, "dt={dt} anchor={peak_anchor}");
            assert!(peak_cone < 1e-3, "dt={dt} cone={peak_cone}");
            assert!(
                energy < initial_energy + 0.02,
                "dt={dt} energy={energy} initial={initial_energy}"
            );
        }
        assert!(cone_active);
        eprintln!(
            "rigid ball pendulum: dt={dt} duration=10s anchor={peak_anchor:e} cone={peak_cone:e} initial_energy={initial_energy:e} peak_energy={peak_energy:e}"
        );
    }
}

fn floating_pair(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let unrelated = scene.insert_body(BodyDesc::default()).unwrap();
    let mut bodies = Vec::new();
    for mass in [2.0, 3.0] {
        bodies.push(
            scene
                .insert_body(BodyDesc {
                    additional_mass: Some(MassProperties {
                        mass,
                        local_com: Vec3::ZERO,
                        inertia: Mat3::IDENTITY * mass,
                    }),
                    ..Default::default()
                })
                .unwrap(),
        );
    }
    let joint = scene
        .insert_rigid_loop_joint(JointDesc {
            parent: Some(bodies[0]),
            child: bodies[1],
            kind: JointType::Ball,
            axis: Vec3::Y,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_cone_limits(
            joint,
            SphericalConeLimits {
                angle0: Some(0.0),
                angle1: Some(0.0),
            },
        )
        .unwrap();
    scene
        .set_body_velocity(
            bodies[0],
            SpatialVector::new(Vec3::X * 3.0, Vec3::Y * 2.0 + Vec3::Z * 3.0),
        )
        .unwrap();
    scene
        .set_body_velocity(
            bodies[1],
            SpatialVector::new(-Vec3::X * 2.0, Vec3::Y - Vec3::Z * 2.0),
        )
        .unwrap();
    let mut world = make(scene);
    for _ in 0..120 {
        world.step(1.0 / 120.0).unwrap();
        let a = world.scene.body_velocity(bodies[0]).unwrap();
        let b = world.scene.body_velocity(bodies[1]).unwrap();
        assert!((a.linear * 2.0 + b.linear * 3.0).length() < tolerance);
        assert!((a.angular * 2.0 + b.angular * 3.0 - Vec3::Y * 7.0).length() < tolerance);
        assert!(a.linear.length() < tolerance && b.linear.length() < tolerance);
        assert!((a.angular.y - 2.0).abs() < tolerance);
        assert!((b.angular.y - 1.0).abs() < tolerance);
        assert!(world.ball_anchor_reading(joint).unwrap().separation < tolerance);
    }
    world.scene.remove_body(unrelated).unwrap();
    assert!(world.ball_anchor_reading(joint).is_none());
    world.step(1.0 / 120.0).unwrap();
    assert!(world.ball_anchor_reading(joint).unwrap().separation < tolerance);
    world.sleep_body(bodies[0]).unwrap();
    assert!(world.is_sleeping(bodies[1]));
    world.step(1.0 / 120.0).unwrap();
    assert!(world.ball_anchor_reading(joint).is_none());
    world
        .scene
        .add_body_wrench(bodies[1], SpatialVector::new(Vec3::X * 2.0, Vec3::ZERO))
        .unwrap();
    assert!(!world.is_sleeping(bodies[0]));
    world.step(1.0 / 120.0).unwrap();
    let a = world.scene.body_velocity(bodies[0]).unwrap();
    let b = world.scene.body_velocity(bodies[1]).unwrap();
    assert!((a.linear * 2.0 + b.linear * 3.0 - Vec3::X / 60.0).length() < tolerance);
    assert!(world.ball_anchor_reading(joint).unwrap().separation < tolerance);
}

fn closed_chain(make: impl Fn(RigidScene) -> RigidWorld) -> Vec<[Transform; 2]> {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(-Vec3::Y * 9.81).unwrap();
    let parent = scene.insert_body(BodyDesc::default()).unwrap();
    let child = scene
        .insert_body(BodyDesc {
            params: BodyParams {
                pose: Transform::from_translation(Vec3::X),
                ..Default::default()
            },
            ..Default::default()
        })
        .unwrap();
    scene
        .insert_joint(JointDesc {
            parent: None,
            child: parent,
            kind: JointType::Ball,
            axis: Vec3::Y,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .insert_joint(JointDesc {
            parent: Some(parent),
            child,
            kind: JointType::Ball,
            axis: Vec3::Y,
            frame_parent: Transform::from_translation(Vec3::X),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let joint = scene
        .insert_rigid_loop_joint(JointDesc {
            parent: None,
            child,
            kind: JointType::Ball,
            axis: Vec3::Y,
            frame_parent: Transform::from_translation(Vec3::X + Vec3::Y),
            frame_child: Transform::from_translation(Vec3::Y),
        })
        .unwrap();
    scene
        .set_cone_limits(
            joint,
            SphericalConeLimits {
                angle0: Some(0.3),
                angle1: Some(0.6),
            },
        )
        .unwrap();
    scene
        .add_body_wrench(child, SpatialVector::new(Vec3::Z * 5.0, Vec3::X * 2.0))
        .unwrap();
    assert_eq!(scene.model().articulation_count, 1);
    let mut world = make(scene);
    let mut poses = Vec::new();
    let mut cone_active = false;
    for _ in 0..120 {
        world.step(1.0 / 120.0).unwrap();
        let a = world.scene.body_pose(parent).unwrap();
        let b = world.scene.body_pose(child).unwrap();
        assert!(a.position.length() < 1e-3);
        assert!((a.position + a.rotation * Vec3::X - b.position).length() < 1e-3);
        assert!(world.ball_anchor_reading(joint).unwrap().separation < 1e-3);
        let cone = world.cone_reading(joint).unwrap();
        assert!(cone.violation < 1e-3);
        cone_active |= cone.active;
        poses.push([a, b]);
    }
    assert!(cone_active);
    assert!((poses.last().unwrap()[1].position - Vec3::X).length() > 0.01);
    poses
}

#[test]
fn cpu_rigid_ball_loop_transmits_loaded_anchor_and_cone_reactions() {
    loaded_anchor(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[test]
fn cpu_rigid_ball_loop_pendulum_ten_seconds_three_step_sizes() {
    pendulum(|scene| RigidWorld::from_scene(scene).unwrap());
}

#[test]
fn cpu_rigid_ball_loop_shares_two_articulation_impulses_and_preserves_twist() {
    floating_pair(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-8);
}

#[test]
fn cpu_rigid_ball_loop_closes_an_existing_articulation() {
    closed_chain(|scene| RigidWorld::from_scene(scene).unwrap());
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_rigid_ball_loop_transmits_loaded_anchor_and_cone_reactions() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    loaded_anchor(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        2e-5,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_rigid_ball_loop_pendulum_ten_seconds_three_step_sizes() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    pendulum(|scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap());
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_rigid_ball_loop_shares_two_articulation_impulses_and_preserves_twist() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    floating_pair(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        1e-4,
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_rigid_ball_loop_closed_chain_matches_cpu_trajectory() {
    let cpu = closed_chain(|scene| RigidWorld::from_scene(scene).unwrap());
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let gpu = closed_chain(|scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap());
    let mut peak_position = 0.0_f64;
    let mut peak_rotation = 0.0_f64;
    for (cpu, gpu) in cpu.into_iter().zip(gpu) {
        for (a, b) in cpu.into_iter().zip(gpu) {
            peak_position = peak_position.max((a.position - b.position).length());
            peak_rotation = peak_rotation.max(a.rotation.angle_between(b.rotation).abs());
        }
    }
    assert!(peak_position < 1e-3, "position={peak_position}");
    assert!(peak_rotation < 1e-3, "rotation={peak_rotation}");
    eprintln!(
        "rigid ball closed-chain CPU/GPU: position={peak_position:e} rotation={peak_rotation:e}"
    );
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_rigid_ball_loop_retains_resources_and_rebuilds_only_on_mutation() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let joint = scene
        .insert_rigid_loop_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::Ball,
            axis: Vec3::Y,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_cone_limits(
            joint,
            SphericalConeLimits {
                angle0: Some(0.0),
                angle1: Some(0.0),
            },
        )
        .unwrap();
    let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
    let resources = || {
        let device = engine.device().unwrap();
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let counters = device.get_internal_counters().hal;
        [
            counters.buffers.read(),
            counters.buffer_memory.read(),
            counters.bind_groups.read(),
            counters.compute_pipelines.read(),
            counters.shader_modules.read(),
        ]
    };
    for _ in 0..4 {
        world.step(0.01).unwrap();
    }
    let warm = resources();
    assert!(warm.iter().all(|&count| count > 0));
    let groups = world
        .gpu_diagnostics()
        .unwrap()
        .current_model_bind_groups_created;
    for _ in 0..20 {
        world
            .scene
            .set_body_velocity(body, SpatialVector::new(Vec3::X, Vec3::Y))
            .unwrap();
        world.step(0.01).unwrap();
        assert_eq!(resources(), warm);
        assert_eq!(world.gpu_diagnostics().unwrap().model_compilations, 1);
        assert_eq!(
            world
                .gpu_diagnostics()
                .unwrap()
                .current_model_bind_groups_created,
            groups
        );
    }
    world
        .scene
        .set_cone_limits(joint, SphericalConeLimits::default())
        .unwrap();
    world.step(0.01).unwrap();
    assert_eq!(world.gpu_diagnostics().unwrap().model_compilations, 2);
    assert!(world.ball_anchor_reading(joint).is_some());
    assert!(world.cone_reading(joint).is_none());
    eprintln!("rigid ball warm [buffers, bytes, groups, pipelines, shaders]={warm:?}");
}
