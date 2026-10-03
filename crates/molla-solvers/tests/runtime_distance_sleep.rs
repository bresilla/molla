use molla_math::{SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, BodyKind, DistanceLimits, JointDesc, JointHandle, RigidScene,
};
use molla_sim::{BodyParams, JointType};
use molla_solvers::rigid_world::{RigidWorld, SleepSettings};

fn fixture() -> (RigidScene, [[BodyHandle; 2]; 2], [JointHandle; 2]) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut bodies = Vec::new();
    let mut links = Vec::new();
    for x in [0.0, 10.0] {
        let mut pair = Vec::new();
        for offset in [0.0, 1.0] {
            pair.push(
                scene
                    .insert_body(BodyDesc {
                        params: BodyParams {
                            pose: Transform::from_translation(Vec3::X * (x + offset)),
                            ..Default::default()
                        },
                        ..Default::default()
                    })
                    .unwrap(),
            );
        }
        let j = scene
            .insert_joint(JointDesc {
                parent: Some(pair[0]),
                child: pair[1],
                kind: JointType::Distance,
                axis: Vec3::X,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        scene
            .set_distance_limits(
                j,
                DistanceLimits {
                    min: Some(1.0),
                    max: Some(1.0),
                },
            )
            .unwrap();
        bodies.push([pair[0], pair[1]]);
        links.push(j);
    }
    (scene, bodies.try_into().unwrap(), links.try_into().unwrap())
}

fn explicit_and_nonwaking(
    mut world: RigidWorld,
    bodies: [[BodyHandle; 2]; 2],
    links: [JointHandle; 2],
) {
    world.solver.substeps = 4;
    world.sleep_body(bodies[0][0]).unwrap();
    assert!(world.is_sleeping(bodies[0][1]));
    world
        .scene
        .set_body_velocity_with_wake(
            bodies[0][1],
            SpatialVector::new(Vec3::Y * 0.25, Vec3::Z * 0.1),
            false,
        )
        .unwrap();
    let poses = bodies[0].map(|b| world.scene.body_pose(b).unwrap());
    let velocities = bodies[0].map(|b| world.scene.body_velocity(b).unwrap());
    world
        .scene
        .add_body_wrench(bodies[1][1], SpatialVector::new(Vec3::X * 4.0, Vec3::ZERO))
        .unwrap();
    for _ in 0..20 {
        world.step(0.01).unwrap();
        for i in 0..2 {
            assert!(world.is_sleeping(bodies[0][i]));
            assert!(
                (world.scene.body_pose(bodies[0][i]).unwrap().position - poses[i].position)
                    .length()
                    < 1e-8
            );
            let v = world.scene.body_velocity(bodies[0][i]).unwrap();
            assert!((v.linear - velocities[i].linear).length() < 1e-7);
            assert!((v.angular - velocities[i].angular).length() < 1e-7);
        }
        assert!(world.distance_reading(links[0]).is_none());
        assert!(world.distance_reading(links[1]).unwrap().violation < 1e-4);
    }
    assert!(world.scene.body_pose(bodies[1][0]).unwrap().position.x > 10.01);
    world
        .scene
        .set_body_velocity(bodies[0][0], SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    assert!(!world.is_sleeping(bodies[0][1]));
    world.step(0.01).unwrap();
    assert!(world.distance_reading(links[0]).is_some());
}

fn automatic_and_boundary(
    mut world: RigidWorld,
    bodies: [[BodyHandle; 2]; 2],
    links: [JointHandle; 2],
) {
    world
        .set_sleep_settings(Some(SleepSettings {
            quiet_time: 0.03,
            ..Default::default()
        }))
        .unwrap();
    world
        .scene
        .set_body_kind(bodies[0][0], BodyKind::Fixed)
        .unwrap();
    for _ in 0..6 {
        world.step(0.01).unwrap();
    }
    assert!(world.is_sleeping(bodies[0][1]));
    assert!(world.is_sleeping(bodies[1][0]));
    world
        .scene
        .set_body_pose(bodies[0][0], Transform::from_translation(Vec3::X * 0.2))
        .unwrap();
    assert!(!world.is_sleeping(bodies[0][1]));
    assert!(world.is_sleeping(bodies[1][0]));
    world.step(0.01).unwrap();
    assert!(world.distance_reading(links[0]).unwrap().violation < 1e-4);
    assert!(
        (world.scene.body_pose(bodies[0][0]).unwrap().position - Vec3::X * 0.2).length() < 1e-7
    );
    assert!(world.is_sleeping(bodies[1][0]));
    world.sleep_body(bodies[0][1]).unwrap();
    assert!(world.is_sleeping(bodies[0][1]));
    world
        .scene
        .set_distance_limits(
            links[1],
            DistanceLimits {
                min: Some(0.8),
                max: Some(0.8),
            },
        )
        .unwrap();
    assert!(!world.is_sleeping(bodies[1][0]));
    assert!(world.is_sleeping(bodies[0][1]));
    world.step(0.01).unwrap();
    assert!(world.distance_reading(links[1]).unwrap().violation < 1e-4);
    assert!(world.is_sleeping(bodies[0][1]));
}

#[test]
fn cpu_distance_sleep_keeps_other_islands_active_and_preserves_nonwaking_edits() {
    let (scene, bodies, links) = fixture();
    explicit_and_nonwaking(RigidWorld::from_scene(scene).unwrap(), bodies, links);
}

#[test]
fn cpu_distance_automatic_sleep_and_fixed_boundary_wake() {
    let (scene, bodies, links) = fixture();
    automatic_and_boundary(RigidWorld::from_scene(scene).unwrap(), bodies, links);
}

#[cfg(feature = "gpu")]
fn gpu(scene: RigidScene) -> RigidWorld {
    let engine =
        std::sync::Arc::new(molla_compute::Engine::new(molla_compute::Backend::AutoGpu).unwrap());
    RigidWorld::from_scene_gpu(scene, engine).unwrap()
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_distance_sleep_keeps_other_islands_active_and_preserves_nonwaking_edits() {
    let (scene, bodies, links) = fixture();
    explicit_and_nonwaking(gpu(scene), bodies, links);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_distance_automatic_sleep_and_fixed_boundary_wake() {
    let (scene, bodies, links) = fixture();
    automatic_and_boundary(gpu(scene), bodies, links);
}

fn moving_scene() -> (RigidScene, [[BodyHandle; 2]; 2], [JointHandle; 2]) {
    let (mut scene, bodies, links) = fixture();
    scene
        .set_body_kind(bodies[0][0], BodyKind::Kinematic)
        .unwrap();
    scene
        .set_body_velocity(
            bodies[0][0],
            SpatialVector::new(Vec3::X * 0.2, Vec3::Z * 0.15),
        )
        .unwrap();
    scene
        .set_joint_frames(
            links[0],
            Transform::from_translation(Vec3::Y * 0.25),
            Transform::IDENTITY,
        )
        .unwrap();
    let length = (Vec3::X - Vec3::Y * 0.25).length();
    scene
        .set_distance_limits(
            links[0],
            DistanceLimits {
                min: Some(length),
                max: Some(length),
            },
        )
        .unwrap();
    (scene, bodies, links)
}

fn prescribe(world: &mut RigidWorld, body: BodyHandle, time: f64) -> Transform {
    let pose = Transform {
        position: Vec3::X * (0.2 * time),
        rotation: molla_math::Quat::from_rotation_z(0.15 * time),
    };
    world.scene.set_body_pose(body, pose).unwrap();
    world.scene.body_pose(body).unwrap()
}

#[test]
fn cpu_distance_kinematic_boundary_preserves_prescribed_state_and_prevents_sleep() {
    let (scene, bodies, links) = moving_scene();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.solver.substeps = 8;
    world.sleep_body(bodies[0][1]).unwrap();
    assert!(!world.is_sleeping(bodies[0][1]));
    world.sleep_body(bodies[1][0]).unwrap();
    for frame in 0..120 {
        let pose = prescribe(&mut world, bodies[0][0], frame as f64 / 120.0);
        world.step(1.0 / 120.0).unwrap();
        assert_eq!(world.scene.body_pose(bodies[0][0]).unwrap(), pose);
        assert!(world.is_sleeping(bodies[1][0]));
        assert!(world.distance_reading(links[0]).unwrap().violation < 1e-3);
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_distance_kinematic_boundary_matches_cpu_without_model_recompilation() {
    for force in [0.0, 8.0] {
        let (scene, bodies, links) = moving_scene();
        let mut world = gpu(scene);
        let (scene, cb, cl) = moving_scene();
        let mut cpu = RigidWorld::from_scene(scene).unwrap();
        world.solver.substeps = 8;
        cpu.solver.substeps = 8;
        world.sleep_body(bodies[0][1]).unwrap();
        assert!(!world.is_sleeping(bodies[0][1]));
        world.sleep_body(bodies[1][0]).unwrap();
        cpu.sleep_body(cb[1][0]).unwrap();
        let applied = Vec3::X * force;
        world
            .scene
            .add_body_wrench(bodies[0][1], SpatialVector::new(applied, Vec3::ZERO))
            .unwrap();
        cpu.scene
            .add_body_wrench(cb[0][1], SpatialVector::new(applied, Vec3::ZERO))
            .unwrap();
        let mut peak: f64 = 0.0;
        let mut force_peak: f64 = 0.0;
        let mut cpu_impulse = Vec3::ZERO;
        let mut gpu_impulse = Vec3::ZERO;
        for frame in 0..120 {
            let pose = prescribe(&mut world, bodies[0][0], frame as f64 / 120.0);
            prescribe(&mut cpu, cb[0][0], frame as f64 / 120.0);
            let before = world.scene.body_velocity(bodies[0][1]).unwrap().linear;
            cpu.step(1.0 / 120.0).unwrap();
            world.step(1.0 / 120.0).unwrap();
            let actual = world.scene.body_pose(bodies[0][0]).unwrap();
            assert!((actual.position - pose.position).length() < 1e-7);
            assert!(actual.rotation.dot(pose.rotation).abs().min(1.0).acos() * 2.0 < 1e-3);
            let velocity = world.scene.body_velocity(bodies[0][0]).unwrap();
            assert!((velocity.linear - Vec3::X * 0.2).length() < 1e-7);
            assert!((velocity.angular - Vec3::Z * 0.15).length() < 1e-7);
            let reading = world.distance_reading(links[0]).unwrap();
            assert!(reading.violation < 1e-3);
            let expected = cpu.distance_reading(cl[0]).unwrap();
            force_peak = force_peak.max((reading.force - expected.force).length());
            if force != 0.0 {
                assert!(
                    (reading.force - expected.force).length()
                        < 0.01 * expected.force.length().max(1.0),
                    "frame={frame} CPU={expected:?} GPU={reading:?}"
                );
            }
            let momentum_change = world.scene.body_velocity(bodies[0][1]).unwrap().linear - before;
            let mass = world.scene.body_mass_properties(bodies[0][1]).unwrap().mass;
            assert!((momentum_change * mass - (reading.force + applied) / 120.0).length() < 1e-5);
            cpu_impulse += expected.force / 120.0;
            gpu_impulse += reading.force / 120.0;
            peak = peak.max(
                (world.scene.body_pose(bodies[0][1]).unwrap().position
                    - cpu.scene.body_pose(cb[0][1]).unwrap().position)
                    .length(),
            );
            assert!(world.is_sleeping(bodies[1][0]));
            assert_eq!(world.gpu_diagnostics().unwrap().model_compilations, 1);
        }
        println!(
            "kinematic distance force={force}N one-second CPU/GPU position peak={peak}, force error peak={force_peak}, impulse error={}",
            (cpu_impulse - gpu_impulse).length()
        );
        assert!(peak < 1e-3);
        assert!((cpu_impulse - gpu_impulse).length() < 1e-3);
    }
}
