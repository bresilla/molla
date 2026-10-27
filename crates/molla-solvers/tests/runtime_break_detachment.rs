use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::{
    JointType, MassProperties,
    runtime::{
        BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, DistanceLimits, JointDesc, RigidScene,
    },
};
use molla_solvers::rigid_world::RigidWorld;

fn body(scene: &mut RigidScene) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap()
}
fn desc(parent: Option<BodyHandle>, child: BodyHandle, kind: JointType) -> JointDesc {
    JointDesc {
        parent,
        child,
        kind,
        axis: Vec3::Z,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
    }
}

fn detached_motion(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    for (kind, external) in [
        (JointType::Fixed, false),
        (JointType::Revolute, false),
        (JointType::Prismatic, false),
        (JointType::Ball, false),
        (JointType::D6, false),
        (JointType::Free, false),
        (JointType::Distance, true),
        (JointType::Ball, true),
    ] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let b = body(&mut scene);
        let joint = desc(None, b, kind);
        let id = if external && kind == JointType::Ball {
            scene.insert_rigid_loop_joint(joint).unwrap()
        } else {
            scene.insert_joint(joint).unwrap()
        };
        if kind == JointType::Distance {
            scene
                .set_distance_limits(
                    id,
                    DistanceLimits {
                        min: None,
                        max: Some(0.0),
                    },
                )
                .unwrap();
        }
        let mut world = make(scene);
        world.set_reaction_readout(true).unwrap();
        world.step(0.01).unwrap();
        let pose = world.scene.body_pose(b).unwrap();
        let velocity = world.scene.body_velocity(b).unwrap();
        world.scene.break_joints(&[id]).unwrap();
        assert_eq!(world.scene.body_pose(b).unwrap(), pose);
        assert_eq!(world.scene.body_velocity(b).unwrap(), velocity);
        assert!(world.joint_reaction_samples().is_none());
        world
            .scene
            .add_body_wrench(b, SpatialVector::new(Vec3::X * 4.0, Vec3::ZERO))
            .unwrap();
        world.step(0.1).unwrap();
        assert!(
            (world.scene.body_velocity(b).unwrap().linear.x - 0.2).abs() < tolerance,
            "{kind:?} external={external}"
        );
        assert!((world.scene.body_pose(b).unwrap().position.x - 0.01125).abs() < tolerance);
        assert!(world.joint_reaction(id).is_none());
        assert!(world.scene.joint_broken(id).unwrap());
    }
}

#[test]
fn cpu_detached_tree_and_loop_joints_leave_no_hidden_constraint() {
    detached_motion(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-10);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_detached_tree_and_loop_joints_leave_no_hidden_constraint() {
    let engine =
        std::sync::Arc::new(molla_compute::Engine::new(molla_compute::Backend::AutoGpu).unwrap());
    detached_motion(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        1e-6,
    );
}

fn loop_break(make: impl Fn(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let a = body(&mut scene);
    let b = body(&mut scene);
    let unrelated = body(&mut scene);
    let tree = scene
        .insert_joint(desc(Some(a), b, JointType::Fixed))
        .unwrap();
    let link = scene
        .insert_rigid_loop_joint(desc(None, b, JointType::Ball))
        .unwrap();
    let mut world = make(scene);
    world.set_reaction_readout(true).unwrap();
    world.sleep_body(unrelated).unwrap();
    world.step(0.01).unwrap();
    let stamp = world.scene.activation_stamp(unrelated).unwrap();
    world.scene.break_joints(&[link]).unwrap();
    assert_eq!(world.scene.activation_stamp(unrelated).unwrap(), stamp);
    assert!(world.scene.joint_index(tree).is_some());
    assert!(!world.scene.joint_broken(tree).unwrap());
    world
        .scene
        .add_body_wrench(b, SpatialVector::new(Vec3::X * 8.0, Vec3::ZERO))
        .unwrap();
    world.step(0.1).unwrap();
    assert!(world.is_sleeping(unrelated));
    for id in [a, b] {
        assert!((world.scene.body_velocity(id).unwrap().linear.x - 0.2).abs() < tolerance);
    }
    assert!((world.joint_reaction(tree).unwrap().child.linear.x + 4.0).abs() < tolerance);
    assert!(world.joint_reaction(link).is_none());
    world.scene.remove_body(unrelated).unwrap();
    world.step(0.1).unwrap();
    assert!(world.scene.joint_broken(link).unwrap());
}

#[test]
fn cpu_loop_break_keeps_tree_load_sharing_and_unrelated_sleep() {
    loop_break(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-10);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_loop_break_keeps_tree_load_sharing_and_unrelated_sleep() {
    let engine =
        std::sync::Arc::new(molla_compute::Engine::new(molla_compute::Backend::AutoGpu).unwrap());
    loop_break(
        |scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap(),
        1e-5,
    );
}

fn compliant_loop_repair(make: impl FnOnce(RigidScene) -> RigidWorld, tolerance: f64) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let b = body(&mut scene);
    let link = scene
        .insert_loop_joint(
            desc(None, b, JointType::Fixed),
            molla_sim::joint_control::JointSoftness {
                natural_frequency: 20.0,
                damping_ratio: 1.0,
            },
        )
        .unwrap();
    scene.break_joints(&[link]).unwrap();
    scene
        .add_body_wrench(b, SpatialVector::new(Vec3::X * 4.0, Vec3::ZERO))
        .unwrap();
    let mut world = make(scene);
    world.solver.substeps = 4;
    world.set_reaction_readout(true).unwrap();
    let dt = 0.01;
    let h = dt / 4.0;
    let mass = 2.0;
    let omega = std::f64::consts::TAU * 20.0;
    let stiffness = mass * omega * omega;
    let damping = 2.0 * mass * omega;
    let mut position = 0.0;
    let mut velocity = 0.0;
    let mut peak_position: f64 = 0.0;
    let mut peak_velocity: f64 = 0.0;
    let mut peak_reaction: f64 = 0.0;
    for (phase, enabled) in [false, true, false, true].into_iter().enumerate() {
        let pose = world.scene.body_pose(b).unwrap();
        let rate = world.scene.body_velocity(b).unwrap();
        if enabled {
            world.scene.repair_joint(link).unwrap();
        } else {
            world.scene.break_joints(&[link]).unwrap();
        }
        assert_eq!(world.scene.body_pose(b).unwrap(), pose);
        assert_eq!(world.scene.body_velocity(b).unwrap(), rate);
        assert_eq!(world.scene.joint_broken(link).unwrap(), !enabled);
        assert!(world.joint_reaction(link).is_none());
        for frame in 0..16 {
            let mut reaction = 0.0;
            for _ in 0..4 {
                let impulse = if enabled {
                    (-velocity - stiffness * position / (damping + h * stiffness))
                        / (1.0 / mass + 1.0 / (h * (damping + h * stiffness)))
                } else {
                    0.0
                };
                velocity += impulse / mass + h * 4.0 / mass;
                position += h * velocity;
                reaction += impulse / dt;
            }
            world.step(dt).unwrap();
            let pose = world.scene.body_pose(b).unwrap();
            let rate = world.scene.body_velocity(b).unwrap();
            let position_error = (pose.position - Vec3::X * position).length();
            let velocity_error = (rate.linear - Vec3::X * velocity).length();
            peak_position = peak_position.max(position_error);
            peak_velocity = peak_velocity.max(velocity_error);
            assert!(
                position_error < tolerance,
                "phase={phase} frame={frame} position={position_error:e}"
            );
            assert!(
                velocity_error < tolerance,
                "phase={phase} frame={frame} velocity={velocity_error:e}"
            );
            assert!(pose.position.is_finite() && pose.rotation.is_finite());
            assert!(rate.linear.is_finite() && rate.angular.is_finite());
            assert!(pose.rotation.to_scaled_axis().length() < tolerance);
            assert!(rate.angular.length() < tolerance);
            assert_eq!(world.scene.joint_broken(link).unwrap(), !enabled);
            assert!(world.joint_break_events().is_empty());
            assert!(world.quarantined_bodies().is_empty());
            assert_eq!(world.completed_steps(), (phase * 16 + frame + 1) as u64);
            if enabled {
                let sample = world.joint_reaction(link).unwrap();
                let error = (sample.child.linear - Vec3::X * reaction).length();
                peak_reaction = peak_reaction.max(error);
                assert!(
                    error < tolerance * (1.0 + reaction.abs()),
                    "phase={phase} frame={frame} reaction={error:e}"
                );
                assert!((sample.parent.linear + sample.child.linear).length() < tolerance);
                assert!(sample.child.angular.length() < tolerance);
                assert!(sample.parent.angular.length() < tolerance);
                assert!((sample.sample_dt - dt).abs() < 1e-12);
            } else {
                assert!(world.joint_reaction(link).is_none());
            }
        }
    }
    println!(
        "compliant break/repair phases=4 frames=64 substeps=4 position_error={peak_position:e} velocity_error={peak_velocity:e} reaction_error={peak_reaction:e}"
    );
}

#[test]
fn cpu_broken_compliant_loop_is_free_until_explicit_repair_restores_response() {
    compliant_loop_repair(|scene| RigidWorld::from_scene(scene).unwrap(), 1e-10);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_broken_compliant_loop_is_free_until_explicit_repair_restores_response() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    compliant_loop_repair(
        |scene| RigidWorld::from_scene_gpu(scene, engine).unwrap(),
        2e-5,
    );
}

#[test]
fn cpu_explicit_pair_filters_survive_break_and_control_actual_contact_response() {
    for policy in 0..4 {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let a = body(&mut scene);
        let b = body(&mut scene);
        let collider = |parent| ColliderDesc {
            parent: Some(parent),
            ..ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 })
        };
        let ca = scene.insert_collider(collider(a)).unwrap();
        let cb = scene.insert_collider(collider(b)).unwrap();
        let id = scene
            .insert_joint(desc(Some(a), b, JointType::Fixed))
            .unwrap();
        if policy == 1 {
            scene.set_body_pair_excluded(a, b, true).unwrap();
        }
        if policy == 2 {
            scene.set_collider_pair_excluded(ca, cb, true).unwrap();
        }
        let mut world = RigidWorld::from_scene(scene).unwrap();
        world.step(0.0001).unwrap();
        assert!(world.contact_pairs().is_empty());
        world.scene.break_joints(&[id]).unwrap();
        world.step_filtered(0.0001, |_, _| policy == 3).unwrap();
        if policy == 0 {
            assert!(!world.contact_pairs().is_empty());
            assert!(world.scene.body_velocity(a).unwrap().linear.length() > 1e-5);
        } else {
            assert!(world.contact_pairs().is_empty());
            assert_eq!(world.scene.body_velocity(a).unwrap(), SpatialVector::ZERO);
            assert_eq!(world.scene.body_velocity(b).unwrap(), SpatialVector::ZERO);
        }
    }
}

#[cfg(feature = "gpu")]
fn threshold_capability(make: impl Fn(RigidScene) -> RigidWorld) {
    let mut scene = RigidScene::new().unwrap();
    let b = body(&mut scene);
    let id = scene.insert_joint(desc(None, b, JointType::Fixed)).unwrap();
    let mut world = make(scene);
    world.set_reaction_readout(true).unwrap();
    world.step(0.01).unwrap();
    world
        .scene
        .set_joint_break_limits(
            id,
            molla_sim::runtime::JointBreakLimits {
                force: Some(1e-100),
                torque: None,
            },
        )
        .unwrap();
    let pose = world.scene.body_pose(b).unwrap();
    let velocity = world.scene.body_velocity(b).unwrap();
    let error = world.step(0.01).unwrap_err();
    assert!(
        error
            .to_string()
            .contains("GPU joint break threshold is outside normal f32 range"),
        "{error}"
    );
    assert_eq!(world.scene.body_pose(b).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(b).unwrap(), velocity);
    assert!(world.joint_reaction_samples().is_none());
    assert!(!world.scene.joint_broken(id).unwrap());
    world
        .scene
        .set_joint_break_limits(
            id,
            molla_sim::runtime::JointBreakLimits {
                force: Some(f64::INFINITY),
                torque: None,
            },
        )
        .unwrap();
    world.step(0.01).unwrap();
    world.scene.break_joints(&[id]).unwrap();
    world.step(0.01).unwrap();
    assert!(world.scene.joint_broken(id).unwrap());
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_unrepresentable_threshold_is_rejected_without_advancing() {
    let engine =
        std::sync::Arc::new(molla_compute::Engine::new(molla_compute::Backend::AutoGpu).unwrap());
    threshold_capability(|scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap());
}
