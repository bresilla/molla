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

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU through make test-gpu"]
fn gpu_ignores_broken_compliant_loop_metadata_but_rejects_explicit_repair() {
    let engine =
        std::sync::Arc::new(molla_compute::Engine::new(molla_compute::Backend::AutoGpu).unwrap());
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
    let mut world = RigidWorld::from_scene_gpu(scene, engine).unwrap();
    world.step(0.01).unwrap();
    world.scene.repair_joint(link).unwrap();
    let pose = world.scene.body_pose(b).unwrap();
    assert!(world.step(0.01).is_err());
    assert_eq!(world.scene.body_pose(b).unwrap(), pose);
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
