use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, BodyKind, ColliderDesc, ColliderGeometry, ColliderHandle, RigidScene,
};
use molla_sim::{BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, MassProperties};
use molla_solvers::rigid_world::{RigidWorld, SleepSettings};

#[cfg(feature = "gpu")]
fn engine() -> std::sync::Arc<molla_compute::Engine> {
    static ENGINE: std::sync::OnceLock<std::sync::Arc<molla_compute::Engine>> =
        std::sync::OnceLock::new();
    ENGINE
        .get_or_init(|| {
            let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
            let adapter = engine.adapter_info().unwrap();
            assert!(
                !adapter.is_software,
                "contact sleep requires a hardware GPU"
            );
            println!("adapter={adapter:?}");
            engine
        })
        .clone()
}

fn world(scene: RigidScene, gpu: bool) -> RigidWorld {
    #[cfg(feature = "gpu")]
    if gpu {
        return RigidWorld::from_scene_gpu(scene, engine()).unwrap();
    }
    assert!(!gpu);
    RigidWorld::from_scene(scene).unwrap()
}

fn body(scene: &mut RigidScene, position: Vec3, kind: BodyKind) -> BodyHandle {
    scene
        .insert_body(BodyDesc {
            kind,
            params: BodyParams::new().with_position(position),
            additional_mass: Some(MassProperties {
                mass: 1.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn material() -> ContactMaterial {
    ContactMaterial {
        stiffness: 1e4,
        damping: 10.,
        friction: Some(FrictionCoefficients::uniform(0.4)),
        ..Default::default()
    }
}

fn sphere(scene: &mut RigidScene, parent: BodyHandle, sensor: bool) -> ColliderHandle {
    let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    collider.parent = Some(parent);
    collider.sensor = sensor;
    collider.mass = ColliderMass::Mass(0.0);
    collider.contact_material = material();
    scene.insert_collider(collider).unwrap()
}

fn frozen_support(gpu: bool) {
    for substeps in [1, 4] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(-Vec3::Y * 9.81).unwrap();
        let a = body(
            &mut scene,
            Vec3::new(0.123456789, 0.4995, 0.),
            BodyKind::Dynamic,
        );
        let remote = body(&mut scene, Vec3::new(8., 4., 0.), BodyKind::Dynamic);
        let shape = sphere(&mut scene, a, false);
        sphere(&mut scene, remote, false);
        let mut support = ColliderDesc::new(ColliderGeometry::Cuboid {
            half_extents: Vec3::new(2., 0.1, 2.),
        });
        support.pose = Transform::from_translation(-Vec3::Y * 0.1);
        support.contact_material = material();
        let support = scene.insert_collider(support).unwrap();
        let mut world = world(scene, gpu);
        world.solver.substeps = substeps;
        world.step(0.001).unwrap();
        assert!(
            world
                .contact_pairs()
                .iter()
                .any(|p| p.colliders.contains(&shape))
        );
        world.sleep_body(a).unwrap();
        world
            .scene
            .set_body_velocity_with_wake(
                a,
                SpatialVector::new(Vec3::X * 0.123456789, Vec3::Z * 0.234567891),
                false,
            )
            .unwrap();
        let pose = world.scene.body_pose(a).unwrap();
        let velocity = world.scene.body_velocity(a).unwrap();
        let remote_y = world.scene.body_pose(remote).unwrap().position.y;
        #[cfg(feature = "gpu")]
        let diagnostics = world.gpu_diagnostics();
        for _ in 0..8 {
            world.step(0.001).unwrap();
            assert!(world.is_sleeping(a));
            assert!(!world.is_sleeping(remote));
            assert_eq!(world.scene.body_pose(a).unwrap(), pose);
            assert_eq!(world.scene.body_velocity(a).unwrap(), velocity);
            let pairs = world.contact_pairs();
            let pair = pairs.iter().find(|p| p.colliders.contains(&shape)).unwrap();
            assert!(pair.colliders.contains(&support));
            assert_eq!(pair.points.len(), substeps);
            for point in &pair.points {
                assert!(!point.acted);
                assert_eq!(point.force_on_b, Vec3::ZERO);
                assert_eq!(point.normal_impulse, 0.0);
                assert_eq!(point.tangent_impulse, Vec3::ZERO);
                assert!(point.penetration > 0.0);
            }
            #[cfg(feature = "gpu")]
            if let Some(previous) = diagnostics {
                let current = world.gpu_diagnostics().unwrap();
                assert_eq!(current.model_compilations, previous.model_compilations);
                assert_eq!(
                    current.current_model_bind_groups_created,
                    previous.current_model_bind_groups_created
                );
            }
        }
        assert!(world.scene.body_pose(remote).unwrap().position.y < remote_y);
        world.scene.remove_collider(support).unwrap();
        world.step(0.001).unwrap();
        assert!(!world.is_sleeping(a));
        assert!(world.scene.body_velocity(a).unwrap().linear.y < 0.0);
    }
}

fn fresh_collision(gpu: bool) {
    for mode in 0..3 {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let a = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
        let b = body(&mut scene, Vec3::X * 1.4, BodyKind::Dynamic);
        let remote = body(&mut scene, Vec3::X * 8., BodyKind::Dynamic);
        let sa = sphere(&mut scene, a, false);
        let sb = sphere(&mut scene, b, mode == 1);
        sphere(&mut scene, remote, false);
        if mode == 2 {
            scene.set_collider_pair_excluded(sa, sb, true).unwrap();
        }
        scene
            .set_body_velocity(b, SpatialVector::new(-Vec3::X * 20., Vec3::ZERO))
            .unwrap();
        let mut world = world(scene, gpu);
        world.solver.substeps = 4;
        world.sleep_body(a).unwrap();
        world.sleep_body(remote).unwrap();
        let pose = world.scene.body_pose(a).unwrap();
        let remote_pose = world.scene.body_pose(remote).unwrap();
        world.step(0.04).unwrap();
        assert_eq!(world.is_sleeping(a), mode != 0, "mode={mode}");
        assert!(world.is_sleeping(remote));
        assert_eq!(world.scene.body_pose(remote).unwrap(), remote_pose);
        if mode == 0 {
            assert!(world.scene.body_velocity(a).unwrap().linear.x < -0.1);
            assert!(
                world
                    .contact_pairs()
                    .iter()
                    .flat_map(|p| &p.points)
                    .any(|p| p.acted)
            );
        } else {
            assert_eq!(world.scene.body_pose(a).unwrap(), pose);
        }
    }
}

fn moving_boundary(gpu: bool) {
    for acceleration in [false, true] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let a = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
        let b = body(&mut scene, Vec3::X * 0.9, BodyKind::Kinematic);
        sphere(&mut scene, a, false);
        sphere(&mut scene, b, false);
        if acceleration {
            scene
                .set_kinematic_acceleration(b, Some(SpatialVector::new(-Vec3::X, Vec3::ZERO)))
                .unwrap();
        } else {
            scene
                .set_body_velocity(b, SpatialVector::new(-Vec3::X, Vec3::ZERO))
                .unwrap();
        }
        let mut world = world(scene, gpu);
        world.sleep_body(a).unwrap();
        assert!(world.is_sleeping(a));
        world.step(0.001).unwrap();
        assert!(!world.is_sleeping(a));
        assert!(world.scene.body_velocity(a).unwrap().linear.x < 0.0);
    }
}

fn automatic_contacts(gpu: bool) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let a = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
    let b = body(&mut scene, Vec3::X * 0.9, BodyKind::Dynamic);
    for parent in [a, b] {
        let mut c = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
        c.parent = Some(parent);
        c.mass = ColliderMass::Mass(0.);
        c.contact_material = ContactMaterial {
            stiffness: 0.,
            damping: 0.,
            ..material()
        };
        scene.insert_collider(c).unwrap();
    }
    let mut world = world(scene, gpu);
    world
        .set_sleep_settings(Some(SleepSettings {
            quiet_time: 0.03,
            ..Default::default()
        }))
        .unwrap();
    for _ in 0..6 {
        world.step(0.01).unwrap();
    }
    assert!(world.is_sleeping(a) && world.is_sleeping(b));
    let poses = [a, b].map(|body| world.scene.body_pose(body).unwrap());
    for _ in 0..3 {
        world.step(0.01).unwrap();
        assert!(world.is_sleeping(a) && world.is_sleeping(b));
        assert_eq!(
            [a, b].map(|body| world.scene.body_pose(body).unwrap()),
            poses
        );
        assert!(!world.contact_pairs().is_empty());
    }
    world
        .scene
        .add_body_wrench(a, SpatialVector::new(Vec3::Y, Vec3::ZERO))
        .unwrap();
    world.step(0.01).unwrap();
    assert!(!world.is_sleeping(a) && !world.is_sleeping(b));
    assert!(world.scene.body_velocity(a).unwrap().linear.y > 0.0);
}

fn joint_activation(gpu: bool) {
    use molla_sim::joint_control::{JointAxes, JointAxis, JointMotor, JointSoftness};
    use molla_sim::runtime::JointDesc;
    use molla_sim::{JointTargetMode, JointType};
    for route in 0..3 {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let a = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
        let b = body(&mut scene, Vec3::X * 2., BodyKind::Dynamic);
        let incoming = body(&mut scene, Vec3::X * 1.4, BodyKind::Dynamic);
        let remote = body(&mut scene, Vec3::X * 8., BodyKind::Dynamic);
        sphere(&mut scene, a, false);
        sphere(&mut scene, incoming, false);
        sphere(&mut scene, remote, false);
        let desc = JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::D6,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(Vec3::X * 2.),
            frame_child: Transform::IDENTITY,
        };
        let joint = if route == 0 {
            scene.insert_joint(desc).unwrap()
        } else {
            scene
                .insert_loop_joint(
                    desc,
                    JointSoftness {
                        natural_frequency: if route == 1 { 0. } else { 4. },
                        damping_ratio: 0.6,
                    },
                )
                .unwrap()
        };
        if route == 1 {
            scene.set_joint_softness(joint, None).unwrap();
        }
        scene.set_joint_locked_axes(joint, JointAxes::NONE).unwrap();
        scene
            .set_joint_motor(
                joint,
                JointAxis::AngZ,
                JointMotor {
                    mode: JointTargetMode::Effort,
                    target_position: 0.3,
                    max_force: f64::INFINITY,
                    ..Default::default()
                },
            )
            .unwrap();
        scene.set_joint_brake(joint, JointAxis::AngZ, 0.2).unwrap();
        let mut world = world(scene, gpu);
        world.solver.substeps = 4;
        world.set_reaction_readout(true).unwrap();
        world.sleep_body(a).unwrap();
        world.sleep_body(remote).unwrap();
        assert!(world.is_sleeping(a) && world.is_sleeping(b));
        let poses = [a, b, remote].map(|b| world.scene.body_pose(b).unwrap());
        world.step(0.001).unwrap();
        assert!(
            world.joint_reaction(joint).is_none(),
            "sleeping route={route}"
        );
        assert_eq!(
            [a, b, remote].map(|b| world.scene.body_pose(b).unwrap()),
            poses
        );
        world
            .scene
            .set_body_pose(incoming, Transform::from_translation(Vec3::X * 0.9))
            .unwrap();
        world
            .scene
            .set_body_velocity(incoming, SpatialVector::new(-Vec3::X * 0.1, Vec3::ZERO))
            .unwrap();
        world.step(0.001).unwrap();
        assert!(
            !world.is_sleeping(a) && !world.is_sleeping(b),
            "wake route={route}"
        );
        assert!(world.is_sleeping(remote));
        assert_eq!(world.scene.body_pose(remote).unwrap(), poses[2]);
        assert!(world.scene.body_velocity(a).unwrap().linear.x < 0.0);
        assert!(
            world.scene.body_velocity(b).unwrap().angular.z > 0.0,
            "motor route={route}"
        );
        let reaction = world.joint_reaction(joint).unwrap();
        assert!(
            reaction.child.angular.z > 0.25,
            "reaction route={route}: {reaction:?}"
        );
        let samples = world.joint_reaction_samples().unwrap();
        assert_eq!(samples.iter().filter(|s| s.joint == joint).count(), 4);
    }
}

#[test]
fn cpu_contact_sleep_retains_geometry_and_nonwaking_state() {
    frozen_support(false);
}
#[test]
fn cpu_new_substep_contacts_wake_only_eligible_components() {
    fresh_collision(false);
}
#[test]
fn cpu_moving_boundaries_wake_new_contact_components() {
    moving_boundary(false);
}
#[test]
fn cpu_automatic_contact_groups_sleep_and_wake_together() {
    automatic_contacts(false);
}
#[test]
fn cpu_collision_wake_activates_tree_hard_soft_drives_and_reactions() {
    joint_activation(false);
}

fn hot_compliance_gain(gpu: bool) {
    use molla_math::Quat;
    use molla_sim::JointType;
    use molla_sim::joint_control::{JointAxis, JointSoftness};
    use molla_sim::runtime::JointDesc;
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = body(&mut scene, Vec3::ZERO, BodyKind::Dynamic);
    scene
        .set_body_pose(body, Transform::from_rotation(Quat::from_rotation_x(-0.1)))
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
        .set_joint_limits(joint, JointAxis::AngX, Some([0., 0.5]))
        .unwrap();
    scene
        .set_joint_softness(
            joint,
            Some(JointSoftness {
                natural_frequency: 0.,
                damping_ratio: 0.7,
            }),
        )
        .unwrap();
    let mut world = world(scene, gpu);
    world.solver.substeps = 1;
    world.sleep_body(body).unwrap();
    let pose = world.scene.body_pose(body).unwrap();
    world.step(0.001).unwrap();
    assert!(world.is_sleeping(body));
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    let stamp = world.scene.model_stamp();
    let revision = world.scene.revision();
    #[cfg(feature = "gpu")]
    let compilations = world.gpu_diagnostics().map(|d| d.model_compilations);
    world
        .scene
        .set_joint_softness(
            joint,
            Some(JointSoftness {
                natural_frequency: 3.,
                damping_ratio: 0.7,
            }),
        )
        .unwrap();
    assert_eq!(world.scene.model_stamp(), stamp);
    assert_eq!(world.scene.revision(), revision);
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert!(!world.is_sleeping(body));
    world.step(0.001).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    let after = world.scene.body_pose(body).unwrap();
    println!("hot_compliance_gain gpu={gpu} velocity={velocity:?} pose={after:?}");
    assert!(velocity.linear.is_finite() && velocity.angular.is_finite());
    assert!(after.position.is_finite() && after.rotation.is_finite());
    let omega = std::f64::consts::TAU * 3.;
    let expected = 0.1 * omega * omega * 0.001
        / (1. + 2. * 0.7 * omega * 0.001 + omega * omega * 0.001 * 0.001);
    println!(
        "hot_compliance_gain rate_error={:.9e} tolerance=1e-6",
        (velocity.angular.x - expected).abs()
    );
    assert!((velocity.angular.x - expected).abs() <= 1e-6);
    assert!(world.quarantined_bodies().is_empty());
    assert!(world.joint_break_events().is_empty());
    assert!(
        after.rotation.angle_between(Quat::IDENTITY) < pose.rotation.angle_between(Quat::IDENTITY)
    );
    assert_eq!(world.scene.model_stamp(), stamp);
    #[cfg(feature = "gpu")]
    assert_eq!(
        world.gpu_diagnostics().map(|d| d.model_compilations),
        compilations
    );
}

#[test]
fn cpu_hot_compliance_gain_wakes_retained_tree() {
    hot_compliance_gain(false);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_hot_compliance_gain_wakes_retained_tree() {
    hot_compliance_gain(true);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_contact_sleep_retains_geometry_and_nonwaking_state() {
    frozen_support(true);
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_new_substep_contacts_wake_only_eligible_components() {
    fresh_collision(true);
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_moving_boundaries_wake_new_contact_components() {
    moving_boundary(true);
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_automatic_contact_groups_sleep_and_wake_together() {
    automatic_contacts(true);
}
#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_collision_wake_activates_tree_hard_soft_drives_and_reactions() {
    joint_activation(true);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires GPU; use make test-gpu"]
fn gpu_contact_wake_fault_rolls_back_and_same_layout_retry_recovers() {
    use molla_solvers::rigid_world::GpuContactCapacity;
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let bodies: Vec<_> = [0., 0.9, 1.8]
        .into_iter()
        .map(|x| {
            let b = body(&mut scene, Vec3::X * x, BodyKind::Dynamic);
            sphere(&mut scene, b, false);
            b
        })
        .collect();
    let mut world = RigidWorld::from_scene_gpu_with_contact_capacity(
        scene,
        engine(),
        GpuContactCapacity {
            pairs: 16,
            contacts_per_island: 1,
        },
    )
    .unwrap();
    world.sleep_body(bodies[0]).unwrap();
    let poses: Vec<_> = bodies
        .iter()
        .map(|&b| world.scene.body_pose(b).unwrap())
        .collect();
    let stamp = world.scene.stamp();
    let error = world.step(0.001).unwrap_err();
    assert!(error.to_string().contains("solver fault"), "{error}");
    assert_eq!(world.scene.stamp(), stamp);
    assert!(world.is_sleeping(bodies[0]));
    assert!(world.contact_pairs().is_empty());
    assert_eq!(
        bodies
            .iter()
            .map(|&b| world.scene.body_pose(b).unwrap())
            .collect::<Vec<_>>(),
        poses
    );
    let compilations = world.gpu_diagnostics().unwrap().model_compilations;
    world
        .scene
        .set_body_pose(bodies[2], Transform::from_translation(Vec3::X * 8.))
        .unwrap();
    world.step(0.001).unwrap();
    assert!(!world.is_sleeping(bodies[0]));
    assert!(world.scene.body_velocity(bodies[0]).unwrap().linear.x < 0.0);
    assert_eq!(
        world.gpu_diagnostics().unwrap().model_compilations,
        compilations
    );
}
