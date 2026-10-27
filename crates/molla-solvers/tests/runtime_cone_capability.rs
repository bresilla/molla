use molla_math::{Mat3, Quat, Transform, Vec3};
use molla_sim::joint_control::JointSoftness;
use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene, SphericalConeLimits};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

fn capability(make: impl Fn(RigidScene) -> molla_core::Result<RigidWorld>) {
    for route in 0..3 {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let child = scene
            .insert_body(BodyDesc {
                params: BodyParams::new().with_pose(Transform::from_rotation(
                    Quat::from_rotation_y(std::f64::consts::PI),
                )),
                additional_mass: Some(MassProperties {
                    mass: 2.,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY,
                }),
                ..Default::default()
            })
            .unwrap();
        let desc = JointDesc {
            parent: None,
            child,
            kind: JointType::Ball,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        };
        let joint = match route {
            0 => scene.insert_joint(desc),
            1 => scene.insert_rigid_loop_joint(desc),
            _ => scene.insert_loop_joint(
                desc,
                JointSoftness {
                    natural_frequency: 30.,
                    damping_ratio: 1.,
                },
            ),
        }
        .unwrap();
        scene
            .set_cone_limits(
                joint,
                SphericalConeLimits {
                    angle0: Some(0.4),
                    angle1: Some(0.9),
                },
            )
            .unwrap();
        let mut world = make(scene).unwrap();
        world.solver.substeps = 1;
        let stamp = world.scene.stamp();
        let pose = world.scene.body_pose(child).unwrap();
        let q = world.scene.state().joint_q.host().unwrap().to_vec();
        let qd = world.scene.state().joint_qd.host().unwrap().to_vec();
        let wrench = world.scene.state().body_f.host().unwrap().to_vec();
        for custom in [false, true] {
            let result = if custom {
                world.step_with_forces(
                    0.01,
                    |_, _| false,
                    |_, _, _, _| Err(molla_core::Error::Solver("injected force failure".into())),
                )
            } else {
                world.step(f64::NAN)
            };
            assert!(result.is_err());
            assert_eq!(world.scene.stamp(), stamp);
            assert_eq!(world.scene.body_pose(child).unwrap(), pose);
            assert_eq!(world.scene.state().joint_q.host().unwrap(), q);
            assert_eq!(world.scene.state().joint_qd.host().unwrap(), qd);
            assert_eq!(world.scene.state().body_f.host().unwrap(), wrench);
            assert!(world.cone_reading(joint).is_none());
        }
        for frame in 0..32 {
            world.step(0.01).unwrap();
            let reading = world.cone_reading(joint).unwrap();
            let pose = world.scene.body_pose(child).unwrap();
            let rate = world.scene.body_velocity(child).unwrap();
            assert!(pose.position.length() <= 1e-3 && pose.rotation.is_finite());
            assert!(rate.linear.is_finite() && rate.angular.is_finite());
            assert!(reading.violation.is_finite() && reading.torque.is_finite());
            assert!(world.quarantined_bodies().is_empty());
            assert!(world.joint_break_events().is_empty());
            if frame == 0 {
                assert!(reading.active && reading.torque.length() > 1.);
            }
        }
        let residual = world.cone_reading(joint).unwrap().violation;
        println!("antipodal_cone route={route} frames=32 residual={residual:.9e}");
        assert!(residual <= 1e-3);
        world.scene.set_body_enabled(child, false).unwrap();
        world.step(0.01).unwrap();
        assert!(world.cone_reading(joint).is_none());
        world.scene.set_body_enabled(child, true).unwrap();
        world
            .scene
            .set_cone_limits(joint, SphericalConeLimits::default())
            .unwrap();
        world.step(0.01).unwrap();
    }
}

#[test]
fn cpu_antipodal_cones_recover_through_tree_rigid_and_compliant_routes() {
    capability(RigidWorld::from_scene);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_antipodal_cones_recover_through_tree_rigid_and_compliant_routes() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    capability(|scene| RigidWorld::from_scene_gpu(scene, engine.clone()));
}

fn admissible_motion(make: impl Fn(RigidScene) -> molla_core::Result<RigidWorld>) {
    for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
        let mut scene = RigidScene::new().unwrap();
        scene.set_gravity(Vec3::ZERO).unwrap();
        let body = scene
            .insert_body(BodyDesc {
                params: BodyParams::new().with_pose(Transform::from_rotation(
                    Quat::from_rotation_y(std::f64::consts::PI),
                )),
                additional_mass: Some(MassProperties {
                    mass: 2.,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY,
                }),
                ..Default::default()
            })
            .unwrap();
        let joint = scene
            .insert_joint(JointDesc {
                parent: None,
                child: body,
                kind: JointType::Ball,
                axis: Vec3::X,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        scene
            .set_cone_limits(
                joint,
                SphericalConeLimits {
                    angle0: None,
                    angle1: Some(0.9),
                },
            )
            .unwrap();
        scene
            .set_body_velocity(body, molla_math::SpatialVector::new(Vec3::ZERO, axis * 0.1))
            .unwrap();
        let mut world = make(scene).unwrap();
        world.solver.substeps = 1;
        let mut peak: f64 = 0.;
        for _ in 0..32 {
            world.step(0.005).unwrap();
            let reading = world.cone_reading(joint).unwrap();
            let rate = world.scene.body_velocity(body).unwrap();
            let pose = world.scene.body_pose(body).unwrap();
            peak = peak.max(reading.violation);
            assert!(reading.violation.is_finite() && reading.violation <= 1e-3);
            assert!(rate.angular.is_finite() && rate.angular.length() <= 0.1001);
            assert!(rate.linear.length() <= 1e-3);
            assert!(pose.position.length() <= 1e-3 && pose.rotation.is_finite());
            assert!(world.quarantined_bodies().is_empty());
            if axis != Vec3::Y {
                assert!((rate.angular - axis * 0.1).length() <= 1e-4);
                assert!(reading.torque.length() <= 1e-4);
            }
        }
        println!("admissible_antipode axis={axis:?} peak={peak:.9e}");
    }
}

#[test]
fn cpu_admissible_antipode_preserves_twist_and_free_swing_without_energy_spikes() {
    admissible_motion(RigidWorld::from_scene);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a hardware GPU"]
fn gpu_admissible_antipode_preserves_twist_and_free_swing_without_energy_spikes() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires a hardware GPU");
    println!("adapter={adapter:?}");
    admissible_motion(|scene| RigidWorld::from_scene_gpu(scene, engine.clone()));
}
