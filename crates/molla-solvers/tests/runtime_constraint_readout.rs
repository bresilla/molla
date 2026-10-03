use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, DistanceLimits, JointDesc, JointHandle, RigidScene, SphericalConeLimits,
};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::RigidWorld;

#[derive(Clone, Copy, Debug)]
enum Kind {
    Distance,
    Cone,
    BallAnchor,
}

fn fixture(kind: Kind) -> (RigidScene, BodyHandle, JointHandle) {
    let mut scene = RigidScene::new().unwrap();
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(match kind {
                Kind::Distance => Vec3::X,
                _ => Vec3::ZERO,
            }),
            additional_mass: Some(MassProperties {
                mass: 2.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let desc = JointDesc {
        parent: None,
        child: body,
        kind: match kind {
            Kind::Distance => JointType::Distance,
            _ => JointType::Ball,
        },
        axis: Vec3::X,
        frame_parent: Transform::IDENTITY,
        frame_child: Transform::IDENTITY,
    };
    let joint = match kind {
        Kind::BallAnchor => scene.insert_rigid_loop_joint(desc),
        _ => scene.insert_joint(desc),
    }
    .unwrap();
    match kind {
        Kind::Distance => scene
            .set_distance_limits(
                joint,
                DistanceLimits {
                    min: Some(1.0),
                    max: Some(1.0),
                },
            )
            .unwrap(),
        Kind::Cone => scene
            .set_cone_limits(
                joint,
                SphericalConeLimits {
                    angle0: Some(0.3),
                    angle1: Some(0.4),
                },
            )
            .unwrap(),
        Kind::BallAnchor => {}
    }
    (scene, body, joint)
}

fn reading(world: &RigidWorld, joint: JointHandle, kind: Kind) -> bool {
    match kind {
        Kind::Distance => world.distance_reading(joint).is_some(),
        Kind::Cone => world.cone_reading(joint).is_some(),
        Kind::BallAnchor => world.ball_anchor_reading(joint).is_some(),
    }
}

fn certify(mut make_world: impl FnMut(RigidScene) -> RigidWorld) {
    for kind in [Kind::Distance, Kind::Cone, Kind::BallAnchor] {
        let (scene, body, joint) = fixture(kind);
        let mut world = make_world(scene);
        world.set_reaction_readout(true).unwrap();
        for edit in 0..5 {
            world.step(0.001).unwrap();
            assert!(reading(&world, joint, kind), "{kind:?} edit={edit}");
            assert!(world.joint_reaction(joint).is_some());
            let compiled = world.scene.stamp();
            let dynamics = world.scene.dynamics_stamp();
            assert!(world.scene.set_gravity(Vec3::splat(f64::NAN)).is_err());
            assert!(reading(&world, joint, kind));
            world.scene.set_gravity(world.scene.gravity()).unwrap();
            assert!(reading(&world, joint, kind));
            assert_eq!(world.scene.dynamics_stamp(), dynamics);
            match edit {
                0 => world
                    .scene
                    .add_body_wrench_with_wake(body, SpatialVector::new(Vec3::Y, Vec3::ZERO), false)
                    .unwrap(),
                1 => world
                    .scene
                    .reset_body_forces_with_wake(body, false)
                    .unwrap(),
                2 => world.scene.set_gravity(Vec3::Y * -4.0).unwrap(),
                3 => {
                    world.scene.control_mut();
                }
                _ => {
                    world.scene.state_mut();
                }
            }
            assert_eq!(world.scene.stamp(), compiled);
            assert_ne!(world.scene.dynamics_stamp(), dynamics);
            assert!(world.joint_reaction(joint).is_none());
            assert!(
                !reading(&world, joint, kind),
                "stale {kind:?} readout after hot edit={edit}"
            );
        }
        world.step(0.001).unwrap();
        assert!(reading(&world, joint, kind));
        assert!(world.joint_reaction(joint).is_some());
    }
}

#[test]
fn hot_dynamics_edits_invalidate_constraint_readouts_without_rebuilding() {
    certify(|scene| RigidWorld::from_scene(scene).unwrap());
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires hardware GPU through make test-gpu"]
fn gpu_hot_dynamics_edits_invalidate_constraint_readouts_without_rebuilding() {
    let engine = std::sync::Arc::new(molla_compute::Engine::new_gpu().unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    certify(|scene| RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap());
}
