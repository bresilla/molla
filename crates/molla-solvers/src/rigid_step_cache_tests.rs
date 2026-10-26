use super::*;
use molla_sim::joint_control::{JointAxis, JointMotor};
use molla_sim::runtime::{BodyDesc, BodyKind, ColliderDesc, ColliderGeometry, JointDesc};
use molla_sim::{
    BodyParams, ContactMaterial, FrictionCoefficients, JointTargetMode, JointType, MassProperties,
};

fn body(world: &mut RigidWorld, position: Vec3) -> BodyHandle {
    world
        .scene
        .insert_body(BodyDesc {
            kind: BodyKind::Dynamic,
            params: BodyParams::new().with_position(position),
            additional_mass: Some(MassProperties {
                mass: 1.0,
                local_com: Vec3::ZERO,
                inertia: molla_math::Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap()
}

fn cuboid(world: &mut RigidWorld, parent: Option<BodyHandle>, half_extents: Vec3, at: Vec3) {
    let mut collider = ColliderDesc::new(ColliderGeometry::Cuboid { half_extents });
    collider.parent = parent;
    collider.pose = Transform::from_translation(at);
    collider.contact_material = ContactMaterial {
        stiffness: 1e4,
        damping: 10.,
        friction: Some(FrictionCoefficients::uniform(0.4)),
        ..Default::default()
    };
    world.scene.insert_collider(collider).unwrap();
}

/// A block resting on a floor with a driven, limited hinge arm, and a block asleep far off;
/// without momentum transport every substep starts at its coordinates' poses.
fn fixture() -> (RigidWorld, BodyHandle) {
    let mut world = RigidWorld::new().unwrap();
    world.solver.momentum_transport = false;
    world.scene.set_gravity(-Vec3::Y * 9.81).unwrap();
    cuboid(&mut world, None, Vec3::new(20., 0.1, 20.), -Vec3::Y * 0.1);
    let base = body(&mut world, Vec3::Y * 0.25);
    cuboid(&mut world, Some(base), Vec3::splat(0.25), Vec3::ZERO);
    let arm = body(&mut world, Vec3::new(0., 0.5, 0.5));
    let hinge = world
        .scene
        .insert_joint(JointDesc {
            parent: Some(base),
            child: arm,
            kind: JointType::Revolute,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(Vec3::new(0., 0.25, 0.25)),
            frame_child: Transform::from_translation(-Vec3::Z * 0.25),
        })
        .unwrap();
    world
        .scene
        .set_joint_limits(hinge, JointAxis::AngX, Some([-0.5, 0.5]))
        .unwrap();
    world
        .scene
        .set_joint_motor(
            hinge,
            JointAxis::AngX,
            JointMotor {
                mode: JointTargetMode::Position,
                target_position: 0.3,
                stiffness: 50.,
                damping: 5.,
                max_force: 100.,
                ..Default::default()
            },
        )
        .unwrap();
    let remote = body(&mut world, Vec3::new(8., 0.25, 0.));
    cuboid(&mut world, Some(remote), Vec3::splat(0.25), Vec3::ZERO);
    for _ in 0..20 {
        world.step(1. / 120.).unwrap();
    }
    world.sleep_body(remote).unwrap();
    (world, base)
}

#[test]
fn a_sleeping_step_builds_each_awake_mass_matrix_once_a_substep() {
    let (mut world, _) = fixture();
    assert!(world.sleeping.any_valid(&world.scene, &world.wheels));
    assert!(crate::mixed_constraints::needed(&world.scene).unwrap());
    let before = crate::crba::RESPONSE_BUILDS.with(std::cell::Cell::get);
    world.step(1. / 120.).unwrap();
    let builds = crate::crba::RESPONSE_BUILDS.with(std::cell::Cell::get) - before;
    assert_eq!(builds, world.solver.substeps);
}

#[test]
fn joint_islands_are_shared_between_steps_and_follow_scene_edits() {
    let (mut world, base) = fixture();
    let first = world.joint_islands().unwrap();
    world.step(1. / 120.).unwrap();
    assert!(Arc::ptr_eq(&first, &world.joint_islands().unwrap()));
    let extra = body(&mut world, Vec3::new(-3., 0.5, 0.));
    world
        .scene
        .insert_joint(JointDesc {
            parent: Some(base),
            child: extra,
            kind: JointType::Revolute,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    let edited = world.joint_islands().unwrap();
    assert!(first.body_island(extra).is_none());
    assert_eq!(
        edited.body_island(extra).map(|island| &island.bodies),
        edited.body_island(base).map(|island| &island.bodies)
    );
}
