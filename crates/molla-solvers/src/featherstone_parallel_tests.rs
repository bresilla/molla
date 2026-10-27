use super::{FORCED_PARTS, split_runs};
use crate::rigid_world::RigidWorld;
use molla_math::{Mat3, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, BodyKind, ColliderDesc, ColliderGeometry, JointDesc,
};
use molla_sim::{
    BodyParams, ContactMaterial, FrictionCoefficients, JointTargetMode, JointType, MassProperties,
};

#[test]
fn runs_cover_every_index_once_in_order_and_at_most_the_parts_asked() {
    let cases: [&[usize]; 7] = [
        &[],
        &[5],
        &[0, 0, 0],
        &[1, 1, 1, 1, 1, 1, 1],
        &[100, 1, 1, 1, 1],
        &[1, 1, 1, 1, 100],
        &[9, 0, 0, 4, 4, 0, 4],
    ];
    for weights in cases {
        for parts in 1..6 {
            let runs = split_runs(weights, parts);
            assert!(runs.len() <= parts, "{weights:?} in {parts}");
            assert!(
                runs.iter().all(|run| !run.is_empty()),
                "{weights:?} in {parts}"
            );
            assert_eq!(
                runs.iter().flat_map(Clone::clone).collect::<Vec<_>>(),
                (0..weights.len()).collect::<Vec<_>>(),
                "{weights:?} in {parts}"
            );
        }
    }
    assert_eq!(split_runs(&[1, 1, 1, 1], 2), vec![0..2, 2..4]);
    assert_eq!(split_runs(&[3, 1, 1, 1], 2), vec![0..1, 1..4]);
}

fn body(world: &mut RigidWorld, position: Vec3, half_extents: Vec3) -> BodyHandle {
    let body = world
        .scene
        .insert_body(BodyDesc {
            kind: BodyKind::Dynamic,
            params: BodyParams::new().with_position(position),
            additional_mass: Some(MassProperties {
                mass: 1.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    collider(world, Some(body), half_extents, Vec3::ZERO);
    body
}

fn collider(world: &mut RigidWorld, parent: Option<BodyHandle>, half_extents: Vec3, at: Vec3) {
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

/// Eight free-standing arms on a floor, each a base and three driven hinges.
fn fleet() -> (RigidWorld, Vec<BodyHandle>) {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(-Vec3::Y * 9.81).unwrap();
    collider(&mut world, None, Vec3::new(40., 0.1, 40.), -Vec3::Y * 0.1);
    let mut bases = Vec::new();
    for robot in 0..8 {
        let x = robot as f64 * 3.0 - 10.5;
        let base = body(&mut world, Vec3::new(x, 0.25, 0.), Vec3::splat(0.25));
        let mut parent = base;
        for link in 0..3 {
            let child = body(
                &mut world,
                Vec3::new(x, 0.75 + link as f64 * 0.5, 0.),
                Vec3::new(0.05, 0.25, 0.05),
            );
            let hinge = world
                .scene
                .insert_joint(JointDesc {
                    parent: Some(parent),
                    child,
                    kind: JointType::Revolute,
                    axis: if link == 1 { Vec3::Z } else { Vec3::X },
                    frame_parent: Transform::from_translation(Vec3::Y * 0.25),
                    frame_child: Transform::from_translation(-Vec3::Y * 0.25),
                })
                .unwrap();
            let motor = if link == 1 {
                JointMotor {
                    mode: JointTargetMode::Velocity,
                    target_velocity: 0.4 + 0.1 * robot as f64,
                    damping: 20.,
                    max_force: 40.,
                    ..Default::default()
                }
            } else {
                JointMotor {
                    mode: JointTargetMode::Position,
                    target_position: 0.2 * (robot as f64 - 3.5),
                    stiffness: 80.,
                    damping: 8.,
                    max_force: 60.,
                    ..Default::default()
                }
            };
            world
                .scene
                .set_joint_motor(hinge, JointAxis::AngX, motor)
                .unwrap();
            parent = child;
        }
        bases.push(base);
    }
    (world, bases)
}

/// The bits of every coordinate and velocity after each step, awake and
/// then with the first arm asleep, split into `parts`.
fn trace(parts: usize) -> Vec<u64> {
    FORCED_PARTS.set(Some(parts));
    let (mut world, bases) = fleet();
    assert!(world.scene.model().articulation_count >= 8);
    let mut bits = Vec::new();
    for step in 0..60 {
        if step == 30 {
            world.sleep_body(bases[0]).unwrap();
        }
        world.step(1. / 120.).unwrap();
        let state = world.scene.state();
        bits.extend(state.joint_q.host().unwrap().iter().map(|v| v.to_bits()));
        bits.extend(state.joint_qd.host().unwrap().iter().map(|v| v.to_bits()));
    }
    FORCED_PARTS.set(None);
    bits
}

#[test]
fn a_fleet_steps_the_same_bits_split_across_workers_as_in_one_run() {
    let serial = trace(1);
    for parts in [2, 3, 5] {
        assert!(serial == trace(parts), "{parts} parts");
    }
}
