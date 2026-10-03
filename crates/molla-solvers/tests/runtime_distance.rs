use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, DistanceLimits, JointDesc, JointHandle};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::rigid_world::{DistanceBound, RigidWorld};

fn body(world: &mut RigidWorld, position: Vec3, mass: f64, velocity: Vec3) -> BodyHandle {
    world
        .scene
        .insert_body(BodyDesc {
            params: BodyParams {
                pose: Transform::from_translation(position),
                ..Default::default()
            },
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * mass,
            }),
            velocity: SpatialVector::new(velocity, Vec3::ZERO),
            ..Default::default()
        })
        .unwrap()
}

fn link(
    world: &mut RigidWorld,
    parent: Option<BodyHandle>,
    child: BodyHandle,
    min: Option<f64>,
    max: Option<f64>,
) -> JointHandle {
    let joint = world
        .scene
        .insert_joint(JointDesc {
            parent,
            child,
            kind: JointType::Distance,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    world
        .scene
        .set_distance_limits(joint, DistanceLimits { min, max })
        .unwrap();
    joint
}

#[test]
fn independent_bounds_remain_slack_and_stop_only_at_their_bound() {
    for (min, max, vx, expected, active) in [
        (None, None, 1.0, 2.0, DistanceBound::Inactive),
        (None, Some(3.0), 1.0, 2.0, DistanceBound::Inactive),
        (Some(0.5), None, 1.0, 2.0, DistanceBound::Inactive),
        (None, Some(1.5), 1.0, 1.5, DistanceBound::Maximum),
        (Some(0.5), None, -1.0, 0.5, DistanceBound::Minimum),
    ] {
        let mut world = RigidWorld::new().unwrap();
        world.scene.set_gravity(Vec3::ZERO).unwrap();
        let b = body(&mut world, Vec3::X, 2.0, Vec3::X * vx);
        let j = link(&mut world, None, b, min, max);
        world.step(1.0).unwrap();
        let reading = world.distance_reading(j).unwrap();
        assert!((reading.separation - expected).abs() < 1e-9, "{reading:?}");
        assert_eq!(reading.active, active);
    }
}

#[test]
fn equality_transfers_load_between_unequal_masses() {
    for masses in [[2.0, 3.0], [1.0, 1000.0], [1000.0, 1.0]] {
        let mut world = RigidWorld::new().unwrap();
        world.scene.set_gravity(Vec3::ZERO).unwrap();
        let a = body(&mut world, Vec3::ZERO, masses[0], Vec3::ZERO);
        let b = body(&mut world, Vec3::X, masses[1], Vec3::ZERO);
        let j = link(&mut world, Some(a), b, Some(1.0), Some(1.0));
        world
            .scene
            .add_body_wrench(b, SpatialVector::new(Vec3::X * 10.0, Vec3::ZERO))
            .unwrap();
        world.step(0.01).unwrap();
        let expected = 0.1 / (masses[0] + masses[1]);
        for id in [a, b] {
            assert!((world.scene.body_velocity(id).unwrap().linear.x - expected).abs() < 1e-9);
        }
        let reading = world.distance_reading(j).unwrap();
        assert!(reading.violation < 1e-9);
        assert!((reading.force.x + 10.0 * masses[0] / (masses[0] + masses[1])).abs() < 1e-7);
    }
}

#[test]
fn coincident_equality_blocks_translation_but_not_rotation() {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let b = body(&mut world, Vec3::ZERO, 2.0, Vec3::new(1.0, -2.0, 3.0));
    world
        .scene
        .set_body_velocity(b, SpatialVector::new(Vec3::new(1.0, -2.0, 3.0), Vec3::Y))
        .unwrap();
    let j = link(&mut world, None, b, Some(0.0), Some(0.0));
    world.step(0.01).unwrap();
    let v = world.scene.body_velocity(b).unwrap();
    assert!(v.linear.length() < 1e-9);
    assert!((v.angular - Vec3::Y).length() < 1e-9);
    assert!(world.distance_reading(j).unwrap().violation < 1e-9);
}

#[test]
fn invalid_bounds_are_atomic_and_mutation_invalidates_readings() {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let b = body(&mut world, Vec3::X, 1.0, Vec3::ZERO);
    let j = link(&mut world, None, b, None, Some(2.0));
    world.step(0.01).unwrap();
    let stamp = world.scene.stamp();
    for limits in [
        DistanceLimits {
            min: Some(-1.0),
            max: None,
        },
        DistanceLimits {
            min: None,
            max: Some(f64::INFINITY),
        },
        DistanceLimits {
            min: Some(f64::NAN),
            max: None,
        },
        DistanceLimits {
            min: Some(2.0),
            max: Some(1.0),
        },
    ] {
        assert!(world.scene.set_distance_limits(j, limits).is_err());
        assert_eq!(world.scene.stamp(), stamp);
        assert!(world.distance_reading(j).is_some());
    }
    world
        .scene
        .set_distance_limits(
            j,
            DistanceLimits {
                min: Some(1.0),
                max: Some(1.0),
            },
        )
        .unwrap();
    assert!(world.distance_reading(j).is_none());
    world.step(0.01).unwrap();
    let mut settings = world.scene.joint_settings(j).unwrap();
    settings.enabled = false;
    world.scene.set_joint_settings(j, settings).unwrap();
    world
        .scene
        .set_body_velocity(b, SpatialVector::new(Vec3::X, Vec3::ZERO))
        .unwrap();
    world.step(0.1).unwrap();
    assert!(world.distance_reading(j).is_none());
    assert!((world.scene.body_pose(b).unwrap().position.x - 1.1).abs() < 1e-9);
    world.scene.remove_joint(j).unwrap();
    assert!(world.scene.distance_limits(j).is_err());
}

#[test]
fn world_pendulum_holds_length_for_ten_seconds() {
    for dt in [1.0 / 60.0, 1.0 / 120.0, 1.0 / 240.0] {
        let mut world = RigidWorld::new().unwrap();
        world.solver.substeps = 8;
        world.scene.set_gravity(Vec3::new(0.0, -9.81, 0.0)).unwrap();
        let b = body(&mut world, Vec3::X, 1.0, Vec3::Z);
        let j = link(&mut world, None, b, Some(1.0), Some(1.0));
        let mut peak: f64 = 0.0;
        for _ in 0..(10.0 / dt) as usize {
            world.step(dt).unwrap();
            peak = peak.max(world.distance_reading(j).unwrap().violation);
        }
        println!("distance pendulum: dt={dt}, substeps=8, duration=10s, peak={peak}");
        assert!(peak <= 1e-3, "dt={dt}, peak={peak}");
    }
}

#[test]
fn zero_minimum_without_maximum_is_free_at_coincident_anchors() {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let v = Vec3::new(-1.0, 2.0, -3.0);
    let b = body(&mut world, Vec3::ZERO, 1.0, v);
    let j = link(&mut world, None, b, Some(0.0), None);
    world.step(0.01).unwrap();
    assert!((world.scene.body_velocity(b).unwrap().linear - v).length() < 1e-10);
    assert_eq!(world.distance_reading(j).unwrap().force, Vec3::ZERO);
}

#[test]
fn zero_maximum_alone_locks_all_translation() {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let b = body(&mut world, Vec3::ZERO, 1.0, Vec3::new(-1.0, 2.0, -3.0));
    let j = link(&mut world, None, b, None, Some(0.0));
    world.step(0.01).unwrap();
    assert!(world.scene.body_velocity(b).unwrap().linear.length() < 1e-10);
    assert!(world.distance_reading(j).unwrap().violation < 1e-10);
}

#[test]
fn distance_and_scalar_stop_share_the_articulated_response() {
    use molla_sim::joint_control::JointAxis;
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let a = body(&mut world, Vec3::ZERO, 2.0, Vec3::ZERO);
    let b = body(&mut world, Vec3::ZERO, 3.0, Vec3::ZERO);
    let c = body(&mut world, Vec3::X, 5.0, Vec3::ZERO);
    let slider = world
        .scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Prismatic,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    world
        .scene
        .set_joint_limits(slider, JointAxis::LinX, Some([-1.0, 0.0]))
        .unwrap();
    let distance = link(&mut world, Some(b), c, Some(1.0), Some(1.0));
    world
        .scene
        .add_body_wrench(c, SpatialVector::new(Vec3::X * 10.0, Vec3::ZERO))
        .unwrap();
    world.step(0.01).unwrap();
    for id in [a, b, c] {
        assert!((world.scene.body_velocity(id).unwrap().linear.x - 0.01).abs() < 1e-9);
    }
    assert!(world.distance_reading(distance).unwrap().violation < 1e-9);
}

#[test]
fn offset_anchor_uses_full_inertia_and_com_moment_arm() {
    let mut world = RigidWorld::new().unwrap();
    world.solver.substeps = 1;
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let b = body(&mut world, Vec3::ZERO, 2.0, Vec3::X);
    let inertia = Mat3::from_cols(
        Vec3::new(3.0, 0.2, 0.3),
        Vec3::new(0.2, 4.0, 0.4),
        Vec3::new(0.3, 0.4, 5.0),
    );
    let com = Vec3::new(0.1, 0.2, -0.1);
    world
        .scene
        .set_body_mass_properties(
            b,
            MassProperties {
                mass: 2.0,
                local_com: com,
                inertia,
            },
        )
        .unwrap();
    let j = world
        .scene
        .insert_joint(JointDesc {
            parent: None,
            child: b,
            kind: JointType::Distance,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(Vec3::new(-1.0, 1.0, 0.0)),
            frame_child: Transform::from_translation(Vec3::Y),
        })
        .unwrap();
    world
        .scene
        .set_distance_limits(
            j,
            DistanceLimits {
                min: Some(1.0),
                max: Some(1.0),
            },
        )
        .unwrap();
    let moment = (Vec3::Y - com).cross(Vec3::X);
    let angular_response = inertia.inverse() * moment;
    let impulse = -1.0 / (0.5 + moment.dot(angular_response));
    world.step(1e-6).unwrap();
    let v = world.scene.body_velocity(b).unwrap();
    assert!(
        (v.linear - Vec3::X * (1.0 + impulse * 0.5)).length() < 1e-6,
        "{v:?}"
    );
    assert!((v.angular - angular_response * impulse).length() < 1e-6);
    assert!((world.distance_reading(j).unwrap().force.x * 1e-6 - impulse).abs() < 1e-9);
}

#[test]
fn closed_triangle_and_redundant_link_preserve_linear_momentum() {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let a = body(&mut world, Vec3::ZERO, 1.0, Vec3::X);
    let b = body(&mut world, Vec3::X, 2.0, Vec3::Y);
    let c = body(&mut world, Vec3::Y, 3.0, -Vec3::X);
    let mut joints = Vec::new();
    for (parent, child, length) in [
        (a, b, 1.0),
        (b, c, 2.0_f64.sqrt()),
        (c, a, 1.0),
        (a, b, 1.0),
    ] {
        joints.push(link(
            &mut world,
            Some(parent),
            child,
            Some(length),
            Some(length),
        ));
    }
    let before = Vec3::new(-2.0, 2.0, 0.0);
    for _ in 0..120 {
        world.step(1.0 / 120.0).unwrap();
        let after = world.scene.body_velocity(a).unwrap().linear
            + world.scene.body_velocity(b).unwrap().linear * 2.0
            + world.scene.body_velocity(c).unwrap().linear * 3.0;
        assert!((after - before).length() < 1e-8);
        for j in &joints {
            assert!(world.distance_reading(*j).unwrap().violation < 1e-3);
        }
    }
}

#[test]
fn infeasible_fixed_constraint_fails_atomically_and_invalidates_readout() {
    let mut world = RigidWorld::new().unwrap();
    world.scene.set_gravity(Vec3::ZERO).unwrap();
    let b = body(&mut world, Vec3::X, 1.0, Vec3::ZERO);
    let j = link(&mut world, None, b, Some(1.0), Some(1.0));
    world.step(0.01).unwrap();
    assert!(world.distance_reading(j).is_some());
    assert!(world.step(f64::NAN).is_err());
    assert!(world.distance_reading(j).is_none());
    world
        .scene
        .set_body_kind(b, molla_sim::runtime::BodyKind::Fixed)
        .unwrap();
    world
        .scene
        .set_distance_limits(
            j,
            DistanceLimits {
                min: Some(2.0),
                max: Some(2.0),
            },
        )
        .unwrap();
    let stamp = world.scene.stamp();
    let pose = world.scene.body_pose(b).unwrap();
    assert!(world.step(0.01).is_err());
    assert_eq!(world.scene.stamp(), stamp);
    assert_eq!(world.scene.body_pose(b).unwrap(), pose);
}
