use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{
    BodyDesc, BodyHandle, ColliderDesc, ColliderGeometry, DistanceLimits, JointDesc, JointHandle,
    RigidScene,
};
use molla_sim::{
    BodyParams, ColliderMass, ContactMaterial, FrictionCoefficients, JointType, MassProperties,
};
use molla_solvers::rigid_world::RigidWorld;

fn body(scene: &mut RigidScene, position: Vec3) -> BodyHandle {
    let body = scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_position(position),
            additional_mass: Some(MassProperties {
                mass: 1.,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    collider.parent = Some(body);
    collider.mass = ColliderMass::Mass(0.);
    collider.contact_material = material();
    scene.insert_collider(collider).unwrap();
    body
}

fn material() -> ContactMaterial {
    ContactMaterial {
        stiffness: 1e5,
        damping: 0.,
        friction: Some(FrictionCoefficients {
            mu_static: 0.125,
            mu_dynamic: 0.1,
        }),
        ..Default::default()
    }
}

fn scene() -> RigidScene {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(-Vec3::Z * 2.).unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(5., 5., 0.1),
    });
    ground.pose = Transform::from_translation(-Vec3::Z * 0.1);
    ground.contact_material = material();
    scene.insert_collider(ground).unwrap();
    scene
}

fn distance(
    scene: &mut RigidScene,
    parent: Option<BodyHandle>,
    child: BodyHandle,
    anchor: Vec3,
    length: f64,
) -> JointHandle {
    let joint = scene
        .insert_joint(JointDesc {
            parent,
            child,
            kind: JointType::Distance,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(anchor),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_distance_limits(
            joint,
            DistanceLimits {
                min: Some(length),
                max: Some(length),
            },
        )
        .unwrap();
    joint
}

#[test]
fn distance_and_contact_share_one_cpu_impulse_solution() {
    for capture in [false, true] {
        let mut scene = scene();
        let b = body(&mut scene, Vec3::new(-1., 1., 0.49998));
        let j = distance(&mut scene, None, b, Vec3::Z * 0.49998, 2.0_f64.sqrt());
        scene
            .set_body_velocity(b, SpatialVector::new(Vec3::Y * 0.01, Vec3::ZERO))
            .unwrap();
        let mut world = RigidWorld::from_scene(scene).unwrap();
        world.solver.substeps = 1;
        world.set_reaction_readout(capture).unwrap();
        world.step(0.01).unwrap();
        let pair = &world.contact_pairs()[0];
        let cap = pair.points[0].normal_force * 0.001;
        let expected = 0.005 - cap / 2.0_f64.sqrt();
        let v = world.scene.body_velocity(b).unwrap().linear;
        assert!(
            (v.x - expected).abs() < 1e-8 && (v.y - expected).abs() < 1e-8,
            "v={v:?} expected={expected}"
        );
        let force = world.distance_reading(j).unwrap().force;
        assert!(
            (force - Vec3::new(0.5, -0.5, 0.)).length() < 1e-6,
            "force={force:?}"
        );
        if capture {
            assert!((world.joint_reaction(j).unwrap().child.linear - force).length() < 1e-10);
        }
    }
}

#[test]
fn distance_between_two_contacts_transfers_friction_and_preserves_endpoint_exchange() {
    let mut trajectories = Vec::new();
    for reverse in [false, true] {
        let mut scene = scene();
        let a = body(&mut scene, Vec3::new(-1., 0., 0.49998));
        let b = body(&mut scene, Vec3::new(1., 0., 0.49998));
        let j = distance(
            &mut scene,
            Some(if reverse { b } else { a }),
            if reverse { a } else { b },
            Vec3::ZERO,
            2.,
        );
        scene
            .set_body_velocity(a, SpatialVector::new(Vec3::X * 0.01, Vec3::ZERO))
            .unwrap();
        let mut world = RigidWorld::from_scene(scene).unwrap();
        world.solver.substeps = 1;
        world.step(0.01).unwrap();
        let velocities = [a, b].map(|id| world.scene.body_velocity(id).unwrap());
        for v in velocities {
            assert!(
                (v.linear.x - 0.003).abs() < 1e-8,
                "reverse={reverse} v={v:?}"
            );
            assert!(
                (v.angular.y - 0.001).abs() < 1e-7,
                "reverse={reverse} v={v:?}"
            );
        }
        let expected = Vec3::X * if reverse { -0.5 } else { 0.5 };
        assert!((world.distance_reading(j).unwrap().force - expected).length() < 1e-6);
        trajectories.push(velocities);
    }
    for i in 0..2 {
        assert!((trajectories[0][i].linear - trajectories[1][i].linear).length() < 1e-10);
        assert!((trajectories[0][i].angular - trajectories[1][i].angular).length() < 1e-10);
    }
}

#[test]
fn mixed_cpu_distance_break_uses_common_reaction_and_continues() {
    let mut scene = scene();
    let b = body(&mut scene, Vec3::new(-1., 1., 0.49998));
    let j = distance(&mut scene, None, b, Vec3::Z * 0.49998, 2.0_f64.sqrt());
    scene
        .set_body_velocity(b, SpatialVector::new(Vec3::Y * 0.01, Vec3::ZERO))
        .unwrap();
    scene
        .set_joint_break_limits(
            j,
            molla_sim::runtime::JointBreakLimits {
                force: Some(0.6),
                torque: None,
            },
        )
        .unwrap();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.solver.substeps = 2;
    world.step(0.02).unwrap();
    assert!(world.scene.joint_broken(j).unwrap());
    assert_eq!(world.joint_break_events().len(), 1);
    let event = &world.joint_break_events()[0];
    assert_eq!(event.sample.substep, 0);
    assert!((event.sample.time_offset - 0.01).abs() < 1e-12);
    assert!((event.sample.child.linear - Vec3::new(0.5, -0.5, 0.)).length() < 1e-6);
    assert_eq!(world.contact_pairs()[0].points.len(), 2);
}

fn contact_load(world: &RigidWorld, body: BodyHandle, center: Vec3) -> SpatialVector {
    let mut load = SpatialVector::ZERO;
    for pair in world.contact_pairs() {
        let sign = if pair.bodies[1] == Some(body) {
            1.
        } else if pair.bodies[0] == Some(body) {
            -1.
        } else {
            continue;
        };
        for point in pair.points {
            let force = point.force_on_b * sign;
            load.linear += force;
            load.angular += (point.point - center).cross(force);
        }
    }
    load
}

#[test]
fn loaded_scalar_contact_stops_release_and_survive_limit_edits() {
    use molla_sim::joint_control::JointAxis;
    let mut scene = scene();
    let center = Vec3::Z * 0.49998;
    let b = body(&mut scene, center);
    let slider = scene
        .insert_joint(JointDesc {
            parent: None,
            child: b,
            kind: JointType::Prismatic,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(center),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_joint_limits(slider, JointAxis::LinX, Some([-0.01, 0.01]))
        .unwrap();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.set_reaction_readout(true).unwrap();
    world.solver.substeps = 1;
    for (phase, (force, bound)) in [(1., 0.01_f64), (-1., -0.01), (1., 0.02)]
        .into_iter()
        .enumerate()
    {
        if phase == 2 {
            world
                .scene
                .set_joint_limits(slider, JointAxis::LinX, Some([-0.02, 0.02]))
                .unwrap();
        }
        world.scene.reset_body_forces(b).unwrap();
        world
            .scene
            .add_body_wrench(b, SpatialVector::new(Vec3::X * force, Vec3::ZERO))
            .unwrap();
        let mut moving = false;
        for frame in 0..100 {
            let before = world.scene.body_velocity(b).unwrap().linear.x;
            world.step(0.005).unwrap();
            let velocity = world.scene.body_velocity(b).unwrap().linear.x;
            moving |= velocity * force > 0.01;
            let contact = contact_load(&world, b, world.scene.body_pose(b).unwrap().position);
            let reaction = world.joint_reaction(slider).unwrap().child;
            let residual =
                (velocity - before) / 0.005 - force - contact.linear.x - reaction.linear.x;
            assert!(
                residual.abs() < 1e-7,
                "phase={phase} frame={frame} residual={residual}"
            );
            assert!(world.scene.body_pose(b).unwrap().position.x.abs() < bound.abs() + 1e-9);
        }
        assert!(moving);
        assert!((world.scene.body_pose(b).unwrap().position.x - bound).abs() < 1e-9);
        assert!(world.scene.body_velocity(b).unwrap().linear.x.abs() < 1e-9);
    }
}

#[test]
fn locked_cone_contact_has_complete_cpu_torque_balance() {
    let mut scene = scene();
    let center = Vec3::Z * 0.49998;
    let b = body(&mut scene, center);
    let ball = scene
        .insert_joint(JointDesc {
            parent: None,
            child: b,
            kind: JointType::Ball,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(center),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_cone_limits(
            ball,
            molla_sim::runtime::SphericalConeLimits {
                angle0: Some(0.),
                angle1: Some(0.),
            },
        )
        .unwrap();
    scene
        .set_body_velocity(b, SpatialVector::new(Vec3::ZERO, Vec3::Y * 0.01))
        .unwrap();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.solver.substeps = 1;
    world.set_reaction_readout(true).unwrap();
    world.step(0.01).unwrap();
    let reaction = world.joint_reaction(ball).unwrap().child;
    let contact = contact_load(&world, b, center);
    assert!((reaction.angular + contact.angular + Vec3::Y).length() < 1e-7);
    assert!(world.scene.body_velocity(b).unwrap().angular.length() < 1e-9);
    assert!((world.cone_reading(ball).unwrap().torque - reaction.angular).length() < 1e-7);
}

#[test]
fn unrelated_sleeping_island_does_not_change_mixed_cpu_response() {
    let mut scene = scene();
    let b = body(&mut scene, Vec3::new(-1., 1., 0.49998));
    distance(&mut scene, None, b, Vec3::Z * 0.49998, 2.0_f64.sqrt());
    let ghost = body(&mut scene, Vec3::new(3., 0., 2.));
    distance(&mut scene, None, ghost, Vec3::new(3., 0., 2.), 0.);
    scene
        .set_body_velocity(b, SpatialVector::new(Vec3::Y * 0.01, Vec3::ZERO))
        .unwrap();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.solver.substeps = 1;
    world.sleep_body(ghost).unwrap();
    let pose = world.scene.body_pose(ghost).unwrap();
    world.step(0.01).unwrap();
    let expected = 0.005 - 0.002 / 2.0_f64.sqrt();
    let v = world.scene.body_velocity(b).unwrap().linear;
    assert!(
        (v.x - expected).abs() < 1e-8 && (v.y - expected).abs() < 1e-8,
        "{v:?}"
    );
    assert!(world.is_sleeping(ghost));
    assert_eq!(world.scene.body_pose(ghost).unwrap(), pose);
}

#[test]
fn incompatible_immovable_mixed_row_rolls_back_and_recovers() {
    let mut scene = scene();
    let center = Vec3::Z * 0.49998;
    let b = body(&mut scene, center);
    scene
        .set_body_kind(b, molla_sim::runtime::BodyKind::Fixed)
        .unwrap();
    let joint = distance(&mut scene, None, b, center, 1.);
    let mut world = RigidWorld::from_scene(scene).unwrap();
    let pose = world.scene.body_pose(b).unwrap();
    assert!(world.step(0.01).is_err());
    assert_eq!(world.completed_steps(), 0);
    assert_eq!(world.scene.body_pose(b).unwrap(), pose);
    assert!(world.distance_reading(joint).is_none());
    world
        .scene
        .set_distance_limits(
            joint,
            DistanceLimits {
                min: Some(0.),
                max: Some(0.),
            },
        )
        .unwrap();
    world.step(0.01).unwrap();
    assert_eq!(world.completed_steps(), 1);
    assert_eq!(world.distance_reading(joint).unwrap().force, Vec3::ZERO);
}
