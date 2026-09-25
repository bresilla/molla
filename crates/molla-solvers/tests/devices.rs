//! Motors, brakes, propellers, belts and connectors in the rigid world,
//! against analytic answers.

use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, BodyKind, ColliderDesc, ColliderGeometry, JointDesc};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::devices::{
    BeltSpec, ConnectorSpec, MotorCommand, MotorLimits, MotorSpec, PropellerSpec,
};
use molla_solvers::rigid_world::RigidWorld;

const DT: f64 = 1.0 / 120.0;

fn body(world: &mut RigidWorld, position: Vec3, mass: f64) -> BodyHandle {
    world
        .scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::from_translation(position)),
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * mass,
            }),
            ..Default::default()
        })
        .unwrap()
}

/// A fixed post at the origin and a 2 kg arm whose centre is 1 m out along
/// +X, hinged at the origin about +Y; gravity along −Z pulls the arm down.
fn arm(gravity: bool) -> (RigidWorld, BodyHandle, molla_sim::runtime::JointHandle) {
    let mut w = RigidWorld::new().unwrap();
    w.scene
        .set_gravity(if gravity { Vec3::new(0.0, 0.0, -9.81) } else { Vec3::ZERO })
        .unwrap();
    let post = body(&mut w, Vec3::ZERO, 1000.0);
    w.scene.set_body_kind(post, BodyKind::Fixed).unwrap();
    let arm = body(&mut w, Vec3::X, 2.0);
    let joint = w
        .scene
        .insert_joint(JointDesc {
            parent: Some(post),
            child: arm,
            kind: JointType::Revolute,
            axis: Vec3::Y,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::from_translation(-Vec3::X),
        })
        .unwrap();
    (w, arm, joint)
}

fn run(w: &mut RigidWorld, seconds: f64) {
    for _ in 0..(seconds / DT).round() as usize {
        w.step(DT).unwrap();
    }
}

#[test]
fn position_control_reaches_the_target_at_the_speed_limit() {
    let (mut w, _, joint) = arm(false);
    let limits = MotorLimits {
        max_velocity: 1.0,
        ..Default::default()
    };
    w.devices
        .motors
        .insert(&w.scene, joint, MotorSpec { limits, max_force: 100.0 })
        .unwrap();
    w.devices.motors.command(joint, MotorCommand::Position(0.5)).unwrap();
    run(&mut w, 0.2);
    let moving = w.devices.motors.reading(joint).unwrap();
    assert!((moving.velocity - 1.0).abs() < 1e-3, "{moving:?}");
    assert!((moving.commanded_velocity - 1.0).abs() < 1e-9);
    run(&mut w, 1.3);
    let settled = w.devices.motors.reading(joint).unwrap();
    assert!((settled.position - 0.5).abs() < 1e-3, "{settled:?}");
    assert!(settled.velocity.abs() < 1e-3 && settled.force.abs() < 1e-3, "{settled:?}");
}

#[test]
fn holding_torque_feedback_carries_the_arm() {
    let (mut w, _, joint) = arm(true);
    w.devices
        .motors
        .insert(&w.scene, joint, MotorSpec { max_force: 100.0, ..Default::default() })
        .unwrap();
    run(&mut w, 1.0);
    let hold = w.devices.motors.reading(joint).unwrap();
    assert!(hold.position.abs() < 0.02, "{hold:?}");
    assert!((hold.force.abs() - 2.0 * 9.81).abs() < 0.1, "m·g·L: {hold:?}");
}

#[test]
fn force_control_spins_the_arm_at_torque_over_inertia() {
    let (mut w, _, joint) = arm(false);
    w.devices
        .motors
        .insert(&w.scene, joint, MotorSpec { max_force: 100.0, ..Default::default() })
        .unwrap();
    w.devices.motors.command(joint, MotorCommand::Force(5.0)).unwrap();
    run(&mut w, 1.0);
    let spin = w.devices.motors.reading(joint).unwrap();
    // I = I_com + m·L² = 2 + 2.
    assert!((spin.velocity - 5.0 / 4.0).abs() < 1e-3, "{spin:?}");
    assert!((spin.force - 5.0).abs() < 1e-9, "{spin:?}");
}

#[test]
fn brake_damps_the_spin_exponentially() {
    let (mut w, arm, joint) = arm(false);
    w.devices.motors.set_brake(&mut w.scene, joint, 4.0).unwrap();
    w.scene
        .set_body_velocity(arm, SpatialVector::new(Vec3::new(0.0, 0.0, -2.0), Vec3::Y * 2.0))
        .unwrap();
    w.step(DT).unwrap();
    let start = w.devices.motors.reading(joint).unwrap();
    assert!((start.brake_force + 4.0 * start.velocity).abs() < 0.05 * start.velocity.abs(), "{start:?}");
    run(&mut w, 1.0);
    let later = w.devices.motors.reading(joint).unwrap();
    // ω(t) = ω₀·e^(−c·t/I) with c = 4, I = 4.
    let expected = start.velocity * (-1.0f64).exp();
    assert!((later.velocity - expected).abs() < 0.01 * start.velocity.abs(), "{later:?} vs {expected}");
    assert_eq!(later.brake_damping, 4.0);
    w.devices.motors.set_brake(&mut w.scene, joint, 0.0).unwrap();
    w.step(DT).unwrap();
    let released = w.devices.motors.reading(joint).unwrap();
    assert_eq!((released.brake_damping, released.brake_force), (0.0, 0.0), "released brakes keep reporting");
    assert!((released.velocity - later.velocity).abs() < 1e-9);
}

#[test]
fn propeller_thrust_and_reaction_torque_drive_a_free_body() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let hull = body(&mut w, Vec3::ZERO, 2.0);
    let id = w
        .devices
        .propellers
        .insert(&w.scene, PropellerSpec {
            body: hull,
            frame: Transform::IDENTITY,
            thrust: [0.01, 0.0],
            torque: [0.001, 0.0],
            limits: MotorLimits::default(),
            max_torque: 10.0,
        })
        .unwrap();
    w.devices.propellers.set_speed(id, 10.0).unwrap();
    run(&mut w, 1.0);
    let reading = w.devices.propellers.reading(id).unwrap();
    assert!((reading.omega - 10.0).abs() < 1e-12 && (reading.thrust - 1.0).abs() < 1e-9, "{reading:?}");
    let v = w.scene.body_velocity(hull).unwrap();
    // T/m·t along the shaft, and −Q/I·t about it.
    assert!((v.linear.x - 0.5).abs() < 1e-6, "{v:?}");
    assert!((v.angular.x + 0.05).abs() < 1e-6, "{v:?}");
}

#[test]
fn a_box_on_a_running_conveyor_rides_at_belt_speed() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::new(0.0, -9.81, 0.0)).unwrap();
    let belt = w
        .scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Cuboid {
            half_extents: Vec3::new(20.0, 0.1, 1.0),
        }))
        .unwrap();
    // Heavy enough for the explicit penalty damping of four contact points.
    let crate_body = body(&mut w, Vec3::new(0.0, 0.3, 0.0), 50.0);
    let mut shape = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::splat(0.2),
    });
    shape.parent = Some(crate_body);
    w.scene.insert_collider(shape).unwrap();
    let id = w
        .devices
        .belts
        .insert(&w.scene, BeltSpec {
            colliders: vec![belt],
            direction: Vec3::X,
            limits: MotorLimits::default(),
        })
        .unwrap();
    w.devices.belts.command(id, MotorCommand::Velocity(1.0)).unwrap();
    run(&mut w, 2.0);
    let v = w.scene.body_velocity(crate_body).unwrap().linear;
    assert!((v.x - 1.0).abs() < 0.01 && v.y.abs() < 0.01 && v.z.abs() < 1e-3, "{v:?}");
    let reading = w.devices.belts.reading(id).unwrap();
    assert!((reading.velocity - 1.0).abs() < 1e-12 && (reading.position - 2.0).abs() < 1e-9, "{reading:?}");
}

#[test]
fn connectors_dock_hold_and_break_over_their_strength() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let dock = w
        .devices
        .connectors
        .insert(&w.scene, ConnectorSpec {
            tensile_strength: Some(50.0),
            ..Default::default()
        })
        .unwrap();
    let pod = body(&mut w, Vec3::new(0.005, 0.002, 0.0), 1.0);
    let plug = w
        .devices
        .connectors
        .insert(&w.scene, ConnectorSpec {
            body: Some(pod),
            frame: Transform::new(Vec3::ZERO, Quat::from_rotation_z(std::f64::consts::PI)),
            ..Default::default()
        })
        .unwrap();
    w.step(DT).unwrap();
    assert_eq!(w.devices.connectors.reading(dock).unwrap().presence, Some(plug));
    assert_eq!(w.devices.connectors.reading(plug).unwrap().presence, Some(dock));
    assert_eq!(w.devices.connectors.reading(dock).unwrap().linked, None);
    w.devices.connectors.lock(dock).unwrap();
    run(&mut w, 0.5);
    assert_eq!(w.devices.connectors.reading(dock).unwrap().linked, Some(plug));
    assert!(w.scene.body_pose(pod).unwrap().position.length() < 1e-3, "snapped flush");

    w.scene.add_body_wrench(pod, SpatialVector::new(Vec3::X * 30.0, Vec3::ZERO)).unwrap();
    run(&mut w, 0.5);
    let held = w.devices.connectors.reading(dock).unwrap();
    assert_eq!(held.linked, Some(plug));
    assert!((held.tensile - 30.0).abs() < 1.0 && held.shear < 1.0, "{held:?}");

    w.scene.add_body_wrench(pod, SpatialVector::new(Vec3::X * 40.0, Vec3::ZERO)).unwrap();
    run(&mut w, 0.5);
    let broken = w.devices.connectors.reading(dock).unwrap();
    assert_eq!(broken.linked, None, "{broken:?}");
    assert!(w.scene.body_pose(pod).unwrap().position.x > 0.5, "pulled away");
}

#[test]
fn unlocking_releases_an_auto_locked_link() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let dock = w
        .devices
        .connectors
        .insert(&w.scene, ConnectorSpec {
            auto_lock: true,
            ..Default::default()
        })
        .unwrap();
    let pod = body(&mut w, Vec3::new(0.5, 0.0, 0.0), 1.0);
    let plug = w
        .devices
        .connectors
        .insert(&w.scene, ConnectorSpec {
            body: Some(pod),
            frame: Transform::new(Vec3::ZERO, Quat::from_rotation_z(std::f64::consts::PI)),
            ..Default::default()
        })
        .unwrap();
    w.devices.connectors.lock(dock).unwrap();
    w.scene
        .set_body_velocity(pod, SpatialVector::new(Vec3::X * -1.0, Vec3::ZERO))
        .unwrap();
    run(&mut w, 1.0);
    assert_eq!(w.devices.connectors.reading(dock).unwrap().linked, Some(plug), "auto-locked on arrival");
    assert!(w.scene.body_velocity(pod).unwrap().linear.length() < 1e-2);
    w.devices.connectors.unlock(dock).unwrap();
    w.scene.add_body_wrench(pod, SpatialVector::new(Vec3::X, Vec3::ZERO)).unwrap();
    run(&mut w, 0.5);
    assert_eq!(w.devices.connectors.reading(dock).unwrap().linked, None);
    assert!(w.scene.body_velocity(pod).unwrap().linear.x > 0.4);
}
