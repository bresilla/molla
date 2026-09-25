//! Motors, brakes, propellers, belts, connectors, copters and the air in
//! the rigid world, against analytic answers.

use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, BodyHandle, BodyKind, ColliderDesc, ColliderGeometry, JointDesc};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_solvers::devices::{
    AIR_DENSITY, BeltSpec, ConnectorSpec, CopterCommand, CopterId, CopterLimits, CopterSpec, DragSpec,
    MotorCommand, MotorLimits, MotorSpec, PropellerSpec,
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

/// A 25 kg quadcopter in X layout 10 m up (Z up): diagonal rotor pairs spin
/// opposite ways, hovering near 151 rad/s.
fn quad() -> (RigidWorld, BodyHandle, CopterId, Vec<molla_solvers::devices::PropellerId>) {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::new(0.0, 0.0, -9.81)).unwrap();
    let hull = w
        .scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(Transform::from_translation(Vec3::Z * 10.0)),
            additional_mass: Some(MassProperties {
                mass: 25.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(Vec3::new(2.0, 2.0, 3.5)),
            }),
            ..Default::default()
        })
        .unwrap();
    let shaft_up = Quat::from_rotation_y(-std::f64::consts::FRAC_PI_2);
    let rotors = [(1.0, 1.0, 1.0), (1.0, -1.0, -1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, -1.0)]
        .map(|(x, y, spin): (f64, f64, f64)| {
            w.devices
                .propellers
                .insert(&w.scene, PropellerSpec {
                    body: hull,
                    frame: Transform::new(Vec3::new(0.64 * x, 0.64 * y, 0.2), shaft_up),
                    thrust: [2.7e-3 * spin, 0.0],
                    torque: [8e-5, 0.0],
                    limits: MotorLimits {
                        max_velocity: 300.0,
                        ..Default::default()
                    },
                    max_torque: f64::INFINITY,
                })
                .unwrap()
        })
        .to_vec();
    let spec = CopterSpec {
        body: hull,
        frame: Transform::IDENTITY,
        propellers: rotors.clone(),
        limits: CopterLimits::default(),
    };
    let copter = w.devices.copters.insert(&w.scene, &w.devices.propellers, spec).unwrap();
    (w, hull, copter, rotors)
}

fn velocity(forward: f64, left: f64, up: f64, yaw_rate: f64) -> CopterCommand {
    CopterCommand::Velocity { forward, left, up, yaw_rate }
}

#[test]
fn a_quad_hovers_climbs_flies_and_yaws() {
    let (mut w, hull, flight, rotors) = quad();
    w.devices.copters.command(flight, velocity(0.0, 0.0, 0.0, 0.0)).unwrap();
    run(&mut w, 5.0);
    let hover = w.devices.copters.reading(flight).unwrap();
    let z = w.scene.body_pose(hull).unwrap().position.z;
    assert!((z - 10.0).abs() < 0.05 && hover.roll.abs() < 1e-3 && hover.pitch.abs() < 1e-3, "{z} {hover:?}");
    assert!((hover.thrust - 25.0 * 9.81).abs() < 0.5 && !hover.saturated, "{hover:?}");
    let omega = w.devices.propellers.reading(rotors[0]).unwrap().omega;
    assert!((omega - (25.0 * 9.81 / 4.0 / 2.7e-3f64).sqrt()).abs() < 1.0, "hover rotor speed {omega}");
    assert!((w.devices.propellers.reading(rotors[1]).unwrap().omega + omega).abs() < 1.0, "counter-rotating");

    w.devices.copters.command(flight, velocity(0.0, 0.0, 1.0, 0.0)).unwrap();
    run(&mut w, 4.0);
    let climb = w.devices.copters.reading(flight).unwrap();
    assert!((climb.velocity[2] - 1.0).abs() < 0.02, "{climb:?}");

    w.devices.copters.command(flight, velocity(3.0, 0.0, 0.0, 0.0)).unwrap();
    run(&mut w, 1.0);
    assert!(w.devices.copters.reading(flight).unwrap().pitch > 0.05, "noses down to speed up");
    run(&mut w, 5.0);
    let cruise = w.devices.copters.reading(flight).unwrap();
    assert!((cruise.velocity[0] - 3.0).abs() < 0.05 && cruise.velocity[1].abs() < 0.05, "{cruise:?}");
    assert!(cruise.velocity[2].abs() < 0.05, "{cruise:?}");

    w.devices.copters.command(flight, velocity(0.0, 0.0, 0.0, 0.5)).unwrap();
    run(&mut w, 3.0);
    let turn = w.devices.copters.reading(flight).unwrap();
    assert!((turn.yaw_rate - 0.5).abs() < 0.02 && turn.velocity[2].abs() < 0.05, "{turn:?}");
}

#[test]
fn attitude_mode_tilts_and_off_drops() {
    let (mut w, hull, flight, _) = quad();
    let command = CopterCommand::Attitude { roll: 0.2, pitch: 0.0, yaw_rate: 0.0, climb: 0.0 };
    w.devices.copters.command(flight, command).unwrap();
    run(&mut w, 2.0);
    let banked = w.devices.copters.reading(flight).unwrap();
    assert!((banked.roll - 0.2).abs() < 0.01 && banked.pitch.abs() < 0.01, "{banked:?}");
    assert!(banked.velocity[1] < -1.0 && banked.velocity[2].abs() < 0.1, "slides right, holds height: {banked:?}");
    w.devices.copters.command(flight, CopterCommand::Off).unwrap();
    let before = w.scene.body_pose(hull).unwrap().position.z;
    run(&mut w, 1.0);
    let after = w.scene.body_pose(hull).unwrap().position.z;
    assert!((before - after - 0.5 * 9.81).abs() < 0.3, "free fall: {before} -> {after}");
}

/// A 5 m/s wind along +X on a quad with 0.4 m² of side drag: it holds its
/// place by leaning its thrust against the drag, nose up by
/// `atan(½·ρ·CdA·w² / (m·g))`.
#[test]
fn a_quad_holds_station_in_a_steady_wind() {
    let (mut w, hull, flight, _) = quad();
    let drag = w
        .devices
        .aero
        .insert(&w.scene, DragSpec {
            body: hull,
            frame: Transform::IDENTITY,
            area: Vec3::new(0.4, 0.4, 0.8),
            density: AIR_DENSITY,
        })
        .unwrap();
    w.devices.aero.set_wind(Vec3::X * 5.0).unwrap();
    w.devices.copters.command(flight, velocity(0.0, 0.0, 0.0, 0.0)).unwrap();
    let start = w.scene.body_pose(hull).unwrap().position;
    run(&mut w, 20.0);
    let held = w.devices.copters.reading(flight).unwrap();
    let load = 0.5 * AIR_DENSITY * 0.4 * 25.0;
    assert!(held.velocity.iter().all(|v| v.abs() < 0.01), "{held:?}");
    assert!((held.pitch + (load / (25.0 * 9.81)).atan()).abs() < 2e-3, "leans into the wind: {held:?}");
    let pushed = w.devices.aero.reading(drag).unwrap();
    assert!((pushed.force.x - load).abs() < 0.05 && (pushed.airspeed.x + 5.0).abs() < 0.01, "{pushed:?}");
    let drift = (w.scene.body_pose(hull).unwrap().position - start).length();
    assert!(drift < 1.0, "the integral brings it back: drifted {drift} m");
}

/// A 1 kg body with 1 m² of drag, released at rest in a 3 m/s wind without
/// gravity, follows `v(t) = w − 1/(1/w + k·t)` with `k = ½·ρ·CdA/m`.
#[test]
fn a_free_body_is_blown_along_by_quadratic_drag() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let puff = body(&mut w, Vec3::ZERO, 1.0);
    w.devices
        .aero
        .insert(&w.scene, DragSpec {
            body: puff,
            frame: Transform::IDENTITY,
            area: Vec3::ONE,
            density: AIR_DENSITY,
        })
        .unwrap();
    w.devices.aero.set_wind(Vec3::X * 3.0).unwrap();
    run(&mut w, 4.0);
    let k = 0.5 * AIR_DENSITY;
    let expected = 3.0 - 1.0 / (1.0 / 3.0 + k * 4.0);
    let v = w.scene.body_velocity(puff).unwrap().linear;
    assert!((v.x - expected).abs() < 5e-3 && v.y.abs() < 1e-12, "{v:?} vs {expected}");
    w.devices.aero.set_body_wind(puff, Some(Vec3::ZERO)).unwrap();
    let before = w.scene.body_velocity(puff).unwrap().linear.x;
    run(&mut w, 1.0);
    let after = w.scene.body_velocity(puff).unwrap().linear.x;
    assert!(after < before, "its own still air slows it: {before} -> {after}");
}

/// A tipper body, its tailgate and a gate on the tailgate, each held by a
/// strong ram exactly at a joint stop under gravity, as a parked trailer
/// is: the stops and the motors share the load step after step.
#[test]
fn strong_motors_holding_a_chain_at_its_stops_keep_stepping() {
    use molla_sim::joint_control::JointAxis;
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::new(0.0, 0.0, -9.81)).unwrap();
    let chassis = body(&mut w, Vec3::ZERO, 2500.0);
    w.scene.set_body_kind(chassis, BodyKind::Fixed).unwrap();
    let (tipper, gate, slide) =
        (body(&mut w, Vec3::new(0.0, 2.0, 1.5), 2000.0), body(&mut w, Vec3::new(0.0, 4.0, 1.5), 450.0), body(&mut w, Vec3::new(0.0, 4.2, 1.0), 40.0));
    let hinge = |w: &mut RigidWorld, parent, child, at: Vec3, kind, limits: [f64; 2]| {
        let joint = w
            .scene
            .insert_joint(JointDesc {
                parent: Some(parent),
                child,
                kind,
                axis: Vec3::X,
                frame_parent: Transform::from_translation(at),
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        let axis = if kind == JointType::Prismatic { JointAxis::LinX } else { JointAxis::AngX };
        w.scene.set_joint_limits(joint, axis, Some(limits)).unwrap();
        w.devices.motors.insert(&w.scene, joint, MotorSpec { max_force: 1.0e6, ..Default::default() }).unwrap();
        w.devices.motors.command(joint, MotorCommand::Position(limits[if limits[0] == 0.0 { 0 } else { 1 }])).unwrap();
        joint
    };
    hinge(&mut w, chassis, tipper, Vec3::new(0.0, 2.0, 1.5), JointType::Revolute, [-0.87, 0.0]);
    hinge(&mut w, tipper, gate, Vec3::new(0.0, 2.0, 0.0), JointType::Revolute, [0.0, 0.87]);
    hinge(&mut w, gate, slide, Vec3::new(0.0, 0.2, -0.5), JointType::Prismatic, [0.0, 0.34]);
    run(&mut w, 60.0);
    for body in [tipper, gate, slide] {
        assert!(w.scene.body_velocity(body).unwrap().linear.length() < 1e-3);
    }
}

#[test]
fn a_motor_lifts_a_heavy_load_hung_on_its_arm_through_a_soft_joint() {
    use molla_sim::joint_control::{JointAxes, JointSoftness};
    let (mut w, arm, joint) = arm(true);
    let limits = MotorLimits { max_velocity: 1.0, ..Default::default() };
    w.devices.motors.insert(&w.scene, joint, MotorSpec { limits, max_force: 1.0e5 }).unwrap();
    run(&mut w, 0.5);
    // 500 kg two metres out: far beyond what the arm's own inertia moves in a step.
    let load = body(&mut w, Vec3::new(2.0, 0.0, 0.0), 500.0);
    let hitch = w
        .scene
        .insert_joint(JointDesc {
            parent: Some(arm),
            child: load,
            kind: JointType::D6,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(Vec3::X),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    w.scene.set_joint_locked_axes(hitch, JointAxes::ALL).unwrap();
    w.scene
        .set_joint_softness(hitch, Some(JointSoftness { natural_frequency: 8.0, damping_ratio: 1.0 }))
        .unwrap();
    w.devices.motors.command(joint, MotorCommand::Position(-0.5)).unwrap();
    run(&mut w, 2.0);
    let reading = w.devices.motors.reading(joint).unwrap();
    assert!((reading.position + 0.5).abs() < 0.02, "{reading:?}");
    assert!(reading.force.abs() > 500.0 * 9.81 * 1.5, "{reading:?}");
}
