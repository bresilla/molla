//! Zero-G and spacecraft physics verification suite:
//! - RCS thrusters (impulse, lever-arm torque, valve lag, MIB, Tsiolkovsky depletion)
//! - Reaction wheels (exact angular momentum conservation, 3D gyroscopic coupling, saturation)
//! - Orbital dynamics (Clohessy–Wiltshire 2:1 ellipse, cross-track harmonic, gravity gradient,
//!   central Newtonian orbits, J2 nodal regression)
//! - 6-DOF Spacecraft GNC controller (pose hold, kill-rot, thruster-fault recovery, hybrid
//!   wheel+RCS allocation, wheel desaturation)
//! - End-to-end space station docking + zero-G free-floating robotic arm momentum conservation.

use std::f64::consts::{FRAC_PI_2, PI, TAU};

use molla_math::{GRAVITY, Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, JointType, MassProperties};
use molla_sim::runtime::{BodyDesc, BodyHandle, JointDesc};
use molla_solvers::devices::{
    ConnectorSpec, EARTH_EQUATORIAL_RADIUS, EARTH_MU, GimbalSpec, J2Perturbation, MotorCommand,
    MotorLimits, MotorSpec, OrbitalField, ReactionWheelCommand, ReactionWheelSpec,
    SpacecraftCommand, SpacecraftLimits, SpacecraftSpec, ThrusterId, ThrusterSpec,
};
use molla_solvers::rigid_world::RigidWorld;

const DT: f64 = 1.0 / 120.0;

fn run(world: &mut RigidWorld, seconds: f64) {
    let steps = (seconds / DT).round() as usize;
    for _ in 0..steps {
        world.step(DT).unwrap();
    }
}

fn free_body_pose(world: &mut RigidWorld, pose: Transform, mass: f64, inertia: Vec3) -> BodyHandle {
    world
        .scene
        .insert_body(BodyDesc {
            params: BodyParams::new().with_pose(pose),
            additional_mass: Some(MassProperties {
                mass,
                local_com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(inertia),
            }),
            ..Default::default()
        })
        .unwrap()
}

fn free_body(world: &mut RigidWorld, pos: Vec3, mass: f64, inertia: Vec3) -> BodyHandle {
    free_body_pose(world, Transform::from_translation(pos), mass, inertia)
}

/// Mounts a symmetric 12-thruster 6-DOF RCS pod on `body`:
/// 2 opposing thrusters per face offset by `arm` so all 3 forces and 3 torques
/// have independent bidirectional authority.
fn mount_12_thruster_rcs(
    world: &mut RigidWorld,
    body: BodyHandle,
    arm: f64,
    max_thrust: f64,
    isp: Option<f64>,
    propellant: f64,
) -> Vec<ThrusterId> {
    let configs: &[(Vec3, Quat)] = &[
        // +/-X thrusters offset in +/-Y (gives +/-Fx and +/-Tz)
        (Vec3::new(0.0, arm, 0.0), Quat::IDENTITY),
        (Vec3::new(0.0, -arm, 0.0), Quat::IDENTITY),
        (Vec3::new(0.0, arm, 0.0), Quat::from_rotation_z(PI)),
        (Vec3::new(0.0, -arm, 0.0), Quat::from_rotation_z(PI)),
        // +/-Y thrusters offset in +/-Z (gives +/-Fy and +/-Tx)
        (Vec3::new(0.0, 0.0, arm), Quat::from_rotation_z(FRAC_PI_2)),
        (Vec3::new(0.0, 0.0, -arm), Quat::from_rotation_z(FRAC_PI_2)),
        (Vec3::new(0.0, 0.0, arm), Quat::from_rotation_z(-FRAC_PI_2)),
        (Vec3::new(0.0, 0.0, -arm), Quat::from_rotation_z(-FRAC_PI_2)),
        // +/-Z thrusters offset in +/-X (gives +/-Fz and +/-Ty)
        (Vec3::new(arm, 0.0, 0.0), Quat::from_rotation_y(-FRAC_PI_2)),
        (Vec3::new(-arm, 0.0, 0.0), Quat::from_rotation_y(-FRAC_PI_2)),
        (Vec3::new(arm, 0.0, 0.0), Quat::from_rotation_y(FRAC_PI_2)),
        (Vec3::new(-arm, 0.0, 0.0), Quat::from_rotation_y(FRAC_PI_2)),
    ];
    configs
        .iter()
        .map(|&(pos, rot)| {
            world
                .devices
                .thrusters
                .insert(
                    &world.scene,
                    ThrusterSpec {
                        body,
                        frame: Transform::new(pos, rot),
                        max_thrust,
                        specific_impulse: isp,
                        propellant_mass: propellant,
                        deplete_body_mass: false,
                        min_impulse_bit: 0.0,
                        response_time: 0.0,
                    },
                )
                .unwrap()
        })
        .collect()
}

// ─── 1. RCS Thrusters ────────────────────────────────────────────────────

#[test]
fn thruster_linear_and_lever_arm_torque_match_newton_euler() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let craft = free_body(&mut w, Vec3::ZERO, 10.0, Vec3::splat(4.0));

    // On-COM thruster along +X (20 N) -> a = 2 m/s².
    let main = w
        .devices
        .thrusters
        .insert(&w.scene, ThrusterSpec::ideal(craft, Transform::IDENTITY, 20.0))
        .unwrap();
    w.devices.thrusters.set_throttle(main, 0.5).unwrap(); // 10 N
    run(&mut w, 2.0);
    let vel = w.scene.body_velocity(craft).unwrap();
    let pose = w.scene.body_pose(craft).unwrap();
    assert!((vel.linear.x - 2.0).abs() < 1e-6, "v = {vel:?}");
    // Symplectic Euler advances position with post-step velocity, adding
    // +0.5 * a * sub_dt * T ≈ 0.00104 m over 2 s at sub_dt = DT/8.
    assert!((pose.position.x - 2.0).abs() < 2e-3, "p = {pose:?}");
    assert!(vel.angular.length() < 1e-12);
    let r = w.devices.thrusters.reading(main).unwrap();
    assert!((r.thrust - 10.0).abs() < 1e-9 && (r.impulse_total - 20.0).abs() < 1e-6);

    // Off-COM couple: +5 N at +Y=1 and -5 N at -Y=1 -> pure -Z torque = -10 N·m.
    w.devices.thrusters.set_throttle(main, 0.0).unwrap();
    let top = w
        .devices
        .thrusters
        .insert(
            &w.scene,
            ThrusterSpec::ideal(craft, Transform::from_translation(Vec3::Y), 5.0),
        )
        .unwrap();
    let bot = w
        .devices
        .thrusters
        .insert(
            &w.scene,
            ThrusterSpec::ideal(
                craft,
                Transform::new(-Vec3::Y, Quat::from_rotation_z(PI)),
                5.0,
            ),
        )
        .unwrap();
    w.devices.thrusters.set_throttle(top, 1.0).unwrap();
    w.devices.thrusters.set_throttle(bot, 1.0).unwrap();
    run(&mut w, 1.0);
    let vel2 = w.scene.body_velocity(craft).unwrap();
    // Net linear force of the couple is 0, so linear velocity stays 2.0 m/s along +X.
    assert!((vel2.linear.x - 2.0).abs() < 1e-5, "{vel2:?}");
    // Angular velocity about Z: τ/I * t = -10 / 4 * 1 = -2.5 rad/s.
    assert!((vel2.angular.z + 2.5).abs() < 1e-5, "{vel2:?}");
}

#[test]
fn thruster_valve_lag_and_minimum_impulse_bit() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let craft = free_body(&mut w, Vec3::ZERO, 2.0, Vec3::splat(1.0));

    // Valve lag τ = 0.2 s, F_max = 10 N: after T = 1 s, analytical impulse is
    // ∫_0^T F_max (1 - e^{-t/τ}) dt = F_max * (T - τ * (1 - e^{-T/τ})).
    let tau = 0.2;
    let lag = w
        .devices
        .thrusters
        .insert(
            &w.scene,
            ThrusterSpec {
                response_time: tau,
                ..ThrusterSpec::ideal(craft, Transform::IDENTITY, 10.0)
            },
        )
        .unwrap();
    w.devices.thrusters.set_throttle(lag, 1.0).unwrap();
    run(&mut w, 1.0);
    let expected_impulse = 10.0 * (1.0 - tau * (1.0 - (-1.0 / tau).exp()));
    let got_v = w.scene.body_velocity(craft).unwrap().linear.x;
    assert!(
        (got_v - expected_impulse / 2.0).abs() < 1e-3,
        "lag velocity {got_v} vs {}",
        expected_impulse / 2.0
    );

    // MIB = 0.5 N·s on a 10 N thruster -> minimum burn duration 0.05 s.
    let craft2 = free_body(&mut w, Vec3::Y * 5.0, 1.0, Vec3::splat(1.0));
    let mib = w
        .devices
        .thrusters
        .insert(
            &w.scene,
            ThrusterSpec {
                min_impulse_bit: 0.5,
                ..ThrusterSpec::ideal(craft2, Transform::IDENTITY, 10.0)
            },
        )
        .unwrap();
    // Command a tiny 0.005 s pulse — MIB must stretch it to deliver 0.5 N·s!
    w.devices.thrusters.fire_pulse(mib, 1.0, 0.005).unwrap();
    run(&mut w, 0.2);
    let v2 = w.scene.body_velocity(craft2).unwrap().linear.x;
    let r2 = w.devices.thrusters.reading(mib).unwrap();
    assert!((v2 - 0.5).abs() < 1e-4, "MIB velocity {v2}");
    assert!((r2.impulse_total - 0.5).abs() < 1e-4, "{r2:?}");
    assert!(r2.thrust.abs() < 1e-12, "turned off cleanly after MIB");
}

#[test]
fn thruster_propellant_depletion_matches_tsiolkovsky_rocket_equation() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let m0 = 100.0;
    let fuel = 40.0;
    let mf = m0 - fuel;
    let isp = 300.0;
    let rocket = free_body(&mut w, Vec3::ZERO, m0, Vec3::splat(50.0));
    let engine = w
        .devices
        .thrusters
        .insert(
            &w.scene,
            ThrusterSpec {
                body: rocket,
                frame: Transform::IDENTITY,
                max_thrust: 5000.0,
                specific_impulse: Some(isp),
                propellant_mass: fuel,
                deplete_body_mass: true,
                min_impulse_bit: 0.0,
                response_time: 0.0,
            },
        )
        .unwrap();
    w.devices.thrusters.set_throttle(engine, 1.0).unwrap();
    // Burn duration is fuel * Isp * g0 / F ≈ 40 * 300 * 9.80665 / 5000 ≈ 23.54 s.
    run(&mut w, 25.0);

    let r = w.devices.thrusters.reading(engine).unwrap();
    assert!(r.starved, "engine should report starved after fuel exhaustion");
    assert_eq!(r.propellant_remaining, 0.0);
    assert_eq!(r.thrust, 0.0);

    let final_mass = w.scene.body_mass_properties(rocket).unwrap().mass;
    assert!(
        (final_mass - mf).abs() < 1e-9,
        "final mass {final_mass} vs {mf}"
    );

    let expected_dv = isp * GRAVITY * (m0 / mf).ln();
    let actual_dv = w.scene.body_velocity(rocket).unwrap().linear.x;
    let rel_err = ((actual_dv - expected_dv) / expected_dv).abs();
    assert!(
        rel_err < 2e-3,
        "Tsiolkovsky Δv mismatch: actual={actual_dv:.4}, expected={expected_dv:.4}, rel_err={rel_err:.5}"
    );
}

// ─── 2. Reaction Wheels & 3D Gyroscopic Coupling ─────────────────────────

#[test]
fn reaction_wheel_conserves_angular_momentum_and_respects_saturation() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let i_body = Vec3::new(2.0, 3.0, 4.0);
    let sat = free_body(&mut w, Vec3::ZERO, 20.0, i_body);
    let rw = w
        .devices
        .reaction_wheels
        .insert(
            &w.scene,
            ReactionWheelSpec {
                body: sat,
                axis: Vec3::Z,
                inertia: 0.05,
                max_torque: 0.5,
                max_speed: 20.0, // saturates at t = 0.05 * 20 / 0.5 = 2.0 s
                initial_speed: 0.0,
                gimbal: None,
            },
        )
        .unwrap();
    w.devices.reaction_wheels.set_torque(rw, 0.5).unwrap();
    run(&mut w, 3.0);

    let read = w.devices.reaction_wheels.reading(rw).unwrap();
    assert!(read.saturated, "wheel should saturate at 20 rad/s");
    assert!((read.speed - 20.0).abs() < 1e-9, "{read:?}");
    assert!((read.applied_torque).abs() < 1e-9, "zero torque once saturated");

    // Exact total angular momentum conservation: I_z * ω_z + I_w * Ω = 0.
    let omega = w.scene.body_velocity(sat).unwrap().angular;
    let total_hz = i_body.z * omega.z + read.momentum;
    assert!(
        total_hz.abs() < 1e-8 && omega.x.abs() < 1e-10 && omega.y.abs() < 1e-10,
        "total H_z = {total_hz}, omega = {omega:?}, wheel = {read:?}"
    );
}

#[test]
fn spinning_reaction_wheel_produces_3d_gyroscopic_precession_conserving_total_h() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let i_body = Vec3::splat(2.0);
    let sat = free_body(&mut w, Vec3::ZERO, 10.0, i_body);
    let wheel_inertia = 0.1;
    let wheel_speed = 100.0; // h_w = 10 N·m·s along +X
    let rw = w
        .devices
        .reaction_wheels
        .insert(
            &w.scene,
            ReactionWheelSpec {
                body: sat,
                axis: Vec3::X,
                inertia: wheel_inertia,
                max_torque: 1.0,
                max_speed: 200.0,
                initial_speed: wheel_speed,
                gimbal: None,
            },
        )
        .unwrap();

    // Give spacecraft an initial angular velocity about +Y (perpendicular to wheel).
    w.apply_body_impulse(sat, SpatialVector::new(Vec3::ZERO, Vec3::Y * 4.0))
        .unwrap();
    // Initial total angular momentum in world space: H_0 = (10, 4, 0).
    let h0 = Vec3::new(wheel_inertia * wheel_speed, 4.0, 0.0);

    // Step for 1 second with zero external torque and zero wheel motor torque.
    run(&mut w, 1.0);

    let pose = w.scene.body_pose(sat).unwrap();
    let vel = w.scene.body_velocity(sat).unwrap();
    let read = w.devices.reaction_wheels.reading(rw).unwrap();

    // Gyroscopic torque must have induced precession into Z!
    assert!(
        vel.angular.z.abs() > 0.2,
        "expected gyroscopic precession into Z, got {vel:?}"
    );
    assert!(read.gyroscopic_torque.length() > 0.1, "{read:?}");

    // Total world angular momentum H = I_world * ω + R * (I_w * Ω * x̂) must equal H_0!
    let r = Mat3::from_quat(pose.rotation);
    let i_world = r * Mat3::from_diagonal(i_body) * r.transpose();
    let h_now = i_world * vel.angular + pose.rotation * (Vec3::X * read.momentum);
    assert!(
        (h_now - h0).length() < 5e-3,
        "total angular momentum drift: h_now={h_now:?} vs h0={h0:?}"
    );
}

#[test]
fn single_gimbal_cmg_produces_torque_amplification_and_conserves_angular_momentum() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let i_body = Vec3::splat(10.0);
    let sat = free_body(&mut w, Vec3::ZERO, 50.0, i_body);

    // Spinning flywheel (h = 0.2 * 100 = 20 N·m·s along +X) gimballed about +Z.
    // Slewing the gimbal at γ̇ = +0.5 rad/s rotates h from +X toward +Y,
    // exerting τ_cmg = -γ̇ (ẑ × h) = -0.5 * 20 ŷ = -10 N·m on the spacecraft!
    let cmg = w
        .devices
        .reaction_wheels
        .insert(
            &w.scene,
            ReactionWheelSpec {
                body: sat,
                axis: Vec3::X,
                inertia: 0.2,
                max_torque: 1.0,
                max_speed: 200.0,
                initial_speed: 100.0,
                gimbal: Some(GimbalSpec {
                    axis: Vec3::Z,
                    max_rate: 1.0,
                    initial_angle: 0.0,
                }),
            },
        )
        .unwrap();
    let h0 = Vec3::new(20.0, 0.0, 0.0);

    w.devices.reaction_wheels.set_gimbal_rate(cmg, 0.5).unwrap();
    // On the first step (ω ≈ 0), verify exact CMG torque amplification:
    // τ_cmg = -γ̇ (ĝ × h_w) = -0.5 * (ẑ × 20 x̂) = -10.0 ŷ N·m.
    w.step(DT).unwrap();
    let first_tau = w.devices.reaction_wheels.reading(cmg).unwrap().world_torque;
    assert!(
        (first_tau.y + 10.0).abs() < 0.05 && first_tau.x.abs() < 0.1,
        "initial CMG torque mismatch: {first_tau:?}"
    );

    run(&mut w, 1.0 - DT);
    let read = w.devices.reaction_wheels.reading(cmg).unwrap();
    assert!((read.gimbal_angle - 0.5).abs() < 1e-6, "{read:?}");

    let pose = w.scene.body_pose(sat).unwrap();
    let vel = w.scene.body_velocity(sat).unwrap();
    // Strong coupled 3D spacecraft slew generated purely by gimballing!
    assert!(
        vel.angular.y < -0.3 && vel.angular.length() > 0.7,
        "expected strong CMG slew: {vel:?}"
    );

    // Exact total angular momentum conservation: I_world ω + R * (h * spin_axis) == h0!
    let r = Mat3::from_quat(pose.rotation);
    let i_world = r * Mat3::from_diagonal(i_body) * r.transpose();
    let h_now = i_world * vel.angular + pose.rotation * (read.spin_axis * read.momentum);
    assert!(
        (h_now - h0).length() < 5e-3,
        "CMG angular momentum drift: h_now={h_now:?} vs h0={h0:?}"
    );
}

// ─── 3. Orbital Dynamics (Clohessy–Wiltshire & Central Gravity) ──────────

#[test]
fn clohessy_wiltshire_2_to_1_football_orbit_and_cross_track_match_analytical() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    // Fast test orbit (n = 0.1 rad/s, period T = 20π ≈ 62.83 s) so we integrate
    // a full orbit accurately in milliseconds.
    let n = 0.1;
    let period = TAU / n;
    w.devices
        .orbital
        .set_field(OrbitalField::ClohessyWiltshire {
            mean_motion: n,
            radial_axis: Vec3::Y,
            along_track_axis: Vec3::X,
            origin: Vec3::ZERO,
            gravity_gradient: false,
        })
        .unwrap();

    // Initial conditions for exact 2:1 closed football ellipse + cross-track cosine:
    // r(t) = r0 cos(nt), v(t) = -2 r0 sin(nt), h(t) = h0 cos(nt).
    let r0 = 10.0;
    let h0 = 4.0;
    let sat = free_body(&mut w, Vec3::new(0.0, r0, h0), 50.0, Vec3::splat(10.0));
    w.apply_body_impulse(
        sat,
        SpatialVector::new(Vec3::new(-2.0 * n * r0 * 50.0, 0.0, 0.0), Vec3::ZERO),
    )
    .unwrap();

    // Quarter orbit: r = 0, v = -2*r0 = -20, h = 0.
    run(&mut w, period * 0.25);
    let q = w.scene.body_pose(sat).unwrap().position;
    assert!(
        (q.x + 2.0 * r0).abs() < 0.05 && q.y.abs() < 0.05 && q.z.abs() < 0.05,
        "quarter orbit pos {q:?}"
    );

    // Full orbit: returns cleanly to (0, r0, h0).
    run(&mut w, period * 0.75);
    let full = w.scene.body_pose(sat).unwrap().position;
    assert!(
        (full - Vec3::new(0.0, r0, h0)).length() < 0.15,
        "full CW orbit return error: {full:?} vs (0, {r0}, {h0})"
    );
}

#[test]
fn gravity_gradient_stabilizes_slender_satellite_at_analytical_libration_frequency() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let n = 0.5;
    w.devices
        .orbital
        .set_field(OrbitalField::ClohessyWiltshire {
            mean_motion: n,
            radial_axis: Vec3::Y,
            along_track_axis: Vec3::X,
            origin: Vec3::ZERO,
            gravity_gradient: true,
        })
        .unwrap();

    // Slender rod aligned near radial (+Y): I_y = 1 (small), I_x = I_z = 10 (large).
    // Small-angle libration about Z has frequency ω_lib = sqrt(3 n² (I_z - I_y) / I_z).
    let i_par = 1.0;
    let i_perp = 10.0;
    let theta0 = 0.1; // 0.1 rad (~5.7°) tilt about Z
    let sat = free_body_pose(
        &mut w,
        Transform::new(Vec3::ZERO, Quat::from_rotation_z(theta0)),
        10.0,
        Vec3::new(i_perp, i_par, i_perp),
    );

    let omega_lib = (3.0 * n * n * (i_perp - i_par) / i_perp).sqrt();
    let t_quarter = FRAC_PI_2 / omega_lib;
    run(&mut w, t_quarter);
    let angle_q = (w.scene.body_pose(sat).unwrap().rotation * Vec3::Y)
        .x
        .atan2((w.scene.body_pose(sat).unwrap().rotation * Vec3::Y).y);
    // At a quarter period the rod swings right through vertical (angle ≈ 0).
    assert!(angle_q.abs() < 0.015, "quarter libration angle {angle_q}");

    run(&mut w, t_quarter);
    let y_half = w.scene.body_pose(sat).unwrap().rotation * Vec3::Y;
    let theta_half = (-y_half.x).atan2(y_half.y);
    // At a half period it reaches -theta0 on the opposite side.
    assert!(
        (theta_half + theta0).abs() < 0.015,
        "half libration angle {theta_half} vs {}",
        -theta0
    );
}

#[test]
fn central_gravity_circular_orbit_and_j2_nodal_precession() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    // Scaled two-body circular orbit: μ = 1000, r = 10 -> v_circ = 10, T = 2π ≈ 6.283 s.
    let mu: f64 = 1000.0;
    let r0: f64 = 10.0;
    let v_circ = (mu / r0).sqrt();
    let period = TAU * (r0 * r0 * r0 / mu).sqrt();
    w.devices
        .orbital
        .set_field(OrbitalField::CentralGravity {
            center: Vec3::ZERO,
            mu,
            min_radius: 0.5,
            j2: None,
            gravity_gradient: false,
        })
        .unwrap();

    let sat = free_body(&mut w, Vec3::new(r0, 0.0, 0.0), 5.0, Vec3::splat(1.0));
    w.apply_body_impulse(
        sat,
        SpatialVector::new(Vec3::new(0.0, 0.0, v_circ * 5.0), Vec3::ZERO),
    )
    .unwrap();

    run(&mut w, period);
    let p = w.scene.body_pose(sat).unwrap().position;
    let v = w.scene.body_velocity(sat).unwrap().linear;
    assert!((p.length() - r0).abs() < 0.05, "radius drift: {}", p.length());
    assert!((v.length() - v_circ).abs() < 0.05, "speed drift: {}", v.length());
    assert!((p - Vec3::new(r0, 0.0, 0.0)).length() < 0.1, "full orbit closure: {p:?}");

    // J2 oblateness check: inclined prograde orbit about +Y pole precesses
    // westward (node drifts in the -Y rotation direction, so angular momentum
    // vector rotates around +Y with dΩ/dt < 0).
    let mut w2 = RigidWorld::new().unwrap();
    w2.scene.set_gravity(Vec3::ZERO).unwrap();
    w2.devices
        .orbital
        .set_field(OrbitalField::CentralGravity {
            center: Vec3::ZERO,
            mu,
            min_radius: 0.5,
            j2: Some(J2Perturbation {
                j2: 0.02,
                equatorial_radius: 5.0,
                pole_axis: Vec3::Y,
            }),
            gravity_gradient: false,
        })
        .unwrap();
    let inc: f64 = 0.7; // ~40° inclination
    let sat2 = free_body(&mut w2, Vec3::new(r0, 0.0, 0.0), 1.0, Vec3::splat(1.0));
    // Prograde around +Y at (r0, 0, 0) moves toward -Z (since x × (-z) = +y).
    let v0 = Vec3::new(0.0, v_circ * inc.sin(), -v_circ * inc.cos());
    w2.apply_body_impulse(sat2, SpatialVector::new(v0, Vec3::ZERO))
        .unwrap();
    let h_init = Vec3::new(r0, 0.0, 0.0).cross(v0);
    // Right-handed azimuth around +Y (prograde direction) is atan2(-h.x, -h.z).
    let raan_init = (-h_init.x).atan2(-h_init.z);
    let duration = period * 2.0;
    run(&mut w2, duration);
    let p2 = w2.scene.body_pose(sat2).unwrap().position;
    let v2 = w2.scene.body_velocity(sat2).unwrap().linear;
    let h_end = p2.cross(v2);
    let raan_end = (-h_end.x).atan2(-h_end.z);
    // Secular J2 nodal regression: dΩ/dt = -3/2 * J2 * (R_e/r)^2 * n * cos(i).
    let mut delta_raan = raan_end - raan_init;
    while delta_raan > PI {
        delta_raan -= TAU;
    }
    while delta_raan < -PI {
        delta_raan += TAU;
    }
    let n = (mu / (r0 * r0 * r0)).sqrt();
    let expected_raan = -1.5 * 0.02 * (5.0_f64 / r0).powi(2) * n * inc.cos() * duration;
    assert!(
        (delta_raan - expected_raan).abs() < 0.005,
        "J2 nodal regression mismatch: got={delta_raan:.5}, expected={expected_raan:.5}"
    );
}

// ─── 4. 6-DOF Spacecraft GNC Controller ──────────────────────────────────

#[test]
fn spacecraft_6dof_pose_hold_and_kill_rot_with_12_thruster_rcs() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let craft = free_body(&mut w, Vec3::ZERO, 20.0, Vec3::new(4.0, 5.0, 6.0));
    let thrusters = mount_12_thruster_rcs(&mut w, craft, 0.5, 40.0, None, 0.0);
    let gnc = w
        .devices
        .spacecraft
        .insert(
            &w.scene,
            &w.devices.thrusters,
            &w.devices.reaction_wheels,
            SpacecraftSpec {
                body: craft,
                frame: Transform::IDENTITY,
                thrusters,
                reaction_wheels: Vec::new(),
                limits: SpacecraftLimits::default(),
            },
        )
        .unwrap();

    // First give the spacecraft an ugly tumble and arrest it with RateHold(0, 0).
    w.apply_body_impulse(
        craft,
        SpatialVector::new(Vec3::new(20.0, -15.0, 10.0), Vec3::new(3.0, -2.5, 2.0)),
    )
    .unwrap();
    w.devices
        .spacecraft
        .command(
            gnc,
            SpacecraftCommand::RateHold {
                linear_velocity: Vec3::ZERO,
                angular_velocity: Vec3::ZERO,
            },
        )
        .unwrap();
    run(&mut w, 4.0);
    let v_stopped = w.scene.body_velocity(craft).unwrap();
    assert!(
        v_stopped.linear.length() < 0.02 && v_stopped.angular.length() < 0.02,
        "kill-rot failed: {v_stopped:?}"
    );

    // Now command a simultaneous 3D translation + 3D attitude slew.
    let target_pos = Vec3::new(4.0, -2.5, 3.0);
    let target_rot =
        Quat::from_euler(glam::EulerRot::XYZ, 0.4, -0.6, 0.8).normalize();
    w.devices
        .spacecraft
        .command(
            gnc,
            SpacecraftCommand::PoseHold {
                position: target_pos,
                orientation: target_rot,
            },
        )
        .unwrap();
    run(&mut w, 12.0);
    let pose = w.scene.body_pose(craft).unwrap();
    let vel = w.scene.body_velocity(craft).unwrap();
    let pos_err = pose.position.distance(target_pos);
    let rot_err = pose.rotation.angle_between(target_rot);
    assert!(
        pos_err < 0.02 && rot_err < 0.01 && vel.linear.length() < 0.02 && vel.angular.length() < 0.02,
        "6-DOF pose hold error: pos={pos_err:.4}, rot={rot_err:.4}, vel={vel:?}"
    );
}

#[test]
fn hybrid_reaction_wheels_and_rcs_slews_fuel_free_then_desaturates_steadily() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let craft = free_body(&mut w, Vec3::ZERO, 30.0, Vec3::splat(6.0));
    let thrusters = mount_12_thruster_rcs(&mut w, craft, 0.6, 30.0, Some(220.0), 5.0);
    let wheels: Vec<_> = [Vec3::X, Vec3::Y, Vec3::Z]
        .into_iter()
        .map(|axis| {
            w.devices
                .reaction_wheels
                .insert(
                    &w.scene,
                    ReactionWheelSpec {
                        body: craft,
                        axis,
                        inertia: 0.1,
                        max_torque: 15.0,
                        max_speed: 100.0,
                        initial_speed: 0.0,
                        gimbal: None,
                    },
                )
                .unwrap()
        })
        .collect();

    let gnc = w
        .devices
        .spacecraft
        .insert(
            &w.scene,
            &w.devices.thrusters,
            &w.devices.reaction_wheels,
            SpacecraftSpec {
                body: craft,
                frame: Transform::IDENTITY,
                thrusters: thrusters.clone(),
                reaction_wheels: wheels.clone(),
                limits: SpacecraftLimits::default(),
            },
        )
        .unwrap();

    // Pure attitude slew at origin: reaction wheels should do ALL the work with 0 fuel burned!
    let target_rot = Quat::from_rotation_y(0.9);
    w.devices
        .spacecraft
        .command(
            gnc,
            SpacecraftCommand::PoseHold {
                position: Vec3::ZERO,
                orientation: target_rot,
            },
        )
        .unwrap();
    run(&mut w, 6.0);
    let rot_err = w.scene.body_pose(craft).unwrap().rotation.angle_between(target_rot);
    let fuel_burned: f64 = thrusters
        .iter()
        .map(|&id| 5.0 - w.devices.thrusters.reading(id).unwrap().propellant_remaining)
        .sum();
    assert!(rot_err < 0.01, "wheel slew error {rot_err}");
    assert!(
        fuel_burned < 1e-6,
        "expected zero RCS fuel burned during wheel slew, burned {fuel_burned} kg"
    );

    // Apply an external disturbance torque about +Y while PoseHold is active:
    // the Y reaction wheel absorbs the disturbance momentum (~5 N·m·s -> 50 rad/s)
    // while holding attitude. Then clear the disturbance and run DesaturateWheels:
    // the wheel bleeds back toward 0 while RCS holds attitude steady!
    w.scene
        .add_body_wrench(craft, SpatialVector::new(Vec3::ZERO, Vec3::Y * 2.5))
        .unwrap();
    run(&mut w, 2.0);
    w.scene
        .add_body_wrench(craft, SpatialVector::new(Vec3::ZERO, Vec3::Y * -2.5))
        .unwrap();
    // Let PoseHold settle back to target_rot; the wheel retains the 5 N·m·s
    // absorbed from the external disturbance.
    run(&mut w, 5.0);
    let before_speed = w.devices.reaction_wheels.reading(wheels[1]).unwrap().speed;
    assert!(before_speed > 45.0, "wheel retained absorbed momentum: {before_speed}");

    let hold_rot = target_rot;
    w.devices
        .spacecraft
        .command(
            gnc,
            SpacecraftCommand::DesaturateWheels {
                position: Some(Vec3::ZERO),
                orientation: hold_rot,
            },
        )
        .unwrap();
    let mut max_att_err: f64 = 0.0;
    for _ in 0..(8.0 / DT) as usize {
        w.step(DT).unwrap();
        let err = w.scene.body_pose(craft).unwrap().rotation.angle_between(hold_rot);
        max_att_err = max_att_err.max(err);
    }
    let after_speed = w.devices.reaction_wheels.reading(wheels[1]).unwrap().speed;
    assert!(
        after_speed.abs() < 3.0,
        "wheel failed to desaturate: before={before_speed}, after={after_speed}"
    );
    assert!(
        max_att_err < 0.01,
        "attitude disturbed during wheel desaturation: max_err={max_att_err:.4} rad"
    );
}

#[test]
fn rcs_allocator_routes_around_starved_thruster() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    let craft = free_body(&mut w, Vec3::ZERO, 10.0, Vec3::splat(2.0));
    // Two parallel +X thrusters on COM: one empty, one full.
    let dead = w
        .devices
        .thrusters
        .insert(
            &w.scene,
            ThrusterSpec {
                body: craft,
                frame: Transform::IDENTITY,
                max_thrust: 20.0,
                specific_impulse: Some(200.0),
                propellant_mass: 0.0, // already empty!
                deplete_body_mass: false,
                min_impulse_bit: 0.0,
                response_time: 0.0,
            },
        )
        .unwrap();
    let live = w
        .devices
        .thrusters
        .insert(
            &w.scene,
            ThrusterSpec {
                body: craft,
                frame: Transform::IDENTITY,
                max_thrust: 20.0,
                specific_impulse: Some(200.0),
                propellant_mass: 10.0,
                deplete_body_mass: false,
                min_impulse_bit: 0.0,
                response_time: 0.0,
            },
        )
        .unwrap();
    let gnc = w
        .devices
        .spacecraft
        .insert(
            &w.scene,
            &w.devices.thrusters,
            &w.devices.reaction_wheels,
            SpacecraftSpec {
                body: craft,
                frame: Transform::IDENTITY,
                thrusters: vec![dead, live],
                reaction_wheels: Vec::new(),
                limits: SpacecraftLimits::default(),
            },
        )
        .unwrap();
    w.devices
        .spacecraft
        .command(
            gnc,
            SpacecraftCommand::Wrench {
                force: Vec3::X * 15.0,
                torque: Vec3::ZERO,
            },
        )
        .unwrap();
    run(&mut w, 1.0);
    let r_dead = w.devices.thrusters.reading(dead).unwrap();
    let r_live = w.devices.thrusters.reading(live).unwrap();
    assert_eq!(r_dead.throttle, 0.0);
    assert!((r_live.thrust - 15.0).abs() < 1e-4, "{r_live:?}");
    assert!((w.scene.body_velocity(craft).unwrap().linear.x - 1.5).abs() < 1e-4);
}

// ─── 5. End-to-End Space Station Docking & Free-Floating Arm ─────────────

#[test]
fn spacecraft_approaches_and_docks_to_space_station_port_in_zero_g() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    // Fixed station docking port at world origin facing +X.
    let station_port = w
        .devices
        .connectors
        .insert(
            &w.scene,
            ConnectorSpec {
                auto_lock: true,
                distance_tolerance: 0.05,
                axis_tolerance: 0.15,
                rotation_tolerance: 0.15,
                ..Default::default()
            },
        )
        .unwrap();

    // Free-flying visiting vehicle starting 2 m away, slightly off-axis and tilted.
    let vehicle = free_body_pose(
        &mut w,
        Transform::new(Vec3::new(2.0, 0.4, -0.3), Quat::from_rotation_z(0.25)),
        15.0,
        Vec3::splat(3.0),
    );

    let vehicle_port = w
        .devices
        .connectors
        .insert(
            &w.scene,
            ConnectorSpec {
                body: Some(vehicle),
                // Port faces -X in vehicle frame so vehicle at identity faces station +X.
                frame: Transform::new(Vec3::ZERO, Quat::from_rotation_z(PI)),
                auto_lock: true,
                distance_tolerance: 0.05,
                axis_tolerance: 0.15,
                rotation_tolerance: 0.15,
                ..Default::default()
            },
        )
        .unwrap();
    let thrusters = mount_12_thruster_rcs(&mut w, vehicle, 0.5, 30.0, None, 0.0);
    let gnc = w
        .devices
        .spacecraft
        .insert(
            &w.scene,
            &w.devices.thrusters,
            &w.devices.reaction_wheels,
            SpacecraftSpec {
                body: vehicle,
                frame: Transform::IDENTITY,
                thrusters,
                reaction_wheels: Vec::new(),
                limits: SpacecraftLimits::default(),
            },
        )
        .unwrap();

    // Arm the station port so it auto-locks as soon as the vehicle lines up.
    w.devices.connectors.lock(station_port).unwrap();

    // Fly into the docking port and lock!
    w.devices
        .spacecraft
        .command(
            gnc,
            SpacecraftCommand::PoseHold {
                position: Vec3::ZERO,
                orientation: Quat::IDENTITY,
            },
        )
        .unwrap();
    run(&mut w, 10.0);

    let dock_reading = w.devices.connectors.reading(station_port).unwrap();
    assert_eq!(
        dock_reading.linked,
        Some(vehicle_port),
        "vehicle should auto-dock to station port: pose={:?}, dock={dock_reading:?}",
        w.scene.body_pose(vehicle).unwrap()
    );
}

#[test]
fn free_floating_spacecraft_arm_conserves_linear_and_angular_momentum_in_zero_g() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let base = free_body(&mut w, Vec3::ZERO, 20.0, Vec3::splat(4.0));
    let arm = free_body(&mut w, Vec3::new(1.0, 0.0, 0.0), 4.0, Vec3::splat(0.5));
    let shoulder = w
        .scene
        .insert_joint(JointDesc {
            parent: Some(base),
            child: arm,
            kind: JointType::Revolute,
            axis: Vec3::Z,
            frame_parent: Transform::from_translation(Vec3::new(0.5, 0.0, 0.0)),
            frame_child: Transform::from_translation(Vec3::new(-0.5, 0.0, 0.0)),
        })
        .unwrap();

    w.devices
        .motors
        .insert(
            &w.scene,
            shoulder,
            MotorSpec {
                limits: MotorLimits {
                    max_velocity: 2.0,
                    ..Default::default()
                },
                max_force: 10.0,
            },
        )
        .unwrap();
    w.devices
        .motors
        .command(shoulder, MotorCommand::Velocity(1.5))
        .unwrap();
    run(&mut w, 1.5);

    // Compute total linear momentum P and total angular momentum L about origin.
    let mut total_p = Vec3::ZERO;
    let mut total_l = Vec3::ZERO;
    for b in [base, arm] {
        let pose = w.scene.body_pose(b).unwrap();
        let vel = w.scene.body_velocity(b).unwrap();
        let mp = w.scene.body_mass_properties(b).unwrap();
        let r = Mat3::from_quat(pose.rotation);
        let i_world = r * mp.inertia * r.transpose();
        let com = pose.transform_point(mp.local_com);
        let p = vel.linear * mp.mass;
        total_p += p;
        total_l += i_world * vel.angular + com.cross(p);
    }
    // Base must recoil in opposite angular direction while total P and L remain 0!
    assert!(
        w.scene.body_velocity(base).unwrap().angular.z < -0.05,
        "base should recoil against arm swing"
    );
    assert!(total_p.length() < 1e-6, "linear momentum drift: {total_p:?}");
    assert!(total_l.length() < 1e-5, "angular momentum drift: {total_l:?}");
}

#[test]
fn spacecraft_tracks_6dof_rendezvous_trajectory_in_orbital_lvlh_frame() {
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w.devices.orbital.set_field(OrbitalField::iss_leo()).unwrap();

    let craft = free_body(&mut w, Vec3::new(-10.0, 2.0, 0.0), 25.0, Vec3::splat(5.0));
    let thrusters = mount_12_thruster_rcs(&mut w, craft, 0.6, 50.0, None, 0.0);
    let gnc = w
        .devices
        .spacecraft
        .insert(
            &w.scene,
            &w.devices.thrusters,
            &w.devices.reaction_wheels,
            SpacecraftSpec {
                body: craft,
                frame: Transform::IDENTITY,
                thrusters,
                reaction_wheels: Vec::new(),
                limits: SpacecraftLimits::default(),
            },
        )
        .unwrap();

    // Smooth cubic approach from (-10, 2, 0) to (0, 0, 0) while rotating +90° about Y over 10 s.
    let start = Vec3::new(-10.0, 2.0, 0.0);
    let total_t = 10.0;
    let steps = (total_t / DT).round() as usize;
    let mut max_pos_err: f64 = 0.0;
    for k in 0..=steps {
        let t = k as f64 * DT;
        let s = (t / total_t).clamp(0.0, 1.0);
        let h = 3.0 * s * s - 2.0 * s * s * s;
        let hd = (6.0 * s - 6.0 * s * s) / total_t;
        let hdd = (6.0 - 12.0 * s) / (total_t * total_t);
        let pos = start * (1.0 - h);
        let vel = -start * hd;
        let acc = -start * hdd;
        let yaw = FRAC_PI_2 * h;
        let yaw_rate = FRAC_PI_2 * hd;
        let yaw_acc = FRAC_PI_2 * hdd;
        w.devices
            .spacecraft
            .command(
                gnc,
                SpacecraftCommand::Trajectory {
                    position: pos,
                    velocity: vel,
                    acceleration: acc,
                    orientation: Quat::from_rotation_y(yaw),
                    angular_velocity: Vec3::Y * yaw_rate,
                    angular_acceleration: Vec3::Y * yaw_acc,
                },
            )
            .unwrap();
        w.step(DT).unwrap();
        let err = w.scene.body_pose(craft).unwrap().position.distance(pos);
        max_pos_err = max_pos_err.max(err);
    }
    let final_pose = w.scene.body_pose(craft).unwrap();
    let final_rot_err = final_pose
        .rotation
        .angle_between(Quat::from_rotation_y(FRAC_PI_2));
    assert!(max_pos_err < 0.08, "max trajectory pos error {max_pos_err}");
    assert!(
        final_pose.position.length() < 0.03 && final_rot_err < 0.02,
        "final rendezvous error: pos={:?}, rot_err={final_rot_err}",
        final_pose.position
    );
}

// ─── 6. Input Validation ─────────────────────────────────────────────────

#[test]
fn space_devices_reject_invalid_specs_and_commands() {
    let mut w = RigidWorld::new().unwrap();
    let craft = free_body(&mut w, Vec3::ZERO, 10.0, Vec3::splat(2.0));

    // Thruster validation
    assert!(
        w.devices
            .thrusters
            .insert(&w.scene, ThrusterSpec::ideal(craft, Transform::IDENTITY, -1.0))
            .is_err()
    );
    assert!(
        w.devices
            .thrusters
            .insert(
                &w.scene,
                ThrusterSpec {
                    specific_impulse: Some(0.0),
                    ..ThrusterSpec::ideal(craft, Transform::IDENTITY, 10.0)
                }
            )
            .is_err()
    );
    let tid = w
        .devices
        .thrusters
        .insert(&w.scene, ThrusterSpec::ideal(craft, Transform::IDENTITY, 10.0))
        .unwrap();
    assert!(w.devices.thrusters.set_throttle(tid, f64::NAN).is_err());
    assert!(w.devices.thrusters.fire_pulse(tid, 1.0, -0.1).is_err());

    // Reaction wheel & CMG validation
    assert!(
        w.devices
            .reaction_wheels
            .insert(
                &w.scene,
                ReactionWheelSpec {
                    body: craft,
                    axis: Vec3::ZERO,
                    inertia: 0.1,
                    max_torque: 1.0,
                    max_speed: 10.0,
                    initial_speed: 0.0,
                    gimbal: None,
                }
            )
            .is_err()
    );
    assert!(
        w.devices
            .reaction_wheels
            .insert(
                &w.scene,
                ReactionWheelSpec {
                    body: craft,
                    axis: Vec3::X,
                    inertia: 0.1,
                    max_torque: 1.0,
                    max_speed: 10.0,
                    initial_speed: 20.0, // exceeds max_speed
                    gimbal: None,
                }
            )
            .is_err()
    );
    assert!(
        w.devices
            .reaction_wheels
            .insert(
                &w.scene,
                ReactionWheelSpec {
                    body: craft,
                    axis: Vec3::X,
                    inertia: 0.1,
                    max_torque: 1.0,
                    max_speed: 10.0,
                    initial_speed: 0.0,
                    gimbal: Some(GimbalSpec {
                        axis: Vec3::X, // not orthogonal to spin axis!
                        max_rate: 1.0,
                        initial_angle: 0.0,
                    }),
                }
            )
            .is_err()
    );
    let wid = w
        .devices
        .reaction_wheels
        .insert(
            &w.scene,
            ReactionWheelSpec {
                body: craft,
                axis: Vec3::X,
                inertia: 0.1,
                max_torque: 1.0,
                max_speed: 10.0,
                initial_speed: 0.0,
                gimbal: None,
            },
        )
        .unwrap();
    assert!(w.devices.reaction_wheels.set_gimbal_rate(wid, 0.5).is_err());
    assert!(
        w.devices
            .reaction_wheels
            .command(wid, ReactionWheelCommand::Torque(f64::INFINITY))
            .is_err()
    );

    // Orbital field validation
    assert!(
        w.devices
            .orbital
            .set_field(OrbitalField::ClohessyWiltshire {
                mean_motion: 0.01,
                radial_axis: Vec3::Y,
                along_track_axis: Vec3::new(0.5, 0.5, 0.0), // non-orthogonal!
                origin: Vec3::ZERO,
                gravity_gradient: false,
            })
            .is_err()
    );
    assert!(
        OrbitalField::circular_lvlh(EARTH_MU, EARTH_EQUATORIAL_RADIUS + 400_000.0, Vec3::Y, Vec3::X, true)
            .is_ok()
    );
    assert!(OrbitalField::circular_lvlh(-1.0, 1000.0, Vec3::Y, Vec3::X, false).is_err());

    // ISS LEO helper + body exclusion
    w.devices.orbital.set_field(OrbitalField::iss_leo()).unwrap();
    assert!(w.devices.orbital.active());
    w.devices.orbital.exclude_body(craft);
    w.step(DT).unwrap();
    assert!(w.devices.orbital.reading(craft).is_none());
    w.devices.orbital.include_body(craft);
    w.step(DT).unwrap();
    assert!(w.devices.orbital.reading(craft).is_some());

    // Spacecraft controller validation
    assert!(
        w.devices
            .spacecraft
            .insert(
                &w.scene,
                &w.devices.thrusters,
                &w.devices.reaction_wheels,
                SpacecraftSpec {
                    body: craft,
                    frame: Transform::IDENTITY,
                    thrusters: Vec::new(),
                    reaction_wheels: Vec::new(),
                    limits: SpacecraftLimits::default(),
                }
            )
            .is_err()
    );
}

// ─── 6. Research-Backed LEO Environment, SRI CMG, Imbalance, Plume & Slosh ─

#[test]
fn sgcmg_sri_steering_slews_spacecraft_and_avoids_singularity() {
    use molla_solvers::devices::GimbalSpec;
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let craft = free_body(&mut w, Vec3::ZERO, 120.0, Vec3::new(40.0, 45.0, 42.0));
    // Standard 4-pyramid SGCMG cluster (skew angle β = 54.73°, symmetric 3-axis authority!).
    let beta = 54.73_f64.to_radians();
    let sb = beta.sin();
    let cb = beta.cos();
    let pyramid = [
        (Vec3::new(0.0, -1.0, 0.0), Vec3::new(sb, 0.0, cb)),
        (Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, sb, cb)),
        (Vec3::new(0.0, 1.0, 0.0), Vec3::new(-sb, 0.0, cb)),
        (Vec3::new(-1.0, 0.0, 0.0), Vec3::new(0.0, -sb, cb)),
    ];
    let cmgs: Vec<_> = pyramid
        .into_iter()
        .map(|(spin, gimbal_axis)| {
            w.devices
                .reaction_wheels
                .insert(
                    &w.scene,
                    ReactionWheelSpec {
                        body: craft,
                        axis: spin,
                        inertia: 0.4,
                        max_torque: 5.0,
                        max_speed: 200.0,
                        initial_speed: 100.0, // 40 N·m·s per CMG
                        gimbal: Some(GimbalSpec {
                            axis: gimbal_axis,
                            max_rate: 1.5,
                            initial_angle: 0.0,
                        }),
                    },
                )
                .unwrap()
        })
        .collect();

    // Pure SGCMG attitude control (NO thrusters, NO fixed reaction wheels!).
    let gnc = w
        .devices
        .spacecraft
        .insert(
            &w.scene,
            &w.devices.thrusters,
            &w.devices.reaction_wheels,
            SpacecraftSpec {
                body: craft,
                frame: Transform::IDENTITY,
                thrusters: Vec::new(),
                reaction_wheels: cmgs.clone(),
                limits: SpacecraftLimits {
                    attitude_frequency: 1.1,
                    ..SpacecraftLimits::default()
                },
            },
        )
        .unwrap();

    let target = (Quat::from_rotation_x(0.45) * Quat::from_rotation_y(-0.35)).normalize();
    w.devices
        .spacecraft
        .command(
            gnc,
            SpacecraftCommand::PoseHold {
                position: Vec3::ZERO,
                orientation: target,
            },
        )
        .unwrap();
    run(&mut w, 8.0);

    let err = w.scene.body_pose(craft).unwrap().rotation.angle_between(target);
    assert!(err < 0.02, "SGCMG SRI failed to converge: err={err:.4} rad");
    for &id in &cmgs {
        let r = w.devices.reaction_wheels.reading(id).unwrap();
        assert!(r.gimbal_angle.is_finite() && r.gimbal_rate.abs() <= 1.5 + 1e-9);
    }
}

#[test]
fn reaction_wheel_imbalance_exerts_analytic_omega_squared_jitter() {
    use molla_solvers::devices::ImbalanceSpec;
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let craft = free_body(&mut w, Vec3::ZERO, 100.0, Vec3::splat(20.0));
    let omega = 120.0;
    let us = 2.0e-4; // kg·m
    let ud = 5.0e-5; // kg·m²
    let rw = w
        .devices
        .reaction_wheels
        .insert(
            &w.scene,
            ReactionWheelSpec {
                body: craft,
                axis: Vec3::Z,
                inertia: 0.1,
                max_torque: 5.0,
                max_speed: 200.0,
                initial_speed: omega,
                gimbal: None,
            },
        )
        .unwrap();
    w.devices
        .reaction_wheels
        .set_imbalance(
            rw,
            ImbalanceSpec {
                static_unbalance: us,
                dynamic_unbalance: ud,
                mount_offset: Vec3::ZERO,
                initial_phase: 0.0,
            },
        )
        .unwrap();
    w.step(1e-4).unwrap();
    let r = w.devices.reaction_wheels.reading(rw).unwrap();
    let expected_f = us * omega * omega;
    let expected_t = ud * omega * omega;
    assert!(
        (r.jitter_force.length() - expected_f).abs() / expected_f < 1e-2,
        "static jitter {:?} vs expected {expected_f}",
        r.jitter_force
    );
    assert!(
        (r.jitter_torque.length() - expected_t).abs() / expected_t < 1e-2,
        "dynamic jitter {:?} vs expected {expected_t}",
        r.jitter_torque
    );
    assert!(r.jitter_force.z.abs() < 1e-9, "jitter must be orthogonal to spin axis");
}

#[test]
fn srp_matches_analytic_photon_momentum_and_earth_eclipse() {
    use molla_solvers::devices::{SOLAR_PRESSURE_1AU, SolarRadiation, SurfaceFacetSpec};
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let craft = free_body(&mut w, Vec3::ZERO, 10.0, Vec3::splat(2.0));
    let area = 20.0;
    let rs = 0.6;
    let rd = 0.3;
    let facet = w
        .devices
        .space_env
        .insert_facet(
            &w.scene,
            SurfaceFacetSpec {
                body: craft,
                local_center: Vec3::ZERO,
                local_normal: Vec3::X,
                area,
                drag_cd: 0.0,
                reflectivity_spec: rs,
                reflectivity_diff: rd,
                sigma_n: 0.9,
                sigma_t: 0.9,
                double_sided: false,
            },
        )
        .unwrap();
    // Sun along +X (normal incidence, cos(theta) = 1).
    // Analytic: F_x = -P0 * A * ((1 - rs) + 2*(rs + rd/3)) = -P0 * A * (1 + rs + 2*rd/3).
    w.devices
        .space_env
        .set_solar_radiation(Some(SolarRadiation::one_au(Vec3::X)))
        .unwrap();
    w.step(DT).unwrap();
    let r = w.devices.space_env.facet_reading(facet).unwrap();
    let expected = -SOLAR_PRESSURE_1AU * area * (1.0 + rs + 2.0 * rd / 3.0);
    assert!(!r.in_eclipse);
    assert!(
        (r.srp_force.x - expected).abs() < 1e-12,
        "SRP {:?} vs analytic {expected}",
        r.srp_force
    );

    // Place Earth between the Sun (+X) and the spacecraft (at x = +1000 m, radius 500 m):
    // spacecraft is now in cylindrical Earth umbra -> zero SRP!
    w.devices
        .space_env
        .set_solar_radiation(Some(SolarRadiation {
            pressure: SOLAR_PRESSURE_1AU,
            sun_direction: Vec3::X,
            eclipse_occluder: Some((Vec3::new(1000.0, 0.0, 0.0), 500.0)),
        }))
        .unwrap();
    w.step(DT).unwrap();
    let r_ecl = w.devices.space_env.facet_reading(facet).unwrap();
    assert!(r_ecl.in_eclipse);
    assert!(r_ecl.srp_force.length() < 1e-15);
}

#[test]
fn leo_faceted_drag_applies_analytic_force_and_torque() {
    use molla_solvers::devices::{OrbitalAtmosphere, SurfaceFacetSpec};
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let craft = free_body(&mut w, Vec3::ZERO, 50.0, Vec3::splat(10.0));
    let atmo = OrbitalAtmosphere::iss_408km();
    let area = 40.0;
    let cd = 2.2;
    let arm_y = 2.5;
    let facet = w
        .devices
        .space_env
        .insert_facet(
            &w.scene,
            SurfaceFacetSpec {
                body: craft,
                local_center: Vec3::new(0.0, arm_y, 0.0),
                local_normal: Vec3::X,
                area,
                drag_cd: cd,
                reflectivity_spec: 0.0,
                reflectivity_diff: 0.0,
                sigma_n: 0.9,
                sigma_t: 0.9,
                double_sided: true,
            },
        )
        .unwrap();
    w.devices.space_env.set_atmosphere(Some(atmo)).unwrap();
    w.step(DT).unwrap();
    let r = w.devices.space_env.facet_reading(facet).unwrap();
    let v = atmo.frame_orbital_velocity.x;
    let expected_fx = -0.5 * atmo.density * cd * area * v * v;
    assert!(
        (r.aero_force.x - expected_fx).abs() / expected_fx.abs() < 1e-5,
        "aero {:?} vs expected {expected_fx}",
        r.aero_force
    );
    // Offset along +Y with -X force creates +Z aerodynamic pitching torque!
    let omega_z = w.scene.body_velocity(craft).unwrap().angular.z;
    assert!(omega_z > 0.0, "expected +Z aero torque, got omega_z={omega_z}");
}

#[test]
fn magnetorquer_applies_m_cross_b_and_dumps_momentum_without_fuel() {
    use molla_solvers::devices::{MagneticField, MagnetorquerSpec, desaturation_dipole_body};
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let craft = free_body(&mut w, Vec3::ZERO, 40.0, Vec3::splat(8.0));
    let b_world = Vec3::new(0.0, 0.0, 4.0e-5); // 40 µT along +Z
    w.devices
        .space_env
        .set_magnetic_field(MagneticField::Uniform(b_world))
        .unwrap();
    let mtb = w
        .devices
        .space_env
        .insert_magnetorquer(
            &w.scene,
            MagnetorquerSpec {
                body: craft,
                max_dipole: Vec3::splat(500.0),
                residual_dipole: Vec3::ZERO,
            },
        )
        .unwrap();

    // Exact single-step torque check: `m = (0, -500, 0)`, `B = (0, 0, 4e-5)` -> `τ = (-0.02, 0, 0)`.
    w.devices
        .space_env
        .set_magnetorquer_dipole(mtb, Vec3::new(0.0, -500.0, 0.0))
        .unwrap();
    w.step(DT).unwrap();
    let r0 = w.devices.space_env.magnetorquer_reading(mtb).unwrap();
    assert!(
        (r0.world_torque.x + 0.02).abs() < 1e-12,
        "m x B mismatch: {:?}",
        r0.world_torque
    );

    // Give the spacecraft an initial spin about +X (orthogonal to B along +Z) and dump momentum
    // purely magnetically (0 propellant burned!).
    w.apply_body_impulse(craft, SpatialVector::new(Vec3::ZERO, Vec3::X * 0.4))
        .unwrap();
    let omega0 = w.scene.body_velocity(craft).unwrap().angular.x;
    let steps = (6.0 / DT) as usize;
    for _ in 0..steps {
        let q = w.scene.body_pose(craft).unwrap().rotation;
        let omega_b = q.conjugate() * w.scene.body_velocity(craft).unwrap().angular;
        let h_b = omega_b * 8.0;
        let b_b = q.conjugate() * b_world;
        let cmd = desaturation_dipole_body(h_b, b_b, 2.0, Vec3::splat(500.0));
        w.devices.space_env.set_magnetorquer_dipole(mtb, cmd).unwrap();
        w.step(DT).unwrap();
    }
    let omega1 = w.scene.body_velocity(craft).unwrap().angular.x;
    assert!(
        omega1 < omega0 - 0.014,
        "magnetorquer failed to damp spin: before={omega0}, after={omega1}"
    );
}

#[test]
fn dettleff_plume_impingement_produces_normal_and_tangential_shear_on_angled_panel() {
    use molla_solvers::devices::SurfaceFacetSpec;
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    // Firing spacecraft at origin with exhaust directed along +X (thrust along -X).
    let firer = free_body(&mut w, Vec3::ZERO, 500.0, Vec3::splat(100.0));
    let noz = w
        .devices
        .thrusters
        .insert(
            &w.scene,
            ThrusterSpec {
                body: firer,
                frame: Transform::new(Vec3::ZERO, Quat::from_rotation_z(PI)),
                max_thrust: 400.0,
                specific_impulse: None,
                propellant_mass: 0.0,
                deplete_body_mass: false,
                min_impulse_bit: 0.0,
                response_time: 0.0,
            },
        )
        .unwrap();
    w.devices.thrusters.set_throttle(noz, 1.0).unwrap();

    // Target panel at x = +2.0 m, tilted 45° so normal = (-1, 1, 0)/sqrt(2).
    let target = free_body(&mut w, Vec3::new(2.0, 0.0, 0.0), 100.0, Vec3::splat(20.0));
    let facet = w
        .devices
        .space_env
        .insert_facet(
            &w.scene,
            SurfaceFacetSpec {
                body: target,
                local_center: Vec3::ZERO,
                local_normal: Vec3::new(-1.0, 1.0, 0.0).normalize(),
                area: 2.0,
                drag_cd: 0.0,
                reflectivity_spec: 0.0,
                reflectivity_diff: 0.0,
                sigma_n: 0.85,
                sigma_t: 0.85,
                double_sided: false,
            },
        )
        .unwrap();
    w.devices.space_env.set_plume_impingement(true, 4.0).unwrap();
    w.step(DT).unwrap();

    let r = w.devices.space_env.facet_reading(facet).unwrap();
    // Normal push has +X and -Y components; tangential shear adds +X and +Y -> net +X > 0!
    assert!(r.plume_force.x > 1.0, "expected strong +X impingement, got {:?}", r.plume_force);
    // Compare specular (sigma_t = 0) vs diffuse (sigma_t = 0.85): tangential shear should reduce |F_y|!
    let mut w_spec = RigidWorld::new().unwrap();
    w_spec.scene.set_gravity(Vec3::ZERO).unwrap();
    let f2 = free_body(&mut w_spec, Vec3::ZERO, 500.0, Vec3::splat(100.0));
    let n2 = w_spec
        .devices
        .thrusters
        .insert(
            &w_spec.scene,
            ThrusterSpec {
                body: f2,
                frame: Transform::new(Vec3::ZERO, Quat::from_rotation_z(PI)),
                max_thrust: 400.0,
                specific_impulse: None,
                propellant_mass: 0.0,
                deplete_body_mass: false,
                min_impulse_bit: 0.0,
                response_time: 0.0,
            },
        )
        .unwrap();
    w_spec.devices.thrusters.set_throttle(n2, 1.0).unwrap();
    let t2 = free_body(&mut w_spec, Vec3::new(2.0, 0.0, 0.0), 100.0, Vec3::splat(20.0));
    let fac2 = w_spec
        .devices
        .space_env
        .insert_facet(
            &w_spec.scene,
            SurfaceFacetSpec {
                body: t2,
                local_center: Vec3::ZERO,
                local_normal: Vec3::new(-1.0, 1.0, 0.0).normalize(),
                area: 2.0,
                drag_cd: 0.0,
                reflectivity_spec: 0.0,
                reflectivity_diff: 0.0,
                sigma_n: 0.0,
                sigma_t: 0.0,
                double_sided: false,
            },
        )
        .unwrap();
    w_spec.devices.space_env.set_plume_impingement(true, 4.0).unwrap();
    w_spec.step(DT).unwrap();
    let r_spec = w_spec.devices.space_env.facet_reading(fac2).unwrap();
    let n_hat = Vec3::new(-1.0, 1.0, 0.0).normalize();
    let t_hat = Vec3::new(1.0, 1.0, 0.0).normalize();
    let normal_spec = (-r_spec.plume_force).dot(n_hat);
    let normal_diff = (-r.plume_force).dot(n_hat);
    let shear_spec = r_spec.plume_force.dot(t_hat);
    let shear_diff = r.plume_force.dot(t_hat);
    assert!(
        normal_spec > normal_diff && r_spec.plume_force.length() > r.plume_force.length(),
        "specular normal momentum transfer should exceed diffuse: spec={:?}, diff={:?}",
        r_spec.plume_force,
        r.plume_force
    );
    assert!(
        shear_spec.abs() < 1e-6 && shear_diff > 50.0,
        "tangential shear must be 0 for specular and positive for diffuse: spec={shear_spec}, diff={shear_diff}"
    );
}

#[test]
fn microgravity_slosh_conserves_momentum_and_dissipates_energy() {
    use molla_solvers::devices::SloshTankSpec;
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let bus_mass = 200.0;
    let slosh_mass = 50.0;
    let craft = free_body(&mut w, Vec3::ZERO, bus_mass, Vec3::splat(40.0));
    let tank = w
        .devices
        .space_env
        .insert_slosh_tank(
            &w.scene,
            SloshTankSpec {
                body: craft,
                tank_center_local: Vec3::ZERO,
                slosh_mass,
                stiffness: 250.0,
                damping: 18.0,
                max_displacement: 0.35,
            },
        )
        .unwrap();

    // Deliver a sharp RCS impulse to the spacecraft bus: initially only the bus moves,
    // then the internal slosh mass lags behind, oscillates, and damps until both share
    // the exact center-of-mass velocity `P / (m_bus + m_slosh)`!
    let impulse_x = 250.0; // Total system momentum = 250 N·s -> final v = 1.0 m/s!
    w.apply_body_impulse(craft, SpatialVector::new(Vec3::X * impulse_x, Vec3::ZERO))
        .unwrap();

    let mut max_disp = 0.0_f64;
    for _ in 0..800 {
        w.step(DT).unwrap();
        let sr = w.devices.space_env.slosh_reading(tank).unwrap();
        max_disp = max_disp.max(sr.displacement_local.length());
    }
    assert!(max_disp > 0.04, "slosh mass never excited: max_disp={max_disp}");

    let v_bus = w.scene.body_velocity(craft).unwrap().linear;
    let p_slosh = w.devices.space_env.slosh_momentum_world(tank).unwrap();
    let p_total = v_bus * bus_mass + p_slosh;
    assert!(
        (p_total.x - impulse_x).abs() < 1e-3,
        "total momentum not conserved: {p_total:?} vs {impulse_x}"
    );
    let sr_end = w.devices.space_env.slosh_reading(tank).unwrap();
    assert!(
        sr_end.velocity_local.length() < 0.05 && (v_bus.x - 1.0).abs() < 0.05,
        "slosh failed to damp to shared COM speed 1.0 m/s: v_bus={v_bus:?}, rel_v={:?}",
        sr_end.velocity_local
    );
}

#[test]
fn flexible_solar_array_csi_bending_mode_matches_analytical_frequency_and_conserves_h() {
    use molla_solvers::devices::FlexJointSpec;
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    // Heavy spacecraft bus (fixed-like ratio 100:1) + flexible hinged solar panel
    let i_bus = 5_000.0;
    let i_panel_about_hinge = 40.0; // panel mass 20 kg at lever 1.2 m + I_local 11.2 = 40.0
    let bus = free_body(&mut w, Vec3::ZERO, 2_000.0, Vec3::splat(i_bus));
    let panel = free_body(
        &mut w,
        Vec3::new(0.0, 0.0, 2.2),
        20.0,
        Vec3::new(11.2, 11.2, 5.0),
    );
    let hinge = w
        .scene
        .insert_joint(JointDesc {
            parent: Some(bus),
            child: panel,
            kind: JointType::Revolute,
            axis: Vec3::Y,
            frame_parent: Transform::from_translation(Vec3::new(0.0, 0.0, 1.0)),
            frame_child: Transform::from_translation(Vec3::new(0.0, 0.0, -1.2)),
        })
        .unwrap();

    let target_hz = 0.30;
    let fj = w
        .devices
        .space_env
        .insert_flex_joint(
            &w.scene,
            FlexJointSpec::from_modal_frequency(hinge, i_panel_about_hinge, target_hz, 0.015),
        )
        .unwrap();

    // Apply a small flap impulse to the outer panel to excite the 0.30 Hz mode.
    w.apply_body_impulse(panel, SpatialVector::new(Vec3::new(12.0, 0.0, 0.0), Vec3::ZERO))
        .unwrap();

    let mut crossings = Vec::new();
    let mut prev_q = 0.0_f64;
    let mut peak_e_early = 0.0_f64;
    let mut peak_e_late = 0.0_f64;
    let mut peak_bus_omega = 0.0_f64;
    let dt = 0.01;
    for step in 0..1_200 {
        w.step(dt).unwrap();
        let t = (step + 1) as f64 * dt;
        let r = w.devices.space_env.flex_joint_reading(fj).unwrap();
        if step > 5 && prev_q <= 0.0 && r.deflection > 0.0 {
            crossings.push(t);
        }
        prev_q = r.deflection;
        if t < 2.0 {
            peak_e_early = peak_e_early.max(r.elastic_energy_j);
        } else if t > 9.0 {
            peak_e_late = peak_e_late.max(r.elastic_energy_j);
        }
        peak_bus_omega = peak_bus_omega.max(w.scene.body_velocity(bus).unwrap().angular.length());
    }
    assert!(
        crossings.len() >= 3,
        "expected >= 3 bending cycles in 12 s, got {}",
        crossings.len()
    );
    let period_meas = (crossings[2] - crossings[0]) * 0.5;
    let freq_meas = 1.0 / period_meas;
    // Free-floating reduced inertia raises frequency slightly above clamped-base 0.30 Hz (~0.303 Hz).
    assert!(
        (freq_meas - target_hz).abs() / target_hz < 0.06,
        "measured flex modal freq {freq_meas:.4} Hz vs target {target_hz:.4} Hz"
    );
    assert!(
        peak_e_early > 0.1 && peak_e_late < peak_e_early * 0.65,
        "modal energy should ring down passively: early={peak_e_early:.4}, late={peak_e_late:.4}"
    );
    assert!(
        peak_bus_omega > 1e-4,
        "CSI reaction torque must visibly couple back into spacecraft bus: peak_bus_omega={peak_bus_omega}"
    );
}

#[test]
fn thruster_pwm_and_pwpf_modulators_produce_discrete_pulses_matching_mean_demand() {
    use molla_solvers::devices::ThrusterModulation;
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let b_pwm = free_body(&mut w, Vec3::ZERO, 100.0, Vec3::splat(20.0));
    let b_pwpf = free_body(&mut w, Vec3::new(0.0, 5.0, 0.0), 100.0, Vec3::splat(20.0));
    let t_pwm = w
        .devices
        .thrusters
        .insert(&w.scene, ThrusterSpec::ideal(b_pwm, Transform::IDENTITY, 100.0))
        .unwrap();
    let t_pwpf = w
        .devices
        .thrusters
        .insert(&w.scene, ThrusterSpec::ideal(b_pwpf, Transform::IDENTITY, 100.0))
        .unwrap();

    w.devices
        .thrusters
        .set_modulation(t_pwm, ThrusterModulation::rcs_pwm_10hz())
        .unwrap();
    w.devices
        .thrusters
        .set_modulation(t_pwpf, ThrusterModulation::aerospace_pwpf())
        .unwrap();

    // 1. Sub-deadband demand produces zero pulses on both PWM and PWPF.
    w.devices.thrusters.set_throttle(t_pwm, 0.05).unwrap(); // 5 ms < 12 ms min_pulse_width
    w.devices.thrusters.set_throttle(t_pwpf, 0.01).unwrap(); // < u_on/Km = 0.025
    for _ in 0..50 {
        w.step(0.01).unwrap();
    }
    assert_eq!(w.devices.thrusters.reading(t_pwm).unwrap().impulse_total, 0.0);
    assert_eq!(w.devices.thrusters.reading(t_pwpf).unwrap().impulse_total, 0.0);

    // 2. Active duty 0.40 over 2.0 s -> expected impulse ≈ 80 N·s via discrete cycles!
    w.devices.thrusters.set_throttle(t_pwm, 0.40).unwrap();
    w.devices.thrusters.set_throttle(t_pwpf, 0.40).unwrap();
    for _ in 0..200 {
        w.step(0.01).unwrap();
    }
    let r_pwm = w.devices.thrusters.reading(t_pwm).unwrap();
    let r_pwpf = w.devices.thrusters.reading(t_pwpf).unwrap();
    assert!(
        (r_pwm.impulse_total - 80.0).abs() < 4.0 && r_pwm.valve_cycles >= 15,
        "PWM mean impulse/cycles mismatch: {r_pwm:?}"
    );
    assert!(
        (r_pwpf.impulse_total - 76.7).abs() < 6.0 && r_pwpf.valve_cycles >= 10,
        "PWPF mean impulse/cycles mismatch: {r_pwpf:?}"
    );
}

#[test]
fn space_imu_reads_zero_in_orbital_freefall_and_tracks_bias_and_thrust() {
    use molla_solvers::devices::ImuSpec;
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let r0 = EARTH_EQUATORIAL_RADIUS + 420_000.0;
    let v0 = (EARTH_MU / r0).sqrt();
    w.devices
        .orbital
        .set_field(OrbitalField::CentralGravity {
            center: Vec3::ZERO,
            mu: EARTH_MU,
            min_radius: 100_000.0,
            j2: None,
            gravity_gradient: false,
        })
        .unwrap();

    let sat = free_body(&mut w, Vec3::new(r0, 0.0, 0.0), 100.0, Vec3::splat(20.0));
    w.scene
        .set_body_velocity(sat, SpatialVector::new(Vec3::new(0.0, v0, 0.0), Vec3::ZERO))
        .unwrap();
    let thruster = w
        .devices
        .thrusters
        .insert(&w.scene, ThrusterSpec::ideal(sat, Transform::IDENTITY, 50.0))
        .unwrap();

    // Ideal zero-noise IMU first: verify Einstein equivalence principle (specific force = 0 in free-fall!).
    let imu_ideal = w
        .devices
        .space_sensors
        .insert_imu(
            &w.scene,
            ImuSpec {
                body: sat,
                mount_frame: Transform::IDENTITY,
                sample_rate_hz: 0.0,
                latency_s: 0.0,
                gyro_bias_init: Vec3::ZERO,
                gyro_arw: 0.0,
                gyro_rrw: 0.0,
                gyro_quantization: 0.0,
                accel_bias_init: Vec3::ZERO,
                accel_vrw: 0.0,
                accel_rrw: 0.0,
                seed: 1,
            },
        )
        .unwrap();
    let imu_noisy = w
        .devices
        .space_sensors
        .insert_imu(&w.scene, ImuSpec::space_mems(sat, 42))
        .unwrap();

    // Free-fall orbit: gravity is ~8.6 m/s² yet specific force must be ~0!
    for _ in 0..10 {
        w.step(0.01).unwrap();
    }
    let sf_freefall = w.devices.space_sensors.imu_reading(imu_ideal).unwrap().specific_force;
    assert!(
        sf_freefall.length() < 0.02,
        "free-fall orbital IMU must read ~0 specific force, got {sf_freefall:?}"
    );

    // Fire 50 N thruster on 100 kg satellite -> specific force = +0.50 m/s² along +X!
    w.devices.thrusters.set_throttle(thruster, 1.0).unwrap();
    for _ in 0..10 {
        w.step(0.01).unwrap();
    }
    let sf_burn = w.devices.space_sensors.imu_reading(imu_ideal).unwrap().specific_force;
    assert!(
        (sf_burn.x - 0.50).abs() < 0.02 && sf_burn.y.abs() < 0.02,
        "burning IMU specific force should equal F/m = 0.50 m/s², got {sf_burn:?}"
    );

    // Verify noisy MEMS IMU bias random walk evolved away from exact init bias.
    let r_noisy = w.devices.space_sensors.imu_reading(imu_noisy).unwrap();
    assert!(
        (r_noisy.true_gyro_bias - ImuSpec::space_mems(sat, 42).gyro_bias_init).length() > 1e-9,
        "gyro bias random walk did not evolve"
    );
}

#[test]
fn star_tracker_anisotropic_noise_latency_sun_and_rate_blinding() {
    use molla_solvers::devices::{SolarRadiation, StarTrackerSpec};
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w.devices
        .space_env
        .set_solar_radiation(Some(SolarRadiation::one_au(Vec3::Z)))
        .unwrap();

    let sat = free_body(&mut w, Vec3::ZERO, 100.0, Vec3::splat(20.0));
    let st = w
        .devices
        .space_sensors
        .insert_star_tracker(
            &w.scene,
            StarTrackerSpec {
                body: sat,
                mount_frame: Transform::IDENTITY, // +X boresight (90° from +Z Sun)
                sample_rate_hz: 50.0,
                latency_s: 0.06,
                cross_axis_noise_rad: 2.0e-4,
                boresight_noise_rad: 1.4e-3, // 7× cross-axis!
                sun_exclusion_rad: 25.0_f64.to_radians(),
                max_slew_rate_rad_s: 0.20,
                seed: 7,
            },
        )
        .unwrap();

    let mut sum_x2 = 0.0;
    let mut sum_y2 = 0.0;
    let mut n = 0;
    let mut first_fresh_t = None;
    for step in 0..120 {
        w.step(0.01).unwrap();
        let r = w.devices.space_sensors.star_tracker_reading(st).unwrap();
        if r.fresh && r.valid {
            if first_fresh_t.is_none() {
                first_fresh_t = Some((step + 1) as f64 * 0.01);
            }
            let e = r.orientation.to_scaled_axis();
            sum_x2 += e.x * e.x;
            sum_y2 += e.y * e.y;
            n += 1;
        }
    }
    assert!(
        first_fresh_t.unwrap() >= 0.06,
        "latency buffer failed: first fresh at {:?}",
        first_fresh_t
    );
    let rms_bore = (sum_x2 / n as f64).sqrt();
    let rms_cross = (sum_y2 / n as f64).sqrt();
    assert!(
        rms_bore > 3.5 * rms_cross,
        "boresight roll noise ({rms_bore:.5}) must greatly exceed cross-axis ({rms_cross:.5})"
    );

    // Slewing faster than 0.20 rad/s triggers rate blinding!
    w.scene
        .set_body_velocity(sat, SpatialVector::new(Vec3::ZERO, Vec3::new(0.0, 0.35, 0.0)))
        .unwrap();
    for _ in 0..15 {
        w.step(0.01).unwrap();
    }
    let r_rate = w.devices.space_sensors.star_tracker_reading(st).unwrap();
    assert!(r_rate.rate_blinded && !r_rate.valid, "expected rate blinding: {r_rate:?}");

    // Stop slewing and point boresight (+X) straight at Sun (+Z) -> Sun blinding!
    w.scene
        .set_body_velocity(sat, SpatialVector::ZERO)
        .unwrap();
    w.scene
        .set_body_pose(
            sat,
            Transform::from_rotation(Quat::from_rotation_y(-std::f64::consts::FRAC_PI_2)),
        )
        .unwrap();
    for _ in 0..15 {
        w.step(0.01).unwrap();
    }
    let r_sun = w.devices.space_sensors.star_tracker_reading(st).unwrap();
    assert!(r_sun.sun_blinded && !r_sun.valid, "expected sun blinding: {r_sun:?}");
}

#[test]
fn rendezvous_lidar_and_navigation_filter_estimate_gyro_bias_and_relative_pose() {
    use molla_solvers::devices::{ImuSpec, LidarSpec, NavigationFilter, StarTrackerSpec};
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();

    let station = free_body(&mut w, Vec3::ZERO, 5_000.0, Vec3::splat(2_000.0));
    let chaser = free_body(&mut w, Vec3::new(12.0, 0.5, -0.4), 1_000.0, Vec3::splat(400.0));
    w.scene
        .set_body_velocity(chaser, SpatialVector::new(Vec3::new(-0.3, 0.0, 0.0), Vec3::ZERO))
        .unwrap();

    // Mount LiDAR pointing along `-X` of chaser toward station at `(0, 0, 0)`.
    let lidar_mount = Transform::new(
        Vec3::new(-0.5, 0.0, 0.0),
        Quat::from_rotation_y(std::f64::consts::PI),
    );
    let true_bias = Vec3::new(0.008, -0.006, 0.005); // Deliberately large gyro bias to test online learning!
    let imu = w
        .devices
        .space_sensors
        .insert_imu(
            &w.scene,
            ImuSpec {
                gyro_bias_init: true_bias,
                gyro_arw: 1e-4,
                gyro_rrw: 0.0,
                latency_s: 0.0,
                ..ImuSpec::tactical_fog(chaser, 101)
            },
        )
        .unwrap();
    let st = w
        .devices
        .space_sensors
        .insert_star_tracker(
            &w.scene,
            StarTrackerSpec {
                sample_rate_hz: 10.0,
                latency_s: 0.02,
                ..StarTrackerSpec::leo_flight(chaser, Transform::IDENTITY, 102)
            },
        )
        .unwrap();
    let lidar = w
        .devices
        .space_sensors
        .insert_lidar(
            &w.scene,
            LidarSpec::rendezvous_proximity(chaser, station, lidar_mount, Vec3::ZERO, 103),
        )
        .unwrap();

    let mut nav = NavigationFilter::new(Quat::IDENTITY);
    let dt = 0.01;
    for _ in 0..800 {
        w.step(dt).unwrap();
        let r_imu = w.devices.space_sensors.imu_reading(imu).unwrap();
        let r_st = w.devices.space_sensors.star_tracker_reading(st).unwrap();
        let r_ld = w.devices.space_sensors.lidar_reading(lidar).unwrap();
        nav.update(dt, &r_imu, Some(&r_st), Some(&r_ld), lidar_mount);
    }
    assert!(nav.lidar_locked, "LiDAR never achieved target lock");
    let bias_err = (nav.gyro_bias_est - true_bias).length();
    assert!(
        bias_err < 1.5e-3,
        "NavigationFilter failed to converge on gyro bias: est={:?}, true={true_bias:?}",
        nav.gyro_bias_est
    );
    let true_rel_body = w
        .scene
        .body_pose(chaser)
        .unwrap()
        .inverse()
        .transform_point(w.scene.body_pose(station).unwrap().position);
    let pos_err = (nav.rel_target_pos_body - true_rel_body).length();
    assert!(
        pos_err < 0.08,
        "NavigationFilter relative position error too high: {pos_err:.4} m"
    );
}

#[test]
fn zonal_harmonics_j2_j3_j4_match_potential_gradient_to_machine_precision() {
    use molla_solvers::devices::ZonalHarmonics;
    let pole = Vec3::new(0.15, 0.92, -0.36).normalize();
    let cases = [
        ZonalHarmonics {
            j2: molla_solvers::devices::EARTH_J2,
            j3: 0.0,
            j4: 0.0,
            equatorial_radius: EARTH_EQUATORIAL_RADIUS,
            pole_axis: pole,
        },
        ZonalHarmonics {
            j2: 0.0,
            j3: molla_solvers::devices::EARTH_J3,
            j4: 0.0,
            equatorial_radius: EARTH_EQUATORIAL_RADIUS,
            pole_axis: pole,
        },
        ZonalHarmonics {
            j2: 0.0,
            j3: 0.0,
            j4: molla_solvers::devices::EARTH_J4,
            equatorial_radius: EARTH_EQUATORIAL_RADIUS,
            pole_axis: pole,
        },
        ZonalHarmonics::earth_egm96(pole),
    ];
    let probes = [
        Vec3::new(6_800_000.0, 1_200_000.0, -900_000.0),
        Vec3::new(-4_100_000.0, 5_800_000.0, 2_300_000.0),
        Vec3::new(7_100_000.0, 0.0, 0.0),
    ];
    // 4th-order central difference on perturbation potential alone verifies J2, J3, AND J4
    // individually to < 1e-10 without central μ/r swamping the 1e-6 J3/J4 terms!
    let eps = 2_000.0;
    for zonal in cases {
        let u_pert = |rv: Vec3| zonal.perturbation_potential(EARTH_MU, rv);
        for r_vec in probes {
            let a_analytic = zonal.perturbation_acceleration(EARTH_MU, r_vec, 1000.0);
            let fd4 = |axis: Vec3| {
                (u_pert(r_vec - axis * (2.0 * eps))
                    - 8.0 * u_pert(r_vec - axis * eps)
                    + 8.0 * u_pert(r_vec + axis * eps)
                    - u_pert(r_vec + axis * (2.0 * eps)))
                    / (12.0 * eps)
            };
            let a_numeric = Vec3::new(fd4(Vec3::X), fd4(Vec3::Y), fd4(Vec3::Z));
            let rel_err = (a_analytic - a_numeric).length() / a_numeric.length();
            assert!(
                rel_err < 1e-11,
                "zonal analytic gradient mismatch for {zonal:?} at {r_vec:?}: rel_err={rel_err:.3e}"
            );
        }
    }
    // Check J3 north-south asymmetry at exact equator (r ⊥ pole): a_J3 along pole = +3/2 J3 μ R_e³ / r⁵.
    let z3 = ZonalHarmonics {
        j2: 0.0,
        j3: molla_solvers::devices::EARTH_J3,
        j4: 0.0,
        equatorial_radius: EARTH_EQUATORIAL_RADIUS,
        pole_axis: Vec3::Z,
    };
    let r_eq = 7_000_000.0;
    let a3 = z3.perturbation_acceleration(EARTH_MU, Vec3::new(r_eq, 0.0, 0.0), 1000.0);
    let expected_z = 1.5 * z3.j3 * EARTH_MU * EARTH_EQUATORIAL_RADIUS.powi(3) / r_eq.powi(5);
    assert!((a3.z - expected_z).abs() < 1e-12 && a3.x.abs() < 1e-15);
}

#[test]
fn yamanaka_ankersen_and_eccentric_relative_match_nonlinear_inertial_and_rendezvous() {
    use molla_solvers::devices::{EccentricOrbit, YamanakaAnkersen};

    // 1. Zero-eccentricity limit reduces to exact analytical Clohessy-Wiltshire (< 1e-10).
    let r0 = EARTH_EQUATORIAL_RADIUS + 410_000.0;
    let circ = EccentricOrbit {
        mu: EARTH_MU,
        semi_major_axis: r0,
        eccentricity: 0.0,
        initial_true_anomaly: 0.0,
    };
    let n = circ.mean_motion();
    let t_q = 0.25 * circ.period_s();
    let (p_ya, v_ya) =
        YamanakaAnkersen::propagate(&circ, 0.0, t_q, Vec3::new(100.0, 0.0, 50.0), Vec3::new(0.0, -2.0 * n * 100.0, 0.0))
            .unwrap();
    // Closed 2:1 CW football orbit at quarter period: x = 0, y = -200 m, z = 0!
    assert!(
        (p_ya - Vec3::new(0.0, -200.0, 0.0)).length() < 1e-7,
        "YA e=0 failed CW check: p={p_ya:?}, v={v_ya:?}"
    );

    // 2. High-eccentricity (e = 0.35) comparison: Yamanaka-Ankersen STM vs RigidWorld
    //    `OrbitalField::EccentricRelative` vs exact nonlinear inertial two-body propagation!
    let ecc_orbit = EccentricOrbit {
        mu: EARTH_MU,
        semi_major_axis: 12_000_000.0,
        eccentricity: 0.35,
        initial_true_anomaly: 0.4,
    };
    let p0_lvlh = Vec3::new(250.0, -400.0, 180.0);
    let v0_lvlh = Vec3::new(0.12, -0.35, -0.08);
    let t_prop = 0.65 * ecc_orbit.period_s();

    let (p_ya_e, v_ya_e) =
        YamanakaAnkersen::propagate(&ecc_orbit, ecc_orbit.initial_true_anomaly, t_prop, p0_lvlh, v0_lvlh)
            .unwrap();

    // Integrate exact nonlinear Chief + Deputy in `OrbitalField::EccentricRelative`!
    let mut w = RigidWorld::new().unwrap();
    w.scene.set_gravity(Vec3::ZERO).unwrap();
    w.devices
        .orbital
        .set_field(
            OrbitalField::eccentric_lvlh(ecc_orbit, Vec3::X, Vec3::Y, false, false).unwrap(),
        )
        .unwrap();
    let dep = free_body(&mut w, p0_lvlh, 100.0, Vec3::splat(20.0));
    w.scene
        .set_body_velocity(dep, SpatialVector::new(v0_lvlh, Vec3::ZERO))
        .unwrap();
    let steps = 1_500;
    let dt = t_prop / steps as f64;
    for _ in 0..steps {
        w.step(dt).unwrap();
    }
    let p_sim = w.scene.body_pose(dep).unwrap().position;
    let v_sim = w.scene.body_velocity(dep).unwrap().linear;
    let pos_diff = (p_sim - p_ya_e).length();
    let vel_diff = (v_sim - v_ya_e).length();
    assert!(
        pos_diff < 5.0 && vel_diff < 0.01,
        "EccentricRelative vs YA mismatch on e=0.35 orbit over 0.65 orbits: pos_diff={pos_diff:.3} m, vel_diff={vel_diff:.5} m/s (p_sim={p_sim:?}, p_ya={p_ya_e:?})"
    );

    // 3. Closed-form two-impulse eccentric rendezvous targeting on e = 0.35 orbit!
    let t_xfer = 0.35 * ecc_orbit.period_s();
    let (dv1, dv2) = YamanakaAnkersen::rendezvous_delta_v(
        &ecc_orbit,
        ecc_orbit.initial_true_anomaly,
        t_xfer,
        p0_lvlh,
        v0_lvlh,
        Vec3::ZERO,
        Vec3::ZERO,
    )
    .unwrap();
    let (p_arr, v_arr) = YamanakaAnkersen::propagate(
        &ecc_orbit,
        ecc_orbit.initial_true_anomaly,
        t_xfer,
        p0_lvlh,
        v0_lvlh + dv1,
    )
    .unwrap();
    assert!(
        p_arr.length() < 1e-6 && (v_arr + dv2).length() < 1e-8,
        "YA two-impulse rendezvous targeting failed: p_arr={p_arr:?}, v_final={:?}",
        v_arr + dv2
    );
}

#[test]
fn nrlmsise_exponential_atmosphere_tracks_altitude_solar_activity_and_perigee_drag() {
    use molla_solvers::devices::{
        EccentricOrbit, NrlmsiseAtmosphere, SolarActivity, SurfaceFacetSpec, nrlmsise_density_kg_m3,
    };

    // 1. Check Vallado / NRLMSISE-00 reference densities & solar activity ordering.
    let rho_200 = nrlmsise_density_kg_m3(200_000.0, SolarActivity::Moderate);
    let rho_400_low = nrlmsise_density_kg_m3(400_000.0, SolarActivity::Low);
    let rho_400_mod = nrlmsise_density_kg_m3(400_000.0, SolarActivity::Moderate);
    let rho_400_high = nrlmsise_density_kg_m3(400_000.0, SolarActivity::High);
    let rho_700 = nrlmsise_density_kg_m3(700_000.0, SolarActivity::Moderate);
    assert!((rho_200 - 2.789e-10).abs() / 2.789e-10 < 0.01);
    assert!((rho_400_mod - 3.725e-12).abs() / 3.725e-12 < 0.01);
    assert!(rho_400_high > 2.0 * rho_400_mod && rho_400_mod > 1.8 * rho_400_low);
    assert!(rho_200 > 50.0 * rho_400_mod && rho_400_mod > 50.0 * rho_700);

    // 2. Eccentric orbit (perigee 240 km, apogee 740 km): verify perigee drag >> apogee drag!
    let r_p = EARTH_EQUATORIAL_RADIUS + 240_000.0;
    let r_a = EARTH_EQUATORIAL_RADIUS + 740_000.0;
    let a = 0.5 * (r_p + r_a);
    let e = (r_a - r_p) / (r_a + r_p);

    let measure_drag_at_nu = |nu0: f64| -> f64 {
        let mut w = RigidWorld::new().unwrap();
        w.scene.set_gravity(Vec3::ZERO).unwrap();
        let orbit = EccentricOrbit {
            mu: EARTH_MU,
            semi_major_axis: a,
            eccentricity: e,
            initial_true_anomaly: nu0,
        };
        w.devices
            .orbital
            .set_field(OrbitalField::eccentric_lvlh(orbit, Vec3::Y, Vec3::X, false, false).unwrap())
            .unwrap();
        w.devices
            .space_env
            .set_nrlmsise_atmosphere(Some(NrlmsiseAtmosphere::earth_lvlh(
                400_000.0,
                SolarActivity::Moderate,
            )))
            .unwrap();
        let sat = free_body(&mut w, Vec3::ZERO, 500.0, Vec3::splat(100.0));
        let f = w
            .devices
            .space_env
            .insert_facet(&w.scene, SurfaceFacetSpec::solar_panel(sat, Vec3::ZERO, Vec3::X, 20.0))
            .unwrap();
        w.step(0.01).unwrap();
        w.devices.space_env.facet_reading(f).unwrap().aero_force.length()
    };

    let drag_perigee = measure_drag_at_nu(0.0);
    let drag_apogee = measure_drag_at_nu(std::f64::consts::PI);
    assert!(
        drag_perigee > 500.0 * drag_apogee && drag_apogee > 0.0,
        "perigee drag ({drag_perigee:.3e} N) should vastly exceed apogee drag ({drag_apogee:.3e} N)"
    );
}
