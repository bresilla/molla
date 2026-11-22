//! Copter controllers (multirotors): every step they read their body's
//! state and set their rotors' speeds to follow a commanded velocity or
//! attitude.
//!
//! The loops cascade: velocity to a desired thrust direction and magnitude,
//! attitude (on SO(3)) to body rates, rates to torque through the body's
//! inertia. A disturbance observer compares the acceleration the body had
//! with what thrust and gravity gave it, and the velocity loop cancels the
//! difference (wind, drag, payload) without touching setpoint changes. A
//! mixer built from the rotors' positions, shafts and constants turns the
//! collective thrust and torque into each rotor's `|ω|·ω`.

use std::collections::BTreeMap;

use glam::{DMat4, DVec4};
use molla_core::{Error, Result};
use molla_math::{Mat3, Quat, Real, Transform, Vec3};
use molla_sim::runtime::{BodyHandle, RigidScene};

use super::{PropellerId, Propellers};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CopterId(pub u64);

/// What a copter controller holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CopterCommand {
    /// Rotors stopped.
    Off,
    /// Velocity in the heading frame (forward, left, up; m/s) and yaw rate
    /// about up (rad/s, counter-clockwise).
    Velocity {
        forward: Real,
        left: Real,
        up: Real,
        yaw_rate: Real,
    },
    /// World-space position, velocity and acceleration references, with a world heading.
    Trajectory {
        position: Vec3,
        velocity: Vec3,
        acceleration: Vec3,
        heading: Vec3,
    },
    /// Roll (right side down) and pitch (nose down) in rad, yaw rate and
    /// climb rate.
    Attitude {
        roll: Real,
        pitch: Real,
        yaw_rate: Real,
        climb: Real,
    },
}

/// Limits and loop gains of a copter controller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CopterLimits {
    /// Largest tilt from level (rad).
    pub max_tilt: Real,
    /// Largest horizontal speed and climb rate (m/s).
    pub max_speed: Real,
    pub max_climb: Real,
    pub max_yaw_rate: Real,
    /// Velocity loop gain (1/s).
    pub velocity_gain: Real,
    /// Position error to velocity correction (1/s).
    pub position_gain: Real,
    /// Time constant (s) of the disturbance observer.
    pub disturbance_time: Real,
    /// Attitude loop natural frequency (rad/s), critically damped.
    pub attitude_frequency: Real,
    /// Yaw-rate loop gain (1/s).
    pub yaw_gain: Real,
}

impl Default for CopterLimits {
    fn default() -> Self {
        Self {
            max_tilt: 0.5,
            max_speed: 10.0,
            max_climb: 3.0,
            max_yaw_rate: 1.5,
            velocity_gain: 1.5,
            position_gain: 1.2,
            disturbance_time: 0.5,
            attitude_frequency: 6.0,
            yaw_gain: 4.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CopterSpec {
    pub body: BodyHandle,
    /// Flight frame in the body frame: +X forward, +Y left, +Z up.
    pub frame: Transform,
    /// Rotors on `body`; at least four that can steer all four axes.
    pub propellers: Vec<PropellerId>,
    pub limits: CopterLimits,
}

/// A copter controller's view of its body after the last update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CopterReading {
    pub command: CopterCommand,
    /// Roll (right side down) and pitch (nose down), rad.
    pub roll: Real,
    pub pitch: Real,
    /// Velocity in the heading frame (forward, left, up), m/s.
    pub velocity: [Real; 3],
    pub yaw_rate: Real,
    /// Collective thrust asked of the rotors (N).
    pub thrust: Real,
    /// Whether a rotor hit its speed limit.
    pub saturated: bool,
}

struct Copter {
    spec: CopterSpec,
    command: CopterCommand,
    /// Estimated disturbance acceleration (m/s², world).
    disturbance: Vec3,
    /// Previous velocity and integration interval.
    last: Option<(Vec3, Real)>,
    reading: CopterReading,
}

/// Copter controllers of a rigid world.
#[derive(Default)]
pub struct Copters {
    next: u64,
    copters: BTreeMap<CopterId, Copter>,
}

/// One rotor in the flight frame: its `[thrust along up, roll, pitch, yaw
/// torque]` per unit `|ω|·ω`, and its speed limit.
struct Rotor {
    id: PropellerId,
    column: DVec4,
    sign: Real,
    max_s: Real,
}

fn rotors(propellers: &Propellers, scene: &RigidScene, spec: &CopterSpec) -> Result<Vec<Rotor>> {
    let mass = scene.body_mass_properties(spec.body)?;
    let to_flight = spec.frame.rotation.conjugate();
    spec.propellers
        .iter()
        .map(|&id| {
            let p = propellers
                .spec(id)
                .ok_or_else(|| Error::Build("copter propeller is missing".into()))?;
            if p.body != spec.body {
                return Err(Error::Build("copter propellers must be on its body".into()));
            }
            let axis = to_flight * (p.frame.rotation * Vec3::X);
            let arm = to_flight * (p.frame.position - mass.local_com - spec.frame.position);
            let [t1, q1] = [p.thrust[0], p.torque[0]];
            let torque = arm.cross(axis) * t1 - axis * q1;
            Ok(Rotor {
                id,
                column: DVec4::new(t1 * axis.z, torque.x, torque.y, torque.z),
                sign: if t1 >= 0.0 { 1.0 } else { -1.0 },
                max_s: p.limits.max_velocity * p.limits.max_velocity,
            })
        })
        .collect()
}

/// `(M·Mᵀ)⁻¹` of the mixer matrix, or an error when the rotors cannot steer
/// thrust and all three torques.
fn mixer_inverse(rotors: &[Rotor]) -> Result<DMat4> {
    let mut gram = DMat4::ZERO;
    for r in rotors {
        let c = r.column;
        gram += DMat4::from_cols(c * c.x, c * c.y, c * c.z, c * c.w);
    }
    let diagonal = DVec4::new(gram.x_axis.x, gram.y_axis.y, gram.z_axis.z, gram.w_axis.w);
    if !diagonal.is_finite() || diagonal.min_element() <= 0.0 {
        return Err(Error::Build("copter rotors cannot steer all axes".into()));
    }
    let scale = DVec4::from_array(diagonal.to_array().map(|v| v.sqrt().recip()));
    let normalize = |m: DMat4| {
        DMat4::from_cols(
            m.x_axis * scale * scale.x,
            m.y_axis * scale * scale.y,
            m.z_axis * scale * scale.z,
            m.w_axis * scale * scale.w,
        )
    };
    let balanced = normalize(gram);
    let det = balanced.determinant();
    if !det.is_finite() || det <= 1e-12 {
        return Err(Error::Build("copter rotors cannot steer all axes".into()));
    }
    Ok(normalize(balanced.inverse()))
}

impl Copters {
    pub(super) fn is_empty(&self) -> bool {
        self.copters.is_empty()
    }

    pub fn insert(
        &mut self,
        scene: &RigidScene,
        propellers: &Propellers,
        spec: CopterSpec,
    ) -> Result<CopterId> {
        let l = spec.limits;
        if [
            l.max_tilt,
            l.max_speed,
            l.max_climb,
            l.max_yaw_rate,
            l.velocity_gain,
            l.position_gain,
            l.disturbance_time,
            l.attitude_frequency,
            l.yaw_gain,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
            || l.max_tilt >= std::f64::consts::FRAC_PI_2
        {
            return Err(Error::Build(
                "copter gains and bounds must be positive, with tilt below pi/2".into(),
            ));
        }
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("copter body is missing".into()));
        }
        if spec.propellers.len() < 4 {
            return Err(Error::Build("a copter needs at least four rotors".into()));
        }
        mixer_inverse(&rotors(propellers, scene, &spec)?)?;
        self.next += 1;
        let id = CopterId(self.next);
        self.copters.insert(
            id,
            Copter {
                spec,
                command: CopterCommand::Off,
                disturbance: Vec3::ZERO,
                last: None,
                reading: CopterReading {
                    command: CopterCommand::Off,
                    roll: 0.0,
                    pitch: 0.0,
                    velocity: [0.0; 3],
                    yaw_rate: 0.0,
                    thrust: 0.0,
                    saturated: false,
                },
            },
        );
        Ok(id)
    }

    pub fn remove(&mut self, id: CopterId) -> bool {
        self.copters.remove(&id).is_some()
    }

    pub fn command(&mut self, id: CopterId, command: CopterCommand) -> Result<()> {
        let finite = match command {
            CopterCommand::Off => true,
            CopterCommand::Trajectory {
                position,
                velocity,
                acceleration,
                heading,
            } => {
                [position, velocity, acceleration, heading]
                    .iter()
                    .all(|v| v.is_finite())
                    && heading.length_squared() > 1e-12
            }
            CopterCommand::Velocity {
                forward,
                left,
                up,
                yaw_rate,
            } => [forward, left, up, yaw_rate].iter().all(|v| v.is_finite()),
            CopterCommand::Attitude {
                roll,
                pitch,
                yaw_rate,
                climb,
            } => [roll, pitch, yaw_rate, climb].iter().all(|v| v.is_finite()),
        };
        if !finite {
            return Err(Error::Build("copter command must be finite".into()));
        }
        let copter = self
            .copters
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such copter".into()))?;
        if copter.command == CopterCommand::Off {
            copter.disturbance = Vec3::ZERO;
            copter.last = None;
        }
        copter.command = command;
        Ok(())
    }

    pub fn reading(&self, id: CopterId) -> Option<CopterReading> {
        self.copters.get(&id).map(|c| c.reading)
    }

    /// Sets every controlled rotor's speed for the coming step of `dt`.
    pub(crate) fn before_step(
        &mut self,
        scene: &RigidScene,
        propellers: &mut Propellers,
        dt: Real,
    ) -> Result<()> {
        for copter in self.copters.values_mut() {
            if scene.body(copter.spec.body).is_none() {
                continue;
            }
            let rotors = rotors(propellers, scene, &copter.spec)?;
            let measured_force = rotors
                .iter()
                .filter_map(|r| propellers.reading(r.id))
                .map(|r| r.world_force)
                .sum();
            let speeds = control(scene, copter, &rotors, measured_force, dt)?;
            for (rotor, omega) in rotors.iter().zip(speeds) {
                propellers.set_speed(rotor.id, omega)?;
            }
        }
        Ok(())
    }
}

/// Largest disturbance the observer cancels, in g along up and in the tilt
/// limit's reach across it.
const MAX_LIFT_DISTURBANCE: Real = 0.3;

/// Unit vector along the part of `v` perpendicular to `up`.
fn horizontal(v: Vec3, up: Vec3) -> Option<Vec3> {
    (v - up * v.dot(up)).try_normalize()
}

/// Shortest quaternion attitude error in the current flight frame.
fn rotation_error(desired: Mat3, current: Mat3) -> Vec3 {
    let q = Quat::from_mat3(&(desired.transpose() * current)).normalize();
    Vec3::new(q.x, q.y, q.z) * (if q.w < 0.0 { -2.0 } else { 2.0 })
}

/// Rotor speeds for this step from the body state and the command.
fn control(
    scene: &RigidScene,
    copter: &mut Copter,
    rotors: &[Rotor],
    measured_force: Vec3,
    dt: Real,
) -> Result<Vec<Real>> {
    let spec = &copter.spec;
    let limits = spec.limits;
    let pose = scene.body_pose(spec.body)?;
    let velocity = scene.body_velocity(spec.body)?;
    let mass = scene.body_mass_properties(spec.body)?;
    let gravity = scene.gravity();
    let g = gravity.length();
    let up = if g > 0.0 { -gravity / g } else { Vec3::Z };
    let rotation: Quat = pose.rotation * spec.frame.rotation;
    let current = Mat3::from_quat(rotation);
    let [fx, fy, fz] = [current.x_axis, current.y_axis, current.z_axis];
    let heading = horizontal(fx, up)
        .or_else(|| horizontal(-fz, up))
        .unwrap_or(Vec3::X);
    let left = up.cross(heading);
    let v = velocity.linear;
    let rate = rotation.conjugate() * velocity.angular;
    // What the air, payload or ground added to the thrust and gravity asked for.
    if let Some((last_v, last_dt)) = copter.last {
        let observed = (v - last_v) / last_dt - measured_force / mass.mass - gravity;
        let blend = 1.0 - (-last_dt / limits.disturbance_time.max(last_dt)).exp();
        let estimate = copter.disturbance + (observed - copter.disturbance) * blend;
        let along = estimate
            .dot(up)
            .clamp(-MAX_LIFT_DISTURBANCE * g, MAX_LIFT_DISTURBANCE * g);
        let across = (estimate - up * estimate.dot(up)).clamp_length_max(g * limits.max_tilt.tan());
        copter.disturbance = across + up * along;
    }
    copter.reading = CopterReading {
        command: copter.command,
        roll: fy.dot(up).atan2(fz.dot(up)),
        pitch: (-fx.dot(up)).clamp(-1.0, 1.0).asin(),
        velocity: [v.dot(heading), v.dot(left), v.dot(up)],
        yaw_rate: velocity.angular.dot(up),
        thrust: 0.0,
        saturated: false,
    };

    let (desired, thrust, yaw_rate) = match copter.command {
        CopterCommand::Off => {
            copter.last = None;
            return Ok(vec![0.0; rotors.len()]);
        }
        CopterCommand::Trajectory {
            position,
            velocity,
            acceleration,
            heading: reference_heading,
        } => {
            let correction = (position - pose.position) * limits.position_gain;
            let target = velocity + correction;
            let horizontal_v = (target - up * target.dot(up)).clamp_length_max(limits.max_speed);
            let target =
                horizontal_v + up * target.dot(up).clamp(-limits.max_climb, limits.max_climb);
            let force =
                acceleration + (target - v) * limits.velocity_gain - copter.disturbance - gravity;
            let vertical = force.dot(up).max(0.1 * g);
            let lateral =
                (force - up * force.dot(up)).clamp_length_max(vertical * limits.max_tilt.tan());
            let force = lateral + up * vertical;
            let z = force.normalize();
            let desired_heading = horizontal(reference_heading, up).unwrap_or(heading);
            let y = z.cross(desired_heading).try_normalize().unwrap_or(left);
            (
                Mat3::from_cols(y.cross(z), y, z),
                mass.mass * force.dot(fz).max(0.0),
                0.0,
            )
        }
        CopterCommand::Velocity {
            forward,
            left: side,
            up: climb,
            yaw_rate,
        } => {
            let mut horizontal_v = heading * forward + left * side;
            let speed = horizontal_v.length();
            if speed > limits.max_speed {
                horizontal_v *= limits.max_speed / speed;
            }
            let target = horizontal_v + up * climb.clamp(-limits.max_climb, limits.max_climb);
            let mut force = (target - v) * limits.velocity_gain - copter.disturbance;
            let vertical = force.dot(up) + g;
            let mut lateral = force - up * force.dot(up);
            let max_lateral = vertical.max(0.0) * limits.max_tilt.tan();
            if lateral.length() > max_lateral {
                lateral *= max_lateral / lateral.length();
            }
            force = lateral + up * vertical.max(0.1 * g);
            let z = force.normalize();
            let y = z.cross(heading).try_normalize().unwrap_or(left);
            let desired = Mat3::from_cols(y.cross(z), y, z);
            (desired, mass.mass * force.dot(fz).max(0.0), yaw_rate)
        }
        CopterCommand::Attitude {
            roll,
            pitch,
            yaw_rate,
            climb,
        } => {
            let tilt = |a: Real| a.clamp(-limits.max_tilt, limits.max_tilt);
            let level = Mat3::from_cols(heading, left, up);
            let desired =
                level * Mat3::from_rotation_y(tilt(pitch)) * Mat3::from_rotation_x(tilt(roll));
            let climb = climb.clamp(-limits.max_climb, limits.max_climb);
            let vertical = g + (climb - v.dot(up)) * limits.velocity_gain;
            let lift = fz.dot(up).max(0.2);
            (desired, mass.mass * vertical.max(0.0) / lift, yaw_rate)
        }
    };

    let omega_n = limits.attitude_frequency;
    let frame = Mat3::from_quat(spec.frame.rotation);
    let inertia = frame.transpose() * mass.inertia * frame;
    let inverse = mixer_inverse(rotors)?;
    let collective = inverse * DVec4::new(thrust, 0.0, 0.0, 0.0);
    let rate_gain = 2.0 * omega_n;
    let attitude_gain = omega_n * omega_n / rate_gain;
    let error = rotation_error(desired, current);
    let mut wanted = -error * attitude_gain;
    if matches!(copter.command, CopterCommand::Trajectory { .. }) {
        let tilt = Quat::from_rotation_arc(fz, desired.z_axis);
        wanted = rotation.conjugate() * Vec3::new(tilt.x, tilt.y, tilt.z) * (2.0 * attitude_gain);
        let target_heading = horizontal(desired.x_axis, up).unwrap_or(heading);
        let yaw_error = up
            .dot(heading.cross(target_heading))
            .atan2(heading.dot(target_heading));
        let unit_yaw = inverse * DVec4::W;
        let authority = rotors
            .iter()
            .filter_map(|r| {
                let coefficient = r.column.dot(unit_yaw).abs();
                let base = (r.column.dot(collective) * r.sign).clamp(0.0, r.max_s);
                (coefficient > 1e-12).then_some(base.min(r.max_s - base) / coefficient)
            })
            .fold(f64::INFINITY, f64::min);
        let braking_rate = (authority / inertia.z_axis.z * yaw_error.abs()).sqrt();
        let bound = limits.max_yaw_rate.min(braking_rate);
        wanted.z = (yaw_error * attitude_gain).clamp(-bound, bound);
    } else {
        wanted.z = yaw_rate.clamp(-limits.max_yaw_rate, limits.max_yaw_rate) * up.dot(fz).max(0.0);
    }
    let gains = Vec3::new(rate_gain, rate_gain, limits.yaw_gain);
    let accel = (wanted - rate) * gains;
    let torque = inertia * accel + rate.cross(inertia * rate);

    let mut saturated = false;
    let mut squared: Vec<_> = rotors
        .iter()
        .map(|r| {
            let requested = r.column.dot(collective) * r.sign;
            let bounded = requested.clamp(0.0, r.max_s);
            saturated |= bounded != requested;
            bounded
        })
        .collect();
    for demand in [
        DVec4::new(0.0, torque.x, torque.y, 0.0),
        DVec4::new(0.0, 0.0, 0.0, torque.z),
    ] {
        let lambda = inverse * demand;
        let increments: Vec<_> = rotors
            .iter()
            .map(|r| r.column.dot(lambda) * r.sign)
            .collect();
        let mut scale = 1.0_f64;
        for ((r, base), delta) in rotors.iter().zip(&squared).zip(&increments) {
            if *delta > 0.0 {
                scale = scale.min((r.max_s - base) / delta);
            } else if *delta < 0.0 {
                scale = scale.min(-base / delta);
            }
        }
        scale = scale.clamp(0.0, 1.0);
        saturated |= scale < 1.0 - 1e-12;
        for (base, delta) in squared.iter_mut().zip(increments) {
            *base += scale * delta;
        }
    }
    let speeds = rotors
        .iter()
        .zip(squared)
        .map(|(r, s)| r.sign * s.clamp(0.0, r.max_s).sqrt())
        .collect();
    copter.last = Some((v, dt));
    copter.reading.thrust = thrust;
    copter.reading.saturated = saturated;
    Ok(speeds)
}
