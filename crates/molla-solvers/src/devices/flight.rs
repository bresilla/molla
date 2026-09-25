//! Multirotor flight controllers: every step they read their body's state
//! and set their rotors' speeds to follow a commanded velocity or attitude.
//!
//! The loops cascade: velocity to a desired thrust direction and magnitude,
//! attitude (on SO(3)) to body rates, rates to torque through the body's
//! inertia. A mixer built from the rotors' positions, shafts and constants
//! turns the collective thrust and torque into each rotor's `|ω|·ω`.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use glam::{DMat4, DVec4};
use molla_math::{Mat3, Quat, Real, Transform, Vec3};
use molla_sim::runtime::{BodyHandle, RigidScene};

use super::{PropellerId, Propellers};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct FlightId(pub u64);

/// What a flight controller holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum FlightCommand {
    /// Rotors stopped.
    Off,
    /// Velocity in the heading frame (forward, left, up; m/s) and yaw rate
    /// about up (rad/s, counter-clockwise).
    Velocity { forward: Real, left: Real, up: Real, yaw_rate: Real },
    /// Roll (right side down) and pitch (nose down) in rad, yaw rate and
    /// climb rate.
    Attitude { roll: Real, pitch: Real, yaw_rate: Real, climb: Real },
}

/// Limits and loop gains of a flight controller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlightLimits {
    /// Largest tilt from level (rad).
    pub max_tilt: Real,
    /// Largest horizontal speed and climb rate (m/s).
    pub max_speed: Real,
    pub max_climb: Real,
    pub max_yaw_rate: Real,
    /// Velocity loop gain (1/s).
    pub velocity_gain: Real,
    /// Attitude loop natural frequency (rad/s), critically damped.
    pub attitude_frequency: Real,
    /// Yaw-rate loop gain (1/s).
    pub yaw_gain: Real,
}

impl Default for FlightLimits {
    fn default() -> Self {
        Self {
            max_tilt: 0.5,
            max_speed: 10.0,
            max_climb: 3.0,
            max_yaw_rate: 1.5,
            velocity_gain: 1.5,
            attitude_frequency: 6.0,
            yaw_gain: 4.0,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct FlightSpec {
    pub body: BodyHandle,
    /// Flight frame in the body frame: +X forward, +Y left, +Z up.
    pub frame: Transform,
    /// Rotors on `body`; at least four that can steer all four axes.
    pub propellers: Vec<PropellerId>,
    pub limits: FlightLimits,
}

/// A flight controller's view of its body after the last update.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FlightReading {
    pub command: FlightCommand,
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

struct Flight {
    spec: FlightSpec,
    command: FlightCommand,
    reading: FlightReading,
}

/// Flight controllers of a rigid world.
#[derive(Default)]
pub struct Flights {
    next: u64,
    flights: BTreeMap<FlightId, Flight>,
}

/// One rotor in the flight frame: its `[thrust along up, roll, pitch, yaw
/// torque]` per unit `|ω|·ω`, and its speed limit.
struct Rotor {
    id: PropellerId,
    column: DVec4,
    sign: Real,
    max_s: Real,
}

fn rotors(propellers: &Propellers, scene: &RigidScene, spec: &FlightSpec) -> Result<Vec<Rotor>> {
    let mass = scene.body_mass_properties(spec.body)?;
    let to_flight = spec.frame.rotation.conjugate();
    spec.propellers
        .iter()
        .map(|&id| {
            let p = propellers
                .spec(id)
                .ok_or_else(|| Error::Build("flight controller propeller is missing".into()))?;
            if p.body != spec.body {
                return Err(Error::Build("flight controller propellers must be on its body".into()));
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
    let det = gram.determinant();
    if !det.is_finite() || det.abs() < 1e-30 {
        return Err(Error::Build("flight controller rotors cannot steer all axes".into()));
    }
    Ok(gram.inverse())
}

impl Flights {
    pub fn insert(&mut self, scene: &RigidScene, propellers: &Propellers, spec: FlightSpec) -> Result<FlightId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("flight controller body is missing".into()));
        }
        if spec.propellers.len() < 4 {
            return Err(Error::Build("a flight controller needs at least four rotors".into()));
        }
        mixer_inverse(&rotors(propellers, scene, &spec)?)?;
        self.next += 1;
        let id = FlightId(self.next);
        self.flights.insert(id, Flight {
            spec,
            command: FlightCommand::Off,
            reading: FlightReading {
                command: FlightCommand::Off,
                roll: 0.0,
                pitch: 0.0,
                velocity: [0.0; 3],
                yaw_rate: 0.0,
                thrust: 0.0,
                saturated: false,
            },
        });
        Ok(id)
    }

    pub fn remove(&mut self, id: FlightId) -> bool {
        self.flights.remove(&id).is_some()
    }

    pub fn command(&mut self, id: FlightId, command: FlightCommand) -> Result<()> {
        let finite = match command {
            FlightCommand::Off => true,
            FlightCommand::Velocity { forward, left, up, yaw_rate } => {
                [forward, left, up, yaw_rate].iter().all(|v| v.is_finite())
            }
            FlightCommand::Attitude { roll, pitch, yaw_rate, climb } => {
                [roll, pitch, yaw_rate, climb].iter().all(|v| v.is_finite())
            }
        };
        if !finite {
            return Err(Error::Build("flight command must be finite".into()));
        }
        self.flights
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such flight controller".into()))?
            .command = command;
        Ok(())
    }

    pub fn reading(&self, id: FlightId) -> Option<FlightReading> {
        self.flights.get(&id).map(|f| f.reading)
    }

    /// Sets every controlled rotor's speed for the coming step.
    pub(crate) fn before_step(&mut self, scene: &RigidScene, propellers: &mut Propellers) -> Result<()> {
        for flight in self.flights.values_mut() {
            if scene.body(flight.spec.body).is_none() {
                continue;
            }
            let rotors = rotors(propellers, scene, &flight.spec)?;
            let speeds = control(scene, flight, &rotors)?;
            for (rotor, omega) in rotors.iter().zip(speeds) {
                propellers.set_speed(rotor.id, omega)?;
            }
        }
        Ok(())
    }
}

/// Unit vector along the part of `v` perpendicular to `up`.
fn horizontal(v: Vec3, up: Vec3) -> Option<Vec3> {
    (v - up * v.dot(up)).try_normalize()
}

/// `vee((Aᵀ·B − Bᵀ·A)/2)`: the rotation error from `A` to `B` in `B`'s frame.
fn rotation_error(desired: Mat3, current: Mat3) -> Vec3 {
    let m = desired.transpose() * current - current.transpose() * desired;
    Vec3::new(m.y_axis.z - m.z_axis.y, m.z_axis.x - m.x_axis.z, m.x_axis.y - m.y_axis.x) * 0.25
}

/// Rotor speeds for this step from the body state and the command.
fn control(scene: &RigidScene, flight: &mut Flight, rotors: &[Rotor]) -> Result<Vec<Real>> {
    let spec = &flight.spec;
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
    let heading = horizontal(fx, up).or_else(|| horizontal(-fz, up)).unwrap_or(Vec3::X);
    let left = up.cross(heading);
    let v = velocity.linear;
    let rate = rotation.conjugate() * velocity.angular;
    flight.reading = FlightReading {
        command: flight.command,
        roll: fy.dot(up).atan2(fz.dot(up)),
        pitch: (-fx.dot(up)).clamp(-1.0, 1.0).asin(),
        velocity: [v.dot(heading), v.dot(left), v.dot(up)],
        yaw_rate: velocity.angular.dot(up),
        thrust: 0.0,
        saturated: false,
    };

    let (desired, thrust, yaw_rate) = match flight.command {
        FlightCommand::Off => return Ok(vec![0.0; rotors.len()]),
        FlightCommand::Velocity { forward, left: side, up: climb, yaw_rate } => {
            let mut horizontal_v = heading * forward + left * side;
            let speed = horizontal_v.length();
            if speed > limits.max_speed {
                horizontal_v *= limits.max_speed / speed;
            }
            let target = horizontal_v + up * climb.clamp(-limits.max_climb, limits.max_climb);
            let mut force = (target - v) * limits.velocity_gain;
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
        FlightCommand::Attitude { roll, pitch, yaw_rate, climb } => {
            let tilt = |a: Real| a.clamp(-limits.max_tilt, limits.max_tilt);
            let level = Mat3::from_cols(heading, left, up);
            let desired = level * Mat3::from_rotation_y(tilt(pitch)) * Mat3::from_rotation_x(tilt(roll));
            let climb = climb.clamp(-limits.max_climb, limits.max_climb);
            let vertical = g + (climb - v.dot(up)) * limits.velocity_gain;
            let lift = fz.dot(up).max(0.2);
            (desired, mass.mass * vertical.max(0.0) / lift, yaw_rate)
        }
    };

    let omega_n = limits.attitude_frequency;
    let rate_gain = 2.0 * omega_n;
    let attitude_gain = omega_n * omega_n / rate_gain;
    let error = rotation_error(desired, current);
    let mut wanted = -error * attitude_gain;
    wanted.z = yaw_rate.clamp(-limits.max_yaw_rate, limits.max_yaw_rate) * up.dot(fz).max(0.0);
    let gains = Vec3::new(rate_gain, rate_gain, limits.yaw_gain);
    let accel = (wanted - rate) * gains;
    let inertia = Mat3::from_quat(spec.frame.rotation).transpose() * mass.inertia * Mat3::from_quat(spec.frame.rotation);
    let torque = inertia * accel + rate.cross(inertia * rate);

    let inverse = mixer_inverse(rotors)?;
    let wrench = DVec4::new(thrust, torque.x, torque.y, torque.z);
    let lambda = inverse * wrench;
    let mut saturated = false;
    let speeds = rotors
        .iter()
        .map(|r| {
            let s = r.column.dot(lambda);
            let bounded = (s * r.sign).clamp(0.0, r.max_s);
            saturated |= bounded != s * r.sign;
            r.sign * bounded.sqrt()
        })
        .collect();
    flight.reading.thrust = thrust;
    flight.reading.saturated = saturated;
    Ok(speeds)
}
