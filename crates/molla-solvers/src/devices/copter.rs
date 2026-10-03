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
    /// Velocity and thrust acceleration at the last update, which the
    /// observer compares against.
    last: Option<(Vec3, Vec3)>,
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
    let det = gram.determinant();
    if !det.is_finite() || det.abs() < 1e-30 {
        return Err(Error::Build("copter rotors cannot steer all axes".into()));
    }
    Ok(gram.inverse())
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
            let speeds = control(scene, copter, &rotors, dt)?;
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

/// `½·vee(Aᵀ·B − Bᵀ·A)`: the rotation error from `A` to `B` in `B`'s frame.
fn rotation_error(desired: Mat3, current: Mat3) -> Vec3 {
    let m = desired.transpose() * current - current.transpose() * desired;
    Vec3::new(
        m.y_axis.z - m.z_axis.y,
        m.z_axis.x - m.x_axis.z,
        m.x_axis.y - m.y_axis.x,
    ) * 0.25
}

/// Rotor speeds for this step from the body state and the command.
fn control(
    scene: &RigidScene,
    copter: &mut Copter,
    rotors: &[Rotor],
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
    if let Some((last_v, last_thrust)) = copter.last {
        let observed = (v - last_v) / dt - last_thrust - gravity;
        let blend = (dt / limits.disturbance_time.max(dt)).min(1.0);
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
            copter.last = Some((v, Vec3::ZERO));
            return Ok(vec![0.0; rotors.len()]);
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
    let rate_gain = 2.0 * omega_n;
    let attitude_gain = omega_n * omega_n / rate_gain;
    let error = rotation_error(desired, current);
    let mut wanted = -error * attitude_gain;
    wanted.z = yaw_rate.clamp(-limits.max_yaw_rate, limits.max_yaw_rate) * up.dot(fz).max(0.0);
    let gains = Vec3::new(rate_gain, rate_gain, limits.yaw_gain);
    let accel = (wanted - rate) * gains;
    let frame = Mat3::from_quat(spec.frame.rotation);
    let inertia = frame.transpose() * mass.inertia * frame;
    let torque = inertia * accel + rate.cross(inertia * rate);

    let inverse = mixer_inverse(rotors)?;
    let lambda = inverse * DVec4::new(thrust, torque.x, torque.y, torque.z);
    let mut saturated = false;
    let mut delivered = 0.0;
    let speeds = rotors
        .iter()
        .map(|r| {
            let s = r.column.dot(lambda);
            let bounded = (s * r.sign).clamp(0.0, r.max_s);
            saturated |= bounded != s * r.sign;
            delivered += r.column.x * r.sign * bounded;
            r.sign * bounded.sqrt()
        })
        .collect();
    copter.last = Some((v, fz * (delivered / mass.mass)));
    copter.reading.thrust = thrust;
    copter.reading.saturated = saturated;
    Ok(speeds)
}
