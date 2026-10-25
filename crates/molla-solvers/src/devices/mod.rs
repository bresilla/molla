//! Actuator devices of the rigid world, after Webots: joint motors and
//! brakes, propellers, belts (conveyors and tank tracks) and connectors.
//!
//! Each device is part of the physics step. Motors turn a position,
//! velocity or force command into the joint drive; brakes damp a joint
//! axis; propellers push their body; belts move their colliders' surfaces;
//! connectors dock bodies with a breakable joint; copter controllers fly a
//! multirotor on its propellers; the air drags bodies and blows with the
//! wind. Every device reports what it did in the last step.

mod aero;
mod belts;
mod connectors;
mod copter;
mod motors;
mod propellers;

pub use aero::{AIR_DENSITY, Aero, DragId, DragReading, DragSpec};
pub use belts::{BeltId, BeltReading, BeltSpec, Belts};
pub use connectors::{ConnectorId, ConnectorKind, ConnectorReading, ConnectorSpec, Connectors};
pub use copter::{CopterCommand, CopterId, CopterLimits, CopterReading, CopterSpec, Copters};
pub use motors::{MotorKey, MotorReading, MotorSpec, Motors};
pub use propellers::{PropellerId, PropellerReading, PropellerSpec, Propellers};

use std::collections::BTreeMap;

use molla_core::Result;
use molla_math::{Real, SpatialVector};
use molla_sim::State;
use molla_sim::runtime::{JointHandle, RigidScene};

use crate::featherstone::JointEffort;
use crate::loop_forces::LoopReaction;

/// How near its target (rad or m) a position-controlled motor stops.
pub const HOLD_TOLERANCE: Real = 1e-5;

/// What a motor is told to do.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MotorCommand {
    /// Reach a position (rad or m) at up to the motor's velocity.
    Position(Real),
    /// Run at a velocity (rad/s or m/s).
    Velocity(Real),
    /// Apply a force or torque (N or N·m), without the controller.
    Force(Real),
}

/// Limits and gains of a Webots-style motor controller.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MotorLimits {
    /// Largest speed, also the default speed of position control.
    pub max_velocity: Real,
    /// Largest change of speed per second; `None` is unlimited.
    pub acceleration: Option<Real>,
    /// PID gains on the position error, giving a speed.
    pub pid: [Real; 3],
    /// Soft position limits the target is clipped to.
    pub position_limits: Option<[Real; 2]>,
}

impl Default for MotorLimits {
    fn default() -> Self {
        Self {
            max_velocity: 10.0,
            acceleration: None,
            pid: [10.0, 0.0, 0.0],
            position_limits: None,
        }
    }
}

/// The controller state of one motor: its command, the speed it runs
/// position control at, and the PID and acceleration memory.
#[derive(Clone, Copy, Debug)]
pub(crate) struct MotorController {
    pub limits: MotorLimits,
    pub command: MotorCommand,
    /// Speed position control runs at (Webots `set_velocity`).
    pub speed: Real,
    integral: Real,
    previous_error: Option<Real>,
    /// Speed commanded last step.
    pub velocity: Real,
}

impl MotorController {
    pub fn new(limits: MotorLimits) -> Self {
        Self {
            limits,
            command: MotorCommand::Velocity(0.0),
            speed: limits.max_velocity,
            integral: 0.0,
            previous_error: None,
            velocity: 0.0,
        }
    }

    pub fn set_command(&mut self, command: MotorCommand) {
        let command = match command {
            MotorCommand::Position(p) => {
                MotorCommand::Position(match self.limits.position_limits {
                    Some([lo, hi]) => p.clamp(lo, hi),
                    None => p,
                })
            }
            MotorCommand::Velocity(v) => {
                MotorCommand::Velocity(v.clamp(-self.limits.max_velocity, self.limits.max_velocity))
            }
            other => other,
        };
        if !matches!(
            (self.command, command),
            (MotorCommand::Position(_), MotorCommand::Position(_))
        ) {
            self.integral = 0.0;
            self.previous_error = None;
        }
        self.command = command;
    }

    pub fn set_pid(&mut self, pid: [Real; 3]) {
        self.limits.pid = pid;
        self.integral = 0.0;
        self.previous_error = None;
    }

    /// The speed to run at this step from the current `position`, or
    /// `None` under force control.
    pub fn velocity(&mut self, position: Real, dt: Real) -> Option<Real> {
        let target = match self.command {
            MotorCommand::Force(_) => return None,
            MotorCommand::Velocity(v) => v,
            MotorCommand::Position(target) => {
                let error = target - position;
                self.integral += error * dt;
                let derivative = self
                    .previous_error
                    .map_or(0.0, |previous| (error - previous) / dt);
                self.previous_error = Some(error);
                let [p, i, d] = self.limits.pid;
                let speed = self.speed.abs().min(self.limits.max_velocity);
                // At the target the motor holds still rather than creeping
                // at a speed that only tends to zero.
                if error.abs() <= HOLD_TOLERANCE {
                    0.0
                } else {
                    (p * error + i * self.integral + d * derivative).clamp(-speed, speed)
                }
            }
        };
        let velocity = match self.limits.acceleration {
            Some(a) => self.velocity + (target - self.velocity).clamp(-a * dt, a * dt),
            None => target,
        };
        self.velocity = velocity;
        Some(velocity)
    }

    /// [`Self::velocity`] of a joint whose travel ends at `stops` (`NaN`
    /// for no end): a position target past a stop is held at the stop, and
    /// no speed drives the joint further into one.
    pub fn velocity_within(&mut self, position: Real, dt: Real, stops: [Real; 2]) -> Option<Real> {
        let command = self.command;
        if let MotorCommand::Position(target) = command {
            let below = if stops[0].is_nan() { target } else { target.max(stops[0]) };
            self.command = MotorCommand::Position(if stops[1].is_nan() { below } else { below.min(stops[1]) });
        }
        let velocity = self.velocity(position, dt);
        self.command = command;
        let velocity = velocity?;
        let pushing = (velocity < 0.0 && position <= stops[0] + HOLD_TOLERANCE)
            || (velocity > 0.0 && position >= stops[1] - HOLD_TOLERANCE);
        if pushing {
            self.velocity = 0.0;
            return Some(0.0);
        }
        Some(velocity)
    }
}

/// Every actuator device of a rigid world.
#[derive(Default)]
pub struct Devices {
    pub motors: Motors,
    pub propellers: Propellers,
    pub belts: Belts,
    pub connectors: Connectors,
    pub copters: Copters,
    pub aero: Aero,
    /// Loop-joint reactions of the last step, averaged over its substeps.
    reactions: BTreeMap<JointHandle, LoopReaction>,
}

impl Devices {
    pub fn is_empty(&self) -> bool {
        self.motors.is_empty()
            && self.propellers.is_empty()
            && self.belts.is_empty()
            && self.connectors.is_empty()
            && self.copters.is_empty()
            && !self.aero.active()
    }

    /// Whether a device needs the per-substep force path this step.
    pub fn needs_forces(&self) -> bool {
        self.propellers.active() || self.aero.active()
    }

    /// Commands the scene for the coming step of `dt`: connectors dock,
    /// undock or break, motors and belts set their drives, copters their
    /// rotors.
    pub fn before_step(&mut self, scene: &mut RigidScene, dt: Real) -> Result<()> {
        self.connectors.update(scene, &self.reactions)?;
        self.motors.before_step(scene, dt)?;
        self.belts.before_step(scene, dt)?;
        self.copters.before_step(scene, &mut self.propellers, dt)?;
        self.propellers.before_step(dt);
        self.aero.before_step();
        Ok(())
    }

    /// Adds the propellers' and the air's wrenches for one substep.
    pub fn apply(
        &mut self,
        scene: &RigidScene,
        state: &State,
        wrenches: &mut [SpatialVector],
        weight: Real,
    ) -> Result<()> {
        let aero = &self.aero;
        self.propellers
            .apply(scene, state, wrenches, weight, &|body| aero.wind_at(body))?;
        self.aero.apply(scene, state, wrenches, weight)
    }

    /// Records one substep's loop-joint reactions with `weight`.
    pub fn record_reactions(
        &mut self,
        reactions: &[(JointHandle, LoopReaction)],
        weight: Real,
        first: bool,
    ) {
        if first {
            self.reactions.clear();
        }
        for (joint, reaction) in reactions {
            let total = self.reactions.entry(*joint).or_default();
            total.force += reaction.force * weight;
            total.torque += reaction.torque * weight;
        }
    }

    /// Records the step's joint efforts for the motor and brake readings.
    pub fn after_step(&mut self, scene: &RigidScene, efforts: &[JointEffort]) -> Result<()> {
        self.motors.after_step(scene, efforts)
    }

    /// A loop joint's reaction on its child in the last step.
    pub fn reaction(&self, joint: JointHandle) -> Option<LoopReaction> {
        self.reactions.get(&joint).copied()
    }

    pub(crate) fn replace_reactions(
        &mut self,
        reactions: BTreeMap<JointHandle, LoopReaction>,
    ) -> BTreeMap<JointHandle, LoopReaction> {
        std::mem::replace(&mut self.reactions, reactions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_control_follows_webots_p_controller() {
        let mut c = MotorController::new(MotorLimits {
            max_velocity: 2.0,
            acceleration: Some(10.0),
            ..Default::default()
        });
        c.set_command(MotorCommand::Position(1.0));
        let v = c.velocity(0.0, 0.01).unwrap();
        assert!((v - 0.1).abs() < 1e-12, "acceleration-limited: {v}");
        c.velocity = 2.0;
        assert!(
            (c.velocity(0.0, 0.01).unwrap() - 2.0).abs() < 1e-12,
            "clamped at max velocity"
        );
        assert!(
            (c.velocity(0.95, 0.01).unwrap() - 1.9).abs() < 1e-12,
            "decelerates at the limit"
        );
        c.set_command(MotorCommand::Force(3.0));
        assert!(c.velocity(0.0, 0.01).is_none());
    }

    #[test]
    fn position_control_stops_dead_within_the_hold_tolerance() {
        let mut c = MotorController::new(MotorLimits::default());
        c.set_command(MotorCommand::Position(1.0));
        assert!((c.velocity(1.0 - 1e-3, 0.01).unwrap() - 1e-2).abs() < 1e-9);
        assert_eq!(c.velocity(1.0 - HOLD_TOLERANCE * 0.5, 0.01), Some(0.0));
        assert_eq!(c.velocity(1.0 + HOLD_TOLERANCE * 0.5, 0.01), Some(0.0));
    }

    #[test]
    fn a_motor_holds_at_its_stops_instead_of_pushing_into_them() {
        let mut c = MotorController::new(MotorLimits::default());
        c.set_command(MotorCommand::Position(2.0));
        assert_eq!(c.velocity_within(1.0, 0.01, [f64::NAN, 1.0]), Some(0.0));
        assert!(c.velocity_within(0.5, 0.01, [f64::NAN, 1.0]).unwrap() > 0.0);
        assert!((c.velocity_within(0.99, 0.01, [0.0, 1.0]).unwrap() - 0.1).abs() < 1e-9);
        c.set_command(MotorCommand::Velocity(-3.0));
        assert_eq!(c.velocity_within(0.0, 0.01, [0.0, 1.0]), Some(0.0));
        assert_eq!(c.velocity_within(0.5, 0.01, [0.0, 1.0]), Some(-3.0));
        assert_eq!(c.velocity_within(0.5, 0.01, [f64::NAN, f64::NAN]), Some(-3.0));
        assert_eq!(c.command, MotorCommand::Velocity(-3.0));
    }

    #[test]
    fn soft_limits_clip_the_target_and_velocity_the_speed() {
        let mut c = MotorController::new(MotorLimits {
            position_limits: Some([-0.5, 0.5]),
            max_velocity: 1.0,
            ..Default::default()
        });
        c.set_command(MotorCommand::Position(2.0));
        assert_eq!(c.command, MotorCommand::Position(0.5));
        c.set_command(MotorCommand::Velocity(-3.0));
        assert_eq!(c.command, MotorCommand::Velocity(-1.0));
    }
}
