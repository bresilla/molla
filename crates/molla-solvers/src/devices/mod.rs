//! Actuator devices of the rigid world, after Webots: joint motors and
//! brakes, propellers, belts (conveyors and tank tracks), connectors,
//! spacecraft thrusters, reaction wheels, orbital force fields and 6-DOF
//! spacecraft controllers.
//!
//! Each device is part of the physics step. Motors turn a position,
//! velocity or force command into the joint drive; brakes damp a joint
//! axis; propellers push their body; belts move their colliders' surfaces;
//! connectors dock bodies with a breakable joint; copter controllers fly a
//! multirotor on its propellers; the air drags bodies and blows with the
//! wind; thrusters push spacecraft with optional valve lag, MIB and propellant
//! depletion; reaction wheels exchange angular momentum with exact gyroscopic
//! coupling; orbital fields model Clohessy–Wiltshire and central gravity; and
//! spacecraft controllers fly 6-DOF free-flyers. Every device reports what it
//! did in the last step.

mod aero;
mod belts;
mod connectors;
mod copter;
mod motors;
mod orbital;
mod propellers;
mod reaction_wheels;
mod rocket;
mod rocket_aero;
mod rocket_structure;
mod sensors;
mod space_env;
mod spacecraft;
mod thrusters;

pub use aero::{AIR_DENSITY, Aero, DragId, DragReading, DragSpec};
pub use belts::{BeltId, BeltReading, BeltSpec, Belts};
pub use connectors::{ConnectorId, ConnectorKind, ConnectorReading, ConnectorSpec, Connectors};
pub use copter::{CopterCommand, CopterId, CopterLimits, CopterReading, CopterSpec, Copters};
pub use motors::{MotorKey, MotorReading, MotorSpec, Motors};
pub use orbital::{
    EARTH_EQUATORIAL_RADIUS, EARTH_J2, EARTH_J3, EARTH_J4, EARTH_MU, EccentricOrbit,
    J2Perturbation, MARS_MU, MOON_MU, Orbital, OrbitalField, OrbitalReading, YamanakaAnkersen,
    ZonalHarmonics,
};
pub use propellers::{PropellerId, PropellerReading, PropellerSpec, Propellers};
pub use reaction_wheels::{
    GimbalSpec, ImbalanceSpec, ReactionWheelCommand, ReactionWheelId, ReactionWheelReading,
    ReactionWheelSpec, ReactionWheels,
};
pub use rocket::{
    AtmosphereState, EARTH_OMEGA_RAD_S, PropellantTankReading, PropellantTankSpec,
    RocketEngineCommand, RocketEngineId, RocketEngineReading, RocketEngineSpec, RocketPropulsion,
    TankId, earth_rotation_at_latitude, us76_atmosphere,
};
pub use rocket_aero::{
    AeroBodyId, AeroBodyReading, AeroBodySpec, AeroSurfaceId, AeroSurfaceKind, AeroSurfaceReading,
    AeroSurfaceSpec, RocketAero, WindField,
};
pub use rocket_structure::{
    BiquadNotchFilter, FlexBodyId, FlexBodyReading, FlexBodySpec, RocketStructure,
    free_free_beam_mode,
};
pub use sensors::{
    ImuId, ImuReading, ImuSpec, LidarId, LidarReading, LidarSpec, NavigationFilter, SpaceSensors,
    StarTrackerId, StarTrackerReading, StarTrackerSpec,
};
pub use space_env::{
    EARTH_MAGNETIC_MU0_M_OVER_4PI, FacetId, FacetReading, FlexJointId, FlexJointReading,
    FlexJointSpec, MagneticField, MagnetorquerId, MagnetorquerReading, MagnetorquerSpec,
    NrlmsiseAtmosphere, OrbitalAtmosphere, SOLAR_PRESSURE_1AU, SloshTankId, SloshTankReading,
    SloshTankSpec, SolarActivity, SolarRadiation, SpaceEnvironment, SurfaceFacetSpec,
    desaturation_dipole_body, nrlmsise_density_kg_m3,
};
pub use spacecraft::{
    SpacecraftCommand, SpacecraftControllers, SpacecraftId, SpacecraftLimits, SpacecraftReading,
    SpacecraftSpec,
};
pub use thrusters::{
    ThrusterId, ThrusterModulation, ThrusterReading, ThrusterSpec, Thrusters,
};

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
    pub thrusters: Thrusters,
    pub reaction_wheels: ReactionWheels,
    pub orbital: Orbital,
    pub space_env: SpaceEnvironment,
    pub space_sensors: SpaceSensors,
    pub rockets: RocketPropulsion,
    pub rocket_aero: RocketAero,
    pub rocket_structure: RocketStructure,
    pub spacecraft: SpacecraftControllers,
    /// Loop-joint reactions of the last step, averaged over its substeps.
    reactions: BTreeMap<JointHandle, LoopReaction>,
    last_dt: Real,
}

impl Devices {
    pub fn is_empty(&self) -> bool {
        self.motors.is_empty()
            && self.propellers.is_empty()
            && self.belts.is_empty()
            && self.connectors.is_empty()
            && self.copters.is_empty()
            && !self.aero.active()
            && self.thrusters.is_empty()
            && self.reaction_wheels.is_empty()
            && !self.orbital.active()
            && self.space_env.is_empty()
            && self.space_sensors.is_empty()
            && self.rockets.is_empty()
            && self.rocket_aero.is_empty()
            && self.rocket_structure.is_empty()
            && self.spacecraft.is_empty()
    }

    /// Whether a device needs the per-substep force path this step.
    pub fn needs_forces(&self) -> bool {
        self.propellers.active()
            || self.aero.active()
            || self.thrusters.active()
            || self.reaction_wheels.active()
            || self.orbital.active()
            || self.space_env.active()
            || self.rockets.active()
            || self.rocket_aero.active()
            || self.rocket_structure.active()
            || self.spacecraft.active()
    }

    /// Commands the scene for the coming step of `dt`: connectors dock,
    /// undock or break, motors and belts set their drives, copters their
    /// rotors, spacecraft controllers their thrusters and reaction wheels.
    pub fn before_step(&mut self, scene: &mut RigidScene, dt: Real) -> Result<()> {
        self.last_dt = dt;
        self.connectors.update(scene, &self.reactions)?;
        self.motors.before_step(scene, dt)?;
        self.belts.before_step(scene, dt)?;
        self.copters.before_step(scene, &mut self.propellers, dt)?;
        self.propellers.before_step(dt);
        self.spacecraft.before_step(
            scene,
            &mut self.thrusters,
            &mut self.reaction_wheels,
            dt,
        )?;
        self.thrusters.before_step(scene, dt)?;
        self.reaction_wheels.before_step();
        self.orbital.before_step();
        self.space_env.before_step(scene)?;
        self.rockets.before_step(scene, dt)?;
        self.rocket_aero.before_step(dt);
        self.aero.before_step();
        Ok(())
    }

    /// Adds the propellers', air's, thrusters', reaction wheels', orbital
    /// fields' and LEO space environment's wrenches for one substep of `sub_dt`.
    pub fn apply(
        &mut self,
        scene: &RigidScene,
        state: &State,
        wrenches: &mut [SpatialVector],
        sub_dt: Real,
        weight: Real,
    ) -> Result<()> {
        let aero = &self.aero;
        self.propellers
            .apply(scene, state, wrenches, weight, &|body| aero.wind_at(body))?;
        self.aero.apply(scene, state, wrenches, weight)?;
        self.thrusters.apply(scene, state, wrenches, weight)?;
        self.reaction_wheels
            .apply(scene, state, wrenches, sub_dt, weight)?;
        self.orbital.apply(scene, state, wrenches, sub_dt, weight)?;
        self.space_env.apply(
            scene,
            state,
            &self.thrusters,
            &self.orbital,
            wrenches,
            sub_dt,
            weight,
        )?;
        self.rockets.apply(scene, state, wrenches, sub_dt, weight)?;
        self.rocket_aero
            .apply(scene, state, &self.rockets, wrenches, weight)?;
        self.rocket_structure.apply(
            scene,
            state,
            &self.rockets,
            &self.rocket_aero,
            wrenches,
            sub_dt,
            weight,
        )
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

    /// Records the step's joint efforts for motors, flexible appendages, and updates
    /// hardware spacecraft sensors (IMU, Star Tracker, LiDAR).
    pub fn after_step(&mut self, scene: &RigidScene, efforts: &[JointEffort]) -> Result<()> {
        self.motors.after_step(scene, efforts)?;
        self.space_env.after_step(scene, efforts)?;
        self.space_sensors.after_step(
            scene,
            &self.orbital,
            &self.space_env,
            &self.rocket_structure,
            self.last_dt,
        )
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
