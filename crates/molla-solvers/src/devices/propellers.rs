//! Propellers: thrust and reaction torque on their body from rotor speed
//! and the speed of advance (Fossen's model, as Webots uses it).

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{Real, SpatialVector, Transform, Vec3};
use molla_sim::State;
use molla_sim::runtime::{BodyHandle, BodyKind, RigidScene};

use super::{MotorCommand, MotorController, MotorLimits};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct PropellerId(pub u64);

/// A propeller on `body`, driven by a rotor speed command.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PropellerSpec {
    pub body: BodyHandle,
    /// Shaft frame in the body frame: +X along the shaft, the origin at the
    /// centre of thrust.
    pub frame: Transform,
    /// `[t1, t2]` of the thrust `T = t1·|ω|·ω − t2·|ω|·V` (N).
    pub thrust: [Real; 2],
    /// `[q1, q2]` of the torque `Q = q1·|ω|·ω − q2·|ω|·V` (N·m); the body
    /// takes `−Q` about the shaft.
    pub torque: [Real; 2],
    /// Rotor speed limit and acceleration (rad/s, rad/s²).
    pub limits: MotorLimits,
    /// Largest `|Q|` the rotor motor supplies (N·m).
    pub max_torque: Real,
}

/// A propeller's rotor speed and averaged loads over the last step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PropellerReading {
    pub omega: Real,
    pub thrust: Real,
    pub torque: Real,
    /// Speed of the centre of thrust along the shaft (m/s).
    pub advance: Real,
}

struct Propeller {
    spec: PropellerSpec,
    controller: MotorController,
    reading: PropellerReading,
}

/// Propellers of a rigid world.
#[derive(Default)]
pub struct Propellers {
    next: u64,
    propellers: BTreeMap<PropellerId, Propeller>,
}

impl Propellers {
    pub fn insert(&mut self, scene: &RigidScene, spec: PropellerSpec) -> Result<PropellerId> {
        if scene.body(spec.body).is_none() {
            return Err(Error::Build("propeller body is missing".into()));
        }
        if spec.thrust.iter().chain(&spec.torque).any(|c| !c.is_finite()) || !(spec.max_torque >= 0.0) {
            return Err(Error::Build("propeller constants must be finite".into()));
        }
        self.next += 1;
        let id = PropellerId(self.next);
        self.propellers.insert(id, Propeller {
            spec,
            controller: MotorController::new(spec.limits),
            reading: PropellerReading::default(),
        });
        Ok(id)
    }

    pub fn remove(&mut self, id: PropellerId) -> bool {
        self.propellers.remove(&id).is_some()
    }

    /// Rotor speed to run at (rad/s), up to the limit.
    pub fn set_speed(&mut self, id: PropellerId, omega: Real) -> Result<()> {
        if !omega.is_finite() {
            return Err(Error::Build("propeller speed must be finite".into()));
        }
        self.propellers
            .get_mut(&id)
            .ok_or_else(|| Error::Build("no such propeller".into()))?
            .controller
            .set_command(MotorCommand::Velocity(omega));
        Ok(())
    }

    pub fn reading(&self, id: PropellerId) -> Option<PropellerReading> {
        self.propellers.get(&id).map(|p| p.reading)
    }

    /// Whether any rotor turns or is told to.
    pub fn active(&self) -> bool {
        self.propellers.values().any(|p| {
            p.controller.velocity != 0.0 || p.controller.command != MotorCommand::Velocity(0.0)
        })
    }

    pub(crate) fn before_step(&mut self, dt: Real) {
        for propeller in self.propellers.values_mut() {
            let omega = propeller.controller.velocity(0.0, dt).unwrap_or(0.0);
            propeller.reading = PropellerReading {
                omega,
                ..PropellerReading::default()
            };
        }
    }

    pub(crate) fn apply(
        &mut self,
        scene: &RigidScene,
        state: &State,
        wrenches: &mut [SpatialVector],
        weight: Real,
    ) -> Result<()> {
        if self.propellers.is_empty() {
            return Ok(());
        }
        let poses = state.body_q.host()?;
        let velocities = state.body_qd.host()?;
        let centers = scene.model().body_com.host()?;
        for propeller in self.propellers.values_mut() {
            let spec = propeller.spec;
            let omega = propeller.reading.omega;
            let (Some(body), Some(desc)) = (scene.body_index(spec.body), scene.body(spec.body)) else {
                continue;
            };
            if omega == 0.0 || !desc.enabled || desc.kind != BodyKind::Dynamic {
                continue;
            }
            let i = body.index();
            let shaft = poses[i].compose(&spec.frame);
            let axis = shaft.rotation * Vec3::X;
            let com = poses[i].transform_point(centers[i]);
            let arm = shaft.position - com;
            let advance = (velocities[i].linear + velocities[i].angular.cross(arm)).dot(axis);
            let spin = omega.abs();
            let thrust = spec.thrust[0] * spin * omega - spec.thrust[1] * spin * advance;
            let torque = (spec.torque[0] * spin * omega - spec.torque[1] * spin * advance)
                .clamp(-spec.max_torque, spec.max_torque);
            let force = axis * thrust;
            wrenches[i].linear += force;
            wrenches[i].angular += arm.cross(force) - axis * torque;
            let reading = &mut propeller.reading;
            reading.thrust += thrust * weight;
            reading.torque += torque * weight;
            reading.advance += advance * weight;
        }
        Ok(())
    }
}
