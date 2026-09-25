//! Belts: collider surfaces that run along a direction, as conveyors or
//! tank tracks (Webots `Track` with its linear motor). Contacts are dragged
//! along the running surface by friction.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_sim::runtime::{ColliderHandle, RigidScene};

use super::{MotorCommand, MotorController, MotorLimits};

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BeltId(pub u64);

#[derive(Clone, Debug, PartialEq)]
pub struct BeltSpec {
    /// Colliders forming the running surface.
    pub colliders: Vec<ColliderHandle>,
    /// Running direction in each collider's frame.
    pub direction: Vec3,
    /// Belt speed limit, acceleration and position-control gains (m/s, m/s²).
    pub limits: MotorLimits,
}

/// A belt's travel and speed after the last step.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BeltReading {
    pub position: Real,
    pub velocity: Real,
}

struct Belt {
    spec: BeltSpec,
    controller: MotorController,
    reading: BeltReading,
}

/// Belts of a rigid world.
#[derive(Default)]
pub struct Belts {
    next: u64,
    belts: BTreeMap<BeltId, Belt>,
}

impl Belts {
    pub fn insert(&mut self, scene: &RigidScene, mut spec: BeltSpec) -> Result<BeltId> {
        if spec.colliders.is_empty() || spec.colliders.iter().any(|&c| scene.collider(c).is_none()) {
            return Err(Error::Build("belt colliders are missing".into()));
        }
        spec.direction = spec
            .direction
            .try_normalize()
            .ok_or_else(|| Error::Build("belt direction must be nonzero".into()))?;
        self.next += 1;
        let id = BeltId(self.next);
        let controller = MotorController::new(spec.limits);
        self.belts.insert(id, Belt {
            spec,
            controller,
            reading: BeltReading::default(),
        });
        Ok(id)
    }

    /// Removes a belt and stops its surfaces.
    pub fn remove(&mut self, scene: &mut RigidScene, id: BeltId) -> Result<bool> {
        let Some(belt) = self.belts.remove(&id) else {
            return Ok(false);
        };
        for &collider in &belt.spec.colliders {
            if scene.collider(collider).is_some() {
                scene.set_collider_surface_velocity(collider, Vec3::ZERO)?;
            }
        }
        Ok(true)
    }

    fn belt(&mut self, id: BeltId) -> Result<&mut Belt> {
        self.belts.get_mut(&id).ok_or_else(|| Error::Build("no such belt".into()))
    }

    /// Position (travel, m) or velocity control; belts take no force command.
    pub fn command(&mut self, id: BeltId, command: MotorCommand) -> Result<()> {
        match command {
            MotorCommand::Force(_) => Err(Error::Build("belts take position or velocity commands".into())),
            MotorCommand::Position(v) | MotorCommand::Velocity(v) if !v.is_finite() => {
                Err(Error::Build("belt command must be finite".into()))
            }
            command => {
                self.belt(id)?.controller.set_command(command);
                Ok(())
            }
        }
    }

    /// Speed position control runs at.
    pub fn set_speed(&mut self, id: BeltId, speed: Real) -> Result<()> {
        let belt = self.belt(id)?;
        belt.controller.speed = speed.abs().min(belt.controller.limits.max_velocity);
        Ok(())
    }

    pub fn set_acceleration(&mut self, id: BeltId, acceleration: Option<Real>) -> Result<()> {
        self.belt(id)?.controller.limits.acceleration = acceleration.map(Real::abs);
        Ok(())
    }

    pub fn reading(&self, id: BeltId) -> Option<BeltReading> {
        self.belts.get(&id).map(|b| b.reading)
    }

    pub(crate) fn before_step(&mut self, scene: &mut RigidScene, dt: Real) -> Result<()> {
        for belt in self.belts.values_mut() {
            let velocity = belt.controller.velocity(belt.reading.position, dt).unwrap_or(0.0);
            belt.reading.velocity = velocity;
            belt.reading.position += velocity * dt;
            for &collider in &belt.spec.colliders {
                if scene.collider(collider).is_some() {
                    scene.set_collider_surface_velocity(collider, belt.spec.direction * velocity)?;
                }
            }
        }
        Ok(())
    }
}
