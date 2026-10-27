//! Touch sensors: bumper, scalar force and 3-axis force on a body.
//!
//! The model is backend-neutral: the host lists the contact forces acting on
//! bodies this step (from any solver that reports them), and the sensor
//! reduces those on its body.

use molla_core::BodyId;
use molla_math::{Quat, Real, Vec3};

/// What a touch sensor reports.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TouchKind {
    /// Contact or none.
    #[default]
    Bumper,
    /// Magnitude of the total contact force.
    Force,
    /// Total contact force in sensor axes.
    Force3d,
}

/// One contact force acting on a body, in world axes and newtons.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContactForceSample {
    pub body: BodyId,
    pub point: Vec3,
    pub force: Vec3,
}

/// A touch reading. All fields are filled whatever the kind; the kind says
/// which one a Webots-style device would expose.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct TouchReading {
    pub touching: bool,
    pub contacts: u32,
    /// Magnitude of the summed contact force (N).
    pub force: Real,
    /// Summed contact force in sensor axes (N).
    pub force_sensor: Vec3,
}

/// A touch sensor on one body, reading forces in its own axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TouchSensor {
    pub body: BodyId,
    pub kind: TouchKind,
    /// Forces below this magnitude (N) do not count as touching.
    pub threshold: Real,
}

impl TouchSensor {
    pub fn new(body: BodyId, kind: TouchKind) -> Self {
        Self {
            body,
            kind,
            threshold: 0.0,
        }
    }

    /// Reduce this step's contact forces on the sensor body; `world_rotation`
    /// is the sensor's rotation in the world.
    pub fn sample(&self, world_rotation: Quat, contacts: &[ContactForceSample]) -> TouchReading {
        let (count, total) = contacts
            .iter()
            .filter(|c| c.body == self.body)
            .fold((0u32, Vec3::ZERO), |(n, sum), c| (n + 1, sum + c.force));
        let force = total.length();
        TouchReading {
            touching: count > 0 && force >= self.threshold,
            contacts: count,
            force,
            force_sensor: world_rotation.inverse() * total,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn touch_sums_forces_on_its_body_in_sensor_axes() {
        let body = BodyId::from_index(3);
        let other = BodyId::from_index(4);
        let contacts = [
            ContactForceSample {
                body,
                point: Vec3::ZERO,
                force: Vec3::new(0.0, 0.0, 30.0),
            },
            ContactForceSample {
                body,
                point: Vec3::X,
                force: Vec3::new(0.0, 0.0, 10.0),
            },
            ContactForceSample {
                body: other,
                point: Vec3::ZERO,
                force: Vec3::new(0.0, 0.0, 500.0),
            },
        ];
        let sensor = TouchSensor::new(body, TouchKind::Force3d);
        let rotated = Quat::from_rotation_x(std::f64::consts::FRAC_PI_2);
        let reading = sensor.sample(rotated, &contacts);
        assert!(reading.touching);
        assert_eq!(reading.contacts, 2);
        assert!((reading.force - 40.0).abs() < 1e-12);
        assert!((reading.force_sensor - Vec3::new(0.0, 40.0, 0.0)).length() < 1e-9);
        let light = TouchSensor {
            threshold: 50.0,
            ..sensor
        };
        assert!(!light.sample(Quat::IDENTITY, &contacts).touching);
        assert_eq!(sensor.sample(Quat::IDENTITY, &[]), TouchReading::default());
    }
}
