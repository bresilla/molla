//! Reduced continuous-belt drive and anisotropic contact laws.

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackMaterial {
    pub longitudinal_friction: f64,
    pub lateral_friction: f64,
    pub slip_damping: f64,
}

impl Default for TrackMaterial {
    fn default() -> Self {
        Self {
            longitudinal_friction: 0.85,
            lateral_friction: 0.65,
            slip_damping: 8000.0,
        }
    }
}

impl TrackMaterial {
    pub fn is_valid(self) -> bool {
        [
            self.longitudinal_friction,
            self.lateral_friction,
            self.slip_damping,
        ]
        .iter()
        .all(|x| x.is_finite() && *x >= 0.0)
    }

    /// Project tangential forces into the load-scaled friction ellipse.
    pub fn limit(self, desired: [f64; 2], normal_load: f64, ground_friction: f64) -> [f64; 2] {
        if !self.is_valid()
            || !normal_load.is_finite()
            || normal_load <= 0.0
            || !ground_friction.is_finite()
            || ground_friction <= 0.0
            || desired.iter().any(|v| !v.is_finite())
        {
            return [0.0; 2];
        }
        let caps = [self.longitudinal_friction, self.lateral_friction]
            .map(|mu| mu.min(ground_friction) * normal_load);
        let force = [0, 1].map(|i| if caps[i] > 0.0 { desired[i] } else { 0.0 });
        let scale = (force[0] / caps[0].max(f64::MIN_POSITIVE))
            .hypot(force[1] / caps[1].max(f64::MIN_POSITIVE))
            .max(1.0);
        force.map(|v| v / scale)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackMotor {
    pub max_torque: f64,
    pub max_power: f64,
    pub speed_gain: f64,
}

impl Default for TrackMotor {
    fn default() -> Self {
        Self {
            max_torque: 300.0,
            max_power: 5000.0,
            speed_gain: 120.0,
        }
    }
}

impl TrackMotor {
    pub fn is_valid(self) -> bool {
        [self.max_torque, self.max_power, self.speed_gain]
            .iter()
            .all(|x| x.is_finite() && *x >= 0.0)
    }

    pub fn torque(self, target_omega: f64, omega: f64) -> f64 {
        if !self.is_valid() || !target_omega.is_finite() || !omega.is_finite() {
            return 0.0;
        }
        let cap = self
            .max_torque
            .min(self.max_power / omega.abs().max(1.0e-12));
        ((target_omega - omega) * self.speed_gain).clamp(-cap, cap)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined_slip_respects_ellipse_and_zero_load() {
        let material = TrackMaterial::default();
        let [x, y] = material.limit([5000.0, -2000.0], 100.0, 1.0);
        assert!(((x / 85.0).powi(2) + (y / 65.0).powi(2) - 1.0).abs() < 1e-12);
        assert_eq!(material.limit([20.0, 10.0], 0.0, 1.0), [0.0; 2]);
        assert_eq!(material.limit([20.0, 10.0], 100.0, 0.0), [0.0; 2]);
        assert_eq!(material.limit([f64::NAN, 0.0], 100.0, 1.0), [0.0; 2]);
    }

    #[test]
    fn drive_and_brake_respect_torque_and_power() {
        let motor = TrackMotor::default();
        for omega in [-100.0, -2.0, 0.0, 2.0, 100.0] {
            for target in [-1000.0, 0.0, 1000.0] {
                let torque = motor.torque(target, omega);
                assert!(torque.abs() <= motor.max_torque);
                assert!((torque * omega).abs() <= motor.max_power + 1e-9);
                assert!(torque * (target - omega) >= 0.0);
            }
        }
    }
}
