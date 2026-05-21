//! V0 — 1-DOF rotational inertia (driveline shaft).
//!
//! A [`Shaft`] is the building block of the V4 driveline: engine,
//! gearbox, and wheel spin are all 1-DOF rotational inertias linked by
//! gear ratios. Torques accumulate over a step and are integrated with
//! semi-implicit Euler.

/// A 1-DOF rotational inertia.
#[derive(Clone, Copy, Debug)]
pub struct Shaft {
    /// Rotational inertia `I` (kg·m²).
    pub inertia: f64,
    /// Angular position (rad).
    pub angle: f64,
    /// Angular velocity (rad/s).
    pub omega: f64,
    /// Torque accumulated since the last [`Shaft::step`] (N·m).
    torque: f64,
}

impl Shaft {
    /// A shaft of inertia `inertia`, at rest.
    pub fn new(inertia: f64) -> Self {
        Self {
            inertia: inertia.max(1e-9),
            angle: 0.0,
            omega: 0.0,
            torque: 0.0,
        }
    }

    /// Accumulate a torque to be applied on the next [`Shaft::step`].
    pub fn apply_torque(&mut self, torque: f64) {
        self.torque += torque;
    }

    /// Integrate one step of size `dt` (semi-implicit Euler) and clear
    /// the accumulated torque.
    pub fn step(&mut self, dt: f64) {
        self.omega += (self.torque / self.inertia) * dt;
        self.angle += self.omega * dt;
        self.torque = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_torque_spins_the_shaft_up_linearly() {
        let mut s = Shaft::new(2.0);
        let dt = 1.0e-3;
        for _ in 0..1000 {
            s.apply_torque(4.0); // tau / I = 2 rad/s²
            s.step(dt);
        }
        // After 1 s: omega ≈ 2 rad/s.
        assert!((s.omega - 2.0).abs() < 1e-6, "omega = {}", s.omega);
        assert!(s.angle > 0.0 && s.angle.is_finite());
    }

    #[test]
    fn torque_is_cleared_each_step() {
        let mut s = Shaft::new(1.0);
        s.apply_torque(10.0);
        s.step(1.0e-3);
        let omega_after_one = s.omega;
        // No torque applied this step → omega unchanged.
        s.step(1.0e-3);
        assert!((s.omega - omega_after_one).abs() < 1e-12);
    }
}
