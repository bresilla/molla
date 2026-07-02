//! V4 — steering, driveline & powertrain.
//!
//! Throttle → [`Engine`] torque at the current rpm → amplified by the
//! [`Transmission`] ratio → split by the [`Driveline`] to the driven
//! wheels. Steering maps a driver input to per-wheel road angles with
//! Ackermann geometry. [`Vehicle::apply_driver`] writes these onto the
//! vehicle's corners each control tick.

use crate::vehicle::Vehicle;
use std::f64::consts::PI;

/// A simple engine: peak torque scaled by throttle, with a mild
/// torque-curve shape and a redline cut.
#[derive(Clone, Copy, Debug)]
pub struct Engine {
    /// Peak torque (N·m) at full throttle.
    pub peak_torque: f64,
    /// Idle speed (rpm).
    pub idle_rpm: f64,
    /// Redline (rpm) — torque is cut above it.
    pub redline: f64,
}

impl Default for Engine {
    fn default() -> Self {
        Self {
            peak_torque: 240.0,
            idle_rpm: 800.0,
            redline: 6500.0,
        }
    }
}

impl Engine {
    /// Torque (N·m) at the given crank `rpm` and `throttle` in `[0, 1]`.
    pub fn torque(&self, rpm: f64, throttle: f64) -> f64 {
        if rpm > self.redline {
            return 0.0;
        }
        let frac = ((rpm - self.idle_rpm) / (self.redline - self.idle_rpm)).clamp(0.0, 1.0);
        // A mild hump: ~1.0 mid-range, ~0.6 at the ends.
        let shape = (1.0 - 1.6 * (frac - 0.5).powi(2)).max(0.55);
        throttle.clamp(0.0, 1.0) * self.peak_torque * shape
    }
}

/// A gearbox: a set of gear ratios plus a final-drive ratio, with an
/// RPM-triggered automatic shifter.
#[derive(Clone, Debug)]
pub struct Transmission {
    /// Gear ratios (gear 0 is first/lowest).
    pub gear_ratios: Vec<f64>,
    /// Final-drive (differential) ratio.
    pub final_drive: f64,
    /// Currently selected gear index.
    pub gear: usize,
    /// Engine speed (rpm) above which [`auto_shift`](Self::auto_shift)
    /// selects the next-higher gear.
    pub upshift_rpm: f64,
    /// Engine speed (rpm) below which [`auto_shift`](Self::auto_shift)
    /// drops a gear. Must sit well below `upshift_rpm` scaled by the
    /// largest adjacent-ratio step, or the shifter will hunt: after an
    /// upshift the rpm falls by `ratio[g+1]/ratio[g]` (≈ 0.6–0.8 for
    /// the default box), so the post-shift rpm must stay above this
    /// threshold.
    pub downshift_rpm: f64,
}

impl Default for Transmission {
    fn default() -> Self {
        Self {
            gear_ratios: vec![3.4, 2.1, 1.4, 1.0, 0.8],
            final_drive: 3.5,
            gear: 0,
            // 5200 → post-upshift rpm ≈ 3200 (worst adjacent step
            // 2.1/3.4); 1800 keeps a wide hysteresis band so the
            // shifter cannot oscillate between two gears at constant
            // speed.
            upshift_rpm: 5200.0,
            downshift_rpm: 1800.0,
        }
    }
}

impl Transmission {
    /// Total ratio from the crankshaft to the wheels.
    pub fn ratio(&self) -> f64 {
        self.gear_ratios[self.gear.min(self.gear_ratios.len() - 1)] * self.final_drive
    }

    /// Advance the automatic shifter one control tick: upshift above
    /// `upshift_rpm`, downshift below `downshift_rpm`, at most one gear
    /// per call so a spin spike cannot skip the box to top gear.
    pub fn auto_shift(&mut self, engine_rpm: f64) {
        if engine_rpm > self.upshift_rpm && self.gear + 1 < self.gear_ratios.len() {
            self.gear += 1;
        } else if engine_rpm < self.downshift_rpm && self.gear > 0 {
            self.gear -= 1;
        }
    }
}

/// Which wheels the driveline powers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Driveline {
    /// Front-wheel drive (corners 0, 1).
    Front,
    /// Rear-wheel drive (corners 2, 3).
    Rear,
    /// All-wheel drive.
    All,
}

impl Driveline {
    /// Corner indices this driveline powers.
    pub fn driven(&self) -> &'static [usize] {
        match self {
            Driveline::Front => &[0, 1],
            Driveline::Rear => &[2, 3],
            Driveline::All => &[0, 1, 2, 3],
        }
    }
}

/// Engine + transmission + driveline.
#[derive(Clone, Debug)]
pub struct Powertrain {
    /// The engine.
    pub engine: Engine,
    /// The gearbox.
    pub transmission: Transmission,
    /// Which wheels are driven.
    pub driveline: Driveline,
    /// Maximum brake torque per wheel (N·m).
    pub max_brake_torque: f64,
}

impl Default for Powertrain {
    fn default() -> Self {
        Self {
            engine: Engine::default(),
            transmission: Transmission::default(),
            driveline: Driveline::Rear,
            max_brake_torque: 2500.0,
        }
    }
}

impl Powertrain {
    /// Engine crank speed (rpm) implied by the driven-wheel spin rate
    /// `wheel_spin` (rad/s) through the current total ratio.
    pub fn engine_rpm(&self, wheel_spin: f64) -> f64 {
        wheel_spin.abs() * self.transmission.ratio() * 60.0 / (2.0 * PI)
    }

    /// Total wheel torque (N·m) for the given `throttle` and the
    /// average driven-wheel spin rate `wheel_spin` (rad/s).
    pub fn wheel_torque(&self, throttle: f64, wheel_spin: f64) -> f64 {
        let ratio = self.transmission.ratio();
        self.engine.torque(self.engine_rpm(wheel_spin), throttle) * ratio
    }
}

/// One control tick of driver input.
#[derive(Clone, Copy, Debug, Default)]
pub struct DriverInput {
    /// Throttle in `[0, 1]`.
    pub throttle: f64,
    /// Brake in `[0, 1]`.
    pub brake: f64,
    /// Steering reference angle (rad); positive turns left.
    pub steer: f64,
}

/// Ackermann steering: a reference angle `steer` for a vehicle of the
/// given `wheelbase` and `track` becomes `(left, right)` road-wheel
/// angles, the inner wheel turning more sharply.
pub fn ackermann_steer(steer: f64, wheelbase: f64, track: f64) -> (f64, f64) {
    if steer.abs() < 1e-6 {
        return (0.0, 0.0);
    }
    // Turn radius of the reference (centre-line) wheel.
    let radius = wheelbase / steer.tan();
    let inner = (wheelbase / (radius.abs() - track * 0.5)).atan();
    let outer = (wheelbase / (radius.abs() + track * 0.5)).atan();
    let s = steer.signum();
    // Positive steer (left turn): the left wheel is the inner one.
    if steer > 0.0 {
        (s * inner, s * outer)
    } else {
        (s * outer, s * inner)
    }
}

impl Vehicle {
    /// Apply one tick of driver input through a [`Powertrain`]: advance
    /// the automatic shifter, set the driven wheels' drive torque, every
    /// wheel's brake torque, and the front wheels' Ackermann steer
    /// angles. Takes the powertrain mutably because the gearbox state
    /// (selected gear) advances with driving — callers that clone the
    /// powertrain per frame must write it back or the gear resets.
    pub fn apply_driver(&mut self, powertrain: &mut Powertrain, input: &DriverInput) {
        let driven = powertrain.driveline.driven();
        let avg_spin =
            driven.iter().map(|&i| self.corners[i].spin).sum::<f64>() / driven.len() as f64;
        let rpm = powertrain.engine_rpm(avg_spin);
        powertrain.transmission.auto_shift(rpm);
        let total = powertrain.wheel_torque(input.throttle, avg_spin);
        let per_wheel = total / driven.len() as f64;
        for c in &mut self.corners {
            c.drive_torque = 0.0;
            c.brake_torque = input.brake.clamp(0.0, 1.0) * powertrain.max_brake_torque;
        }
        for &i in driven {
            self.corners[i].drive_torque = per_wheel;
        }
        // Wheelbase / track from the corner mount geometry.
        let wheelbase = (self.corners[0].attach[0] - self.corners[2].attach[0]).abs();
        let track = (self.corners[0].attach[2] - self.corners[1].attach[2]).abs();
        let (left, right) = ackermann_steer(input.steer, wheelbase, track);
        self.corners[0].steer = left;
        self.corners[1].steer = right;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::terrain::FlatTerrain;

    #[test]
    fn engine_torque_responds_to_throttle_and_redline() {
        let e = Engine::default();
        assert_eq!(e.torque(3000.0, 0.0), 0.0, "closed throttle → no torque");
        assert!(e.torque(3000.0, 1.0) > 0.0, "open throttle → torque");
        assert!(
            e.torque(3000.0, 1.0) > e.torque(3000.0, 0.5),
            "more throttle → more torque"
        );
        assert_eq!(e.torque(9000.0, 1.0), 0.0, "past redline → cut");
    }

    #[test]
    fn transmission_auto_shifts_with_hysteresis() {
        let mut t = Transmission::default();
        assert_eq!(t.gear, 0);
        // Past the upshift point → next gear, one at a time.
        t.auto_shift(6000.0);
        assert_eq!(t.gear, 1, "high rpm → upshift");
        // Post-upshift rpm for the same road speed drops by the ratio
        // step (2.1/3.4 → ~3700 rpm): inside the hysteresis band, no
        // hunting back down.
        t.auto_shift(6000.0 * 2.1 / 3.4);
        assert_eq!(t.gear, 1, "post-upshift rpm must not trigger a downshift");
        // Slowing to near idle drops a gear...
        t.auto_shift(1000.0);
        assert_eq!(t.gear, 0, "low rpm → downshift");
        // ...and the post-downshift rpm (1000·3.4/2.1 ≈ 1600) stays
        // below the upshift point: no oscillation.
        t.auto_shift(1000.0 * 3.4 / 2.1);
        assert_eq!(t.gear, 0, "post-downshift rpm must not re-upshift");
        // Bounds: can't shift past top gear or below first.
        t.gear = 4;
        t.auto_shift(9000.0);
        assert_eq!(t.gear, 4, "no shift past top gear");
        t.gear = 0;
        t.auto_shift(0.0);
        assert_eq!(t.gear, 0, "no shift below first gear");
    }

    #[test]
    fn apply_driver_upshifts_at_high_rpm() {
        let mut pt = Powertrain::default();
        let mut car = Vehicle::car();
        // Wheel spin that puts the engine past the upshift point in
        // first gear: ω = rpm · 2π / (60 · ratio).
        let spin =
            (pt.transmission.upshift_rpm + 300.0) * 2.0 * PI / (60.0 * pt.transmission.ratio());
        for c in &mut car.corners {
            c.spin = spin;
        }
        let input = DriverInput {
            throttle: 1.0,
            ..Default::default()
        };
        car.apply_driver(&mut pt, &input);
        assert_eq!(
            pt.transmission.gear, 1,
            "driving past the upshift rpm must engage second gear"
        );
    }

    #[test]
    fn ackermann_inner_wheel_turns_more() {
        let (left, right) = ackermann_steer(0.3, 2.6, 1.5);
        assert!(left > 0.0 && right > 0.0, "left turn → both wheels left");
        assert!(left > right, "inner (left) wheel turns more sharply");
        assert_eq!(ackermann_steer(0.0, 2.6, 1.5), (0.0, 0.0));
    }

    #[test]
    fn vehicle_accelerates_under_throttle() {
        let ground = FlatTerrain::new(0.0, 1.0);
        let mut pt = Powertrain::default();
        let mut car = Vehicle::car();
        let throttle = DriverInput {
            throttle: 1.0,
            ..Default::default()
        };
        for _ in 0..2000 {
            car.apply_driver(&mut pt, &throttle);
            car.step(&ground, 1.0e-3);
        }
        assert!(
            car.forward_speed() > 2.0,
            "car did not accelerate: speed = {}",
            car.forward_speed()
        );
    }

    #[test]
    fn vehicle_brakes_to_a_stop() {
        let ground = FlatTerrain::new(0.0, 1.0);
        let mut pt = Powertrain::default();
        let mut car = Vehicle::car();
        // Get rolling.
        let go = DriverInput {
            throttle: 1.0,
            ..Default::default()
        };
        for _ in 0..2000 {
            car.apply_driver(&mut pt, &go);
            car.step(&ground, 1.0e-3);
        }
        let moving = car.forward_speed();
        assert!(moving > 2.0);
        // Then brake.
        let brake = DriverInput {
            brake: 1.0,
            ..Default::default()
        };
        for _ in 0..4000 {
            car.apply_driver(&mut pt, &brake);
            car.step(&ground, 1.0e-3);
        }
        assert!(
            car.forward_speed() < moving * 0.4,
            "brakes did not slow the car: {} → {}",
            moving,
            car.forward_speed()
        );
    }

    #[test]
    fn steering_produces_yaw() {
        let ground = FlatTerrain::new(0.0, 1.0);
        let mut pt = Powertrain::default();
        let mut car = Vehicle::car();
        let drive = DriverInput {
            throttle: 0.8,
            ..Default::default()
        };
        for _ in 0..2500 {
            car.apply_driver(&mut pt, &drive);
            car.step(&ground, 1.0e-3);
        }
        assert!(
            car.forward_speed() > 3.0,
            "car not rolling for the turn test"
        );
        let yaw_before = car.yaw;
        let turn = DriverInput {
            throttle: 0.6,
            steer: 0.25,
            ..Default::default()
        };
        for _ in 0..2000 {
            car.apply_driver(&mut pt, &turn);
            car.step(&ground, 1.0e-3);
        }
        assert!(
            (car.yaw - yaw_before).abs() > 0.1,
            "steering produced no yaw: Δyaw = {}",
            car.yaw - yaw_before
        );
    }
}
