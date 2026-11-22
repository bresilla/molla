use super::invalid;
use molla_core::Result;
use molla_math::Vec3;

/// Isotropic Ornstein–Uhlenbeck air-velocity fluctuations for particle drag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TurbulentDispersion {
    velocity: Vec3,
    seed: u64,
}

impl TurbulentDispersion {
    pub fn new(seed: u64) -> Self {
        Self {
            velocity: Vec3::ZERO,
            seed,
        }
    }

    /// Advances correlated fluctuations with per-component RMS speed and correlation time.
    pub fn advance(&mut self, dt: f64, rms: f64, correlation: f64) -> Result<Vec3> {
        if !dt.is_finite()
            || !(0.0..=0.1).contains(&dt)
            || !rms.is_finite()
            || rms < 0.0
            || !correlation.is_finite()
            || correlation <= 0.0
        {
            return Err(invalid("invalid turbulent dispersion step"));
        }
        if dt == 0.0 {
            return Ok(self.velocity);
        }
        let mut next = *self;
        let (x, y) = next.normal_pair();
        let (z, _) = next.normal_pair();
        let decay = (-dt / correlation).exp();
        let innovation = (-(-2.0 * dt / correlation).exp_m1()).sqrt() * rms;
        next.velocity = self.velocity * decay + Vec3::new(x, y, z) * innovation;
        if !next.velocity.is_finite() {
            return Err(invalid("turbulent dispersion overflow"));
        }
        *self = next;
        Ok(self.velocity)
    }

    fn uniform(&mut self) -> f64 {
        self.seed = self.seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut value = self.seed;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d049bb133111eb);
        (((value ^ (value >> 31)) >> 12) as f64 + 0.5) / ((1u64 << 52) as f64)
    }

    fn normal_pair(&mut self) -> (f64, f64) {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        let (sine, cosine) = (std::f64::consts::TAU * self.uniform()).sin_cos();
        (radius * cosine, radius * sine)
    }
}

/// Spherical water droplet with Schiller–Naumann drag in prescribed air.
#[derive(Clone, Copy, Debug)]
pub struct WaterDroplet {
    pub position: Vec3,
    pub velocity: Vec3,
    diameter: f64,
}

impl WaterDroplet {
    pub fn new(position: Vec3, velocity: Vec3, diameter: f64) -> Result<Self> {
        if !position.is_finite()
            || !velocity.is_finite()
            || !diameter.is_finite()
            || !(1e-6..=0.005).contains(&diameter)
        {
            return Err(invalid("invalid water droplet"));
        }
        Ok(Self {
            position,
            velocity,
            diameter,
        })
    }

    pub fn diameter(&self) -> f64 {
        self.diameter
    }

    /// Integrates gravity and air drag with frozen-coefficient exponential substeps.
    pub fn advance(&mut self, dt: f64, air: Vec3) -> Result<()> {
        if !dt.is_finite()
            || !(0.0..=0.1).contains(&dt)
            || !air.is_finite()
            || !self.position.is_finite()
            || !self.velocity.is_finite()
        {
            return Err(invalid("invalid droplet step"));
        }
        let steps = (dt / 0.001).ceil().max(1.0) as usize;
        let h = dt / steps as f64;
        let mut next = *self;
        for _ in 0..steps {
            let re = (air - next.velocity).length() * self.diameter / 1.5e-5;
            if !re.is_finite() {
                return Err(invalid("droplet drag overflow"));
            }
            let cd_re = if re < 1000.0 {
                24.0 * (1.0 + 0.15 * re.powf(0.687))
            } else {
                0.44 * re
            };
            let rate = 0.75 * 1.8e-5 * cd_re / (998.0 * self.diameter.powi(2));
            let equilibrium = air - Vec3::Y * (9.81 * (1.0 - 1.2 / 998.0) / rate);
            let decay = (-rate * h).exp();
            next.position +=
                equilibrium * h + (next.velocity - equilibrium) * (-(-rate * h).exp_m1() / rate);
            next.velocity = equilibrium + (next.velocity - equilibrium) * decay;
        }
        if !next.position.is_finite() || !next.velocity.is_finite() {
            return Err(invalid("droplet step overflow"));
        }
        *self = next;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn turbulent_dispersion_has_zero_mean_target_variance_and_time_correlation() {
        let mut eddy = TurbulentDispersion::new(123);
        let mut sum = Vec3::ZERO;
        let mut square = 0.0;
        let mut covariance = 0.0;
        let mut previous = Vec3::ZERO;
        for step in 0..50_000 {
            let velocity = eddy.advance(0.01, 1.2, 0.12).unwrap();
            if step >= 1000 {
                sum += velocity;
                square += velocity.length_squared() / 3.0;
                covariance += velocity.dot(previous) / 3.0;
            }
            previous = velocity;
        }
        let variance = square / 49_000.0;
        assert!((sum / 49_000.0).length() < 0.12);
        assert!((variance - 1.44).abs() < 0.12, "variance={variance}");
        assert!((covariance / square - (-0.01_f64 / 0.12).exp()).abs() < 0.015);
        let before = eddy;
        assert_eq!(eddy.advance(0.0, 1.2, 0.12).unwrap(), before.velocity);
        assert_eq!(eddy, before);
        assert!(eddy.advance(0.01, f64::NAN, 0.12).is_err());
        assert!(eddy.advance(0.01, 1.0, 0.0).is_err());
        assert_eq!(eddy, before);
        let mut quiet = TurbulentDispersion::new(1);
        for _ in 0..100 {
            assert_eq!(quiet.advance(0.01, 0.0, 0.12).unwrap(), Vec3::ZERO);
        }
        let mut copy = before;
        assert_eq!(
            eddy.advance(0.01, 1.2, 0.12).unwrap(),
            copy.advance(0.01, 1.2, 0.12).unwrap()
        );
    }

    #[test]
    fn turbulent_dispersion_variance_is_consistent_across_timesteps() {
        for dt in [1.0_f64 / 60.0, 1.0 / 120.0] {
            let mut sum = 0.0;
            for seed in 0..2048 {
                let mut eddy = TurbulentDispersion::new(seed * 0x100000001b3);
                let mut velocity = Vec3::ZERO;
                for _ in 0..(0.5 / dt).round() as usize {
                    velocity = eddy.advance(dt, 1.2, 0.12).unwrap();
                }
                sum += velocity.length_squared() / 3.0;
            }
            let expected = 1.44 * (1.0 - (-1.0_f64 / 0.12).exp());
            assert!((sum / 2048.0 - expected).abs() < 0.08);
        }
    }

    #[test]
    fn droplets_settle_follow_crosswind_and_reject_invalid_steps() {
        let mut fine = WaterDroplet::new(Vec3::Y * 4.0, Vec3::ZERO, 100e-6).unwrap();
        let mut coarse = WaterDroplet::new(Vec3::Y * 4.0, Vec3::ZERO, 400e-6).unwrap();
        for _ in 0..120 {
            fine.advance(1.0 / 120.0, Vec3::X * 2.0).unwrap();
            coarse.advance(1.0 / 120.0, Vec3::X * 2.0).unwrap();
        }
        assert!(fine.position.x > 1.8 && coarse.position.x > 1.0);
        assert!(fine.velocity.y < -0.2 && fine.velocity.y > -0.3);
        assert!(coarse.velocity.y < -1.4 && coarse.velocity.y > -1.8);
        assert!(coarse.position.y < fine.position.y);
        let before = fine.position;
        assert!(fine.advance(f64::NAN, Vec3::ZERO).is_err());
        assert_eq!(before, fine.position);
        assert!(fine.advance(0.01, Vec3::splat(f64::MAX)).is_err());
        assert_eq!(before, fine.position);
        assert!(WaterDroplet::new(Vec3::ZERO, Vec3::ZERO, 0.0).is_err());
    }

    #[test]
    fn droplet_drag_is_stable_and_converges_under_strong_downwash() {
        let run = |dt: f64| {
            let mut p = WaterDroplet::new(Vec3::Y * 5.0, -Vec3::Y * 3.0, 250e-6).unwrap();
            for _ in 0..(1.0 / dt).round() as usize {
                p.advance(dt, Vec3::new(1.0, -8.0, 0.3)).unwrap();
            }
            p
        };
        let a = run(1.0 / 60.0);
        let b = run(1.0 / 240.0);
        assert!(a.position.distance(b.position) < 0.002);
        assert!(a.velocity.y < -8.0 && a.velocity.y > -10.0);
    }

    #[test]
    fn application_droplets_settle_through_weak_updrafts_but_follow_strong_ones() {
        let run = |diameter: f64, air: Vec3, dt: f64| {
            let mut drop = WaterDroplet::new(Vec3::Y * 6.0, Vec3::ZERO, diameter).unwrap();
            for _ in 0..(2.0 / dt).round() as usize {
                drop.advance(dt, air).unwrap();
            }
            drop
        };
        let weak = Vec3::new(2.0, 0.35, 0.0);
        let mist = run(45e-6, weak, 1.0 / 60.0);
        assert!(mist.position.y > 6.5);
        for diameter in [160e-6, 200e-6, 240e-6] {
            let calm = run(diameter, Vec3::ZERO, 1.0 / 60.0);
            let drop = run(diameter, weak, 1.0 / 60.0);
            let refined = run(diameter, weak, 1.0 / 1200.0);
            eprintln!(
                "diameter={diameter}, settling_speed={}, weak_updraft_y={}",
                -calm.velocity.y, drop.velocity.y
            );
            assert!(calm.velocity.y < -0.5 && calm.velocity.y > -1.0);
            assert!(drop.velocity.y < -0.15 && drop.position.y < 5.8);
            assert!(drop.position.x > 3.5 && drop.velocity.x > 1.99);
            assert!(drop.position.distance(refined.position) < 0.003);
            let rising = run(diameter, Vec3::Y * 2.0, 1.0 / 60.0);
            assert!(rising.position.y > 7.0 && rising.velocity.y > 1.0);
            let jet = |dt: f64| {
                let mut drop =
                    WaterDroplet::new(Vec3::Y * 6.0, -Vec3::Y * 36.374, diameter).unwrap();
                for step in 0..(0.5 / dt).round() as usize {
                    let lateral = if (step as f64 + 0.5) * dt < 0.25 {
                        2.0
                    } else {
                        -2.0
                    };
                    drop.advance(dt, Vec3::new(lateral, -3.5, 0.4)).unwrap();
                }
                drop
            };
            let coarse = jet(1.0 / 60.0);
            let fine = jet(1.0 / 1200.0);
            assert!(coarse.position.distance(fine.position) < 0.006);
            assert!(coarse.velocity.x < -1.0 && coarse.velocity.y < -3.5);
        }
    }

    #[test]
    fn atomized_mist_relaxes_to_changing_air_and_converges() {
        for diameter in [30e-6, 45e-6, 60e-6] {
            let run = |dt: f64| {
                let mut p = WaterDroplet::new(Vec3::Y * 6.0, -Vec3::Y * 36.374, diameter).unwrap();
                for step in 0..(0.5 / dt).round() as usize {
                    let side = if (step as f64 + 0.5) * dt < 0.25 {
                        1.0
                    } else {
                        -1.0
                    };
                    p.advance(dt, Vec3::new(2.0 * side, -3.5, 0.4)).unwrap();
                }
                p
            };
            let coarse = run(1.0 / 60.0);
            let fine = run(1.0 / 1200.0);
            assert!(
                coarse.position.distance(fine.position) < 0.006,
                "diameter={diameter}, position error={}",
                coarse.position.distance(fine.position)
            );
            assert!((coarse.velocity.x + 2.0).abs() < 1e-6);
            assert!((coarse.velocity.z - 0.4).abs() < 1e-6);
            assert!(coarse.velocity.y < -3.5 && coarse.velocity.y > -3.62);
        }
    }
}
