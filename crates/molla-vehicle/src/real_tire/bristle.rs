//! Persistent, combined-slip elastic/plastic tread shear in a tangent frame.

#[derive(Clone, Copy, Debug)]
pub struct TreadBristleParams {
    /// Integrated patch shear stiffness in N/m.
    pub stiffness: f64,
    /// Combined longitudinal/lateral friction budget in N.
    pub friction_limit: f64,
    /// Replacement rate of loaded tread material in 1/s; zero at rest.
    pub renewal_rate: f64,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TreadBristle {
    displacement: [f64; 2],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TreadBristleOutput {
    pub force: [f64; 2],
    pub elastic_energy: f64,
    pub sliding: bool,
}

impl TreadBristle {
    pub fn displacement(&self) -> [f64; 2] {
        self.displacement
    }

    /// Clear contact history on contact loss or a discontinuous frame change.
    pub fn reset(&mut self) {
        self.displacement = [0.0; 2];
    }

    /// Advance shear with backward-Euler tread renewal and a circular plastic return.
    /// Velocity is tread relative to ground in the retained orthonormal tangent frame.
    /// Invalid inputs leave state unchanged. Changing stiffness supplies/removes elastic
    /// energy externally; fixed-parameter updates dissipate work plus stored energy.
    pub fn advance(
        &mut self,
        velocity: [f64; 2],
        dt: f64,
        params: TreadBristleParams,
    ) -> Result<TreadBristleOutput, &'static str> {
        if !dt.is_finite()
            || dt <= 0.0
            || velocity.iter().any(|v| !v.is_finite())
            || [params.stiffness, params.friction_limit, params.renewal_rate]
                .iter()
                .any(|v| !v.is_finite() || *v < 0.0)
        {
            return Err("invalid tread bristle input");
        }
        if params.stiffness == 0.0 || params.friction_limit == 0.0 {
            self.reset();
            return Ok(TreadBristleOutput::default());
        }
        let denominator = 1.0 + dt * params.renewal_rate;
        let displacement = std::array::from_fn::<_, 2, _>(|i| {
            self.displacement[i] / denominator + velocity[i] * (dt / denominator)
        });
        let mut force = displacement.map(|x| -params.stiffness * x);
        let magnitude = force[0].hypot(force[1]);
        if !denominator.is_finite() || !magnitude.is_finite() {
            return Err("non-finite tread bristle trial");
        }
        let sliding = magnitude > params.friction_limit;
        if sliding {
            let scale = params.friction_limit / magnitude;
            force = force.map(|f| f * scale);
        }
        let displacement = if sliding {
            force.map(|f| -f / params.stiffness)
        } else {
            displacement
        };
        let elastic_energy = -0.5 * (force[0] * displacement[0] + force[1] * displacement[1]);
        if !elastic_energy.is_finite() || displacement.iter().any(|v| !v.is_finite()) {
            return Err("non-finite tread bristle energy");
        }
        self.displacement = displacement;
        Ok(TreadBristleOutput {
            force,
            elastic_energy,
            sliding,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> TreadBristleParams {
        TreadBristleParams {
            stiffness: 100_000.0,
            friction_limit: 1000.0,
            renewal_rate: 0.0,
        }
    }

    #[test]
    fn standstill_retains_force_and_reset_releases_it() {
        let mut state = TreadBristle::default();
        let first = state.advance([0.1, 0.2], 0.01, params()).unwrap();
        assert_eq!(first.force, [-100.0, -200.0]);
        assert!(!first.sliding);
        for _ in 0..7200 {
            assert_eq!(
                state.advance([0.0; 2], 1.0 / 120.0, params()).unwrap(),
                first
            );
        }
        state.reset();
        assert_eq!(
            state.advance([0.0; 2], 0.01, params()).unwrap(),
            TreadBristleOutput::default()
        );
    }

    #[test]
    fn combined_slip_has_one_budget_and_no_plastic_windup() {
        let mut state = TreadBristle::default();
        for _ in 0..10_000 {
            let out = state.advance([3.0, 4.0], 0.01, params()).unwrap();
            assert!(out.sliding);
            assert!((out.force[0] + 600.0).abs() < 1e-10);
            assert!((out.force[1] + 800.0).abs() < 1e-10);
            assert!((out.elastic_energy - 5.0).abs() < 1e-12);
        }
        let displacement = state.displacement();
        let unloaded = state
            .advance(displacement.map(|x| -x / 0.01), 0.01, params())
            .unwrap();
        assert!(unloaded.force[0].hypot(unloaded.force[1]) < 1e-10);
    }

    #[test]
    fn unloading_can_return_stored_energy_without_generating_it() {
        let mut state = TreadBristle::default();
        let loaded = state.advance([0.1, 0.0], 0.01, params()).unwrap();
        let velocity = [-0.05, 0.0];
        let unloaded = state.advance(velocity, 0.01, params()).unwrap();
        let work = unloaded.force[0] * velocity[0] * 0.01;
        assert!(work > 0.0);
        assert!(work + unloaded.elastic_energy <= loaded.elastic_energy);
    }

    #[test]
    fn tread_renewal_converges_to_transport_balance() {
        let mut state = TreadBristle::default();
        let config = TreadBristleParams {
            renewal_rate: 20.0,
            friction_limit: 1e6,
            ..params()
        };
        let dt = 0.001;
        for step in 1..=1000 {
            state.advance([0.3, -0.1], dt, config).unwrap();
            let fraction = 1.0 - (1.0 + dt * config.renewal_rate).powi(-step);
            for (actual, velocity) in state.displacement().into_iter().zip([0.3, -0.1]) {
                assert!((actual - fraction * velocity / config.renewal_rate).abs() < 1e-14);
            }
        }
    }

    #[test]
    fn fixed_parameter_updates_obey_discrete_work_energy_bound() {
        for rate in [0.0, 0.5, 100.0] {
            let config = TreadBristleParams {
                renewal_rate: rate,
                ..params()
            };
            let mut state = TreadBristle::default();
            let mut energy = 0.0;
            for step in 0..10_000 {
                let t = step as f64;
                let velocity = [(0.31 * t).sin(), (0.17 * t).cos()];
                let dt = 0.0001 + (step % 7) as f64 * 0.002;
                let out = state.advance(velocity, dt, config).unwrap();
                let work = dt * (out.force[0] * velocity[0] + out.force[1] * velocity[1]);
                assert!(work + out.elastic_energy - energy <= 1e-10);
                energy = out.elastic_energy;
            }
        }
    }

    #[test]
    fn tangent_rotation_rotates_force_without_changing_energy() {
        let (s, c) = 0.37_f64.sin_cos();
        let rotate = |v: [f64; 2]| [c * v[0] - s * v[1], s * v[0] + c * v[1]];
        let mut original = TreadBristle::default();
        let mut rotated = TreadBristle::default();
        for step in 0..1000 {
            let v = [(step as f64 * 0.13).sin(), (step as f64 * 0.29).cos()];
            let a = original.advance(v, 0.003, params()).unwrap();
            let b = rotated.advance(rotate(v), 0.003, params()).unwrap();
            for (expected, actual) in rotate(a.force).into_iter().zip(b.force) {
                assert!((expected - actual).abs() < 1e-9);
            }
            assert!((a.elastic_energy - b.elastic_energy).abs() < 1e-11);
        }
    }

    #[test]
    fn invalid_updates_are_atomic_and_zero_budget_clears_history() {
        let mut state = TreadBristle::default();
        state.advance([0.1; 2], 0.01, params()).unwrap();
        let saved = state;
        for value in [f64::NAN, f64::INFINITY, -1.0] {
            for config in [
                TreadBristleParams {
                    stiffness: value,
                    ..params()
                },
                TreadBristleParams {
                    friction_limit: value,
                    ..params()
                },
                TreadBristleParams {
                    renewal_rate: value,
                    ..params()
                },
            ] {
                assert!(state.advance([0.0; 2], 0.01, config).is_err());
                assert_eq!(state, saved);
            }
            assert!(state.advance([0.0; 2], value, params()).is_err());
            assert_eq!(state, saved);
        }
        for velocity in [[f64::NAN, 0.0], [0.0, f64::INFINITY], [f64::MAX; 2]] {
            assert!(state.advance(velocity, 0.01, params()).is_err());
            assert_eq!(state, saved);
        }
        assert!(state.advance([0.0; 2], 0.0, params()).is_err());
        assert_eq!(state, saved);
        let overflow = TreadBristleParams {
            renewal_rate: f64::MAX,
            ..params()
        };
        assert!(state.advance([0.0; 2], 2.0, overflow).is_err());
        assert_eq!(state, saved);
        let zero = TreadBristleParams {
            friction_limit: 0.0,
            ..params()
        };
        assert_eq!(
            state.advance([0.0; 2], 0.01, zero).unwrap(),
            TreadBristleOutput::default()
        );
        assert_eq!(state, TreadBristle::default());
        state = saved;
        let zero = TreadBristleParams {
            stiffness: 0.0,
            ..params()
        };
        assert_eq!(
            state.advance([0.0; 2], 0.01, zero).unwrap(),
            TreadBristleOutput::default()
        );
        assert_eq!(state, TreadBristle::default());
    }

    #[test]
    fn reducing_contact_budget_projects_existing_shear() {
        let mut state = TreadBristle::default();
        state.advance([0.3, 0.4], 0.01, params()).unwrap();
        let reduced = TreadBristleParams {
            friction_limit: 100.0,
            ..params()
        };
        let out = state.advance([0.0; 2], 0.01, reduced).unwrap();
        assert!(out.sliding);
        assert!((out.force[0] + 60.0).abs() < 1e-12);
        assert!((out.force[1] + 80.0).abs() < 1e-12);
        assert!((out.elastic_energy - 0.05).abs() < 1e-14);
    }

    #[test]
    fn loaded_shear_settles_but_overload_cannot_freeze_a_mass() {
        for external in [300.0_f64, 1500.0] {
            let mut state = TreadBristle::default();
            let mut velocity = 0.0;
            let mass = 100.0;
            let damping = 2.0 * (params().stiffness * mass).sqrt();
            for _ in 0..60_000 {
                let out = state.advance([velocity, 0.0], 0.001, params()).unwrap();
                velocity += 0.001 * (external + out.force[0] - damping * velocity) / mass;
            }
            if external < params().friction_limit {
                assert!(velocity.abs() < 1e-12);
                assert!((state.displacement()[0] - external / params().stiffness).abs() < 1e-12);
            } else {
                assert!((velocity - (external - params().friction_limit) / damping).abs() < 1e-12);
            }
        }
    }
}
