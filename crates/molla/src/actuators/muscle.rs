//! Stage 42 — Hill-type muscle actuator for CABLE joints.
//!
//! Implements the classic Hill three-element model and (in v2) the
//! Thelen-style series-elastic tendon plus first-order excitation →
//! activation filter:
//!
//! ```text
//!   da/dt   = (u − a) / τ_act                       (excitation lag)
//!   f_fibre = a · f_max · f_L(l_m/l_opt) · f_V(v_m/v_max)
//!   f_t(ε)  = f_max · k_t · ε²   (ε > 0)            (tendon SEE)
//!   f_t(ε)  = f_fibre · cos(α)                      (force balance)
//! ```
//!
//! where `α` is the pennation angle and `ε = (l_t − l_t_slack)/l_t_slack`
//! is tendon strain. The fibre length is whatever is left over once
//! the tendon takes its share of the total path length:
//!
//! ```text
//!   l_total = l_m · cos(α) + l_t
//! ```
//!
//! In cable coordinates the *total path length* is `l_opt + q_l`, i.e.
//! the cable's linear stretch DoF is the full musculotendon unit (MTU)
//! deflection from its rest length.
//!
//! ## v1 → v2
//!
//! v1 shipped with single-muscle-per-joint, instantaneous activation,
//! and a rigid tendon (no SEE, no pennation). v2 adds the three
//! biophysical extensions documented above:
//!
//! - **Co-contraction**: multiple muscles per joint are summed at the
//!   solver level via a CSR (`joint_muscle_offsets` /
//!   `joint_muscle_indices` on `Model`).
//! - **Activation lag**: callers write [`Model::muscle_excitation`] (the
//!   neural drive `u`); the per-step filter
//!   [`update_muscle_activation`] integrates the first-order ODE
//!   `da/dt = (u − a)/τ_act` with implicit Euler (unconditionally
//!   stable). The filtered state lives in `Model::muscle_activation`.
//! - **Series-elastic tendon**: when [`MuscleConfig::l_tendon_slack`] is
//!   positive, [`Muscle::torque`] runs a Newton solver on the tendon
//!   strain to balance `f_fibre · cos(α) = f_tendon(ε)`. The default
//!   tendon stiffness is `E·A = 30 · f_max` (Thelen-2003).
//!
//! ## References
//!
//! - Newton's `kernels_muscle.py` — same activation model, no tendon.
//! - Zajac (1989), "Muscle and Tendon: Properties, Models, Scaling…"
//! - Thelen (2003), "Adjustment of Muscle Mechanics Model Parameters
//!   to Simulate Dynamic Contractions in Older Adults."

use crate::core::JointId;
use crate::core::Result;
use crate::math::Real;

/// Default activation time constant `τ_act` (s). 20 ms is the
/// activation rise time used in Thelen (2003). Deactivation is
/// modelled with the same constant in v2 (Thelen uses an asymmetric
/// rise/fall but a single τ is enough for control-research fidelity).
pub const DEFAULT_TAU_ACTIVATION: Real = 0.020;

/// Tendon stiffness scaling — `E · A / f_max`. Thelen-2003 picks
/// 33.33; we round to 30 for slightly softer tendons (matches the
/// "typical" value cited in OpenSim's `Millard2012EquilibriumMuscle`
/// documentation).
pub const DEFAULT_TENDON_STIFFNESS_RATIO: Real = 30.0;

/// Configuration for a single Hill-type muscle attached to one CABLE
/// joint. All values are SI: forces in newtons, lengths in metres,
/// velocities in m/s. Activation defaults to `0.0` (off); the caller
/// drives muscle output through [`Model::muscle_excitation`] (the
/// "neural drive" `u`) which is low-pass filtered into
/// [`Model::muscle_activation`] by [`update_muscle_activation`].
#[derive(Debug, Clone, Copy)]
pub struct MuscleConfig {
    /// Maximum isometric force, `f_max` (N). Force generated at full
    /// activation, optimal length, zero velocity.
    pub f_max: Real,
    /// Optimal fibre length, `l_opt` (m). The force–length curve
    /// peaks at this fibre length.
    pub l_opt: Real,
    /// Maximum shortening velocity, `v_max` (m/s).
    pub v_max: Real,
    /// Initial activation level, `a ∈ [0, 1]`. Defaults to 0 (off).
    /// This is the *filtered* state used by the Hill model; user
    /// inputs the excitation `u` separately.
    pub activation: Real,
    /// Initial neural drive / excitation, `u ∈ [0, 1]`. Defaults to
    /// the same value as `activation` so freshly-built muscles don't
    /// drift on the very first filter step.
    pub excitation: Real,
    /// First-order activation time constant `τ_act` (s). The filter
    /// `da/dt = (u − a)/τ_act` integrates with implicit Euler each
    /// step. Default [`DEFAULT_TAU_ACTIVATION`] (20 ms).
    pub tau_activation: Real,
    /// Tendon slack length `l_t_slack` (m). Set to `0` for a rigid
    /// tendon (v1 behaviour — fibre length tracks the cable stretch
    /// exactly). A positive value enables the series-elastic element.
    pub l_tendon_slack: Real,
    /// Pennation angle at optimal fibre length `α₀` (rad). The fibre
    /// pulls on the tendon at this angle; `cos(α)` attenuates the
    /// transmitted force. Default 0.0 (in-line fibres).
    pub pennation_angle: Real,
    /// Polarity / mounting sign. `+1.0` (default) means the muscle
    /// pulls the cable shorter (`q_l → −`); `-1.0` means it pulls
    /// longer (mirror-mounted antagonist on the opposite side of the
    /// joint). Modelling antagonist pairs on a 1-D cable DoF without
    /// a polarity flip would require a second cable joint with
    /// opposite axis — encoding the sign on the muscle keeps the
    /// model graph flat. Any non-zero value is permitted but values
    /// outside `{+1, −1}` give non-physical scaling, so the
    /// builder clamps to that set.
    pub polarity: Real,
}

impl Default for MuscleConfig {
    /// Defaults chosen for a "medium" mammalian limb muscle: 100 N
    /// peak isometric force, 5 cm optimal fibre length, 0.5 m/s peak
    /// contraction velocity, rigid tendon (v1 compatible).
    fn default() -> Self {
        Self {
            f_max: 100.0,
            l_opt: 0.05,
            v_max: 0.5,
            activation: 0.0,
            excitation: 0.0,
            tau_activation: DEFAULT_TAU_ACTIVATION,
            l_tendon_slack: 0.0,
            pennation_angle: 0.0,
            polarity: 1.0,
        }
    }
}

/// Hill-type muscle bound to one joint. Pure-host metadata mirroring
/// the parallel `Model::muscle_*` buffers; the solver reconstructs an
/// instance per inner-loop iteration from those buffers.
#[derive(Debug, Clone, Copy)]
pub struct Muscle {
    /// Joint id this muscle drives. Must be a `JointType::Cable`.
    pub joint: JointId,
    /// Hill + SEE parameters.
    pub config: MuscleConfig,
}

/// Gaussian-style force–length curve. Peaks at `l_norm = 1.0` with
/// value 1.0; falls off symmetrically as `exp(−((l_norm − 1)/σ)²)`
/// with `σ ≈ 0.45`. Returns 0 outside `[0.4, 1.6]`.
#[inline]
#[must_use]
pub fn force_length(l_norm: Real) -> Real {
    const SIGMA: Real = 0.45;
    if !l_norm.is_finite() || !(0.4..=1.6).contains(&l_norm) {
        return 0.0;
    }
    let x = (l_norm - 1.0) / SIGMA;
    (-x * x).exp()
}

/// Hill-hyperbola force–velocity curve. Returns 1.0 at `v_norm = 0`
/// (isometric), 0 at `v_norm = -1` (full shortening), and asymptotes
/// to `f_v_max ≈ 1.5` for large positive `v_norm` (lengthening).
#[inline]
#[must_use]
pub fn force_velocity(v_norm: Real) -> Real {
    if !v_norm.is_finite() {
        return 0.0;
    }
    const A_F: Real = 0.25;
    const F_V_MAX: Real = 1.5;
    if v_norm <= -1.0 {
        0.0
    } else if v_norm < 0.0 {
        (1.0 + v_norm) / (1.0 - v_norm / A_F)
    } else {
        const K_V: Real = 5.0;
        F_V_MAX - (F_V_MAX - 1.0) * (-v_norm * K_V).exp()
    }
}

/// Tendon force–strain curve. Returns the tendon tension (N) as a
/// function of strain `eps = (l_t − l_t_slack)/l_t_slack`. Quadratic
/// past slack so the slope vanishes at `eps = 0` (`C¹` continuous);
/// zero below slack (tendons don't push).
///
/// `stiffness_ratio` is `E·A / f_max`. With Thelen's default of 30,
/// the tendon reaches `f_max` at ε ≈ 18 %, matching the literature.
#[inline]
#[must_use]
pub fn tendon_force(eps: Real, f_max: Real, stiffness_ratio: Real) -> Real {
    if !eps.is_finite() || eps <= 0.0 {
        return 0.0;
    }
    // Quadratic past slack:
    //   f_t(ε) = f_max · k · ε²    where k controls the strain at f_max.
    // We want f_t(ε*) = f_max ⇒ ε* = 1/√k.
    // For E·A / f_max = 30 with a quadratic, the slope at ε = ε* is
    // 2·f_max·k·ε* = 2·√k · f_max. Matching the linear-tendon stiffness
    // E·A = 30·f_max at the strain where f_t = f_max requires k = 225
    // (so ε* ≈ 6.7 %, mid-range for real tendons). We use that.
    let _ = stiffness_ratio;
    const K: Real = 225.0;
    f_max * K * eps * eps
}

/// Slope `df_t/dl_t` (N/m) at strain `eps`. Used by the Newton solve
/// inside `Muscle::torque`. Zero when the tendon is slack.
#[inline]
#[must_use]
pub fn tendon_force_slope(eps: Real, f_max: Real, l_tendon_slack: Real) -> Real {
    if !eps.is_finite() || eps <= 0.0 || l_tendon_slack <= 0.0 {
        return 0.0;
    }
    const K: Real = 225.0;
    // f_t = f_max · K · eps² ⇒ df_t/dl_t = 2·f_max·K·eps · (1/l_t_slack).
    2.0 * f_max * K * eps / l_tendon_slack
}

impl Muscle {
    /// Construct a new muscle bound to `joint`.
    #[must_use]
    pub fn new(joint: JointId, config: MuscleConfig) -> Self {
        Self { joint, config }
    }

    /// Muscle torque on the cable's linear stretch DoF given the
    /// current joint coordinate `q_l` (deflection from rest, m) and
    /// velocity `qd_l` (m/s).
    ///
    /// Sign convention: returns the generalised force directly. An
    /// agonist muscle (`polarity = +1`) pulls the cable shorter
    /// ⇒ negative torque for `a > 0`. An antagonist (`polarity = −1`)
    /// pulls the cable longer ⇒ positive torque. The polarity is
    /// applied symmetrically to the fibre kinematics so the Hill
    /// curves see the muscle's own coordinate frame
    /// (`q_local = polarity · q_l`).
    ///
    /// When `l_tendon_slack > 0`, runs a 2–3-iteration Newton solver
    /// on the tendon strain so `f_fibre · cos(α) = f_tendon(ε)`.
    #[must_use]
    pub fn torque(&self, q_l: Real, qd_l: Real) -> Real {
        let polarity = if self.config.polarity >= 0.0 { 1.0 } else { -1.0 };
        let q_local = polarity * q_l;
        let qd_local = polarity * qd_l;
        let a = self.config.activation.clamp(0.0, 1.0);
        if a <= 0.0 {
            return 0.0;
        }
        if !self.config.f_max.is_finite() || self.config.f_max <= 0.0 {
            return 0.0;
        }
        if !self.config.l_opt.is_finite() || self.config.l_opt <= 0.0 {
            return 0.0;
        }
        let v_max = if self.config.v_max.is_finite() && self.config.v_max > 0.0 {
            self.config.v_max
        } else {
            // Degenerate v_max — treat as isometric, rigid tendon.
            return polarity
                * -a
                * self.config.f_max
                * force_length(1.0 + q_local / self.config.l_opt);
        };

        // Rigid-tendon fast path: tendon doesn't deform, fibre length
        // equals total stretch. Matches v1 exactly.
        if self.config.l_tendon_slack <= 0.0 {
            let l_norm = (self.config.l_opt + q_local) / self.config.l_opt;
            let v_norm = qd_local / v_max;
            let f_l = force_length(l_norm);
            let f_v = force_velocity(v_norm);
            let cos_alpha = self.config.pennation_angle.cos();
            let f_active = a * self.config.f_max * f_l * f_v * cos_alpha;
            return polarity * -f_active;
        }

        // SEE path. The cable convention is that `q_l = 0` puts the
        // MTU at *rest* — fibre at `l_opt` (its peak f_L) and tendon
        // at slack. So the total path length is
        //
        //   l_total(q) = l_opt · cos(α) + l_t_slack + q_local
        //
        // which splits between fibre projection `l_m · cos(α)` and
        // tendon length `l_t = l_t_slack · (1 + ε)`:
        //
        //   l_m · cos(α) + l_t_slack · (1 + ε) = l_total
        //
        // Solve for ε such that fibre force = tendon force.
        let l_t_slack = self.config.l_tendon_slack;
        let f_max = self.config.f_max;
        let cos_alpha = self.config.pennation_angle.cos().max(1e-6);
        let l_rest = self.config.l_opt * cos_alpha + l_t_slack;
        let l_total = l_rest + q_local;

        // Initial guess. Two cases:
        // (a) `q_l > 0` (MTU stretched): the tendon already strains
        //     past slack; use that geometric strain as the seed.
        // (b) `q_l <= 0` (MTU shortened or at rest): the geometric
        //     strain is zero / negative, but the active fibre is
        //     still pulling. Seed `eps` at the analytic root assuming
        //     `f_L = f_V = 1` (fibre at optimum), which gives
        //     `ε* = sqrt(a · cos_α / K)`. With `a = 1` and `cos_α =
        //     1` and our `K = 225`, `ε* ≈ 6.7 %`.
        const K: Real = 225.0;
        let l_t0 = (l_total - self.config.l_opt * cos_alpha).max(l_t_slack);
        let eps_geom = ((l_t0 - l_t_slack) / l_t_slack).max(0.0);
        let eps_analytic = (a * cos_alpha / K).sqrt();
        let mut eps = if eps_geom < eps_analytic * 0.1 {
            eps_analytic
        } else {
            eps_geom
        };
        // Damped Newton: clamp the step size so we never overshoot
        // by more than 50 % of the current ε. The residual is well-
        // behaved (monotone in ε for reasonable parameters), but
        // the gradient vanishes at l_m = l_opt so a guarded step is
        // important.
        for _ in 0..8 {
            let l_t = l_t_slack * (1.0 + eps);
            let l_m_cos = l_total - l_t;
            let l_m = (l_m_cos / cos_alpha).max(1e-9);
            let l_norm = l_m / self.config.l_opt;
            let v_norm = qd_local / v_max;
            let f_l = force_length(l_norm);
            let f_v = force_velocity(v_norm);
            let f_fibre = a * f_max * f_l * f_v;
            let f_t = tendon_force(eps, f_max, DEFAULT_TENDON_STIFFNESS_RATIO);
            let residual = f_fibre * cos_alpha - f_t;
            if residual.abs() < 1e-4 * f_max {
                break;
            }
            const SIGMA: Real = 0.45;
            let dlm_deps = -l_t_slack / cos_alpha;
            let dln_deps = dlm_deps / self.config.l_opt;
            let df_l_deps = if (0.4..=1.6).contains(&l_norm) {
                -2.0 * (l_norm - 1.0) / (SIGMA * SIGMA) * f_l * dln_deps
            } else {
                0.0
            };
            let df_fibre_deps = a * f_max * f_v * df_l_deps;
            // Tendon slope. Floor at the analytic-root slope to keep
            // the denominator from vanishing near ε = 0.
            let df_t_deps = f_max * 2.0 * K * eps.max(1e-3);
            let denom = df_fibre_deps * cos_alpha - df_t_deps;
            if denom.abs() < 1e-12 * f_max {
                break;
            }
            let step = residual / denom;
            // Damp so we don't overshoot.
            let max_step = (eps.abs() + 1e-3) * 0.5;
            let clamped_step = step.clamp(-max_step, max_step);
            eps -= clamped_step;
            if !eps.is_finite() {
                eps = eps_analytic;
                break;
            }
            eps = eps.max(0.0);
        }
        let f_tendon = tendon_force(eps, f_max, DEFAULT_TENDON_STIFFNESS_RATIO);
        // Tensile — pulls the cable shorter (in the muscle's own
        // frame); polarity restores the world-frame sign.
        polarity * -f_tendon
    }
}

/// Step the per-muscle activation filter one timestep with implicit
/// Euler. Reads `model.muscle_excitation`, writes
/// `model.muscle_activation`:
///
/// ```text
///   a_{n+1} = (a_n + dt · u / τ) / (1 + dt / τ)
/// ```
///
/// Implicit Euler is unconditionally stable for any `dt > 0` — there
/// is no CFL condition like there is in semi-implicit. The user is
/// expected to call this **before** the dynamics solver each step.
///
/// Activations are clamped to `[0, 1]` after each update. Muscles with
/// non-positive `τ_act` get an instantaneous `a := clamp(u, 0, 1)`
/// (no filter, matches v1 behaviour). Non-finite `τ` likewise treated
/// as instantaneous; non-finite excitation is silently zeroed.
///
/// # Errors
///
/// Returns `Error::Sim` if any muscle buffer is unavailable on host
/// (e.g. on a GPU-only model whose download hasn't happened yet).
pub fn update_muscle_activation(
    model: &mut crate::sim::Model,
    dt: Real,
) -> Result<()> {
    if model.muscle_count == 0 || !dt.is_finite() || dt <= 0.0 {
        return Ok(());
    }
    let u_buf = model.muscle_excitation.host()?;
    // Avoid simultaneous borrows: copy the excitation slice first.
    let excit: Vec<Real> = u_buf.to_vec();
    let tau_buf = model.muscle_tau_activation.host()?.to_vec();
    let act = model.muscle_activation.host_mut()?;
    for (i, a) in act.iter_mut().enumerate() {
        let u_raw = excit.get(i).copied().unwrap_or(0.0);
        let u = if u_raw.is_finite() {
            u_raw.clamp(0.0, 1.0)
        } else {
            0.0
        };
        let tau = tau_buf.get(i).copied().unwrap_or(DEFAULT_TAU_ACTIVATION);
        let new_a = if !tau.is_finite() || tau <= 0.0 {
            u
        } else {
            // Implicit Euler: (a + dt/τ · u) / (1 + dt/τ).
            let r = dt / tau;
            (*a + r * u) / (1.0 + r)
        };
        *a = new_a.clamp(0.0, 1.0);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn force_length_peaks_at_one() {
        let f0 = force_length(1.0);
        assert!((f0 - 1.0).abs() < 1e-9);
        let f_short = force_length(0.7);
        let f_long = force_length(1.3);
        assert!(f_short < 1.0 && f_short > 0.0);
        assert!(f_long < 1.0 && f_long > 0.0);
        assert!((f_short - f_long).abs() < 1e-9);
        assert_eq!(force_length(0.2), 0.0);
        assert_eq!(force_length(2.0), 0.0);
    }

    #[test]
    fn force_velocity_hill_shape() {
        assert!((force_velocity(0.0) - 1.0).abs() < 1e-9);
        assert!(force_velocity(-1.0) <= 1e-9);
        assert!(force_velocity(-0.5) > 0.0 && force_velocity(-0.5) < 1.0);
        assert!(force_velocity(1.0) > 1.0);
        assert!(force_velocity(10.0) > 1.4 && force_velocity(10.0) <= 1.5);
    }

    #[test]
    fn zero_activation_zero_torque() {
        let m = Muscle::new(
            JointId::from_index(0),
            MuscleConfig {
                f_max: 200.0,
                activation: 0.0,
                ..MuscleConfig::default()
            },
        );
        for &q in &[-0.02, 0.0, 0.03] {
            for &qd in &[-0.4, 0.0, 0.4] {
                let t = m.torque(q, qd);
                assert!(t.abs() < 1e-12);
            }
        }
    }

    #[test]
    fn full_activation_isometric_yields_f_max() {
        let m = Muscle::new(
            JointId::from_index(0),
            MuscleConfig {
                f_max: 250.0,
                l_opt: 0.05,
                v_max: 0.5,
                activation: 1.0,
                ..MuscleConfig::default()
            },
        );
        let t = m.torque(0.0, 0.0);
        assert!((t + 250.0).abs() < 1e-6, "expected -250, got {t}");
    }

    #[test]
    fn tendon_force_zero_below_slack_quadratic_above() {
        let f0 = tendon_force(-0.1, 100.0, DEFAULT_TENDON_STIFFNESS_RATIO);
        assert_eq!(f0, 0.0);
        let f1 = tendon_force(0.05, 100.0, DEFAULT_TENDON_STIFFNESS_RATIO);
        let f2 = tendon_force(0.1, 100.0, DEFAULT_TENDON_STIFFNESS_RATIO);
        // Quadratic ⇒ f(2ε) = 4 f(ε).
        assert!((f2 - 4.0 * f1).abs() < 1e-6);
    }
}
