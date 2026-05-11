//! Stage 42 — Hill-type muscle activation for CABLE joints.
//!
//! Adds a per-joint biomechanical muscle actuator that produces a
//! tensile force on a `JointType::Cable`'s **linear stretch DoF**. The
//! force is computed from the classic Hill three-element model
//!
//! ```text
//!   f_active(a, l, v) = a · f_max · f_L(l / l_opt) · f_V(v / v_max)
//! ```
//!
//! where:
//!   - `a ∈ [0, 1]` is the muscle activation (instantaneous in v1);
//!   - `f_max` is the maximum isometric force (N);
//!   - `f_L` is a length–tension curve, Gaussian-shaped around the
//!     optimal length `l_opt` (m);
//!   - `f_V` is a force–velocity curve, the Hill hyperbola in m/s
//!     scaled by `v_max` (m/s).
//!
//! The cable's linear coordinate `q_l` is **deflection from rest**.
//! Hence the current muscle fibre length is `l = l_opt + q_l`. A
//! contracting muscle (positive activation) pulls the cable shorter
//! → it produces a **negative** torque on `q_l` (i.e. towards
//! `q_l < 0`). This mirrors `JointType::Cable`'s existing convention
//! that `τ_l = −k_l · q_l` is a restoring spring (the muscle simply
//! biases the rest length downward at full activation).
//!
//! ## Design notes
//!
//! - **Always compiled** — no `cfg(feature = "gpu")` gate. The host
//!   path is consulted from `solvers::featherstone` (CPU) directly,
//!   and from the GPU kernel via the model's parallel buffers (when
//!   the GPU Featherstone lands, Stage 43+).
//! - **Single muscle per joint** in v1. Co-contraction (antagonist
//!   pair on the same DoF) is left for v2 — see the module docs.
//! - **No activation dynamics**: caller writes
//!   `model.muscle_activation[id]` directly. Excitation → activation
//!   lag (~10–30 ms first-order filter in biological muscle) is left
//!   for v2; document in the user-facing release notes that fast
//!   activation transients are unfiltered.
//! - **No series elasticity / pennation / passive curve** — the v1
//!   model is purely active-contractile. Passive stretch beyond
//!   `l_opt` returns zero force; the user can supplement with the
//!   built-in cable spring (`stiffness_linear`) for passive recoil.
//!
//! ## References
//!
//! - Newton's `kernels_muscle.py` (`xtra/newton/_src/solvers/
//!   semi_implicit/kernels_muscle.py`) — the per-cable activation
//!   pattern this module mirrors, simplified to a single fibre per
//!   joint.
//! - Zajac (1989), "Muscle and Tendon: Properties, Models, Scaling,
//!   and Application to Biomechanics and Motor Control."
//! - Thelen (2003), "Adjustment of Muscle Mechanics Model Parameters
//!   to Simulate Dynamic Contractions in Older Adults."

use crate::core::JointId;
use crate::math::Real;

/// Configuration for a single Hill-type muscle attached to one
/// CABLE joint. All values are SI: forces in newtons, lengths in
/// metres, velocities in m/s. The activation defaults to `0.0`
/// (off); the caller drives it through `Model::set_muscle_activation`
/// (or the `model.muscle_activation` buffer directly) once the model
/// is built.
#[derive(Debug, Clone, Copy)]
pub struct MuscleConfig {
    /// Maximum isometric force, `f_max` (N). Force generated at full
    /// activation, optimal length, zero velocity.
    pub f_max: Real,
    /// Optimal fibre length, `l_opt` (m). The force–length curve
    /// peaks at the cable length `l_opt`; in cable coordinates the
    /// fibre is at `l_opt` when `q_l = 0`.
    pub l_opt: Real,
    /// Maximum shortening velocity, `v_max` (m/s). Force drops to
    /// zero when the fibre contracts at `−v_max`.
    pub v_max: Real,
    /// Initial activation level, `a ∈ [0, 1]`. Defaults to 0 (off).
    pub activation: Real,
}

impl Default for MuscleConfig {
    /// Defaults chosen for a "medium" mammalian limb muscle: 100 N
    /// max isometric force, 5 cm optimal fibre length, 0.5 m/s
    /// peak contraction velocity (≈ 10·l_opt/s — within the
    /// biological 8–12·l_opt/s window for fast-twitch muscle).
    fn default() -> Self {
        Self {
            f_max: 100.0,
            l_opt: 0.05,
            v_max: 0.5,
            activation: 0.0,
        }
    }
}

/// Hill-type single-muscle model attached to one joint. Pure-host
/// data; the parallel `model.muscle_*` buffers store the same fields
/// indexed by muscle id for the solver.
#[derive(Debug, Clone, Copy)]
pub struct Muscle {
    /// Joint id this muscle drives. Must be a `JointType::Cable`.
    pub joint: JointId,
    /// Hill model parameters.
    pub config: MuscleConfig,
}

/// Gaussian-style force–length curve. Peaks at `l_norm = 1.0` with
/// value 1.0; falls off symmetrically as `exp(−((l_norm − 1)/σ)²)`
/// with `σ ≈ 0.45`. Returns 0 outside the active range.
///
/// `σ = 0.45` is the Thelen-2003 default — gives F/F_max ≈ 0.4 at
/// `l_norm = 0.6` and `l_norm = 1.4`, matching empirical sarcomere
/// force–length curves.
#[inline]
#[must_use]
pub fn force_length(l_norm: Real) -> Real {
    const SIGMA: Real = 0.45;
    // Clamp to the physiologically active range; outside [0.4, 1.6]
    // the curve is essentially zero, and we don't want spurious
    // floating-point tails to drive the solver.
    if !l_norm.is_finite() || !(0.4..=1.6).contains(&l_norm) {
        return 0.0;
    }
    let x = (l_norm - 1.0) / SIGMA;
    (-x * x).exp()
}

/// Hill-hyperbola force–velocity curve. Returns 1.0 at `v_norm = 0`
/// (isometric), drops to 0 at `v_norm = -1` (full shortening), and
/// asymptotes to ~1.5 at large positive `v_norm` (lengthening). The
/// sign convention: **negative** `v_norm` is shortening (which is
/// what a contracting muscle wants), **positive** is lengthening.
///
/// Standard Hill parameters from Zajac (1989): shortening slope
/// curvature `a_f = 0.25`, lengthening asymptote `f_v_max = 1.5`,
/// blend curvature `k_pe = 5.0`.
#[inline]
#[must_use]
pub fn force_velocity(v_norm: Real) -> Real {
    if !v_norm.is_finite() {
        return 0.0;
    }
    const A_F: Real = 0.25;
    const F_V_MAX: Real = 1.5;

    if v_norm <= -1.0 {
        // Fully shortening beyond v_max — no active force.
        0.0
    } else if v_norm < 0.0 {
        // Hyperbolic shortening branch (Hill):
        //   f_v(v) = (1 + v_n) / (1 - v_n / a_f)
        // Returns 1 at v=0, 0 at v=-1. Equivalent to the classic
        // Hill (v_max + v) / (v_max - v / a) form normalised.
        (1.0 + v_norm) / (1.0 - v_norm / A_F)
    } else {
        // Lengthening branch — asymptotes to f_v_max.
        //   f_v(v) = (f_v_max · v_n + 1) / (v_n + 1)  (simple, smooth at 0)
        // Tweaked to a Thelen-style sigmoid:
        //   f_v(v) = f_v_max − (f_v_max − 1) · exp(−v_n · k_v)
        // where k_v ≈ 5 keeps the slope continuous with shortening.
        const K_V: Real = 5.0;
        F_V_MAX - (F_V_MAX - 1.0) * (-v_norm * K_V).exp()
    }
}

impl Muscle {
    /// Construct a new muscle bound to `joint`. Validate-on-build
    /// only enforces that the joint is a cable; bounds-checks on
    /// activation happen lazily in `torque` (clamped to `[0, 1]`).
    #[must_use]
    pub fn new(joint: JointId, config: MuscleConfig) -> Self {
        Self { joint, config }
    }

    /// Muscle torque on the cable's linear stretch DoF given the
    /// current joint coordinate `q_l` (deflection from rest, m) and
    /// velocity `qd_l` (m/s).
    ///
    /// Sign convention: the returned value is added directly into the
    /// generalised force `τ_l`. Active muscle force is **tensile** —
    /// it pulls the cable shorter, i.e. `τ_l < 0` for `a > 0`.
    #[must_use]
    pub fn torque(&self, q_l: Real, qd_l: Real) -> Real {
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
            // Degenerate v_max — treat as isometric only.
            return -a * self.config.f_max * force_length(1.0 + q_l / self.config.l_opt);
        };

        let l_norm = (self.config.l_opt + q_l) / self.config.l_opt;
        // `qd_l` is dl/dt: positive = lengthening, negative = shortening.
        // The Hill force–velocity convention is the same.
        let v_norm = qd_l / v_max;
        let f_l = force_length(l_norm);
        let f_v = force_velocity(v_norm);
        let f_active = a * self.config.f_max * f_l * f_v;
        // Tensile force pulls the cable shorter → negative torque on
        // the stretch coordinate.
        -f_active
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Force-length curve peaks at l = l_opt and falls off symmetrically.
    #[test]
    fn force_length_peaks_at_one() {
        let f0 = force_length(1.0);
        assert!((f0 - 1.0).abs() < 1e-9, "f_L(1.0) should be 1.0, got {f0}");

        let f_short = force_length(0.7);
        let f_long = force_length(1.3);
        assert!(f_short < 1.0 && f_short > 0.0);
        assert!(f_long < 1.0 && f_long > 0.0);
        // Symmetric.
        assert!((f_short - f_long).abs() < 1e-9);

        // Out of range → zero (defensive).
        assert_eq!(force_length(0.2), 0.0);
        assert_eq!(force_length(2.0), 0.0);
    }

    /// Force-velocity is 1 at v=0, 0 at v=-v_max, > 1 for lengthening.
    #[test]
    fn force_velocity_hill_shape() {
        assert!((force_velocity(0.0) - 1.0).abs() < 1e-9);
        assert!(force_velocity(-1.0) <= 1e-9);
        assert!(force_velocity(-0.5) > 0.0 && force_velocity(-0.5) < 1.0);
        // Lengthening: > 1, approaches f_v_max = 1.5.
        assert!(force_velocity(1.0) > 1.0);
        assert!(force_velocity(10.0) > 1.4 && force_velocity(10.0) <= 1.5);
    }

    /// Zero activation always gives zero force regardless of length / velocity.
    #[test]
    fn zero_activation_zero_torque() {
        let m = Muscle::new(
            JointId::from_index(0),
            MuscleConfig {
                f_max: 200.0,
                l_opt: 0.05,
                v_max: 0.5,
                activation: 0.0,
            },
        );
        for &q in &[-0.02, 0.0, 0.03] {
            for &qd in &[-0.4, 0.0, 0.4] {
                let t = m.torque(q, qd);
                assert!(t.abs() < 1e-12, "expected 0, got {t} at q={q} qd={qd}");
            }
        }
    }

    /// Full activation at rest length and zero velocity → -f_max torque
    /// (tensile sign).
    #[test]
    fn full_activation_isometric_yields_f_max() {
        let m = Muscle::new(
            JointId::from_index(0),
            MuscleConfig {
                f_max: 250.0,
                l_opt: 0.05,
                v_max: 0.5,
                activation: 1.0,
            },
        );
        let t = m.torque(0.0, 0.0);
        assert!((t + 250.0).abs() < 1e-6, "expected -250, got {t}");
    }
}
