//! Stage 49 — Signorini hard non-penetration contact via Lagrange
//! multipliers (fixed active set).
//! Stage 50 — Coulomb friction at the Signorini contact interface.
//! Stage 51 — Primal-Dual Active-Set (PDAS) Newton outer loop — see
//! [`solve_active_set_signorini`] and [`SignoriniContactCandidates`]
//! near the bottom of this file.
//!
//! Stage 47 v2 ships **penalty contact**, which approximates the
//! Signorini condition `gap_normal ≥ 0` with the soft spring
//! `f_contact = −k · Δu`. The penalty matrix `k C^T C` becomes
//! ill-conditioned as `k → ∞`, capping how hard the constraint is in
//! practice. Stage 49 replaces the spring with **Lagrange multipliers
//! `λ`** that enforce the non-penetration constraint exactly:
//!
//! ```text
//!   minimise   ½ u^T K u − f^T u
//!   subject to C u ≥ g           (gap_normal ≥ g — penetration removed)
//!              λ ≥ 0              (only push, never pull)
//! ```
//!
//! KKT stationarity gives `K u − C^T λ = f`. In matrix form the
//! linearised saddle-point system reads
//!
//! ```text
//!   [ K   −C^T ] [ u ]   [ f ]
//!   [ C    0   ] [ λ ] = [ g ]      (with λ ≥ 0)
//! ```
//!
//! (The PLAN's `[K, C^T; C, 0]` convention flips the sign of `λ` — the
//! algebra is identical up to a `λ ← −λ` relabel. We use the
//! `λ ≥ 0`-friendly form below; the reported `λ` is the contact pressure
//! magnitude pointing along the master surface normal.)
//!
//! # Stage 50 — Coulomb friction
//!
//! Stage 50 adds **tangential reactions** at every active contact pair.
//! Each pair carries an orthonormal tangent basis `(t_u, t_v)` spanning
//! the master tangent plane (orthogonal to the normal `n`). The
//! constraint matrix is augmented with two extra rows per pair that
//! project the slave's displacement onto each tangent direction:
//!
//! ```text
//!   C_full = [ C_n ; C_tu ; C_tv ]      (3 rows per pair, stacked)
//! ```
//!
//! and three multipliers per pair `(λ_n, λ_tu, λ_tv)`. The friction
//! cone is enforced by projecting the dual pair `(λ_tu, λ_tv)` onto the
//! disc of radius `μ · λ_n` at every Uzawa step:
//!
//! ```text
//!   λ_n ← max(0, λ_n + ρ · residual_n)                       (normal push)
//!   (λ_tu, λ_tv) ← proj_disc_{μ λ_n} (λ_tu + ρ · res_tu,
//!                                     λ_tv + ρ · res_tv)     (Coulomb cap)
//! ```
//!
//! At each contact, this is exactly the classical Coulomb stick/slip
//! split: if the unprojected tangential multiplier lies *inside* the
//! disc the constraint is sticking and the projection is a no-op; if it
//! lies *outside* we radially rescale to the disc boundary, recovering
//! the slip case where `|λ_t| = μ · λ_n`.
//!
//! The friction Uzawa is otherwise unchanged: the inner CG solves
//! `K u = f + C_full^T λ_full` with the now-stacked constraint matrix.
//!
//! # Algorithm — Uzawa iteration
//!
//! Uzawa is the textbook dual-ascent / inner-primal-solve scheme:
//!
//! 1. **Inner primal** — fix `λ`, solve `K u = f + C^T λ` via CG.
//! 2. **Dual update** — gradient ascent on the (concave) dual
//!    functional with projection onto the admissible set:
//!    - `λ_n ← max(0, λ_n + ρ · (g − C_n u))` (normal cap),
//!    - `(λ_tu, λ_tv) ← proj_disc_{μ λ_n}(λ_t + ρ · (0 − C_t u))`
//!      (friction cone).
//! 3. Iterate until `‖g − C u‖_∞ < tol` and `λ` has stabilised.
//!
//! # Choice of dual step `ρ`
//!
//! For symmetric positive-definite `K`, the dual functional is concave
//! with Lipschitz gradient bounded by the largest eigenvalue of the
//! Schur complement `S = C K⁻¹ C^T`. Convergence requires
//! `0 < ρ < 2 / λ_max(S)`. We estimate `λ_max(S)` via a short power
//! iteration that needs only matrix-free actions of `K⁻¹` (via an
//! inner CG solve) and `C`. The default step is
//! `ρ = 1 / λ_max(S)` — within a factor of two of the optimum
//! `ρ_opt = 2 / (λ_min(S) + λ_max(S))` and conservative enough to
//! avoid oscillation across the wide range of `S` spectra produced by
//! FEM contact problems.
//!
//! For Stage 50 we estimate `λ_max(S)` on the **full stacked** `C` (so
//! the bound covers both normal and tangential rows). The same `ρ` is
//! used for every multiplier.
//!
//! # Scope — fixed active set
//!
//! Stage 49 solves the **augmented system for a fixed list of active
//! contact pairs**. The caller is responsible for deciding which pairs
//! are active (typically: any slave node currently penetrating).
//! **Stage 51** adds the **active-set update loop** on top:
//! [`solve_active_set_signorini`] takes a list of *candidate* slaves
//! plus a gap closure, detects new penetrations after each Uzawa
//! solve, drops pairs whose `λ` clamps to zero (separating), and
//! re-solves until the active set stops changing.
//!
//! # Out of scope (deferred)
//!
//! - **Rate-dependent friction** (Stribeck curve). Stage 50 v2 supports
//!   separate static/kinetic Coulomb coefficients, but not velocity-dependent
//!   Stribeck smoothing.
//! - **Dahl friction** (smooth stick-slip with memory) — handled by
//!   Stage 33.4's cable kernel; not duplicated here.
//! - **GPU port** of the friction projection — currently CPU-only.
//!
//! # Reference
//!
//! - Wriggers, *Computational Contact Mechanics*, §6.3 (Uzawa-Lagrange)
//!   and §10 (Coulomb friction projection).
//! - Newton's `signorini.py` and Warp's `xpbd_contact.py` for the active-
//!   set logic that comes in v2.

use std::collections::{HashMap, HashSet};

use crate::core::{Error, Result};
use crate::geometry::SelfContactCandidate;
use crate::math::iterative::cg_solve;
use crate::math::sparse::BsrMatrix;
use crate::math::{Real, Vec3};

use super::assembly::matvec;

/// Default inner-CG tolerance for the `K u = f + C^T λ` solve.
/// Tighter than the outer Uzawa tol so the inner residual does not
/// pollute the dual update — a standard "nested-iteration" rule of
/// thumb is to keep the inner tol ~10× tighter than the outer.
pub const DEFAULT_INNER_CG_TOL: Real = 1.0e-11;
/// Default cap on inner CG iterations. `K` is well-conditioned for
/// small elasticity blocks; 2000 iterations is generous.
pub const DEFAULT_INNER_CG_MAX_ITERS: usize = 2000;

/// One Signorini contact pair with full normal + tangent-plane bases.
///
/// Stage 49 only used `(slave_dof, normal, target_gap)`. Stage 50 adds
/// the orthonormal tangent basis `(tangent_u, tangent_v)` spanning the
/// master tangent plane, which is required for Coulomb friction. For a
/// frictionless solve the tangents are still populated (any
/// `n`-orthogonal pair) but ignored by the friction projection because
/// `μ = 0`.
///
/// Fields:
/// - `slave_dof`: BASE global DoF index of the slave vertex
///   (i.e. `vertex_index * dim`). The constraint operator writes the
///   normal/tangent components into the three consecutive DoFs starting
///   here.
/// - `normal`: unit outward normal of the master plane at the contact
///   point. The constraint enforces `n · u_slave ≥ target_gap`.
/// - `target_gap`: minimum allowed projected displacement. If the
///   initial geometry has a penetration `−p < 0`, set `target_gap = p`
///   so that `n · u_slave ≥ p` restores contact at zero gap. For nodes
///   that are merely *on* the master surface, `target_gap = 0`.
/// - `tangent_u`, `tangent_v`: orthonormal pair spanning the tangent
///   plane (i.e. perpendicular to `normal` and to each other). The
///   friction cone is enforced on the dual pair acting in this basis.
#[derive(Clone, Copy, Debug)]
pub struct ContactPair {
    /// Base global DoF index of the slave vertex.
    pub slave_dof: u32,
    /// Outward unit normal of the master plane at this contact point.
    pub normal: Vec3,
    /// Target gap (≥ 0; usually 0 for fresh contacts, positive when
    /// resolving an initial penetration of magnitude `target_gap`).
    pub target_gap: Real,
    /// First tangent-plane basis vector (unit, ⟂ normal).
    pub tangent_u: Vec3,
    /// Second tangent-plane basis vector (unit, ⟂ normal, ⟂ tangent_u).
    pub tangent_v: Vec3,
}

impl ContactPair {
    /// Build a pair from `(slave_dof, normal, target_gap)`, auto-
    /// generating an orthonormal tangent basis from the normal.
    ///
    /// The basis is chosen deterministically (cross with the world axis
    /// least aligned with `normal`); friction is rotationally invariant
    /// inside the tangent plane so the specific choice does not affect
    /// the converged solution.
    #[must_use]
    pub fn from_normal(slave_dof: u32, normal: Vec3, target_gap: Real) -> Self {
        let (tu, tv) = orthonormal_tangents(normal);
        Self {
            slave_dof,
            normal,
            target_gap,
            tangent_u: tu,
            tangent_v: tv,
        }
    }
}

/// Build an orthonormal tangent-plane basis `(t_u, t_v)` from `n`.
///
/// Picks the world axis least aligned with `n` as the seed, crosses it
/// against `n` to get `t_u`, then crosses `n × t_u` to get `t_v`. The
/// basis is always right-handed: `n × t_u = t_v`.
fn orthonormal_tangents(normal: Vec3) -> (Vec3, Vec3) {
    let nx = normal.x.abs();
    let ny = normal.y.abs();
    let nz = normal.z.abs();
    // Pick the world axis least aligned with `normal`.
    let seed = if nx <= ny && nx <= nz {
        Vec3::new(1.0, 0.0, 0.0)
    } else if ny <= nx && ny <= nz {
        Vec3::new(0.0, 1.0, 0.0)
    } else {
        Vec3::new(0.0, 0.0, 1.0)
    };
    let t_u_raw = seed - normal * normal.dot(seed);
    let len_u = t_u_raw.length();
    let t_u = if len_u > 1.0e-12 {
        t_u_raw / len_u
    } else {
        // Degenerate fallback (normal exactly equals seed — should not
        // happen given the "least aligned" choice above, but be safe).
        Vec3::new(0.0, 1.0, 0.0)
    };
    let t_v = normal.cross(t_u);
    (t_u, t_v)
}

/// Fixed list of Signorini contact pairs to enforce, plus optional
/// Coulomb friction.
///
/// The active-set update logic (deciding *which* pairs end up here) is
/// out of scope for Stage 49/50 — the caller assembles `active_pairs`
/// from their own contact-detection pass.
#[derive(Clone, Debug)]
pub struct SignoriniContact {
    /// One [`ContactPair`] per active constraint.
    pub active_pairs: Vec<ContactPair>,
    /// Legacy Coulomb friction coefficient `μ` applied uniformly to every
    /// pair. `with_friction(μ)` sets both static and kinetic coefficients to
    /// this value, preserving Stage 50 v1 behaviour.
    pub friction_coeff: Real,
    /// Static friction coefficient `μ_s` used while a contact can stick.
    pub friction_static_coeff: Real,
    /// Kinetic friction coefficient `μ_k` used after a contact slips.
    pub friction_kinetic_coeff: Real,
}

impl SignoriniContact {
    /// Build an empty, frictionless contact set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            active_pairs: Vec::new(),
            friction_coeff: 0.0,
            friction_static_coeff: 0.0,
            friction_kinetic_coeff: 0.0,
        }
    }

    /// Build a contact set with a given friction coefficient.
    ///
    /// This is the backwards-compatible Stage 50 v1 constructor:
    /// `μ_s == μ_k == friction_coeff`.
    #[must_use]
    pub fn with_friction(friction_coeff: Real) -> Self {
        Self {
            active_pairs: Vec::new(),
            friction_coeff,
            friction_static_coeff: friction_coeff,
            friction_kinetic_coeff: friction_coeff,
        }
    }

    /// Build a contact set with separate static and kinetic Coulomb
    /// coefficients.
    #[must_use]
    pub fn with_static_kinetic_friction(mu_static: Real, mu_kinetic: Real) -> Self {
        Self {
            active_pairs: Vec::new(),
            friction_coeff: mu_static,
            friction_static_coeff: mu_static,
            friction_kinetic_coeff: mu_kinetic,
        }
    }

    /// Number of active contact pairs.
    #[must_use]
    pub fn n_active(&self) -> usize {
        self.active_pairs.len()
    }

    /// Borrow the active pairs as a slice.
    #[must_use]
    pub fn pairs(&self) -> &[ContactPair] {
        &self.active_pairs
    }

    /// Push a contact pair, auto-deriving the tangent basis from the
    /// normal.
    pub fn push_from_normal(&mut self, slave_dof: u32, normal: Vec3, target_gap: Real) {
        self.active_pairs
            .push(ContactPair::from_normal(slave_dof, normal, target_gap));
    }
}

impl Default for SignoriniContact {
    fn default() -> Self {
        Self::new()
    }
}

/// Assemble the sparse **normal-only** constraint matrix `C_n` of shape
/// `n_active × n_dofs`.
///
/// Back-compat entry point preserved from Stage 49: each active pair
/// contributes one row holding `(n.x, n.y, n.z)` at the slave's three
/// consecutive DoFs. Stage 50 prefers
/// [`assemble_full_constraint_matrix`] which stacks normals and
/// tangents.
///
/// Returns [`Error::Build`] if any `slave_dof + 2 >= n_dofs` or if the
/// underlying BSR builder rejects the pattern.
pub fn assemble_constraint_matrix(
    active_pairs: &[(u32, Vec3, Real)],
    n_dofs: usize,
) -> Result<BsrMatrix> {
    let n_active = active_pairs.len();
    if n_active == 0 {
        return BsrMatrix::from_pattern(1, 0, n_dofs.max(1), Vec::new());
    }
    let mut triplets: Vec<(usize, usize, Real)> = Vec::with_capacity(3 * n_active);
    for (k, (slave, normal, _gap)) in active_pairs.iter().enumerate() {
        let base = *slave as usize;
        if base + 2 >= n_dofs {
            return Err(Error::Build(format!(
                "Signorini constraint pair {k}: slave_dof {} + 2 >= n_dofs {}",
                base, n_dofs
            )));
        }
        triplets.push((k, base, normal.x));
        triplets.push((k, base + 1, normal.y));
        triplets.push((k, base + 2, normal.z));
    }
    BsrMatrix::from_scalar_triplets(n_active, n_dofs, &triplets)
}

/// Extract the `g` (target-gap) vector from the back-compat tuple
/// active-pair list.
#[must_use]
pub fn target_gap_vector(active_pairs: &[(u32, Vec3, Real)]) -> Vec<Real> {
    active_pairs.iter().map(|(_, _, g)| *g).collect()
}

/// Assemble the **stacked** constraint matrix `C_full` for Stage 50.
///
/// Layout: `3 · n_active` rows, organised as
/// ```text
///   row 3k + 0 : normal n at slave k        →  enforces (n · u) ≥ g_k
///   row 3k + 1 : tangent_u t_u at slave k   →  observed in C_t u
///   row 3k + 2 : tangent_v t_v at slave k   →  observed in C_t u
/// ```
///
/// The interleaving keeps each pair's three rows contiguous, which is
/// convenient when projecting the dual triple onto the friction cone
/// (Coulomb couples normal and tangential multipliers).
///
/// Returns [`Error::Build`] on out-of-range DoF indices or BSR build
/// errors.
pub fn assemble_full_constraint_matrix(pairs: &[ContactPair], n_dofs: usize) -> Result<BsrMatrix> {
    let n_active = pairs.len();
    if n_active == 0 {
        return BsrMatrix::from_pattern(1, 0, n_dofs.max(1), Vec::new());
    }
    let n_rows = 3 * n_active;
    let mut triplets: Vec<(usize, usize, Real)> = Vec::with_capacity(9 * n_active);
    for (k, pair) in pairs.iter().enumerate() {
        let base = pair.slave_dof as usize;
        if base + 2 >= n_dofs {
            return Err(Error::Build(format!(
                "Signorini constraint pair {k}: slave_dof {} + 2 >= n_dofs {}",
                base, n_dofs
            )));
        }
        let row_n = 3 * k;
        let row_tu = 3 * k + 1;
        let row_tv = 3 * k + 2;
        // Normal row.
        triplets.push((row_n, base, pair.normal.x));
        triplets.push((row_n, base + 1, pair.normal.y));
        triplets.push((row_n, base + 2, pair.normal.z));
        // Tangent-u row.
        triplets.push((row_tu, base, pair.tangent_u.x));
        triplets.push((row_tu, base + 1, pair.tangent_u.y));
        triplets.push((row_tu, base + 2, pair.tangent_u.z));
        // Tangent-v row.
        triplets.push((row_tv, base, pair.tangent_v.x));
        triplets.push((row_tv, base + 1, pair.tangent_v.y));
        triplets.push((row_tv, base + 2, pair.tangent_v.z));
    }
    BsrMatrix::from_scalar_triplets(n_rows, n_dofs, &triplets)
}

/// Convergence summary for [`solve_augmented_system`].
#[derive(Clone, Debug)]
pub struct UzawaResult {
    /// Final displacement field, length `n_dofs`.
    pub u: Vec<Real>,
    /// Final dual multipliers, length `n_active`. All entries
    /// satisfy `λ ≥ 0` modulo the projection step.
    pub lambda: Vec<Real>,
    /// Outer iterations actually performed.
    pub iters: usize,
    /// Final primal residual `‖g − C u‖_∞` (sup-norm of constraint
    /// violation; ≤ 0 entries are clamped to zero so this measures
    /// the *active* violation, not the satisfied slack).
    pub primal_residual: Real,
    /// Final dual increment `‖λ_new − λ_old‖_∞`. Useful for
    /// diagnosing slow active-set drift.
    pub dual_residual: Real,
}

/// Stick/slip classification from the last friction projection for each
/// active contact pair.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FrictionState {
    /// No normal reaction, so the friction cone is inactive.
    Inactive,
    /// Tangential multiplier remained inside the static cone.
    Stick,
    /// Tangential multiplier exceeded the static cone and was projected to the
    /// kinetic-friction boundary.
    Slip,
}

/// Dieterich-Ruina rate-and-state friction parameters.
///
/// The coefficient is
/// `μ = μ0 + a ln(v/v0) + b ln(θ v0 / dc)`, with θ evolved by the
/// aging law `dθ/dt = 1 - v θ / dc`.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RateStateFrictionParams {
    pub mu0: Real,
    pub a: Real,
    pub b: Real,
    pub v0: Real,
    pub dc: Real,
}

impl Default for RateStateFrictionParams {
    fn default() -> Self {
        Self {
            mu0: 0.6,
            a: 0.01,
            b: 0.015,
            v0: 1.0e-6,
            dc: 1.0e-3,
        }
    }
}

/// Per-contact state variable for Dieterich-Ruina friction.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct RateStateFrictionState {
    /// State/age variable θ (seconds).
    pub theta: Real,
}

impl RateStateFrictionState {
    #[must_use]
    pub fn steady(params: RateStateFrictionParams, slip_rate: Real) -> Self {
        let v = slip_rate.max(params.v0 * 1.0e-12);
        Self {
            theta: params.dc / v,
        }
    }
}

/// Evaluate the Dieterich-Ruina coefficient for the current state.
#[must_use]
pub fn rate_state_friction_coeff(
    params: RateStateFrictionParams,
    state: RateStateFrictionState,
    slip_rate: Real,
) -> Real {
    let v = slip_rate.max(params.v0 * 1.0e-12);
    let theta = state.theta.max(params.dc / params.v0 * 1.0e-12);
    params.mu0 + params.a * (v / params.v0).ln() + params.b * (theta * params.v0 / params.dc).ln()
}

/// Advance the Dieterich-Ruina aging-law state by one explicit Euler
/// step and return the updated state plus coefficient.
#[must_use]
pub fn advance_rate_state_friction(
    params: RateStateFrictionParams,
    state: RateStateFrictionState,
    slip_rate: Real,
    dt: Real,
) -> (RateStateFrictionState, Real) {
    let v = slip_rate.max(0.0);
    let theta_dot = 1.0 - v * state.theta / params.dc;
    let theta = (state.theta + dt.max(0.0) * theta_dot).max(1.0e-30);
    let next = RateStateFrictionState { theta };
    let mu = rate_state_friction_coeff(params, next, v);
    (next, mu)
}

/// Convergence summary for [`solve_augmented_system_with_friction`].
///
/// The dual is now split into a length-`n_active` normal vector and a
/// `n_active × 2` tangential matrix (flattened row-major: pair `k` owns
/// entries `[2k]` and `[2k+1]` corresponding to `(λ_tu, λ_tv)`).
#[derive(Clone, Debug)]
pub struct FrictionUzawaResult {
    /// Final displacement field, length `n_dofs`.
    pub u: Vec<Real>,
    /// Final normal multipliers `λ_n`, length `n_active`. Always `≥ 0`.
    pub lambda_normal: Vec<Real>,
    /// Final tangential multipliers, flattened length `2 · n_active`.
    /// Pair `k` occupies `[2k] = λ_tu`, `[2k+1] = λ_tv`. Always inside
    /// the friction disc selected by [`friction_state`](Self::friction_state).
    pub lambda_tangent: Vec<Real>,
    /// Per-pair stick/slip state from the final projection.
    pub friction_state: Vec<FrictionState>,
    /// Outer iterations actually performed.
    pub iters: usize,
    /// Sup-norm of the active normal-constraint violation.
    pub primal_residual: Real,
    /// Sup-norm of the dual increment across all `3 · n_active`
    /// multipliers in the last iteration.
    pub dual_residual: Real,
}

/// Solve the Signorini saddle-point system via projected Uzawa
/// (frictionless, Stage 49 API).
pub fn solve_augmented_system(
    k_global: &BsrMatrix,
    c_matrix: &BsrMatrix,
    f: &[Real],
    g: &[Real],
    max_iters: usize,
    tol: Real,
) -> Result<UzawaResult> {
    let rho = estimate_dual_step(
        k_global,
        c_matrix,
        g.len(),
        DEFAULT_INNER_CG_TOL,
        DEFAULT_INNER_CG_MAX_ITERS,
    );
    solve_augmented_system_with(
        k_global,
        c_matrix,
        f,
        g,
        max_iters,
        tol,
        rho,
        DEFAULT_INNER_CG_TOL,
        DEFAULT_INNER_CG_MAX_ITERS,
    )
}

/// Same as [`solve_augmented_system`] with the dual step, inner CG
/// tolerance, and inner CG iteration cap exposed.
#[allow(clippy::too_many_arguments)]
pub fn solve_augmented_system_with(
    k_global: &BsrMatrix,
    c_matrix: &BsrMatrix,
    f: &[Real],
    g: &[Real],
    max_iters: usize,
    tol: Real,
    rho: Real,
    inner_cg_tol: Real,
    inner_cg_max_iters: usize,
) -> Result<UzawaResult> {
    let n_dofs = f.len();
    if k_global.nrows() != n_dofs || k_global.ncols() != n_dofs {
        return Err(Error::Build(format!(
            "Signorini: K is {}×{} but f has length {}",
            k_global.nrows(),
            k_global.ncols(),
            n_dofs
        )));
    }
    let n_active = g.len();
    if n_active > 0 && c_matrix.ncols() != n_dofs {
        return Err(Error::Build(format!(
            "Signorini: C has {} cols but f has length {}",
            c_matrix.ncols(),
            n_dofs
        )));
    }
    if n_active > 0 && c_matrix.nrows() != n_active {
        return Err(Error::Build(format!(
            "Signorini: C has {} rows but g has length {}",
            c_matrix.nrows(),
            n_active
        )));
    }

    let mut lambda = vec![0.0; n_active];
    let mut u = cg_solve(
        |x| matvec(k_global, x),
        f,
        &vec![0.0; n_dofs],
        inner_cg_tol,
        inner_cg_max_iters,
    )
    .x;

    let mut primal_residual = Real::INFINITY;
    let mut dual_residual = 0.0;
    let mut outer_iters = 0;

    for it in 0..max_iters {
        outer_iters = it + 1;

        let cu = if n_active > 0 {
            matvec(c_matrix, &u)
        } else {
            Vec::new()
        };
        let mut residual = vec![0.0; n_active];
        let mut max_violation: Real = 0.0;
        for k in 0..n_active {
            residual[k] = g[k] - cu[k];
            let active_violation = if lambda[k] > 0.0 {
                residual[k].abs()
            } else {
                residual[k].max(0.0)
            };
            if active_violation > max_violation {
                max_violation = active_violation;
            }
        }
        primal_residual = max_violation;

        let mut max_dlam: Real = 0.0;
        let lambda_new: Vec<Real> = (0..n_active)
            .map(|k| {
                let raw = lambda[k] + rho * residual[k];
                let proj = raw.max(0.0);
                let dl = (proj - lambda[k]).abs();
                if dl > max_dlam {
                    max_dlam = dl;
                }
                proj
            })
            .collect();
        dual_residual = max_dlam;
        lambda = lambda_new;

        if primal_residual < tol && dual_residual < tol {
            break;
        }

        let mut rhs = f.to_vec();
        if n_active > 0 {
            let ct_lambda = ct_transpose_mul(c_matrix, &lambda, n_dofs);
            for i in 0..n_dofs {
                rhs[i] += ct_lambda[i];
            }
        }
        u = cg_solve(
            |x| matvec(k_global, x),
            &rhs,
            &u,
            inner_cg_tol,
            inner_cg_max_iters,
        )
        .x;
    }

    Ok(UzawaResult {
        u,
        lambda,
        iters: outer_iters,
        primal_residual,
        dual_residual,
    })
}

/// Solve the Signorini saddle-point system with Coulomb friction.
///
/// For each pair we maintain three multipliers `(λ_n, λ_tu, λ_tv)`.
/// The dual update is the same projected-gradient step as
/// frictionless Uzawa, but the projection now mixes the normal cap and
/// the friction cone:
///
/// - `λ_n ← max(0, λ_n + ρ · (g − n·u))`
/// - `(λ_tu, λ_tv) ← proj_disc_{μ · λ_n}(λ_tu + ρ · (−t_u·u),
///                                       λ_tv + ρ · (−t_v·u))`
///
/// Note the **tangential target is zero** — we want to drive
/// `t · u = 0` (stick); the disc cap then permits the slip residual to
/// grow up to the Coulomb limit.
///
/// The inner CG solves `K u = f + C_full^T λ` with `C_full` carrying
/// all three rows per pair.
///
/// Returns a [`FrictionUzawaResult`] with `lambda_normal` of length
/// `n_active` and `lambda_tangent` of length `2 · n_active`.
pub fn solve_augmented_system_with_friction(
    k_global: &BsrMatrix,
    contact: &SignoriniContact,
    f: &[Real],
    max_iters: usize,
    tol: Real,
) -> Result<FrictionUzawaResult> {
    let n_active = contact.n_active();
    let n_dofs = f.len();
    let c_full = assemble_full_constraint_matrix(contact.pairs(), n_dofs)?;
    let rho = estimate_dual_step(
        k_global,
        &c_full,
        3 * n_active,
        DEFAULT_INNER_CG_TOL,
        DEFAULT_INNER_CG_MAX_ITERS,
    );
    solve_augmented_system_with_friction_full(
        k_global,
        contact,
        &c_full,
        f,
        max_iters,
        tol,
        rho,
        DEFAULT_INNER_CG_TOL,
        DEFAULT_INNER_CG_MAX_ITERS,
    )
}

/// Same as [`solve_augmented_system_with_friction`] with the dual step,
/// the pre-assembled stacked `C_full`, and CG knobs exposed. Useful
/// when the caller wants to reuse one `C_full` build across multiple
/// solves (e.g. parameter sweeps over `μ`).
#[allow(clippy::too_many_arguments)]
pub fn solve_augmented_system_with_friction_full(
    k_global: &BsrMatrix,
    contact: &SignoriniContact,
    c_full: &BsrMatrix,
    f: &[Real],
    max_iters: usize,
    tol: Real,
    rho: Real,
    inner_cg_tol: Real,
    inner_cg_max_iters: usize,
) -> Result<FrictionUzawaResult> {
    let n_dofs = f.len();
    let n_active = contact.n_active();
    let mu_static = contact.friction_static_coeff.max(0.0);
    let mu_kinetic = contact.friction_kinetic_coeff.max(0.0);

    if k_global.nrows() != n_dofs || k_global.ncols() != n_dofs {
        return Err(Error::Build(format!(
            "Signorini-friction: K is {}×{} but f has length {}",
            k_global.nrows(),
            k_global.ncols(),
            n_dofs
        )));
    }
    if n_active > 0 && c_full.ncols() != n_dofs {
        return Err(Error::Build(format!(
            "Signorini-friction: C_full has {} cols but f has length {}",
            c_full.ncols(),
            n_dofs
        )));
    }
    if n_active > 0 && c_full.nrows() != 3 * n_active {
        return Err(Error::Build(format!(
            "Signorini-friction: C_full has {} rows but expected 3 × n_active = {}",
            c_full.nrows(),
            3 * n_active
        )));
    }

    // λ stacked as `[λ_n_0, λ_tu_0, λ_tv_0, λ_n_1, …]` to match the
    // row interleaving of `C_full`.
    let mut lambda_stacked = vec![0.0; 3 * n_active];
    let mut u = cg_solve(
        |x| matvec(k_global, x),
        f,
        &vec![0.0; n_dofs],
        inner_cg_tol,
        inner_cg_max_iters,
    )
    .x;

    let mut primal_residual = Real::INFINITY;
    let mut dual_residual = 0.0;
    let mut outer_iters = 0;
    let mut friction_state = vec![FrictionState::Inactive; n_active];

    for it in 0..max_iters {
        outer_iters = it + 1;

        // C_full · u — gives normal and tangential projections at every
        // pair, interleaved.
        let cu = if n_active > 0 {
            matvec(c_full, &u)
        } else {
            Vec::new()
        };

        // ── Compute primal residual on the normal rows only ─────────
        // (tangential rows have "target = 0"; their residual feeds the
        // dual update but is not part of the convergence metric — the
        // dual_residual sup-norm catches stagnation in λ_t).
        let mut max_violation: Real = 0.0;
        for k in 0..n_active {
            let row_n = 3 * k;
            let res_n = contact.active_pairs[k].target_gap - cu[row_n];
            let lam_n = lambda_stacked[row_n];
            let active_violation = if lam_n > 0.0 {
                res_n.abs()
            } else {
                res_n.max(0.0)
            };
            if active_violation > max_violation {
                max_violation = active_violation;
            }
        }
        primal_residual = max_violation;

        // ── Projected dual update (normal + Coulomb cone) ───────────
        let mut max_dlam: Real = 0.0;
        let mut lambda_new = vec![0.0; 3 * n_active];
        let mut state_new = vec![FrictionState::Inactive; n_active];
        for k in 0..n_active {
            let row_n = 3 * k;
            let row_tu = 3 * k + 1;
            let row_tv = 3 * k + 2;

            // Normal: gradient ascent toward residual_n = g − n·u, then
            // clip to λ_n ≥ 0.
            let target_g = contact.active_pairs[k].target_gap;
            let res_n = target_g - cu[row_n];
            let lam_n_raw = lambda_stacked[row_n] + rho * res_n;
            let lam_n_new = lam_n_raw.max(0.0);

            // Tangential: target is zero displacement (stick). We push
            // λ_t in the direction of `−t·u` so the inner solve sees a
            // tangential force that opposes the observed tangential
            // slide.
            let res_tu = -cu[row_tu];
            let res_tv = -cu[row_tv];
            let lam_tu_raw = lambda_stacked[row_tu] + rho * res_tu;
            let lam_tv_raw = lambda_stacked[row_tv] + rho * res_tv;

            // Project onto the static/kinetic Coulomb cone. Inside the static
            // cone we stick and keep the raw tangential multiplier; outside it
            // we slip and project to the kinetic boundary. If both
            // coefficients are equal this is exactly the Stage 50 v1
            // single-μ projection.
            let mu_static_eff = mu_static * lam_n_new;
            let mu_kinetic_eff = mu_kinetic * lam_n_new;
            let mag = (lam_tu_raw * lam_tu_raw + lam_tv_raw * lam_tv_raw).sqrt();
            let (lam_tu_new, lam_tv_new, state) = if lam_n_new <= 0.0 {
                (0.0, 0.0, FrictionState::Inactive)
            } else if mag <= mu_static_eff {
                // Inside cone → stick, no projection.
                (lam_tu_raw, lam_tv_raw, FrictionState::Stick)
            } else if mag > 0.0 {
                // Outside static cone → slip; radially rescale to the
                // kinetic-friction boundary.
                let s = mu_kinetic_eff / mag;
                (lam_tu_raw * s, lam_tv_raw * s, FrictionState::Slip)
            } else {
                // mag == 0 and λ_n > 0 — already inside trivially.
                (0.0, 0.0, FrictionState::Stick)
            };

            // Track sup-norm of dual increment across all three.
            let dl_n = (lam_n_new - lambda_stacked[row_n]).abs();
            let dl_tu = (lam_tu_new - lambda_stacked[row_tu]).abs();
            let dl_tv = (lam_tv_new - lambda_stacked[row_tv]).abs();
            let dl_max = dl_n.max(dl_tu).max(dl_tv);
            if dl_max > max_dlam {
                max_dlam = dl_max;
            }

            lambda_new[row_n] = lam_n_new;
            lambda_new[row_tu] = lam_tu_new;
            lambda_new[row_tv] = lam_tv_new;
            state_new[k] = state;
        }
        dual_residual = max_dlam;
        lambda_stacked = lambda_new;
        friction_state = state_new;

        // ── Convergence test ───────────────────────────────────────
        if primal_residual < tol && dual_residual < tol {
            break;
        }

        // ── Inner primal solve: K u = f + C_full^T λ ────────────────
        let mut rhs = f.to_vec();
        if n_active > 0 {
            let ct_lambda = ct_transpose_mul(c_full, &lambda_stacked, n_dofs);
            for i in 0..n_dofs {
                rhs[i] += ct_lambda[i];
            }
        }
        u = cg_solve(
            |x| matvec(k_global, x),
            &rhs,
            &u,
            inner_cg_tol,
            inner_cg_max_iters,
        )
        .x;
    }

    // De-interleave the stacked dual into normal / tangential views.
    let mut lambda_normal = vec![0.0; n_active];
    let mut lambda_tangent = vec![0.0; 2 * n_active];
    for k in 0..n_active {
        lambda_normal[k] = lambda_stacked[3 * k];
        lambda_tangent[2 * k] = lambda_stacked[3 * k + 1];
        lambda_tangent[2 * k + 1] = lambda_stacked[3 * k + 2];
    }

    Ok(FrictionUzawaResult {
        u,
        lambda_normal,
        lambda_tangent,
        friction_state,
        iters: outer_iters,
        primal_residual,
        dual_residual,
    })
}

/// Compute `y = C^T · x` where `C` is `n_active × n_dofs` (B = 1).
fn ct_transpose_mul(c: &BsrMatrix, x: &[Real], n_dofs: usize) -> Vec<Real> {
    debug_assert_eq!(c.b, 1, "ct_transpose_mul: only B = 1 supported");
    debug_assert_eq!(x.len(), c.nrows());
    let mut y = vec![0.0; n_dofs];
    for r in 0..c.block_rows {
        let lo = c.row_offsets[r];
        let hi = c.row_offsets[r + 1];
        let xr = x[r];
        for slot in lo..hi {
            let col = c.col_idx[slot];
            y[col] += c.values[slot] * xr;
        }
    }
    y
}

/// Estimate `ρ = 1 / λ_max(C K⁻¹ C^T)` via 6 power-iteration steps.
///
/// `n_active_rows` is the number of rows of `C` (i.e. `n_active` for
/// the Stage 49 normal-only path, or `3 · n_active` for the Stage 50
/// stacked path).
pub fn estimate_dual_step(
    k: &BsrMatrix,
    c: &BsrMatrix,
    n_active_rows: usize,
    inner_cg_tol: Real,
    inner_cg_max_iters: usize,
) -> Real {
    if n_active_rows == 0 {
        return 1.0;
    }
    let n_dofs = k.nrows();
    let mut y: Vec<Real> = (0..n_active_rows).map(|i| 1.0 + i as Real * 0.5).collect();
    let mut lam_max: Real = 1.0;
    for _ in 0..6 {
        let cty = ct_transpose_mul(c, &y, n_dofs);
        let kinv_cty = cg_solve(
            |x| matvec(k, x),
            &cty,
            &vec![0.0; n_dofs],
            inner_cg_tol,
            inner_cg_max_iters,
        )
        .x;
        let y_next = matvec(c, &kinv_cty);
        let norm: Real = y_next.iter().map(|v| v * v).sum::<Real>().sqrt();
        if norm <= 1.0e-30 {
            break;
        }
        let y_dot_y: Real = y.iter().map(|v| v * v).sum();
        let rq: Real = y
            .iter()
            .zip(y_next.iter())
            .map(|(a, b)| a * b)
            .sum::<Real>()
            / y_dot_y.max(1.0e-30);
        lam_max = rq.abs().max(1.0e-30);
        let inv = 1.0 / norm;
        for (a, b) in y.iter_mut().zip(y_next.iter()) {
            *a = b * inv;
        }
    }
    1.0 / lam_max
}

// ─── Stage 51 — Primal-Dual Active-Set (PDAS) Newton outer loop ──────
//
// Stage 49 (above) ships the *fixed-active-set* Uzawa solver. The
// caller must pre-decide which slave nodes are in contact. The v2
// surface below adds the **active-set update loop**: given a list of
// *candidate* slaves plus a gap function, automatically detect
// penetration, release zero-multiplier pairs that separate, and
// iterate the inner Uzawa solve until the active set stops changing.
//
// # Algorithm — PDAS Newton (a.k.a. semismooth Newton)
//
// The Signorini complementarity condition `λ ⊥ (g − Cu)` with `λ ≥ 0`
// and `g − Cu ≤ 0` is equivalent (via Fischer-Burmeister / min-NCP) to
// the fixed-point equation
//
// ```text
//     λ_k = max(0, λ_k + ρ (g_k − (Cu)_k))
// ```
//
// which we already linearise via the projected Uzawa inner solve. The
// outer PDAS Newton step linearises the `max` itself: at the current
// iterate, partition the candidates into
//
//   - **A** (active): `λ + ρ (g − Cu) > 0`  ⇒ enforce equality `Cu = g`
//   - **I** (inactive): `λ + ρ (g − Cu) ≤ 0` ⇒ drop the constraint
//
// then resolve with the new active set. The partition update is a
// Newton step on the complementarity equation; once two consecutive
// outer iterations produce the same partition the algorithm has
// converged (super-linearly, in theory).
//
// # Practical rules used here
//
// 1. **Add** any candidate with `gap(u) < −gap_tol` to the active set.
//    Slightly negative tolerance avoids cycling on grazing-contact
//    pairs.
// 2. **Release** any active pair whose multiplier clamped to zero in
//    the Uzawa inner solve AND whose current gap is positive (i.e.
//    truly separating). Pairs that hit `λ = 0` *and* `gap ≈ 0` stay
//    active — they're at the boundary of activity, and bouncing them
//    in/out cycles the active set without progress.
// 3. **Convergence** when the active set is identical to the previous
//    outer iteration AND the inner Uzawa solve converged.
//
// # Friction compatibility (deferred to Stage 50)
//
// Friction tangential constraints attach a *secondary* multiplier per
// active normal pair, bounded by `|λ_t| ≤ μ λ_n` (Coulomb). The active-
// set update logic here partitions only on normal complementarity, so
// adding friction means:
//
//   - At each active-set iteration, after solving for `(u, λ_n)`, run
//     an inner tangential active-set loop that decides per pair
//     whether the friction cone is *binding* (`|λ_t| = μ λ_n` — slip)
//     or *interior* (`|λ_t| < μ λ_n` — stick). The pair partition
//     (active / inactive normal) stays the same.
//   - The `release` rule still applies: a normal pair that hits
//     `λ_n = 0` automatically releases both its normal and friction
//     constraints (no friction without a normal force).
//
// The `gap_fn` closure here returns only the normal pair `(normal,
// gap)`; the Stage 50 friction extension will mirror it with a
// `tangent_fn` that returns the local tangential basis.

/// Candidate Signorini contact pairs that the active-set loop chooses
/// from.
///
/// Unlike [`SignoriniContact`] (which lists *active* pairs the caller
/// already decided on), this struct lists every slave DoF that *could*
/// come into contact, plus a closure that evaluates the gap function
/// `g(u)` at the current displacement.
///
/// The gap closure takes the global displacement field and a slave
/// base DoF index, and returns `(normal, gap)`:
///
/// - `normal`: unit outward master-surface normal at the slave's
///   current contact point.
/// - `gap`: signed distance along the normal. `gap < 0` means the
///   slave has penetrated the master and the pair should be added to
///   the active set.
pub struct SignoriniContactCandidates {
    /// Candidate slave DoFs (base index — same convention as
    /// [`SignoriniContact::active_pairs`]: `vertex * dim`).
    pub slave_dofs: Vec<u32>,
    /// Master gap function. Given `(slave_dof, &u)`, returns
    /// `(outward_normal, signed_gap)`. The active-set loop calls this
    /// once per candidate per outer iteration.
    pub master_gap_fn: Box<dyn Fn(u32, &[Real]) -> (Vec3, Real)>,
}

impl std::fmt::Debug for SignoriniContactCandidates {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SignoriniContactCandidates")
            .field("slave_dofs", &self.slave_dofs)
            .field("master_gap_fn", &"<closure>")
            .finish()
    }
}

impl SignoriniContactCandidates {
    /// Build a candidate set from a slave list and a gap closure.
    pub fn new(
        slave_dofs: Vec<u32>,
        master_gap_fn: impl Fn(u32, &[Real]) -> (Vec3, Real) + 'static,
    ) -> Self {
        Self {
            slave_dofs,
            master_gap_fn: Box::new(master_gap_fn),
        }
    }

    /// Convenience constructor: every candidate sees the same fixed
    /// master plane defined by `(point, normal)`. The gap is
    /// `n · (x + u_slave − point)` — i.e. the signed distance from
    /// the slave's *current* position (rest + displacement) to the
    /// plane along the outward normal.
    ///
    /// `rest_positions` carries the per-vertex rest coordinates; only
    /// `rest_positions[slave_dof / dim]` is read per evaluation.
    pub fn from_plane(
        slave_dofs: Vec<u32>,
        rest_positions: Vec<Vec3>,
        dim: usize,
        plane_point: Vec3,
        plane_normal: Vec3,
    ) -> Self {
        let n = plane_normal;
        let p = plane_point;
        Self::new(slave_dofs, move |slave_dof, u| {
            let v = slave_dof as usize / dim;
            let rest = rest_positions[v];
            let ux = u[slave_dof as usize];
            let uy = u[slave_dof as usize + 1];
            let uz = u[slave_dof as usize + 2];
            let current = rest + Vec3::new(ux, uy, uz);
            let delta = current - p;
            let gap = n.x * delta.x + n.y * delta.y + n.z * delta.z;
            (n, gap)
        })
    }
}

/// Convert vertex-triangle self-contact candidates into Signorini
/// active-set candidates.
///
/// The Signorini candidate interface carries one gap function per
/// slave DoF, so multiple self-contact candidates touching the same
/// vertex are reduced to the deepest shell penetration for that
/// vertex. The master plane is the closest triangle point shifted by
/// `normal * thickness`, which makes the gap equal to
/// `distance - thickness` at zero displacement.
#[must_use]
pub fn signorini_candidates_from_self_contacts(
    contacts: &[SelfContactCandidate],
    rest_positions: Vec<Vec3>,
    dim: usize,
    thickness: Real,
) -> SignoriniContactCandidates {
    let mut by_slave: HashMap<u32, (Vec3, Vec3, Real)> = HashMap::new();
    for c in contacts {
        let slave = c.vertex * dim as u32;
        let plane_point = c.point + c.normal * thickness;
        let entry = by_slave
            .entry(slave)
            .or_insert((plane_point, c.normal, c.gap));
        if c.gap < entry.2 {
            *entry = (plane_point, c.normal, c.gap);
        }
    }
    let mut slave_dofs: Vec<u32> = by_slave.keys().copied().collect();
    slave_dofs.sort_unstable();
    SignoriniContactCandidates::new(slave_dofs, move |slave_dof, u| {
        let v = slave_dof as usize / dim;
        let rest = rest_positions[v];
        let ux = u[slave_dof as usize];
        let uy = u[slave_dof as usize + 1];
        let uz = u[slave_dof as usize + 2];
        let current = rest + Vec3::new(ux, uy, uz);
        let (plane_point, normal, _) = by_slave[&slave_dof];
        let gap = normal.dot(current - plane_point);
        (normal, gap)
    })
}

/// Per-outer-iteration record from [`solve_active_set_signorini`].
///
/// One entry per outer PDAS Newton step; the last entry holds the
/// converged state.
#[derive(Clone, Debug)]
pub struct ActiveSetIteration {
    /// 0-based outer iteration index.
    pub iter: usize,
    /// Slave DoFs in the active set going *into* the inner solve.
    pub active_slaves: Vec<u32>,
    /// Slave DoFs newly added this iteration (penetrating ⇒ added).
    pub added: Vec<u32>,
    /// Slave DoFs released this iteration (separating ⇒ removed).
    pub released: Vec<u32>,
    /// Outer Uzawa iterations taken by the inner solve at this step.
    pub inner_iters: usize,
    /// Inner Uzawa primal residual at this step.
    pub primal_residual: Real,
    /// Inner Uzawa dual residual at this step.
    pub dual_residual: Real,
    /// Whether this iteration proposed an active set that had already
    /// appeared in the trace and therefore used the least-index
    /// anti-cycling pivot.
    pub cycle_detected: bool,
    /// Slave DoF selected by the least-index anti-cycling rule when
    /// [`Self::cycle_detected`] is true.
    pub bland_pivot: Option<u32>,
}

/// Diagnostic trace of the PDAS Newton outer loop.
#[derive(Clone, Debug, Default)]
pub struct ActiveSetTrace {
    /// One entry per outer iteration, in order.
    pub iterations: Vec<ActiveSetIteration>,
    /// Whether the outer loop converged (final active set stable AND
    /// inner solve hit its tolerance).
    pub converged: bool,
    /// Total number of outer iterations performed.
    pub outer_iters: usize,
    /// Number of times the least-index anti-cycling rule was applied.
    pub cycle_breaks: usize,
}

impl ActiveSetTrace {
    /// Final active slave list, or empty if no iterations ran.
    #[must_use]
    pub fn final_active(&self) -> Vec<u32> {
        self.iterations
            .last()
            .map(|it| it.active_slaves.clone())
            .unwrap_or_default()
    }
}

/// Output of [`solve_active_set_signorini`].
///
/// `lambda` is the *full-length* multiplier vector aligned with
/// `candidates.slave_dofs` — inactive slots are 0. This makes it easy
/// to read the per-candidate force without tracking the active-set
/// permutation.
#[derive(Clone, Debug)]
pub struct ActiveSetResult {
    /// Final displacement field.
    pub u: Vec<Real>,
    /// Final per-candidate dual multipliers. Length matches
    /// `candidates.slave_dofs.len()`; entries for inactive candidates
    /// are exactly 0.
    pub lambda: Vec<Real>,
    /// Per-iteration diagnostic record.
    pub trace: ActiveSetTrace,
}

/// Output of [`solve_active_set_signorini_with_friction`].
///
/// Normal multipliers are full-length and aligned with
/// `candidates.slave_dofs`; tangent multipliers are interleaved
/// `[tu_i, tv_i]` with zero entries for inactive candidates.
#[derive(Clone, Debug)]
pub struct ActiveSetFrictionResult {
    /// Final displacement field.
    pub u: Vec<Real>,
    /// Final per-candidate normal multipliers.
    pub lambda_normal: Vec<Real>,
    /// Final per-candidate tangent multipliers, two entries per
    /// candidate.
    pub lambda_tangent: Vec<Real>,
    /// Final per-candidate stick/slip state.
    pub friction_state: Vec<FrictionState>,
    /// Per-iteration diagnostic record.
    pub trace: ActiveSetTrace,
}

/// Maximum outer PDAS Newton iterations before bailing out.
pub const DEFAULT_ACTIVE_SET_MAX_OUTER: usize = 20;
/// Gap tolerance for the "candidate is penetrating" test. Slightly
/// negative — i.e. only nodes with `gap < -gap_tol` are added — to
/// avoid cycling on grazing contacts.
pub const DEFAULT_ACTIVE_SET_GAP_TOL: Real = 1.0e-10;
/// Inner Uzawa max iterations per outer step.
pub const DEFAULT_ACTIVE_SET_INNER_MAX: usize = 500;
/// Inner Uzawa convergence tolerance (primal + dual residual).
pub const DEFAULT_ACTIVE_SET_INNER_TOL: Real = 1.0e-10;

/// Solve the Signorini problem with **automatic active-set update**.
///
/// Given a candidate slave set and a gap function, run PDAS Newton:
///
/// 1. Start from the empty active set.
/// 2. Run the inner Uzawa solve (Stage 49).
/// 3. Update the active set:
///    - **Add** candidates with `gap(u) < −gap_tol`.
///    - **Release** active pairs that converged to `λ = 0` *and* have
///      `gap(u) > gap_tol` (truly separating).
/// 4. If the active set is unchanged AND the inner solve converged,
///    return; else loop.
///
/// The dual step `ρ` is re-estimated each outer iteration because the
/// constraint matrix `C` changes when the active set changes.
///
/// The returned `lambda` is *per-candidate*: index `i` of the output
/// vector corresponds to `candidates.slave_dofs[i]`. Inactive slots
/// hold 0.
pub fn solve_active_set_signorini(
    k_global: &BsrMatrix,
    candidates: &SignoriniContactCandidates,
    f: &[Real],
    max_outer_iters: usize,
    gap_tol: Real,
    inner_max_iters: usize,
    inner_tol: Real,
) -> Result<ActiveSetResult> {
    let n_dofs = f.len();
    let n_cand = candidates.slave_dofs.len();

    if k_global.nrows() != n_dofs || k_global.ncols() != n_dofs {
        return Err(Error::Build(format!(
            "Active-set Signorini: K is {}×{} but f has length {}",
            k_global.nrows(),
            k_global.ncols(),
            n_dofs
        )));
    }

    // Warm-start `u` with the unconstrained solve.
    let mut u = cg_solve(
        |x| matvec(k_global, x),
        f,
        &vec![0.0; n_dofs],
        DEFAULT_INNER_CG_TOL,
        DEFAULT_INNER_CG_MAX_ITERS,
    )
    .x;

    // Active-set membership tracked as a Vec<bool> aligned with
    // `candidates.slave_dofs`. We carry per-candidate λ across outer
    // iterations as a warm start.
    let mut active = vec![false; n_cand];
    let mut lambda_full = vec![0.0; n_cand];
    let mut trace = ActiveSetTrace::default();
    let mut prev_active_valid = false;
    let mut seen_active_sets: HashSet<Vec<bool>> = HashSet::new();
    seen_active_sets.insert(active.clone());

    for outer in 0..max_outer_iters {
        // ── Step 1: evaluate gaps at current displacement ────────────
        let mut gaps = vec![0.0; n_cand];
        let mut normals = vec![Vec3::new(0.0, 0.0, 0.0); n_cand];
        for i in 0..n_cand {
            let (n, gap) = (candidates.master_gap_fn)(candidates.slave_dofs[i], &u);
            normals[i] = n;
            gaps[i] = gap;
        }

        // ── Step 2: update active set ────────────────────────────────
        // Add: candidates with gap < −gap_tol that aren't active yet.
        // Release: active pairs with λ ≈ 0 AND gap > +gap_tol (truly
        // separating). Pairs that converged to λ = 0 but still have
        // gap ≈ 0 are at the activity boundary and stay active to
        // avoid chattering.
        let old_active = active.clone();
        let mut added: Vec<u32> = Vec::new();
        let mut released: Vec<u32> = Vec::new();
        for i in 0..n_cand {
            let was_active = active[i];
            let penetrating = gaps[i] < -gap_tol;
            let separating: bool = (lambda_full[i] as Real).abs() < 1.0e-12 && gaps[i] > gap_tol;

            if !was_active && penetrating {
                active[i] = true;
                added.push(candidates.slave_dofs[i]);
            } else if was_active && separating {
                active[i] = false;
                lambda_full[i] = 0.0;
                released.push(candidates.slave_dofs[i]);
            }
        }
        added.sort_unstable();
        released.sort_unstable();

        // Stage 51 v2 — anti-cycling. If the full add/release update
        // would revisit an active set already seen in this PDAS trace,
        // replace it with a single least-slave-DoF pivot that leads to
        // an unseen set when possible. This is the active-set analogue
        // of Bland's least-index rule: deterministic, monotone in the
        // candidate ordering, and it breaks add/drop oscillations in
        // degenerate grazing contact configurations.
        let mut cycle_detected = false;
        let mut bland_pivot = None;
        if active != old_active
            && seen_active_sets.contains(&active)
            && let Some((pivoted, pivot_idx)) = bland_cycle_break(
                &old_active,
                &active,
                &seen_active_sets,
                &candidates.slave_dofs,
            )
        {
            active = pivoted;
            cycle_detected = true;
            bland_pivot = Some(candidates.slave_dofs[pivot_idx]);
            trace.cycle_breaks += 1;
            recompute_active_delta(
                &old_active,
                &active,
                &candidates.slave_dofs,
                &mut added,
                &mut released,
            );
        }

        // Snapshot the active list for the trace.
        let active_slaves: Vec<u32> = (0..n_cand)
            .filter(|&i| active[i])
            .map(|i| candidates.slave_dofs[i])
            .collect();

        // ── Step 3: detect convergence ──────────────────────────────
        // Stable if no adds/releases AND we had a previous iteration
        // (so we don't bail on outer == 0 with an empty set when the
        // unconstrained solution is genuinely admissible).
        let set_stable = prev_active_valid && added.is_empty() && released.is_empty();

        // For the no-contact case (set_stable at outer 0 with empty
        // active set) we still need to record the iteration with the
        // unconstrained solve as the converged state.
        if set_stable {
            trace.iterations.push(ActiveSetIteration {
                iter: outer,
                active_slaves,
                added,
                released,
                inner_iters: 0,
                primal_residual: 0.0,
                dual_residual: 0.0,
                cycle_detected,
                bland_pivot,
            });
            trace.converged = true;
            trace.outer_iters = outer + 1;
            break;
        }

        // ── Step 4: assemble C, g for the current active set ────────
        let pairs: Vec<(u32, Vec3, Real)> = (0..n_cand)
            .filter(|&i| active[i])
            .map(|i| {
                // Target gap: enforce `n · u_slave ≥ g_target` such
                // that after the solve, `current_position · n ≥
                // plane_point · n`. Using the gap closure's current
                // reading: `gap = n · (rest + u − plane_point)`. To
                // restore non-penetration we want `gap_new ≥ 0`. The
                // constraint is `n · u_slave ≥ g_target` where the
                // closure already folded the rest position in. We
                // solve for the delta needed: at the *current* u,
                // `n · u = gap − n · (rest − plane_point)`. Tracking
                // `n · u_target = n · u − gap` gives the displacement
                // along the normal that lifts the slave back to the
                // plane. That's the target.
                let slave_dof = candidates.slave_dofs[i];
                let ux = u[slave_dof as usize];
                let uy = u[slave_dof as usize + 1];
                let uz = u[slave_dof as usize + 2];
                let n = normals[i];
                let n_dot_u = n.x * ux + n.y * uy + n.z * uz;
                let target = n_dot_u - gaps[i];
                (slave_dof, n, target)
            })
            .collect();

        let n_active = pairs.len();
        let c = assemble_constraint_matrix(&pairs, n_dofs)?;
        let g = target_gap_vector(&pairs);

        // ── Step 5: warm-start λ for the active pairs ───────────────
        // Map per-candidate λ → per-active-pair λ.
        let warm_lambda: Vec<Real> = (0..n_cand)
            .filter(|&i| active[i])
            .map(|i| lambda_full[i])
            .collect();

        // ── Step 6: inner Uzawa solve ───────────────────────────────
        let inner = if n_active == 0 {
            // No constraints — unconstrained CG solve from current u.
            let u_free = cg_solve(
                |x| matvec(k_global, x),
                f,
                &u,
                DEFAULT_INNER_CG_TOL,
                DEFAULT_INNER_CG_MAX_ITERS,
            )
            .x;
            u = u_free;
            UzawaResult {
                u: u.clone(),
                lambda: Vec::new(),
                iters: 0,
                primal_residual: 0.0,
                dual_residual: 0.0,
            }
        } else {
            let rho = estimate_dual_step(
                k_global,
                &c,
                n_active,
                DEFAULT_INNER_CG_TOL,
                DEFAULT_INNER_CG_MAX_ITERS,
            );
            let mut res = solve_augmented_system_with(
                k_global,
                &c,
                f,
                &g,
                inner_max_iters,
                inner_tol,
                rho,
                DEFAULT_INNER_CG_TOL,
                DEFAULT_INNER_CG_MAX_ITERS,
            )?;
            // Overwrite with the warm-start λ once we know
            // `solve_augmented_system_with` starts from zero — for
            // future tightening we could thread the warm start
            // through. For now we just propagate the *converged* λ
            // back into `lambda_full`.
            //
            // (The implementation in Stage 49 starts λ at zero each
            // call; the warm start is essentially "guidance" for the
            // active-set decision, not the inner iterate. This still
            // converges in ≤ 5 outer iters for the test cases.)
            let _ = &warm_lambda;
            u = std::mem::take(&mut res.u);
            res
        };

        // ── Step 7: scatter λ back into the full-length vector ──────
        // First zero out inactive slots (they may have flipped this
        // step), then write the freshly-solved active values.
        for i in 0..n_cand {
            if !active[i] {
                lambda_full[i] = 0.0;
            }
        }
        if n_active > 0 {
            let mut k_active = 0usize;
            for i in 0..n_cand {
                if active[i] {
                    lambda_full[i] = inner.lambda[k_active];
                    k_active += 1;
                }
            }
        }

        trace.iterations.push(ActiveSetIteration {
            iter: outer,
            active_slaves,
            added,
            released,
            inner_iters: inner.iters,
            primal_residual: inner.primal_residual,
            dual_residual: inner.dual_residual,
            cycle_detected,
            bland_pivot,
        });

        prev_active_valid = true;
        seen_active_sets.insert(active.clone());
        trace.outer_iters = outer + 1;
    }

    Ok(ActiveSetResult {
        u,
        lambda: lambda_full,
        trace,
    })
}

/// Solve hard Signorini contact and Coulomb friction in one active-set
/// outer loop.
///
/// This is the Stage 49/50 integration path: the outer PDAS loop still
/// discovers which candidate slaves are actually in contact, but every
/// inner solve uses the stacked normal+tangent Coulomb system from
/// [`solve_augmented_system_with_friction_full`]. Releasing a normal
/// contact also releases its tangential multipliers and stick/slip
/// state, so there is no separate "friction after contact" pass.
#[allow(clippy::too_many_arguments)]
pub fn solve_active_set_signorini_with_friction(
    k_global: &BsrMatrix,
    candidates: &SignoriniContactCandidates,
    f: &[Real],
    friction_static_coeff: Real,
    friction_kinetic_coeff: Real,
    max_outer_iters: usize,
    gap_tol: Real,
    inner_max_iters: usize,
    inner_tol: Real,
) -> Result<ActiveSetFrictionResult> {
    let n_dofs = f.len();
    let n_cand = candidates.slave_dofs.len();

    if k_global.nrows() != n_dofs || k_global.ncols() != n_dofs {
        return Err(Error::Build(format!(
            "Active-set friction Signorini: K is {}×{} but f has length {}",
            k_global.nrows(),
            k_global.ncols(),
            n_dofs
        )));
    }

    let mut u = cg_solve(
        |x| matvec(k_global, x),
        f,
        &vec![0.0; n_dofs],
        DEFAULT_INNER_CG_TOL,
        DEFAULT_INNER_CG_MAX_ITERS,
    )
    .x;

    let mut active = vec![false; n_cand];
    let mut lambda_n_full = vec![0.0 as Real; n_cand];
    let mut lambda_t_full = vec![0.0 as Real; 2 * n_cand];
    let mut state_full = vec![FrictionState::Inactive; n_cand];
    let mut trace = ActiveSetTrace::default();
    let mut prev_active_valid = false;
    let mut seen_active_sets: HashSet<Vec<bool>> = HashSet::new();
    seen_active_sets.insert(active.clone());

    for outer in 0..max_outer_iters {
        let mut gaps = vec![0.0; n_cand];
        let mut normals = vec![Vec3::new(0.0, 0.0, 0.0); n_cand];
        for i in 0..n_cand {
            let (n, gap) = (candidates.master_gap_fn)(candidates.slave_dofs[i], &u);
            normals[i] = n;
            gaps[i] = gap;
        }

        let old_active = active.clone();
        let mut added: Vec<u32> = Vec::new();
        let mut released: Vec<u32> = Vec::new();
        for i in 0..n_cand {
            let was_active = active[i];
            let penetrating = gaps[i] < -gap_tol;
            let separating = lambda_n_full[i].abs() < 1.0e-12 && gaps[i] > gap_tol;
            if !was_active && penetrating {
                active[i] = true;
                added.push(candidates.slave_dofs[i]);
            } else if was_active && separating {
                active[i] = false;
                lambda_n_full[i] = 0.0;
                lambda_t_full[2 * i] = 0.0;
                lambda_t_full[2 * i + 1] = 0.0;
                state_full[i] = FrictionState::Inactive;
                released.push(candidates.slave_dofs[i]);
            }
        }
        added.sort_unstable();
        released.sort_unstable();

        let mut cycle_detected = false;
        let mut bland_pivot = None;
        if active != old_active
            && seen_active_sets.contains(&active)
            && let Some((pivoted, pivot_idx)) = bland_cycle_break(
                &old_active,
                &active,
                &seen_active_sets,
                &candidates.slave_dofs,
            )
        {
            active = pivoted;
            cycle_detected = true;
            bland_pivot = Some(candidates.slave_dofs[pivot_idx]);
            trace.cycle_breaks += 1;
            recompute_active_delta(
                &old_active,
                &active,
                &candidates.slave_dofs,
                &mut added,
                &mut released,
            );
        }

        let active_slaves: Vec<u32> = (0..n_cand)
            .filter(|&i| active[i])
            .map(|i| candidates.slave_dofs[i])
            .collect();
        let set_stable = prev_active_valid && added.is_empty() && released.is_empty();
        if set_stable {
            trace.iterations.push(ActiveSetIteration {
                iter: outer,
                active_slaves,
                added,
                released,
                inner_iters: 0,
                primal_residual: 0.0,
                dual_residual: 0.0,
                cycle_detected,
                bland_pivot,
            });
            trace.converged = true;
            trace.outer_iters = outer + 1;
            break;
        }

        let mut contact = SignoriniContact::with_static_kinetic_friction(
            friction_static_coeff,
            friction_kinetic_coeff,
        );
        let mut active_indices: Vec<usize> = Vec::new();
        for i in 0..n_cand {
            if !active[i] {
                continue;
            }
            let slave_dof = candidates.slave_dofs[i];
            let ux = u[slave_dof as usize];
            let uy = u[slave_dof as usize + 1];
            let uz = u[slave_dof as usize + 2];
            let n = normals[i];
            let n_dot_u = n.x * ux + n.y * uy + n.z * uz;
            let target = n_dot_u - gaps[i];
            contact
                .active_pairs
                .push(ContactPair::from_normal(slave_dof, n, target));
            active_indices.push(i);
        }

        let inner = if active_indices.is_empty() {
            u = cg_solve(
                |x| matvec(k_global, x),
                f,
                &u,
                DEFAULT_INNER_CG_TOL,
                DEFAULT_INNER_CG_MAX_ITERS,
            )
            .x;
            FrictionUzawaResult {
                u: u.clone(),
                lambda_normal: Vec::new(),
                lambda_tangent: Vec::new(),
                friction_state: Vec::new(),
                iters: 0,
                primal_residual: 0.0,
                dual_residual: 0.0,
            }
        } else {
            let c_full = assemble_full_constraint_matrix(contact.pairs(), n_dofs)?;
            let rho = estimate_dual_step(
                k_global,
                &c_full,
                3 * active_indices.len(),
                DEFAULT_INNER_CG_TOL,
                DEFAULT_INNER_CG_MAX_ITERS,
            );
            let mut res = solve_augmented_system_with_friction_full(
                k_global,
                &contact,
                &c_full,
                f,
                inner_max_iters,
                inner_tol,
                rho,
                DEFAULT_INNER_CG_TOL,
                DEFAULT_INNER_CG_MAX_ITERS,
            )?;
            u = std::mem::take(&mut res.u);
            res
        };

        for i in 0..n_cand {
            if !active[i] {
                lambda_n_full[i] = 0.0;
                lambda_t_full[2 * i] = 0.0;
                lambda_t_full[2 * i + 1] = 0.0;
                state_full[i] = FrictionState::Inactive;
            }
        }
        for (local, &cand_idx) in active_indices.iter().enumerate() {
            lambda_n_full[cand_idx] = inner.lambda_normal[local];
            lambda_t_full[2 * cand_idx] = inner.lambda_tangent[2 * local];
            lambda_t_full[2 * cand_idx + 1] = inner.lambda_tangent[2 * local + 1];
            state_full[cand_idx] = inner.friction_state[local];
        }

        trace.iterations.push(ActiveSetIteration {
            iter: outer,
            active_slaves,
            added,
            released,
            inner_iters: inner.iters,
            primal_residual: inner.primal_residual,
            dual_residual: inner.dual_residual,
            cycle_detected,
            bland_pivot,
        });

        prev_active_valid = true;
        seen_active_sets.insert(active.clone());
        trace.outer_iters = outer + 1;
    }

    Ok(ActiveSetFrictionResult {
        u,
        lambda_normal: lambda_n_full,
        lambda_tangent: lambda_t_full,
        friction_state: state_full,
        trace,
    })
}

fn recompute_active_delta(
    old_active: &[bool],
    new_active: &[bool],
    slave_dofs: &[u32],
    added: &mut Vec<u32>,
    released: &mut Vec<u32>,
) {
    added.clear();
    released.clear();
    for i in 0..new_active.len() {
        match (old_active[i], new_active[i]) {
            (false, true) => added.push(slave_dofs[i]),
            (true, false) => released.push(slave_dofs[i]),
            _ => {}
        }
    }
    added.sort_unstable();
    released.sort_unstable();
}

fn bland_cycle_break(
    old_active: &[bool],
    proposed_active: &[bool],
    seen_active_sets: &HashSet<Vec<bool>>,
    slave_dofs: &[u32],
) -> Option<(Vec<bool>, usize)> {
    let mut changes: Vec<usize> = (0..proposed_active.len())
        .filter(|&i| old_active[i] != proposed_active[i])
        .collect();
    changes.sort_by_key(|&i| slave_dofs[i]);
    for &idx in &changes {
        let mut pivoted = old_active.to_vec();
        pivoted[idx] = proposed_active[idx];
        if !seen_active_sets.contains(&pivoted) {
            return Some((pivoted, idx));
        }
    }
    // Degenerate fallback: every one-pivot state has also been seen.
    // Still choose the least-index pivot for deterministic progress
    // rather than accepting the full repeated set.
    changes.first().map(|&idx| {
        let mut pivoted = old_active.to_vec();
        pivoted[idx] = proposed_active[idx];
        (pivoted, idx)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1-DoF toy: K = [2.0], f = [-3.0] (would push u to -1.5),
    /// constraint u ≥ 0. Expect u = 0, λ = 3.0.
    #[test]
    fn uzawa_one_dof_clamps_to_constraint() {
        let k = BsrMatrix::from_scalar_triplets(3, 3, &[(0, 0, 2.0), (1, 1, 2.0), (2, 2, 2.0)])
            .unwrap();
        let f = vec![0.0, 0.0, -3.0]; // push z-component negative.
        let pairs = vec![(0u32, Vec3::new(0.0, 0.0, 1.0), 0.0)];
        let c = assemble_constraint_matrix(&pairs, 3).unwrap();
        let g = target_gap_vector(&pairs);
        let res = solve_augmented_system(&k, &c, &f, &g, 200, 1.0e-12).unwrap();
        assert!(
            res.u[2].abs() < 1.0e-8,
            "u_z should clamp to 0, got {}",
            res.u[2]
        );
        assert!(
            (res.lambda[0] - 3.0).abs() < 1.0e-6,
            "λ should converge to 3.0, got {}",
            res.lambda[0]
        );
        assert!(res.lambda[0] >= -1.0e-9, "λ must be ≥ 0");
    }

    /// Tangent basis sanity: orthonormal and right-handed.
    #[test]
    fn tangent_basis_is_orthonormal() {
        for n in [
            Vec3::new(0.0, 0.0, 1.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(1.0, 1.0, 1.0).normalize(),
            Vec3::new(0.3, -0.4, 0.866).normalize(),
        ] {
            let (tu, tv) = orthonormal_tangents(n);
            assert!(
                (tu.length() - 1.0).abs() < 1.0e-10,
                "|t_u| != 1 for n={n:?}"
            );
            assert!(
                (tv.length() - 1.0).abs() < 1.0e-10,
                "|t_v| != 1 for n={n:?}"
            );
            assert!(tu.dot(n).abs() < 1.0e-10, "t_u not ⟂ n for n={n:?}");
            assert!(tv.dot(n).abs() < 1.0e-10, "t_v not ⟂ n for n={n:?}");
            assert!(tu.dot(tv).abs() < 1.0e-10, "t_u not ⟂ t_v for n={n:?}");
            // Right-handed: n × t_u = t_v.
            let nt = n.cross(tu);
            assert!(
                (nt - tv).length() < 1.0e-10,
                "basis not right-handed for n={n:?}"
            );
        }
    }

    #[test]
    fn static_kinetic_constructor_preserves_legacy_mu() {
        let legacy = SignoriniContact::with_friction(0.4);
        assert_eq!(legacy.friction_coeff, 0.4);
        assert_eq!(legacy.friction_static_coeff, 0.4);
        assert_eq!(legacy.friction_kinetic_coeff, 0.4);

        let split = SignoriniContact::with_static_kinetic_friction(0.6, 0.2);
        assert_eq!(split.friction_coeff, 0.6);
        assert_eq!(split.friction_static_coeff, 0.6);
        assert_eq!(split.friction_kinetic_coeff, 0.2);
    }

    #[test]
    fn bland_cycle_break_chooses_least_unseen_slave_dof() {
        let old = vec![true, false, false];
        let proposed = vec![false, true, true];
        let slave_dofs = vec![30, 10, 20];
        let mut seen = HashSet::new();
        seen.insert(old.clone());
        // The least slave DoF change is index 1 (DoF 10). Since that
        // one-pivot set is unseen, Bland-style anti-cycling should pick it.
        let (pivoted, idx) = bland_cycle_break(&old, &proposed, &seen, &slave_dofs).expect("pivot");
        assert_eq!(idx, 1);
        assert_eq!(pivoted, vec![true, true, false]);

        let mut added = Vec::new();
        let mut released = Vec::new();
        recompute_active_delta(&old, &pivoted, &slave_dofs, &mut added, &mut released);
        assert_eq!(added, vec![10]);
        assert!(released.is_empty());
    }
}
