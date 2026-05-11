//! Stage 49 — Signorini hard non-penetration contact via Lagrange
//! multipliers (fixed active set).
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
//! # Algorithm — Uzawa iteration
//!
//! Uzawa is the textbook dual-ascent / inner-primal-solve scheme:
//!
//! 1. **Inner primal** — fix `λ`, solve `K u = f + C^T λ` via CG.
//! 2. **Dual update** — gradient ascent on the (concave) dual
//!    functional with projection onto `λ ≥ 0`:
//!    `λ ← max(0, λ + ρ · (g − C u))`.
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
//! # Scope — fixed active set
//!
//! This stage solves the **augmented system for a fixed list of active
//! contact pairs**. The caller is responsible for deciding which pairs
//! are active (typically: any slave node currently penetrating). The
//! v2 follow-up adds the **active-set update loop**: detect new
//! penetrations after each Uzawa solve, drop pairs whose `λ` clamps to
//! zero (separating), and re-solve.
//!
//! # Reference
//!
//! - Wriggers, *Computational Contact Mechanics*, §6.3 (Uzawa-Lagrange).
//! - Newton's `signorini.py` and Warp's `xpbd_contact.py` for the active-
//!   set logic that comes in v2.

use crate::core::{Error, Result};
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

/// Fixed list of Signorini contact pairs to enforce.
///
/// Each entry is `(slave_dof_global, master_normal, target_gap)`:
/// - `slave_dof_global`: BASE global DoF index of the slave vertex
///   (i.e. `vertex_index * dim`). The constraint operator writes the
///   normal components into the three consecutive DoFs starting here.
/// - `master_normal`: unit outward normal of the master plane at the
///   contact point. The constraint enforces `n · u_slave ≥ g`.
/// - `target_gap`: minimum allowed projected displacement. If the
///   initial geometry has a penetration `−p < 0`, set
///   `target_gap = p` so that `n · u_slave ≥ p` restores contact at
///   zero gap. For nodes that are merely *on* the master surface,
///   `target_gap = 0`.
///
/// The active-set update logic (deciding *which* pairs end up here) is
/// out of scope for Stage 49 — the caller assembles `active_pairs`
/// from their own contact-detection pass.
#[derive(Clone, Debug)]
pub struct SignoriniContact {
    /// `(slave_dof_global, master_normal, target_gap)` triplets.
    pub active_pairs: Vec<(u32, Vec3, Real)>,
}

impl SignoriniContact {
    /// Build an empty contact set.
    #[must_use]
    pub fn new() -> Self {
        Self {
            active_pairs: Vec::new(),
        }
    }

    /// Number of active constraint rows.
    #[must_use]
    pub fn n_active(&self) -> usize {
        self.active_pairs.len()
    }

    /// Borrow the active pairs as a slice.
    #[must_use]
    pub fn pairs(&self) -> &[(u32, Vec3, Real)] {
        &self.active_pairs
    }
}

impl Default for SignoriniContact {
    fn default() -> Self {
        Self::new()
    }
}

/// Assemble the sparse constraint matrix `C` of shape `n_active × n_dofs`.
///
/// Row `k` corresponds to active pair `k`: three entries at columns
/// `(slave_dof, slave_dof+1, slave_dof+2)` equal to
/// `(n.x, n.y, n.z)`. All other entries are zero.
///
/// The matrix is built with [`BsrMatrix::from_scalar_triplets`]
/// (`B = 1`, scalar CSR-style storage), which sorts and dedupes the
/// `(row, col)` pattern. Multiple pairs that share the same slave node
/// land in distinct rows, so duplicates within a row only occur if the
/// caller put two entries pointing at the same column inside one pair
/// (impossible for the `(slave, normal)` writer below).
///
/// Returns [`Error::Build`] if any `slave_dof + 2 >= n_dofs` or if the
/// underlying BSR builder rejects the pattern.
pub fn assemble_constraint_matrix(
    active_pairs: &[(u32, Vec3, Real)],
    n_dofs: usize,
) -> Result<BsrMatrix> {
    let n_active = active_pairs.len();
    if n_active == 0 {
        // Empty constraint matrix: 0 rows × n_dofs cols. Still has to
        // satisfy `block_rows >= 1` for the from_pattern path — use a
        // dummy single-row matrix with no entries. The downstream
        // Uzawa loop skips the dual update entirely in this case.
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

/// Extract the `g` (target-gap) vector from the active pairs.
#[must_use]
pub fn target_gap_vector(active_pairs: &[(u32, Vec3, Real)]) -> Vec<Real> {
    active_pairs.iter().map(|(_, _, g)| *g).collect()
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

/// Solve the Signorini saddle-point system via projected Uzawa.
///
/// Inputs:
/// - `k_global`: global stiffness matrix `K` (assumed SPD after
///   Dirichlet BCs are baked in — the caller applies BCs *before*
///   handing `K` here).
/// - `c_matrix`: contact constraint matrix from
///   [`assemble_constraint_matrix`]. Shape `n_active × n_dofs`.
/// - `f`: external force / Dirichlet-shifted RHS, length `n_dofs`.
/// - `g`: target gaps, length `n_active`.
/// - `max_iters`: outer Uzawa iteration cap.
/// - `tol`: outer convergence tolerance on `‖g − C u‖_∞` (primal
///   residual of the *active* constraints).
///
/// Returns the final `(u, λ)` plus iteration diagnostics.
///
/// Optional knobs default to:
/// - Dual step `ρ = 1 / λ_max(C K⁻¹ C^T)` — Lipschitz-inverse estimated
///   via a short matrix-free power iteration (see
///   [`estimate_dual_step`]).
/// - Inner CG tolerance [`DEFAULT_INNER_CG_TOL`].
/// - Inner CG iteration cap [`DEFAULT_INNER_CG_MAX_ITERS`].
///
/// Use [`solve_augmented_system_with`] to override any of those.
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
    // `c_matrix` may carry a dummy 1×n_dofs "no-active" pattern (see
    // `assemble_constraint_matrix`); the dual update path handles
    // that gracefully via the `n_active == 0` short-circuit.
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
    // Warm start `u` from the unconstrained solve `K u = f`. This is
    // the same first inner CG that Uzawa would do anyway with
    // `λ = 0`, so caching it costs nothing.
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

        // ── Compute primal residual on the *active* constraints ───
        // residual_k = g_k − (C u)_k. A negative entry means the
        // constraint is over-satisfied (gap larger than target) and
        // we should release the multiplier; a positive entry means
        // the constraint is violated (penetration) and λ must grow.
        let cu = if n_active > 0 {
            matvec(c_matrix, &u)
        } else {
            Vec::new()
        };
        let mut residual = vec![0.0; n_active];
        let mut max_violation: Real = 0.0;
        for k in 0..n_active {
            residual[k] = g[k] - cu[k];
            // Track only the active violation: λ > 0 entries care
            // about both signs, but λ = 0 entries only care about
            // *positive* residual (i.e. still penetrating).
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

        // ── Projected dual update ────────────────────────────────
        // λ ← max(0, λ + ρ · residual). The `max(0, ·)` projection
        // enforces λ ≥ 0 — physically: contact can push but not
        // pull. A pair whose multiplier clamps to zero is a
        // candidate for active-set release in v2.
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

        // ── Convergence test ────────────────────────────────────
        if primal_residual < tol && dual_residual < tol {
            break;
        }

        // ── Inner primal solve: K u = f + C^T λ ─────────────────
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
            &u, // warm-start from previous iterate
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

/// Compute `y = C^T · x` where `C` is `n_active × n_dofs` (B = 1).
/// We don't materialise `C^T` — the constraint matrix is short and
/// fat (`n_active << n_dofs`) so walking its rows once per dual
/// update is cheaper than transposing.
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
/// Each step computes `y ← C K⁻¹ C^T y / ‖y‖`, using inner CG to apply
/// `K⁻¹`. Three to five iterations are normally enough to nail
/// `λ_max` to ~10 % — the inverse of which is a *conservative*
/// step (always inside the `(0, 2 / λ_max)` stability band, never
/// optimal but never oscillating).
///
/// Returns `1.0` if `n_active == 0` (the caller never enters the dual
/// update path in that case so the value is irrelevant).
pub fn estimate_dual_step(
    k: &BsrMatrix,
    c: &BsrMatrix,
    n_active: usize,
    inner_cg_tol: Real,
    inner_cg_max_iters: usize,
) -> Real {
    if n_active == 0 {
        return 1.0;
    }
    let n_dofs = k.nrows();
    // Seed with a deterministic vector (not the all-ones — that would
    // be a near-eigenvector for translational rigid-body modes after
    // BCs, which can confuse the estimate). A length-`n_active` ramp
    // is fine.
    let mut y: Vec<Real> = (0..n_active).map(|i| 1.0 + i as Real * 0.5).collect();
    let mut lam_max: Real = 1.0;
    for _ in 0..6 {
        // y_next = C · K⁻¹ · C^T · y
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
        // Rayleigh quotient: λ ≈ y_next · y / (y · y)
        let y_dot_y: Real = y.iter().map(|v| v * v).sum();
        let rq: Real = y.iter().zip(y_next.iter()).map(|(a, b)| a * b).sum::<Real>()
            / y_dot_y.max(1.0e-30);
        lam_max = rq.abs().max(1.0e-30);
        // Normalise for the next step.
        for v in y.iter_mut() {
            *v /= norm;
        }
        // Replace y with the normalised image for the next iteration.
        // (Power iteration: y ← y_next / ‖y_next‖.)
        let inv = 1.0 / norm;
        for (a, b) in y.iter_mut().zip(y_next.iter()) {
            *a = b * inv;
        }
    }
    1.0 / lam_max
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1-DoF toy: K = [2.0], f = [-3.0] (would push u to -1.5),
    /// constraint u ≥ 0. Expect u = 0, λ = 3.0.
    #[test]
    fn uzawa_one_dof_clamps_to_constraint() {
        let k = BsrMatrix::from_scalar_triplets(3, 3, &[
            (0, 0, 2.0),
            (1, 1, 2.0),
            (2, 2, 2.0),
        ])
        .unwrap();
        let f = vec![0.0, 0.0, -3.0]; // push z-component negative.
        // One contact pair: slave_dof = 0 (3 components 0..3),
        // normal = +z, target_gap = 0.
        let pairs = vec![(0u32, Vec3::new(0.0, 0.0, 1.0), 0.0)];
        let c = assemble_constraint_matrix(&pairs, 3).unwrap();
        let g = target_gap_vector(&pairs);
        let res = solve_augmented_system(&k, &c, &f, &g, 200, 1.0e-12).unwrap();
        assert!(
            res.u[2].abs() < 1.0e-8,
            "u_z should clamp to 0, got {}",
            res.u[2]
        );
        // λ = 3.0 because K u + C^T λ = f, with C^T λ = [0,0,λ] and
        // K u = 0 (clamped), gives λ = −f_z = 3.0... but our sign
        // convention has K u = f + C^T λ ⇒ 0 = -3 + λ ⇒ λ = 3.
        assert!(
            (res.lambda[0] - 3.0).abs() < 1.0e-6,
            "λ should converge to 3.0, got {}",
            res.lambda[0]
        );
        assert!(res.lambda[0] >= -1.0e-9, "λ must be ≥ 0");
    }
}
