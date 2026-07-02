//! Top-level FEM solvers — steady-state Laplace / Poisson and
//! transient parabolic (heat-equation) backward Euler.
//!
//! Both routines layer on top of [`crate::math::iterative::cg_solve`]:
//! the global stiffness / mass matrices are assembled into CSR
//! storage but never copied into a dense form. CG only needs the
//! matvec product, which the closure-driven API expresses naturally.
//!
//! # Algorithms
//!
//! - **Steady state** — assemble `K`, apply Dirichlet BCs to `(K, f)`,
//!   then solve `K · u = f` via CG.
//! - **Transient** — backward Euler:
//!
//!   `(M / Δt + K) · uⁿ⁺¹ = f + (M / Δt) · uⁿ`
//!
//!   The Dirichlet enforcement is applied to the LHS = `M / Δt + K`
//!   and the assembled RHS each step, so the solution is consistent
//!   with the BC at every time level.
//!
//! [`super::bc::apply_dirichlet`] does symmetric reduction (zeroes
//! the column too and shifts the RHS) so the reduced matrix is
//! still SPD when the parent is. CG retains its convergence
//! guarantee on the BC rows; non-zero Dirichlet values produce the
//! correct RHS shift instead of silently relying on row-only
//! elimination.

use crate::math::Real;
use crate::math::iterative::{CgResult, JacobiPreconditioner, cg_solve, pcg_solve};
use crate::math::sparse::BsrMatrix;

use super::assembly::matvec;
use super::bc::{BoundaryCondition, apply_dirichlet};

#[cfg(feature = "gpu")]
use crate::compute::Engine;
#[cfg(feature = "gpu")]
use crate::core::Error;
#[cfg(any(feature = "gpu", feature = "russell"))]
use crate::core::Result;
#[cfg(feature = "gpu")]
use crate::fem::gpu_data::FemGpuData;
#[cfg(feature = "gpu")]
use crate::gpu_kernels::{FemCgGpuKernel, FemGpuKernel, TransientStepOpsKernel, WaveStepOpsKernel};
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// Default tolerance for the FEM CG solves — `1e-8` is sufficient for
/// the v1 contract tests at 16³ resolution.
pub const DEFAULT_TOL: Real = 1.0e-8;

/// Default iteration cap — generous for v1 mesh sizes (≤16³).
pub const DEFAULT_MAX_ITERS: usize = 2000;

/// Small recycled-Krylov cache for repeated transient solves with the
/// same left-hand side.
///
/// The cache stores recent correction directions from previous time
/// steps. Before launching CG/PCG, the transient solver computes a
/// tiny minimum-residual projection in `x0 + span(cache)` and uses that
/// projected vector as the restarted initial guess. This is deliberately
/// lightweight: it is a CPU oracle / API contract for "Krylov restart"
/// behavior, not a full GCRO-DR implementation.
#[derive(Clone, Debug)]
pub struct KrylovRestartCache {
    max_vectors: usize,
    vectors: Vec<Vec<Real>>,
}

impl KrylovRestartCache {
    #[must_use]
    pub fn new(max_vectors: usize) -> Self {
        Self {
            max_vectors,
            vectors: Vec::new(),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    pub fn clear(&mut self) {
        self.vectors.clear();
    }

    fn push_direction(&mut self, mut direction: Vec<Real>) {
        if self.max_vectors == 0 || direction.is_empty() {
            return;
        }
        let norm = direction.iter().map(|v| v * v).sum::<Real>().sqrt();
        if norm <= 1.0e-30 {
            return;
        }
        for v in &mut direction {
            *v /= norm;
        }
        if self.vectors.len() == self.max_vectors {
            self.vectors.remove(0);
        }
        self.vectors.push(direction);
    }
}

/// Solve `K · u = f` with Dirichlet BCs applied to the row-reduced
/// system. Returns the full DoF-length solution vector.
///
/// `k` is consumed (reduced in place); pass a fresh assembly per call.
/// Use [`solve_steady_state_with_initial`] if you have a warm start.
pub fn solve_steady_state(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
    max_iters: usize,
    tol: Real,
) -> Vec<Real> {
    solve_steady_state_with_initial(k, f, bcs, &vec![0.0; f.len()], max_iters, tol).x
}

/// Stage 36.7 — direct-solver sibling of [`solve_steady_state`].
///
/// Same input + output contract as [`solve_steady_state`]: assemble
/// `K`, run the symmetric Dirichlet reduction, then solve the
/// reduced `K_red · u = f_red` system. The only difference is the
/// inner solve — instead of iterating with CG it routes through a
/// direct LU factorisation via [`crate::fem::russell::DirectSolver`].
/// That makes the routine a drop-in alternative when CG stalls
/// (stiff problems, indefinite systems) or pays more overhead than
/// a one-shot factor (very small systems, repeated solves with the
/// same matrix).
///
/// For the "factor-once, back-solve-many" workflow (one matrix,
/// many RHS — e.g. adjoint loops) construct a [`crate::fem::russell::DirectSolver`]
/// directly: call `factorize(&k_red)` once, then `back_solve(&b)`
/// in a loop. This wrapper rebuilds the factorisation every call.
#[cfg(feature = "russell")]
pub fn solve_steady_state_direct(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
) -> Result<Vec<Real>> {
    let mut k_red = k.clone();
    let mut f_red = f.to_vec();
    apply_dirichlet(&mut k_red, &mut f_red, bcs);
    let mut solver = super::russell::DirectSolver::new()?;
    solver.solve(&k_red, &f_red)
}

/// Variant of [`solve_steady_state`] that returns the full
/// [`CgResult`] (residual + iteration count) and accepts an initial
/// guess for warm-started repeated solves.
pub fn solve_steady_state_with_initial(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
    x0: &[Real],
    max_iters: usize,
    tol: Real,
) -> CgResult {
    let mut k_red = k.clone();
    let mut f_red = f.to_vec();
    apply_dirichlet(&mut k_red, &mut f_red, bcs);
    let mut x_init = x0.to_vec();
    // Make sure the initial guess is consistent with the Dirichlet
    // values — otherwise CG burns iterations correcting it.
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            x_init[*vertex as usize] = *value;
        }
    }
    cg_solve(|x| matvec(&k_red, x), &f_red, &x_init, tol, max_iters)
}

/// Jacobi-preconditioned variant of [`solve_steady_state_with_initial`].
///
/// Same input/output contract — applies symmetric Dirichlet reduction
/// then solves `K_red · u = f_red` via [`pcg_solve`] using
/// `M⁻¹ = diag(K_red)⁻¹` (Jacobi). For Laplacian-like SPD systems
/// this typically cuts CG iter count by 25-40% with no per-iter cost
/// beyond the diagonal apply (one elementwise multiply per row).
///
/// The preconditioner is built once from `K_red`'s diagonal — this
/// is `O(n)` and dwarfed by even a single CG iteration.
pub fn solve_steady_state_pcg(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
    x0: &[Real],
    max_iters: usize,
    tol: Real,
) -> CgResult {
    let mut k_red = k.clone();
    let mut f_red = f.to_vec();
    apply_dirichlet(&mut k_red, &mut f_red, bcs);
    let pre = JacobiPreconditioner::new(&k_red.diagonal());
    let mut x_init = x0.to_vec();
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            x_init[*vertex as usize] = *value;
        }
    }
    pcg_solve(
        |x| matvec(&k_red, x),
        |r| pre.apply(r),
        &f_red,
        &x_init,
        tol,
        max_iters,
    )
}

/// One backward-Euler step of the parabolic heat equation:
///
/// `(M / Δt + K) · uⁿ⁺¹ = f + (M / Δt) · uⁿ`
///
/// `m` is the mass matrix, `k` the stiffness (already scaled by the
/// diffusivity `α`); `bcs` are enforced on the LHS + RHS. Returns the
/// new state vector. Use [`solve_transient_with_initial`] if you want
/// to plug in a warm start (otherwise the previous `u_old` is the
/// natural initial guess and is used here).
pub fn solve_transient(
    m: &BsrMatrix,
    k: &BsrMatrix,
    f: &[Real],
    u_old: &[Real],
    dt: Real,
    bcs: &[BoundaryCondition],
) -> Vec<Real> {
    solve_transient_with_initial(
        m,
        k,
        f,
        u_old,
        dt,
        bcs,
        u_old,
        DEFAULT_MAX_ITERS,
        DEFAULT_TOL,
    )
    .x
}

/// Variant of [`solve_transient`] exposing the iteration count +
/// residual + initial guess. Useful when stepping repeatedly in a
/// viewer loop where you want to display residual telemetry.
#[allow(clippy::too_many_arguments)]
pub fn solve_transient_with_initial(
    m: &BsrMatrix,
    k: &BsrMatrix,
    f: &[Real],
    u_old: &[Real],
    dt: Real,
    bcs: &[BoundaryCondition],
    x0: &[Real],
    max_iters: usize,
    tol: Real,
) -> CgResult {
    debug_assert_eq!(m.b, 1);
    debug_assert_eq!(k.b, 1);
    debug_assert_eq!(m.nrows(), k.nrows());
    debug_assert_eq!(m.nrows(), f.len());
    debug_assert_eq!(m.nrows(), u_old.len());

    let inv_dt = 1.0 / dt;
    // Build the LHS A = M/Δt + K. We materialise this CSR so the
    // closure-driven `cg_solve` can call `matvec(&a, x)` directly.
    let a = add_csr_scaled(m, inv_dt, k, 1.0);
    // RHS b = f + (M/Δt) · u_old.
    let m_uold = matvec(m, u_old);
    let mut b: Vec<Real> = f
        .iter()
        .zip(m_uold.iter())
        .map(|(fi, mi)| fi + inv_dt * mi)
        .collect();

    // Apply Dirichlet to the LHS + RHS in tandem.
    let mut a_red = a;
    apply_dirichlet(&mut a_red, &mut b, bcs);
    let mut x_init = x0.to_vec();
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            x_init[*vertex as usize] = *value;
        }
    }
    cg_solve(|x| matvec(&a_red, x), &b, &x_init, tol, max_iters)
}

/// Jacobi-preconditioned variant of [`solve_transient_with_initial`].
/// See [`solve_steady_state_pcg`] for the rationale — the per-step
/// LHS `A = M/Δt + K` is rebuilt and BC-reduced as in the
/// non-preconditioned path; the diagonal extraction adds `O(n)` per
/// step and the iter-count cut typically pays for it many times over.
#[allow(clippy::too_many_arguments)]
pub fn solve_transient_pcg(
    m: &BsrMatrix,
    k: &BsrMatrix,
    f: &[Real],
    u_old: &[Real],
    dt: Real,
    bcs: &[BoundaryCondition],
    x0: &[Real],
    max_iters: usize,
    tol: Real,
) -> CgResult {
    debug_assert_eq!(m.b, 1);
    debug_assert_eq!(k.b, 1);
    debug_assert_eq!(m.nrows(), k.nrows());

    let inv_dt = 1.0 / dt;
    let a = add_csr_scaled(m, inv_dt, k, 1.0);
    let m_uold = matvec(m, u_old);
    let mut b: Vec<Real> = f
        .iter()
        .zip(m_uold.iter())
        .map(|(fi, mi)| fi + inv_dt * mi)
        .collect();
    let mut a_red = a;
    apply_dirichlet(&mut a_red, &mut b, bcs);
    let pre = JacobiPreconditioner::new(&a_red.diagonal());
    let mut x_init = x0.to_vec();
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            x_init[*vertex as usize] = *value;
        }
    }
    pcg_solve(
        |x| matvec(&a_red, x),
        |r| pre.apply(r),
        &b,
        &x_init,
        tol,
        max_iters,
    )
}

/// Jacobi-PCG transient solve with a recycled Krylov restart cache.
///
/// This builds the same reduced system as [`solve_transient_pcg`], but
/// first improves `x0` by a small minimum-residual projection over the
/// directions retained in `cache`. After the PCG solve converges, the
/// accepted correction is normalised and pushed back into the cache for
/// the next time step.
#[allow(clippy::too_many_arguments)]
pub fn solve_transient_pcg_restarted(
    m: &BsrMatrix,
    k: &BsrMatrix,
    f: &[Real],
    u_old: &[Real],
    dt: Real,
    bcs: &[BoundaryCondition],
    x0: &[Real],
    cache: &mut KrylovRestartCache,
    max_iters: usize,
    tol: Real,
) -> CgResult {
    debug_assert_eq!(m.b, 1);
    debug_assert_eq!(k.b, 1);
    debug_assert_eq!(m.nrows(), k.nrows());

    let inv_dt = 1.0 / dt;
    let a = add_csr_scaled(m, inv_dt, k, 1.0);
    let m_uold = matvec(m, u_old);
    let mut b: Vec<Real> = f
        .iter()
        .zip(m_uold.iter())
        .map(|(fi, mi)| fi + inv_dt * mi)
        .collect();
    let mut a_red = a;
    apply_dirichlet(&mut a_red, &mut b, bcs);
    let pre = JacobiPreconditioner::new(&a_red.diagonal());

    let mut x_init = x0.to_vec();
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            x_init[*vertex as usize] = *value;
        }
    }
    let projected_x0 = krylov_project_initial_guess(&a_red, &b, &x_init, cache);
    let result = pcg_solve(
        |x| matvec(&a_red, x),
        |r| pre.apply(r),
        &b,
        &projected_x0,
        tol,
        max_iters,
    );
    let direction: Vec<Real> = result
        .x
        .iter()
        .zip(projected_x0.iter())
        .map(|(x, x0)| x - x0)
        .collect();
    cache.push_direction(direction);
    result
}

fn krylov_project_initial_guess(
    a: &BsrMatrix,
    b: &[Real],
    x0: &[Real],
    cache: &KrylovRestartCache,
) -> Vec<Real> {
    let m = cache.vectors.len();
    if m == 0 {
        return x0.to_vec();
    }
    let ax0 = matvec(a, x0);
    let residual: Vec<Real> = b.iter().zip(ax0.iter()).map(|(bi, ai)| bi - ai).collect();
    let av: Vec<Vec<Real>> = cache.vectors.iter().map(|v| matvec(a, v)).collect();
    let mut normal = vec![vec![0.0; m]; m];
    let mut rhs = vec![0.0; m];
    for i in 0..m {
        rhs[i] = dot(&av[i], &residual);
        for j in 0..m {
            normal[i][j] = dot(&av[i], &av[j]);
        }
        normal[i][i] += 1.0e-24;
    }
    let Some(coeff) = solve_small_dense(normal, rhs) else {
        return x0.to_vec();
    };
    let mut x = x0.to_vec();
    for (c, v) in coeff.iter().zip(cache.vectors.iter()) {
        for (xi, vi) in x.iter_mut().zip(v.iter()) {
            *xi += c * vi;
        }
    }
    x
}

fn dot(a: &[Real], b: &[Real]) -> Real {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn solve_small_dense(mut a: Vec<Vec<Real>>, mut b: Vec<Real>) -> Option<Vec<Real>> {
    let n = b.len();
    for k in 0..n {
        let mut pivot = k;
        let mut pivot_abs = a[k][k].abs();
        for (r, row) in a.iter().enumerate().skip(k + 1) {
            if row[k].abs() > pivot_abs {
                pivot = r;
                pivot_abs = row[k].abs();
            }
        }
        if pivot_abs <= 1.0e-30 {
            return None;
        }
        if pivot != k {
            a.swap(k, pivot);
            b.swap(k, pivot);
        }
        let diag = a[k][k];
        for j in k..n {
            a[k][j] /= diag;
        }
        b[k] /= diag;
        for r in 0..n {
            if r == k {
                continue;
            }
            let factor = a[r][k];
            if factor == 0.0 {
                continue;
            }
            for j in k..n {
                a[r][j] -= factor * a[k][j];
            }
            b[r] -= factor * b[k];
        }
    }
    Some(b)
}

/// State produced by one [`solve_wave_transient`] step — the new
/// displacement, velocity and acceleration vectors at `t + Δt`.
#[derive(Clone, Debug)]
pub struct WaveStepResult {
    pub u_new: Vec<Real>,
    pub v_new: Vec<Real>,
    pub a_new: Vec<Real>,
    /// Inner CG residual on the implicit solve.
    pub residual: Real,
    /// Inner CG iteration count.
    pub iters: usize,
}

/// Default Newmark-β parameters — `(β = 1/4, γ = 1/2)` is the
/// constant-average-acceleration (a.k.a. "trapezoidal-rule") variant,
/// unconditionally stable for linear undamped wave problems.
pub const NEWMARK_BETA_DEFAULT: Real = 0.25;
pub const NEWMARK_GAMMA_DEFAULT: Real = 0.5;

/// One implicit Newmark-β step of the second-order wave equation
/// `M · ä + K · u = 0`.
///
/// Algorithm (textbook Hughes 9.1.2):
///
/// 1. **Predictor** `u_pred = u_old + Δt·v_old + (½ − β)·Δt²·a_old`.
/// 2. **Solve** `(M / (β·Δt²) + K) · u_new = (M / (β·Δt²)) · u_pred`
///    using `cg_solve` (CSR LHS materialised via [`add_csr_scaled`]).
/// 3. **Corrector**
///    `a_new = (u_new − u_pred) / (β·Δt²)`,
///    `v_new = v_old + Δt · ((1 − γ)·a_old + γ·a_new)`.
///
/// `β = 0.25`, `γ = 0.5` (constant-average-acceleration) is
/// unconditionally stable; smaller `β` introduces numerical damping
/// at the price of conditional stability.
///
/// Dirichlet BCs are enforced exactly the same way as in
/// [`solve_steady_state`] / [`solve_transient`]: zero the row, set
/// the diagonal to 1, and overwrite the matching RHS entry with the
/// prescribed `value`. The corrector step then derives `a_new` and
/// `v_new` from the BC-consistent `u_new`.
///
/// After the corrector, `v_new` and `a_new` are zeroed on every
/// Dirichlet DoF. Without that, a non-zero `v_old` or `a_old` on a
/// pinned DoF (e.g. seeded from a consistent initial-acceleration
/// solve) gets carried into `a_new` via the predictor `u_pred` and
/// then integrated into spurious boundary motion across steps. For
/// time-varying Dirichlet data, callers should still overwrite the
/// BC entries of `v_new` / `a_new` with the prescribed velocity /
/// acceleration after this routine returns.
#[allow(clippy::too_many_arguments)]
pub fn solve_wave_transient(
    m: &BsrMatrix,
    k: &BsrMatrix,
    u_old: &[Real],
    v_old: &[Real],
    a_old: &[Real],
    dt: Real,
    beta: Real,
    gamma: Real,
    bcs: &[BoundaryCondition],
) -> WaveStepResult {
    debug_assert_eq!(m.b, 1);
    debug_assert_eq!(k.b, 1);
    debug_assert_eq!(m.nrows(), k.nrows());
    debug_assert_eq!(m.nrows(), u_old.len());
    debug_assert_eq!(m.nrows(), v_old.len());
    debug_assert_eq!(m.nrows(), a_old.len());
    debug_assert!(beta > 0.0, "Newmark β must be > 0 for the implicit solve");
    debug_assert!(dt > 0.0, "Newmark Δt must be > 0");

    let n = u_old.len();
    let bdt2_inv = 1.0 / (beta * dt * dt);
    let half_minus_beta = 0.5 - beta;

    // Predictor — `u_pred = u_old + Δt·v_old + (½ − β)·Δt²·a_old`.
    let mut u_pred = vec![0.0; n];
    for i in 0..n {
        u_pred[i] = u_old[i] + dt * v_old[i] + half_minus_beta * dt * dt * a_old[i];
    }

    // LHS A = M / (β·Δt²) + K.
    let a_lhs = add_csr_scaled(m, bdt2_inv, k, 1.0);
    // RHS b = (M / (β·Δt²)) · u_pred.
    let m_upred = matvec(m, &u_pred);
    let mut b: Vec<Real> = m_upred.iter().map(|v| bdt2_inv * v).collect();

    // Apply Dirichlet to (A, b) in tandem.
    let mut a_red = a_lhs;
    apply_dirichlet(&mut a_red, &mut b, bcs);

    // Initial guess: `u_old` is the natural warm-start; seed Dirichlet
    // entries so the first CG residual on those rows is already zero.
    let mut x_init = u_old.to_vec();
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = bc {
            x_init[*vertex as usize] = *value;
        }
    }
    let cg = cg_solve(
        |x| matvec(&a_red, x),
        &b,
        &x_init,
        DEFAULT_TOL,
        DEFAULT_MAX_ITERS,
    );
    let u_new = cg.x;

    // Corrector — `a_new = (u_new − u_pred) / (β·Δt²)`,
    // `v_new = v_old + Δt · ((1 − γ)·a_old + γ·a_new)`.
    let mut a_new = vec![0.0; n];
    let mut v_new = vec![0.0; n];
    for i in 0..n {
        a_new[i] = (u_new[i] - u_pred[i]) * bdt2_inv;
        v_new[i] = v_old[i] + dt * ((1.0 - gamma) * a_old[i] + gamma * a_new[i]);
    }

    // Zero `v_new` and `a_new` on every Dirichlet DoF — the pinned
    // displacement makes `u_new − u_pred` non-zero through `u_pred`'s
    // dependence on `v_old` and `a_old`, which would otherwise leak
    // spurious motion into the boundary across steps.
    for bc in bcs {
        match bc {
            BoundaryCondition::Dirichlet { vertex, .. } => {
                let idx = *vertex as usize;
                v_new[idx] = 0.0;
                a_new[idx] = 0.0;
            }
            BoundaryCondition::DirichletVector { vertex, values } => {
                let base = *vertex as usize * values.len();
                for c in 0..values.len() {
                    v_new[base + c] = 0.0;
                    a_new[base + c] = 0.0;
                }
            }
            _ => {}
        }
    }

    WaveStepResult {
        u_new,
        v_new,
        a_new,
        residual: cg.residual,
        iters: cg.iters,
    }
}

// ─── GPU wave transient (Stage 36.5 Slice E + Slice H) ──────────────

/// Per-step diagnostic returned by [`WaveStepperGpu::step`] — inner
/// CG residual + iteration count. The new (Slice H) API leaves
/// `(u, v, a)` resident on the GPU; call [`WaveStepperGpu::read_state`]
/// (or [`WaveStepperGpu::read_u`] alone) to copy them back to host
/// when the viewer needs to render.
#[cfg(feature = "gpu")]
#[derive(Copy, Clone, Debug)]
pub struct WaveStepStats {
    /// Inner CG residual on the implicit solve.
    pub residual: Real,
    /// Inner CG iteration count.
    pub iters: usize,
}

/// Cached state for a [`solve_wave_transient`] simulation that runs
/// the implicit-Newmark inner CG on the GPU. Build once per scene
/// (fixes mesh, BCs, β, γ, Δt); call [`Self::step`] every frame.
///
/// **Slice H** keeps `u`, `v`, `a` resident on the GPU as `f32`
/// storage buffers across timesteps. The predictor, RHS construction
/// (with BC shift + pin overwrite), CG solve, and corrector all run
/// on GPU dispatches with **zero** host↔device traffic per step. The
/// viewer reads back `u` (or the full state) only when it needs to
/// render — typically once per render frame, not once per substep.
///
/// Supports **arbitrary** Dirichlet values (zero or non-zero) via a
/// pre-computed column-elimination shift `bc_shift` baked at
/// construction — the per-step RHS is `b_unreduced + bc_shift` on
/// unpinned rows, with pinned rows overwritten to the prescribed
/// values. This matches what `apply_bcs` would do on every step but
/// without the per-step matrix touch.
#[cfg(feature = "gpu")]
pub struct WaveStepperGpu {
    /// BC-reduced `A = M / (β·Δt²) + K` uploaded as GPU CSR.
    a_gpu: FemGpuData,
    /// Mass matrix uploaded as GPU CSR — used by the per-step
    /// `M · u_pred` matvec on the GPU. Slice H removed the CPU
    /// matvec + per-step f32 upload that the original code did.
    m_gpu: FemGpuData,
    /// Linear-system size.
    n: usize,
    /// Newmark parameters (kept for API introspection — the per-step
    /// values are baked into `bdt2_inv` / `half_minus_beta_dt2`
    /// floats below).
    beta: Real,
    gamma: Real,
    dt: Real,
    bdt2_inv: Real,
    half_minus_beta_dt2: Real,
    /// Persistent GPU state buffers (length `n`, `f32`). Survive
    /// across calls to [`Self::step`].
    u_buf: Arc<wgpu::Buffer>,
    v_buf: Arc<wgpu::Buffer>,
    a_buf: Arc<wgpu::Buffer>,
    /// Per-step scratch — `u_pred`, `m_upred`, `b`, `x_init` (a copy
    /// of `u` used to seed CG). Allocated once at construction.
    u_pred_buf: Arc<wgpu::Buffer>,
    m_upred_buf: Arc<wgpu::Buffer>,
    b_buf: Arc<wgpu::Buffer>,
    /// Precomputed BC column-shift uploaded once (length `n`, `f32`).
    bc_shift_buf: Arc<wgpu::Buffer>,
    /// Per-row pin bitset (1 = pinned, 0 = free), `u32` storage buffer.
    is_pinned_buf: Arc<wgpu::Buffer>,
    /// Per-row prescribed Dirichlet value; unused on free rows.
    pinned_value_buf: Arc<wgpu::Buffer>,
    /// Pre-computed `1 / diag(A_red)` for the Jacobi preconditioner.
    /// Built once at construction by extracting the GPU CSR diagonal,
    /// inverting on host (zeroing rows where the diagonal is zero
    /// or near-singular), and uploading. The per-step CG runs as PCG
    /// using this — typical 25-30% iteration reduction with no
    /// per-step diag-extract overhead.
    inv_diag_buf: Arc<wgpu::Buffer>,
    cg_kernel: FemCgGpuKernel,
    mv_kernel: FemGpuKernel,
    ops_kernel: WaveStepOpsKernel,
}

#[cfg(feature = "gpu")]
impl WaveStepperGpu {
    /// Build the stepper from initial state. `m`, `k`, `dt`, `beta`,
    /// `gamma`, `bcs` are fixed for the lifetime of the stepper — if
    /// any of them change, build a new one. The initial `(u, v, a)`
    /// are uploaded once and survive across `step` calls; mutate them
    /// later via [`Self::write_state`] (e.g. on user reset).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        engine: &Engine,
        mesh: &super::FemMesh,
        m: &BsrMatrix,
        k: &BsrMatrix,
        dt: Real,
        beta: Real,
        gamma: Real,
        bcs: &[BoundaryCondition],
        u: &[Real],
        v: &[Real],
        a: &[Real],
    ) -> Result<Self> {
        debug_assert_eq!(m.b, 1, "WaveStepperGpu: M must be B=1");
        debug_assert_eq!(k.b, 1, "WaveStepperGpu: K must be B=1");
        debug_assert_eq!(m.nrows(), k.nrows(), "WaveStepperGpu: M, K row mismatch");
        debug_assert!(beta > 0.0, "WaveStepperGpu: β must be > 0");
        debug_assert!(dt > 0.0, "WaveStepperGpu: Δt must be > 0");
        debug_assert_eq!(u.len(), m.nrows(), "WaveStepperGpu: u length mismatch");
        debug_assert_eq!(v.len(), m.nrows(), "WaveStepperGpu: v length mismatch");
        debug_assert_eq!(a.len(), m.nrows(), "WaveStepperGpu: a length mismatch");

        let n = m.nrows();
        let bdt2_inv = 1.0 / (beta * dt * dt);
        let half_minus_beta_dt2 = (0.5 - beta) * dt * dt;

        // Per-vertex DoF count inferred from the assembled system —
        // scalar wave ⇒ dim = 1, interleaved vector fields ⇒
        // n = vert_count · dim (same layout `FemGpuData::upload`
        // accepts and the same indexing `apply_bcs` pins).
        let vc = mesh.vert_count();
        if vc == 0 || n == 0 || n % vc != 0 {
            return Err(Error::Gpu(format!(
                "WaveStepperGpu: system rows ({n}) is not a positive integer \
                 multiple of mesh vertices ({vc})"
            )));
        }
        let dim = n / vc;

        // Build A = M / (β·Δt²) + K, then run BC reduction on a
        // **zero** RHS to extract the column-elimination shift. After
        // the reduction, `bc_shift[i]` for unpinned `i` equals
        // `-Σ_r A_orig[i, r] · value[r]` (sum over pinned rows) and
        // for pinned `i` equals `value[i]`. The reduced `a_red` is
        // what we upload to the GPU; the shift becomes the per-step
        // RHS correction.
        let mut a_red = add_csr_scaled(m, bdt2_inv, k, 1.0);
        let mut bc_shift = vec![0.0; n];
        super::bc::apply_bcs(&mut a_red, &mut bc_shift, bcs, &mesh.vertices, dim);

        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("WaveStepperGpu::new: engine has no GPU device".into()))?;
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("WaveStepperGpu::new: engine has no GPU queue".into()))?;
        let a_gpu = FemGpuData::upload(device, queue, mesh, &a_red)?;
        // Mass matrix uploaded unmodified — used for `M · u_pred` on
        // the GPU. The matvec only reads `csr_*`; we don't apply the
        // BC reduction here.
        let m_gpu = FemGpuData::upload(device, queue, mesh, m)?;

        // Build the per-row pin bitset + prescribed-value array. Must
        // mirror the `apply_bcs` indexing above exactly or the GPU pin
        // metadata addresses rows the BC-reduced matrix doesn't pin:
        // scalar `Dirichlet` pins component 0 of the vertex
        // (`vertex · dim`), `DirichletVector` pins every component.
        let mut is_pinned_u32 = vec![0u32; n];
        let mut pinned_value_f32 = vec![0.0f32; n];
        for bc in bcs {
            match bc {
                BoundaryCondition::Dirichlet { vertex, value } => {
                    let dof = *vertex as usize * dim;
                    is_pinned_u32[dof] = 1;
                    pinned_value_f32[dof] = *value as f32;
                }
                BoundaryCondition::DirichletVector { vertex, values } => {
                    if values.len() != dim {
                        return Err(Error::Gpu(format!(
                            "WaveStepperGpu: DirichletVector with {} components on a \
                             dim-{dim} system; values.len() must equal the per-vertex \
                             DoF count",
                            values.len()
                        )));
                    }
                    let base = *vertex as usize * dim;
                    for (ci, val) in values.iter().enumerate() {
                        is_pinned_u32[base + ci] = 1;
                        pinned_value_f32[base + ci] = *val as f32;
                    }
                }
                _ => {}
            }
        }

        let bytes = (n.max(4) * 4) as u64;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;

        let make_buf = |label: &str| {
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: usage_rw,
                mapped_at_creation: false,
            }))
        };

        let u_buf = make_buf("molla::fem::wave_gpu::u");
        let v_buf = make_buf("molla::fem::wave_gpu::v");
        let a_buf = make_buf("molla::fem::wave_gpu::a");
        let u_pred_buf = make_buf("molla::fem::wave_gpu::u_pred");
        let m_upred_buf = make_buf("molla::fem::wave_gpu::m_upred");
        let b_buf = make_buf("molla::fem::wave_gpu::b");
        let bc_shift_buf = make_buf("molla::fem::wave_gpu::bc_shift");
        let is_pinned_buf = make_buf("molla::fem::wave_gpu::is_pinned");
        let pinned_value_buf = make_buf("molla::fem::wave_gpu::pinned_value");
        let inv_diag_buf = make_buf("molla::fem::wave_gpu::inv_diag");

        // Upload initial state + the constant per-row metadata.
        let u_f32: Vec<f32> = u.iter().map(|&v| v as f32).collect();
        let v_f32: Vec<f32> = v.iter().map(|&v| v as f32).collect();
        let a_f32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        let bc_shift_f32: Vec<f32> = bc_shift.iter().map(|&v| v as f32).collect();
        queue.write_buffer(&u_buf, 0, bytemuck::cast_slice(&u_f32));
        queue.write_buffer(&v_buf, 0, bytemuck::cast_slice(&v_f32));
        queue.write_buffer(&a_buf, 0, bytemuck::cast_slice(&a_f32));
        queue.write_buffer(&bc_shift_buf, 0, bytemuck::cast_slice(&bc_shift_f32));
        queue.write_buffer(&is_pinned_buf, 0, bytemuck::cast_slice(&is_pinned_u32));
        queue.write_buffer(
            &pinned_value_buf,
            0,
            bytemuck::cast_slice(&pinned_value_f32),
        );

        let cg_kernel = FemCgGpuKernel::new(device.clone(), queue.clone());
        let mv_kernel = FemGpuKernel::new(device.clone(), queue.clone());
        let ops_kernel = WaveStepOpsKernel::new(device.clone(), queue.clone());

        // Extract diag(A_red), invert host-side, upload — once. Per-step
        // CG runs as PCG with this preconditioner; ~30% iter reduction
        // on Laplacian-like systems (Jacobi sweep).
        mv_kernel.extract_diagonal(&a_gpu, &inv_diag_buf)?;
        let diag_f32 = cg_kernel.read_buffer(&inv_diag_buf, n as u32)?;
        let inv_diag_f32: Vec<f32> = diag_f32
            .iter()
            .map(|&d| if d.abs() > 1.0e-30 { 1.0 / d } else { 0.0 })
            .collect();
        queue.write_buffer(&inv_diag_buf, 0, bytemuck::cast_slice(&inv_diag_f32));

        Ok(Self {
            a_gpu,
            m_gpu,
            n,
            beta,
            gamma,
            dt,
            bdt2_inv,
            half_minus_beta_dt2,
            u_buf,
            v_buf,
            a_buf,
            u_pred_buf,
            m_upred_buf,
            b_buf,
            bc_shift_buf,
            is_pinned_buf,
            pinned_value_buf,
            inv_diag_buf,
            cg_kernel,
            mv_kernel,
            ops_kernel,
        })
    }

    /// Suppress unused-field warnings — `beta` is stored for API
    /// introspection (the per-step value is precomputed into
    /// `bdt2_inv` and `half_minus_beta_dt2`).
    #[allow(dead_code)]
    fn _keep_warnings_quiet(&self) -> Real {
        self.beta
    }

    /// Linear-system size (number of DoFs).
    pub fn n(&self) -> usize {
        self.n
    }

    /// One implicit Newmark-β step running entirely on the GPU.
    /// `(u, v, a)` are read + updated in their persistent storage
    /// buffers — no host↔device traffic per step. Returns CG
    /// diagnostics (residual + iters).
    pub fn step(&mut self) -> Result<WaveStepStats> {
        // 1. Predictor — `u_pred = u + dt·v + (½−β)·dt²·a`.
        self.ops_kernel.predictor(
            self.n as u32,
            self.dt as f32,
            self.half_minus_beta_dt2 as f32,
            &self.u_buf,
            &self.v_buf,
            &self.a_buf,
            &self.u_pred_buf,
        )?;

        // 2. m_upred = M · u_pred (GPU sparse matvec, unreduced M).
        self.mv_kernel
            .matvec(&self.m_gpu, &self.u_pred_buf, &self.m_upred_buf)?;

        // 3. RHS construction — `b[i] = bdt2_inv · m_upred[i] +
        //    bc_shift[i]` for unpinned i; `b[i] = pinned_value[i]`
        //    for pinned i.
        self.ops_kernel.rhs(
            self.n as u32,
            self.bdt2_inv as f32,
            &self.m_upred_buf,
            &self.bc_shift_buf,
            &self.is_pinned_buf,
            &self.pinned_value_buf,
            &self.b_buf,
        )?;

        // 4. PCG solve `(A_red) · u_new = b` with `u` as the warm
        //    start. The diag preconditioner was extracted at
        //    construction so the per-step cost is identical to plain
        //    CG except for the `z = M⁻¹·r` step (one tiny dispatch +
        //    one extra dot per iter) — net ~30% iter reduction on
        //    this Laplacian-like operator with no per-step diag-
        //    extract overhead. cg_solve_resident is still available
        //    for callers that want to compare.
        let mv_kernel = &self.mv_kernel;
        let a_gpu = &self.a_gpu;
        let cg_result = self.cg_kernel.pcg_solve_resident(
            self.n as u32,
            &self.b_buf,
            Some(&self.u_buf),
            &self.inv_diag_buf,
            DEFAULT_TOL as f32,
            DEFAULT_MAX_ITERS,
            |x_buf, y_buf| mv_kernel.matvec(a_gpu, x_buf, y_buf),
        )?;

        // 5. Corrector — `a_new = (u_new − u_pred) · bdt2_inv` and
        //    `v_new = v_old + dt · ((1−γ)·a_old + γ·a_new)`. Read
        //    `v` and `a` in-place (the kernel reads then writes).
        self.ops_kernel.corrector(
            self.n as u32,
            self.dt as f32,
            self.bdt2_inv as f32,
            self.gamma as f32,
            &cg_result.x,
            &self.u_pred_buf,
            &self.v_buf,
            &self.a_buf,
            &self.is_pinned_buf,
        )?;

        // 6. Promote the CG iterate into the persistent `u` buffer.
        self.ops_kernel
            .copy_u(self.n as u32, &cg_result.x, &self.u_buf)?;

        Ok(WaveStepStats {
            residual: cg_result.residual as Real,
            iters: cg_result.iters,
        })
    }

    /// Read the persistent `u` buffer back to host as `f64`. Call
    /// once per render frame, not once per substep.
    pub fn read_u(&self) -> Result<Vec<Real>> {
        let f32_buf = self.cg_kernel.read_buffer(&self.u_buf, self.n as u32)?;
        Ok(f32_buf.into_iter().map(|v| v as Real).collect())
    }

    /// Read the persistent `v` buffer.
    pub fn read_v(&self) -> Result<Vec<Real>> {
        let f32_buf = self.cg_kernel.read_buffer(&self.v_buf, self.n as u32)?;
        Ok(f32_buf.into_iter().map(|v| v as Real).collect())
    }

    /// Read the persistent `a` buffer.
    pub fn read_a(&self) -> Result<Vec<Real>> {
        let f32_buf = self.cg_kernel.read_buffer(&self.a_buf, self.n as u32)?;
        Ok(f32_buf.into_iter().map(|v| v as Real).collect())
    }

    /// Read the full `(u, v, a)` triple in one shot. Three sequential
    /// readbacks — same wall-clock as calling them individually.
    pub fn read_state(&self) -> Result<(Vec<Real>, Vec<Real>, Vec<Real>)> {
        Ok((self.read_u()?, self.read_v()?, self.read_a()?))
    }

    /// Overwrite the persistent state buffers from host. Used when
    /// the viewer wants to push a fresh initial condition without
    /// rebuilding the stepper (e.g. user clicked "Reset").
    pub fn write_state(&mut self, u: &[Real], v: &[Real], a: &[Real]) -> Result<()> {
        debug_assert_eq!(u.len(), self.n);
        debug_assert_eq!(v.len(), self.n);
        debug_assert_eq!(a.len(), self.n);
        let queue = self.cg_kernel.queue();
        let u_f32: Vec<f32> = u.iter().map(|&v| v as f32).collect();
        let v_f32: Vec<f32> = v.iter().map(|&v| v as f32).collect();
        let a_f32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
        queue.write_buffer(&self.u_buf, 0, bytemuck::cast_slice(&u_f32));
        queue.write_buffer(&self.v_buf, 0, bytemuck::cast_slice(&v_f32));
        queue.write_buffer(&self.a_buf, 0, bytemuck::cast_slice(&a_f32));
        Ok(())
    }
}

// ─── GPU transient (Stage 36.5 Slice F + Slice H) ────────────────────

/// Per-step diagnostic returned by [`TransientStepperGpu::step`].
/// Mirrors the [`WaveStepStats`] shape; see that type for the
/// rationale.
#[cfg(feature = "gpu")]
#[derive(Copy, Clone, Debug)]
pub struct TransientStepStats {
    pub residual: Real,
    pub iters: usize,
}

/// Cached state for a [`solve_transient`] simulation that runs the
/// implicit-Euler inner CG on the GPU. Build once per scene (fixes
/// mesh, BCs, Δt); call [`Self::step`] every frame.
///
/// **Slice H** keeps `u` resident on the GPU as an `f32` storage
/// buffer across timesteps. The RHS construction (with mass matvec
/// + BC shift + pin overwrite) and CG solve all run on GPU
/// dispatches — zero host↔device traffic per step. Read `u` back via
/// [`Self::read_u`] when the viewer needs to render.
///
/// Supports arbitrary Dirichlet values via a precomputed column-
/// elimination shift baked at construction (same trick as
/// [`WaveStepperGpu`]).
#[cfg(feature = "gpu")]
pub struct TransientStepperGpu {
    /// BC-reduced `A = M / Δt + K` uploaded as GPU CSR.
    a_gpu: FemGpuData,
    /// Mass matrix uploaded as GPU CSR — used for the per-step
    /// `M · u_old` matvec on the GPU.
    m_gpu: FemGpuData,
    n: usize,
    inv_dt: Real,
    /// Persistent state buffer (length `n`, `f32`).
    u_buf: Arc<wgpu::Buffer>,
    /// Per-step source-term buffer (length `n`, `f32`). Updated each
    /// step via [`Self::write_f`] when non-zero; defaults to zero.
    f_buf: Arc<wgpu::Buffer>,
    /// Per-step scratch buffers.
    m_uold_buf: Arc<wgpu::Buffer>,
    b_buf: Arc<wgpu::Buffer>,
    /// Constant per-row metadata.
    bc_shift_buf: Arc<wgpu::Buffer>,
    is_pinned_buf: Arc<wgpu::Buffer>,
    pinned_value_buf: Arc<wgpu::Buffer>,
    /// `1 / diag(A_red)` for the Jacobi preconditioner — extracted
    /// once at construction (see [`WaveStepperGpu::inv_diag_buf`] for
    /// the full rationale).
    inv_diag_buf: Arc<wgpu::Buffer>,
    cg_kernel: FemCgGpuKernel,
    mv_kernel: FemGpuKernel,
    ops_kernel: TransientStepOpsKernel,
}

#[cfg(feature = "gpu")]
impl TransientStepperGpu {
    /// Build the stepper from initial state. The initial `u` is
    /// uploaded once and survives across `step` calls; mutate it
    /// later via [`Self::write_u`] (e.g. on user reset).
    pub fn new(
        engine: &Engine,
        mesh: &super::FemMesh,
        m: &BsrMatrix,
        k: &BsrMatrix,
        dt: Real,
        bcs: &[BoundaryCondition],
        u: &[Real],
    ) -> Result<Self> {
        debug_assert_eq!(m.b, 1, "TransientStepperGpu: M must be B=1");
        debug_assert_eq!(k.b, 1, "TransientStepperGpu: K must be B=1");
        debug_assert_eq!(m.nrows(), k.nrows());
        debug_assert!(dt > 0.0, "TransientStepperGpu: Δt must be > 0");
        debug_assert_eq!(u.len(), m.nrows(), "TransientStepperGpu: u length mismatch");

        let n = m.nrows();
        let inv_dt = 1.0 / dt;
        // Per-vertex DoF count inferred from the assembled system
        // (see `WaveStepperGpu::new` — same layout contract).
        let vc = mesh.vert_count();
        if vc == 0 || n == 0 || n % vc != 0 {
            return Err(Error::Gpu(format!(
                "TransientStepperGpu: system rows ({n}) is not a positive integer \
                 multiple of mesh vertices ({vc})"
            )));
        }
        let dim = n / vc;
        // Build A = M/Δt + K, then BC-reduce on a zero RHS to extract
        // the column-elimination shift.
        let mut a_red = add_csr_scaled(m, inv_dt, k, 1.0);
        let mut bc_shift = vec![0.0; n];
        super::bc::apply_bcs(&mut a_red, &mut bc_shift, bcs, &mesh.vertices, dim);

        let device = engine.device().ok_or_else(|| {
            Error::Gpu("TransientStepperGpu::new: engine has no GPU device".into())
        })?;
        let queue = engine.queue().ok_or_else(|| {
            Error::Gpu("TransientStepperGpu::new: engine has no GPU queue".into())
        })?;
        let a_gpu = FemGpuData::upload(device, queue, mesh, &a_red)?;
        let m_gpu = FemGpuData::upload(device, queue, mesh, m)?;

        // Pin metadata must mirror the `apply_bcs` indexing above
        // (scalar `Dirichlet` → component 0; `DirichletVector` → every
        // component of the vertex).
        let mut is_pinned_u32 = vec![0u32; n];
        let mut pinned_value_f32 = vec![0.0f32; n];
        for bc in bcs {
            match bc {
                BoundaryCondition::Dirichlet { vertex, value } => {
                    let dof = *vertex as usize * dim;
                    is_pinned_u32[dof] = 1;
                    pinned_value_f32[dof] = *value as f32;
                }
                BoundaryCondition::DirichletVector { vertex, values } => {
                    if values.len() != dim {
                        return Err(Error::Gpu(format!(
                            "TransientStepperGpu: DirichletVector with {} components on \
                             a dim-{dim} system; values.len() must equal the per-vertex \
                             DoF count",
                            values.len()
                        )));
                    }
                    let base = *vertex as usize * dim;
                    for (ci, val) in values.iter().enumerate() {
                        is_pinned_u32[base + ci] = 1;
                        pinned_value_f32[base + ci] = *val as f32;
                    }
                }
                _ => {}
            }
        }

        let bytes = (n.max(4) * 4) as u64;
        let usage_rw = wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST;
        let make_buf = |label: &str| {
            Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: usage_rw,
                mapped_at_creation: false,
            }))
        };
        let u_buf = make_buf("molla::fem::transient_gpu::u");
        let f_buf = make_buf("molla::fem::transient_gpu::f");
        let m_uold_buf = make_buf("molla::fem::transient_gpu::m_uold");
        let b_buf = make_buf("molla::fem::transient_gpu::b");
        let bc_shift_buf = make_buf("molla::fem::transient_gpu::bc_shift");
        let is_pinned_buf = make_buf("molla::fem::transient_gpu::is_pinned");
        let pinned_value_buf = make_buf("molla::fem::transient_gpu::pinned_value");
        let inv_diag_buf = make_buf("molla::fem::transient_gpu::inv_diag");

        let u_f32: Vec<f32> = u.iter().map(|&v| v as f32).collect();
        let bc_shift_f32: Vec<f32> = bc_shift.iter().map(|&v| v as f32).collect();
        queue.write_buffer(&u_buf, 0, bytemuck::cast_slice(&u_f32));
        // f defaults to zero — explicitly write so any later
        // dispatch reads consistent data.
        queue.write_buffer(&f_buf, 0, &vec![0u8; bytes as usize]);
        queue.write_buffer(&bc_shift_buf, 0, bytemuck::cast_slice(&bc_shift_f32));
        queue.write_buffer(&is_pinned_buf, 0, bytemuck::cast_slice(&is_pinned_u32));
        queue.write_buffer(
            &pinned_value_buf,
            0,
            bytemuck::cast_slice(&pinned_value_f32),
        );

        let cg_kernel = FemCgGpuKernel::new(device.clone(), queue.clone());
        let mv_kernel = FemGpuKernel::new(device.clone(), queue.clone());
        let ops_kernel = TransientStepOpsKernel::new(device.clone(), queue.clone());

        // Extract diag(A_red) once + invert host-side for the Jacobi
        // preconditioner. Same pattern as WaveStepperGpu.
        mv_kernel.extract_diagonal(&a_gpu, &inv_diag_buf)?;
        let diag_f32 = cg_kernel.read_buffer(&inv_diag_buf, n as u32)?;
        let inv_diag_f32: Vec<f32> = diag_f32
            .iter()
            .map(|&d| if d.abs() > 1.0e-30 { 1.0 / d } else { 0.0 })
            .collect();
        queue.write_buffer(&inv_diag_buf, 0, bytemuck::cast_slice(&inv_diag_f32));

        Ok(Self {
            a_gpu,
            m_gpu,
            n,
            inv_dt,
            u_buf,
            f_buf,
            m_uold_buf,
            b_buf,
            bc_shift_buf,
            is_pinned_buf,
            pinned_value_buf,
            inv_diag_buf,
            cg_kernel,
            mv_kernel,
            ops_kernel,
        })
    }

    /// Linear-system size (number of DoFs).
    pub fn n(&self) -> usize {
        self.n
    }

    /// One backward-Euler step running entirely on the GPU. Returns
    /// CG diagnostics. The persistent `u` buffer is updated in place.
    pub fn step(&mut self) -> Result<TransientStepStats> {
        // 1. m_uold = M · u (GPU sparse matvec).
        self.mv_kernel
            .matvec(&self.m_gpu, &self.u_buf, &self.m_uold_buf)?;

        // 2. RHS construction.
        self.ops_kernel.rhs(
            self.n as u32,
            self.inv_dt as f32,
            &self.m_uold_buf,
            &self.f_buf,
            &self.bc_shift_buf,
            &self.is_pinned_buf,
            &self.pinned_value_buf,
            &self.b_buf,
        )?;

        // 3. PCG solve (warm-start from u, Jacobi preconditioner
        //    extracted at construction). ~30% iter reduction over
        //    plain CG with no per-step diag-extract overhead.
        let mv_kernel = &self.mv_kernel;
        let a_gpu = &self.a_gpu;
        let cg_result = self.cg_kernel.pcg_solve_resident(
            self.n as u32,
            &self.b_buf,
            Some(&self.u_buf),
            &self.inv_diag_buf,
            DEFAULT_TOL as f32,
            DEFAULT_MAX_ITERS,
            |x_buf, y_buf| mv_kernel.matvec(a_gpu, x_buf, y_buf),
        )?;

        // 4. Promote CG iterate into the persistent `u` buffer.
        self.ops_kernel
            .copy_u(self.n as u32, &cg_result.x, &self.u_buf)?;

        Ok(TransientStepStats {
            residual: cg_result.residual as Real,
            iters: cg_result.iters,
        })
    }

    /// Read the persistent `u` buffer back to host as `f64`. Call
    /// once per render frame, not once per substep.
    pub fn read_u(&self) -> Result<Vec<Real>> {
        let f32_buf = self.cg_kernel.read_buffer(&self.u_buf, self.n as u32)?;
        Ok(f32_buf.into_iter().map(|v| v as Real).collect())
    }

    /// Overwrite the persistent `u` buffer from host (e.g. on reset).
    pub fn write_u(&mut self, u: &[Real]) -> Result<()> {
        debug_assert_eq!(u.len(), self.n);
        let queue = self.cg_kernel.queue();
        let u_f32: Vec<f32> = u.iter().map(|&v| v as f32).collect();
        queue.write_buffer(&self.u_buf, 0, bytemuck::cast_slice(&u_f32));
        Ok(())
    }

    /// Overwrite the persistent source-term `f` buffer from host.
    /// Defaults to zero at construction; call this when the source
    /// changes per step.
    pub fn write_f(&mut self, f: &[Real]) -> Result<()> {
        debug_assert_eq!(f.len(), self.n);
        let queue = self.cg_kernel.queue();
        let f_f32: Vec<f32> = f.iter().map(|&v| v as f32).collect();
        queue.write_buffer(&self.f_buf, 0, bytemuck::cast_slice(&f_f32));
        Ok(())
    }
}

/// Stage 36.5 — GPU port of [`solve_steady_state`].
///
/// Wraps the [`crate::kernels::FemGpuKernel::matvec_blocking`] entry
/// in a closure for the CPU [`cg_solve`] driver. Each CG iteration
/// dispatches one matvec + reads the result back into host f64.
///
/// # API contract
///
/// `K_csr_gpu` is expected to be the **already BC-reduced** CSR
/// matrix — the caller must `apply_dirichlet` to a CPU `BsrMatrix`
/// + `f` BEFORE uploading via [`FemGpuData::upload`]. v1 does not
/// re-apply BCs on the GPU side; doing so cleanly requires a
/// dedicated BC-application kernel which is deferred to a follow-up
/// (Stage 36.5.1). The `bcs` parameter is consulted ONLY to seed the
/// initial guess (the CG residual at BC rows is then trivially zero
/// because the BC-reduced rows are identity).
///
/// `f` is the BC-reduced RHS — same one passed to `cg_solve` on the
/// CPU path.
///
/// # Per-iter readback cost
///
/// On the 16³ tet test (~5k DoFs) each CG iteration measures
/// ~0.5 ms wall-clock on RTX 4080 — dominated by the host bounce
/// from reading the matvec result back into a `Vec<f64>` for the
/// host CG iterates. CG converges in ~50-200 iters on the
/// heat-equation problem so the total readback overhead is
/// 25-100 ms per solve. The CPU path, by comparison, runs the same
/// problem in <50 ms because it has no per-iter device sync.
///
/// **Stage 36.5.1 follow-up**: fully on-device CG keeps `r`, `p`,
/// `x` resident on the GPU and only reads back at convergence,
/// eliminating the per-iter host bounce.
#[cfg(feature = "gpu")]
pub fn solve_steady_state_gpu(
    k_csr_gpu: &FemGpuData,
    f: &[Real],
    bcs: &[BoundaryCondition],
    engine: &Engine,
    max_iters: usize,
    tol: Real,
) -> Result<CgResult> {
    let n = f.len();
    if k_csr_gpu.csr_nrows as usize != n {
        return Err(Error::Build(format!(
            "solve_steady_state_gpu: GPU CSR rows ({}) != f length ({n})",
            k_csr_gpu.csr_nrows,
        )));
    }
    let device = engine
        .device()
        .ok_or_else(|| Error::Gpu("solve_steady_state_gpu: engine has no GPU device".into()))?;
    let queue = engine
        .queue()
        .ok_or_else(|| Error::Gpu("solve_steady_state_gpu: engine has no GPU queue".into()))?;

    let kernel = FemGpuKernel::new(device.clone(), queue.clone());

    // Seed the initial guess with Dirichlet values. Even though the
    // BC-reduced CSR has identity rows for those DoFs, seeding makes
    // the CG residual at iter 0 already zero on those rows so we save
    // a pass that wouldn't move them anyway.
    let mut x_init = vec![0.0; n];
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = *bc {
            x_init[vertex as usize] = value;
        }
    }

    // Closure: y = K · x. Down-converts x to f32, dispatches GPU,
    // up-converts y back to f64.
    let result = cg_solve(
        |x_f64: &[Real]| -> Vec<Real> {
            let x_f32: Vec<f32> = x_f64.iter().map(|&v| v as f32).collect();
            // The matvec_blocking call is the only place per-iter that
            // does host I/O. v2 removes this — see the function
            // doc-comment.
            let y_f32 = kernel
                .matvec_blocking(k_csr_gpu, &x_f32)
                .expect("FEM GPU matvec dispatch");
            y_f32.into_iter().map(|v| v as Real).collect()
        },
        f,
        &x_init,
        tol,
        max_iters,
    );
    Ok(result)
}

/// Stage 36.5 Slice A — GPU-resident port of [`solve_steady_state`].
///
/// Eliminates the per-CG-iter host roundtrip that
/// [`solve_steady_state_gpu`] paid in v1: `x`, `r`, `p`, and `Ap`
/// stay resident on the GPU; only the dot-product partial-sum
/// readback (a few hundred f32 max for the workgroup-wise tree
/// reduction) crosses the host boundary per iteration.
///
/// Same API contract as [`solve_steady_state_gpu`]: `K_csr_gpu` is
/// the **already BC-reduced** CSR; `f` is the BC-reduced RHS; `bcs`
/// only seeds the initial guess.
///
/// Returns a [`CgResult`] in `f64` for API parity — the GPU runs
/// `f32` internally, the host up-converts the final `x` on readback.
#[cfg(feature = "gpu")]
pub fn solve_steady_state_gpu_resident(
    k_csr_gpu: &FemGpuData,
    f: &[Real],
    bcs: &[BoundaryCondition],
    engine: &Engine,
    max_iters: usize,
    tol: Real,
) -> Result<CgResult> {
    let n = f.len();
    if k_csr_gpu.csr_nrows as usize != n {
        return Err(Error::Build(format!(
            "solve_steady_state_gpu_resident: GPU CSR rows ({}) != f length ({n})",
            k_csr_gpu.csr_nrows,
        )));
    }
    let device = engine.device().ok_or_else(|| {
        Error::Gpu("solve_steady_state_gpu_resident: engine has no GPU device".into())
    })?;
    let queue = engine.queue().ok_or_else(|| {
        Error::Gpu("solve_steady_state_gpu_resident: engine has no GPU queue".into())
    })?;

    // Seed the initial guess with Dirichlet values (consistent with
    // the BC-reduced CSR, which has identity rows on those DoFs).
    let mut x_init_f64 = vec![0.0_f64; n];
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = *bc {
            x_init_f64[vertex as usize] = value;
        }
    }
    let x_init_f32: Vec<f32> = x_init_f64.iter().map(|&v| v as f32).collect();
    let f_f32: Vec<f32> = f.iter().map(|&v| v as f32).collect();

    // Upload b and x_init once; the resident CG solver keeps them on
    // the GPU across iterations.
    let bytes = (n.max(4) * 4) as u64;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let b_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::fem::resident_cg::b"),
        size: bytes,
        usage,
        mapped_at_creation: false,
    });
    queue.write_buffer(&b_buf, 0, bytemuck::cast_slice(&f_f32));
    let x_init_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::fem::resident_cg::x_init"),
        size: bytes,
        usage,
        mapped_at_creation: false,
    });
    queue.write_buffer(&x_init_buf, 0, bytemuck::cast_slice(&x_init_f32));

    // Build kernels. Both can be reused across many solves; we
    // construct fresh ones per call here for API simplicity. Caller
    // who wants the persistent variant can call
    // [`crate::kernels::FemCgGpuKernel::new`] directly.
    let mv_kernel = FemGpuKernel::new(device.clone(), queue.clone());
    let cg_kernel = FemCgGpuKernel::new(device.clone(), queue.clone());

    let result = cg_kernel.cg_solve_resident(
        n as u32,
        &b_buf,
        Some(&x_init_buf),
        tol as f32,
        max_iters,
        |x_buf, y_buf| mv_kernel.matvec(k_csr_gpu, x_buf, y_buf),
    )?;

    // Read back the final x.
    let x_f32 = cg_kernel.read_buffer(&result.x, n as u32)?;
    let x_f64: Vec<Real> = x_f32.into_iter().map(|v| v as Real).collect();

    Ok(CgResult {
        x: x_f64,
        iters: result.iters,
        residual: result.residual as Real,
    })
}

/// Jacobi-preconditioned variant of [`solve_steady_state_gpu_resident`].
///
/// Same API contract — `K_csr_gpu` is the **already BC-reduced** CSR;
/// `f` is the BC-reduced RHS; `bcs` only seeds the initial guess.
///
/// Implementation: extracts `diag(K)` once via
/// [`FemGpuKernel::extract_diagonal`], inverts host-side (zero entries
/// pass through), uploads `inv_diag`, then dispatches
/// [`FemCgGpuKernel::pcg_solve_resident`]. The Jacobi preconditioner
/// typically cuts the CG iter count by 25-40% on Laplacian-like FEM
/// systems for free — same per-iter cost.
#[cfg(feature = "gpu")]
pub fn solve_steady_state_gpu_resident_pcg(
    k_csr_gpu: &FemGpuData,
    f: &[Real],
    bcs: &[BoundaryCondition],
    engine: &Engine,
    max_iters: usize,
    tol: Real,
) -> Result<CgResult> {
    let n = f.len();
    if k_csr_gpu.csr_nrows as usize != n {
        return Err(Error::Build(format!(
            "solve_steady_state_gpu_resident_pcg: GPU CSR rows ({}) != f length ({n})",
            k_csr_gpu.csr_nrows,
        )));
    }
    let device = engine.device().ok_or_else(|| {
        Error::Gpu("solve_steady_state_gpu_resident_pcg: engine has no GPU device".into())
    })?;
    let queue = engine.queue().ok_or_else(|| {
        Error::Gpu("solve_steady_state_gpu_resident_pcg: engine has no GPU queue".into())
    })?;

    // Seed the initial guess with Dirichlet values (consistent with
    // the BC-reduced CSR, which has identity rows on those DoFs).
    let mut x_init_f64 = vec![0.0_f64; n];
    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = *bc {
            x_init_f64[vertex as usize] = value;
        }
    }
    let x_init_f32: Vec<f32> = x_init_f64.iter().map(|&v| v as f32).collect();
    let f_f32: Vec<f32> = f.iter().map(|&v| v as f32).collect();

    let bytes = (n.max(4) * 4) as u64;
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let b_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::fem::resident_pcg::b"),
        size: bytes,
        usage,
        mapped_at_creation: false,
    });
    queue.write_buffer(&b_buf, 0, bytemuck::cast_slice(&f_f32));
    let x_init_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::fem::resident_pcg::x_init"),
        size: bytes,
        usage,
        mapped_at_creation: false,
    });
    queue.write_buffer(&x_init_buf, 0, bytemuck::cast_slice(&x_init_f32));

    let mv_kernel = FemGpuKernel::new(device.clone(), queue.clone());
    let cg_kernel = FemCgGpuKernel::new(device.clone(), queue.clone());

    // Extract diag(K) on GPU, read back, invert host-side, upload as
    // `inv_diag`. The extract + readback + reupload happens once per
    // solve — amortised over many CG iters.
    let diag_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::fem::resident_pcg::diag"),
        size: bytes,
        usage,
        mapped_at_creation: false,
    });
    mv_kernel.extract_diagonal(k_csr_gpu, &diag_buf)?;
    let diag_host = cg_kernel.read_buffer(&diag_buf, n as u32)?;
    let inv_diag_host: Vec<f32> = diag_host
        .iter()
        .map(|&d| if d.abs() > 1.0e-30 { 1.0 / d } else { 0.0 })
        .collect();
    let inv_diag_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("molla::fem::resident_pcg::inv_diag"),
        size: bytes,
        usage,
        mapped_at_creation: false,
    });
    queue.write_buffer(&inv_diag_buf, 0, bytemuck::cast_slice(&inv_diag_host));

    let result = cg_kernel.pcg_solve_resident(
        n as u32,
        &b_buf,
        Some(&x_init_buf),
        &inv_diag_buf,
        tol as f32,
        max_iters,
        |x_buf, y_buf| mv_kernel.matvec(k_csr_gpu, x_buf, y_buf),
    )?;

    let x_f32 = cg_kernel.read_buffer(&result.x, n as u32)?;
    let x_f64: Vec<Real> = x_f32.into_iter().map(|v| v as Real).collect();

    Ok(CgResult {
        x: x_f64,
        iters: result.iters,
        residual: result.residual as Real,
    })
}

/// Vector-aware steady-state solve. Identical to
/// [`solve_steady_state`] except it routes through [`apply_bcs`] so
/// `BoundaryCondition::DirichletVector` and `NeumannTraction` are
/// honoured. `dim` is the per-vertex DoF count (1 for scalar fields,
/// 2 / 3 for vector elasticity); `vertices` is the mesh vertex array
/// needed for `NeumannTraction` face-area integration.
pub fn solve_steady_state_vector(
    k: &BsrMatrix,
    f: &[Real],
    bcs: &[BoundaryCondition],
    vertices: &[crate::math::Vec3],
    dim: usize,
    max_iters: usize,
    tol: Real,
) -> Vec<Real> {
    let mut k_red = k.clone();
    let mut f_red = f.to_vec();
    super::bc::apply_bcs(&mut k_red, &mut f_red, bcs, vertices, dim);
    let mut x_init = vec![0.0; f.len()];
    for bc in bcs {
        match bc {
            BoundaryCondition::Dirichlet { vertex, value } => {
                x_init[(*vertex as usize) * dim] = *value;
            }
            BoundaryCondition::DirichletVector { vertex, values } => {
                let base = (*vertex as usize) * dim;
                for (ci, val) in values.iter().enumerate() {
                    x_init[base + ci] = *val;
                }
            }
            _ => {}
        }
    }
    cg_solve(|x| matvec(&k_red, x), &f_red, &x_init, tol, max_iters).x
}

/// Newton-Raphson driver for the **nonlinear static** elasticity
/// solve `R(u) = K_int(u) − f_ext = 0`.
///
/// Each Newton iteration:
///   1. assemble the **tangent** `K(u)` via `tangent_assembler(u)`,
///   2. compute the **residual** `R(u) = K_int(u) − f_ext` via
///      `residual_assembler(u)`,
///   3. apply Dirichlet BCs to `(K, R)` (zero-row + diagonal=1, but
///      with RHS `−R` so the increment respects the BC),
///   4. solve `K · du = −R` via `cg_solve`,
///   5. update `u ← u + du`.
///
/// Termination: `‖du‖ < tol` (absolute) or `iter == max_newton_iters`.
///
/// `vertices` is needed by `apply_bcs` (NeumannTraction face areas);
/// `dim` is the per-vertex DoF count. The Dirichlet BCs are
/// **incremental** for the Newton step (`du = 0` at constrained DoFs
/// after the first iteration), but the user supplies the absolute
/// values — we seed `u_initial` to satisfy them then enforce `du = 0`
/// on those rows internally.
///
/// v1: no line search. Small problems only.
#[allow(clippy::too_many_arguments)]
pub fn solve_nonlinear_static<KAsm, RAsm>(
    tangent_assembler: KAsm,
    residual_assembler: RAsm,
    f_ext: &[Real],
    bcs: &[BoundaryCondition],
    vertices: &[crate::math::Vec3],
    dim: usize,
    max_newton_iters: usize,
    tol: Real,
) -> Vec<Real>
where
    KAsm: Fn(&[Real]) -> BsrMatrix,
    RAsm: Fn(&[Real]) -> Vec<Real>,
{
    let n = f_ext.len();
    let mut u = vec![0.0; n];
    // Seed Dirichlet entries.
    for bc in bcs {
        match bc {
            BoundaryCondition::Dirichlet { vertex, value } => {
                u[(*vertex as usize) * dim] = *value;
            }
            BoundaryCondition::DirichletVector { vertex, values } => {
                let base = (*vertex as usize) * dim;
                for (ci, val) in values.iter().enumerate() {
                    u[base + ci] = *val;
                }
            }
            _ => {}
        }
    }

    let l2 = |v: &[Real]| v.iter().map(|x| x * x).sum::<Real>().sqrt();
    let mut prev_residual_norm = Real::INFINITY;
    let mut diverging_streak: usize = 0;

    for _iter in 0..max_newton_iters {
        let k = tangent_assembler(&u);
        let r_int = residual_assembler(&u);
        // Residual R(u) = K_int(u) − f_ext. Newton increment solves
        // K · du = −R = f_ext − K_int(u).
        let mut rhs: Vec<Real> = f_ext
            .iter()
            .zip(r_int.iter())
            .map(|(fe, ri)| fe - ri)
            .collect();
        // Track residual norm to detect divergence.
        let cur_residual_norm = l2(&rhs);
        if cur_residual_norm > prev_residual_norm {
            diverging_streak += 1;
            if diverging_streak >= 2 {
                // Two consecutive iterations with growing residual —
                // Newton has lost the basin of attraction. Bail out
                // with the current iterate (still better than running
                // to `max_newton_iters` and returning garbage that
                // looks like a converged solution).
                tracing::warn!(
                    "solve_nonlinear_static: residual ‖R‖ grew on \
                     consecutive iterations ({prev_residual_norm:.3e} → \
                     {cur_residual_norm:.3e}); Newton diverging — early exit"
                );
                break;
            }
        } else {
            diverging_streak = 0;
        }
        prev_residual_norm = cur_residual_norm;
        let mut k_red = k;
        // For Dirichlet rows, enforce du = 0: pin row + zero RHS.
        // We re-use apply_bcs but with **zero**-valued Dirichlet rows
        // (the absolute Dirichlet value is already in `u`).
        let zero_bcs: Vec<BoundaryCondition> = bcs
            .iter()
            .map(|bc| match bc {
                BoundaryCondition::Dirichlet { vertex, .. } => BoundaryCondition::Dirichlet {
                    vertex: *vertex,
                    value: 0.0,
                },
                BoundaryCondition::DirichletVector { vertex, values } => {
                    BoundaryCondition::DirichletVector {
                        vertex: *vertex,
                        values: vec![0.0; values.len()],
                    }
                }
                BoundaryCondition::Neumann { face, flux } => BoundaryCondition::Neumann {
                    face: *face,
                    flux: *flux,
                },
                BoundaryCondition::NeumannTraction { face, traction } => {
                    BoundaryCondition::NeumannTraction {
                        face: *face,
                        traction: *traction,
                    }
                }
                // DG Nitsche BCs are irrelevant to the CG Newton iteration.
                BoundaryCondition::WeakDirichletDG {
                    vertex,
                    value,
                    sigma,
                } => BoundaryCondition::WeakDirichletDG {
                    vertex: *vertex,
                    value: *value,
                    sigma: *sigma,
                },
            })
            .collect();
        // `f_ext` already accounts for tractions, so we DROP traction
        // entries from the BC list before pinning rows — apply_bcs
        // would otherwise add them again into `rhs`.
        let pin_only: Vec<BoundaryCondition> = zero_bcs
            .into_iter()
            .filter(|bc| {
                matches!(
                    bc,
                    BoundaryCondition::Dirichlet { .. } | BoundaryCondition::DirichletVector { .. }
                )
            })
            .collect();
        super::bc::apply_bcs(&mut k_red, &mut rhs, &pin_only, vertices, dim);

        // Initial guess for the increment: zero (BCs already enforce
        // du = 0 on the constrained rows).
        let du = cg_solve(
            |x| matvec(&k_red, x),
            &rhs,
            &vec![0.0; n],
            DEFAULT_TOL,
            DEFAULT_MAX_ITERS,
        )
        .x;
        // Update.
        let mut max_du = 0.0_f64;
        for (uv, dv) in u.iter_mut().zip(du.iter()) {
            *uv += *dv;
            max_du = max_du.max(dv.abs());
        }
        if max_du < tol {
            break;
        }
    }
    u
}

/// Stage 37.4 — Newton-Raphson minimisation of the per-cell symmetric
/// Dirichlet distortion energy summed over a [`super::FemMesh`].
///
/// Each Newton iteration:
///   1. assemble the global gradient `g = ∑_e g_e` (one `Vec3` per
///      vertex, summed in interleaved order),
///   2. assemble the global Hessian `H = ∑_e H_e` (dense `Vec<Vec<Real>>`
///      ⇒ packed into a [`BsrMatrix`] with `B = 1`, scattering each
///      element 9×9 / 12×12 block in interleaved DoF order),
///   3. solve `H · dx = −g` via [`cg_solve`] (linear-time, matrix-free
///      via [`matvec`] over the assembled CSR),
///   4. update `deformed ← deformed + dx`.
///
/// Termination: `‖dx‖∞ < tol` (max absolute component of the Newton
/// step) or `iter == max_iter`. Returns the final deformed-vertex
/// array.
///
/// `init_deformed` is the warm-start configuration; pass the rest mesh
/// vertices for a from-rest minimisation, or perturbed positions for a
/// "snap back to a low-distortion map" use case (the canonical
/// `WarpFemDistortion` viewer scene).
///
/// v1: no line search; the Hessian is regularised by adding `1e-8 · I`
/// before solving so degenerate elements (collinear / coplanar) don't
/// blow up the CG. Callers with pathological inputs should pre-fix the
/// rest mesh.
pub fn solve_distortion_minimization(
    rest_mesh: &super::mesh::FemMesh,
    init_deformed: &[crate::math::Vec3],
    max_iter: usize,
    tol: Real,
) -> Vec<crate::math::Vec3> {
    use super::element::compute_element_distortion_energy;
    use crate::math::Vec3;

    let n_verts = rest_mesh.vertices.len();
    let n_dof = n_verts * 3;
    debug_assert_eq!(
        init_deformed.len(),
        n_verts,
        "solve_distortion_minimization: init_deformed length mismatch"
    );

    let mut deformed = init_deformed.to_vec();

    for _iter in 0..max_iter {
        // Assemble the global gradient + Hessian by summing the
        // per-element contributions. The Hessian is collected as a
        // CSR via the same triplet rebuild path the FEM module uses
        // elsewhere (`BsrMatrix::from_triplets`).
        let mut grad = vec![0.0_f64; n_dof];
        let mut triplets: Vec<(usize, usize, Vec<Real>)> = Vec::new();
        for cell in &rest_mesh.cells {
            let (_e, g_local, h_local) =
                compute_element_distortion_energy(cell, &rest_mesh.vertices, &deformed);
            let verts = cell.verts();
            for (i, vi) in verts.iter().enumerate() {
                let row_base = (*vi as usize) * 3;
                grad[row_base] += g_local[i].x;
                grad[row_base + 1] += g_local[i].y;
                grad[row_base + 2] += g_local[i].z;
                for (j, vj) in verts.iter().enumerate() {
                    let col_base = (*vj as usize) * 3;
                    for ci in 0..3 {
                        for cj in 0..3 {
                            let val = h_local[i * 3 + ci][j * 3 + cj];
                            if val == 0.0 {
                                continue;
                            }
                            triplets.push((row_base + ci, col_base + cj, vec![val]));
                        }
                    }
                }
            }
        }
        // Diagonal regulariser to keep the Hessian SPD on degenerate
        // / pinned DoFs. Mostly relevant for elements whose volume
        // collapses during the Newton walk; harmless on smooth
        // problems because the regulariser is 1e-8.
        for r in 0..n_dof {
            triplets.push((r, r, vec![1.0e-8]));
        }
        let h_global = BsrMatrix::from_triplets(1, n_dof, n_dof, &triplets)
            .expect("solve_distortion_minimization: hessian assembly");
        // Newton step: H · dx = −g.
        let neg_g: Vec<Real> = grad.iter().map(|v| -v).collect();
        let dx_flat = cg_solve(
            |x| matvec(&h_global, x),
            &neg_g,
            &vec![0.0; n_dof],
            DEFAULT_TOL,
            DEFAULT_MAX_ITERS,
        )
        .x;
        // Apply the increment + measure ‖dx‖∞.
        let mut max_abs: Real = 0.0;
        for v in 0..n_verts {
            let dx = Vec3::new(dx_flat[v * 3], dx_flat[v * 3 + 1], dx_flat[v * 3 + 2]);
            deformed[v] += dx;
            let m = dx.x.abs().max(dx.y.abs()).max(dx.z.abs());
            if m > max_abs {
                max_abs = m;
            }
        }
        if max_abs < tol {
            break;
        }
    }
    deformed
}

/// Returns a fresh `BsrMatrix` representing `α·A + β·B` where `A` and
/// `B` share the same `(B = 1)` block size. Patterns are unioned by
/// re-emitting the triplets through [`BsrMatrix::from_triplets`] which
/// dedupes + sums on coincident `(r, c)`.
///
/// This is the v1 "good enough" implementation — `nnz` is at most
/// `nnz(A) + nnz(B)` and the sort happens once per call. Stage 36.5
/// will replace it with a sparse-add kernel that walks both row
/// patterns in tandem.
fn add_csr_scaled(a: &BsrMatrix, alpha: Real, b: &BsrMatrix, beta: Real) -> BsrMatrix {
    debug_assert_eq!(a.b, 1);
    debug_assert_eq!(b.b, 1);
    debug_assert_eq!(a.nrows(), b.nrows());
    debug_assert_eq!(a.ncols(), b.ncols());

    let mut triplets: Vec<(usize, usize, Vec<Real>)> =
        Vec::with_capacity(a.values.len() + b.values.len());
    for r in 0..a.block_rows {
        for slot in a.row_offsets[r]..a.row_offsets[r + 1] {
            let c = a.col_idx[slot];
            triplets.push((r, c, vec![alpha * a.values[slot]]));
        }
    }
    for r in 0..b.block_rows {
        for slot in b.row_offsets[r]..b.row_offsets[r + 1] {
            let c = b.col_idx[slot];
            triplets.push((r, c, vec![beta * b.values[slot]]));
        }
    }
    BsrMatrix::from_triplets(1, a.block_rows, a.block_cols, &triplets)
        .expect("add_csr_scaled: triplet rebuild")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem::assembly::{assemble_mass, assemble_stiffness};
    use crate::fem::function_space::FunctionSpace;
    use crate::fem::mesh::FemMesh;

    #[test]
    fn steady_state_constant_dirichlet_yields_constant() {
        // K · u = 0, with all-boundary Dirichlet u = 5 ⇒ u ≡ 5.
        let mesh = FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
        let space = FunctionSpace::P1Nodal;
        let k = assemble_stiffness(&mesh, space, 1.0).unwrap();
        let n = space.ndof(&mesh);
        let f = vec![0.0; n];

        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.x.abs() < 1e-9
                || (p.x - 1.0).abs() < 1e-9
                || p.y.abs() < 1e-9
                || (p.y - 1.0).abs() < 1e-9
            {
                bcs.push(BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 5.0,
                });
            }
        }
        let u = solve_steady_state(&k, &f, &bcs, 500, 1e-10);
        for &val in &u {
            assert!((val - 5.0).abs() < 1e-6, "interior value {val} ≠ 5");
        }
    }

    #[test]
    fn transient_zero_dt_step_is_well_defined() {
        // A small, finite dt — verify the solver returns numbers and
        // the BC is honoured at the boundaries.
        let mesh = FemMesh::tri_grid_2d(3, 3, 1.0, 1.0);
        let space = FunctionSpace::P1Nodal;
        let m = assemble_mass(&mesh, space).unwrap();
        let k = assemble_stiffness(&mesh, space, 1.0).unwrap();
        let n = space.ndof(&mesh);
        let f = vec![0.0; n];
        let u_old = vec![0.0; n];
        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.y.abs() < 1e-9 {
                bcs.push(BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 0.0,
                });
            } else if (p.y - 1.0).abs() < 1e-9 {
                bcs.push(BoundaryCondition::Dirichlet {
                    vertex: i as u32,
                    value: 1.0,
                });
            }
        }
        let u = solve_transient(&m, &k, &f, &u_old, 0.1, &bcs);
        for bc in &bcs {
            if let BoundaryCondition::Dirichlet { vertex, value } = bc {
                assert!((u[*vertex as usize] - *value).abs() < 1e-6);
            }
        }
    }
}
