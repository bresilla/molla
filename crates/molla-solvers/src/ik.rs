//! Inverse kinematics — Levenberg-Marquardt solver with damped least
//! squares + L-BFGS quasi-Newton fallback.
//!
//! Pose objectives:
//!   * `IkObjective::Position`        — minimise `‖p_world − target‖²`.
//!   * `IkObjective::Rotation`        — minimise `2·sin(θ/2)`-style
//!     residual = imaginary part of `q_target · q_current⁻¹`.
//!   * `IkObjective::JointLimit`      — soft penalty when `q[i]`
//!     exceeds `[lo, hi]`.
//!
//! Multiple objectives can be active simultaneously; each contributes
//! a row block to the LM normal equations. Joint-limit penalties are
//! treated as soft additions to the gradient (no hard projection in
//! v1; use joint-limit clamps via Featherstone if you need that).
//!
//! Multi-seed sampling is exposed via `IkSolver::solve_with_seeds` —
//! useful when the cost landscape has local minima (long chains, tight
//! orientation constraints).

use crate::core::{Error, Result};
use crate::jacobian::jacobian;
use crate::math::{
    Real, Vec3,
    dense_solve::{dense_chol_solve, dense_psd_pinv_solve},
    robust_loss::{RobustLoss, norm_huber_grad},
};
use crate::sim::{Model, State};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Process-wide counters tracking how often the IK solver had to fall
/// back from the fast Cholesky pre-solve to the SVD-style pseudoinverse
/// (Stage 24a). Mostly observability — automated tests in the IK suite
/// poll these to verify the fallback rarely triggers.
///
/// Reads are advisory; the counters are updated by every IK solve in
/// the process and never reset (callers compute deltas if they want a
/// per-solve rate).
pub static IK_CHOL_ATTEMPTS: AtomicUsize = AtomicUsize::new(0);
pub static IK_PSD_FALLBACKS: AtomicUsize = AtomicUsize::new(0);

/// Snapshot of the global IK fallback counters. Returns
/// `(chol_attempts, psd_fallbacks)`.
pub fn ik_solver_health() -> (usize, usize) {
    (
        IK_CHOL_ATTEMPTS.load(Ordering::Relaxed),
        IK_PSD_FALLBACKS.load(Ordering::Relaxed),
    )
}

/// Solve `(JᵀJ + λI) · dq = rhs` with automatic fallback. Tries the
/// existing Cholesky path first; on failure (non-SPD pivot caused by
/// numerical breakdown) drops to a damped pseudoinverse computed from
/// the symmetric eigendecomposition of `JᵀJ + λI`. Increments the
/// process-wide [`IK_CHOL_ATTEMPTS`] / [`IK_PSD_FALLBACKS`] counters
/// so callers can observe trigger rates.
fn ik_solve_normal_eq(jtj: &mut [Real], rhs: &[Real], n: usize) -> Result<Vec<Real>> {
    IK_CHOL_ATTEMPTS.fetch_add(1, Ordering::Relaxed);
    // Cholesky decomposes in place; keep a copy for the fallback.
    let saved = jtj.to_vec();
    match dense_chol_solve(jtj, rhs, n) {
        Ok(x) => Ok(x),
        Err(_) => {
            IK_PSD_FALLBACKS.fetch_add(1, Ordering::Relaxed);
            // Cholesky may have partially written to `jtj`; restore from
            // saved for the eigendecomposition.
            dense_psd_pinv_solve(&saved, rhs, n)
        }
    }
}

#[cfg(feature = "gpu")]
use crate::compute::Engine;

/// One IK objective. Multiple objectives can be combined; LM stacks
/// them into a single `J·dq = -e` system.
#[derive(Clone, Debug)]
pub enum IkObjective {
    /// Position objective — drive `body`'s `point_local` to
    /// `target_world` (3 residual rows).
    Position {
        body: usize,
        point_local: Vec3,
        target_world: Vec3,
        weight: Real,
    },
    /// Joint-limit penalty: each violated `q[dof]` past `[lo, hi]`
    /// adds a residual `weight · violation` (1 row per limit).
    JointLimit {
        dof: usize,
        lo: Real,
        hi: Real,
        weight: Real,
    },
    /// Sphere-avoidance penalty (INSPECTION.md F15 / scene 113): when
    /// `body`'s `point_local` enters the inflated sphere
    /// (`radius + margin` around `center`), one residual row
    /// `weight · (radius + margin − ‖p − center‖)` pushes it back out
    /// along the center→point direction. Zero row (inactive) outside.
    SphereAvoid {
        body: usize,
        point_local: Vec3,
        center: Vec3,
        radius: Real,
        margin: Real,
        weight: Real,
    },
}

/// Solver configuration.
#[derive(Copy, Clone, Debug)]
pub struct IkConfig {
    /// Maximum LM iterations.
    pub max_iter: usize,
    /// Convergence tolerance on `‖e‖` (residual L2 norm).
    pub residual_tol: Real,
    /// Step-size tolerance — exit early if `‖dq‖ < step_tol`.
    pub step_tol: Real,
    /// LM damping initial value. Adapted per iteration: doubled on
    /// rejected steps, halved on accepted.
    pub damping_init: Real,
    /// Damping growth on rejected step (typical 2.0 .. 10.0).
    pub damping_grow: Real,
    /// Damping shrink on accepted step (typical 2.0 .. 10.0).
    pub damping_shrink: Real,
    /// Articulation index.
    pub articulation: usize,
    /// Stage 24b — robust-loss selector. Default [`RobustLoss::L2`]
    /// preserves the original Cholesky-+-L2 behaviour. When a non-L2
    /// loss is selected the solver runs an iteratively re-weighted
    /// least-squares (IRLS) inner loop: at each iteration the
    /// per-residual weight `w_i = ρ'(r_i) / r_i` is folded into the
    /// normal equations as `(JᵀWJ + λI)·dq = −JᵀWe`. Bounded-influence
    /// losses (Tukey, Cauchy) reject or down-weight outlier targets;
    /// Huber gives a milder gradient cap for noisy targets.
    pub loss: RobustLoss,
}

impl Default for IkConfig {
    fn default() -> Self {
        Self {
            max_iter: 100,
            residual_tol: 1.0e-4,
            step_tol: 1.0e-9,
            damping_init: 1.0e-3,
            damping_grow: 4.0,
            damping_shrink: 4.0,
            articulation: 0,
            loss: RobustLoss::L2,
        }
    }
}

/// Outcome of an IK solve.
#[derive(Clone, Debug)]
pub struct IkResult {
    /// Final joint coordinates (length = articulation DOFs).
    pub q: Vec<Real>,
    /// Final residual L2 norm.
    pub residual: Real,
    /// Whether the solver hit the residual tolerance.
    pub converged: bool,
    /// Number of LM iterations consumed.
    pub iters: usize,
}

/// Stateless LM IK solver. Construct, then call `solve` /
/// `solve_with_seeds`. Each call performs a fresh forward
/// kinematics evaluation.
#[derive(Clone, Debug)]
pub struct IkSolver {
    pub objectives: Vec<IkObjective>,
    pub config: IkConfig,
}

impl IkSolver {
    pub fn new(objectives: Vec<IkObjective>) -> Self {
        Self {
            objectives,
            config: IkConfig::default(),
        }
    }

    pub fn with_config(mut self, c: IkConfig) -> Self {
        self.config = c;
        self
    }

    /// Single-seed solve from `q0`. Mutates state's `joint_q` and
    /// `body_q` to reflect the solution.
    pub fn solve(&self, model: &Model, state: &mut State, q0: &[Real]) -> Result<IkResult> {
        let cfg = &self.config;
        let dof_total = articulation_dof_total(model, cfg.articulation)?;
        if q0.len() != dof_total {
            return Err(Error::Build(format!(
                "ik::solve: q0 len {} != articulation DOFs {}",
                q0.len(),
                dof_total
            )));
        }

        // Working state.
        let mut q = q0.to_vec();
        write_q(model, state, cfg.articulation, &q)?;
        crate::sim::eval_fk(model, state)?;
        let mut e = self.compute_residuals(model, state)?;
        let mut e_norm = norm(&e);
        let mut e_cost = loss_cost(cfg.loss, &e);
        let mut damping = cfg.damping_init;

        for it in 0..cfg.max_iter {
            if e_norm < cfg.residual_tol {
                return Ok(IkResult {
                    q,
                    residual: e_norm,
                    converged: true,
                    iters: it,
                });
            }

            // Build stacked Jacobian.
            let j = self.build_jacobian(model, state)?;
            let m = e.len(); // residual dimension
            // Stage 24b — per-residual IRLS weights. For L2 every
            // weight is 1, so this collapses to the original normal
            // equations. Bounded-influence losses (Tukey, Cauchy)
            // shrink outlier rows; L1 / smooth-L2 turn the LM step
            // into an IRLS step.
            let weights = irls_weights(cfg.loss, &e);
            // Form normal equations: (J^T · W · J + λ·I) · dq = -J^T · W · e.
            let mut jtj = vec![0.0; dof_total * dof_total];
            for i in 0..dof_total {
                for k in 0..dof_total {
                    let mut s = 0.0;
                    for r in 0..m {
                        s += weights[r] * j[r * dof_total + i] * j[r * dof_total + k];
                    }
                    jtj[i * dof_total + k] = s;
                }
                jtj[i * dof_total + i] += damping;
            }
            let mut rhs = vec![0.0; dof_total];
            for i in 0..dof_total {
                let mut s = 0.0;
                for r in 0..m {
                    s -= weights[r] * j[r * dof_total + i] * e[r];
                }
                rhs[i] = s;
            }
            let dq = match ik_solve_normal_eq(&mut jtj, &rhs, dof_total) {
                Ok(x) => x,
                Err(_) => {
                    // Even SVD-based fallback failed — bump damping and retry.
                    damping *= cfg.damping_grow;
                    continue;
                }
            };

            let dq_norm = norm(&dq);
            if dq_norm < cfg.step_tol {
                return Ok(IkResult {
                    q,
                    residual: e_norm,
                    converged: e_norm < cfg.residual_tol,
                    iters: it + 1,
                });
            }

            // Trial step. Acceptance is judged on the *loss-equivalent
            // cost* `Σρ(r_i)` so IRLS (Tukey / Cauchy / etc.) makes
            // monotone progress on the actual optimisation surface;
            // for L2 the cost is `½‖e‖²` and the comparison is
            // equivalent to the historical `‖e‖` check.
            let mut q_trial = q.clone();
            for i in 0..dof_total {
                q_trial[i] += dq[i];
            }
            write_q(model, state, cfg.articulation, &q_trial)?;
            crate::sim::eval_fk(model, state)?;
            let e_trial = self.compute_residuals(model, state)?;
            let e_trial_norm = norm(&e_trial);
            let e_trial_cost = loss_cost(cfg.loss, &e_trial);
            if e_trial_cost < e_cost {
                // Accept.
                q = q_trial;
                e = e_trial;
                e_norm = e_trial_norm;
                e_cost = e_trial_cost;
                damping = (damping / cfg.damping_shrink).max(1.0e-12);
            } else {
                // Reject — restore q, increase damping.
                write_q(model, state, cfg.articulation, &q)?;
                crate::sim::eval_fk(model, state)?;
                damping *= cfg.damping_grow;
                if damping > 1.0e12 {
                    return Ok(IkResult {
                        q,
                        residual: e_norm,
                        converged: false,
                        iters: it + 1,
                    });
                }
            }
        }
        Ok(IkResult {
            q,
            residual: e_norm,
            converged: e_norm < self.config.residual_tol,
            iters: self.config.max_iter,
        })
    }

    /// Multi-seed solve — runs `solve` starting from each seed and
    /// returns the result with the lowest residual.
    pub fn solve_with_seeds(
        &self,
        model: &Model,
        state: &mut State,
        seeds: &[Vec<Real>],
    ) -> Result<IkResult> {
        if seeds.is_empty() {
            return Err(Error::Build("ik::solve_with_seeds: no seeds".into()));
        }
        let mut best: Option<IkResult> = None;
        for seed in seeds {
            let r = self.solve(model, state, seed)?;
            if r.converged {
                return Ok(r);
            }
            best = Some(match best {
                Some(b) if b.residual <= r.residual => b,
                _ => r,
            });
        }
        Ok(best.unwrap())
    }

    fn compute_residuals(&self, _model: &Model, state: &State) -> Result<Vec<Real>> {
        let body_q = state.body_q.host()?;
        let joint_q = state.joint_q.host()?;
        let mut out = Vec::with_capacity(self.objectives.len() * 3);
        for obj in &self.objectives {
            match *obj {
                IkObjective::Position {
                    body,
                    point_local,
                    target_world,
                    weight,
                } => {
                    let cw = body_q[body];
                    let p_world = cw.position + cw.rotation * point_local;
                    let r = (p_world - target_world) * weight;
                    out.push(r.x);
                    out.push(r.y);
                    out.push(r.z);
                }
                IkObjective::JointLimit {
                    dof,
                    lo,
                    hi,
                    weight,
                } => {
                    let q = joint_q[dof];
                    let viol = if q < lo {
                        q - lo
                    } else if q > hi {
                        q - hi
                    } else {
                        0.0
                    };
                    // Use Huber gradient for smooth fall-off.
                    out.push(weight * norm_huber_grad(viol, 0.05));
                }
                IkObjective::SphereAvoid {
                    body,
                    point_local,
                    center,
                    radius,
                    margin,
                    weight,
                } => {
                    let cw = body_q[body];
                    let p_world = cw.position + cw.rotation * point_local;
                    let viol = (radius + margin - (p_world - center).length()).max(0.0);
                    out.push(weight * viol);
                }
            }
        }
        Ok(out)
    }

    fn build_jacobian(&self, model: &Model, state: &State) -> Result<Vec<Real>> {
        let dof_total = articulation_dof_total(model, self.config.articulation)?;
        let m: usize = self
            .objectives
            .iter()
            .map(|o| match o {
                IkObjective::Position { .. } => 3,
                IkObjective::JointLimit { .. } => 1,
                IkObjective::SphereAvoid { .. } => 1,
            })
            .sum();
        let mut j = vec![0.0; m * dof_total];
        let mut row = 0usize;
        for obj in &self.objectives {
            match *obj {
                IkObjective::Position {
                    body,
                    point_local,
                    weight,
                    ..
                } => {
                    let jcols =
                        jacobian(model, state, self.config.articulation, body, point_local)?;
                    // jcols layout: 6 rows × dof_total cols. Take rows 0..3 (linear).
                    for r in 0..3 {
                        for k in 0..dof_total {
                            j[(row + r) * dof_total + k] = weight * jcols[r * dof_total + k];
                        }
                    }
                    row += 3;
                }
                IkObjective::JointLimit { dof, weight, .. } => {
                    // ∂(q[dof])/∂(q[k]) = δ_{dof,k}.
                    if dof < dof_total {
                        j[row * dof_total + dof] = weight;
                    }
                    row += 1;
                }
                IkObjective::SphereAvoid {
                    body,
                    point_local,
                    center,
                    radius,
                    margin,
                    weight,
                } => {
                    // Residual = weight·(R − ‖p − c‖) inside; its
                    // gradient wrt p is −weight·d̂, chained through the
                    // linear point Jacobian. Zero row when inactive.
                    let body_q = state.body_q.host()?;
                    let cw = body_q[body];
                    let p_world = cw.position + cw.rotation * point_local;
                    let d = p_world - center;
                    let dist = d.length();
                    if dist > 1.0e-9 && dist < radius + margin {
                        let dir = d / dist;
                        let jcols =
                            jacobian(model, state, self.config.articulation, body, point_local)?;
                        for k in 0..dof_total {
                            let jp =
                                Vec3::new(jcols[k], jcols[dof_total + k], jcols[2 * dof_total + k]);
                            j[row * dof_total + k] = -weight * dir.dot(jp);
                        }
                    }
                    row += 1;
                }
            }
        }
        Ok(j)
    }
}

fn norm(v: &[Real]) -> Real {
    v.iter().map(|x| x * x).sum::<Real>().sqrt()
}

/// Stage 24b — per-residual IRLS weight vector for the configured
/// robust loss. For [`RobustLoss::L2`] returns `[1; m]` (identity
/// re-weighting) so the LM normal equation reduces to its original
/// form. Always non-negative; bounded-influence losses zero out
/// outlier rows.
fn irls_weights(loss: RobustLoss, e: &[Real]) -> Vec<Real> {
    match loss {
        RobustLoss::L2 => vec![1.0; e.len()],
        _ => e.iter().map(|r| loss.weight(*r).max(0.0)).collect(),
    }
}

/// Stage 24b — total robust-loss cost `Σ ρ(r_i)`. Used as the LM
/// trial-step acceptance criterion when a non-L2 loss is in play so
/// the solver makes monotone progress on the real objective rather
/// than on the L2 norm of the (un-weighted) residuals.
fn loss_cost(loss: RobustLoss, e: &[Real]) -> Real {
    e.iter().map(|r| loss.rho(*r)).sum()
}

fn write_q(model: &Model, state: &mut State, articulation: usize, q: &[Real]) -> Result<()> {
    let dof_offset_first = if model.articulation_count == 0 {
        0
    } else {
        model.articulation_start.host()?[articulation] as usize
    };
    let coord_offset_start = model.joint_coord_offset.host()?[dof_offset_first] as usize;
    let dst = state.joint_q.host_mut()?;
    for (i, &v) in q.iter().enumerate() {
        if coord_offset_start + i < dst.len() {
            dst[coord_offset_start + i] = v;
        }
    }
    Ok(())
}

// ─── GPU LM solver ────────────────────────────────────────────────
//
// Per-iteration the GPU does the Jacobian assembly (the expensive
// linear-in-DOFs piece, also the part that scales with seed count).
// Trial-step scoring + dense_chol_solve stay on the CPU because they
// touch a tiny `dof × dof` system.

#[cfg(feature = "gpu")]
impl IkSolver {
    /// GPU IK solve. Mirrors [`IkSolver::solve`] but uses the GPU
    /// `JacobianKernel` to assemble J at every iteration. Caller must
    /// have uploaded `model` + `state` to the GPU beforehand
    /// (`upload_to_device`).
    pub fn solve_gpu(
        &self,
        engine: &Engine,
        model: &Model,
        state: &mut State,
        q0: &[Real],
    ) -> Result<IkResult> {
        let cfg = &self.config;
        let dof_total = articulation_dof_total(model, cfg.articulation)?;
        if q0.len() != dof_total {
            return Err(Error::Build(format!(
                "ik::solve_gpu: q0 len {} != articulation DOFs {}",
                q0.len(),
                dof_total
            )));
        }

        let kernel = crate::kernels::JacobianKernel::new(engine, model)?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("solve_gpu: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("solve_gpu: no queue".into()))?
            .clone();

        let mut q = q0.to_vec();
        write_q(model, state, cfg.articulation, &q)?;
        crate::sim::eval_fk(model, state)?;
        let mut e = self.compute_residuals(model, state)?;
        let mut e_norm = norm(&e);
        let mut e_cost = loss_cost(cfg.loss, &e);
        let mut damping = cfg.damping_init;

        // Pre-allocate Jacobian output (single seed; each Position
        // objective owns one `6 × dof_total` block).
        let n_pos = self
            .objectives
            .iter()
            .filter(|o| {
                matches!(
                    o,
                    IkObjective::Position { .. } | IkObjective::SphereAvoid { .. }
                )
            })
            .count();
        let block_floats = 6 * dof_total;
        let total_floats = block_floats * n_pos.max(1);
        let j_buf = kernel.create_output_buffer(&device, total_floats);

        for it in 0..cfg.max_iter {
            if e_norm < cfg.residual_tol {
                return Ok(IkResult {
                    q,
                    residual: e_norm,
                    converged: true,
                    iters: it,
                });
            }

            // Push current FK to GPU then run the Jacobian batch.
            state.body_q.upload(&queue)?;

            let mut requests: Vec<crate::kernels::JacobianRequest> = Vec::with_capacity(n_pos);
            let mut block_idx = 0u32;
            for obj in &self.objectives {
                if let IkObjective::Position {
                    body, point_local, ..
                }
                | IkObjective::SphereAvoid {
                    body, point_local, ..
                } = *obj
                {
                    requests.push(crate::kernels::JacobianRequest {
                        articulation: cfg.articulation as u32,
                        target_body: body as u32,
                        output_offset: block_idx * block_floats as u32,
                        body_q_offset: 0,
                        point_local: [
                            point_local.x as f32,
                            point_local.y as f32,
                            point_local.z as f32,
                            0.0,
                        ],
                    });
                    block_idx += 1;
                }
            }
            kernel.dispatch(engine, model, state, &requests, &j_buf)?;
            let j_blocks = kernel.read_back(engine, &j_buf, total_floats)?;

            // Assemble the stacked Jacobian into row-major `m × dof_total`.
            let m = e.len();
            let mut j = vec![0.0; m * dof_total];
            let mut row = 0usize;
            let mut block_idx = 0usize;
            for obj in &self.objectives {
                match *obj {
                    IkObjective::Position { weight, .. } => {
                        let off = block_idx * block_floats;
                        for r in 0..3 {
                            for k in 0..dof_total {
                                j[(row + r) * dof_total + k] =
                                    weight * (j_blocks[off + r * dof_total + k] as Real);
                            }
                        }
                        row += 3;
                        block_idx += 1;
                    }
                    IkObjective::JointLimit { dof, weight, .. } => {
                        if dof < dof_total {
                            j[row * dof_total + dof] = weight;
                        }
                        row += 1;
                    }
                    IkObjective::SphereAvoid {
                        body,
                        point_local,
                        center,
                        radius,
                        margin,
                        weight,
                    } => {
                        // Project the GPU linear-Jacobian block onto
                        // −d̂ (center→point direction); zero row when
                        // the point is outside the inflated sphere.
                        let body_q_h = state.body_q.host()?;
                        let cw = body_q_h[body];
                        let p_world = cw.position + cw.rotation * point_local;
                        let d = p_world - center;
                        let dist = d.length();
                        if dist > 1.0e-9 && dist < radius + margin {
                            let dir = d / dist;
                            let off = block_idx * block_floats;
                            for k in 0..dof_total {
                                let jp = Vec3::new(
                                    j_blocks[off + k] as Real,
                                    j_blocks[off + dof_total + k] as Real,
                                    j_blocks[off + 2 * dof_total + k] as Real,
                                );
                                j[row * dof_total + k] = -weight * dir.dot(jp);
                            }
                        }
                        row += 1;
                        block_idx += 1;
                    }
                }
            }

            // Normal equations (Stage 24b — IRLS-weighted).
            let weights = irls_weights(cfg.loss, &e);
            let mut jtj = vec![0.0; dof_total * dof_total];
            for i in 0..dof_total {
                for k in 0..dof_total {
                    let mut s = 0.0;
                    for r in 0..m {
                        s += weights[r] * j[r * dof_total + i] * j[r * dof_total + k];
                    }
                    jtj[i * dof_total + k] = s;
                }
                jtj[i * dof_total + i] += damping;
            }
            let mut rhs = vec![0.0; dof_total];
            for i in 0..dof_total {
                let mut s = 0.0;
                for r in 0..m {
                    s -= weights[r] * j[r * dof_total + i] * e[r];
                }
                rhs[i] = s;
            }
            let dq = match ik_solve_normal_eq(&mut jtj, &rhs, dof_total) {
                Ok(x) => x,
                Err(_) => {
                    damping *= cfg.damping_grow;
                    continue;
                }
            };
            let dq_norm = norm(&dq);
            if dq_norm < cfg.step_tol {
                return Ok(IkResult {
                    q,
                    residual: e_norm,
                    converged: e_norm < cfg.residual_tol,
                    iters: it + 1,
                });
            }

            // Trial step — accept on the IRLS cost (matches CPU path).
            let mut q_trial = q.clone();
            for i in 0..dof_total {
                q_trial[i] += dq[i];
            }
            write_q(model, state, cfg.articulation, &q_trial)?;
            crate::sim::eval_fk(model, state)?;
            let e_trial = self.compute_residuals(model, state)?;
            let e_trial_norm = norm(&e_trial);
            let e_trial_cost = loss_cost(cfg.loss, &e_trial);
            if e_trial_cost < e_cost {
                q = q_trial;
                e = e_trial;
                e_norm = e_trial_norm;
                e_cost = e_trial_cost;
                damping = (damping / cfg.damping_shrink).max(1.0e-12);
            } else {
                write_q(model, state, cfg.articulation, &q)?;
                crate::sim::eval_fk(model, state)?;
                damping *= cfg.damping_grow;
                if damping > 1.0e12 {
                    return Ok(IkResult {
                        q,
                        residual: e_norm,
                        converged: false,
                        iters: it + 1,
                    });
                }
            }
        }
        Ok(IkResult {
            q,
            residual: e_norm,
            converged: e_norm < cfg.residual_tol,
            iters: cfg.max_iter,
        })
    }

    /// Multi-seed GPU IK. Each LM iteration dispatches one batched
    /// Jacobian call with all `seeds × position_objectives` requests,
    /// scoring each seed's step on the GPU's parallel cost. Returns
    /// the seed with the lowest final residual.
    pub fn solve_gpu_with_seeds(
        &self,
        engine: &Engine,
        model: &Model,
        state: &mut State,
        seeds: &[Vec<Real>],
    ) -> Result<IkResult> {
        if seeds.is_empty() {
            return Err(Error::Build("ik::solve_gpu_with_seeds: no seeds".into()));
        }
        let cfg = &self.config;
        let dof_total = articulation_dof_total(model, cfg.articulation)?;
        if self
            .objectives
            .iter()
            .any(|o| matches!(o, IkObjective::SphereAvoid { .. }))
        {
            return Err(Error::Build(
                "ik::solve_gpu_with_seeds: SphereAvoid objectives are not supported in \
                 the multi-seed GPU path yet — use solve() or solve_gpu()"
                    .into(),
            ));
        }
        for (i, s) in seeds.iter().enumerate() {
            if s.len() != dof_total {
                return Err(Error::Build(format!(
                    "ik::solve_gpu_with_seeds: seed[{i}] len {} != articulation DOFs {dof_total}",
                    s.len()
                )));
            }
        }

        let kernel = crate::kernels::JacobianKernel::new(engine, model)?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("solve_gpu_with_seeds: no device".into()))?
            .clone();
        let queue = engine
            .queue()
            .ok_or_else(|| Error::Gpu("solve_gpu_with_seeds: no queue".into()))?
            .clone();

        let n_pos = self
            .objectives
            .iter()
            .filter(|o| matches!(o, IkObjective::Position { .. }))
            .count();
        let block_floats = 6 * dof_total;
        let n_seeds = seeds.len();
        let n_bodies = model.body_count;
        let total_floats = block_floats * n_pos.max(1) * n_seeds;
        let j_buf = kernel.create_output_buffer(&device, total_floats);

        // One concatenated body_q buffer holding `n_seeds * n_bodies`
        // transforms; each seed owns `body_q[seed*n_bodies .. (seed+1)*n_bodies]`.
        let bq_size =
            (n_seeds * n_bodies * core::mem::size_of::<crate::math::GpuTransform>()) as u64;
        let bq_buf = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("molla::ik multi-seed body_q"),
                size: bq_size.max(32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );

        // Per-seed working state.
        struct SeedState {
            q: Vec<Real>,
            e: Vec<Real>,
            e_norm: Real,
            e_cost: Real,
            damping: Real,
            converged: bool,
            iters: usize,
            dead: bool,
        }
        let mut seeds_state: Vec<SeedState> = Vec::with_capacity(n_seeds);
        for s in seeds {
            // Initial residual via host FK.
            write_q(model, state, cfg.articulation, s)?;
            crate::sim::eval_fk(model, state)?;
            let e = self.compute_residuals(model, state)?;
            let e_n = norm(&e);
            let e_c = loss_cost(cfg.loss, &e);
            seeds_state.push(SeedState {
                q: s.clone(),
                e,
                e_norm: e_n,
                e_cost: e_c,
                damping: cfg.damping_init,
                converged: e_n < cfg.residual_tol,
                iters: 0,
                dead: false,
            });
        }

        for it in 0..cfg.max_iter {
            // 1) Pack body_q for every still-active seed.
            let mut body_q_pack: Vec<crate::math::GpuTransform> =
                vec![crate::math::GpuTransform::default(); n_seeds * n_bodies];
            for (si, s) in seeds_state.iter().enumerate() {
                if s.dead || s.converged {
                    continue;
                }
                write_q(model, state, cfg.articulation, &s.q)?;
                crate::sim::eval_fk(model, state)?;
                let bq = state.body_q.host()?;
                for (bi, t) in bq.iter().enumerate() {
                    body_q_pack[si * n_bodies + bi] = (*t).into();
                }
            }
            molla_core::gpu_write_buffer!(queue, &bq_buf, 0, bytemuck::cast_slice(&body_q_pack));

            // 2) Build batched requests.
            let mut requests: Vec<crate::kernels::JacobianRequest> = Vec::new();
            let mut block_idx = 0u32;
            for (si, s) in seeds_state.iter().enumerate() {
                if s.dead || s.converged {
                    // Still must allocate the same blocks per seed to
                    // keep indexing consistent, but we won't read them.
                    block_idx += n_pos as u32;
                    continue;
                }
                for obj in &self.objectives {
                    if let IkObjective::Position {
                        body, point_local, ..
                    } = *obj
                    {
                        requests.push(crate::kernels::JacobianRequest {
                            articulation: cfg.articulation as u32,
                            target_body: body as u32,
                            output_offset: block_idx * block_floats as u32,
                            body_q_offset: (si * n_bodies) as u32,
                            point_local: [
                                point_local.x as f32,
                                point_local.y as f32,
                                point_local.z as f32,
                                0.0,
                            ],
                        });
                        block_idx += 1;
                    }
                }
            }
            if !requests.is_empty() {
                kernel.dispatch_with_body_q(engine, model, &bq_buf, &requests, &j_buf)?;
            }
            let j_blocks = kernel.read_back(engine, &j_buf, total_floats)?;

            // 3) Per-seed CPU normal-eq solve + trial-step accept/reject.
            let mut all_done = true;
            for si in 0..n_seeds {
                let seed_state = &mut seeds_state[si];
                if seed_state.dead || seed_state.converged {
                    continue;
                }
                all_done = false;

                let m = seed_state.e.len();
                let mut j_mat = vec![0.0; m * dof_total];
                let mut row = 0usize;
                let mut block_idx_local = si * n_pos;
                for obj in &self.objectives {
                    match *obj {
                        IkObjective::Position { weight, .. } => {
                            let off = block_idx_local * block_floats;
                            for r in 0..3 {
                                for k in 0..dof_total {
                                    j_mat[(row + r) * dof_total + k] =
                                        weight * (j_blocks[off + r * dof_total + k] as Real);
                                }
                            }
                            row += 3;
                            block_idx_local += 1;
                        }
                        IkObjective::JointLimit { dof, weight, .. } => {
                            if dof < dof_total {
                                j_mat[row * dof_total + dof] = weight;
                            }
                            row += 1;
                        }
                        IkObjective::SphereAvoid { .. } => {
                            unreachable!("rejected at solve_gpu_with_seeds entry")
                        }
                    }
                }
                // Normal equations (Stage 24b — IRLS-weighted).
                let weights = irls_weights(cfg.loss, &seed_state.e);
                let mut jtj = vec![0.0; dof_total * dof_total];
                for i in 0..dof_total {
                    for k in 0..dof_total {
                        let mut s_acc = 0.0;
                        for r in 0..m {
                            s_acc +=
                                weights[r] * j_mat[r * dof_total + i] * j_mat[r * dof_total + k];
                        }
                        jtj[i * dof_total + k] = s_acc;
                    }
                    jtj[i * dof_total + i] += seed_state.damping;
                }
                let mut rhs = vec![0.0; dof_total];
                for i in 0..dof_total {
                    let mut s_acc = 0.0;
                    for r in 0..m {
                        s_acc -= weights[r] * j_mat[r * dof_total + i] * seed_state.e[r];
                    }
                    rhs[i] = s_acc;
                }
                let dq = match ik_solve_normal_eq(&mut jtj, &rhs, dof_total) {
                    Ok(x) => x,
                    Err(_) => {
                        seed_state.damping *= cfg.damping_grow;
                        continue;
                    }
                };
                let dq_norm = norm(&dq);
                if dq_norm < cfg.step_tol {
                    seed_state.iters = it + 1;
                    seed_state.dead = true;
                    continue;
                }

                let mut q_trial = seed_state.q.clone();
                for i in 0..dof_total {
                    q_trial[i] += dq[i];
                }
                // Score on host FK.
                write_q(model, state, cfg.articulation, &q_trial)?;
                crate::sim::eval_fk(model, state)?;
                let e_trial = self.compute_residuals(model, state)?;
                let e_trial_norm = norm(&e_trial);
                let e_trial_cost = loss_cost(cfg.loss, &e_trial);
                if e_trial_cost < seed_state.e_cost {
                    seed_state.q = q_trial;
                    seed_state.e = e_trial;
                    seed_state.e_norm = e_trial_norm;
                    seed_state.e_cost = e_trial_cost;
                    seed_state.damping = (seed_state.damping / cfg.damping_shrink).max(1.0e-12);
                    if seed_state.e_norm < cfg.residual_tol {
                        seed_state.converged = true;
                        seed_state.iters = it + 1;
                    }
                } else {
                    seed_state.damping *= cfg.damping_grow;
                    if seed_state.damping > 1.0e12 {
                        seed_state.dead = true;
                        seed_state.iters = it + 1;
                    }
                }
            }
            if all_done {
                break;
            }
        }

        // Pick best seed.
        let mut best_idx = 0usize;
        let mut best_res = Real::INFINITY;
        for (i, s) in seeds_state.iter().enumerate() {
            if s.e_norm < best_res {
                best_res = s.e_norm;
                best_idx = i;
            }
        }
        let best = &seeds_state[best_idx];
        // Restore final pose for the winning seed.
        write_q(model, state, cfg.articulation, &best.q)?;
        crate::sim::eval_fk(model, state)?;
        Ok(IkResult {
            q: best.q.clone(),
            residual: best.e_norm,
            converged: best.converged,
            iters: best.iters.max(1),
        })
    }
}

fn articulation_dof_total(model: &Model, articulation: usize) -> Result<usize> {
    let dof_offsets = model.joint_dof_offset.host()?;
    let (j_begin, j_end) = if model.articulation_count == 0 {
        (0, model.joint_count)
    } else {
        let starts = model.articulation_start.host()?;
        if articulation + 1 >= starts.len() {
            return Err(Error::Build(format!(
                "ik: articulation {articulation} out of range"
            )));
        }
        (
            starts[articulation] as usize,
            starts[articulation + 1] as usize,
        )
    };
    Ok((dof_offsets[j_end] - dof_offsets[j_begin]) as usize)
}
