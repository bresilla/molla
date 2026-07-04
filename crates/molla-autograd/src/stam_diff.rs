//! Differentiable Eulerian fluid — the reverse-mode adjoint of a 2D
//! periodic inviscid Stam step (the last diffsim family: cloth,
//! softbody, MPM, Featherstone, XPBD/VBD and rays all have adjoints;
//! this adds smoke).
//!
//! Forward pipeline per step (f64, matching `molla-fem`'s
//! `StamFluidState::step_cpu` minus diffusion — the inviscid
//! diffsim-standard subset; Gauss-Seidel diffusion is not
//! self-adjoint and stays out of scope until needed):
//!
//! 1. `u += dt·force` (the control field being optimised),
//!    `ρ += dt·source`;
//! 2. pressure projection (K Jacobi sweeps on the periodic
//!    central-difference Laplacian);
//! 3. semi-Lagrangian velocity advection `u ← interp(u, x − dt·u)`;
//! 4. projection again;
//! 5. semi-Lagrangian density advection along the projected field.
//!
//! Adjoint structure, term by term:
//! * **Projection is SELF-ADJOINT here** — `P = I − G·L̃⁻¹·D` with
//!   central-difference `D`/`G` satisfying `Dᵀ = −G` on the periodic
//!   grid, and the K-sweep Jacobi approximation `L̃⁻¹` symmetric
//!   because the periodic Laplacian's diagonal is CONSTANT (the sweep
//!   map is a polynomial in the symmetric `L`). So the reverse pass
//!   just re-applies the forward projection to the adjoint field —
//!   no stored state.
//! * **Advection** splits into the linear gather (adjoint =
//!   scatter-transpose with the same bilinear weights) and the
//!   nonlinear backtrace (adjoint feeds `−dt·(∇̃field)(x_back)` into
//!   the advecting velocity's adjoint, where `∇̃` is the gradient of
//!   the bilinear interpolant — exact a.e., kinks on cell boundaries,
//!   the same measure-zero caveat as the barrier stencils).
//! * Force/source adds are `dt`-scaled identities.
//!
//! Checkpoints per step: the pre-advection velocity field and the
//! pre-advection density (everything else is linear or recomputable).

use molla_math::Real;

/// Configuration for [`StamFluidDiff`].
#[derive(Copy, Clone, Debug)]
pub struct StamDiffConfig {
    pub grid_n: usize,
    pub dt: Real,
    /// Jacobi sweeps per projection.
    pub pressure_iters: usize,
}

impl Default for StamDiffConfig {
    fn default() -> Self {
        Self {
            grid_n: 16,
            dt: 0.05,
            pressure_iters: 60,
        }
    }
}

/// Reverse-mode differentiable 2D periodic Stam fluid (see module
/// docs). `force` is a per-cell, time-constant control field.
pub struct StamFluidDiff {
    pub cfg: StamDiffConfig,
    /// Current state.
    pub u: Vec<[Real; 2]>,
    pub density: Vec<Real>,
    /// Per-step checkpoints (pre-velocity-advection field, projected
    /// advecting field, pre-density-advection density).
    ckpt_u_pre: Vec<Vec<[Real; 2]>>,
    ckpt_u_adv: Vec<Vec<[Real; 2]>>,
    ckpt_rho_pre: Vec<Vec<Real>>,
}

fn wrap(i: isize, n: usize) -> usize {
    i.rem_euclid(n as isize) as usize
}

/// Bilinear sample of a scalar field at grid coordinates `(x, y)`
/// (cell units, periodic), returning (value, ∂/∂x, ∂/∂y) — the
/// gradient of the INTERPOLANT (exact a.e.).
fn sample(field: &[Real], n: usize, x: Real, y: Real) -> (Real, Real, Real) {
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let i0 = wrap(x0 as isize, n);
    let j0 = wrap(y0 as isize, n);
    let i1 = wrap(x0 as isize + 1, n);
    let j1 = wrap(y0 as isize + 1, n);
    let f00 = field[j0 * n + i0];
    let f10 = field[j0 * n + i1];
    let f01 = field[j1 * n + i0];
    let f11 = field[j1 * n + i1];
    let v = f00 * (1.0 - fx) * (1.0 - fy)
        + f10 * fx * (1.0 - fy)
        + f01 * (1.0 - fx) * fy
        + f11 * fx * fy;
    let dx = (f10 - f00) * (1.0 - fy) + (f11 - f01) * fy;
    let dy = (f01 - f00) * (1.0 - fx) + (f11 - f10) * fx;
    (v, dx, dy)
}

/// Scatter `adj` through the bilinear stencil at `(x, y)` — the exact
/// transpose of the gather in [`sample`].
fn scatter(field_adj: &mut [Real], n: usize, x: Real, y: Real, adj: Real) {
    let x0 = x.floor();
    let y0 = y.floor();
    let fx = x - x0;
    let fy = y - y0;
    let i0 = wrap(x0 as isize, n);
    let j0 = wrap(y0 as isize, n);
    let i1 = wrap(x0 as isize + 1, n);
    let j1 = wrap(y0 as isize + 1, n);
    field_adj[j0 * n + i0] += adj * (1.0 - fx) * (1.0 - fy);
    field_adj[j0 * n + i1] += adj * fx * (1.0 - fy);
    field_adj[j1 * n + i0] += adj * (1.0 - fx) * fy;
    field_adj[j1 * n + i1] += adj * fx * fy;
}

/// Pressure projection with K Jacobi sweeps (periodic central
/// differences). Linear and self-adjoint — used verbatim by both the
/// forward and the reverse pass.
fn project(u: &mut [[Real; 2]], n: usize, iters: usize) {
    let nn = n * n;
    let half = 0.5; // central difference over 2·dx with dx = 1 (cell units)
    let mut div = vec![0.0; nn];
    for j in 0..n {
        for i in 0..n {
            let ip = wrap(i as isize + 1, n);
            let im = wrap(i as isize - 1, n);
            let jp = wrap(j as isize + 1, n);
            let jm = wrap(j as isize - 1, n);
            div[j * n + i] = half * (u[j * n + ip][0] - u[j * n + im][0])
                + half * (u[jp * n + i][1] - u[jm * n + i][1]);
        }
    }
    // Jacobi on the wide (2·dx) Laplacian implied by the
    // central-difference div/grad pair: L p = div, stencil
    // (p[i±2] etc.)/4 — matching molla-fem's projection choice.
    let mut p = vec![0.0; nn];
    let mut p_new = vec![0.0; nn];
    for _ in 0..iters {
        for j in 0..n {
            for i in 0..n {
                let ip2 = wrap(i as isize + 2, n);
                let im2 = wrap(i as isize - 2, n);
                let jp2 = wrap(j as isize + 2, n);
                let jm2 = wrap(j as isize - 2, n);
                let nb = p[j * n + ip2] + p[j * n + im2] + p[jp2 * n + i] + p[jm2 * n + i];
                p_new[j * n + i] = (nb - 4.0 * div[j * n + i]) / 4.0;
            }
        }
        std::mem::swap(&mut p, &mut p_new);
    }
    for j in 0..n {
        for i in 0..n {
            let ip = wrap(i as isize + 1, n);
            let im = wrap(i as isize - 1, n);
            let jp = wrap(j as isize + 1, n);
            let jm = wrap(j as isize - 1, n);
            u[j * n + i][0] -= half * (p[j * n + ip] - p[j * n + im]);
            u[j * n + i][1] -= half * (p[jp * n + i] - p[jm * n + i]);
        }
    }
}

impl StamFluidDiff {
    pub fn new(cfg: StamDiffConfig) -> Self {
        let nn = cfg.grid_n * cfg.grid_n;
        Self {
            cfg,
            u: vec![[0.0; 2]; nn],
            density: vec![0.0; nn],
            ckpt_u_pre: Vec::new(),
            ckpt_u_adv: Vec::new(),
            ckpt_rho_pre: Vec::new(),
        }
    }

    /// One forward step, recording checkpoints.
    pub fn step(&mut self, force: &[[Real; 2]], source: &[Real]) {
        let n = self.cfg.grid_n;
        let dt = self.cfg.dt;
        for k in 0..n * n {
            self.u[k][0] += dt * force[k][0];
            self.u[k][1] += dt * force[k][1];
            self.density[k] += dt * source[k];
        }
        project(&mut self.u, n, self.cfg.pressure_iters);
        self.ckpt_u_pre.push(self.u.clone());

        // Advect u by itself (component-wise semi-Lagrangian).
        let ux: Vec<Real> = self.u.iter().map(|v| v[0]).collect();
        let uy: Vec<Real> = self.u.iter().map(|v| v[1]).collect();
        let mut u_new = vec![[0.0; 2]; n * n];
        for j in 0..n {
            for i in 0..n {
                let k = j * n + i;
                let xb = i as Real - dt * self.u[k][0];
                let yb = j as Real - dt * self.u[k][1];
                u_new[k][0] = sample(&ux, n, xb, yb).0;
                u_new[k][1] = sample(&uy, n, xb, yb).0;
            }
        }
        self.u = u_new;
        project(&mut self.u, n, self.cfg.pressure_iters);
        self.ckpt_u_adv.push(self.u.clone());
        self.ckpt_rho_pre.push(self.density.clone());

        // Advect density along the projected field.
        let rho_old = self.density.clone();
        for j in 0..n {
            for i in 0..n {
                let k = j * n + i;
                let xb = i as Real - dt * self.u[k][0];
                let yb = j as Real - dt * self.u[k][1];
                self.density[k] = sample(&rho_old, n, xb, yb).0;
            }
        }
    }

    /// Reverse pass over all recorded steps. `adj_u` / `adj_rho` seed
    /// ∂J/∂u_T and ∂J/∂ρ_T; returns (∂J/∂force, ∂J/∂source,
    /// ∂J/∂u₀, ∂J/∂ρ₀) for the time-constant control fields.
    #[allow(clippy::type_complexity)]
    pub fn backward(
        &self,
        mut adj_u: Vec<[Real; 2]>,
        mut adj_rho: Vec<Real>,
    ) -> (Vec<[Real; 2]>, Vec<Real>, Vec<[Real; 2]>, Vec<Real>) {
        let n = self.cfg.grid_n;
        let dt = self.cfg.dt;
        let nn = n * n;
        let mut adj_force = vec![[0.0; 2]; nn];
        let mut adj_source = vec![0.0; nn];

        for s in (0..self.ckpt_u_pre.len()).rev() {
            let u_adv = &self.ckpt_u_adv[s]; // projected advecting field
            let rho_pre = &self.ckpt_rho_pre[s];

            // ── reverse 5: density advection.
            let mut adj_rho_pre = vec![0.0; nn];
            let mut adj_u_from_rho = vec![[0.0; 2]; nn];
            for j in 0..n {
                for i in 0..n {
                    let k = j * n + i;
                    let a = adj_rho[k];
                    if a == 0.0 {
                        continue;
                    }
                    let xb = i as Real - dt * u_adv[k][0];
                    let yb = j as Real - dt * u_adv[k][1];
                    scatter(&mut adj_rho_pre, n, xb, yb, a);
                    let (_, gx, gy) = sample(rho_pre, n, xb, yb);
                    adj_u_from_rho[k][0] -= dt * a * gx;
                    adj_u_from_rho[k][1] -= dt * a * gy;
                }
            }
            adj_rho = adj_rho_pre;
            for k in 0..nn {
                adj_u[k][0] += adj_u_from_rho[k][0];
                adj_u[k][1] += adj_u_from_rho[k][1];
            }

            // ── reverse 4: projection (self-adjoint).
            project(&mut adj_u, n, self.cfg.pressure_iters);

            // ── reverse 3: velocity self-advection.
            let u_pre = &self.ckpt_u_pre[s];
            let ux: Vec<Real> = u_pre.iter().map(|v| v[0]).collect();
            let uy: Vec<Real> = u_pre.iter().map(|v| v[1]).collect();
            let mut adj_u_pre = vec![[0.0; 2]; nn];
            let mut adj_ux = vec![0.0; nn];
            let mut adj_uy = vec![0.0; nn];
            for j in 0..n {
                for i in 0..n {
                    let k = j * n + i;
                    let (ax, ay) = (adj_u[k][0], adj_u[k][1]);
                    if ax == 0.0 && ay == 0.0 {
                        continue;
                    }
                    let xb = i as Real - dt * u_pre[k][0];
                    let yb = j as Real - dt * u_pre[k][1];
                    // Gather-transpose for each advected component.
                    scatter(&mut adj_ux, n, xb, yb, ax);
                    scatter(&mut adj_uy, n, xb, yb, ay);
                    // Backtrace dependence on the advecting u[k].
                    let (_, gxx, gxy) = sample(&ux, n, xb, yb);
                    let (_, gyx, gyy) = sample(&uy, n, xb, yb);
                    adj_u_pre[k][0] -= dt * (ax * gxx + ay * gyx);
                    adj_u_pre[k][1] -= dt * (ax * gxy + ay * gyy);
                }
            }
            for k in 0..nn {
                adj_u_pre[k][0] += adj_ux[k];
                adj_u_pre[k][1] += adj_uy[k];
            }
            adj_u = adj_u_pre;

            // ── reverse 2: first projection (self-adjoint).
            project(&mut adj_u, n, self.cfg.pressure_iters);

            // ── reverse 1: force / source adds.
            for k in 0..nn {
                adj_force[k][0] += dt * adj_u[k][0];
                adj_force[k][1] += dt * adj_u[k][1];
                adj_source[k] += dt * adj_rho[k];
            }
        }
        (adj_force, adj_source, adj_u, adj_rho)
    }
}
