//! C0 phase 1 (roadmap Tier-C) — barrier-Newton contact, CPU f64 oracle.
//!
//! The incremental-potential formulation that makes interpenetration
//! impossible at EVERY iterate, not just at convergence:
//!
//! ```text
//!   E(x) = ½‖x − x̃‖²_M + h²·[ elastic(x) + Σ_c barrier(d_c(x)) ]
//!   x̃    = x_prev + h·v + h²·g
//! ```
//!
//! minimized by damped Newton where the line search is truncated at a
//! conservative time-of-impact BEFORE the Armijo backtracking — so no
//! trial state, accepted or rejected, ever crosses a surface. This is
//! the IPC recipe (Li et al.) with the cubic compact-support barrier
//! (Ando, ACM TOG 43.6 family):
//!
//! ```text
//!   b(d) = κ·(d̂ − d)³ / (3·d̂²)      for d < d̂,   0 otherwise
//! ```
//!
//! C² at `d̂` (b = b' = b'' = 0 there), convex on (0, d̂], and — unlike
//! the IPC log barrier — zero outside the activation band, so distant
//! geometry contributes nothing. Phase-1 κ is the mass-consistent
//! placeholder `κ_c = kappa_scale · m_c / h²` (per contact); the
//! elasticity-inclusive eigen-stiffness scheme of the paper replaces
//! this in a later phase — the plumbing is per-contact from day one so
//! that upgrade is local.
//!
//! Phase-1 scope (deliberately minimal, correctness-anchored):
//! * elastic model: stretch springs (the same J as the XPBD/cloth-diff
//!   spring Jacobian, compression-clamped for SPD);
//! * contacts: vertices vs ANALYTIC obstacles (half-space + sphere) —
//!   the full Newton/barrier/CCD structure without VT/EE pair
//!   management, which arrives in phase 3 over the LBVH;
//! * solve: matrix-free PCG on the (PD) Newton system with a block
//!   mass-diagonal preconditioner;
//! * invariant: `min_gap()` is exposed and the tests assert it stays
//!   positive after EVERY step, plus FD-validate ∇E and ∇²E·v.
//!
//! Everything is f64: this module is the parity oracle the GPU port
//! (phase 2) will be measured against.

use crate::core::{Error, Result};
use crate::math::{Real, Vec3};

/// Analytic obstacle for phase 1 (static, infinite mass).
#[derive(Copy, Clone, Debug)]
pub enum BarrierObstacle {
    /// Half-space: points with `n·x − offset < 0` are inside. `n` must
    /// be unit length.
    HalfSpace { n: Vec3, offset: Real },
    /// Solid sphere.
    Sphere { center: Vec3, radius: Real },
}

impl BarrierObstacle {
    /// Signed distance (> 0 outside) and its gradient at `x`.
    fn distance(&self, x: Vec3) -> (Real, Vec3) {
        match *self {
            BarrierObstacle::HalfSpace { n, offset } => (n.dot(x) - offset, n),
            BarrierObstacle::Sphere { center, radius } => {
                let d = x - center;
                let len = d.length().max(1.0e-12);
                (len - radius, d / len)
            }
        }
    }
}
// Distance-Hessian note: the `b'·∇²d` term of the barrier Hessian is
// dropped (standard IPC PSD projection) — for the half-space ∇²d = 0
// exactly; for the sphere ∇²d is PSD outside but is multiplied by
// b' ≤ 0, so the product is NSD and projecting it away keeps the
// Newton operator positive definite.

/// One stretch spring.
#[derive(Copy, Clone, Debug)]
pub struct BarrierSpring {
    pub a: u32,
    pub b: u32,
    pub rest: Real,
    pub k: Real,
}

/// Solver parameters.
#[derive(Copy, Clone, Debug)]
pub struct BarrierParams {
    /// Barrier activation distance d̂ (m). Contacts farther than this
    /// contribute nothing.
    pub dhat: Real,
    /// Dimensionless barrier stiffness scale (κ_c = scale·m_c/h²).
    pub kappa_scale: Real,
    /// Newton convergence: ‖∇E‖_∞ ≤ tol · (characteristic force).
    pub newton_tol: Real,
    pub max_newton_iters: usize,
    /// Inner PCG relative tolerance / iteration cap.
    pub cg_tol: Real,
    pub max_cg_iters: usize,
    pub gravity: Vec3,
}

impl Default for BarrierParams {
    fn default() -> Self {
        Self {
            dhat: 1.0e-3,
            kappa_scale: 1.0e3,
            newton_tol: 1.0e-6,
            max_newton_iters: 40,
            cg_tol: 1.0e-10,
            max_cg_iters: 2000,
            gravity: Vec3::new(0.0, -9.81, 0.0),
        }
    }
}

/// Per-step diagnostics.
#[derive(Copy, Clone, Debug, Default)]
pub struct BarrierStepStats {
    pub newton_iters: usize,
    /// Smallest obstacle gap seen at ANY accepted iterate this step.
    pub min_gap: Real,
    /// Final incremental-potential value.
    pub energy: Real,
    /// True when the Newton loop hit `max_newton_iters` before the
    /// gradient tolerance.
    pub hit_iter_cap: bool,
}

/// CPU f64 barrier-Newton shell (phase-1 oracle).
pub struct BarrierShell {
    pub positions: Vec<Vec3>,
    pub velocities: Vec<Vec3>,
    pub masses: Vec<Real>,
    pub springs: Vec<BarrierSpring>,
    pub obstacles: Vec<BarrierObstacle>,
    pub params: BarrierParams,
}

impl BarrierShell {
    pub fn new(
        positions: Vec<Vec3>,
        masses: Vec<Real>,
        springs: Vec<BarrierSpring>,
        obstacles: Vec<BarrierObstacle>,
        params: BarrierParams,
    ) -> Result<Self> {
        if positions.len() != masses.len() {
            return Err(Error::Solver(
                "barrier_contact: positions/masses length mismatch".into(),
            ));
        }
        if masses.iter().any(|&m| !m.is_finite() || m <= 0.0) {
            return Err(Error::Solver(
                "barrier_contact: all masses must be positive (pin via huge mass in v1)".into(),
            ));
        }
        let n = positions.len();
        Ok(Self {
            velocities: vec![Vec3::ZERO; n],
            positions,
            masses,
            springs,
            obstacles,
            params,
        })
    }

    fn barrier_kappa(&self, vertex: usize, h: Real) -> Real {
        self.params.kappa_scale * self.masses[vertex] / (h * h)
    }

    /// b, b', b'' of the cubic compact-support barrier at gap `d`.
    fn barrier(&self, d: Real, kappa: Real) -> (Real, Real, Real) {
        let dhat = self.params.dhat;
        if d >= dhat {
            return (0.0, 0.0, 0.0);
        }
        let s = dhat - d;
        let inv = 1.0 / (dhat * dhat);
        (
            kappa * s * s * s * inv / 3.0,
            -kappa * s * s * inv,
            2.0 * kappa * s * inv,
        )
    }

    /// Incremental potential at `x` (with predicted target `x_tilde`).
    fn energy(&self, x: &[Vec3], x_tilde: &[Vec3], h: Real) -> Real {
        let mut e = 0.0;
        for i in 0..x.len() {
            e += 0.5 * self.masses[i] * (x[i] - x_tilde[i]).length_squared();
        }
        let h2 = h * h;
        for s in &self.springs {
            let d = x[s.b as usize] - x[s.a as usize];
            let c = d.length() - s.rest;
            e += h2 * 0.5 * s.k * c * c;
        }
        for (i, xi) in x.iter().enumerate() {
            let kappa = self.barrier_kappa(i, h);
            for ob in &self.obstacles {
                let (dist, _) = ob.distance(*xi);
                let (b, _, _) = self.barrier(dist, kappa);
                e += h2 * b;
            }
        }
        e
    }

    /// ∇E at `x`.
    fn gradient(&self, x: &[Vec3], x_tilde: &[Vec3], h: Real, out: &mut [Vec3]) {
        for g in out.iter_mut() {
            *g = Vec3::ZERO;
        }
        for i in 0..x.len() {
            out[i] += (x[i] - x_tilde[i]) * self.masses[i];
        }
        let h2 = h * h;
        for s in &self.springs {
            let d = x[s.b as usize] - x[s.a as usize];
            let len = d.length().max(1.0e-12);
            let dir = d / len;
            // ∂E/∂x_b = +h²k(len − rest)·d̂ ; ∂E/∂x_a is the negation.
            let f = h2 * s.k * (len - s.rest);
            out[s.a as usize] -= dir * f;
            out[s.b as usize] += dir * f;
        }
        for (i, xi) in x.iter().enumerate() {
            let kappa = self.barrier_kappa(i, h);
            for ob in &self.obstacles {
                let (dist, grad_d) = ob.distance(*xi);
                let (_, db, _) = self.barrier(dist, kappa);
                out[i] += grad_d * (h2 * db);
            }
        }
    }

    /// Hessian-vector product `y = ∇²E · v` with the standard SPD
    /// projections (compression-clamped spring blocks, convex-only
    /// barrier blocks) — the operator PCG solves against.
    fn hessian_mul(&self, x: &[Vec3], h: Real, v: &[Vec3], y: &mut [Vec3]) {
        for yi in y.iter_mut() {
            *yi = Vec3::ZERO;
        }
        for i in 0..x.len() {
            y[i] += v[i] * self.masses[i];
        }
        let h2 = h * h;
        for s in &self.springs {
            let d = x[s.b as usize] - x[s.a as usize];
            let len = d.length().max(1.0e-12);
            let dir = d / len;
            // J = k[t(I − d̂d̂ᵀ) + d̂d̂ᵀ], t = max(1 − rest/len, 0):
            // clamping t keeps the block PSD in compression (standard
            // spring-Hessian projection).
            let t = (1.0 - s.rest / len).max(0.0);
            let dv = v[s.b as usize] - v[s.a as usize];
            let dv_par = dir * dir.dot(dv);
            let jdv = (dv - dv_par) * (h2 * s.k * t) + dv_par * (h2 * s.k);
            y[s.a as usize] -= jdv;
            y[s.b as usize] += jdv;
        }
        for (i, xi) in x.iter().enumerate() {
            let kappa = self.barrier_kappa(i, h);
            for ob in &self.obstacles {
                let (dist, grad_d) = ob.distance(*xi);
                let (_, _, ddb) = self.barrier(dist, kappa);
                if ddb > 0.0 {
                    // Convex part only: b''·∇d∇dᵀ (b'·∇²d dropped —
                    // see `distance_hessian_psd`).
                    y[i] += grad_d * (h2 * ddb * grad_d.dot(v[i]));
                }
            }
        }
    }

    /// Conservative step-size cap: no vertex may cross an obstacle
    /// surface along `x + α·p`. Exact for half-spaces; conservative
    /// (straight-line vs ball) for spheres. Returns α ∈ (0, 1].
    fn ccd_alpha(&self, x: &[Vec3], p: &[Vec3]) -> Real {
        let mut alpha: Real = 1.0;
        for (i, xi) in x.iter().enumerate() {
            for ob in &self.obstacles {
                match *ob {
                    BarrierObstacle::HalfSpace { n, offset } => {
                        let gap = n.dot(*xi) - offset;
                        let approach = -n.dot(p[i]);
                        if approach > 0.0 {
                            alpha = alpha.min(0.9 * gap / approach);
                        }
                    }
                    BarrierObstacle::Sphere { center, radius } => {
                        let gap = (*xi - center).length() - radius;
                        let speed = p[i].length();
                        if speed > 0.0 {
                            // Fully conservative: the vertex cannot
                            // close more than |p|·α of gap.
                            alpha = alpha.min(0.9 * gap / speed);
                        }
                    }
                }
            }
        }
        alpha.clamp(1.0e-12, 1.0)
    }

    /// Test probes — thin public wrappers so the FD contract tests can
    /// exercise the energy/derivative internals directly.
    pub fn fd_probe_energy(&self, x: &[Vec3], x_tilde: &[Vec3], h: Real) -> Real {
        self.energy(x, x_tilde, h)
    }

    pub fn fd_probe_gradient(&self, x: &[Vec3], x_tilde: &[Vec3], h: Real, out: &mut [Vec3]) {
        self.gradient(x, x_tilde, h, out)
    }

    pub fn fd_probe_hessian_mul(&self, x: &[Vec3], h: Real, v: &[Vec3], y: &mut [Vec3]) {
        self.hessian_mul(x, h, v, y)
    }

    /// Smallest gap between any vertex and any obstacle — THE
    /// invariant. Positive at all times or the solver is wrong.
    pub fn min_gap(&self) -> Real {
        let mut g = Real::INFINITY;
        for xi in &self.positions {
            for ob in &self.obstacles {
                g = g.min(ob.distance(*xi).0);
            }
        }
        g
    }

    /// One implicit step of size `h`.
    pub fn step(&mut self, h: Real) -> Result<BarrierStepStats> {
        let n = self.positions.len();
        let x_prev = self.positions.clone();
        let x_tilde: Vec<Vec3> = (0..n)
            .map(|i| x_prev[i] + self.velocities[i] * h + self.params.gravity * (h * h))
            .collect();

        let mut x = x_prev.clone();
        let mut grad = vec![Vec3::ZERO; n];
        let mut stats = BarrierStepStats {
            min_gap: self.min_gap(),
            ..Default::default()
        };

        // Characteristic gradient scale for the stopping test: the
        // inertia term of moving one dhat.
        let m_max = self.masses.iter().cloned().fold(0.0, Real::max);
        let tol = self.params.newton_tol * (m_max * self.params.dhat).max(1.0e-30);

        for it in 0..self.params.max_newton_iters {
            self.gradient(&x, &x_tilde, h, &mut grad);
            let g_inf = grad
                .iter()
                .map(|g| g.x.abs().max(g.y.abs()).max(g.z.abs()))
                .fold(0.0, Real::max);
            if g_inf <= tol {
                stats.newton_iters = it;
                break;
            }
            stats.newton_iters = it + 1;

            // PCG on H p = −g with block mass-diagonal preconditioner.
            let p = self.pcg(&x, h, &grad)?;

            // CCD-truncated Armijo backtracking.
            let alpha_max = self.ccd_alpha(&x, &p);
            let e0 = self.energy(&x, &x_tilde, h);
            let slope: Real = grad.iter().zip(p.iter()).map(|(g, d)| g.dot(*d)).sum();
            let mut alpha = alpha_max;
            let mut accepted = false;
            for _ in 0..30 {
                let x_try: Vec<Vec3> = (0..n).map(|i| x[i] + p[i] * alpha).collect();
                let e_try = self.energy(&x_try, &x_tilde, h);
                if e_try <= e0 + 1.0e-4 * alpha * slope {
                    x = x_try;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                // Gradient is descent-aligned by construction (PD
                // operator); a full backtracking failure means we're
                // at numerical stagnation — stop with what we have.
                stats.hit_iter_cap = true;
                break;
            }
            // Invariant bookkeeping at the ACCEPTED iterate.
            let gap_now = {
                let mut g = Real::INFINITY;
                for xi in &x {
                    for ob in &self.obstacles {
                        g = g.min(ob.distance(*xi).0);
                    }
                }
                g
            };
            stats.min_gap = stats.min_gap.min(gap_now);
            if it + 1 == self.params.max_newton_iters {
                stats.hit_iter_cap = true;
            }
        }

        stats.energy = self.energy(&x, &x_tilde, h);
        for i in 0..n {
            self.velocities[i] = (x[i] - x_prev[i]) / h;
        }
        self.positions = x;
        Ok(stats)
    }

    /// Matrix-free PCG for `H p = −g` (H is PD: mass + PSD projections).
    fn pcg(&self, x: &[Vec3], h: Real, grad: &[Vec3]) -> Result<Vec<Vec3>> {
        let n = x.len();
        let mut p_out = vec![Vec3::ZERO; n];
        let mut r: Vec<Vec3> = grad.iter().map(|g| -*g).collect();
        // Block mass-diagonal preconditioner.
        let precond = |r: &[Vec3], z: &mut Vec<Vec3>| {
            for i in 0..r.len() {
                z[i] = r[i] / self.masses[i];
            }
        };
        let mut z = vec![Vec3::ZERO; n];
        precond(&r, &mut z);
        let mut d = z.clone();
        let mut hd = vec![Vec3::ZERO; n];
        let dot = |a: &[Vec3], b: &[Vec3]| -> Real {
            a.iter().zip(b.iter()).map(|(x, y)| x.dot(*y)).sum()
        };
        let mut rz = dot(&r, &z);
        let r0: Real = dot(&r, &r);
        if r0 <= 0.0 {
            return Ok(p_out);
        }
        for _ in 0..self.params.max_cg_iters {
            self.hessian_mul(x, h, &d, &mut hd);
            let dhd = dot(&d, &hd);
            if dhd <= 0.0 {
                break; // numerically flat — accept current iterate
            }
            let alpha = rz / dhd;
            for i in 0..n {
                p_out[i] += d[i] * alpha;
                r[i] -= hd[i] * alpha;
            }
            if dot(&r, &r) <= self.params.cg_tol * self.params.cg_tol * r0 {
                break;
            }
            precond(&r, &mut z);
            let rz_new = dot(&r, &z);
            let beta = rz_new / rz;
            rz = rz_new;
            for i in 0..n {
                d[i] = z[i] + d[i] * beta;
            }
        }
        Ok(p_out)
    }
}
