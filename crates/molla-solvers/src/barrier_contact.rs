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
    /// Collision triangles (phase 3 — self-contact). Empty = obstacle
    /// contacts only.
    pub triangles: Vec<[u32; 3]>,
    /// Unique triangle edges, derived by [`Self::with_triangles`].
    pub collision_edges: Vec<[u32; 2]>,
    /// Pinned vertices are true DoF removals: x̃ = x (no predictor
    /// gravity — a huge mass does NOT pin under an incremental
    /// potential, since the predictor free-falls independently of
    /// mass), zero gradient row, identity Hessian row, zero velocity.
    pub pinned: Vec<bool>,
    pub params: BarrierParams,
}

/// One self-contact candidate: a non-adjacent vertex-triangle or
/// edge-edge pair within the candidate margin, with its clamped
/// closest-point weights frozen for the current iterate (the clamped
/// closest point makes constant-weight gradients EXACT away from
/// region boundaries — the a.e.-exact IPC treatment).
///
/// At EXACTLY degenerate configurations (a vertex precisely above a
/// mesh vertex/edge — e.g. perfectly aligned grids) the closest-point
/// map sits on a region boundary and the gradient is one-sided:
/// still repulsive, still descent-compatible, but an FD probe
/// straddling the kink will disagree (measured: 3–6% on aligned
/// grids, 3.5e-7 after any generic jitter). Newton tolerates this —
/// the configuration is measure-zero and the line search only needs
/// descent.
#[derive(Copy, Clone, Debug)]
enum SelfPair {
    /// (vertex, tri-index, barycentric weights of the closest point)
    VertTri(u32, u32, [Real; 3]),
    /// (edge-a index, edge-b index, param on a, param on b)
    EdgeEdge(u32, u32, Real, Real),
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
                "barrier_contact: all masses must be positive (use pin_vertices to pin)".into(),
            ));
        }
        let n = positions.len();
        Ok(Self {
            velocities: vec![Vec3::ZERO; n],
            positions,
            masses,
            springs,
            obstacles,
            triangles: Vec::new(),
            collision_edges: Vec::new(),
            pinned: vec![false; n],
            params,
        })
    }

    /// Pin vertices in place (true DoF removal — see `pinned`).
    pub fn pin_vertices(&mut self, indices: &[usize]) {
        for &i in indices {
            self.pinned[i] = true;
        }
    }

    /// Attach collision triangles (derives the unique edge list).
    pub fn with_triangles(mut self, triangles: Vec<[u32; 3]>) -> Self {
        let mut edges = std::collections::BTreeSet::new();
        for t in &triangles {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                edges.insert((a.min(b), a.max(b)));
            }
        }
        self.collision_edges = edges.into_iter().map(|(a, b)| [a, b]).collect();
        self.triangles = triangles;
        self
    }

    /// Closest point on triangle (a,b,c) to p, as barycentric weights.
    fn tri_closest_weights(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> [Real; 3] {
        // Ericson, Real-Time Collision Detection §5.1.5.
        let ab = b - a;
        let ac = c - a;
        let ap = p - a;
        let d1 = ab.dot(ap);
        let d2 = ac.dot(ap);
        if d1 <= 0.0 && d2 <= 0.0 {
            return [1.0, 0.0, 0.0];
        }
        let bp = p - b;
        let d3 = ab.dot(bp);
        let d4 = ac.dot(bp);
        if d3 >= 0.0 && d4 <= d3 {
            return [0.0, 1.0, 0.0];
        }
        let vc = d1 * d4 - d3 * d2;
        if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
            let v = d1 / (d1 - d3);
            return [1.0 - v, v, 0.0];
        }
        let cp = p - c;
        let d5 = ab.dot(cp);
        let d6 = ac.dot(cp);
        if d6 >= 0.0 && d5 <= d6 {
            return [0.0, 0.0, 1.0];
        }
        let vb = d5 * d2 - d1 * d6;
        if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
            let w = d2 / (d2 - d6);
            return [1.0 - w, 0.0, w];
        }
        let va = d3 * d6 - d5 * d4;
        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
            return [0.0, 1.0 - w, w];
        }
        let denom = 1.0 / (va + vb + vc);
        let v = vb * denom;
        let w = vc * denom;
        [1.0 - v - w, v, w]
    }

    /// Closest parameters (s, t) between segments p1+s(q1−p1), p2+t(q2−p2).
    fn seg_closest_params(p1: Vec3, q1: Vec3, p2: Vec3, q2: Vec3) -> (Real, Real) {
        // Ericson §5.1.9.
        let d1 = q1 - p1;
        let d2 = q2 - p2;
        let r = p1 - p2;
        let a = d1.length_squared();
        let e = d2.length_squared();
        let f = d2.dot(r);
        let tiny = 1.0e-18;
        if a <= tiny && e <= tiny {
            return (0.0, 0.0);
        }
        if a <= tiny {
            return (0.0, (f / e).clamp(0.0, 1.0));
        }
        let c = d1.dot(r);
        if e <= tiny {
            return ((-c / a).clamp(0.0, 1.0), 0.0);
        }
        let b = d1.dot(d2);
        let denom = a * e - b * b;
        let mut s = if denom > tiny {
            ((b * f - c * e) / denom).clamp(0.0, 1.0)
        } else {
            0.0
        };
        let mut t = (b * s + f) / e;
        if t < 0.0 {
            t = 0.0;
            s = (-c / a).clamp(0.0, 1.0);
        } else if t > 1.0 {
            t = 1.0;
            s = ((b - c) / a).clamp(0.0, 1.0);
        }
        (s, t)
    }

    /// Difference vector (from the OTHER primitive's closest point to
    /// this pair's "first" primitive point) and the 4-vertex stencil
    /// (index, signed weight) of a pair at `x`.
    fn pair_geometry(&self, pair: &SelfPair, x: &[Vec3]) -> (Vec3, [(u32, Real); 4]) {
        match *pair {
            SelfPair::VertTri(v, t, w) => {
                let tri = self.triangles[t as usize];
                let cp = x[tri[0] as usize] * w[0]
                    + x[tri[1] as usize] * w[1]
                    + x[tri[2] as usize] * w[2];
                (
                    x[v as usize] - cp,
                    [(v, 1.0), (tri[0], -w[0]), (tri[1], -w[1]), (tri[2], -w[2])],
                )
            }
            SelfPair::EdgeEdge(ea, eb, s, t) => {
                let e1 = self.collision_edges[ea as usize];
                let e2 = self.collision_edges[eb as usize];
                let pa = x[e1[0] as usize] * (1.0 - s) + x[e1[1] as usize] * s;
                let pb = x[e2[0] as usize] * (1.0 - t) + x[e2[1] as usize] * t;
                (
                    pa - pb,
                    [
                        (e1[0], 1.0 - s),
                        (e1[1], s),
                        (e2[0], -(1.0 - t)),
                        (e2[1], -t),
                    ],
                )
            }
        }
    }

    /// Rebuild the self-contact candidate set at `x`: every
    /// non-adjacent VT and EE pair within `3·d̂`. Combined with the
    /// global travel cap `α·max|p| ≤ d̂` in the line search, a pair
    /// OUTSIDE the margin can close at most 2·d̂ in one Newton step
    /// and therefore can never reach contact unseen (3d̂ − 2d̂ = d̂ of
    /// slack) — the additive-CCD safety argument.
    fn self_candidates(&self, x: &[Vec3]) -> Vec<SelfPair> {
        let margin = 3.0 * self.params.dhat;
        let mut out = Vec::new();
        for (t_idx, tri) in self.triangles.iter().enumerate() {
            let (a, b, c) = (x[tri[0] as usize], x[tri[1] as usize], x[tri[2] as usize]);
            for v in 0..x.len() as u32 {
                if tri.contains(&v) {
                    continue;
                }
                let w = Self::tri_closest_weights(x[v as usize], a, b, c);
                let cp = a * w[0] + b * w[1] + c * w[2];
                if (x[v as usize] - cp).length() < margin {
                    out.push(SelfPair::VertTri(v, t_idx as u32, w));
                }
            }
        }
        for i in 0..self.collision_edges.len() {
            let e1 = self.collision_edges[i];
            for j in (i + 1)..self.collision_edges.len() {
                let e2 = self.collision_edges[j];
                if e1[0] == e2[0] || e1[0] == e2[1] || e1[1] == e2[0] || e1[1] == e2[1] {
                    continue;
                }
                let (s, t) = Self::seg_closest_params(
                    x[e1[0] as usize],
                    x[e1[1] as usize],
                    x[e2[0] as usize],
                    x[e2[1] as usize],
                );
                let pa = x[e1[0] as usize] * (1.0 - s) + x[e1[1] as usize] * s;
                let pb = x[e2[0] as usize] * (1.0 - t) + x[e2[1] as usize] * t;
                if (pa - pb).length() < margin {
                    out.push(SelfPair::EdgeEdge(i as u32, j as u32, s, t));
                }
            }
        }
        out
    }

    /// Reduced mass of a pair's stencil (harmonic mean of the two
    /// sides' effective masses) for the per-contact κ.
    fn pair_kappa(&self, pair: &SelfPair, h: Real) -> Real {
        let (m1, m2) = match *pair {
            SelfPair::VertTri(v, t, _) => {
                let tri = self.triangles[t as usize];
                let mt = (self.masses[tri[0] as usize]
                    + self.masses[tri[1] as usize]
                    + self.masses[tri[2] as usize])
                    / 3.0;
                (self.masses[v as usize], mt)
            }
            SelfPair::EdgeEdge(ea, eb, _, _) => {
                let e1 = self.collision_edges[ea as usize];
                let e2 = self.collision_edges[eb as usize];
                (
                    0.5 * (self.masses[e1[0] as usize] + self.masses[e1[1] as usize]),
                    0.5 * (self.masses[e2[0] as usize] + self.masses[e2[1] as usize]),
                )
            }
        };
        let m_red = 2.0 * m1 * m2 / (m1 + m2).max(1.0e-30);
        self.params.kappa_scale * m_red / (h * h)
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
    fn energy(&self, x: &[Vec3], x_tilde: &[Vec3], h: Real, pairs: &[SelfPair]) -> Real {
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
        for pair in pairs {
            let (diff, _) = self.pair_geometry(pair, x);
            let (b, _, _) = self.barrier(diff.length(), self.pair_kappa(pair, h));
            e += h2 * b;
        }
        e
    }

    /// ∇E at `x`.
    fn gradient(
        &self,
        x: &[Vec3],
        x_tilde: &[Vec3],
        h: Real,
        pairs: &[SelfPair],
        out: &mut [Vec3],
    ) {
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
        for pair in pairs {
            let (diff, stencil) = self.pair_geometry(pair, x);
            let dist = diff.length().max(1.0e-12);
            let n = diff / dist;
            let (_, db, _) = self.barrier(dist, self.pair_kappa(pair, h));
            if db != 0.0 {
                for (idx, w) in stencil {
                    out[idx as usize] += n * (h2 * db * w);
                }
            }
        }
        for (i, g) in out.iter_mut().enumerate() {
            if self.pinned[i] {
                *g = Vec3::ZERO;
            }
        }
    }

    /// Hessian-vector product `y = ∇²E · v` with the standard SPD
    /// projections (compression-clamped spring blocks, convex-only
    /// barrier blocks) — the operator PCG solves against.
    fn hessian_mul(&self, x: &[Vec3], h: Real, pairs: &[SelfPair], v: &[Vec3], y: &mut [Vec3]) {
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
        for pair in pairs {
            let (diff, stencil) = self.pair_geometry(pair, x);
            let dist = diff.length().max(1.0e-12);
            let n = diff / dist;
            let (_, _, ddb) = self.barrier(dist, self.pair_kappa(pair, h));
            if ddb > 0.0 {
                // Rank-1 convex block over the 4-vertex stencil:
                // (Σ w_j n·v_j) scattered back with the same weights.
                let mut s = 0.0;
                for (idx, w) in stencil {
                    s += w * n.dot(v[idx as usize]);
                }
                for (idx, w) in stencil {
                    y[idx as usize] += n * (h2 * ddb * w * s);
                }
            }
        }
        for (i, yi) in y.iter_mut().enumerate() {
            if self.pinned[i] {
                *yi = v[i] * self.masses[i]; // identity (scaled) row
            }
        }
    }

    /// Conservative step-size cap: no vertex may cross an obstacle
    /// surface along `x + α·p`. Exact for half-spaces; conservative
    /// (straight-line vs ball) for spheres. Returns α ∈ (0, 1].
    fn ccd_alpha(&self, x: &[Vec3], p: &[Vec3], pairs: &[SelfPair]) -> Real {
        let mut alpha: Real = 1.0;
        // Global travel cap: no vertex moves more than d̂ in one
        // accepted line-search step. This is what makes the 3·d̂
        // candidate margin airtight (see `self_candidates`).
        if !self.triangles.is_empty() {
            let p_max = p.iter().map(|q| q.length()).fold(0.0, Real::max);
            if p_max > 0.0 {
                alpha = alpha.min(self.params.dhat / p_max);
            }
        }
        // Per-candidate conservative TOI: the pair gap can close at
        // most at the summed max speeds of its two sides.
        for pair in pairs {
            let (diff, _) = self.pair_geometry(pair, x);
            let gap = diff.length();
            let closing = match *pair {
                SelfPair::VertTri(v, t, _) => {
                    let tri = self.triangles[t as usize];
                    p[v as usize].length()
                        + p[tri[0] as usize]
                            .length()
                            .max(p[tri[1] as usize].length())
                            .max(p[tri[2] as usize].length())
                }
                SelfPair::EdgeEdge(ea, eb, _, _) => {
                    let e1 = self.collision_edges[ea as usize];
                    let e2 = self.collision_edges[eb as usize];
                    p[e1[0] as usize].length().max(p[e1[1] as usize].length())
                        + p[e2[0] as usize].length().max(p[e2[1] as usize].length())
                }
            };
            if closing > 0.0 {
                alpha = alpha.min(0.9 * gap / closing);
            }
        }
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
        self.energy(x, x_tilde, h, &self.self_candidates(x))
    }

    pub fn fd_probe_gradient(&self, x: &[Vec3], x_tilde: &[Vec3], h: Real, out: &mut [Vec3]) {
        // NOTE for FD probing: candidates are rebuilt at the probe
        // point, so central differences straddle the same active set
        // as long as the probe step is far smaller than the margin
        // slack — 1e-7 vs d̂/2, three orders of headroom.
        self.gradient(x, x_tilde, h, &self.self_candidates(x), out)
    }

    pub fn fd_probe_hessian_mul(&self, x: &[Vec3], h: Real, v: &[Vec3], y: &mut [Vec3]) {
        self.hessian_mul(x, h, &self.self_candidates(x), v, y)
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
        g.min(self.min_self_gap(&self.positions))
    }

    /// Smallest non-adjacent VT/EE distance — the SELF-intersection
    /// scan half of the invariant (O(V·T + E²), oracle-grade).
    pub fn min_self_gap(&self, x: &[Vec3]) -> Real {
        let mut g = Real::INFINITY;
        for tri in &self.triangles {
            let (a, b, c) = (x[tri[0] as usize], x[tri[1] as usize], x[tri[2] as usize]);
            for v in 0..x.len() as u32 {
                if tri.contains(&v) {
                    continue;
                }
                let w = Self::tri_closest_weights(x[v as usize], a, b, c);
                g = g.min((x[v as usize] - (a * w[0] + b * w[1] + c * w[2])).length());
            }
        }
        for i in 0..self.collision_edges.len() {
            let e1 = self.collision_edges[i];
            for j in (i + 1)..self.collision_edges.len() {
                let e2 = self.collision_edges[j];
                if e1[0] == e2[0] || e1[0] == e2[1] || e1[1] == e2[0] || e1[1] == e2[1] {
                    continue;
                }
                let (s, t) = Self::seg_closest_params(
                    x[e1[0] as usize],
                    x[e1[1] as usize],
                    x[e2[0] as usize],
                    x[e2[1] as usize],
                );
                let pa = x[e1[0] as usize] * (1.0 - s) + x[e1[1] as usize] * s;
                let pb = x[e2[0] as usize] * (1.0 - t) + x[e2[1] as usize] * t;
                g = g.min((pa - pb).length());
            }
        }
        g
    }

    /// One implicit step of size `h`.
    pub fn step(&mut self, h: Real) -> Result<BarrierStepStats> {
        let n = self.positions.len();
        let x_prev = self.positions.clone();
        let x_tilde: Vec<Vec3> = (0..n)
            .map(|i| {
                if self.pinned[i] {
                    x_prev[i]
                } else {
                    x_prev[i] + self.velocities[i] * h + self.params.gravity * (h * h)
                }
            })
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
            // Fresh candidate set every iterate (IPC-style active set;
            // the 3·d̂ margin + travel cap make between-rebuild motion
            // safe).
            let pairs = self.self_candidates(&x);
            self.gradient(&x, &x_tilde, h, &pairs, &mut grad);
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
            let p = self.pcg(&x, h, &pairs, &grad)?;

            // CCD-truncated Armijo backtracking.
            let alpha_max = self.ccd_alpha(&x, &p, &pairs);
            let e0 = self.energy(&x, &x_tilde, h, &pairs);
            let slope: Real = grad.iter().zip(p.iter()).map(|(g, d)| g.dot(*d)).sum();
            let mut alpha = alpha_max;
            let mut accepted = false;
            for _ in 0..30 {
                let x_try: Vec<Vec3> = (0..n).map(|i| x[i] + p[i] * alpha).collect();
                let e_try = self.energy(&x_try, &x_tilde, h, &pairs);
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
                g.min(self.min_self_gap(&x))
            };
            stats.min_gap = stats.min_gap.min(gap_now);
            if it + 1 == self.params.max_newton_iters {
                stats.hit_iter_cap = true;
            }
        }

        stats.energy = self.energy(&x, &x_tilde, h, &self.self_candidates(&x));
        for i in 0..n {
            self.velocities[i] = if self.pinned[i] {
                Vec3::ZERO
            } else {
                (x[i] - x_prev[i]) / h
            };
        }
        self.positions = x;
        Ok(stats)
    }

    /// Matrix-free PCG for `H p = −g` (H is PD: mass + PSD projections).
    fn pcg(&self, x: &[Vec3], h: Real, pairs: &[SelfPair], grad: &[Vec3]) -> Result<Vec<Vec3>> {
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
            self.hessian_mul(x, h, pairs, &d, &mut hd);
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
