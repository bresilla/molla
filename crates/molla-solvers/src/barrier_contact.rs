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
use crate::math::{Mat3, Real, Vec3};

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

/// One affine body (ABD, Lan et al. 2022) for STRONG rigid coupling:
/// 12 DOF — translation `p` and a free 3×3 `a_mat` held near rotation
/// by the stiff orthogonality potential `κ_o·‖AᵀA − I‖²_F`. The body's
/// surface samples live as PROXY vertices in the shell's position
/// array (positions = A·x̄ + p, exactly, since the map is linear), so
/// barriers, self-pairs, friction and the CCD line search all apply
/// unchanged — one Newton solve couples everything, with no explicit
/// force exchange and no rigid-side TOI patch.
#[derive(Clone, Debug)]
pub struct AffineBody {
    /// Proxy range in `positions`.
    pub proxy_start: usize,
    pub proxy_count: usize,
    /// COM-centred rest shape x̄.
    pub rest: Vec<Vec3>,
    pub p: Vec3,
    pub a_mat: Mat3,
    pub v_p: Vec3,
    pub v_a: Mat3,
    /// Total mass and the affine inertia J = Σ mᵢ x̄ᵢx̄ᵢᵀ.
    pub mass: Real,
    pub j_mat: Mat3,
    /// Orthogonality stiffness κ_o.
    pub kappa_ortho: Real,
}

/// One volumetric tet with a volume-preservation energy
/// `½·k_v·(V − V₀)²` (rest volume measured at attach time). Same
/// energy family as `molla-kernels::SoftTet`, whose gradients this
/// oracle re-derives and the FD test re-verifies.
#[derive(Copy, Clone, Debug)]
pub struct BarrierTet {
    pub ids: [u32; 4],
    pub kv: Real,
}

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
    /// Hard per-spring strain limit (length/rest), e.g. 1.05 = 5%.
    /// Enforced the same way contact is: a strain barrier keeps
    /// springs from pressing against the bound, and the LINE SEARCH
    /// truncates any step that would cross it — the finite cubic
    /// barrier alone cannot guarantee a hard bound; the truncation
    /// can. `None` disables.
    pub strain_limit: Option<Real>,
    /// Barrier stiffness scale for the strain limit, multiplying the
    /// elasticity-derived κ_s = k_spring·rest² (the same
    /// "stiffness from the material, not a global knob" philosophy
    /// as the contact κ).
    pub strain_barrier_scale: Real,
    /// Kinetic Coulomb coefficient μ_k for obstacle contacts
    /// (lagged IPC friction). 0 disables.
    pub friction_mu: Real,
    /// μ_s = scale·μ_k with scale ≥ 1. The stick well peaks at
    /// μ_s·λ before blending down to the kinetic plateau μ_k·λ —
    /// the split the roadmap A1 attempt showed velocity-level
    /// solvers cannot express.
    pub friction_mu_s_scale: Real,
    /// Stick-well half-width as a velocity (m/s): the per-step
    /// displacement well is e = ε_v·h. Smaller ⇒ less creep,
    /// stiffer friction Hessian.
    pub friction_epsv: Real,
    /// Elasticity-inclusive dynamic stiffness (the TOG 43.6 scheme):
    /// κ_c = scale·(m_c + h²·k_local)/h², where k_local is the Σ of
    /// incident spring stiffnesses — an upper bound on the local
    /// elastic eigen-stiffness. A stiff material then meets a
    /// proportionally stiff barrier instead of pressing through the
    /// activation band under its own elastic forces. `false` falls
    /// back to the mass-consistent κ = scale·m/h².
    pub elasticity_inclusive_kappa: bool,
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
            strain_limit: None,
            strain_barrier_scale: 10.0,
            friction_mu: 0.0,
            friction_mu_s_scale: 1.0,
            friction_epsv: 1.0e-3,
            elasticity_inclusive_kappa: true,
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
    /// contacts only. SET VIA [`Self::with_triangles`] — assigning
    /// this field directly desyncs the derived `collision_edges`.
    pub triangles: Vec<[u32; 3]>,
    /// Unique triangle edges, derived by [`Self::with_triangles`].
    pub collision_edges: Vec<[u32; 2]>,
    /// Chain-run provenance per collision edge: `Some((run, index))`
    /// for explicit rod edges, `None` for triangle-derived ones.
    edge_run: Vec<Option<(u32, u32)>>,
    /// Volumetric tets (volume-preservation energy). SET VIA
    /// [`Self::with_tets`] — assigning directly desyncs the derived
    /// rest volumes `tet_v0`.
    pub tets: Vec<BarrierTet>,
    /// Rest volumes, measured by [`Self::with_tets`].
    pub tet_v0: Vec<Real>,
    /// Lagged friction contacts, rebuilt at each step start from the
    /// committed positions (IPC semi-implicit friction: the normal
    /// force λ and the contact frame are FROZEN for the whole step's
    /// Newton solve, which keeps the potential smooth and
    /// FD-checkable).
    lagged_friction: Vec<FrictionContact>,
    lagged_pair_friction: Vec<PairFrictionContact>,
    /// Per-vertex Σ of incident spring stiffnesses (the local elastic
    /// eigen-stiffness bound feeding the elasticity-inclusive κ).
    vertex_ksum: Vec<Real>,
    /// Affine bodies (strong coupling); `proxy_of[i]` marks vertex i
    /// as a derived proxy of body `proxy_of[i]`.
    pub affine_bodies: Vec<AffineBody>,
    pub proxy_of: Vec<Option<u32>>,
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
/// One lagged obstacle-friction contact (see `lagged_friction`).
#[derive(Copy, Clone, Debug)]
struct FrictionContact {
    vertex: u32,
    /// Index into `obstacles` (reaction attribution).
    obstacle: u32,
    /// Contact normal at the lag anchor.
    n: Vec3,
    /// Anchor position (u = tangential part of x − anchor).
    anchor: Vec3,
    /// Lagged normal force magnitude in energy-gradient units
    /// (h²·(−b′) at the anchor gap).
    lambda: Real,
}

/// One lagged PAIR-friction contact (cloth-on-cloth): the 4-vertex
/// stencil, contact normal, per-vertex anchors and λ are all frozen at
/// step start; friction acts on the RELATIVE tangential slip
/// `T·Σ w_j (x_j − anchor_j)` of the stencil.
#[derive(Copy, Clone, Debug)]
struct PairFrictionContact {
    stencil: [(u32, Real); 4],
    n: Vec3,
    anchors: [Vec3; 4],
    lambda: Real,
}

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
        let mut vertex_ksum = vec![0.0; n];
        for s in &springs {
            vertex_ksum[s.a as usize] += s.k;
            vertex_ksum[s.b as usize] += s.k;
        }
        Ok(Self {
            velocities: vec![Vec3::ZERO; n],
            positions,
            masses,
            springs,
            obstacles,
            triangles: Vec::new(),
            collision_edges: Vec::new(),
            edge_run: Vec::new(),
            tets: Vec::new(),
            tet_v0: Vec::new(),
            lagged_friction: Vec::new(),
            lagged_pair_friction: Vec::new(),
            vertex_ksum,
            affine_bodies: Vec::new(),
            proxy_of: vec![None; n],
            pinned: vec![false; n],
            params,
        })
    }

    /// Attach volumetric tets; rest volumes measured from the
    /// CURRENT positions.
    pub fn with_tets(mut self, tets: Vec<BarrierTet>) -> Self {
        self.tet_v0 = tets
            .iter()
            .map(|t| Self::tet_volume(&self.positions, t.ids))
            .collect();
        self.tets = tets;
        self
    }

    fn tet_volume(x: &[Vec3], ids: [u32; 4]) -> Real {
        let a = x[ids[0] as usize];
        (x[ids[1] as usize] - a)
            .cross(x[ids[2] as usize] - a)
            .dot(x[ids[3] as usize] - a)
            / 6.0
    }

    /// Volume gradients g_b, g_c, g_d (g_a = −Σ) at `x`.
    fn tet_volume_grads(x: &[Vec3], ids: [u32; 4]) -> [Vec3; 4] {
        let pa = x[ids[0] as usize];
        let pb = x[ids[1] as usize];
        let pc = x[ids[2] as usize];
        let pd = x[ids[3] as usize];
        let gb = (pc - pa).cross(pd - pa) / 6.0;
        let gc = (pd - pa).cross(pb - pa) / 6.0;
        let gd = (pb - pa).cross(pc - pa) / 6.0;
        [-(gb + gc + gd), gb, gc, gd]
    }

    /// Add an affine body from world-space surface samples. Returns
    /// the body index; the samples are appended to `positions` as
    /// PROXY vertices (their triangles/edges can then be registered
    /// through [`Self::with_triangles`] exactly like cloth).
    pub fn add_affine_body(
        &mut self,
        samples: Vec<Vec3>,
        sample_masses: Vec<Real>,
        kappa_ortho: Real,
    ) -> usize {
        assert_eq!(samples.len(), sample_masses.len());
        assert!(!samples.is_empty(), "affine body needs samples");
        let mass: Real = sample_masses.iter().sum();
        assert!(
            mass > 0.0 && mass.is_finite(),
            "affine body needs positive total mass"
        );
        let com = samples
            .iter()
            .zip(sample_masses.iter())
            .fold(Vec3::ZERO, |acc, (x, &m)| acc + *x * m)
            / mass;
        let rest: Vec<Vec3> = samples.iter().map(|x| *x - com).collect();
        let mut j_mat = Mat3::ZERO;
        for (x, &m) in rest.iter().zip(sample_masses.iter()) {
            j_mat += Mat3::from_cols(*x * x.x, *x * x.y, *x * x.z) * m;
        }
        let body_idx = self.affine_bodies.len();
        let start = self.positions.len();
        for (x, &m) in samples.iter().zip(sample_masses.iter()) {
            self.positions.push(*x);
            self.velocities.push(Vec3::ZERO);
            self.masses.push(m);
            self.vertex_ksum.push(0.0);
            self.pinned.push(false);
            self.proxy_of.push(Some(body_idx as u32));
        }
        self.affine_bodies.push(AffineBody {
            proxy_start: start,
            proxy_count: samples.len(),
            rest,
            p: com,
            a_mat: Mat3::IDENTITY,
            v_p: Vec3::ZERO,
            v_a: Mat3::ZERO,
            mass,
            j_mat,
            kappa_ortho,
        });
        body_idx
    }

    /// Orthogonality energy, gradient and Gauss-projected
    /// Hessian-vector product for `A` (κ_o·‖AᵀA − I‖²_F; the
    /// residual-times-second-derivative term vanishes at AᵀA = I and
    /// is projected away — same SPD discipline as everything else).
    fn ortho_terms(a: Mat3, kappa: Real) -> (Real, Mat3) {
        let r = a.transpose() * a - Mat3::IDENTITY;
        let e = kappa
            * (r.col(0).length_squared() + r.col(1).length_squared() + r.col(2).length_squared());
        let grad = a * r * (4.0 * kappa);
        (e, grad)
    }

    fn ortho_hess_mul(a: Mat3, x: Mat3, kappa: Real) -> Mat3 {
        // Gauss part: 4κ·A·(XᵀA + AᵀX).
        a * (x.transpose() * a + a.transpose() * x) * (4.0 * kappa)
    }

    /// Append triangles (and their derived unique edges) to the
    /// EXISTING collision topology — used when several sources
    /// contribute surfaces (cloth + promoted ABD boxes).
    pub fn with_triangles_appended(mut self, triangles: Vec<[u32; 3]>) -> Self {
        let mut edges = std::collections::BTreeSet::new();
        for t in &triangles {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                edges.insert((a.min(b), a.max(b)));
            }
        }
        let added: Vec<[u32; 2]> = edges.into_iter().map(|(a, b)| [a, b]).collect();
        self.edge_run.extend(std::iter::repeat_n(None, added.len()));
        self.collision_edges.extend(added);
        self.triangles.extend(triangles);
        self
    }

    /// Register EXPLICIT collision edges (rod/chain segments without
    /// triangles — noodles, yarns, knots). They join the EE pair
    /// machinery exactly like triangle edges; call after
    /// `with_triangles` if both are used.
    ///
    /// Chain RUNS are detected (consecutive edges sharing a vertex)
    /// and near-neighbours along the same run (≤ 3 hops) are EXCLUDED
    /// from EE pairing: a straight rod's non-adjacent segments are
    /// COLINEAR — distance exactly zero — and a taut strand would
    /// otherwise fight itself with internal barrier pressure. This is
    /// the standard rod-IPC neighbour exclusion.
    pub fn with_collision_edges(mut self, edges: Vec<[u32; 2]>) -> Self {
        let base = self.collision_edges.len();
        // Assign run ids: consecutive appended edges sharing a vertex
        // belong to the same chain run.
        let mut run = self
            .edge_run
            .iter()
            .flatten()
            .map(|r| r.0)
            .max()
            .map_or(0, |m| m + 1);
        let mut pos = 0u32;
        for (k, e) in edges.iter().enumerate() {
            if k > 0 {
                let prev = edges[k - 1];
                let shares =
                    e[0] == prev[0] || e[0] == prev[1] || e[1] == prev[0] || e[1] == prev[1];
                if shares {
                    pos += 1;
                } else {
                    run += 1;
                    pos = 0;
                }
            }
            self.edge_run.push(Some((run, pos)));
            let _ = base;
        }
        self.collision_edges.extend(edges);
        self
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
        self.edge_run = vec![None; self.collision_edges.len()];
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
    /// Same-rigid-body exclusion: an affine body's proxies are exactly
    /// ON its own surface triangles (face centres!) — internal pairs
    /// are meaningless and would report zero gaps.
    /// Same-chain near-neighbour exclusion for explicit rod edges.
    fn chain_neighbors(&self, ei: usize, ej: usize) -> bool {
        match (
            self.edge_run.get(ei).copied().flatten(),
            self.edge_run.get(ej).copied().flatten(),
        ) {
            (Some((ra, pa)), Some((rb, pb))) => ra == rb && pa.abs_diff(pb) <= 3,
            _ => false,
        }
    }

    fn same_body(&self, a: u32, b: u32) -> bool {
        match (self.proxy_of[a as usize], self.proxy_of[b as usize]) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        }
    }

    fn self_candidates(&self, x: &[Vec3]) -> Vec<SelfPair> {
        let margin = 3.0 * self.params.dhat;
        let mut out = Vec::new();
        for (t_idx, tri) in self.triangles.iter().enumerate() {
            let (a, b, c) = (x[tri[0] as usize], x[tri[1] as usize], x[tri[2] as usize]);
            for v in 0..x.len() as u32 {
                if tri.contains(&v) || self.same_body(v, tri[0]) {
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
                if self.same_body(e1[0], e2[0]) {
                    continue;
                }
                if self.chain_neighbors(i, j) {
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
        // Per-side effective stiffness: mass, plus h²·(local elastic
        // eigen-stiffness bound) under the elasticity-inclusive
        // scheme; sides combine harmonically (series compliance).
        let side = |m: Real, ks: Real| -> Real {
            if self.params.elasticity_inclusive_kappa {
                m + h * h * ks
            } else {
                m
            }
        };
        let (s1, s2) = match *pair {
            SelfPair::VertTri(v, t, _) => {
                let tri = self.triangles[t as usize];
                let mt = (self.masses[tri[0] as usize]
                    + self.masses[tri[1] as usize]
                    + self.masses[tri[2] as usize])
                    / 3.0;
                let kt = (self.vertex_ksum[tri[0] as usize]
                    + self.vertex_ksum[tri[1] as usize]
                    + self.vertex_ksum[tri[2] as usize])
                    / 3.0;
                (
                    side(self.masses[v as usize], self.vertex_ksum[v as usize]),
                    side(mt, kt),
                )
            }
            SelfPair::EdgeEdge(ea, eb, _, _) => {
                let e1 = self.collision_edges[ea as usize];
                let e2 = self.collision_edges[eb as usize];
                (
                    side(
                        0.5 * (self.masses[e1[0] as usize] + self.masses[e1[1] as usize]),
                        0.5 * (self.vertex_ksum[e1[0] as usize] + self.vertex_ksum[e1[1] as usize]),
                    ),
                    side(
                        0.5 * (self.masses[e2[0] as usize] + self.masses[e2[1] as usize]),
                        0.5 * (self.vertex_ksum[e2[0] as usize] + self.vertex_ksum[e2[1] as usize]),
                    ),
                )
            }
        };
        let s_red = 2.0 * s1 * s2 / (s1 + s2).max(1.0e-30);
        self.params.kappa_scale * s_red / (h * h)
    }

    fn barrier_kappa(&self, vertex: usize, h: Real) -> Real {
        let m_eff = if self.params.elasticity_inclusive_kappa {
            self.masses[vertex] + h * h * self.vertex_ksum[vertex]
        } else {
            self.masses[vertex]
        };
        self.params.kappa_scale * m_eff / (h * h)
    }

    /// b, b', b'' of the cubic compact-support barrier at gap `d`.
    fn barrier(&self, d: Real, kappa: Real) -> (Real, Real, Real) {
        Self::barrier_with(d, self.params.dhat, kappa)
    }

    fn barrier_with(d: Real, dhat: Real, kappa: Real) -> (Real, Real, Real) {
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

    /// Strain-limit barrier band and stiffness for a spring: the gap
    /// is `limit − strain` (dimensionless), the activation band the
    /// last 20% of the allowed over-strain, κ_s from the spring's own
    /// stiffness (`k·rest²` is the energy scale of a unit strain).
    fn strain_barrier_terms(&self, s: &BarrierSpring, strain: Real) -> (Real, Real, Real) {
        let Some(limit) = self.params.strain_limit else {
            return (0.0, 0.0, 0.0);
        };
        // A limit at or below the rest strain (1.0) is nonsensical —
        // treat as disabled, matching the GPU convention.
        if limit <= 1.0 {
            return (0.0, 0.0, 0.0);
        }
        let gap = limit - strain;
        let band = 0.2 * (limit - 1.0).max(1.0e-6);
        let kappa = self.params.strain_barrier_scale * s.k * s.rest * s.rest;
        Self::barrier_with(gap, band, kappa)
    }

    /// Incremental potential at `x` (with predicted target `x_tilde`).
    fn energy(&self, x: &[Vec3], x_tilde: &[Vec3], h: Real, pairs: &[SelfPair]) -> Real {
        let mut e = 0.0;
        for i in 0..x.len() {
            // Proxy vertices carry no per-vertex inertia — theirs
            // lives in the owning body's 12-DOF mass terms.
            if self.proxy_of[i].is_none() {
                e += 0.5 * self.masses[i] * (x[i] - x_tilde[i]).length_squared();
            }
        }
        let h2 = h * h;
        for s in &self.springs {
            let d = x[s.b as usize] - x[s.a as usize];
            let len = d.length();
            let c = len - s.rest;
            e += h2 * 0.5 * s.k * c * c;
            if self.params.strain_limit.is_some() {
                let (b, _, _) = self.strain_barrier_terms(s, len / s.rest);
                e += h2 * b;
            }
        }
        for (t, &v0) in self.tets.iter().zip(self.tet_v0.iter()) {
            let dv = Self::tet_volume(x, t.ids) - v0;
            e += h2 * 0.5 * t.kv * dv * dv;
        }
        for (i, xi) in x.iter().enumerate() {
            if self.pinned[i] {
                continue; // pinned = boundary condition, not a contact
            }
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
        let ew = self.params.friction_epsv * h;
        for fc in &self.lagged_friction {
            let u = x[fc.vertex as usize] - fc.anchor;
            let ut = u - fc.n * fc.n.dot(u);
            e += self.friction_energy_of(fc.lambda, ut.length(), ew);
        }
        for pf in &self.lagged_pair_friction {
            let ut = self.pair_slip(pf, x);
            e += self.friction_energy_of(pf.lambda, ut.length(), ew);
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
            if self.proxy_of[i].is_none() {
                out[i] += (x[i] - x_tilde[i]) * self.masses[i];
            }
        }
        let h2 = h * h;
        for s in &self.springs {
            let d = x[s.b as usize] - x[s.a as usize];
            let len = d.length().max(1.0e-12);
            let dir = d / len;
            // ∂E/∂x_b = +h²k(len − rest)·d̂ ; ∂E/∂x_a is the negation.
            let mut f = h2 * s.k * (len - s.rest);
            if self.params.strain_limit.is_some() {
                // gap = limit − len/rest ⇒ ∂gap/∂len = −1/rest, so the
                // barrier ADDS +(−b′)/rest of tension along d̂.
                let (_, db, _) = self.strain_barrier_terms(s, len / s.rest);
                f -= h2 * db / s.rest;
            }
            out[s.a as usize] -= dir * f;
            out[s.b as usize] += dir * f;
        }
        for (t, &v0) in self.tets.iter().zip(self.tet_v0.iter()) {
            let dv = Self::tet_volume(x, t.ids) - v0;
            if dv != 0.0 {
                let g = Self::tet_volume_grads(x, t.ids);
                for (k, gk) in g.iter().enumerate() {
                    out[t.ids[k] as usize] += *gk * (h2 * t.kv * dv);
                }
            }
        }
        for (i, xi) in x.iter().enumerate() {
            if self.pinned[i] {
                continue;
            }
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
        let ew = self.params.friction_epsv * h;
        for fc in &self.lagged_friction {
            let i = fc.vertex as usize;
            let u = x[i] - fc.anchor;
            let ut = u - fc.n * fc.n.dot(u);
            let s = ut.length();
            if s > 1.0e-15 {
                let (g, _) = self.friction_g(fc.lambda, s, ew);
                out[i] += (ut / s) * g;
            }
        }
        for pf in &self.lagged_pair_friction {
            let ut = self.pair_slip(pf, x);
            let s = ut.length();
            if s > 1.0e-15 {
                let (g, _) = self.friction_g(pf.lambda, s, ew);
                let dir = ut / s;
                for (idx, w) in pf.stencil {
                    out[idx as usize] += dir * (g * w);
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
            if self.proxy_of[i].is_none() {
                y[i] += v[i] * self.masses[i];
            }
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
            let mut jdv = (dv - dv_par) * (h2 * s.k * t) + dv_par * (h2 * s.k);
            if self.params.strain_limit.is_some() {
                // Convex part of the strain barrier: b″/rest² along
                // d̂d̂ᵀ (the b′ curvature term is dropped — same
                // projection discipline as contact).
                let (_, _, ddb) = self.strain_barrier_terms(s, len / s.rest);
                if ddb > 0.0 {
                    jdv += dv_par * (h2 * ddb / (s.rest * s.rest));
                }
            }
            y[s.a as usize] -= jdv;
            y[s.b as usize] += jdv;
        }
        for t in &self.tets {
            // Gauss (SPD) part only: k_v·g·gᵀ — the geometric part
            // k_v(V−V₀)·∂g/∂x is indefinite and projected away, same
            // discipline as every other Hessian here.
            let g = Self::tet_volume_grads(x, t.ids);
            let mut s_acc = 0.0;
            for (k, gk) in g.iter().enumerate() {
                s_acc += gk.dot(v[t.ids[k] as usize]);
            }
            for (k, gk) in g.iter().enumerate() {
                y[t.ids[k] as usize] += *gk * (h2 * t.kv * s_acc);
            }
        }
        for (i, xi) in x.iter().enumerate() {
            if self.pinned[i] {
                continue;
            }
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
        let ew = self.params.friction_epsv * h;
        for fc in &self.lagged_friction {
            let i = fc.vertex as usize;
            let u = x[i] - fc.anchor;
            let ut = u - fc.n * fc.n.dot(u);
            let s = ut.length();
            let vt = v[i] - fc.n * fc.n.dot(v[i]);
            if s > 1.0e-15 {
                let (g, gp) = self.friction_g(fc.lambda, s, ew);
                let uhat = ut / s;
                let v_par = uhat * uhat.dot(vt);
                // H = g′₊·ûûᵀ + (g/s)·(T − ûûᵀ) on the tangent plane
                // T = I − nnᵀ; both coefficients are ≥ 0 (g′ already
                // projected in friction_g).
                y[i] += v_par * gp + (vt - v_par) * (g / s);
            } else {
                // s → 0 limit: (μ_s λ / e)·T.
                let mu_s = self.params.friction_mu * self.params.friction_mu_s_scale.max(1.0);
                y[i] += vt * (mu_s * fc.lambda / ew);
            }
        }
        for pf in &self.lagged_pair_friction {
            // Relative-slip chain rule: v_rel = T·Σ w_j v_j, block
            // H_jk = w_j w_k · H_scalar — apply as gather+scatter.
            let mut vrel = Vec3::ZERO;
            for (k, (idx, w)) in pf.stencil.iter().enumerate() {
                let _ = k;
                vrel += v[*idx as usize] * *w;
            }
            let vt = vrel - pf.n * pf.n.dot(vrel);
            let ut = self.pair_slip(pf, x);
            let s = ut.length();
            let hv = if s > 1.0e-15 {
                let (g, gp) = self.friction_g(pf.lambda, s, ew);
                let uhat = ut / s;
                let v_par = uhat * uhat.dot(vt);
                v_par * gp + (vt - v_par) * (g / s)
            } else {
                let mu_s = self.params.friction_mu * self.params.friction_mu_s_scale.max(1.0);
                vt * (mu_s * pf.lambda / ew)
            };
            for (idx, w) in pf.stencil {
                y[idx as usize] += hv * w;
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
        // (Pinned vertices never move — p is zero on their rows — so
        // their below-surface anchors cannot truncate the step.)
        // Global travel cap: no vertex moves more than d̂ in one
        // accepted line-search step. This is what makes the 3·d̂
        // candidate margin airtight (see `self_candidates`).
        if !self.triangles.is_empty() {
            let p_max = p.iter().map(|q| q.length()).fold(0.0, Real::max);
            if p_max > 0.0 {
                alpha = alpha.min(self.params.dhat / p_max);
            }
        }
        // HARD strain bound: truncate any step that would cross the
        // limit. Conservative: |Δstrain| ≤ (|p_a| + |p_b|)/rest.
        if let Some(limit) = self.params.strain_limit {
            for s in &self.springs {
                let len = (x[s.b as usize] - x[s.a as usize]).length();
                let gap = limit - len / s.rest;
                let rate = (p[s.a as usize].length() + p[s.b as usize].length()) / s.rest;
                if rate > 0.0 && gap > 0.0 {
                    alpha = alpha.min(0.9 * gap / rate);
                }
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

    /// Rebuild the lagged friction state from the CURRENT positions
    /// (called by [`Self::step`]; public so FD tests can pin the lag).
    pub fn rebuild_friction_lag(&mut self, h: Real) {
        self.lagged_friction.clear();
        self.lagged_pair_friction.clear();
        if self.params.friction_mu <= 0.0 {
            return;
        }
        // Self-pair friction: freeze the pair set, frames and λ at the
        // committed positions.
        if !self.triangles.is_empty() {
            let x = self.positions.clone();
            for pair in self.self_candidates(&x) {
                let (diff, stencil) = self.pair_geometry(&pair, &x);
                let dist = diff.length();
                if dist >= self.params.dhat || dist <= 1.0e-12 {
                    continue;
                }
                let s = self.params.dhat - dist;
                // λ = h²·(−b′) with the SAME pair κ the barrier uses.
                let lambda = h * h * self.pair_kappa(&pair, h) * s * s
                    / (self.params.dhat * self.params.dhat);
                let mut anchors = [Vec3::ZERO; 4];
                for (k, (idx, _)) in stencil.iter().enumerate() {
                    anchors[k] = x[*idx as usize];
                }
                self.lagged_pair_friction.push(PairFrictionContact {
                    stencil,
                    n: diff / dist,
                    anchors,
                    lambda,
                });
            }
        }
        // λ needs a step scale; use the same h the barrier κ uses
        // implicitly — the caller's h is not known here, so store λ
        // in h-free units (−b′·h² has κ ∝ 1/h², cancelling h²): with
        // κ = kappa_scale·m/h², h²·(−b′) = kappa_scale·m·(d̂−g)²/d̂².
        for (i, xi) in self.positions.iter().enumerate() {
            if self.pinned[i] {
                continue;
            }
            for (o_idx, ob) in self.obstacles.iter().enumerate() {
                let (gap, n) = ob.distance(*xi);
                if gap < self.params.dhat {
                    let s = (self.params.dhat - gap).max(0.0);
                    // λ = h²·(−b′) with THE SAME κ the barrier uses —
                    // a mass-only λ under the elasticity-inclusive
                    // barrier underestimates the true normal force by
                    // the (m + h²k_sum)/m factor and friction goes
                    // proportionally weak (measured: the incline
                    // discriminator slid in BOTH cases).
                    let lambda = h * h * self.barrier_kappa(i, h) * s * s
                        / (self.params.dhat * self.params.dhat);
                    self.lagged_friction.push(FrictionContact {
                        vertex: i as u32,
                        obstacle: o_idx as u32,
                        n,
                        anchor: *xi,
                        lambda,
                    });
                }
            }
        }
    }

    /// Relative tangential slip of a lagged pair at `x`.
    fn pair_slip(&self, pf: &PairFrictionContact, x: &[Vec3]) -> Vec3 {
        let mut u = Vec3::ZERO;
        for (k, (idx, w)) in pf.stencil.iter().enumerate() {
            u += (x[*idx as usize] - pf.anchors[k]) * *w;
        }
        u - pf.n * pf.n.dot(u)
    }

    /// Friction force-magnitude profile g(s) and its (projected)
    /// slope g′₊(s) at tangential slip s, with stick well e:
    /// `s ≤ e`: ramp to μ_s·λ; `e < s ≤ 2e`: C⁰ blend down to the
    /// kinetic plateau; `s > 2e`: μ_k·λ. The blend slope is negative
    /// (the stick-slip drop) and is PROJECTED to 0 in the Hessian —
    /// the same discipline as every other indefinite term here.
    fn friction_g(&self, lambda: Real, s: Real, e: Real) -> (Real, Real) {
        let mu_k = self.params.friction_mu;
        let mu_s = mu_k * self.params.friction_mu_s_scale.max(1.0);
        if s <= e {
            let slope = mu_s * lambda / e;
            (slope * s, slope)
        } else if s <= 2.0 * e {
            let g = mu_s * lambda + (mu_k - mu_s) * lambda * (s - e) / e;
            (g, 0.0)
        } else {
            (mu_k * lambda, 0.0)
        }
    }

    /// ∫g — the friction potential at slip s.
    fn friction_energy_of(&self, lambda: Real, s: Real, e: Real) -> Real {
        let mu_k = self.params.friction_mu;
        let mu_s = mu_k * self.params.friction_mu_s_scale.max(1.0);
        let d_stick = |t: Real| mu_s * lambda * t * t / (2.0 * e);
        if s <= e {
            d_stick(s)
        } else if s <= 2.0 * e {
            let t = s - e;
            d_stick(e) + mu_s * lambda * t + (mu_k - mu_s) * lambda * t * t / (2.0 * e)
        } else {
            let t2 = e;
            d_stick(e)
                + mu_s * lambda * t2
                + (mu_k - mu_s) * lambda * t2 * t2 / (2.0 * e)
                + mu_k * lambda * (s - 2.0 * e)
        }
    }

    /// Physical reaction wrench each obstacle receives from the shell
    /// at the CURRENT positions: per obstacle, (Σ force, Σ torque
    /// about the WORLD ORIGIN) in Newtons / N·m. Includes the barrier
    /// normal forces and the lagged friction forces. Because implicit
    /// Euler gives the shell exactly `h·F(x_new)` of impulse, sampling
    /// the same final-state forces here makes the exchange satisfy
    /// Newton's third law to Newton-solver tolerance — the weak
    /// rigid-coupling contract.
    pub fn obstacle_reactions(&self, h: Real) -> Vec<(Vec3, Vec3)> {
        let mut out = vec![(Vec3::ZERO, Vec3::ZERO); self.obstacles.len()];
        for (i, xi) in self.positions.iter().enumerate() {
            if self.pinned[i] {
                continue;
            }
            let kappa = self.barrier_kappa(i, h);
            for (o, ob) in self.obstacles.iter().enumerate() {
                let (gap, n) = ob.distance(*xi);
                let (_, db, _) = self.barrier(gap, kappa);
                if db != 0.0 {
                    // Force ON the vertex is (−b′)·n; the obstacle
                    // receives the negation.
                    let f_obstacle = n * db; // db < 0 ⇒ pushes obstacle away
                    out[o].0 += f_obstacle;
                    out[o].1 += xi.cross(f_obstacle);
                }
            }
        }
        // Friction: lagged contacts know their obstacle only through
        // the anchor frame; re-associate by matching normals is
        // fragile, so the lag records the obstacle index.
        let ew = self.params.friction_epsv * h;
        for fc in &self.lagged_friction {
            let i = fc.vertex as usize;
            let u = self.positions[i] - fc.anchor;
            let ut = u - fc.n * fc.n.dot(u);
            let s = ut.length();
            if s > 1.0e-15 {
                let (g, _) = self.friction_g(fc.lambda, s, ew);
                // Friction force on the VERTEX is −(g/h²)·û (energy
                // has no h² prefix); the obstacle receives +.
                let f_obstacle = (ut / s) * (g / (h * h));
                let o = fc.obstacle as usize;
                out[o].0 += f_obstacle;
                out[o].1 += self.positions[i].cross(f_obstacle);
            }
        }
        out
    }

    /// Largest spring strain (len/rest) — the strain-limit invariant
    /// counterpart to [`Self::min_gap`].
    pub fn max_strain(&self) -> Real {
        self.springs
            .iter()
            .map(|s| {
                (self.positions[s.b as usize] - self.positions[s.a as usize]).length() / s.rest
            })
            .fold(0.0, Real::max)
    }

    /// Smallest gap between any vertex and any obstacle — THE
    /// invariant. Positive at all times or the solver is wrong.
    pub fn min_gap(&self) -> Real {
        let mut g = Real::INFINITY;
        for (i, xi) in self.positions.iter().enumerate() {
            if self.pinned[i] {
                continue; // boundary condition, not a contact
            }
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
                if tri.contains(&v) || self.same_body(v, tri[0]) {
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
                if self.same_body(e1[0], e2[0]) {
                    continue;
                }
                if self.chain_neighbors(i, j) {
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

    /// Expand a composite direction (cloth verts free, proxies
    /// derived from their body's (dp, dA)) into the full vertex array.
    fn abd_expand(&self, dv: &[Vec3], dbody: &[(Vec3, Mat3)], out: &mut [Vec3]) {
        out.copy_from_slice(dv);
        for (bi, body) in self.affine_bodies.iter().enumerate() {
            let (dp, da) = dbody[bi];
            for k in 0..body.proxy_count {
                out[body.proxy_start + k] = da * body.rest[k] + dp;
            }
        }
    }

    /// Reduce per-proxy vectors into per-body 12-DOF blocks:
    /// (Σ yᵢ, Σ yᵢ x̄ᵢᵀ).
    fn abd_reduce(&self, y: &[Vec3]) -> Vec<(Vec3, Mat3)> {
        self.affine_bodies
            .iter()
            .map(|body| {
                let mut fp = Vec3::ZERO;
                let mut fa = Mat3::ZERO;
                for k in 0..body.proxy_count {
                    let yi = y[body.proxy_start + k];
                    fp += yi;
                    let xb = body.rest[k];
                    fa += Mat3::from_cols(yi * xb.x, yi * xb.y, yi * xb.z);
                }
                (fp, fa)
            })
            .collect()
    }

    /// Body-side incremental energy (inertia + h²·orthogonality) for
    /// trial poses.
    fn abd_energy(&self, poses: &[(Vec3, Mat3)], tilde: &[(Vec3, Mat3)], h: Real) -> Real {
        let mut e = 0.0;
        for (bi, body) in self.affine_bodies.iter().enumerate() {
            let (p, a) = poses[bi];
            let (pt, at) = tilde[bi];
            e += 0.5 * body.mass * (p - pt).length_squared();
            let da = a - at;
            // ½ tr(ΔA J ΔAᵀ) — affine rotational inertia.
            let daj = da * body.j_mat;
            e += 0.5
                * (daj.col(0).dot(da.col(0))
                    + daj.col(1).dot(da.col(1))
                    + daj.col(2).dot(da.col(2)));
            let (eo, _) = Self::ortho_terms(a, body.kappa_ortho);
            e += h * h * eo;
        }
        e
    }

    /// One implicit step of size `h`.
    pub fn step(&mut self, h: Real) -> Result<BarrierStepStats> {
        if !self.affine_bodies.is_empty() {
            return self.step_abd(h);
        }
        let n = self.positions.len();
        self.rebuild_friction_lag(h);
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
                for (i, xi) in x.iter().enumerate() {
                    if self.pinned[i] {
                        continue;
                    }
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

    /// The ABD-coupled Newton step: ONE incremental potential over
    /// cloth vertices AND affine-body DOFs — barriers/pairs/friction
    /// evaluated on the derived proxy vertices, directions expanded /
    /// forces reduced through the linear map, the SAME CCD-truncated
    /// line search protecting everything (this is what removes the
    /// weak path's rigid-side TOI patch: a fast body simply shortens
    /// the unified step).
    fn step_abd(&mut self, h: Real) -> Result<BarrierStepStats> {
        let n = self.positions.len();
        self.rebuild_friction_lag(h);
        let x_prev = self.positions.clone();
        let bodies_prev: Vec<(Vec3, Mat3)> =
            self.affine_bodies.iter().map(|b| (b.p, b.a_mat)).collect();
        let x_tilde: Vec<Vec3> = (0..n)
            .map(|i| {
                if self.pinned[i] || self.proxy_of[i].is_some() {
                    x_prev[i]
                } else {
                    x_prev[i] + self.velocities[i] * h + self.params.gravity * (h * h)
                }
            })
            .collect();
        let body_tilde: Vec<(Vec3, Mat3)> = self
            .affine_bodies
            .iter()
            .map(|b| {
                (
                    b.p + b.v_p * h + self.params.gravity * (h * h),
                    b.a_mat + b.v_a * h,
                )
            })
            .collect();

        let mut x = x_prev.clone();
        let mut poses = bodies_prev.clone();
        let mut grad = vec![Vec3::ZERO; n];
        let mut stats = BarrierStepStats {
            min_gap: self.min_gap(),
            ..Default::default()
        };
        let m_max = self.masses.iter().cloned().fold(0.0, Real::max);
        let tol = self.params.newton_tol * (m_max * self.params.dhat).max(1.0e-30);
        let nb = self.affine_bodies.len();
        let j_inv: Vec<Mat3> = self
            .affine_bodies
            .iter()
            .map(|b| b.j_mat.inverse())
            .collect();

        for it in 0..self.params.max_newton_iters {
            // Sync proxies from poses (exact — linear map).
            for (bi, body) in self.affine_bodies.iter().enumerate() {
                let (p, a) = poses[bi];
                for k in 0..body.proxy_count {
                    x[body.proxy_start + k] = a * body.rest[k] + p;
                }
            }
            let pairs = self.self_candidates(&x);
            self.gradient(&x, &x_tilde, h, &pairs, &mut grad);
            // Body gradients: reduced proxy forces + inertia + ortho.
            let mut gbody = self.abd_reduce(&grad);
            for bi in 0..nb {
                let body = &self.affine_bodies[bi];
                let (p, a) = poses[bi];
                let (pt, at) = body_tilde[bi];
                gbody[bi].0 += (p - pt) * body.mass;
                gbody[bi].1 += (a - at) * body.j_mat;
                let (_, go) = Self::ortho_terms(a, body.kappa_ortho);
                gbody[bi].1 += go * (h * h);
            }
            let mut g_inf: Real = 0.0;
            for (i, g) in grad.iter().enumerate() {
                if self.proxy_of[i].is_none() {
                    g_inf = g_inf.max(g.x.abs().max(g.y.abs()).max(g.z.abs()));
                }
            }
            for (gp, ga) in &gbody {
                g_inf = g_inf.max(gp.x.abs().max(gp.y.abs()).max(gp.z.abs()));
                for c in 0..3 {
                    let col = ga.col(c);
                    g_inf = g_inf.max(col.x.abs().max(col.y.abs()).max(col.z.abs()));
                }
            }
            if g_inf <= tol {
                stats.newton_iters = it;
                break;
            }
            stats.newton_iters = it + 1;

            // ── composite PCG on H·d = −g.
            let mut dv = vec![Vec3::ZERO; n];
            let mut dbody = vec![(Vec3::ZERO, Mat3::ZERO); nb];
            {
                let mut r_v: Vec<Vec3> = grad
                    .iter()
                    .enumerate()
                    .map(|(i, g)| {
                        if self.proxy_of[i].is_none() {
                            -*g
                        } else {
                            Vec3::ZERO
                        }
                    })
                    .collect();
                let mut r_b: Vec<(Vec3, Mat3)> =
                    gbody.iter().map(|(p, a)| (-*p, *a * -1.0)).collect();
                let prec = |rv: &[Vec3], rb: &[(Vec3, Mat3)]| -> (Vec<Vec3>, Vec<(Vec3, Mat3)>) {
                    let zv = rv
                        .iter()
                        .enumerate()
                        .map(|(i, r)| {
                            if self.proxy_of[i].is_none() {
                                *r / self.masses[i]
                            } else {
                                Vec3::ZERO
                            }
                        })
                        .collect();
                    let zb = rb
                        .iter()
                        .enumerate()
                        .map(|(bi, (rp, ra))| (*rp / self.affine_bodies[bi].mass, *ra * j_inv[bi]))
                        .collect();
                    (zv, zb)
                };
                let dot =
                    |av: &[Vec3], ab: &[(Vec3, Mat3)], bv: &[Vec3], bb: &[(Vec3, Mat3)]| -> Real {
                        let mut s: Real = av.iter().zip(bv.iter()).map(|(a, b)| a.dot(*b)).sum();
                        for ((ap, aa), (bp, ba)) in ab.iter().zip(bb.iter()) {
                            s += ap.dot(*bp);
                            for c in 0..3 {
                                s += aa.col(c).dot(ba.col(c));
                            }
                        }
                        s
                    };
                let (mut z_v, mut z_b) = prec(&r_v, &r_b);
                let mut d_v = z_v.clone();
                let mut d_b = z_b.clone();
                let mut rz = dot(&r_v, &r_b, &z_v, &z_b);
                let r0 = dot(&r_v, &r_b, &r_v, &r_b);
                let mut full = vec![Vec3::ZERO; n];
                let mut hy = vec![Vec3::ZERO; n];
                if r0 > 0.0 {
                    for _ in 0..self.params.max_cg_iters {
                        // H·d: expand, per-vertex Hessian, reduce, add
                        // body inertia + ortho blocks.
                        self.abd_expand(&d_v, &d_b, &mut full);
                        self.hessian_mul(&x, h, &pairs, &full, &mut hy);
                        let mut hb = self.abd_reduce(&hy);
                        for bi in 0..nb {
                            let body = &self.affine_bodies[bi];
                            let (dp, da) = d_b[bi];
                            hb[bi].0 += dp * body.mass;
                            hb[bi].1 += da * body.j_mat;
                            hb[bi].1 +=
                                Self::ortho_hess_mul(poses[bi].1, da, body.kappa_ortho) * (h * h);
                        }
                        let hv: Vec<Vec3> = hy
                            .iter()
                            .enumerate()
                            .map(|(i, y)| {
                                if self.proxy_of[i].is_none() {
                                    *y
                                } else {
                                    Vec3::ZERO
                                }
                            })
                            .collect();
                        let dhd = dot(&d_v, &d_b, &hv, &hb);
                        if dhd <= 0.0 {
                            break;
                        }
                        let alpha = rz / dhd;
                        for i in 0..n {
                            dv[i] += d_v[i] * alpha;
                            r_v[i] -= hv[i] * alpha;
                        }
                        for bi in 0..nb {
                            dbody[bi].0 += d_b[bi].0 * alpha;
                            dbody[bi].1 += d_b[bi].1 * alpha;
                            r_b[bi].0 -= hb[bi].0 * alpha;
                            r_b[bi].1 -= hb[bi].1 * alpha;
                        }
                        if dot(&r_v, &r_b, &r_v, &r_b)
                            <= self.params.cg_tol * self.params.cg_tol * r0
                        {
                            break;
                        }
                        let (z2v, z2b) = prec(&r_v, &r_b);
                        z_v = z2v;
                        z_b = z2b;
                        let rz_new = dot(&r_v, &r_b, &z_v, &z_b);
                        let beta = rz_new / rz;
                        rz = rz_new;
                        for i in 0..n {
                            d_v[i] = z_v[i] + d_v[i] * beta;
                        }
                        for bi in 0..nb {
                            d_b[bi].0 = z_b[bi].0 + d_b[bi].0 * beta;
                            d_b[bi].1 = z_b[bi].1 + d_b[bi].1 * beta;
                        }
                    }
                }
            }

            // ── unified CCD-truncated Armijo line search.
            let mut full_dir = vec![Vec3::ZERO; n];
            self.abd_expand(&dv, &dbody, &mut full_dir);
            let alpha_max = self.ccd_alpha(&x, &full_dir, &pairs);
            let e0 = self.energy(&x, &x_tilde, h, &pairs) + self.abd_energy(&poses, &body_tilde, h);
            let slope: Real = grad
                .iter()
                .zip(full_dir.iter())
                .enumerate()
                .filter(|&(i, _)| self.proxy_of[i].is_none())
                .map(|(_, (g, d))| g.dot(*d))
                .sum::<Real>()
                + gbody
                    .iter()
                    .zip(dbody.iter())
                    .map(|((gp, ga), (dp, da))| {
                        gp.dot(*dp) + (0..3).map(|c| ga.col(c).dot(da.col(c))).sum::<Real>()
                    })
                    .sum::<Real>();
            let mut alpha = alpha_max;
            let mut accepted = false;
            for _ in 0..30 {
                let x_try: Vec<Vec3> = (0..n).map(|i| x[i] + full_dir[i] * alpha).collect();
                let poses_try: Vec<(Vec3, Mat3)> = poses
                    .iter()
                    .zip(dbody.iter())
                    .map(|((p, a), (dp, da))| (*p + *dp * alpha, *a + *da * alpha))
                    .collect();
                let e_try = self.energy(&x_try, &x_tilde, h, &pairs)
                    + self.abd_energy(&poses_try, &body_tilde, h);
                if e_try <= e0 + 1.0e-4 * alpha * slope {
                    x = x_try;
                    poses = poses_try;
                    accepted = true;
                    break;
                }
                alpha *= 0.5;
            }
            if !accepted {
                stats.hit_iter_cap = true;
                break;
            }
            let gap_now = {
                let mut g = Real::INFINITY;
                for (i, xi) in x.iter().enumerate() {
                    if self.pinned[i] {
                        continue;
                    }
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

        stats.energy = self.energy(&x, &x_tilde, h, &self.self_candidates(&x))
            + self.abd_energy(&poses, &body_tilde, h);
        for i in 0..n {
            self.velocities[i] = if self.pinned[i] {
                Vec3::ZERO
            } else {
                (x[i] - x_prev[i]) / h
            };
        }
        for (bi, body) in self.affine_bodies.iter_mut().enumerate() {
            let (p, a) = poses[bi];
            let (pp, ap) = bodies_prev[bi];
            body.v_p = (p - pp) / h;
            body.v_a = (a - ap) * (1.0 / h);
            body.p = p;
            body.a_mat = a;
        }
        self.positions = x;
        Ok(stats)
    }
}
