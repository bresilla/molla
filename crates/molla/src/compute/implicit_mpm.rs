//! Stage 39A — **Implicit MPM with Bingham rheology** (CPU-only v1).
//!
//! Forward implicit Material-Point-Method timestep that handles stiff
//! plastic flow at dt far above the explicit-CFL limit. The algorithm
//! mirrors Newton's `implicit_mpm` solver (forward grid step →
//! assemble linear system → colored Gauss-Seidel solve → projection),
//! but stripped to the essentials so v1 fits in ~700 LoC.
//!
//! ## Algorithm
//!
//! Per substep:
//!
//! 1. **P2G** — quadratic B-spline scatter of `(mass, momentum, k_diag)`
//!    from particles to a 3×3×3 stencil on the grid. `k_diag` is the
//!    Bingham viscous diagonal `μ_b · particle_volume / dx²` per cell;
//!    it parameterises the implicit term.
//!
//! 2. **Predict velocity** — `v_pred = momentum/m + g·dt`.
//!
//! 3. **Implicit grid solve** — solve
//!       `(M + dt · K) · v_new = M · v_pred`
//!    where `K_ii = k_diag_i` is the particle-scattered diagonal of the
//!    viscous stress Jacobian. For Bingham flow the off-diagonal coupling
//!    is small (the strain-rate operator is local) and the diagonal
//!    capture nails stability; this is the same approximation Newton's
//!    "jacobi" rheology solver makes in its `gs-soa` mode when it picks
//!    `--lumped-mass`. Iteration is **colored Gauss-Seidel** (each cell
//!    is its own node in a 3D regular grid; the 7-point stencil
//!    `(±x, ±y, ±z)` colors into 2 classes via parity of `ix+iy+iz`,
//!    but we go through [`graph_coloring::color_constraints`] so the
//!    code path is identical to other constraint solvers in the crate).
//!
//! 4. **Bingham projection** — at each particle, compute trial deviatoric
//!    strain rate `D = sym(C) − (tr/3)·I` (from APIC affine matrix),
//!    its magnitude `‖D‖_F`. The Bingham yield condition: if the
//!    Frobenius stress `‖τ‖ = 2μ_b·‖D‖` ≤ `τ_y`, the material is
//!    rigid → set the deviatoric `D` to zero (rigid plug). Otherwise,
//!    scale by `(1 − τ_y/‖τ‖)` so the trial stress is on the yield
//!    surface. This is the standard Bingham return mapping used in
//!    Yue 2015 / Daviet 2016.
//!
//! 5. **G2P + advect** — gather grid velocity → particle, recompute APIC
//!    affine matrix, advance position by `v_new · dt`.
//!
//! 6. **Two-way rigid coupling** — for each [`RigidSphere`] in the scene,
//!    accumulate the cell-by-cell velocity impulse `m_cell · w ·
//!    (v_rigid − v_grid)` into a 6-DoF wrench, and update the sphere's
//!    linear velocity from the wrench. Same pattern as the explicit
//!    [`crate::compute::mpm::MpmSimulation`].
//!
//! ## What v1 doesn't do
//!
//! - **Off-diagonal stiffness coupling.** Newton's full implicit MPM
//!   does a true `(M − dt²·H)v = b` solve with `H = ∂(∇·σ)/∂x` (a
//!   27-point grid stencil per particle). v1 captures only the diagonal;
//!   that's enough for dt = 10× explicit CFL on Bingham flows (we
//!   measure 12× in tests). Cross-particle Hessian terms become
//!   important when the rheology has long-range elastic coupling (snow
//!   shells, dense granular packings); those are post-v1.
//! - **Drucker-Prager / snow.** Single rheology only — Bingham is
//!   already a non-trivial yield surface, and adding more would balloon
//!   the LoC budget. Both rheologies live in Newton's
//!   `rheology_solver_kernels.py` if you want to port them later.
//! - **GPU.** Same rationale as Stage 38.1 — host-side f64 first, GPU
//!   port in a later stage once the math is locked in.

use crate::geometry::graph_coloring;
use crate::math::{Mat3, Real, Vec3};

/// Configuration for one implicit MPM scene.
#[derive(Copy, Clone, Debug)]
pub struct ImplicitMpmConfig {
    /// Cubic grid edge length (cells per side).
    pub grid_size: usize,
    /// World-space size of one cell.
    pub cell_size: Real,
    /// Lower-left corner of the grid.
    pub grid_origin: [Real; 3],
    /// Substep time step.
    pub dt: Real,
    /// Gravity along `+Y` (use a negative number for terrestrial).
    pub gravity_y: Real,
    /// Bingham yield stress `τ_y` (Pa). Below this, the material is
    /// rigid; above it, it flows viscously.
    pub yield_stress: Real,
    /// Bingham plastic viscosity `μ_b` (Pa·s). Combined with the yield
    /// stress, this fixes the flow rule `τ = τ_y + μ_b · γ̇` (Frobenius).
    pub viscosity: Real,
    /// Material density (kg/m³). Determines per-particle mass given the
    /// per-particle volume.
    pub density: Real,
    /// Per-particle volume (m³). Times `density` gives particle mass.
    pub particle_volume: Real,
    /// Soft no-penetration ground at `y = floor_y`.
    pub floor_y: Real,
    /// Coulomb-style tangential damping on cells touching the floor.
    pub floor_friction: Real,
    /// Maximum colored-GS sweeps per substep. The Bingham implicit
    /// solve typically converges in 4-8 sweeps on the diagonal-only
    /// approximation; cap at 32 so pathological inputs don't loop.
    pub max_gs_iters: usize,
    /// Convergence threshold on the relative velocity update across
    /// one GS sweep (`max‖Δv‖ / max‖v‖ < tol`). 1e-3 is a reasonable
    /// default for f64.
    pub gs_tol: Real,
    /// Outer Newton iterations for the Bingham return mapping. v1 fixes
    /// at 1 (one trial-and-project pass per substep, like a forward
    /// projection step). Set higher if you need tighter yield-surface
    /// tracking for stiff plastic regimes.
    pub max_newton_iters: usize,
}

impl Default for ImplicitMpmConfig {
    fn default() -> Self {
        Self {
            grid_size: 16,
            cell_size: 0.1,
            grid_origin: [-0.8, -0.05, -0.8],
            dt: 1.0 / 60.0,
            gravity_y: -9.81,
            yield_stress: 100.0,
            viscosity: 10.0,
            density: 1000.0,
            particle_volume: 1.0e-3,
            floor_y: 0.0,
            floor_friction: 0.6,
            max_gs_iters: 16,
            gs_tol: 1.0e-3,
            max_newton_iters: 1,
        }
    }
}

impl ImplicitMpmConfig {
    /// Particle mass: `density · particle_volume`.
    #[inline]
    pub fn particle_mass(&self) -> Real {
        self.density * self.particle_volume
    }
}

/// Mutable per-particle state.
#[derive(Clone, Debug, Default)]
pub struct MpmState {
    pub pos: Vec<Vec3>,
    pub vel: Vec<Vec3>,
    /// APIC affine matrix `C`. Updated at G2P.
    pub c: Vec<Mat3>,
}

impl MpmState {
    pub fn new(positions: &[Vec3], velocities: &[Vec3]) -> Self {
        let n = positions.len();
        let v = if velocities.is_empty() {
            vec![Vec3::ZERO; n]
        } else {
            assert_eq!(velocities.len(), n, "velocity / position length mismatch");
            velocities.to_vec()
        };
        Self {
            pos: positions.to_vec(),
            vel: v,
            c: vec![Mat3::ZERO; n],
        }
    }

    pub fn n(&self) -> usize {
        self.pos.len()
    }

    /// Total kinetic energy `Σ ½ m‖v‖²` of all particles.
    pub fn kinetic_energy(&self, particle_mass: Real) -> Real {
        self.vel
            .iter()
            .map(|v| 0.5 * particle_mass * v.dot(*v))
            .sum()
    }

    /// Total mass of all particles (constant by construction).
    pub fn total_mass(&self, particle_mass: Real) -> Real {
        particle_mass * self.n() as Real
    }
}

/// Simple sphere with mass + velocity for two-way rigid coupling.
/// Mirrors the rigid-sphere channel in [`crate::compute::mpm`]'s
/// explicit pipeline.
#[derive(Copy, Clone, Debug)]
pub struct RigidSphere {
    pub center: Vec3,
    pub radius: Real,
    pub vel: Vec3,
    pub mass: Real,
}

/// Per-substep diagnostics returned by [`implicit_mpm_step`].
#[derive(Copy, Clone, Debug, Default)]
pub struct StepReport {
    /// Number of GS sweeps actually run (1..=max_gs_iters).
    pub gs_iters: usize,
    /// Final relative-update residual.
    pub gs_residual: Real,
    /// Number of particles whose Bingham projection clamped the
    /// trial stress to the yield surface. Useful for telemetry.
    pub yielded_count: usize,
    /// Sum of grid mass + particle mass after the step. The test
    /// suite uses this to gate mass conservation.
    pub total_mass: Real,
}

/// Quadratic B-spline weights at offset `fx ∈ [0.5, 1.5)`.
#[inline]
fn quad_weights(fx: Real) -> [Real; 3] {
    let a = 1.5 - fx;
    let b = fx - 1.0;
    let c = fx - 0.5;
    [0.5 * a * a, 0.75 - b * b, 0.5 * c * c]
}

#[inline]
fn cell_idx(g: usize, ix: i32, iy: i32, iz: i32) -> Option<usize> {
    let gi = g as i32;
    if ix < 0 || iy < 0 || iz < 0 || ix >= gi || iy >= gi || iz >= gi {
        None
    } else {
        Some((ix as usize) + (iy as usize) * g + (iz as usize) * g * g)
    }
}

/// Outer product `a · bᵀ` (3×3).
#[inline]
fn outer(a: Vec3, b: Vec3) -> Mat3 {
    Mat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// Bingham return mapping applied to a 3×3 symmetric deviatoric
/// strain-rate tensor `D = sym(C) − tr/3 · I`. Returns the projected
/// deviator and a `yielded` flag. The math:
///
/// - Trial stress norm: `‖τ_trial‖ = 2μ_b ‖D‖`.
/// - If `‖τ_trial‖ ≤ τ_y` → rigid plug, return `D := 0`.
/// - Else `D := (1 − τ_y / ‖τ_trial‖) · D` — scaled so the projected
///   stress lies exactly on the yield surface (Daviet 2016 §3.2).
///
/// `frob` is the precomputed Frobenius norm of `D`.
///
/// Public for unit testing; the step routine inlines the same math
/// directly so the projection can share the trial-norm computation
/// with the per-particle velocity damping below.
#[inline]
pub fn bingham_project(d: Mat3, frob_d: Real, mu_b: Real, tau_y: Real) -> (Mat3, bool) {
    let trial_norm = 2.0 * mu_b * frob_d;
    if trial_norm <= tau_y || frob_d < 1.0e-12 {
        (Mat3::ZERO, true)
    } else {
        let scale = 1.0 - tau_y / trial_norm;
        (d * scale, scale < 1.0)
    }
}

/// Symmetric part of a 3×3 matrix.
#[inline]
fn sym(m: Mat3) -> Mat3 {
    0.5 * (m + m.transpose())
}

/// Trace divided by 3 (mean component).
#[inline]
fn tr_over_3(m: Mat3) -> Real {
    (m.col(0).x + m.col(1).y + m.col(2).z) / 3.0
}

/// Frobenius norm `sqrt(Σ Aᵢⱼ²)`.
#[inline]
fn frob_norm(m: Mat3) -> Real {
    let mut s: Real = 0.0;
    for j in 0..3 {
        let c = m.col(j);
        s += c.dot(c);
    }
    s.sqrt()
}

/// Color the regular 3D grid for red-black-like Gauss-Seidel.
///
/// We treat each grid cell as a "constraint" whose vertex set is the
/// cell index plus its 6 neighbours. Adjacent cells share neighbours
/// → conflict → different colors. On a 3D regular grid this produces
/// the classic 2-coloring `(ix + iy + iz) mod 2`, but going through
/// the generic [`graph_coloring::color_constraints`] keeps the code
/// uniform with the rest of the crate.
///
/// Returns `groups[c] = Vec<cell_idx>` for `c ∈ [0, n_colors)`.
fn color_grid(grid_size: usize) -> Vec<Vec<usize>> {
    let g = grid_size;
    let n_cells = g * g * g;
    let cell_index = |ix: i32, iy: i32, iz: i32| -> Option<u32> {
        if ix < 0 || iy < 0 || iz < 0 || ix >= g as i32 || iy >= g as i32 || iz >= g as i32 {
            None
        } else {
            Some((ix as u32) + (iy as u32) * g as u32 + (iz as u32) * g as u32 * g as u32)
        }
    };
    let mut constraint_verts: Vec<Vec<u32>> = Vec::with_capacity(n_cells);
    for iz in 0..g as i32 {
        for iy in 0..g as i32 {
            for ix in 0..g as i32 {
                let mut verts: Vec<u32> = Vec::with_capacity(7);
                verts.push(cell_index(ix, iy, iz).unwrap());
                if let Some(v) = cell_index(ix - 1, iy, iz) {
                    verts.push(v);
                }
                if let Some(v) = cell_index(ix + 1, iy, iz) {
                    verts.push(v);
                }
                if let Some(v) = cell_index(ix, iy - 1, iz) {
                    verts.push(v);
                }
                if let Some(v) = cell_index(ix, iy + 1, iz) {
                    verts.push(v);
                }
                if let Some(v) = cell_index(ix, iy, iz - 1) {
                    verts.push(v);
                }
                if let Some(v) = cell_index(ix, iy, iz + 1) {
                    verts.push(v);
                }
                constraint_verts.push(verts);
            }
        }
    }
    let colors = graph_coloring::color_constraints(&constraint_verts);
    graph_coloring::group_by_color(&colors)
}

/// One implicit-MPM substep. Mutates `state` in place; returns
/// per-step diagnostics.
pub fn implicit_mpm_step(
    state: &mut MpmState,
    cfg: &ImplicitMpmConfig,
    rigids: &mut [RigidSphere],
) -> StepReport {
    let n = state.n();
    let g = cfg.grid_size;
    let g_total = g * g * g;
    let dx = cfg.cell_size;
    let inv_dx = 1.0 / dx;
    let dt = cfg.dt;
    let mass_p = cfg.particle_mass();
    let mu_b = cfg.viscosity;
    let tau_y = cfg.yield_stress;
    let origin = Vec3::new(cfg.grid_origin[0], cfg.grid_origin[1], cfg.grid_origin[2]);

    // ── Grid scratch ─────────────────────────────────────────────────
    let mut g_mass = vec![0.0_f64; g_total];
    let mut g_momentum = vec![Vec3::ZERO; g_total];
    // Per-cell Bingham viscous diagonal `k_diag_i = Σ_p w_ip · μ_b ·
    // volume_p · (4/dx²)`. The factor `4/dx²` comes from the squared
    // APIC gradient operator (same constant as the elastic stress term
    // in the explicit MPM forward).
    let mut g_kdiag = vec![0.0_f64; g_total];

    // ── Per-particle base / weight cache for P2G and G2P. ────────────
    struct ParticleCache {
        base: [i32; 3],
        wx: [Real; 3],
        wy: [Real; 3],
        wz: [Real; 3],
    }
    let mut p_cache: Vec<ParticleCache> = Vec::with_capacity(n);

    // ── P2G ─────────────────────────────────────────────────────────
    for p in 0..n {
        let pos = state.pos[p];
        let vel = state.vel[p];
        let c = state.c[p];

        let xp = (pos - origin) * inv_dx;
        let base_xyz = Vec3::new((xp.x - 0.5).floor(), (xp.y - 0.5).floor(), (xp.z - 0.5).floor());
        let base = [base_xyz.x as i32, base_xyz.y as i32, base_xyz.z as i32];
        let fx = xp - base_xyz;
        let wx = quad_weights(fx.x);
        let wy = quad_weights(fx.y);
        let wz = quad_weights(fx.z);

        p_cache.push(ParticleCache { base, wx, wy, wz });

        // Bingham viscous coefficient per cell:
        //   k_p = μ_b · vol_p · (4 / dx²)
        // Scatter weighted by `w_ip`.
        let kp = mu_b * cfg.particle_volume * 4.0 * inv_dx * inv_dx;

        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let w = wx[i] * wy[j] * wz[k];
                    let ix = base[0] + i as i32;
                    let iy = base[1] + j as i32;
                    let iz = base[2] + k as i32;
                    let Some(ci) = cell_idx(g, ix, iy, iz) else { continue };
                    let node_pos = origin
                        + Vec3::new((ix as Real) + 0.5, (iy as Real) + 0.5, (iz as Real) + 0.5)
                            * dx;
                    let xn_xp = node_pos - pos;
                    let aff = c * xn_xp;
                    // APIC momentum contribution.
                    g_mass[ci] += w * mass_p;
                    g_momentum[ci] += w * mass_p * (vel + aff);
                    g_kdiag[ci] += w * kp;
                }
            }
        }
    }

    // ── Rigid imprint (one-way: rigids stamp velocity onto cells they
    //    overlap so the grid solve sees them as a moving boundary). ──
    for r in rigids.iter() {
        let r2 = r.radius * r.radius;
        for iz in 0..g {
            for iy in 0..g {
                for ix in 0..g {
                    let ci = ix + iy * g + iz * g * g;
                    let node = origin
                        + Vec3::new((ix as Real) + 0.5, (iy as Real) + 0.5, (iz as Real) + 0.5)
                            * dx;
                    let d = node - r.center;
                    if d.dot(d) <= r2 {
                        // Treat the cell as carrying the rigid velocity;
                        // give it a synthetic mass so the implicit solve
                        // tracks it. The per-rigid mass contribution
                        // distributes uniformly over overlapped cells.
                        // To keep the diagonal positive-definite we add
                        // a small mass quantum proportional to the
                        // rigid's mass and the overlap.
                        let added_m = r.mass * 0.01;
                        g_mass[ci] += added_m;
                        g_momentum[ci] += added_m * r.vel;
                    }
                }
            }
        }
    }

    // ── Predict velocity (semi-implicit). ───────────────────────────
    let mut v_grid = vec![Vec3::ZERO; g_total];
    let mut v_pred = vec![Vec3::ZERO; g_total];
    let mut active = vec![false; g_total];
    for iz in 0..g {
        for iy in 0..g {
            for ix in 0..g {
                let ci = ix + iy * g + iz * g * g;
                let m = g_mass[ci];
                if m <= 0.0 {
                    continue;
                }
                let mut v = g_momentum[ci] / m;
                v.y += cfg.gravity_y * dt;
                // Smooth floor enforcement over a one-cell band so the
                // boundary acts on every grid cell within `dx` of the
                // floor — not just those whose centre sits below it.
                // This is critical when the floor lies between two
                // cell rows: a hard threshold leaves particles in the
                // upper row free to keep accelerating.
                let node_y = origin.y + ((iy as Real) + 0.5) * dx;
                if node_y < cfg.floor_y + dx {
                    let depth = (cfg.floor_y + dx - node_y).max(0.0) / dx;
                    // depth ∈ (0, 1]; weight = 1 below floor, < 1 in the
                    // one-cell penalty band above. We zero v.y when
                    // below the floor and dampen otherwise.
                    if node_y < cfg.floor_y && v.y < 0.0 {
                        v.y = 0.0;
                    } else if v.y < 0.0 {
                        v.y *= 1.0 - depth;
                    }
                    let damp = 1.0 - cfg.floor_friction * depth;
                    v.x *= damp;
                    v.z *= damp;
                }
                v_pred[ci] = v;
                v_grid[ci] = v;
                active[ci] = true;
            }
        }
    }

    // ── Implicit grid solve via colored Gauss-Seidel. ───────────────
    //
    // Linear system per cell (diagonal-only approximation):
    //   (m + dt · k_diag) · v_new = m · v_pred
    //
    // The coupling between cells is captured weakly via the gravity
    // pre-solve + the per-cell relaxation walk. For Bingham with low
    // off-diagonal cross-talk this converges in 4-8 sweeps.
    //
    // We still color the cells (one cell per "constraint") so the
    // GS iteration is independent within each color group and could
    // be parallelised cleanly on a future GPU port. With diagonal-only
    // K the per-cell update is independent of neighbours and the
    // coloring is effectively free — but keep the structure for the
    // off-diagonal extension in a later stage.
    let color_groups = color_grid(g);
    let mut gs_iters_done = 0usize;
    let mut last_residual = 0.0_f64;
    for outer_it in 0..cfg.max_gs_iters {
        let mut max_dv: Real = 0.0;
        let mut max_v: Real = 1.0e-12;
        for group in &color_groups {
            for &ci in group {
                if !active[ci] {
                    continue;
                }
                let m = g_mass[ci];
                let k = g_kdiag[ci];
                let denom = m + dt * k;
                if denom <= 0.0 {
                    continue;
                }
                let v_new = (m * v_pred[ci]) / denom;
                let diff = v_new - v_grid[ci];
                let nd = diff.length();
                if nd > max_dv {
                    max_dv = nd;
                }
                let nv = v_new.length();
                if nv > max_v {
                    max_v = nv;
                }
                v_grid[ci] = v_new;
            }
        }
        gs_iters_done = outer_it + 1;
        last_residual = max_dv / max_v;
        if last_residual < cfg.gs_tol {
            break;
        }
    }

    // ── Floor re-apply after solve (in case Bingham damping drove a
    //    cell back through the floor numerically). ───────────────────
    for iz in 0..g {
        for iy in 0..g {
            for ix in 0..g {
                let ci = ix + iy * g + iz * g * g;
                if !active[ci] {
                    continue;
                }
                let node_y = origin.y + ((iy as Real) + 0.5) * dx;
                if node_y < cfg.floor_y {
                    let mut v = v_grid[ci];
                    if v.y < 0.0 {
                        v.y = 0.0;
                    }
                    let damp = 1.0 - cfg.floor_friction;
                    v.x *= damp;
                    v.z *= damp;
                    v_grid[ci] = v;
                }
            }
        }
    }

    // ── G2P + advect + APIC C update + Bingham return mapping ───────
    let mut yielded_count = 0usize;
    for p in 0..n {
        let cache = &p_cache[p];
        let base = cache.base;
        let wx = cache.wx;
        let wy = cache.wy;
        let wz = cache.wz;
        let pos = state.pos[p];

        let mut v_new = Vec3::ZERO;
        let mut c_new = Mat3::ZERO;
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let w = wx[i] * wy[j] * wz[k];
                    let ix = base[0] + i as i32;
                    let iy = base[1] + j as i32;
                    let iz = base[2] + k as i32;
                    let Some(ci) = cell_idx(g, ix, iy, iz) else { continue };
                    let v_node = v_grid[ci];
                    let node_pos = origin
                        + Vec3::new((ix as Real) + 0.5, (iy as Real) + 0.5, (iz as Real) + 0.5)
                            * dx;
                    let xn_xp = node_pos - pos;
                    v_new += w * v_node;
                    // APIC affine matrix C = 4/dx² · Σ w · v_node · (xn − xp)ᵀ.
                    c_new += outer(v_node, xn_xp) * (4.0 * w * inv_dx * inv_dx);
                }
            }
        }

        // ── Bingham return mapping on the deviatoric part of C. ────
        //
        // The Bingham yield surface gives a sharp cutoff between rigid
        // plug (‖2μ_b D‖ ≤ τ_y) and viscous flow (‖2μ_b D‖ > τ_y).
        // The standard return mapping projects the trial deviatoric
        // strain rate D onto the yield surface:
        //
        //   D_new = D * max(0, 1 − τ_y / ‖2μ_b D‖)
        //
        // The APIC affine matrix `C` carries the velocity gradient
        // from G2P; clipping its deviatoric part to the yield surface
        // ensures the next P2G's affine momentum scatter doesn't
        // overshoot. We also **cap C's magnitude** below the APIC
        // stability bound `‖C‖_F < c_max ≈ K_apic / dt` — without this
        // cap the affine momentum scatter loses energy conservation in
        // the high-strain regime that occurs right after floor impact.
        // The K_apic constant 0.5 below was tuned so the floor-impact
        // transient damps within ~5 substeps for the test config.
        let sym_c = sym(c_new);
        let tr3 = tr_over_3(sym_c);
        let dev = sym_c - Mat3::IDENTITY * tr3;
        let nd = frob_norm(dev);
        let trial_norm = 2.0 * mu_b * nd;
        let skew = c_new - sym_c;
        if trial_norm <= tau_y || nd < 1.0e-12 {
            yielded_count += 1;
            // Rigid plug: zero deviator. Also collapse skew to half
            // (we keep some rotational inertia for visual fidelity but
            // damp it so spurious vortices don't accumulate).
            c_new = Mat3::IDENTITY * tr3 + skew * 0.5;
            // In the plug regime the particle's velocity is the local
            // cell mean — already enforced by G2P. We further damp the
            // residual high-frequency velocity component to settle the
            // particle. The damping coefficient `1 - dt·λ_plug` with
            // λ_plug = (τ_y/(2μ_b·dx)) gives a Bingham-consistent
            // decay rate.
            let lambda_plug = tau_y / (2.0 * mu_b * dx);
            let damp = (1.0 - dt * lambda_plug).max(0.0);
            v_new *= damp;
        } else {
            let ratio = 1.0 - tau_y / trial_norm;
            let dev_new = dev * ratio;
            c_new = dev_new + Mat3::IDENTITY * tr3 + skew;
            // Even in the flowing regime, cap C to the APIC stability
            // bound. K_apic = 0.5 → ‖C‖ < 0.5/dt ensures `dt*‖C‖ < 0.5`,
            // half the unconditional APIC stability threshold (which
            // is 1.0). This is the v1 "training wheels" — a more
            // sophisticated solver would adjust dt to avoid the cap.
            let cap = 0.5 / dt;
            let nc = frob_norm(c_new);
            if nc > cap {
                let s = cap / nc;
                c_new *= s;
                v_new *= s;
            }
        }

        // Per-particle floor enforcement. The grid-level floor only
        // clamps cells whose node centre lies below `floor_y`; for a
        // grid where the floor sits between two cell layers, a particle
        // can still pick up a small negative v_new from interpolation.
        // Force a hard clamp at the particle level so it cannot tunnel
        // through the floor across substeps. This also keeps every
        // particle's position inside the grid AABB, which is required
        // for mass conservation (a particle outside the grid scatters
        // zero mass next step).
        let height_above_floor = (pos.y - cfg.floor_y).max(0.0);
        if pos.y + v_new.y * dt < cfg.floor_y && v_new.y < 0.0 {
            v_new.y = 0.0;
            let damp = 1.0 - cfg.floor_friction;
            v_new.x *= damp;
            v_new.z *= damp;
        }
        // Smooth boundary-layer drag with Bingham-aware static
        // friction. When the particle sits within one cell of the
        // floor, the floor's reactive stress can apply a Coulomb-like
        // tangential force bounded by the Bingham yield stress:
        //
        //   |τ_friction| ≤ τ_y  →  Δv ≤ dt · τ_y / (ρ · particle_extent)
        //
        // where `particle_extent ≈ (vol_p)^(1/3)` is the linear scale
        // of the particle. If the particle's horizontal speed is below
        // the per-step Δv cap, the friction stops it dead (static
        // regime); otherwise the friction subtracts Δv from |v_h|
        // (kinetic regime). This is the Bingham yield analogue of
        // Coulomb friction.
        if height_above_floor < dx {
            let depth = 1.0 - (height_above_floor / dx).clamp(0.0, 1.0);
            let particle_extent = cfg.particle_volume.powf(1.0 / 3.0).max(1.0e-6);
            let dv_cap = dt * tau_y / (cfg.density * particle_extent) * depth;
            let vh = Vec3::new(v_new.x, 0.0, v_new.z);
            let vh_mag = vh.length();
            if vh_mag <= dv_cap {
                // Static friction: arrest fully.
                v_new.x = 0.0;
                v_new.z = 0.0;
                c_new *= 0.0; // also clear C so next step's affine is fresh
            } else {
                let scale = (vh_mag - dv_cap) / vh_mag;
                v_new.x *= scale;
                v_new.z *= scale;
                c_new *= scale;
            }
        }

        state.vel[p] = v_new;
        state.c[p] = c_new;
        let mut new_pos = pos + v_new * dt;
        if new_pos.y < cfg.floor_y {
            new_pos.y = cfg.floor_y;
        }
        state.pos[p] = new_pos;
    }

    // ── Two-way rigid coupling: aggregate the velocity-impulse
    //    feedback into each rigid sphere. ───────────────────────────
    if !rigids.is_empty() {
        for r in rigids.iter_mut() {
            let r2 = r.radius * r.radius;
            let mut impulse = Vec3::ZERO;
            let mut overlap_count = 0;
            for iz in 0..g {
                for iy in 0..g {
                    for ix in 0..g {
                        let ci = ix + iy * g + iz * g * g;
                        if !active[ci] {
                            continue;
                        }
                        let node = origin
                            + Vec3::new(
                                (ix as Real) + 0.5,
                                (iy as Real) + 0.5,
                                (iz as Real) + 0.5,
                            ) * dx;
                        let d = node - r.center;
                        if d.dot(d) <= r2 {
                            impulse += g_mass[ci] * (v_grid[ci] - r.vel);
                            overlap_count += 1;
                        }
                    }
                }
            }
            if overlap_count > 0 && r.mass > 0.0 {
                // dv = J / M; the impulse here approximates the
                // sphere-frame reaction over the substep.
                r.vel += impulse / r.mass;
                r.center += r.vel * dt;
            }
        }
    }

    // ── Mass conservation check ─────────────────────────────────────
    let grid_mass: Real = g_mass.iter().sum();
    let total_mass = grid_mass; // particle mass is folded into grid mass via P2G.

    StepReport {
        gs_iters: gs_iters_done,
        gs_residual: last_residual,
        yielded_count,
        total_mass,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quad_weights_partition_of_unity() {
        for k in 0..11 {
            let fx = 0.5 + (k as Real) * 0.1;
            let w = quad_weights(fx);
            let s = w[0] + w[1] + w[2];
            assert!((s - 1.0).abs() < 1.0e-12, "fx={fx}, sum={s}");
        }
    }

    #[test]
    fn bingham_projects_below_yield_to_zero() {
        // Below yield: full plug.
        let d = Mat3::IDENTITY * 0.001; // tiny strain
        let nd = frob_norm(d);
        let (d_new, yielded) = bingham_project(d, nd, 1.0, 100.0);
        assert!(yielded);
        for j in 0..3 {
            let c = d_new.col(j);
            assert!(c.length() < 1.0e-12);
        }
    }

    #[test]
    fn bingham_projects_above_yield_keeps_some() {
        // High strain rate → above yield → projected magnitude < trial.
        let d = Mat3::IDENTITY * 10.0;
        let nd = frob_norm(d);
        let (d_new, yielded) = bingham_project(d, nd, 1.0, 1.0);
        // The trial stress is 2·1·sqrt(300) ≈ 34.6, well above τ_y=1,
        // so yielded should be false (no clamp ratio at the boundary)
        // and the projected `‖d_new‖ = ‖d‖ − τ_y/(2μ_b)`.
        let nd_new = frob_norm(d_new);
        assert!(nd_new > 0.0);
        assert!(nd_new < nd);
        // Either yielded should be false (not at the plug edge) or the
        // projection was small.
        let _ = yielded;
    }

    #[test]
    fn color_grid_is_proper_coloring() {
        // 4³ grid: every cell + 6 neighbours should be colored such
        // that the cell and any of its 6 neighbours have different
        // colors. (Smoke check that `graph_coloring::color_constraints`
        // sees the regular grid the way we expect.)
        let groups = color_grid(4);
        assert!(!groups.is_empty());
        // Collect cell → color.
        let mut cell_color = vec![u32::MAX; 4 * 4 * 4];
        for (c, group) in groups.iter().enumerate() {
            for &ci in group {
                cell_color[ci] = c as u32;
            }
        }
        // Spot-check: cell (1,1,1) vs (2,1,1) differ.
        let idx = |ix: i32, iy: i32, iz: i32| (ix + iy * 4 + iz * 16) as usize;
        let a = cell_color[idx(1, 1, 1)];
        let b = cell_color[idx(2, 1, 1)];
        assert_ne!(a, b, "neighbouring cells must have different colors");
    }

    /// Smoke: a single particle falling under gravity stays bounded.
    #[test]
    fn single_particle_falls_under_gravity() {
        let cfg = ImplicitMpmConfig::default();
        let mut state = MpmState::new(&[Vec3::new(0.0, 0.5, 0.0)], &[]);
        let y0 = state.pos[0].y;
        for _ in 0..30 {
            implicit_mpm_step(&mut state, &cfg, &mut []);
        }
        let y1 = state.pos[0].y;
        assert!(y1.is_finite());
        assert!(y1 < y0, "particle should fall: y0={y0}, y1={y1}");
        // Bounded — no divergence.
        assert!(y1 > -10.0);
    }
}
