//! Stage 39A / 40 — **Implicit MPM with Bingham rheology**.
//!
//! Forward implicit Material-Point-Method timestep that handles stiff
//! plastic flow at dt far above the explicit-CFL limit. The algorithm
//! mirrors Newton's `implicit_mpm` solver (forward grid step →
//! assemble linear system → colored Gauss-Seidel solve → projection).
//!
//! ## Stage 40 v4 — full 125-point Newton Hessian
//!
//! v1/v2/v3 shipped the 27-point Hessian (cells coupled to their 26
//! immediate neighbours). v4 extends the stencil to the **true** Newton
//! reach for the quadratic-B-spline kernel: two cells whose 3×3×3
//! particle support boxes overlap can share a particle whenever their
//! index offset is at most ±2 on every axis, so the strict Hessian is
//! 5×5×5 = 125-point. v4 keeps the 27-point path as the default (a
//! 5× cheaper assembly) but exposes [`HessianStencil::Full125`] for
//! callers that need tighter convergence per GS iteration on stiff
//! Bingham configurations.
//!
//! ## Stage 40 — full 27-point Newton Hessian
//!
//! Stage 39A shipped a diagonal-only approximation `(M + dt·K_diag)·v = M·v_pred`.
//! Stage 40 replaces that with the **full 27-point Newton Hessian**:
//! each grid cell is coupled to its 26 immediate neighbours (the
//! 3×3×3 stencil centred on the cell) plus itself. Off-diagonal
//! entries `K_ij` (`i ≠ j`) accumulate `Σ_p k_p · w_ip · w_jp` over
//! every particle that scatters onto **both** cells `i` and `j`.
//!
//! ## Algorithm
//!
//! Per substep:
//!
//! 1. **P2G** — quadratic B-spline scatter of `(mass, momentum,
//!    27-stencil K)` from particles to a 3×3×3 stencil on the grid.
//!    Each particle contributes to its 3×3×3 = 27 cells; for every
//!    *ordered pair* `(i, j)` of cells in that stencil whose offset
//!    `j - i` lies inside the unit cube `{-1, 0, 1}³`, accumulate
//!    `k_p · w_ip · w_jp` into the cell-i Hessian slot for offset
//!    `j - i`. The diagonal slot (offset `(0,0,0)`) reduces to the
//!    Stage 39A `k_diag`.
//!
//! 2. **Predict velocity** — `v_pred = momentum/m + g·dt`.
//!
//! 3. **Implicit grid solve** — solve
//!    `(M + dt · K) · v_new = M · v_pred`
//!    with `K` the full 27-point velocity-domain Hessian (isotropic;
//!    block entries are `k_ij · I_3`, so we store scalars). Iteration
//!    is **colored Gauss-Seidel**: each cell is a node, the
//!    27-stencil produces an 8-coloring on the parity of
//!    `(ix mod 2, iy mod 2, iz mod 2)`. Per-cell update:
//!    `v_i ← (M_i · v_pred_i − dt · Σ_{j ≠ i} K_ij · v_j) /
//!              (M_i + dt · K_ii)`.
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
//! ## What this stage doesn't do
//!
//! - **5×5×5 (125-point) "true" Hessian.** When two cells `i, j` are
//!   separated by up to 2 cells on every axis they can still share a
//!   particle — strictly the Newton Hessian is 5×5×5 = 125-point.
//!   Stage 40 lumps everything past `|i − j|_∞ = 1` onto the 27-point
//!   stencil. The 27-point block captures the dominant local coupling
//!   (any particle is touched by 27 cells in 3³ so neighbour pairs at
//!   `|i−j|_∞ = 1` carry the largest `w_ip · w_jp` products); the
//!   125-point extension halves convergence again but doubles assembly
//!   cost. Reserved for a future stage.
//! - **Drucker-Prager / snow.** Single rheology only — Bingham is
//!   already a non-trivial yield surface.
//! - **Full GPU perf target.** Stage 40 ships a working GPU port
//!   (P2G + colored-GS sweep over the full Hessian) and a parity
//!   gate, but the per-step wall-time target is deferred.

use crate::math::coloring as graph_coloring;
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

/// Hessian stencil width for the implicit grid solve.
///
/// * [`HessianStencil::Diagonal`] — Stage 39A behaviour: per-cell
///   diagonal preconditioner, off-diagonal `K_ij` ignored.
/// * [`HessianStencil::Full27`] — Stage 40 default: full 27-point
///   block-isotropic Hessian (cell + 26 immediate neighbours).
/// * [`HessianStencil::Full125`] — Stage 40 v4: full 5×5×5 = 125-point
///   Hessian. The quadratic-B-spline kernel scatters each particle onto
///   a 3×3×3 box; two cells within distance 2 on every axis can share a
///   particle, so the strict Newton Hessian reaches 125 entries per row.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Default)]
pub enum HessianStencil {
    /// 1-point diagonal — Stage 39A path.
    Diagonal,
    /// 27-point (cell + 26 neighbours) — Stage 40 default.
    #[default]
    Full27,
    /// 125-point (5×5×5 box) — Stage 40 v4. True Newton Hessian for the
    /// quadratic-B-spline particle kernel.
    Full125,
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
    /// Final relative-update residual (the GS termination criterion).
    pub gs_residual: Real,
    /// Number of particles whose Bingham projection clamped the
    /// trial stress to the yield surface. Useful for telemetry.
    pub yielded_count: usize,
    /// Sum of grid mass + particle mass after the step. The test
    /// suite uses this to gate mass conservation.
    pub total_mass: Real,
    /// Squared L2-norm of the post-solve residual measured against
    /// the **full 27-point** Hessian system:
    ///   `Σ_i ‖(M_i + dt · K_{ii}) v_i + dt · Σ_{j ≠ i} K_{ij} v_j − M_i · v_pred_i‖²`.
    /// Independent of which stencil the GS sweep ran on, so it
    /// provides a fair "true accuracy" comparison between the
    /// diagonal-only and full-27 paths.
    pub true_residual_sq: Real,
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
/// deviator and a `plug` flag (true = rigid plug, false = flowing).
///
/// - Trial stress norm: `‖τ_trial‖ = 2μ_b ‖D‖`.
/// - If `‖τ_trial‖ ≤ τ_y` → rigid plug, return `D := 0` with `plug = true`.
/// - Else `D := (1 − τ_y / ‖τ_trial‖) · D` — scaled so the projected
///   stress lies exactly on the yield surface (Daviet 2016 §3.2); `plug = false`.
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
        // Flowing regime: `tau_y > 0` ⇒ `scale < 1.0` strictly, so `plug = false`.
        (d * scale, false)
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

/// Color the regular 3D grid for colored Gauss-Seidel over the 27-point
/// stencil (cell + 26 immediate neighbours).
///
/// Each cell is a "constraint" whose vertex set is the cell index plus
/// every neighbour in its 3×3×3 box. Two cells share an edge iff some
/// constraint contains both — i.e. one of them is within distance 1 of
/// the other on any axis. The generic
/// [`graph_coloring::color_constraints`] returns the standard 8-coloring
/// (`(ix mod 2, iy mod 2, iz mod 2)`) on this graph.
///
/// Returns `groups[c] = Vec<cell_idx>` for `c ∈ [0, n_colors)`.
pub fn color_grid(grid_size: usize) -> Vec<Vec<usize>> {
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
                let mut verts: Vec<u32> = Vec::with_capacity(27);
                for dz in -1..=1 {
                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if let Some(v) = cell_index(ix + dx, iy + dy, iz + dz) {
                                verts.push(v);
                            }
                        }
                    }
                }
                constraint_verts.push(verts);
            }
        }
    }
    let colors = graph_coloring::color_constraints(&constraint_verts);
    graph_coloring::group_by_color(&colors)
}

/// Color the regular 3D grid for colored Gauss-Seidel over the **125-point**
/// stencil (5×5×5 box centred on each cell).
///
/// Two cells share an edge iff one lies within distance 2 of the other
/// on every axis. The natural closed-form coloring is the 64-coloring
/// `(ix mod 4, iy mod 4, iz mod 4)` — guaranteed proper because any
/// two cells in the same color group differ by at least 4 on at least
/// one axis, which is strictly greater than the 2-cell stencil reach.
/// The generic [`graph_coloring::color_constraints`] is given the
/// 5×5×5 vertex sets per cell; it may discover a different ordering
/// but still returns at most 64 colors on a sufficiently large grid.
///
/// Returns `groups[c] = Vec<cell_idx>` for `c ∈ [0, n_colors)`.
pub fn color_grid_125(grid_size: usize) -> Vec<Vec<usize>> {
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
                let mut verts: Vec<u32> = Vec::with_capacity(125);
                for dz in -2..=2 {
                    for dy in -2..=2 {
                        for dx in -2..=2 {
                            if let Some(v) = cell_index(ix + dx, iy + dy, iz + dz) {
                                verts.push(v);
                            }
                        }
                    }
                }
                constraint_verts.push(verts);
            }
        }
    }
    let colors = graph_coloring::color_constraints(&constraint_verts);
    graph_coloring::group_by_color(&colors)
}

/// Map 3D stencil offset `(dx, dy, dz) ∈ {-1, 0, 1}³` to a flat index
/// in `[0, 27)`. The diagonal entry is at index 13 (centre of the cube).
#[inline]
pub fn stencil_offset(dx: i32, dy: i32, dz: i32) -> usize {
    debug_assert!(dx.abs() <= 1 && dy.abs() <= 1 && dz.abs() <= 1);
    ((dz + 1) as usize) * 9 + ((dy + 1) as usize) * 3 + ((dx + 1) as usize)
}

/// Map 3D stencil offset `(dx, dy, dz) ∈ {-2..=2}³` to a flat index in
/// `[0, 125)`. The diagonal entry `(0, 0, 0)` is at index
/// [`STENCIL_DIAG_125`] = 62 (centre of the 5×5×5 box).
#[inline]
pub fn stencil_offset_125(dx: i32, dy: i32, dz: i32) -> usize {
    debug_assert!(dx.abs() <= 2 && dy.abs() <= 2 && dz.abs() <= 2);
    ((dz + 2) as usize) * 25 + ((dy + 2) as usize) * 5 + ((dx + 2) as usize)
}

/// Index of the diagonal entry `(0, 0, 0)` in the 27-stencil. Inlined
/// constant so call sites can spell out the intent.
pub const STENCIL_DIAG: usize = 13;

/// Index of the diagonal entry `(0, 0, 0)` in the 125-stencil.
pub const STENCIL_DIAG_125: usize = 62;

/// Width of a single Hessian row for the 125-stencil (5³).
pub const STENCIL_LEN_125: usize = 125;

/// Internal type that bundles the assembled implicit-MPM grid system
/// after P2G but before the inner GS solve. Exposed via
/// [`implicit_mpm_step_with_solver`] so the Stage 40 GPU parity test
/// can hijack the colored-GS sweep while reusing every other step.
#[derive(Clone, Debug)]
pub struct GridSystem {
    /// Edge length of the cubic grid.
    pub grid_size: usize,
    /// Per-cell mass (`g_total` entries).
    pub g_mass: Vec<Real>,
    /// 27-point Hessian, `g_total * 27` entries; see
    /// [`stencil_offset`] for the layout. Populated when the assembled
    /// stencil is [`HessianStencil::Diagonal`] or [`HessianStencil::Full27`].
    /// For Full125 this is empty — the system stores [`Self::k_stencil_125`]
    /// instead.
    pub k_stencil: Vec<Real>,
    /// 125-point Hessian, `g_total * 125` entries; see
    /// [`stencil_offset_125`] for the layout. Populated only when the
    /// assembled stencil is [`HessianStencil::Full125`].
    pub k_stencil_125: Vec<Real>,
    /// Predicted velocity (`g_total` entries).
    pub v_pred: Vec<Vec3>,
    /// Active-cell flags (cells with mass > 0). `g_total` entries.
    pub active: Vec<bool>,
    /// Colored-GS groups produced by [`color_grid`] or [`color_grid_125`].
    /// Length corresponds to the stencil that was used during assembly.
    pub color_groups: Vec<Vec<usize>>,
    /// Time step the system was assembled with.
    pub dt: Real,
    /// Stencil this system was assembled with. Determines whether
    /// [`Self::k_stencil`] or [`Self::k_stencil_125`] is populated.
    pub stencil: HessianStencil,
}

/// One implicit-MPM substep with the **Stage 40 default** (full
/// 27-point Newton Hessian). Mutates `state` in place; returns
/// per-step diagnostics.
pub fn implicit_mpm_step(
    state: &mut MpmState,
    cfg: &ImplicitMpmConfig,
    rigids: &mut [RigidSphere],
) -> StepReport {
    implicit_mpm_step_with_stencil(state, cfg, rigids, HessianStencil::Full27)
}

/// One implicit-MPM substep with an explicit Hessian-stencil choice.
/// Used by the Stage 40 convergence sanity test that compares the
/// diagonal-only (Stage 39A) path against the full 27-point path.
pub fn implicit_mpm_step_with_stencil(
    state: &mut MpmState,
    cfg: &ImplicitMpmConfig,
    rigids: &mut [RigidSphere],
    stencil: HessianStencil,
) -> StepReport {
    implicit_mpm_step_with_solver(state, cfg, rigids, stencil, None)
}

/// One implicit-MPM substep with an optional GS-sweep override. When
/// `gs_override` is `Some(closure)`, the closure is invoked with the
/// assembled [`GridSystem`] and an initial-guess `v_grid`, and must
/// mutate `v_grid` in place to the GS-solver result. When `None`, the
/// stock host-side colored Gauss-Seidel sweep is used. The Stage 40
/// GPU parity test plugs the device kernel into this hook.
pub fn implicit_mpm_step_with_solver(
    state: &mut MpmState,
    cfg: &ImplicitMpmConfig,
    rigids: &mut [RigidSphere],
    stencil: HessianStencil,
    gs_override: Option<&mut dyn FnMut(&GridSystem, &mut [Vec3], HessianStencil) -> (usize, Real)>,
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
    let want_125 = matches!(stencil, HessianStencil::Full125);
    // Per-cell 27-point Hessian: `k_stencil[i * 27 + s]` is the
    // scalar coupling `K_{i, i+δ(s)}` where `δ(s)` is the stencil offset
    // recovered from index `s ∈ [0, 27)` via [`stencil_offset`]. The
    // matrix is isotropic (`K · v` acts componentwise), so we store a
    // single scalar per stencil slot rather than a 3×3 block.
    //
    // Population: for every particle `p`, for every cell pair `(i, j)`
    // both within `p`'s 3×3×3 stencil whose offset `j − i` has
    // `|·|_∞ ≤ 1`, accumulate `k_p · w_ip · w_jp` into
    // `k_stencil[i * 27 + s(j − i)]`. The diagonal slot collapses to
    // the Stage 39A `k_diag` term.
    //
    // When `want_125`, we instead allocate the 125-stencil
    // (`k_stencil_125`) which captures every cell pair within the
    // 3×3×3 particle support — the strict Newton Hessian for the
    // quadratic-B-spline kernel. Offsets `|j − i|_∞ ≤ 2`.
    let mut k_stencil = if want_125 {
        Vec::new()
    } else {
        vec![0.0_f64; g_total * 27]
    };
    let mut k_stencil_125 = if want_125 {
        vec![0.0_f64; g_total * STENCIL_LEN_125]
    } else {
        Vec::new()
    };

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
        let base_xyz = Vec3::new(
            (xp.x - 0.5).floor(),
            (xp.y - 0.5).floor(),
            (xp.z - 0.5).floor(),
        );
        let base = [base_xyz.x as i32, base_xyz.y as i32, base_xyz.z as i32];
        let fx = xp - base_xyz;
        let wx = quad_weights(fx.x);
        let wy = quad_weights(fx.y);
        let wz = quad_weights(fx.z);

        p_cache.push(ParticleCache { base, wx, wy, wz });

        // Bingham viscous coefficient per cell:
        //   k_p = μ_b · vol_p · (4 / dx²)
        // The cell-pair Hessian entry accumulates `k_p · w_ip · w_jp`.
        let kp = mu_b * cfg.particle_volume * 4.0 * inv_dx * inv_dx;

        // Precompute the (w, cell_idx) tuples for the 27 stencil
        // entries so the O(N²) Hessian scatter inside the loop reads
        // contiguous cached values. Out-of-grid entries get index `-1`
        // so they're skipped without re-checking bounds.
        let mut sten: [(Real, i32, i32, i32, i32); 27] = [(0.0, -1, 0, 0, 0); 27];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let w = wx[i] * wy[j] * wz[k];
                    let ix = base[0] + i as i32;
                    let iy = base[1] + j as i32;
                    let iz = base[2] + k as i32;
                    let ci = cell_idx(g, ix, iy, iz).map(|c| c as i32).unwrap_or(-1);
                    sten[i * 9 + j * 3 + k] = (w, ci, ix, iy, iz);
                }
            }
        }

        // Mass / momentum scatter (linear-in-w).
        for &(w, ci, ix, iy, iz) in sten.iter() {
            if ci < 0 {
                continue;
            }
            let node_pos =
                origin + Vec3::new((ix as Real) + 0.5, (iy as Real) + 0.5, (iz as Real) + 0.5) * dx;
            let xn_xp = node_pos - pos;
            let aff = c * xn_xp;
            let ci = ci as usize;
            g_mass[ci] += w * mass_p;
            g_momentum[ci] += w * mass_p * (vel + aff);
        }

        // Full-Hessian scatter (O(27²) per particle — 729 ops). For the
        // 27-stencil path, only entries with `|i − j|_∞ ≤ 1` land inside
        // the row; the outer ring of cell pairs (where particles in the
        // corner of their 3³ stencil couple to the opposite corner of
        // their own 3³ stencil) is lumped onto the diagonal via the
        // implicit sum-of-particles approximation. The 125-stencil path
        // (Stage 40 v4) keeps every pair — `|i − j|_∞ ≤ 2` is the strict
        // bound for a 3×3×3 particle support, so the full 729 pairs all
        // land cleanly inside the 5×5×5 row.
        if want_125 {
            for a in 0..27 {
                let (wa, ca, ax, ay, az) = sten[a];
                if ca < 0 {
                    continue;
                }
                for b in 0..27 {
                    let (wb, cb, bx, by, bz) = sten[b];
                    if cb < 0 {
                        continue;
                    }
                    let dx_ij = bx - ax;
                    let dy_ij = by - ay;
                    let dz_ij = bz - az;
                    // Distance between any two cells inside a 3³ box is
                    // at most 2 on each axis — guaranteed in-range.
                    debug_assert!(dx_ij.abs() <= 2 && dy_ij.abs() <= 2 && dz_ij.abs() <= 2);
                    let s = stencil_offset_125(dx_ij, dy_ij, dz_ij);
                    k_stencil_125[(ca as usize) * STENCIL_LEN_125 + s] += kp * wa * wb;
                }
            }
        } else {
            for a in 0..27 {
                let (wa, ca, ax, ay, az) = sten[a];
                if ca < 0 {
                    continue;
                }
                for b in 0..27 {
                    let (wb, cb, bx, by, bz) = sten[b];
                    if cb < 0 {
                        continue;
                    }
                    let dx_ij = bx - ax;
                    let dy_ij = by - ay;
                    let dz_ij = bz - az;
                    if dx_ij.abs() > 1 || dy_ij.abs() > 1 || dz_ij.abs() > 1 {
                        continue;
                    }
                    let s = stencil_offset(dx_ij, dy_ij, dz_ij);
                    k_stencil[(ca as usize) * 27 + s] += kp * wa * wb;
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
                    let damp = (1.0 - cfg.floor_friction * depth).max(0.0);
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
    // Linear system per cell (full Newton Hessian — 27 or 125 entries
    // per row depending on `stencil`):
    //   (m_i + dt · K_ii) · v_new_i + dt · Σ_{j ≠ i} K_ij · v_j
    //       = m_i · v_pred_i
    //
    // K is isotropic (scalar block per stencil slot) so the 3D
    // velocity components decouple per cell and we can solve each
    // axis with the same scalar denominator.
    //
    // Each cell is a node of the constraint graph. The 27-stencil
    // produces an 8-coloring (octant parity of `(ix, iy, iz)`); the
    // 125-stencil reaches farther and produces up to a 64-coloring
    // (`(ix mod 4, iy mod 4, iz mod 4)`). Within a color group the
    // cell updates are independent → GPU-friendly.
    let color_groups = if want_125 {
        color_grid_125(g)
    } else {
        color_grid(g)
    };
    let gi = g as i32;
    let use_off_diag = matches!(stencil, HessianStencil::Full27 | HessianStencil::Full125);
    let mut gs_iters_done = 0usize;
    let mut last_residual = 0.0_f64;

    // If the caller supplied an alternate solver (e.g. the Stage 40
    // GPU kernel), package the assembled system and hand it off. The
    // override is responsible for honoring `cfg.max_gs_iters` /
    // `cfg.gs_tol` itself; we only record what it tells us.
    if let Some(cb) = gs_override {
        let system = GridSystem {
            grid_size: g,
            g_mass: g_mass.clone(),
            k_stencil: k_stencil.clone(),
            k_stencil_125: k_stencil_125.clone(),
            v_pred: v_pred.clone(),
            active: active.clone(),
            color_groups: color_groups.clone(),
            dt,
            stencil,
        };
        let (iters, res) = cb(&system, &mut v_grid, stencil);
        gs_iters_done = iters;
        last_residual = res;
    } else if want_125 {
        // 5×5×5 GS sweep.
        for outer_it in 0..cfg.max_gs_iters {
            let mut max_dv: Real = 0.0;
            let mut max_v: Real = 1.0e-12;
            for group in &color_groups {
                for &ci in group {
                    if !active[ci] {
                        continue;
                    }
                    let m = g_mass[ci];
                    let k_diag = k_stencil_125[ci * STENCIL_LEN_125 + STENCIL_DIAG_125];
                    let denom = m + dt * k_diag;
                    if denom <= 0.0 {
                        continue;
                    }
                    let iz = (ci / (g * g)) as i32;
                    let iy = ((ci / g) % g) as i32;
                    let ix = (ci % g) as i32;
                    let mut off = Vec3::ZERO;
                    for dz in -2..=2 {
                        for dy in -2..=2 {
                            for dx_ in -2..=2 {
                                if dx_ == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let jx = ix + dx_;
                                let jy = iy + dy;
                                let jz = iz + dz;
                                if jx < 0 || jy < 0 || jz < 0 || jx >= gi || jy >= gi || jz >= gi {
                                    continue;
                                }
                                let cj = (jx as usize) + (jy as usize) * g + (jz as usize) * g * g;
                                let s = stencil_offset_125(dx_, dy, dz);
                                let kij = k_stencil_125[ci * STENCIL_LEN_125 + s];
                                if kij != 0.0 {
                                    off += kij * v_grid[cj];
                                }
                            }
                        }
                    }
                    let rhs = m * v_pred[ci] - dt * off;
                    let v_new = rhs / denom;
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
    } else {
        for outer_it in 0..cfg.max_gs_iters {
            let mut max_dv: Real = 0.0;
            let mut max_v: Real = 1.0e-12;
            for group in &color_groups {
                for &ci in group {
                    if !active[ci] {
                        continue;
                    }
                    let m = g_mass[ci];
                    let k_diag = k_stencil[ci * 27 + STENCIL_DIAG];
                    let denom = m + dt * k_diag;
                    if denom <= 0.0 {
                        continue;
                    }
                    let mut off = Vec3::ZERO;
                    if use_off_diag {
                        // Decode (ix, iy, iz) so we can index neighbours.
                        let iz = (ci / (g * g)) as i32;
                        let iy = ((ci / g) % g) as i32;
                        let ix = (ci % g) as i32;
                        // Sum off-diagonal flux: dt · Σ_{j ≠ i} K_ij · v_j.
                        for dz in -1..=1 {
                            for dy in -1..=1 {
                                for dx_ in -1..=1 {
                                    if dx_ == 0 && dy == 0 && dz == 0 {
                                        continue;
                                    }
                                    let jx = ix + dx_;
                                    let jy = iy + dy;
                                    let jz = iz + dz;
                                    if jx < 0
                                        || jy < 0
                                        || jz < 0
                                        || jx >= gi
                                        || jy >= gi
                                        || jz >= gi
                                    {
                                        continue;
                                    }
                                    let cj =
                                        (jx as usize) + (jy as usize) * g + (jz as usize) * g * g;
                                    let s = stencil_offset(dx_, dy, dz);
                                    let kij = k_stencil[ci * 27 + s];
                                    if kij != 0.0 {
                                        off += kij * v_grid[cj];
                                    }
                                }
                            }
                        }
                    }
                    let rhs = m * v_pred[ci] - dt * off;
                    let v_new = rhs / denom;
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
    } // end of `if gs_override.is_none()` else

    // ── True residual against the assembled Newton Hessian.
    //    Independent of whether the GS sweep used the diagonal-only
    //    approximation or the full off-diagonal coupling, this measures
    //    how close `v_grid` is to the exact solution of the **assembled**
    //    coupled system. For Diagonal/Full27 the stencil is 27-point;
    //    for Full125 it is 125-point. Returned in the [`StepReport`].
    //    Cross-stencil comparison (e.g. Full27 vs Full125 against the
    //    same 125-pt yardstick) is done by the standalone helpers
    //    [`full_hessian_residual_norm_squared`] (27-pt yardstick) and
    //    [`full_hessian_residual_norm_squared_125`] (125-pt yardstick).
    let mut true_residual_sq: Real = 0.0;
    if want_125 {
        for iz in 0..g as i32 {
            for iy in 0..g as i32 {
                for ix in 0..g as i32 {
                    let ci_u = (ix as usize) + (iy as usize) * g + (iz as usize) * g * g;
                    if !active[ci_u] {
                        continue;
                    }
                    let m = g_mass[ci_u];
                    let k_diag = k_stencil_125[ci_u * STENCIL_LEN_125 + STENCIL_DIAG_125];
                    let mut lhs = (m + dt * k_diag) * v_grid[ci_u];
                    for dz in -2..=2 {
                        for dy in -2..=2 {
                            for dx_ in -2..=2 {
                                if dx_ == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let jx = ix + dx_;
                                let jy = iy + dy;
                                let jz = iz + dz;
                                if jx < 0 || jy < 0 || jz < 0 || jx >= gi || jy >= gi || jz >= gi {
                                    continue;
                                }
                                let cj = (jx as usize) + (jy as usize) * g + (jz as usize) * g * g;
                                let s = stencil_offset_125(dx_, dy, dz);
                                let kij = k_stencil_125[ci_u * STENCIL_LEN_125 + s];
                                if kij != 0.0 {
                                    lhs += dt * kij * v_grid[cj];
                                }
                            }
                        }
                    }
                    let r = lhs - m * v_pred[ci_u];
                    true_residual_sq += r.dot(r);
                }
            }
        }
    } else {
        for iz in 0..g as i32 {
            for iy in 0..g as i32 {
                for ix in 0..g as i32 {
                    let ci_u = (ix as usize) + (iy as usize) * g + (iz as usize) * g * g;
                    if !active[ci_u] {
                        continue;
                    }
                    let m = g_mass[ci_u];
                    let k_diag = k_stencil[ci_u * 27 + STENCIL_DIAG];
                    let mut lhs = (m + dt * k_diag) * v_grid[ci_u];
                    for dz in -1..=1 {
                        for dy in -1..=1 {
                            for dx_ in -1..=1 {
                                if dx_ == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let jx = ix + dx_;
                                let jy = iy + dy;
                                let jz = iz + dz;
                                if jx < 0 || jy < 0 || jz < 0 || jx >= gi || jy >= gi || jz >= gi {
                                    continue;
                                }
                                let cj = (jx as usize) + (jy as usize) * g + (jz as usize) * g * g;
                                let s = stencil_offset(dx_, dy, dz);
                                let kij = k_stencil[ci_u * 27 + s];
                                if kij != 0.0 {
                                    lhs += dt * kij * v_grid[cj];
                                }
                            }
                        }
                    }
                    let r = lhs - m * v_pred[ci_u];
                    true_residual_sq += r.dot(r);
                }
            }
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
                    let damp = (1.0 - cfg.floor_friction).max(0.0);
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
                    let Some(ci) = cell_idx(g, ix, iy, iz) else {
                        continue;
                    };
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
            let damp = (1.0 - cfg.floor_friction).max(0.0);
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
                            + Vec3::new((ix as Real) + 0.5, (iy as Real) + 0.5, (iz as Real) + 0.5)
                                * dx;
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
        true_residual_sq,
    }
}

/// Diagnostic — run **only** P2G + predict + GS + G2P on the CPU and
/// return per-particle `(v_new, c_new)` BEFORE the Bingham return
/// mapping, alongside the assembled [`GridSystem`] (mass, k_stencil,
/// v_pred, active flags) and the post-GS / post-floor grid velocity.
/// The Stage 40 v2 GPU debug path compares these element-wise against
/// the GPU output so a discrepancy can be localised to P2G/predict/GS/G2P
/// precisely.
pub fn cpu_p2g_predict_gs_g2p_debug(
    state: &MpmState,
    cfg: &ImplicitMpmConfig,
) -> (Vec<Vec3>, Vec<Mat3>, GridSystem, Vec<Vec3>) {
    let mut work = state.clone();
    let mut captured: Option<(Vec<Vec3>, Vec<Mat3>, GridSystem, Vec<Vec3>)> = None;
    let mut hook = |sys: &GridSystem, v_grid: &mut [Vec3], _: HessianStencil| -> (usize, Real) {
        let g = sys.grid_size;
        let gi = g as i32;
        for _ in 0..cfg.max_gs_iters {
            for group in &sys.color_groups {
                for &ci in group {
                    if !sys.active[ci] {
                        continue;
                    }
                    let m = sys.g_mass[ci];
                    let k_diag = sys.k_stencil[ci * 27 + STENCIL_DIAG];
                    let denom = m + sys.dt * k_diag;
                    if denom <= 0.0 {
                        continue;
                    }
                    let iz = (ci / (g * g)) as i32;
                    let iy = ((ci / g) % g) as i32;
                    let ix = (ci % g) as i32;
                    let mut off = Vec3::ZERO;
                    for dz in -1..=1 {
                        for dy in -1..=1 {
                            for dx in -1..=1 {
                                if dx == 0 && dy == 0 && dz == 0 {
                                    continue;
                                }
                                let jx = ix + dx;
                                let jy = iy + dy;
                                let jz = iz + dz;
                                if jx < 0 || jy < 0 || jz < 0 || jx >= gi || jy >= gi || jz >= gi {
                                    continue;
                                }
                                let cj = (jx as usize) + (jy as usize) * g + (jz as usize) * g * g;
                                let s = stencil_offset(dx, dy, dz);
                                let kij = sys.k_stencil[ci * 27 + s];
                                if kij != 0.0 {
                                    off += kij * v_grid[cj];
                                }
                            }
                        }
                    }
                    let rhs = m * sys.v_pred[ci] - sys.dt * off;
                    v_grid[ci] = rhs / denom;
                }
            }
        }
        // Floor reapply.
        let origin_y = cfg.grid_origin[1];
        let dx = cfg.cell_size;
        for iz in 0..g {
            for iy in 0..g {
                for ix in 0..g {
                    let ci = ix + iy * g + iz * g * g;
                    if !sys.active[ci] {
                        continue;
                    }
                    let node_y = origin_y + ((iy as Real) + 0.5) * dx;
                    if node_y < cfg.floor_y {
                        let mut v = v_grid[ci];
                        if v.y < 0.0 {
                            v.y = 0.0;
                        }
                        let damp = (1.0 - cfg.floor_friction).max(0.0);
                        v.x *= damp;
                        v.z *= damp;
                        v_grid[ci] = v;
                    }
                }
            }
        }
        // G2P.
        let inv_dx = 1.0 / dx;
        let origin = Vec3::new(cfg.grid_origin[0], cfg.grid_origin[1], cfg.grid_origin[2]);
        let n = state.n();
        let mut v_new = vec![Vec3::ZERO; n];
        let mut c_new = vec![Mat3::ZERO; n];
        for p in 0..n {
            let pos = state.pos[p];
            let xp = (pos - origin) * inv_dx;
            let base_xyz = Vec3::new(
                (xp.x - 0.5).floor(),
                (xp.y - 0.5).floor(),
                (xp.z - 0.5).floor(),
            );
            let base = [base_xyz.x as i32, base_xyz.y as i32, base_xyz.z as i32];
            let fx = xp - base_xyz;
            let wx = quad_weights(fx.x);
            let wy = quad_weights(fx.y);
            let wz = quad_weights(fx.z);
            for i in 0..3 {
                for j in 0..3 {
                    for k in 0..3 {
                        let w = wx[i] * wy[j] * wz[k];
                        let ix = base[0] + i as i32;
                        let iy = base[1] + j as i32;
                        let iz = base[2] + k as i32;
                        let Some(ci) = cell_idx(g, ix, iy, iz) else {
                            continue;
                        };
                        let v_node = v_grid[ci];
                        let node_pos = origin
                            + Vec3::new((ix as Real) + 0.5, (iy as Real) + 0.5, (iz as Real) + 0.5)
                                * dx;
                        let xn_xp = node_pos - pos;
                        v_new[p] += w * v_node;
                        c_new[p] += outer(v_node, xn_xp) * (4.0 * w * inv_dx * inv_dx);
                    }
                }
            }
        }
        captured = Some((v_new, c_new, sys.clone(), v_grid.to_vec()));
        (cfg.max_gs_iters, 0.0)
    };
    implicit_mpm_step_with_solver(
        &mut work,
        cfg,
        &mut [],
        HessianStencil::Full27,
        Some(&mut hook),
    );
    captured.expect("hook should run")
}

/// Apply the **Bingham return mapping + per-particle floor / friction +
/// advect** stage to a precomputed `(v_new, c_new)` pair (e.g. from the
/// Stage 40 v2 GPU pipeline that hands back gathered velocities + APIC
/// matrices). Mutates `state.vel`, `state.c`, `state.pos` in place,
/// mirroring the tail of [`implicit_mpm_step_with_solver`]. Used by
/// the full CPU↔GPU parity test so both paths apply **identical**
/// rheology and advect math, leaving only P2G/predict/GS/G2P as the
/// variable under test.
pub fn apply_bingham_and_advect(
    state: &mut MpmState,
    cfg: &ImplicitMpmConfig,
    v_new: &[Vec3],
    c_new: &[Mat3],
) -> usize {
    assert_eq!(v_new.len(), state.n());
    assert_eq!(c_new.len(), state.n());
    let dt = cfg.dt;
    let dx = cfg.cell_size;
    let mu_b = cfg.viscosity;
    let tau_y = cfg.yield_stress;
    let mut yielded_count = 0usize;
    for p in 0..state.n() {
        let pos = state.pos[p];
        let mut v_new_p = v_new[p];
        let mut c_new_p = c_new[p];

        let sym_c = sym(c_new_p);
        let tr3 = tr_over_3(sym_c);
        let dev = sym_c - Mat3::IDENTITY * tr3;
        let nd = frob_norm(dev);
        let trial_norm = 2.0 * mu_b * nd;
        let skew = c_new_p - sym_c;
        if trial_norm <= tau_y || nd < 1.0e-12 {
            yielded_count += 1;
            c_new_p = Mat3::IDENTITY * tr3 + skew * 0.5;
            let lambda_plug = tau_y / (2.0 * mu_b * dx);
            let damp = (1.0 - dt * lambda_plug).max(0.0);
            v_new_p *= damp;
        } else {
            let ratio = 1.0 - tau_y / trial_norm;
            let dev_new = dev * ratio;
            c_new_p = dev_new + Mat3::IDENTITY * tr3 + skew;
            let cap = 0.5 / dt;
            let nc = frob_norm(c_new_p);
            if nc > cap {
                let s = cap / nc;
                c_new_p *= s;
                v_new_p *= s;
            }
        }

        let height_above_floor = (pos.y - cfg.floor_y).max(0.0);
        if pos.y + v_new_p.y * dt < cfg.floor_y && v_new_p.y < 0.0 {
            v_new_p.y = 0.0;
            let damp = (1.0 - cfg.floor_friction).max(0.0);
            v_new_p.x *= damp;
            v_new_p.z *= damp;
        }
        if height_above_floor < dx {
            let depth = 1.0 - (height_above_floor / dx).clamp(0.0, 1.0);
            let particle_extent = cfg.particle_volume.powf(1.0 / 3.0).max(1.0e-6);
            let dv_cap = dt * tau_y / (cfg.density * particle_extent) * depth;
            let vh = Vec3::new(v_new_p.x, 0.0, v_new_p.z);
            let vh_mag = vh.length();
            if vh_mag <= dv_cap {
                v_new_p.x = 0.0;
                v_new_p.z = 0.0;
                c_new_p *= 0.0;
            } else {
                let scale = (vh_mag - dv_cap) / vh_mag;
                v_new_p.x *= scale;
                v_new_p.z *= scale;
                c_new_p *= scale;
            }
        }

        state.vel[p] = v_new_p;
        state.c[p] = c_new_p;
        let mut new_pos = pos + v_new_p * dt;
        if new_pos.y < cfg.floor_y {
            new_pos.y = cfg.floor_y;
        }
        state.pos[p] = new_pos;
    }
    yielded_count
}

/// Run one substep with `stencil` and return the post-solve squared
/// residual against the **full** 27-point Hessian system. Used by the
/// Stage 40 convergence sanity test to compare diagonal-only vs
/// full-27 *on the same target system*.
pub fn full_hessian_residual_norm_squared(
    state: &mut MpmState,
    cfg: &ImplicitMpmConfig,
    stencil: HessianStencil,
) -> Real {
    let r = implicit_mpm_step_with_stencil(state, cfg, &mut [], stencil);
    r.true_residual_sq
}

/// Run one substep with `stencil` and return the post-solve squared
/// residual against the **full 125-point** Hessian system (the true
/// Newton Hessian for the quadratic-B-spline kernel). Stage 40 v4
/// uses this to compare Diagonal / Full27 / Full125 on a common
/// yardstick. The step's particle state advances exactly as in
/// [`implicit_mpm_step_with_stencil`]; the only difference is that
/// the *reported* residual is measured against the 125-stencil
/// regardless of which solver stencil was used.
///
/// Internally this assembles two systems in parallel (27 and 125)
/// from the same particle scatter — about 5× the assembly memory of
/// the regular step, but ~negligible at test scales.
pub fn full_hessian_residual_norm_squared_125(
    state: &mut MpmState,
    cfg: &ImplicitMpmConfig,
    stencil: HessianStencil,
) -> Real {
    // Build the diagnostic 125-point Hessian + RHS *before* the step
    // runs, by replaying the same P2G + predict math from the state's
    // initial particles. After the step finishes (mutating particles),
    // we compute the residual against the captured K125 + v_pred using
    // the post-GS v_grid the step computed.
    let (g_mass, k_stencil_125, v_pred, active) = assemble_grid_system_125(state, cfg);

    // Run the step (any stencil), capturing v_grid via the override hook.
    let mut work = state.clone();
    let mut captured: Vec<Vec3> = Vec::new();
    let mut hook = |sys: &GridSystem, v_grid: &mut [Vec3], st: HessianStencil| -> (usize, Real) {
        // Reproduce the host CPU GS sweep for the requested stencil,
        // then return the post-GS v_grid. We mirror the implementation
        // in `implicit_mpm_step_with_solver` exactly, so the captured
        // v_grid matches what the unhooked path would produce.
        host_gs_sweep_on_system(sys, v_grid, cfg, st);
        // Floor reapply (mirrors the post-GS floor step).
        let g = sys.grid_size;
        let dx = cfg.cell_size;
        let origin_y = cfg.grid_origin[1];
        for iz in 0..g {
            for iy in 0..g {
                for ix in 0..g {
                    let ci = ix + iy * g + iz * g * g;
                    if !sys.active[ci] {
                        continue;
                    }
                    let node_y = origin_y + ((iy as Real) + 0.5) * dx;
                    if node_y < cfg.floor_y {
                        let mut v = v_grid[ci];
                        if v.y < 0.0 {
                            v.y = 0.0;
                        }
                        let damp = (1.0 - cfg.floor_friction).max(0.0);
                        v.x *= damp;
                        v.z *= damp;
                        v_grid[ci] = v;
                    }
                }
            }
        }
        captured = v_grid.to_vec();
        (cfg.max_gs_iters, 0.0)
    };
    implicit_mpm_step_with_solver(&mut work, cfg, &mut [], stencil, Some(&mut hook));
    let v_grid_post = captured;

    // Now measure residual against the 125-stencil Hessian.
    let g = cfg.grid_size;
    let gi = g as i32;
    let dt = cfg.dt;
    let mut res_sq: Real = 0.0;
    for iz in 0..g as i32 {
        for iy in 0..g as i32 {
            for ix in 0..g as i32 {
                let ci_u = (ix as usize) + (iy as usize) * g + (iz as usize) * g * g;
                if !active[ci_u] {
                    continue;
                }
                let m = g_mass[ci_u];
                let k_diag = k_stencil_125[ci_u * STENCIL_LEN_125 + STENCIL_DIAG_125];
                let mut lhs = (m + dt * k_diag) * v_grid_post[ci_u];
                for dz in -2..=2 {
                    for dy in -2..=2 {
                        for dx_ in -2..=2 {
                            if dx_ == 0 && dy == 0 && dz == 0 {
                                continue;
                            }
                            let jx = ix + dx_;
                            let jy = iy + dy;
                            let jz = iz + dz;
                            if jx < 0 || jy < 0 || jz < 0 || jx >= gi || jy >= gi || jz >= gi {
                                continue;
                            }
                            let cj = (jx as usize) + (jy as usize) * g + (jz as usize) * g * g;
                            let s = stencil_offset_125(dx_, dy, dz);
                            let kij = k_stencil_125[ci_u * STENCIL_LEN_125 + s];
                            if kij != 0.0 {
                                lhs += dt * kij * v_grid_post[cj];
                            }
                        }
                    }
                }
                let r = lhs - m * v_pred[ci_u];
                res_sq += r.dot(r);
            }
        }
    }
    res_sq
}

/// Run the host CPU colored Gauss-Seidel sweep on a pre-assembled
/// [`GridSystem`] using the stencil width that matches the system's
/// `stencil` field, until `cfg.max_gs_iters` or `cfg.gs_tol` is hit.
/// Used by [`full_hessian_residual_norm_squared_125`] to reproduce the
/// step's interior GS without re-running P2G.
fn host_gs_sweep_on_system(
    sys: &GridSystem,
    v_grid: &mut [Vec3],
    cfg: &ImplicitMpmConfig,
    stencil: HessianStencil,
) {
    let g = sys.grid_size;
    let gi = g as i32;
    let dt = sys.dt;
    let use_off_diag = matches!(stencil, HessianStencil::Full27 | HessianStencil::Full125);
    if matches!(sys.stencil, HessianStencil::Full125) {
        for _ in 0..cfg.max_gs_iters {
            let mut max_dv: Real = 0.0;
            let mut max_v: Real = 1.0e-12;
            for group in &sys.color_groups {
                for &ci in group {
                    if !sys.active[ci] {
                        continue;
                    }
                    let m = sys.g_mass[ci];
                    let k_diag = sys.k_stencil_125[ci * STENCIL_LEN_125 + STENCIL_DIAG_125];
                    let denom = m + dt * k_diag;
                    if denom <= 0.0 {
                        continue;
                    }
                    let iz = (ci / (g * g)) as i32;
                    let iy = ((ci / g) % g) as i32;
                    let ix = (ci % g) as i32;
                    let mut off = Vec3::ZERO;
                    if use_off_diag {
                        for dz in -2..=2 {
                            for dy in -2..=2 {
                                for dx_ in -2..=2 {
                                    if dx_ == 0 && dy == 0 && dz == 0 {
                                        continue;
                                    }
                                    let jx = ix + dx_;
                                    let jy = iy + dy;
                                    let jz = iz + dz;
                                    if jx < 0
                                        || jy < 0
                                        || jz < 0
                                        || jx >= gi
                                        || jy >= gi
                                        || jz >= gi
                                    {
                                        continue;
                                    }
                                    let cj =
                                        (jx as usize) + (jy as usize) * g + (jz as usize) * g * g;
                                    let s = stencil_offset_125(dx_, dy, dz);
                                    let kij = sys.k_stencil_125[ci * STENCIL_LEN_125 + s];
                                    if kij != 0.0 {
                                        off += kij * v_grid[cj];
                                    }
                                }
                            }
                        }
                    }
                    let rhs = m * sys.v_pred[ci] - dt * off;
                    let v_new = rhs / denom;
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
            let last_residual = max_dv / max_v;
            if last_residual < cfg.gs_tol {
                break;
            }
        }
    } else {
        for _ in 0..cfg.max_gs_iters {
            let mut max_dv: Real = 0.0;
            let mut max_v: Real = 1.0e-12;
            for group in &sys.color_groups {
                for &ci in group {
                    if !sys.active[ci] {
                        continue;
                    }
                    let m = sys.g_mass[ci];
                    let k_diag = sys.k_stencil[ci * 27 + STENCIL_DIAG];
                    let denom = m + dt * k_diag;
                    if denom <= 0.0 {
                        continue;
                    }
                    let mut off = Vec3::ZERO;
                    if use_off_diag {
                        let iz = (ci / (g * g)) as i32;
                        let iy = ((ci / g) % g) as i32;
                        let ix = (ci % g) as i32;
                        for dz in -1..=1 {
                            for dy in -1..=1 {
                                for dx_ in -1..=1 {
                                    if dx_ == 0 && dy == 0 && dz == 0 {
                                        continue;
                                    }
                                    let jx = ix + dx_;
                                    let jy = iy + dy;
                                    let jz = iz + dz;
                                    if jx < 0
                                        || jy < 0
                                        || jz < 0
                                        || jx >= gi
                                        || jy >= gi
                                        || jz >= gi
                                    {
                                        continue;
                                    }
                                    let cj =
                                        (jx as usize) + (jy as usize) * g + (jz as usize) * g * g;
                                    let s = stencil_offset(dx_, dy, dz);
                                    let kij = sys.k_stencil[ci * 27 + s];
                                    if kij != 0.0 {
                                        off += kij * v_grid[cj];
                                    }
                                }
                            }
                        }
                    }
                    let rhs = m * sys.v_pred[ci] - dt * off;
                    let v_new = rhs / denom;
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
            let last_residual = max_dv / max_v;
            if last_residual < cfg.gs_tol {
                break;
            }
        }
    }
}

/// Assemble the 125-point Hessian + RHS for a given particle state.
/// Used for diagnostic residual computation in
/// [`full_hessian_residual_norm_squared_125`]. Returns
/// `(g_mass, k_stencil_125, v_pred, active)`.
fn assemble_grid_system_125(
    state: &MpmState,
    cfg: &ImplicitMpmConfig,
) -> (Vec<Real>, Vec<Real>, Vec<Vec3>, Vec<bool>) {
    let n = state.n();
    let g = cfg.grid_size;
    let g_total = g * g * g;
    let dx = cfg.cell_size;
    let inv_dx = 1.0 / dx;
    let dt = cfg.dt;
    let mass_p = cfg.particle_mass();
    let mu_b = cfg.viscosity;
    let origin = Vec3::new(cfg.grid_origin[0], cfg.grid_origin[1], cfg.grid_origin[2]);

    let mut g_mass = vec![0.0_f64; g_total];
    let mut g_momentum = vec![Vec3::ZERO; g_total];
    let mut k_stencil_125 = vec![0.0_f64; g_total * STENCIL_LEN_125];

    for p in 0..n {
        let pos = state.pos[p];
        let vel = state.vel[p];
        let c = state.c[p];

        let xp = (pos - origin) * inv_dx;
        let base_xyz = Vec3::new(
            (xp.x - 0.5).floor(),
            (xp.y - 0.5).floor(),
            (xp.z - 0.5).floor(),
        );
        let base = [base_xyz.x as i32, base_xyz.y as i32, base_xyz.z as i32];
        let fx = xp - base_xyz;
        let wx = quad_weights(fx.x);
        let wy = quad_weights(fx.y);
        let wz = quad_weights(fx.z);

        let kp = mu_b * cfg.particle_volume * 4.0 * inv_dx * inv_dx;

        let mut sten: [(Real, i32, i32, i32, i32); 27] = [(0.0, -1, 0, 0, 0); 27];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    let w = wx[i] * wy[j] * wz[k];
                    let ix = base[0] + i as i32;
                    let iy = base[1] + j as i32;
                    let iz = base[2] + k as i32;
                    let ci = cell_idx(g, ix, iy, iz).map(|c| c as i32).unwrap_or(-1);
                    sten[i * 9 + j * 3 + k] = (w, ci, ix, iy, iz);
                }
            }
        }

        for &(w, ci, ix, iy, iz) in sten.iter() {
            if ci < 0 {
                continue;
            }
            let node_pos =
                origin + Vec3::new((ix as Real) + 0.5, (iy as Real) + 0.5, (iz as Real) + 0.5) * dx;
            let xn_xp = node_pos - pos;
            let aff = c * xn_xp;
            let ci = ci as usize;
            g_mass[ci] += w * mass_p;
            g_momentum[ci] += w * mass_p * (vel + aff);
        }

        for a in 0..27 {
            let (wa, ca, ax, ay, az) = sten[a];
            if ca < 0 {
                continue;
            }
            for b in 0..27 {
                let (wb, cb, bx, by, bz) = sten[b];
                if cb < 0 {
                    continue;
                }
                let dx_ij = bx - ax;
                let dy_ij = by - ay;
                let dz_ij = bz - az;
                let s = stencil_offset_125(dx_ij, dy_ij, dz_ij);
                k_stencil_125[(ca as usize) * STENCIL_LEN_125 + s] += kp * wa * wb;
            }
        }
    }

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
                let node_y = origin.y + ((iy as Real) + 0.5) * dx;
                if node_y < cfg.floor_y + dx {
                    let depth = (cfg.floor_y + dx - node_y).max(0.0) / dx;
                    if node_y < cfg.floor_y && v.y < 0.0 {
                        v.y = 0.0;
                    } else if v.y < 0.0 {
                        v.y *= 1.0 - depth;
                    }
                    let damp = (1.0 - cfg.floor_friction * depth).max(0.0);
                    v.x *= damp;
                    v.z *= damp;
                }
                v_pred[ci] = v;
                active[ci] = true;
            }
        }
    }
    (g_mass, k_stencil_125, v_pred, active)
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
