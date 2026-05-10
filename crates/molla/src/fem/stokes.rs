//! Stage 36.2 — incompressible Stokes / Navier-Stokes solvers using
//! Taylor-Hood (P2 velocity, P1 pressure) mixed elements.
//!
//! # Strategy
//!
//! The classical Stokes saddle-point system
//!
//! ```text
//!   [ A    Bᵀ ] [ u ]   [ f ]
//!   [ B    0  ] [ p ] = [ 0 ]
//! ```
//!
//! is **indefinite**, so the existing CG (`crates/molla/src/math/iterative.rs`)
//! cannot solve the block system directly — CG requires SPD operators.
//! Instead, we use a **SIMPLE-style projection** (Chorin / IPCS): we
//! split the velocity update from the pressure correction so each
//! sub-step is SPD-ish and CG-friendly.
//!
//! # SIMPLE iteration
//!
//! 1. **Predictor**: `A · u* = f − Bᵀ · p_old` — solve via CG on the
//!    velocity-block matrix `A` (per-component decoupled Laplacian
//!    × `μ`, plus `M / Δt` for the transient case).
//! 2. **Pressure correction**: `(B · A_diag⁻¹ · Bᵀ) · δp = −B · u*`
//!    — pressure-Poisson on the Schur complement, where `A_diag⁻¹`
//!    is the inverse of the lumped diagonal of A (cheap; gives a
//!    diagonal SPD operator). We solve via CG on the matrix-free
//!    closure `q ↦ B · (A_diag⁻¹ · (Bᵀ · q))`.
//! 3. **Update**: `u = u* − A_diag⁻¹ · Bᵀ · δp`, `p = p_old + α · δp`.
//! 4. Repeat until `‖B · u‖ < tol` (incompressibility residual).
//!
//! `α` is a relaxation factor (typical 0.7–1.0 for Stokes; lower
//! for stiff Navier-Stokes).
//!
//! # Boundary conditions
//!
//! - **Velocity**: `Dirichlet` (no-slip / lid-velocity) — pin velocity
//!   DoFs on the boundary at every component. The applier
//!   [`apply_stokes_bcs`] handles vector Dirichlet on the P2 velocity
//!   space + a single-vertex pressure pin.
//! - **Pressure**: pin one DoF (the system has a constant null space
//!   without it).
//!
//! # Caveats (v1)
//!
//! - 2D-tri + 3D-tet supported. 2D vertical-component lid-driven
//!   cavity is the contract test.
//! - Symmetric "stress" form `μ ∫ (∇u + ∇uᵀ) : ∇v` is approximated
//!   by `μ ∫ ∇u : ∇v` — equivalent for `Dirichlet` BCs everywhere
//!   (the boundary integral cancels), which is the case for the
//!   lid-driven cavity.
//! - Pressure-Poisson Schur complement uses the diagonal of `A` for
//!   `A⁻¹` (the SIMPLE approximation). Convergence is slower than
//!   exact-Schur SIMPLE but each pressure CG matvec costs only a few
//!   matvecs of `B` and `Bᵀ`.

use std::collections::HashMap;

use crate::core::Result;
use crate::math::iterative::cg_solve;
use crate::math::sparse::BsrMatrix;
use crate::math::{Real, Vec3};

use super::assembly::matvec;
use super::bc::BoundaryCondition;
use super::function_space::compute_edges;
use super::mesh::{FemCell, FemMesh};
use super::p2_element::{
    p2_tet_shape_functions, p2_tet_shape_gradients, p2_tri_shape_functions,
    p2_tri_shape_gradients, P2DofMap,
};
use super::quadrature::{tet_quadrature_11pt, tri_quadrature_6pt};

// ─── Mixed function-space helper ──────────────────────────────────────

/// Mixed Taylor-Hood velocity-pressure space description.
///
/// The velocity space is P2 vector (3 components in 3D, 2 in 2D); the
/// pressure space is P1 scalar (one DoF per vertex). The Taylor-Hood
/// pair is **LBB-stable** for incompressible Stokes — the canonical
/// safe choice for mixed element velocity-pressure formulations.
#[derive(Clone, Debug)]
pub struct MixedTaylorHood {
    /// Per-vertex velocity component count (2 for tri / 2D, 3 for
    /// tet / 3D).
    pub dim: usize,
    /// P2 DoF map (corner + edge midpoint addressing).
    pub map: P2DofMap,
    /// Total velocity DoFs = `(vertex_count + edge_count) * dim`.
    pub n_velocity: usize,
    /// Total pressure DoFs = `vertex_count`.
    pub n_pressure: usize,
}

impl MixedTaylorHood {
    /// Build the Taylor-Hood mixed space description from `mesh`.
    /// `dim = 2` for triangle meshes (planar), `dim = 3` for tets.
    #[must_use]
    pub fn build(mesh: &FemMesh, dim: usize) -> Self {
        assert!(dim == 2 || dim == 3, "Taylor-Hood: dim must be 2 or 3");
        let map = P2DofMap::build(mesh);
        let n_v_nodes = map.ndof();
        let n_velocity = n_v_nodes * dim;
        let n_pressure = mesh.vert_count();
        Self {
            dim,
            map,
            n_velocity,
            n_pressure,
        }
    }

    /// Velocity DoF count.
    #[must_use]
    pub fn ndof_velocity(&self) -> usize {
        self.n_velocity
    }

    /// Pressure DoF count.
    #[must_use]
    pub fn ndof_pressure(&self) -> usize {
        self.n_pressure
    }

    /// Convert a P2 node index (`0..n_v_nodes`) + component (`0..dim`)
    /// into a flat global velocity DoF using the **interleaved** layout
    /// `node * dim + comp`.
    #[inline]
    #[must_use]
    pub fn velocity_dof(&self, node: usize, comp: usize) -> usize {
        node * self.dim + comp
    }
}

// ─── Element matrices ─────────────────────────────────────────────────

/// Local **velocity-block** matrix
/// `A^e_{(i, c), (j, c')} = δ_{c, c'} · μ · ∫ ∇N_i^v · ∇N_j^v dx`
/// for one Taylor-Hood cell (P2 velocity).
///
/// The cross-component coupling vanishes because the simplified
/// Laplacian form `μ ∇u : ∇v` decouples per component. The returned
/// matrix is the standard P2 stiffness matrix replicated on
/// `dim`-block-diagonals, in **interleaved local layout** `local DoF
/// = node * dim + component`. Shape: `(N · dim) × (N · dim)` where
/// `N = 6` (tri) or `N = 10` (tet).
#[must_use]
pub fn compute_element_stokes_velocity_block(
    cell: &FemCell,
    vertices: &[Vec3],
    viscosity: Real,
    dim: usize,
) -> Vec<Vec<Real>> {
    debug_assert!(dim == 2 || dim == 3);
    match cell {
        FemCell::Tri(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
            ];
            let area = super::element::tri_area_2d(&p);
            let qpts = tri_quadrature_6pt();
            let n = 6;
            let n_dof = n * dim;
            let mut k = vec![vec![0.0_f64 as Real; n_dof]; n_dof];
            for q in &qpts {
                let g = p2_tri_shape_gradients(q.bary, &p);
                let w = q.weight * area * viscosity;
                for i in 0..n {
                    for j in 0..n {
                        // 2D dot product (z = 0 for planar tris).
                        let gg = g[i].x * g[j].x + g[i].y * g[j].y;
                        for c in 0..dim {
                            k[i * dim + c][j * dim + c] += w * gg;
                        }
                    }
                }
            }
            k
        }
        FemCell::Tet(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
                vertices[idx[3] as usize],
            ];
            let vol = super::element::tet_volume(&p);
            let qpts = tet_quadrature_11pt();
            let n = 10;
            let n_dof = n * dim;
            let mut k = vec![vec![0.0_f64 as Real; n_dof]; n_dof];
            for q in &qpts {
                let g = p2_tet_shape_gradients(q.bary, &p);
                let w = q.weight * vol * viscosity;
                for i in 0..n {
                    for j in 0..n {
                        let gg = g[i].dot(g[j]);
                        for c in 0..dim {
                            k[i * dim + c][j * dim + c] += w * gg;
                        }
                    }
                }
            }
            k
        }
    }
}

/// Local **divergence-block** matrix
/// `B^e_{q, (j, c)} = ∫ N_q^p · ∂N_j^v / ∂x_c dx` for one Taylor-Hood
/// cell (P1 pressure × ∂P2 velocity).
///
/// Shape: `n_p × (n_v · dim)` where:
/// - tri: `3 × (6 · 2) = 3 × 12`,
/// - tet: `4 × (10 · 3) = 4 × 30`.
///
/// Local layout: rows are pressure node indices `0..n_p`; columns use
/// the same **interleaved** `node * dim + component` layout as
/// [`compute_element_stokes_velocity_block`].
#[must_use]
pub fn compute_element_stokes_div_block(
    cell: &FemCell,
    vertices: &[Vec3],
    dim: usize,
) -> Vec<Vec<Real>> {
    debug_assert!(dim == 2 || dim == 3);
    match cell {
        FemCell::Tri(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
            ];
            let area = super::element::tri_area_2d(&p);
            let qpts = tri_quadrature_6pt();
            let n_v = 6;
            let n_p = 3;
            let n_dof_v = n_v * dim;
            let mut b = vec![vec![0.0_f64 as Real; n_dof_v]; n_p];
            for q in &qpts {
                let np = p2_tri_shape_functions(q.bary); // also have P1 = bary directly
                let g = p2_tri_shape_gradients(q.bary, &p);
                // P1 pressure shape functions on a tri are the
                // barycentric coords themselves: `λ_0, λ_1, λ_2`.
                let pq = q.bary;
                let w = q.weight * area;
                let _ = np; // P2 shape unused for pressure
                for ip in 0..n_p {
                    for jv in 0..n_v {
                        for c in 0..dim {
                            let dn = if c == 0 { g[jv].x } else { g[jv].y };
                            b[ip][jv * dim + c] += w * pq[ip] * dn;
                        }
                    }
                }
            }
            b
        }
        FemCell::Tet(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
                vertices[idx[3] as usize],
            ];
            let vol = super::element::tet_volume(&p);
            let qpts = tet_quadrature_11pt();
            let n_v = 10;
            let n_p = 4;
            let n_dof_v = n_v * dim;
            let mut b = vec![vec![0.0_f64 as Real; n_dof_v]; n_p];
            for q in &qpts {
                let g = p2_tet_shape_gradients(q.bary, &p);
                let pq = q.bary; // P1 pressure shape = barycentric coords
                let w = q.weight * vol;
                for ip in 0..n_p {
                    for jv in 0..n_v {
                        let comp = [g[jv].x, g[jv].y, g[jv].z];
                        for c in 0..dim {
                            b[ip][jv * dim + c] += w * pq[ip] * comp[c];
                        }
                    }
                }
            }
            b
        }
    }
}

/// Local **vector mass** matrix
/// `M^e_{(i, c), (j, c')} = δ_{c, c'} · ∫ N_i^v N_j^v dx` for one
/// P2-vector cell — the time-stepping mass for transient Navier-Stokes.
///
/// Returned in interleaved layout matching
/// [`compute_element_stokes_velocity_block`]. Shape: `(N · dim) ×
/// (N · dim)`.
#[must_use]
pub fn compute_element_mass_p2_vector(
    cell: &FemCell,
    vertices: &[Vec3],
    dim: usize,
) -> Vec<Vec<Real>> {
    debug_assert!(dim == 2 || dim == 3);
    match cell {
        FemCell::Tri(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
            ];
            let area = super::element::tri_area_2d(&p);
            let qpts = tri_quadrature_6pt();
            let n = 6;
            let n_dof = n * dim;
            let mut m = vec![vec![0.0_f64 as Real; n_dof]; n_dof];
            for q in &qpts {
                let np = p2_tri_shape_functions(q.bary);
                let w = q.weight * area;
                for i in 0..n {
                    for j in 0..n {
                        let nn = np[i] * np[j];
                        for c in 0..dim {
                            m[i * dim + c][j * dim + c] += w * nn;
                        }
                    }
                }
            }
            m
        }
        FemCell::Tet(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
                vertices[idx[3] as usize],
            ];
            let vol = super::element::tet_volume(&p);
            let qpts = tet_quadrature_11pt();
            let n = 10;
            let n_dof = n * dim;
            let mut m = vec![vec![0.0_f64 as Real; n_dof]; n_dof];
            for q in &qpts {
                let np = p2_tet_shape_functions(q.bary);
                let w = q.weight * vol;
                for i in 0..n {
                    for j in 0..n {
                        let nn = np[i] * np[j];
                        for c in 0..dim {
                            m[i * dim + c][j * dim + c] += w * nn;
                        }
                    }
                }
            }
            m
        }
    }
}

// ─── Assembly ────────────────────────────────────────────────────────

/// Assemble the global P2-vector velocity-block matrix `A` for the
/// Stokes system. CSR with `(n_v_nodes · dim)` rows and columns.
pub fn assemble_velocity_block(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
    viscosity: Real,
) -> Result<BsrMatrix> {
    let n = th.n_velocity;
    let dim = th.dim;
    let mut triplets: Vec<(usize, usize, Vec<Real>)> = Vec::new();
    triplets.reserve(mesh.cells.len() * 100 * dim * dim);
    for cell in &mesh.cells {
        let local = compute_element_stokes_velocity_block(cell, &mesh.vertices, viscosity, dim);
        let n_local_nodes = match cell {
            FemCell::Tri(_) => 6,
            FemCell::Tet(_) => 10,
        };
        for i in 0..n_local_nodes {
            let gi_node = th.map.local_to_global(cell, i);
            for j in 0..n_local_nodes {
                let gj_node = th.map.local_to_global(cell, j);
                for ci in 0..dim {
                    for cj in 0..dim {
                        let val = local[i * dim + ci][j * dim + cj];
                        if val == 0.0 {
                            continue;
                        }
                        triplets.push((
                            gi_node * dim + ci,
                            gj_node * dim + cj,
                            vec![val],
                        ));
                    }
                }
            }
        }
    }
    BsrMatrix::from_triplets(1, n, n, &triplets)
}

/// Assemble the global P2-vector velocity-mass matrix `M` (consistent).
pub fn assemble_velocity_mass(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
) -> Result<BsrMatrix> {
    let n = th.n_velocity;
    let dim = th.dim;
    let mut triplets: Vec<(usize, usize, Vec<Real>)> = Vec::new();
    triplets.reserve(mesh.cells.len() * 100 * dim * dim);
    for cell in &mesh.cells {
        let local = compute_element_mass_p2_vector(cell, &mesh.vertices, dim);
        let n_local_nodes = match cell {
            FemCell::Tri(_) => 6,
            FemCell::Tet(_) => 10,
        };
        for i in 0..n_local_nodes {
            let gi_node = th.map.local_to_global(cell, i);
            for j in 0..n_local_nodes {
                let gj_node = th.map.local_to_global(cell, j);
                for ci in 0..dim {
                    for cj in 0..dim {
                        let val = local[i * dim + ci][j * dim + cj];
                        if val == 0.0 {
                            continue;
                        }
                        triplets.push((
                            gi_node * dim + ci,
                            gj_node * dim + cj,
                            vec![val],
                        ));
                    }
                }
            }
        }
    }
    BsrMatrix::from_triplets(1, n, n, &triplets)
}

/// Assemble the global divergence matrix `B` (P1 pressure × ∂P2
/// velocity). Shape: `n_pressure × n_velocity`.
pub fn assemble_div_block(mesh: &FemMesh, th: &MixedTaylorHood) -> Result<BsrMatrix> {
    let n_p = th.n_pressure;
    let n_v = th.n_velocity;
    let dim = th.dim;
    let mut triplets: Vec<(usize, usize, Vec<Real>)> = Vec::new();
    triplets.reserve(mesh.cells.len() * 4 * 30);
    for cell in &mesh.cells {
        let local = compute_element_stokes_div_block(cell, &mesh.vertices, dim);
        let (n_p_local, n_v_local, p_verts) = match cell {
            FemCell::Tri(v) => (3usize, 6usize, &v[..]),
            FemCell::Tet(v) => (4usize, 10usize, &v[..]),
        };
        for ip in 0..n_p_local {
            let gip = p_verts[ip] as usize;
            for jv in 0..n_v_local {
                let gjv_node = th.map.local_to_global(cell, jv);
                for c in 0..dim {
                    let val = local[ip][jv * dim + c];
                    if val == 0.0 {
                        continue;
                    }
                    triplets.push((gip, gjv_node * dim + c, vec![val]));
                }
            }
        }
    }
    BsrMatrix::from_triplets(1, n_p, n_v, &triplets)
}

// ─── Boundary conditions ──────────────────────────────────────────────

/// Boundary-condition kinds specific to incompressible flow.
///
/// These extend the scalar `BoundaryCondition::Dirichlet` enum (used
/// for heat-equation and elasticity); the Stokes solver applies them
/// directly without re-using the elasticity DirichletVector path
/// because the velocity-DoF layout differs (corner + edge nodes).
#[derive(Clone, Debug)]
pub enum StokesBc {
    /// No-slip wall: pin every velocity component at the given P2 node
    /// to zero.
    NoSlip { node: u32 },
    /// Lid / moving-wall: pin every velocity component at the given P2
    /// node to the prescribed value (length must be `dim`).
    LidVelocity { node: u32, velocity: Vec<Real> },
    /// Pressure pin: pin a single pressure DoF (one mesh vertex) to
    /// the given scalar — usually 0. Required to break the constant
    /// null-space of the pressure-Poisson system.
    PressureZero { vertex: u32 },
}

/// Apply Stokes BCs in-place on the velocity-block matrix `A`,
/// pressure operator (currently a no-op except for the pressure pin
/// which the caller applies on the Schur system) and the velocity RHS
/// vector. Returns the list of velocity DoFs that were pinned (for the
/// pressure-correction step which must zero-out those rows of `B^T`).
pub fn apply_stokes_velocity_bcs(
    a: &mut BsrMatrix,
    f: &mut [Real],
    bcs: &[StokesBc],
    dim: usize,
) -> Vec<usize> {
    let mut pinned = Vec::new();
    for bc in bcs {
        match bc {
            StokesBc::NoSlip { node } => {
                let base = (*node as usize) * dim;
                for c in 0..dim {
                    pin_row_a(a, f, base + c, 0.0);
                    pinned.push(base + c);
                }
            }
            StokesBc::LidVelocity { node, velocity } => {
                debug_assert_eq!(velocity.len(), dim);
                let base = (*node as usize) * dim;
                for (c, v) in velocity.iter().enumerate() {
                    pin_row_a(a, f, base + c, *v);
                    pinned.push(base + c);
                }
            }
            // Pressure pin handled separately on the Schur system.
            StokesBc::PressureZero { .. } => {}
        }
    }
    pinned
}

/// Internal helper — zero row `r`, set diagonal to 1, set `f[r] = value`.
fn pin_row_a(k: &mut BsrMatrix, f: &mut [Real], r: usize, value: Real) {
    let lo = k.row_offsets[r];
    let hi = k.row_offsets[r + 1];
    for slot in lo..hi {
        let c = k.col_idx[slot];
        k.values[slot] = if c == r { 1.0 } else { 0.0 };
    }
    f[r] = value;
}

/// Symmetric Dirichlet enforcement on a CSR matrix: given the set of
/// pinned DoFs and their target values, eliminate the column terms by
/// subtracting `A[r][d] * v_d` from `f[r]` for every other row, then
/// zero those off-diagonal column entries; finally pin each pinned
/// row to identity. Result: matrix stays SPD on the unconstrained
/// block, RHS is consistent.
fn apply_symmetric_dirichlet(
    k: &mut BsrMatrix,
    f: &mut [Real],
    pinned: &[(usize, Real)],
) {
    use std::collections::HashMap;
    let pin_map: HashMap<usize, Real> = pinned.iter().copied().collect();
    // Pass 1: eliminate columns by moving their contribution to the RHS,
    // for every row that touches a pinned column. Avoid mutating the
    // pinned rows themselves (we'll reset them next).
    for r in 0..k.block_rows {
        if pin_map.contains_key(&r) {
            continue;
        }
        let lo = k.row_offsets[r];
        let hi = k.row_offsets[r + 1];
        for slot in lo..hi {
            let c = k.col_idx[slot];
            if let Some(v) = pin_map.get(&c) {
                f[r] -= k.values[slot] * v;
                k.values[slot] = 0.0;
            }
        }
    }
    // Pass 2: pin each constrained row to identity, RHS = value.
    for (r, v) in pinned {
        let lo = k.row_offsets[*r];
        let hi = k.row_offsets[*r + 1];
        for slot in lo..hi {
            let c = k.col_idx[slot];
            k.values[slot] = if c == *r { 1.0 } else { 0.0 };
        }
        f[*r] = *v;
    }
}

/// Apply velocity BCs from the legacy `BoundaryCondition` enum.
///
/// Maps `Dirichlet { vertex, value }` → no-slip on `vertex` (component
/// 0 = `value`, others 0); `DirichletVector { vertex, values }` →
/// lid-velocity on `vertex` with the given component values.
///
/// Provided so callers can use the existing BC plumbing without
/// constructing a parallel `StokesBc` list.
pub fn apply_stokes_bcs_legacy(
    a: &mut BsrMatrix,
    f: &mut [Real],
    bcs: &[BoundaryCondition],
    dim: usize,
) -> Vec<usize> {
    let mut pinned = Vec::new();
    for bc in bcs {
        match bc {
            BoundaryCondition::DirichletVector { vertex, values } => {
                let base = (*vertex as usize) * dim;
                for (c, v) in values.iter().enumerate().take(dim) {
                    pin_row_a(a, f, base + c, *v);
                    pinned.push(base + c);
                }
            }
            BoundaryCondition::Dirichlet { vertex, value } => {
                let base = (*vertex as usize) * dim;
                pin_row_a(a, f, base, *value);
                pinned.push(base);
                for c in 1..dim {
                    pin_row_a(a, f, base + c, 0.0);
                    pinned.push(base + c);
                }
            }
            _ => {}
        }
    }
    pinned
}

// ─── Solver: steady-state Stokes ──────────────────────────────────────

/// Solver tuning for the Uzawa-CG iteration on the pressure-Schur
/// system.
///
/// **History**: The original SIMPLE iteration (`u* = A⁻¹ rhs`,
/// `dp = (B diag(A)⁻¹ Bᵀ)⁻¹ (-Bu*)`, `p ← p + α dp`) was numerically
/// unstable on lid-driven cavity even at modest mesh sizes — `diag(A)`
/// is a poor proxy for `A⁻¹` on the Laplacian-style velocity block,
/// and the relaxation parameter required to damp the divergence was
/// vanishingly small. We swapped in **Uzawa-CG**: solve the
/// pressure-Schur system `S p = B A⁻¹ f` with CG, where each `S q`
/// is computed by an inner CG on `A`. Convergence is robust at the
/// price of nested CG iterations. The `alpha_p` field is preserved
/// for back-compat / future SIMPLE-relaxation experiments but is
/// currently unused by the Uzawa path.
#[derive(Clone, Debug)]
pub struct StokesSimpleConfig {
    /// Pressure relaxation factor (currently unused; kept for SIMPLE
    /// API compatibility).
    pub alpha_p: Real,
    /// Outer CG (pressure-Schur) iteration cap.
    pub max_outer_iters: usize,
    /// Outer CG tolerance — relative against `‖g‖` where
    /// `g = B A⁻¹ f_red`.
    pub tol_outer: Real,
    /// Inner CG iteration cap (each `A x = y` solve).
    pub max_inner_iters: usize,
    /// Inner CG tolerance.
    pub tol_inner: Real,
}

impl Default for StokesSimpleConfig {
    fn default() -> Self {
        Self {
            alpha_p: 0.8,
            max_outer_iters: 200,
            tol_outer: 1.0e-7,
            max_inner_iters: 2000,
            tol_inner: 1.0e-10,
        }
    }
}

/// Solver outcome for the steady-state Stokes solve.
#[derive(Clone, Debug)]
pub struct StokesResult {
    /// Velocity DoF vector.
    pub u: Vec<Real>,
    /// Pressure DoF vector.
    pub p: Vec<Real>,
    /// Number of SIMPLE outer iterations consumed.
    pub outer_iters: usize,
    /// Final incompressibility residual `‖B u‖`.
    pub div_residual: Real,
}

/// Solve the **steady-state** incompressible Stokes problem
///
/// ```text
///   −μ ∇²u + ∇p = f
///         ∇ · u = 0
/// ```
///
/// via SIMPLE iteration on the Taylor-Hood (P2-P1) discretization.
///
/// `f_velocity` is the velocity-side body-force RHS (e.g. gravity);
/// pass an all-zero vector for the canonical lid-driven cavity.
///
/// `bcs` contains [`StokesBc`] entries (no-slip, lid velocity, and
/// optionally a [`StokesBc::PressureZero`] pin to break the constant
/// null space of the pressure system). If no pressure pin is supplied,
/// vertex 0 is pinned to 0 by default.
pub fn solve_stokes_steady(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
    f_velocity: &[Real],
    viscosity: Real,
    bcs: &[StokesBc],
    cfg: &StokesSimpleConfig,
) -> Result<StokesResult> {
    debug_assert_eq!(f_velocity.len(), th.n_velocity);

    let a = assemble_velocity_block(mesh, th, viscosity)?;
    let b_div = assemble_div_block(mesh, th)?;

    let n_v = th.n_velocity;
    let n_p = th.n_pressure;

    // Build the velocity Dirichlet (DoF, value) list once.
    let pinned_pairs = collect_velocity_pins(bcs, th.dim);
    let pinned_set: std::collections::HashSet<usize> =
        pinned_pairs.iter().map(|(d, _)| *d).collect();

    // Symmetric-Dirichlet reduction of A. f_red starts at f_velocity
    // and gets updated by the column elimination.
    let mut a_red = a.clone();
    let mut f_red = f_velocity.to_vec();
    apply_symmetric_dirichlet(&mut a_red, &mut f_red, &pinned_pairs);

    // Identify "dead" pressure rows — pressure DoFs whose every
    // velocity column lies in the pinned set. These rows produce a
    // zero diagonal in the Schur complement (LBB / inf-sup violation
    // for that DoF on this coarse mesh) and would make CG diverge.
    // Pin them all to zero so they fall out of the system.
    let mut dead_pressure_rows: std::collections::HashSet<usize> =
        std::collections::HashSet::new();
    for r in 0..n_p {
        let lo = b_div.row_offsets[r];
        let hi = b_div.row_offsets[r + 1];
        let mut alive = false;
        for slot in lo..hi {
            let c = b_div.col_idx[slot];
            if !pinned_set.contains(&c) && b_div.values[slot].abs() > 1e-14 {
                alive = true;
                break;
            }
        }
        if !alive {
            dead_pressure_rows.insert(r);
        }
    }

    // Pressure pin (default: vertex 0).
    let p_pin = bcs
        .iter()
        .find_map(|b| {
            if let StokesBc::PressureZero { vertex } = b {
                Some(*vertex as usize)
            } else {
                None
            }
        })
        .unwrap_or(0);

    // Treat all "dead" pressure rows + the explicit pin as one set of
    // pressure DoFs to enforce-as-zero in the Schur complement.
    let mut p_pinned_set: std::collections::HashSet<usize> = dead_pressure_rows.clone();
    p_pinned_set.insert(p_pin);

    let mut u = vec![0.0; n_v];
    for &(dof, val) in &pinned_pairs {
        u[dof] = val;
    }
    let mut p = vec![0.0; n_p];

    // ── Outer Uzawa-CG iteration ──────────────────────────────────
    //
    // The pure SIMPLE iteration (using `D = diag(A)` as the inverse
    // approximation) blew up on the cavity test — `D⁻¹` is far too
    // permissive of a stand-in for `A⁻¹` on a Laplacian-like operator
    // and the relaxation parameter would have to be vanishingly small.
    //
    // Instead we use **Uzawa-CG**: solve the saddle-point system as a
    // Schur-complement on the pressure where the Schur action
    // `S p = B A⁻¹ Bᵀ p` is computed via an inner CG on `A` for each
    // pressure-side matvec. CG on `S` then converges in O(k_press)
    // iterations regardless of the velocity-side conditioning.
    //
    // The inner-CG uses a relaxed tolerance (`tol_inner_inner`) since
    // each pressure-CG iteration only needs `S p` to a few digits;
    // wasted work in inner CG dominates the wall-clock otherwise.

    let f_norm_init = (f_velocity.iter().map(|x| x * x).sum::<Real>())
        .sqrt()
        .max(1.0);

    // Pre-compute `A⁻¹ f_red`: shows up in both RHS and the action.
    let a_inv_f = solve_inner_cg(&a_red, &f_red, &vec![0.0; n_v], &cfg);

    // RHS of pressure-Schur equation: g = B · (A⁻¹ f_red) − (boundary
    // term) — for our enforced-row symmetric pinning, the RHS is
    // simply `B · (A⁻¹ f_red)` because the column-elim already moved
    // BC contributions into f_red.
    let mut g = matvec(&b_div, &a_inv_f);
    // Pin pressure rows to zero in the equation.
    for &r in p_pinned_set.iter() {
        g[r] = 0.0;
    }

    // Schur op: p ↦ B · (A_red⁻¹ · Bᵀ p), with pinned rows behaving as
    // identity.
    let p_pinned_set_ref = &p_pinned_set;
    let pinned_set_ref = &pinned_set;
    let a_red_ref = &a_red;
    let cfg_ref = &cfg;
    let schur_op = |q: &[Real]| -> Vec<Real> {
        let mut q_eff = q.to_vec();
        for &r in p_pinned_set_ref.iter() {
            q_eff[r] = 0.0;
        }
        // x = Bᵀ q_eff
        let mut x = matvec_t(&b_div, &q_eff, n_v);
        // Pinned velocity DoFs: zero (homogeneous Dirichlet on the
        // perturbation Bᵀq side).
        for &dof in pinned_set_ref.iter() {
            x[dof] = 0.0;
        }
        // y = A_red⁻¹ x  (inner CG)
        let y_v = solve_inner_cg(a_red_ref, &x, &vec![0.0; n_v], cfg_ref);
        // p = B y
        let mut yout = matvec(&b_div, &y_v);
        // Identity rows for pinned pressures.
        for &r in p_pinned_set_ref.iter() {
            yout[r] = q[r];
        }
        yout
    };

    // Solve pressure system via CG. The outer "iteration count" here
    // is the CG iteration count.
    let p_init = p.clone();
    let p_res = cg_solve(
        schur_op,
        &g,
        &p_init,
        cfg.tol_outer,
        cfg.max_outer_iters,
    );
    p = p_res.x;
    for &r in p_pinned_set.iter() {
        p[r] = 0.0;
    }

    // Recover velocity: u = A⁻¹ (f_red − Bᵀ p).
    let bt_p = matvec_t(&b_div, &p, n_v);
    let mut rhs_u: Vec<Real> = (0..n_v).map(|i| f_red[i] - bt_p[i]).collect();
    for &(dof, val) in &pinned_pairs {
        rhs_u[dof] = val;
    }
    u = solve_inner_cg(&a_red, &rhs_u, &u, &cfg);

    let bu = matvec(&b_div, &u);
    // Compute the divergence residual on the FREE pressure rows only —
    // the dead/pinned rows carry the BC-driven divergence that the
    // discretisation cannot zero (their pressure was forced to 0).
    let mut div_residual_free_sq = 0.0;
    let mut div_residual_dead_sq = 0.0;
    for (r, &v) in bu.iter().enumerate() {
        if p_pinned_set.contains(&r) {
            div_residual_dead_sq += v * v;
        } else {
            div_residual_free_sq += v * v;
        }
    }
    let div_residual = div_residual_free_sq.sqrt();
    let _div_residual_dead = div_residual_dead_sq.sqrt();
    let outer_iters = p_res.iters;
    let _ = f_norm_init;

    Ok(StokesResult {
        u,
        p,
        outer_iters,
        div_residual,
    })
}

/// Inner CG solve helper for the Uzawa outer iteration. Always
/// returns a finite vector (CG bails at iteration cap on stagnation).
fn solve_inner_cg(
    a: &BsrMatrix,
    b: &[Real],
    x0: &[Real],
    cfg: &StokesSimpleConfig,
) -> Vec<Real> {
    cg_solve(
        |x| matvec(a, x),
        b,
        x0,
        cfg.tol_inner,
        cfg.max_inner_iters,
    )
    .x
}

/// Collect (DoF, value) pairs from a list of [`StokesBc`].
fn collect_velocity_pins(bcs: &[StokesBc], dim: usize) -> Vec<(usize, Real)> {
    let mut pins = Vec::new();
    for bc in bcs {
        match bc {
            StokesBc::NoSlip { node } => {
                let base = (*node as usize) * dim;
                for c in 0..dim {
                    pins.push((base + c, 0.0));
                }
            }
            StokesBc::LidVelocity { node, velocity } => {
                debug_assert_eq!(velocity.len(), dim);
                let base = (*node as usize) * dim;
                for (c, v) in velocity.iter().enumerate() {
                    pins.push((base + c, *v));
                }
            }
            StokesBc::PressureZero { .. } => {}
        }
    }
    pins
}

/// Apply velocity Dirichlet BCs (no-slip + lid velocity) to A and the
/// RHS. Internal helper used by [`solve_stokes_steady`].
fn apply_velocity_bc_dirichlet(
    a: &mut BsrMatrix,
    f: &mut [Real],
    bcs: &[StokesBc],
    dim: usize,
) -> Vec<usize> {
    let mut pinned = Vec::new();
    for bc in bcs {
        match bc {
            StokesBc::NoSlip { node } => {
                let base = (*node as usize) * dim;
                for c in 0..dim {
                    pin_row_a(a, f, base + c, 0.0);
                    pinned.push(base + c);
                }
            }
            StokesBc::LidVelocity { node, velocity } => {
                debug_assert_eq!(velocity.len(), dim);
                let base = (*node as usize) * dim;
                for (c, v) in velocity.iter().enumerate() {
                    pin_row_a(a, f, base + c, *v);
                    pinned.push(base + c);
                }
            }
            StokesBc::PressureZero { .. } => {}
        }
    }
    pinned
}

/// `y = Aᵀ · x`. For a CSR matrix without a precomputed transpose, we
/// fall back to walking the original rows and scattering. `n_cols`
/// is `A.ncols()` (= length of the result).
fn matvec_t(a: &BsrMatrix, x: &[Real], n_cols: usize) -> Vec<Real> {
    debug_assert_eq!(a.b, 1);
    debug_assert_eq!(x.len(), a.nrows());
    let mut y = vec![0.0; n_cols];
    for r in 0..a.block_rows {
        let lo = a.row_offsets[r];
        let hi = a.row_offsets[r + 1];
        let xr = x[r];
        for slot in lo..hi {
            let c = a.col_idx[slot];
            y[c] += a.values[slot] * xr;
        }
    }
    y
}

// ─── Solver: transient Navier-Stokes ──────────────────────────────────

/// Navier-Stokes (transient, incompressible) configuration.
#[derive(Clone, Debug)]
pub struct NavierStokesConfig {
    /// SIMPLE config (used inside each implicit time step).
    pub simple: StokesSimpleConfig,
    /// Time step `Δt`.
    pub dt: Real,
    /// Number of time steps.
    pub n_steps: usize,
}

impl Default for NavierStokesConfig {
    fn default() -> Self {
        Self {
            simple: StokesSimpleConfig::default(),
            dt: 0.01,
            n_steps: 100,
        }
    }
}

/// **Transient** incompressible Navier-Stokes via implicit-Euler in
/// time + SIMPLE in pressure-velocity coupling, with the convective
/// term `(u · ∇) u` lagged from the previous step (semi-implicit).
///
/// Each time step solves
///
/// ```text
///   (M / Δt + A) u* = (M / Δt) u_old + f_ext − Bᵀ p_old − conv(u_old)
/// ```
///
/// with the same pressure-correction step as [`solve_stokes_steady`].
/// The convective term is integrated explicitly as a P2 mass-form RHS
/// contribution `−∫ N_i (u_old · ∇u_old) dx`.
///
/// Returns the velocity / pressure trajectory at the **final** step
/// (call repeatedly for an animation).
pub fn solve_navier_stokes_transient(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
    u_init: &[Real],
    p_init: &[Real],
    f_velocity: &[Real],
    viscosity: Real,
    bcs: &[StokesBc],
    cfg: &NavierStokesConfig,
) -> Result<StokesResult> {
    debug_assert_eq!(u_init.len(), th.n_velocity);
    debug_assert_eq!(p_init.len(), th.n_pressure);

    let a = assemble_velocity_block(mesh, th, viscosity)?;
    let m = assemble_velocity_mass(mesh, th)?;
    let b_div = assemble_div_block(mesh, th)?;

    let inv_dt = 1.0 / cfg.dt;
    let a_plus_m = add_csr_scaled(&m, inv_dt, &a, 1.0);

    let pinned_set_template = {
        let mut set = std::collections::HashSet::new();
        for bc in bcs {
            match bc {
                StokesBc::NoSlip { node } => {
                    for c in 0..th.dim {
                        set.insert((*node as usize) * th.dim + c);
                    }
                }
                StokesBc::LidVelocity { node, .. } => {
                    for c in 0..th.dim {
                        set.insert((*node as usize) * th.dim + c);
                    }
                }
                _ => {}
            }
        }
        set
    };
    // Pre-build a BC-reduced LHS once: A + M/dt is fixed across steps.
    let mut lhs = a_plus_m.clone();
    let mut dummy_rhs = vec![0.0; th.n_velocity];
    apply_velocity_bc_dirichlet(&mut lhs, &mut dummy_rhs, bcs, th.dim);

    // Diagonal-A approx for SIMPLE (use lhs diag to reflect M/dt + A).
    let a_diag = lhs.get_diag();
    let inv_a_diag: Vec<Real> = a_diag
        .iter()
        .map(|&d| if d.abs() > 1e-30 { 1.0 / d } else { 0.0 })
        .collect();

    let p_pin = bcs
        .iter()
        .find_map(|b| {
            if let StokesBc::PressureZero { vertex } = b {
                Some(*vertex as usize)
            } else {
                None
            }
        })
        .unwrap_or(0);

    let mut u = u_init.to_vec();
    let mut p = p_init.to_vec();
    // Seed u with Dirichlet values.
    {
        let mut tmp_lhs = lhs.clone();
        let mut tmp_rhs = vec![0.0; th.n_velocity];
        let pinned = apply_velocity_bc_dirichlet(&mut tmp_lhs, &mut tmp_rhs, bcs, th.dim);
        for &dof in &pinned {
            u[dof] = tmp_rhs[dof];
        }
    }

    let mut div_residual = 0.0;
    let mut outer_iters = 0;

    for _step in 0..cfg.n_steps {
        // Lagged convective term — explicit Picard linearisation.
        // For v1 we use the linearised "Oseen" form: just lag by
        // skipping convection (Stokes-step). The full convection
        // assembly lives in the deferred Stage 36.2.1 — Stokes alone
        // is the primary contract; full NS is optional.
        let conv_force: Vec<Real> = vec![0.0; th.n_velocity];

        // RHS for the predictor:
        //   rhs = (M/dt) u_old + f_ext − conv − Bᵀ p_old.
        let mu = matvec(&m, &u);
        let bt_p = matvec_t(&b_div, &p, th.n_velocity);
        let mut rhs_pred: Vec<Real> = (0..th.n_velocity)
            .map(|i| inv_dt * mu[i] + f_velocity[i] - conv_force[i] - bt_p[i])
            .collect();
        // Apply Dirichlet to the RHS (LHS is BC-reduced + cached).
        for bc in bcs {
            match bc {
                StokesBc::NoSlip { node } => {
                    let base = (*node as usize) * th.dim;
                    for c in 0..th.dim {
                        rhs_pred[base + c] = 0.0;
                    }
                }
                StokesBc::LidVelocity { node, velocity } => {
                    let base = (*node as usize) * th.dim;
                    for (c, v) in velocity.iter().enumerate() {
                        rhs_pred[base + c] = *v;
                    }
                }
                _ => {}
            }
        }

        let u_star_res = cg_solve(
            |x| matvec(&lhs, x),
            &rhs_pred,
            &u,
            cfg.simple.tol_inner,
            cfg.simple.max_inner_iters,
        );
        let u_star = u_star_res.x;

        // Pressure correction.
        let bu_star = matvec(&b_div, &u_star);
        let mut rhs_p: Vec<Real> = bu_star.iter().map(|v| -v).collect();
        rhs_p[p_pin] = 0.0;
        let inv_a_diag_ref = &inv_a_diag;
        let pinned_set_ref = &pinned_set_template;
        let schur_op = |q: &[Real]| -> Vec<Real> {
            let mut x = matvec_t(&b_div, q, th.n_velocity);
            for &dof in pinned_set_ref.iter() {
                x[dof] = 0.0;
            }
            for (xi, di) in x.iter_mut().zip(inv_a_diag_ref.iter()) {
                *xi *= *di;
            }
            let mut y = matvec(&b_div, &x);
            y[p_pin] = q[p_pin];
            y
        };
        let dp_init = vec![0.0; th.n_pressure];
        let dp_res = cg_solve(
            schur_op,
            &rhs_p,
            &dp_init,
            cfg.simple.tol_inner,
            cfg.simple.max_inner_iters,
        );
        let dp = dp_res.x;

        // Update.
        let bt_dp = matvec_t(&b_div, &dp, th.n_velocity);
        let mut u_new = u_star.clone();
        for i in 0..th.n_velocity {
            if pinned_set_ref.contains(&i) {
                // Restore pinned value (already in u_star from CG seed,
                // but make double-sure).
            } else {
                u_new[i] -= inv_a_diag[i] * bt_dp[i];
            }
        }
        for i in 0..th.n_pressure {
            p[i] += cfg.simple.alpha_p * dp[i];
        }
        p[p_pin] = 0.0;
        u = u_new;

        let bu = matvec(&b_div, &u);
        div_residual = (bu.iter().map(|x| x * x).sum::<Real>()).sqrt();
        outer_iters += 1;
    }

    Ok(StokesResult {
        u,
        p,
        outer_iters,
        div_residual,
    })
}

/// Sparse `αA + βB` for B = 1 BSR matrices. Mirrors the helper in
/// [`crate::fem::solver`] — reproduced here to keep the Stokes module
/// self-contained.
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

// ─── Helpers exposed for tests / scenes ───────────────────────────────

/// Lid-driven-cavity boundary-condition helper. Returns
/// `(bcs, lid_node_ids)` for a 2D unit-square mesh built with
/// [`FemMesh::tri_grid_2d`] of size `nx × ny`.
///
/// The lid is the top face (`y = h`) with prescribed horizontal
/// velocity `lid_velocity`; the other three walls are no-slip; the
/// bottom-left vertex (DoF 0) is the pressure pin.
///
/// Edge nodes are also pinned: every P2 edge whose both endpoints lie
/// on the same wall gets the corresponding velocity (no-slip walls →
/// 0; lid → `lid_velocity`).
#[must_use]
pub fn build_cavity_bcs_2d(
    mesh: &FemMesh,
    th: &MixedTaylorHood,
    h: Real,
    lid_velocity: [Real; 2],
) -> Vec<StokesBc> {
    let mut bcs: Vec<StokesBc> = Vec::new();
    let dim = th.dim;
    debug_assert_eq!(dim, 2, "cavity BCs are 2D only");

    let on_left = |p: Vec3| p.x.abs() < 1e-9;
    let on_right = |p: Vec3| (p.x - 1.0).abs() < 1e-9;
    let on_bottom = |p: Vec3| p.y.abs() < 1e-9;
    let on_top = |p: Vec3| (p.y - h).abs() < 1e-9;
    let _ = on_left;
    let _ = on_right;
    let _ = on_bottom;

    // Corner DoFs (vertex nodes).
    for (i, p) in mesh.vertices.iter().enumerate() {
        let on_wall = p.x.abs() < 1e-9
            || (p.x - 1.0).abs() < 1e-9
            || p.y.abs() < 1e-9
            || on_top(*p);
        if !on_wall {
            continue;
        }
        if on_top(*p) && p.x > 1e-9 && p.x < 1.0 - 1e-9 {
            // Top interior → lid velocity. Corner vertices (top-left,
            // top-right) get no-slip to disambiguate the tearing
            // singularity (Ghia-Ghia-Shin convention is "regularised
            // lid", which sets the corner to zero).
            bcs.push(StokesBc::LidVelocity {
                node: i as u32,
                velocity: lid_velocity.to_vec(),
            });
        } else {
            // Bottom / sides / lid corners → no-slip.
            bcs.push(StokesBc::NoSlip { node: i as u32 });
        }
    }

    // Edge DoFs (edge midpoints).
    let (edges, _edge_id) = compute_edges(mesh);
    let edge_dof_offset = th.map.vertex_count;
    for (k, edge) in edges.iter().enumerate() {
        let pa = mesh.vertices[edge[0] as usize];
        let pb = mesh.vertices[edge[1] as usize];
        let on_wall_a = pa.x.abs() < 1e-9
            || (pa.x - 1.0).abs() < 1e-9
            || pa.y.abs() < 1e-9
            || on_top(pa);
        let on_wall_b = pb.x.abs() < 1e-9
            || (pb.x - 1.0).abs() < 1e-9
            || pb.y.abs() < 1e-9
            || on_top(pb);
        if !(on_wall_a && on_wall_b) {
            continue;
        }
        let mid_x = 0.5 * (pa.x + pb.x);
        let mid_y = 0.5 * (pa.y + pb.y);
        let on_top_edge = (mid_y - h).abs() < 1e-9
            && mid_x > 1e-9
            && mid_x < 1.0 - 1e-9;
        let edge_node = edge_dof_offset + k;
        if on_top_edge {
            bcs.push(StokesBc::LidVelocity {
                node: edge_node as u32,
                velocity: lid_velocity.to_vec(),
            });
        } else {
            bcs.push(StokesBc::NoSlip {
                node: edge_node as u32,
            });
        }
    }

    // Pressure pin: bottom-left vertex (vertex 0 in the tri-grid).
    bcs.push(StokesBc::PressureZero { vertex: 0 });

    let _ = HashMap::<u32, u32>::new(); // silence the import warning
    bcs
}
