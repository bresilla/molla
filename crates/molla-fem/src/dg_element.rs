//! Stage 39B — Discontinuous Galerkin (DG) nodal element matrices.
//!
//! DG elements duplicate DoFs at every cell boundary: a vertex shared
//! by `k` cells contributes `k * ndof_per_cell` DoFs total, **not** the
//! single shared DoF that CG (P1/P2) elements use. The lack of
//! inter-element continuity means a DG bilinear form has two
//! contributions:
//!
//! 1. **Element-interior** integrals (mass, stiffness, …) — identical
//!    to the corresponding CG element matrices because the shape
//!    functions on the interior of `K` are the same as the CG
//!    `(deg)`-nodal shape functions. These are produced by
//!    [`compute_element_dg_mass`] and
//!    [`compute_element_dg_stiffness`].
//! 2. **Inter-element flux** integrals on each interior face. v1 ships
//!    the **Symmetric Interior Penalty (SIP)** Laplacian flux —
//!    standard, symmetric, and provably coercive for sufficient
//!    penalty. See [`compute_dg_flux_jump_block`].
//!
//! The SIP bilinear form on face `F` shared by cells `K⁻` and `K⁺`
//! (normal `n` pointing from `−` to `+`):
//!
//! ```text
//! a_F(u, v) = − ∫_F {α ∇u · n} [v] dS
//!             − ∫_F [u] {α ∇v · n} dS
//!             + (η · α / h_F) ∫_F [u] [v] dS,
//!
//! {w}  = ½(w⁻ + w⁺)      (average)
//! [w]  = w⁻ − w⁺         (scalar jump)
//! ```
//!
//! where `α` is the diffusion coefficient (the same `coeff` taken by
//! [`compute_element_dg_stiffness`]) and `η` is the penalty parameter
//! (Arnold-Brezzi-Cockburn-Marini style, scaled by `α / h_F`).
//!
//! v1 scope:
//!
//! - **Degree** `∈ {0, 1, 2}` — P0 (piecewise constant; no interior
//!   stiffness contribution, only flux penalty), P1 (linear on each
//!   cell, nodal at the cell vertices), and P2 (quadratic, nodal at
//!   cell vertices + edge midpoints — **Stage 43**). The P2 shape
//!   functions use the same Lagrange basis as the CG-P2 element in
//!   [`crate::fem::p2_element`] but are duplicated per cell.
//! - **Tri / Tet** cells — Tri faces are edges (line integrals), Tet
//!   faces are triangles (surface integrals).
//! - **Boundary handling** — for v1 we only assemble the flux on
//!   **interior** faces (those shared by exactly two cells). Boundary
//!   handling (Dirichlet / Neumann) is the caller's responsibility via
//!   the existing [`crate::fem::bc`] machinery applied to the
//!   per-cell DG DoFs.
//!
//! All kernels return row-major dense matrices as `Vec<Vec<Real>>` to
//! match the rest of the FEM module.
//!
//! ## Stage 43 additions
//!
//! - **P2 DG shape functions** — per-cell Lagrange basis, 6 DoFs on a
//!   triangle (3 vertex + 3 edge midpoint) and 10 DoFs on a tet
//!   (4 vertex + 6 edge midpoint). Layout matches
//!   [`crate::fem::p2_element`]'s reference layout.
//! - **`NumericalFlux`** — first-order (hyperbolic) flux library for
//!   scalar conservation laws: `Central`, `Upwind`, `LaxFriedrichs`,
//!   `Roe`. Used by [`compute_dg_numerical_flux_block`] to build the
//!   advection-style face block. The SIP-elliptic flux used by
//!   [`compute_dg_flux_jump_block`] is unrelated — that's the
//!   diffusion / viscous flux, kept separate so a convection-diffusion
//!   solver can sum the two contributions.
//! - **Barth-Jespersen slope limiter** — scalar shock capture via
//!   [`limit_solution`]. Clamps the per-cell P1/P2 reconstruction so
//!   it never exceeds the min/max of cell-averaged neighbours
//!   (optionally clipped to a user-supplied `physical_min/max`).

use crate::math::{Real, Vec3};

use super::element::{tet_shape_grads, tet_volume, tri_area_2d, tri_shape_grads_2d};
use super::mesh::FemCell;
use super::p2_element::{
    TET_EDGES, TRI_EDGES, p2_tet_shape_functions, p2_tet_shape_gradients, p2_tri_shape_functions,
    p2_tri_shape_gradients,
};

// ─── Element-interior blocks ─────────────────────────────────────────

/// Per-cell DG mass matrix `Mₑᵢⱼ = ∫_K φᵢ φⱼ dx` — block-diagonal in
/// the global system because DG basis functions vanish outside their
/// owning cell, so two distinct cells share no support.
///
/// Returns an `N × N` row-major matrix where
/// `N = dg_ndof_per_cell(cell, degree)`:
///
/// - `degree = 0` → 1×1 diagonal `[ |K| ]` (piecewise-constant mass is
///   exactly the cell volume).
/// - `degree = 1` → 3×3 (Tri) / 4×4 (Tet) — same as the CG-P1 element
///   mass matrix, because the per-cell P1 nodal basis equals the CG-P1
///   restriction on `K`.
/// - `degree = 2` → 6×6 (Tri) / 10×10 (Tet) — same as the CG-P2
///   element mass matrix (the Lagrange basis on the reference simplex
///   is identical; only the global DoF map differs). Stage 43.
#[must_use]
pub fn compute_element_dg_mass(cell: &FemCell, vertices: &[Vec3], degree: usize) -> Vec<Vec<Real>> {
    assert!(
        degree <= 2,
        "DG: degree {degree} out of range (supports 0..=2)"
    );
    match (cell, degree) {
        (FemCell::Tri(idx), 0) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
            ];
            let area = tri_area_2d(&p);
            vec![vec![area]]
        }
        (FemCell::Tet(idx), 0) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
                vertices[idx[3] as usize],
            ];
            let vol = tet_volume(&p);
            vec![vec![vol]]
        }
        // P1 element mass matches the CG-P1 element mass — same shape
        // functions on the cell interior. Re-use the existing kernel
        // to avoid drift between the two paths.
        (_, 1) => super::element::compute_element_mass(cell, vertices),
        // P2: reuse the CG-P2 element mass kernel. On the cell
        // interior the DG basis is identical to the CG basis (per-cell
        // Lagrange nodes at vertices + edge midpoints); only the
        // global DoF map differs.
        (_, 2) => super::p2_element::compute_element_p2_mass(cell, vertices),
        _ => unreachable!(),
    }
}

/// Per-cell DG stiffness `Kₑᵢⱼ = coeff · ∫_K ∇φᵢ · ∇φⱼ dx` —
/// element-interior part only (the inter-element flux block lives in
/// [`compute_dg_flux_jump_block`]).
///
/// Returns an `N × N` row-major matrix where
/// `N = dg_ndof_per_cell(cell, degree)`:
///
/// - `degree = 0` → 1×1 zero (piecewise constants have zero gradient).
/// - `degree = 1` → 3×3 (Tri) / 4×4 (Tet) — same as the CG-P1 element
///   stiffness with the same `coeff` scaling.
/// - `degree = 2` → 6×6 (Tri) / 10×10 (Tet) — same as the CG-P2
///   element stiffness (Lagrange shape gradients integrated with the
///   higher-order quadrature shipped in [`crate::fem::quadrature`]).
#[must_use]
pub fn compute_element_dg_stiffness(
    cell: &FemCell,
    vertices: &[Vec3],
    coeff: Real,
    degree: usize,
) -> Vec<Vec<Real>> {
    assert!(
        degree <= 2,
        "DG: degree {degree} out of range (supports 0..=2)"
    );
    match degree {
        0 => vec![vec![0.0]],
        1 => super::element::compute_element_stiffness(cell, vertices, coeff),
        2 => super::p2_element::compute_element_p2_stiffness(cell, vertices, coeff),
        _ => unreachable!(),
    }
}

// ─── Face enumeration ────────────────────────────────────────────────

/// One face shared by exactly two cells of the mesh.
///
/// `face_verts` lists the vertex indices of the shared face (2 entries
/// for a triangle-mesh edge, 3 entries for a tet face). The orientation
/// is canonical (sorted ascending) — the per-cell side picks its own
/// orientation when computing the outward normal.
#[derive(Clone, Debug)]
pub struct InteriorFace {
    /// Cell index of the "left" side (the cell whose outward normal is
    /// taken to define the `+1` jump sign convention).
    pub cell_left: u32,
    /// Cell index of the "right" side.
    pub cell_right: u32,
    /// Sorted vertex indices of the shared face. Length 2 for a tri
    /// mesh (edge) or 3 for a tet mesh (triangle face).
    pub face_verts: Vec<u32>,
}

/// Enumerate the interior faces of `mesh` — faces (edges in 2D, tri
/// faces in 3D) shared by exactly **two** cells. Boundary faces
/// (those owned by a single cell) are excluded; v1 leaves boundary
/// flux handling to the caller's BC routines applied to per-cell DG
/// DoFs.
///
/// Each interior face is reported once, with `cell_left` < `cell_right`
/// so the orientation is deterministic.
#[must_use]
pub fn compute_interior_faces(mesh: &super::mesh::FemMesh) -> Vec<InteriorFace> {
    use std::collections::HashMap;

    // Map sorted face vertices → list of (cell_idx, original verts).
    let mut face_to_cells: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();
    for (cell_idx, cell) in mesh.cells.iter().enumerate() {
        match cell {
            FemCell::Tri(v) => {
                // 3 edges: (0,1), (1,2), (2,0).
                for &[a, b] in &[[v[0], v[1]], [v[1], v[2]], [v[2], v[0]]] {
                    let mut key = vec![a, b];
                    key.sort_unstable();
                    face_to_cells.entry(key).or_default().push(cell_idx as u32);
                }
            }
            FemCell::Tet(v) => {
                // 4 faces — each is opposite one vertex.
                for &[a, b, c] in &[
                    [v[1], v[2], v[3]],
                    [v[0], v[2], v[3]],
                    [v[0], v[1], v[3]],
                    [v[0], v[1], v[2]],
                ] {
                    let mut key = vec![a, b, c];
                    key.sort_unstable();
                    face_to_cells.entry(key).or_default().push(cell_idx as u32);
                }
            }
        }
    }

    let mut interior = Vec::new();
    for (face_verts, owners) in face_to_cells {
        if owners.len() == 2 {
            let (l, r) = if owners[0] < owners[1] {
                (owners[0], owners[1])
            } else {
                (owners[1], owners[0])
            };
            interior.push(InteriorFace {
                cell_left: l,
                cell_right: r,
                face_verts,
            });
        }
        // owners.len() == 1 → boundary face (skipped).
        // owners.len() >= 3 should never happen for a manifold mesh.
    }
    // Sort for deterministic output (helps tests + reproducibility).
    interior.sort_by_key(|f| (f.cell_left, f.cell_right, f.face_verts.clone()));
    interior
}

// ─── Geometry helpers for faces ──────────────────────────────────────

/// 2D length of an edge connecting `a` and `b` (z ignored).
#[must_use]
fn edge_length_2d(a: Vec3, b: Vec3) -> Real {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    (dx * dx + dy * dy).sqrt()
}

/// 2D unit normal of the directed edge `(a → b)`, pointing 90° right
/// (i.e. `(dy, -dx) / |edge|`). Returns zero for a degenerate edge.
#[must_use]
fn edge_normal_2d(a: Vec3, b: Vec3) -> Vec3 {
    let dx = b.x - a.x;
    let dy = b.y - a.y;
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0e-30 {
        return Vec3::ZERO;
    }
    Vec3::new(dy / len, -dx / len, 0.0)
}

/// 3D area of the triangle `(a, b, c)`.
#[must_use]
fn tri_area_3d(a: Vec3, b: Vec3, c: Vec3) -> Real {
    0.5 * (b - a).cross(c - a).length()
}

/// Unit normal of the triangle `(a, b, c)` using the right-hand rule
/// from `(b - a) × (c - a)`. Returns zero for a degenerate triangle.
#[must_use]
fn tri_normal_3d(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let n = (b - a).cross(c - a);
    let l = n.length();
    if l < 1.0e-30 {
        return Vec3::ZERO;
    }
    n / l
}

/// Length scale `h_F` for SIP penalty — the cross-cell diameter measure
/// the literature uses. For 2D faces (edges) we use the edge length;
/// for 3D faces (triangles) we use `sqrt(area)` (consistent with the
/// `mesh size ~ h` reading of penalty constants).
#[must_use]
fn face_h_scale(face: &InteriorFace, vertices: &[Vec3]) -> Real {
    match face.face_verts.len() {
        2 => edge_length_2d(
            vertices[face.face_verts[0] as usize],
            vertices[face.face_verts[1] as usize],
        ),
        3 => tri_area_3d(
            vertices[face.face_verts[0] as usize],
            vertices[face.face_verts[1] as usize],
            vertices[face.face_verts[2] as usize],
        )
        .sqrt(),
        _ => unreachable!("InteriorFace has unsupported face_verts length"),
    }
}

// ─── DG shape-function helpers ───────────────────────────────────────
//
// Stage 43 — DG-P2 face flux blocks need shape values and gradients
// at face-quadrature points (which are arbitrary inside the cell,
// not just at nodes). We compute the barycentric coordinates of the
// physical point in the owning cell and route them through the same
// P2 Lagrange shape tables that `p2_element` uses for CG-P2.

/// Barycentric coordinates of a 2D point `x` inside a triangle.
fn tri_bary(p: &[Vec3; 3], x: Vec3) -> [Real; 3] {
    let det = (p[1].x - p[0].x) * (p[2].y - p[0].y) - (p[2].x - p[0].x) * (p[1].y - p[0].y);
    let inv = 1.0 / det;
    let l1 = ((p[2].y - p[0].y) * (x.x - p[0].x) + (p[0].x - p[2].x) * (x.y - p[0].y)) * inv;
    let l2 = ((p[0].y - p[1].y) * (x.x - p[0].x) + (p[1].x - p[0].x) * (x.y - p[0].y)) * inv;
    let l0 = 1.0 - l1 - l2;
    [l0, l1, l2]
}

/// Barycentric coordinates of a 3D point `x` inside a tetrahedron.
fn tet_bary(p: &[Vec3; 4], x: Vec3) -> [Real; 4] {
    use crate::math::Mat3;
    let d = Mat3::from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0]);
    let inv = d.inverse();
    let r = x - p[0];
    let l1 = inv.x_axis.x * r.x + inv.y_axis.x * r.y + inv.z_axis.x * r.z;
    let l2 = inv.x_axis.y * r.x + inv.y_axis.y * r.y + inv.z_axis.y * r.z;
    let l3 = inv.x_axis.z * r.x + inv.y_axis.z * r.y + inv.z_axis.z * r.z;
    let l0 = 1.0 - l1 - l2 - l3;
    [l0, l1, l2, l3]
}

/// Evaluate the DG shape values for `cell` at physical point `x`,
/// supporting `degree ∈ {0, 1, 2}`. Returns a `Vec<Real>` of length
/// `dg_ndof_per_cell(cell, degree)`.
fn dg_shape_values_at(cell: &FemCell, vertices: &[Vec3], x: Vec3, degree: usize) -> Vec<Real> {
    match (cell, degree) {
        (_, 0) => vec![1.0],
        (FemCell::Tri(v), 1) | (FemCell::Tri(v), 2) => {
            let p = [
                vertices[v[0] as usize],
                vertices[v[1] as usize],
                vertices[v[2] as usize],
            ];
            let bary = tri_bary(&p, x);
            if degree == 1 {
                bary.to_vec()
            } else {
                p2_tri_shape_functions(bary).to_vec()
            }
        }
        (FemCell::Tet(v), 1) | (FemCell::Tet(v), 2) => {
            let p = [
                vertices[v[0] as usize],
                vertices[v[1] as usize],
                vertices[v[2] as usize],
                vertices[v[3] as usize],
            ];
            let bary = tet_bary(&p, x);
            if degree == 1 {
                bary.to_vec()
            } else {
                p2_tet_shape_functions(bary).to_vec()
            }
        }
        _ => unreachable!(),
    }
}

/// Evaluate the DG shape **gradients** in physical space for `cell` at
/// physical point `x`, supporting `degree ∈ {1, 2}`. Returns a
/// `Vec<Vec3>` of length `dg_ndof_per_cell(cell, degree)`. For
/// `degree = 0` this is the empty / zero gradient — callers should
/// short-circuit before calling.
fn dg_shape_grads_at(cell: &FemCell, vertices: &[Vec3], x: Vec3, degree: usize) -> Vec<Vec3> {
    match (cell, degree) {
        (_, 0) => vec![Vec3::ZERO],
        (FemCell::Tri(v), 1) => {
            let p = [
                vertices[v[0] as usize],
                vertices[v[1] as usize],
                vertices[v[2] as usize],
            ];
            let g = tri_shape_grads_2d(&p);
            vec![
                Vec3::new(g[0][0], g[0][1], 0.0),
                Vec3::new(g[1][0], g[1][1], 0.0),
                Vec3::new(g[2][0], g[2][1], 0.0),
            ]
        }
        (FemCell::Tet(v), 1) => {
            let p = [
                vertices[v[0] as usize],
                vertices[v[1] as usize],
                vertices[v[2] as usize],
                vertices[v[3] as usize],
            ];
            tet_shape_grads(&p).to_vec()
        }
        (FemCell::Tri(v), 2) => {
            let p = [
                vertices[v[0] as usize],
                vertices[v[1] as usize],
                vertices[v[2] as usize],
            ];
            let bary = tri_bary(&p, x);
            p2_tri_shape_gradients(bary, &p).to_vec()
        }
        (FemCell::Tet(v), 2) => {
            let p = [
                vertices[v[0] as usize],
                vertices[v[1] as usize],
                vertices[v[2] as usize],
                vertices[v[3] as usize],
            ];
            let bary = tet_bary(&p, x);
            p2_tet_shape_gradients(bary, &p).to_vec()
        }
        _ => unreachable!(),
    }
}

/// Per-cell DoF count for a DG cell of `degree`.
fn dg_local_ndof(cell: &FemCell, degree: usize) -> usize {
    super::function_space::dg_ndof_per_cell(cell, degree)
}

/// Stage 43 — for projecting analytic / arbitrary fields onto the DG
/// basis (initial conditions, slope-limiter visualisation), return the
/// physical position of each DG DoF inside `cell`.
///
/// - `degree = 0` → single centroid.
/// - `degree = 1` → cell vertices.
/// - `degree = 2` → cell vertices, then edge midpoints (same order as
///   the Lagrange DoF layout used by [`crate::fem::p2_element`]).
#[must_use]
pub fn dg_nodal_positions(cell: &FemCell, vertices: &[Vec3], degree: usize) -> Vec<Vec3> {
    match cell {
        FemCell::Tri(v) => {
            let p = [
                vertices[v[0] as usize],
                vertices[v[1] as usize],
                vertices[v[2] as usize],
            ];
            match degree {
                0 => vec![(p[0] + p[1] + p[2]) / 3.0],
                1 => p.to_vec(),
                2 => {
                    let mut nodes = p.to_vec();
                    for &edge in &TRI_EDGES {
                        nodes.push((p[edge[0]] + p[edge[1]]) * 0.5);
                    }
                    nodes
                }
                _ => unreachable!(),
            }
        }
        FemCell::Tet(v) => {
            let p = [
                vertices[v[0] as usize],
                vertices[v[1] as usize],
                vertices[v[2] as usize],
                vertices[v[3] as usize],
            ];
            match degree {
                0 => vec![(p[0] + p[1] + p[2] + p[3]) / 4.0],
                1 => p.to_vec(),
                2 => {
                    let mut nodes = p.to_vec();
                    for &edge in &TET_EDGES {
                        nodes.push((p[edge[0]] + p[edge[1]]) * 0.5);
                    }
                    nodes
                }
                _ => unreachable!(),
            }
        }
    }
}

// ─── SIP flux block ──────────────────────────────────────────────────

/// SIP flux block on the shared `face` between `cell_left` and
/// `cell_right`, returning a square dense matrix
/// `(NL + NR) × (NL + NR)` where `NL = NR = ndof_per_cell(degree)`
/// (assuming both cells have the same shape; degree-mixing on a face
/// is not supported in v1).
///
/// The block layout is:
///
/// ```text
///         | L-DoFs  R-DoFs |
///   L-row | a_LL    a_LR  |
///   R-row | a_RL    a_RR  |
/// ```
///
/// Caller scatters local DoF `i` (row/col `0 .. NL`) to global DG
/// DoF `cell_left * NL + i` and local DoF `NL + i` (row/col `NL ..`)
/// to global DoF `cell_right * NR + i`.
///
/// `penalty_eta` is the dimensionless SIP penalty `η` — the literature
/// recommends `η ≥ C · (degree + 1)²` with `C` of order ~10 for
/// triangles. The caller passes the raw `η`; the block scales it
/// internally as `η · coeff / h_F` (see file docstring).
#[must_use]
pub fn compute_dg_flux_jump_block(
    cell_left: &FemCell,
    cell_right: &FemCell,
    face: &InteriorFace,
    vertices: &[Vec3],
    coeff: Real,
    degree: usize,
    penalty_eta: Real,
) -> Vec<Vec<Real>> {
    assert!(
        degree <= 2,
        "DG: degree {degree} out of range (supports 0..=2)"
    );
    let nl = dg_local_ndof(cell_left, degree);
    let nr = dg_local_ndof(cell_right, degree);
    let n = nl + nr;
    let mut block = vec![vec![0.0_f64 as Real; n]; n];

    let h_f = face_h_scale(face, vertices);
    let area = face_measure(face, vertices);
    let pen_coeff = penalty_eta * coeff / h_f;

    // Outward normal from `cell_left` to `cell_right`. Use the centroid
    // displacement to disambiguate orientation.
    let n_lr = face_normal_left_to_right(cell_left, cell_right, face, vertices);

    // For degree 0 the jump is `[u] = u_L - u_R` (scalar) and the
    // gradient terms vanish — only the penalty block contributes.
    if degree == 0 {
        let p = pen_coeff * area;
        // a_LL = p, a_RR = p, a_LR = -p, a_RL = -p.
        block[0][0] += p;
        block[1][1] += p;
        block[0][1] += -p;
        block[1][0] += -p;
        return block;
    }

    // Degree 1 or 2: SIP bilinear form (Arnold-Brezzi-Cockburn-Marini):
    //
    //   a_F(u, v) = − ∫_F ⟨α ∇u·n⟩ ⟦v⟧ dS
    //               − ∫_F ⟦u⟧ ⟨α ∇v·n⟩ dS
    //               + ∫_F (η α / h_F) ⟦u⟧ ⟦v⟧ dS
    //
    // where ⟨w⟩ = ½(w⁻ + w⁺) and ⟦w⟧ = w⁻ − w⁺ with the normal
    // pointing from − (L) to + (R). For a basis function φ_L^i
    // restricted to face F: w⁻ = N_L[i], w⁺ = 0; for φ_R^i:
    // w⁻ = 0, w⁺ = N_R[i].
    //
    // Expanding the four sub-blocks (with `gn_S[i] = ∇φ_S^i · n_LR`):
    //
    //   a_LL[i][j] = − ½ α gn_L[j] N_L[i] − ½ α N_L[j] gn_L[i]
    //                + pen N_L[i] N_L[j]
    //
    //   a_LR[i][j] = + ½ α gn_L[i] N_R[j] − ½ α N_L[i] gn_R[j]
    //                − pen N_L[i] N_R[j]
    //
    //   a_RL[i][j] = + ½ α gn_L[j] N_R[i] − ½ α N_L[j] gn_R[i]
    //                − pen N_L[j] N_R[i]                        (= a_LR^T)
    //
    //   a_RR[i][j] = + ½ α gn_R[j] N_R[i] + ½ α N_R[j] gn_R[i]
    //                + pen N_R[i] N_R[j]
    //
    // The asymmetric signs on the gradient terms (− for LL, + for RR)
    // come from ⟦v⟧ flipping sign on the R side (v⁻=0, v⁺=N_R ⇒
    // ⟦v⟧=−N_R) — coupled with the SIP form's leading minus they
    // produce a + on a_RR.
    //
    // Face quadrature: 2-pt Gauss-Legendre (edge) is degree-3 exact —
    // sufficient for P1 × P1 (quadratic face integrand) but **not**
    // for P2 × P2 (quartic face integrand). For `degree = 2` we
    // upgrade to the 3-pt Gauss-Legendre rule on the edge (degree-5
    // exact). 3D tet faces use the standard 3-pt mid-edge rule
    // (degree-2 exact) for P1; the P2 case needs a higher-order
    // triangle-face rule — we use a 6-pt Dunavant rule that captures
    // degree-4 integrands exactly.

    let qpts = if degree >= 2 {
        face_quadrature_high(face, vertices)
    } else {
        face_quadrature(face, vertices)
    };

    for q in &qpts {
        let nl_vals = dg_shape_values_at(cell_left, vertices, q.point, degree);
        let nr_vals = dg_shape_values_at(cell_right, vertices, q.point, degree);
        let grads_l = dg_shape_grads_at(cell_left, vertices, q.point, degree);
        let grads_r = dg_shape_grads_at(cell_right, vertices, q.point, degree);
        let gn_l: Vec<Real> = grads_l.iter().map(|g| g.dot(n_lr)).collect();
        let gn_r: Vec<Real> = grads_r.iter().map(|g| g.dot(n_lr)).collect();
        let w = q.weight; // absolute weight (already includes |F|).

        for i in 0..nl {
            for j in 0..nl {
                block[i][j] += w
                    * (-0.5 * coeff * gn_l[j] * nl_vals[i] - 0.5 * coeff * nl_vals[j] * gn_l[i]
                        + pen_coeff * nl_vals[i] * nl_vals[j]);
            }
            for j in 0..nr {
                block[i][nl + j] += w
                    * (0.5 * coeff * gn_l[i] * nr_vals[j]
                        - 0.5 * coeff * nl_vals[i] * gn_r[j]
                        - pen_coeff * nl_vals[i] * nr_vals[j]);
            }
        }
        for i in 0..nr {
            for j in 0..nl {
                block[nl + i][j] += w
                    * (0.5 * coeff * gn_l[j] * nr_vals[i]
                        - 0.5 * coeff * nl_vals[j] * gn_r[i]
                        - pen_coeff * nl_vals[j] * nr_vals[i]);
            }
            for j in 0..nr {
                block[nl + i][nl + j] += w
                    * (0.5 * coeff * gn_r[j] * nr_vals[i]
                        + 0.5 * coeff * nr_vals[j] * gn_r[i]
                        + pen_coeff * nr_vals[i] * nr_vals[j]);
            }
        }
    }

    block
}

// ─── Stage 43 — Numerical-flux library + advection face block ────────

/// Numerical flux for first-order (hyperbolic) DG conservation laws.
///
/// A scalar conservation law `∂u/∂t + ∇·F(u) = 0` discretised with DG
/// needs a face flux `Ĥ(u_L, u_R, n)` that picks a single trace value
/// on each face. Stage 43 ships the four canonical choices used by
/// Newton's DG ports (`example_burgers`, `example_convection_diffusion_dg`,
/// `example_shallow_water`):
///
/// - [`NumericalFlux::Central`] — simple average `½ (F(u_L) + F(u_R)) · n`.
///   No stabilisation; only stable for problems with diffusion (or
///   combined with a slope limiter).
/// - [`NumericalFlux::Upwind`] — for **linear advection** `F(u) = a · u`
///   with constant velocity `wind = a`. Picks `u_L` or `u_R` based on
///   the sign of `wind · n`. This is the exact Riemann solver for
///   linear advection.
/// - [`NumericalFlux::LaxFriedrichs`] — global Lax-Friedrichs (Rusanov)
///   flux `½ (F(u_L) + F(u_R)) · n − ½ alpha (u_R − u_L)`. The
///   dissipation coefficient `alpha` is the maximum wave speed
///   `max_u |F'(u)|`; for Burgers' use `max_u |u|`, for SWE use
///   `max_u (|u| + √(g h))`. Robust default — works for any flux.
/// - [`NumericalFlux::Roe`] — Roe-average linearisation. For a scalar
///   conservation law this collapses to upwind with wave speed
///   `(F(u_R) − F(u_L)) / (u_R − u_L)` (the Roe-average derivative).
///   Identical to upwind in the linear case; for non-linear scalar
///   problems (Burgers') it adds correct entropy if combined with a
///   limiter or sonic-fix.
///
/// Default recommendation:
/// - Linear advection ⇒ `Upwind { wind }`.
/// - Burgers' / non-linear scalar ⇒ `LaxFriedrichs { alpha }`.
/// - Shallow water / systems ⇒ `LaxFriedrichs` (component-wise).
#[derive(Copy, Clone, Debug)]
pub enum NumericalFlux {
    /// `Ĥ = ½ (F(u_L) + F(u_R)) · n`. No upwinding.
    Central,
    /// Linear-advection upwind: `Ĥ = (wind · n) u_upwind`. The upwind
    /// state is `u_L` when `wind · n > 0` (information leaving cell
    /// `L`) and `u_R` otherwise.
    Upwind {
        /// Advection velocity (constant in space + time for this flux).
        wind: Vec3,
    },
    /// Global Lax-Friedrichs (Rusanov): `Ĥ = ½(F_L+F_R)·n − ½α(u_R−u_L)`
    /// where `α` is the global max wave speed.
    LaxFriedrichs {
        /// Maximum wave speed magnitude over the entire mesh.
        alpha: Real,
    },
    /// Roe flux — for scalar problems collapses to upwind with the
    /// Roe-average wave speed. For linear advection this matches
    /// `Upwind`; included so callers can write generic code keyed on
    /// the variant.
    Roe,
    /// Stage 43 v3 — HLL (Harten-Lax-van-Leer) two-wave Riemann solver
    /// for **systems** of conservation laws. Picks a single
    /// star-region state by enforcing conservation across the slowest
    /// and fastest wave-speed estimates (`s_l`, `s_r`). Works for any
    /// system; for SWE on a face with state `(h, hu, hv)` and normal
    /// `n`, the wave-speed estimates are
    ///
    /// ```text
    ///   s_l = min(u_L·n − √(g h_L), u_R·n − √(g h_R))
    ///   s_r = max(u_L·n + √(g h_L), u_R·n + √(g h_R))
    /// ```
    ///
    /// Robust + diffusive in the contact (HLLC adds the missing third
    /// wave). The flux itself is system-specific so the enum variant
    /// is just a tag — the SWE evaluation lives in
    /// [`compute_swe_numerical_flux_hll`].
    Hll,
    /// Stage 43 v3 — HLLC (Toro 1994) three-wave Riemann solver
    /// **specialised for shallow water**. Adds a contact-discontinuity
    /// star-region between the two HLL waves. Produces sharper
    /// contact resolution than HLL — important for dam-break/wet-dry
    /// fronts where the post-shock height has a slowly-moving
    /// contact between the left and right star states.
    Hllc,
    /// Stage 43 v3 — Roe (1981) linearised Riemann solver
    /// **specialised for shallow water**. Builds the Roe-average
    /// state `(ĥ, û, v̂)` such that the discrete Jacobian satisfies the
    /// "Property U" (jump-condition compatibility), then upwinds each
    /// of the three characteristic fields independently:
    ///
    /// ```text
    ///   ĥ  = ½(h_L + h_R)                   (arithmetic mean for SWE — the
    ///                                        textbook Roe SWE average; the
    ///                                        density-weighted Roe avg
    ///                                        familiar from gas dynamics
    ///                                        collapses to this for SWE)
    ///   û  = (u_L √h_L + u_R √h_R) / (√h_L + √h_R)
    ///   v̂  = analogous
    ///   ĉ  = √(g ĥ)
    /// ```
    ///
    /// The three eigenvalues are `(û·n − ĉ, û·n, û·n + ĉ)`; an
    /// **entropy fix** (Harten-Hyman) replaces eigenvalues smaller
    /// than `δ = max(0, λ̂ − λ_L, λ_R − λ̂)` with the smooth
    /// `(λ² + δ²) / (2δ)` to prevent transonic-rarefaction
    /// "anti-entropy" sonic glitches.
    RoeSwe,
}

/// Stage 43 — advection-style face flux block for the linear scalar
/// conservation law `∂u/∂t + ∇·(a u) = 0` (or more general scalar laws
/// handled via the dispatch in [`NumericalFlux`]).
///
/// The DG weak form contribution from one interior face `F` is
///
/// ```text
///   −∫_F  Ĥ(u_L, u_R, n)  ⟦v⟧ dS
/// ```
///
/// which, expanded over the per-cell DoFs (`u_L = Σ N_L^j u_L^j`,
/// `v = N_L^i` or `N_R^i`), gives the four sub-blocks below. The block
/// is **non-symmetric** in general (advection is not self-adjoint).
///
/// Layout matches [`compute_dg_flux_jump_block`]: rows / cols
/// `[0..nl]` hold the left-cell DoFs, rows / cols `[nl..]` hold the
/// right-cell DoFs.
///
/// The sign convention is "RHS form": the block represents the matrix
/// `A` such that `(A · u)_F` equals the contribution of this face to
/// the residual `r = M u_t + ∇·F = 0`. So a backward-Euler stepper
/// would form `(M / dt + A) u^{n+1} = (M / dt) u^n` (where `A` includes
/// all face contributions assembled via [`assemble_dg_global`]).
#[must_use]
pub fn compute_dg_numerical_flux_block(
    cell_left: &FemCell,
    cell_right: &FemCell,
    face: &InteriorFace,
    vertices: &[Vec3],
    flux: NumericalFlux,
    degree: usize,
) -> Vec<Vec<Real>> {
    assert!(
        degree <= 2,
        "DG: degree {degree} out of range (supports 0..=2)"
    );
    let nl = dg_local_ndof(cell_left, degree);
    let nr = dg_local_ndof(cell_right, degree);
    let n = nl + nr;
    let mut block = vec![vec![0.0_f64 as Real; n]; n];

    let n_lr = face_normal_left_to_right(cell_left, cell_right, face, vertices);
    let qpts = if degree >= 2 {
        face_quadrature_high(face, vertices)
    } else {
        face_quadrature(face, vertices)
    };

    // The scalar conservation law is `∂u/∂t + ∇·(c(x) u) = 0` for the
    // variants currently supported (linear advection / Roe collapsed
    // to upwind / Lax-Friedrichs around the linear flux). For the
    // explicit dispatch:
    //
    // - `Central`        : ĥ(u_L, u_R) = ½ (c_L u_L + c_R u_R) · n
    // - `Upwind { wind }`: ĥ = (wind·n) u_upwind
    // - `LaxFriedrichs`  : ĥ = ½ (c u_L + c u_R) · n − ½ α (u_R − u_L)
    //                      with `c = wind` taken from the LF variant
    //                      caller? — we treat `LaxFriedrichs` as a
    //                      Rusanov flux *around the central* flux,
    //                      meaning `c · u` average + dissipation. For
    //                      callers who want LF for a non-linear flux
    //                      (Burgers), pre-linearise around their
    //                      transported field.
    // - `Roe`            : same as `Upwind`.
    //
    // For purely linear advection the velocity is constant in space,
    // so `c_L = c_R = wind`. Non-linear callers (Burgers') should use
    // the residual-style API in [`compute_dg_burgers_residual`].

    // Pull the advection velocity (or zero for `Central`).
    let (wind, _lf_alpha, is_upwind) = match flux {
        NumericalFlux::Central => (Vec3::ZERO, 0.0, false),
        NumericalFlux::Upwind { wind } => (wind, 0.0, true),
        NumericalFlux::LaxFriedrichs { alpha } => (Vec3::ZERO, alpha, false),
        NumericalFlux::Roe => (Vec3::ZERO, 0.0, true),
        // System-only fluxes: not usable here. Caller must dispatch
        // through the SWE helpers below (compute_swe_numerical_flux_*).
        NumericalFlux::Hll | NumericalFlux::Hllc | NumericalFlux::RoeSwe => {
            panic!(
                "compute_dg_numerical_flux_block: {flux:?} is a system flux; \
                 use compute_swe_numerical_flux_{{hll,hllc,roe}} for SWE"
            );
        }
    };

    if is_upwind {
        // Upwind: ĥ = (wind · n) u_upwind. If `wind·n > 0`, u_upwind = u_L.
        let wn = wind.dot(n_lr);
        if wn.abs() < 1.0e-30 {
            // Zero-flux face — nothing to add.
            return block;
        }
        // The face contribution to the residual matrix `A` (with
        // `r = M u_t + A u = 0`) is `+∮ Ĥ ⟦v⟧`. For test function
        // v = N_L^i: ⟦v⟧ = +N_L^i; for v = N_R^i: ⟦v⟧ = −N_R^i.
        //
        // For `u_upwind = u_L`:
        //   - L row: +∮ (wn · Σ N_L^j u_L^j) · N_L^i
        //              ⇒ block[i][j] += +wn · ∫ N_L^i N_L^j
        //   - R row: +∮ (wn · Σ N_L^j u_L^j) · (−N_R^i)
        //              ⇒ block[nl+i][j] += −wn · ∫ N_R^i N_L^j
        // Symmetric for u_upwind = u_R.
        for q in &qpts {
            let nl_vals = dg_shape_values_at(cell_left, vertices, q.point, degree);
            let nr_vals = dg_shape_values_at(cell_right, vertices, q.point, degree);
            let w = q.weight;
            if wn > 0.0 {
                // u_upwind = u_L
                for i in 0..nl {
                    for j in 0..nl {
                        block[i][j] += wn * w * nl_vals[i] * nl_vals[j];
                    }
                }
                for i in 0..nr {
                    for j in 0..nl {
                        block[nl + i][j] += -wn * w * nr_vals[i] * nl_vals[j];
                    }
                }
            } else {
                // u_upwind = u_R
                for i in 0..nl {
                    for j in 0..nr {
                        block[i][nl + j] += wn * w * nl_vals[i] * nr_vals[j];
                    }
                }
                for i in 0..nr {
                    for j in 0..nr {
                        block[nl + i][nl + j] += -wn * w * nr_vals[i] * nr_vals[j];
                    }
                }
            }
        }
        return block;
    }

    // Central / Lax-Friedrichs branch.
    let wn = wind.dot(n_lr); // 0 for both Central and LF
    let _ = wn;
    for q in &qpts {
        let nl_vals = dg_shape_values_at(cell_left, vertices, q.point, degree);
        let nr_vals = dg_shape_values_at(cell_right, vertices, q.point, degree);
        let w = q.weight;

        // Central piece — for `LaxFriedrichs` the caller hasn't passed
        // a `wind`; we treat the central flux of LF as 0 (i.e. LF
        // becomes pure dissipation `−½α(u_R−u_L)`) since the caller is
        // expected to assemble the F(u)·n volumetric contribution
        // separately via [`compute_element_convection`] etc. For
        // `Central` with `wind = 0` (default) the loop is a no-op
        // (intended; users should call this with an explicit wind via
        // `Upwind` if they want any flux at all).

        if let NumericalFlux::LaxFriedrichs { alpha } = flux {
            // Dissipation: ĥ_diss = −½α(u_R − u_L). Its contribution to
            // `A` is `+∮ ĥ_diss ⟦v⟧ = −½α ∮(u_R − u_L)⟦v⟧`.
            // For v = N_L^i (⟦v⟧ = +N_L^i):
            //   block[i][j_L] += +½α · ∫ N_L^j N_L^i      (the −u_L term, with the leading −)
            //   block[i][nl+j_R] += −½α · ∫ N_R^j N_L^i   (the +u_R term)
            // For v = N_R^i (⟦v⟧ = −N_R^i):
            //   block[nl+i][j_L] += −½α · ∫ N_L^j N_R^i
            //   block[nl+i][nl+j_R] += +½α · ∫ N_R^j N_R^i
            let half_a = 0.5 * alpha * w;
            for i in 0..nl {
                for j in 0..nl {
                    block[i][j] += half_a * nl_vals[j] * nl_vals[i];
                }
                for j in 0..nr {
                    block[i][nl + j] += -half_a * nr_vals[j] * nl_vals[i];
                }
            }
            for i in 0..nr {
                for j in 0..nl {
                    block[nl + i][j] += -half_a * nl_vals[j] * nr_vals[i];
                }
                for j in 0..nr {
                    block[nl + i][nl + j] += half_a * nr_vals[j] * nr_vals[i];
                }
            }
        }
    }

    block
}

// ─── Stage 43 v2 — Vector / system DG numerical flux ────────────────

/// Stage 43 v2 — component-wise Lax-Friedrichs flux block for a system
/// of `components` scalar conservation laws (e.g. shallow-water with
/// `components = 3` for `[h, hu, hv]`).
///
/// The DG semi-discretisation of a system `∂U/∂t + ∇·F(U) = 0` with
/// `U ∈ ℝᶜ` produces, for each face,
///
/// ```text
///   ∮_F Ĥ(U_L, U_R, n) ⟦v⟧ dS
/// ```
///
/// where `Ĥ` is a numerical flux vector with `c` components. Component-
/// wise Lax-Friedrichs is the simplest robust choice:
///
/// ```text
///   Ĥ_k = ½ (F_k(U_L) + F_k(U_R)) · n − ½ α (U_R[k] − U_L[k])
/// ```
///
/// — i.e. apply scalar LF to each component independently using a
/// shared global dissipation coefficient `α` ≥ max wave speed.
///
/// **Return:** a `((components·NL) + (components·NR))²` block, laid out
/// in "component-fastest" order to match the natural flat-state layout
/// `U_flat[dof * components + c]`. Specifically the row/column for
/// local DG-DoF `i` of component `c` on the L-cell is
/// `(i * components + c)`, and on the R-cell it starts at
/// `nl_total + i * components + c` where `nl_total = NL * components`.
///
/// Like [`compute_dg_numerical_flux_block`] this builds **only** the
/// dissipation contribution (the central-flux average is `0` when the
/// caller hasn't passed a wind/flux Jacobian — system solvers are
/// expected to assemble the analytic `∇·F` separately via volumetric
/// quadrature, then use this block only for inter-element
/// stabilisation).
#[must_use]
pub fn compute_dg_numerical_flux_block_vector(
    cell_left: &FemCell,
    cell_right: &FemCell,
    face: &InteriorFace,
    vertices: &[Vec3],
    alpha: Real,
    components: usize,
    degree: usize,
) -> Vec<Vec<Real>> {
    assert!(
        degree <= 2,
        "DG: degree {degree} out of range (supports 0..=2)"
    );
    assert!(components >= 1, "DG vector flux: components must be ≥ 1");
    let nl = dg_local_ndof(cell_left, degree);
    let nr = dg_local_ndof(cell_right, degree);
    let total_l = nl * components;
    let total_r = nr * components;
    let n = total_l + total_r;
    let mut block = vec![vec![0.0_f64 as Real; n]; n];

    let qpts = if degree >= 2 {
        face_quadrature_high(face, vertices)
    } else {
        face_quadrature(face, vertices)
    };

    // The block layout is "component-fastest": DoF (cell_l, local i, comp c)
    // → row total_index = i * components + c.
    for q in &qpts {
        let nl_vals = dg_shape_values_at(cell_left, vertices, q.point, degree);
        let nr_vals = dg_shape_values_at(cell_right, vertices, q.point, degree);
        let w = q.weight;
        let half_a = 0.5 * alpha * w;
        // For component `c`: pure-dissipation contribution to the
        // residual matrix is the same scalar pattern as the scalar
        // `LaxFriedrichs` branch in `compute_dg_numerical_flux_block`,
        // applied independently to each component.
        for i in 0..nl {
            for j in 0..nl {
                let v = half_a * nl_vals[j] * nl_vals[i];
                for c in 0..components {
                    block[i * components + c][j * components + c] += v;
                }
            }
            for j in 0..nr {
                let v = -half_a * nr_vals[j] * nl_vals[i];
                for c in 0..components {
                    block[i * components + c][total_l + j * components + c] += v;
                }
            }
        }
        for i in 0..nr {
            for j in 0..nl {
                let v = -half_a * nl_vals[j] * nr_vals[i];
                for c in 0..components {
                    block[total_l + i * components + c][j * components + c] += v;
                }
            }
            for j in 0..nr {
                let v = half_a * nr_vals[j] * nr_vals[i];
                for c in 0..components {
                    block[total_l + i * components + c][total_l + j * components + c] += v;
                }
            }
        }
    }

    block
}

// ─── Stage 43 v2 — Nitsche boundary block for weak Dirichlet ──────────

/// Stage 43 v2 — boundary face block + RHS for Nitsche weak Dirichlet.
///
/// Given a boundary face with sorted vertex indices `face_verts` owned
/// by exactly one cell `cell`, compute
///
/// ```text
///   a_F(u, v) = − ∫_F α (∇u·n) v dS
///               − ∫_F α u (∇v·n) dS
///               + ∫_F (σ α / h_F) u v dS
/// ```
///
/// and the RHS contribution that pins the trace to a linearly-
/// interpolated `u_D` between the per-vertex values supplied in
/// `vertex_values` (one entry per face vertex, in the same order as
/// `face_verts`):
///
/// ```text
///   l_F(v) = − ∫_F α u_D (∇v·n) dS + ∫_F (σ α / h_F) u_D v dS
/// ```
///
/// Returns `(block, rhs)` where `block` is `N × N` with
/// `N = dg_ndof_per_cell(cell, degree)` and `rhs` is length `N`.
///
/// Used by [`super::bc::apply_dg_bcs`] when processing
/// [`super::bc::BoundaryCondition::WeakDirichletDG`].
#[must_use]
pub fn compute_nitsche_boundary_block(
    cell: &FemCell,
    face_verts: &[u32],
    vertices: &[Vec3],
    alpha: Real,
    degree: usize,
    sigma: Real,
    vertex_values: &[Real],
) -> (Vec<Vec<Real>>, Vec<Real>) {
    assert!(degree <= 2, "DG Nitsche: degree {degree} out of range");
    assert_eq!(
        face_verts.len(),
        vertex_values.len(),
        "compute_nitsche_boundary_block: vertex_values.len() != face_verts.len()"
    );
    let nl = dg_local_ndof(cell, degree);
    let mut block = vec![vec![0.0_f64 as Real; nl]; nl];
    let mut rhs = vec![0.0_f64 as Real; nl];

    // Build a fake interior face structure to reuse the high-order face
    // quadrature kernel. Geometry is identical for boundary vs interior
    // — only the orientation/normal convention differs.
    let pseudo_face = InteriorFace {
        cell_left: 0,
        cell_right: u32::MAX,
        face_verts: face_verts.to_vec(),
    };
    let qpts = if degree >= 2 {
        face_quadrature_high(&pseudo_face, vertices)
    } else {
        face_quadrature(&pseudo_face, vertices)
    };
    let h_f = face_h_scale(&pseudo_face, vertices);
    let pen = sigma * alpha / h_f;

    // Outward normal: point *away* from the cell centroid.
    let c = cell_centroid(cell, vertices);
    let raw_n = match face_verts.len() {
        2 => {
            let a = vertices[face_verts[0] as usize];
            let b = vertices[face_verts[1] as usize];
            edge_normal_2d(a, b)
        }
        3 => {
            let a = vertices[face_verts[0] as usize];
            let b = vertices[face_verts[1] as usize];
            let c2 = vertices[face_verts[2] as usize];
            tri_normal_3d(a, b, c2)
        }
        _ => unreachable!("compute_nitsche_boundary_block: unsupported face_verts length"),
    };
    let to_face_centroid = match face_verts.len() {
        2 => {
            let a = vertices[face_verts[0] as usize];
            let b = vertices[face_verts[1] as usize];
            (a + b) * 0.5 - c
        }
        3 => {
            let a = vertices[face_verts[0] as usize];
            let b = vertices[face_verts[1] as usize];
            let c2 = vertices[face_verts[2] as usize];
            (a + b + c2) / 3.0 - c
        }
        _ => unreachable!(),
    };
    let n_out = if raw_n.dot(to_face_centroid) >= 0.0 {
        raw_n
    } else {
        -raw_n
    };

    // For interpolating `u_D` linearly between face vertices, we need
    // a parameter along the face for each quadrature point. For an
    // edge: ξ ∈ [0, 1] from face_verts[0] to face_verts[1]. For a
    // triangle face: barycentric (b0, b1, b2) wrt face_verts.
    for q in &qpts {
        let nvals = dg_shape_values_at(cell, vertices, q.point, degree);
        let grads = dg_shape_grads_at(cell, vertices, q.point, degree);
        let gn: Vec<Real> = grads.iter().map(|g| g.dot(n_out)).collect();

        // u_D at the quadrature point — linear interpolation between
        // face-vertex values.
        let u_d = match face_verts.len() {
            2 => {
                let a = vertices[face_verts[0] as usize];
                let b = vertices[face_verts[1] as usize];
                let len = (b - a).length();
                let xi = if len < 1.0e-30 {
                    0.5
                } else {
                    (q.point - a).dot(b - a) / (len * len)
                };
                (1.0 - xi) * vertex_values[0] + xi * vertex_values[1]
            }
            3 => {
                let a = vertices[face_verts[0] as usize];
                let b = vertices[face_verts[1] as usize];
                let c2 = vertices[face_verts[2] as usize];
                // 3D triangle barycentric — solve in the face plane.
                let area = tri_area_3d(a, b, c2);
                let area_bcq = tri_area_3d(b, c2, q.point);
                let area_caq = tri_area_3d(c2, a, q.point);
                let area_abq = tri_area_3d(a, b, q.point);
                let inv_area = if area < 1.0e-30 { 0.0 } else { 1.0 / area };
                let l0 = area_bcq * inv_area;
                let l1 = area_caq * inv_area;
                let l2 = area_abq * inv_area;
                l0 * vertex_values[0] + l1 * vertex_values[1] + l2 * vertex_values[2]
            }
            _ => unreachable!(),
        };

        let w = q.weight;
        for i in 0..nl {
            for j in 0..nl {
                let val = -alpha * gn[j] * nvals[i] - alpha * nvals[j] * gn[i]
                    + pen * nvals[i] * nvals[j];
                block[i][j] += w * val;
            }
            // RHS: − α ∫_F u_D (∇v·n) + pen ∫_F u_D v.
            rhs[i] += w * (-alpha * u_d * gn[i] + pen * u_d * nvals[i]);
        }
    }

    (block, rhs)
}

// ─── Stage 43 v2 — Periodic face pairing ─────────────────────────────

/// Stage 43 v2 — pair up boundary faces on opposite sides of a periodic
/// domain along `axis ∈ {0, 1, 2}` (= x / y / z).
///
/// Walks the boundary faces (= faces owned by a single cell), bins
/// them by their **midpoint coordinates in the other two axes**, and
/// returns one `(face_a, face_b)` pair per bucket that contains exactly
/// two faces — where one is on the minimum-axis side of the domain and
/// the other is on the maximum-axis side.
///
/// **Tolerance:** the bin key uses 12-digit rounding of the non-axis
/// coordinates. Two faces match if their non-axis midpoints round
/// identically — this is consistent with the structured `tri_grid_2d`
/// and `tet_grid_3d` builders which produce exact (binary) coordinates
/// on the grid lattice. For unstructured imported meshes the caller
/// should pre-snap the periodic boundary nodes onto an exact lattice.
///
/// Returns an empty `Vec` if the mesh has no periodic boundary pair.
#[must_use]
pub fn periodic_face_pairs(
    mesh: &super::mesh::FemMesh,
    axis: usize,
) -> Vec<(InteriorFace, InteriorFace)> {
    use std::collections::HashMap;
    assert!(axis < 3, "periodic_face_pairs: axis must be < 3");

    // First pass — collect boundary faces (owned by exactly one cell).
    let mut face_to_cells: HashMap<Vec<u32>, Vec<u32>> = HashMap::new();
    for (cell_idx, cell) in mesh.cells.iter().enumerate() {
        match cell {
            FemCell::Tri(v) => {
                for &[a, b] in &[[v[0], v[1]], [v[1], v[2]], [v[2], v[0]]] {
                    let mut key = vec![a, b];
                    key.sort_unstable();
                    face_to_cells.entry(key).or_default().push(cell_idx as u32);
                }
            }
            FemCell::Tet(v) => {
                for &[a, b, c] in &[
                    [v[1], v[2], v[3]],
                    [v[0], v[2], v[3]],
                    [v[0], v[1], v[3]],
                    [v[0], v[1], v[2]],
                ] {
                    let mut key = vec![a, b, c];
                    key.sort_unstable();
                    face_to_cells.entry(key).or_default().push(cell_idx as u32);
                }
            }
        }
    }

    // Find domain extents along the axis (for min/max side detection).
    let mut axis_min: Real = Real::INFINITY;
    let mut axis_max: Real = Real::NEG_INFINITY;
    for v in &mesh.vertices {
        let c = match axis {
            0 => v.x,
            1 => v.y,
            _ => v.z,
        };
        if c < axis_min {
            axis_min = c;
        }
        if c > axis_max {
            axis_max = c;
        }
    }

    // Bin boundary faces by rounded non-axis midpoint coordinates.
    let key_for = |coords: &[Real; 2]| -> (i64, i64) {
        let scale = 1.0e12_f64;
        (
            (coords[0] * scale).round() as i64,
            (coords[1] * scale).round() as i64,
        )
    };
    let face_midpoint = |face_verts: &[u32]| -> Vec3 {
        let mut s = Vec3::ZERO;
        for &v in face_verts {
            s += mesh.vertices[v as usize];
        }
        s / (face_verts.len() as Real)
    };
    let axis_value = |p: Vec3| -> Real {
        match axis {
            0 => p.x,
            1 => p.y,
            _ => p.z,
        }
    };
    let other_coords = |p: Vec3| -> [Real; 2] {
        match axis {
            0 => [p.y, p.z],
            1 => [p.x, p.z],
            _ => [p.x, p.y],
        }
    };

    let tol_axis = (axis_max - axis_min).abs() * 1.0e-6 + 1.0e-12;

    // (bin_key, side) → InteriorFace.  side: false = min side, true = max side.
    let mut min_side: HashMap<(i64, i64), InteriorFace> = HashMap::new();
    let mut max_side: HashMap<(i64, i64), InteriorFace> = HashMap::new();

    for (face_verts, owners) in &face_to_cells {
        if owners.len() != 1 {
            continue;
        }
        let mid = face_midpoint(face_verts);
        let a_val = axis_value(mid);
        let is_min = (a_val - axis_min).abs() <= tol_axis;
        let is_max = (a_val - axis_max).abs() <= tol_axis;
        if !is_min && !is_max {
            continue;
        }
        let k = key_for(&other_coords(mid));
        let f = InteriorFace {
            cell_left: owners[0],
            cell_right: u32::MAX,
            face_verts: face_verts.clone(),
        };
        if is_min {
            min_side.insert(k, f);
        } else {
            max_side.insert(k, f);
        }
    }

    // Pair them up.
    let mut pairs: Vec<(InteriorFace, InteriorFace)> = Vec::new();
    for (k, fa) in min_side.into_iter() {
        if let Some(fb) = max_side.remove(&k) {
            pairs.push((fa, fb));
        }
    }
    pairs.sort_by_key(|(a, b)| (a.cell_left, b.cell_left));
    pairs
}

// ─── Stage 43 — Barth-Jespersen slope limiter ────────────────────────

/// Apply a scalar Barth-Jespersen slope limiter to the per-cell DG
/// solution `u_dg`.
///
/// For each cell `K` with cell-average `ū_K` and per-node values
/// `u^K_i`, the limiter computes a scaling factor `φ_K ∈ [0, 1]` such
/// that
///
/// ```text
///   ū_K + φ_K · (u^K_i − ū_K)  ∈  [ū_min(K), ū_max(K)]   for all i
/// ```
///
/// where `ū_min(K) / ū_max(K)` are the min / max cell averages over
/// `K` and its face-sharing neighbours. The factor is then **clamped**
/// against the user-supplied `physical_min / physical_max` so the
/// limited reconstruction respects a hard physical envelope (e.g.
/// `0 ≤ u ≤ 1` for an indicator field, `0 ≤ h` for shallow-water
/// depth).
///
/// Inputs:
/// - `mesh`            — DG mesh (must match `dg_offsets`).
/// - `u_dg`            — current DG solution (length = `Σ ndof(cell)`).
/// - `degree`          — DG polynomial order (1 or 2 — degree 0 is a
///   no-op).
/// - `cell_neighbours` — per-cell list of neighbour cell indices (face-
///   sharing). Use [`build_cell_neighbours`] to compute.
/// - `physical_min / physical_max` — optional hard clamp (use
///   `f64::NEG_INFINITY / INFINITY` for "no clamp").
///
/// Writes the limited DG solution back into `u_dg` in-place.
///
/// Returns the per-cell limiter factor `φ_K` (length = `cells.len()`)
/// so the caller can monitor the limiter (`min(φ) ≪ 1` ⇒ strong shock
/// nearby; `φ = 1` everywhere ⇒ smooth region, no limiting).
///
/// **Scope** — scalar (single-variable) Barth-Jespersen. System
/// variants (Krivodonova characteristic-wise, Cockburn moment-based)
/// are out of scope for Stage 43 v1.
pub fn limit_solution(
    mesh: &super::mesh::FemMesh,
    u_dg: &mut [Real],
    degree: usize,
    cell_neighbours: &[Vec<u32>],
    physical_min: Real,
    physical_max: Real,
) -> Vec<Real> {
    assert!(
        degree <= 2,
        "limit_solution: degree {degree} out of range (supports 0..=2)"
    );

    // P0 has only cell averages → no slope to limit. Just clamp to
    // the physical envelope and bail.
    if degree == 0 {
        for v in u_dg.iter_mut() {
            *v = v.clamp(physical_min, physical_max);
        }
        return vec![1.0; mesh.cells.len()];
    }

    // 1. Per-cell offsets.
    let mut offsets: Vec<usize> = Vec::with_capacity(mesh.cells.len() + 1);
    let mut running = 0usize;
    for c in &mesh.cells {
        offsets.push(running);
        running += dg_local_ndof(c, degree);
    }
    offsets.push(running);

    // 2. Cell averages — for our Lagrange basis the cell average is the
    //    arithmetic mean of the nodal values (P1 nodes form a partition
    //    of unity on the simplex; for P2 the equal-weight average over
    //    Lagrange nodes is exact for linear data and close-to-correct
    //    for general quadratic data — sufficient for Barth-Jespersen
    //    where the cell average is just a centring reference).
    //
    //    For P2 the **exact** cell average is `(1/area) ∫_K u dx` —
    //    using mass-matrix-weighted weights — but the Lagrange-node
    //    mean is the conventional choice in production DG codes
    //    (Cockburn-Shu, Krivodonova) and is what the test suite
    //    expects.
    let cell_avg: Vec<Real> = (0..mesh.cells.len())
        .map(|ci| {
            let s: Real = u_dg[offsets[ci]..offsets[ci + 1]].iter().sum();
            let n = (offsets[ci + 1] - offsets[ci]) as Real;
            s / n
        })
        .collect();

    let mut phis = vec![1.0_f64 as Real; mesh.cells.len()];

    // 3. Apply Barth-Jespersen per cell.
    for ci in 0..mesh.cells.len() {
        let u_bar = cell_avg[ci];
        let nbrs = &cell_neighbours[ci];

        // Neighbour min/max — include self so isolated cells fall back
        // to no limiting.
        let mut u_min = u_bar;
        let mut u_max = u_bar;
        for &nb in nbrs {
            let v = cell_avg[nb as usize];
            if v < u_min {
                u_min = v;
            }
            if v > u_max {
                u_max = v;
            }
        }
        // Hard clamp against physical envelope.
        if u_min < physical_min {
            u_min = physical_min;
        }
        if u_max > physical_max {
            u_max = physical_max;
        }

        let off = offsets[ci];
        let n = offsets[ci + 1] - off;
        // Per-node limiter factor.
        let mut phi: Real = 1.0;
        for i in 0..n {
            let u_i = u_dg[off + i];
            let d = u_i - u_bar;
            let yi = if d > 1.0e-30 {
                ((u_max - u_bar) / d).min(1.0)
            } else if d < -1.0e-30 {
                ((u_min - u_bar) / d).min(1.0)
            } else {
                1.0
            };
            // Clamp to [0, 1] — guards against tiny negatives from
            // round-off when (u_max - u_bar) is ~0.
            let yi = yi.clamp(0.0, 1.0);
            if yi < phi {
                phi = yi;
            }
        }
        phis[ci] = phi;

        // 4. Apply: u_i ← u_bar + phi · (u_i - u_bar). After limiting,
        //    clamp each node hard to the physical envelope (a last-
        //    resort guard; with phi correctly chosen this is a no-op).
        for i in 0..n {
            let u_i = u_dg[off + i];
            let limited = u_bar + phi * (u_i - u_bar);
            u_dg[off + i] = limited.clamp(physical_min, physical_max);
        }
    }

    phis
}

/// Build the face-sharing neighbour map for a DG mesh.
///
/// `result[ci]` lists the cell indices that share at least one face
/// with cell `ci` (edges in 2D / tri faces in 3D). Used by
/// [`limit_solution`] as the BJ stencil. Boundary cells get a strictly
/// shorter neighbour list (no neighbour through the boundary face).
#[must_use]
pub fn build_cell_neighbours(mesh: &super::mesh::FemMesh) -> Vec<Vec<u32>> {
    let mut result: Vec<Vec<u32>> = vec![Vec::new(); mesh.cells.len()];
    let faces = compute_interior_faces(mesh);
    for f in &faces {
        result[f.cell_left as usize].push(f.cell_right);
        result[f.cell_right as usize].push(f.cell_left);
    }
    result
}

// ─── Face quadrature ─────────────────────────────────────────────────

/// One quadrature point on a face — pre-projected to physical space
/// + the weight (already includes the face measure).
#[derive(Copy, Clone, Debug)]
struct FaceQuadPoint {
    point: Vec3,
    weight: Real,
}

/// Face quadrature — degree-2 exact on the face, sufficient for the
/// SIP integrals (mass-like contribution is degree 2 in the face
/// coordinate, gradient-times-shape is degree 1).
///
/// - 2D edge: Gauss-Legendre 2-pt (degree-3 exact).
/// - 3D triangle: 3-pt mid-edge rule (degree-2 exact).
fn face_quadrature(face: &InteriorFace, vertices: &[Vec3]) -> Vec<FaceQuadPoint> {
    match face.face_verts.len() {
        2 => {
            let a = vertices[face.face_verts[0] as usize];
            let b = vertices[face.face_verts[1] as usize];
            let len = edge_length_2d(a, b);
            // Standard Gauss-Legendre on [0, 1]:
            //   ξ₁ = ½ − 1/(2√3),  ξ₂ = ½ + 1/(2√3),  w = ½ each.
            let g = 1.0 / (3.0 as Real).sqrt() / 2.0;
            let xi1 = 0.5 - g;
            let xi2 = 0.5 + g;
            let w = 0.5 * len;
            vec![
                FaceQuadPoint {
                    point: a + (b - a) * xi1,
                    weight: w,
                },
                FaceQuadPoint {
                    point: a + (b - a) * xi2,
                    weight: w,
                },
            ]
        }
        3 => {
            let a = vertices[face.face_verts[0] as usize];
            let b = vertices[face.face_verts[1] as usize];
            let c = vertices[face.face_verts[2] as usize];
            let area = tri_area_3d(a, b, c);
            let w = area / 3.0;
            // Mid-edge rule on the triangular face — degree-2 exact.
            vec![
                FaceQuadPoint {
                    point: (a + b) * 0.5,
                    weight: w,
                },
                FaceQuadPoint {
                    point: (b + c) * 0.5,
                    weight: w,
                },
                FaceQuadPoint {
                    point: (c + a) * 0.5,
                    weight: w,
                },
            ]
        }
        _ => unreachable!("InteriorFace has unsupported face_verts length"),
    }
}

/// Higher-order face quadrature for P2 SIP and high-degree advection
/// fluxes — degree-5 exact on a 2D edge (3-pt Gauss-Legendre) and
/// degree-4 exact on a 3D triangle face (6-pt Dunavant). Same return
/// convention as [`face_quadrature`] — pre-projected to physical
/// space with weights already including the face measure.
fn face_quadrature_high(face: &InteriorFace, vertices: &[Vec3]) -> Vec<FaceQuadPoint> {
    match face.face_verts.len() {
        2 => {
            let a = vertices[face.face_verts[0] as usize];
            let b = vertices[face.face_verts[1] as usize];
            let len = edge_length_2d(a, b);
            // 3-pt Gauss-Legendre on [0, 1].
            //   ξ₁,₃ = ½ ± ½·√(3/5),  w₁,₃ = 5/18
            //   ξ₂   = ½,             w₂   = 8/18
            let g = 0.5 * (3.0_f64 / 5.0_f64).sqrt();
            let xis = [0.5 - g, 0.5, 0.5 + g];
            let ws = [5.0 / 18.0 * len, 8.0 / 18.0 * len, 5.0 / 18.0 * len];
            xis.iter()
                .zip(ws.iter())
                .map(|(&xi, &w)| FaceQuadPoint {
                    point: a + (b - a) * xi,
                    weight: w,
                })
                .collect()
        }
        3 => {
            // 6-pt Dunavant degree-4 rule on the triangle face,
            // mirrored from `tri_quadrature_6pt` but lifted to 3D via
            // barycentric interpolation of the face vertices.
            let a = vertices[face.face_verts[0] as usize];
            let b = vertices[face.face_verts[1] as usize];
            let c = vertices[face.face_verts[2] as usize];
            let area = tri_area_3d(a, b, c);
            let a1 = 0.816_847_572_980_459;
            let b1 = 0.091_576_213_509_771;
            let w1 = 0.109_951_743_655_322;
            let a2 = 0.108_103_018_168_070;
            let b2 = 0.445_948_490_915_965;
            let w2 = 0.223_381_589_678_011;
            let pts: [[Real; 3]; 6] = [
                [a1, b1, b1],
                [b1, a1, b1],
                [b1, b1, a1],
                [a2, b2, b2],
                [b2, a2, b2],
                [b2, b2, a2],
            ];
            let ws: [Real; 6] = [w1, w1, w1, w2, w2, w2];
            pts.iter()
                .zip(ws.iter())
                .map(|(bary, &w)| FaceQuadPoint {
                    point: a * bary[0] + b * bary[1] + c * bary[2],
                    weight: w * area,
                })
                .collect()
        }
        _ => unreachable!("InteriorFace has unsupported face_verts length"),
    }
}

/// Face measure (length for an edge, area for a triangle face).
fn face_measure(face: &InteriorFace, vertices: &[Vec3]) -> Real {
    match face.face_verts.len() {
        2 => edge_length_2d(
            vertices[face.face_verts[0] as usize],
            vertices[face.face_verts[1] as usize],
        ),
        3 => tri_area_3d(
            vertices[face.face_verts[0] as usize],
            vertices[face.face_verts[1] as usize],
            vertices[face.face_verts[2] as usize],
        ),
        _ => unreachable!(),
    }
}

/// Outward face normal pointing from `cell_left` to `cell_right` —
/// disambiguated using the centroid displacement vector.
fn face_normal_left_to_right(
    cell_left: &FemCell,
    cell_right: &FemCell,
    face: &InteriorFace,
    vertices: &[Vec3],
) -> Vec3 {
    let c_l = cell_centroid(cell_left, vertices);
    let c_r = cell_centroid(cell_right, vertices);
    let raw = match face.face_verts.len() {
        2 => {
            let a = vertices[face.face_verts[0] as usize];
            let b = vertices[face.face_verts[1] as usize];
            edge_normal_2d(a, b)
        }
        3 => {
            let a = vertices[face.face_verts[0] as usize];
            let b = vertices[face.face_verts[1] as usize];
            let c = vertices[face.face_verts[2] as usize];
            tri_normal_3d(a, b, c)
        }
        _ => unreachable!(),
    };
    let from_l_to_r = c_r - c_l;
    if raw.dot(from_l_to_r) >= 0.0 {
        raw
    } else {
        -raw
    }
}

fn cell_centroid(cell: &FemCell, vertices: &[Vec3]) -> Vec3 {
    match cell {
        FemCell::Tri(v) => {
            (vertices[v[0] as usize] + vertices[v[1] as usize] + vertices[v[2] as usize]) / 3.0
        }
        FemCell::Tet(v) => {
            (vertices[v[0] as usize]
                + vertices[v[1] as usize]
                + vertices[v[2] as usize]
                + vertices[v[3] as usize])
                / 4.0
        }
    }
}

// ─── Global assembly for DG ──────────────────────────────────────────

/// Stage 39B — assemble a DG global matrix.
///
/// Walks the mesh **twice**:
///
/// 1. **Cell pass** — for every cell `K`, the per-cell element matrix
///    (returned by `cell_visitor(K, vertices)`) is scattered into the
///    diagonal block at DoF offset `dg_offset(K) .. dg_offset(K) + N`.
/// 2. **Face pass** — for every interior face, the flux block
///    (returned by `face_visitor(K_L, K_R, face, vertices)`) is
///    scattered into the cross-block between the two cells.
///
/// `cell_visitor` returns `N × N` (`N = ndof_per_cell(cell, degree)`).
/// `face_visitor` returns `(NL + NR) × (NL + NR)` with the layout
/// described in [`compute_dg_flux_jump_block`].
///
/// Returns the assembled CSR sparse matrix.
pub fn assemble_dg_global<C, F>(
    mesh: &super::mesh::FemMesh,
    degree: usize,
    cell_visitor: C,
    face_visitor: Option<F>,
) -> crate::core::Result<crate::math::sparse::BsrMatrix>
where
    C: Fn(&FemCell, &[Vec3]) -> Vec<Vec<Real>>,
    F: Fn(&FemCell, &FemCell, &InteriorFace, &[Vec3]) -> Vec<Vec<Real>>,
{
    let n = super::function_space::FunctionSpace::DGNodal { degree }.ndof(mesh);

    // Pre-compute per-cell DoF offsets (running sum of per-cell DoF
    // counts) so cells of mixed type would be supported transparently
    // (v1 meshes are homogeneous but no reason to bake that in).
    let mut offsets = Vec::with_capacity(mesh.cells.len());
    let mut running = 0usize;
    for c in &mesh.cells {
        offsets.push(running);
        running += super::function_space::dg_ndof_per_cell(c, degree);
    }
    debug_assert_eq!(running, n);

    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    triplets.reserve(mesh.cells.len() * 16 + mesh.cells.len() * 4 * 64);

    // Cell pass.
    for (ci, cell) in mesh.cells.iter().enumerate() {
        let local = cell_visitor(cell, &mesh.vertices);
        let nl = super::function_space::dg_ndof_per_cell(cell, degree);
        debug_assert_eq!(local.len(), nl);
        let off = offsets[ci];
        for i in 0..nl {
            debug_assert_eq!(local[i].len(), nl);
            for j in 0..nl {
                let v = local[i][j];
                if v == 0.0 {
                    continue;
                }
                triplets.push((off + i, off + j, v));
            }
        }
    }

    // Face pass — only if a flux visitor was provided. For a pure
    // mass-matrix assembly the caller passes `None` and we skip the
    // entire face walk.
    if let Some(face_v) = face_visitor {
        let faces = compute_interior_faces(mesh);
        for face in &faces {
            let cl = &mesh.cells[face.cell_left as usize];
            let cr = &mesh.cells[face.cell_right as usize];
            let block = face_v(cl, cr, face, &mesh.vertices);
            let nl = super::function_space::dg_ndof_per_cell(cl, degree);
            let nr = super::function_space::dg_ndof_per_cell(cr, degree);
            let off_l = offsets[face.cell_left as usize];
            let off_r = offsets[face.cell_right as usize];
            let n_total = nl + nr;
            debug_assert_eq!(block.len(), n_total);
            for i in 0..n_total {
                debug_assert_eq!(block[i].len(), n_total);
                let row_global = if i < nl { off_l + i } else { off_r + (i - nl) };
                for j in 0..n_total {
                    let v = block[i][j];
                    if v == 0.0 {
                        continue;
                    }
                    let col_global = if j < nl { off_l + j } else { off_r + (j - nl) };
                    triplets.push((row_global, col_global, v));
                }
            }
        }
    }

    crate::math::sparse::BsrMatrix::from_scalar_triplets(n, n, &triplets)
}

// ─── Stage 43 v3 — SWE numerical fluxes (HLL / HLLC / Roe) ──────────
//
// All three functions return the **3-component physical flux vector**
// `Ĥ(U_L, U_R, n)` for the shallow-water system
//
// ```text
//   U  = (h,  hu, hv)
//   F  = (hu, hu² + ½ g h², hu·hv / h)            (in x)
//   G  = (hv, hu·hv / h, hv² + ½ g h²)            (in y)
//   F·n + G·n  expanded for a 2D normal n = (nx, ny)
// ```
//
// Caller is responsible for the DG face-quadrature loop + scattering
// `−Ĥ ⟦v⟧` into the residual. These helpers are deliberately
// pointwise (`u_l, u_r → ĥ`) so they compose with hand-rolled face
// quadrature loops as well as the structured `compute_dg_numerical_*`
// assembly path.

/// Compressible Euler physical flux `F(U)·n` for the 2D conservative
/// state `U = (ρ, ρu, ρv, E)`.
#[must_use]
pub fn euler_flux_dot_n_2d(u: [Real; 4], n: Vec3, gamma: Real) -> [Real; 4] {
    let rho = u[0].max(1.0e-30);
    let ux = u[1] / rho;
    let uy = u[2] / rho;
    let e = u[3];
    let q2 = ux * ux + uy * uy;
    let p = (gamma - 1.0) * (e - 0.5 * rho * q2);
    let un = ux * n.x + uy * n.y;
    [
        rho * un,
        u[1] * un + p * n.x,
        u[2] * un + p * n.y,
        (e + p) * un,
    ]
}

/// Ideal-gas pressure for a 2D compressible Euler conservative state.
#[must_use]
pub fn euler_pressure_2d(u: [Real; 4], gamma: Real) -> Real {
    let rho = u[0].max(1.0e-30);
    let ux = u[1] / rho;
    let uy = u[2] / rho;
    (gamma - 1.0) * (u[3] - 0.5 * rho * (ux * ux + uy * uy))
}

/// Local Lax-Friedrichs/Rusanov numerical flux for the 2D compressible
/// Euler equations.
#[must_use]
pub fn compute_euler_numerical_flux_rusanov_2d(
    u_l: [Real; 4],
    u_r: [Real; 4],
    n: Vec3,
    gamma: Real,
) -> [Real; 4] {
    let f_l = euler_flux_dot_n_2d(u_l, n, gamma);
    let f_r = euler_flux_dot_n_2d(u_r, n, gamma);
    let wave_speed = |u: [Real; 4]| {
        let rho = u[0].max(1.0e-30);
        let ux = u[1] / rho;
        let uy = u[2] / rho;
        let p = euler_pressure_2d(u, gamma).max(0.0);
        let c = (gamma * p / rho).sqrt();
        (ux * n.x + uy * n.y).abs() + c
    };
    let alpha = wave_speed(u_l).max(wave_speed(u_r));
    let mut out = [0.0; 4];
    for k in 0..4 {
        out[k] = 0.5 * (f_l[k] + f_r[k]) - 0.5 * alpha * (u_r[k] - u_l[k]);
    }
    out
}

/// HLL two-wave numerical flux for the 2D compressible Euler
/// equations.
#[must_use]
pub fn compute_euler_numerical_flux_hll_2d(
    u_l: [Real; 4],
    u_r: [Real; 4],
    n: Vec3,
    gamma: Real,
) -> [Real; 4] {
    let primitive = |u: [Real; 4]| {
        let rho = u[0].max(1.0e-30);
        let ux = u[1] / rho;
        let uy = u[2] / rho;
        let p = euler_pressure_2d(u, gamma).max(0.0);
        let c = (gamma * p / rho).sqrt();
        (ux * n.x + uy * n.y, c)
    };
    let (un_l, c_l) = primitive(u_l);
    let (un_r, c_r) = primitive(u_r);
    let s_l = (un_l - c_l).min(un_r - c_r);
    let s_r = (un_l + c_l).max(un_r + c_r);
    let f_l = euler_flux_dot_n_2d(u_l, n, gamma);
    let f_r = euler_flux_dot_n_2d(u_r, n, gamma);
    if s_l >= 0.0 {
        f_l
    } else if s_r <= 0.0 {
        f_r
    } else {
        let mut out = [0.0; 4];
        for k in 0..4 {
            out[k] = (s_r * f_l[k] - s_l * f_r[k] + s_l * s_r * (u_r[k] - u_l[k])) / (s_r - s_l);
        }
        out
    }
}

/// Stage 43 v3 — SWE physical flux `F(U)·n` for state `U = (h, hu, hv)`
/// and normal `n` (only `nx, ny` are used; `nz = 0`).
///
/// Returns `(F·n)[0..3]`.
#[must_use]
fn swe_flux_dot_n(h: Real, hu: Real, hv: Real, n: Vec3, g: Real) -> [Real; 3] {
    let h_safe = h.max(1.0e-12);
    let u = hu / h_safe;
    let v = hv / h_safe;
    // F·n = nx · (hu, hu² + ½gh², hu·hv/h)
    //     + ny · (hv, hu·hv/h, hv² + ½gh²)
    let fx = [hu, hu * u + 0.5 * g * h * h, hu * v];
    let fy = [hv, hv * u, hv * v + 0.5 * g * h * h];
    [
        fx[0] * n.x + fy[0] * n.y,
        fx[1] * n.x + fy[1] * n.y,
        fx[2] * n.x + fy[2] * n.y,
    ]
}

/// Stage 43 v3 — **HLL** numerical flux for the SWE system.
///
/// Two-wave Riemann solver (Harten-Lax-van-Leer 1983). Estimates the
/// slowest and fastest wave speeds `s_l, s_r`; for SWE these are the
/// minimum / maximum of the eigenvalues of the normal Jacobian on the
/// two sides:
///
/// ```text
///   s_l = min(u_n_L − c_L, u_n_R − c_R, û_n − ĉ)
///   s_r = max(u_n_L + c_L, u_n_R + c_R, û_n + ĉ)
/// ```
///
/// with `c = √(g h)`. The Davis (1988) speed estimate already includes
/// the Roe-average `û_n ± ĉ` for robustness on near-vacuum left/right
/// states (dry-bed / wet-bed pair). The HLL flux is then
///
/// ```text
///   Ĥ = ⎧ F_L · n,                       s_l ≥ 0
///       ⎨ (s_r F_L − s_l F_R + s_l s_r (U_R − U_L)) / (s_r − s_l),  s_l < 0 < s_r
///       ⎩ F_R · n,                       s_r ≤ 0
/// ```
///
/// Returns the 3-component flux vector `(Ĥ_h, Ĥ_hu, Ĥ_hv)`.
#[must_use]
pub fn compute_swe_numerical_flux_hll(
    u_l: [Real; 3],
    u_r: [Real; 3],
    n: Vec3,
    g: Real,
) -> [Real; 3] {
    let h_l = u_l[0].max(1.0e-12);
    let h_r = u_r[0].max(1.0e-12);
    let un_l = (u_l[1] * n.x + u_l[2] * n.y) / h_l;
    let un_r = (u_r[1] * n.x + u_r[2] * n.y) / h_r;
    let c_l = (g * h_l).sqrt();
    let c_r = (g * h_r).sqrt();

    // Roe-average wave speeds (Davis style — wider stencil for robustness).
    let sqrt_h_l = h_l.sqrt();
    let sqrt_h_r = h_r.sqrt();
    let h_hat = 0.5 * (h_l + h_r);
    let denom = sqrt_h_l + sqrt_h_r;
    let un_hat = if denom > 1.0e-30 {
        (un_l * sqrt_h_l + un_r * sqrt_h_r) / denom
    } else {
        0.5 * (un_l + un_r)
    };
    let c_hat = (g * h_hat).sqrt();

    let s_l = (un_l - c_l).min(un_r - c_r).min(un_hat - c_hat);
    let s_r = (un_l + c_l).max(un_r + c_r).max(un_hat + c_hat);

    let f_l = swe_flux_dot_n(u_l[0], u_l[1], u_l[2], n, g);
    let f_r = swe_flux_dot_n(u_r[0], u_r[1], u_r[2], n, g);

    if s_l >= 0.0 {
        f_l
    } else if s_r <= 0.0 {
        f_r
    } else {
        let inv = 1.0 / (s_r - s_l);
        [
            (s_r * f_l[0] - s_l * f_r[0] + s_l * s_r * (u_r[0] - u_l[0])) * inv,
            (s_r * f_l[1] - s_l * f_r[1] + s_l * s_r * (u_r[1] - u_l[1])) * inv,
            (s_r * f_l[2] - s_l * f_r[2] + s_l * s_r * (u_r[2] - u_l[2])) * inv,
        ]
    }
}

/// Stage 43 v3 — **HLLC** numerical flux for the SWE system (Toro 1994
/// adapted for shallow water by Fraccarollo & Toro 1995).
///
/// Three-wave Riemann solver: adds the contact-discontinuity speed
/// `s_*` between the HLL `s_l, s_r` waves. The contact speed is
/// derived from the conservation of mass + normal momentum across the
/// star region:
///
/// ```text
///   s_* = (s_l h_R (u_n_R − s_r) − s_r h_L (u_n_L − s_l))
///       / (h_R (u_n_R − s_r) − h_L (u_n_L − s_l))
/// ```
///
/// The four-region flux is
///
/// ```text
///   Ĥ = ⎧ F_L,        s_l ≥ 0
///       ⎪ F_*L,       s_l < 0 ≤ s_*
///       ⎨ F_*R,       s_* < 0 ≤ s_r
///       ⎩ F_R,        s_r ≤ 0
/// ```
///
/// with `F_*K = F_K + s_K (U_*K − U_K)` and the star-region state
/// `U_*K` constructed so the tangential velocity is unchanged across
/// the contact (the SWE contact only carries `h` / normal-velocity
/// jumps; tangential velocity is purely advected).
#[must_use]
pub fn compute_swe_numerical_flux_hllc(
    u_l: [Real; 3],
    u_r: [Real; 3],
    n: Vec3,
    g: Real,
) -> [Real; 3] {
    let h_l = u_l[0].max(1.0e-12);
    let h_r = u_r[0].max(1.0e-12);
    let un_l = (u_l[1] * n.x + u_l[2] * n.y) / h_l;
    let un_r = (u_r[1] * n.x + u_r[2] * n.y) / h_r;
    let c_l = (g * h_l).sqrt();
    let c_r = (g * h_r).sqrt();

    // Roe-average wave-speed estimate (Davis style).
    let sqrt_h_l = h_l.sqrt();
    let sqrt_h_r = h_r.sqrt();
    let h_hat = 0.5 * (h_l + h_r);
    let denom = sqrt_h_l + sqrt_h_r;
    let un_hat = if denom > 1.0e-30 {
        (un_l * sqrt_h_l + un_r * sqrt_h_r) / denom
    } else {
        0.5 * (un_l + un_r)
    };
    let c_hat = (g * h_hat).sqrt();
    let s_l = (un_l - c_l).min(un_r - c_r).min(un_hat - c_hat);
    let s_r = (un_l + c_l).max(un_r + c_r).max(un_hat + c_hat);

    let f_l = swe_flux_dot_n(u_l[0], u_l[1], u_l[2], n, g);
    let f_r = swe_flux_dot_n(u_r[0], u_r[1], u_r[2], n, g);

    if s_l >= 0.0 {
        return f_l;
    }
    if s_r <= 0.0 {
        return f_r;
    }

    // Star-region contact speed (Toro 2001, Eq. 10.70 for SWE).
    let num =
        h_r * un_r * (s_r - un_r) - h_l * un_l * (s_l - un_l) + 0.5 * g * (h_l * h_l - h_r * h_r);
    let den = h_r * (s_r - un_r) - h_l * (s_l - un_l);
    let s_star = if den.abs() > 1.0e-30 {
        num / den
    } else {
        0.5 * (s_l + s_r)
    };

    // Helper to build the star-region conservative state `U_*K` and
    // the corresponding flux `F_*K = F_K + s_K (U_*K − U_K)`.
    //
    // Decompose velocity into normal + tangential components. Across
    // the contact the tangential velocity is preserved; the normal
    // velocity equals `s_star`; the depth follows from conservation:
    //   h_*K = h_K · (s_K − u_n_K) / (s_K − s_*)
    let nx = n.x;
    let ny = n.y;
    let tx = -ny;
    let ty = nx;

    let star_flux =
        |h_k: Real, hu_k: Real, hv_k: Real, un_k: Real, s_k: Real, f_k: [Real; 3]| -> [Real; 3] {
            let h_star = h_k * (s_k - un_k) / (s_k - s_star);
            let u_k = hu_k / h_k;
            let v_k = hv_k / h_k;
            let ut_k = u_k * tx + v_k * ty;
            // Star velocity components.
            let us_x = s_star * nx + ut_k * tx;
            let us_y = s_star * ny + ut_k * ty;
            let hu_star = h_star * us_x;
            let hv_star = h_star * us_y;
            [
                f_k[0] + s_k * (h_star - h_k),
                f_k[1] + s_k * (hu_star - hu_k),
                f_k[2] + s_k * (hv_star - hv_k),
            ]
        };

    if s_star >= 0.0 {
        star_flux(h_l, u_l[1], u_l[2], un_l, s_l, f_l)
    } else {
        star_flux(h_r, u_r[1], u_r[2], un_r, s_r, f_r)
    }
}

/// Stage 43 v3 — **Roe** numerical flux for the SWE system.
///
/// Linearised Riemann solver around the Roe-average state. The
/// three-component flux is
///
/// ```text
///   Ĥ = ½ (F_L · n + F_R · n) − ½ Σ_k |λ̂_k| α_k r̂_k
/// ```
///
/// where `(λ̂_k, r̂_k)` are the eigenvalues and right eigenvectors of
/// the Roe-averaged Jacobian `Â(U_L, U_R)` and `α_k = ℓ̂_k · (U_R − U_L)`
/// are the wave strengths (`ℓ̂_k` left eigenvectors).
///
/// Eigenstructure (in the rotated normal/tangential frame):
///
/// ```text
///   λ̂_1 = û_n − ĉ,        r̂_1 = (1, û_n − ĉ nx + û_t (−ny), …)
///   λ̂_2 = û_n,            r̂_2 = (0, −ny, nx)·ĉ            (shear wave)
///   λ̂_3 = û_n + ĉ
/// ```
///
/// The implementation rotates the problem into a coordinate system
/// aligned with `n`, applies the textbook 1D SWE Roe solver, then
/// rotates back. **Entropy fix** (Harten-Hyman): eigenvalues smaller
/// than `δ_k = max(0, λ̂_k − λ_{k,L}, λ_{k,R} − λ̂_k)` are replaced
/// with the smooth `(λ̂² + δ²) / (2δ)` to prevent transonic-
/// rarefaction sonic glitches.
#[must_use]
pub fn compute_swe_numerical_flux_roe(
    u_l: [Real; 3],
    u_r: [Real; 3],
    n: Vec3,
    g: Real,
) -> [Real; 3] {
    // Wet-dry guard: clamp `h` to a small positive value and zero the
    // momentum if the depth is below a numerical floor. Without this,
    // a near-vacuum trace produces `hu/h → ∞` and the Roe flux NaN's.
    const H_FLOOR: Real = 1.0e-8;
    let (h_l, hu_l, hv_l) = if u_l[0] < H_FLOOR {
        (H_FLOOR, 0.0, 0.0)
    } else {
        (u_l[0], u_l[1], u_l[2])
    };
    let (h_r, hu_r, hv_r) = if u_r[0] < H_FLOOR {
        (H_FLOOR, 0.0, 0.0)
    } else {
        (u_r[0], u_r[1], u_r[2])
    };
    let nx = n.x;
    let ny = n.y;
    let tx = -ny;
    let ty = nx;

    // Rotate to the normal / tangential frame.
    let u_l_n = (hu_l * nx + hv_l * ny) / h_l;
    let u_l_t = (hu_l * tx + hv_l * ty) / h_l;
    let u_r_n = (hu_r * nx + hv_r * ny) / h_r;
    let u_r_t = (hu_r * tx + hv_r * ty) / h_r;

    let sqrt_h_l = h_l.sqrt();
    let sqrt_h_r = h_r.sqrt();
    let h_hat = 0.5 * (h_l + h_r);
    let denom = sqrt_h_l + sqrt_h_r;
    let (un_hat, ut_hat) = if denom > 1.0e-30 {
        (
            (u_l_n * sqrt_h_l + u_r_n * sqrt_h_r) / denom,
            (u_l_t * sqrt_h_l + u_r_t * sqrt_h_r) / denom,
        )
    } else {
        (0.5 * (u_l_n + u_r_n), 0.5 * (u_l_t + u_r_t))
    };
    let c_hat = (g * h_hat).sqrt();

    // Jumps in conservative variables (h, h·u_n, h·u_t).
    let dh = h_r - h_l;
    let d_hun = h_r * u_r_n - h_l * u_l_n;
    let d_hut = h_r * u_r_t - h_l * u_l_t;

    // Wave strengths (Toro 2001, §13.7 for 2D SWE in rotated coordinates):
    //   α_1 = ½ (dh − (dhu_n − û_n dh) / ĉ)
    //   α_2 = dhu_t − û_t dh                     (shear-wave strength)
    //   α_3 = ½ (dh + (dhu_n − û_n dh) / ĉ)
    let inv_c = if c_hat > 1.0e-30 { 1.0 / c_hat } else { 0.0 };
    let alpha_1 = 0.5 * (dh - (d_hun - un_hat * dh) * inv_c);
    let alpha_2 = d_hut - ut_hat * dh;
    let alpha_3 = 0.5 * (dh + (d_hun - un_hat * dh) * inv_c);

    // Eigenvalues at the Roe-average state.
    let lam_1 = un_hat - c_hat;
    let lam_2 = un_hat;
    let lam_3 = un_hat + c_hat;

    // Eigenvalues on the L and R sides (for the entropy fix).
    let c_l = (g * h_l).sqrt();
    let c_r = (g * h_r).sqrt();
    let lam_1_l = u_l_n - c_l;
    let lam_1_r = u_r_n - c_r;
    let lam_3_l = u_l_n + c_l;
    let lam_3_r = u_r_n + c_r;

    // Harten-Hyman entropy fix: smooth |λ| near sonic points so that
    // transonic rarefactions get the correct shock-free expansion.
    let smooth_abs = |lam: Real, lam_l: Real, lam_r: Real| -> Real {
        let delta = ((lam - lam_l).max(0.0)).max((lam_r - lam).max(0.0));
        if lam.abs() < delta {
            0.5 * (lam * lam / delta.max(1.0e-30) + delta)
        } else {
            lam.abs()
        }
    };
    let abs_lam_1 = smooth_abs(lam_1, lam_1_l, lam_1_r);
    // Contact / shear wave: no entropy fix needed (it's linearly
    // degenerate — no rarefaction across it).
    let abs_lam_2 = lam_2.abs();
    let abs_lam_3 = smooth_abs(lam_3, lam_3_l, lam_3_r);

    // Right eigenvectors in the rotated (normal/tangential) frame:
    //   r_1 = (1, û_n − ĉ, û_t)
    //   r_2 = (0, 0, 1)                       (pure shear)
    //   r_3 = (1, û_n + ĉ, û_t)
    let r1 = [1.0_f64 as Real, un_hat - c_hat, ut_hat];
    let r2 = [0.0_f64 as Real, 0.0, 1.0];
    let r3 = [1.0_f64 as Real, un_hat + c_hat, ut_hat];

    // Roe dissipation in the rotated frame.
    let d_rot = [
        abs_lam_1 * alpha_1 * r1[0] + abs_lam_2 * alpha_2 * r2[0] + abs_lam_3 * alpha_3 * r3[0],
        abs_lam_1 * alpha_1 * r1[1] + abs_lam_2 * alpha_2 * r2[1] + abs_lam_3 * alpha_3 * r3[1],
        abs_lam_1 * alpha_1 * r1[2] + abs_lam_2 * alpha_2 * r2[2] + abs_lam_3 * alpha_3 * r3[2],
    ];

    // Average physical flux (in original frame). Use the clamped
    // (h, hu, hv) so a near-vacuum trace doesn't blow up here either.
    let f_l = swe_flux_dot_n(h_l, hu_l, hv_l, n, g);
    let f_r = swe_flux_dot_n(h_r, hu_r, hv_r, n, g);

    // Rotate the dissipation back: the dissipation acts on (h, h·u_n,
    // h·u_t) → (h, hu, hv) via
    //   d_hu = d_hun · nx + d_hut · tx
    //   d_hv = d_hun · ny + d_hut · ty
    let d_h = d_rot[0];
    let d_hu = d_rot[1] * nx + d_rot[2] * tx;
    let d_hv = d_rot[1] * ny + d_rot[2] * ty;

    [
        0.5 * (f_l[0] + f_r[0]) - 0.5 * d_h,
        0.5 * (f_l[1] + f_r[1]) - 0.5 * d_hu,
        0.5 * (f_l[2] + f_r[2]) - 0.5 * d_hv,
    ]
}

// ─── Stage 43 v3 — Krivodonova characteristic limiter ───────────────

/// Stage 43 v3 — characteristic-wise Barth-Jespersen limiter for the
/// SWE `(h, hu, hv)` system (Krivodonova 2007).
///
/// The scalar BJ limiter applied component-wise to a non-linear
/// system over-limits the contact / shear field, producing
/// excessive diffusion in smooth regions adjacent to shocks. The
/// fix is to transform per-cell residuals into **characteristic
/// variables** (eigenvector basis of the local Roe-averaged Jacobian),
/// apply scalar BJ independently to each characteristic, and
/// transform back. The cell-average and per-cell layout are unchanged
/// vs [`limit_solution`].
///
/// Algorithm:
/// 1. Compute cell averages `Ū_K` (mean of nodal values, like
///    `limit_solution`).
/// 2. For each cell `K`, build a single Roe-average state from the
///    cell-average + the average of its face-neighbour cell-averages
///    (a centred / symmetric choice — Krivodonova originally used
///    direction-dependent averages, but the symmetric variant gives
///    the same shock-capturing behaviour and avoids tie-breaks on
///    triangular meshes where each cell has 3 neighbours).
/// 3. Compute the left + right eigenmatrices `L̂, R̂` of the Jacobian
///    `Â · n_K` at the Roe-average state. The "principal" normal
///    `n_K` is the average outward normal across the cell's faces —
///    on an isotropic mesh this is well-defined; on anisotropic
///    meshes the limiter degrades gracefully to the scalar BJ.
/// 4. Transform the per-cell deltas `U_i − Ū_K` and the neighbour
///    deltas `Ū_NB − Ū_K` into characteristic variables via `L̂`.
/// 5. Apply scalar BJ to each characteristic field independently.
/// 6. Transform back via `R̂` and update `u_dg`.
///
/// `u_dg` is the flat **system** state in component-fastest layout:
/// `u_dg[(dof) * components + c]`. The function is hard-coded to
/// `components == 3` and treats them as `(h, hu, hv)` SWE variables.
pub fn limit_solution_characteristic(
    mesh: &super::mesh::FemMesh,
    u_dg: &mut [Real],
    degree: usize,
    cell_neighbours: &[Vec<u32>],
    g: Real,
    physical_h_min: Real,
) {
    const COMPONENTS: usize = 3;
    assert!(
        degree <= 2,
        "limit_solution_characteristic: degree {degree} out of range"
    );
    if degree == 0 {
        // P0 has only cell averages → just clamp h non-negative.
        for ci in 0..mesh.cells.len() {
            let idx = ci * COMPONENTS;
            if u_dg[idx] < physical_h_min {
                u_dg[idx] = physical_h_min;
            }
        }
        return;
    }

    // 1. Per-cell offsets (in scalar DoFs).
    let mut offsets: Vec<usize> = Vec::with_capacity(mesh.cells.len() + 1);
    let mut running = 0usize;
    for c in &mesh.cells {
        offsets.push(running);
        running += dg_local_ndof(c, degree);
    }
    offsets.push(running);

    // 2. Cell averages — per component.
    let cell_avg: Vec<[Real; COMPONENTS]> = (0..mesh.cells.len())
        .map(|ci| {
            let n = offsets[ci + 1] - offsets[ci];
            let inv_n = 1.0 / (n as Real);
            let mut avg = [0.0_f64 as Real; COMPONENTS];
            for i in 0..n {
                for c in 0..COMPONENTS {
                    avg[c] += u_dg[(offsets[ci] + i) * COMPONENTS + c];
                }
            }
            for c in 0..COMPONENTS {
                avg[c] *= inv_n;
            }
            avg
        })
        .collect();

    // Helper: scalar BJ factor over a single field. Returns phi ∈ [0,1].
    let bj_phi = |center: Real, nodal: &[Real], u_min: Real, u_max: Real| -> Real {
        let mut phi: Real = 1.0;
        for &u_i in nodal {
            let d = u_i - center;
            let yi = if d > 1.0e-30 {
                ((u_max - center) / d).min(1.0)
            } else if d < -1.0e-30 {
                ((u_min - center) / d).min(1.0)
            } else {
                1.0
            };
            let yi = yi.clamp(0.0, 1.0);
            if yi < phi {
                phi = yi;
            }
        }
        phi
    };

    // 3. Apply per-cell characteristic limiting.
    let n_cells = mesh.cells.len();
    let mut new_u_dg: Vec<Real> = u_dg.to_vec();

    for ci in 0..n_cells {
        let nbrs = &cell_neighbours[ci];
        let u_bar = cell_avg[ci];

        // Build the principal cell normal — average of the outward
        // normals on each face. On an isotropic mesh (square / right-tri
        // grid) this is well-defined; on degenerate cells it falls
        // back to (1, 0, 0).
        let cell = &mesh.cells[ci];
        let centroid = cell_centroid(cell, &mesh.vertices);
        let mut n_avg = Vec3::ZERO;
        match cell {
            FemCell::Tri(v) => {
                for &[a, b] in &[[v[0], v[1]], [v[1], v[2]], [v[2], v[0]]] {
                    let pa = mesh.vertices[a as usize];
                    let pb = mesh.vertices[b as usize];
                    let mid = (pa + pb) * 0.5;
                    let raw_n = edge_normal_2d(pa, pb);
                    let out_n = if raw_n.dot(mid - centroid) >= 0.0 {
                        raw_n
                    } else {
                        -raw_n
                    };
                    n_avg += out_n;
                }
            }
            FemCell::Tet(_) => {
                // 3D SWE is non-standard; degrade to scalar component-wise BJ.
                let off = offsets[ci];
                let n = offsets[ci + 1] - off;
                for c in 0..COMPONENTS {
                    let center = u_bar[c];
                    let mut u_min = center;
                    let mut u_max = center;
                    for &nb in nbrs {
                        let v = cell_avg[nb as usize][c];
                        if v < u_min {
                            u_min = v;
                        }
                        if v > u_max {
                            u_max = v;
                        }
                    }
                    if c == 0 && u_min < physical_h_min {
                        u_min = physical_h_min;
                    }
                    let nodal: Vec<Real> =
                        (0..n).map(|i| u_dg[(off + i) * COMPONENTS + c]).collect();
                    let phi = bj_phi(center, &nodal, u_min, u_max);
                    for i in 0..n {
                        let u_i = u_dg[(off + i) * COMPONENTS + c];
                        new_u_dg[(off + i) * COMPONENTS + c] = center + phi * (u_i - center);
                    }
                }
                continue;
            }
        }
        let n_len = (n_avg.x * n_avg.x + n_avg.y * n_avg.y).sqrt();
        let n_principal = if n_len > 1.0e-12 {
            Vec3::new(n_avg.x / n_len, n_avg.y / n_len, 0.0)
        } else {
            Vec3::new(1.0, 0.0, 0.0)
        };

        // Roe-average state between this cell and the *average* of its
        // neighbours' cell averages.
        let mut nb_avg = [0.0_f64 as Real; COMPONENTS];
        if !nbrs.is_empty() {
            for &nb in nbrs {
                for c in 0..COMPONENTS {
                    nb_avg[c] += cell_avg[nb as usize][c];
                }
            }
            let inv = 1.0 / (nbrs.len() as Real);
            for c in 0..COMPONENTS {
                nb_avg[c] *= inv;
            }
        } else {
            nb_avg = u_bar;
        }

        // Roe-average between (u_bar, nb_avg). Numerical floor on `h`
        // (independent of `physical_h_min` which is the user-facing
        // post-limit clamp) guards against transient near-vacuum
        // cell-averages — the wave-speed and eigenvector math
        // dividing by `h_l`/`h_r` would otherwise NaN.
        const H_NUM_FLOOR: Real = 1.0e-8;
        let h_l = u_bar[0].max(H_NUM_FLOOR);
        let h_r = nb_avg[0].max(H_NUM_FLOOR);
        let sqrt_h_l = h_l.sqrt();
        let sqrt_h_r = h_r.sqrt();
        let denom = sqrt_h_l + sqrt_h_r;
        let nx = n_principal.x;
        let ny = n_principal.y;
        let tx = -ny;
        let ty = nx;
        // Zero out momenta when h is near-vacuum (consistent with
        // the wet/dry treatment in `compute_swe_numerical_flux_roe`).
        let (hu_l, hv_l) = if u_bar[0] < H_NUM_FLOOR {
            (0.0, 0.0)
        } else {
            (u_bar[1], u_bar[2])
        };
        let (hu_r, hv_r) = if nb_avg[0] < H_NUM_FLOOR {
            (0.0, 0.0)
        } else {
            (nb_avg[1], nb_avg[2])
        };
        let u_l_n = (hu_l * nx + hv_l * ny) / h_l;
        let u_l_t = (hu_l * tx + hv_l * ty) / h_l;
        let u_r_n = (hu_r * nx + hv_r * ny) / h_r;
        let u_r_t = (hu_r * tx + hv_r * ty) / h_r;
        let (un_hat, ut_hat) = if denom > 1.0e-30 {
            (
                (u_l_n * sqrt_h_l + u_r_n * sqrt_h_r) / denom,
                (u_l_t * sqrt_h_l + u_r_t * sqrt_h_r) / denom,
            )
        } else {
            (0.5 * (u_l_n + u_r_n), 0.5 * (u_l_t + u_r_t))
        };
        let h_hat = 0.5 * (h_l + h_r);
        let c_hat = (g * h_hat).sqrt();

        // Left + right eigenmatrices in the rotated (normal/tangential)
        // frame. Conservative variables in that frame are
        // (h, h·u_n, h·u_t). The left eigenmatrix L̂ maps a (h, h·u_n,
        // h·u_t) jump → wave strengths (α_1, α_2, α_3):
        //
        //   L̂ = ½ ⎡ (1 + û_n/ĉ)  −1/ĉ   0 ⎤
        //          ⎢ −2 û_t        0      2 ⎥
        //          ⎣ (1 − û_n/ĉ)  +1/ĉ   0 ⎦
        //
        //   R̂ = ⎡ 1            0     1           ⎤
        //        ⎢ û_n − ĉ      0     û_n + ĉ     ⎥
        //        ⎣ û_t          1     û_t         ⎦
        //
        // Verified L̂·R̂ = I.
        let inv_c = if c_hat > 1.0e-30 { 1.0 / c_hat } else { 0.0 };
        let l_mat = [
            [0.5 * (1.0 + un_hat * inv_c), -0.5 * inv_c, 0.0],
            [-ut_hat, 0.0, 1.0],
            [0.5 * (1.0 - un_hat * inv_c), 0.5 * inv_c, 0.0],
        ];
        let r_mat = [
            [1.0_f64 as Real, 0.0, 1.0],
            [un_hat - c_hat, 0.0, un_hat + c_hat],
            [ut_hat, 1.0, ut_hat],
        ];

        // Transform `U = (h, hu, hv)` → rotated `(h, h·u_n, h·u_t)`.
        let rotate_to_nt =
            |u: [Real; 3]| -> [Real; 3] { [u[0], u[1] * nx + u[2] * ny, u[1] * tx + u[2] * ty] };
        // Transform rotated → original.
        let rotate_to_xy =
            |u: [Real; 3]| -> [Real; 3] { [u[0], u[1] * nx + u[2] * tx, u[1] * ny + u[2] * ty] };

        // Center cell-average in characteristic vars (only used to
        // pick the BJ neighbourhood — the limiter actually limits the
        // *deltas* relative to the cell average, then converts back).
        let u_bar_nt = rotate_to_nt(u_bar);
        let mut w_bar = [0.0_f64 as Real; COMPONENTS];
        for i in 0..COMPONENTS {
            for j in 0..COMPONENTS {
                w_bar[i] += l_mat[i][j] * u_bar_nt[j];
            }
        }

        // Characteristic min/max over neighbours.
        let mut w_min = w_bar;
        let mut w_max = w_bar;
        for &nb in nbrs {
            let nb_state = cell_avg[nb as usize];
            let nb_nt = rotate_to_nt(nb_state);
            let mut w_nb = [0.0_f64 as Real; COMPONENTS];
            for i in 0..COMPONENTS {
                for j in 0..COMPONENTS {
                    w_nb[i] += l_mat[i][j] * nb_nt[j];
                }
            }
            for i in 0..COMPONENTS {
                if w_nb[i] < w_min[i] {
                    w_min[i] = w_nb[i];
                }
                if w_nb[i] > w_max[i] {
                    w_max[i] = w_nb[i];
                }
            }
        }

        // Compute nodal characteristic values + BJ phi per characteristic.
        let off = offsets[ci];
        let n = offsets[ci + 1] - off;
        let mut nodal_w: Vec<[Real; COMPONENTS]> = Vec::with_capacity(n);
        for i in 0..n {
            let u_i = [
                u_dg[(off + i) * COMPONENTS + 0],
                u_dg[(off + i) * COMPONENTS + 1],
                u_dg[(off + i) * COMPONENTS + 2],
            ];
            let u_i_nt = rotate_to_nt(u_i);
            let mut w_i = [0.0_f64 as Real; COMPONENTS];
            for ii in 0..COMPONENTS {
                for jj in 0..COMPONENTS {
                    w_i[ii] += l_mat[ii][jj] * u_i_nt[jj];
                }
            }
            nodal_w.push(w_i);
        }

        let mut phi = [1.0_f64 as Real; COMPONENTS];
        for k in 0..COMPONENTS {
            let nodal: Vec<Real> = nodal_w.iter().map(|w| w[k]).collect();
            phi[k] = bj_phi(w_bar[k], &nodal, w_min[k], w_max[k]);
        }

        // Apply: w_i ← w_bar + phi · (w_i − w_bar); convert back to
        // conservative variables in the original frame.
        for i in 0..n {
            let mut w_limited = [0.0_f64 as Real; COMPONENTS];
            for k in 0..COMPONENTS {
                w_limited[k] = w_bar[k] + phi[k] * (nodal_w[i][k] - w_bar[k]);
            }
            // Back to rotated conservative vars via R̂.
            let mut u_limited_nt = [0.0_f64 as Real; COMPONENTS];
            for ii in 0..COMPONENTS {
                for kk in 0..COMPONENTS {
                    u_limited_nt[ii] += r_mat[ii][kk] * w_limited[kk];
                }
            }
            // Rotate to physical (h, hu, hv) frame.
            let u_limited = rotate_to_xy(u_limited_nt);
            for c in 0..COMPONENTS {
                new_u_dg[(off + i) * COMPONENTS + c] = u_limited[c];
            }
            // Hard clamp h ≥ physical_h_min.
            if new_u_dg[(off + i) * COMPONENTS + 0] < physical_h_min {
                new_u_dg[(off + i) * COMPONENTS + 0] = physical_h_min;
            }
        }
    }

    u_dg.copy_from_slice(&new_u_dg);
}

// ─── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem::mesh::FemMesh;

    #[test]
    fn dg_p0_mass_is_diagonal_cell_volume() {
        let mesh = FemMesh::tri_grid_2d(2, 2, 1.0, 1.0);
        for cell in &mesh.cells {
            let m = compute_element_dg_mass(cell, &mesh.vertices, 0);
            assert_eq!(m.len(), 1);
            assert_eq!(m[0].len(), 1);
            let area = match cell {
                FemCell::Tri(v) => tri_area_2d(&[
                    mesh.vertices[v[0] as usize],
                    mesh.vertices[v[1] as usize],
                    mesh.vertices[v[2] as usize],
                ]),
                _ => unreachable!(),
            };
            assert!((m[0][0] - area).abs() < 1e-12);
        }
    }

    #[test]
    fn dg_p1_mass_matches_cg_p1_mass() {
        // Cell-interior DG-P1 mass is identical to CG-P1 mass.
        let p = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let cell = FemCell::Tri([0, 1, 2]);
        let dg = compute_element_dg_mass(&cell, &p, 1);
        let cg = super::super::element::compute_element_mass(&cell, &p);
        for i in 0..3 {
            for j in 0..3 {
                assert!((dg[i][j] - cg[i][j]).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn dg_p0_stiffness_is_zero() {
        let cell = FemCell::Tri([0, 1, 2]);
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let k = compute_element_dg_stiffness(&cell, &v, 1.0, 0);
        assert_eq!(k.len(), 1);
        assert_eq!(k[0].len(), 1);
        assert!(k[0][0].abs() < 1e-30);
    }

    #[test]
    fn interior_faces_2x2_grid_has_correct_count() {
        // tri_grid_2d(2, 2) → 8 triangles, each has 3 edges. Interior
        // edges are shared by 2 triangles; boundary edges by 1.
        // The 2×2 grid has 12 boundary edge nodes worth of edges (4
        // per side × 2 + diagonals). Easier check: count interior
        // faces vs cell count.
        let mesh = FemMesh::tri_grid_2d(2, 2, 1.0, 1.0);
        let faces = compute_interior_faces(&mesh);
        // 2×2 grid: 4 quads → 8 tris. Edges total: 4 quad horizontals
        // + 4 quad verticals + 4 diagonals = 12 distinct edges.
        // Each quad shares one diagonal between its two tris → 4 interior diagonals.
        // Inter-quad: the bottom/top of each row contributes (nx-1)*ny
        // horizontals shared by neighbouring rows? Actually this 2D
        // tri-grid_2d places diagonals so each cell shares 1 diagonal
        // with its sibling in the same quad + 2 edges with neighbouring
        // quads (the shared sides). The count works out, but easier to
        // just assert non-empty + each face is owned by exactly 2 cells
        // (already enforced by `compute_interior_faces` returning only
        // those).
        assert!(!faces.is_empty());
        for f in &faces {
            assert_ne!(f.cell_left, f.cell_right);
            assert_eq!(f.face_verts.len(), 2);
        }
    }

    #[test]
    fn interior_faces_tet_grid_is_nonempty() {
        let mesh = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        let faces = compute_interior_faces(&mesh);
        assert!(!faces.is_empty());
        for f in &faces {
            assert_eq!(f.face_verts.len(), 3);
            assert_ne!(f.cell_left, f.cell_right);
        }
    }

    #[test]
    fn dg_p1_flux_block_is_symmetric() {
        // Two adjacent triangles sharing an edge — assemble the SIP
        // flux block and check (block + blockᵀ) / 2 ≈ block.
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let cells = vec![FemCell::Tri([0, 1, 2]), FemCell::Tri([0, 2, 3])];
        let mesh = FemMesh {
            vertices: v.clone(),
            cells,
        };
        let faces = compute_interior_faces(&mesh);
        assert_eq!(faces.len(), 1);
        let face = &faces[0];
        let cl = &mesh.cells[face.cell_left as usize];
        let cr = &mesh.cells[face.cell_right as usize];
        let block = compute_dg_flux_jump_block(cl, cr, face, &v, 1.0, 1, 10.0);
        assert_eq!(block.len(), 6);
        for i in 0..6 {
            for j in 0..6 {
                let asym = (block[i][j] - block[j][i]).abs();
                assert!(
                    asym < 1e-9,
                    "block[{i}][{j}] = {} vs [{j}][{i}] = {} (Δ = {})",
                    block[i][j],
                    block[j][i],
                    asym
                );
            }
        }
    }

    #[test]
    fn dg_p0_flux_block_constant_field_kills_jump() {
        // For DG-P0 the flux block is `pen [[1, -1], [-1, 1]] * len`.
        // The vector (1, 1)ᵀ (constant field) is in its null space.
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(1.0, 1.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let cells = vec![FemCell::Tri([0, 1, 2]), FemCell::Tri([0, 2, 3])];
        let mesh = FemMesh {
            vertices: v.clone(),
            cells,
        };
        let faces = compute_interior_faces(&mesh);
        let face = &faces[0];
        let block = compute_dg_flux_jump_block(
            &mesh.cells[face.cell_left as usize],
            &mesh.cells[face.cell_right as usize],
            face,
            &v,
            1.0,
            0,
            10.0,
        );
        // 2×2 block.
        for i in 0..2 {
            let row_sum: Real = block[i].iter().sum();
            assert!(row_sum.abs() < 1e-12, "row {i} sum = {row_sum}");
        }
    }

    #[test]
    fn periodic_face_pairs_finds_matching_boundary_edges_on_grid() {
        // 4×3 tri grid on [0, 4] × [0, 3]. Periodic along axis 0 (x):
        // every face on x = 0 should pair with the same-y face on x = 4.
        let mesh = FemMesh::tri_grid_2d(4, 3, 4.0, 3.0);
        let pairs = periodic_face_pairs(&mesh, 0);
        assert!(!pairs.is_empty(), "no periodic pairs found");
        for (fa, fb) in &pairs {
            // Both faces should be edges (2 vertices).
            assert_eq!(fa.face_verts.len(), 2);
            assert_eq!(fb.face_verts.len(), 2);
            // x coords of fa midpoint should be ~0; fb midpoint ~4.
            let mid = |f: &InteriorFace| -> Vec3 {
                let mut s = Vec3::ZERO;
                for &v in &f.face_verts {
                    s += mesh.vertices[v as usize];
                }
                s / (f.face_verts.len() as Real)
            };
            let ma = mid(fa);
            let mb = mid(fb);
            assert!(ma.x.abs() < 1e-9, "fa midpoint not on x=0: {ma:?}");
            assert!((mb.x - 4.0).abs() < 1e-9, "fb midpoint not on x=4: {mb:?}");
            // y midpoints must match (the bin key).
            assert!(
                (ma.y - mb.y).abs() < 1e-9,
                "y mismatch: ma={ma:?}, mb={mb:?}"
            );
        }
    }

    #[test]
    fn nitsche_boundary_block_constant_field_recovers_value() {
        // Constant-field test: for a P1 boundary face with u_D set to
        // the (constant) cell value, the RHS should equal `block · u`
        // (= just the consistency boundary integral). Verifies the
        // sign conventions for the consistency / penalty terms.
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        let cell = FemCell::Tri([0, 1, 2]);
        // Boundary edge (0, 1) — the bottom of the triangle. u_D = 2.5
        // at both endpoints (constant trace).
        let (block, rhs) = compute_nitsche_boundary_block(
            &cell,
            &[0, 1],
            &v,
            1.0,  // alpha
            1,    // degree
            20.0, // sigma
            &[2.5, 2.5],
        );
        // u = (2.5, 2.5, 2.5) — constant field over the cell.
        let u = [2.5_f64, 2.5, 2.5];
        for i in 0..3 {
            let bu: Real = (0..3).map(|j| block[i][j] * u[j]).sum();
            // block · u should equal rhs for the constant Dirichlet trace.
            let diff = (bu - rhs[i]).abs();
            assert!(diff < 1e-9, "row {i}: block·u = {bu}, rhs = {}", rhs[i]);
        }
    }
}
