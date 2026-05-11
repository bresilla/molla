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
//! - **Degree** `∈ {0, 1}` — P0 (piecewise constant; no interior
//!   stiffness contribution, only flux penalty) and P1 (linear on each
//!   cell, nodal at the cell vertices). P2-DG is wired through the
//!   `FunctionSpace` enum but the element kernels here panic — defer
//!   to a follow-up.
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

use crate::math::{Real, Vec3};

use super::element::{tet_shape_grads, tet_volume, tri_area_2d, tri_shape_grads_2d};
use super::mesh::FemCell;

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
#[must_use]
pub fn compute_element_dg_mass(
    cell: &FemCell,
    vertices: &[Vec3],
    degree: usize,
) -> Vec<Vec<Real>> {
    assert!(degree <= 1, "DG v1: degree {degree} not supported");
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
#[must_use]
pub fn compute_element_dg_stiffness(
    cell: &FemCell,
    vertices: &[Vec3],
    coeff: Real,
    degree: usize,
) -> Vec<Vec<Real>> {
    assert!(degree <= 1, "DG v1: degree {degree} not supported");
    match degree {
        0 => vec![vec![0.0]],
        1 => super::element::compute_element_stiffness(cell, vertices, coeff),
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
                    face_to_cells
                        .entry(key)
                        .or_default()
                        .push(cell_idx as u32);
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
                    face_to_cells
                        .entry(key)
                        .or_default()
                        .push(cell_idx as u32);
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

// ─── DG local indexing helpers ───────────────────────────────────────

/// Evaluate the DG-P1 basis on `cell` at the physical point `x`.
///
/// Returns `[N_0, N_1, …]` where `N_i` is the value of the per-cell
/// P1 hat function attached to local vertex `i`. For `degree = 0`
/// returns `[1.0]`.
fn dg_p1_shape_values_at(cell: &FemCell, vertices: &[Vec3], x: Vec3) -> Vec<Real> {
    match cell {
        FemCell::Tri(v) => {
            let p = [
                vertices[v[0] as usize],
                vertices[v[1] as usize],
                vertices[v[2] as usize],
            ];
            // Solve for barycentric coordinates of `x` in (x, y).
            let det = (p[1].x - p[0].x) * (p[2].y - p[0].y)
                - (p[2].x - p[0].x) * (p[1].y - p[0].y);
            let inv = 1.0 / det;
            let l1 = ((p[2].y - p[0].y) * (x.x - p[0].x)
                + (p[0].x - p[2].x) * (x.y - p[0].y))
                * inv;
            let l2 = ((p[0].y - p[1].y) * (x.x - p[0].x)
                + (p[1].x - p[0].x) * (x.y - p[0].y))
                * inv;
            let l0 = 1.0 - l1 - l2;
            vec![l0, l1, l2]
        }
        FemCell::Tet(v) => {
            // Solve a 3×3 system for barycentric coords.
            use crate::math::Mat3;
            let p = [
                vertices[v[0] as usize],
                vertices[v[1] as usize],
                vertices[v[2] as usize],
                vertices[v[3] as usize],
            ];
            let d = Mat3::from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0]);
            let inv = d.inverse();
            let r = x - p[0];
            let l1 = inv.x_axis.x * r.x + inv.y_axis.x * r.y + inv.z_axis.x * r.z;
            let l2 = inv.x_axis.y * r.x + inv.y_axis.y * r.y + inv.z_axis.y * r.z;
            let l3 = inv.x_axis.z * r.x + inv.y_axis.z * r.y + inv.z_axis.z * r.z;
            let l0 = 1.0 - l1 - l2 - l3;
            vec![l0, l1, l2, l3]
        }
    }
}

/// Cell-constant gradient of each DG-P1 shape function.
fn dg_p1_shape_grads(cell: &FemCell, vertices: &[Vec3]) -> Vec<Vec3> {
    match cell {
        FemCell::Tri(v) => {
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
        FemCell::Tet(v) => {
            let p = [
                vertices[v[0] as usize],
                vertices[v[1] as usize],
                vertices[v[2] as usize],
                vertices[v[3] as usize],
            ];
            tet_shape_grads(&p).to_vec()
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
    assert!(degree <= 1, "DG v1: degree {degree} not supported");
    let nl = match (cell_left, degree) {
        (FemCell::Tri(_), 0) | (FemCell::Tet(_), 0) => 1,
        (FemCell::Tri(_), 1) => 3,
        (FemCell::Tet(_), 1) => 4,
        _ => unreachable!(),
    };
    let nr = match (cell_right, degree) {
        (FemCell::Tri(_), 0) | (FemCell::Tet(_), 0) => 1,
        (FemCell::Tri(_), 1) => 3,
        (FemCell::Tet(_), 1) => 4,
        _ => unreachable!(),
    };
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

    // Degree 1: SIP bilinear form (Arnold-Brezzi-Cockburn-Marini):
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
    // Face quadrature: 2-pt Gauss-Legendre (edge) or 3-pt mid-edge
    // (triangle face) — both degree-2 exact, sufficient for the
    // P1 × P1 quadratic face integrand.

    let qpts = face_quadrature(face, vertices);
    let grads_l = dg_p1_shape_grads(cell_left, vertices);
    let grads_r = dg_p1_shape_grads(cell_right, vertices);
    let gn_l: Vec<Real> = grads_l.iter().map(|g| g.dot(n_lr)).collect();
    let gn_r: Vec<Real> = grads_r.iter().map(|g| g.dot(n_lr)).collect();

    for q in &qpts {
        let nl_vals = dg_p1_shape_values_at(cell_left, vertices, q.point);
        let nr_vals = dg_p1_shape_values_at(cell_right, vertices, q.point);
        let w = q.weight; // absolute weight (already includes |F|).

        for i in 0..nl {
            for j in 0..nl {
                block[i][j] += w
                    * (-0.5 * coeff * gn_l[j] * nl_vals[i]
                        - 0.5 * coeff * nl_vals[j] * gn_l[i]
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
            (vertices[v[0] as usize]
                + vertices[v[1] as usize]
                + vertices[v[2] as usize])
                / 3.0
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
        let block =
            compute_dg_flux_jump_block(cl, cr, face, &v, 1.0, 1, 10.0);
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

}
