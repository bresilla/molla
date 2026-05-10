//! Local element matrices — the "weak form" visitor pattern.
//!
//! Stage 36 v1 shipped two scalar kernels:
//!
//! - [`compute_element_mass`]      — local mass matrix `Mₑ`
//!   (consistent, `(area / 12) · M_ref` for triangles, `(vol / 20) ·
//!   M_ref` for tetrahedra).
//! - [`compute_element_stiffness`] — local Laplacian `Kₑ = α · ∫ ∇φᵢ ·
//!   ∇φⱼ dx`. P1 ⇒ gradients are constant per element ⇒ no
//!   quadrature loop needed; the integral collapses to `α · Bᵀ·B ·
//!   |Ω|`.
//!
//! Stage 36.1 adds two **vector-valued** kernels:
//!
//! - [`compute_element_elasticity_stiffness`] — linear isotropic
//!   elasticity `Kₑ = ∫ Bᵀ D B dx`. Returns a `(N · dim) × (N · dim)`
//!   block where each row/col groups the per-vertex displacement
//!   components in **interleaved** layout: `[ux₀, uy₀, uz₀, ux₁,
//!   uy₁, uz₁, …]`. Use parameters `λ, μ` (Lamé coefficients).
//! - [`compute_element_st_venant_kirchhoff_stress`] — geometrically
//!   nonlinear elastic force from the Green strain `E = ½(FᵀF − I)`
//!   and 2nd Piola-Kirchhoff stress `S = λ tr(E) I + 2 μ E`.
//!
//! All kernels return a row-major dense matrix as `Vec<Vec<Real>>`.
//! The choice of `Vec<Vec<…>>` over a packed `Vec<Real>` keeps the
//! v1 assembly readable; Stage 36.5 (GPU port) will switch to a flat
//! `[Real; 144]` style local-matrix payload.
//!
//! Stage 36.2 will add `compute_element_stokes_block` (Taylor-Hood
//! velocity-pressure block) under the same shape.

use crate::math::{Real, Vec3};

use super::mesh::FemCell;

/// Triangle reference mass matrix (before scaling by `area / 12`).
const TRI_MASS_REF: [[Real; 3]; 3] = [
    [2.0, 1.0, 1.0],
    [1.0, 2.0, 1.0],
    [1.0, 1.0, 2.0],
];

/// Tetrahedron reference mass matrix (before scaling by `vol / 20`).
/// Diagonal entries are 2, off-diagonals are 1 — the standard
/// consistent-mass coefficients (see Hughes, "The Finite Element
/// Method", 1987, eq. 4.5.13).
const TET_MASS_REF: [[Real; 4]; 4] = [
    [2.0, 1.0, 1.0, 1.0],
    [1.0, 2.0, 1.0, 1.0],
    [1.0, 1.0, 2.0, 1.0],
    [1.0, 1.0, 1.0, 2.0],
];

/// Per-cell mass matrix (consistent, P1 nodal). Returns an `N × N`
/// matrix as `Vec<Vec<Real>>` where `N = 3` (Tri) or `N = 4` (Tet).
///
/// Triangles use the `(x, y)` projection of the vertex coordinates;
/// the embedding `z` is ignored. Tets use the full 3D positions.
#[must_use]
pub fn compute_element_mass(cell: &FemCell, vertices: &[Vec3]) -> Vec<Vec<Real>> {
    match cell {
        FemCell::Tri(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
            ];
            let area = tri_area_2d(&p);
            let scale = area / 12.0;
            (0..3)
                .map(|i| (0..3).map(|j| scale * TRI_MASS_REF[i][j]).collect())
                .collect()
        }
        FemCell::Tet(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
                vertices[idx[3] as usize],
            ];
            let vol = tet_volume(&p);
            let scale = vol / 20.0;
            (0..4)
                .map(|i| (0..4).map(|j| scale * TET_MASS_REF[i][j]).collect())
                .collect()
        }
    }
}

/// Per-cell stiffness matrix `Kₑ = coeff · ∫ ∇φᵢ · ∇φⱼ dx`.
/// `coeff` lets the caller scale by a thermal diffusivity / viscosity
/// without rebuilding a custom kernel. P1 ⇒ shape-function gradients
/// are piecewise-constant, so the integral collapses to `coeff · Bᵀ·B
/// · |Ω|`.
///
/// Triangle elements project to the `(x, y)` plane; tetrahedral
/// elements use the full 3D vertex positions.
#[must_use]
pub fn compute_element_stiffness(
    cell: &FemCell,
    vertices: &[Vec3],
    coeff: Real,
) -> Vec<Vec<Real>> {
    match cell {
        FemCell::Tri(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
            ];
            let area = tri_area_2d(&p);
            let grads = tri_shape_grads_2d(&p);
            // Kᵢⱼ = coeff · area · ∇φᵢ · ∇φⱼ (2D dot product).
            (0..3)
                .map(|i| {
                    (0..3)
                        .map(|j| coeff * area * (grads[i][0] * grads[j][0] + grads[i][1] * grads[j][1]))
                        .collect()
                })
                .collect()
        }
        FemCell::Tet(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
                vertices[idx[3] as usize],
            ];
            let vol = tet_volume(&p);
            let grads = tet_shape_grads(&p);
            (0..4)
                .map(|i| {
                    (0..4)
                        .map(|j| coeff * vol * grads[i].dot(grads[j]))
                        .collect()
                })
                .collect()
        }
    }
}

// ─── Geometric helpers ───────────────────────────────────────────────

/// 2D signed area of a triangle (positive when CCW), computed from
/// the `(x, y)` projection of the input 3D points.
#[must_use]
pub fn tri_area_2d(p: &[Vec3; 3]) -> Real {
    let (x1, y1) = (p[0].x, p[0].y);
    let (x2, y2) = (p[1].x, p[1].y);
    let (x3, y3) = (p[2].x, p[2].y);
    0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)).abs()
}

/// Signed volume of a tetrahedron (positive when (b-a, c-a, d-a) is
/// right-handed). v1 takes the absolute value to be robust to
/// orientation; element matrices only depend on `|vol|`.
#[must_use]
pub fn tet_volume(p: &[Vec3; 4]) -> Real {
    let v = (p[1] - p[0]).dot((p[2] - p[0]).cross(p[3] - p[0])) / 6.0;
    v.abs()
}

/// Gradients of the three P1 hat-functions on a 2D triangle.
/// Returns `[∇φ₀, ∇φ₁, ∇φ₂]` as `[gx, gy]` pairs (only the in-plane
/// components — the normal-out-of-plane component is zero by
/// construction for a planar triangle).
fn tri_shape_grads_2d(p: &[Vec3; 3]) -> [[Real; 2]; 3] {
    // φᵢ(x, y) = (aᵢ + bᵢ·x + cᵢ·y) / (2A); ∇φᵢ = (bᵢ, cᵢ) / (2A).
    // Standard expressions (see Zienkiewicz):
    //   bᵢ = y_{i+1} − y_{i+2}
    //   cᵢ = x_{i+2} − x_{i+1}
    let (x0, y0) = (p[0].x, p[0].y);
    let (x1, y1) = (p[1].x, p[1].y);
    let (x2, y2) = (p[2].x, p[2].y);
    let two_area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
    let inv = 1.0 / two_area;
    [
        [(y1 - y2) * inv, (x2 - x1) * inv],
        [(y2 - y0) * inv, (x0 - x2) * inv],
        [(y0 - y1) * inv, (x1 - x0) * inv],
    ]
}

/// Gradients of the four P1 hat-functions on a tetrahedron, returned
/// as 3D vectors. Standard formula: solve `[∇φ₁, ∇φ₂, ∇φ₃] = D⁻ᵀ`
/// where `D = [v₁−v₀ | v₂−v₀ | v₃−v₀]`, and `∇φ₀ = −∑ᵢ₌₁ ∇φᵢ`.
fn tet_shape_grads(p: &[Vec3; 4]) -> [Vec3; 4] {
    use crate::math::Mat3;
    let d = Mat3::from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0]);
    // Inverse-transpose.
    let inv = d.inverse();
    let inv_t = inv.transpose();
    let g1 = inv_t.x_axis;
    let g2 = inv_t.y_axis;
    let g3 = inv_t.z_axis;
    let g0 = -(g1 + g2 + g3);
    [g0, g1, g2, g3]
}

// ─── Linear isotropic elasticity ─────────────────────────────────────

/// Per-cell stiffness matrix for linear isotropic elasticity:
/// `Kₑ = ∫ Bᵀ D B dV`, where `B` is the strain-displacement matrix
/// (Voigt notation) and `D` is the elasticity tensor in matrix form.
///
/// Returns a `(N · dim) × (N · dim)` row-major matrix in **interleaved**
/// layout: row `i * dim + c` corresponds to local vertex `i`'s
/// component `c`. For a tetrahedron, that's a 12×12 matrix; for a
/// triangle (plane strain, `dim = 2`), it's a 6×6 matrix.
///
/// Lamé parameters: `lambda = E ν / ((1 + ν)(1 − 2ν))`,
/// `mu = E / (2(1 + ν))`.
///
/// Triangle elements use plane-strain: the out-of-plane strain
/// `εzz` is held at zero and the in-plane part of `D` becomes
/// `λ + 2μ` on the diagonal (xx, yy), `λ` off-diagonal, with shear
/// modulus `μ` in the (3,3) slot.
#[must_use]
pub fn compute_element_elasticity_stiffness(
    cell: &FemCell,
    vertices: &[Vec3],
    lambda: Real,
    mu: Real,
) -> Vec<Vec<Real>> {
    match cell {
        FemCell::Tri(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
            ];
            let area = tri_area_2d(&p);
            let grads = tri_shape_grads_2d(&p);
            // 2D plane-strain D matrix (3×3 in Voigt: [εxx, εyy, 2εxy]).
            let d11 = lambda + 2.0 * mu;
            let d12 = lambda;
            let d33 = mu;
            // Build B (3 × 6): for each node i, the columns
            // (i*2, i*2+1) are
            //   [ ∂φᵢ/∂x , 0       ]
            //   [ 0       , ∂φᵢ/∂y ]
            //   [ ∂φᵢ/∂y , ∂φᵢ/∂x ]
            let n_dof = 3 * 2;
            let mut b = vec![vec![0.0; n_dof]; 3];
            for i in 0..3 {
                let gx = grads[i][0];
                let gy = grads[i][1];
                b[0][2 * i] = gx;
                b[1][2 * i + 1] = gy;
                b[2][2 * i] = gy;
                b[2][2 * i + 1] = gx;
            }
            // K = (Bᵀ D B) · area. D is diagonal-with-d12-coupling on
            // the (0,1) (1,0) slots; everything else is diagonal.
            // Compute D·B first (3 × n_dof), then Bᵀ · (D·B).
            let mut db = vec![vec![0.0; n_dof]; 3];
            for c in 0..n_dof {
                db[0][c] = d11 * b[0][c] + d12 * b[1][c];
                db[1][c] = d12 * b[0][c] + d11 * b[1][c];
                db[2][c] = d33 * b[2][c];
            }
            let mut k = vec![vec![0.0; n_dof]; n_dof];
            for r in 0..n_dof {
                for c in 0..n_dof {
                    let mut s = 0.0;
                    for q in 0..3 {
                        s += b[q][r] * db[q][c];
                    }
                    k[r][c] = s * area;
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
            let vol = tet_volume(&p);
            let grads = tet_shape_grads(&p);
            // 3D D matrix in Voigt order [εxx, εyy, εzz, γyz, γxz, γxy]
            // — 6×6, but only the leading 3×3 (axial) and trailing 3×3
            // (shear, all = μ on the diagonal) are non-trivial.
            let d11 = lambda + 2.0 * mu;
            let d12 = lambda;
            // Build B (6 × 12): for each node i (cols i*3, i*3+1, i*3+2)
            //   [ ∂x, 0 , 0  ]
            //   [ 0 , ∂y, 0  ]
            //   [ 0 , 0 , ∂z ]
            //   [ 0 , ∂z, ∂y ]
            //   [ ∂z, 0 , ∂x ]
            //   [ ∂y, ∂x, 0  ]
            let n_dof = 4 * 3;
            let mut b = vec![vec![0.0; n_dof]; 6];
            for i in 0..4 {
                let gx = grads[i].x;
                let gy = grads[i].y;
                let gz = grads[i].z;
                let cx = 3 * i;
                let cy = 3 * i + 1;
                let cz = 3 * i + 2;
                b[0][cx] = gx;
                b[1][cy] = gy;
                b[2][cz] = gz;
                b[3][cy] = gz;
                b[3][cz] = gy;
                b[4][cx] = gz;
                b[4][cz] = gx;
                b[5][cx] = gy;
                b[5][cy] = gx;
            }
            // D · B
            let mut db = vec![vec![0.0; n_dof]; 6];
            for c in 0..n_dof {
                let exx = b[0][c];
                let eyy = b[1][c];
                let ezz = b[2][c];
                db[0][c] = d11 * exx + d12 * eyy + d12 * ezz;
                db[1][c] = d12 * exx + d11 * eyy + d12 * ezz;
                db[2][c] = d12 * exx + d12 * eyy + d11 * ezz;
                db[3][c] = mu * b[3][c];
                db[4][c] = mu * b[4][c];
                db[5][c] = mu * b[5][c];
            }
            // K = vol · Bᵀ · D · B
            let mut k = vec![vec![0.0; n_dof]; n_dof];
            for r in 0..n_dof {
                for c in 0..n_dof {
                    let mut s = 0.0;
                    for q in 0..6 {
                        s += b[q][r] * db[q][c];
                    }
                    k[r][c] = s * vol;
                }
            }
            k
        }
    }
}

/// Per-cell internal nodal force from the **St. Venant-Kirchhoff
/// (SVK)** hyperelastic model, i.e. Green strain
/// `E = ½(Fᵀ F − I)` and 2nd Piola-Kirchhoff stress
/// `S = λ tr(E) I + 2μ E`. The first Piola stress is `P = F · S`,
/// and the nodal force on vertex `a` is
/// `fₐ = − vol · P · ∇φₐ`. (Sign convention: positive force is what
/// the material pushes back when displaced — matches the linear
/// stiffness `K · u` in the small-strain limit.)
///
/// `u_local` is the **interleaved** displacement vector:
/// `[ux₀, uy₀, uz₀, ux₁, …]`. For a tet, `u_local.len() == 12`.
///
/// Tris are projected to (x, y); we treat the third coordinate as
/// rigid (the small-strain plane-strain result reduces to the 2D
/// linear K matrix). This is "v1 SVK" — no plane-strain
/// regularisation in the SVK case.
#[must_use]
pub fn compute_element_st_venant_kirchhoff_stress(
    cell: &FemCell,
    vertices: &[Vec3],
    u_local: &[Real],
    lambda: Real,
    mu: Real,
) -> Vec<Real> {
    match cell {
        FemCell::Tet(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
                vertices[idx[3] as usize],
            ];
            assert_eq!(u_local.len(), 12, "SVK: u_local must be 12-component");
            let vol = tet_volume(&p);
            let grads = tet_shape_grads(&p);

            // Deformation gradient F = ∂x/∂X = I + ∑ᵢ uᵢ ⊗ ∇φᵢ.
            use crate::math::Mat3;
            let mut f = Mat3::IDENTITY;
            for i in 0..4 {
                let ux = u_local[3 * i];
                let uy = u_local[3 * i + 1];
                let uz = u_local[3 * i + 2];
                let g = grads[i];
                // f_jk += u_j * g_k.
                f.x_axis.x += ux * g.x;
                f.y_axis.x += ux * g.y;
                f.z_axis.x += ux * g.z;
                f.x_axis.y += uy * g.x;
                f.y_axis.y += uy * g.y;
                f.z_axis.y += uy * g.z;
                f.x_axis.z += uz * g.x;
                f.y_axis.z += uz * g.y;
                f.z_axis.z += uz * g.z;
            }
            // Green strain E = ½(FᵀF − I).
            let ftf = f.transpose() * f;
            let e = Mat3::from_cols(
                Vec3::new(
                    0.5 * (ftf.x_axis.x - 1.0),
                    0.5 * ftf.x_axis.y,
                    0.5 * ftf.x_axis.z,
                ),
                Vec3::new(
                    0.5 * ftf.y_axis.x,
                    0.5 * (ftf.y_axis.y - 1.0),
                    0.5 * ftf.y_axis.z,
                ),
                Vec3::new(
                    0.5 * ftf.z_axis.x,
                    0.5 * ftf.z_axis.y,
                    0.5 * (ftf.z_axis.z - 1.0),
                ),
            );
            // S = λ tr(E) I + 2 μ E.
            let tr_e = e.x_axis.x + e.y_axis.y + e.z_axis.z;
            let mut s = e * (2.0 * mu);
            s.x_axis.x += lambda * tr_e;
            s.y_axis.y += lambda * tr_e;
            s.z_axis.z += lambda * tr_e;
            // First Piola P = F · S.
            let p_first = f * s;
            // Nodal forces: f_a = vol · P · ∇φ_a (returned as positive
            // "internal force"; for residual-form Newton-Raphson the
            // caller flips sign — see solver::solve_nonlinear_static).
            let mut out = vec![0.0; 12];
            for i in 0..4 {
                let g = grads[i];
                // P · g: column-major glam.
                let fx = p_first.x_axis.x * g.x + p_first.y_axis.x * g.y + p_first.z_axis.x * g.z;
                let fy = p_first.x_axis.y * g.x + p_first.y_axis.y * g.y + p_first.z_axis.y * g.z;
                let fz = p_first.x_axis.z * g.x + p_first.y_axis.z * g.y + p_first.z_axis.z * g.z;
                out[3 * i] = vol * fx;
                out[3 * i + 1] = vol * fy;
                out[3 * i + 2] = vol * fz;
            }
            out
        }
        FemCell::Tri(_) => {
            // 2D plane-strain SVK is intentionally deferred — Stage
            // 36.1 cantilever uses the linear element for tris and SVK
            // for tets only (the bonus large-deformation test runs in
            // 3D). Return zeros so the assembly visitor signature
            // stays stable.
            vec![0.0; cell.vert_count() * 2]
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;

    fn approx_eq(a: Real, b: Real, eps: Real) -> bool {
        (a - b).abs() <= eps
    }

    #[test]
    fn unit_triangle_area_is_one_half() {
        let p = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
        ];
        assert!(approx_eq(tri_area_2d(&p), 0.5, 1e-12));
    }

    #[test]
    fn unit_tet_volume_is_one_sixth() {
        let p = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        assert!(approx_eq(tet_volume(&p), 1.0 / 6.0, 1e-12));
    }

    #[test]
    fn p1_tri_shape_grad_partition_of_zero() {
        // ∑ ∇φᵢ ≡ 0 (because ∑ φᵢ ≡ 1).
        let p = [
            Vec3::new(0.1, 0.2, 0.0),
            Vec3::new(1.3, 0.0, 0.0),
            Vec3::new(0.4, 1.7, 0.0),
        ];
        let g = tri_shape_grads_2d(&p);
        let sx = g[0][0] + g[1][0] + g[2][0];
        let sy = g[0][1] + g[1][1] + g[2][1];
        assert!(sx.abs() < 1e-12);
        assert!(sy.abs() < 1e-12);
    }

    #[test]
    fn p1_tet_shape_grad_partition_of_zero() {
        let p = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let g = tet_shape_grads(&p);
        let s = g[0] + g[1] + g[2] + g[3];
        assert!(s.length() < 1e-12);
    }

    #[test]
    fn unit_tet_stiffness_kills_constant_field() {
        // K · 1 = 0 because Kᵢⱼ ∝ (∇φᵢ · ∇φⱼ) and ∑ ∇φⱼ = 0.
        let cell = FemCell::Tet([0, 1, 2, 3]);
        let v = vec![
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ];
        let k = compute_element_stiffness(&cell, &v, 1.0);
        let one = vec![1.0, 1.0, 1.0, 1.0];
        for i in 0..4 {
            let s: Real = (0..4).map(|j| k[i][j] * one[j]).sum();
            assert!(s.abs() < 1e-12, "row {i}: K·1 = {s}");
        }
    }
}
