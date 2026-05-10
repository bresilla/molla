//! Local element matrices — the "weak form" visitor pattern.
//!
//! Stage 36 v1 ships two element kernels:
//!
//! - [`compute_element_mass`]      — local mass matrix `Mₑ`
//!   (consistent, `(area / 12) · M_ref` for triangles, `(vol / 20) ·
//!   M_ref` for tetrahedra).
//! - [`compute_element_stiffness`] — local Laplacian `Kₑ = α · ∫ ∇φᵢ ·
//!   ∇φⱼ dx`. P1 ⇒ gradients are constant per element ⇒ no
//!   quadrature loop needed; the integral collapses to `α · Bᵀ·B ·
//!   |Ω|` where `B` is the (D × N) matrix of shape-function
//!   gradients.
//!
//! Both kernels return a row-major dense matrix as `Vec<Vec<Real>>`
//! (`N × N` where `N` is the cell's vertex count). The choice of
//! `Vec<Vec<…>>` over a packed `Vec<Real>` keeps the v1 assembly
//! readable; Stage 36.5 (GPU port) will switch to a flat `[Real; 16]`
//! style local-matrix payload.
//!
//! Stage 36.1 will add `compute_element_elasticity_stiffness` here and
//! Stage 36.2 will add `compute_element_stokes_block` — both follow
//! the same `(cell, vertices, params) → Vec<Vec<Real>>` shape.

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
pub(super) fn tri_shape_grads_2d(p: &[Vec3; 3]) -> [[Real; 2]; 3] {
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
pub(super) fn tet_shape_grads(p: &[Vec3; 4]) -> [Vec3; 4] {
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
