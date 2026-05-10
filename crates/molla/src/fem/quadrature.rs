//! Gauss quadrature rules for the v1 cell types.
//!
//! Each rule returns barycentric points + weights pre-normalised so
//! `∑ wᵢ = 1`. The assembly multiplies by the cell's measure (area
//! for triangles, signed volume for tets) so the rule itself is
//! "shape-agnostic".
//!
//! P1 elements have constant gradients ⇒ stiffness assembly is exact
//! with the 1-point rule. The 3-point (tri) / 4-point (tet) rules are
//! provided for users that integrate higher-order quantities (e.g.
//! L²-projected source terms) on the same elements.
//!
//! Rules used in v1:
//! - [`tri_quadrature_1pt`] — barycentric `(1/3, 1/3, 1/3)`, weight 1.
//! - [`tri_quadrature_3pt`] — three edge midpoints, weight 1/3 each.
//! - [`tet_quadrature_1pt`] — barycentric `(1/4, 1/4, 1/4, 1/4)`,
//!   weight 1.
//! - [`tet_quadrature_4pt`] — four corners, weight 1/4 each.
//!
//! Higher-order rules (5-point tri, 11-point tet, etc.) live in
//! Stage 36.3 alongside P2 / P3 elements.

use crate::math::Real;

/// One quadrature point on a triangle, in barycentric coordinates
/// `(λ₀, λ₁, λ₂)` plus a weight.
#[derive(Copy, Clone, Debug)]
pub struct TriQuadPoint {
    pub bary: [Real; 3],
    pub weight: Real,
}

/// One quadrature point on a tetrahedron, in barycentric coordinates
/// `(λ₀, λ₁, λ₂, λ₃)` plus a weight.
#[derive(Copy, Clone, Debug)]
pub struct TetQuadPoint {
    pub bary: [Real; 4],
    pub weight: Real,
}

/// Single-point Gauss rule for a triangle (centroid).
/// Exact for polynomials up to degree 1.
#[must_use]
pub fn tri_quadrature_1pt() -> Vec<TriQuadPoint> {
    vec![TriQuadPoint {
        bary: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        weight: 1.0,
    }]
}

/// Three-point Gauss rule for a triangle (edge midpoints).
/// Exact for polynomials up to degree 2.
#[must_use]
pub fn tri_quadrature_3pt() -> Vec<TriQuadPoint> {
    let w = 1.0 / 3.0;
    vec![
        TriQuadPoint { bary: [0.5, 0.5, 0.0], weight: w },
        TriQuadPoint { bary: [0.0, 0.5, 0.5], weight: w },
        TriQuadPoint { bary: [0.5, 0.0, 0.5], weight: w },
    ]
}

/// Single-point Gauss rule for a tetrahedron (centroid).
/// Exact for polynomials up to degree 1.
#[must_use]
pub fn tet_quadrature_1pt() -> Vec<TetQuadPoint> {
    vec![TetQuadPoint {
        bary: [0.25, 0.25, 0.25, 0.25],
        weight: 1.0,
    }]
}

/// Four-point Gauss rule for a tetrahedron (vertex sampling).
/// Each vertex contributes weight 1/4. Exact for polynomials up to
/// degree 1; sufficient for diagonal "lumped mass" experiments.
#[must_use]
pub fn tet_quadrature_4pt() -> Vec<TetQuadPoint> {
    let w = 0.25;
    vec![
        TetQuadPoint { bary: [1.0, 0.0, 0.0, 0.0], weight: w },
        TetQuadPoint { bary: [0.0, 1.0, 0.0, 0.0], weight: w },
        TetQuadPoint { bary: [0.0, 0.0, 1.0, 0.0], weight: w },
        TetQuadPoint { bary: [0.0, 0.0, 0.0, 1.0], weight: w },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: Real, b: Real) -> bool {
        (a - b).abs() < 1.0e-12
    }

    #[test]
    fn tri_rules_have_unit_weight_sum() {
        let s1: Real = tri_quadrature_1pt().iter().map(|p| p.weight).sum();
        let s3: Real = tri_quadrature_3pt().iter().map(|p| p.weight).sum();
        assert!(approx_eq(s1, 1.0));
        assert!(approx_eq(s3, 1.0));
    }

    #[test]
    fn tet_rules_have_unit_weight_sum() {
        let s1: Real = tet_quadrature_1pt().iter().map(|p| p.weight).sum();
        let s4: Real = tet_quadrature_4pt().iter().map(|p| p.weight).sum();
        assert!(approx_eq(s1, 1.0));
        assert!(approx_eq(s4, 1.0));
    }

    #[test]
    fn bary_coords_partition_unity() {
        for p in tri_quadrature_1pt() {
            let s: Real = p.bary.iter().sum();
            assert!(approx_eq(s, 1.0));
        }
        for p in tri_quadrature_3pt() {
            let s: Real = p.bary.iter().sum();
            assert!(approx_eq(s, 1.0));
        }
        for p in tet_quadrature_1pt() {
            let s: Real = p.bary.iter().sum();
            assert!(approx_eq(s, 1.0));
        }
        for p in tet_quadrature_4pt() {
            let s: Real = p.bary.iter().sum();
            assert!(approx_eq(s, 1.0));
        }
    }
}
