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
//! Stage 36.3 (P2 nodal) adds higher-order rules:
//! - [`tri_quadrature_6pt`]  — degree-4 exact, for P2 stiffness/mass.
//! - [`tet_quadrature_5pt`]  — degree-3 exact (Keast P5 (5)).
//! - [`tet_quadrature_11pt`] — degree-4 exact (Keast P11 (4)).

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

/// Six-point Gauss rule for a triangle (Dunavant, degree-4 exact).
///
/// Standard "two-orbit" rule: three points clustered near the
/// vertices (`a₁ ≈ 0.8168, b₁ ≈ 0.0916`, weight `w₁ ≈ 0.1100`) and
/// three points near the edge midpoints (`a₂ ≈ 0.1081, b₂ ≈ 0.4459`,
/// weight `w₂ ≈ 0.2234`). Weights pre-normalised to sum to 1.
///
/// Used by the Stage 36.3 P2 element matrices: P2 stiffness
/// integrand `∇φᵢ · ∇φⱼ` is degree 2 (gradients are linear) and the
/// mass integrand `φᵢ · φⱼ` is degree 4 — both exactly captured here.
#[must_use]
pub fn tri_quadrature_6pt() -> Vec<TriQuadPoint> {
    // Dunavant 1985 degree-4 rule, in barycentric coordinates.
    let a1 = 0.816_847_572_980_459;
    let b1 = 0.091_576_213_509_771;
    let w1 = 0.109_951_743_655_322;
    let a2 = 0.108_103_018_168_070;
    let b2 = 0.445_948_490_915_965;
    let w2 = 0.223_381_589_678_011;
    vec![
        TriQuadPoint { bary: [a1, b1, b1], weight: w1 },
        TriQuadPoint { bary: [b1, a1, b1], weight: w1 },
        TriQuadPoint { bary: [b1, b1, a1], weight: w1 },
        TriQuadPoint { bary: [a2, b2, b2], weight: w2 },
        TriQuadPoint { bary: [b2, a2, b2], weight: w2 },
        TriQuadPoint { bary: [b2, b2, a2], weight: w2 },
    ]
}

/// Five-point Gauss rule for a tetrahedron (Keast, degree-3 exact).
///
/// One negative-weight centroid point (`w = -4/5`) plus four
/// equally-weighted off-centre points (`w = 9/20`) at
/// `(1/2, 1/6, 1/6, 1/6)` and cyclic permutations. Weights sum to 1.
///
/// Sufficient for P2 tet stiffness (`∇φᵢ · ∇φⱼ` is degree 2). Use
/// [`tet_quadrature_11pt`] when integrating P2 mass (degree-4
/// integrand) is required to high accuracy.
#[must_use]
pub fn tet_quadrature_5pt() -> Vec<TetQuadPoint> {
    let q = 1.0 / 6.0;
    let p = 1.0 / 2.0;
    let w_corner = 9.0 / 20.0;
    let w_centre = -4.0 / 5.0;
    vec![
        TetQuadPoint { bary: [0.25, 0.25, 0.25, 0.25], weight: w_centre },
        TetQuadPoint { bary: [p, q, q, q], weight: w_corner },
        TetQuadPoint { bary: [q, p, q, q], weight: w_corner },
        TetQuadPoint { bary: [q, q, p, q], weight: w_corner },
        TetQuadPoint { bary: [q, q, q, p], weight: w_corner },
    ]
}

/// 11-point Gauss rule for a tetrahedron (Keast, degree-4 exact).
///
/// Combines:
/// - 1 centroid point with negative weight `w₁ = -148/1875`,
/// - 4 "edge-cluster" points at `(a, a, a, 1−3a)` permutations
///   (`a = 1/14`), weight `w₂ = 343/7500`,
/// - 6 "face-cluster" points at `(b, b, c, c)` permutations with
///   `b = (5−√5)/20`, `c = (5+3√5)/20`, weight `w₃ = 56/375`.
///
/// Weights sum to 1 (post-normalisation by `6 × volume_ref`).
/// Captures P2 mass integrals on tetrahedra exactly.
#[must_use]
pub fn tet_quadrature_11pt() -> Vec<TetQuadPoint> {
    // All weights pre-normalised so ∑ wᵢ = 1.
    let w1 = -148.0 / 1875.0;
    let w2 = 343.0 / 7500.0;
    let w3 = 56.0 / 375.0;

    let a = 1.0 / 14.0;
    let aa = 11.0 / 14.0; // 1 − 3a

    // Keast-tabulated values for the (b, b, c, c) orbit; b + c = 1/2.
    let b = 0.399_403_576_166_799;
    let c = 0.100_596_423_833_201;

    vec![
        // Centroid.
        TetQuadPoint { bary: [0.25, 0.25, 0.25, 0.25], weight: w1 },
        // 4 edge-cluster points: one coord is `aa`, the other three `a`.
        TetQuadPoint { bary: [aa, a, a, a], weight: w2 },
        TetQuadPoint { bary: [a, aa, a, a], weight: w2 },
        TetQuadPoint { bary: [a, a, aa, a], weight: w2 },
        TetQuadPoint { bary: [a, a, a, aa], weight: w2 },
        // 6 face-cluster points: each has two coords = c, two = b.
        // Corresponds to choosing 2 of 4 indices for `c`.
        TetQuadPoint { bary: [c, c, b, b], weight: w3 },
        TetQuadPoint { bary: [c, b, c, b], weight: w3 },
        TetQuadPoint { bary: [c, b, b, c], weight: w3 },
        TetQuadPoint { bary: [b, c, c, b], weight: w3 },
        TetQuadPoint { bary: [b, c, b, c], weight: w3 },
        TetQuadPoint { bary: [b, b, c, c], weight: w3 },
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
        let s6: Real = tri_quadrature_6pt().iter().map(|p| p.weight).sum();
        assert!(approx_eq(s1, 1.0));
        assert!(approx_eq(s3, 1.0));
        assert!((s6 - 1.0).abs() < 1.0e-10);
    }

    #[test]
    fn tet_rules_have_unit_weight_sum() {
        let s1: Real = tet_quadrature_1pt().iter().map(|p| p.weight).sum();
        let s4: Real = tet_quadrature_4pt().iter().map(|p| p.weight).sum();
        let s5: Real = tet_quadrature_5pt().iter().map(|p| p.weight).sum();
        let s11: Real = tet_quadrature_11pt().iter().map(|p| p.weight).sum();
        assert!(approx_eq(s1, 1.0));
        assert!(approx_eq(s4, 1.0));
        assert!((s5 - 1.0).abs() < 1.0e-12);
        assert!((s11 - 1.0).abs() < 1.0e-12);
    }

    #[test]
    fn bary_coords_partition_unity() {
        let all_tri = [
            tri_quadrature_1pt(),
            tri_quadrature_3pt(),
            tri_quadrature_6pt(),
        ];
        for rule in &all_tri {
            for p in rule {
                let s: Real = p.bary.iter().sum();
                assert!((s - 1.0).abs() < 1.0e-10, "tri bary sum {s}");
            }
        }
        let all_tet = [
            tet_quadrature_1pt(),
            tet_quadrature_4pt(),
            tet_quadrature_5pt(),
            tet_quadrature_11pt(),
        ];
        for rule in &all_tet {
            for p in rule {
                let s: Real = p.bary.iter().sum();
                assert!((s - 1.0).abs() < 1.0e-10, "tet bary sum {s}");
            }
        }
    }
}
