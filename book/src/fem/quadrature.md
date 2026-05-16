# Quadrature rules

Quadrature rules turn a continuous integral
\\( \int_K f(x) \, dx \\) over a reference simplex into a finite sum
\\( |K| \cdot \sum_q w_q \, f(x_q) \\) at a handful of barycentric
sample points. Molla ships rules of increasing degree for triangles
and tetrahedra in `crates/molla-fem/src/quadrature.rs`, all
pre-normalised so that \\( \sum_q w_q = 1 \\) — the assembly path
multiplies by the cell measure (area for triangles, volume for tets)
so the rules themselves are "shape-agnostic".

## Reference structures

```rust
pub struct TriQuadPoint {
    pub bary: [Real; 3],
    pub weight: Real,
}

pub struct TetQuadPoint {
    pub bary: [Real; 4],
    pub weight: Real,
}
```

Barycentric coordinates always satisfy \\( \sum_i \lambda_i = 1 \\),
which the `bary_coords_partition_unity` test verifies on every rule
shipped.

## Triangle rules

| Function                 | Points | Exact for `≤ degree` | Use                       |
|--------------------------|--------|----------------------|---------------------------|
| `tri_quadrature_1pt()`   | 1 (centroid)              | 1 | P1 stiffness (gradients constant) |
| `tri_quadrature_3pt()`   | 3 (edge midpoints)        | 2 | P1 mass; P2 stiffness with caveats |
| `tri_quadrature_6pt()`   | 6 (Dunavant two-orbit)    | 4 | P2 mass + P2 stiffness   |

The Dunavant degree-4 rule packs three "vertex-cluster" points at
`(a₁, b₁, b₁)` permutations with \\( a_1 \approx 0.8168 \\),
\\( b_1 \approx 0.0916 \\) and three "edge-cluster" points at
`(a₂, b₂, b₂)` permutations with \\( a_2 \approx 0.1081 \\),
\\( b_2 \approx 0.4459 \\). Weights are
\\( w_1 \approx 0.110 \\) and \\( w_2 \approx 0.223 \\), pre-normalised
so they sum to one. P2 mass integrates a quartic and P2 stiffness a
quadratic — both captured exactly by the degree-4 rule.

## Tetrahedron rules

| Function                 | Points | Exact for `≤ degree` | Use                       |
|--------------------------|--------|----------------------|---------------------------|
| `tet_quadrature_1pt()`   | 1 (centroid)               | 1 | P1 stiffness            |
| `tet_quadrature_4pt()`   | 4 (corner sampling)        | 1 | lumped-mass experiments |
| `tet_quadrature_5pt()`   | 5 (Keast, signed weights)  | 3 | Nedelec mass, P2 stiffness, nonconforming cross-mesh integration |
| `tet_quadrature_11pt()`  | 11 (Keast)                 | 4 | P2 mass                  |

The 5-point Keast rule is unusual in having a **negative** centroid
weight of \\( -4/5 \\) balanced by four positive
\\( 9/20 \\) weights at the
\\( (\frac{1}{2}, \frac{1}{6}, \frac{1}{6}, \frac{1}{6}) \\)
permutations. The 11-point rule combines a (negative-weighted)
centroid with four "edge-cluster" points at
\\( (\frac{11}{14}, \frac{1}{14}, \frac{1}{14}, \frac{1}{14}) \\)
permutations and six "face-cluster" points at the
\\( (b, b, c, c) \\) orbit (with \\( b + c = \frac{1}{2} \\)).

Both rules are tabulated directly from Keast's 1986 paper; the v1 test
suite verifies `∑ wᵢ = 1` to numerical roundoff.

## 3D-triangle surface rules — Stage 47 v3

Surface integrals — e.g. the `M_st` cross-mesh mass matrix used by
[`NonconformingField::new_surface`](./nonconforming.md) — need to
integrate over a triangle embedded in 3D. The dedicated `Tri3dQuadPoint`
type carries the world-space point, the area-pre-multiplied weight,
and the original barycentric coordinates:

```rust
pub struct Tri3dQuadPoint {
    pub point: Vec3,      // world-space integration point
    pub weight: Real,     // w_q · area(T)
    pub bary: [Real; 3],  // barycentric coords for shape evals
}

pub fn tri3d_quadrature_3pt(p: &[Vec3; 3]) -> Vec<Tri3dQuadPoint>;
pub fn tri3d_quadrature_6pt(p: &[Vec3; 3]) -> Vec<Tri3dQuadPoint>;
```

`tri3d_area` uses the 3D cross product so a triangle lying parallel to
the z-axis still gets the right area — the `tri3d_area_works_on_tilted_triangle`
test pins this with the classic 3-4-5 example tilted out of plane.

The 3-point rule is degree-2 exact (enough for `P1 × P1` mass
integrands); the 6-point rule is degree-4 exact and is the conservative
fallback when the cross-mesh field becomes higher order.

## When to use which rule

| Integrand on Tet              | Polynomial degree | Minimum rule needed |
|-------------------------------|-------------------|---------------------|
| `1`                           | 0                 | 1-pt                |
| `∇φᵢ · ∇φⱼ` (P1)              | 0 (gradients constant) | 1-pt           |
| `φᵢ · φⱼ` (P1)                | 2                 | 4-pt or higher      |
| `∇φᵢ · ∇φⱼ` (P2)              | 2                 | 5-pt                |
| `φᵢ · φⱼ` (P2)                | 4                 | 11-pt               |
| `Nᵢ · Nⱼ` (Nedelec)           | 2                 | closed-form below   |

The Nedelec mass matrix actually avoids a quadrature loop entirely —
the integral
\\( \int_K \lambda_p \lambda_q \, dV = \frac{(1 + \delta_{pq})}{20} \cdot |K| \\)
is closed-form, and the Nedelec basis is built from products of
barycentric coordinates with constant gradients, so the element matrix
collapses to a closed-form sum of `c(p, q) · ∇λᵢ · ∇λⱼ` terms. See
the [magnetostatics chapter](./magneto.md) for the full derivation.

## Cross-references

- The P1 element kernels in [elements](./elements.md) use the 1-point
  rule (exact for piecewise-linear gradient products).
- The P2 element kernels in [p2](./p2.md) use the 6-pt (triangle) and
  11-pt (tet) rules.
- The Stokes Taylor-Hood assembly in [stokes_ns](./stokes_ns.md) uses
  the same higher-order rules for the P2 velocity block.
- The cross-mesh `M_st` assembler in [nonconforming](./nonconforming.md)
  uses the 5-pt tet rule for volume integration and the 6-pt tri-3D
  rule for the Stage 47 v3 surface-only variant.
- The DG numerical-flux blocks in [dg_flux](./dg_flux.md) use the
  triangle 3-pt rule to integrate jump terms on faces.
