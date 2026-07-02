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
const TRI_MASS_REF: [[Real; 3]; 3] = [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]];

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
pub fn compute_element_stiffness(cell: &FemCell, vertices: &[Vec3], coeff: Real) -> Vec<Vec<Real>> {
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
                        .map(|j| {
                            coeff * area * (grads[i][0] * grads[j][0] + grads[i][1] * grads[j][1])
                        })
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

/// Per-cell **convection** matrix `Cₑ = ∫ Nᵢ · (b · ∇Nⱼ) dx`.
///
/// Stage 37.2 — first-order advection operator for the
/// convection-diffusion / Navier-Stokes momentum block. With P1
/// (linear) shape functions the gradients `∇Nⱼ` are constant per
/// element so `b · ∇Nⱼ` is a single number for each column `j`; the
/// integral collapses to that constant times `∫ Nᵢ dx = |Ω| / (N+1)`
/// for every row `i` (the well-known "barycentric mean" for P1
/// elements over a simplex). The block is therefore rank-1 in the
/// row direction, with all rows equal to the column vector
/// `(|Ω| / (N+1)) · (b · ∇Nⱼ)`.
///
/// Returns an `N × N` matrix as `Vec<Vec<Real>>` where `N = 3` (Tri)
/// or `N = 4` (Tet). The convection direction `b` is a 3D vector;
/// triangle elements use only the in-plane components `(b.x, b.y)`.
///
/// Without stabilisation this operator is **non-symmetric** and
/// becomes oscillatory when the cell Péclet number `|b|·h / (2α)`
/// exceeds 1. Pair with [`compute_element_supg_convection`] to
/// recover monotonicity in the convection-dominated regime.
#[must_use]
pub fn compute_element_convection(cell: &FemCell, vertices: &[Vec3], b: Vec3) -> Vec<Vec<Real>> {
    match cell {
        FemCell::Tri(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
            ];
            let area = tri_area_2d(&p);
            let grads = tri_shape_grads_2d(&p);
            // ∫ Nᵢ dx = area / 3 for P1 on a triangle.
            let mean = area / 3.0;
            let bg: [Real; 3] = [
                b.x * grads[0][0] + b.y * grads[0][1],
                b.x * grads[1][0] + b.y * grads[1][1],
                b.x * grads[2][0] + b.y * grads[2][1],
            ];
            (0..3)
                .map(|_i| (0..3).map(|j| mean * bg[j]).collect())
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
            // ∫ Nᵢ dx = vol / 4 for P1 on a tetrahedron.
            let mean = vol / 4.0;
            let bg: [Real; 4] = [
                b.dot(grads[0]),
                b.dot(grads[1]),
                b.dot(grads[2]),
                b.dot(grads[3]),
            ];
            (0..4)
                .map(|_i| (0..4).map(|j| mean * bg[j]).collect())
                .collect()
        }
    }
}

/// Per-cell **SUPG** (Streamline-Upwind Petrov-Galerkin) stabilisation
/// block `Sₑ = ∫ τ · (b · ∇Nᵢ) · (b · ∇Nⱼ) dx`.
///
/// Stage 37.2 — Hughes' classic textbook stabilisation: bias the test
/// function in the streamline direction so the discretised
/// convection-diffusion operator stays monotone in the
/// convection-dominated regime (cell Péclet > 1). The stabilisation
/// parameter is `τ = h / (2 |b|)` (caller computes `h` as the cell
/// length scale; `alpha` is exposed as a multiplier so callers can
/// blend in additional dampening or follow alternative `τ` formulas).
///
/// The block is added on top of [`compute_element_convection`] —
/// together they replace the standard Galerkin convection term with a
/// Petrov-Galerkin one that retains the conservation properties at
/// `Pe ≪ 1` and removes the cell-Péclet oscillations at `Pe ≫ 1`.
///
/// With P1 shape functions the gradients are constant per element so
/// the integral collapses to `τ · (b · ∇Nᵢ) · (b · ∇Nⱼ) · |Ω|`. The
/// resulting block is **symmetric** (rank-1 outer product of `b·∇N`
/// with itself, scaled by `τ · |Ω|`).
///
/// Returns an `N × N` matrix as `Vec<Vec<Real>>` where `N = 3` (Tri)
/// or `N = 4` (Tet). When `b == 0` the block is identically zero
/// (and skipping the call is the obvious micro-optimisation).
#[must_use]
pub fn compute_element_supg_convection(
    cell: &FemCell,
    vertices: &[Vec3],
    b: Vec3,
    alpha: Real,
    h: Real,
) -> Vec<Vec<Real>> {
    let bmag = b.length();
    if bmag <= 1.0e-30 {
        let n = cell.vert_count();
        return vec![vec![0.0; n]; n];
    }
    // Hughes SUPG parameter — the textbook `h / (2 |b|)` choice. The
    // optional `alpha` multiplier lets callers tune the stabilisation
    // strength without rebuilding the kernel.
    let tau = alpha * h / (2.0 * bmag);
    match cell {
        FemCell::Tri(idx) => {
            let p = [
                vertices[idx[0] as usize],
                vertices[idx[1] as usize],
                vertices[idx[2] as usize],
            ];
            let area = tri_area_2d(&p);
            let grads = tri_shape_grads_2d(&p);
            let bg: [Real; 3] = [
                b.x * grads[0][0] + b.y * grads[0][1],
                b.x * grads[1][0] + b.y * grads[1][1],
                b.x * grads[2][0] + b.y * grads[2][1],
            ];
            let scale = tau * area;
            (0..3)
                .map(|i| (0..3).map(|j| scale * bg[i] * bg[j]).collect())
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
            let bg: [Real; 4] = [
                b.dot(grads[0]),
                b.dot(grads[1]),
                b.dot(grads[2]),
                b.dot(grads[3]),
            ];
            let scale = tau * vol;
            (0..4)
                .map(|i| (0..4).map(|j| scale * bg[i] * bg[j]).collect())
                .collect()
        }
    }
}

/// Per-cell **wave** stiffness matrix `Kₑ = c² · ∫ ∇φᵢ · ∇φⱼ dx`.
///
/// Stage 37.1 — symmetric Helmholtz operator used as the spatial
/// operator of the second-order wave equation `∂²u/∂t² = c² ∇²u`. The
/// math is identical to [`compute_element_stiffness`]; only the
/// physical interpretation of the scaling factor differs (here `c²` is
/// the squared wave speed instead of a thermal diffusivity). This is
/// kept as a separate entry-point so callers reading the assembly
/// visitor list can tell at a glance which PDE is being assembled.
///
/// Triangle elements project to the `(x, y)` plane; tetrahedral
/// elements use the full 3D vertex positions.
#[must_use]
pub fn compute_element_wave_stiffness(
    cell: &FemCell,
    vertices: &[Vec3],
    c2: Real,
) -> Vec<Vec<Real>> {
    // Same math as the Laplacian — just rebadged with `c²` as the
    // physical coefficient. Re-using the implementation keeps the two
    // kernels bit-for-bit identical (see
    // `wave_element_stiffness_proportional_to_laplacian`).
    compute_element_stiffness(cell, vertices, c2)
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

/// Absolute volume of a tetrahedron — always non-negative. Use this
/// for SPD element matrices (mass, Laplacian, linear elasticity)
/// where the integration measure must be positive regardless of
/// vertex orientation; pair it with shape-gradient routines that
/// also work in absolute terms (`tet_shape_grads` returns signed
/// gradients, so for SPD operators the per-element products
/// `vol · ∇φᵢ · ∇φⱼ` cancel the sign on flipped tets).
#[must_use]
pub fn tet_volume(p: &[Vec3; 4]) -> Real {
    tet_signed_volume(p).abs()
}

/// Signed volume of a tetrahedron — positive when `(b−a, c−a, d−a)`
/// is right-handed, negative when the vertices are in a left-handed
/// orientation. Required by routines that pair the volume with
/// signed shape gradients (e.g. SVK / hyperelastic internal forces)
/// where mismatched signs flip the internal-force direction on a
/// flipped rest tet.
#[must_use]
pub fn tet_signed_volume(p: &[Vec3; 4]) -> Real {
    (p[1] - p[0]).dot((p[2] - p[0]).cross(p[3] - p[0])) / 6.0
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
            // SVK pairs the volume with signed shape gradients, so
            // the volume must be signed too — otherwise a flipped
            // rest tet produces a wrong-sign internal force (the
            // hyperelastic energy itself doesn't care about parent
            // orientation, but the assembled nodal force does).
            let vol = tet_signed_volume(&p);
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
            // for tets only. Earlier this returned `vec![0.0; nv*2]`
            // which silently produced wrong-size output for 3D callers
            // indexing as `3*i + c` (out-of-bounds reads in release
            // mode); panic explicitly instead so misuse is loud.
            unimplemented!(
                "compute_element_st_venant_kirchhoff_stress: Tri (plane-strain 2D) \
                 SVK is not implemented — use Tet cells in 3D, or extend this \
                 function for 2D plane-strain elasticity"
            )
        }
    }
}

/// Compressible Neo-Hookean strain energy for a P1 tetrahedron.
///
/// Energy density:
///
/// ```text
/// Ψ(F) = μ/2 (I₁ - 3) - μ ln J + λ/2 (ln J)²
/// ```
///
/// where `J = det(F)`. This model is stress-free at `F = I` and its
/// small-strain tangent matches linear isotropic elasticity with Lamé
/// parameters `(λ, μ)`.
#[must_use]
pub fn compute_element_neo_hookean_energy(
    cell: &FemCell,
    vertices: &[Vec3],
    u_local: &[Real],
    lambda: Real,
    mu: Real,
) -> Real {
    let (f, vol) = tet_deformation_gradient(cell, vertices, u_local, "Neo-Hookean");
    let j = f.determinant();
    assert!(j > 0.0, "Neo-Hookean: inverted element with det(F) = {j}");
    let i1 = (f.transpose() * f).to_cols_array()[0]
        + (f.transpose() * f).to_cols_array()[4]
        + (f.transpose() * f).to_cols_array()[8];
    let ln_j = j.ln();
    vol.abs() * (0.5 * mu * (i1 - 3.0) - mu * ln_j + 0.5 * lambda * ln_j * ln_j)
}

/// Compressible Mooney-Rivlin strain energy for a P1 tetrahedron.
///
/// This is a compact two-invariant model with a volumetric penalty:
///
/// ```text
/// Ψ(F) = C10(I₁ - 3) + C01(I₂ - 3)
///        - (2C10 + 4C01) ln J + κ/2 (J - 1)²
/// ```
///
/// The logarithmic term makes the first derivative vanish at
/// `F = I`, while `κ` controls the bulk response.
#[must_use]
pub fn compute_element_mooney_rivlin_energy(
    cell: &FemCell,
    vertices: &[Vec3],
    u_local: &[Real],
    c10: Real,
    c01: Real,
    kappa: Real,
) -> Real {
    let (f, vol) = tet_deformation_gradient(cell, vertices, u_local, "Mooney-Rivlin");
    let j = f.determinant();
    assert!(j > 0.0, "Mooney-Rivlin: inverted element with det(F) = {j}");
    let c = f.transpose() * f;
    let cols = c.to_cols_array();
    let i1 = cols[0] + cols[4] + cols[8];
    let c2 = c * c;
    let c2_cols = c2.to_cols_array();
    let tr_c2 = c2_cols[0] + c2_cols[4] + c2_cols[8];
    let i2 = 0.5 * (i1 * i1 - tr_c2);
    let ln_j = j.ln();
    vol.abs()
        * (c10 * (i1 - 3.0) + c01 * (i2 - 3.0) - (2.0 * c10 + 4.0 * c01) * ln_j
            + 0.5 * kappa * (j - 1.0) * (j - 1.0))
}

/// Internal nodal force for the Neo-Hookean tet, evaluated as the
/// central finite-difference gradient of
/// [`compute_element_neo_hookean_energy`].
#[must_use]
pub fn compute_element_neo_hookean_force(
    cell: &FemCell,
    vertices: &[Vec3],
    u_local: &[Real],
    lambda: Real,
    mu: Real,
) -> Vec<Real> {
    finite_difference_hyperelastic_force(u_local, |u| {
        compute_element_neo_hookean_energy(cell, vertices, u, lambda, mu)
    })
}

/// Consistent material tangent for the Neo-Hookean tet, evaluated by
/// central finite differences of [`compute_element_neo_hookean_force`].
#[must_use]
pub fn compute_element_neo_hookean_tangent(
    cell: &FemCell,
    vertices: &[Vec3],
    u_local: &[Real],
    lambda: Real,
    mu: Real,
) -> Vec<Vec<Real>> {
    finite_difference_hyperelastic_tangent(u_local, |u| {
        compute_element_neo_hookean_force(cell, vertices, u, lambda, mu)
    })
}

/// Internal nodal force for the Mooney-Rivlin tet.
#[must_use]
pub fn compute_element_mooney_rivlin_force(
    cell: &FemCell,
    vertices: &[Vec3],
    u_local: &[Real],
    c10: Real,
    c01: Real,
    kappa: Real,
) -> Vec<Real> {
    finite_difference_hyperelastic_force(u_local, |u| {
        compute_element_mooney_rivlin_energy(cell, vertices, u, c10, c01, kappa)
    })
}

/// Consistent material tangent for the Mooney-Rivlin tet.
#[must_use]
pub fn compute_element_mooney_rivlin_tangent(
    cell: &FemCell,
    vertices: &[Vec3],
    u_local: &[Real],
    c10: Real,
    c01: Real,
    kappa: Real,
) -> Vec<Vec<Real>> {
    finite_difference_hyperelastic_tangent(u_local, |u| {
        compute_element_mooney_rivlin_force(cell, vertices, u, c10, c01, kappa)
    })
}

fn tet_deformation_gradient(
    cell: &FemCell,
    vertices: &[Vec3],
    u_local: &[Real],
    label: &str,
) -> (crate::math::Mat3, Real) {
    let FemCell::Tet(idx) = cell else {
        unimplemented!("{label}: only Tet cells are implemented");
    };
    assert_eq!(u_local.len(), 12, "{label}: u_local must be 12-component");
    let p = [
        vertices[idx[0] as usize],
        vertices[idx[1] as usize],
        vertices[idx[2] as usize],
        vertices[idx[3] as usize],
    ];
    let vol = tet_signed_volume(&p);
    let grads = tet_shape_grads(&p);
    let mut f = crate::math::Mat3::IDENTITY;
    for i in 0..4 {
        let ux = u_local[3 * i];
        let uy = u_local[3 * i + 1];
        let uz = u_local[3 * i + 2];
        let g = grads[i];
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
    (f, vol)
}

fn finite_difference_hyperelastic_force(
    u_local: &[Real],
    energy: impl Fn(&[Real]) -> Real,
) -> Vec<Real> {
    let n = u_local.len();
    let mut out = vec![0.0; n];
    let eps = 1.0e-6;
    let mut up = u_local.to_vec();
    let mut um = u_local.to_vec();
    for i in 0..n {
        up[i] += eps;
        um[i] -= eps;
        out[i] = (energy(&up) - energy(&um)) / (2.0 * eps);
        up[i] = u_local[i];
        um[i] = u_local[i];
    }
    out
}

fn finite_difference_hyperelastic_tangent(
    u_local: &[Real],
    force: impl Fn(&[Real]) -> Vec<Real>,
) -> Vec<Vec<Real>> {
    let n = u_local.len();
    let eps = 1.0e-6;
    let mut k = vec![vec![0.0; n]; n];
    let mut up = u_local.to_vec();
    let mut um = u_local.to_vec();
    for j in 0..n {
        up[j] += eps;
        um[j] -= eps;
        let fp = force(&up);
        let fm = force(&um);
        for i in 0..n {
            k[i][j] = (fp[i] - fm[i]) / (2.0 * eps);
        }
        up[j] = u_local[j];
        um[j] = u_local[j];
    }
    k
}

// ─── Symmetric Dirichlet distortion energy ──────────────────────────

/// Per-cell **symmetric Dirichlet distortion energy** along with its
/// gradient and Hessian with respect to the deformed-vertex positions.
///
/// `E = ½ tr(FᵀF + (FᵀF)⁻¹) − dim`
///
/// where `F = D_def · D_rest⁻¹` is the deformation gradient mapping
/// the rest-pose simplex to the deformed simplex, and `dim` is 2 (Tri)
/// or 3 (Tet). The energy is zero on the identity map and grows
/// monotonically away from it, with a barrier at every degenerate
/// (zero-volume) configuration. It's the canonical "distortion-aware"
/// objective for parameterisation / inverse-deformation tasks.
///
/// The returned tuple is:
/// - **energy** — scalar `E` for this element.
/// - **gradient** — one `Vec3` per local node holding `∂E/∂x_a`.
///   Length matches `cell.vert_count()`.
/// - **hessian** — `(N · 3) × (N · 3)` row-major dense matrix as
///   `Vec<Vec<Real>>`, storing `∂²E/∂x_{a,α} ∂x_{b,β}` in interleaved
///   layout `(node_a · 3 + α, node_b · 3 + β)`. For a tri we get a
///   9×9 block; for a tet we get a 12×12 block.
///
/// v1 implementation: the gradient and Hessian are evaluated by
/// **central finite differences** on the analytic energy. The energy
/// itself is closed-form (uses [`crate::math::Mat3::inverse`] /
/// `Mat3::determinant` for the 3D path, and an inline 2×2 inverse for
/// the in-plane triangle path). Finite differences keep the v1
/// implementation short, easy to audit, and correct to roundoff
/// thanks to the smooth analytic energy. A follow-up stage can swap
/// in the analytic gradient `P = ∂Ψ/∂F` + tangent stiffness
/// `K = ∂²Ψ/∂F²` once the `dF/dx` chain rule is wired through.
///
/// Triangles are handled in plane: we work with the 2D in-plane part
/// of `F` (so `dim = 2`); the z-component of vertex perturbations is
/// ignored (gradient z-entries stay zero, Hessian z-rows / -cols
/// stay zero). This matches the convention used throughout the
/// `element` module for tri kernels.
#[must_use]
pub fn compute_element_distortion_energy(
    cell: &FemCell,
    rest_vertices: &[Vec3],
    deformed_vertices: &[Vec3],
) -> (Real, Vec<Vec3>, Vec<Vec<Real>>) {
    let nv = cell.vert_count();
    let n_dof = nv * 3;
    // Closed-form energy in terms of deformed positions only.
    let energy_at = |x: &[Vec3]| -> Real {
        match cell {
            FemCell::Tri(idx) => {
                let r = [
                    rest_vertices[idx[0] as usize],
                    rest_vertices[idx[1] as usize],
                    rest_vertices[idx[2] as usize],
                ];
                let d = [x[0], x[1], x[2]];
                tri_distortion_energy(&r, &d)
            }
            FemCell::Tet(idx) => {
                let r = [
                    rest_vertices[idx[0] as usize],
                    rest_vertices[idx[1] as usize],
                    rest_vertices[idx[2] as usize],
                    rest_vertices[idx[3] as usize],
                ];
                let d = [x[0], x[1], x[2], x[3]];
                tet_distortion_energy(&r, &d)
            }
        }
    };

    // Materialise the deformed positions for this cell into a local array.
    let mut local: Vec<Vec3> = cell
        .verts()
        .iter()
        .map(|&v| deformed_vertices[v as usize])
        .collect();
    let energy = energy_at(&local);

    // Central finite differences. `eps` is a small absolute step; for
    // an `f64` energy with magnitudes around 1e-3 .. 1e3 the central
    // gradient is accurate to ~1e-9 with eps=1e-5 (truncation O(eps²)
    // ~1e-10, roundoff O(1/eps) ~1e-11).
    let eps: Real = 1.0e-5;
    let mut gradient = vec![Vec3::ZERO; nv];
    for a in 0..nv {
        for alpha in 0..3 {
            let saved = local[a];
            let mut perturbed = saved;
            perturbed[alpha] = saved[alpha] + eps;
            local[a] = perturbed;
            let e_plus = energy_at(&local);
            let mut perturbed = saved;
            perturbed[alpha] = saved[alpha] - eps;
            local[a] = perturbed;
            let e_minus = energy_at(&local);
            local[a] = saved;
            let g = (e_plus - e_minus) / (2.0 * eps);
            gradient[a][alpha] = g;
        }
    }

    // Hessian: direct two-point central finite difference on the
    // **energy**, not on the gradient — avoids FD-on-FD roundoff
    // compounding. Stencils:
    //   H_ii ≈ (E(x + h·e_i) − 2 E(x) + E(x − h·e_i)) / h²
    //   H_ij ≈ (E(x + h·e_i + h·e_j) − E(x + h·e_i) − E(x + h·e_j) + E(x)) / h²  for i ≠ j
    // (the standard first-order forward mixed-partial formula; we
    // symmetrise the (i, j) and (j, i) entries afterwards.)
    let mut hessian = vec![vec![0.0; n_dof]; n_dof];
    let h_eps: Real = 1.0e-4;
    let h2 = h_eps * h_eps;
    let saved_local = local.clone();
    let perturb = |coord: usize, delta: Real, base: &[Vec3]| -> Vec<Vec3> {
        let mut out = base.to_vec();
        let node = coord / 3;
        let comp = coord % 3;
        let mut p = out[node];
        p[comp] += delta;
        out[node] = p;
        out
    };
    let e0 = energy_at(&saved_local);
    let mut e_plus_i = vec![0.0_f64; n_dof];
    for i in 0..n_dof {
        let p = perturb(i, h_eps, &saved_local);
        e_plus_i[i] = energy_at(&p);
    }
    let mut e_minus_i = vec![0.0_f64; n_dof];
    for i in 0..n_dof {
        let p = perturb(i, -h_eps, &saved_local);
        e_minus_i[i] = energy_at(&p);
    }
    // Diagonal entries.
    for i in 0..n_dof {
        hessian[i][i] = (e_plus_i[i] - 2.0 * e0 + e_minus_i[i]) / h2;
    }
    // Off-diagonals via the forward mixed-partial formula, then
    // symmetrise.
    for i in 0..n_dof {
        for j in (i + 1)..n_dof {
            // Build x + h·e_i + h·e_j by perturbing twice.
            let mut p = saved_local.clone();
            let ni = i / 3;
            let ci = i % 3;
            let nj = j / 3;
            let cj = j % 3;
            let mut tmp = p[ni];
            tmp[ci] += h_eps;
            p[ni] = tmp;
            let mut tmp = p[nj];
            tmp[cj] += h_eps;
            p[nj] = tmp;
            let e_pp = energy_at(&p);
            let h_ij = (e_pp - e_plus_i[i] - e_plus_i[j] + e0) / h2;
            hessian[i][j] = h_ij;
            hessian[j][i] = h_ij;
        }
    }

    (energy, gradient, hessian)
}

/// 2D symmetric Dirichlet energy of a triangle map (rest → deformed).
/// Uses only the in-plane components of the input positions.
fn tri_distortion_energy(rest: &[Vec3; 3], def: &[Vec3; 3]) -> Real {
    // 2×2 D matrix [v1 - v0 | v2 - v0] in plane.
    let dr00 = rest[1].x - rest[0].x;
    let dr10 = rest[1].y - rest[0].y;
    let dr01 = rest[2].x - rest[0].x;
    let dr11 = rest[2].y - rest[0].y;
    let det_r = dr00 * dr11 - dr01 * dr10;
    if det_r.abs() < 1.0e-30 {
        return Real::INFINITY;
    }
    let inv = 1.0 / det_r;
    let r_inv00 = dr11 * inv;
    let r_inv01 = -dr01 * inv;
    let r_inv10 = -dr10 * inv;
    let r_inv11 = dr00 * inv;
    let dd00 = def[1].x - def[0].x;
    let dd10 = def[1].y - def[0].y;
    let dd01 = def[2].x - def[0].x;
    let dd11 = def[2].y - def[0].y;
    // F = D_def · R_inv  (2×2).
    let f00 = dd00 * r_inv00 + dd01 * r_inv10;
    let f01 = dd00 * r_inv01 + dd01 * r_inv11;
    let f10 = dd10 * r_inv00 + dd11 * r_inv10;
    let f11 = dd10 * r_inv01 + dd11 * r_inv11;
    // ‖F‖² = tr(FᵀF).
    let frob_sq = f00 * f00 + f01 * f01 + f10 * f10 + f11 * f11;
    // det(F) and ‖F⁻¹‖² (uses the closed-form 2×2 inverse).
    let det_f = f00 * f11 - f01 * f10;
    if det_f.abs() < 1.0e-30 {
        return Real::INFINITY;
    }
    let inv_f = 1.0 / det_f;
    let fi00 = f11 * inv_f;
    let fi01 = -f01 * inv_f;
    let fi10 = -f10 * inv_f;
    let fi11 = f00 * inv_f;
    let frob_inv_sq = fi00 * fi00 + fi01 * fi01 + fi10 * fi10 + fi11 * fi11;
    0.5 * (frob_sq + frob_inv_sq) - 2.0
}

/// 3D symmetric Dirichlet energy of a tetrahedron map (rest → deformed).
fn tet_distortion_energy(rest: &[Vec3; 4], def: &[Vec3; 4]) -> Real {
    use crate::math::Mat3;
    let dr = Mat3::from_cols(rest[1] - rest[0], rest[2] - rest[0], rest[3] - rest[0]);
    let det_r = dr.determinant();
    if det_r.abs() < 1.0e-30 {
        return Real::INFINITY;
    }
    let dr_inv = dr.inverse();
    let dd = Mat3::from_cols(def[1] - def[0], def[2] - def[0], def[3] - def[0]);
    let f = dd * dr_inv;
    let det_f = f.determinant();
    if det_f.abs() < 1.0e-30 {
        return Real::INFINITY;
    }
    let f_inv = f.inverse();
    // ‖F‖² = sum of squares of all entries.
    let frob_sq = f.x_axis.length_squared() + f.y_axis.length_squared() + f.z_axis.length_squared();
    let frob_inv_sq = f_inv.x_axis.length_squared()
        + f_inv.y_axis.length_squared()
        + f_inv.z_axis.length_squared();
    0.5 * (frob_sq + frob_inv_sq) - 3.0
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
        let one = [1.0, 1.0, 1.0, 1.0];
        for i in 0..4 {
            let s: Real = (0..4).map(|j| k[i][j] * one[j]).sum();
            assert!(s.abs() < 1e-12, "row {i}: K·1 = {s}");
        }
    }
}
