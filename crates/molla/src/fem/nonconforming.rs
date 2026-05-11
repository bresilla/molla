//! Stage 47 v1 — L² projection between non-matching tetrahedral meshes.
//!
//! Two FEM meshes that share no DoFs need a transfer operator to move a
//! scalar field from one to the other. The textbook (and Warp-reference)
//! recipe is **L²-projection mediated by a cross-mesh mass matrix**:
//!
//! ```text
//!   M_st_ij = ∫_target φ_target_i(x) · φ_source_j(x) dx
//!   M_tt_ij = ∫_target φ_target_i(x) · φ_target_j(x) dx
//!   u_target = M_tt⁻¹ · (M_st · u_source)
//! ```
//!
//! `M_st` is rectangular (`n_target × n_source`); `M_tt` is the **target
//! mesh's own consistent mass matrix** (SPD, square, CG-solvable). The
//! integral that defines `M_st` is computed on the **target** cells —
//! that way `φ_target_i` is a simple polynomial in each integration
//! sub-domain, and we only need to evaluate `φ_source_j` at each
//! quadrature point via barycentric tet-location.
//!
//! # Properties (P1 nodal, target-side quadrature)
//!
//! - **Constant reproduction**: `u_source ≡ 1` ⇒ `u_target ≡ 1` exactly
//!   (modulo CG tolerance). Reason: `M_st · 1_source` reproduces the
//!   target lumped row sums (`∫ φ_target_i · ∑φ_source_j dx = ∫ φ_target_i dx`,
//!   which is also `M_tt · 1_target`), so `M_tt⁻¹ · M_st · 1 = 1`.
//! - **Linear reproduction**: with degree-≥2 quadrature, the projection
//!   is exact on affine source fields up to floating-point round-off
//!   (P1 contains all affine functions exactly, and the L² projector
//!   onto a space that includes the source field is the identity).
//! - **Quadratic loss**: a quadratic source field is not in P1, so the
//!   projection introduces an O(h²) error; halving `h` cuts the error
//!   by ~4× — the standard P1 best-approximation rate.
//!
//! # Out of scope (v1)
//!
//! - **Elasticity coupling** — the cross-mesh field that matters for the
//!   non-matching contact paper is the *displacement* (vector-valued).
//!   The scalar operator here is the building block; v2 will (a) extend
//!   `coupling_matrix` to a block-1 `BsrMatrix` with `B = 3` and (b)
//!   wire it into the residual / Jacobian of the contact problem.
//! - **Contact pressure formulation** — Hertz / Signorini contact across
//!   non-matching surfaces wants an additional surface mass operator on
//!   the slave side, plus a Lagrange multiplier or penalty term. v3.
//! - **Tet-location acceleration** — v1 uses the same brute-force
//!   barycentric search as Stage 37.6 (`apic_coupling::locate_particle_tet`).
//!   A `TetBvh` (`crates/molla/src/geometry/tet_bvh.rs`) would drop the
//!   build cost from O(N_target · N_source) to O(N_target · log N_source)
//!   without changing the public API.
//!
//! # Reference
//!
//! - Warp `example_nonconforming_contact.py` (the `NonconformingField`
//!   class).
//! - Quarteroni & Valli, *Domain Decomposition Methods for Partial
//!   Differential Equations*, §1.4 (mortar L² projection).

use crate::core::{Error, Result};
use crate::math::iterative::cg_solve;
use crate::math::sparse::BsrMatrix;
use crate::math::{Mat3, Real, Vec3};

use super::assembly::{assemble_global, matvec};
use super::element::compute_element_mass;
use super::function_space::FunctionSpace;
use super::mesh::{FemCell, FemMesh};
use super::quadrature::tet_quadrature_5pt;

/// Default tolerance for the inner CG solve against `M_tt`.
const DEFAULT_CG_TOL: Real = 1.0e-12;
/// Default cap on CG iterations — the mass matrix is well-conditioned
/// (κ ~ O(1) independent of mesh size) so 200 iterations is generous.
const DEFAULT_CG_MAX_ITERS: usize = 1000;
/// Slack used when classifying a quadrature point as "inside" a source
/// tet. Matches the apic_coupling convention.
const BARY_EPS: Real = 1.0e-9;

/// Scalar L²-projection operator between two non-matching P1 nodal
/// tetrahedral meshes.
///
/// The constructor builds the cross-mesh mass matrix `M_st` once;
/// repeated [`NonconformingField::project`] calls reuse it and solve
/// `M_tt · u_target = M_st · u_source` via CG.
///
/// Lifetimes: `source` and `target` are borrowed for the lifetime of
/// the operator. Cloning the field is intentionally not supported —
/// the user holds the meshes alive externally.
pub struct NonconformingField<'a> {
    /// Source mesh — the field `u_source` lives on this mesh's vertices.
    pub source: &'a FemMesh,
    /// Target mesh — the projected field `u_target` lives here.
    pub target: &'a FemMesh,
    /// Cross-mesh mass matrix `M_st` (`n_target × n_source`, scalar).
    coupling_matrix: BsrMatrix,
    /// Target mesh's consistent mass matrix `M_tt` (`n_target × n_target`).
    /// Stored so each `project` call doesn't reassemble it.
    target_mass: BsrMatrix,
}

impl<'a> NonconformingField<'a> {
    /// Build the cross-mesh L² projector.
    ///
    /// Quadrature: 5-point Keast rule on every target tet (degree-3
    /// exact in barycentric coordinates ⇒ the P1·P1 product is captured
    /// exactly when source and target overlap on a single tet; for
    /// partial overlaps the rule still bounds the error at O(h⁴), well
    /// below the O(h²) interpolation error of the projection itself).
    ///
    /// Tet-location: brute-force linear scan over `source.cells` — see
    /// the module-level rationale. Returns [`Error::Fem`] only when the
    /// underlying CSR assembly hits a degenerate row (e.g. a target
    /// mesh with no tets).
    pub fn new(source: &'a FemMesh, target: &'a FemMesh) -> Result<Self> {
        // Build the source-side AABBs once: barycentric solves are
        // cheap but pruning candidates with an AABB filter typically
        // wins anyway because most quadrature points fall in only a
        // handful of source tets. (We deliberately do *not* build a
        // BVH at v1 — see the module-level note on tet-location
        // acceleration.)
        let source_aabbs = build_tet_aabbs(source)?;

        let coupling_matrix = assemble_cross_mass(source, target, &source_aabbs)?;
        let target_mass = assemble_global(
            target,
            FunctionSpace::P1Nodal,
            compute_element_mass,
        )?;

        Ok(Self {
            source,
            target,
            coupling_matrix,
            target_mass,
        })
    }

    /// Number of source DoFs (target rows of `M_st`).
    #[must_use]
    pub fn source_dofs(&self) -> usize {
        self.source.vertices.len()
    }

    /// Number of target DoFs (rows of the projection output).
    #[must_use]
    pub fn target_dofs(&self) -> usize {
        self.target.vertices.len()
    }

    /// Borrow the assembled cross-mesh mass matrix `M_st`.
    #[must_use]
    pub fn coupling_matrix(&self) -> &BsrMatrix {
        &self.coupling_matrix
    }

    /// Borrow the target-side consistent mass matrix `M_tt`.
    #[must_use]
    pub fn target_mass_matrix(&self) -> &BsrMatrix {
        &self.target_mass
    }

    /// Project a scalar field defined on the source vertices onto the
    /// target vertices via `u_target = M_tt⁻¹ · M_st · u_source`.
    ///
    /// Internally:
    /// 1. Compute the RHS `b = M_st · u_source` (sparse matvec).
    /// 2. Solve `M_tt · u_target = b` with CG starting from a zero
    ///    initial guess (tolerance [`DEFAULT_CG_TOL`], max
    ///    [`DEFAULT_CG_MAX_ITERS`] iterations).
    ///
    /// Panics on a length mismatch between `u_source` and the source
    /// mesh's DoF count (treat as a programming error — projection
    /// targets are fixed at construction time).
    #[must_use]
    pub fn project(&self, u_source: &[Real]) -> Vec<Real> {
        self.project_with(u_source, DEFAULT_CG_TOL, DEFAULT_CG_MAX_ITERS)
    }

    /// Same as [`project`](Self::project) but lets the caller override
    /// the CG tolerance and iteration cap. Useful for the
    /// `project_constant_field_is_constant` test which demands
    /// 1e-10 absolute accuracy.
    #[must_use]
    pub fn project_with(
        &self,
        u_source: &[Real],
        tol: Real,
        max_iters: usize,
    ) -> Vec<Real> {
        assert_eq!(
            u_source.len(),
            self.source_dofs(),
            "NonconformingField::project: u_source length {} != source DoF count {}",
            u_source.len(),
            self.source_dofs(),
        );
        let rhs = matvec(&self.coupling_matrix, u_source);
        let x0 = vec![0.0; self.target_dofs()];
        let result = cg_solve(
            |x| matvec(&self.target_mass, x),
            &rhs,
            &x0,
            tol,
            max_iters,
        );
        result.x
    }
}

// ─── Cross-mesh mass assembly ────────────────────────────────────────

/// Compact AABB record for each tet in a mesh — used as a pruning
/// hint before the full barycentric containment test.
#[derive(Copy, Clone)]
struct TetAabb {
    min: Vec3,
    max: Vec3,
    verts: [u32; 4],
}

/// Walk a mesh and return one [`TetAabb`] per tetrahedral cell. Tri
/// cells are skipped — v1 of the operator is 3D-only. Returns
/// [`Error::Fem`] if the mesh contains zero tetrahedra (a target /
/// source mesh built from triangles alone is not supported here).
fn build_tet_aabbs(mesh: &FemMesh) -> Result<Vec<TetAabb>> {
    let mut out = Vec::with_capacity(mesh.cells.len());
    for cell in &mesh.cells {
        if let FemCell::Tet(v) = cell {
            let p = [
                mesh.vertices[v[0] as usize],
                mesh.vertices[v[1] as usize],
                mesh.vertices[v[2] as usize],
                mesh.vertices[v[3] as usize],
            ];
            let mut lo = p[0];
            let mut hi = p[0];
            for q in &p[1..] {
                lo.x = lo.x.min(q.x);
                lo.y = lo.y.min(q.y);
                lo.z = lo.z.min(q.z);
                hi.x = hi.x.max(q.x);
                hi.y = hi.y.max(q.y);
                hi.z = hi.z.max(q.z);
            }
            out.push(TetAabb {
                min: lo,
                max: hi,
                verts: *v,
            });
        }
    }
    if out.is_empty() {
        return Err(Error::Fem(
            "NonconformingField: mesh has no tetrahedral cells".to_string(),
        ));
    }
    Ok(out)
}

/// Assemble the cross-mesh mass matrix `M_st_ij = ∫ φ_target_i · φ_source_j dV`.
///
/// Integration scheme:
/// - Loop over every target tet `T`.
/// - For each Keast 5-point quadrature point `q` (barycentric on `T`),
///   compute its world position `x`.
/// - Brute-force locate the source tet `S(q)` containing `x` (skip the
///   point if no source tet contains it — the source mesh may only
///   partly overlap the target domain).
/// - Accumulate `M_st_(target_vert_i, source_vert_j) +=
///   w_q · vol(T) · φ_target_i(q) · φ_source_j(q)` for all 4×4 (i, j)
///   pairs. `φ_*_i(q)` are just the barycentric coordinates of `q` on
///   the respective tet.
fn assemble_cross_mass(
    source: &FemMesh,
    target: &FemMesh,
    source_aabbs: &[TetAabb],
) -> Result<BsrMatrix> {
    let nrows = target.vertices.len();
    let ncols = source.vertices.len();

    let rule = tet_quadrature_5pt();
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    triplets.reserve(target.cells.len() * rule.len() * 16);

    for cell in &target.cells {
        let FemCell::Tet(tv) = cell else { continue };
        let tp = [
            target.vertices[tv[0] as usize],
            target.vertices[tv[1] as usize],
            target.vertices[tv[2] as usize],
            target.vertices[tv[3] as usize],
        ];
        let target_vol = tet_volume(&tp);
        if target_vol <= 0.0 {
            // Degenerate target tet — skip.
            continue;
        }

        for q in &rule {
            // World-space location of the quadrature point.
            let x = bary_to_world(&tp, &q.bary);
            // Locate the source tet containing `x`.
            let Some((source_verts, source_bary)) = locate_in_source(source, source_aabbs, x)
            else {
                continue;
            };
            // Element-volume-weighted quadrature contribution. The
            // Keast rule's weights are pre-normalised so `∑ w_q = 1`
            // over the *reference* tet; multiplying by `target_vol`
            // gives the world-space integration measure for this tet.
            let scale = q.weight * target_vol;
            // φ_target_i(x) = q.bary[i],  φ_source_j(x) = source_bary[j].
            for i in 0..4 {
                let phi_t = q.bary[i];
                if phi_t == 0.0 {
                    continue;
                }
                let row = tv[i] as usize;
                for j in 0..4 {
                    let phi_s = source_bary[j];
                    if phi_s == 0.0 {
                        continue;
                    }
                    let col = source_verts[j] as usize;
                    triplets.push((row, col, scale * phi_t * phi_s));
                }
            }
        }
    }

    BsrMatrix::from_scalar_triplets(nrows, ncols, &triplets)
}

/// World-space position of a quadrature point given its barycentric
/// coords on a tet.
fn bary_to_world(p: &[Vec3; 4], bary: &[Real; 4]) -> Vec3 {
    p[0] * bary[0] + p[1] * bary[1] + p[2] * bary[2] + p[3] * bary[3]
}

/// Absolute tet volume.
fn tet_volume(p: &[Vec3; 4]) -> Real {
    ((p[1] - p[0]).dot((p[2] - p[0]).cross(p[3] - p[0])) / 6.0).abs()
}

/// Locate the source tet containing world-space point `x` via a
/// brute-force barycentric scan. Returns the source tet's vertex
/// indices + the barycentric coordinates of `x` within it, or `None`
/// if `x` is not inside any source tet.
///
/// An AABB prefilter is applied before the (more expensive)
/// barycentric solve; the prefilter is generous by `BARY_EPS` so a
/// quadrature point lying exactly on a face won't be rejected.
fn locate_in_source(
    source: &FemMesh,
    aabbs: &[TetAabb],
    x: Vec3,
) -> Option<([u32; 4], [Real; 4])> {
    for aabb in aabbs {
        if x.x < aabb.min.x - BARY_EPS
            || x.y < aabb.min.y - BARY_EPS
            || x.z < aabb.min.z - BARY_EPS
            || x.x > aabb.max.x + BARY_EPS
            || x.y > aabb.max.y + BARY_EPS
            || x.z > aabb.max.z + BARY_EPS
        {
            continue;
        }
        let p = [
            source.vertices[aabb.verts[0] as usize],
            source.vertices[aabb.verts[1] as usize],
            source.vertices[aabb.verts[2] as usize],
            source.vertices[aabb.verts[3] as usize],
        ];
        let Some(weights) = barycentric_tet(&p, x) else {
            continue;
        };
        if weights.iter().all(|w| *w >= -BARY_EPS && *w <= 1.0 + BARY_EPS) {
            return Some((aabb.verts, weights));
        }
    }
    None
}

/// Barycentric coordinates of `x` w.r.t. tet `p`. Mirrors the helper
/// in `apic_coupling.rs` (which is module-private there).
fn barycentric_tet(p: &[Vec3; 4], x: Vec3) -> Option<[Real; 4]> {
    let d = Mat3::from_cols(p[1] - p[0], p[2] - p[0], p[3] - p[0]);
    let det = d.determinant();
    if det.abs() < 1.0e-18 {
        return None;
    }
    let inv = d.inverse();
    let lam = inv * (x - p[0]);
    Some([1.0 - lam.x - lam.y - lam.z, lam.x, lam.y, lam.z])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_projection_preserves_total_mass() {
        // Sanity: M_st · 1_source = M_tt · 1_target when the source
        // covers the target completely. We use two identical unit-cube
        // meshes at different resolutions so each target quadrature
        // point falls in *some* source tet.
        let source = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
        let target = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        let field = NonconformingField::new(&source, &target).unwrap();
        let ones_source = vec![1.0; field.source_dofs()];
        let rhs = matvec(field.coupling_matrix(), &ones_source);
        let ones_target = vec![1.0; field.target_dofs()];
        let mtt_one = matvec(field.target_mass_matrix(), &ones_target);
        // Both vectors are "lumped row sums" of the respective mass
        // matrices; for total mass on the unit cube each sums to 1.
        let s_rhs: Real = rhs.iter().sum();
        let s_mtt: Real = mtt_one.iter().sum();
        assert!(
            (s_rhs - 1.0).abs() < 1.0e-9,
            "M_st · 1 row sum {s_rhs} != 1.0"
        );
        assert!(
            (s_mtt - 1.0).abs() < 1.0e-12,
            "M_tt · 1 row sum {s_mtt} != 1.0"
        );
    }
}
