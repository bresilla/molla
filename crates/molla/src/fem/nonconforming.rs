//! Stage 47 v1/v2 — L² projection between non-matching tetrahedral meshes
//! + vector-valued projection + a simple penalty-contact force.
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
//! # v2 additions
//!
//! - **Vector-valued projection** — [`NonconformingField::project_vector`]
//!   pushes a `dim`-component (e.g. displacement) field through the same
//!   scalar `M_st` / `M_tt` once per component. Storage choice: we keep
//!   the **single scalar `BsrMatrix`** and loop over components rather
//!   than constructing a `B = 3` block matrix. The math is identical
//!   (the cross-mesh mass operator is **scalar** — it couples a target
//!   scalar dof to a source scalar dof; for an isotropic vector field
//!   the components decouple and a `B = 3` block matrix would just hold
//!   three identical scalar copies on its block diagonal with zero off-
//!   diagonal blocks, i.e. 3× the storage for the same numerics). The
//!   3× CG solves still reuse the cached SPD `M_tt` factorisation-free
//!   so the cost is `3 ·` the scalar path with zero re-assembly.
//! - **Penalty contact force** —
//!   [`NonconformingPenaltyContact::compute_penalty_forces`] computes a
//!   pair of nodal force vectors from a displacement gap
//!   `Δu = u_target − project_vector(u_source)`. The target receives
//!   `f_target = − k · M_tt · Δu` (consistent-mass weighting); the
//!   source receives `f_source = + k · M_stᵀ · Δu` (the transpose of
//!   the projection operator, so total force vanishes by Newton's third
//!   law). The relation `1ᵀ M_tt = 1ᵀ M_st` (both reproduce the target
//!   volume measure) makes the cancellation exact.
//!
//! # Out of scope (v2 — deferred to v3)
//!
//! - **Surface-only quadrature** — v2 still integrates over the target
//!   *volume*; surface-only quadrature requires extracting the contact
//!   surface mesh and integrating on that 2-manifold instead.
//! - **Full elasticity residual + Jacobian** for an implicit solve.
//! - **Contact pressure formulation** — Hertz / Signorini contact across
//!   non-matching surfaces wants an additional surface mass operator on
//!   the slave side, plus a Lagrange multiplier formulation.
//! - **Friction** at the contact interface.
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

    /// Project a vector-valued field defined on the source vertices onto
    /// the target vertices. The input is laid out **interleaved**:
    /// `u_source[i * dim + c]` is component `c` of the displacement at
    /// source vertex `i`. The output uses the same interleaved layout,
    /// length `target_dofs() * dim`.
    ///
    /// Implementation: the scalar L² projector commutes with the
    /// component-wise extraction (the cross-mesh mass matrix `M_st`
    /// doesn't couple components — see the v2 storage-choice note in
    /// the module header). So we deinterleave, run [`project_with`] on
    /// each scalar component, and reinterleave the results. The CG
    /// solves share the cached `M_tt` so the work is `dim ·` the
    /// scalar path with zero re-assembly.
    ///
    /// Panics on a length mismatch or `dim == 0`.
    #[must_use]
    pub fn project_vector(&self, u_source: &[Real], dim: usize) -> Vec<Real> {
        self.project_vector_with(u_source, dim, DEFAULT_CG_TOL, DEFAULT_CG_MAX_ITERS)
    }

    /// Same as [`project_vector`](Self::project_vector) but exposes the
    /// CG tolerance / iteration cap (matches [`project_with`] for the
    /// scalar path).
    #[must_use]
    pub fn project_vector_with(
        &self,
        u_source: &[Real],
        dim: usize,
        tol: Real,
        max_iters: usize,
    ) -> Vec<Real> {
        assert!(dim > 0, "NonconformingField::project_vector: dim must be > 0");
        assert_eq!(
            u_source.len(),
            self.source_dofs() * dim,
            "NonconformingField::project_vector: u_source length {} != source_dofs ({}) · dim ({})",
            u_source.len(),
            self.source_dofs(),
            dim,
        );
        let n_src = self.source_dofs();
        let n_tgt = self.target_dofs();
        let mut out = vec![0.0; n_tgt * dim];
        let mut comp = vec![0.0; n_src];
        for c in 0..dim {
            for i in 0..n_src {
                comp[i] = u_source[i * dim + c];
            }
            let projected = self.project_with(&comp, tol, max_iters);
            for i in 0..n_tgt {
                out[i * dim + c] = projected[i];
            }
        }
        out
    }
}

/// Stage 47 v2 — simple **penalty-spring** contact handler between two
/// non-matching meshes coupled through a [`NonconformingField`].
///
/// Conceptual model: the source mesh and the target mesh are treated as
/// two elastic bodies in (one-sided) contact. We measure the
/// displacement gap on the **target** mesh,
///
/// ```text
///   Δu_target = u_target − Π_st · u_source
/// ```
///
/// where `Π_st = M_tt⁻¹ M_st` is the cross-mesh projection operator
/// from [`NonconformingField::project_vector`]. The penalty contact
/// force on the target is a consistent-mass-weighted spring force,
///
/// ```text
///   f_target = − k · M_tt · Δu_target
/// ```
///
/// and the reaction on the source is the **transpose** of that load
/// pushed through the cross-mesh operator,
///
/// ```text
///   f_source = + k · M_stᵀ · Δu_target
/// ```
///
/// (one component at a time — see the storage-choice note in the
/// module header). Newton's third law is exact because
/// `1ᵀ M_tt 1 = 1ᵀ M_st 1` (both reproduce the target volume) — see
/// the [`tests::constant_projection_preserves_total_mass`] check in
/// v1.
///
/// **Caveat (v2 scope):** the integration is still over the **target
/// volume**, not just the contact surface — v3 will swap in a
/// surface-only quadrature. This means the v2 penalty is a "soft
/// volumetric coupling" suitable for thin overlapping bodies; for thick
/// volumetric contact, v3 should be used.
pub struct NonconformingPenaltyContact<'a> {
    /// The underlying projection operator (re-used as-is).
    pub field: NonconformingField<'a>,
    /// Spring stiffness `k`. The units are `[force] / [displacement]`
    /// per unit volume on the target side (consistent-mass weighting).
    pub stiffness: Real,
}

impl<'a> NonconformingPenaltyContact<'a> {
    /// Build a penalty-contact coupling on top of a fresh
    /// [`NonconformingField`].
    pub fn new(source: &'a FemMesh, target: &'a FemMesh, stiffness: Real) -> Result<Self> {
        Ok(Self {
            field: NonconformingField::new(source, target)?,
            stiffness,
        })
    }

    /// Wrap a caller-built [`NonconformingField`] (useful when the
    /// projector is already built for another purpose).
    pub fn from_field(field: NonconformingField<'a>, stiffness: Real) -> Self {
        Self { field, stiffness }
    }

    /// Compute the penalty-spring contact force on both meshes given the
    /// current displacement state.
    ///
    /// `u_source` is interleaved 3D displacement at source vertices
    /// (length `n_source · 3`); `u_target` is the same on the target
    /// side. Returns `(f_source, f_target)` — both interleaved 3D
    /// nodal forces — with `Σ f_source + Σ f_target = 0` (Newton's
    /// third law, exact to round-off because `M_stᵀ · 1 = M_tt · 1` on
    /// the target side, see [`tests::constant_projection_preserves_total_mass`]
    /// in v1).
    #[must_use]
    pub fn compute_penalty_forces(
        &self,
        u_source: &[Real],
        u_target: &[Real],
    ) -> (Vec<Real>, Vec<Real>) {
        self.compute_penalty_forces_with(
            u_source,
            u_target,
            DEFAULT_CG_TOL,
            DEFAULT_CG_MAX_ITERS,
        )
    }

    /// Same as [`compute_penalty_forces`](Self::compute_penalty_forces)
    /// but exposes the inner CG tolerance / iteration cap.
    #[must_use]
    pub fn compute_penalty_forces_with(
        &self,
        u_source: &[Real],
        u_target: &[Real],
        tol: Real,
        max_iters: usize,
    ) -> (Vec<Real>, Vec<Real>) {
        let dim = 3usize;
        let n_src = self.field.source_dofs();
        let n_tgt = self.field.target_dofs();
        assert_eq!(
            u_source.len(),
            n_src * dim,
            "compute_penalty_forces: u_source length {} != source_dofs ({}) · 3",
            u_source.len(),
            n_src,
        );
        assert_eq!(
            u_target.len(),
            n_tgt * dim,
            "compute_penalty_forces: u_target length {} != target_dofs ({}) · 3",
            u_target.len(),
            n_tgt,
        );

        // Per-component work. We don't call project_vector_with so we
        // can re-use the projected-source intermediate for both the
        // f_target (M_tt · Δu) and the f_source (M_stᵀ · Δu) products
        // without re-projecting.
        let mut f_source = vec![0.0; n_src * dim];
        let mut f_target = vec![0.0; n_tgt * dim];
        let mut u_src_c = vec![0.0; n_src];
        let mut u_tgt_c = vec![0.0; n_tgt];
        let mut delta_c = vec![0.0; n_tgt];

        for c in 0..dim {
            for i in 0..n_src {
                u_src_c[i] = u_source[i * dim + c];
            }
            for i in 0..n_tgt {
                u_tgt_c[i] = u_target[i * dim + c];
            }
            let projected = self.field.project_with(&u_src_c, tol, max_iters);
            for i in 0..n_tgt {
                delta_c[i] = u_tgt_c[i] - projected[i];
            }

            // f_target = − k · M_tt · Δu_target.
            let mtt_delta = matvec(&self.field.target_mass, &delta_c);
            for i in 0..n_tgt {
                f_target[i * dim + c] = -self.stiffness * mtt_delta[i];
            }

            // f_source = + k · M_stᵀ · Δu_target.
            let mst_t_delta = matvec_transpose(&self.field.coupling_matrix, &delta_c);
            assert_eq!(mst_t_delta.len(), n_src);
            for i in 0..n_src {
                f_source[i * dim + c] = self.stiffness * mst_t_delta[i];
            }
        }

        (f_source, f_target)
    }
}

// ─── Sparse helpers ─────────────────────────────────────────────────

/// Compute `y = Aᵀ · x` for a scalar (`B = 1`) [`BsrMatrix`]. Used by
/// [`NonconformingPenaltyContact`] to push the target-side gap back
/// through the cross-mesh operator's transpose (Newton's third law).
///
/// Output length is `a.ncols()`; the input length must equal `a.nrows()`.
/// Panics on a shape mismatch (a programming error — both meshes are
/// fixed at construction time).
fn matvec_transpose(a: &BsrMatrix, x: &[Real]) -> Vec<Real> {
    debug_assert_eq!(a.b, 1, "matvec_transpose: only B=1 supported");
    assert_eq!(
        x.len(),
        a.nrows(),
        "matvec_transpose: x length {} != A.nrows {}",
        x.len(),
        a.nrows()
    );
    let mut y = vec![0.0; a.ncols()];
    for r in 0..a.block_rows {
        let lo = a.row_offsets[r];
        let hi = a.row_offsets[r + 1];
        let xr = x[r];
        if xr == 0.0 {
            continue;
        }
        for slot in lo..hi {
            let c = a.col_idx[slot];
            y[c] += a.values[slot] * xr;
        }
    }
    y
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
