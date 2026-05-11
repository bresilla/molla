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
use super::bc::{apply_bcs, BoundaryCondition};
use super::element::{compute_element_elasticity_stiffness, compute_element_mass};
use super::function_space::FunctionSpace;
use super::mesh::{FemCell, FemMesh};
use super::quadrature::{tet_quadrature_5pt, tri3d_quadrature_6pt};
use crate::geometry::{boundary_facets, INVALID_VERT};

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
/// Stage 47 v3 — which domain `M_st` was integrated over.
///
/// `Volume` (the v1/v2 default) integrates over all target tets;
/// `Surface` integrates over the target's outer triangle facets
/// (returned by [`crate::geometry::boundary_facets`]) and is what
/// contact formulations want.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum IntegrationDomain {
    Volume,
    Surface,
}

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
    /// Which domain was integrated over — exposed via [`is_surface`].
    domain: IntegrationDomain,
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
            domain: IntegrationDomain::Volume,
        })
    }

    /// Build the **surface-only** cross-mesh L² projector — Stage 47 v3.
    ///
    /// Same operator definition as [`Self::new`] except `M_st_ij` is
    /// integrated over the **target mesh's outer surface** (the
    /// triangle facets returned by
    /// [`crate::geometry::boundary_facets`]) instead of the target
    /// volume. Quadrature: 6-point degree-4 Dunavant rule on every
    /// facet. Interior target DoFs receive zero contribution.
    pub fn new_surface(source: &'a FemMesh, target: &'a FemMesh) -> Result<Self> {
        let source_aabbs = build_tet_aabbs(source)?;
        let coupling_matrix = assemble_cross_mass_surface(source, target, &source_aabbs)?;
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
            domain: IntegrationDomain::Surface,
        })
    }

    /// `true` iff this operator integrates `M_st` over the target's
    /// outer surface (Stage 47 v3). `false` for the volumetric path.
    #[must_use]
    pub fn is_surface(&self) -> bool {
        matches!(self.domain, IntegrationDomain::Surface)
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

    /// Stage 47 v4 — assemble the **tangent stiffness blocks** of the
    /// penalty-contact term `Π_c(u) = ½ k · ‖u_t − M_tt⁻¹ M_st u_s‖²_Mtt`.
    ///
    /// Returns the two scalar (`B = 1`) blocks of the contact Jacobian:
    ///
    /// - **`K_ss = k · M_stᵀ · M_st`** (`n_s × n_s`, SPSD).
    /// - **`K_st = −k · M_stᵀ · M_tt`** (`n_s × n_t`).
    ///
    /// The full stacked contact tangent (sized `(n_s + n_t) × (n_s + n_t)`)
    /// has the structure
    ///
    /// ```text
    ///   K_contact = k · [  M_stᵀ M_st ,  − M_stᵀ M_tt ;
    ///                       − M_tt M_st ,    M_tt M_tt   ]
    /// ```
    ///
    /// The remaining two blocks are obtained from the returned ones by
    /// symmetry / transpose plus the cached `M_tt`:
    /// - `K_ts = K_stᵀ = −k · M_tt · M_st`,
    /// - `K_tt = k · M_tt · M_tt` (re-uses [`Self::field`]`.target_mass`).
    ///
    /// **Scalar storage**: per the v2 storage-choice note (the cross-mesh
    /// mass is component-decoupled), these are scalar (`B = 1`) blocks.
    /// The vector-valued contact tangent for `dim = 3` is the
    /// block-diagonal embedding of these scalar matrices into the
    /// interleaved `(vertex · 3 + comp)` layout — applied component-wise
    /// in [`solve_coupled_elasticity`].
    ///
    /// **Symmetry**: `K_ss = K_ssᵀ`, `K_tt = K_ttᵀ`, and `K_st = K_tsᵀ`,
    /// so the full contact tangent is symmetric. The penalty residual
    /// `R_c(u) = K_contact · u` makes the coupled elasticity tangent SPD
    /// for any `k > 0` (it adds a rank-deficient PSD perturbation to the
    /// SPD elasticity block — the rigid-body kernel of `K_s + K_t` may
    /// remain in the kernel of `K_contact` if both meshes translate
    /// equally, which is why Dirichlet BCs are still required on at
    /// least one body to make the stacked system SPD).
    pub fn compute_jacobian(&self) -> (BsrMatrix, BsrMatrix) {
        let mst = &self.field.coupling_matrix;
        let mtt = &self.field.target_mass;
        let k = self.stiffness;

        // K_ss = k · M_stᵀ · M_st (size n_s × n_s).
        let k_ss = mat_transpose_times_mat(mst, mst, k)
            .expect("compute_jacobian: K_ss assembly");
        // K_st = − k · M_stᵀ · M_tt (size n_s × n_t).
        let k_st = mat_transpose_times_mat(mst, mtt, -k)
            .expect("compute_jacobian: K_st assembly");
        (k_ss, k_st)
    }
}

// ─── Stage 47 v4 — coupled Newton-Raphson elasticity solve ──────────

/// Stage 47 v4 — solve the **stacked elasticity + non-conforming
/// penalty-contact** system via CG.
///
/// Conceptual model: two linearly-elastic bodies (`source` and `target`)
/// each with their own tangent stiffness `K_s`, `K_t`, external load
/// `f_source`, `f_target`, and Dirichlet BC sets. They are coupled
/// through the penalty-spring contact `K_contact` (built once from
/// `contact.compute_jacobian()` plus the cached `M_tt`).
///
/// The stacked linear system is
///
/// ```text
///   ┌ K_s + k·Kss   k·Kst   ┐ ┌ u_s ┐   ┌ f_s ┐
///   │                       │ │     │ = │     │
///   └ k·Kts        K_t + k·Ktt ┘ └ u_t ┘   └ f_t ┘
/// ```
///
/// where `Kss = M_stᵀ M_st`, `Kst = −M_stᵀ M_tt`, `Kts = Kstᵀ`,
/// `Ktt = M_tt M_tt`. With Dirichlet BCs applied on both bodies the
/// stacked operator is **symmetric positive definite** for any
/// `k_penalty ≥ 0`, so CG is the right solver (and the system is
/// linear — Newton-Raphson reduces to a single linear solve).
///
/// **Vector layout** (interleaved 3D for each body):
/// - `f_source` / `u_source` length: `source.vert_count() · 3`.
/// - `f_target` / `u_target` length: `target.vert_count() · 3`.
/// - The stacked DoF vector is `[u_source ; u_target]` (source first,
///   target second).
///
/// **BC handling**: `bcs_source` and `bcs_target` are applied **only**
/// to their respective elasticity blocks (`K_s`, `K_t`). The CG iterate
/// is kept in `du`-form on the pinned DoFs (pinned increments are zero,
/// so the prescribed values flow through unchanged from the seed).
/// `NeumannTraction` BCs are integrated into `f_*` *before* the solve.
///
/// **Args:**
/// - `k_source` (Lamé `λ`, `μ`) — source material parameters.
/// - `k_target` (Lamé `λ`, `μ`) — target material parameters.
/// - `contact` — the [`NonconformingPenaltyContact`] coupling.
/// - `f_source`, `f_target` — external load vectors (interleaved 3D).
/// - `bcs_source`, `bcs_target` — Dirichlet / Neumann BC lists.
/// - `k_penalty` — contact stiffness multiplier. **Overrides**
///   `contact.stiffness`; `k_penalty = 0` recovers two independent
///   elasticity solves to round-off.
/// - `max_iters`, `tol` — outer CG cap + relative residual tolerance.
///
/// Returns `(u_source, u_target)` — interleaved 3D displacement
/// vectors, length matching the input loads.
///
/// **Newton iterations**: linear elasticity ⇒ a single linear solve
/// is enough; the `max_iters` argument caps the **CG** (Krylov) inner
/// iterations, not Newton outer steps. The "Newton-Raphson" label in
/// the name is for consistency with v5 (which generalises to nonlinear
/// elasticity / Signorini hard constraint).
#[allow(clippy::too_many_arguments)]
pub fn solve_coupled_elasticity(
    source_mesh: &FemMesh,
    target_mesh: &FemMesh,
    k_source: (Real, Real),
    k_target: (Real, Real),
    contact: &NonconformingPenaltyContact<'_>,
    f_source: &[Real],
    f_target: &[Real],
    bcs_source: &[BoundaryCondition],
    bcs_target: &[BoundaryCondition],
    k_penalty: Real,
    max_iters: usize,
    tol: Real,
) -> (Vec<Real>, Vec<Real>) {
    let dim = 3usize;
    let n_s = source_mesh.vertices.len();
    let n_t = target_mesh.vertices.len();
    assert_eq!(
        f_source.len(),
        n_s * dim,
        "solve_coupled_elasticity: f_source length {} != n_source ({}) · 3",
        f_source.len(),
        n_s,
    );
    assert_eq!(
        f_target.len(),
        n_t * dim,
        "solve_coupled_elasticity: f_target length {} != n_target ({}) · 3",
        f_target.len(),
        n_t,
    );
    assert!(
        std::ptr::eq(contact.field.source as *const _, source_mesh as *const _),
        "solve_coupled_elasticity: contact source mesh must equal source_mesh"
    );
    assert!(
        std::ptr::eq(contact.field.target as *const _, target_mesh as *const _),
        "solve_coupled_elasticity: contact target mesh must equal target_mesh"
    );

    // ── 1. Assemble per-body elasticity stiffness blocks ────────────
    let (lam_s, mu_s) = k_source;
    let (lam_t, mu_t) = k_target;
    let space = FunctionSpace::P1Vector { dim };
    let k_s = assemble_global(source_mesh, space, |c, v| {
        compute_element_elasticity_stiffness(c, v, lam_s, mu_s)
    })
    .expect("solve_coupled_elasticity: source elasticity assembly");
    let k_t = assemble_global(target_mesh, space, |c, v| {
        compute_element_elasticity_stiffness(c, v, lam_t, mu_t)
    })
    .expect("solve_coupled_elasticity: target elasticity assembly");

    // ── 2. Pre-compute the contact block matrices (scalar) ──────────
    // K_ss = k_penalty · M_stᵀ · M_st (n_s × n_s).
    // K_st = -k_penalty · M_stᵀ · M_tt (n_s × n_t).
    // K_tt_contact = k_penalty · M_tt · M_tt (n_t × n_t).
    // K_ts = K_stᵀ (used by transpose matvec).
    let mst = contact.field.coupling_matrix.clone();
    let mtt = contact.field.target_mass.clone();
    let k_ss_scalar = mat_transpose_times_mat(&mst, &mst, k_penalty)
        .expect("solve_coupled_elasticity: K_ss");
    let k_st_scalar = mat_transpose_times_mat(&mst, &mtt, -k_penalty)
        .expect("solve_coupled_elasticity: K_st");
    let k_tt_scalar = mat_times_mat(&mtt, &mtt, k_penalty)
        .expect("solve_coupled_elasticity: K_tt_contact");

    // ── 3. Apply BCs on the elasticity blocks (column-elim + pin) ───
    // The BCs modify `K_s`, `f_s`, `K_t`, `f_t` in the usual symmetric
    // way (zero the row + column of every pinned DoF, set diagonal to 1,
    // RHS to the prescribed value). We snapshot the pinned indices +
    // their values so we can (a) absorb the contact block contribution
    // of the prescribed values into the RHS, and (b) zero the contact
    // block contribution at pinned rows/columns during the matvec.
    let mut k_s_red = k_s;
    let mut f_s_red = f_source.to_vec();
    apply_bcs(
        &mut k_s_red,
        &mut f_s_red,
        bcs_source,
        &source_mesh.vertices,
        dim,
    );
    let mut k_t_red = k_t;
    let mut f_t_red = f_target.to_vec();
    apply_bcs(
        &mut k_t_red,
        &mut f_t_red,
        bcs_target,
        &target_mesh.vertices,
        dim,
    );

    // Index the Dirichlet pin set for each body — flat per-DoF
    // (interleaved layout). `None` = free, `Some(v)` = pinned to `v`.
    let pinned_s = collect_pinned_dofs(bcs_source, n_s, dim);
    let pinned_t = collect_pinned_dofs(bcs_target, n_t, dim);

    // ── 4. Absorb the contact block's column shift on Dirichlet pins ─
    // The standard symmetric BC reduction (already applied to K_s/K_t)
    // shifts the RHS by `K_block · u_pin` for every pinned column.
    // We need the analogous shift for the K_contact block:
    //   f_red[i, free] -= K_contact[i, j] · u_pin[j]  for every pinned j.
    // We build `u_pin = [u_pin_s; u_pin_t]` (zero on free DoFs, pin
    // value on pinned DoFs) and matvec the contact tangent against it.
    let mut u_pin_s = vec![0.0; n_s * dim];
    for (i, v) in pinned_s.iter().enumerate() {
        if let Some(val) = v {
            u_pin_s[i] = *val;
        }
    }
    let mut u_pin_t = vec![0.0; n_t * dim];
    for (i, v) in pinned_t.iter().enumerate() {
        if let Some(val) = v {
            u_pin_t[i] = *val;
        }
    }
    // Compute K_contact · u_pin (block-wise, component-decoupled).
    let (mut shift_s, mut shift_t) = matvec_stacked_contact(
        &k_ss_scalar,
        &k_st_scalar,
        &k_tt_scalar,
        &u_pin_s,
        &u_pin_t,
        dim,
    );
    // Zero out the pinned rows of the shift (those rows get their
    // diagonal-1 BC contribution from K_block, not from K_contact —
    // we don't want to double-count the Dirichlet value).
    for (i, v) in pinned_s.iter().enumerate() {
        if v.is_some() {
            shift_s[i] = 0.0;
        }
    }
    for (i, v) in pinned_t.iter().enumerate() {
        if v.is_some() {
            shift_t[i] = 0.0;
        }
    }
    for i in 0..f_s_red.len() {
        f_s_red[i] -= shift_s[i];
    }
    for i in 0..f_t_red.len() {
        f_t_red[i] -= shift_t[i];
    }

    // ── 5. Build the stacked RHS + matvec closure ───────────────────
    let n_total = (n_s + n_t) * dim;
    let mut rhs = vec![0.0; n_total];
    rhs[..f_s_red.len()].copy_from_slice(&f_s_red);
    rhs[n_s * dim..n_s * dim + f_t_red.len()].copy_from_slice(&f_t_red);

    // Initial guess: seed pinned DoFs at their prescribed values, zero
    // elsewhere. This makes the converged CG iterate satisfy the
    // Dirichlet condition exactly (the pinned rows of the matvec map
    // `u_pin → u_pin` because of the diagonal-1 BC pin in K_block, and
    // the contact-row mask below).
    let mut x0 = vec![0.0; n_total];
    for (i, v) in pinned_s.iter().enumerate() {
        if let Some(val) = v {
            x0[i] = *val;
            // Also force the RHS at pinned rows to `val` (overwrite
            // apply_bcs' choice in case the BC list re-uses non-zero
            // values; apply_bcs already sets f_red[pin] = val, but the
            // earlier shift step zeroed the contact part so this is a
            // no-op verification step).
            rhs[i] = *val;
        }
    }
    for (i, v) in pinned_t.iter().enumerate() {
        if let Some(val) = v {
            x0[n_s * dim + i] = *val;
            rhs[n_s * dim + i] = *val;
        }
    }

    // Stacked matvec — built block-by-block:
    //   y_s = (K_s · u_s)        + (K_ss · u_s + K_st · u_t)_masked
    //   y_t = (K_t · u_t)        + (K_ts · u_s + K_tt · u_t)_masked
    // where _masked = "pinned rows zeroed".
    let pinned_s_clone = pinned_s.clone();
    let pinned_t_clone = pinned_t.clone();
    let op = move |x: &[Real]| -> Vec<Real> {
        debug_assert_eq!(x.len(), n_total);
        let (u_s, u_t) = x.split_at(n_s * dim);
        // Block-block (elasticity).
        let mut y_s = matvec(&k_s_red, u_s);
        let mut y_t = matvec(&k_t_red, u_t);
        // For the contact contribution we need to mask the input at
        // pinned DoFs (column elimination) — zeroing pinned components
        // of u_s, u_t before applying the contact blocks ensures the
        // pinned columns don't contribute to free rows.
        let mut u_s_masked = u_s.to_vec();
        for (i, v) in pinned_s_clone.iter().enumerate() {
            if v.is_some() {
                u_s_masked[i] = 0.0;
            }
        }
        let mut u_t_masked = u_t.to_vec();
        for (i, v) in pinned_t_clone.iter().enumerate() {
            if v.is_some() {
                u_t_masked[i] = 0.0;
            }
        }
        let (cs, ct) = matvec_stacked_contact(
            &k_ss_scalar,
            &k_st_scalar,
            &k_tt_scalar,
            &u_s_masked,
            &u_t_masked,
            dim,
        );
        // Mask pinned output rows: K_block already enforces
        // `y[pin] = u[pin]` via its diagonal-1 pin, so the contact
        // contribution at pinned rows must be zeroed.
        for i in 0..(n_s * dim) {
            if pinned_s_clone[i].is_some() {
                continue;
            }
            y_s[i] += cs[i];
        }
        for i in 0..(n_t * dim) {
            if pinned_t_clone[i].is_some() {
                continue;
            }
            y_t[i] += ct[i];
        }

        // Stack.
        let mut y = Vec::with_capacity(n_total);
        y.extend_from_slice(&y_s);
        y.extend_from_slice(&y_t);
        y
    };

    // ── 6. Solve the stacked system via CG ──────────────────────────
    let result = cg_solve(op, &rhs, &x0, tol, max_iters);
    if !result.converged(tol) {
        tracing::warn!(
            "solve_coupled_elasticity: CG did not converge — \
             iters = {}/{}, residual = {:.3e} (tol = {:.3e})",
            result.iters,
            max_iters,
            result.residual,
            tol,
        );
    }
    tracing::debug!(
        "solve_coupled_elasticity: CG iters = {}, residual = {:.3e}",
        result.iters,
        result.residual,
    );
    let mut u_s = vec![0.0; n_s * dim];
    let mut u_t = vec![0.0; n_t * dim];
    u_s.copy_from_slice(&result.x[..n_s * dim]);
    u_t.copy_from_slice(&result.x[n_s * dim..]);
    (u_s, u_t)
}

/// Apply the **stacked contact matvec** to an interleaved 3D
/// displacement pair `(u_s, u_t)`, component-by-component (the contact
/// blocks are scalar / component-decoupled — see the v2 storage-choice
/// note in the module header).
///
/// Returns `(y_s, y_t)` where
/// ```text
///   y_s = k_ss · u_s + k_st · u_t
///   y_t = k_stᵀ · u_s + k_tt · u_t
/// ```
/// `k_ss`, `k_st`, `k_tt` are the **scalar** (`B = 1`) blocks built by
/// [`NonconformingPenaltyContact::compute_jacobian`] (or the inline
/// equivalents in [`solve_coupled_elasticity`]); `dim` is the per-vertex
/// component count (3 for elasticity).
fn matvec_stacked_contact(
    k_ss: &BsrMatrix,
    k_st: &BsrMatrix,
    k_tt: &BsrMatrix,
    u_s: &[Real],
    u_t: &[Real],
    dim: usize,
) -> (Vec<Real>, Vec<Real>) {
    let n_s = k_ss.nrows();
    let n_t = k_tt.nrows();
    debug_assert_eq!(u_s.len(), n_s * dim);
    debug_assert_eq!(u_t.len(), n_t * dim);
    debug_assert_eq!(k_st.nrows(), n_s);
    debug_assert_eq!(k_st.ncols(), n_t);

    let mut y_s = vec![0.0; n_s * dim];
    let mut y_t = vec![0.0; n_t * dim];
    let mut comp_s = vec![0.0; n_s];
    let mut comp_t = vec![0.0; n_t];

    for c in 0..dim {
        for i in 0..n_s {
            comp_s[i] = u_s[i * dim + c];
        }
        for i in 0..n_t {
            comp_t[i] = u_t[i * dim + c];
        }
        // y_s_c = K_ss · u_s_c + K_st · u_t_c.
        let a = matvec(k_ss, &comp_s);
        let b = matvec(k_st, &comp_t);
        for i in 0..n_s {
            y_s[i * dim + c] = a[i] + b[i];
        }
        // y_t_c = K_stᵀ · u_s_c + K_tt · u_t_c.
        let at = matvec_transpose(k_st, &comp_s);
        let d = matvec(k_tt, &comp_t);
        for i in 0..n_t {
            y_t[i * dim + c] = at[i] + d[i];
        }
    }
    (y_s, y_t)
}

/// Collect Dirichlet pins from a BC list into a flat
/// `Vec<Option<Real>>` of length `n_verts · dim`. `None` = free DoF,
/// `Some(value)` = pinned. Handles both `Dirichlet` (component 0 only)
/// and `DirichletVector` (every component). Neumann variants are
/// silently ignored (they affect the RHS, not the matrix pattern).
fn collect_pinned_dofs(
    bcs: &[BoundaryCondition],
    n_verts: usize,
    dim: usize,
) -> Vec<Option<Real>> {
    let mut out = vec![None; n_verts * dim];
    for bc in bcs {
        match bc {
            BoundaryCondition::Dirichlet { vertex, value } => {
                let base = (*vertex as usize) * dim;
                if base < out.len() {
                    out[base] = Some(*value);
                }
            }
            BoundaryCondition::DirichletVector { vertex, values } => {
                let base = (*vertex as usize) * dim;
                for (ci, val) in values.iter().enumerate() {
                    if base + ci < out.len() {
                        out[base + ci] = Some(*val);
                    }
                }
            }
            BoundaryCondition::Neumann { .. }
            | BoundaryCondition::NeumannTraction { .. }
            | BoundaryCondition::WeakDirichletDG { .. } => {}
        }
    }
    out
}

/// Compute `C = α · Aᵀ · B` as a scalar (`B = 1`) [`BsrMatrix`].
///
/// Shapes: `A` is `m × p`, `B` is `m × q`, output is `p × q` (the
/// transposition swaps `A`'s row/column roles). Used by
/// [`NonconformingPenaltyContact::compute_jacobian`] to build
/// `k · M_stᵀ · M_st` and `−k · M_stᵀ · M_tt`.
///
/// Implementation: walk every non-zero entry of `A` as triplets
/// `(r, c, v)`, then for each entry in row `r` of `B` (call it
/// `(r, c2, v2)`) accumulate `α · v · v2` into `C[c, c2]`. Final
/// triplets are coalesced via [`BsrMatrix::from_scalar_triplets`] so
/// repeated `(row, col)` pairs sum.
///
/// Complexity: `O(nnz(A) · avg_row_nnz(B))` — fine for the
/// non-conforming contact scale (target meshes have a few thousand
/// vertices in practice).
fn mat_transpose_times_mat(a: &BsrMatrix, b: &BsrMatrix, alpha: Real) -> Result<BsrMatrix> {
    debug_assert_eq!(a.b, 1, "mat_transpose_times_mat: only B = 1 supported");
    debug_assert_eq!(b.b, 1, "mat_transpose_times_mat: only B = 1 supported");
    assert_eq!(
        a.nrows(),
        b.nrows(),
        "mat_transpose_times_mat: row mismatch {} vs {}",
        a.nrows(),
        b.nrows()
    );
    let p = a.ncols();
    let q = b.ncols();
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for r in 0..a.block_rows {
        let a_lo = a.row_offsets[r];
        let a_hi = a.row_offsets[r + 1];
        let b_lo = b.row_offsets[r];
        let b_hi = b.row_offsets[r + 1];
        if a_lo == a_hi || b_lo == b_hi {
            continue;
        }
        for a_slot in a_lo..a_hi {
            let c_a = a.col_idx[a_slot];
            let v_a = a.values[a_slot];
            if v_a == 0.0 {
                continue;
            }
            for b_slot in b_lo..b_hi {
                let c_b = b.col_idx[b_slot];
                let v_b = b.values[b_slot];
                if v_b == 0.0 {
                    continue;
                }
                triplets.push((c_a, c_b, alpha * v_a * v_b));
            }
        }
    }
    BsrMatrix::from_scalar_triplets(p, q, &triplets)
}

/// Compute `C = α · A · B` as a scalar (`B = 1`) [`BsrMatrix`].
///
/// Shapes: `A` is `m × p`, `B` is `p × q`, output is `m × q`. Used by
/// [`solve_coupled_elasticity`] to build the contact `K_tt = k · M_tt²`
/// block (target-target coupling).
fn mat_times_mat(a: &BsrMatrix, b: &BsrMatrix, alpha: Real) -> Result<BsrMatrix> {
    debug_assert_eq!(a.b, 1, "mat_times_mat: only B = 1 supported");
    debug_assert_eq!(b.b, 1, "mat_times_mat: only B = 1 supported");
    assert_eq!(
        a.ncols(),
        b.nrows(),
        "mat_times_mat: inner dim mismatch {} vs {}",
        a.ncols(),
        b.nrows()
    );
    let m = a.nrows();
    let q = b.ncols();
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for i in 0..a.block_rows {
        let a_lo = a.row_offsets[i];
        let a_hi = a.row_offsets[i + 1];
        for a_slot in a_lo..a_hi {
            let k = a.col_idx[a_slot];
            let v_a = a.values[a_slot];
            if v_a == 0.0 {
                continue;
            }
            let b_lo = b.row_offsets[k];
            let b_hi = b.row_offsets[k + 1];
            for b_slot in b_lo..b_hi {
                let j = b.col_idx[b_slot];
                let v_b = b.values[b_slot];
                if v_b == 0.0 {
                    continue;
                }
                triplets.push((i, j, alpha * v_a * v_b));
            }
        }
    }
    BsrMatrix::from_scalar_triplets(m, q, &triplets)
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

/// Stage 47 v3 — surface-only variant of [`assemble_cross_mass`].
///
/// Integrates `M_st_ij` over the target mesh's outer triangle facets
/// (from [`boundary_facets`]) instead of the target volume. 2D-tri
/// boundary edges (encoded as `[a, b, INVALID_VERT]`) are skipped —
/// 1D edge quadrature is deferred.
fn assemble_cross_mass_surface(
    source: &FemMesh,
    target: &FemMesh,
    source_aabbs: &[TetAabb],
) -> Result<BsrMatrix> {
    let nrows = target.vertices.len();
    let ncols = source.vertices.len();

    let facets = boundary_facets(target);
    if facets.is_empty() {
        return Err(Error::Fem(
            "NonconformingField::new_surface: target mesh has no outer facets".into(),
        ));
    }

    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    let mut n_tri_facets = 0usize;

    for facet in &facets {
        if facet[2] == INVALID_VERT {
            continue;
        }
        n_tri_facets += 1;
        let tp = [
            target.vertices[facet[0] as usize],
            target.vertices[facet[1] as usize],
            target.vertices[facet[2] as usize],
        ];
        let rule = tri3d_quadrature_6pt(&tp);
        for q in &rule {
            let Some((source_verts, source_bary)) =
                locate_in_source(source, source_aabbs, q.point)
            else {
                continue;
            };
            let scale = q.weight;
            for k in 0..3 {
                let phi_t = q.bary[k];
                if phi_t == 0.0 {
                    continue;
                }
                let row = facet[k] as usize;
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

    if n_tri_facets == 0 {
        return Err(Error::Fem(
            "NonconformingField::new_surface: target mesh has no triangle facets \
             (2D tri-only target meshes are not supported in v3)"
                .into(),
        ));
    }

    BsrMatrix::from_scalar_triplets(nrows, ncols, &triplets)
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
