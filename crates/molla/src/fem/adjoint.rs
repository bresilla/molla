//! Stage 36.4 — Adjoint / differentiable FEM.
//!
//! Reverse-mode through the **linear-elasticity** assembly + solve.
//! Given a forward solve `K(p) · u = f(p)` with design parameters `p`
//! (Young's modulus, applied traction, …) and a scalar loss `J(u, p)`,
//! the adjoint method computes `∂J/∂p` with one extra linear solve of
//! the same matrix — independent of the dimensionality of `p`.
//!
//! # Algorithm (textbook)
//!
//! 1. Forward: `u = K(p)⁻¹ f(p)`.
//! 2. Adjoint: solve `Kᵀ · λ = ∂J/∂u` (for SPD `K`, `Kᵀ = K`, so we
//!    reuse the same matrix and the same `cg_solve` closure).
//! 3. Gradient:
//!    `∂J/∂p = ∂J/∂p|_explicit + λᵀ · (∂f/∂p − ∂K/∂p · u)`.
//!
//! For linear elasticity the per-element stiffness scales **linearly**
//! with Young's modulus `E` (`λ = E·ν / ((1+ν)(1−2ν))`,
//! `μ = E / (2(1+ν))` — both proportional to `E`), so
//! `∂K/∂E = K / E` globally and the gradient collapses to
//! `∂J/∂E = −λᵀ · K · u / E` for explicitly load-independent loss
//! functions.
//!
//! For the applied **traction vector** `t` over a face, the consistent
//! P1 lumping `f_v += (area / 3) · t` per face vertex, component-by-
//! component, gives `∂f/∂t_c = (area / 3) · e_c` summed over the three
//! face vertex DoFs. The gradient is therefore
//! `∂J/∂t_c = (area / 3) · ∑_{v ∈ face} λ_{v · dim + c}`.
//!
//! # Boundary-condition handling
//!
//! The adjoint system inherits the same Dirichlet BC reduction as the
//! forward solve (zero out the row, set the diagonal to 1, RHS = 0 on
//! constrained rows). The seed `∂J/∂u` may have nonzero entries on
//! constrained rows (e.g. when the loss reads tip-deflection at a free
//! vertex), but for any pinned DoF the row reduction forces the
//! corresponding adjoint entry to zero — which is correct: a pinned
//! DoF can't contribute to the loss because its forward value is
//! prescribed, not computed.
//!
//! # Scope (v1 of Stage 36.4)
//!
//! - Linear elasticity only. SVK / plasticity adjoints (Stage 36.4.2)
//!   require differentiating through the Newton iteration; deferred.
//! - SIMPLE / Navier-Stokes adjoint (Stage 36.4.1) requires
//!   differentiating the pressure-velocity coupling; deferred.
//! - CPU only. The adjoint solve is structurally identical to the
//!   forward solve, so the GPU port (Stage 36.5.x) plugs the same way.

use crate::math::iterative::cg_solve;
use crate::math::sparse::BsrMatrix;
use crate::math::{Real, Vec3};

use super::assembly::matvec;
use super::bc::{apply_bcs, BoundaryCondition};
use super::mesh::{FemCell, FemMesh};

/// Default tolerance for the adjoint CG solve. Matches the forward
/// default; gradients computed at this tolerance agree with central
/// finite differences to ~1e-3 relative on the v1 contract tests.
pub const DEFAULT_ADJOINT_TOL: Real = 1.0e-10;

/// Default iteration cap for the adjoint CG solve.
pub const DEFAULT_ADJOINT_MAX_ITERS: usize = 50_000;

/// Solve the adjoint system `Kᵀ · λ = ∂J/∂u` with the same Dirichlet
/// reduction as the forward solve.
///
/// `k` is the **un-reduced** stiffness matrix (the full assembled `K`
/// before BCs); the routine clones it and applies BCs internally so the
/// caller doesn't have to bookkeep two copies. `dj_du` is the seed —
/// the partial derivative of the loss w.r.t. the displacement field
/// (length = `n_dof`). `bcs` are the same Dirichlet / vector-Dirichlet
/// constraints used by the forward solve. `vertices` + `dim` are
/// passed through to [`apply_bcs`] (the routine ignores any
/// `NeumannTraction` BC entries — the adjoint RHS is `dj_du`, not the
/// physical load).
///
/// For symmetric `K` (linear elasticity, Laplacian, …), `Kᵀ = K` so
/// the routine never explicitly transposes. If you need a non-
/// symmetric adjoint in a follow-up stage, swap the inner solver for
/// `bicgstab_solve` and pass the transposed matrix.
#[must_use]
pub fn solve_adjoint(
    k: &BsrMatrix,
    dj_du: &[Real],
    bcs: &[BoundaryCondition],
    vertices: &[Vec3],
    dim: usize,
) -> Vec<Real> {
    solve_adjoint_with(
        k,
        dj_du,
        bcs,
        vertices,
        dim,
        DEFAULT_ADJOINT_MAX_ITERS,
        DEFAULT_ADJOINT_TOL,
    )
}

/// Variant of [`solve_adjoint`] that exposes the CG iteration cap +
/// tolerance for callers that want tighter / looser solves.
#[must_use]
pub fn solve_adjoint_with(
    k: &BsrMatrix,
    dj_du: &[Real],
    bcs: &[BoundaryCondition],
    vertices: &[Vec3],
    dim: usize,
    max_iters: usize,
    tol: Real,
) -> Vec<Real> {
    debug_assert_eq!(k.nrows(), dj_du.len(), "solve_adjoint: dj_du length mismatch");
    let mut k_red = k.clone();
    let mut rhs = dj_du.to_vec();
    // Zero RHS on Dirichlet rows + identity-row in K. Strip any
    // `NeumannTraction` entries first — those would inject the physical
    // traction into the adjoint RHS, which is wrong (the adjoint RHS is
    // `∂J/∂u`, not the physical load).
    let pin_only: Vec<BoundaryCondition> = bcs
        .iter()
        .filter_map(|bc| match bc {
            BoundaryCondition::Dirichlet { vertex, .. } => Some(BoundaryCondition::Dirichlet {
                vertex: *vertex,
                value: 0.0,
            }),
            BoundaryCondition::DirichletVector { vertex, values } => {
                Some(BoundaryCondition::DirichletVector {
                    vertex: *vertex,
                    values: vec![0.0; values.len()],
                })
            }
            BoundaryCondition::Neumann { .. } | BoundaryCondition::NeumannTraction { .. } => None,
            // DG Nitsche BCs are irrelevant to the CG adjoint solve.
            BoundaryCondition::WeakDirichletDG { .. } => None,
        })
        .collect();
    apply_bcs(&mut k_red, &mut rhs, &pin_only, vertices, dim);
    cg_solve(|x| matvec(&k_red, x), &rhs, &vec![0.0; rhs.len()], tol, max_iters).x
}

/// Compute `∂J/∂E` for **linear** isotropic elasticity, given the
/// forward displacement `u` and the adjoint `λ`.
///
/// Derivation: the global stiffness `K(E)` is linear in `E`
/// (Lamé `λ_l = E ν / ((1+ν)(1−2ν))` and `μ = E / (2(1+ν))` are both
/// proportional to `E`), so `∂K/∂E = K / E`. The adjoint identity
/// gives `∂J/∂E = −λᵀ · (∂K/∂E) · u = −(λᵀ · K · u) / E`.
///
/// The implementation reuses the **un-reduced** assembled `K` (passed
/// indirectly by re-assembling from `mesh` + `vertices` + `(λ_l, μ)`
/// would be wasteful; we rebuild only the local matrices and contract
/// with the global `u` / `λ` element-by-element). This avoids the
/// global sparse matvec entirely and keeps the gradient computation
/// `O(n_cells)`.
///
/// Caveats:
/// - Assumes `dim == 3` (tet) or `dim == 2` (tri plane-strain) — the
///   per-cell elasticity kernel handles both.
/// - Assumes the loss `J` has no explicit `E` dependence (which is the
///   common case: compliance, tip deflection, target-displacement
///   norms all read `u` only).
#[must_use]
pub fn compute_grad_youngs(
    mesh: &FemMesh,
    vertices: &[Vec3],
    u: &[Real],
    lambda: &[Real],
    poisson: Real,
    youngs: Real,
) -> Real {
    debug_assert_eq!(u.len(), lambda.len(), "compute_grad_youngs: u/λ length mismatch");
    debug_assert!(youngs > 0.0, "compute_grad_youngs: E must be > 0");

    // Per-cell contribution: λᵀ_local · K_local · u_local; sum then
    // divide by E and negate.
    let lam_l = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
    let mu_l = youngs / (2.0 * (1.0 + poisson));
    let mut accum: Real = 0.0;
    for cell in &mesh.cells {
        let dim = match cell {
            FemCell::Tri(_) => 2,
            FemCell::Tet(_) => 3,
        };
        let k_local = super::element::compute_element_elasticity_stiffness(
            cell, vertices, lam_l, mu_l,
        );
        let verts = cell.verts();
        let nv = verts.len();
        let n_local = nv * dim;
        // Gather local u, λ in interleaved layout.
        let mut u_l = vec![0.0; n_local];
        let mut l_l = vec![0.0; n_local];
        for (i, &v) in verts.iter().enumerate() {
            let base = (v as usize) * dim;
            for c in 0..dim {
                let local_idx = i * dim + c;
                u_l[local_idx] = u[base + c];
                l_l[local_idx] = lambda[base + c];
            }
        }
        // λᵀ_local · K_local · u_local.
        for r in 0..n_local {
            let mut row_dot: Real = 0.0;
            for c in 0..n_local {
                row_dot += k_local[r][c] * u_l[c];
            }
            accum += l_l[r] * row_dot;
        }
    }
    -accum / youngs
}

/// Compute `∂J/∂t` for a **uniform traction vector** `t` applied to a
/// triangular `face` (3 vertex indices). Returns the 3-component
/// gradient.
///
/// Derivation: the consistent P1 lumping makes the contribution to the
/// global RHS `f_{v · dim + c} += (area / 3) · t_c` for every
/// `v ∈ face` and every component `c`. So `∂f/∂t_c = (area / 3)`
/// summed over the three face vertices, projected on component `c`.
/// The adjoint identity (with no explicit `t` dependence in `J` and
/// `K` independent of `t`) gives
/// `∂J/∂t_c = λᵀ · ∂f/∂t_c = (area / 3) · ∑_v λ_{v · dim + c}`.
///
/// Triangle face area is computed in 3D as `½ · |(b − a) × (c − a)|`.
///
/// `dim` is the per-vertex DoF count of the displacement field
/// (typically 3 for tets, 2 for plane-strain tris). Components beyond
/// `dim` are returned as 0.
#[must_use]
pub fn compute_grad_load(
    vertices: &[Vec3],
    lambda: &[Real],
    traction_face: [u32; 3],
    dim: usize,
) -> [Real; 3] {
    let a = vertices[traction_face[0] as usize];
    let b = vertices[traction_face[1] as usize];
    let c = vertices[traction_face[2] as usize];
    let area = 0.5 * (b - a).cross(c - a).length();
    let third = area / 3.0;
    let mut sums = [0.0_f64; 3];
    for &v in &traction_face {
        let base = (v as usize) * dim;
        for ci in 0..dim.min(3) {
            sums[ci] += lambda[base + ci];
        }
    }
    [
        third * sums[0],
        third * sums[1],
        third * sums[2],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem::assembly::assemble_global;
    use crate::fem::element::compute_element_elasticity_stiffness;
    use crate::fem::function_space::FunctionSpace;

    fn lame(youngs: Real, poisson: Real) -> (Real, Real) {
        let mu = youngs / (2.0 * (1.0 + poisson));
        let lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson));
        (lam, mu)
    }

    #[test]
    fn adjoint_self_test_kt_lambda_equals_dj_du() {
        // Build a tiny 1×1×1 tet beam, solve forward, build a random-
        // looking dJ/du seed, run adjoint, and verify Kᵀ·λ ≈ dj_du
        // with the constrained rows zeroed (since we pin them).
        let mesh = FemMesh::tet_grid_3d(2, 1, 1, 2.0, 1.0, 1.0);
        let dim = 3usize;
        let space = FunctionSpace::P1Vector { dim };
        let (lam_l, mu_l) = lame(1.0e6, 0.3);
        let k = assemble_global(&mesh, space, |c, v| {
            compute_element_elasticity_stiffness(c, v, lam_l, mu_l)
        })
        .unwrap();
        let mut bcs = Vec::new();
        for (i, p) in mesh.vertices.iter().enumerate() {
            if p.x.abs() < 1e-9 {
                bcs.push(BoundaryCondition::DirichletVector {
                    vertex: i as u32,
                    values: vec![0.0; dim],
                });
            }
        }
        let n = mesh.vert_count() * dim;
        let mut dj_du = vec![0.0; n];
        // Pick a tip-vertex z-DoF as the seed (compliance-like, but
        // local).
        let tip = (mesh.vert_count() - 1) * dim + 2;
        dj_du[tip] = 1.0;
        let lam_a = solve_adjoint(&k, &dj_du, &bcs, &mesh.vertices, dim);
        // Verify K · λ ≈ dj_du except on Dirichlet rows.
        let kl = matvec(&k, &lam_a);
        // Build a mask of free DoFs.
        let mut free = vec![true; n];
        for bc in &bcs {
            if let BoundaryCondition::DirichletVector { vertex, .. } = bc {
                let base = (*vertex as usize) * dim;
                for c in 0..dim {
                    free[base + c] = false;
                }
            }
        }
        for i in 0..n {
            if free[i] {
                let err = (kl[i] - dj_du[i]).abs();
                // Relax slightly because we use the un-reduced matrix
                // for the residual check; the reduced + un-reduced
                // matrices agree on free rows up to floating noise.
                assert!(err < 1e-4, "K·λ != dj_du at free DoF {i}: err = {err}");
            }
        }
    }
}
