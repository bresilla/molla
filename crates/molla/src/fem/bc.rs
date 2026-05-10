//! Boundary conditions for the v1 FEM solvers.
//!
//! v1 supports:
//! - **Dirichlet** — fix `u(vertex) = value`. The "reduce" form
//!   (zero out the row, set the diagonal to 1, overwrite the RHS)
//!   is the textbook simplest enforcement; we don't symmetrise the
//!   column in v1, so the resulting matrix is no longer SPD on the
//!   constrained rows. CG still converges in practice on these
//!   diagonally-dominant Laplacian-like systems; symmetric reduction
//!   is a follow-up.
//! - **Neumann** — `n · ∇u = flux` over a face. v1 records the
//!   constraint but does not yet integrate the surface contribution
//!   into the RHS — the homogeneous (zero-flux) case happens
//!   automatically because P1 + bilinear weak form drops the
//!   boundary term when no Neumann is specified. The Neumann variant
//!   is wired so callers can store the data; full integration lands
//!   in Stage 36.1 alongside elasticity tractions.
//!
//! All BCs are addressed by global DoF index (= vertex index for
//! `FunctionSpace::P1Nodal`). The caller is responsible for picking
//! the right vertices given the mesh.

use crate::math::sparse::BsrMatrix;
use crate::math::Real;

/// A single boundary-condition entry.
///
/// The scalar variants (`Dirichlet`, `Neumann`) target one DoF per
/// vertex (the `P1Nodal` Stage 36 v1 case). The vector variants
/// (`DirichletVector`, `NeumannTraction`) target every component of a
/// vector field at a vertex / face (the Stage 36.1 elasticity case).
///
/// Both forms address the global DoF using the interleaved layout
/// `dof = vertex * dim + component`. The applier
/// ([`apply_dirichlet`]) infers `dim` from the matrix size + mesh
/// caller-side (the BC payload carries the per-component values).
#[derive(Clone, Debug)]
pub enum BoundaryCondition {
    /// Scalar Dirichlet: pin DoF at `vertex` to exactly `value`. For
    /// `P1Vector { dim }` this only constrains component 0; use
    /// `DirichletVector` to pin every component.
    Dirichlet { vertex: u32, value: Real },
    /// Vector Dirichlet: pin every component at `vertex` to the
    /// prescribed `values`. The slice length must match the function
    /// space's `dofs_per_vertex` (typically 2 or 3 for elasticity).
    DirichletVector { vertex: u32, values: Vec<Real> },
    /// Neumann: prescribe a scalar flux `flux` over the triangular
    /// `face` (3 vertex indices). Stage 36 v1 stores but does not
    /// integrate this (zero-flux is the homogeneous default).
    Neumann { face: [u32; 3], flux: Real },
    /// Neumann traction: prescribe a 3D traction vector over the
    /// triangular `face`. The applier integrates `∫ N · t dA` and
    /// adds the contribution to the RHS for every face vertex,
    /// component-by-component. Used by Stage 36.1 elasticity to push
    /// loads onto a free surface.
    NeumannTraction { face: [u32; 3], traction: [Real; 3] },
}

/// Apply Dirichlet BCs + Neumann tractions in-place.
///
/// **Dirichlet:** zero out the row, set the diagonal entry to 1,
/// overwrite the matching RHS entry with `value`. For
/// `DirichletVector`, the same operation is repeated for each
/// component DoF `vertex * dim + c`.
///
/// **Neumann tractions:** for `NeumannTraction { face, traction }`,
/// add `(area / 3) * traction[c]` to the RHS at each face vertex
/// `v`'s component DoF `v * dim + c`. This is the consistent (P1
/// surface integral) lumping for a constant traction over a flat
/// triangular face. `vertices` is the mesh vertex array (needed for
/// the face area).
///
/// The scalar `Neumann { face, flux }` variant is still a no-op —
/// zero flux is the homogeneous default and the v1 weak form
/// already drops the boundary integral.
///
/// `dim` is the per-vertex component count (1 for scalar fields,
/// `space.dofs_per_vertex()` for vector fields). Pass 1 for the
/// classical `P1Nodal` use-case.
pub fn apply_dirichlet(
    k: &mut BsrMatrix,
    f: &mut [Real],
    bcs: &[BoundaryCondition],
) {
    apply_bcs(k, f, bcs, &[], 1);
}

/// Vector-aware applier — same as [`apply_dirichlet`] but takes the
/// per-vertex DoF count and the mesh vertex positions (required for
/// `NeumannTraction` face-area integration).
pub fn apply_bcs(
    k: &mut BsrMatrix,
    f: &mut [Real],
    bcs: &[BoundaryCondition],
    vertices: &[crate::math::Vec3],
    dim: usize,
) {
    debug_assert_eq!(k.b, 1, "apply_bcs: only B = 1 supported in v1");
    debug_assert_eq!(k.nrows(), f.len(), "RHS length mismatch");
    debug_assert!(dim >= 1, "apply_bcs: dim must be >= 1");

    // Pass 1: integrate Neumann tractions into the RHS first. (Order
    // matters — Dirichlet overwrites RHS rows it pins, so any traction
    // spuriously assigned to a constrained DoF is silently dropped,
    // matching the textbook "Dirichlet wins" convention.)
    for bc in bcs {
        if let BoundaryCondition::NeumannTraction { face, traction } = bc {
            // Triangle face area in 3D.
            let a = vertices[face[0] as usize];
            let b = vertices[face[1] as usize];
            let c = vertices[face[2] as usize];
            let area = 0.5 * (b - a).cross(c - a).length();
            let third = area / 3.0;
            for v in face {
                let v = *v as usize;
                for ci in 0..dim.min(3) {
                    f[v * dim + ci] += third * traction[ci];
                }
            }
        }
    }

    // Pass 2: enforce Dirichlet rows.
    for bc in bcs {
        match bc {
            BoundaryCondition::Dirichlet { vertex, value } => {
                pin_row(k, f, *vertex as usize * dim, *value);
            }
            BoundaryCondition::DirichletVector { vertex, values } => {
                debug_assert_eq!(
                    values.len(),
                    dim,
                    "DirichletVector: values.len() ({}) != dim ({})",
                    values.len(),
                    dim,
                );
                let base = *vertex as usize * dim;
                for (ci, val) in values.iter().enumerate() {
                    pin_row(k, f, base + ci, *val);
                }
            }
            // Scalar Neumann: still a no-op — zero flux is the
            // homogeneous default.
            BoundaryCondition::Neumann { .. } => {}
            // Already handled in Pass 1.
            BoundaryCondition::NeumannTraction { .. } => {}
        }
    }
}

/// Internal helper — zero row `r`, set diagonal to 1, set `f[r] = value`.
fn pin_row(k: &mut BsrMatrix, f: &mut [Real], r: usize, value: Real) {
    let lo = k.row_offsets[r];
    let hi = k.row_offsets[r + 1];
    for slot in lo..hi {
        let c = k.col_idx[slot];
        k.values[slot] = if c == r { 1.0 } else { 0.0 };
    }
    f[r] = value;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::sparse::BsrMatrix;

    fn build_test_csr() -> BsrMatrix {
        // 4×4 dense-ish (full pattern), values just so we can check
        // they get zeroed correctly.
        let mut triplets = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                triplets.push((i, j, vec![(1 + i * 4 + j) as Real]));
            }
        }
        BsrMatrix::from_triplets(1, 4, 4, &triplets).unwrap()
    }

    #[test]
    fn apply_dirichlet_zeros_row_and_sets_diag() {
        let mut k = build_test_csr();
        let mut f = vec![10.0, 20.0, 30.0, 40.0];
        let bcs = [BoundaryCondition::Dirichlet {
            vertex: 2,
            value: 7.5,
        }];
        apply_dirichlet(&mut k, &mut f, &bcs);
        // Row 2 should be (0, 0, 1, 0).
        let lo = k.row_offsets[2];
        let hi = k.row_offsets[2 + 1];
        for slot in lo..hi {
            let c = k.col_idx[slot];
            let v = k.values[slot];
            if c == 2 {
                assert!((v - 1.0).abs() < 1e-15);
            } else {
                assert!(v.abs() < 1e-15);
            }
        }
        assert!((f[2] - 7.5).abs() < 1e-15);
        // Other rows untouched.
        assert!((f[0] - 10.0).abs() < 1e-15);
    }

    #[test]
    fn neumann_is_a_noop_in_v1() {
        let mut k = build_test_csr();
        let snapshot_values = k.values.clone();
        let mut f = vec![1.0, 2.0, 3.0, 4.0];
        let f_before = f.clone();
        let bcs = [BoundaryCondition::Neumann {
            face: [0, 1, 2],
            flux: 0.5,
        }];
        apply_dirichlet(&mut k, &mut f, &bcs);
        assert_eq!(k.values, snapshot_values);
        assert_eq!(f, f_before);
    }
}
