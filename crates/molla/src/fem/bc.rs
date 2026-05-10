//! Boundary conditions for the v1 FEM solvers.
//!
//! v1 supports:
//! - **Dirichlet** — fix `u(vertex) = value`. The "reduce" form is
//!   symmetric: for every pinned DoF `r` with value `v`, the column
//!   `K[:, r]` is also eliminated by shifting `f[i] -= K[i, r] · v`
//!   for every non-pinned row `i` and zeroing `K[i, r]`. The pinned
//!   row is then `K[r, r] = 1, K[r, j≠r] = 0, f[r] = v`. The
//!   resulting reduced matrix is symmetric (and SPD when the parent
//!   is) so CG retains its convergence guarantee — required for the
//!   adjoint solve and for non-zero Dirichlet values that previously
//!   silently relied on row-only elimination.
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
/// **Dirichlet:** symmetric reduction — for every pinned DoF `r`
/// with value `v`, the column `K[:, r]` is eliminated by shifting
/// `f[i] -= K[i, r] · v` for every non-pinned row `i` and zeroing
/// `K[i, r]`, then the row `K[r, :]` is zeroed except for
/// `K[r, r] = 1` and `f[r] = v`. The reduced matrix retains the
/// symmetry of the parent — CG (which assumes self-adjoint) is safe
/// on the result, and the same RHS shift makes the solution
/// independent of whether `v` is zero or non-zero. For
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

    // Pass 2: collect every pinned DoF + its value into a flat array.
    // We need the full pin set up-front so the column-elimination
    // pass can shift the RHS by the right Dirichlet values in one
    // sweep of the matrix.
    let n = k.nrows();
    let mut pinned: Vec<Option<Real>> = vec![None; n];
    for bc in bcs {
        match bc {
            BoundaryCondition::Dirichlet { vertex, value } => {
                pinned[*vertex as usize * dim] = Some(*value);
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
                    pinned[base + ci] = Some(*val);
                }
            }
            BoundaryCondition::Neumann { .. } => {}
            BoundaryCondition::NeumannTraction { .. } => {}
        }
    }

    // Pass 3: symmetric column elimination — for every non-pinned
    // row i, look for entries `K[i, j]` where j is pinned; shift
    // `f[i] -= K[i, j] · value[j]` and zero the matrix entry. This
    // restores the symmetry the textbook row-only "pin" breaks.
    for i in 0..n {
        if pinned[i].is_some() {
            continue;
        }
        let lo = k.row_offsets[i];
        let hi = k.row_offsets[i + 1];
        for slot in lo..hi {
            let j = k.col_idx[slot];
            if let Some(val_j) = pinned[j] {
                f[i] -= k.values[slot] * val_j;
                k.values[slot] = 0.0;
            }
        }
    }

    // Pass 4: pin each constrained row — zero off-diagonals, set
    // diagonal to 1, set the RHS to the prescribed value.
    for r in 0..n {
        let Some(value) = pinned[r] else { continue };
        pin_row(k, f, r, value);
    }
}

/// Internal helper — zero row `r`, set diagonal to 1, set `f[r] = value`.
/// The caller must already have done the symmetric column elimination
/// (see `apply_bcs` pass 3) before invoking this; otherwise the
/// reduced matrix loses symmetry.
fn pin_row(k: &mut BsrMatrix, f: &mut [Real], r: usize, value: Real) {
    let lo = k.row_offsets[r];
    let hi = k.row_offsets[r + 1];
    let mut diag_found = false;
    for slot in lo..hi {
        let c = k.col_idx[slot];
        if c == r {
            k.values[slot] = 1.0;
            diag_found = true;
        } else {
            k.values[slot] = 0.0;
        }
    }
    debug_assert!(
        diag_found,
        "pin_row({r}): no diagonal entry in the CSR pattern — \
         row has no assembled connectivity, system is singular",
    );
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
        // With symmetric reduction the column is also eliminated, so
        // every other row's RHS absorbs `K[i, 2] · 7.5`. For the test
        // matrix `K[i, j] = 1 + i*4 + j`, that gives `K[0, 2] = 3`,
        // so `f[0]` shifts from 10 → 10 − 3·7.5 = −12.5.
        assert!((f[0] - (10.0 - 3.0 * 7.5)).abs() < 1e-15, "f[0] = {}", f[0]);
    }

    #[test]
    fn dirichlet_symmetric_reduction_preserves_symmetry() {
        // SPD matrix: 4×4 tridiagonal Laplacian + identity boost.
        let mut triplets = Vec::new();
        for i in 0..4usize {
            triplets.push((i, i, vec![3.0_f64]));
            if i + 1 < 4 {
                triplets.push((i, i + 1, vec![-1.0_f64]));
                triplets.push((i + 1, i, vec![-1.0_f64]));
            }
        }
        let mut k = BsrMatrix::from_triplets(1, 4, 4, &triplets).unwrap();
        let mut f = vec![1.0, 1.0, 1.0, 1.0];

        // Pin DoF 0 to a non-zero value — exercises the column shift.
        let bcs = [BoundaryCondition::Dirichlet { vertex: 0, value: 2.0 }];
        apply_dirichlet(&mut k, &mut f, &bcs);

        // K[0, 1] should now be zero (column elimination).
        let lo = k.row_offsets[0];
        let hi = k.row_offsets[0 + 1];
        for slot in lo..hi {
            let c = k.col_idx[slot];
            if c == 0 {
                assert!((k.values[slot] - 1.0).abs() < 1e-15);
            } else {
                assert!(k.values[slot].abs() < 1e-15, "K[0, {c}] not zeroed");
            }
        }
        // K[1, 0] should also now be zero.
        let lo = k.row_offsets[1];
        let hi = k.row_offsets[1 + 1];
        for slot in lo..hi {
            if k.col_idx[slot] == 0 {
                assert!(k.values[slot].abs() < 1e-15, "K[1, 0] not zeroed");
            }
        }
        // RHS at row 1 should have absorbed K[1, 0] · value = -1 · 2 = -2.
        // f[1] was 1, now should be 1 - (-1)·2 = 3.
        assert!((f[1] - 3.0).abs() < 1e-15, "f[1] = {} (expected 3.0)", f[1]);
        // f[0] should be the pin value.
        assert!((f[0] - 2.0).abs() < 1e-15);
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
