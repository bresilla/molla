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
#[derive(Copy, Clone, Debug)]
pub enum BoundaryCondition {
    /// Dirichlet: pin DoF at `vertex` to exactly `value`.
    Dirichlet { vertex: u32, value: Real },
    /// Neumann: prescribe a flux `flux` over the triangular `face`
    /// (3 vertex indices). v1 stores but does not yet integrate this
    /// (zero flux is the homogeneous default and arises naturally —
    /// no work needed).
    Neumann { face: [u32; 3], flux: Real },
}

/// Apply Dirichlet BCs in-place: zero out the corresponding row of
/// `k`, set the diagonal entry to 1, and overwrite the matching RHS
/// entry with `value`. Neumann entries are ignored here — they
/// would either contribute to `f` (Stage 36.1) or be silently zero
/// in the v1 homogeneous case.
///
/// We mutate the CSR's `values` array directly. The pattern is left
/// unchanged (the zeroed-out off-diagonal slots stay in the sparsity
/// pattern, just with zero numerical values).
pub fn apply_dirichlet(
    k: &mut BsrMatrix,
    f: &mut [Real],
    bcs: &[BoundaryCondition],
) {
    debug_assert_eq!(k.b, 1, "apply_dirichlet: only B = 1 supported in v1");
    debug_assert_eq!(k.nrows(), f.len(), "RHS length mismatch");

    for bc in bcs {
        if let BoundaryCondition::Dirichlet { vertex, value } = *bc {
            let r = vertex as usize;
            let lo = k.row_offsets[r];
            let hi = k.row_offsets[r + 1];
            for slot in lo..hi {
                let c = k.col_idx[slot];
                k.values[slot] = if c == r { 1.0 } else { 0.0 };
            }
            f[r] = value;
        }
    }
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
