//! Global assembly — scatter per-cell local matrices into a CSR.
//!
//! The visitor pattern lets the caller supply *any* function from
//! `(cell, vertices) → local-matrix`; v1 uses
//! [`crate::fem::element::compute_element_mass`] and
//! [`crate::fem::element::compute_element_stiffness`] but Stage 36.1+
//! kernels (elasticity, Stokes, etc.) plug in unchanged.
//!
//! Sparse layout: we use the existing [`crate::math::sparse::BsrMatrix`]
//! with block size `B = 1` — i.e. the same storage as a plain CSR
//! matrix. We collect `(row, col, value)` triplets while iterating
//! cells, then rely on `BsrMatrix::from_triplets` to dedupe pattern
//! entries and accumulate values at each `(row, col)` slot.
//!
//! For v1 the COO → CSR step costs `O(nnz · log nnz)` (sort + scan
//! inside `from_triplets`); plenty fast at the Stage-36 mesh sizes
//! (16³ tets ⇒ ~5k vertices, ~25k cells, ~100k triplets in <10 ms).
//! Larger meshes would benefit from a row-grouped pre-pass; deferred.

use crate::core::Result;
use crate::math::sparse::BsrMatrix;
use crate::math::{Real, Vec3};

use super::function_space::FunctionSpace;
use super::mesh::FemMesh;
use super::mesh::FemCell;

/// Assemble a global matrix `A_global` by scattering the visitor's
/// local matrix into the CSR for each cell.
///
/// The visitor receives the cell + the full vertex array (so it can
/// compute geometric quantities) and must return an `N × N` row-major
/// `Vec<Vec<Real>>` where `N = cell.vert_count()`.
///
/// The DoF map for `FunctionSpace::P1Nodal` is the identity:
/// vertex index `i` ↔ global DoF `i`.
pub fn assemble_global<F>(
    mesh: &FemMesh,
    space: FunctionSpace,
    element_visitor: F,
) -> Result<BsrMatrix>
where
    F: Fn(&FemCell, &[Vec3]) -> Vec<Vec<Real>>,
{
    let n = space.ndof(mesh);
    let mut triplets: Vec<(usize, usize, Vec<Real>)> = Vec::new();
    triplets.reserve(mesh.cells.len() * 16);

    for cell in &mesh.cells {
        let local = element_visitor(cell, &mesh.vertices);
        let verts = cell.verts();
        debug_assert_eq!(local.len(), verts.len());
        for (i, vi) in verts.iter().enumerate() {
            debug_assert_eq!(local[i].len(), verts.len());
            for (j, vj) in verts.iter().enumerate() {
                let val = local[i][j];
                if val == 0.0 {
                    continue;
                }
                triplets.push((*vi as usize, *vj as usize, vec![val]));
            }
        }
    }

    BsrMatrix::from_triplets(1, n, n, &triplets)
}

/// Convenience: assemble the (consistent) mass matrix for `mesh`.
pub fn assemble_mass(mesh: &FemMesh, space: FunctionSpace) -> Result<BsrMatrix> {
    assemble_global(mesh, space, super::element::compute_element_mass)
}

/// Convenience: assemble the Laplacian stiffness matrix for `mesh`,
/// scaled by `coeff` (e.g. thermal diffusivity α).
pub fn assemble_stiffness(
    mesh: &FemMesh,
    space: FunctionSpace,
    coeff: Real,
) -> Result<BsrMatrix> {
    assemble_global(mesh, space, |cell, verts| {
        super::element::compute_element_stiffness(cell, verts, coeff)
    })
}

/// `y = A · x` over a [`BsrMatrix`] with `B = 1`. Allocating wrapper
/// matched to the closure-style API of [`crate::math::iterative`].
#[must_use]
pub fn matvec(a: &BsrMatrix, x: &[Real]) -> Vec<Real> {
    let mut y = vec![0.0; a.nrows()];
    a.mv(x, &mut y).expect("BsrMatrix::mv");
    y
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem::element::{compute_element_mass, compute_element_stiffness};

    #[test]
    fn mass_matrix_row_sum_equals_lumped_mass() {
        // ∑ⱼ Mᵢⱼ over all vertices ⇒ vertex-lumped mass; sum of those
        // ⇒ total mesh measure (area for tris, volume for tets).
        let mesh = FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
        let space = FunctionSpace::P1Nodal;
        let m = assemble_global(&mesh, space, compute_element_mass).unwrap();
        let n = space.ndof(&mesh);
        let mut total: Real = 0.0;
        let ones = vec![1.0; n];
        let row_sums = matvec(&m, &ones);
        for v in &row_sums {
            total += v;
        }
        assert!((total - 1.0).abs() < 1e-12, "total area = {total} ≠ 1");
    }

    #[test]
    fn stiffness_kills_constant_field() {
        let mesh = FemMesh::tet_grid_3d(3, 3, 3, 1.0, 1.0, 1.0);
        let space = FunctionSpace::P1Nodal;
        let k = assemble_global(&mesh, space, |cell, verts| {
            compute_element_stiffness(cell, verts, 1.0)
        })
        .unwrap();
        let n = space.ndof(&mesh);
        let ones = vec![1.0; n];
        let kx = matvec(&k, &ones);
        for v in kx {
            assert!(v.abs() < 1e-10, "K · 1 should be zero, got {v}");
        }
    }
}
