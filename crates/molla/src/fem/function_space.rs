//! Function-space tag — describes how DoFs are attached to the mesh.
//!
//! v1 shipped `P1Nodal` (one scalar DoF per vertex). Stage 36.1 adds
//! `P1Vector { dim }` (a vector field of `dim` scalar DoFs per vertex)
//! to support linear elasticity (`dim = 3` in 3D, `dim = 2` for plane-
//! strain in 2D).
//!
//! Follow-up stages will extend this further:
//!
//! - `P0CellWise`     — one DoF per cell (DG / FV variants).
//! - `P2Nodal`        — vertex + edge DoFs (Stage 36.3).
//! - `RaviartThomas`  — face-normal DoFs (Stage 36.2 mixed Stokes).
//!
//! Keeping the enum closed for now means we can pattern-match on it in
//! [`crate::fem::assembly`] without leaving an `_` arm dangling.

use super::mesh::FemMesh;

/// Discrete function-space tag.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FunctionSpace {
    /// Piecewise-linear nodal basis. One scalar DoF per vertex.
    P1Nodal,
    /// Piecewise-linear nodal basis with `dim` vector components per
    /// vertex. DoF layout is **interleaved**: vertex `v` component `c`
    /// lives at global DoF `v * dim + c`. Stage 36.1 elasticity uses
    /// `dim = 3` in 3D and `dim = 2` for plane-strain 2D.
    P1Vector { dim: usize },
}

impl FunctionSpace {
    /// Global DoF count for `mesh` under this space.
    #[must_use]
    pub fn ndof(&self, mesh: &FemMesh) -> usize {
        match self {
            FunctionSpace::P1Nodal => mesh.vert_count(),
            FunctionSpace::P1Vector { dim } => mesh.vert_count() * dim,
        }
    }

    /// Number of scalar DoFs per vertex (1 for scalar fields, `dim`
    /// for vector fields). Used by the assembly path to compute the
    /// DoF stride.
    #[must_use]
    pub fn dofs_per_vertex(&self) -> usize {
        match self {
            FunctionSpace::P1Nodal => 1,
            FunctionSpace::P1Vector { dim } => *dim,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem::mesh::FemMesh;

    #[test]
    fn p1_nodal_ndof_matches_vertex_count() {
        let m = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        assert_eq!(FunctionSpace::P1Nodal.ndof(&m), m.vert_count());
    }

    #[test]
    fn p1_vector_ndof_scales_with_dim() {
        let m = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        let v = m.vert_count();
        assert_eq!(FunctionSpace::P1Vector { dim: 2 }.ndof(&m), 2 * v);
        assert_eq!(FunctionSpace::P1Vector { dim: 3 }.ndof(&m), 3 * v);
        assert_eq!(FunctionSpace::P1Vector { dim: 3 }.dofs_per_vertex(), 3);
        assert_eq!(FunctionSpace::P1Nodal.dofs_per_vertex(), 1);
    }
}
