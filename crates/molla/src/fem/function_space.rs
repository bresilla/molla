//! Function-space tag — describes how DoFs are attached to the mesh.
//!
//! v1 ships a single variant, `P1Nodal` (one DoF per vertex,
//! piecewise-linear basis). Follow-up stages will extend this:
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
}

impl FunctionSpace {
    /// Global DoF count for `mesh` under this space.
    #[must_use]
    pub fn ndof(&self, mesh: &FemMesh) -> usize {
        match self {
            FunctionSpace::P1Nodal => mesh.vert_count(),
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
}
