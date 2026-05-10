//! Function-space tag — describes how DoFs are attached to the mesh.
//!
//! v1 shipped `P1Nodal` (one scalar DoF per vertex).
//! Stage 36.1 adds `P1Vector { dim }` (a vector field of `dim` scalar
//! DoFs per vertex) — supports linear elasticity (`dim = 3` in 3D,
//! `dim = 2` for plane-strain 2D).
//! Stage 36.3 adds `P2Nodal` (vertex + edge-midpoint DoFs).
//!
//! Follow-up stages will extend this further:
//!
//! - `P0CellWise`     — one DoF per cell (DG / FV variants).
//! - `RaviartThomas`  — face-normal DoFs (Stage 36.2 mixed Stokes).
//!
//! Keeping the enum closed for now means we can pattern-match on it in
//! [`crate::fem::assembly`] without leaving an `_` arm dangling.

use std::collections::HashMap;

use super::mesh::{FemCell, FemMesh};

/// Discrete function-space tag.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FunctionSpace {
    /// Piecewise-linear nodal basis. One scalar DoF per vertex.
    P1Nodal,
    /// Piecewise-quadratic nodal basis. One scalar DoF per vertex +
    /// one DoF per unique mesh edge (Stage 36.3).
    P2Nodal,
    /// Piecewise-linear nodal basis with `dim` vector components per
    /// vertex. DoF layout is **interleaved**: vertex `v` component `c`
    /// lives at global DoF `v * dim + c`. Stage 36.1 elasticity uses
    /// `dim = 3` in 3D and `dim = 2` for plane-strain 2D.
    P1Vector { dim: usize },
}

impl FunctionSpace {
    /// Global DoF count for `mesh` under this space.
    ///
    /// - `P1Nodal` → `vertex_count`.
    /// - `P2Nodal` → `vertex_count + edge_count` (one extra DoF at the
    ///   midpoint of every unique undirected edge in the mesh).
    #[must_use]
    pub fn ndof(&self, mesh: &FemMesh) -> usize {
        match self {
            FunctionSpace::P1Nodal => mesh.vert_count(),
            FunctionSpace::P2Nodal => {
                let (edges, _) = compute_edges(mesh);
                mesh.vert_count() + edges.len()
            }
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
            FunctionSpace::P2Nodal => 1,
            FunctionSpace::P1Vector { dim } => *dim,
        }
    }
}

/// Enumerate the unique undirected edges of `mesh`. Each triangle
/// contributes its 3 edges; each tet contributes its 6 edges. Each
/// edge is stored as `[min, max]` and deduplicated.
///
/// Returns `(edges, edge_id)` where `edge_id[[a, b]]` (with `a < b`)
/// is the position of edge `[a, b]` inside `edges`. The lookup is
/// `O(1)` so call sites can resolve edge → DoF in tight loops without
/// re-scanning. Pattern matches `compute_surface_edges` in
/// `crate::geometry::soft_surface`.
#[must_use]
pub fn compute_edges(mesh: &FemMesh) -> (Vec<[u32; 2]>, HashMap<[u32; 2], u32>) {
    let mut edges: Vec<[u32; 2]> = Vec::new();
    let mut edge_id: HashMap<[u32; 2], u32> = HashMap::new();
    edges.reserve(mesh.cells.len() * 6);
    edge_id.reserve(mesh.cells.len() * 6);

    let push = |a: u32, b: u32, edges: &mut Vec<[u32; 2]>, ids: &mut HashMap<[u32; 2], u32>| {
        if a == b {
            return;
        }
        let key = if a < b { [a, b] } else { [b, a] };
        if let std::collections::hash_map::Entry::Vacant(slot) = ids.entry(key) {
            let id = edges.len() as u32;
            edges.push(key);
            slot.insert(id);
        }
    };

    for cell in &mesh.cells {
        match cell {
            FemCell::Tri(v) => {
                push(v[0], v[1], &mut edges, &mut edge_id);
                push(v[1], v[2], &mut edges, &mut edge_id);
                push(v[2], v[0], &mut edges, &mut edge_id);
            }
            FemCell::Tet(v) => {
                push(v[0], v[1], &mut edges, &mut edge_id);
                push(v[0], v[2], &mut edges, &mut edge_id);
                push(v[0], v[3], &mut edges, &mut edge_id);
                push(v[1], v[2], &mut edges, &mut edge_id);
                push(v[1], v[3], &mut edges, &mut edge_id);
                push(v[2], v[3], &mut edges, &mut edge_id);
            }
        }
    }
    (edges, edge_id)
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
    fn p2_nodal_ndof_is_vertex_plus_edge_count() {
        let m = FemMesh::tri_grid_2d(2, 2, 1.0, 1.0);
        let (edges, _) = compute_edges(&m);
        assert_eq!(
            FunctionSpace::P2Nodal.ndof(&m),
            m.vert_count() + edges.len()
        );
        assert!(edges.len() < 3 * m.cell_count());
    }

    #[test]
    fn compute_edges_single_tet_has_six_edges() {
        let m = FemMesh {
            vertices: vec![
                crate::math::Vec3::new(0.0, 0.0, 0.0),
                crate::math::Vec3::new(1.0, 0.0, 0.0),
                crate::math::Vec3::new(0.0, 1.0, 0.0),
                crate::math::Vec3::new(0.0, 0.0, 1.0),
            ],
            cells: vec![FemCell::Tet([0, 1, 2, 3])],
        };
        let (edges, ids) = compute_edges(&m);
        assert_eq!(edges.len(), 6);
        assert_eq!(ids.len(), 6);
        for a in 0u32..4 {
            for b in (a + 1)..4 {
                assert!(ids.contains_key(&[a, b]));
            }
        }
    }

    #[test]
    fn compute_edges_single_tri_has_three_edges() {
        let m = FemMesh {
            vertices: vec![
                crate::math::Vec3::new(0.0, 0.0, 0.0),
                crate::math::Vec3::new(1.0, 0.0, 0.0),
                crate::math::Vec3::new(0.0, 1.0, 0.0),
            ],
            cells: vec![FemCell::Tri([0, 1, 2])],
        };
        let (edges, ids) = compute_edges(&m);
        assert_eq!(edges.len(), 3);
        assert_eq!(ids.len(), 3);
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
