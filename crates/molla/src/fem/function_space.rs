//! Function-space tag — describes how DoFs are attached to the mesh.
//!
//! v1 shipped `P1Nodal` (one scalar DoF per vertex).
//! Stage 36.1 adds `P1Vector { dim }` (a vector field of `dim` scalar
//! DoFs per vertex) — supports linear elasticity (`dim = 3` in 3D,
//! `dim = 2` for plane-strain 2D).
//! Stage 36.3 adds `P2Nodal` (vertex + edge-midpoint DoFs).
//! Stage 39B adds `DGNodal { degree }` — discontinuous Galerkin nodal
//! basis with per-cell DoFs (no inter-cell sharing).
//!
//! Follow-up stages will extend this further:
//!
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
    /// Piecewise-quadratic nodal basis with `dim` vector components
    /// per node. DoF layout: corner DoFs first (interleaved by
    /// component), then edge-midpoint DoFs (interleaved by component),
    /// matching the [`crate::fem::p2_element::P2DofMap`] layout
    /// extended for `dim` components. Stage 36.2 Taylor-Hood velocity
    /// uses `dim = 3` in 3D, `dim = 2` for 2D.
    P2Vector { dim: usize },
    /// Stage 39B — Discontinuous Galerkin nodal basis of the given
    /// polynomial `degree`. DoFs are **per-cell**: a vertex shared by
    /// `k` cells contributes `k * ndof_per_cell` DoFs total — one set
    /// per cell. Inter-element coupling is via numerical flux at face
    /// quadrature points, NOT via shared DoFs.
    ///
    /// `ndof_per_cell` is `(deg+1)(deg+2)/2` for triangles (1 for P0,
    /// 3 for P1, 6 for P2) and `C(deg+3, 3)` for tetrahedra (1 for P0,
    /// 4 for P1, 10 for P2). v1 ships support for `degree ∈ {0, 1}`;
    /// higher-degree DG is a follow-up stage.
    ///
    /// The DoF offset for cell `c` is `c * ndof_per_cell` and the
    /// local-`i`-th DoF of cell `c` lives at global index
    /// `c * ndof_per_cell + i`.
    DGNodal { degree: usize },
}

impl FunctionSpace {
    /// Global DoF count for `mesh` under this space.
    ///
    /// - `P1Nodal` → `vertex_count`.
    /// - `P2Nodal` → `vertex_count + edge_count` (one extra DoF at the
    ///   midpoint of every unique undirected edge in the mesh).
    /// - `P2Vector { dim }` → `(vertex_count + edge_count) * dim`.
    /// - `DGNodal { degree }` → `Σ_cells ndof_per_cell(cell, degree)`.
    ///   For a homogeneous mesh this collapses to `cell_count *
    ///   ndof_per_cell(degree)`. Returns 0 on an empty mesh.
    #[must_use]
    pub fn ndof(&self, mesh: &FemMesh) -> usize {
        match self {
            FunctionSpace::P1Nodal => mesh.vert_count(),
            FunctionSpace::P2Nodal => {
                let (edges, _) = compute_edges(mesh);
                mesh.vert_count() + edges.len()
            }
            FunctionSpace::P1Vector { dim } => mesh.vert_count() * dim,
            FunctionSpace::P2Vector { dim } => {
                let (edges, _) = compute_edges(mesh);
                (mesh.vert_count() + edges.len()) * dim
            }
            FunctionSpace::DGNodal { degree } => mesh
                .cells
                .iter()
                .map(|c| dg_ndof_per_cell(c, *degree))
                .sum(),
        }
    }

    /// Number of scalar DoFs per node (1 for scalar fields, `dim`
    /// for vector fields). For `P2*` spaces a "node" is either a
    /// corner or an edge-midpoint. For DG spaces the per-cell layout
    /// is exposed via [`FunctionSpace::dg_ndof_per_cell`] and the
    /// concept of "per-vertex" doesn't apply; returns 1 as a sensible
    /// default so DG-aware paths can fall through CG-style assembly
    /// helpers safely.
    #[must_use]
    pub fn dofs_per_vertex(&self) -> usize {
        match self {
            FunctionSpace::P1Nodal => 1,
            FunctionSpace::P2Nodal => 1,
            FunctionSpace::P1Vector { dim } => *dim,
            FunctionSpace::P2Vector { dim } => *dim,
            FunctionSpace::DGNodal { .. } => 1,
        }
    }

    /// Per-cell DoF count for a DG function space — `(deg+1)(deg+2)/2`
    /// for triangles and `C(deg+3, 3)` for tetrahedra.
    ///
    /// Returns `None` for non-DG spaces; the caller should use
    /// [`FunctionSpace::ndof`] / `dofs_per_vertex` for those.
    ///
    /// # Panics
    /// Panics for DG `degree > 2` — those polynomial orders aren't
    /// shipped in v1 (the per-cell shape-function tables are only
    /// built for `deg ∈ {0, 1, 2}`). The DG element kernels in v1
    /// further restrict `degree ≤ 1`.
    #[must_use]
    pub fn dg_ndof_per_cell(&self, cell: &FemCell) -> Option<usize> {
        match self {
            FunctionSpace::DGNodal { degree } => Some(dg_ndof_per_cell(cell, *degree)),
            _ => None,
        }
    }
}

/// Per-cell DoF count for a DG element of polynomial `degree`.
///
/// - Tri P0 = 1, P1 = 3, P2 = 6 → `(deg+1)(deg+2)/2`
/// - Tet P0 = 1, P1 = 4, P2 = 10 → `C(deg+3, 3) = (deg+1)(deg+2)(deg+3)/6`
///
/// Panics for `degree > 2` — the v1 DG element kernels (see
/// [`crate::fem::dg_element`]) only support `degree ∈ {0, 1}`.
#[must_use]
pub fn dg_ndof_per_cell(cell: &FemCell, degree: usize) -> usize {
    assert!(
        degree <= 2,
        "DG: degree {degree} out of range (v1 supports 0..=2)"
    );
    match cell {
        FemCell::Tri(_) => (degree + 1) * (degree + 2) / 2,
        FemCell::Tet(_) => (degree + 1) * (degree + 2) * (degree + 3) / 6,
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

    #[test]
    fn p2_vector_ndof_scales_with_dim() {
        let m = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        let (edges, _) = compute_edges(&m);
        let n_nodes = m.vert_count() + edges.len();
        assert_eq!(FunctionSpace::P2Vector { dim: 3 }.ndof(&m), 3 * n_nodes);
        assert_eq!(FunctionSpace::P2Vector { dim: 2 }.ndof(&m), 2 * n_nodes);
        assert_eq!(FunctionSpace::P2Vector { dim: 3 }.dofs_per_vertex(), 3);
    }

    #[test]
    fn dg_p0_ndof_equals_cell_count() {
        let m = FemMesh::tri_grid_2d(3, 3, 1.0, 1.0);
        let space = FunctionSpace::DGNodal { degree: 0 };
        assert_eq!(space.ndof(&m), m.cell_count());
    }

    #[test]
    fn dg_p1_ndof_equals_three_times_tri_count() {
        let m = FemMesh::tri_grid_2d(4, 4, 1.0, 1.0);
        let space = FunctionSpace::DGNodal { degree: 1 };
        // Each tri contributes 3 DG-P1 DoFs (no sharing).
        assert_eq!(space.ndof(&m), 3 * m.cell_count());
    }

    #[test]
    fn dg_p1_tet_ndof_equals_four_times_tet_count() {
        let m = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        let space = FunctionSpace::DGNodal { degree: 1 };
        assert_eq!(space.ndof(&m), 4 * m.cell_count());
    }
}
