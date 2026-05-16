//! FEM surface extraction helpers.

use std::collections::HashMap;

use molla_sim::{INVALID_VERT, compute_tet_surface_triangles};

use crate::mesh::{FemCell, FemMesh};

/// Outer-surface facets of a [`FemMesh`].
///
/// Tet meshes return true boundary triangles. Triangle meshes return boundary
/// edges encoded as degenerate triangles `[a, b, INVALID_VERT]` so callers can
/// share one `Vec<[u32; 3]>` surface-facet API.
#[must_use]
pub fn boundary_facets(mesh: &FemMesh) -> Vec<[u32; 3]> {
    let mut tets: Vec<[u32; 4]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for cell in &mesh.cells {
        match *cell {
            FemCell::Tet(v) => tets.push(v),
            FemCell::Tri(v) => tris.push(v),
        }
    }

    let mut out = compute_tet_surface_triangles(&tets);

    if !tris.is_empty() {
        let mut edges: HashMap<[u32; 2], (u32, [u32; 2])> = HashMap::new();
        for t in &tris {
            for raw in [[t[0], t[1]], [t[1], t[2]], [t[2], t[0]]] {
                if raw[0] == raw[1] {
                    continue;
                }
                let mut key = raw;
                key.sort_unstable();
                let entry = edges.entry(key).or_insert((0, raw));
                entry.0 += 1;
            }
        }
        let mut bnd_edges: Vec<[u32; 3]> = edges
            .into_iter()
            .filter_map(|(_, (count, edge))| {
                (count == 1).then_some([edge[0], edge[1], INVALID_VERT])
            })
            .collect();
        bnd_edges.sort_unstable_by_key(|f| {
            let mut k = [f[0], f[1]];
            k.sort_unstable();
            k
        });
        out.extend(bnd_edges);
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_facets_unit_cube_has_twelve_tris() {
        let mesh = FemMesh::tet_grid_3d(1, 1, 1, 1.0, 1.0, 1.0);
        let facets = boundary_facets(&mesh);
        assert_eq!(facets.len(), 12);
        for f in &facets {
            assert!(f[2] != INVALID_VERT);
        }
    }

    #[test]
    fn boundary_facets_tri_grid_returns_invalid_third_slot() {
        let mesh = FemMesh::tri_grid_2d(3, 2, 1.0, 1.0);
        let facets = boundary_facets(&mesh);
        assert_eq!(facets.len(), 2 * (3 + 2));
        for f in &facets {
            assert_eq!(f[2], INVALID_VERT);
        }
    }
}
