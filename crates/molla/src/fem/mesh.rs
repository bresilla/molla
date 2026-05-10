//! `FemMesh` + `FemCell` — the geometric substrate of the FEM module.
//!
//! v1 stores vertices as 3D points (`Vec3`) and cells as either
//! triangles (`Tri([u32; 3])`) or tetrahedra (`Tet([u32; 4])`). 2D
//! problems use triangles embedded in the `z = 0` plane — element
//! routines for triangles look at `(x, y)` only, so the embedding is
//! transparent.
//!
//! Two builder helpers cover the common test geometries:
//! - [`FemMesh::tri_grid_2d`] — `nx × ny` quad grid split into right
//!   triangles.
//! - [`FemMesh::tet_grid_3d`] — `nx × ny × nz` cube grid where each
//!   hex is split into 6 congruent tets ("Kuhn / Coxeter" subdivision).
//!
//! Stages 36.1 + 36.2 will likely add more builders (cantilever beam,
//! lid-driven cavity) but those scenes can lean on the same primitive
//! `Vec<Vec3> + Vec<FemCell>` layout — no schema change needed.

use crate::math::{Real, Vec3};

/// A single FEM cell. v1 supports two shapes; follow-up stages can
/// add wedge / hex / pyramid by extending this enum.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FemCell {
    /// Triangle — 3 vertex indices into [`FemMesh::vertices`].
    Tri([u32; 3]),
    /// Tetrahedron — 4 vertex indices into [`FemMesh::vertices`].
    Tet([u32; 4]),
}

impl FemCell {
    /// Number of vertices this cell touches (3 for tri, 4 for tet).
    #[must_use]
    pub fn vert_count(&self) -> usize {
        match self {
            FemCell::Tri(_) => 3,
            FemCell::Tet(_) => 4,
        }
    }

    /// Vertex indices (slice form) — useful for assembly scatter.
    #[must_use]
    pub fn verts(&self) -> &[u32] {
        match self {
            FemCell::Tri(v) => v,
            FemCell::Tet(v) => v,
        }
    }
}

/// Unstructured mesh of triangles / tetrahedra used by the FEM solver.
///
/// `vertices[i]` carries the position of vertex `i`; `cells[k]` lists
/// the vertex indices of cell `k`. There is no per-cell material slot
/// in v1 — that arrives in Stage 36.1 with elasticity.
#[derive(Clone, Debug)]
pub struct FemMesh {
    pub vertices: Vec<Vec3>,
    pub cells: Vec<FemCell>,
}

impl FemMesh {
    /// Empty mesh — useful for incremental construction.
    #[must_use]
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            cells: Vec::new(),
        }
    }

    /// `nx × ny` triangulated quad grid spanning `[0, w] × [0, h]` in
    /// the `z = 0` plane. Each quad is split into two right-triangles
    /// along its `(low-left, top-right)` diagonal.
    ///
    /// Returns a mesh with `(nx + 1) × (ny + 1)` vertices and
    /// `2 · nx · ny` triangles.
    #[must_use]
    pub fn tri_grid_2d(nx: usize, ny: usize, w: Real, h: Real) -> Self {
        assert!(nx > 0 && ny > 0, "tri_grid_2d: nx, ny must be > 0");
        let mut vertices = Vec::with_capacity((nx + 1) * (ny + 1));
        let dx = w / nx as Real;
        let dy = h / ny as Real;
        for j in 0..=ny {
            for i in 0..=nx {
                vertices.push(Vec3::new(i as Real * dx, j as Real * dy, 0.0));
            }
        }
        let idx = |i: usize, j: usize| -> u32 { (j * (nx + 1) + i) as u32 };

        let mut cells = Vec::with_capacity(2 * nx * ny);
        for j in 0..ny {
            for i in 0..nx {
                let v00 = idx(i, j);
                let v10 = idx(i + 1, j);
                let v11 = idx(i + 1, j + 1);
                let v01 = idx(i, j + 1);
                // Diagonal v00 → v11; lower triangle (v00, v10, v11)
                // and upper triangle (v00, v11, v01) — both
                // counter-clockwise so the 2D area is positive.
                cells.push(FemCell::Tri([v00, v10, v11]));
                cells.push(FemCell::Tri([v00, v11, v01]));
            }
        }
        Self { vertices, cells }
    }

    /// `nx × ny × nz` hex grid spanning `[0, w] × [0, h] × [0, d]`,
    /// each hex split into 6 tets via the standard Coxeter / Kuhn
    /// subdivision.
    ///
    /// Returns a mesh with `(nx + 1) × (ny + 1) × (nz + 1)` vertices
    /// and `6 · nx · ny · nz` tetrahedra. All tets are oriented with
    /// positive (right-handed) volume so [`crate::fem::element`]
    /// doesn't need to flip signs.
    #[must_use]
    pub fn tet_grid_3d(nx: usize, ny: usize, nz: usize, w: Real, h: Real, d: Real) -> Self {
        assert!(nx > 0 && ny > 0 && nz > 0, "tet_grid_3d: dims must be > 0");
        let mut vertices = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
        let dx = w / nx as Real;
        let dy = h / ny as Real;
        let dz = d / nz as Real;
        for k in 0..=nz {
            for j in 0..=ny {
                for i in 0..=nx {
                    vertices.push(Vec3::new(
                        i as Real * dx,
                        j as Real * dy,
                        k as Real * dz,
                    ));
                }
            }
        }
        let stride_y = nx + 1;
        let stride_z = (nx + 1) * (ny + 1);
        let idx = |i: usize, j: usize, k: usize| -> u32 {
            (k * stride_z + j * stride_y + i) as u32
        };

        let mut cells = Vec::with_capacity(6 * nx * ny * nz);
        for k in 0..nz {
            for j in 0..ny {
                for i in 0..nx {
                    // 8 corners of the hex cell.
                    let v000 = idx(i, j, k);
                    let v100 = idx(i + 1, j, k);
                    let v010 = idx(i, j + 1, k);
                    let v110 = idx(i + 1, j + 1, k);
                    let v001 = idx(i, j, k + 1);
                    let v101 = idx(i + 1, j, k + 1);
                    let v011 = idx(i, j + 1, k + 1);
                    let v111 = idx(i + 1, j + 1, k + 1);

                    // Six-tet decomposition along the v000 → v111
                    // body diagonal. This is the standard
                    // "Kuhn triangulation"; all tets share the
                    // diagonal edge and have positive volume under
                    // the standard `(a, b-a) · ((c-a) × (d-a)) > 0`
                    // orientation convention.
                    cells.push(FemCell::Tet([v000, v100, v110, v111]));
                    cells.push(FemCell::Tet([v000, v110, v010, v111]));
                    cells.push(FemCell::Tet([v000, v010, v011, v111]));
                    cells.push(FemCell::Tet([v000, v011, v001, v111]));
                    cells.push(FemCell::Tet([v000, v001, v101, v111]));
                    cells.push(FemCell::Tet([v000, v101, v100, v111]));
                }
            }
        }
        Self { vertices, cells }
    }

    /// Number of vertices.
    #[must_use]
    pub fn vert_count(&self) -> usize {
        self.vertices.len()
    }

    /// Number of cells.
    #[must_use]
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }
}

impl Default for FemMesh {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tri_grid_2d_counts() {
        let m = FemMesh::tri_grid_2d(2, 3, 1.0, 1.0);
        assert_eq!(m.vertices.len(), 3 * 4);
        assert_eq!(m.cells.len(), 2 * 2 * 3);
    }

    #[test]
    fn tet_grid_3d_counts() {
        let m = FemMesh::tet_grid_3d(2, 3, 4, 1.0, 1.0, 1.0);
        assert_eq!(m.vertices.len(), 3 * 4 * 5);
        assert_eq!(m.cells.len(), 6 * 2 * 3 * 4);
    }

    #[test]
    fn tet_grid_3d_positive_volumes() {
        let m = FemMesh::tet_grid_3d(2, 2, 2, 1.0, 1.0, 1.0);
        for c in &m.cells {
            if let FemCell::Tet(v) = c {
                let a = m.vertices[v[0] as usize];
                let b = m.vertices[v[1] as usize];
                let cc = m.vertices[v[2] as usize];
                let d = m.vertices[v[3] as usize];
                let vol = (b - a).dot((cc - a).cross(d - a)) / 6.0;
                assert!(vol > 0.0, "tet has non-positive volume {vol}");
            }
        }
    }
}
