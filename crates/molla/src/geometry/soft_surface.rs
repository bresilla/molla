//! Soft-surface registry — Stage 33.
//!
//! Unifies tet boundary tris and cloth tris under one logical "soft
//! surface" so the cross-mesh VT contact pipeline can treat them
//! interchangeably. The downstream pieces (BVH wrapper, VT contact
//! gather, VBD penalty) walk the registry rather than re-deriving
//! per-mesh tessellation each substep.
//!
//! v1 ships CPU-only; the GPU port (Stage 33.2) reuses these data
//! structures verbatim and adds a per-substep tri-AABB shader.

use crate::sim::TetMeshId;

/// Whether a soft surface comes from a tet mesh (boundary tris) or a
/// cloth mesh (tessellation tris). Drives nothing on the contact path
/// itself — it's there for diagnostics and to scope future stage
/// extensions (e.g. self-collision Stage 33.3 needs adjacency that
/// differs by kind).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SoftSurfaceKind {
    /// Boundary triangulation of a volumetric tet mesh.
    TetMesh,
    /// Cloth tessellation tris (already in `Model.tri_indices`).
    Cloth,
}

/// Stable identifier for a cloth surface added through the cloth
/// builder. 0-based dense index into the cloth surface registry — the
/// builder stamps one of these per `add_cloth_grid*` call so future
/// stages can address "the i-th cloth" the way `TetMeshId` addresses
/// the i-th tet mesh.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ClothMeshId(pub u32);

impl ClothMeshId {
    /// Raw 0-based index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// One soft-surface registry entry. References slices of two flat
/// arrays:
/// - `vertex_offset..vertex_offset + vertex_count` indexes the model's
///   shared `particle_q` (and `particle_inv_mass`); both tet and cloth
///   surfaces live in the unified particle pool.
/// - `tri_offset..tri_offset + tri_count` indexes
///   `Model.soft_surface_tris` — flat per-surface tri lists are
///   concatenated so the VT contact pipeline can broad-phase across
///   all soft surfaces with one BVH.
#[derive(Copy, Clone, Debug)]
pub struct SoftSurface {
    /// Tet boundary tris vs cloth tris.
    pub kind: SoftSurfaceKind,
    /// First vertex (particle) belonging to this surface — index into
    /// `Model.particle_q`. The vertex range is contiguous because the
    /// builder allocates each soft-body's particles consecutively.
    pub vertex_offset: u32,
    /// Number of vertices owned by this surface.
    pub vertex_count: u32,
    /// First triangle in `Model.soft_surface_tris`.
    pub tri_offset: u32,
    /// Number of triangles in `Model.soft_surface_tris` belonging to
    /// this surface.
    pub tri_count: u32,
    /// Source tet mesh identifier when `kind == TetMesh`; `None` for
    /// cloth surfaces.
    pub tet_mesh_id: Option<TetMeshId>,
    /// Source cloth mesh identifier when `kind == Cloth`; `None` for
    /// tet surfaces.
    pub cloth_mesh_id: Option<ClothMeshId>,
    /// Stage 33.3 — whether this surface participates in self-
    /// collision. Default `false` preserves Stage 33's cross-mesh-
    /// only behaviour (same-surface VT pairs are skipped). When
    /// `true`, the VT contact gather emits same-surface pairs except
    /// for triangles topologically incident to the vertex (looked up
    /// via `Model.soft_vertex_incident_tris`).
    pub self_collision: bool,
}

/// Compute the boundary (outer-surface) triangles of a tet soup via
/// the standard "tris that belong to exactly one tet" pass.
///
/// Each tet contributes its 4 triangular faces (CCW from outside the
/// tet, so the cross product `(v1-v0) × (v2-v0)` points outward).
/// We group faces by the sorted triple of vertex indices: faces that
/// appear exactly once are on the boundary; faces shared by two tets
/// cancel out (they're internal). The original CCW ordering is
/// preserved on the surviving boundary face so downstream normal
/// queries get the outward-pointing normal.
///
/// Faces follow the same convention as `tet_closest_face` in
/// [`crate::geometry::tet_bvh`]: each tet `(v0, v1, v2, v3)` produces
/// faces `(v1, v2, v3)`, `(v0, v3, v2)`, `(v0, v1, v3)`, `(v0, v2, v1)`.
/// With the rest-state ordering the builder enforces (`det(Dm) > 0`),
/// each of these is CCW when viewed from outside.
#[must_use]
pub fn compute_tet_surface_triangles(tets: &[[u32; 4]]) -> Vec<[u32; 3]> {
    use std::collections::HashMap;

    if tets.is_empty() {
        return Vec::new();
    }

    // Map sorted-tri-key → (count, original_ccw_tri).
    let mut faces: HashMap<[u32; 3], (u32, [u32; 3])> = HashMap::new();
    for t in tets {
        let v0 = t[0];
        let v1 = t[1];
        let v2 = t[2];
        let v3 = t[3];
        let raw_faces: [[u32; 3]; 4] = [
            [v1, v2, v3],
            [v0, v3, v2],
            [v0, v1, v3],
            [v0, v2, v1],
        ];
        for raw in raw_faces {
            let mut key = raw;
            key.sort_unstable();
            let entry = faces.entry(key).or_insert((0, raw));
            entry.0 += 1;
            // Keep the *first* CCW orientation seen so the surviving
            // boundary face inherits the correct outward winding. If a
            // face is shared by two tets they'll both supply opposite
            // winding (and the count goes to 2 ⇒ excluded below); the
            // first-write wins is fine because either orientation
            // would be discarded.
        }
    }

    let mut out = Vec::new();
    for (_, (count, tri)) in faces {
        if count == 1 {
            out.push(tri);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One tet has exactly 4 boundary tris (every face is on the boundary).
    #[test]
    fn single_tet_has_four_boundary_tris() {
        let tets = vec![[0u32, 1, 2, 3]];
        let bnd = compute_tet_surface_triangles(&tets);
        assert_eq!(bnd.len(), 4);
    }

    /// Two tets sharing one face produce 6 boundary tris (4 + 4 -
    /// 2·1 shared face).
    #[test]
    fn two_tets_sharing_a_face_have_six_boundary_tris() {
        // Vertex layout: t0 = (0,1,2,3); t1 = (1,2,3,4). Shared face
        // is the triangle (1,2,3).
        let tets = vec![[0u32, 1, 2, 3], [1u32, 2, 3, 4]];
        let bnd = compute_tet_surface_triangles(&tets);
        assert_eq!(bnd.len(), 6);
    }

    #[test]
    fn empty_input_returns_empty() {
        let bnd = compute_tet_surface_triangles(&[]);
        assert!(bnd.is_empty());
    }
}
