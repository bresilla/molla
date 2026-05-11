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

use crate::fem::mesh::{FemCell, FemMesh};
use crate::math::Real;
use crate::sim::TetMeshId;

/// Sentinel for the "no third vertex" slot used when a `FemMesh` is
/// purely 2D and [`boundary_facets`] returns boundary edges encoded as
/// degenerate triangles `[a, b, INVALID_VERT]`. Callers should test for
/// this value before treating the slot as a real vertex index.
///
/// Picked as `u32::MAX` so it cannot collide with a real mesh vertex
/// index (which is bounded by `mesh.vertices.len() < u32::MAX` in
/// practice).
pub const INVALID_VERT: u32 = u32::MAX;

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
    /// First edge in `Model.soft_surface_edges` (Stage 33.1 — EE
    /// contact). Edges are unique undirected `[min, max]` pairs
    /// derived from the surface's tris.
    pub edge_offset: u32,
    /// Number of edges in `Model.soft_surface_edges` belonging to
    /// this surface (Stage 33.1).
    pub edge_count: u32,
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
    /// Stage 33.5 — Coulomb friction coefficient at contacts on this
    /// surface. Two contacting surfaces' values combine via the
    /// geometric mean (`μ = sqrt(μ_a · μ_b)`); 0 disables tangential
    /// resistance for any contact involving the surface. Default 0.4.
    pub friction: Real,
    /// Stage 33.5 — restitution coefficient (0 = perfectly inelastic,
    /// 1 = full velocity reversal). Two contacting surfaces' values
    /// combine via `min(e_a, e_b)`. Defaults per kind: `Cloth = 0.0`,
    /// `TetMesh = 0.1`.
    pub restitution: Real,
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

    // Collect surviving boundary tris and sort deterministically by
    // sorted-vertex key. HashMap iteration is non-deterministic across
    // runs; without this sort the resulting tri ordering — and the
    // global tri IDs that downstream code (LBVH, soft-VT contact,
    // friction warm-start LUT keyed on `(vertex_global_id, tri_global_id)`)
    // computes from it — would shuffle frame-to-frame.
    let mut out: Vec<[u32; 3]> = faces
        .into_iter()
        .filter_map(|(_, (count, tri))| if count == 1 { Some(tri) } else { None })
        .collect();
    out.sort_unstable_by_key(|tri| {
        let mut k = *tri;
        k.sort_unstable();
        k
    });
    out
}

/// Compute the unique undirected edges from a triangle set.
///
/// Each tri contributes its 3 edges; we sort each pair as
/// `[min, max]` and dedup so the result is a flat `Vec<[u32; 2]>`
/// with one entry per unique edge. Used by Stage 33.1's EE contact
/// gather to enumerate per-surface edges without re-deriving them
/// per substep.
#[must_use]
pub fn compute_surface_edges(tris: &[[u32; 3]]) -> Vec<[u32; 2]> {
    use std::collections::HashSet;
    if tris.is_empty() {
        return Vec::new();
    }
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(tris.len() * 3);
    let mut out: Vec<[u32; 2]> = Vec::with_capacity(tris.len() * 3);
    for tri in tris {
        let pairs = [
            (tri[0], tri[1]),
            (tri[1], tri[2]),
            (tri[2], tri[0]),
        ];
        for (a, b) in pairs {
            if a == b {
                continue; // degenerate edge; skip.
            }
            let key = if a < b { (a, b) } else { (b, a) };
            if seen.insert(key) {
                out.push([key.0, key.1]);
            }
        }
    }
    out
}

/// Outer-surface facets of a [`FemMesh`] — Stage 47 v3.
///
/// The L²-projection operator integrates the cross-mesh mass either on
/// the interior volume (Stage 47 v1/v2) or on the **outer surface** of
/// the target mesh (v3). The latter is the right integration domain
/// when the cross-mesh field carries a *contact pressure* rather than a
/// volumetric source — the integrand has support only on the contact
/// patch, and integrating it over the volume would dilute the result
/// by O(h/L) per layer of interior tets.
///
/// This helper picks the right boundary extraction per cell type:
///
/// - **Tet meshes** — delegate to
///   [`compute_tet_surface_triangles`]: collect every tet's 4 faces,
///   keep those that belong to a single tet (outer boundary), preserve
///   their CCW-from-outside winding.
/// - **Tri meshes** — return the **boundary edges** encoded as
///   degenerate triangles `[a, b, INVALID_VERT]`. The caller treats
///   each such "facet" as a line segment in 3D and integrates a 1D
///   quadrature along it; this lets the surface operator API uniformly
///   return `Vec<[u32; 3]>` regardless of the source dimension.
///   Boundary edges are the ones that appear in exactly one triangle,
///   detected via the same one-vs-many-incidence count as for tets.
/// - **Mixed cell kinds** (tris + tets in the same mesh) — both
///   contributions are concatenated; the surface integrator skips
///   `INVALID_VERT`-tagged facets when running the 2D tri rule.
///
/// Returns an empty `Vec` when the mesh has no cells. Faces / edges are
/// sorted deterministically (lexicographic on sorted-vertex key) so the
/// output is reproducible across runs — important for downstream code
/// that warm-starts a contact LUT keyed on facet identity.
#[must_use]
pub fn boundary_facets(mesh: &FemMesh) -> Vec<[u32; 3]> {
    use std::collections::HashMap;

    // Split cells by kind so each path can use its specialised
    // counter. Cloning the index buffers is cheap; the alternative
    // (parameterising `compute_tet_surface_triangles` over a generic
    // face count) bloats the API for one caller.
    let mut tets: Vec<[u32; 4]> = Vec::new();
    let mut tris: Vec<[u32; 3]> = Vec::new();
    for cell in &mesh.cells {
        match *cell {
            FemCell::Tet(v) => tets.push(v),
            FemCell::Tri(v) => tris.push(v),
        }
    }

    let mut out: Vec<[u32; 3]> = compute_tet_surface_triangles(&tets);

    if !tris.is_empty() {
        // Mirror the "exactly-one-incident-cell" rule used for tets,
        // but on edges. We keep the *first* directed edge seen so the
        // boundary edge inherits an oriented winding (the caller can
        // use the original tri's normal if it needs an outward
        // direction; v1 of the surface operator only needs the edge
        // length and midpoint, so the orientation is informational).
        let mut edges: HashMap<[u32; 2], (u32, [u32; 2])> = HashMap::new();
        for t in &tris {
            let raw_edges: [[u32; 2]; 3] = [
                [t[0], t[1]],
                [t[1], t[2]],
                [t[2], t[0]],
            ];
            for raw in raw_edges {
                if raw[0] == raw[1] {
                    continue; // degenerate edge.
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
                if count == 1 {
                    Some([edge[0], edge[1], INVALID_VERT])
                } else {
                    None
                }
            })
            .collect();
        bnd_edges.sort_unstable_by_key(|f| {
            // INVALID_VERT is `u32::MAX` and sorts last, so the sort
            // key is effectively `[min(a,b), max(a,b)]` — the same key
            // we used inside the HashMap.
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

    /// A unit-cube tet mesh has exactly 12 boundary triangles
    /// (6 cube faces × 2 triangles each).
    #[test]
    fn boundary_facets_unit_cube_has_twelve_tris() {
        let mesh = FemMesh::tet_grid_3d(1, 1, 1, 1.0, 1.0, 1.0);
        let facets = boundary_facets(&mesh);
        assert_eq!(facets.len(), 12, "expected 12 boundary tris, got {}", facets.len());
        for f in &facets {
            assert!(
                f[2] != INVALID_VERT,
                "pure tet mesh should not produce INVALID_VERT facets"
            );
        }
    }

    /// A 2D triangle mesh on the unit square — boundary is the outer
    /// 4-edge loop. `tri_grid_2d(nx, ny)` puts `2·nx + 2·ny` triangles
    /// on the boundary, contributing `2·(nx + ny)` boundary edges.
    #[test]
    fn boundary_facets_tri_grid_returns_invalid_third_slot() {
        let mesh = FemMesh::tri_grid_2d(3, 2, 1.0, 1.0);
        let facets = boundary_facets(&mesh);
        let n_boundary = 2 * (3 + 2);
        assert_eq!(facets.len(), n_boundary, "got {} boundary edges", facets.len());
        for f in &facets {
            assert_eq!(
                f[2], INVALID_VERT,
                "tri-mesh boundary edge must have INVALID_VERT in slot 2"
            );
        }
    }
}
