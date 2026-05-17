//! Raw soft-surface descriptors shared by simulation and geometry.
//!
//! `sim` owns the registry shape because these descriptors are model data:
//! contiguous particle ranges plus flat triangle/edge slices. Geometry owns the
//! acceleration structures and contact queries derived from this raw data.

use molla_math::Real;

use super::TetMeshId;

/// Sentinel for the "no third vertex" slot used when a surface extractor returns
/// boundary edges encoded as degenerate triangles `[a, b, INVALID_VERT]`.
pub const INVALID_VERT: u32 = u32::MAX;

/// Whether a soft surface comes from a tet mesh (boundary tris) or a cloth mesh
/// (tessellation tris).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SoftSurfaceKind {
    /// Boundary triangulation of a volumetric tet mesh.
    TetMesh,
    /// Cloth tessellation tris (already in `Model.tri_indices`).
    Cloth,
}

/// Stable identifier for a cloth surface added through the cloth builder.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ClothMeshId(pub u32);

impl ClothMeshId {
    /// Raw 0-based index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// One soft-surface registry entry. References slices of two flat arrays:
/// - `vertex_offset..vertex_offset + vertex_count` indexes the model's shared
///   `particle_q` and `particle_inv_mass` buffers.
/// - `tri_offset..tri_offset + tri_count` indexes `Model.soft_surface_tris`.
#[derive(Copy, Clone, Debug)]
pub struct SoftSurface {
    /// Tet boundary tris vs cloth tris.
    pub kind: SoftSurfaceKind,
    /// First vertex (particle) belonging to this surface — index into
    /// `Model.particle_q`.
    pub vertex_offset: u32,
    /// Number of vertices owned by this surface.
    pub vertex_count: u32,
    /// First triangle in `Model.soft_surface_tris`.
    pub tri_offset: u32,
    /// Number of triangles in `Model.soft_surface_tris` belonging to this
    /// surface.
    pub tri_count: u32,
    /// First edge in `Model.soft_surface_edges`.
    pub edge_offset: u32,
    /// Number of edges in `Model.soft_surface_edges` belonging to this surface.
    pub edge_count: u32,
    /// Source tet mesh identifier when `kind == TetMesh`; `None` for cloth
    /// surfaces.
    pub tet_mesh_id: Option<TetMeshId>,
    /// Source cloth mesh identifier when `kind == Cloth`; `None` for tet
    /// surfaces.
    pub cloth_mesh_id: Option<ClothMeshId>,
    /// Whether this surface participates in self-collision.
    pub self_collision: bool,
    /// Coulomb friction coefficient at contacts on this surface.
    pub friction: Real,
    /// Restitution coefficient at contacts on this surface.
    pub restitution: Real,
}

/// Compute the boundary (outer-surface) triangles of a tet soup via the standard
/// "tris that belong to exactly one tet" pass.
#[must_use]
pub fn compute_tet_surface_triangles(tets: &[[u32; 4]]) -> Vec<[u32; 3]> {
    use std::collections::HashMap;

    if tets.is_empty() {
        return Vec::new();
    }

    let mut faces: HashMap<[u32; 3], (u32, [u32; 3])> = HashMap::new();
    for t in tets {
        let v0 = t[0];
        let v1 = t[1];
        let v2 = t[2];
        let v3 = t[3];
        let raw_faces: [[u32; 3]; 4] = [[v1, v2, v3], [v0, v3, v2], [v0, v1, v3], [v0, v2, v1]];
        for raw in raw_faces {
            let mut key = raw;
            key.sort_unstable();
            let entry = faces.entry(key).or_insert((0, raw));
            entry.0 += 1;
        }
    }

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
#[must_use]
pub fn compute_surface_edges(tris: &[[u32; 3]]) -> Vec<[u32; 2]> {
    use std::collections::HashSet;
    if tris.is_empty() {
        return Vec::new();
    }
    let mut seen: HashSet<(u32, u32)> = HashSet::with_capacity(tris.len() * 3);
    let mut out: Vec<[u32; 2]> = Vec::with_capacity(tris.len() * 3);
    for tri in tris {
        let pairs = [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])];
        for (a, b) in pairs {
            if a == b {
                continue;
            }
            let key = if a < b { (a, b) } else { (b, a) };
            if seen.insert(key) {
                out.push([key.0, key.1]);
            }
        }
    }
    out
}
