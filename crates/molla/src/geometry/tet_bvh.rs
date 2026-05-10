//! Per-tet AABB BVH (Stage 20.4) for tet-vs-tet collision.
//!
//! ## Design choice — flat AABB list + GPU brute force
//!
//! The original Stage 20.4 spec called for a tet-AABB BVH with GPU
//! traversal, mirroring [`crate::geometry::mesh_bvh::MeshBvh`]. For
//! deformable meshes the BVH must be *refit* every substep (each tet's
//! AABB changes as vertices move), and a per-vertex GPU traversal of a
//! per-substep BVH is the central technical risk the spec flags.
//!
//! v1 ships the safer fallback the spec explicitly authorises: CPU
//! per-substep AABB recompute + GPU per-vertex *brute-force-against-
//! flat-AABB-array* check (with a coarse top-level AABB-pair cull on
//! the host so cross-mesh dispatches are skipped when the meshes don't
//! overlap). For the demo workload (two ~150-tet jelly cubes) the
//! per-vertex × per-tet quadratic cost is ~22.5 k checks per substep,
//! which an L2 GPU dispatches in microseconds. A real-BVH refit is
//! deferred to Stage 20.5; this module just lays out the per-tet AABB
//! pool the future per-mesh tree builder will consume.
//!
//! Public entry points:
//!  - [`recompute_tet_aabbs`]: refresh per-tet AABBs from current vertex
//!    positions. CPU-side, parallelisable.
//!  - [`tet_meshes_aabb_overlap`]: top-level cull — do the coarse
//!    bounding boxes of two tet meshes overlap?
//!  - [`tet_mesh_root_aabb`] / [`tet_closest_face`]: per-mesh root AABB
//!    helper and per-tet point-to-face query used by the host fixup pass.

use crate::math::{Real, Vec3};

/// One per-tet AABB. Padded to 32 bytes so the GPU upload is a clean
/// `vec4 + vec4` layout (`min.xyz` + `tet_id`, `max.xyz` + pad).
#[repr(C, align(16))]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TetAabb {
    pub min_x: f32,
    pub min_y: f32,
    pub min_z: f32,
    pub tet_id: u32,
    pub max_x: f32,
    pub max_y: f32,
    pub max_z: f32,
    pub _pad: f32,
}

impl TetAabb {
    #[must_use]
    pub fn min(self) -> Vec3 {
        Vec3::new(self.min_x as Real, self.min_y as Real, self.min_z as Real)
    }
    #[must_use]
    pub fn max(self) -> Vec3 {
        Vec3::new(self.max_x as Real, self.max_y as Real, self.max_z as Real)
    }
    #[must_use]
    pub fn from_corners(tet_id: u32, lo: Vec3, hi: Vec3) -> Self {
        Self {
            min_x: lo.x as f32,
            min_y: lo.y as f32,
            min_z: lo.z as f32,
            tet_id,
            max_x: hi.x as f32,
            max_y: hi.y as f32,
            max_z: hi.z as f32,
            _pad: 0.0,
        }
    }
    /// Inflate the AABB symmetrically by `margin` on every face (used
    /// to give the broad-phase a small contact buffer so vertices that
    /// are *just* poking into a tet still trigger the narrow-phase
    /// closest-face check).
    #[must_use]
    pub fn inflated(self, margin: Real) -> Self {
        let lo = self.min() - Vec3::splat(margin);
        let hi = self.max() + Vec3::splat(margin);
        Self::from_corners(self.tet_id, lo, hi)
    }
    /// World-AABB-vs-AABB overlap test (closed intervals).
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.min_x <= other.max_x
            && self.max_x >= other.min_x
            && self.min_y <= other.max_y
            && self.max_y >= other.min_y
            && self.min_z <= other.max_z
            && self.max_z >= other.min_z
    }
    /// Point-in-AABB test (closed).
    #[must_use]
    pub fn contains_point(&self, p: Vec3) -> bool {
        p.x as f32 >= self.min_x
            && p.x as f32 <= self.max_x
            && p.y as f32 >= self.min_y
            && p.y as f32 <= self.max_y
            && p.z as f32 >= self.min_z
            && p.z as f32 <= self.max_z
    }
}

/// Refresh per-tet AABBs from current vertex positions. `tet_indices`
/// is the flat `4 · tet_count` layout from `Model.tet_indices`.
/// `out` must be `tet_count` long.
pub fn recompute_tet_aabbs(
    tet_indices: &[u32],
    vertices: &[Vec3],
    margin: Real,
    out: &mut [TetAabb],
) {
    debug_assert_eq!(tet_indices.len(), out.len() * 4);
    for (i, slot) in out.iter_mut().enumerate() {
        let i0 = tet_indices[i * 4] as usize;
        let i1 = tet_indices[i * 4 + 1] as usize;
        let i2 = tet_indices[i * 4 + 2] as usize;
        let i3 = tet_indices[i * 4 + 3] as usize;
        let v0 = vertices[i0];
        let v1 = vertices[i1];
        let v2 = vertices[i2];
        let v3 = vertices[i3];
        let lo = v0.min(v1).min(v2).min(v3) - Vec3::splat(margin);
        let hi = v0.max(v1).max(v2).max(v3) + Vec3::splat(margin);
        *slot = TetAabb::from_corners(i as u32, lo, hi);
    }
}

/// Coarse mesh-vs-mesh AABB cull. Each tet mesh has its own root AABB
/// (the union of its tet AABBs); if the roots don't overlap, the
/// downstream per-vertex × per-tet pass can be skipped entirely.
#[must_use]
pub fn tet_mesh_root_aabb(aabbs: &[TetAabb]) -> Option<(Vec3, Vec3)> {
    if aabbs.is_empty() {
        return None;
    }
    let mut lo = aabbs[0].min();
    let mut hi = aabbs[0].max();
    for a in &aabbs[1..] {
        lo = lo.min(a.min());
        hi = hi.max(a.max());
    }
    Some((lo, hi))
}

/// Top-level mesh AABB overlap test.
#[must_use]
pub fn tet_meshes_aabb_overlap(aabbs_a: &[TetAabb], aabbs_b: &[TetAabb]) -> bool {
    let (lo_a, hi_a) = match tet_mesh_root_aabb(aabbs_a) {
        Some(r) => r,
        None => return false,
    };
    let (lo_b, hi_b) = match tet_mesh_root_aabb(aabbs_b) {
        Some(r) => r,
        None => return false,
    };
    lo_a.x <= hi_b.x
        && hi_a.x >= lo_b.x
        && lo_a.y <= hi_b.y
        && hi_a.y >= lo_b.y
        && lo_a.z <= hi_b.z
        && hi_a.z >= lo_b.z
}

/// Closest-point on a tet's interior to `p`. Returns `(distance,
/// outward_normal)` where `distance >= 0` if `p` is outside the tet
/// and `< 0` if inside (penetration depth = `-distance`). Outward
/// normal is the closest-face normal pointing away from the tet
/// centroid (so penalty = `outward_normal * (-distance)` pushes a
/// penetrating vertex out the way it came in).
///
/// CPU oracle for the GPU kernel; both walk the four faces and return
/// the smallest signed distance. We use the canonical
/// `(face_normal, face_plane_d)` parameterisation: each face's plane
/// is `n · x = d` with `n` outward-pointing, so the signed distance is
/// `n · p - d` (positive outside).
///
/// Returns `None` if the tet is degenerate (any face produces a
/// zero-length normal).
#[must_use]
pub fn tet_closest_face(
    p: Vec3,
    v0: Vec3,
    v1: Vec3,
    v2: Vec3,
    v3: Vec3,
) -> Option<(Real, Vec3)> {
    // Tet centroid (used to flip face normals to outward).
    let c = (v0 + v1 + v2 + v3) * 0.25;
    // Four faces, each given as a (a, b, cc) CCW triangle. The cross
    // product `(b - a) × (cc - a)` is the face normal up to sign;
    // we flip if it points toward the centroid.
    let faces = [
        (v1, v2, v3),
        (v0, v3, v2),
        (v0, v1, v3),
        (v0, v2, v1),
    ];
    // The signed distance to a convex tet is `max_f (n_f · p - d_f)`
    // when every n_f points outward — a point is inside the tet iff
    // every n_f · p ≤ d_f, so the max defines the SDF (positive
    // outside, negative inside) and the achieving face's normal is the
    // closest-face outward direction.
    let mut max_signed = -Real::INFINITY;
    let mut max_normal = Vec3::Y;
    for (a, b, cc) in faces {
        let mut n = (b - a).cross(cc - a);
        let len = n.length();
        if len < 1.0e-12 {
            return None;
        }
        n /= len;
        let to_inside = c - a;
        if n.dot(to_inside) > 0.0 {
            n = -n;
        }
        let d_plane = n.dot(a);
        let signed = n.dot(p) - d_plane;
        if signed > max_signed {
            max_signed = signed;
            max_normal = n;
        }
    }
    Some((max_signed, max_normal))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_tet() -> [Vec3; 4] {
        [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(1.0, 0.0, 0.0),
            Vec3::new(0.0, 1.0, 0.0),
            Vec3::new(0.0, 0.0, 1.0),
        ]
    }

    #[test]
    fn aabb_recompute_round_trips() {
        let verts = unit_tet();
        let tet_idx = vec![0u32, 1, 2, 3];
        let mut out = vec![TetAabb::from_corners(0, Vec3::ZERO, Vec3::ZERO); 1];
        recompute_tet_aabbs(&tet_idx, &verts, 0.0, &mut out);
        assert!((out[0].min() - Vec3::ZERO).length() < 1e-9);
        assert!((out[0].max() - Vec3::ONE).length() < 1e-9);
    }

    #[test]
    fn aabb_inflate_expands_symmetrically() {
        // f32-precision tolerance — internal storage is f32 even when
        // `Real == f64`, so 0.1 cannot be represented exactly.
        let a = TetAabb::from_corners(0, Vec3::ZERO, Vec3::ONE).inflated(0.1);
        assert!((a.min() - Vec3::splat(-0.1)).length() < 1e-6);
        assert!((a.max() - Vec3::splat(1.1)).length() < 1e-6);
    }

    #[test]
    fn mesh_aabb_overlap_disjoint_returns_false() {
        let a = vec![TetAabb::from_corners(0, Vec3::ZERO, Vec3::ONE)];
        let b = vec![TetAabb::from_corners(
            0,
            Vec3::splat(2.0),
            Vec3::splat(3.0),
        )];
        assert!(!tet_meshes_aabb_overlap(&a, &b));
    }

    #[test]
    fn mesh_aabb_overlap_touching_returns_true() {
        let a = vec![TetAabb::from_corners(0, Vec3::ZERO, Vec3::ONE)];
        let b = vec![TetAabb::from_corners(0, Vec3::ONE, Vec3::splat(2.0))];
        assert!(tet_meshes_aabb_overlap(&a, &b));
    }

    #[test]
    fn closest_face_inside_returns_negative_signed_distance() {
        let v = unit_tet();
        // Centre of the tet (centroid of unit tet at (0.25, 0.25, 0.25)).
        let p = Vec3::splat(0.25);
        let (signed, _n) = tet_closest_face(p, v[0], v[1], v[2], v[3]).unwrap();
        assert!(signed < 0.0, "centroid should be inside (signed = {signed})");
    }

    #[test]
    fn closest_face_outside_returns_positive_signed_distance() {
        let v = unit_tet();
        // Far above the tet on the +Y axis.
        let p = Vec3::new(0.0, 5.0, 0.0);
        let (signed, n) = tet_closest_face(p, v[0], v[1], v[2], v[3]).unwrap();
        assert!(signed > 0.0, "outside point should have signed > 0");
        // Outward normal should point roughly upward (+Y).
        assert!(n.y > 0.0, "outward normal should have +Y component, got {n:?}");
    }
}
