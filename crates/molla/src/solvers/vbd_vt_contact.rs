//! Vertex-triangle (VT) cross-mesh contact gather — Stage 33.
//!
//! Walks every soft-surface vertex through the cross-mesh
//! [`crate::geometry::SoftBvh`] and emits one [`VtContact`] per
//! penetrating (or near-touching) vertex/triangle pair. The VBD
//! per-color solver consumes these contacts and adds a Hookean
//! penalty energy on the vertex side + an opposing barycentric-
//! weighted force distribution on the triangle's three vertices
//! (Newton's third law).
//!
//! Same-surface pairs are filtered out at this stage (self-collision
//! lands in Stage 33.3 with proper adjacency exclusion).

use crate::geometry::{closest_point_on_triangle, SoftBvh};
use crate::math::{Real, Vec3};
use crate::sim::{Model, State};

/// One VT contact emitted by [`gather_vt_contacts`]. Both `vertex_*`
/// and `tri_*` references are global indices (into the flat
/// soft-surface arrays + `Model.particle_q`).
#[derive(Copy, Clone, Debug)]
pub struct VtContact {
    /// Global particle index of the contact vertex (into
    /// `Model.particle_q`).
    pub vertex_global_id: u32,
    /// Index into `Model.soft_surfaces` — the surface owning the
    /// vertex. Used by the per-color solve to skip same-surface
    /// contacts (until self-collision Stage 33.3).
    pub vertex_surface: u32,
    /// Global tri index (into `Model.soft_surface_tris`).
    pub tri_global_id: u32,
    /// Index into `Model.soft_surfaces` — the surface owning the tri.
    pub tri_surface: u32,
    /// Barycentric weights of the closest-point on the triangle.
    /// `bary[0..3]` correspond to triangle vertices `(v0, v1, v2)` in
    /// the order stored in `Model.soft_surface_tris`. Always
    /// `bary[0] + bary[1] + bary[2] ≈ 1`.
    pub bary: [Real; 3],
    /// Outward-facing contact normal at the contact (unit length),
    /// pointing from the triangle toward the vertex side. The VBD
    /// penalty pushes the vertex along `+normal` and the triangle
    /// vertices along `-normal · bary[i]`.
    pub normal: [Real; 3],
    /// Penetration depth: `(margin - distance_to_tri)` if positive,
    /// 0 otherwise. Larger `depth` ⇒ stronger penalty force.
    pub depth: Real,
}

/// Walk every soft-surface vertex; for each, BVH-query for nearby
/// triangles within `margin`; for each candidate tri (skipping
/// same-surface), compute the closest point on the triangle, the
/// signed distance, and the contact normal; emit a [`VtContact`] when
/// the depth is positive.
///
/// `margin` is the contact buffer (typically `particle_radius +
/// contact_slop`). A vertex `margin` beyond the tri still triggers
/// a contact so the penalty can engage *before* full interpenetration.
pub fn gather_vt_contacts(
    model: &Model,
    state: &State,
    bvh: &SoftBvh,
    margin: Real,
) -> Vec<VtContact> {
    let mut out = Vec::new();
    if model.soft_surfaces.is_empty() {
        return out;
    }
    let q = match state.particle_q.host() {
        Ok(q) => q,
        Err(_) => return out,
    };
    let tris: &[[u32; 3]] = &model.soft_surface_tris;
    if tris.is_empty() {
        return out;
    }
    let inv_mass = match model.particle_inv_mass.host() {
        Ok(m) => m,
        Err(_) => return out,
    };

    // For each soft surface, walk its vertex range.
    for (s_idx, s) in model.soft_surfaces.iter().enumerate() {
        let v_lo = s.vertex_offset as usize;
        let v_hi = v_lo + s.vertex_count as usize;
        for v_global in v_lo..v_hi {
            // Pinned vertices still produce contacts (the *other*
            // surface's tri verts get a force) but their own block
            // solve skips the displacement.
            // We still need to surface them so the tri side moves.
            let _ = inv_mass[v_global];

            let p = q[v_global];
            let candidates = bvh.query_point(p, margin);
            if candidates.is_empty() {
                continue;
            }
            for tri_global in candidates {
                // Same-surface filter: defer to Stage 33.3.
                let tri_surface = bvh.tri_to_surface[tri_global as usize];
                if tri_surface == s_idx as u32 {
                    continue;
                }
                let tri = tris[tri_global as usize];
                let v0 = q[tri[0] as usize];
                let v1 = q[tri[1] as usize];
                let v2 = q[tri[2] as usize];
                // Topological filter: skip when the vertex IS one of
                // the triangle's own corners. This can happen when
                // two surfaces accidentally share a particle (the
                // builder allows particle reuse across meshes); the
                // contact would be vacuous + numerically dangerous.
                if tri[0] as usize == v_global
                    || tri[1] as usize == v_global
                    || tri[2] as usize == v_global
                {
                    continue;
                }

                // Closest point + barycentric weights.
                let (q_close, bary) = closest_point_on_triangle_bary(p, v0, v1, v2);
                let diff = p - q_close;
                let dist = diff.length();
                if !dist.is_finite() {
                    continue;
                }
                if dist >= margin {
                    continue;
                }
                let depth = margin - dist;
                // Normal direction: from the triangle's surface to
                // the vertex side. When the vertex is exactly on the
                // tri, fall back to the triangle's geometric normal.
                let normal = if dist > 1.0e-9 {
                    diff / dist
                } else {
                    let geo = (v1 - v0).cross(v2 - v0);
                    let len = geo.length();
                    if len > 1.0e-9 {
                        geo / len
                    } else {
                        Vec3::Y
                    }
                };

                out.push(VtContact {
                    vertex_global_id: v_global as u32,
                    vertex_surface: s_idx as u32,
                    tri_global_id: tri_global,
                    tri_surface,
                    bary: [bary.x, bary.y, bary.z],
                    normal: [normal.x, normal.y, normal.z],
                    depth,
                });
            }
        }
    }
    out
}

/// Closest-point + barycentric weights wrapper. Wraps
/// [`crate::geometry::closest_point_on_triangle`] which only returns
/// the closest point + squared distance — we recompute the
/// barycentric coordinates on the closest point so the VT penalty
/// can distribute the reaction force across the triangle's verts
/// proportionally.
fn closest_point_on_triangle_bary(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Vec3) {
    let (_, q) = closest_point_on_triangle(p, a, b, c);
    // Barycentric of `q` w.r.t. `(a, b, c)` via the standard
    // projected-area formulation (Crane 2018, "Discrete Differential
    // Geometry" §4.3): solve
    //   q = α·a + β·b + γ·c, α + β + γ = 1
    // by computing per-corner triangle areas relative to `q`.
    let v0 = b - a;
    let v1 = c - a;
    let v2 = q - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1.0e-20 {
        // Degenerate triangle — return uniform weights so the force
        // distribution doesn't blow up.
        return (q, Vec3::splat(1.0 / 3.0));
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    // Clamp tiny negative weights from edge / vertex regions; the
    // total still equals 1 because we re-derive `u` from the
    // residual.
    let bary = Vec3::new(u.max(0.0), v.max(0.0), w.max(0.0));
    let sum = bary.x + bary.y + bary.z;
    if sum > 1.0e-12 {
        (q, bary / sum)
    } else {
        (q, Vec3::splat(1.0 / 3.0))
    }
}
