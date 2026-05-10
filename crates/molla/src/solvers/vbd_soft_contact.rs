//! Cross-mesh soft-vs-soft contact (VT + EE) — Stages 33 / 33.1.
//!
//! ### Stage 33 — Vertex-Triangle (VT) contact
//!
//! Walks every soft-surface vertex through the cross-mesh
//! [`crate::geometry::SoftBvh`] and emits one [`VtContact`] per
//! penetrating (or near-touching) vertex/triangle pair. The VBD
//! per-color solver consumes these contacts and adds a Hookean
//! penalty energy on the vertex side + an opposing barycentric-
//! weighted force distribution on the triangle's three vertices
//! (Newton's third law).
//!
//! ### Stage 33.1 — Edge-Edge (EE) contact
//!
//! VT alone misses cloth folds passing sideways through a tet.
//! EE catches the case where two soft edges (one from each mesh)
//! come within `margin` of each other. The current commit ships
//! the [`SoftContact`] enum + the dispatch path so subsequent
//! commits can drop in the EE narrow-phase + projection without
//! touching call sites.
//!
//! Same-surface pairs are filtered out at this stage (self-collision
//! lands in Stage 33.3 with proper adjacency exclusion).

use crate::core::Result;
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

/// One EE (edge-edge) contact emitted by [`gather_ee_contacts`].
/// Both edges are referenced by their global edge index (into
/// `Model.soft_surface_edges`) plus the surface they belong to so the
/// projection sweep can apply the right per-vertex inverse mass.
#[derive(Copy, Clone, Debug)]
pub struct EeContact {
    /// Global edge index for the first edge (into
    /// `Model.soft_surface_edges`).
    pub edge_a_global_id: u32,
    /// Surface owning edge A (index into `Model.soft_surfaces`).
    pub edge_a_surface: u32,
    /// Global edge index for the second edge.
    pub edge_b_global_id: u32,
    /// Surface owning edge B.
    pub edge_b_surface: u32,
    /// Barycentric coordinate on edge A, `0 ≤ t_a ≤ 1`. The closest
    /// point on edge A is `(1 - t_a) · A0 + t_a · A1`.
    pub t_a: Real,
    /// Barycentric coordinate on edge B, `0 ≤ t_b ≤ 1`.
    pub t_b: Real,
    /// Unit-length contact normal pointing from edge B toward
    /// edge A. The penalty pushes edge-A's verts along `+normal`
    /// and edge-B's verts along `-normal`.
    pub normal: [Real; 3],
    /// Penetration depth: `(margin - segment_distance)` if positive,
    /// 0 otherwise.
    pub depth: Real,
}

/// Tagged union over the two cross-mesh soft contact flavours. The
/// single [`gather_soft_contacts`] / [`project_soft_contacts`] entry
/// points operate on `Vec<SoftContact>` so future contact kinds (e.g.
/// hybrid VE) can be threaded through the same pipeline without yet
/// another parallel type.
#[derive(Copy, Clone, Debug)]
pub enum SoftContact {
    Vt(VtContact),
    Ee(EeContact),
}

/// Gather both VT and EE contacts for the current state. Internally
/// concatenates the VT + EE narrow-phase results. Callers consume
/// `Vec<SoftContact>` directly via [`project_soft_contacts`].
///
/// Stage 33: VT only — the EE half lights up in Stage 33.1 once the
/// edge extraction + segment-segment narrow-phase land.
pub fn gather_soft_contacts(
    model: &Model,
    state: &State,
    bvh: &SoftBvh,
    margin: Real,
) -> Vec<SoftContact> {
    let mut out: Vec<SoftContact> = Vec::new();
    for c in gather_vt_contacts(model, state, bvh, margin) {
        out.push(SoftContact::Vt(c));
    }
    for c in gather_ee_contacts(model, state, bvh, margin) {
        out.push(SoftContact::Ee(c));
    }
    out
}

/// Walk every soft-surface vertex; for each, BVH-query for nearby
/// triangles within `margin`; for each candidate tri (skipping
/// same-surface), compute the closest point on the triangle, the
/// signed distance, and the contact normal; emit a [`VtContact`] when
/// the depth is positive.
///
/// Per-vertex deduplication: only the **deepest** contact per
/// `(vertex, other_surface)` pair is emitted. Multiple candidate tris
/// near one vertex would otherwise stack penalty forces and over-
/// correct the position. This matches Newton's `vbd_self_contact`
/// strategy of "one contact per vertex per other-surface".
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
            // Best contact per `(vertex, other_surface)`: only the
            // deepest penetration survives. Stacking multiple tris
            // near one vertex would compound the penalty and over-
            // correct.
            let mut best_per_surface: Vec<Option<VtContact>> =
                vec![None; model.soft_surfaces.len()];
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

                let new_contact = VtContact {
                    vertex_global_id: v_global as u32,
                    vertex_surface: s_idx as u32,
                    tri_global_id: tri_global,
                    tri_surface,
                    bary: [bary.x, bary.y, bary.z],
                    normal: [normal.x, normal.y, normal.z],
                    depth,
                };
                let slot = &mut best_per_surface[tri_surface as usize];
                match slot {
                    Some(existing) if existing.depth >= depth => {}
                    _ => *slot = Some(new_contact),
                }
            }
            for slot in best_per_surface {
                if let Some(c) = slot {
                    out.push(c);
                }
            }
        }
    }
    out
}

/// EE narrow-phase — Stage 33.1 stub. Returns no contacts; the real
/// implementation lands once the edge extraction + segment query
/// support are in place.
pub fn gather_ee_contacts(
    _model: &Model,
    _state: &State,
    _bvh: &SoftBvh,
    _margin: Real,
) -> Vec<EeContact> {
    Vec::new()
}

/// Multi-sweep position projection for `Vec<SoftContact>`. Refits the
/// BVH each sweep, re-gathers contacts, and applies the per-variant
/// projection. Returns the active contact count from the final sweep.
pub fn project_soft_contacts(
    model: &Model,
    state: &mut State,
    bvh: &mut SoftBvh,
    margin: Real,
    sweeps: usize,
) -> Result<usize> {
    if model.soft_surfaces.is_empty() {
        return Ok(0);
    }
    let inv_mass = model.particle_inv_mass.host()?.to_vec();
    let mut last_active: usize = 0;
    let n_sweeps = sweeps.max(1);

    // Per-vertex accumulated normal direction (for velocity damping
    // at the end). VT and EE contacts both contribute. The end-of-
    // routine pass projects out any inward-facing velocity component.
    let mut contact_normal_sum: Vec<Vec3> = vec![Vec3::ZERO; model.particle_count];

    for _ in 0..n_sweeps {
        // Refit the BVH from the current positions. Cheap when the
        // tri count hasn't changed (just AABB recompute + bottom-up
        // refit); falls back to a full rebuild on a topology change.
        bvh.refit(model, state);
        let contacts = gather_soft_contacts(model, state, bvh, margin);
        last_active = contacts.len();
        if contacts.is_empty() {
            break;
        }
        // Apply position projection. Read q via host_mut() once per
        // sweep so the Jacobi update sees a consistent snapshot.
        let q = state.particle_q.host_mut()?;
        let tris: &[[u32; 3]] = &model.soft_surface_tris;
        for c in &contacts {
            match c {
                SoftContact::Vt(vt) => {
                    project_one_vt(vt, q, tris, &inv_mass, &mut contact_normal_sum);
                }
                SoftContact::Ee(ee) => {
                    project_one_ee(ee, q, model, &inv_mass, &mut contact_normal_sum);
                }
            }
        }
    }

    // Velocity damping along the contact normal. For each particle
    // that participated in a contact at any sweep this substep,
    // compute the dominant normal direction and zero out the
    // velocity component pointing INTO the contact. Same logic
    // shared by VT + EE so we do it once at the end.
    let any_contact = contact_normal_sum
        .iter()
        .any(|n| n.length_squared() > 1.0e-12);
    if any_contact {
        let qd = state.particle_qd.host_mut()?;
        for i in 0..model.particle_count {
            let n_sum = contact_normal_sum[i];
            let n_len2 = n_sum.length_squared();
            if n_len2 < 1.0e-12 {
                continue;
            }
            if inv_mass[i] <= 0.0 {
                continue;
            }
            let n = n_sum / n_len2.sqrt();
            let v = qd[i];
            let vn = v.dot(n);
            if vn < 0.0 {
                qd[i] = v - n * vn;
            }
        }
    }

    Ok(last_active)
}

/// Back-compat wrapper around [`project_soft_contacts`] for any
/// caller that only wants VT projection. Prefer
/// [`project_soft_contacts`] for new code so EE catches cloth-fold
/// pokethroughs.
pub fn project_vt_contacts(
    model: &Model,
    state: &mut State,
    bvh: &mut SoftBvh,
    margin: Real,
    sweeps: usize,
) -> Result<usize> {
    project_soft_contacts(model, state, bvh, margin, sweeps)
}

fn project_one_vt(
    c: &VtContact,
    q: &mut [Vec3],
    tris: &[[u32; 3]],
    inv_mass: &[Real],
    contact_normal_sum: &mut [Vec3],
) {
    let v = c.vertex_global_id as usize;
    let tri = tris[c.tri_global_id as usize];
    let i0 = tri[0] as usize;
    let i1 = tri[1] as usize;
    let i2 = tri[2] as usize;
    let n_vec = Vec3::new(c.normal[0], c.normal[1], c.normal[2]);
    contact_normal_sum[v] += n_vec;
    contact_normal_sum[i0] -= n_vec;
    contact_normal_sum[i1] -= n_vec;
    contact_normal_sum[i2] -= n_vec;

    let w_v = inv_mass[v];
    let w0 = inv_mass[i0];
    let w1 = inv_mass[i1];
    let w2 = inv_mass[i2];
    let b0 = c.bary[0];
    let b1 = c.bary[1];
    let b2 = c.bary[2];
    let w_tri_eff = b0 * b0 * w0 + b1 * b1 * w1 + b2 * b2 * w2;
    let total = w_v + w_tri_eff;
    if total <= 1.0e-20 {
        return;
    }
    let push = c.depth * 0.5;
    let alpha = w_v / total;
    if w_v > 0.0 {
        q[v] += n_vec * (alpha * push);
    }
    if w0 > 0.0 {
        let f0 = b0 * b0 * w0 / total;
        q[i0] -= n_vec * (f0 * push);
    }
    if w1 > 0.0 {
        let f1 = b1 * b1 * w1 / total;
        q[i1] -= n_vec * (f1 * push);
    }
    if w2 > 0.0 {
        let f2 = b2 * b2 * w2 / total;
        q[i2] -= n_vec * (f2 * push);
    }
}

/// EE projection — Stage 33.1 stub. Currently unreachable because
/// [`gather_ee_contacts`] returns empty; once the EE narrow-phase
/// lands the body distributes the depth across the four edge verts
/// via segment-segment barycentric weights.
fn project_one_ee(
    _c: &EeContact,
    _q: &mut [Vec3],
    _model: &Model,
    _inv_mass: &[Real],
    _contact_normal_sum: &mut [Vec3],
) {
}

/// Closest-point + barycentric weights wrapper. Wraps
/// [`crate::geometry::closest_point_on_triangle`] which only returns
/// the closest point + squared distance — we recompute the
/// barycentric coordinates on the closest point so the VT penalty
/// can distribute the reaction force across the triangle's verts
/// proportionally.
fn closest_point_on_triangle_bary(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> (Vec3, Vec3) {
    let (_, q) = closest_point_on_triangle(p, a, b, c);
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
        return (q, Vec3::splat(1.0 / 3.0));
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    let bary = Vec3::new(u.max(0.0), v.max(0.0), w.max(0.0));
    let sum = bary.x + bary.y + bary.z;
    if sum > 1.0e-12 {
        (q, bary / sum)
    } else {
        (q, Vec3::splat(1.0 / 3.0))
    }
}
