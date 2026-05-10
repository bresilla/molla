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
//! VT alone misses cloth folds passing sideways through a tet —
//! e.g. a rope sliding edge-first through a triangle gap, where no
//! single vertex pierces a tri but two edges cross. EE catches the
//! case where two soft edges (one from each mesh) come within
//! `margin` of each other and applies a position projection that
//! pushes them apart along the edge-cross normal.
//!
//! Same-surface pairs are filtered out at this stage (self-collision
//! lands in Stage 33.3 with proper adjacency exclusion).

use crate::core::Result;
use crate::geometry::{closest_point_on_triangle, vt_ccd, vt_ccd_normal, SoftBvh};
use crate::math::{Real, Vec3};
use crate::sim::{Model, State};

/// One VT contact emitted by [`gather_vt_contacts`]. Both `vertex_*`
/// and `tri_*` references are global indices (into the flat
/// soft-surface arrays + `Model.particle_q`).
#[derive(Copy, Clone, Debug)]
pub struct VtContact {
    pub vertex_global_id: u32,
    pub vertex_surface: u32,
    pub tri_global_id: u32,
    pub tri_surface: u32,
    pub bary: [Real; 3],
    pub normal: [Real; 3],
    pub depth: Real,
    /// Stage 33.5 — Coulomb friction coefficient at the contact,
    /// `μ = sqrt(μ_a · μ_b)` (geometric mean).
    pub friction: Real,
    /// Stage 33.5 — restitution coefficient at the contact,
    /// `min(e_a, e_b)`.
    pub restitution: Real,
    /// Stage 33.5 — warm-start tangential impulse (m/s units).
    pub prev_friction_impulse: [Real; 3],
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
    // For each soft surface, walk its vertex range. Pinned vertices
    // still produce contacts (the *other* surface's tri verts get a
    // force from the projection sweep) but their own block solve
    // skips the displacement; the inverse-mass weighting in the
    // projection step picks up the pin via `w_v = 0`.
    for (s_idx, s) in model.soft_surfaces.iter().enumerate() {
        let v_lo = s.vertex_offset as usize;
        let v_hi = v_lo + s.vertex_count as usize;
        for v_global in v_lo..v_hi {
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
                // Stage 33.3 — same-surface filter:
                //   * cross-mesh   → emit
                //   * same surface AND self_collision OFF → skip
                //   * same surface AND self_collision ON  → skip only
                //     incident triangles (vertex IS one of the corners)
                //     so non-adjacent same-surface tris (folded layers)
                //     do produce contacts.
                let tri_surface = bvh.tri_to_surface[tri_global as usize];
                if tri_surface == s_idx as u32 {
                    if !s.self_collision {
                        continue;
                    }
                    // self-collision ON: filter out incident tris.
                    let off = model.soft_vertex_incident_tris_offsets[v_global] as usize;
                    let off_next =
                        model.soft_vertex_incident_tris_offsets[v_global + 1] as usize;
                    if model.soft_vertex_incident_tris[off..off_next].contains(&tri_global) {
                        continue;
                    }
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

                // Stage 33.5 — friction/restitution combination.
                let s_v = &model.soft_surfaces[s_idx];
                let s_t = &model.soft_surfaces[tri_surface as usize];
                let mu = (s_v.friction.max(0.0) * s_t.friction.max(0.0)).sqrt();
                let restitution = s_v.restitution.min(s_t.restitution).clamp(0.0, 1.0);
                let new_contact = VtContact {
                    vertex_global_id: v_global as u32,
                    vertex_surface: s_idx as u32,
                    tri_global_id: tri_global,
                    tri_surface,
                    bary: [bary.x, bary.y, bary.z],
                    normal: [normal.x, normal.y, normal.z],
                    depth,
                    friction: mu,
                    restitution,
                    prev_friction_impulse: [0.0; 3],
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

/// EE narrow-phase — Stage 33.1.
///
/// Walks every soft-surface edge; broad-phases against the cross-mesh
/// BVH via [`SoftBvh::query_segment`], then derives candidate edges
/// from the returned tris. For each candidate cross-surface edge,
/// runs a segment-segment closest-point computation; emits an
/// [`EeContact`] when the distance falls within `margin`.
///
/// Same-surface pairs are skipped (self-collision is Stage 33.3);
/// edge pairs sharing a vertex are skipped (they meet at a single
/// point — that's a VT contact, not an edge crossing); near-parallel
/// pairs are skipped (the segment-segment minimization is ill-
/// conditioned and the resulting contact is already covered by VT
/// at the endpoints).
///
/// Per-edge deduplication: only the deepest contact per
/// `(edge, other_surface)` pair is emitted. This mirrors VT's
/// `(vertex, other_surface)` dedup — multiple candidate partners
/// near one edge would otherwise stack penalty forces and over-
/// correct the position.
pub fn gather_ee_contacts(
    model: &Model,
    state: &State,
    bvh: &SoftBvh,
    margin: Real,
) -> Vec<EeContact> {
    let mut out = Vec::new();
    if model.soft_surfaces.is_empty() {
        return out;
    }
    let edges: &[[u32; 2]] = &model.soft_surface_edges;
    if edges.is_empty() {
        return out;
    }
    let tris: &[[u32; 3]] = &model.soft_surface_tris;
    let q = match state.particle_q.host() {
        Ok(q) => q,
        Err(_) => return out,
    };

    // Map (min_vert, max_vert) → global edge index. Each tri-AABB
    // hit fans out into 3 candidate edges; we look them up here
    // rather than searching `edges` linearly.
    use std::collections::HashMap;
    let mut edge_lookup: HashMap<(u32, u32), u32> = HashMap::with_capacity(edges.len());
    for (i, e) in edges.iter().enumerate() {
        let key = if e[0] <= e[1] {
            (e[0], e[1])
        } else {
            (e[1], e[0])
        };
        edge_lookup.entry(key).or_insert(i as u32);
    }
    let edge_of = |a: u32, b: u32| -> Option<u32> {
        let k = if a <= b { (a, b) } else { (b, a) };
        edge_lookup.get(&k).copied()
    };

    // Per-edge surface ownership: edges are extracted per-surface
    // sequentially in the builder, so we can resolve via the
    // SoftSurface.edge_offset/edge_count window.
    let mut edge_to_surface = vec![0u32; edges.len()];
    for (s_idx, s) in model.soft_surfaces.iter().enumerate() {
        let lo = s.edge_offset as usize;
        let hi = lo + s.edge_count as usize;
        for e in lo..hi {
            edge_to_surface[e] = s_idx as u32;
        }
    }

    for (s_idx, s) in model.soft_surfaces.iter().enumerate() {
        let e_lo = s.edge_offset as usize;
        let e_hi = e_lo + s.edge_count as usize;
        for e_a_idx in e_lo..e_hi {
            let ea = edges[e_a_idx];
            let a0_id = ea[0] as usize;
            let a1_id = ea[1] as usize;
            let a0 = q[a0_id];
            let a1 = q[a1_id];

            let tri_hits = bvh.query_segment(a0, a1, margin);
            if tri_hits.is_empty() {
                continue;
            }

            let mut best_per_surface: Vec<Option<EeContact>> =
                vec![None; model.soft_surfaces.len()];

            // Dedup candidate edges so we don't run the segment
            // computation twice when two adjacent tris share an edge.
            let mut seen_edges: Vec<u32> = Vec::with_capacity(tri_hits.len() * 3);
            for tri_global in tri_hits {
                let tri_surface = bvh.tri_to_surface[tri_global as usize];
                if tri_surface == s_idx as u32 {
                    continue; // same-surface — defer to Stage 33.3.
                }
                let tri = tris[tri_global as usize];
                for &(va, vb) in &[
                    (tri[0], tri[1]),
                    (tri[1], tri[2]),
                    (tri[2], tri[0]),
                ] {
                    if let Some(eid) = edge_of(va, vb) {
                        if !seen_edges.contains(&eid) {
                            seen_edges.push(eid);
                        }
                    }
                }
            }

            for &e_b_idx in &seen_edges {
                let s_b = edge_to_surface[e_b_idx as usize];
                if s_b == s_idx as u32 {
                    continue;
                }
                let eb = edges[e_b_idx as usize];
                let b0_id = eb[0] as usize;
                let b1_id = eb[1] as usize;
                if a0_id == b0_id
                    || a0_id == b1_id
                    || a1_id == b0_id
                    || a1_id == b1_id
                {
                    continue;
                }
                let b0 = q[b0_id];
                let b1 = q[b1_id];

                let res = match closest_point_segment_segment(a0, a1, b0, b1) {
                    Some(r) => r,
                    None => continue, // near-parallel: defer to VT.
                };
                if !res.dist.is_finite() {
                    continue;
                }
                if res.dist >= margin {
                    continue;
                }
                // Endpoint filter: when the segment-segment minimum
                // sits at an endpoint of either edge, the contact is
                // really vertex-edge (or vertex-vertex), which the VT
                // narrow-phase already covers. Limiting EE to interior
                // (t ∈ (eps, 1-eps)) closest points stops the EE pass
                // from double-counting alongside VT.
                const EE_INTERIOR_EPS: Real = 0.05;
                if res.t_a < EE_INTERIOR_EPS
                    || res.t_a > 1.0 - EE_INTERIOR_EPS
                    || res.t_b < EE_INTERIOR_EPS
                    || res.t_b > 1.0 - EE_INTERIOR_EPS
                {
                    continue;
                }
                let depth = margin - res.dist;

                // Normal: cross-product of the two edge directions
                // (the natural "edge plane" perpendicular). This is
                // the standard EE penalty normal — using
                // `(point_a - point_b) / dist` instead picks up any
                // local geometry artefact and can flip sign once the
                // edges cross, which would PULL them through one
                // another instead of pushing them apart.
                //
                // Sign convention: we want the normal to point from
                // edge B toward edge A. Aligning it with
                // `(point_a - point_b)` does this whenever the edges
                // are on opposite sides of their shared perpendicular
                // (the "haven't crossed yet" regime that EE penalty
                // handles correctly).
                let da = a1 - a0;
                let db = b1 - b0;
                let cross = da.cross(db);
                let cross_len = cross.length();
                let normal = if cross_len > 1.0e-9 {
                    let mut n = cross / cross_len;
                    // Sign-align with `(p_a - p_b)` so the penalty
                    // pushes apart along the same direction the
                    // closest-point displacement implies.
                    let pa_pb = res.point_a - res.point_b;
                    if pa_pb.dot(n) < 0.0 {
                        n = -n;
                    }
                    n
                } else if res.dist > 1.0e-9 {
                    (res.point_a - res.point_b) / res.dist
                } else {
                    Vec3::Y
                };

                let c = EeContact {
                    edge_a_global_id: e_a_idx as u32,
                    edge_a_surface: s_idx as u32,
                    edge_b_global_id: e_b_idx,
                    edge_b_surface: s_b,
                    t_a: res.t_a,
                    t_b: res.t_b,
                    normal: [normal.x, normal.y, normal.z],
                    depth,
                };
                let slot = &mut best_per_surface[s_b as usize];
                match slot {
                    Some(existing) if existing.depth >= depth => {}
                    _ => *slot = Some(c),
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
        let edges: &[[u32; 2]] = &model.soft_surface_edges;
        for c in &contacts {
            match c {
                SoftContact::Vt(vt) => {
                    project_one_vt(vt, q, tris, &inv_mass, &mut contact_normal_sum);
                }
                SoftContact::Ee(ee) => {
                    project_one_ee(ee, q, edges, &inv_mass, &mut contact_normal_sum);
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

// project_vt_contacts (Stage 33.5 friction-aware version) is appended
// below after the helpers it depends on.

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

/// EE projection — Stage 33.1.
///
/// For an EE contact the per-vertex Hessian contribution is
/// `+k · w_i² · n n^T` with `w_i ∈ {1-t_a, t_a, 1-t_b, t_b}` for the
/// four edge endpoints. Distribute the depth across the four verts
/// via the same inverse-mass weighted barycentric partitioning the
/// VT path uses, signed `+n` on edge A and `-n` on edge B.
fn project_one_ee(
    c: &EeContact,
    q: &mut [Vec3],
    edges: &[[u32; 2]],
    inv_mass: &[Real],
    contact_normal_sum: &mut [Vec3],
) {
    let ea = edges[c.edge_a_global_id as usize];
    let eb = edges[c.edge_b_global_id as usize];
    let ia0 = ea[0] as usize;
    let ia1 = ea[1] as usize;
    let ib0 = eb[0] as usize;
    let ib1 = eb[1] as usize;
    let n_vec = Vec3::new(c.normal[0], c.normal[1], c.normal[2]);
    let ta = c.t_a;
    let tb = c.t_b;
    let wa0 = 1.0 - ta;
    let wa1 = ta;
    let wb0 = 1.0 - tb;
    let wb1 = tb;

    // Accumulate the contact normal weighted by each vertex's
    // barycentric share so the post-loop velocity damping in
    // `project_vt_contacts` zeroes out the inward-bound component on
    // EE-touching verts the same way it does for VT-touching verts.
    // Edge A points along `+n` (`+wa{0,1} · n`); edge B along `-n`.
    contact_normal_sum[ia0] += n_vec * wa0;
    contact_normal_sum[ia1] += n_vec * wa1;
    contact_normal_sum[ib0] -= n_vec * wb0;
    contact_normal_sum[ib1] -= n_vec * wb1;

    let m_a0 = inv_mass[ia0];
    let m_a1 = inv_mass[ia1];
    let m_b0 = inv_mass[ib0];
    let m_b1 = inv_mass[ib1];
    let w_eff = wa0 * wa0 * m_a0
        + wa1 * wa1 * m_a1
        + wb0 * wb0 * m_b0
        + wb1 * wb1 * m_b1;
    if w_eff <= 1.0e-20 {
        return;
    }
    // Same underrelaxation factor VT uses; the multi-sweep Jacobi
    // settles for the v1 cloth-fold workload at this rate.
    let push = c.depth * 0.5;
    if m_a0 > 0.0 {
        let f = wa0 * wa0 * m_a0 / w_eff;
        q[ia0] += n_vec * (f * push);
    }
    if m_a1 > 0.0 {
        let f = wa1 * wa1 * m_a1 / w_eff;
        q[ia1] += n_vec * (f * push);
    }
    if m_b0 > 0.0 {
        let f = wb0 * wb0 * m_b0 / w_eff;
        q[ib0] -= n_vec * (f * push);
    }
    if m_b1 > 0.0 {
        let f = wb1 * wb1 * m_b1 / w_eff;
        q[ib1] -= n_vec * (f * push);
    }
}

/// Result of [`closest_point_segment_segment`].
struct ClosestSegResult {
    /// Barycentric on segment A: `0 = a0`, `1 = a1`.
    t_a: Real,
    /// Barycentric on segment B.
    t_b: Real,
    /// Closest point on segment A.
    point_a: Vec3,
    /// Closest point on segment B.
    point_b: Vec3,
    /// Distance between `point_a` and `point_b`.
    dist: Real,
}

/// Closest-point computation between two finite segments
/// `(a0, a1)` and `(b0, b1)`. Returns `None` for near-parallel
/// configurations (`|a·e − b·b|` below `1e-12`); near-parallel
/// pairs are handled by VT contacts at the endpoints — the
/// segment-segment minimum is degenerate (any point along the
/// overlap is a minimum).
///
/// Reference: Real-Time Collision Detection §5.1.9 (Ericson 2005).
fn closest_point_segment_segment(
    a0: Vec3,
    a1: Vec3,
    b0: Vec3,
    b1: Vec3,
) -> Option<ClosestSegResult> {
    let d1 = a1 - a0;
    let d2 = b1 - b0;
    let r = a0 - b0;
    let a = d1.dot(d1);
    let e = d2.dot(d2);
    let f = d2.dot(r);
    if a <= 1.0e-20 && e <= 1.0e-20 {
        // Both segments degenerate to points.
        let dist = (a0 - b0).length();
        return Some(ClosestSegResult {
            t_a: 0.0,
            t_b: 0.0,
            point_a: a0,
            point_b: b0,
            dist,
        });
    }
    let (t_a, t_b);
    if a <= 1.0e-20 {
        t_a = 0.0;
        t_b = (f / e).clamp(0.0, 1.0);
    } else {
        let c = d1.dot(r);
        if e <= 1.0e-20 {
            t_b = 0.0;
            t_a = (-c / a).clamp(0.0, 1.0);
        } else {
            let b = d1.dot(d2);
            let denom = a * e - b * b;
            if denom.abs() < 1.0e-12 {
                return None; // parallel; defer to VT.
            }
            let mut s = (b * f - c * e) / denom;
            s = s.clamp(0.0, 1.0);
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            t_a = s;
            t_b = t;
        }
    }
    let point_a = a0 + d1 * t_a;
    let point_b = b0 + d2 * t_b;
    let dist = (point_a - point_b).length();
    Some(ClosestSegResult {
        t_a,
        t_b,
        point_a,
        point_b,
        dist,
    })
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

/// Stage 33.4 — pre-substep continuous collision detection sweep.
///
/// Operates on raw `(particle_q, particle_qd)` arrays so the VBD CPU
/// substep loop can call us BEFORE building the implicit-Euler
/// predictor. Steps:
///
/// 1. Build the t1-predicted position `q_predict[i] = q[i] + v[i] * dt`
///    for every particle.
/// 2. Refit the cross-mesh BVH using SWEPT AABBs (union of t0 + t1
///    tri AABBs).
/// 3. For every fast-moving vertex (`‖v · dt‖ > velocity_threshold ·
///    dt`):
///    - Query the swept BVH for candidate tris.
///    - For each candidate (skipping same-surface + own-vertex tris),
///      run [`vt_ccd`] to find the earliest TOI.
///    - If a vertex has TOI < 1, snap its position to `TOI - eps`
///      along its trajectory and zero its velocity component diving
///      INTO the contact (along `-normal`). Tangential motion is
///      preserved.
///
/// Returns `(invocations, snaps)`:
/// - `invocations` = total number of `vt_ccd` calls (used by perf
///   tests to verify the velocity threshold gate is working).
/// - `snaps` = number of vertices whose TOI fell strictly inside
///   `[0, 1)` (i.e. were snapped back).
///
/// Same-surface contacts are skipped (self-collision is Stage 33.3).
/// Own-vertex contacts (the vertex IS one of the tri's corners) are
/// skipped — these arise when two soft surfaces share a particle.
#[allow(clippy::too_many_arguments)]
pub fn ccd_pass(
    model: &Model,
    bvh: &mut SoftBvh,
    q_in: &mut [Vec3],
    qd_in: &mut [Vec3],
    margin: Real,
    velocity_threshold: Real,
    dt: Real,
) -> (usize, usize) {
    let mut invocations: usize = 0;
    let mut snaps: usize = 0;

    if model.soft_surfaces.is_empty() {
        return (invocations, snaps);
    }
    let tris: &[[u32; 3]] = &model.soft_surface_tris;
    if tris.is_empty() {
        return (invocations, snaps);
    }

    let n = q_in.len();
    let mut q_t1: Vec<Vec3> = Vec::with_capacity(n);
    for i in 0..n {
        q_t1.push(q_in[i] + qd_in[i] * dt);
    }
    bvh.refit_swept_arrays(q_in, &q_t1, tris);

    let v_threshold_sq = velocity_threshold * velocity_threshold;
    const SNAP_EPS: Real = 1.0e-4;
    let dt2 = dt * dt;

    // Stage 33.4 — per-soft-surface "shock-stop" accumulator. When any
    // vertex of a soft surface is snapped by CCD, we also dampen the
    // contact-normal velocity component of EVERY vertex of that
    // surface in the same direction. This treats the mesh as a
    // quasi-rigid body for impact-arrest purposes, which prevents the
    // common failure mode where an unsnapped top-of-mesh vertex (whose
    // own swept envelope didn't reach the contact tri) re-launches the
    // snapped bottom vertices via tet elastic restoration in the very
    // same substep. v1's per-vertex CCD without this shock-stop loses
    // soft tet bodies entirely on hard impacts; with it the body
    // arrests within 1-2 substeps.
    let mut surface_shock_normal: Vec<Vec3> = vec![Vec3::ZERO; model.soft_surfaces.len()];

    for (s_idx, s) in model.soft_surfaces.iter().enumerate() {
        let v_lo = s.vertex_offset as usize;
        let v_hi = v_lo + s.vertex_count as usize;
        for v_global in v_lo..v_hi {
            let v_t0 = q_in[v_global];
            let v_t1 = q_t1[v_global];
            let dv = v_t1 - v_t0;
            // Velocity gate: a vertex moving slower than threshold
            // can't tunnel meaningfully. The check is in displacement
            // form so the threshold has m/s units regardless of dt.
            if dv.length_squared() < v_threshold_sq * dt2 {
                continue;
            }
            let candidates = bvh.query_point_swept(v_t0, v_t1, margin);
            if candidates.is_empty() {
                continue;
            }
            let mut best_toi: Option<Real> = None;
            let mut best_tri_idx: u32 = 0;
            for tri_global in candidates {
                let tri_surface = bvh.tri_to_surface[tri_global as usize];
                if tri_surface == s_idx as u32 {
                    continue;
                }
                let tri = tris[tri_global as usize];
                if tri[0] as usize == v_global
                    || tri[1] as usize == v_global
                    || tri[2] as usize == v_global
                {
                    continue;
                }
                let a0 = q_in[tri[0] as usize];
                let b0 = q_in[tri[1] as usize];
                let c0 = q_in[tri[2] as usize];
                let a1 = q_t1[tri[0] as usize];
                let b1 = q_t1[tri[1] as usize];
                let c1 = q_t1[tri[2] as usize];
                invocations += 1;
                if let Some(toi) = vt_ccd(v_t0, v_t1, [a0, b0, c0], [a1, b1, c1]) {
                    if toi >= 0.0 && toi < 1.0 {
                        match best_toi {
                            Some(t) if t <= toi => {}
                            _ => {
                                best_toi = Some(toi);
                                best_tri_idx = tri_global;
                            }
                        }
                    }
                }
            }
            if let Some(toi) = best_toi {
                let t_safe = (toi - SNAP_EPS).max(0.0);
                let snapped = v_t0 + dv * t_safe;
                let tri = tris[best_tri_idx as usize];
                let a0 = q_in[tri[0] as usize];
                let b0 = q_in[tri[1] as usize];
                let c0 = q_in[tri[2] as usize];
                let a1 = q_t1[tri[0] as usize];
                let b1 = q_t1[tri[1] as usize];
                let c1 = q_t1[tri[2] as usize];
                let normal = vt_ccd_normal(toi, dv, [a0, b0, c0], [a1, b1, c1]);
                // Snap position to just before contact.
                q_in[v_global] = snapped;
                // Zero the velocity component diving INTO the contact
                // (along `-normal`). Tangential motion is preserved.
                let v_now = qd_in[v_global];
                let vn = v_now.dot(normal);
                if vn < 0.0 {
                    qd_in[v_global] = v_now - normal * vn;
                }
                // Accumulate a shock-stop normal for the whole soft
                // surface. We add the snap normal weighted by the
                // depth of the dive (how fast the vert was moving INTO
                // the contact); the post-pass averages and applies it
                // to every vert of the surface.
                surface_shock_normal[s_idx] += normal * vn.abs();
                snaps += 1;
            }
        }
    }

    // Stage 33.4 — apply the per-surface shock-stop. For each surface
    // that had at least one snapped vertex, kill the contact-normal
    // velocity component of EVERY vertex of the surface (so the
    // unsnapped top of the mesh doesn't re-launch the snapped bottom
    // via tet elastic restoration in the same substep). This is the
    // quasi-rigid impact-arrest approximation.
    for (s_idx, s) in model.soft_surfaces.iter().enumerate() {
        let shock = surface_shock_normal[s_idx];
        let shock_len = shock.length();
        if shock_len < 1.0e-9 {
            continue;
        }
        let n = shock / shock_len;
        let v_lo = s.vertex_offset as usize;
        let v_hi = v_lo + s.vertex_count as usize;
        for v_global in v_lo..v_hi {
            let v_now = qd_in[v_global];
            let vn = v_now.dot(n);
            // Kill the dive component (the vertex moving into the
            // contact normal direction). Tangential motion preserved.
            if vn < 0.0 {
                qd_in[v_global] = v_now - n * vn;
            }
        }
    }
    (invocations, snaps)
}

// Stage 33.5 — friction + restitution (ported from the pre-rename
// vbd_vt_contact.rs). The previous merge included a
// `gather_vt_contacts_with_hint` shim that took a `prev_normal_lut`
// argument and immediately discarded it; it has been removed because
// `gather_vt_contacts` already performs the same-surface filter and
// self-collision exclusion that the hint was meant to refine. A
// future hint-aware gather (anti-tunneling normal sign refinement)
// can ship as a new function rather than a silently-broken wrapper.

/// Stage 33.5 — copy `prev_friction_impulse` from `prev` into the
/// matching contact in `current` (matched by `(vertex_global_id,
/// tri_global_id)`).
pub fn warm_start_friction(prev: &[VtContact], current: &mut [VtContact]) {
    if prev.is_empty() || current.is_empty() {
        return;
    }
    let mut lut: std::collections::HashMap<(u32, u32), usize> =
        std::collections::HashMap::with_capacity(prev.len());
    for (i, c) in prev.iter().enumerate() {
        lut.insert((c.vertex_global_id, c.tri_global_id), i);
    }
    for c in current.iter_mut() {
        if let Some(&i) = lut.get(&(c.vertex_global_id, c.tri_global_id)) {
            c.prev_friction_impulse = prev[i].prev_friction_impulse;
        }
    }
}

/// Stage 33.5 — friction-aware VT projection + Stage 33.1 EE
/// projection. Returns the substep's final VT contact set (with
/// `prev_friction_impulse` populated) so the caller can persist it
/// for warm-starting the next substep.
///
/// Each Jacobi sweep runs:
/// 1. BVH refit against the latest positions.
/// 2. VT gather + friction-aware position projection (Stage 33.5).
/// 3. EE gather + position projection (Stage 33.1) — frictionless
///    in v1; friction along an edge-edge contact is geometrically
///    ambiguous and the EE pass exists primarily to stop ropes /
///    cloth folds from threading through tet faces.
///
/// Both VT and EE feed the per-vertex `contact_normal_sum`, so the
/// post-loop velocity recovery (restitution + Coulomb tangential
/// damping) sees normals from both contact flavours.
pub fn project_vt_contacts(
    model: &Model,
    state: &mut State,
    bvh: &mut SoftBvh,
    margin: Real,
    sweeps: usize,
    dt: Real,
    prev_contacts: &[VtContact],
    q_step_start: &[Vec3],
    qd_step_start: &[Vec3],
) -> Result<Vec<VtContact>> {
    if model.soft_surfaces.is_empty() {
        return Ok(Vec::new());
    }
    let inv_mass = model.particle_inv_mass.host()?.to_vec();
    let n_sweeps = sweeps.max(1);

    let qd_pre = qd_step_start.to_vec();
    let q_pre = q_step_start.to_vec();
    let mut contact_normal_sum: Vec<Vec3> = vec![Vec3::ZERO; model.particle_count];
    let mut restitution_per_v: Vec<Real> = vec![0.0; model.particle_count];
    let mut friction_per_v: Vec<Real> = vec![0.0; model.particle_count];
    let v_eps: Real = 1.0e-3;

    let mut final_contacts: Vec<VtContact> = Vec::new();

    for sweep_idx in 0..n_sweeps {
        bvh.refit(model, state);
        let mut contacts = gather_vt_contacts(model, state, bvh, margin);
        if sweep_idx == 0 {
            warm_start_friction(prev_contacts, &mut contacts);
        } else {
            warm_start_friction(&final_contacts, &mut contacts);
        }
        let ee_contacts = gather_ee_contacts(model, state, bvh, margin);
        if contacts.is_empty() && ee_contacts.is_empty() {
            break;
        }
        let q = state.particle_q.host_mut()?;
        let tris: &[[u32; 3]] = &model.soft_surface_tris;
        let edges: &[[u32; 2]] = &model.soft_surface_edges;
        for c in contacts.iter_mut() {
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
            for &i in &[v, i0, i1, i2] {
                if c.restitution > restitution_per_v[i] {
                    restitution_per_v[i] = c.restitution;
                }
                if c.friction > friction_per_v[i] {
                    friction_per_v[i] = c.friction;
                }
            }
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
                continue;
            }
            let push = c.depth * 0.5;
            let alpha = w_v / total;
            let f0 = b0 * b0 * w0 / total;
            let f1 = b1 * b1 * w1 / total;
            let f2 = b2 * b2 * w2 / total;
            if w_v > 0.0 { q[v] += n_vec * (alpha * push); }
            if w0 > 0.0 { q[i0] -= n_vec * (f0 * push); }
            if w1 > 0.0 { q[i1] -= n_vec * (f1 * push); }
            if w2 > 0.0 { q[i2] -= n_vec * (f2 * push); }

            // Position-form Coulomb friction (first sweep only).
            if c.friction > 0.0 && sweep_idx == 0 {
                let dq_v = q[v] - q_pre[v];
                let dq_tri = (q[i0] - q_pre[i0]) * b0
                    + (q[i1] - q_pre[i1]) * b1
                    + (q[i2] - q_pre[i2]) * b2;
                let mut dq_t = dq_v - dq_tri;
                let dn = dq_t.dot(n_vec);
                dq_t -= n_vec * dn;
                let dq_t_len = dq_t.length();
                if dq_t_len > 1.0e-9 {
                    let coulomb_max = c.friction * push.abs();
                    let smooth = (dq_t_len / (v_eps * dt.max(1.0e-6))).tanh();
                    let mag = (dq_t_len * smooth).min(coulomb_max);
                    let dir = dq_t / dq_t_len;
                    let push_t = -dir * mag;
                    if w_v > 0.0 { q[v] += push_t * alpha; }
                    if w0 > 0.0 { q[i0] -= push_t * f0; }
                    if w1 > 0.0 { q[i1] -= push_t * f1; }
                    if w2 > 0.0 { q[i2] -= push_t * f2; }
                    c.prev_friction_impulse = [push_t.x, push_t.y, push_t.z];
                }
            }
        }
        // EE projection — frictionless penalty + contact-normal
        // accumulation so the post-loop velocity damping handles
        // both VT and EE the same way. Without this, ropes / cloth
        // folds whose edges thread between two tris (no single
        // vertex piercing a tri) tunnel through the other surface.
        for ee in &ee_contacts {
            project_one_ee(ee, q, edges, &inv_mass, &mut contact_normal_sum);
        }
        final_contacts = contacts;
    }

    // Velocity recovery — restitution + Coulomb friction.
    let any_contact = contact_normal_sum.iter().any(|n| n.length_squared() > 1.0e-12);
    if any_contact {
        let qd = state.particle_qd.host_mut()?;
        for i in 0..model.particle_count {
            let n_sum = contact_normal_sum[i];
            let n_len2 = n_sum.length_squared();
            if n_len2 < 1.0e-12 || inv_mass[i] <= 0.0 {
                continue;
            }
            let n = n_sum / n_len2.sqrt();
            let vn_pre = qd_pre[i].dot(n);
            let mut v_now = qd[i];
            let vn_now = v_now.dot(n);
            if vn_pre < 0.0 {
                let e = restitution_per_v[i];
                let target_vn = -e * vn_pre;
                if vn_now < target_vn {
                    v_now += n * (target_vn - vn_now);
                }
            } else if vn_now < 0.0 {
                v_now -= n * vn_now;
            }
            let mu_v = friction_per_v[i];
            if mu_v > 0.0 {
                let v_t = v_now - n * v_now.dot(n);
                let v_t_len = v_t.length();
                if v_t_len > 1.0e-9 {
                    let dv_t_max = mu_v * 9.81 * dt;
                    let dv_t = dv_t_max.min(v_t_len);
                    v_now -= (v_t / v_t_len) * dv_t;
                }
            }
            qd[i] = v_now;
        }
    }

    Ok(final_contacts)
}
