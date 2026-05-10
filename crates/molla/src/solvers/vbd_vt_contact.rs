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
        // Stage 33.3 — capture the per-surface self-collision flag
        // here so the inner loop can branch cheaply (vs re-reading
        // the registry per candidate tri).
        let self_coll = s.self_collision;
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
            // Stage 33.3 — incident-tri set lookup for the current
            // vertex. Empty when no soft surface owns the vertex
            // (shouldn't happen given the outer surface walk, but
            // keeps the indexing safe).
            let inc_lo = model
                .soft_vertex_incident_tris_offsets
                .get(v_global)
                .copied()
                .unwrap_or(0) as usize;
            let inc_hi = model
                .soft_vertex_incident_tris_offsets
                .get(v_global + 1)
                .copied()
                .unwrap_or(inc_lo as u32) as usize;
            let incident_tris: &[u32] = &model.soft_vertex_incident_tris[inc_lo..inc_hi];
            for tri_global in candidates {
                let tri_surface = bvh.tri_to_surface[tri_global as usize];
                // Stage 33.3 — same-surface filter. Three regimes:
                //   (a) cross-mesh (tri_surface != s_idx): always emit.
                //   (b) same-surface, self-collision OFF: skip
                //       (preserves Stage 33's default behaviour).
                //   (c) same-surface, self-collision ON: skip iff the
                //       triangle is topologically incident to the
                //       vertex (vertex IS one of the tri's three
                //       corners — a degenerate, always-in-contact
                //       pair). Adjacent-but-not-incident tris (one
                //       ring away) ARE valid contacts — see Stage 33.3
                //       caveat list re: thin-shell folds.
                if tri_surface == s_idx as u32 {
                    if !self_coll {
                        continue;
                    }
                    if incident_tris.contains(&tri_global) {
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

/// Jacobi-style position projection of every active VT contact.
///
/// For each contact, computes a separation displacement `Δ = depth · n`
/// and distributes it as a mass-weighted impulse across the four
/// participating vertices: the contact vertex moves `+α · Δ` while
/// each tri vertex `i` moves `-β_i · Δ` with weights summing to 1
/// (Newton's third law in position-projection form). The weights
/// follow the standard inverse-mass partitioning used by XPBD-style
/// contact solvers:
///   `total_w = w_v + Σ_i (b_i² · w_{tri_i})`
///   `α       = w_v / total_w`
///   `β_i     = (b_i² · w_{tri_i}) / total_w`
///
/// where `w_v = 1/m_v`, `b_i` is the i-th barycentric weight, and the
/// `b_i²` term comes from the per-vertex Hessian contribution `+k · b²
/// · n n^T` (so the projection is consistent with the implicit-
/// integrator block-solve interpretation).
///
/// Multi-sweep so successive contacts converge: the v1 workload
/// (jelly cube on cloth) settles in 4-6 sweeps for typical mass
/// ratios. Returns the number of *active* contacts (depth > 0) for
/// the caller to surface in diagnostics.
pub fn project_vt_contacts(
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

    // Snapshot pre-projection positions so we can update velocities
    // after the position correction. The implicit-Euler `qd = (q -
    // q_prev_step) / dt` semantics need the velocity to reflect the
    // contact correction, otherwise the next substep's predict step
    // shoots the vertex back through the contact.
    let q_pre = state.particle_q.host()?.to_vec();
    // Track per-vertex accumulated normal direction (for velocity
    // damping at the end). We sum the most-recent normal each sweep;
    // for a single contact this is just that contact's normal.
    let mut contact_normal_sum: Vec<Vec3> = vec![Vec3::ZERO; model.particle_count];

    for _ in 0..n_sweeps {
        // Refit the BVH from the current positions. Cheap when the
        // tri count hasn't changed (just AABB recompute + bottom-up
        // refit); falls back to a full rebuild on a topology change.
        bvh.refit(model, state);
        let contacts = gather_vt_contacts(model, state, bvh, margin);
        last_active = contacts.len();
        if contacts.is_empty() {
            break;
        }
        // Apply position projection. Read q via host_mut() once per
        // sweep so the Jacobi update sees a consistent snapshot.
        let q = state.particle_q.host_mut()?;
        let tris: &[[u32; 3]] = &model.soft_surface_tris;
        for c in &contacts {
            let v = c.vertex_global_id as usize;
            let tri = tris[c.tri_global_id as usize];
            let i0 = tri[0] as usize;
            let i1 = tri[1] as usize;
            let i2 = tri[2] as usize;
            let n_vec = Vec3::new(c.normal[0], c.normal[1], c.normal[2]);
            // Accumulate normal for later velocity damping. We track
            // the last contact direction per vertex (for v1 the
            // typical case is one dominant contact per vertex).
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
                continue; // all four are pinned — nothing to do.
            }
            // Underrelaxation: take a fraction of the depth each
            // sweep so the multi-sweep Jacobi converges from above
            // (no overshoot) for typical mass ratios. A full depth
            // correction can cascade across neighbouring contacts on
            // the same particle and produce visible cloth tearing.
            let push = c.depth * 0.5;
            let alpha = w_v / total;
            // Vertex side: move +α·push along n.
            if w_v > 0.0 {
                q[v] += n_vec * (alpha * push);
            }
            // Tri side: distribute -(1-α)·push across the three tri
            // verts via the inverse-mass weighted barycentric
            // partition. The `b_i² · w_i / total` term is the
            // Hessian-consistent share (matches the implicit-
            // integrator AVBD penalty's per-vertex Hessian
            // contribution `+k·b_i²·n n^T`).
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
    }

    // Velocity damping along the contact normal. For each particle
    // that participated in a contact at any sweep this substep,
    // compute the dominant normal direction and zero out the
    // velocity component pointing INTO the contact (i.e. the
    // component along -contact_normal). This prevents the predict
    // step from re-driving the vertex through the contact next
    // substep.
    //
    // We run this even when the last sweep returned 0 contacts —
    // the per-particle `contact_normal_sum` aggregates over every
    // sweep, so a particle whose contact resolved mid-sweep still
    // gets the velocity correction.
    let any_contact = contact_normal_sum
        .iter()
        .any(|n| n.length_squared() > 1.0e-12);
    if any_contact {
        let _ = q_pre; // unused; documents pre-projection snapshot intent
        let qd = state.particle_qd.host_mut()?;
        for i in 0..model.particle_count {
            let n_sum = contact_normal_sum[i];
            let n_len2 = n_sum.length_squared();
            if n_len2 < 1.0e-12 {
                continue;
            }
            // Pinned particles never integrate; skip the qd update.
            if inv_mass[i] <= 0.0 {
                continue;
            }
            let n = n_sum / n_len2.sqrt();
            // The vertex side of a contact has its dominant n
            // pointing AWAY from the tri (so the vertex is being
            // pushed in the +n direction). A velocity component
            // along -n is "diving in"; project it out.
            let v = qd[i];
            let vn = v.dot(n);
            if vn < 0.0 {
                qd[i] = v - n * vn;
            }
        }
    }

    Ok(last_active)
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
