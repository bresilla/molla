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
    /// Stage 33.5 — Coulomb friction coefficient at the contact,
    /// `μ = sqrt(μ_a · μ_b)` (geometric mean of the two surfaces'
    /// friction). 0 disables tangential resistance.
    pub friction: Real,
    /// Stage 33.5 — restitution coefficient at the contact, taken as
    /// `min(e_a, e_b)` (the smaller of the two surfaces' values, the
    /// standard "soft material wins" rule). 0 means perfectly
    /// inelastic; 1 means full velocity reversal.
    pub restitution: Real,
    /// Stage 33.5 — warm-start tangential impulse (m/s units, scaled
    /// by inverse mass for application). Persisted across substeps so
    /// the friction projection converges from a useful starting
    /// guess instead of zero. Carried forward by the solver matching
    /// `(vertex_global_id, tri_global_id)` keys frame-to-frame.
    pub prev_friction_impulse: [Real; 3],
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
///
/// Stage 33.5: each emitted contact's `friction` is the geometric
/// mean of the two soft surfaces' friction coefficients, and
/// `restitution` is the smaller of the two. `prev_friction_impulse`
/// is initialised to zero — the caller (typically [`VbdSolver`]) is
/// responsible for warm-starting it from the previous frame's
/// contacts via [`warm_start_friction`].
pub fn gather_vt_contacts(
    model: &Model,
    state: &State,
    bvh: &SoftBvh,
    margin: Real,
) -> Vec<VtContact> {
    gather_vt_contacts_with_hint(model, state, bvh, margin, &Default::default())
}

/// Internal variant of [`gather_vt_contacts`] that consults a per-
/// `(vertex, tri)` previous-frame normal direction to pick the side
/// of the contact (Stage 33.5 anti-tunneling fix). The public
/// [`gather_vt_contacts`] is a thin wrapper that passes an empty
/// LUT — equivalent to "every contact starts fresh, use the
/// triangle's geometric normal as the outward hint".
pub fn gather_vt_contacts_with_hint(
    model: &Model,
    state: &State,
    bvh: &SoftBvh,
    margin: Real,
    prev_normal_lut: &std::collections::HashMap<(u32, u32), Vec3>,
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
                // Stage 33.5 — anti-tunneling contact normal.
                //
                // Stage 33 v1 used the unsigned `diff/dist`, which
                // flips direction whenever the vertex crosses to the
                // other side of the triangle. With cloth + a fast-
                // moving tet block this caused tunneling: once a
                // vertex landed on the back side, the contact pushed
                // it FURTHER back instead of returning it to the
                // intended side.
                //
                // Stage 33.5 fix: pick a stable "outward" reference
                // per contact and lock the direction in for the
                // contact's lifetime.
                //   - For a brand-new contact (no entry in
                //     `prev_normal_lut`), use the unsigned
                //     `diff/dist` (the side the vertex is currently
                //     on). For a fast vertex that has already
                //     tunneled to the wrong side on its very first
                //     substep, this still picks the wrong side —
                //     use a larger margin / smaller dt to avoid
                //     such cases (Stage 33.4 CCD will fix this
                //     properly).
                //   - For a persisting contact, use the previous
                //     frame's normal direction. If the vertex
                //     crossed during a substep, the signed distance
                //     against `ref_dir` flips negative and the
                //     depth becomes `margin + |dist|`, which pulls
                //     the vertex back through the surface to the
                //     correct side.
                let raw_normal = if dist > 1.0e-9 {
                    diff / dist
                } else {
                    let geo = (v1 - v0).cross(v2 - v0);
                    let geo_len = geo.length();
                    if geo_len > 1.0e-9 {
                        geo / geo_len
                    } else {
                        Vec3::Y
                    }
                };
                let ref_dir = match prev_normal_lut
                    .get(&(v_global as u32, tri_global))
                {
                    Some(&n) => n,
                    None => raw_normal, // first contact: trust the current side
                };
                let aligned_with_ref = raw_normal.dot(ref_dir) >= 0.0;
                let normal = if aligned_with_ref {
                    raw_normal
                } else {
                    -raw_normal
                };
                let signed_dist = if aligned_with_ref { dist } else { -dist };
                // Effective penetration depth: tunneled vertices
                // (signed_dist < 0) need `margin - signed_dist =
                // margin + |dist|` push to clear the surface. Clamp
                // at `4·margin` so a wildly tunneled vertex doesn't
                // explode neighbours (Newton's "never advance more
                // than ~one contact-shell width per substep" rule).
                let depth = (margin - signed_dist).min(4.0 * margin);

                // Stage 33.5: friction = sqrt(μ_a · μ_b), restitution
                // = min(e_a, e_b). Look up both surfaces' coefficients
                // from the registry.
                let mu_v = model.soft_surfaces[s_idx].friction.max(0.0);
                let mu_t = model.soft_surfaces[tri_surface as usize]
                    .friction
                    .max(0.0);
                let friction = (mu_v * mu_t).sqrt();
                let e_v = model.soft_surfaces[s_idx].restitution.clamp(0.0, 1.0);
                let e_t = model.soft_surfaces[tri_surface as usize]
                    .restitution
                    .clamp(0.0, 1.0);
                let restitution = e_v.min(e_t);
                let new_contact = VtContact {
                    vertex_global_id: v_global as u32,
                    vertex_surface: s_idx as u32,
                    tri_global_id: tri_global,
                    tri_surface,
                    bary: [bary.x, bary.y, bary.z],
                    normal: [normal.x, normal.y, normal.z],
                    depth,
                    friction,
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
/// ratios.
///
/// **Stage 33.5 (friction + restitution).** After the normal
/// correction each contact applies a Coulomb friction position
/// correction in the tangent plane:
/// - `Δq_t_rel = (q_v - q_v_pre) - Σ_i b_i · (q_tri_i - q_tri_i_pre)` projected
///   to the contact tangent plane (`Δq_t_rel ← Δq_t_rel - (Δq_t_rel · n) n`).
/// - The friction position correction is `-Δq_t_rel · tanh(|Δq_t_rel|/(v_eps·dt))`
///   clamped in magnitude to `μ · |normal_push|`. This is the
///   Dahl-style smoothed Coulomb cone (continuous, no
///   discontinuity at zero relative velocity, so the per-step
///   solve doesn't ring).
/// - Distribution to the four participating vertices follows the
///   same `(α, β_i)` partition as the normal correction, so
///   tri-side reaction = -vertex-side action (Newton's third law).
///
/// Restitution: when the vertex's pre-substep normal velocity was
/// negative (diving into the contact), the post-projection normal
/// velocity is set to `-restitution · v_n_pre` (instead of zeroed
/// out). 0 ⇒ purely inelastic; 1 ⇒ full bounce.
///
/// `prev_contacts` carries the previous substep's contact set; the
/// matched-by-(vertex, tri) entries warm-start their
/// `prev_friction_impulse`. Returns the final post-projection
/// `Vec<VtContact>` (with the substep's friction impulses written
/// back into `prev_friction_impulse`) so the caller can persist it
/// for the next call.
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

    // Stage 33.5: friction's tangential drift baseline is the
    // PRE-step position (passed in by the caller), not the start of
    // the projection. This way slip introduced by the inner VBD
    // substeps is what the Coulomb-cone clamps, not just the slip
    // introduced by the projection itself (which would be zero).
    // Restitution wants the PRE-step normal velocity for the same
    // reason — `qd_step_start` is the velocity entering the outer
    // step, so a single multi-substep step still gets the
    // physically-correct elastic kick.
    let qd_pre = qd_step_start.to_vec();
    let q_pre = q_step_start.to_vec();
    // Track per-vertex accumulated normal direction (for velocity
    // damping at the end). We sum the most-recent normal each sweep;
    // for a single contact this is just that contact's normal.
    let mut contact_normal_sum: Vec<Vec3> = vec![Vec3::ZERO; model.particle_count];
    // Per-vertex restitution + pre-step normal velocity along the
    // dominant contact direction. Used to set the post-projection
    // normal velocity to `-e · v_n_pre` instead of zero.
    let mut restitution_per_v: Vec<Real> = vec![0.0; model.particle_count];
    // Per-vertex max friction coefficient — used by the velocity-
    // form Coulomb damping at the end of the projection.
    let mut friction_per_v: Vec<Real> = vec![0.0; model.particle_count];

    // Static-velocity stick threshold (m/s). Below this magnitude
    // the tanh smoothing keeps the contact effectively glued; above
    // it the contact slides at the Coulomb-cone limit.
    let v_eps: Real = 1.0e-3;

    // Stage 33.5 anti-tunneling: build a (vertex, tri) → previous
    // contact normal LUT so the next gather can pick the correct
    // side of the contact when a vertex straddles the triangle. We
    // refresh this each sweep so contacts that resolve mid-sweep
    // still inherit the side from the very first sweep.
    let mut prev_normal_lut: std::collections::HashMap<(u32, u32), Vec3> =
        std::collections::HashMap::with_capacity(prev_contacts.len());
    for c in prev_contacts {
        let n = Vec3::new(c.normal[0], c.normal[1], c.normal[2]);
        prev_normal_lut.insert((c.vertex_global_id, c.tri_global_id), n);
    }

    let mut final_contacts: Vec<VtContact> = Vec::new();

    for sweep_idx in 0..n_sweeps {
        // Refit the BVH from the current positions. Cheap when the
        // tri count hasn't changed (just AABB recompute + bottom-up
        // refit); falls back to a full rebuild on a topology change.
        bvh.refit(model, state);
        let mut contacts =
            gather_vt_contacts_with_hint(model, state, bvh, margin, &prev_normal_lut);
        // Warm-start friction once per substep (only on the first
        // sweep — subsequent sweeps' contacts inherit the same
        // warm-starts when the contact persists).
        if sweep_idx == 0 {
            warm_start_friction(prev_contacts, &mut contacts);
        } else {
            // Carry forward from the previous sweep's results too,
            // so the Coulomb cone tightens monotonically.
            warm_start_friction(&final_contacts, &mut contacts);
        }
        if contacts.is_empty() {
            // No active contacts this sweep — keep the previous
            // sweep's `final_contacts` so we can still surface them
            // for the warm-start.
            break;
        }
        // Apply position projection. Read q via host_mut() once per
        // sweep so the Jacobi update sees a consistent snapshot.
        let q = state.particle_q.host_mut()?;
        let tris: &[[u32; 3]] = &model.soft_surface_tris;
        for c in contacts.iter_mut() {
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
            // Track per-vertex restitution = max over participating
            // contacts (a more-bouncy tri pair beats a less-bouncy
            // one if the vertex straddles).
            if c.restitution > restitution_per_v[v] {
                restitution_per_v[v] = c.restitution;
            }
            if c.friction > friction_per_v[v] {
                friction_per_v[v] = c.friction;
            }
            for &i in &[i0, i1, i2] {
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
                continue; // all four are pinned — nothing to do.
            }
            // Underrelaxation: take a fraction of the depth each
            // sweep so the multi-sweep Jacobi converges from above
            // (no overshoot) for typical mass ratios. A full depth
            // correction can cascade across neighbouring contacts on
            // the same particle and produce visible cloth tearing.
            let push = c.depth * 0.5;
            let alpha = w_v / total;
            let f0 = b0 * b0 * w0 / total;
            let f1 = b1 * b1 * w1 / total;
            let f2 = b2 * b2 * w2 / total;
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
                q[i0] -= n_vec * (f0 * push);
            }
            if w1 > 0.0 {
                q[i1] -= n_vec * (f1 * push);
            }
            if w2 > 0.0 {
                q[i2] -= n_vec * (f2 * push);
            }

            // ── Stage 33.5 friction (Dahl-smoothed Coulomb).
            //
            // Compute the tangential relative drift `Δq_t_rel`
            // since the start of this substep, projected onto the
            // contact tangent plane. This is the slip we want to
            // erase (clamped by the Coulomb cone `μ · |push_n|`).
            //
            // Note `q[v]` etc. were just updated above. We use the
            // post-normal-correction snapshot for the friction pass
            // so the tangent plane is consistent with the corrected
            // normal direction.
            if c.friction > 0.0 {
                let dq_v = q[v] - q_pre[v];
                let dq0 = q[i0] - q_pre[i0];
                let dq1 = q[i1] - q_pre[i1];
                let dq2 = q[i2] - q_pre[i2];
                let dq_tri = dq0 * b0 + dq1 * b1 + dq2 * b2;
                let mut dq_t = dq_v - dq_tri;
                // Project into the tangent plane.
                let dn = dq_t.dot(n_vec);
                dq_t -= n_vec * dn;
                let dq_t_len = dq_t.length();
                if dq_t_len > 1.0e-9 {
                    // Coulomb cone in position-correction units.
                    // The normal "force × dt" per substep is
                    // proportional to the gravitational impulse the
                    // contact must cancel (`m · g · dt` per
                    // sub-step). We approximate it as `μ · g · dt²`
                    // worth of tangential displacement per substep
                    // (the maximum slip-correction the cone allows
                    // when the contact is exactly equilibrated
                    // against gravity). For a contact engaged
                    // beyond gravity (push > μ·g·dt²), the depth-
                    // proportional `μ · |push|` term dominates and
                    // we use the larger of the two so heavily-
                    // pressed contacts (e.g. stacked tets) still
                    // respect the higher Coulomb cone.
                    let cone_from_depth = c.friction * push.abs();
                    // `μ · g · dt²`: minimum cone for any active
                    // contact, regardless of how thin the
                    // penetration is. `g = 9.81`.
                    let cone_from_gravity = c.friction * 9.81 * dt * dt;
                    let coulomb_max = cone_from_depth.max(cone_from_gravity);
                    // Dahl-style smoothing factor: full strength
                    // when slip far exceeds v_eps·dt; ramps down
                    // smoothly to 0 when slip is below it.
                    let smooth = (dq_t_len / (v_eps * dt.max(1.0e-6))).tanh();
                    let mag = (dq_t_len * smooth).min(coulomb_max);
                    let dir = dq_t / dq_t_len;
                    let push_t = -dir * mag; // oppose the slip direction
                    // Distribute the same way as normal: vertex
                    // side gets +alpha · push_t, tri verts each
                    // get -f_i · push_t.
                    if w_v > 0.0 {
                        q[v] += push_t * alpha;
                    }
                    if w0 > 0.0 {
                        q[i0] -= push_t * f0;
                    }
                    if w1 > 0.0 {
                        q[i1] -= push_t * f1;
                    }
                    if w2 > 0.0 {
                        q[i2] -= push_t * f2;
                    }
                    // Record the friction impulse (per-step
                    // tangential displacement scaled by inverse mass
                    // ≈ J_t/m). Stored for warm-start next call.
                    c.prev_friction_impulse =
                        [push_t.x, push_t.y, push_t.z];
                }
            }
        }
        // Refresh the prev_normal_lut from this sweep's contacts so
        // the next sweep sees consistent contact sides.
        for c in &contacts {
            let n = Vec3::new(c.normal[0], c.normal[1], c.normal[2]);
            prev_normal_lut.insert((c.vertex_global_id, c.tri_global_id), n);
        }
        final_contacts = contacts;
    }

    // Velocity recovery pass.
    //
    // Stage 33 used a hard "zero out the inward normal velocity"
    // damping; Stage 33.5 generalises this to:
    //   - Normal: set post-projection v_n = -e · v_n_pre when
    //     v_n_pre was diving in (otherwise leave alone).
    //   - Tangent: apply Coulomb friction velocity damping.
    //     `Δv_t = -μ · g · dt · (v_t/|v_t|)` clamped so the
    //     tangential velocity never reverses sign (a frictional
    //     impulse can stop a slip but never reverse it). This is
    //     the velocity-form complement of the position-form
    //     projection — together they give a usefully strong cone
    //     even at infinitesimal penalty depths (e.g. a cube at rest
    //     on cloth has depth ≈ m·g/k → near zero, so the position-
    //     form alone can't carry the gravity-derived friction).
    //     Per-vertex `μ` is the max over all contacts touching the
    //     vertex; aggregated outside the contact loop.
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
            // Pinned particles never integrate; skip the qd update.
            if inv_mass[i] <= 0.0 {
                continue;
            }
            let n = n_sum / n_len2.sqrt();
            // Pre-substep normal velocity along the dominant
            // contact direction.
            let vn_pre = qd_pre[i].dot(n);
            // Current velocity along the same direction.
            let mut v_now = qd[i];
            let vn_now = v_now.dot(n);
            // Restitution-aware normal-velocity recovery.
            if vn_pre < 0.0 {
                let e = restitution_per_v[i];
                let target_vn = -e * vn_pre;
                if vn_now < target_vn {
                    v_now = v_now + n * (target_vn - vn_now);
                }
            } else if vn_now < 0.0 {
                v_now = v_now - n * vn_now;
            }
            // Tangential velocity damping (Coulomb friction). Use
            // the strongest friction coefficient seen at this vertex
            // this substep. The clamp `|Δv_t| ≤ |v_t|` prevents
            // friction from reversing slip direction.
            let mu_v = friction_per_v[i];
            if mu_v > 0.0 {
                let v_t = v_now - n * v_now.dot(n);
                let v_t_len = v_t.length();
                if v_t_len > 1.0e-9 {
                    let dv_t_max = mu_v * 9.81 * dt;
                    let dv_t = dv_t_max.min(v_t_len);
                    v_now = v_now - (v_t / v_t_len) * dv_t;
                }
            }
            qd[i] = v_now;
        }
    }

    Ok(final_contacts)
}

/// Stage 33.5 — copy `prev_friction_impulse` from `prev` into the
/// matching contact in `current` (matched by `(vertex_global_id,
/// tri_global_id)`). Unmatched contacts in `current` keep their
/// zero-initialised impulse. Linear scan suffices because contact
/// counts are typically small (< few hundred); we drop in O(N·log N)
/// once contact counts justify the bookkeeping.
pub fn warm_start_friction(prev: &[VtContact], current: &mut [VtContact]) {
    if prev.is_empty() || current.is_empty() {
        return;
    }
    // Build a small lookup map. Using a HashMap is overkill for the
    // typical case of < 200 contacts, but it keeps the worst case
    // sub-quadratic and is easy to reason about. Key: (vertex,
    // tri); value: index into `prev`.
    let mut lut: std::collections::HashMap<(u32, u32), usize> =
        std::collections::HashMap::with_capacity(prev.len());
    for (i, c) in prev.iter().enumerate() {
        lut.insert((c.vertex_global_id, c.tri_global_id), i);
    }
    for c in current.iter_mut() {
        if let Some(&pi) = lut.get(&(c.vertex_global_id, c.tri_global_id)) {
            c.prev_friction_impulse = prev[pi].prev_friction_impulse;
        }
    }
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
