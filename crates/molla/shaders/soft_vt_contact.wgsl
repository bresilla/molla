// Soft-on-soft vertex-triangle (VT) contact pipeline — Stage 33.2.
//
// Five compute entry points run per VBD substep, after the per-vertex
// elastic block solve writes `particle_q`:
//
//   compute_tri_aabbs     — one thread per soft tri. Gathers the three
//                           tri verts from `particle_q`, builds the
//                           per-tri AABB used as input to the LBVH
//                           refit pass.
//   gather_vt_aabbs       — one thread per soft-surface vertex. Writes
//                           the per-vertex inflated AABB (point ±
//                           margin) into `vert_aabbs` (currently
//                           reserved for downstream batched-query
//                           kernels; the gather pass below walks the
//                           BVH directly with the live position so this
//                           is purely book-keeping for v1).
//   gather_vt             — one thread per soft-surface vertex. DFS-
//                           walks the BVH `nodes` buffer (built /
//                           refit by `kernels::lbvh.rs` over the per-
//                           tri AABBs above), enumerates candidate
//                           tris, computes barycentric closest-point
//                           + signed distance + outward normal, and
//                           atomically appends a `VtContactGpu` to
//                           `out_contacts` whenever `dist < margin`.
//                           Same per-vertex deduplication as the CPU
//                           reference: only the deepest contact per
//                           `(vertex, other_surface)` survives.
//   apply_vt_penalty      — one thread per emitted contact. Atomically
//                           accumulates the position-projection delta
//                           `±n · α · depth` into `particle_dq` (a
//                           scratch buffer of the same size as
//                           `particle_q`). Writes go through a
//                           `atomicCompareExchangeWeak` CAS loop
//                           because WGSL has no `atomic<f32>`.
//   commit_dq             — one thread per particle. Adds the
//                           accumulated `particle_dq` delta into
//                           `particle_q` and zeroes `particle_dq` for
//                           the next gather/apply round. This is the
//                           commit step that materialises the contact
//                           response into the position state.
//
// Layout choices:
//   - `tri_indices: vec4<u32>` — `(v0, v1, v2, surface_owner)`. The
//     surface owner is folded in at upload so the per-vertex same-
//     surface filter is one direct compare (no second indirection).
//   - The LBVH `nodes` buffer is bound read-only at group 4. Layout
//     mirrors `lbvh.wgsl::LbvhNode` exactly (8 × vec4<f32> = 64 bytes).
//   - Atomic VT contact sink: `out_contact_count[0]` is the high-water
//     mark; up to `arrayLength(&out_contacts)` entries fit before the
//     next slot is dropped (host pre-sizes for `vertex_count * 8` —
//     well above the typical density of 1-2 contacts/vertex).

struct Aabb {
    min: vec4<f32>,
    max: vec4<f32>,
};

struct LbvhNode {
    aabb_min: vec4<f32>,
    aabb_max: vec4<f32>,
    left:     u32,
    right:    u32,
    parent:   u32,
    is_leaf:  u32,
};

/// `kind` mirrors the host `SoftSurfaceKind` enum (0 = TetMesh,
/// 1 = Cloth). 32 bytes total — Stage 33.2.1 reused two of the
/// padding slots for `friction` + `restitution` so the kernel can
/// read per-surface contact material properties without a separate
/// upload.
struct SoftSurfaceGpu {
    kind:          u32,
    vertex_offset: u32,
    vertex_count:  u32,
    tri_offset:    u32,
    tri_count:     u32,
    friction:      f32,
    restitution:   f32,
    pad0:          u32,
};

/// One emitted VT contact. Fields mirror the CPU `VtContact` exactly
/// (modulo f32 vs f64 for `bary` / `normal` / `depth`); the host
/// downloads this for diagnostics and the `apply_vt_penalty` kernel
/// reads it back to apply the position projection.
struct VtContactGpu {
    vertex_global_id: u32,
    vertex_surface:   u32,
    tri_global_id:    u32,
    tri_surface:      u32,
    bary:             vec4<f32>, // .w = depth (packs into vec4)
    normal:           vec4<f32>, // .w unused
};

struct Uniforms {
    /// Total particle count — for vertex-loop bounds.
    particle_count:  u32,
    /// Total soft tri count — for tri-loop bounds.
    tri_count:       u32,
    /// Total soft-surface count — for the vertex→surface ownership
    /// lookup in the gather kernel.
    surface_count:   u32,
    /// Total emitted-contact count — for the apply kernel's bound.
    /// Host re-uploads after gather completes (or apply reads
    /// `out_contact_count[0]` directly via atomicLoad).
    contact_count:   u32,
    /// Contact buffer margin (m). Verts within this distance of an
    /// other-surface tri trigger a contact.
    margin:          f32,
    /// Penalty stiffness scalar `k_contact` (N/m). Reserved for an
    /// energy-based variant; v1 uses the position-projection variant
    /// and ignores this field.
    k_contact:       f32,
    /// Capacity of the contact output buffer; the gather kernel skips
    /// emission once this many contacts have been written.
    out_capacity:    u32,
    /// LBVH primitive count — needed by the BVH traversal to detect
    /// leaves (leaves live at `[0, primitive_count)`).
    primitive_count: u32,
};

// ── Group 0: per-particle state + delta scratch ──────────────────────
// `particle_q` is bound read_write because `commit_dq` updates it in
// place. The `apply_vt_penalty` pass writes to `particle_dq` instead
// (atomic float-add via CAS); `commit_dq` reads `particle_dq`, adds
// it into `particle_q`, and zeroes `particle_dq` for the next round.
// `particle_qd` is read_write because `commit_dq` damps the inward
// velocity component along the dominant contact direction (the dq
// direction proxies for the contact normal sum — points the same
// way for the contact-vertex side, opposite for the tri-vertex side,
// either way the inward component should be zeroed).
@group(0) @binding(0) var<storage, read_write> particle_q:        array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>       particle_inv_mass: array<f32>;
@group(0) @binding(2) var<storage, read_write> particle_dq:       array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> particle_qd:       array<vec4<f32>>;

// ── Group 1: soft-surface registry + tri indices ─────────────────────
@group(1) @binding(0) var<storage, read>       soft_surface_tris: array<vec4<u32>>;
@group(1) @binding(1) var<storage, read>       soft_surfaces:     array<SoftSurfaceGpu>;

// ── Group 2: per-tri AABB output (refit input) + per-vert AABB ───────
@group(2) @binding(0) var<storage, read_write> tri_aabbs:  array<Aabb>;
@group(2) @binding(1) var<storage, read_write> vert_aabbs: array<Aabb>;

// ── Group 3: LBVH read-only — nodes + sorted_idx + root ──────────────
@group(3) @binding(0) var<storage, read>       bvh_nodes:      array<LbvhNode>;
@group(3) @binding(1) var<storage, read>       bvh_sorted_idx: array<u32>;
/// `pair_count[1]` holds the LBVH root index (written by the build /
/// refit kernels in `lbvh.wgsl`). We read it once per gather thread.
@group(3) @binding(2) var<storage, read>       bvh_pair_count: array<u32>;

// ── Group 4: contact sink + uniforms ─────────────────────────────────
@group(4) @binding(0) var<storage, read_write> out_contacts:      array<VtContactGpu>;
@group(4) @binding(1) var<storage, read_write> out_contact_count: array<atomic<u32>>;
@group(4) @binding(2) var<uniform>             u: Uniforms;

// ── Helpers ─────────────────────────────────────────────────────────

fn aabb_contains_point(amin: vec3<f32>, amax: vec3<f32>, p: vec3<f32>, m: f32) -> bool {
    return p.x >= amin.x - m && p.x <= amax.x + m
        && p.y >= amin.y - m && p.y <= amax.y + m
        && p.z >= amin.z - m && p.z <= amax.z + m;
}

/// Closest point on triangle `(a, b, c)` to `p` + barycentric weights.
/// Mirrors `closest_point_on_triangle_bary` in
/// `solvers/vbd_vt_contact.rs`. Output `.xyz` is the closest point;
/// barycentric weights are `(u, v, w)` matching `(a, b, c)`.
fn closest_point_on_triangle(
    p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>
) -> vec3<f32> {
    // Voronoi region tests — Eberly's "Real-Time Collision Detection"
    // §5.1.5. Returns the closest point only; barycentrics are
    // recomputed from the closest point in the gather pass.
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) { return a; } // vertex-A region
    let bp = p - b;
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) { return b; } // vertex-B region
    let vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        let v = d1 / (d1 - d3);
        return a + v * ab; // edge-AB region
    }
    let cp = p - c;
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) { return c; } // vertex-C region
    let vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        let w = d2 / (d2 - d6);
        return a + w * ac; // edge-AC region
    }
    let va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b + w * (c - b); // edge-BC region
    }
    // Face region — barycentric inside the triangle.
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    return a + ab * v + ac * w;
}

/// Compute `(u, v, w)` barycentric weights of `q` w.r.t. triangle
/// `(a, b, c)` via the standard projected-area formulation. Mirrors
/// the CPU helper.
fn barycentric(q: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = q - a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if (abs(denom) < 1.0e-20) {
        return vec3<f32>(1.0 / 3.0);
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let uu = 1.0 - v - w;
    var bary = vec3<f32>(max(uu, 0.0), max(v, 0.0), max(w, 0.0));
    let s = bary.x + bary.y + bary.z;
    if (s > 1.0e-12) {
        bary = bary / s;
    } else {
        bary = vec3<f32>(1.0 / 3.0);
    }
    return bary;
}

/// Surface-id of vertex `v_global`. Linear scan — surface_count is
/// tiny (1-4 in v1 demos) so this is far cheaper than a CSR lookup.
fn vertex_surface(v_global: u32) -> u32 {
    for (var s: u32 = 0u; s < u.surface_count; s = s + 1u) {
        let surf = soft_surfaces[s];
        let lo = surf.vertex_offset;
        let hi = lo + surf.vertex_count;
        if (v_global >= lo && v_global < hi) {
            return s;
        }
    }
    return 0xffffffffu;
}

// ── Entry: compute per-tri AABBs from particle_q ─────────────────────
//
// One thread per soft tri. Reads the three vert positions from
// `particle_q`, writes the per-tri AABB into `tri_aabbs`. Output
// feeds the LBVH refit pass directly.
@compute @workgroup_size(64)
fn compute_tri_aabbs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = gid.x;
    if (t >= u.tri_count) { return; }
    let tri = soft_surface_tris[t];
    let p0 = particle_q[tri.x].xyz;
    let p1 = particle_q[tri.y].xyz;
    let p2 = particle_q[tri.z].xyz;
    let mn = min(p0, min(p1, p2));
    let mx = max(p0, max(p1, p2));
    tri_aabbs[t] = Aabb(vec4<f32>(mn, 0.0), vec4<f32>(mx, 0.0));
}

// ── Entry: compute per-vert AABBs (point ± margin) ───────────────────
//
// One thread per particle. Writes the per-vertex AABB into
// `vert_aabbs`. v1's gather kernel walks the BVH directly with the
// live position; this entry point is wired in for downstream batched-
// query kernels and parity with the spec's three-entry-point design.
@compute @workgroup_size(64)
fn gather_vt_aabbs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= u.particle_count) { return; }
    let p = particle_q[v].xyz;
    let m = u.margin;
    vert_aabbs[v] = Aabb(
        vec4<f32>(p - vec3<f32>(m), 0.0),
        vec4<f32>(p + vec3<f32>(m), 0.0),
    );
}

// ── Entry: gather VT contacts ────────────────────────────────────────
//
// One thread per soft-surface vertex. DFS-walks the LBVH, enumerates
// every candidate tri whose AABB contains the vert (with margin),
// computes the barycentric closest-point + signed distance + outward
// normal, and emits one `VtContactGpu` per `(vertex, other_surface)`
// pair (deepest depth wins). Same dedup strategy as the CPU
// `gather_vt_contacts`.
@compute @workgroup_size(64)
fn gather_vt(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= u.particle_count) { return; }
    if (u.primitive_count < 1u) { return; }
    let v_surface = vertex_surface(v);
    if (v_surface == 0xffffffffu) {
        // Vertex isn't part of any soft surface — nothing to do.
        return;
    }
    let p = particle_q[v].xyz;
    let n = u.primitive_count;
    let root = bvh_pair_count[1];

    // Per-other-surface "best contact" tracker (deepest depth wins).
    // surface_count is small (≤ 8 in v1 demos); cap to 8 for the
    // fixed-size local array. If a future scene needs more, raise
    // this bound — atomic emission is the only correctness cost.
    var best_depth: array<f32, 8>;
    var best_tri:   array<u32, 8>;
    var best_bary:  array<vec3<f32>, 8>;
    var best_n:     array<vec3<f32>, 8>;
    let max_surf: u32 = min(u.surface_count, 8u);
    for (var s: u32 = 0u; s < max_surf; s = s + 1u) {
        best_depth[s] = -1.0;
        best_tri[s] = 0xffffffffu;
        best_bary[s] = vec3<f32>(0.0);
        best_n[s] = vec3<f32>(0.0, 1.0, 0.0);
    }

    // Iterative DFS over the LBVH. Stack depth 64 mirrors lbvh.wgsl's
    // find_pairs — sufficient for the v1 tri counts (~10k tris ⇒
    // ~14-deep tree).
    var stack: array<u32, 64>;
    var sp: i32 = 0;
    stack[0] = root;
    sp = 1;
    let m = u.margin;

    loop {
        if (sp <= 0) { break; }
        sp = sp - 1;
        let node_idx = stack[sp];
        // Leaf? Layout puts leaves at [0, n).
        if (node_idx < n) {
            let tri_id = bvh_sorted_idx[node_idx];
            let tri = soft_surface_tris[tri_id];
            let tri_owner = tri.w;
            // Same-surface filter (Stage 33.3 will replace with
            // per-vertex adjacency exclusion).
            if (tri_owner == v_surface) { continue; }
            // Topological filter — vertex IS one of the tri corners
            // (rare but possible when surfaces share a particle).
            if (tri.x == v || tri.y == v || tri.z == v) { continue; }
            let p0 = particle_q[tri.x].xyz;
            let p1 = particle_q[tri.y].xyz;
            let p2 = particle_q[tri.z].xyz;
            let q = closest_point_on_triangle(p, p0, p1, p2);
            let diff = p - q;
            let d2 = dot(diff, diff);
            if (d2 >= m * m) { continue; }
            let dist = sqrt(d2);
            let depth = m - dist;
            // Normal: from tri toward vert. Fall back to the geometric
            // normal when the vert sits exactly on the tri.
            var nrm: vec3<f32>;
            if (dist > 1.0e-9) {
                nrm = diff / dist;
            } else {
                let g = cross(p1 - p0, p2 - p0);
                let gl = length(g);
                if (gl > 1.0e-9) {
                    nrm = g / gl;
                } else {
                    nrm = vec3<f32>(0.0, 1.0, 0.0);
                }
            }
            let bary = barycentric(q, p0, p1, p2);
            // Update per-other-surface best-contact slot.
            if (tri_owner < max_surf) {
                if (depth > best_depth[tri_owner]) {
                    best_depth[tri_owner] = depth;
                    best_tri[tri_owner] = tri_id;
                    best_bary[tri_owner] = bary;
                    best_n[tri_owner] = nrm;
                }
            }
            continue;
        }
        // Internal: AABB-vs-point (with margin) test, recurse.
        let node = bvh_nodes[node_idx];
        if (!aabb_contains_point(node.aabb_min.xyz, node.aabb_max.xyz, p, m)) {
            continue;
        }
        if (sp + 2 <= 64) {
            stack[sp] = node.left;
            sp = sp + 1;
            stack[sp] = node.right;
            sp = sp + 1;
        }
    }

    // Emit one contact per surface that found a hit.
    for (var s: u32 = 0u; s < max_surf; s = s + 1u) {
        if (best_depth[s] <= 0.0) { continue; }
        let slot = atomicAdd(&out_contact_count[0], 1u);
        if (slot >= u.out_capacity) { continue; }
        var c: VtContactGpu;
        c.vertex_global_id = v;
        c.vertex_surface = v_surface;
        c.tri_global_id = best_tri[s];
        c.tri_surface = s;
        c.bary = vec4<f32>(best_bary[s], best_depth[s]);
        c.normal = vec4<f32>(best_n[s], 0.0);
        out_contacts[slot] = c;
    }
}

// ── Entry: apply VT penalty (atomic delta accumulation) ──────────────
//
// One thread per emitted contact. Atomically accumulates the
// inverse-mass-weighted underrelaxed projection into `particle_dq`
// (NOT `particle_q` — that gets updated by `commit_dq`). v1 uses
// `atomicAdd` semantics emulated via `atomicCompareExchangeWeak` on
// the f32-bit-cast u32 view of `particle_dq`. Atomics serialise the
// writes so colouring is unnecessary.
@compute @workgroup_size(64)
fn apply_vt_penalty(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.contact_count) { return; }
    let c = out_contacts[i];
    let v = c.vertex_global_id;
    let tri = soft_surface_tris[c.tri_global_id];
    let i0 = tri.x;
    let i1 = tri.y;
    let i2 = tri.z;
    let n = c.normal.xyz;
    let depth = c.bary.w;
    let b0 = c.bary.x;
    let b1 = c.bary.y;
    let b2 = c.bary.z;

    let w_v = particle_inv_mass[v];
    let w0 = particle_inv_mass[i0];
    let w1 = particle_inv_mass[i1];
    let w2 = particle_inv_mass[i2];
    let w_tri = b0 * b0 * w0 + b1 * b1 * w1 + b2 * b2 * w2;
    let total = w_v + w_tri;
    if (total <= 1.0e-20) { return; }
    // Underrelaxation: 0.5 ⇒ 2-sweep convergence on typical mass
    // ratios (matches CPU `project_vt_contacts`).
    let push = depth * 0.5;
    let alpha = w_v / total;

    // Vertex side: +α·push along n.
    if (w_v > 0.0) {
        atomic_add_f32_xyz(v, n * (alpha * push));
    }
    if (w0 > 0.0) {
        let f0 = b0 * b0 * w0 / total;
        atomic_add_f32_xyz(i0, -n * (f0 * push));
    }
    if (w1 > 0.0) {
        let f1 = b1 * b1 * w1 / total;
        atomic_add_f32_xyz(i1, -n * (f1 * push));
    }
    if (w2 > 0.0) {
        let f2 = b2 * b2 * w2 / total;
        atomic_add_f32_xyz(i2, -n * (f2 * push));
    }
}

// ── Entry: commit accumulated dq into q + zero scratch ───────────────
//
// One thread per particle. Adds `particle_dq` into `particle_q` and
// zeroes `particle_dq` so the next gather/apply round starts from a
// clean slate. Non-atomic — each thread owns one particle, so there
// are no races at this stage (reads of `particle_dq` happen-after
// the apply pass via the queue.submit boundary between dispatches).
@compute @workgroup_size(64)
fn commit_dq(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if (v >= u.particle_count) { return; }
    // particle_dq is bound as atomic<u32> for the apply pass; here we
    // read it via atomicLoad to drain the lane and then atomicStore 0.
    let base = v * 4u;
    let dx_bits = atomicLoad(&particle_dq[base + 0u]);
    let dy_bits = atomicLoad(&particle_dq[base + 1u]);
    let dz_bits = atomicLoad(&particle_dq[base + 2u]);
    let dx = bitcast<f32>(dx_bits);
    let dy = bitcast<f32>(dy_bits);
    let dz = bitcast<f32>(dz_bits);
    if (dx == 0.0 && dy == 0.0 && dz == 0.0) { return; }
    let dq = vec3<f32>(dx, dy, dz);
    let q_old = particle_q[v].xyz;
    particle_q[v] = vec4<f32>(q_old + dq, 0.0);
    // Velocity damping along the dq direction. CPU
    // `project_vt_contacts` zeroes the inward velocity component
    // along the per-particle contact-normal sum (vertex side: +n;
    // tri side: -n weighted by bary). The per-substep accumulated
    // dq is a mass-weighted proxy for that direction (same sign on
    // each side — vertex moves +n·α·push, tri verts move
    // -n·β_i·push). Project out the inward component
    // (negative-dq direction) of velocity.
    let dq_len2 = dot(dq, dq);
    if (dq_len2 > 1.0e-20 && particle_inv_mass[v] > 0.0) {
        let n_dir = dq / sqrt(dq_len2);
        let qd_old = particle_qd[v].xyz;
        let vn = dot(qd_old, n_dir);
        if (vn < 0.0) {
            // Inward velocity (against push direction) — zero it.
            particle_qd[v] = vec4<f32>(qd_old - n_dir * vn, 0.0);
        }
    }
    atomicStore(&particle_dq[base + 0u], 0u);
    atomicStore(&particle_dq[base + 1u], 0u);
    atomicStore(&particle_dq[base + 2u], 0u);
}

// WGSL has no `atomic<f32>`; emulate via CAS loop on the f32-bit-cast
// `atomic<u32>` view of each lane (same pattern as `mpm.wgsl`).
fn atomic_add_f32_xyz(particle: u32, delta: vec3<f32>) {
    // particle_dq layout matches particle_q: vec4<f32> = 16 bytes per
    // particle, bound as `array<atomic<u32>>` so 4 u32s per particle.
    // Lane 0 = .x, 1 = .y, 2 = .z, 3 = pad.
    let base = particle * 4u;
    atomic_add_f32(base + 0u, delta.x);
    atomic_add_f32(base + 1u, delta.y);
    atomic_add_f32(base + 2u, delta.z);
}

fn atomic_add_f32(idx: u32, value: f32) {
    if (value == 0.0) { return; }
    var old_bits = atomicLoad(&particle_dq[idx]);
    loop {
        let old_f = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_f + value);
        let r = atomicCompareExchangeWeak(&particle_dq[idx], old_bits, new_bits);
        if (r.exchanged) {
            return;
        }
        old_bits = r.old_value;
    }
}
