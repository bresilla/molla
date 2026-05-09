// GPU EPA — Expanding Polytope Algorithm in WGSL.
//
// Per-pair: read the 4-vertex GJK simplex from gjk_simplex_in,
// expand it iteratively into a polytope, and write the contact
// (normal, depth, point_a, point_b, status) into epa_out.
//
// Bounded-polytope variant: per-thread fixed arrays cap vertex /
// face counts. Pairs that overflow or fail to converge emit a
// `status = STATUS_FALLBACK` flag so the host can re-run them on
// CPU EPA.
//
// Reference: crates/molla/src/geometry/epa.rs (CPU port that this
// kernel mirrors).

struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,
};

const KIND_SPHERE:    u32 = 0u;
const KIND_PLANE:     u32 = 1u;
const KIND_BOX:       u32 = 2u;
const KIND_CAPSULE:   u32 = 3u;
const KIND_CYLINDER:  u32 = 4u;
const KIND_CONE:      u32 = 5u;
const KIND_ELLIPSOID: u32 = 6u;
const KIND_SKIP:      u32 = 0xffu;
const STATIC_BODY:   i32 = -1;

const MAX_VERTICES: u32 = 32u;
const MAX_FACES:    u32 = 64u;
const MAX_ITERS:    u32 = 32u;

const STATUS_OK:       u32 = 1u;  // contact valid
const STATUS_FALLBACK: u32 = 2u;  // host EPA on CPU
const STATUS_NONE:     u32 = 0u;  // pair was disjoint or skipped

// Stride matches GjkKernel::GJK_SIMPLEX_STRIDE — 12 vec4 per pair
// (4 vertices × 3 vec3 (w, a, b)).
const SIMPLEX_STRIDE: u32 = 12u;

struct Uniforms {
    pair_count:  u32,
    shape_count: u32,
    _pad0:       u32,
    _pad1:       u32,
};

@group(0) @binding(0) var<storage, read>  shape_geom_kind:  array<u32>;
@group(0) @binding(1) var<storage, read>  shape_geom_data:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>  shape_body:       array<i32>;
@group(0) @binding(3) var<storage, read>  shape_transform:  array<Transform>;
@group(0) @binding(4) var<storage, read>  body_q:           array<Transform>;

@group(1) @binding(0) var<storage, read>  pair_a:           array<u32>;
@group(1) @binding(1) var<storage, read>  pair_b:           array<u32>;
@group(1) @binding(2) var<storage, read>  overlap_in:       array<u32>;
@group(1) @binding(3) var<storage, read>  gjk_simplex_in:   array<vec4<f32>>;

// Output stride: 16 f32 per pair. Layout per pair `i`
// (base = i*16):
//   [0..3]   normal.xyz, depth
//   [4..7]   point_a.xyz, _
//   [8..11]  point_b.xyz, _
//   [12..15] status_bits_as_f32, _, _, _
@group(2) @binding(0) var<storage, read_write> epa_out: array<f32>;
const EPA_OUT_STRIDE: u32 = 16u;

@group(3) @binding(0) var<uniform> u: Uniforms;

// ── Quat / transform helpers (mirror gjk.wgsl). ─────────────────
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}
fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}
fn xf_compose(parent: Transform, local: Transform) -> Transform {
    var out: Transform;
    out.position = vec4<f32>(parent.position.xyz + quat_rotate(parent.rotation, local.position.xyz), 0.0);
    out.rotation = quat_mul(parent.rotation, local.rotation);
    return out;
}
fn world_xf_of(shape: u32) -> Transform {
    let body = shape_body[shape];
    let local = shape_transform[shape];
    if (body == STATIC_BODY) { return local; }
    return xf_compose(body_q[u32(body)], local);
}

// ── Inline support functions in shape-local frame. ──────────────
fn local_support(kind: u32, data: vec3<f32>, dir_local: vec3<f32>) -> vec3<f32> {
    if (kind == KIND_SPHERE) {
        let r = data.x;
        let len_sq = dot(dir_local, dir_local);
        if (len_sq < 1e-24) { return vec3<f32>(0.0); }
        return dir_local * (r / sqrt(len_sq));
    }
    if (kind == KIND_BOX) {
        return vec3<f32>(
            select(-data.x, data.x, dir_local.x >= 0.0),
            select(-data.y, data.y, dir_local.y >= 0.0),
            select(-data.z, data.z, dir_local.z >= 0.0),
        );
    }
    if (kind == KIND_CAPSULE) {
        let half_h = data.x;
        let r = data.y;
        let cap_y = select(-half_h, half_h, dir_local.y >= 0.0);
        let len_sq = dot(dir_local, dir_local);
        var sphere = vec3<f32>(0.0);
        if (len_sq > 1e-24) {
            sphere = dir_local * (r / sqrt(len_sq));
        }
        return vec3<f32>(sphere.x, cap_y + sphere.y, sphere.z);
    }
    if (kind == KIND_CYLINDER) {
        let half_h = data.x;
        let r = data.y;
        let cap_y = select(-half_h, half_h, dir_local.y >= 0.0);
        let xz = vec2<f32>(dir_local.x, dir_local.z);
        let xz_len = length(xz);
        var edge = vec2<f32>(r, 0.0);
        if (xz_len > 1e-12) {
            edge = xz * (r / xz_len);
        }
        return vec3<f32>(edge.x, cap_y, edge.y);
    }
    if (kind == KIND_CONE) {
        let half_h = data.x;
        let r = data.y;
        let apex_dot = half_h * dir_local.y;
        let xz = vec2<f32>(dir_local.x, dir_local.z);
        let xz_len = length(xz);
        var edge = vec2<f32>(r, 0.0);
        if (xz_len > 1e-12) {
            edge = xz * (r / xz_len);
        }
        let base_dot = edge.x * dir_local.x + edge.y * dir_local.z - half_h * dir_local.y;
        if (apex_dot >= base_dot) {
            return vec3<f32>(0.0, half_h, 0.0);
        }
        return vec3<f32>(edge.x, -half_h, edge.y);
    }
    if (kind == KIND_ELLIPSOID) {
        // Mirror of CPU `support_ellipsoid` (per-axis scale → norm →
        // rescale). Stage 30.
        let scaled = vec3<f32>(
            data.x * dir_local.x,
            data.y * dir_local.y,
            data.z * dir_local.z,
        );
        let len_sq = dot(scaled, scaled);
        if (len_sq < 1e-24) {
            return vec3<f32>(data.x, 0.0, 0.0);
        }
        let inv_len = inverseSqrt(len_sq);
        return vec3<f32>(
            data.x * scaled.x * inv_len,
            data.y * scaled.y * inv_len,
            data.z * scaled.z * inv_len,
        );
    }
    return vec3<f32>(0.0);
}
fn world_support(shape: u32, dir_world: vec3<f32>) -> vec3<f32> {
    let xf = world_xf_of(shape);
    let inv_rot = quat_conj(xf.rotation);
    let dir_local = quat_rotate(inv_rot, dir_world);
    let kind = shape_geom_kind[shape];
    let data = shape_geom_data[shape].xyz;
    let local_p = local_support(kind, data, dir_local);
    return xf.position.xyz + quat_rotate(xf.rotation, local_p);
}

struct SupportPair { w: vec3<f32>, a: vec3<f32>, b: vec3<f32>, };
fn minkowski_support_pair(sa: u32, sb: u32, dir: vec3<f32>) -> SupportPair {
    var out: SupportPair;
    out.a = world_support(sa, dir);
    out.b = world_support(sb, -dir);
    out.w = out.a - out.b;
    return out;
}

// ── Per-thread polytope storage. WGSL doesn't allow per-thread
// dynamic-size arrays, so we cap and store in `var` private. ────
var<private> v_w: array<vec3<f32>, MAX_VERTICES>;
var<private> v_a: array<vec3<f32>, MAX_VERTICES>;
var<private> v_b: array<vec3<f32>, MAX_VERTICES>;
var<private> v_count: u32;

// Faces: 3 vertex indices + outward unit normal + signed distance.
var<private> f_i0: array<u32, MAX_FACES>;
var<private> f_i1: array<u32, MAX_FACES>;
var<private> f_i2: array<u32, MAX_FACES>;
var<private> f_normal: array<vec3<f32>, MAX_FACES>;
var<private> f_dist:   array<f32, MAX_FACES>;
var<private> f_alive:  array<u32, MAX_FACES>;
var<private> f_count: u32;

fn add_vertex(w: vec3<f32>, a: vec3<f32>, b: vec3<f32>) -> i32 {
    if (v_count >= MAX_VERTICES) { return -1; }
    let idx = v_count;
    v_w[idx] = w;
    v_a[idx] = a;
    v_b[idx] = b;
    v_count = v_count + 1u;
    return i32(idx);
}

fn add_face(i0: u32, i1: u32, i2: u32) -> i32 {
    if (f_count >= MAX_FACES) { return -1; }
    let v0 = v_w[i0];
    let v1 = v_w[i1];
    let v2 = v_w[i2];
    let n_raw = cross(v1 - v0, v2 - v0);
    let len = length(n_raw);
    if (len < 1e-12) { return -2; }
    let n = n_raw / len;
    let d = dot(v0, n);
    let idx = f_count;
    f_i0[idx] = i0;
    f_i1[idx] = i1;
    f_i2[idx] = i2;
    f_normal[idx] = n;
    f_dist[idx] = max(d, 0.0);
    f_alive[idx] = 1u;
    f_count = f_count + 1u;
    return i32(idx);
}

// Build the initial 4-face tetrahedron from 4 vertices, orienting
// each face's normal away from the opposite vertex. Returns 0 on
// success, !=0 on degeneracy.
fn build_initial_tetra() -> i32 {
    if (v_count < 4u) { return 1; }
    f_count = 0u;
    // Combos: (face vertices, opposite vertex).
    var i0a = array<u32, 4>(0u, 0u, 0u, 1u);
    var i1a = array<u32, 4>(1u, 2u, 3u, 3u);
    var i2a = array<u32, 4>(2u, 3u, 1u, 2u);
    var oppa = array<u32, 4>(3u, 1u, 2u, 0u);
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        var i0 = i0a[k]; var i1 = i1a[k]; var i2 = i2a[k]; let opp = oppa[k];
        let v0 = v_w[i0]; let v1 = v_w[i1]; let v2 = v_w[i2]; let vop = v_w[opp];
        let n_raw = cross(v1 - v0, v2 - v0);
        let len = length(n_raw);
        if (len < 1e-12) { return 2; }
        var n = n_raw / len;
        if (dot(n, vop - v0) > 0.0) {
            // Flip winding so the normal faces outward.
            let tmp = i1; i1 = i2; i2 = tmp;
            n = -n;
        }
        let d = max(dot(v_w[i0], n), 0.0);
        if (d < 1e-6) { return 3; } // origin on face plane → degenerate
        let idx = f_count;
        f_i0[idx] = i0; f_i1[idx] = i1; f_i2[idx] = i2;
        f_normal[idx] = n; f_dist[idx] = d; f_alive[idx] = 1u;
        f_count = f_count + 1u;
    }
    return 0;
}

// Find the live face nearest the origin. Returns -1 if none.
fn nearest_face() -> i32 {
    var best: i32 = -1;
    var best_d: f32 = 1e30;
    for (var i: u32 = 0u; i < f_count; i = i + 1u) {
        if (f_alive[i] == 0u) { continue; }
        if (f_dist[i] < best_d) {
            best_d = f_dist[i];
            best = i32(i);
        }
    }
    return best;
}

// Compute barycentric (u, v, w) of `p` projected onto triangle (a,b,c).
fn barycentric(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, p: vec3<f32>) -> vec3<f32> {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    if (abs(denom) < 1e-24) { return vec3<f32>(1.0, 0.0, 0.0); }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let uu = 1.0 - v - w;
    return vec3<f32>(uu, v, w);
}

fn write_contact(i: u32, normal: vec3<f32>, depth: f32, pa: vec3<f32>, pb: vec3<f32>, status: u32) {
    let base = i * EPA_OUT_STRIDE;
    epa_out[base + 0u]  = normal.x;
    epa_out[base + 1u]  = normal.y;
    epa_out[base + 2u]  = normal.z;
    epa_out[base + 3u]  = depth;
    epa_out[base + 4u]  = pa.x;
    epa_out[base + 5u]  = pa.y;
    epa_out[base + 6u]  = pa.z;
    epa_out[base + 7u]  = 0.0;
    epa_out[base + 8u]  = pb.x;
    epa_out[base + 9u]  = pb.y;
    epa_out[base + 10u] = pb.z;
    epa_out[base + 11u] = 0.0;
    epa_out[base + 12u] = bitcast<f32>(status);
    epa_out[base + 13u] = 0.0;
    epa_out[base + 14u] = 0.0;
    epa_out[base + 15u] = 0.0;
}

// Edge silhouette tracking. We push directed edges (a→b) from each
// visible face; for a pair (b, a) already in the list, we cancel
// (the edge was interior, not silhouette).
const MAX_EDGES: u32 = 192u; // 3 * MAX_FACES is the worst-case
var<private> e_a: array<u32, MAX_EDGES>;
var<private> e_b: array<u32, MAX_EDGES>;
var<private> e_count: u32;

fn push_edge(a: u32, b: u32) -> u32 {
    // Cancel a matching reverse edge if present.
    for (var i: u32 = 0u; i < e_count; i = i + 1u) {
        if (e_a[i] == b && e_b[i] == a) {
            // Swap-remove with last edge.
            let last = e_count - 1u;
            e_a[i] = e_a[last];
            e_b[i] = e_b[last];
            e_count = last;
            return 1u;
        }
    }
    if (e_count >= MAX_EDGES) { return 0u; }
    e_a[e_count] = a;
    e_b[e_count] = b;
    e_count = e_count + 1u;
    return 1u;
}

@compute @workgroup_size(64)
fn epa_solve(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.pair_count) { return; }
    // Default: write a zero status so callers see "no contact" for
    // pairs we skipped.
    write_contact(i, vec3<f32>(0.0, 1.0, 0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_NONE);

    // Only run on pairs the GJK pass marked as overlapping.
    if (overlap_in[i] != 1u) { return; }

    let sa = pair_a[i];
    let sb = pair_b[i];
    if (sa >= u.shape_count || sb >= u.shape_count) { return; }
    let ka = shape_geom_kind[sa];
    let kb = shape_geom_kind[sb];
    if (ka == KIND_PLANE || kb == KIND_PLANE
        || ka == KIND_SKIP || kb == KIND_SKIP) {
        write_contact(i, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_FALLBACK);
        return;
    }

    // ── Seed polytope from the GJK simplex. ────────────────────
    v_count = 0u;
    f_count = 0u;
    let base = i * SIMPLEX_STRIDE;
    // Each vertex = 3 vec4 (w, a, b). The third's .w lane carries
    // the simplex vertex count.
    let count_f = gjk_simplex_in[base + 2u].w;
    let n_seed = u32(count_f);
    if (n_seed == 0u) {
        write_contact(i, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_FALLBACK);
        return;
    }
    for (var k: u32 = 0u; k < n_seed && k < 4u; k = k + 1u) {
        let off = base + k * 3u;
        let w = gjk_simplex_in[off + 0u].xyz;
        let a = gjk_simplex_in[off + 1u].xyz;
        let b = gjk_simplex_in[off + 2u].xyz;
        _ = add_vertex(w, a, b);
    }

    // Pad to a 4-vertex tetra if needed by querying cardinal-axis
    // supports — mirrors the CPU fallback. The GJK kernel almost
    // always returns 4-vertex simplices but defends against partials.
    if (v_count < 4u) {
        // Drop the partial seed (it may be linearly degenerate) and
        // build a fresh cardinal-axis tetra. Cheap, robust.
        v_count = 0u;
        var dirs = array<vec3<f32>, 4>(
            vec3<f32>( 1.0,  1.0,  1.0),
            vec3<f32>(-1.0, -1.0,  1.0),
            vec3<f32>(-1.0,  1.0, -1.0),
            vec3<f32>( 1.0, -1.0, -1.0),
        );
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let p = minkowski_support_pair(sa, sb, dirs[k]);
            _ = add_vertex(p.w, p.a, p.b);
        }
    }

    // Build initial faces; on degeneracy fall back to CPU.
    let init = build_initial_tetra();
    if (init != 0) {
        // Try the cardinal-axis tetra once.
        v_count = 0u;
        var dirs = array<vec3<f32>, 4>(
            vec3<f32>( 1.0,  1.0,  1.0),
            vec3<f32>(-1.0, -1.0,  1.0),
            vec3<f32>(-1.0,  1.0, -1.0),
            vec3<f32>( 1.0, -1.0, -1.0),
        );
        for (var k: u32 = 0u; k < 4u; k = k + 1u) {
            let p = minkowski_support_pair(sa, sb, dirs[k]);
            _ = add_vertex(p.w, p.a, p.b);
        }
        let init2 = build_initial_tetra();
        if (init2 != 0) {
            write_contact(i, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_FALLBACK);
            return;
        }
    }

    let TOL: f32 = 1e-4;

    var converged_face: i32 = -1;
    for (var iter: u32 = 0u; iter < MAX_ITERS; iter = iter + 1u) {
        let face_idx = nearest_face();
        if (face_idx < 0) {
            write_contact(i, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_FALLBACK);
            return;
        }
        let fi = u32(face_idx);
        let n = f_normal[fi];
        let d = f_dist[fi];

        let support = minkowski_support_pair(sa, sb, n);
        let dist_new = dot(support.w, n);
        if (dist_new - d < TOL) {
            converged_face = face_idx;
            break;
        }

        // Add the new vertex.
        let v_new = add_vertex(support.w, support.a, support.b);
        if (v_new < 0) {
            write_contact(i, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_FALLBACK);
            return;
        }
        let new_idx = u32(v_new);

        // Mark all faces visible from the new vertex as dead and
        // collect their boundary edges as silhouette.
        e_count = 0u;
        var any_visible = false;
        for (var k: u32 = 0u; k < f_count; k = k + 1u) {
            if (f_alive[k] == 0u) { continue; }
            let v0 = v_w[f_i0[k]];
            if (dot(f_normal[k], v_w[new_idx] - v0) > 0.0) {
                any_visible = true;
                f_alive[k] = 0u;
                let i0 = f_i0[k]; let i1 = f_i1[k]; let i2 = f_i2[k];
                _ = push_edge(i0, i1);
                _ = push_edge(i1, i2);
                _ = push_edge(i2, i0);
            }
        }
        if (!any_visible) {
            // Numerical issue — return current best.
            converged_face = face_idx;
            break;
        }

        // Stitch new fan: each silhouette edge (a, b) becomes a
        // triangle (a, b, new_idx).
        var stitched_any = false;
        for (var k: u32 = 0u; k < e_count; k = k + 1u) {
            let r = add_face(e_a[k], e_b[k], new_idx);
            if (r >= 0) { stitched_any = true; }
            if (r == -1) {
                // Polytope cap hit.
                write_contact(i, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_FALLBACK);
                return;
            }
        }
        if (!stitched_any) {
            converged_face = face_idx;
            break;
        }
    }

    if (converged_face < 0) {
        // Hit iter cap without convergence — return best face we have.
        converged_face = nearest_face();
        if (converged_face < 0) {
            write_contact(i, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_FALLBACK);
            return;
        }
    }

    let fi = u32(converged_face);
    let n = f_normal[fi];
    let d = f_dist[fi];
    let v0w = v_w[f_i0[fi]];
    let v1w = v_w[f_i1[fi]];
    let v2w = v_w[f_i2[fi]];
    let proj = n * d;
    let bc = barycentric(v0w, v1w, v2w, proj);
    let pa = v_a[f_i0[fi]] * bc.x + v_a[f_i1[fi]] * bc.y + v_a[f_i2[fi]] * bc.z;
    let pb = v_b[f_i0[fi]] * bc.x + v_b[f_i1[fi]] * bc.y + v_b[f_i2[fi]] * bc.z;
    write_contact(i, n, max(d, 0.0), pa, pb, STATUS_OK);
}
