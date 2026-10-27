// GPU MPR (Minkowski Portal Refinement) — alternative narrow-phase
// to GJK + EPA. Each thread handles one shape pair (matches the
// pattern used by gjk.wgsl / epa.wgsl). MPR computes the portal
// directly off the Minkowski difference and refines it until the
// portal face hugs the origin; this avoids EPA's polytope expansion
// and is more robust on deep penetration / degenerate simplices.
//
// Reference: crates/molla/src/geometry/mpr.rs (CPU port that this
// kernel mirrors line-for-line).
//
// Algorithm choice: ONE THREAD PER PAIR. The portal loop is purely
// scalar per pair (3 vertices + 1 interior point + one new sample
// per iter), so workgroup-level parallelism wouldn't help. 64 threads
// per workgroup matches the GJK/EPA kernels.
//
// Termination criterion (matches CPU):
//   `new_dist - face_dist < tolerance`
// where `face_dist = v1.w · n` (portal-face plane offset along its
// outward normal) and `new_dist = v4.w · n` (newest sample's offset).
// If the iteration cap is hit we emit the current portal with the
// last valid normal — STATUS_OK, not FALLBACK, so callers can
// inspect the depth and decide. Portal-edge locking is mitigated by
// the same sub-tetra selection logic used on CPU (Snethen §3.3).

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
const STATIC_BODY:    i32 = -1;

const MAX_ITERS: u32 = 64u;

const STATUS_NONE:     u32 = 0u; // pair skipped / disjoint
const STATUS_OK:       u32 = 1u; // converged manifold
const STATUS_FALLBACK: u32 = 2u; // host CPU should re-run

struct Uniforms {
    pair_count:  u32,
    shape_count: u32,
    tolerance:   f32, // portal-refinement convergence threshold
    _pad:        u32,
};

@group(0) @binding(0) var<storage, read> shape_geom_kind: array<u32>;
@group(0) @binding(1) var<storage, read> shape_geom_data: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> shape_body:      array<i32>;
@group(0) @binding(3) var<storage, read> shape_transform: array<Transform>;
@group(0) @binding(4) var<storage, read> body_q:          array<Transform>;

@group(1) @binding(0) var<storage, read> pair_a: array<u32>;
@group(1) @binding(1) var<storage, read> pair_b: array<u32>;

// Output stride: 16 f32 per pair, identical to EPA so callers can
// share decode helpers if convenient.
//   [0..3]   normal.xyz, depth
//   [4..7]   point_a.xyz, _
//   [8..11]  point_b.xyz, _
//   [12..15] status_bits_as_f32, _, _, _
@group(2) @binding(0) var<storage, read_write> mpr_out: array<f32>;
const MPR_OUT_STRIDE: u32 = 16u;

@group(3) @binding(0) var<uniform> u: Uniforms;

// ── Quat / transform helpers. ────────────────────────────────────
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

// ── Inline support functions (identical to gjk.wgsl / epa.wgsl). ─
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

// Per-vertex Minkowski-difference sample. `w = a - b` is the actual
// portal vertex; `a`, `b` retain the per-shape supports so we can
// recover witness points by barycentric mixing on the final portal.
struct Sp {
    w: vec3<f32>,
    a: vec3<f32>,
    b: vec3<f32>,
};

fn pair_support(sa: u32, sb: u32, dir: vec3<f32>) -> Sp {
    var out: Sp;
    out.a = world_support(sa, dir);
    out.b = world_support(sb, -dir);
    out.w = out.a - out.b;
    return out;
}

// Interior point of A − B: average opposing-direction supports along
// the 6 cardinal axes per shape, return the centroid difference.
// Always lies inside the Minkowski difference by convexity.
fn interior_point(sa: u32, sb: u32) -> Sp {
    var a_sum = vec3<f32>(0.0);
    var b_sum = vec3<f32>(0.0);
    var dirs = array<vec3<f32>, 6>(
        vec3<f32>( 1.0, 0.0, 0.0),
        vec3<f32>(-1.0, 0.0, 0.0),
        vec3<f32>(0.0,  1.0, 0.0),
        vec3<f32>(0.0, -1.0, 0.0),
        vec3<f32>(0.0, 0.0,  1.0),
        vec3<f32>(0.0, 0.0, -1.0),
    );
    for (var k: u32 = 0u; k < 6u; k = k + 1u) {
        a_sum = a_sum + world_support(sa, dirs[k]);
        b_sum = b_sum + world_support(sb, -dirs[k]);
    }
    let a_mid = a_sum / 6.0;
    let b_mid = b_sum / 6.0;
    var out: Sp;
    out.a = a_mid;
    out.b = b_mid;
    out.w = a_mid - b_mid;
    return out;
}

// Pick a non-zero vector perpendicular to `v` for portal-seed fallback.
fn perp_to(v: vec3<f32>) -> vec3<f32> {
    let ax = abs(v.x);
    let ay = abs(v.y);
    let az = abs(v.z);
    var other = vec3<f32>(0.0, 0.0, 1.0);
    if (ax <= ay && ax <= az) {
        other = vec3<f32>(1.0, 0.0, 0.0);
    } else if (ay <= az) {
        other = vec3<f32>(0.0, 1.0, 0.0);
    }
    var p = cross(v, other);
    if (dot(p, p) < 1e-24) {
        p = cross(v, vec3<f32>(0.0, 0.0, 1.0));
    }
    return p;
}

// Barycentric coords of `p` on triangle (a, b, c). Used to recover
// per-shape witnesses by mixing v1.a/v2.a/v3.a (and v1.b/...).
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

fn write_out(i: u32, normal: vec3<f32>, depth: f32, pa: vec3<f32>, pb: vec3<f32>, status: u32) {
    let base = i * MPR_OUT_STRIDE;
    mpr_out[base + 0u]  = normal.x;
    mpr_out[base + 1u]  = normal.y;
    mpr_out[base + 2u]  = normal.z;
    mpr_out[base + 3u]  = depth;
    mpr_out[base + 4u]  = pa.x;
    mpr_out[base + 5u]  = pa.y;
    mpr_out[base + 6u]  = pa.z;
    mpr_out[base + 7u]  = 0.0;
    mpr_out[base + 8u]  = pb.x;
    mpr_out[base + 9u]  = pb.y;
    mpr_out[base + 10u] = pb.z;
    mpr_out[base + 11u] = 0.0;
    mpr_out[base + 12u] = bitcast<f32>(status);
    mpr_out[base + 13u] = 0.0;
    mpr_out[base + 14u] = 0.0;
    mpr_out[base + 15u] = 0.0;
}

// Build the final manifold from the converged portal (v1, v2, v3).
fn emit_contact(i: u32, v1: Sp, v2: Sp, v3: Sp, normal_in: vec3<f32>) {
    let face_dist = dot(v1.w, normal_in);
    let proj = normal_in * face_dist;
    let bc = barycentric(v1.w, v2.w, v3.w, proj);
    let pa = v1.a * bc.x + v2.a * bc.y + v3.a * bc.z;
    let pb = v1.b * bc.x + v2.b * bc.y + v3.b * bc.z;
    var n_final = normal_in;
    if (dot(normal_in, normal_in) < 0.5) {
        let d = pb - pa;
        let dl = length(d);
        if (dl > 1e-12) {
            n_final = d / dl;
        } else {
            n_final = vec3<f32>(0.0, 1.0, 0.0);
        }
    }
    write_out(i, n_final, max(face_dist, 0.0), pa, pb, STATUS_OK);
}

@compute @workgroup_size(64)
fn mpr_collide_pair(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.pair_count) { return; }
    // Default: status_none (no contact / skipped).
    write_out(i, vec3<f32>(0.0, 1.0, 0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_NONE);

    let sa = pair_a[i];
    let sb = pair_b[i];
    if (sa >= u.shape_count || sb >= u.shape_count) { return; }
    let ka = shape_geom_kind[sa];
    let kb = shape_geom_kind[sb];
    if (ka == KIND_PLANE || kb == KIND_PLANE
        || ka == KIND_SKIP || kb == KIND_SKIP) {
        write_out(i, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_FALLBACK);
        return;
    }

    let tol = u.tolerance;

    // ── Phase 1: Interior point v0 of A − B. ───────────────────
    let v0 = interior_point(sa, sb);
    if (dot(v0.w, v0.w) < 1e-24) {
        // Origin sits at the centroid difference → bodies coincident.
        // Probe along +X to recover a stable normal.
        let probe = pair_support(sa, sb, vec3<f32>(1.0, 0.0, 0.0));
        var n = vec3<f32>(0.0, 1.0, 0.0);
        let pw_sq = dot(probe.w, probe.w);
        if (pw_sq > 1e-24) {
            n = probe.w * inverseSqrt(pw_sq);
        }
        write_out(i, n, 0.0, v0.a, v0.b, STATUS_OK);
        return;
    }

    // ── Phase 2: First portal vertex v1 — support along −v0. ───
    var dir = -v0.w;
    var v1 = pair_support(sa, sb, dir);
    if (dot(v1.w, dir) <= 0.0) {
        // Origin outside the supporting half-space → disjoint.
        write_out(i, vec3<f32>(0.0, 1.0, 0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_NONE);
        return;
    }

    // ── Phase 3: Second portal vertex v2. ──────────────────────
    dir = cross(v0.w, v1.w);
    if (dot(dir, dir) < 1e-24) {
        // v0, v1, origin colinear — pick a perpendicular.
        dir = perp_to(v1.w - v0.w);
        if (dot(dir, dir) < 1e-24) {
            write_out(i, vec3<f32>(0.0, 1.0, 0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_NONE);
            return;
        }
    }
    var v2 = pair_support(sa, sb, dir);
    if (dot(v2.w, dir) <= 0.0) {
        write_out(i, vec3<f32>(0.0, 1.0, 0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_NONE);
        return;
    }

    // ── Phase 4: Find v3 such that (v1, v2, v3) brackets the ray
    // from v0 through the origin. ──────────────────────────────
    var v3: Sp;
    var found_bracket = false;
    for (var iter: u32 = 0u; iter < MAX_ITERS; iter = iter + 1u) {
        dir = cross(v1.w - v0.w, v2.w - v0.w);
        if (dot(dir, v0.w) > 0.0) {
            dir = -dir;
        }
        if (dot(dir, dir) < 1e-24) {
            write_out(i, vec3<f32>(0.0, 1.0, 0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_NONE);
            return;
        }
        let v3_try = pair_support(sa, sb, dir);
        if (dot(v3_try.w, dir) <= 0.0) {
            write_out(i, vec3<f32>(0.0, 1.0, 0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_NONE);
            return;
        }
        let cross_v0_v3 = cross(v0.w, v3_try.w);
        if (dot(cross_v0_v3, v1.w) < 0.0) {
            // Origin on the v2 side of v0-v1-v3 plane → replace v2.
            v2 = v3_try;
            continue;
        }
        if (dot(cross_v0_v3, v2.w) > 0.0) {
            // Origin on the v1 side of v0-v2-v3 plane → replace v1.
            v1 = v3_try;
            continue;
        }
        v3 = v3_try;
        found_bracket = true;
        break;
    }
    if (!found_bracket) {
        // Couldn't bracket within MAX_ITERS — host fallback.
        write_out(i, vec3<f32>(0.0), 0.0, vec3<f32>(0.0), vec3<f32>(0.0), STATUS_FALLBACK);
        return;
    }

    // ── Phase 5: Refine the portal. ────────────────────────────
    var last_n = vec3<f32>(0.0, 1.0, 0.0);
    for (var iter: u32 = 0u; iter < MAX_ITERS; iter = iter + 1u) {
        var n_raw = cross(v2.w - v1.w, v3.w - v1.w);
        // Orient outward (away from v0).
        if (dot(n_raw, v1.w - v0.w) < 0.0) {
            n_raw = -n_raw;
        }
        let n_len_sq = dot(n_raw, n_raw);
        if (n_len_sq < 1e-24) {
            // Degenerate portal — emit best-effort contact.
            emit_contact(i, v1, v2, v3, last_n);
            return;
        }
        let n = n_raw * inverseSqrt(n_len_sq);
        last_n = n;

        let face_dist = dot(v1.w, n);
        let v4 = pair_support(sa, sb, n);
        let new_dist = dot(v4.w, n);

        // Termination criterion — new support no further past the
        // portal than `tolerance`. Identical to CPU `mpr.rs` L308.
        if (new_dist - face_dist < tol) {
            emit_contact(i, v1, v2, v3, n);
            return;
        }

        // Choose which of v1/v2/v3 to drop (Snethen §3.3): test the
        // sign of (v4 × v0) · v_i to find the sub-tetra containing
        // the ray v0 → origin.
        let cross_v4_v0 = cross(v4.w, v0.w);
        let d1 = dot(v1.w, cross_v4_v0);
        let d2 = dot(v2.w, cross_v4_v0);
        let d3 = dot(v3.w, cross_v4_v0);

        if (d1 >= 0.0) {
            if (d2 >= 0.0) {
                v3 = v4;
            } else {
                v2 = v4;
            }
        } else {
            if (d3 >= 0.0) {
                v1 = v4;
            } else {
                v3 = v4;
            }
        }
    }

    // Iter cap hit. The portal-edge lock case ends up here: we keep
    // the last valid normal and emit current best as STATUS_OK
    // (mirrors CPU `mpr.rs` which also returns `Some(...)` from the
    // iter-cap branch).
    emit_contact(i, v1, v2, v3, last_n);
}
