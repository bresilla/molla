// GPU GJK boolean overlap test — one thread per candidate shape pair
// from the broad phase. Inlines support functions for the supported
// primitives (Sphere/Box/Capsule/Cylinder/Cone). ConvexHull / TriMesh
// stay on CPU (would need additional storage for hull vertex tables).
// Plane is also CPU-handled because GJK doesn't terminate against
// half-spaces.
//
// Output: per-pair `overlap[i] = 1` if the two shapes overlap, `0`
// otherwise. Caller (the host pipeline) feeds the overlapping subset
// to GPU EPA (epa.wgsl) for contact generation. When overlap == 1 the
// kernel also writes the 4-vertex Minkowski-difference simplex into
// `gjk_simplex[i]` for EPA to bootstrap its initial polytope.

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
const STATIC_BODY: i32 = -1;

struct Uniforms {
    pair_count:  u32,
    shape_count: u32,
    _pad0:       u32,
    _pad1:       u32,
};

@group(0) @binding(0) var<storage, read>      shape_geom_kind:  array<u32>;
@group(0) @binding(1) var<storage, read>      shape_geom_data:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>      shape_body:       array<i32>;
@group(0) @binding(3) var<storage, read>      shape_transform:  array<Transform>;
@group(0) @binding(4) var<storage, read>      body_q:           array<Transform>;

@group(1) @binding(0) var<storage, read>      pair_a:           array<u32>;
@group(1) @binding(1) var<storage, read>      pair_b:           array<u32>;

@group(2) @binding(0) var<storage, read_write> overlap:         array<u32>;
// Optional simplex output. Each pair writes 12 vec4<f32>: 4 vertices
// × 3 vec3 (w, a, b). The kernel writes nothing when the host hasn't
// bound a real buffer (see GjkKernel::dispatch_with_simplex). 12 ×
// vec4 = 192 bytes per pair, well under typical storage limits.
@group(2) @binding(1) var<storage, read_write> gjk_simplex:     array<vec4<f32>>;

@group(3) @binding(0) var<uniform>             u: Uniforms;

// ── Quat / transform helpers (mirror raycast.wgsl). ──────────────
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

// ── Inline support functions in shape-local frame. ───────────────
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
        // Ellipsoid support: rescale dir by per-axis semi-axes,
        // normalise, then rescale again. See `support_ellipsoid` in
        // crates/molla/src/geometry/support.rs.
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
    // KIND_PLANE / KIND_SKIP: caller filters before calling.
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

fn minkowski_support(a: u32, b: u32, dir: vec3<f32>) -> vec3<f32> {
    return world_support(a, dir) - world_support(b, -dir);
}

// Variant that also returns the per-shape support points used for
// Minkowski difference. EPA needs the (a, b) pair to recover contact
// witnesses — boolean GJK only needed `w = a - b`.
struct SupportPair {
    w: vec3<f32>,
    a: vec3<f32>,
    b: vec3<f32>,
};

fn minkowski_support_pair(sa: u32, sb: u32, dir: vec3<f32>) -> SupportPair {
    var out: SupportPair;
    out.a = world_support(sa, dir);
    out.b = world_support(sb, -dir);
    out.w = out.a - out.b;
    return out;
}

// ── Simplex update routines (Bullet-style). ──────────────────────
//
// Each `update_simplex_*` operates on a fixed 4-slot stack and
// returns the new search direction (toward the origin from the
// closest sub-feature). The "ContainsOrigin" case sets `overlap`
// and bails out of the loop in the caller.

fn triple(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    return cross(cross(a, b), c);
}

// Result encoding: w-component of return value is non-zero when the
// simplex contains the origin (caller should report overlap). xyz =
// new search direction. The per-vertex `a`/`b`/`c`/`d` slots carry
// (w, sa, sb) triples so EPA can recover the original support points
// without re-querying.
struct SimplexVtx {
    w: vec3<f32>,
    a: vec3<f32>,
    b: vec3<f32>,
};
struct SimplexStep {
    dir:           vec3<f32>,
    contains:      u32, // 1 = origin enclosed, 0 = continue
    new_count:     u32,
    a: SimplexVtx,
    b: SimplexVtx,
    c: SimplexVtx,
    d: SimplexVtx,
};

fn line_step(a: SimplexVtx, b: SimplexVtx) -> SimplexStep {
    // Latest support is `a`, previous was `b`. Direction = ⊥AB toward origin.
    let ab = b.w - a.w;
    let ao = -a.w;
    var out: SimplexStep;
    if (dot(ab, ao) > 0.0) {
        out.dir = triple(ab, ao, ab);
        if (dot(out.dir, out.dir) < 1e-20) {
            // origin lies on AB
            out.dir = vec3<f32>(0.0, 1.0, 0.0);
        }
        out.contains = 0u;
        out.new_count = 2u;
        out.a = a; out.b = b;
    } else {
        out.dir = ao;
        out.contains = 0u;
        out.new_count = 1u;
        out.a = a;
    }
    return out;
}

fn triangle_step(a: SimplexVtx, b: SimplexVtx, c: SimplexVtx) -> SimplexStep {
    let ab = b.w - a.w;
    let ac = c.w - a.w;
    let ao = -a.w;
    let abc = cross(ab, ac);
    var out: SimplexStep;
    out.contains = 0u;

    let ac_perp = cross(abc, ac);
    if (dot(ac_perp, ao) > 0.0) {
        if (dot(ac, ao) > 0.0) {
            out.dir = triple(ac, ao, ac);
            out.new_count = 2u;
            out.a = a; out.b = c;
            return out;
        }
        return line_step(a, b);
    }
    let ab_perp = cross(ab, abc);
    if (dot(ab_perp, ao) > 0.0) {
        return line_step(a, b);
    }
    // Origin is above or below the triangle. Pick the side facing it.
    if (dot(abc, ao) > 0.0) {
        out.dir = abc;
        out.new_count = 3u;
        out.a = a; out.b = b; out.c = c;
    } else {
        out.dir = -abc;
        out.new_count = 3u;
        out.a = a; out.b = c; out.c = b;
    }
    return out;
}

fn tetra_step(a: SimplexVtx, b: SimplexVtx, c: SimplexVtx, d: SimplexVtx) -> SimplexStep {
    let ab = b.w - a.w;
    let ac = c.w - a.w;
    let ad = d.w - a.w;
    let ao = -a.w;

    let abc = cross(ab, ac);
    let acd = cross(ac, ad);
    let adb = cross(ad, ab);

    var out: SimplexStep;
    out.contains = 0u;
    if (dot(abc, ao) > 0.0) {
        return triangle_step(a, b, c);
    }
    if (dot(acd, ao) > 0.0) {
        return triangle_step(a, c, d);
    }
    if (dot(adb, ao) > 0.0) {
        return triangle_step(a, d, b);
    }
    out.contains = 1u;
    out.new_count = 4u;
    out.a = a; out.b = b; out.c = c; out.d = d;
    return out;
}

// Write a 4-vertex simplex into `gjk_simplex[i]`. Layout per pair:
// 12 vec4<f32> = 4 vertices × 3 vec3 (w, a, b). The `.w` lane of each
// vec4 is left as 0; EPA reads `.xyz`. The third vec4 of vertex 0
// stores `count` in its `.w` lane so EPA knows how many simplex
// vertices are valid.
fn write_simplex(i: u32, count: u32, v0: SimplexVtx, v1: SimplexVtx, v2: SimplexVtx, v3: SimplexVtx) {
    let base = i * 12u;
    var verts: array<SimplexVtx, 4>;
    verts[0] = v0;
    if (count >= 2u) { verts[1] = v1; } else {
        verts[1] = SimplexVtx(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    }
    if (count >= 3u) { verts[2] = v2; } else {
        verts[2] = SimplexVtx(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    }
    if (count >= 4u) { verts[3] = v3; } else {
        verts[3] = SimplexVtx(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    }
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let v = verts[k];
        gjk_simplex[base + k * 3u + 0u] = vec4<f32>(v.w, 0.0);
        gjk_simplex[base + k * 3u + 1u] = vec4<f32>(v.a, 0.0);
        // Stash the simplex-vertex count in vertex 0's .w lane (EPA
        // reads vertex 0's third vec4 to recover it).
        let pad = select(0.0, f32(count), k == 0u);
        gjk_simplex[base + k * 3u + 2u] = vec4<f32>(v.b, pad);
    }
}

// ── GJK kernel. ─────────────────────────────────────────────────
@compute @workgroup_size(64)
fn gjk_overlap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.pair_count) { return; }
    overlap[i] = 0u;

    let sa = pair_a[i];
    let sb = pair_b[i];
    if (sa >= u.shape_count || sb >= u.shape_count) { return; }
    let ka = shape_geom_kind[sa];
    let kb = shape_geom_kind[sb];
    // Plane / TriMesh / unsupported: signal a sentinel by setting
    // `overlap = 2` so the host knows to fall back to CPU narrow phase.
    if (ka == KIND_PLANE || kb == KIND_PLANE
        || ka == KIND_SKIP || kb == KIND_SKIP) {
        overlap[i] = 2u;
        return;
    }

    // Initial direction.
    let xf_a = world_xf_of(sa);
    let xf_b = world_xf_of(sb);
    var dir = xf_a.position.xyz - xf_b.position.xyz;
    if (dot(dir, dir) < 1e-12) {
        dir = vec3<f32>(1.0, 0.0, 0.0);
    }

    let p0 = minkowski_support_pair(sa, sb, dir);
    var sv1: SimplexVtx = SimplexVtx(p0.w, p0.a, p0.b);
    var sv2: SimplexVtx = SimplexVtx(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
    var sv3: SimplexVtx = sv2;
    var sv4: SimplexVtx = sv2;
    var count: u32 = 1u;
    dir = -sv1.w;

    let max_iters = 32u;
    for (var iter: u32 = 0u; iter < max_iters; iter = iter + 1u) {
        if (dot(dir, dir) < 1e-20) {
            // Search direction collapsed — origin is on the simplex
            // boundary. Treat as overlap.
            overlap[i] = 1u;
            write_simplex(i, count, sv1, sv2, sv3, sv4);
            return;
        }
        let new_pair = minkowski_support_pair(sa, sb, dir);
        if (dot(new_pair.w, dir) < 0.0) {
            // No support past origin in this direction → disjoint.
            overlap[i] = 0u;
            return;
        }
        let new_v: SimplexVtx = SimplexVtx(new_pair.w, new_pair.a, new_pair.b);
        // Push new point as the latest (`a`) and shift the old ones.
        sv4 = sv3; sv3 = sv2; sv2 = sv1; sv1 = new_v;
        count = min(count + 1u, 4u);

        var step: SimplexStep;
        if (count == 2u) {
            step = line_step(sv1, sv2);
        } else if (count == 3u) {
            step = triangle_step(sv1, sv2, sv3);
        } else {
            step = tetra_step(sv1, sv2, sv3, sv4);
        }
        if (step.contains == 1u) {
            overlap[i] = 1u;
            write_simplex(i, step.new_count, step.a, step.b, step.c, step.d);
            return;
        }
        dir = step.dir;
        count = step.new_count;
        sv1 = step.a;
        if (count >= 2u) { sv2 = step.b; }
        if (count >= 3u) { sv3 = step.c; }
        if (count >= 4u) { sv4 = step.d; }
    }
    // Hit iteration cap without converging → fall back to CPU.
    overlap[i] = 2u;
}
