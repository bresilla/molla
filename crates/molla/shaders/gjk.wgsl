// GPU GJK boolean overlap test — one thread per candidate shape pair
// from the broad phase. Inlines support functions for the supported
// primitives (Sphere/Box/Capsule/Cylinder/Cone). ConvexHull / TriMesh
// stay on CPU (would need additional storage for hull vertex tables).
// Plane is also CPU-handled because GJK doesn't terminate against
// half-spaces.
//
// Output: per-pair `overlap[i] = 1` if the two shapes overlap, `0`
// otherwise. Caller (the host pipeline) feeds the overlapping subset
// to CPU EPA for contact generation. This pass cuts narrow-phase
// work for scenes where the broad phase emits many pairs that don't
// actually overlap.

struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,
};

const KIND_SPHERE:   u32 = 0u;
const KIND_PLANE:    u32 = 1u;
const KIND_BOX:      u32 = 2u;
const KIND_CAPSULE:  u32 = 3u;
const KIND_CYLINDER: u32 = 4u;
const KIND_CONE:     u32 = 5u;
const KIND_SKIP:     u32 = 0xffu;
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
// new search direction.
struct SimplexStep {
    dir:           vec3<f32>,
    contains:      u32, // 1 = origin enclosed, 0 = continue
    new_count:     u32,
    a: vec3<f32>,
    b: vec3<f32>,
    c: vec3<f32>,
    d: vec3<f32>,
};

fn line_step(a: vec3<f32>, b: vec3<f32>) -> SimplexStep {
    // Latest support is `a`, previous was `b`. Direction = ⊥AB toward origin.
    let ab = b - a;
    let ao = -a;
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

fn triangle_step(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> SimplexStep {
    let ab = b - a;
    let ac = c - a;
    let ao = -a;
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

fn tetra_step(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> SimplexStep {
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let ao = -a;

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

    var sa1 = minkowski_support(sa, sb, dir);
    var sa2 = vec3<f32>(0.0);
    var sa3 = vec3<f32>(0.0);
    var sa4 = vec3<f32>(0.0);
    var count: u32 = 1u;
    dir = -sa1;

    let max_iters = 32u;
    for (var iter: u32 = 0u; iter < max_iters; iter = iter + 1u) {
        if (dot(dir, dir) < 1e-20) {
            // Search direction collapsed — origin is on the simplex
            // boundary. Treat as overlap.
            overlap[i] = 1u;
            return;
        }
        let new_p = minkowski_support(sa, sb, dir);
        if (dot(new_p, dir) < 0.0) {
            // No support past origin in this direction → disjoint.
            overlap[i] = 0u;
            return;
        }
        // Push new point as the latest (`a`) and shift the old ones.
        sa4 = sa3; sa3 = sa2; sa2 = sa1; sa1 = new_p;
        count = min(count + 1u, 4u);

        var step: SimplexStep;
        if (count == 2u) {
            step = line_step(sa1, sa2);
        } else if (count == 3u) {
            step = triangle_step(sa1, sa2, sa3);
        } else {
            step = tetra_step(sa1, sa2, sa3, sa4);
        }
        if (step.contains == 1u) {
            overlap[i] = 1u;
            return;
        }
        dir = step.dir;
        count = step.new_count;
        sa1 = step.a;
        if (count >= 2u) { sa2 = step.b; }
        if (count >= 3u) { sa3 = step.c; }
        if (count >= 4u) { sa4 = step.d; }
    }
    // Hit iteration cap without converging → fall back to CPU.
    overlap[i] = 2u;
}
