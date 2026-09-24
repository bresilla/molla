// Ray-cast compute kernel — closest-hit ray vs every collision shape
// in a `Model`. One thread per ray, sequential scan over shapes.
//
// Mirrors Newton/Warp's `ray_intersect_sphere/_box/_capsule/_cylinder/
// _plane` device funcs (xtra/newton/_src/geometry/raycast.py) and
// `sensor_raycast_kernel`. Triangle / TriMesh shapes are flagged as
// kind=0xff in `shape_geom_kind` and skipped here (host fallback for
// trimesh raycasts until stage 11 ports the mesh BVH).
//
// Inputs:
//   shape_geom_kind  : per-shape discriminant (0..4 = supported, 0xff = skip)
//   shape_geom_data  : per-shape packed parameters (vec3 padded to vec4)
//   shape_body       : per-shape owning body index (-1 = static world)
//   shape_transform  : per-shape transform in body frame (Transform = pos+quat)
//   body_q           : per-body world-frame pose (Transform)
//   ray_origin       : per-ray world-frame origin
//   ray_direction    : per-ray world-frame direction (need not be unit)
//   uniforms         : ray_count, shape_count, max_distance
//
// Output:
//   hit_t            : per-ray distance along ray, or -1 if no hit
//   hit_shape        : per-ray shape index, or -1 if no hit
//   hit_point        : per-ray world-space hit point
//   hit_normal       : per-ray world-space outward surface normal

// ── Shape-kind constants (must match `Model::shape_geom_kind` doc). ──
const KIND_SPHERE:    u32 = 0u;
const KIND_PLANE:     u32 = 1u;
const KIND_BOX:       u32 = 2u;
const KIND_CAPSULE:   u32 = 3u;
const KIND_CYLINDER:  u32 = 4u;
const KIND_CONE:      u32 = 5u;
const KIND_ELLIPSOID: u32 = 6u;
const KIND_SKIP:      u32 = 0xffu;

const STATIC_BODY: i32 = -1;
const NO_HIT_T: f32 = -1.0;
const EPS: f32 = 1e-9;

// ── Layouts ───────────────────────────────────────────────────────
struct Transform {
    position: vec4<f32>,  // xyz used, w padded
    rotation: vec4<f32>,  // (x, y, z, w) quaternion
};

struct Uniforms {
    ray_count:    u32,
    shape_count:  u32,
    max_distance: f32,
    _pad:         f32,
};

@group(0) @binding(0) var<storage, read>  shape_geom_kind:  array<u32>;
@group(0) @binding(1) var<storage, read>  shape_geom_data:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>  shape_body:       array<i32>;
@group(0) @binding(3) var<storage, read>  shape_transform:  array<Transform>;
@group(0) @binding(4) var<storage, read>  body_q:           array<Transform>;

@group(1) @binding(0) var<storage, read>  ray_origin:    array<vec4<f32>>;
@group(1) @binding(1) var<storage, read>  ray_direction: array<vec4<f32>>;

@group(2) @binding(0) var<storage, read_write> hit_t:      array<f32>;
@group(2) @binding(1) var<storage, read_write> hit_shape:  array<i32>;
@group(2) @binding(2) var<storage, read_write> hit_point:  array<vec4<f32>>;
@group(2) @binding(3) var<storage, read_write> hit_normal: array<vec4<f32>>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

// ── Quaternion + transform helpers ───────────────────────────────
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

fn transform_point(xf: Transform, p: vec3<f32>) -> vec3<f32> {
    return xf.position.xyz + quat_rotate(xf.rotation, p);
}

fn transform_compose(parent: Transform, local: Transform) -> Transform {
    var out: Transform;
    out.position = vec4<f32>(parent.position.xyz + quat_rotate(parent.rotation, local.position.xyz), 0.0);
    out.rotation = quat_mul(parent.rotation, local.rotation);
    return out;
}

fn transform_inverse(xf: Transform) -> Transform {
    let inv_rot = quat_conj(xf.rotation);
    var out: Transform;
    out.rotation = inv_rot;
    out.position = vec4<f32>(-quat_rotate(inv_rot, xf.position.xyz), 0.0);
    return out;
}

// ── Per-shape ray intersection in shape-local frame. ─────────────
// Each returns t > 0 if hit, < 0 if miss. `out_normal` is the
// outward shape-local normal at the hit point.

fn ray_sphere_local(o: vec3<f32>, d: vec3<f32>, r: f32) -> f32 {
    let b = dot(o, d);
    let c = dot(o, o) - r * r;
    let disc = b * b - c;
    if (disc < 0.0) { return NO_HIT_T; }
    let sd = sqrt(disc);
    let t1 = -b - sd;
    if (t1 > 0.0) { return t1; }
    let t2 = -b + sd;
    if (t2 > 0.0) { return t2; }
    return NO_HIT_T;
}

fn ray_plane_local(o: vec3<f32>, d: vec3<f32>, n: vec3<f32>) -> f32 {
    let denom = dot(n, d);
    if (abs(denom) < EPS) { return NO_HIT_T; }
    let t = -dot(n, o) / denom;
    if (t > 0.0) { return t; }
    return NO_HIT_T;
}

fn ray_box_local(o: vec3<f32>, d: vec3<f32>, h: vec3<f32>) -> f32 {
    // Slab method.
    var tmin = -1.0e30;
    var tmax =  1.0e30;
    for (var k: u32 = 0u; k < 3u; k = k + 1u) {
        let oi = o[k];
        let di = d[k];
        let hi = h[k];
        if (abs(di) < EPS) {
            if (abs(oi) > hi) { return NO_HIT_T; }
        } else {
            let inv = 1.0 / di;
            let t1 = (-hi - oi) * inv;
            let t2 = ( hi - oi) * inv;
            let lo = min(t1, t2);
            let hi2 = max(t1, t2);
            tmin = max(tmin, lo);
            tmax = min(tmax, hi2);
            if (tmax < tmin) { return NO_HIT_T; }
        }
    }
    if (tmax < 0.0) { return NO_HIT_T; }
    if (tmin > 0.0) { return tmin; }
    return tmax;
}

fn ray_capsule_local(o: vec3<f32>, d: vec3<f32>, half_h: f32, r: f32) -> f32 {
    // Capsule axis = +Y. Cylindrical body of half-height h plus
    // hemispheres at ±h. Take min of three ray-primitive tests.
    var best = -1.0;

    // Side: infinite cylinder around Y, then check |y| <= half_h.
    let a = d.x * d.x + d.z * d.z;
    if (a > 1e-18) {
        let b = o.x * d.x + o.z * d.z;
        let c = o.x * o.x + o.z * o.z - r * r;
        let disc = b * b - a * c;
        if (disc >= 0.0) {
            let sd = sqrt(disc);
            let inv = 1.0 / a;
            let t1 = (-b - sd) * inv;
            let t2 = (-b + sd) * inv;
            // Pick the closer positive one whose y is within the cylinder.
            if (t1 > 0.0) {
                let y = o.y + d.y * t1;
                if (abs(y) <= half_h) { best = t1; }
            }
            if (best < 0.0 && t2 > 0.0) {
                let y = o.y + d.y * t2;
                if (abs(y) <= half_h) { best = t2; }
            }
        }
    }

    // Endcap spheres at (0, ±half_h, 0).
    for (var s = 0u; s < 2u; s = s + 1u) {
        var cap_y = half_h;
        if (s == 1u) { cap_y = -half_h; }
        let oc = vec3<f32>(o.x, o.y - cap_y, o.z);
        let t = ray_sphere_local(oc, d, r);
        if (t > 0.0 && (best < 0.0 || t < best)) {
            best = t;
        }
    }
    return best;
}

fn ray_cylinder_local(o: vec3<f32>, d: vec3<f32>, half_h: f32, r: f32) -> f32 {
    // Y-axis cylinder with flat caps at y = ±half_h.
    var best = -1.0;

    // Side surface.
    let a = d.x * d.x + d.z * d.z;
    if (a > 1e-18) {
        let b = o.x * d.x + o.z * d.z;
        let c = o.x * o.x + o.z * o.z - r * r;
        let disc = b * b - a * c;
        if (disc >= 0.0) {
            let sd = sqrt(disc);
            let inv = 1.0 / a;
            let t1 = (-b - sd) * inv;
            let t2 = (-b + sd) * inv;
            if (t1 > 0.0) {
                let y = o.y + d.y * t1;
                if (abs(y) <= half_h) { best = t1; }
            }
            if (best < 0.0 && t2 > 0.0) {
                let y = o.y + d.y * t2;
                if (abs(y) <= half_h) { best = t2; }
            }
        }
    }

    // Caps.
    if (abs(d.y) >= EPS) {
        for (var s = 0u; s < 2u; s = s + 1u) {
            var cap_y = half_h;
            if (s == 1u) { cap_y = -half_h; }
            let t = (cap_y - o.y) / d.y;
            if (t > 0.0) {
                let x = o.x + d.x * t;
                let z = o.z + d.z * t;
                if (x * x + z * z <= r * r && (best < 0.0 || t < best)) {
                    best = t;
                }
            }
        }
    }
    return best;
}

// ── Ellipsoid (per-axis semi-axes `data.xyz`). Scale space by
// 1/half_axes so the surface becomes a unit sphere; intersect; the
// ray-parameter t is preserved by the diagonal scale.
fn ray_cone_local(o: vec3<f32>, d: vec3<f32>, half_h: f32, radius: f32) -> f32 {
    let k = radius / (2.0 * half_h);
    let k2 = k * k;
    let shifted_y = o.y - half_h;
    let a = dot(d.xz, d.xz) - k2 * d.y * d.y;
    let b = dot(o.xz, d.xz) - k2 * shifted_y * d.y;
    let c = dot(o.xz, o.xz) - k2 * shifted_y * shifted_y;
    var roots = vec2<f32>(NO_HIT_T);
    if (abs(a) <= 1e-7 * max(1.0, k2)) {
        if (abs(b) > EPS) { roots.x = -c / (2.0 * b); }
    } else {
        let discriminant = b * b - a * c;
        if (discriminant >= 0.0) {
            let q = -b - select(-sqrt(discriminant), sqrt(discriminant), b >= 0.0);
            if (abs(q) > EPS) {
                roots = vec2<f32>(q / a, c / q);
            } else {
                roots = vec2<f32>(-b / a);
            }
        }
    }
    var best = NO_HIT_T;
    for (var i = 0u; i < 2u; i += 1u) {
        let t = roots[i];
        let y = o.y + t * d.y;
        if (t > 0.0 && y >= -half_h && y <= half_h && (best < 0.0 || t < best)) {
            best = t;
        }
    }
    if (abs(d.y) > EPS) {
        let t = (-half_h - o.y) / d.y;
        let p = o.xz + t * d.xz;
        if (t > 0.0 && dot(p, p) <= radius * radius && (best < 0.0 || t < best)) {
            best = t;
        }
    }
    return best;
}

fn ray_ellipsoid_local(o: vec3<f32>, d: vec3<f32>, half: vec3<f32>) -> f32 {
    let inv = vec3<f32>(1.0 / half.x, 1.0 / half.y, 1.0 / half.z);
    let o_s = o * inv;
    let d_s = d * inv;
    let a = dot(d_s, d_s);
    if (a < 1e-24) { return 1e30; }
    let b = dot(o_s, d_s);
    let c = dot(o_s, o_s) - 1.0;
    let disc = b * b - a * c;
    if (disc < 0.0) { return 1e30; }
    let sqd = sqrt(disc);
    let t0 = (-b - sqd) / a;
    let t1 = (-b + sqd) / a;
    if (t0 > 0.0) { return t0; }
    if (t1 > 0.0) { return t1; }
    return 1e30;
}

// ── Outward normal at a shape-local hit point. ─────────────────
fn shape_normal(kind: u32, data: vec3<f32>, p_local: vec3<f32>) -> vec3<f32> {
    if (kind == KIND_SPHERE) {
        return normalize(p_local);
    } else if (kind == KIND_PLANE) {
        return data;
    } else if (kind == KIND_BOX) {
        let half = data;
        let nx = abs(p_local.x / half.x);
        let ny = abs(p_local.y / half.y);
        let nz = abs(p_local.z / half.z);
        if (nx >= ny && nx >= nz) {
            return vec3<f32>(sign(p_local.x), 0.0, 0.0);
        } else if (ny >= nz) {
            return vec3<f32>(0.0, sign(p_local.y), 0.0);
        } else {
            return vec3<f32>(0.0, 0.0, sign(p_local.z));
        }
    } else if (kind == KIND_CAPSULE) {
        let half_h = data.x;
        if (abs(p_local.y) <= half_h) {
            return normalize(vec3<f32>(p_local.x, 0.0, p_local.z));
        }
        var cap_y = half_h;
        if (p_local.y < 0.0) { cap_y = -half_h; }
        return normalize(vec3<f32>(p_local.x, p_local.y - cap_y, p_local.z));
    } else if (kind == KIND_CYLINDER) {
        let half_h = data.x;
        if (abs(p_local.y - half_h) < 1e-5) { return vec3<f32>(0.0, 1.0, 0.0); }
        if (abs(p_local.y + half_h) < 1e-5) { return vec3<f32>(0.0, -1.0, 0.0); }
        return normalize(vec3<f32>(p_local.x, 0.0, p_local.z));
    } else if (kind == KIND_CONE) {
        if (abs(p_local.y + data.x) < 1e-5) { return vec3<f32>(0.0, -1.0, 0.0); }
        let k = data.y / (2.0 * data.x);
        let n = vec3<f32>(p_local.x, k * k * (data.x - p_local.y), p_local.z);
        if (dot(n, n) < 1e-20) { return vec3<f32>(0.0, 1.0, 0.0); }
        return normalize(n);
    } else if (kind == KIND_ELLIPSOID) {
        // Normal at a surface point on an axis-aligned ellipsoid is
        // gradient(p²/a²) = (px/a², py/b², pz/c²), normalised.
        return normalize(vec3<f32>(
            p_local.x / (data.x * data.x),
            p_local.y / (data.y * data.y),
            p_local.z / (data.z * data.z),
        ));
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

// ── Master kernel — one thread per ray. ─────────────────────────
@compute @workgroup_size(64)
fn cast_rays(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.ray_count) { return; }

    let ro = ray_origin[tid].xyz;
    let rd_raw = ray_direction[tid].xyz;
    let rd_len = length(rd_raw);
    if (rd_len < 1e-12) {
        hit_t[tid] = NO_HIT_T;
        hit_shape[tid] = -1;
        hit_point[tid] = vec4<f32>(0.0);
        hit_normal[tid] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
        return;
    }
    let rd = rd_raw / rd_len;

    var best_t: f32 = uniforms.max_distance;
    var best_shape: i32 = -1;
    var best_point: vec3<f32> = vec3<f32>(0.0);
    var best_normal: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

    for (var i: u32 = 0u; i < uniforms.shape_count; i = i + 1u) {
        let kind = shape_geom_kind[i];
        if (kind == KIND_SKIP) { continue; }

        // World transform = body × shape-local.
        let body = shape_body[i];
        var world_xform: Transform;
        if (body >= 0) {
            world_xform = transform_compose(body_q[u32(body)], shape_transform[i]);
        } else {
            world_xform = shape_transform[i];
        }

        // Ray in shape-local frame.
        let inv = transform_inverse(world_xform);
        let lo = transform_point(inv, ro);
        let ld = quat_rotate(inv.rotation, rd);

        let data = shape_geom_data[i].xyz;
        var t: f32 = NO_HIT_T;
        if      (kind == KIND_SPHERE)    { t = ray_sphere_local(lo, ld, data.x); }
        else if (kind == KIND_PLANE)     { t = ray_plane_local(lo, ld, data); }
        else if (kind == KIND_BOX)       { t = ray_box_local(lo, ld, data); }
        else if (kind == KIND_CAPSULE)   { t = ray_capsule_local(lo, ld, data.x, data.y); }
        else if (kind == KIND_CYLINDER)  { t = ray_cylinder_local(lo, ld, data.x, data.y); }
        else if (kind == KIND_CONE)      { t = ray_cone_local(lo, ld, data.x, data.y); }
        else if (kind == KIND_ELLIPSOID) { t = ray_ellipsoid_local(lo, ld, data); }

        if (t > 0.0 && t < best_t) {
            best_t = t;
            best_shape = i32(i);
            let p_local = lo + ld * t;
            best_point  = transform_point(world_xform, p_local);
            let n_local = shape_normal(kind, data, p_local);
            best_normal = normalize(quat_rotate(world_xform.rotation, n_local));
        }
    }

    if (best_shape < 0) {
        hit_t[tid] = NO_HIT_T;
        hit_shape[tid] = -1;
        hit_point[tid] = vec4<f32>(0.0);
        hit_normal[tid] = vec4<f32>(0.0, 1.0, 0.0, 0.0);
    } else {
        hit_t[tid] = best_t;
        hit_shape[tid] = best_shape;
        hit_point[tid] = vec4<f32>(best_point, 0.0);
        hit_normal[tid] = vec4<f32>(best_normal, 0.0);
    }
}
