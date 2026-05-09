// Multi-channel camera ray-trace kernel — one thread per pixel.
//
// Inputs:
//   * Per-pixel ray bundle from camera_rays.wgsl (`ray_origin`,
//     `ray_direction`, sized `cam × H × W`).
//   * Same shape arrays raycast.wgsl consumes: `shape_geom_kind`,
//     `shape_geom_data`, `shape_body`, `shape_transform`, `body_q`.
//   * Per-shape visual material: `shape_color`, `shape_metallic`,
//     `shape_roughness` (Vec3 / f32 / f32).
//   * Lighting + channel uniforms.
//
// Outputs (each conditionally written based on `channels` bitmask):
//   * `out_depth`       — `t` along the ray, < 0 = miss.
//   * `out_normal`      — outward surface normal (vec4<f32>).
//   * `out_albedo`      — RGBA8 packed material colour.
//   * `out_color`       — RGBA8 packed Lambertian-shaded colour.
//   * `out_shape_index` — hit shape id (`0xFFFFFFFF` = miss).
//
// Channel bitmask values mirror `CameraChannels` in Rust:
//   COLOR=1, DEPTH=2, NORMAL=4, ALBEDO=8, SHAPE_INDEX=16

const KIND_SPHERE:   u32 = 0u;
const KIND_PLANE:    u32 = 1u;
const KIND_BOX:      u32 = 2u;
const KIND_CAPSULE:  u32 = 3u;
const KIND_CYLINDER: u32 = 4u;
const KIND_CONE:     u32 = 5u;
const KIND_SKIP:     u32 = 0xffu;

const CH_COLOR:       u32 = 1u;
const CH_DEPTH:       u32 = 2u;
const CH_NORMAL:      u32 = 4u;
const CH_ALBEDO:      u32 = 8u;
const CH_SHAPE_INDEX: u32 = 16u;

const NO_HIT_T: f32 = -1.0;
const EPS: f32 = 1e-9;

struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,
};

struct Uniforms {
    width:        u32,
    height:       u32,
    camera_count: u32,
    shape_count:  u32,
    max_distance: f32,
    channels:     u32,
    _pad0:        f32,
    _pad1:        f32,
    ambient:      vec4<f32>,
    light_dir:    vec4<f32>,    // world-space, expected unit length
    light_color:  vec4<f32>,
};

@group(0) @binding(0) var<storage, read> shape_geom_kind:  array<u32>;
@group(0) @binding(1) var<storage, read> shape_geom_data:  array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> shape_body:       array<i32>;
@group(0) @binding(3) var<storage, read> shape_transform:  array<Transform>;
@group(0) @binding(4) var<storage, read> body_q:           array<Transform>;
@group(0) @binding(5) var<storage, read> shape_color:      array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> shape_metallic:   array<f32>;
@group(0) @binding(7) var<storage, read> shape_roughness:  array<f32>;

@group(1) @binding(0) var<storage, read> ray_origin:    array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> ray_direction: array<vec4<f32>>;

@group(2) @binding(0) var<storage, read_write> out_depth:       array<f32>;
@group(2) @binding(1) var<storage, read_write> out_color:       array<u32>;
@group(2) @binding(2) var<storage, read_write> out_normal:      array<vec4<f32>>;
@group(2) @binding(3) var<storage, read_write> out_albedo:      array<u32>;
@group(2) @binding(4) var<storage, read_write> out_shape_index: array<u32>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

// ── Quat / transform helpers ──
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}
fn quat_conj(q: vec4<f32>) -> vec4<f32> { return vec4<f32>(-q.x, -q.y, -q.z, q.w); }
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

// ── Per-shape ray intersection in shape-local frame. ──
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
    var tmin = -1.0e30;
    var tmax =  1.0e30;
    for (var k: u32 = 0u; k < 3u; k = k + 1u) {
        let oi = o[k]; let di = d[k]; let hi = h[k];
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
    var best = -1.0;
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
    for (var s = 0u; s < 2u; s = s + 1u) {
        var cap_y = half_h;
        if (s == 1u) { cap_y = -half_h; }
        let oc = vec3<f32>(o.x, o.y - cap_y, o.z);
        let t = ray_sphere_local(oc, d, r);
        if (t > 0.0 && (best < 0.0 || t < best)) { best = t; }
    }
    return best;
}
fn ray_cylinder_local(o: vec3<f32>, d: vec3<f32>, half_h: f32, r: f32) -> f32 {
    var best = -1.0;
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
    if (abs(d.y) >= EPS) {
        for (var s = 0u; s < 2u; s = s + 1u) {
            var cap_y = half_h;
            if (s == 1u) { cap_y = -half_h; }
            let t = (cap_y - o.y) / d.y;
            if (t > 0.0) {
                let x = o.x + d.x * t;
                let z = o.z + d.z * t;
                if (x * x + z * z <= r * r && (best < 0.0 || t < best)) { best = t; }
            }
        }
    }
    return best;
}

// Solid cone with axis +Y. Apex at (0, +half_h, 0); circular base
// of radius `r` at y = -half_h. Ray test = side surface (truncated
// double cone, then range-clamped) ∪ base disc.
fn ray_cone_local(o: vec3<f32>, d: vec3<f32>, half_h: f32, r: f32) -> f32 {
    var best = -1.0;

    // Side surface: cone equation in cone-local space (apex at origin
    // pointing -y in cone-local) — easier to shift the ray to the
    // apex frame: o' = o - (0, half_h, 0).
    // Equation: x² + z² = k² · (apex_y - y')² where apex_y = 0 and y'
    // is measured downward → y' negative inside cone.
    // k = r / (2·half_h) = base_radius / cone_height.
    let h_total = 2.0 * half_h;
    if (h_total > EPS && r > EPS) {
        let k = r / h_total;
        let k2 = k * k;
        // Shifted origin: o' = o - (0, half_h, 0). Solve
        //   (o'.x + d.x t)² + (o'.z + d.z t)² = k² · (-(o'.y + d.y t))²
        // ⇒ A t² + 2 B t + C = 0 with
        let ox = o.x;
        let oy = o.y - half_h;
        let oz = o.z;
        let dx = d.x;
        let dy = d.y;
        let dz = d.z;
        let a = dx * dx + dz * dz - k2 * dy * dy;
        let b = ox * dx + oz * dz - k2 * oy * dy;
        let c = ox * ox + oz * oz - k2 * oy * oy;
        // Two roots if a ≠ 0. Walk both, accept those with y in
        // [-half_h, +half_h] (i.e. between base and apex).
        if (abs(a) > 1e-18) {
            let disc = b * b - a * c;
            if (disc >= 0.0) {
                let sd = sqrt(disc);
                let inv = 1.0 / a;
                let t1 = (-b - sd) * inv;
                let t2 = (-b + sd) * inv;
                // Validate that the hit y is within the cone slab
                // AND that the ray is on the "below apex" side (apex_y
                // - y > 0). Since apex_y = half_h, that means
                // (half_h - y_hit) > 0, i.e. y_hit < half_h.
                if (t1 > 0.0) {
                    let y = o.y + dy * t1;
                    if (y >= -half_h && y <= half_h) { best = t1; }
                }
                if (t2 > 0.0 && (best < 0.0 || t2 < best)) {
                    let y = o.y + dy * t2;
                    if (y >= -half_h && y <= half_h) { best = t2; }
                }
            }
        }
    }

    // Base disc at y = -half_h.
    if (abs(d.y) >= EPS) {
        let t = (-half_h - o.y) / d.y;
        if (t > 0.0) {
            let x = o.x + d.x * t;
            let z = o.z + d.z * t;
            if (x * x + z * z <= r * r && (best < 0.0 || t < best)) { best = t; }
        }
    }
    return best;
}

// ── Outward normal at a shape-local hit point. ──
fn shape_normal(kind: u32, data: vec3<f32>, p_local: vec3<f32>) -> vec3<f32> {
    if (kind == KIND_SPHERE) {
        return normalize(p_local);
    } else if (kind == KIND_PLANE) {
        return data;
    } else if (kind == KIND_BOX) {
        let half = data;
        let nx = abs(p_local.x / max(half.x, EPS));
        let ny = abs(p_local.y / max(half.y, EPS));
        let nz = abs(p_local.z / max(half.z, EPS));
        if (nx >= ny && nx >= nz) { return vec3<f32>(sign(p_local.x), 0.0, 0.0); }
        else if (ny >= nz)        { return vec3<f32>(0.0, sign(p_local.y), 0.0); }
        else                       { return vec3<f32>(0.0, 0.0, sign(p_local.z)); }
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
        if (abs(p_local.y - half_h) < 1e-4) { return vec3<f32>(0.0, 1.0, 0.0); }
        if (abs(p_local.y + half_h) < 1e-4) { return vec3<f32>(0.0, -1.0, 0.0); }
        return normalize(vec3<f32>(p_local.x, 0.0, p_local.z));
    } else if (kind == KIND_CONE) {
        let half_h = data.x;
        let r = data.y;
        // Base disc?
        if (abs(p_local.y + half_h) < 1e-4) {
            return vec3<f32>(0.0, -1.0, 0.0);
        }
        // Side surface: gradient of (x² + z² − k²·(half_h - y)²) = 0
        // is (2x, 2 k² (half_h - y), 2z); flip sign so it points OUT.
        let h_total = 2.0 * half_h;
        let k = r / max(h_total, EPS);
        let k2 = k * k;
        let n = vec3<f32>(p_local.x, k2 * (half_h - p_local.y), p_local.z);
        let l2 = dot(n, n);
        if (l2 > 1e-18) { return n / sqrt(l2); }
        return vec3<f32>(0.0, 1.0, 0.0);
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}

// ── RGBA8 packing helper (linear-space, no gamma). ──
fn pack_rgba8(c: vec3<f32>) -> u32 {
    let r = u32(clamp(c.r, 0.0, 1.0) * 255.0 + 0.5);
    let g = u32(clamp(c.g, 0.0, 1.0) * 255.0 + 0.5);
    let b = u32(clamp(c.b, 0.0, 1.0) * 255.0 + 0.5);
    return (255u << 24u) | (b << 16u) | (g << 8u) | r;
}

@compute @workgroup_size(8, 8, 1)
fn trace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let px = gid.x;
    let py = gid.y;
    let cam = gid.z;
    if (px >= uniforms.width || py >= uniforms.height || cam >= uniforms.camera_count) {
        return;
    }

    let pixels_per_cam = uniforms.width * uniforms.height;
    let idx = cam * pixels_per_cam + py * uniforms.width + px;

    let ro = ray_origin[idx].xyz;
    let rd_raw = ray_direction[idx].xyz;
    let rd_len = length(rd_raw);
    let channels = uniforms.channels;

    if (rd_len < 1e-12) {
        if ((channels & CH_DEPTH) != 0u)       { out_depth[idx] = NO_HIT_T; }
        if ((channels & CH_SHAPE_INDEX) != 0u) { out_shape_index[idx] = 0xffffffffu; }
        if ((channels & CH_NORMAL) != 0u)      { out_normal[idx] = vec4<f32>(0.0); }
        if ((channels & CH_ALBEDO) != 0u)      { out_albedo[idx] = 0u; }
        if ((channels & CH_COLOR) != 0u)       { out_color[idx] = 0u; }
        return;
    }
    let rd = rd_raw / rd_len;

    var best_t: f32 = uniforms.max_distance;
    var best_shape: i32 = -1;
    var best_point_world: vec3<f32> = vec3<f32>(0.0);
    var best_normal_world: vec3<f32> = vec3<f32>(0.0, 1.0, 0.0);

    for (var i: u32 = 0u; i < uniforms.shape_count; i = i + 1u) {
        let kind = shape_geom_kind[i];
        if (kind == KIND_SKIP) { continue; }

        let body = shape_body[i];
        var world_xform: Transform;
        if (body >= 0) {
            world_xform = transform_compose(body_q[u32(body)], shape_transform[i]);
        } else {
            world_xform = shape_transform[i];
        }

        let inv = transform_inverse(world_xform);
        let lo = transform_point(inv, ro);
        let ld = quat_rotate(inv.rotation, rd);

        let data = shape_geom_data[i].xyz;
        var t: f32 = NO_HIT_T;
        if      (kind == KIND_SPHERE)   { t = ray_sphere_local(lo, ld, data.x); }
        else if (kind == KIND_PLANE)    { t = ray_plane_local(lo, ld, data); }
        else if (kind == KIND_BOX)      { t = ray_box_local(lo, ld, data); }
        else if (kind == KIND_CAPSULE)  { t = ray_capsule_local(lo, ld, data.x, data.y); }
        else if (kind == KIND_CYLINDER) { t = ray_cylinder_local(lo, ld, data.x, data.y); }
        else if (kind == KIND_CONE)     { t = ray_cone_local(lo, ld, data.x, data.y); }

        if (t > 0.0 && t < best_t) {
            best_t = t;
            best_shape = i32(i);
            let p_local = lo + ld * t;
            best_point_world  = transform_point(world_xform, p_local);
            let n_local = shape_normal(kind, data, p_local);
            best_normal_world = normalize(quat_rotate(world_xform.rotation, n_local));
        }
    }

    if (best_shape < 0) {
        if ((channels & CH_DEPTH) != 0u)       { out_depth[idx] = NO_HIT_T; }
        if ((channels & CH_SHAPE_INDEX) != 0u) { out_shape_index[idx] = 0xffffffffu; }
        if ((channels & CH_NORMAL) != 0u)      { out_normal[idx] = vec4<f32>(0.0); }
        if ((channels & CH_ALBEDO) != 0u)      { out_albedo[idx] = 0u; }
        if ((channels & CH_COLOR) != 0u)       { out_color[idx] = 0u; }
        return;
    }

    if ((channels & CH_DEPTH) != 0u)       { out_depth[idx] = best_t; }
    if ((channels & CH_SHAPE_INDEX) != 0u) { out_shape_index[idx] = u32(best_shape); }
    if ((channels & CH_NORMAL) != 0u) {
        out_normal[idx] = vec4<f32>(best_normal_world, 0.0);
    }

    let albedo = shape_color[u32(best_shape)].xyz;
    if ((channels & CH_ALBEDO) != 0u) {
        out_albedo[idx] = pack_rgba8(albedo);
    }

    if ((channels & CH_COLOR) != 0u) {
        // Lambertian shading: albedo * (ambient + light · max(0, n · -light_dir)).
        // Light shines "from light_dir" → surface gets light from -light_dir.
        let l = -uniforms.light_dir.xyz;
        let ndotl = max(0.0, dot(best_normal_world, l));
        let lit = uniforms.ambient.xyz + uniforms.light_color.xyz * ndotl;
        let shaded = albedo * lit;
        out_color[idx] = pack_rgba8(shaded);
    }
}
