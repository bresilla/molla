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

struct Transform {
    position: vec4<f32>,  // xyz used, w padded
    rotation: vec4<f32>,  // (x, y, z, w) quaternion
};

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
        let axis = select(1.0, -1.0, s == 1u);
        let cap_y = axis * half_h;
        let oc = vec3<f32>(o.x, o.y - cap_y, o.z);
        let sphere_a = dot(d, d);
        let b = dot(oc, d);
        let c = dot(oc, oc) - r * r;
        let disc = b * b - sphere_a * c;
        if (disc < 0.0) { continue; }
        let roots = vec2<f32>(-b - sqrt(disc), -b + sqrt(disc)) / sphere_a;
        for (var root = 0u; root < 2u; root++) {
            let t = roots[root];
            let hemisphere_y = axis * (oc.y + d.y * t);
            if (t > 0.0 && hemisphere_y >= 0.0 && (best < 0.0 || t < best)) {
                best = t;
            }
        }
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
                if (x * x + z * z <= r * r && (best < 0.0 || t < best)) {
                    best = t;
                }
            }
        }
    }
    return best;
}

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
        return normalize(vec3<f32>(
            p_local.x / (data.x * data.x),
            p_local.y / (data.y * data.y),
            p_local.z / (data.z * data.z),
        ));
    }
    return vec3<f32>(0.0, 1.0, 0.0);
}
