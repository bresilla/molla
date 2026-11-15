struct Uniforms {
    vertices: u32, triangles: u32, edges: u32, springs: u32,
    tets: u32, obstacles: u32, contacts: u32, final_state: u32,
    h: f32, strain_limit: f32, pad0: u32, pad1: u32,
}
struct Aabb { lo: vec4<f32>, hi: vec4<f32> }
struct Node {
    lo: vec4<f32>, hi: vec4<f32>, left: u32, right: u32, parent: u32, leaf: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> previous: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> velocity: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> predicted: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> gradient: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> triangles: array<vec4<u32>>;
@group(0) @binding(7) var<storage, read> edges: array<vec4<u32>>;
@group(0) @binding(8) var<storage, read> pinned: array<u32>;
@group(0) @binding(9) var<storage, read> obstacles: array<vec4<f32>>;
@group(0) @binding(10) var<storage, read> spring_a: array<u32>;
@group(0) @binding(11) var<storage, read> spring_b: array<u32>;
@group(0) @binding(12) var<storage, read> rest: array<f32>;
@group(0) @binding(13) var<storage, read> tets: array<vec4<u32>>;
@group(0) @binding(14) var<storage, read> tet_rest: array<vec2<f32>>;
@group(0) @binding(15) var<storage, read_write> bounds: array<Aabb>;
@group(0) @binding(16) var<storage, read> nodes: array<Node>;
@group(0) @binding(17) var<storage, read> sorted_ids: array<u32>;
@group(0) @binding(18) var<storage, read> tree_info: array<u32>;
@group(0) @binding(19) var<storage, read_write> flags: array<atomic<u32>>;

const NUMERIC: u32 = 1u;
const OBSTACLE: u32 = 2u;
const STRAIN: u32 = 4u;
const DEGENERATE: u32 = 8u;
const INTERSECTION: u32 = 16u;
const TRAVERSAL: u32 = 32u;
const INVERTED: u32 = 64u;
const ROUNDING: f32 = 9.5367431640625e-7;

fn finite(v: f32) -> bool { return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u; }
fn finite3(v: vec3<f32>) -> bool {
    return all((bitcast<vec3<u32>>(v) & vec3<u32>(0x7f800000u)) != vec3<u32>(0x7f800000u));
}
fn maxabs(v: vec3<f32>) -> f32 { return max(abs(v.x), max(abs(v.y), abs(v.z))); }
fn uncertainty(a: vec3<f32>, b: vec3<f32>) -> f32 {
    return ROUNDING * max(max(maxabs(a), maxabs(b)), 1.0e-12);
}
fn report(mask: u32) { atomicOr(&flags[0], mask); }

@compute @workgroup_size(64)
fn validate_primitives(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i < u.vertices {
        let p = x[i].xyz;
        if !finite3(p) || !finite3(previous[i].xyz) || !finite3(predicted[i].xyz) || !finite3(velocity[i].xyz) {
            report(NUMERIC);
        }
        if u.final_state != 0u {
            if !finite3(gradient[i].xyz) { report(NUMERIC); }
            if pinned[i] == 0u && !finite3((p - previous[i].xyz) / u.h) { report(NUMERIC); }
        }
        if u.contacts != 0u && pinned[i] == 0u {
            for (var o = 0u; o < u.obstacles; o++) {
                let shape = obstacles[2u * o];
                let parameter = obstacles[2u * o + 1u].x;
                var gap = dot(shape.yzw, p) - parameter;
                if shape.x == 1.0 { gap = length(p - shape.yzw) - parameter; }
                if !finite(gap) { report(NUMERIC); }
                else if gap <= 0.0 { report(OBSTACLE); }
            }
        }
    }
    if i < u.springs && u.strain_limit > 0.0 {
        let ratio = length(x[spring_a[i]].xyz - x[spring_b[i]].xyz) / rest[i];
        if !finite(ratio) { report(NUMERIC); }
        else if ratio >= u.strain_limit { report(STRAIN); }
    }
    if i < u.tets {
        let t = tets[i];
        let a = x[t.x].xyz;
        let v = dot(cross(x[t.y].xyz - a, x[t.z].xyz - a), x[t.w].xyz - a) / 6.0;
        if !finite(v) { report(NUMERIC); }
        else if v == 0.0 { report(DEGENERATE); }
        else if (v < 0.0) != (tet_rest[i].y < 0.0) { report(INVERTED); }
    }
    if i < u.triangles {
        let t = triangles[i];
        let a = x[t.x].xyz;
        let b = x[t.y].xyz;
        let c = x[t.z].xyz;
        var lo = min(a, min(b, c));
        var hi = max(a, max(b, c));
        let area2 = dot(cross(b - a, c - a), cross(b - a, c - a));
        let eps = uncertainty(lo, hi);
        lo -= vec3<f32>(eps);
        hi += vec3<f32>(eps);
        if !finite3(lo) || !finite3(hi) || !finite(area2) {
            report(NUMERIC);
            lo = vec3<f32>(0.0);
            hi = vec3<f32>(0.0);
        } else if area2 <= 0.0 { report(DEGENERATE); }
        bounds[i] = Aabb(vec4<f32>(lo, 0.0), vec4<f32>(hi, 0.0));
    }
}

fn project(p: vec3<f32>, axis: u32) -> vec2<f32> {
    if axis == 0u { return p.yz; }
    if axis == 1u { return p.xz; }
    return p.xy;
}
fn cross2(a: vec2<f32>, b: vec2<f32>) -> f32 { return a.x * b.y - a.y * b.x; }

fn clipped_interval(p: vec2<f32>, q: vec2<f32>, b: vec2<f32>, c: vec2<f32>, eps: f32) -> vec2<f32> {
    let corners = array<vec2<f32>, 3>(vec2<f32>(0.0), b, c);
    let orientation = select(-1.0, 1.0, cross2(b, c) > 0.0);
    var interval = vec2<f32>(0.0, 1.0);
    for (var k = 0u; k < 3u; k++) {
        let a = corners[k];
        let e = corners[(k + 1u) % 3u] - a;
        let tolerance = eps * length(e);
        let f0 = orientation * cross2(e, p - a) + tolerance;
        let f1 = orientation * cross2(e, q - a) + tolerance;
        if f0 < 0.0 && f1 < 0.0 { return vec2<f32>(1.0, -1.0); }
        if f0 < 0.0 { interval.x = max(interval.x, f0 / (f0 - f1)); }
        if f1 < 0.0 { interval.y = min(interval.y, f0 / (f0 - f1)); }
    }
    return interval;
}

fn intersects(ids: vec2<u32>, tri: vec3<u32>) -> bool {
    let shared_a = any(tri == vec3<u32>(ids.x));
    let shared_b = any(tri == vec3<u32>(ids.y));
    if shared_a && shared_b { return false; }
    let a = x[tri.x].xyz;
    let p_world = x[ids.x].xyz;
    let q_world = x[ids.y].xyz;
    var b = x[tri.y].xyz - a;
    var c = x[tri.z].xyz - a;
    var p = p_world - a;
    var q = q_world - a;
    let scale = max(max(maxabs(b), maxabs(c)), max(maxabs(p), maxabs(q)));
    if !finite(scale) || !finite3(b) || !finite3(c) || !finite3(p) || !finite3(q) || scale == 0.0 {
        report(NUMERIC);
        return true;
    }
    b /= scale;
    c /= scale;
    p /= scale;
    q /= scale;
    let normal = cross(b, c);
    let n_abs = abs(normal);
    if maxabs(normal) < 1.0e-20 {
        report(DEGENERATE);
        return true;
    }
    var axis = 2u;
    if n_abs.x >= n_abs.y && n_abs.x >= n_abs.z { axis = 0u; }
    else if n_abs.y >= n_abs.z { axis = 1u; }
    let eps = max(ROUNDING, max(uncertainty(p_world, q_world), uncertainty(a, a + b * scale)) / scale);
    let plane_eps = eps * (n_abs.x + n_abs.y + n_abs.z);
    let d0 = dot(normal, p);
    let d1 = dot(normal, q);
    if (d0 > plane_eps && d1 > plane_eps) || (d0 < -plane_eps && d1 < -plane_eps) { return false; }
    let endpoint_eps = min(0.25, 4.0 * eps / max(length(q - p), eps));
    let b2 = project(b, axis);
    let c2 = project(c, axis);
    if abs(d0) <= plane_eps && abs(d1) <= plane_eps {
        let interval = clipped_interval(project(p, axis), project(q, axis), b2, c2, eps);
        if interval.x > interval.y { return false; }
        if shared_a && interval.y <= endpoint_eps { return false; }
        if shared_b && interval.x >= 1.0 - endpoint_eps { return false; }
        return true;
    }
    let t = d0 / (d0 - d1);
    if !finite(t) { report(NUMERIC); return true; }
    if t < 0.0 || t > 1.0 { return false; }
    if (shared_a && t <= endpoint_eps) || (shared_b && t >= 1.0 - endpoint_eps) { return false; }
    let point = project(p + t * (q - p), axis);
    let interval = clipped_interval(point, point, b2, c2, eps);
    return interval.x <= interval.y;
}

@compute @workgroup_size(64)
fn validate_intersections(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= u.vertices + u.edges || u.triangles == 0u { return; }
    if atomicLoad(&flags[0]) != 0u { return; }
    var ids: vec2<u32>;
    if i < u.edges { ids = edges[i].xy; }
    else { ids = vec2<u32>(i - u.edges); }
    let p = x[ids.x].xyz;
    let q = x[ids.y].xyz;
    let eps = uncertainty(p, q);
    let lo = min(p, q) - vec3<f32>(eps);
    let hi = max(p, q) + vec3<f32>(eps);
    var stack: array<u32, 64>;
    stack[0] = tree_info[1];
    var size = 1u;
    var visited = 0u;
    loop {
        if size == 0u { break; }
        if visited >= 2u * u.triangles { report(TRAVERSAL); return; }
        visited++;
        size--;
        let index = stack[size];
        if index >= 2u * u.triangles - 1u { report(TRAVERSAL); return; }
        let node = nodes[index];
        if any(lo > node.hi.xyz) || any(hi < node.lo.xyz) { continue; }
        if index < u.triangles {
            let tri = sorted_ids[index];
            if tri >= u.triangles { report(TRAVERSAL); return; }
            if intersects(ids, triangles[tri].xyz) { report(INTERSECTION); return; }
        } else {
            if size + 2u > 64u { report(TRAVERSAL); return; }
            stack[size] = node.left;
            stack[size + 1u] = node.right;
            size += 2u;
        }
    }
}
