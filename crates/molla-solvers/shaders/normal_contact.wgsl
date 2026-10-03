struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Spatial { linear: vec4<f32>, angular: vec4<f32> }
struct Shape { penalty: vec4<f32>, coefficients: vec4<f32>, surface: vec4<f32>, tags: vec4<u32> }
struct Body { com: vec4<f32>, tags: vec4<u32> }
struct Contact {
    point_depth: vec4<f32>, normal_force: vec4<f32>, coefficients: vec4<f32>,
    endpoints: vec4<u32>, relative_velocity: vec4<f32>, surface_velocity: vec4<f32>,
    force_on_b: vec4<f32>, penalty: vec4<f32>,
}
struct Params { capacity: u32, bodies: u32, shapes: u32, points: u32 }

@group(0) @binding(0) var<storage, read> pair_a: array<u32>;
@group(0) @binding(1) var<storage, read> pair_b: array<u32>;
@group(0) @binding(2) var<storage, read> pair_count: array<u32>;
@group(0) @binding(3) var<storage, read> manifold_counts: array<u32>;
@group(0) @binding(4) var<storage, read> manifold: array<vec4<f32>>;
@group(1) @binding(0) var<storage, read> shapes: array<Shape>;
@group(1) @binding(1) var<storage, read> bodies: array<Body>;
@group(1) @binding(2) var<storage, read> poses: array<Transform>;
@group(1) @binding(3) var<storage, read> velocities: array<Spatial>;
@group(1) @binding(4) var<storage, read> external_forces: array<Spatial>;
@group(2) @binding(0) var<storage, read_write> contacts: array<Contact>;
@group(2) @binding(1) var<storage, read_write> heads: array<atomic<u32>>;
@group(2) @binding(2) var<storage, read_write> links: array<u32>;
@group(2) @binding(3) var<storage, read_write> forces: array<Spatial>;
@group(3) @binding(0) var<uniform> u: Params;
@group(3) @binding(1) var<storage, read_write> status: array<atomic<u32>>;

const INPUT: u32 = 32u;
const NONFINITE: u32 = 64u;
const MATERIAL: u32 = 128u;
const NONE: u32 = 0xffffffffu;

fn finite(v: vec4<f32>) -> bool {
    return all(v == v) && all(abs(v) <= vec4<f32>(3.4028234e38));
}

fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

fn combine(a: vec2<f32>, b: vec2<f32>, rule: u32) -> vec2<f32> {
    if (rule == 1u) { return min(a, b); }
    if (rule == 2u) { return a * b; }
    if (rule == 3u) { return max(a, b); }
    return min(a, b) + 0.5 * (max(a, b) - min(a, b));
}

@compute @workgroup_size(64)
fn reset_heads(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x < max(u.bodies, 1u)) { atomicStore(&heads[gid.x], NONE); }
}

@compute @workgroup_size(64)
fn normal_contacts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= u.capacity * u.points) { return; }
    var row: Contact;
    contacts[slot] = row;
    if (atomicLoad(&status[0]) != 0u) { return; }
    if (pair_count[0] > u.capacity) { atomicOr(&status[0], INPUT); return; }
    let pair = slot / u.points;
    let point_index = slot % u.points;
    if (pair >= pair_count[0]) { return; }
    if (manifold_counts[pair] > u.points) { atomicOr(&status[0], INPUT); return; }
    if (point_index >= manifold_counts[pair]) { return; }
    let a = pair_a[pair];
    let b = pair_b[pair];
    if (a >= u.shapes || b >= u.shapes || a == b) { atomicOr(&status[0], INPUT); return; }
    let ma = shapes[a];
    let mb = shapes[b];
    let normal_depth = manifold[slot * 4u];
    let pa = manifold[slot * 4u + 1u];
    let pb = manifold[slot * 4u + 2u];
    if (!finite(normal_depth) || !finite(pa) || !finite(pb)) { atomicOr(&status[0], NONFINITE); return; }
    let normal_scale = max(abs(normal_depth.x), max(abs(normal_depth.y), abs(normal_depth.z)));
    if (bitcast<u32>(manifold[slot * 4u + 3u].x) != 1u || normal_scale == 0.0 || normal_depth.w < 0.0) {
        atomicOr(&status[0], INPUT); return;
    }
    let scaled_normal = normal_depth.xyz / normal_scale;
    let normal = scaled_normal / length(scaled_normal);
    let point = 0.5 * pa.xyz + 0.5 * pb.xyz;
    let rule = max(ma.tags.z, mb.tags.z);
    let mu = combine(ma.coefficients.xy, mb.coefficients.xy, rule);
    if (!finite(vec4<f32>(mu, 0.0, 0.0)) || mu.x < mu.y || mu.y < 0.0) { atomicOr(&status[0], MATERIAL); return; }
    for (var lane = 0u; lane < 2u; lane += 1u) {
        if ((mu[lane] > 0.0 && mu[lane] < 1.17549435e-38)
            || (rule == 2u && ma.coefficients[lane] > 0.0 && mb.coefficients[lane] > 0.0 && mu[lane] == 0.0)
            || (rule == 0u && max(ma.coefficients[lane], mb.coefficients[lane]) > 0.0 && mu[lane] == 0.0)) {
            atomicOr(&status[0], MATERIAL); return;
        }
    }
    row.point_depth = vec4<f32>(point, normal_depth.w);
    row.normal_force = vec4<f32>(normal, 0.0);
    row.coefficients = vec4<f32>(mu, max(ma.coefficients.z, mb.coefficients.z), 0.0);
    row.endpoints = vec4<u32>(a, b, ma.tags.x, mb.tags.x);
    row.penalty = min(ma.penalty, mb.penalty) + 0.5 * (max(ma.penalty, mb.penalty) - min(ma.penalty, mb.penalty));
    if (!finite(row.penalty)) { atomicOr(&status[0], MATERIAL); return; }
    for (var lane = 0u; lane < 4u; lane += 1u) {
        if ((row.penalty[lane] > 0.0 && row.penalty[lane] < 1.17549435e-38)
            || (max(ma.penalty[lane], mb.penalty[lane]) > 0.0 && row.penalty[lane] == 0.0)) {
            atomicOr(&status[0], MATERIAL); return;
        }
    }
    var speed: array<vec3<f32>, 2>;
    var belt = array<vec3<f32>, 2>(ma.surface.xyz, mb.surface.xyz);
    var dynamic: array<bool, 2>;
    for (var side = 0u; side < 2u; side += 1u) {
        let body = row.endpoints[side + 2u];
        if (body == NONE) { continue; }
        if (body >= u.bodies) { atomicOr(&status[0], INPUT); return; }
        let pose = poses[body];
        let velocity = velocities[body];
        if (!finite(pose.position) || !finite(pose.rotation) || !finite(velocity.linear) || !finite(velocity.angular)) {
            atomicOr(&status[0], NONFINITE); return;
        }
        let com = pose.position.xyz + rotate(pose.rotation, bodies[body].com.xyz);
        let arm = point - com;
        let flags = bodies[body].tags.x;
        if ((flags & 2u) == 0u) { speed[side] = velocity.linear.xyz + cross(velocity.angular.xyz, arm); }
        dynamic[side] = (flags & 3u) == 0u;
        belt[side] = rotate(pose.rotation, belt[side]);
    }
    let relative = speed[1] - speed[0];
    let vn = dot(relative, normal);
    row.relative_velocity = vec4<f32>(relative, vn);
    row.surface_velocity = vec4<f32>(belt[1] - belt[0], 0.0);
    let enabled = (ma.tags.y & 3u) == 1u && (mb.tags.y & 3u) == 1u;
    let acting = enabled && ma.tags.x != mb.tags.x && (dynamic[0] || dynamic[1]);
    if (acting) {
        row.coefficients.w = 1.0;
    }
    if (acting && normal_depth.w > 0.0) {
        let spring = row.penalty.x * normal_depth.w;
        let compression = row.penalty.y * max(-vn, 0.0);
        let magnitude = spring + compression;
        if ((row.penalty.x > 0.0 && spring == 0.0) || (row.penalty.y > 0.0 && vn < 0.0 && compression == 0.0)) {
            atomicOr(&status[0], NONFINITE); return;
        }
        row.normal_force.w = magnitude;
        row.force_on_b = vec4<f32>(normal * magnitude, 0.0);
    }
    if (!finite(row.normal_force) || !finite(row.force_on_b) || !finite(row.relative_velocity) || !finite(row.surface_velocity)) {
        atomicOr(&status[0], NONFINITE); return;
    }
    contacts[slot] = row;
    if (acting) {
        for (var side = 0u; side < 2u; side += 1u) {
            let body = row.endpoints[side + 2u];
            if (body == NONE) { continue; }
            let node = slot * 2u + side;
            links[node] = atomicExchange(&heads[body], node);
        }
    }
}

@compute @workgroup_size(64)
fn gather_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let body = gid.x;
    if (body >= max(u.bodies, 1u)) { return; }
    var total: Spatial;
    forces[body] = total;
    if (body >= u.bodies || atomicLoad(&status[0]) != 0u) { return; }
    total = external_forces[body];
    if ((bodies[body].tags.x & 3u) != 0u) {
        if (!finite(total.linear) || !finite(total.angular)) { atomicOr(&status[0], NONFINITE); return; }
        forces[body] = total;
        return;
    }
    let com = poses[body].position.xyz + rotate(poses[body].rotation, bodies[body].com.xyz);
    var node = atomicLoad(&heads[body]);
    var visited = 0u;
    while (node != NONE) {
        if (node >= u.capacity * u.points * 2u || visited >= u.capacity * u.points * 2u) {
            atomicOr(&status[0], INPUT); return;
        }
        let row = contacts[node / 2u];
        let force = row.force_on_b.xyz * select(-1.0, 1.0, (node & 1u) == 1u);
        total.linear = vec4<f32>(total.linear.xyz + force, 0.0);
        total.angular = vec4<f32>(total.angular.xyz + cross(row.point_depth.xyz - com, force), 0.0);
        node = links[node];
        visited += 1u;
    }
    if (!finite(total.linear) || !finite(total.angular)) { atomicOr(&status[0], NONFINITE); return; }
    forces[body] = total;
}
