struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Spatial { linear: vec4<f32>, angular: vec4<f32> }
struct Contact {
    point_depth: vec4<f32>, normal_force: vec4<f32>, coefficients: vec4<f32>,
    endpoints: vec4<u32>, relative_velocity: vec4<f32>, surface_velocity: vec4<f32>,
    force_on_b: vec4<f32>, penalty: vec4<f32>,
}
@group(0) @binding(0) var<storage, read> contacts: array<Contact>;
@group(0) @binding(1) var<storage, read> heads: array<u32>;
@group(0) @binding(2) var<storage, read> links: array<u32>;
@group(0) @binding(3) var<storage, read> impulses: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> poses: array<Transform>;
@group(0) @binding(5) var<storage, read> centers: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> external_loads: array<Spatial>;
@group(0) @binding(7) var<storage, read> predicted_delta: array<f32>;
@group(0) @binding(8) var<storage, read> response_delta: array<f32>;
@group(0) @binding(9) var<storage, read_write> loads: array<Spatial>;
@group(0) @binding(10) var<storage, read_write> total_delta: array<f32>;
@group(0) @binding(11) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(12) var<storage, read> dt: f32;
@group(0) @binding(13) var<storage, read> params: vec4<u32>;

const NONE: u32 = 0xffffffffu;
const FAULT: u32 = 33554432u;
fn finite(v: vec4<f32>) -> bool {
    return all(v == v) && all(abs(v) <= vec4<f32>(3.4028234e38));
}
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}

@compute @workgroup_size(64)
fn assemble(@builtin(global_invocation_id) id: vec3<u32>) {
    if atomicLoad(&status) != 0u { return; }
    if !(dt > 0.0) || !finite(vec4<f32>(dt)) { atomicOr(&status, FAULT); return; }
    let body = id.x;
    if body < params.y {
        let change = predicted_delta[body] + response_delta[body];
        if !finite(vec4<f32>(change)) { atomicOr(&status, FAULT); return; }
        total_delta[body] = change;
    }
    if body >= params.x { return; }
    var load = external_loads[body];
    let center = poses[body].position.xyz + rotate(poses[body].rotation, centers[body].xyz);
    var node = heads[body];
    var visited = 0u;
    while node != NONE {
        if node >= params.z * 2u || visited >= params.z * 2u { atomicOr(&status, FAULT); return; }
        let row = node / 2u;
        let contact = contacts[row];
        if contact.endpoints[2u + (node & 1u)] != body { atomicOr(&status, FAULT); return; }
        let force = (contact.force_on_b.xyz + impulses[row].xyz / dt) * select(-1.0, 1.0, (node & 1u) == 1u);
        load.linear = vec4<f32>(load.linear.xyz + force, 0.0);
        load.angular = vec4<f32>(load.angular.xyz + cross(contact.point_depth.xyz - center, force), 0.0);
        node = links[node];
        visited += 1u;
    }
    if !finite(load.linear) || !finite(load.angular) { atomicOr(&status, FAULT); return; }
    loads[body] = load;
}
