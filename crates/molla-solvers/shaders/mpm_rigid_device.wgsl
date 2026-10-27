struct Binding { index: vec4<u32>, geometry: vec4<f32>, com_radius: vec4<f32> }
struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Spatial { linear: vec4<f32>, angular: vec4<f32> }
@group(0) @binding(0) var<storage, read> bindings: array<Binding>;
@group(0) @binding(1) var<storage, read> poses: array<Transform>;
@group(0) @binding(2) var<storage, read> velocities: array<Spatial>;
@group(0) @binding(3) var<storage, read_write> centers: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> motion: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> reactions: array<Spatial>;
@group(0) @binding(6) var<storage, read_write> forces: array<Spatial>;
@group(0) @binding(7) var<uniform> counts: vec4<u32>;
fn rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    return v + 2.0 * cross(q.xyz, cross(q.xyz, v) + q.w * v);
}
@compute @workgroup_size(64)
fn kinematics(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x;
    if slot >= counts.x { return; }
    let info = bindings[slot];
    let pose = poses[info.index.x];
    let velocity = velocities[info.index.x];
    centers[slot] = vec4<f32>(pose.position.xyz + rotate(pose.rotation, info.com_radius.xyz), info.com_radius.w);
    motion[4u*slot] = velocity.linear;
    motion[4u*slot+1u] = velocity.angular;
    motion[4u*slot+2u] = pose.rotation;
    motion[4u*slot+3u] = info.geometry;
}
@compute @workgroup_size(64)
fn apply_reactions(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x;
    if slot >= counts.x { return; }
    let body = bindings[slot].index.x;
    let reaction = reactions[slot];
    forces[body].linear = vec4<f32>(forces[body].linear.xyz + reaction.linear.xyz, 0.0);
    forces[body].angular = vec4<f32>(forces[body].angular.xyz + reaction.angular.xyz, 0.0);
}
