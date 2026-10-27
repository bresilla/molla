struct Wrench { linear: vec4<f32>, angular: vec4<f32> }
struct Impulse { linear: vec4<f32>, angular: vec4<f32>, world_origin_angular: vec4<f32> }
struct Params { dt: f32, count: u32, pad: vec2<u32> }
@group(0) @binding(0) var<storage, read> reactions: array<Wrench>;
@group(0) @binding(1) var<storage, read_write> impulses: array<Impulse>;
@group(0) @binding(2) var<uniform> params: Params;
@group(0) @binding(3) var<storage, read> centers: array<vec4<f32>>;

@compute @workgroup_size(64)
fn integrate(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot = id.x;
    if slot >= params.count { return; }
    impulses[slot].linear += vec4<f32>(reactions[slot].linear.xyz * params.dt, 0.0);
    impulses[slot].angular += vec4<f32>(reactions[slot].angular.xyz * params.dt, 0.0);
    let torque = reactions[slot].angular.xyz + cross(centers[slot].xyz, reactions[slot].linear.xyz);
    impulses[slot].world_origin_angular += vec4<f32>(torque * params.dt, 0.0);
}
