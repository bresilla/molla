struct Velocity { linear: vec4<f32>, angular: vec4<f32> }
struct Params { particles: u32, bodies: u32, final_iteration: u32, pad: u32,
    linear_tolerance: f32, angular_tolerance: f32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<storage, read_write> input_velocity: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> predicted_velocity: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> corrected_velocity: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> previous_body_velocity: array<Velocity>;
@group(0) @binding(4) var<storage, read> corrected_body_velocity: array<Velocity>;
@group(0) @binding(5) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(6) var<storage, read_write> residual: array<atomic<u32>>;
@group(0) @binding(7) var<uniform> params: Params;

fn monitor(delta: vec3<f32>, index: u32, tolerance: f32) {
    let value = length(delta);
    if !(value >= 0.0 && value <= 3.402823e38) { atomicOr(&status, 4u); return; }
    atomicMax(&residual[index], bitcast<u32>(value));
    if params.final_iteration != 0u && value > tolerance { atomicOr(&status, 64u); }
}

@compute @workgroup_size(64)
fn accumulate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.particles { return; }
    let delta = corrected_velocity[i].xyz - predicted_velocity[i].xyz;
    let value = input_velocity[i].xyz + delta;
    if !all(abs(value) <= vec3<f32>(3.402823e38)) { atomicOr(&status, 4u); return; }
    input_velocity[i] = vec4<f32>(value, 0.0);
    monitor(delta, 0u, params.linear_tolerance);
}

@compute @workgroup_size(64)
fn check_bodies(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.bodies { return; }
    monitor(corrected_body_velocity[i].linear.xyz - previous_body_velocity[i].linear.xyz, 1u, params.linear_tolerance);
    monitor(corrected_body_velocity[i].angular.xyz - previous_body_velocity[i].angular.xyz, 2u, params.angular_tolerance);
}
