struct Params { count: u32, dt: f32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> velocities: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> output: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn drift(@builtin(global_invocation_id) gid: vec3<u32>) {
    let node = gid.x;
    if node >= params.count || atomicLoad(&status) != 0u { return; }
    let q = positions[node].xyz + params.dt * velocities[node].xyz;
    if !all(abs(q) <= vec3<f32>(3.402823e38)) { atomicOr(&status, 4u); return; }
    output[node] = vec4<f32>(q, 0.0);
}
