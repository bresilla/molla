struct Params { count: u32, tolerance: f32, pad: vec2<u32> }
@group(0) @binding(0) var<storage, read> values: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(2) var<storage, read_write> maximum: atomic<u32>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(64)
fn check_force(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.count { return; }
    let value = values[gid.x].w;
    if !(value >= 0.0 && value <= 3.402823e38) {
        atomicOr(&status, 4u);
        return;
    }
    atomicMax(&maximum, bitcast<u32>(value));
    if value > params.tolerance { atomicOr(&status, 64u); }
}
