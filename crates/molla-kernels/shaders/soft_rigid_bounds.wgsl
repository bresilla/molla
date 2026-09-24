struct Bound { indices: vec4<u32>, limits: vec4<f32> }
struct Contact {
    normal_gap: vec4<f32>, material: vec4<f32>, ids: vec4<u32>,
    support: vec4<u32>, weights: vec4<f32>,
}
struct Params { dt: f32, base: u32, count: u32, pad: u32 }
@group(0) @binding(0) var<storage, read> bounds: array<Bound>;
@group(0) @binding(1) var<storage, read> coordinates: array<f32>;
@group(0) @binding(2) var<storage, read_write> contacts: array<Contact>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(64)
fn update_bounds(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count || atomicLoad(&status) != 0u { return; }
    let bound = bounds[i];
    let q = coordinates[bound.indices.y];
    if !(abs(q) <= 3.402823e38) { atomicOr(&status, 32u); return; }
    var lo = -bound.limits.z;
    var hi = bound.limits.z;
    if bound.limits.x > -1e30 { lo = max(lo, (bound.limits.x-q)/params.dt); }
    if bound.limits.y < 1e30 { hi = min(hi, (bound.limits.y-q)/params.dt); }
    if !(lo <= hi && abs(lo) <= 3.402823e38 && abs(hi) <= 3.402823e38) {
        atomicOr(&status, 32u); return;
    }
    let row = params.base+2u*i;
    contacts[row].material.y = lo;
    contacts[row+1u].material.y = -hi;
}
