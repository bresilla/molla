struct Transform { position: vec4<f32>, rotation: vec4<f32> }
struct Instance { lo: vec4<f32>, hi: vec4<f32>, pose: Transform, ids: vec4<i32> }
struct Params { count: u32, root: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<storage, read> lower: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> upper: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> instances: array<Instance>;
@group(0) @binding(3) var<storage, read_write> scattered: array<Instance>;
@group(0) @binding(4) var<storage, read_write> topology: array<vec4<u32>>;
@group(0) @binding(5) var<storage, read_write> keys: array<u32>;
@group(0) @binding(6) var<storage, read_write> values: array<u32>;
@group(0) @binding(7) var<storage, read> leaf_slots: array<u32>;
@group(1) @binding(0) var<uniform> params: Params;

fn spread(value: u32) -> u32 {
    var v = value & 1023u;
    v = (v | (v << 16u)) & 0x030000ffu;
    v = (v | (v << 8u)) & 0x0300f00fu;
    v = (v | (v << 4u)) & 0x030c30c3u;
    return (v | (v << 2u)) & 0x09249249u;
}

@compute @workgroup_size(64)
fn morton(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.count { return; }
    let center = lower[i].xyz*0.5 + upper[i].xyz*0.5;
    let lo = lower[params.root].xyz*0.5;
    let span = upper[params.root].xyz*0.5 - lo;
    var cell = vec3<u32>(0u);
    for (var axis = 0u; axis < 3u; axis++) {
        if span[axis] > 1e-30 {
            cell[axis] = u32(clamp((center[axis]*0.5-lo[axis])/span[axis], 0.0, 1.0)*1023.0);
        }
    }
    keys[i] = spread(cell.x) | (spread(cell.y)<<1u) | (spread(cell.z)<<2u);
    values[i] = i;
}

@compute @workgroup_size(64)
fn reorder(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rank = gid.x;
    if rank >= params.count { return; }
    let slot = leaf_slots[rank];
    let instance = instances[values[rank]];
    scattered[slot] = instance;
    topology[slot].z = bitcast<u32>(instance.ids.x);
}
