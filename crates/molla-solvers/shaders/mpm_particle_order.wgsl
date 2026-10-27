struct Params {
    counts: vec4<u32>,
    origin: vec4<f32>,
    grid_window_cells: vec4<i32>,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> keys: array<u32>;
@group(0) @binding(3) var<storage, read_write> identities: array<u32>;

@compute @workgroup_size(64)
fn tile_keys(@builtin(global_invocation_id) gid: vec3<u32>) {
    let id = gid.x;
    if id >= params.counts.y { return; }
    identities[id] = id;
    if id >= params.counts.x {
        keys[id] = 0xffffffffu;
        return;
    }
    let cell = vec3<i32>(floor((positions[id].xyz - params.origin.xyz) * params.origin.w)) - params.grid_window_cells.xyz;
    let tile = vec3<u32>(clamp(cell, vec3<i32>(0), vec3<i32>(i32(params.counts.z - 1u)))) / 4u;
    let width = params.counts.w;
    keys[id] = tile.x + width * (tile.y + width * tile.z);
}
