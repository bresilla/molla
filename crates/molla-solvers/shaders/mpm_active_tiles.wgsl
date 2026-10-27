struct Params {
    dimensions: vec4<u32>,
    origin_inverse_dx: vec4<f32>,
    counts: vec4<u32>,
    grid_window_cells: vec4<i32>,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> positions: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> spheres: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> tiles: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> dispatch: array<u32>;
override SPARSE_MAP: bool = false;

fn tile_hash(id: u32) -> u32 {
    var value = (id ^ (id >> 16u)) * 0x7feb352du;
    value = (value ^ (value >> 15u)) * 0x846ca68bu;
    return value ^ (value >> 16u);
}

fn mark_sparse_tile(id: u32) {
    let base = params.counts.z + 4u;
    let width = params.counts.w;
    let mask = width - 1u;
    var bucket = tile_hash(id) & mask;
    for (var probe = 0u; probe < width; probe++) {
        let key = atomicLoad(&tiles[base + bucket]);
        if key == id + 1u { return; }
        if key == 0u {
            loop {
                let claim = atomicCompareExchangeWeak(&tiles[base + bucket], 0u, id + 1u);
                if claim.exchanged {
                    let slot = atomicAdd(&tiles[params.counts.z], 1u);
                    if slot >= params.dimensions.z {
                        atomicOr(&tiles[params.counts.z + 2u], 1u);
                        return;
                    }
                    atomicStore(&tiles[base + width + bucket], slot + 1u);
                    atomicStore(&tiles[base + 2u * width + slot], id);
                    atomicStore(&tiles[base + 2u * width + params.dimensions.z + slot], bucket);
                    return;
                }
                if claim.old_value == id + 1u { return; }
                if claim.old_value != 0u { break; }
            }
        }
        bucket = (bucket + 1u) & mask;
    }
    atomicOr(&tiles[params.counts.z + 2u], 1u);
}

fn mark_tile(coordinate: vec3<u32>) {
    if atomicLoad(&tiles[params.counts.z + 2u]) != 0u { return; }
    let width = params.dimensions.y;
    let capacity = params.dimensions.z;
    let id = coordinate.x + width * (coordinate.y + width * coordinate.z);
    if SPARSE_MAP {
        mark_sparse_tile(id);
        return;
    }
    let base = params.counts.z;
    if atomicLoad(&tiles[base + 4u + id]) != 0u { return; }
    loop {
        let claim = atomicCompareExchangeWeak(&tiles[base + 4u + id], 0u, 0xffffffffu);
        if !claim.exchanged {
            if claim.old_value != 0u { return; }
            continue;
        }
        let slot = atomicAdd(&tiles[base], 1u);
        if slot >= capacity {
            atomicOr(&tiles[base + 2u], 1u);
            return;
        }
        atomicStore(&tiles[base + 4u + params.counts.w + slot], id);
        atomicStore(&tiles[base + 4u + id], slot + 1u);
        return;
    }
}

@compute @workgroup_size(64)
fn mark_particles(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x >= params.counts.x { return; }
    let xp = (positions[gid.x].xyz - params.origin_inverse_dx.xyz) * params.origin_inverse_dx.w;
    if any((bitcast<vec3<u32>>(xp) & vec3<u32>(0x7f800000u)) == vec3<u32>(0x7f800000u)) {
        atomicOr(&tiles[params.counts.z + 2u], 2u);
        return;
    }
    let base = vec3<i32>(floor(xp - vec3<f32>(0.5))) - params.grid_window_cells.xyz;
    let last = vec3<i32>(i32(params.dimensions.x) - 1);
    let lo = max(base - vec3<i32>(1), vec3<i32>(0));
    let hi = min(base + vec3<i32>(3), last);
    if any(lo > hi) { return; }
    for (var z = u32(lo.z) / 4u; z <= u32(hi.z) / 4u; z++) {
        for (var y = u32(lo.y) / 4u; y <= u32(hi.y) / 4u; y++) {
            for (var x = u32(lo.x) / 4u; x <= u32(hi.x) / 4u; x++) {
                mark_tile(vec3<u32>(x, y, z));
            }
        }
    }
}

@compute @workgroup_size(64)
fn mark_rigids(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    if wid.x >= params.counts.y { return; }
    let sphere = spheres[wid.x];
    let center = (sphere.xyz - params.origin_inverse_dx.xyz) * params.origin_inverse_dx.w;
    let extent = max(sphere.w, 0.0) * params.origin_inverse_dx.w + 2.0;
    let lo_cell = max(vec3<i32>(floor(center - vec3<f32>(extent))) - params.grid_window_cells.xyz, vec3<i32>(0));
    let hi_cell = min(vec3<i32>(ceil(center + vec3<f32>(extent))) - params.grid_window_cells.xyz, vec3<i32>(i32(params.dimensions.x) - 1));
    if any(lo_cell > hi_cell) { return; }
    let lo = vec3<u32>(lo_cell) / vec3<u32>(4u);
    let hi = vec3<u32>(hi_cell) / vec3<u32>(4u);
    let size = hi - lo + vec3<u32>(1u);
    let count = size.x * size.y * size.z;
    for (var index = lid.x; index < count; index += 64u) {
        if atomicLoad(&tiles[params.counts.z + 2u]) != 0u { return; }
        let offset = vec3<u32>(index % size.x, (index / size.x) % size.y, index / (size.x * size.y));
        mark_tile(lo + offset);
    }
}

@compute @workgroup_size(1)
fn finalize_tiles() {
    var count = min(atomicLoad(&tiles[params.counts.z]), params.dimensions.z);
    if atomicLoad(&tiles[params.counts.z + 2u]) != 0u { count = 0u; }
    let width = min(count, params.dimensions.w);
    atomicStore(&tiles[params.counts.z + 1u], width);
    dispatch[0] = width;
    dispatch[1] = (count + max(width, 1u) - 1u) / max(width, 1u);
    dispatch[2] = 1u;
}
