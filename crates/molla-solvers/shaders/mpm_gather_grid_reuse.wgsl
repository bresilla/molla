struct GatherReuseKey { words: array<u32, 28> }
struct GatherReuseTile { dirty: atomic<u32>, valid: atomic<u32> }
@group(1) @binding(12) var<storage, read_write> reuse_keys: array<GatherReuseKey>;
@group(1) @binding(13) var<storage, read_write> reuse_nodes: array<u32>;
@group(1) @binding(14) var<storage, read_write> reuse_tiles: array<GatherReuseTile>;
@group(1) @binding(15) var<storage, read_write> reuse_header: array<atomic<u32>>;
var<workgroup> reuse_tile_hit: u32;

fn reuse_signature() -> array<u32, 9> {
    return array<u32, 9>(u.grid_size, bitcast<u32>(u.grid_origin.x), bitcast<u32>(u.grid_origin.y),
        bitcast<u32>(u.grid_origin.z), bitcast<u32>(u.cell_size), bitcast<u32>(u.inv_cell_size),
        bitcast<u32>(u.grid_window_cells.x), bitcast<u32>(u.grid_window_cells.y), bitcast<u32>(u.grid_window_cells.z));
}

@compute @workgroup_size(64)
fn reuse_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let width = (u.grid_size + 3u) / 4u;
    let tile = gid.x;
    if tile >= width * width * width { return; }
    let signature = reuse_signature();
    var same = atomicLoad(&reuse_header[14]) != 0u;
    for (var i = 0u; i < 9u; i++) { same = same && atomicLoad(&reuse_header[i]) == signature[i]; }
    if !same { atomicStore(&reuse_tiles[tile].valid, 0u); }
    atomicStore(&reuse_tiles[tile].dirty, select(1u, 0u, atomicLoad(&reuse_tiles[tile].valid) != 0u));
    if tile == 0u { atomicStore(&reuse_header[15], 0u); }
}

fn reuse_position_supported(pos: vec3<f32>) -> bool {
    let xp = (pos - u.grid_origin) * u.inv_cell_size;
    let base_float = floor(xp - vec3<f32>(0.5));
    return finite_vector(base_float) && all(abs(base_float) <= vec3<f32>(536870912.0))
        && all(u.grid_window_cells.xyz <= vec3<i32>(536870912)) && all(u.grid_window_cells.xyz >= vec3<i32>(-536870912));
}

fn reuse_invalidate_position(pos: vec3<f32>) {
    if !reuse_position_supported(pos) {
        atomicOr(&reuse_header[15], 1u);
        return;
    }
    let base_float = floor((pos - u.grid_origin) * u.inv_cell_size - vec3<f32>(0.5));
    let base = vec3<i32>(base_float) - u.grid_window_cells.xyz;
    let lo = max(base, vec3<i32>(0));
    let hi = min(base + vec3<i32>(2), vec3<i32>(i32(u.grid_size) - 1));
    if any(hi < lo) { return; }
    let first = vec3<u32>(lo) / 4u;
    let last = vec3<u32>(hi) / 4u;
    let width = (u.grid_size + 3u) / 4u;
    for (var z = first.z; z <= last.z; z++) {
        for (var y = first.y; y <= last.y; y++) {
            for (var x = first.x; x <= last.x; x++) {
                let tile = x + width * (y + width * z);
                atomicOr(&reuse_tiles[tile].dirty, 1u);
                atomicStore(&reuse_tiles[tile].valid, 0u);
            }
        }
    }
}

@compute @workgroup_size(64)
fn reuse_mark(@builtin(global_invocation_id) gid: vec3<u32>) {
    if !grid_valid() { return; }
    let pid = gid.x;
    if pid == 0u {
        let signature = reuse_signature();
        for (var i = 0u; i < 9u; i++) { atomicStore(&reuse_header[i], signature[i]); }
        atomicStore(&reuse_header[14], 1u);
    }
    if pid >= arrayLength(&reuse_keys) { return; }
    let was_active = reuse_keys[pid].words[26] != 0u;
    let old_position = bitcast<vec3<f32>>(vec3<u32>(reuse_keys[pid].words[0], reuse_keys[pid].words[1], reuse_keys[pid].words[2]));
    if pid >= u.particle_count {
        if was_active { reuse_invalidate_position(old_position); reuse_keys[pid].words[26] = 0u; }
        return;
    }
    var key: array<u32, 26>;
    for (var a = 0u; a < 3u; a++) {
        key[a] = bitcast<u32>(p_pos[pid][a]);
        key[3u + a] = bitcast<u32>(p_vel[pid][a]);
        for (var b = 0u; b < 3u; b++) {
            key[7u + 3u * a + b] = bitcast<u32>(p_C[pid][a][b]);
            key[16u + 3u * a + b] = bitcast<u32>(gather_particles[pid].stress[a][b]);
        }
    }
    key[6] = bitcast<u32>(p_mass[pid]);
    key[25] = bitcast<u32>(gather_particles[pid].stiffness);
    var same = was_active;
    for (var i = 0u; i < 26u; i++) {
        if (key[i] & 0x7f800000u) == 0x7f800000u { atomicOr(&reuse_header[15], 1u); }
        same = same && reuse_keys[pid].words[i] == key[i];
    }
    if !reuse_position_supported(p_pos[pid].xyz) { atomicOr(&reuse_header[15], 1u); }
    if same { return; }
    if was_active { reuse_invalidate_position(old_position); }
    reuse_invalidate_position(p_pos[pid].xyz);
    for (var i = 0u; i < 26u; i++) { reuse_keys[pid].words[i] = key[i]; }
    reuse_keys[pid].words[26] = 1u;
}

fn reuse_restore_node(tile_id: u32, lane: u32, gid: vec3<u32>) {
    let width = (u.grid_size + 3u) / 4u;
    let tile = vec3<u32>(tile_id % width, (tile_id / width) % width, tile_id / (width * width));
    let cell = 4u * tile + vec3<u32>(lane % 4u, (lane / 4u) % 4u, lane / 16u);
    if any(cell >= vec3<u32>(u.grid_size)) { return; }
    let world = cell.x + u.grid_size * (cell.y + u.grid_size * cell.z);
    let index = dispatched_grid_index(world, gid);
    atomicStore(&g_mass[index], bitcast<GridAtomicScalar>(reuse_nodes[world * 5u]));
    for (var a = 0u; a < 3u; a++) {
        atomicStore(&g_momentum[index * 3u + a], bitcast<GridAtomicScalar>(reuse_nodes[world * 5u + 1u + a]));
    }
    atomicStore(&g_kdiag[index], reuse_nodes[world * 5u + 4u]);
}
