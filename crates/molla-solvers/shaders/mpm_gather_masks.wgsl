var<workgroup> gather_mask_tile: vec2<u32>;
var<workgroup> gather_masks: array<atomic<u32>, 128>;

@compute @workgroup_size(64)
fn gather_grid_masked(@builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let width = (u.grid_size + 3u) / 4u;
    if lane == 0u {
        var tile_id = 0u;
        var valid = false;
        if DENSE_GATHER {
            tile_id = group.x + groups.x * group.y;
            valid = tile_id < width * width * width && grid_valid();
        } else {
            let slot = active_tile_slot(gid);
            valid = slot < atomicLoad(&g_kdiag[grid_metadata()]) && grid_valid();
            if valid { tile_id = atomicLoad(&g_kdiag[grid_tile_ids() + slot]); }
        }
        gather_mask_tile = vec2<u32>(tile_id, select(0u, 1u, valid));
    }
    let shared_tile = workgroupUniformLoad(&gather_mask_tile);
    if shared_tile.y == 0u { return; }
    let tile_id = shared_tile.x;
    let tile = vec3<i32>(i32(tile_id % width), i32((tile_id / width) % width), i32(tile_id / (width * width)));
    if lane < 27u {
        let neighbor = tile + vec3<i32>(i32(lane % 3u), i32((lane / 3u) % 3u), i32(lane / 9u)) - vec3<i32>(1);
        var range = vec2<u32>(0u);
        if all(neighbor >= vec3<i32>(0)) && all(neighbor < vec3<i32>(i32(width))) {
            let key = u32(neighbor.x) + width * (u32(neighbor.y) + width * u32(neighbor.z));
            range = vec2<u32>(gather_lower_bound(key), gather_lower_bound(key + 1u));
        }
        gather_ranges[lane] = range;
    }
    workgroupBarrier();
    let tile_base = tile * 4;
    let cell = tile_base + vec3<i32>(i32(lane % 4u), i32((lane / 4u) % 4u), i32(lane / 16u));
    let valid_cell = all(cell < vec3<i32>(i32(u.grid_size)));
    let inv_dx2 = u.inv_cell_size * u.inv_cell_size;
    var mass_sum = 0.0;
    var momentum_sum = vec3<f32>(0.0);
    var stiffness_sum = 0.0;
    for (var tile_index = 0u; tile_index < 27u; tile_index++) {
        let range = workgroupUniformLoad(&gather_ranges[tile_index]);
        for (var first = range.x; first < range.y; first += 64u) {
            atomicStore(&gather_masks[lane * 2u], 0u);
            atomicStore(&gather_masks[lane * 2u + 1u], 0u);
            workgroupBarrier();
            let sorted = first + lane;
            if sorted < range.y {
                let pid = p_material[u.grid_layout.w + sorted];
                let xp = (p_pos[pid].xyz - u.grid_origin) * u.inv_cell_size;
                let base = vec3<i32>(floor(xp - vec3<f32>(0.5))) - u.grid_window_cells.xyz;
                let lo = max(base, tile_base);
                let hi = min(base + vec3<i32>(2), tile_base + vec3<i32>(3));
                for (var z = lo.z; z <= hi.z; z++) {
                    for (var y = lo.y; y <= hi.y; y++) {
                        for (var x = lo.x; x <= hi.x; x++) {
                            let local = vec3<u32>(vec3<i32>(x, y, z) - tile_base);
                            let node = local.x + 4u * (local.y + 4u * local.z);
                            atomicOr(&gather_masks[node * 2u + lane / 32u], 1u << (lane % 32u));
                        }
                    }
                }
            }
            workgroupBarrier();
            if valid_cell {
                for (var word = 0u; word < 2u; word++) {
                    var mask = atomicLoad(&gather_masks[lane * 2u + word]);
                    while mask != 0u {
                        let bit = firstTrailingBit(mask);
                        mask &= mask - 1u;
                        let pid = p_material[u.grid_layout.w + first + word * 32u + bit];
                        let pos = p_pos[pid].xyz;
                        let xp = (pos - u.grid_origin) * u.inv_cell_size;
                        let base_xyz = floor(xp - vec3<f32>(0.5));
                        let offset = cell - (vec3<i32>(base_xyz) - u.grid_window_cells.xyz);
                        let fx = xp - base_xyz;
                        let wx = quad_weights(fx.x);
                        let wy = quad_weights(fx.y);
                        let wz = quad_weights(fx.z);
                        let w = wx[u32(offset.x)] * wy[u32(offset.y)] * wz[u32(offset.z)];
                        let mass = p_mass[pid];
                        let xn_xp = (vec3<f32>(offset) - fx) * u.cell_size;
                        let aff = p_C[pid] * xn_xp;
                        let stress_force = gather_particles[pid].stress * xn_xp * (4.0 * inv_dx2);
                        let mom = w * mass * (p_vel[pid].xyz + aff) + w * stress_force;
                        mass_sum += w * mass;
                        momentum_sum += mom;
                        stiffness_sum += w * w * gather_particles[pid].stiffness;
                    }
                }
            }
            workgroupBarrier();
        }
    }
    if !valid_cell { return; }
    let world = u32(cell.x) + u.grid_size * (u32(cell.y) + u.grid_size * u32(cell.z));
    let index = dispatched_grid_index(world, gid);
    atomicStore(&g_mass[index], bitcast<GridAtomicScalar>(mass_sum));
    atomicStore(&g_momentum[index * 3u], bitcast<GridAtomicScalar>(momentum_sum.x));
    atomicStore(&g_momentum[index * 3u + 1u], bitcast<GridAtomicScalar>(momentum_sum.y));
    atomicStore(&g_momentum[index * 3u + 2u], bitcast<GridAtomicScalar>(momentum_sum.z));
    atomicStore(&g_kdiag[index], bitcast<u32>(select(stiffness_sum, 0.0, DENSE_GATHER || ZERO_GATHER_STIFFNESS)));
}
