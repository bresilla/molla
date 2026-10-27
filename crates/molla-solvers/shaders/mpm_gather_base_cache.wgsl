struct GatherParticle {
    stress: mat3x3<f32>,
    stiffness: f32,
    base_x: i32,
    base_y: i32,
    base_z: i32,
}
override DENSE_GATHER: bool = false;
override ZERO_GATHER_STIFFNESS: bool = false;
override PRUNE_GATHER_RANGES: bool = true;
override CACHE_GATHER_BASE: bool = true;
@group(1) @binding(9) var<storage, read_write> gather_particles: array<GatherParticle>;
@group(1) @binding(10) var<storage, read> gather_keys: array<u32>;
var<workgroup> gather_ranges: array<vec2<u32>, 27>;

@compute @workgroup_size(64)
fn gather_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= u.particle_count || !grid_valid() { return; }
    let vol = p_volume[pid];
    let F = p_F[pid];
    let raw_svd = svd3_jacobi(F);
    let s_clamped = vec3<f32>(
        clamp(raw_svd.s.x, 0.1, 10.0),
        clamp(raw_svd.s.y, 0.1, 10.0),
        clamp(raw_svd.s.z, 0.1, 10.0),
    );
    let svd = Svd3(raw_svd.u, s_clamped, raw_svd.v);
    let R = svd.u * transpose(svd.v);
    let J = svd.s.x * svd.s.y * svd.s.z;

    let mu = u.youngs / (2.0 * (1.0 + u.poisson));
    let lam = u.youngs * u.poisson / ((1.0 + u.poisson) * (1.0 - 2.0 * u.poisson));
    let mrow = materials[p_material[pid]];
    let rheo = u32(mrow.p0.x + 0.5);
    var h: f32 = 1.0;
    if (rheo == 1u) {
        h = exp(clamp(-mrow.p1.x * p_jp[pid], -3.0, 3.0));
    } else if (rheo == 0u) {
        h = exp(clamp(p_jp[pid], -3.0, 3.0));
    }
    let mu_h = mu * h;
    let lam_h = lam * h;

    let F_clamped = svd.u * mat3x3<f32>(
        vec3<f32>(svd.s.x, 0.0, 0.0),
        vec3<f32>(0.0, svd.s.y, 0.0),
        vec3<f32>(0.0, 0.0, svd.s.z),
    ) * transpose(svd.v);
    let f_minus_r = F_clamped - R;
    let cof = svd.u * mat3x3<f32>(
        vec3<f32>(svd.s.y * svd.s.z, 0.0, 0.0),
        vec3<f32>(0.0, svd.s.x * svd.s.z, 0.0),
        vec3<f32>(0.0, 0.0, svd.s.x * svd.s.y),
    ) * transpose(svd.v);
    let pk1 = 2.0 * mu_h * f_minus_r + lam_h * (J - 1.0) * cof;

    let stress = -u.dt * vol * pk1 * transpose(F_clamped);

    let inv_dx2 = u.inv_cell_size * u.inv_cell_size;

    let kdiag_p = 4.0 * vol * (2.0 * mu_h + lam_h) * inv_dx2;


    gather_particles[pid].stress = stress;
    gather_particles[pid].stiffness = kdiag_p;
    if CACHE_GATHER_BASE {
        let xp = (p_pos[pid].xyz - u.grid_origin) * u.inv_cell_size;
        let base = vec3<i32>(floor(xp - vec3<f32>(0.5))) - u.grid_window_cells.xyz;
        gather_particles[pid].base_x = base.x;
        gather_particles[pid].base_y = base.y;
        gather_particles[pid].base_z = base.z;
    }
}

fn gather_lower_bound(key: u32) -> u32 {
    var first = 0u;
    var last = u.particle_count;
    while first < last {
        let middle = first + (last - first) / 2u;
        if gather_keys[middle] < key { first = middle + 1u; }
        else { last = middle; }
    }
    return first;
}

@compute @workgroup_size(64)
fn gather_grid(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32,
    @builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>) {
    let width = (u.grid_size + 3u) / 4u;
    var tile_id = 0u;
    var valid_slot = false;
    if DENSE_GATHER {
        tile_id = group.x + groups.x * group.y;
        valid_slot = tile_id < width * width * width && grid_valid();
    } else {
        let slot = active_tile_slot(gid);
        valid_slot = slot < atomicLoad(&g_kdiag[grid_metadata()]) && grid_valid();
        if valid_slot { tile_id = atomicLoad(&g_kdiag[grid_tile_ids() + slot]); }
    }
    let tile = vec3<i32>(i32(tile_id % width), i32((tile_id / width) % width), i32(tile_id / (width * width)));
    if lane < 27u {
        let neighbor = tile + vec3<i32>(i32(lane % 3u), i32((lane / 3u) % 3u), i32(lane / 9u)) - vec3<i32>(1);
        var range = vec2<u32>(0u);
        if valid_slot && all(neighbor >= vec3<i32>(0)) && all(neighbor < vec3<i32>(i32(width))) {
            let key = u32(neighbor.x) + width * (u32(neighbor.y) + width * u32(neighbor.z));
            range = vec2<u32>(gather_lower_bound(key), gather_lower_bound(key + 1u));
        }
        gather_ranges[lane] = range;
    }
    workgroupBarrier();
    if !valid_slot { return; }
    var world = grid_cell_id(gid);
    if DENSE_GATHER {
        let xyz = vec3<u32>(tile) * 4u + vec3<u32>(lane % 4u, (lane / 4u) % 4u, lane / 16u);
        if any(xyz >= vec3<u32>(u.grid_size)) { return; }
        world = xyz.x + u.grid_size * (xyz.y + u.grid_size * xyz.z);
    }
    if world >= u.grid_total { return; }
    let index = dispatched_grid_index(world, gid);
    let cell = vec3<i32>(i32(world % u.grid_size), i32((world / u.grid_size) % u.grid_size), i32(world / (u.grid_size * u.grid_size)));
    let inv_dx2 = u.inv_cell_size * u.inv_cell_size;
    var mass_sum = 0.0;
    var momentum_sum = vec3<f32>(0.0);
    var stiffness_sum = 0.0;
    let first_tile = max(cell - vec3<i32>(2), vec3<i32>(0)) / vec3<i32>(4);
    let last_tile = min(cell + vec3<i32>(1), vec3<i32>(i32(u.grid_size) - 1)) / vec3<i32>(4);
    for (var tile_index = 0u; tile_index < 27u; tile_index++) {
        if PRUNE_GATHER_RANGES {
            let neighbor = tile + vec3<i32>(i32(tile_index % 3u), i32((tile_index / 3u) % 3u), i32(tile_index / 9u)) - vec3<i32>(1);
            if any(neighbor < first_tile) || any(neighbor > last_tile) { continue; }
        }
        let range = gather_ranges[tile_index];
        for (var sorted = range.x; sorted < range.y; sorted++) {
            let pid = p_material[u.grid_layout.w + sorted];
            var offset = vec3<i32>(0);
            if CACHE_GATHER_BASE {
                offset = cell - vec3<i32>(gather_particles[pid].base_x, gather_particles[pid].base_y, gather_particles[pid].base_z);
                if any(offset < vec3<i32>(0)) || any(offset > vec3<i32>(2)) { continue; }
            }
            let pos = p_pos[pid].xyz;
            let xp = (pos - u.grid_origin) * u.inv_cell_size;
            let base_xyz = floor(xp - vec3<f32>(0.5));
            if !CACHE_GATHER_BASE {
                offset = cell - (vec3<i32>(base_xyz) - u.grid_window_cells.xyz);
                if any(offset < vec3<i32>(0)) || any(offset > vec3<i32>(2)) { continue; }
            }
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
    atomicStore(&g_mass[index], bitcast<GridAtomicScalar>(mass_sum));
    atomicStore(&g_momentum[index * 3u], bitcast<GridAtomicScalar>(momentum_sum.x));
    atomicStore(&g_momentum[index * 3u + 1u], bitcast<GridAtomicScalar>(momentum_sum.y));
    atomicStore(&g_momentum[index * 3u + 2u], bitcast<GridAtomicScalar>(momentum_sum.z));
    atomicStore(&g_kdiag[index], bitcast<u32>(select(stiffness_sum, 0.0, DENSE_GATHER || ZERO_GATHER_STIFFNESS)));
}
