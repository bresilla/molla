// GPU Bekker tire/soil rut writeback.
//
// One invocation owns one heightfield cell and gathers all tire contact cells
// whose nearest grid sample maps to that terrain cell. This avoids float
// atomics and mirrors the CPU Bekker ratchet: terrain only plastically sinks.

struct Uniforms {
    // x=nx, y=nz, z=contact_count, w=reserved
    dims: vec4<u32>,
    // x=origin_x, y=origin_z, z=spacing, w=reserved
    origin: vec4<f32>,
    // x=stiffness, y=max_sinkage, z=shear_strength, w=shear_sinkage_gain
    soil0: vec4<f32>,
    // x=bekker_n, yzw=reserved
    soil1: vec4<f32>,
};

@group(0) @binding(0) var<storage, read>       rest_heights: array<f32>;
@group(0) @binding(1) var<storage, read>       heights_in:   array<f32>;
@group(0) @binding(2) var<storage, read>       sinkage_in:   array<f32>;
// (world_x, world_z, normal_pressure_pa, shear_pressure_pa)
@group(0) @binding(3) var<storage, read>       contacts:     array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> heights_out:  array<f32>;
@group(0) @binding(5) var<storage, read_write> sinkage_out:  array<f32>;
@group(1) @binding(0) var<uniform> u: Uniforms;

fn nearest_cell_index(world_x: f32, world_z: f32) -> u32 {
    let spacing = max(u.origin.z, 1.0e-6);
    let ix_i = i32(round((world_x - u.origin.x) / spacing));
    let iz_i = i32(round((world_z - u.origin.y) / spacing));
    let ix = u32(clamp(ix_i, 0, i32(u.dims.x) - 1));
    let iz = u32(clamp(iz_i, 0, i32(u.dims.y) - 1));
    return iz * u.dims.x + ix;
}

@compute @workgroup_size(64)
fn bekker_writeback(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let n_cells = u.dims.x * u.dims.y;
    if (idx >= n_cells) {
        return;
    }

    let rest_h = rest_heights[idx];
    let current_h = heights_in[idx];
    var sinkage = max(max(sinkage_in[idx], rest_h - current_h), 0.0);
    let stiffness = max(u.soil0.x, 1.0e-9);
    let max_sinkage = max(u.soil0.y, 0.0);
    let shear_strength = max(u.soil0.z, 1.0);
    let shear_gain = max(u.soil0.w, 0.0);
    let bekker_n = max(u.soil1.x, 1.0e-6);

    for (var c: u32 = 0u; c < u.dims.z; c = c + 1u) {
        let row = contacts[c];
        if (nearest_cell_index(row.x, row.y) != idx) {
            continue;
        }
        let pressure = max(row.z, 0.0);
        let shear_pressure = max(row.w, 0.0);
        let normal_sinkage = min(pow(pressure / stiffness, 1.0 / bekker_n), max_sinkage);
        let overshear = max(shear_pressure / shear_strength - 1.0, 0.0);
        let shear_sinkage = min(overshear * shear_gain * max_sinkage, max_sinkage * 0.45);
        sinkage = max(sinkage, min(normal_sinkage + shear_sinkage, max_sinkage));
    }

    sinkage_out[idx] = sinkage;
    heights_out[idx] = rest_h - sinkage;
}
