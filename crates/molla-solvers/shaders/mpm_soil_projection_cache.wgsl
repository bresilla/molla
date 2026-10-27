struct SoilProjectionRecord {
    input_f: mat3x3<f32>,
    output_f: mat3x3<f32>,
    input_jp: f32,
    output_jp: f32,
    dt: u32,
    valid: u32,
}

@group(0) @binding(3) var<storage, read_write> soil_projection_cache: array<SoilProjectionRecord>;
@group(0) @binding(4) var<storage, read> soil_projection_quiet: array<vec2<u32>>;

fn soil_projection_hit(pid: u32, f: mat3x3<f32>, jp: f32) -> bool {
    if soil_projection_cache[pid].valid == 0u || soil_projection_cache[pid].dt != bitcast<u32>(u.dt) { return false; }
    if bitcast<u32>(jp) != bitcast<u32>(soil_projection_cache[pid].input_jp) { return false; }
    for (var column = 0u; column < 3u; column++) {
        if any(bitcast<vec3<u32>>(f[column]) != bitcast<vec3<u32>>(soil_projection_cache[pid].input_f[column])) { return false; }
    }
    return true;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn soil_project_cached(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= u.particle_count || !grid_valid() { return; }
    if soil_projection_quiet[pid].x == 0u {
        soil_project_uncached(gid);
        return;
    }
    let f = soil_motion_proposals[pid].deformation;
    let jp = soil_motion_proposals[pid].plastic;
    if soil_projection_hit(pid, f, jp) {
        soil_motion_proposals[pid].deformation = soil_projection_cache[pid].output_f;
        soil_motion_proposals[pid].plastic = soil_projection_cache[pid].output_jp;
        return;
    }
    soil_projection_cache[pid].valid = 0u;
    soil_project_uncached(gid);
    if !grid_valid() { return; }
    soil_projection_cache[pid].input_f = f;
    soil_projection_cache[pid].input_jp = jp;
    soil_projection_cache[pid].output_f = soil_motion_proposals[pid].deformation;
    soil_projection_cache[pid].output_jp = soil_motion_proposals[pid].plastic;
    soil_projection_cache[pid].dt = bitcast<u32>(u.dt);
    soil_projection_cache[pid].valid = 1u;
}
