struct GatherStressKey {
    deformation: mat3x3<f32>,
    scalars0: vec4<u32>,
    scalars1: vec4<u32>,
    valid: u32,
}
@group(1) @binding(11) var<storage, read_write> gather_stress_keys: array<GatherStressKey>;

fn gather_stress_hit(pid: u32, f: mat3x3<f32>, a: vec4<u32>, b: vec4<u32>) -> bool {
    if gather_stress_keys[pid].valid == 0u { return false; }
    if any(gather_stress_keys[pid].scalars0 != a) || any(gather_stress_keys[pid].scalars1 != b) { return false; }
    for (var column = 0u; column < 3u; column++) {
        if any(bitcast<vec3<u32>>(f[column]) != bitcast<vec3<u32>>(gather_stress_keys[pid].deformation[column])) { return false; }
    }
    return true;
}

@compute @workgroup_size(64)
fn gather_prepare(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= u.particle_count || !grid_valid() { return; }
    let f = p_F[pid];
    let material = materials[p_material[pid]];
    let a = bitcast<vec4<u32>>(vec4<f32>(p_volume[pid], p_jp[pid], u.dt, u.youngs));
    let b = bitcast<vec4<u32>>(vec4<f32>(u.poisson, u.inv_cell_size, material.p0.x, material.p1.x));
    if gather_stress_hit(pid, f, a, b) { return; }
    gather_prepare_uncached(gid);
    gather_stress_keys[pid].deformation = f;
    gather_stress_keys[pid].scalars0 = a;
    gather_stress_keys[pid].scalars1 = b;
    gather_stress_keys[pid].valid = 1u;
}
