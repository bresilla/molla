@group(0) @binding(3) var<storage, read_write> soil_quiet: array<vec2<u32>>;

fn soil_exact_rest(pid: u32, proposal: SoilMotionProposal) -> bool {
    if any(p_vel[pid].xyz != vec3<f32>(0.0)) || any(proposal.velocity.xyz != vec3<f32>(0.0)) { return false; }
    if any(bitcast<vec3<u32>>(p_pos[pid].xyz) != bitcast<vec3<u32>>(proposal.position)) { return false; }
    if bitcast<u32>(p_jp[pid]) != bitcast<u32>(proposal.plastic) { return false; }
    for (var column = 0u; column < 3u; column++) {
        if any(p_C[pid][column] != vec3<f32>(0.0)) || any(proposal.affine[column] != vec3<f32>(0.0)) { return false; }
        if any(bitcast<vec3<u32>>(p_F[pid][column]) != bitcast<vec3<u32>>(proposal.deformation[column])) { return false; }
    }
    return true;
}

fn soil_quiet_observe(pid: u32, proposal: SoilMotionProposal) {
    let previous = soil_quiet[pid];
    let dt = bitcast<u32>(u.dt);
    var streak = 0u;
    if soil_exact_rest(pid, proposal) {
        streak = 1u;
        if previous.y == dt {
            streak = previous.x + select(1u, 0u, previous.x == 0xffffffffu);
        }
    }
    soil_quiet[pid] = vec2<u32>(streak, dt);
}
