struct SoilMotionProposal {
    position: vec3<f32>,
    plastic: f32,
    velocity: vec4<f32>,
    affine: mat3x3<f32>,
    deformation: mat3x3<f32>,
}

@group(0) @binding(2) var<storage, read_write> soil_motion_proposals: array<SoilMotionProposal>;

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn soil_motion_commit(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if pid >= u.particle_count || !grid_valid() { return; }
    let proposal = soil_motion_proposals[pid];
    p_pos[pid] = vec4<f32>(proposal.position, 0.0);
    p_vel[pid] = proposal.velocity;
    p_C[pid] = proposal.affine;
    p_F[pid] = proposal.deformation;
    p_jp[pid] = proposal.plastic;
}
