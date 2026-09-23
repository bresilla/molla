fn stabilize_scalar_feedback(
    begin: u32, n: u32,
    counts: ptr<function, StopCounts>,
    sl: ptr<function, StopVectors>, sa: ptr<function, StopVectors>,
    ul: ptr<function, StopVectors>, ua: ptr<function, StopVectors>,
    inverse: ptr<function, StopBlocks>,
    acceleration: ptr<function, StopScalars>,
) -> bool {
    var parents: StopParents;
    for (var k = 0u; k < n; k += 1u) {
        parents[k] = -1;
        for (var p = 0u; p < k; p += 1u) {
            if joint_child[begin+p] == joint_parent[begin+k] { parents[k] = i32(p); }
        }
    }
    var correction: StopScalars;
    var unit: StopScalars;
    var changed = false;
    for (var k = 0u; k < n; k += 1u) {
        let j = begin + k;
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            let dof = joint_dof_offset[j] + d;
            let mode = joint_target_mode[dof];
            if mode < TARGET_POS || mode > TARGET_PV { continue; }
            let kind = joint_type[j];
            if kind != JOINT_REVOLUTE && kind != JOINT_PRISMATIC && kind != JOINT_CABLE { return false; }
            if abs(joint_limit_upper[dof] - joint_limit_lower[dof]) < 1e-12 { continue; }
            let kp = joint_target_kp[dof];
            let kd = joint_target_kd[dof];
            if !(kp >= 0.0 && kp <= 3.402823e38 && kd >= 0.0 && kd <= 3.402823e38) { return false; }
            unit[k][d] = 1.0;
            let response = stop_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &unit);
            unit[k][d] = 0.0;
            let inverse_inertia = response[k][d];
            if !(inverse_inertia > 0.0 && inverse_inertia <= 3.402823e38) { return false; }
            let q = joint_q[joint_coord_offset[j] + d];
            let qd = joint_qd[dof];
            let drive = drive_torque(mode, q, qd, kp, kd, joint_target_pos[dof], joint_target_vel[dof]);
            let stiffness = select(kp, 0.0, mode == TARGET_VEL);
            let denominator = 1.0 + (kd + stiffness * uniforms.dt) * uniforms.dt * inverse_inertia;
            if !(denominator >= 1.0 && denominator <= 3.402823e38) { return false; }
            let stabilized = (drive - stiffness * uniforms.dt * qd) / denominator;
            let cap = joint_motor_max_force[dof];
            if !(cap >= 0.0) { return false; }
            var cable = 0.0;
            if kind == JOINT_CABLE {
                let spring = select(joint_cable_k_lin[j], joint_cable_k_bend[j], d != 0u);
                let damper = select(joint_cable_c_lin[j], joint_cable_c_bend[j], d != 0u);
                cable = -spring * q - damper * qd;
            }
            let before = joint_force[dof] + clamp(drive, -cap, cap) - joint_friction[dof] * qd
                + cable + joint_muscle_torque[dof];
            let after = joint_force[dof] + clamp(stabilized, -cap, cap) - joint_friction[dof] * qd
                + cable + joint_muscle_torque[dof];
            let delta = clamp_or_passthrough(after, joint_effort_limit[dof])
                - clamp_or_passthrough(before, joint_effort_limit[dof]);
            if !(abs(drive) <= 3.402823e38 && abs(stabilized) <= 3.402823e38
                && abs(before) <= 3.402823e38 && abs(after) <= 3.402823e38 && abs(delta) <= 3.402823e38) { return false; }
            correction[k][d] = delta;
            changed = changed || delta != 0.0;
        }
    }
    if !changed { return true; }
    let delta = stop_mass_response(n, &parents, counts, sl, sa, ul, ua, inverse, &correction);
    for (var k = 0u; k < n; k += 1u) {
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            if !(abs((*acceleration)[k][d] + delta[k][d]) <= 3.402823e38) { return false; }
        }
    }
    for (var k = 0u; k < n; k += 1u) {
        for (var d = 0u; d < (*counts)[k]; d += 1u) {
            (*acceleration)[k][d] += delta[k][d];
        }
    }
    return true;
}
