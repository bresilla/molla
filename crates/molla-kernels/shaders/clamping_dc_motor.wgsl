// DC-motor four-quadrant effort-speed saturation.
//
// Mirrors Newton's `_clamp_dc_motor_kernel`
// (xtra/newton/_src/actuators/clamping/clamping_dc_motor.py).
//
// Models a brushed DC motor's torque-vs-speed curve directly from
// three motor parameters:
//
//   stall_torque       — peak torque at zero velocity (= Kt · V / R)
//   no_load_speed      — velocity at which the motor produces zero
//                         torque (= V / Ke ≈ V / Kt for SI units)
//   max_motor_effort   — absolute torque ceiling (clamps the family
//                         of lines to a horizontal band — matches a
//                         motor controller's current limit).
//
// Effort envelope (linear effort-speed):
//
//   effort_max(qd) = min( stall · (1 - qd/no_load),  max_effort )
//   effort_min(qd) = max( stall · (-1 - qd/no_load), -max_effort )
//
// At |qd| beyond a "corner velocity" the envelope would push you
// past the max-effort cap; the host pre-computes that corner and we
// clamp `qd` into it before evaluating the line.
//
// Inputs/outputs match `clamping_max_effort.wgsl` so the orchestrator
// can swap them transparently.

struct Uniforms {
    actuator_count: u32,
    _pad0:          u32,
    _pad1:          u32,
    _pad2:          u32,
};

@group(0) @binding(0) var<storage, read>       dof_indices:        array<u32>;
@group(0) @binding(1) var<storage, read>       stall_torque:       array<f32>;
@group(0) @binding(2) var<storage, read>       no_load_speed:      array<f32>;
@group(0) @binding(3) var<storage, read>       max_motor_effort:   array<f32>;
@group(0) @binding(4) var<storage, read>       corner_velocity:    array<f32>;

@group(1) @binding(0) var<storage, read>       joint_qd:    array<f32>;
@group(1) @binding(1) var<storage, read_write> joint_force: array<f32>;

@group(2) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn clamp_dc_motor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.actuator_count) { return; }

    let dof = dof_indices[i];
    let stall = stall_torque[i];
    let nload = no_load_speed[i];
    let cap   = max_motor_effort[i];
    let corner = corner_velocity[i];

    var qd = joint_qd[dof];
    qd = clamp(qd, -corner, corner);

    var emax = cap;
    var emin = -cap;
    if (nload > 0.0) {
        let line_pos = stall * (1.0 - qd / nload);
        let line_neg = stall * (-1.0 - qd / nload);
        if (line_pos < emax) { emax = line_pos; }
        if (line_neg > emin) { emin = line_neg; }
    }

    let f = joint_force[dof];
    joint_force[dof] = clamp(f, emin, emax);
}
