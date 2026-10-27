// PID actuator kernel — extends [`actuator_pd.wgsl`] with an
// integral term (with anti-windup clamp). Mirrors Newton's
// `_pid_effort_kernel`.
//
// Effort law:
//
//   integral ← clamp(integral + (target_pos - current_pos) · dt,
//                    -integral_max, +integral_max)
//   effort = kp·(target_pos - current_pos)
//          + ki·integral
//          + kd·(target_vel - current_vel)
//   effort = clamp(effort, -max_effort, +max_effort)
//
// `integral` is per-actuator state owned by the controller; the
// kernel reads from `current_integral` and writes the updated value
// to `next_integral` so the host can ping-pong the buffers (or use
// the same buffer if it doesn't mind the in-place update — there's
// no race here because each thread touches a distinct slot).

struct Uniforms {
    actuator_count: u32,
    dt:             f32,
    _pad0:          u32,
    _pad1:          u32,
};

@group(0) @binding(0) var<storage, read> joint_q:          array<f32>;
@group(0) @binding(1) var<storage, read> joint_qd:         array<f32>;
@group(0) @binding(2) var<storage, read> joint_target_pos: array<f32>;
@group(0) @binding(3) var<storage, read> joint_target_vel: array<f32>;

@group(1) @binding(0) var<storage, read> dof_indices:  array<u32>;
@group(1) @binding(1) var<storage, read> kp:           array<f32>;
@group(1) @binding(2) var<storage, read> ki:           array<f32>;
@group(1) @binding(3) var<storage, read> kd:           array<f32>;
@group(1) @binding(4) var<storage, read> integral_max: array<f32>;
@group(1) @binding(5) var<storage, read> max_effort:   array<f32>;
@group(1) @binding(6) var<storage, read> current_integral: array<f32>;

@group(2) @binding(0) var<storage, read_write> joint_force:   array<f32>;
@group(2) @binding(1) var<storage, read_write> next_integral: array<f32>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn compute_pid_effort(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.actuator_count) { return; }

    let dof = dof_indices[tid];
    let q = joint_q[dof];
    let qdot = joint_qd[dof];
    let tp = joint_target_pos[dof];
    let tv = joint_target_vel[dof];

    let pos_err = tp - q;
    let vel_err = tv - qdot;

    var integ = current_integral[tid] + pos_err * uniforms.dt;
    let imax = integral_max[tid];
    if (imax > 0.0) {
        integ = clamp(integ, -imax, imax);
    }

    var effort = kp[tid] * pos_err + ki[tid] * integ + kd[tid] * vel_err;
    let cap = max_effort[tid];
    if (cap > 0.0) {
        effort = clamp(effort, -cap, cap);
    }

    joint_force[dof] = effort;
    next_integral[tid] = integ;
}
