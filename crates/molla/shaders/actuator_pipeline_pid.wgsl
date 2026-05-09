// PID effort kernel — pipeline variant.
//
// Like `actuator_pipeline_pd.wgsl` but adds an integral term with
// anti-windup. Reads target pos/vel from per-actuator arrays
// (typically the output of the Delay stage) rather than the global
// `joint_target_*` arrays. Effort cap is **not** applied here —
// clamping is its own pipeline stage. This keeps the integral term
// honest: capping the effort post-clamp would alias the integrator
// anti-windup against the downstream clamp.
//
// Effort law:
//
//   integral ← clamp(integral + (delayed_target_pos - q) · dt,
//                    -integral_max, +integral_max)   (when imax > 0)
//   effort = kp·(delayed_target_pos - q)
//          + ki·integral
//          + kd·(delayed_target_vel - qdot)
//
// `integral` is per-actuator state owned by the controller; the
// kernel reads from `current_integral` and writes the updated value
// to `next_integral` so the host can ping-pong the buffers (no race
// because each thread touches a distinct slot).

struct Uniforms {
    actuator_count: u32,
    dt:             f32,
    _pad0:          u32,
    _pad1:          u32,
};

@group(0) @binding(0) var<storage, read> joint_q:           array<f32>;
@group(0) @binding(1) var<storage, read> joint_qd:          array<f32>;
@group(0) @binding(2) var<storage, read> delayed_target_pos: array<f32>;
@group(0) @binding(3) var<storage, read> delayed_target_vel: array<f32>;

@group(1) @binding(0) var<storage, read> dof_indices:      array<u32>;
@group(1) @binding(1) var<storage, read> kp:               array<f32>;
@group(1) @binding(2) var<storage, read> ki:               array<f32>;
@group(1) @binding(3) var<storage, read> kd:               array<f32>;
@group(1) @binding(4) var<storage, read> integral_max:     array<f32>;
@group(1) @binding(5) var<storage, read> current_integral: array<f32>;

@group(2) @binding(0) var<storage, read_write> joint_force:   array<f32>;
@group(2) @binding(1) var<storage, read_write> next_integral: array<f32>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn pipeline_pid_effort(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.actuator_count) { return; }

    let dof = dof_indices[i];
    let q = joint_q[dof];
    let qdot = joint_qd[dof];
    let tp = delayed_target_pos[i];
    let tv = delayed_target_vel[i];

    let pos_err = tp - q;
    let vel_err = tv - qdot;

    var integ = current_integral[i] + pos_err * uniforms.dt;
    let imax = integral_max[i];
    if (imax > 0.0) {
        integ = clamp(integ, -imax, imax);
    }

    let effort = kp[i] * pos_err + ki[i] * integ + kd[i] * vel_err;
    joint_force[dof] = effort;
    next_integral[i] = integ;
}
