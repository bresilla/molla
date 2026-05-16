// PD effort kernel — pipeline variant.
//
// Like `actuator_pd.wgsl` but reads target pos/vel from per-actuator
// arrays (typically the output of the Delay stage) rather than from
// the global `joint_target_pos / joint_target_vel` arrays. This lets
// the `Actuator` orchestrator chain Delay → PD → Clamping without
// reaching back into the global control buffers between stages.
//
// Effort law (no clamp here — clamping is its own stage):
//
//   effort = kp · (delayed_target_pos - joint_q[dof])
//          + kd · (delayed_target_vel - joint_qd[dof])
//
// Output is **assigned** to `joint_force[dof]`. Caller must ensure
// every actuator targets a unique DOF.

struct Uniforms {
    actuator_count: u32,
    _pad0:          u32,
    _pad1:          u32,
    _pad2:          u32,
};

@group(0) @binding(0) var<storage, read>       joint_q:           array<f32>;
@group(0) @binding(1) var<storage, read>       joint_qd:          array<f32>;
@group(0) @binding(2) var<storage, read>       delayed_target_pos: array<f32>;
@group(0) @binding(3) var<storage, read>       delayed_target_vel: array<f32>;

@group(1) @binding(0) var<storage, read> dof_indices: array<u32>;
@group(1) @binding(1) var<storage, read> kp:          array<f32>;
@group(1) @binding(2) var<storage, read> kd:          array<f32>;

@group(2) @binding(0) var<storage, read_write> joint_force: array<f32>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn pipeline_pd_effort(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.actuator_count) { return; }

    let dof = dof_indices[i];
    let q = joint_q[dof];
    let qdot = joint_qd[dof];
    let tp = delayed_target_pos[i];
    let tv = delayed_target_vel[i];

    let effort = kp[i] * (tp - q) + kd[i] * (tv - qdot);
    joint_force[dof] = effort;
}
