// PD actuator kernel — drives joint DOFs toward a target position
// + velocity using stateless proportional-derivative control.
//
// Mirrors Newton's `_pd_effort_kernel`
// (xtra/newton/_src/actuators/controllers/controller_pd.py) with
// these simplifications:
//   - one actuator per DOF (no sparse pos/vel index lookup arrays
//     since every actuator addresses a distinct DOF)
//   - effort cap baked into the same kernel (Newton runs clamping
//     as a separate stage; for our first iteration that's overkill)
//   - no const_effort / feedforward inputs (re-add when an example
//     actually needs them)
//
// Effort law:
//
//   effort = kp · (target_pos - current_pos) + kd · (target_vel - current_vel)
//   effort = clamp(effort, -max_effort, +max_effort)   (when max_effort > 0)
//
// Output is **assigned** (not accumulated) to `joint_force[dof]`.
// Caller should zero `joint_force` before stepping the actuator and
// must ensure every actuator targets a unique DOF.

struct Uniforms {
    actuator_count: u32,
    _pad0:          u32,
    _pad1:          u32,
    _pad2:          u32,
};

@group(0) @binding(0) var<storage, read> joint_q:          array<f32>;
@group(0) @binding(1) var<storage, read> joint_qd:         array<f32>;
@group(0) @binding(2) var<storage, read> joint_target_pos: array<f32>;
@group(0) @binding(3) var<storage, read> joint_target_vel: array<f32>;

@group(1) @binding(0) var<storage, read> dof_indices:  array<u32>;
@group(1) @binding(1) var<storage, read> kp:           array<f32>;
@group(1) @binding(2) var<storage, read> kd:           array<f32>;
@group(1) @binding(3) var<storage, read> max_effort:   array<f32>;

@group(2) @binding(0) var<storage, read_write> joint_force: array<f32>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn compute_pd_effort(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.actuator_count) { return; }

    let dof = dof_indices[tid];

    let q = joint_q[dof];
    let qdot = joint_qd[dof];
    let tp = joint_target_pos[dof];
    let tv = joint_target_vel[dof];

    var effort = kp[tid] * (tp - q) + kd[tid] * (tv - qdot);

    let cap = max_effort[tid];
    if (cap > 0.0) {
        effort = clamp(effort, -cap, cap);
    }

    joint_force[dof] = effort;
}
