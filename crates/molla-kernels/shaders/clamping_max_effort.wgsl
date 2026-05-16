// Symmetric per-actuator effort clamp.
//
// Mirrors Newton's `_box_clamp_kernel`
// (xtra/newton/_src/actuators/clamping/clamping_max_effort.py).
// Reads `src[i]` (typically the controller output, sparse over DOF
// indices) and writes the clamped value back into `joint_force[dof]`
// using the actuator → DOF index map. `max_effort[i] <= 0` disables
// clamping (passthrough), matching the "0 ⇒ off" convention used by
// `actuator_pd.wgsl`.

struct Uniforms {
    actuator_count: u32,
    _pad0:          u32,
    _pad1:          u32,
    _pad2:          u32,
};

@group(0) @binding(0) var<storage, read>       dof_indices: array<u32>;
@group(0) @binding(1) var<storage, read>       max_effort:  array<f32>;
@group(1) @binding(0) var<storage, read_write> joint_force: array<f32>;
@group(2) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn clamp_max_effort(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.actuator_count) { return; }

    let dof = dof_indices[i];
    let cap = max_effort[i];
    let f = joint_force[dof];
    if (cap > 0.0) {
        joint_force[dof] = clamp(f, -cap, cap);
    }
}
