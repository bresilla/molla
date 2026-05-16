// Per-DOF command-input delay — circular buffer rotate + read.
//
// Mirrors Newton's `_delay_buffer_state_kernel` + `_delay_read_kernel`
// (xtra/newton/_src/actuators/delay.py) collapsed into two entry
// points sharing a single bind-group set:
//
//   - `delay_push`  — copy the previous slot, write the current
//                      target_pos / target_vel into the next slot,
//                      bump per-DOF push counter and the global write
//                      index.
//   - `delay_read`  — sample the per-DOF lag offset from the buffer
//                      and write into `out_pos / out_vel`. Falls back
//                      to current targets if the buffer hasn't been
//                      filled or the configured delay is zero.
//
// The buffer is laid out row-major as `buf_depth × N_actuators`,
// indexed `buffer[row * actuator_count + i]`. We track the global
// write index in a single-element u32 buffer so it survives across
// dispatches without a host round-trip.
//
// Single-DOF assumption: for revolute / prismatic the same `dof`
// indexes both `joint_q` and `joint_qd`, so we use one `dof_indices`
// array for both reads (matching `actuator_pd.wgsl`).

struct Uniforms {
    actuator_count: u32,
    buf_depth:      u32,
    _pad0:          u32,
    _pad1:          u32,
};

// Inputs (read-only).
@group(0) @binding(0) var<storage, read>       target_pos:     array<f32>;
@group(0) @binding(1) var<storage, read>       target_vel:     array<f32>;
@group(0) @binding(2) var<storage, read>       dof_indices:    array<u32>;
@group(0) @binding(3) var<storage, read>       delay_steps:    array<u32>;

// Buffer state (read-write).
@group(1) @binding(0) var<storage, read_write> buffer_pos:     array<f32>;
@group(1) @binding(1) var<storage, read_write> buffer_vel:     array<f32>;
@group(1) @binding(2) var<storage, read_write> num_pushes:     array<u32>;
@group(1) @binding(3) var<storage, read_write> write_idx_arr:  array<u32>;

// Outputs of the read pass (written by `delay_read`).
@group(2) @binding(0) var<storage, read_write> out_pos:        array<f32>;
@group(2) @binding(1) var<storage, read_write> out_vel:        array<f32>;

@group(3) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(64)
fn delay_push(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.actuator_count) { return; }

    let depth = uniforms.buf_depth;
    let cur = write_idx_arr[0];
    let new_idx = (cur + 1u) % depth;

    let dof = dof_indices[i];
    let row_off = new_idx * uniforms.actuator_count;
    buffer_pos[row_off + i] = target_pos[dof];
    buffer_vel[row_off + i] = target_vel[dof];

    let pc = num_pushes[i] + 1u;
    if (pc < depth) {
        num_pushes[i] = pc;
    } else {
        num_pushes[i] = depth;
    }

    if (i == 0u) {
        write_idx_arr[0] = new_idx;
    }
}

@compute @workgroup_size(64)
fn delay_read(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= uniforms.actuator_count) { return; }

    let depth = uniforms.buf_depth;
    let n     = num_pushes[i];
    let lag   = delay_steps[i];

    if (n == 0u || lag == 0u) {
        let dof = dof_indices[i];
        out_pos[i] = target_pos[dof];
        out_vel[i] = target_vel[dof];
        return;
    }

    var available = n;
    if (available > depth) { available = depth; }
    var clamped_lag = lag - 1u;
    if (clamped_lag > available - 1u) { clamped_lag = available - 1u; }

    let write_idx = write_idx_arr[0];
    let read_idx = (write_idx + depth - clamped_lag) % depth;
    let row_off = read_idx * uniforms.actuator_count;
    out_pos[i] = buffer_pos[row_off + i];
    out_vel[i] = buffer_vel[row_off + i];
}
