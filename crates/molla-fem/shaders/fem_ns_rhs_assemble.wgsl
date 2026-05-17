// Stage 41 v4 — fused RHS assembly for the IMEX Navier-Stokes time
// stepper. Replaces the per-step host roundtrip that Stage 41 v3 paid
// for `m_u`, `f_conv`, and `bt_p`. The host stays out of the inner loop.
//
// Two entry points share one shader so they can co-locate the pin
// override logic and the constant uniform layout:
//
//   * `assemble_step_rhs` — writes
//       rhs[i] = inv_dt * m_u[i] + f_ext[i] - f_conv[i] - bc_delta[i]
//     then overrides rhs[i] = pin_val[i] where pin_mask[i] != 0.
//
//   * `assemble_recovery_rhs` — writes
//       rhs[i] = rhs_old[i] - bt_p[i]
//     then overrides rhs[i] = pin_val[i] where pin_mask[i] != 0.
//
// Layout:
//   group(0):  the four GPU-resident sources (the LHS RHS and step path
//             both pull from these; the recovery path remaps two of
//             them via different binding semantics).
//   group(1):  pin mask + pin values (constant for the cavity / TG BC
//             sets — uploaded once at setup).
//   group(2):  uniforms (n_v, inv_dt).
//
// One workgroup per 64 DoFs.

struct RhsUniforms {
    n_v: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    inv_dt: f32,
    weight_a: f32,
    weight_b: f32,
    _pad5: f32,
};

// Group 0 — per-step inputs.
// For `assemble_step_rhs`:
//   src_a = m_u       (M · u_old)
//   src_b = f_ext     (external body force, constant per step)
//   src_c = f_conv    (GPU-assembled convection RHS)
//   src_d = bc_delta  (BC column-elim correction, constant)
// For `assemble_recovery_rhs`:
//   src_a = rhs_old   (the step RHS already on the device)
//   src_b = bt_p      (Bᵀ · p_new from the Schur solve)
//   src_c, src_d = unused (bound to dummy buffer by host).
// For `combine_states`:
//   src_a = first state
//   src_b = second state
//   src_c, src_d = unused (bound to dummy buffer by host).
@group(0) @binding(0) var<storage, read> src_a : array<f32>;
@group(0) @binding(1) var<storage, read> src_b : array<f32>;
@group(0) @binding(2) var<storage, read> src_c : array<f32>;
@group(0) @binding(3) var<storage, read> src_d : array<f32>;

// Group 1 — pin mask / values (constant after setup).
@group(1) @binding(0) var<storage, read> pin_mask : array<u32>;
@group(1) @binding(1) var<storage, read> pin_val  : array<f32>;

// Group 2 — destination + uniforms.
@group(2) @binding(0) var<storage, read_write> rhs : array<f32>;
@group(2) @binding(1) var<uniform> u : RhsUniforms;

const WG_SIZE: u32 = 64u;

@compute @workgroup_size(WG_SIZE)
fn assemble_step_rhs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n_v) {
        return;
    }
    let m_u_i    = src_a[i];
    let f_ext_i  = src_b[i];
    let f_conv_i = src_c[i];
    let bc_delta_i = src_d[i];

    var value = u.inv_dt * m_u_i + f_ext_i - f_conv_i - bc_delta_i;
    if (pin_mask[i] != 0u) {
        value = pin_val[i];
    }
    rhs[i] = value;
}

@compute @workgroup_size(WG_SIZE)
fn assemble_recovery_rhs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n_v) {
        return;
    }
    let rhs_old_i = src_a[i];
    let bt_p_i    = src_b[i];

    var value = rhs_old_i - bt_p_i;
    if (pin_mask[i] != 0u) {
        value = pin_val[i];
    }
    rhs[i] = value;
}

// Stage 41 v4 — in-place zero-mask: `rhs[i] = 0` where `pin_mask[i] != 0`.
// Replaces the v3 host loop that issued one `queue.write_buffer(zero)`
// per pinned DoF inside the Uzawa Schur inner loop (3 calls × ~hundreds
// of pinned DoFs per Schur op — a measurable share of small-N
// wall-clock). The `src_*` group(0) bindings are unused; the host binds
// dummy buffers to satisfy the shared pipeline layout.
@compute @workgroup_size(WG_SIZE)
fn apply_zero_mask(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n_v) {
        return;
    }
    if (pin_mask[i] != 0u) {
        rhs[i] = 0.0;
    }
}

// Stage 41 v7 — on-device SSP-RK state combination. The host supplies
// convex weights for the two stage buffers and this kernel writes the
// combined velocity/pressure vector without any stage readback. For
// velocity vectors, pinned values are restored from the constant pin
// buffer; for pressure vectors callers bind a zero pin mask.
@compute @workgroup_size(WG_SIZE)
fn combine_states(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n_v) {
        return;
    }
    var value = u.weight_a * src_a[i] + u.weight_b * src_b[i];
    if (pin_mask[i] != 0u) {
        value = pin_val[i];
    }
    rhs[i] = value;
}
