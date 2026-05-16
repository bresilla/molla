// Stage 36.5 Slice H — WGSL element-wise kernels for the GPU-resident
// backward-Euler transient (heat) stepper. Mirrors the predictor /
// corrector free path: we only need an RHS-construction kernel since
// the transient step has no integrator state beyond `u_old`.
//
// CPU formula (see [`fem::solver::solve_transient`]):
//
//   b[i] = f[i] + (1/dt) · (M · u_old)[i] + bc_shift[i]   (unpinned)
//   b[i] = pinned_value[i]                                (pinned)
//
// All ops are `f32` to match the rest of the FEM GPU stack.
//
// Workgroup size: 256.

const WG_SIZE: u32 = 256u;

struct TransientStepUniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    inv_dt: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
};

// ───────────────────────── transient_rhs ──────────────────────────

@group(0) @binding(0) var<storage, read>       trh_m_uold       : array<f32>;
@group(0) @binding(1) var<storage, read>       trh_f            : array<f32>;
@group(0) @binding(2) var<storage, read>       trh_bc_shift     : array<f32>;
@group(0) @binding(3) var<storage, read>       trh_is_pinned    : array<u32>;
@group(0) @binding(4) var<storage, read>       trh_pinned_value : array<f32>;
@group(0) @binding(5) var<storage, read_write> trh_out          : array<f32>;
@group(1) @binding(0) var<uniform>             trh_u_in         : TransientStepUniforms;

@compute @workgroup_size(WG_SIZE)
fn transient_rhs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= trh_u_in.n) {
        return;
    }
    if (trh_is_pinned[i] != 0u) {
        trh_out[i] = trh_pinned_value[i];
    } else {
        trh_out[i] = trh_f[i] + trh_u_in.inv_dt * trh_m_uold[i] + trh_bc_shift[i];
    }
}

// ───────────────────────── transient_copy_u ───────────────────────
//
// `u_dst[i] = u_src[i]` — same role as `wave_copy_u`. Used to copy
// the GPU CG result into the persistent `u` storage at the end of
// step.

@group(0) @binding(0) var<storage, read>       trc_src : array<f32>;
@group(0) @binding(1) var<storage, read_write> trc_dst : array<f32>;
@group(1) @binding(0) var<uniform>             trc_u   : TransientStepUniforms;

@compute @workgroup_size(WG_SIZE)
fn transient_copy_u(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= trc_u.n) {
        return;
    }
    trc_dst[i] = trc_src[i];
}
