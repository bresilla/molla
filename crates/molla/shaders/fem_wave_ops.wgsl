// Stage 36.5 Slice H — WGSL element-wise kernels for the GPU-resident
// Newmark-β wave stepper. Three small kernels keep `u`, `v`, `a`
// resident on the GPU across timesteps; CPU↔GPU traffic per step is
// reduced to **zero** (the viewer can read `u` once per render frame).
//
// Mirrors the CPU code in [`fem::solver::solve_wave_transient`]:
//
//   predictor:  u_pred[i] = u[i] + dt·v[i] + (½ − β)·dt²·a[i]
//   rhs:        b[i]      = (1/(β·dt²)) · (M·u_pred)[i] + bc_shift[i]
//                            for unpinned i; b[i] = pinned_value[i] for pinned.
//   corrector:  a_new[i]  = (u_new[i] − u_pred[i]) · (1/(β·dt²))
//               v_new[i]  = v_old[i] + dt · ((1−γ)·a_old[i] + γ·a_new[i])
//               then v_new, a_new are zeroed on every Dirichlet DoF.
//
// All ops are `f32` to match the rest of the FEM GPU stack
// (see `fem::gpu_data` for the precision rationale).
//
// Workgroup size: 256 — vector-style element-wise dispatch.

const WG_SIZE: u32 = 256u;

// ───────────────────────── wave_predictor ─────────────────────────
//
// `u_pred[i] = u[i] + dt·v[i] + half_minus_beta · dt² · a[i]`

struct WaveStepUniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    dt: f32,
    half_minus_beta_dt2: f32,
    bdt2_inv: f32,
    gamma: f32,
};

@group(0) @binding(0) var<storage, read>       pred_u    : array<f32>;
@group(0) @binding(1) var<storage, read>       pred_v    : array<f32>;
@group(0) @binding(2) var<storage, read>       pred_a    : array<f32>;
@group(0) @binding(3) var<storage, read_write> pred_out  : array<f32>;
@group(1) @binding(0) var<uniform>             pred_u_in : WaveStepUniforms;

@compute @workgroup_size(WG_SIZE)
fn wave_predictor(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= pred_u_in.n) {
        return;
    }
    pred_out[i] = pred_u[i]
        + pred_u_in.dt * pred_v[i]
        + pred_u_in.half_minus_beta_dt2 * pred_a[i];
}

// ───────────────────────── wave_rhs ───────────────────────────────
//
// For unpinned i:  b[i] = bdt2_inv · m_upred[i] + bc_shift[i]
// For pinned   i:  b[i] = pinned_value[i]
//
// `is_pinned` is a u32 buffer (1 = pinned, 0 = free). `pinned_value`
// is unused on free rows.

@group(0) @binding(0) var<storage, read>       rhs_m_upred      : array<f32>;
@group(0) @binding(1) var<storage, read>       rhs_bc_shift     : array<f32>;
@group(0) @binding(2) var<storage, read>       rhs_is_pinned    : array<u32>;
@group(0) @binding(3) var<storage, read>       rhs_pinned_value : array<f32>;
@group(0) @binding(4) var<storage, read_write> rhs_out          : array<f32>;
@group(1) @binding(0) var<uniform>             rhs_u_in         : WaveStepUniforms;

@compute @workgroup_size(WG_SIZE)
fn wave_rhs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= rhs_u_in.n) {
        return;
    }
    if (rhs_is_pinned[i] != 0u) {
        rhs_out[i] = rhs_pinned_value[i];
    } else {
        rhs_out[i] = rhs_u_in.bdt2_inv * rhs_m_upred[i] + rhs_bc_shift[i];
    }
}

// ───────────────────────── wave_corrector ─────────────────────────
//
// a_new[i] = (u_new[i] − u_pred[i]) · bdt2_inv
// v_new[i] = v_old[i] + dt · ((1−γ)·a_old[i] + γ·a_new[i])
// then on every pinned DoF: v_new[i] = 0; a_new[i] = 0.
//
// In-place semantics: `corr_v` is read as v_old + written as v_new;
// `corr_a` is read as a_old + written as a_new. Safe because we read
// the old value into a register before writing the new one.

@group(0) @binding(0) var<storage, read>       corr_u_new    : array<f32>;
@group(0) @binding(1) var<storage, read>       corr_u_pred   : array<f32>;
@group(0) @binding(2) var<storage, read_write> corr_v        : array<f32>;
@group(0) @binding(3) var<storage, read_write> corr_a        : array<f32>;
@group(0) @binding(4) var<storage, read>       corr_is_pinned: array<u32>;
@group(1) @binding(0) var<uniform>             corr_u_in     : WaveStepUniforms;

@compute @workgroup_size(WG_SIZE)
fn wave_corrector(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= corr_u_in.n) {
        return;
    }
    let v_old = corr_v[i];
    let a_old = corr_a[i];
    let a_new = (corr_u_new[i] - corr_u_pred[i]) * corr_u_in.bdt2_inv;
    let v_new = v_old + corr_u_in.dt
        * ((1.0 - corr_u_in.gamma) * a_old + corr_u_in.gamma * a_new);
    if (corr_is_pinned[i] != 0u) {
        corr_v[i] = 0.0;
        corr_a[i] = 0.0;
    } else {
        corr_v[i] = v_new;
        corr_a[i] = a_new;
    }
}

// ───────────────────────── wave_copy_u ────────────────────────────
//
// `u_dst[i] = u_src[i]` for `i < n`. Used to copy the GPU CG result
// (`x` buffer) into the persistent `u` storage at the end of step.
// Defined here so the WaveStepOpsKernel doesn't need a separate
// pipeline; an axpy with α=0 + read-of-self would also work but a
// dedicated copy is clearer and saves the read.

@group(0) @binding(0) var<storage, read>       copy_src : array<f32>;
@group(0) @binding(1) var<storage, read_write> copy_dst : array<f32>;
@group(1) @binding(0) var<uniform>             copy_u   : WaveStepUniforms;

@compute @workgroup_size(WG_SIZE)
fn wave_copy_u(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= copy_u.n) {
        return;
    }
    copy_dst[i] = copy_src[i];
}
