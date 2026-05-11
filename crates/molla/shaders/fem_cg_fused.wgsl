// Stage 41 v6 — fused CG iter kernels.
//
// Cuts the per-iter dispatch count from v5's 7 (matvec + dot + finalize
// + axpy_xr + dot + finalize + axpby_p) down to **3**:
//
//   1. `bsr_matvec_pap_fused`     — BSR matvec `Ap = A·p` AND, in the
//                                    same dispatch, atomic-CAS-adds each
//                                    row's `p[row] * Ap[row]` into the
//                                    scalar `pap_atomic`. Replaces v5's
//                                    (matvec + dot + finalize).
//   2. `cg_alpha_xr_rr_fused`     — reads `pap_atomic` + `rr_old`,
//                                    derives α = rr / pap, does
//                                    `x += α·p; r -= α·Ap`, then
//                                    workgroup-reduces `r[i]²` and
//                                    atomic-CAS-adds to `rr_new_atomic`.
//                                    Replaces v5's (axpy_xr + dot +
//                                    finalize).
//   3. `cg_axpby_p_with_save`     — `β = rr_new / rr_old`; `p ← r + β·p`;
//                                    saves `rr_old ← rr_new`. Reads
//                                    `rr_new` from `atomic<u32>` so it
//                                    chains directly off kernel 2.
//
// Atomic dot reductions use the `bitcast<f32>(atomic<u32>)` CAS-loop
// pattern (same as `bsr_spgemm_symbolic.wgsl`'s `atomic_add_f32`).
// WGSL does NOT expose f32 atomics portably; CAS via u32 is the canonical
// fallback. CAS contention is bounded: only one thread per workgroup
// performs the atomic-add (the workgroup-local tree-reduce produces a
// single f32 per workgroup), so the contention is `n_workgroups` writes
// per dot, ≤ a few dozen even at 16×16 cavity sizes.
//
// **Pre-zeroing**: the host issues `encoder.clear_buffer(pap_atomic, ..)`
// and `encoder.clear_buffer(rr_atomic, ..)` between iters. Transfer
// commands, not compute dispatches — much cheaper than a 1-thread zero
// kernel. The clears happen between compute passes; wgpu inserts the
// necessary barriers.

const WG_SIZE: u32 = 256u;
const MAX_DIM: u32 = 65535u;

// ──────────────────────────────────────────────────────────────────────
// Kernel 1: `bsr_matvec_pap_fused`
//
// `Ap = A · p` (BSR matvec, one workgroup per row, WG-local tree reduce)
// PLUS atomic-CAS-add of `p[row] * Ap[row]` into `pap_atomic[0]`.
//
// Bindings (group 0 = io):
//   0  row_offsets : array<u32>           — BSR row offsets
//   1  col_indices : array<u32>           — BSR col indices
//   2  values      : array<f32>           — BSR values
//   3  p           : array<f32>           — CG search direction (x in matvec)
//   4  ap          : array<f32, rw>       — output Ap (y in matvec)
//   5  pap_atomic  : array<atomic<u32>>   — scalar `pap` accumulator
//                                          (bit-cast f32 via atomic CAS)
//
// Bindings (group 1 = uniforms):
//   0  u : MvUniforms { nrows; alpha; beta; }
//
// Dispatch grid: same 2D `(min(nrows, 65535), ⌈nrows / 65535⌉)` layout
// as `bsr_mv_gpu.wgsl` so we can lift past the 65535-per-dim cap.
// ──────────────────────────────────────────────────────────────────────

struct MvUniforms {
    nrows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f32,
    beta:  f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<storage, read>            mv_row_offsets : array<u32>;
@group(0) @binding(1) var<storage, read>            mv_col_indices : array<u32>;
@group(0) @binding(2) var<storage, read>            mv_values      : array<f32>;
@group(0) @binding(3) var<storage, read>            mv_p           : array<f32>;
@group(0) @binding(4) var<storage, read_write>      mv_ap          : array<f32>;
@group(0) @binding(5) var<storage, read_write>      pap_atomic     : array<atomic<u32>>;
@group(1) @binding(0) var<uniform>                  mv_u           : MvUniforms;

var<workgroup> mv_row_acc : array<f32, WG_SIZE>;

// f32 atomic-add via CAS — WGSL exposes only `atomic<u32>` / `atomic<i32>`.
// We bitcast back-and-forth. Native f32 atomics would be faster on
// Volta+/RDNA2+ adapters but are not portably exposed by wgpu.
fn atomic_add_f32_pap(addend: f32) {
    var old_u = atomicLoad(&pap_atomic[0]);
    loop {
        let old_f = bitcast<f32>(old_u);
        let new_f = old_f + addend;
        let new_u = bitcast<u32>(new_f);
        let res = atomicCompareExchangeWeak(&pap_atomic[0], old_u, new_u);
        if (res.exchanged) { break; }
        old_u = res.old_value;
    }
}

@compute @workgroup_size(WG_SIZE)
fn bsr_matvec_pap_fused(
    @builtin(workgroup_id)        wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.y * MAX_DIM + wid.x;
    let lane = lid.x;
    if (row >= mv_u.nrows) {
        return;
    }
    let lo = mv_row_offsets[row];
    let hi = mv_row_offsets[row + 1u];

    var local_sum: f32 = 0.0;
    for (var s: u32 = lo + lane; s < hi; s = s + WG_SIZE) {
        let col = mv_col_indices[s];
        local_sum = local_sum + mv_values[s] * mv_p[col];
    }
    mv_row_acc[lane] = local_sum;
    workgroupBarrier();

    var stride: u32 = WG_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            mv_row_acc[lane] = mv_row_acc[lane] + mv_row_acc[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        let row_val = mv_u.alpha * mv_row_acc[0] + mv_u.beta * mv_ap[row];
        mv_ap[row] = row_val;
        // Atomic-add p[row] * Ap[row] to pap_atomic. One CAS per row
        // (per WG, since only thread 0 fires). Contention = nrows.
        let prod = mv_p[row] * row_val;
        atomic_add_f32_pap(prod);
    }
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 2: `cg_alpha_xr_rr_fused`
//
// Reads `pap_in[0]` + `rr_old[0]`, derives α = rr_old / pap, does
// `x[i] += α·p[i]; r[i] -= α·Ap[i]`, then workgroup-reduces `r[i]²` and
// atomic-CAS-adds to `rr_out[0]`.
//
// Bindings (group 0):
//   0  x        : array<f32, rw>           — CG iterate, updated in-place
//   1  r        : array<f32, rw>           — CG residual, updated in-place
//   2  p        : array<f32>                — CG search direction
//   3  ap       : array<f32>                — A·p
//   4  pap_in   : array<atomic<u32>>        — scalar pap (bit-cast f32)
//   5  rr_old   : array<f32>                — previous ‖r‖²
//   6  rr_out   : array<atomic<u32>>        — scalar rr_new accumulator
//
// Bindings (group 1):
//   0  u : XrUniforms { n; pad; }
//
// Dispatch grid: 1D, `⌈n / WG_SIZE⌉` workgroups, WG_SIZE threads each.
//
// Each WG does element-wise updates for its slice of indices, then
// tree-reduces `r[i]²` into a single per-WG sum. Thread 0 of each WG
// atomic-CAS-adds that sum to `rr_out`. Contention bounded by
// `⌈n / WG_SIZE⌉` (≤ 10 at 16×16).
//
// **Guard**: if `|pap| < 1e-30`, α = 0 → x, r unchanged. The host
// observes a stagnant ‖r‖ next iter and exits the batch loop on tol.
// ──────────────────────────────────────────────────────────────────────

struct XrUniforms {
    n:    u32,
    _pad0:u32,
    _pad1:u32,
    _pad2:u32,
};

@group(0) @binding(0) var<storage, read_write>      xr_x       : array<f32>;
@group(0) @binding(1) var<storage, read_write>      xr_r       : array<f32>;
@group(0) @binding(2) var<storage, read>            xr_p       : array<f32>;
@group(0) @binding(3) var<storage, read>            xr_ap      : array<f32>;
@group(0) @binding(4) var<storage, read_write>      xr_pap_in  : array<atomic<u32>>;
@group(0) @binding(5) var<storage, read>            xr_rr_old  : array<f32>;
@group(0) @binding(6) var<storage, read_write>      xr_rr_out  : array<atomic<u32>>;
@group(1) @binding(0) var<uniform>                  xr_u       : XrUniforms;

var<workgroup> xr_sdata : array<f32, WG_SIZE>;

fn atomic_add_f32_xr_rr(addend: f32) {
    var old_u = atomicLoad(&xr_rr_out[0]);
    loop {
        let old_f = bitcast<f32>(old_u);
        let new_f = old_f + addend;
        let new_u = bitcast<u32>(new_f);
        let res = atomicCompareExchangeWeak(&xr_rr_out[0], old_u, new_u);
        if (res.exchanged) { break; }
        old_u = res.old_value;
    }
}

@compute @workgroup_size(WG_SIZE)
fn cg_alpha_xr_rr_fused(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    let i    = gid.x;
    let lane = lid.x;
    let n    = xr_u.n;

    // Read scalars — every thread does the same load + divide. The
    // compiler hoists into a uniform — single FP-divide per WG warp.
    let pap_u = atomicLoad(&xr_pap_in[0]);
    let pap_f = bitcast<f32>(pap_u);
    let ro    = xr_rr_old[0];
    var alpha: f32 = 0.0;
    if (abs(pap_f) > 1.0e-30) {
        alpha = ro / pap_f;
    }

    // Element-wise update.
    var r_local: f32 = 0.0;
    if (i < n) {
        let pi  = xr_p[i];
        let api = xr_ap[i];
        let xi  = xr_x[i] + alpha * pi;
        let ri  = xr_r[i] - alpha * api;
        xr_x[i] = xi;
        xr_r[i] = ri;
        r_local = ri * ri;
    }

    // WG-local tree reduce of r².
    xr_sdata[lane] = r_local;
    workgroupBarrier();
    var stride: u32 = WG_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            xr_sdata[lane] = xr_sdata[lane] + xr_sdata[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    // One atomic-CAS-add per WG.
    if (lane == 0u) {
        atomic_add_f32_xr_rr(xr_sdata[0]);
    }
}

// ──────────────────────────────────────────────────────────────────────
// Kernel 3: `cg_axpby_p_with_save`
//
// `β = rr_new / rr_old; p[i] = r[i] + β · p[i]; rr_old ← rr_new`.
//
// Mirror of v5's `cg_axpby_p` (in `fem_vector_ops.wgsl`) but reads
// `rr_new` from an atomic<u32> (bit-cast f32) instead of a plain f32
// storage buffer, so the kernel can chain directly off
// `cg_alpha_xr_rr_fused`'s atomic write without an intervening
// finalize/copy dispatch.
//
// Bindings (group 0):
//   0  p          : array<f32, rw>
//   1  r          : array<f32>
//   2  rr_new_in  : array<atomic<u32>>  — bit-cast f32, from kernel 2
//   3  rr_old     : array<f32, rw>      — overwritten with rr_new at end
//
// Bindings (group 1):
//   0  u : XrUniforms { n; pad; }
// ──────────────────────────────────────────────────────────────────────

@group(0) @binding(0) var<storage, read_write>      bp_p       : array<f32>;
@group(0) @binding(1) var<storage, read>            bp_r       : array<f32>;
@group(0) @binding(2) var<storage, read_write>      bp_rr_new  : array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write>      bp_rr_old  : array<f32>;
@group(1) @binding(0) var<uniform>                  bp_u       : XrUniforms;

@compute @workgroup_size(WG_SIZE)
fn cg_axpby_p_with_save(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bp_u.n) {
        return;
    }
    let ro = bp_rr_old[0];
    let rn_u = atomicLoad(&bp_rr_new[0]);
    let rn = bitcast<f32>(rn_u);
    var b: f32 = 0.0;
    if (abs(ro) > 1.0e-30) {
        b = rn / ro;
    }
    bp_p[i] = bp_r[i] + b * bp_p[i];
    // Thread 0 saves rr_old ← rr_new. Single-writer; benign race with
    // any other thread that might do the same (all compute the same
    // value). We let only thread 0 write to minimise memory traffic.
    if (i == 0u) {
        bp_rr_old[0] = rn;
    }
}
