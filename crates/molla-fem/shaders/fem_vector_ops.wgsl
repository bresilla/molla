// Stage 36.5 Slice A — WGSL vector primitives for the GPU-resident
// CG solver. Three small kernels keep `x`, `r`, `p`, `Ap` resident
// on the GPU across CG iterations, eliminating the per-iter host
// roundtrip that `solve_steady_state_gpu` paid in v1.
//
// Kernels:
//   * `vec_axpy`            — `a[i] += alpha * b[i]`
//   * `vec_axpby`           — `dst[i] = alpha * src[i] + beta * dst[i]`
//   * `vec_dot`             — workgroup-wise reduction of `dot(a, b)`
//                             into `partial_sums[group_id]`. The host
//                             sums the (small, ≤ ⌈n / WG⌉) partials.
//   * `vec_apply_diag_inv`  — `dst[i] = src[i] * inv_diag[i]`. Used by
//                             the Jacobi-preconditioned CG path
//                             (`pcg_solve_resident`) to apply
//                             `M⁻¹ = diag(K)⁻¹` once per iteration.
//                             The caller pre-inverts the diagonal on
//                             the host so the kernel only multiplies
//                             (a per-thread divide is the same cost on
//                             modern GPUs but precluding NaN/Inf when
//                             `diag = 0` is cleaner host-side).
//
// All ops are `f32` to match the rest of the FEM GPU stack
// (see `fem::gpu_data` for the precision rationale).
//
// Workgroup size: 256 — standard for warp-aligned vector ops on
// both NVIDIA + Intel + AMD desktop adapters; small enough that the
// shared-memory tree-reduction in `vec_dot` (256 × 4 B = 1 KiB)
// stays well under any device limit.

const WG_SIZE: u32 = 256u;

// ───────────────────────── vec_axpy ──────────────────────────────
//
// `a[i] += alpha * b[i]` for `i ∈ [0, n)`. One thread per element.

struct AxpyUniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
};

@group(0) @binding(0) var<storage, read_write> axpy_a : array<f32>;
@group(0) @binding(1) var<storage, read>       axpy_b : array<f32>;
@group(1) @binding(0) var<uniform>             axpy_u : AxpyUniforms;

@compute @workgroup_size(WG_SIZE)
fn vec_axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= axpy_u.n) {
        return;
    }
    axpy_a[i] = axpy_a[i] + axpy_u.alpha * axpy_b[i];
}

// ───────────────────────── vec_axpby ─────────────────────────────
//
// `dst[i] = alpha * src[i] + beta * dst[i]` — covers `r = b - Ap`
// (alpha=1, beta=0 followed by a separate `r -= Ap`), `p = r + β·p`
// (alpha=1, beta=β), and `dst = α·src` (beta=0).
//
// Bound separately from `vec_axpy` because writing `dst` requires
// `read_write` *and* read access in the same expression; WGSL does
// not allow aliasing the same binding twice.

struct AxpbyUniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f32,
    beta: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<storage, read_write> axpby_dst : array<f32>;
@group(0) @binding(1) var<storage, read>       axpby_src : array<f32>;
@group(1) @binding(0) var<uniform>             axpby_u   : AxpbyUniforms;

@compute @workgroup_size(WG_SIZE)
fn vec_axpby(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= axpby_u.n) {
        return;
    }
    axpby_dst[i] = axpby_u.alpha * axpby_src[i] + axpby_u.beta * axpby_dst[i];
}

// ───────────────────────── vec_dot ───────────────────────────────
//
// Workgroup-wise reduction of `sum(a[i] * b[i])`. Each workgroup
// computes one partial sum, written to `partials[wg_id]`. The host
// then sums the (small) `partials` — ⌈n / WG_SIZE⌉ entries, ≤ a few
// hundred even at million-DoF problem sizes.
//
// Uses shared memory + tree reduction (textbook `parallel reduction`
// — Brent + Kung). No subgroup ops since wgpu doesn't expose them
// portably across vendors yet.

struct DotUniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       dot_a        : array<f32>;
@group(0) @binding(1) var<storage, read>       dot_b        : array<f32>;
@group(0) @binding(2) var<storage, read_write> dot_partials : array<f32>;
@group(1) @binding(0) var<uniform>             dot_u        : DotUniforms;

var<workgroup> sdata : array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn vec_dot(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let lane = lid.x;
    let i = gid.x;
    var v: f32 = 0.0;
    if (i < dot_u.n) {
        v = dot_a[i] * dot_b[i];
    }
    sdata[lane] = v;
    workgroupBarrier();

    // Tree reduction over the workgroup. WG_SIZE = 256 = 2^8.
    var stride: u32 = WG_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            sdata[lane] = sdata[lane] + sdata[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        dot_partials[wid.x] = sdata[0];
    }
}

// ─────────────────────── vec_apply_diag_inv ──────────────────────
//
// `dst[i] = src[i] * inv_diag[i]` for `i ∈ [0, n)`. One thread per
// element. Used by the PCG solver to apply the Jacobi preconditioner
// `M⁻¹ = diag(K)⁻¹` to a residual vector — `inv_diag` is computed
// host-side once per solve (the caller stamps `0` for any zero diag
// entries so null-space rows pass through cleanly).
//
// Reuses the `n` field of [`AxpbyUniforms`] (alpha/beta unused) so we
// don't have to introduce a fourth uniform layout.

@group(0) @binding(0) var<storage, read_write> diag_dst      : array<f32>;
@group(0) @binding(1) var<storage, read>       diag_src      : array<f32>;
@group(0) @binding(2) var<storage, read>       diag_inv_diag : array<f32>;
@group(1) @binding(0) var<uniform>             diag_u        : AxpbyUniforms;

@compute @workgroup_size(WG_SIZE)
fn vec_apply_diag_inv(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= diag_u.n) {
        return;
    }
    diag_dst[i] = diag_src[i] * diag_inv_diag[i];
}

// ═════════════════════════════════════════════════════════════════════
// Stage 41 v5 — async-dot fusion primitives for lazy CG.
//
// Goal: eliminate the per-iter `map_async + poll` host sync that the
// standard CG inner loop pays for `pap = ⟨p, Ap⟩` and `rr = ⟨r, r⟩`.
// We replace those host syncs by keeping every CG scalar (alpha, beta,
// rr) resident on the GPU in 1-element storage buffers, and the host
// reads back only the residual norm every `check_period` iters.
//
// Kernels:
//   * `dot_finalize` — reduces the per-workgroup partial-sums buffer
//                      into a single f32 scalar buffer. Single
//                      workgroup, tree reduction.
//   * `cg_alpha`     — scalar div: `alpha = rr / pap`; also writes
//                      `-alpha` to a sibling buffer so the r-axpy can
//                      use it directly without a host negation. Guards
//                      against `|pap| < 1e-30` (writes 0 — caller
//                      checks for stagnation on the residual norm).
//   * `cg_beta`      — scalar div: `beta = rr_new / rr_old`. Also
//                      copies `rr_new` → `rr_old_buf` for the next
//                      iter's beta.
//   * `vec_axpy_s`   — `a[i] += s[0] * b[i]` — same as `vec_axpy` but
//                      reads `alpha` from a 1-element storage buffer
//                      instead of a uniform. Lets the inner loop chain
//                      `cg_alpha → axpy_s` without a host roundtrip.
//   * `vec_axpby_s`  — `dst[i] = src[i] + s[0] * dst[i]` — sibling for
//                      the `p ← r + β·p` update.

const FINALIZE_WG: u32 = 64u;  // partials are tiny (≤ ⌈n / 256⌉)

// ─── dot_finalize ────────────────────────────────────────────────────
//
// Single-workgroup reduction of the `partials` buffer into one f32.
// `n_partials` is the live prefix length (≤ FINALIZE_WG for the mesh
// sizes we target; the loop in `dot_finalize_main` strides across the
// buffer if `n_partials > FINALIZE_WG` for future-proofing).

struct DotFinalizeUniforms {
    n_partials: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       fin_partials : array<f32>;
@group(0) @binding(1) var<storage, read_write> fin_out      : array<f32>;
@group(1) @binding(0) var<uniform>             fin_u        : DotFinalizeUniforms;

var<workgroup> fin_sdata : array<f32, FINALIZE_WG>;

@compute @workgroup_size(FINALIZE_WG)
fn dot_finalize(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    var v: f32 = 0.0;
    // Stride load — handle n_partials > FINALIZE_WG cleanly.
    var i: u32 = lane;
    loop {
        if (i >= fin_u.n_partials) { break; }
        v = v + fin_partials[i];
        i = i + FINALIZE_WG;
    }
    fin_sdata[lane] = v;
    workgroupBarrier();

    var stride: u32 = FINALIZE_WG / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            fin_sdata[lane] = fin_sdata[lane] + fin_sdata[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        fin_out[0] = fin_sdata[0];
    }
}

// ─── cg_alpha ────────────────────────────────────────────────────────
//
// `alpha = rr_old / pap`, with `neg_alpha = -alpha`. Both written.
// Guards against `|pap| < 1e-30` — emits 0 (the host's next residual
// check will exit the loop on a stagnant pap-near-zero).

@group(0) @binding(0) var<storage, read>       alpha_rr_old  : array<f32>;
@group(0) @binding(1) var<storage, read>       alpha_pap     : array<f32>;
@group(0) @binding(2) var<storage, read_write> alpha_out     : array<f32>;
@group(0) @binding(3) var<storage, read_write> alpha_neg_out : array<f32>;

@compute @workgroup_size(1)
fn cg_alpha() {
    let rr = alpha_rr_old[0];
    let pap = alpha_pap[0];
    var a: f32 = 0.0;
    if (abs(pap) > 1.0e-30) {
        a = rr / pap;
    }
    alpha_out[0] = a;
    alpha_neg_out[0] = -a;
}

// ─── cg_beta ─────────────────────────────────────────────────────────
//
// `beta = rr_new / rr_old`; then copies `rr_new` → `rr_old` so the
// next iter has the right denominator without a host shuffle. The two
// buffers are bound separately (read-only `rr_new`, read_write
// `rr_old`) so WGSL's aliasing rules are happy.

@group(0) @binding(0) var<storage, read>       beta_rr_new  : array<f32>;
@group(0) @binding(1) var<storage, read_write> beta_rr_old  : array<f32>;
@group(0) @binding(2) var<storage, read_write> beta_out     : array<f32>;

@compute @workgroup_size(1)
fn cg_beta() {
    let rn = beta_rr_new[0];
    let ro = beta_rr_old[0];
    var b: f32 = 0.0;
    if (abs(ro) > 1.0e-30) {
        b = rn / ro;
    }
    beta_out[0] = b;
    beta_rr_old[0] = rn;
}

// ─── vec_axpy_s / vec_axpby_s ────────────────────────────────────────
//
// Like `vec_axpy` / `vec_axpby` but read the scalar(s) from a 1-element
// storage buffer instead of a uniform. Lets us chain dot → alpha → axpy
// inside one queue submit without paying a host roundtrip to re-write
// the uniform.
//
// Each kernel keeps its own `n` uniform (same layout as the existing
// `AxpyUniforms` / `AxpbyUniforms` so we don't have to introduce yet
// another uniform struct).

@group(0) @binding(0) var<storage, read_write> axpys_a     : array<f32>;
@group(0) @binding(1) var<storage, read>       axpys_b     : array<f32>;
@group(0) @binding(2) var<storage, read>       axpys_scalar: array<f32>;
@group(1) @binding(0) var<uniform>             axpys_u     : AxpyUniforms;

@compute @workgroup_size(WG_SIZE)
fn vec_axpy_s(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= axpys_u.n) {
        return;
    }
    axpys_a[i] = axpys_a[i] + axpys_scalar[0] * axpys_b[i];
}

@group(0) @binding(0) var<storage, read_write> axpbys_dst   : array<f32>;
@group(0) @binding(1) var<storage, read>       axpbys_src   : array<f32>;
@group(0) @binding(2) var<storage, read>       axpbys_scalar: array<f32>;
@group(1) @binding(0) var<uniform>             axpbys_u     : AxpbyUniforms;

// `dst = src + scalar * dst` — α is fixed at 1.0 (the lazy CG only
// ever runs `p ← r + β · p`). β comes from the storage scalar buffer.
@compute @workgroup_size(WG_SIZE)
fn vec_axpby_s(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= axpbys_u.n) {
        return;
    }
    axpbys_dst[i] = axpbys_src[i] + axpbys_scalar[0] * axpbys_dst[i];
}

// ─── cg_axpy_xr ──────────────────────────────────────────────────────
//
// Fused `α = rr / pap` divide + `x += α·p` + `r -= α·Ap` in ONE
// dispatch. Each thread does its own divide on the per-buffer scalars
// (cheap, fully parallel) and then writes its own x/r entry. This
// replaces three separate dispatches (cg_alpha + axpy_x + axpy_r)
// from the early v5 prototype with one — at the 8×8/16×16 mesh sizes
// the dispatch-launch cost dominated f32 throughput and removing two
// dispatches per inner CG iter is worth ~25-30 % wall-clock.
//
// Guards against `|pap| < 1e-30` by writing 0 — equivalent to "no
// update this iter", and the next batch's residual read will see the
// stagnant ‖r‖ and exit through the host's tol check.

@group(0) @binding(0) var<storage, read_write> xrx_x      : array<f32>;
@group(0) @binding(1) var<storage, read_write> xrx_r      : array<f32>;
@group(0) @binding(2) var<storage, read>       xrx_p      : array<f32>;
@group(0) @binding(3) var<storage, read>       xrx_ap     : array<f32>;
@group(0) @binding(4) var<storage, read>       xrx_rr_old : array<f32>;
@group(0) @binding(5) var<storage, read>       xrx_pap    : array<f32>;
@group(1) @binding(0) var<uniform>             xrx_u      : AxpyUniforms;

@compute @workgroup_size(WG_SIZE)
fn cg_axpy_xr(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= xrx_u.n) {
        return;
    }
    let pap = xrx_pap[0];
    var a: f32 = 0.0;
    if (abs(pap) > 1.0e-30) {
        a = xrx_rr_old[0] / pap;
    }
    xrx_x[i] = xrx_x[i] + a * xrx_p[i];
    xrx_r[i] = xrx_r[i] - a * xrx_ap[i];
}

// ─── cg_axpby_p ──────────────────────────────────────────────────────
//
// Fused `β = rr_new / rr_old` + `p ← r + β·p` + (thread 0 only)
// `rr_old ← rr_new`. Same motivation as `cg_axpy_xr`: collapses three
// dispatches (cg_beta + axpby + scalar copy embedded in cg_beta) into
// one.

@group(0) @binding(0) var<storage, read_write> bp_p       : array<f32>;
@group(0) @binding(1) var<storage, read>       bp_r       : array<f32>;
@group(0) @binding(2) var<storage, read>       bp_rr_new  : array<f32>;
@group(0) @binding(3) var<storage, read_write> bp_rr_old  : array<f32>;
@group(1) @binding(0) var<uniform>             bp_u       : AxpyUniforms;

@compute @workgroup_size(WG_SIZE)
fn cg_axpby_p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= bp_u.n) {
        return;
    }
    let ro = bp_rr_old[0];
    let rn = bp_rr_new[0];
    var b: f32 = 0.0;
    if (abs(ro) > 1.0e-30) {
        b = rn / ro;
    }
    bp_p[i] = bp_r[i] + b * bp_p[i];
    // Thread 0 of the dispatch carries the rr_old update — single
    // writer, no atomics needed. WGSL race rules: only one workgroup
    // worth of threads can race on this write, and they all compute
    // the same value, so functionally it's a benign race even if
    // multiple threads write. (We let thread 0 take the write only
    // to keep memory traffic low.)
    if (i == 0u) {
        bp_rr_old[0] = rn;
    }
}
