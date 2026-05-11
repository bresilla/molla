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
