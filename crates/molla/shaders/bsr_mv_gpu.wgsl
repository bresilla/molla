// Stage 38a — generalised sparse `y = α·A·x + β·y`.
//
// One workgroup per row, tree-reduction over the row's nonzeros within
// the workgroup. Mirrors `fem_matvec.wgsl` for the row-walk pattern but
// adds the (α, β) BLAS-style coefficients so the kernel can serve as
// the generic BSR matvec primitive across the FEM stack and any future
// optimisation pipelines that need α·A·x updates.
//
// `α` and `β` come from the uniform block (NOT the values buffer) so
// the caller can sweep coefficients without rewriting the matrix.
// `β = 0` matches the FEM matvec contract; `β = 1, α = -1` is the
// CG residual update `r ← b - A·x` etc.
//
// 2D dispatch grid (`(min(n, 65535), ⌈n / 65535⌉, 1)`) mirrors the
// existing `fem_matvec.wgsl` so the dispatcher can lift the row count
// over the WebGPU 65535-per-dim cap.

struct Uniforms {
    nrows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f32,
    beta: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<storage, read> row_offsets : array<u32>;
@group(0) @binding(1) var<storage, read> col_indices : array<u32>;
@group(0) @binding(2) var<storage, read> values : array<f32>;
@group(0) @binding(3) var<storage, read> x : array<f32>;
@group(0) @binding(4) var<storage, read_write> y : array<f32>;

@group(1) @binding(0) var<uniform> u : Uniforms;

const WG_SIZE: u32 = 256u;
const MAX_DIM: u32 = 65535u;

var<workgroup> row_acc : array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn bsr_mv_gpu(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.y * MAX_DIM + wid.x;
    let lane = lid.x;
    if (row >= u.nrows) {
        return;
    }
    let lo = row_offsets[row];
    let hi = row_offsets[row + 1u];

    var local_sum: f32 = 0.0;
    for (var s: u32 = lo + lane; s < hi; s = s + WG_SIZE) {
        let col = col_indices[s];
        local_sum = local_sum + values[s] * x[col];
    }
    row_acc[lane] = local_sum;
    workgroupBarrier();

    var stride: u32 = WG_SIZE / 2u;
    loop {
        if (stride == 0u) { break; }
        if (lane < stride) {
            row_acc[lane] = row_acc[lane] + row_acc[lane + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    if (lane == 0u) {
        // y[row] = α · (A·x)[row] + β · y[row]
        let prev = y[row];
        y[row] = u.alpha * row_acc[0] + u.beta * prev;
    }
}
