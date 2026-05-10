// Stage 36.5 — WGSL CSR matvec for FEM `K_csr · x = y`.
//
// Standard SpMV pattern: one workgroup per row, each lane in the
// workgroup walks a stride of nonzeros in that row, accumulating into
// a workgroup-shared scratch, then the leader writes the row sum.
//
// Output `y` is f32; input `x` is f32. The host wrapper converts
// from / to f64 for the closure-driven CG (Stage 27 `cg_solve` is
// f64-internal; the f32 round-trip on the matvec is the documented
// precision tradeoff).
//
// One workgroup per row makes this trivially correct without
// inter-workgroup synchronisation. For very long rows (say > 64
// nonzeros) we'd want a tile-per-row pattern; v1 tet meshes have
// rows of size ~14-25, well under the workgroup size.

struct Uniforms {
    nrows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> csr_row_offsets : array<u32>;
@group(0) @binding(1) var<storage, read> csr_col_indices : array<u32>;
@group(0) @binding(2) var<storage, read> csr_values : array<f32>;
@group(0) @binding(3) var<storage, read> x : array<f32>;
@group(0) @binding(4) var<storage, read_write> y : array<f32>;

@group(1) @binding(0) var<uniform> u : Uniforms;

const WG_SIZE: u32 = 64u;

// Workgroup-shared accumulator for the per-row reduction.
var<workgroup> row_acc : array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn csr_matvec(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.x;
    let lane = lid.x;
    if (row >= u.nrows) {
        return;
    }
    let lo = csr_row_offsets[row];
    let hi = csr_row_offsets[row + 1u];

    // Each lane sums a stride of nonzeros.
    var local_sum: f32 = 0.0;
    for (var s: u32 = lo + lane; s < hi; s = s + WG_SIZE) {
        let col = csr_col_indices[s];
        local_sum = local_sum + csr_values[s] * x[col];
    }
    row_acc[lane] = local_sum;
    workgroupBarrier();

    // Tree reduction over the workgroup. WG_SIZE = 64 = 2^6.
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
        y[row] = row_acc[0];
    }
}
