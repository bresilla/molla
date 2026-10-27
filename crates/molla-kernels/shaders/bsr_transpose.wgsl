// Stage 38a — transpose value scatter `B.values[dst_map[i]] = A.values[i]`.
//
// The transpose `B = Aᵀ` factors into a SYMBOLIC step (counts per
// column → row_offsets / col_indices of B) and a NUMERIC step
// (scatter values). v1 does the symbolic step on the HOST — see
// `BsrAlgebraKernel::transpose` — because the count + prefix-sum
// pattern is short and the result is needed for buffer allocation
// anyway. The host then builds `dst_map: [u32; nnz]` mapping each
// slot in `A.values` to its destination slot in `B.values`, uploads
// it, and dispatches THIS kernel for the numeric scatter.
//
// One thread per source slot — completely embarrassingly parallel.

struct Uniforms {
    nnz: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> src_values : array<f32>;
@group(0) @binding(1) var<storage, read> dst_map : array<u32>;
@group(0) @binding(2) var<storage, read_write> dst_values : array<f32>;

@group(1) @binding(0) var<uniform> u : Uniforms;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn bsr_transpose_scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.nnz) {
        return;
    }
    dst_values[dst_map[i]] = src_values[i];
}
