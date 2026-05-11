// Stage 38a — sparse axpy `B = α·A + β·B` for the SAME-PATTERN case.
//
// Caller asserts that A and B share identical `row_offsets` +
// `col_indices` (i.e. A.values and B.values are aligned per-slot). The
// kernel is a trivial element-wise blend over the nnz slots — one
// thread per nonzero. This is the common-case for FEM where multiple
// matrices with the same sparsity pattern get combined (mass + stiffness,
// implicit MPM updates, etc.).
//
// The pattern-union case (different sparsities) stays host-orchestrated
// for v1 — see `BsrAlgebraKernel::axpy` for the dispatch.

struct Uniforms {
    nnz: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    alpha: f32,
    beta: f32,
    _pad3: f32,
    _pad4: f32,
};

@group(0) @binding(0) var<storage, read> a_values : array<f32>;
@group(0) @binding(1) var<storage, read_write> b_values : array<f32>;

@group(1) @binding(0) var<uniform> u : Uniforms;

const WG_SIZE: u32 = 256u;

@compute @workgroup_size(WG_SIZE)
fn bsr_axpy(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.nnz) {
        return;
    }
    b_values[i] = u.alpha * a_values[i] + u.beta * b_values[i];
}
