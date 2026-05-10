// Stage 37.8 — single-workgroup Cholesky factorization.
//
// Computes lower-triangular `L` (row-major, N × N) such that
// `L · Lᵀ = A` for an SPD matrix `A`. Single workgroup of 64 threads
// cooperates per column j: thread 0 computes the diagonal sqrt; the
// rest fan out across `i ∈ [j+1, N)` to compute the off-diagonal
// entries of column j.
//
// Limits: `N ≤ 1024` (single workgroup; storage memory is the
// scratch). Multi-workgroup tiling (true block-Cholesky) is a v2.

const WG_SIZE: u32 = 64u;

struct Uniforms {
    n: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

// `a_buf` is read on entry (row-major SPD input) and overwritten with
// `L` on exit (lower triangle = factor; upper triangle = 0).
@group(0) @binding(0) var<storage, read_write> a_buf: array<f32>;
@group(1) @binding(0) var<uniform> u: Uniforms;

@compute @workgroup_size(64, 1, 1)
fn tile_block_cholesky(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let n = u.n;
    let tid = lid.x;

    // Strict upper triangle is zero in `L`. Each thread strides through
    // the upper triangle to clear it before factorisation begins. This
    // is also defensive for callers that re-use a buffer.
    let total = n * n;
    var k0: u32 = tid;
    while (k0 < total) {
        let r = k0 / n;
        let c = k0 % n;
        if (c > r) {
            // Stash the original (we'll need symmetric pairs) … but
            // since we're SPD, A[r][c] = A[c][r], and we already use
            // the lower triangle below. Just zero it out for the final
            // result.
            a_buf[k0] = 0.0;
        }
        k0 = k0 + WG_SIZE;
    }
    workgroupBarrier();

    for (var j: u32 = 0u; j < n; j = j + 1u) {
        // Step 1: diagonal element. Thread 0 computes
        // L[j,j] = sqrt(A[j,j] - Σ_{k<j} L[j,k]²)
        if (tid == 0u) {
            var s: f32 = a_buf[j * n + j];
            for (var k: u32 = 0u; k < j; k = k + 1u) {
                let v = a_buf[j * n + k];
                s = s - v * v;
            }
            // Clamp to non-negative for numerical safety on
            // marginally-SPD inputs (caller-side responsibility).
            if (s < 0.0) {
                s = 0.0;
            }
            a_buf[j * n + j] = sqrt(s);
        }
        workgroupBarrier();

        let ljj = a_buf[j * n + j];
        // WGSL `select(false_val, true_val, condition)` evaluates BOTH
        // arguments — `1.0 / ljj` would compute `+inf` when `ljj = 0`
        // (the singular / non-SPD path the clamp above produces) and
        // some drivers flag this as a robustness violation. Use a
        // branched scalar instead so the divide never executes when
        // the diagonal is degenerate.
        var inv_ljj: f32 = 0.0;
        if (ljj > 1.0e-30) {
            inv_ljj = 1.0 / ljj;
        }

        // Step 2: off-diagonal entries below the diagonal in column j.
        // Each thread takes one i = j+1+tid, j+1+tid+WG, ...
        var i: u32 = j + 1u + tid;
        while (i < n) {
            var s: f32 = a_buf[i * n + j];
            for (var k: u32 = 0u; k < j; k = k + 1u) {
                s = s - a_buf[i * n + k] * a_buf[j * n + k];
            }
            a_buf[i * n + j] = s * inv_ljj;
            i = i + WG_SIZE;
        }
        workgroupBarrier();
    }
}
