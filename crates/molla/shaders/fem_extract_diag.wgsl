// Stage 36.5 — WGSL kernel for extracting the diagonal of a CSR
// matrix into a flat `f32` vector.
//
// For each row `i ∈ [0, nrows)`, walks `csr_row_offsets[i..i+1]` and
// finds the slot `s` where `csr_col_indices[s] == i`, writing
// `csr_values[s]` into `dst[i]`. If no diagonal entry exists for the
// row (assembled CSR from a P1 stiffness matrix always has one, but
// the kernel handles the missing case defensively) `dst[i]` is left
// at its prior value — the host pre-zeroes the buffer so missing
// diagonals show up as `0` (mapped to pass-through inverse downstream).
//
// One workgroup per row, packed into a 2D dispatch grid identical to
// the matvec kernel because WebGPU caps any single dispatch dim at
// 65535. Within a workgroup only lane 0 does work — the row is tiny
// (≤ 25 nonzeros for v1 tet meshes) and the search is cheaper than
// any tree reduction we could mount across the workgroup. Sized to
// match `fem_matvec.wgsl` for layout consistency (WG_SIZE = 64).

struct Uniforms {
    nrows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read>       csr_row_offsets : array<u32>;
@group(0) @binding(1) var<storage, read>       csr_col_indices : array<u32>;
@group(0) @binding(2) var<storage, read>       csr_values      : array<f32>;
@group(0) @binding(3) var<storage, read_write> diag            : array<f32>;

@group(1) @binding(0) var<uniform> u : Uniforms;

const WG_SIZE: u32 = 64u;
const MAX_DIM: u32 = 65535u;

@compute @workgroup_size(WG_SIZE)
fn extract_diag(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = wid.y * MAX_DIM + wid.x;
    if (row >= u.nrows) {
        return;
    }
    if (lid.x != 0u) {
        return;
    }
    let lo = csr_row_offsets[row];
    let hi = csr_row_offsets[row + 1u];
    for (var s: u32 = lo; s < hi; s = s + 1u) {
        if (csr_col_indices[s] == row) {
            diag[row] = csr_values[s];
            return;
        }
    }
}
