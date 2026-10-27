// One bounded-irreversible Jacobi sweep of the AT2 phase-field damage
// solve. Mirrors the inner loop of the CPU
// `solve_phase_field_fracture_jacobi`: per row,
//   d_new = clamp((rhs - Σ_{c≠row} A[row,c]·d[c]) / A[row,row],
//                 lower[row], 1).
// One thread per row; the host ping-pongs damage_in/out across sweeps.

struct Uniforms {
    n: vec4<u32>,   // x = row count
};

@group(0) @binding(0) var<storage, read> row_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> col_idx:     array<u32>;
@group(0) @binding(2) var<storage, read> values:      array<f32>;
@group(0) @binding(3) var<storage, read> rhs:         array<f32>;
@group(0) @binding(4) var<storage, read> lower:       array<f32>;

@group(1) @binding(0) var<storage, read> damage_in:        array<f32>;
@group(1) @binding(1) var<storage, read_write> damage_out: array<f32>;

@group(2) @binding(0) var<uniform> u: Uniforms;

@compute @workgroup_size(64)
fn jacobi_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= u.n.x) { return; }

    let lo = row_offsets[row];
    let hi = row_offsets[row + 1u];
    var diag = 0.0;
    var off = 0.0;
    for (var slot = lo; slot < hi; slot = slot + 1u) {
        let col = col_idx[slot];
        let val = values[slot];
        if (col == row) {
            diag = diag + val;
        } else {
            off = off + val * damage_in[col];
        }
    }
    // Degenerate diagonal — leave the value unchanged rather than NaN.
    if (abs(diag) <= 1.0e-30) {
        damage_out[row] = damage_in[row];
        return;
    }
    let unconstrained = (rhs[row] - off) / diag;
    damage_out[row] = clamp(unconstrained, lower[row], 1.0);
}
