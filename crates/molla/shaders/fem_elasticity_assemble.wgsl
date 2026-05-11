// Stage 41.1 — WGSL FEM linear isotropic elasticity assembly (tet).
//
// Each invocation handles one tetrahedral cell:
//   * compute the 12×12 element stiffness `K_e = ∫ Bᵀ D B dV`,
//   * scatter the 144 entries into a global CSR matrix sized for the
//     vector field (`nrows = vertex_count * 3`, interleaved DoF
//     layout `dof = vertex * 3 + component`),
//   * skip rows that are tagged as Dirichlet via `is_bc_row[row]`.
//
// The CSR pattern + values are owned by the caller (a `FemGpuData`
// built against the BC-applied K_red, so the pattern itself is
// identical between the BC-reduced and unreduced cases — `apply_bcs`
// only zeroes values, never rewrites the column structure).
//
// Mirrors `fem::element::compute_element_elasticity_stiffness` in the
// CPU stack — the 6-row B matrix (Voigt notation), the 6×6 D matrix
// in plane-strain form, and the `Bᵀ D B · vol` quadrature collapse.
//
// Caller responsibility:
//   1. Initialise `csr_values` so BC rows already contain `1.0` on
//      their diagonal and `0.0` everywhere else, and non-BC rows are
//      all-zero. (`FemElasticityGpuKernel::assemble` does this via a
//      host write_buffer call before submitting the dispatch.)
//   2. Provide `is_bc_row[row]` — a `u32` array (1 = BC, 0 = free)
//      sized `nrows`.
//   3. Dispatch `assemble_elasticity_tet` over `cell_count`
//      invocations.

struct Uniforms {
    cell_count: u32,
    nrows: u32,           // CSR row count = vertex_count * dim
    lambda: f32,
    mu: f32,
    dim: u32,             // 3 for tets (12 DoFs/element)
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> mesh_vertices : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> mesh_cells : array<u32>;
@group(0) @binding(2) var<storage, read> mesh_cell_offsets : array<u32>;

@group(1) @binding(0) var<storage, read> csr_row_offsets : array<u32>;
@group(1) @binding(1) var<storage, read> csr_col_indices : array<u32>;
@group(1) @binding(2) var<storage, read_write> csr_values : array<atomic<u32>>;
@group(1) @binding(3) var<storage, read> is_bc_row : array<u32>;

@group(2) @binding(0) var<uniform> u : Uniforms;

const WG_SIZE: u32 = 64u;

// Atomic-add into `csr_values[slot]` via the standard f32-bit-cast
// CAS loop. WGSL has no `atomic<f32>` so we bit-cast through `u32`.
fn atomic_add_csr(slot: u32, value: f32) {
    if (value == 0.0) { return; }
    var old_bits = atomicLoad(&csr_values[slot]);
    loop {
        let old_f = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_f + value);
        let r = atomicCompareExchangeWeak(&csr_values[slot], old_bits, new_bits);
        if (r.exchanged) {
            return;
        }
        old_bits = r.old_value;
    }
}

// Linear-search row `row` of the CSR pattern for column `col`. Returns
// `0xFFFFFFFFu` if not present (silently dropped — the BC-reduced
// pattern omits BC columns from non-BC rows, and that's intentional
// since the CG correctness argument doesn't need them — see
// crates/molla/src/fem/elasticity_gpu.rs doc-comment).
fn find_slot(row: u32, col: u32) -> u32 {
    let lo = csr_row_offsets[row];
    let hi = csr_row_offsets[row + 1u];
    for (var s: u32 = lo; s < hi; s = s + 1u) {
        if (csr_col_indices[s] == col) {
            return s;
        }
    }
    return 0xFFFFFFFFu;
}

fn tet_volume(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> f32 {
    let v = dot(b - a, cross(c - a, d - a)) / 6.0;
    return abs(v);
}

// Per-shape-function gradients for a P1 tet — mirrors
// `tet_shape_grads` in `crates/molla/src/fem/element.rs`.
fn tet_shape_grads(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>)
    -> array<vec3<f32>, 4>
{
    let d0 = p1 - p0;
    let d1 = p2 - p0;
    let d2 = p3 - p0;
    let det = dot(d0, cross(d1, d2));
    let inv_det = 1.0 / det;
    let g1 = cross(d1, d2) * inv_det;
    let g2 = cross(d2, d0) * inv_det;
    let g3 = cross(d0, d1) * inv_det;
    let g0 = -(g1 + g2 + g3);
    return array<vec3<f32>, 4>(g0, g1, g2, g3);
}

// One workgroup invocation per cell. Computes a 12×12 local
// elasticity stiffness and scatters it into the global CSR.
@compute @workgroup_size(WG_SIZE)
fn assemble_elasticity_tet(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cid = gid.x;
    if (cid >= u.cell_count) {
        return;
    }

    let off = mesh_cell_offsets[cid];
    let i0 = mesh_cells[off + 0u];
    let i1 = mesh_cells[off + 1u];
    let i2 = mesh_cells[off + 2u];
    let i3 = mesh_cells[off + 3u];

    let p0 = mesh_vertices[i0].xyz;
    let p1 = mesh_vertices[i1].xyz;
    let p2 = mesh_vertices[i2].xyz;
    let p3 = mesh_vertices[i3].xyz;

    let vol = tet_volume(p0, p1, p2, p3);
    let grads = tet_shape_grads(p0, p1, p2, p3);

    // Material constants. plane-strain D matrix in Voigt form
    //   [εxx, εyy, εzz, γyz, γxz, γxy] —
    // diagonal d11 = λ + 2μ on the three axial slots, off-diagonal
    // d12 = λ for the cross-coupling, μ for the three shear slots.
    let d11 = u.lambda + 2.0 * u.mu;
    let d12 = u.lambda;
    let mu_loc = u.mu;

    // Build B as 6 rows × 12 cols (flattened); each node i contributes
    // columns (3i, 3i+1, 3i+2). For node i with gradient (gx, gy, gz):
    //   B[0, 3i  ] = gx
    //   B[1, 3i+1] = gy
    //   B[2, 3i+2] = gz
    //   B[3, 3i+1] = gz, B[3, 3i+2] = gy
    //   B[4, 3i  ] = gz, B[4, 3i+2] = gx
    //   B[5, 3i  ] = gy, B[5, 3i+1] = gx
    var b: array<array<f32, 12>, 6>;
    for (var r: u32 = 0u; r < 6u; r = r + 1u) {
        for (var c: u32 = 0u; c < 12u; c = c + 1u) {
            b[r][c] = 0.0;
        }
    }
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let g = grads[i];
        let cx = 3u * i;
        let cy = 3u * i + 1u;
        let cz = 3u * i + 2u;
        b[0u][cx] = g.x;
        b[1u][cy] = g.y;
        b[2u][cz] = g.z;
        b[3u][cy] = g.z;
        b[3u][cz] = g.y;
        b[4u][cx] = g.z;
        b[4u][cz] = g.x;
        b[5u][cx] = g.y;
        b[5u][cy] = g.x;
    }
    // D · B → 6 × 12.
    var db: array<array<f32, 12>, 6>;
    for (var c: u32 = 0u; c < 12u; c = c + 1u) {
        let exx = b[0u][c];
        let eyy = b[1u][c];
        let ezz = b[2u][c];
        db[0u][c] = d11 * exx + d12 * eyy + d12 * ezz;
        db[1u][c] = d12 * exx + d11 * eyy + d12 * ezz;
        db[2u][c] = d12 * exx + d12 * eyy + d11 * ezz;
        db[3u][c] = mu_loc * b[3u][c];
        db[4u][c] = mu_loc * b[4u][c];
        db[5u][c] = mu_loc * b[5u][c];
    }
    // K_e[r, c] = vol · Σ_q b[q, r] · db[q, c]. Build only what we
    // need on-the-fly while scattering — keeps register pressure
    // bounded (a full 12×12 in flat array<f32, 144> would otherwise
    // spill).
    let verts = array<u32, 4>(i0, i1, i2, i3);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        for (var ci: u32 = 0u; ci < 3u; ci = ci + 1u) {
            let row_local = i * 3u + ci;
            let row_global = verts[i] * 3u + ci;
            // Skip the scatter entirely for BC rows — those keep the
            // (row, row) = 1.0 / off-diag = 0.0 initialisation that
            // the host wrote before dispatch.
            if (is_bc_row[row_global] != 0u) {
                continue;
            }
            for (var j: u32 = 0u; j < 4u; j = j + 1u) {
                for (var cj: u32 = 0u; cj < 3u; cj = cj + 1u) {
                    let col_local = j * 3u + cj;
                    let col_global = verts[j] * 3u + cj;
                    // Inner Σ_q over 6 Voigt rows.
                    var sum: f32 = 0.0;
                    for (var q: u32 = 0u; q < 6u; q = q + 1u) {
                        sum = sum + b[q][row_local] * db[q][col_local];
                    }
                    let val = sum * vol;
                    if (val == 0.0) { continue; }
                    let slot = find_slot(row_global, col_global);
                    if (slot != 0xFFFFFFFFu) {
                        atomic_add_csr(slot, val);
                    }
                }
            }
        }
    }
}
