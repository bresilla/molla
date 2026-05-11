// Stage 41 v2 — WGSL Taylor-Hood Stokes element assembly (P2 velocity,
// P1 pressure) for triangle meshes (2D).
//
// Two entry points:
//
//   1. `assemble_velocity_block` — scatters the 12×12 per-cell P2-vector
//      velocity stiffness `A^e_{(i, c), (j, c')} = δ_{c, c'} · μ · ∫
//      ∇N_i · ∇N_j dx` into the global velocity CSR. The matrix has
//      `dim`-block-diagonal structure (cross-component coupling is
//      zero for the simplified `μ ∇u : ∇v` Laplacian form — equivalent
//      to the symmetric stress form for fully-Dirichlet boundary
//      conditions like the lid-driven cavity).
//
//   2. `assemble_div_block` — scatters the 3×12 per-cell P1×∂P2
//      divergence coupling `B^e_{q, (j, c)} = ∫ N_q^p · ∂N_j^v / ∂x_c
//      dx` into the global divergence CSR. Rows are pressure DoFs
//      (P1 vertex indices); columns are interleaved velocity DoFs
//      (`node * dim + c`).
//
// **Tri-only v1**: Stage 41 v1 (elasticity) shipped a tet kernel; the
// lid-driven cavity contract test is 2D, so we ship the tri path now
// and defer the tet path to v2. The pattern is identical (different
// number of nodes + 3D dot-product); copy & adjust when needed.
//
// Mirrors `compute_element_stokes_velocity_block` and
// `compute_element_stokes_div_block` in `crates/molla/src/fem/stokes.rs`
// — same Dunavant 6-point quadrature, same P2 shape gradients.
//
// Caller contract (see `crates/molla/src/fem/stokes_gpu.rs`):
//   * `csr_values` is **zero-initialised** before dispatch. Both
//     kernels use atomic-add scatter, so any pre-existing content
//     would accumulate. (The dispatcher calls `clear_csr_values`
//     between assembles.)
//   * `p2_dof_table` maps each cell's local P2 node index (`0..6`)
//     to its global P2 node index; flat layout `cell_idx * 6 + local`.
//     The dispatcher builds this CPU-side from the mesh + P2DofMap.

struct StokesUniforms {
    cell_count: u32,
    n_v: u32,            // velocity DoF count = n_v_nodes * dim
    n_p: u32,            // pressure DoF count = vertex_count
    dim: u32,            // 2 for tri / planar
    viscosity: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

@group(0) @binding(0) var<storage, read> mesh_vertices : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> mesh_cells : array<u32>;
@group(0) @binding(2) var<storage, read> mesh_cell_offsets : array<u32>;
@group(0) @binding(3) var<storage, read> p2_dof_table : array<u32>;

@group(1) @binding(0) var<storage, read> csr_row_offsets : array<u32>;
@group(1) @binding(1) var<storage, read> csr_col_indices : array<u32>;
@group(1) @binding(2) var<storage, read_write> csr_values : array<atomic<u32>>;

@group(2) @binding(0) var<uniform> u : StokesUniforms;

const WG_SIZE: u32 = 64u;

// Atomic-add f32 via u32 bit-cast CAS loop — WGSL has no atomic<f32>.
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

fn tri_area_2d(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> f32 {
    let dx1 = p1.x - p0.x;
    let dy1 = p1.y - p0.y;
    let dx2 = p2.x - p0.x;
    let dy2 = p2.y - p0.y;
    return 0.5 * abs(dx1 * dy2 - dx2 * dy1);
}

// P1 shape-function gradients on a 2D triangle. Returns three vec2<f32>
// (`x, y` partials). Mirrors `tri_shape_grads_2d` in
// `crates/molla/src/fem/element.rs`.
fn tri_shape_grads_2d(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> array<vec2<f32>, 3> {
    let two_a_signed = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
    let inv = 1.0 / two_a_signed;
    let g0 = vec2<f32>(p1.y - p2.y, p2.x - p1.x) * inv;
    let g1 = vec2<f32>(p2.y - p0.y, p0.x - p2.x) * inv;
    let g2 = vec2<f32>(p0.y - p1.y, p1.x - p0.x) * inv;
    return array<vec2<f32>, 3>(g0, g1, g2);
}

// P2 triangle shape-function gradients at barycentric point `bary`,
// given the three P1 (corner) gradients. Mirrors
// `p2_tri_grad_from_bary_grads` in `crates/molla/src/fem/p2_element.rs`.
//
// Layout: [c0, c1, c2, e01, e12, e20]
fn p2_tri_grads_at(
    bary: vec3<f32>,
    g_lambda: array<vec2<f32>, 3>,
) -> array<vec2<f32>, 6> {
    // Corner gradients: ∇N_i = (4λᵢ - 1) ∇λᵢ.
    let c0 = g_lambda[0] * (4.0 * bary.x - 1.0);
    let c1 = g_lambda[1] * (4.0 * bary.y - 1.0);
    let c2 = g_lambda[2] * (4.0 * bary.z - 1.0);
    // Edge gradients: ∇N_ij = 4(λⱼ ∇λᵢ + λᵢ ∇λⱼ).
    let e01 = (g_lambda[0] * bary.y + g_lambda[1] * bary.x) * 4.0;
    let e12 = (g_lambda[1] * bary.z + g_lambda[2] * bary.y) * 4.0;
    let e20 = (g_lambda[2] * bary.x + g_lambda[0] * bary.z) * 4.0;
    return array<vec2<f32>, 6>(c0, c1, c2, e01, e12, e20);
}

// 6-point Dunavant rule on a triangle (degree-4 exact). Pre-normalised
// so ∑ wᵢ = 1; assembly multiplies by the cell area.
//
// Returns barycentric coordinates for quadrature point `q ∈ 0..6`.
fn qpt_bary(q: u32) -> vec3<f32> {
    // Two-orbit Dunavant rule. The first three points are the "near-
    // corner" cluster with weight w1, the last three are the "near-
    // edge-midpoint" cluster with weight w2.
    let a1 = 0.816847572980459;
    let b1 = 0.091576213509771;
    let a2 = 0.108103018168070;
    let b2 = 0.445948490915965;
    switch q {
        case 0u: { return vec3<f32>(a1, b1, b1); }
        case 1u: { return vec3<f32>(b1, a1, b1); }
        case 2u: { return vec3<f32>(b1, b1, a1); }
        case 3u: { return vec3<f32>(a2, b2, b2); }
        case 4u: { return vec3<f32>(b2, a2, b2); }
        default: { return vec3<f32>(b2, b2, a2); }
    }
}

fn qpt_w(q: u32) -> f32 {
    // w1 for q ∈ [0..3), w2 for q ∈ [3..6).
    if (q < 3u) {
        return 0.109951743655322;
    }
    return 0.223381589678011;
}

// One workgroup invocation per cell — assemble the local 12×12 P2
// velocity stiffness and scatter into the global velocity CSR. Layout:
// local DoF `i * dim + c` → global DoF `p2_node(i) * dim + c`.
//
// Cross-component coupling vanishes for the Laplacian form, so the
// inner-most scatter only touches `c == c'` slots.
@compute @workgroup_size(WG_SIZE)
fn assemble_velocity_block(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cid = gid.x;
    if (cid >= u.cell_count) {
        return;
    }
    let off = mesh_cell_offsets[cid];
    let i0 = mesh_cells[off + 0u];
    let i1 = mesh_cells[off + 1u];
    let i2 = mesh_cells[off + 2u];
    let p0 = mesh_vertices[i0].xyz;
    let p1 = mesh_vertices[i1].xyz;
    let p2 = mesh_vertices[i2].xyz;

    let area = tri_area_2d(p0, p1, p2);
    let g_lambda = tri_shape_grads_2d(p0, p1, p2);

    // Quadrature loop: accumulate the 6×6 P2 scalar stiffness `k_scalar`
    // in registers; replicate on the `dim` block-diagonal during scatter.
    var k_scalar: array<f32, 36>;  // 6×6 row-major
    for (var i: u32 = 0u; i < 36u; i = i + 1u) {
        k_scalar[i] = 0.0;
    }
    for (var q: u32 = 0u; q < 6u; q = q + 1u) {
        let bary = qpt_bary(q);
        let g = p2_tri_grads_at(bary, g_lambda);
        let w = qpt_w(q) * area * u.viscosity;
        for (var i: u32 = 0u; i < 6u; i = i + 1u) {
            for (var j: u32 = 0u; j < 6u; j = j + 1u) {
                let gg = g[i].x * g[j].x + g[i].y * g[j].y;
                k_scalar[i * 6u + j] = k_scalar[i * 6u + j] + w * gg;
            }
        }
    }

    // Scatter: for each (local_i, local_j) replicate on `dim` diagonals.
    let p2_base = cid * 6u;
    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let gi_node = p2_dof_table[p2_base + i];
        for (var j: u32 = 0u; j < 6u; j = j + 1u) {
            let gj_node = p2_dof_table[p2_base + j];
            let val_scalar = k_scalar[i * 6u + j];
            if (val_scalar == 0.0) { continue; }
            for (var c: u32 = 0u; c < u.dim; c = c + 1u) {
                let row = gi_node * u.dim + c;
                let col = gj_node * u.dim + c;
                let slot = find_slot(row, col);
                if (slot != 0xFFFFFFFFu) {
                    atomic_add_csr(slot, val_scalar);
                }
            }
        }
    }
}

// Divergence block: rows = P1 pressure DoFs (corner vertex indices),
// columns = interleaved P2 velocity DoFs (`node * dim + c`). One
// workgroup invocation per cell; computes the 3×12 local block and
// scatters into the global B CSR.
//
// P1 pressure shape functions on a triangle are the barycentric
// coordinates `(λ₀, λ₁, λ₂)` evaluated at each corner — pressure
// DoF `q` corresponds to corner `q`, and its shape value at a
// quadrature point is just `bary[q]`.
@compute @workgroup_size(WG_SIZE)
fn assemble_div_block(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cid = gid.x;
    if (cid >= u.cell_count) {
        return;
    }
    let off = mesh_cell_offsets[cid];
    let i0 = mesh_cells[off + 0u];
    let i1 = mesh_cells[off + 1u];
    let i2 = mesh_cells[off + 2u];
    let p0 = mesh_vertices[i0].xyz;
    let p1 = mesh_vertices[i1].xyz;
    let p2 = mesh_vertices[i2].xyz;

    let area = tri_area_2d(p0, p1, p2);
    let g_lambda = tri_shape_grads_2d(p0, p1, p2);

    // The local B block is `3 × (6·dim)`; with cross-component decoupling
    // for the divergence operator `∂N_j^v / ∂x_c` selects exactly one
    // component per (j, c) pair. Accumulate the local block in registers
    // then scatter.
    //
    // Layout: `b_local[ip * 6 * dim + jv * dim + c]`.
    //
    // For `dim = 2` that's `3 × 12 = 36` floats, fits easily in regs.
    var b_local: array<f32, 36>;  // 3 rows × 12 cols, only dim=2 path
    for (var i: u32 = 0u; i < 36u; i = i + 1u) {
        b_local[i] = 0.0;
    }
    for (var q: u32 = 0u; q < 6u; q = q + 1u) {
        let bary = qpt_bary(q);
        let g = p2_tri_grads_at(bary, g_lambda);
        let w = qpt_w(q) * area;
        for (var ip: u32 = 0u; ip < 3u; ip = ip + 1u) {
            let pq = bary[ip];
            for (var jv: u32 = 0u; jv < 6u; jv = jv + 1u) {
                for (var c: u32 = 0u; c < u.dim; c = c + 1u) {
                    let dn = select(g[jv].y, g[jv].x, c == 0u);
                    let idx = ip * (6u * u.dim) + jv * u.dim + c;
                    b_local[idx] = b_local[idx] + w * pq * dn;
                }
            }
        }
    }

    // Scatter into global B CSR. Rows are corner P1 indices `i0, i1, i2`.
    let p_verts = array<u32, 3>(i0, i1, i2);
    let p2_base = cid * 6u;
    for (var ip: u32 = 0u; ip < 3u; ip = ip + 1u) {
        let row = p_verts[ip];
        for (var jv: u32 = 0u; jv < 6u; jv = jv + 1u) {
            let gj_node = p2_dof_table[p2_base + jv];
            for (var c: u32 = 0u; c < u.dim; c = c + 1u) {
                let val = b_local[ip * (6u * u.dim) + jv * u.dim + c];
                if (val == 0.0) { continue; }
                let col = gj_node * u.dim + c;
                let slot = find_slot(row, col);
                if (slot != 0xFFFFFFFFu) {
                    atomic_add_csr(slot, val);
                }
            }
        }
    }
}
