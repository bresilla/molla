// Stage 36.5 — WGSL FEM P1 stiffness assembly (tet meshes only).
//
// Each invocation handles one cell. The per-cell stiffness matrix
// `K_e[i,j] = coeff * vol * grad_phi_i . grad_phi_j` is computed in
// registers from the four vertex positions; entries are then
// scattered into the global CSR via an atomicAdd (f32-bit-cast u32
// CAS loop — WGSL has no native atomic<f32>; same pattern as
// `soft_vt_contact.wgsl`).
//
// Caller responsibility:
//   1. Zero `csr_values` first (`FemGpuKernel::assemble` does this via
//      a host write_buffer call before submitting the dispatch).
//   2. Dispatch `assemble_p1_stiffness_tet` over `cell_count`
//      invocations.
//
// Triangle elements are NOT yet supported on the GPU (v1 demo is the
// 3D heat-equation problem on tets). Follow-up stages can add a
// `assemble_p1_stiffness_tri` entry point — the scatter logic is
// identical, only the local stiffness math changes.

struct Uniforms {
    cell_count: u32,
    nrows: u32,           // CSR row count (= mesh.vert_count() for P1Nodal)
    coeff: f32,           // diffusivity / Laplacian scale
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read> mesh_vertices : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> mesh_cells : array<u32>;
@group(0) @binding(2) var<storage, read> mesh_cell_offsets : array<u32>;

@group(1) @binding(0) var<storage, read> csr_row_offsets : array<u32>;
@group(1) @binding(1) var<storage, read> csr_col_indices : array<u32>;
@group(1) @binding(2) var<storage, read_write> csr_values : array<atomic<u32>>;

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

// Linear-search the CSR row `row` for column `col` and return the
// slot index, or `0xFFFFFFFFu` if not found (which would indicate a
// bug — the assembly visitor only scatters to (row, col) pairs that
// the CPU side already inserted into the pattern).
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

// Compute the absolute volume of the tetrahedron (a, b, c, d).
fn tet_volume(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> f32 {
    let v = dot(b - a, cross(c - a, d - a)) / 6.0;
    return abs(v);
}

// Per-shape-function gradients for a P1 tet element. Mirrors
// `tet_shape_grads` in `crates/molla/src/fem/element.rs`:
//   D = [v1-v0 | v2-v0 | v3-v0]   (3x3 matrix, columns)
//   G = D^{-T}                     (gradients of phi_1, phi_2, phi_3)
//   grad_phi_0 = -(grad_phi_1 + grad_phi_2 + grad_phi_3)
fn tet_shape_grads(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>)
    -> array<vec3<f32>, 4>
{
    let d0 = p1 - p0;
    let d1 = p2 - p0;
    let d2 = p3 - p0;

    // 3x3 cofactor expansion for the inverse.
    // det = d0 . (d1 x d2)
    let det = dot(d0, cross(d1, d2));
    let inv_det = 1.0 / det;

    // Inverse of D (column-major): rows of D^{-1} are scaled cofactors.
    // For D = [d0|d1|d2] (cols), D^{-1} rows are
    //   row0 = (d1 x d2) / det
    //   row1 = (d2 x d0) / det
    //   row2 = (d0 x d1) / det
    // The transpose D^{-T} has those as columns. We want column-i of
    // D^{-T} = row-i of D^{-1} = (cross of the other two cols) / det.
    let g1 = cross(d1, d2) * inv_det;
    let g2 = cross(d2, d0) * inv_det;
    let g3 = cross(d0, d1) * inv_det;
    let g0 = -(g1 + g2 + g3);

    return array<vec3<f32>, 4>(g0, g1, g2, g3);
}

@compute @workgroup_size(WG_SIZE)
fn assemble_p1_stiffness_tet(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cid = gid.x;
    if (cid >= u.cell_count) {
        return;
    }

    // Pull the four vertex indices for this tet.
    let off = mesh_cell_offsets[cid];
    // (Triangle cells would have 3 indices; v1 tet kernel asserts 4
    //  via the cell_offsets stride at upload time — no need to check
    //  here.)
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

    // K_e[i,j] = coeff * vol * grads[i] . grads[j]
    // Scatter to the global CSR.
    let verts = array<u32, 4>(i0, i1, i2, i3);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let row = verts[i];
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let col = verts[j];
            let val = u.coeff * vol * dot(grads[i], grads[j]);
            if (val == 0.0) { continue; }
            let slot = find_slot(row, col);
            // `slot == 0xFFFFFFFFu` means the CPU pattern didn't
            // include (row, col) — should never happen because the
            // CPU `assemble_global` walks the same cells. We just
            // skip silently to avoid a wild atomic write.
            if (slot != 0xFFFFFFFFu) {
                atomic_add_csr(slot, val);
            }
        }
    }
}

// ─── P1 mass matrix (tet) ────────────────────────────────────────────
// Consistent mass M_e[i, j] = coeff * vol * (1 + delta_ij) / 20
// (see Hughes, "The Finite Element Method", 1987, eq. 4.5.13). The
// `coeff` slot doubles as a density / `c²` scale here — pass `1.0`
// for the standard mass matrix, `rho` for inertial scaling. Same
// scatter / atomic-add logic as the stiffness pass; only the per-cell
// math differs.
@compute @workgroup_size(WG_SIZE)
fn assemble_p1_mass_tet(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    let scale = u.coeff * vol / 20.0;

    let verts = array<u32, 4>(i0, i1, i2, i3);
    for (var i: u32 = 0u; i < 4u; i = i + 1u) {
        let row = verts[i];
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let col = verts[j];
            // Diagonal entries get coeff 2, off-diagonals get coeff 1.
            let mass_coeff = select(1.0, 2.0, i == j);
            let val = scale * mass_coeff;
            let slot = find_slot(row, col);
            if (slot != 0xFFFFFFFFu) {
                atomic_add_csr(slot, val);
            }
        }
    }
}
