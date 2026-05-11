// Stage 41 v2 — APIC FEM-MPM GPU coupling kernels.
//
// One WGSL file, several @compute entry points, mirroring
// `crates/molla/src/fem/apic_coupling.rs` semantics step-by-step:
//
//   1. `locate_tet`        — per particle: brute-force O(T) barycentric
//      scan over every tet; write owning tet + the four (vertex_index,
//      weight) pairs into the `loc` buffer. Out-of-mesh particles get
//      tet = 0xFFFFFFFF.
//   2. `p2g`               — per particle: scatter `weight·mass` and
//      `weight·mass·(v + C·dx)` into atomic node mass + 3-component
//      momentum buffers.
//   3. `grid_update`       — per node: velocity = momentum / mass; add
//      `dt · gravity`. Empty cells stay zero.
//   4. `viscous_smoothing` — per node: one Jacobi sweep that replaces
//      `v[i] ← v[i] + dt·ν·(mass-weighted neighbour average − v[i])`,
//      with neighbours read from precomputed CSR-style offsets / indices.
//      Operates on `g_velocity` → writes into `g_velocity_next` (caller
//      copies back). Stable when `dt·ν ≤ 1`.
//   5. `apply_bcs`         — pin Dirichlet rows to user-supplied values.
//   6. `g2p`               — per particle: gather node velocities into
//      new v + new C affine matrix; advect x by `dt · v_new`.
//
// CPU mirror lives in `crates/molla/src/fem/apic_coupling.rs`; this
// shader is a 1-to-1 translation (no algorithmic improvements) so the
// CPU oracle + GPU output match within f32 round-off — the parity test
// in `crates/molla/tests/fem_apic_gpu.rs` enforces 1e-4 relative.
//
// # Memory layout
//
//   `loc` buffer (per-particle, stride = 8 u32 + 4 f32 = 8 vec4<u32>):
//     vec4<u32>: (tet_id, vert0, vert1, vert2)
//     vec4<u32>: (vert3, _pad0, _pad1, _pad2)
//     vec4<f32>: (w0, w1, w2, w3)
//   Total per particle: 12 × 4 bytes = 48 bytes = 3 × vec4.
//
//   Atomic accumulators on the grid (length = vertex_count):
//     g_mass     : array<atomic<u32>>  (bit-cast f32)
//     g_mom_xyz  : array<atomic<u32>>  three separate arrays so we can
//                  use the same `atomic_add_f32` helper.
//
//   Particle SoA:
//     p_x      : array<vec4<f32>>       (xyz + pad)
//     p_v      : array<vec4<f32>>       (xyz + pad)
//     p_C      : array<vec4<f32>>       3 entries per particle (rows
//                                       of the 3×3 affine matrix; pad
//                                       per row).
//     p_mass   : array<f32>             (per particle)

struct Uniforms {
    particle_count: u32,
    vert_count: u32,
    cell_count: u32,
    _pad0: u32,

    dt: f32,
    viscosity: f32,
    gravity_x: f32,
    gravity_y: f32,

    gravity_z: f32,
    _pad1: f32,
    _pad2: f32,
    _pad3: f32,
};

@group(0) @binding(0) var<uniform> u : Uniforms;

@group(1) @binding(0) var<storage, read> mesh_vertices : array<vec4<f32>>;
@group(1) @binding(1) var<storage, read> mesh_cells : array<u32>;
@group(1) @binding(2) var<storage, read> mesh_cell_offsets : array<u32>;

@group(2) @binding(0) var<storage, read_write> p_x : array<vec4<f32>>;
@group(2) @binding(1) var<storage, read_write> p_v : array<vec4<f32>>;
// p_C stored row-major: 3 vec4 per particle (r0 r1 r2; w slot padding).
@group(2) @binding(2) var<storage, read_write> p_C : array<vec4<f32>>;
@group(2) @binding(3) var<storage, read> p_mass : array<f32>;
// Per-particle location: 3 vec4 per particle (see layout above).
@group(2) @binding(4) var<storage, read_write> p_loc : array<vec4<u32>>;

@group(3) @binding(0) var<storage, read_write> g_mass : array<atomic<u32>>;
@group(3) @binding(1) var<storage, read_write> g_mom_x : array<atomic<u32>>;
@group(3) @binding(2) var<storage, read_write> g_mom_y : array<atomic<u32>>;
@group(3) @binding(3) var<storage, read_write> g_mom_z : array<atomic<u32>>;
@group(3) @binding(4) var<storage, read_write> g_vel : array<vec4<f32>>;
@group(3) @binding(5) var<storage, read_write> g_vel_next : array<vec4<f32>>;
@group(3) @binding(6) var<storage, read> g_neighbour_offsets : array<u32>;
@group(3) @binding(7) var<storage, read> g_neighbour_indices : array<u32>;
@group(3) @binding(8) var<storage, read> g_bc_mask : array<u32>;       // 1 if BC
@group(3) @binding(9) var<storage, read> g_bc_values : array<vec4<f32>>;

const WG_SIZE : u32 = 64u;
const INVALID_TET : u32 = 0xFFFFFFFFu;
const BARY_EPS : f32 = 1.0e-5;

// ─── Atomic-add helper (f32 via u32 bit-cast CAS loop) ────────────────

fn atomic_add_mass(slot: u32, value: f32) {
    if (value == 0.0) { return; }
    var old_bits = atomicLoad(&g_mass[slot]);
    loop {
        let old_f = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_f + value);
        let r = atomicCompareExchangeWeak(&g_mass[slot], old_bits, new_bits);
        if (r.exchanged) { return; }
        old_bits = r.old_value;
    }
}

fn atomic_add_mom_x(slot: u32, value: f32) {
    if (value == 0.0) { return; }
    var old_bits = atomicLoad(&g_mom_x[slot]);
    loop {
        let old_f = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_f + value);
        let r = atomicCompareExchangeWeak(&g_mom_x[slot], old_bits, new_bits);
        if (r.exchanged) { return; }
        old_bits = r.old_value;
    }
}
fn atomic_add_mom_y(slot: u32, value: f32) {
    if (value == 0.0) { return; }
    var old_bits = atomicLoad(&g_mom_y[slot]);
    loop {
        let old_f = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_f + value);
        let r = atomicCompareExchangeWeak(&g_mom_y[slot], old_bits, new_bits);
        if (r.exchanged) { return; }
        old_bits = r.old_value;
    }
}
fn atomic_add_mom_z(slot: u32, value: f32) {
    if (value == 0.0) { return; }
    var old_bits = atomicLoad(&g_mom_z[slot]);
    loop {
        let old_f = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_f + value);
        let r = atomicCompareExchangeWeak(&g_mom_z[slot], old_bits, new_bits);
        if (r.exchanged) { return; }
        old_bits = r.old_value;
    }
}

// ─── Barycentric tet test (Cramer rule) ──────────────────────────────
// Returns (w0, w1, w2, w3) with .w-component = 1 on success / 0 on
// degenerate. Negative weights are normal (point outside): the caller
// checks the [−eps, 1+eps] range.
fn barycentric_tet(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>, p3: vec3<f32>, x: vec3<f32>)
    -> vec4<f32>
{
    let d0 = p1 - p0;
    let d1 = p2 - p0;
    let d2 = p3 - p0;
    let det = dot(d0, cross(d1, d2));
    if (abs(det) < 1.0e-18) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    let inv = 1.0 / det;
    // Cramer rule columns
    let r = x - p0;
    let lam_x = dot(r, cross(d1, d2)) * inv;
    let lam_y = dot(r, cross(d2, d0)) * inv;
    let lam_z = dot(r, cross(d0, d1)) * inv;
    let lam_0 = 1.0 - lam_x - lam_y - lam_z;
    return vec4<f32>(lam_0, lam_x, lam_y, lam_z);
}

// Brute-force O(T) tet scan, mirroring CPU `locate_particle_tet`.
@compute @workgroup_size(WG_SIZE)
fn locate_tet(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= u.particle_count) { return; }
    let x = p_x[pid].xyz;

    var found_tet: u32 = INVALID_TET;
    var found_w: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var found_v: vec4<u32> = vec4<u32>(0u, 0u, 0u, 0u);
    var found_v3: u32 = 0u;

    for (var cid: u32 = 0u; cid < u.cell_count; cid = cid + 1u) {
        let off = mesh_cell_offsets[cid];
        let i0 = mesh_cells[off + 0u];
        let i1 = mesh_cells[off + 1u];
        let i2 = mesh_cells[off + 2u];
        let i3 = mesh_cells[off + 3u];
        let p0 = mesh_vertices[i0].xyz;
        let p1 = mesh_vertices[i1].xyz;
        let p2 = mesh_vertices[i2].xyz;
        let p3 = mesh_vertices[i3].xyz;
        let w = barycentric_tet(p0, p1, p2, p3, x);
        // Degenerate tet returns (0,0,0,0). Accept any tet whose four
        // weights all fall inside [−eps, 1+eps].
        let ok = (w.x >= -BARY_EPS) && (w.x <= 1.0 + BARY_EPS)
              && (w.y >= -BARY_EPS) && (w.y <= 1.0 + BARY_EPS)
              && (w.z >= -BARY_EPS) && (w.z <= 1.0 + BARY_EPS)
              && (w.w >= -BARY_EPS) && (w.w <= 1.0 + BARY_EPS);
        // Also reject the degenerate case where all components are
        // exactly zero (det ≈ 0 → bary returned a zeroed vector).
        let nonzero = w.x != 0.0 || w.y != 0.0 || w.z != 0.0 || w.w != 0.0;
        if (ok && nonzero) {
            found_tet = cid;
            found_w = w;
            found_v = vec4<u32>(i0, i1, i2, i3);
            found_v3 = i3;
            break;
        }
    }

    let base = pid * 3u;
    p_loc[base + 0u] = vec4<u32>(found_tet, found_v.x, found_v.y, found_v.z);
    p_loc[base + 1u] = vec4<u32>(found_v.w, 0u, 0u, 0u);
    p_loc[base + 2u] = vec4<u32>(
        bitcast<u32>(found_w.x),
        bitcast<u32>(found_w.y),
        bitcast<u32>(found_w.z),
        bitcast<u32>(found_w.w),
    );
}

// Zero-out grid mass + momentum atomics. One thread per node.
@compute @workgroup_size(WG_SIZE)
fn reset_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= u.vert_count) { return; }
    atomicStore(&g_mass[n], bitcast<u32>(0.0));
    atomicStore(&g_mom_x[n], bitcast<u32>(0.0));
    atomicStore(&g_mom_y[n], bitcast<u32>(0.0));
    atomicStore(&g_mom_z[n], bitcast<u32>(0.0));
    g_vel[n] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    g_vel_next[n] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
}

// Particle → grid scatter (APIC formula): each of the four owning
// tet's vertices receives `w_i · m_p · (v_p + C_p · (x_i − x_p))`.
@compute @workgroup_size(WG_SIZE)
fn p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= u.particle_count) { return; }

    let base = pid * 3u;
    let head = p_loc[base + 0u];
    let tail = p_loc[base + 1u];
    if (head.x == INVALID_TET) { return; }

    let vidx = array<u32, 4>(head.y, head.z, head.w, tail.x);
    let w_packed = p_loc[base + 2u];
    let w = vec4<f32>(
        bitcast<f32>(w_packed.x),
        bitcast<f32>(w_packed.y),
        bitcast<f32>(w_packed.z),
        bitcast<f32>(w_packed.w),
    );
    let weights = array<f32, 4>(w.x, w.y, w.z, w.w);

    let x = p_x[pid].xyz;
    let v = p_v[pid].xyz;
    let cr0 = p_C[pid * 3u + 0u].xyz;
    let cr1 = p_C[pid * 3u + 1u].xyz;
    let cr2 = p_C[pid * 3u + 2u].xyz;
    let mass = p_mass[pid];

    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let nid = vidx[k];
        let xk = mesh_vertices[nid].xyz;
        let dx = xk - x;
        // c · dx (each row of C is a 3-vector).
        let cdx = vec3<f32>(dot(cr0, dx), dot(cr1, dx), dot(cr2, dx));
        let momentum = weights[k] * mass * (v + cdx);
        atomic_add_mass(nid, weights[k] * mass);
        atomic_add_mom_x(nid, momentum.x);
        atomic_add_mom_y(nid, momentum.y);
        atomic_add_mom_z(nid, momentum.z);
    }
}

// Per-node: v = mom / mass + dt·gravity. Empty cells stay zero. The
// caller's CPU mirror only adds gravity to cells with mass — match that
// exactly so the parity test stays tight.
@compute @workgroup_size(WG_SIZE)
fn grid_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= u.vert_count) { return; }
    let m = bitcast<f32>(atomicLoad(&g_mass[n]));
    if (m <= 1.0e-30) {
        g_vel[n] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }
    let mx = bitcast<f32>(atomicLoad(&g_mom_x[n]));
    let my = bitcast<f32>(atomicLoad(&g_mom_y[n]));
    let mz = bitcast<f32>(atomicLoad(&g_mom_z[n]));
    var v = vec3<f32>(mx, my, mz) / m;
    v.x = v.x + u.dt * u.gravity_x;
    v.y = v.y + u.dt * u.gravity_y;
    v.z = v.z + u.dt * u.gravity_z;
    g_vel[n] = vec4<f32>(v.x, v.y, v.z, m);  // stash mass in .w for the smoothing pass
}

// One mass-weighted Jacobi sweep over the velocity field. Reads from
// `g_vel`, writes into `g_vel_next`. Caller copies back before applying
// BCs.
@compute @workgroup_size(WG_SIZE)
fn viscous_smoothing(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= u.vert_count) { return; }
    let v_in = g_vel[n];
    let m_self = v_in.w;
    if (m_self <= 1.0e-30) {
        g_vel_next[n] = v_in;
        return;
    }
    let lo = g_neighbour_offsets[n];
    let hi = g_neighbour_offsets[n + 1u];
    var weighted = vec3<f32>(0.0, 0.0, 0.0);
    var total_m: f32 = 0.0;
    for (var s: u32 = lo; s < hi; s = s + 1u) {
        let j = g_neighbour_indices[s];
        let vj = g_vel[j];
        if (vj.w > 1.0e-30) {
            weighted = weighted + vj.w * vj.xyz;
            total_m = total_m + vj.w;
        }
    }
    if (total_m <= 1.0e-30) {
        g_vel_next[n] = v_in;
        return;
    }
    let v_avg = weighted / total_m;
    let damping = min(u.dt * u.viscosity, 1.0);
    let v_new = v_in.xyz + damping * (v_avg - v_in.xyz);
    g_vel_next[n] = vec4<f32>(v_new.x, v_new.y, v_new.z, m_self);
}

// Copy g_vel_next → g_vel (ping-pong). Single sweep over the grid.
@compute @workgroup_size(WG_SIZE)
fn velocity_swap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= u.vert_count) { return; }
    g_vel[n] = g_vel_next[n];
}

// Apply Dirichlet boundary conditions. `g_bc_mask[i] == 1` → overwrite
// velocity with `g_bc_values[i].xyz`; otherwise leave the cell alone.
@compute @workgroup_size(WG_SIZE)
fn apply_bcs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = gid.x;
    if (n >= u.vert_count) { return; }
    if (g_bc_mask[n] == 0u) { return; }
    let bc = g_bc_values[n].xyz;
    let mass = g_vel[n].w;
    g_vel[n] = vec4<f32>(bc.x, bc.y, bc.z, mass);
}

// G2P: gather node velocities back to each particle, recompute the
// APIC affine matrix `C = Σ v_k ⊗ ∇N_k` using the same closed-form
// shape gradients the CPU mirror uses.
@compute @workgroup_size(WG_SIZE)
fn g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= u.particle_count) { return; }
    let base = pid * 3u;
    let head = p_loc[base + 0u];
    let tail = p_loc[base + 1u];
    if (head.x == INVALID_TET) {
        // Just advect by current velocity (CPU mirror semantics).
        let x = p_x[pid].xyz;
        let v = p_v[pid].xyz;
        let x_new = x + u.dt * v;
        p_x[pid] = vec4<f32>(x_new.x, x_new.y, x_new.z, 0.0);
        return;
    }

    let vidx = array<u32, 4>(head.y, head.z, head.w, tail.x);
    let w_packed = p_loc[base + 2u];
    let w = vec4<f32>(
        bitcast<f32>(w_packed.x),
        bitcast<f32>(w_packed.y),
        bitcast<f32>(w_packed.z),
        bitcast<f32>(w_packed.w),
    );
    let weights = array<f32, 4>(w.x, w.y, w.z, w.w);

    // Shape gradients (P1 tet) — D = [v1-v0|v2-v0|v3-v0], rows of
    // D^{-1} are the gradients of phi_{1,2,3}; phi_0 is the negative sum.
    let p0 = mesh_vertices[vidx[0]].xyz;
    let p1 = mesh_vertices[vidx[1]].xyz;
    let p2 = mesh_vertices[vidx[2]].xyz;
    let p3 = mesh_vertices[vidx[3]].xyz;
    let d0 = p1 - p0;
    let d1 = p2 - p0;
    let d2 = p3 - p0;
    let det = dot(d0, cross(d1, d2));
    let inv_det = 1.0 / det;
    let g1 = cross(d1, d2) * inv_det;
    let g2 = cross(d2, d0) * inv_det;
    let g3 = cross(d0, d1) * inv_det;
    let g0 = -(g1 + g2 + g3);
    let grads = array<vec3<f32>, 4>(g0, g1, g2, g3);

    var v_new = vec3<f32>(0.0, 0.0, 0.0);
    // Outer product accumulator for C (3 rows × 3 cols).
    var cr0 = vec3<f32>(0.0, 0.0, 0.0);
    var cr1 = vec3<f32>(0.0, 0.0, 0.0);
    var cr2 = vec3<f32>(0.0, 0.0, 0.0);
    for (var k: u32 = 0u; k < 4u; k = k + 1u) {
        let v_k = g_vel[vidx[k]].xyz;
        v_new = v_new + weights[k] * v_k;
        let g_k = grads[k];
        // outer = v_k · g_kᵀ — row i = v_k[i] · g_k
        cr0 = cr0 + v_k.x * g_k;
        cr1 = cr1 + v_k.y * g_k;
        cr2 = cr2 + v_k.z * g_k;
    }
    let x = p_x[pid].xyz;
    let x_new = x + u.dt * v_new;
    p_v[pid] = vec4<f32>(v_new.x, v_new.y, v_new.z, 0.0);
    p_x[pid] = vec4<f32>(x_new.x, x_new.y, x_new.z, 0.0);
    p_C[pid * 3u + 0u] = vec4<f32>(cr0.x, cr0.y, cr0.z, 0.0);
    p_C[pid * 3u + 1u] = vec4<f32>(cr1.x, cr1.y, cr1.z, 0.0);
    p_C[pid * 3u + 2u] = vec4<f32>(cr2.x, cr2.y, cr2.z, 0.0);
}
