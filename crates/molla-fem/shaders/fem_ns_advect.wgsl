// Stage 41 v3 — WGSL Taylor-Hood **convective** RHS for transient
// Navier-Stokes. Mirrors `assemble_convection_block_with_supg` in
// `crates/molla-fem/src/stokes.rs`.
//
// One workgroup invocation per cell. For each tri:
//   * load the 6-node P2 velocity from `u_old`,
//   * loop the 6-point Dunavant rule (same as the velocity/divergence
//     assembly kernels),
//   * compute `(u·∇)u` at each quadrature point,
//   * scatter `∫ N_i · (u·∇)u dx` into the global `f_conv` buffer via
//     `atomic_add_f32`.
//
// SUPG: if `alpha > 0` the test function becomes `N_i + τ · (u·∇N_i)`
// with `τ = α · (h / (2|u|)) · ξ(Pe)` and `ξ(Pe) = coth(Pe) − 1/Pe`.
// Set `alpha = 0` for plain Galerkin (used in the parity test —
// CPU vs GPU run with identical alpha so the two paths exercise the
// same code path).
//
// IMEX context: the host computes
//   `rhs = (M/dt) · u_old + f_ext − f_conv`
// and feeds that into the Uzawa-CG solve. The shader writes ONLY
// `f_conv` — the host owns the rest of the RHS assembly.

struct NsAdvectUniforms {
    cell_count: u32,
    n_v: u32,            // velocity DoF count = n_v_nodes * dim
    dim: u32,            // tri-only v3 → always 2
    _pad0: u32,
    viscosity: f32,
    supg_alpha: f32,     // 0 = plain Galerkin; 1 = textbook Hughes
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read> mesh_vertices : array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> mesh_cells : array<u32>;
@group(0) @binding(2) var<storage, read> mesh_cell_offsets : array<u32>;
@group(0) @binding(3) var<storage, read> p2_dof_table : array<u32>;

@group(1) @binding(0) var<storage, read> u_old : array<f32>;
@group(1) @binding(1) var<storage, read_write> f_conv : array<atomic<u32>>;

@group(2) @binding(0) var<uniform> u : NsAdvectUniforms;

const WG_SIZE: u32 = 64u;

// Atomic-add f32 via u32 bit-cast CAS loop — WGSL has no atomic<f32>.
fn atomic_add_f32(slot: u32, value: f32) {
    if (value == 0.0) { return; }
    var old_bits = atomicLoad(&f_conv[slot]);
    loop {
        let old_f = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_f + value);
        let r = atomicCompareExchangeWeak(&f_conv[slot], old_bits, new_bits);
        if (r.exchanged) {
            return;
        }
        old_bits = r.old_value;
    }
}

fn tri_area_2d(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> f32 {
    let dx1 = p1.x - p0.x;
    let dy1 = p1.y - p0.y;
    let dx2 = p2.x - p0.x;
    let dy2 = p2.y - p0.y;
    return 0.5 * abs(dx1 * dy2 - dx2 * dy1);
}

fn tri_shape_grads_2d(p0: vec3<f32>, p1: vec3<f32>, p2: vec3<f32>) -> array<vec2<f32>, 3> {
    let two_a_signed = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
    let inv = 1.0 / two_a_signed;
    let g0 = vec2<f32>(p1.y - p2.y, p2.x - p1.x) * inv;
    let g1 = vec2<f32>(p2.y - p0.y, p0.x - p2.x) * inv;
    let g2 = vec2<f32>(p0.y - p1.y, p1.x - p0.x) * inv;
    return array<vec2<f32>, 3>(g0, g1, g2);
}

// P2 triangle shape values at barycentric `bary`. Layout matches the
// CPU: [c0, c1, c2, e01, e12, e20]. Mirrors `p2_tri_shape_functions`.
fn p2_tri_shape_at(bary: vec3<f32>) -> array<f32, 6> {
    let l0 = bary.x;
    let l1 = bary.y;
    let l2 = bary.z;
    let c0 = l0 * (2.0 * l0 - 1.0);
    let c1 = l1 * (2.0 * l1 - 1.0);
    let c2 = l2 * (2.0 * l2 - 1.0);
    let e01 = 4.0 * l0 * l1;
    let e12 = 4.0 * l1 * l2;
    let e20 = 4.0 * l2 * l0;
    return array<f32, 6>(c0, c1, c2, e01, e12, e20);
}

// P2 triangle gradients at `bary` given the three corner gradients.
fn p2_tri_grads_at(
    bary: vec3<f32>,
    g_lambda: array<vec2<f32>, 3>,
) -> array<vec2<f32>, 6> {
    let c0 = g_lambda[0] * (4.0 * bary.x - 1.0);
    let c1 = g_lambda[1] * (4.0 * bary.y - 1.0);
    let c2 = g_lambda[2] * (4.0 * bary.z - 1.0);
    let e01 = (g_lambda[0] * bary.y + g_lambda[1] * bary.x) * 4.0;
    let e12 = (g_lambda[1] * bary.z + g_lambda[2] * bary.y) * 4.0;
    let e20 = (g_lambda[2] * bary.x + g_lambda[0] * bary.z) * 4.0;
    return array<vec2<f32>, 6>(c0, c1, c2, e01, e12, e20);
}

// 6-point Dunavant rule barycentrics — identical to fem_stokes_assemble.wgsl.
fn qpt_bary(q: u32) -> vec3<f32> {
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
    if (q < 3u) {
        return 0.109951743655322;
    }
    return 0.223381589678011;
}

// Hughes ξ(Pe) — branched for accuracy near 0 / ∞. Matches the CPU
// `hughes_xi` helper in stokes.rs.
fn hughes_xi(pe: f32) -> f32 {
    if (pe < 1.0e-3) {
        return pe / 3.0;
    }
    if (pe > 50.0) {
        return 1.0 - 1.0 / pe;
    }
    let e2 = exp(2.0 * pe);
    let coth = (e2 + 1.0) / (e2 - 1.0);
    return coth - 1.0 / pe;
}

@compute @workgroup_size(WG_SIZE)
fn assemble_convection(@builtin(global_invocation_id) gid: vec3<u32>) {
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
    // Characteristic length — same as the CPU SUPG path.
    let h = sqrt(area);

    // Cache the six node u-values (dim=2 → two floats per node).
    let p2_base = cid * 6u;
    var ux: array<f32, 6>;
    var uy: array<f32, 6>;
    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let node = p2_dof_table[p2_base + i];
        let base = node * u.dim;
        ux[i] = u_old[base + 0u];
        uy[i] = u_old[base + 1u];
    }

    // Local RHS contribution `f_local[i * dim + c]` (12 floats).
    var f_local: array<f32, 12>;
    for (var i: u32 = 0u; i < 12u; i = i + 1u) {
        f_local[i] = 0.0;
    }

    for (var q: u32 = 0u; q < 6u; q = q + 1u) {
        let bary = qpt_bary(q);
        let n_vals = p2_tri_shape_at(bary);
        let g_vals = p2_tri_grads_at(bary, g_lambda);

        // Interpolate u and ∇u at quadrature point.
        var uxq: f32 = 0.0;
        var uyq: f32 = 0.0;
        var dux_dx: f32 = 0.0;
        var dux_dy: f32 = 0.0;
        var duy_dx: f32 = 0.0;
        var duy_dy: f32 = 0.0;
        for (var i: u32 = 0u; i < 6u; i = i + 1u) {
            uxq = uxq + n_vals[i] * ux[i];
            uyq = uyq + n_vals[i] * uy[i];
            dux_dx = dux_dx + g_vals[i].x * ux[i];
            dux_dy = dux_dy + g_vals[i].y * ux[i];
            duy_dx = duy_dx + g_vals[i].x * uy[i];
            duy_dy = duy_dy + g_vals[i].y * uy[i];
        }
        // Convection: c = (u·∇)u.
        let cx = uxq * dux_dx + uyq * dux_dy;
        let cy = uxq * duy_dx + uyq * duy_dy;

        let umag = sqrt(uxq * uxq + uyq * uyq);
        var tau: f32 = 0.0;
        if (u.supg_alpha > 0.0 && umag > 1.0e-30) {
            var pe: f32;
            if (u.viscosity > 1.0e-30) {
                pe = umag * h / (2.0 * u.viscosity);
            } else {
                pe = 1.0e30;
            }
            tau = u.supg_alpha * (h / (2.0 * umag)) * hughes_xi(pe);
        }

        let w = qpt_w(q) * area;
        for (var i: u32 = 0u; i < 6u; i = i + 1u) {
            // SUPG test function: N_i + τ · (u·∇N_i).
            let u_dot_grad_ni = uxq * g_vals[i].x + uyq * g_vals[i].y;
            let test = n_vals[i] + tau * u_dot_grad_ni;
            f_local[i * 2u + 0u] = f_local[i * 2u + 0u] + w * test * cx;
            f_local[i * 2u + 1u] = f_local[i * 2u + 1u] + w * test * cy;
        }
    }

    // Scatter local into global f_conv (atomic add).
    for (var i: u32 = 0u; i < 6u; i = i + 1u) {
        let gi_node = p2_dof_table[p2_base + i];
        for (var c: u32 = 0u; c < u.dim; c = c + 1u) {
            let slot = gi_node * u.dim + c;
            let val = f_local[i * 2u + c];
            atomic_add_f32(slot, val);
        }
    }
}
