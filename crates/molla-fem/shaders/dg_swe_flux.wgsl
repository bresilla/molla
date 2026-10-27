// Stage 43 v4 — Per-face SWE numerical flux for DG-P1 on triangle
// meshes. Three entry points (HLL / HLLC / Roe) share a common bind
// group layout. Each kernel processes one interior face per thread:
//
//   1. Loads the left + right cell states (3 P1 nodes × 3 SWE
//      components = 9 floats per side) into registers.
//   2. Evaluates a 2-point Gauss-Legendre line quadrature along the
//      shared edge. At each quadrature point, interpolates the trace
//      state from both sides via linear (P1) edge basis functions,
//      calls the chosen Riemann solver, and accumulates a per-face
//      contribution scattered to four target DoFs (2 left, 2 right).
//   3. Atomic-adds the four contributions into the per-cell residual
//      buffer.
//
// The atomic add is a CAS-loop on u32 bitcast of f32 (WGSL has no
// `atomic<f32>`); same pattern as `mpm.wgsl` / `fem_elasticity_assemble.wgsl`.
//
// CPU oracle path lives in `fem::dg_element::compute_swe_numerical_flux_*`.
// Test driver lives in `tests/fem_dg_swe_gpu.rs`.
//
// Buffer layout (all f32 / u32, 9 floats per cell):
//   state[cell * 9 + i * 3 + c]   — trace state at node `i ∈ 0..3`,
//                                   component c ∈ {0:h, 1:hu, 2:hv}.
//   residual[cell * 9 + i * 3 + c] — atomic-add target (same shape).
//   faces[face * 8 + ...]          — packed face record:
//        [0] cell_l (u32, bitcast)
//        [1] cell_r (u32, bitcast)
//        [2] local_l0 (u32, bitcast)
//        [3] local_l1 (u32, bitcast)
//        [4] local_r0 (u32, bitcast)
//        [5] local_r1 (u32, bitcast)
//        [6] n.x (f32)
//        [7] n.y (f32)
//        [8] len (f32, stored at byte offset 32 in a separate slot;
//                see below — the packed layout uses 10 lanes).
//
// To keep alignment trivial we store one face as 10 f32 lanes (40 B),
// with `cell_*` / `local_*` lanes interpreted via `bitcast<u32>(...)`:
//   lane 0 : cell_l_u32
//   lane 1 : cell_r_u32
//   lane 2 : local_l0
//   lane 3 : local_l1
//   lane 4 : local_r0
//   lane 5 : local_r1
//   lane 6 : n.x
//   lane 7 : n.y
//   lane 8 : len
//   lane 9 : 0.0    (pad)

struct SweUniforms {
    n_faces: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    g: f32,
    _pad3: f32,
    _pad4: f32,
    _pad5: f32,
};

@group(0) @binding(0) var<storage, read>       state    : array<f32>;
@group(0) @binding(1) var<storage, read>       faces    : array<f32>;
@group(1) @binding(0) var<storage, read_write> residual : array<atomic<u32>>;
@group(1) @binding(1) var<uniform>             u        : SweUniforms;

const WG_SIZE     : u32 = 64u;
const FACE_STRIDE : u32 = 10u;
const COMPONENTS  : u32 = 3u;
const NODES_PER_C : u32 = 3u;     // P1 triangle → 3 nodes per cell.
const H_FLOOR     : f32 = 1.0e-8;
const H_SAFE      : f32 = 1.0e-12;

// f32 atomic-add via CAS loop on u32 bitcast. Same pattern as
// `mpm.wgsl` / `fem_elasticity_assemble.wgsl`. WGSL has no
// `atomic<f32>`.
fn atomic_add_f32(idx: u32, val: f32) {
    var old_bits = atomicLoad(&residual[idx]);
    loop {
        let new_val = bitcast<f32>(old_bits) + val;
        let new_bits = bitcast<u32>(new_val);
        let ex = atomicCompareExchangeWeak(&residual[idx], old_bits, new_bits);
        if (ex.exchanged) { break; }
        old_bits = ex.old_value;
    }
}

fn load_state(cell: u32, local: u32, out_h: ptr<function, f32>, out_hu: ptr<function, f32>, out_hv: ptr<function, f32>) {
    let base = cell * NODES_PER_C * COMPONENTS + local * COMPONENTS;
    *out_h  = state[base + 0u];
    *out_hu = state[base + 1u];
    *out_hv = state[base + 2u];
}

// SWE physical flux F(U)·n.
fn swe_flux_dot_n(h: f32, hu: f32, hv: f32, nx: f32, ny: f32, g: f32) -> vec3<f32> {
    let h_safe = max(h, H_SAFE);
    let u_vel  = hu / h_safe;
    let v_vel  = hv / h_safe;
    let fx = vec3<f32>(hu, hu * u_vel + 0.5 * g * h * h, hu * v_vel);
    let fy = vec3<f32>(hv, hv * u_vel,                   hv * v_vel + 0.5 * g * h * h);
    return vec3<f32>(
        fx.x * nx + fy.x * ny,
        fx.y * nx + fy.y * ny,
        fx.z * nx + fy.z * ny,
    );
}

// ─── HLL ─────────────────────────────────────────────────────────────
fn flux_hll(
    h_l_in: f32, hu_l_in: f32, hv_l_in: f32,
    h_r_in: f32, hu_r_in: f32, hv_r_in: f32,
    nx: f32, ny: f32, g: f32,
) -> vec3<f32> {
    let h_l = max(h_l_in, H_SAFE);
    let h_r = max(h_r_in, H_SAFE);
    let un_l = (hu_l_in * nx + hv_l_in * ny) / h_l;
    let un_r = (hu_r_in * nx + hv_r_in * ny) / h_r;
    let c_l = sqrt(g * h_l);
    let c_r = sqrt(g * h_r);

    let sqrt_h_l = sqrt(h_l);
    let sqrt_h_r = sqrt(h_r);
    let h_hat = 0.5 * (h_l + h_r);
    let denom = sqrt_h_l + sqrt_h_r;
    var un_hat: f32;
    if (denom > 1.0e-30) {
        un_hat = (un_l * sqrt_h_l + un_r * sqrt_h_r) / denom;
    } else {
        un_hat = 0.5 * (un_l + un_r);
    }
    let c_hat = sqrt(g * h_hat);
    let s_l = min(min(un_l - c_l, un_r - c_r), un_hat - c_hat);
    let s_r = max(max(un_l + c_l, un_r + c_r), un_hat + c_hat);

    let f_l = swe_flux_dot_n(h_l_in, hu_l_in, hv_l_in, nx, ny, g);
    let f_r = swe_flux_dot_n(h_r_in, hu_r_in, hv_r_in, nx, ny, g);

    if (s_l >= 0.0) {
        return f_l;
    }
    if (s_r <= 0.0) {
        return f_r;
    }
    let inv = 1.0 / (s_r - s_l);
    return vec3<f32>(
        (s_r * f_l.x - s_l * f_r.x + s_l * s_r * (h_r_in - h_l_in)) * inv,
        (s_r * f_l.y - s_l * f_r.y + s_l * s_r * (hu_r_in - hu_l_in)) * inv,
        (s_r * f_l.z - s_l * f_r.z + s_l * s_r * (hv_r_in - hv_l_in)) * inv,
    );
}

// ─── HLLC (SWE) ──────────────────────────────────────────────────────
fn flux_hllc(
    h_l_in: f32, hu_l_in: f32, hv_l_in: f32,
    h_r_in: f32, hu_r_in: f32, hv_r_in: f32,
    nx: f32, ny: f32, g: f32,
) -> vec3<f32> {
    let h_l = max(h_l_in, H_SAFE);
    let h_r = max(h_r_in, H_SAFE);
    let un_l = (hu_l_in * nx + hv_l_in * ny) / h_l;
    let un_r = (hu_r_in * nx + hv_r_in * ny) / h_r;
    let c_l = sqrt(g * h_l);
    let c_r = sqrt(g * h_r);

    let sqrt_h_l = sqrt(h_l);
    let sqrt_h_r = sqrt(h_r);
    let h_hat = 0.5 * (h_l + h_r);
    let denom = sqrt_h_l + sqrt_h_r;
    var un_hat: f32;
    if (denom > 1.0e-30) {
        un_hat = (un_l * sqrt_h_l + un_r * sqrt_h_r) / denom;
    } else {
        un_hat = 0.5 * (un_l + un_r);
    }
    let c_hat = sqrt(g * h_hat);
    let s_l = min(min(un_l - c_l, un_r - c_r), un_hat - c_hat);
    let s_r = max(max(un_l + c_l, un_r + c_r), un_hat + c_hat);

    let f_l = swe_flux_dot_n(h_l_in, hu_l_in, hv_l_in, nx, ny, g);
    let f_r = swe_flux_dot_n(h_r_in, hu_r_in, hv_r_in, nx, ny, g);

    if (s_l >= 0.0) { return f_l; }
    if (s_r <= 0.0) { return f_r; }

    // Star-region contact speed.
    let num = h_r * un_r * (s_r - un_r) - h_l * un_l * (s_l - un_l)
            + 0.5 * g * (h_l * h_l - h_r * h_r);
    let den = h_r * (s_r - un_r) - h_l * (s_l - un_l);
    var s_star: f32;
    if (abs(den) > 1.0e-30) {
        s_star = num / den;
    } else {
        s_star = 0.5 * (s_l + s_r);
    }

    let tx = -ny;
    let ty =  nx;

    // Pick the star-region side and build the star-state flux
    // `F_*K = F_K + s_K (U_*K - U_K)`.
    var h_k: f32;
    var hu_k: f32;
    var hv_k: f32;
    var un_k: f32;
    var s_k: f32;
    var f_k: vec3<f32>;
    if (s_star >= 0.0) {
        h_k = h_l; hu_k = hu_l_in; hv_k = hv_l_in; un_k = un_l; s_k = s_l; f_k = f_l;
    } else {
        h_k = h_r; hu_k = hu_r_in; hv_k = hv_r_in; un_k = un_r; s_k = s_r; f_k = f_r;
    }
    let h_star = h_k * (s_k - un_k) / (s_k - s_star);
    let u_vel = hu_k / h_k;
    let v_vel = hv_k / h_k;
    let ut_k  = u_vel * tx + v_vel * ty;
    let us_x  = s_star * nx + ut_k * tx;
    let us_y  = s_star * ny + ut_k * ty;
    let hu_star = h_star * us_x;
    let hv_star = h_star * us_y;
    return vec3<f32>(
        f_k.x + s_k * (h_star  - h_k),
        f_k.y + s_k * (hu_star - hu_k),
        f_k.z + s_k * (hv_star - hv_k),
    );
}

// ─── Roe (SWE) with Harten-Hyman entropy fix ─────────────────────────
fn smooth_abs(lam: f32, lam_l: f32, lam_r: f32) -> f32 {
    let delta = max(max(lam - lam_l, 0.0), max(lam_r - lam, 0.0));
    if (abs(lam) < delta) {
        return 0.5 * (lam * lam / max(delta, 1.0e-30) + delta);
    }
    return abs(lam);
}

fn flux_roe(
    h_l_in: f32, hu_l_in: f32, hv_l_in: f32,
    h_r_in: f32, hu_r_in: f32, hv_r_in: f32,
    nx: f32, ny: f32, g: f32,
) -> vec3<f32> {
    // Wet-dry guard.
    var h_l: f32; var hu_l: f32; var hv_l: f32;
    if (h_l_in < H_FLOOR) { h_l = H_FLOOR; hu_l = 0.0; hv_l = 0.0; }
    else { h_l = h_l_in; hu_l = hu_l_in; hv_l = hv_l_in; }
    var h_r: f32; var hu_r: f32; var hv_r: f32;
    if (h_r_in < H_FLOOR) { h_r = H_FLOOR; hu_r = 0.0; hv_r = 0.0; }
    else { h_r = h_r_in; hu_r = hu_r_in; hv_r = hv_r_in; }

    let tx = -ny;
    let ty =  nx;

    // Rotate to normal/tangential frame (note these are velocities, not
    // momenta — divided by h).
    let u_l_n = (hu_l * nx + hv_l * ny) / h_l;
    let u_l_t = (hu_l * tx + hv_l * ty) / h_l;
    let u_r_n = (hu_r * nx + hv_r * ny) / h_r;
    let u_r_t = (hu_r * tx + hv_r * ty) / h_r;

    let sqrt_h_l = sqrt(h_l);
    let sqrt_h_r = sqrt(h_r);
    let h_hat = 0.5 * (h_l + h_r);
    let denom = sqrt_h_l + sqrt_h_r;
    var un_hat: f32;
    var ut_hat: f32;
    if (denom > 1.0e-30) {
        un_hat = (u_l_n * sqrt_h_l + u_r_n * sqrt_h_r) / denom;
        ut_hat = (u_l_t * sqrt_h_l + u_r_t * sqrt_h_r) / denom;
    } else {
        un_hat = 0.5 * (u_l_n + u_r_n);
        ut_hat = 0.5 * (u_l_t + u_r_t);
    }
    let c_hat = sqrt(g * h_hat);

    let dh    = h_r - h_l;
    let d_hun = h_r * u_r_n - h_l * u_l_n;
    let d_hut = h_r * u_r_t - h_l * u_l_t;

    var inv_c: f32;
    if (c_hat > 1.0e-30) { inv_c = 1.0 / c_hat; } else { inv_c = 0.0; }
    let alpha_1 = 0.5 * (dh - (d_hun - un_hat * dh) * inv_c);
    let alpha_2 = d_hut - ut_hat * dh;
    let alpha_3 = 0.5 * (dh + (d_hun - un_hat * dh) * inv_c);

    let lam_1 = un_hat - c_hat;
    let lam_2 = un_hat;
    let lam_3 = un_hat + c_hat;

    let c_l = sqrt(g * h_l);
    let c_r = sqrt(g * h_r);
    let lam_1_l = u_l_n - c_l;
    let lam_1_r = u_r_n - c_r;
    let lam_3_l = u_l_n + c_l;
    let lam_3_r = u_r_n + c_r;

    let abs_lam_1 = smooth_abs(lam_1, lam_1_l, lam_1_r);
    let abs_lam_2 = abs(lam_2);
    let abs_lam_3 = smooth_abs(lam_3, lam_3_l, lam_3_r);

    // Right eigenvectors in the rotated frame.
    let r1 = vec3<f32>(1.0, un_hat - c_hat, ut_hat);
    let r2 = vec3<f32>(0.0, 0.0, 1.0);
    let r3 = vec3<f32>(1.0, un_hat + c_hat, ut_hat);

    let d_rot = vec3<f32>(
        abs_lam_1 * alpha_1 * r1.x + abs_lam_2 * alpha_2 * r2.x + abs_lam_3 * alpha_3 * r3.x,
        abs_lam_1 * alpha_1 * r1.y + abs_lam_2 * alpha_2 * r2.y + abs_lam_3 * alpha_3 * r3.y,
        abs_lam_1 * alpha_1 * r1.z + abs_lam_2 * alpha_2 * r2.z + abs_lam_3 * alpha_3 * r3.z,
    );

    let f_l = swe_flux_dot_n(h_l, hu_l, hv_l, nx, ny, g);
    let f_r = swe_flux_dot_n(h_r, hu_r, hv_r, nx, ny, g);

    // Rotate dissipation back to (h, hu, hv).
    let d_h  = d_rot.x;
    let d_hu = d_rot.y * nx + d_rot.z * tx;
    let d_hv = d_rot.y * ny + d_rot.z * ty;

    return vec3<f32>(
        0.5 * (f_l.x + f_r.x) - 0.5 * d_h,
        0.5 * (f_l.y + f_r.y) - 0.5 * d_hu,
        0.5 * (f_l.z + f_r.z) - 0.5 * d_hv,
    );
}

// ─── Common per-face driver ──────────────────────────────────────────
//
// Reads the face record + per-cell P1 trace, runs the 2-point
// Gauss-Legendre line quadrature, and atomic-adds the four-node
// contributions to the residual buffer. `kind == 0` HLL, `1` HLLC,
// `2` Roe.
fn process_face(face_idx: u32, kind: u32) {
    let base = face_idx * FACE_STRIDE;
    let cell_l   = bitcast<u32>(faces[base + 0u]);
    let cell_r   = bitcast<u32>(faces[base + 1u]);
    let local_l0 = bitcast<u32>(faces[base + 2u]);
    let local_l1 = bitcast<u32>(faces[base + 3u]);
    let local_r0 = bitcast<u32>(faces[base + 4u]);
    let local_r1 = bitcast<u32>(faces[base + 5u]);
    let nx       = faces[base + 6u];
    let ny       = faces[base + 7u];
    let length   = faces[base + 8u];

    // Load the four trace states (two per side at the face vertices).
    var h_l0: f32; var hu_l0: f32; var hv_l0: f32;
    var h_l1: f32; var hu_l1: f32; var hv_l1: f32;
    var h_r0: f32; var hu_r0: f32; var hv_r0: f32;
    var h_r1: f32; var hu_r1: f32; var hv_r1: f32;
    load_state(cell_l, local_l0, &h_l0, &hu_l0, &hv_l0);
    load_state(cell_l, local_l1, &h_l1, &hu_l1, &hv_l1);
    load_state(cell_r, local_r0, &h_r0, &hu_r0, &hv_r0);
    load_state(cell_r, local_r1, &h_r1, &hu_r1, &hv_r1);

    // 2-point Gauss-Legendre on [0, 1] (parametric edge coord).
    let gp = 1.0 / sqrt(3.0) / 2.0;
    var xis: array<f32, 2>;
    xis[0] = 0.5 - gp;
    xis[1] = 0.5 + gp;
    let w_q = 0.5 * length;

    // Accumulate per-DoF contributions (4 nodes × 3 components).
    var acc_l0 = vec3<f32>(0.0, 0.0, 0.0);
    var acc_l1 = vec3<f32>(0.0, 0.0, 0.0);
    var acc_r0 = vec3<f32>(0.0, 0.0, 0.0);
    var acc_r1 = vec3<f32>(0.0, 0.0, 0.0);

    for (var qi: u32 = 0u; qi < 2u; qi = qi + 1u) {
        let xi = xis[qi];
        // Trace interpolation on left.
        let h_l_q  = (1.0 - xi) * h_l0  + xi * h_l1;
        let hu_l_q = (1.0 - xi) * hu_l0 + xi * hu_l1;
        let hv_l_q = (1.0 - xi) * hv_l0 + xi * hv_l1;
        // Trace interpolation on right.
        let h_r_q  = (1.0 - xi) * h_r0  + xi * h_r1;
        let hu_r_q = (1.0 - xi) * hu_r0 + xi * hu_r1;
        let hv_r_q = (1.0 - xi) * hv_r0 + xi * hv_r1;

        var hhat = vec3<f32>(0.0, 0.0, 0.0);
        if (kind == 0u) {
            hhat = flux_hll(h_l_q, hu_l_q, hv_l_q, h_r_q, hu_r_q, hv_r_q, nx, ny, u.g);
        } else if (kind == 1u) {
            hhat = flux_hllc(h_l_q, hu_l_q, hv_l_q, h_r_q, hu_r_q, hv_r_q, nx, ny, u.g);
        } else {
            hhat = flux_roe(h_l_q, hu_l_q, hv_l_q, h_r_q, hu_r_q, hv_r_q, nx, ny, u.g);
        }

        // Edge basis at left node 0 / 1 and right node 0 / 1 at the
        // same physical quadrature point. Right side uses the same
        // (1−xi, xi) parameterisation because `local_r0` is the
        // image of the same physical vertex as `local_l0`.
        let nl_a = 1.0 - xi;
        let nl_b = xi;
        let nr_a = 1.0 - xi;
        let nr_b = xi;

        // Left side: r -= w_q * Ĥ · v_left, with v_left jump sign +1.
        acc_l0 = acc_l0 - vec3<f32>(w_q * hhat.x * nl_a, w_q * hhat.y * nl_a, w_q * hhat.z * nl_a);
        acc_l1 = acc_l1 - vec3<f32>(w_q * hhat.x * nl_b, w_q * hhat.y * nl_b, w_q * hhat.z * nl_b);
        // Right side: r -= w_q * Ĥ · v_right, with v_right jump sign −1.
        acc_r0 = acc_r0 - vec3<f32>(w_q * hhat.x * (-nr_a), w_q * hhat.y * (-nr_a), w_q * hhat.z * (-nr_a));
        acc_r1 = acc_r1 - vec3<f32>(w_q * hhat.x * (-nr_b), w_q * hhat.y * (-nr_b), w_q * hhat.z * (-nr_b));
    }

    // Atomic-scatter the four-node contributions.
    let base_l0 = cell_l * NODES_PER_C * COMPONENTS + local_l0 * COMPONENTS;
    let base_l1 = cell_l * NODES_PER_C * COMPONENTS + local_l1 * COMPONENTS;
    let base_r0 = cell_r * NODES_PER_C * COMPONENTS + local_r0 * COMPONENTS;
    let base_r1 = cell_r * NODES_PER_C * COMPONENTS + local_r1 * COMPONENTS;

    atomic_add_f32(base_l0 + 0u, acc_l0.x);
    atomic_add_f32(base_l0 + 1u, acc_l0.y);
    atomic_add_f32(base_l0 + 2u, acc_l0.z);
    atomic_add_f32(base_l1 + 0u, acc_l1.x);
    atomic_add_f32(base_l1 + 1u, acc_l1.y);
    atomic_add_f32(base_l1 + 2u, acc_l1.z);
    atomic_add_f32(base_r0 + 0u, acc_r0.x);
    atomic_add_f32(base_r0 + 1u, acc_r0.y);
    atomic_add_f32(base_r0 + 2u, acc_r0.z);
    atomic_add_f32(base_r1 + 0u, acc_r1.x);
    atomic_add_f32(base_r1 + 1u, acc_r1.y);
    atomic_add_f32(base_r1 + 2u, acc_r1.z);
}

@compute @workgroup_size(WG_SIZE)
fn compute_swe_hll_flux_per_face(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n_faces) { return; }
    process_face(i, 0u);
}

@compute @workgroup_size(WG_SIZE)
fn compute_swe_hllc_flux_per_face(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n_faces) { return; }
    process_face(i, 1u);
}

@compute @workgroup_size(WG_SIZE)
fn compute_swe_roe_flux_per_face(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.n_faces) { return; }
    process_face(i, 2u);
}
