// Stage 40 — Implicit MPM kernels.
//
// Entry points:
//   gs_sweep       — colored Gauss-Seidel update over the 27-point
//                    Newton Hessian. One thread per active cell in the
//                    current color group.
//   reset_grid     — zero per-cell mass / momentum / k_stencil.
//   p2g            — one thread per particle. Atomically scatters mass,
//                    momentum, and the 27×27 = 729-pair Hessian
//                    contribution into the grid. Uses f32-CAS atomics
//                    over u32 bitcast (same pattern as `mpm.wgsl`).
//   predict_grid   — one thread per cell. Computes
//                      v_pred = momentum / mass + g·dt
//                    with floor enforcement, sets the active flag, and
//                    seeds v_grid := v_pred for the GS sweep.
//   floor_reapply  — one thread per cell. Re-runs the post-solve floor
//                    clamp that the CPU path applies after GS.
//   g2p            — one thread per particle. Gathers v_grid into
//                    per-particle velocity, plus rebuilds the APIC
//                    affine matrix `p_C` in-place.
//   bingham_advect — Stage 40 v3 — one thread per particle. Applies the
//                    Bingham deviatoric return mapping to `p_C` (with
//                    C-magnitude capping), the per-particle floor +
//                    Bingham-Coulomb friction, then advects
//                    `p_pos += dt·p_vel`. After this pass the device
//                    state is fully consistent with the CPU's
//                    `apply_bingham_and_advect`, so the host only needs
//                    `p_pos` + `p_vel` (skip the `p_C` readback).
//
// Hessian stencil layout: `k_stencil[ci * 27 + s(δ)]` where
//   s(δx, δy, δz) = (δz + 1) * 9 + (δy + 1) * 3 + (δx + 1)
// and `STENCIL_DIAG = 13` indexes the on-diagonal entry.
//
// Atomic float layout: mass and k_stencil are flat `array<atomic<u32>>`
// of length g_total and g_total*27 respectively. Momentum is a single
// `array<atomic<u32>>` of length 3*g_total (xyz interleaved).
//
// Uniforms:
//   grid_size, color_n, dt, use_off_diag,
//   cell_size, inv_cell_size, gravity_y, floor_y, floor_friction,
//   mass_p, particle_volume, viscosity (μ_b), particle_count

const WG_SIZE: u32 = 64u;
const WG_PARTICLES: u32 = 64u;
const STENCIL_DIAG: u32 = 13u;

struct Uniforms {
    grid_size: u32,
    color_n: u32,
    use_off_diag: u32,
    particle_count: u32,

    dt: f32,
    cell_size: f32,
    inv_cell_size: f32,
    gravity_y: f32,

    floor_y: f32,
    floor_friction: f32,
    mass_p: f32,
    particle_volume: f32,

    viscosity: f32,
    grid_origin_x: f32,
    grid_origin_y: f32,
    grid_origin_z: f32,

    yield_stress: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<storage, read_write> g_mass_atomic: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> k_stencil_atomic: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> v_pred: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read> color_cells: array<u32>;
@group(0) @binding(4) var<storage, read_write> v_grid: array<vec3<f32>>;
@group(0) @binding(5) var<storage, read_write> g_momentum_atomic: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> p_pos: array<vec3<f32>>;
@group(0) @binding(7) var<storage, read_write> p_vel: array<vec3<f32>>;
// p_C is 9 floats per particle (mat3 stored as 3 vec3 columns); use
// a flat f32 array to dodge std430 padding surprises.
@group(0) @binding(8) var<storage, read_write> p_C: array<f32>;
@group(0) @binding(9) var<storage, read_write> g_active: array<u32>;
@group(1) @binding(0) var<uniform> U: Uniforms;

fn cell_flat(ix: i32, iy: i32, iz: i32, g: i32) -> i32 {
    if (ix < 0 || iy < 0 || iz < 0 || ix >= g || iy >= g || iz >= g) {
        return -1;
    }
    return ix + iy * g + iz * g * g;
}

fn stencil_idx(dx: i32, dy: i32, dz: i32) -> u32 {
    return u32((dz + 1) * 9 + (dy + 1) * 3 + (dx + 1));
}

fn quad_weights(fx: f32) -> vec3<f32> {
    let a = 1.5 - fx;
    let b = fx - 1.0;
    let c = fx - 0.5;
    return vec3<f32>(0.5 * a * a, 0.75 - b * b, 0.5 * c * c);
}

// ── f32 atomic-add via CAS on u32 bitcast. ─────────────────────────
fn atomic_add_mass(idx: u32, val: f32) {
    var old_bits = atomicLoad(&g_mass_atomic[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let ex = atomicCompareExchangeWeak(&g_mass_atomic[idx], old_bits, new_bits);
        if (ex.exchanged) { break; }
        old_bits = ex.old_value;
    }
}

fn atomic_add_momentum(idx: u32, val: f32) {
    var old_bits = atomicLoad(&g_momentum_atomic[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let ex = atomicCompareExchangeWeak(&g_momentum_atomic[idx], old_bits, new_bits);
        if (ex.exchanged) { break; }
        old_bits = ex.old_value;
    }
}

fn atomic_add_kstencil(idx: u32, val: f32) {
    var old_bits = atomicLoad(&k_stencil_atomic[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let ex = atomicCompareExchangeWeak(&k_stencil_atomic[idx], old_bits, new_bits);
        if (ex.exchanged) { break; }
        old_bits = ex.old_value;
    }
}

// ── reset_grid ─────────────────────────────────────────────────────
@compute @workgroup_size(WG_SIZE)
fn reset_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let g = U.grid_size;
    let g_total = g * g * g;
    if (ci >= g_total) {
        return;
    }
    atomicStore(&g_mass_atomic[ci], 0u);
    let mb = ci * 3u;
    atomicStore(&g_momentum_atomic[mb + 0u], 0u);
    atomicStore(&g_momentum_atomic[mb + 1u], 0u);
    atomicStore(&g_momentum_atomic[mb + 2u], 0u);
    for (var s: u32 = 0u; s < 27u; s = s + 1u) {
        atomicStore(&k_stencil_atomic[ci * 27u + s], 0u);
    }
    g_active[ci] = 0u;
    v_pred[ci] = vec3<f32>(0.0, 0.0, 0.0);
    v_grid[ci] = vec3<f32>(0.0, 0.0, 0.0);
}

// ── p2g ────────────────────────────────────────────────────────────
@compute @workgroup_size(WG_PARTICLES)
fn p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= U.particle_count) {
        return;
    }
    let g = i32(U.grid_size);
    let origin = vec3<f32>(U.grid_origin_x, U.grid_origin_y, U.grid_origin_z);
    let dx = U.cell_size;
    let inv_dx = U.inv_cell_size;
    let mass_p = U.mass_p;
    let mu_b = U.viscosity;
    let vol_p = U.particle_volume;

    let pos = p_pos[pid];
    let vel = p_vel[pid];

    // APIC affine matrix C — 9 floats per particle, stored as
    //   p_C[pid*9 + 0..9] = [c00, c10, c20, c01, c11, c21, c02, c12, c22]
    // i.e. column-major. Mirrors the Mat3 layout on the CPU.
    let cb = pid * 9u;
    let c0 = vec3<f32>(p_C[cb + 0u], p_C[cb + 1u], p_C[cb + 2u]);
    let c1 = vec3<f32>(p_C[cb + 3u], p_C[cb + 4u], p_C[cb + 5u]);
    let c2 = vec3<f32>(p_C[cb + 6u], p_C[cb + 7u], p_C[cb + 8u]);
    let C = mat3x3<f32>(c0, c1, c2);

    let xp = (pos - origin) * inv_dx;
    let base_xyz = vec3<f32>(floor(xp.x - 0.5), floor(xp.y - 0.5), floor(xp.z - 0.5));
    let base = vec3<i32>(i32(base_xyz.x), i32(base_xyz.y), i32(base_xyz.z));
    let fx = xp - base_xyz;
    let wx = quad_weights(fx.x);
    let wy = quad_weights(fx.y);
    let wz = quad_weights(fx.z);

    // Cache 27 stencil entries (weight, cell_idx, ix, iy, iz).
    var w27: array<f32, 27>;
    var ci27: array<i32, 27>;
    var ix27: array<i32, 27>;
    var iy27: array<i32, 27>;
    var iz27: array<i32, 27>;
    for (var i: i32 = 0; i < 3; i = i + 1) {
        for (var j: i32 = 0; j < 3; j = j + 1) {
            for (var k: i32 = 0; k < 3; k = k + 1) {
                let s = u32(i * 9 + j * 3 + k);
                let wi = wx[i] * wy[j] * wz[k];
                let xi = base.x + i;
                let yi = base.y + j;
                let zi = base.z + k;
                w27[s] = wi;
                ix27[s] = xi;
                iy27[s] = yi;
                iz27[s] = zi;
                ci27[s] = cell_flat(xi, yi, zi, g);
            }
        }
    }

    // Mass + momentum scatter.
    for (var s: u32 = 0u; s < 27u; s = s + 1u) {
        let ci = ci27[s];
        if (ci < 0) { continue; }
        let cu = u32(ci);
        let w = w27[s];
        let node_pos = origin + vec3<f32>(
            f32(ix27[s]) + 0.5,
            f32(iy27[s]) + 0.5,
            f32(iz27[s]) + 0.5,
        ) * dx;
        let xn_xp = node_pos - pos;
        let aff = C * xn_xp;
        let mom = w * mass_p * (vel + aff);
        atomic_add_mass(cu, w * mass_p);
        let mb = cu * 3u;
        atomic_add_momentum(mb + 0u, mom.x);
        atomic_add_momentum(mb + 1u, mom.y);
        atomic_add_momentum(mb + 2u, mom.z);
    }

    // 27×27 Hessian scatter.
    let kp = mu_b * vol_p * 4.0 * inv_dx * inv_dx;
    for (var a: u32 = 0u; a < 27u; a = a + 1u) {
        let ca = ci27[a];
        if (ca < 0) { continue; }
        let wa = w27[a];
        let ax = ix27[a];
        let ay = iy27[a];
        let az = iz27[a];
        for (var b: u32 = 0u; b < 27u; b = b + 1u) {
            let cb_idx = ci27[b];
            if (cb_idx < 0) { continue; }
            let bx = ix27[b];
            let by = iy27[b];
            let bz = iz27[b];
            let dxi = bx - ax;
            let dyi = by - ay;
            let dzi = bz - az;
            if (dxi < -1 || dxi > 1 || dyi < -1 || dyi > 1 || dzi < -1 || dzi > 1) {
                continue;
            }
            let s = stencil_idx(dxi, dyi, dzi);
            let wb = w27[b];
            atomic_add_kstencil(u32(ca) * 27u + s, kp * wa * wb);
        }
    }
}

// ── predict_grid ───────────────────────────────────────────────────
// One thread per cell. Reads atomic mass/momentum buffers (as plain
// u32 bitcast → f32) and writes v_pred + v_grid + active flag.
@compute @workgroup_size(WG_SIZE)
fn predict_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let g = U.grid_size;
    let g_total = g * g * g;
    if (ci >= g_total) {
        return;
    }
    let m_bits = atomicLoad(&g_mass_atomic[ci]);
    let m = bitcast<f32>(m_bits);
    if (m <= 0.0) {
        g_active[ci] = 0u;
        v_pred[ci] = vec3<f32>(0.0, 0.0, 0.0);
        v_grid[ci] = vec3<f32>(0.0, 0.0, 0.0);
        return;
    }
    let mb = ci * 3u;
    let mx = bitcast<f32>(atomicLoad(&g_momentum_atomic[mb + 0u]));
    let my = bitcast<f32>(atomicLoad(&g_momentum_atomic[mb + 1u]));
    let mz = bitcast<f32>(atomicLoad(&g_momentum_atomic[mb + 2u]));
    var v = vec3<f32>(mx, my, mz) / m;
    v.y = v.y + U.gravity_y * U.dt;

    // Decode (ix, iy, iz).
    let gi = i32(g);
    let iz = i32(ci) / (gi * gi);
    let iy = (i32(ci) / gi) % gi;
    let ix = i32(ci) % gi;
    let node_y = U.grid_origin_y + (f32(iy) + 0.5) * U.cell_size;
    if (node_y < U.floor_y + U.cell_size) {
        let depth = max(U.floor_y + U.cell_size - node_y, 0.0) / U.cell_size;
        if (node_y < U.floor_y && v.y < 0.0) {
            v.y = 0.0;
        } else if (v.y < 0.0) {
            v.y = v.y * (1.0 - depth);
        }
        let damp = 1.0 - U.floor_friction * depth;
        v.x = v.x * damp;
        v.z = v.z * damp;
    }
    v_pred[ci] = v;
    v_grid[ci] = v;
    g_active[ci] = 1u;
    // Suppress unused-binding warning when neither branch touches ix.
    let _unused = ix;
}

// ── floor_reapply ──────────────────────────────────────────────────
@compute @workgroup_size(WG_SIZE)
fn floor_reapply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let ci = gid.x;
    let g = U.grid_size;
    let g_total = g * g * g;
    if (ci >= g_total) {
        return;
    }
    if (g_active[ci] == 0u) {
        return;
    }
    let gi = i32(g);
    let iy = (i32(ci) / gi) % gi;
    let node_y = U.grid_origin_y + (f32(iy) + 0.5) * U.cell_size;
    if (node_y < U.floor_y) {
        var v = v_grid[ci];
        if (v.y < 0.0) { v.y = 0.0; }
        let damp = 1.0 - U.floor_friction;
        v.x = v.x * damp;
        v.z = v.z * damp;
        v_grid[ci] = v;
    }
}

// ── g2p ────────────────────────────────────────────────────────────
// Gather v_grid → particle velocity. APIC affine update is NOT done
// here — the CPU still owns the Bingham return-mapping path in v2,
// and it recomputes C from the read-back velocities to avoid a
// second device→host trip for C.
@compute @workgroup_size(WG_PARTICLES)
fn g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= U.particle_count) {
        return;
    }
    let g = i32(U.grid_size);
    let origin = vec3<f32>(U.grid_origin_x, U.grid_origin_y, U.grid_origin_z);
    let dx = U.cell_size;
    let inv_dx = U.inv_cell_size;

    let pos = p_pos[pid];
    let xp = (pos - origin) * inv_dx;
    let base_xyz = vec3<f32>(floor(xp.x - 0.5), floor(xp.y - 0.5), floor(xp.z - 0.5));
    let base = vec3<i32>(i32(base_xyz.x), i32(base_xyz.y), i32(base_xyz.z));
    let fx = xp - base_xyz;
    let wx = quad_weights(fx.x);
    let wy = quad_weights(fx.y);
    let wz = quad_weights(fx.z);

    var v_new = vec3<f32>(0.0, 0.0, 0.0);
    // Reconstruct APIC C (column-major). The Bingham return mapping
    // on the CPU side reads this back through `p_C` so we update it
    // here to keep the device the single source of truth for the
    // pre-projection affine matrix.
    var c0 = vec3<f32>(0.0, 0.0, 0.0);
    var c1 = vec3<f32>(0.0, 0.0, 0.0);
    var c2 = vec3<f32>(0.0, 0.0, 0.0);

    for (var i: i32 = 0; i < 3; i = i + 1) {
        for (var j: i32 = 0; j < 3; j = j + 1) {
            for (var k: i32 = 0; k < 3; k = k + 1) {
                let w = wx[i] * wy[j] * wz[k];
                let xi = base.x + i;
                let yi = base.y + j;
                let zi = base.z + k;
                let cf = cell_flat(xi, yi, zi, g);
                if (cf < 0) { continue; }
                let cu = u32(cf);
                let v_node = v_grid[cu];
                let node_pos = origin + vec3<f32>(f32(xi) + 0.5, f32(yi) + 0.5, f32(zi) + 0.5) * dx;
                let xn_xp = node_pos - pos;
                v_new = v_new + w * v_node;
                // outer(v_node, xn_xp) columns: col0 = v_node * xn_xp.x, etc.
                let scale = 4.0 * w * inv_dx * inv_dx;
                c0 = c0 + v_node * (xn_xp.x * scale);
                c1 = c1 + v_node * (xn_xp.y * scale);
                c2 = c2 + v_node * (xn_xp.z * scale);
            }
        }
    }

    p_vel[pid] = v_new;
    let cb = pid * 9u;
    p_C[cb + 0u] = c0.x; p_C[cb + 1u] = c0.y; p_C[cb + 2u] = c0.z;
    p_C[cb + 3u] = c1.x; p_C[cb + 4u] = c1.y; p_C[cb + 5u] = c1.z;
    p_C[cb + 6u] = c2.x; p_C[cb + 7u] = c2.y; p_C[cb + 8u] = c2.z;
}

// ── gs_sweep — colored Gauss-Seidel sweep over the 27-stencil. ────
//
// Reads g_mass via the atomic buffer (as a regular u32→f32 bitcast,
// no atomic semantics needed: nothing else writes to it during the
// sweep). Same for k_stencil.
@compute @workgroup_size(WG_SIZE)
fn gs_sweep(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= U.color_n) {
        return;
    }
    let ci = color_cells[i];
    let g = i32(U.grid_size);
    let g2 = g * g;
    let m = bitcast<f32>(atomicLoad(&g_mass_atomic[ci]));
    let k_diag = bitcast<f32>(atomicLoad(&k_stencil_atomic[ci * 27u + STENCIL_DIAG]));
    let denom = m + U.dt * k_diag;
    if (denom <= 0.0) {
        return;
    }

    let ix = i32(ci) % g;
    let iy = (i32(ci) / g) % g;
    let iz = i32(ci) / g2;

    var off = vec3<f32>(0.0, 0.0, 0.0);
    if (U.use_off_diag != 0u) {
        for (var dz = -1; dz <= 1; dz = dz + 1) {
            for (var dy = -1; dy <= 1; dy = dy + 1) {
                for (var dx = -1; dx <= 1; dx = dx + 1) {
                    if (dx == 0 && dy == 0 && dz == 0) {
                        continue;
                    }
                    let cj = cell_flat(ix + dx, iy + dy, iz + dz, g);
                    if (cj < 0) {
                        continue;
                    }
                    let s = stencil_idx(dx, dy, dz);
                    let kij = bitcast<f32>(atomicLoad(&k_stencil_atomic[ci * 27u + s]));
                    if (kij != 0.0) {
                        off = off + kij * v_grid[cj];
                    }
                }
            }
        }
    }

    let rhs = m * v_pred[ci] - U.dt * off;
    let v_new = rhs / denom;
    v_grid[ci] = v_new;
}

// ── bingham_advect — Stage 40 v3 ───────────────────────────────────
//
// Mirror of `apply_bingham_and_advect` in
// `crates/molla/src/compute/implicit_mpm.rs`. Inputs are the
// post-G2P `p_vel` + `p_C` written by the `g2p` pass and `p_pos`
// (untouched since upload). Outputs are the projected `p_vel`,
// `p_C`, and advected `p_pos`. One thread per particle.
@compute @workgroup_size(WG_PARTICLES)
fn bingham_advect(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= U.particle_count) {
        return;
    }
    let dt = U.dt;
    let dx = U.cell_size;
    let mu_b = U.viscosity;
    let tau_y_ = 2.0 * mu_b; // we never need 2μ_b on its own; keep μ_b only
    // (μ_b already in scope; this local is unused but kept inline so
    // the compiler folds it away — it's a marker so future readers
    // see the 2μ_b factor.)
    let _unused = tau_y_;

    var pos = p_pos[pid];
    var vel = p_vel[pid];
    let cb = pid * 9u;
    // p_C is column-major: cb..cb+8 → [c0.x, c0.y, c0.z, c1.x, c1.y, c1.z, c2.x, c2.y, c2.z]
    var c0 = vec3<f32>(p_C[cb + 0u], p_C[cb + 1u], p_C[cb + 2u]);
    var c1 = vec3<f32>(p_C[cb + 3u], p_C[cb + 4u], p_C[cb + 5u]);
    var c2 = vec3<f32>(p_C[cb + 6u], p_C[cb + 7u], p_C[cb + 8u]);

    // sym(C) = 0.5 * (C + Cᵀ). For column-major mat3 with columns
    // (c0, c1, c2), Cᵀ has columns (row0, row1, row2). Element-wise:
    //   sym[i, j] = 0.5 * (C[i, j] + C[j, i])
    // For each column k, sym.col(k)[i] = 0.5 * (c_k[i] + c_i[k]).
    let s00 = c0.x;
    let s11 = c1.y;
    let s22 = c2.z;
    let s01 = 0.5 * (c1.x + c0.y);
    let s02 = 0.5 * (c2.x + c0.z);
    let s12 = 0.5 * (c2.y + c1.z);
    let tr3 = (s00 + s11 + s22) / 3.0;

    // dev = sym - I * tr3.
    let d00 = s00 - tr3;
    let d11 = s11 - tr3;
    let d22 = s22 - tr3;
    let d01 = s01;
    let d02 = s02;
    let d12 = s12;

    // Frobenius norm of a symmetric 3×3 ‖dev‖² = Σᵢ dii² + 2·Σᵢ<j dij².
    let nd2 = d00 * d00 + d11 * d11 + d22 * d22
            + 2.0 * (d01 * d01 + d02 * d02 + d12 * d12);
    let nd = sqrt(nd2);
    let trial_norm = 2.0 * mu_b * nd;

    // skew = C - sym. Columns:
    //   skew.col(0) = (c0.x - s00, c0.y - s01, c0.z - s02)
    //   skew.col(1) = (c1.x - s01, c1.y - s11, c1.z - s12)
    //   skew.col(2) = (c2.x - s02, c2.y - s12, c2.z - s22)
    let sk0 = vec3<f32>(c0.x - s00, c0.y - s01, c0.z - s02);
    let sk1 = vec3<f32>(c1.x - s01, c1.y - s11, c1.z - s12);
    let sk2 = vec3<f32>(c2.x - s02, c2.y - s12, c2.z - s22);

    if (trial_norm <= U.yield_stress || nd < 1.0e-12) {
        // Plug regime. c_new = I*tr3 + skew*0.5.
        c0 = vec3<f32>(tr3, 0.0, 0.0) + sk0 * 0.5;
        c1 = vec3<f32>(0.0, tr3, 0.0) + sk1 * 0.5;
        c2 = vec3<f32>(0.0, 0.0, tr3) + sk2 * 0.5;
        let lambda_plug = U.yield_stress / (2.0 * mu_b * dx);
        let damp = max(1.0 - dt * lambda_plug, 0.0);
        vel = vel * damp;
    } else {
        let ratio = 1.0 - U.yield_stress / trial_norm;
        // dev_new = dev * ratio. Symmetric.
        let dn00 = d00 * ratio;
        let dn11 = d11 * ratio;
        let dn22 = d22 * ratio;
        let dn01 = d01 * ratio;
        let dn02 = d02 * ratio;
        let dn12 = d12 * ratio;
        // c_new = dev_new + I*tr3 + skew.
        c0 = vec3<f32>(dn00 + tr3, dn01, dn02) + sk0;
        c1 = vec3<f32>(dn01, dn11 + tr3, dn12) + sk1;
        c2 = vec3<f32>(dn02, dn12, dn22 + tr3) + sk2;
        // ‖c_new‖_F: sum of squares of all 9 entries.
        let nc2 = dot(c0, c0) + dot(c1, c1) + dot(c2, c2);
        let nc = sqrt(nc2);
        let cap = 0.5 / dt;
        if (nc > cap) {
            let s = cap / nc;
            c0 = c0 * s;
            c1 = c1 * s;
            c2 = c2 * s;
            vel = vel * s;
        }
    }

    // Per-particle floor enforcement.
    let height_above_floor = max(pos.y - U.floor_y, 0.0);
    if (pos.y + vel.y * dt < U.floor_y && vel.y < 0.0) {
        vel.y = 0.0;
        let damp = 1.0 - U.floor_friction;
        vel.x = vel.x * damp;
        vel.z = vel.z * damp;
    }
    if (height_above_floor < dx) {
        var ratio = height_above_floor / dx;
        ratio = clamp(ratio, 0.0, 1.0);
        let depth = 1.0 - ratio;
        // density = mass_p / particle_volume.
        let density = U.mass_p / U.particle_volume;
        let particle_extent = max(pow(U.particle_volume, 1.0 / 3.0), 1.0e-6);
        let dv_cap = dt * U.yield_stress / (density * particle_extent) * depth;
        let vh = vec3<f32>(vel.x, 0.0, vel.z);
        let vh_mag = length(vh);
        if (vh_mag <= dv_cap) {
            vel.x = 0.0;
            vel.z = 0.0;
            c0 = vec3<f32>(0.0, 0.0, 0.0);
            c1 = vec3<f32>(0.0, 0.0, 0.0);
            c2 = vec3<f32>(0.0, 0.0, 0.0);
        } else {
            let scale = (vh_mag - dv_cap) / vh_mag;
            vel.x = vel.x * scale;
            vel.z = vel.z * scale;
            c0 = c0 * scale;
            c1 = c1 * scale;
            c2 = c2 * scale;
        }
    }

    p_vel[pid] = vel;
    var new_pos = pos + vel * dt;
    if (new_pos.y < U.floor_y) {
        new_pos.y = U.floor_y;
    }
    p_pos[pid] = new_pos;
    p_C[cb + 0u] = c0.x; p_C[cb + 1u] = c0.y; p_C[cb + 2u] = c0.z;
    p_C[cb + 3u] = c1.x; p_C[cb + 4u] = c1.y; p_C[cb + 5u] = c1.z;
    p_C[cb + 6u] = c2.x; p_C[cb + 7u] = c2.y; p_C[cb + 8u] = c2.z;
}
