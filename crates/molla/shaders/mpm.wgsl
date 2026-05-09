// Stage 24d v1 — Material-Point-Method (MPM) for granular materials.
//
// **Implementation choice: explicit symplectic time-stepping.** The PLAN
// originally called for an implicit backward-Euler grid solve via CG;
// we deliberately fall back to explicit because:
//   1. The textbook implicit MPM requires a matrix-free CG solver,
//      multi-pass Hessian-vector products on the grid, and a
//      line-search / damping strategy — easily 10× the WGSL surface
//      of the explicit path. Multi-week scope on its own.
//   2. Sand-grade Drucker-Prager rheology is dissipative; explicit MPM
//      with APIC transfer + a small dt remains stable in practice.
//      Stomakhin 2013 (MPM snow) and most production sand demos run
//      explicit.
//   3. Per the task brief: "explicit MPM is a valid fallback" — so we
//      ship a working v1 and leave the implicit upgrade for 24d.1.
//
// CFL caveat: explicit MPM needs `dt < cell_size / max(c_p, |v|)`
// where `c_p ≈ sqrt(E / ρ)` is the wave speed. For E = 1e5 Pa,
// ρ = 1500 kg/m³, cell_size = 0.05 m we get dt ≲ 6 ms — fine for
// real-time at 240 Hz inner / 60 Hz outer.
//
// ── Pipeline ──────────────────────────────────────────────────────
//   reset_grid     — zero out grid mass + momentum.
//   p2g            — particle → grid scatter (atomic adds; computes
//                    elastic PK1 stress per particle, splats forces +
//                    momentum into 3³ = 27 grid nodes per particle).
//   rigid_imprint  — write rigid-body velocities into overlapped grid
//                    cells (penalty coupling, v1).
//   grid_update    — gravity, boundary clamps, divide momentum by mass
//                    to get velocity.
//   g2p            — grid → particle gather; computes APIC affine C,
//                    integrates particle position, accumulates
//                    deformation gradient F via dF/dt = ∇v · F.
//   plasticity     — SVD return mapping (Drucker-Prager) on F.
//
// ── Storage layout ────────────────────────────────────────────────
//   Particle SoA: pos[N], vel[N], C[N] (3×3 affine), F[N] (3×3
//   elastic deformation gradient), volume[N], mass[N], jp[N]
//   (plastic deformation, log of det Fp, scalar — sand is volumetric
//   one-DOF cohesion).
//
//   Grid AoS: per-cell { mass: f32, momentum: vec3<f32> }, packed as
//   4 floats. Stored as a flat buffer of size grid_size³.
//   Atomic operations on grid cells use a u32 reinterpret of the f32
//   storage with a bitcast-compare-exchange loop (WGSL has no
//   atomic<f32>; we emulate with atomic<u32> + bitcasts).
//
// ── References ────────────────────────────────────────────────────
//   * Stomakhin et al. 2013 — Disney "snow" MPM. Provides the
//     APIC transfer, the elastic-plastic split, and the SVD return
//     mapping pattern.
//   * Klár et al. 2016 — Drucker-Prager elasto-plasticity. Gives the
//     "project SVD onto cone" yield criterion used in `plasticity`.
//   * Jiang et al. 2015 — APIC affine particle-in-cell.
//   * Newton's `xtra/newton/.../implicit_mpm/` — reference for buffer
//     layout (we follow the SoA grain shape but transfer is APIC, not
//     PolyPIC).

const WORKGROUP_SIZE: u32 = 64u;

// Drucker-Prager friction-cone constant, computed host-side from the
// friction angle: alpha = sqrt(2/3) * 2 sin(phi) / (3 - sin(phi)).
struct Uniforms {
    // Particle bookkeeping
    particle_count: u32,
    rigid_count: u32,
    grid_size: u32,        // N for an N×N×N grid
    grid_total: u32,       // = N³

    // Grid placement in world space
    grid_origin: vec3<f32>,
    cell_size: f32,

    // Time integration
    dt: f32,
    gravity_y: f32,        // -9.81 typical

    // Material constants (sand)
    youngs: f32,
    poisson: f32,

    // Drucker-Prager yield α (precomputed host-side from phi)
    dp_alpha: f32,
    cohesion: f32,         // c (Pa); 0 for cohesion-less sand

    // Boundary friction
    boundary_friction: f32,
    inv_cell_size: f32,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Particle buffers (SoA, all read_write so kernels share the layout).
@group(1) @binding(0) var<storage, read_write> p_pos:    array<vec4<f32>>; // (xyz, _)
@group(1) @binding(1) var<storage, read_write> p_vel:    array<vec4<f32>>; // (xyz, _)
@group(1) @binding(2) var<storage, read_write> p_C:      array<mat3x3<f32>>;
@group(1) @binding(3) var<storage, read_write> p_F:      array<mat3x3<f32>>;
@group(1) @binding(4) var<storage, read_write> p_volume: array<f32>;
@group(1) @binding(5) var<storage, read_write> p_mass:   array<f32>;
@group(1) @binding(6) var<storage, read_write> p_jp:     array<f32>; // log(det Fp), tracks hardening

// Grid storage. mass = atomic<u32> (bitcast f32). momentum xyz = 3
// atomic<u32> (bitcast f32). Layout: [mass, mx, my, mz] per cell.
@group(2) @binding(0) var<storage, read_write> g_mass:      array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> g_momentum:  array<atomic<u32>>; // 3 entries per cell
@group(2) @binding(2) var<storage, read_write> g_velocity:  array<vec4<f32>>;   // (vx, vy, vz, mass) post grid_update
// Rigid imprint mask — non-zero where rigid bodies have stamped a
// velocity, blocking further updates from p2g/grid_update.
@group(2) @binding(3) var<storage, read_write> g_rigid_mask: array<u32>;

// Rigid bodies (sphere primitives only in v1): center + radius +
// linear velocity. Used by `rigid_imprint`.
@group(3) @binding(0) var<storage, read> rigid_center: array<vec4<f32>>; // (cx, cy, cz, radius)
@group(3) @binding(1) var<storage, read> rigid_vel:    array<vec4<f32>>; // (vx, vy, vz, _)

// ── Helpers ───────────────────────────────────────────────────────

fn cell_idx(ix: i32, iy: i32, iz: i32) -> i32 {
    let n = i32(u.grid_size);
    return ix + n * (iy + n * iz);
}

fn in_bounds(ix: i32, iy: i32, iz: i32) -> bool {
    let n = i32(u.grid_size);
    return ix >= 0 && iy >= 0 && iz >= 0 && ix < n && iy < n && iz < n;
}

// Quadratic B-spline kernel (Stomakhin §4.2). Three weights per axis,
// indices base, base+1, base+2 in cell space.
fn quad_weights(fx: f32) -> vec3<f32> {
    // fx in [0.5, 1.5) — distance from base node
    return vec3<f32>(
        0.5 * (1.5 - fx) * (1.5 - fx),
        0.75 - (fx - 1.0) * (fx - 1.0),
        0.5 * (fx - 0.5) * (fx - 0.5),
    );
}

// f32 atomic add via CAS loop on u32 bitcast. WGSL has no
// atomic<f32>; this is the standard workaround used by Houdini's
// vellum and Taichi's MPM. Naga rejects pointer-to-storage as a
// function argument so we have one helper per buffer.
fn atomic_add_mass(idx: u32, val: f32) {
    var old_bits = atomicLoad(&g_mass[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let ex = atomicCompareExchangeWeak(&g_mass[idx], old_bits, new_bits);
        if (ex.exchanged) { break; }
        old_bits = ex.old_value;
    }
}

fn atomic_add_mom(idx: u32, val: f32) {
    var old_bits = atomicLoad(&g_momentum[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let ex = atomicCompareExchangeWeak(&g_momentum[idx], old_bits, new_bits);
        if (ex.exchanged) { break; }
        old_bits = ex.old_value;
    }
}

// Symmetric 3×3 SVD (textbook Jacobi, mirrors `dense3.rs::svd3`).
// Returns U, sigma, V s.t. M = U * diag(sigma) * Vᵀ.
struct Svd3 { u: mat3x3<f32>, s: vec3<f32>, v: mat3x3<f32> };

fn givens(a: f32, b: f32) -> vec2<f32> {
    if (b == 0.0) { return vec2<f32>(1.0, 0.0); }
    if (abs(b) > abs(a)) {
        let t = -a / b;
        let s = 1.0 / sqrt(1.0 + t * t);
        return vec2<f32>(s * t, s);
    }
    let t = -b / a;
    let c = 1.0 / sqrt(1.0 + t * t);
    return vec2<f32>(c, c * t);
}

fn svd3_jacobi(m_in: mat3x3<f32>) -> Svd3 {
    // Polar-style: form A = MᵀM (symmetric), Jacobi-diagonalize ⇒ V.
    // Then sigmas = ‖M·v_i‖, U columns = (M·v_i)/sigma_i.
    var v = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    var a = transpose(m_in) * m_in;
    let max_sweeps: u32 = 16u;
    for (var sweep: u32 = 0u; sweep < max_sweeps; sweep = sweep + 1u) {
        var max_off: f32 = 0.0;
        for (var p: u32 = 0u; p < 3u; p = p + 1u) {
            for (var q: u32 = p + 1u; q < 3u; q = q + 1u) {
                let apq = a[q][p]; // mat3x3 is column-major: a[col][row]
                if (abs(apq) > max_off) { max_off = abs(apq); }
                if (abs(apq) < 1.0e-7) { continue; }
                let app = a[p][p];
                let aqq = a[q][q];
                let theta = (aqq - app) / (2.0 * apq);
                var t: f32;
                if (theta >= 0.0) {
                    t = 1.0 / (theta + sqrt(1.0 + theta * theta));
                } else {
                    t = 1.0 / (theta - sqrt(1.0 + theta * theta));
                }
                let c = 1.0 / sqrt(1.0 + t * t);
                let s = t * c;

                // Update A: rotate rows/cols p, q.
                let app_new = app - t * apq;
                let aqq_new = aqq + t * apq;
                a[p][p] = app_new;
                a[q][q] = aqq_new;
                a[q][p] = 0.0;
                a[p][q] = 0.0;
                for (var k: u32 = 0u; k < 3u; k = k + 1u) {
                    if (k != p && k != q) {
                        let akp = a[p][k];
                        let akq = a[q][k];
                        a[p][k] = c * akp - s * akq;
                        a[q][k] = s * akp + c * akq;
                        a[k][p] = a[p][k];
                        a[k][q] = a[q][k];
                    }
                }
                // Update V columns p, q.
                for (var k: u32 = 0u; k < 3u; k = k + 1u) {
                    let vkp = v[p][k];
                    let vkq = v[q][k];
                    v[p][k] = c * vkp - s * vkq;
                    v[q][k] = s * vkp + c * vkq;
                }
            }
        }
        if (max_off < 1.0e-7) { break; }
    }
    // Singular values from M·v_i.
    var sigma: vec3<f32>;
    var u_out = mat3x3<f32>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        let vi = vec3<f32>(v[i][0], v[i][1], v[i][2]);
        let mvi = m_in * vi;
        let s = length(mvi);
        sigma[i] = s;
        if (s > 1.0e-12) {
            let col = mvi / s;
            u_out[i][0] = col.x;
            u_out[i][1] = col.y;
            u_out[i][2] = col.z;
        } else {
            u_out[i][0] = select(0.0, 1.0, i == 0u);
            u_out[i][1] = select(0.0, 1.0, i == 1u);
            u_out[i][2] = select(0.0, 1.0, i == 2u);
        }
    }
    // Make det(U) = +1 (flip the smallest-singular-value column if needed).
    let detu = determinant(u_out);
    if (detu < 0.0) {
        // Flip column with smallest sigma.
        var min_i: u32 = 0u;
        if (sigma[1] < sigma[min_i]) { min_i = 1u; }
        if (sigma[2] < sigma[min_i]) { min_i = 2u; }
        u_out[min_i][0] = -u_out[min_i][0];
        u_out[min_i][1] = -u_out[min_i][1];
        u_out[min_i][2] = -u_out[min_i][2];
        sigma[min_i] = -sigma[min_i];
    }
    let detv = determinant(v);
    if (detv < 0.0) {
        var min_i: u32 = 0u;
        if (abs(sigma[1]) < abs(sigma[min_i])) { min_i = 1u; }
        if (abs(sigma[2]) < abs(sigma[min_i])) { min_i = 2u; }
        v[min_i][0] = -v[min_i][0];
        v[min_i][1] = -v[min_i][1];
        v[min_i][2] = -v[min_i][2];
        sigma[min_i] = -sigma[min_i];
    }
    return Svd3(u_out, sigma, v);
}

// ── Kernel: reset_grid ────────────────────────────────────────────
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn reset_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.grid_total) { return; }
    atomicStore(&g_mass[idx], 0u);
    let base = idx * 3u;
    atomicStore(&g_momentum[base + 0u], 0u);
    atomicStore(&g_momentum[base + 1u], 0u);
    atomicStore(&g_momentum[base + 2u], 0u);
    g_velocity[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    g_rigid_mask[idx] = 0u;
}

// ── Kernel: p2g (particle → grid scatter) ─────────────────────────
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn p2g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= u.particle_count) { return; }

    let pos = p_pos[pid].xyz;
    let vel = p_vel[pid].xyz;
    let mass = p_mass[pid];
    let vol = p_volume[pid];
    let F = p_F[pid];
    let C = p_C[pid];

    // Cell-space coordinates relative to grid_origin. We use the
    // "node centred on integer grid index" convention: a cell at
    // index (ix, iy, iz) is centred at grid_origin + (ix + 0.5) * dx
    // for the quadratic kernel — but to keep stencils symmetric we
    // shift by 0.5 and use the standard `base = floor(x - 0.5)` form
    // from Stomakhin §4.2.
    let xp = (pos - u.grid_origin) * u.inv_cell_size;
    let base_xyz = vec3<f32>(floor(xp.x - 0.5), floor(xp.y - 0.5), floor(xp.z - 0.5));
    let base = vec3<i32>(i32(base_xyz.x), i32(base_xyz.y), i32(base_xyz.z));
    let fx = xp - base_xyz; // fx in [0.5, 1.5)

    let wx = quad_weights(fx.x);
    let wy = quad_weights(fx.y);
    let wz = quad_weights(fx.z);

    // Elastic Cauchy stress via a Neo-Hookean potential (deviatoric)
    // + Drucker-Prager-style isotropic part (handled in `plasticity`).
    // PK1: P = 2μ (F − R) + λ J (J−1) F^{-T} where R = polar(F).
    // We use SVD: R = U Vᵀ, J = det(F) = sigma_x*sigma_y*sigma_z.
    //
    // Singular values clamped to [0.1, 10] to defuse the hourglass /
    // inverted-element pathology: a particle that gets pancaked
    // (sigma → 0) would otherwise see an enormous F−R term and
    // explode. Clamping caps the elastic stress at a finite value,
    // letting plasticity catch up.
    let raw_svd = svd3_jacobi(F);
    let s_clamped = vec3<f32>(
        clamp(raw_svd.s.x, 0.1, 10.0),
        clamp(raw_svd.s.y, 0.1, 10.0),
        clamp(raw_svd.s.z, 0.1, 10.0),
    );
    let svd = Svd3(raw_svd.u, s_clamped, raw_svd.v);
    let R = svd.u * transpose(svd.v);
    let J = svd.s.x * svd.s.y * svd.s.z;

    let mu = u.youngs / (2.0 * (1.0 + u.poisson));
    let lam = u.youngs * u.poisson / ((1.0 + u.poisson) * (1.0 - 2.0 * u.poisson));
    // Hardening — sand stiffens with accumulated plastic strain
    // (Klár 2016). p_jp tracks Σ‖dγ‖ ≥ 0; multiplier = exp(h0 * jp)
    // with h0 small (1.0) so the modulus doesn't blow up after a few
    // substeps. Disney's snow paper uses 10 here but they track
    // log(det Fp) which is bounded near 0; we track plastic flow
    // increment cumulative so we're more conservative.
    let h = exp(min(p_jp[pid], 5.0));
    let mu_h = mu * h;
    let lam_h = lam * h;

    // Reconstruct F from clamped SVD so the elastic-stress arithmetic
    // doesn't see degenerate (near-zero or near-infinite) sigmas.
    let F_clamped = svd.u * mat3x3<f32>(
        vec3<f32>(svd.s.x, 0.0, 0.0),
        vec3<f32>(0.0, svd.s.y, 0.0),
        vec3<f32>(0.0, 0.0, svd.s.z),
    ) * transpose(svd.v);
    let f_minus_r = F_clamped - R;
    // J F^{-T} via SVD cofactor (no explicit inverse).
    let cof = svd.u * mat3x3<f32>(
        vec3<f32>(svd.s.y * svd.s.z, 0.0, 0.0),
        vec3<f32>(0.0, svd.s.x * svd.s.z, 0.0),
        vec3<f32>(0.0, 0.0, svd.s.x * svd.s.y),
    ) * transpose(svd.v);
    let pk1 = 2.0 * mu_h * f_minus_r + lam_h * (J - 1.0) * cof;

    // APIC stress force per node = -4 dt vol / dx² · P · F^T.
    let stress = -u.dt * vol * pk1 * transpose(F_clamped);

    // APIC: per-node momentum contribution = mass * (vel + C * (xn - xp))
    let dx = u.cell_size;
    let inv_dx2 = u.inv_cell_size * u.inv_cell_size;

    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        for (var j: u32 = 0u; j < 3u; j = j + 1u) {
            for (var k: u32 = 0u; k < 3u; k = k + 1u) {
                let w = wx[i] * wy[j] * wz[k];
                let ix = base.x + i32(i);
                let iy = base.y + i32(j);
                let iz = base.z + i32(k);
                if (!in_bounds(ix, iy, iz)) { continue; }
                let cidx = u32(cell_idx(ix, iy, iz));
                // node position in world coords (cell centre)
                let node_pos = u.grid_origin + vec3<f32>(
                    f32(ix) + 0.5,
                    f32(iy) + 0.5,
                    f32(iz) + 0.5,
                ) * dx;
                let xn_xp = node_pos - pos;

                // mass deposit
                atomic_add_mass(cidx, w * mass);
                // momentum deposit: w * mass * (vel + C * (xn - xp))
                //                  + 4 * w * stress * (xn - xp) / dx²
                let aff = C * xn_xp;
                let stress_force = stress * xn_xp * (4.0 * inv_dx2);
                let mom = w * mass * (vel + aff) + w * stress_force;
                let mb = cidx * 3u;
                atomic_add_mom(mb + 0u, mom.x);
                atomic_add_mom(mb + 1u, mom.y);
                atomic_add_mom(mb + 2u, mom.z);
            }
        }
    }
}

// ── Kernel: rigid_imprint ─────────────────────────────────────────
// For each grid cell, check overlap with each rigid sphere (small N
// in v1; brute force is fine). If overlapped, write the rigid's
// velocity directly into momentum (mass = 1, mom = v) and tag the
// cell so grid_update doesn't apply gravity / friction to it.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn rigid_imprint(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cidx = gid.x;
    if (cidx >= u.grid_total) { return; }
    if (u.rigid_count == 0u) { return; }
    let n = u.grid_size;
    let ix = cidx % n;
    let iy = (cidx / n) % n;
    let iz = cidx / (n * n);
    let node_pos = u.grid_origin + vec3<f32>(
        f32(ix) + 0.5,
        f32(iy) + 0.5,
        f32(iz) + 0.5,
    ) * u.cell_size;
    for (var r: u32 = 0u; r < u.rigid_count; r = r + 1u) {
        let cr = rigid_center[r];
        let v  = rigid_vel[r].xyz;
        let center = cr.xyz;
        let radius = cr.w;
        let d = distance(node_pos, center);
        if (d < radius) {
            // Stamp velocity. mass = some constant so g2p sees a
            // "heavy" cell that pushes particles away.
            let stamp_mass: f32 = 1.0e6;
            let mom = stamp_mass * v;
            atomicStore(&g_mass[cidx], bitcast<u32>(stamp_mass));
            let mb = cidx * 3u;
            atomicStore(&g_momentum[mb + 0u], bitcast<u32>(mom.x));
            atomicStore(&g_momentum[mb + 1u], bitcast<u32>(mom.y));
            atomicStore(&g_momentum[mb + 2u], bitcast<u32>(mom.z));
            g_rigid_mask[cidx] = 1u;
            return;
        }
    }
}

// ── Kernel: grid_update ───────────────────────────────────────────
// Convert momentum → velocity (divide by mass), apply gravity, clamp
// to grid bounds with friction. Output goes into g_velocity for g2p.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn grid_update(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cidx = gid.x;
    if (cidx >= u.grid_total) { return; }
    let m_bits = atomicLoad(&g_mass[cidx]);
    let m = bitcast<f32>(m_bits);
    // Mass-tolerance gate: cells with negligible mass produce a
    // numerically explosive velocity = momentum/mass. Skip them.
    if (m <= 1.0e-9 || !(m == m)) {
        g_velocity[cidx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }
    let mb = cidx * 3u;
    let mom = vec3<f32>(
        bitcast<f32>(atomicLoad(&g_momentum[mb + 0u])),
        bitcast<f32>(atomicLoad(&g_momentum[mb + 1u])),
        bitcast<f32>(atomicLoad(&g_momentum[mb + 2u])),
    );
    var vel = mom / m;

    // Skip integration on rigid-imprinted cells.
    if (g_rigid_mask[cidx] != 0u) {
        g_velocity[cidx] = vec4<f32>(vel.x, vel.y, vel.z, m);
        return;
    }

    // Gravity (semi-implicit).
    vel.y = vel.y + u.dt * u.gravity_y;

    // Boundary friction / sticking. Outer 2 layers clamp normal vel
    // and apply Coulomb friction on the tangent.
    let n = i32(u.grid_size);
    let ix = i32(cidx) % n;
    let iy = (i32(cidx) / n) % n;
    let iz = i32(cidx) / (n * n);
    let pad: i32 = 2;
    if (ix < pad)        { vel.x = max(vel.x, 0.0); vel.y = vel.y * u.boundary_friction; vel.z = vel.z * u.boundary_friction; }
    if (ix >= n - pad)   { vel.x = min(vel.x, 0.0); vel.y = vel.y * u.boundary_friction; vel.z = vel.z * u.boundary_friction; }
    if (iy < pad)        { vel.y = max(vel.y, 0.0); vel.x = vel.x * u.boundary_friction; vel.z = vel.z * u.boundary_friction; }
    if (iy >= n - pad)   { vel.y = min(vel.y, 0.0); vel.x = vel.x * u.boundary_friction; vel.z = vel.z * u.boundary_friction; }
    if (iz < pad)        { vel.z = max(vel.z, 0.0); vel.x = vel.x * u.boundary_friction; vel.y = vel.y * u.boundary_friction; }
    if (iz >= n - pad)   { vel.z = min(vel.z, 0.0); vel.x = vel.x * u.boundary_friction; vel.y = vel.y * u.boundary_friction; }

    g_velocity[cidx] = vec4<f32>(vel.x, vel.y, vel.z, m);
}

// ── Kernel: g2p (grid → particle gather + APIC + F update) ────────
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn g2p(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= u.particle_count) { return; }
    let pos = p_pos[pid].xyz;
    let xp = (pos - u.grid_origin) * u.inv_cell_size;
    let base_xyz = vec3<f32>(floor(xp.x - 0.5), floor(xp.y - 0.5), floor(xp.z - 0.5));
    let base = vec3<i32>(i32(base_xyz.x), i32(base_xyz.y), i32(base_xyz.z));
    let fx = xp - base_xyz;
    let wx = quad_weights(fx.x);
    let wy = quad_weights(fx.y);
    let wz = quad_weights(fx.z);

    var v_new = vec3<f32>(0.0, 0.0, 0.0);
    var C_new = mat3x3<f32>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );
    let inv_dx = u.inv_cell_size;
    let dx = u.cell_size;
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        for (var j: u32 = 0u; j < 3u; j = j + 1u) {
            for (var k: u32 = 0u; k < 3u; k = k + 1u) {
                let w = wx[i] * wy[j] * wz[k];
                let ix = base.x + i32(i);
                let iy = base.y + i32(j);
                let iz = base.z + i32(k);
                if (!in_bounds(ix, iy, iz)) { continue; }
                let cidx = u32(cell_idx(ix, iy, iz));
                let gv = g_velocity[cidx].xyz;
                v_new = v_new + w * gv;
                let node_pos = u.grid_origin + vec3<f32>(
                    f32(ix) + 0.5,
                    f32(iy) + 0.5,
                    f32(iz) + 0.5,
                ) * dx;
                let xn_xp = node_pos - pos;
                // C += 4 * w * v_node * (xn - xp)ᵀ / dx²
                let scale = 4.0 * w * inv_dx * inv_dx;
                C_new[0][0] = C_new[0][0] + scale * gv.x * xn_xp.x;
                C_new[0][1] = C_new[0][1] + scale * gv.y * xn_xp.x;
                C_new[0][2] = C_new[0][2] + scale * gv.z * xn_xp.x;
                C_new[1][0] = C_new[1][0] + scale * gv.x * xn_xp.y;
                C_new[1][1] = C_new[1][1] + scale * gv.y * xn_xp.y;
                C_new[1][2] = C_new[1][2] + scale * gv.z * xn_xp.y;
                C_new[2][0] = C_new[2][0] + scale * gv.x * xn_xp.z;
                C_new[2][1] = C_new[2][1] + scale * gv.y * xn_xp.z;
                C_new[2][2] = C_new[2][2] + scale * gv.z * xn_xp.z;
            }
        }
    }

    // Integrate position.
    var new_pos = pos + u.dt * v_new;
    // Soft position clamp. If the particle is escaping the active
    // grid region (within 2 cells of the boundary on the floor and
    // walls, far side of grid), bring it back and zero the offending
    // velocity component so it doesn't keep ramming. Position-clamp
    // is a safety net only — the velocity-clamp in `grid_update`
    // does the real work for stable contact.
    let pad: f32 = 2.0;
    let lo = u.grid_origin + vec3<f32>(pad * u.cell_size);
    let hi = u.grid_origin + vec3<f32>(f32(u.grid_size) - pad - 1.0) * u.cell_size;
    let clamped_pos = clamp(new_pos, lo, hi);
    if (new_pos.x < lo.x || new_pos.x > hi.x) { v_new.x = 0.0; }
    if (new_pos.y < lo.y || new_pos.y > hi.y) { v_new.y = 0.0; }
    if (new_pos.z < lo.z || new_pos.z > hi.z) { v_new.z = 0.0; }
    new_pos = clamped_pos;

    // Update F via dF/dt = (∇v) F. For APIC, ∇v ≈ C (at this scale).
    let F_old = p_F[pid];
    let identity = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    let F_new = (identity + u.dt * C_new) * F_old;

    // NaN safety: if the gather produced non-finite output (typically
    // because a single rogue grid cell spat out Inf), reset to zero
    // velocity / identity affine matrix. Lossy but bounded.
    if (!(v_new.x == v_new.x) || !(v_new.y == v_new.y) || !(v_new.z == v_new.z)) {
        v_new = vec3<f32>(0.0, 0.0, 0.0);
        new_pos = pos;
    }
    p_pos[pid] = vec4<f32>(new_pos, 0.0);
    p_vel[pid] = vec4<f32>(v_new, 0.0);
    let c00 = C_new[0][0];
    if (c00 == c00) {
        p_C[pid] = C_new;
    } else {
        p_C[pid] = mat3x3<f32>(
            vec3<f32>(0.0, 0.0, 0.0),
            vec3<f32>(0.0, 0.0, 0.0),
            vec3<f32>(0.0, 0.0, 0.0),
        );
    }
    let f00 = F_new[0][0];
    if (f00 == f00) {
        p_F[pid] = F_new;
    } else {
        p_F[pid] = mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0),
        );
    }
}

// ── Kernel: plasticity (Drucker-Prager return mapping) ────────────
// Klár et al. 2016. SVD F = U Σ Vᵀ. Project Σ → Σ_e (elastic part)
// based on the friction cone, log the plastic deviation into jp.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn plasticity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= u.particle_count) { return; }
    let F = p_F[pid];
    let svd = svd3_jacobi(F);
    let s = svd.s;

    let mu = u.youngs / (2.0 * (1.0 + u.poisson));
    let lam = u.youngs * u.poisson / ((1.0 + u.poisson) * (1.0 - 2.0 * u.poisson));
    let d = 3.0;

    // log strains (Hencky)
    let eps_log = vec3<f32>(log(max(s.x, 1.0e-6)), log(max(s.y, 1.0e-6)), log(max(s.z, 1.0e-6)));
    let trace_eps = eps_log.x + eps_log.y + eps_log.z;
    let mean_eps = trace_eps / d;
    let dev_eps = eps_log - vec3<f32>(mean_eps, mean_eps, mean_eps);
    let dev_norm = length(dev_eps);

    // Klár 2016 §3.4 — Drucker-Prager return mapping.
    //
    // Yield function (in log-space deviator and mean):
    //   f = ‖dev_eps‖ + α · (d·λ + 2μ)/(2μ) · trace_eps
    //
    // Three regimes:
    //  (a) Tensile (trace_eps > 0)  ⇒ project F → I (no tension in
    //      granular media). dq = ‖eps_log‖.
    //  (b) f ≤ 0                    ⇒ inside / on cone, elastic.
    //  (c) f > 0                    ⇒ project onto cone surface
    //      ‖dev_eps‖ = -α (d·λ + 2μ)/(2μ) · trace_eps. Keep mean
    //      strain; scale deviator down to the cone radius.
    var new_s = s;
    var dq: f32 = 0.0;
    let kappa = (d * lam + 2.0 * mu) / (2.0 * mu);
    if (trace_eps > 0.0) {
        new_s = vec3<f32>(1.0, 1.0, 1.0);
        dq = length(eps_log);
    } else if (dev_norm == 0.0) {
        new_s = s;
    } else {
        let f_yield = dev_norm + kappa * trace_eps * u.dp_alpha;
        if (f_yield <= 0.0) {
            // Inside the cone — fully elastic.
            new_s = s;
        } else {
            // Cone surface radius for this trace_eps:
            //   R(trace) = -α κ · trace_eps    (positive since trace_eps < 0)
            let r_cone = max(0.0, -u.dp_alpha * kappa * trace_eps);
            let new_dev = dev_eps * (r_cone / dev_norm);
            let new_eps = new_dev + vec3<f32>(mean_eps, mean_eps, mean_eps);
            new_s = vec3<f32>(exp(new_eps.x), exp(new_eps.y), exp(new_eps.z));
            dq = f_yield;
        }
    }

    // Reconstruct F_e (clamped to a sane range; an inverted /
    // pancaked particle gets reset to identity so the next p2g
    // doesn't see Inf stress).
    let s_safe = vec3<f32>(
        clamp(new_s.x, 0.05, 20.0),
        clamp(new_s.y, 0.05, 20.0),
        clamp(new_s.z, 0.05, 20.0),
    );
    var F_new = svd.u * mat3x3<f32>(
        vec3<f32>(s_safe.x, 0.0, 0.0),
        vec3<f32>(0.0, s_safe.y, 0.0),
        vec3<f32>(0.0, 0.0, s_safe.z),
    ) * transpose(svd.v);
    // NaN sentinel: any non-finite element resets F to identity.
    let f00 = F_new[0][0];
    if (!(f00 == f00) || !(F_new[1][1] == F_new[1][1]) || !(F_new[2][2] == F_new[2][2])) {
        F_new = mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0),
        );
    }
    p_F[pid] = F_new;
    p_jp[pid] = clamp(p_jp[pid] + dq, 0.0, 5.0);
}
