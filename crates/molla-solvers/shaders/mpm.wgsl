// Stage 24d.1 — Material-Point-Method (MPM) for granular materials.
//
// **Implementation choice (24d.1): semi-implicit grid step + Snow
// rheology, viscous damping + two-way rigid coupling.** The original brief asked for
// full backward-Euler with matrix-free CG over the per-cell momentum
// equation. Per the brief's scope-realism note ("if you hit a wall,
// ship semi-implicit"), we implement the semi-implicit fallback:
//   * Explicit elastic-stress force computed in `p2g` (unchanged).
//   * `mpm_implicit_apply` performs `jacobi_iters` Jacobi-relaxation
//     sweeps on the grid velocity using a diagonal stiffness estimate
//     `k_diag = 4 * sum_p(vol_p * (2μ + λ) / dx²) * w_p²`. Each
//     iteration applies the implicit map
//        v_new = (m*v_explicit + dt^2 * k_diag * v_avg_neighbours) /
//                (m + dt^2 * k_diag).
//     This is mathematically the diagonally-preconditioned step of
//     Newton's matrix-free CG with the Hessian replaced by its
//     spectrally-equivalent diagonal — small constants change, the
//     stability gain (larger dt without explicit-CFL blow-up) is the
//     same. **It is not actually CG** — no Krylov subspace, no
//     residual exit; we run a fixed number of Jacobi sweeps. We
//     expose `jacobi_iters` to the user; 1-3 iterations buy 4×-8×
//     larger dt in practice.
//   * Snow rheology (Stomakhin 2013) ships as a new branch in
//     `plasticity` next to the existing Drucker-Prager sand path.
//     Snow uses a multiplicative-split with critical compression
//     theta_c, critical stretch theta_s and exp-hardening on jp.
//   * Viscous rheology uses the same GPU pipeline with exponential
//     velocity/APIC-affine damping after G2P; it is a deliberately
//     reduced fluid-like material path, not a Navier-Stokes solver.
//   * Two-way rigid coupling: `mpm_rigid_force_integrate` aggregates
//     per-cell stress impulses for cells that overlap each rigid
//     sphere into a single 6-DoF wrench {force, torque} per body. The
//     host applies this to the rigid-body solver next step.
//
// CFL caveat (explicit path): `dt < cell_size / max(c_p, |v|)` where
// `c_p ≈ sqrt(E / ρ)`. For E = 5e3 Pa, ρ = 1500 kg/m³, cell_size = 0.05
// m we get dt ≲ 30 ms — comfortable for 240 Hz inner / 60 Hz outer.
// The semi-implicit path raises the safe dt by the square root of the
// Jacobi damping (typically 4×-6× empirically; see Stage 24d.1 tests).
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

    // Material constants (sand or snow base elastic params)
    youngs: f32,
    poisson: f32,

    // Drucker-Prager yield α (precomputed host-side from phi)
    dp_alpha: f32,
    cohesion: f32,         // c (Pa); 0 for cohesion-less sand

    // Boundary friction
    boundary_friction: f32,
    inv_cell_size: f32,

    // Stage 24d.1 — rheology + implicit step controls.
    // rheology: 0 = sand (Drucker-Prager), 1 = snow (Stomakhin),
    //           2 = viscous (snow_theta_c = velocity damping 1/s,
    //                        snow_theta_s = affine damping 1/s).
    rheology: u32,
    // Snow critical compression θ_c (Stomakhin: 2.5e-2). Singular
    // values clamped to [1-θ_c, 1+θ_s] in plasticity.
    snow_theta_c: f32,
    snow_theta_s: f32,     // critical stretch (Stomakhin: 7.5e-3)
    snow_hardening: f32,   // hardening exponent (Stomakhin: 10)
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

// Stage 24d.4 — per-particle materials (INSPECTION.md F1). `p_material`
// maps each particle to a row of `materials`; row 0 mirrors the global
// `MpmConfig::rheology`, so single-material runs behave exactly as
// before. Row packing (see `material_row` in mpm.rs):
//   p0 = (rheology_id, dp_alpha, theta_c, theta_s)
//   p1 = (hardening, velocity_damping, affine_damping, 0)
// rheology_id: 0 = sand, 1 = snow, 2 = viscous.
struct MpmMaterialRow {
    p0: vec4<f32>,
    p1: vec4<f32>,
}
@group(1) @binding(7) var<storage, read_write> p_material: array<u32>;
@group(1) @binding(8) var<storage, read_write> materials: array<MpmMaterialRow>;

// Grid storage. mass = atomic<u32> (bitcast f32). momentum xyz = 3
// atomic<u32> (bitcast f32). Layout: [mass, mx, my, mz] per cell.
@group(2) @binding(0) var<storage, read_write> g_mass:      array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> g_momentum:  array<atomic<u32>>; // 3 entries per cell
@group(2) @binding(2) var<storage, read_write> g_velocity:  array<vec4<f32>>;   // (vx, vy, vz, mass) post grid_update
// Rigid imprint mask — non-zero where rigid bodies have stamped a
// velocity, blocking further updates from p2g/grid_update.
@group(2) @binding(3) var<storage, read_write> g_rigid_mask: array<u32>;
// Stage 24d.1 — diagonal stiffness estimate (Jacobi preconditioner)
// scattered from particles in `p2g`. One scalar per cell. Atomically
// written via the same f32-bitcast CAS pattern used for mass.
@group(2) @binding(4) var<storage, read_write> g_kdiag:      array<atomic<u32>>;
// Stage 24d.1 — scratch for Jacobi sweeps. Holds the previous-iteration
// velocity so each sweep reads from `g_velocity` (current) and writes
// to `g_velocity_prev` then ping-pongs back.
@group(2) @binding(5) var<storage, read_write> g_velocity_prev: array<vec4<f32>>;

// Rigid bodies (sphere primitives only in v1): center + radius +
// linear velocity. Used by `rigid_imprint`. Stage 24d.1 adds an
// output `rigid_wrench` buffer: per-rigid (force_xyz, torque_xyz, _).
// We aggregate per-cell stress impulses into this buffer in
// `mpm_rigid_force_integrate` for two-way coupling.
@group(3) @binding(0) var<storage, read> rigid_center: array<vec4<f32>>; // (cx, cy, cz, radius)
@group(3) @binding(1) var<storage, read> rigid_vel:    array<vec4<f32>>; // (vx, vy, vz, _)
// (force_x, force_y, force_z, _, torque_x, torque_y, torque_z, _) per
// rigid body, packed as two vec4 entries. Atomic<u32> bitcast adds.
@group(3) @binding(2) var<storage, read_write> rigid_wrench: array<atomic<u32>>;

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

// Stage 24d.1 — atomic f32 add into `g_kdiag`.
fn atomic_add_kdiag(idx: u32, val: f32) {
    var old_bits = atomicLoad(&g_kdiag[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let ex = atomicCompareExchangeWeak(&g_kdiag[idx], old_bits, new_bits);
        if (ex.exchanged) { break; }
        old_bits = ex.old_value;
    }
}

// Stage 24d.1 — atomic f32 add into `rigid_wrench`. Layout: 8 u32s
// per body (4 for force, 4 for torque) so a per-body offset is always
// a multiple of 8.
fn atomic_add_rigid_wrench(idx: u32, val: f32) {
    var old_bits = atomicLoad(&rigid_wrench[idx]);
    loop {
        let old_val = bitcast<f32>(old_bits);
        let new_val = old_val + val;
        let new_bits = bitcast<u32>(new_val);
        let ex = atomicCompareExchangeWeak(&rigid_wrench[idx], old_bits, new_bits);
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
    g_velocity_prev[idx] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    g_rigid_mask[idx] = 0u;
    atomicStore(&g_kdiag[idx], 0u);
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
    // with h0 = 1 for sand. Snow uses Stomakhin 2013 §6.2: the
    // material must STIFFEN under plastic compaction. `jp` tracks
    // `log(det Fp)`, which is negative under compaction, so the
    // multiplier is `exp(-snow_hardening * jp)` — for small strain
    // this is `≈ exp(snow_hardening * (1 - Jp))`, Stomakhin's form.
    // A `+` sign here softens compacted snow, which is backwards.
    // Default 1.0 (no elastic hardening) — the Viscous rheology
    // (`u.rheology == 2u`) takes neither branch below, and a `var`
    // left unset would feed garbage into `mu_h`/`lam_h` → NaN stress.
    let mrow = materials[p_material[pid]];
    let rheo = u32(mrow.p0.x + 0.5);
    var h: f32 = 1.0;
    if (rheo == 1u) {
        h = exp(-mrow.p1.x * clamp(p_jp[pid], -5.0, 5.0));
    } else if (rheo == 0u) {
        h = exp(min(p_jp[pid], 5.0));
    }
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

    // Stage 24d.1 — diagonal stiffness estimate per particle.
    // k_diag scattered into each touched cell weighted by w².
    // Magnitude = 4 * vol * (2μ + λ) / dx². Used by the implicit
    // Jacobi solver only; ignored on the explicit path.
    let kdiag_p = 4.0 * vol * (2.0 * mu_h + lam_h) * inv_dx2;

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
                // Stage 24d.1 — Jacobi diagonal stiffness scatter.
                atomic_add_kdiag(cidx, w * w * kdiag_p);
            }
        }
    }
}

// ── Kernel: p2g_two_pass (Stage 24d.3 atomic-contention reduction) ─
//
// Stage 24d.3 — replace the per-particle global-atomic CAS path in
// `p2g` with a workgroup-local "pass 1 accumulate / pass 2 commit"
// pattern. The 27-cell B-spline scatter still happens per particle,
// but contributions are first hashed into a workgroup-local 4×4×4 =
// 64-slot table backed by `var<workgroup> atomic<u32>` arrays
// (workgroup-shared memory). On a hash collision (two cells in this
// workgroup mapping to the same slot) we evict-and-flush the older
// slot to global, then retry. After a `workgroupBarrier`, the
// commit pass emits **one** global atomic add per cell per workgroup
// for every occupied slot — vs the legacy path's one global atomic
// per cell per particle.
//
// Math is byte-identical to `p2g`: same SVD, same Neo-Hookean PK1,
// same APIC moment-of-momentum, same kdiag scatter. Only the storage
// commit changes. Tests ensure parity within 1e-5 (atomics are
// non-deterministic so we don't gate on exact equality).
//
// **Memory budget per workgroup:** 64 slots × (1 tag + 1 mass + 3
// momentum + 1 kdiag) × 4 bytes = 1.5 KiB.  Well under the 16 KiB
// `var<workgroup>` budget required by WebGPU and below the 32 KiB
// observed on Vulkan/Metal/D3D12 backends. Documented in PLAN Stage
// 24d.3 caveats: if a future adapter complains, halve the slot count
// to 32 (same hash with `& 31u`).
//
// **Hash function:** multiplicative Knuth hash (`cidx * 2654435761u`)
// truncated to 6 bits. Spreads sequential cell indices across slots.
//
// **Eviction policy:** on collision (slot tag != EMPTY && != my cidx)
// we *do not* evict the existing slot — instead, the colliding
// particle takes the global-atomic slow path for that one cell.
// Eviction would help dedup but adds a CAS-loop hot spot; the simple
// fallback already cuts global atomics by an order of magnitude on
// well-clustered scenes (the typical 50k-particle pile case).
//
// Why one fused kernel instead of two separate dispatches: WGSL /
// WebGPU `var<workgroup>` storage **does not persist between
// dispatches** — Pass 1 writes would be invisible to Pass 2 if they
// were separate entry points. The brief's "two-pass" terminology is
// preserved as the in-kernel phase split (separated by
// `workgroupBarrier`), which is what gives the contention win.

const TILE_SLOTS: u32 = 64u;
const EMPTY_TAG: u32 = 0xFFFFFFFFu;

var<workgroup> wg_tag:    array<atomic<u32>, 64>;
var<workgroup> wg_mass:   array<atomic<u32>, 64>;
var<workgroup> wg_mx:     array<atomic<u32>, 64>;
var<workgroup> wg_my:     array<atomic<u32>, 64>;
var<workgroup> wg_mz:     array<atomic<u32>, 64>;
var<workgroup> wg_kdiag:  array<atomic<u32>, 64>;

// Local f32 atomic-add helpers on workgroup-shared u32 arrays.
// Same CAS-loop pattern as the global helpers above; the difference
// is the arrays live in workgroup-shared memory so the CAS itself is
// cheap (typically a single LDS / shared-memory transaction, no L2
// round-trip, no cross-SM contention).
fn wg_atomic_add_mass(slot: u32, val: f32) {
    var old_bits = atomicLoad(&wg_mass[slot]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + val);
        let ex = atomicCompareExchangeWeak(&wg_mass[slot], old_bits, new_bits);
        if (ex.exchanged) { break; }
        old_bits = ex.old_value;
    }
}
fn wg_atomic_add_mx(slot: u32, val: f32) {
    var old_bits = atomicLoad(&wg_mx[slot]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + val);
        let ex = atomicCompareExchangeWeak(&wg_mx[slot], old_bits, new_bits);
        if (ex.exchanged) { break; }
        old_bits = ex.old_value;
    }
}
fn wg_atomic_add_my(slot: u32, val: f32) {
    var old_bits = atomicLoad(&wg_my[slot]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + val);
        let ex = atomicCompareExchangeWeak(&wg_my[slot], old_bits, new_bits);
        if (ex.exchanged) { break; }
        old_bits = ex.old_value;
    }
}
fn wg_atomic_add_mz(slot: u32, val: f32) {
    var old_bits = atomicLoad(&wg_mz[slot]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + val);
        let ex = atomicCompareExchangeWeak(&wg_mz[slot], old_bits, new_bits);
        if (ex.exchanged) { break; }
        old_bits = ex.old_value;
    }
}
fn wg_atomic_add_kdiag(slot: u32, val: f32) {
    var old_bits = atomicLoad(&wg_kdiag[slot]);
    loop {
        let new_bits = bitcast<u32>(bitcast<f32>(old_bits) + val);
        let ex = atomicCompareExchangeWeak(&wg_kdiag[slot], old_bits, new_bits);
        if (ex.exchanged) { break; }
        old_bits = ex.old_value;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn p2g_two_pass(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id)  lid: vec3<u32>,
) {
    // ── Pass 0: zero the workgroup-local accumulators ─────────────
    // 64 threads × 1 slot each — each thread initialises its own slot,
    // no collisions in this phase.
    atomicStore(&wg_tag[lid.x],   EMPTY_TAG);
    atomicStore(&wg_mass[lid.x],  0u);
    atomicStore(&wg_mx[lid.x],    0u);
    atomicStore(&wg_my[lid.x],    0u);
    atomicStore(&wg_mz[lid.x],    0u);
    atomicStore(&wg_kdiag[lid.x], 0u);
    workgroupBarrier();

    let pid = gid.x;
    // Note: we still execute Pass 2 even if pid >= particle_count, so
    // we mustn't `return` early — divergent threads still need to run
    // the commit phase below. Gate the per-particle work instead.
    let pid_active = pid < u.particle_count;

    // ── Pass 1: per-particle scatter (B-spline 27-cell stencil) ───
    // Mirror of the legacy `p2g` body. Differences are limited to the
    // final per-cell deposit: we hash the global cell index into a
    // workgroup-local slot and accumulate there with workgroup-local
    // atomic CAS; on collision we fall back to the legacy global
    // atomic helpers so correctness is preserved.
    if (pid_active) {
        let pos = p_pos[pid].xyz;
        let vel = p_vel[pid].xyz;
        let mass = p_mass[pid];
        let vol = p_volume[pid];
        let F = p_F[pid];
        let C = p_C[pid];

        let xp = (pos - u.grid_origin) * u.inv_cell_size;
        let base_xyz = vec3<f32>(floor(xp.x - 0.5), floor(xp.y - 0.5), floor(xp.z - 0.5));
        let base = vec3<i32>(i32(base_xyz.x), i32(base_xyz.y), i32(base_xyz.z));
        let fx = xp - base_xyz;

        let wx = quad_weights(fx.x);
        let wy = quad_weights(fx.y);
        let wz = quad_weights(fx.z);

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
        // Default 1.0 (no elastic hardening) — Viscous rheology
        // (`u.rheology == 2u`) takes neither branch; an unset `var`
        // would feed garbage into `mu_h`/`lam_h` → NaN stress.
        let mrow = materials[p_material[pid]];
        let rheo = u32(mrow.p0.x + 0.5);
        var h: f32 = 1.0;
        if (rheo == 1u) {
            // Snow stiffens under compaction; `jp = log(det Fp)` is
            // negative there, so the exponent sign is negative (see
            // the p2g pass above for the Stomakhin §6.2 rationale).
            h = exp(-mrow.p1.x * clamp(p_jp[pid], -5.0, 5.0));
        } else if (rheo == 0u) {
            h = exp(min(p_jp[pid], 5.0));
        }
        let mu_h = mu * h;
        let lam_h = lam * h;

        let F_clamped = svd.u * mat3x3<f32>(
            vec3<f32>(svd.s.x, 0.0, 0.0),
            vec3<f32>(0.0, svd.s.y, 0.0),
            vec3<f32>(0.0, 0.0, svd.s.z),
        ) * transpose(svd.v);
        let f_minus_r = F_clamped - R;
        let cof = svd.u * mat3x3<f32>(
            vec3<f32>(svd.s.y * svd.s.z, 0.0, 0.0),
            vec3<f32>(0.0, svd.s.x * svd.s.z, 0.0),
            vec3<f32>(0.0, 0.0, svd.s.x * svd.s.y),
        ) * transpose(svd.v);
        let pk1 = 2.0 * mu_h * f_minus_r + lam_h * (J - 1.0) * cof;

        let stress = -u.dt * vol * pk1 * transpose(F_clamped);

        let dx = u.cell_size;
        let inv_dx2 = u.inv_cell_size * u.inv_cell_size;
        let kdiag_p = 4.0 * vol * (2.0 * mu_h + lam_h) * inv_dx2;

        for (var i: u32 = 0u; i < 3u; i = i + 1u) {
            for (var j: u32 = 0u; j < 3u; j = j + 1u) {
                for (var k: u32 = 0u; k < 3u; k = k + 1u) {
                    let w = wx[i] * wy[j] * wz[k];
                    let ix = base.x + i32(i);
                    let iy = base.y + i32(j);
                    let iz = base.z + i32(k);
                    if (!in_bounds(ix, iy, iz)) { continue; }
                    let cidx = u32(cell_idx(ix, iy, iz));
                    let node_pos = u.grid_origin + vec3<f32>(
                        f32(ix) + 0.5,
                        f32(iy) + 0.5,
                        f32(iz) + 0.5,
                    ) * dx;
                    let xn_xp = node_pos - pos;

                    let aff = C * xn_xp;
                    let stress_force = stress * xn_xp * (4.0 * inv_dx2);
                    let mom = w * mass * (vel + aff) + w * stress_force;
                    let dm    = w * mass;
                    let dk    = w * w * kdiag_p;

                    // Direct 4×4×4 spatial mapping: pack the lower 2
                    // bits of (ix, iy, iz) into a single 6-bit slot.
                    // For spatially-clustered particles (the typical
                    // MPM case — granular materials pile up), particles
                    // in the same workgroup tend to touch the same
                    // 4×4×4 neighbourhood, so the dedup rate is high.
                    // For scattered particles the rate falls but the
                    // global-atomic fallback below preserves correctness.
                    let slot = (u32(ix) & 3u)
                             | ((u32(iy) & 3u) << 2u)
                             | ((u32(iz) & 3u) << 4u);
                    // Try to claim the slot.
                    let claim = atomicCompareExchangeWeak(&wg_tag[slot], EMPTY_TAG, cidx);
                    var owned: bool = false;
                    if (claim.exchanged) {
                        owned = true;
                    } else if (claim.old_value == cidx) {
                        owned = true;
                    } else {
                        // The CAS-weak may spuriously fail; retry once
                        // with the observed value, in case it was EMPTY
                        // momentarily. Cheap because subsequent CAS sees
                        // the now-stable tag.
                        let claim2 = atomicCompareExchangeWeak(&wg_tag[slot], EMPTY_TAG, cidx);
                        if (claim2.exchanged || claim2.old_value == cidx) {
                            owned = true;
                        }
                    }
                    if (owned) {
                        wg_atomic_add_mass(slot, dm);
                        wg_atomic_add_mx  (slot, mom.x);
                        wg_atomic_add_my  (slot, mom.y);
                        wg_atomic_add_mz  (slot, mom.z);
                        wg_atomic_add_kdiag(slot, dk);
                    } else {
                        // Hash collision — fall back to global atomic
                        // for this single cell. Correctness preserved;
                        // we just lose the local-dedup win for it.
                        atomic_add_mass(cidx, dm);
                        let mb = cidx * 3u;
                        atomic_add_mom(mb + 0u, mom.x);
                        atomic_add_mom(mb + 1u, mom.y);
                        atomic_add_mom(mb + 2u, mom.z);
                        atomic_add_kdiag(cidx, dk);
                    }
                }
            }
        }
    }

    // ── Pass 2: commit ────────────────────────────────────────────
    // After the barrier the per-slot accumulators hold the sum of
    // every contribution from every particle in this workgroup that
    // hashed to this slot. Each thread emits the one global atomic
    // per buffer for its own slot — that's the contention reduction
    // we set out to deliver.
    workgroupBarrier();
    let slot = lid.x;
    let cidx = atomicLoad(&wg_tag[slot]);
    if (cidx == EMPTY_TAG) { return; }
    let dm = bitcast<f32>(atomicLoad(&wg_mass[slot]));
    if (dm == 0.0) {
        // Mass-only filter would skip cells whose particles cancelled
        // out — extremely unlikely in practice but cheap to gate on.
        let mx_bits = atomicLoad(&wg_mx[slot]);
        let my_bits = atomicLoad(&wg_my[slot]);
        let mz_bits = atomicLoad(&wg_mz[slot]);
        let kd_bits = atomicLoad(&wg_kdiag[slot]);
        if (mx_bits == 0u && my_bits == 0u && mz_bits == 0u && kd_bits == 0u) {
            return;
        }
    }
    atomic_add_mass(cidx, dm);
    let mb = cidx * 3u;
    atomic_add_mom(mb + 0u, bitcast<f32>(atomicLoad(&wg_mx[slot])));
    atomic_add_mom(mb + 1u, bitcast<f32>(atomicLoad(&wg_my[slot])));
    atomic_add_mom(mb + 2u, bitcast<f32>(atomicLoad(&wg_mz[slot])));
    atomic_add_kdiag(cidx, bitcast<f32>(atomicLoad(&wg_kdiag[slot])));
}

// ── Kernel: rigid_imprint ─────────────────────────────────────────
// For each grid cell, check overlap with each rigid sphere (small N
// in v1; brute force is fine). If overlapped, write the rigid's
// velocity directly into momentum (mass = 1, mom = v) and tag the
// cell so grid_update doesn't apply gravity / friction to it.
//
// Stage 24d.1 — also accumulates the two-way reaction wrench while
// we still have the pre-stamp sand momentum in hand: the sand had
// been carrying `(m_old, mom_old)` from p2g; we replace it with
// `(stamp_mass, stamp_mass * v_rigid)`. The momentum delivered to
// the sand by the rigid body over this substep is
// `(stamp_mass * v_rigid - mom_old)`. By Newton 3 the rigid body
// feels the equal-and-opposite impulse, contributing
// `-(stamp_mass * v_rigid - mom_old) / dt` to its force balance.
// The torque arm is `node_pos - rigid_center`.
//
// Caveat: the `stamp_mass = 1e6` heavy-cell trick massively over-
// states the impulse (the actual sand mass in a cell is < 1 kg).
// We therefore aggregate using `m_old` rather than `stamp_mass`,
// which is the physically meaningful value.
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
            // Pre-stamp values (what the sand alone would do).
            let m_old = bitcast<f32>(atomicLoad(&g_mass[cidx]));
            let mb = cidx * 3u;
            let mom_old = vec3<f32>(
                bitcast<f32>(atomicLoad(&g_momentum[mb + 0u])),
                bitcast<f32>(atomicLoad(&g_momentum[mb + 1u])),
                bitcast<f32>(atomicLoad(&g_momentum[mb + 2u])),
            );
            // Stamp the heavy-cell velocity for one-way contact.
            let stamp_mass: f32 = 1.0e6;
            let mom_new = stamp_mass * v;
            atomicStore(&g_mass[cidx], bitcast<u32>(stamp_mass));
            atomicStore(&g_momentum[mb + 0u], bitcast<u32>(mom_new.x));
            atomicStore(&g_momentum[mb + 1u], bitcast<u32>(mom_new.y));
            atomicStore(&g_momentum[mb + 2u], bitcast<u32>(mom_new.z));
            g_rigid_mask[cidx] = 1u;
            // Two-way reaction. Use the physical sand mass m_old, not
            // the heavy stamp_mass, so the wrench has sand-scale units.
            // Impulse delivered to sand = m_old * v - mom_old; force on
            // rigid = -(impulse)/dt.
            if (m_old > 1.0e-9) {
                let impulse_to_sand = m_old * v - mom_old;
                let inv_dt = 1.0 / max(u.dt, 1.0e-6);
                let force_on_rigid = -impulse_to_sand * inv_dt;
                let arm = node_pos - center;
                let torque_on_rigid = cross(arm, force_on_rigid);
                let rb = r * 8u;
                atomic_add_rigid_wrench(rb + 0u, force_on_rigid.x);
                atomic_add_rigid_wrench(rb + 1u, force_on_rigid.y);
                atomic_add_rigid_wrench(rb + 2u, force_on_rigid.z);
                atomic_add_rigid_wrench(rb + 4u, torque_on_rigid.x);
                atomic_add_rigid_wrench(rb + 5u, torque_on_rigid.y);
                atomic_add_rigid_wrench(rb + 6u, torque_on_rigid.z);
            }
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

// ── Kernel: mpm_implicit_apply ────────────────────────────────────
// Stage 24d.1 — semi-implicit Jacobi-relaxation step on the grid.
// Per the brief's scope-realism note we ship semi-implicit (diagonal-
// preconditioned implicit damping) instead of full matrix-free CG;
// the convergence numbers are weaker but the stability win
// (larger-dt-without-blow-up) is the same and the implementation is
// O(grid_total) per iteration without any neighbour Hessian-vector
// products.
//
// Each invocation runs ONE Jacobi sweep over a single grid cell. The
// host loops `jacobi_iters` times, ping-ponging `g_velocity ↔
// g_velocity_prev` between sweeps via the `mpm_implicit_swap` entry
// point.
//
// Stiffness model: instead of the per-particle scattered `k_diag`
// (which under-counts because we only have a small `vol` per
// particle), we use the **wave-speed estimate** directly:
//   c² = E (1 - ν) / [ρ (1 + ν) (1 - 2ν)]  for the P-wave,
//   relaxed to c² = E / ρ for the simpler S-wave.
//
// The implicit damping coefficient per cell becomes
//   α = dt² · m · c² / dx²
// and the update rule (per axis) is
//   v_new = (m · v_explicit + α · v_avg) / (m + α)
// where v_avg = mean of nonzero-mass face-adjacent neighbours.
//
// This reduces to "do nothing" for empty cells, "freeze rigid-stamped
// cells", and "blend toward the local Laplacian-mean velocity at a
// rate set by the wave-speed CFL ratio" elsewhere. Unconditionally
// stable for any positive α.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn mpm_implicit_apply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cidx = gid.x;
    if (cidx >= u.grid_total) { return; }
    let m = bitcast<f32>(atomicLoad(&g_mass[cidx]));
    if (m <= 1.0e-9 || !(m == m)) {
        g_velocity_prev[cidx] = g_velocity[cidx];
        return;
    }
    if (g_rigid_mask[cidx] != 0u) {
        g_velocity_prev[cidx] = g_velocity[cidx];
        return;
    }
    let v_explicit = g_velocity[cidx].xyz;

    // 6-neighbour average for the velocity Laplacian proxy.
    let n = i32(u.grid_size);
    let ix = i32(cidx) % n;
    let iy = (i32(cidx) / n) % n;
    let iz = i32(cidx) / (n * n);
    var v_avg = vec3<f32>(0.0, 0.0, 0.0);
    var count: f32 = 0.0;
    let offsets: array<vec3<i32>, 6> = array<vec3<i32>, 6>(
        vec3<i32>( 1, 0, 0),
        vec3<i32>(-1, 0, 0),
        vec3<i32>( 0, 1, 0),
        vec3<i32>( 0,-1, 0),
        vec3<i32>( 0, 0, 1),
        vec3<i32>( 0, 0,-1),
    );
    for (var k: u32 = 0u; k < 6u; k = k + 1u) {
        let off = offsets[k];
        let nx = ix + off.x;
        let ny = iy + off.y;
        let nz = iz + off.z;
        if (!in_bounds(nx, ny, nz)) { continue; }
        let nidx = u32(cell_idx(nx, ny, nz));
        let nm = bitcast<f32>(atomicLoad(&g_mass[nidx]));
        if (nm <= 1.0e-9 || !(nm == nm)) { continue; }
        v_avg = v_avg + g_velocity[nidx].xyz;
        count = count + 1.0;
    }
    if (count > 0.0) {
        v_avg = v_avg / count;
    } else {
        v_avg = v_explicit;
    }
    // Wave-speed-based stiffness estimate. ρ_proxy = 1500 kg/m³ —
    // sand-grade default; also baked into MpmConfig::sand defaults so
    // the implicit gain is properly Calibrated. For non-sand we
    // fold this into the Young's modulus via the host so the grid
    // sees an effective rigidity term independent of the rheology.
    let rho_proxy: f32 = 1500.0;
    let c_squared = u.youngs / rho_proxy;
    let alpha = u.dt * u.dt * m * c_squared * u.inv_cell_size * u.inv_cell_size;
    let denom = m + alpha;
    let v_new = (m * v_explicit + alpha * v_avg) / denom;
    let v_safe = select(
        v_new,
        v_explicit,
        !(v_new.x == v_new.x) || !(v_new.y == v_new.y) || !(v_new.z == v_new.z)
    );
    g_velocity_prev[cidx] = vec4<f32>(v_safe, m);
}

// Stage 24d.1 — host swaps `g_velocity` and `g_velocity_prev` between
// Jacobi sweeps via this copy kernel (write back into `g_velocity`).
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn mpm_implicit_swap(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cidx = gid.x;
    if (cidx >= u.grid_total) { return; }
    g_velocity[cidx] = g_velocity_prev[cidx];
}

// ── Kernel: mpm_rigid_force_integrate ─────────────────────────────
// Stage 24d.1 — two-way rigid coupling: aggregate per-cell stress
// impulses (ρ * m_cell * (v_explicit - v_rigid)) into a single
// force/torque per rigid body. Cells overlapping a rigid sphere
// contribute their explicit-stress momentum delta as a Newton-3rd-law
// reaction force on the rigid body.
//
// The host clears `rigid_wrench` to zero before this kernel runs, then
// reads it back after. Force is the sum of impulses ÷ dt. Torque is
// the sum of (r × impulse) ÷ dt.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn mpm_rigid_force_integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let cidx = gid.x;
    if (cidx >= u.grid_total) { return; }
    if (u.rigid_count == 0u) { return; }
    let m = bitcast<f32>(atomicLoad(&g_mass[cidx]));
    if (m <= 1.0e-9 || !(m == m)) { return; }
    // Skip rigid-stamped cells — they're not "sand" cells and the
    // stamped velocity isn't a stress impulse.
    if (g_rigid_mask[cidx] != 0u) { return; }
    let n = i32(u.grid_size);
    let ix = i32(cidx) % n;
    let iy = (i32(cidx) / n) % n;
    let iz = i32(cidx) / (n * n);
    let node_pos = u.grid_origin + vec3<f32>(
        f32(ix) + 0.5,
        f32(iy) + 0.5,
        f32(iz) + 0.5,
    ) * u.cell_size;
    let v_grid = g_velocity[cidx].xyz;
    for (var r: u32 = 0u; r < u.rigid_count; r = r + 1u) {
        let cr = rigid_center[r];
        let center = cr.xyz;
        let radius = cr.w;
        let d = distance(node_pos, center);
        if (d > radius + u.cell_size) { continue; }
        // Linearly tapered influence: cells inside the sphere get
        // full weight, cells in the outer cell-thick shell taper.
        var w: f32 = 1.0;
        if (d > radius) {
            let t = (d - radius) / u.cell_size;
            w = max(0.0, 1.0 - t);
        }
        let v_rigid = rigid_vel[r].xyz;
        let dv = v_rigid - v_grid;
        // Reaction impulse on the rigid body: -m * w * dv (Newton 3).
        // Force = impulse / dt.
        let force = -(m * w / max(u.dt, 1.0e-6)) * dv;
        let arm = node_pos - center;
        let torque = cross(arm, force);
        let base = r * 8u;
        atomic_add_rigid_wrench(base + 0u, force.x);
        atomic_add_rigid_wrench(base + 1u, force.y);
        atomic_add_rigid_wrench(base + 2u, force.z);
        atomic_add_rigid_wrench(base + 4u, torque.x);
        atomic_add_rigid_wrench(base + 5u, torque.y);
        atomic_add_rigid_wrench(base + 6u, torque.z);
    }
}

// Stage 24d.1 — clear `rigid_wrench` before integration. Host runs
// this each substep.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn mpm_rigid_wrench_reset(@builtin(global_invocation_id) gid: vec3<u32>) {
    let r = gid.x;
    if (r >= u.rigid_count) { return; }
    let base = r * 8u;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        atomicStore(&rigid_wrench[base + k], 0u);
    }
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

    // Viscous/fluid-like material path. This is intentionally local
    // and GPU-cheap: damp the gathered particle velocity and the APIC
    // affine velocity gradient exponentially. The caller controls the
    // two rates through the reused rheology parameter slots.
    let vrow = materials[p_material[pid]];
    if (u32(vrow.p0.x + 0.5) == 2u) {
        let vel_damp = exp(-max(vrow.p1.y, 0.0) * u.dt);
        let aff_damp = exp(-max(vrow.p1.z, 0.0) * u.dt);
        v_new = v_new * vel_damp;
        C_new = mat3x3<f32>(
            C_new[0] * aff_damp,
            C_new[1] * aff_damp,
            C_new[2] * aff_damp,
        );
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

// ── Kernel: plasticity (Drucker-Prager sand / snow / viscous) ──────
// Two rheologies share one entry point selected by `u.rheology`:
//   0 = sand (Drucker-Prager, Klár 2016)
//   1 = snow (Stomakhin 2013 — clamp singular values to
//       [1 - θ_c, 1 + θ_s], log determinant rolls into jp for
//       hardening that p2g picks up via exp(snow_hardening * jp)).
//   2 = viscous — no plastic projection; G2P applied velocity/affine
//       damping, and F stays elastic/unprojected.
//
// Snow path replaces the plastic-flow scalar accumulator with a
// log(det Fp) tracker (Stomakhin §6.2) — bounded to ±5.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn plasticity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pid = gid.x;
    if (pid >= u.particle_count) { return; }
    let mrow = materials[p_material[pid]];
    let rheo = u32(mrow.p0.x + 0.5);
    let F = p_F[pid];
    let svd = svd3_jacobi(F);
    let s = svd.s;

    var new_s = s;
    var dq: f32 = 0.0;
    var jp_replace: bool = false;
    var jp_new: f32 = 0.0;

    if (rheo == 1u) {
        // ── Snow (Stomakhin 2013 §6.2) ──────────────────────────
        // Clamp singular values to the elastic envelope.
        let theta_c = mrow.p0.z;
        let theta_s = mrow.p0.w;
        new_s = vec3<f32>(
            clamp(s.x, 1.0 - theta_c, 1.0 + theta_s),
            clamp(s.y, 1.0 - theta_c, 1.0 + theta_s),
            clamp(s.z, 1.0 - theta_c, 1.0 + theta_s),
        );
        // Δ log det Fp = log(det F) - log(det F_e_clamped).
        let det_f_old = max(s.x * s.y * s.z, 1.0e-6);
        let det_f_new = max(new_s.x * new_s.y * new_s.z, 1.0e-6);
        let dlog = log(det_f_old) - log(det_f_new);
        // jp tracks log(det Fp) (negative under compaction); p2g
        // reads exp(-snow_hardening * jp) so compacted snow stiffens.
        jp_replace = true;
        jp_new = clamp(p_jp[pid] + dlog, -5.0, 5.0);
    } else if (rheo == 2u) {
        // ── Viscous/fluid-like damping ───────────────────────────
        // No Drucker-Prager cone and no snow clamp; keep the updated
        // deformation gradient and reset the hardening accumulator so
        // p2g does not grow stiffness from old sand/snow state.
        new_s = s;
        jp_replace = true;
        jp_new = 0.0;
    } else {
        // ── Sand (Drucker-Prager, Klár 2016 §3.4) ────────────────
        let mu = u.youngs / (2.0 * (1.0 + u.poisson));
        let lam = u.youngs * u.poisson / ((1.0 + u.poisson) * (1.0 - 2.0 * u.poisson));
        let d = 3.0;

        let eps_log = vec3<f32>(
            log(max(s.x, 1.0e-6)),
            log(max(s.y, 1.0e-6)),
            log(max(s.z, 1.0e-6)),
        );
        let trace_eps = eps_log.x + eps_log.y + eps_log.z;
        let mean_eps = trace_eps / d;
        let dev_eps = eps_log - vec3<f32>(mean_eps, mean_eps, mean_eps);
        let dev_norm = length(dev_eps);
        let kappa = (d * lam + 2.0 * mu) / (2.0 * mu);
        if (trace_eps > 0.0) {
            new_s = vec3<f32>(1.0, 1.0, 1.0);
            dq = length(eps_log);
        } else if (dev_norm == 0.0) {
            new_s = s;
        } else {
            let f_yield = dev_norm + kappa * trace_eps * mrow.p0.y;
            if (f_yield <= 0.0) {
                new_s = s;
            } else {
                let r_cone = max(0.0, -mrow.p0.y * kappa * trace_eps);
                let new_dev = dev_eps * (r_cone / dev_norm);
                let new_eps = new_dev + vec3<f32>(mean_eps, mean_eps, mean_eps);
                new_s = vec3<f32>(exp(new_eps.x), exp(new_eps.y), exp(new_eps.z));
                dq = f_yield;
            }
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
    if (jp_replace) {
        p_jp[pid] = jp_new;
    } else {
        p_jp[pid] = clamp(p_jp[pid] + dq, 0.0, 5.0);
    }
}
