// Stage F1 — Weakly-compressible SPH (WCSPH) fluid.
//
// GPU-resident SPH fluid. Pipeline, one substep:
//   reset_grid       — zero the per-cell particle counters.
//   build_grid       — scatter each particle index into its cell bucket
//                      (fixed-capacity uniform grid; atomic append).
//   compute_density  — ρ_i = Σ_j m W(r_ij,h) over the 27 neighbour
//                      cells; Tait pressure p_i from ρ_i.
//   compute_forces   — symmetric SPH pressure accel + Monaghan
//                      artificial viscosity + gravity → p_accel.
//   integrate        — symplectic Euler; clamp to the domain box.
//
// Neighbour search is a fixed-capacity uniform grid (cell size = 2h =
// the cubic-spline support radius), so a 3×3×3 cell stencil captures
// every pair with W > 0. A radix-sort cell sort is the unbounded-count
// alternative (see PLAN_ENGINE.md F1); the fixed grid keeps F1 free of
// extra crate dependencies and is robust for WCSPH's bounded density.

const WORKGROUP_SIZE: u32 = 64u;
const PI: f32 = 3.14159265358979;

// F4 — granular regularisation safety factor. The sub-yield effective
// viscosity τ_y/(2·reg) is pinned to the explicit-stability ceiling
// ρh²/(STRESS_STAB_K·dt); a larger factor is safer but lets the pile
// creep faster, and the yield sharpens toward ideal plasticity as the
// time step shrinks.
const STRESS_STAB_K: f32 = 16.0;

struct Uniforms {
    particle_count: u32,
    grid_res: u32,
    grid_total: u32,
    max_per_cell: u32,

    grid_origin: vec3<f32>,
    cell_size: f32,

    domain_min: vec3<f32>,
    h: f32,

    domain_max: vec3<f32>,
    dt: f32,

    mass: f32,
    rest_density: f32,
    stiffness_b: f32,
    gamma: f32,

    viscosity: f32,
    wall_restitution: f32,
    sound_speed: f32,
    n_fluid: u32,

    gravity: vec3<f32>,
    friction: f32,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

// Particle SoA.
@group(1) @binding(0) var<storage, read_write> p_pos:      array<vec4<f32>>; // (xyz, _)
@group(1) @binding(1) var<storage, read_write> p_vel:      array<vec4<f32>>; // (xyz, _)
@group(1) @binding(2) var<storage, read_write> p_density:  array<f32>;
@group(1) @binding(3) var<storage, read_write> p_pressure: array<f32>;
@group(1) @binding(4) var<storage, read_write> p_accel:    array<vec4<f32>>; // (xyz, _)
// F4 — per-particle symmetric deviatoric stress: 6 floats per particle,
// [xx, yy, zz, xy, xz, yz]. Zero for ordinary fluid (friction == 0).
@group(1) @binding(5) var<storage, read_write> p_stress:   array<f32>;

// Fixed-capacity uniform grid: g_count[c] particles in cell c, their
// indices in g_cell[c * max_per_cell + slot].
@group(2) @binding(0) var<storage, read_write> g_count: array<atomic<u32>>;
@group(2) @binding(1) var<storage, read_write> g_cell:  array<u32>;

// ── Helpers ───────────────────────────────────────────────────────

fn cell_coord(pos: vec3<f32>) -> vec3<i32> {
    let n = i32(u.grid_res);
    let c = vec3<i32>(floor((pos - u.grid_origin) / u.cell_size));
    return clamp(c, vec3<i32>(0, 0, 0), vec3<i32>(n - 1, n - 1, n - 1));
}

fn cell_index(c: vec3<i32>) -> i32 {
    let n = i32(u.grid_res);
    return c.x + n * (c.y + n * c.z);
}

// Cubic-spline kernel (Monaghan), 3D normalisation. Support is q < 2.
fn kernel_w(r: f32) -> f32 {
    let h = u.h;
    let q = r / h;
    let sigma = 1.0 / (PI * h * h * h);
    if (q < 1.0) {
        return sigma * (1.0 - 1.5 * q * q + 0.75 * q * q * q);
    } else if (q < 2.0) {
        let t = 2.0 - q;
        return sigma * 0.25 * t * t * t;
    }
    return 0.0;
}

// ∇_i W(r_i - r_j); points from i toward j (W decreases with r).
fn kernel_grad_w(rvec: vec3<f32>, r: f32) -> vec3<f32> {
    let h = u.h;
    if (r < 1e-7) {
        return vec3<f32>(0.0, 0.0, 0.0);
    }
    let q = r / h;
    let sigma = 1.0 / (PI * h * h * h);
    var dwdq = 0.0;
    if (q < 1.0) {
        dwdq = sigma * (-3.0 * q + 2.25 * q * q);
    } else if (q < 2.0) {
        let t = 2.0 - q;
        dwdq = sigma * (-0.75 * t * t);
    }
    let dwdr = dwdq / h;
    return (dwdr / r) * rvec;
}

// ── Kernels ───────────────────────────────────────────────────────

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn reset_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= u.grid_total) { return; }
    atomicStore(&g_count[idx], 0u);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn build_grid(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.particle_count) { return; }
    let cell = u32(cell_index(cell_coord(p_pos[i].xyz)));
    let slot = atomicAdd(&g_count[cell], 1u);
    if (slot < u.max_per_cell) {
        g_cell[cell * u.max_per_cell + slot] = i;
    }
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn compute_density(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.particle_count) { return; }
    let xi = p_pos[i].xyz;
    let c = cell_coord(xi);
    let n = i32(u.grid_res);
    var rho = 0.0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let cc = c + vec3<i32>(dx, dy, dz);
                if (cc.x < 0 || cc.y < 0 || cc.z < 0 ||
                    cc.x >= n || cc.y >= n || cc.z >= n) { continue; }
                let cell = u32(cell_index(cc));
                let cnt = min(atomicLoad(&g_count[cell]), u.max_per_cell);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = g_cell[cell * u.max_per_cell + s];
                    let r = length(xi - p_pos[j].xyz);
                    rho = rho + u.mass * kernel_w(r);
                }
            }
        }
    }
    rho = max(rho, 1e-6);
    p_density[i] = rho;
    // Tait equation of state. Clamp ≥ 0 so the free surface does not
    // pull particles together with negative pressure.
    let ratio = rho / u.rest_density;
    p_pressure[i] = max(u.stiffness_b * (pow(ratio, u.gamma) - 1.0), 0.0);
}

// F4 — granular deviatoric stress: a regularised perfectly-plastic
// Drucker-Prager rheology. The SPH velocity gradient gives the
// deviatoric strain rate D'; the deviatoric stress is
//   S = τ_y · D' / (‖D'‖ + reg),   τ_y = friction · pressure,
// so ‖S‖ ≤ τ_y (the Drucker-Prager yield) and S stays aligned with D'.
// Alignment makes S:D' ≥ 0 — the stress is dissipative by construction:
// it always opposes the flow and can never pump energy into it. (That
// pumping was the failure mode of the earlier hypoelastic model, whose
// frozen leftover stress drove the pile to keep spreading.) `reg` is a
// per-particle, dt-scaled regularisation that pins the sub-yield
// effective viscosity τ_y/(2·reg) to the explicit-stability ceiling
// ρh²/(STRESS_STAB_K·dt), so the kernel is stable at any dt and the
// yield sharpens toward ideal plasticity as dt → 0. friction == 0 →
// ordinary fluid, S stays zero.
@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn compute_stress(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.particle_count) { return; }
    let base = i * 6u;
    if (u.friction <= 0.0 || i >= u.n_fluid) {
        for (var k = 0u; k < 6u; k = k + 1u) {
            p_stress[base + k] = 0.0;
        }
        return;
    }
    let xi = p_pos[i].xyz;
    let vi = p_vel[i].xyz;
    let c = cell_coord(xi);
    let n = i32(u.grid_res);
    // Velocity gradient L: row a = ∂v_a/∂x = Σ_j vol_j (v_j−v_i)_a ∇W.
    var l0 = vec3<f32>(0.0, 0.0, 0.0);
    var l1 = vec3<f32>(0.0, 0.0, 0.0);
    var l2 = vec3<f32>(0.0, 0.0, 0.0);
    // SPH-smoothed (Shepard) pressure for the yield: the raw Tait
    // pressure of a barely-compressed granular pile is dominated by
    // per-particle density noise, so the kernel-interpolated pressure
    // p_y = Σ V_j p_j W / Σ V_j W is used to set τ_y instead. This also
    // lends a surface particle (own pressure ≈ 0) a little yield
    // strength from the supported material around it. Seeded with the
    // particle's own contribution.
    let voli = u.mass / max(p_density[i], 1e-6);
    let w0 = kernel_w(0.0);
    var psum = voli * max(p_pressure[i], 0.0) * w0;
    var wsum = voli * w0;
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let cc = c + vec3<i32>(dx, dy, dz);
                if (cc.x < 0 || cc.y < 0 || cc.z < 0 ||
                    cc.x >= n || cc.y >= n || cc.z >= n) { continue; }
                let cell = u32(cell_index(cc));
                let cnt = min(atomicLoad(&g_count[cell]), u.max_per_cell);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = g_cell[cell * u.max_per_cell + s];
                    if (j == i) { continue; }
                    let rvec = xi - p_pos[j].xyz;
                    let r = length(rvec);
                    if (r >= 2.0 * u.h || r < 1e-7) { continue; }
                    let gw = kernel_grad_w(rvec, r);
                    let dv = p_vel[j].xyz - vi;
                    let vol = u.mass / max(p_density[j], 1e-6);
                    l0 = l0 + vol * dv.x * gw;
                    l1 = l1 + vol * dv.y * gw;
                    l2 = l2 + vol * dv.z * gw;
                    let wj = kernel_w(r);
                    psum = psum + vol * max(p_pressure[j], 0.0) * wj;
                    wsum = wsum + vol * wj;
                }
            }
        }
    }
    // Deviatoric strain rate D' = dev(sym(L)).
    let dxx = l0.x;
    let dyy = l1.y;
    let dzz = l2.z;
    let dxy = 0.5 * (l0.y + l1.x);
    let dxz = 0.5 * (l0.z + l2.x);
    let dyz = 0.5 * (l1.z + l2.y);
    let tr = (dxx + dyy + dzz) / 3.0;
    let exx = dxx - tr;
    let eyy = dyy - tr;
    let ezz = dzz - tr;
    let dnorm = sqrt(exx * exx + eyy * eyy + ezz * ezz
        + 2.0 * (dxy * dxy + dxz * dxz + dyz * dyz));
    // Drucker-Prager yield τ_y (from the smoothed pressure) and the
    // dt-scaled regularisation. `reg` pins the sub-yield effective
    // viscosity to the stable ceiling.
    let p_yield = psum / max(wsum, 1e-6);
    let tau_y = u.friction * max(p_yield, 0.0);
    let reg = max(1.0e-4,
        tau_y * STRESS_STAB_K * u.dt / (2.0 * u.rest_density * u.h * u.h));
    // S = τ_y · D'/(‖D'‖+reg): magnitude ≤ τ_y, aligned with D'.
    let scale = tau_y / (dnorm + reg);
    p_stress[base + 0u] = scale * exx;
    p_stress[base + 1u] = scale * eyy;
    p_stress[base + 2u] = scale * ezz;
    p_stress[base + 3u] = scale * dxy;
    p_stress[base + 4u] = scale * dxz;
    p_stress[base + 5u] = scale * dyz;
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn compute_forces(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.particle_count) { return; }
    // Fluid particles get the full force (gravity + all neighbours).
    // Boundary particles accumulate only the force from fluid
    // neighbours — that is the per-particle fluid load the FSI wrench
    // reduction sums over a rigid body. Static walls compute it too but
    // the host ignores their result.
    let i_is_fluid = i < u.n_fluid;
    let xi = p_pos[i].xyz;
    let vi = p_vel[i].xyz;
    let rho_i = p_density[i];
    let p_i = p_pressure[i];
    var accel = vec3<f32>(0.0, 0.0, 0.0);
    if (i_is_fluid) { accel = u.gravity; }
    let c = cell_coord(xi);
    let n = i32(u.grid_res);
    for (var dz = -1; dz <= 1; dz = dz + 1) {
        for (var dy = -1; dy <= 1; dy = dy + 1) {
            for (var dx = -1; dx <= 1; dx = dx + 1) {
                let cc = c + vec3<i32>(dx, dy, dz);
                if (cc.x < 0 || cc.y < 0 || cc.z < 0 ||
                    cc.x >= n || cc.y >= n || cc.z >= n) { continue; }
                let cell = u32(cell_index(cc));
                let cnt = min(atomicLoad(&g_count[cell]), u.max_per_cell);
                for (var s = 0u; s < cnt; s = s + 1u) {
                    let j = g_cell[cell * u.max_per_cell + s];
                    if (j == i) { continue; }
                    // Boundary particles feel only fluid neighbours.
                    if (!i_is_fluid && j >= u.n_fluid) { continue; }
                    let rvec = xi - p_pos[j].xyz;
                    let r = length(rvec);
                    if (r >= 2.0 * u.h || r < 1e-7) { continue; }
                    let rho_j = p_density[j];
                    let p_j = p_pressure[j];
                    let gw = kernel_grad_w(rvec, r);
                    // Symmetric SPH pressure acceleration. For a
                    // boundary particle the self-pressure is meaningless,
                    // so mirror the fluid neighbour's pressure (Akinci-
                    // style): the body then feels the true fluid pressure
                    // field, which recovers buoyancy.
                    var pterm = p_i / (rho_i * rho_i) + p_j / (rho_j * rho_j);
                    if (!i_is_fluid) {
                        pterm = 2.0 * p_j / (rho_j * rho_j);
                    }
                    accel = accel - u.mass * pterm * gw;
                    // Monaghan artificial viscosity (linear term only).
                    let vij = vi - p_vel[j].xyz;
                    let vr = dot(vij, rvec);
                    if (vr < 0.0) {
                        let mu = u.h * vr / (dot(rvec, rvec) + 0.01 * u.h * u.h);
                        let rho_bar = 0.5 * (rho_i + rho_j);
                        let visc = -u.viscosity * u.sound_speed * mu / rho_bar;
                        accel = accel - u.mass * visc * gw;
                    }
                    // F4 — granular deviatoric stress force, the SPH
                    // form a_i += Σ_j m_j (S_i/ρ_i² + S_j/ρ_j²)·∇W.
                    // S carries the material's shear strength up to the
                    // Drucker-Prager yield, so a pile holds its slope.
                    if (i_is_fluid && u.friction > 0.0) {
                        let si = i * 6u;
                        let sj = j * 6u;
                        let wi = 1.0 / (rho_i * rho_i);
                        let wj = 1.0 / (rho_j * rho_j);
                        let mxx = p_stress[si + 0u] * wi + p_stress[sj + 0u] * wj;
                        let myy = p_stress[si + 1u] * wi + p_stress[sj + 1u] * wj;
                        let mzz = p_stress[si + 2u] * wi + p_stress[sj + 2u] * wj;
                        let mxy = p_stress[si + 3u] * wi + p_stress[sj + 3u] * wj;
                        let mxz = p_stress[si + 4u] * wi + p_stress[sj + 4u] * wj;
                        let myz = p_stress[si + 5u] * wi + p_stress[sj + 5u] * wj;
                        accel = accel + u.mass * vec3<f32>(
                            mxx * gw.x + mxy * gw.y + mxz * gw.z,
                            mxy * gw.x + myy * gw.y + myz * gw.z,
                            mxz * gw.x + myz * gw.y + mzz * gw.z,
                        );
                    }
                }
            }
        }
    }
    p_accel[i] = vec4<f32>(accel, 0.0);
}

@compute @workgroup_size(WORKGROUP_SIZE, 1, 1)
fn integrate(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    // Only fluid particles (index < n_fluid) move; boundary is static.
    if (i >= u.n_fluid) { return; }
    var v = p_vel[i].xyz + p_accel[i].xyz * u.dt;
    var x = p_pos[i].xyz + v * u.dt;
    let e = u.wall_restitution;
    if (x.x < u.domain_min.x) { x.x = u.domain_min.x; v.x = -e * v.x; }
    if (x.x > u.domain_max.x) { x.x = u.domain_max.x; v.x = -e * v.x; }
    // Floor: reflect the normal velocity, and — for the granular preset
    // — apply Coulomb basal friction so the pile cannot slide off the
    // floor as a rigid block (a frictionless floor flattens any pile
    // regardless of the internal rheology). The tangential velocity is
    // cut by at most friction × the normal impulse, the impulse floored
    // at the gravity-support value |g|·dt so a resting layer is still
    // held. The cut is a bounded [0,1] factor → unconditionally stable.
    if (x.y < u.domain_min.y) {
        let vy_in = v.y;
        x.y = u.domain_min.y;
        v.y = -e * vy_in;
        if (u.friction > 0.0) {
            let dvn = max(abs(vy_in) * (1.0 + e), abs(u.gravity.y) * u.dt);
            let vt = length(vec2<f32>(v.x, v.z));
            if (vt > 1.0e-8) {
                let factor = max(0.0, 1.0 - u.friction * dvn / vt);
                v.x = v.x * factor;
                v.z = v.z * factor;
            }
        }
    }
    if (x.y > u.domain_max.y) { x.y = u.domain_max.y; v.y = -e * v.y; }
    if (x.z < u.domain_min.z) { x.z = u.domain_min.z; v.z = -e * v.z; }
    if (x.z > u.domain_max.z) { x.z = u.domain_max.z; v.z = -e * v.z; }
    p_vel[i] = vec4<f32>(v, 0.0);
    p_pos[i] = vec4<f32>(x, 0.0);
}
