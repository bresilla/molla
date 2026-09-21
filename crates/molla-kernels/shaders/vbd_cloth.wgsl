// Vertex Block Descent (VBD) cloth solver — WGSL port.
//
// Reference: Chen, Liu, Yang & Yuksel, "Vertex Block Descent" (2024,
// SIGGRAPH); Giles, Diaz & Yuksel, "Augmented Vertex Block Descent"
// (2025). Algorithm structure copied from Newton's
// `xtra/newton/newton/_src/solvers/vbd/particle_vbd_kernels.py`
// (see `forward_step`, `solve_elasticity`, `update_velocity`).
//
// Outer loop (host-side): for each substep,
//   1. predict        — inertial target y = x + dt·v + dt²·g.
//   2. for `iters` block iterations:
//        for each colour k:
//           solve_color(k) — every vertex in colour k assembles its
//             3×3 Hessian H = m/dt² · I + Σ H_e and gradient
//             g = m/dt² · (x - y) + Σ f_e from incident springs
//             (stretch + bending, both as Hookean springs; bending
//             springs are the skip-one diagonals already laid out by
//             the `add_cloth_grid` builder), and writes
//                x ← x - H⁻¹ · g
//             back into `particle_q`. Within a colour, no two threads
//             share a vertex => race-free.
//        end for-colour
//   3. velocity_recovery — v = (x - x_prev) / dt.
//
// Augmented VBD (rigid coupling) is implemented as a soft penalty
// against shape SDFs in the same per-vertex block: penetration
// depth contributes a spring-like force + Hessian to the vertex
// block. This is the simplest AVBD variant — no augmented-Lagrangian
// dual updates, no friction; sufficient for a kinematic-rigid drape
// demo, and the right scope for v1 per the stage's deferral note.

struct Uniforms {
    particle_count:    u32,
    spring_count:      u32,
    /// Number of vertex colours. The host dispatches one
    /// `solve_color` pass per colour per outer iteration.
    color_count:       u32,
    /// Slot in `particle_color_offsets` for the colour the current
    /// `solve_color` dispatch targets. Repurposed as `shape_count`
    /// for `apply_shape_contacts`.
    color_index:       u32,
    /// Number of valid invocations in this dispatch — particle count
    /// for predict / velocity / shape contacts, particles-in-colour
    /// for `solve_color`.
    work_count:        u32,
    dt:                f32,
    linear_damping:    f32,
    /// Stiffness of the soft-contact penalty against rigid SDFs.
    /// 0 = disable contacts entirely. Recommended ~1e5 N/m for a
    /// 1cm particle radius cloth.
    contact_stiffness: f32,
    /// AVBD mode: 0 = soft (Stage 20), 1 = hard (Stage 20.1
    /// augmented-Lagrangian dual updates).
    is_hard:           u32,
    _pad1:             f32,
};

const MAX_STRETCH_DISABLED: f32 = 1.0;

// ── Group 0: read-only model topology + state-in ────────────────────
@group(0) @binding(0) var<storage, read>      particle_q_in:    array<vec4<f32>>;
@group(0) @binding(1) var<storage, read>      particle_qd_in:   array<vec4<f32>>;
@group(0) @binding(2) var<storage, read>      particle_inv_mass: array<f32>;
@group(0) @binding(3) var<storage, read>      particle_radius:  array<f32>;
@group(0) @binding(4) var<storage, read>      particle_world:   array<i32>;
@group(0) @binding(5) var<storage, read>      gravity:          array<vec4<f32>>;

// ── Group 1: spring topology + per-vertex CSR adjacency ─────────────
@group(1) @binding(0) var<storage, read>      spring_a:                 array<u32>;
@group(1) @binding(1) var<storage, read>      spring_b:                 array<u32>;
@group(1) @binding(2) var<storage, read>      spring_rest_length:       array<f32>;
/// Per-spring stiffness (N/m). Computed from XPBD compliance at
/// build time as `k = 1 / max(α, ε)`.
@group(1) @binding(3) var<storage, read>      spring_stiffness:         array<f32>;
/// Per-vertex CSR offsets into `vertex_adjacent_springs`. Length
/// `particle_count + 1`.
@group(1) @binding(4) var<storage, read>      vertex_adj_offsets:       array<u32>;
/// Flat list of spring indices for each vertex.
@group(1) @binding(5) var<storage, read>      vertex_adj_springs:       array<u32>;
/// CSR offsets into `particle_indices_by_color`.
@group(1) @binding(6) var<storage, read>      particle_color_offsets:   array<u32>;
/// Particle indices sorted by colour.
@group(1) @binding(7) var<storage, read>      particle_indices_by_color: array<u32>;

// ── Group 2: read/write working state ───────────────────────────────
/// Current iterate of vertex positions (overwritten each block solve).
@group(2) @binding(0) var<storage, read_write> particle_q:  array<vec4<f32>>;
/// Output velocities (filled by `velocity_recovery`).
@group(2) @binding(1) var<storage, read_write> particle_qd: array<vec4<f32>>;
/// Per-particle previous-step position `x_n` (for v = (x - x_n)/dt).
@group(2) @binding(2) var<storage, read_write> particle_q_prev: array<vec4<f32>>;
/// Inertial target `y = x_n + dt·(v_n + dt·g)`. Held constant during
/// the block iterations (the "right-hand side" of the implicit step).
@group(2) @binding(3) var<storage, read_write> inertia_y:   array<vec4<f32>>;
/// Hard-mode AVBD per-particle augmented-Lagrangian multiplier λ
/// (one scalar per particle, applied along the most-penetrating
/// contact normal). Updated once per outer iteration by
/// `update_dual`. Soft mode leaves it untouched.
@group(2) @binding(4) var<storage, read_write> contact_lambda: array<f32>;

// ── Group 3: per-dispatch uniforms ──────────────────────────────────
@group(3) @binding(0) var<uniform>             u: Uniforms;

// ── Group 4: rigid shape collision data (Augmented VBD soft path) ──
struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,
};
@group(4) @binding(0) var<storage, read>       shape_geom_kind:  array<u32>;
@group(4) @binding(1) var<storage, read>       shape_geom_data:  array<vec4<f32>>;
@group(4) @binding(2) var<storage, read>       shape_body:       array<i32>;
@group(4) @binding(3) var<storage, read>       shape_transform:  array<Transform>;
@group(4) @binding(4) var<storage, read>       body_q:           array<Transform>;

// ── Group 5: tetrahedra (Stage 20.2 volumetric soft bodies) ────────
// All arrays length-1 dummies when `tet_count == 0` so the bind
// group always has 8 entries; the per-vertex tet loop bails out
// immediately on an empty CSR.
//
// `tet_indices` is flat 4 u32s per tet (matching the host model).
// `tet_dm_inv_x/y/z` are the columns of the inverse rest-state basis.
// `tet_rest_volume / tet_mu / tet_lambda` carry the per-element
// material params.
// `vertex_adj_tets_offsets / vertex_adj_tets` are the per-vertex CSR
// of (tet_id, v_order) pairs.
// Mat3 stored as three padded columns (matches `GpuMat3`, same
// layout as `body_inertia` in `integrate_bodies.wgsl`).
struct Mat3Cols {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
};
@group(5) @binding(0) var<storage, read>       tet_indices:                  array<u32>;
@group(5) @binding(1) var<storage, read>       tet_dm_inv:                   array<Mat3Cols>;
@group(5) @binding(2) var<storage, read>       tet_rest_volume:              array<f32>;
@group(5) @binding(3) var<storage, read>       tet_mu:                       array<f32>;
@group(5) @binding(4) var<storage, read>       tet_lambda:                   array<f32>;
@group(5) @binding(5) var<storage, read>       vertex_adj_tets_offsets:      array<u32>;
@group(5) @binding(6) var<storage, read>       vertex_adj_tets:              array<u32>;
// Stage 20.2 — per-vertex CABLE bending CSR. Each entry is 4 u32s
// `(v_prev, v_mid, v_next, role)` packed flat; per-entry stiffness
// is stored in the parallel `vertex_cable_bending_k` array. Empty
// when no cable has a positive bending stiffness.
@group(5) @binding(7) var<storage, read>       vertex_cable_bend_offsets:    array<u32>;
@group(5) @binding(8) var<storage, read>       vertex_cable_bend_entries:    array<u32>;
@group(5) @binding(9) var<storage, read>       vertex_cable_bend_k:          array<f32>;
// Stage 20.3 — per-tet plasticity state. F_p and ε_p are read-write
// because the `plasticity_return_map` entry point mutates them in
// place once per substep; `solve_color` only reads F_p (for the
// elastic Neo-Hookean computation). Yield + hardening are read-only
// material params.
@group(5) @binding(10) var<storage, read_write> tet_f_plastic:                array<Mat3Cols>;
@group(5) @binding(11) var<storage, read_write> tet_plastic_strain:           array<f32>;
@group(5) @binding(12) var<storage, read>       tet_yield_stress:             array<f32>;
@group(5) @binding(13) var<storage, read>       tet_hardening:                array<f32>;
// Stage 20.4 — kinematic-hardening back-stress α (3×3 symmetric tensor)
// + the per-tet kinematic-hardening modulus c. Both default-zero in
// back-compat models (Stage 20.3 path); the return-mapping kernel
// short-circuits the back-stress branch when c == 0.
@group(5) @binding(14) var<storage, read_write> tet_back_stress:              array<Mat3Cols>;
@group(5) @binding(15) var<storage, read>       tet_kinematic_hardening:      array<f32>;

const KIND_SPHERE:   u32 = 0u;
const KIND_PLANE:    u32 = 1u;
const KIND_BOX:      u32 = 2u;
const KIND_CAPSULE:  u32 = 3u;
const KIND_CYLINDER: u32 = 4u;
const KIND_SKIP:     u32 = 0xffu;
const STATIC_BODY:   i32 = -1;

// ─────────────────────────────────────────────────────────────────
// Predict: inertial target y = x + dt·v + dt²·g.
// Snapshot x_prev = x and store y; particle_q starts the iteration
// at the warm-start `y` (Newton's `forward_step` does exactly this).
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }

    let x = particle_q_in[i].xyz;
    particle_q_prev[i] = vec4<f32>(x, 0.0);
    // Reset the AVBD dual variable at the start of each substep —
    // contact lambdas are *substep*-local (they decay back through
    // velocity recovery; persisting them across substeps over-pushes
    // particles out of the surface).
    contact_lambda[i] = 0.0;

    let inv_m = particle_inv_mass[i];
    if (inv_m <= 0.0) {
        // Pinned: target = current position; velocities zero.
        particle_q[i]  = vec4<f32>(x, 0.0);
        particle_qd[i] = vec4<f32>(0.0);
        inertia_y[i]   = vec4<f32>(x, 0.0);
        return;
    }
    let world = max(particle_world[i], 0);
    let g = gravity[u32(world)].xyz;
    var v = particle_qd_in[i].xyz + g * u.dt;
    if (u.linear_damping > 0.0) {
        v = v * max(0.0, 1.0 - u.linear_damping * u.dt);
    }
    let y = x + v * u.dt;
    inertia_y[i]    = vec4<f32>(y, 0.0);
    particle_q[i]   = vec4<f32>(y, 0.0); // warm start
    particle_qd[i]  = vec4<f32>(v, 0.0); // tentative; rewritten by velocity_recovery
}

// ─────────────────────────────────────────────────────────────────
// Quaternion + transform helpers (mirror xpbd_particles.wgsl).
// ─────────────────────────────────────────────────────────────────
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = q.xyz;
    let t = 2.0 * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}
fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}
fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}
fn xf_compose(parent: Transform, local: Transform) -> Transform {
    var out: Transform;
    out.position = vec4<f32>(parent.position.xyz + quat_rotate(parent.rotation, local.position.xyz), 0.0);
    out.rotation = quat_mul(parent.rotation, local.rotation);
    return out;
}
fn xf_inverse(xf: Transform) -> Transform {
    let inv_rot = quat_conj(xf.rotation);
    var out: Transform;
    out.rotation = inv_rot;
    out.position = vec4<f32>(-quat_rotate(inv_rot, xf.position.xyz), 0.0);
    return out;
}
fn world_xf_of(shape: u32) -> Transform {
    let body = shape_body[shape];
    let local = shape_transform[shape];
    if (body == STATIC_BODY) {
        return local;
    }
    return xf_compose(body_q[u32(body)], local);
}

// SDF helpers — kept identical to xpbd_particles.wgsl so the contact
// model stays consistent across solvers.
fn sd_sphere(p: vec3<f32>, xf: Transform, radius: f32) -> vec4<f32> {
    let d = p - xf.position.xyz;
    let len = length(d);
    if (len < 1e-9) { return vec4<f32>(0.0, 1.0, 0.0, -radius); }
    return vec4<f32>(d / len, len - radius);
}
fn sd_plane(p: vec3<f32>, xf: Transform, n_local: vec3<f32>) -> vec4<f32> {
    let n_world = normalize(quat_rotate(xf.rotation, n_local));
    return vec4<f32>(n_world, dot(p - xf.position.xyz, n_world));
}
fn sd_box(p: vec3<f32>, xf: Transform, half: vec3<f32>) -> vec4<f32> {
    let inv = xf_inverse(xf);
    let p_local = inv.position.xyz + quat_rotate(inv.rotation, p);
    let q = abs(p_local) - half;
    let outside = max(q, vec3<f32>(0.0));
    let outside_len = length(outside);
    let inside_d = min(max(q.x, max(q.y, q.z)), 0.0);
    let sd = outside_len + inside_d;
    var n_local: vec3<f32>;
    if (outside_len > 1e-9) {
        n_local = sign(p_local) * (outside / outside_len);
    } else {
        var n = vec3<f32>(0.0, 1.0, 0.0);
        if (q.x >= q.y && q.x >= q.z) {
            n = vec3<f32>(sign(p_local.x), 0.0, 0.0);
        } else if (q.y >= q.z) {
            n = vec3<f32>(0.0, sign(p_local.y), 0.0);
        } else {
            n = vec3<f32>(0.0, 0.0, sign(p_local.z));
        }
        n_local = n;
    }
    let n_world = normalize(quat_rotate(xf.rotation, n_local));
    return vec4<f32>(n_world, sd);
}

fn shape_sd(p: vec3<f32>, s: u32) -> vec4<f32> {
    let kind = shape_geom_kind[s];
    let xf = world_xf_of(s);
    let data = shape_geom_data[s].xyz;
    if (kind == KIND_SPHERE) { return sd_sphere(p, xf, data.x); }
    if (kind == KIND_PLANE)  { return sd_plane(p, xf, data); }
    if (kind == KIND_BOX)    { return sd_box(p, xf, data); }
    // Capsule + cylinder fall back to "no contact" for v1.
    return vec4<f32>(0.0, 1.0, 0.0, 1e6);
}

// 3x3 inverse — small, dense, no pivoting (Hessians are SPD).
fn mat3_inverse(m: mat3x3<f32>) -> mat3x3<f32> {
    let a = m[0]; let b = m[1]; let c = m[2];
    let det =
          a.x * (b.y * c.z - b.z * c.y)
        - a.y * (b.x * c.z - b.z * c.x)
        + a.z * (b.x * c.y - b.y * c.x);
    if (abs(det) < 1e-20) {
        // Pseudo-fallback: identity (no update). Caller treats this
        // as "skip the block update".
        return mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0),
        );
    }
    let inv_det = 1.0 / det;
    let r0 = vec3<f32>(
         (b.y * c.z - b.z * c.y) * inv_det,
        -(a.y * c.z - a.z * c.y) * inv_det,
         (a.y * b.z - a.z * b.y) * inv_det,
    );
    let r1 = vec3<f32>(
        -(b.x * c.z - b.z * c.x) * inv_det,
         (a.x * c.z - a.z * c.x) * inv_det,
        -(a.x * b.z - a.z * b.x) * inv_det,
    );
    let r2 = vec3<f32>(
         (b.x * c.y - b.y * c.x) * inv_det,
        -(a.x * c.y - a.y * c.x) * inv_det,
         (a.x * b.y - a.y * b.x) * inv_det,
    );
    // Construct columns of the inverse from rows above (transpose).
    return mat3x3<f32>(
        vec3<f32>(r0.x, r1.x, r2.x),
        vec3<f32>(r0.y, r1.y, r2.y),
        vec3<f32>(r0.z, r1.z, r2.z),
    );
}

// Determinant gradient, including singular matrices.
fn cofactor3(m: mat3x3<f32>) -> mat3x3<f32> {
    return mat3x3<f32>(
        cross(m[1], m[2]),
        cross(m[2], m[0]),
        cross(m[0], m[1]),
    );
}

// Stage 20.3 — 3×3 SVD via cyclic one-sided Jacobi (mirrors
// `mpm.wgsl`'s `svd3_jacobi` and the CPU oracle in `dense3.rs::svd3`).
// Returns `(U, σ, V)` such that `M = U · diag(σ) · Vᵀ`.
struct Svd3 { u: mat3x3<f32>, s: vec3<f32>, v: mat3x3<f32> };

fn svd3_jacobi(m_in: mat3x3<f32>) -> Svd3 {
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
                let apq = a[q][p];
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
    var sigma: vec3<f32>;
    var u_out = mat3x3<f32>(
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
        vec3<f32>(0.0, 0.0, 0.0),
    );
    for (var i: u32 = 0u; i < 3u; i = i + 1u) {
        let vi = vec3<f32>(v[i][0], v[i][1], v[i][2]);
        let mvi = m_in * vi;
        let s_norm = length(mvi);
        sigma[i] = s_norm;
        if (s_norm > 1.0e-12) {
            let col = mvi / s_norm;
            u_out[i][0] = col.x;
            u_out[i][1] = col.y;
            u_out[i][2] = col.z;
        } else {
            u_out[i][0] = select(0.0, 1.0, i == 0u);
            u_out[i][1] = select(0.0, 1.0, i == 1u);
            u_out[i][2] = select(0.0, 1.0, i == 2u);
        }
    }
    let detu = determinant(u_out);
    if (detu < 0.0) {
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

// Per-vertex Neo-Hookean force + 3×3 Hessian for tet `tet_id` /
// vertex order `v_order ∈ {0..3}`. Mirrors the CPU
// `neo_hookean_vertex_block`. Uses the Smith et al. 2018 stable-NH
// reparameterisation `μ_NH = μ`, `λ_NH = λ + μ` and drops the
// `s · ∂²J/∂F²` term from the Hessian (zero by Levi-Civita
// antisymmetry on the per-vertex 3×3 projection).
fn tet_neo_hookean_block(
    tet_id: u32,
    v_order: u32,
    out_force: ptr<function, vec3<f32>>,
    out_hess:  ptr<function, mat3x3<f32>>,
) {
    let i0 = tet_indices[tet_id * 4u];
    let i1 = tet_indices[tet_id * 4u + 1u];
    let i2 = tet_indices[tet_id * 4u + 2u];
    let i3 = tet_indices[tet_id * 4u + 3u];
    let v0 = particle_q[i0].xyz;
    let v1 = particle_q[i1].xyz;
    let v2 = particle_q[i2].xyz;
    let v3 = particle_q[i3].xyz;

    // Inverse rest-state basis: stored as 3 padded vec4 columns
    // (Mat3Cols / GpuMat3 layout, same as body_inertia uploads).
    let dm_inv_raw = tet_dm_inv[tet_id];
    let dm_inv = mat3x3<f32>(dm_inv_raw.col0.xyz, dm_inv_raw.col1.xyz, dm_inv_raw.col2.xyz);

    let rest_vol = tet_rest_volume[tet_id];
    let mu_l = tet_mu[tet_id];
    let lam_l = tet_lambda[tet_id];
    let mu = mu_l;
    let lam = lam_l + mu_l;

    // Deformation gradient F = Ds · Dm⁻¹. Stage 20.3 — for plastic
    // tets the elastic Neo-Hookean energy is evaluated on
    // `F_e = F · F_p⁻¹` (multiplicative split). Pure-elastic path
    // (F_p = I) short-circuits the inverse + multiply.
    let ds = mat3x3<f32>(v1 - v0, v2 - v0, v3 - v0);
    let f_full = ds * dm_inv;
    let fp_raw = tet_f_plastic[tet_id];
    let fp_mat = mat3x3<f32>(fp_raw.col0.xyz, fp_raw.col1.xyz, fp_raw.col2.xyz);
    let fp_det = determinant(fp_mat);
    var f_mat = f_full;
    // NaN/degenerate sentinel — bail to identity-F_p when the prior
    // step wrote something we can't safely invert (defensive — the
    // plasticity kernel clamps F_p, but a bug elsewhere shouldn't
    // poison the per-vertex block solve).
    let fp00 = fp_raw.col0.x;
    if (!(fp00 == fp00) || abs(fp_det) < 1e-6) {
        f_mat = f_full;
    } else if (abs(fp_det - 1.0) > 1e-6) {
        // Real plastic deformation — multiply by F_p⁻¹ (3×3 inverse).
        f_mat = f_full * mat3_inverse(fp_mat);
    }

    let j = determinant(f_mat);
    let lam_safe = select(sign(lam) * 1e-6, lam, abs(lam) > 1e-6);
    let alpha = 1.0 + mu / lam_safe;

    let cof = cofactor3(f_mat);
    let s_scalar = lam * (j - alpha);

    // First Piola-Kirchhoff stress P = μ·F + s·cof(F), scaled by rest
    // volume.
    let p_mat = mat3x3<f32>(
        f_mat[0] * mu + cof[0] * s_scalar,
        f_mat[1] * mu + cof[1] * s_scalar,
        f_mat[2] * mu + cof[2] * s_scalar,
    );
    let p_vec = mat3x3<f32>(
        p_mat[0] * rest_vol,
        p_mat[1] * rest_vol,
        p_mat[2] * rest_vol,
    );

    // Per-vertex gradient column `m` from `Dm⁻¹`. Translation of
    // Newton's `compute_G_matrix` to WGSL's column-major mat3x3 — see
    // the matching CPU comment in `solvers/vbd.rs`.
    var m: vec3<f32>;
    if (v_order == 0u) {
        m = vec3<f32>(
            -(dm_inv[0].x + dm_inv[0].y + dm_inv[0].z),
            -(dm_inv[1].x + dm_inv[1].y + dm_inv[1].z),
            -(dm_inv[2].x + dm_inv[2].y + dm_inv[2].z),
        );
    } else if (v_order == 1u) {
        m = vec3<f32>(dm_inv[0].x, dm_inv[1].x, dm_inv[2].x);
    } else if (v_order == 2u) {
        m = vec3<f32>(dm_inv[0].y, dm_inv[1].y, dm_inv[2].y);
    } else {
        m = vec3<f32>(dm_inv[0].z, dm_inv[1].z, dm_inv[2].z);
    }

    // Force = -P · m (column-major reduction).
    let f_clean = -(p_vec[0] * m.x + p_vec[1] * m.y + p_vec[2] * m.z);

    // Hessian per-vertex 3×3 block: μ·|m|²·I + λ·(cof·m)·(cof·m)ᵀ,
    // multiplied by rest volume.
    let cof_m = cof[0] * m.x + cof[1] * m.y + cof[2] * m.z;
    let mu_term = mat3x3<f32>(
        vec3<f32>(mu * dot(m, m), 0.0, 0.0),
        vec3<f32>(0.0, mu * dot(m, m), 0.0),
        vec3<f32>(0.0, 0.0, mu * dot(m, m)),
    );
    let lam_term = mat3x3<f32>(
        cof_m * cof_m.x,
        cof_m * cof_m.y,
        cof_m * cof_m.z,
    );
    var h_block = mat3x3<f32>(
        (mu_term[0] + lam_term[0] * lam) * rest_vol,
        (mu_term[1] + lam_term[1] * lam) * rest_vol,
        (mu_term[2] + lam_term[2] * lam) * rest_vol,
    );
    // Tikhonov-like diagonal floor — keep the per-vertex block SPD
    // even when an inverted tet would otherwise contribute a
    // sign-indefinite term. Negligible vs the inertia diagonal in
    // the outer assembly.
    let h_floor = 1e-8 * max(rest_vol, 1e-12);
    h_block[0].x = max(h_block[0].x, h_floor);
    h_block[1].y = max(h_block[1].y, h_floor);
    h_block[2].z = max(h_block[2].z, h_floor);

    // Stage 20.3 — NaN/Inf sentinel. With plasticity in the loop,
    // an exotic F_p combined with a degenerate F can produce a
    // NaN in `f_clean`/`h_block`. Returning zero force + identity-
    // multiplied Hessian for this tet's contribution makes the
    // per-vertex block solve see "no contribution" — the inertia
    // term dominates and the position update is sane (the particle
    // won't move much, but at least it won't go NaN).
    let any_nan = !(f_clean.x == f_clean.x)
        || !(f_clean.y == f_clean.y)
        || !(f_clean.z == f_clean.z)
        || !(h_block[0].x == h_block[0].x)
        || !(h_block[1].y == h_block[1].y)
        || !(h_block[2].z == h_block[2].z);
    if (any_nan) {
        *out_force = vec3<f32>(0.0);
        *out_hess = mat3x3<f32>(
            vec3<f32>(h_floor, 0.0, 0.0),
            vec3<f32>(0.0, h_floor, 0.0),
            vec3<f32>(0.0, 0.0, h_floor),
        );
        return;
    }

    *out_force = f_clean;
    *out_hess  = h_block;
}

// Spring contribution to a vertex's local force/hessian. Mirrors
// Newton's `evaluate_spring_force_and_hessian_both_vertices` but
// returns just the "self-vertex" side.
//
// For spring (a, b) with current length L, rest length L0, stiffness k:
//
//   diff = x_self - x_other
//   f_self = -k · (1 - L0/L) · diff
//   H = k · ( I - (L0/L) · (I - diff⊗diff/L²) )
//
// (Identical for both endpoints; sign of force flips because diff
// flips.)
fn spring_force_hessian(
    self_idx: u32,
    spring_idx: u32,
    pos_self: vec3<f32>,
    out_force: ptr<function, vec3<f32>>,
    out_hess:  ptr<function, mat3x3<f32>>,
) {
    let a = spring_a[spring_idx];
    let b = spring_b[spring_idx];
    let other = select(a, b, self_idx == a);
    let pos_other = particle_q[other].xyz;
    let diff = pos_self - pos_other;
    let len = max(length(diff), 1e-8);
    let rest = spring_rest_length[spring_idx];
    let k = spring_stiffness[spring_idx];

    let f = -k * (1.0 - rest / len) * diff;
    let outer = mat3x3<f32>(
        vec3<f32>(diff.x * diff.x, diff.y * diff.x, diff.z * diff.x),
        vec3<f32>(diff.x * diff.y, diff.y * diff.y, diff.z * diff.y),
        vec3<f32>(diff.x * diff.z, diff.y * diff.z, diff.z * diff.z),
    );
    let inv_l2 = 1.0 / (len * len);
    let id = mat3x3<f32>(
        vec3<f32>(1.0, 0.0, 0.0),
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
    );
    let outer_scaled = mat3x3<f32>(
        outer[0] * inv_l2, outer[1] * inv_l2, outer[2] * inv_l2,
    );
    let bracket = mat3x3<f32>(
        id[0] - outer_scaled[0],
        id[1] - outer_scaled[1],
        id[2] - outer_scaled[2],
    );
    let l0_over_l = rest / len;
    let h = mat3x3<f32>(
        k * (id[0] - l0_over_l * bracket[0]),
        k * (id[1] - l0_over_l * bracket[1]),
        k * (id[2] - l0_over_l * bracket[2]),
    );

    *out_force = f;
    *out_hess  = h;
}

// ─────────────────────────────────────────────────────────────────
// Per-colour vertex block solve. Race-free under vertex colouring:
// no two particles in the same colour share a spring.
//
// Contact penalty is folded into the block (not a post-pass) so the
// velocity recovery sees a balanced inertia + spring + contact state
// — otherwise the per-iteration contact correction leaks straight
// into velocity and the cloth flies upward.
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn solve_color(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    if (slot >= u.work_count) { return; }
    let base = particle_color_offsets[u.color_index];
    let v = particle_indices_by_color[base + slot];
    if (v >= u.particle_count) { return; }

    let inv_m = particle_inv_mass[v];
    if (inv_m <= 0.0) {
        return; // Pinned: position frozen at `inertia_y` (= x_prev).
    }
    let mass = 1.0 / inv_m;
    let dt2 = u.dt * u.dt;
    let m_over_dt2 = mass / dt2;

    let x = particle_q[v].xyz;
    let y = inertia_y[v].xyz;

    // Inertia term: f_inertia = m/dt² · (y - x), H_inertia = m/dt² · I.
    var f = m_over_dt2 * (y - x);
    var h = mat3x3<f32>(
        vec3<f32>(m_over_dt2, 0.0, 0.0),
        vec3<f32>(0.0, m_over_dt2, 0.0),
        vec3<f32>(0.0, 0.0, m_over_dt2),
    );

    // Sum incident-spring contributions.
    let off_lo = vertex_adj_offsets[v];
    let off_hi = vertex_adj_offsets[v + 1u];
    for (var s = off_lo; s < off_hi; s = s + 1u) {
        let spring_idx = vertex_adj_springs[s];
        var f_s: vec3<f32>;
        var h_s: mat3x3<f32>;
        spring_force_hessian(v, spring_idx, x, &f_s, &h_s);
        f = f + f_s;
        h = mat3x3<f32>(h[0] + h_s[0], h[1] + h_s[1], h[2] + h_s[2]);
    }

    // Stage 20.2: per-tet Neo-Hookean contributions. The CSR
    // entries are (tet_id, v_order) pairs — two u32s per tet. We
    // skip the loop entirely when the per-vertex slice is empty
    // (cloth-only models pay no GPU cost here).
    let to_lo = vertex_adj_tets_offsets[v];
    let to_hi = vertex_adj_tets_offsets[v + 1u];
    for (var t = to_lo; t < to_hi; t = t + 2u) {
        let tet_id = vertex_adj_tets[t];
        let v_order = vertex_adj_tets[t + 1u];
        var f_t: vec3<f32>;
        var h_t: mat3x3<f32>;
        tet_neo_hookean_block(tet_id, v_order, &f_t, &h_t);
        f = f + f_t;
        h = mat3x3<f32>(h[0] + h_t[0], h[1] + h_t[1], h[2] + h_t[2]);
    }

    // Stage 20.2: per-vertex CABLE bending. Each entry is 4 u32s
    // `(v_prev, v_mid, v_next, role)` + a parallel `k_b`. The
    // discrete hinge energy is `½·k_b·|κ|²` where
    // `κ = v_prev + v_next - 2·v_mid`. Force per role:
    //   role 0/2 (endpoint): -k_b · κ           Hessian: k_b · I
    //   role 1   (mid):      +2·k_b · κ         Hessian: 4·k_b · I
    let cb_lo = vertex_cable_bend_offsets[v];
    let cb_hi = vertex_cable_bend_offsets[v + 1u];
    for (var e = cb_lo; e < cb_hi; e = e + 4u) {
        let i_prev = vertex_cable_bend_entries[e];
        let i_mid = vertex_cable_bend_entries[e + 1u];
        let i_next = vertex_cable_bend_entries[e + 2u];
        let role = vertex_cable_bend_entries[e + 3u];
        let kb = vertex_cable_bend_k[e / 4u];
        let kappa = particle_q[i_prev].xyz + particle_q[i_next].xyz - 2.0 * particle_q[i_mid].xyz;
        var f_b: vec3<f32>;
        var h_diag: f32;
        if (role == 1u) {
            f_b = 2.0 * kb * kappa;
            h_diag = 4.0 * kb;
        } else {
            f_b = -kb * kappa;
            h_diag = kb;
        }
        f = f + f_b;
        h = mat3x3<f32>(
            h[0] + vec3<f32>(h_diag, 0.0, 0.0),
            h[1] + vec3<f32>(0.0, h_diag, 0.0),
            h[2] + vec3<f32>(0.0, 0.0, h_diag),
        );
    }

    // Soft rigid-contact penalty (Augmented VBD soft-mode).
    // For each rigid shape with sd(p) < pr (penetration), add a
    // quadratic penalty `½ k_c · (pr - sd)²` along the outward
    // normal. Contributes:
    //   f_c = +k_c · (pr - sd) · n      (push outward)
    //   H_c = +k_c · n nᵀ               (rank-1 outer product)
    // We use the gradient at the current iterate `x`, which is
    // a Gauss-Newton (rather than full Newton) approximation.
    // u.color_index doubles as `shape_count` in the contact dispatch
    // — but here in `solve_color` it's the colour index, so we read
    // shape_count from u.work_count when contact_stiffness > 0?
    // No — work_count is the colour size. Use a separate uniform
    // field `contact_shape_count`? Avoid bloating the uniform by
    // reading the global shape count from `arrayLength(&shape_geom_kind)`.
    if (u.contact_stiffness > 0.0) {
        let pr = particle_radius[v];
        // Hard-mode AVBD adds the per-particle multiplier λ on top of
        // the quadratic penalty. λ is updated once per outer iteration
        // by `update_dual` and is always 0 in soft mode.
        let lam = select(0.0, contact_lambda[v], u.is_hard == 1u);
        let shape_count = arrayLength(&shape_geom_kind);
        for (var s: u32 = 0u; s < shape_count; s = s + 1u) {
            let kind = shape_geom_kind[s];
            if (kind == KIND_SKIP) { continue; }
            let nd = shape_sd(x, s);
            let pen = pr - nd.w;
            if (pen > 0.0) {
                let n_dir = nd.xyz;
                let mag = lam + u.contact_stiffness * pen;
                f = f + n_dir * mag;
                let outer = mat3x3<f32>(
                    vec3<f32>(n_dir.x * n_dir.x, n_dir.y * n_dir.x, n_dir.z * n_dir.x),
                    vec3<f32>(n_dir.x * n_dir.y, n_dir.y * n_dir.y, n_dir.z * n_dir.y),
                    vec3<f32>(n_dir.x * n_dir.z, n_dir.y * n_dir.z, n_dir.z * n_dir.z),
                );
                h = mat3x3<f32>(
                    h[0] + outer[0] * u.contact_stiffness,
                    h[1] + outer[1] * u.contact_stiffness,
                    h[2] + outer[2] * u.contact_stiffness,
                );
            }
        }
    }

    // Newton-Raphson step in position space. Skip if the Hessian is
    // numerically singular (mass/dt² guarantees it isn't, but floats).
    let h_inv = mat3_inverse(h);
    let dx_raw = h_inv * f;
    // Stage 20.3 — NaN/Inf sentinel + magnitude clamp on the position
    // update. With plasticity in the loop, an exotic per-tet F_p can
    // poison the assembled `f`/`h` block in pathological cases
    // despite the per-call guards in `tet_neo_hookean_block`. Refuse
    // to write a non-finite update; cap |dx| at 0.5 m to defuse
    // runaway geometric blow-ups (the per-substep dt is small so any
    // physically-meaningful dx is much less than 0.5 m anyway).
    let dx_ok = (dx_raw.x == dx_raw.x) && (dx_raw.y == dx_raw.y) && (dx_raw.z == dx_raw.z);
    if (dx_ok) {
        let dx_len = length(dx_raw);
        let dx = select(dx_raw, dx_raw * (0.5 / dx_len), dx_len > 0.5);
        particle_q[v] = vec4<f32>(x + dx, 0.0);
    }
}

// ─────────────────────────────────────────────────────────────────
// Augmented VBD soft contact: project each particle out of every
// rigid SDF it penetrates, with a stiffness-weighted line solve.
// One thread per particle. Uses `color_index` as `shape_count` so
// the kernel doesn't need an extra uniform field.
//
// This is the simplest AVBD penalty term — for each penetrating
// shape it solves a 1-D problem along the contact normal:
//   x_new = x + (k_c · pen) / (k_c + m/dt²) · n
// which is the closed-form minimiser of the per-vertex objective
// when the only forces are the inertia term + a unilateral
// quadratic penalty `½ k_c · max(pen, 0)²`.
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn apply_shape_contacts(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }
    let inv_m = particle_inv_mass[i];
    if (inv_m <= 0.0) { return; }
    let pr = particle_radius[i];
    let mass = 1.0 / inv_m;
    let m_over_dt2 = mass / (u.dt * u.dt);

    var p = particle_q[i].xyz;
    let shape_count = u.color_index;
    let kc = u.contact_stiffness;
    if (kc <= 0.0) { return; }
    for (var s: u32 = 0u; s < shape_count; s = s + 1u) {
        let kind = shape_geom_kind[s];
        if (kind == KIND_SKIP) { continue; }
        let nd = shape_sd(p, s);
        let pen = pr - nd.w;
        if (pen > 0.0) {
            let denom = kc + m_over_dt2;
            let alpha = kc / denom;
            p = p + nd.xyz * (pen * alpha);
        }
    }
    particle_q[i] = vec4<f32>(p, 0.0);
}

// ─────────────────────────────────────────────────────────────────
// Hard-mode AVBD dual update — once per outer iteration.
// For each particle, find the maximum residual penetration over all
// shape SDFs and bump the per-particle multiplier:
//   λ ← max(λ + k_c · pen, 0)     (unilateral contact: clamp at 0)
// One thread per particle. Soft mode skips this dispatch entirely.
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn update_dual(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }
    let inv_m = particle_inv_mass[i];
    if (inv_m <= 0.0) { return; }
    if (u.contact_stiffness <= 0.0) { return; }
    let pr = particle_radius[i];
    let p = particle_q[i].xyz;
    var max_pen: f32 = 0.0;
    let shape_count = arrayLength(&shape_geom_kind);
    for (var s: u32 = 0u; s < shape_count; s = s + 1u) {
        let kind = shape_geom_kind[s];
        if (kind == KIND_SKIP) { continue; }
        let nd = shape_sd(p, s);
        let pen = pr - nd.w;
        if (pen > max_pen) { max_pen = pen; }
    }
    let new_lam = contact_lambda[i] + u.contact_stiffness * max_pen;
    contact_lambda[i] = max(new_lam, 0.0);
}

// ─────────────────────────────────────────────────────────────────
// Velocity recovery: v = (x - x_prev) / dt.
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn velocity_recovery(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= u.work_count) { return; }
    if (particle_inv_mass[i] <= 0.0) {
        particle_qd[i] = vec4<f32>(0.0);
        return;
    }
    let dx = particle_q[i].xyz - particle_q_prev[i].xyz;
    particle_qd[i] = vec4<f32>(dx / u.dt, 0.0);
}

// ─────────────────────────────────────────────────────────────────
// Stage 20.3 — von Mises return mapping (one thread per tet).
//
// Reads the post-step positions to evaluate the trial elastic
// gradient `F_e_trial = F · F_p⁻¹`. SVD into `U·Σ·Vᵀ`, take Hencky
// log strains, split into mean + deviator. Trial Kirchhoff deviator
// magnitude `‖s‖ = 2μ · ‖dev_ε‖` checked against `√(2/3)·σ_y(ε_p)`.
// If yielding, project deviator back along its trial direction by
// `Δγ = f_yield / (2μ + (2/3)·H)` (consistent return mapping for
// linear isotropic hardening). Reconstruct `F_e_new = U · Σ_e · Vᵀ`,
// set `F_p_new = F_e_new⁻¹ · F`, accumulate `ε_p`.
//
// Damping: cap the plastic strain increment at half the trial
// deviator norm to defuse overshoot oscillations (the discrete
// implicit-Euler step can pull the trial state past the yield
// surface; halving the increment keeps the iteration stable without
// the cost of a full line search).
// ─────────────────────────────────────────────────────────────────
@compute @workgroup_size(64)
fn plasticity_return_map(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tet = gid.x;
    if (tet >= u.work_count) { return; }
    let sigma_y0 = tet_yield_stress[tet];
    // Pure-elastic short-circuit: σ_y₀ = INFINITY (encoded as a
    // very large finite number on upload to avoid f32 inf-bit
    // surprises).
    if (sigma_y0 > 1.0e29) { return; }
    let mu_l = tet_mu[tet];
    let h_mod = tet_hardening[tet];

    let i0 = tet_indices[tet * 4u];
    let i1 = tet_indices[tet * 4u + 1u];
    let i2 = tet_indices[tet * 4u + 2u];
    let i3 = tet_indices[tet * 4u + 3u];
    let p0 = particle_q[i0].xyz;
    let p1 = particle_q[i1].xyz;
    let p2 = particle_q[i2].xyz;
    let p3 = particle_q[i3].xyz;

    let dm_inv_raw = tet_dm_inv[tet];
    let dm_inv = mat3x3<f32>(dm_inv_raw.col0.xyz, dm_inv_raw.col1.xyz, dm_inv_raw.col2.xyz);
    let ds = mat3x3<f32>(p1 - p0, p2 - p0, p3 - p0);
    let f_full = ds * dm_inv;

    let fp_raw = tet_f_plastic[tet];
    var fp_old = mat3x3<f32>(fp_raw.col0.xyz, fp_raw.col1.xyz, fp_raw.col2.xyz);
    let fp_det = determinant(fp_old);
    let fp00 = fp_raw.col0.x;
    if (!(fp00 == fp00) || abs(fp_det) < 1e-12 || abs(fp_det) > 50.0) {
        // Reset on degenerate / divergent F_p.
        tet_f_plastic[tet] = Mat3Cols(
            vec4<f32>(1.0, 0.0, 0.0, 0.0),
            vec4<f32>(0.0, 1.0, 0.0, 0.0),
            vec4<f32>(0.0, 0.0, 1.0, 0.0),
        );
        fp_old = mat3x3<f32>(
            vec3<f32>(1.0, 0.0, 0.0),
            vec3<f32>(0.0, 1.0, 0.0),
            vec3<f32>(0.0, 0.0, 1.0),
        );
    }
    let fe_trial = f_full * mat3_inverse(fp_old);

    let svd = svd3_jacobi(fe_trial);
    let sigma = vec3<f32>(
        clamp(svd.s.x, 1e-3, 1e3),
        clamp(svd.s.y, 1e-3, 1e3),
        clamp(svd.s.z, 1e-3, 1e3),
    );
    let eps = vec3<f32>(log(sigma.x), log(sigma.y), log(sigma.z));
    let trace_eps = eps.x + eps.y + eps.z;
    let mean = trace_eps / 3.0;
    let dev = eps - vec3<f32>(mean, mean, mean);

    // Stage 20.4 — kinematic-hardening back-stress (Prager rule).
    // Project α into the SVD's left-singular frame (`Uᵀ·α·U`), then
    // take the diagonal as the principal back-stress vector. This
    // assumes α stays approximately aligned with the trial-strain
    // principal axes substep-to-substep — true under monotonic
    // loading; small misalignment under cyclic / rotating load is
    // bounded by the per-substep dt being small.
    let c_kin = tet_kinematic_hardening[tet];
    let alpha_raw = tet_back_stress[tet];
    let alpha_old = mat3x3<f32>(alpha_raw.col0.xyz, alpha_raw.col1.xyz, alpha_raw.col2.xyz);
    let alpha_local = transpose(svd.u) * alpha_old * svd.u;
    let alpha_p = vec3<f32>(alpha_local[0].x, alpha_local[1].y, alpha_local[2].z);

    // Shifted principal deviator: ξ = 2μ·dev - α_p (already in stress
    // units; the scalar 2μ converts log-strain to Kirchhoff stress).
    let two_mu = 2.0 * mu_l;
    let xi = two_mu * dev - alpha_p;
    let xi_norm = length(xi);

    let sigma_y = sigma_y0 + h_mod * tet_plastic_strain[tet];
    let sqrt_2_3: f32 = 0.81649658;
    let f_yield = xi_norm - sqrt_2_3 * sigma_y;
    if (f_yield <= 0.0 || xi_norm < 1e-12) {
        return;
    }
    // Plastic multiplier: closed-form for linear isotropic + kinematic
    // hardening (consistent return mapping). The (2/3)·c contribution
    // grows the denominator by the same shape as `H`, so a positive
    // `c` shrinks Δγ proportionally — desirable, that's what makes
    // back-stress well-behaved under cyclic loading.
    let denom = two_mu + (2.0 / 3.0) * h_mod + (2.0 / 3.0) * c_kin;
    let dgamma_raw = max(f_yield / denom, 0.0);
    // Damping cap (overshoot defuse) — same structure as Stage 20.3:
    //   (a) Δγ ≤ 0.1 · (‖ξ‖ / 2μ) — proportional to the (now shifted)
    //       trial state (scaled by 1/2μ to keep units of strain).
    //   (b) Δγ ≤ 0.01           — absolute strain-rate cap.
    let dgamma = min(min(dgamma_raw, 0.1 * xi_norm / max(two_mu, 1.0)), 0.01);
    // Plastic flow direction in the principal frame: n̂ = ξ / ‖ξ‖.
    let n_hat = xi / xi_norm;
    // Pull the principal log-strain deviator back along n̂ by Δγ.
    let new_dev = dev - n_hat * dgamma;
    let new_eps = new_dev + vec3<f32>(mean, mean, mean);
    let new_sigma = vec3<f32>(
        clamp(exp(new_eps.x), 0.05, 20.0),
        clamp(exp(new_eps.y), 0.05, 20.0),
        clamp(exp(new_eps.z), 0.05, 20.0),
    );
    let sigma_diag = mat3x3<f32>(
        vec3<f32>(new_sigma.x, 0.0, 0.0),
        vec3<f32>(0.0, new_sigma.y, 0.0),
        vec3<f32>(0.0, 0.0, new_sigma.z),
    );
    let fe_new = svd.u * sigma_diag * transpose(svd.v);
    let fe_det = determinant(fe_new);
    if (!(fe_det == fe_det) || abs(fe_det) < 1e-12) {
        return;
    }
    let fp_new = mat3_inverse(fe_new) * f_full;
    let fp_new_det = determinant(fp_new);
    if (!(fp_new_det == fp_new_det) || abs(fp_new_det) < 0.05 || abs(fp_new_det) > 20.0) {
        // Sanity-clamp: skip the update if F_p_new is degenerate. The
        // CPU oracle does the same.
        return;
    }
    // Robust SVD-based clamp on F_p_new's own singular values, in
    // place. The closed-form return mapping with a single SVD per
    // step can produce an F_p with σ ∈ [σ_min, σ_max] outside a
    // useful range; clamping σ ∈ [0.5, 2.0] keeps the next step's
    // elastic computation `F · F_p⁻¹` numerically sane (the inverse
    // blows up proportionally to 1/σ_min). Documented in
    // `solvers/vbd.rs` plasticity comments — this is the cheap
    // alternative to a full per-step line search.
    //
    // Note: svd3_jacobi can return negative σ when it absorbs a
    // det(U) < 0 sign flip. We force F_p to det > 0 by taking
    // |σ| then absorbing any sign flip into U (flipping the
    // smallest-σ U column if det(U·diag(σ)·Vᵀ) ends up negative).
    let fp_svd = svd3_jacobi(fp_new);
    let fp_sigma_clamped = vec3<f32>(
        clamp(abs(fp_svd.s.x), 0.5, 2.0),
        clamp(abs(fp_svd.s.y), 0.5, 2.0),
        clamp(abs(fp_svd.s.z), 0.5, 2.0),
    );
    let fp_diag = mat3x3<f32>(
        vec3<f32>(fp_sigma_clamped.x, 0.0, 0.0),
        vec3<f32>(0.0, fp_sigma_clamped.y, 0.0),
        vec3<f32>(0.0, 0.0, fp_sigma_clamped.z),
    );
    var fp_clamped = fp_svd.u * fp_diag * transpose(fp_svd.v);
    // Flip the sign of one column of fp_clamped to get det(F_p) > 0
    // if the SVD's sign convention flipped it negative. We pick the
    // column with the smallest |σ| (the rotation impact is smallest).
    if (determinant(fp_clamped) < 0.0) {
        var min_i: u32 = 0u;
        if (fp_sigma_clamped.y < fp_sigma_clamped[min_i]) { min_i = 1u; }
        if (fp_sigma_clamped.z < fp_sigma_clamped[min_i]) { min_i = 2u; }
        fp_clamped[min_i] = -fp_clamped[min_i];
    }
    // Final NaN sentinel — refuse to write a non-finite F_p.
    let test_val = fp_clamped[0].x;
    if (!(test_val == test_val)) {
        return;
    }
    tet_f_plastic[tet] = Mat3Cols(
        vec4<f32>(fp_clamped[0], 0.0),
        vec4<f32>(fp_clamped[1], 0.0),
        vec4<f32>(fp_clamped[2], 0.0),
    );
    tet_plastic_strain[tet] = min(tet_plastic_strain[tet] + sqrt_2_3 * dgamma, 10.0);

    // Stage 20.4 — back-stress evolution (Prager rule). No-op when
    // c == 0 (Stage 20.3 path). The principal-frame update
    // `α_p_new = α_p + (2/3)·c·Δγ·n̂` is rotated back to world via
    // `α = U·diag(α_p_new)·Uᵀ`. Cap |α_p| at 100·σ_y0 (or 1e6, whichever
    // is larger) so a misbehaving substep doesn't accumulate unbounded.
    if (c_kin > 0.0) {
        let dalpha_p = n_hat * ((2.0 / 3.0) * c_kin * dgamma);
        let alpha_p_new = alpha_p + dalpha_p;
        let alpha_cap = max(100.0 * sigma_y0, 1.0e6);
        let alpha_p_clamped = vec3<f32>(
            clamp(alpha_p_new.x, -alpha_cap, alpha_cap),
            clamp(alpha_p_new.y, -alpha_cap, alpha_cap),
            clamp(alpha_p_new.z, -alpha_cap, alpha_cap),
        );
        let alpha_diag = mat3x3<f32>(
            vec3<f32>(alpha_p_clamped.x, 0.0, 0.0),
            vec3<f32>(0.0, alpha_p_clamped.y, 0.0),
            vec3<f32>(0.0, 0.0, alpha_p_clamped.z),
        );
        let alpha_world = svd.u * alpha_diag * transpose(svd.u);
        // Symmetrise — guard against numerical drift introducing
        // an antisymmetric component.
        let alpha_sym = mat3x3<f32>(
            (alpha_world[0] + vec3<f32>(alpha_world[0].x, alpha_world[1].x, alpha_world[2].x)) * 0.5,
            (alpha_world[1] + vec3<f32>(alpha_world[0].y, alpha_world[1].y, alpha_world[2].y)) * 0.5,
            (alpha_world[2] + vec3<f32>(alpha_world[0].z, alpha_world[1].z, alpha_world[2].z)) * 0.5,
        );
        // NaN sentinel.
        let av = alpha_sym[0].x;
        if (av == av) {
            tet_back_stress[tet] = Mat3Cols(
                vec4<f32>(alpha_sym[0], 0.0),
                vec4<f32>(alpha_sym[1], 0.0),
                vec4<f32>(alpha_sym[2], 0.0),
            );
        }
    }
}
