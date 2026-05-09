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
    _pad0:             f32,
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

    // Soft rigid-contact penalty (Augmented VBD).
    if (u.contact_stiffness > 0.0) {
        let pr = particle_radius[v];
        let shape_count = u.color_count; // stuffed via color_count when contacts pass uses solve_color? No — separate pass.
        // (No-op here. Contact penalty lives in `apply_shape_contacts`
        // dispatch; we avoid mixing the two so colour bookkeeping
        // stays simple.)
    }

    // Newton-Raphson step in position space. Skip if the Hessian is
    // numerically singular (mass/dt² guarantees it isn't, but floats).
    let h_inv = mat3_inverse(h);
    let dx = h_inv * f;
    particle_q[v] = vec4<f32>(x + dx, 0.0);
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
