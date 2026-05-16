// XPBD compute kernels — predict, position projection, velocity
// recovery, velocity projection. Matches the CPU implementation in
// `src/solvers/xpbd.rs` and Newton's `solve_body_contact_positions` +
// `apply_body_deltas`.
//
// **Per-body Jacobi**: each thread owns one body and accumulates
// linear+angular impulse contributions from every contact involving
// its body, then averages by the per-body contact count. This avoids
// the WGSL `atomic<f32>` gap and is more stable than Gauss-Seidel for
// stacks (boxes especially).
//
// Off-COM contacts produce angular response via the standard
// generalised inverse mass formulation:
//     w = 1/m + (r × n)ᵀ · I⁻¹_world · (r × n)
//     Δposition_lin = -n · λ · 1/m
//     Δrotation     = -I⁻¹_world · (r × n) · λ
// with `λ = current_penetration / w_total` and per-body averaging
// `dp = sum(impulse_lin) / count(contacts) / mass`.
//
// Iteration coupling: the host runs N position iterations by
// dispatching `project_position` N times with ping-pong buffers.

struct Transform {
    position: vec4<f32>, // xyz + pad
    rotation: vec4<f32>, // quat (x,y,z,w)
};

struct SpatialVec {
    linear:  vec4<f32>, // xyz + pad
    angular: vec4<f32>, // xyz + pad
};

struct Mat3Cols {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
};

struct Uniforms {
    dt:                 f32,
    angular_damping:    f32,
    body_count:         u32,
    has_external_force: u32,
    linear_damping:     f32,
    _pad0:              f32,
    _pad1:              f32,
    _pad2:              f32,
};

const FLAG_KINEMATIC: u32 = 1u;
const STATIC_BODY:    u32 = 0xffffffffu;
const B_SIDE_BIT:     u32 = 0x80000000u;

// ── Group 0 — Model (read-only) ──────────────────────────────────────
@group(0) @binding(0) var<storage, read> body_inv_mass:    array<f32>;
@group(0) @binding(1) var<storage, read> body_flags:       array<u32>;
@group(0) @binding(2) var<storage, read> body_world:       array<i32>;
@group(0) @binding(3) var<storage, read> gravity:          array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> body_com:         array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> body_inv_inertia: array<Mat3Cols>;

// ── Group 1 — State input + ping-pong working buffers ───────────────
@group(1) @binding(0) var<storage, read>       body_q_in:    array<Transform>;
@group(1) @binding(1) var<storage, read>       body_qd_in:   array<SpatialVec>;
@group(1) @binding(2) var<storage, read>       body_f_in:    array<SpatialVec>;
@group(1) @binding(3) var<storage, read_write> body_q_pred:  array<Transform>;
@group(1) @binding(4) var<storage, read_write> body_v_pred:  array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> body_q_out:   array<Transform>; // ping
@group(1) @binding(6) var<storage, read_write> body_qd_out:  array<SpatialVec>;
@group(1) @binding(7) var<storage, read_write> body_q_alt:   array<Transform>; // pong
@group(1) @binding(8) var<storage, read_write> body_v_post:  array<vec4<f32>>;
@group(1) @binding(9) var<storage, read_write> body_w_pred:  array<vec4<f32>>;
@group(1) @binding(10) var<storage, read_write> body_w_post: array<vec4<f32>>;

// ── Group 2 — Control (read-only) ────────────────────────────────────
@group(2) @binding(0) var<storage, read> body_external_f: array<SpatialVec>;

// ── Group 3 — Contacts + per-body CSR ────────────────────────────────
@group(3) @binding(0) var<storage, read> contact_a:        array<u32>;
@group(3) @binding(1) var<storage, read> contact_b:        array<u32>;
@group(3) @binding(2) var<storage, read> contact_normal:   array<vec4<f32>>;
@group(3) @binding(3) var<storage, read> contact_depth:    array<f32>;
@group(3) @binding(4) var<storage, read> contact_friction: array<f32>;
@group(3) @binding(5) var<storage, read> contact_restitution: array<f32>;
@group(3) @binding(6) var<storage, read> body_contact_offsets: array<u32>;
@group(3) @binding(7) var<storage, read> body_contact_indices: array<u32>;
@group(3) @binding(8) var<storage, read> contact_point:    array<vec4<f32>>;
@group(3) @binding(9) var<storage, read> contact_rolling_friction:   array<f32>;
@group(3) @binding(10) var<storage, read> contact_torsional_friction: array<f32>;

// ── Group 4 — Uniforms ───────────────────────────────────────────────
@group(4) @binding(0) var<uniform> uniforms: Uniforms;

// ─── Quaternion / matrix helpers ────────────────────────────────────

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    );
}
fn quat_conj(q: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(-q.x, -q.y, -q.z, q.w);
}
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s*s - dot(u,u)) * v + 2.0 * s * cross(u, v);
}
fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    let n = length(q);
    if (n < 1e-12) { return vec4<f32>(0.0, 0.0, 0.0, 1.0); }
    return q / n;
}
fn integrate_rotation(q: vec4<f32>, omega: vec3<f32>, dt: f32) -> vec4<f32> {
    let dq = vec4<f32>(omega * 0.5 * dt, 0.0);
    return quat_normalize(quat_mul(dq, q) + q);
}
fn apply_rotation_delta(q: vec4<f32>, dtheta: vec3<f32>) -> vec4<f32> {
    let dq = vec4<f32>(dtheta * 0.5, 0.0);
    return quat_normalize(quat_mul(dq, q) + q);
}
fn quat_to_omega(dq: vec4<f32>) -> vec3<f32> {
    let s = select(1.0, -1.0, dq.w < 0.0);
    return vec3<f32>(dq.x, dq.y, dq.z) * (2.0 * s);
}
fn mat3_from_cols(m: Mat3Cols) -> mat3x3<f32> {
    return mat3x3<f32>(m.col0.xyz, m.col1.xyz, m.col2.xyz);
}
fn rotate_inertia_inv(q: vec4<f32>, i_inv_body: mat3x3<f32>) -> mat3x3<f32> {
    // R · I_inv_body · R^T. Build R as col vectors of rotated basis.
    let cx = quat_rotate(q, vec3<f32>(1.0, 0.0, 0.0));
    let cy = quat_rotate(q, vec3<f32>(0.0, 1.0, 0.0));
    let cz = quat_rotate(q, vec3<f32>(0.0, 0.0, 1.0));
    let r = mat3x3<f32>(cx, cy, cz);
    return r * i_inv_body * transpose(r);
}
fn com_world(t: Transform, com: vec3<f32>) -> vec3<f32> {
    return t.position.xyz + quat_rotate(t.rotation, com);
}

// ─────────────────────────────────────────────────────────────────────
// 1. PREDICT — semi-implicit Euler (linear AND angular).
@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.body_count) { return; }

    if ((body_flags[tid] & FLAG_KINEMATIC) != 0u) {
        body_q_pred[tid] = body_q_in[tid];
        body_v_pred[tid] = vec4<f32>(0.0);
        body_w_pred[tid] = vec4<f32>(0.0);
        return;
    }

    let inv_m = body_inv_mass[tid];
    let world = max(body_world[tid], 0);
    let g = select(vec3<f32>(0.0), gravity[u32(world)].xyz, inv_m > 0.0);

    var applied_lin = body_f_in[tid].linear.xyz;
    var applied_ang = body_f_in[tid].angular.xyz;
    if (uniforms.has_external_force == 1u) {
        applied_lin = applied_lin + body_external_f[tid].linear.xyz;
        applied_ang = applied_ang + body_external_f[tid].angular.xyz;
    }

    let v0 = body_qd_in[tid].linear.xyz;
    let w0 = body_qd_in[tid].angular.xyz;
    var v1 = v0 + (applied_lin * inv_m + g) * uniforms.dt;
    // Linear damping: v *= max(0, 1 - λ·dt). Models air drag /
    // rolling resistance crudely; small-dt approximation of exp().
    let lin_damp_factor = max(0.0, 1.0 - uniforms.linear_damping * uniforms.dt);
    v1 = v1 * lin_damp_factor;
    let p0 = body_q_in[tid].position.xyz;
    let p1 = p0 + v1 * uniforms.dt;

    let i_inv = rotate_inertia_inv(body_q_in[tid].rotation, mat3_from_cols(body_inv_inertia[tid]));
    let dw = i_inv * applied_ang * uniforms.dt;
    let w1 = w0 + dw;
    let q1 = integrate_rotation(body_q_in[tid].rotation, w1, uniforms.dt);

    var t: Transform;
    t.position = vec4<f32>(p1, 0.0);
    t.rotation = q1;
    body_q_pred[tid] = t;
    body_v_pred[tid] = vec4<f32>(v1, 0.0);
    body_w_pred[tid] = vec4<f32>(w1, 0.0);
    body_q_out[tid]  = t;
}

// ─────────────────────────────────────────────────────────────────────
// 2. POSITION PROJECTION — single iteration, Jacobi.
//    Reads body_q_out (ping), writes body_q_alt (pong). Host swaps.
@compute @workgroup_size(64)
fn project_position(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.body_count) { return; }

    if ((body_flags[tid] & FLAG_KINEMATIC) != 0u) {
        body_q_alt[tid] = body_q_out[tid];
        return;
    }

    let inv_m_self = body_inv_mass[tid];
    let q_self     = body_q_out[tid];
    let q_self_in  = body_q_in[tid];
    let com_self   = body_com[tid].xyz;
    let i_inv_self = rotate_inertia_inv(q_self.rotation, mat3_from_cols(body_inv_inertia[tid]));

    var pos_self = q_self.position.xyz;
    var rot_self = q_self.rotation;

    var lin_acc = vec3<f32>(0.0);
    var ang_acc = vec3<f32>(0.0);
    var weight  = 0.0;

    let begin = body_contact_offsets[tid];
    let end   = body_contact_offsets[tid + 1u];

    for (var i = begin; i < end; i = i + 1u) {
        let packed = body_contact_indices[i];
        let c = packed & ~B_SIDE_BIT;
        let is_b = (packed & B_SIDE_BIT) != 0u;

        let n_ab = contact_normal[c].xyz;
        let initial = contact_depth[c];
        let p_w = contact_point[c].xyz;

        let a = contact_a[c];
        let b = contact_b[c];
        let other = select(b, a, is_b);
        let other_kin = (other == STATIC_BODY) ||
            ((body_flags[other] & FLAG_KINEMATIC) != 0u);

        // Fetch other-body data with proper static-body fallbacks.
        var inv_m_other = 0.0;
        var q_other_pos = p_w;
        var q_other_rot = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        var q_other_in_pos = p_w;
        var q_other_in_rot = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        var com_other = vec3<f32>(0.0);
        var i_inv_other = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
        if (!other_kin) {
            inv_m_other = body_inv_mass[other];
            q_other_pos = body_q_out[other].position.xyz;
            q_other_rot = body_q_out[other].rotation;
            q_other_in_pos = body_q_in[other].position.xyz;
            q_other_in_rot = body_q_in[other].rotation;
            com_other = body_com[other].xyz;
            i_inv_other = rotate_inertia_inv(q_other_rot, mat3_from_cols(body_inv_inertia[other]));
        }

        // body-local contact points (frozen at narrow-phase).
        let p_self_local = quat_rotate(quat_conj(q_self_in.rotation),
                                       p_w - q_self_in.position.xyz);
        let p_self_now  = pos_self + quat_rotate(rot_self, p_self_local);

        var p_other_now = p_w;
        if (!other_kin) {
            let p_other_local = quat_rotate(quat_conj(q_other_in_rot),
                                            p_w - q_other_in_pos);
            p_other_now = q_other_pos + quat_rotate(q_other_rot, p_other_local);
        }

        let r_self  = p_self_now  - (pos_self + quat_rotate(rot_self, com_self));
        var r_other = vec3<f32>(0.0);
        if (!other_kin) {
            r_other = p_other_now - (q_other_pos + quat_rotate(q_other_rot, com_other));
        }

        // a → b convention.
        var r_a = r_self;
        var r_b = r_other;
        var p_a_now = p_self_now;
        var p_b_now = p_other_now;
        var inv_m_a = inv_m_self;
        var inv_m_b = inv_m_other;
        var i_inv_a = i_inv_self;
        var i_inv_b = i_inv_other;
        if (is_b) {
            r_a = r_other;
            r_b = r_self;
            p_a_now = p_other_now;
            p_b_now = p_self_now;
            inv_m_a = inv_m_other;
            inv_m_b = inv_m_self;
            i_inv_a = i_inv_other;
            i_inv_b = i_inv_self;
        }

        let ra_x_n = cross(r_a, n_ab);
        let rb_x_n = cross(r_b, n_ab);
        let denom = inv_m_a + inv_m_b
                  + dot(ra_x_n, i_inv_a * ra_x_n)
                  + dot(rb_x_n, i_inv_b * rb_x_n);
        if (denom <= 0.0) { continue; }

        let current = initial + dot(p_a_now - p_b_now, n_ab);
        if (current <= 0.0) { continue; }

        let lambda = current / denom;

        let lin_imp = select(-n_ab, n_ab, is_b) * lambda;
        let ang_imp = select(-ra_x_n, rb_x_n, is_b) * lambda;
        lin_acc = lin_acc + lin_imp;
        ang_acc = ang_acc + ang_imp;
        weight = weight + 1.0;
    }

    if (weight > 0.0 && inv_m_self > 0.0) {
        let inv_w = 1.0 / weight;
        let dp = lin_acc * (inv_m_self * inv_w);
        let dtheta = i_inv_self * ang_acc * inv_w;
        pos_self = pos_self + dp;
        rot_self = apply_rotation_delta(rot_self, dtheta);
    }

    var t: Transform;
    t.position = vec4<f32>(pos_self, 0.0);
    t.rotation = rot_self;
    body_q_alt[tid] = t;
}

// ─────────────────────────────────────────────────────────────────────
// 3. VELOCITY RECOVERY — linear from pos delta, angular from rot delta.
@compute @workgroup_size(64)
fn recover_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.body_count) { return; }
    if ((body_flags[tid] & FLAG_KINEMATIC) != 0u) {
        body_v_post[tid] = vec4<f32>(0.0);
        body_w_post[tid] = vec4<f32>(0.0);
        return;
    }
    let dp = body_q_out[tid].position.xyz - body_q_in[tid].position.xyz;
    body_v_post[tid] = vec4<f32>(dp / uniforms.dt, 0.0);
    let dq = quat_mul(body_q_out[tid].rotation, quat_conj(body_q_in[tid].rotation));
    body_w_post[tid] = vec4<f32>(quat_to_omega(dq) / uniforms.dt, 0.0);
}

// ─────────────────────────────────────────────────────────────────────
// 4. VELOCITY PROJECTION — Jacobi over contacts (linear + angular).
@compute @workgroup_size(64)
fn project_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.body_count) { return; }
    if ((body_flags[tid] & FLAG_KINEMATIC) != 0u) {
        return;
    }
    let inv_m_self = body_inv_mass[tid];
    let q_self_pred = body_q_out[tid];
    let q_self_in   = body_q_in[tid];
    let com_self    = body_com[tid].xyz;
    let i_inv_self  = rotate_inertia_inv(q_self_pred.rotation, mat3_from_cols(body_inv_inertia[tid]));
    let v_self_pred = body_v_pred[tid].xyz;
    let w_self_pred = body_w_pred[tid].xyz;

    var v_self = body_v_post[tid].xyz;
    var w_self = body_w_post[tid].xyz;

    var lin_acc = vec3<f32>(0.0);
    var ang_acc = vec3<f32>(0.0);
    var weight  = 0.0;

    let begin = body_contact_offsets[tid];
    let end   = body_contact_offsets[tid + 1u];

    for (var i = begin; i < end; i = i + 1u) {
        let packed = body_contact_indices[i];
        let c = packed & ~B_SIDE_BIT;
        let is_b = (packed & B_SIDE_BIT) != 0u;

        let n_ab = contact_normal[c].xyz;
        let p_w  = contact_point[c].xyz;
        let e    = contact_restitution[c];
        let mu   = contact_friction[c];

        let a = contact_a[c];
        let b = contact_b[c];
        let other = select(b, a, is_b);
        let other_kin = (other == STATIC_BODY) ||
            ((body_flags[other] & FLAG_KINEMATIC) != 0u);

        var inv_m_other = 0.0;
        var q_other_pos = p_w;
        var q_other_rot = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        var q_other_in_pos = p_w;
        var q_other_in_rot = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        var com_other = vec3<f32>(0.0);
        var i_inv_other = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
        var v_other_pre = vec3<f32>(0.0);
        var v_other_post = vec3<f32>(0.0);
        if (!other_kin) {
            inv_m_other = body_inv_mass[other];
            q_other_pos = body_q_out[other].position.xyz;
            q_other_rot = body_q_out[other].rotation;
            q_other_in_pos = body_q_in[other].position.xyz;
            q_other_in_rot = body_q_in[other].rotation;
            com_other = body_com[other].xyz;
            i_inv_other = rotate_inertia_inv(q_other_rot, mat3_from_cols(body_inv_inertia[other]));
        }

        let p_self_local = quat_rotate(quat_conj(q_self_in.rotation),
                                       p_w - q_self_in.position.xyz);
        let p_self_now  = q_self_pred.position.xyz +
                          quat_rotate(q_self_pred.rotation, p_self_local);
        var p_other_now = p_w;
        if (!other_kin) {
            let p_other_local = quat_rotate(quat_conj(q_other_in_rot),
                                            p_w - q_other_in_pos);
            p_other_now = q_other_pos + quat_rotate(q_other_rot, p_other_local);
        }

        let r_self  = p_self_now  - (q_self_pred.position.xyz +
                                     quat_rotate(q_self_pred.rotation, com_self));
        var r_other = vec3<f32>(0.0);
        if (!other_kin) {
            r_other = p_other_now - (q_other_pos + quat_rotate(q_other_rot, com_other));
            v_other_pre = body_v_pred[other].xyz + cross(body_w_pred[other].xyz, r_other);
            v_other_post = body_v_post[other].xyz + cross(body_w_post[other].xyz, r_other);
        }

        var r_a = r_self;
        var r_b = r_other;
        var inv_m_a = inv_m_self;
        var inv_m_b = inv_m_other;
        var i_inv_a = i_inv_self;
        var i_inv_b = i_inv_other;
        if (is_b) {
            r_a = r_other;
            r_b = r_self;
            inv_m_a = inv_m_other;
            inv_m_b = inv_m_self;
            i_inv_a = i_inv_other;
            i_inv_b = i_inv_self;
        }

        let v_self_pre = v_self_pred + cross(w_self_pred, r_self);
        var v_a_pre = v_self_pre;
        var v_b_pre = v_other_pre;
        if (is_b) { v_a_pre = v_other_pre; v_b_pre = v_self_pre; }
        let v_n_pre = dot(v_b_pre - v_a_pre, n_ab);
        if (v_n_pre >= 0.0) { continue; }

        let v_self_now = v_self + cross(w_self, r_self);
        var v_a = v_self_now;
        var v_b = v_other_post;
        if (is_b) { v_a = v_other_post; v_b = v_self_now; }
        let v_rel = v_b - v_a;
        let v_n = dot(v_rel, n_ab);
        let v_n_target = -e * v_n_pre;

        let ra_x_n = cross(r_a, n_ab);
        let rb_x_n = cross(r_b, n_ab);
        let denom = inv_m_a + inv_m_b
                  + dot(ra_x_n, i_inv_a * ra_x_n)
                  + dot(rb_x_n, i_inv_b * rb_x_n);
        if (denom <= 0.0) { continue; }

        let lambda_n = (v_n_target - v_n) / denom;
        let lin_imp_self = select(-n_ab, n_ab, is_b) * lambda_n;
        let ang_imp_self = select(-ra_x_n, rb_x_n, is_b) * lambda_n;
        lin_acc = lin_acc + lin_imp_self;
        ang_acc = ang_acc + ang_imp_self;
        weight  = weight + 1.0;

        if (mu > 0.0) {
            let v_t = v_rel - n_ab * v_n;
            let v_t_len = length(v_t);
            if (v_t_len > 1e-9) {
                let t_hat = v_t / v_t_len;
                let ra_x_t = cross(r_a, t_hat);
                let rb_x_t = cross(r_b, t_hat);
                let denom_t = inv_m_a + inv_m_b
                            + dot(ra_x_t, i_inv_a * ra_x_t)
                            + dot(rb_x_t, i_inv_b * rb_x_t);
                if (denom_t > 0.0) {
                    // Coulomb cone caps the friction *impulse* λ_t at
                    // μ|λ_n|. λ_t = dv_t/denom_t, so dv_t must be
                    // capped at μ|λ_n|·denom_t (matches CPU xpbd.rs) —
                    // capping dv_t at μ|λ_n| alone is wrong by denom_t.
                    let cap = mu * abs(lambda_n);
                    let dv_t = min(v_t_len, max(cap, 0.0) * denom_t);
                    let lambda_t = dv_t / denom_t;
                    // self impulses: if self is a, +t_hat ; if self is b, -t_hat.
                    let lin_t_self = select(t_hat, -t_hat, is_b) * lambda_t;
                    let ang_t_self = select(ra_x_t, -rb_x_t, is_b) * lambda_t;
                    lin_acc = lin_acc + lin_t_self;
                    ang_acc = ang_acc + ang_t_self;
                }
            }
        }

        // ── Rolling friction: damp angular velocity perpendicular ──
        // to the contact normal. Cap = μ_rolling · |λ_n|. Per-body
        // Jacobi: each thread accumulates only its own ang_acc.
        let mu_roll = contact_rolling_friction[c];
        if (mu_roll > 0.0) {
            // w_a / w_b in global terms (kinematic = 0 for missing body).
            var w_a_g = w_self;
            var w_b_g = vec3<f32>(0.0);
            if (!other_kin) {
                w_b_g = body_w_post[other].xyz;
            }
            if (is_b) {
                let tmp = w_a_g; w_a_g = w_b_g; w_b_g = tmp;
            }
            let w_rel = w_b_g - w_a_g;
            let w_n = dot(w_rel, n_ab);
            let w_perp = w_rel - n_ab * w_n;
            let w_perp_len = length(w_perp);
            if (w_perp_len > 1e-9) {
                let dir = w_perp / w_perp_len;
                let denom_r = dot(dir, i_inv_a * dir) + dot(dir, i_inv_b * dir);
                if (denom_r > 0.0) {
                    let cap_r = mu_roll * abs(lambda_n);
                    // Cap the rolling-friction impulse at μ_roll|λ_n|
                    // (= dw capped at cap_r·denom_r), matching CPU.
                    let dw = min(w_perp_len, max(cap_r, 0.0) * denom_r);
                    let lambda_r = dw / denom_r;
                    // Self gets +dir if self == a, -dir if self == b
                    // (relative velocity sign flips).
                    let ang_r_self = select(dir, -dir, is_b) * lambda_r;
                    ang_acc = ang_acc + ang_r_self;
                }
            }
        }

        // ── Torsional friction: damp angular velocity component ──
        // ALONG the normal. Cap = μ_torsional · |λ_n|.
        let mu_tors = contact_torsional_friction[c];
        if (mu_tors > 0.0) {
            var w_a_g = w_self;
            var w_b_g = vec3<f32>(0.0);
            if (!other_kin) {
                w_b_g = body_w_post[other].xyz;
            }
            if (is_b) {
                let tmp = w_a_g; w_a_g = w_b_g; w_b_g = tmp;
            }
            let w_rel = w_b_g - w_a_g;
            let w_n = dot(w_rel, n_ab);
            let w_n_abs = abs(w_n);
            if (w_n_abs > 1e-9) {
                let dir = n_ab * sign(w_n);
                let denom_t = dot(dir, i_inv_a * dir) + dot(dir, i_inv_b * dir);
                if (denom_t > 0.0) {
                    let cap_t = mu_tors * abs(lambda_n);
                    // Cap the torsional-friction impulse at μ_tors|λ_n|
                    // (= dw capped at cap_t·denom_t), matching CPU.
                    let dw = min(w_n_abs, max(cap_t, 0.0) * denom_t);
                    let lambda_t = dw / denom_t;
                    let ang_t_self = select(dir, -dir, is_b) * lambda_t;
                    ang_acc = ang_acc + ang_t_self;
                }
            }
        }
    }

    if (weight > 0.0 && inv_m_self > 0.0) {
        let inv_w = 1.0 / weight;
        v_self = v_self + lin_acc * (inv_m_self * inv_w);
        w_self = w_self + i_inv_self * ang_acc * inv_w;
    }

    body_v_post[tid] = vec4<f32>(v_self, 0.0);
    body_w_post[tid] = vec4<f32>(w_self, 0.0);
}

// ─────────────────────────────────────────────────────────────────────
// 5. WRITE FINAL — copy projected pose + final v/ω into outputs.
@compute @workgroup_size(64)
fn write_final(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.body_count) { return; }

    if ((body_flags[tid] & FLAG_KINEMATIC) != 0u) {
        body_q_out[tid]  = body_q_in[tid];
        body_qd_out[tid] = body_qd_in[tid];
        return;
    }

    var qd: SpatialVec;
    qd.linear  = vec4<f32>(body_v_post[tid].xyz, 0.0);
    qd.angular = vec4<f32>(body_w_post[tid].xyz, 0.0);
    body_qd_out[tid] = qd;
}
