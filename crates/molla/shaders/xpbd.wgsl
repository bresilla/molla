// XPBD compute kernels — predict, position projection, velocity
// recovery, velocity projection. Mirrors the CPU implementation in
// `src/solvers/xpbd.rs` exactly; see that file for the algorithmic
// reference and constraint derivations.
//
// Per-body parallelism: each thread owns one body and serially
// processes only the contacts that involve its body (via the per-body
// CSR built in `geometry::pipeline::build_body_contact_csr`). This
// avoids cross-thread races and the WGSL `atomic<f32>` gap.
//
// Iteration coupling: position projection iterates internally on the
// CPU side because each projection iteration ping-pongs through a
// pair of `q_pred` buffers. The WGSL kernel processes one iteration
// per dispatch.

struct Transform {
    position: vec4<f32>, // xyz + pad
    rotation: vec4<f32>, // quat (x,y,z,w)
};

struct SpatialVec {
    linear:  vec4<f32>, // xyz + pad
    angular: vec4<f32>, // xyz + pad
};

struct Uniforms {
    dt:                 f32,
    angular_damping:    f32,
    body_count:         u32,
    has_external_force: u32,
};

const FLAG_KINEMATIC: u32 = 1u;
const STATIC_BODY:    u32 = 0xffffffffu;
const B_SIDE_BIT:     u32 = 0x80000000u;

// ── Group 0 — Model (read-only) ──────────────────────────────────────
@group(0) @binding(0) var<storage, read> body_inv_mass:  array<f32>;
@group(0) @binding(1) var<storage, read> body_flags:     array<u32>;
@group(0) @binding(2) var<storage, read> body_world:     array<i32>;
@group(0) @binding(3) var<storage, read> gravity:        array<vec4<f32>>;

// ── Group 1 — State input + ping-pong working buffers ───────────────
@group(1) @binding(0) var<storage, read>       body_q_in:    array<Transform>;
@group(1) @binding(1) var<storage, read>       body_qd_in:   array<SpatialVec>;
@group(1) @binding(2) var<storage, read>       body_f_in:    array<SpatialVec>;
@group(1) @binding(3) var<storage, read_write> body_q_pred:  array<Transform>;
@group(1) @binding(4) var<storage, read_write> body_v_pred:  array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> body_q_out:   array<Transform>; // ping
@group(1) @binding(6) var<storage, read_write> body_qd_out:  array<SpatialVec>;
@group(1) @binding(7) var<storage, read_write> body_q_alt:   array<Transform>; // pong (also reused for v_post)
@group(1) @binding(8) var<storage, read_write> body_v_post:  array<vec4<f32>>;

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

// ── Group 4 — Uniforms ───────────────────────────────────────────────
@group(4) @binding(0) var<uniform> uniforms: Uniforms;

// ─────────────────────────────────────────────────────────────────────
// 1. PREDICT — semi-implicit Euler. Writes body_q_pred + body_v_pred.
@compute @workgroup_size(64)
fn predict(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.body_count) { return; }

    if ((body_flags[tid] & FLAG_KINEMATIC) != 0u) {
        body_q_pred[tid] = body_q_in[tid];
        body_v_pred[tid] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
        return;
    }

    let inv_m = body_inv_mass[tid];
    let world = max(body_world[tid], 0);
    let g = select(vec3<f32>(0.0), gravity[u32(world)].xyz, inv_m > 0.0);

    var applied = body_f_in[tid].linear.xyz;
    if (uniforms.has_external_force == 1u) {
        applied = applied + body_external_f[tid].linear.xyz;
    }

    let v0 = body_qd_in[tid].linear.xyz;
    let v1 = v0 + (applied * inv_m + g) * uniforms.dt;
    let p0 = body_q_in[tid].position.xyz;
    let p1 = p0 + v1 * uniforms.dt;

    var t: Transform;
    t.position = vec4<f32>(p1, 0.0);
    t.rotation = body_q_in[tid].rotation;
    body_q_pred[tid] = t;
    body_v_pred[tid] = vec4<f32>(v1, 0.0);
    body_q_out[tid] = t;  // initial ping value
}

// ─────────────────────────────────────────────────────────────────────
// 2. POSITION PROJECTION — single iteration. Reads body_q_out (ping),
// writes body_q_alt (pong). Host swaps the two on each invocation.
@compute @workgroup_size(64)
fn project_position(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.body_count) { return; }

    if ((body_flags[tid] & FLAG_KINEMATIC) != 0u) {
        body_q_alt[tid] = body_q_out[tid];
        return;
    }

    let inv_m_self = body_inv_mass[tid];
    var pos_self = body_q_out[tid].position.xyz;
    let p0 = body_q_in[tid].position.xyz;

    let begin = body_contact_offsets[tid];
    let end   = body_contact_offsets[tid + 1u];

    for (var i = begin; i < end; i = i + 1u) {
        let packed = body_contact_indices[i];
        let c = packed & ~B_SIDE_BIT;
        let is_b = (packed & B_SIDE_BIT) != 0u;

        let n_ab = contact_normal[c].xyz;
        let initial = contact_depth[c];

        let a = contact_a[c];
        let b = contact_b[c];
        let other = select(b, a, is_b);
        let other_kin = (other == STATIC_BODY) ||
            ((body_flags[other] & FLAG_KINEMATIC) != 0u);
        let other_inv_m = select(body_inv_mass[other], 0.0, other_kin);

        let w_sum = inv_m_self + other_inv_m;
        if (w_sum <= 0.0) { continue; }

        // Δq for both bodies, w.r.t. their input positions.
        let delta_self = pos_self - p0;
        var delta_other = vec3<f32>(0.0);
        if (!other_kin) {
            // Read from the SAME ping buffer the host had us reading
            // — body_q_out — for symmetric Jacobi-style relaxation.
            delta_other = body_q_out[other].position.xyz - body_q_in[other].position.xyz;
        }

        // Convention: contact_normal points a → b.
        let delta_a = select(delta_other, delta_self, !is_b);
        let delta_b = select(delta_self, delta_other, !is_b);

        let current = initial + dot(delta_a - delta_b, n_ab);
        if (current <= 0.0) { continue; }

        let lambda = current / w_sum;
        // Self gets pushed AWAY from the other along the contact axis.
        // If self is `a`, push along -n_ab. If self is `b`, push along +n_ab.
        let dir = select(-n_ab, n_ab, is_b);
        pos_self = pos_self + dir * (lambda * inv_m_self);
    }

    var t: Transform;
    t.position = vec4<f32>(pos_self, 0.0);
    t.rotation = body_q_out[tid].rotation;
    body_q_alt[tid] = t;
}

// ─────────────────────────────────────────────────────────────────────
// 3. VELOCITY RECOVERY — v_post = (q_pred_final - q_in) / dt.
// Reads body_q_out (final ping after position projection).
@compute @workgroup_size(64)
fn recover_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.body_count) { return; }
    if ((body_flags[tid] & FLAG_KINEMATIC) != 0u) {
        body_v_post[tid] = vec4<f32>(0.0);
        return;
    }
    let dp = body_q_out[tid].position.xyz - body_q_in[tid].position.xyz;
    body_v_post[tid] = vec4<f32>(dp / uniforms.dt, 0.0);
}

// ─────────────────────────────────────────────────────────────────────
// 4. VELOCITY PROJECTION — single iteration. Per-body, processes
// contacts via CSR; uses pre-projection v_pred for the bounce target
// so position-correction-induced separation isn't read as a real bounce.
@compute @workgroup_size(64)
fn project_velocity(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.body_count) { return; }
    if ((body_flags[tid] & FLAG_KINEMATIC) != 0u) {
        return;
    }
    let inv_m_self = body_inv_mass[tid];
    var v_self = body_v_post[tid].xyz;
    let v_pre_self = body_v_pred[tid].xyz;

    let begin = body_contact_offsets[tid];
    let end   = body_contact_offsets[tid + 1u];

    for (var i = begin; i < end; i = i + 1u) {
        let packed = body_contact_indices[i];
        let c = packed & ~B_SIDE_BIT;
        let is_b = (packed & B_SIDE_BIT) != 0u;

        let n_ab = contact_normal[c].xyz;
        let e = contact_restitution[c];
        let mu = contact_friction[c];

        let a = contact_a[c];
        let b = contact_b[c];
        let other = select(b, a, is_b);
        let other_kin = (other == STATIC_BODY) ||
            ((body_flags[other] & FLAG_KINEMATIC) != 0u);
        let other_inv_m = select(body_inv_mass[other], 0.0, other_kin);
        let w_sum = inv_m_self + other_inv_m;
        if (w_sum <= 0.0) { continue; }

        // Pre-projection rel velocity.
        var v_other_pre = vec3<f32>(0.0);
        if (!other_kin) { v_other_pre = body_v_pred[other].xyz; }
        let v_a_pre = select(v_other_pre, v_pre_self, !is_b);
        let v_b_pre = select(v_pre_self, v_other_pre, !is_b);
        let v_rel_pre = v_b_pre - v_a_pre;
        let v_n_pre = dot(v_rel_pre, n_ab);
        if (v_n_pre >= 0.0) { continue; } // already separating

        let v_n_target = -e * v_n_pre;

        // Current (post-projection) rel velocity.
        var v_other = vec3<f32>(0.0);
        if (!other_kin) { v_other = body_v_post[other].xyz; }
        let v_a = select(v_other, v_self, !is_b);
        let v_b = select(v_self, v_other, !is_b);
        let v_rel = v_b - v_a;
        let v_n = dot(v_rel, n_ab);

        let dv_n = v_n_target - v_n;
        let lambda_n = dv_n / w_sum;
        let n_dir = select(-n_ab, n_ab, is_b);
        v_self = v_self + n_dir * (lambda_n * inv_m_self);

        if (mu > 0.0) {
            // Re-read post-projection rel velocity since v_self changed.
            let v_a2 = select(v_other, v_self, !is_b);
            let v_b2 = select(v_self, v_other, !is_b);
            let v_rel2 = v_b2 - v_a2;
            let v_n2 = dot(v_rel2, n_ab);
            let v_t = v_rel2 - n_ab * v_n2;
            let v_t_len = length(v_t);
            if (v_t_len > 1e-9) {
                let t_hat = v_t / v_t_len;
                let cap = mu * abs(lambda_n);
                let dv_t = min(v_t_len, max(cap, 0.0));
                let lambda_t = dv_t / w_sum;
                let t_dir = select(t_hat, -t_hat, !is_b);
                v_self = v_self + t_dir * (lambda_t * inv_m_self);
            }
        }
    }

    body_v_post[tid] = vec4<f32>(v_self, 0.0);
}

// ─────────────────────────────────────────────────────────────────────
// 5. WRITE FINAL — copies q_pred (in body_q_out) + v_post into the
// official body_q_out / body_qd_out output slots, preserving angular.
@compute @workgroup_size(64)
fn write_final(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= uniforms.body_count) { return; }

    if ((body_flags[tid] & FLAG_KINEMATIC) != 0u) {
        body_q_out[tid]  = body_q_in[tid];
        body_qd_out[tid] = body_qd_in[tid];
        return;
    }

    // body_q_out already holds the projected pose (last position
    // projection ping wrote it there). Just write velocities.
    var qd: SpatialVec;
    qd.linear  = vec4<f32>(body_v_post[tid].xyz, 0.0);
    qd.angular = body_qd_in[tid].angular;
    body_qd_out[tid] = qd;
}
