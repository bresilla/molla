// Featherstone forward dynamics on GPU — full Articulated Body
// Algorithm (ABA) per Featherstone (2008) ch. 7. Mirrors the CPU
// path in `solvers/featherstone.rs::cpu_step` and the structure of
// Newton's `solvers/featherstone/kernels.py` (with spatial-vector
// primitives ported from Warp's `_src/math.py` / `builtins.py`).
//
// Per-articulation parallelism: one workgroup invocation per chain.
// Each thread runs the three ABA passes (forward kinematics +
// articulated-inertia accumulation backward + acceleration solve
// forward) over its articulation's joint range, then integrates
// q,q̇ via semi-implicit Euler.
//
// The articulated inertia is a 6×6 matrix per body — we store it
// inline as 36 floats. For molla's stage-3 demos there's typically
// one articulation per scene (one robot), so the per-thread state
// fits in registers / local memory comfortably.

struct Transform {
    position: vec4<f32>, // xyz + pad
    rotation: vec4<f32>, // (qx, qy, qz, qw)
};

struct Mat3Cols {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
};

struct Uniforms {
    dt:                 f32,
    joint_damping:      f32,
    joint_count:        u32,
    articulation_count: u32,
};

const JOINT_PRISMATIC: u32 = 0u;
const JOINT_REVOLUTE:  u32 = 1u;
const JOINT_FIXED:     u32 = 3u;
const JOINT_FREE:      u32 = 4u;

const MAX_JOINTS_PER_ART: u32 = 64u;

// Group 0 — Model
@group(0) @binding(0) var<storage, read> joint_type:         array<u32>;
@group(0) @binding(1) var<storage, read> joint_parent:       array<i32>;
@group(0) @binding(2) var<storage, read> joint_child:        array<i32>;
@group(0) @binding(3) var<storage, read> joint_x_p:          array<Transform>;
@group(0) @binding(4) var<storage, read> joint_x_c:          array<Transform>;
@group(0) @binding(5) var<storage, read> joint_axis:         array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> joint_coord_offset: array<u32>;
@group(0) @binding(7) var<storage, read> joint_dof_offset:   array<u32>;
// Per-DOF limits / drives. NaN sentinel = unlimited (signalled
// by `isFinite(...) == false`). Length = joint_dof_total.
@group(0) @binding(8) var<storage, read> joint_limit_lower:    array<f32>;
@group(0) @binding(9) var<storage, read> joint_limit_upper:    array<f32>;
@group(0) @binding(10) var<storage, read> joint_friction:      array<f32>;
@group(0) @binding(11) var<storage, read> joint_effort_limit:  array<f32>;
@group(0) @binding(12) var<storage, read> joint_velocity_limit: array<f32>;

// Group 1 — Body data + State
@group(1) @binding(0) var<storage, read>       body_mass:    array<f32>;
@group(1) @binding(1) var<storage, read>       body_inertia: array<Mat3Cols>;
@group(1) @binding(2) var<storage, read>       body_com:     array<vec4<f32>>;
@group(1) @binding(3) var<storage, read>       body_world:   array<i32>;
@group(1) @binding(4) var<storage, read>       gravity:      array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> body_q:       array<Transform>;
@group(1) @binding(6) var<storage, read_write> joint_q:      array<f32>;
@group(1) @binding(7) var<storage, read_write> joint_qd:     array<f32>;

// Group 2 — Control
@group(2) @binding(0) var<storage, read> joint_force:        array<f32>;
@group(2) @binding(1) var<storage, read> joint_target_mode:  array<u32>;
@group(2) @binding(2) var<storage, read> joint_target_pos:   array<f32>;
@group(2) @binding(3) var<storage, read> joint_target_vel:   array<f32>;
@group(2) @binding(4) var<storage, read> joint_target_kp:    array<f32>;
@group(2) @binding(5) var<storage, read> joint_target_kd:    array<f32>;

// Joint target modes (mirror src/sim/joint.rs:JointTargetMode).
const TARGET_NONE: u32 = 0u;
const TARGET_POS:  u32 = 1u;
const TARGET_VEL:  u32 = 2u;
const TARGET_PV:   u32 = 3u;
const TARGET_EFF:  u32 = 4u;

// NaN-aware clamp: if `limit` is non-finite (NaN), passthrough.
fn clamp_or_passthrough(x: f32, limit: f32) -> f32 {
    if (limit != limit) { return x; } // NaN check via self-ne
    if (limit > 1e30) { return x; }   // inf passthrough
    return clamp(x, -limit, limit);
}

fn drive_torque(mode: u32, q: f32, qd: f32, kp: f32, kd: f32, pos_t: f32, vel_t: f32) -> f32 {
    if (mode == TARGET_POS) { return kp * (pos_t - q) - kd * qd; }
    if (mode == TARGET_VEL) { return kd * (vel_t - qd); }
    if (mode == TARGET_PV)  { return kp * (pos_t - q) + kd * (vel_t - qd); }
    if (mode == TARGET_EFF) { return pos_t; }
    return 0.0;
}

// Group 3 — Articulation ranges
@group(3) @binding(0) var<storage, read> art_offsets: array<u32>;

// Group 4 — Uniforms
@group(4) @binding(0) var<uniform> uniforms: Uniforms;

// ─── Spatial-vector primitives (matches Warp `_src/math.py` shape)
//
// Convention: spatial vector = (linear, angular) — `top` = linear,
// `bottom` = angular. Stored here as `vec3` pairs.

fn spatial_cross_lin(va: vec3<f32>, wa: vec3<f32>, vb: vec3<f32>, wb: vec3<f32>) -> vec3<f32> {
    return cross(wa, vb) + cross(va, wb);
}
fn spatial_cross_ang(_va: vec3<f32>, wa: vec3<f32>, _vb: vec3<f32>, wb: vec3<f32>) -> vec3<f32> {
    return cross(wa, wb);
}
fn spatial_cross_dual_lin(_va: vec3<f32>, wa: vec3<f32>, vb: vec3<f32>, _wb: vec3<f32>) -> vec3<f32> {
    return cross(wa, vb);
}
fn spatial_cross_dual_ang(va: vec3<f32>, wa: vec3<f32>, vb: vec3<f32>, wb: vec3<f32>) -> vec3<f32> {
    return cross(wa, wb) + cross(va, vb);
}

// 6×6 spatial inertia. Indexed (row, col) with `m[r * 6 + c]`.

fn smat6_zero() -> array<f32, 36> {
    return array<f32, 36>(
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    );
}

fn smat6_mul_sv(m: array<f32, 36>, v_lin: vec3<f32>, v_ang: vec3<f32>)
    -> array<f32, 6>
{
    let x = array<f32, 6>(v_lin.x, v_lin.y, v_lin.z, v_ang.x, v_ang.y, v_ang.z);
    var out: array<f32, 6>;
    for (var i = 0u; i < 6u; i = i + 1u) {
        var s = 0.0;
        for (var j = 0u; j < 6u; j = j + 1u) {
            s = s + m[i * 6u + j] * x[j];
        }
        out[i] = s;
    }
    return out;
}

fn smat6_from_body(mass: f32, com: vec3<f32>, i_world: mat3x3<f32>) -> array<f32, 36> {
    // Top-left: m·I
    // Top-right: m·skew(c)
    // Bottom-left: -m·skew(c)
    // Bottom-right: I_C - m·skew(c)·skew(c)
    var m: array<f32, 36>;
    for (var i = 0u; i < 36u; i = i + 1u) {
        m[i] = 0.0;
    }
    // Top-left m·I
    m[0u]  = mass; m[7u]  = mass; m[14u] = mass;
    // skew(com)
    let sx = mat3x3<f32>(
        vec3<f32>(0.0, com.z, -com.y),
        vec3<f32>(-com.z, 0.0, com.x),
        vec3<f32>(com.y, -com.x, 0.0),
    );
    // skew(c)·skew(c) — store in s2
    let s2 = sx * sx;
    // Top-right: m * skew (rows 0..2, cols 3..5)
    m[3u]  = mass * sx[0][0]; m[4u]  = mass * sx[1][0]; m[5u]  = mass * sx[2][0];
    m[9u]  = mass * sx[0][1]; m[10u] = mass * sx[1][1]; m[11u] = mass * sx[2][1];
    m[15u] = mass * sx[0][2]; m[16u] = mass * sx[1][2]; m[17u] = mass * sx[2][2];
    // Bottom-left: -m * skew (rows 3..5, cols 0..2)
    m[18u] = -mass * sx[0][0]; m[19u] = -mass * sx[1][0]; m[20u] = -mass * sx[2][0];
    m[24u] = -mass * sx[0][1]; m[25u] = -mass * sx[1][1]; m[26u] = -mass * sx[2][1];
    m[30u] = -mass * sx[0][2]; m[31u] = -mass * sx[1][2]; m[32u] = -mass * sx[2][2];
    // Bottom-right: I_C - m * s2
    m[21u] = i_world[0][0] - mass * s2[0][0];
    m[22u] = i_world[1][0] - mass * s2[1][0];
    m[23u] = i_world[2][0] - mass * s2[2][0];
    m[27u] = i_world[0][1] - mass * s2[0][1];
    m[28u] = i_world[1][1] - mass * s2[1][1];
    m[29u] = i_world[2][1] - mass * s2[2][1];
    m[33u] = i_world[0][2] - mass * s2[0][2];
    m[34u] = i_world[1][2] - mass * s2[1][2];
    m[35u] = i_world[2][2] - mass * s2[2][2];
    return m;
}

// ─── Helpers ───────────────────────────────────────────────────────

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
fn quat_from_axis_angle(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let h = 0.5 * angle;
    let s = sin(h);
    return vec4<f32>(axis * s, cos(h));
}
fn transform_compose(a: Transform, b: Transform) -> Transform {
    var r: Transform;
    r.rotation = normalize(quat_mul(a.rotation, b.rotation));
    r.position = vec4<f32>(a.position.xyz + quat_rotate(a.rotation, b.position.xyz), 0.0);
    return r;
}
fn transform_inverse(t: Transform) -> Transform {
    var r: Transform;
    r.rotation = quat_conj(t.rotation);
    r.position = vec4<f32>(-quat_rotate(r.rotation, t.position.xyz), 0.0);
    return r;
}
fn mat3_from_cols(m: Mat3Cols) -> mat3x3<f32> {
    return mat3x3<f32>(m.col0.xyz, m.col1.xyz, m.col2.xyz);
}
fn rotate_inertia(q: vec4<f32>, i_local: mat3x3<f32>) -> mat3x3<f32> {
    // R · I · R^T as concrete columns.
    let r0 = quat_rotate(q, vec3<f32>(1.0, 0.0, 0.0));
    let r1 = quat_rotate(q, vec3<f32>(0.0, 1.0, 0.0));
    let r2 = quat_rotate(q, vec3<f32>(0.0, 0.0, 1.0));
    let r = mat3x3<f32>(r0, r1, r2);
    let rt = transpose(r);
    return r * i_local * rt;
}
fn joint_local_transform(jt: u32, q_start: u32, axis: vec3<f32>) -> Transform {
    var t: Transform;
    t.position = vec4<f32>(0.0);
    t.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (jt == JOINT_REVOLUTE) {
        t.rotation = quat_from_axis_angle(axis, joint_q[q_start]);
    } else if (jt == JOINT_PRISMATIC) {
        t.position = vec4<f32>(axis * joint_q[q_start], 0.0);
    } else if (jt == JOINT_FREE) {
        t.position = vec4<f32>(joint_q[q_start], joint_q[q_start+1u], joint_q[q_start+2u], 0.0);
        t.rotation = normalize(vec4<f32>(joint_q[q_start+3u], joint_q[q_start+4u], joint_q[q_start+5u], joint_q[q_start+6u]));
    }
    return t;
}

// ─── Kernel ────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn featherstone_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let art = gid.x;
    if (art >= uniforms.articulation_count) {
        return;
    }
    let j_begin = art_offsets[art];
    let j_end   = art_offsets[art + 1u];
    let n = j_end - j_begin;
    if (n > MAX_JOINTS_PER_ART) {
        return; // hard limit per articulation; silently skip oversize chains
    }

    // Per-joint scratch — sized to MAX_JOINTS_PER_ART.
    var s_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var s_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var v_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var v_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var c_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var c_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var i_a:      array<array<f32, 36>, MAX_JOINTS_PER_ART>;
    var p_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var p_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var u_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var u_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var d_inv:    array<f32, MAX_JOINTS_PER_ART>;
    var u_tau:    array<f32, MAX_JOINTS_PER_ART>;
    var a_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var a_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;

    // Pass 1 — FK + body kinematics.
    for (var k = 0u; k < n; k = k + 1u) {
        let j = j_begin + k;
        let parent = joint_parent[j];
        let child  = u32(joint_child[j]);
        let coord  = joint_coord_offset[j];
        let dof    = joint_dof_offset[j];
        let jt     = joint_type[j];
        let axis   = joint_axis[j].xyz;

        var parent_world: Transform;
        if (parent < 0) {
            parent_world.position = vec4<f32>(0.0);
            parent_world.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        } else {
            parent_world = body_q[u32(parent)];
        }

        // Forward kinematics for this body.
        let local = joint_local_transform(jt, coord, axis);
        let xp = joint_x_p[j];
        let xc_inv = transform_inverse(joint_x_c[j]);
        let xpw = transform_compose(parent_world, xp);
        let after = transform_compose(xpw, local);
        body_q[child] = transform_compose(after, xc_inv);

        // Joint motion subspace in world frame (`xpw` = joint frame in world).
        let axis_w = normalize(quat_rotate(xpw.rotation, axis));
        var s_l = vec3<f32>(0.0);
        var s_a = vec3<f32>(0.0);
        if (jt == JOINT_REVOLUTE) {
            s_a = axis_w;
        } else if (jt == JOINT_PRISMATIC) {
            s_l = axis_w;
        }
        s_lin[k] = s_l;
        s_ang[k] = s_a;

        let qd_j = joint_qd[dof];
        let v_j_l = s_l * qd_j;
        let v_j_a = s_a * qd_j;

        // Parent body velocity (zero if root or parent isn't the
        // child of an earlier joint in this articulation).
        var vp_l = vec3<f32>(0.0);
        var vp_a = vec3<f32>(0.0);
        if (parent >= 0) {
            for (var kk = 0u; kk < k; kk = kk + 1u) {
                let jj = j_begin + kk;
                if (u32(joint_child[jj]) == u32(parent)) {
                    vp_l = v_lin[kk];
                    vp_a = v_ang[kk];
                    break;
                }
            }
        }
        let vi_l = vp_l + v_j_l;
        let vi_a = vp_a + v_j_a;
        v_lin[k] = vi_l;
        v_ang[k] = vi_a;

        // Coriolis term c = v ×  v_j.
        c_lin[k] = spatial_cross_lin(vi_l, vi_a, v_j_l, v_j_a);
        c_ang[k] = spatial_cross_ang(vi_l, vi_a, v_j_l, v_j_a);

        // Body spatial inertia in world frame, expressed at world origin.
        let cw = body_q[child];
        let com_world = cw.position.xyz + quat_rotate(cw.rotation, body_com[child].xyz);
        let i_local = mat3_from_cols(body_inertia[child]);
        let i_world = rotate_inertia(cw.rotation, i_local);
        var i_s = smat6_from_body(body_mass[child], com_world, i_world);

        // Bias = v ×* (I · v) - f_grav.
        let isv = smat6_mul_sv(i_s, vi_l, vi_a);
        let isv_l = vec3<f32>(isv[0u], isv[1u], isv[2u]);
        let isv_a = vec3<f32>(isv[3u], isv[4u], isv[5u]);
        let bias_l = spatial_cross_dual_lin(vi_l, vi_a, isv_l, isv_a);
        let bias_a = spatial_cross_dual_ang(vi_l, vi_a, isv_l, isv_a);
        let world_idx = max(body_world[child], 0);
        let g = gravity[u32(world_idx)].xyz;
        let f_g_lin = g * body_mass[child];
        let f_g_ang = cross(com_world, f_g_lin);
        i_a[k] = i_s;
        p_lin[k] = bias_l - f_g_lin;
        p_ang[k] = bias_a - f_g_ang;
    }

    // Pass 2 — backward articulated-inertia accumulation.
    for (var ki = 0u; ki < n; ki = ki + 1u) {
        let k = n - 1u - ki;
        let j = j_begin + k;
        let parent = joint_parent[j];
        let jt = joint_type[j];
        let supported = (jt == JOINT_REVOLUTE) || (jt == JOINT_PRISMATIC);
        if (!supported) { continue; }

        // U = I^A · S, D = S^T · U.
        let s_l = s_lin[k];
        let s_a = s_ang[k];
        let u_arr = smat6_mul_sv(i_a[k], s_l, s_a);
        let u_l = vec3<f32>(u_arr[0u], u_arr[1u], u_arr[2u]);
        let u_a = vec3<f32>(u_arr[3u], u_arr[4u], u_arr[5u]);
        let d = dot(s_l, u_l) + dot(s_a, u_a);
        let d_i = select(0.0, 1.0 / d, abs(d) > 1e-12);
        u_lin[k] = u_l;
        u_ang[k] = u_a;
        d_inv[k] = d_i;
        let dof = joint_dof_offset[j];
        let coord = joint_coord_offset[j];
        let q_now = joint_q[coord];
        let qd_now = joint_qd[dof];
        let mode = joint_target_mode[dof];
        let drive = drive_torque(
            mode,
            q_now, qd_now,
            joint_target_kp[dof], joint_target_kd[dof],
            joint_target_pos[dof], joint_target_vel[dof],
        );
        let mu_jt = joint_friction[dof];
        let tau_raw = joint_force[dof] + drive - mu_jt * qd_now;
        let tau_app = clamp_or_passthrough(tau_raw, joint_effort_limit[dof]);
        u_tau[k] = tau_app - (dot(s_l, p_lin[k]) + dot(s_a, p_ang[k]));

        if (parent >= 0) {
            // Find this joint's parent-joint within the articulation.
            var pj: i32 = -1;
            for (var kk = 0u; kk < k; kk = kk + 1u) {
                let jj = j_begin + kk;
                if (u32(joint_child[jj]) == u32(parent)) {
                    pj = i32(kk);
                    break;
                }
            }
            if (pj >= 0) {
                // I^a = I^A - U · U^T / D
                // p^a = p^A + I^a · c + U · (u_tau / D)
                // Outer product U · U^T / D.
                let ua = array<f32, 6>(u_l.x, u_l.y, u_l.z, u_a.x, u_a.y, u_a.z);
                var i_articulated: array<f32, 36>;
                for (var rr = 0u; rr < 6u; rr = rr + 1u) {
                    for (var cc = 0u; cc < 6u; cc = cc + 1u) {
                        i_articulated[rr * 6u + cc] = i_a[k][rr * 6u + cc] - ua[rr] * ua[cc] * d_i;
                    }
                }
                let cv = smat6_mul_sv(i_articulated, c_lin[k], c_ang[k]);
                let cv_l = vec3<f32>(cv[0u], cv[1u], cv[2u]);
                let cv_a = vec3<f32>(cv[3u], cv[4u], cv[5u]);
                let scale_u = u_tau[k] * d_i;
                let p_articulated_l = p_lin[k] + cv_l + u_l * scale_u;
                let p_articulated_a = p_ang[k] + cv_a + u_a * scale_u;
                // Accumulate into parent's I^A and p^A.
                let pj_u = u32(pj);
                for (var rr = 0u; rr < 36u; rr = rr + 1u) {
                    i_a[pj_u][rr] = i_a[pj_u][rr] + i_articulated[rr];
                }
                p_lin[pj_u] = p_lin[pj_u] + p_articulated_l;
                p_ang[pj_u] = p_ang[pj_u] + p_articulated_a;
            }
        }
    }

    // Pass 3 — forward acceleration solve + integrate.
    for (var k = 0u; k < n; k = k + 1u) {
        let j = j_begin + k;
        let parent = joint_parent[j];
        let jt = joint_type[j];
        let supported = (jt == JOINT_REVOLUTE) || (jt == JOINT_PRISMATIC);

        var ap_l = vec3<f32>(0.0);
        var ap_a = vec3<f32>(0.0);
        if (parent >= 0) {
            for (var kk = 0u; kk < k; kk = kk + 1u) {
                let jj = j_begin + kk;
                if (u32(joint_child[jj]) == u32(parent)) {
                    ap_l = a_lin[kk];
                    ap_a = a_ang[kk];
                    break;
                }
            }
        }

        if (supported) {
            let dof = joint_dof_offset[j];
            let coord = joint_coord_offset[j];
            let q_dd = (u_tau[k] - (dot(u_lin[k], ap_l) + dot(u_ang[k], ap_a))) * d_inv[k];

            a_lin[k] = ap_l + s_lin[k] * q_dd + c_lin[k];
            a_ang[k] = ap_a + s_ang[k] * q_dd + c_ang[k];

            // Semi-implicit Euler integrate.
            let qd_old = joint_qd[dof];
            let qd_int = qd_old + q_dd * uniforms.dt;
            let damp = max(1.0 - uniforms.joint_damping * uniforms.dt, 0.0);
            let qd_damped = qd_int * damp;
            // Per-DOF velocity cap.
            let qd_capped = clamp_or_passthrough(qd_damped, joint_velocity_limit[dof]);
            let q_pred = joint_q[coord] + qd_capped * uniforms.dt;
            // Hard joint-position limits (NaN = unlimited).
            let lo = joint_limit_lower[dof];
            let hi = joint_limit_upper[dof];
            var q_clamped = q_pred;
            var qd_after = qd_capped;
            if ((lo == lo) && (q_clamped < lo)) {
                q_clamped = lo;
                if (qd_after < 0.0) { qd_after = 0.0; }
            }
            if ((hi == hi) && (q_clamped > hi)) {
                q_clamped = hi;
                if (qd_after > 0.0) { qd_after = 0.0; }
            }
            joint_qd[dof] = qd_after;
            joint_q[coord] = q_clamped;
        } else {
            a_lin[k] = ap_l + c_lin[k];
            a_ang[k] = ap_a + c_ang[k];
        }
    }

    // Pass 4 — re-evaluate FK so body_q reflects integrated q.
    for (var k = 0u; k < n; k = k + 1u) {
        let j = j_begin + k;
        let parent = joint_parent[j];
        let child  = u32(joint_child[j]);
        let coord  = joint_coord_offset[j];
        let jt     = joint_type[j];
        let axis   = joint_axis[j].xyz;

        var parent_world: Transform;
        if (parent < 0) {
            parent_world.position = vec4<f32>(0.0);
            parent_world.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        } else {
            parent_world = body_q[u32(parent)];
        }

        let local = joint_local_transform(jt, coord, axis);
        let xp = joint_x_p[j];
        let xc_inv = transform_inverse(joint_x_c[j]);
        let xpw = transform_compose(parent_world, xp);
        let after = transform_compose(xpw, local);
        body_q[child] = transform_compose(after, xc_inv);
    }
}
