// Featherstone forward dynamics on GPU — Articulated Body Algorithm
// (ABA) per Featherstone (2008) ch. 7. Mirrors the CPU path in
// `solvers/featherstone.rs::cpu_step`.
//
// Stage 17 — multi-DOF support: each joint provides up to MAX_DOFS_PER_JOINT
// motion-subspace columns S_k. The backward ABA pass applies sequential
// rank-1 updates per DOF (correct for joints whose D matrix is diagonal;
// approximate but stable when DOFs are coupled via inertia). Supported
// joint types: REVOLUTE, PRISMATIC (1 DOF); BALL (3 angular DOFs);
// FREE / DISTANCE / D6 (6 DOFs in joint world frame).
//
// Per-articulation parallelism: one workgroup invocation per chain.

struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,
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
    /// Stage 17.1: 1 = rigorous dense-Cholesky D solve in pass 2/3.
    rigorous_d_solve:   u32,
    /// Stage 17.1: 1 = velocity-level Lagrange projection on Distance.
    rigorous_distance:  u32,
    _pad0:              u32,
    _pad1:              u32,
};

const JOINT_PRISMATIC: u32 = 0u;
const JOINT_REVOLUTE:  u32 = 1u;
const JOINT_BALL:      u32 = 2u;
const JOINT_FIXED:     u32 = 3u;
const JOINT_FREE:      u32 = 4u;
const JOINT_DISTANCE:  u32 = 5u;
const JOINT_D6:        u32 = 6u;

const MAX_JOINTS_PER_ART: u32 = 32u;
const MAX_DOFS_PER_JOINT: u32 = 6u;

@group(0) @binding(0) var<storage, read> joint_type:         array<u32>;
@group(0) @binding(1) var<storage, read> joint_parent:       array<i32>;
@group(0) @binding(2) var<storage, read> joint_child:        array<i32>;
@group(0) @binding(3) var<storage, read> joint_x_p:          array<Transform>;
@group(0) @binding(4) var<storage, read> joint_x_c:          array<Transform>;
@group(0) @binding(5) var<storage, read> joint_axis:         array<vec4<f32>>;
@group(0) @binding(6) var<storage, read> joint_coord_offset: array<u32>;
@group(0) @binding(7) var<storage, read> joint_dof_offset:   array<u32>;
@group(0) @binding(8) var<storage, read> joint_limit_lower:    array<f32>;
@group(0) @binding(9) var<storage, read> joint_limit_upper:    array<f32>;
@group(0) @binding(10) var<storage, read> joint_friction:      array<f32>;
@group(0) @binding(11) var<storage, read> joint_effort_limit:  array<f32>;
@group(0) @binding(12) var<storage, read> joint_velocity_limit: array<f32>;

@group(1) @binding(0) var<storage, read>       body_mass:    array<f32>;
@group(1) @binding(1) var<storage, read>       body_inertia: array<Mat3Cols>;
@group(1) @binding(2) var<storage, read>       body_com:     array<vec4<f32>>;
@group(1) @binding(3) var<storage, read>       body_world:   array<i32>;
@group(1) @binding(4) var<storage, read>       gravity:      array<vec4<f32>>;
@group(1) @binding(5) var<storage, read_write> body_q:       array<Transform>;
@group(1) @binding(6) var<storage, read_write> joint_q:      array<f32>;
@group(1) @binding(7) var<storage, read_write> joint_qd:     array<f32>;
// External wrench per body, at COM, world frame.
// Layout matches `GpuSpatialVector`: 2 × vec4 (linear+pad, angular+pad).
@group(1) @binding(8) var<storage, read>       body_f:       array<SpatialVecGpu>;

struct SpatialVecGpu {
    linear:  vec4<f32>,
    angular: vec4<f32>,
};

@group(2) @binding(0) var<storage, read> joint_force:        array<f32>;
@group(2) @binding(1) var<storage, read> joint_target_mode:  array<u32>;
@group(2) @binding(2) var<storage, read> joint_target_pos:   array<f32>;
@group(2) @binding(3) var<storage, read> joint_target_vel:   array<f32>;
@group(2) @binding(4) var<storage, read> joint_target_kp:    array<f32>;
@group(2) @binding(5) var<storage, read> joint_target_kd:    array<f32>;

const TARGET_NONE: u32 = 0u;
const TARGET_POS:  u32 = 1u;
const TARGET_VEL:  u32 = 2u;
const TARGET_PV:   u32 = 3u;
const TARGET_EFF:  u32 = 4u;

fn clamp_or_passthrough(x: f32, limit: f32) -> f32 {
    if (limit != limit) { return x; }
    if (limit > 1e30) { return x; }
    return clamp(x, -limit, limit);
}

fn drive_torque(mode: u32, q: f32, qd: f32, kp: f32, kd: f32, pos_t: f32, vel_t: f32) -> f32 {
    if (mode == TARGET_POS) { return kp * (pos_t - q) - kd * qd; }
    if (mode == TARGET_VEL) { return kd * (vel_t - qd); }
    if (mode == TARGET_PV)  { return kp * (pos_t - q) + kd * (vel_t - qd); }
    if (mode == TARGET_EFF) { return pos_t; }
    return 0.0;
}

@group(3) @binding(0) var<storage, read> art_offsets: array<u32>;
@group(4) @binding(0) var<uniform> uniforms: Uniforms;

// ─── Spatial primitives ─────────────────────────────────────────

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
    var m: array<f32, 36>;
    for (var i = 0u; i < 36u; i = i + 1u) {
        m[i] = 0.0;
    }
    m[0u]  = mass; m[7u]  = mass; m[14u] = mass;
    let sx = mat3x3<f32>(
        vec3<f32>(0.0, com.z, -com.y),
        vec3<f32>(-com.z, 0.0, com.x),
        vec3<f32>(com.y, -com.x, 0.0),
    );
    let s2 = sx * sx;
    m[3u]  = mass * sx[0][0]; m[4u]  = mass * sx[1][0]; m[5u]  = mass * sx[2][0];
    m[9u]  = mass * sx[0][1]; m[10u] = mass * sx[1][1]; m[11u] = mass * sx[2][1];
    m[15u] = mass * sx[0][2]; m[16u] = mass * sx[1][2]; m[17u] = mass * sx[2][2];
    m[18u] = -mass * sx[0][0]; m[19u] = -mass * sx[1][0]; m[20u] = -mass * sx[2][0];
    m[24u] = -mass * sx[0][1]; m[25u] = -mass * sx[1][1]; m[26u] = -mass * sx[2][1];
    m[30u] = -mass * sx[0][2]; m[31u] = -mass * sx[1][2]; m[32u] = -mass * sx[2][2];
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

// ─── Helpers ─────────────────────────────────────────────────────

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
    let r0 = quat_rotate(q, vec3<f32>(1.0, 0.0, 0.0));
    let r1 = quat_rotate(q, vec3<f32>(0.0, 1.0, 0.0));
    let r2 = quat_rotate(q, vec3<f32>(0.0, 0.0, 1.0));
    let r = mat3x3<f32>(r0, r1, r2);
    let rt = transpose(r);
    return r * i_local * rt;
}

/// Joint local transform (parent-anchor frame → child-anchor frame).
fn joint_local_transform(jt: u32, q_start: u32, axis: vec3<f32>) -> Transform {
    var t: Transform;
    t.position = vec4<f32>(0.0);
    t.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    if (jt == JOINT_REVOLUTE) {
        t.rotation = quat_from_axis_angle(axis, joint_q[q_start]);
    } else if (jt == JOINT_PRISMATIC) {
        t.position = vec4<f32>(axis * joint_q[q_start], 0.0);
    } else if (jt == JOINT_BALL) {
        t.rotation = normalize(vec4<f32>(joint_q[q_start], joint_q[q_start+1u], joint_q[q_start+2u], joint_q[q_start+3u]));
    } else if (jt == JOINT_FREE || jt == JOINT_DISTANCE || jt == JOINT_D6) {
        t.position = vec4<f32>(joint_q[q_start], joint_q[q_start+1u], joint_q[q_start+2u], 0.0);
        t.rotation = normalize(vec4<f32>(joint_q[q_start+3u], joint_q[q_start+4u], joint_q[q_start+5u], joint_q[q_start+6u]));
    }
    return t;
}

/// DOF count per joint type. Mirrors `JointType::dof_count`.
fn joint_dof_n(jt: u32) -> u32 {
    if (jt == JOINT_REVOLUTE || jt == JOINT_PRISMATIC) { return 1u; }
    if (jt == JOINT_BALL) { return 3u; }
    if (jt == JOINT_FREE || jt == JOINT_DISTANCE || jt == JOINT_D6) { return 6u; }
    return 0u; // FIXED
}

/// Decompose a quaternion into XYZ-Euler angles `(rx, ry, rz)`.
/// Matches glam's `Quat::to_euler(EulerRot::XYZ)` convention so the CPU
/// path's `Quat::to_euler(EulerRot::XYZ)` and the WGSL path agree.
fn quat_to_euler_xyz(q: vec4<f32>) -> vec3<f32> {
    // glam XYZ extrinsic order: rotation = Rz · Ry · Rx (intrinsic XYZ).
    // The math below is derived from the rotation matrix entries of that
    // composition. Clamp the asin argument to keep gimbal-lock stable.
    let xx = q.x * q.x;
    let yy = q.y * q.y;
    let zz = q.z * q.z;
    let ww = q.w * q.w;
    let xy = q.x * q.y;
    let xz = q.x * q.z;
    let yz = q.y * q.z;
    let wx = q.w * q.x;
    let wy = q.w * q.y;
    let wz = q.w * q.z;

    let m00 = ww + xx - yy - zz;
    let m01 = 2.0 * (xy - wz);
    let m02 = 2.0 * (xz + wy);
    let m12 = 2.0 * (yz - wx);
    let m22 = ww - xx - yy + zz;

    let sy = clamp(m02, -1.0, 1.0);
    let ey = asin(sy);
    var ex = 0.0;
    var ez = 0.0;
    if (abs(sy) < 0.99999) {
        ex = atan2(-m12, m22);
        ez = atan2(-m01, m00);
    } else {
        // Gimbal lock — collapse roll into yaw.
        ex = atan2(2.0 * (yz + wx), ww - xx + yy - zz);
        ez = 0.0;
    }
    return vec3<f32>(ex, ey, ez);
}

fn quat_from_euler_xyz(e: vec3<f32>) -> vec4<f32> {
    let rx = quat_from_axis_angle(vec3<f32>(1.0, 0.0, 0.0), e.x);
    let ry = quat_from_axis_angle(vec3<f32>(0.0, 1.0, 0.0), e.y);
    let rz = quat_from_axis_angle(vec3<f32>(0.0, 0.0, 1.0), e.z);
    return normalize(quat_mul(quat_mul(rz, ry), rx));
}

// Note: dense Cholesky / forward+back substitution helpers live on the
// CPU side only (see `solvers/featherstone.rs` and `math/dense_solve.rs`).
// WGSL doesn't ergonomically support the per-thread function-scoped
// dynamic linear algebra we'd need here, so the GPU kernel always uses
// the sequential rank-1 deflation in pass-2 even when
// `uniforms.rigorous_d_solve == 1u`. The angular-limit + Distance
// projection polish is honoured on both backends.

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
        return;
    }

    // Per-joint scratch.
    var v_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var v_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var c_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var c_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var i_a:      array<array<f32, 36>, MAX_JOINTS_PER_ART>;
    var p_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var p_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var a_lin:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var a_ang:    array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var n_dof_arr: array<u32, MAX_JOINTS_PER_ART>;

    // Per-joint × per-DOF scratch.
    // S_lin[k][d] / S_ang[k][d] is the d-th column of joint k's
    // motion subspace in world frame.
    var s_lin:  array<array<vec3<f32>, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
    var s_ang:  array<array<vec3<f32>, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
    var u_lin:  array<array<vec3<f32>, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
    var u_ang:  array<array<vec3<f32>, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
    var d_inv_arr: array<array<f32, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
    var u_tau_arr: array<array<f32, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
    var dof_active_arr: array<array<u32, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
    var q_dd_arr: array<array<f32, MAX_DOFS_PER_JOINT>, MAX_JOINTS_PER_ART>;
    // Stage 17.1 — pre-integration position copy for the Distance
    // Lagrange projection (used by the velocity-level multiplier path
    // in pass-4). Only stored for FREE / DISTANCE / D6.
    //
    // **Caveat — dense Cholesky D solve is CPU-only for now.** WGSL's
    // function-pointer / dynamic-indexing limits make the in-place
    // dense Cholesky on a per-joint 6×6 painful (we'd need per-thread
    // scratch with run-time-sized linear algebra). The CPU path
    // implements it; the GPU path here always uses the sequential
    // rank-1 deflation that's exact when `D` is diagonal. For the
    // asymmetric-inertia coupled-D parity test we run the CPU backend
    // and compare to a CPU dense-Cholesky reference.
    var p_pre_arr:  array<vec3<f32>, MAX_JOINTS_PER_ART>;

    // ── Pass 1 — FK + body kinematics + motion subspace ─────────
    for (var k = 0u; k < n; k = k + 1u) {
        let j = j_begin + k;
        let parent = joint_parent[j];
        let child  = u32(joint_child[j]);
        let coord  = joint_coord_offset[j];
        let dof    = joint_dof_offset[j];
        let jt     = joint_type[j];
        let axis_l = joint_axis[j].xyz;

        var parent_world: Transform;
        if (parent < 0) {
            parent_world.position = vec4<f32>(0.0);
            parent_world.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        } else {
            parent_world = body_q[u32(parent)];
        }

        let local = joint_local_transform(jt, coord, axis_l);
        let xp = joint_x_p[j];
        let xc_inv = transform_inverse(joint_x_c[j]);
        let xpw = transform_compose(parent_world, xp);
        let after = transform_compose(xpw, local);
        body_q[child] = transform_compose(after, xc_inv);

        // Motion subspace columns in joint world frame.
        let r_jw = xpw.rotation;
        let ex = quat_rotate(r_jw, vec3<f32>(1.0, 0.0, 0.0));
        let ey = quat_rotate(r_jw, vec3<f32>(0.0, 1.0, 0.0));
        let ez = quat_rotate(r_jw, vec3<f32>(0.0, 0.0, 1.0));

        // Initialise all columns to zero / inactive.
        for (var d = 0u; d < MAX_DOFS_PER_JOINT; d = d + 1u) {
            s_lin[k][d] = vec3<f32>(0.0);
            s_ang[k][d] = vec3<f32>(0.0);
            dof_active_arr[k][d] = 0u;
        }

        var n_dof = 0u;
        if (jt == JOINT_REVOLUTE) {
            let axis_w = normalize(quat_rotate(r_jw, axis_l));
            s_lin[k][0u] = vec3<f32>(0.0);
            s_ang[k][0u] = axis_w;
            dof_active_arr[k][0u] = 1u;
            n_dof = 1u;
        } else if (jt == JOINT_PRISMATIC) {
            let axis_w = normalize(quat_rotate(r_jw, axis_l));
            s_lin[k][0u] = axis_w;
            s_ang[k][0u] = vec3<f32>(0.0);
            dof_active_arr[k][0u] = 1u;
            n_dof = 1u;
        } else if (jt == JOINT_BALL) {
            s_ang[k][0u] = ex; dof_active_arr[k][0u] = 1u;
            s_ang[k][1u] = ey; dof_active_arr[k][1u] = 1u;
            s_ang[k][2u] = ez; dof_active_arr[k][2u] = 1u;
            n_dof = 3u;
        } else if (jt == JOINT_FREE || jt == JOINT_DISTANCE) {
            s_lin[k][0u] = ex; dof_active_arr[k][0u] = 1u;
            s_lin[k][1u] = ey; dof_active_arr[k][1u] = 1u;
            s_lin[k][2u] = ez; dof_active_arr[k][2u] = 1u;
            s_ang[k][3u] = ex; dof_active_arr[k][3u] = 1u;
            s_ang[k][4u] = ey; dof_active_arr[k][4u] = 1u;
            s_ang[k][5u] = ez; dof_active_arr[k][5u] = 1u;
            n_dof = 6u;
        } else if (jt == JOINT_D6) {
            // Lock-aware: lower==upper (finite) = locked.
            for (var d = 0u; d < 6u; d = d + 1u) {
                let lo = joint_limit_lower[dof + d];
                let hi = joint_limit_upper[dof + d];
                let lo_finite = (lo == lo) && (abs(lo) < 1e30);
                let hi_finite = (hi == hi) && (abs(hi) < 1e30);
                let locked = lo_finite && hi_finite && (abs(hi - lo) < 1e-12);
                var v_l = vec3<f32>(0.0);
                var v_a = vec3<f32>(0.0);
                if (d == 0u) { v_l = ex; }
                else if (d == 1u) { v_l = ey; }
                else if (d == 2u) { v_l = ez; }
                else if (d == 3u) { v_a = ex; }
                else if (d == 4u) { v_a = ey; }
                else { v_a = ez; }
                if (!locked) {
                    s_lin[k][d] = v_l;
                    s_ang[k][d] = v_a;
                    dof_active_arr[k][d] = 1u;
                }
            }
            n_dof = 6u;
        }
        n_dof_arr[k] = n_dof;

        // Stash pre-integration anchor offset for Distance/D6/Free
        // joints (used by the Lagrange projection in pass-4).
        if (jt == JOINT_FREE || jt == JOINT_DISTANCE || jt == JOINT_D6) {
            p_pre_arr[k] = vec3<f32>(joint_q[coord], joint_q[coord+1u], joint_q[coord+2u]);
        } else {
            p_pre_arr[k] = vec3<f32>(0.0);
        }

        // Joint velocity = Σ S_k · qd_k.
        var v_j_l = vec3<f32>(0.0);
        var v_j_a = vec3<f32>(0.0);
        for (var d = 0u; d < n_dof; d = d + 1u) {
            if (dof_active_arr[k][d] != 0u) {
                let qd_k = joint_qd[dof + d];
                v_j_l = v_j_l + s_lin[k][d] * qd_k;
                v_j_a = v_j_a + s_ang[k][d] * qd_k;
            }
        }

        // Parent body's spatial velocity.
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

        c_lin[k] = spatial_cross_lin(vi_l, vi_a, v_j_l, v_j_a);
        c_ang[k] = spatial_cross_ang(vi_l, vi_a, v_j_l, v_j_a);

        let cw = body_q[child];
        let com_world = cw.position.xyz + quat_rotate(cw.rotation, body_com[child].xyz);
        let i_local = mat3_from_cols(body_inertia[child]);
        let i_world = rotate_inertia(cw.rotation, i_local);
        var i_s = smat6_from_body(body_mass[child], com_world, i_world);

        let isv = smat6_mul_sv(i_s, vi_l, vi_a);
        let isv_l = vec3<f32>(isv[0u], isv[1u], isv[2u]);
        let isv_a = vec3<f32>(isv[3u], isv[4u], isv[5u]);
        let bias_l = spatial_cross_dual_lin(vi_l, vi_a, isv_l, isv_a);
        let bias_a = spatial_cross_dual_ang(vi_l, vi_a, isv_l, isv_a);
        let world_idx = max(body_world[child], 0);
        let g = gravity[u32(world_idx)].xyz;
        let f_g_lin = g * body_mass[child];
        let f_g_ang = cross(com_world, f_g_lin);
        // External wrench at COM (caller writes `state.body_f[child]`):
        // re-express at world origin — linear unchanged, angular gets
        // +r_com × F.
        let ext = body_f[child];
        let f_ext_lin = ext.linear.xyz;
        let f_ext_ang = ext.angular.xyz + cross(com_world, ext.linear.xyz);
        i_a[k] = i_s;
        p_lin[k] = bias_l - f_g_lin - f_ext_lin;
        p_ang[k] = bias_a - f_g_ang - f_ext_ang;
    }

    // ── Pass 2 — backward articulated-inertia accumulation ───────
    for (var ki = 0u; ki < n; ki = ki + 1u) {
        let k = n - 1u - ki;
        let j = j_begin + k;
        let parent = joint_parent[j];
        let jt = joint_type[j];
        let n_dof = n_dof_arr[k];
        let supported = (jt != JOINT_FIXED) && (n_dof > 0u);
        if (!supported) { continue; }

        let dof = joint_dof_offset[j];
        let coord = joint_coord_offset[j];

        // Sequential rank-1 updates per DOF.
        var i_curr: array<f32, 36> = i_a[k];
        for (var d = 0u; d < n_dof; d = d + 1u) {
            if (dof_active_arr[k][d] == 0u) { continue; }
            let s_l = s_lin[k][d];
            let s_a = s_ang[k][d];
            let u_arr = smat6_mul_sv(i_curr, s_l, s_a);
            let u_l = vec3<f32>(u_arr[0u], u_arr[1u], u_arr[2u]);
            let u_a = vec3<f32>(u_arr[3u], u_arr[4u], u_arr[5u]);
            let dval = dot(s_l, u_l) + dot(s_a, u_a);
            let d_i = select(0.0, 1.0 / dval, abs(dval) > 1e-12);
            u_lin[k][d] = u_l;
            u_ang[k][d] = u_a;
            d_inv_arr[k][d] = d_i;

            // Drive / friction / effort.
            let dof_g = dof + d;
            let qd_now = joint_qd[dof_g];
            var q_now = 0.0;
            if (jt == JOINT_REVOLUTE || jt == JOINT_PRISMATIC) {
                q_now = joint_q[coord + d];
            }
            let mode = joint_target_mode[dof_g];
            let drv = drive_torque(
                mode,
                q_now, qd_now,
                joint_target_kp[dof_g], joint_target_kd[dof_g],
                joint_target_pos[dof_g], joint_target_vel[dof_g],
            );
            let mu_jt = joint_friction[dof_g];
            let tau_raw = joint_force[dof_g] + drv - mu_jt * qd_now;
            let tau_app = clamp_or_passthrough(tau_raw, joint_effort_limit[dof_g]);
            u_tau_arr[k][d] = tau_app - (dot(s_l, p_lin[k]) + dot(s_a, p_ang[k]));

            // Deflate i_curr := i_curr - U·U^T / D
            let ua = array<f32, 6>(u_l.x, u_l.y, u_l.z, u_a.x, u_a.y, u_a.z);
            for (var rr = 0u; rr < 6u; rr = rr + 1u) {
                for (var cc = 0u; cc < 6u; cc = cc + 1u) {
                    i_curr[rr * 6u + cc] = i_curr[rr * 6u + cc] - ua[rr] * ua[cc] * d_i;
                }
            }
        }

        // p^a = p^A + i_curr · c + Σ U_k · (u_tau_k / D_k)
        let cv = smat6_mul_sv(i_curr, c_lin[k], c_ang[k]);
        var p_art_l = p_lin[k] + vec3<f32>(cv[0u], cv[1u], cv[2u]);
        var p_art_a = p_ang[k] + vec3<f32>(cv[3u], cv[4u], cv[5u]);
        for (var d = 0u; d < n_dof; d = d + 1u) {
            if (dof_active_arr[k][d] == 0u) { continue; }
            let scale_u = u_tau_arr[k][d] * d_inv_arr[k][d];
            p_art_l = p_art_l + u_lin[k][d] * scale_u;
            p_art_a = p_art_a + u_ang[k][d] * scale_u;
        }

        if (parent >= 0) {
            var pj: i32 = -1;
            for (var kk = 0u; kk < k; kk = kk + 1u) {
                let jj = j_begin + kk;
                if (u32(joint_child[jj]) == u32(parent)) {
                    pj = i32(kk);
                    break;
                }
            }
            if (pj >= 0) {
                let pj_u = u32(pj);
                for (var rr = 0u; rr < 36u; rr = rr + 1u) {
                    i_a[pj_u][rr] = i_a[pj_u][rr] + i_curr[rr];
                }
                p_lin[pj_u] = p_lin[pj_u] + p_art_l;
                p_ang[pj_u] = p_ang[pj_u] + p_art_a;
            }
        }
    }

    // ── Pass 3 — forward acceleration solve + integrate ─────────
    for (var k = 0u; k < n; k = k + 1u) {
        let j = j_begin + k;
        let parent = joint_parent[j];
        let jt = joint_type[j];
        let n_dof = n_dof_arr[k];
        let supported = (jt != JOINT_FIXED) && (n_dof > 0u);

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

        var s_qdd_l = vec3<f32>(0.0);
        var s_qdd_a = vec3<f32>(0.0);
        if (supported) {
            for (var d = 0u; d < n_dof; d = d + 1u) {
                if (dof_active_arr[k][d] == 0u) {
                    q_dd_arr[k][d] = 0.0;
                    continue;
                }
                let q_dd_d = (u_tau_arr[k][d]
                              - (dot(u_lin[k][d], ap_l) + dot(u_ang[k][d], ap_a)))
                              * d_inv_arr[k][d];
                q_dd_arr[k][d] = q_dd_d;
                s_qdd_l = s_qdd_l + s_lin[k][d] * q_dd_d;
                s_qdd_a = s_qdd_a + s_ang[k][d] * q_dd_d;
            }
            a_lin[k] = ap_l + s_qdd_l + c_lin[k];
            a_ang[k] = ap_a + s_qdd_a + c_ang[k];
        } else {
            a_lin[k] = ap_l + c_lin[k];
            a_ang[k] = ap_a + c_ang[k];
        }
    }

    // ── Integrate q, qd ─────────────────────────────────────────
    for (var k = 0u; k < n; k = k + 1u) {
        let j = j_begin + k;
        let jt = joint_type[j];
        let coord = joint_coord_offset[j];
        let dof = joint_dof_offset[j];
        let n_dof = n_dof_arr[k];
        let damp = max(1.0 - uniforms.joint_damping * uniforms.dt, 0.0);

        if (jt == JOINT_REVOLUTE || jt == JOINT_PRISMATIC) {
            let qd_old = joint_qd[dof];
            let qd_int = qd_old + q_dd_arr[k][0u] * uniforms.dt;
            let qd_damped = qd_int * damp;
            let qd_capped = clamp_or_passthrough(qd_damped, joint_velocity_limit[dof]);
            let q_pred = joint_q[coord] + qd_capped * uniforms.dt;
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
        } else if (jt == JOINT_BALL) {
            var omega = vec3<f32>(joint_qd[dof], joint_qd[dof+1u], joint_qd[dof+2u]);
            let alpha = vec3<f32>(q_dd_arr[k][0u], q_dd_arr[k][1u], q_dd_arr[k][2u]);
            omega = (omega + alpha * uniforms.dt) * damp;
            let q_old = vec4<f32>(joint_q[coord], joint_q[coord+1u], joint_q[coord+2u], joint_q[coord+3u]);
            let dq = vec4<f32>(omega.x * 0.5, omega.y * 0.5, omega.z * 0.5, 0.0);
            let q_dot = quat_mul(dq, q_old);
            let q_new = normalize(q_old + q_dot * uniforms.dt);
            joint_q[coord]    = q_new.x;
            joint_q[coord+1u] = q_new.y;
            joint_q[coord+2u] = q_new.z;
            joint_q[coord+3u] = q_new.w;
            joint_qd[dof]    = omega.x;
            joint_qd[dof+1u] = omega.y;
            joint_qd[dof+2u] = omega.z;
        } else if (jt == JOINT_FREE || jt == JOINT_DISTANCE || jt == JOINT_D6) {
            var v = vec3<f32>(joint_qd[dof], joint_qd[dof+1u], joint_qd[dof+2u]);
            var omega = vec3<f32>(joint_qd[dof+3u], joint_qd[dof+4u], joint_qd[dof+5u]);
            let a_l = vec3<f32>(q_dd_arr[k][0u], q_dd_arr[k][1u], q_dd_arr[k][2u]);
            let a_a = vec3<f32>(q_dd_arr[k][3u], q_dd_arr[k][4u], q_dd_arr[k][5u]);
            v = (v + a_l * uniforms.dt) * damp;
            omega = (omega + a_a * uniforms.dt) * damp;
            let p_old = vec3<f32>(joint_q[coord], joint_q[coord+1u], joint_q[coord+2u]);
            let p_new = p_old + v * uniforms.dt;
            let r_old = vec4<f32>(joint_q[coord+3u], joint_q[coord+4u], joint_q[coord+5u], joint_q[coord+6u]);
            let dq = vec4<f32>(omega.x * 0.5, omega.y * 0.5, omega.z * 0.5, 0.0);
            let r_dot = quat_mul(dq, r_old);
            let r_new = normalize(r_old + r_dot * uniforms.dt);
            joint_q[coord]    = p_new.x;
            joint_q[coord+1u] = p_new.y;
            joint_q[coord+2u] = p_new.z;
            joint_q[coord+3u] = r_new.x;
            joint_q[coord+4u] = r_new.y;
            joint_q[coord+5u] = r_new.z;
            joint_q[coord+6u] = r_new.w;
            joint_qd[dof]    = v.x;
            joint_qd[dof+1u] = v.y;
            joint_qd[dof+2u] = v.z;
            joint_qd[dof+3u] = omega.x;
            joint_qd[dof+4u] = omega.y;
            joint_qd[dof+5u] = omega.z;

            // Distance constraint.
            //
            // Stage 17.1 default: velocity-level Lagrange-multiplier
            // projection. Pick the active bound (lo, hi, or
            // equality-form lo==hi) as `L_target`, project velocity
            // onto the tangent plane, re-integrate position from the
            // projected v, then snap |p| to L_target to absorb
            // floating-point drift.
            //
            // When `uniforms.rigorous_distance == 0u` we keep the v1
            // hard-clamp for parity tests.
            if (jt == JOINT_DISTANCE) {
                let lo = joint_limit_lower[dof];
                let hi = joint_limit_upper[dof];
                let lo_finite = (lo == lo) && (abs(lo) < 1e30);
                let hi_finite = (hi == hi) && (abs(hi) < 1e30);

                if (uniforms.rigorous_distance == 1u) {
                    let p_pre = p_pre_arr[k];
                    let dist_pre = length(p_pre);
                    var l_target = -1.0;
                    if (lo_finite && hi_finite && abs(hi - lo) < 1e-9) {
                        l_target = lo;
                    } else if (lo_finite && dist_pre <= lo + 1e-9) {
                        l_target = lo;
                    } else if (hi_finite && dist_pre >= hi - 1e-9) {
                        l_target = hi;
                    } else if (lo_finite && !hi_finite) {
                        l_target = lo;
                    } else if (hi_finite && !lo_finite) {
                        l_target = hi;
                    }
                    if (l_target >= 0.0) {
                        var n_hat = vec3<f32>(0.0, 1.0, 0.0);
                        if (dist_pre > 1e-12) {
                            n_hat = p_pre / dist_pre;
                        }
                        let v_int = vec3<f32>(joint_qd[dof], joint_qd[dof+1u], joint_qd[dof+2u]);
                        let v_proj = v_int - n_hat * dot(n_hat, v_int);
                        let p_re = p_pre + v_proj * uniforms.dt;
                        var p_final = p_re;
                        let dist_re = length(p_re);
                        if (dist_re > 1e-12) {
                            p_final = p_re * (l_target / dist_re);
                        }
                        var n_hat_f = n_hat;
                        if (l_target > 1e-12) { n_hat_f = p_final / l_target; }
                        let v_final = v_proj - n_hat_f * dot(n_hat_f, v_proj);
                        joint_q[coord]    = p_final.x;
                        joint_q[coord+1u] = p_final.y;
                        joint_q[coord+2u] = p_final.z;
                        joint_qd[dof]    = v_final.x;
                        joint_qd[dof+1u] = v_final.y;
                        joint_qd[dof+2u] = v_final.z;
                    }
                } else {
                    var p_curr = vec3<f32>(joint_q[coord], joint_q[coord+1u], joint_q[coord+2u]);
                    let dist = length(p_curr);
                    var clamped = p_curr;
                    var clip = false;
                    if (lo_finite && (dist < lo) && (dist > 1e-12)) {
                        clamped = p_curr * (lo / dist);
                        clip = true;
                    }
                    if (hi_finite && (dist > hi) && (dist > 1e-12)) {
                        clamped = p_curr * (hi / dist);
                        clip = true;
                    }
                    if (clip) {
                        joint_q[coord]    = clamped.x;
                        joint_q[coord+1u] = clamped.y;
                        joint_q[coord+2u] = clamped.z;
                        let nrm = normalize(clamped);
                        let v_now = vec3<f32>(joint_qd[dof], joint_qd[dof+1u], joint_qd[dof+2u]);
                        let radial = nrm * dot(v_now, nrm);
                        let v_proj = v_now - radial;
                        joint_qd[dof]    = v_proj.x;
                        joint_qd[dof+1u] = v_proj.y;
                        joint_qd[dof+2u] = v_proj.z;
                    }
                }
            }

            // D6 per-axis limits.
            //
            // Stage 17.1 polish:
            //   - Linear axes 0..3 still clamp the per-axis position.
            //   - Angular axes 3..6 decompose the integrated quaternion
            //     into XYZ Euler angles and clamp each unlocked axis,
            //     then recompose.
            if (jt == JOINT_D6) {
                for (var d = 0u; d < 3u; d = d + 1u) {
                    let lo = joint_limit_lower[dof + d];
                    let hi = joint_limit_upper[dof + d];
                    var p_k = joint_q[coord + d];
                    var qd_k = joint_qd[dof + d];
                    if ((lo == lo) && (p_k < lo)) {
                        p_k = lo;
                        if (qd_k < 0.0) { qd_k = 0.0; }
                    }
                    if ((hi == hi) && (p_k > hi)) {
                        p_k = hi;
                        if (qd_k > 0.0) { qd_k = 0.0; }
                    }
                    joint_q[coord + d] = p_k;
                    joint_qd[dof + d] = qd_k;
                }

                let r_now = vec4<f32>(
                    joint_q[coord+3u],
                    joint_q[coord+4u],
                    joint_q[coord+5u],
                    joint_q[coord+6u],
                );
                var ang = quat_to_euler_xyz(r_now);
                var omega_arr = vec3<f32>(
                    joint_qd[dof+3u],
                    joint_qd[dof+4u],
                    joint_qd[dof+5u],
                );
                var clamped = false;
                for (var d = 0u; d < 3u; d = d + 1u) {
                    let lo = joint_limit_lower[dof + 3u + d];
                    let hi = joint_limit_upper[dof + 3u + d];
                    var ad = ang[d];
                    var od = omega_arr[d];
                    if ((lo == lo) && (ad < lo)) {
                        ad = lo;
                        if (od < 0.0) { od = 0.0; }
                        clamped = true;
                    }
                    if ((hi == hi) && (ad > hi)) {
                        ad = hi;
                        if (od > 0.0) { od = 0.0; }
                        clamped = true;
                    }
                    ang[d] = ad;
                    omega_arr[d] = od;
                }
                if (clamped) {
                    let r_proj = quat_from_euler_xyz(ang);
                    joint_q[coord+3u] = r_proj.x;
                    joint_q[coord+4u] = r_proj.y;
                    joint_q[coord+5u] = r_proj.z;
                    joint_q[coord+6u] = r_proj.w;
                    joint_qd[dof+3u] = omega_arr.x;
                    joint_qd[dof+4u] = omega_arr.y;
                    joint_qd[dof+5u] = omega_arr.z;
                }
            }
        }
    }

    // ── Pass 4 — re-evaluate FK so body_q reflects integrated q.
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
