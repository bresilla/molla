// Recursive Newton-Euler — joint-space inverse dynamics on the GPU.
//
// Mirrors `crates/molla/src/solvers/rnea.rs::inverse_dynamics`. One
// workgroup invocation per articulation; per-joint scratch sized to
// MAX_JOINTS_PER_ART = 64 (matches `featherstone.wgsl`).
//
// Two passes:
//   1. Forward (root → leaf): per-body spatial velocity, acceleration,
//      and net spatial force `f = I·a + v ×* (I·v)`.
//   2. Backward (leaf → root): `τ_k = S_kᵀ·f_k`; force transferred to
//      parent via simple summation (vectors expressed at world origin).
//
// Gravity folded into base acceleration `a_root = (-g, 0)`.
//
// Output: per-articulation `tau` packed into `tau` buffer using the
// same per-articulation DOF offsets as the input `qdd` buffer (CSR
// indexed via `dof_offsets`).

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
    articulation_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
};

const JOINT_PRISMATIC: u32 = 0u;
const JOINT_REVOLUTE:  u32 = 1u;
const JOINT_FIXED:     u32 = 3u;

const MAX_JOINTS_PER_ART: u32 = 64u;

// Group 0 — Model
@group(0) @binding(0)  var<storage, read> joint_type:        array<u32>;
@group(0) @binding(1)  var<storage, read> joint_parent:      array<i32>;
@group(0) @binding(2)  var<storage, read> joint_child:       array<i32>;
@group(0) @binding(3)  var<storage, read> joint_x_p:         array<Transform>;
@group(0) @binding(4)  var<storage, read> joint_axis:        array<vec4<f32>>;
@group(0) @binding(5)  var<storage, read> joint_dof_offset:  array<u32>;
@group(0) @binding(6)  var<storage, read> body_mass:         array<f32>;
@group(0) @binding(7)  var<storage, read> body_inertia:      array<Mat3Cols>;
@group(0) @binding(8)  var<storage, read> body_com:          array<vec4<f32>>;
@group(0) @binding(9)  var<storage, read> body_world:        array<i32>;
@group(0) @binding(10) var<storage, read> gravity:           array<vec4<f32>>;

// Group 1 — State
@group(1) @binding(0) var<storage, read> body_q:   array<Transform>;
@group(1) @binding(1) var<storage, read> joint_qd: array<f32>;

// Group 2 — Articulation CSR + qdd input + tau output + dof offsets.
@group(2) @binding(0) var<storage, read>       art_offsets: array<u32>;
@group(2) @binding(1) var<storage, read>       dof_offsets: array<u32>;
@group(2) @binding(2) var<storage, read>       qdd_in:      array<f32>;
@group(2) @binding(3) var<storage, read_write> tau_out:     array<f32>;

// Group 3 — Uniforms
@group(3) @binding(0) var<uniform> uniforms: Uniforms;

// ─── Spatial primitives ────────────────────────────────────────────

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z,
    );
}
fn quat_rotate(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let u = q.xyz;
    let s = q.w;
    return 2.0 * dot(u, v) * u + (s*s - dot(u,u)) * v + 2.0 * s * cross(u, v);
}
fn transform_compose(a: Transform, b: Transform) -> Transform {
    var r: Transform;
    r.rotation = normalize(quat_mul(a.rotation, b.rotation));
    r.position = vec4<f32>(a.position.xyz + quat_rotate(a.rotation, b.position.xyz), 0.0);
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

struct SpatialV {
    lin: vec3<f32>,
    ang: vec3<f32>,
};

fn jacobian_col_world_origin(jt: u32, axis_w: vec3<f32>, origin_w: vec3<f32>) -> SpatialV {
    var s: SpatialV;
    s.lin = vec3<f32>(0.0);
    s.ang = vec3<f32>(0.0);
    if (jt == JOINT_REVOLUTE) {
        s.lin = cross(axis_w, -origin_w);
        s.ang = axis_w;
    } else if (jt == JOINT_PRISMATIC) {
        s.lin = axis_w;
    }
    return s;
}

// 6×6 spatial inertia at world origin, [linear; angular] order.
fn smat6_world_origin(mass: f32, c: vec3<f32>, i_world: mat3x3<f32>) -> array<f32, 36> {
    var m: array<f32, 36>;
    for (var i = 0u; i < 36u; i = i + 1u) { m[i] = 0.0; }
    m[0u]  = mass; m[7u]  = mass; m[14u] = mass;
    let sx00 = 0.0;   let sx01 = -c.z;  let sx02 = c.y;
    let sx10 = c.z;   let sx11 = 0.0;   let sx12 = -c.x;
    let sx20 = -c.y;  let sx21 = c.x;   let sx22 = 0.0;
    m[3u]  = -mass*sx00; m[4u]  = -mass*sx01; m[5u]  = -mass*sx02;
    m[9u]  = -mass*sx10; m[10u] = -mass*sx11; m[11u] = -mass*sx12;
    m[15u] = -mass*sx20; m[16u] = -mass*sx21; m[17u] = -mass*sx22;
    m[18u] = mass*sx00; m[19u] = mass*sx01; m[20u] = mass*sx02;
    m[24u] = mass*sx10; m[25u] = mass*sx11; m[26u] = mass*sx12;
    m[30u] = mass*sx20; m[31u] = mass*sx21; m[32u] = mass*sx22;
    let s2_00 = sx00*sx00 + sx01*sx10 + sx02*sx20;
    let s2_01 = sx00*sx01 + sx01*sx11 + sx02*sx21;
    let s2_02 = sx00*sx02 + sx01*sx12 + sx02*sx22;
    let s2_10 = sx10*sx00 + sx11*sx10 + sx12*sx20;
    let s2_11 = sx10*sx01 + sx11*sx11 + sx12*sx21;
    let s2_12 = sx10*sx02 + sx11*sx12 + sx12*sx22;
    let s2_20 = sx20*sx00 + sx21*sx10 + sx22*sx20;
    let s2_21 = sx20*sx01 + sx21*sx11 + sx22*sx21;
    let s2_22 = sx20*sx02 + sx21*sx12 + sx22*sx22;
    m[21u] = i_world[0][0] - mass * s2_00;
    m[22u] = i_world[1][0] - mass * s2_01;
    m[23u] = i_world[2][0] - mass * s2_02;
    m[27u] = i_world[0][1] - mass * s2_10;
    m[28u] = i_world[1][1] - mass * s2_11;
    m[29u] = i_world[2][1] - mass * s2_12;
    m[33u] = i_world[0][2] - mass * s2_20;
    m[34u] = i_world[1][2] - mass * s2_21;
    m[35u] = i_world[2][2] - mass * s2_22;
    return m;
}

fn smat6_mul_sv(m: array<f32, 36>, v_lin: vec3<f32>, v_ang: vec3<f32>) -> SpatialV {
    let x = array<f32, 6>(v_lin.x, v_lin.y, v_lin.z, v_ang.x, v_ang.y, v_ang.z);
    var out: array<f32, 6>;
    for (var i = 0u; i < 6u; i = i + 1u) {
        var s = 0.0;
        for (var j = 0u; j < 6u; j = j + 1u) {
            s = s + m[i * 6u + j] * x[j];
        }
        out[i] = s;
    }
    var sv: SpatialV;
    sv.lin = vec3<f32>(out[0u], out[1u], out[2u]);
    sv.ang = vec3<f32>(out[3u], out[4u], out[5u]);
    return sv;
}

// spatial_cross — motion vectors. linear = w_a×v_b + v_a×w_b; angular = w_a×w_b.
fn spatial_cross(a_lin: vec3<f32>, a_ang: vec3<f32>, b_lin: vec3<f32>, b_ang: vec3<f32>) -> SpatialV {
    var s: SpatialV;
    s.lin = cross(a_ang, b_lin) + cross(a_lin, b_ang);
    s.ang = cross(a_ang, b_ang);
    return s;
}
// spatial_cross_dual — force vectors. linear = w_a×v_b; angular = w_a×w_b + v_a×v_b.
fn spatial_cross_dual(a_lin: vec3<f32>, a_ang: vec3<f32>, b_lin: vec3<f32>, b_ang: vec3<f32>) -> SpatialV {
    var s: SpatialV;
    s.lin = cross(a_ang, b_lin);
    s.ang = cross(a_ang, b_ang) + cross(a_lin, b_lin);
    return s;
}

// ─── Kernel ────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn rnea_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let art = gid.x;
    if (art >= uniforms.articulation_count) { return; }
    let j_begin = art_offsets[art];
    let j_end   = art_offsets[art + 1u];
    let n = j_end - j_begin;
    if (n == 0u) { return; }
    if (n > MAX_JOINTS_PER_ART) { return; }

    // Per-joint scratch.
    var parent_local: array<i32, MAX_JOINTS_PER_ART>;
    var s_lin:        array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var s_ang:        array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var v_lin:        array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var v_ang:        array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var a_lin:        array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var a_ang:        array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var f_lin:        array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var f_ang:        array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var dof_local_offset: array<u32, MAX_JOINTS_PER_ART>;
    var dof_local_width:  array<u32, MAX_JOINTS_PER_ART>;

    let dof_base = dof_offsets[art];
    let qd_base = joint_dof_offset[j_begin];

    // Pass 1 — forward sweep.
    var dof_total: u32 = 0u;
    for (var k = 0u; k < n; k = k + 1u) {
        let j = j_begin + k;
        let jt = joint_type[j];
        let parent_body = joint_parent[j];
        let child = u32(joint_child[j]);
        let axis  = joint_axis[j].xyz;

        var pl: i32 = -1;
        if (parent_body >= 0) {
            for (var kk = 0u; kk < k; kk = kk + 1u) {
                if (joint_child[j_begin + kk] == parent_body) {
                    pl = i32(kk);
                    break;
                }
            }
        }
        parent_local[k] = pl;

        var dw: u32 = 0u;
        if (jt == JOINT_REVOLUTE || jt == JOINT_PRISMATIC) { dw = 1u; }
        dof_local_offset[k] = dof_total;
        dof_local_width[k]  = dw;
        dof_total = dof_total + dw;

        // Joint origin & axis in world.
        var parent_world: Transform;
        if (parent_body < 0) {
            parent_world.position = vec4<f32>(0.0);
            parent_world.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        } else {
            parent_world = body_q[u32(parent_body)];
        }
        let xpw = transform_compose(parent_world, joint_x_p[j]);
        let origin_w = xpw.position.xyz;
        let axis_w = quat_rotate(xpw.rotation, axis);

        let s = jacobian_col_world_origin(jt, axis_w, origin_w);
        s_lin[k] = s.lin;
        s_ang[k] = s.ang;

        // Parent body's spatial velocity / acceleration.
        var vp_l = vec3<f32>(0.0);
        var vp_a = vec3<f32>(0.0);
        var ap_l = vec3<f32>(0.0);
        var ap_a = vec3<f32>(0.0);
        if (pl >= 0) {
            vp_l = v_lin[u32(pl)];
            vp_a = v_ang[u32(pl)];
            ap_l = a_lin[u32(pl)];
            ap_a = a_ang[u32(pl)];
        } else {
            // Root: gravity-loaded acceleration, linear part = -g.
            let world_idx = max(body_world[child], 0);
            let g = gravity[u32(world_idx)].xyz;
            ap_l = -g;
            ap_a = vec3<f32>(0.0);
        }

        // qd / qdd lookup (skip for fixed joints).
        var qd_k: f32 = 0.0;
        var qdd_k: f32 = 0.0;
        if (dw > 0u) {
            let dof_global = joint_dof_offset[j];
            qd_k  = joint_qd[dof_global];
            // qdd indexed locally within this articulation.
            let dof_idx = dof_global - qd_base;
            qdd_k = qdd_in[dof_base + dof_idx];
        }
        let v_j_l = s.lin * qd_k;
        let v_j_a = s.ang * qd_k;
        let vi_l = vp_l + v_j_l;
        let vi_a = vp_a + v_j_a;
        v_lin[k] = vi_l;
        v_ang[k] = vi_a;
        // a_i = a_p + S·qdd + v × S·qd
        let cor = spatial_cross(vi_l, vi_a, v_j_l, v_j_a);
        let ai_l = ap_l + s.lin * qdd_k + cor.lin;
        let ai_a = ap_a + s.ang * qdd_k + cor.ang;
        a_lin[k] = ai_l;
        a_ang[k] = ai_a;

        // Body spatial inertia at world origin.
        let cw = body_q[child];
        let com_world = cw.position.xyz + quat_rotate(cw.rotation, body_com[child].xyz);
        let i_local = mat3_from_cols(body_inertia[child]);
        let i_world = rotate_inertia(cw.rotation, i_local);
        let i_s = smat6_world_origin(body_mass[child], com_world, i_world);

        // f = I·a + v ×* (I·v)
        let i_a_sv = smat6_mul_sv(i_s, ai_l, ai_a);
        let i_v_sv = smat6_mul_sv(i_s, vi_l, vi_a);
        let bias = spatial_cross_dual(vi_l, vi_a, i_v_sv.lin, i_v_sv.ang);
        f_lin[k] = i_a_sv.lin + bias.lin;
        f_ang[k] = i_a_sv.ang + bias.ang;
    }

    // Pass 2 — backward sweep. τ_k = S_kᵀ · f_k; propagate f to parent.
    for (var ki = 0u; ki < n; ki = ki + 1u) {
        let k = n - 1u - ki;
        if (dof_local_width[k] > 0u) {
            let tau_k = dot(s_lin[k], f_lin[k]) + dot(s_ang[k], f_ang[k]);
            tau_out[dof_base + dof_local_offset[k]] = tau_k;
        }
        let pl = parent_local[k];
        if (pl >= 0) {
            let p = u32(pl);
            f_lin[p] = f_lin[p] + f_lin[k];
            f_ang[p] = f_ang[p] + f_ang[k];
        }
    }
}
