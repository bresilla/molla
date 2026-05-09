// Composite Rigid Body Algorithm — joint-space mass matrix M(q).
//
// Mirrors `crates/molla/src/solvers/crba.rs::mass_matrix`. One workgroup
// invocation per articulation; per-thread scratch sized to
// MAX_JOINTS_PER_ART = 64 (matches `featherstone.wgsl`).
//
// Output: dense `N×N` row-major mass matrix per articulation, packed
// into a flat `f32` buffer with per-articulation offsets given by
// `mass_offsets[a..a+1]`. `N` is the articulation's local DOF total
// (revolute + prismatic only — `Fixed` skipped, multi-DOF not yet).
//
// Spatial-vector convention: `[linear; angular]`. Spatial inertia is
// expressed about the world origin via `smat6_world_origin` — the
// CRBA-style helper used by the CPU port.

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
@group(0) @binding(0)  var<storage, read> joint_type:         array<u32>;
@group(0) @binding(1)  var<storage, read> joint_parent:       array<i32>;
@group(0) @binding(2)  var<storage, read> joint_child:        array<i32>;
@group(0) @binding(3)  var<storage, read> joint_x_p:          array<Transform>;
@group(0) @binding(4)  var<storage, read> joint_axis:         array<vec4<f32>>;
@group(0) @binding(5)  var<storage, read> joint_dof_offset:   array<u32>;
@group(0) @binding(6)  var<storage, read> body_mass:          array<f32>;
@group(0) @binding(7)  var<storage, read> body_inertia:       array<Mat3Cols>;
@group(0) @binding(8)  var<storage, read> body_com:           array<vec4<f32>>;
@group(0) @binding(9)  var<storage, read> body_q:             array<Transform>;

// Group 1 — Articulation CSR + output
@group(1) @binding(0) var<storage, read>       art_offsets:  array<u32>;
@group(1) @binding(1) var<storage, read>       mass_offsets: array<u32>;
@group(1) @binding(2) var<storage, read_write> mass_matrix:  array<f32>;

// Group 2 — Uniforms
@group(2) @binding(0) var<uniform> uniforms: Uniforms;

// ─── Quaternion / transform helpers ────────────────────────────────

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

// ─── Spatial Jacobian column at world origin ──────────────────────
// Matches `solvers/crba.rs`: `body_point_world = ZERO`, so the linear
// component reduces to `axis × (-origin)` for revolute, `axis` for
// prismatic. Returned as (lin, ang).

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

// 6×6 spatial inertia at world origin, `[linear; angular]` order.
//   I_s = [ m·I_3       -m·[c]×        ]
//         [ m·[c]×       I_C - m·[c]×² ]
fn smat6_world_origin(mass: f32, c: vec3<f32>, i_world: mat3x3<f32>) -> array<f32, 36> {
    var m: array<f32, 36>;
    for (var i = 0u; i < 36u; i = i + 1u) { m[i] = 0.0; }
    // Top-left m·I.
    m[0u]  = mass; m[7u]  = mass; m[14u] = mass;
    // skew(c).
    let sx00 = 0.0;   let sx01 = -c.z;  let sx02 = c.y;
    let sx10 = c.z;   let sx11 = 0.0;   let sx12 = -c.x;
    let sx20 = -c.y;  let sx21 = c.x;   let sx22 = 0.0;
    // Top-right: -m · skew(c)
    m[3u]  = -mass*sx00; m[4u]  = -mass*sx01; m[5u]  = -mass*sx02;
    m[9u]  = -mass*sx10; m[10u] = -mass*sx11; m[11u] = -mass*sx12;
    m[15u] = -mass*sx20; m[16u] = -mass*sx21; m[17u] = -mass*sx22;
    // Bottom-left: +m · skew(c)
    m[18u] = mass*sx00; m[19u] = mass*sx01; m[20u] = mass*sx02;
    m[24u] = mass*sx10; m[25u] = mass*sx11; m[26u] = mass*sx12;
    m[30u] = mass*sx20; m[31u] = mass*sx21; m[32u] = mass*sx22;
    // skew(c) · skew(c).
    let s2_00 = sx00*sx00 + sx01*sx10 + sx02*sx20;
    let s2_01 = sx00*sx01 + sx01*sx11 + sx02*sx21;
    let s2_02 = sx00*sx02 + sx01*sx12 + sx02*sx22;
    let s2_10 = sx10*sx00 + sx11*sx10 + sx12*sx20;
    let s2_11 = sx10*sx01 + sx11*sx11 + sx12*sx21;
    let s2_12 = sx10*sx02 + sx11*sx12 + sx12*sx22;
    let s2_20 = sx20*sx00 + sx21*sx10 + sx22*sx20;
    let s2_21 = sx20*sx01 + sx21*sx11 + sx22*sx21;
    let s2_22 = sx20*sx02 + sx21*sx12 + sx22*sx22;
    // Bottom-right: I_C − m · s2.  I_world is column-major mat3.
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

fn smat6_mul_sv(m: array<f32, 36>, v_lin: vec3<f32>, v_ang: vec3<f32>) -> array<f32, 6> {
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

fn smat6_add(a: array<f32, 36>, b: array<f32, 36>) -> array<f32, 36> {
    var o: array<f32, 36>;
    for (var i = 0u; i < 36u; i = i + 1u) { o[i] = a[i] + b[i]; }
    return o;
}

// ─── Kernel ────────────────────────────────────────────────────────

@compute @workgroup_size(64)
fn crba_step(@builtin(global_invocation_id) gid: vec3<u32>) {
    let art = gid.x;
    if (art >= uniforms.articulation_count) { return; }
    let j_begin = art_offsets[art];
    let j_end   = art_offsets[art + 1u];
    let n = j_end - j_begin;
    if (n == 0u) { return; }
    if (n > MAX_JOINTS_PER_ART) { return; }

    // Parent joint within articulation (-1 for root or unmatched).
    var parent_local: array<i32, MAX_JOINTS_PER_ART>;
    var s_lin:  array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var s_ang:  array<vec3<f32>, MAX_JOINTS_PER_ART>;
    var dof_local_offset: array<u32, MAX_JOINTS_PER_ART>;
    var dof_local_width:  array<u32, MAX_JOINTS_PER_ART>;
    var i_a:    array<array<f32, 36>, MAX_JOINTS_PER_ART>;

    // Pass 1 — per-joint S, parent linkage, body inertia at world origin.
    var dof_total: u32 = 0u;
    for (var k = 0u; k < n; k = k + 1u) {
        let j = j_begin + k;
        let jt = joint_type[j];
        let parent_body = joint_parent[j];
        let child = u32(joint_child[j]);
        let axis  = joint_axis[j].xyz;

        // Walk earlier joints in this articulation.
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

        // Body spatial inertia at world origin.
        let cw = body_q[child];
        let com_world = cw.position.xyz + quat_rotate(cw.rotation, body_com[child].xyz);
        let i_local = mat3_from_cols(body_inertia[child]);
        let i_world = rotate_inertia(cw.rotation, i_local);
        i_a[k] = smat6_world_origin(body_mass[child], com_world, i_world);
    }

    // Pass 2 — composite-inertia accumulation, leaf to root.
    for (var ki = 0u; ki < n; ki = ki + 1u) {
        let k = n - 1u - ki;
        let pl = parent_local[k];
        if (pl >= 0) {
            i_a[u32(pl)] = smat6_add(i_a[u32(pl)], i_a[k]);
        }
    }

    // Pass 3 — fill M.
    let n_dof = dof_total;
    let m_off = mass_offsets[art];
    // Zero the output tile.
    for (var i = 0u; i < n_dof; i = i + 1u) {
        for (var jj = 0u; jj < n_dof; jj = jj + 1u) {
            mass_matrix[m_off + i * n_dof + jj] = 0.0;
        }
    }
    for (var k = 0u; k < n; k = k + 1u) {
        if (dof_local_width[k] == 0u) { continue; }
        let s_l = s_lin[k];
        let s_a = s_ang[k];
        // F = Iᶜ · S
        let f_arr = smat6_mul_sv(i_a[k], s_l, s_a);
        let f_l = vec3<f32>(f_arr[0u], f_arr[1u], f_arr[2u]);
        let f_a = vec3<f32>(f_arr[3u], f_arr[4u], f_arr[5u]);

        let dof_k = dof_local_offset[k];
        // Diagonal
        let m_kk = dot(s_l, f_l) + dot(s_a, f_a);
        mass_matrix[m_off + dof_k * n_dof + dof_k] = m_kk;

        // Walk ancestors.
        var cur = parent_local[k];
        loop {
            if (cur < 0) { break; }
            let p = u32(cur);
            if (dof_local_width[p] > 0u) {
                let m_kp = dot(s_lin[p], f_l) + dot(s_ang[p], f_a);
                let dof_p = dof_local_offset[p];
                mass_matrix[m_off + dof_k * n_dof + dof_p] = m_kp;
                mass_matrix[m_off + dof_p * n_dof + dof_k] = m_kp;
            }
            cur = parent_local[p];
        }
    }
}
