// Stage 23.1.2 — multi-articulation closed-form Jacobian kernel.
//
// One workgroup per articulation; one thread per workgroup (n ≤ 6 keeps
// per-thread registers happy). Each invocation:
//   1. Reads its chain header + per-link params from the global params
//      buffer (offset by `chain_offset[gid.x]`).
//   2. Reads `(q_t, qd_t, qdd_t)` for this chain at this step from the
//      per-step state buffer.
//   3. Builds M(q) via CRBA, inverts it (n ≤ 6 dense Cholesky), and
//      writes M⁻¹ to `qdd_tau` (∂q̈/∂τ block).
//   4. Builds dh/dq, dh/dqd via central FD on RNEA at `qdd_t` (the
//      Carpentier-Mansard chain rule), then writes
//      `qdd_q  = -M⁻¹ · dh/dq` and `qdd_qd = -M⁻¹ · dh/dqd`.
//
// **This kernel ships the closed-form path on GPU** — same algorithm
// the CPU `fill_jacobians_closed_form` runs. The pure-analytic
// `∂h/∂qd` chain-rule sweep (Stage 23.1.2's CPU win) currently lives
// only on CPU; porting the analytic sweep to WGSL is Stage 23.1.3
// follow-up (the per-thread register pressure for the qd sweep is
// non-trivial and the closed-form GPU path already wins via batching).
//
// Per-articulation buffer layout (stride = MAX_LINKS² × 3):
//   qdd_tau  : MAX_LINKS² floats  (∂q̈/∂τ = M⁻¹)
//   qdd_q    : MAX_LINKS² floats  (-M⁻¹ · ∂h/∂q)
//   qdd_qd   : MAX_LINKS² floats  (-M⁻¹ · ∂h/∂qd)
// Each block is row-major n × n with n = chain.n_links; cells past
// `n` in any direction are left untouched (callers slice by n).
//
// CPU mirror: `crate::kernels::featherstone_adjoint::FeatherstoneAdjointDispatcher::reverse_with_strategy`
// at `JacobianStrategy::ClosedForm` for the per-step block content.

const MAX_LINKS: u32 = 8u;
const MAX_LINKS2: u32 = 64u;            // MAX_LINKS * MAX_LINKS
const FD_EPS: f32 = 1.0e-3f;            // f32 sweet-spot for central FD on RNEA

struct LinkParams {
    // 0 = Revolute, 1 = Prismatic.  (Ball / Free are stacked single-DoF
    // upstream — the kernel never sees a multi-DoF kind directly.)
    kind: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
    axis_local: vec3<f32>,
    mass: f32,
    x_p_pos: vec3<f32>,
    _pad3: f32,
    x_p_rot: vec4<f32>,
    com_local: vec3<f32>,
    _pad4: f32,
    inertia_local_row0: vec3<f32>,
    _pad5: f32,
    inertia_local_row1: vec3<f32>,
    _pad6: f32,
    inertia_local_row2: vec3<f32>,
    _pad7: f32,
}

struct ChainHeader {
    n_links: u32,
    link_offset: u32,    // index into g_links where this chain's links start
    state_offset: u32,   // index into g_qstate where this chain's q/qd/qdd start
    out_offset: u32,     // index into g_out_jac where this chain's blocks start (in floats; stride = MAX_LINKS² * 3)
    gravity: vec3<f32>,
    damping: f32,
}

@group(0) @binding(0) var<storage, read> g_chains: array<ChainHeader>;
@group(0) @binding(1) var<storage, read> g_links: array<LinkParams>;
// Per-chain state buffer: (q, qd, qdd) packed sequentially per chain,
// each block of MAX_LINKS floats.
@group(0) @binding(2) var<storage, read> g_qstate: array<f32>;
// Per-chain output: 3 × MAX_LINKS² floats — qdd_tau / qdd_q / qdd_qd.
@group(0) @binding(3) var<storage, read_write> g_out_jac: array<f32>;
@group(1) @binding(0) var<uniform> g_uniforms: Uniforms;

struct Uniforms {
    chain_count: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// ──────────────────────────────────────────────────────────────────
// Quaternion / transform helpers (f32 mirrors of math::Transform).
// ──────────────────────────────────────────────────────────────────

fn quat_mul(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    return vec4<f32>(
        a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
        a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
    );
}

fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    let n = sqrt(dot(q, q));
    if (n > 1.0e-20f) {
        return q / n;
    }
    return vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
}

fn quat_from_axis_angle(axis: vec3<f32>, angle: f32) -> vec4<f32> {
    let half = angle * 0.5f;
    let s = sin(half);
    let c = cos(half);
    let an = normalize(axis);
    return vec4<f32>(an.x * s, an.y * s, an.z * s, c);
}

fn quat_rotate_vec(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
    let qv = vec3<f32>(q.x, q.y, q.z);
    let t = 2.0f * cross(qv, v);
    return v + q.w * t + cross(qv, t);
}

fn quat_to_rot_mat(q: vec4<f32>) -> mat3x3<f32> {
    let x = q.x; let y = q.y; let z = q.z; let w = q.w;
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;
    return mat3x3<f32>(
        vec3<f32>(1.0f - 2.0f * (yy + zz),       2.0f * (xy + wz),       2.0f * (xz - wy)),
        vec3<f32>(      2.0f * (xy - wz), 1.0f - 2.0f * (xx + zz),       2.0f * (yz + wx)),
        vec3<f32>(      2.0f * (xz + wy),       2.0f * (yz - wx), 1.0f - 2.0f * (xx + yy)),
    );
}

// ──────────────────────────────────────────────────────────────────
// Per-chain inline scratch.  All thread-local arrays sized to
// MAX_LINKS so the WGSL register allocator can flatten them.
// ──────────────────────────────────────────────────────────────────

// Spatial vector packed as 6 floats: [linear; angular].
struct SV { v: array<f32, 6> }

fn sv_zero() -> SV { return SV(array<f32, 6>(0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)); }

fn sv_add(a: SV, b: SV) -> SV {
    var r: SV;
    for (var i = 0u; i < 6u; i = i + 1u) { r.v[i] = a.v[i] + b.v[i]; }
    return r;
}

fn sv_scale(a: SV, s: f32) -> SV {
    var r: SV;
    for (var i = 0u; i < 6u; i = i + 1u) { r.v[i] = a.v[i] * s; }
    return r;
}

fn sv_dot(a: SV, b: SV) -> f32 {
    var r: f32 = 0.0f;
    for (var i = 0u; i < 6u; i = i + 1u) { r = r + a.v[i] * b.v[i]; }
    return r;
}

fn sv_cross_motion(a: SV, b: SV) -> SV {
    // Motion x: lin = a_ang × b_lin + a_lin × b_ang ; ang = a_ang × b_ang.
    let a_lin = vec3<f32>(a.v[0], a.v[1], a.v[2]);
    let a_ang = vec3<f32>(a.v[3], a.v[4], a.v[5]);
    let b_lin = vec3<f32>(b.v[0], b.v[1], b.v[2]);
    let b_ang = vec3<f32>(b.v[3], b.v[4], b.v[5]);
    let lin = cross(a_ang, b_lin) + cross(a_lin, b_ang);
    let ang = cross(a_ang, b_ang);
    return SV(array<f32, 6>(lin.x, lin.y, lin.z, ang.x, ang.y, ang.z));
}

fn sv_cross_dual(a: SV, b: SV) -> SV {
    // Force x* : lin = a_ang × b_lin ; ang = a_ang × b_ang + a_lin × b_lin.
    let a_lin = vec3<f32>(a.v[0], a.v[1], a.v[2]);
    let a_ang = vec3<f32>(a.v[3], a.v[4], a.v[5]);
    let b_lin = vec3<f32>(b.v[0], b.v[1], b.v[2]);
    let b_ang = vec3<f32>(b.v[3], b.v[4], b.v[5]);
    let lin = cross(a_ang, b_lin);
    let ang = cross(a_ang, b_ang) + cross(a_lin, b_lin);
    return SV(array<f32, 6>(lin.x, lin.y, lin.z, ang.x, ang.y, ang.z));
}

// Body spatial inertia 6×6, stored row-major as 36 floats (matches
// math::SpatialMatrix6 packing in spatial.rs).
struct SI {
    rows: array<f32, 36>,
}

fn si_mul_sv(m: SI, v: SV) -> SV {
    var r: SV;
    for (var i = 0u; i < 6u; i = i + 1u) {
        var s: f32 = 0.0f;
        for (var j = 0u; j < 6u; j = j + 1u) {
            s = s + m.rows[i * 6u + j] * v.v[j];
        }
        r.v[i] = s;
    }
    return r;
}

fn si_zero() -> SI {
    var s: SI;
    for (var i = 0u; i < 36u; i = i + 1u) { s.rows[i] = 0.0f; }
    return s;
}

fn si_add(a: SI, b: SI) -> SI {
    var r: SI;
    for (var i = 0u; i < 36u; i = i + 1u) { r.rows[i] = a.rows[i] + b.rows[i]; }
    return r;
}

// Build SpatialMatrix6 from (mass, com_world, inertia_world).
// Mirrors `SpatialMatrix6::from_body` in math/spatial.rs.
fn si_from_body(mass: f32, com: vec3<f32>, i_world: mat3x3<f32>) -> SI {
    // Layout: [m·I3, m·skew(com)ᵀ ; m·skew(com), I_world − m·skew(com)·skew(com)]
    // (matches the host-side spatial inertia composition).
    var m: SI = si_zero();
    // Top-left = m·I3.
    m.rows[0u * 6u + 0u] = mass;
    m.rows[1u * 6u + 1u] = mass;
    m.rows[2u * 6u + 2u] = mass;
    // Top-right = -m · skew(com) (sign matches host: −skew · m for ω→linear).
    let cx = com.x; let cy = com.y; let cz = com.z;
    m.rows[0u * 6u + 4u] = -mass * cz;
    m.rows[0u * 6u + 5u] =  mass * cy;
    m.rows[1u * 6u + 3u] =  mass * cz;
    m.rows[1u * 6u + 5u] = -mass * cx;
    m.rows[2u * 6u + 3u] = -mass * cy;
    m.rows[2u * 6u + 4u] =  mass * cx;
    // Bottom-left = -m · skew(com) (host convention; skew is antisymmetric → -skew = skewᵀ).
    m.rows[3u * 6u + 1u] =  mass * cz;
    m.rows[3u * 6u + 2u] = -mass * cy;
    m.rows[4u * 6u + 0u] = -mass * cz;
    m.rows[4u * 6u + 2u] =  mass * cx;
    m.rows[5u * 6u + 0u] =  mass * cy;
    m.rows[5u * 6u + 1u] = -mass * cx;
    // Bottom-right = I_world + m · (com·comᵀ · I − com·comᵀ ?) — simpler:
    // I_world − m · skew(com) · skew(com).  Compute skew·skew first.
    // skew(c) · skew(c) = c·cᵀ − |c|² · I (standard identity).
    let cc = dot(com, com);
    let cct = mat3x3<f32>(
        vec3<f32>(cx * cx, cy * cx, cz * cx),
        vec3<f32>(cx * cy, cy * cy, cz * cy),
        vec3<f32>(cx * cz, cy * cz, cz * cz),
    );
    // skew·skew = c cᵀ − cc · I.
    let ss = mat3x3<f32>(
        cct[0] - vec3<f32>(cc, 0.0f, 0.0f),
        cct[1] - vec3<f32>(0.0f, cc, 0.0f),
        cct[2] - vec3<f32>(0.0f, 0.0f, cc),
    );
    // I_world − mass · ss.
    let i_corr = mat3x3<f32>(
        i_world[0] - mass * ss[0],
        i_world[1] - mass * ss[1],
        i_world[2] - mass * ss[2],
    );
    // Pack into bottom-right 3×3.
    for (var i = 0u; i < 3u; i = i + 1u) {
        for (var j = 0u; j < 3u; j = j + 1u) {
            m.rows[(3u + i) * 6u + (3u + j)] = i_corr[j][i];
        }
    }
    return m;
}

// ──────────────────────────────────────────────────────────────────
// Forward-kinematics one chain link: returns (body_world_pos,
// body_world_rot, S column, joint_axis_world, com_world).
// ──────────────────────────────────────────────────────────────────

struct Kin {
    body_pos: vec3<f32>,
    body_rot: vec4<f32>,
    s: SV,
    com_world: vec3<f32>,
    i_world: mat3x3<f32>,
}

fn link_kinematics(parent_pos: vec3<f32>, parent_rot: vec4<f32>, link: LinkParams, q_j: f32) -> Kin {
    // parent_joint = parent_t · x_p
    let pj_pos = parent_pos + quat_rotate_vec(parent_rot, link.x_p_pos);
    let pj_rot = quat_normalize(quat_mul(parent_rot, link.x_p_rot));

    var body_pos: vec3<f32> = pj_pos;
    var body_rot: vec4<f32> = pj_rot;
    var s_lin: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var s_ang: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var axis_world: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    if (link.kind == 0u) {
        // Revolute
        let q_rot = quat_from_axis_angle(link.axis_local, q_j);
        body_rot = quat_normalize(quat_mul(pj_rot, q_rot));
        axis_world = quat_rotate_vec(pj_rot, link.axis_local);
        s_ang = axis_world;
    } else {
        // Prismatic
        axis_world = quat_rotate_vec(pj_rot, link.axis_local);
        body_pos = pj_pos + axis_world * q_j;
        body_rot = pj_rot;
        s_lin = axis_world;
    }
    let rm = quat_to_rot_mat(body_rot);
    let inertia_local = mat3x3<f32>(
        link.inertia_local_row0,
        link.inertia_local_row1,
        link.inertia_local_row2,
    );
    let i_world = rm * inertia_local * transpose(rm);
    let com_world = body_pos + rm * link.com_local;
    var s_v: SV;
    s_v.v[0] = s_lin.x; s_v.v[1] = s_lin.y; s_v.v[2] = s_lin.z;
    s_v.v[3] = s_ang.x; s_v.v[4] = s_ang.y; s_v.v[5] = s_ang.z;
    return Kin(body_pos, body_rot, s_v, com_world, i_world);
}

// ──────────────────────────────────────────────────────────────────
// CRBA: build M(q) ∈ ℝⁿˣⁿ for this articulation, n ≤ MAX_LINKS.
// Output stored in a thread-local n×n array.
// ──────────────────────────────────────────────────────────────────

struct MassMat {
    m: array<f32, MAX_LINKS2>,
}

fn fill_mass_matrix(header: ChainHeader, q: ptr<function, array<f32, MAX_LINKS>>) -> MassMat {
    let n = header.n_links;
    var s_cols: array<SV, MAX_LINKS>;
    var i_c: array<SI, MAX_LINKS>;
    var parent_pos: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var parent_rot: vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    for (var j = 0u; j < n; j = j + 1u) {
        let link = g_links[header.link_offset + j];
        let kin = link_kinematics(parent_pos, parent_rot, link, (*q)[j]);
        s_cols[j] = kin.s;
        i_c[j] = si_from_body(link.mass, kin.com_world, kin.i_world);
        parent_pos = kin.body_pos;
        parent_rot = kin.body_rot;
    }
    // Backward composite-inertia accumulation.
    if (n > 1u) {
        for (var jj = n - 1u; jj > 0u; jj = jj - 1u) {
            let i_j = i_c[jj];
            i_c[jj - 1u] = si_add(i_c[jj - 1u], i_j);
        }
    }
    var out: MassMat;
    for (var i = 0u; i < MAX_LINKS2; i = i + 1u) { out.m[i] = 0.0f; }
    for (var i = 0u; i < n; i = i + 1u) {
        for (var j = 0u; j < n; j = j + 1u) {
            // r = max(i, j)
            var r: u32 = i;
            if (j > i) { r = j; }
            let f = si_mul_sv(i_c[r], s_cols[j]);
            out.m[i * MAX_LINKS + j] = sv_dot(s_cols[i], f);
        }
    }
    return out;
}

// Cholesky-invert a symmetric positive-definite n×n matrix laid out
// in `MAX_LINKS × MAX_LINKS` with non-zero entries only in the top-
// left n×n. Returns inverse with the same layout. n ≤ MAX_LINKS.
//
// On Cholesky breakdown returns identity (gradient survives, signal
// is silent — same fallback policy as the CPU `invert_spd`).
fn invert_spd_inline(mat: MassMat, n: u32) -> MassMat {
    // In-place factorisation into L (lower triangle).
    var a: MassMat = mat;
    for (var i = 0u; i < n; i = i + 1u) {
        for (var j = 0u; j <= i; j = j + 1u) {
            var s = a.m[i * MAX_LINKS + j];
            for (var k = 0u; k < j; k = k + 1u) {
                s = s - a.m[i * MAX_LINKS + k] * a.m[j * MAX_LINKS + k];
            }
            if (i == j) {
                if (s <= 1.0e-20f) {
                    // Breakdown — return identity.
                    var id: MassMat;
                    for (var ii = 0u; ii < MAX_LINKS2; ii = ii + 1u) { id.m[ii] = 0.0f; }
                    for (var ii = 0u; ii < n; ii = ii + 1u) { id.m[ii * MAX_LINKS + ii] = 1.0f; }
                    return id;
                }
                a.m[i * MAX_LINKS + i] = sqrt(s);
            } else {
                a.m[i * MAX_LINKS + j] = s / a.m[j * MAX_LINKS + j];
            }
        }
    }
    // Solve M · x_k = e_k for each column k.
    var inv: MassMat;
    for (var ii = 0u; ii < MAX_LINKS2; ii = ii + 1u) { inv.m[ii] = 0.0f; }
    var y: array<f32, MAX_LINKS>;
    var x: array<f32, MAX_LINKS>;
    for (var k = 0u; k < n; k = k + 1u) {
        for (var i = 0u; i < n; i = i + 1u) {
            var s: f32;
            if (i == k) { s = 1.0f; } else { s = 0.0f; }
            for (var jj = 0u; jj < i; jj = jj + 1u) {
                s = s - a.m[i * MAX_LINKS + jj] * y[jj];
            }
            y[i] = s / a.m[i * MAX_LINKS + i];
        }
        // Back-sub on Lᵀ. Iterate i from n-1 down to 0.
        var i: u32 = n;
        loop {
            if (i == 0u) { break; }
            i = i - 1u;
            var s = y[i];
            for (var kk = i + 1u; kk < n; kk = kk + 1u) {
                s = s - a.m[kk * MAX_LINKS + i] * x[kk];
            }
            x[i] = s / a.m[i * MAX_LINKS + i];
        }
        for (var jj = 0u; jj < n; jj = jj + 1u) {
            inv.m[jj * MAX_LINKS + k] = x[jj];
        }
    }
    return inv;
}

// ──────────────────────────────────────────────────────────────────
// Recursive Newton-Euler: τ = M·q̈ + h(q,qd). Mirrors the CPU
// `inverse_dynamics` in featherstone_adjoint.rs.
// ──────────────────────────────────────────────────────────────────

fn rnea(
    header: ChainHeader,
    q: ptr<function, array<f32, MAX_LINKS>>,
    qd: ptr<function, array<f32, MAX_LINKS>>,
    qdd: ptr<function, array<f32, MAX_LINKS>>,
    out_tau: ptr<function, array<f32, MAX_LINKS>>,
) {
    let n = header.n_links;
    var s_cols: array<SV, MAX_LINKS>;
    var f_net: array<SV, MAX_LINKS>;
    var parent_pos: vec3<f32> = vec3<f32>(0.0f, 0.0f, 0.0f);
    var parent_rot: vec4<f32> = vec4<f32>(0.0f, 0.0f, 0.0f, 1.0f);
    var v_parent: SV = sv_zero();
    var a_parent: SV = sv_zero();
    let g = header.gravity;
    for (var j = 0u; j < n; j = j + 1u) {
        let link = g_links[header.link_offset + j];
        let kin = link_kinematics(parent_pos, parent_rot, link, (*q)[j]);
        let s_j = kin.s;
        s_cols[j] = s_j;
        let v_j = sv_scale(s_j, (*qd)[j]);
        let v_i = sv_add(v_parent, v_j);
        let a_j_local = sv_scale(s_j, (*qdd)[j]);
        let c_j = sv_cross_motion(sv_add(v_parent, v_j), v_j);
        let a_i = sv_add(sv_add(a_parent, a_j_local), c_j);

        let i_s = si_from_body(link.mass, kin.com_world, kin.i_world);
        let i_a = si_mul_sv(i_s, a_i);
        let bias = sv_cross_dual(v_i, si_mul_sv(i_s, v_i));
        // f_grav as a wrench at the COM.
        let f_g_lin = g * link.mass;
        let f_g_ang = cross(kin.com_world, f_g_lin);
        let f_grav = SV(array<f32, 6>(f_g_lin.x, f_g_lin.y, f_g_lin.z, f_g_ang.x, f_g_ang.y, f_g_ang.z));
        var fnet: SV;
        for (var ii = 0u; ii < 6u; ii = ii + 1u) {
            fnet.v[ii] = i_a.v[ii] + bias.v[ii] - f_grav.v[ii];
        }
        f_net[j] = fnet;

        parent_pos = kin.body_pos;
        parent_rot = kin.body_rot;
        v_parent = v_i;
        a_parent = a_i;
    }

    // Backward sweep: propagate child wrenches up + project.
    var f_acc: array<SV, MAX_LINKS>;
    for (var j = 0u; j < n; j = j + 1u) { f_acc[j] = sv_zero(); }
    var jj: u32 = n;
    loop {
        if (jj == 0u) { break; }
        jj = jj - 1u;
        let f_total = sv_add(f_net[jj], f_acc[jj]);
        (*out_tau)[jj] = sv_dot(s_cols[jj], f_total);
        if (jj > 0u) {
            f_acc[jj - 1u] = sv_add(f_acc[jj - 1u], f_total);
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Main entry point: one workgroup per articulation.
// ──────────────────────────────────────────────────────────────────

@compute @workgroup_size(1)
fn jacobians(@builtin(global_invocation_id) gid: vec3<u32>) {
    let chain_idx = gid.x;
    if (chain_idx >= g_uniforms.chain_count) { return; }
    let header = g_chains[chain_idx];
    let n = header.n_links;
    let so = header.state_offset;
    let oo = header.out_offset;

    // Load q, qd, qdd into per-thread scratch.
    var q: array<f32, MAX_LINKS>;
    var qd: array<f32, MAX_LINKS>;
    var qdd: array<f32, MAX_LINKS>;
    for (var i = 0u; i < n; i = i + 1u) {
        q[i] = g_qstate[so + i];
        qd[i] = g_qstate[so + MAX_LINKS + i];
        qdd[i] = g_qstate[so + 2u * MAX_LINKS + i];
    }
    for (var i = n; i < MAX_LINKS; i = i + 1u) {
        q[i] = 0.0f; qd[i] = 0.0f; qdd[i] = 0.0f;
    }

    // 1) M(q) via CRBA + invert.
    let m_mat = fill_mass_matrix(header, &q);
    let inv_m = invert_spd_inline(m_mat, n);
    // Write qdd_tau = M⁻¹.
    for (var j = 0u; j < n; j = j + 1u) {
        for (var k = 0u; k < n; k = k + 1u) {
            g_out_jac[oo + j * MAX_LINKS + k] = inv_m.m[j * MAX_LINKS + k];
        }
    }

    // 2) dh/dq via central FD on RNEA at qdd_t.
    var did_dq: array<f32, MAX_LINKS2>;
    for (var i = 0u; i < MAX_LINKS2; i = i + 1u) { did_dq[i] = 0.0f; }
    var tau_p: array<f32, MAX_LINKS>;
    var tau_m: array<f32, MAX_LINKS>;
    for (var k = 0u; k < n; k = k + 1u) {
        let saved = q[k];
        q[k] = saved + FD_EPS;
        rnea(header, &q, &qd, &qdd, &tau_p);
        q[k] = saved - FD_EPS;
        rnea(header, &q, &qd, &qdd, &tau_m);
        q[k] = saved;
        let inv_two_eps = 1.0f / (2.0f * FD_EPS);
        for (var j = 0u; j < n; j = j + 1u) {
            did_dq[j * MAX_LINKS + k] = (tau_p[j] - tau_m[j]) * inv_two_eps;
        }
    }
    // qdd_q = -inv_m · did_dq, written into the second n×n block.
    let qdd_q_off = oo + MAX_LINKS2;
    for (var j = 0u; j < n; j = j + 1u) {
        for (var k = 0u; k < n; k = k + 1u) {
            var s: f32 = 0.0f;
            for (var r = 0u; r < n; r = r + 1u) {
                s = s + inv_m.m[j * MAX_LINKS + r] * did_dq[r * MAX_LINKS + k];
            }
            g_out_jac[qdd_q_off + j * MAX_LINKS + k] = -s;
        }
    }

    // 3) dh/dqd via central FD on RNEA.
    var did_dqd: array<f32, MAX_LINKS2>;
    for (var i = 0u; i < MAX_LINKS2; i = i + 1u) { did_dqd[i] = 0.0f; }
    for (var k = 0u; k < n; k = k + 1u) {
        let saved = qd[k];
        qd[k] = saved + FD_EPS;
        rnea(header, &q, &qd, &qdd, &tau_p);
        qd[k] = saved - FD_EPS;
        rnea(header, &q, &qd, &qdd, &tau_m);
        qd[k] = saved;
        let inv_two_eps = 1.0f / (2.0f * FD_EPS);
        for (var j = 0u; j < n; j = j + 1u) {
            did_dqd[j * MAX_LINKS + k] = (tau_p[j] - tau_m[j]) * inv_two_eps;
        }
    }
    let qdd_qd_off = oo + 2u * MAX_LINKS2;
    for (var j = 0u; j < n; j = j + 1u) {
        for (var k = 0u; k < n; k = k + 1u) {
            var s: f32 = 0.0f;
            for (var r = 0u; r < n; r = r + 1u) {
                s = s + inv_m.m[j * MAX_LINKS + r] * did_dqd[r * MAX_LINKS + k];
            }
            g_out_jac[qdd_qd_off + j * MAX_LINKS + k] = -s;
        }
    }
}
