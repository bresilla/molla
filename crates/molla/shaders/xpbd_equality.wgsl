// XPBD equality-constraint projection — CONNECT (3-DoF positional
// pin) and WELD (6-DoF position + orientation pin). Replaces the
// host-side `apply_equality_post_gpu` shim with a true GPU kernel,
// keeping body_q on the device through the whole step.
//
// Algorithm: same Gauss-Seidel loop as
// `solvers::xpbd::equality_projection_pass`, but running on the
// device. Single-threaded — equality chains (e.g. ropes, equality
// graphs) propagate sequentially through Gauss-Seidel anyway, and
// our typical workload is ≤ 100 constraints, so workgroup_size(1)
// is fine. The big win is eliminating the CPU↔GPU round-trip on
// every substep.
//
// Pipeline order (in `XpbdKernel::dispatch`):
//   predict → project_position × N → **project_equality** →
//   recover_velocity → project_velocity → write_final
//
// `recover_velocity` reads the post-equality `body_q_out` so the
// equality correction shows up in the next-step velocity naturally.

const FLAG_KINEMATIC: u32 = 1u;
const EQ_TYPE_CONNECT: u32     = 0u;
const EQ_TYPE_WELD: u32        = 1u;
const EQ_TYPE_JOINT_MIMIC: u32 = 2u;

struct Transform {
    position: vec4<f32>,
    rotation: vec4<f32>,
};

struct GpuMat3 {
    col0: vec4<f32>,
    col1: vec4<f32>,
    col2: vec4<f32>,
};

struct Uniforms {
    body_count:     u32,
    equality_count: u32,
    iterations:     u32,
    dt:             f32,
};

struct SpatialVec {
    linear:  vec4<f32>,
    angular: vec4<f32>,
};

@group(0) @binding(0) var<storage, read_write> body_q:        array<Transform>;
@group(0) @binding(1) var<storage, read>       body_inv_mass: array<f32>;
@group(0) @binding(2) var<storage, read>       body_inv_inertia: array<GpuMat3>;
@group(0) @binding(3) var<storage, read>       body_com:      array<vec4<f32>>;
@group(0) @binding(4) var<storage, read>       body_flags:    array<u32>;
// Snapshot of `body_q` taken on the host side immediately BEFORE the
// `project_equality` pass runs. Used by `recover_velocity_after_equality`
// to compute the equality-only displacement delta — see that entry
// point's comment for why this is additive (preserving the contact /
// friction / restitution velocity that the main XPBD kernel already
// wrote into `body_qd`).
@group(0) @binding(5) var<storage, read>       body_q_pre_eq: array<Transform>;
@group(0) @binding(6) var<storage, read_write> body_qd:       array<SpatialVec>;

@group(1) @binding(0) var<storage, read> eq_type:               array<u32>;
@group(1) @binding(1) var<storage, read> eq_body_a:             array<i32>;
@group(1) @binding(2) var<storage, read> eq_body_b:             array<i32>;
@group(1) @binding(3) var<storage, read> eq_anchor_a:           array<vec4<f32>>;
@group(1) @binding(4) var<storage, read> eq_anchor_b:           array<vec4<f32>>;
@group(1) @binding(5) var<storage, read> eq_relative_rotation:  array<vec4<f32>>;
@group(1) @binding(6) var<storage, read> eq_enabled:            array<u32>;

@group(2) @binding(0) var<uniform> u: Uniforms;

// ── Joint-mimic bind group (Stage 32) ─────────────────────────────
// Distinct from the body-equality bind groups above so the existing
// `project_equality` / `recover_velocity_after_equality` entry points
// stay binding-layout-compatible with the pre-Stage-32 dispatcher.
// `project_joint_mimic` is the only entry point that touches these.
@group(3) @binding(0) var<storage, read_write> joint_q:                 array<f32>;
@group(3) @binding(1) var<storage, read_write> joint_qd:                array<f32>;
@group(3) @binding(2) var<storage, read>       joint_coord_offset:      array<u32>;
@group(3) @binding(3) var<storage, read>       joint_dof_offset:        array<u32>;
@group(3) @binding(4) var<storage, read>       eq_mimic_driver_joint:   array<i32>;
@group(3) @binding(5) var<storage, read>       eq_mimic_driven_joint:   array<i32>;
@group(3) @binding(6) var<storage, read>       eq_mimic_poly:           array<f32>;

// ── Quaternion + matrix helpers ──────────────────────────────────
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
fn quat_normalize(q: vec4<f32>) -> vec4<f32> {
    let n = length(q);
    return select(vec4<f32>(0.0, 0.0, 0.0, 1.0), q / n, n > 1e-9);
}
/// `q_new = exp(0.5·dθ) ⊗ q_old`, small-angle form to match CPU.
fn apply_rotation_delta(q: vec4<f32>, dtheta: vec3<f32>) -> vec4<f32> {
    let dq = vec4<f32>(0.5 * dtheta.x, 0.5 * dtheta.y, 0.5 * dtheta.z, 0.0);
    return quat_normalize(quat_mul(dq, q) + q);
}
fn mat3_from_cols(g: GpuMat3) -> mat3x3<f32> {
    return mat3x3<f32>(g.col0.xyz, g.col1.xyz, g.col2.xyz);
}
/// World-frame inverse inertia: R · I_body⁻¹ · Rᵀ.
fn world_inv_inertia(rotation: vec4<f32>, body_inv_inertia_mat: mat3x3<f32>) -> mat3x3<f32> {
    // Build the rotation matrix from quaternion.
    let x = rotation.x;
    let y = rotation.y;
    let z = rotation.z;
    let w = rotation.w;
    let xx = x * x; let yy = y * y; let zz = z * z;
    let xy = x * y; let xz = x * z; let yz = y * z;
    let wx = w * x; let wy = w * y; let wz = w * z;
    let r = mat3x3<f32>(
        vec3<f32>(1.0 - 2.0*(yy+zz), 2.0*(xy+wz),     2.0*(xz-wy)),
        vec3<f32>(2.0*(xy-wz),       1.0 - 2.0*(xx+zz), 2.0*(yz+wx)),
        vec3<f32>(2.0*(xz+wy),       2.0*(yz-wx),     1.0 - 2.0*(xx+yy)),
    );
    return r * body_inv_inertia_mat * transpose(r);
}
fn transform_point(xf: Transform, p: vec3<f32>) -> vec3<f32> {
    return xf.position.xyz + quat_rotate(xf.rotation, p);
}
fn com_world(xf: Transform, com: vec3<f32>) -> vec3<f32> {
    return xf.position.xyz + quat_rotate(xf.rotation, com);
}

// ── Single-thread Gauss-Seidel projection. ───────────────────────
@compute @workgroup_size(1)
fn project_equality() {
    if (u.equality_count == 0u) { return; }

    for (var iter: u32 = 0u; iter < u.iterations; iter = iter + 1u) {
        for (var c: u32 = 0u; c < u.equality_count; c = c + 1u) {
            if (eq_enabled[c] == 0u) { continue; }

            let a = eq_body_a[c];
            let b = eq_body_b[c];
            let anchor_a_local = eq_anchor_a[c].xyz;
            let anchor_b_local = eq_anchor_b[c].xyz;

            let a_idx = u32(a);
            let b_idx = u32(b);
            let kin_a = (a < 0) || ((body_flags[a_idx] & FLAG_KINEMATIC) != 0u);
            let kin_b = (b < 0) || ((body_flags[b_idx] & FLAG_KINEMATIC) != 0u);

            var inv_m_a: f32 = 0.0;
            var inv_m_b: f32 = 0.0;
            if (!kin_a) { inv_m_a = body_inv_mass[a_idx]; }
            if (!kin_b) { inv_m_b = body_inv_mass[b_idx]; }
            if (inv_m_a + inv_m_b <= 0.0) { continue; }

            // Read current poses.
            var xf_a: Transform;
            var xf_b: Transform;
            if (a >= 0) { xf_a = body_q[a_idx]; }
            else        { xf_a.position = vec4<f32>(0.0); xf_a.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0); }
            if (b >= 0) { xf_b = body_q[b_idx]; }
            else        { xf_b.position = vec4<f32>(0.0); xf_b.rotation = vec4<f32>(0.0, 0.0, 0.0, 1.0); }

            let p_a = transform_point(xf_a, anchor_a_local);
            let p_b = transform_point(xf_b, anchor_b_local);
            let delta = p_b - p_a;
            let len = length(delta);

            // Lever arms from each body's COM-world.
            var r_a = vec3<f32>(0.0);
            var r_b = vec3<f32>(0.0);
            if (!kin_a) { r_a = p_a - com_world(xf_a, body_com[a_idx].xyz); }
            if (!kin_b) { r_b = p_b - com_world(xf_b, body_com[b_idx].xyz); }

            // Positional correction (CONNECT / first part of WELD).
            if (len > 1e-9) {
                let n_dir = delta / len;
                var i_inv_a = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
                var i_inv_b = mat3x3<f32>(vec3<f32>(0.0), vec3<f32>(0.0), vec3<f32>(0.0));
                if (!kin_a) {
                    i_inv_a = world_inv_inertia(xf_a.rotation, mat3_from_cols(body_inv_inertia[a_idx]));
                }
                if (!kin_b) {
                    i_inv_b = world_inv_inertia(xf_b.rotation, mat3_from_cols(body_inv_inertia[b_idx]));
                }
                let ra_x_n = cross(r_a, n_dir);
                let rb_x_n = cross(r_b, n_dir);
                let w_a = inv_m_a + dot(ra_x_n, i_inv_a * ra_x_n);
                let w_b = inv_m_b + dot(rb_x_n, i_inv_b * rb_x_n);
                let w_sum = w_a + w_b;
                if (w_sum > 0.0) {
                    let lambda = len / w_sum;
                    if (!kin_a) {
                        let new_pos = xf_a.position.xyz + n_dir * (lambda * inv_m_a);
                        let dtheta = i_inv_a * ra_x_n * lambda;
                        let new_rot = apply_rotation_delta(xf_a.rotation, dtheta);
                        body_q[a_idx].position = vec4<f32>(new_pos, 0.0);
                        body_q[a_idx].rotation = new_rot;
                        // Refresh local copy for the WELD step below.
                        xf_a.position = vec4<f32>(new_pos, 0.0);
                        xf_a.rotation = new_rot;
                    }
                    if (!kin_b) {
                        let new_pos = xf_b.position.xyz - n_dir * (lambda * inv_m_b);
                        let dtheta = i_inv_b * rb_x_n * (-lambda);
                        let new_rot = apply_rotation_delta(xf_b.rotation, dtheta);
                        body_q[b_idx].position = vec4<f32>(new_pos, 0.0);
                        body_q[b_idx].rotation = new_rot;
                        xf_b.position = vec4<f32>(new_pos, 0.0);
                        xf_b.rotation = new_rot;
                    }
                }
            }

            // WELD: also align orientations.
            if (eq_type[c] == EQ_TYPE_WELD) {
                let r_target = eq_relative_rotation[c];
                let q_a_now = xf_a.rotation;
                let q_b_now = xf_b.rotation;
                let q_b_target = quat_mul(q_a_now, r_target);
                let dq = quat_mul(q_b_target, quat_conj(q_b_now));
                let sign_w = select(1.0, -1.0, dq.w < 0.0);
                let theta_full = vec3<f32>(
                    dq.x * 2.0 * sign_w,
                    dq.y * 2.0 * sign_w,
                    dq.z * 2.0 * sign_w,
                );
                // Mass-weighted by trace(I_inv).
                var tr_a: f32 = 0.0;
                var tr_b: f32 = 0.0;
                if (!kin_a) {
                    let i_inv = world_inv_inertia(xf_a.rotation, mat3_from_cols(body_inv_inertia[a_idx]));
                    tr_a = i_inv[0].x + i_inv[1].y + i_inv[2].z;
                }
                if (!kin_b) {
                    let i_inv = world_inv_inertia(xf_b.rotation, mat3_from_cols(body_inv_inertia[b_idx]));
                    tr_b = i_inv[0].x + i_inv[1].y + i_inv[2].z;
                }
                let tr_sum = tr_a + tr_b;
                if (tr_sum > 0.0) {
                    let frac_a = tr_a / tr_sum;
                    let frac_b = tr_b / tr_sum;
                    if (!kin_a) {
                        body_q[a_idx].rotation =
                            apply_rotation_delta(xf_a.rotation, -theta_full * frac_a);
                    }
                    if (!kin_b) {
                        body_q[b_idx].rotation =
                            apply_rotation_delta(xf_b.rotation, theta_full * frac_b);
                    }
                }
            }
        }
    }
}

/// Add the equality-projection displacement to `body_qd` so the
/// velocity reflects the equality-corrected pose **without losing the
/// contact / friction / restitution components** that the main XPBD
/// kernel already wrote into `body_qd`.
///
/// **Bug fix (Stage X):** the previous version did
/// `body_qd = (body_q − body_q_in) / dt`, where `body_q_in` was the
/// PRE-STEP pose. That OVERWROTE the post-contact velocity from the
/// main kernel — destroying friction & restitution for every body
/// touched by an equality. Symptom: rope spheres tunnelled through the
/// floor; FourBar bars lost contact friction; arm-with-mimic chains
/// jittered. The CPU `cpu_step` runs the equality projection BEFORE
/// the velocity recovery (see `solvers/xpbd.rs::cpu_step`), so its
/// `(q_post_eq − q_pre_step) / dt` already accounts for the equality.
///
/// Surgical fix: snapshot `body_q` to `body_q_pre_eq` immediately
/// before `project_equality`, then recover by ADDING the
/// equality-only displacement on top of the existing `body_qd`:
///
/// ```text
/// body_qd[i].linear  += (body_q[i].pos - body_q_pre_eq[i].pos) / dt
/// body_qd[i].angular += quat_to_omega(q_now * conj(q_pre_eq)) / dt
/// ```
///
/// For bodies with no equality touching them, the snapshot equals the
/// post-equality pose and the additive term is zero — so non-equality
/// bodies are unaffected. For equality-bearing bodies, the contact
/// velocity from the main kernel is preserved and the equality
/// constraint adds its own velocity correction on top, matching the
/// CPU reference.
@compute @workgroup_size(64)
fn recover_velocity_after_equality(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tid = gid.x;
    if (tid >= u.body_count) { return; }
    if ((body_flags[tid] & FLAG_KINEMATIC) != 0u) { return; }

    let q_now = body_q[tid];
    let q_pre = body_q_pre_eq[tid];
    let inv_dt = select(0.0, 1.0 / u.dt, u.dt > 1e-9);

    let dp = (q_now.position.xyz - q_pre.position.xyz) * inv_dt;
    let dq = quat_mul(q_now.rotation, quat_conj(q_pre.rotation));
    let sign_w = select(1.0, -1.0, dq.w < 0.0);
    let omega = vec3<f32>(
        dq.x * 2.0 * sign_w * inv_dt,
        dq.y * 2.0 * sign_w * inv_dt,
        dq.z * 2.0 * sign_w * inv_dt,
    );

    // Additive — preserves the contact-friction velocity from the main
    // XPBD kernel. See entry-point comment for the bug-fix history.
    var qd = body_qd[tid];
    qd.linear = vec4<f32>(qd.linear.xyz + dp, 0.0);
    qd.angular = vec4<f32>(qd.angular.xyz + omega, 0.0);
    body_qd[tid] = qd;
}

// ── Joint-mimic projection (Stage 32). ───────────────────────────
// Driver-authoritative: snap `joint_q[driven] = poly(joint_q[driver])`
// for every JointMimic equality, then propagate the velocity
// `joint_qd[driven] = poly'(driver) · joint_qd[driver]`. Single-thread
// Gauss-Seidel iteration, like the body equality kernel — joint-mimic
// counts are tiny (≤ a handful per articulation) and chains
// propagate sequentially in WGSL's serial model anyway.
//
// `iterations` re-evaluates the polynomial at the latest driver value
// so a CHAIN of mimics (driver → mid → tail) converges in
// `chain_length` sweeps; for an isolated pair, one sweep is exact.
@compute @workgroup_size(1)
fn project_joint_mimic() {
    if (u.equality_count == 0u) { return; }
    for (var iter: u32 = 0u; iter < u.iterations; iter = iter + 1u) {
        for (var c: u32 = 0u; c < u.equality_count; c = c + 1u) {
            if (eq_type[c] != EQ_TYPE_JOINT_MIMIC) { continue; }
            if (eq_enabled[c] == 0u) { continue; }
            let dr_j_signed = eq_mimic_driver_joint[c];
            let dn_j_signed = eq_mimic_driven_joint[c];
            if (dr_j_signed < 0 || dn_j_signed < 0) { continue; }
            let dr_j = u32(dr_j_signed);
            let dn_j = u32(dn_j_signed);
            let dr_q = joint_coord_offset[dr_j];
            let dn_q = joint_coord_offset[dn_j];
            let dr_dof = joint_dof_offset[dr_j];
            let dn_dof = joint_dof_offset[dn_j];
            // 5 polynomial coefficients per slot, flat layout.
            let base = c * 5u;
            let p0 = eq_mimic_poly[base];
            let p1 = eq_mimic_poly[base + 1u];
            let p2 = eq_mimic_poly[base + 2u];
            let p3 = eq_mimic_poly[base + 3u];
            let p4 = eq_mimic_poly[base + 4u];

            let q_dr = joint_q[dr_q];
            // Horner-form polynomial eval + derivative at q_dr.
            let q_target = (((p4 * q_dr + p3) * q_dr + p2) * q_dr + p1) * q_dr + p0;
            let dp       = ((4.0 * p4 * q_dr + 3.0 * p3) * q_dr + 2.0 * p2) * q_dr + p1;

            joint_q[dn_q] = q_target;
            let qd_dr = joint_qd[dr_dof];
            joint_qd[dn_dof] = dp * qd_dr;
        }
    }
}

