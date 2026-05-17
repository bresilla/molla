// Hill-type muscle kernel — Stage 42 v5.
//
// Two entry points:
//
//   * `update_muscle_activation` — first-order excitation→activation
//     filter, integrated with implicit Euler so it is unconditionally
//     stable for any `dt > 0`:
//
//         a_{n+1} = (a_n + (dt/τ) · u) / (1 + dt/τ)
//
//     `τ_act ≤ 0` ⇒ instantaneous `a := clamp(u, 0, 1)`.
//
//   * `muscle_torque_sum` — one workgroup invocation per joint. Sums
//     the Hill three-element force across every muscle attached to
//     this joint via the CSR `(joint_muscle_offsets,
//     joint_muscle_indices)`. Writes the result to
//     `joint_muscle_torque[joint_dof_offset[j]]` (only the cable
//     linear-stretch DoF — k == 0). All other entries stay zero so
//     the consumer (featherstone.wgsl) can blindly add this buffer
//     into its per-DOF τ_raw without per-joint type tests.
//
// Mirrors `actuators::muscle::Muscle::torque` and `update_muscle_
// activation` in the CPU path bit-for-bit (modulo f32 precision).

const JOINT_CABLE: u32 = 7u;

const TAU_DEFAULT: f32 = 0.02;
const SIGMA_FL: f32 = 0.45;
const A_F_FV:  f32 = 0.25;
const F_V_MAX: f32 = 1.5;
const K_V_FV:  f32 = 5.0;

// ── Uniforms / shared bindings ─────────────────────────────────────

struct Uniforms {
    muscle_count: u32,
    joint_count:  u32,
    dof_total:    u32,
    dt:           f32,
};

@group(0) @binding(0) var<uniform> uniforms: Uniforms;

// Muscle parameter buffers (length = muscle_count).
@group(1) @binding(0) var<storage, read>       muscle_f_max:           array<f32>;
@group(1) @binding(1) var<storage, read>       muscle_l_opt:           array<f32>;
@group(1) @binding(2) var<storage, read>       muscle_v_max:           array<f32>;
@group(1) @binding(3) var<storage, read_write> muscle_activation:      array<f32>;
@group(1) @binding(4) var<storage, read>       muscle_excitation:      array<f32>;
@group(1) @binding(5) var<storage, read>       muscle_tau_activation:  array<f32>;
@group(1) @binding(6) var<storage, read>       muscle_l_tendon_slack:  array<f32>;
@group(1) @binding(7) var<storage, read>       muscle_pennation_angle: array<f32>;
@group(1) @binding(8) var<storage, read>       muscle_polarity:        array<f32>;
@group(1) @binding(9) var<storage, read>       muscle_tendon_stiffness_ratio: array<f32>;

// Joint / CSR + state (used by torque-sum entry point only, but bound
// for both for layout uniformity).
@group(2) @binding(0) var<storage, read>       joint_type:            array<u32>;
@group(2) @binding(1) var<storage, read>       joint_coord_offset:    array<u32>;
@group(2) @binding(2) var<storage, read>       joint_dof_offset:      array<u32>;
@group(2) @binding(3) var<storage, read>       joint_muscle_offsets:  array<u32>;
@group(2) @binding(4) var<storage, read>       joint_muscle_indices:  array<u32>;
@group(2) @binding(5) var<storage, read>       state_joint_q:         array<f32>;
@group(2) @binding(6) var<storage, read>       state_joint_qd:        array<f32>;

// Output: per-DOF muscle torque. Length = dof_total. Zeroed by the
// host before each dispatch (or by the entry point itself if joint
// has no muscles), then summed in place by each joint's invocation.
@group(3) @binding(0) var<storage, read_write> joint_muscle_torque:   array<f32>;

// ── Hill curves (mirror CPU `force_length` / `force_velocity`) ────

fn force_length(l_norm: f32) -> f32 {
    if (l_norm < 0.4 || l_norm > 1.6) { return 0.0; }
    let x = (l_norm - 1.0) / SIGMA_FL;
    return exp(-x * x);
}

fn force_velocity(v_norm: f32) -> f32 {
    if (v_norm <= -1.0) { return 0.0; }
    if (v_norm < 0.0) {
        return (1.0 + v_norm) / (1.0 - v_norm / A_F_FV);
    }
    return F_V_MAX - (F_V_MAX - 1.0) * exp(-v_norm * K_V_FV);
}

/// Compute muscle torque (single muscle) at cable coords `(q_l, qd_l)`.
/// Mirrors `actuators::muscle::Muscle::torque`. Returns the generalised
/// force contribution to the cable's linear stretch DoF.
fn muscle_force(
    f_max:           f32,
    l_opt:           f32,
    v_max:           f32,
    a:               f32,
    l_tendon_slack:  f32,
    pennation_angle: f32,
    polarity:        f32,
    q_l:             f32,
    qd_l:            f32,
    stiffness_ratio: f32,
) -> f32 {
    let pol = select(-1.0, 1.0, polarity >= 0.0);
    let q_local = pol * q_l;
    let qd_local = pol * qd_l;
    let a_c = clamp(a, 0.0, 1.0);
    if (a_c <= 0.0) { return 0.0; }
    if (!(f_max > 0.0)) { return 0.0; }
    if (!(l_opt > 0.0)) { return 0.0; }

    // Degenerate v_max ⇒ isometric, rigid-tendon. Apply the pennation
    // cos(α) projection, matching the non-degenerate rigid path below
    // and the CPU `Muscle::torque`.
    if (!(v_max > 0.0)) {
        let l_norm0 = 1.0 + q_local / l_opt;
        return pol * (-a_c) * f_max * force_length(l_norm0) * cos(pennation_angle);
    }

    // Rigid-tendon fast path.
    if (l_tendon_slack <= 0.0) {
        let l_norm = (l_opt + q_local) / l_opt;
        let v_norm = qd_local / v_max;
        let f_l = force_length(l_norm);
        let f_v = force_velocity(v_norm);
        let cos_alpha = cos(pennation_angle);
        return pol * (-a_c * f_max * f_l * f_v * cos_alpha);
    }

    // SEE path. Newton solve on tendon strain ε with damped step.
    let cos_alpha = max(cos(pennation_angle), 1e-6);
    let l_rest = l_opt * cos_alpha + l_tendon_slack;
    let l_total = l_rest + q_local;

    let l_t0 = max(l_total - l_opt * cos_alpha, l_tendon_slack);
    let eps_geom = max((l_t0 - l_tendon_slack) / l_tendon_slack, 0.0);
    let ratio = select(30.0, stiffness_ratio, stiffness_ratio > 0.0 && stiffness_ratio == stiffness_ratio);
    let k_tendon = 0.25 * ratio * ratio;
    let eps_analytic = sqrt(max(a_c * cos_alpha / k_tendon, 0.0));
    var eps = select(eps_geom, eps_analytic, eps_geom < eps_analytic * 0.1);

    for (var it = 0u; it < 8u; it = it + 1u) {
        let l_t = l_tendon_slack * (1.0 + eps);
        let l_m_cos = l_total - l_t;
        let l_m = max(l_m_cos / cos_alpha, 1e-9);
        let l_norm = l_m / l_opt;
        let v_norm = qd_local / v_max;
        let f_l = force_length(l_norm);
        let f_v = force_velocity(v_norm);
        let f_fibre = a_c * f_max * f_l * f_v;
        let f_t = f_max * k_tendon * eps * eps;
        let residual = f_fibre * cos_alpha - f_t;
        if (abs(residual) < 1e-4 * f_max) { break; }
        let dlm_deps = -l_tendon_slack / cos_alpha;
        let dln_deps = dlm_deps / l_opt;
        var df_l_deps = 0.0;
        if (l_norm >= 0.4 && l_norm <= 1.6) {
            df_l_deps = -2.0 * (l_norm - 1.0) / (SIGMA_FL * SIGMA_FL) * f_l * dln_deps;
        }
        let df_fibre_deps = a_c * f_max * f_v * df_l_deps;
        let df_t_deps = f_max * 2.0 * k_tendon * max(eps, 1e-3);
        let denom = df_fibre_deps * cos_alpha - df_t_deps;
        if (abs(denom) < 1e-12 * f_max) { break; }
        let step = residual / denom;
        let max_step = (abs(eps) + 1e-3) * 0.5;
        let clamped_step = clamp(step, -max_step, max_step);
        eps = eps - clamped_step;
        if (!(eps == eps) || (abs(eps) > 1e30)) {
            eps = eps_analytic;
            break;
        }
        eps = max(eps, 0.0);
    }
    let f_tend = f_max * k_tendon * eps * eps;
    return pol * (-f_tend);
}

// ── Entry: activation low-pass filter ──────────────────────────────

@compute @workgroup_size(64)
fn update_muscle_activation(@builtin(global_invocation_id) gid: vec3<u32>) {
    let mi = gid.x;
    if (mi >= uniforms.muscle_count) { return; }
    let u_raw = muscle_excitation[mi];
    var u = u_raw;
    if (!(u == u)) {
        u = 0.0;
    }
    u = clamp(u, 0.0, 1.0);
    let tau = muscle_tau_activation[mi];
    let a = muscle_activation[mi];
    var new_a: f32;
    let tau_ok = (tau == tau) && (tau > 0.0);
    if (!tau_ok) {
        new_a = u;
    } else {
        let r = uniforms.dt / tau;
        new_a = (a + r * u) / (1.0 + r);
    }
    muscle_activation[mi] = clamp(new_a, 0.0, 1.0);
}

// ── Entry: per-joint Hill-torque sum ───────────────────────────────

@compute @workgroup_size(64)
fn muscle_torque_sum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if (j >= uniforms.joint_count) { return; }
    let dof_off = joint_dof_offset[j];
    // Always seed this joint's linear-stretch DoF (k == 0) to zero so
    // a stale value from a previous step doesn't carry. The host has
    // already zero-filled the buffer, but if the layout ever grows
    // a second DoF column for muscles this stays safe.
    if (dof_off < uniforms.dof_total) {
        joint_muscle_torque[dof_off] = 0.0;
    }
    if (joint_type[j] != JOINT_CABLE) { return; }

    let lo = joint_muscle_offsets[j];
    let hi = joint_muscle_offsets[j + 1u];
    if (hi <= lo) { return; }

    // Read the joint's linear-stretch coord + velocity once.
    let coord = joint_coord_offset[j];
    let q_l  = state_joint_q[coord];
    let qd_l = state_joint_qd[dof_off];

    var sum: f32 = 0.0;
    for (var s = lo; s < hi; s = s + 1u) {
        let m_id = joint_muscle_indices[s];
        if (m_id >= uniforms.muscle_count) { continue; }
        let f_max  = muscle_f_max[m_id];
        let l_opt  = muscle_l_opt[m_id];
        let v_max  = muscle_v_max[m_id];
        let a_v    = muscle_activation[m_id];
        let lts    = muscle_l_tendon_slack[m_id];
        let pen    = muscle_pennation_angle[m_id];
        let pol    = muscle_polarity[m_id];
        let k_ratio = muscle_tendon_stiffness_ratio[m_id];
        sum = sum + muscle_force(
            f_max, l_opt, v_max, a_v, lts, pen, pol,
            q_l, qd_l, k_ratio,
        );
    }
    joint_muscle_torque[dof_off] = sum;
}
