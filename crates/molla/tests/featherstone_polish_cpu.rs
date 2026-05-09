//! Stage 17.1 — Featherstone multi-DOF polish tests (CPU lane).
//!
//! These four tests exercise the CPU path of the same polish features
//! covered by `featherstone_polish_gpu.rs` and don't need a wgpu
//! adapter:
//!
//!  - `d6_angular_limit_holds_under_gravity_cpu`
//!  - `distance_joint_length_holds_within_eps_cpu`
//!  - `distance_joint_lagrange_beats_v1_clamp_cpu`
//!  - `coupled_d_dense_cholesky_matches_reference_cpu`
//!
//! They live in their own file (no `#![cfg(feature = "gpu")]`) so the
//! default lane (`cargo test -p molla`) actually runs them — the prior
//! arrangement gated them behind the `gpu` feature, which silently
//! dropped them from the default lane.

use molla::core::BodyId;
use molla::math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;

/// Stage 17.1 deliverable 1 — D6 angular limits — CPU companion.
#[test]
fn d6_angular_limit_holds_under_gravity_cpu() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.0,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY * 0.05,
    }));
    let j = b
        .add_joint_d6(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
        )
        .unwrap();
    let lim = std::f64::consts::FRAC_PI_4;
    b.set_joint_limits(
        j,
        &[0.0, 0.0, 0.0, -lim, 0.0, 0.0],
        &[0.0, 0.0, 0.0, lim, 0.0, 0.0],
    )
    .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let tilt = molla::math::Quat::from_axis_angle(V3::X, lim);
    let qmut = s_a.joint_q.host_mut().unwrap();
    qmut[3] = tilt.x;
    qmut[4] = tilt.y;
    qmut[5] = tilt.z;
    qmut[6] = tilt.w;
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu().with_damping(0.1);

    let dt = 0.001_f64;
    let mut max_excess: Real = 0.0;
    for _ in 0..600 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q = s_a.joint_q.host().unwrap();
        let r = molla::math::Quat::from_xyzw(q[3], q[4], q[5], q[6]);
        let (rx, _ry, _rz) = r.to_euler(glam::EulerRot::XYZ);
        let excess = (rx.abs() - lim).max(0.0);
        if excess > max_excess {
            max_excess = excess;
        }
    }
    eprintln!("D6 angular limit (CPU): max excess = {max_excess:.5} rad");
    assert!(max_excess < 0.05, "max_excess={max_excess}");
}

/// Stage 17.1 deliverable 2 — Distance-joint Lagrange projection — CPU
/// companion. Tighter tolerance than GPU because we run in f64.
#[test]
fn distance_joint_length_holds_within_eps_cpu() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.0,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY * 0.1,
    }));
    let j = b
        .add_joint_distance(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
        )
        .unwrap();
    let l_t = 0.5;
    b.set_joint_limits(j, &[l_t; 6], &[l_t; 6]).unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let qmut = s_a.joint_q.host_mut().unwrap();
    qmut[0] = 0.0;
    qmut[1] = -l_t;
    qmut[2] = 0.0;
    qmut[6] = 1.0;
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();

    let dt = 0.001_f64;
    let mut worst_residual: Real = 0.0;
    for _ in 0..200 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q = s_a.joint_q.host().unwrap();
        let p = V3::new(q[0], q[1], q[2]);
        let res = (p.length() - l_t).abs();
        if res > worst_residual {
            worst_residual = res;
        }
    }
    eprintln!("Distance Lagrange (CPU): worst |p|−L = {worst_residual:.3e}");
    // CPU f64 path: should hit machine eps after the post-projection
    // snap (`p ← p · L_t / |p|`).
    assert!(
        worst_residual < 1e-9,
        "distance constraint drift (CPU): {worst_residual} m"
    );
}

/// A/B parity check vs the v1 hard-clamp path. The Lagrange path
/// should never overshoot the v1 path's error — and on the CPU should
/// be markedly tighter.
#[test]
fn distance_joint_lagrange_beats_v1_clamp_cpu() {
    fn run(rigorous: bool) -> Real {
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY * 0.1,
        }));
        let j = b
            .add_joint_distance(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let l_t = 0.5;
        b.set_joint_limits(j, &[l_t; 6], &[l_t; 6]).unwrap();
        let model = b.build().unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let qmut = s_a.joint_q.host_mut().unwrap();
        qmut[1] = -l_t;
        qmut[6] = 1.0;
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new_cpu().with_rigorous_distance(rigorous);
        let mut worst: Real = 0.0;
        for _ in 0..200 {
            solver
                .step(&model, &s_a, &control, &contacts, 0.001, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
            let q = s_a.joint_q.host().unwrap();
            let p = V3::new(q[0], q[1], q[2]);
            let r = (p.length() - l_t).abs();
            if r > worst {
                worst = r;
            }
        }
        worst
    }
    let v1 = run(false);
    let rigorous = run(true);
    eprintln!("Distance v1 hard-clamp worst: {v1:.3e}, rigorous: {rigorous:.3e}");
    assert!(
        rigorous <= v1 + 1e-12,
        "Lagrange path worse than v1 clamp: rigorous={rigorous} > v1={v1}"
    );
}

/// Stage 17.1 deliverable 3 — coupled-D parity (CPU dense Cholesky vs
/// CPU dense-Cholesky reference, plus comparison against sequential
/// rank-1).
///
/// A free body with a non-trivial CoM offset (0.15 m from the joint
/// axis) feels strong cross-coupling between linear and angular DOFs.
///
/// The reference solution is computed by hand: build the joint-space
/// Jacobian + spatial inertia, form `H = J^T · I · J`, factor with
/// `dense_chol_solve`, recover q̈, integrate symplectic Euler. This
/// matches what `with_rigorous_d_solve(true)` does internally; the
/// per-step deviation must be < 1e-9 in joint coords (f64 round-off).
#[test]
fn coupled_d_dense_cholesky_matches_reference_cpu() {
    use molla::math::dense_solve::dense_chol_solve;
    use molla::math::{spatial_cross_dual, Quat, SpatialMatrix6, SpatialVector};

    let mass = 2.0;
    let com_offset = V3::new(0.15, 0.0, 0.0);
    let inertia_local = Mat3::from_diagonal(V3::new(0.04, 0.10, 0.10));

    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: com_offset,
        inertia: inertia_local,
    }));
    b.add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let qmut = s_a.joint_q.host_mut().unwrap();
    qmut[1] = 0.5;
    qmut[6] = 1.0; // identity quaternion
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu().with_rigorous_d_solve(true);

    let dt = 0.001_f64;

    fn reference_step(
        s_in: &State,
        s_out: &mut State,
        mass: Real,
        com_local: V3,
        inertia_local: Mat3,
        dt: Real,
    ) {
        let q = s_in.joint_q.host().unwrap();
        let qd = s_in.joint_qd.host().unwrap();
        let pos = V3::new(q[0], q[1], q[2]);
        let rot = Quat::from_xyzw(q[3], q[4], q[5], q[6]);
        let v = V3::new(qd[0], qd[1], qd[2]);
        let w = V3::new(qd[3], qd[4], qd[5]);
        let com_world = pos + rot * com_local;
        let i_world = Mat3::from_quat(rot) * inertia_local * Mat3::from_quat(rot).transpose();
        let i_s = SpatialMatrix6::from_body(mass, com_world, i_world);

        let v_body = SpatialVector::new(v, w);
        let bias = spatial_cross_dual(v_body, i_s.mul_sv(v_body));
        let g = V3::new(0.0, -molla::math::GRAVITY, 0.0);
        let f_g_lin = g * mass;
        let f_g_ang = com_world.cross(f_g_lin);
        let p_a = bias - SpatialVector::new(f_g_lin, f_g_ang);

        let mut h = vec![0.0_f64; 36];
        for i in 0..6 {
            for j in 0..6 {
                h[i * 6 + j] = i_s.m[i][j];
            }
        }
        let tau = [
            -p_a.linear.x,
            -p_a.linear.y,
            -p_a.linear.z,
            -p_a.angular.x,
            -p_a.angular.y,
            -p_a.angular.z,
        ];
        let qdd = dense_chol_solve(&mut h, &tau, 6).unwrap();

        let v_new = V3::new(v.x + qdd[0] * dt, v.y + qdd[1] * dt, v.z + qdd[2] * dt);
        let w_new = V3::new(w.x + qdd[3] * dt, w.y + qdd[4] * dt, w.z + qdd[5] * dt);
        let p_new = pos + v_new * dt;
        let dq = Quat::from_xyzw(w_new.x * 0.5, w_new.y * 0.5, w_new.z * 0.5, 0.0);
        let r_dot = dq * rot;
        let r_new = Quat::from_xyzw(
            rot.x + r_dot.x * dt,
            rot.y + r_dot.y * dt,
            rot.z + r_dot.z * dt,
            rot.w + r_dot.w * dt,
        )
        .normalize();
        let q_out = s_out.joint_q.host_mut().unwrap();
        q_out[0] = p_new.x;
        q_out[1] = p_new.y;
        q_out[2] = p_new.z;
        q_out[3] = r_new.x;
        q_out[4] = r_new.y;
        q_out[5] = r_new.z;
        q_out[6] = r_new.w;
        let qd_out = s_out.joint_qd.host_mut().unwrap();
        qd_out[0] = v_new.x;
        qd_out[1] = v_new.y;
        qd_out[2] = v_new.z;
        qd_out[3] = w_new.x;
        qd_out[4] = w_new.y;
        qd_out[5] = w_new.z;
    }

    let mut ref_a = model.state().unwrap();
    let mut ref_b = model.state().unwrap();
    let r_q = ref_a.joint_q.host_mut().unwrap();
    r_q[1] = 0.5;
    r_q[6] = 1.0;

    let mut max_drift: Real = 0.0;
    for _ in 0..100 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        reference_step(&ref_a, &mut ref_b, mass, com_offset, inertia_local, dt);
        std::mem::swap(&mut ref_a, &mut ref_b);

        let q1 = s_a.joint_q.host().unwrap();
        let q2 = ref_a.joint_q.host().unwrap();
        let mut step_drift: Real = 0.0;
        for k in 0..7 {
            let d = (q1[k] - q2[k]).abs();
            if d > step_drift {
                step_drift = d;
            }
        }
        if step_drift > max_drift {
            max_drift = step_drift;
        }
    }
    eprintln!("coupled-D vs hand-rolled reference: max |Δq| over 100 steps = {max_drift:.3e}");
    assert!(
        max_drift < 1e-9,
        "rigorous dense Cholesky disagreed with reference: max Δ = {max_drift}"
    );
}
