//! Stage 39E — `JointType::Cable` integration tests.
//!
//! Three end-to-end checks on the CPU Featherstone solver:
//!
//!   1. `cable_joint_2dof_count`
//!      A model with N cable joints exposes `2·N` DoFs (1 linear + 1
//!      bend per joint).
//!
//!   2. `cable_stretch_oscillates_at_correct_frequency`
//!      Single child body attached to the world by a cable. With unit
//!      mass and the cable axis perpendicular to gravity, the linear
//!      DoF behaves like a 1-D mass-spring oscillator with `ω = √(k/m)`.
//!      We track the first zero crossing and compare its time to the
//!      analytic quarter-period `T/4 = (π/2)/ω`.
//!
//!   3. `cable_bend_returns_to_rest`
//!      Perturb the bend DoF by ~10°, watch it decay back toward zero
//!      under the configured bend damping. Final `|q_b|` and `|q̇_b|`
//!      must be < 1 % of the initial bend.
//!
//! The cable convention (Stage 39E) is fixed: cable axis = joint-local
//! **+z**, bend axis = joint-local **+x**.

use molla::core::BodyId;
use molla::math::{Mat3, Quat, Real, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;

/// A spherical-inertia body with mass `m` and diagonal inertia tensor
/// `I·diag` — used by both oscillator tests.
fn unit_body(b: &mut ModelBuilder, mass: Real, inertia_scalar: Real) -> molla::core::BodyId {
    b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY * inertia_scalar,
    }))
}

#[test]
fn cable_joint_2dof_count() {
    let mut b = ModelBuilder::new();
    // Each joint connects world → its own body; we add four independent
    // articulations so the DoF accounting isn't entangled by tree
    // topology — purely the joint's own (coord, dof) layout.
    let mut prev_body: Option<BodyId> = None;
    let mut joint_count = 0usize;
    for i in 0..4 {
        let body = unit_body(&mut b, 1.0, 0.05);
        let parent = if i == 0 { BodyId::NONE } else { prev_body.unwrap() };
        b.add_joint_cable(
            parent,
            body,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.0, -0.5)),
            10.0, // stiffness_linear
            0.5,  // stiffness_bend
            0.0,
            0.0,
        )
        .unwrap();
        prev_body = Some(body);
        joint_count += 1;
    }
    let model = b.build().unwrap();

    // 2 DoFs per cable joint.
    assert_eq!(model.joint_count, joint_count);
    assert_eq!(model.joint_dof_total, 2 * joint_count);
    // 2 coords per cable joint (linear stretch + bend angle).
    assert_eq!(model.joint_coord_total, 2 * joint_count);
    // CSR offsets monotonically increase by 2 each entry.
    let dof_off = model.joint_dof_offset.host().unwrap();
    let coord_off = model.joint_coord_offset.host().unwrap();
    for j in 0..joint_count {
        assert_eq!(
            dof_off[j + 1] - dof_off[j],
            2,
            "joint {j} should expose 2 DoF, got {}",
            dof_off[j + 1] - dof_off[j]
        );
        assert_eq!(
            coord_off[j + 1] - coord_off[j],
            2,
            "joint {j} should expose 2 coords"
        );
    }
}

/// Single mass on a cable spring. With gravity (-y), the cable axis
/// pinned along world +z, and the body's COM placed on the cable
/// axis, the linear DoF decouples from gravity and reduces to a 1-D
/// mass-spring oscillator
///
/// ```text
///     m · q̈_l + k · q_l = 0     →     ω = √(k / m).
/// ```
///
/// We seed `q_l(0) = q0` and `q̇_l(0) = 0`, integrate until the first
/// zero-crossing of `q_l`, and check the elapsed time matches the
/// analytic quarter-period `T/4 = (π/2) / ω` within 5 %.
#[test]
fn cable_stretch_oscillates_at_correct_frequency() {
    let mass: Real = 1.0;
    let k_lin: Real = 100.0; // → ω = 10 rad/s, T = 0.628 s, T/4 = 0.157 s
    let omega = (k_lin / mass).sqrt();
    let quarter_period = std::f64::consts::FRAC_PI_2 / omega;

    // Single body. Place the joint frame so the cable axis (+z) is
    // perpendicular to gravity (-y). No `x_c` offset → child COM sits
    // on the cable axis at q_l = 0, so the stretch DoF doesn't couple
    // into the gravitational potential energy at the linearised order.
    let mut b = ModelBuilder::new();
    let body = unit_body(&mut b, mass, 0.1);
    let joint = b
        .add_joint_cable(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
            k_lin,
            0.0, // no bend stiffness (we only excite the linear DoF)
            0.0, // no linear damping
            0.0,
        )
        .unwrap();
    let _ = joint;
    // Disable gravity on the body's world so the test is a pure
    // mass-spring — the cable axis is perpendicular to default
    // gravity but any tiny coupling through the second-order ABA
    // recursion would still bias the period at the percent level.
    b.set_world_gravity(WorldId(0), V3::ZERO);
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let q0: Real = 0.05;
    s_a.joint_q.host_mut().unwrap()[0] = q0;
    s_a.joint_qd.host_mut().unwrap()[0] = 0.0;
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();

    let dt: Real = 0.0001;
    let max_steps = (2.5 * quarter_period / dt) as usize;
    let mut zero_cross_time: Option<Real> = None;
    let mut prev_q = q0;
    for step in 1..=max_steps {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q = s_a.joint_q.host().unwrap()[0];
        if zero_cross_time.is_none() && prev_q > 0.0 && q <= 0.0 {
            // Linear interp between the last two samples to refine the
            // zero-crossing time.
            let frac = prev_q / (prev_q - q);
            zero_cross_time = Some((step as Real - 1.0 + frac) * dt);
            break;
        }
        prev_q = q;
    }

    let t_zero = zero_cross_time.expect(
        "cable stretch never crossed zero — solver may be over-damped or unstable",
    );
    let err = (t_zero - quarter_period).abs() / quarter_period;
    eprintln!(
        "cable stretch ω = {:.3} rad/s, T/4 = {:.5}s, observed zero crossing = {:.5}s, err = {:.2}%",
        omega,
        quarter_period,
        t_zero,
        err * 100.0
    );
    assert!(
        err < 0.05,
        "cable stretch frequency off by {:.2}% (T/4 expected {:.5}s, got {:.5}s)",
        err * 100.0,
        quarter_period,
        t_zero
    );
}

/// Perturb the bend DoF by ~10° and verify it decays back to zero
/// under the configured bend damping. The body has spherical inertia
/// and we set bend stiffness `k_b` plus damping `c_b` such that the
/// linearised bend ODE
///
/// ```text
///     I · q̈_b + c_b · q̇_b + k_b · q_b = 0
/// ```
///
/// is well into the under-damped regime (`ζ ≈ 0.4`). After integrating
/// for roughly 6 natural periods we expect both `|q_b|` and `|q̇_b|`
/// to be < 1 % of the initial bend amplitude.
#[test]
fn cable_bend_returns_to_rest() {
    let mass: Real = 1.0;
    let inertia: Real = 0.05;
    let k_bend: Real = 5.0; // ω_b = √(k/I) ≈ 10 rad/s
    let c_bend: Real = 0.4; // ζ = c/(2·√(k·I)) ≈ 0.4

    let mut b = ModelBuilder::new();
    let body = unit_body(&mut b, mass, inertia);
    b.add_joint_cable(
        BodyId::NONE,
        body,
        Transform::IDENTITY,
        Transform::IDENTITY,
        0.0,
        k_bend,
        0.0,
        c_bend,
    )
    .unwrap();
    b.set_world_gravity(WorldId(0), V3::ZERO);
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let q_b0: Real = 0.175; // ~10°
    s_a.joint_q.host_mut().unwrap()[1] = q_b0; // bend coord
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();

    let dt: Real = 0.0005;
    // Run for ≈ 6 natural periods → 6 · 2π/ω ≈ 3.8 s.
    let total_time: Real = 4.0;
    let n_steps = (total_time / dt) as usize;
    let mut max_q_abs: Real = 0.0;
    for _ in 0..n_steps {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q_b = s_a.joint_q.host().unwrap()[1];
        if q_b.abs() > max_q_abs {
            max_q_abs = q_b.abs();
        }
    }
    let q_b_final = s_a.joint_q.host().unwrap()[1];
    let qd_b_final = s_a.joint_qd.host().unwrap()[1];
    eprintln!(
        "cable bend: q0 = {:.4} rad, max|q| during run = {:.4}, final q = {:.5}, final qd = {:.5}",
        q_b0, max_q_abs, q_b_final, qd_b_final
    );

    // Sanity: never exceeded the initial amplitude (no energy gain).
    assert!(
        max_q_abs <= q_b0 * 1.05,
        "bend amplitude grew during decay: max = {} > 1.05·q0 = {}",
        max_q_abs,
        q_b0 * 1.05
    );
    // Final state is within 1 % of rest on both coord and velocity.
    assert!(
        q_b_final.abs() < 0.01 * q_b0,
        "bend did not decay: final |q| = {} ≥ 1% of q0 = {}",
        q_b_final.abs(),
        q_b0
    );
    assert!(
        qd_b_final.abs() < 0.01 * q_b0 * (k_bend / inertia).sqrt(),
        "bend velocity did not decay: final |qd| = {}",
        qd_b_final.abs()
    );

    // Final body orientation should be near identity rotation.
    let body_q = s_a.body_q.host().unwrap();
    let r_final = body_q[0].rotation;
    let r_identity_err = (Quat::IDENTITY * r_final.inverse()).to_axis_angle().1.abs();
    assert!(
        r_identity_err < 0.02,
        "body rotation didn't return near identity: residual angle = {} rad",
        r_identity_err
    );
}
