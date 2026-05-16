//! RNEA inverse-dynamics tests.

use molla_core::BodyId;
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, ModelBuilder};
use molla_solvers::{inverse_dynamics, mass_matrix};

/// Pendulum at q = π/4, q̇ = 0, q̈ = 0. RNEA should return the joint
/// torque τ that holds the link in place against gravity, which
/// equals the gravity torque about the joint axis.
#[test]
fn pendulum_gravity_torque_matches_rnea() {
    let mass = 1.0_f64;
    let l = 0.5_f64;
    let i_xx = mass * (2.0 * l).powi(2) / 12.0;
    let inertia = Mat3::from_diagonal(V3::new(i_xx, 0.0, i_xx));

    let mut b = ModelBuilder::new();
    let link = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia,
    }));
    b.add_joint_revolute(
        BodyId::NONE,
        link,
        Transform::IDENTITY,
        Transform::from_translation(V3::new(0.0, -l, 0.0)),
        V3::Z,
    )
    .unwrap();
    let model = b.build().unwrap();

    let mut state = model.state().unwrap();
    state.joint_q.host_mut().unwrap()[0] = std::f64::consts::FRAC_PI_4;
    molla_sim::eval_fk(&model, &mut state).unwrap();

    // q̈ = 0: RNEA returns the torque required to hold the
    // pendulum static at q = π/4 against gravity. From the
    // Lagrangian: τ_static = m·g·L·sin(q).
    // The RNEA convention returns τ such that M·q̈ + C·q̇ + G(q) = τ.
    // With q̇ = 0 and q̈ = 0: τ = G(q) = -gravity torque (the torque
    // YOU must apply to cancel gravity).
    let qdd = vec![0.0_f64];
    let tau = inverse_dynamics(&model, &state, 0, &qdd).unwrap();
    assert_eq!(tau.len(), 1);

    let q = state.joint_q.host().unwrap()[0];
    let g = molla_math::GRAVITY;
    let tau_g_external = mass * g * l * q.sin(); // gravity-driven swing torque
    // To hold static, applied τ must oppose gravity → τ = -tau_g_external.
    let expected = -tau_g_external;
    let got = tau[0];
    eprintln!("pendulum: τ_RNEA={got:.6}, expected={expected:.6}");
    assert!(
        (got - expected).abs() < 1e-9,
        "RNEA gravity torque off: got {got}, expected {expected}"
    );
}

/// CRBA + RNEA closed loop: pick an arbitrary q, q̇, q̈; compute
/// τ via RNEA; then verify M(q)·q̈ + bias = τ where bias = RNEA(q̇, q̈=0).
/// The relation `RNEA(q, q̇, q̈) = M(q)·q̈ + RNEA(q, q̇, 0)` is the
/// classical decomposition of inverse dynamics.
#[test]
fn rnea_decomposes_into_mass_matrix_plus_bias() {
    let mass = 1.5_f64;
    let l = 0.4_f64;
    let i_xx = mass * (2.0 * l).powi(2) / 12.0;
    let inertia = Mat3::from_diagonal(V3::new(i_xx, 0.0, i_xx));

    let mut b = ModelBuilder::new();
    let l0 = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia,
    }));
    let l1 = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia,
    }));
    b.add_joint_revolute(
        BodyId::NONE,
        l0,
        Transform::IDENTITY,
        Transform::from_translation(V3::new(0.0, -l, 0.0)),
        V3::Z,
    )
    .unwrap();
    b.add_joint_revolute(
        l0,
        l1,
        Transform::from_translation(V3::new(0.0, l, 0.0)),
        Transform::from_translation(V3::new(0.0, -l, 0.0)),
        V3::Z,
    )
    .unwrap();
    let model = b.build().unwrap();
    let mut state = model.state().unwrap();
    // Non-trivial config + velocities.
    state
        .joint_q
        .host_mut()
        .unwrap()
        .copy_from_slice(&[0.3, -0.2]);
    state
        .joint_qd
        .host_mut()
        .unwrap()
        .copy_from_slice(&[0.4, 0.1]);
    molla_sim::eval_fk(&model, &mut state).unwrap();

    let qdd_test = [1.7_f64, -0.6];
    let tau_full = inverse_dynamics(&model, &state, 0, &qdd_test).unwrap();
    let tau_bias = inverse_dynamics(&model, &state, 0, &[0.0, 0.0]).unwrap();
    let m = mass_matrix(&model, &state, 0).unwrap();
    let n = (m.len() as f64).sqrt() as usize;
    assert_eq!(n, 2);

    // M·q̈ + bias should equal tau_full.
    let mut mqdd = [0.0_f64; 2];
    for i in 0..n {
        for j in 0..n {
            mqdd[i] += m[i * n + j] * qdd_test[j];
        }
    }
    for i in 0..n {
        let expected = mqdd[i] + tau_bias[i];
        let got = tau_full[i];
        let err = (got - expected).abs();
        eprintln!("i={i}: τ_full={got:.6}, M·q̈+bias={expected:.6}, |Δ|={err:.2e}");
        assert!(
            err < 1e-9,
            "RNEA decomposition broken at i={i}: |Δ|={err:.4e}"
        );
    }
}

/// Empty model returns empty.
#[test]
fn rnea_empty_returns_empty() {
    let model = ModelBuilder::new().build().unwrap();
    let state = model.state().unwrap();
    let tau = inverse_dynamics(&model, &state, 0, &[]).unwrap();
    assert!(tau.is_empty());
    let _: Real = 0.0;
}
