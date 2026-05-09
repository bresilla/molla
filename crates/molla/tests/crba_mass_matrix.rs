//! CPU CRBA mass-matrix tests against analytical formulas.
//!
//! `make test` runs these (no `--ignored`, no GPU). All are
//! single-articulation, revolute-only.

use molla::core::BodyId;
use molla::math::{Mat3, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;
use molla::solvers::mass_matrix;

/// Single revolute pendulum: link of mass `m` and length `2L` rotating
/// about a horizontal axis through one end. Inertia about that axis
/// is `m·L² + I_com` where `I_com` is the link's own inertia about its
/// COM. For a thin rod: `I_com = m·(2L)² / 12 = m·L²/3`. Total about
/// the pivot via parallel-axis: `m·L² + m·L²/3 = 4·m·L²/3`. With our
/// joint anchor at the proximal end (-Y by L) the lever arm is `L`,
/// matching this calculation.
#[test]
fn pendulum_mass_matrix_matches_parallel_axis() {
    let mass = 1.5_f64;
    let l = 0.4_f64; // half-length
    // Rod inertia tensor at COM: I_xx = I_zz = m·(2L)²/12, I_yy ≈ 0.
    let i_xx = mass * (2.0 * l).powi(2) / 12.0;
    let inertia = Mat3::from_diagonal(V3::new(i_xx, 0.0, i_xx));

    let mut b = ModelBuilder::new();
    let link = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia,
    }));
    // Joint at world origin, axis +Z, child anchor at -Y by L.
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
    molla::sim::eval_fk(&model, &mut state).unwrap();

    let m = mass_matrix(&model, &state, 0).unwrap();
    assert_eq!(m.len(), 1);
    let expected = mass * l * l + i_xx;
    let got = m[0];
    assert!(
        (got - expected).abs() < 1e-9,
        "M[0,0] = {got}, expected {expected}"
    );
}

/// Double pendulum at rest pose with both joints aligned (both links
/// hang straight down along world -Y, axis +Z). At this configuration
/// the cross-coupling M[0,1] equals the mass times the lever arms.
/// See Featherstone (2008) §6.5 for the closed form. We just verify
/// CRBA is symmetric, positive-definite, and matches the M[0,0] from
/// CRBA-on-the-truncated-system equals the mass moment of the chain.
#[test]
fn double_pendulum_mass_matrix_is_spd_and_symmetric() {
    let mass = 1.0_f64;
    let l = 0.5_f64;
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
    // First joint: world origin, axis +Z, child anchor at -Y by L.
    b.add_joint_revolute(
        BodyId::NONE,
        l0,
        Transform::IDENTITY,
        Transform::from_translation(V3::new(0.0, -l, 0.0)),
        V3::Z,
    )
    .unwrap();
    // Second joint: parent's distal tip (+Y by L in body 0 frame),
    // child's proximal end (-Y by L in body 1 frame). Same axis.
    b.add_joint_revolute(
        l0,
        l1,
        Transform::from_translation(V3::new(0.0, l, 0.0)),
        Transform::from_translation(V3::new(0.0, -l, 0.0)),
        V3::Z,
    )
    .unwrap();
    let model = b.build().unwrap();
    // FK so body_q reflects q = 0 (both links straight down).
    let mut state = model.state().unwrap();
    molla::sim::eval_fk(&model, &mut state).unwrap();

    let m = mass_matrix(&model, &state, 0).unwrap();
    assert_eq!(m.len(), 4);
    let m00 = m[0];
    let m01 = m[1];
    let m10 = m[2];
    let m11 = m[3];
    // Symmetry.
    assert!((m01 - m10).abs() < 1e-9, "M not symmetric: M[0,1]={m01}, M[1,0]={m10}");
    // Positive-definite: diagonal positive, det positive.
    assert!(m00 > 0.0 && m11 > 0.0, "diagonal non-positive");
    let det = m00 * m11 - m01 * m10;
    assert!(det > 0.0, "M not SPD: det={det}");

    // M[1,1] for the second joint alone (with q1's link only) should
    // equal the same parallel-axis term we tested above for the
    // single pendulum: m·L² + I_xx. Featherstone CRBA at q=0 gives
    // exactly this for the leaf joint.
    let expected_11 = mass * l * l + i_xx;
    assert!(
        (m11 - expected_11).abs() < 1e-9,
        "M[1,1] = {m11}, expected {expected_11}"
    );
}

/// CRBA + Featherstone consistency check: at zero velocity and zero
/// applied force on a pendulum at rest, the joint acceleration from
/// Featherstone forward dynamics must equal `M⁻¹ · (-G(q))` where
/// `G(q)` is the gravity term. We extract gravity by solving
/// `M·q̈ + G = 0  →  G = -M·q̈`. Then we verify Featherstone's
/// reported `qd_new` after one tiny step matches `qd_old + dt · q̈`.
#[test]
fn crba_consistent_with_featherstone_pendulum_acceleration() {
    use molla::core::Result;
    use molla::math::Real;
    use molla::solvers::Solver as _;

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
    let control = model.control();
    let contacts = molla::sim::Contacts::with_capacity(0);
    // Start at q = π/4 (link offset from straight up by 45°).
    state.joint_q.host_mut().unwrap()[0] = std::f64::consts::FRAC_PI_4;
    molla::sim::eval_fk(&model, &mut state).unwrap();

    let m = mass_matrix(&model, &state, 0).unwrap();
    let m00 = m[0];

    // Step Featherstone by tiny dt and read joint_qd.
    let mut state_b = model.state().unwrap();
    let mut solver = molla::solvers::FeatherstoneSolver::new_cpu();
    let dt: Real = 1.0e-5;
    solver
        .step(&model, &state, &control, &contacts, dt, &mut state_b)
        .unwrap();
    let qd_after = state_b.joint_qd.host().unwrap()[0];
    // qd_old = 0, so q̈ ≈ qd_after / dt.
    let q_dd_featherstone = qd_after / dt;

    // Independent gravity wrench projection onto joint axis.
    // Joint at world origin, axis +Z, child anchor at -Y by L. At q=0
    // the link points along +Y (inverted pendulum). At q rad, body
    // COM is at (-L·sin q, L·cos q, 0). Gravity force = (0, -m·g, 0).
    //     τ_z = r × F · ẑ = (-L sin q)(-m·g) - (L cos q)(0) = m·g·L·sin q
    // So `M·q̈ = τ_g  →  q̈ = m·g·L·sin q / M`.
    let q = state.joint_q.host().unwrap()[0];
    let g = molla::math::GRAVITY;
    let tau_g = mass * g * l * q.sin();
    let q_dd_expected = tau_g / m00;

    // Tolerance reflects f64 + the Featherstone / CRBA pipeline both
    // doing ~ε rounding; absolute bound is plenty.
    let err = (q_dd_featherstone - q_dd_expected).abs();
    eprintln!(
        "M[0,0]={m00:.6}, τ_g={tau_g:.6}, q̈_FS={q_dd_featherstone:.6}, q̈_CRBA={q_dd_expected:.6}"
    );
    assert!(
        err < 1.0e-5,
        "Featherstone q̈ disagrees with M⁻¹·τ_g: |Δ|={err:.4e}"
    );
    let _: Result<()> = Ok(());
}

/// Empty model returns an empty matrix.
#[test]
fn empty_model_returns_empty() {
    let model = ModelBuilder::new().build().unwrap();
    let state = model.state().unwrap();
    let m = mass_matrix(&model, &state, 0).unwrap();
    assert!(m.is_empty());
}
