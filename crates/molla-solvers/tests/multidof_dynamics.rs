//! Multi-DOF joint support in CRBA / RNEA / the analytic Jacobian
//! (INSPECTION.md F6). All three now expand every joint type through
//! the same `build_motion_subspace` Featherstone forward dynamics
//! uses, so Ball / Free / D6(locked) / Cable articulations get mass
//! matrices, inverse dynamics, and Jacobians instead of
//! "not yet supported" errors.
//!
//! Oracles:
//! - Ball pendulum M against the closed-form world-origin spatial
//!   inertia bottom block `I_C − m·[c]×²` (independent arithmetic).
//! - CRBA columns against RNEA with unit q̈, zero q̇, zero gravity
//!   (`τ = M·q̈` exactly) — the classic cross-validation, run on
//!   ball+revolute, free-base, and locked-D6 chains.
//! - Ball Jacobian columns against hand-built target-shifted twists.

use molla_core::{BodyId, WorldId};
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, ModelBuilder};
use molla_solvers::{inverse_dynamics, jacobian, mass_matrix};

fn rod_body(b: &mut ModelBuilder, mass: Real, half_len: Real) -> BodyId {
    let i_xx = mass * (2.0 * half_len).powi(2) / 12.0;
    b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia: Mat3::from_diagonal(V3::new(i_xx, 0.0, i_xx)),
    }))
}

/// Expected ball-joint-at-origin mass matrix: with the joint at the
/// world origin every subspace column is a pure rotation `(0; e_a)`,
/// so `M = I_C(world) − m·[c]×[c]×` = `I_C + m(|c|²·δ − c·cᵀ)`.
fn ball_at_origin_expected(mass: Real, c: V3, i_world: Mat3) -> [[Real; 3]; 3] {
    let i = [
        [i_world.x_axis.x, i_world.y_axis.x, i_world.z_axis.x],
        [i_world.x_axis.y, i_world.y_axis.y, i_world.z_axis.y],
        [i_world.x_axis.z, i_world.y_axis.z, i_world.z_axis.z],
    ];
    let cv = [c.x, c.y, c.z];
    let c2 = c.dot(c);
    let mut m = [[0.0; 3]; 3];
    for (a, row) in m.iter_mut().enumerate() {
        for (b, entry) in row.iter_mut().enumerate() {
            let delta = if a == b { 1.0 } else { 0.0 };
            *entry = i[a][b] + mass * (c2 * delta - cv[a] * cv[b]);
        }
    }
    m
}

#[test]
fn ball_pendulum_mass_matrix_matches_analytic() {
    let mass = 2.3_f64;
    let l = 0.6_f64;
    let mut b = ModelBuilder::new();
    let link = rod_body(&mut b, mass, l);
    b.add_joint_ball(
        BodyId::NONE,
        link,
        Transform::IDENTITY,
        Transform::from_translation(V3::new(0.0, -l, 0.0)),
    )
    .unwrap();
    let model = b.build().unwrap();
    let mut state = model.state().unwrap();
    molla_sim::eval_fk(&model, &mut state).unwrap();

    let m = mass_matrix(&model, &state, 0).unwrap();
    assert_eq!(m.len(), 9, "ball joint must yield a 3×3 mass matrix");

    // World COM + world inertia from the FK-resolved pose.
    let bq = state.body_q.host().unwrap()[link.index()];
    let c = bq.position; // com_local = ZERO
    let r = Mat3::from_quat(bq.rotation);
    let i_body = model.body_inertia.host().unwrap()[link.index()];
    let i_world = r * i_body * r.transpose();
    let expected = ball_at_origin_expected(mass, c, i_world);

    for a in 0..3 {
        for bb in 0..3 {
            let got = m[a * 3 + bb];
            let want = expected[a][bb];
            assert!(
                (got - want).abs() < 1e-9,
                "M[{a}][{bb}] = {got}, expected {want}"
            );
        }
    }
}

/// τ(q̈ = eᵢ, q̇ = 0, g = 0) must reproduce column i of M exactly.
fn crba_rnea_cross_check(model: &molla_sim::Model, dof_total: usize) {
    let mut state = model.state().unwrap();
    molla_sim::eval_fk(model, &mut state).unwrap();
    let m = mass_matrix(model, &state, 0).unwrap();
    assert_eq!(m.len(), dof_total * dof_total);
    for i in 0..dof_total {
        let mut qdd = vec![0.0; dof_total];
        qdd[i] = 1.0;
        let tau = inverse_dynamics(model, &state, 0, &qdd).unwrap();
        assert_eq!(tau.len(), dof_total);
        for r in 0..dof_total {
            let want = m[r * dof_total + i];
            assert!(
                (tau[r] - want).abs() < 1e-9,
                "column {i}, row {r}: RNEA τ = {}, CRBA M = {want}",
                tau[r]
            );
        }
    }
    // Symmetry while we're here.
    for a in 0..dof_total {
        for bb in 0..dof_total {
            assert!(
                (m[a * dof_total + bb] - m[bb * dof_total + a]).abs() < 1e-12,
                "M not symmetric at ({a},{bb})"
            );
        }
    }
}

#[test]
fn ball_revolute_chain_crba_matches_rnea() {
    let mut b = ModelBuilder::new();
    let l0 = rod_body(&mut b, 1.4, 0.5);
    let l1 = rod_body(&mut b, 0.9, 0.35);
    b.add_joint_ball(
        BodyId::NONE,
        l0,
        Transform::IDENTITY,
        Transform::from_translation(V3::new(0.0, -0.5, 0.0)),
    )
    .unwrap();
    b.add_joint_revolute(
        l0,
        l1,
        Transform::from_translation(V3::new(0.0, -0.5, 0.0)),
        Transform::from_translation(V3::new(0.0, -0.35, 0.0)),
        V3::Z,
    )
    .unwrap();
    b.set_world_gravity(WorldId(0), V3::ZERO);
    let model = b.build().unwrap();
    crba_rnea_cross_check(&model, 4);
}

#[test]
fn free_base_chain_crba_matches_rnea() {
    let mut b = ModelBuilder::new();
    let base = rod_body(&mut b, 3.0, 0.4);
    let arm = rod_body(&mut b, 1.1, 0.3);
    b.add_joint_free(
        BodyId::NONE,
        base,
        Transform::from_translation(V3::new(0.2, 0.9, -0.1)),
        Transform::IDENTITY,
    )
    .unwrap();
    b.add_joint_revolute(
        base,
        arm,
        Transform::from_translation(V3::new(0.4, 0.0, 0.0)),
        Transform::from_translation(V3::new(0.0, -0.3, 0.0)),
        V3::X,
    )
    .unwrap();
    b.set_world_gravity(WorldId(0), V3::ZERO);
    let model = b.build().unwrap();
    crba_rnea_cross_check(&model, 7);
}

#[test]
fn d6_locked_axes_zero_their_rows_and_cross_check() {
    let mut b = ModelBuilder::new();
    let body = rod_body(&mut b, 1.7, 0.45);
    let j = b
        .add_joint_d6(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, -0.45, 0.0)),
        )
        .unwrap();
    // Lock the three linear axes (lower == upper == 0) — leaves a
    // ball-like 3-rotational-DOF joint in D6 clothing.
    b.set_joint_limits(
        j,
        &[0.0, 0.0, 0.0, -10.0, -10.0, -10.0],
        &[0.0, 0.0, 0.0, 10.0, 10.0, 10.0],
    )
    .unwrap();
    b.set_world_gravity(WorldId(0), V3::ZERO);
    let model = b.build().unwrap();

    let mut state = model.state().unwrap();
    molla_sim::eval_fk(&model, &mut state).unwrap();
    let m = mass_matrix(&model, &state, 0).unwrap();
    assert_eq!(m.len(), 36);
    // Locked linear rows/cols must be exactly zero.
    for locked in 0..3 {
        for k in 0..6 {
            assert_eq!(m[locked * 6 + k], 0.0, "locked row {locked} col {k}");
            assert_eq!(m[k * 6 + locked], 0.0, "col {locked} row {k}");
        }
    }
    // The rotational block must be non-trivial (the pendulum's
    // inertia) — guards against the whole matrix silently zeroing.
    assert!(m[3 * 6 + 3] > 1e-3, "rotational block vanished");
    crba_rnea_cross_check(&model, 6);
}

#[test]
fn ball_jacobian_columns_are_target_shifted_twists() {
    let l = 0.6_f64;
    let mut b = ModelBuilder::new();
    let link = rod_body(&mut b, 1.0, l);
    b.add_joint_ball(
        BodyId::NONE,
        link,
        Transform::IDENTITY,
        Transform::from_translation(V3::new(0.0, -l, 0.0)),
    )
    .unwrap();
    let model = b.build().unwrap();
    let mut state = model.state().unwrap();
    molla_sim::eval_fk(&model, &mut state).unwrap();

    // Target: the link's distal tip in local coordinates.
    let tip_local = V3::new(0.0, l, 0.0);
    let bq = state.body_q.host().unwrap()[link.index()];
    let tip_world = bq.position + bq.rotation * tip_local;

    let jm = jacobian(&model, &state, 0, link.index(), tip_local).unwrap();
    assert_eq!(jm.len(), 6 * 3);

    // Ball at origin, identity joint-world rotation: column a is the
    // twist about world axis e_a through the origin, referenced at the
    // tip: linear = e_a × tip, angular = e_a.
    let axes = [V3::X, V3::Y, V3::Z];
    for (a, e) in axes.iter().enumerate() {
        let lin = e.cross(tip_world);
        let got_lin = V3::new(jm[a], jm[3 + a], jm[6 + a]);
        let got_ang = V3::new(jm[9 + a], jm[12 + a], jm[15 + a]);
        assert!(
            (got_lin - lin).length() < 1e-12,
            "col {a} linear: got {got_lin:?}, want {lin:?}"
        );
        assert!(
            (got_ang - *e).length() < 1e-12,
            "col {a} angular: got {got_ang:?}, want {e:?}"
        );
    }
}
