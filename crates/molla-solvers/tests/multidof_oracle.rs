//! INSPECTION.md / 2026-07-03 critic audit — analytic oracle for the
//! multi-DoF motion-subspace path in CRBA + RNEA.
//!
//! Oracle: a D6 joint with five axes locked (only rotation-Z free) is
//! EXACTLY a revolute-Z joint. Build the same double pendulum twice —
//! once with revolute joints, once with lock-filtered D6 joints — and
//! require the mass matrix (on the free-axis submatrix) and the
//! inverse-dynamics torques (free slots) to match to solver epsilon,
//! with all locked slots exactly zero. This exercises the shared
//! `build_motion_subspace` multi-DoF machinery (6-column D6 blocks,
//! lock filtering, per-joint dof offsets) against the independently
//! verified single-DoF path.

use molla_core::BodyId;
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, ModelBuilder};
use molla_solvers::{inverse_dynamics, mass_matrix};

fn link_inertia(mass: Real, l: Real) -> SpatialInertia {
    let i_xx = mass * (2.0 * l).powi(2) / 12.0;
    SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia: Mat3::from_diagonal(V3::new(i_xx, 0.0, i_xx)),
    }
}

fn build_chain(d6: bool) -> molla_sim::Model {
    let mass = 1.3_f64;
    let l = 0.5_f64;
    let mut b = ModelBuilder::new();
    let l0 = b.add_body(BodyParams::new().with_spatial_inertia(link_inertia(mass, l)));
    let l1 = b.add_body(BodyParams::new().with_spatial_inertia(link_inertia(mass, l)));
    let x_p0 = Transform::IDENTITY;
    let x_c0 = Transform::from_translation(V3::new(0.0, -l, 0.0));
    let x_p1 = Transform::from_translation(V3::new(0.0, l, 0.0));
    let x_c1 = Transform::from_translation(V3::new(0.0, -l, 0.0));
    if d6 {
        // Lock everything except rotation-Z (dof slot 5 of
        // [linX, linY, linZ, rotX, rotY, rotZ]).
        let lock = [0.0, 0.0, 0.0, 0.0, 0.0, f64::NEG_INFINITY];
        let free_hi = [0.0, 0.0, 0.0, 0.0, 0.0, f64::INFINITY];
        let j0 = b.add_joint_d6(BodyId::NONE, l0, x_p0, x_c0).unwrap();
        b.set_joint_limits(j0, &lock, &free_hi).unwrap();
        let j1 = b.add_joint_d6(l0, l1, x_p1, x_c1).unwrap();
        b.set_joint_limits(j1, &lock, &free_hi).unwrap();
    } else {
        b.add_joint_revolute(BodyId::NONE, l0, x_p0, x_c0, V3::Z)
            .unwrap();
        b.add_joint_revolute(l0, l1, x_p1, x_c1, V3::Z).unwrap();
    }
    b.build().unwrap()
}

#[test]
fn d6_locked_to_revolute_matches_crba_and_rnea() {
    let model_rev = build_chain(false);
    let model_d6 = build_chain(true);
    let mut s_rev = model_rev.state().unwrap();
    let mut s_d6 = model_d6.state().unwrap();
    molla_sim::eval_fk(&model_rev, &mut s_rev).unwrap();
    molla_sim::eval_fk(&model_d6, &mut s_d6).unwrap();

    // ── CRBA: free-axis submatrix must match the revolute 2×2.
    let m_rev = mass_matrix(&model_rev, &s_rev, 0).unwrap();
    assert_eq!(m_rev.len(), 4, "revolute chain must have 2 DOFs");
    let m_d6 = mass_matrix(&model_d6, &s_d6, 0).unwrap();
    assert_eq!(m_d6.len(), 144, "D6 chain must have 12 DOF slots");
    let free = [5usize, 11];
    for (ri, &r) in free.iter().enumerate() {
        for (ci, &c) in free.iter().enumerate() {
            let got = m_d6[r * 12 + c];
            let want = m_rev[ri * 2 + ci];
            assert!(
                (got - want).abs() < 1.0e-10,
                "M[{r},{c}] = {got} but revolute M[{ri},{ci}] = {want}"
            );
        }
    }
    // Locked rows/cols must be exactly zero (zero subspace columns).
    for r in 0..12 {
        for c in 0..12 {
            if free.contains(&r) && free.contains(&c) {
                continue;
            }
            assert!(
                m_d6[r * 12 + c].abs() < 1.0e-12,
                "locked M[{r},{c}] = {} must be zero",
                m_d6[r * 12 + c]
            );
        }
    }

    // ── RNEA: same joint accelerations through the free axes.
    let qdd_rev = [0.7, -1.3];
    let mut qdd_d6 = [0.0; 12];
    qdd_d6[5] = qdd_rev[0];
    qdd_d6[11] = qdd_rev[1];
    let tau_rev = inverse_dynamics(&model_rev, &s_rev, 0, &qdd_rev).unwrap();
    let tau_d6 = inverse_dynamics(&model_d6, &s_d6, 0, &qdd_d6).unwrap();
    assert_eq!(tau_rev.len(), 2);
    assert_eq!(tau_d6.len(), 12);
    for (i, &slot) in free.iter().enumerate() {
        assert!(
            (tau_d6[slot] - tau_rev[i]).abs() < 1.0e-10,
            "tau[{slot}] = {} but revolute tau[{i}] = {}",
            tau_d6[slot],
            tau_rev[i]
        );
    }
    for (slot, tau) in tau_d6.iter().enumerate() {
        if !free.contains(&slot) {
            assert!(
                tau.abs() < 1.0e-12,
                "locked tau[{slot}] = {tau} must be zero"
            );
        }
    }
    // Non-vacuous: gravity must load the free axes at this pose? Both
    // links hang straight down (gravity along the chain) — the
    // acceleration terms alone must be nonzero.
    assert!(tau_rev.iter().any(|t| t.abs() > 1.0e-6));
}
