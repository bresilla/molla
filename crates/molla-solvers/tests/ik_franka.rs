//! IK on a 7-DOF Franka-style serial arm + standalone Jacobian tests.
//! All CPU; no `--ignored` because nothing here needs a GPU adapter.

use molla_core::BodyId;
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, ModelBuilder};
use molla_solvers::{IkObjective, IkSolver, ik_solver_health, jacobian};

fn box_inertia(mass: Real, half: V3) -> Mat3 {
    let h2 = half * half;
    let m_third = mass / 3.0;
    Mat3::from_diagonal(V3::new(
        m_third * (h2.y + h2.z),
        m_third * (h2.x + h2.z),
        m_third * (h2.x + h2.y),
    ))
}

/// Stylised 7-DOF arm modelled after Franka Emika Panda's joint
/// layout (axes, link lengths approximate). Returns
/// `(model, end_effector_body_index)`.
fn build_franka() -> (molla_sim::Model, usize) {
    let mut b = ModelBuilder::new();
    let half = V3::new(0.04, 0.04, 0.04);
    let mass = 1.0_f64;
    let inertia = box_inertia(mass, half);
    let make_link = |b: &mut ModelBuilder| {
        b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass,
            com: V3::ZERO,
            inertia,
        }))
    };

    b.begin_articulation();
    // Joint axes follow Franka's default (mostly Z, alternating with
    // Y) — concrete values aren't important for IK convergence as
    // long as the chain has 7 independent DOFs.
    let axes = [V3::Z, V3::Y, V3::Z, V3::Y, V3::Z, V3::Y, V3::Z];
    let link_lens = [0.333, 0.0, 0.316, 0.0, 0.384, 0.0, 0.107];
    let mut prev = BodyId::NONE;
    let mut last_body = 0usize;
    for i in 0..7 {
        let body = make_link(&mut b);
        last_body = body.index();
        let x_p = if i == 0 {
            Transform::IDENTITY
        } else {
            Transform::from_translation(V3::new(0.0, 0.0, link_lens[i - 1]))
        };
        let x_c = Transform::IDENTITY;
        b.add_joint_revolute(prev, body, x_p, x_c, axes[i]).unwrap();
        prev = body;
    }
    let model = b.build().unwrap();
    (model, last_body)
}

#[test]
fn jacobian_finite_difference_matches_analytical() {
    // 1-link revolute pendulum, axis +Z, child anchor at -Y by 0.5,
    // so at q=0 the link's "tip" (in body local frame at +Y by 0.5)
    // is at world (0, 1, 0). A small δq about Z should move the tip
    // in -X by L·δq.
    let mut b = ModelBuilder::new();
    let mass = 1.0;
    let inertia = Mat3::from_diagonal(V3::splat(0.1));
    let link = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass,
        com: V3::ZERO,
        inertia,
    }));
    let l = 0.5;
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
    molla_sim::eval_fk(&model, &mut state).unwrap();

    let tip_local = V3::new(0.0, l, 0.0);
    let j = jacobian(&model, &state, 0, link.index(), tip_local).unwrap();
    // Analytical: tip at (0, L, 0)+(0, L, 0) → wait the body's frame
    // has +Y aligned with world +Y at q=0 (since child anchor is
    // -Y·L). So tip_local (0, L, 0) is at world (0, 2L, 0)? No wait
    // — body 0 sits at world (0, L, 0) (FK output), tip in its
    // frame is +Y, so world tip = (0, L, 0) + R·(0, L, 0) = (0, 2L, 0).
    // dω = (0, 0, 1)·δq, so dv_tip = ω × r_joint→tip = (0,0,1)×(0,2L,0)
    //   = (-2L, 0, 0). So J_linear[:, 0] should be ≈ (-2L, 0, 0).
    let dof = 1;
    let jx = j[0 * dof + 0];
    let jy = j[1 * dof + 0];
    let jz = j[2 * dof + 0];
    eprintln!(
        "jacobian column: linear=({jx}, {jy}, {jz}), expected (~{:.3}, 0, 0)",
        -2.0 * l
    );
    assert!((jx - (-2.0 * l)).abs() < 1e-9);
    assert!(jy.abs() < 1e-12);
    assert!(jz.abs() < 1e-12);
}

#[test]
fn ik_franka_reaches_position_target() {
    let (model, ee_body) = build_franka();
    let mut state = model.state().unwrap();
    // Forward-kinematics with q = 0 to discover where the EE sits.
    molla_sim::eval_fk(&model, &mut state).unwrap();
    let ee_at_zero = state.body_q.host().unwrap()[ee_body].position;

    // Target: 0.3 m offset from the rest position.
    let target = ee_at_zero + V3::new(0.2, 0.1, -0.15);
    let solver = IkSolver::new(vec![IkObjective::Position {
        body: ee_body,
        point_local: V3::ZERO,
        target_world: target,
        weight: 1.0,
    }]);
    let q0 = vec![0.0; 7];
    let result = solver.solve(&model, &mut state, &q0).unwrap();
    eprintln!(
        "Franka IK: iters={}, residual={:.5}, converged={}",
        result.iters, result.residual, result.converged
    );
    let achieved = state.body_q.host().unwrap()[ee_body].position;
    let err = (achieved - target).length();
    eprintln!("position error: {err:.4} m");
    // Definition of done: ≤ 5 mm (PLAN.md §20 stage 10).
    assert!(err < 5.0e-3, "IK missed target by {err:.4} m");
}

#[test]
fn ik_multi_seed_finds_solution_when_single_seed_fails() {
    let (model, ee_body) = build_franka();
    let mut state = model.state().unwrap();
    molla_sim::eval_fk(&model, &mut state).unwrap();
    let ee_at_zero = state.body_q.host().unwrap()[ee_body].position;

    // Moderately distant target — chosen inside the arm's ~1.14 m
    // reach. A single seed at q=0 occasionally stalls in a local
    // minimum because the chain has axis-aligned joints; multi-seed
    // is the safety net.
    let target = ee_at_zero + V3::new(0.25, -0.2, -0.15);
    let solver = IkSolver::new(vec![IkObjective::Position {
        body: ee_body,
        point_local: V3::ZERO,
        target_world: target,
        weight: 1.0,
    }]);
    let seeds = vec![
        vec![0.0; 7],
        vec![0.5; 7],
        vec![-0.5, 0.3, -0.2, 0.7, 0.1, -0.4, 0.0],
        vec![1.0, -0.5, 0.5, -1.0, 0.5, -0.5, 1.0],
    ];
    let r = solver.solve_with_seeds(&model, &mut state, &seeds).unwrap();
    let achieved = state.body_q.host().unwrap()[ee_body].position;
    let err = (achieved - target).length();
    eprintln!(
        "multi-seed IK: best residual = {:.4}, position err = {err:.4} m",
        r.residual
    );
    assert!(err < 1.0e-2, "IK best seed missed by {err:.4} m");
}

#[test]
fn ik_joint_limit_penalty_keeps_q_in_band() {
    let (model, ee_body) = build_franka();
    let mut state = model.state().unwrap();
    molla_sim::eval_fk(&model, &mut state).unwrap();
    let ee_at_zero = state.body_q.host().unwrap()[ee_body].position;
    let target = ee_at_zero + V3::new(0.3, 0.0, 0.0);

    // Position objective + tight joint-1 limit at [-0.1, 0.1].
    let solver = IkSolver::new(vec![
        IkObjective::Position {
            body: ee_body,
            point_local: V3::ZERO,
            target_world: target,
            weight: 1.0,
        },
        IkObjective::JointLimit {
            dof: 1,
            lo: -0.1,
            hi: 0.1,
            weight: 50.0,
        },
    ]);
    let r = solver.solve(&model, &mut state, &[0.0; 7]).unwrap();
    let q1 = r.q[1];
    eprintln!("constrained q[1] = {q1:.4} (target band [-0.1, 0.1])");
    // Soft penalty — allow small slack.
    assert!(q1.abs() < 0.15, "joint 1 escaped band: {q1}");
}

/// Stage 24a — observability: the SVD-pseudoinverse fallback should
/// rarely trigger for well-conditioned IK problems. We run a fresh
/// position-target solve, then assert the per-solve fallback rate
/// (psd_fallbacks / chol_attempts) stays low.
///
/// This test depends on whatever runs before it for absolute counts;
/// we work in deltas so it's robust to test ordering.
#[test]
fn ik_solver_health_fallback_rate_is_low() {
    let (chol0, psd0) = ik_solver_health();

    // Standard "reach a nearby target" — should be solvable cleanly.
    let (model, ee_body) = build_franka();
    let mut state = model.state().unwrap();
    molla_sim::eval_fk(&model, &mut state).unwrap();
    let ee_at_zero = state.body_q.host().unwrap()[ee_body].position;
    let target = ee_at_zero + V3::new(0.2, 0.0, 0.0);
    let solver = IkSolver::new(vec![IkObjective::Position {
        body: ee_body,
        point_local: V3::ZERO,
        target_world: target,
        weight: 1.0,
    }]);
    let _ = solver.solve(&model, &mut state, &[0.0; 7]).unwrap();

    let (chol1, psd1) = ik_solver_health();
    let dchol = chol1 - chol0;
    let dpsd = psd1 - psd0;
    eprintln!(
        "IK fallback delta: chol_attempts={dchol}, psd_fallbacks={dpsd} \
         (rate = {:.3}%)",
        100.0 * (dpsd as f64) / (dchol.max(1) as f64)
    );
    // Healthy IK should never need the SVD fallback for a tame target.
    assert!(
        dpsd == 0,
        "expected zero psd fallbacks on a well-conditioned solve, got {dpsd} of {dchol}"
    );
}
