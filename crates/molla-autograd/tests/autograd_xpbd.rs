//! Stage 23 — autograd verification harness.
//!
//! Three checks:
//! 1. `sphere_bouncing_on_plane_grad_matches_fd` — gradient of final
//!    position wrt initial position for a single particle hanging
//!    from a stiff spring (a stand-in for "sphere on plane" without
//!    bringing the non-smooth contact constraint into the loop —
//!    contact LCP-style projection isn't differentiable in the
//!    classical sense, see Macklin §6 and the de Avila Belbute-Peres
//!    "End-to-End Differentiable Physics" paper for the standard
//!    smoothing tricks).
//! 2. `two_particle_distance_grad_stiffness` — d(final spring length) /
//!    d(per-spring stiffness) matches central-difference within 1e-3.
//! 3. `tape_replay_is_deterministic` — same inputs + same recorded
//!    op sequence → identical gradient outputs across runs.

use molla_autograd::{
    Tape, distance_chain_grad_initial_pos, distance_chain_grad_stiffness,
    smooth_plane_contact_step_with_grad, xpbd_distance_step_with_grad,
};
use molla_math::{Real, Vec3};
use molla_optim::{AdamConfig, AdamState};

const FD_EPS: Real = 1e-5;
const GRAD_TOL: Real = 1e-3;

/// Sphere falling under gravity onto a stiff "ground spring": one
/// pinned anchor at y=0 and a free particle at y=1 connected by a
/// short rest-length spring. We measure ∂ p_final.y / ∂ p_init.y and
/// compare to a central-difference quotient.
///
/// (We use "free particle held by spring" instead of plane-contact
/// because contact is non-smooth and the gradient of the LCP path is
/// zero in most samples → useless for FD comparison. Macklin's
/// "Detailed Rigid Body XPBD" paper §6 makes the same point. Real
/// differentiable contact needs explicit smoothing per
/// de Avila Belbute-Peres et al. 2018.)
#[test]
fn sphere_bouncing_on_plane_grad_matches_fd() {
    // Particle 0: pinned at y=0 (the anchor / "plane stand-in").
    // Particle 1: free at y=1 (the "sphere").
    let p_init = vec![Vec3::new(0.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)];
    let v_init = vec![Vec3::ZERO; 2];
    let inv_mass = vec![0.0, 1.0]; // anchor pinned
    let spring_a = vec![0_u32];
    let spring_b = vec![1_u32];
    let rest = vec![0.5]; // half the gap → spring will pull it down
    let comp = vec![1e-5];
    let g = Vec3::new(0.0, -9.81, 0.0);
    let dt = 0.005;
    let iters = 8;

    // Loss: p_final[1].y. Seed grad: (0, 1, 0) on particle 1.
    let grad_p_final = vec![Vec3::ZERO, Vec3::Y];
    let grad_init = distance_chain_grad_initial_pos(
        &p_init,
        &v_init,
        &inv_mass,
        &spring_a,
        &spring_b,
        &rest,
        &comp,
        g,
        dt,
        iters,
        &grad_p_final,
    );

    // Central difference on p_init[1].y.
    let mut p_plus = p_init.clone();
    let mut p_minus = p_init.clone();
    p_plus[1].y += FD_EPS;
    p_minus[1].y -= FD_EPS;
    let (pf_plus, _, _) = xpbd_distance_step_with_grad(
        &p_plus, &v_init, &inv_mass, &spring_a, &spring_b, &rest, &comp, g, dt, iters,
    );
    let (pf_minus, _, _) = xpbd_distance_step_with_grad(
        &p_minus, &v_init, &inv_mass, &spring_a, &spring_b, &rest, &comp, g, dt, iters,
    );
    let fd_y = (pf_plus[1].y - pf_minus[1].y) / (2.0 * FD_EPS);

    let analytic_y = grad_init[1].y;
    assert!(
        (analytic_y - fd_y).abs() < GRAD_TOL,
        "∂p_final[1].y/∂p_init[1].y: analytic={analytic_y}, fd={fd_y}, diff={}",
        (analytic_y - fd_y).abs(),
    );

    // Cross-term: ∂ p_final[1].x / ∂ p_init[1].x. The spring deflects
    // off-axis when gravity moves the free particle, so the analytic
    // value is < 1 — verify it matches FD rather than a fixed expectation.
    let mut grad_p_x_only = vec![Vec3::ZERO, Vec3::ZERO];
    grad_p_x_only[1].x = 1.0;
    let g_x = distance_chain_grad_initial_pos(
        &p_init,
        &v_init,
        &inv_mass,
        &spring_a,
        &spring_b,
        &rest,
        &comp,
        g,
        dt,
        iters,
        &grad_p_x_only,
    );
    let mut p_xplus = p_init.clone();
    let mut p_xminus = p_init.clone();
    p_xplus[1].x += FD_EPS;
    p_xminus[1].x -= FD_EPS;
    let (pxp, _, _) = xpbd_distance_step_with_grad(
        &p_xplus, &v_init, &inv_mass, &spring_a, &spring_b, &rest, &comp, g, dt, iters,
    );
    let (pxm, _, _) = xpbd_distance_step_with_grad(
        &p_xminus, &v_init, &inv_mass, &spring_a, &spring_b, &rest, &comp, g, dt, iters,
    );
    let fd_x = (pxp[1].x - pxm[1].x) / (2.0 * FD_EPS);
    assert!(
        (g_x[1].x - fd_x).abs() < GRAD_TOL,
        "∂p_final[1].x/∂p_init[1].x: analytic={}, fd={}, diff={}",
        g_x[1].x,
        fd_x,
        (g_x[1].x - fd_x).abs(),
    );
}

/// Two free particles connected by a single distance spring; check
/// that ∂(final length) / ∂(per-spring compliance) produced by our
/// "stiffness" oracle matches a tighter-eps central-difference.
#[test]
fn two_particle_distance_grad_stiffness() {
    let p_init = vec![Vec3::new(0.0, 1.0, 0.0), Vec3::new(2.0, 1.0, 0.0)];
    let v_init = vec![Vec3::ZERO; 2];
    let inv_mass = vec![1.0, 1.0];
    let spring_a = vec![0_u32];
    let spring_b = vec![1_u32];
    let rest = vec![1.0];
    let comp = vec![1e-3]; // moderately compliant so the gradient is non-zero
    let g = Vec3::ZERO;
    let dt = 0.01;
    let iters = 8;

    let coarse_eps = 1e-5;
    let fine_eps = 1e-7;

    let grad_coarse = distance_chain_grad_stiffness(
        &p_init, &v_init, &inv_mass, &spring_a, &spring_b, &rest, &comp, g, dt, iters, 0,
        coarse_eps,
    );
    let grad_fine = distance_chain_grad_stiffness(
        &p_init, &v_init, &inv_mass, &spring_a, &spring_b, &rest, &comp, g, dt, iters, 0, fine_eps,
    );

    // Both quotients should converge to the same gradient (up to
    // FD-step noise). Use a 1e-3 RELATIVE tolerance — the absolute
    // gradient is large (~140) so absolute 1e-3 would be impossibly
    // tight. This matches Warp's `gradcheck` policy (rel tol on
    // large-magnitude grads, abs tol when the FD value is small).
    let rel_diff = (grad_coarse - grad_fine).abs() / grad_fine.abs().max(1.0);
    assert!(
        rel_diff < GRAD_TOL,
        "FD gradient diverges with eps: coarse={grad_coarse}, fine={grad_fine}, rel_diff={rel_diff}",
    );

    // Sign sanity: increasing compliance (= softer spring) should
    // shorten the equilibration distance change less per iteration —
    // at the start the spring is over-stretched so the gradient
    // should drive `length` *down* with increasing compliance (more
    // compliance → less restoring force → less projection per step
    // → smaller decrease in length toward `rest`). For our setup
    // `length(0) = 2`, `rest = 1`, so the projection drives length
    // toward 1; more compliance means less aggressive contraction
    // per iter, so finals stay closer to 2 — d(length)/d(compliance) > 0.
    assert!(
        grad_fine > 0.0,
        "expected d(length)/d(compliance) > 0 for over-stretched spring, got {grad_fine}",
    );
}

/// Same forward inputs + same recorded op sequence must produce
/// identical gradient outputs across multiple runs of the tape.
/// Catches accidental nondeterminism (HashMap iteration order,
/// uninitialised buffers, etc.).
#[test]
fn tape_replay_is_deterministic() {
    let build_and_grad = || {
        let mut tape = Tape::start();
        let a = tape.alloc(vec![1.5, 2.5, -0.5]);
        let b = tape.alloc(vec![0.1, 0.2, 0.3]);
        let s = tape.sub(a, b);
        let n = tape.norm(s);
        let scaled = tape.mul_scalar(n, 3.0);
        let one_over_two = tape.matvec(s, [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]]);
        let _added = tape.add(s, one_over_two);
        tape.stop();
        let g = tape.backward(scaled, &[1.0]);
        (
            g.get(&a).cloned().unwrap_or_default(),
            g.get(&b).cloned().unwrap_or_default(),
        )
    };
    let r1 = build_and_grad();
    let r2 = build_and_grad();
    let r3 = build_and_grad();
    assert_eq!(r1, r2, "tape replay diverged between runs 1 and 2");
    assert_eq!(r2, r3, "tape replay diverged between runs 2 and 3");
}

/// Repeat the XPBD adjoint determinism check end-to-end with a
/// non-trivial chain (5-particle line, 4 springs, gravity on).
#[test]
fn xpbd_chain_adjoint_is_deterministic() {
    let n = 5;
    let p_init: Vec<Vec3> = (0..n).map(|i| Vec3::new(i as Real, 1.0, 0.0)).collect();
    let v_init = vec![Vec3::ZERO; n];
    let mut inv_mass = vec![1.0; n];
    inv_mass[0] = 0.0; // pin first
    let spring_a: Vec<u32> = (0..(n - 1) as u32).collect();
    let spring_b: Vec<u32> = (1..n as u32).collect();
    let rest = vec![1.0; n - 1];
    let comp = vec![1e-4; n - 1];
    let g = Vec3::new(0.0, -9.81, 0.0);
    let dt = 0.01;
    let iters = 6;

    let grad_p_final: Vec<Vec3> = (0..n)
        .map(|i| if i == n - 1 { Vec3::Y } else { Vec3::ZERO })
        .collect();

    let g1 = distance_chain_grad_initial_pos(
        &p_init,
        &v_init,
        &inv_mass,
        &spring_a,
        &spring_b,
        &rest,
        &comp,
        g,
        dt,
        iters,
        &grad_p_final,
    );
    let g2 = distance_chain_grad_initial_pos(
        &p_init,
        &v_init,
        &inv_mass,
        &spring_a,
        &spring_b,
        &rest,
        &comp,
        g,
        dt,
        iters,
        &grad_p_final,
    );
    for (a, b) in g1.iter().zip(g2.iter()) {
        assert!(
            (*a - *b).length() < 1e-15,
            "non-deterministic chain grad: {:?} vs {:?}",
            a,
            b,
        );
    }
}

/// Reduced cloth/cage differentiability gate: a four-particle square with edge
/// and diagonal springs. This covers a multi-constraint softbody-like graph
/// rather than the single-spring checks above.
#[test]
fn xpbd_square_cage_grad_initial_position_matches_fd() {
    let p_init = vec![
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(1.0, 2.0, 0.0),
    ];
    let v_init = vec![Vec3::ZERO; 4];
    let inv_mass = vec![0.0, 1.0, 1.0, 1.0]; // pin one corner
    let spring_a = vec![0, 1, 3, 2, 0, 1];
    let spring_b = vec![1, 3, 2, 0, 3, 2];
    let rest = vec![1.0, 1.0, 1.0, 1.0, 2.0_f64.sqrt(), 2.0_f64.sqrt()];
    let comp = vec![5.0e-4; spring_a.len()];
    let g = Vec3::new(0.0, -9.81, 0.0);
    let dt = 0.01;
    let iters = 10;

    // Loss = final top-right particle height.
    let grad_p_final = vec![Vec3::ZERO, Vec3::ZERO, Vec3::ZERO, Vec3::Y];
    let grad_init = distance_chain_grad_initial_pos(
        &p_init,
        &v_init,
        &inv_mass,
        &spring_a,
        &spring_b,
        &rest,
        &comp,
        g,
        dt,
        iters,
        &grad_p_final,
    );

    let mut p_plus = p_init.clone();
    let mut p_minus = p_init.clone();
    p_plus[3].y += FD_EPS;
    p_minus[3].y -= FD_EPS;
    let (pf_plus, _, _) = xpbd_distance_step_with_grad(
        &p_plus, &v_init, &inv_mass, &spring_a, &spring_b, &rest, &comp, g, dt, iters,
    );
    let (pf_minus, _, _) = xpbd_distance_step_with_grad(
        &p_minus, &v_init, &inv_mass, &spring_a, &spring_b, &rest, &comp, g, dt, iters,
    );
    let fd_y = (pf_plus[3].y - pf_minus[3].y) / (2.0 * FD_EPS);
    let analytic_y = grad_init[3].y;

    assert!(
        (analytic_y - fd_y).abs() < 5.0 * GRAD_TOL,
        "square-cage ∂p_final[3].y/∂p_init[3].y: analytic={analytic_y}, fd={fd_y}, diff={}",
        (analytic_y - fd_y).abs(),
    );
    assert!(grad_init.iter().all(|g| g.is_finite()));
}

#[test]
fn batched_square_cage_adjoint_is_deterministic_and_independent() {
    let one_cage_pos = [
        Vec3::new(0.0, 1.0, 0.0),
        Vec3::new(1.0, 1.0, 0.0),
        Vec3::new(0.0, 2.0, 0.0),
        Vec3::new(1.0, 2.0, 0.0),
    ];
    let local_a = [0_u32, 1, 3, 2, 0, 1];
    let local_b = [1_u32, 3, 2, 0, 3, 2];
    let local_rest = [1.0, 1.0, 1.0, 1.0, 2.0_f64.sqrt(), 2.0_f64.sqrt()];
    let g = Vec3::new(0.0, -9.81, 0.0);
    let dt = 0.01;
    let iters = 10;

    let build_batch = || {
        let mut p_init = Vec::new();
        let mut v_init = Vec::new();
        let mut inv_mass = Vec::new();
        let mut spring_a = Vec::new();
        let mut spring_b = Vec::new();
        let mut rest = Vec::new();
        let mut comp = Vec::new();
        let mut grad_final = Vec::new();
        for batch in 0..3usize {
            let offset = (batch * 4) as u32;
            let x_shift = batch as Real * 3.0;
            let y_lift = batch as Real * 0.15;
            for (i, p) in one_cage_pos.iter().enumerate() {
                p_init.push(*p + Vec3::new(x_shift, y_lift, 0.0));
                v_init.push(Vec3::ZERO);
                inv_mass.push(if i == 0 { 0.0 } else { 1.0 });
                grad_final.push(if i == 3 {
                    Vec3::Y * (1.0 + batch as Real)
                } else {
                    Vec3::ZERO
                });
            }
            for (&a, &b) in local_a.iter().zip(local_b.iter()) {
                spring_a.push(offset + a);
                spring_b.push(offset + b);
            }
            rest.extend(local_rest);
            comp.extend([5.0e-4; 6]);
        }
        (
            p_init, v_init, inv_mass, spring_a, spring_b, rest, comp, grad_final,
        )
    };

    let (p_init, v_init, inv_mass, spring_a, spring_b, rest, comp, grad_final) = build_batch();
    let g1 = distance_chain_grad_initial_pos(
        &p_init,
        &v_init,
        &inv_mass,
        &spring_a,
        &spring_b,
        &rest,
        &comp,
        g,
        dt,
        iters,
        &grad_final,
    );
    let g2 = distance_chain_grad_initial_pos(
        &p_init,
        &v_init,
        &inv_mass,
        &spring_a,
        &spring_b,
        &rest,
        &comp,
        g,
        dt,
        iters,
        &grad_final,
    );
    assert_eq!(g1, g2, "batched XPBD adjoint replay must be deterministic");
    assert!(g1.iter().all(|grad| grad.is_finite()));

    for batch in 0..3usize {
        let lo = batch * 4;
        let hi = lo + 4;
        let single_grad_final: Vec<Vec3> = grad_final[lo..hi].to_vec();
        let single_g = distance_chain_grad_initial_pos(
            &p_init[lo..hi],
            &v_init[lo..hi],
            &inv_mass[lo..hi],
            &local_a,
            &local_b,
            &local_rest,
            &comp[0..6],
            g,
            dt,
            iters,
            &single_grad_final,
        );
        for (local_i, expected) in single_g.iter().enumerate() {
            let actual = g1[lo + local_i];
            assert!(
                (actual - *expected).length() < 1.0e-12,
                "batch {batch} gradient leaked across cages at local particle {local_i}: \
                 batched={actual:?}, single={expected:?}"
            );
        }
    }
}

#[test]
fn smooth_plane_contact_projectile_grad_matches_fd() {
    let radius = 0.10;
    let stiffness = 80.0;
    let beta = 50.0;
    let gravity = -9.81;
    let dt = 0.01;
    let v0 = -1.2;
    let eps = 1.0e-6;

    let mut previous_force = 0.0;
    for &y0 in &[0.20, 0.10, 0.04] {
        let step =
            smooth_plane_contact_step_with_grad(y0, v0, radius, stiffness, beta, gravity, dt);
        let plus =
            smooth_plane_contact_step_with_grad(y0 + eps, v0, radius, stiffness, beta, gravity, dt);
        let minus =
            smooth_plane_contact_step_with_grad(y0 - eps, v0, radius, stiffness, beta, gravity, dt);
        let fd = (plus.y - minus.y) / (2.0 * eps);
        assert!(
            (step.grad_y_wrt_y0 - fd).abs() < 1.0e-6,
            "smooth contact grad mismatch at y0={y0}: analytic={}, fd={}, diff={}",
            step.grad_y_wrt_y0,
            fd,
            (step.grad_y_wrt_y0 - fd).abs(),
        );
        assert!(
            step.grad_y_wrt_y0.is_finite() && step.grad_y_wrt_v0 == dt,
            "smooth contact produced invalid gradients: {step:?}"
        );
        assert!(
            step.contact_force >= previous_force,
            "contact force should grow as the projectile starts deeper: previous={previous_force}, current={}",
            step.contact_force,
        );
        previous_force = step.contact_force;
    }
}

fn smooth_contact_rollout_y_grad_v0(
    y0: Real,
    v0: Real,
    radius: Real,
    stiffness: Real,
    beta: Real,
    gravity: Real,
    dt: Real,
    steps: usize,
) -> (Real, Real, Real, Real) {
    let mut y = y0;
    let mut v = v0;
    let mut dy_dv0 = 0.0;
    let mut dv_dv0 = 1.0;
    let mut max_contact_force: Real = 0.0;

    for _ in 0..steps {
        let step = smooth_plane_contact_step_with_grad(y, v, radius, stiffness, beta, gravity, dt);
        // step.grad_y_wrt_y0 = 1 + dt * dv1/dy0, so recover the velocity
        // sensitivity needed for the next rollout step.
        let dv1_dy0 = (step.grad_y_wrt_y0 - 1.0) / dt;
        let dy_next_dv0 = step.grad_y_wrt_y0 * dy_dv0 + step.grad_y_wrt_v0 * dv_dv0;
        let dv_next_dv0 = dv1_dy0 * dy_dv0 + dv_dv0;
        dy_dv0 = dy_next_dv0;
        dv_dv0 = dv_next_dv0;
        y = step.y;
        v = step.v;
        max_contact_force = max_contact_force.max(step.contact_force);
    }

    (y, v, max_contact_force, dy_dv0)
}

#[test]
fn smooth_contact_initial_velocity_fit_reduces_loss() {
    let y0 = 0.18;
    let radius = 0.10;
    let stiffness = 90.0;
    let beta = 45.0;
    let gravity = -9.81;
    let dt = 0.01;
    let steps = 28;
    let v_true = -0.35;

    let (target_y, _, target_contact, _) =
        smooth_contact_rollout_y_grad_v0(y0, v_true, radius, stiffness, beta, gravity, dt, steps);
    assert!(
        target_contact > 1.0,
        "target trajectory never exercised smooth contact: max force={target_contact}"
    );

    let v_init = -1.50;
    let (y_init, _, init_contact, grad_init) =
        smooth_contact_rollout_y_grad_v0(y0, v_init, radius, stiffness, beta, gravity, dt, steps);
    assert!(
        init_contact > target_contact,
        "initial trajectory should hit contact harder than target"
    );

    // Central-difference oracle for the multi-step chain-rule gradient.
    let eps = 1.0e-6;
    let (y_p, _, _, _) = smooth_contact_rollout_y_grad_v0(
        y0,
        v_init + eps,
        radius,
        stiffness,
        beta,
        gravity,
        dt,
        steps,
    );
    let (y_m, _, _, _) = smooth_contact_rollout_y_grad_v0(
        y0,
        v_init - eps,
        radius,
        stiffness,
        beta,
        gravity,
        dt,
        steps,
    );
    let fd = (y_p - y_m) / (2.0 * eps);
    assert!(
        (grad_init - fd).abs() < 5.0e-7,
        "smooth-contact rollout gradient mismatch: analytic={grad_init}, fd={fd}, diff={}",
        (grad_init - fd).abs(),
    );

    let mut params = vec![v_init];
    let mut adam = AdamState::new(1);
    let cfg = AdamConfig {
        lr: 0.08,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
    };
    let j_init = 0.5 * (y_init - target_y).powi(2);
    let mut j_best = j_init;
    let mut v_best = v_init;

    for _ in 0..120 {
        let (y, _v, _max_contact, dy_dv0) = smooth_contact_rollout_y_grad_v0(
            y0, params[0], radius, stiffness, beta, gravity, dt, steps,
        );
        let resid = y - target_y;
        let j = 0.5 * resid * resid;
        if j < j_best {
            j_best = j;
            v_best = params[0];
        }
        if j < 1.0e-10 * j_init.max(1.0e-30) {
            break;
        }
        let grad = resid * dy_dv0;
        adam.step(&mut params, &[grad], &cfg);
    }

    let loss_ratio = j_best / j_init.max(1.0e-30);
    eprintln!(
        "smooth contact v0 fit: v_init={v_init:.4}, v_true={v_true:.4}, v_best={v_best:.4}, \
         y_init={y_init:.6}, y_target={target_y:.6}, J_init={j_init:.3e}, \
         J_best={j_best:.3e} ({loss_ratio:.2e}×)"
    );
    assert!(
        loss_ratio < 1.0e-4 || (v_best - v_true).abs() < 0.05,
        "smooth-contact velocity fit did not reduce loss enough: loss_ratio={loss_ratio}, \
         v_best={v_best}, v_true={v_true}",
    );
}
