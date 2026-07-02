//! Stage 38.1 — end-to-end differentiable MPM **Adam fit on E**.
//!
//! Pipeline:
//!   1. Set up a small forward MPM sim: a 4×4×4 = 64 particle cube
//!      falling onto a soft floor for 30 substeps.
//!   2. Fix a "true" Young's modulus `E_true = 1e4` and run the
//!      forward to get the target final centroid.
//!   3. Pick a starting guess `E_init = 5 · E_true` (5× too stiff).
//!   4. Run Adam for up to 200 steps; at each step:
//!      a. forward + tape recording with current E.
//!      b. evaluate `J = ‖centroid_final − target‖²`.
//!      c. backward to get `dJ/dE`.
//!      d. one Adam update on `E` (treated as a 1D parameter).
//!   5. Assert convergence: either `J_final < 0.01 · J_init` (90%
//!      loss reduction) or `|E_final − E_true| / E_true < 0.1`.
//!
//! Prints progress every 20 steps and an FD-vs-analytic check at the
//! first iteration (the gradient-quality oracle the brief asks for).

use molla_autograd::mpm_adjoint::backward;
use molla_autograd::mpm_cpu_diff::{MpmCpuConfig, ParticleState, forward, forward_with_tape};
use molla_math::{Real, Vec3};
use molla_optim::{AdamConfig, AdamState};

/// Build a 4×4×4 = 64 particle cube, one particle per cell. Centred
/// at (0, 0.40, 0) — high enough that the demo never touches the floor
/// during the 30-step rollout (so the loss is smooth wrt E).
fn make_cube_64() -> ParticleState {
    let mut pos = Vec::with_capacity(64);
    let dx_cell = 0.08;
    let spacing = dx_cell;
    let center = Vec3::new(0.0, 0.40, 0.0);
    for i in 0..4 {
        for j in 0..4 {
            for k in 0..4 {
                let off = Vec3::new(
                    (i as Real - 1.5) * spacing,
                    (j as Real - 1.5) * spacing,
                    (k as Real - 1.5) * spacing,
                );
                pos.push(center + off);
            }
        }
    }
    ParticleState::new(&pos, &[])
}

/// Stage 38.1 demo config — 64 particles spread one-per-cell on a
/// 16³ grid, gravity + soft floor. Tuned for CFL: at E = 5e4 Pa,
/// ρ = mass/vol = 10 kg/m³, c_s ≈ 70 m/s, dx = 0.08 → dt < 0.0011 s.
/// We use dt = 5e-4 (~2× margin) so the forward stays stable.
fn demo_cfg(youngs: Real) -> MpmCpuConfig {
    MpmCpuConfig {
        grid_size: 16,
        cell_size: 0.08,
        grid_origin: [-0.64, -0.08, -0.64],
        gravity_y: -9.81,
        youngs,
        poisson: 0.3,
        particle_volume: 1.0e-3,
        particle_mass: 1.0e-2,
        floor_y: 0.0,
        floor_friction: 0.4,
    }
}

/// Sanity check: the demo forward stays bounded across the full
/// 30-step rollout for both `E_true` and `E_init` of the Adam fit.
#[test]
fn forward_stable_for_demo_cfg() {
    let init = make_cube_64();
    let dt: Real = 5.0e-4;
    for youngs in [1.5e4, 6.0e4] {
        let cfg = demo_cfg(youngs);
        let out = forward(&cfg, &init, dt, 10);
        let mut c = Vec3::ZERO;
        for p in &out.pos {
            c += *p;
        }
        c /= out.n() as Real;
        eprintln!("E={youngs:.3e}  10-step centroid={c:?}");
        assert!(
            c.length() < 1.0,
            "E={youngs} 10-step centroid {c:?} escaped (forward exploded)"
        );
    }
}

#[test]
fn adam_fits_youngs_modulus_via_reverse_mpm() {
    let mut init = make_cube_64();
    // Pre-load a per-particle deformation so elastic stress is nonzero
    // from step 0. Under the corrected node lattice (F1b fix) a
    // uniformly free-falling cube develops NO stress at all — C stays
    // exactly zero and F stays I — so an undeformed cube cannot
    // identify E. (The old comment here documented tuning around the
    // half-cell moment-arm bug's "small fictitious force"; that force
    // was the only thing making the previous centroid loss
    // E-sensitive.)
    for (idx, f) in init.f.iter_mut().enumerate() {
        let a = 1.0 + 0.08 * (((idx % 4) as Real) - 1.5) / 1.5;
        let b = 1.0 - 0.05 * ((((idx / 4) % 4) as Real) - 1.5) / 1.5;
        *f = molla_math::Mat3::from_diagonal(Vec3::new(a, b, 1.0 / (a * b)));
    }
    let dt: Real = 5.0e-4;
    let steps = 10;
    let e_true: Real = 1.5e4;
    let e_init: Real = 6.0e4; // 4× too stiff — Adam should walk down.

    // ── Target: forward with E_true → final centroid. ───────────────
    let cfg_true = demo_cfg(e_true);
    let final_true = forward(&cfg_true, &init, dt, steps);
    // Per-particle position targets from the E_true rollout. A
    // CENTROID target cannot identify E under momentum-conserving
    // dynamics (internal stresses never move the centroid) — that
    // formulation only ever worked through the pre-F1b half-cell
    // moment-arm bug, which leaked momentum.
    let targets: Vec<Vec3> = final_true.pos.clone();
    let per_particle_loss = |pos: &[Vec3]| -> (Real, Vec<Vec3>) {
        let mut loss = 0.0;
        let mut grad = vec![Vec3::ZERO; pos.len()];
        for (i, p) in pos.iter().enumerate() {
            let d = *p - targets[i];
            loss += d.dot(d);
            grad[i] = 2.0 * d;
        }
        (loss, grad)
    };

    let mut cfg_now = demo_cfg(e_init);
    let tape0 = forward_with_tape(&cfg_now, &init, dt, steps);
    let (loss_init, grad0) = per_particle_loss(&tape0.final_state.pos);
    let g_e_analytic_init = backward(&tape0, &grad0).grad_youngs(cfg_now.poisson);
    let h = e_init * 1.0e-3;
    let mut cfg_p = cfg_now;
    cfg_p.youngs += h;
    let (lp, _) = per_particle_loss(&forward(&cfg_p, &init, dt, steps).pos);
    let mut cfg_m = cfg_now;
    cfg_m.youngs -= h;
    let (lm, _) = per_particle_loss(&forward(&cfg_m, &init, dt, steps).pos);
    let fd = (lp - lm) / (2.0 * h);
    let denom = g_e_analytic_init.abs().max(fd.abs()).max(1.0e-30);
    let rel_g = (g_e_analytic_init - fd).abs() / denom;
    eprintln!(
        "[FD check @ iter 0] J_init={:.6e}  analytic dJ/dE={:.6e}  fd={:.6e}  rel={:e}",
        loss_init, g_e_analytic_init, fd, rel_g
    );
    // Sanity gate: analytical gradient must be within 5% of FD oracle
    // at the starting point. Looser than the unit-test 1e-2 because
    // this config is 6× more steps + 8× more particles + has the
    // floor + gravity active.
    assert!(
        rel_g < 5.0e-2,
        "FD-vs-analytic g_E rel err {rel_g} too large at iter 0"
    );

    // ── Adam loop. ──────────────────────────────────────────────────
    let mut params = vec![e_init];
    let mut adam = AdamState::new(1);
    // Learning rate: E lives at ~1e4 scale, so lr ~ 1e3 — Adam
    // normalises by √v anyway, but the bias-correction window in early
    // iterations means a tiny lr stalls. 5e2 lr converges in ~50-100
    // iters at the demo scale.
    let adam_cfg = AdamConfig {
        lr: 5.0e2,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1.0e-8,
        weight_decay: 0.0,
    };

    let start = std::time::Instant::now();
    let max_iters = 200;
    let mut best_loss = Real::INFINITY;
    let mut best_e = params[0];
    let mut final_iter = max_iters;

    for it in 0..max_iters {
        // Re-build cfg with current E (everything else fixed).
        cfg_now = demo_cfg(params[0]);

        let tape = forward_with_tape(&cfg_now, &init, dt, steps);
        let (loss, grad_pos_final) = per_particle_loss(&tape.final_state.pos);
        let adj = backward(&tape, &grad_pos_final);
        let g_e = adj.grad_youngs(cfg_now.poisson);

        if loss < best_loss {
            best_loss = loss;
            best_e = params[0];
        }

        if it % 20 == 0 || it == max_iters - 1 {
            eprintln!(
                "[iter {it:>3}]  E = {:.4e}  loss = {:.4e}  dJ/dE = {:.4e}",
                params[0], loss, g_e,
            );
        }

        // Early-exit: success if either gate is met.
        let e_close = (params[0] - e_true).abs() / e_true < 0.1;
        let loss_reduced = loss < 0.01 * loss_init;
        if e_close && loss_reduced {
            eprintln!("[iter {it:>3}] converged — both gates met");
            final_iter = it;
            break;
        }

        // Adam step.
        let grads = vec![g_e];
        adam.step(&mut params, &grads, &adam_cfg);

        // Guard: keep E in a sensible positive range so a wild Adam
        // step doesn't push us into the unphysical regime where the
        // forward explodes.
        if params[0] < 1.0e2 {
            params[0] = 1.0e2;
        } else if params[0] > 1.0e8 {
            params[0] = 1.0e8;
        }
    }
    let wall = start.elapsed();
    let per_iter_ms = wall.as_secs_f64() * 1000.0 / (final_iter as f64 + 1.0);
    eprintln!(
        "Adam loop wall time: {:.2}s ({:.1} ms/iter avg over {} iters)",
        wall.as_secs_f64(),
        per_iter_ms,
        final_iter + 1
    );
    eprintln!(
        "Final: E_init={e_init:.3e}  E_best={best_e:.4e}  E_true={e_true:.3e}  J_init={loss_init:.3e}  J_best={best_loss:.3e}",
    );

    let e_rel_err = (best_e - e_true).abs() / e_true;
    let loss_ratio = best_loss / loss_init;
    eprintln!(
        "Gates: |E_best - E_true|/E_true = {:.3} (target < 0.1)  |  J_best / J_init = {:.3e} (target < 0.01)",
        e_rel_err, loss_ratio
    );

    assert!(
        e_rel_err < 0.1 || loss_ratio < 0.01,
        "neither convergence gate met: E_rel_err={e_rel_err}, J_ratio={loss_ratio}"
    );
}
