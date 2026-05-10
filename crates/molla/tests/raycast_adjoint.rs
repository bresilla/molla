//! Stage 37.7 — contract tests for the finite-difference raycast
//! adjoint. CPU-only, no GPU dependencies; runs on every CI matrix.
//!
//! Layout mirrors the spec in `PLAN.md` Stage 37.7 § C:
//!   1. `fd_grad_recovers_loss_minimum` — gradient sign / magnitude.
//!   2. `gradient_descent_converges_to_target` — the *contract* —
//!      50 SGD steps on `params` recover the target sphere position
//!      to within 0.1.
//!   3. `loss_is_zero_at_target` — smoke check.
//!
//! Scene = single sphere we're optimising + a `+Y` plane at `y=0` as
//! a backdrop so off-silhouette rays still terminate against
//! something measurable. The depth image is rendered from a small
//! `8×8` grid of orthographic rays pointing down `−Z`, which gives
//! ~64 pixels with hit information across both objects.

use molla::math::{Real, Vec3};
use molla::sensors::raycast_adjoint::{
    compute_grad_raycast, compute_loss_l2, RayScene,
};
use molla::sensors::DiffRay as Ray;

/// Build the camera grid used by every test in this file. Rays look
/// down `−Z` from `z = +5`, sweeping a `±half` × `±half`-metre window
/// centred at the origin. The window is sized so the sphere (r = 0.5)
/// occupies a meaningful fraction of the image at both start + target
/// positions, and the ray spacing is well below the sphere radius so
/// the FD gradient can pick up sub-pixel motion.
fn build_rays(grid_n: usize, half: Real) -> Vec<Ray> {
    let mut rays = Vec::with_capacity(grid_n * grid_n);
    for iy in 0..grid_n {
        for ix in 0..grid_n {
            let u = (ix as Real + 0.5) / grid_n as Real;
            let v = (iy as Real + 0.5) / grid_n as Real;
            let x = -half + 2.0 * half * u;
            let y = -half + 2.0 * half * v;
            rays.push(Ray::new(
                Vec3::new(x, y, 5.0),
                Vec3::new(0.0, 0.0, -1.0),
            ));
        }
    }
    rays
}

/// Build the canonical test scene: one differentiable sphere with its
/// centre at `params[0..3]` plus a static `+Z` plane backdrop (so
/// rays that miss the sphere still register a finite hit, which lets
/// silhouette pixels carry gradient information through the
/// inf-skipping L2 loss).
fn build_scene() -> RayScene {
    let mut scene = RayScene::new();
    scene.add_sphere(0, 0.5);
    scene.add_plane_z(-2.0); // 7 m in front of the rays at z=+5; behind the sphere.
    scene
}

#[test]
fn loss_is_zero_at_target() {
    let scene = build_scene();
    let rays = build_rays(32, 1.5);
    let target_params = vec![0.5, 0.2, 0.0];
    let target_image = scene.trace(&target_params, &rays);
    let loss = compute_loss_l2(&target_image, &target_image);
    assert!(loss <= 1.0e-6, "loss at target should be ~0 but was {loss}");
}

#[test]
fn fd_grad_recovers_loss_minimum() {
    // Target sphere at +X, optimisation starts at origin: gradient on
    // params[0] (= sphere centre x) must point toward `+X` decrease,
    // i.e. be **negative** so SGD step `params -= lr · grad` moves
    // params[0] toward `+0.4`. Target is close enough (0.4 vs sphere
    // radius 0.5) that the silhouettes overlap and FD has signal.
    let scene = build_scene();
    let rays = build_rays(32, 1.5);
    let target_params = vec![0.4, 0.0, 0.0];
    let target_image = scene.trace(&target_params, &rays);

    let mut params = vec![0.0, 0.0, 0.0];
    let grad = compute_grad_raycast(&scene, &mut params, &target_image, &rays, 1.0e-3);
    // Params must be restored.
    assert_eq!(params, vec![0.0, 0.0, 0.0]);
    assert_eq!(grad.len(), 3);

    // Sign check: ∂J/∂x must be negative (loss drops as x grows
    // toward the target). The magnitude is dominated by silhouette-
    // boundary pixels — large for typical setups, but the spec only
    // demands "right sign + non-trivial magnitude" so we test for a
    // safely-below-zero floor.
    assert!(
        grad[0] < -1.0e-3,
        "expected ∂J/∂x < −1e-3 (sphere should move toward +x), got {}",
        grad[0],
    );
    // Y gradient should be tiny — both spheres lie on the X axis and
    // the ray grid is symmetric in Y.
    assert!(grad[1].abs() < 1.0e-1, "grad[y] = {} unexpectedly large", grad[1]);
    // Z gradient is **not** zero in general: shifting the current
    // sphere along Z translates every silhouette-pixel's hit-distance
    // uniformly, and the X-misalignment between current vs target
    // produces a net depth mismatch that FD legitimately picks up. We
    // don't constrain it here; the convergence test below is the
    // contract.
    let _ = grad[2];
}

#[test]
fn gradient_descent_converges_to_target() {
    // The contract: 50 SGD steps on `params` should land within 0.1
    // of the target sphere position. Target is offset in XY (the
    // silhouette dimensions); the camera looks straight down `−Z` so
    // XY motion deforms the silhouette in a way the FD gradient can
    // chase. Pure-Z translation is intentionally absent — for an
    // orthographic-down-Z camera the depth shift is uniform across
    // every silhouette pixel, which couples poorly with the silhouette-
    // boundary aliasing of the L2 loss + finite-difference probe (the
    // natural workaround is a perspective camera or pseudo-Huber loss
    // — both are fair game for `Stage 37.7.1`'s analytic adjoint).
    let scene = build_scene();
    let rays = build_rays(32, 1.5);
    let target_params = vec![0.2, 0.2, 0.0];
    let target_image = scene.trace(&target_params, &rays);

    let mut params = vec![0.0, 0.0, 0.0];
    // Vanilla SGD per the PLAN.md spec. The lr is tuned for the
    // `O(residual · silhouette_pixels)` gradient magnitude — a very
    // small step keeps us within the silhouette-overlap basin of
    // attraction and avoids the FD aliasing that would otherwise let
    // the optimiser oscillate across silhouette boundaries.
    let lr: Real = 5.0e-4;
    let eps: Real = 1.0e-3;

    for _iter in 0..50 {
        let grad = compute_grad_raycast(&scene, &mut params, &target_image, &rays, eps);
        for (p, g) in params.iter_mut().zip(grad.iter()) {
            *p -= lr * g;
        }
    }

    let dx = params[0] - target_params[0];
    let dy = params[1] - target_params[1];
    let dz = params[2] - target_params[2];
    let err = (dx * dx + dy * dy + dz * dz).sqrt();
    assert!(
        err < 0.1,
        "after 50 steps params={params:?} target={target_params:?} err={err}",
    );
}
