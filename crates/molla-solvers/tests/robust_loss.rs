//! Stage 24b — robust-loss CPU verification suite.
//!
//! Three groups:
//!   1. **Gradient parity** — for each loss, sample 1000 random `r`
//!      values and check the analytic gradient against a 5-point
//!      central difference. L1 (sub-gradient at 0) gets a tolerance
//!      band around the kink; everything else hits 1e-4 cleanly.
//!   2. **Pose-fit outlier resistance** — fit a 1-D translation to a
//!      noisy point cloud with 10% outliers; verify Tukey converges
//!      to the inlier mean while plain L2 is pulled toward the
//!      outliers.
//!   3. **IkSolver outlier resistance** — synthetic 7-DOF arm with
//!      multiple position objectives, 10% of which are outliers far
//!      from the true target. Tukey-loss IK beats L2-loss IK on
//!      inlier-only error.

use molla_core::BodyId;
use molla_math::robust_loss::{
    RobustLoss, norm_cauchy, norm_cauchy_grad, norm_huber, norm_huber_grad, norm_l1, norm_l1_grad,
    norm_l2_smooth, norm_l2_smooth_grad, norm_pseudo_huber, norm_pseudo_huber_grad, norm_tukey,
    norm_tukey_grad,
};
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, ModelBuilder};
use molla_solvers::{IkConfig, IkObjective, IkSolver};

// ── Deterministic RNG ─────────────────────────────────────────────

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xCAFE_F00D)
    }
    fn next(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.0
    }
    /// Uniform `[-1, 1]`.
    fn signed(&mut self) -> Real {
        let x = self.next() >> 11;
        let u = (x as Real + 1.0) / ((1u64 << 53) as Real + 1.0);
        2.0 * u - 1.0
    }
}

// ── Gradient parity (group 1) ──────────────────────────────────────

const N_GRAD_SAMPLES: usize = 1_000;
const GRAD_TOL: Real = 1.0e-4;
/// Half-width band around zero where L1's sub-gradient is undefined;
/// FD straddles the kink for any sample in `(-FD_STEP, FD_STEP)`.
const L1_KINK_BAND: Real = 1.0e-2;
const FD_STEP: Real = 1.0e-4;

fn central_diff(loss: impl Fn(Real) -> Real, r: Real) -> Real {
    // 5-point stencil — matches the 1e-4 budget cleanly for f64.
    let h = FD_STEP;
    let f1 = loss(r - 2.0 * h);
    let f2 = loss(r - h);
    let f3 = loss(r + h);
    let f4 = loss(r + 2.0 * h);
    (f1 - 8.0 * f2 + 8.0 * f3 - f4) / (12.0 * h)
}

fn check_gradient<F, G>(name: &str, loss: F, grad: G, range: Real)
where
    F: Fn(Real) -> Real + Copy,
    G: Fn(Real) -> Real + Copy,
{
    let mut rng = Lcg::new(0x4242_BEEF + name.len() as u64);
    let mut max_err: Real = 0.0;
    for _ in 0..N_GRAD_SAMPLES {
        let r = range * rng.signed();
        let analytic = grad(r);
        let fd = central_diff(loss, r);
        let err = (analytic - fd).abs();
        if err > max_err {
            max_err = err;
        }
        // Skip kink samples for L1 — the FD stencil straddles the
        // discontinuity in `sign`.
        if name == "l1" && r.abs() < L1_KINK_BAND {
            continue;
        }
        assert!(
            err < GRAD_TOL,
            "{name}: grad mismatch at r={r}: analytic={analytic}, fd={fd}, err={err}",
        );
    }
    eprintln!("{name}: max grad err vs FD = {max_err:e}");
}

#[test]
fn l1_gradient_matches_fd_outside_kink() {
    check_gradient("l1", norm_l1, norm_l1_grad, 1.0);
}

#[test]
fn smooth_l2_gradient_matches_fd() {
    let eps = 0.05;
    check_gradient(
        "smooth_l2",
        |r| norm_l2_smooth(r, eps),
        |r| norm_l2_smooth_grad(r, eps),
        1.0,
    );
}

#[test]
fn huber_gradient_matches_fd() {
    let delta = 0.5;
    check_gradient(
        "huber",
        |r| norm_huber(r, delta),
        |r| norm_huber_grad(r, delta),
        1.5,
    );
}

#[test]
fn pseudo_huber_gradient_matches_fd() {
    let delta = 0.5;
    check_gradient(
        "pseudo_huber",
        |r| norm_pseudo_huber(r, delta),
        |r| norm_pseudo_huber_grad(r, delta),
        2.0,
    );
}

#[test]
fn tukey_gradient_matches_fd() {
    let c = 1.0;
    // Sample inside the band so FD doesn't straddle the saturation
    // boundary (gradient drops to 0 at |r| = c). Outside-band samples
    // are covered separately by `tukey_zero_outside_band`.
    check_gradient(
        "tukey",
        |r| norm_tukey(r, c),
        |r| norm_tukey_grad(r, c),
        0.95 * c,
    );
}

#[test]
fn cauchy_gradient_matches_fd() {
    let c = 1.0;
    check_gradient(
        "cauchy",
        |r| norm_cauchy(r, c),
        |r| norm_cauchy_grad(r, c),
        2.0,
    );
}

// ── Smooth L2 → L2 limit ───────────────────────────────────────────

#[test]
fn smooth_l2_reduces_to_l2_as_eps_shrinks() {
    // Take a residual r and watch ρ(r) - |r| → 0 as eps → 0.
    let r: Real = 0.5;
    let prev = norm_l2_smooth(r, 1e-2);
    let mid = norm_l2_smooth(r, 1e-4);
    let near_zero = norm_l2_smooth(r, 1e-8);
    eprintln!(
        "smooth_l2(0.5, eps): eps=1e-2 → {prev:.6}, eps=1e-4 → {mid:.6}, eps=1e-8 → {near_zero:.6}",
    );
    // The error scales like O(eps): for |r| ≫ eps,
    // ρ(r) = √(r²+ε²) − ε ≈ |r| + ε²/(2|r|) − ε ≈ |r| − ε.
    // So the limit gap is dominated by `eps` itself; assert each
    // value sits inside a tolerance band of `eps + 1e-12`.
    assert!((mid - r.abs()).abs() < 1.0e-4 + 1e-12);
    assert!((near_zero - r.abs()).abs() < 1.0e-8 + 1e-12);
    // Smaller eps → tighter approximation.
    assert!((near_zero - r.abs()).abs() < (mid - r.abs()).abs() + 1e-12);
    // Gradient → sign(r).
    let g = norm_l2_smooth_grad(r, 1e-9);
    assert!((g - 1.0).abs() < 1e-6);
}

// ── 2D point-cloud pose fit (group 2) ──────────────────────────────

/// Fit a 2D translation `t` to a target point cloud by minimising
/// `Σ ρ(‖x_i - t‖)`. Returns the fitted `t` after `iters` IRLS
/// updates (closed-form on the diagonal weight matrix). Hard-cutoff
/// losses (Tukey) need a warm start that's already inside the
/// inlier band, so we initialise from the plain L2 mean — same
/// convention as Levenberg-Marquardt-IRLS in the IkSolver.
fn fit_translation(loss: RobustLoss, points: &[(Real, Real)], iters: usize) -> (Real, Real) {
    // L2 warm start.
    let mut tx0 = 0.0;
    let mut ty0 = 0.0;
    for &(x, y) in points {
        tx0 += x;
        ty0 += y;
    }
    let n = points.len() as Real;
    let mut t = (tx0 / n, ty0 / n);
    for _ in 0..iters {
        let mut wsum = 0.0;
        let mut tx = 0.0;
        let mut ty = 0.0;
        for &(x, y) in points {
            let rx = x - t.0;
            let ry = y - t.1;
            let r = (rx * rx + ry * ry).sqrt();
            // For a *radial* loss in 2D, the per-residual weight on
            // each component is ρ'(r)/r — the IRLS weight from the
            // module API.
            let w = loss.weight(r);
            wsum += w;
            tx += w * x;
            ty += w * y;
        }
        if wsum > 1e-12 {
            t = (tx / wsum, ty / wsum);
        }
    }
    t
}

#[test]
fn tukey_rejects_outliers_in_point_cloud_fit() {
    // 90 inlier points clustered around (1, 1) with σ=0.05; 10
    // outliers placed at (5, 5) (5σ away in any reasonable scale).
    let mut rng = Lcg::new(0xABCD_1234);
    let mut points: Vec<(Real, Real)> = Vec::with_capacity(100);
    for _ in 0..90 {
        let nx = 1.0 + 0.05 * rng.signed();
        let ny = 1.0 + 0.05 * rng.signed();
        points.push((nx, ny));
    }
    for _ in 0..10 {
        points.push((5.0, 5.0));
    }

    let l2_t = fit_translation(RobustLoss::L2, &points, 1);
    // For Tukey we need IRLS iterations because the weights depend
    // on the current estimate. `c = 1.0` is comfortably larger than
    // the L2-mean residual to the inliers (~0.6 m) so the warm
    // start sees the inliers with non-zero weight, while the
    // outlier residual (~5 m) is far past the rejection boundary.
    // 30 iterations is comfortably enough to converge.
    let tukey_t = fit_translation(RobustLoss::Tukey { c: 1.0 }, &points, 30);

    let l2_err = ((l2_t.0 - 1.0).powi(2) + (l2_t.1 - 1.0).powi(2)).sqrt();
    let tukey_err = ((tukey_t.0 - 1.0).powi(2) + (tukey_t.1 - 1.0).powi(2)).sqrt();

    eprintln!(
        "L2 fit  → ({:.4}, {:.4}), err vs inlier mean = {l2_err:.4}",
        l2_t.0, l2_t.1
    );
    eprintln!(
        "Tukey fit → ({:.4}, {:.4}), err vs inlier mean = {tukey_err:.4}",
        tukey_t.0, tukey_t.1
    );

    // Tukey should land near (1, 1); L2 is pulled toward (1.4, 1.4)
    // because each outlier carries the same weight as an inlier.
    assert!(tukey_err < 0.05, "Tukey err too large: {tukey_err}");
    assert!(
        l2_err > 0.2,
        "L2 should be pulled toward outliers: {l2_err}"
    );
    assert!(
        l2_err > 4.0 * tukey_err,
        "Tukey should beat L2 by at least 4×: tukey={tukey_err} vs l2={l2_err}",
    );
}

// ── IkSolver outlier resistance (group 3) ──────────────────────────

fn box_inertia(mass: Real, half: V3) -> Mat3 {
    let h2 = half * half;
    let m_third = mass / 3.0;
    Mat3::from_diagonal(V3::new(
        m_third * (h2.y + h2.z),
        m_third * (h2.x + h2.z),
        m_third * (h2.x + h2.y),
    ))
}

/// 7-DOF stylised Franka arm, copied from `tests/ik_franka.rs`.
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
fn ik_tukey_beats_l2_on_outlier_targets() {
    // Build a Franka arm. Discover the EE rest position; the "true"
    // target is a small offset from rest. We feed the solver 10
    // *copies* of the same Position objective — 9 at the true
    // target, 1 at an outlier 1 m away. Tukey-loss IK should ignore
    // the outlier and reach the true target; plain L2 IK (the
    // historical default) should split the difference and finish
    // with measurably worse inlier-only error.
    let (model, ee_body) = build_franka();
    let mut state = model.state().unwrap();
    molla_sim::eval_fk(&model, &mut state).unwrap();
    let ee_at_zero = state.body_q.host().unwrap()[ee_body].position;
    let true_target = ee_at_zero + V3::new(0.15, 0.05, -0.10);
    let outlier = ee_at_zero + V3::new(1.2, 0.8, -0.6);

    let mk_objectives = |outliers: usize, weight_per: Real| -> Vec<IkObjective> {
        let mut objs = Vec::with_capacity(10);
        for _ in 0..(10 - outliers) {
            objs.push(IkObjective::Position {
                body: ee_body,
                point_local: V3::ZERO,
                target_world: true_target,
                weight: weight_per,
            });
        }
        for _ in 0..outliers {
            objs.push(IkObjective::Position {
                body: ee_body,
                point_local: V3::ZERO,
                target_world: outlier,
                weight: weight_per,
            });
        }
        objs
    };
    // 10 objectives, weight 1 each. We use the *same* weight for
    // both runs so the comparison is fair — the loss does the
    // outlier-rejection work, not the user-provided weight.
    let objs = mk_objectives(1, 1.0);
    let q0 = vec![0.0; 7];

    // L2 baseline.
    let cfg_l2 = IkConfig {
        loss: RobustLoss::L2,
        max_iter: 200,
        ..Default::default()
    };
    let solver_l2 = IkSolver::new(objs.clone()).with_config(cfg_l2);
    let mut state_l2 = model.state().unwrap();
    let _ = solver_l2.solve(&model, &mut state_l2, &q0).unwrap();
    let pos_l2 = state_l2.body_q.host().unwrap()[ee_body].position;
    let inlier_err_l2 = (pos_l2 - true_target).length();

    // Tukey: c chosen between the inlier residual (~0) and the
    // outlier residual (~1 m) so the outlier sits well past the
    // rejection boundary.
    let cfg_tukey = IkConfig {
        loss: RobustLoss::Tukey { c: 0.4 },
        max_iter: 200,
        ..Default::default()
    };
    let solver_tukey = IkSolver::new(objs).with_config(cfg_tukey);
    let mut state_tukey = model.state().unwrap();
    let _ = solver_tukey.solve(&model, &mut state_tukey, &q0).unwrap();
    let pos_tukey = state_tukey.body_q.host().unwrap()[ee_body].position;
    let inlier_err_tukey = (pos_tukey - true_target).length();

    eprintln!("IK L2    → inlier err = {inlier_err_l2:.4} m, EE at {pos_l2:?}",);
    eprintln!("IK Tukey → inlier err = {inlier_err_tukey:.4} m, EE at {pos_tukey:?}",);
    // Tukey should land within a few mm of the true target.
    assert!(
        inlier_err_tukey < 5.0e-3,
        "Tukey IK didn't reach inlier target: err = {inlier_err_tukey} m",
    );
    // L2 should be measurably worse — the outlier pulls the EE 10+
    // cm away from the true target. We assert a 5× ratio so the
    // assertion isn't fragile to integration-step jitter.
    assert!(
        inlier_err_l2 > 5.0 * inlier_err_tukey + 5.0e-3,
        "L2 should be pulled toward outlier: l2={inlier_err_l2}, tukey={inlier_err_tukey}",
    );
}
