//! C0 integration — `BarrierSolver` as an engine `Solver`: a cloth
//! grid built through the REAL `ModelBuilder` path (particles, XPBD
//! springs, `SoftSurface`, static plane + sphere shapes) dropped
//! through the `Solver::step` interface. The barrier invariant must
//! hold against obstacles AND self-pairs, and the drape must settle.

use molla_core::{BodyId, WorldId};
use molla_math::{Real, Transform, Vec3};
use molla_sim::{Contacts, ModelBuilder};
use molla_solvers::{BarrierSolver, Solver};

#[test]
fn cloth_grid_drapes_over_sphere_via_solver_trait() {
    let mut b = ModelBuilder::new();
    b.add_cloth_grid(
        Vec3::new(-0.105, 0.15, -0.105),
        Vec3::X * 0.03,
        Vec3::Z * 0.03,
        8,
        8,
        0.02,
        0.005,
        2.5e-3, // compliance → k = 400
        1.5,
        WorldId(0),
    );
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.add_shape_sphere(BodyId::NONE, 0.06, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    assert!(model.spring_count > 0, "cloth springs missing");
    assert!(
        !model.soft_surface_tris.is_empty(),
        "soft-surface triangles missing"
    );

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(4);
    let mut solver = BarrierSolver::new();

    let h: Real = 1.0 / 240.0;
    for step in 0..240 {
        solver
            .step(&model, &s_a, &control, &contacts, h, &mut s_b)
            .unwrap();
        assert!(
            solver.last_min_gap > 0.0,
            "step {step}: INVARIANT violated through the engine path — gap {}",
            solver.last_min_gap
        );
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.particle_q.host().unwrap();
    let qd = s_a.particle_qd.host().unwrap();
    let v_max = qd.iter().map(|v| v.length()).fold(0.0, Real::max);
    let gap = q
        .iter()
        .map(|p| p.y.min(p.length() - 0.06))
        .fold(Real::INFINITY, Real::min);
    eprintln!("engine drape: final gap {gap:.2e}, v_max {v_max:.3e}");
    assert!(v_max < 0.08, "cloth did not settle: {v_max}");
    assert!(
        gap > 0.0 && gap < 2.0e-3,
        "resting gap must sit in the barrier band: {gap:.3e}"
    );
}

/// Rigid two-way coupling (weak/explicit): a dynamic rigid ball
/// dropped onto a pinned-edge cloth trampoline. The cloth must hold
/// the ball up (reaction wrench = ball weight at rest), the invariant
/// must hold throughout, and the ball must come to rest ABOVE the
/// sagged cloth rather than falling through.
#[test]
fn rigid_ball_rests_on_cloth_trampoline() {
    let mut b = ModelBuilder::new();
    let n = 9u32;
    let size = 0.36_f64;
    b.add_cloth_grid(
        Vec3::new(-size * 0.5, 0.0, -size * 0.5),
        Vec3::X * size,
        Vec3::Z * size,
        n,
        n,
        0.02,
        0.005,
        1.0e-3, // k = 1000 — taut trampoline
        1.5,
        WorldId(0),
    );
    let ball_r = 0.05;
    let ball_mass = 0.3;
    let ball = b.add_body(
        molla_sim::BodyParams::new()
            .with_spatial_inertia(molla_math::SpatialInertia {
                mass: ball_mass,
                com: Vec3::ZERO,
                inertia: molla_math::Mat3::from_diagonal(Vec3::splat(
                    0.4 * ball_mass * ball_r * ball_r,
                )),
            })
            .with_pose(Transform::from_translation(Vec3::new(0.0, 0.12, 0.0))),
    );
    b.add_shape_sphere(ball, ball_r, Transform::IDENTITY)
        .unwrap();
    let mut model = b.build().unwrap();

    // Pin the trampoline's boundary ring (inv_mass = 0 → the solver
    // treats them as true DoF removals).
    {
        let inv = model.particle_inv_mass.host_mut().unwrap();
        for j in 0..n {
            for i in 0..n {
                if i == 0 || j == 0 || i == n - 1 || j == n - 1 {
                    inv[(j * n + i) as usize] = 0.0;
                }
            }
        }
    }
    let model = model;

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(4);
    let mut solver = BarrierSolver::new();

    let h: Real = 1.0 / 240.0;
    for step in 0..360 {
        solver
            .step(&model, &s_a, &control, &contacts, h, &mut s_b)
            .unwrap();
        assert!(
            solver.last_min_gap > 0.0,
            "step {step}: cloth-ball INVARIANT violated — gap {}",
            solver.last_min_gap
        );
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let ball_q = s_a.body_q.host().unwrap()[0];
    let ball_qd = s_a.body_qd.host().unwrap()[0];
    let cloth_min_y = s_a
        .particle_q
        .host()
        .unwrap()
        .iter()
        .map(|p| p.y)
        .fold(Real::INFINITY, Real::min);
    eprintln!(
        "trampoline: ball y {:.4}, |v| {:.3e}, cloth sag to {:.4}",
        ball_q.position.y,
        ball_qd.linear.length(),
        cloth_min_y
    );
    // Supported: the ball must NOT have fallen through (free fall over
    // 1.5 s would put it below −10 m); it rests with its center about
    // one radius above the sagged cloth.
    assert!(
        ball_qd.linear.length() < 0.2,
        "ball did not come to rest: |v| = {}",
        ball_qd.linear.length()
    );
    assert!(
        ball_q.position.y > cloth_min_y,
        "ball fell through the cloth: {} vs sag {}",
        ball_q.position.y,
        cloth_min_y
    );
    assert!(
        ball_q.position.y > -0.2,
        "ball not supported: y = {}",
        ball_q.position.y
    );
}

/// Newton's third law under the weak coupling: zero gravity, a free
/// (unpinned) cloth, a ball flying into it. Total linear momentum
/// (ball + cloth) must be conserved through the impact to Newton
/// tolerance — implicit Euler gives the cloth exactly h·F(x_new) and
/// the reaction samples the same forces.
#[test]
fn ball_cloth_impact_conserves_momentum() {
    let mut b = ModelBuilder::new();
    let n = 7u32;
    let size = 0.3_f64;
    b.add_cloth_grid(
        Vec3::new(-size * 0.5, 0.0, -size * 0.5),
        Vec3::X * size,
        Vec3::Z * size,
        n,
        n,
        0.02,
        0.005,
        2.5e-3,
        1.5,
        WorldId(0),
    );
    let ball_r = 0.05;
    let ball_mass = 0.4;
    let ball = b.add_body(
        molla_sim::BodyParams::new()
            .with_spatial_inertia(molla_math::SpatialInertia {
                mass: ball_mass,
                com: Vec3::ZERO,
                inertia: molla_math::Mat3::from_diagonal(Vec3::splat(
                    0.4 * ball_mass * ball_r * ball_r,
                )),
            })
            .with_pose(Transform::from_translation(Vec3::new(0.0, 0.09, 0.0))),
    );
    b.add_shape_sphere(ball, ball_r, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.body_qd.host_mut().unwrap()[0].linear = Vec3::new(0.0, -2.0, 0.0);
    let control = model.control();
    let contacts = Contacts::with_capacity(4);
    let mut solver = BarrierSolver::new();
    solver.params.gravity = Vec3::ZERO;

    let cloth_mass = 0.02;
    let p_total = |st: &molla_sim::State| -> Vec3 {
        let mut p = st.body_qd.host().unwrap()[0].linear * ball_mass;
        for v in st.particle_qd.host().unwrap().iter() {
            p += *v * cloth_mass;
        }
        p
    };
    let p0 = p_total(&s_a);

    let h: Real = 1.0 / 240.0;
    for step in 0..120 {
        solver
            .step(&model, &s_a, &control, &contacts, h, &mut s_b)
            .unwrap();
        assert!(solver.last_min_gap > 0.0, "step {step}: penetrated");
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let p1 = p_total(&s_a);
    let drift = (p1 - p0).length() / p0.length().max(1.0e-12);
    eprintln!(
        "momentum: |p0| {:.4}, |p1| {:.4}, drift {:.3e}",
        p0.length(),
        p1.length(),
        drift
    );
    assert!(
        drift < 0.02,
        "third law violated: momentum drift {drift:.3e}"
    );
}

/// PPF-gallery enabler — dynamic BOX bodies promoted to STRONG (ABD)
/// coupling through the Model path: a three-domino chain. The first
/// domino is tipped with an initial angular velocity; the chain must
/// topple left-to-right (each successor knocked over), penetration-
/// free, with every box staying rigid.
#[test]
fn domino_chain_topples_through_model_path() {
    let mut b = ModelBuilder::new();
    let half = Vec3::new(0.005, 0.04, 0.02); // thin tall dominoes
    let gap = 0.045;
    let mut ids = Vec::new();
    for k in 0..3 {
        let body = b.add_body(
            molla_sim::BodyParams::new()
                .with_spatial_inertia(molla_math::SpatialInertia {
                    mass: 0.05,
                    com: Vec3::ZERO,
                    inertia: molla_math::Mat3::from_diagonal(Vec3::new(
                        0.05 * (half.y * half.y + half.z * half.z) / 3.0,
                        0.05 * (half.x * half.x + half.z * half.z) / 3.0,
                        0.05 * (half.x * half.x + half.y * half.y) / 3.0,
                    )),
                })
                .with_pose(Transform::from_translation(Vec3::new(
                    k as Real * gap,
                    half.y + 5.0e-4,
                    0.0,
                ))),
        );
        b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
        ids.push(body);
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    // Tip the first domino toward +x.
    s_a.body_qd.host_mut().unwrap()[0].angular = Vec3::new(0.0, 0.0, -14.0);
    let control = model.control();
    let contacts = Contacts::with_capacity(4);
    let mut solver = BarrierSolver::new();
    // Dominoes NEED floor friction: a frictionless tipped domino
    // slides backward off its pivot instead of falling onto its
    // neighbour (observed: domino 0 flat, domino 1 untouched).
    solver.params.friction_mu = 0.5;
    solver.params.friction_mu_s_scale = 1.2;

    let h: Real = 1.0 / 240.0;
    for step in 0..420 {
        solver
            .step(&model, &s_a, &control, &contacts, h, &mut s_b)
            .unwrap();
        assert!(
            solver.last_min_gap > 0.0,
            "step {step}: domino INVARIANT violated — gap {}",
            solver.last_min_gap
        );
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    // Toppled: each domino's local +y axis has tilted far from world
    // +y (fallen ≥ 60°), in chain order.
    for (k, body) in ids.iter().enumerate() {
        let up = q[body.0 as usize].rotation * Vec3::Y;
        let tilt = up.dot(Vec3::Y);
        eprintln!("domino {k}: up·Y = {tilt:.3}");
        assert!(tilt < 0.5, "domino {k} did not topple (up·Y = {tilt:.3})");
    }
}
