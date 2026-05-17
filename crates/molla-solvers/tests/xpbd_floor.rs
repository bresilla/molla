//! Stage 4.0 sanity test: drop a sphere on a static plane and check
//! that it settles above the plane after enough simulation time.

use molla_core::BodyId;
use molla_geometry::collide;
use molla_math::{SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, ModelBuilder};
use molla_solvers::{Solver, XpbdSolver};

fn bounce_peak_for_restitution(restitution: f64) -> (f64, f64, f64) {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);
    b.set_linear_damping(0.0);
    b.set_default_friction(0.0).unwrap();
    b.set_default_restitution(restitution).unwrap();

    let radius = 0.25;
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, radius))
            .with_position(Vec3::new(0.0, 1.2, 0.0)),
    );
    b.add_shape_sphere(body, radius, Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(32);
    let mut pair_buf = Vec::new();
    let mut solver = XpbdSolver::new().with_iterations(10);
    let dt = 0.0025;

    let mut seen_contact = false;
    let mut peak_after_contact = radius;
    let mut max_vy_after_contact = -f64::INFINITY;
    for _ in 0..700 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        if contacts.count > 0 {
            seen_contact = true;
        }
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let q = s_a.body_q.host().unwrap();
        let qd = s_a.body_qd.host().unwrap();
        if seen_contact {
            peak_after_contact = peak_after_contact.max(q[body.index()].position.y);
            max_vy_after_contact = max_vy_after_contact.max(qd[body.index()].linear.y);
        }
    }
    assert!(seen_contact, "drop test never generated a floor contact");
    (peak_after_contact, max_vy_after_contact, radius)
}

#[test]
fn sphere_settles_on_plane() {
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);

    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(0.0, 5.0, 0.0)),
    );
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();

    // Static plane at y = 0 with +Y normal, attached to the world.
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pair_buf = Vec::new();

    let mut solver = XpbdSolver::new();
    let dt = 0.005;

    // Run 4 seconds — generous enough to bounce-and-settle.
    for _ in 0..800 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    let y = q[0].position.y;
    let v = qd[0].linear.y;

    // Sphere of radius 0.5 should sit at y >= 0.5 (with slight slack
    // tolerance from XPBD's contact slop).
    assert!(
        (0.5 - 1e-3..0.6).contains(&y),
        "sphere settled at y={y}, expected ~0.5"
    );
    // And its vertical velocity should be near zero.
    assert!(v.abs() < 0.1, "sphere still moving, vy={v}");
}

#[test]
fn rigid_restitution_controls_first_bounce_height() {
    let (peak_dead, vy_dead, radius) = bounce_peak_for_restitution(0.0);
    let (peak_bouncy, vy_bouncy, _) = bounce_peak_for_restitution(0.8);

    eprintln!(
        "rigid restitution: e=0 peak={peak_dead:.4}, max_vy={vy_dead:.4}; e=0.8 peak={peak_bouncy:.4}, max_vy={vy_bouncy:.4}"
    );
    assert!(
        peak_dead < radius + 0.08,
        "zero-restitution sphere bounced too high: peak={peak_dead}, radius={radius}"
    );
    assert!(
        vy_dead < 0.25,
        "zero-restitution sphere retained too much upward velocity: max_vy={vy_dead}"
    );
    assert!(
        peak_bouncy > peak_dead + 0.25,
        "high restitution did not increase bounce height enough: e0 peak={peak_dead}, e0.8 peak={peak_bouncy}"
    );
    assert!(
        vy_bouncy > vy_dead + 0.5,
        "high restitution did not produce stronger upward rebound: e0 max_vy={vy_dead}, e0.8 max_vy={vy_bouncy}"
    );
}

#[test]
fn stack_settles_without_growing() {
    // Stack four spheres above a plane — exercises sphere-sphere
    // *and* sphere-plane contacts on the same body. The bug this
    // test catches: under-corrected position projection leaving
    // residual penetration each frame, which an XPBD velocity
    // projection then converts into spurious upward velocity →
    // the stack bounces ever higher instead of settling.
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);
    // Restitution + friction must be set on the builder so that the
    // finalize step bakes them into the per-shape NaN-sentinel slots.
    // Setting `model.default_*` post-build no longer affects shapes
    // (the substitution is one-shot at `build()`).
    b.set_default_restitution(0.3).unwrap();
    b.set_default_friction(0.5).unwrap();

    for i in 0..4 {
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
                .with_position(Vec3::new(0.0, 5.0 + i as f64 * 1.05, 0.0)),
        );
        b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pair_buf = Vec::new();
    let mut solver = XpbdSolver::new().with_iterations(8);

    // Track the high-water mark of the topmost sphere across the
    // simulation. If the bug is back, this peak grows over time.
    let mut max_top_y_after_start = 0.0_f64;
    let warmup_steps = 200;
    for step in 0..1500 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        if step > warmup_steps {
            let q = s_a.body_q.host().unwrap();
            let top_y = q[3].position.y;
            if top_y > max_top_y_after_start {
                max_top_y_after_start = top_y;
            }
        }
    }

    // Initial top sphere centre at y = 5 + 3*1.05 = 8.15. After
    // settle it should sit at ~3.5 (4 stacked spheres of radius 0.5,
    // bottom at 0.5). It must never exceed the initial drop height.
    assert!(
        max_top_y_after_start < 8.5,
        "stack bouncing higher than start: peak top y = {max_top_y_after_start}",
    );

    let q = s_a.body_q.host().unwrap();
    let qd = s_a.body_qd.host().unwrap();
    let top_v = qd[3].linear.length();
    assert!(top_v < 0.5, "top sphere still moving fast: |v| = {top_v}");
    eprintln!(
        "stack settled: bottom y={:.3} top y={:.3} peak top y={:.3}",
        q[0].position.y, q[3].position.y, max_top_y_after_start
    );
}

#[test]
fn two_spheres_dont_overlap() {
    // Two spheres dropped side-by-side; check they don't penetrate.
    let mut b = ModelBuilder::new();
    b.set_angular_damping(0.0);

    let a = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(-0.4, 5.0, 0.0)),
    );
    b.add_shape_sphere(a, 0.5, Transform::IDENTITY).unwrap();

    let c = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(0.4, 5.0, 0.0)),
    );
    b.add_shape_sphere(c, 0.5, Transform::IDENTITY).unwrap();

    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pair_buf = Vec::new();
    let mut solver = XpbdSolver::new().with_iterations(8);

    for _ in 0..800 {
        collide(&model, &s_a, &mut contacts, &mut pair_buf).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let dist = (q[0].position - q[1].position).length();
    assert!(
        dist >= 1.0 - 5e-3,
        "spheres overlap: distance {dist}, expected >= 1.0"
    );
}
