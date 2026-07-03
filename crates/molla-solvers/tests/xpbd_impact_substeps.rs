//! INSPECTION.md B1b regression — impact behavior under internal
//! substepping.
//!
//! Root cause (found 2026-07-03): the velocity pass ran a SINGLE
//! Jacobi sweep. For multi-contact bodies the per-contact effective
//! mass under-relaxes the correction (a unit box landing on four
//! corner contacts has denom ≈ 4·inv_m → one sweep removes ~1/4 of
//! the position-projection pop), and under internal substepping the
//! surviving pop — already amplified by dt/n — escaped because later
//! substeps gate out on `v_n_pre ≥ 0`. Measured: a restitution-0 box
//! dropped 0.1 m at substeps=4 left the impact at +2.0 m/s after
//! arriving at −1.4 m/s (an energy PUMP reaching y = 4.6 m by step
//! 400). With iterated sweeps (default 8; friction applied once on
//! the final sweep so the Coulomb caps bound the TOTAL impulse) the
//! same drop settles with a ≤ 1 cm decaying hop.
use molla_core::BodyId;
use molla_geometry::CollisionPipeline;
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, ModelBuilder};
use molla_solvers::{Solver, XpbdSolver};

fn box_inertia(mass: Real, half: Vec3) -> Mat3 {
    let f = mass / 3.0;
    Mat3::from_diagonal(Vec3::new(
        f * (half.y * half.y + half.z * half.z),
        f * (half.x * half.x + half.z * half.z),
        f * (half.x * half.x + half.y * half.y),
    ))
}

#[test]
fn restitution_zero_drop_settles_under_internal_substeps() {
    for substeps in [1usize, 4] {
        let half = Vec3::splat(0.5);
        let mut b = ModelBuilder::new();
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: 1.0,
                    com: Vec3::ZERO,
                    inertia: box_inertia(1.0, half),
                })
                .with_position(Vec3::new(0.0, 0.6, 0.0)),
        );
        b.add_shape_box(body, half, Transform::IDENTITY).unwrap();
        b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
            .unwrap();
        let mut model = b.build().unwrap();
        model.default_friction = 0.4;
        model.default_restitution = 0.0;
        model.contact_margin = 0.02;

        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let mut contacts = Contacts::with_capacity(64);
        let mut pipeline = CollisionPipeline::new();
        let mut solver = XpbdSolver::new().with_iterations(8).with_substeps(substeps);
        let dt: Real = 0.005;
        let mut max_y_after_touch = 0.0_f64;
        let mut touched = false;
        for step in 0..400 {
            pipeline.collide(&model, &s_a, &mut contacts).unwrap();
            solver
                .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
            let q = s_a.body_q.host().unwrap()[0];
            let v = s_a.body_qd.host().unwrap()[0].linear;
            if q.position.y < 0.505 {
                touched = true;
            }
            if touched {
                max_y_after_touch = max_y_after_touch.max(q.position.y);
            }
            if substeps == 4 && step == 28 {
                unsafe { std::env::set_var("B1B_DEBUG", "1") };
            }
            if substeps == 4 && step == 30 {
                unsafe { std::env::remove_var("B1B_DEBUG") };
            }
            if substeps == 4 && (28..32).contains(&step) {
                eprintln!(
                    "substeps {substeps} step {step}: y = {:.5}, vy = {:.5}, contacts = {}",
                    q.position.y, v.y, contacts.count
                );
            }
        }
        let q = s_a.body_q.host().unwrap()[0];
        let v = s_a.body_qd.host().unwrap()[0].linear;
        eprintln!(
            "substeps {substeps}: max y after touchdown = {max_y_after_touch:.4}, final y = {:.4}, vy = {:.4}",
            q.position.y, v.y
        );
        assert!(touched, "substeps {substeps}: box never touched down");
        assert!(
            max_y_after_touch < 0.53,
            "substeps {substeps}: bounce too high ({max_y_after_touch:.4}) — impact pop escaping the velocity pass"
        );
        assert!(
            (q.position.y - 0.5).abs() < 0.01 && v.y.abs() < 0.1,
            "substeps {substeps}: box did not settle (y = {:.4}, vy = {:.4})",
            q.position.y,
            v.y
        );
    }
}
