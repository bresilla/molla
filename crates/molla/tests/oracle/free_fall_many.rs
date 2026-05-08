//! 100 spheres, free fall, semi-implicit Euler. Stage 1 oracle target:
//! per-body position drift below 1e-5 m at t=1.0s.

use molla::prelude::*;

use crate::harness::{run_with_newton, OracleBody, Scene};

#[test]
#[ignore = "requires MOLLA_NEWTON_PYTHON pointing at a newton-physics venv"]
fn free_fall_many_matches_newton_within_1e_5() {
    let n = 100usize;
    let bodies: Vec<OracleBody> = (0..n)
        .map(|i| OracleBody {
            mass: 1.0,
            radius: 0.5,
            position: Vec3::new(i as f64 * 1.5, 10.0, 0.0),
            linear_velocity: Vec3::ZERO,
        })
        .collect();

    let scene = Scene {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        bodies: bodies.clone(),
        solver: "semi_implicit",
        dt: 0.01,
        steps: 100,
    };

    let reference = run_with_newton(&scene).expect("newton oracle failed");

    let mut b = ModelBuilder::new();
    b.set_world_gravity(WorldId(0), scene.gravity);
    b.set_angular_damping(0.0);
    for body in &bodies {
        let id = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(body.mass, body.radius))
                .with_position(body.position),
        );
        b.add_shape_sphere(id, body.radius, Transform::IDENTITY).unwrap();
    }
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = SemiImplicitSolver::new();

    for step in 0..scene.steps {
        solver
            .step(&model, &s_a, &control, &contacts, scene.dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);

        let q = s_a.body_q.host().unwrap();
        for (j, our) in q.iter().enumerate().take(n) {
            let ref_q = reference.body_q[step][j];
            let dx = (our.position - ref_q.position).length();
            assert!(
                dx < 1e-5,
                "step {step} body {j}: drift {dx} > 1e-5 (got {:?}, ref {:?})",
                our.position,
                ref_q.position,
            );
        }
    }
}
