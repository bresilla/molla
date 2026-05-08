//! One sphere, free fall, semi-implicit Euler. Stage 1 oracle target:
//! trajectory matches reference within 1e-6 m at t=1.0s.

use molla::prelude::*;

use crate::harness::{run_with_newton, OracleBody, Scene};

#[test]
#[ignore = "requires MOLLA_NEWTON_PYTHON pointing at a newton-physics venv"]
fn free_fall_matches_newton_within_1e_6() {
    let scene = Scene {
        gravity: Vec3::new(0.0, -9.81, 0.0),
        bodies: vec![OracleBody {
            mass: 1.0,
            radius: 0.5,
            position: Vec3::new(0.0, 10.0, 0.0),
            linear_velocity: Vec3::ZERO,
        }],
        solver: "semi_implicit",
        dt: 0.01,
        steps: 100,
    };

    let reference = run_with_newton(&scene).expect("newton oracle failed");

    // Run molla.
    let mut b = ModelBuilder::new();
    b.set_world_gravity(WorldId(0), scene.gravity);
    b.set_angular_damping(0.0);
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(scene.bodies[0].position),
    );
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
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

        let our_q = s_a.body_q.host().unwrap()[0];
        let ref_q = reference.body_q[step][0];
        let dx = (our_q.position - ref_q.position).length();
        assert!(
            dx < 1e-6,
            "step {step}: position drift {dx} > 1e-6 (got {:?}, ref {:?})",
            our_q.position,
            ref_q.position,
        );
    }
}
