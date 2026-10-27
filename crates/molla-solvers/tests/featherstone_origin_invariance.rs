//! Featherstone forward-dynamics origin-invariance check.
//!
//! A single revolute pendulum is a closed physical system: its angular
//! acceleration about the joint axis depends only on the geometry
//! *relative to the joint* (lever arm, inertia, axis) and on gravity —
//! never on where the world coordinate origin happens to sit. This
//! test builds the identical pendulum at two different world
//! translations and asserts Featherstone produces the same `qdd`.
//!
//! It also cross-checks the magnitude against the closed-form rigid
//! pendulum `qdd = -(m g d sinθ) / (I_com + m d²)`.

use molla_core::BodyId;
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder};
use molla_solvers::{FeatherstoneSolver, Solver};

const MASS: Real = 1.5;
const I_COM: Real = 0.04; // isotropic inertia about the COM
const LEVER: Real = 0.5; // joint -> body COM distance (along body -Y)

/// Single revolute pendulum (axis +Z) whose joint sits at `origin` in
/// world space and whose body COM hangs `LEVER` below it at rest.
fn build_pendulum(origin: Vec3) -> Model {
    let mut b = ModelBuilder::new();
    b.begin_articulation();
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: MASS,
                com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * I_COM,
            })
            .with_position(origin + Vec3::new(0.0, -LEVER, 0.0)),
    );
    // Joint frame in the world: at `origin`. Joint frame in the body:
    // `LEVER` above the body COM. At q = 0 the body hangs straight down.
    let x_p = Transform::from_translation(origin);
    let x_c = Transform::from_translation(Vec3::new(0.0, LEVER, 0.0));
    b.add_joint_revolute(BodyId::NONE, body, x_p, x_c, Vec3::Z)
        .unwrap();
    b.build().unwrap()
}

/// Featherstone angular acceleration of the pendulum at joint angle
/// `q0`, recovered from a single tiny step started from rest.
fn pendulum_qdd(origin: Vec3, q0: Real) -> Real {
    let model = build_pendulum(origin);
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.joint_q.host_mut().unwrap()[0] = q0;
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();
    let dt: Real = 1.0e-5;
    solver
        .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
        .unwrap();
    // Started from qd = 0, so qd_after ≈ qdd · dt.
    s_b.joint_qd.host().unwrap()[0] / dt
}

#[test]
fn featherstone_single_pendulum_qdd_is_origin_invariant() {
    let q0: Real = 0.7;
    let at_origin = pendulum_qdd(Vec3::ZERO, q0);
    let translated = pendulum_qdd(Vec3::new(5.0, 2.0, -3.0), q0);

    // Closed-form reference: I about the joint = I_com + m·d² (parallel
    // axis); restoring torque = -m·g·d·sinθ.
    let g: Real = 9.81;
    let i_joint = I_COM + MASS * LEVER * LEVER;
    let analytic = -(MASS * g * LEVER * q0.sin()) / i_joint;

    assert!(
        (at_origin - translated).abs() <= 1.0e-6 * at_origin.abs().max(1.0),
        "Featherstone pendulum qdd depends on world-origin placement \
         (it must not): at_origin={at_origin}, translated={translated}"
    );
    assert!(
        (at_origin - analytic).abs() <= 1.0e-3 * analytic.abs().max(1.0),
        "Featherstone pendulum qdd disagrees with the closed-form rigid \
         pendulum: featherstone={at_origin}, analytic={analytic}"
    );
}
