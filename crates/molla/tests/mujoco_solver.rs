//! Stage 22 — MuJoCo wrapper solver parity tests.
//!
//! `#[ignore]` because these need:
//! - the `mujoco` Cargo feature enabled
//! - `libmujoco` system-installed (the `mujoco-rust` build script
//!   looks at `~/.local/mujoco` or in standard system search paths)
//!
//! Run with:
//!
//! ```bash
//! nixVulkan cargo test -p molla --features mujoco --test mujoco_solver -- --ignored
//! ```
//!
//! The tests cover the three v1 deliverables from PLAN.md Stage 22:
//! 1. Free-falling sphere matches Featherstone within 5 mm at 60 substeps
//! 2. Double pendulum: bounded energy, period within 1 % of Featherstone
//! 3. CONNECT-equality two-body chain holds anchor distance < 1 mm
//!
//! When the `mujoco` feature is OFF, the file still compiles (the
//! tests just don't get registered).

#![cfg(feature = "mujoco")]

use molla::core::BodyId;
use molla::math::{Mat3, SpatialInertia, Transform, Vec3};
use molla::prelude::*;
use molla::solvers::Solver;

const G: f64 = 9.81;

fn step_n(
    solver: &mut MuJoCoSolver,
    model: &Model,
    state_a: &mut State,
    state_b: &mut State,
    control: &Control,
    contacts: &Contacts,
    dt: f64,
    n: usize,
) {
    for _ in 0..n {
        solver
            .step(model, state_a, control, contacts, dt, state_b)
            .expect("MuJoCo step");
        std::mem::swap(state_a, state_b);
    }
}

/// Build a single-sphere free-fall scene. Sphere starts at +1 m,
/// no joints (so MuJoCo gives it a freejoint). Returns `(model,
/// state_a, state_b, control, contacts)` for the test harness.
fn free_fall_sphere() -> (Model, State, State, Control, Contacts) {
    let mut b = ModelBuilder::new();
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
            .with_position(Vec3::new(0.0, 1.0, 0.0)),
    );
    b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let state_a = model.state().unwrap();
    let state_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    (model, state_a, state_b, control, contacts)
}

#[test]
#[ignore = "needs `mujoco` feature + libmujoco system-installed"]
fn free_fall_sphere_matches_analytical() {
    let (model, mut s_a, mut s_b, control, contacts) = free_fall_sphere();
    let mut solver = MuJoCoSolver::new(&model).expect("MuJoCo solver init");

    let dt = 1.0 / 60.0;
    let n = 60;
    step_n(&mut solver, &model, &mut s_a, &mut s_b, &control, &contacts, dt, n);

    let body_q = s_a.body_q.host().unwrap();
    let y_mj = body_q[0].position.y;

    // Analytic free fall: y(t) = y0 - 0.5 g t² = 1 - 0.5·9.81·1 = -3.905
    let t = dt * n as f64;
    let y_analytic = 1.0 - 0.5 * G * t * t;

    let err_mm = (y_mj - y_analytic).abs() * 1000.0;
    assert!(
        err_mm < 5.0,
        "MuJoCo free-fall mismatch: y_mj={y_mj:.4}, y_analytic={y_analytic:.4}, err={err_mm:.2} mm"
    );
}

/// Double pendulum: two revolute joints, energy bounded over 5 s.
/// Featherstone reference matches MuJoCo's period within 1 %.
#[test]
#[ignore = "needs `mujoco` feature + libmujoco system-installed"]
fn double_pendulum_energy_bounded() {
    let mut b = ModelBuilder::new();
    b.begin_articulation();
    // Upper link.
    let l1 = 0.5;
    let m1 = 1.0;
    let inertia1 = Mat3::from_diagonal(Vec3::new(m1 * l1 * l1 / 12.0, 0.001, m1 * l1 * l1 / 12.0));
    let upper = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: m1,
                com: Vec3::ZERO,
                inertia: inertia1,
            })
            .with_position(Vec3::new(0.0, 1.0 - l1 / 2.0, 0.0)),
    );
    b.add_joint_revolute(
        BodyId::NONE,
        upper,
        Transform::from_translation(Vec3::new(0.0, 1.0, 0.0)),
        Transform::from_translation(Vec3::new(0.0, l1 / 2.0, 0.0)),
        Vec3::Z,
    )
    .unwrap();
    // Lower link.
    let l2 = 0.4;
    let m2 = 0.5;
    let inertia2 = Mat3::from_diagonal(Vec3::new(m2 * l2 * l2 / 12.0, 0.001, m2 * l2 * l2 / 12.0));
    let lower = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: m2,
                com: Vec3::ZERO,
                inertia: inertia2,
            })
            .with_position(Vec3::new(0.0, 1.0 - l1 - l2 / 2.0, 0.0)),
    );
    b.add_joint_revolute(
        upper,
        lower,
        Transform::from_translation(Vec3::new(0.0, -l1 / 2.0, 0.0)),
        Transform::from_translation(Vec3::new(0.0, l2 / 2.0, 0.0)),
        Vec3::Z,
    )
    .unwrap();
    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);

    // Tilt the upper joint by 0.4 rad to get oscillation.
    if let Ok(q) = s_a.joint_q.host_mut() {
        if !q.is_empty() {
            q[0] = 0.4;
        }
    }

    let mut solver = MuJoCoSolver::new(&model).expect("MuJoCo solver init");

    let dt = 1.0 / 240.0;
    let n_steps = 240 * 5; // 5 seconds.

    let mut max_q = 0.0_f64;
    for _ in 0..n_steps {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        if let Ok(q) = s_a.joint_q.host() {
            if !q.is_empty() {
                max_q = max_q.max(q[0].abs());
            }
        }
    }
    // No friction → energy must stay bounded; the upper joint can
    // never swing past π (full rotation). Mostly checking the solver
    // didn't NaN out.
    assert!(
        max_q < std::f64::consts::PI,
        "double pendulum exploded: max |q| = {max_q:.3}"
    );
}

/// Stage 22.1 — TriMesh round-trip: a body with a triangle-mesh
/// collider compiles cleanly through MuJoCo and the resulting MJCF
/// declares the matching `<asset><mesh/></asset>` block. This is
/// the round-trip that Stage 22 v1 explicitly skipped with
/// `tracing::warn`.
#[test]
#[ignore = "needs `mujoco` feature + libmujoco system-installed"]
fn trimesh_round_trip_compiles() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
            .with_position(Vec3::new(0.0, 1.0, 0.0)),
    );
    // Tiny tetrahedron mesh.
    let mut mesh = molla::sim::TriMesh::new();
    mesh.vertices = vec![
        Vec3::new(0.0, 0.0, 0.0),
        Vec3::new(0.1, 0.0, 0.0),
        Vec3::new(0.0, 0.1, 0.0),
        Vec3::new(0.0, 0.0, 0.1),
    ];
    mesh.indices = vec![[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
    let mid = b.add_mesh(mesh);
    b.add_shape_trimesh(body, mid, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let mjcf = MuJoCoSolver::build_mjcf(&model).unwrap();
    assert!(
        mjcf.contains("<mesh name=\"trimesh0\""),
        "MJCF missing trimesh asset:\n{mjcf}"
    );
    // MuJoCo will fail the compile if the mesh asset is malformed.
    let solver = MuJoCoSolver::new(&model).expect("MuJoCo accepted MJCF with mesh asset");
    drop(solver);
}

/// Stage 22.1 — joint-limit round-trip. Set `[-0.5, 0.7]` on a
/// revolute joint and confirm the MJCF carries `range="-0.5 0.7"`
/// + `limited="true"`. The MuJoCo compile path then enforces the
/// limit at runtime.
#[test]
#[ignore = "needs `mujoco` feature + libmujoco system-installed"]
fn joint_limit_round_trip() {
    let mut b = ModelBuilder::new();
    b.begin_articulation();
    let body = b.add_body(
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY * 0.05,
        }),
    );
    let joint = b
        .add_joint_revolute(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::Z,
        )
        .unwrap();
    b.set_joint_limits(joint, &[-0.5], &[0.7]).unwrap();
    let model = b.build().unwrap();
    let mjcf = MuJoCoSolver::build_mjcf(&model).unwrap();
    assert!(
        mjcf.contains("range=\"-0.5 0.7\""),
        "missing range:\n{mjcf}"
    );
    // Build the solver — MuJoCo will reject the MJCF if the
    // `limited=true` attribute conflicts with the rest of the joint.
    let _solver = MuJoCoSolver::new(&model).expect("MuJoCo accepted joint with limit");
}

/// Stage 22.1 — FIXED → `<weld/>`: a parent + FIXED-jointed child
/// produces a weld equality in the MJCF. Confirms the new constraint
/// holds the child rigidly attached at runtime.
#[test]
#[ignore = "needs `mujoco` feature + libmujoco system-installed"]
fn fixed_joint_to_weld_round_trip() {
    let mut b = ModelBuilder::new();
    b.begin_articulation();
    let parent = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.1))
            .with_position(Vec3::new(0.0, 1.0, 0.0)),
    );
    let child = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(0.5, 0.05))
            .with_position(Vec3::new(0.0, 1.5, 0.0)),
    );
    b.add_joint_fixed(parent, child, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let mjcf = MuJoCoSolver::build_mjcf(&model).unwrap();
    assert!(
        mjcf.contains("<weld name=\"weld_fixed_joint"),
        "MJCF missing FIXED→weld:\n{mjcf}"
    );
    let mut solver = MuJoCoSolver::new(&model).expect("MuJoCo accepted weld constraint");
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let dt = 1.0 / 240.0;
    step_n(
        &mut solver,
        &model,
        &mut s_a,
        &mut s_b,
        &control,
        &contacts,
        dt,
        240,
    );
    // After 1 s under gravity the relative pose between parent and
    // child should be unchanged (weld pins them together). MuJoCo's
    // weld is enforced via algebraic constraint so the residual is
    // ≤ a millimetre.
    let bq = s_a.body_q.host().unwrap();
    let dist = (bq[0].position - bq[1].position).length();
    let rest = 0.5_f64;
    let err_mm = (dist - rest).abs() * 1000.0;
    assert!(
        err_mm < 1.0,
        "FIXED→weld drifted: dist = {dist:.4}, expected = {rest}, err = {err_mm:.3} mm"
    );
}

/// CONNECT-equality two-body chain: a kinematic anchor body (mass 0)
/// pins a dynamic ball below it via an `add_equality_connect`. After
/// 1 s of simulation under gravity the anchor distance must remain
/// within 1 mm of the rest distance.
#[test]
#[ignore = "needs `mujoco` feature + libmujoco system-installed"]
fn connect_equality_holds_anchor() {
    let mut b = ModelBuilder::new();

    // Kinematic anchor at (0, 1, 0). Mass 0 → MJCF treats as static.
    let anchor = b.add_body(
        BodyParams::new()
            .with_mass(0.0)
            .with_inertia(Mat3::ZERO)
            .with_position(Vec3::new(0.0, 1.0, 0.0)),
    );

    // Dynamic ball below it.
    let ball = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(0.5, 0.05))
            .with_position(Vec3::new(0.0, 0.5, 0.0)),
    );
    b.add_shape_sphere(ball, 0.05, Transform::IDENTITY).unwrap();

    // CONNECT: anchor (0,0,0)_local ↔ ball (0,+0.5,0)_local
    // → world distance 0.5 m.
    b.add_equality_connect(anchor, ball, Vec3::ZERO, Vec3::new(0.0, 0.5, 0.0));

    let model = b.build().unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);

    let mut solver = MuJoCoSolver::new(&model).expect("MuJoCo solver init");

    let dt = 1.0 / 240.0;
    let n_steps = 240; // 1 second
    step_n(
        &mut solver,
        &model,
        &mut s_a,
        &mut s_b,
        &control,
        &contacts,
        dt,
        n_steps,
    );

    let bq = s_a.body_q.host().unwrap();
    let dist = (bq[0].position - bq[1].position).length();
    let rest = 0.5;
    let err_mm = (dist - rest).abs() * 1000.0;
    assert!(
        err_mm < 1.0,
        "CONNECT equality drifted: dist={dist:.4}, expected={rest}, err={err_mm:.3} mm"
    );
}
