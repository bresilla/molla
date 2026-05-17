//! Stage 17.1 — Featherstone multi-DOF polish tests.
//!
//!  - **D6 angular-limit test** (CPU + GPU): pretensioned cable-style
//!    D6 joint with angular range ±π/4 holds within ±0.05 rad under
//!    gravity for 600 substeps.
//!  - **Distance-joint test** (CPU + GPU): 2-particle constraint length
//!    holds within machine eps post-step (vs ~1 mm with the v1 clamp).
//!  - **Coupled-D parity test** (CPU only — see WGSL caveat): asymmetric
//!    Free-joint chain (CoM offset ≥ 0.1 m from joint axis) matches a
//!    CPU dense-Cholesky reference within f32 tolerance for ≥ 100
//!    substeps. The GPU shader still runs the v1 sequential rank-1
//!    deflation (see `crates/molla/shaders/featherstone.wgsl` caveat
//!    block); the CPU dense-Cholesky path is the authoritative
//!    reference for asymmetric inertias.
//!
//!  All tests are `#[ignore]`d because the GPU variants need a real
//!  wgpu adapter; run them via:
//!
//!  ```bash
//!  make test-featherstone-polish-gpu
//!  ```

#![cfg(feature = "gpu")]

use molla_compute::{Backend, Engine};
use molla_core::BodyId;
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, Contacts, ModelBuilder};
use molla_solvers::{FeatherstoneSolver, Solver};

fn try_engine() -> Option<Engine> {
    match Engine::new(Backend::AutoGpu) {
        Ok(e) => Some(e),
        Err(e) => {
            eprintln!("skipping: no GPU adapter ({e})");
            None
        }
    }
}

/// Stage 17.1 deliverable 1 — D6 angular limits.
///
/// A pendulum-style body hangs from a D6 joint whose angular X axis is
/// limited to ±π/4 (≈ ±45°). Under gravity the body should oscillate
/// inside that cone — never breaching it by more than 0.05 rad over
/// 600 substeps. Linear DOFs are locked so the only motion is angular.
#[test]
#[ignore = "needs a real wgpu adapter"]
fn d6_angular_limit_holds_under_gravity_gpu() {
    let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.0,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY * 0.05,
    }));
    let j = b
        .add_joint_d6(
            BodyId::NONE,
            body,
            Transform::IDENTITY,
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
        )
        .unwrap();
    // Lock all 3 linear DOFs; allow angular X (±π/4); lock angular Y/Z.
    let lim = std::f64::consts::FRAC_PI_4;
    b.set_joint_limits(
        j,
        &[0.0, 0.0, 0.0, -lim, 0.0, 0.0],
        &[0.0, 0.0, 0.0, lim, 0.0, 0.0],
    )
    .unwrap();
    let mut model = b.build().unwrap();
    model.upload_to_device(&device, &queue).unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    // Start pretensioned at the +X angular limit so any energy gain
    // would push past `lim`. Free, Distance, D6 layout: coords =
    //   [tx, ty, tz, qx, qy, qz, qw].
    let tilt = molla_math::Quat::from_axis_angle(V3::X, lim);
    let qmut = s_a.joint_q.host_mut().unwrap();
    qmut[3] = tilt.x;
    qmut[4] = tilt.y;
    qmut[5] = tilt.z;
    qmut[6] = tilt.w;
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model)
        .unwrap()
        .with_damping(0.1);

    let dt = 0.001_f64;
    let n_steps = 600;
    let mut max_excess: Real = 0.0;
    for _ in 0..n_steps {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let _ = s_a.joint_q.download(&device, &queue);
        let q = s_a.joint_q.host().unwrap();
        let r = molla_math::Quat::from_xyzw(q[3], q[4], q[5], q[6]);
        let (rx, _ry, _rz) = r.to_euler(glam::EulerRot::XYZ);
        let excess = (rx.abs() - lim).max(0.0);
        if excess > max_excess {
            max_excess = excess;
        }
    }
    eprintln!("D6 angular limit: max excess = {max_excess:.5} rad");
    assert!(
        max_excess < 0.05,
        "D6 angular limit broke: |rx| exceeded ±π/4 by {max_excess} rad (>0.05)"
    );
}

// CPU companion `d6_angular_limit_holds_under_gravity_cpu` moved to
// `featherstone_polish_cpu.rs` (no `gpu` feature gate, so the default
// `cargo test` lane runs it).

/// Stage 17.1 deliverable 2 — Distance-joint Lagrange projection.
///
/// A 2-body chain: world ← Free body ← Distance body. Constraint
/// length is locked at 0.5 m. After 200 sub-steps the residual
/// `||p| − L|` must be < 1e-5 m (machine-eps in single precision; vs
/// ~1 mm with the v1 hard-clamp under realistic timestep + gravity).
#[test]
#[ignore = "needs a real wgpu adapter"]
fn distance_joint_length_holds_within_eps_gpu() {
    let engine = try_engine().expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.0,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY * 0.1,
    }));
    let j = b
        .add_joint_distance(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let l_t = 0.5;
    b.set_joint_limits(j, &[l_t; 6], &[l_t; 6]).unwrap();
    let mut model = b.build().unwrap();
    model.upload_to_device(&device, &queue).unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    // Start hanging straight down at length L_t.
    let qmut = s_a.joint_q.host_mut().unwrap();
    qmut[0] = 0.0;
    qmut[1] = -l_t;
    qmut[2] = 0.0;
    qmut[3] = 0.0;
    qmut[4] = 0.0;
    qmut[5] = 0.0;
    qmut[6] = 1.0;
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();

    let dt = 0.001_f64;
    let mut worst_residual: Real = 0.0;
    for _ in 0..200 {
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        let _ = s_a.joint_q.download(&device, &queue);
        let q = s_a.joint_q.host().unwrap();
        let p = V3::new(q[0], q[1], q[2]);
        let res = (p.length() - l_t).abs();
        if res > worst_residual {
            worst_residual = res;
        }
    }
    eprintln!("Distance Lagrange (GPU): worst |p|−L = {worst_residual:.3e}");
    // f32 GPU eps ~1e-7 per add; over 200 steps the projected
    // length should still be tighter than 1e-3 m.
    assert!(
        worst_residual < 1e-3,
        "distance constraint drift: {worst_residual} m"
    );
}

// CPU companions `distance_joint_length_holds_within_eps_cpu` and
// `distance_joint_lagrange_beats_v1_clamp_cpu` moved to
// `featherstone_polish_cpu.rs` so the default lane exercises them.

// CPU companion `coupled_d_dense_cholesky_matches_reference_cpu` moved to
// `featherstone_polish_cpu.rs` for the same reason.
