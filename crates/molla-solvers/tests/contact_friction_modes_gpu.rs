//! GPU XPBD rolling + torsional friction parity. Mirrors the CPU
//! tests in `contact_friction_modes.rs` — both `μ_rolling` and
//! `μ_torsional` should damp the matching angular DOF on the GPU
//! kernel just like the CPU one.
//!
//! Both tests use a per-test `Engine::new_gpu()`. When the cargo
//! `--workspace -- --ignored` lane runs both in parallel inside the
//! same binary (the rustc default `--test-threads=N`) the wgpu
//! Vulkan path SIGSEGVs on adapter destruction. We serialise the
//! `run_gpu` call sites behind `GPU_LOCK` so the two tests share
//! the GPU sequentially even when cargo runs them on separate
//! threads. Make-target invocations
//! (`make test-gpu` -> `--test-threads=1`) hit the same path.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_core::BodyId;
use molla_geometry::CollisionPipeline;
use molla_math::{Mat3, SpatialInertia, SpatialVector, Transform, Vec3 as V3};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder};
use molla_solvers::{Solver, XpbdSolver};

/// Serialises the wgpu adapter lifetime across the two `#[test]` fns
/// in this binary. See module docstring for why.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn sphere_inertia(mass: f64, r: f64) -> Mat3 {
    let i = (2.0 / 5.0) * mass * r * r;
    Mat3::from_diagonal(V3::new(i, i, i))
}

fn build(coulomb: f64, rolling: f64, torsional: f64) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    b.set_angular_damping(0.0);
    let radius = 0.1_f64;
    let mass = 1.0_f64;
    let inertia = sphere_inertia(mass, radius);
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass,
                com: V3::ZERO,
                inertia,
            })
            .with_position(V3::new(0.0, radius, 0.0)),
    );
    let sphere = b
        .add_shape_sphere(body, radius, Transform::IDENTITY)
        .unwrap();
    let plane = b
        .add_shape_plane(BodyId::NONE, V3::Y, Transform::IDENTITY)
        .unwrap();
    b.set_shape_material(sphere, coulomb, 0.0).unwrap();
    b.set_shape_material(plane, coulomb, 0.0).unwrap();
    b.set_shape_friction_modes(sphere, rolling, torsional)
        .unwrap();
    b.build().unwrap()
}

fn run_gpu(mut model: Model, w0: V3, n_steps: usize) -> V3 {
    // Serialise the GPU path across tests in this binary; the
    // wgpu Vulkan adapter does not survive concurrent
    // create/destroy from two threads on the dev driver. Holding
    // the lock for the whole `run_gpu` body keeps the engine
    // alive for the duration of the test (the lock is poisoned-
    // safe via `unwrap_or_else(PoisonError::into_inner)`; we
    // accept poisoned state since the failing test's panic is
    // already the visible signal we care about).
    let _gpu_lock = GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("gpu engine");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.body_qd.host_mut().unwrap()[0] = SpatialVector::new(V3::ZERO, w0);
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(&device, &queue).unwrap();
    let mut contacts = Contacts::with_capacity(64);
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut solver = XpbdSolver::new_gpu(&engine, model.body_count.max(1) as u32)
        .unwrap()
        .with_iterations(8);
    let mut pipeline = CollisionPipeline::new();
    let dt = 0.005_f64;
    for _ in 0..n_steps {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        contacts.upload_to_device(&device, &queue).unwrap();
        // Re-upload qd because the mouse-grab path expects host to
        // be authoritative; the rigid solver reads device-side qd
        // via the bind group. Initial frame: state was just
        // uploaded so device matches host. Subsequent frames: solver
        // wrote device, we swap, then download — both stay in sync.
        s_a.body_qd.upload(&queue).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
        s_a.body_q.download(&device, &queue).unwrap();
        s_a.body_qd.download(&device, &queue).unwrap();
    }
    s_a.body_qd.host().unwrap()[0].angular
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn rolling_friction_damps_horizontal_axis_spin_gpu() {
    let w_zero = run_gpu(build(0.0, 0.0, 0.0), V3::new(10.0, 0.0, 0.0), 60);
    let w_with = run_gpu(build(0.0, 0.5, 0.0), V3::new(10.0, 0.0, 0.0), 60);
    eprintln!(
        "GPU horizontal spin |ω| after 0.3 s: roll=0 → {:.4}, roll=0.5 → {:.4}",
        w_zero.length(),
        w_with.length()
    );
    assert!(
        w_zero.length() > 9.0,
        "no Coulomb + no rolling on GPU: should keep ~10 rad/s, got {}",
        w_zero.length()
    );
    assert!(
        w_with.length() < w_zero.length() - 1.0,
        "GPU rolling friction did nothing: zero={}, with={}",
        w_zero.length(),
        w_with.length()
    );
}

#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn torsional_friction_damps_normal_axis_spin_gpu() {
    let w_zero = run_gpu(build(0.7, 0.0, 0.0), V3::new(0.0, 10.0, 0.0), 60);
    let w_with = run_gpu(build(0.7, 0.0, 2.0), V3::new(0.0, 10.0, 0.0), 60);
    eprintln!(
        "GPU normal-axis spin |ω.y| after 0.3 s: tors=0 → {:.4}, tors=2.0 → {:.4}",
        w_zero.y.abs(),
        w_with.y.abs()
    );
    assert!(
        w_zero.y.abs() > 9.0,
        "Coulomb shouldn't damp normal-axis spin on GPU, got {}",
        w_zero.y.abs()
    );
    assert!(
        w_with.y.abs() < w_zero.y.abs() - 0.3,
        "GPU torsional friction did nothing: zero={}, with={}",
        w_zero.y.abs(),
        w_with.y.abs()
    );
}
