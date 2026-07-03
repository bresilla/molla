//! GPU ray-cast sensor — golden-value tests against hand-computed
//! expectations. Each test is `#[ignore]` because it requires a real
//! wgpu adapter (CI doesn't have one); run with
//! `make test-gpu P=molla-sensors T=raycast_trimesh_gpu` (forwards `-- --ignored` to the test binary).
//!
//! Each `#[test]` brings up its own `Engine::new_gpu()`. When the
//! cargo `--workspace -- --ignored` lane runs all five concurrently
//! inside this binary (rustc's default `--test-threads=N`) the wgpu
//! Vulkan path can hang or crash on adapter creation. The
//! `GPU_LOCK` mutex serialises adapter ownership so the suite's
//! parallel runner can't trip over itself.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sensors::Raycaster;
use molla_sim::{BodyParams, Model, ModelBuilder, State};

/// Serialises wgpu adapter lifetime across the tests in this binary.
/// See module docstring.
static GPU_LOCK: Mutex<()> = Mutex::new(());

/// Bring up GPU + upload the model + a default state, returning the
/// pieces every test needs. Returns `None` if no adapter is
/// available (the caller `expect`s on the result so CI without a
/// GPU surfaces an explicit "GPU adapter required" error).
fn setup(model: Model) -> Option<(Engine, Model, State)> {
    let engine = match Engine::new_gpu() {
        Ok(e) => e,
        Err(_) => return None, // no adapter
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut model = model;
    model.upload_to_device(&device, &queue).ok()?;
    let mut state = model.state().ok()?;
    state.upload_to_device(&device, &queue).ok()?;
    Some((engine, model, state))
}

/// Helper: acquire the GPU lock and return a guard that the caller
/// drops at the end of the test scope. See module docstring.
fn lock_gpu() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Sphere of radius 0.5 at origin; ray from (0, 5, 0) straight down.
/// Expected hit at y = 0.5, t = 4.5, normal = +Y.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu P=molla-sensors T=raycast_trimesh_gpu"]
fn ray_hits_sphere_at_expected_distance() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut sensor = Raycaster::new(&engine, 8).unwrap();
    let hits = sensor
        .cast(
            &engine,
            &model,
            &state,
            &[Vec3::new(0.0, 5.0, 0.0)],
            &[Vec3::new(0.0, -1.0, 0.0)],
            100.0,
        )
        .unwrap();

    assert_eq!(hits.len(), 1);
    let h = hits[0];
    assert!(h.shape_id.is_some(), "expected a hit, got miss");
    assert!((h.t - 4.5).abs() < 1.0e-3, "t = {}, expected 4.5", h.t);
    assert!(
        (h.point - Vec3::new(0.0, 0.5, 0.0)).length() < 1.0e-3,
        "point {:?}",
        h.point
    );
    assert!(
        (h.normal - Vec3::Y).length() < 1.0e-3,
        "normal {:?}",
        h.normal
    );
}

/// Ray that misses the sphere returns `shape_id = None`.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu P=molla-sensors T=raycast_trimesh_gpu"]
fn ray_misses_sphere() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut sensor = Raycaster::new(&engine, 8).unwrap();
    let hits = sensor
        .cast(
            &engine,
            &model,
            &state,
            &[Vec3::new(10.0, 5.0, 0.0)], // far off to the side
            &[Vec3::new(0.0, -1.0, 0.0)],
            100.0,
        )
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert!(
        hits[0].shape_id.is_none(),
        "expected miss, got {:?}",
        hits[0]
    );
}

/// Box of half-extents (1, 0.5, 1) at origin; ray from (0, 5, 0) down.
/// Expected hit at y = 0.5, t = 4.5, normal = +Y.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu P=molla-sensors T=raycast_trimesh_gpu"]
fn ray_hits_box_top_face() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_box(body, Vec3::new(1.0, 0.5, 1.0), Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut sensor = Raycaster::new(&engine, 8).unwrap();
    let hits = sensor
        .cast(
            &engine,
            &model,
            &state,
            &[Vec3::new(0.0, 5.0, 0.0)],
            &[Vec3::new(0.0, -1.0, 0.0)],
            100.0,
        )
        .unwrap();
    let h = hits[0];
    assert!(h.shape_id.is_some());
    assert!((h.t - 4.5).abs() < 1.0e-3, "box top hit t = {}", h.t);
    assert!(
        (h.normal - Vec3::Y).length() < 1.0e-3,
        "normal {:?}",
        h.normal
    );
}

/// Multiple shapes: ray takes the closer hit. Two spheres on the
/// same vertical line at y = 0 (radius 0.5) and y = 3 (radius 0.4).
/// Ray from (0, 5, 0) down should hit the upper (closer) sphere
/// first.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu P=molla-sensors T=raycast_trimesh_gpu"]
fn ray_takes_closer_of_two_spheres() {
    let mut b = ModelBuilder::new();
    let lower = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_sphere(lower, 0.5, Transform::IDENTITY).unwrap();
    let upper = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 3.0, 0.0)));
    b.add_shape_sphere(upper, 0.4, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut sensor = Raycaster::new(&engine, 8).unwrap();
    let hits = sensor
        .cast(
            &engine,
            &model,
            &state,
            &[Vec3::new(0.0, 5.0, 0.0)],
            &[Vec3::new(0.0, -1.0, 0.0)],
            100.0,
        )
        .unwrap();
    let h = hits[0];
    assert!(h.shape_id.is_some());
    // Upper sphere top is at y = 3.4, so t = 5 - 3.4 = 1.6.
    assert!((h.t - 1.6).abs() < 1.0e-3, "t = {}", h.t);
}

/// Ray-cast respects the body pose stored in `state.body_q`. We move
/// a sphere to (5, 0, 0) by writing its initial pose, then cast a
/// horizontal ray from the world origin and expect to hit at t ≈ 4.5.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu P=molla-sensors T=raycast_trimesh_gpu"]
fn ray_hits_translated_body() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(
        BodyParams::new().with_pose(Transform::new(Vec3::new(5.0, 0.0, 0.0), Quat::IDENTITY)),
    );
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut sensor = Raycaster::new(&engine, 8).unwrap();
    let hits = sensor
        .cast(
            &engine,
            &model,
            &state,
            &[Vec3::new(0.0, 0.0, 0.0)],
            &[Vec3::new(1.0, 0.0, 0.0)],
            100.0,
        )
        .unwrap();
    let h = hits[0];
    assert!(h.shape_id.is_some(), "expected a hit, got miss");
    let expected_t: Real = 4.5;
    assert!(
        (h.t - expected_t).abs() < 1.0e-3,
        "t = {}, expected {}",
        h.t,
        expected_t
    );
}
