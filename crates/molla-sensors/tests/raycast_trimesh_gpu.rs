//! H1 — TriMesh raycast integration tests (GPU).
//!
//! The `Raycaster` sensor used to skip TriMesh shapes via the `0xff`
//! sentinel; H1 closes the gap by dispatching the existing
//! `MeshBvhRaycastKernel` against each static TriMesh shape and
//! merging the closest hit into the analytical result. These tests
//! verify that:
//!
//!   * a ray that hits *only* a static TriMesh returns a mesh hit
//!     with the correct `t`, `point`, and `normal`,
//!   * in a mixed scene the closer of the analytical and mesh hits
//!     wins,
//!   * the sentinel + sign conventions match the analytical kernel
//!     (no false positives from the miss path).
//!
//! Each test is `#[ignore]` because it requires a real wgpu adapter
//! (CI doesn't have one); run with
//! `make test-gpu P=molla-sensors T=raycast_trimesh_gpu`.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Transform, Vec3};
use molla_sensors::Raycaster;
use molla_sim::{Model, ModelBuilder, State, TriMesh};

/// Serialise wgpu adapter lifetime across the tests in this binary —
/// the same pattern as `sensor_raycast_gpu.rs`.
static GPU_LOCK: Mutex<()> = Mutex::new(());

fn lock_gpu() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// Bring up GPU + upload the model + a default state.
fn setup(model: Model) -> Option<(Engine, Model, State)> {
    let engine = Engine::new_gpu().ok()?;
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut model = model;
    model.upload_to_device(&device, &queue).ok()?;
    let mut state = model.state().ok()?;
    state.upload_to_device(&device, &queue).ok()?;
    Some((engine, model, state))
}

/// Horizontal quad (two triangles) at `y == y0`, centred at the
/// origin, half-width `r`, CCW from above so the geometric normal
/// points `+Y`.
fn ground_quad(r: f32, y0: f32) -> TriMesh {
    TriMesh {
        vertices: vec![
            Vec3::new(-r as f64, y0 as f64, -r as f64),
            Vec3::new(r as f64, y0 as f64, -r as f64),
            Vec3::new(r as f64, y0 as f64, r as f64),
            Vec3::new(-r as f64, y0 as f64, r as f64),
        ],
        indices: vec![[0, 1, 2], [0, 2, 3]],
    }
}

/// A ray straight down onto a static TriMesh ground quad hits at the
/// expected distance, with the expected world-space hit point and
/// outward normal.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=raycast_trimesh_gpu`"]
fn ray_hits_static_trimesh_ground_at_expected_distance() {
    let mut b = ModelBuilder::new();
    let mesh_id = b.add_mesh(ground_quad(2.0, 0.0));
    b.add_shape_trimesh(BodyId::NONE, mesh_id, Transform::IDENTITY)
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

    assert_eq!(hits.len(), 1);
    let h = hits[0];
    assert!(
        h.shape_id.is_some(),
        "ray missed the TriMesh ground (got {h:?})"
    );
    assert!((h.t - 5.0).abs() < 1.0e-3, "t = {}, expected 5.0", h.t);
    assert!(
        (h.point - Vec3::new(0.0, 0.0, 0.0)).length() < 1.0e-3,
        "hit point {:?}, expected (0,0,0)",
        h.point
    );
    assert!(
        (h.normal - Vec3::Y).length() < 1.0e-3,
        "normal {:?}, expected +Y",
        h.normal
    );
}

/// A ray that misses every primitive must still be reported as a miss
/// after the TriMesh pass — the H1 merge must not invent hits.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=raycast_trimesh_gpu`"]
fn ray_missing_everything_remains_a_miss() {
    let mut b = ModelBuilder::new();
    let mesh_id = b.add_mesh(ground_quad(0.5, 0.0));
    b.add_shape_trimesh(BodyId::NONE, mesh_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut sensor = Raycaster::new(&engine, 8).unwrap();
    // Ray fired horizontally well above the quad — must miss.
    let hits = sensor
        .cast(
            &engine,
            &model,
            &state,
            &[Vec3::new(0.0, 5.0, 0.0)],
            &[Vec3::new(1.0, 0.0, 0.0)],
            100.0,
        )
        .unwrap();

    assert_eq!(hits.len(), 1);
    assert!(
        hits[0].shape_id.is_none(),
        "expected a miss but got {:?}",
        hits[0]
    );
}

/// In a mixed analytical-+-TriMesh scene the closer hit wins. The
/// analytical sphere is the closer hit for the downward ray; a second
/// horizontal ray that clears the sphere must fall through to the
/// TriMesh ground hit.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=raycast_trimesh_gpu`"]
fn analytical_and_trimesh_merge_picks_the_closer_hit() {
    let mut b = ModelBuilder::new();
    // Sphere of radius 0.5 sitting on the quad ground.
    let sphere_body =
        b.add_body(molla_sim::BodyParams::new().with_position(Vec3::new(0.0, 1.0, 0.0)));
    b.add_shape_sphere(sphere_body, 0.5, Transform::IDENTITY)
        .unwrap();
    // Ground quad at y == 0.
    let mesh_id = b.add_mesh(ground_quad(2.0, 0.0));
    b.add_shape_trimesh(BodyId::NONE, mesh_id, Transform::IDENTITY)
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
            &[
                // Straight down through the sphere centre — hits sphere first.
                Vec3::new(0.0, 5.0, 0.0),
                // Down beside the sphere — falls through to the quad.
                Vec3::new(1.5, 5.0, 0.0),
            ],
            &[Vec3::new(0.0, -1.0, 0.0), Vec3::new(0.0, -1.0, 0.0)],
            100.0,
        )
        .unwrap();

    assert_eq!(hits.len(), 2);
    // First ray: closest hit is the top of the sphere at y=1.5.
    let a = hits[0];
    assert!(
        a.shape_id.is_some(),
        "first ray expected to hit the sphere, got miss"
    );
    assert!(
        (a.t - 3.5).abs() < 1.0e-3,
        "sphere t = {}, expected 3.5",
        a.t
    );
    assert!(
        (a.point - Vec3::new(0.0, 1.5, 0.0)).length() < 1.0e-3,
        "sphere hit point {:?}",
        a.point
    );
    // Second ray: clears the sphere, hits the quad at y=0, so t=5.
    let g = hits[1];
    assert!(
        g.shape_id.is_some(),
        "second ray expected to hit the TriMesh, got miss"
    );
    assert!(
        (g.t - 5.0).abs() < 1.0e-3,
        "ground t = {}, expected 5.0",
        g.t
    );
    assert!(
        (g.normal - Vec3::Y).length() < 1.0e-3,
        "ground normal {:?}, expected +Y",
        g.normal
    );
}
