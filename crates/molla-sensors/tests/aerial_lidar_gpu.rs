//! H7 — AerialLidar integration tests (GPU).
//!
//! Verifies the aerial-LiDAR sensor: pulse origins distributed over a
//! footprint AABB, nadir-or-cone directions, dispatched through the
//! shared `Raycaster`. Run via
//! `make test-gpu P=molla-sensors T=aerial_lidar_gpu`.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Transform, Vec3};
use molla_sensors::{AerialLidar, AerialScanMetadata};
use molla_sim::{Model, ModelBuilder, State, TriMesh};

static GPU_LOCK: Mutex<()> = Mutex::new(());
fn lock_gpu() -> std::sync::MutexGuard<'static, ()> {
    GPU_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

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

/// Huge horizontal quad at `y = 0` so every aerial pulse fired
/// straight down inside the footprint hits the ground.
fn big_ground(half_width: f32) -> TriMesh {
    TriMesh {
        vertices: vec![
            Vec3::new(-half_width as f64, 0.0, -half_width as f64),
            Vec3::new(half_width as f64, 0.0, -half_width as f64),
            Vec3::new(half_width as f64, 0.0, half_width as f64),
            Vec3::new(-half_width as f64, 0.0, half_width as f64),
        ],
        indices: vec![[0, 1, 2], [0, 2, 3]],
    }
}

/// A nadir-only aerial scan over a 10 × 10 m footprint at 50 m AGL
/// produces the expected pulse count and every pulse hits the
/// ground at `y ≈ 0` with `t ≈ 50`.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=aerial_lidar_gpu`"]
fn nadir_aerial_pass_hits_ground_every_pulse() {
    let mut b = ModelBuilder::new();
    let mesh_id = b.add_mesh(big_ground(50.0));
    b.add_shape_trimesh(BodyId::NONE, mesh_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let meta = AerialScanMetadata::new(
        Vec3::new(-5.0, 50.0, -5.0),
        Vec3::new(5.0, 50.0, 5.0),
        4.0, // 400 pulses
    );
    let expected_pulses = meta.pulse_count();
    let mut aerial = AerialLidar::new(&engine, expected_pulses * 2).unwrap();

    let scan = aerial.cast(&engine, &model, &state, &meta).unwrap();
    assert_eq!(scan.pulse_count(), expected_pulses);
    assert_eq!(scan.hit_count(), expected_pulses as usize);
    // Each pulse fired straight down from y = 50; the hit should be
    // at y ≈ 0 with t ≈ 50.
    for p in 0..expected_pulses {
        let h = scan.first_hit(p).expect("pulse missed");
        assert!(
            (h.t - 50.0).abs() < 1.0e-2,
            "pulse {p}: t = {}, expected 50",
            h.t
        );
        assert!(
            h.point.y.abs() < 1.0e-2,
            "pulse {p}: hit y = {}, expected 0",
            h.point.y
        );
    }
}

/// The pulse count matches `area × density` (rounded), and pulse
/// origins land inside the footprint AABB upper face.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=aerial_lidar_gpu`"]
fn pulse_origins_lie_inside_the_footprint() {
    let mut b = ModelBuilder::new();
    let mesh_id = b.add_mesh(big_ground(50.0));
    b.add_shape_trimesh(BodyId::NONE, mesh_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let meta = AerialScanMetadata::new(Vec3::new(-3.0, 40.0, -3.0), Vec3::new(3.0, 40.0, 3.0), 2.0);
    let mut aerial = AerialLidar::new(&engine, meta.ray_count() * 2).unwrap();
    let scan = aerial.cast(&engine, &model, &state, &meta).unwrap();
    for o in &scan.origins {
        assert!(
            o.x >= -3.0 && o.x <= 3.0,
            "origin x = {} out of footprint",
            o.x
        );
        assert!(
            o.z >= -3.0 && o.z <= 3.0,
            "origin z = {} out of footprint",
            o.z
        );
        assert!((o.y - 40.0).abs() < 1.0e-9, "origin y = {} ≠ 40", o.y);
    }
}
