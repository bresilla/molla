//! H5 — Cone sub-ray sampling (multi-return) tests.
//!
//! Verifies the H5 layer above [`Raycaster`]:
//!
//!   * `subray_count > 1` fires the central ray plus `K - 1`
//!     Fibonacci-spiral cone samples per cell.
//!   * The per-cell `first_hit_at` / `last_hit_at` / `sub_hits_at`
//!     selectors mirror helios's first / last / all-hit filters.
//!   * Increasing the beam-divergence cone widens the per-cell hit
//!     footprint on a fine-detail target — the macroscopic
//!     "divergence > 0 widens the per-cell hit count" acceptance
//!     from `PLAN_HELIOS.md` H5.
//!
//! Each test is `#[ignore]`; run via
//! `make test-gpu P=molla-sensors T=lidar_multi_return_gpu`.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_math::{Transform, Vec3};
use molla_sensors::{Lidar, ScanMetadata};
use molla_sim::{BodyParams, Model, ModelBuilder, State};

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

/// All sub-rays of a cell aimed at a large flat sphere hit the same
/// surface; first / last / sub_hits_at therefore agree on the
/// shape, and the sub-ray count matches `subray_count`.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=lidar_multi_return_gpu`"]
fn cone_sub_rays_share_surface_when_target_is_large() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    // Big sphere so every sub-ray inside the cone still hits it.
    b.add_shape_sphere(body, 1.0, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let subray_count: u32 = 7;
    let mut lidar = Lidar::new(&engine, subray_count).unwrap();

    // One cell aimed straight down at the sphere with a narrow cone.
    let mut scan = ScanMetadata::new(Vec3::new(0.0, 5.0, 0.0), 1, 1);
    scan.theta_range = (std::f64::consts::PI, std::f64::consts::PI);
    scan.phi_range = (0.0, 0.0);
    scan.subray_count = subray_count;
    scan.beam_divergence = 0.05; // ~3°
    scan.max_distance = 100.0;

    let result = lidar.cast(&engine, &model, &state, &scan).unwrap();
    assert_eq!(result.len(), subray_count as usize);
    let subs = result.sub_hits_at(0, 0);
    assert_eq!(subs.len(), subray_count as usize);
    // Every sub-ray hits the sphere.
    assert!(
        subs.iter().all(|h| h.shape_id == Some(0)),
        "expected all sub-rays to hit shape 0, got {:?}",
        subs.iter().map(|h| h.shape_id).collect::<Vec<_>>()
    );
    // Sub-rays must agree on `first_hit_at <= last_hit_at`.
    let first = result.first_hit_at(0, 0).unwrap();
    let last = result.last_hit_at(0, 0).unwrap();
    assert!(first.t <= last.t, "first.t {} > last.t {}", first.t, last.t);
    // Coverage helpers.
    assert_eq!(result.hit_count(), 1);
    assert_eq!(result.subray_hit_count(), subray_count as usize);
}

/// Increasing the beam-divergence cone widens the per-cell hit
/// footprint on a small target. We scan a 16×16 angular grid at a
/// tiny sphere — without divergence only a few cells aimed near the
/// centre hit it; with a wider cone, more cells record at least one
/// sub-ray hit.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=lidar_multi_return_gpu`"]
fn larger_cone_widens_hit_footprint_on_small_target() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    // Small target so cone widening matters.
    b.add_shape_sphere(body, 0.05, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let subray_count: u32 = 8;
    let n_theta: u32 = 16;
    let n_phi: u32 = 16;
    let mut lidar = Lidar::new(&engine, n_theta * n_phi * subray_count).unwrap();

    let base = |divergence: f64| {
        let mut scan = ScanMetadata::new(Vec3::new(0.0, 5.0, 0.0), n_theta, n_phi);
        // Narrow downward cone centred on the target.
        scan.theta_range = (std::f64::consts::PI - 0.04, std::f64::consts::PI);
        scan.phi_range = (0.0, 2.0 * std::f64::consts::PI);
        scan.subray_count = subray_count;
        scan.beam_divergence = divergence;
        scan.max_distance = 100.0;
        scan
    };

    // Tiny "no-divergence" cone — sub-rays are effectively coincident.
    let narrow = lidar.cast(&engine, &model, &state, &base(0.0)).unwrap();
    let wide = lidar.cast(&engine, &model, &state, &base(0.02)).unwrap();

    let narrow_cells = narrow.hit_count();
    let wide_cells = wide.hit_count();
    eprintln!(
        "cone footprint — narrow (div=0): {narrow_cells} cells, wide (div=0.02): {wide_cells} cells"
    );
    assert!(
        wide_cells > narrow_cells,
        "expected wider cone to hit MORE cells: narrow={narrow_cells} wide={wide_cells}"
    );
}
