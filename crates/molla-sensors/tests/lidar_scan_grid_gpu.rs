//! H4 — Lidar scan-grid integration tests (GPU).
//!
//! Verify that the [`Lidar`] sensor wraps the [`Raycaster`] correctly:
//! a (θ, φ) angular grid is laid out, rays are dispatched, and the
//! resulting [`Scan`] back-references the metadata so range-image
//! lookups work. The test scenes use a TriMesh ground quad so both
//! the H1 mesh-BVH path and the H4 scan-grid layout exercise together.
//!
//! Each test is `#[ignore]` because it requires a real wgpu adapter;
//! run with `make test-gpu P=molla-sensors T=lidar_scan_grid_gpu`.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Transform, Vec3};
use molla_sensors::{Lidar, ScanMetadata};
use molla_sim::{Model, ModelBuilder, State, TriMesh};

/// Serialise wgpu adapter lifetime across the tests in this binary —
/// the same pattern as `sensor_raycast_gpu.rs`.
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

/// Large horizontal quad at `y == y0` (two triangles, CCW from above).
fn big_ground(half_width: f32, y0: f32) -> TriMesh {
    TriMesh {
        vertices: vec![
            Vec3::new(-half_width as f64, y0 as f64, -half_width as f64),
            Vec3::new(half_width as f64, y0 as f64, -half_width as f64),
            Vec3::new(half_width as f64, y0 as f64, half_width as f64),
            Vec3::new(-half_width as f64, y0 as f64, half_width as f64),
        ],
        indices: vec![[0, 1, 2], [0, 2, 3]],
    }
}

/// A downward-hemisphere scan from (0, 5, 0) hits the ground at every
/// cell; the row at `θ = π` (straight down) hits at `t = 5` regardless
/// of φ, and the row with the widest `θ` (most-horizontal) hits at the
/// largest `t`.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=lidar_scan_grid_gpu`"]
fn downward_hemisphere_scan_hits_ground_at_every_cell() {
    let mut b = ModelBuilder::new();
    let mesh_id = b.add_mesh(big_ground(20.0, 0.0));
    b.add_shape_trimesh(BodyId::NONE, mesh_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    // 32 (θ) × 64 (φ) — small but exercises a 2-D grid layout.
    let n_theta: u32 = 32;
    let n_phi: u32 = 64;
    let mut lidar = Lidar::new(&engine, n_theta * n_phi).unwrap();

    let mut scan = ScanMetadata::new(Vec3::new(0.0, 5.0, 0.0), n_theta, n_phi);
    // Steeply-downward cone only — keep `θ` far enough above π/2 that
    // every ray reaches `y = 0` within the 20 m half-width quad and
    // within `max_distance`. At θ = 0.7 π, `t = 5/|cos θ| ≈ 8.5` m,
    // so we stay well inside the 100 m budget.
    scan.theta_range = (0.7 * std::f64::consts::PI, std::f64::consts::PI);
    scan.phi_range = (0.0, 2.0 * std::f64::consts::PI);
    scan.max_distance = 100.0;

    let result = lidar.cast(&engine, &model, &state, &scan).unwrap();
    assert_eq!(result.len(), (n_theta * n_phi) as usize);

    // Every ray in this steep cone hits the ground quad.
    let hits = result.hit_count();
    assert_eq!(
        hits,
        result.len(),
        "expected every cell to hit ({} / {} did)",
        hits,
        result.len()
    );

    // Straight-down row (θ ≈ π) hits at t = 5 for every φ.
    let last_theta = n_theta - 1;
    for col in 0..n_phi {
        let h = result.hit_at(last_theta, col);
        let t = h.t;
        assert!(
            (t - 5.0).abs() < 1.0e-3,
            "θ≈π col {col}: t = {t}, expected 5.0"
        );
        assert!(
            (h.normal - Vec3::Y).length() < 1.0e-2,
            "normal {:?} ≠ +Y at θ≈π col {col}",
            h.normal
        );
    }

    // Most-horizontal row (θ just above π/2) hits at the largest t —
    // strictly greater than the straight-down distance.
    let first_t = result.hit_at(0, 0).t;
    let last_t = result.hit_at(last_theta, 0).t;
    assert!(
        first_t > last_t,
        "expected first-row t {first_t} > last-row t {last_t}"
    );
}

/// Out-of-capacity scans must error rather than silently truncate.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=lidar_scan_grid_gpu`"]
fn scan_exceeding_capacity_errors() {
    let mut b = ModelBuilder::new();
    let mesh_id = b.add_mesh(big_ground(20.0, 0.0));
    b.add_shape_trimesh(BodyId::NONE, mesh_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut lidar = Lidar::new(&engine, 16).unwrap();
    let scan = ScanMetadata::new(Vec3::new(0.0, 5.0, 0.0), 8, 8); // 64 cells > 16.
    let res = lidar.cast(&engine, &model, &state, &scan);
    assert!(res.is_err(), "expected an over-capacity error, got Ok");
}

/// `Scan::hit_at(row, col)` must round-trip the row-major-over-θ
/// layout: the cell direction returned by the back-pointer matches
/// what the metadata says.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=lidar_scan_grid_gpu`"]
fn scan_hit_at_back_pointer_is_consistent() {
    let mut b = ModelBuilder::new();
    let mesh_id = b.add_mesh(big_ground(20.0, 0.0));
    b.add_shape_trimesh(BodyId::NONE, mesh_id, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let n_theta: u32 = 4;
    let n_phi: u32 = 8;
    let mut lidar = Lidar::new(&engine, n_theta * n_phi).unwrap();

    let mut scan = ScanMetadata::new(Vec3::new(0.0, 5.0, 0.0), n_theta, n_phi);
    scan.theta_range = (
        std::f64::consts::FRAC_PI_2 + 0.1,
        std::f64::consts::PI - 0.1,
    );
    scan.phi_range = (0.0, 2.0 * std::f64::consts::PI - 0.1);
    scan.max_distance = 100.0;

    let result = lidar.cast(&engine, &model, &state, &scan).unwrap();
    assert_eq!(result.len(), (n_theta * n_phi) as usize);
    let mut tested = 0u32;
    for row in 0..n_theta {
        for col in 0..n_phi {
            let h = result.hit_at(row, col);
            if h.shape_id.is_none() {
                continue;
            }
            let d = scan.rc_to_direction(row, col);
            // Hit point should lie on the ray `origin + t * d` (within
            // floating-point slop).
            let predicted = Vec3::new(
                scan.origin.x + d.x * h.t,
                scan.origin.y + d.y * h.t,
                scan.origin.z + d.z * h.t,
            );
            let err = (predicted - h.point).length();
            assert!(
                err < 1.0e-3,
                "row {row} col {col}: predicted {predicted:?} ≠ hit point {:?} (err {err})",
                h.point
            );
            tested += 1;
        }
    }
    assert!(
        tested > 0,
        "no cells were tested — the test scene produced zero hits"
    );
}
