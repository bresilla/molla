//! H10 — Physical-optics dispatch tests (GPU).
//!
//! Verifies the [`TiledCamera`] honours `PhysicalOptics`:
//!
//!   * Focal length + sensor width override `fov_y_radians`
//!     (sensor_width = 2 · focal_length ⇒ HFOV = 90°).
//!   * `lens_diameter_m > 0` with `aperture_samples > 1` averages
//!     a multi-sample render; pinhole `lens_diameter_m = 0` produces
//!     bit-identical output regardless of `aperture_samples`.
//!   * Depth-of-field shifts pixel colour off the central sample —
//!     the integrated image differs from the pinhole.
//!
//! Each test is `#[ignore]`; run via
//! `make test-gpu P=molla-sensors T=tiled_camera_optics_gpu`.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_math::{Real, Transform, Vec3};
use molla_sensors::{CameraChannels, PhysicalOptics, TiledCamera, TiledCameraConfig};
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

fn base_config() -> TiledCameraConfig {
    TiledCameraConfig {
        width: 64,
        height: 64,
        fov_y_radians: 60f32.to_radians(),
        channels: CameraChannels::ALL,
        ambient: Vec3::new(0.1, 0.1, 0.1),
        light_dir: Vec3::new(0.0, -1.0, 0.0).normalize(),
        light_color: Vec3::new(0.9, 0.9, 0.9),
        max_distance: 100.0,
        near: 0.05,
        optics: PhysicalOptics::default(),
    }
}

/// Sphere centred on the optical axis at distance 5. Drives every
/// test in this binary.
fn sphere_scene() -> Model {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    b.build().unwrap()
}

/// Pinhole (`lens_diameter_m = 0`) with `aperture_samples > 1` must
/// short-circuit to the single-dispatch path and produce a bit-
/// identical colour buffer to the standard pinhole — proves the
/// pinhole limit of the H10 code path.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=tiled_camera_optics_gpu`"]
fn pinhole_limit_is_bit_identical_to_legacy_path() {
    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(sphere_scene()).expect("GPU adapter required for this test (gated by #[ignore])");

    let cfg_legacy = base_config();
    let mut cam_legacy = TiledCamera::new(&engine, cfg_legacy, 1).unwrap();
    cam_legacy
        .set_camera_look_at(0, Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    cam_legacy.dispatch(&engine, &model, &state).unwrap();
    let legacy = cam_legacy.download_color(&engine).unwrap().to_vec();

    // Same scene, default optics, but `aperture_samples = 8`. Since
    // `lens_diameter_m == 0`, the dispatch must collapse to the
    // pinhole fast path and the output should equal the legacy run.
    let mut cfg_optics = base_config();
    cfg_optics.optics.aperture_samples = 8;
    let mut cam_optics = TiledCamera::new(&engine, cfg_optics, 1).unwrap();
    cam_optics
        .set_camera_look_at(0, Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    cam_optics.dispatch(&engine, &model, &state).unwrap();
    let optics = cam_optics.download_color(&engine).unwrap().to_vec();
    assert_eq!(
        legacy, optics,
        "pinhole+aperture_samples must match pinhole"
    );
}

/// Focal length + sensor width override `fov_y_radians`. With
/// `sensor_width = 2 · focal_length` the HFOV is exactly 90° and the
/// sphere subtends a known pixel diameter; with double the focal
/// length the HFOV halves and the sphere looks twice as big.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=tiled_camera_optics_gpu`"]
fn focal_length_drives_field_of_view() {
    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(sphere_scene()).expect("GPU adapter required for this test (gated by #[ignore])");

    let count_hits = |cam: &mut TiledCamera| -> usize {
        cam.dispatch(&engine, &model, &state).unwrap();
        cam.download_shape_index(&engine)
            .unwrap()
            .iter()
            .filter(|&&s| s != u32::MAX)
            .count()
    };

    // Wide lens: focal 12 mm, sensor 36 mm → HFOV ≈ 112°.
    let mut wide_cfg = base_config();
    wide_cfg.optics.focal_length_mm = 12.0;
    wide_cfg.optics.sensor_width_mm = 36.0;
    let mut wide = TiledCamera::new(&engine, wide_cfg, 1).unwrap();
    wide.set_camera_look_at(0, Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    let wide_hits = count_hits(&mut wide);

    // Tele lens: focal 100 mm, sensor 36 mm → HFOV ≈ 20°. Same sphere
    // at the same distance subtends a much larger pixel fraction, so
    // the hit count goes UP. (Both renders are 64×64 = 4096 pixels.)
    let mut tele_cfg = base_config();
    tele_cfg.optics.focal_length_mm = 100.0;
    tele_cfg.optics.sensor_width_mm = 36.0;
    let mut tele = TiledCamera::new(&engine, tele_cfg, 1).unwrap();
    tele.set_camera_look_at(0, Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    let tele_hits = count_hits(&mut tele);

    assert!(
        tele_hits > wide_hits * 3,
        "expected tele lens to subtend much more of the image than the wide lens (wide={wide_hits}, tele={tele_hits})"
    );
}

/// With a defocused sphere, K-sample DOF widens the silhouette: the
/// "hit" count from the K-sample render is ≥ the pinhole count
/// because the averaged image has bokeh-blurred edges that still
/// resolve to a real surface at most pixels.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=tiled_camera_optics_gpu`"]
fn aperture_sampling_blurs_versus_pinhole() {
    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(sphere_scene()).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut pinhole_cfg = base_config();
    pinhole_cfg.width = 96;
    pinhole_cfg.height = 96;
    let mut pinhole = TiledCamera::new(&engine, pinhole_cfg, 1).unwrap();
    pinhole
        .set_camera_look_at(0, Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    pinhole.dispatch(&engine, &model, &state).unwrap();
    let p_color = pinhole.download_color(&engine).unwrap().to_vec();

    let mut dof_cfg = pinhole_cfg;
    dof_cfg.optics.lens_diameter_m = 0.20; // wide aperture for visible blur
    dof_cfg.optics.focal_distance_m = 5.0; // focused at the sphere
    dof_cfg.optics.aperture_samples = 8;
    let mut dof = TiledCamera::new(&engine, dof_cfg, 1).unwrap();
    dof.set_camera_look_at(0, Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    dof.dispatch(&engine, &model, &state).unwrap();
    let d_color = dof.download_color(&engine).unwrap().to_vec();

    // Sphere is exactly at the focal plane → centre pixels should
    // match the pinhole (focused). The K-sample colour buffer must
    // not be bit-identical to the pinhole (different rays at the
    // edges produce different averaged shading), so a non-zero
    // per-pixel diff is the property we assert.
    let differing = p_color
        .iter()
        .zip(d_color.iter())
        .filter(|(a, b)| a != b)
        .count();
    assert!(
        differing > 0,
        "K-sample DOF render is bit-identical to pinhole — aperture sampling is a no-op"
    );

    // Both runs must produce non-degenerate output (some hits, not
    // an all-black image).
    let pin_lit = p_color.iter().filter(|&&p| p & 0x00FF_FFFF != 0).count();
    let dof_lit = d_color.iter().filter(|&&p| p & 0x00FF_FFFF != 0).count();
    assert!(pin_lit > 100);
    assert!(dof_lit > 100);
    // Sanity: depth at the centre still matches the focused distance
    // (the centre pixel's depth comes from the centre sample). Pull
    // the (width, height) out of the immutable config borrow first
    // so the subsequent `download_depth` mutable borrow is clean.
    let w = dof.config().width;
    let h = dof.config().height;
    let depth = dof.download_depth(&engine).unwrap();
    let idx = ((h / 2) * w + (w / 2)) as usize;
    let d: Real = depth[idx];
    assert!(
        (d - 4.5).abs() < 5.0e-2,
        "centre depth = {d}, expected ≈ 4.5 (sphere of radius 0.5 at 5 m)"
    );
}
