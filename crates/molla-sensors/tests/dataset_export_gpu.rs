//! H9 — Synthetic-annotation export integration test (GPU).
//!
//! Renders a small scene through [`TiledCamera`], labels one shape,
//! and verifies that the [`Dataset`] post-process produces the right
//! per-instance bounding box and a non-trivial PGM class image. Run
//! via `make test-gpu P=molla-sensors T=dataset_export_gpu`.

#![cfg(feature = "gpu")]

use std::sync::Mutex;

use molla_compute::Engine;
use molla_math::{Real, Transform, Vec3};
use molla_sensors::{BoundingBox2D, CameraChannels, Dataset, TiledCamera, TiledCameraConfig};
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

/// 32×32 image so the projected sphere is several pixels wide.
fn small_camera_config() -> TiledCameraConfig {
    TiledCameraConfig {
        width: 32,
        height: 32,
        fov_y_radians: 60f32.to_radians(),
        channels: CameraChannels::ALL,
        ambient: Vec3::new(0.1, 0.1, 0.1),
        light_dir: Vec3::new(0.0, -1.0, 0.0).normalize(),
        light_color: Vec3::new(0.8, 0.8, 0.8),
        max_distance: 100.0,
        near: 0.05,
        optics: Default::default(),
    }
}

/// A labelled sphere returns exactly one bounding box centred on the
/// image, with the expected pixel-AABB extent within a 1-pixel slop.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=dataset_export_gpu`"]
fn labelled_sphere_yields_centred_bbox() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    let sphere_shape = b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let cfg = small_camera_config();
    let mut cam = TiledCamera::new(&engine, cfg, 1).unwrap();
    cam.set_camera_look_at(0, Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    cam.dispatch(&engine, &model, &state).unwrap();
    let shape_image = cam.download_shape_index(&engine).unwrap().to_vec();

    // Label the sphere as class 1 (= "target").
    let mut ds = Dataset::new();
    ds.label_shapes(&[sphere_shape.0], 1);

    let width = cam.config().width;
    let height = cam.config().height;
    let frame = ds.annotate(0, &shape_image[..(width * height) as usize], width, height);

    // Exactly one labelled instance → exactly one bbox.
    assert_eq!(
        frame.bboxes.len(),
        1,
        "expected one bbox, got {:#?}",
        frame.bboxes
    );
    let b: BoundingBox2D = frame.bboxes[0];
    assert_eq!(b.class_id, 1);
    assert_eq!(b.shape_id, sphere_shape.0);

    // The sphere projects to a roughly centred disc. Verify the
    // bbox centre is within 2 pixels of the image centre.
    let cx_target = width / 2;
    let cy_target = height / 2;
    let bcx = (b.x_min + b.x_max) / 2;
    let bcy = (b.y_min + b.y_max) / 2;
    let dx = (bcx as i32 - cx_target as i32).abs();
    let dy = (bcy as i32 - cy_target as i32).abs();
    assert!(
        dx <= 2,
        "bbox centre {bcx} too far from image centre {cx_target} (dx={dx})"
    );
    assert!(
        dy <= 2,
        "bbox centre {bcy} too far from image centre {cy_target} (dy={dy})"
    );

    // The projected sphere subtends an angular diameter
    // `2·atan(r / d) = 2·atan(0.5/5) ≈ 11.4°`. At `fov_y = 60°` and
    // height 32, the sphere covers ≈ 32 · 11.4 / 60 ≈ 6 pixels.
    assert!(
        b.width() >= 4 && b.width() <= 10,
        "bbox width {} outside expected [4, 10]",
        b.width()
    );
    assert!(
        b.height() >= 4 && b.height() <= 10,
        "bbox height {} outside expected [4, 10]",
        b.height()
    );

    // JSON output must contain the shape id + class id strings.
    let mut json = Vec::new();
    frame.write_bbox_json(&mut json).unwrap();
    let s = std::str::from_utf8(&json).unwrap();
    assert!(s.contains("\"shape_id\""));
    assert!(s.contains("\"class_id\": 1"));
    assert!(s.contains("\"bbox\":"));

    // PGM class-image header is well-formed.
    let mut pgm = Vec::new();
    frame.write_class_image_pgm(&mut pgm).unwrap();
    assert!(pgm.starts_with(b"P2\n"));

    // Compute pixel-IOU against the bbox's own pixel-count footprint
    // — the labelled-pixel count should match the bbox `pixel_count`.
    let labelled_pixels: u32 = frame.class_image.iter().filter(|&&c| c == 1).count() as u32;
    assert_eq!(
        labelled_pixels, b.pixel_count,
        "class-image pixel count ≠ bbox pixel_count"
    );

    // The full state.body_q + max_distance keep the test silent on
    // floating-point noise; `_` to mark them as intentionally unused.
    let _ = (state, model, engine, Real::default());
}

/// A scene with no labelled shapes returns an empty bbox list, but a
/// PGM class image of the right shape.
#[test]
#[ignore = "needs a real wgpu adapter — run via `make test-gpu P=molla-sensors T=dataset_export_gpu`"]
fn unlabelled_scene_returns_empty_bbox_list() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let _gpu = lock_gpu();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let cfg = small_camera_config();
    let mut cam = TiledCamera::new(&engine, cfg, 1).unwrap();
    cam.set_camera_look_at(0, Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    cam.dispatch(&engine, &model, &state).unwrap();
    let shape_image = cam.download_shape_index(&engine).unwrap().to_vec();

    let ds = Dataset::new(); // no labels
    let frame = ds.annotate(
        0,
        &shape_image[..(cfg.width * cfg.height) as usize],
        cfg.width,
        cfg.height,
    );
    assert!(frame.bboxes.is_empty());
    assert_eq!(frame.class_image.len() as u32, cfg.width * cfg.height);
}
