//! Stage 14 — multi-channel raytraced camera sensor (GPU).
//!
//! All `#[ignore]`-marked because they need a real wgpu adapter; run
//! via `make test-tiled-camera-gpu` (or the workspace-wide
//! `make test-gpu`).

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::math::{Quat, Real, Transform, Vec3};
use molla::prelude::*;
use molla::sensors::{CameraChannels, TiledCamera, TiledCameraConfig};

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

/// Tiny config: 17×17 image (odd so the centre pixel sits exactly
/// on the optical axis), 4 cameras max — enough resolution to verify
/// hits without dragging GPU latency.
fn tiny_config() -> TiledCameraConfig {
    TiledCameraConfig {
        width: 17,
        height: 17,
        fov_y_radians: 60f32.to_radians(),
        channels: CameraChannels::ALL,
        ambient: Vec3::new(0.1, 0.1, 0.1),
        light_dir: Vec3::new(0.0, -1.0, 0.0).normalize(),
        light_color: Vec3::new(0.8, 0.8, 0.8),
        max_distance: 100.0,
        near: 0.05,
    }
}

/// Camera at +Z looking down -Z at a sphere at the origin. Centre
/// pixel should hit the sphere at depth ≈ (eye_z - radius).
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-tiled-camera-gpu"]
fn camera_hits_sphere_centre_returns_correct_depth() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let (engine, model, state) = setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut cam = TiledCamera::new(&engine, tiny_config(), 1).unwrap();
    // Eye at (0, 0, 5), looking at the origin (so toward -Z, with +Y up).
    cam.set_camera_look_at(0, Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    cam.dispatch(&engine, &model, &state).unwrap();

    let cx = cam.config().width / 2;
    let cy = cam.config().height / 2;
    let centre_idx = cam.pixel_index(0, cx, cy);
    let corner_idx = cam.pixel_index(0, 0, 0);
    let depth = cam.download_depth(&engine).unwrap().to_vec();
    let centre = depth[centre_idx];
    let expected: Real = 4.5;
    assert!(
        centre > 0.0 && (centre - expected).abs() < 5.0e-2,
        "centre depth {centre} expected ≈ {expected}"
    );

    // A corner pixel should miss (sphere subtends a small angle from 5 m away).
    let corner = depth[corner_idx];
    assert!(corner < 0.0, "corner pixel should miss, got depth {corner}");
}

/// Two spheres side-by-side, one camera centred between them. Left
/// half of the image should report the left sphere's index; right
/// half the right sphere's index.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-tiled-camera-gpu"]
fn shape_index_distinguishes_two_spheres() {
    let mut b = ModelBuilder::new();
    let left_body = b.add_body(BodyParams::new().with_position(Vec3::new(-1.0, 0.0, 0.0)));
    let _left = b.add_shape_sphere(left_body, 0.6, Transform::IDENTITY).unwrap();
    let right_body = b.add_body(BodyParams::new().with_position(Vec3::new(1.0, 0.0, 0.0)));
    let _right = b.add_shape_sphere(right_body, 0.6, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let (engine, model, state) = setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut cam = TiledCamera::new(&engine, tiny_config(), 1).unwrap();
    // Eye on +Z looking at origin; the spheres span the horizontal FOV.
    cam.set_camera_look_at(0, Vec3::new(0.0, 0.0, 4.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    cam.dispatch(&engine, &model, &state).unwrap();
    let shape_idx = cam.download_shape_index(&engine).unwrap().to_vec();

    let w = cam.config().width;
    let h = cam.config().height;
    let cy = h / 2;
    // Use the quarter / three-quarter pixel columns: those land
    // squarely on the left and right spheres respectively (centre
    // ~±1 m at 4 m distance ≈ ±14° from optical axis; FOV is 60°).
    let left_x = w / 4;
    let right_x = w - 1 - (w / 4);
    let left_pixel = shape_idx[(cy * w + left_x) as usize];
    let right_pixel = shape_idx[(cy * w + right_x) as usize];
    assert_ne!(
        left_pixel, right_pixel,
        "left col {left_x} ({left_pixel:#x}) and right col {right_x} ({right_pixel:#x}) should hit different shapes"
    );
    // Both halves should hit *something* (no miss sentinel).
    assert_ne!(left_pixel, 0xffff_ffff, "left column missed");
    assert_ne!(right_pixel, 0xffff_ffff, "right column missed");
    // Both should be valid shape ids in {0, 1}.
    assert!(left_pixel < 2, "left_pixel = {left_pixel}");
    assert!(right_pixel < 2, "right_pixel = {right_pixel}");
}

/// Sphere coloured red via `set_shape_visual_material`. The albedo
/// channel at the centre pixel should pack red as `(255, 0, 0, 255)`.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-tiled-camera-gpu"]
fn albedo_matches_set_color() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    let sphere = b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    b.set_shape_visual_material(sphere, Vec3::new(1.0, 0.0, 0.0), 0.0, 0.5)
        .unwrap();
    let model = b.build().unwrap();
    let (engine, model, state) = setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut cam = TiledCamera::new(&engine, tiny_config(), 1).unwrap();
    cam.set_camera_look_at(0, Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    cam.dispatch(&engine, &model, &state).unwrap();
    let cx = cam.config().width / 2;
    let cy = cam.config().height / 2;
    let centre_idx = cam.pixel_index(0, cx, cy);
    let albedo = cam.download_albedo(&engine).unwrap().to_vec();
    let centre = albedo[centre_idx];
    // RGBA8: r in low byte, g next, b next, a in high byte.
    let r = centre & 0xff;
    let g = (centre >> 8) & 0xff;
    let b_ch = (centre >> 16) & 0xff;
    let a = (centre >> 24) & 0xff;
    assert_eq!(r, 255, "red channel = {r}, packed = {centre:#x}");
    assert_eq!(g, 0, "green channel = {g}");
    assert_eq!(b_ch, 0, "blue channel = {b_ch}");
    assert_eq!(a, 255, "alpha channel = {a}");
}

/// 4 cameras in one batch, each looking at the origin from a distinct
/// world-space direction. Each camera's centre pixel should hit the
/// shared sphere at the same depth.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-tiled-camera-gpu"]
fn multi_camera_dispatches_in_one_call() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let (engine, model, state) = setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut cam = TiledCamera::new(&engine, tiny_config(), 4).unwrap();
    let r: Real = 5.0;
    cam.set_camera_look_at(0, Vec3::new(r, 0.0, 0.0), Vec3::ZERO, Vec3::Y).unwrap();
    cam.set_camera_look_at(1, Vec3::new(-r, 0.0, 0.0), Vec3::ZERO, Vec3::Y).unwrap();
    cam.set_camera_look_at(2, Vec3::new(0.0, 0.0, r), Vec3::ZERO, Vec3::Y).unwrap();
    cam.set_camera_look_at(3, Vec3::new(0.0, 0.0, -r), Vec3::ZERO, Vec3::Y).unwrap();
    cam.dispatch(&engine, &model, &state).unwrap();

    let depth = cam.download_depth(&engine).unwrap().to_vec();
    let cx = cam.config().width / 2;
    let cy = cam.config().height / 2;
    let expected: Real = 4.5;
    for c in 0..4u32 {
        let d = depth[cam.pixel_index(c, cx, cy)];
        assert!(
            d > 0.0 && (d - expected).abs() < 5.0e-2,
            "camera {c} centre depth {d}, expected ≈ {expected}"
        );
    }

    // Each camera's tile should be independent — clobber camera 0's
    // pose and re-dispatch; cameras 1..4 still hit at expected depth.
    cam.set_camera_pose(
        0,
        Transform::new(Vec3::new(0.0, 100.0, 0.0), Quat::IDENTITY),
    )
    .unwrap();
    cam.dispatch(&engine, &model, &state).unwrap();
    let depth2 = cam.download_depth(&engine).unwrap().to_vec();
    let centre0 = depth2[cam.pixel_index(0, cx, cy)];
    assert!(
        centre0 < 0.0 || centre0 > 50.0,
        "camera 0 should miss (or be far) after teleport, got {centre0}"
    );
    for c in 1..4u32 {
        let d = depth2[cam.pixel_index(c, cx, cy)];
        assert!(
            d > 0.0 && (d - expected).abs() < 5.0e-2,
            "camera {c} (post-teleport of cam 0) depth {d}"
        );
    }
}
