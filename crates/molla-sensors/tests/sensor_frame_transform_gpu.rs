//! GPU frame-transform sensor — golden-value tests against
//! hand-computed expectations. `#[ignore]` because they need a real
//! wgpu adapter; run with `make test-frame-transform-gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Quat, Transform, Vec3};
use molla_sensors::{FrameTarget, FrameTransformSensor};
use molla_sim::{BodyParams, Model, ModelBuilder, State};

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

/// Body at (5, 1, 0); world pose for an identity local-pose target
/// is just the body pose.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-frame-transform-gpu"]
fn world_pose_matches_body_pose() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(
        BodyParams::new().with_pose(Transform::new(Vec3::new(5.0, 1.0, 0.0), Quat::IDENTITY)),
    );
    b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut sensor = FrameTransformSensor::new(
        &engine,
        &model,
        None,
        &[FrameTarget {
            body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();
    let r = sensor.update(&engine, &state).unwrap();
    assert_eq!(r.len(), 1);
    let p = r[0].world.position;
    assert!(
        (p - Vec3::new(5.0, 1.0, 0.0)).length() < 1.0e-4,
        "world pos {:?}",
        p
    );
    // No reference → relative == world.
    assert!((r[0].relative.position - r[0].world.position).length() < 1.0e-4);
}

/// A target with a local +X offset of 2.0 on a body at the origin
/// reads world (2, 0, 0).
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-frame-transform-gpu"]
fn local_offset_is_applied_in_world() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new());
    b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut sensor = FrameTransformSensor::new(
        &engine,
        &model,
        None,
        &[FrameTarget {
            body,
            local_pose: Transform::new(Vec3::new(2.0, 0.0, 0.0), Quat::IDENTITY),
        }],
    )
    .unwrap();
    let r = sensor.update(&engine, &state).unwrap();
    let p = r[0].world.position;
    assert!(
        (p - Vec3::new(2.0, 0.0, 0.0)).length() < 1.0e-4,
        "got {:?}",
        p
    );
}

/// Two bodies, target on body B expressed relative to body A:
/// body A at (10, 0, 0), body B at (15, 0, 0); relative pose
/// of B (in A's frame) should be (5, 0, 0).
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-frame-transform-gpu"]
fn relative_pose_subtracts_reference() {
    let mut b = ModelBuilder::new();
    let a = b.add_body(BodyParams::new().with_position(Vec3::new(10.0, 0.0, 0.0)));
    b.add_shape_sphere(a, 0.1, Transform::IDENTITY).unwrap();
    let b_body = b.add_body(BodyParams::new().with_position(Vec3::new(15.0, 0.0, 0.0)));
    b.add_shape_sphere(b_body, 0.1, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut sensor = FrameTransformSensor::new(
        &engine,
        &model,
        Some(FrameTarget {
            body: a,
            local_pose: Transform::IDENTITY,
        }),
        &[FrameTarget {
            body: b_body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();
    let r = sensor.update(&engine, &state).unwrap();
    assert_eq!(r.len(), 1);
    let p_world = r[0].world.position;
    let p_rel = r[0].relative.position;
    assert!(
        (p_world - Vec3::new(15.0, 0.0, 0.0)).length() < 1.0e-4,
        "world {:?}",
        p_world
    );
    assert!(
        (p_rel - Vec3::new(5.0, 0.0, 0.0)).length() < 1.0e-4,
        "relative {:?}",
        p_rel
    );
}

/// Reference frame rotated 90° around Z: a target at world (1, 0, 0)
/// expressed in the rotated reference reads (0, -1, 0) — the +X world
/// direction is the reference's -Y.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-frame-transform-gpu"]
fn relative_pose_handles_reference_rotation() {
    let mut b = ModelBuilder::new();
    let a = b.add_body(BodyParams::new().with_pose(Transform::new(
        Vec3::ZERO,
        Quat::from_rotation_z(std::f64::consts::FRAC_PI_2),
    )));
    b.add_shape_sphere(a, 0.1, Transform::IDENTITY).unwrap();
    let b_body = b.add_body(BodyParams::new().with_position(Vec3::new(1.0, 0.0, 0.0)));
    b.add_shape_sphere(b_body, 0.1, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

    let mut sensor = FrameTransformSensor::new(
        &engine,
        &model,
        Some(FrameTarget {
            body: a,
            local_pose: Transform::IDENTITY,
        }),
        &[FrameTarget {
            body: b_body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();
    let r = sensor.update(&engine, &state).unwrap();
    let p_rel = r[0].relative.position;
    eprintln!("relative position: {:?}", p_rel);
    assert!(
        (p_rel - Vec3::new(0.0, -1.0, 0.0)).length() < 1.0e-4,
        "expected (0, -1, 0), got {:?}",
        p_rel
    );
}
