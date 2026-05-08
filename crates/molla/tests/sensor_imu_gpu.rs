//! GPU IMU sensor — golden-value tests against hand-computed
//! expectations. `#[ignore]` because they need a real wgpu adapter;
//! run with `make test-imu-gpu`.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::math::{Quat, Real, SpatialVector, Transform, Vec3};
use molla::prelude::*;
use molla::sensors::{ImuMount, ImuSensor};

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

/// A body at rest with **zero linear velocity** in successive frames
/// has `a_lin = 0` from finite difference. The accelerometer should
/// then read `-gravity` (gravity in molla's default world is -Y, so
/// the sensor reads +9.81 along its local Y-axis with identity mount).
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-imu-gpu"]
fn imu_at_rest_reads_minus_gravity() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 1.0, 0.0)));
    b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let Some((engine, model, mut state)) = setup(model) else {
        return;
    };

    let mut imu = ImuSensor::new(
        &engine,
        &model,
        &[ImuMount {
            body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();

    // Prime the finite-difference: first update returns empty.
    let first = imu.update(&engine, &model, &state, 0.005).unwrap();
    assert!(first.is_empty(), "first update should be empty (priming)");

    // Second update: body_qd is still zero (we never stepped), so
    // a_lin_com = 0, and accelerometer = -gravity = (0, 9.81, 0).
    let r = imu.update(&engine, &model, &state, 0.005).unwrap();
    assert_eq!(r.len(), 1);
    let a = r[0].accelerometer;
    eprintln!("at-rest accel: {a:?}");
    assert!(
        (a.y - 9.806649).abs() < 1.0e-3,
        "expected accel.y ≈ 9.81, got {}",
        a.y
    );
    assert!(a.x.abs() < 1.0e-4 && a.z.abs() < 1.0e-4);

    // Gyro is just ω in sensor frame; ω = 0 at rest.
    let g = r[0].gyroscope;
    assert!(g.length() < 1.0e-6, "gyro should be zero, got {:?}", g);
    let _ = state; // keep in scope
}

/// A body in **free fall** has `body_qd.linear` falling at g per
/// second; the finite-difference linear accel approaches `(0, -g, 0)`.
/// Specific force = a - g = 0, so the accelerometer reads zero.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-imu-gpu"]
fn imu_in_free_fall_reads_zero() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 5.0, 0.0)));
    b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let Some((engine, model, mut state)) = setup(model) else {
        return;
    };

    let mut imu = ImuSensor::new(
        &engine,
        &model,
        &[ImuMount {
            body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();

    // Prime.
    let _ = imu.update(&engine, &model, &state, 0.005).unwrap();

    // Manually advance state.body_qd by a free-fall step, with
    // dt = 0.005, g = -9.81: Δv.y = -9.81 · 0.005 = -0.04905.
    let dt: Real = 0.005;
    {
        let qd = state.body_qd.host_mut().unwrap();
        qd[body.index()] = SpatialVector::new(
            Vec3::new(0.0, -9.806649 * dt, 0.0),
            Vec3::ZERO,
        );
        let device = engine.device().unwrap().clone();
        let queue = engine.queue().unwrap().clone();
        state.body_qd.upload(&queue).unwrap();
        // Note: positions don't matter for accel calc with identity
        // local pose at COM, but keep them coherent anyway.
        let _ = device;
    }

    let r = imu.update(&engine, &model, &state, dt).unwrap();
    assert_eq!(r.len(), 1);
    let a = r[0].accelerometer;
    eprintln!("free-fall accel: {a:?}");
    // Specific force should be ~0 — finite difference of (Δv/Δt) gives
    // -g, then we subtract gravity to get specific force = 0.
    assert!(
        a.length() < 1.0e-3,
        "expected ≈ zero specific force in free fall, got {:?}",
        a
    );
}

/// IMU mounted with a 90° rotation around Z reads the same magnitudes
/// but on swapped axes — verifies the world→sensor frame transform.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-imu-gpu"]
fn imu_local_rotation_swaps_axes() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 1.0, 0.0)));
    b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let Some((engine, model, state)) = setup(model) else {
        return;
    };

    // Sensor rotated 90° around Z — its local +Y axis points along
    // world -X, so gravity (along world -Y) projects onto sensor +X.
    let q = Quat::from_rotation_z(std::f64::consts::FRAC_PI_2);
    let mut imu = ImuSensor::new(
        &engine,
        &model,
        &[ImuMount {
            body,
            local_pose: Transform::new(Vec3::ZERO, q),
        }],
    )
    .unwrap();
    let _ = imu.update(&engine, &model, &state, 0.005).unwrap(); // prime
    let r = imu.update(&engine, &model, &state, 0.005).unwrap();
    let a = r[0].accelerometer;
    eprintln!("rotated-mount accel: {a:?}");
    // World specific force is (0, 9.81, 0). Inverse rotation
    // (-90° around Z) maps +Y → +X.
    assert!(
        (a.x - 9.806649).abs() < 1.0e-3,
        "expected accel.x ≈ 9.81, got {}",
        a.x
    );
    assert!(a.y.abs() < 1.0e-3 && a.z.abs() < 1.0e-3);
}

/// IMU on a body with non-zero angular velocity reports it in the
/// sensor frame. A body spinning at ω_world = (0, 1, 0) rad/s with
/// identity sensor mount should read gyro = (0, 1, 0).
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-imu-gpu"]
fn imu_gyro_reads_body_angular_velocity() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 1.0, 0.0)));
    b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let Some((engine, model, mut state)) = setup(model) else {
        return;
    };

    {
        let qd = state.body_qd.host_mut().unwrap();
        qd[body.index()] = SpatialVector::new(Vec3::ZERO, Vec3::new(0.0, 1.0, 0.0));
        let queue = engine.queue().unwrap().clone();
        state.body_qd.upload(&queue).unwrap();
    }

    let mut imu = ImuSensor::new(
        &engine,
        &model,
        &[ImuMount {
            body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();
    let _ = imu.update(&engine, &model, &state, 0.005).unwrap(); // prime
    let r = imu.update(&engine, &model, &state, 0.005).unwrap();
    let g = r[0].gyroscope;
    eprintln!("spinning gyro: {g:?}");
    assert!(
        (g.y - 1.0).abs() < 1.0e-3,
        "expected gyro.y ≈ 1.0, got {}",
        g.y
    );
    assert!(g.x.abs() < 1.0e-3 && g.z.abs() < 1.0e-3);
}
