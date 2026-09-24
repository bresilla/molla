//! GPU IMU sensor — golden-value tests against hand-computed
//! expectations. `#[ignore]` because they need a real wgpu adapter;
//! run with `make test-gpu P=molla-sensors T=sensor_imu_gpu`.

#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_math::{Quat, Real, SpatialVector, Transform, Vec3};
use molla_sensors::{ImuMount, ImuSensor};
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

/// A body at rest with **zero linear velocity** in successive frames
/// has `a_lin = 0` from finite difference. The accelerometer should
/// then read `-gravity` (gravity in molla's default world is -Y, so
/// the sensor reads +9.81 along its local Y-axis with identity mount).
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu P=molla-sensors T=sensor_imu_gpu"]
fn imu_at_rest_reads_minus_gravity() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 1.0, 0.0)));
    b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

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
#[ignore = "needs a real wgpu adapter — run via make test-gpu P=molla-sensors T=sensor_imu_gpu"]
fn imu_in_free_fall_reads_zero() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 5.0, 0.0)));
    b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let (engine, model, mut state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

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
        qd[body.index()] = SpatialVector::new(Vec3::new(0.0, -9.806649 * dt, 0.0), Vec3::ZERO);
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
#[ignore = "needs a real wgpu adapter — run via make test-gpu P=molla-sensors T=sensor_imu_gpu"]
fn imu_local_rotation_swaps_axes() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 1.0, 0.0)));
    b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let (engine, model, state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

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
#[ignore = "needs a real wgpu adapter — run via make test-gpu P=molla-sensors T=sensor_imu_gpu"]
fn imu_gyro_reads_body_angular_velocity() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 1.0, 0.0)));
    b.add_shape_sphere(body, 0.1, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let (engine, model, mut state) =
        setup(model).expect("GPU adapter required for this test (gated by #[ignore])");

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

#[test]
#[ignore = "needs a real wgpu adapter"]
fn imu_reads_gpu_integrated_velocity_without_host_readback() {
    use molla_kernels::IntegrateBodiesKernel;
    use molla_sim::Control;

    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_position(Vec3::Y * 5.0));
    builder
        .add_shape_sphere(body, 0.1, Transform::IDENTITY)
        .unwrap();
    let (engine, model, mut current) = setup(builder.build().unwrap()).expect("GPU required");
    eprintln!("IMU integration adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut next = model.state().unwrap();
    next.upload_to_device(device, queue).unwrap();
    let mut control = Control::for_model(&model);
    control.upload_to_device(device, queue).unwrap();
    let dummy = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("imu integration dummy force"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });
    let integrator = IntegrateBodiesKernel::new(device);
    let mut imu = ImuSensor::new(
        &engine,
        &model,
        &[ImuMount {
            body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();
    let dt = 0.005;

    for step in 0..6 {
        integrator
            .dispatch(
                device, queue, &model, &current, &control, &dummy, &mut next, dt,
            )
            .unwrap();
        assert!(next.body_qd.host().is_err());
        if step == 3 {
            imu.reset();
        }
        let readings = imu.update(&engine, &model, &next, dt).unwrap();
        if step == 0 || step == 3 {
            assert!(readings.is_empty());
        } else {
            assert_eq!(readings.len(), 1);
            assert!(readings[0].accelerometer.length() < 1.0e-3, "{readings:?}");
            assert!(readings[0].gyroscope.length() < 1.0e-6);
        }
        assert!(next.body_qd.host().is_err());
        std::mem::swap(&mut current, &mut next);
    }
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn imu_rejects_invalid_time_and_unuploaded_velocity_without_advancing_history() {
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    builder
        .add_shape_sphere(body, 0.1, Transform::IDENTITY)
        .unwrap();
    let (engine, model, mut state) = setup(builder.build().unwrap()).expect("GPU required");
    let mut imu = ImuSensor::new(
        &engine,
        &model,
        &[ImuMount {
            body,
            local_pose: Transform::IDENTITY,
        }],
    )
    .unwrap();
    for dt in [
        0.0,
        -0.1,
        f64::NAN,
        f64::INFINITY,
        f64::MAX,
        f64::MIN_POSITIVE,
    ] {
        assert!(
            imu.update(&engine, &model, &state, dt).is_err(),
            "accepted dt={dt}"
        );
    }
    state.body_qd.host_mut().unwrap()[0].linear = Vec3::Y;
    assert!(imu.update(&engine, &model, &state, 0.01).is_err());
    state.body_qd.upload(engine.queue().unwrap()).unwrap();
    assert!(
        imu.update(&engine, &model, &state, 0.01)
            .unwrap()
            .is_empty()
    );
    let readings = imu.update(&engine, &model, &state, 0.01).unwrap();
    assert!((readings[0].accelerometer.y - 9.806649).abs() < 1.0e-3);
}

#[test]
#[ignore = "needs a real wgpu adapter"]
fn imu_batches_mount_lever_arms_and_rejects_scene_changes_until_reset() {
    let build = || {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(BodyParams::new());
        builder
            .add_shape_sphere(body, 0.1, Transform::IDENTITY)
            .unwrap();
        (builder.build().unwrap(), body)
    };
    let (model, body) = build();
    let (engine, model, mut state) = setup(model).expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mounts: Vec<_> = (0..65)
        .map(|i| ImuMount {
            body,
            local_pose: Transform::new(Vec3::X * (i as f64 / 64.0), Quat::IDENTITY),
        })
        .collect();
    let mut imu = ImuSensor::new(&engine, &model, &mounts).unwrap();
    state.body_qd.host_mut().unwrap()[0].angular = Vec3::Y;
    state.body_qd.upload(queue).unwrap();
    assert!(imu.update(&engine, &model, &state, 0.5).unwrap().is_empty());
    state.body_qd.host_mut().unwrap()[0].angular = Vec3::Y * 2.0;
    state.body_qd.upload(queue).unwrap();
    let readings = imu.update(&engine, &model, &state, 0.5).unwrap();
    assert_eq!(readings.len(), 65);
    for (i, reading) in readings.iter().enumerate() {
        let offset = i as f64 / 64.0;
        let expected = Vec3::new(-4.0 * offset, 9.806649, -2.0 * offset);
        assert!(
            (reading.accelerometer - expected).length() < 1.0e-3,
            "mount {i}: {reading:?}"
        );
        assert!((reading.gyroscope - Vec3::Y * 2.0).length() < 1.0e-4);
    }

    let (mut replacement, _) = build();
    replacement.upload_to_device(device, queue).unwrap();
    let mut replacement_state = replacement.state().unwrap();
    replacement_state.upload_to_device(device, queue).unwrap();
    assert!(
        imu.update(&engine, &replacement, &replacement_state, 0.5)
            .is_err()
    );
    imu.reset();
    assert!(
        imu.update(&engine, &replacement, &replacement_state, 0.5)
            .unwrap()
            .is_empty()
    );
    let readings = imu
        .update(&engine, &replacement, &replacement_state, 0.5)
        .unwrap();
    for reading in readings {
        assert!((reading.accelerometer - Vec3::Y * 9.806649).length() < 1.0e-3);
        assert!(reading.gyroscope.length() < 1.0e-6);
    }

    let mut builder = ModelBuilder::new();
    builder.add_body(BodyParams::new());
    builder.add_body(BodyParams::new());
    let different_size = builder.build().unwrap();
    let different_state = different_size.state().unwrap();
    assert!(
        imu.update(&engine, &different_size, &different_state, 0.5)
            .is_err()
    );
}
