#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Transform, Vec3};
use molla_sensors::{
    AerialLidar, AerialScanMetadata, CameraBatch, CameraBatchLimits, CameraChannels, CameraSpec,
    CameraStorageUsage, FrameTarget, FrameTransformSensor, ImuMount, ImuSensor, Lidar, LidarBatch,
    RayStorageUsage, Raycaster, RaycasterLimits, ScanMetadata, TiledCamera, TiledCameraConfig,
    TiledCameraLimits,
};
use molla_sim::ModelBuilder;

#[test]
#[ignore = "requires GPU"]
fn legacy_lidar_layout_caps_preserve_terrestrial_and_aerial_results() {
    use molla_sensors::LidarLayoutLimits;
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut state = model.state().unwrap();
    state.upload_to_device(device, queue).unwrap();
    let mut lidar = Lidar::new(&engine, 3).unwrap();
    let mut terrestrial = ScanMetadata::new(Vec3::Y, 1, 3);
    terrestrial.theta_range = (std::f64::consts::PI, std::f64::consts::PI);
    let bytes = terrestrial.layout_storage_bytes().unwrap();
    assert!(
        lidar
            .cast_with_layout_limits(
                &engine,
                &model,
                &state,
                &terrestrial,
                LidarLayoutLimits {
                    max_layout_bytes: bytes - 1
                }
            )
            .is_err()
    );
    let exact = lidar
        .cast_with_layout_limits(
            &engine,
            &model,
            &state,
            &terrestrial,
            LidarLayoutLimits {
                max_layout_bytes: bytes,
            },
        )
        .unwrap();
    let legacy = lidar.cast(&engine, &model, &state, &terrestrial).unwrap();
    assert_eq!(exact.hits.len(), 3);
    for (a, b) in exact.hits.iter().zip(&legacy.hits) {
        assert!(a.shape_id.is_some());
        assert_eq!(a.shape_id, b.shape_id);
        assert_eq!(a.t, b.t);
        assert_eq!(a.point, b.point);
    }
    let mut lidar = AerialLidar::new(&engine, 3).unwrap();
    let aerial = AerialScanMetadata::new(Vec3::Y, Vec3::ONE, 3.0);
    let bytes = aerial.layout_storage_bytes().unwrap();
    assert!(
        lidar
            .cast_with_layout_limits(
                &engine,
                &model,
                &state,
                &aerial,
                LidarLayoutLimits {
                    max_layout_bytes: bytes - 1
                }
            )
            .is_err()
    );
    let exact = lidar
        .cast_with_layout_limits(
            &engine,
            &model,
            &state,
            &aerial,
            LidarLayoutLimits {
                max_layout_bytes: bytes,
            },
        )
        .unwrap();
    let legacy = lidar.cast(&engine, &model, &state, &aerial).unwrap();
    assert_eq!(exact.origins, legacy.origins);
    assert_eq!(exact.directions, legacy.directions);
    assert_eq!(exact.hits.len(), 3);
    for (a, b) in exact.hits.iter().zip(&legacy.hits) {
        assert!(a.shape_id.is_some());
        assert_eq!(a.shape_id, b.shape_id);
        assert_eq!(a.t, b.t);
        assert_eq!(a.point, b.point);
    }
}

#[test]
#[ignore = "requires GPU"]
fn lidar_combined_budget_includes_returns_and_metadata_before_dispatch() {
    use molla_sensors::{
        LidarBatchLimits, LidarBatchStorage, LidarPattern, LidarScan, SensorScene,
    };
    let engine = Engine::new_gpu().unwrap();
    let usage = LidarBatchStorage::for_capacity(2, 3).unwrap();
    let exact = LidarBatchLimits {
        max_storage_bytes: usage.total_bytes,
        ..Default::default()
    };
    assert!(
        LidarBatch::new_with_limits(
            &engine,
            2,
            3,
            LidarBatchLimits {
                max_storage_bytes: usage.total_bytes - 1,
                ..exact
            }
        )
        .is_err()
    );
    let mut lidar = LidarBatch::new_with_limits(&engine, 2, 3, exact).unwrap();
    assert_eq!(lidar.storage_usage(), usage);
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = builder.build().unwrap();
    let state = model.state().unwrap();
    let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let scan = LidarScan {
        sensor_id: 17,
        pattern: LidarPattern::Terrestrial(ScanMetadata::new(Vec3::Y, 1, 3)),
    };
    assert!(
        lidar
            .dispatch_scene(&engine, &scene, &state, &[scan; 2])
            .is_err()
    );
    assert_eq!(lidar.stats().submissions, 0);
    let frame = lidar
        .dispatch_scene(&engine, &scene, &state, &[scan])
        .unwrap();
    assert_eq!(frame.hits.ray_count, 3);
    assert_eq!(frame.pulse_count, 3);
    assert_eq!(frame.scans[0].scan.sensor_id, 17);
    assert_eq!(lidar.storage_usage(), usage);
    assert_eq!(
        lidar
            .dispatch_scene(&engine, &scene, &state, &[])
            .unwrap()
            .hits
            .ray_count,
        0
    );
}

#[test]
#[ignore = "requires GPU"]
fn auxiliary_sensor_arrays_enforce_combined_budgets_before_sampling() {
    use molla_sensors::{ContactSensor, SensorArrayLimits, SensorArrayStorage};
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(Default::default());
    builder.add_body(Default::default());
    let model = builder.build().unwrap();
    let mounts = [ImuMount {
        body,
        local_pose: Transform::IDENTITY,
    }; 3];
    let usage = SensorArrayStorage::for_imu(mounts.len(), model.body_count).unwrap();
    assert!(
        ImuSensor::new_with_limits(
            &engine,
            &model,
            &mounts,
            SensorArrayLimits {
                max_storage_bytes: usage.total_bytes - 1
            }
        )
        .is_err()
    );
    let imu = ImuSensor::new_with_limits(
        &engine,
        &model,
        &mounts,
        SensorArrayLimits {
            max_storage_bytes: usage.total_bytes,
        },
    )
    .unwrap();
    assert_eq!(imu.storage_usage(), usage);
    let usage = SensorArrayStorage::for_contact(model.body_count).unwrap();
    assert!(
        ContactSensor::new_with_limits(
            &engine,
            &model,
            SensorArrayLimits {
                max_storage_bytes: usage.total_bytes - 1
            }
        )
        .is_err()
    );
    let contacts = ContactSensor::new_with_limits(
        &engine,
        &model,
        SensorArrayLimits {
            max_storage_bytes: usage.total_bytes,
        },
    )
    .unwrap();
    assert_eq!(contacts.storage_usage(), usage);
    let target = FrameTarget {
        body,
        local_pose: Transform::IDENTITY,
    };
    for reference in [None, Some(target)] {
        let usage = SensorArrayStorage::for_frames(3, reference.is_some()).unwrap();
        assert!(
            FrameTransformSensor::new_with_limits(
                &engine,
                &model,
                reference,
                &[target; 3],
                SensorArrayLimits {
                    max_storage_bytes: usage.total_bytes - 1
                }
            )
            .is_err()
        );
        let frames = FrameTransformSensor::new_with_limits(
            &engine,
            &model,
            reference,
            &[target; 3],
            SensorArrayLimits {
                max_storage_bytes: usage.total_bytes,
            },
        )
        .unwrap();
        assert_eq!(frames.storage_usage(), usage);
    }
    let empty = ModelBuilder::new().build().unwrap();
    let contact = ContactSensor::new_with_limits(
        &engine,
        &empty,
        SensorArrayLimits {
            max_storage_bytes: 0,
        },
    )
    .unwrap();
    assert_eq!(contact.storage_usage(), SensorArrayStorage::default());
}

#[test]
#[ignore = "requires GPU"]
fn camera_storage_budget_covers_all_channels_optics_and_heterogeneous_groups() {
    let engine = Engine::new_gpu().unwrap();
    let config = TiledCameraConfig {
        width: 8,
        height: 8,
        channels: CameraChannels::NONE,
        ..Default::default()
    };
    let usage = CameraStorageUsage::for_dimensions(8, 8, 1).unwrap();
    let exact = TiledCameraLimits {
        max_storage_bytes: usage.total_bytes,
        ..Default::default()
    };
    assert!(
        TiledCamera::new_with_limits(
            &engine,
            config,
            1,
            TiledCameraLimits {
                max_storage_bytes: usage.total_bytes - 1,
                ..exact
            }
        )
        .is_err()
    );
    let mut camera = TiledCamera::new_with_limits(&engine, config, 1, exact).unwrap();
    let mut active = config;
    active.channels = CameraChannels::ALL;
    active.optics.lens_diameter_m = 0.01;
    active.optics.aperture_samples = 4;
    camera.set_config(active).unwrap();
    assert_eq!(camera.storage_usage(), usage);
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = builder.build().unwrap();
    let state = model.state().unwrap();
    let scene =
        molla_sensors::SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    camera
        .set_camera_look_at(0, Vec3::new(0.0, 2.0, 2.0), Vec3::ZERO, Vec3::Y)
        .unwrap();
    camera.dispatch_scene(&engine, &scene, &state).unwrap();
    assert!(
        camera
            .download_depth(&engine)
            .unwrap()
            .iter()
            .any(|&d| d > 0.0)
    );
    assert_eq!(camera.dispatch_stats().accumulator_buffer_allocations, 2);
    let specs = [
        CameraSpec {
            sensor_id: 1,
            config,
            pose: Transform::IDENTITY,
        },
        CameraSpec {
            sensor_id: 2,
            config: TiledCameraConfig {
                fov_y_radians: 0.8,
                ..config
            },
            pose: Transform::IDENTITY,
        },
    ];
    assert!(
        CameraBatch::new_with_storage_limits(&engine, &specs, CameraBatchLimits::default(), exact)
            .is_err()
    );
    let aggregate = TiledCameraLimits {
        max_storage_bytes: 2 * usage.total_bytes,
        ..exact
    };
    let mut batch =
        CameraBatch::new_with_storage_limits(&engine, &specs, Default::default(), aggregate)
            .unwrap();
    assert_eq!(batch.storage_bytes(), 2 * usage.total_bytes);
    let old = batch.layouts().to_vec();
    let larger = [
        specs[0],
        specs[1],
        CameraSpec {
            sensor_id: 3,
            config: TiledCameraConfig {
                fov_y_radians: 0.7,
                ..config
            },
            ..specs[0]
        },
    ];
    assert!(batch.reconfigure(&engine, &larger).is_err());
    assert_eq!(batch.layouts(), old);
    assert_eq!(batch.storage_bytes(), 2 * usage.total_bytes);
    batch.reconfigure(&engine, &specs[..1]).unwrap();
    assert_eq!(batch.storage_bytes(), usage.total_bytes);
    batch.reconfigure(&engine, &[]).unwrap();
    assert_eq!(batch.storage_bytes(), 0);
    assert!(batch.reconfigure(&engine, &larger).is_err());
}

#[test]
#[ignore = "requires GPU"]
fn imu_and_frame_mounts_reject_nonfinite_and_nonrigid_poses() {
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(Default::default());
    let model = builder.build().unwrap();
    let poses = [
        Transform::from_translation(Vec3::splat(f64::NAN)),
        Transform::from_translation(Vec3::splat(f64::INFINITY)),
        Transform::from_translation(Vec3::splat(f64::MAX)),
        Transform::new(Vec3::ZERO, molla_math::Quat::from_xyzw(0.0, 0.0, 0.0, 0.0)),
        Transform::new(Vec3::ZERO, molla_math::Quat::from_xyzw(0.0, 0.0, 0.0, 2.0)),
        Transform::new(
            Vec3::ZERO,
            molla_math::Quat::from_xyzw(f64::NAN, 0.0, 0.0, 1.0),
        ),
    ];
    for local_pose in poses {
        assert!(ImuSensor::new(&engine, &model, &[ImuMount { body, local_pose }]).is_err());
        let target = FrameTarget { body, local_pose };
        assert!(FrameTransformSensor::new(&engine, &model, None, &[target]).is_err());
        assert!(
            FrameTransformSensor::new(
                &engine,
                &model,
                Some(target),
                &[FrameTarget {
                    body,
                    local_pose: Transform::IDENTITY
                }]
            )
            .is_err()
        );
    }
    let local_pose = Transform::new(
        Vec3::new(1.0, 2.0, 3.0),
        molla_math::Quat::from_rotation_y(0.7),
    );
    assert!(ImuSensor::new(&engine, &model, &[ImuMount { body, local_pose }]).is_ok());
    assert!(
        FrameTransformSensor::new(&engine, &model, None, &[FrameTarget { body, local_pose }])
            .is_ok()
    );
}

#[test]
#[ignore = "requires GPU"]
fn explicit_ray_array_budgets_reject_all_shared_producers_before_rendering() {
    let engine = Engine::new_gpu().unwrap();
    let usage = RayStorageUsage::for_capacity(1).unwrap();
    let too_small = RaycasterLimits {
        max_ray_storage_bytes: usage.total_bytes - 1,
    };
    let exact = RaycasterLimits {
        max_ray_storage_bytes: usage.total_bytes,
    };
    let camera = TiledCameraConfig {
        width: 1,
        height: 1,
        ..Default::default()
    };
    assert!(Raycaster::new_with_limits(&engine, 1, too_small).is_err());
    assert!(Lidar::new_with_limits(&engine, 1, too_small).is_err());
    assert!(AerialLidar::new_with_limits(&engine, 1, too_small).is_err());
    assert!(LidarBatch::new_with_ray_limits(&engine, 1, 1, too_small).is_err());
    assert!(TiledCamera::new_with_ray_limits(&engine, camera, 1, too_small).is_err());
    assert!(Raycaster::new(&engine, u32::MAX).is_err());
    assert!(Lidar::new(&engine, u32::MAX).is_err());
    assert!(AerialLidar::new(&engine, u32::MAX).is_err());
    assert!(Raycaster::new_with_limits(&engine, 0, exact).is_ok());
    assert_eq!(
        Lidar::new_with_limits(&engine, 0, exact)
            .unwrap()
            .max_rays(),
        1
    );
    assert_eq!(
        AerialLidar::new_with_limits(&engine, 0, exact)
            .unwrap()
            .max_rays(),
        1
    );
    assert!(LidarBatch::new_with_ray_limits(&engine, 1, 1, exact).is_ok());
    assert!(TiledCamera::new_with_ray_limits(&engine, camera, 1, exact).is_ok());

    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let state = model.state().unwrap();
    let mut ray = Raycaster::new_with_limits(&engine, 1, exact).unwrap();
    let hits = ray
        .cast(&engine, &model, &state, &[Vec3::Y * 3.0], &[-Vec3::Y], 10.0)
        .unwrap();
    assert_eq!(hits[0].shape_id, Some(0));
    assert!((hits[0].t - 3.0).abs() < 1e-6);
    let mut lidar = Lidar::new_with_limits(&engine, 1, exact).unwrap();
    let mut recipe = ScanMetadata::new(Vec3::Y * 3.0, 1, 1);
    recipe.up = -Vec3::Y;
    let scan = lidar.cast(&engine, &model, &state, &recipe).unwrap();
    assert_eq!(scan.hits[0].shape_id, Some(0));
    assert!((scan.hits[0].t - 3.0).abs() < 1e-6);
    let mut aerial = AerialLidar::new_with_limits(&engine, 1, exact).unwrap();
    let recipe = AerialScanMetadata::new(Vec3::new(-0.5, 3.0, -0.5), Vec3::new(0.5, 3.0, 0.5), 1.0);
    let scan = aerial.cast(&engine, &model, &state, &recipe).unwrap();
    assert_eq!(scan.hits[0].shape_id, Some(0));
    assert!((scan.hits[0].t - 3.0).abs() < 1e-6);
}
