#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Transform, Vec3};
use molla_sensors::{
    CloudFilter, CloudInput, CloudJob, CloudPoint, CloudProcessor, CloudProcessorConfig,
    CloudReadLimits, CloudSummary, LidarBatch, LidarPattern, LidarScan, ScanMetadata, SensorScene,
    SensorSceneOptions,
    cloud_export::{read_cloud_ply, write_cloud_ply, write_cloud_xyz_rgb},
    readback::{DeviceBuffer, ReadbackConfig, ReadbackRing, ReadbackStamp},
};
use molla_sim::ModelBuilder;
use molla_storage::BufferUsage;
use std::{io::Cursor, sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());

fn ring(engine: &Engine, bytes: u64) -> ReadbackRing {
    ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: bytes,
            max_total_bytes: bytes,
        },
    )
    .unwrap()
}

fn roundtrip(points: &[CloudPoint]) {
    let mut file = Vec::new();
    write_cloud_ply(points, &mut file).unwrap();
    let limits = CloudReadLimits {
        max_points: points.len(),
        max_file_bytes: file.len() as u64,
    };
    let decoded = read_cloud_ply(Cursor::new(file), limits).unwrap();
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&decoded),
        bytemuck::cast_slice::<_, u8>(points)
    );
    let mut xyz = Vec::new();
    write_cloud_xyz_rgb(points, &mut xyz).unwrap();
    let decoded = molla_sensors::pointcloud::read_xyz_rgb(Cursor::new(xyz)).unwrap();
    assert_eq!(decoded.len(), points.len());
    for ((position, rgb), point) in decoded.into_iter().zip(points) {
        let expected = Vec3::new(
            point.position[0] as f64,
            point.position[1] as f64,
            point.position[2] as f64,
        );
        assert!(position.distance(expected) < 1e-6);
        let color: [f32; 3] = if point.extra[2] & 2 == 0 {
            [1.; 3]
        } else {
            std::array::from_fn(|i| ((point.extra[1] >> (i * 8)) & 255) as f32 / 255.)
        };
        for (actual, expected) in rgb.into_iter().zip(color) {
            assert!((actual - expected).abs() < 0.000051);
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn selected_compacted_job_exports_without_collecting_other_jobs_or_tail() {
    let _lock = GPU.lock().unwrap();
    let engine = Engine::new_gpu().unwrap();
    let points: Vec<_> = (0..259)
        .map(|i| {
            let mut p = CloudPoint::new(
                Vec3::new(i as f64, -0.25, 1.5),
                u64::MAX - i as u64,
                0xfedcba9876543210,
            );
            p.position[3] = if i == 257 { 1. } else { 0. };
            p.normal = [0., 1., 0., 12.5];
            p.direction = [0., -1., 0., 42.25];
            p.grid = [3, 4, 2, i];
            p.extra = [77, 0x80604020, 15, 42];
            p.angles = [1.25, -0.25, 3., 4.];
            p
        })
        .collect();
    let input =
        DeviceBuffer::from_host(&engine, points.clone(), BufferUsage::StorageReadWrite).unwrap();
    let stamp = ReadbackStamp {
        sensor_id: u64::MAX - 257,
        sample_id: 0xfedcba9876543210,
        scene_revision: 91,
        simulation_time: Some(12.5),
    };
    let jobs = [
        CloudJob {
            input: CloudInput::Points(input.readback_source(0..256).unwrap()),
            transform: Transform::IDENTITY,
            filter: Default::default(),
            stamp: Default::default(),
        },
        CloudJob {
            input: CloudInput::Points(input.readback_source(256..259).unwrap()),
            transform: Transform::IDENTITY,
            filter: CloudFilter {
                min_intensity: Some(0.5),
                ..Default::default()
            },
            stamp,
        },
    ];
    let mut processor = CloudProcessor::new(
        &engine,
        CloudProcessorConfig {
            max_points: 259,
            max_jobs: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let frame = processor.process(&jobs).unwrap();
    let mut ring = ring(&engine, 112);
    let ticket = ring
        .try_enqueue(&[frame.summary_source(1).unwrap()], stamp)
        .unwrap()
        .unwrap();
    let summary_read = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    let summary = CloudSummary::from_words(&summary_read.channel::<u32>(0).unwrap()).unwrap();
    assert_eq!(
        summary,
        CloudSummary {
            start: 256,
            count: 1,
            misses: 0,
            invalid: 0,
            filtered: 2
        }
    );
    let ticket = ring
        .try_enqueue(
            &[frame
                .points_source(summary.start as usize..(summary.start + summary.count) as usize)
                .unwrap()],
            stamp,
        )
        .unwrap()
        .unwrap();
    let read = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(read.stamp, stamp);
    let selected = read.channel::<CloudPoint>(0).unwrap();
    assert_eq!(selected, points[257..258]);
    assert!(read.channel::<u32>(1).is_err());
    roundtrip(&selected);
    let mut file = Vec::new();
    write_cloud_ply(&selected, &mut file).unwrap();
    let restored = read_cloud_ply(Cursor::new(file), CloudReadLimits::default()).unwrap();
    let uploaded =
        DeviceBuffer::from_host(&engine, restored, BufferUsage::StorageReadWrite).unwrap();
    let frame = processor
        .process(&[CloudJob {
            input: CloudInput::Points(uploaded.readback_source(0..1).unwrap()),
            transform: Transform::IDENTITY,
            filter: Default::default(),
            stamp,
        }])
        .unwrap();
    let ticket = ring
        .try_enqueue(&[frame.points_source(0..1).unwrap()], stamp)
        .unwrap()
        .unwrap();
    assert_eq!(
        ring.wait(&ticket, Duration::from_secs(10))
            .unwrap()
            .channel::<CloudPoint>(0)
            .unwrap(),
        selected
    );
}

#[test]
#[ignore = "requires GPU"]
fn lidar_device_cloud_exports_empty_and_hit_scans_with_full_metadata() {
    let _lock = GPU.lock().unwrap();
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = builder.build().unwrap();
    let state = model.state().unwrap();
    let scene = SensorScene::from_model(&engine, &model, SensorSceneOptions::default()).unwrap();
    let mut up = ScanMetadata::new(Vec3::new(0., 2., 0.), 3, 5);
    up.theta_range = (0.1, 0.5);
    up.phi_range = (-1., 1.);
    up.subray_count = 2;
    up.beam_divergence = 0.01;
    let mut down = up;
    down.theta_range = (2., 2.8);
    let scans = [
        LidarScan {
            sensor_id: 7,
            pattern: LidarPattern::Terrestrial(up),
        },
        LidarScan {
            sensor_id: u64::MAX,
            pattern: LidarPattern::Terrestrial(down),
        },
    ];
    let mut lidar = LidarBatch::new(&engine, 2, 60).unwrap();
    let lidar_frame = lidar
        .dispatch_scene(&engine, &scene, &state, &scans)
        .unwrap();
    let jobs: Vec<_> = scans
        .iter()
        .enumerate()
        .map(|(i, scan)| CloudJob {
            input: CloudInput::from_lidar(&lidar_frame, i).unwrap(),
            transform: Transform::IDENTITY,
            filter: Default::default(),
            stamp: ReadbackStamp {
                sensor_id: scan.sensor_id,
                sample_id: 0x123456789abcdef0,
                scene_revision: scene.stamp().geometry_revision,
                simulation_time: Some(0.25),
            },
        })
        .collect();
    let mut processor = CloudProcessor::new(
        &engine,
        CloudProcessorConfig {
            max_points: 60,
            max_jobs: 2,
            ..Default::default()
        },
    )
    .unwrap();
    let frame = processor.process(&jobs).unwrap();
    let mut ring = ring(&engine, 30 * 112);
    for (i, job) in jobs.iter().enumerate() {
        let ticket = ring
            .try_enqueue(&[frame.summary_source(i).unwrap()], job.stamp)
            .unwrap()
            .unwrap();
        let read = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
        let summary = CloudSummary::from_words(&read.channel::<u32>(0).unwrap()).unwrap();
        assert_eq!(summary.count, if i == 0 { 0 } else { 30 });
        assert_eq!(summary.misses, if i == 0 { 30 } else { 0 });
        let ticket = ring
            .try_enqueue(
                &[frame
                    .points_source(summary.start as usize..(summary.start + summary.count) as usize)
                    .unwrap()],
                job.stamp,
            )
            .unwrap()
            .unwrap();
        let read = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
        let points = read.channel::<CloudPoint>(0).unwrap();
        roundtrip(&points);
        for (j, p) in points.iter().enumerate() {
            assert_eq!(p.sensor_id(), u64::MAX);
            assert_eq!(p.sample_id(), job.stamp.sample_id);
            assert_eq!(
                p.grid,
                [
                    (j / 10) as u32,
                    (j / 2 % 5) as u32,
                    (j % 2) as u32,
                    30 + j as u32
                ]
            );
            assert_eq!(p.extra, [0, u32::MAX, 9, 0]);
            assert!(p.position[1].abs() < 1e-6);
            assert!(p.normal[3] > 0.);
            assert!((2.0..=2.8).contains(&p.angles[0]));
        }
    }
    assert_eq!(processor.stats().submissions, 1);
}
