#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Quat, Real, Transform, Vec3};
use molla_sensors::{
    CameraBatch, CameraBatchLimits, CameraChannels as Ch, CameraImage, CameraSpec,
    DeviceSnapshotConfig, DeviceSnapshotPool, PhysicalOptics, SensorScene, SensorSceneOptions,
    TiledCamera, TiledCameraConfig,
    readback::{ReadbackConfig, ReadbackFrame, ReadbackRing, ReadbackStamp, ReadbackTicket},
};
use molla_sim::{BodyParams, Model, ModelBuilder, State, TriMesh};
use std::{sync::Mutex, time::Duration};

static GPU: Mutex<()> = Mutex::new(());
const CHANNELS: [Ch; 5] = [
    Ch::DEPTH,
    Ch::COLOR,
    Ch::NORMAL,
    Ch::ALBEDO,
    Ch::SHAPE_INDEX,
];

fn fixture(engine: &Engine) -> (Model, State, SensorScene) {
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    builder
        .add_shape_sphere(body, 0.7, Transform::IDENTITY)
        .unwrap();
    let mesh = builder.add_mesh(TriMesh {
        vertices: vec![
            Vec3::new(-6.0, -6.0, -1.0),
            Vec3::new(6.0, -6.0, -1.0),
            Vec3::new(6.0, 6.0, -1.0),
            Vec3::new(-6.0, 6.0, -1.0),
        ],
        indices: vec![[0, 1, 2], [0, 2, 3]],
    });
    builder
        .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .shape_color
        .host_mut()
        .unwrap()
        .copy_from_slice(&[Vec3::X, Vec3::Y]);
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let scene = SensorScene::from_model(engine, &model, SensorSceneOptions::default()).unwrap();
    (model, state, scene)
}

fn specs() -> Vec<CameraSpec> {
    let a = TiledCameraConfig {
        width: 13,
        height: 9,
        ambient: Vec3::ONE,
        light_color: Vec3::ZERO,
        ..Default::default()
    };
    let b = TiledCameraConfig {
        width: 7,
        height: 11,
        fov_y_radians: 0.55,
        near: 9.0,
        ambient: Vec3::splat(0.15),
        light_color: Vec3::new(0.4, 0.7, 1.0),
        light_dir: -Vec3::Z,
        optics: PhysicalOptics {
            lens_diameter_m: 0.3,
            focal_distance_m: 1.5,
            aperture_samples: 7,
            ..Default::default()
        },
        ..a
    };
    vec![
        CameraSpec {
            sensor_id: 91,
            config: a,
            pose: Transform::new(Vec3::Z * 3.0, Quat::IDENTITY),
        },
        CameraSpec {
            sensor_id: 7,
            config: b,
            pose: Transform::new(Vec3::new(0.2, 0.1, 3.0), Quat::from_rotation_y(0.1)),
        },
        CameraSpec {
            sensor_id: 53,
            config: a,
            pose: Transform::new(Vec3::new(1.2, 0.0, 3.0), Quat::IDENTITY),
        },
        CameraSpec {
            sensor_id: 0,
            config: TiledCameraConfig {
                width: 5,
                height: 3,
                max_distance: 2.0,
                channels: Ch::DEPTH | Ch::SHAPE_INDEX,
                ..a
            },
            pose: Transform::new(Vec3::Z * 3.0, Quat::IDENTITY),
        },
    ]
}

fn ring(engine: &Engine, slots: usize) -> ReadbackRing {
    ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots,
            bytes_per_slot: 8192,
            max_total_bytes: slots as u64 * 8192,
        },
    )
    .unwrap()
}

fn enqueue(image: &CameraImage<'_>, ring: &mut ReadbackRing) -> ReadbackTicket {
    let sources: Vec<_> = CHANNELS
        .iter()
        .copied()
        .filter(|ch| image.config().channels.contains(*ch))
        .map(|ch| image.readback_source(ch).unwrap())
        .collect();
    let stamp = image.readback_stamp(Ch::DEPTH, Some(0.125)).unwrap();
    ring.try_enqueue(&sources, stamp).unwrap().unwrap()
}

fn compare(actual: &ReadbackFrame, expected: &ReadbackFrame, mask: Ch, pixels: usize) {
    for (index, channel) in CHANNELS
        .iter()
        .copied()
        .filter(|ch| mask.contains(*ch))
        .enumerate()
    {
        match channel {
            Ch::DEPTH => {
                let a = actual.channel::<Real>(index).unwrap();
                assert_eq!(a.len(), pixels);
                assert_eq!(a, expected.channel::<Real>(index).unwrap());
            }
            Ch::NORMAL => assert_eq!(
                actual.channel::<Vec3>(index).unwrap(),
                expected.channel::<Vec3>(index).unwrap()
            ),
            _ => assert_eq!(
                actual.channel::<u32>(index).unwrap(),
                expected.channel::<u32>(index).unwrap()
            ),
        }
    }
}

#[test]
#[ignore = "needs a real GPU"]
fn heterogeneous_groups_match_independent_cameras_and_pose_changes_stay_local() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let (_model, state, scene) = fixture(&engine);
    let mut specs = specs();
    let mut batch = CameraBatch::new(&engine, &specs, CameraBatchLimits::default()).unwrap();
    let mut independent: Vec<_> = specs
        .iter()
        .map(|spec| {
            let mut camera = TiledCamera::new(&engine, spec.config, 1).unwrap();
            camera.set_camera_pose(0, spec.pose).unwrap();
            camera
        })
        .collect();
    assert_eq!(batch.group_count(), 3);
    assert_eq!(
        batch
            .layouts()
            .iter()
            .map(|l| l.sensor_id)
            .collect::<Vec<_>>(),
        [91, 7, 53, 0]
    );
    assert_eq!(batch.layouts()[2].pixels, 117..234);
    assert_eq!(
        (
            batch.layouts()[2].group_index,
            batch.layouts()[2].camera_index
        ),
        (0, 1)
    );
    let mut ring = ring(&engine, 8);
    let mut initial_depth = Vec::new();
    let mut initial_stamp = None;
    for sample in 0..3 {
        if sample == 1 {
            specs[2].pose.position.x = -1.2;
            batch
                .set_camera_pose(specs[2].sensor_id, specs[2].pose)
                .unwrap();
            independent[2].set_camera_pose(0, specs[2].pose).unwrap();
        }
        if sample == 2 {
            let eye = Vec3::new(-0.3, 1.0, 3.0);
            batch
                .set_camera_look_at(53, eye, Vec3::ZERO, Vec3::Y)
                .unwrap();
            independent[2]
                .set_camera_look_at(0, eye, Vec3::ZERO, Vec3::Y)
                .unwrap();
        }
        let frame = batch.dispatch_scene(&engine, &scene, &state).unwrap();
        assert_eq!(frame.stamp().sample_id(), sample);
        initial_stamp.get_or_insert_with(|| frame.stamp().clone());
        assert!(frame.image(999).is_none());
        let mut tickets = Vec::new();
        for (i, spec) in specs.iter().enumerate() {
            let image = frame.image(spec.sensor_id).unwrap();
            assert_eq!(*image.config(), spec.config);
            assert_eq!(image.scene_stamp(Ch::DEPTH).unwrap(), Some(scene.stamp()));
            let ticket = enqueue(&image, &mut ring);
            independent[i]
                .dispatch_scene(&engine, &scene, &state)
                .unwrap();
            let pixels = (spec.config.width * spec.config.height) as usize;
            let sources: Vec<_> = CHANNELS
                .iter()
                .copied()
                .filter(|ch| spec.config.channels.contains(*ch))
                .map(|ch| independent[i].readback_source(ch, 0..pixels).unwrap())
                .collect();
            let reference = ring
                .try_enqueue(&sources, ReadbackStamp::default())
                .unwrap()
                .unwrap();
            tickets.push((ticket, reference));
        }
        for (i, (ticket, reference)) in tickets.into_iter().enumerate().rev() {
            let actual = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
            let expected = ring.wait(&reference, Duration::from_secs(10)).unwrap();
            let spec = specs[i];
            compare(
                &actual,
                &expected,
                spec.config.channels,
                (spec.config.width * spec.config.height) as usize,
            );
            assert_eq!(actual.stamp.sensor_id, spec.sensor_id);
            assert_eq!(actual.stamp.sample_id, sample);
            assert_eq!(actual.stamp.scene_revision, scene.stamp().geometry_revision);
            assert_eq!(actual.stamp.simulation_time, Some(0.125));
            let depth = actual.channel::<Real>(0).unwrap();
            if sample == 0 {
                initial_depth.push((i, depth));
            } else {
                let old = &initial_depth
                    .iter()
                    .find(|(index, _)| *index == i)
                    .unwrap()
                    .1;
                if i == 2 {
                    assert_ne!(&depth, old);
                } else {
                    assert_eq!(&depth, old);
                }
            }
            if i == 3 {
                assert!(
                    actual
                        .channel::<Real>(0)
                        .unwrap()
                        .iter()
                        .all(|d| *d == -1.0)
                );
            }
        }
        assert!(batch.is_current(initial_stamp.as_ref().unwrap()));
        assert_eq!(batch.stats().submissions, (sample + 1) * 3);
        for g in 0..3 {
            let stats = batch.group_stats(g).unwrap();
            assert_eq!((stats.rays_preparations, stats.trace_preparations), (1, 1));
            assert_eq!(
                stats.accumulator_buffer_allocations,
                if g == 1 { 2 } else { 0 }
            );
            assert_eq!(batch.query_stats(g).unwrap().submissions, 0);
        }
    }
    assert!(batch.group_stats(3).is_none());
    assert!(batch.query_stats(3).is_none());
}

#[test]
#[ignore = "needs a real GPU"]
fn reconfiguration_preserves_queued_images_but_invalidates_old_receipts() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let (model, state, scene) = fixture(&engine);
    let specs = specs();
    let mut batch = CameraBatch::new(&engine, &specs, Default::default()).unwrap();
    let mut ring = ring(&engine, 3);
    let mut snapshots = DeviceSnapshotPool::new(
        &engine,
        DeviceSnapshotConfig {
            slots: 1,
            bytes_per_slot: 8192,
            max_total_bytes: 8192,
        },
    )
    .unwrap();
    let frame = batch.dispatch_scene(&engine, &scene, &state).unwrap();
    let old = frame.stamp().clone();
    let image = frame.image(91).unwrap();
    let ticket = enqueue(&image, &mut ring);
    let stamp = image.readback_stamp(Ch::DEPTH, Some(1.0)).unwrap();
    let snapshot = snapshots
        .try_capture(&[image.readback_source(Ch::DEPTH).unwrap()], stamp)
        .unwrap()
        .unwrap();
    for channel in [Ch::NONE, Ch::ALL, Ch(32)] {
        assert!(image.readback_source(channel).is_err());
    }
    for time in [f64::NAN, f64::INFINITY, -0.1] {
        assert!(image.readback_stamp(Ch::DEPTH, Some(time)).is_err());
    }
    let partial = frame.image(0).unwrap();
    assert!(partial.readback_source(Ch::COLOR).is_err());
    assert!(partial.scene_stamp(Ch::COLOR).is_err());
    assert!(partial.readback_stamp(Ch::COLOR, None).is_err());

    let layouts = batch.layouts().to_vec();
    let before = batch.stats();
    let mut invalid = specs.clone();
    invalid[1].sensor_id = invalid[0].sensor_id;
    assert!(batch.reconfigure(&engine, &invalid).is_err());
    assert_eq!(batch.layouts(), layouts);
    assert_eq!(batch.stats(), before);
    assert!(batch.is_current(&old));
    assert!(batch.set_camera_pose(999, Transform::IDENTITY).is_err());
    assert!(
        batch
            .set_camera_look_at(999, Vec3::Z, Vec3::ZERO, Vec3::Y)
            .is_err()
    );
    assert!(
        batch
            .set_camera_pose(91, Transform::new(Vec3::splat(f64::NAN), Quat::IDENTITY))
            .is_err()
    );

    let mut changed = vec![specs[2], specs[0]];
    changed[0].config.width = 3;
    changed[1].pose.position.z = 5.0;
    batch.reconfigure(&engine, &changed).unwrap();
    assert!(!batch.is_current(&old));
    assert_eq!(batch.stats().reconfigurations, 1);
    assert_eq!(batch.layouts()[0].sensor_id, 53);
    let frame = batch.dispatch(&engine, &model, &state).unwrap();
    assert_eq!(frame.stamp().sample_id(), 1);
    assert_eq!(frame.stamp().generation(), 1);
    let current = frame.stamp().clone();
    let image = frame.image(91).unwrap();
    assert_eq!(image.scene_stamp(Ch::DEPTH).unwrap(), None);
    let newest = enqueue(&image, &mut ring);
    let preserved = ring
        .try_enqueue(
            &[snapshot
                .channel::<Real>(0)
                .unwrap()
                .readback_source(0..117)
                .unwrap()],
            snapshot.stamp(),
        )
        .unwrap()
        .unwrap();
    drop(snapshot);
    let newest = ring.wait(&newest, Duration::from_secs(10)).unwrap();
    let preserved = ring.wait(&preserved, Duration::from_secs(10)).unwrap();
    let historical = ring.wait(&ticket, Duration::from_secs(10)).unwrap();
    assert_eq!(
        historical.channel::<Real>(0).unwrap(),
        preserved.channel::<Real>(0).unwrap()
    );
    assert_ne!(
        newest.channel::<Real>(0).unwrap(),
        historical.channel::<Real>(0).unwrap()
    );
    assert_eq!(historical.stamp.sensor_id, 91);
    assert_eq!(historical.stamp.sample_id, 0);
    assert_eq!(newest.stamp.sample_id, 1);
    let mut unrelated = CameraBatch::new(&engine, &[], Default::default()).unwrap();
    let foreign_receipt = unrelated
        .dispatch(&engine, &model, &state)
        .unwrap()
        .stamp()
        .clone();
    assert!(!batch.is_current(&foreign_receipt));
    batch.reset().unwrap();
    assert!(!batch.is_current(&current));
    let frame = batch.dispatch(&engine, &model, &state).unwrap();
    assert_eq!(
        (frame.stamp().generation(), frame.stamp().sample_id()),
        (2, 2)
    );
}

#[test]
#[ignore = "needs a real GPU"]
fn empty_batches_disabled_groups_and_foreign_engines_have_explicit_behavior() {
    let _lock = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().expect("GPU required");
    let (model, state, scene) = fixture(&engine);
    let mut batch = CameraBatch::new(&engine, &[], Default::default()).unwrap();
    let frame = batch.dispatch_scene(&engine, &scene, &state).unwrap();
    assert!(frame.images().is_empty());
    assert!(frame.image(0).is_none());
    assert_eq!(batch.stats().submissions, 0);
    assert_eq!(batch.stats().frames, 1);
    let mut specs = specs();
    specs[1].config.channels = Ch::NONE;
    batch.reconfigure(&engine, &specs).unwrap();
    let frame = batch.dispatch_scene(&engine, &scene, &state).unwrap();
    assert!(frame.image(7).unwrap().readback_source(Ch::DEPTH).is_err());
    assert_eq!(batch.stats().submissions, 2);
    let foreign = Engine::new_gpu().expect("second GPU engine required");
    let before = batch.stats();
    let layouts = batch.layouts().to_vec();
    assert!(batch.dispatch(&foreign, &model, &state).is_err());
    assert!(batch.dispatch_scene(&foreign, &scene, &state).is_err());
    assert!(batch.reconfigure(&foreign, &specs).is_err());
    assert_eq!(batch.stats(), before);
    assert_eq!(batch.layouts(), layouts);
}
