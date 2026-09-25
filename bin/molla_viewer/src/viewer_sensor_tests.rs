use super::*;
use bevy::ecs::system::RunSystemOnce;
use std::time::Duration;

fn drain(world: &World) {
    world
        .resource::<Sim>()
        .engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
}

fn velocity(world: &mut World, tick: u64, x: f32) {
    let mut sim = world.resource_mut::<Sim>();
    let bytes: Vec<u8> = [x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        .into_iter()
        .flat_map(f32::to_ne_bytes)
        .collect();
    sim.engine.queue().unwrap().write_buffer(
        sim.state_a.body_qd.device_buffer().unwrap(),
        0,
        &bytes,
    );
    sim.state_a.body_qd.mark_device_dirty();
    sim.sensor_clock.tick = tick;
}

fn collect_imu(world: &mut World) {
    world.run_system_once(update_imu_example).unwrap();
    drain(world);
    world.run_system_once(update_imu_example).unwrap();
    assert!(world.resource::<ImuExampleState>().error.is_none());
}

#[test]
#[ignore = "requires GPU"]
fn viewer_sensor_lab_shared_mesh_cadence_occlusion_imu_and_lifecycle() {
    let (model, visuals) = Sim::build_scene_model(Scene::SensorLab);
    let (sim, _) = Sim::finalize(Scene::SensorLab, model, visuals);
    let mut world = World::new();
    world.insert_resource(sim);
    world.insert_resource(SensorLabState::default());
    world.insert_resource(Assets::<Image>::default());
    world.insert_resource(EguiUserTextures::default());
    let mut ribbon = RibbonOpen::default();
    ribbon.set(RIBBON_RIGHT, PANE_SCENE);
    world.insert_resource(ribbon);
    world.insert_resource(RibbonPlacement::default());
    world.insert_resource(SimRunState {
        paused: true,
        step_once: false,
    });
    world.insert_resource(SimStats::default());
    world.insert_resource(GpuNpExampleState::default());
    world.insert_resource(GpuBpExampleState::default());
    world.insert_resource(ReferencePendulumState::default());
    let collect = |world: &mut World| {
        world.run_system_once(update_sensor_lab_motion).unwrap();
        world.run_system_once(update_sensor_lab).unwrap();
        drain(world);
        world.run_system_once(update_sensor_lab).unwrap();
        assert!(
            world.resource::<SensorLabState>().error.is_none(),
            "{:?}",
            world.resource::<SensorLabState>().error
        );
    };
    collect(&mut world);
    let state = world.resource::<SensorLabState>();
    let runtime = state.runtime.as_ref().unwrap();
    assert_eq!(runtime.cloud.len(), 325);
    assert_eq!(runtime.cloud[2 * 65 + 32].1, 0);
    assert!((runtime.cloud[2 * 65 + 32].0.z - 1.35).abs() < 1e-5);
    assert!(runtime.reading.is_none());
    let scene_stamp = runtime.scene.stamp();
    let handle = state.image.clone().unwrap();
    let initial_image = world
        .resource::<Assets<Image>>()
        .get(&handle)
        .unwrap()
        .data
        .clone()
        .unwrap();
    world.run_system_once(step_simulation).unwrap();
    collect(&mut world);
    assert_eq!(world.resource::<Sim>().sensor_clock.tick, 0);
    for _ in 0..60 {
        world.resource_mut::<SimRunState>().step_once = true;
        world.run_system_once(step_simulation).unwrap();
        collect(&mut world);
    }
    let state = world.resource::<SensorLabState>();
    let runtime = state.runtime.as_ref().unwrap();
    assert_eq!(
        runtime.displays.each_ref().map(|p| p.submitted),
        [11, 6, 31]
    );
    assert_eq!(runtime.skipped, [0; 3]);
    assert_eq!(runtime.scene.stamp(), scene_stamp);
    assert_eq!(runtime.lidar.query_stats().scene_builds, 1);
    assert_eq!(runtime.camera.query_stats().scene_builds, 1);
    assert_eq!(runtime.cloud[2 * 65 + 32].1, 1);
    assert!((runtime.cloud[2 * 65 + 32].0.z + 0.3).abs() < 1e-5);
    assert_eq!(
        runtime.displays[0].readback.stats().copied_bytes,
        11 * 325 * 20
    );
    assert_eq!(
        runtime.displays[1].readback.stats().copied_bytes,
        6 * 160 * 120 * 4
    );
    assert_eq!(runtime.displays[2].readback.stats().copied_bytes, 30 * 32);
    for preview in &runtime.displays {
        assert_eq!(preview.readback.stats().blocking_waits, 0);
        assert_eq!(preview.latest.unwrap().simulation_time, Some(1.0));
        if let Some(timer) = &preview.timer {
            assert!(timer.latest.is_some());
        }
    }
    let expected_accel =
        (sensor_lab_velocity(1.0) - sensor_lab_velocity(58.0 / 60.0)) / (2.0 / 60.0);
    let expected_force = sensor_lab_pose(1.0).rotation.inverse()
        * (expected_accel - world.resource::<Sim>().model.gravity.host().unwrap()[0]);
    let imu = runtime.reading.unwrap();
    assert!(
        (imu.accelerometer - expected_force).length() < 5e-4,
        "{imu:?} vs {expected_force:?}"
    );
    assert!((imu.gyroscope - MVec3::Y * 0.6).length() < 1e-5);
    let actual_image = world
        .resource::<Assets<Image>>()
        .get(&handle)
        .unwrap()
        .data
        .clone()
        .unwrap();
    assert_ne!(actual_image, initial_image);
    {
        let sim = world.resource::<Sim>();
        let mut camera = TiledCamera::new(&sim.engine, state.settings.camera(), 1).unwrap();
        camera
            .set_camera_look_at(
                0,
                MVec3::new(0.0, 2.0, 6.0),
                MVec3::new(0.0, 1.0, 0.0),
                MVec3::Y,
            )
            .unwrap();
        camera
            .dispatch(&sim.engine, &sim.model, &sim.state_a)
            .unwrap();
        let expected = camera
            .download_color(&sim.engine)
            .unwrap()
            .iter()
            .flat_map(|pixel| pixel.to_le_bytes())
            .collect::<Vec<_>>();
        assert_eq!(actual_image, expected);
    }
    let copied = [
        runtime.displays[1].readback.stats().copied_bytes,
        runtime.displays[2].readback.stats().copied_bytes,
    ];
    world.insert_resource(RibbonOpen::default());
    for _ in 0..20 {
        world.resource_mut::<SimRunState>().step_once = true;
        world.run_system_once(step_simulation).unwrap();
        collect(&mut world);
    }
    let runtime = world.resource::<SensorLabState>().runtime.as_ref().unwrap();
    assert_eq!(
        [
            runtime.displays[1].readback.stats().copied_bytes,
            runtime.displays[2].readback.stats().copied_bytes
        ],
        copied
    );
    let submitted = runtime.displays.each_ref().map(|p| p.submitted);
    world
        .resource_mut::<RibbonOpen>()
        .set(RIBBON_RIGHT, PANE_SCENE);
    collect(&mut world);
    assert_eq!(
        world
            .resource::<SensorLabState>()
            .runtime
            .as_ref()
            .unwrap()
            .displays
            .each_ref()
            .map(|p| p.submitted),
        submitted
    );
    {
        let mut state = world.resource_mut::<SensorLabState>();
        state.settings.rates = [6, 3, 12];
        state.settings.columns = 129;
        state.settings.camera_fov = 0.8;
    }
    collect(&mut world);
    let runtime = world.resource::<SensorLabState>().runtime.as_ref().unwrap();
    assert_eq!(runtime.cloud.len(), 645);
    assert_eq!(runtime.displays[2].submitted, submitted[2]);
    assert_eq!(runtime.scene.stamp(), scene_stamp);
    world.resource_mut::<Sim>().sensor_clock.tick = 1200;
    collect(&mut world);
    assert!(
        world
            .resource::<SensorLabState>()
            .runtime
            .as_ref()
            .unwrap()
            .skipped
            .iter()
            .all(|&n| n > 0)
    );

    let mut state = world.remove_resource::<SensorLabState>().unwrap();
    let runtime = state.runtime.as_mut().unwrap();
    let settings = state.settings;
    let mut held = Vec::new();
    {
        let sim = world.resource::<Sim>();
        for _ in 0..3 {
            let mut batch = SensorCommandBatch::new(&sim.engine).unwrap();
            let sample = runtime
                .lidar
                .encode_scene(
                    &mut batch,
                    &sim.engine,
                    &runtime.scene,
                    &sim.state_a,
                    &[molla::sensors::LidarScan {
                        sensor_id: 0,
                        pattern: molla::sensors::LidarPattern::Terrestrial(settings.scan()),
                    }],
                    &mut runtime.displays[0].pool,
                    sim.sensor_clock.stamp(0),
                )
                .unwrap()
                .unwrap();
            held.push(
                batch
                    .submit()
                    .unwrap()
                    .frame(&sample.frame)
                    .unwrap()
                    .clone(),
            );
        }
    }
    world.resource_mut::<Sim>().sensor_clock.tick = 1800;
    let submitted = runtime.displays[0].submitted;
    runtime
        .update(world.resource::<Sim>(), settings, true)
        .unwrap();
    assert_eq!(runtime.displays[0].submitted, submitted);
    assert!(runtime.displays[0].deferred > 0);
    drop(held);
    drain(&world);
    runtime
        .update(world.resource::<Sim>(), settings, true)
        .unwrap();
    assert_eq!(runtime.displays[0].submitted, submitted);
    world.insert_resource(state);
    world.resource_mut::<Sim>().sensor_clock.tick = 2000;
    collect(&mut world);
    assert!(
        world
            .resource::<SensorLabState>()
            .runtime
            .as_ref()
            .unwrap()
            .displays[0]
            .submitted
            > submitted
    );
    world.resource_mut::<Sim>().sensor_clock.tick = 2200;
    world.run_system_once(update_sensor_lab_motion).unwrap();
    world.run_system_once(update_sensor_lab).unwrap();
    world.resource_mut::<Sim>().reset();
    world.run_system_once(update_sensor_lab_motion).unwrap();
    assert!(!world.resource::<SensorLabState>().image_valid);
    assert!(world.resource::<SensorLabState>().runtime.is_none());
    collect(&mut world);
    assert_eq!(world.resource::<Assets<Image>>().len(), 1);
    assert_eq!(
        world.resource::<SensorLabState>().image.as_ref(),
        Some(&handle)
    );
    world.resource_mut::<SensorLabState>().settings.columns = 0;
    world.run_system_once(update_sensor_lab).unwrap();
    assert!(world.resource::<SensorLabState>().error.is_some());
    world.resource_mut::<SensorLabState>().settings.columns = 65;
    world.resource_mut::<Sim>().reset();
    collect(&mut world);
    world.resource_mut::<Sim>().rebuild_scene(Scene::Lidar);
    world.run_system_once(update_sensor_lab_motion).unwrap();
    assert!(world.resource::<SensorLabState>().runtime.is_none());
    world.resource_mut::<Sim>().rebuild_scene(Scene::SensorLab);
    collect(&mut world);
    assert_eq!(world.resource::<Assets<Image>>().len(), 1);
}

#[test]
#[ignore = "requires GPU"]
fn viewer_sensor_annotation_gpu_chain_parity_empty_pause_reset_and_switch() {
    let (model, visuals) = Sim::build_scene_model(Scene::SyntheticAnnotation);
    let (sim, _) = Sim::finalize(Scene::SyntheticAnnotation, model, visuals);
    let mut world = World::new();
    world.insert_resource(sim);
    world.insert_resource(SynthAnnotationExampleState::default());
    world.insert_resource(SimRunState {
        paused: true,
        step_once: false,
    });
    world.insert_resource(SimStats::default());
    world.insert_resource(GpuNpExampleState::default());
    world.insert_resource(GpuBpExampleState::default());
    world.insert_resource(ReferencePendulumState::default());
    let tuples = |boxes: &[BoundingBox2D]| {
        let mut result = boxes
            .iter()
            .map(|b| {
                (
                    b.shape_id,
                    b.class_id,
                    b.x_min,
                    b.y_min,
                    b.x_max,
                    b.y_max,
                    b.pixel_count,
                )
            })
            .collect::<Vec<_>>();
        result.sort_unstable();
        result
    };
    let reference = |world: &World| {
        let sim = world.resource::<Sim>();
        let mut config = synth_annotation_config();
        config.channels = CameraChannels::ALL;
        let mut camera = TiledCamera::new(&sim.engine, config, 1).unwrap();
        camera
            .set_camera_look_at(
                0,
                MVec3::new(0.0, 2.6, 5.0),
                MVec3::new(0.3, 0.5, 0.0),
                MVec3::Y,
            )
            .unwrap();
        camera
            .dispatch(&sim.engine, &sim.model, &sim.state_a)
            .unwrap();
        let shapes = camera.download_shape_index(&sim.engine).unwrap();
        let mut dataset = molla::sensors::Dataset::default();
        for id in 1..=3 {
            dataset.label_shapes(&[id], id);
        }
        tuples(
            &dataset
                .annotate(0, shapes, config.width, config.height)
                .bboxes,
        )
    };
    let collect = |world: &mut World| {
        world
            .run_system_once(update_synth_annotation_example)
            .unwrap();
        drain(world);
        world
            .run_system_once(update_synth_annotation_example)
            .unwrap();
        assert!(
            world
                .resource::<SynthAnnotationExampleState>()
                .error
                .is_none(),
            "{:?}",
            world.resource::<SynthAnnotationExampleState>().error
        );
    };
    let initial = reference(&world);
    assert_eq!(initial.len(), 3);
    collect(&mut world);
    assert_eq!(
        tuples(&world.resource::<SynthAnnotationExampleState>().boxes),
        initial
    );
    world.run_system_once(step_simulation).unwrap();
    collect(&mut world);
    assert_eq!(
        world
            .resource::<SynthAnnotationExampleState>()
            .runtime
            .as_ref()
            .unwrap()
            .display
            .submitted,
        1
    );
    assert_eq!(world.resource::<Sim>().sensor_clock.tick, 0);

    {
        let mut sim = world.resource_mut::<Sim>();
        let queue = sim.engine.queue().unwrap().clone();
        for pose in sim.state_a.body_q.host_mut().unwrap() {
            pose.position.x += 100.0;
        }
        sim.state_a.body_q.upload(&queue).unwrap();
        sim.state_a.body_q.mark_device_dirty();
    }
    let empty = reference(&world);
    assert!(empty.is_empty());
    world.resource_mut::<SimRunState>().step_once = true;
    world.run_system_once(step_simulation).unwrap();
    collect(&mut world);
    let state = world.resource::<SynthAnnotationExampleState>();
    assert!(state.boxes.is_empty());
    let runtime = state.runtime.as_ref().unwrap();
    assert_eq!(runtime.camera.query_stats().scene_builds, 1);
    assert_eq!(runtime.camera.dispatch_stats().submissions, 0);
    assert_eq!(runtime.camera.dispatch_stats().encoded_frames, 2);
    assert_eq!(runtime.annotations.stats().encoded_frames, 2);
    assert_eq!(runtime.annotations.stats().submissions, 0);
    assert_eq!(runtime.annotations.stats().label_uploads, 1);
    assert_eq!(runtime.display.readback.stats().copied_bytes, 2 * 96);
    assert_eq!(runtime.display.readback.stats().blocking_waits, 0);
    assert_eq!(runtime.display.pool.stats().buffer_allocations, 12);
    assert_eq!(
        runtime.display.latest.unwrap().simulation_time,
        Some(SUBSTEPS_PER_FRAME as f64 * FIXED_DT)
    );
    assert!(world.resource::<Sim>().state_a.body_q.host().is_err());

    world.resource_mut::<SimRunState>().step_once = true;
    world.run_system_once(step_simulation).unwrap();
    world
        .run_system_once(update_synth_annotation_example)
        .unwrap();
    let revision = world.resource::<Sim>().sensor_clock.revision;
    world.resource_mut::<Sim>().reset();
    world
        .run_system_once(update_synth_annotation_example)
        .unwrap();
    assert!(
        world
            .resource::<SynthAnnotationExampleState>()
            .boxes
            .is_empty()
    );
    collect(&mut world);
    assert_eq!(
        tuples(&world.resource::<SynthAnnotationExampleState>().boxes),
        initial
    );
    assert_ne!(
        world
            .resource::<SynthAnnotationExampleState>()
            .runtime
            .as_ref()
            .unwrap()
            .display
            .latest
            .unwrap()
            .scene_revision,
        revision
    );
    {
        let mut state = world.resource_mut::<SynthAnnotationExampleState>();
        let camera = &mut state.runtime.as_mut().unwrap().camera;
        let mut config = *camera.config();
        config.channels = CameraChannels::COLOR;
        camera.set_config(config).unwrap();
    }
    world.resource_mut::<SimRunState>().step_once = true;
    world.run_system_once(step_simulation).unwrap();
    world
        .run_system_once(update_synth_annotation_example)
        .unwrap();
    assert!(
        world
            .resource::<SynthAnnotationExampleState>()
            .error
            .is_some()
    );
    world.resource_mut::<Sim>().reset();
    collect(&mut world);
    assert_eq!(
        tuples(&world.resource::<SynthAnnotationExampleState>().boxes),
        initial
    );
    world.resource_mut::<Sim>().rebuild_scene(Scene::Lidar);
    world
        .run_system_once(update_synth_annotation_example)
        .unwrap();
    assert!(
        world
            .resource::<SynthAnnotationExampleState>()
            .runtime
            .is_none()
    );
    assert!(
        world
            .resource::<SynthAnnotationExampleState>()
            .boxes
            .is_empty()
    );
    world
        .resource_mut::<Sim>()
        .rebuild_scene(Scene::SyntheticAnnotation);
    collect(&mut world);
    assert_eq!(
        tuples(&world.resource::<SynthAnnotationExampleState>().boxes),
        initial
    );
}

#[test]
#[ignore = "requires GPU"]
fn viewer_sensor_camera_gpu_processing_parity_hidden_collection_and_paused_edits() {
    let (model, visuals) = Sim::build_scene_model(Scene::Camera);
    let (mut sim, _) = Sim::finalize(Scene::Camera, model, visuals);
    let state = CameraExampleState::default();
    let mut request = state.request(&sim);
    request.config.width = 24;
    request.config.height = 16;
    let mut preview = CameraPreview::new(&sim, request.config).unwrap();
    let wait = |sim: &Sim| {
        sim.engine
            .device()
            .unwrap()
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap();
    };
    for case in 0..4 {
        request.exposure = [1.0, 1.7, 0.8, 2.0][case];
        request.tone_map = case >= 2;
        request.gamma = case >= 1;
        if case == 3 {
            request.config.optics = molla::sensors::PhysicalOptics {
                lens_diameter_m: 0.2,
                focal_distance_m: 5.0,
                aperture_samples: 3,
                focal_length_mm: 35.0,
                sensor_width_mm: 36.0,
            };
        }
        assert!(preview.update(&sim, request, true).unwrap().is_none());
        wait(&sim);
        let (frame, pose) = preview.update(&sim, request, true).unwrap().unwrap();
        assert!((pose.position - request.eye).length() < 1e-6);
        assert_eq!(frame.stamp.simulation_time, Some(0.0));
        let mut old = TiledCamera::new(&sim.engine, request.config, 1).unwrap();
        old.set_camera_look_at(0, request.eye, request.target, MVec3::Y)
            .unwrap();
        old.dispatch(&sim.engine, &sim.model, &sim.state_a).unwrap();
        let mut color = old.download_color(&sim.engine).unwrap().to_vec();
        if (request.exposure - 1.0).abs() > 1e-3 {
            molla::sensors::camera_post::apply_exposure(&mut color, request.exposure);
        }
        if request.tone_map {
            molla::sensors::camera_post::apply_reinhard_tone_map(&mut color);
        }
        if request.gamma {
            molla::sensors::camera_post::apply_gamma_srgb(&mut color);
        }
        for (actual, expected) in frame.channel::<u32>(1).unwrap().iter().zip(color) {
            for shift in [0, 8, 16, 24] {
                let a = ((actual >> shift) & 255) as i32;
                let b = ((expected >> shift) & 255) as i32;
                assert!((a - b).abs() <= 1, "case {case}: {a} != {b}");
            }
        }
        assert_eq!(
            frame.channel::<u32>(3).unwrap(),
            old.download_albedo(&sim.engine).unwrap()
        );
        assert_eq!(
            frame.channel::<u32>(4).unwrap(),
            old.download_shape_index(&sim.engine).unwrap()
        );
        for (a, b) in frame
            .channel::<f64>(0)
            .unwrap()
            .iter()
            .zip(old.download_depth(&sim.engine).unwrap())
        {
            assert!((a - b).abs() < 1e-5);
        }
        for (a, b) in frame
            .channel::<MVec3>(2)
            .unwrap()
            .iter()
            .zip(old.download_normal(&sim.engine).unwrap())
        {
            assert!((*a - *b).length() < 1e-5);
        }
        assert_eq!(preview.display.submitted, case as u64 + 1);
        assert_eq!(preview.display.readback.stats().blocking_waits, 0);
        assert_eq!(
            preview.display.readback.stats().copied_bytes,
            (case as u64 + 1) * 24 * 16 * 32
        );
        assert_eq!(preview.display.pool.stats().buffer_allocations, 15);
        assert_eq!(preview.camera.dispatch_stats().submissions, 0);
        assert_eq!(preview.camera.query_stats().scene_builds, 1);
        assert!(preview.update(&sim, request, true).unwrap().is_none());
        assert_eq!(preview.display.submitted, case as u64 + 1);
    }
    let copied = preview.display.readback.stats().copied_bytes;
    for tick in [10, 20, 30, 40] {
        sim.sensor_clock.tick = tick;
        request.tick = tick;
        request.eye.z += 0.1;
        assert!(preview.update(&sim, request, false).unwrap().is_none());
        wait(&sim);
        assert!(preview.update(&sim, request, false).unwrap().is_none());
    }
    assert_eq!(preview.display.submitted, 8);
    assert_eq!(preview.display.readback.stats().copied_bytes, copied);
    assert!(preview.update(&sim, request, true).unwrap().is_none());
    wait(&sim);
    let (frame, pose) = preview.update(&sim, request, true).unwrap().unwrap();
    assert_eq!(preview.display.submitted, 8);
    assert_eq!(frame.stamp.simulation_time, Some(40.0 * FIXED_DT));
    assert!((pose.position - request.eye).length() < 1e-6);
    assert_eq!(
        preview.display.readback.stats().copied_bytes,
        copied + 24 * 16 * 32
    );
}

#[test]
#[ignore = "requires GPU"]
fn viewer_sensor_camera_thumbnails_reset_reuse_faults_and_switch() {
    let (model, visuals) = Sim::build_scene_model(Scene::Camera);
    let (sim, _) = Sim::finalize(Scene::Camera, model, visuals);
    let mut world = World::new();
    world.insert_resource(sim);
    world.insert_resource(CameraExampleState::default());
    world.insert_resource(Assets::<Image>::default());
    world.insert_resource(EguiUserTextures::default());
    let mut ribbon = RibbonOpen::default();
    ribbon.set(RIBBON_RIGHT, PANE_SCENE);
    world.insert_resource(ribbon);
    world.insert_resource(RibbonPlacement::default());
    let collect = |world: &mut World| {
        world.run_system_once(update_camera_example).unwrap();
        drain(world);
        world.run_system_once(update_camera_example).unwrap();
        assert!(
            world.resource::<CameraExampleState>().error.is_none(),
            "{:?}",
            world.resource::<CameraExampleState>().error
        );
    };
    collect(&mut world);
    let state = world.resource::<CameraExampleState>();
    let handles = [
        state.image_color.clone().unwrap(),
        state.image_depth.clone().unwrap(),
        state.image_normal.clone().unwrap(),
        state.image_albedo.clone().unwrap(),
        state.image_index.clone().unwrap(),
    ];
    assert!(state.last_pose.is_some());
    assert_eq!(world.resource::<Assets<Image>>().len(), 5);
    assert_eq!(
        state
            .runtime
            .as_ref()
            .unwrap()
            .display
            .readback
            .stats()
            .copied_bytes,
        u64::from(CAMERA_IMG_W) * u64::from(CAMERA_IMG_H) * 32
    );
    for handle in &handles {
        let data = world
            .resource::<Assets<Image>>()
            .get(handle)
            .unwrap()
            .data
            .as_ref()
            .unwrap();
        assert_eq!(data.len(), (CAMERA_IMG_W * CAMERA_IMG_H * 4) as usize);
        assert!(data.chunks_exact(4).any(|rgba| rgba[..3] != [0, 0, 0]));
    }
    let eye = state.last_pose.unwrap().position;
    collect(&mut world);
    assert_eq!(
        world
            .resource::<CameraExampleState>()
            .last_pose
            .unwrap()
            .position,
        eye
    );
    world.resource_mut::<Sim>().sensor_clock.tick = 10;
    collect(&mut world);
    assert_ne!(
        world
            .resource::<CameraExampleState>()
            .last_pose
            .unwrap()
            .position,
        eye
    );
    world.insert_resource(RibbonOpen::default());
    let copied = world
        .resource::<CameraExampleState>()
        .runtime
        .as_ref()
        .unwrap()
        .display
        .readback
        .stats()
        .copied_bytes;
    world.resource_mut::<Sim>().sensor_clock.tick = 20;
    collect(&mut world);
    let runtime = world
        .resource::<CameraExampleState>()
        .runtime
        .as_ref()
        .unwrap();
    assert_eq!(runtime.display.readback.stats().copied_bytes, copied);
    let submitted = runtime.display.submitted;
    world
        .resource_mut::<RibbonOpen>()
        .set(RIBBON_RIGHT, PANE_SCENE);
    collect(&mut world);
    let runtime = world
        .resource::<CameraExampleState>()
        .runtime
        .as_ref()
        .unwrap();
    assert_eq!(runtime.display.submitted, submitted);
    assert_eq!(
        runtime.display.latest.unwrap().simulation_time,
        Some(20.0 * FIXED_DT)
    );
    assert_eq!(
        runtime.display.readback.stats().copied_bytes - copied,
        u64::from(CAMERA_IMG_W) * u64::from(CAMERA_IMG_H) * 32
    );
    world.resource_mut::<CameraExampleState>().exposure = 1.5;
    world.run_system_once(update_camera_example).unwrap();
    world.resource_mut::<Sim>().reset();
    world.run_system_once(update_camera_example).unwrap();
    assert!(world.resource::<CameraExampleState>().last_pose.is_none());
    collect(&mut world);
    assert_eq!(world.resource::<Assets<Image>>().len(), 5);
    assert_eq!(
        world.resource::<CameraExampleState>().image_color.as_ref(),
        Some(&handles[0])
    );
    world.resource_mut::<CameraExampleState>().fov_y = f32::NAN;
    world.run_system_once(update_camera_example).unwrap();
    assert!(world.resource::<CameraExampleState>().error.is_some());
    world.resource_mut::<CameraExampleState>().fov_y = 0.8;
    world.resource_mut::<Sim>().reset();
    collect(&mut world);
    world.resource_mut::<Sim>().rebuild_scene(Scene::Lidar);
    world.run_system_once(update_camera_example).unwrap();
    assert!(world.resource::<CameraExampleState>().runtime.is_none());
    assert!(world.resource::<CameraExampleState>().last_pose.is_none());
    world.resource_mut::<Sim>().rebuild_scene(Scene::Camera);
    collect(&mut world);
    assert_eq!(world.resource::<Assets<Image>>().len(), 5);
    assert_eq!(
        world.resource::<CameraExampleState>().image_color.as_ref(),
        Some(&handles[0])
    );
}

#[test]
fn viewer_sensor_configurable_lidar_validates_layout_limits() {
    let mut config = LidarConfig::default();
    config.ray_count_h = u32::MAX;
    config.ray_count_v = 2;
    assert!(lidar_metadata(&config).is_err());
    config.ray_count_h = 0;
    config.ray_count_v = 0;
    assert_eq!(
        lidar_metadata(&config)
            .unwrap()
            .checked_ray_count()
            .unwrap(),
        1
    );
    config.max_range = f32::NAN;
    assert!(lidar_metadata(&config).is_err());
}

#[test]
#[ignore = "requires GPU"]
fn viewer_sensor_configurable_lidar_angles_edits_pause_resize_and_reset() {
    let (model, visuals) = Sim::build_scene_model(Scene::Lidar);
    let (sim, _) = Sim::finalize(Scene::Lidar, model, visuals);
    let mut world = World::new();
    world.insert_resource(sim);
    world.insert_resource(LidarState::default());
    world.insert_resource(LidarConfig {
        auto_rotate_speed: 0.0,
        ..Default::default()
    });
    let collect = |world: &mut World| {
        world.run_system_once(update_lidar).unwrap();
        drain(world);
        world.run_system_once(update_lidar).unwrap();
        assert!(
            world.resource::<LidarState>().error.is_none(),
            "{:?}",
            world.resource::<LidarState>().error
        );
    };
    for (nh, nv, yaw, pitch) in [
        (13, 3, 0.23, -0.13),
        (1, 1, 0.43, -0.21),
        (1, 9, 6.1, 1.4),
        (11, 1, -0.2, -1.4),
        (32, 16, 3.7, 0.1),
        (7, 3, 1.3, -0.6),
    ] {
        {
            let mut config = world.resource_mut::<LidarConfig>();
            config.ray_count_h = nh;
            config.ray_count_v = nv;
            config.yaw = yaw;
            config.pitch = pitch;
            config.fov_h = 5.8;
            config.fov_v = 1.8;
        }
        collect(&mut world);
        let config = *world.resource::<LidarConfig>();
        let mut origins = Vec::new();
        let mut directions = Vec::new();
        for iy in 0..nv {
            let pitch = config.pitch
                + if nv == 1 {
                    0.0
                } else {
                    -0.5 * config.fov_v + iy as f32 * config.fov_v / (nv - 1) as f32
                };
            for ix in 0..nh {
                let yaw = config.yaw
                    + if nh == 1 {
                        0.0
                    } else {
                        -0.5 * config.fov_h + ix as f32 * config.fov_h / (nh - 1) as f32
                    };
                origins.push(config.origin.as_dvec3());
                directions.push(
                    Vec3::new(
                        yaw.sin() * pitch.cos(),
                        pitch.sin(),
                        yaw.cos() * pitch.cos(),
                    )
                    .as_dvec3(),
                );
            }
        }
        let sim = world.resource::<Sim>();
        let hits = Raycaster::new(&sim.engine, nh * nv)
            .unwrap()
            .cast(
                &sim.engine,
                &sim.model,
                &sim.state_a,
                &origins,
                &directions,
                f64::from(config.max_range),
            )
            .unwrap();
        let state = world.resource::<LidarState>();
        assert_eq!(state.rays.len(), (nh * nv) as usize);
        assert_eq!(state.pending.len(), 0);
        for ((ray, direction), hit) in state.rays.iter().zip(directions).zip(hits) {
            assert!((ray.direction - direction.as_vec3()).length() < 2e-6);
            assert_eq!(ray.origin, config.origin);
            assert_eq!(ray.hit.is_some(), hit.shape_id.is_some());
            if let Some(point) = ray.hit {
                assert!((point - hit.point.as_vec3()).length() < 2e-4);
            }
        }
        let runtime = state.runtime.as_ref().unwrap();
        assert_eq!(runtime.display.latest.unwrap().simulation_time, Some(0.0));
        assert_eq!(runtime.display.readback.stats().blocking_waits, 0);
        assert_eq!(runtime.sensor.query_stats().scene_builds, 1);
        assert_eq!(runtime.display.pool.stats().buffer_allocations, 27);
        let submitted = runtime.display.submitted;
        world.run_system_once(update_lidar).unwrap();
        assert_eq!(
            world
                .resource::<LidarState>()
                .runtime
                .as_ref()
                .unwrap()
                .display
                .submitted,
            submitted
        );
    }
    let previous = world.resource::<LidarState>().displayed.unwrap();
    let bytes = world
        .resource::<LidarState>()
        .runtime
        .as_ref()
        .unwrap()
        .display
        .readback
        .stats()
        .copied_bytes;
    {
        let mut config = world.resource_mut::<LidarConfig>();
        config.origin.x = 1.2;
        config.max_range = 7.0;
    }
    world.run_system_once(update_lidar).unwrap();
    let state = world.resource::<LidarState>();
    assert_eq!(state.displayed.unwrap().origin, previous.origin);
    assert_eq!(state.displayed.unwrap().max_range, previous.max_range);
    assert_eq!(state.pending.len(), 1);
    collect(&mut world);
    let state = world.resource::<LidarState>();
    assert_eq!(state.displayed.unwrap().max_range, 7.0);
    assert_eq!(state.rays[0].origin.x, 1.2);
    assert_eq!(
        state
            .runtime
            .as_ref()
            .unwrap()
            .display
            .readback
            .stats()
            .copied_bytes
            - bytes,
        21 * 52
    );

    world.resource_mut::<LidarConfig>().auto_rotate_speed = 0.7;
    collect(&mut world);
    let yaw = world.resource::<LidarConfig>().yaw;
    collect(&mut world);
    assert_eq!(world.resource::<LidarConfig>().yaw, yaw);
    world.resource_mut::<Sim>().sensor_clock.tick = SUBSTEPS_PER_FRAME as u64;
    collect(&mut world);
    assert!(
        (world.resource::<LidarConfig>().yaw
            - yaw
            - 0.7 * SUBSTEPS_PER_FRAME as f32 * FIXED_DT as f32)
            .abs()
            < 1e-6
    );
    assert_eq!(
        world
            .resource::<LidarState>()
            .runtime
            .as_ref()
            .unwrap()
            .display
            .latest
            .unwrap()
            .simulation_time,
        Some(SUBSTEPS_PER_FRAME as f64 * FIXED_DT)
    );

    world.resource_mut::<LidarConfig>().max_range = 8.0;
    world.run_system_once(update_lidar).unwrap();
    let displayed_count = world.resource::<LidarState>().rays.len();
    {
        let mut config = world.resource_mut::<LidarConfig>();
        config.ray_count_h = 512;
        config.ray_count_v = 64;
    }
    world.run_system_once(update_lidar).unwrap();
    assert_eq!(world.resource::<LidarState>().pending.len(), 1);
    assert_eq!(world.resource::<LidarState>().rays.len(), displayed_count);
    collect(&mut world);
    let state = world.resource::<LidarState>();
    assert_eq!(state.rays.len(), 512 * 64);
    assert!(state.pending.is_empty());
    assert_eq!(
        state
            .runtime
            .as_ref()
            .unwrap()
            .display
            .readback
            .stats()
            .copied_bytes,
        512 * 64 * 52
    );

    world.resource_mut::<LidarConfig>().max_range = 6.0;
    world.run_system_once(update_lidar).unwrap();
    let revision = world.resource::<Sim>().sensor_clock.revision;
    world.resource_mut::<Sim>().reset();
    world.run_system_once(update_lidar).unwrap();
    assert!(world.resource::<LidarState>().rays.is_empty());
    collect(&mut world);
    assert_ne!(
        world
            .resource::<LidarState>()
            .runtime
            .as_ref()
            .unwrap()
            .display
            .latest
            .unwrap()
            .scene_revision,
        revision
    );
    world.resource_mut::<Sim>().rebuild_scene(Scene::LidarV2);
    world.run_system_once(update_lidar).unwrap();
    assert!(world.resource::<LidarState>().runtime.is_none());
    assert!(world.resource::<LidarState>().rays.is_empty());
    world.resource_mut::<Sim>().rebuild_scene(Scene::Lidar);
    collect(&mut world);
    assert!(!world.resource::<LidarState>().rays.is_empty());
}

#[test]
#[ignore = "requires GPU"]
fn viewer_sensor_lidar_layout_parity_live_poses_pause_reset_and_one_shot() {
    let (model, visuals) = Sim::build_scene_model(Scene::LidarV2);
    let (sim, _) = Sim::finalize(Scene::LidarV2, model, visuals);
    let mut world = World::new();
    world.insert_resource(sim);
    world.insert_resource(LidarV2ExampleState::default());
    world.insert_resource(AerialLidarExampleState::default());
    world.insert_resource(SimRunState {
        paused: true,
        step_once: false,
    });
    world.insert_resource(SimStats::default());
    world.insert_resource(GpuNpExampleState::default());
    world.insert_resource(GpuBpExampleState::default());
    world.insert_resource(ReferencePendulumState::default());
    let meta = lidar_v2_metadata();
    let reference = |world: &World| {
        let sim = world.resource::<Sim>();
        molla::sensors::Lidar::new(&sim.engine, meta.checked_ray_count().unwrap())
            .unwrap()
            .cast(&sim.engine, &sim.model, &sim.state_a, &meta)
            .unwrap()
            .hits
            .into_iter()
            .filter(|h| h.shape_id.is_some())
            .map(|h| (h.point.as_vec3(), h.t as f32))
            .collect::<Vec<_>>()
    };
    let initial = reference(&world);
    assert!(!initial.is_empty());
    world.run_system_once(update_lidar_v2_example).unwrap();
    assert_eq!(
        world
            .resource::<LidarV2ExampleState>()
            .runtime
            .as_ref()
            .unwrap()
            .display
            .submitted,
        0
    );
    for iteration in 0..2 {
        if iteration == 1 {
            let mut sim = world.resource_mut::<Sim>();
            let queue = sim.engine.queue().unwrap().clone();
            sim.state_a.body_q.host_mut().unwrap()[0].position.x += 3.0;
            sim.state_a.body_q.upload(&queue).unwrap();
            sim.state_a.body_q.mark_device_dirty();
        }
        let expected = reference(&world);
        if iteration == 1 {
            assert_ne!(expected, initial);
        }
        world.resource_mut::<SimRunState>().step_once = true;
        world.run_system_once(step_simulation).unwrap();
        world.run_system_once(update_lidar_v2_example).unwrap();
        drain(&world);
        world.run_system_once(update_lidar_v2_example).unwrap();
        let state = world.resource::<LidarV2ExampleState>();
        assert!(state.error.is_none(), "{:?}", state.error);
        assert_eq!(state.hits.len(), expected.len());
        for ((point, t), (ep, et)) in state.hits.iter().zip(&expected) {
            assert!((*point - *ep).length() < 1e-4);
            assert!((*t - *et).abs() < 1e-4);
        }
        let runtime = state.runtime.as_ref().unwrap();
        assert_eq!(runtime.sensor.query_stats().scene_builds, 1);
        assert_eq!(
            runtime.sensor.stats().descriptor_upload_bytes,
            128 * (iteration + 1)
        );
        assert_eq!(
            runtime.display.readback.stats().copied_bytes,
            u64::from(meta.ray_count()) * 24 * (iteration + 1)
        );
        assert_eq!(runtime.display.readback.stats().blocking_waits, 0);
        assert_eq!(runtime.display.pool.stats().buffer_allocations, 27);
        if iteration == 1 {
            assert!(world.resource::<Sim>().state_a.body_q.host().is_err());
        }
    }
    for _ in 0..3 {
        world.run_system_once(update_lidar_v2_example).unwrap();
    }
    assert_eq!(
        world
            .resource::<LidarV2ExampleState>()
            .runtime
            .as_ref()
            .unwrap()
            .display
            .submitted,
        2
    );
    world.resource_mut::<SimRunState>().step_once = true;
    world.run_system_once(step_simulation).unwrap();
    world.run_system_once(update_lidar_v2_example).unwrap();
    world.resource_mut::<Sim>().reset();
    world.run_system_once(update_lidar_v2_example).unwrap();
    assert!(world.resource::<LidarV2ExampleState>().hits.is_empty());
    assert_eq!(
        world
            .resource::<LidarV2ExampleState>()
            .runtime
            .as_ref()
            .unwrap()
            .display
            .submitted,
        0
    );
    world.resource_mut::<SimRunState>().step_once = true;
    world.run_system_once(step_simulation).unwrap();
    world.run_system_once(update_lidar_v2_example).unwrap();
    drain(&world);
    world.run_system_once(update_lidar_v2_example).unwrap();
    assert_eq!(
        world.resource::<LidarV2ExampleState>().hits.len(),
        initial.len()
    );

    world
        .resource_mut::<Sim>()
        .rebuild_scene(Scene::AerialLidarPass);
    world.run_system_once(update_lidar_v2_example).unwrap();
    assert!(world.resource::<LidarV2ExampleState>().runtime.is_none());
    let meta = aerial_lidar_metadata();
    let expected = {
        let sim = world.resource::<Sim>();
        molla::sensors::AerialLidar::new(&sim.engine, meta.checked_ray_count().unwrap())
            .unwrap()
            .cast(&sim.engine, &sim.model, &sim.state_a, &meta)
            .unwrap()
    };
    world.run_system_once(update_aerial_lidar_example).unwrap();
    drain(&world);
    world.run_system_once(update_aerial_lidar_example).unwrap();
    let state = world.resource::<AerialLidarExampleState>();
    assert!(state.error.is_none(), "{:?}", state.error);
    assert_eq!(state.origins.len(), expected.origins.len());
    for (origin, expected) in state.origins.iter().zip(&expected.origins) {
        assert!((*origin - expected.as_vec3()).length() < 1e-4);
    }
    let hits: Vec<_> = expected
        .hits
        .iter()
        .filter(|h| h.shape_id.is_some())
        .collect();
    assert_eq!(state.hits.len(), hits.len());
    for (point, expected) in state.hits.iter().zip(hits) {
        assert!((*point - expected.point.as_vec3()).length() < 1e-4);
    }
    let runtime = state.runtime.as_ref().unwrap();
    assert_eq!(
        runtime.display.readback.stats().copied_bytes,
        u64::from(meta.ray_count()) * 36
    );
    assert_eq!(runtime.display.readback.stats().blocking_waits, 0);
    assert_eq!(runtime.display.cadence_hz, None);
    for _ in 0..3 {
        world.resource_mut::<SimRunState>().step_once = true;
        world.run_system_once(step_simulation).unwrap();
        world.run_system_once(update_aerial_lidar_example).unwrap();
    }
    assert_eq!(
        world
            .resource::<AerialLidarExampleState>()
            .runtime
            .as_ref()
            .unwrap()
            .display
            .submitted,
        1
    );
    world.resource_mut::<Sim>().reset();
    world.run_system_once(update_aerial_lidar_example).unwrap();
    assert!(world.resource::<AerialLidarExampleState>().hits.is_empty());
    drain(&world);
    world.run_system_once(update_aerial_lidar_example).unwrap();
    assert_eq!(
        world.resource::<AerialLidarExampleState>().origins.len(),
        expected.origins.len()
    );
    world.resource_mut::<Sim>().rebuild_scene(Scene::Imu);
    world.run_system_once(update_aerial_lidar_example).unwrap();
    assert!(
        world
            .resource::<AerialLidarExampleState>()
            .runtime
            .is_none()
    );
}

#[test]
#[ignore = "requires GPU"]
fn viewer_sensor_reset_preserves_buffer_authority_for_bodies_and_particles() {
    let mut builder = mp::ModelBuilder::new();
    let initial = MVec3::new(1.0, 2.0, 3.0);
    builder.add_body(mp::BodyParams::new().with_position(initial));
    builder.add_particle(initial, 1.0, 0.1, mp::WorldId(0));
    let (mut sim, _) = Sim::finalize(Scene::Imu, builder.build().unwrap(), Vec::new());
    sim.state_a.body_q.host_mut().unwrap()[0].position = MVec3::splat(42.0);
    sim.state_a.body_qd.host_mut().unwrap()[0] = mp::SpatialVector::new(MVec3::X, MVec3::Y);
    sim.state_b.body_q.host_mut().unwrap()[0].position = MVec3::splat(17.0);
    sim.state_b.body_qd.mark_device_dirty();
    sim.state_a.particle_q.mark_device_dirty();
    sim.state_a.particle_qd.host_mut().unwrap()[0] = MVec3::Y;
    sim.state_b.particle_q.host_mut().unwrap()[0] = MVec3::splat(13.0);
    sim.state_b.particle_qd.mark_device_dirty();
    sim.sensor_clock.tick = 10;
    sim.reset();
    assert_eq!(sim.state_a.body_q.host().unwrap()[0].position, initial);
    assert_eq!(sim.state_a.body_qd.host().unwrap()[0].linear, MVec3::ZERO);
    assert_eq!(sim.state_a.particle_q.host().unwrap()[0], initial);
    assert_eq!(sim.state_a.particle_qd.host().unwrap()[0], MVec3::ZERO);
    assert_eq!(
        sim.state_b.body_q.host().unwrap()[0].position,
        MVec3::splat(17.0)
    );
    assert_eq!(
        sim.state_b.particle_q.host().unwrap()[0],
        MVec3::splat(13.0)
    );
    assert_eq!(sim.sensor_clock.tick, 0);
    assert_eq!(sim.sensor_clock.revision, 1);
}

#[test]
#[ignore = "requires GPU"]
fn viewer_sensor_async_imu_contacts_pause_reset_backpressure_and_switch() {
    let (model, visuals) = Sim::build_scene_model(Scene::Imu);
    let (sim, _) = Sim::finalize(Scene::Imu, model, visuals);
    let mut world = World::new();
    world.insert_resource(sim);
    world.insert_resource(ImuExampleState::default());
    world.insert_resource(ContactExampleState::default());
    let mut ribbon = RibbonOpen::default();
    ribbon.set(RIBBON_RIGHT, PANE_SCENE);
    world.insert_resource(ribbon);
    world.insert_resource(RibbonPlacement::default());
    world.insert_resource(SimRunState {
        paused: true,
        step_once: false,
    });
    world.insert_resource(SimStats::default());
    world.insert_resource(GpuNpExampleState::default());
    world.insert_resource(GpuBpExampleState::default());
    world.insert_resource(ReferencePendulumState::default());
    collect_imu(&mut world);
    assert!(world.resource::<ImuExampleState>().last.is_none());

    velocity(&mut world, 10, 0.2);
    collect_imu(&mut world);
    let imu = world.resource::<ImuExampleState>();
    let first = imu.last.unwrap();
    assert!(
        (first.accelerometer - MVec3::new(12.0, molla::math::GRAVITY, 0.0)).length() < 1e-4,
        "{first:?}"
    );
    assert_eq!(imu.preview.as_ref().unwrap().submitted, 2);
    assert_eq!(
        imu.preview.as_ref().unwrap().readback.stats().copied_bytes,
        32
    );
    assert!(world.resource::<Sim>().state_a.body_qd.host().is_err());
    for _ in 0..3 {
        world.run_system_once(update_imu_example).unwrap();
    }
    assert_eq!(
        world
            .resource::<ImuExampleState>()
            .preview
            .as_ref()
            .unwrap()
            .submitted,
        2
    );

    let held = world.resource_scope(|world, mut state: Mut<ImuExampleState>| {
        let sim = world.resource::<Sim>();
        let ImuExampleState {
            sensor, preview, ..
        } = &mut *state;
        let mut frames = Vec::new();
        for _ in 0..preview.as_mut().unwrap().pool.available_slots().unwrap() {
            let mut batch = SensorCommandBatch::new(&sim.engine).unwrap();
            let out = sensor
                .as_mut()
                .unwrap()
                .encode(
                    &mut batch,
                    &sim.engine,
                    &sim.model,
                    &sim.state_a,
                    FIXED_DT,
                    &mut preview.as_mut().unwrap().pool,
                    sim.sensor_clock.stamp(6),
                )
                .unwrap()
                .unwrap();
            frames.push(batch.submit().unwrap().frame(&out.frame).unwrap().clone());
        }
        frames
    });
    drain(&world);
    velocity(&mut world, 20, 1.0);
    world.run_system_once(update_imu_example).unwrap();
    let preview = world
        .resource::<ImuExampleState>()
        .preview
        .as_ref()
        .unwrap();
    assert_eq!(preview.deferred, 1);
    assert_eq!(preview.submitted_tick, Some(10));
    world.run_system_once(update_imu_example).unwrap();
    assert_eq!(
        world
            .resource::<ImuExampleState>()
            .preview
            .as_ref()
            .unwrap()
            .deferred,
        1
    );
    drop(held);
    velocity(&mut world, 30, 1.4);
    collect_imu(&mut world);
    let imu = world.resource::<ImuExampleState>();
    assert!((imu.last.unwrap().accelerometer.x - 36.0).abs() < 1e-4);
    assert_eq!(
        imu.preview
            .as_ref()
            .unwrap()
            .readback
            .stats()
            .blocking_waits,
        0
    );
    assert_eq!(
        imu.sensor.as_ref().unwrap().readback_stats().copied_bytes,
        0
    );

    world.insert_resource(RibbonOpen::default());
    let copied = world
        .resource::<ImuExampleState>()
        .preview
        .as_ref()
        .unwrap()
        .readback
        .stats()
        .copied_bytes;
    velocity(&mut world, 35, 1.5);
    world.run_system_once(update_imu_example).unwrap();
    drain(&world);
    world.run_system_once(update_imu_example).unwrap();
    let imu = world.resource::<ImuExampleState>();
    assert_eq!(imu.preview.as_ref().unwrap().submitted_tick, Some(35));
    assert_eq!(
        imu.preview.as_ref().unwrap().readback.stats().copied_bytes,
        copied
    );
    let submissions = imu.preview.as_ref().unwrap().submitted;
    world
        .resource_mut::<RibbonOpen>()
        .set(RIBBON_RIGHT, PANE_SCENE);
    collect_imu(&mut world);
    let imu = world.resource::<ImuExampleState>();
    assert_eq!(imu.preview.as_ref().unwrap().submitted, submissions);
    assert_eq!(imu.preview.as_ref().unwrap().latest.unwrap().sample_id, 35);
    assert_eq!(
        imu.preview.as_ref().unwrap().readback.stats().copied_bytes,
        copied + 32
    );
    assert!((imu.last.unwrap().accelerometer.x - 12.0).abs() < 1e-4);

    velocity(&mut world, 40, 1.6);
    world.run_system_once(update_imu_example).unwrap();
    let revision = world.resource::<Sim>().sensor_clock.revision;
    world.resource_mut::<Sim>().reset();
    collect_imu(&mut world);
    assert!(world.resource::<ImuExampleState>().last.is_none());
    assert_eq!(world.resource::<Sim>().sensor_clock.revision, revision + 1);
    let before = world.resource::<Sim>().sensor_clock.tick;
    world.run_system_once(step_simulation).unwrap();
    assert_eq!(world.resource::<Sim>().sensor_clock.tick, before);
    world.resource_mut::<SimRunState>().step_once = true;
    world.run_system_once(step_simulation).unwrap();
    assert_eq!(
        world.resource::<Sim>().sensor_clock.tick,
        SUBSTEPS_PER_FRAME as u64
    );
    collect_imu(&mut world);
    assert!(
        world
            .resource::<ImuExampleState>()
            .last
            .unwrap()
            .accelerometer
            .is_finite()
    );
    world.resource_mut::<Sim>().reset();
    collect_imu(&mut world);
    {
        let mut sim = world.resource_mut::<Sim>();
        sim.state_a.body_qd.host_mut().unwrap()[0] = mp::SpatialVector::ZERO;
        sim.sensor_clock.tick = 10;
    }
    world.run_system_once(update_imu_example).unwrap();
    assert!(world.resource::<ImuExampleState>().error.is_some());
    world.resource_mut::<Sim>().reset();
    collect_imu(&mut world);
    assert!(world.resource::<ImuExampleState>().error.is_none());

    world.resource_mut::<Sim>().rebuild_scene(Scene::Contact);
    world.run_system_once(update_imu_example).unwrap();
    assert!(world.resource::<ImuExampleState>().sensor.is_none());
    world.run_system_once(update_contact_example).unwrap();
    assert!(world.resource::<ContactExampleState>().readings.is_empty());
    assert_eq!(
        world
            .resource::<ContactExampleState>()
            .preview
            .as_ref()
            .unwrap()
            .submitted,
        0
    );
    let expected = {
        let mut sim = world.resource_mut::<Sim>();
        let device = sim.engine.device().unwrap().clone();
        let queue = sim.engine.queue().unwrap().clone();
        sim.state_a.body_q.host_mut().unwrap()[0].position.y = 0.2;
        sim.state_a.body_q.upload(&queue).unwrap();
        let Sim {
            model,
            state_a,
            contacts,
            pipeline,
            sensor_clock,
            ..
        } = &mut *sim;
        pipeline.collide(model, state_a, contacts).unwrap();
        contacts.upload_to_device(&device, &queue).unwrap();
        assert!(contacts.count > 0);
        sensor_clock.tick = 10;
        let mut expected = vec![BodyContactReading::default(); model.body_count];
        for i in 0..contacts.count as usize {
            for (body, sign) in [
                (contacts.a_index.host().unwrap()[i], -1.0),
                (contacts.b_index.host().unwrap()[i], 1.0),
            ] {
                if let Some(out) = expected.get_mut(body as usize) {
                    out.count += 1;
                    out.max_depth = out.max_depth.max(contacts.depth.host().unwrap()[i]);
                    out.normal_sum += contacts.normal.host().unwrap()[i] * sign;
                    out.centroid += contacts.point.host().unwrap()[i];
                }
            }
        }
        for out in &mut expected {
            if out.count > 0 {
                out.centroid /= out.count as f64;
            }
        }
        expected
    };
    world.run_system_once(update_contact_example).unwrap();
    drain(&world);
    world.run_system_once(update_contact_example).unwrap();
    let state = world.resource::<ContactExampleState>();
    assert!(state.error.is_none(), "{:?}", state.error);
    assert_eq!(state.readings.len(), expected.len());
    for (actual, expected) in state.readings.iter().zip(&expected) {
        assert_eq!(actual.count, expected.count);
        assert!((actual.max_depth - expected.max_depth).abs() < 1e-5);
        assert!((actual.normal_sum - expected.normal_sum).length() < 1e-5);
        assert!((actual.centroid - expected.centroid).length() < 1e-5);
    }
    assert_eq!(state.preview.as_ref().unwrap().submitted, 1);
    assert_eq!(
        state
            .preview
            .as_ref()
            .unwrap()
            .readback
            .stats()
            .copied_bytes,
        120
    );
    assert_eq!(
        state
            .preview
            .as_ref()
            .unwrap()
            .readback
            .stats()
            .blocking_waits,
        0
    );
    assert_eq!(
        state.sensor.as_ref().unwrap().readback_stats().copied_bytes,
        0
    );
    {
        let mut sim = world.resource_mut::<Sim>();
        sim.contacts.clear();
        sim.sensor_clock.tick = 20;
    }
    world.run_system_once(update_contact_example).unwrap();
    drain(&world);
    world.run_system_once(update_contact_example).unwrap();
    assert!(
        world
            .resource::<ContactExampleState>()
            .readings
            .iter()
            .all(|r| r.count == 0 && r.normal_sum == MVec3::ZERO)
    );
    world.resource_mut::<Sim>().sensor_clock.tick = 30;
    world.run_system_once(update_contact_example).unwrap();
    world.resource_mut::<Sim>().reset();
    world.run_system_once(update_contact_example).unwrap();
    assert!(world.resource::<ContactExampleState>().readings.is_empty());
    world.resource_mut::<Sim>().rebuild_scene(Scene::Imu);
    world.run_system_once(update_contact_example).unwrap();
    assert!(world.resource::<ContactExampleState>().sensor.is_none());
    collect_imu(&mut world);
    assert!(world.resource::<ImuExampleState>().last.is_none());
}
