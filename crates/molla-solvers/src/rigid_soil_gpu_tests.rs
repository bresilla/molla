use super::*;
use crate::mpm::{MpmConfig, MpmSolidShape};
use molla_math::{Mat3, SpatialVector, Vec3};
use molla_sim::MassProperties;

#[test]
#[ignore = "requires a real GPU adapter"]
fn bindings_disable_reenable_remove_and_contact_ownership() {
    binding_lifecycle(false);
}

#[test]
#[ignore = "requires timestamp GPU support"]
fn bindings_profile_retains_committed_capture_and_discards_rollback() {
    binding_lifecycle(true);
}

fn binding_lifecycle(profile: bool) {
    let features = if profile {
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
    } else {
        wgpu::Features::empty()
    };
    let engine = Arc::new(Engine::new_gpu_with_features(features).unwrap());
    let mut scene = RigidScene::new().unwrap();
    let desc = BodyDesc {
        additional_mass: Some(MassProperties {
            mass: 1.0,
            local_com: Vec3::new(0.03125, -0.0625, 0.015625),
            inertia: Mat3::IDENTITY,
        }),
        ..Default::default()
    };
    let body = scene.insert_body(desc.clone()).unwrap();
    scene.insert_body(desc).unwrap();
    let states = |scene: &RigidScene| {
        std::array::from_fn(|_| {
            let mut state = scene.state().clone_host().unwrap();
            state
                .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
                .unwrap();
            state
        })
    };
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 8;
    cfg.grid_origin = [-0.5; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 4;
    cfg.max_rigid = 1;
    let mut soil = MpmSimulation::new(&engine, cfg).unwrap();
    soil.set_particles(engine.queue().unwrap(), &[[0.0; 3]])
        .unwrap();
    if profile {
        soil.prepare_step_profiling(1).unwrap();
    }
    let backup = MpmFrameBackup::new(engine.clone(), &soil, 1_000_000).unwrap();
    let mut runtime = RuntimeSoil {
        soil,
        bindings: vec![MpmRigidDeviceBinding {
            body,
            shape: MpmSolidShape::Sphere { radius: 0.25 },
        }],
        bridge: None,
        projection: None,
        machine_fault: None,
        projection_config: None,
        prepared_stamp: None,
        published: None,
        substep_dt: 0.0,
        completion_timeout: std::time::Duration::from_secs(5),
        backup,
        max_bytes: 1_000_000,
        applied: false,
        stats: Default::default(),
        profile_enabled: profile,
        profiling_current: false,
        profile_frame: None,
        profile_host_ms: HOST_STAGES,
        profile_completion_start: None,
        machine_profiler: profile
            .then(|| profile::MachineProfiler::new(engine.device().unwrap(), 1).unwrap()),
        loads: None,
        fail_after_steps: None,
        contact_trace: None,
        force_separate_submissions: false,
    };
    runtime.prepare(&engine, &scene, &states(&scene)).unwrap();
    assert_eq!(runtime.stats.live_bindings, 1);
    scene.set_body_enabled(body, false).unwrap();
    let mut disabled = states(&scene);
    runtime.prepare(&engine, &scene, &disabled).unwrap();
    assert_eq!(runtime.stats.live_bindings, 0);
    let rejected = PreparedMpmRigidDeviceBridge::new(
        engine.clone(),
        &mut runtime.soil,
        &scene,
        &disabled,
        &runtime.bindings,
        1_000_000,
    );
    assert!(
        rejected
            .err()
            .unwrap()
            .to_string()
            .contains("disabled body")
    );
    let before = runtime.soil.read_positions(&engine).unwrap();
    runtime.begin(&engine, 0.0001).unwrap();
    runtime.begin_profile(engine.queue().unwrap(), 0.0001, 1);
    let host_start = runtime.profile_host_start();
    runtime
        .before_step(&engine, &scene, &mut disabled, 0, 0.0001)
        .unwrap();
    runtime.profile_host_end(ProfileHostStage::Soil, host_start);
    if let Some((queries, slots)) = runtime.machine_timestamp("fixture", Some(0)) {
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        encoder.write_timestamp(&queries, slots[0]);
        encoder.write_timestamp(&queries, slots[1]);
        runtime.resolve_machine_profile(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
    }
    runtime.after_step(&mut disabled).unwrap();
    runtime.profile_submissions_finished();
    runtime.commit();
    let committed_host = runtime.profile_host_ms;
    if profile {
        assert_eq!(runtime.profile_frame, Some(1));
        assert!(runtime.soil.step_profile_pending());
        assert!(committed_host[ProfileHostStage::Soil as usize].1 > 0.0);
        assert!(committed_host[ProfileHostStage::Completion as usize].1 > 0.0);
    } else {
        assert_eq!(committed_host, HOST_STAGES);
    }
    assert_ne!(before, runtime.soil.read_positions(&engine).unwrap());
    scene.set_body_enabled(body, true).unwrap();
    let mut active = states(&scene);
    runtime.prepare(&engine, &scene, &active).unwrap();
    assert_eq!(runtime.stats.live_bindings, 1);
    let mut expected_impulse = [0.0f32; 6];
    if !profile {
        runtime.loads = Some(
            loads::FrameLoads::new(
                engine.device().unwrap(),
                runtime.soil.rigid_wrench_buffer(),
                runtime.soil.rigid_center_buffer(),
                vec![body],
                scene.model_stamp(),
            )
            .unwrap(),
        );
        runtime.stats.retained_bytes += loads::FrameLoads::bytes(1);
        runtime.begin(&engine, 0.0001).unwrap();
        for step in 0..3 {
            let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            runtime
                .before_step(&engine, &scene, &mut active, step % 2, 0.0001)
                .unwrap();
            runtime.after_step(&mut active).unwrap();
            assert_eq!(transfers.finish().readback_bytes, 0);
            assert_eq!(allocations.finish().observations, 0);
            let wrench = runtime.soil.read_rigid_wrenches(&engine).unwrap()[0];
            for (sum, value) in expected_impulse
                .iter_mut()
                .zip(wrench.0.into_iter().chain(wrench.1))
            {
                *sum += value * 0.0001f32;
            }
        }
        runtime.commit();
    }
    runtime.begin(&engine, 0.0001).unwrap();
    runtime.begin_profile(engine.queue().unwrap(), 0.0001, 1);
    assert!(runtime.profile_host_start().is_none());
    runtime
        .before_step(&engine, &scene, &mut active, 0, 0.0001)
        .unwrap();
    runtime.rollback(Some(&mut active)).unwrap();
    assert_eq!(runtime.profile_host_ms, committed_host);
    assert_eq!(runtime.stats.rolled_back_frames, 1);
    if !profile {
        for retry in 0..2 {
            let sample = runtime
                .loads
                .as_ref()
                .unwrap()
                .read(&engine, &scene.model_stamp(), runtime.completion_timeout)
                .unwrap();
            assert_eq!(sample.committed_frame, 2);
            assert_eq!(sample.simulated_seconds, f64::from(0.0001f32) * 3.0);
            let actual = sample.bodies[0]
                .linear
                .into_iter()
                .chain(sample.bodies[0].angular);
            for (actual, expected) in actual.zip(expected_impulse) {
                assert!((actual - expected).abs() <= 1e-6 * expected.abs().max(1.0));
            }
            let expected_f64 = expected_impulse.map(f64::from);
            let linear = Vec3::from_slice(&expected_f64[..3]);
            let angular = Vec3::from_slice(&expected_f64[3..]);
            let expected_origin = angular + scene.body_world_com(body).unwrap().cross(linear);
            for (actual, expected) in sample.bodies[0]
                .world_origin_angular
                .into_iter()
                .zip(expected_origin.to_array())
            {
                assert!((f64::from(actual) - expected).abs() <= 1e-6 * expected.abs().max(1.0));
            }
            assert_eq!(
                runtime.stats.retained_bytes,
                runtime.backup.bytes() + loads::FrameLoads::bytes(1)
            );
            if retry == 0 {
                runtime.prepare(&engine, &scene, &active).unwrap();
                runtime.begin(&engine, 0.0001).unwrap();
                runtime
                    .before_step(&engine, &scene, &mut active, 0, 0.0001)
                    .unwrap();
                runtime.rollback(Some(&mut active)).unwrap();
            }
        }
    }
    if profile {
        assert_eq!(runtime.stats.dropped_profile_frames, 1);
        assert_eq!(runtime.profile_frame, Some(1));
        let machine = runtime
            .machine_profiler
            .as_mut()
            .unwrap()
            .read(&engine, runtime.completion_timeout)
            .unwrap();
        assert_eq!(machine.stages.len(), 1);
        assert_eq!(machine.stages[0].stage, "fixture");
        assert!(machine.work.is_empty());
        assert_eq!(machine.readback_bytes, 16);
        assert_eq!(
            runtime
                .soil
                .read_step_profile()
                .unwrap()
                .unwrap()
                .stages
                .len(),
            5
        );
        runtime.profile_frame = None;
        runtime.prepare(&engine, &scene, &active).unwrap();
        runtime.begin(&engine, 0.0001).unwrap();
        runtime.begin_profile(engine.queue().unwrap(), 0.0001, 1);
        assert_eq!(runtime.profile_host_ms, HOST_STAGES);
        runtime.machine_timestamp("failed", Some(0)).unwrap();
        runtime
            .before_step(&engine, &scene, &mut active, 0, 0.0001)
            .unwrap();
        assert!(runtime.soil.step_profile_pending());
        runtime.rollback(Some(&mut active)).unwrap();
        assert!(!runtime.soil.step_profile_pending());
        assert_eq!(runtime.profile_frame, None);
        assert!(
            runtime
                .machine_profiler
                .as_mut()
                .unwrap()
                .read(&engine, runtime.completion_timeout)
                .unwrap()
                .stages
                .is_empty()
        );
    }
    let mut collider =
        molla_sim::runtime::ColliderDesc::new(molla_sim::runtime::ColliderGeometry::Cuboid {
            half_extents: Vec3::splat(0.1),
        });
    collider.parent = Some(body);
    scene.insert_collider(collider).unwrap();
    assert!(
        runtime
            .validate(&scene)
            .unwrap_err()
            .to_string()
            .contains("rigid colliders")
    );
    scene.remove_body(body).unwrap();
    let removed = states(&scene);
    runtime.prepare(&engine, &scene, &removed).unwrap();
    assert!(runtime.bindings.is_empty());
    assert_eq!(runtime.stats.live_bindings, 0);
    let scope = molla_core::gpu_metrics::TransferScope::start().unwrap();
    runtime.prepare(&engine, &scene, &removed).unwrap();
    assert_eq!(scope.finish().upload_bytes, 0);
    println!(
        "binding lifecycle: disable/free soil/re-enable/pending rollback/removal/contact ownership pass"
    );
}
use molla_sim::runtime::BodyDesc;

fn rejected_checkpoint_without_gpu_work(world: &mut RigidWorld, limit: u32) {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    assert!(world.read_mpm_soil_checkpoint(limit).is_err());
    assert_eq!(allocations.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
    assert_eq!(dispatches.finish().encoded_dispatches, 0);
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn machine_soil_frame_rollback_and_resume() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene
        .insert_body(BodyDesc {
            velocity: SpatialVector::new(Vec3::X * 0.05, Vec3::ZERO),
            additional_mass: Some(MassProperties {
                mass: 10.0,
                local_com: Vec3::ZERO,
                inertia: Mat3::IDENTITY,
            }),
            ..Default::default()
        })
        .unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 16;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 8;
    cfg.max_rigid = 1;
    cfg.gravity_y = 0.0;
    cfg.youngs = 100.0;
    cfg.particle_mass = 0.25;
    cfg.particle_volume = 0.25 / 1600.0;
    let mut soil = MpmSimulation::new(&engine, cfg).unwrap();
    soil.set_particles(engine.queue().unwrap(), &[[0.0; 3], [0.0625; 3]])
        .unwrap();
    println!("runtime fixture: soil ready, constructing GPU machine");
    let mut world = RigidWorld::from_scene_gpu(scene, engine.clone()).unwrap();
    rejected_checkpoint_without_gpu_work(&mut world, 2);
    world.solver.substeps = 2;
    println!("runtime fixture: GPU machine ready");
    world
        .attach_mpm_soil(
            soil,
            vec![MpmRigidDeviceBinding {
                body,
                shape: MpmSolidShape::Box {
                    half_extents: [0.25; 3],
                },
            }],
            2_000_000,
        )
        .unwrap();
    let before = world.read_mpm_soil_positions().unwrap();
    rejected_checkpoint_without_gpu_work(&mut world, 2);
    world.step(0.0002).unwrap();
    assert_ne!(before, world.read_mpm_soil_positions().unwrap());
    assert!(world.scene.body_velocity(body).unwrap().linear.x < 0.0499);
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 1);
    rejected_checkpoint_without_gpu_work(&mut world, 1);
    let stats_before = world.mpm_soil_stats();
    let (frame, observed) = world.read_mpm_soil_checkpoint(2).unwrap();
    assert_eq!(frame, 1);
    assert_eq!(observed.particle_count(), 2);
    assert_eq!(world.mpm_soil_stats(), stats_before);
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    let checkpoint = world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .soil
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap()
        .to_bytes();
    let diagnostics = world.gpu_diagnostics().unwrap();
    assert_eq!(observed.to_bytes(), checkpoint);
    assert_eq!(
        world.read_mpm_soil_checkpoint(2).unwrap().1.to_bytes(),
        checkpoint
    );
    let published = world.read_mpm_soil_wrenches().unwrap();
    assert_eq!(published.committed_frame, 1);
    assert_eq!(published.substep_dt, 0.0001);
    world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .fail_after_steps = Some(2);
    let error = world.step(0.0002).unwrap_err();
    assert!(error.to_string().contains("injected failure"), "{error}");
    assert_eq!(pose, world.scene.body_pose(body).unwrap());
    assert_eq!(velocity, world.scene.body_velocity(body).unwrap());
    assert_eq!(
        checkpoint,
        world
            .gpu
            .as_mut()
            .unwrap()
            .soil
            .as_mut()
            .unwrap()
            .soil
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .to_bytes()
    );
    assert_eq!(
        world.gpu_diagnostics().unwrap().completed_frames,
        diagnostics.completed_frames
    );
    assert_eq!(world.mpm_soil_stats().unwrap().rolled_back_frames, 1);
    let (frame, restored_checkpoint) = world.read_mpm_soil_checkpoint(2).unwrap();
    assert_eq!(frame, 1);
    assert_eq!(restored_checkpoint.to_bytes(), checkpoint);
    let restored = world.read_mpm_soil_wrenches().unwrap();
    assert_eq!(restored.committed_frame, published.committed_frame);
    assert_eq!(restored.substep_dt, published.substep_dt);
    assert_eq!(restored.bodies.len(), published.bodies.len());
    for (a, b) in restored.bodies.iter().zip(&published.bodies) {
        assert_eq!(a.body, b.body);
        assert_eq!(a.force.map(f32::to_bits), b.force.map(f32::to_bits));
        assert_eq!(a.torque.map(f32::to_bits), b.torque.map(f32::to_bits));
    }
    world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .fail_after_steps = None;
    world.step(0.0002).unwrap();
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 2);
    assert!(!world.mpm_soil_stats().unwrap().faulted);
    assert_ne!(pose, world.scene.body_pose(body).unwrap());
    assert_eq!(world.read_mpm_soil_checkpoint(2).unwrap().0, 2);
    world.scene.set_body_enabled(body, false).unwrap();
    rejected_checkpoint_without_gpu_work(&mut world, 2);
    world.scene.set_body_enabled(body, true).unwrap();
    let mut soil = world.detach_mpm_soil().unwrap().unwrap();
    assert!(world.mpm_soil_stats().is_none());
    assert_eq!(soil.particle_count(), 2);
    soil.step(&engine, 0.0001).unwrap();
    println!(
        "runtime fixture: coupled advancement, bit-exact failed-frame rollback, retry and detach pass"
    );
    world
        .attach_mpm_soil(
            soil,
            vec![MpmRigidDeviceBinding {
                body,
                shape: MpmSolidShape::Box {
                    half_extents: [0.25; 3],
                },
            }],
            2_000_000,
        )
        .unwrap();
    assert!(
        world
            .set_mpm_soil_completion_timeout(std::time::Duration::ZERO)
            .is_err()
    );
    world.step(0.0002).unwrap();
    let pose = world.scene.body_pose(body).unwrap();
    let velocity = world.scene.body_velocity(body).unwrap();
    let stats = world.mpm_soil_stats().unwrap();
    assert!(world.read_mpm_soil_wrenches().is_ok());
    world
        .set_mpm_soil_completion_timeout(std::time::Duration::from_nanos(1))
        .unwrap();
    let error = world.step(0.0002).unwrap_err();
    assert!(error.to_string().contains("quarantined"), "{error}");
    assert!(engine.check_device().is_err());
    assert_eq!(world.scene.body_pose(body).unwrap(), pose);
    assert_eq!(world.scene.body_velocity(body).unwrap(), velocity);
    let faulted = world.mpm_soil_stats().unwrap();
    assert!(faulted.faulted);
    rejected_checkpoint_without_gpu_work(&mut world, 2);
    assert_eq!(faulted.committed_frames, stats.committed_frames);
    assert_eq!(faulted.rolled_back_frames, stats.rolled_back_frames);
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    assert!(world.read_mpm_soil_wrenches().is_err());
    assert!(world.read_mpm_soil_positions().is_err());
    assert!(world.detach_mpm_soil().is_err());
    assert!(world.step(0.0002).is_err());
    assert!(
        world
            .set_mpm_soil_completion_timeout(std::time::Duration::from_secs(5))
            .is_err()
    );
    let transfers = transfers.finish();
    assert_eq!(transfers.upload_bytes, 0);
    assert_eq!(transfers.readback_bytes, 0);
    assert_eq!(dispatches.finish().encoded_dispatches, 0);
    println!(
        "runtime fixture: completion expiry quarantines the shared device; CPU frame restored; no commit, GPU rollback or reuse"
    );
}
