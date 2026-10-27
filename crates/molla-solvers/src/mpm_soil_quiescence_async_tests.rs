use super::*;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};

fn await_result<T>(mut poll: impl FnMut() -> Result<Option<T>>) -> Result<T> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if let Some(result) = poll()? {
            return Ok(result);
        }
        assert!(
            std::time::Instant::now() < deadline,
            "quiescence observation deadline"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn step(sim: &mut MpmSimulation, engine: &Engine, dt: f32, count: usize) {
    let step = sim.prepare_step(dt, count).unwrap();
    let mut batch = sim.begin_command_batch(
        engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default()),
    );
    batch.encode_step(&step).unwrap();
    batch.submit(engine.queue().unwrap());
}

fn identified_selection(checkpoint: &MpmSoilCheckpoint, base: u64) -> MpmSoilSelection {
    use molla_vehicle::{SoilMaterialId, SoilParticleId, SoilTileLayout};
    let ids: Vec<_> = (0..checkpoint.particle_count())
        .map(|i| SoilParticleId(base + u64::from(i)))
        .collect();
    let materials: Vec<_> = (0..checkpoint.materials().len().max(1))
        .map(|i| SoilMaterialId(i as u64 + 1))
        .collect();
    let archive = checkpoint
        .partition_tiles(SoilTileLayout::new([0.; 3], 0.5).unwrap(), &ids, &materials)
        .unwrap();
    archive
        .select(&archive.tiles().collect::<Vec<_>>(), 129)
        .unwrap()
}

#[test]
fn quiescence_async_request_and_poll_contract() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<MpmSimulation>();
    send_sync::<crate::mpm::MpmSoilWorld>();
    send_sync::<crate::mpm::MpmSoilDiskWorld>();
    let source = include_str!("mpm_soil_quiescence_async.rs");
    let request = source
        .split_once("fn try_request_soil_quiescence(")
        .unwrap()
        .1
        .split_once("fn poll_soil_quiescence(")
        .unwrap()
        .0;
    assert!(!request.contains(".poll(") && !request.contains(".recv("));
    assert!(
        request.find("observer.pending.is_some()").unwrap()
            < request.find("create_command_encoder").unwrap()
    );
    assert!(request.find("self.step_epoch.clone()").unwrap() < request.find(".submit(").unwrap());
    let poll = source
        .split_once("fn poll_soil_quiescence(")
        .unwrap()
        .1
        .split_once("fn cancel_soil_quiescence(")
        .unwrap()
        .0;
    assert!(poll.contains("poll_soil_quiescence_inner(engine, false)"));
    assert!(!poll.contains("wait_indefinitely"));
    let inner = source
        .split_once("fn poll_soil_quiescence_inner(")
        .unwrap()
        .1;
    assert!(!inner.contains("self.step_epoch") && !inner.contains("soil_identity_snapshot()"));
    assert!(!inner.contains(".recv("));
}

#[test]
#[ignore = "requires GPU; bounded quiescence snapshots and request-time provenance"]
fn quiescence_gpu_async_readback_lifecycle() {
    let engine = Engine::new_gpu().unwrap();
    let foreign = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..3 {
        let cfg = MpmConfig {
            max_particles: 129,
            max_rigid: 1,
            grid_size: 16,
            grid_origin: [-0.8, -0.5, -0.8],
            cell_size: 0.1,
            gravity_y: 0.0,
            time_step: MpmTimeStep::Explicit,
            ..MpmConfig::sand()
        };
        let mut sim = if mode == 2 {
            MpmSimulation::new_sparse(&engine, cfg, 64).unwrap()
        } else {
            MpmSimulation::new(&engine, cfg).unwrap()
        };
        if mode == 1 {
            sim.set_active_grid(true).unwrap();
        }
        assert!(sim.poll_soil_quiescence(&engine).is_err());
        assert!(!sim.cancel_soil_quiescence());
        sim.set_particles(queue, &vec![[0.0, 0.3, 0.0]; 129])
            .unwrap();
        sim.prepare_soil_motion_guard().unwrap();
        sim.set_soil_motion_bound(Some([0.01; 3]));
        sim.prepare_soil_quiescence(2080).unwrap();
        assert_eq!(sim.soil_quiescence_bytes(), 2080);
        assert!(sim.try_request_soil_quiescence(&engine, 129).is_err());
        assert!(sim.poll_soil_quiescence(&engine).unwrap().is_none());
        step(&mut sim, &engine, 0.0001, 2);
        let baseline = sim.read_soil_quiescence(&engine, 129).unwrap();
        assert!(
            baseline
                .particles()
                .iter()
                .all(|p| p.consecutive_steps == 2 && p.dt == 0.0001)
        );
        let before = sim.checkpoint_soil_with(queue).unwrap().to_bytes();
        let allocation = DeviceScope::start().unwrap();
        let transfer = TransferScope::start().unwrap();
        let dispatch = DispatchScope::start().unwrap();
        assert!(sim.try_request_soil_quiescence(&engine, 128).is_err());
        assert!(sim.try_request_soil_quiescence(&foreign, 129).is_err());
        assert!(sim.try_request_soil_quiescence(&engine, 129).unwrap());
        assert!(!sim.try_request_soil_quiescence(&engine, 129).unwrap());
        assert!(sim.read_soil_quiescence(&engine, 129).is_err());
        assert!(sim.poll_soil_quiescence(&foreign).is_err());
        let result = await_result(|| sim.poll_soil_quiescence(&engine)).unwrap();
        assert_eq!(result.particles(), baseline.particles());
        assert!(Arc::ptr_eq(&result.epoch, &baseline.epoch));
        assert_eq!(result.attempt, baseline.attempt);
        sim.validate_soil_quiescence(&result).unwrap();
        assert!(sim.poll_soil_quiescence(&engine).unwrap().is_none());
        assert_eq!(allocation.finish().observations, 0);
        assert_eq!(dispatch.finish().encoded_dispatches, 0);
        let traffic = transfer.finish();
        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 1048));
        assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), before);
        sim.prepare_soil_support(17496).unwrap();
        assert!(
            sim.try_request_soil_support_closure(&engine, &result, 2, 129, 2)
                .unwrap()
        );
        let support = await_result(|| sim.poll_soil_support_closure(&engine)).unwrap();
        assert!(
            support
                .particles()
                .iter()
                .all(|p| p.quiet && !p.connected_to_nonquiet)
        );
        sim.validate_soil_quiescence(support.quiescence()).unwrap();
        for dt in [0.0002, 0.0001, 0.0002] {
            assert!(sim.try_request_soil_quiescence(&engine, 129).unwrap());
            assert!(sim.cancel_soil_quiescence());
            assert!(!sim.cancel_soil_quiescence());
            step(&mut sim, &engine, dt, 1);
            assert!(!sim.try_request_soil_quiescence(&engine, 129).unwrap());
            await_result(|| {
                assert!(sim.poll_soil_quiescence(&engine)?.is_none());
                Ok(sim.try_request_soil_quiescence(&engine, 129)?.then_some(()))
            })
            .unwrap();
            let current = await_result(|| sim.poll_soil_quiescence(&engine)).unwrap();
            assert!(
                current
                    .particles()
                    .iter()
                    .all(|p| p.consecutive_steps == 1 && p.dt == dt)
            );
            assert_eq!(
                current.particles(),
                sim.read_soil_quiescence(&engine, 129).unwrap().particles()
            );
        }
        assert!(sim.try_request_soil_quiescence(&engine, 129).unwrap());
        step(&mut sim, &engine, 0.0002, 2);
        let historical = await_result(|| sim.poll_soil_quiescence(&engine)).unwrap();
        assert!(
            historical
                .particles()
                .iter()
                .all(|p| p.consecutive_steps == 1)
        );
        assert!(sim.validate_soil_quiescence(&historical).is_err());
        assert!(
            sim.try_request_soil_support_closure(&engine, &historical, 1, 129, 2)
                .is_err()
        );
        let current = sim.read_soil_quiescence(&engine, 129).unwrap();
        assert!(current.particles().iter().all(|p| p.consecutive_steps == 3));
        assert!(sim.try_request_soil_quiescence(&engine, 129).unwrap());
        let prepared = sim.prepare_step(0.0002, 1).unwrap();
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        batch.encode_step(&prepared).unwrap();
        drop(batch);
        let cancelled_step = await_result(|| sim.poll_soil_quiescence(&engine)).unwrap();
        assert_eq!(cancelled_step.particles(), current.particles());
        assert_eq!(cancelled_step.attempt, current.attempt);
        assert!(sim.validate_soil_quiescence(&cancelled_step).is_err());
        let checkpoint = sim.checkpoint_soil_with(queue).unwrap();
        let first = identified_selection(&checkpoint, 1 << 40);
        let second = identified_selection(&checkpoint, 1 << 44);
        sim.restore_soil_selection(&engine, &first).unwrap();
        step(&mut sim, &engine, 0.0001, 1);
        let identified_before = sim.read_soil_quiescence(&engine, 129).unwrap();
        assert!(sim.try_request_soil_quiescence(&engine, 129).unwrap());
        sim.restore_soil_selection(&engine, &second).unwrap();
        let restored = await_result(|| sim.poll_soil_quiescence(&engine)).unwrap();
        assert_eq!(restored.particles(), identified_before.particles());
        assert_eq!(
            restored.identity().unwrap().particle_ids(),
            first.particle_ids()
        );
        assert_eq!(
            restored.identity().unwrap().material_ids(),
            first.material_ids()
        );
        assert_eq!(
            sim.soil_identity_snapshot().unwrap().particle_ids(),
            second.particle_ids()
        );
        assert!(Arc::ptr_eq(&restored.epoch, &identified_before.epoch));
        assert!(sim.validate_soil_quiescence(&restored).is_err());
        assert!(sim.try_request_soil_quiescence(&engine, 129).is_err());
        step(&mut sim, &engine, 0.0001, 1);
        assert!(sim.try_request_soil_quiescence(&engine, 129).unwrap());
        sim.set_particles(queue, &[[0.0, 0.3, 0.0]; 3]).unwrap();
        let old_count = await_result(|| sim.poll_soil_quiescence(&engine)).unwrap();
        assert_eq!(old_count.particles().len(), 129);
        assert_eq!(
            old_count.identity().unwrap().particle_ids(),
            second.particle_ids()
        );
        assert!(sim.soil_identity_snapshot().is_none());
        assert!(sim.validate_soil_quiescence(&old_count).is_err());
        assert!(sim.try_request_soil_quiescence(&engine, 129).is_err());
        step(&mut sim, &engine, 0.0001, 1);
        let current = sim.read_soil_quiescence(&engine, 129).unwrap();
        assert_eq!(current.particles().len(), 3);
        queue.write_buffer(
            &sim.active_tiles.list,
            sim.active_tiles.metadata_offset + 8,
            &1u32.to_le_bytes(),
        );
        assert!(sim.try_request_soil_quiescence(&engine, 129).unwrap());
        assert!(await_result(|| sim.poll_soil_quiescence(&engine)).is_err());
        assert!(sim.poll_soil_quiescence(&engine).unwrap().is_none());
        queue.write_buffer(
            &sim.active_tiles.list,
            sim.active_tiles.metadata_offset + 8,
            &0u32.to_le_bytes(),
        );
        assert!(sim.try_request_soil_quiescence(&engine, 129).unwrap());
        assert_eq!(
            await_result(|| sim.poll_soil_quiescence(&engine))
                .unwrap()
                .particles(),
            current.particles()
        );
        assert!(sim.try_request_soil_quiescence(&engine, 129).unwrap());
        let (send, receive) = std::sync::mpsc::channel();
        sim.motion_guard
            .as_mut()
            .unwrap()
            .quiescence
            .as_mut()
            .unwrap()
            .pending
            .as_mut()
            .unwrap()
            .receive = std::sync::Mutex::new(receive);
        drop(send);
        assert!(await_result(|| sim.poll_soil_quiescence(&engine)).is_err());
        assert!(sim.try_request_soil_quiescence(&engine, 129).is_err());
        assert!(sim.poll_soil_quiescence(&engine).is_err());
        sim.prepare_soil_quiescence(2080).unwrap();
        assert!(sim.poll_soil_quiescence(&engine).is_err());
        sim.disable_soil_quiescence();
        sim.prepare_soil_quiescence(2080).unwrap();
        assert!(sim.try_request_soil_quiescence(&engine, 129).is_err());
        step(&mut sim, &engine, 0.0001, 1);
        assert!(sim.try_request_soil_quiescence(&engine, 129).unwrap());
        sim.disable_soil_quiescence();
        assert!(sim.poll_soil_quiescence(&engine).is_err());
        sim.prepare_soil_quiescence(2080).unwrap();
        step(&mut sim, &engine, 0.0001, 1);
        assert!(sim.try_request_soil_quiescence(&engine, 129).unwrap());
        assert_eq!(
            await_result(|| sim.poll_soil_quiescence(&engine))
                .unwrap()
                .particles(),
            current.particles()
        );
        assert!(sim.try_request_soil_quiescence(&engine, 129).unwrap());
        if mode == 2 {
            engine.destroy_device().unwrap();
            assert!(sim.try_request_soil_quiescence(&engine, 129).is_err());
            assert!(sim.poll_soil_quiescence(&engine).is_err());
            assert!(sim.cancel_soil_quiescence());
        }
        drop(sim);
        if mode != 2 {
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        }
        println!(
            "async quiescence mode={mode} exact_parity=true request_time_provenance=true upload=0 readback=1048 gpu_allocations=0 dispatches=0"
        );
    }
}
