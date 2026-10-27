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
            "support observation deadline"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

fn complete(sim: &mut MpmSimulation, engine: &Engine) -> Result<MpmSoilSupportSnapshot> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if let Some(result) = sim.poll_soil_support(engine)? {
            return Ok(result);
        }
        assert!(
            std::time::Instant::now() < deadline,
            "support observation deadline"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
fn quiescence_support_async_nonblocking_source_contract() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<MpmSimulation>();
    send_sync::<crate::mpm::MpmSoilWorld>();
    send_sync::<crate::mpm::MpmSoilDiskWorld>();
    let source = include_str!("mpm_soil_support.rs");
    let request = source
        .split_once("    fn request_soil_support_words(")
        .unwrap()
        .1
        .split_once("    fn poll_soil_support_words(")
        .unwrap()
        .0;
    assert!(!request.contains(".poll(") && !request.contains(".recv("));
    assert!(
        request.find("soil_support_request_available(").unwrap()
            < request.find("queue.write_buffer").unwrap()
    );
    let poll = source
        .split_once("pub(in crate::mpm) fn poll_soil_support(")
        .unwrap()
        .1
        .split_once("pub(in crate::mpm) fn cancel_soil_support(")
        .unwrap()
        .0;
    assert!(poll.contains("poll_soil_support_words(engine, false, Some(SupportKind::Direct))"));
    assert!(!poll.contains("wait_indefinitely"));
    let retained_request = include_str!("mpm_soil_retained_support.rs")
        .split_once("fn try_request_soil_retained_support(")
        .unwrap()
        .1;
    assert!(
        retained_request
            .find("soil_support_request_available(")
            .unwrap()
            < retained_request.find("retained_support_seeds(").unwrap()
    );
    for (source, kind) in [
        (include_str!("mpm_soil_support.rs"), "Closure"),
        (include_str!("mpm_soil_retained_support.rs"), "Retained"),
        (include_str!("mpm_soil_support_regions.rs"), "Region"),
    ] {
        assert!(source.contains(&format!(
            "poll_soil_support_words(engine, false, Some(SupportKind::{kind}))"
        )));
    }
}

#[test]
#[ignore = "requires GPU; typed asynchronous support closure and retention lifecycle"]
fn quiescence_gpu_support_async_kinds() {
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let positions = [
        [-1., 0.5, 0.],
        [-0.8, 0.5, 0.],
        [-0.6, 0.5, 0.],
        [1., 0.5, 0.],
    ];
    let boxes = [[-1.01, 0.49, -0.01, 0., -0.99, 0.51, 0.01, 0.]];
    for mode in 0..3 {
        let (mut sim, snapshot, ids) = retained::tests::identified(&engine, &positions, mode);
        sim.prepare_soil_support_regions(2, 64).unwrap();
        let before = sim.checkpoint_soil_with(queue).unwrap().to_bytes();
        let closure = sim
            .read_soil_support_closure(&engine, &snapshot, 2, 129, 2)
            .unwrap();
        let retained = sim
            .read_soil_retained_support(&engine, &snapshot, 2, 129, 2, &ids[..1])
            .unwrap();
        let region = sim
            .read_soil_region_support(&engine, &snapshot, 2, 129, 2, &boxes)
            .unwrap();
        assert_eq!(retained.particles(), region.particles());
        for kind in [
            SupportKind::Closure,
            SupportKind::Retained,
            SupportKind::Region,
        ] {
            let allocation = DeviceScope::start().unwrap();
            let transfer = TransferScope::start().unwrap();
            let dispatch = DispatchScope::start().unwrap();
            let mut requested = ids[..1].to_vec();
            assert!(
                match kind {
                    SupportKind::Closure =>
                        sim.try_request_soil_support_closure(&engine, &snapshot, 2, 129, 2),
                    SupportKind::Retained => sim.try_request_soil_retained_support(
                        &engine, &snapshot, 2, 129, 2, &requested
                    ),
                    SupportKind::Region =>
                        sim.try_request_soil_region_support(&engine, &snapshot, 2, 129, 2, &boxes),
                    _ => unreachable!(),
                }
                .unwrap()
            );
            requested.clear();
            assert!(
                !sim.try_request_soil_support(&engine, &snapshot, 2, 129)
                    .unwrap()
            );
            assert!(
                !sim.try_request_soil_support_closure(&engine, &snapshot, 2, 129, 2)
                    .unwrap()
            );
            assert!(
                !sim.try_request_soil_retained_support(&engine, &snapshot, 2, 129, 2, &ids[..1])
                    .unwrap()
            );
            assert!(
                !sim.try_request_soil_region_support(&engine, &snapshot, 2, 129, 2, &boxes)
                    .unwrap()
            );
            assert!(sim.read_soil_support(&engine, &snapshot, 2, 129).is_err());
            assert!(sim.poll_soil_support(&engine).is_err());
            if kind != SupportKind::Closure {
                assert!(sim.poll_soil_support_closure(&engine).is_err());
            }
            if kind != SupportKind::Retained {
                assert!(sim.poll_soil_retained_support(&engine).is_err());
            }
            if kind != SupportKind::Region {
                assert!(sim.poll_soil_region_support(&engine).is_err());
            }
            if kind == SupportKind::Closure {
                let result = await_result(|| sim.poll_soil_support_closure(&engine)).unwrap();
                assert_eq!(result.particles(), closure.particles());
                assert_eq!((result.minimum_quiet_steps(), result.rounds()), (2, 2));
                sim.validate_soil_quiescence(result.quiescence()).unwrap();
            } else {
                let result = await_result(|| {
                    if kind == SupportKind::Retained {
                        sim.poll_soil_retained_support(&engine)
                    } else {
                        sim.poll_soil_region_support(&engine)
                    }
                })
                .unwrap();
                let expected = if kind == SupportKind::Retained {
                    &retained
                } else {
                    &region
                };
                assert_eq!(result.particles(), expected.particles());
                assert_eq!(result.requested_particles(), expected.requested_particles());
                assert_eq!((result.minimum_quiet_steps(), result.rounds()), (2, 2));
                sim.validate_soil_quiescence(result.quiescence()).unwrap();
            }
            assert!(sim.poll_soil_support_closure(&engine).unwrap().is_none());
            assert!(sim.poll_soil_retained_support(&engine).unwrap().is_none());
            assert!(sim.poll_soil_region_support(&engine).unwrap().is_none());
            assert_eq!(allocation.finish().observations, 0);
            let traffic = transfer.finish();
            assert_eq!(
                (traffic.upload_bytes, traffic.readback_bytes),
                (
                    match kind {
                        SupportKind::Closure => 64,
                        SupportKind::Retained => 80,
                        _ => 96,
                    },
                    32
                )
            );
            assert_eq!(
                dispatch.finish().encoded_dispatches,
                if kind == SupportKind::Region { 7 } else { 6 }
            );
        }
        let transfer = TransferScope::start().unwrap();
        for rounds in [0, 65] {
            assert!(
                sim.try_request_soil_support_closure(&engine, &snapshot, 2, 129, rounds)
                    .is_err()
            );
            assert!(
                sim.try_request_soil_retained_support(
                    &engine,
                    &snapshot,
                    2,
                    129,
                    rounds,
                    &ids[..1]
                )
                .is_err()
            );
            assert!(
                sim.try_request_soil_region_support(&engine, &snapshot, 2, 129, rounds, &boxes)
                    .is_err()
            );
        }
        assert!(
            sim.try_request_soil_retained_support(&engine, &snapshot, 2, 129, 2, &[ids[0], ids[0]])
                .is_err()
        );
        assert!(
            sim.try_request_soil_retained_support(
                &engine,
                &snapshot,
                2,
                129,
                2,
                &[molla_vehicle::SoilParticleId(0)]
            )
            .is_err()
        );
        assert!(
            sim.try_request_soil_region_support(&engine, &snapshot, 2, 129, 2, &[[f32::NAN; 8]])
                .is_err()
        );
        let traffic = transfer.finish();
        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
        assert!(
            sim.try_request_soil_retained_support(&engine, &snapshot, 2, 129, 1, &ids[..1])
                .unwrap()
        );
        assert!(
            await_result(|| sim.poll_soil_retained_support(&engine))
                .unwrap_err()
                .to_string()
                .contains("did not converge")
        );
        assert!(sim.poll_soil_retained_support(&engine).unwrap().is_none());
        assert!(
            sim.try_request_soil_retained_support(&engine, &snapshot, 2, 129, 2, &ids[..1])
                .unwrap()
        );
        assert!(sim.cancel_soil_support());
        assert!(
            !sim.try_request_soil_region_support(&engine, &snapshot, 2, 129, 2, &boxes)
                .unwrap()
        );
        await_result(|| {
            assert!(sim.poll_soil_support_closure(&engine)?.is_none());
            Ok(sim
                .try_request_soil_region_support(&engine, &snapshot, 2, 129, 2, &boxes)?
                .then_some(()))
        })
        .unwrap();
        assert_eq!(
            await_result(|| sim.poll_soil_region_support(&engine))
                .unwrap()
                .particles(),
            region.particles()
        );
        assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), before);
        assert!(
            sim.try_request_soil_region_support(&engine, &snapshot, 2, 129, 2, &boxes)
                .unwrap()
        );
        let step = sim.prepare_step(0.0001, 2).unwrap();
        let mut batch = sim.begin_command_batch(
            engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default()),
        );
        batch.encode_step(&step).unwrap();
        drop(batch);
        let historical = await_result(|| sim.poll_soil_region_support(&engine)).unwrap();
        assert_eq!(historical.particles(), region.particles());
        assert_eq!(
            historical.requested_particles(),
            region.requested_particles()
        );
        assert!(
            sim.validate_soil_quiescence(historical.quiescence())
                .is_err()
        );
        assert!(
            sim.try_request_soil_support_closure(&engine, &snapshot, 2, 129, 2)
                .is_err()
        );
        assert!(
            sim.try_request_soil_retained_support(&engine, &snapshot, 2, 129, 2, &ids[..1])
                .is_err()
        );
        assert!(
            sim.try_request_soil_region_support(&engine, &snapshot, 2, 129, 2, &boxes)
                .is_err()
        );
    }
}

#[test]
#[ignore = "requires GPU; bounded async support parity, backpressure and cancellation"]
fn quiescence_gpu_support_async_lifecycle() {
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let device = engine.device().unwrap();
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
    let positions = [[-0.3, 0.3, 0.0], [-0.1, 0.3, 0.0], [0.3, 0.3, 0.0]];
    let mut sim = MpmSimulation::new(&engine, cfg).unwrap();
    assert!(sim.poll_soil_support(&engine).is_err());
    assert!(!sim.cancel_soil_support());
    sim.set_particles(queue, &positions).unwrap();
    sim.prepare_soil_motion_guard().unwrap();
    sim.set_soil_motion_bound(Some([0.01; 3]));
    sim.prepare_soil_quiescence(2080).unwrap();
    sim.prepare_soil_support(17496).unwrap();
    assert_eq!(sim.soil_support_bytes(), 17496);
    let step = sim.prepare_step(0.0001, 2).unwrap();
    let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
    batch.encode_step(&step).unwrap();
    batch.submit(queue);
    let snapshot = sim.read_soil_quiescence(&engine, 129).unwrap();
    let baseline = sim.read_soil_support(&engine, &snapshot, 2, 129).unwrap();
    eprintln!("async support quiet records={:?}", snapshot.particles());
    assert!(
        snapshot
            .particles()
            .iter()
            .all(|p| p.consecutive_steps == 2)
    );
    let nonquiet = sim.read_soil_support(&engine, &snapshot, 3, 129).unwrap();
    assert!(
        nonquiet
            .particles()
            .iter()
            .all(|p| !p.quiet && p.touches_nonquiet)
    );
    let before = sim.checkpoint_soil_with(queue).unwrap().to_bytes();
    assert!(sim.poll_soil_support(&engine).unwrap().is_none());
    let allocation = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let dispatches = DispatchScope::start().unwrap();
    assert!(
        sim.try_request_soil_support(&engine, &snapshot, 2, 129)
            .unwrap()
    );
    assert!(
        !sim.try_request_soil_support(&engine, &snapshot, 2, 129)
            .unwrap()
    );
    assert!(sim.read_soil_support(&engine, &snapshot, 2, 129).is_err());
    assert!(
        sim.read_soil_support_closure(&engine, &snapshot, 2, 129, 2)
            .is_err()
    );
    let async_result = complete(&mut sim, &engine).unwrap();
    assert_eq!(async_result.particles(), baseline.particles());
    assert_eq!(async_result.minimum_quiet_steps(), 2);
    sim.validate_soil_quiescence(async_result.quiescence())
        .unwrap();
    assert!(sim.poll_soil_support(&engine).unwrap().is_none());
    assert_eq!(allocation.finish().observations, 0);
    let traffic = transfers.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (64, 28));
    assert_eq!(dispatches.finish().encoded_dispatches, 2);
    assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), before);

    for _ in 0..4 {
        assert!(
            sim.try_request_soil_support(&engine, &snapshot, 2, 129)
                .unwrap()
        );
        assert!(sim.cancel_soil_support());
        assert!(!sim.cancel_soil_support());
        assert!(
            !sim.try_request_soil_support(&engine, &snapshot, 3, 129)
                .unwrap()
        );
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
        loop {
            assert!(sim.poll_soil_support(&engine).unwrap().is_none());
            if sim
                .try_request_soil_support(&engine, &snapshot, 3, 129)
                .unwrap()
            {
                break;
            }
            assert!(std::time::Instant::now() < deadline);
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        let result = complete(&mut sim, &engine).unwrap();
        assert_eq!(result.minimum_quiet_steps(), 3);
        assert_eq!(result.particles(), nonquiet.particles());
        assert!(
            result
                .particles()
                .iter()
                .all(|p| !p.quiet && p.touches_nonquiet)
        );
    }
    for (minimum, bound) in [(0, 129), (2, 2)] {
        let transfers = TransferScope::start().unwrap();
        assert!(
            sim.try_request_soil_support(&engine, &snapshot, minimum, bound)
                .is_err()
        );
        let traffic = transfers.finish();
        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
        assert!(sim.poll_soil_support(&engine).unwrap().is_none());
    }
    assert!(
        sim.try_request_soil_support(&engine, &snapshot, 2, 129)
            .unwrap()
    );
    let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
    batch.encode_step(&step).unwrap();
    drop(batch);
    let historical = complete(&mut sim, &engine).unwrap();
    assert_eq!(historical.particles(), baseline.particles());
    assert!(
        sim.validate_soil_quiescence(historical.quiescence())
            .is_err()
    );
    assert!(
        sim.try_request_soil_support(&engine, &snapshot, 2, 129)
            .is_err()
    );
    let step = sim.prepare_step(0.0001, 2).unwrap();
    let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
    batch.encode_step(&step).unwrap();
    batch.submit(queue);
    let current = sim.read_soil_quiescence(&engine, 129).unwrap();
    queue.write_buffer(
        &sim.active_tiles.list,
        sim.active_tiles.metadata_offset + 8,
        &1u32.to_le_bytes(),
    );
    assert!(
        sim.try_request_soil_support(&engine, &current, 2, 129)
            .unwrap()
    );
    assert!(complete(&mut sim, &engine).is_err());
    assert!(sim.poll_soil_support(&engine).unwrap().is_none());
    queue.write_buffer(
        &sim.active_tiles.list,
        sim.active_tiles.metadata_offset + 8,
        &0u32.to_le_bytes(),
    );
    assert!(
        sim.try_request_soil_support(&engine, &current, 2, 129)
            .unwrap()
    );
    let (send, receive) = std::sync::mpsc::channel();
    sim.motion_guard
        .as_mut()
        .unwrap()
        .quiescence
        .as_mut()
        .unwrap()
        .support
        .as_mut()
        .unwrap()
        .pending
        .as_mut()
        .unwrap()
        .receive = std::sync::Mutex::new(receive);
    drop(send);
    assert!(complete(&mut sim, &engine).is_err());
    assert!(sim.poll_soil_support(&engine).is_err());
    assert!(
        sim.try_request_soil_support(&engine, &current, 2, 129)
            .is_err()
    );
    sim.disable_soil_support();
    sim.prepare_soil_support(17496).unwrap();
    assert!(
        sim.try_request_soil_support(&engine, &current, 2, 129)
            .unwrap()
    );
    sim.disable_soil_support();
    assert!(sim.poll_soil_support(&engine).is_err());
    sim.prepare_soil_support(17496).unwrap();
    assert!(
        sim.try_request_soil_support(&engine, &current, 2, 129)
            .unwrap()
    );
    assert_eq!(
        complete(&mut sim, &engine).unwrap().particles(),
        baseline.particles()
    );
    assert!(
        sim.try_request_soil_support(&engine, &current, 2, 129)
            .unwrap()
    );
    drop(sim);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}
