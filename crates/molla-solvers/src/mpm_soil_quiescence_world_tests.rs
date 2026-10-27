use super::*;

fn finish_quiet_read<T>(mut poll: impl FnMut() -> Result<Option<T>>) -> Result<T> {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if let Some(value) = poll()? {
            return Ok(value);
        }
        assert!(
            std::time::Instant::now() < deadline,
            "owner quiescence deadline"
        );
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
}

#[test]
#[ignore = "requires GPU; identified quiescence reads on memory and disk owners"]
fn quiescence_gpu_world_identity_and_pending_boundaries() {
    let engine = Engine::new_gpu().unwrap();
    let dir = Directory::new();
    let (store, archive, cfg) = seed(&dir);
    let residency = SoilResidencyLimits {
        regions: 3,
        tiles: 4096,
        claims: 12288,
    };
    let mut disk = MpmSoilDiskWorld::new(
        &engine,
        MpmSimulation::new_sparse(&engine, cfg, 128).unwrap(),
        store,
        residency,
        MpmSoilStoreReadLimits {
            tiles: 4096,
            encoded_bytes: 1 << 20,
            ..io_limits()
        },
        20656,
    )
    .unwrap();
    let mut memory = MpmSoilWorld::new(
        &engine,
        MpmSimulation::new_sparse(&engine, cfg, 128).unwrap(),
        archive,
        residency,
        20656,
    )
    .unwrap();
    assert!(disk.prepare_swept_quiescence(2080).is_err());
    assert!(memory.prepare_swept_quiescence(2080).is_err());
    assert!(disk.prepare_swept_projection_cache(14448).is_err());
    assert!(memory.prepare_swept_projection_cache(14448).is_err());
    disk.prepare_swept_motion_guard(16 + 129 * 128).unwrap();
    memory.prepare_swept_motion_guard(16 + 129 * 128).unwrap();
    disk.prepare_swept_quiescence(2080).unwrap();
    memory.prepare_swept_quiescence(2080).unwrap();
    disk.prepare_swept_projection_cache(14448).unwrap();
    memory.prepare_swept_projection_cache(14448).unwrap();
    assert_eq!(disk.swept_projection_cache_bytes(), 14448);
    assert_eq!(memory.swept_projection_cache_bytes(), 14448);
    assert_eq!(disk.swept_quiescence_bytes(), 2080);
    assert_eq!(memory.swept_quiescence_bytes(), 2080);
    let envelope = MpmSoilRegionEnvelope {
        id: SoilRegionId(5),
        minimum: [0.02, 0.3, -0.07],
        maximum: [0.02, 0.3, -0.07],
        particle_travel_per_substep: [0.01; 3],
        substeps: 3,
    };
    disk.begin_swept_region_update(&[envelope]).unwrap();
    memory.begin_swept_region_update(&[envelope]).unwrap();
    assert!(disk.read_swept_quiescence(&engine, 129).is_err());
    assert!(memory.read_swept_quiescence(&engine, 129).is_err());
    assert!(disk.disable_swept_quiescence().is_err());
    assert!(memory.disable_swept_quiescence().is_err());
    assert!(disk.disable_swept_projection_cache().is_err());
    assert!(memory.disable_swept_projection_cache().is_err());
    poll(&mut disk, &engine).unwrap();
    poll_memory(&mut memory, &engine).unwrap();
    assert!(disk.read_swept_quiescence(&engine, 129).is_err());
    assert!(memory.read_swept_quiescence(&engine, 129).is_err());
    disk.step_n(&engine, 0.0001, 3).unwrap();
    memory.step_n(&engine, 0.0001, 3).unwrap();
    let d = disk.read_swept_quiescence(&engine, 129).unwrap();
    let m = memory.read_swept_quiescence(&engine, 129).unwrap();
    assert_eq!(
        d.identity().unwrap().particle_ids(),
        disk.identity().particle_ids()
    );
    assert_eq!(
        m.identity().unwrap().particle_ids(),
        memory.identity().particle_ids()
    );
    assert_eq!(d.particles().len(), disk.active_particle_count() as usize);
    assert_eq!(m.particles().len(), memory.active_particle_count() as usize);
    assert!(!d.particles().is_empty() && !m.particles().is_empty());
    assert!(
        d.particles()
            .iter()
            .chain(m.particles())
            .all(|p| p.consecutive_steps == 0 && p.dt == 0.0001)
    );
    disk.validate_swept_quiescence(&d).unwrap();
    memory.validate_swept_quiescence(&m).unwrap();
    assert!(disk.validate_swept_quiescence(&m).is_err());
    assert!(memory.validate_swept_quiescence(&d).is_err());
    macro_rules! async_quiet_boundary {
        ($owner:ident, $baseline:ident) => {{
            assert!($owner.try_request_swept_quiescence(&engine, 129).unwrap());
            assert!(!$owner.try_request_swept_quiescence(&engine, 129).unwrap());
            assert!($owner.read_swept_quiescence(&engine, 129).is_err());
            assert!($owner.cancel_swept_quiescence().unwrap());
            assert!(!$owner.cancel_swept_quiescence().unwrap());
            assert!(!$owner.try_request_swept_quiescence(&engine, 129).unwrap());
            finish_quiet_read(|| {
                assert!($owner.poll_swept_quiescence(&engine)?.is_none());
                Ok($owner
                    .try_request_swept_quiescence(&engine, 129)?
                    .then_some(()))
            })
            .unwrap();
            let result = finish_quiet_read(|| $owner.poll_swept_quiescence(&engine)).unwrap();
            assert_eq!(result.particles(), $baseline.particles());
            assert_eq!(
                result.identity().unwrap().particle_ids(),
                $baseline.identity().unwrap().particle_ids()
            );
            assert_eq!(
                result.identity().unwrap().material_ids(),
                $baseline.identity().unwrap().material_ids()
            );
            $owner.validate_swept_quiescence(&result).unwrap();
            assert!($owner.poll_swept_quiescence(&engine).unwrap().is_none());
            result
        }};
    }
    let d = async_quiet_boundary!(disk, d);
    let m = async_quiet_boundary!(memory, m);
    disk.prepare_swept_support(1 << 25).unwrap();
    memory.prepare_swept_support(1 << 25).unwrap();
    let ds = disk.read_swept_support(&engine, &d, 1, 129).unwrap();
    let ms = memory.read_swept_support(&engine, &m, 1, 129).unwrap();
    assert_eq!(ds.particles(), ms.particles());
    assert!(
        ds.particles()
            .iter()
            .all(|p| !p.quiet && p.touches_nonquiet)
    );
    disk.validate_swept_support(&ds).unwrap();
    memory.validate_swept_support(&ms).unwrap();
    assert!(disk.try_request_swept_support(&engine, &m, 1, 129).is_err());
    assert!(
        memory
            .try_request_swept_support(&engine, &d, 1, 129)
            .is_err()
    );
    assert!(disk.try_request_swept_support(&engine, &d, 1, 129).unwrap());
    assert!(
        memory
            .try_request_swept_support(&engine, &m, 1, 129)
            .unwrap()
    );
    assert!(!disk.try_request_swept_support(&engine, &d, 1, 129).unwrap());
    assert!(memory.cancel_swept_support().unwrap());
    assert!(!memory.cancel_swept_support().unwrap());
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    assert!(
        !memory
            .try_request_swept_support(&engine, &m, 1, 129)
            .unwrap()
    );
    loop {
        assert!(memory.poll_swept_support(&engine).unwrap().is_none());
        if memory
            .try_request_swept_support(&engine, &m, 1, 129)
            .unwrap()
        {
            break;
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let (mut disk_result, mut memory_result) = (None, None);
    while disk_result.is_none() || memory_result.is_none() {
        if disk_result.is_none() {
            disk_result = disk.poll_swept_support(&engine).unwrap();
        }
        if memory_result.is_none() {
            memory_result = memory.poll_swept_support(&engine).unwrap();
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    let disk_result = disk_result.unwrap();
    let memory_result = memory_result.unwrap();
    assert_eq!(disk_result.particles(), ds.particles());
    assert_eq!(memory_result.particles(), ms.particles());
    disk.validate_swept_support(&disk_result).unwrap();
    memory.validate_swept_support(&memory_result).unwrap();
    assert_eq!(
        disk_result.quiescence().identity().unwrap().particle_ids(),
        disk.identity().particle_ids()
    );
    assert!(disk.poll_swept_support(&engine).unwrap().is_none());
    assert!(memory.poll_swept_support(&engine).unwrap().is_none());
    let dc = disk
        .read_swept_support_closure(&engine, &d, 1, 129, 1)
        .unwrap();
    let mc = memory
        .read_swept_support_closure(&engine, &m, 1, 129, 1)
        .unwrap();
    assert_eq!(dc.particles(), mc.particles());
    assert!(
        dc.particles()
            .iter()
            .all(|p| !p.quiet && p.connected_to_nonquiet)
    );
    disk.validate_swept_support_closure(&dc).unwrap();
    memory.validate_swept_support_closure(&mc).unwrap();
    let requested = [d.identity().unwrap().particle_ids()[0]];
    let dr = disk
        .read_swept_retained_support(&engine, &d, 1, 129, 1, &requested)
        .unwrap();
    let mr = memory
        .read_swept_retained_support(&engine, &m, 1, 129, 1, &requested)
        .unwrap();
    assert_eq!(dr.particles(), mr.particles());
    assert_eq!(dr.requested_particles(), requested);
    assert!(dr.particles().iter().all(|p| !p.quiet && p.retain));
    assert_eq!(dr.particles().iter().filter(|p| p.requested).count(), 1);
    disk.validate_swept_retained_support(&dr).unwrap();
    memory.validate_swept_retained_support(&mr).unwrap();
    assert!(disk.validate_swept_retained_support(&mr).is_err());
    assert!(memory.validate_swept_retained_support(&dr).is_err());
    assert!(
        disk.read_swept_region_support(&engine, &d, 1, 129, 1, &[envelope])
            .is_err()
    );
    disk.prepare_swept_support_regions(2, 64).unwrap();
    memory.prepare_swept_support_regions(2, 64).unwrap();
    assert_eq!(disk.swept_support_region_bytes(), 64);
    assert_eq!(memory.swept_support_region_bytes(), 64);
    let dg = disk
        .read_swept_region_support(&engine, &d, 1, 129, 1, &[envelope])
        .unwrap();
    let mg = memory
        .read_swept_region_support(&engine, &m, 1, 129, 1, &[envelope])
        .unwrap();
    assert_eq!(dg.particles(), mg.particles());
    assert_eq!(dg.requested_particles(), mg.requested_particles());
    assert!(!dg.requested_particles().is_empty());
    disk.validate_swept_retained_support(&dg).unwrap();
    memory.validate_swept_retained_support(&mg).unwrap();
    assert!(disk.validate_swept_retained_support(&mg).is_err());
    assert!(
        disk.read_swept_region_support(&engine, &d, 1, 129, 1, &[envelope, envelope])
            .is_err()
    );
    assert!(disk.validate_swept_support_closure(&mc).is_err());
    assert!(memory.validate_swept_support_closure(&dc).is_err());
    assert!(disk.validate_swept_support(&ms).is_err());
    assert!(memory.validate_swept_support(&ds).is_err());
    macro_rules! check_async_support {
        ($owner:ident, $snapshot:ident, $foreign:ident, $baseline:ident, $request:ident, $poll:ident, $validate:ident $(, $arg:expr)*) => {{
            assert!($owner.$request(&engine, &$foreign, 1, 129, 1 $(, $arg)*).is_err());
            assert!($owner.$request(&engine, &$snapshot, 1, 129, 1 $(, $arg)*).unwrap());
            assert!(!$owner.$request(&engine, &$snapshot, 1, 129, 1 $(, $arg)*).unwrap());
            assert!($owner.poll_swept_support(&engine).is_err());
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
            let result = loop {
                if let Some(result) = $owner.$poll(&engine).unwrap() { break result; }
                assert!(std::time::Instant::now() < deadline);
                std::thread::sleep(std::time::Duration::from_millis(1));
            };
            assert_eq!(result.particles(), $baseline.particles());
            assert_eq!((result.minimum_quiet_steps(), result.rounds()), (1, 1));
            assert_eq!(result.quiescence().identity().unwrap().particle_ids(), $snapshot.identity().unwrap().particle_ids());
            $owner.$validate(&result).unwrap();
            assert!($owner.$poll(&engine).unwrap().is_none());
        }};
    }
    check_async_support!(
        disk,
        d,
        m,
        dc,
        try_request_swept_support_closure,
        poll_swept_support_closure,
        validate_swept_support_closure
    );
    check_async_support!(
        memory,
        m,
        d,
        mc,
        try_request_swept_support_closure,
        poll_swept_support_closure,
        validate_swept_support_closure
    );
    check_async_support!(
        disk,
        d,
        m,
        dr,
        try_request_swept_retained_support,
        poll_swept_retained_support,
        validate_swept_retained_support,
        &requested
    );
    check_async_support!(
        memory,
        m,
        d,
        mr,
        try_request_swept_retained_support,
        poll_swept_retained_support,
        validate_swept_retained_support,
        &requested
    );
    check_async_support!(
        disk,
        d,
        m,
        dg,
        try_request_swept_region_support,
        poll_swept_region_support,
        validate_swept_retained_support,
        &[envelope]
    );
    check_async_support!(
        memory,
        m,
        d,
        mg,
        try_request_swept_region_support,
        poll_swept_region_support,
        validate_swept_retained_support,
        &[envelope]
    );
    assert!(disk.try_request_swept_quiescence(&engine, 129).unwrap());
    assert!(memory.try_request_swept_quiescence(&engine, 129).unwrap());
    disk.begin_swept_region_update(&[envelope]).unwrap();
    memory.begin_swept_region_update(&[envelope]).unwrap();
    assert!(disk.try_request_swept_quiescence(&engine, 129).is_err());
    assert!(memory.try_request_swept_quiescence(&engine, 129).is_err());
    assert!(disk.poll_swept_quiescence(&engine).is_err());
    assert!(memory.poll_swept_quiescence(&engine).is_err());
    assert!(disk.cancel_swept_quiescence().is_err());
    assert!(memory.cancel_swept_quiescence().is_err());
    macro_rules! pending_support_rejected {
        ($owner:ident, $snapshot:ident) => {{
            assert!(
                $owner
                    .try_request_swept_support_closure(&engine, &$snapshot, 1, 129, 1)
                    .is_err()
            );
            assert!(
                $owner
                    .try_request_swept_retained_support(&engine, &$snapshot, 1, 129, 1, &requested)
                    .is_err()
            );
            assert!(
                $owner
                    .try_request_swept_region_support(&engine, &$snapshot, 1, 129, 1, &[envelope])
                    .is_err()
            );
            assert!($owner.poll_swept_support_closure(&engine).is_err());
            assert!($owner.poll_swept_retained_support(&engine).is_err());
            assert!($owner.poll_swept_region_support(&engine).is_err());
        }};
    }
    pending_support_rejected!(disk, d);
    pending_support_rejected!(memory, m);
    assert!(disk.poll_swept_support(&engine).is_err());
    assert!(memory.poll_swept_support(&engine).is_err());
    assert!(disk.cancel_swept_support().is_err());
    assert!(memory.cancel_swept_support().is_err());
    assert!(disk.validate_swept_quiescence(&d).is_err());
    assert!(disk.validate_swept_support(&ds).is_err());
    assert!(memory.validate_swept_support(&ms).is_err());
    assert!(disk.validate_swept_support_closure(&dc).is_err());
    assert!(memory.validate_swept_support_closure(&mc).is_err());
    assert!(disk.validate_swept_retained_support(&dr).is_err());
    assert!(memory.validate_swept_retained_support(&mr).is_err());
    assert!(disk.validate_swept_retained_support(&dg).is_err());
    assert!(memory.validate_swept_retained_support(&mg).is_err());
    assert!(
        disk.read_swept_region_support(&engine, &d, 1, 129, 1, &[envelope])
            .is_err()
    );
    assert!(
        memory
            .read_swept_region_support(&engine, &m, 1, 129, 1, &[envelope])
            .is_err()
    );
    assert!(
        disk.read_swept_retained_support(&engine, &d, 1, 129, 1, &requested)
            .is_err()
    );
    assert!(
        memory
            .read_swept_retained_support(&engine, &m, 1, 129, 1, &requested)
            .is_err()
    );
    assert!(
        disk.read_swept_support_closure(&engine, &d, 1, 129, 1)
            .is_err()
    );
    assert!(
        memory
            .read_swept_support_closure(&engine, &m, 1, 129, 1)
            .is_err()
    );
    assert!(disk.disable_swept_support().is_err());
    assert!(memory.disable_swept_support().is_err());
    assert!(memory.validate_swept_quiescence(&m).is_err());
    poll(&mut disk, &engine).unwrap();
    poll_memory(&mut memory, &engine).unwrap();
    let historical_disk = finish_quiet_read(|| disk.poll_swept_quiescence(&engine)).unwrap();
    let historical_memory = finish_quiet_read(|| memory.poll_swept_quiescence(&engine)).unwrap();
    assert_eq!(historical_disk.particles(), d.particles());
    assert_eq!(historical_memory.particles(), m.particles());
    assert_eq!(
        historical_disk.identity().unwrap().particle_ids(),
        d.identity().unwrap().particle_ids()
    );
    assert_eq!(
        historical_memory.identity().unwrap().particle_ids(),
        m.identity().unwrap().particle_ids()
    );
    assert_eq!(
        disk.last_transfer_stats().path,
        MpmSoilTransferPath::Unchanged
    );
    assert_eq!(
        memory.last_transfer_stats().path,
        MpmSoilTransferPath::Unchanged
    );
    disk.validate_swept_quiescence(&historical_disk).unwrap();
    memory
        .validate_swept_quiescence(&historical_memory)
        .unwrap();
    disk.step_n(&engine, 0.0001, 1).unwrap();
    memory.step_n(&engine, 0.0001, 1).unwrap();
    assert!(disk.validate_swept_quiescence(&historical_disk).is_err());
    assert!(
        memory
            .validate_swept_quiescence(&historical_memory)
            .is_err()
    );
    assert!(disk.validate_swept_quiescence(&d).is_err());
    assert!(memory.validate_swept_quiescence(&m).is_err());
    let d = disk.read_swept_quiescence(&engine, 129).unwrap();
    let m = memory.read_swept_quiescence(&engine, 129).unwrap();
    disk.validate_swept_quiescence(&d).unwrap();
    memory.validate_swept_quiescence(&m).unwrap();
    disk.disable_swept_quiescence().unwrap();
    memory.disable_swept_quiescence().unwrap();
    assert!(disk.validate_swept_quiescence(&d).is_err());
    assert!(memory.validate_swept_quiescence(&m).is_err());
    assert!(disk.read_swept_quiescence(&engine, 129).is_err());
    assert!(memory.read_swept_quiescence(&engine, 129).is_err());
    assert_eq!(disk.swept_quiescence_bytes(), 0);
    assert_eq!(memory.swept_quiescence_bytes(), 0);
    assert_eq!(disk.swept_projection_cache_bytes(), 0);
    assert_eq!(memory.swept_projection_cache_bytes(), 0);
    assert_eq!(disk.swept_support_bytes(), 0);
    assert_eq!(memory.swept_support_bytes(), 0);
    println!(
        "quiescence owners=memory,disk identified_particles={} moving_streaks_zero=true pending_reads_rejected=true",
        d.particles().len()
    );
}
