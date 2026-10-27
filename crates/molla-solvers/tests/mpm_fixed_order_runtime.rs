#![cfg(all(feature = "gpu", feature = "resource-counters"))]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, TransferScope};
use molla_solvers::mpm::{
    MpmConfig, MpmSimulation, MpmSoilCanonicalStore, MpmSoilDiskWorld, MpmSoilFrameCheckpoint,
    MpmSoilStoreLimits, MpmSoilStoreReadLimits,
};
use molla_vehicle::{
    SoilMaterialId, SoilParticleId, SoilRegionId, SoilRegionRequest, SoilResidencyLimits,
    SoilTileLayout,
};
use std::{
    fs,
    path::PathBuf,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

fn config() -> MpmConfig {
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 32;
    cfg.grid_origin = [-2.0; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 129;
    cfg.max_rigid = 1;
    cfg.youngs = 1000.0;
    cfg
}

fn solver(engine: &Engine, sparse: bool, cfg: MpmConfig) -> MpmSimulation {
    let mut sim = if sparse {
        MpmSimulation::new_sparse(engine, cfg, 128).unwrap()
    } else {
        MpmSimulation::new(engine, cfg).unwrap()
    };
    assert!(!sim.deterministic_p2g_enabled());
    sim.set_active_grid(true).unwrap();
    sim.prepare_parallel_particle_tile_ordering(engine.queue().unwrap(), 129)
        .unwrap();
    sim.set_particle_tile_ordering(true).unwrap();
    sim.prepare_deterministic_p2g_with_budget(129 * 64).unwrap();
    sim.set_deterministic_p2g(true).unwrap();
    sim
}

fn seed(sim: &mut MpmSimulation, engine: &Engine, count: usize) {
    let positions: Vec<_> = (0..count)
        .map(|i| {
            [
                0.125 + (i % 4) as f32 * 0.125,
                0.5 + (i / 4 % 4) as f32 * 0.125,
                0.125 + (i / 16 % 4) as f32 * 0.125,
            ]
        })
        .collect();
    sim.set_particles_with_velocity(
        engine.queue().unwrap(),
        &positions,
        &vec![[0.01, -0.007, 0.002]; count],
    )
    .unwrap();
}

#[test]
#[ignore = "requires GPU; non-test library runtime gather lifecycle and exact repeats"]
fn fixed_order_public_lifecycle_repeatability_and_hot_resources() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for sparse in [false, true] {
        let mut a = solver(&engine, sparse, config());
        let mut b = solver(&engine, sparse, config());
        let scope = DeviceScope::start().unwrap();
        assert!(
            a.prepare_deterministic_p2g_with_budget(129 * 64 - 1)
                .is_err()
        );
        a.prepare_deterministic_p2g_with_budget(129 * 64).unwrap();
        assert_eq!(scope.finish().observations, 0);
        assert!(a.release_deterministic_p2g().is_err());
        assert!(a.set_active_grid(false).is_err());
        assert!(a.set_particle_tile_ordering(false).is_err());
        assert!(a.set_deferred_p2g_lookup(true).is_err());
        assert!(a.prepare_particle_tile_ordering(queue, 129).is_err());
        for count in [129, 1, 0, 129] {
            seed(&mut a, &engine, count);
            seed(&mut b, &engine, count);
            for sim in [&mut a, &mut b] {
                sim.step_n(&engine, 0.0001, 1).unwrap();
                let record = sim.prepare_step(0.0001, 2).unwrap();
                let before = device.get_internal_counters().hal;
                let encoder = device.create_command_encoder(&Default::default());
                let resources = DeviceScope::start().unwrap();
                let transfers = TransferScope::start().unwrap();
                let mut batch = sim.begin_command_batch(encoder);
                batch.encode_step(&record).unwrap();
                batch.submit(queue);
                assert_eq!(resources.finish().observations, 0);
                let transfers = transfers.finish();
                assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
                let after = device.get_internal_counters().hal;
                assert_eq!(before.bind_groups.read(), after.bind_groups.read());
                assert_eq!(
                    before.compute_pipelines.read(),
                    after.compute_pipelines.read()
                );
            }
            assert_eq!(
                a.checkpoint_soil_with(queue).unwrap().to_bytes(),
                b.checkpoint_soil_with(queue).unwrap().to_bytes()
            );
        }
        let old = a.prepare_step(0.0001, 1).unwrap();
        a.set_deterministic_p2g(false).unwrap();
        a.release_deterministic_p2g().unwrap();
        assert_eq!(a.deterministic_p2g_storage_bytes(), 0);
        assert!(a.set_deterministic_p2g(true).is_err());
        let mut batch = a.begin_command_batch(device.create_command_encoder(&Default::default()));
        assert!(batch.encode_step(&old).is_err());
        drop(batch);
        let resources = DeviceScope::start().unwrap();
        assert!(a.prepare_deterministic_p2g_with_budget(0).is_err());
        assert_eq!(resources.finish().observations, 0);
        a.prepare_deterministic_p2g().unwrap();
        a.prepare_particle_tile_ordering(queue, 129).unwrap();
        assert_eq!(a.deterministic_p2g_storage_bytes(), 0);
        a.prepare_deterministic_p2g_with_budget(129 * 64).unwrap();
        a.set_deterministic_p2g(true).unwrap();
        a.step_n(&engine, 0.0001, 1).unwrap();
        a.set_p2g_mode(true);
        assert!(!a.deterministic_p2g_enabled());
        println!(
            "runtime sparse={sparse}: 4 reseed sizes, exact history, zero hot owned resources/transfers, guarded replacement/release"
        );
    }
}

struct Directory(PathBuf);
impl Directory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let root = std::env::var_os("MOLLA_STORE_TEST_ROOT")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir);
        let path = root.join(format!(
            "fixed-order-runtime-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir(&path).unwrap();
        Self(path)
    }
}
impl Drop for Directory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn boundary(live: &mut MpmSoilDiskWorld, engine: &Engine) -> MpmSoilFrameCheckpoint {
    live.begin_region_update(&[]).unwrap();
    let until = Instant::now() + Duration::from_secs(10);
    let mut captured = None;
    loop {
        let ready = live.poll_region_update(engine).unwrap();
        if let Some((state, ids)) = live.captured_state() {
            assert_eq!(ids.len(), 129);
            captured = Some(
                MpmSoilFrameCheckpoint::new(state.clone(), live.world_origin().unwrap()).unwrap(),
            );
        }
        if ready {
            return captured.unwrap();
        }
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn complete(live: &mut MpmSoilDiskWorld, engine: &Engine) {
    let until = Instant::now() + Duration::from_secs(10);
    while !live.poll_region_update(engine).unwrap() {
        assert!(Instant::now() < until);
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
#[ignore = "requires GPU; non-test library canonical runtime storage/rebase/evolution repeatability"]
fn fixed_order_public_canonical_recenter_and_durable_evolution() {
    let engine = Engine::new_gpu().unwrap();
    let cfg = config();
    let mut seed_sim = solver(&engine, false, cfg);
    seed(&mut seed_sim, &engine, 129);
    let initial = MpmSoilFrameCheckpoint::new(
        seed_sim
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap(),
        [1e12; 3],
    )
    .unwrap();
    let archive = initial
        .partition_canonical_tiles(
            SoilTileLayout::new([0.0; 3], 1.0).unwrap(),
            &(0..129)
                .map(|i| SoilParticleId((i << 40) + 7))
                .collect::<Vec<_>>(),
            &[SoilMaterialId(1 << 50)],
        )
        .unwrap();
    let keys: Vec<_> = archive.tiles().collect();
    let limits = MpmSoilStoreLimits {
        particles: 129,
        tiles: 256,
    };
    let dirs = [Directory::new(), Directory::new()];
    let mut worlds: Vec<_> = dirs
        .iter()
        .map(|dir| {
            let mut store = MpmSoilCanonicalStore::open(&dir.0, limits).unwrap();
            store.save(&archive).unwrap();
            let mut live = MpmSoilDiskWorld::new_canonical(
                &engine,
                solver(&engine, true, cfg),
                store,
                [1e12; 3],
                SoilResidencyLimits {
                    regions: 2,
                    tiles: 256,
                    claims: 512,
                },
                MpmSoilStoreReadLimits {
                    tiles: 32,
                    particles: 129,
                    encoded_bytes: 32 * 1052 + 129 * 181,
                },
                129 * 160 + 16,
            )
            .unwrap();
            live.begin_region_update(&[SoilRegionRequest {
                id: SoilRegionId(1),
                tiles: &keys,
            }])
            .unwrap();
            complete(&mut live, &engine);
            live.begin_recenter([-3.0, -2.0, -2.0]).unwrap();
            complete(&mut live, &engine);
            assert_eq!(
                boundary(&mut live, &engine).to_bytes(),
                initial.recentered([-3.0, -2.0, -2.0]).unwrap().to_bytes()
            );
            live
        })
        .collect();
    let mut last = None;
    for step in 0..8 {
        for index in if step % 2 == 0 { [0, 1] } else { [1, 0] } {
            worlds[index].step_n(&engine, 0.0001, 1).unwrap();
        }
        let a = boundary(&mut worlds[0], &engine);
        let b = boundary(&mut worlds[1], &engine);
        assert_eq!(a.to_bytes(), b.to_bytes());
        assert_eq!(
            worlds[0].identity().particle_ids(),
            worlds[1].identity().particle_ids()
        );
        last = Some(a);
    }
    let last = last.unwrap();
    drop(worlds);
    for dir in &dirs {
        let mut store = MpmSoilCanonicalStore::open(&dir.0, limits).unwrap();
        let loaded = store.load().unwrap().unwrap();
        let all: Vec<_> = loaded.tiles().collect();
        let selected = loaded
            .select(
                &all,
                129,
                last.world_origin(),
                last.soil().config().grid_origin,
            )
            .unwrap();
        assert_eq!(
            selected
                .framed_selection()
                .to_frame_checkpoint()
                .unwrap()
                .to_bytes(),
            last.to_bytes()
        );
    }
    println!(
        "runtime canonical: 8 exact repeated evolved boundaries, preserved high IDs, exact rebase and durable reopen"
    );
}
