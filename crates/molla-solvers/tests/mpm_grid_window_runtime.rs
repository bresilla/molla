#![cfg(all(feature = "gpu", feature = "resource-counters"))]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, TransferScope};
use molla_solvers::mpm::*;
use molla_vehicle::{
    SoilMaterialId, SoilParticleId, SoilRegionId, SoilRegionRequest, SoilResidencyLimits,
    SoilTileLayout,
};
use std::{
    fs,
    path::PathBuf,
    time::{Duration, Instant},
};

fn poll(live: &mut MpmSoilDiskWorld, engine: &Engine) {
    let deadline = Instant::now() + Duration::from_secs(10);
    while !live.poll_region_update(engine).unwrap() {
        assert!(Instant::now() < deadline);
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
#[ignore = "requires GPU; public canonical physical window persistence and stepping"]
fn grid_window_public_runtime() {
    run_window_public(false);
}

#[test]
#[ignore = "requires GPU; exact repeats and original physical tolerance for public windows"]
fn grid_window_public_repeat_physical() {
    run_window_public(true);
}

fn physical_errors(a: &[u8], b: &[u8]) -> [f64; 9] {
    let a_state = MpmSoilCheckpoint::from_bytes(a, 65).unwrap();
    let b_state = MpmSoilCheckpoint::from_bytes(b, 65).unwrap();
    assert_eq!(a_state.config(), b_state.config());
    assert_eq!(a_state.materials(), b_state.materials());
    assert_eq!(a_state.particle_count(), b_state.particle_count());
    let mut offset = a.len() - a_state.particle_bytes();
    assert_eq!(offset, b.len() - b_state.particle_bytes());
    assert_eq!(&a[44..offset], &b[44..offset]);
    let mut errors = [0.0_f64; 9];
    for (stream, stride) in [4, 4, 12, 12, 1, 1, 1, 1, 4].into_iter().enumerate() {
        for _ in 0..stride * a_state.particle_count() {
            let x = u32::from_le_bytes(a[offset..offset + 4].try_into().unwrap());
            let y = u32::from_le_bytes(b[offset..offset + 4].try_into().unwrap());
            if stream == 7 {
                assert_eq!(x, y);
            } else {
                let x = f64::from(f32::from_bits(x));
                let y = f64::from(f32::from_bits(y));
                assert!(x.is_finite() && y.is_finite());
                let scale = if stream == 0 {
                    1.0
                } else {
                    x.abs().max(y.abs()).max(1.0)
                };
                errors[stream] = errors[stream].max((x - y).abs() / scale);
            }
            offset += 4;
        }
    }
    assert_eq!(offset, a.len());
    errors
}

fn run_window_public(repeat_physical: bool) {
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let root = std::env::var_os("MOLLA_STORE_TEST_ROOT")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    let limits = MpmSoilStoreLimits {
        particles: 65,
        tiles: 32,
    };
    let io = MpmSoilStoreReadLimits {
        particles: 65,
        tiles: 32,
        encoded_bytes: 32 * 1052 + 65 * 181,
    };
    let output = root.join(format!("window-public-observations-{}", std::process::id()));
    if repeat_physical {
        fs::create_dir(&output).unwrap();
    }
    let mut reports = Vec::new();
    let mut accepted = true;
    for (sparse, implicit) in [false, true].into_iter().flat_map(|sparse| {
        [false, true]
            .into_iter()
            .map(move |implicit| (sparse, implicit))
    }) {
        let mut results = vec![Vec::new(); if repeat_physical { 4 } else { 2 }];
        let arms = if repeat_physical {
            vec![0, 2, 1, 3]
        } else {
            vec![0, 1]
        };
        for arm in arms {
            let moved = if repeat_physical { arm >= 2 } else { arm == 1 };
            let path = root.join(format!(
                "window-public-{}-{sparse}-{implicit}-arm-{arm}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            let mut cfg = MpmConfig::sand();
            cfg.grid_size = 64;
            cfg.grid_origin = [-4.0; 3];
            cfg.cell_size = 0.125;
            cfg.max_particles = 65;
            cfg.max_rigid = 1;
            cfg.youngs = 1000.0;
            cfg.time_step = if implicit {
                MpmTimeStep::Implicit { jacobi_iters: 2 }
            } else {
                MpmTimeStep::Explicit
            };
            let mut sim = if sparse {
                MpmSimulation::new_sparse(&engine, cfg, 128).unwrap()
            } else {
                MpmSimulation::new(&engine, cfg).unwrap()
            };
            sim.set_active_grid(true).unwrap();
            let points: Vec<_> = (0..65)
                .map(|i| [0.125 + (i % 4) as f32 * 0.125, 2.0, 2.0])
                .collect();
            sim.set_particles_with_velocity(queue, &points, &vec![[0.01, -0.01, 0.0]; 65])
                .unwrap();
            let checkpoint = sim.checkpoint_soil_with(queue).unwrap();
            let shifted = checkpoint.with_grid_window([-3.875, -4.0, -4.0]).unwrap();
            assert_eq!(
                shifted
                    .with_grid_window(cfg.grid_origin)
                    .unwrap()
                    .to_bytes(),
                checkpoint.to_bytes()
            );
            let frame = MpmSoilFrameCheckpoint::new(checkpoint, [1e12; 3]).unwrap();
            let archive = frame
                .partition_canonical_tiles(
                    SoilTileLayout::new([0.0; 3], 1.0).unwrap(),
                    &(0..65).map(SoilParticleId).collect::<Vec<_>>(),
                    &[SoilMaterialId(0)],
                )
                .unwrap();
            let keys: Vec<_> = archive.tiles().collect();
            let mut store = MpmSoilCanonicalStore::open(&path, limits).unwrap();
            store.save(&archive).unwrap();
            sim.set_particles_with_velocity(queue, &[], &[]).unwrap();
            sim.prepare_parallel_particle_tile_ordering(queue, 65)
                .unwrap();
            sim.set_particle_tile_ordering(true).unwrap();
            sim.prepare_deterministic_p2g_with_compensated_stiffness(65 * 64)
                .unwrap();
            sim.set_deterministic_p2g(true).unwrap();
            let mut live = MpmSoilDiskWorld::new_canonical(
                &engine,
                sim,
                store,
                [1e12; 3],
                SoilResidencyLimits {
                    regions: 2,
                    tiles: 64,
                    claims: 128,
                },
                io,
                65 * 160 + 16,
            )
            .unwrap();
            live.begin_region_update(&[SoilRegionRequest {
                id: SoilRegionId(1),
                tiles: &keys,
            }])
            .unwrap();
            poll(&mut live, &engine);
            let identity = live.identity().particle_ids().to_vec();
            if moved {
                let resources = DeviceScope::start().unwrap();
                let transfers = TransferScope::start().unwrap();
                live.begin_grid_window_move([-3.875, -4.0, -4.0]).unwrap();
                assert!(live.step_n(&engine, 0.0001, 1).is_err());
                poll(&mut live, &engine);
                assert_eq!(resources.finish().observations, 0);
                assert_eq!(transfers.finish().upload_bytes, 0);
                assert_eq!(live.last_transfer_stats().upload_bytes, 0);
                assert_eq!(live.world_origin(), Some([1e12; 3]));
                assert_eq!(live.identity().particle_ids(), identity);
            }
            live.step_n(&engine, 0.0001, 4).unwrap();
            live.begin_region_update(&[SoilRegionRequest {
                id: SoilRegionId(1),
                tiles: &keys,
            }])
            .unwrap();
            poll(&mut live, &engine);
            assert_eq!(live.active_particle_count(), 65);
            drop(live);
            let mut store = MpmSoilCanonicalStore::open(&path, limits).unwrap();
            let saved = store.load().unwrap().unwrap();
            let keys = saved.tiles().collect::<Vec<_>>();
            let selected = saved.select(&keys, 65, [1e12; 3], cfg.grid_origin).unwrap();
            results[arm] = selected.framed_selection().soil().to_bytes();
            if repeat_physical {
                fs::write(
                    output.join(format!(
                        "sparse-{sparse}-implicit-{implicit}-arm-{arm}.checkpoint"
                    )),
                    &results[arm],
                )
                .unwrap();
            }
            assert_eq!(selected.framed_selection().particle_ids(), identity);
            println!(
                "public_grid_window sparse={sparse} implicit={implicit} moved={moved} arm={arm} particles=65 persisted=true"
            );
            drop(store);
            fs::remove_dir_all(&path).unwrap();
        }
        if repeat_physical {
            let repeats = [results[0] == results[1], results[2] == results[3]];
            let errors = physical_errors(&results[0], &results[2]);
            let physical = errors.iter().all(|&x| x <= 1e-6);
            accepted &= repeats.iter().all(|&x| x) && physical;
            reports.push(serde_json::json!({"sparse":sparse,"implicit":implicit,"exact_repeats":repeats,"physical_errors":errors,"physical_passed":physical}));
        } else {
            assert_eq!(results[0], results[1]);
        }
    }
    if repeat_physical {
        let report = serde_json::json!({"accepted":accepted,"particles":65,"substeps":4,"dt":0.0001,"tolerance":1e-6,"self_repeat":"exact for shifted and unshifted owners","records":reports,"scope":"non-test public API, persisted terminal state; no per-step attribution or long traversal"});
        fs::write(
            output.join("report.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        println!("window_public_repeat_report={report}");
        assert!(accepted);
    }
}
