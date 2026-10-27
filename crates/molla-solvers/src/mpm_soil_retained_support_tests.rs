use super::*;

#[test]
fn quiescence_retained_support_seed_and_decode_contract() {
    let ids = [
        SoilParticleId(u64::MAX),
        SoilParticleId(7),
        SoilParticleId(1 << 40),
    ];
    assert_eq!(seed_mask(&ids, &[ids[2], ids[0]]).unwrap(), [1, 0, 1]);
    assert_eq!(seed_mask(&ids, &[]).unwrap(), [0, 0, 0]);
    for bad in [vec![ids[0]; 2], vec![SoilParticleId(0)], vec![ids[0]; 4]] {
        assert!(seed_mask(&ids, &bad).is_err());
    }
    assert_eq!(
        decode(&[1, 65, 66, 97, 98]).unwrap(),
        [
            MpmSoilRetainedSupportParticle {
                quiet: true,
                requested: false,
                retain: false
            },
            MpmSoilRetainedSupportParticle {
                quiet: true,
                requested: false,
                retain: true
            },
            MpmSoilRetainedSupportParticle {
                quiet: false,
                requested: false,
                retain: true
            },
            MpmSoilRetainedSupportParticle {
                quiet: true,
                requested: true,
                retain: true
            },
            MpmSoilRetainedSupportParticle {
                quiet: false,
                requested: true,
                retain: true
            },
        ]
    );
    for bad in [0, 2, 4, 9, 18, 32, 33, 64, 67, 96, 99, 129, 193, u32::MAX] {
        assert!(decode(&[1, bad]).is_err());
    }
}

#[cfg(feature = "resource-counters")]
fn step(sim: &mut MpmSimulation, engine: &Engine) {
    let prepared = sim.prepare_step(0.0001, 2).unwrap();
    let mut batch = sim.begin_command_batch(
        engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default()),
    );
    batch.encode_step(&prepared).unwrap();
    batch.submit(engine.queue().unwrap());
}

#[cfg(feature = "resource-counters")]
pub(in super::super) fn identified(
    engine: &Engine,
    positions: &[[f32; 3]],
    mode: u32,
) -> (
    MpmSimulation,
    MpmSoilQuiescenceSnapshot,
    Vec<SoilParticleId>,
) {
    use molla_vehicle::{SoilMaterialId, SoilTileLayout};
    let cfg = MpmConfig {
        max_particles: 129,
        max_rigid: 1,
        grid_size: 64,
        grid_origin: [-3., -0.5, -3.],
        cell_size: 0.1,
        gravity_y: 0.,
        time_step: MpmTimeStep::Explicit,
        ..MpmConfig::sand()
    };
    let mut sim = if mode == 2 {
        MpmSimulation::new_sparse(engine, cfg, 128).unwrap()
    } else {
        MpmSimulation::new(engine, cfg).unwrap()
    };
    if mode == 1 {
        sim.set_active_grid(true).unwrap();
    }
    sim.set_particles(engine.queue().unwrap(), positions)
        .unwrap();
    step(&mut sim, engine);
    let checkpoint = sim.checkpoint_soil_with(engine.queue().unwrap()).unwrap();
    let ids: Vec<_> = (0..positions.len())
        .map(|i| {
            SoilParticleId(if i + 1 == positions.len() {
                u64::MAX
            } else {
                (1u64 << 40) + i as u64
            })
        })
        .collect();
    let materials: Vec<_> = (0..checkpoint.materials().len().max(1))
        .map(|i| SoilMaterialId(i as u64 + 1))
        .collect();
    let archive = checkpoint
        .partition_tiles(SoilTileLayout::new([0.; 3], 0.5).unwrap(), &ids, &materials)
        .unwrap();
    let selection = archive
        .select(&archive.tiles().collect::<Vec<_>>(), 129)
        .unwrap();
    sim.restore_soil_selection(engine, &selection).unwrap();
    assert_eq!(sim.soil_identity_snapshot().unwrap().particle_ids(), ids);
    sim.prepare_soil_motion_guard().unwrap();
    sim.set_soil_motion_bound(Some([0.01; 3]));
    sim.prepare_soil_quiescence(2080).unwrap();
    sim.prepare_soil_support(1049688).unwrap();
    step(&mut sim, engine);
    let snapshot = sim.read_soil_quiescence(engine, 129).unwrap();
    assert!(
        snapshot
            .particles()
            .iter()
            .all(|p| p.consecutive_steps == 2)
    );
    (sim, snapshot, ids)
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; identified retained quiet chains, storage reuse and lifecycle"]
fn quiescence_gpu_retained_support_seeds_and_lifecycle() {
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let positions = [
        [-1., 0.5, 0.],
        [-0.8, 0.5, 0.],
        [-0.6, 0.5, 0.],
        [1., 0.5, 0.],
    ];
    for mode in 0..3 {
        let (mut sim, snapshot, ids) = identified(&engine, &positions, mode);
        let before = sim
            .checkpoint_soil_with(engine.queue().unwrap())
            .unwrap()
            .to_bytes();
        assert!(
            sim.read_soil_support_closure(&engine, &snapshot, 2, 129, 2)
                .unwrap()
                .particles()
                .iter()
                .all(|p| !p.connected_to_nonquiet)
        );
        assert!(
            sim.read_soil_retained_support(&engine, &snapshot, 2, 129, 1, &ids[..1])
                .unwrap_err()
                .to_string()
                .contains("did not converge")
        );
        let allocations = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let dispatches = DispatchScope::start().unwrap();
        let seeded = sim
            .read_soil_retained_support(&engine, &snapshot, 2, 129, 2, &ids[..1])
            .unwrap();
        assert_eq!(allocations.finish().observations, 0);
        let transfers = transfers.finish();
        assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (80, 32));
        assert_eq!(dispatches.finish().encoded_dispatches, 6);
        assert_eq!(
            seeded
                .particles()
                .iter()
                .map(|p| p.retain)
                .collect::<Vec<_>>(),
            [true, true, true, false]
        );
        assert_eq!(
            seeded
                .particles()
                .iter()
                .map(|p| p.requested)
                .collect::<Vec<_>>(),
            [true, false, false, false]
        );
        assert_eq!(seeded.requested_particles(), &ids[..1]);
        assert_eq!((seeded.minimum_quiet_steps(), seeded.rounds()), (2, 2));
        assert!(seeded.particles().iter().all(|p| p.quiet));
        for rounds in [2, 3, 64] {
            assert_eq!(
                sim.read_soil_retained_support(&engine, &snapshot, 2, 129, rounds, &ids[..1])
                    .unwrap()
                    .particles(),
                seeded.particles()
            );
        }
        let isolated = sim
            .read_soil_retained_support(&engine, &snapshot, 2, 129, 1, &ids[3..])
            .unwrap();
        assert_eq!(
            isolated
                .particles()
                .iter()
                .map(|p| p.retain)
                .collect::<Vec<_>>(),
            [false, false, false, true]
        );
        let union = sim
            .read_soil_retained_support(&engine, &snapshot, 2, 129, 2, &[ids[3], ids[0]])
            .unwrap();
        assert!(union.particles().iter().all(|p| p.retain));
        let empty = sim
            .read_soil_retained_support(&engine, &snapshot, 2, 129, 2, &[])
            .unwrap();
        assert!(empty.particles().iter().all(|p| !p.retain && !p.requested));
        let nonquiet = sim
            .read_soil_retained_support(&engine, &snapshot, 3, 129, 1, &ids[3..])
            .unwrap();
        assert!(nonquiet.particles().iter().all(|p| !p.quiet && p.retain));
        let direct = sim.read_soil_support(&engine, &snapshot, 2, 129).unwrap();
        assert!(
            direct
                .particles()
                .iter()
                .all(|p| p.quiet && !p.touches_nonquiet)
        );
        assert!(
            sim.read_soil_support_closure(&engine, &snapshot, 2, 129, 2)
                .unwrap()
                .particles()
                .iter()
                .all(|p| !p.connected_to_nonquiet)
        );
        for bad in [vec![ids[0]; 2], vec![SoilParticleId(0)], vec![ids[0]; 5]] {
            let transfers = TransferScope::start().unwrap();
            let dispatches = DispatchScope::start().unwrap();
            assert!(
                sim.read_soil_retained_support(&engine, &snapshot, 2, 129, 2, &bad)
                    .is_err()
            );
            let transfers = transfers.finish();
            assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
            assert_eq!(dispatches.finish().encoded_dispatches, 0);
        }
        assert_eq!(
            sim.checkpoint_soil_with(engine.queue().unwrap())
                .unwrap()
                .to_bytes(),
            before
        );
        sim.validate_soil_quiescence(seeded.quiescence()).unwrap();
        sim.disable_soil_support();
        assert!(
            sim.read_soil_retained_support(&engine, &snapshot, 2, 129, 2, &ids[..1])
                .is_err()
        );
        sim.prepare_soil_support(1049688).unwrap();
        assert_eq!(
            sim.read_soil_retained_support(&engine, &snapshot, 2, 129, 2, &ids[..1])
                .unwrap()
                .particles(),
            seeded.particles()
        );
        sim.set_particles(engine.queue().unwrap(), &positions)
            .unwrap();
        assert!(
            sim.read_soil_retained_support(&engine, &snapshot, 2, 129, 2, &ids[..1])
                .is_err()
        );
        step(&mut sim, &engine);
        let unidentified = sim.read_soil_quiescence(&engine, 129).unwrap();
        assert!(unidentified.identity().is_none());
        assert!(
            sim.read_soil_retained_support(&engine, &unidentified, 2, 129, 2, &[])
                .unwrap_err()
                .to_string()
                .contains("global identities")
        );
        println!(
            "retained-support mode={mode} quiet_chain=3 isolated=1 stable_ids=true exact_physics=true dispatches=6 upload=80 readback=32"
        );
    }
    let (mut sim, snapshot, ids) = identified(&engine, &vec![[0., 0.5, 0.]; 129], 0);
    let result = sim
        .read_soil_retained_support(&engine, &snapshot, 2, 129, 1, &ids[128..])
        .unwrap();
    assert_eq!(result.particles().len(), 129);
    assert!(result.particles().iter().all(|p| p.quiet && p.retain));
    assert_eq!(result.particles().iter().filter(|p| p.requested).count(), 1);
    assert!(result.particles()[128].requested);
    println!("retained-support cross_workgroup=129 high_id_seed=true");
}
