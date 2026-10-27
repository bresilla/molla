use super::*;
use molla_vehicle::{SoilMaterialId, SoilParticleId, SoilTileLayout};

fn layout() -> SoilTileLayout {
    SoilTileLayout::new([0.0; 3], 1.0).unwrap()
}

fn selection(count: u32, tag: u64) -> MpmSoilFrameSelection {
    let mut state = super::super::tests::fixture(count);
    state.cfg.grid_size = 64;
    state.cfg.grid_origin = [-4.0; 3];
    state.cfg.cell_size = 0.125;
    for (i, p) in state.words[..count as usize * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        for v in &mut p[..3] {
            *v = (2.0 + (i % 17) as f32 / 64.0).to_bits();
        }
    }
    let frame = MpmSoilFrameCheckpoint::new(state, [1e12; 3]).unwrap();
    let ids: Vec<_> = (0..count)
        .map(|i| SoilParticleId((u64::from(i) << 40) | tag))
        .collect();
    let materials = [
        SoilMaterialId(u64::MAX),
        SoilMaterialId(0),
        SoilMaterialId(tag),
    ];
    let archive = frame.partition_tiles(layout(), &ids, &materials).unwrap();
    archive
        .select(&archive.tiles().collect::<Vec<_>>(), count)
        .unwrap()
}

fn assert_device_ids(engine: &Engine, frame: &MpmSoilFrame, expected: &MpmSoilFrameSelection) {
    let sim = frame.simulation();
    let buffers = sim.soil_identity_buffers().unwrap();
    let snapshot = sim.soil_identity_snapshot().unwrap();
    assert_eq!(snapshot.particle_ids(), expected.particle_ids());
    assert_eq!(snapshot.material_ids(), expected.material_ids());
    for (buffer, ids) in [
        (
            buffers.particles,
            expected
                .particle_ids()
                .iter()
                .map(|i| i.0)
                .collect::<Vec<_>>(),
        ),
        (
            buffers.materials,
            expected
                .material_ids()
                .iter()
                .map(|i| i.0)
                .collect::<Vec<_>>(),
        ),
    ] {
        let words = crate::mpm::replay_tests::read_words(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            buffer,
        );
        let expected: Vec<u32> = ids
            .into_iter()
            .flat_map(|id| [id as u32, (id >> 32) as u32])
            .collect();
        assert_eq!(&words[..expected.len()], expected);
    }
}

#[test]
#[ignore = "requires GPU; frame and global IDs restore/recenter/capture together"]
fn framed_identity_restore_recenter_capture_reload_and_reuse() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..4 {
        let selected = selection(129, 17);
        let mut cfg = selected.soil().config();
        cfg.grid_origin = [-8.0; 3];
        let sim = super::super::tests::new_sim(&engine, cfg, mode);
        let mut live = MpmSoilFrame::new(sim, [0.0; 3]).unwrap();
        assert!(live.checkpoint_tiles_with(queue, layout()).is_err());
        let allocations = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        live.restore_selection(&engine, &selected).unwrap();
        assert_eq!(allocations.finish().observations, 4);
        let transfer = transfers.finish();
        assert_eq!(
            (transfer.upload_bytes, transfer.readback_bytes),
            (129 * 168 + 3 * 8, 0)
        );
        assert_eq!(live.world_origin(), selected.world_origin());
        assert_eq!(
            live.checkpoint_with(queue).unwrap().to_bytes(),
            selected.to_frame_checkpoint().unwrap().to_bytes()
        );
        assert_device_ids(&engine, &live, &selected);
        let old = live.simulation.soil_identity_snapshot().unwrap().clone();
        let view = live.simulation.soil_identity_buffers().unwrap();
        let buffers = [view.particles.clone(), view.materials.clone()];
        let original_archive = live.checkpoint_tiles_with(queue, layout()).unwrap();
        let prepared = live.simulation.prepare_step(0.0001, 1).unwrap();
        live.recenter_with(queue, [-8.0; 3]).unwrap();
        assert!(live.simulation.is_soil_identity_current(&old));
        assert_eq!(live.world_origin(), [1e12 + 4.0; 3]);
        let view = live.simulation.soil_identity_buffers().unwrap();
        assert_eq!((&buffers[0], &buffers[1]), (view.particles, view.materials));
        assert_device_ids(&engine, &live, &selected);
        let rebased = live.checkpoint_tiles_with(queue, layout()).unwrap();
        assert_eq!(
            rebased.to_bytes(),
            original_archive.recentered([-8.0; 3]).unwrap().to_bytes()
        );
        {
            let mut batch = live
                .simulation
                .begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&prepared).is_err());
        }
        live.recenter_with(queue, [-8.0; 3]).unwrap();
        assert!(live.simulation.is_soil_identity_current(&old));
        live.step_n_with(device, queue, 0.0001, 2).unwrap();
        assert!(live.simulation.is_soil_identity_current(&old));
        let evolved = live.checkpoint_tiles_with(queue, layout()).unwrap();
        let loaded = MpmSoilFrameTileArchive::from_bytes(&evolved.to_bytes(), 129).unwrap();
        let loaded_selection = loaded
            .select(&loaded.tiles().collect::<Vec<_>>(), 129)
            .unwrap();
        let sim = super::super::tests::new_sim(&engine, cfg, mode);
        let mut restored = MpmSoilFrame::new(sim, [0.0; 3]).unwrap();
        restored
            .restore_selection(&engine, &loaded_selection)
            .unwrap();
        assert_device_ids(&engine, &restored, &selected);
        assert_eq!(
            restored
                .checkpoint_tiles_with(queue, layout())
                .unwrap()
                .to_bytes(),
            evolved.to_bytes()
        );
        restored.step_n_with(device, queue, 0.0001, 2).unwrap();
        restored
            .checkpoint_with(queue)
            .unwrap()
            .soil()
            .validate()
            .unwrap();
        let changed = selection(129, 23);
        let allocations = DeviceScope::start().unwrap();
        live.restore_selection(&engine, &changed).unwrap();
        assert_eq!(allocations.finish().observations, 2);
        assert!(!live.simulation.is_soil_identity_current(&old));
        let view = live.simulation.soil_identity_buffers().unwrap();
        assert_eq!((&buffers[0], &buffers[1]), (view.particles, view.materials));
        assert_device_ids(&engine, &live, &changed);
        live.restore_with(queue, &changed.to_frame_checkpoint().unwrap())
            .unwrap();
        assert!(live.simulation.soil_identity_snapshot().is_none());
        assert!(live.checkpoint_tiles_with(queue, layout()).is_err());
    }
}

#[test]
#[ignore = "requires GPU; invalid identified frame operations preserve prior authority"]
fn framed_identity_invalid_restore_and_recenter_preserve_state() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let selected = selection(3, 17);
    let mut cfg = selected.soil().config();
    cfg.max_particles = 3;
    let sim = MpmSimulation::new(&engine, cfg).unwrap();
    let mut live = MpmSoilFrame::new(sim, [0.0; 3]).unwrap();
    live.restore_selection(&engine, &selected).unwrap();
    let before = live
        .checkpoint_tiles_with(queue, layout())
        .unwrap()
        .to_bytes();
    let identity = live.simulation.soil_identity_snapshot().unwrap().clone();
    let oversized = selection(4, 17);
    let allocations = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    assert!(live.restore_selection(&engine, &oversized).is_err());
    assert_eq!(allocations.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
    assert!(live.recenter_with(queue, [16_777_216.0; 3]).is_err());
    assert!(live.recenter_with(queue, [f32::NAN; 3]).is_err());
    assert!(live.simulation.is_soil_identity_current(&identity));
    assert_device_ids(&engine, &live, &selected);
    assert_eq!(
        live.checkpoint_tiles_with(queue, layout())
            .unwrap()
            .to_bytes(),
        before
    );
    let foreign = Engine::new_gpu().unwrap();
    assert!(live.restore_selection(&foreign, &selected).is_err());
    assert_eq!(
        live.checkpoint_tiles_with(queue, layout())
            .unwrap()
            .to_bytes(),
        before
    );
    let mut changed = selected.soil().clone();
    changed.cfg.cell_size *= 2.0;
    let other = MpmSoilFrameCheckpoint::new(changed, selected.world_origin())
        .unwrap()
        .partition_tiles(layout(), selected.particle_ids(), selected.material_ids())
        .unwrap();
    let other = other.select(&other.tiles().collect::<Vec<_>>(), 3).unwrap();
    assert!(live.restore_selection(&engine, &other).is_err());
    assert_eq!(
        live.checkpoint_tiles_with(queue, layout())
            .unwrap()
            .to_bytes(),
        before
    );
    let empty = selection(0, 23);
    live.restore_selection(&engine, &empty).unwrap();
    assert_device_ids(&engine, &live, &empty);
    let archive = live.checkpoint_tiles_with(queue, layout()).unwrap();
    assert_eq!(archive.particle_count(), 0);
    assert_eq!(archive.material_ids(), empty.material_ids());
}
