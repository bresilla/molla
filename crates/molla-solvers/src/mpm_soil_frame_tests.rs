use super::*;

fn soil(count: u32) -> MpmSoilCheckpoint {
    let mut result = super::super::tests::fixture(count);
    result.cfg.grid_size = 64;
    result.cfg.cell_size = 0.125;
    result.cfg.grid_origin = [1024.0, -2048.0, 4096.0];
    for (i, particle) in result.words[..count as usize * 4]
        .chunks_exact_mut(4)
        .enumerate()
    {
        for axis in 0..3 {
            particle[axis] =
                (result.cfg.grid_origin[axis] + 2.0 + (i % 17) as f32 / 64.0).to_bits();
        }
    }
    result
}

fn rehash(bytes: &mut [u8]) {
    let digest = blake3::hash(&bytes[HEADER_BYTES..]);
    bytes[12..HEADER_BYTES].copy_from_slice(digest.as_bytes());
}

#[test]
fn framed_particle_tiles_preserve_small_offsets_and_recentered_membership() {
    use molla_vehicle::{SoilTileCoord, SoilTileLayout};
    let mut state = soil(3);
    state.cfg.grid_origin = [-4.0; 3];
    for (i, p) in state.words[..12].chunks_exact_mut(4).enumerate() {
        for v in &mut p[..3] {
            *v = [-1e-6f32, 0.0, 1e-6][i].to_bits();
        }
    }
    let frame = MpmSoilFrameCheckpoint::new(state, [1e12; 3]).unwrap();
    let layout = SoilTileLayout::new([0.0; 3], 1.0).unwrap();
    let expected = vec![
        SoilTileCoord([999_999_999_999; 3]),
        SoilTileCoord([1_000_000_000_000; 3]),
        SoilTileCoord([1_000_000_000_000; 3]),
    ];
    assert_eq!(frame.particle_tiles(layout).unwrap(), expected);
    let restored = MpmSoilFrameCheckpoint::from_bytes(&frame.to_bytes(), 3).unwrap();
    assert_eq!(restored.particle_tiles(layout).unwrap(), expected);
    let frame = MpmSoilFrameCheckpoint::new(soil(129), [1e12, -1e12, 1e12]).unwrap();
    let before = frame.particle_tiles(layout).unwrap();
    let rebased = frame.recentered([-4.0; 3]).unwrap();
    assert_eq!(rebased.particle_tiles(layout).unwrap(), before);
    assert_eq!(
        MpmSoilFrameCheckpoint::new(soil(0), [1e12; 3])
            .unwrap()
            .particle_tiles(layout)
            .unwrap(),
        Vec::<SoilTileCoord>::new()
    );
    assert!(
        frame
            .particle_tiles(SoilTileLayout::new([0.0; 3], 1e-20).unwrap())
            .is_err()
    );
}

#[test]
fn framed_codec_is_bounded_lossless_and_covers_world_origin() {
    for count in [0, 1, 129] {
        let frame = MpmSoilFrameCheckpoint::new(soil(count), [1e12, -0.0, -1e12]).unwrap();
        let bytes = frame.to_bytes();
        let decoded = MpmSoilFrameCheckpoint::from_bytes(&bytes, count).unwrap();
        assert_eq!(decoded.to_bytes(), bytes);
        assert_eq!(decoded.soil().to_bytes(), frame.soil().to_bytes());
        assert_eq!(
            decoded.world_origin().map(f64::to_bits),
            frame.world_origin().map(f64::to_bits)
        );
        assert!(MpmSoilCheckpoint::from_bytes(&bytes, count).is_err());
        assert!(MpmSoilFrameCheckpoint::from_bytes(&frame.soil.to_bytes(), count).is_err());
        if count != 0 {
            assert!(MpmSoilFrameCheckpoint::from_bytes(&bytes, count - 1).is_err());
        }
    }
    let frame = MpmSoilFrameCheckpoint::new(soil(1), [1e12, -0.0, -1e12]).unwrap();
    let bytes = frame.to_bytes();
    for length in 0..bytes.len() {
        assert!(MpmSoilFrameCheckpoint::from_bytes(&bytes[..length], 129).is_err());
    }
    for index in 0..bytes.len() {
        let mut bad = bytes.clone();
        bad[index] ^= 1;
        assert!(MpmSoilFrameCheckpoint::from_bytes(&bad, 129).is_err());
    }
    for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut bad = bytes.clone();
        bad[HEADER_BYTES..HEADER_BYTES + 8].copy_from_slice(&value.to_le_bytes());
        rehash(&mut bad);
        assert!(MpmSoilFrameCheckpoint::from_bytes(&bad, 129).is_err());
        assert!(MpmSoilFrameCheckpoint::new(soil(1), [value; 3]).is_err());
    }
    let mut trailing = bytes.clone();
    trailing.extend_from_slice(&[0; 4]);
    rehash(&mut trailing);
    assert!(MpmSoilFrameCheckpoint::from_bytes(&trailing, 129).is_err());
    let mut inner = bytes;
    inner[FRAME_HEADER + 12] ^= 1;
    rehash(&mut inner);
    assert!(MpmSoilFrameCheckpoint::from_bytes(&inner, 129).is_err());
}

#[test]
fn frame_recenter_preserves_world_coordinates_and_rejects_offset_rounding() {
    let frame = MpmSoilFrameCheckpoint::new(soil(129), [1e12, -1e12, 1e12]).unwrap();
    let bytes = frame.to_bytes();
    let shifted = frame.recentered([-4.0; 3]).unwrap();
    assert_eq!(
        shifted.origin,
        [1e12 + 1028.0, -1e12 - 2044.0, 1e12 + 4100.0]
    );
    for (a, b) in frame
        .soil
        .stream(0)
        .chunks_exact(4)
        .zip(shifted.soil.stream(0).chunks_exact(4))
    {
        for axis in 0..3 {
            assert_eq!(
                frame.origin[axis] + f64::from(f32::from_bits(a[axis])),
                shifted.origin[axis] + f64::from(f32::from_bits(b[axis]))
            );
        }
    }
    assert_eq!(
        shifted
            .recentered(frame.soil.cfg.grid_origin)
            .unwrap()
            .to_bytes(),
        bytes
    );
    assert_eq!(frame.to_bytes(), bytes);
    let reference = [
        frame.origin[0] + 1024.0,
        frame.origin[1] - 2048.0,
        frame.origin[2] + 4096.0,
    ];
    let a = frame.momenta_about_world(reference).unwrap();
    let b = shifted.momenta_about_world(reference).unwrap();
    assert_eq!(a.origin, reference);
    assert_eq!(b.origin, reference);
    assert_eq!(a.mass, b.mass);
    assert_eq!(a.linear_momentum, b.linear_momentum);
    assert_eq!(a.angular_momentum, b.angular_momentum);
    assert_eq!(a.affine_angular_momentum, b.affine_angular_momentum);
    let a = frame.energy_at_world_height(reference[1]).unwrap();
    let b = shifted.energy_at_world_height(reference[1]).unwrap();
    assert_eq!(a.reference_height, reference[1]);
    assert_eq!(b.reference_height, reference[1]);
    assert_eq!(a.mechanical_energy(), b.mechanical_energy());
    assert!(frame.momenta_about_world([f64::NAN; 3]).is_err());
    assert!(frame.energy_at_world_height(f64::INFINITY).is_err());
    let huge = MpmSoilFrameCheckpoint::new(soil(1), [2f64.powi(60); 3]).unwrap();
    assert!(huge.recentered([-4.0; 3]).is_err());
    assert!(huge.momenta_about_world([1.0; 3]).is_err());
    assert!(huge.energy_at_world_height(1.0).is_err());
    let rounding = MpmSoilFrameCheckpoint::new(super::super::tests::fixture(1), [0.0; 3]).unwrap();
    assert!(rounding.recentered([16_777_216.0; 3]).is_err());
    let zero = MpmSoilFrameCheckpoint::new(soil(0), [-0.0; 3]).unwrap();
    assert_eq!(
        zero.recentered(zero.soil.cfg.grid_origin)
            .unwrap()
            .to_bytes(),
        zero.to_bytes()
    );
    fn send_sync<T: Send + Sync>() {}
    send_sync::<MpmSoilFrame>();
    send_sync::<MpmSoilFrameCheckpoint>();
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn live_frame_save_recenter_restore_and_resume_preserve_state() {
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let device = engine.device().unwrap();
    for mode in 0..4 {
        let state = soil(129);
        let mut simulation = super::super::tests::new_sim(&engine, state.cfg, mode);
        simulation.restore_soil_with(queue, &state).unwrap();
        let mut frame = MpmSoilFrame::new(simulation, [1e12, -1e12, 1e12]).unwrap();
        let initial = frame.checkpoint_with(queue).unwrap();
        let buffers: Vec<_> = frame
            .simulation
            .checkpoint_buffers()
            .into_iter()
            .cloned()
            .collect();
        let record = frame.simulation.prepare_step(0.0001, 1).unwrap();
        frame.recenter_with(queue, [-4.0; 3]).unwrap();
        let local = frame.checkpoint_with(queue).unwrap();
        assert_eq!(
            local.to_bytes(),
            initial.recentered([-4.0; 3]).unwrap().to_bytes()
        );
        let saved = MpmSoilFrameCheckpoint::from_bytes(&local.to_bytes(), 129).unwrap();
        frame.step_n_with(device, queue, 0.0001, 2).unwrap();
        assert_ne!(
            frame.checkpoint_with(queue).unwrap().to_bytes(),
            saved.to_bytes()
        );
        frame.restore_with(queue, &initial).unwrap();
        assert_eq!(
            frame.checkpoint_with(queue).unwrap().to_bytes(),
            initial.to_bytes()
        );
        frame.restore_with(queue, &saved).unwrap();
        assert_eq!(
            frame.checkpoint_with(queue).unwrap().to_bytes(),
            saved.to_bytes()
        );
        for (a, b) in buffers.iter().zip(frame.simulation.checkpoint_buffers()) {
            assert_eq!(a, b);
        }
        let mut batch = frame
            .simulation
            .begin_command_batch(device.create_command_encoder(&Default::default()));
        assert!(batch.encode_step(&record).is_err());
        drop(batch);
        let mut incompatible = saved.clone();
        incompatible.origin[0] += 64.0;
        incompatible.soil.cfg.gravity_y -= 1.0;
        let epoch = frame.simulation.step_epoch.clone();
        assert!(frame.restore_with(queue, &incompatible).is_err());
        assert!(Arc::ptr_eq(&epoch, &frame.simulation.step_epoch));
        assert_eq!(
            frame.checkpoint_with(queue).unwrap().to_bytes(),
            saved.to_bytes()
        );
        let (simulation, origin) = frame.into_parts();
        let mut reowned = MpmSoilFrame::new(simulation, origin).unwrap();
        assert_eq!(
            reowned.checkpoint_with(queue).unwrap().to_bytes(),
            saved.to_bytes()
        );
        println!(
            "mode={mode}: framed 129-particle save/recenter/restore, history and buffer identities, stale work and physical-config rejection"
        );
    }
    let state = soil(1);
    let mut simulation = super::super::tests::new_sim(&engine, state.cfg, 2);
    simulation.restore_soil_with(queue, &state).unwrap();
    let mut frame = MpmSoilFrame::new(simulation, [1e12; 3]).unwrap();
    frame.recenter_with(queue, [-4.0; 3]).unwrap();
    let saved = frame.checkpoint_with(queue).unwrap();
    frame.step_n_with(device, queue, 0.0001, 8).unwrap();
    let evolved = frame.checkpoint_with(queue).unwrap();
    drop(frame);
    let simulation = super::super::tests::new_sim(&engine, state.cfg, 2);
    let mut restored = MpmSoilFrame::new(simulation, [0.0; 3]).unwrap();
    restored.restore_with(queue, &saved).unwrap();
    restored.step_n_with(device, queue, 0.0001, 8).unwrap();
    assert_eq!(
        restored.checkpoint_with(queue).unwrap().to_bytes(),
        evolved.to_bytes()
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn rejected_world_offset_loses_no_state_and_performs_no_gpu_work() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let state = soil(1);
    let mut simulation = super::super::tests::new_sim(&engine, state.cfg, 2);
    simulation.restore_soil_with(queue, &state).unwrap();
    let mut frame = MpmSoilFrame::new(simulation, [2f64.powi(60); 3]).unwrap();
    let before = frame.checkpoint_with(queue).unwrap().to_bytes();
    let epoch = frame.simulation.step_epoch.clone();
    let resources = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    assert!(frame.recenter_with(queue, [-4.0; 3]).is_err());
    let traffic = transfers.finish();
    assert_eq!(traffic.upload_bytes, 0);
    assert_eq!(traffic.readback_bytes, 0);
    assert_eq!(resources.finish().observations, 0);
    assert!(Arc::ptr_eq(&epoch, &frame.simulation.step_epoch));
    assert_eq!(frame.checkpoint_with(queue).unwrap().to_bytes(), before);
    let mut rigid = super::super::tests::new_sim(&engine, state.cfg, 2);
    rigid
        .set_rigid_spheres(queue, &[([1026.0, -2046.0, 4098.0], 0.1, [0.0; 3])])
        .unwrap();
    assert!(MpmSoilFrame::new(rigid, [0.0; 3]).is_err());
    let mut guarded = super::super::tests::new_sim(&engine, state.cfg, 2);
    guarded.prepare_soil_motion_guard().unwrap();
    assert!(MpmSoilFrame::new(guarded, [0.0; 3]).is_err());
    let archive = state
        .partition_tiles(
            molla_vehicle::SoilTileLayout::new([0.0; 3], 8.0).unwrap(),
            &[molla_vehicle::SoilParticleId(9)],
            &[
                molla_vehicle::SoilMaterialId(1),
                molla_vehicle::SoilMaterialId(2),
                molla_vehicle::SoilMaterialId(3),
            ],
        )
        .unwrap();
    let selection = archive
        .select(&archive.tiles().collect::<Vec<_>>(), 1)
        .unwrap();
    let mut owned = super::super::tests::new_sim(&engine, state.cfg, 2);
    owned.restore_soil_selection(&engine, &selection).unwrap();
    assert!(MpmSoilFrame::new(owned, [0.0; 3]).is_err());
}
