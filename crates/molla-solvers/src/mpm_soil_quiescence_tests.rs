use super::*;
use crate::mpm::replay_tests::read_words;

#[path = "mpm_soil_quiescence_probe_tests.rs"]
mod probes;

#[test]
fn quiescence_layout_budget_and_shader_validation() {
    let limits = wgpu::Limits::default();
    assert_eq!(sizes(129, &limits, 2080).unwrap(), [1032, 1048]);
    assert!(sizes(129, &limits, 2079).is_err());
    assert!(sizes(u32::MAX, &limits, u64::MAX).is_err());
    let module = naga::front::wgsl::parse_str(&source()).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; per-substep exact-rest streaks, reset and guarded publication"]
fn quiescence_gpu_streaks_dt_cancel_restore_and_fault() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..3 {
        let cfg = MpmConfig {
            max_particles: 129,
            max_rigid: 1,
            grid_size: 64,
            grid_origin: [-3.0, -0.5, -3.0],
            cell_size: 0.1,
            gravity_y: 0.0,
            time_step: MpmTimeStep::Explicit,
            ..MpmConfig::sand()
        };
        let mut sim = if mode == 2 {
            MpmSimulation::new_sparse(&engine, cfg, 128).unwrap()
        } else {
            MpmSimulation::new(&engine, cfg).unwrap()
        };
        if mode == 1 {
            sim.set_active_grid(true).unwrap();
        }
        sim.set_particles(queue, &vec![[0.0, 0.3, 0.0]; 129])
            .unwrap();
        let initial = sim.checkpoint_soil_with(queue).unwrap();
        assert!(sim.prepare_soil_quiescence(u64::MAX).is_err());
        sim.prepare_soil_motion_guard().unwrap();
        sim.set_soil_motion_bound(Some([0.0; 3]));
        let old = sim.prepare_step(0.0001, 1).unwrap();
        let resources = DeviceScope::start().unwrap();
        assert!(sim.prepare_soil_quiescence(2079).is_err());
        assert_eq!(resources.finish().observations, 0);
        sim.prepare_soil_quiescence(2080).unwrap();
        assert_eq!(sim.soil_quiescence_bytes(), 2080);
        assert!(sim.read_soil_quiescence(&engine, 129).is_err());
        let step = sim.prepare_step(0.0001, 3).unwrap();
        {
            let mut batch =
                sim.begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&old).is_err());
            batch.encode_step(&step).unwrap();
        }
        assert!(sim.quiescence_needs_reset());
        assert!(sim.read_soil_quiescence(&engine, 129).is_err());
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        batch.encode_step(&step).unwrap();
        batch.submit(queue);
        assert_eq!(resources.finish().observations, 0);
        assert_eq!(transfers.finish().readback_bytes, 0);
        let snapshot = sim.read_soil_quiescence(&engine, 129).unwrap();
        sim.validate_soil_quiescence(&snapshot).unwrap();
        sim.validate_soil_quiescence(&snapshot.clone()).unwrap();
        assert!(snapshot.identity.is_none());
        assert_eq!(
            snapshot.particles(),
            vec![
                MpmSoilQuiescence {
                    consecutive_steps: 3,
                    dt: 0.0001
                };
                129
            ]
        );
        assert_eq!(
            sim.checkpoint_soil_with(queue).unwrap().to_bytes(),
            initial.to_bytes()
        );
        let transfers = TransferScope::start().unwrap();
        assert!(sim.read_soil_quiescence(&engine, 128).is_err());
        assert_eq!(transfers.finish().readback_bytes, 0);
        sim.step_n(&engine, 0.0001, 0).unwrap();
        sim.validate_soil_quiescence(&snapshot).unwrap();
        {
            let mut batch =
                sim.begin_command_batch(device.create_command_encoder(&Default::default()));
            batch.encode_step(&step).unwrap();
        }
        assert!(sim.validate_soil_quiescence(&snapshot).is_err());
        let cancelled = sim.read_soil_quiescence(&engine, 129).unwrap();
        assert_eq!(cancelled.particles(), snapshot.particles());
        sim.validate_soil_quiescence(&cancelled).unwrap();
        sim.step_n(&engine, 0.0001, 2).unwrap();
        assert!(sim.validate_soil_quiescence(&cancelled).is_err());
        assert!(
            sim.read_soil_quiescence(&engine, 129)
                .unwrap()
                .particles
                .iter()
                .all(|p| p.consecutive_steps == 5)
        );
        sim.step_n(&engine, 0.0002, 1).unwrap();
        assert!(
            sim.read_soil_quiescence(&engine, 129)
                .unwrap()
                .particles
                .iter()
                .all(|p| p.consecutive_steps == 1 && p.dt == 0.0002)
        );
        let before_restore = sim.read_soil_quiescence(&engine, 129).unwrap();
        sim.restore_soil_with(queue, &initial).unwrap();
        assert!(sim.validate_soil_quiescence(&before_restore).is_err());
        assert!(sim.read_soil_quiescence(&engine, 129).is_err());
        sim.step_n(&engine, 0.0001, 2).unwrap();
        assert!(
            sim.read_soil_quiescence(&engine, 129)
                .unwrap()
                .particles
                .iter()
                .all(|p| p.consecutive_steps == 2)
        );
        let observer = sim
            .motion_guard
            .as_ref()
            .unwrap()
            .quiescence
            .as_ref()
            .unwrap();
        let before = read_words(device, queue, &observer.records);
        let before_fault = sim.read_soil_quiescence(&engine, 129).unwrap();
        queue.write_buffer(
            &sim.p_vel,
            128 * 16,
            bytemuck::cast_slice(&[1.0_f32, 0.0, 0.0, 0.0]),
        );
        sim.step_n(&engine, 0.0001, 1).unwrap();
        assert!(sim.validate_soil_quiescence(&before_fault).is_err());
        assert!(sim.read_soil_quiescence(&engine, 129).is_err());
        let observer = sim
            .motion_guard
            .as_ref()
            .unwrap()
            .quiescence
            .as_ref()
            .unwrap();
        assert_eq!(read_words(device, queue, &observer.records), before);
        assert!(before.iter().any(|&word| word != 0));
        sim.restore_soil_with(queue, &initial).unwrap();
        sim.set_soil_motion_bound(Some([0.01; 3]));
        sim.step_n(&engine, 0.0001, 1).unwrap();
        assert!(
            sim.read_soil_quiescence(&engine, 129)
                .unwrap()
                .particles
                .iter()
                .all(|p| p.consecutive_steps == 1)
        );
        let stale = sim.prepare_step(0.0001, 1).unwrap();
        let before_external = sim.read_soil_quiescence(&engine, 129).unwrap();
        {
            let mut batch =
                sim.begin_command_batch(device.create_command_encoder(&Default::default()));
            let _ = batch.encoder_mut();
        }
        assert!(sim.validate_soil_quiescence(&before_external).is_err());
        assert!(sim.read_soil_quiescence(&engine, 129).is_err());
        sim.step_n(&engine, 0.0001, 1).unwrap();
        assert!(sim.validate_soil_quiescence(&before_external).is_err());
        assert!(
            sim.read_soil_quiescence(&engine, 129)
                .unwrap()
                .particles()
                .iter()
                .all(|p| p.consecutive_steps == 1)
        );
        sim.motion_guard
            .as_mut()
            .unwrap()
            .quiescence
            .as_mut()
            .unwrap()
            .attempt = Some(u64::MAX);
        let exhausted = sim.read_soil_quiescence(&engine, 129).unwrap();
        sim.validate_soil_quiescence(&exhausted).unwrap();
        sim.step_n(&engine, 0.0001, 1).unwrap();
        assert!(sim.validate_soil_quiescence(&exhausted).is_err());
        assert!(sim.read_soil_quiescence(&engine, 129).is_err());
        sim.disable_soil_quiescence();
        assert_eq!(sim.soil_quiescence_bytes(), 0);
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        assert!(batch.encode_step(&stale).is_err());
        drop(batch);
        println!(
            "quiescence mode={mode} particles=129 streaks=3,5,1 resets_exact=true cancelled_batch_preserved=true retained_bytes=2080"
        );
    }
}
