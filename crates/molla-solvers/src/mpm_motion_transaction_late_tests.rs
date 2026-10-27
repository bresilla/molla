use super::*;

#[test]
#[ignore = "requires GPU; failure on the second substep preserves the first publication"]
fn motion_guard_gpu_later_failure_preserves_last_committed_substep() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..3 {
        let cfg = MpmConfig {
            max_particles: 129,
            max_rigid: 1,
            grid_size: 64,
            cell_size: 0.1,
            grid_origin: [-3.0, -0.5, -3.0],
            gravity_y: -9.81,
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
        sim.step_n(&engine, 0.001, 1).unwrap();
        let first = sim.checkpoint_soil_with(queue).unwrap();
        let read = |sim: &MpmSimulation| {
            sim.checkpoint_buffers()
                .iter()
                .map(|buffer| read_words(device, queue, buffer))
                .collect::<Vec<_>>()
        };
        let expected = read(&sim);
        let first_y = f32::from_bits(expected[0][1]);
        sim.step_n(&engine, 0.001, 1).unwrap();
        let second_y = f32::from_bits(read_words(device, queue, &sim.p_pos)[1]);
        let bound = 0.000015;
        assert!((f64::from(first_y) - f64::from(0.3f32)).abs() < bound);
        assert!((f64::from(second_y) - f64::from(first_y)).abs() > bound);
        sim.restore_soil_with(queue, &initial).unwrap();
        sim.prepare_soil_motion_guard().unwrap();
        sim.set_soil_motion_bound(Some([1.0, bound, 1.0]));
        let step = sim.prepare_step(0.001, 3).unwrap();
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        batch.encode_step(&step).unwrap();
        batch.submit(queue);
        let status = sim.read_active_grid_status_with(device, queue).unwrap();
        assert!(status.motion_bound_exceeded && !status.numerical_fault && !status.overflow);
        assert_eq!(read(&sim), expected);
        if let Some(path) = std::env::var_os("MOLLA_MOTION_TRANSACTION_OUTPUT") {
            let path = std::path::PathBuf::from(path);
            std::fs::create_dir_all(&path).unwrap();
            std::fs::write(
                path.join(format!("mode-{mode}-late-initial.soil")),
                initial.to_bytes(),
            )
            .unwrap();
            std::fs::write(
                path.join(format!("mode-{mode}-late-first.soil")),
                first.to_bytes(),
            )
            .unwrap();
            let actual: Vec<u32> = read(&sim).into_iter().flatten().collect();
            std::fs::write(
                path.join(format!("mode-{mode}-late-rejected.words")),
                bytemuck::cast_slice(&actual),
            )
            .unwrap();
        }
        println!(
            "motion_transaction_late mode={mode} particles=129 second_substep_rejected=true first_substep_exact=true"
        );
    }
}
