use super::*;
use replay_tests::read_words;

fn dispatch(sim: &MpmSimulation, device: &wgpu::Device, queue: &wgpu::Queue, plastic: bool) {
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        sim.bind_step(&mut pass);
        pass.set_pipeline(if plastic {
            &sim.pipe_plastic
        } else {
            &sim.pipe_g2p
        });
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn nonfinite_state_latches_without_repair_and_restore_recovers() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cases = 0;
    for mode in 0..3 {
        let cfg = MpmConfig {
            max_particles: 1,
            max_rigid: 1,
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
        sim.set_particles(queue, &[[0.0, 0.3, 0.0]]).unwrap();
        sim.step_n_with(device, queue, 0.0001, 1).unwrap();
        let clean = sim.checkpoint_soil_with(queue).unwrap();
        for plastic in [false, true] {
            for bits in [
                f32::NAN.to_bits(),
                f32::INFINITY.to_bits(),
                f32::NEG_INFINITY.to_bits(),
            ] {
                for column in 0..3 {
                    for row in 0..3 {
                        sim.restore_soil_with(queue, &clean).unwrap();
                        sim.step_n_with(device, queue, 0.0001, 1).unwrap();
                        queue.write_buffer(
                            &sim.p_f,
                            (column * 4 + row) * 4,
                            bytemuck::bytes_of(&bits),
                        );
                        let before: Vec<_> = sim
                            .checkpoint_buffers()
                            .iter()
                            .map(|b| read_words(device, queue, b))
                            .collect();
                        dispatch(&sim, device, queue, plastic);
                        let status = sim.read_active_grid_status_with(device, queue).unwrap();
                        assert!(
                            status.numerical_fault && status.faulted() && !status.overflow,
                            "mode={mode} plastic={plastic} column={column} row={row} bits={bits:x}"
                        );
                        assert!(sim.checkpoint_soil_with(queue).is_err());
                        sim.step_n_with(device, queue, 0.0001, 3).unwrap();
                        let after: Vec<_> = sim
                            .checkpoint_buffers()
                            .iter()
                            .map(|b| read_words(device, queue, b))
                            .collect();
                        assert_eq!(before, after, "faulted state must not be silently repaired");
                        assert!(
                            sim.read_active_grid_status_with(device, queue)
                                .unwrap()
                                .numerical_fault
                        );
                        cases += 1;
                    }
                }
            }
        }
        sim.restore_soil_with(queue, &clean).unwrap();
        sim.step_n_with(device, queue, 0.0001, 1).unwrap();
        assert!(
            !sim.read_active_grid_status_with(device, queue)
                .unwrap()
                .faulted()
        );
        sim.checkpoint_soil_with(queue).unwrap();
    }
    assert_eq!(cases, 162);
    println!(
        "{cases} dense/active/sparse gather and plasticity matrix faults; exact stream preservation, latch, capture rejection, restore recovery"
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn gather_overflow_cannot_be_hidden_by_boundary_clamp() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let cfg = MpmConfig {
        max_particles: 1,
        max_rigid: 1,
        ..MpmConfig::sand()
    };
    let mut sim = MpmSimulation::new(&engine, cfg).unwrap();
    sim.set_particles(queue, &[[0.0, 0.3, 0.0]]).unwrap();
    sim.step_n_with(device, queue, 0.0001, 1).unwrap();
    let before = sim.checkpoint_soil_with(queue).unwrap();
    let grid = vec![[f32::INFINITY, 0.0, 0.0, 1.0]; cfg.grid_size.pow(3) as usize];
    queue.write_buffer(&sim.g_velocity, 0, bytemuck::cast_slice(&grid));
    let particles: Vec<_> = sim
        .checkpoint_buffers()
        .iter()
        .map(|b| read_words(device, queue, b))
        .collect();
    dispatch(&sim, device, queue, false);
    assert!(
        sim.read_active_grid_status_with(device, queue)
            .unwrap()
            .numerical_fault
    );
    assert!(sim.checkpoint_soil_with(queue).is_err());
    let after: Vec<_> = sim
        .checkpoint_buffers()
        .iter()
        .map(|b| read_words(device, queue, b))
        .collect();
    assert_eq!(particles, after);
    sim.restore_soil_with(queue, &before).unwrap();
    sim.step_n_with(device, queue, 0.0001, 1).unwrap();
    assert!(
        !sim.read_active_grid_status_with(device, queue)
            .unwrap()
            .faulted()
    );
}
