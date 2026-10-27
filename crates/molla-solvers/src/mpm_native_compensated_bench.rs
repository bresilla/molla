use super::*;

#[test]
#[ignore = "requires native/f64 GPU and fresh SOIL_PRECONTACT_OUTPUT; whole-step proxy, not tractor acceptance"]
fn native_precontact_whole_step_cost() {
    use super::super::replay_tests::read_words;
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_PRECONTACT_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::SHADER_FLOAT32_ATOMIC | wgpu::Features::SHADER_F64,
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let wait = || {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
    };
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 64;
    cfg.grid_origin = [-2.0; 3];
    cfg.cell_size = 0.0625;
    cfg.max_particles = 50_000;
    cfg.max_rigid = 1;
    cfg.youngs = 1000.0;
    cfg.particle_mass = 0.002;
    cfg.particle_volume = 0.000002;
    let points: Vec<_> = (0..50_000)
        .map(|i| {
            [
                -0.3 + (i % 50) as f32 * 0.0125,
                0.05 + (i / 50 % 20) as f32 * 0.0125,
                -0.3 + (i / 1000) as f32 * 0.0125,
            ]
        })
        .collect();
    let dt = 1.0 / (60.0 * 34.0);
    let mut samples = Vec::new();
    for layout in ["dense", "active", "compact"] {
        let mut pair = [false, true].map(|compensated| {
            let mut sim = if layout == "compact" {
                MpmSimulation::new_native_sparse(&engine, cfg, 256).unwrap()
            } else {
                MpmSimulation::new_native_explicit(&engine, cfg).unwrap()
            };
            if layout == "active" {
                sim.set_active_grid(true).unwrap();
            }
            sim.set_particles(queue, &points).unwrap();
            sim.set_rigid_spheres(queue, &[([0.0, -0.05, 0.0], 0.1, [0.0; 3])])
                .unwrap();
            if compensated {
                sim.install_compensated_active_transfer();
            }
            sim
        });
        let initial = pair[0]
            .checkpoint_buffers()
            .map(|buffer| read_words(device, queue, buffer));
        for (field, (words, buffer)) in initial.iter().zip(pair[1].checkpoint_buffers()).enumerate()
        {
            assert_eq!(words, &read_words(device, queue, buffer));
            std::fs::write(
                output.join(format!("{layout}-input-{field}.bin")),
                bytemuck::cast_slice(words),
            )
            .unwrap();
        }
        let records = pair.each_ref().map(|sim| sim.prepare_step(dt, 34).unwrap());
        for round in 0..7 {
            for mode in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
                let sim = &mut pair[mode];
                for (buffer, words) in sim.checkpoint_buffers().into_iter().zip(&initial) {
                    queue.write_buffer(buffer, 0, bytemuck::cast_slice(words));
                }
                sim.reset_active_grid();
                sim.step(&engine, dt).unwrap();
                wait();
                let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
                let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
                let started = std::time::Instant::now();
                let encoder = device.create_command_encoder(&Default::default());
                let mut batch = sim.begin_command_batch(encoder);
                batch.encode_step(&records[mode]).unwrap();
                batch.submit(queue);
                wait();
                let wall_ms = started.elapsed().as_secs_f64() * 1000.0;
                assert_eq!(allocations.finish().observations, 0);
                let traffic = traffic.finish();
                assert_eq!(traffic.upload_bytes, 0);
                assert_eq!(traffic.readback_bytes, 0);
                let status = sim.read_active_grid_status_with(device, queue).unwrap();
                assert!(!status.faulted());
                assert_eq!(status.completed_substeps, 35);
                for (field, buffer) in sim.checkpoint_buffers().into_iter().enumerate() {
                    for word in read_words(device, queue, buffer) {
                        if field == 7 {
                            assert_eq!(word, 0);
                        } else {
                            assert!(f32::from_bits(word).is_finite());
                        }
                    }
                }
                assert!(
                    read_words(device, queue, &sim.rigid_wrench)
                        .iter()
                        .all(|&word| f32::from_bits(word).is_finite())
                );
                let residual_bytes = sim
                    .compensated_active_transfer
                    .as_ref()
                    .map_or(0, |transfer| transfer.error.size());
                samples.push(serde_json::json!({
                    "layout":layout, "compensated":mode == 1, "round":round,
                    "warmup":round == 0, "wall_ms":wall_ms, "particles":50_000,
                    "substeps":34, "dt":dt, "residual_bytes":residual_bytes,
                    "completed_substeps":status.completed_substeps,
                    "hot_buffer_allocations":0, "hot_upload_bytes":0, "hot_readback_bytes":0
                }));
                std::fs::write(
                    output.join("samples.json"),
                    serde_json::to_vec_pretty(&samples).unwrap(),
                )
                .unwrap();
                eprintln!(
                    "precontact-cost layout={layout} compensated={} round={round} wall_ms={wall_ms:.6} residual_bytes={residual_bytes}",
                    mode == 1
                );
            }
        }
    }
    assert_eq!(samples.len(), 42);
}
