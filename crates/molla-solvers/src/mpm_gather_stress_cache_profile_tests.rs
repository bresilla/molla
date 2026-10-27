use super::super::tests as transfer_tests;
use super::*;
use crate::mpm::replay_tests::read_words;

#[test]
#[ignore = "requires timestamp GPU and frozen 50k fixture; not tractor acceptance"]
fn gather_stress_cache_profile_50k() {
    profile_50k(
        |sim| {
            sim.prepare_deterministic_p2g_stress_cache(8_000_000, false, false)
                .unwrap();
        },
        "stress-cache",
    );
}

pub(crate) fn profile_50k(prepare: fn(&mut MpmSimulation), label: &str) {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let input = std::path::PathBuf::from(std::env::var_os("SOIL_CACHE_INPUT").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_CACHE_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let initial = MpmSoilCheckpoint::from_bytes(&std::fs::read(input).unwrap(), 50_000).unwrap();
    assert_eq!(initial.particle_count(), 50_000);
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
    )
    .unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut sims = [false, true].map(|cached| {
        let mut sim = MpmSimulation::new_sparse(&engine, initial.config(), 512).unwrap();
        sim.restore_soil_with(queue, &initial).unwrap();
        sim.prepare_parallel_particle_tile_ordering(queue, 50_000)
            .unwrap();
        sim.set_particle_tile_ordering(true).unwrap();
        if cached {
            prepare(&mut sim);
        } else {
            sim.prepare_deterministic_p2g_with_budget(3_200_000)
                .unwrap();
        }
        sim.set_deterministic_p2g(true).unwrap();
        sim.prepare_step_profiling(1).unwrap();
        sim
    });
    let mut records = Vec::new();
    for step in 0..12 {
        let mut checkpoints = [Vec::new(), Vec::new()];
        for arm in if step % 2 == 0 { [0, 1] } else { [1, 0] } {
            let allocations = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            let begin = std::time::Instant::now();
            sims[arm]
                .step_n_profiled_with(device, queue, 0.0001, 1)
                .unwrap();
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let wall_ms = begin.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(allocations.finish().observations, 0);
            let transfers = transfers.finish();
            assert_eq!(transfers.readback_bytes, 0);
            let profile = sims[arm].read_step_profile().unwrap().unwrap();
            checkpoints[arm] = sims[arm].checkpoint_soil_with(queue).unwrap().to_bytes();
            let stages: Vec<_> = profile
                .stages
                .iter()
                .map(|s| serde_json::json!({"name":s.stage.name(),"ms":s.elapsed_ms}))
                .collect();
            records.push(serde_json::json!({"lane":"evolving", "step":step,"arm":arm,"warmup":step<2,"wall_ms":wall_ms,"stages":stages,"checkpoint_blake3":blake3::hash(&checkpoints[arm]).to_hex().to_string(),"hot_upload_bytes":transfers.upload_bytes,"hot_readback_bytes":transfers.readback_bytes,"hot_allocations":0,"dt":0.0001_f32}));
        }
        assert_eq!(checkpoints[0], checkpoints[1], "evolving step {step}");
        std::fs::write(
            output.join(format!("evolving-{step}.json")),
            serde_json::to_vec_pretty(&records[records.len() - 2..]).unwrap(),
        )
        .unwrap();
    }
    let frozen = sims[0].checkpoint_soil_with(queue).unwrap().to_bytes();
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("stress cache transfer pairs"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let times = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut grid_reference = None;
    for pair in 0..12 {
        for arm in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
            let allocations = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.write_timestamp(&queries, 0);
            transfer_tests::encode_transfer(&sims[arm], &mut encoder, true);
            encoder.write_timestamp(&queries, 1);
            encoder.resolve_query_set(&queries, 0..2, &times, 0);
            queue.submit([encoder.finish()]);
            assert_eq!(allocations.finish().observations, 0);
            let transfers = transfers.finish();
            assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
            let ticks = read_words(device, queue, &times);
            let tick = |i| u64::from(ticks[i]) | (u64::from(ticks[i + 1]) << 32);
            let ms = (tick(2) - tick(0)) as f64 * f64::from(queue.get_timestamp_period()) / 1e6;
            let grid = transfer_tests::world_grid(&engine, &sims[arm]);
            if let Some(reference) = &grid_reference {
                assert_eq!(&grid, reference, "pair={pair} arm={arm}");
            } else {
                grid_reference = Some(grid);
            }
            records.push(serde_json::json!({"lane":"held_transfer","pair":pair,"arm":arm,"warmup":pair<2,"transfer_gpu_ms":ms,"hot_allocations":0,"hot_upload_bytes":0,"hot_readback_bytes":0}));
        }
        std::fs::write(
            output.join(format!("held-{pair}.json")),
            serde_json::to_vec_pretty(&records[records.len() - 2..]).unwrap(),
        )
        .unwrap();
    }
    for sim in &mut sims {
        assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), frozen);
    }
    std::fs::write(output.join("final.soil"), frozen).unwrap();
    std::fs::write(
        output.join("records.json"),
        serde_json::to_vec_pretty(&records).unwrap(),
    )
    .unwrap();
    println!(
        "{label} particles=50000 evolving_exact_boundaries=12 held_exact_pairs=12 hot_allocations=0 hot_readbacks=0"
    );
}
