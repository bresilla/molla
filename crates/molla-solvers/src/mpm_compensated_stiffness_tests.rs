use super::*;
use crate::mpm::replay_tests::read_words;

#[test]
fn retained_stiffness_shader_contract() {
    let source = compensated_stiffness::source().unwrap();
    assert!(!source.contains("stiffness_records"));
    assert!(!source.contains("GatherParticle"));
    assert!(
        source.contains("@group(2) @binding(6) var<storage, read_write> stiffness_global_tail")
    );
    assert_eq!(source.matches("fn clear_stiffness(").count(), 1);
    assert_eq!(source.matches("fn resolve_stiffness(").count(), 1);
    let original = include_str!("../shaders/mpm.wgsl");
    let body = |s: &str, name: &str| {
        let start = s.find(&format!("fn {name}(")).unwrap();
        let end = start + s[start..].find("\n}\n").unwrap() + 2;
        s[start..end].to_owned()
    };
    for name in [
        "quad_weights",
        "svd3_jacobi",
        "atomic_add_mass",
        "atomic_add_mom",
        "wg_atomic_add_mass",
        "wg_atomic_add_mx",
        "wg_atomic_add_my",
        "wg_atomic_add_mz",
    ] {
        assert_eq!(body(&source, name), body(original, name));
    }
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
#[ignore = "requires GPU and frozen 50k stiffness evidence"]
fn retained_stiffness_frozen_50k() {
    let input =
        std::path::PathBuf::from(std::env::var_os("SOIL_RETAINED_STIFFNESS_INPUT").unwrap());
    let output =
        std::path::PathBuf::from(std::env::var_os("SOIL_RETAINED_STIFFNESS_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = tests::seeded_benchmark(&engine);
    sim.set_p2g_mode(true);
    let frozen = std::fs::read(input.join("soil.bin")).unwrap();
    sim.restore_soil_with(
        queue,
        &MpmSoilCheckpoint::from_bytes(&frozen, 50_000).unwrap(),
    )
    .unwrap();
    sim.prepare_compensated_stiffness(131072).unwrap();
    sim.set_compensated_stiffness(true).unwrap();
    assert_eq!(sim.compensated_stiffness_storage_bytes(), 131072);
    let mut u = sim.step_uniforms(0.0001);
    u.grid_layout[2] = 0;
    queue.write_buffer(&sim.uniform_buf, 0, bytemuck::bytes_of(&u));
    let expected = std::fs::read(input.join("gather-2.bin")).unwrap();
    let reference: Vec<_> = std::fs::read(input.join("sum.bin"))
        .unwrap()
        .chunks_exact(8)
        .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
        .collect();
    let mut maximum = 0_f64;
    for round in 0..6 {
        let mut encoder = device.create_command_encoder(&Default::default());
        sim.initialize_active_grid(&mut encoder, false);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            sim.bind_step(&mut pass);
            pass.set_pipeline(&sim.active_pipelines[0]);
            pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
        }
        sim.active_tiles
            .encode_build(&mut encoder, sim.particle_count, 0);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            sim.bind_step(&mut pass);
            sim.encode_p2g(&mut pass);
        }
        queue.submit([encoder.finish()]);
        let grid = tests::world_grid(&engine, &sim);
        std::fs::write(
            output.join(format!("stiffness-{round}.bin")),
            bytemuck::cast_slice(&grid[2]),
        )
        .unwrap();
        assert_eq!(
            bytemuck::cast_slice::<u32, u8>(&grid[2]),
            expected,
            "uninstrumented stiffness identity"
        );
        assert_eq!(grid[2].len(), reference.len());
        for (&bits, &value) in grid[2].iter().zip(&reference) {
            let v = f32::from_bits(bits);
            assert!(v.is_finite());
            maximum = maximum.max((f64::from(v) - value).abs());
        }
    }
    assert_eq!(sim.checkpoint_soil_with(queue).unwrap().to_bytes(), frozen);
    assert!(maximum <= 2e-5);
    let report = serde_json::json!({"rounds":6,"stiffness_exact":true,"maximum_reference_error":maximum,"tail_bytes":sim.compensated_stiffness_storage_bytes(),"particle_checkpoint_exact":true,"dispatches_per_transfer":3,"diagnostic_contribution_buffers":0,"scope":"uninstrumented retained transfer on frozen 50k state; not whole-frame performance"});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("retained_stiffness={report}");
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; dense/active/compact retained stiffness lifecycle"]
fn retained_stiffness_lifecycle_and_evolution() {
    retained_transfer_lifecycle_and_evolution(false);
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU; all-field single/two-pass dense/active/compact lifecycle"]
fn retained_grid_transfer_lifecycle_and_evolution() {
    retained_transfer_lifecycle_and_evolution(true);
}

#[cfg(feature = "resource-counters")]
fn retained_transfer_lifecycle_and_evolution(all_fields: bool) {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let prepare = if all_fields {
        MpmSimulation::prepare_compensated_grid_transfer
    } else {
        MpmSimulation::prepare_compensated_stiffness
    };
    let enable = if all_fields {
        MpmSimulation::set_compensated_grid_transfer
    } else {
        MpmSimulation::set_compensated_stiffness
    };
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for variant in 0..if all_fields { 7 } else { 4 } {
        let two_pass = variant < 4;
        let mode = variant % 4;
        let mut sims = [false, true].map(|_| {
            let mut sim = tests::seeded(&engine, mode >= 2);
            sim.set_p2g_mode(two_pass);
            if mode == 0 {
                sim.set_active_grid(false).unwrap();
            }
            if mode == 3 {
                sim.set_deferred_p2g_lookup(true).unwrap();
            }
            if !two_pass && mode == 2 {
                assert!(sim.set_deferred_p2g_lookup(true).is_err());
            }
            sim
        });
        let positions = sims[0].read_positions_with(device, queue).unwrap();
        let bytes = sims[1].g_mass.size() * if all_fields { 5 } else { 1 };
        let scope = DeviceScope::start().unwrap();
        assert!(prepare(&mut sims[1], bytes - 1).is_err());
        assert_eq!(scope.finish().observations, 0);
        assert_eq!(sims[1].compensated_stiffness_storage_bytes(), 0);
        let stale = sims[1].prepare_step(0.0001, 1).unwrap();
        prepare(&mut sims[1], bytes).unwrap();
        enable(&mut sims[1], true).unwrap();
        {
            let mut batch =
                sims[1].begin_command_batch(device.create_command_encoder(&Default::default()));
            assert!(batch.encode_step(&stale).is_err());
        }
        assert!(sims[1].release_compensated_stiffness().is_err());
        assert!(sims[1].set_active_grid(mode == 0).is_err());
        assert!(sims[1].set_deferred_p2g_lookup(mode != 3).is_err());
        for count in [129, 1, 0, 129] {
            for sim in &mut sims {
                sim.set_particles_with_velocity(
                    queue,
                    &positions[..count],
                    &vec![[0.02, -0.01, 0.005]; count],
                )
                .unwrap();
                let record = sim.prepare_step(0.0001, 2).unwrap();
                let encoder = device.create_command_encoder(&Default::default());
                let before = device.get_internal_counters().hal;
                let resources = DeviceScope::start().unwrap();
                let transfers = TransferScope::start().unwrap();
                let mut batch = sim.begin_command_batch(encoder);
                batch.encode_step(&record).unwrap();
                batch.submit(queue);
                assert_eq!(resources.finish().observations, 0);
                let transfers = transfers.finish();
                assert_eq!(transfers.upload_bytes, 0);
                assert_eq!(transfers.readback_bytes, 0);
                let after = device.get_internal_counters().hal;
                assert_eq!(before.bind_groups.read(), after.bind_groups.read());
                assert_eq!(
                    before.compute_pipelines.read(),
                    after.compute_pipelines.read()
                );
            }
            let mut worst = 0_f32;
            for (field, (a, b)) in sims[0]
                .checkpoint_buffers()
                .into_iter()
                .zip(sims[1].checkpoint_buffers())
                .enumerate()
            {
                let a = read_words(device, queue, a);
                let b = read_words(device, queue, b);
                if field == 7 {
                    assert_eq!(a, b);
                    continue;
                }
                for (&a, &b) in a.iter().zip(&b) {
                    let a = f32::from_bits(a);
                    let b = f32::from_bits(b);
                    assert!(a.is_finite() && b.is_finite());
                    worst = worst.max((a - b).abs());
                }
            }
            println!(
                "retained_transfer_evolution all_fields={all_fields} two_pass={two_pass} mode={mode} count={count} max_absolute={worst} hot_buffers=0 uploads=0 readbacks=0"
            );
            assert!(worst <= 2e-5);
        }
        if mode >= 2 {
            assert!(sims[1].reserve_sparse_tiles(256).is_err());
            enable(&mut sims[1], false).unwrap();
            sims[1].reserve_sparse_tiles(256).unwrap();
        } else {
            enable(&mut sims[1], false).unwrap();
            sims[1].set_active_grid(mode == 0).unwrap();
        }
        assert!(enable(&mut sims[1], true).is_err());
        let bytes = sims[1].g_mass.size() * if all_fields { 5 } else { 1 };
        prepare(&mut sims[1], bytes).unwrap();
        enable(&mut sims[1], true).unwrap();
        sims[1].set_p2g_mode(!two_pass);
        assert!(!sims[1].compensated_stiffness_enabled());
        assert!(enable(&mut sims[1], true).is_err());
        sims[1].release_compensated_stiffness().unwrap();
        assert_eq!(sims[1].compensated_stiffness_storage_bytes(), 0);
    }
    let mut native = MpmSimulation::new_native_explicit(&engine, MpmConfig::sand()).unwrap();
    let resources = DeviceScope::start().unwrap();
    assert!(prepare(&mut native, u64::MAX).is_err());
    assert!(enable(&mut native, true).is_err());
    assert_eq!(resources.finish().observations, 0);
}
