use super::*;
use molla_core::gpu_metrics::{DeviceScope, TransferScope};

#[test]
#[ignore = "requires GPU and frozen50k input; evolving grid transfer with prescribed sphere"]
fn compensated_grid_50k_contact_evolution() {
    let input = std::path::PathBuf::from(std::env::var_os("SOIL_CORRECTED_INPUT").unwrap());
    let output = std::path::PathBuf::from(std::env::var_os("SOIL_CORRECTED_OUTPUT").unwrap());
    std::fs::create_dir(&output).unwrap();
    let checkpoint =
        MpmSoilCheckpoint::from_bytes(&std::fs::read(input.join("soil.bin")).unwrap(), 50_000)
            .unwrap();
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sims = [0, 1, 2].map(|arm| {
        let mut sim = tests::seeded_benchmark(&engine);
        sim.restore_soil_with(queue, &checkpoint).unwrap();
        if arm == 0 {
            sim.prepare_deterministic_p2g_with_compensated_stiffness(3_200_000)
                .unwrap();
            sim.set_deterministic_p2g(true).unwrap();
        } else {
            sim.set_p2g_mode(true);
            sim.set_particle_tile_ordering(false).unwrap();
            sim.prepare_compensated_grid_transfer(sim.g_mass.size() * 5)
                .unwrap();
            sim.set_compensated_grid_transfer(true).unwrap();
        }
        sim.set_rigid_spheres(queue, &[([-0.15, 0.52, -0.15], 0.12, [0.01, -0.005, 0.0])])
            .unwrap();
        if std::env::var_os("MOLLA_COMPENSATED_WRENCH").is_some() {
            sim.prepare_compensated_rigid_wrench(sim.rigid_wrench.size())
                .unwrap();
            sim.set_compensated_rigid_wrench(true).unwrap();
        }
        assert_eq!(sim.particle_count, 50_000);
        sim
    });
    let mut observations = Vec::new();
    let mut maxima = [[0_f64; 10]; 2];
    let mut changed = [0_usize; 10];
    for step in 0..12 {
        let mut states: [Option<Vec<Vec<u32>>>; 3] = Default::default();
        for arm in if step % 2 == 0 { [0, 1, 2] } else { [2, 1, 0] } {
            let sim = &mut sims[arm];
            let record = sim.prepare_step(0.0001, 1).unwrap();
            let encoder = device.create_command_encoder(&Default::default());
            let resources = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            let mut batch = sim.begin_command_batch(encoder);
            batch.encode_step(&record).unwrap();
            batch.submit(queue);
            assert_eq!(resources.finish().observations, 0);
            let transfers = transfers.finish();
            assert_eq!(transfers.upload_bytes, 0);
            assert_eq!(transfers.readback_bytes, 0);
            let mut state: Vec<_> = sim
                .checkpoint_buffers()
                .into_iter()
                .map(|buffer| read_words(device, queue, buffer))
                .collect();
            state.push(read_words(device, queue, &sim.rigid_wrench));
            assert_eq!(state.len(), 10);
            let saved = sim.checkpoint_soil_with(queue).unwrap();
            assert_eq!(saved.particle_count(), 50_000);
            let bytes = saved.to_bytes();
            let hash = blake3::hash(&bytes).to_hex().to_string();
            if step == 11 {
                std::fs::write(output.join(format!("final-{arm}.soil")), &bytes).unwrap();
            }
            std::fs::write(
                output.join(format!("wrench-{step}-{arm}.bin")),
                bytemuck::cast_slice(&state[9]),
            )
            .unwrap();
            observations.push(serde_json::json!({"step":step,"arm":arm,"checkpoint_hash":hash}));
            states[arm] = Some(state);
        }
        let mut errors = [[0_f64; 10]; 2];
        for (comparison, (a, b)) in [(1, 0), (1, 2)].into_iter().enumerate() {
            for field in 0..10 {
                let left = &states[a].as_ref().unwrap()[field];
                let right = &states[b].as_ref().unwrap()[field];
                assert_eq!(left.len(), right.len());
                for (&a, &b) in left.iter().zip(right) {
                    if comparison == 1 {
                        changed[field] += usize::from(a != b);
                    }
                    if field == 7 {
                        assert_eq!(a, b);
                        continue;
                    }
                    let a = f64::from(f32::from_bits(a));
                    let b = f64::from(f32::from_bits(b));
                    assert!(a.is_finite() && b.is_finite());
                    errors[comparison][field] = errors[comparison][field].max((a - b).abs());
                }
                maxima[comparison][field] =
                    maxima[comparison][field].max(errors[comparison][field]);
            }
        }
        observations.push(serde_json::json!({"step":step,"candidate_gather_errors":errors[0],"candidate_repeat_errors":errors[1]}));
    }
    let passed = maxima.iter().flatten().all(|&v| v <= 2e-5);
    let report = serde_json::json!({"particles":50_000,"steps":12,"dt":0.0001,"prescribed_spheres":1,"bound":2e-5,"passed":passed,"maxima":maxima,"repeat_changed_words":changed,"observations":observations,"hot_buffers":0,"uploads":0,"readbacks_in_encode":0,"scope":"50k detached soil with prescribed sphere, corrected gather versus two independent compensated scatter arms; not articulated tractor, accuracy calibration or timing acceptance"});
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!(
        "compensated_grid_50k_evolution maxima={maxima:?} repeat_changed={changed:?} passed={passed}"
    );
    assert!(passed);
}
