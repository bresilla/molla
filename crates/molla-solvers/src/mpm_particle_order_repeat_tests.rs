use super::*;

const COUNT: usize = 129;
const STRIDES: [usize; 9] = [4, 4, 12, 12, 1, 1, 1, 1, 4];

fn snapshot(device: &wgpu::Device, queue: &wgpu::Queue, sim: &MpmSimulation) -> Vec<u32> {
    let packed = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("particle-order-replay"),
        size: (COUNT * 40 * 4) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    let mut offset = 0;
    for (buffer, stride) in sim.checkpoint_buffers().into_iter().zip(STRIDES) {
        let bytes = (COUNT * stride * 4) as u64;
        encoder.copy_buffer_to_buffer(buffer, 0, &packed, offset, bytes);
        offset += bytes;
    }
    queue.submit([encoder.finish()]);
    read_words(device, queue, &packed)
}

fn errors(a: &[u32], b: &[u32]) -> [f32; 2] {
    assert_eq!(a.len(), COUNT * 40);
    assert_eq!(a.len(), b.len());
    std::array::from_fn(|field| {
        let start = field * COUNT * 4;
        (start..start + COUNT * 4)
            .map(|i| {
                let (a, b) = (f32::from_bits(a[i]), f32::from_bits(b[i]));
                assert!(a.is_finite() && b.is_finite());
                (a - b).abs()
            })
            .fold(0.0, f32::max)
    })
}

#[test]
fn ordering_replay_error_contract() {
    let a = vec![0; COUNT * 40];
    let mut b = a.clone();
    b[COUNT * 4 + 7] = 3e-5_f32.to_bits();
    assert_eq!(errors(&a, &b), [0., 3e-5]);
    b[3] = 1e-5_f32.to_bits();
    assert_eq!(errors(&a, &b), [1e-5, 3e-5]);
}

#[test]
#[ignore = "bounded GPU diagnostic; collects all comparisons before asserting parity"]
fn ordering_matched_inputs_and_trajectory_repeats() {
    let output = std::path::PathBuf::from(
        std::env::var_os("MOLLA_ORDER_REPLAY_ARTIFACTS").expect("artifact directory required"),
    );
    std::fs::create_dir(&output).expect("artifact directory must be new");
    let engine = Engine::new_gpu().unwrap();
    println!("adapter={:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = config(64);
    cfg.youngs = 0.;
    cfg.gravity_y = -0.5;
    let positions = points(COUNT);
    let velocities = vec![[4., 0., 0.]; COUNT];
    let local_transfer = match std::env::var("MOLLA_ORDER_LOCAL_TRANSFER").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        Err(std::env::VarError::NotPresent) => true,
        _ => panic!("MOLLA_ORDER_LOCAL_TRANSFER must be absent, 0 or 1"),
    };
    let names = [
        "unordered-a",
        "unordered-b",
        "reference-a",
        "reference-b",
        "parallel-a",
        "parallel-b",
    ];
    let mut sims: Vec<_> = (0..6)
        .map(|mode| {
            let mut sim = MpmSimulation::new_sparse(&engine, cfg, 1).unwrap();
            sim.set_particles_with_velocity(queue, &positions, &velocities)
                .unwrap();
            if mode >= 4 {
                sim.prepare_parallel_particle_tile_ordering(queue, COUNT as u32)
                    .unwrap();
            } else if mode >= 2 {
                sim.prepare_particle_tile_ordering(queue, COUNT as u32)
                    .unwrap();
            }
            if mode >= 2 {
                sim.set_particle_tile_ordering(true).unwrap();
            }
            sim.step_n_with(device, queue, 0.0001, 2).unwrap();
            let status = sim.read_active_grid_status_with(device, queue).unwrap();
            assert!(status.overflow);
            assert_eq!(status.completed_substeps, 0);
            sim.reserve_sparse_tiles(128).unwrap();
            if !local_transfer {
                sim.install_world_transfer_coordinates();
            }
            sim
        })
        .collect();
    let initial = snapshot(device, queue, &sims[0]);
    for sim in &sims[1..] {
        assert_eq!(initial, snapshot(device, queue, sim));
    }
    println!(
        "{{\"record\":\"manifest\",\"particles\":129,\"steps\":128,\"dt\":0.001953125,\"grid_size\":64,\"tile_capacity\":128,\"youngs\":0,\"gravity_y\":-0.5,\"tolerance\":0.00002,\"local_transfer_coordinates\":{local_transfer},\"all_modes_overflow_then_grow\":true,\"scope\":\"single-step observed diagnostic; not batched equivalence or timing\"}}"
    );
    let pairs = [(0, 1), (2, 3), (4, 5), (0, 2), (0, 4), (2, 4)];
    let mut total_failures = 0;
    for matched in [false, true] {
        let phase = if matched {
            "matched-input"
        } else {
            "trajectory"
        };
        if matched {
            for sim in &mut sims {
                sim.set_particles_with_velocity(queue, &positions, &velocities)
                    .unwrap();
                sim.reset_active_grid();
            }
        }
        let mut first_bits = [0; 6];
        let mut first_failed = [0; 6];
        let mut maxima = [[0_f32; 2]; 6];
        for step in 1..=128 {
            if matched {
                let mut encoder = device.create_command_encoder(&Default::default());
                for sim in &sims[1..] {
                    for ((src, dst), stride) in sims[0]
                        .checkpoint_buffers()
                        .into_iter()
                        .zip(sim.checkpoint_buffers())
                        .zip(STRIDES)
                    {
                        encoder.copy_buffer_to_buffer(src, 0, dst, 0, (COUNT * stride * 4) as u64);
                    }
                }
                queue.submit([encoder.finish()]);
                let input = snapshot(device, queue, &sims[0]);
                for sim in &sims[1..] {
                    assert_eq!(input, snapshot(device, queue, sim));
                }
            }
            let input = snapshot(device, queue, &sims[0]);
            for offset in 0..6 {
                let mode = (step + offset) % 6;
                sims[mode]
                    .step_n_with(device, queue, 0.001953125, 1)
                    .unwrap();
            }
            let states: Vec<_> = sims
                .iter()
                .map(|sim| snapshot(device, queue, sim))
                .collect();
            let mut capture = step == 128;
            for (index, &(a, b)) in pairs.iter().enumerate() {
                let error = errors(&states[a], &states[b]);
                if states[a] != states[b] && first_bits[index] == 0 {
                    first_bits[index] = step;
                    capture = true;
                }
                let passed = error.into_iter().all(|e| e <= 2e-5);
                if !passed {
                    total_failures += 1;
                    if first_failed[index] == 0 {
                        first_failed[index] = step;
                        capture = true;
                    }
                }
                for axis in 0..2 {
                    maxima[index][axis] = maxima[index][axis].max(error[axis]);
                }
                println!(
                    "{{\"record\":\"comparison\",\"phase\":\"{phase}\",\"step\":{step},\"a\":\"{}\",\"b\":\"{}\",\"position_error\":{},\"velocity_error\":{},\"passed\":{passed}}}",
                    names[a], names[b], error[0], error[1]
                );
            }
            for sim in &sims {
                let status = sim.read_active_grid_status_with(device, queue).unwrap();
                assert!(!status.overflow);
                assert_eq!(status.completed_substeps, step as u32);
            }
            if capture {
                std::fs::write(
                    output.join(format!("{phase}-{step}-input.bin")),
                    bytemuck::cast_slice(&input),
                )
                .unwrap();
                for (name, state) in names.iter().zip(&states) {
                    std::fs::write(
                        output.join(format!("{phase}-{step}-{name}.bin")),
                        bytemuck::cast_slice(state),
                    )
                    .unwrap();
                }
            }
        }
        for (i, &(a, b)) in pairs.iter().enumerate() {
            println!(
                "{{\"record\":\"summary\",\"phase\":\"{phase}\",\"a\":\"{}\",\"b\":\"{}\",\"first_word_difference\":{},\"first_tolerance_failure\":{},\"max_position_error\":{},\"max_velocity_error\":{}}}",
                names[a], names[b], first_bits[i], first_failed[i], maxima[i][0], maxima[i][1]
            );
        }
    }
    assert_eq!(
        total_failures, 0,
        "collected all matched and evolving comparisons; no ordering promotion on failure"
    );
}
