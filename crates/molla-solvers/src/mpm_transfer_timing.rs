use super::*;
use crate::mpm::replay_tests::read_words;

fn complete(device: &wgpu::Device) {
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires native atomics, timestamps and retained 50k input"]
fn bounded_centered_transfer_timing() {
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    let input = std::path::PathBuf::from(
        std::env::var_os("MOLLA_TRANSFER_TIMING_INPUT").expect("captured input"),
    );
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::SHADER_FLOAT32_ATOMIC
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )
    .unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let times = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    for native in [false, true] {
        for two_pass in [false, true] {
            let case = input.join(format!("implicit-false-two-pass-{two_pass}"));
            let fields: Vec<_> = (0..9)
                .map(|i| std::fs::read(case.join(format!("particle-{i}.bin"))).unwrap())
                .collect();
            let mut cfg = MpmConfig::sand();
            cfg.grid_size = 32;
            cfg.grid_origin = [-1.; 3];
            cfg.cell_size = 0.0625;
            cfg.max_particles = 50_000;
            cfg.max_rigid = 4;
            cfg.youngs = 3000.;
            cfg.particle_volume = 0.025_f32.powi(3);
            cfg.particle_mass = 1600. * cfg.particle_volume;
            let mut owners: Vec<_> = (0..2)
                .map(|variant| {
                    let mut sim = if native {
                        MpmSimulation::new_native_explicit(&engine, cfg).unwrap()
                    } else {
                        MpmSimulation::new(&engine, cfg).unwrap()
                    };
                    if variant == 1 {
                        sim.install_bounded_centered_transfer();
                    }
                    sim.set_p2g_mode(two_pass);
                    sim.set_particles(queue, &vec![[0.; 3]; 50_000]).unwrap();
                    sim.set_rigid_spheres(
                        queue,
                        &[
                            ([0.2, -0.2, -0.2], 0.1, [0.02, -0.01, 0.]),
                            ([-0.87, 0.8, -0.87], 0.08, [0.; 3]),
                        ],
                    )
                    .unwrap();
                    queue.write_buffer(
                        &sim.uniform_buf,
                        0,
                        bytemuck::bytes_of(&sim.step_uniforms(0.0001)),
                    );
                    queue.write_buffer(
                        &sim.material_table,
                        0,
                        bytemuck::cast_slice(&sim.step_material_table()),
                    );
                    sim
                })
                .collect();
            for round in 0..9 {
                for offset in 0..2 {
                    let variant = (round + offset) % 2;
                    let sim = &mut owners[variant];
                    for (buffer, bytes) in sim.checkpoint_buffers().iter().zip(&fields) {
                        assert_eq!(buffer.size(), bytes.len() as u64);
                        queue.write_buffer(buffer, 0, bytes);
                    }
                    queue.submit([]);
                    complete(device);
                    let allocations = DeviceScope::start().unwrap();
                    let transfers = TransferScope::start().unwrap();
                    let dispatches = DispatchScope::start().unwrap();
                    let start = std::time::Instant::now();
                    let mut encoder = device.create_command_encoder(&Default::default());
                    encoder.write_timestamp(&queries, 0);
                    sim.encode_step_work(&mut encoder, 4, 0, false);
                    encoder.write_timestamp(&queries, 1);
                    encoder.resolve_query_set(&queries, 0..2, &times, 0);
                    queue.submit([encoder.finish()]);
                    complete(device);
                    let wall_ms = start.elapsed().as_secs_f64() * 1000.;
                    let allocations = allocations.finish();
                    let transfers = transfers.finish();
                    let dispatches = dispatches.finish();
                    assert_eq!(allocations.observations, 0);
                    assert!(
                        !allocations.invalid && !transfers.overflowed && !dispatches.overflowed
                    );
                    assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
                    assert_eq!(
                        dispatches.encoded_dispatches,
                        if variant == 1 { 40 } else { 32 }
                    );
                    let ticks = read_words(device, queue, &times);
                    let tick = |i| u64::from(ticks[i]) | (u64::from(ticks[i + 1]) << 32);
                    let gpu_ms =
                        (tick(2) - tick(0)) as f64 * f64::from(queue.get_timestamp_period()) / 1e6;
                    assert!(gpu_ms.is_finite() && gpu_ms > 0.);
                    for buffer in [
                        &sim.p_pos,
                        &sim.p_vel,
                        &sim.p_c,
                        &sim.p_f,
                        &sim.rigid_wrench,
                    ] {
                        assert!(
                            read_words(device, queue, buffer)
                                .iter()
                                .all(|x| f32::from_bits(*x).is_finite())
                        );
                    }
                    println!(
                        "transfer-timing native={native} two_pass={two_pass} round={round} variant={variant} warmup={} particles=50000 substeps=4 dt=0.0001 gpu_ms={gpu_ms:.9} wall_ms={wall_ms:.9} dispatches={} allocations=0 uploads=0 readbacks=0",
                        round == 0,
                        dispatches.encoded_dispatches
                    );
                }
            }
        }
    }
}
