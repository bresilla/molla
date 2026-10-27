use super::replay_tests::{grid_buffers, read_words};
use super::*;

fn scatter(sim: &MpmSimulation, encoder: &mut wgpu::CommandEncoder) {
    sim.encode_bounded_reference(encoder);
    let mut pass = encoder.begin_compute_pass(&Default::default());
    sim.bind_step(&mut pass);
    for (pipeline, groups) in [
        (
            &sim.pipe_reset,
            sim.cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE),
        ),
        (
            sim.p2g_pipeline(),
            sim.particle_count.div_ceil(WORKGROUP_SIZE),
        ),
    ] {
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

fn coupling(sim: &MpmSimulation, encoder: &mut wgpu::CommandEncoder, implicit: bool) {
    let mut pass = encoder.begin_compute_pass(&Default::default());
    sim.bind_step(&mut pass);
    pass.set_pipeline(&sim.pipe_rigid_wrench_reset);
    pass.dispatch_workgroups(1, 1, 1);
    let grid = sim.cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE);
    for pipeline in [&sim.pipe_rigid, &sim.pipe_grid] {
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(grid, 1, 1);
    }
    if implicit {
        for _ in 0..3 {
            for pipeline in [&sim.pipe_implicit_apply, &sim.pipe_implicit_swap] {
                pass.set_pipeline(pipeline);
                pass.dispatch_workgroups(grid, 1, 1);
            }
        }
    }
    pass.set_pipeline(&sim.pipe_rigid_force);
    pass.dispatch_workgroups(grid, 1, 1);
}

fn error(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(&a, &b)| {
            assert!(a.is_finite() && b.is_finite());
            f64::from((a - b).abs() / (2e-5 * b.abs().max(1.)))
        })
        .fold(0., f64::max)
}

#[test]
#[ignore = "requires a GPU adapter and a fresh capture directory"]
fn scatter_wrench_identical_particles_and_grid() {
    let output = std::path::PathBuf::from(
        std::env::var_os("MOLLA_SCATTER_WRENCH_ARTIFACTS").expect("capture directory required"),
    );
    std::fs::create_dir(&output).expect("fresh capture directory");
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let relative = std::env::var_os("MOLLA_SCATTER_RELATIVE").is_some();
    let bounded = std::env::var_os("MOLLA_SCATTER_BOUNDED").is_some();
    let centered = std::env::var_os("MOLLA_SCATTER_CENTERED").is_some();
    assert!(!bounded || relative);
    assert!(!centered || bounded);
    let input = std::env::var_os("MOLLA_SCATTER_WRENCH_INPUT").map(std::path::PathBuf::from);
    println!(
        "scatter-wrench relative_momentum={relative} bounded_reference={bounded} centered_affine={centered} restored_input={}",
        input.is_some()
    );
    let mut failures = [0_usize; 2];
    for implicit in [false, true] {
        for two_pass in [false, true] {
            let case = output.join(format!("implicit-{implicit}-two-pass-{two_pass}"));
            std::fs::create_dir(&case).unwrap();
            let mut cfg = MpmConfig::sand();
            cfg.grid_size = 32;
            cfg.grid_origin = [-1.; 3];
            cfg.cell_size = 0.0625;
            cfg.max_particles = 50_000;
            cfg.max_rigid = 4;
            cfg.youngs = 3000.;
            cfg.particle_volume = 0.025_f32.powi(3);
            cfg.particle_mass = 1600. * cfg.particle_volume;
            cfg.time_step = if implicit {
                MpmTimeStep::Implicit { jacobi_iters: 3 }
            } else {
                MpmTimeStep::Explicit
            };
            let points: Vec<_> = (0..50_000)
                .map(|i| {
                    [
                        -0.65 + (i % 40) as f32 * 0.025,
                        -0.65 + (i / 40 % 40) as f32 * 0.025,
                        -0.65 + (i / 1600) as f32 * 0.025,
                    ]
                })
                .collect();
            let mut sim = MpmSimulation::new(&engine, cfg).unwrap();
            if relative {
                if centered {
                    sim.install_bounded_centered_transfer();
                } else if bounded {
                    sim.install_bounded_relative_transfer();
                } else {
                    sim.install_relative_momentum_transfer();
                }
            }
            sim.set_p2g_mode(two_pass);
            sim.set_particles_with_velocity(
                queue,
                &points,
                &vec![[0.03, -0.02, 0.01]; points.len()],
            )
            .unwrap();
            sim.set_rigid_spheres(
                queue,
                &[
                    ([0.2, -0.2, -0.2], 0.1, [0.02, -0.01, 0.]),
                    ([-0.87, 0.8, -0.87], 0.08, [0.; 3]),
                ],
            )
            .unwrap();
            if let Some(input) = &input {
                let input = input.join(format!("implicit-{implicit}-two-pass-{two_pass}"));
                for (i, buffer) in sim.checkpoint_buffers().iter().enumerate() {
                    let bytes = std::fs::read(input.join(format!("particle-{i}.bin"))).unwrap();
                    assert_eq!(bytes.len() as u64, buffer.size());
                    queue.write_buffer(buffer, 0, &bytes);
                }
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
            } else {
                sim.step_n_with(device, queue, 0.0001, 3).unwrap();
            }
            let particle_input: Vec<_> = sim
                .checkpoint_buffers()
                .iter()
                .map(|b| read_words(device, queue, b))
                .collect();
            for (i, words) in particle_input.iter().enumerate() {
                std::fs::write(
                    case.join(format!("particle-{i}.bin")),
                    bytemuck::cast_slice(words),
                )
                .unwrap();
            }
            let saved: Vec<_> = grid_buffers(&sim)
                .iter()
                .map(|b| {
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("identical pre-imprint grid"),
                        size: b.size(),
                        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    })
                })
                .collect();
            let mut encoder = device.create_command_encoder(&Default::default());
            scatter(&sim, &mut encoder);
            for (source, target) in grid_buffers(&sim).iter().zip(&saved) {
                encoder.copy_buffer_to_buffer(source, 0, target, 0, source.size());
            }
            queue.submit([encoder.finish()]);
            let canonical: Vec<_> = saved.iter().map(|b| read_words(device, queue, b)).collect();
            for (i, words) in canonical.iter().enumerate() {
                std::fs::write(
                    case.join(format!("grid-{i}.bin")),
                    bytemuck::cast_slice(words),
                )
                .unwrap();
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            coupling(&sim, &mut encoder, implicit);
            queue.submit([encoder.finish()]);
            let reference = sim.read_rigid_wrenches_with(device, queue).unwrap();
            if bounded {
                std::fs::write(
                    case.join("uniforms.bin"),
                    bytemuck::cast_slice(&read_words(device, queue, &sim.uniform_buf)),
                )
                .unwrap();
            }
            std::fs::write(
                case.join("reference.json"),
                serde_json::to_vec_pretty(&reference).unwrap(),
            )
            .unwrap();
            for repeat in 0..12 {
                for offset in 0..2 {
                    let matched_grid = (repeat + offset) % 2 == 1;
                    let mut encoder = device.create_command_encoder(&Default::default());
                    if matched_grid {
                        for (source, target) in saved.iter().zip(grid_buffers(&sim)) {
                            encoder.copy_buffer_to_buffer(source, 0, target, 0, source.size());
                        }
                    } else {
                        scatter(&sim, &mut encoder);
                    }
                    queue.submit([encoder.finish()]);
                    let fields: Vec<_> = grid_buffers(&sim)
                        .iter()
                        .map(|b| read_words(device, queue, b))
                        .collect();
                    if matched_grid {
                        assert_eq!(fields, canonical);
                    }
                    let grid_error: Vec<_> = fields[..2]
                        .iter()
                        .zip(&canonical[..2])
                        .map(|(a, b)| {
                            error(
                                &a.iter().map(|x| f32::from_bits(*x)).collect::<Vec<_>>(),
                                &b.iter().map(|x| f32::from_bits(*x)).collect::<Vec<_>>(),
                            )
                        })
                        .collect();
                    let mut encoder = device.create_command_encoder(&Default::default());
                    coupling(&sim, &mut encoder, implicit);
                    queue.submit([encoder.finish()]);
                    let wrenches = sim.read_rigid_wrenches_with(device, queue).unwrap();
                    let mut wrench_error = [0_f64; 2];
                    for (a, b) in wrenches.iter().zip(&reference) {
                        wrench_error[0] = wrench_error[0].max(error(&a.0, &b.0));
                        wrench_error[1] = wrench_error[1].max(error(&a.1, &b.1));
                    }
                    let passed = wrench_error.into_iter().all(|e| e <= 1.);
                    failures[usize::from(matched_grid)] += usize::from(!passed);
                    println!(
                        "scatter-wrench implicit={implicit} two_pass={two_pass} repeat={repeat} matched_grid={matched_grid} grid_error={grid_error:?} wrench_error={wrench_error:?} passed={passed}"
                    );
                    if !passed {
                        for (i, words) in fields[..2].iter().enumerate() {
                            std::fs::write(
                                case.join(format!("failure-{repeat}-{matched_grid}-grid-{i}.bin")),
                                bytemuck::cast_slice(words),
                            )
                            .unwrap();
                        }
                        std::fs::write(
                            case.join(format!("failure-{repeat}-{matched_grid}-wrench.json")),
                            serde_json::to_vec_pretty(&wrenches).unwrap(),
                        )
                        .unwrap();
                    }
                }
            }
            let after: Vec<_> = sim
                .checkpoint_buffers()
                .iter()
                .map(|b| read_words(device, queue, b))
                .collect();
            assert_eq!(
                particle_input, after,
                "all replays must share exact evolved particle state"
            );
            std::fs::write(
                case.join("gather-grid-velocity.bin"),
                bytemuck::cast_slice(&read_words(device, queue, &sim.g_velocity)),
            )
            .unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                sim.bind_step(&mut pass);
                pass.set_pipeline(&sim.pipe_g2p);
                pass.dispatch_workgroups(sim.particle_count.div_ceil(WORKGROUP_SIZE), 1, 1);
            }
            queue.submit([encoder.finish()]);
            for (name, buffer) in [
                ("position", &sim.p_pos),
                ("velocity", &sim.p_vel),
                ("affine", &sim.p_c),
            ] {
                std::fs::write(
                    case.join(format!("gather-{name}.bin")),
                    bytemuck::cast_slice(&read_words(device, queue, buffer)),
                )
                .unwrap();
            }
        }
    }
    println!(
        "scatter-wrench failures fresh_scatter={} matched_grid={}",
        failures[0], failures[1]
    );
    assert_eq!(failures[1], 0, "identical-grid downstream wrench gate");
    assert_eq!(
        failures[0], 0,
        "fresh-scatter wrench repeatability gate remains open"
    );
}
