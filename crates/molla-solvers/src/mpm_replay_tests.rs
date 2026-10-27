use super::*;

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires timestamp and native float-atomic GPU support"]
fn frame_profile_window_accumulates_and_skips_without_changing_physics() {
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::SHADER_FLOAT32_ATOMIC
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
    )
    .unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 8;
    cfg.grid_origin = [-0.5; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 4;
    let mut sim = MpmSimulation::new_native_explicit(&engine, cfg).unwrap();
    let mut reference = MpmSimulation::new_native_explicit(&engine, cfg).unwrap();
    reference
        .set_particles(queue, &[[0.0; 3], [0.03; 3]])
        .unwrap();
    sim.set_particles(queue, &[[0.0; 3], [0.03; 3]]).unwrap();
    sim.prepare_step_profiling(3).unwrap();
    assert!(!sim.begin_frame_profile(queue, 0.0001, 4));
    assert!(sim.begin_frame_profile(queue, 0.0001, 3));
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    for _ in 0..2 {
        sim.step(&engine, 0.0001).unwrap();
        assert!(!sim.step_profile_pending());
    }
    sim.step(&engine, 0.0001).unwrap();
    assert!(sim.step_profile_pending());
    assert!(!sim.begin_frame_profile(queue, 0.0001, 3));
    sim.step(&engine, 0.0001).unwrap();
    assert_eq!(transfers.finish().readback_bytes, 0);
    assert_eq!(allocations.finish().observations, 0);
    let profile = sim.read_step_profile().unwrap().unwrap();
    assert_eq!(profile.requested_substeps, 3);
    assert!(profile.native_float_atomics);
    assert_eq!(profile.stages.len(), 15);
    assert_eq!(profile.readback_bytes, 15 * 16);
    assert_eq!(profile.stages[0].start_ms, 0.0);
    assert!(
        profile
            .stages
            .windows(2)
            .all(|s| s[1].start_ms >= s[0].start_ms)
    );
    for step in 0..3 {
        assert_eq!(
            profile
                .stages
                .iter()
                .filter(|s| s.substep == Some(step))
                .count(),
            5
        );
    }
    assert!(
        profile
            .stages
            .iter()
            .all(|s| s.elapsed_ms.is_finite() && s.elapsed_ms >= 0.0)
    );
    assert!(sim.read_step_profile().unwrap().is_none());
    assert!(sim.begin_frame_profile(queue, 0.0001, 3));
    sim.step(&engine, 0.0001).unwrap();
    sim.discard_step_profile();
    assert!(sim.read_step_profile().unwrap().is_none());
    assert!(sim.begin_frame_profile(queue, 0.0001, 3));
    sim.step(&engine, 0.0002).unwrap();
    assert!(!sim.step_profile_pending());
    assert!(sim.begin_frame_profile(queue, 0.0001, 1));
    sim.set_time_step(MpmTimeStep::Explicit);
    sim.step(&engine, 0.0001).unwrap();
    assert!(!sim.step_profile_pending());
    assert!(sim.begin_frame_profile(queue, 0.0001, 1));
    sim.step(&engine, 0.0001).unwrap();
    assert_eq!(sim.read_step_profile().unwrap().unwrap().stages.len(), 5);
    for dt in [
        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0002, 0.0001, 0.0001,
    ] {
        reference.step(&engine, dt).unwrap();
    }
    let device = engine.device().unwrap();
    for (actual, expected) in sim
        .checkpoint_buffers()
        .into_iter()
        .zip(reference.checkpoint_buffers())
    {
        for (a, b) in read_words(device, queue, actual)
            .into_iter()
            .zip(read_words(device, queue, expected))
        {
            let (a, b) = (f32::from_bits(a), f32::from_bits(b));
            assert!(a.is_finite() && b.is_finite());
            assert!((a - b).abs() <= 2e-5 * b.abs().max(1.0));
        }
    }
    println!(
        "cross-submission profile: distinct substeps, no inner readback/allocation, retained unread capture and cancellation pass"
    );
}

#[test]
#[ignore = "requires native float-atomic GPU support"]
fn native_explicit_matches_portable_particles_and_wrenches() {
    native_portable_parity(false);
}

#[test]
#[ignore = "requires native float-atomic GPU support"]
fn bounded_centered_native_lifecycle() {
    native_portable_parity(true);
}

fn native_portable_parity(combined: bool) {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 16;
    cfg.grid_origin = [-0.5; 3];
    cfg.cell_size = 0.0625;
    cfg.max_particles = 193;
    cfg.max_rigid = 1;
    cfg.youngs = 3000.0;
    cfg.particle_mass = 0.1;
    cfg.particle_volume = 0.0000625;
    let points: Vec<_> = (0..193)
        .map(|i| {
            [
                -0.16 + (i % 9) as f32 * 0.04,
                -0.08 + (i / 9 % 5) as f32 * 0.04,
                (i / 45) as f32 * 0.025,
            ]
        })
        .collect();
    let velocities: Vec<_> = (0..193)
        .map(|i| [0.03 * (i % 7) as f32, -0.01 * (i % 5) as f32, 0.02])
        .collect();
    let mut portable = MpmSimulation::new(&engine, cfg).unwrap();
    let mut native = MpmSimulation::new_native_explicit(&engine, cfg).unwrap();
    if combined {
        portable.install_bounded_centered_transfer();
        native.install_bounded_centered_transfer();
    }
    assert!(native.uses_native_float_atomics());
    assert!(!native.uses_two_pass_p2g());
    assert!(!portable.uses_native_float_atomics());
    let solids = [MpmPrescribedSolid {
        center: [0.0, 0.0, 0.0],
        orientation: [0.0, 0.0, 0.0, 1.0],
        linear_velocity: [0.03, 0.0, 0.0],
        angular_velocity: [0.0, 0.0, 0.1],
        shape: MpmSolidShape::Sphere { radius: 0.1 },
    }];
    for sim in [&mut portable, &mut native] {
        sim.set_particles_with_velocity(queue, &points, &velocities)
            .unwrap();
        sim.set_prescribed_solids(queue, &solids).unwrap();
    }
    for step in 0..16 {
        let count = if combined {
            [193, 65, 0, 1][step / 4]
        } else {
            193
        };
        if combined && step % 4 == 0 {
            for sim in [&mut portable, &mut native] {
                sim.set_particles_with_velocity(queue, &points[..count], &velocities[..count])
                    .unwrap();
            }
        }
        #[cfg(feature = "resource-counters")]
        let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
        for sim in [&mut portable, &mut native] {
            sim.step(&engine, 0.0001).unwrap();
        }
        #[cfg(feature = "resource-counters")]
        assert_eq!(
            dispatches.finish().encoded_dispatches,
            if count == 0 {
                0
            } else if combined {
                18
            } else {
                14
            }
        );
        if let Err(error) = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        }) {
            eprintln!("native MPM parity completion failed: {error}");
            std::process::exit(76);
        }
        for (field, (a, b)) in [
            (&native.p_pos, &portable.p_pos),
            (&native.p_vel, &portable.p_vel),
            (&native.p_c, &portable.p_c),
            (&native.p_f, &portable.p_f),
            (&native.p_volume, &portable.p_volume),
            (&native.p_mass, &portable.p_mass),
            (&native.p_jp, &portable.p_jp),
            (&native.rigid_wrench, &portable.rigid_wrench),
        ]
        .into_iter()
        .enumerate()
        {
            for (word, (a, b)) in read_words(device, queue, a)
                .into_iter()
                .zip(read_words(device, queue, b))
                .enumerate()
            {
                let (a, b) = (f32::from_bits(a), f32::from_bits(b));
                assert!(a.is_finite() && b.is_finite());
                assert!(
                    (a - b).abs() <= 2e-5 * b.abs().max(1.0),
                    "step {step} field {field} word {word}: {a} != {b}"
                );
            }
        }
        assert_eq!(
            read_words(device, queue, &native.p_material),
            read_words(device, queue, &portable.p_material)
        );
        assert!(
            read_words(device, queue, &native.g_kdiag)[..cfg.grid_size.pow(3) as usize]
                .iter()
                .all(|&v| v == 0)
        );
    }
    native.set_active_grid(true).unwrap();
    native.set_active_grid(false).unwrap();
    native.set_time_step(MpmTimeStep::Implicit { jacobi_iters: 1 });
    assert!(native.step(&engine, 0.0001).is_err());
    assert!(native.prepare_step(0.0001, 1).is_err());
    cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 1 };
    assert!(MpmSimulation::new_native_explicit(&engine, cfg).is_err());
    println!(
        "native explicit: combined={combined}, 16-step particle/history/wrench parity and unsupported-mode rejection pass"
    );
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn native_explicit_rejects_device_without_enabled_feature() {
    let engine = Engine::new_gpu().unwrap();
    assert!(MpmSimulation::new_native_explicit(&engine, MpmConfig::sand()).is_err());
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires native float-atomic GPU support"]
fn native_explicit_prescribed_shell_pass_is_noop_and_legacy_keeps_it() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 8;
    cfg.grid_origin = [-0.5; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 2;
    cfg.max_rigid = 1;
    let mut sim = MpmSimulation::new_native_explicit(&engine, cfg).unwrap();
    sim.set_particles(queue, &[[0.0; 3], [0.25; 3]]).unwrap();
    sim.set_prescribed_solids(
        queue,
        &[MpmPrescribedSolid {
            shape: MpmSolidShape::Sphere { radius: 0.15 },
            center: [0.0; 3],
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.1, 0.0, 0.0],
            angular_velocity: [0.0; 3],
        }],
    )
    .unwrap();
    let counts = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    sim.step(&engine, 0.0001).unwrap();
    assert_eq!(counts.finish().encoded_dispatches, 7);
    let before = read_words(device, queue, &sim.rigid_wrench);
    assert!(before.iter().any(|&word| f32::from_bits(word).abs() > 1e-6));
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        sim.bind_step(&mut pass);
        pass.set_pipeline(&sim.pipe_rigid_force);
        pass.dispatch_workgroups(cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    queue.submit([encoder.finish()]);
    assert_eq!(before, read_words(device, queue, &sim.rigid_wrench));
    sim.set_rigid_spheres(queue, &[([0.0; 3], 0.15, [0.1, 0.0, 0.0])])
        .unwrap();
    let counts = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    sim.step(&engine, 0.0001).unwrap();
    assert_eq!(counts.finish().encoded_dispatches, 8);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    println!(
        "prescribed shell integration is bit-exactly inert; prescribed steps encode 7 passes and legacy steps retain 8"
    );
}

#[test]
#[ignore = "requires native float-atomic GPU support"]
fn native_float_atomic_capability_probe() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let source = r#"
alias GridScalar = f32;
@group(0) @binding(0) var<storage, read_write> values: array<atomic<GridScalar>>;
@compute @workgroup_size(64)
fn main() {
    atomicAdd(&values[0], 0.25);
    atomicAdd(&values[1], -0.03125);
}
"#;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("soil native float atomic capability"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("soil native float atomic capability"),
        layout: None,
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("soil atomic probe values"),
        size: 8,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(64, 1, 1);
    }
    queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    let values: Vec<_> = read_words(device, queue, &buffer)
        .into_iter()
        .map(f32::from_bits)
        .collect();
    assert_eq!(values, [1024.0, -128.0]);
    println!(
        "native f32 buffer atomics: 4096 contended positive/negative additions are exact; not an MPM implementation or performance gate"
    );
}

pub(super) fn instrument_global_cas(source: &'static str) -> std::borrow::Cow<'static, str> {
    if std::env::var_os("MOLLA_SOIL_CAS_PROBE").is_none() {
        return source.into();
    }
    let limit: u32 = std::env::var("MOLLA_SOIL_CAS_PROBE_LIMIT")
        .map(|v| v.parse().expect("integer CAS probe limit"))
        .unwrap_or(4096);
    assert!((1..=4096).contains(&limit));
    eprintln!("MPM global CAS diagnostic retry limit={limit}");
    let mut source = source.to_owned();
    for (name, buffer, bit) in [
        ("mass", "g_mass", 1),
        ("mom", "g_momentum", 2),
        ("kdiag", "g_kdiag", 4),
    ] {
        let start = source.find(&format!("fn atomic_add_{name}(")).unwrap();
        let end = start + source[start..].find("\n}").unwrap() + 2;
        let replacement = format!(
            r#"fn atomic_add_{name}(idx: u32, val: f32) {{
    let header = arrayLength(&g_mass);
    atomicAdd(&g_kdiag[header], 1u);
    if idx >= arrayLength(&{buffer}) {{
        atomicOr(&g_kdiag[header + 2u], {bit}u);
        atomicMax(&g_kdiag[header + 1u], idx);
        return;
    }}
    var old_bits = atomicLoad(&{buffer}[idx]);
    for (var attempt = 0u; attempt < {limit}u; attempt++) {{
        let next = bitcast<u32>(bitcast<f32>(old_bits) + val);
        let ex = atomicCompareExchangeWeak(&{buffer}[idx], old_bits, next);
        if ex.exchanged {{ return; }}
        old_bits = ex.old_value;
    }}
    atomicOr(&g_kdiag[header + 2u], {bit}u << 8u);
    atomicMax(&g_kdiag[header + 1u], idx);
    atomicStore(&g_kdiag[header + 3u], old_bits);
}}"#
        );
        source.replace_range(start..end, &replacement);
    }
    source.into()
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn p2g_shared_accumulation_matches_global_scatter() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 16;
    cfg.grid_origin = [-0.5; 3];
    cfg.cell_size = 0.0625;
    cfg.max_particles = 193;
    cfg.particle_mass = 0.1;
    cfg.particle_volume = 0.0000625;
    let points: Vec<_> = (0..193)
        .map(|i| {
            [
                -0.16 + (i % 9) as f32 * 0.04,
                -0.08 + (i / 9 % 5) as f32 * 0.04,
                (i / 45) as f32 * 0.025,
            ]
        })
        .collect();
    let velocities: Vec<_> = (0..193)
        .map(|i| [0.03 * (i % 7) as f32, -0.01 * (i % 5) as f32, 0.02])
        .collect();
    let mut sim = MpmSimulation::new(&engine, cfg).unwrap();
    sim.set_particles_with_velocity(queue, &points, &velocities)
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
    let mut reference: Option<Vec<Vec<f32>>> = None;
    for two_pass in [false, true] {
        sim.set_p2g_mode(two_pass);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            sim.bind_step(&mut pass);
            pass.set_pipeline(&sim.pipe_reset);
            pass.dispatch_workgroups(cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE), 1, 1);
            pass.set_pipeline(sim.p2g_pipeline());
            pass.dispatch_workgroups(sim.particle_count.div_ceil(WORKGROUP_SIZE), 1, 1);
        }
        queue.submit([encoder.finish()]);
        if let Err(error) = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        }) {
            eprintln!("P2G parity completion failed: {error}");
            std::process::exit(76);
        }
        let fields: Vec<Vec<f32>> = [&sim.g_mass, &sim.g_momentum, &sim.g_kdiag]
            .into_iter()
            .map(|buffer| {
                read_words(device, queue, buffer)
                    .into_iter()
                    .map(f32::from_bits)
                    .collect()
            })
            .collect();
        let total_mass: f64 = fields[0].iter().map(|&v| f64::from(v)).sum();
        assert!((total_mass - 193.0 * f64::from(cfg.particle_mass)).abs() < 2e-5);
        if let Some(reference) = &reference {
            for (field, (actual, expected)) in fields.iter().zip(reference).enumerate() {
                for (cell, (&a, &b)) in actual.iter().zip(expected).enumerate() {
                    assert!(a.is_finite() && b.is_finite());
                    assert!(
                        (a - b).abs() <= 2e-5 * b.abs().max(1.0),
                        "field {field} cell {cell}: {a} != {b}"
                    );
                }
            }
        } else {
            reference = Some(fields);
        }
        println!("P2G shared={two_pass}: 193-particle mass={total_mass:.9}");
    }
}

#[test]
#[ignore = "requires a captured tractor boundary and real GPU"]
fn captured_tractor_boundary_replay() {
    let path = std::path::PathBuf::from(
        std::env::var_os("MOLLA_SOIL_RUNTIME_REPLAY").expect("capture directory required"),
    );
    let checkpoint =
        MpmSoilCheckpoint::from_bytes(&std::fs::read(path.join("soil.bin")).unwrap(), 1_000_000)
            .unwrap();
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = MpmSimulation::new(&engine, checkpoint.config()).unwrap();
    sim.restore_soil_with(queue, &checkpoint).unwrap();
    let centers = std::fs::read(path.join("centers.bin")).unwrap();
    let motion = std::fs::read(path.join("motion.bin")).unwrap();
    assert_eq!(centers.len() as u64, sim.rigid_center.size());
    assert_eq!(motion.len() as u64, sim.rigid_vel.size());
    sim.rigid_count = 5;
    sim.prescribed_solids = true;
    queue.write_buffer(&sim.rigid_center, 0, &centers);
    queue.write_buffer(&sim.rigid_vel, 0, &motion);
    let wait = |stage: &str| {
        eprintln!("captured tractor replay: {stage} submitted");
        if let Err(error) = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        }) {
            eprintln!("captured tractor replay: {stage} failed: {error}");
            std::process::exit(76);
        }
        eprintln!("captured tractor replay: {stage} completed");
    };
    sim.step_with(device, queue, 0.0001).unwrap();
    wait("whole soil step");
    let wrenches = sim.read_rigid_wrenches(&engine).unwrap();
    assert_eq!(wrenches.len(), 5);
    assert!(
        wrenches
            .iter()
            .flat_map(|(f, t)| f.iter().chain(t))
            .all(|v| v.is_finite())
    );
    sim.checkpoint_soil_with(queue).unwrap();
    println!("captured tractor soil boundary completed: {wrenches:?}");
}

pub(super) fn split_step(sim: &MpmSimulation, device: &wgpu::Device, queue: &wgpu::Queue, dt: f32) {
    let expected = sim.step_uniforms(dt);
    let actual = read_words(device, queue, &sim.uniform_buf);
    assert_eq!(actual, bytemuck::cast_slice::<Uniforms, u32>(&[expected]));
    eprintln!(
        "MPM entry: two_pass={} uniforms={expected:?}",
        sim.uses_two_pass_p2g()
    );
    let particles_before: Vec<_> = sim
        .checkpoint_buffers()
        .into_iter()
        .map(|buffer| read_words(device, queue, buffer))
        .collect();
    let grid = sim.cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE);
    let particles = sim.particle_count.div_ceil(WORKGROUP_SIZE);
    for (name, pipeline, count) in [
        ("grid reset", &sim.pipe_reset, grid),
        (
            "wrench reset",
            &sim.pipe_rigid_wrench_reset,
            sim.rigid_count.div_ceil(WORKGROUP_SIZE),
        ),
        ("P2G", sim.p2g_pipeline(), particles),
        ("rigid imprint", &sim.pipe_rigid, grid),
        ("grid update", &sim.pipe_grid, grid),
        ("rigid force", &sim.pipe_rigid_force, grid),
        ("G2P", &sim.pipe_g2p, particles),
        ("plasticity", &sim.pipe_plastic, particles),
    ] {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            sim.bind_step(&mut pass);
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(count, 1, 1);
        }
        queue.submit([encoder.finish()]);
        eprintln!("MPM split step: {name} submitted");
        if let Err(error) = device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        }) {
            eprintln!("MPM split step: {name} failed: {error}");
            std::process::exit(76);
        }
        eprintln!("MPM split step: {name} completed");
        if name == "grid reset" {
            for (i, (buffer, before)) in sim
                .checkpoint_buffers()
                .into_iter()
                .zip(&particles_before)
                .enumerate()
            {
                assert_eq!(
                    &read_words(device, queue, buffer),
                    before,
                    "grid reset changed particle stream {i}"
                );
            }
            for (buffer, cells) in [
                (&sim.g_mass, sim.cfg.grid_size.pow(3)),
                (&sim.g_momentum, 3 * sim.cfg.grid_size.pow(3)),
                (&sim.g_kdiag, sim.cfg.grid_size.pow(3)),
            ] {
                let values = read_words(device, queue, buffer);
                assert!(
                    values[..cells as usize].iter().all(|&v| v == 0),
                    "grid reset left nonzero physical storage"
                );
            }
            eprintln!(
                "MPM reset: particle streams unchanged; physical mass/momentum/stiffness zero"
            );
        }
        if name == "P2G" && std::env::var_os("MOLLA_SOIL_CAS_PROBE").is_some() {
            let grid = read_words(device, queue, &sim.g_kdiag);
            let meta = sim.g_mass.size() as usize / 4;
            let status = &grid[meta..meta + 4];
            eprintln!(
                "P2G global CAS probe [calls, largest failed index, flags, old bits]={status:?}"
            );
            assert!(status[0] > 0, "CAS probe did not execute");
            assert_eq!(
                status[2], 0,
                "diagnostic CAS failure; not a physical result"
            );
        }
    }
}

pub(super) fn grid_buffers(sim: &MpmSimulation) -> [&wgpu::Buffer; 6] {
    [
        &sim.g_mass,
        &sim.g_momentum,
        &sim.g_velocity,
        &sim.g_rigid_mask,
        &sim.g_kdiag,
        &sim.g_velocity_prev,
    ]
}

pub(super) fn read_words(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    source: &wgpu::Buffer,
) -> Vec<u32> {
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MPM replay diagnostic staging"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &stage, 0, source.size());
    queue.submit([encoder.finish()]);
    let slice = stage.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        tx.send(result).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let words = bytemuck::cast_slice::<u8, u32>(&slice.get_mapped_range()).to_vec();
    stage.unmap();
    words
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn mpm_identical_grid_wrench_replay() {
    let engine = Engine::new_gpu().expect("GPU required for grid replay");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 32;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.0625;
    cfg.max_particles = 50_000;
    cfg.max_rigid = 4;
    cfg.youngs = 3000.0;
    cfg.particle_volume = 0.025_f32.powi(3);
    cfg.particle_mass = 1600.0 * cfg.particle_volume;
    cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 3 };
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
    sim.set_particles_with_velocity(queue, &points, &vec![[0.03, -0.02, 0.01]; points.len()])
        .unwrap();
    sim.set_rigid_spheres(
        queue,
        &[
            ([0.2, -0.2, -0.2], 0.1, [0.02, -0.01, 0.0]),
            ([-0.87, 0.8, -0.87], 0.08, [0.0; 3]),
        ],
    )
    .unwrap();
    sim.step_n_with(device, queue, 0.0001, 3).unwrap();
    sim.active_tiles
        .upload(
            queue,
            &cfg,
            sim.particle_count,
            sim.rigid_count,
            sim.grid_window_cells,
        )
        .unwrap();
    let saved: Vec<_> = grid_buffers(&sim)
        .iter()
        .map(|source| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MPM pre-imprint grid replay"),
                size: source.size(),
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        })
        .collect();
    let groups = cfg.grid_size.pow(3).div_ceil(WORKGROUP_SIZE);
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        sim.bind_step(&mut pass);
        pass.set_pipeline(&sim.pipe_reset);
        pass.dispatch_workgroups(groups, 1, 1);
    }
    sim.active_tiles
        .encode_build(&mut encoder, sim.particle_count, sim.rigid_count);
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        sim.bind_step(&mut pass);
        pass.set_pipeline(&sim.pipe_p2g_two_pass);
        pass.dispatch_workgroups(sim.particle_count.div_ceil(WORKGROUP_SIZE), 1, 1);
    }
    for (source, dest) in grid_buffers(&sim).iter().zip(&saved) {
        encoder.copy_buffer_to_buffer(source, 0, dest, 0, source.size());
    }
    queue.submit([encoder.finish()]);
    let mut reference_grid: Option<Vec<Vec<u32>>> = None;
    let mut reference_wrench: Option<Vec<([f32; 3], [f32; 3])>> = None;
    let mut maximum_error = [0.0_f32; 2];
    for repeat in 0..6 {
        for active in [false, true] {
            let mut encoder = device.create_command_encoder(&Default::default());
            for (source, dest) in saved.iter().zip(grid_buffers(&sim)) {
                encoder.copy_buffer_to_buffer(source, 0, dest, 0, source.size());
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                sim.bind_step(&mut pass);
                pass.set_pipeline(&sim.pipe_rigid_wrench_reset);
                pass.dispatch_workgroups(1, 1, 1);
                let dispatch = |pass: &mut wgpu::ComputePass<'_>,
                                dense: &wgpu::ComputePipeline,
                                tile_index: usize| {
                    if active {
                        pass.set_pipeline(&sim.active_pipelines[tile_index]);
                        pass.dispatch_workgroups_indirect(&sim.active_tiles.indirect, 0);
                    } else {
                        pass.set_pipeline(dense);
                        pass.dispatch_workgroups(groups, 1, 1);
                    }
                };
                dispatch(&mut pass, &sim.pipe_rigid, 1);
                dispatch(&mut pass, &sim.pipe_grid, 2);
                for _ in 0..3 {
                    dispatch(&mut pass, &sim.pipe_implicit_apply, 3);
                    dispatch(&mut pass, &sim.pipe_implicit_swap, 4);
                }
                dispatch(&mut pass, &sim.pipe_rigid_force, 5);
            }
            queue.submit([encoder.finish()]);
            let grid: Vec<_> = grid_buffers(&sim)
                .into_iter()
                .map(|buffer| read_words(device, queue, buffer))
                .collect();
            if let Some(reference) = &reference_grid {
                for (buffer, (a, b)) in grid.iter().zip(reference).enumerate() {
                    assert_eq!(a.len(), b.len());
                    assert!(
                        a == b,
                        "replay grid buffer {buffer} first differing word: {:?}",
                        a.iter().zip(b).position(|(a, b)| a != b)
                    );
                }
            } else {
                reference_grid = Some(grid);
            }
            let wrench = sim.read_rigid_wrenches_with(device, queue).unwrap();
            let reference = reference_wrench.get_or_insert_with(|| wrench.clone());
            let mut errors = [0.0_f32; 2];
            for (reference, value) in reference.iter().zip(&wrench) {
                for (kind, (a, b)) in [(&reference.0, &value.0), (&reference.1, &value.1)]
                    .into_iter()
                    .enumerate()
                {
                    for (&a, &b) in a.iter().zip(b) {
                        assert!(a.is_finite() && b.is_finite());
                        errors[kind] = errors[kind].max((a - b).abs() / (2e-5 * a.abs().max(1.0)));
                        maximum_error[kind] = maximum_error[kind].max(errors[kind]);
                    }
                }
            }
            println!(
                "replay={repeat} active={active} force_error_over_tolerance={} torque_error_over_tolerance={} torque={:?}",
                errors[0], errors[1], wrench[0].1
            );
        }
    }
    println!("identical_grid_replay_maximum_error_over_tolerance={maximum_error:?}");
    assert!(
        maximum_error.into_iter().all(|error| error <= 1.0),
        "identical-grid wrench reduction exceeds frozen gate"
    );
}
