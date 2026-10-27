use super::super::replay_tests::read_words;
use super::*;

#[path = "mpm_repeatability_tests.rs"]
mod repeatability;

fn config() -> MpmConfig {
    let mut cfg = MpmConfig::sand();
    cfg.max_particles = 129;
    cfg.max_rigid = 2;
    cfg.youngs = 1000.;
    cfg.particle_mass = 0.02;
    cfg.particle_volume = 0.0000125;
    cfg
}

fn points(n: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            [
                -0.15 + (i % 7) as f32 * 0.021,
                0.2 + (i / 7 % 7) as f32 * 0.021,
                -0.1 + (i / 49) as f32 * 0.021,
            ]
        })
        .collect()
}

fn seeded(engine: &Engine, mode: usize, implicit: bool) -> MpmSimulation {
    let queue = engine.queue().unwrap();
    let mut cfg = config();
    if implicit {
        cfg.time_step = MpmTimeStep::Implicit { jacobi_iters: 2 };
    }
    let mut sim = if mode >= 3 {
        MpmSimulation::new_sparse(engine, cfg, 64).unwrap()
    } else {
        MpmSimulation::new(engine, cfg).unwrap()
    };
    sim.set_particles_with_velocity(queue, &points(129), &vec![[0.02, -0.01, 0.005]; 129])
        .unwrap();
    sim.set_materials(
        &[
            MpmRheology::Sand {
                friction_angle: 0.5,
            },
            MpmRheology::Snow {
                theta_c: 0.025,
                theta_s: 0.0075,
                hardening: 10.,
            },
        ],
        (0..129).map(|i| i % 2).collect(),
    )
    .unwrap();
    sim.set_rigid_spheres(queue, &[([-0.09, 0.23, -0.07], 0.045, [0.015, -0.005, 0.])])
        .unwrap();
    match mode {
        1 => {
            sim.prepare_particle_tile_ordering(queue, 129).unwrap();
            sim.set_particle_tile_ordering(true).unwrap();
        }
        2 => {
            sim.set_active_grid(true).unwrap();
            sim.set_p2g_mode(false);
        }
        4 => sim.set_deferred_p2g_lookup(true).unwrap(),
        5 => {
            sim.prepare_parallel_particle_tile_ordering(queue, 129)
                .unwrap();
            sim.set_particle_tile_ordering(true).unwrap();
        }
        _ => {}
    }
    if std::env::var_os("MOLLA_COMMAND_BATCH_COMPENSATED_TRANSFER").is_some() {
        sim.prepare_compensated_grid_transfer(sim.g_mass.size() * 5)
            .unwrap();
        sim.set_compensated_grid_transfer(true).unwrap();
    }
    sim
}

fn near(name: &str, a: &[u32], b: &[u32]) {
    assert_eq!(a.len(), b.len());
    for (i, (&a, &b)) in a.iter().zip(b).enumerate() {
        let a = f32::from_bits(a);
        let b = f32::from_bits(b);
        assert!(
            a.is_finite() && b.is_finite() && (a - b).abs() <= 2e-5,
            "{name}[{i}]: {a} != {b}"
        );
    }
}

fn compare(engine: &Engine, a: &MpmSimulation, b: &MpmSimulation) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (name, a, b) in [
        ("position", &a.p_pos, &b.p_pos),
        ("velocity", &a.p_vel, &b.p_vel),
        ("affine", &a.p_c, &b.p_c),
        ("deformation", &a.p_f, &b.p_f),
        ("plasticity", &a.p_jp, &b.p_jp),
        ("compaction", &a.tire_compaction_buf, &b.tire_compaction_buf),
        ("wrench", &a.rigid_wrench, &b.rigid_wrench),
    ] {
        near(
            name,
            &read_words(device, queue, a),
            &read_words(device, queue, b),
        );
    }
    assert_eq!(
        &read_words(device, queue, &a.p_material)[..129],
        &read_words(device, queue, &b.p_material)[..129]
    );
}

#[test]
fn prepared_records_are_send_sync() {
    fn check<T: Send + Sync>() {}
    check::<PreparedMpmStep>();
}

#[test]
#[ignore = "needs a real GPU adapter; repeatability diagnostic"]
fn command_batch_reference_repeatability_diagnostic() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut maximum = 0_f32;
    for pair in 0..6 {
        let mut a = seeded(&engine, 2, true);
        let mut b = seeded(&engine, 2, true);
        for sim in [&mut a, &mut b] {
            for (dt, n) in [(0.0001, 2), (0.0002, 2), (0.00008, 1)] {
                sim.step_n_with(device, queue, dt, n).unwrap();
            }
        }
        let a = read_words(device, queue, &a.rigid_wrench);
        let b = read_words(device, queue, &b.rigid_wrench);
        let mut pair_maximum = 0_f32;
        for (i, (&a, &b)) in a.iter().zip(&b).enumerate() {
            let a = f32::from_bits(a);
            let b = f32::from_bits(b);
            assert!(a.is_finite() && b.is_finite());
            let error = (a - b).abs();
            pair_maximum = pair_maximum.max(error);
            if error > 2e-5 {
                println!("reference_pair={pair} wrench[{i}]={a} vs {b} error={error}");
            }
        }
        maximum = maximum.max(pair_maximum);
        println!("reference_pair={pair} max_abs_wrench_error={pair_maximum}");
    }
    assert!(
        maximum <= 2e-5,
        "unchanged submission reference is not repeatable within 2e-5: {maximum}"
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn command_batch_matches_submission_reference_across_solver_modes() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for implicit in [false, true] {
        for mode in 0..6 {
            let mut control = seeded(&engine, mode, implicit);
            let mut candidate = seeded(&engine, mode, implicit);
            let steps = [(0.0001, 2), (0.0002, 2), (0.00008, 1)];
            let records: Vec<_> = steps
                .iter()
                .map(|&(dt, n)| candidate.prepare_step(dt, n).unwrap())
                .collect();
            for &(dt, n) in &steps {
                control.step_n_with(device, queue, dt, n).unwrap();
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let before = device.get_internal_counters().hal;
            let encoder = device.create_command_encoder(&Default::default());
            let resources = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            let mut batch = candidate.begin_command_batch(encoder);
            for record in &records {
                batch.encode_step(record).unwrap();
            }
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
            compare(&engine, &candidate, &control);
            assert!(candidate.particle_material_dirty.is_none());
            if mode >= 2 {
                let status = candidate
                    .read_active_grid_status_with(device, queue)
                    .unwrap();
                assert!(!status.overflow);
                assert_eq!(status.completed_substeps, 5);
            }
            println!(
                "mode={mode} implicit={implicit}: 3 immutable records / 5 completed substeps, state/history/wrench <=2e-5, zero hot buffers/transfers"
            );
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
#[cfg(feature = "resource-counters")]
fn command_batch_cancellation_stale_records_and_empty_work_are_safe() {
    use molla_core::gpu_metrics::{DeviceScope, TransferScope};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut sim = seeded(&engine, 3, false);
    let mut control = seeded(&engine, 3, false);
    let record = sim.prepare_step(0.0001, 2).unwrap();
    let before = read_words(device, queue, &sim.p_pos);
    {
        let mut cancelled =
            sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        cancelled.encode_step(&record).unwrap();
    }
    assert!(sim.active_grid_needs_reset);
    assert!(sim.particle_material_dirty.is_some());
    assert_eq!(read_words(device, queue, &sim.p_pos), before);
    let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
    batch.encode_step(&record).unwrap();
    batch.submit(queue);
    control.step_n_with(device, queue, 0.0001, 2).unwrap();
    compare(&engine, &sim, &control);
    assert!(!sim.active_grid_needs_reset);
    assert!(sim.particle_material_dirty.is_none());
    let mut cancelled = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
    cancelled.encode_step(&record).unwrap();
    drop(cancelled);
    assert!(!sim.active_grid_needs_reset);
    let foreign = control.prepare_step(0.0001, 1).unwrap();
    let encoder = device.create_command_encoder(&Default::default());
    let resources = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let mut batch = sim.begin_command_batch(encoder);
    assert!(batch.encode_step(&foreign).is_err());
    drop(batch);
    assert_eq!(resources.finish().observations, 0);
    assert_eq!(transfers.finish().upload_bytes, 0);
    for mutation in 0..10 {
        let stale = sim.prepare_step(0.0001, 1).unwrap();
        match mutation {
            0 => sim.set_time_step(MpmTimeStep::Implicit { jacobi_iters: 1 }),
            1 => sim.set_material(900., 0.4),
            2 => sim.set_rheology(MpmRheology::Sand {
                friction_angle: 0.4,
            }),
            3 => sim.set_p2g_mode(false),
            4 => {
                sim.set_p2g_mode(true);
                sim.set_deferred_p2g_lookup(true).unwrap();
            }
            5 => sim.reserve_sparse_tiles(128).unwrap(),
            6 => sim
                .set_materials(
                    &[MpmRheology::Sand {
                        friction_angle: 0.4,
                    }],
                    vec![0; 129],
                )
                .unwrap(),
            7 => sim.set_rigid_spheres(queue, &[]).unwrap(),
            8 => {
                sim.prepare_particle_tile_ordering(queue, 129).unwrap();
                sim.set_particle_tile_ordering(true).unwrap();
            }
            _ => sim.set_particles(queue, &points(129)).unwrap(),
        }
        let mut batch = sim.begin_command_batch(device.create_command_encoder(&Default::default()));
        assert!(batch.encode_step(&stale).is_err(), "mutation {mutation}");
    }
    for dt in [0., -0.1, f32::NAN, f32::INFINITY] {
        assert!(sim.prepare_step(dt, 1).is_err());
    }
    sim.set_particles(queue, &[]).unwrap();
    let empty = sim.prepare_step(0.0001, 9).unwrap();
    let zero_steps = sim.prepare_step(0.0001, 0).unwrap();
    let encoder = device.create_command_encoder(&Default::default());
    let resources = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let mut batch = sim.begin_command_batch(encoder);
    batch.encode_step(&empty).unwrap();
    batch.encode_step(&zero_steps).unwrap();
    batch.submit(queue);
    assert!(sim.active_grid_needs_reset);
    assert_eq!(resources.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!(transfers.upload_bytes, 0);
    assert_eq!(transfers.readback_bytes, 0);
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn command_batch_chains_feedback_solver_and_export_and_retains_queued_resources() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireMpmGpuKernel::new(device);
    let mut candidate = seeded(&engine, 3, false);
    let mut control = seeded(&engine, 3, false);
    let contacts = [TireMpmContactCell {
        center_radius: [-0.1, 0.23, -0.08, 0.2],
        force_pressure: [0.05, -0.1, 0.02, 20.],
    }];
    let contact_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("batch GPU contacts"),
        size: 32,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&contact_buffer, 0, bytemuck::cast_slice(&contacts));
    let params = MpmTireFeedbackParams {
        dt: 0.0001,
        plastic_compaction_gain: 0.00001,
        max_compaction: 0.1,
        damping: 1.,
    };
    let buffers = candidate.tire_particle_buffers();
    let feedback = kernel
        .prepare_in_place_from_buffers(
            device,
            buffers[0],
            buffers[1],
            buffers[2],
            &contact_buffer,
            1,
            &TireMpmGpuInPlaceParams {
                particle_count: 129,
                particle_mass: candidate.cfg.particle_mass,
                dt: params.dt,
                plastic_compaction_gain: params.plastic_compaction_gain,
                max_compaction: params.max_compaction,
                damping: params.damping,
            },
        )
        .unwrap();
    let outputs = [64_u64, 64, 256].map(|size| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("batch terrain"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    });
    let raster_params = TireMpmTerrainRasterParams {
        particle_count: 129,
        origin: [-0.3, -0.3],
        spacing: 0.15,
        nx: 4,
        nz: 4,
        rest_height: 0.0,
    };
    let raster = kernel
        .prepare_raster_to_buffers(
            device,
            buffers[0],
            buffers[1],
            buffers[2],
            &outputs[0],
            &outputs[1],
            &outputs[2],
            &raster_params,
            [0.; 3],
        )
        .unwrap();
    let record = candidate.prepare_step(0.0001, 2).unwrap();
    let mut batch =
        candidate.begin_command_batch(device.create_command_encoder(&Default::default()));
    for _ in 0..3 {
        feedback.encode(batch.encoder_mut());
        batch.encode_step(&record).unwrap();
    }
    raster.encode(batch.encoder_mut());
    drop((feedback, raster, contact_buffer, record));
    batch.submit(queue);
    for _ in 0..3 {
        control
            .apply_tire_contacts_gpu_resident_with(device, queue, &kernel, &contacts, params)
            .unwrap();
        control.step_n_with(device, queue, 0.0001, 2).unwrap();
    }
    compare(&engine, &candidate, &control);
    let to_rows = |b| {
        read_words(device, queue, b)
            .chunks_exact(4)
            .map(|v| {
                [
                    f32::from_bits(v[0]),
                    f32::from_bits(v[1]),
                    f32::from_bits(v[2]),
                    f32::from_bits(v[3]),
                ]
            })
            .collect::<Vec<_>>()
    };
    let expected = kernel
        .dispatch_raster_blocking(
            device,
            queue,
            &to_rows(&control.p_pos),
            &to_rows(&control.p_vel),
            &to_rows(&control.tire_compaction_buf),
            &raster_params,
            [0.; 3],
        )
        .unwrap();
    near(
        "height",
        &read_words(device, queue, &outputs[0]),
        bytemuck::cast_slice(&expected.heights),
    );
    near(
        "sinkage",
        &read_words(device, queue, &outputs[1]),
        bytemuck::cast_slice(&expected.sinkage),
    );
    near(
        "surface velocity",
        &read_words(device, queue, &outputs[2]),
        bytemuck::cast_slice(&expected.surface_velocity_map),
    );
    let record = candidate.prepare_step(0.0001, 2).unwrap();
    let position_view = candidate.p_pos.clone();
    let mut batch =
        candidate.begin_command_batch(device.create_command_encoder(&Default::default()));
    batch.encode_step(&record).unwrap();
    batch.submit(queue);
    drop((candidate, record));
    control.step_n_with(device, queue, 0.0001, 2).unwrap();
    near(
        "owner drop",
        &read_words(device, queue, &position_view),
        &read_words(device, queue, &control.p_pos),
    );
}
