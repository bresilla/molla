use super::*;
use crate::mpm::replay_tests::read_words;

fn buffer(engine: &Engine, values: &[f32]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    engine
        .device()
        .unwrap()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paired MPM coupling fixture"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
}

#[test]
#[ignore = "requires real GPU native atomics"]
fn paired_soil_consumer_native_mpm_publication_and_feedback() {
    native_mpm_feedback(false, false);
}

#[test]
#[ignore = "requires real GPU native atomics; prepared Newton MPM feedback"]
fn newton_native_mpm_publication_and_feedback() {
    native_mpm_feedback(true, false);
}

#[test]
#[ignore = "requires GPU native atomics and nested timestamp queries"]
fn newton_native_mpm_profile_equivalence() {
    assert_eq!(
        native_mpm_feedback(true, false),
        native_mpm_feedback(true, true)
    );
}

fn native_mpm_feedback(newton: bool, profiled: bool) -> Vec<Vec<u32>> {
    native_mpm_feedback_layout(feedback_engine(newton, profiled), newton, profiled, 0)
}

#[test]
#[ignore = "requires native GPU; dense/active/compact paired friction publication parity"]
fn active_unilateral_paired_feedback() {
    let engine = feedback_engine(false, false);
    let dense = native_mpm_feedback_layout(engine.clone(), false, false, 0);
    for layout in [1, 2] {
        let actual = native_mpm_feedback_layout(engine.clone(), false, false, layout);
        assert_eq!(actual, dense, "paired wall friction layout={layout}");
    }
}

#[test]
#[ignore = "requires native GPU; dense/active/compact Newton publication parity"]
fn active_unilateral_newton_feedback() {
    let engine = feedback_engine(true, false);
    let dense = native_mpm_feedback_layout(engine.clone(), true, false, 0);
    for layout in [1, 2] {
        let actual = native_mpm_feedback_layout(engine.clone(), true, false, layout);
        assert_eq!(actual, dense, "paired Newton layout={layout}");
    }
}

fn feedback_engine(newton: bool, profiled: bool) -> Arc<Engine> {
    let features = wgpu::Features::SHADER_FLOAT32_ATOMIC;
    let features = if profiled {
        features
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
    } else {
        features
    };
    #[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
    let features = if newton {
        features | wgpu::Features::PASSTHROUGH_SHADERS
    } else {
        features
    };
    let engine = Arc::new(Engine::new_gpu_with_features(features).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let _ = newton;
    engine
}

fn native_mpm_feedback_layout(
    engine: Arc<Engine>,
    newton: bool,
    profiled: bool,
    layout: u32,
) -> Vec<Vec<u32>> {
    native_mpm_feedback_method(
        engine,
        if newton {
            ContactMethod::Newton
        } else {
            ContactMethod::Projection
        },
        profiled,
        layout,
    )
}

#[test]
#[ignore = "requires native GPU; experimental ADMM MPM integration"]
fn admm_mpm_feedback_layouts_and_profile() {
    let engine = feedback_engine(false, true);
    let root = std::path::PathBuf::from(std::env::var("MOLLA_ADMM_MPM_OUTPUT").unwrap());
    std::fs::create_dir_all(&root).unwrap();
    let dense = native_mpm_feedback_method(engine.clone(), ContactMethod::Admm, false, 0);
    for (name, layout, profiled) in [
        ("active", 1, false),
        ("compact", 2, false),
        ("profiled", 0, true),
    ] {
        let actual =
            native_mpm_feedback_method(engine.clone(), ContactMethod::Admm, profiled, layout);
        for (i, values) in actual.iter().enumerate() {
            std::fs::write(
                root.join(format!("{name}-{i}.bin")),
                bytemuck::cast_slice(values),
            )
            .unwrap();
        }
        assert_eq!(actual, dense, "ADMM MPM {name} exact parity");
    }
    for (i, values) in dense.iter().enumerate() {
        std::fs::write(
            root.join(format!("dense-{i}.bin")),
            bytemuck::cast_slice(values),
        )
        .unwrap();
    }
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "method": "experimental-admm",
            "particles": 65,
            "dofs": 3,
            "slots": 128,
            "steps_per_case": 4,
            "layouts": ["dense", "active", "compact", "dense-profiled"],
            "exact_layout_and_profile_parity": true,
            "snapshot_buffers": dense.len(),
            "momentum_bound": 1e-6,
            "paired_delta_bound": 2e-14,
            "acceptance": false
        }))
        .unwrap(),
    )
    .unwrap();
}

fn native_mpm_feedback_method(
    engine: Arc<Engine>,
    method: ContactMethod,
    profiled: bool,
    layout: u32,
) -> Vec<Vec<u32>> {
    let newton = method == ContactMethod::Newton;
    let admm = method == ContactMethod::Admm;
    let setup = std::time::Instant::now();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 16;
    cfg.grid_origin = [-1.; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 65;
    cfg.max_rigid = 1;
    cfg.particle_mass = 0.5 / 65.;
    cfg.particle_volume = cfg.particle_mass / 1600.;
    let mut soil = if layout == 2 {
        MpmSimulation::new_native_sparse(&engine, cfg, 64)
    } else {
        MpmSimulation::new_native_explicit(&engine, cfg)
    }
    .unwrap();
    if layout == 1 {
        soil.set_active_grid(true).unwrap();
    }
    soil.set_particles_with_velocity(
        engine.queue().unwrap(),
        &[[0., 0., 0.]; 65],
        &[[1., -1., 0.]; 65],
    )
    .unwrap();
    soil.set_prescribed_solids(
        engine.queue().unwrap(),
        &[MpmPrescribedSolid {
            shape: MpmSolidShape::Sphere { radius: 0.25 },
            center: [0., -0.25, 0.],
            orientation: [0., 0., 0., 1.],
            linear_velocity: [1., 0., 0.],
            angular_velocity: [0.; 3],
        }],
    )
    .unwrap();
    let mass = buffer(&engine, &[1e6, 0., 0., 0., 1e6, 0., 0., 0., 1e6]);
    let jac = buffer(
        &engine,
        &[
            1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
    );
    let free = buffer(&engine, &[1., 2f32.powi(-30), 0., 2f32.powi(-31), 0., 0.]);
    let delta = buffer(&engine, &[0.; 6]);
    let machine_status = buffer(&engine, &[0.]);
    let config = MpmUnilateralConfig {
        node_capacity: 128,
        search_distance: 0.25,
        max_iterations: 64,
    };
    let bytes = if admm {
        PreparedMpmUnilateralCoupling::required_admm_bytes(config, 3, 16, 1)
    } else if newton {
        PreparedMpmUnilateralCoupling::required_newton_bytes(config, 3, 16, 1)
    } else {
        PreparedMpmUnilateralCoupling::required_wall_friction_bytes_for_rates(
            config,
            3,
            16,
            1,
            RateEncoding::Paired,
        )
    }
    .unwrap();
    let bytes = bytes
        + if layout > 0 {
            ExperimentalSoilCandidatesGpu::grid_storage_bytes(4096, true)
                - ExperimentalSoilCandidatesGpu::grid_storage_bytes(4096, false)
        } else {
            0
        };
    let input = || MpmUnilateralInputs {
        mass: &mass,
        jacobians: &jac,
        free_rates: &free,
        dofs: 3,
        max_owned_bytes: bytes,
    };
    if admm {
        assert_eq!(
            PreparedMpmUnilateralCoupling::required_admm_bytes_for_soil(&soil, config, 3).unwrap(),
            bytes
        );
    }
    if admm && layout == 0 && !profiled {
        let before: Vec<_> = soil
            .checkpoint_buffers()
            .into_iter()
            .map(|b| read_words(engine.device().unwrap(), engine.queue().unwrap(), b))
            .collect();
        let mut overflow = PreparedMpmUnilateralCoupling::new_experimental_admm(
            engine.clone(),
            &mut soil,
            input(),
            MpmUnilateralConfig {
                node_capacity: 1,
                ..config
            },
            &[0.3],
        )
        .unwrap();
        overflow.step(&mut soil, 0.0001).unwrap();
        assert_ne!(
            read_words(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                overflow.status_buffer()
            )[0] & 512,
            0
        );
        assert!(
            read_words(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                overflow.rates_buffer()
            )
            .iter()
            .all(|&v| v == 0)
        );
        assert_eq!(
            soil.checkpoint_buffers()
                .into_iter()
                .map(|b| read_words(engine.device().unwrap(), engine.queue().unwrap(), b))
                .collect::<Vec<_>>(),
            before
        );
        eprintln!("ADMM MPM candidate overflow rejected without advancing particles");
    }
    if newton || admm {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        assert!(
            (if admm {
                PreparedMpmUnilateralCoupling::new_experimental_admm
            } else {
                PreparedMpmUnilateralCoupling::new_with_newton_rates
            })(
                engine.clone(),
                &mut soil,
                MpmUnilateralInputs {
                    max_owned_bytes: bytes - 1,
                    ..input()
                },
                config,
                &[0.3]
            )
            .is_err()
        );
        assert_eq!(allocations.finish().observations, 0);
    }
    let mut coupling = if admm {
        PreparedMpmUnilateralCoupling::new_experimental_admm(
            engine.clone(),
            &mut soil,
            input(),
            config,
            &[0.3],
        )
    } else if newton {
        PreparedMpmUnilateralCoupling::new_with_newton_rates(
            engine.clone(),
            &mut soil,
            input(),
            config,
            &[0.3],
        )
    } else {
        PreparedMpmUnilateralCoupling::new_with_friction_rates(
            engine.clone(),
            &mut soil,
            input(),
            config,
            &[0.3],
            true,
            RateEncoding::Paired,
        )
    }
    .unwrap();
    assert_eq!(coupling.owned_buffer_bytes(), bytes);
    eprintln!(
        "native feedback layout={layout} newton={newton} admm={admm} constructor_ms={:.3}",
        setup.elapsed().as_secs_f64() * 1000.0
    );
    if newton {
        assert!(coupling.local_inputs_buffer().is_none());
        assert!(coupling.failure_solver().is_none());
    }
    if admm {
        assert!(coupling.history.is_none());
        assert!(coupling.newton_method_name().is_none());
        assert!(coupling.newton_failure_buffer().is_none());
        assert!(coupling.newton_preparation_failure_buffer().is_none());
        assert!(coupling.admm_preparation_failure_buffer().is_some());
        assert!(coupling.local_inputs_buffer().is_none());
        assert!(coupling.failure_solver().is_none());
    }
    assert!(
        coupling
            .prepare_machine_publication(SoilRatePublicationInputs {
                rates: &free,
                velocity_delta: &delta,
                machine_status: &machine_status,
                max_owned_bytes: 16
            })
            .is_err()
    );
    let publication = coupling
        .prepare_paired_machine_publication(SoilPairedRatePublicationInputs {
            rates: &free,
            velocity_delta: &delta,
            machine_status: &machine_status,
            max_owned_bytes: 16,
        })
        .unwrap();
    let floats = |b: &wgpu::Buffer| {
        read_words(engine.device().unwrap(), engine.queue().unwrap(), b)
            .into_iter()
            .map(f32::from_bits)
            .collect::<Vec<_>>()
    };
    let reconstruct = |v: &[f32]| {
        v.chunks_exact(2)
            .map(|q| q[0] as f64 + q[1] as f64)
            .collect::<Vec<_>>()
    };
    let mut worst = 0f64;
    if profiled {
        soil.prepare_step_profiling(4).unwrap();
        assert!(soil.begin_frame_profile(engine.queue().unwrap(), 0.0001, 4));
    }
    for _ in 0..4 {
        let before = reconstruct(&floats(&free));
        engine
            .queue()
            .unwrap()
            .write_buffer(&delta, 0, bytemuck::cast_slice(&[0f32; 6]));
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        coupling.step(&mut soil, 0.0001).unwrap();
        assert_eq!(transfers.finish().readback_bytes, 0);
        assert_eq!(allocations.finish().observations, 0);
        assert_eq!(
            read_words(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                coupling.status_buffer()
            )[0],
            0
        );
        let after = reconstruct(&floats(coupling.rates_buffer()));
        let wrench = floats(&soil.rigid_wrench);
        for axis in 0..3 {
            let impulse = wrench[axis] as f64 * 0.0001f32 as f64;
            let change = 1e6 * (after[axis] - before[axis]);
            let error = (change - impulse).abs() / change.abs().max(impulse.abs()).max(1.);
            worst = worst.max(error);
            assert!(error < 1e-6, "axis {axis}: {change} vs {impulse}");
        }
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        publication.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(
            read_words(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                &machine_status
            )[0],
            0
        );
        assert_eq!(floats(&free), floats(coupling.rates_buffer()));
        let change = reconstruct(&floats(&delta));
        for axis in 0..3 {
            assert!((change[axis] - (after[axis] - before[axis])).abs() < 2e-14);
        }
    }
    eprintln!("native paired MPM four-step feedback maximum normalized momentum error={worst}");
    if let Some(history) = &coupling.history {
        let counts = read_words(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            history.counters(),
        );
        assert_eq!(&counts[..2], &[1, 0]);
        assert!(
            counts[2] > 0 && counts[3] > 0,
            "native feedback must restore nonzero matched history: {counts:?}"
        );
        eprintln!(
            "native Newton history matched={} nonzero={}",
            counts[2], counts[3]
        );
        coupling.invalidate_newton_history(engine.queue().unwrap());
        assert_eq!(
            read_words(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                history.counters()
            ),
            [0; 4]
        );
    }
    if profiled {
        let profile = soil.read_step_profile().unwrap().unwrap();
        assert_eq!(profile.stages.len(), 4 * 7);
        assert_eq!(profile.contact_stages.len(), 4 * 3);
        assert_eq!(profile.readback_bytes, 4 * 10 * 16);
        for step in 0..4 {
            let parent = profile
                .stages
                .iter()
                .find(|s| s.stage == MpmProfileStage::ContactProjection && s.substep == Some(step))
                .unwrap();
            let children = &profile.contact_stages[3 * step..3 * step + 3];
            assert_eq!(
                children.iter().map(|s| s.stage.name()).collect::<Vec<_>>(),
                [
                    "contact-preparation",
                    "contact-solve",
                    "contact-publication"
                ]
            );
            let epsilon = f64::from(profile.timestamp_period_ns) * 1e-6;
            for child in children {
                assert_eq!(child.substep, Some(step));
                assert!(child.elapsed_ms.is_finite() && child.elapsed_ms > 0.0);
                assert!(child.start_ms >= parent.start_ms);
                assert!(
                    child.start_ms + child.elapsed_ms
                        <= parent.start_ms + parent.elapsed_ms + epsilon
                );
            }
            assert!(
                children
                    .windows(2)
                    .all(|s| s[0].start_ms + s[0].elapsed_ms <= s[1].start_ms + epsilon)
            );
            assert!(
                children.iter().map(|s| s.elapsed_ms).sum::<f64>() <= parent.elapsed_ms + epsilon
            );
        }
        assert!(soil.read_step_profile().unwrap().is_none());
    }
    let mut snapshot: Vec<_> = soil
        .checkpoint_buffers()
        .into_iter()
        .map(|b| read_words(engine.device().unwrap(), engine.queue().unwrap(), b))
        .collect();
    for b in [
        &free,
        coupling.rates_buffer(),
        coupling.status_buffer(),
        &soil.rigid_wrench,
    ] {
        snapshot.push(read_words(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            b,
        ));
    }
    if admm {
        admm_mpm_failure_checks(
            &engine,
            &mut soil,
            &mut coupling,
            &mass,
            &free,
            &delta,
            &machine_status,
            &publication,
        );
    }
    if newton {
        let ContactSolve::Newton(owner) = &coupling.contact else {
            panic!("wrong contact backend")
        };
        let encode = || {
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            owner.encode(&mut encoder);
            engine.queue().unwrap().submit([encoder.finish()]);
            assert_eq!(allocations.finish().observations, 0);
            let traffic = traffic.finish();
            assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
            read_words(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                owner.status_buffer(),
            )
        };
        engine
            .queue()
            .unwrap()
            .write_buffer(&mass, 0, bytemuck::bytes_of(&-1.0f32));
        let failure = encode();
        assert_ne!(failure[0] & 4096, 0);
        assert_eq!(failure[1], 2);
        let packet = read_words(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            owner.preparation_failure_buffer(),
        );
        assert_eq!(packet[0], 1);
        assert_eq!(packet[4], 2);
        assert_eq!(packet[16], (-1.0f32).to_bits());
        for b in [
            owner.rates_buffer(),
            owner.velocities_buffer(),
            owner.contact_impulses_buffer(),
            owner.wall_impulses_buffer(),
        ] {
            assert!(
                read_words(engine.device().unwrap(), engine.queue().unwrap(), b)
                    .iter()
                    .all(|&v| v == 0)
            );
        }
        engine
            .queue()
            .unwrap()
            .write_buffer(&mass, 0, bytemuck::bytes_of(&1e6f32));
        assert_eq!(encode(), failure);
        assert_eq!(
            read_words(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                owner.preparation_failure_buffer()
            ),
            packet
        );
        engine.queue().unwrap().write_buffer(
            owner.status_buffer(),
            0,
            bytemuck::cast_slice(&[0u32; 8]),
        );
        assert_eq!(encode()[0], 0);
        assert_eq!(
            read_words(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                owner.preparation_failure_buffer()
            )[0],
            0
        );
    }
    snapshot
}

fn admm_mpm_failure_checks(
    engine: &Engine,
    soil: &mut MpmSimulation,
    coupling: &mut PreparedMpmUnilateralCoupling,
    mass: &wgpu::Buffer,
    free: &wgpu::Buffer,
    delta: &wgpu::Buffer,
    machine_status: &wgpu::Buffer,
    publication: &SoilRatePublicationGpu,
) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let read = |b: &wgpu::Buffer| read_words(device, queue, b);
    let particles: Vec<_> = soil.checkpoint_buffers().into_iter().map(read).collect();
    let free_before = read(free);
    let delta_before = read(delta);
    queue.write_buffer(mass, 0, bytemuck::bytes_of(&-1.0f32));
    coupling.step(soil, 0.0001).unwrap();
    let failed = read(coupling.status_buffer());
    assert_ne!(failed[0] & 4096, 0);
    assert_eq!(failed[1], 2);
    let packet = read(coupling.admm_preparation_failure_buffer().unwrap());
    assert_eq!(
        (packet[0], packet[4], packet[16]),
        (1, 2, (-1.0f32).to_bits())
    );
    let ContactSolve::Admm(owner) = &coupling.contact else {
        panic!("wrong backend")
    };
    for buffer in [
        owner.rates_buffer(),
        owner.velocities_buffer(),
        owner.contact_impulses_buffer(),
        owner.wall_impulses_buffer(),
        &soil.rigid_wrench,
    ] {
        assert!(read(buffer).iter().all(|&v| v == 0));
    }
    assert_eq!(
        soil.checkpoint_buffers()
            .into_iter()
            .map(read)
            .collect::<Vec<_>>(),
        particles
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    publication.encode(&mut encoder);
    queue.submit([encoder.finish()]);
    assert_eq!(read(machine_status)[0], 32);
    assert_eq!(read(free), free_before);
    assert_eq!(read(delta), delta_before);
    queue.write_buffer(mass, 0, bytemuck::bytes_of(&1e6f32));
    coupling.step(soil, 0.0001).unwrap();
    assert_ne!(read(coupling.status_buffer())[0] & 4096, 0);
    assert_eq!(
        soil.checkpoint_buffers()
            .into_iter()
            .map(read)
            .collect::<Vec<_>>(),
        particles
    );
    let ContactSolve::Admm(owner) = &coupling.contact else {
        unreachable!()
    };
    let mut encoder = device.create_command_encoder(&Default::default());
    owner.encode_reset(&mut encoder);
    encoder.clear_buffer(machine_status, 0, None);
    queue.submit([encoder.finish()]);
    coupling.step(soil, 0.0001).unwrap();
    assert_eq!(read(coupling.status_buffer())[0], 0);
    assert_eq!(
        read(coupling.admm_preparation_failure_buffer().unwrap())[0],
        0
    );

    queue.write_buffer(
        &soil.g_kdiag,
        soil.active_tiles.metadata_offset + 8,
        bytemuck::bytes_of(&1u32),
    );
    coupling.step(soil, 0.0001).unwrap();
    assert_ne!(
        read(coupling.status_buffer())[0] & particle_fault_bridge::MPM_COUPLING_PARTICLE_FAULT,
        0
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    publication.encode(&mut encoder);
    queue.submit([encoder.finish()]);
    assert_eq!(read(machine_status)[0], 32);
    assert_eq!(read(free), free_before);
    assert_eq!(read(delta), delta_before);
    eprintln!(
        "ADMM MPM invalid-mass fail-closed, particle freeze, publication rejection, latch/reset and injected particle-fault bridge passed"
    );
}
