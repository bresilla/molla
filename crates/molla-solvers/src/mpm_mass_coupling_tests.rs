use super::*;
use crate::mpm::replay_tests::read_words;

#[path = "mpm_ordered_coupling_tests.rs"]
mod ordered;

#[path = "mpm_lattice_coupling_tests.rs"]
mod lattice;

#[path = "mpm_projected_commands_tests.rs"]
mod commands;

#[path = "mpm_projected_repeat_tests.rs"]
mod command_repeats;

fn data(engine: &Engine, words: &[f32]) -> wgpu::Buffer {
    let buffer = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("MPM mass coupling test"),
            size: words.len() as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
    engine
        .queue()
        .unwrap()
        .write_buffer(&buffer, 0, bytemuck::cast_slice(words));
    buffer
}

fn fixture(engine: &Engine, native: bool, count: usize, center: [f32; 3]) -> MpmSimulation {
    fixture_layout(engine, native, count, center, 0, 16, 64)
}

fn fixture_layout(
    engine: &Engine,
    native: bool,
    count: usize,
    center: [f32; 3],
    layout: u32,
    size: u32,
    capacity: u32,
) -> MpmSimulation {
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = size;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = count as u32;
    cfg.max_rigid = 2;
    cfg.particle_mass = 0.5 / count as f32;
    cfg.particle_volume = cfg.particle_mass / 1600.0;
    let mut soil = if layout == 2 && native {
        MpmSimulation::new_native_sparse(engine, cfg, capacity)
    } else if layout == 2 {
        MpmSimulation::new_sparse(engine, cfg, capacity)
    } else if native {
        MpmSimulation::new_native_explicit(engine, cfg)
    } else {
        MpmSimulation::new(engine, cfg)
    }
    .unwrap();
    if layout == 1 {
        soil.set_active_grid(true).unwrap();
    }
    soil.set_particles_with_velocity(
        engine.queue().unwrap(),
        &vec![center; count],
        &vec![[1.0, 0.0, 0.0]; count],
    )
    .unwrap();
    soil.set_prescribed_solids(
        engine.queue().unwrap(),
        &[MpmPrescribedSolid {
            shape: MpmSolidShape::Box {
                half_extents: [0.4; 3],
            },
            center,
            orientation: [0.0, 0.0, 0.0, 1.0],
            linear_velocity: [0.0; 3],
            angular_velocity: [0.0; 3],
        }],
    )
    .unwrap();
    soil
}

enum FaultTestCoupling {
    Sticking(PreparedMpmMassCoupling),
    Unilateral(PreparedMpmUnilateralCoupling),
}

impl FaultTestCoupling {
    fn step_with_commands(
        &mut self,
        soil: &mut MpmSimulation,
        dt: f32,
        before: impl FnOnce(&mut wgpu::CommandEncoder),
        after: impl FnOnce(&mut wgpu::CommandEncoder),
    ) -> Result<()> {
        match self {
            Self::Sticking(c) => c.step_with_commands(soil, dt, before, after),
            Self::Unilateral(c) => c.step_with_commands(soil, dt, before, after),
        }
    }

    fn step_and_publish(
        &mut self,
        engine: &Engine,
        soil: &mut MpmSimulation,
        publication: &SoilRatePublicationGpu,
        batched: bool,
    ) {
        if batched {
            self.step_with_commands(soil, 0.0001, |_| {}, |e| publication.encode(e))
                .unwrap();
        } else {
            self.step(soil);
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            publication.encode(&mut encoder);
            engine.queue().unwrap().submit([encoder.finish()]);
        }
    }
    fn step(&mut self, soil: &mut MpmSimulation) {
        match self {
            Self::Sticking(c) => c.step(soil, 0.0001),
            Self::Unilateral(c) => c.step(soil, 0.0001),
        }
        .unwrap();
    }

    fn status(&self) -> &wgpu::Buffer {
        match self {
            Self::Sticking(c) => c.status_buffer(),
            Self::Unilateral(c) => c.status_buffer(),
        }
    }

    fn publication(&self, inputs: SoilRatePublicationInputs<'_>) -> SoilRatePublicationGpu {
        match self {
            Self::Sticking(c) => c.prepare_machine_publication(inputs),
            Self::Unilateral(c) => c.prepare_machine_publication(inputs),
        }
        .unwrap()
    }
}

#[test]
#[ignore = "requires real GPU native atomics"]
fn late_particle_fault_blocks_machine_publication_and_frame_rollback_recovers() {
    late_particle_fault(false);
}

#[test]
#[ignore = "requires real native GPU; batched publication and exact rollback"]
fn projected_commands_late_fault_rollback() {
    late_particle_fault(true);
}

fn late_particle_fault(batched: bool) {
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mass = data(&engine, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]);
    let jac = data(
        &engine,
        &[
            1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
    );
    for (native, ordered, gather) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (true, true, false),
        (true, true, true),
    ] {
        for unilateral in [false, true] {
            let free = data(&engine, &[0.; 3]);
            let delta = data(&engine, &[7., 8., 9.]);
            let machine_status = data(&engine, &[0.]);
            let mut soil = fixture(&engine, native, 1, [0.; 3]);
            let original = soil.inject_late_particle_fault();
            if ordered {
                ordered::prepare_order(&mut soil, queue, gather);
            }
            let mut coupling = if unilateral {
                FaultTestCoupling::Unilateral(
                    PreparedMpmUnilateralCoupling::new(
                        engine.clone(),
                        &mut soil,
                        MpmUnilateralInputs {
                            mass: &mass,
                            jacobians: &jac,
                            free_rates: &free,
                            dofs: 3,
                            max_owned_bytes: 4_000_000,
                        },
                        MpmUnilateralConfig {
                            node_capacity: 128,
                            search_distance: 0.05,
                            max_iterations: 64,
                        },
                    )
                    .unwrap(),
                )
            } else {
                FaultTestCoupling::Sticking(
                    PreparedMpmMassCoupling::new(
                        engine.clone(),
                        &mut soil,
                        MpmMassCouplingInputs {
                            mass: &mass,
                            jacobians: &jac,
                            free_rates: &free,
                            dofs: 3,
                            nodes_per_body: 128,
                            max_owned_bytes: 4_000_000,
                        },
                    )
                    .unwrap(),
                )
            };
            let publication = coupling.publication(SoilRatePublicationInputs {
                rates: &free,
                velocity_delta: &delta,
                machine_status: &machine_status,
                max_owned_bytes: 16,
            });
            let before = soil.checkpoint_soil_with(queue).unwrap().to_bytes();
            let metadata_before = read_words(device, queue, &soil.g_kdiag);
            let mut backup = MpmFrameBackup::new(engine.clone(), &soil, 2_000_000).unwrap();
            backup.begin(&soil).unwrap();
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
            coupling.step_and_publish(&engine, &mut soil, &publication, batched);
            assert_eq!(allocations.finish().observations, 0);
            assert_eq!(transfers.finish().readback_bytes, 0);
            assert_eq!(
                read_words(device, queue, coupling.status())[0],
                MPM_COUPLING_PARTICLE_FAULT,
                "native={native} unilateral={unilateral}"
            );
            assert!(
                soil.read_active_grid_status_with(device, queue)
                    .unwrap()
                    .numerical_fault
            );
            assert_eq!(read_words(device, queue, &machine_status), vec![32]);
            assert_eq!(read_words(device, queue, &free), vec![0; 3]);
            assert_eq!(
                read_words(device, queue, &delta),
                [7.0_f32, 8., 9.].map(f32::to_bits)
            );
            assert!(soil.checkpoint_soil_with(queue).is_err());
            assert!(backup.rollback(&mut soil).unwrap());
            assert_eq!(soil.checkpoint_soil_with(queue).unwrap().to_bytes(), before);
            assert_eq!(read_words(device, queue, &soil.g_kdiag), metadata_before);
            soil.pipe_g2p = original;
            queue.write_buffer(&machine_status, 0, bytemuck::bytes_of(&0u32));
            queue.write_buffer(coupling.status(), 0, bytemuck::bytes_of(&0u32));
            backup.begin(&soil).unwrap();
            coupling.step_and_publish(&engine, &mut soil, &publication, batched);
            assert_eq!(read_words(device, queue, coupling.status())[0], 0);
            assert_eq!(read_words(device, queue, &machine_status), vec![0]);
            assert!(
                !soil
                    .read_active_grid_status_with(device, queue)
                    .unwrap()
                    .faulted()
            );
            assert_ne!(soil.checkpoint_soil_with(queue).unwrap().to_bytes(), before);
            assert_ne!(
                read_words(device, queue, &delta),
                [7.0_f32, 8., 9.].map(f32::to_bits)
            );
            backup.commit();
            println!(
                "native={native} ordered={ordered} gather={gather} unilateral={unilateral}: post-contact NaN blocks machine rates/delta, exact rollback restores latch and history, valid retry publishes"
            );
        }
    }
}

#[test]
#[ignore = "requires real GPU native atomics and timestamp features"]
fn projected_grid_drives_all_particles_and_reports_one_reaction() {
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        )
        .unwrap(),
    );
    assert!(!engine.adapter_info().unwrap().is_software);
    let dt = 0.0001f32;
    let mass = data(&engine, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]);
    let jac = data(
        &engine,
        &[
            1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
    );
    let free = data(&engine, &[0., -9.81 * dt, 0.]);
    for (native, ordered, gather) in [
        (false, false, false),
        (true, false, false),
        (false, true, false),
        (true, true, false),
        (true, true, true),
    ] {
        engine
            .queue()
            .unwrap()
            .write_buffer(&free, 0, bytemuck::cast_slice(&[0., -9.81 * dt, 0.]));
        let mut soil = fixture(&engine, native, 65, [0.; 3]);
        if ordered {
            ordered::prepare_order(&mut soil, engine.queue().unwrap(), gather);
        }
        soil.prepare_step_profiling(1).unwrap();
        let mut coupling = PreparedMpmMassCoupling::new(
            engine.clone(),
            &mut soil,
            MpmMassCouplingInputs {
                mass: &mass,
                jacobians: &jac,
                free_rates: &free,
                dofs: 3,
                nodes_per_body: 128,
                max_owned_bytes: 32_000,
            },
        )
        .unwrap();
        assert_eq!(
            coupling.owned_buffer_bytes(),
            16 * 16 * 16 * 4 + 128 * 32 + 8 + 76 + 160 + 36 + 24 + 20
        );
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        coupling.step_profiled(&mut soil, dt).unwrap();
        assert_eq!(allocations.finish().observations, 0);
        assert_eq!(transfers.finish().readback_bytes, 0);
        assert_eq!(
            read_words(
                engine.device().unwrap(),
                engine.queue().unwrap(),
                coupling.status_buffer()
            ),
            vec![0]
        );
        let rates: Vec<_> = read_words(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            coupling.rates_buffer(),
        )
        .into_iter()
        .map(f32::from_bits)
        .collect();
        let expected = [1.0 / 3.0, -9.81 * dt, 0.0];
        for (a, b) in rates.iter().zip(expected) {
            assert!((a - b).abs() < 2e-5, "native={native}: {rates:?}");
        }
        let velocities: Vec<_> = read_words(
            engine.device().unwrap(),
            engine.queue().unwrap(),
            &soil.p_vel,
        )
        .into_iter()
        .map(f32::from_bits)
        .collect();
        for (i, v) in velocities.chunks_exact(4).take(65).enumerate() {
            for k in 0..3 {
                assert!(
                    (v[k] - rates[k]).abs() < 2e-5,
                    "particle {i}, axis {k}: {} vs {}",
                    v[k],
                    rates[k]
                );
            }
        }
        let positions = soil.read_positions(&engine).unwrap();
        assert_eq!(positions.len(), 65);
        for p in positions {
            assert!((p[0] - dt * rates[0]).abs() < 1e-8);
        }
        let wrench = soil.read_rigid_wrenches(&engine).unwrap()[0];
        assert!((wrench.0[0] * dt - rates[0]).abs() < 2e-5);
        let profile = soil.read_step_profile().unwrap().unwrap();
        let names: Vec<_> = profile.stages.iter().map(|s| s.stage.name()).collect();
        let mut expected_names = vec![
            "grid-reset",
            "rigid-reset",
            "p2g",
            "grid-update",
            "contact-projection",
            "g2p",
            "plasticity",
        ];
        if ordered {
            expected_names.insert(0, "particle-ordering");
        }
        assert_eq!(names, expected_names);
        assert_eq!(profile.requested_substeps, 1);
        let delta = data(&engine, &[0., -9.81 * dt, 0.]);
        let status = data(&engine, &[0.]);
        let publication = coupling
            .prepare_machine_publication(SoilRatePublicationInputs {
                rates: &free,
                velocity_delta: &delta,
                machine_status: &status,
                max_owned_bytes: 16,
            })
            .unwrap();
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        publication.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        let words = |buffer| read_words(engine.device().unwrap(), engine.queue().unwrap(), buffer);
        assert_eq!(words(&status), vec![0]);
        assert_eq!(
            words(&free),
            rates.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        for (value, expected) in words(&delta).iter().zip(&rates) {
            assert!((f32::from_bits(*value) - expected).abs() < 2e-6);
        }
        soil.set_material(2000.0, 0.3);
        assert!(coupling.step(&mut soil, dt).is_err());
        soil.set_prescribed_solids(engine.queue().unwrap(), &[])
            .unwrap();
        assert!(coupling.step(&mut soil, dt).is_err());
        println!(
            "native={native} ordered={ordered} gather={gather}: projected grid -> 65 particles, shared velocity {rates:?}, single impulse and profile verified"
        );
    }
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn capacity_faults_preserve_particles_and_rollback_grid() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mass = data(&engine, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]);
    let jac = data(
        &engine,
        &[
            1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
    );
    let free = data(&engine, &[0.; 3]);
    for (center, capacity, fault) in [
        ([0.; 3], 1, MPM_COUPLING_CAPACITY_FAULT),
        ([-0.875, 0., 0.], 1, MPM_COUPLING_CAPACITY_FAULT),
    ] {
        let mut soil = fixture(&engine, false, 2, center);
        let particle_before: Vec<_> = soil
            .checkpoint_buffers()
            .iter()
            .map(|b| read_words(device, queue, b))
            .collect();
        let read_all = |soil: &MpmSimulation| {
            soil.checkpoint_buffers()
                .into_iter()
                .chain([
                    &soil.g_mass,
                    &soil.g_momentum,
                    &soil.g_velocity,
                    &soil.g_rigid_mask,
                    &soil.g_kdiag,
                    &soil.g_velocity_prev,
                    &soil.rigid_center,
                    &soil.rigid_vel,
                    &soil.rigid_wrench,
                ])
                .map(|b| read_words(device, queue, b))
                .collect::<Vec<_>>()
        };
        let before = read_all(&soil);
        let mut backup = MpmFrameBackup::new(engine.clone(), &soil, 2_000_000).unwrap();
        let mut coupling = PreparedMpmMassCoupling::new(
            engine.clone(),
            &mut soil,
            MpmMassCouplingInputs {
                mass: &mass,
                jacobians: &jac,
                free_rates: &free,
                dofs: 3,
                nodes_per_body: capacity,
                max_owned_bytes: 32_000,
            },
        )
        .unwrap();
        backup.begin(&soil).unwrap();
        coupling.step(&mut soil, 0.0001).unwrap();
        assert_ne!(
            read_words(device, queue, coupling.status_buffer())[0] & fault,
            0
        );
        assert_eq!(read_words(device, queue, &coupling.indirect), vec![0, 1, 1]);
        let after: Vec<_> = soil
            .checkpoint_buffers()
            .iter()
            .map(|b| read_words(device, queue, b))
            .collect();
        assert_eq!(after, particle_before);
        assert!(backup.rollback(&mut soil).unwrap());
        assert_eq!(read_all(&soil), before);
        assert_ne!(
            read_words(device, queue, coupling.status_buffer())[0] & fault,
            0
        );
    }
}

#[test]
#[ignore = "requires real GPU native atomics"]
fn all_fixed_grid_faces_project_machine_and_particles_with_external_impulse() {
    fixed_grid_faces(&[0]);
}

#[test]
#[ignore = "requires real native GPU; active/compact independent mass and boundary projection"]
fn active_mass_coupling_fixed_faces() {
    fixed_grid_faces(&[1, 2]);
}

#[test]
#[ignore = "requires native GPU; unilateral active-grid fault and capacity lifecycle"]
fn active_unilateral_coupling_lifecycle() {
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mass = data(&engine, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]);
    let jac = data(
        &engine,
        &[
            1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
    );
    let free = data(&engine, &[0.; 3]);
    let config = MpmUnilateralConfig {
        node_capacity: 128,
        search_distance: 0.05,
        max_iterations: 64,
    };
    let prepare = |soil: &mut MpmSimulation, budget| {
        PreparedMpmUnilateralCoupling::new(
            engine.clone(),
            soil,
            MpmUnilateralInputs {
                mass: &mass,
                jacobians: &jac,
                free_rates: &free,
                dofs: 3,
                max_owned_bytes: budget,
            },
            config,
        )
    };
    for native in [false, true] {
        let mut soil = fixture_layout(&engine, native, 1, [0.; 3], 2, 17, 1);
        let mut coupling = prepare(&mut soil, 1_000_000).unwrap();
        let delta = data(&engine, &[7., 8., 9.]);
        let machine_status = data(&engine, &[0.]);
        let publication = coupling
            .prepare_machine_publication(SoilRatePublicationInputs {
                rates: &free,
                velocity_delta: &delta,
                machine_status: &machine_status,
                max_owned_bytes: 16,
            })
            .unwrap();
        let before = soil
            .checkpoint_buffers()
            .map(|b| read_words(device, queue, b));
        coupling.step(&mut soil, 0.0001).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        publication.encode(&mut encoder);
        queue.submit([encoder.finish()]);
        assert_eq!(
            read_words(device, queue, coupling.status_buffer())[0],
            MPM_COUPLING_PARTICLE_FAULT
        );
        assert_eq!(read_words(device, queue, &free), vec![0; 3]);
        assert_eq!(
            read_words(device, queue, &delta),
            [7.0f32, 8., 9.].map(f32::to_bits)
        );
        assert_eq!(read_words(device, queue, &machine_status), vec![32]);
        for (b, words) in soil.checkpoint_buffers().into_iter().zip(&before) {
            assert_eq!(&read_words(device, queue, b), words);
        }
        assert_eq!(
            soil.read_active_grid_status_with(device, queue)
                .unwrap()
                .completed_substeps,
            0
        );
        soil.reserve_sparse_tiles(125).unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        assert!(coupling.step(&mut soil, 0.0001).is_err());
        assert_eq!(traffic.finish().upload_bytes, 0);
        drop(publication);
        drop(coupling);
        let mut coupling = prepare(&mut soil, 1_000_000).unwrap();
        coupling.step(&mut soil, 0.0001).unwrap();
        assert_eq!(read_words(device, queue, coupling.status_buffer())[0], 0);
        assert_eq!(
            soil.read_active_grid_status_with(device, queue)
                .unwrap()
                .completed_substeps,
            1
        );
        drop(coupling);
        soil.set_particles(queue, &[]).unwrap();
        let mut coupling = prepare(&mut soil, 1_000_000).unwrap();
        coupling.step(&mut soil, 0.0001).unwrap();
        assert_eq!(read_words(device, queue, coupling.status_buffer())[0], 0);
        assert_eq!(
            read_words(device, queue, coupling.rates_buffer()),
            vec![0; 3]
        );
        let mut expected_bytes = None;
        for size in [17, 96, 1536] {
            let center = if size == 1536 {
                [0., 0., 150.]
            } else {
                [0.; 3]
            };
            let mut soil = fixture_layout(&engine, native, 1, center, 2, size, 125);
            let mut coupling = prepare(&mut soil, 1_000_000).unwrap();
            let bytes = coupling.owned_buffer_bytes();
            assert_eq!(
                bytes,
                PreparedMpmUnilateralCoupling::required_bytes_for_grid(&soil, |size| {
                    PreparedMpmUnilateralCoupling::required_bytes(config, 3, size, 1)
                })
                .unwrap()
            );
            assert_eq!(*expected_bytes.get_or_insert(bytes), bytes);
            let allocation = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            assert!(prepare(&mut soil, bytes - 1).is_err());
            assert_eq!(allocation.finish().observations, 0);
            let allocation = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
            coupling.step(&mut soil, 0.0001).unwrap();
            assert_eq!(allocation.finish().observations, 0);
            assert_eq!(traffic.finish().readback_bytes, 0);
            assert_eq!(read_words(device, queue, coupling.status_buffer())[0], 0);
            assert!(
                !soil
                    .read_active_grid_status_with(device, queue)
                    .unwrap()
                    .faulted()
            );
            let velocity = soil.read_velocities_with(device, queue).unwrap();
            assert!(velocity.iter().flatten().all(|v| v.is_finite()));
            eprintln!(
                "unilateral compact native={native} world={size} owned_bytes={bytes}: bounded storage, budget, successful publication pass"
            );
        }
    }
}

#[test]
#[ignore = "requires real native GPU; compact coupling faults, growth, empty state and domain scaling"]
fn active_mass_coupling_lifecycle() {
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mass = data(&engine, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]);
    let jac = data(
        &engine,
        &[
            1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
    );
    let free = data(&engine, &[0.; 3]);
    let prepare = |soil: &mut MpmSimulation| {
        PreparedMpmMassCoupling::new(
            engine.clone(),
            soil,
            MpmMassCouplingInputs {
                mass: &mass,
                jacobians: &jac,
                free_rates: &free,
                dofs: 3,
                nodes_per_body: 128,
                max_owned_bytes: 100_000,
            },
        )
        .unwrap()
    };
    for native in [false, true] {
        let mut soil = fixture_layout(&engine, native, 1, [0.; 3], 2, 17, 1);
        let mut coupling = prepare(&mut soil);
        let delta = data(&engine, &[7., 8., 9.]);
        let machine_status = data(&engine, &[0.]);
        let publication = coupling
            .prepare_machine_publication(SoilRatePublicationInputs {
                rates: &free,
                velocity_delta: &delta,
                machine_status: &machine_status,
                max_owned_bytes: 16,
            })
            .unwrap();
        let before = soil
            .checkpoint_buffers()
            .map(|b| read_words(device, queue, b));
        coupling.step(&mut soil, 0.0001).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        publication.encode(&mut encoder);
        queue.submit([encoder.finish()]);
        assert_eq!(
            read_words(device, queue, coupling.status_buffer()),
            vec![MPM_COUPLING_PARTICLE_FAULT]
        );
        assert_eq!(read_words(device, queue, &free), vec![0; 3]);
        assert_eq!(
            read_words(device, queue, &delta),
            [7.0f32, 8., 9.].map(f32::to_bits)
        );
        assert_eq!(read_words(device, queue, &machine_status), vec![32]);
        let status = soil.read_active_grid_status_with(device, queue).unwrap();
        assert!(status.overflow);
        assert_eq!(status.completed_substeps, 0);
        for (words, buffer) in before.iter().zip(soil.checkpoint_buffers()) {
            assert_eq!(words, &read_words(device, queue, buffer));
        }
        soil.reserve_sparse_tiles(125).unwrap();
        assert!(coupling.step(&mut soil, 0.0001).is_err());
        drop(publication);
        drop(coupling);
        let mut coupling = prepare(&mut soil);
        coupling.step(&mut soil, 0.0001).unwrap();
        assert_eq!(read_words(device, queue, coupling.status_buffer()), vec![0]);
        let status = soil.read_active_grid_status_with(device, queue).unwrap();
        assert!(!status.faulted());
        assert_eq!(status.completed_substeps, 1);
        drop(coupling);
        soil.set_particles(queue, &[]).unwrap();
        queue.write_buffer(&free, 0, bytemuck::cast_slice(&[0.1f32, -0.2, 0.3]));
        let mut coupling = prepare(&mut soil);
        coupling.step(&mut soil, 0.0001).unwrap();
        assert_eq!(read_words(device, queue, coupling.status_buffer()), vec![0]);
        assert_eq!(
            read_words(device, queue, coupling.rates_buffer()),
            [0.1f32, -0.2, 0.3].map(f32::to_bits)
        );
        assert_eq!(
            read_words(device, queue, coupling.body_spans_buffer())[1],
            0
        );
        queue.write_buffer(&free, 0, bytemuck::cast_slice(&[0.0f32; 3]));
        let mut bytes = None;
        for size in [17, 96, 1536] {
            let center = if size == 1536 {
                [93.0, 0.0, 0.0]
            } else {
                [0.; 3]
            };
            let mut soil = fixture_layout(&engine, native, 1, center, 2, size, 125);
            let mut coupling = prepare(&mut soil);
            assert_eq!(
                *bytes.get_or_insert(coupling.owned_buffer_bytes()),
                coupling.owned_buffer_bytes()
            );
            assert_eq!(
                coupling.owned_buffer_bytes(),
                PreparedMpmMassCoupling::required_bytes(&soil, 3, 1, 128).unwrap()
            );
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
            coupling.step(&mut soil, 0.0001).unwrap();
            assert_eq!(allocations.finish().observations, 0);
            assert_eq!(traffic.finish().readback_bytes, 0);
            assert_eq!(read_words(device, queue, coupling.status_buffer()), vec![0]);
            let rates = read_words(device, queue, coupling.rates_buffer())
                .into_iter()
                .map(f32::from_bits)
                .collect::<Vec<_>>();
            let expected = [1.0 / 3.0, soil.cfg.gravity_y * 0.0001 / 3.0, 0.0];
            for (actual, expected) in rates.iter().zip(expected) {
                assert!((actual - expected).abs() <= 2e-5);
            }
            let spans = read_words(device, queue, coupling.body_spans_buffer());
            let nodes = read_words(device, queue, coupling.contact_nodes_buffer());
            let mass: f64 = nodes
                .chunks_exact(8)
                .take(spans[1] as usize)
                .map(|n| f64::from(f32::from_bits(n[3])))
                .sum();
            assert!((mass - 0.5).abs() <= 2e-5);
            eprintln!(
                "active mass coupling native={native} world={size} owned_bytes={} contact_nodes={}",
                coupling.owned_buffer_bytes(),
                spans[1]
            );
        }
    }
}

fn fixed_grid_faces(layouts: &[u32]) {
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mass = data(&engine, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]);
    let jac = data(
        &engine,
        &[
            1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ],
    );
    let free = data(&engine, &[0.; 3]);
    let dt = 0.0001;
    for (&layout, native) in layouts
        .iter()
        .flat_map(|layout| [false, true].map(|native| (layout, native)))
    {
        for face in 0..6 {
            let axis = face / 2;
            let inward = if face % 2 == 0 { 1.0f32 } else { -1.0 };
            let mut center = [0.; 3];
            center[axis] = if inward > 0.0 { -0.875 } else { 0.75 };
            let mut velocity = [0.; 3];
            velocity[axis] = -inward;
            velocity[(axis + 1) % 3] = 0.2;
            let mut free_rates = [0.; 3];
            free_rates[axis] = -2.0 * inward;
            queue.write_buffer(&free, 0, bytemuck::cast_slice(&free_rates));
            let mut soil = fixture_layout(&engine, native, 65, center, layout, 16, 64);
            soil.set_particles_with_velocity(queue, &vec![center; 65], &vec![velocity; 65])
                .unwrap();
            let mut coupling = PreparedMpmMassCoupling::new(
                engine.clone(),
                &mut soil,
                MpmMassCouplingInputs {
                    mass: &mass,
                    jacobians: &jac,
                    free_rates: &free,
                    dofs: 3,
                    nodes_per_body: 128,
                    max_owned_bytes: 32_000,
                },
            )
            .unwrap();
            let allocation = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
            coupling.step(&mut soil, dt).unwrap();
            assert_eq!(allocation.finish().observations, 0);
            assert_eq!(transfers.finish().readback_bytes, 0);
            assert_eq!(
                read_words(device, queue, coupling.status_buffer()),
                vec![0],
                "face {face}, native={native}"
            );
            let read_floats = |buffer: &wgpu::Buffer| {
                read_words(device, queue, buffer)
                    .into_iter()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>()
            };
            let spans = read_words(device, queue, &coupling.spans);
            let nodes = read_floats(&coupling.nodes);
            let mut soil_mass = 0.0f64;
            let mut momentum = [0.0f64; 3];
            let mut kinetic = 2.0f64;
            let mut boundary_nodes = 0;
            for node in nodes.chunks_exact(8).take(spans[1] as usize) {
                soil_mass += f64::from(node[3]);
                for k in 0..3 {
                    momentum[k] += f64::from(node[4 + k]);
                    kinetic += 0.5 * f64::from(node[4 + k]).powi(2) / f64::from(node[3]);
                }
                if node[7] as u32 & (1 << face) != 0 {
                    boundary_nodes += 1;
                }
            }
            assert!(boundary_nodes > 0);
            assert!((soil_mass - 0.5).abs() < 2e-5);
            let rates = read_floats(coupling.rates_buffer());
            let ground = read_floats(coupling.boundary_impulses_buffer());
            let impulses = read_floats(coupling.impulses_buffer());
            let wrenches = read_floats(&soil.rigid_wrench);
            for k in 0..3 {
                let expected = if k == axis {
                    0.0
                } else {
                    (momentum[k] + f64::from(free_rates[k])) / (1.0 + soil_mass)
                };
                assert!(
                    (f64::from(rates[k]) - expected).abs() < 2e-5,
                    "face {face}, rates={rates:?}"
                );
                assert!((impulses[k] - (rates[k] - free_rates[k])).abs() < 2e-5);
                assert!((wrenches[k] * dt - impulses[k]).abs() < 2e-5);
                let external = (1.0 + soil_mass) * f64::from(rates[k])
                    - f64::from(free_rates[k])
                    - momentum[k];
                assert!((f64::from(ground[k]) - external).abs() < 2e-5);
            }
            assert!(ground[axis] * inward > 0.0);
            let velocities = read_floats(&soil.p_vel);
            for particle in velocities.chunks_exact(4).take(65) {
                for k in 0..3 {
                    assert!((particle[k] - rates[k]).abs() < 2e-5);
                }
                assert!(particle[axis] * inward >= -2e-5);
            }
            let final_kinetic =
                0.5 * (1.0 + soil_mass) * rates.iter().map(|v| f64::from(*v).powi(2)).sum::<f64>();
            assert!(final_kinetic <= kinetic + 2e-5);
        }
    }
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn empty_soil_overlap_ownership_and_budget_rejection() {
    let engine = Arc::new(Engine::new_gpu().unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mass = data(&engine, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]);
    let single = [
        1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    ];
    let jac = data(&engine, &single.repeat(2));
    let free = data(&engine, &[0.1, -0.2, 0.3]);
    let mut soil = fixture(&engine, false, 2, [0.; 3]);
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        PreparedMpmMassCoupling::new(
            engine.clone(),
            &mut soil,
            MpmMassCouplingInputs {
                mass: &mass,
                jacobians: &jac,
                free_rates: &free,
                dofs: 3,
                nodes_per_body: 128,
                max_owned_bytes: 20_755,
            }
        )
        .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    soil.set_particles(queue, &[]).unwrap();
    let mut empty = PreparedMpmMassCoupling::new(
        engine.clone(),
        &mut soil,
        MpmMassCouplingInputs {
            mass: &mass,
            jacobians: &jac,
            free_rates: &free,
            dofs: 3,
            nodes_per_body: 128,
            max_owned_bytes: 32_000,
        },
    )
    .unwrap();
    empty.step(&mut soil, 0.0001).unwrap();
    assert_eq!(read_words(device, queue, empty.status_buffer()), vec![0]);
    assert_eq!(
        read_words(device, queue, empty.rates_buffer()),
        [0.1f32, -0.2, 0.3].map(f32::to_bits)
    );
    assert!(soil.read_positions(&engine).unwrap().is_empty());
    assert!(
        soil.read_rigid_wrenches(&engine)
            .unwrap()
            .iter()
            .all(|(f, t)| f.iter().chain(t).all(|v| *v == 0.0))
    );
    drop(empty);
    let mut soil = fixture(&engine, false, 2, [0.; 3]);
    let solid = MpmPrescribedSolid {
        shape: MpmSolidShape::Box {
            half_extents: [0.4; 3],
        },
        center: [0.; 3],
        orientation: [0., 0., 0., 1.],
        linear_velocity: [0.; 3],
        angular_velocity: [0.; 3],
    };
    soil.set_prescribed_solids(queue, &[solid, solid]).unwrap();
    let mut coupling = PreparedMpmMassCoupling::new(
        engine.clone(),
        &mut soil,
        MpmMassCouplingInputs {
            mass: &mass,
            jacobians: &jac,
            free_rates: &free,
            dofs: 3,
            nodes_per_body: 128,
            max_owned_bytes: 64_000,
        },
    )
    .unwrap();
    coupling.step(&mut soil, 0.0001).unwrap();
    assert_eq!(read_words(device, queue, coupling.status_buffer()), vec![0]);
    let spans = read_words(device, queue, coupling.body_spans_buffer());
    assert_eq!(spans[0], 0);
    assert!(spans[1] > 0);
    assert_eq!(&spans[2..], &[128, 0]);
    let impulses = read_words(device, queue, coupling.impulses_buffer());
    assert!(impulses[8..].iter().all(|&v| f32::from_bits(v) == 0.0));
}
