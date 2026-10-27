use super::*;

pub(super) fn prepare_order(soil: &mut MpmSimulation, queue: &wgpu::Queue, gather: bool) {
    soil.prepare_parallel_particle_tile_ordering(queue, soil.cfg.max_particles)
        .unwrap();
    soil.set_particle_tile_ordering(true).unwrap();
    if gather {
        soil.prepare_deterministic_p2g().unwrap();
        soil.set_deterministic_p2g(true).unwrap();
    }
}

pub(super) fn coupling(
    engine: &Arc<Engine>,
    soil: &mut MpmSimulation,
    buffers: &[wgpu::Buffer; 3],
    unilateral: bool,
) -> FaultTestCoupling {
    let [mass, jacobians, free_rates] = buffers;
    if unilateral {
        FaultTestCoupling::Unilateral(
            PreparedMpmUnilateralCoupling::new(
                engine.clone(),
                soil,
                MpmUnilateralInputs {
                    mass,
                    jacobians,
                    free_rates,
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
                soil,
                MpmMassCouplingInputs {
                    mass,
                    jacobians,
                    free_rates,
                    dofs: 3,
                    nodes_per_body: 128,
                    max_owned_bytes: 4_000_000,
                },
            )
            .unwrap(),
        )
    }
}

pub(super) fn step(c: &mut FaultTestCoupling, soil: &mut MpmSimulation, dt: f32) -> Result<()> {
    match c {
        FaultTestCoupling::Sticking(c) => c.step(soil, dt),
        FaultTestCoupling::Unilateral(c) => c.step(soil, dt),
    }
}

fn rates(c: &FaultTestCoupling) -> &wgpu::Buffer {
    match c {
        FaultTestCoupling::Sticking(c) => c.rates_buffer(),
        FaultTestCoupling::Unilateral(c) => c.rates_buffer(),
    }
}

pub(super) fn inputs(engine: &Engine) -> [wgpu::Buffer; 3] {
    [
        data(engine, &[1., 0., 0., 0., 1., 0., 0., 0., 1.]),
        data(
            engine,
            &[
                1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            ],
        ),
        data(engine, &[0.; 3]),
    ]
}

fn expected_order(soil: &mut MpmSimulation, engine: &Engine) -> Vec<(u32, u32)> {
    let positions = soil.read_positions(engine).unwrap();
    let width = soil.cfg.grid_size.div_ceil(4);
    let mut pairs: Vec<_> = positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let tile: [u32; 3] = std::array::from_fn(|k| {
                ((p[k] - soil.cfg.grid_origin[k]) / soil.cfg.cell_size)
                    .floor()
                    .clamp(0., (soil.cfg.grid_size - 1) as f32) as u32
                    / 4
            });
            (tile[0] + width * (tile[1] + width * tile[2]), i as u32)
        })
        .collect();
    pairs.sort_unstable();
    pairs
}

#[test]
#[ignore = "requires native GPU and resource counters"]
fn ordered_projection_rebuilds_indices_and_preserves_feedback() {
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let buffers = inputs(&engine);
    for (native, gather) in [(false, false), (true, false), (true, true)] {
        for unilateral in [false, true] {
            let points: Vec<_> = (0..129)
                .map(|i| {
                    [
                        if i % 3 == 0 { -0.000002 } else { 0.05 },
                        if i % 2 == 0 { -0.03 } else { 0.03 },
                        if i % 5 == 0 { -0.03 } else { 0.03 },
                    ]
                })
                .collect();
            let mut reference = fixture(&engine, native, 129, [0.; 3]);
            let mut candidate = fixture(&engine, native, 129, [0.; 3]);
            for soil in [&mut reference, &mut candidate] {
                soil.set_particles_with_velocity(queue, &points, &vec![[1., 0., 0.]; 129])
                    .unwrap();
            }
            prepare_order(&mut candidate, queue, gather);
            let mut a = coupling(&engine, &mut reference, &buffers, unilateral);
            let mut b = coupling(&engine, &mut candidate, &buffers, unilateral);
            let initial_order = expected_order(&mut candidate, &engine);
            let mut order_changed = false;
            let mut max_error = 0.0f32;
            for dt in [0.0001, 0.0002, 0.00008] {
                let expected = expected_order(&mut candidate, &engine);
                order_changed |= expected != initial_order;
                queue.write_buffer(
                    &candidate.particle_order.as_ref().unwrap().params,
                    0,
                    &[0; 32],
                );
                queue.write_buffer(&candidate.p_material, 129 * 4, &[255; 129 * 4]);
                let reference_traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
                step(&mut a, &mut reference, dt).unwrap();
                let reference_traffic = reference_traffic.finish();
                let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
                let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
                step(&mut b, &mut candidate, dt).unwrap();
                let allocations = allocations.finish();
                let traffic = traffic.finish();
                assert!(!allocations.invalid && !traffic.overflowed);
                assert_eq!(allocations.observations, 0);
                assert_eq!(traffic.readback_bytes, 0);
                assert_eq!(traffic.upload_bytes, reference_traffic.upload_bytes + 32);
                assert_eq!(read_words(device, queue, a.status())[0], 0);
                assert_eq!(read_words(device, queue, b.status())[0], 0);
                let keys = read_words(
                    device,
                    queue,
                    &candidate.particle_order.as_ref().unwrap().keys,
                );
                let material = read_words(device, queue, &candidate.p_material);
                assert!(material[..129].iter().all(|&id| id == 0));
                assert_eq!(keys, expected.iter().map(|p| p.0).collect::<Vec<_>>());
                assert_eq!(
                    &material[129..],
                    expected.iter().map(|p| p.1).collect::<Vec<_>>()
                );
                for (left, right) in [
                    (&reference.p_pos, &candidate.p_pos),
                    (&reference.p_vel, &candidate.p_vel),
                    (&reference.p_c, &candidate.p_c),
                    (&reference.p_f, &candidate.p_f),
                    (&reference.p_jp, &candidate.p_jp),
                    (rates(&a), rates(&b)),
                ] {
                    for (left, right) in read_words(device, queue, left)
                        .into_iter()
                        .zip(read_words(device, queue, right))
                    {
                        let error = (f32::from_bits(left) - f32::from_bits(right)).abs();
                        assert!(
                            error.is_finite() && error <= 2e-5,
                            "native={native} gather={gather} unilateral={unilateral} error={error}"
                        );
                        max_error = max_error.max(error);
                    }
                }
            }
            assert!(order_changed, "fixture must cross a tile boundary");
            if gather {
                candidate.set_deterministic_p2g(false).unwrap();
            } else {
                candidate.set_particle_tile_ordering(false).unwrap();
            }
            let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
            assert!(step(&mut b, &mut candidate, 0.0001).is_err());
            assert_eq!(traffic.finish().upload_bytes, 0);
            println!(
                "native={native} gather={gather} unilateral={unilateral}: refreshed tile-crossing order, feedback/history max_error={max_error}, zero hot allocations/readbacks, stale mode rejected before upload"
            );
        }
    }
}

#[test]
#[ignore = "requires native GPU and resource counters"]
fn ordered_projection_empty_counts_and_capacity_guards() {
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let buffers = inputs(&engine);
    for (native, gather) in [(false, false), (true, false), (true, true)] {
        for unilateral in [false, true] {
            let mut soil = fixture(&engine, native, 129, [0.; 3]);
            prepare_order(&mut soil, queue, gather);
            for count in [0, 1, 129, 0] {
                soil.set_particles(queue, &vec![[0.; 3]; count]).unwrap();
                let mut owner = coupling(&engine, &mut soil, &buffers, unilateral);
                step(&mut owner, &mut soil, 0.0001).unwrap();
                assert_eq!(read_words(device, queue, owner.status())[0], 0);
                let keys = read_words(device, queue, &soil.particle_order.as_ref().unwrap().keys);
                assert!(keys[..count].iter().all(|&v| v == 42));
                assert!(keys[count..].iter().all(|&v| v == u32::MAX));
                assert_eq!(soil.read_positions(&engine).unwrap().len(), count);
                if count == 0 {
                    assert!(
                        read_words(device, queue, &soil.g_mass)
                            .iter()
                            .all(|&v| v == 0)
                    );
                    assert_eq!(read_words(device, queue, rates(&owner)), vec![0; 3]);
                }
                soil.set_particles(queue, &[]).unwrap();
                let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
                assert!(step(&mut owner, &mut soil, 0.0001).is_err());
                assert_eq!(traffic.finish().upload_bytes, 0);
            }
            if gather {
                soil.set_deterministic_p2g(false).unwrap();
            }
            soil.set_particle_tile_ordering(false).unwrap();
            soil.set_particles(queue, &[[0.; 3]]).unwrap();
            soil.prepare_parallel_particle_tile_ordering(queue, 1)
                .unwrap();
            soil.set_particle_tile_ordering(true).unwrap();
            if gather {
                soil.prepare_deterministic_p2g().unwrap();
                soil.set_deterministic_p2g(true).unwrap();
            }
            let mut owner = coupling(&engine, &mut soil, &buffers, unilateral);
            soil.set_particles(queue, &[[0.; 3]; 129]).unwrap();
            let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let error = step(&mut owner, &mut soil, 0.0001).unwrap_err();
            assert!(error.to_string().contains("ordering capacity"));
            assert!(PreparedMpmMassCoupling::validate_mode(&soil).is_err());
            assert_eq!(traffic.finish().upload_bytes, 0);
            println!(
                "native={native} gather={gather} unilateral={unilateral}: empty/count/reseed and ordering capacity guards pass"
            );
        }
    }
}
