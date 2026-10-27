use crate::soil_newton_gpu::{
    ExperimentalSoilNewtonGpu as Solver, SoilNewtonCapacity, SoilNewtonInputs,
};
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

#[test]
#[ignore = "requires protected GPU and MOLLA_NEWTON_NORM_COMPARE=publication-active"]
fn newton_active_publication_gpu_lifecycle() {
    assert_eq!(
        std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref(),
        Ok("publication-active")
    );
    lifecycle(false, 1, &[0, 1, 63, 64, 65, 255, 256, 257, 601, 4096]);
}

#[test]
#[ignore = "requires protected GPU and MOLLA_NEWTON_NORM_COMPARE=publication-zero"]
fn newton_zero_publication_gpu_lifecycle() {
    assert_eq!(
        std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref(),
        Ok("publication-zero")
    );
    lifecycle(true, 1, &[0, 1, 63, 64, 65, 255, 256, 257, 601, 4096]);
}

#[test]
#[ignore = "requires protected GPU and MOLLA_NEWTON_NORM_COMPARE=publication-products"]
fn newton_publication_products_gpu_lifecycle() {
    assert_eq!(
        std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref(),
        Ok("publication-products")
    );
    for width in [1, 3, 10, 31, 32] {
        lifecycle(true, width, &[0, 1, 15, 16, 17, 255, 256, 257, 601, 4096]);
    }
}

#[test]
#[cfg(feature = "retained-local-newton")]
#[ignore = "requires protected GPU and MOLLA_NEWTON_NORM_COMPARE=publication-adaptive"]
fn newton_publication_adaptive_gpu_lifecycle() {
    assert_eq!(
        std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref(),
        Ok("publication-adaptive")
    );
    for width in [1, 3, 10, 31, 32] {
        let mut counts = vec![0, 1, 16, 256, 4096];
        for axes in [1, 3] {
            let tile = 2080 / (axes * width);
            counts.extend([tile - 1, tile, tile + 1]);
        }
        counts.sort_unstable();
        counts.dedup();
        lifecycle(true, width, &counts);
    }
}

#[test]
#[cfg(feature = "retained-local-newton")]
#[ignore = "requires protected GPU and MOLLA_NEWTON_NORM_COMPARE=publication-validation"]
fn newton_publication_validation_gpu_lifecycle() {
    assert_eq!(
        std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref(),
        Ok("publication-validation")
    );
    for width in [1, 3, 10, 31, 32] {
        lifecycle(true, width, &[0, 1, 15, 16, 17, 255, 256, 257, 601, 4096]);
    }
}

#[test]
#[ignore = "requires protected GPU and MOLLA_NEWTON_NORM_COMPARE=publication-energy"]
fn newton_publication_energy_gpu_lifecycle() {
    assert_eq!(
        std::env::var("MOLLA_NEWTON_NORM_COMPARE").as_deref(),
        Ok("publication-energy")
    );
    for width in [1, 3, 10, 31, 32] {
        lifecycle(true, width, &[0, 1, 15, 16, 17, 255, 256, 257, 601, 4096]);
    }
}

fn lifecycle(zero: bool, width: usize, counts: &[usize]) {
    let free_cases = [
        [0.0, -0.0],
        [-0.0, -0.0],
        [1.0, 1.0],
        [1.0, -1.0],
        [1.0, 2.0f32.powi(-24)],
        [1.0e20, 1.0],
        [f32::MIN_POSITIVE, f32::from_bits(1)],
        [-3.0, 2.0],
        [1.0e-20, -1.0e-20],
        [f32::MAX, -f32::MAX],
        [0.0, 0.0],
    ];
    let engine =
        molla_compute::Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let storage = |data: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("publication lifecycle fixture"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    const C: usize = 4096;
    let mut identity = vec![0.0; width * width];
    for i in 0..width {
        identity[i * width + i] = 1.0;
    }
    let mass = storage(&identity);
    let rates = storage(&vec![0.0; 2 * width]);
    let upstream = storage(&[0.0]);
    let rows = storage(&vec![0.0; C * 4 * width]);
    let contacts = storage(&vec![0.0; C * 16]);
    let solvers = [false, true].map(|candidate| {
        Solver::new_norm_tree_edge_probe(
            &engine,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &rates,
                point_rows: &rows,
                point_response: &rows,
                contacts: &contacts,
                upstream_status: &upstream,
            },
            SoilNewtonCapacity {
                dofs: width as u32,
                contacts: C as u32,
                newton_iterations: 64,
                krylov_iterations: 64,
                max_owned_bytes: u64::MAX,
            },
            candidate,
        )
        .unwrap()
    });
    let mut cases = 0;
    for &count in counts {
        for mode in 0..(7 + if zero { free_cases.len() } else { 0 }) {
            if count == 0 && matches!(mode, 2 | 3 | 4 | 6) {
                continue;
            }
            if count == C && mode == 1 {
                continue;
            }
            let mut nodes = vec![0.0f32; C * 16];
            let mut jacobian = vec![0.0f32; C * 4 * width];
            let mut states = vec![f32::NAN; C * 16];
            let mut active = vec![false; C];
            let impulse = 1.0 / (count * width + 1) as f64;
            for slot in 0..count {
                let c = slot * C / count;
                active[c] = true;
                nodes[c * 16 + 1] = -1.0;
                nodes[c * 16 + 3] = 1.0;
                nodes[c * 16 + 5] = 1.0;
                for i in 0..width {
                    jacobian[(c * width + i) * 4 + 1] = 1.0;
                }
                states[c * 16..c * 16 + 16].fill(0.0);
                states[c * 16 + 1] = impulse as f32;
                states[c * 16 + 9] = (impulse - impulse as f32 as f64) as f32;
            }
            match mode {
                1 => nodes[(C - 1) * 16 + 3] = f32::NAN,
                2 => states[1] = f32::NAN,
                3 => nodes[8] = -1.0,
                4 => {
                    states[1] = -1.0;
                    states[9] = 0.0;
                }
                6 => {
                    states[1] += 0.25;
                }
                _ => {}
            }
            let free = if mode >= 7 {
                for slot in 0..count {
                    let c = slot * C / count;
                    states[c * 16..c * 16 + 16].fill(0.0);
                    nodes[c * 16 + 1] = 1.0;
                    jacobian[c * width * 4..(c + 1) * width * 4].fill(0.0);
                    if mode == 7 {
                        jacobian[c * width * 4..c * width * 4 + 3].copy_from_slice(&[
                            3.0e38,
                            -3.0e38,
                            f32::MIN_POSITIVE,
                        ]);
                    }
                    if mode == 17 {
                        states[c * 16..c * 16 + 3].fill(-0.0);
                        jacobian[c * width * 4..c * width * 4 + 3].copy_from_slice(&[
                            -0.0,
                            f32::from_bits(1),
                            -f32::from_bits(1),
                        ]);
                    }
                }
                free_cases[mode - 7]
            } else {
                [0.0; 2]
            };
            queue.write_buffer(&rates, 0, bytemuck::cast_slice(&free.repeat(width)));
            queue.write_buffer(&contacts, 0, bytemuck::cast_slice(&nodes));
            queue.write_buffer(&rows, 0, bytemuck::cast_slice(&jacobian));
            queue.write_buffer(&upstream, 0, bytemuck::bytes_of(&u32::from(mode == 5)));
            let mut control = None;
            for solver in &solvers {
                for _ in 0..2 {
                    queue.write_buffer(solver.state_buffer(), 0, bytemuck::cast_slice(&states));
                    let mut encoder = device.create_command_encoder(&Default::default());
                    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
                    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
                    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
                    solver.encode(&mut encoder);
                    queue.submit([encoder.finish()]);
                    let a = allocations.finish();
                    let t = traffic.finish();
                    let d = dispatch.finish();
                    assert!(!a.invalid && !t.overflowed && !d.overflowed);
                    assert_eq!(
                        (
                            a.observations,
                            t.upload_bytes,
                            t.readback_bytes,
                            d.encoded_dispatches
                        ),
                        (0, 0, 0, 1)
                    );
                    let outputs = capture::read_many(
                        &engine,
                        &[
                            solver.outputs_buffer(),
                            solver.rates_buffer(),
                            solver.publication_status_buffer(),
                            solver.state_buffer(),
                            solver.status_buffer(),
                        ],
                    );
                    if let Some(old) = &control {
                        assert_eq!(&outputs, old, "count={count} mode={mode}");
                    } else {
                        control = Some(outputs.clone());
                    }
                    let status: [u32; 8] = bytemuck::pod_read_unaligned(&outputs[2]);
                    if mode == 16 {
                        assert_ne!(status[0], 0, "count={count} mode={mode}");
                        assert!(outputs[0].iter().chain(&outputs[1]).all(|&b| b == 0));
                    } else if mode >= 7 {
                        assert_eq!(status[0], 0, "count={count} mode={mode}");
                        assert_eq!(status[2], count as u32);
                        let expected = free[0] as f64 + free[1] as f64;
                        for bytes in outputs[1].chunks_exact(8) {
                            let q: [f32; 2] = bytemuck::pod_read_unaligned(bytes);
                            assert!(
                                (q[0] as f64 + q[1] as f64 - expected).abs()
                                    <= 1e-6 * expected.abs().max(1.0),
                                "width={width} count={count} mode={mode} q={q:?} free={free:?}"
                            );
                        }
                        for (c, bytes) in outputs[0].chunks_exact(80).enumerate() {
                            let values: [f32; 20] = bytemuck::pod_read_unaligned(bytes);
                            assert!(values[..3].iter().all(|&v| v == 0.0));
                            if active[c] {
                                assert_eq!(values[5], 1.0);
                            } else {
                                assert!(bytes.iter().all(|&b| b == 0));
                            }
                        }
                    } else if mode == 0 {
                        assert_eq!(status[0], 0, "count={count}");
                        assert_eq!(status[2], count as u32);
                        let expected_rate = -(count as f64) * impulse;
                        for bytes in outputs[1].chunks_exact(8) {
                            let q: [f32; 2] = bytemuck::pod_read_unaligned(bytes);
                            assert!((q[0] as f64 + q[1] as f64 - expected_rate).abs() < 1e-6);
                        }
                        for (c, bytes) in outputs[0].chunks_exact(80).enumerate() {
                            if !active[c] {
                                assert!(bytes.iter().all(|&b| b == 0));
                                continue;
                            }
                            let values: [f32; 20] = bytemuck::pod_read_unaligned(bytes);
                            assert!((values[1] as f64 + impulse).abs() < 1e-6);
                            assert!((values[5] as f64 - (impulse - 1.0)).abs() < 1e-6);
                        }
                    } else {
                        assert_ne!(status[0], 0, "count={count} mode={mode}");
                        assert!(outputs[0].iter().chain(&outputs[1]).all(|&b| b == 0));
                    }
                }
            }
            cases += 1;
            println!(
                "publication-edge zero={zero} width={width} count={count} mode={mode} exact=true"
            );
        }
    }
    assert_eq!(
        cases,
        7 * counts.len() - 5
            + if zero {
                counts.len() * free_cases.len()
            } else {
                0
            }
    );
}
