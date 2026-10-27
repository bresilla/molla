use crate::soil_newton_gpu::{
    ExperimentalSoilNewtonGpu as Solver, SoilNewtonCapacity, SoilNewtonInputs,
};
use crate::soil_shared_wall_friction_gpu::capture;
use wgpu::util::DeviceExt;

fn x(c: usize, i: usize, paired: bool) -> f64 {
    ((c * 3 + i * 5) % 17) as f64 / 16.0 - 0.5
        + if paired {
            (((c + i) % 5) as f64 - 2.0) / 268435456.0
        } else {
            0.0
        }
}

#[test]
#[ignore = "requires protected GPU; tiled row products and independent f64 oracle"]
fn newton_row_products_gpu_edges() {
    product_edges(false);
}

#[test]
#[ignore = "requires protected GPU; multi-workgroup products and f64 oracle"]
fn newton_distributed_gpu_edges() {
    product_edges(true);
}

enum ProductProbe {
    Resident(Solver),
    Distributed(crate::soil_newton_gpu::distributed::Probe),
}

impl ProductProbe {
    fn encode_reset(&self, encoder: &mut wgpu::CommandEncoder) {
        match self {
            Self::Resident(p) => p.encode_reset(encoder),
            Self::Distributed(p) => p.encode_reset(encoder),
        }
    }
    fn encode(&self, encoder: &mut wgpu::CommandEncoder) {
        match self {
            Self::Resident(p) => p.encode(encoder),
            Self::Distributed(p) => p.encode(encoder),
        }
    }
    fn status_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::Resident(p) => p.status_buffer(),
            Self::Distributed(p) => p.status_buffer(),
        }
    }
    fn state_buffer(&self) -> &wgpu::Buffer {
        match self {
            Self::Resident(p) => p.state_buffer(),
            Self::Distributed(p) => p.state_buffer(),
        }
    }
}

fn product_edges(distributed: bool) {
    let engine =
        molla_compute::Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let storage = |data: &[f32]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("row product edge fixture"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
    };
    const C: usize = 601;
    let contacts = storage(&vec![0.0; C * 16]);
    let upstream = storage(&[0.0]);
    for n in [1usize, 3, 32] {
        let mut rows = vec![0.0; C * n * 4];
        let mut response = rows.clone();
        for c in 0..C {
            for i in 0..n {
                for a in 0..3 {
                    rows[4 * (c * n + i) + a] =
                        (((c * 3 + i * 5 + a) % 17) as i32 - 8) as f32 / 128.0;
                    response[4 * (c * n + i) + a] =
                        (((c * 7 + i * 3 + a) % 13) as i32 - 6) as f32 / 256.0;
                }
            }
        }
        let row_buffer = storage(&rows);
        let response_buffer = storage(&response);
        let mass = storage(&vec![0.0; n * n]);
        let rates = storage(&vec![0.0; 2 * n]);
        let solvers = [false, true].map(|candidate| {
            let input = SoilNewtonInputs {
                mass: &mass,
                free_rates: &rates,
                point_rows: &row_buffer,
                point_response: &response_buffer,
                contacts: &contacts,
                upstream_status: &upstream,
            };
            let capacity = SoilNewtonCapacity {
                dofs: n as u32,
                contacts: C as u32,
                newton_iterations: 64,
                krylov_iterations: 64,
                max_owned_bytes: u64::MAX,
            };
            if distributed && candidate {
                ProductProbe::Distributed(
                    crate::soil_newton_gpu::distributed::Probe::new(&engine, input, capacity)
                        .unwrap(),
                )
            } else {
                ProductProbe::Resident(
                    Solver::new_row_products_edge_probe(&engine, input, capacity, candidate)
                        .unwrap(),
                )
            }
        });
        let counts: &[usize] = if distributed {
            &[0, 1, 9, 10, 11, 42, 43, 255, 256, 257, 301, C]
        } else {
            &[0, 1, 42, 43, 255, 256, 257, 301, C]
        };
        for &count in counts {
            queue.write_buffer(&contacts, 48, &(count as u32).to_le_bytes());
            let active: Vec<_> = (0..count).map(|s| (2 * s + 7) % C).collect();
            let mut expected = None;
            for (variant, solver) in solvers.iter().enumerate() {
                for repeat in 0..2 {
                    let mut encoder = device.create_command_encoder(&Default::default());
                    solver.encode_reset(&mut encoder);
                    solver.encode(&mut encoder);
                    queue.submit([encoder.finish()]);
                    let status = capture::read(&engine, solver.status_buffer());
                    assert_eq!(u32::from_le_bytes(status[..4].try_into().unwrap()), 0);
                    let bytes = capture::read(&engine, solver.state_buffer());
                    if let Some(old) = &expected {
                        assert_eq!(
                            &bytes, old,
                            "n={n} count={count} variant={variant} repeat={repeat}"
                        );
                    } else {
                        expected = Some(bytes.clone());
                    }
                    let values: Vec<f32> = bytes
                        .chunks_exact(4)
                        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                        .collect();
                    for paired in [false, true] {
                        let q: Vec<f64> = (0..n)
                            .map(|i| {
                                active
                                    .iter()
                                    .map(|&c| {
                                        (0..3)
                                            .map(|a| {
                                                -(response[4 * (c * n + i) + a] as f64)
                                                    * x(c, a, paired)
                                            })
                                            .sum::<f64>()
                                    })
                                    .sum()
                            })
                            .collect();
                        for c in 0..C {
                            let offset = c * 16 + if paired { 8 } else { 0 };
                            if !active.contains(&c) {
                                assert_eq!(
                                    &values[offset..offset + 8],
                                    &[17.0, 17.0, 17.0, 17.0, 19.0, 19.0, 19.0, 19.0]
                                );
                                continue;
                            }
                            assert_eq!(values[offset + 3].to_bits(), 0);
                            assert_eq!(values[offset + 7].to_bits(), 0);
                            let point: [f64; 3] = std::array::from_fn(|a| {
                                (0..n)
                                    .map(|i| rows[4 * (c * n + i) + a] as f64 * q[i])
                                    .sum()
                            });
                            for i in 0..6 {
                                let direct: f64 = (0..6)
                                    .map(|j| {
                                        let d = (((c + i * 3 + j) % 19) as f64 - 9.0) / 32.0
                                            + (((i + j) % 3) as f64 - 1.0) / 134217728.0;
                                        d * x(c, j, paired)
                                    })
                                    .sum();
                                let projected: f64 = (0..3)
                                    .map(|j| {
                                        let m = (((c + i + j * 7) % 13) as f64 - 6.0) / 64.0
                                            + (((i + j) % 3) as f64 - 1.0) / 134217728.0;
                                        m * point[j]
                                    })
                                    .sum();
                                let reference = direct + projected;
                                let actual = values[offset + i + usize::from(i >= 3)] as f64;
                                assert!(
                                    actual.is_finite()
                                        && (actual - reference).abs()
                                            <= 1e-6 * (1.0 + reference.abs()),
                                    "n={n} count={count} c={c} row={i} paired={paired} actual={actual} expected={reference}"
                                );
                            }
                        }
                    }
                }
            }
            eprintln!(
                "row-product edges distributed={distributed} n={n} active={count}: exact variants/repeats, padding/inactive preservation and independent f64 oracle pass"
            );
        }
    }
}
