use super::*;
use crate::impact_constraints::{Contact, Domain, solve_midpoint};
use molla_compute::Engine;

#[test]
#[ignore = "requires Vulkan GPU; use make test-primitives-impact-gpu"]
fn bounded_scalars_match_independent_coupled_box_oracle_and_validate_domains() {
    let mut fixtures = Vec::new();
    let mut expected = Vec::new();
    for mass in [0.01f32, 1.0, 100.0] {
        for speed in [-3.0f32, 0.0, 3.0] {
            for feedback in [0.0, 0.5 / mass, 2.0 / mass] {
                for limits in [
                    [-0.7f32, 0.7],
                    [-2.0, 2.0],
                    [0.2, 0.2],
                    [0.0, 0.7],
                    [-0.7, 0.0],
                    [0.0, 0.0],
                    [f32::NEG_INFINITY, f32::INFINITY],
                    [f32::NEG_INFINITY, 0.7],
                    [-0.7, f32::INFINITY],
                ] {
                    let bounds = limits.map(|value| value * mass);
                    let mut gram = vec![0.0; 36];
                    gram[0] = 2.0 / mass;
                    gram[3] = 0.5 / mass;
                    gram[18] = 0.5 / mass;
                    gram[21] = 1.5 / mass;
                    let a = f64::from(gram[0]) + f64::from(feedback);
                    let b = f64::from(gram[3]);
                    let d = f64::from(gram[21]);
                    let free = [speed, 0.0, 0.0, 0.4, 0.0, 0.0];
                    let first = ((-f64::from(speed) + b * f64::from(free[3]) / d)
                        / (a - b * b / d))
                        .clamp(f64::from(bounds[0]), f64::from(bounds[1]));
                    expected.push([first, (-f64::from(free[3]) - b * first) / d]);
                    fixtures.push(Fixture {
                        gram,
                        free: free.to_vec(),
                        blocks: vec![
                            ImpactGpuBlock {
                                contact: [bounds[0], bounds[1], feedback, 0.0],
                                metadata: [3, 0, 0, 0],
                            },
                            ImpactGpuBlock {
                                contact: [0.0; 4],
                                metadata: [1, 0, 0, 0],
                            },
                        ],
                    });
                }
            }
        }
    }
    let harness = Harness::with_tolerance(&fixtures, 4096, 1e-6);
    let (status, result) = harness.run(2);
    assert_eq!(status, 0);
    for ((fixture, expected), actual) in fixtures.iter().zip(expected).zip(result.chunks_exact(2)) {
        for i in 0..2 {
            let bound = 2e-5 * (1.0 + expected[i].abs());
            assert!(
                (f64::from(actual[i].impulse[0]) - expected[i]).abs() < bound,
                "free={:?} blocks={:?} row={i} actual={} expected={} bound={bound}",
                fixture.free,
                fixture.blocks,
                actual[i].impulse[0],
                expected[i]
            );
            assert_eq!(actual[i].impulse[1..], [0.0; 3]);
            for lane in 0..3 {
                let row = 3 * i + lane;
                let velocity = f64::from(fixture.free[row])
                    + (0..2)
                        .map(|j| {
                            f64::from(fixture.gram[row * 6 + 3 * j])
                                * f64::from(actual[j].impulse[0])
                        })
                        .sum::<f64>();
                assert!((f64::from(actual[i].velocity[lane]) - velocity).abs() < 3e-5);
            }
        }
    }
    let fixture = Fixture {
        gram: vec![0.0; 9],
        free: vec![7.0, 0.0, 0.0],
        blocks: vec![ImpactGpuBlock {
            contact: [0.3, 0.3, 0.0, 0.0],
            metadata: [3, 0, 0, 0],
        }],
    };
    let harness = Harness::new(&[fixture], 64);
    let (status, result) = harness.run(1);
    assert_eq!(status, 0);
    assert_eq!(result[0].impulse[0], 0.3);
    assert_eq!(result[0].velocity[0], 7.0);
    for bounds in [
        [1.0, -1.0],
        [f32::NAN, 1.0],
        [0.0, f32::NAN],
        [f32::INFINITY, f32::INFINITY],
        [f32::NEG_INFINITY, f32::NEG_INFINITY],
    ] {
        harness.write(
            2,
            &[ImpactGpuBlock {
                contact: [bounds[0], bounds[1], 0.0, 0.0],
                metadata: [3, 0, 0, 0],
            }],
        );
        harness.write(5, &[0u32]);
        let (status, _) = harness.run(1);
        assert_ne!(status & IMPACT_FAULT_INPUT, 0);
    }
    eprintln!(
        "243 coupled box/bilateral feedback systems plus prescribed null-response effort and invalid domains passed"
    );
}

#[derive(Clone)]
struct Fixture {
    gram: Vec<f32>,
    free: Vec<f32>,
    blocks: Vec<ImpactGpuBlock>,
}

fn single(mass: f32, coupling: [f32; 2], mu: f32, restitution: f32, slip: f32) -> Fixture {
    let [b, c] = coupling;
    Fixture {
        gram: [1., b, c, b, 1., 0., c, 0., 1.].map(|v| v / mass).to_vec(),
        free: vec![-1., slip, -0.7 * slip],
        blocks: vec![ImpactGpuBlock {
            contact: [1.5 * mu, mu, restitution, -1.],
            metadata: [0, 1, 0, 0],
        }],
    }
}

fn buffer(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("impact test input"),
            size: bytes.len().max(4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }
    );
    molla_core::gpu_write_buffer!(queue, &buffer, 0, bytes);
    buffer
}

struct Harness {
    engine: Engine,
    solver: ResidentImpactConstraints,
    inputs: Vec<wgpu::Buffer>,
    blocks: usize,
    islands: usize,
}

impl Harness {
    fn new(fixtures: &[Fixture], iterations: u32) -> Self {
        Self::with_tolerance(fixtures, iterations, 2e-5)
    }

    fn with_tolerance(fixtures: &[Fixture], iterations: u32, tolerance: f32) -> Self {
        let engine = Engine::new_gpu().unwrap();
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let islands = fixtures.len();
        let blocks = fixtures[0].blocks.len();
        assert!(fixtures.iter().all(|f| f.blocks.len() == blocks));
        let grams: Vec<_> = fixtures
            .iter()
            .flat_map(|f| f.gram.iter().copied())
            .collect();
        let free: Vec<_> = fixtures
            .iter()
            .flat_map(|f| f.free.iter().copied())
            .collect();
        let data: Vec<_> = fixtures
            .iter()
            .flat_map(|f| f.blocks.iter().copied())
            .collect();
        let counts = vec![blocks as u32; islands];
        let inputs: Vec<_> = [
            bytemuck::cast_slice(&grams),
            bytemuck::cast_slice(&free),
            bytemuck::cast_slice(&data),
            bytemuck::cast_slice(&counts),
            bytemuck::bytes_of(&(islands as u32)),
            bytemuck::bytes_of(&0u32),
        ]
        .into_iter()
        .map(|bytes| buffer(device, queue, bytes))
        .collect();
        let solver = ResidentImpactConstraints::new(
            device,
            &inputs[0],
            &inputs[1],
            &inputs[2],
            &inputs[3],
            &inputs[4],
            &inputs[5],
            ImpactGpuConfig {
                island_capacity: islands as u32,
                blocks_per_island: blocks as u32,
                max_iterations: iterations,
                velocity_tolerance: tolerance,
            },
        )
        .unwrap();
        Self {
            engine,
            solver,
            inputs,
            blocks,
            islands,
        }
    }

    fn write<T: bytemuck::Pod>(&self, index: usize, values: &[T]) {
        molla_core::gpu_write_buffer!(
            self.engine.queue().unwrap(),
            &self.inputs[index],
            0,
            bytemuck::cast_slice(values),
        );
    }

    fn run(&self, repeats: usize) -> (u32, Vec<ImpactGpuSolution>) {
        let device = self.engine.device().unwrap();
        let size = (self.blocks * self.islands * 48) as u64;
        let readback = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("impact test readback"),
                size: size + 4 + self.islands as u64 * 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        for _ in 0..repeats {
            self.solver.encode(&mut encoder);
        }
        encoder.copy_buffer_to_buffer(self.solver.solution_buffer(), 0, &readback, 0, size);
        encoder.copy_buffer_to_buffer(&self.inputs[5], 0, &readback, size, 4);
        encoder.copy_buffer_to_buffer(
            self.solver.diagnostic_buffer(),
            0,
            &readback,
            size + 4,
            self.islands as u64 * 16,
        );
        self.engine.queue().unwrap().submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = molla_core::gpu_readback!(readback.slice(..));
        let status = bytemuck::pod_read_unaligned(&bytes[size as usize..size as usize + 4]);
        if status != 0 {
            for (i, bytes) in bytes[size as usize + 4..].chunks_exact(16).enumerate() {
                let diagnostic = bytemuck::pod_read_unaligned::<[u32; 4]>(bytes);
                if diagnostic[0] != 0 {
                    eprintln!(
                        "island={i}, diagnostic={diagnostic:?}, residual={}",
                        f32::from_bits(diagnostic[2])
                    );
                }
            }
        }
        assert_eq!(self.solver.resource_counts(), [4, 1, 1]);
        (
            status,
            bytes[..size as usize]
                .chunks_exact(48)
                .map(bytemuck::pod_read_unaligned)
                .collect(),
        )
    }
}

fn check(f: &Fixture, results: &[ImpactGpuSolution]) {
    let n = f.free.len();
    let lambda: Vec<_> = results
        .iter()
        .flat_map(|s| s.impulse[..3].iter().map(|&v| v as f64))
        .collect();
    let after: Vec<_> = (0..n)
        .map(|i| {
            f.free[i] as f64
                + (0..n)
                    .map(|j| f.gram[i * n + j] as f64 * lambda[j])
                    .sum::<f64>()
        })
        .collect();
    let mut indices = Vec::new();
    let mut domains = Vec::new();
    for (b, block) in f.blocks.iter().enumerate() {
        let r = 3 * b;
        assert!(
            results[b]
                .impulse
                .iter()
                .chain(&results[b].velocity)
                .all(|v| v.is_finite())
        );
        assert!(f32::from_bits(results[b].diagnostics[3]) <= 2e-5);
        for i in 0..3 {
            assert!((results[b].velocity[i] as f64 - after[r + i]).abs() < 3e-5);
        }
        indices.push(r);
        match block.metadata[0] {
            0 => {
                indices.extend([r + 1, r + 2]);
                domains.push(Domain::Contact(Contact {
                    static_friction: block.contact[0] as f64,
                    dynamic_friction: block.contact[1] as f64,
                    restitution: block.contact[2] as f64,
                    incoming_normal_velocity: block.contact[3] as f64,
                    moving: block.metadata[1] != 0,
                    base_normal_impulse: 0.,
                }));
                assert!(lambda[r] >= 0.);
                let normal_error = after[r] + block.contact[2] as f64 * block.contact[3] as f64;
                assert!(normal_error >= -3e-5 && (lambda[r] == 0. || normal_error.abs() < 3e-5));
                let mu = block.contact[results[b].diagnostics[0] as usize] as f64;
                let radius = mu * lambda[r];
                let tangent = lambda[r + 1].hypot(lambda[r + 2]);
                assert!(tangent <= radius * (1. + 3e-6) + 1e-12);
                let average = [
                    (after[r + 1] + f.free[r + 1] as f64) * 0.5,
                    (after[r + 2] + f.free[r + 2] as f64) * 0.5,
                ];
                if radius > 0. {
                    let multiplier = if tangent > 0. {
                        (-(average[0] * lambda[r + 1] + average[1] * lambda[r + 2]) / tangent)
                            .max(0.)
                    } else {
                        0.
                    };
                    let error = if tangent > 0. {
                        (average[0] + multiplier * lambda[r + 1] / tangent)
                            .hypot(average[1] + multiplier * lambda[r + 2] / tangent)
                    } else {
                        average[0].hypot(average[1])
                    };
                    assert!(error < 3e-5, "friction stationarity={error}");
                    assert!(multiplier * (radius - tangent).max(0.) / radius < 3e-5);
                }
            }
            1 => {
                domains.push(Domain::Bilateral);
                assert!(after[r].abs() < 3e-5);
            }
            2 => {
                domains.push(Domain::Nonnegative);
                assert!(
                    lambda[r] >= 0.
                        && after[r] >= -3e-5
                        && (lambda[r] == 0. || after[r].abs() < 3e-5)
                );
            }
            _ => unreachable!(),
        }
    }
    let gram: Vec<_> = indices
        .iter()
        .flat_map(|&i| indices.iter().map(move |&j| f.gram[i * n + j] as f64))
        .collect();
    let free: Vec<_> = indices.iter().map(|&i| f.free[i] as f64).collect();
    let cpu = solve_midpoint(&gram, &free, &domains).unwrap();
    for (i, &row) in indices.iter().enumerate() {
        let expected = free[i]
            + (0..indices.len())
                .map(|j| gram[i * indices.len() + j] * cpu.impulses[j])
                .sum::<f64>();
        assert!(
            (after[row] - expected).abs() < 2e-4,
            "row={row}, actual={}, cpu={expected}",
            after[row]
        );
    }
    if f.blocks.iter().all(|b| b.metadata[0] == 0) {
        let work: f64 = (0..n)
            .map(|i| 0.5 * lambda[i] * (f.free[i] as f64 + after[i]))
            .sum();
        let scale = lambda.iter().map(|v| v.abs()).sum::<f64>().max(1e-12);
        assert!(work <= 3e-5 * scale, "work={work}, scale={scale}");
    }
}

#[test]
#[ignore = "requires GPU"]
fn midpoint_oblique_mass_friction_restitution_sweep() {
    let mut fixtures = Vec::new();
    for mass in [1e-6, 1., 1e6] {
        for coupling in [[0., 0.], [0.8, 0.], [-0.6, 0.5], [0.95, 0.2]] {
            for mu in [0.05, 0.4, 2.] {
                for restitution in [0., 0.5, 1.] {
                    for slip in [-3., -0.8, 0., 0.8, 3.] {
                        fixtures.push(single(mass, coupling, mu, restitution, slip));
                    }
                }
            }
        }
    }
    let harness = Harness::new(&fixtures, 4096);
    let (status, result) = harness.run(1);
    assert_eq!(status, 0);
    for (i, f) in fixtures.iter().enumerate() {
        eprintln!("checking fixture {i}");
        check(f, &result[i..i + 1]);
    }
}

#[test]
#[ignore = "requires GPU"]
fn mixed_joint_rows_and_global_mode_promotion() {
    let mut fixtures = Vec::new();
    for kind in [1, 2] {
        let mut gram = vec![0.; 36];
        for i in [0, 1, 2] {
            gram[i * 6 + i] = 1.;
        }
        gram[6 + 3] = 1.;
        gram[3 * 6 + 1] = 1.;
        gram[3 * 6 + 3] = 1.;
        let f = Fixture {
            gram,
            free: vec![-1., 0.001, 0., -0.5, 0., 0.],
            blocks: vec![
                ImpactGpuBlock {
                    contact: [0.8, 0.4, 0.5, -1.],
                    metadata: [0, 0, 0, 0],
                },
                ImpactGpuBlock {
                    metadata: [kind, 0, 0, 0],
                    ..Default::default()
                },
            ],
        };
        fixtures.push(f);
    }
    let harness = Harness::new(&fixtures, 4096);
    let (status, result) = harness.run(4);
    assert_eq!(status, 0);
    for (i, f) in fixtures.iter().enumerate() {
        check(f, &result[i * 2..i * 2 + 2]);
        assert_eq!(result[i * 2].diagnostics[0], 1);
    }
}

#[test]
#[ignore = "requires GPU"]
fn coupled_manifolds_scalar_rows_and_block_permutations() {
    let mut fixtures = Vec::new();
    let indices = [0, 1, 2, 3, 4, 5, 6, 9];
    for restitution in [0., 0.5, 1.] {
        for mass in [1e-6, 1., 1e6] {
            let mut f = Fixture {
                gram: vec![0.; 144],
                free: vec![-1., 0.6, -0.3, -0.7, -0.5, 0.4, 0., 0., 0., -0.2, 0., 0.],
                blocks: vec![
                    ImpactGpuBlock {
                        contact: [0.6, 0.4, restitution, -1.],
                        metadata: [0, 1, 0, 0],
                    },
                    ImpactGpuBlock {
                        contact: [0.6, 0.4, restitution, -0.7],
                        metadata: [0, 1, 0, 0],
                    },
                    ImpactGpuBlock {
                        metadata: [1, 0, 0, 0],
                        ..Default::default()
                    },
                    ImpactGpuBlock {
                        metadata: [2, 0, 0, 0],
                        ..Default::default()
                    },
                ],
            };
            for &i in &indices {
                f.gram[i * 12 + i] = 1. / mass;
            }
            for (i, j, value) in [
                (0, 3, 0.25),
                (1, 4, -0.2),
                (0, 4, 0.1),
                (2, 5, 0.2),
                (0, 6, 0.15),
                (1, 6, 0.2),
                (3, 9, -0.2),
            ] {
                f.gram[i * 12 + j] = value / mass;
                f.gram[j * 12 + i] = value / mass;
            }
            let mut permuted = f.clone();
            let order = [9, 10, 11, 3, 4, 5, 6, 7, 8, 0, 1, 2];
            for i in 0..12 {
                permuted.free[i] = f.free[order[i]];
                for j in 0..12 {
                    permuted.gram[i * 12 + j] = f.gram[order[i] * 12 + order[j]];
                }
            }
            permuted.blocks.swap(0, 3);
            fixtures.extend([f, permuted]);
        }
    }
    let harness = Harness::new(&fixtures, 4096);
    let (status, result) = harness.run(1);
    assert_eq!(status, 0);
    for (i, f) in fixtures.iter().enumerate() {
        check(f, &result[i * 4..i * 4 + 4]);
    }
    for pair in result.chunks_exact(8) {
        for (a, b) in [(0, 7), (1, 5), (2, 6), (3, 4)] {
            for axis in 0..3 {
                assert!((pair[a].velocity[axis] - pair[b].velocity[axis]).abs() < 8e-5);
            }
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn normal_bounce_midpoint_capture_and_singular_tangents() {
    let mut fixtures = Vec::new();
    for mass in [1e-9, 1., 1e9] {
        for e in [0., 0.5, 1.] {
            fixtures.push(single(mass, [0., 0.], 0., e, 0.));
        }
    }
    let mut capture = single(1., [0., 0.], 0.4, 0., 0.);
    capture.free[1] = 0.2;
    capture.blocks[0].metadata[1] = 0;
    fixtures.push(capture);
    for mu in [0.4, 10.] {
        let mut f = single(1., [0., 0.], mu, 0., 0.);
        f.gram[5] = 1.;
        f.gram[7] = 1.;
        f.free[1] = 2.;
        f.free[2] = 2.;
        fixtures.push(f);
    }
    let harness = Harness::new(&fixtures, 4096);
    let (status, result) = harness.run(1);
    assert_eq!(status, 0);
    for (i, f) in fixtures.iter().enumerate() {
        check(f, &result[i..i + 1]);
    }
    assert_eq!(result[9].diagnostics[0], 0);
    assert_eq!(result[9].diagnostics[1], 1);
    assert!((result[9].velocity[1] + 0.2).abs() < 2e-5);
}

#[test]
#[ignore = "requires GPU"]
fn uncoupled_normal_rebound_does_not_round_through_tangent_mass_scaling() {
    let mut fixtures = Vec::new();
    for tangent in [1.7475, 2.31, 5.7, 101.3] {
        for friction in [0., 0.2] {
            let mut f = single(2., [0., 0.], friction, 1., 0.);
            f.gram[4] = tangent;
            f.gram[8] = tangent;
            fixtures.push(f);
        }
    }
    let harness = Harness::with_tolerance(&fixtures, 32, 1e-7);
    let (status, solutions) = harness.run(1);
    assert_eq!(status, 0);
    for solution in solutions {
        assert_eq!(solution.impulse[0], 4.);
        assert_eq!(solution.velocity[0], 1.);
        assert_eq!(solution.diagnostics[2], 1);
    }
    let mut fixture = single(2., [0., 0.], 0., 1., 0.);
    fixture.gram[0] = f32::from_bits(0.5f32.to_bits() - 1);
    let harness = Harness::with_tolerance(std::slice::from_ref(&fixture), 32, 1e-7);
    let (status, solutions) = harness.run(1);
    assert_eq!(status, 0);
    let impulse = solutions[0].impulse[0];
    let residual = |value: f32| (fixture.gram[0] as f64 * value as f64 - 2.).abs();
    assert!(residual(impulse) > 1e-7);
    for neighbour in [impulse.to_bits() - 1, impulse.to_bits() + 1] {
        assert!(residual(impulse) <= residual(f32::from_bits(neighbour)));
    }
    assert_eq!(solutions[0].diagnostics[2], 1);
}

#[test]
#[ignore = "requires GPU"]
fn sliding_isotropic_tangents_converge_at_runtime_tolerance() {
    let fixtures: Vec<_> = [1.75, 1.747_501_3, 1.7475]
        .into_iter()
        .map(|tangent| Fixture {
            gram: vec![0.5, 0., 0., 0., tangent, 0., 0., 0., tangent],
            free: vec![-1., 1., 0.2],
            blocks: vec![ImpactGpuBlock {
                contact: [0.4, 0.2, 0.5, -1.],
                metadata: [0, 1, 0, 0],
            }],
        })
        .collect();
    let harness = Harness::with_tolerance(&fixtures, 8192, 1e-7);
    let (status, solutions) = harness.run(1);
    assert_eq!(status, 0);
    for (fixture, solution) in fixtures.iter().zip(solutions) {
        assert_eq!(solution.diagnostics[2], 1);
        let tangent = [solution.impulse[1] as f64, solution.impulse[2] as f64];
        let gradient = [
            2.0 * fixture.free[1] as f64 + fixture.gram[4] as f64 * tangent[0],
            2.0 * fixture.free[2] as f64 + fixture.gram[8] as f64 * tangent[1],
        ];
        let length = tangent[0].hypot(tangent[1]);
        let stationary = (gradient[0] * tangent[1] - gradient[1] * tangent[0]).abs() / length;
        assert!(0.5 * stationary <= 1e-7, "stationarity={stationary:e}");
        check(fixture, &[solution]);
    }
}

#[test]
#[ignore = "requires GPU"]
fn tangent_basis_covariance_and_zero_normal_capacity() {
    let base = single(1., [0.6, 0.3], 0.7, 1., -0.8);
    let mut fixtures = vec![base.clone()];
    let mut rotations = Vec::new();
    for angle in [0.3_f64, 1.2, 2.8] {
        let (s, c) = angle.sin_cos();
        let rotation = [1., 0., 0., 0., c, -s, 0., s, c];
        let mut f = base.clone();
        for i in 0..3 {
            f.free[i] = (0..3)
                .map(|j| rotation[i * 3 + j] * base.free[j] as f64)
                .sum::<f64>() as f32;
            for j in 0..=i {
                let mut value = 0.;
                for k in 0..3 {
                    for l in 0..3 {
                        value +=
                            rotation[i * 3 + k] * base.gram[k * 3 + l] as f64 * rotation[j * 3 + l];
                    }
                }
                f.gram[i * 3 + j] = value as f32;
                f.gram[j * 3 + i] = value as f32;
            }
        }
        fixtures.push(f);
        rotations.push(rotation);
    }
    let mut stationary = single(1., [0., 0.], 0.4, 1., 2.);
    stationary.free[0] = 0.;
    stationary.blocks[0].contact[3] = 0.;
    fixtures.push(stationary);
    let harness = Harness::new(&fixtures, 4096);
    let (status, result) = harness.run(1);
    assert_eq!(status, 0);
    for (i, f) in fixtures.iter().enumerate() {
        check(f, &result[i..i + 1]);
    }
    for (i, rotation) in rotations.iter().enumerate() {
        for r in 0..3 {
            let actual: f64 = (0..3)
                .map(|j| rotation[j * 3 + r] * result[i + 1].impulse[j] as f64)
                .sum();
            assert!((actual - result[0].impulse[r] as f64).abs() < 4e-5);
        }
    }
    assert_eq!(result[4].impulse, [0.; 4]);
}

#[test]
#[ignore = "requires GPU"]
fn invalid_capacity_padding_materials_and_separating_contacts_fail_closed() {
    let fixture = single(1., [0., 0.], 0.4, 0.5, 0.);
    let harness = Harness::new(std::slice::from_ref(&fixture), 4096);
    let valid = ImpactGpuConfig {
        island_capacity: 1,
        blocks_per_island: 1,
        max_iterations: 4096,
        velocity_tolerance: 2e-5,
    };
    for config in [
        ImpactGpuConfig {
            island_capacity: 0,
            ..valid
        },
        ImpactGpuConfig {
            blocks_per_island: 0,
            ..valid
        },
        ImpactGpuConfig {
            blocks_per_island: u32::MAX,
            ..valid
        },
        ImpactGpuConfig {
            blocks_per_island: 2,
            ..valid
        },
        ImpactGpuConfig {
            max_iterations: 0,
            ..valid
        },
        ImpactGpuConfig {
            velocity_tolerance: 0.,
            ..valid
        },
        ImpactGpuConfig {
            velocity_tolerance: f32::NAN,
            ..valid
        },
    ] {
        assert!(
            ResidentImpactConstraints::new(
                harness.engine.device().unwrap(),
                &harness.inputs[0],
                &harness.inputs[1],
                &harness.inputs[2],
                &harness.inputs[3],
                &harness.inputs[4],
                &harness.inputs[5],
                config
            )
            .is_err()
        );
    }
    let mut invalids = Vec::new();
    for (field, value) in [(0, 0.1), (1, -0.1), (2, 1.1), (3, 1.)] {
        let mut block = fixture.blocks[0];
        block.contact[field] = value;
        invalids.push(block);
    }
    for (field, value) in [(0, 3), (1, 2), (2, 1), (3, 1)] {
        let mut block = fixture.blocks[0];
        block.metadata[field] = value;
        invalids.push(block);
    }
    invalids.push(ImpactGpuBlock {
        metadata: [1, 0, 0, 0],
        ..Default::default()
    });
    for block in invalids {
        harness.write(5, &[0u32]);
        harness.write(2, &[block]);
        assert_eq!(harness.run(1).0, IMPACT_FAULT_INPUT);
    }
    harness.write(5, &[0u32]);
    harness.write(2, &fixture.blocks);
    harness.write(4, &[2u32]);
    assert_eq!(harness.run(1).0, IMPACT_FAULT_INPUT);
    harness.write(5, &[0u32]);
    harness.write(4, &[0u32]);
    let (status, result) = harness.run(1);
    assert_eq!(status, 0);
    assert_eq!(result[0].impulse, [0.; 4]);
}

#[test]
#[ignore = "requires GPU"]
fn iteration_exhaustion_and_external_halt_do_not_publish_valid_solutions() {
    let one = single(1., [0., 0.], 0.4, 0.5, 0.2);
    let mut f = Fixture {
        gram: vec![0.; 36],
        free: one.free.repeat(2),
        blocks: one.blocks.repeat(2),
    };
    for i in 0..6 {
        f.gram[i * 6 + i] = 1.;
    }
    for i in 0..3 {
        f.gram[i * 6 + i + 3] = 0.5;
        f.gram[(i + 3) * 6 + i] = 0.5;
    }
    let harness = Harness::new(&[f.clone()], 1);
    let (status, result) = harness.run(1);
    assert_eq!(status, IMPACT_FAULT_CONVERGENCE);
    assert!(result.iter().all(|s| s.impulse == [0.; 4]));
    for i in 0..3 {
        f.gram[i * 6 + i + 3] = 0.;
        f.gram[(i + 3) * 6 + i] = 0.;
    }
    harness.write(0, &f.gram);
    harness.write(5, &[0u32]);
    let (status, result) = harness.run(8);
    assert_eq!(status, 0);
    check(&f, &result);
    harness.write(5, &[1u32 << 30]);
    let (status, result) = harness.run(2);
    assert_eq!(status, 1u32 << 30);
    assert!(
        result
            .iter()
            .all(|s| s.impulse == [0.; 4] && s.velocity == [0.; 4])
    );
}

#[test]
#[ignore = "requires GPU"]
fn faults_are_sticky_and_counts_clear_old_solutions() {
    let fixture = single(1., [0., 0.], 0.4, 0.5, 0.);
    let harness = Harness::new(std::slice::from_ref(&fixture), 4096);
    let (status, result) = harness.run(2);
    assert_eq!(status, 0);
    check(&fixture, &result);
    harness.write(3, &[0u32]);
    let (status, result) = harness.run(1);
    assert_eq!(status, 0);
    assert_eq!(result[0].impulse, [0.; 4]);
    harness.write(3, &[2u32]);
    assert_eq!(harness.run(1).0, IMPACT_FAULT_INPUT);
    harness.write(3, &[1u32]);
    assert_eq!(harness.run(1).0, IMPACT_FAULT_INPUT);
    harness.write(5, &[0u32]);
    let (status, result) = harness.run(1);
    assert_eq!(status, 0);
    check(&fixture, &result);
    let mut invalid = fixture.blocks.clone();
    invalid[0].contact[3] = -2.;
    harness.write(2, &invalid);
    assert_eq!(harness.run(1).0, IMPACT_FAULT_INPUT);
    harness.write(5, &[0u32]);
    harness.write(2, &fixture.blocks);
    harness.write(0, &[1f32, -0.75, -0.75, -0.75, 1., -0.75, -0.75, -0.75, 1.]);
    assert_eq!(harness.run(1).0, IMPACT_FAULT_PSD);
    harness.write(5, &[0u32]);
    harness.write(0, &fixture.gram);
    harness.write(1, &[f32::NAN, 0., 0.]);
    assert_eq!(harness.run(1).0, IMPACT_FAULT_NONFINITE);
    harness.write(5, &[0u32]);
    harness.write(1, &fixture.free);
    harness.write(0, &[0f32; 9]);
    assert_eq!(harness.run(1).0, IMPACT_FAULT_CONVERGENCE);
}
