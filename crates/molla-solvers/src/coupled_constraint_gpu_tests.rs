use super::*;
use crate::coulomb_disk_gpu::{
    CoulombDiskInput, CoulombDiskSolution, FRICTION_FAULT_CONVERGENCE, FRICTION_FAULT_INPUT,
    FRICTION_FAULT_NONFINITE, FRICTION_FAULT_PSD,
};
use molla_compute::Engine;

#[derive(Clone)]
struct Fixture {
    gram: Vec<f32>,
    free: Vec<f32>,
    data: Vec<CoulombDiskInput>,
    kinds: Vec<u32>,
}

fn mixed(scale: f32, kind: ConstraintBlockKind, sign: f32) -> Fixture {
    Fixture {
        gram: vec![
            1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
        .into_iter()
        .map(|v| v / scale)
        .collect(),
        free: vec![sign * 0.01, 0.0, sign * 0.01, 0.0],
        data: vec![
            CoulombDiskInput {
                static_cap: 0.0025 * scale,
                dynamic_cap: 0.002 * scale,
                initial_velocity: [sign * 0.01, 0.0],
            },
            CoulombDiskInput::default(),
        ],
        kinds: vec![0, kind as u32],
    }
}

fn buffer(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("coupled constraint test input"),
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
    solver: ResidentCoupledConstraints,
    inputs: Vec<wgpu::Buffer>,
    blocks: usize,
    islands: usize,
}

impl Harness {
    fn new(fixtures: &[Fixture]) -> Self {
        let engine = Engine::new_gpu().unwrap();
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let blocks = fixtures[0].kinds.len();
        let islands = fixtures.len();
        let gram: Vec<_> = fixtures
            .iter()
            .flat_map(|f| f.gram.iter().copied())
            .collect();
        let free: Vec<_> = fixtures
            .iter()
            .flat_map(|f| f.free.iter().copied())
            .collect();
        let data: Vec<_> = fixtures
            .iter()
            .flat_map(|f| f.data.iter().copied())
            .collect();
        let kinds: Vec<_> = fixtures
            .iter()
            .flat_map(|f| f.kinds.iter().copied())
            .collect();
        let counts = vec![blocks as u32; islands];
        let inputs: Vec<_> = [
            bytemuck::cast_slice(&gram),
            bytemuck::cast_slice(&free),
            bytemuck::cast_slice(&data),
            bytemuck::cast_slice(&kinds),
            bytemuck::cast_slice(&counts),
            bytemuck::bytes_of(&(islands as u32)),
            bytemuck::bytes_of(&0u32),
        ]
        .into_iter()
        .map(|bytes| buffer(device, queue, bytes))
        .collect();
        let solver = ResidentCoupledConstraints::new(
            device,
            &inputs[0],
            &inputs[1],
            &inputs[2],
            &inputs[3],
            &inputs[4],
            &inputs[5],
            &inputs[6],
            CoupledConstraintConfig {
                island_capacity: islands as u32,
                blocks_per_island: blocks as u32,
                max_iterations: 4096,
                velocity_tolerance: 1e-7,
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

    fn write<T: bytemuck::Pod>(&self, slot: usize, data: &[T]) {
        molla_core::gpu_write_buffer!(
            self.engine.queue().unwrap(),
            &self.inputs[slot],
            0,
            bytemuck::cast_slice(data),
        );
    }

    fn run(&self) -> (u32, Vec<CoulombDiskSolution>) {
        let (status, solutions, _) = self.run_with_residues();
        (status, solutions)
    }

    fn run_with_residues(&self) -> (u32, Vec<CoulombDiskSolution>, Vec<[f32; 2]>) {
        let device = self.engine.device().unwrap();
        let size = (self.blocks * self.islands * 32) as u64;
        let residue_start = size + 4 + self.islands as u64 * 16;
        let residue_size = (self.blocks * self.islands * 8) as u64;
        let snapshot = molla_core::gpu_create_buffer!(
            device,
            &wgpu::BufferDescriptor {
                label: Some("coupled test readback"),
                size: residue_start + residue_size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }
        );
        let mut encoder = device.create_command_encoder(&Default::default());
        self.solver.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(self.solver.solution_buffer(), 0, &snapshot, 0, size);
        encoder.copy_buffer_to_buffer(&self.inputs[6], 0, &snapshot, size, 4);
        encoder.copy_buffer_to_buffer(
            self.solver.diagnostic_buffer(),
            0,
            &snapshot,
            size + 4,
            self.islands as u64 * 16,
        );
        encoder.copy_buffer_to_buffer(
            self.solver.solution_buffer(),
            self.solver.impulse_residue_offset(),
            &snapshot,
            residue_start,
            residue_size,
        );
        self.engine.queue().unwrap().submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        snapshot
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let bytes = molla_core::gpu_readback!(snapshot.slice(..));
        let solutions = bytes[..size as usize]
            .chunks_exact(32)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let status = bytemuck::pod_read_unaligned(&bytes[size as usize..size as usize + 4]);
        if status != 0 {
            for (island, raw) in bytes[size as usize + 4..residue_start as usize]
                .chunks_exact(16)
                .enumerate()
            {
                let diagnostic = bytemuck::pod_read_unaligned::<[u32; 4]>(raw);
                if diagnostic[0] != 0 {
                    eprintln!(
                        "island={island} diagnostic={diagnostic:?} residual={}",
                        f32::from_bits(diagnostic[2])
                    );
                }
            }
        }
        assert_eq!(self.solver.resource_counts(), [4, 1, 1]);
        let residues = bytes[residue_start as usize..]
            .chunks_exact(8)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        (status, solutions, residues)
    }
}

fn check_kkt(f: &Fixture, solution: &[CoulombDiskSolution]) {
    check_kkt_with_residues(f, solution, &vec![[0.; 2]; solution.len()]);
}

fn check_kkt_with_residues(f: &Fixture, solution: &[CoulombDiskSolution], residues: &[[f32; 2]]) {
    let lambda: Vec<f64> = solution
        .iter()
        .zip(&f.kinds)
        .zip(residues)
        .flat_map(|((s, kind), low)| {
            let impulse = s.impulse.map(f64::from);
            if *kind == 0 {
                [
                    impulse[0] + f64::from(low[0]),
                    impulse[1] + f64::from(low[1]),
                ]
            } else {
                [impulse[0] + impulse[1], 0.]
            }
        })
        .collect();
    let n = lambda.len();
    let gradient: Vec<f64> = (0..n)
        .map(|i| {
            f.free[i] as f64
                + (0..n)
                    .map(|j| f.gram[i * n + j] as f64 * lambda[j])
                    .sum::<f64>()
        })
        .collect();
    for (i, s) in solution.iter().enumerate() {
        let v = [gradient[2 * i], gradient[2 * i + 1]];
        let l = [lambda[2 * i], lambda[2 * i + 1]];
        let error = match f.kinds[i] {
            0 => {
                let cap = s.applied_cap as f64;
                let length = l[0].hypot(l[1]);
                assert!(length <= cap * (1.0 + 2e-6));
                if cap == 0.0 {
                    0.0
                } else if length == 0.0 {
                    v[0].hypot(v[1])
                } else {
                    let angular = if v[0] * l[0] + v[1] * l[1] > 0.0 {
                        v[0].hypot(v[1])
                    } else {
                        (v[0] * l[1] - v[1] * l[0]).abs() / length
                    };
                    let a = f.gram[(2 * i) * n + 2 * i].abs() as f64;
                    let b = f.gram[(2 * i) * n + 2 * i + 1].abs() as f64;
                    let d = f.gram[(2 * i + 1) * n + 2 * i + 1].abs() as f64;
                    let radial = v[0]
                        .hypot(v[1])
                        .min((a + b).max(d + b) * (cap - length).abs());
                    angular
                        .max(v[0].hypot(v[1]) * (1.0 - length / cap).abs())
                        .max(radial)
                }
            }
            1 => {
                assert_eq!(s.regime, 4);
                assert_eq!(l[1], 0.0);
                v[0].abs()
            }
            2 => {
                assert_eq!(s.regime, 5);
                assert_eq!(l[1], 0.0);
                assert!(l[0] >= 0.0);
                if l[0] == 0.0 {
                    (-v[0]).max(0.0)
                } else {
                    v[0].abs()
                }
            }
            _ => panic!("invalid fixture"),
        };
        assert!(error <= 1e-7, "block={i} f64 KKT={error:e} solution={s:?}");
        assert!((s.slip[0] - v[0] as f32).abs() < 1e-7);
        assert!((s.slip[1] - v[1] as f32).abs() < 1e-7);
    }
    let energy: f64 = (0..n)
        .map(|i| 0.5 * lambda[i] * (f.free[i] as f64 + gradient[i]))
        .sum();
    assert!(energy <= 1e-7 * lambda.iter().map(|v| v.abs()).sum::<f64>());
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn large_scalar_recovery_preserves_absolute_kkt_precision() {
    let mut fixtures = Vec::new();
    for kind in [1, 2, 3] {
        for gram in [0.37_f32, 1.17, 7.31] {
            fixtures.push(Fixture {
                gram: vec![gram, 0., 0., 0.],
                free: vec![-1537.173_f32, 0.],
                data: vec![if kind == 3 {
                    CoulombDiskInput {
                        static_cap: 0.,
                        dynamic_cap: 1e5,
                        initial_velocity: [0.13, 0.],
                    }
                } else {
                    CoulombDiskInput::default()
                }],
                kinds: vec![kind],
            });
        }
    }
    let harness = Harness::new(&fixtures);
    let adapter = harness.engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires hardware GPU");
    println!("adapter={adapter:?} scalar_absolute_kkt=1e-7");
    let (status, solutions) = harness.run();
    assert_eq!(status, 0);
    for (fixture, solution) in fixtures.iter().zip(&solutions) {
        let impulse = f64::from(solution.impulse[0]) + f64::from(solution.impulse[1]);
        let diagonal = f64::from(fixture.gram[0])
            + if fixture.kinds[0] == 3 {
                f64::from(fixture.data[0].initial_velocity[0])
            } else {
                0.
            };
        let residual = (f64::from(fixture.free[0]) + diagonal * impulse).abs();
        assert!(
            residual <= 1e-7,
            "kind={} impulse={:?} residual={residual:e}",
            fixture.kinds[0],
            solution.impulse
        );
        assert!(solution.residual <= 1e-7);
        assert!(impulse > 0.);
        println!(
            "kind={} gram={} impulse={:?} host_kkt={residual:e}",
            fixture.kinds[0], fixture.gram[0], solution.impulse
        );
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn large_scalar_recovery_coupled_to_sliding_disk_preserves_absolute_kkt_precision() {
    let cap = 0.0006_f32;
    let cross = 0.37_f32;
    let target = 1537.173_f32;
    let fixture = Fixture {
        gram: vec![
            1., 0., 1., 0., 0., 1., cross, 0., 1., cross, 2., 0., 0., 0., 0., 0.,
        ],
        free: vec![0., 0., -target, 0.],
        data: vec![
            CoulombDiskInput {
                static_cap: cap,
                dynamic_cap: cap,
                initial_velocity: [1., 0.],
            },
            CoulombDiskInput::default(),
        ],
        kinds: vec![0, 2],
    };
    let magnitude = 1_f64.hypot(f64::from(cross));
    let expected_disk = [
        -f64::from(cap) / magnitude,
        -f64::from(cap) * f64::from(cross) / magnitude,
    ];
    let expected_scalar = (f64::from(target) + f64::from(cap) * magnitude) / 2.;
    assert!(
        (2. * expected_scalar + expected_disk[0] + f64::from(cross) * expected_disk[1]
            - f64::from(target))
        .abs()
            < 1e-10
    );
    println!(
        "large_mixed_reference disk={expected_disk:?} scalar={expected_scalar} gram_min_schur={}",
        1. - f64::from(cross).powi(2)
    );
    let harness = Harness::new(std::slice::from_ref(&fixture));
    assert!(
        !harness.engine.adapter_info().unwrap().is_software,
        "requires hardware GPU"
    );
    let (status, solutions, residues) = harness.run_with_residues();
    assert_eq!(
        status, 0,
        "large scalar recovery and sliding disk must satisfy the unchanged1e-7 KKT gate"
    );
    println!("large_mixed_before_oracle solutions={solutions:?} residues={residues:?}");
    check_kkt_with_residues(&fixture, &solutions, &residues);
    for ((actual, low), expected) in solutions[0]
        .impulse
        .iter()
        .zip(residues[0])
        .zip(expected_disk)
    {
        assert!((f64::from(*actual) + f64::from(low) - expected).abs() < 1e-7 * f64::from(cap));
    }
    let scalar = f64::from(solutions[1].impulse[0]) + f64::from(solutions[1].impulse[1]);
    assert!((scalar - expected_scalar).abs() < 1e-7);
    println!(
        "large_mixed_solution primary={:?} residues={residues:?} scalar={scalar} residual={}",
        solutions[0].impulse, solutions[0].residual
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn bounded_feedback_couples_to_friction_and_publishes_physical_slip() {
    let mut fixtures = Vec::new();
    let mut expected = Vec::new();
    for scale in [1e-12_f32, 0.01, 1., 100., 1e12] {
        for feedback in [0., 0.5, 2.] {
            for (u, v) in [
                (-0.03, -0.04),
                (-0.03, 0.04),
                (0., 0.),
                (0.03, -0.04),
                (0.03, 0.04),
            ] {
                for (lower, upper) in [
                    (-0.003, 0.007),
                    (0.002, 0.002),
                    (-0.005, -0.002),
                    (0.002, 0.006),
                    (f32::NEG_INFINITY, 0.001),
                    (-0.001, f32::INFINITY),
                    (f32::NEG_INFINITY, f32::INFINITY),
                ] {
                    let f = Fixture {
                        gram: [
                            2., 0., 0.5, 0., 0., 1., 0., 0., 0.5, 0., 1.5, 0., 0., 0., 0., 0.,
                        ]
                        .map(|g| g / scale)
                        .to_vec(),
                        free: vec![u, 0., v, 0.],
                        data: vec![
                            CoulombDiskInput {
                                static_cap: 0.006 * scale,
                                dynamic_cap: 0.004 * scale,
                                initial_velocity: [0.01, 0.],
                            },
                            CoulombDiskInput {
                                static_cap: lower * scale,
                                dynamic_cap: upper * scale,
                                initial_velocity: [feedback / scale, 0.],
                            },
                        ],
                        kinds: vec![0, 3],
                    };
                    let a = f.gram[0] as f64;
                    let b = f.gram[2] as f64;
                    let d = f.gram[10] as f64 + f.data[1].initial_velocity[0] as f64;
                    let radius = f.data[0].dynamic_cap as f64;
                    let lower = f.data[1].static_cap as f64;
                    let upper = f.data[1].dynamic_cap as f64;
                    let u = u as f64;
                    let v = v as f64;
                    let objective = |x: f64, z: f64| {
                        0.5 * (a * x * x + 2. * b * x * z + d * z * z) + u * x + v * z
                    };
                    let determinant = a * d - b * b;
                    let (x, z) = ((b * v - d * u) / determinant, (b * u - a * v) / determinant);
                    let mut candidates = Vec::new();
                    if x.abs() <= radius && z >= lower && z <= upper {
                        candidates.push((x, z));
                    }
                    for x in [-radius, radius] {
                        candidates.push((x, (-(v + b * x) / d).clamp(lower, upper)));
                    }
                    for z in [lower, upper].into_iter().filter(|z| z.is_finite()) {
                        candidates.push(((-(u + b * z) / a).clamp(-radius, radius), z));
                    }
                    expected.push(
                        candidates
                            .into_iter()
                            .min_by(|&(x, z), &(y, w)| objective(x, z).total_cmp(&objective(y, w)))
                            .unwrap(),
                    );
                    fixtures.push(f);
                }
            }
        }
    }
    let harness = Harness::new(&fixtures);
    for repeat in 0..2 {
        let (status, solutions) = harness.run();
        assert_eq!(status, 0, "repeat={repeat}");
        for ((f, solution), &(x, z)) in fixtures
            .iter()
            .zip(solutions.chunks_exact(2))
            .zip(&expected)
        {
            let scale = 2. / f.gram[0] as f64;
            for (actual, expected) in solution.iter().flat_map(|s| s.impulse).zip([x, 0., z, 0.]) {
                assert!(
                    (actual as f64 - expected).abs() / scale < 2e-7,
                    "actual={actual} expected={expected} scale={scale} fixture={:?}",
                    f.free
                );
            }
            assert_eq!(solution[1].regime, 6);
            assert_eq!(solution[1].applied_cap, 0.);
            let lambda: Vec<f64> = solution
                .iter()
                .flat_map(|s| s.impulse.map(f64::from))
                .collect();
            for (i, actual) in solution.iter().flat_map(|s| s.slip).enumerate() {
                let physical = f.free[i] as f64
                    + (0..4)
                        .map(|j| f.gram[4 * i + j] as f64 * lambda[j])
                        .sum::<f64>();
                assert!((actual as f64 - physical).abs() < 1e-7);
            }
            let impulse = lambda[2];
            let lower = f.data[1].static_cap as f64;
            let upper = f.data[1].dynamic_cap as f64;
            assert!(impulse >= lower && impulse <= upper);
            let gradient = f.free[2] as f64
                + f.gram[8] as f64 * lambda[0]
                + (f.gram[10] as f64 + f.data[1].initial_velocity[0] as f64) * impulse;
            let error = if lower == upper {
                0.
            } else if impulse == lower {
                (-gradient).max(0.)
            } else if impulse == upper {
                gradient.max(0.)
            } else {
                gradient.abs()
            };
            assert!(
                error <= 1e-7,
                "interval KKT={error} impulse={impulse} bounds=[{lower},{upper}]"
            );
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn redundant_bounded_allocation_matches_independent_capped_minimum_norm() {
    let mut fixtures = Vec::new();
    let mut expected = Vec::new();
    for scale in [1e-12_f32, 0.01, 1., 100., 1e12] {
        for cap in [0.002, 0.02] {
            for (lower, upper) in [
                (-0.02, 0.02),
                (-0.003, -0.001),
                (0.002, 0.002),
                (0.001, 0.003),
                (f32::NEG_INFINITY, f32::INFINITY),
            ] {
                for reverse in [false, true] {
                    let disk = CoulombDiskInput {
                        static_cap: cap * scale,
                        dynamic_cap: cap * scale,
                        initial_velocity: [0.01, 0.],
                    };
                    let interval = CoulombDiskInput {
                        static_cap: lower * scale,
                        dynamic_cap: upper * scale,
                        initial_velocity: [0.; 2],
                    };
                    let order = if reverse { [2, 3, 0, 1] } else { [0, 1, 2, 3] };
                    let base = [
                        1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
                    ];
                    let mut f = Fixture {
                        gram: vec![0.; 16],
                        free: order.map(|i| [0.01, 0., 0.01, 0.][i]).to_vec(),
                        data: if reverse {
                            vec![interval, disk]
                        } else {
                            vec![disk, interval]
                        },
                        kinds: if reverse { vec![3, 0] } else { vec![0, 3] },
                    };
                    for i in 0..4 {
                        for j in 0..4 {
                            f.gram[4 * i + j] = base[4 * order[i] + order[j]] / scale;
                        }
                    }
                    let radius = disk.dynamic_cap as f64;
                    let lower = interval.static_cap as f64;
                    let upper = interval.dynamic_cap as f64;
                    let sum = (-(0.01_f32 as f64) / (1. / scale) as f64)
                        .clamp(lower - radius, upper + radius);
                    let z = (sum / 2.).clamp(lower.max(sum - radius), upper.min(sum + radius));
                    let lambda = [sum - z, 0., z, 0.];
                    expected.push((scale, order.map(|i| lambda[i])));
                    fixtures.push(f);
                }
            }
        }
    }
    let harness = Harness::new(&fixtures);
    for repeat in 0..2 {
        let (status, solutions) = harness.run();
        assert_eq!(status, 0, "repeat={repeat}");
        for ((f, s), (scale, expected)) in fixtures
            .iter()
            .zip(solutions.chunks_exact(2))
            .zip(&expected)
        {
            for (actual, expected) in s.iter().flat_map(|s| s.impulse).zip(expected) {
                assert!(
                    (actual as f64 - expected).abs() / (*scale as f64) < 2e-7,
                    "scale={scale} actual={actual} expected={expected} kinds={:?}",
                    f.kinds
                );
            }
            let interval = if f.kinds[0] == 3 { 0 } else { 1 };
            assert_eq!(s[interval].regime, 6);
            assert!(s[interval].impulse[0] >= f.data[interval].static_cap);
            assert!(s[interval].impulse[0] <= f.data[interval].dynamic_cap);
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn bounded_effort_and_invalid_domains_are_explicit() {
    let fixture = Fixture {
        gram: vec![0.; 4],
        free: vec![7., 0.],
        data: vec![CoulombDiskInput {
            static_cap: 0.3,
            dynamic_cap: 0.3,
            initial_velocity: [0.; 2],
        }],
        kinds: vec![3],
    };
    let harness = Harness::new(&[fixture]);
    let (status, solution) = harness.run();
    assert_eq!(status, 0);
    assert_eq!(solution[0].impulse, [0.3, 0.]);
    assert_eq!(solution[0].slip, [7., 0.]);
    assert_eq!(solution[0].regime, 6);
    for (free, lower, upper, expected) in [
        (7., -0.3, 0.3, -0.3),
        (-7., -0.3, 0.3, 0.3),
        (0., 0.2, 0.4, 0.2),
        (0., f32::NEG_INFINITY, f32::INFINITY, 0.),
        (-7., f32::NEG_INFINITY, 0.3, 0.3),
        (7., -0.3, f32::INFINITY, -0.3),
    ] {
        harness.write(6, &[0u32]);
        harness.write(1, &[free, 0.]);
        harness.write(
            2,
            &[CoulombDiskInput {
                static_cap: lower,
                dynamic_cap: upper,
                initial_velocity: [0.; 2],
            }],
        );
        let (status, solution) = harness.run();
        assert_eq!(status, 0);
        assert_eq!(solution[0].impulse, [expected, 0.]);
        assert_eq!(solution[0].slip, [free, 0.]);
    }
    harness.write(6, &[0u32]);
    harness.write(1, &[7.0_f32, 0.]);
    harness.write(
        2,
        &[CoulombDiskInput {
            static_cap: f32::NEG_INFINITY,
            dynamic_cap: f32::INFINITY,
            initial_velocity: [0.; 2],
        }],
    );
    assert_eq!(harness.run().0, FRICTION_FAULT_CONVERGENCE);
    for (lower, upper, diagonal) in [
        (0.3, -0.3, 0.),
        (f32::NAN, 0.3, 0.),
        (-0.3, f32::NAN, 0.),
        (f32::INFINITY, f32::INFINITY, 0.),
        (f32::NEG_INFINITY, f32::NEG_INFINITY, 0.),
        (-0.3, 0.3, -1.),
        (-0.3, 0.3, f32::NAN),
        (-0.3, 0.3, f32::INFINITY),
    ] {
        harness.write(6, &[0u32]);
        harness.write(
            2,
            &[CoulombDiskInput {
                static_cap: lower,
                dynamic_cap: upper,
                initial_velocity: [diagonal, 0.],
            }],
        );
        for _ in 0..2 {
            let (status, solution) = harness.run();
            assert_eq!(status, FRICTION_FAULT_INPUT);
            assert!(
                bytemuck::cast_slice::<_, u8>(&solution)
                    .iter()
                    .all(|&v| v == 0)
            );
        }
    }
    harness.write(6, &[0u32]);
    harness.write(
        2,
        &[CoulombDiskInput {
            static_cap: 0.3,
            dynamic_cap: 0.3,
            initial_velocity: [0.; 2],
        }],
    );
    let (status, solution) = harness.run();
    assert_eq!(status, 0);
    assert_eq!(solution[0].impulse, [0.3, 0.]);
    assert_eq!(solution[0].slip, [7., 0.]);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn redundant_allocation_matches_minimum_norm_across_scales_rotations_and_order() {
    let mut fixtures = Vec::new();
    let mut expected = Vec::new();
    for scale in [1e-12_f32, 1e-6, 1., 1e6, 1e12] {
        for angle in [0.0_f64, 0.4, 1.2] {
            for cap in [0.002, 0.02] {
                for reverse in [false, true] {
                    let (s, c) = angle.sin_cos();
                    let base = [1., 0., c, 0., 0., 1., s, 0., c, s, 1., 0., 0., 0., 0., 0.];
                    let free = [0.01 * c, 0.01 * s, 0.01, 0.];
                    let order = if reverse { [2, 3, 0, 1] } else { [0, 1, 2, 3] };
                    let disk = CoulombDiskInput {
                        static_cap: cap * scale,
                        dynamic_cap: cap * scale,
                        initial_velocity: [0.01, 0.],
                    };
                    let mut f = Fixture {
                        gram: vec![0.; 16],
                        free: order.map(|i| free[i] as f32).to_vec(),
                        data: if reverse {
                            vec![CoulombDiskInput::default(), disk]
                        } else {
                            vec![disk, CoulombDiskInput::default()]
                        },
                        kinds: if reverse { vec![1, 0] } else { vec![0, 1] },
                    };
                    for i in 0..4 {
                        for j in 0..4 {
                            f.gram[4 * i + j] = base[4 * order[i] + order[j]] as f32 / scale;
                        }
                    }
                    let amplitude = (cap as f64).min(0.005);
                    let solution = [-amplitude * c, -amplitude * s, -0.01 + amplitude, 0.];
                    expected.push((scale, order.map(|i| solution[i])));
                    fixtures.push(f);
                }
            }
        }
    }
    let harness = Harness::new(&fixtures);
    for repeat in 0..2 {
        let (status, solutions) = harness.run();
        assert_eq!(status, 0, "repeat={repeat}");
        for ((f, s), (scale, expected)) in fixtures
            .iter()
            .zip(solutions.chunks_exact(2))
            .zip(&expected)
        {
            check_kkt(f, s);
            for (actual, expected) in s.iter().flat_map(|s| s.impulse).zip(expected) {
                assert!(
                    (actual as f64 / *scale as f64 - expected).abs() < 2e-7,
                    "scale={scale} actual={actual} expected={expected}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn redundant_bilateral_rows_keep_inactive_stop_exactly_zero() {
    let mut gram = vec![0.; 36];
    for i in [0, 2, 4] {
        for j in [0, 2, 4] {
            gram[6 * i + j] = 1.;
        }
    }
    gram[7] = 1.;
    let fixture = Fixture {
        gram,
        free: vec![0.01, 0., 0.02, 0., 0.01, 0.],
        data: vec![
            CoulombDiskInput {
                static_cap: 0.02,
                dynamic_cap: 0.02,
                initial_velocity: [0.01, 0.],
            },
            CoulombDiskInput::default(),
            CoulombDiskInput::default(),
        ],
        kinds: vec![0, 2, 1],
    };
    let harness = Harness::new(std::slice::from_ref(&fixture));
    let (status, solution) = harness.run();
    assert_eq!(status, 0);
    check_kkt(&fixture, &solution);
    assert!((solution[0].impulse[0] + 0.005).abs() < 2e-7);
    assert!((solution[2].impulse[0] + 0.005).abs() < 2e-7);
    assert_eq!(solution[1].impulse[0], 0.);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn mixed_constraints_match_analytic_impulses_across_scales() {
    let mut fixtures = Vec::new();
    for scale in [1e-6, 1.0, 1e6] {
        for (kind, sign) in [
            (ConstraintBlockKind::Bilateral, 1.0),
            (ConstraintBlockKind::Nonnegative, -1.0),
            (ConstraintBlockKind::Nonnegative, 1.0),
        ] {
            fixtures.push(mixed(scale, kind, sign));
        }
    }
    let harness = Harness::new(&fixtures);
    let (status, solutions) = harness.run();
    assert_eq!(status, 0);
    for (i, (f, s)) in fixtures.iter().zip(solutions.chunks_exact(2)).enumerate() {
        check_kkt(f, s);
        let scale = [1e-6, 1.0, 1e6][i / 3];
        let d = 0.002 / 2.0_f64.sqrt();
        let expected = match i % 3 {
            0 => [-d, d, -0.005],
            1 => [d, -d, 0.005],
            _ => [-0.002, 0.0, 0.0],
        };
        for (actual, expected) in [s[0].impulse[0], s[0].impulse[1], s[1].impulse[0]]
            .into_iter()
            .zip(expected)
        {
            assert!(
                (actual as f64 / scale - expected).abs() < 2e-7,
                "case={i} actual={actual} expected={expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn bilateral_coupling_prevents_premature_sliding() {
    let f = Fixture {
        gram: vec![
            1., 0., 1., 0., 0., 1., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
        ],
        free: vec![0.02, 0., 0.02, 0.],
        data: vec![
            CoulombDiskInput {
                static_cap: 0.001,
                dynamic_cap: 0.0005,
                initial_velocity: [0.; 2],
            },
            CoulombDiskInput::default(),
        ],
        kinds: vec![0, 1],
    };
    let harness = Harness::new(std::slice::from_ref(&f));
    let (status, solution) = harness.run();
    assert_eq!(status, 0);
    check_kkt(&f, &solution);
    assert_eq!(solution[0].regime, 2);
    assert_eq!(solution[0].applied_cap, 0.001);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn sticky_faults_retire_and_recover_mixed_outputs() {
    let f = mixed(1.0, ConstraintBlockKind::Bilateral, 1.0);
    let harness = Harness::new(std::slice::from_ref(&f));
    for fault in 0..7 {
        harness.write(0, &f.gram);
        harness.write(1, &f.free);
        harness.write(2, &f.data);
        harness.write(3, &f.kinds);
        harness.write(4, &[2u32]);
        harness.write(6, &[0u32]);
        assert_eq!(harness.run().0, 0);
        let expected = match fault {
            0 => {
                harness.write(3, &[0u32, 4]);
                FRICTION_FAULT_INPUT
            }
            1 => {
                harness.write(1, &[f32::NAN, 0., 0., 0.]);
                FRICTION_FAULT_NONFINITE
            }
            2 => {
                let mut g = f.gram.clone();
                g[15] = 1.;
                harness.write(0, &g);
                FRICTION_FAULT_INPUT
            }
            3 => {
                let mut g = f.gram.clone();
                g[0] = -1.;
                harness.write(0, &g);
                FRICTION_FAULT_PSD
            }
            4 => {
                harness.write(1, &[0.01_f32, 0., 0.01, 1.]);
                FRICTION_FAULT_INPUT
            }
            5 => {
                let mut d = f.data.clone();
                d[1].static_cap = 1.;
                harness.write(2, &d);
                FRICTION_FAULT_INPUT
            }
            _ => {
                harness.write(4, &[3u32]);
                FRICTION_FAULT_INPUT
            }
        };
        for _ in 0..2 {
            let (status, solution) = harness.run();
            assert_eq!(status, expected, "fault={fault}");
            assert!(
                bytemuck::cast_slice::<_, u8>(&solution)
                    .iter()
                    .all(|&v| v == 0)
            );
        }
    }
    harness.write(4, &[2u32]);
    harness.write(6, &[0u32]);
    let (status, solution) = harness.run();
    assert_eq!(status, 0);
    check_kkt(&f, &solution);
    harness.write(5, &[0u32]);
    let (status, solution) = harness.run();
    assert_eq!(status, 0);
    assert!(
        bytemuck::cast_slice::<_, u8>(&solution)
            .iter()
            .all(|&v| v == 0)
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn zero_response_scalar_domains_fail_only_when_unbounded() {
    for (kind, free, expected) in [
        (1, 0., 0),
        (1, 1., FRICTION_FAULT_CONVERGENCE),
        (2, 1., 0),
        (2, -1., FRICTION_FAULT_CONVERGENCE),
    ] {
        let f = Fixture {
            gram: vec![0.; 16],
            free: vec![1., 0., free, 0.],
            data: vec![
                CoulombDiskInput {
                    static_cap: 0.1,
                    dynamic_cap: 0.1,
                    initial_velocity: [1., 0.],
                },
                CoulombDiskInput::default(),
            ],
            kinds: vec![0, kind],
        };
        let harness = Harness::new(std::slice::from_ref(&f));
        let (status, solution) = harness.run();
        assert_eq!(status, expected, "kind={kind} free={free}");
        if status == 0 {
            check_kkt(&f, &solution);
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn manufactured_mixed_kkt_permutations_cross_workgroups() {
    let mut fixtures = Vec::new();
    for island in 0..70 {
        let factor = [
            [1.0_f64, 0., 0., 0.],
            [0.3, 1.2, 0., 0.],
            [-0.2, 0.5, 0.9, 0.],
            [0.4, -0.1, 0.6, 1.1],
        ];
        let mut gram = [0.0_f32; 36];
        let rows = [0, 1, 2, 4];
        for i in 0..4 {
            for j in 0..4 {
                gram[rows[i] * 6 + rows[j]] =
                    (0..4).map(|k| factor[i][k] * factor[j][k]).sum::<f64>() as f32;
            }
        }
        let active = island % 2 == 0;
        let lambda = [
            0.075_f64,
            0.1,
            -0.2,
            0.,
            if active { 0.15 } else { 0.0 },
            0.,
        ];
        let gradient = [-0.03, -0.04, 0., 0., if active { 0. } else { 0.07 }, 0.];
        let free: Vec<f32> = (0..6)
            .map(|i| {
                (gradient[i]
                    - (0..6)
                        .map(|j| gram[i * 6 + j] as f64 * lambda[j])
                        .sum::<f64>()) as f32
            })
            .collect();
        let data = [
            CoulombDiskInput {
                static_cap: 0.15,
                dynamic_cap: 0.125,
                initial_velocity: [1., 0.],
            },
            CoulombDiskInput::default(),
            CoulombDiskInput::default(),
        ];
        let order = [
            [0, 1, 2],
            [1, 2, 0],
            [2, 0, 1],
            [0, 2, 1],
            [1, 0, 2],
            [2, 1, 0],
        ][island % 6];
        let scale = [1e-12_f32, 1e-6, 1., 1e6, 1e12][(island / 6) % 5];
        let mut f = Fixture {
            gram: vec![0.; 36],
            free: vec![0.; 6],
            data: Vec::new(),
            kinds: Vec::new(),
        };
        for i in 0..3 {
            let mut d = data[order[i]];
            d.static_cap *= scale;
            d.dynamic_cap *= scale;
            f.data.push(d);
            f.kinds.push(order[i] as u32);
            for axis in 0..2 {
                f.free[2 * i + axis] = free[2 * order[i] + axis];
                for j in 0..3 {
                    for other in 0..2 {
                        f.gram[(2 * i + axis) * 6 + 2 * j + other] =
                            gram[(2 * order[i] + axis) * 6 + 2 * order[j] + other] / scale;
                    }
                }
            }
        }
        fixtures.push(f);
    }
    let harness = Harness::new(&fixtures);
    let (status, solutions) = harness.run();
    assert_eq!(status, 0);
    for (f, s) in fixtures.iter().zip(solutions.chunks_exact(3)) {
        check_kkt(f, s);
    }
    harness.write(4, &vec![0u32; 70]);
    let (status, solutions) = harness.run();
    assert_eq!(status, 0);
    assert!(
        bytemuck::cast_slice::<_, u8>(&solutions)
            .iter()
            .all(|&v| v == 0)
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn contradictory_bilateral_rows_fail_closed() {
    let f = Fixture {
        gram: vec![
            1., 0., 1., 0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0., 0.,
        ],
        free: vec![1., 0., -1., 0.],
        data: vec![CoulombDiskInput::default(); 2],
        kinds: vec![1, 1],
    };
    let harness = Harness::new(&[f]);
    assert_eq!(harness.run().0, FRICTION_FAULT_CONVERGENCE);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn retained_device_sequence_changes_domains_counts_and_status() {
    let fixtures: Vec<_> = (0..64)
        .map(|step| {
            mixed(
                1.,
                if step % 2 == 0 {
                    ConstraintBlockKind::Bilateral
                } else {
                    ConstraintBlockKind::Nonnegative
                },
                if step % 3 == 0 { -1. } else { 1. },
            )
        })
        .collect();
    let harness = Harness::new(&fixtures[..1]);
    let device = harness.engine.device().unwrap();
    let queue = harness.engine.queue().unwrap();
    let snapshot = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("coupled device sequence"),
            size: 64 * 68,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    for (step, f) in fixtures.iter().enumerate() {
        let count = match step % 8 {
            3 => 1u32,
            4 => 0u32,
            _ => 2u32,
        };
        let status = if step % 8 == 5 {
            FRICTION_FAULT_INPUT
        } else {
            0u32
        };
        for (slot, bytes) in [
            bytemuck::cast_slice(&f.gram),
            bytemuck::cast_slice(&f.free),
            bytemuck::cast_slice(&f.data),
            bytemuck::cast_slice(&f.kinds),
            bytemuck::bytes_of(&count),
            bytemuck::bytes_of(&1u32),
            bytemuck::bytes_of(&status),
        ]
        .into_iter()
        .enumerate()
        {
            if slot == 6 && step % 8 == 6 {
                continue;
            }
            let source = buffer(device, queue, bytes);
            encoder.copy_buffer_to_buffer(&source, 0, &harness.inputs[slot], 0, bytes.len() as u64);
        }
        harness.solver.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(
            harness.solver.solution_buffer(),
            0,
            &snapshot,
            step as u64 * 68,
            64,
        );
        encoder.copy_buffer_to_buffer(&harness.inputs[6], 0, &snapshot, step as u64 * 68 + 64, 4);
    }
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    snapshot
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = molla_core::gpu_readback!(snapshot.slice(..));
    for (step, (f, frame)) in fixtures.iter().zip(bytes.chunks_exact(68)).enumerate() {
        let status = bytemuck::pod_read_unaligned::<u32>(&frame[64..]);
        let halted = matches!(step % 8, 5 | 6);
        assert_eq!(
            status,
            if halted { FRICTION_FAULT_INPUT } else { 0 },
            "step={step}"
        );
        let solutions: Vec<CoulombDiskSolution> = frame[..64]
            .chunks_exact(32)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        if halted || step % 8 == 4 {
            assert!(frame[..64].iter().all(|&v| v == 0));
        } else if step % 8 == 3 {
            assert!(frame[32..64].iter().all(|&v| v == 0));
            let reduced = Fixture {
                gram: vec![1., 0., 0., 1.],
                free: f.free[..2].to_vec(),
                data: f.data[..1].to_vec(),
                kinds: vec![0],
            };
            check_kkt(&reduced, &solutions[..1]);
        } else {
            check_kkt(f, &solutions);
        }
    }
    assert_eq!(harness.solver.resource_counts(), [4, 1, 1]);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-constraints-gpu"]
fn mixed_constructor_rejects_invalid_domain_buffers_and_capacity() {
    let harness = Harness::new(&[mixed(1., ConstraintBlockKind::Bilateral, 1.)]);
    let device = harness.engine.device().unwrap();
    let small = buffer(
        device,
        harness.engine.queue().unwrap(),
        bytemuck::bytes_of(&0u32),
    );
    let wrong_usage = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }
    );
    for case in 0..7 {
        let mut config = CoupledConstraintConfig {
            island_capacity: 1,
            blocks_per_island: 2,
            max_iterations: 4096,
            velocity_tolerance: 1e-7,
        };
        let kinds = match case {
            0 => &small,
            1 => &wrong_usage,
            _ => &harness.inputs[3],
        };
        match case {
            2 => config.blocks_per_island = 0,
            3 => config.island_capacity = u32::MAX,
            4 => config.max_iterations = 0,
            5 => config.velocity_tolerance = f32::NAN,
            6 => config.velocity_tolerance = 0.,
            _ => {}
        }
        assert!(
            ResidentCoupledConstraints::new(
                device,
                &harness.inputs[0],
                &harness.inputs[1],
                &harness.inputs[2],
                kinds,
                &harness.inputs[4],
                &harness.inputs[5],
                &harness.inputs[6],
                config
            )
            .is_err(),
            "case={case}"
        );
    }
}
