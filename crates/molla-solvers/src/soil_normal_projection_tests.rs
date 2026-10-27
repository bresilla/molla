use super::*;

pub(super) fn buffer(engine: &Engine, words: &[f32]) -> wgpu::Buffer {
    let result = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("normal contact fixture"),
            size: (words.len().max(1) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    if !words.is_empty() {
        engine
            .queue()
            .unwrap()
            .write_buffer(&result, 0, bytemuck::cast_slice(words));
    }
    result
}

pub(super) fn read(engine: &Engine, source: &wgpu::Buffer) -> Vec<f32> {
    let device = engine.device().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("normal contact observation"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let result = bytemuck::cast_slice::<u8, f32>(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    result
}

struct Fixture {
    buffers: [wgpu::Buffer; 4],
    projection: ExperimentalSoilNormalProjectionGpu,
}

impl Fixture {
    fn new(
        engine: &Engine,
        mass: &[f32],
        free: &[f32],
        rows: &[f32],
        contacts: &[[f32; 4]],
        iterations: u32,
    ) -> Self {
        let row_words = if contacts.is_empty() {
            vec![0.0; free.len()]
        } else {
            rows.to_vec()
        };
        let contact_words = if contacts.is_empty() {
            vec![0.0; 4]
        } else {
            contacts.iter().flatten().copied().collect()
        };
        let buffers = [
            buffer(engine, mass),
            buffer(engine, free),
            buffer(engine, &row_words),
            buffer(engine, &contact_words),
        ];
        let projection = ExperimentalSoilNormalProjectionGpu::new(
            engine,
            SoilNormalInputs {
                mass: &buffers[0],
                free_rates: &buffers[1],
                rows: &buffers[2],
                contacts: &buffers[3],
            },
            SoilNormalCapacity {
                dofs: free.len() as u32,
                contacts: contacts.len() as u32,
                max_iterations: iterations,
                max_owned_bytes: u64::MAX,
            },
        )
        .unwrap();
        Self {
            buffers,
            projection,
        }
    }

    fn run(&self, engine: &Engine, reset: bool) -> (Vec<f32>, Vec<f32>, Vec<f32>, [u32; 4]) {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        if reset {
            encoder.clear_buffer(self.projection.status_buffer(), 0, None);
        }
        self.projection.encode(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let traffic = transfers.finish();
        assert_eq!(traffic.upload_bytes, 0);
        assert_eq!(traffic.readback_bytes, 0);
        let status: Vec<u32> = read(engine, self.projection.status_buffer())
            .iter()
            .map(|v| v.to_bits())
            .collect();
        (
            read(engine, self.projection.rates_buffer()),
            read(engine, self.projection.impulses_buffer()),
            read(engine, self.projection.normal_velocities_buffer()),
            status.try_into().unwrap(),
        )
    }
}

pub(super) fn solve(mut matrix: Vec<f64>, mut rhs: Vec<f64>) -> Vec<f64> {
    let n = rhs.len();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| matrix[i * n + k].abs().total_cmp(&matrix[j * n + k].abs()))
            .unwrap();
        for j in 0..n {
            matrix.swap(k * n + j, pivot * n + j);
        }
        rhs.swap(k, pivot);
        assert!(matrix[k * n + k].abs() > 1e-15);
        for i in k + 1..n {
            let scale = matrix[i * n + k] / matrix[k * n + k];
            for j in k..n {
                matrix[i * n + j] -= scale * matrix[k * n + j];
            }
            rhs[i] -= scale * rhs[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (rhs[i] - (i + 1..n).map(|j| matrix[i * n + j] * x[j]).sum::<f64>())
            / matrix[i * n + i];
    }
    x
}

fn oracle(mass: &[f32], free: &[f32], rows: &[f32], contacts: &[[f32; 4]]) -> (Vec<f64>, Vec<f64>) {
    let n = free.len();
    assert!(contacts.len() <= 10);
    for mask in 0..1usize << contacts.len() {
        let mut h: Vec<f64> = mass.iter().map(|&x| x as f64).collect();
        let mut rhs: Vec<_> = (0..n)
            .map(|i| (0..n).map(|j| h[i * n + j] * free[j] as f64).sum())
            .collect();
        for (c, node) in contacts
            .iter()
            .enumerate()
            .filter(|(c, _)| mask & (1 << c) != 0)
        {
            for i in 0..n {
                let ai = rows[c * n + i] as f64;
                rhs[i] += node[0] as f64 * ai * (node[1] as f64 + node[2] as f64);
                for j in 0..n {
                    h[i * n + j] += node[0] as f64 * ai * rows[c * n + j] as f64;
                }
            }
        }
        let q = solve(h, rhs);
        let violations: Vec<_> = contacts
            .iter()
            .enumerate()
            .map(|(c, node)| {
                (0..n).map(|i| rows[c * n + i] as f64 * q[i]).sum::<f64>()
                    - node[1] as f64
                    - node[2] as f64
            })
            .collect();
        if violations.iter().enumerate().all(|(c, &z)| {
            contacts[c][0] == 0.0
                || (if mask & (1 << c) != 0 {
                    z >= -1e-10
                } else {
                    z <= 1e-10
                })
        }) {
            return (
                q,
                violations
                    .iter()
                    .zip(contacts)
                    .map(|(&z, node)| node[0] as f64 * z.max(0.0))
                    .collect(),
            );
        }
    }
    panic!("no feasible exhaustive active set");
}

fn verify(
    mass: &[f32],
    free: &[f32],
    rows: &[f32],
    contacts: &[[f32; 4]],
    actual: &(Vec<f32>, Vec<f32>, Vec<f32>, [u32; 4]),
) {
    let n = free.len();
    assert_eq!(actual.3[0], 0, "normal solver fault: {:?}", actual.3);
    let (expected, impulses) = oracle(mass, free, rows, contacts);
    for (&a, b) in actual.0.iter().zip(expected) {
        assert!((a as f64 - b).abs() <= 2e-5, "rate {a} vs {b}");
    }
    for (&a, b) in actual.1.iter().zip(impulses) {
        assert!((a as f64 - b).abs() <= 2e-5, "impulse {a} vs {b}");
    }
    for i in 0..n {
        let error = (0..n)
            .map(|j| mass[i * n + j] as f64 * (actual.0[j] as f64 - free[j] as f64))
            .sum::<f64>()
            + contacts
                .iter()
                .enumerate()
                .map(|(c, _)| rows[c * n + i] as f64 * actual.1[c] as f64)
                .sum::<f64>();
        let scale = (0..n)
            .map(|j| (mass[i * n + j] as f64 * free[j] as f64).abs())
            .sum::<f64>()
            .max(1.0);
        assert!(
            error.abs() <= 1e-6 * scale,
            "generalized momentum error {error} scale {scale}"
        );
    }
    for (c, node) in contacts.iter().enumerate() {
        assert!(actual.1[c] >= 0.0);
        if node[0] == 0.0 {
            assert_eq!(actual.1[c], 0.0);
            continue;
        }
        let gap = actual.2[c] as f64
            - (0..n)
                .map(|i| rows[c * n + i] as f64 * actual.0[i] as f64)
                .sum::<f64>()
            + node[2] as f64;
        assert!(gap >= -2e-5, "negative gap velocity {gap}");
        assert!(
            (gap * actual.1[c] as f64).abs() <= 2e-5,
            "complementarity {gap}"
        );
        assert!(
            (node[0] as f64 * (actual.2[c] as f64 - node[1] as f64) - actual.1[c] as f64).abs()
                <= 2e-5
        );
    }
    let energy = |q: &[f32], v: &[f32]| {
        0.5 * (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| q[i] as f64 * mass[i * n + j] as f64 * q[j] as f64)
                    .sum::<f64>()
            })
            .sum::<f64>()
            + 0.5
                * contacts
                    .iter()
                    .zip(v)
                    .map(|(node, &v)| node[0] as f64 * (v as f64).powi(2))
                    .sum::<f64>()
    };
    let before = energy(
        free,
        &contacts.iter().map(|node| node[1]).collect::<Vec<_>>(),
    );
    assert!(
        energy(&actual.0, &actual.2) <= before + 1e-6 * before.max(1.0),
        "kinetic energy increased"
    );
}

#[test]
fn normal_workgroup_storage_reserves_per_variable_alignment() {
    assert_eq!(storage_bytes(1), 528);
    assert_eq!(storage_bytes(6), 928);
    assert_eq!(storage_bytes(32), 9632);
    assert_eq!(hard_storage_bytes(1), 528);
    assert_eq!(hard_storage_bytes(6), 1232);
    assert_eq!(hard_storage_bytes(32), 18128);
}

#[test]
#[ignore = "requires GPU"]
fn ordered_contact_cache_preserves_overflow_hard_rows_and_reuse() {
    let engine = Engine::new_gpu().unwrap();
    let source = include_str!("../shaders/soil_normal_cached_candidate.wgsl");
    assert_eq!(source.matches("    cache_contacts(lane);").count(), 1);
    let reference = source.replace("    cache_contacts(lane);", "    cached_count=257u;");
    let slots = 2049usize;
    let inputs = [
        buffer(&engine, &[3.0, 0.0, 0.0, 2.0]),
        buffer(&engine, &[2.0, -0.3]),
        buffer(&engine, &vec![0.0; 2 * slots]),
        buffer(&engine, &vec![0.0; 4 * slots]),
    ];
    let variants = [reference.as_str(), source].map(|shader| Fixture {
        buffers: inputs.clone(),
        projection: ExperimentalSoilNormalProjectionGpu::new_shader(
            &engine,
            SoilNormalInputs {
                mass: &inputs[0],
                free_rates: &inputs[1],
                rows: &inputs[2],
                contacts: &inputs[3],
            },
            SoilNormalCapacity {
                dofs: 2,
                contacts: slots as u32,
                max_iterations: 64,
                max_owned_bytes: u64::MAX,
            },
            true,
            shader,
        )
        .unwrap(),
    });
    for (count, hard) in [
        (0, false),
        (1, false),
        (255, false),
        (256, false),
        (257, false),
        (512, false),
        (0, true),
        (255, true),
        (256, true),
        (512, true),
        (1, true),
        (0, false),
    ] {
        let mut rows = vec![0.0f32; 2 * slots];
        let mut contacts = vec![0.0f32; 4 * slots];
        let mut soil_mass = 0.0f64;
        for i in 0..count {
            let slot = if i + 1 == count { slots - 2 } else { 3 * i + 1 };
            let mass = 0.25 + (i % 7) as f32 * 0.125;
            contacts[4 * slot] = mass;
            rows[2 * slot] = 1.0;
            soil_mass += f64::from(mass);
        }
        if hard {
            contacts[4 * (slots - 1) + 3] = 1.0;
            rows[2 * (slots - 1) + 1] = -1.0;
        }
        let queue = engine.queue().unwrap();
        queue.write_buffer(&inputs[2], 0, bytemuck::cast_slice(&rows));
        queue.write_buffer(&inputs[3], 0, bytemuck::cast_slice(&contacts));
        let a = variants[0].run(&engine, true);
        let b = variants[1].run(&engine, true);
        assert_eq!(a.3, b.3, "status count={count} hard={hard}");
        assert_eq!(b.3[0], 0, "count={count} hard={hard}");
        for (old, new) in [(&a.0, &b.0), (&a.1, &b.1), (&a.2, &b.2)] {
            assert_eq!(
                old.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                new.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                "count={count} hard={hard}"
            );
        }
        assert!((f64::from(b.0[0]) - 6.0 / (3.0 + soil_mass)).abs() <= 1e-6);
        assert!((b.0[1] - if hard { 0.0 } else { -0.3 }).abs() <= 1e-6);
    }
    engine.queue().unwrap().write_buffer(
        &inputs[2],
        ((2 * slots - 2) * 4) as u64,
        bytemuck::cast_slice(&[f32::NAN]),
    );
    for fixture in &variants {
        assert_eq!(fixture.run(&engine, true).3[0], NORMAL_INPUT);
    }
    engine.queue().unwrap().write_buffer(
        &inputs[2],
        ((2 * slots - 2) * 4) as u64,
        bytemuck::cast_slice(&[0.0f32]),
    );
    for fixture in &variants {
        assert_eq!(fixture.run(&engine, true).3[0], 0);
    }
}

#[test]
fn normal_hinge_curvature_matches_objective_across_activation() {
    for z in [-2.0f64, -0.1, 0.0, 0.1, 2.0] {
        for delta in [-4.0, -0.25, 0.0, 0.25, 4.0] {
            for step in [0.125, 0.5, 1.0] {
                let next = z + step * delta;
                let curvature = if z > 0.0 {
                    if next > 0.0 {
                        step * step * delta * delta
                    } else {
                        z * (-z - 2.0 * step * delta)
                    }
                } else if next > 0.0 {
                    next * next
                } else {
                    0.0
                };
                let direct = 0.5 * (next.max(0.0).powi(2) - z.max(0.0).powi(2));
                assert!(curvature >= 0.0);
                assert!((direct - step * z.max(0.0) * delta - 0.5 * curvature).abs() < 1e-12);
            }
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn normal_free_body_conserves_linear_and_angular_impulse() {
    let engine = Engine::new_gpu().unwrap();
    let diagonal = [860.0f32, 860.0, 860.0, 300.0, 500.0, 400.0];
    let mut mass = vec![0.0; 36];
    for i in 0..6 {
        mass[6 * i + i] = diagonal[i];
    }
    let free = [0.2, -0.3, 0.1, 0.05, -0.1, 0.15];
    let mut rows = Vec::new();
    let mut contacts = Vec::new();
    for c in 0..6 {
        let mut normal = molla_math::Vec3::ZERO;
        normal[c / 2] = if c % 2 == 0 { 1.0 } else { -1.0 };
        let arm = molla_math::Vec3::new(0.7 - 0.2 * c as f64, -0.4, 0.3);
        let angular = arm.cross(normal);
        rows.extend(
            normal
                .to_array()
                .into_iter()
                .chain(angular.to_array())
                .map(|x| x as f32),
        );
        contacts.push([0.47464246 + 0.1 * c as f32, 0.0, 0.01, 0.0]);
    }
    let actual = Fixture::new(&engine, &mass, &free, &rows, &contacts, 32).run(&engine, true);
    verify(&mass, &free, &rows, &contacts, &actual);
    let mut error = [0.0f64; 6];
    for i in 0..6 {
        error[i] = diagonal[i] as f64 * (actual.0[i] as f64 - free[i] as f64);
        for c in 0..6 {
            error[i] += rows[c * 6 + i] as f64
                * contacts[c][0] as f64
                * (actual.2[c] as f64 - contacts[c][1] as f64);
        }
        assert!(
            error[i].abs() < 2e-5,
            "spatial impulse error axis {i}: {}",
            error[i]
        );
    }
    println!("free-body linear/angular impulse errors={error:?}");
}

#[test]
#[ignore = "requires GPU"]
fn normal_mass_gap_separation_and_continuity() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    for (machine, soil, free, velocity, gap) in [
        (2.0, 1.0, 1.0, 0.0, 0.0),
        (1000.0, 1.0, 1.0, 0.0, 0.0),
        (1.0, 1000.0, 1.0, 0.0, 0.0),
        (2.0, 0.001, 1.0, 0.0, 0.0),
        (2.0, 1.0, -1.0, 0.0, 0.0),
        (2.0, 1.0, 0.0, 1.0, 0.0),
        (2.0, 1.0, 1.0, 0.0, 2.0),
        (2.0, 1.0, 1.0, 0.0, 0.5),
        (2.0, 0.0, 1.0, -2.0, 0.0),
    ] {
        let contacts = [[soil, velocity, gap, 0.0]];
        let fixture = Fixture::new(&engine, &[machine], &[free], &[1.0], &contacts, 32);
        let actual = fixture.run(&engine, true);
        println!(
            "scalar mass={machine} soil={soil} free={free} velocity={velocity} gap={gap} status={:?}",
            actual.3
        );
        if actual.3[0] != 0 {
            if let Some(path) = std::env::var_os("SOIL_NORMAL_SCALAR_TRACE_DIR") {
                super::replay::trace_replay(
                    &engine,
                    &fixture.buffers,
                    &fixture.projection,
                    SoilNormalCapacity {
                        dofs: 1,
                        contacts: 1,
                        max_iterations: 32,
                        max_owned_bytes: u64::MAX,
                    },
                    false,
                    &std::path::PathBuf::from(path),
                );
            }
        }
        verify(&[machine], &[free], &[1.0], &contacts, &actual);
        let denominator = 1.0f64 / machine as f64 + 1.0 / soil as f64;
        let expected = ((free - velocity - gap) as f64).max(0.0) / denominator;
        assert!((actual.1[0] as f64 - expected).abs() < 2e-5);
        if free <= velocity + gap || soil == 0.0 {
            assert_eq!(actual.0[0].to_bits(), free.to_bits());
            assert_eq!(actual.1[0], 0.0);
        }
        let repeated = fixture.run(&engine, true);
        assert_eq!(actual, repeated);
    }
    let fixture = Fixture::new(
        &engine,
        &[2.0],
        &[1.0],
        &[1.0],
        &[[0.47464246, 0.0, 1.0, 0.0]],
        32,
    );
    let mut last = 0.0f32;
    for gap in [1.000002, 1.000001, 1.0, 0.999999, 0.999998] {
        let contact = [[0.47464246, 0.0, gap, 0.0]];
        engine.queue().unwrap().write_buffer(
            &fixture.buffers[3],
            0,
            bytemuck::cast_slice(&contact),
        );
        let actual = fixture.run(&engine, true);
        verify(&[2.0], &[1.0], &[1.0], &contact, &actual);
        assert!((actual.1[0] - last).abs() < 2e-6);
        last = actual.1[0];
    }
    let empty = Fixture::new(&engine, &[2.0], &[-0.0], &[], &[], 32).run(&engine, true);
    assert_eq!(empty.3[0], 0);
    assert_eq!(empty.0[0].to_bits(), (-0.0f32).to_bits());
    println!(
        "normal contact: analytic masses, gap activation, withdrawal, empty state and continuity pass"
    );
}

#[test]
#[ignore = "requires GPU"]
fn normal_multi_contact_matches_exhaustive_f64() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut seed = 0x473f186du32;
    let mut random = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16777216.0 * 2.0 - 1.0
    };
    let mut maximum_iterations = 0;
    let mut backtracks = 0;
    let switching = [[10.0, -1.0, 0.0, 0.0], [100.0, 0.1, 0.0, 0.0]];
    let fixture = Fixture::new(&engine, &[1.0], &[0.0], &[1.0, -1.0], &switching, 32);
    let actual = fixture.run(&engine, true);
    verify(&[1.0], &[0.0], &[1.0, -1.0], &switching, &actual);
    assert!(
        actual.3[2] > 0,
        "active-set crossing must exercise backtracking"
    );
    backtracks += actual.3[2];
    for case in 0..36 {
        let n = if case < 32 { 1 + case % 8 } else { 32 };
        let basis: Vec<_> = (0..n * n).map(|_| random()).collect();
        let mut mass = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..=i {
                let value = (0..n)
                    .map(|k| basis[k * n + i] * basis[k * n + j])
                    .sum::<f32>()
                    + if i == j { 0.5 } else { 0.0 };
                mass[i * n + j] = value;
                mass[j * n + i] = value;
            }
        }
        let free: Vec<_> = (0..n).map(|_| random() * 2.0).collect();
        let rows: Vec<_> = (0..6 * n).map(|_| random()).collect();
        let contacts: Vec<_> = (0..6)
            .map(|_| {
                [
                    0.1 + random().abs() * 3.0,
                    random(),
                    random().abs() * 0.25,
                    0.0,
                ]
            })
            .collect();
        let fixture = Fixture::new(&engine, &mass, &free, &rows, &contacts, 32);
        let actual = fixture.run(&engine, true);
        println!("case={case} width={n} status={:?}", actual.3);
        verify(&mass, &free, &rows, &contacts, &actual);
        maximum_iterations = maximum_iterations.max(actual.3[1]);
        backtracks += actual.3[2];
    }
    println!(
        "36 exhaustive f64 cases, max iterations={maximum_iterations}, backtracks={backtracks}"
    );
}

#[test]
#[ignore = "requires GPU"]
fn normal_contact_extent_resets_outputs_and_validates_padding() {
    let engine = Engine::new_gpu().unwrap();
    let mut contacts = vec![[0.0f32, 2.0, 0.0, 0.0]; 129];
    let fixture = Fixture::new(&engine, &[2.0], &[1.0], &[1.0; 129], &contacts, 32);
    for active in [Some(128), Some(64), Some(63), Some(0), None] {
        contacts.fill([0.0, 2.0, 0.0, 0.0]);
        if let Some(i) = active {
            contacts[i] = [1.0, 0.0, 0.0, 0.0];
        }
        engine.queue().unwrap().write_buffer(
            &fixture.buffers[3],
            0,
            bytemuck::cast_slice(&contacts),
        );
        let actual = fixture.run(&engine, true);
        assert_eq!(actual.3[0], 0);
        let expected = if active.is_some() { 2.0 / 3.0 } else { 1.0 };
        assert!((actual.0[0] - expected).abs() < 1e-6);
        for i in 0..129 {
            if Some(i) == active {
                assert!((actual.1[i] - 2.0 / 3.0).abs() < 1e-6);
                assert!((actual.2[i] - 2.0 / 3.0).abs() < 1e-6);
            } else {
                assert_eq!(actual.1[i], 0.0);
                assert_eq!(actual.2[i], 2.0);
            }
        }
    }
    engine.queue().unwrap().write_buffer(
        &fixture.buffers[2],
        128 * 4,
        bytemuck::cast_slice(&[f32::NAN]),
    );
    assert_eq!(fixture.run(&engine, true).3[0], NORMAL_INPUT);
}

#[test]
#[ignore = "requires GPU"]
fn normal_faults_are_sticky_and_preflight_is_allocation_free() {
    let engine = Engine::new_gpu().unwrap();
    let fixture = Fixture::new(&engine, &[2.0], &[1.0], &[1.0], &[[1.0, 0.0, 0.0, 0.0]], 1);
    engine.queue().unwrap().write_buffer(
        fixture.projection.rates_buffer(),
        0,
        bytemuck::cast_slice(&[99.0f32]),
    );
    let exhausted = fixture.run(&engine, true);
    assert_eq!(exhausted.3[0], NORMAL_ITERATIONS);
    assert_eq!(exhausted.0, [99.0]);
    engine
        .queue()
        .unwrap()
        .write_buffer(&fixture.buffers[1], 0, bytemuck::cast_slice(&[-1.0f32]));
    assert_eq!(fixture.run(&engine, false).0, [99.0]);
    assert_eq!(fixture.run(&engine, true).0, [-1.0]);
    engine.queue().unwrap().write_buffer(
        &fixture.buffers[3],
        0,
        bytemuck::cast_slice(&[1.0f32, 0.0, -1.0, 0.0]),
    );
    assert_eq!(fixture.run(&engine, true).3[0], NORMAL_INPUT);
    engine
        .queue()
        .unwrap()
        .write_buffer(&fixture.buffers[0], 0, bytemuck::cast_slice(&[-2.0f32]));
    assert_eq!(fixture.run(&engine, true).3[0], NORMAL_MASS);
    engine
        .queue()
        .unwrap()
        .write_buffer(&fixture.buffers[0], 0, bytemuck::cast_slice(&[2.0f32]));
    engine.queue().unwrap().write_buffer(
        &fixture.buffers[3],
        0,
        bytemuck::cast_slice(&[1.0f32, f32::NAN, 0.0, 0.0]),
    );
    assert_eq!(fixture.run(&engine, true).3[0], NORMAL_INPUT);
    let asymmetric = Fixture::new(
        &engine,
        &[2.0, 0.5, 0.5001, 2.0],
        &[0.0, 0.0],
        &[1.0, 0.0],
        &[[1.0, 0.0, 0.0, 0.0]],
        32,
    );
    assert_eq!(asymmetric.run(&engine, true).3[0], NORMAL_MASS);
    engine.queue().unwrap().write_buffer(
        &fixture.buffers[3],
        0,
        bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, 1.0]),
    );
    assert_eq!(fixture.run(&engine, true).3[0], NORMAL_INPUT);
    assert_eq!(fixture.projection.owned_buffer_bytes(), 44);
    let create = |capacity| {
        ExperimentalSoilNormalProjectionGpu::new(
            &engine,
            SoilNormalInputs {
                mass: &fixture.buffers[0],
                free_rates: &fixture.buffers[1],
                rows: &fixture.buffers[2],
                contacts: &fixture.buffers[3],
            },
            capacity,
        )
    };
    for capacity in [
        SoilNormalCapacity {
            dofs: u32::MAX,
            contacts: 1,
            max_iterations: 32,
            max_owned_bytes: u64::MAX,
        },
        SoilNormalCapacity {
            dofs: 1,
            contacts: 65537,
            max_iterations: 32,
            max_owned_bytes: u64::MAX,
        },
        SoilNormalCapacity {
            dofs: 1,
            contacts: 1,
            max_iterations: 0,
            max_owned_bytes: u64::MAX,
        },
        SoilNormalCapacity {
            dofs: 1,
            contacts: 1,
            max_iterations: 32,
            max_owned_bytes: fixture.projection.owned_buffer_bytes() - 1,
        },
        SoilNormalCapacity {
            dofs: 2,
            contacts: 1,
            max_iterations: 32,
            max_owned_bytes: u64::MAX,
        },
    ] {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        assert!(create(capacity).is_err());
        assert_eq!(allocations.finish().observations, 0);
    }
    println!(
        "normal contact: nonconvergence, sticky faults, reset and allocation-free preflight pass"
    );
}
