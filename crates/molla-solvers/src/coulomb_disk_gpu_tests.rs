use super::*;
use molla_compute::Engine;

fn input(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = storage(device, "disk test input", bytes.len().max(4) as u64);
    if !bytes.is_empty() {
        molla_core::gpu_write_buffer!(queue, &buffer, 0, bytes);
    }
    buffer
}
fn staging(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("disk test readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    )
}
fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = molla_core::gpu_readback!(buffer.slice(..)).to_vec();
    buffer.unmap();
    bytes
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-friction-gpu"]
fn rounded_disk_boundary_gap_matches_f64() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cases = Vec::new();
    for scale in [1e-12_f32, 1.0, 1e12] {
        let radius = 0.00078484364_f32 * scale;
        for i in 0..65 {
            let angle = i as f64 * std::f64::consts::TAU / 64.0;
            for offset in [-1_i32, 0, 1] {
                let length = f32::from_bits(radius.to_bits().wrapping_add_signed(offset));
                cases.push([
                    length * angle.cos() as f32,
                    length * angle.sin() as f32,
                    radius,
                    0.0,
                ]);
            }
        }
    }
    let source = format!(
        "{}\n{}",
        include_str!("../shaders/coulomb_disk.wgsl"),
        r#"
@compute @workgroup_size(1)
fn probe(@builtin(global_invocation_id) id: vec3<u32>) {
    let data = contact_data[id.x];
    solution_words[id.x*8u] = bitcast<u32>(relative_disk_gap(data.xy, data.z, id.x));
}
"#
    );
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("disk gap probe"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("disk gap probe"),
        layout: None,
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let data = input(device, queue, bytemuck::cast_slice(&cases));
    let scratch = storage(device, "disk gap scratch", cases.len() as u64 * 4);
    let output = storage(device, "disk gap output", cases.len() as u64 * 32);
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("disk gap probe"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 2,
                resource: data.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: scratch.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let snapshot = staging(device, cases.len() as u64 * 32);
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups(cases.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&output, 0, &snapshot, 0, snapshot.size());
    queue.submit([encoder.finish()]);
    let bytes = read(device, &snapshot);
    let mut peak: f64 = 0.0;
    for (i, (data, frame)) in cases.iter().zip(bytes.chunks_exact(32)).enumerate() {
        let got = bytemuck::pod_read_unaligned::<f32>(&frame[..4]) as f64;
        let [x, y, radius, _] = data.map(f64::from);
        let expected = (radius * radius - x * x - y * y) / (radius * radius);
        peak = peak.max((got - expected).abs());
        assert!(
            (got - expected).abs() < 1e-13,
            "case={i} input={data:?} got={got:e} expected={expected:e}"
        );
    }
    eprintln!("disk boundary squared-gap peak error={peak:e}");
}

#[derive(Clone)]
struct Fixture {
    gram: Vec<f32>,
    free: Vec<f32>,
    contacts: Vec<CoulombDiskInput>,
}

fn fixture(which: usize) -> Fixture {
    let uniform = |cap: f32, moving: bool| CoulombDiskInput {
        static_cap: cap,
        dynamic_cap: cap,
        initial_velocity: if moving { [1.0, 0.0] } else { [0.0; 2] },
    };
    match which {
        0 => Fixture {
            gram: vec![2.0, 0.5, 0.5, 3.0],
            free: vec![0.1, 0.2],
            contacts: vec![CoulombDiskInput {
                static_cap: 0.4,
                dynamic_cap: 0.1,
                initial_velocity: [0.0; 2],
            }],
        },
        1 => Fixture {
            gram: vec![0.1, 0.02, 0.02, 3.0],
            free: vec![1.0, -0.6],
            contacts: vec![uniform(0.1, true)],
        },
        2 => Fixture {
            gram: vec![
                1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0,
            ],
            free: vec![0.3, 0.4, 0.3, 0.4],
            contacts: vec![uniform(0.3, true), uniform(0.1, true)],
        },
        3 => Fixture {
            gram: vec![0.0; 4],
            free: vec![1.0, 2.0],
            contacts: vec![uniform(0.2, true)],
        },
        4 => Fixture {
            gram: vec![1.0, 0.0, 0.0, 1.0],
            free: vec![0.3, 0.0],
            contacts: vec![CoulombDiskInput {
                static_cap: 0.2,
                dynamic_cap: 0.05,
                initial_velocity: [0.0; 2],
            }],
        },
        5 => {
            use molla_math::Vec3;
            let mut jacobian = Vec::new();
            let mut free = Vec::new();
            for x in [-0.4, 0.4] {
                for z in [-0.4, 0.4] {
                    let arm = Vec3::new(x, -0.5, z);
                    for direction in [Vec3::Z, Vec3::X] {
                        let angular = arm.cross(direction);
                        jacobian.push([
                            direction.x,
                            direction.y,
                            direction.z,
                            angular.x,
                            angular.y,
                            angular.z,
                        ]);
                        free.push(
                            (direction.dot(Vec3::new(0.003, 0.0, 0.002)) + angular.y * 0.001)
                                as f32,
                        );
                    }
                }
            }
            let mut gram = vec![0.0; 64];
            for i in 0..8 {
                for j in 0..8 {
                    gram[i * 8 + j] = (0..6)
                        .map(|k| jacobian[i][k] * jacobian[j][k] * if k < 3 { 0.5 } else { 2.0 })
                        .sum::<f64>() as f32;
                }
            }
            Fixture {
                gram,
                free,
                contacts: vec![
                    CoulombDiskInput {
                        static_cap: 0.02,
                        dynamic_cap: 0.005,
                        initial_velocity: [0.0; 2]
                    };
                    4
                ],
            }
        }
        _ => {
            let n = 12;
            let j: Vec<f32> = (0..n * 4)
                .map(|i| ((i * 13 % 19) as f32 - 9.0) * 0.05)
                .collect();
            let mut gram = vec![0.0; n * n];
            for r in 0..n {
                for c in 0..n {
                    gram[r * n + c] = (0..4).map(|k| j[r * 4 + k] * j[c * 4 + k]).sum::<f32>()
                        + if r == c { 0.05 } else { 0.0 };
                }
            }
            Fixture {
                gram,
                free: (0..n).map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.07).collect(),
                contacts: (0..6)
                    .map(|i| uniform(0.15 + i as f32 * 0.05, true))
                    .collect(),
            }
        }
    }
}

fn oracle(f: &Fixture) -> (Vec<f64>, Vec<f64>, Vec<bool>) {
    let count = f.contacts.len();
    let n = count * 2;
    let gram: Vec<_> = f.gram.iter().map(|&x| x as f64).collect();
    let free: Vec<_> = f.free.iter().map(|&x| x as f64).collect();
    let mut moving: Vec<_> = f
        .contacts
        .iter()
        .map(|c| (c.initial_velocity[0] as f64).hypot(c.initial_velocity[1] as f64) > 1e-4)
        .collect();
    for _ in 0..=count {
        let caps: Vec<_> = f
            .contacts
            .iter()
            .zip(&moving)
            .map(|(c, &m)| {
                if m {
                    c.dynamic_cap as f64
                } else {
                    c.static_cap as f64
                }
            })
            .collect();
        let (impulse, _) = crate::contact_friction::disks(&gram, &free, &caps).unwrap();
        let slip: Vec<_> = (0..n)
            .map(|i| free[i] + (0..n).map(|j| gram[i * n + j] * impulse[j]).sum::<f64>())
            .collect();
        let mut changed = false;
        for c in 0..count {
            if !moving[c] && slip[2 * c].hypot(slip[2 * c + 1]) > 1e-4 {
                moving[c] = true;
                changed = true;
            }
        }
        if !changed {
            return (impulse, slip, moving);
        }
    }
    panic!("CPU reference regime transition did not converge");
}

struct Sequence {
    gram: wgpu::Buffer,
    free: wgpu::Buffer,
    contacts: wgpu::Buffer,
    counts: wgpu::Buffer,
    count: wgpu::Buffer,
    status: wgpu::Buffer,
    solver: ResidentCoulombDisks,
    config: CoulombDiskConfig,
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-friction-gpu"]
fn nearly_loaded_static_disks_cancel_slip_across_scales_and_directions() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let sequence = Sequence::new(device, 8192);
    let mut frames = Vec::new();
    for scale in [1e-6_f32, 1.0, 1e6] {
        for angle in [0.0_f32, 0.37, 1.2] {
            for load in [0.8_f32, 0.99, 0.999, 0.9999] {
                let velocity = [0.005 * load * angle.cos(), 0.005 * load * angle.sin()];
                let f = Fixture {
                    gram: vec![
                        1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0,
                        1.0,
                    ]
                    .into_iter()
                    .map(|g| g / scale)
                    .collect(),
                    free: velocity.into_iter().cycle().take(4).collect(),
                    contacts: [0.001, 0.004]
                        .into_iter()
                        .map(|cap| CoulombDiskInput {
                            static_cap: cap * scale,
                            dynamic_cap: cap * scale * 0.5,
                            initial_velocity: [0.0; 2],
                        })
                        .collect(),
                };
                frames.push((f, 1, 2, Some(0)));
            }
        }
    }
    let output = sequence.run(device, queue, &frames);
    let mut peak: f64 = 0.0;
    for (case, ((status, rows, diagnostic), (f, _, _, _))) in output.iter().zip(&frames).enumerate()
    {
        assert_eq!(*status, 0, "case={case} diagnostic={diagnostic:?}");
        assert!(host_kkt(f, rows) <= 1e-7);
        let impulse: Vec<_> = rows[..2]
            .iter()
            .flat_map(|r| r.impulse.map(f64::from))
            .collect();
        for c in 0..2 {
            assert_eq!(rows[c].regime, 2);
            let slip = [2 * c, 2 * c + 1].map(|r| {
                f.free[r] as f64
                    + (0..4)
                        .map(|k| f.gram[r * 4 + k] as f64 * impulse[k])
                        .sum::<f64>()
            });
            peak = peak.max(slip[0].hypot(slip[1]));
        }
    }
    eprintln!("nearly loaded static peak slip={peak:e}");
    assert!(peak <= 1e-7, "feasible static speed={peak:e}");
}

impl Sequence {
    fn new(device: &wgpu::Device, iterations: u32) -> Self {
        let config = CoulombDiskConfig {
            island_capacity: 1,
            contacts_per_island: 8,
            max_iterations: iterations,
            velocity_tolerance: 1e-7,
        };
        let gram = storage(device, "sequence gram", 16 * 16 * 4);
        let free = storage(device, "sequence free", 16 * 4);
        let contacts = storage(device, "sequence contacts", 8 * 16);
        let counts = storage(device, "sequence counts", 4);
        let count = storage(device, "sequence count", 4);
        let status = storage(device, "sequence status", 4);
        let solver = ResidentCoulombDisks::new(
            device, &gram, &free, &contacts, &counts, &count, &status, config,
        )
        .unwrap();
        Self {
            gram,
            free,
            contacts,
            counts,
            count,
            status,
            solver,
            config,
        }
    }

    fn run(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        frames: &[(Fixture, u32, u32, Option<u32>)],
    ) -> Vec<(u32, Vec<CoulombDiskSolution>, [u32; 4])> {
        let output_stride = 4 + 8 * 32 + 16;
        let output = staging(device, frames.len() as u64 * output_stride);
        let mut encoder = device.create_command_encoder(&Default::default());
        for (step, (f, islands, count, reset)) in frames.iter().enumerate() {
            let n = f.free.len();
            let mut gram = vec![0_f32; 256];
            let mut free = [0_f32; 16];
            let mut contacts = [CoulombDiskInput::default(); 8];
            for r in 0..n {
                gram[r * 16..r * 16 + n].copy_from_slice(&f.gram[r * n..(r + 1) * n]);
            }
            free[..n].copy_from_slice(&f.free);
            contacts[..f.contacts.len()].copy_from_slice(&f.contacts);
            for (target, bytes) in [
                (&self.gram, bytemuck::cast_slice(&gram)),
                (&self.free, bytemuck::cast_slice(&free)),
                (&self.contacts, bytemuck::cast_slice(&contacts)),
                (&self.counts, bytemuck::bytes_of(count)),
                (&self.count, bytemuck::bytes_of(islands)),
            ] {
                let source = input(device, queue, bytes);
                encoder.copy_buffer_to_buffer(&source, 0, target, 0, bytes.len() as u64);
            }
            if let Some(value) = reset {
                let source = input(device, queue, bytemuck::bytes_of(value));
                encoder.copy_buffer_to_buffer(&source, 0, &self.status, 0, 4);
            }
            self.solver.encode(&mut encoder);
            let offset = step as u64 * output_stride;
            encoder.copy_buffer_to_buffer(&self.status, 0, &output, offset, 4);
            encoder.copy_buffer_to_buffer(
                self.solver.solution_buffer(),
                0,
                &output,
                offset + 4,
                256,
            );
            encoder.copy_buffer_to_buffer(
                self.solver.diagnostic_buffer(),
                0,
                &output,
                offset + 260,
                16,
            );
        }
        queue.submit([encoder.finish()]);
        read(device, &output)
            .chunks_exact(output_stride as usize)
            .map(|frame| {
                (
                    bytemuck::pod_read_unaligned(&frame[..4]),
                    frame[4..260]
                        .chunks_exact(32)
                        .map(bytemuck::pod_read_unaligned)
                        .collect(),
                    bytemuck::pod_read_unaligned(&frame[260..]),
                )
            })
            .collect()
    }
}

fn host_kkt(f: &Fixture, rows: &[CoulombDiskSolution]) -> f64 {
    let n = f.free.len();
    let mut peak: f64 = 0.0;
    for c in 0..f.contacts.len() {
        let cap = rows[c].applied_cap as f64;
        if cap == 0.0 {
            continue;
        }
        let impulse = rows[c].impulse.map(f64::from);
        let length = impulse[0].hypot(impulse[1]);
        let slip = [2 * c, 2 * c + 1].map(|r| {
            f.free[r] as f64
                + (0..n)
                    .map(|j| f.gram[r * n + j] as f64 * rows[j / 2].impulse[j % 2] as f64)
                    .sum::<f64>()
        });
        let speed = slip[0].hypot(slip[1]);
        let error = if length == 0.0 {
            speed
        } else {
            let direction = impulse.map(|v| v / length);
            let pressure = (-(slip[0] * direction[0] + slip[1] * direction[1])).max(0.0);
            let r = 2 * c;
            let curvature = (f.gram[r * n + r].abs() as f64 + f.gram[r * n + r + 1].abs() as f64)
                .max(f.gram[(r + 1) * n + r + 1].abs() as f64 + f.gram[r * n + r + 1].abs() as f64);
            (slip[0] + pressure * direction[0])
                .hypot(slip[1] + pressure * direction[1])
                .max(speed * (1.0 - length / cap).abs())
                .max(speed.min(curvature * (cap - length).abs()))
        };
        assert!(length <= cap * (1.0 + 2e-6));
        if error > 1e-7 {
            let naive = [2 * c, 2 * c + 1].map(|r| {
                (0..n).fold(f.free[r], |sum, j| {
                    sum + f.gram[r * n + j] * rows[j / 2].impulse[j % 2]
                })
            });
            eprintln!("naive slip={naive:?}");
            eprintln!(
                "contact={c} host slip={slip:?} gpu slip={:?} host KKT={error:e}",
                rows[c].slip
            );
        }
        peak = peak.max(error);
    }
    peak
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-friction-gpu"]
fn rank_deficient_grams_pass_and_indefinite_null_blocks_fail() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let sequence = Sequence::new(device, 4096);
    let mut frames = Vec::new();
    for scale in [1e-12_f32, 1.0, 1e12] {
        for seed in 0..32 {
            let columns: Vec<[f64; 3]> = (0..8)
                .map(|i| {
                    std::array::from_fn(|k| {
                        ((i * 7 + seed * 13) as f64 * 0.37 + k as f64 * 1.17).sin()
                    })
                })
                .collect();
            let mut gram = vec![0.0; 64];
            for i in 0..8 {
                for j in 0..8 {
                    gram[i * 8 + j] =
                        (0..3).map(|k| columns[i][k] * columns[j][k]).sum::<f64>() as f32 * scale;
                }
            }
            frames.push((
                Fixture {
                    gram,
                    free: vec![0.0; 8],
                    contacts: vec![
                        CoulombDiskInput {
                            static_cap: 1.0,
                            dynamic_cap: 0.5,
                            initial_velocity: [0.0; 2]
                        };
                        4
                    ],
                },
                1,
                4,
                Some(0),
            ));
        }
    }
    let valid = frames.len();
    for diagonal in [0.0, 1e-8, -1.0] {
        frames.push((
            Fixture {
                gram: vec![diagonal, 1.0, 1.0, diagonal],
                free: vec![0.0; 2],
                contacts: vec![CoulombDiskInput {
                    static_cap: 1.0,
                    dynamic_cap: 0.5,
                    initial_velocity: [0.0; 2],
                }],
            },
            1,
            1,
            Some(0),
        ));
    }
    for (i, (status, rows, diagnostic)) in
        sequence.run(device, queue, &frames).into_iter().enumerate()
    {
        assert_eq!(
            status,
            if i < valid { 0 } else { FRICTION_FAULT_PSD },
            "case={i} diagnostic={diagnostic:?}"
        );
        assert!(rows.iter().all(|row| row.impulse == [0.0; 2]));
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-friction-gpu"]
fn coupled_disk_gpu_rotations_permutations_and_reciprocal_scales() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let sequence = Sequence::new(device, 8192);
    let base = fixture(6);
    let n = base.free.len();
    let mut frames = Vec::new();
    let mut transforms = Vec::new();
    for scale in [1e-30_f32, 1e-15, 1.0, 1e15, 1e30] {
        for angle in [0_f64, 0.37, 1.2] {
            for reverse in [false, true] {
                let mut transform = vec![0_f64; n * n];
                let mut transformed = base.clone();
                for c in 0..n / 2 {
                    let old = if reverse { n / 2 - 1 - c } else { c };
                    let (s, co) = (angle * (c + 1) as f64).sin_cos();
                    transform[(2 * c) * n + 2 * old] = co;
                    transform[(2 * c) * n + 2 * old + 1] = -s;
                    transform[(2 * c + 1) * n + 2 * old] = s;
                    transform[(2 * c + 1) * n + 2 * old + 1] = co;
                    transformed.contacts[c] = base.contacts[old];
                    transformed.contacts[c].static_cap *= scale;
                    transformed.contacts[c].dynamic_cap *= scale;
                    transformed.contacts[c].initial_velocity = [co as f32, s as f32];
                }
                for r in 0..n {
                    transformed.free[r] = (0..n)
                        .map(|i| transform[r * n + i] * base.free[i] as f64)
                        .sum::<f64>() as f32;
                    for c in 0..n {
                        transformed.gram[r * n + c] = ((0..n)
                            .map(|i| {
                                (0..n)
                                    .map(|j| {
                                        transform[r * n + i]
                                            * base.gram[i * n + j] as f64
                                            * transform[c * n + j]
                                    })
                                    .sum::<f64>()
                            })
                            .sum::<f64>()
                            / scale as f64)
                            as f32;
                    }
                }
                frames.push((transformed, 1, 6, Some(0)));
                transforms.push((transform, scale));
            }
        }
    }
    let (reference, _, _) = oracle(&base);
    let objective = |f: &Fixture, impulse: &[f64]| -> f64 {
        (0..n)
            .map(|r| {
                impulse[r]
                    * (f.free[r] as f64
                        + 0.5
                            * (0..n)
                                .map(|c| f.gram[r * n + c] as f64 * impulse[c])
                                .sum::<f64>())
            })
            .sum()
    };
    let reference_energy = objective(&base, &reference);
    let output = sequence.run(device, queue, &frames);
    let mut peak: f64 = 0.0;
    for (step, ((status, rows, diagnostic), (transform, scale))) in
        output.iter().zip(&transforms).enumerate()
    {
        assert_eq!(
            *status, 0,
            "step={step} scale={scale:e} diagnostic={diagnostic:?}"
        );
        let f = &frames[step].0;
        let residual = host_kkt(f, rows);
        assert!(
            residual <= 1e-7,
            "step={step} host KKT={residual:e} rows={rows:?}"
        );
        peak = peak.max(residual);
        let impulse: Vec<_> = rows[..n / 2]
            .iter()
            .flat_map(|r| r.impulse.map(f64::from))
            .collect();
        let energy = objective(f, &impulse) / *scale as f64;
        assert!(energy <= 0.0, "step={step} energy={energy:e}");
        assert!((energy - reference_energy).abs() < 3e-6);
        for r in 0..n {
            let expected: f64 = (0..n).map(|j| transform[r * n + j] * reference[j]).sum();
            let actual = rows[r / 2].impulse[r % 2] as f64 / *scale as f64;
            assert!(
                (expected - actual).abs() < 4e-6,
                "step={step} row={r}: {actual:e} != {expected:e}"
            );
        }
    }
    eprintln!(
        "disk basis/permutation/scale: {} cases, peak host KKT={peak:e}",
        frames.len()
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-friction-gpu"]
fn coupled_disk_gpu_device_sequence_regimes_and_contact_retirement() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let sequence = Sequence::new(device, 8192);
    let frames: Vec<_> = (0..100)
        .map(|step| {
            let mut f = fixture(4);
            match step % 8 {
                0 => f = fixture(6),
                1 => f.free = vec![0.02, 0.03],
                2 => {}
                3 => {
                    f.free = vec![0.02, 0.03];
                    f.contacts[0].initial_velocity = [0.1, 0.0];
                }
                4 => {
                    f.free = vec![-0.02, -0.03];
                }
                5 => {
                    f.free = vec![-0.3, 0.0];
                    f.contacts[0].initial_velocity = [-0.1, 0.0];
                }
                6 => {
                    f.contacts[0].static_cap = 0.0;
                    f.contacts[0].dynamic_cap = 0.0;
                }
                _ => {}
            }
            let count = if step % 8 == 7 {
                0
            } else {
                f.contacts.len() as u32
            };
            (f, 1, count, None)
        })
        .collect();
    let output = sequence.run(device, engine.queue().unwrap(), &frames);
    for (step, ((f, _, count, _), (status, rows, diagnostic))) in
        frames.iter().zip(&output).enumerate()
    {
        assert_eq!(*status, 0, "step={step} diagnostic={diagnostic:?}");
        for row in &rows[*count as usize..] {
            assert!(bytemuck::bytes_of(row).iter().all(|&v| v == 0));
        }
        if *count == 0 {
            continue;
        }
        let (impulses, _, moving) = oracle(f);
        assert!(host_kkt(f, rows) <= 1e-7);
        for c in 0..*count as usize {
            let cap = if moving[c] {
                f.contacts[c].dynamic_cap
            } else {
                f.contacts[c].static_cap
            };
            assert_eq!(rows[c].applied_cap, cap);
            for k in 0..2 {
                assert!((rows[c].impulse[k] as f64 - impulses[2 * c + k]).abs() < 3e-6);
            }
        }
        if step % 8 == 3 {
            assert_eq!(rows[0].regime, 2);
            assert_eq!(rows[0].applied_cap, 0.05);
        }
        if step % 8 == 4 {
            assert_eq!(rows[0].regime, 2);
            assert_eq!(rows[0].applied_cap, 0.2);
        }
        if step % 8 == 6 {
            assert_eq!(rows[0].regime, 0);
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-friction-gpu"]
fn coupled_disk_gpu_validation_sticky_faults_and_reset() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let sequence = Sequence::new(device, 8192);
    let build = |config, gram: &wgpu::Buffer| {
        ResidentCoulombDisks::new(
            device,
            gram,
            &sequence.free,
            &sequence.contacts,
            &sequence.counts,
            &sequence.count,
            &sequence.status,
            config,
        )
    };
    for config in [
        CoulombDiskConfig {
            island_capacity: 0,
            ..sequence.config
        },
        CoulombDiskConfig {
            contacts_per_island: 0,
            ..sequence.config
        },
        CoulombDiskConfig {
            max_iterations: 0,
            ..sequence.config
        },
        CoulombDiskConfig {
            velocity_tolerance: f32::NAN,
            ..sequence.config
        },
        CoulombDiskConfig {
            velocity_tolerance: -1.0,
            ..sequence.config
        },
        CoulombDiskConfig {
            velocity_tolerance: f32::from_bits(1),
            ..sequence.config
        },
        CoulombDiskConfig {
            contacts_per_island: u32::MAX,
            ..sequence.config
        },
        CoulombDiskConfig {
            island_capacity: u32::MAX,
            ..sequence.config
        },
    ] {
        assert!(build(config, &sequence.gram).is_err());
    }
    assert!(build(sequence.config, &storage(device, "undersized gram", 4)).is_err());
    assert!(build(sequence.config, &staging(device, 1024)).is_err());

    let valid = fixture(0);
    let mut invalid = Vec::new();
    invalid.push((valid.clone(), 2, 1, FRICTION_FAULT_INPUT));
    invalid.push((valid.clone(), 1, 9, FRICTION_FAULT_INPUT));
    for field in 0..8 {
        let mut f = valid.clone();
        let fault = match field {
            0 => {
                f.contacts[0].dynamic_cap = -1.0;
                FRICTION_FAULT_INPUT
            }
            1 => {
                f.contacts[0].dynamic_cap = 1.0;
                FRICTION_FAULT_INPUT
            }
            2 => {
                f.contacts[0].static_cap = f32::NAN;
                FRICTION_FAULT_NONFINITE
            }
            3 => {
                f.contacts[0].initial_velocity[0] = f32::INFINITY;
                FRICTION_FAULT_NONFINITE
            }
            4 => {
                f.free[0] = f32::NAN;
                FRICTION_FAULT_NONFINITE
            }
            5 => {
                f.gram[0] = f32::INFINITY;
                FRICTION_FAULT_NONFINITE
            }
            6 => {
                f.gram[0] = -2.0;
                FRICTION_FAULT_PSD
            }
            _ => {
                f.gram[1] = 0.9;
                FRICTION_FAULT_PSD
            }
        };
        invalid.push((f, 1, 1, fault));
    }
    let mut overflow = valid.clone();
    overflow.gram = vec![1e30, 0.0, 0.0, 1e30];
    overflow.contacts[0].static_cap = 1e30;
    invalid.push((overflow, 1, 1, FRICTION_FAULT_NONFINITE));
    let mut underflow = valid.clone();
    underflow.gram = vec![1e-30, 0.0, 0.0, 1e-30];
    underflow.contacts[0].static_cap = 1e-30;
    underflow.contacts[0].dynamic_cap = 1e-30;
    invalid.push((underflow, 1, 1, FRICTION_FAULT_NONFINITE));
    let mut frames = Vec::new();
    for (f, islands, count, _) in &invalid {
        frames.push((valid.clone(), 1, 1, Some(0)));
        frames.push((f.clone(), *islands, *count, None));
        frames.push((valid.clone(), 0, 0, None));
        frames.push((valid.clone(), 1, 1, Some(0)));
    }
    let output = sequence.run(device, queue, &frames);
    for (case, chunk) in output.chunks_exact(4).enumerate() {
        assert_eq!(chunk[0].0, 0);
        assert!(chunk[0].1[0].impulse.iter().any(|&v| v != 0.0));
        assert_eq!(
            chunk[1].0, invalid[case].3,
            "case={case} diagnostic={:?}",
            chunk[1].2
        );
        assert_eq!(chunk[2].0, chunk[1].0);
        assert_eq!(chunk[2].2, chunk[1].2);
        for frame in &chunk[1..3] {
            assert!(
                bytemuck::cast_slice::<_, u8>(&frame.1)
                    .iter()
                    .all(|&v| v == 0)
            );
        }
        assert_eq!(chunk[3].0, 0);
        assert_eq!(chunk[3].2[0], 0);
        assert!(chunk[3].1[0].impulse.iter().any(|&v| v != 0.0));
    }
    let short = Sequence::new(device, 1);
    let output = short.run(device, queue, &[(fixture(6), 1, 6, Some(0))]);
    assert_eq!(output[0].0, FRICTION_FAULT_CONVERGENCE);
    assert_eq!(output[0].2[1], 1);
    assert_eq!(output[0].2[3], 1);
    assert!(
        bytemuck::cast_slice::<_, u8>(&output[0].1)
            .iter()
            .all(|&v| v == 0)
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-friction-gpu"]
fn coupled_disk_gpu_matches_cpu_and_retires_batched_island_slots() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let config = CoulombDiskConfig {
        island_capacity: 130,
        contacts_per_island: 8,
        max_iterations: 8192,
        velocity_tolerance: 1e-7,
    };
    let width = config.contacts_per_island as usize;
    let stride = width * 2;
    let fixtures: Vec<_> = (0..config.island_capacity)
        .map(|i| fixture(i as usize % 7))
        .collect();
    let references: Vec<_> = fixtures.iter().map(oracle).collect();
    let mut grams = vec![0_f32; fixtures.len() * stride * stride];
    let mut free = vec![0_f32; fixtures.len() * stride];
    let mut data = vec![CoulombDiskInput::default(); fixtures.len() * width];
    let counts: Vec<_> = fixtures.iter().map(|f| f.contacts.len() as u32).collect();
    for (island, f) in fixtures.iter().enumerate() {
        let n = f.free.len();
        for r in 0..n {
            free[island * stride + r] = f.free[r];
            grams[island * stride * stride + r * stride..island * stride * stride + r * stride + n]
                .copy_from_slice(&f.gram[r * n..r * n + n]);
        }
        data[island * width..island * width + f.contacts.len()].copy_from_slice(&f.contacts);
    }
    let gram = input(device, queue, bytemuck::cast_slice(&grams));
    let free = input(device, queue, bytemuck::cast_slice(&free));
    let contacts = input(device, queue, bytemuck::cast_slice(&data));
    let counts = input(device, queue, bytemuck::cast_slice(&counts));
    let count = input(device, queue, bytemuck::bytes_of(&config.island_capacity));
    let status = input(device, queue, &[0; 4]);
    let solver = ResidentCoulombDisks::new(
        device, &gram, &free, &contacts, &counts, &count, &status, config,
    )
    .unwrap();
    assert_eq!(solver.resource_counts(), [4, 1, 1]);
    let bytes_per_frame = 4 + fixtures.len() as u64 * width as u64 * 32;
    let output = staging(device, bytes_per_frame * 3);
    let diagnostics = staging(device, fixtures.len() as u64 * 16);
    let new_counts = input(device, queue, bytemuck::cast_slice(&[130_u32, 1, 0]));
    let mut encoder = device.create_command_encoder(&Default::default());
    for step in 0..3_u64 {
        encoder.copy_buffer_to_buffer(&new_counts, step * 4, &count, 0, 4);
        solver.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&status, 0, &output, step * bytes_per_frame, 4);
        encoder.copy_buffer_to_buffer(
            solver.solution_buffer(),
            0,
            &output,
            step * bytes_per_frame + 4,
            bytes_per_frame - 4,
        );
    }
    encoder.copy_buffer_to_buffer(
        solver.diagnostic_buffer(),
        0,
        &diagnostics,
        0,
        fixtures.len() as u64 * 16,
    );
    queue.submit([encoder.finish()]);
    let bytes = read(device, &output);
    for (island, row) in read(device, &diagnostics).chunks_exact(16).enumerate() {
        let values = bytemuck::pod_read_unaligned::<[u32; 4]>(row);
        if values[0] != 0 {
            eprintln!(
                "island={island} fixture={} fault={} iterations={} residual={:e} phase={}",
                island % 7,
                values[0],
                values[1],
                f32::from_bits(values[2]),
                values[3]
            );
        }
    }
    let mut peak_iterations = 0;
    for (step, frame) in bytes.chunks_exact(bytes_per_frame as usize).enumerate() {
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&frame[..4]),
            0,
            "step={step}"
        );
        for (island, bytes) in frame[4..].chunks_exact(width * 32).enumerate() {
            let f = &fixtures[island];
            let (cpu_impulse, cpu_slip, cpu_moving) = &references[island];
            let mut impulses = Vec::new();
            for (c, row) in bytes.chunks_exact(32).enumerate() {
                if step == 2 || (step == 1 && island > 0) || c >= f.contacts.len() {
                    assert!(row.iter().all(|&v| v == 0));
                    continue;
                }
                let result = bytemuck::pod_read_unaligned::<CoulombDiskSolution>(row);
                peak_iterations = peak_iterations.max(result.iterations);
                let cap = if cpu_moving[c] {
                    f.contacts[c].dynamic_cap
                } else {
                    f.contacts[c].static_cap
                };
                assert_eq!(result.applied_cap, cap, "island={island} contact={c}");
                let length = result.impulse[0].hypot(result.impulse[1]);
                assert!(length <= cap * (1.0 + 2e-6));
                assert!(result.residual <= 1e-7);
                for k in 0..2 {
                    assert!(
                        (result.slip[k] as f64 - cpu_slip[c * 2 + k]).abs() < 3e-6,
                        "island={island} contact={c} slip={:?} cpu={cpu_slip:?}",
                        result.slip
                    );
                    impulses.push(result.impulse[k] as f64);
                }
            }
            if impulses.is_empty() {
                continue;
            }
            let n = impulses.len();
            let energy = |values: &[f64]| {
                (0..n)
                    .map(|i| {
                        values[i]
                            * (f.free[i] as f64
                                + 0.5
                                    * (0..n)
                                        .map(|j| f.gram[i * n + j] as f64 * values[j])
                                        .sum::<f64>())
                    })
                    .sum::<f64>()
            };
            assert!(energy(&impulses) <= 1e-8);
            assert!((energy(&impulses) - energy(cpu_impulse)).abs() < 3e-6);
            for i in 0..n {
                let response: f64 = (0..n)
                    .map(|j| f.gram[i * n + j] as f64 * (impulses[j] - cpu_impulse[j]))
                    .sum();
                assert!(
                    response.abs() < 3e-6,
                    "island={island} row={i} response={response:e}"
                );
            }
            for c in 0..f.contacts.len() {
                let cap = if cpu_moving[c] {
                    f.contacts[c].dynamic_cap
                } else {
                    f.contacts[c].static_cap
                } as f64;
                let lambda = [impulses[2 * c], impulses[2 * c + 1]];
                let slip = [2 * c, 2 * c + 1].map(|r| {
                    f.free[r] as f64
                        + (0..n)
                            .map(|j| f.gram[r * n + j] as f64 * impulses[j])
                            .sum::<f64>()
                });
                let speed = slip[0].hypot(slip[1]);
                let length = lambda[0].hypot(lambda[1]);
                let error = if cap == 0.0 {
                    0.0
                } else if length == 0.0 {
                    speed
                } else {
                    let direction = [lambda[0] / length, lambda[1] / length];
                    let pressure = (-(slip[0] * direction[0] + slip[1] * direction[1])).max(0.0);
                    (slip[0] + pressure * direction[0])
                        .hypot(slip[1] + pressure * direction[1])
                        .max(speed * (1.0 - length / cap).abs())
                };
                assert!(
                    error <= 1e-7,
                    "host KKT island={island} contact={c}: {error:e}"
                );
            }
        }
    }
    eprintln!("coupled disk batch: peak iterations={peak_iterations}");
}
