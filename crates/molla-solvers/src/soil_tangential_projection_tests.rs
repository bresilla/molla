use super::*;

fn buffer(engine: &Engine, values: &[f32]) -> wgpu::Buffer {
    let buffer = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("coupled friction fixture"),
            size: (values.len().max(4) * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    if !values.is_empty() {
        engine
            .queue()
            .unwrap()
            .write_buffer(&buffer, 0, bytemuck::cast_slice(values));
    }
    buffer
}

fn run(
    engine: &Engine,
    solver: &ExperimentalSoilTangentialProjectionGpu,
    n: usize,
    c: usize,
    reset: bool,
) -> (Vec<f32>, Vec<[f32; 4]>, [u32; 4]) {
    let device = engine.device().unwrap();
    let size = 4 * n as u64 + 16 * c.max(1) as u64 + 16;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("coupled friction fixture readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    if reset {
        encoder.clear_buffer(&solver.status, 0, None);
    }
    solver.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = transfers.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(dispatches.finish().encoded_dispatches, 1);
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&solver.rates, 0, &staging, 0, 4 * n as u64);
    encoder.copy_buffer_to_buffer(
        &solver.outputs,
        0,
        &staging,
        4 * n as u64,
        16 * c.max(1) as u64,
    );
    encoder.copy_buffer_to_buffer(&solver.status, 0, &staging, size - 16, 16);
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
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
    let mapped = staging.slice(..).get_mapped_range();
    let rates = bytemuck::cast_slice::<u8, f32>(&mapped[..4 * n]).to_vec();
    let contacts = bytemuck::cast_slice::<u8, [f32; 4]>(&mapped[4 * n..4 * n + 16 * c]).to_vec();
    let status = *bytemuck::from_bytes::<[u32; 4]>(&mapped[size as usize - 16..]);
    drop(mapped);
    staging.unmap();
    (rates, contacts, status)
}

fn inverse(mass: &[f32], n: usize) -> Vec<f64> {
    let mut matrix = mass.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        result[i * n + i] = 1.0;
    }
    for i in 0..n {
        let pivot = matrix[i * n + i];
        assert!(pivot > 0.0);
        for j in 0..n {
            matrix[i * n + j] /= pivot;
            result[i * n + j] /= pivot;
        }
        for k in 0..n {
            if k != i {
                let factor = matrix[k * n + i];
                for j in 0..n {
                    matrix[k * n + j] -= factor * matrix[i * n + j];
                    result[k * n + j] -= factor * result[i * n + j];
                }
            }
        }
    }
    result
}

fn oracle(
    mass: &[f32],
    free: &[f32],
    rows: &[[f32; 2]],
    contacts: &[SoilTangentialContact],
) -> (Vec<f64>, Vec<[f64; 2]>) {
    let n = free.len();
    let c = contacts.len();
    let inv = inverse(mass, n);
    let mut q = free.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let mut impulses = vec![[0.0; 2]; c];
    for sweep in 0..10000 {
        let mut change = 0.0f64;
        for (k, contact) in contacts.iter().enumerate() {
            if contact.mass == 0.0 || contact.max_impulse == 0.0 {
                continue;
            }
            let mut response = vec![[0.0; 2]; n];
            for i in 0..n {
                for j in 0..n {
                    for axis in 0..2 {
                        response[i][axis] += inv[i * n + j] * f64::from(rows[k * n + j][axis]);
                    }
                }
            }
            let mut w = [
                1.0 / f64::from(contact.mass),
                0.0,
                1.0 / f64::from(contact.mass),
            ];
            let mut v = contact.free_velocity.map(|v| -f64::from(v));
            for i in 0..n {
                let a = rows[k * n + i].map(f64::from);
                w[0] += a[0] * response[i][0];
                w[1] += a[0] * response[i][1];
                w[2] += a[1] * response[i][1];
                for axis in 0..2 {
                    v[axis] += a[axis]
                        * (q[i]
                            - response[i][0] * impulses[k][0]
                            - response[i][1] * impulses[k][1]);
                }
            }
            let (p, _) = crate::contact_friction::impulse(
                w,
                v,
                f64::from(contact.max_impulse),
                f64::from(contact.max_impulse),
            );
            let delta = [p[0] - impulses[k][0], p[1] - impulses[k][1]];
            change = change.max(delta[0].abs()).max(delta[1].abs());
            for i in 0..n {
                q[i] += response[i][0] * delta[0] + response[i][1] * delta[1];
            }
            impulses[k] = p;
        }
        if change < 1e-12 {
            return (q, impulses);
        }
        assert!(sweep < 9999, "oracle did not converge");
    }
    unreachable!()
}

fn check(
    engine: &Engine,
    mass: &[f32],
    free: &[f32],
    rows: &[[f32; 2]],
    contacts: &[SoilTangentialContact],
) {
    let n = free.len();
    let c = contacts.len();
    let buffers = [
        buffer(engine, mass),
        buffer(engine, free),
        buffer(engine, bytemuck::cast_slice(rows)),
        buffer(engine, bytemuck::cast_slice(contacts)),
    ];
    let budget = 4 * n as u64 + 32 * c.max(1) as u64 + 32;
    let solver = ExperimentalSoilTangentialProjectionGpu::new(
        engine,
        SoilTangentialInputs {
            mass: &buffers[0],
            free_rates: &buffers[1],
            rows: &buffers[2],
            contacts: &buffers[3],
        },
        SoilTangentialCapacity {
            dofs: n as u32,
            contacts: c as u32,
            max_iterations: 64,
            max_owned_bytes: budget,
        },
    )
    .unwrap();
    assert_eq!(solver.owned_buffer_bytes(), budget);
    let (q, output, status) = run(engine, &solver, n, c, true);
    assert_eq!(status[0], 0, "n={n}, c={c}, status={status:?}, q={q:?}");
    let (reference, impulses) = oracle(mass, free, rows, contacts);
    for i in 0..n {
        assert!(
            (f64::from(q[i]) - reference[i]).abs() <= 2e-5 * (1.0 + reference[i].abs()),
            "rate {i}: {} != {}",
            q[i],
            reference[i]
        );
    }
    let mut energy = 0.0;
    for i in 0..n {
        let mut momentum = 0.0;
        let mut scale = 1.0;
        for j in 0..n {
            momentum += f64::from(mass[i * n + j]) * (f64::from(q[j]) - f64::from(free[j]));
            scale += (f64::from(mass[i * n + j]) * f64::from(free[j])).abs();
            energy += 0.5
                * f64::from(mass[i * n + j])
                * (f64::from(q[i]) * f64::from(q[j]) - f64::from(free[i]) * f64::from(free[j]));
        }
        for k in 0..c {
            for axis in 0..2 {
                momentum -= f64::from(rows[k * n + i][axis]) * f64::from(output[k][axis]);
            }
        }
        assert!(
            momentum.abs() <= 2e-6 * scale,
            "momentum {i}: {momentum}, scale {scale}"
        );
    }
    for k in 0..c {
        let cap = f64::from(contacts[k].max_impulse);
        let m = f64::from(contacts[k].mass);
        let p = [f64::from(output[k][0]), f64::from(output[k][1])];
        assert!(p[0].hypot(p[1]) <= cap * (1.0 + 2e-6));
        for axis in 0..2 {
            assert!(
                (p[axis] - impulses[k][axis]).abs() <= 2e-5 * (1.0 + cap),
                "contact {k}: {p:?} != {:?}",
                impulses[k]
            );
            if m > 0.0 {
                let corrected = f64::from(contacts[k].free_velocity[axis]) - p[axis] / m;
                assert!(
                    (corrected - f64::from(output[k][2 + axis])).abs()
                        < 2e-6 * (1.0 + corrected.abs())
                );
                energy += 0.5
                    * m
                    * (corrected * corrected - f64::from(contacts[k].free_velocity[axis]).powi(2));
            }
        }
    }
    assert!(energy <= 1e-5, "energy increase {energy}");
    let repeated = run(engine, &solver, n, c, true);
    assert_eq!(q, repeated.0);
    assert_eq!(output, repeated.1);
    println!(
        "shared tangential n={n} c={c}: {} iterations, {} backtracks, energy change {energy}",
        status[1], status[2]
    );
}

#[test]
#[ignore = "requires GPU"]
fn coupled_friction_shared_response_reference_and_energy() {
    let engine = Engine::new_gpu().unwrap();
    let shared = [SoilTangentialContact {
        mass: 1.0,
        max_impulse: 4.0,
        free_velocity: [0.0; 2],
    }; 2];
    check(&engine, &[1.0], &[3.0], &[[1.0, 0.0]; 2], &shared);
    let (q, p) = oracle(&[1.0], &[3.0], &[[1.0, 0.0]; 2], &shared);
    assert!((q[0] - 1.0).abs() < 1e-10);
    assert!(p.iter().all(|p| (p[0] + 1.0).abs() < 1e-10));
    check(
        &engine,
        &[7.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 3.0],
        &[2.0, -1.0, 1.2],
        &[
            [1.0, 0.0],
            [0.0, 1.0],
            [0.8, 0.6],
            [1.0, 0.0],
            [0.0, 1.0],
            [-0.4, -0.7],
        ],
        &[
            SoilTangentialContact {
                mass: 2.0,
                max_impulse: 0.4,
                free_velocity: [-0.5, 0.75],
            },
            SoilTangentialContact {
                mass: 1.5,
                max_impulse: 10.0,
                free_velocity: [0.25, -0.5],
            },
        ],
    );
    for mass in [0.001, 1.0, 1000.0] {
        let contacts = [
            SoilTangentialContact {
                mass,
                max_impulse: 100.0,
                free_velocity: [-0.5, 0.75],
            },
            SoilTangentialContact {
                mass: 2.0,
                max_impulse: 0.4,
                free_velocity: [1.0, -1.0],
            },
        ];
        check(
            &engine,
            &[7.0, 0.5, 0.5, 3.0],
            &[2.0, -1.0],
            &[[1.0, 0.0], [0.0, 1.0], [0.5, 0.8], [-0.4, 1.0]],
            &contacts,
        );
    }
    let mut seed = 90210u32;
    let mut sample = || {
        seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
        (seed >> 8) as f32 / 16777216.0
    };
    for count in [1, 3, 65] {
        let rows = (0..count * 3)
            .map(|_| [sample() - 0.5, sample() - 0.5])
            .collect::<Vec<_>>();
        let contacts = (0..count)
            .map(|i| SoilTangentialContact {
                mass: 0.2 + sample(),
                max_impulse: if i % 7 == 0 { 0.0 } else { 0.05 + sample() },
                free_velocity: [sample() - 0.5, sample() - 0.5],
            })
            .collect::<Vec<_>>();
        check(
            &engine,
            &[4.0, 0.5, -0.25, 0.5, 3.0, 0.125, -0.25, 0.125, 2.0],
            &[2.0, -1.0, 0.5],
            &rows,
            &contacts,
        );
    }
    check(&engine, &[2.0], &[3.0], &[], &[]);
    let mut mass = vec![0.0; 32 * 32];
    for i in 0..32 {
        mass[i * 32 + i] = 1.0 + i as f32;
    }
    let rows = (0..32)
        .map(|i| {
            [
                if i == 0 { 1.0 } else { 0.0 },
                if i == 31 { 1.0 } else { 0.0 },
            ]
        })
        .collect::<Vec<_>>();
    check(
        &engine,
        &mass,
        &[1.0; 32],
        &rows,
        &[SoilTangentialContact {
            mass: 2.0,
            max_impulse: 0.3,
            free_velocity: [0.0; 2],
        }],
    );
}

#[test]
#[ignore = "requires GPU"]
fn coupled_friction_faults_reset_and_preflight() {
    let engine = Engine::new_gpu().unwrap();
    let buffers = [
        buffer(&engine, &[1.0]),
        buffer(&engine, &[3.0]),
        buffer(&engine, &[1.0, 0.0]),
        buffer(&engine, &[1.0, 0.2, 0.0, 0.0]),
    ];
    let new = |iterations, budget| {
        ExperimentalSoilTangentialProjectionGpu::new(
            &engine,
            SoilTangentialInputs {
                mass: &buffers[0],
                free_rates: &buffers[1],
                rows: &buffers[2],
                contacts: &buffers[3],
            },
            SoilTangentialCapacity {
                dofs: 1,
                contacts: 1,
                max_iterations: iterations,
                max_owned_bytes: budget,
            },
        )
    };
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(new(64, 67).is_err());
    assert!(new(0, 68).is_err());
    assert!(new(65, 68).is_err());
    assert_eq!(scope.finish().observations, 0);
    let short = new(1, 68).unwrap();
    assert_eq!(run(&engine, &short, 1, 1, true).2[0], 16);
    let solver = new(64, 68).unwrap();
    for (index, values, expected) in [
        (0, vec![-1.0], 2u32),
        (1, vec![f32::NAN], 1),
        (2, vec![f32::INFINITY, 0.0], 1),
        (3, vec![1.0, -0.2, 0.0, 0.0], 1),
    ] {
        engine
            .queue()
            .unwrap()
            .write_buffer(&buffers[index], 0, bytemuck::cast_slice(&values));
        assert_ne!(run(&engine, &solver, 1, 1, true).2[0] & expected, 0);
        let originals: [Vec<f32>; 4] = [
            vec![1.0],
            vec![3.0],
            vec![1.0, 0.0],
            vec![1.0, 0.2, 0.0, 0.0],
        ];
        engine.queue().unwrap().write_buffer(
            &buffers[index],
            0,
            bytemuck::cast_slice(&originals[index]),
        );
        assert_ne!(run(&engine, &solver, 1, 1, false).2[0], 0);
        assert_eq!(run(&engine, &solver, 1, 1, true).2[0], 0);
    }
}

#[test]
fn tangential_contact_layout() {
    assert_eq!(std::mem::size_of::<SoilTangentialContact>(), 16);
}
