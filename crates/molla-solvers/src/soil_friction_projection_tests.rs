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
    solver: &ExperimentalSoilFrictionProjectionGpu,
    n: usize,
    c: usize,
    reset: bool,
) -> (Vec<f32>, Vec<[f32; 4]>, Vec<[f32; 4]>, [u32; 4]) {
    let device = engine.device().unwrap();
    let size = 4 * n as u64 + 32 * c.max(1) as u64 + 16;
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
    encoder.copy_buffer_to_buffer(
        &solver.normals,
        0,
        &staging,
        4 * n as u64 + 16 * c.max(1) as u64,
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
    let normal_offset = 4 * n + 16 * c.max(1);
    let normals =
        bytemuck::cast_slice::<u8, [f32; 4]>(&mapped[normal_offset..normal_offset + 16 * c])
            .to_vec();
    let status = *bytemuck::from_bytes::<[u32; 4]>(&mapped[size as usize - 16..]);
    drop(mapped);
    staging.unmap();
    (rates, contacts, normals, status)
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

struct Fixture {
    mass: Vec<f32>,
    free: Vec<f32>,
    rows: Vec<[f32; 2]>,
    normal_rows: Vec<f32>,
    contacts: Vec<SoilFrictionContact>,
    normals: Vec<SoilFrictionNormal>,
}

fn oracle(f: &Fixture) -> (Vec<f64>, Vec<[f64; 2]>, Vec<f64>) {
    let n = f.free.len();
    let c = f.contacts.len();
    let inv = inverse(&f.mass, n);
    let mut q = f.free.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    let mut p = vec![[0.0; 2]; c];
    let mut lambda = vec![0.0; c];
    let mut responses = vec![vec![[0.0; 3]; n]; c];
    for k in 0..c {
        for i in 0..n {
            for j in 0..n {
                for a in 0..2 {
                    responses[k][i][a] += inv[i * n + j] * f64::from(f.rows[k * n + j][a]);
                }
                responses[k][i][2] += inv[i * n + j] * f64::from(f.normal_rows[k * n + j]);
            }
        }
    }
    for sweep in 0..20000 {
        let mut error = 0.0f64;
        for k in 0..c {
            let m = f64::from(f.contacts[k].mass);
            if m == 0.0 {
                continue;
            }
            let response = &responses[k];
            let mut w = 1.0 / m;
            let mut v = -f64::from(f.normals[k].free_velocity) - f64::from(f.normals[k].gap_speed);
            for i in 0..n {
                w += f64::from(f.normal_rows[k * n + i]) * response[i][2];
                v += f64::from(f.normal_rows[k * n + i]) * (q[i] + response[i][2] * lambda[k]);
            }
            let next = (v / w).max(0.0);
            let change = next - lambda[k];
            lambda[k] = next;
            error = error.max(change.abs());
            for i in 0..n {
                q[i] -= response[i][2] * change;
            }
            let mut w = [1.0 / m, 0.0, 1.0 / m];
            let mut v = f.contacts[k].free_velocity.map(|x| -f64::from(x));
            for i in 0..n {
                let a = f.rows[k * n + i].map(f64::from);
                w[0] += a[0] * response[i][0];
                w[1] += a[0] * response[i][1];
                w[2] += a[1] * response[i][1];
                for axis in 0..2 {
                    v[axis] +=
                        a[axis] * (q[i] - response[i][0] * p[k][0] - response[i][1] * p[k][1]);
                }
            }
            let cap = f64::from(f.contacts[k].friction) * lambda[k];
            let (next, _) = crate::contact_friction::impulse(w, v, cap, cap);
            let change = [next[0] - p[k][0], next[1] - p[k][1]];
            p[k] = next;
            error = error.max(change[0].abs()).max(change[1].abs());
            for i in 0..n {
                q[i] += response[i][0] * change[0] + response[i][1] * change[1];
            }
        }
        if error < 1e-12 {
            return (q, p, lambda);
        }
        assert!(sweep < 19999, "dual oracle did not converge");
    }
    unreachable!()
}

fn inputs(engine: &Engine, f: &Fixture) -> [wgpu::Buffer; 6] {
    [
        buffer(engine, &f.mass),
        buffer(engine, &f.free),
        buffer(engine, bytemuck::cast_slice(&f.rows)),
        buffer(engine, bytemuck::cast_slice(&f.contacts)),
        buffer(engine, &f.normal_rows),
        buffer(engine, bytemuck::cast_slice(&f.normals)),
    ]
}
fn solver(
    engine: &Engine,
    b: &[wgpu::Buffer; 6],
    n: usize,
    c: usize,
    iterations: u32,
    budget: u64,
) -> Result<ExperimentalSoilFrictionProjectionGpu> {
    ExperimentalSoilFrictionProjectionGpu::new(
        engine,
        SoilFrictionInputs {
            mass: &b[0],
            free_rates: &b[1],
            rows: &b[2],
            contacts: &b[3],
            normal_rows: &b[4],
            normals: &b[5],
        },
        SoilFrictionCapacity {
            dofs: n as u32,
            contacts: c as u32,
            max_iterations: iterations,
            max_owned_bytes: budget,
        },
    )
}
fn check(engine: &Engine, f: &Fixture) -> [u32; 4] {
    let n = f.free.len();
    let c = f.contacts.len();
    let b = inputs(engine, f);
    let budget = 4 * n as u64 + 48 * c.max(1) as u64 + 32;
    let solver = solver(engine, &b, n, c, 64, budget).unwrap();
    assert_eq!(solver.owned_buffer_bytes(), budget);
    let (q, p, normal, status) = run(engine, &solver, n, c, true);
    let control = full_assembly_control(engine, &b, n, c);
    let expected = run(engine, &control, n, c, true);
    assert_eq!(
        q.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.0.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    assert_eq!(
        bytemuck::cast_slice::<_, u32>(&p),
        bytemuck::cast_slice::<_, u32>(&expected.1)
    );
    assert_eq!(
        bytemuck::cast_slice::<_, u32>(&normal),
        bytemuck::cast_slice::<_, u32>(&expected.2)
    );
    assert_eq!(status, expected.3);
    assert_eq!(status[0], 0, "n={n} c={c} status={status:?}");
    let (reference, expected_p, expected_lambda) = oracle(f);
    let mut energy = 0.0;
    for i in 0..n {
        assert!(
            (f64::from(q[i]) - reference[i]).abs() <= 2e-5 * (1.0 + reference[i].abs()),
            "rate {i}: {} vs {}",
            q[i],
            reference[i]
        );
        let mut residual = 0.0;
        let mut scale = 1.0;
        for j in 0..n {
            residual += f64::from(f.mass[i * n + j]) * (f64::from(q[j]) - f64::from(f.free[j]));
            scale += (f64::from(f.mass[i * n + j]) * f64::from(f.free[j])).abs();
            energy += 0.5
                * f64::from(f.mass[i * n + j])
                * (f64::from(q[i]) * f64::from(q[j]) - f64::from(f.free[i]) * f64::from(f.free[j]));
        }
        for k in 0..c {
            residual += f64::from(f.normal_rows[k * n + i]) * f64::from(normal[k][0]);
            for axis in 0..2 {
                residual -= f64::from(f.rows[k * n + i][axis]) * f64::from(p[k][axis]);
            }
        }
        assert!(
            residual.abs() <= 2e-6 * scale,
            "published momentum {i}: {residual}, scale {scale}"
        );
    }
    for k in 0..c {
        let m = f64::from(f.contacts[k].mass);
        let lambda = f64::from(normal[k][0]);
        let cap = f64::from(f.contacts[k].friction) * lambda;
        assert!(lambda >= 0.0);
        assert!(
            (lambda - expected_lambda[k]).abs() <= 2e-5 * (1.0 + expected_lambda[k]),
            "normal {k}: {lambda} vs {}",
            expected_lambda[k]
        );
        let radius = f64::from(p[k][0]).hypot(f64::from(p[k][1]));
        assert!(radius <= cap + 2e-6 * (1.0 + cap));
        let machine_normal = (0..n)
            .map(|i| f64::from(f.normal_rows[k * n + i]) * f64::from(q[i]))
            .sum::<f64>();
        let separation =
            f64::from(normal[k][1]) + f64::from(f.normals[k].gap_speed) - machine_normal;
        if m > 0.0 {
            assert!(
                separation >= -2e-5 * (1.0 + machine_normal.abs()),
                "normal feasibility {k}: {separation}"
            );
            assert!((lambda * separation).abs() <= 2e-5 * (1.0 + lambda));
            let soil = f64::from(f.normals[k].free_velocity) + lambda / m;
            assert!((soil - f64::from(normal[k][1])).abs() <= 2e-6 * (1.0 + soil.abs()));
            energy += 0.5 * m * (soil * soil - f64::from(f.normals[k].free_velocity).powi(2));
        }
        let mut slip = [0.0; 2];
        for axis in 0..2 {
            assert!((f64::from(p[k][axis]) - expected_p[k][axis]).abs() <= 2e-5 * (1.0 + cap));
            let soil = if m > 0.0 {
                f64::from(f.contacts[k].free_velocity[axis]) - f64::from(p[k][axis]) / m
            } else {
                f64::from(f.contacts[k].free_velocity[axis])
            };
            assert!((soil - f64::from(p[k][axis + 2])).abs() <= 2e-6 * (1.0 + soil.abs()));
            slip[axis] = (0..n)
                .map(|i| f64::from(f.rows[k * n + i][axis]) * f64::from(q[i]))
                .sum::<f64>()
                - soil;
            energy +=
                0.5 * m * (soil * soil - f64::from(f.contacts[k].free_velocity[axis]).powi(2));
        }
        let speed = slip[0].hypot(slip[1]);
        if speed > 1e-5 && m > 0.0 {
            for axis in 0..2 {
                assert!(
                    (f64::from(p[k][axis]) + cap * slip[axis] / speed).abs() <= 2e-5 * (1.0 + cap),
                    "friction KKT {k}"
                );
            }
        }
    }
    assert!(energy <= 1e-5, "kinetic energy increased {energy}");
    let repeated = run(engine, &solver, n, c, true);
    assert_eq!((q, p, normal), (repeated.0, repeated.1, repeated.2));
    println!(
        "joint friction n={n}, c={c}: iterations={}, backtracks={}, pivot exchanges={}, energy={energy}",
        status[1], status[2], status[3]
    );
    status
}

fn full_assembly_control(
    engine: &Engine,
    b: &[wgpu::Buffer; 6],
    n: usize,
    c: usize,
) -> ExperimentalSoilFrictionProjectionGpu {
    let source = include_str!("../shaders/soil_friction_projection.wgsl");
    assert_eq!(source.matches("assemble(lane,false)").count(), 1);
    let source = source.replace("assemble(lane,false)", "assemble(lane,true)");
    assert_eq!(source.matches("return atomicLoad(&active_end);").count(), 1);
    let source = source.replace("return atomicLoad(&active_end);", "return params.contacts;");
    let source = source.replace(
        "if contacts[c].x==0.0 { continue; }",
        "if false { continue; }",
    );
    source_solver(engine, b, n, c, &source)
}

fn source_solver(
    engine: &Engine,
    b: &[wgpu::Buffer; 6],
    n: usize,
    c: usize,
    source: &str,
) -> ExperimentalSoilFrictionProjectionGpu {
    ExperimentalSoilFrictionProjectionGpu::with_source(
        engine,
        SoilFrictionInputs {
            mass: &b[0],
            free_rates: &b[1],
            rows: &b[2],
            contacts: &b[3],
            normal_rows: &b[4],
            normals: &b[5],
        },
        SoilFrictionCapacity {
            dofs: n as u32,
            contacts: c as u32,
            max_iterations: 64,
            max_owned_bytes: 4 * n as u64 + 48 * c.max(1) as u64 + 32,
        },
        source,
    )
    .unwrap()
}

#[test]
#[ignore = "requires GPU timestamps"]
fn residual_only_line_search_matches_full_assembly_at_capacity_width() {
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    println!("adapter={:?}", engine.adapter_info().unwrap());
    let n = 32;
    let c = 1024;
    let mut f = Fixture {
        mass: vec![0.0; n * n],
        free: vec![0.0; n],
        rows: vec![[0.0; 2]; n * c],
        normal_rows: vec![0.0; n * c],
        contacts: vec![
            SoilFrictionContact {
                mass: 0.1,
                friction: 0.3,
                free_velocity: [0.0; 2]
            };
            c
        ],
        normals: vec![SoilFrictionNormal::default(); c],
    };
    for i in 0..n {
        f.mass[i * n + i] = 10.0;
    }
    f.free[0] = 2.0;
    f.free[1] = 1.0;
    f.free[n - 1] = 1.0;
    for i in 0..c {
        f.rows[i * n] = [1.0, 0.0];
        f.rows[i * n + 1] = [0.0, 1.0];
        f.normal_rows[i * n + n - 1] = 1.0;
    }
    let b = inputs(&engine, &f);
    let candidate = solver(&engine, &b, n, c, 64, u64::MAX).unwrap();
    let control = full_assembly_control(&engine, &b, n, c);
    let expected = run(&engine, &control, n, c, true);
    let actual = run(&engine, &candidate, n, c, true);
    assert_eq!(actual.3[0], 0, "{:?}", actual.3);
    assert_eq!(actual.3, expected.3);
    assert_eq!(
        bytemuck::cast_slice::<_, u32>(&actual.0),
        bytemuck::cast_slice::<_, u32>(&expected.0)
    );
    assert_eq!(
        bytemuck::cast_slice::<_, u32>(&actual.1),
        bytemuck::cast_slice::<_, u32>(&expected.1)
    );
    assert_eq!(
        bytemuck::cast_slice::<_, u32>(&actual.2),
        bytemuck::cast_slice::<_, u32>(&expected.2)
    );
    let soil_mass = c as f64 * f64::from(f.contacts[0].mass);
    let normal_rate = 10.0 / (10.0 + soil_mass);
    let tangent_scale =
        1.0 - f64::from(f.contacts[0].friction) * soil_mass * normal_rate / (10.0 * 5.0_f64.sqrt());
    for (i, reference) in [
        (0, 2.0 * tangent_scale),
        (1, tangent_scale),
        (n - 1, normal_rate),
    ] {
        assert!(
            (f64::from(actual.0[i]) - reference).abs() < 2e-5,
            "rate {i}"
        );
    }
    paired_timing(&engine, &control, &candidate, n, c, "residual");
    let source = include_str!("../shaders/soil_friction_projection.wgsl")
        .replace("return atomicLoad(&active_end);", "return params.contacts;")
        .replace(
            "if contacts[c].x==0.0 { continue; }",
            "if false { continue; }",
        );
    let dense_control = source_solver(&engine, &b, n, c, &source);
    assert_eq!(run(&engine, &dense_control, n, c, true), actual);
    paired_timing(
        &engine,
        &dense_control,
        &candidate,
        n,
        c,
        "dense-sparse-overhead",
    );
    assert_eq!(run(&engine, &dense_control, n, c, false), actual);
    assert_eq!(run(&engine, &candidate, n, c, false), actual);
    assert_eq!(run(&engine, &control, n, c, false), expected);
}

fn paired_timing(
    engine: &Engine,
    control: &ExperimentalSoilFrictionProjectionGpu,
    candidate: &ExperimentalSoilFrictionProjectionGpu,
    n: usize,
    c: usize,
    label: &str,
) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    assert!(
        device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    );
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("friction residual assembly timing"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("friction query resolve"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("friction query readback"),
        size: 16,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut samples = [Vec::<f64>::new(), Vec::<f64>::new()];
    for pair in 0..14 {
        for index in if pair % 2 == 0 { [0, 1] } else { [1, 0] } {
            let program = [&control, &candidate][index];
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.clear_buffer(&program.status, 0, None);
            encoder.write_timestamp(&queries, 0);
            program.encode(&mut encoder);
            encoder.write_timestamp(&queries, 1);
            encoder.resolve_query_set(&queries, 0..2, &resolved, 0);
            encoder.copy_buffer_to_buffer(&resolved, 0, &staging, 0, 16);
            queue.submit([encoder.finish()]);
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
            let ticks: &[u64] = bytemuck::cast_slice(&mapped);
            let ms = (ticks[1].checked_sub(ticks[0]).unwrap()) as f64
                * f64::from(queue.get_timestamp_period())
                / 1e6;
            drop(mapped);
            staging.unmap();
            println!("friction {label} pair={pair} variant={index} gpu_ms={ms:.6}");
            if pair >= 2 {
                samples[index].push(ms);
            }
        }
    }
    for (index, values) in samples.iter_mut().enumerate() {
        values.sort_by(f64::total_cmp);
        println!(
            "friction {label} n={n} contacts={c} variant={index} median_gpu_ms={:.6}",
            (values[5] + values[6]) / 2.0
        );
    }
    println!(
        "Fixed-input component only; no MPM, machine integration, rendering or simulated-time advance."
    );
}

#[test]
#[ignore = "requires GPU timestamps"]
fn active_contact_prefix_preserves_padding_reuse_and_faults() {
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    println!("prefix adapter={:?}", engine.adapter_info().unwrap());
    let original = planar(0.3);
    let reference = oracle(&original);
    let n = original.free.len();
    for c in [1024, 16384] {
        let padding = SoilFrictionContact {
            mass: 0.0,
            friction: -0.0,
            free_velocity: [0.25, -0.75],
        };
        let mut f = Fixture {
            mass: original.mass.clone(),
            free: original.free.clone(),
            rows: vec![[0.5, -0.25]; n * c],
            normal_rows: vec![0.125; n * c],
            contacts: vec![padding; c],
            normals: vec![
                SoilFrictionNormal {
                    free_velocity: -0.375,
                    gap_speed: 0.5,
                    padding: [0.0; 2]
                };
                c
            ],
        };
        let b = inputs(&engine, &f);
        let candidate = solver(&engine, &b, n, c, 64, u64::MAX).unwrap();
        let source = include_str!("../shaders/soil_friction_projection.wgsl");
        assert_eq!(source.matches("return atomicLoad(&active_end);").count(), 1);
        let source = source.replace("return atomicLoad(&active_end);", "return params.contacts;");
        assert_eq!(
            source
                .matches("if contacts[c].x==0.0 { continue; }")
                .count(),
            1
        );
        let source = source.replace(
            "if contacts[c].x==0.0 { continue; }",
            "if false { continue; }",
        );
        let control = source_solver(&engine, &b, n, c, &source);
        for second in [Some(1), Some(63), Some(c - 1), Some(1), None] {
            f.contacts.fill(padding);
            if let Some(second) = second {
                for (original_index, index) in [0, second].into_iter().enumerate() {
                    f.contacts[index] = original.contacts[original_index];
                    f.normals[index] = original.normals[original_index];
                    f.rows[index * n..(index + 1) * n].copy_from_slice(
                        &original.rows[original_index * n..(original_index + 1) * n],
                    );
                    f.normal_rows[index * n..(index + 1) * n].copy_from_slice(
                        &original.normal_rows[original_index * n..(original_index + 1) * n],
                    );
                }
            }
            for (index, bytes) in [
                (2, bytemuck::cast_slice(&f.rows)),
                (3, bytemuck::cast_slice(&f.contacts)),
                (4, bytemuck::cast_slice(&f.normal_rows)),
                (5, bytemuck::cast_slice(&f.normals)),
            ] {
                engine.queue().unwrap().write_buffer(&b[index], 0, bytes);
            }
            let actual = run(&engine, &candidate, n, c, true);
            let expected = run(&engine, &control, n, c, true);
            assert_eq!(
                actual.3[0], 0,
                "c={c} second={second:?} status={:?}",
                actual.3
            );
            assert_eq!(actual.3, expected.3);
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&actual.0),
                bytemuck::cast_slice::<_, u32>(&expected.0)
            );
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&actual.1),
                bytemuck::cast_slice::<_, u32>(&expected.1)
            );
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&actual.2),
                bytemuck::cast_slice::<_, u32>(&expected.2)
            );
            for i in 0..n {
                let reference = if second.is_some() {
                    reference.0[i]
                } else {
                    f64::from(f.free[i])
                };
                assert!((f64::from(actual.0[i]) - reference).abs() < 2e-5);
            }
            for index in 0..c {
                if f.contacts[index].mass == 0.0 {
                    assert_eq!(
                        actual.1[index],
                        [0.0, 0.0, padding.free_velocity[0], padding.free_velocity[1]]
                    );
                    assert_eq!(actual.2[index][0], 0.0);
                    assert_eq!(actual.2[index][1], f.normals[index].free_velocity);
                    assert_eq!(actual.2[index][2], 0.0);
                    assert_eq!(actual.2[index][3], 0.0);
                }
            }
            println!(
                "prefix capacity={c} second={second:?}: exact outputs/status and f64 rates pass"
            );
            if c == 1024 && (second == Some(63) || second == Some(c - 1)) {
                let label = if second == Some(63) {
                    "prefix-64"
                } else {
                    "gapped-full-prefix"
                };
                paired_timing(&engine, &control, &candidate, n, c, label);
                assert_eq!(run(&engine, &candidate, n, c, false), actual);
                assert_eq!(run(&engine, &control, n, c, false), expected);
            }
        }
        for (index, offset, invalid, restored) in [
            (2, (c - 1) * n * 8, f32::NAN, f.rows[(c - 1) * n][0]),
            (3, (c - 1) * 16 + 4, -1.0, f.contacts[c - 1].friction),
            (
                4,
                (c - 1) * n * 4,
                f32::INFINITY,
                f.normal_rows[(c - 1) * n],
            ),
            (5, (c - 1) * 16 + 4, -1.0, f.normals[c - 1].gap_speed),
        ] {
            engine.queue().unwrap().write_buffer(
                &b[index],
                offset as u64,
                bytemuck::bytes_of(&invalid),
            );
            assert_eq!(
                run(&engine, &candidate, n, c, true).3[0] & FRICTION_INPUT,
                FRICTION_INPUT
            );
            engine.queue().unwrap().write_buffer(
                &b[index],
                offset as u64,
                bytemuck::bytes_of(&restored),
            );
            assert_eq!(
                run(&engine, &candidate, n, c, false).3[0] & FRICTION_INPUT,
                FRICTION_INPUT
            );
            assert_eq!(run(&engine, &candidate, n, c, true).3[0], 0);
        }
        println!("prefix capacity={c}: four invalid-tail/sticky/reset cases pass");
    }
}

fn planar(mu: f32) -> Fixture {
    Fixture {
        mass: vec![7.0, 0.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0, 3.0],
        free: vec![2.0, 1.0, 1.2],
        rows: vec![
            [1.0, 0.0],
            [0.0, 0.0],
            [0.8, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.4, 0.0],
        ],
        normal_rows: vec![0.0, 1.0, 0.6, 0.0, 1.0, -0.7],
        contacts: vec![
            SoilFrictionContact {
                mass: 2.0,
                friction: mu,
                free_velocity: [-0.5, 0.0],
            },
            SoilFrictionContact {
                mass: 1.5,
                friction: mu,
                free_velocity: [0.25, 0.0],
            },
        ],
        normals: vec![SoilFrictionNormal::default(); 2],
    }
}
#[test]
#[ignore = "requires GPU"]
fn joint_normal_friction_matches_independent_dual_and_physical_checks() {
    let engine = Engine::new_gpu().unwrap();
    let frictionless = oracle(&planar(0.0));
    let frictional = oracle(&planar(0.3));
    assert!(
        frictionless
            .2
            .iter()
            .zip(&frictional.2)
            .any(|(a, b)| (a - b).abs() > 0.01)
    );
    for mu in [0.0, 0.1, 0.3, 1.0] {
        check(&engine, &planar(mu));
    }
    let mut f = planar(0.3);
    f.free[1] = -4.0;
    check(&engine, &f);
    let mut f = planar(0.3);
    f.normals[0].gap_speed = 0.25;
    f.normals[1].gap_speed = 0.1;
    check(&engine, &f);
    let mut f = planar(0.3);
    f.contacts[0].mass = 1000.0;
    check(&engine, &f);
    let mut f = planar(0.3);
    f.contacts[0].mass = 0.0;
    check(&engine, &f);
    check(
        &engine,
        &Fixture {
            mass: vec![2.0],
            free: vec![1.0],
            rows: vec![],
            normal_rows: vec![],
            contacts: vec![],
            normals: vec![],
        },
    );
    let f = Fixture {
        mass: vec![8.0, 0.0, 0.0, 5.0],
        free: vec![2.0, 1.0],
        rows: vec![[1.0, 0.0], [0.0, 0.0]].repeat(65),
        normal_rows: vec![0.0, 1.0].repeat(65),
        contacts: vec![
            SoilFrictionContact {
                mass: 0.1,
                friction: 0.3,
                free_velocity: [0.0; 2]
            };
            65
        ],
        normals: vec![SoilFrictionNormal::default(); 65],
    };
    check(&engine, &f);
    for mu in [0.5, 10.0] {
        let f = Fixture {
            mass: vec![2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0],
            free: vec![3.0, 4.0, 1.0],
            rows: vec![[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
            normal_rows: vec![0.0, 0.0, 1.0],
            contacts: vec![SoilFrictionContact {
                mass: 1.0,
                friction: mu,
                free_velocity: [0.0; 2],
            }],
            normals: vec![SoilFrictionNormal::default()],
        };
        let (q, p, normal) = oracle(&f);
        assert!((normal[0] - 2.0 / 3.0).abs() < 1e-10);
        if mu == 10.0 {
            assert!((q[0] + p[0][0]).abs() < 1e-10);
            assert!((q[1] + p[0][1]).abs() < 1e-10);
        }
        check(&engine, &f);
    }
    let mut mass = vec![0.0; 32 * 32];
    for i in 0..32 {
        mass[i * 32 + i] = 1.0 + i as f32;
    }
    let rows = (0..32)
        .map(|i| {
            [
                if i == 0 { 1.0 } else { 0.0 },
                if i == 1 { 1.0 } else { 0.0 },
            ]
        })
        .collect();
    let mut normal_rows = vec![0.0; 32];
    normal_rows[31] = 1.0;
    check(
        &engine,
        &Fixture {
            mass,
            free: vec![1.0; 32],
            rows,
            normal_rows,
            contacts: vec![SoilFrictionContact {
                mass: 2.0,
                friction: 0.3,
                free_velocity: [0.0; 2],
            }],
            normals: vec![SoilFrictionNormal::default()],
        },
    );
}
#[test]
#[ignore = "requires GPU"]
fn joint_friction_faults_reset_budget_and_iteration_bound() {
    let engine = Engine::new_gpu().unwrap();
    let f = planar(0.3);
    let b = inputs(&engine, &f);
    let budget = 140;
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(solver(&engine, &b, 3, 2, 64, budget - 1).is_err());
    assert!(solver(&engine, &b, 3, 2, 0, budget).is_err());
    assert_eq!(scope.finish().observations, 0);
    let short = solver(&engine, &b, 3, 2, 1, budget).unwrap();
    assert_eq!(run(&engine, &short, 3, 2, true).3[0], FRICTION_ITERATIONS);
    let s = solver(&engine, &b, 3, 2, 64, budget).unwrap();
    for invalid in [
        [0.0, -0.1, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [f32::NAN, 0.0, 0.0, 0.0],
    ] {
        engine
            .queue()
            .unwrap()
            .write_buffer(&b[5], 0, bytemuck::cast_slice(&invalid));
        assert_ne!(run(&engine, &s, 3, 2, true).3[0] & FRICTION_INPUT, 0);
        engine
            .queue()
            .unwrap()
            .write_buffer(&b[5], 0, bytemuck::cast_slice(&[0.0f32; 4]));
        assert_ne!(run(&engine, &s, 3, 2, false).3[0], 0);
        assert_eq!(run(&engine, &s, 3, 2, true).3[0], 0);
    }
}
#[test]
fn friction_contact_layout() {
    assert_eq!(std::mem::size_of::<SoilFrictionContact>(), 16);
    assert_eq!(std::mem::size_of::<SoilFrictionNormal>(), 16);
}

#[test]
#[ignore = "requires GPU"]
fn joint_friction_solves_former_cap_cycle() {
    let engine = Engine::new_gpu().unwrap();
    let f = Fixture {
        mass: vec![1.0],
        free: vec![1.0],
        rows: vec![[10.0, 0.0]],
        normal_rows: vec![1.0],
        contacts: vec![SoilFrictionContact {
            mass: 1.0,
            friction: 1.0,
            free_velocity: [0.0; 2],
        }],
        normals: vec![SoilFrictionNormal::default()],
    };
    let q = 1.0f64 / 12.0;
    let lambda = q;
    let p = -q;
    assert!((q - 1.0 + lambda - 10.0 * p).abs() < 1e-12);
    assert_eq!(lambda, q);
    assert_eq!(p.abs(), lambda);
    assert!(10.0 * q + p > 0.0);
    let b = inputs(&engine, &f);
    let s = solver(&engine, &b, 1, 1, 64, 84).unwrap();
    let result = run(&engine, &s, 1, 1, true);
    assert_eq!(result.3[0], 0, "unexpected status {:?}", result.3);
    assert!((f64::from(result.0[0]) - q).abs() < 1e-6);
    assert!((f64::from(result.1[0][0]) - p).abs() < 1e-6);
    assert!((f64::from(result.2[0][0]) - lambda).abs() < 1e-6);
    let residual =
        f64::from(result.0[0]) - 1.0 + f64::from(result.2[0][0]) - 10.0 * f64::from(result.1[0][0]);
    assert!(residual.abs() < 2e-6);
    println!(
        "former cap-cycle solved: {:?}; analytic q=lambda=1/12, p=-1/12",
        result.3
    );
}

#[test]
#[ignore = "requires GPU"]
fn joint_friction_pivoted_direction_and_merit_backtracking() {
    let engine = Engine::new_gpu().unwrap();
    let f = Fixture {
        mass: vec![1.0, 0.0, 0.0, 1.0],
        free: vec![1.0, 10.0],
        rows: vec![[0.0, 0.0], [10.0, 0.0]],
        normal_rows: vec![1.0, 0.0],
        contacts: vec![SoilFrictionContact {
            mass: 1.0,
            friction: 10.0,
            free_velocity: [0.0; 2],
        }],
        normals: vec![SoilFrictionNormal::default()],
    };
    let status = check(&engine, &f);
    assert!(status[2] > 0, "fixture must exercise residual backtracking");
    assert!(status[3] > 0, "fixture must exercise pivot exchanges");
    let reference = oracle(&f);
    assert!((reference.0[0] - 0.5).abs() < 1e-10);
    assert!((reference.0[1] - 10.0 / 101.0).abs() < 1e-10);
}
