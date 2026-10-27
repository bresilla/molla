use super::*;
use molla_math::Vec3;

#[path = "soil_boundary_projection_tests.rs"]
mod boundary_tests;

#[path = "soil_projection_replay_tests.rs"]
pub(crate) mod replay;

#[path = "soil_projection_scratch_tests.rs"]
mod scratch;

fn buffer(engine: &Engine, bytes: &[u8]) -> wgpu::Buffer {
    let result = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("soil projection fixture"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    engine.queue().unwrap().write_buffer(&result, 0, bytes);
    result
}

fn read(engine: &Engine, source: &wgpu::Buffer) -> Vec<f32> {
    let device = engine.device().unwrap();
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("soil projection observation"),
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &stage, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let slice = stage.slice(..);
    let (tx, rx) = std::sync::mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .unwrap()
        .unwrap();
    let result = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
    stage.unmap();
    result
}

fn submit(engine: &Engine, projection: &SoilContactProjectionGpu, reset: bool) {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if reset {
        encoder.clear_buffer(projection.status_buffer(), 0, None);
    }
    projection.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!(transfers.readback_bytes, 0);
    assert_eq!(transfers.upload_bytes, 0);
}

fn solve_cpu(mut a: Vec<f64>, mut b: Vec<f64>) -> Vec<f64> {
    let n = b.len();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))
            .unwrap();
        for j in 0..n {
            a.swap(k * n + j, pivot * n + j);
        }
        b.swap(k, pivot);
        assert!(a[k * n + k].abs() > 1e-12);
        for i in k + 1..n {
            let f = a[i * n + k] / a[k * n + k];
            for j in k..n {
                a[i * n + j] -= f * a[k * n + j];
            }
            b[i] -= f * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|j| a[i * n + j] * x[j]).sum::<f64>()) / a[i * n + i];
    }
    x
}

fn point_columns(j: &[f64], n: usize, arm: Vec3) -> Vec<Vec3> {
    (0..n)
        .map(|i| {
            Vec3::new(j[i], j[n + i], j[2 * n + i])
                + Vec3::new(j[3 * n + i], j[4 * n + i], j[5 * n + i]).cross(arm)
        })
        .collect()
}

fn oracle(
    mass: &[f64],
    j: &[f64],
    free: &[f64],
    nodes: &[SoilProjectionNode],
    spans: &[[u32; 2]],
) -> Vec<f64> {
    let n = free.len();
    let mut a = mass.to_vec();
    let mut rhs: Vec<_> = (0..n)
        .map(|i| (0..n).map(|k| mass[i * n + k] * free[k]).sum::<f64>())
        .collect();
    for (body, span) in spans.iter().enumerate() {
        for node in &nodes[span[0] as usize..(span[0] + span[1]) as usize] {
            let p = Vec3::new(
                node.momentum[0] as f64,
                node.momentum[1] as f64,
                node.momentum[2] as f64,
            );
            let arm = Vec3::new(
                node.arm_mass[0] as f64,
                node.arm_mass[1] as f64,
                node.arm_mass[2] as f64,
            );
            let cols = point_columns(&j[body * 6 * n..(body + 1) * 6 * n], n, arm);
            for i in 0..n {
                rhs[i] += cols[i].dot(p);
                for k in 0..n {
                    a[i * n + k] += f64::from(node.arm_mass[3]) * cols[i].dot(cols[k]);
                }
            }
        }
    }
    solve_cpu(a, rhs)
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn finite_mass_projection_is_passive_and_rejects_invalid_inputs() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let node = buffer(
        &engine,
        bytemuck::cast_slice(&[SoilProjectionNode {
            arm_mass: [0.0, 0.0, 0.0, 1.0],
            momentum: [0.0; 4],
        }]),
    );
    let spans = buffer(&engine, bytemuck::cast_slice(&[[0u32, 1]]));
    let mass = buffer(&engine, bytemuck::cast_slice(&[1.0f32]));
    let j = buffer(
        &engine,
        bytemuck::cast_slice(&[1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0]),
    );
    let free = buffer(&engine, bytemuck::cast_slice(&[1.0f32]));
    let input = || SoilProjectionInputs {
        nodes: &node,
        spans: &spans,
        mass: &mass,
        jacobians: &j,
        free_rates: &free,
    };
    let capacity = SoilProjectionCapacity {
        dofs: 1,
        bodies: 1,
        nodes: 1,
        max_owned_bytes: 192,
    };
    assert!(
        SoilContactProjectionGpu::new(
            &engine,
            input(),
            SoilProjectionCapacity {
                max_owned_bytes: 191,
                ..capacity
            }
        )
        .is_err()
    );
    let projection = SoilContactProjectionGpu::new(&engine, input(), capacity).unwrap();
    assert_eq!(projection.owned_buffer_bytes(), 192);
    for ratio in [0.0f32, 0.0625, 0.5, 1.0, 2.0, 16.0, 256.0] {
        engine
            .queue()
            .unwrap()
            .write_buffer(&node, 12, bytemuck::bytes_of(&ratio));
        submit(&engine, &projection, true);
        assert_eq!(read(&engine, projection.status_buffer())[0].to_bits(), 0);
        let v = read(&engine, projection.rates_buffer())[0];
        assert!((v - 1.0 / (1.0 + ratio)).abs() < 2e-6);
        let kinetic = 0.5 * (1.0 + f64::from(ratio)) * f64::from(v).powi(2);
        assert!(kinetic <= 0.5 + 1e-6);
        let impulse = read(&engine, projection.impulses_buffer())[0];
        assert!((impulse - (v - 1.0)).abs() < 2e-6);
        if ratio > 1.0 {
            assert!(0.5 * (1.0 - ratio).powi(2) + 0.5 * ratio > 0.5);
        }
    }
    for (target, offset, value, bit) in [
        (&node, 12, -1.0, SOIL_PROJECTION_INPUT),
        (&mass, 0, -1.0, SOIL_PROJECTION_MASS),
        (&free, 0, f32::NAN, SOIL_PROJECTION_NONFINITE),
    ] {
        engine
            .queue()
            .unwrap()
            .write_buffer(&node, 12, bytemuck::bytes_of(&1.0f32));
        engine
            .queue()
            .unwrap()
            .write_buffer(&mass, 0, bytemuck::bytes_of(&1.0f32));
        engine
            .queue()
            .unwrap()
            .write_buffer(&free, 0, bytemuck::bytes_of(&1.0f32));
        let before = read(&engine, projection.rates_buffer());
        engine
            .queue()
            .unwrap()
            .write_buffer(target, offset, bytemuck::bytes_of(&value));
        if bit == SOIL_PROJECTION_MASS {
            engine
                .queue()
                .unwrap()
                .write_buffer(&node, 12, bytemuck::bytes_of(&256.0f32));
        }
        submit(&engine, &projection, true);
        assert_ne!(
            read(&engine, projection.status_buffer())[0].to_bits() & bit,
            0
        );
        assert_eq!(read(&engine, projection.rates_buffer()), before);
        for (target, offset) in [(&node, 12), (&mass, 0), (&free, 0)] {
            engine
                .queue()
                .unwrap()
                .write_buffer(target, offset, bytemuck::bytes_of(&1.0f32));
        }
        submit(&engine, &projection, false);
        assert_ne!(
            read(&engine, projection.status_buffer())[0].to_bits() & bit,
            0
        );
        assert_eq!(read(&engine, projection.rates_buffer()), before);
    }
    engine
        .queue()
        .unwrap()
        .write_buffer(&spans, 0, bytemuck::cast_slice(&[u32::MAX, 2u32]));
    submit(&engine, &projection, true);
    assert_ne!(
        read(&engine, projection.status_buffer())[0].to_bits() & SOIL_PROJECTION_INPUT,
        0
    );
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn tractor_projection_consumes_gpu_mass_and_com_jacobians() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut scene, handles) = crate::soil_tractor::build_soil_tractor_scene().unwrap();
    let n = scene.model().joint_dof_total;
    assert_eq!(n, 10);
    let bodies: Vec<_> = handles.wheels.into_iter().chain([handles.tine]).collect();
    let cpu_mass = crate::crba::mass_matrix(scene.model(), scene.state(), 0).unwrap();
    let cpu_j: Vec<_> = bodies
        .iter()
        .flat_map(|&body| {
            let index = scene.body_index(body).unwrap().index();
            crate::jacobian::jacobian(
                scene.model(),
                scene.state(),
                0,
                index,
                scene.model().body_com.host().unwrap()[index],
            )
            .unwrap()
        })
        .collect();
    let free: Vec<_> = (0..n).map(|i| (i as f64 - 4.0) * 0.3).collect();
    let spans: Vec<_> = (0..5).map(|i| [i * 73, 73]).collect();
    let nodes: Vec<_> = (0..365)
        .map(|i| {
            let m = 0.25 + (i % 7) as f32 * 0.125;
            SoilProjectionNode {
                arm_mass: [
                    (i % 3) as f32 * 0.1 - 0.1,
                    (i % 5) as f32 * 0.07 - 0.3,
                    (i % 11) as f32 * 0.02 - 0.1,
                    m,
                ],
                momentum: [
                    m * ((i % 13) as f32 * 0.2 - 0.5),
                    m * ((i % 7) as f32 * 0.1),
                    m * ((i % 17) as f32 * 0.01 - 0.03),
                    0.0,
                ],
            }
        })
        .collect();
    let expected = oracle(&cpu_mass, &cpu_j, &free, &nodes, &spans);
    scene.upload_model_to_device(device, queue).unwrap();
    let mut state = scene.state().clone_host().unwrap();
    state.upload_to_device(device, queue).unwrap();
    let mass = buffer(&engine, bytemuck::cast_slice(&vec![0.0f32; n * n]));
    let jacobians = buffer(&engine, bytemuck::cast_slice(&vec![0.0f32; 5 * 6 * n]));
    let node_buffer = buffer(&engine, bytemuck::cast_slice(&nodes));
    let span_buffer = buffer(&engine, bytemuck::cast_slice(&spans));
    let free_buffer = buffer(
        &engine,
        bytemuck::cast_slice(&free.iter().map(|&v| v as f32).collect::<Vec<_>>()),
    );
    let crba = molla_kernels::MassMatrixKernel::new(&engine, scene.model()).unwrap();
    let mass_dispatch = crba
        .prepare_on_device(device, queue, scene.model(), &state, &mass)
        .unwrap();
    let jac = molla_kernels::JacobianKernel::new(&engine, scene.model()).unwrap();
    let requests: Vec<_> = bodies
        .iter()
        .enumerate()
        .map(|(slot, &body)| {
            let index = scene.body_index(body).unwrap().index();
            let com = scene.model().body_com.host().unwrap()[index];
            molla_kernels::JacobianRequest {
                articulation: 0,
                target_body: index as u32,
                output_offset: (slot * 6 * n) as u32,
                body_q_offset: 0,
                point_local: [com.x as f32, com.y as f32, com.z as f32, 0.0],
            }
        })
        .collect();
    let jac_dispatch = jac
        .prepare_with_body_q_on_device(
            device,
            queue,
            scene.model(),
            state.body_q.device_buffer().unwrap(),
            &requests,
            &jacobians,
        )
        .unwrap()
        .unwrap();
    let projection = SoilContactProjectionGpu::new(
        &engine,
        SoilProjectionInputs {
            nodes: &node_buffer,
            spans: &span_buffer,
            mass: &mass,
            jacobians: &jacobians,
            free_rates: &free_buffer,
        },
        SoilProjectionCapacity {
            dofs: n as u32,
            bodies: 5,
            nodes: 365,
            max_owned_bytes: 4096,
        },
    )
    .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    mass_dispatch.encode(&mut encoder);
    jac_dispatch.encode(&mut encoder);
    projection.encode(&mut encoder);
    queue.submit([encoder.finish()]);
    assert_eq!(read(&engine, projection.status_buffer())[0].to_bits(), 0);
    let actual = read(&engine, projection.rates_buffer());
    for (i, (&a, &b)) in actual.iter().zip(&expected).enumerate() {
        assert!((f64::from(a) - b).abs() < 2e-5, "dof {i}: GPU={a} CPU={b}");
    }
    let impulses = read(&engine, projection.impulses_buffer());
    let motion = read(&engine, projection.motion_buffer());
    let actual64: Vec<_> = actual.iter().map(|&x| f64::from(x)).collect();
    let quadratic = |v: &[f64]| {
        (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| v[i] * cpu_mass[i * n + j] * v[j])
                    .sum::<f64>()
            })
            .sum::<f64>()
            * 0.5
    };
    let delta: Vec<_> = actual64.iter().zip(&free).map(|(a, b)| a - b).collect();
    let mut before = quadratic(&free);
    let mut after = quadratic(&actual64);
    let mut loss = quadratic(&delta);
    for b in 0..5 {
        let v = Vec3::new(
            motion[b * 8] as f64,
            motion[b * 8 + 1] as f64,
            motion[b * 8 + 2] as f64,
        );
        let w = Vec3::new(
            motion[b * 8 + 4] as f64,
            motion[b * 8 + 5] as f64,
            motion[b * 8 + 6] as f64,
        );
        for r in 0..6 {
            let expected_motion = (0..n)
                .map(|i| cpu_j[(b * 6 + r) * n + i] * actual64[i])
                .sum::<f64>();
            assert!(
                (expected_motion - motion[b * 8 + r + usize::from(r >= 3)] as f64).abs() < 2e-5
            );
        }
        for node in &nodes[b * 73..(b + 1) * 73] {
            let arm = Vec3::new(
                node.arm_mass[0] as f64,
                node.arm_mass[1] as f64,
                node.arm_mass[2] as f64,
            );
            let mass = f64::from(node.arm_mass[3]);
            let p = Vec3::new(
                node.momentum[0] as f64,
                node.momentum[1] as f64,
                node.momentum[2] as f64,
            );
            let velocity = v + w.cross(arm);
            before += 0.5 * p.length_squared() / mass;
            after += 0.5 * mass * velocity.length_squared();
            loss += 0.5 * mass * (velocity - p / mass).length_squared();
        }
    }
    assert!(after <= before + 2e-5 * before.max(1.0));
    assert!((before - after - loss).abs() <= 2e-5 * before.max(1.0));
    println!("combined kinetic energy: {before} -> {after}; projection dissipation {loss}");
    for i in 0..n {
        let machine = (0..n)
            .map(|k| cpu_mass[i * n + k] * (f64::from(actual[k]) - free[k]))
            .sum::<f64>();
        let contact = (0..5)
            .flat_map(|b| (0..6).map(move |r| (b, r)))
            .map(|(b, r)| {
                cpu_j[(b * 6 + r) * n + i] * f64::from(impulses[b * 8 + r + usize::from(r >= 3)])
            })
            .sum::<f64>();
        assert!(
            (machine - contact).abs() < 2e-5 * machine.abs().max(contact.abs()).max(1.0),
            "generalized momentum {i}: {machine} vs {contact}"
        );
    }
    println!(
        "ten-DOF tractor projection matches independent f64 node assembly; GPU CRBA/Jacobian inputs, 365 nodes, five bodies"
    );
    queue.write_buffer(&span_buffer, 8, bytemuck::cast_slice(&[0u32, 73]));
    submit(&engine, &projection, true);
    assert_ne!(
        read(&engine, projection.status_buffer())[0].to_bits() & SOIL_PROJECTION_INPUT,
        0
    );
    assert_eq!(read(&engine, projection.rates_buffer()), actual);
    queue.write_buffer(&span_buffer, 0, bytemuck::cast_slice(&spans));
    queue.write_buffer(&node_buffer, 0, &vec![0u8; nodes.len() * 32]);
    submit(&engine, &projection, true);
    assert_eq!(read(&engine, projection.status_buffer())[0].to_bits(), 0);
    assert_eq!(
        read(&engine, projection.rates_buffer()),
        free.iter().map(|&v| v as f32).collect::<Vec<_>>()
    );
    assert!(
        read(&engine, projection.impulses_buffer())
            .iter()
            .all(|&v| v == 0.0)
    );

    use crate::{FeatherstoneSolver, Solver};
    scene.upload_controls_to_device(device, queue).unwrap();
    let mut output = scene.state().clone_host().unwrap();
    output.upload_to_device(device, queue).unwrap();
    let machine_start = std::time::Instant::now();
    eprintln!("tractor publication: creating Featherstone pipelines");
    let mut solver = FeatherstoneSolver::new_gpu(&engine, scene.model()).unwrap();
    eprintln!(
        "tractor publication: pipelines ready after {:?}; free prediction",
        machine_start.elapsed()
    );
    let dt = 0.0001;
    solver
        .step(
            scene.model(),
            &state,
            scene.control(),
            &molla_sim::Contacts::with_capacity(0),
            dt,
            &mut output,
        )
        .unwrap();
    let predicted = read(&engine, output.joint_qd.device_buffer_current().unwrap());
    let initial_delta = read(&engine, solver.gpu_velocity_delta_buffer().unwrap());
    eprintln!(
        "tractor publication: free prediction observed after {:?}",
        machine_start.elapsed()
    );
    queue.write_buffer(&node_buffer, 0, bytemuck::cast_slice(&nodes));
    let live = SoilContactProjectionGpu::new(
        &engine,
        SoilProjectionInputs {
            nodes: &node_buffer,
            spans: &span_buffer,
            mass: &mass,
            jacobians: &jacobians,
            free_rates: output.joint_qd.device_buffer_current().unwrap(),
        },
        capacity_for_tractor(n),
    )
    .unwrap();
    let publication = live
        .prepare_publication(
            &engine,
            SoilRatePublicationInputs {
                rates: output.joint_qd.device_buffer_current().unwrap(),
                velocity_delta: solver.gpu_velocity_delta_buffer().unwrap(),
                machine_status: solver.gpu_status_buffer().unwrap(),
                max_owned_bytes: 16,
            },
        )
        .unwrap();
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    live.encode(&mut encoder);
    publication.encode(&mut encoder);
    queue.submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    assert_eq!(transfers.finish().readback_bytes, 0);
    let corrected = read(&engine, live.rates_buffer());
    eprintln!(
        "tractor publication: projected rates observed after {:?}",
        machine_start.elapsed()
    );
    assert_eq!(
        read(&engine, output.joint_qd.device_buffer_current().unwrap()),
        corrected
    );
    let corrected_delta = read(&engine, solver.gpu_velocity_delta_buffer().unwrap());
    for i in 0..n {
        assert!(
            (corrected_delta[i] - (initial_delta[i] + corrected[i] - predicted[i])).abs() < 2e-6
        );
    }
    let reference = oracle(
        &cpu_mass,
        &cpu_j,
        &predicted.iter().map(|&v| f64::from(v)).collect::<Vec<_>>(),
        &nodes,
        &spans,
    );
    for (actual, expected) in corrected.iter().zip(reference) {
        assert!((f64::from(*actual) - expected).abs() < 2e-5);
    }
    solver
        .drift_gpu(scene.model(), &state, scene.control(), dt, &mut output)
        .unwrap();
    eprintln!(
        "tractor publication: corrected drift submitted after {:?}",
        machine_start.elapsed()
    );
    assert_eq!(
        read(&engine, solver.gpu_status_buffer().unwrap())[0].to_bits(),
        0
    );
    assert_eq!(
        read(&engine, output.joint_qd.device_buffer_current().unwrap()),
        corrected
    );
    assert_eq!(
        read(&engine, solver.gpu_velocity_delta_buffer().unwrap()),
        corrected_delta
    );
    let q = read(&engine, output.joint_q.device_buffer_current().unwrap());
    let q0 = read(&engine, state.joint_q.device_buffer_current().unwrap());
    for i in 0..3 {
        assert!(
            (f64::from(q[i]) - f64::from(q0[i]) - dt * f64::from(corrected[i])).abs() < 1e-7,
            "root drift {i}"
        );
    }
    for i in 6..10 {
        assert!(
            (f64::from(q[i + 1]) - f64::from(q0[i + 1]) - dt * f64::from(corrected[i])).abs()
                < 1e-7,
            "wheel drift {i}"
        );
    }
    println!(
        "actual tractor free prediction -> projection -> rate/delta publication -> compensated drift passes; no second reaction force"
    );
}

fn capacity_for_tractor(n: usize) -> SoilProjectionCapacity {
    SoilProjectionCapacity {
        dofs: n as u32,
        bodies: 5,
        nodes: 365,
        max_owned_bytes: 4096,
    }
}

#[test]
#[ignore = "requires a real GPU adapter"]
fn rate_publication_is_all_or_nothing_and_preserves_stop_status() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let data = |words: &[f32]| buffer(&engine, bytemuck::cast_slice(words));
    let nodes = data(&[0.0; 8]);
    let spans = data(&[0.0; 2]);
    let mass = data(&[0.0; 32 * 32]);
    let jac = data(&[0.0; 6 * 32]);
    let rates = data(&[1.0; 33]);
    let delta = data(&[-0.0; 33]);
    let status = data(&[0.0]);
    let projection = SoilContactProjectionGpu::new(
        &engine,
        SoilProjectionInputs {
            nodes: &nodes,
            spans: &spans,
            mass: &mass,
            jacobians: &jac,
            free_rates: &rates,
        },
        SoilProjectionCapacity {
            dofs: 32,
            bodies: 1,
            nodes: 1,
            max_owned_bytes: 8192,
        },
    )
    .unwrap();
    let input = || SoilRatePublicationInputs {
        rates: &rates,
        velocity_delta: &delta,
        machine_status: &status,
        max_owned_bytes: 16,
    };
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    for bad in [
        SoilRatePublicationInputs {
            max_owned_bytes: 15,
            ..input()
        },
        SoilRatePublicationInputs {
            velocity_delta: &rates,
            ..input()
        },
        SoilRatePublicationInputs {
            rates: projection.rates_buffer(),
            ..input()
        },
        SoilRatePublicationInputs {
            machine_status: &spans,
            velocity_delta: &spans,
            ..input()
        },
        SoilRatePublicationInputs {
            rates: &status,
            ..input()
        },
    ] {
        assert!(projection.prepare_publication(&engine, bad).is_err());
    }
    assert_eq!(allocations.finish().observations, 0);
    let publisher = projection.prepare_publication(&engine, input()).unwrap();
    assert_eq!(publisher.owned_buffer_bytes(), 16);
    let publish = || {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        publisher.encode(&mut encoder);
        queue.submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let transfers = transfers.finish();
        assert_eq!(transfers.readback_bytes, 0);
        assert_eq!(transfers.upload_bytes, 0);
    };
    let bits = |b| {
        read(&engine, b)
            .iter()
            .map(|v| v.to_bits())
            .collect::<Vec<_>>()
    };
    queue.write_buffer(
        projection.rates_buffer(),
        0,
        bytemuck::cast_slice(&[1.0f32; 32]),
    );
    publish();
    assert_eq!(bits(&delta), vec![(-0.0f32).to_bits(); 33]);
    let expected: Vec<_> = (0..32).map(|i| i as f32 * 0.125 - 2.0).collect();
    queue.write_buffer(
        projection.rates_buffer(),
        0,
        bytemuck::cast_slice(&expected),
    );
    publish();
    assert_eq!(&read(&engine, &rates)[..32], &expected);
    assert_eq!(read(&engine, &rates)[32], 1.0);
    let published_delta = read(&engine, &delta);
    for i in 0..32 {
        assert_eq!(published_delta[i], expected[i] - 1.0);
    }
    assert_eq!(published_delta[32].to_bits(), (-0.0f32).to_bits());
    let before_rates = bits(&rates);
    let before_delta = bits(&delta);
    for (buffer, offset, value, expected_status) in [
        (projection.status_buffer(), 0, 16u32, 32u32),
        (&status, 0, 1u32 << 30, 1u32 << 30),
        (projection.rates_buffer(), 31 * 4, f32::NAN.to_bits(), 32u32),
        (&delta, 31 * 4, f32::INFINITY.to_bits(), 32u32),
        (&rates, 31 * 4, f32::NAN.to_bits(), 32u32),
    ] {
        queue.write_buffer(&status, 0, bytemuck::bytes_of(&0u32));
        queue.write_buffer(projection.status_buffer(), 0, bytemuck::bytes_of(&0u32));
        queue.write_buffer(
            projection.rates_buffer(),
            0,
            bytemuck::cast_slice(&[3.0f32; 32]),
        );
        queue.write_buffer(&delta, 0, bytemuck::cast_slice(&before_delta));
        queue.write_buffer(&rates, 0, bytemuck::cast_slice(&before_rates));
        queue.write_buffer(buffer, offset, bytemuck::bytes_of(&value));
        let expected_delta = bits(&delta);
        let expected_rates = bits(&rates);
        publish();
        assert_eq!(bits(&status)[0], expected_status);
        assert_eq!(bits(&rates), expected_rates);
        assert_eq!(bits(&delta), expected_delta);
    }
    queue.write_buffer(&status, 0, bytemuck::bytes_of(&0u32));
    queue.write_buffer(&rates, 31 * 4, bytemuck::bytes_of(&(-3e38f32)));
    queue.write_buffer(
        projection.rates_buffer(),
        31 * 4,
        bytemuck::bytes_of(&3e38f32),
    );
    let overflow_rates = bits(&rates);
    let overflow_delta = bits(&delta);
    publish();
    assert_eq!(bits(&status)[0], 32);
    assert_eq!(bits(&rates), overflow_rates);
    assert_eq!(bits(&delta), overflow_delta);
    queue.write_buffer(&status, 0, bytemuck::bytes_of(&(1u32 << 30)));
    queue.write_buffer(projection.status_buffer(), 0, bytemuck::bytes_of(&32u32));
    publish();
    assert_eq!(bits(&status)[0], (1u32 << 30) | 32);
    println!(
        "32-DOF all-or-nothing publication, signed-zero/tail preservation, upstream stop and local fault propagation pass"
    );
}
