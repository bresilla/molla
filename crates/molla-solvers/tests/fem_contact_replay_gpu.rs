#![cfg(feature = "gpu")]

use molla_kernels::soft_rigid_contact::SoftRigidContactGpu;
use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveProfile, SoftRigidSolveSettings,
};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    struct WakeThread(std::thread::Thread);
    impl std::task::Wake for WakeThread {
        fn wake(self: std::sync::Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(std::sync::Arc::new(WakeThread(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}

fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, values: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("contact preparation fixture"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn check_static_schedule(inputs: &[Vec<u8>], bytes: &[u8]) {
    let contacts: &[SoftRigidContactGpu] = bytemuck::cast_slice(&inputs[3]);
    let words: &[u32] = bytemuck::cast_slice(bytes);
    let base = contacts.len() + 1 + contacts.len().div_ceil(64);
    let enabled: Vec<_> = contacts
        .iter()
        .enumerate()
        .filter(|(_, c)| matches!(c.ids[3], 1 | 3))
        .collect();
    let nodes = |c: &SoftRigidContactGpu| {
        [c.ids[0], c.support[0], c.support[1]]
            .into_iter()
            .take(c.support[2] as usize)
    };
    let mut levels = vec![0u32; inputs[0].len() / 16];
    let dynamic: Vec<_> = enabled
        .iter()
        .filter(|(_, c)| c.ids[1] != u32::MAX)
        .map(|(r, c)| {
            for n in nodes(c) {
                levels[n as usize] = u32::MAX;
            }
            *r as u32
        })
        .collect();
    let mut layers = vec![Vec::new(); 64];
    let mut supported = true;
    for &(r, c) in &enabled {
        if c.ids[1] != u32::MAX {
            continue;
        }
        let level = nodes(c).map(|n| levels[n as usize]).max().unwrap();
        if level >= 64 {
            supported = false;
            break;
        }
        layers[level as usize].push(r as u32);
        for n in nodes(c) {
            levels[n as usize] = level + 1;
        }
    }
    assert_eq!(words[base], u32::from(supported));
    if !supported {
        return;
    }
    let count = layers
        .iter()
        .rposition(|v| !v.is_empty())
        .map_or(0, |n| n + 1);
    assert_eq!(words[base + 1] as usize, dynamic.len());
    assert_eq!(words[base + 2] as usize, count);
    let dynamic_start = base + 132 + levels.len() + contacts.len();
    assert_eq!(
        &words[dynamic_start..dynamic_start + dynamic.len()],
        dynamic
    );
    let static_start = dynamic_start + contacts.len();
    let mut offset = 0usize;
    for (level, rows) in layers.iter().enumerate() {
        assert_eq!(words[base + 3 + level] as usize, offset);
        assert_eq!(
            &words[static_start + offset..static_start + offset + rows.len()],
            rows
        );
        offset += rows.len();
    }
    assert_eq!(words[base + 67] as usize, offset);
}

#[test]
#[ignore = "requires GPU timestamp queries through make test-gpu"]
fn separated_static_layers_cover_overlap_depth_and_wide_supports() {
    for mode in [
        "wide",
        "overlap",
        "depth64",
        "depth65",
        "static_only",
        "dynamic_only",
    ] {
        let point = SoftRigidContactGpu {
            normal_gap: [0.0, 1.0, 0.0, -0.00001],
            material: [0.6, 0.0, 0.0, 0.0],
            ids: [0, 0, 0, 1],
            support: [0, 0, 1, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        let mut contacts = Vec::new();
        if mode != "static_only" {
            contacts.push(point);
        }
        if mode != "dynamic_only" {
            let first = if mode == "overlap" { 0 } else { 1 };
            for node in first..130 {
                contacts.push(SoftRigidContactGpu {
                    ids: [node, u32::MAX, 0, 1],
                    ..point
                });
            }
            if mode.starts_with("depth") {
                let depth = if mode == "depth64" { 64 } else { 65 };
                for _ in 1..depth {
                    contacts.push(SoftRigidContactGpu {
                        ids: [1, u32::MAX, 0, 1],
                        ..point
                    });
                }
            } else {
                for node in (1..128).step_by(3) {
                    contacts.push(SoftRigidContactGpu {
                        normal_gap: [0.6, 0.8, 0.0, -0.00002],
                        ids: [node, u32::MAX, 0, 1],
                        support: [node + 1, 0, 2, 0],
                        weights: [0.4, 0.6, 0.0, 0.0],
                        ..point
                    });
                    contacts.push(SoftRigidContactGpu {
                        normal_gap: [0.0, 0.0, 1.0, -0.00003],
                        ids: [node, u32::MAX, 0, 1],
                        support: [node + 1, node + 2, 3, 0],
                        weights: [0.2, 0.3, 0.5, 0.0],
                        ..point
                    });
                }
            }
        }
        let bytes = |v: &[f32]| bytemuck::cast_slice(v).to_vec();
        eprintln!("separated static fixture: {mode}");
        replay_inputs(
            true,
            12,
            vec![
                bytes(&[0.2, 0.1, 0.3, 0.0].repeat(130)),
                bytes(&[0.2, -0.3, -0.1, 0.0].repeat(130)),
                bytes(&[0.7; 130]),
                bytemuck::cast_slice(&contacts).to_vec(),
                bytes(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
                bytes(&[0.0; 4]),
                bytes(&[0.2, 0.3, 0.4, 0.1, 0.2, 0.3]),
                bytes(&[2.0]),
                bytes(&[0.2]),
                bytes(&[0.0; 8]),
            ],
        );
    }
}

#[test]
#[ignore = "requires MOLLA_CONTACT_FIXTURE and a real GPU through make test-gpu"]
fn captured_contact_preparation_completes() {
    replay(false, 1);
}

#[test]
#[ignore = "requires MOLLA_CONTACT_FIXTURE and GPU timestamp queries through make test-gpu"]
fn captured_full_contact_profile_preserves_solver_outputs() {
    replay(true, 12);
}

#[test]
#[ignore = "requires MOLLA_CONTACT_FIXTURE and GPU timestamp queries through make test-gpu"]
fn captured_single_sweep_profile_preserves_solver_outputs() {
    replay(true, 1);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn cooperative_contact_checks_workgroup_limits_and_dimensions() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = block_on(instance.request_adapter(&Default::default())).unwrap();
    for (invocations, width, storage, supported) in [
        (64, 64, 2048, true),
        (32, 64, 2048, false),
        (64, 32, 2048, false),
        (64, 64, 1024, false),
    ] {
        let mut descriptor = wgpu::DeviceDescriptor::default();
        molla_solvers::fem_rigid_gpu::request_fem_device_limits(
            &mut descriptor.required_limits,
            &adapter.limits(),
        )
        .unwrap();
        descriptor
            .required_limits
            .max_compute_invocations_per_workgroup = invocations;
        descriptor.required_limits.max_compute_workgroup_size_x = width;
        descriptor
            .required_limits
            .max_compute_workgroup_storage_size = storage;
        let (device, _) = block_on(adapter.request_device(&descriptor)).unwrap();
        if storage == 1024 {
            assert!(SoftRigidSolveKernel::new_serial(&device, 64, 1).is_ok());
        }
        for (dofs, rows) in [
            (1, 0),
            (31, 1),
            (32, 1),
            (33, 1),
            (64, 1),
            (0, 1),
            (65, 1),
            (1, u32::MAX),
        ] {
            let valid = supported && (1..=64).contains(&dofs) && rows != u32::MAX;
            assert_eq!(
                SoftRigidSolveKernel::new(&device, dofs, rows).is_ok(),
                valid
            );
            assert_eq!(
                SoftRigidSolveKernel::new_cooperative(&device, dofs, rows).is_ok(),
                valid,
                "invocations={invocations}, width={width}, storage={storage}, dofs={dofs}, rows={rows}"
            );
            assert_eq!(
                SoftRigidSolveKernel::new_separated_static(&device, dofs, rows, 3).is_ok(),
                valid
            );
        }
        assert!(SoftRigidSolveKernel::new_separated_static(&device, 1, 1, u32::MAX).is_err());
    }
}

fn replay(profile: bool, iterations: u32) {
    let path = std::path::PathBuf::from(
        std::env::var_os("MOLLA_CONTACT_FIXTURE").expect("MOLLA_CONTACT_FIXTURE"),
    );
    let names = [
        "positions",
        "velocities",
        "inverse_mass",
        "contacts",
        "body_poses",
        "body_com",
        "jacobians",
        "mass_matrix",
        "rigid_velocities",
        "body_velocities",
    ];
    let inputs: Vec<_> = names
        .iter()
        .map(|name| std::fs::read(path.join(format!("{name}.bin"))).unwrap())
        .collect();
    replay_inputs(profile, iterations, inputs);
}

#[test]
#[ignore = "requires a real GPU with timestamp queries through make test-gpu"]
fn cooperative_contact_dof_boundaries_match_serial_bitwise() {
    for dofs in [1, 29, 31, 32, 33, 64] {
        let mut jacobians = vec![0.0_f32; 12 * dofs];
        let mut mass = vec![0.0_f32; dofs * dofs];
        for i in 0..dofs {
            for j in 0..dofs {
                mass[i * dofs + j] = if i == j { 2.0 + i as f32 * 0.1 } else { 0.01 };
            }
            for body in 0..2 {
                for axis in 0..6 {
                    jacobians[(body * 6 + axis) * dofs + i] =
                        ((i * 7 + axis * 3 + body * 11) % 17) as f32 * 0.02 - 0.15;
                }
            }
        }
        let point = SoftRigidContactGpu {
            normal_gap: [0.0, 1.0, 0.0, -0.00001],
            material: [0.6, 0.0, 0.0, 0.0],
            ids: [0, 0, 0, 1],
            support: [0, 0, 1, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        let mut contacts = vec![
            SoftRigidContactGpu {
                ids: [0; 4],
                ..point
            };
            131
        ];
        contacts[0] = point;
        contacts[63] = SoftRigidContactGpu {
            normal_gap: [0.0, 0.0, 1.0, -0.00002],
            ids: [0, 1, 0, 1],
            support: [1, 0, 2, 0],
            weights: [0.35, 0.65, 0.0, 0.0],
            ..point
        };
        contacts[64] = SoftRigidContactGpu {
            ids: [0, u32::MAX, 0, 1],
            support: [1, 2, 3, 0],
            weights: [0.2, 0.3, 0.5, 0.0],
            ..point
        };
        contacts[130] = SoftRigidContactGpu {
            ids: [0, 1, 0, 3],
            material: [0.02, -0.01, 0.03, 0.0],
            support: [0; 4],
            weights: [0.0; 4],
            ..point
        };
        let bytes = |values: &[f32]| bytemuck::cast_slice(values).to_vec();
        replay_inputs(
            true,
            12,
            vec![
                bytes(&[0.2, 0.0, 0.1, 0.0, 0.3, 0.1, 0.2, 0.0, 0.1, 0.2, 0.0, 0.0]),
                bytes(&[
                    0.3, -0.8, -0.6, 0.0, -0.2, -0.5, 0.1, 0.0, 0.1, -0.3, 0.2, 0.0,
                ]),
                bytes(&[1.0, 2.0, 0.5]),
                bytemuck::cast_slice(&contacts).to_vec(),
                bytes(&[
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
                ]),
                bytes(&[0.0; 8]),
                bytes(&jacobians),
                bytes(&mass),
                bytes(&(0..dofs).map(|i| 0.1 + i as f32 * 0.01).collect::<Vec<_>>()),
                bytes(&[0.0; 16]),
            ],
        );
    }
}

fn replay_inputs(profile: bool, iterations: u32, inputs: Vec<Vec<u8>>) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = block_on(instance.request_adapter(&Default::default())).unwrap();
    eprintln!("contact replay adapter: {:?}", adapter.get_info());
    let mut descriptor = wgpu::DeviceDescriptor::default();
    if profile {
        assert!(adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY));
        descriptor.required_features |= wgpu::Features::TIMESTAMP_QUERY;
    }
    molla_solvers::fem_rigid_gpu::request_fem_device_limits(
        &mut descriptor.required_limits,
        &adapter.limits(),
    )
    .unwrap();
    let (device, queue) = block_on(adapter.request_device(&descriptor)).unwrap();
    let (device, queue) = (&device, &queue);
    if !profile {
        assert!(SoftRigidSolveProfile::new(device).is_err());
    }
    let buffers: Vec<_> = inputs.iter().map(|bytes| buffer(device, bytes)).collect();
    let dofs = (buffers[8].size() / 4) as u32;
    let particles = (buffers[0].size() / 16) as u32;
    let bodies = (buffers[4].size() / 32) as u32;
    let rows = (buffers[3].size() / 80) as u32;
    let _featherstone = {
        let mut builder = molla_sim::ModelBuilder::new();
        builder.begin_articulation();
        let body = builder.add_body(molla_sim::BodyParams::new());
        builder
            .add_joint_revolute(
                molla_core::BodyId::NONE,
                body,
                molla_math::Transform::IDENTITY,
                molla_math::Transform::IDENTITY,
                molla_math::Vec3::X,
            )
            .unwrap();
        molla_kernels::FeatherstoneKernel::new(device, queue, &builder.build().unwrap())
    };
    assert_eq!(buffers[7].size(), dofs as u64 * dofs as u64 * 4);
    let kernel = SoftRigidSolveKernel::new_serial(device, dofs, rows).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let bindings = || SoftRigidSolveBindings {
        positions: &buffers[0],
        velocities: &buffers[1],
        inverse_mass: &buffers[2],
        contacts: &buffers[3],
        body_poses: &buffers[4],
        body_com: &buffers[5],
        jacobians: &buffers[6],
        mass_matrix: &buffers[7],
        rigid_velocities: &buffers[8],
        body_velocities: &buffers[9],
        particle_count: particles,
        body_count: bodies,
    };
    let settings = SoftRigidSolveSettings {
        iterations,
        dt: 1.0 / 19200.0,
        penetration_recovery: 0.2,
        max_recovery_speed: 0.5,
    };
    if profile {
        kernel
            .encode(device, &mut encoder, bindings(), settings)
            .unwrap();
    } else {
        kernel
            .encode_preparation(device, &mut encoder, bindings(), settings)
            .unwrap();
    }
    let start = Instant::now();
    eprintln!(
        "captured contact submit: full={profile}, dofs={dofs}, rows={rows}, particles={particles}, bodies={bodies}"
    );
    let submission = queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(30)),
        })
        .expect("captured contact preparation exceeded 30 seconds");
    eprintln!("captured contact completed in {:?}", start.elapsed());
    let read = |source: &wgpu::Buffer| {
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("captured contact acceptance"),
            size: source.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        staging.slice(..).get_mapped_range().to_vec()
    };
    assert_eq!(read(kernel.status()), 0u32.to_le_bytes());
    let bytes = read(kernel.row_state());
    let rows: &[[f32; 8]] = bytemuck::cast_slice(&bytes);
    assert!(rows.iter().flatten().all(|v| v.is_finite()));
    assert!(rows.iter().any(|r| r[0] > 0.0));
    if profile {
        let expected = [
            read(&buffers[1]),
            read(&buffers[8]),
            read(&buffers[9]),
            bytes,
        ];
        let cooperative =
            SoftRigidSolveKernel::new_cooperative(device, dofs, (buffers[3].size() / 80) as u32)
                .unwrap();
        let separated = SoftRigidSolveKernel::new_separated_static(
            device,
            dofs,
            (buffers[3].size() / 80) as u32,
            particles,
        )
        .unwrap();
        for (mode, kernel) in [
            ("serial", &kernel),
            ("cooperative", &cooperative),
            ("separated", &separated),
        ]
        .into_iter()
        .cycle()
        .take(if iterations == 1 { 9 } else { 3 })
        {
            for (destination, bytes) in buffers.iter().zip(&inputs) {
                queue.write_buffer(destination, 0, bytes);
            }
            let profile = SoftRigidSolveProfile::new(device).unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            let mut empty = bindings();
            empty.body_count = 0;
            assert!(
                kernel
                    .encode_profiled(device, &mut encoder, empty, settings, &profile)
                    .is_err()
            );
            if mode == "separated" {
                let mut wrong = bindings();
                wrong.particle_count += 1;
                assert!(
                    kernel
                        .encode(device, &mut encoder, wrong, settings)
                        .is_err()
                );
            }
            kernel
                .encode_profiled(device, &mut encoder, bindings(), settings, &profile)
                .unwrap();
            let submitted = Instant::now();
            let submission = queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(submission),
                    timeout: Some(Duration::from_secs(30)),
                })
                .expect("profiled contact solve exceeded 30 seconds");
            let elapsed = submitted.elapsed();
            assert_eq!(read(kernel.status()), 0u32.to_le_bytes());
            for (field, (actual, expected)) in [
                read(&buffers[1]),
                read(&buffers[8]),
                read(&buffers[9]),
                read(kernel.row_state()),
            ]
            .iter()
            .zip(&expected)
            .enumerate()
            {
                let difference = actual
                    .chunks_exact(4)
                    .zip(expected.chunks_exact(4))
                    .enumerate()
                    .find(|(_, (a, b))| a != b)
                    .map(|(index, (a, b))| {
                        (
                            index,
                            f32::from_le_bytes(a.try_into().unwrap()),
                            f32::from_le_bytes(b.try_into().unwrap()),
                        )
                    });
                assert!(
                    actual == expected,
                    "{mode} contact state mismatch field={field}, first word={difference:?}"
                );
                assert!(
                    actual
                        .chunks_exact(4)
                        .all(|b| f32::from_le_bytes(b.try_into().unwrap()).is_finite())
                );
            }
            let ticks = read(profile.ticks());
            let ticks: Vec<_> = ticks
                .chunks_exact(8)
                .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                .collect();
            assert_eq!(ticks.len(), 14);
            let mut total = 0.0;
            for (pass, pair) in [
                "factor", "prepare", "count", "offset", "scatter", "solve", "recover",
            ]
            .iter()
            .zip(ticks.chunks_exact(2))
            {
                let milliseconds = pair[1].wrapping_sub(pair[0]) as f64
                    * queue.get_timestamp_period() as f64
                    / 1e6;
                assert!(milliseconds.is_finite() && (0.0..30000.0).contains(&milliseconds));
                total += milliseconds;
                eprintln!("{mode} contact GPU pass {pass}: {milliseconds:.6}ms");
            }
            assert!(total > 0.0);
            let active = read(kernel.active_rows());
            if mode == "separated" {
                check_static_schedule(&inputs, &active);
                let base = buffers[3].size() as usize / 80;
                let base = base + 1 + base.div_ceil(64);
                let words: &[u32] = bytemuck::cast_slice(&active);
                eprintln!(
                    "separated enabled={}, dynamic={}, static_layers={}",
                    words[base],
                    words[base + 1],
                    words[base + 2]
                );
            }
            let active = u32::from_le_bytes(active[..4].try_into().unwrap());
            assert!(active > 0 && active <= (buffers[3].size() / 80) as u32);
            eprintln!(
                "contact active rows={active}, sweeps={}",
                settings.iterations
            );
            eprintln!(
                "{mode} contact GPU total={total:.6}ms, submit/wait={elapsed:?}; outputs bitwise identical"
            );
            if iterations == 1 {
                continue;
            }
            for (destination, bytes) in buffers.iter().zip(&inputs) {
                queue.write_buffer(destination, 0, bytes);
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            kernel
                .encode(
                    device,
                    &mut encoder,
                    bindings(),
                    SoftRigidSolveSettings {
                        iterations: 4,
                        ..settings
                    },
                )
                .unwrap();
            kernel
                .encode_continuation(
                    device,
                    &mut encoder,
                    bindings(),
                    SoftRigidSolveSettings {
                        iterations: 8,
                        ..settings
                    },
                )
                .unwrap();
            queue.submit([encoder.finish()]);
            assert_eq!(read(kernel.status()), 0u32.to_le_bytes());
            for (actual, expected) in [
                read(&buffers[1]),
                read(&buffers[8]),
                read(&buffers[9]),
                read(kernel.row_state()),
            ]
            .iter()
            .zip(&expected)
            {
                assert_eq!(
                    actual, expected,
                    "{mode} continued sweeps changed contact outputs"
                );
            }
            eprintln!("{mode} continued 4+8 sweeps match uninterrupted 12 sweeps bitwise");
        }
    }
}
