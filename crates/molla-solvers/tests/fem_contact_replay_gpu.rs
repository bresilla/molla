#![cfg(feature = "gpu")]

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

#[test]
#[ignore = "requires MOLLA_CONTACT_FIXTURE and a real GPU through make test-gpu"]
fn captured_contact_preparation_completes() {
    replay(false);
}

#[test]
#[ignore = "requires MOLLA_CONTACT_FIXTURE and GPU timestamp queries through make test-gpu"]
fn captured_full_contact_profile_preserves_solver_outputs() {
    replay(true);
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
            assert!(SoftRigidSolveKernel::new(&device, 64, 1).is_ok());
        }
        for (dofs, rows) in [(1, 0), (64, 1), (0, 1), (65, 1), (1, u32::MAX)] {
            let valid = supported && (1..=64).contains(&dofs) && rows != u32::MAX;
            assert_eq!(
                SoftRigidSolveKernel::new_cooperative(&device, dofs, rows).is_ok(),
                valid,
                "invocations={invocations}, width={width}, storage={storage}, dofs={dofs}, rows={rows}"
            );
        }
    }
}

fn replay(profile: bool) {
    let path = std::path::PathBuf::from(
        std::env::var_os("MOLLA_CONTACT_FIXTURE").expect("MOLLA_CONTACT_FIXTURE"),
    );
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
    let kernel = SoftRigidSolveKernel::new(device, dofs, rows).unwrap();
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
        iterations: if profile { 12 } else { 1 },
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
        for (mode, kernel) in [("serial", &kernel), ("cooperative", &cooperative)] {
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
            let active = u32::from_le_bytes(active[..4].try_into().unwrap());
            assert!(active > 0 && active <= (buffers[3].size() / 80) as u32);
            eprintln!(
                "contact active rows={active}, sweeps={}",
                settings.iterations
            );
            eprintln!(
                "{mode} contact GPU total={total:.6}ms, submit/wait={elapsed:?}; outputs bitwise identical"
            );
        }
    }
}
