#![cfg(feature = "gpu")]

use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
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
    let path = std::path::PathBuf::from(
        std::env::var_os("MOLLA_CONTACT_FIXTURE").expect("MOLLA_CONTACT_FIXTURE"),
    );
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = block_on(instance.request_adapter(&Default::default())).unwrap();
    eprintln!("contact replay adapter: {:?}", adapter.get_info());
    let mut descriptor = wgpu::DeviceDescriptor::default();
    molla_solvers::fem_rigid_gpu::request_fem_device_limits(
        &mut descriptor.required_limits,
        &adapter.limits(),
    )
    .unwrap();
    let (device, queue) = block_on(adapter.request_device(&descriptor)).unwrap();
    let (device, queue) = (&device, &queue);
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
    let buffers: Vec<_> = names
        .iter()
        .map(|name| {
            let bytes = std::fs::read(path.join(format!("{name}.bin"))).unwrap();
            buffer(device, &bytes)
        })
        .collect();
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
    kernel
        .encode_preparation(
            device,
            &mut encoder,
            SoftRigidSolveBindings {
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
            },
            SoftRigidSolveSettings {
                iterations: 1,
                dt: 1.0 / 19200.0,
                penetration_recovery: 0.2,
                max_recovery_speed: 0.5,
            },
        )
        .unwrap();
    let start = Instant::now();
    eprintln!(
        "captured preparation submit: dofs={dofs}, rows={rows}, particles={particles}, bodies={bodies}"
    );
    let submission = queue.submit([encoder.finish()]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(Duration::from_secs(30)),
        })
        .expect("captured contact preparation exceeded 30 seconds");
    eprintln!("captured preparation completed in {:?}", start.elapsed());
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
}
