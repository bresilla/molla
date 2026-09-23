#![cfg(feature = "gpu")]

use molla_solvers::fem_rigid_gpu::{
    FemRigidGpuSystem, MaterialContactConfig, SoftRigidShapeGpu, request_fem_device_limits,
};
use std::sync::Arc;

#[path = "common/fem_rotor.rs"]
mod fem_rotor;

fn block_on<T>(future: impl std::future::Future<Output = T>) -> T {
    struct WakeThread(std::thread::Thread);
    impl std::task::Wake for WakeThread {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = std::task::Waker::from(Arc::new(WakeThread(std::thread::current())));
    let mut context = std::task::Context::from_waker(&waker);
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut context) {
            std::task::Poll::Ready(value) => return value,
            std::task::Poll::Pending => std::thread::park(),
        }
    }
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u8> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("FEM timing acceptance"),
        size: buffer.size(),
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |v| tx.send(v).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    staging.slice(..).get_mapped_range().to_vec()
}

#[test]
#[ignore = "requires GPU timestamp queries through make test-gpu"]
fn timing_pages_preserve_state_and_enforce_idle_configuration() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = block_on(instance.request_adapter(&Default::default())).unwrap();
    assert!(adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY));
    for enabled_feature in [false, true] {
        let mut descriptor = wgpu::DeviceDescriptor::default();
        request_fem_device_limits(&mut descriptor.required_limits, &adapter.limits()).unwrap();
        if enabled_feature {
            descriptor.required_features = wgpu::Features::TIMESTAMP_QUERY;
        }
        let (device, queue) = block_on(adapter.request_device(&descriptor)).unwrap();
        let (device, queue) = (Arc::new(device), Arc::new(queue));
        let make = || {
            let mut scene = fem_rotor::scene();
            scene.shapes.push(SoftRigidShapeGpu {
                position: [0.0, -0.1, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                data: [10.0, 0.1, 10.0, 0.85],
                ids: [1, u32::MAX, 0, 0],
            });
            FemRigidGpuSystem::new_on_device(
                device.clone(),
                queue.clone(),
                scene,
                fem_rotor::config(),
            )
            .unwrap()
        };
        if !enabled_feature {
            let mut system = make();
            assert!(system.configure_gpu_timing(true).is_err());
            assert!(system.gpu_timing_ticks().is_none());
            system.configure_gpu_timing(false).unwrap();
            continue;
        }
        for outer in [None, Some(2), Some(130)] {
            let mut plain = make();
            let mut timed = make();
            for system in [&mut plain, &mut timed] {
                if let Some(iterations) = outer {
                    system
                        .configure_material_contact(Some(MaterialContactConfig {
                            iterations,
                            linear_tolerance: 1e-4,
                            angular_tolerance: 1e-4,
                        }))
                        .unwrap();
                    system.configure_material_contact_reuse(Some(1)).unwrap();
                }
                system.set_drive_effort(0, 0.25).unwrap();
            }
            timed.configure_gpu_timing(true).unwrap();
            assert!(timed.gpu_timing_ticks().is_none());
            let count = 2 * outer.unwrap_or(1) + 3;
            for _ in 0..2 {
                plain.step(fem_rotor::config().max_substep).unwrap();
                assert!(
                    timed
                        .try_step_substep(fem_rotor::config().max_substep)
                        .unwrap()
                );
                assert!(timed.configure_gpu_timing(false).is_err());
                while !timed.poll_completion().unwrap() {
                    std::thread::yield_now();
                }
                let raw = read(&device, &queue, timed.gpu_timing_ticks().unwrap());
                let ticks: &[u64] = bytemuck::cast_slice(&raw);
                assert_eq!(ticks.len(), count as usize);
                assert!(
                    ticks[0] > 0 && ticks.windows(2).all(|v| v[1] >= v[0]),
                    "{ticks:?}"
                );
                assert!(ticks[ticks.len() - 1] > ticks[0]);
                for (a, b) in [
                    (
                        plain.soft_state().particle_q.device_buffer().unwrap(),
                        timed.soft_state().particle_q.device_buffer().unwrap(),
                    ),
                    (
                        plain.soft_state().particle_qd.device_buffer().unwrap(),
                        timed.soft_state().particle_qd.device_buffer().unwrap(),
                    ),
                    (
                        plain.rigid_state().joint_qd.device_buffer().unwrap(),
                        timed.rigid_state().joint_qd.device_buffer().unwrap(),
                    ),
                    (
                        plain.rigid_state().body_q.device_buffer().unwrap(),
                        timed.rigid_state().body_q.device_buffer().unwrap(),
                    ),
                    (plain.status_buffer(), timed.status_buffer()),
                ] {
                    assert_eq!(read(&device, &queue, a), read(&device, &queue, b));
                }
                assert_eq!(
                    read(&device, &queue, timed.status_buffer()),
                    0u32.to_le_bytes()
                );
            }
            timed.configure_gpu_timing(false).unwrap();
            assert!(timed.gpu_timing_ticks().is_none());
            timed.configure_gpu_timing(true).unwrap();
            timed.configure_material_contact(None).unwrap();
            assert!(timed.gpu_timing_ticks().is_none());
            timed
                .configure_material_contact(Some(MaterialContactConfig {
                    iterations: 4096,
                    linear_tolerance: 1e-4,
                    angular_tolerance: 1e-4,
                }))
                .unwrap();
            timed.configure_gpu_timing(true).unwrap();
            timed.configure_gpu_timing(false).unwrap();
        }
    }
}
