use molla_compute::Engine;
use molla_math::{Quat, Real, Transform, Vec3};
use molla_solvers::fem_rigid_gpu::FemRigidGpuSystem;

pub fn read<T: bytemuck::Pod>(engine: &Engine, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let size = (std::mem::size_of::<T>() * count) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("track acceptance"),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec()
}

pub fn body_poses(engine: &Engine, system: &FemRigidGpuSystem, count: usize) -> Vec<Transform> {
    read::<[f32; 8]>(
        engine,
        system.rigid_state().body_q.device_buffer().unwrap(),
        count,
    )
    .iter()
    .map(|p| Transform {
        position: Vec3::new(p[0] as Real, p[1] as Real, p[2] as Real),
        rotation: Quat::from_xyzw(p[4] as Real, p[5] as Real, p[6] as Real, p[7] as Real)
            .normalize(),
    })
    .collect()
}
