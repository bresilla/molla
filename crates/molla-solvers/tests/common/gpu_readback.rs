use molla_compute::Engine;

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
