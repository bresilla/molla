#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::tet_volume::{TetVolumeBindings, TetVolumeMonitor};
use wgpu::util::DeviceExt;

fn read(engine: &Engine, buffer: &wgpu::Buffer, count: usize) -> Vec<u32> {
    let device = engine.device().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("tet volume acceptance"),
        size: count as u64 * 4,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, count as u64 * 4);
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec()
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_minimum_remembers_transient_compression_and_faults() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let reference = [
        [0.0_f32, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 0.0, 0.0],
        [3.0, 1.0, 0.0, 0.0],
        [3.0, 0.0, 1.0, 0.0],
    ];
    let allocate = |bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tet volume fixture"),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
    };
    let rest = allocate(bytemuck::cast_slice(&reference));
    let q = allocate(bytemuck::cast_slice(&reference));
    let indices = allocate(bytemuck::cast_slice(&[0_u32, 1, 2, 3, 4, 5, 6, 7]));
    let inv = allocate(bytemuck::cast_slice(
        &[[
            1.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]; 2],
    ));
    let status = allocate(bytemuck::cast_slice(&[0_u32]));
    let sample = |monitor: &TetVolumeMonitor, positions: &[[f32; 4]; 8]| {
        queue.write_buffer(&q, 0, bytemuck::cast_slice(positions));
        let mut encoder = device.create_command_encoder(&Default::default());
        monitor
            .encode(
                device,
                &mut encoder,
                TetVolumeBindings {
                    positions: &q,
                    reference: &rest,
                    indices: &indices,
                    inverse_rest: &inv,
                    status: &status,
                    particle_count: 8,
                    tet_count: 2,
                },
            )
            .unwrap();
        queue.submit([encoder.finish()]);
    };
    let monitor = TetVolumeMonitor::new(device);
    for scale in [1.0_f32, 0.4, 1.2] {
        let mut positions = reference;
        positions[7][2] = scale;
        sample(&monitor, &positions);
    }
    let recorded = read(&engine, monitor.diagnostics(), 2);
    assert!((f32::from_bits(recorded[0]) - 0.4).abs() < 1e-6);
    assert_eq!(recorded[1], u32::MAX);
    assert_eq!(read(&engine, &status, 1), [0]);

    for (scale, flag) in [(-0.1, 8), (0.0, 8), (f32::NAN, 4), (f32::INFINITY, 4)] {
        let monitor = TetVolumeMonitor::new(device);
        queue.write_buffer(&status, 0, bytemuck::cast_slice(&[1_u32]));
        let mut positions = reference;
        positions[7][2] = scale;
        sample(&monitor, &positions);
        sample(&monitor, &reference);
        assert_eq!(read(&engine, monitor.diagnostics(), 2), [0, 1]);
        assert_eq!(read(&engine, &status, 1), [1 | flag]);
    }
    let monitor = TetVolumeMonitor::new(device);
    queue.write_buffer(&status, 0, bytemuck::cast_slice(&[0_u32]));
    queue.write_buffer(
        &indices,
        0,
        bytemuck::cast_slice(&[0_u32, 1, 2, 999, 4, 5, 6, 7]),
    );
    sample(&monitor, &reference);
    assert_eq!(read(&engine, monitor.diagnostics(), 2), [0, 0]);
    assert_eq!(read(&engine, &status, 1), [8]);
}
