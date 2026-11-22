#![cfg(feature = "gpu")]

use molla_compute::{Engine, copy_to_mapped};

#[test]
#[ignore = "requires a GPU"]
fn bounded_mapped_copy_preserves_every_byte_including_partial_blocks() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for len in [4, 128, 256, 260, 1024, 65_540, 2_000_012] {
        let expected: Vec<u8> = (0..len).map(|i| (i * 163 + i / 251) as u8).collect();
        let source = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mapped copy source"),
            size: len as u64,
            usage: wgpu::BufferUsages::MAP_WRITE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: true,
        });
        let target = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mapped copy readback"),
            size: len as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let start = std::time::Instant::now();
        copy_to_mapped(
            &mut source.slice(..).get_mapped_range_mut().unwrap(),
            &expected,
        );
        println!(
            "mapped copy {len} bytes: {:.3} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
        source.unmap();
        let mut commands = device.create_command_encoder(&Default::default());
        commands.copy_buffer_to_buffer(&source, 0, &target, 0, len as u64);
        queue.submit([commands.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        target
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        {
            let actual = target.slice(..).get_mapped_range().unwrap();
            assert_eq!(&*actual, expected.as_slice());
        }
        target.unmap();
    }
}
