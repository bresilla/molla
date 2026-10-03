#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_geometry::LbvhBroadPhase;
use molla_math::Vec3;

fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u32> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = buffer.slice(..).get_mapped_range();
    let result = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    result
}

fn staging(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("LBVH test snapshots"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_refits_share_encoder_preserve_root_and_bindings() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut lbvh = LbvhBroadPhase::new(device.clone(), queue.clone(), 4, 64).unwrap();
    let bounds = [
        (Vec3::ZERO, Vec3::ONE),
        (Vec3::splat(3.0), Vec3::splat(4.0)),
        (Vec3::splat(6.0), Vec3::splat(7.0)),
    ];
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(lbvh.encode_refit(&mut encoder, 3).is_err());
    lbvh.dispatch_no_readback(&bounds).unwrap();
    assert!(lbvh.encode_refit(&mut encoder, 2).is_err());
    let diagnostics = lbvh.diagnostics();
    assert_eq!(diagnostics.pair_readbacks, 0);
    assert_eq!(diagnostics.bind_groups_created, 3);

    let variants = [
        [0.0_f32, 3.0, 6.0],
        [0.0, 0.5, 6.0],
        [0.0, 3.0, 0.5],
        [0.0, 0.25, 0.5],
        [100.0, 200.0, 300.0],
    ];
    let data: Vec<[f32; 4]> = variants
        .iter()
        .flat_map(|positions| {
            positions
                .iter()
                .flat_map(|&p| [[p, p, p, 0.0], [p + 1.0, p + 1.0, p + 1.0, 0.0]])
        })
        .collect();
    let source = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("LBVH AABB snapshots"),
        size: (data.len() * 16) as u64,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&source, 0, bytemuck::cast_slice(&data));
    let frames = 100;
    let output = staging(device, frames * 40);
    for frame in 0..frames {
        encoder.copy_buffer_to_buffer(
            &source,
            (frame % 5) * 96,
            lbvh.primitive_aabb_buffer(),
            0,
            96,
        );
        lbvh.encode_refit(&mut encoder, 3).unwrap();
        let (count, a, b) = lbvh.pair_buffers();
        encoder.copy_buffer_to_buffer(count, 0, &output, frame * 40, 16);
        encoder.copy_buffer_to_buffer(a, 0, &output, frame * 40 + 16, 12);
        encoder.copy_buffer_to_buffer(b, 0, &output, frame * 40 + 28, 12);
    }
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    let root = words[1];
    assert!((3..5).contains(&root));
    for (frame, row) in words.chunks_exact(10).enumerate() {
        assert_eq!(row[1], root);
        let count = row[0] as usize;
        assert!(count <= 3);
        let mut pairs: Vec<_> = (0..count).map(|i| (row[4 + i], row[7 + i])).collect();
        pairs.sort();
        let expected = match frame % 5 {
            1 => vec![(0, 1)],
            2 => vec![(0, 2)],
            3 => vec![(0, 1), (0, 2), (1, 2)],
            _ => vec![],
        };
        assert_eq!(pairs, expected, "frame {frame}");
    }
    lbvh.refit_in_place(3).unwrap();
    assert_eq!(lbvh.diagnostics(), diagnostics);
    assert!(lbvh.refit_only(&bounds).unwrap().a.is_empty());
    assert_eq!(lbvh.diagnostics().pair_readbacks, 1);
    assert_eq!(lbvh.diagnostics().bind_groups_created, 3);
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_overflow_stays_bounded_and_empty_build_clears_count() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut lbvh = LbvhBroadPhase::new(device.clone(), queue.clone(), 2, 64).unwrap();
    let dense = vec![(Vec3::ZERO, Vec3::ONE); 130];
    lbvh.dispatch_no_readback(&dense).unwrap();
    let warmed = lbvh.diagnostics();
    assert_eq!(warmed.bind_groups_created, 6);
    let output = staging(device, 32);
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(lbvh.pair_count_buffer(), 0, &output, 0, 16);
    lbvh.encode_refit(&mut encoder, 130).unwrap();
    encoder.copy_buffer_to_buffer(lbvh.pair_count_buffer(), 0, &output, 16, 16);
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    assert_eq!(words[0], 130 * 129 / 2);
    assert_eq!(words[4], words[0]);
    assert_eq!(words[1], words[5]);
    assert_eq!(lbvh.pair_capacity(), 64);
    assert_eq!(lbvh.diagnostics(), warmed);
    for bounds in [&dense[..1], &dense[..0]] {
        lbvh.dispatch_no_readback(bounds).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        assert!(lbvh.encode_refit(&mut encoder, 130).is_err());
        if bounds.len() == 1 {
            lbvh.encode_refit(&mut encoder, 1).unwrap();
        }
        encoder.copy_buffer_to_buffer(lbvh.pair_count_buffer(), 0, &output, 0, 16);
        queue.submit([encoder.finish()]);
        assert_eq!(&read(device, &output)[..4], &[0; 4]);
        if bounds.len() == 1 {
            let leaf = lbvh.debug_dump_nodes(1).unwrap()[0];
            assert_eq!(leaf.is_leaf, 1);
            assert_eq!(leaf.parent, u32::MAX);
            assert_eq!(leaf.aabb_min, [0.0; 3]);
            assert_eq!(leaf.aabb_max, [1.0; 3]);
        }
    }
    let truncated = lbvh.dispatch(&dense).unwrap();
    assert!(truncated.overflowed);
    assert_eq!(truncated.a.len(), 64);
    assert!(lbvh.pair_capacity() >= 8385);
    let complete = lbvh.refit_only(&dense).unwrap();
    assert!(!complete.overflowed);
    assert_eq!(complete.a.len(), 8385);
    lbvh.refit_in_place(130).unwrap();
    assert_eq!(lbvh.diagnostics().pair_readbacks, 2);
}
