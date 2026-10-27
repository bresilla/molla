use super::*;
use molla_compute::Engine;

fn storage(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("publication fixture"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    )
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-publication-gpu"]
fn sticky_faults_and_break_stop_preserve_outputs_until_explicit_reset() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let lengths = [257_usize, 129];
    let sources = lengths.map(|n| storage(device, (n + 8) as u64 * 4));
    let destinations = lengths.map(|n| storage(device, (n + 8) as u64 * 4));
    let status = storage(device, 8);
    molla_core::gpu_write_buffer!(queue, &status, 0, bytemuck::cast_slice(&[0_u32, u32::MAX]));
    let values = lengths.map(|n| {
        (0..n + 8)
            .map(|i| 0x7f80_0001_u32.wrapping_add(i as u32 * 7919))
            .collect::<Vec<_>>()
    });
    for i in 0..2 {
        molla_core::gpu_write_buffer!(queue, &sources[i], 0, bytemuck::cast_slice(&values[i]));
        molla_core::gpu_write_buffer!(
            queue,
            &destinations[i],
            0,
            bytemuck::cast_slice(&vec![0xdead_beef_u32; lengths[i] + 8]),
        );
    }
    let bindings: Vec<_> = (0..2)
        .map(|i| ContactPublicationBinding {
            source: &sources[i],
            destination: &destinations[i],
            bytes: lengths[i] as u64 * 4,
        })
        .collect();
    let publication = ResidentContactPublication::new(device, &status, &bindings).unwrap();
    let statuses: Vec<u32> = std::iter::once(0)
        .chain((0..32).map(|bit| 1_u32 << bit))
        .chain([u32::MAX, 1 << 30, 0])
        .collect();
    let schedule = storage(device, statuses.len() as u64 * 4);
    molla_core::gpu_write_buffer!(queue, &schedule, 0, bytemuck::cast_slice(&statuses));
    let stride = 8 + destinations.iter().map(wgpu::Buffer::size).sum::<u64>();
    let readback = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("publication snapshots"),
            size: stride * statuses.len() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(publication.encode(&mut encoder, 2).is_err());
    for case in 0..statuses.len() {
        if case == 1 {
            for source in &sources {
                encoder.clear_buffer(source, 0, None);
            }
        }
        encoder.copy_buffer_to_buffer(&schedule, case as u64 * 4, &status, 0, 4);
        let mut offset = case as u64 * stride;
        encoder.copy_buffer_to_buffer(&status, 0, &readback, offset, 8);
        offset += 8;
        for (i, destination) in destinations.iter().enumerate() {
            publication.encode(&mut encoder, i).unwrap();
            publication.encode(&mut encoder, i).unwrap();
            encoder.copy_buffer_to_buffer(destination, 0, &readback, offset, destination.size());
            offset += destination.size();
        }
    }
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = molla_core::gpu_readback!(readback.slice(..));
    for (case, frame) in bytes.chunks_exact(stride as usize).enumerate() {
        let words: &[u32] = bytemuck::cast_slice(frame);
        assert_eq!(&words[..2], &[statuses[case], u32::MAX]);
        let mut offset = 2;
        for i in 0..2 {
            let n = lengths[i];
            let expected = if case == statuses.len() - 1 {
                vec![0; n]
            } else {
                values[i][..n].to_vec()
            };
            assert_eq!(
                &words[offset..offset + n],
                expected,
                "case={case} binding={i}"
            );
            assert_eq!(&words[offset + n..offset + n + 8], &[0xdead_beef; 8]);
            offset += n + 8;
        }
    }
    drop(bytes);
    readback.unmap();
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-publication-gpu"]
fn invalid_publication_bindings_fail_before_encoding() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let status = storage(device, 4);
    let source = storage(device, 64);
    let destination = storage(device, 64);
    let non_storage = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("invalid publication buffer"),
            size: 64,
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    );
    assert!(ResidentContactPublication::new(device, &status, &[]).is_err());
    for (s, from, to, bytes) in [
        (&status, &source, &destination, 0),
        (&status, &source, &destination, 3),
        (&status, &source, &destination, 68),
        (&status, &source, &destination, u64::MAX - 3),
        (&status, &source, &source, 4),
        (&status, &source, &status, 4),
        (&status, &status, &destination, 4),
        (&status, &non_storage, &destination, 4),
        (&status, &source, &non_storage, 4),
        (&non_storage, &source, &destination, 4),
    ] {
        assert!(
            ResidentContactPublication::new(
                device,
                s,
                &[ContactPublicationBinding {
                    source: from,
                    destination: to,
                    bytes
                }],
            )
            .is_err()
        );
    }
    ResidentContactPublication::new(
        device,
        &status,
        &[ContactPublicationBinding {
            source: &source,
            destination: &destination,
            bytes: 64,
        }],
    )
    .unwrap();
}
