use super::*;

#[test]
#[ignore = "requires GPU; alternating history owners and isolated failure seeds"]
fn newton_history_gpu_adjacent_owners() {
    use wgpu::util::DeviceExt;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut data = [
        vec![10, 30],
        vec![0u32; 24],
        vec![0; 32],
        vec![0; 32],
        vec![0; 8],
        vec![0; 8],
        vec![0; 8],
    ];
    for c in 0..2 {
        data[1][12 * c + 7] = 0.001f32.to_bits();
        data[1][12 * c + 8] = (2 + c) as u32;
        data[2][16 * c + 3] = 2f32.to_bits();
        data[2][16 * c + 5] = 1f32.to_bits();
        data[2][16 * c + 8] = 0.3f32.to_bits();
        data[2][16 * c + 12] = 4;
    }
    data[3][0] = 2f32.to_bits();
    data[3][16] = 3f32.to_bits();
    let make = |data: &[Vec<u32>; 7]| {
        data.each_ref().map(|words| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("adjacent history fixture"),
                contents: bytemuck::cast_slice(words),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
        })
    };
    let a_buffers = make(&data);
    let mut b_data = data.clone();
    b_data[0] = vec![5, 10];
    b_data[1][8] = 9;
    b_data[1][20] = 2;
    b_data[3].fill(0);
    let b_buffers = make(&b_data);
    let bytes = NewtonHistory::required_bytes(2).unwrap();
    let mut a = NewtonHistory::new(&engine, a_buffers.each_ref(), 2, 100, bytes).unwrap();
    let mut b = NewtonHistory::new(&engine, b_buffers.each_ref(), 2, 100, bytes).unwrap();
    let incompatible = NewtonHistory::new(&engine, b_buffers.each_ref(), 2, 101, bytes).unwrap();
    assert!(a.use_previous(device, &incompatible).is_err());
    a.use_previous(device, &b).unwrap();
    b.use_previous(device, &a).unwrap();
    let run = |history: &NewtonHistory, save| {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        if save {
            history.save(&mut encoder);
        } else {
            history.restore(&mut encoder);
        }
        queue.submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let traffic = transfers.finish();
        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    };
    run(&a, true);
    run(&b, false);
    let mut b_seed = vec![0u32; 32];
    b_seed[16] = 2f32.to_bits();
    assert_eq!(read_words(device, queue, b.seed_buffer()), b_seed);
    assert_eq!(read_words(device, queue, &b_buffers[3]), b_seed);
    queue.write_buffer(&b_buffers[3], 64, bytemuck::bytes_of(&7f32));
    run(&b, true);
    run(&a, false);
    let mut a_seed = vec![0u32; 32];
    a_seed[0] = 7f32.to_bits();
    assert_eq!(read_words(device, queue, a.seed_buffer()), a_seed);
    assert_eq!(read_words(device, queue, &a_buffers[3]), a_seed);
    assert_eq!(read_words(device, queue, b.seed_buffer()), b_seed);
    assert_eq!(&read_words(device, queue, a.counters())[2..], &[1, 1]);
    assert_eq!(&read_words(device, queue, b.counters())[2..], &[1, 1]);
    queue.write_buffer(&b_buffers[4], 0, bytemuck::bytes_of(&8u32));
    run(&b, false);
    run(&b, true);
    run(&a, false);
    assert_eq!(read_words(device, queue, b.seed_buffer()), b_seed);
    assert_eq!(read_words(device, queue, &a_buffers[3]), vec![0u32; 32]);
    a.invalidate(queue);
    b.invalidate(queue);
    assert_eq!(read_words(device, queue, a.counters()), [0; 4]);
    assert_eq!(read_words(device, queue, b.counters()), [0; 4]);
    run(&a, false);
    assert_eq!(read_words(device, queue, &a_buffers[3]), vec![0u32; 32]);
    eprintln!(
        "adjacent parity consumes newest matching identity, preserves per-owner seeds, rejects donor faults and resets without hot allocation or transfer"
    );
}
fn read_words(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    use wgpu::util::DownloadBuffer;
    let (tx, rx) = std::sync::mpsc::channel();
    DownloadBuffer::read_buffer(device, queue, &buffer.slice(..), move |result| {
        let _ = tx.send(result);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(15)),
        })
        .unwrap();
    let bytes = rx
        .recv_timeout(std::time::Duration::from_secs(15))
        .unwrap()
        .unwrap();
    bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

#[test]
fn newton_history_shader_and_budget() {
    let source = include_str!("../shaders/soil_newton_history.wgsl");
    let module = wgpu::naga::front::wgsl::parse_str(source).unwrap();
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(NewtonHistory::required_bytes(4).unwrap(), 736);
    assert!(NewtonHistory::required_bytes(0).is_err());
    assert!(NewtonHistory::required_bytes(16385).is_err());
}

#[test]
#[ignore = "requires GPU; identity-safe history and invalidation"]
fn newton_history_gpu_keys_faults_and_reset() {
    use wgpu::util::DeviceExt;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut data = [
        vec![10, 30, u32::MAX, u32::MAX],
        vec![0; 48],
        vec![0; 64],
        vec![0; 64],
        vec![0; 8],
        vec![0; 8],
        vec![0; 8],
    ];
    for c in 0..4 {
        data[1][12 * c + 7] = 0.001f32.to_bits();
        data[1][12 * c + 8] = (2 + c) as u32;
        data[2][16 * c + 3] = 2f32.to_bits();
        data[2][16 * c + 5] = 1f32.to_bits();
        data[2][16 * c + 8] = 0.3f32.to_bits();
        data[2][16 * c + 12] = 4;
        data[3][16 * c] = ((1 + 2 * c) as f32).to_bits();
    }
    let buffers = data.each_ref().map(|values| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("history fixture"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    });
    let inputs = buffers.each_ref();
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(NewtonHistory::new(&engine, inputs, 4, 100, 735).is_err());
    assert_eq!(scope.finish().observations, 0);
    let history = NewtonHistory::new(&engine, inputs, 4, 100, 736).unwrap();
    let run = |save| {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        if save {
            history.save(&mut encoder);
        } else {
            history.restore(&mut encoder);
        }
        queue.submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let traffic = transfers.finish();
        assert_eq!(traffic.readback_bytes, 0);
        assert_eq!(traffic.upload_bytes, 0);
    };
    let write = |data: &[Vec<u32>; 7]| {
        for (b, words) in buffers.iter().zip(data) {
            queue.write_buffer(b, 0, bytemuck::cast_slice(words));
        }
    };
    let original = data.clone();
    run(true);
    data[0] = vec![5, 10, 30, u32::MAX];
    data[1][20] = 2;
    data[1][32] = 3;
    data[2][19] = 1f32.to_bits();
    data[2][35] = 4f32.to_bits();
    data[3].fill(0);
    let current = data.clone();
    write(&data);
    run(false);
    let restored = read_words(device, queue, &buffers[3]);
    let mut expected = vec![0; 64];
    expected[16] = 0.5f32.to_bits();
    expected[32] = 6f32.to_bits();
    assert_eq!(restored, expected);
    assert_eq!(read_words(device, queue, history.seed_buffer()), expected);
    data[4][0] = 8;
    write(&data);
    run(false);
    assert_eq!(read_words(device, queue, history.seed_buffer()), expected);
    assert_eq!(&read_words(device, queue, history.counters())[2..], [2, 2]);
    for case in 0..6 {
        data = current.clone();
        match case {
            0 => data[1][20] = 9,
            1 => data[2][28] = 0,
            2 => data[2][24] = 0.4f32.to_bits(),
            3 => data[1][19] = 0.002f32.to_bits(),
            4 => data[2][21] = (-1f32).to_bits(),
            _ => data[2][19] = 5f32.to_bits(),
        }
        write(&data);
        run(false);
        let restored = read_words(device, queue, &buffers[3]);
        assert!(restored[16..32].iter().all(|&v| v == 0), "case {case}");
        assert_eq!(restored[32], 6f32.to_bits());
    }
    for fault in 4..7 {
        data = original.clone();
        data[fault][0] = 8;
        write(&data);
        run(true);
        data = current.clone();
        write(&data);
        run(false);
        assert!(
            read_words(device, queue, &buffers[3])
                .iter()
                .all(|&v| v == 0)
        );
        assert_ne!(read_words(device, queue, history.counters())[1], 0);
    }
    data = original.clone();
    data[0][1] = 10;
    write(&data);
    run(true);
    data = current.clone();
    write(&data);
    run(false);
    assert!(
        read_words(device, queue, &buffers[3])
            .iter()
            .all(|&v| v == 0)
    );
    write(&original);
    run(true);
    history.invalidate(queue);
    write(&current);
    run(false);
    assert!(
        read_words(device, queue, &buffers[3])
            .iter()
            .all(|&v| v == 0)
    );
    assert_eq!(read_words(device, queue, history.counters()), [0; 4]);
    eprintln!(
        "history key insertion/reordering, confidence guards, certificate faults, duplicate rejection and invalidation pass; no hot allocation or transfer"
    );
}
