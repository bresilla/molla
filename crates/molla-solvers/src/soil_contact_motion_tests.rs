use super::*;

#[test]
#[ignore = "requires real GPU"]
fn paired_soil_consumer_motion_retains_cancelled_rate_tails() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let inputs = [12, 4, 4, 4, 12, 4, 4, 4].map(|n| buffer(&engine, &vec![0; n]));
    let write = |i: usize, values: &[f32]| {
        engine
            .queue()
            .unwrap()
            .write_buffer(&inputs[i], 0, bytemuck::cast_slice(values))
    };
    write(0, &[0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0.]);
    write(1, &[0., 1., 0., 0.]);
    write(3, &[0., 1., 0., 1.]);
    write(4, &[1., 1.]);
    let tail = 2f32.powi(-30);
    write(5, &[1., tail, -1., 0.]);
    let reader = ContactMotionGpu::with_rate_encoding(
        &engine,
        inputs.each_ref(),
        2,
        1,
        1,
        112,
        RateEncoding::Paired,
    )
    .unwrap();
    let sample = run(&engine, &reader).unwrap()[0];
    assert_eq!(sample.supporting_nodes, 1);
    assert!((sample.tangential_slip_velocity.unwrap()[0] - tail).abs() < 1e-15);
    assert!((sample.mean_tangential_slip_speed.unwrap() - tail).abs() < 1e-15);
    let short = buffer(&engine, &[0; 2]);
    let mut bad = inputs.each_ref();
    bad[5] = &short;
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        ContactMotionGpu::with_rate_encoding(&engine, bad, 2, 1, 1, 112, RateEncoding::Paired)
            .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    write(5, &[1., f32::NAN, -1., 0.]);
    assert!(run(&engine, &reader).is_err());
}

fn buffer(engine: &Engine, words: &[u32]) -> wgpu::Buffer {
    let result = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("contact motion fixture"),
            size: words.len() as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    engine
        .queue()
        .unwrap()
        .write_buffer(&result, 0, bytemuck::cast_slice(words));
    result
}

fn run(engine: &Engine, reader: &ContactMotionGpu) -> Result<Vec<SoilContactMotion>> {
    let device = engine.device().unwrap();
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    reader.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
    let (tx, rx) = std::sync::mpsc::channel();
    reader
        .staging()
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let result = reader.decode(&reader.staging().slice(..).get_mapped_range());
    reader.staging().unmap();
    result
}

#[test]
#[ignore = "requires GPU"]
fn moving_soil_slip_preserves_frames_support_and_faults() {
    let engine = Engine::new_gpu().unwrap();
    let sizes = [65 * 12, 65 * 4, 65 * 4, 65 * 4, 3 * 6 * 6, 6, 3 * 4, 4];
    let inputs = sizes.map(|size| buffer(&engine, &vec![0; size]));
    let reader = ContactMotionGpu::new(&engine, inputs.each_ref(), 6, 3, 65, 304).unwrap();
    assert_eq!(ContactMotionGpu::bytes(3), 304);
    for (rotated, offset, boost) in [
        (false, 0.0, [0.0; 3]),
        (false, 1024.0, [8.0, -4.0, 2.0]),
        (true, 0.0, [0.0; 3]),
    ] {
        let rotate = |v: [f32; 3]| if rotated { [v[2], v[1], -v[0]] } else { v };
        let mut queries = vec![0u32; 65 * 12];
        let mut normals = vec![0f32; 65 * 4];
        let mut velocities = vec![0f32; 65 * 4];
        let mut impulses = vec![0f32; 65 * 4];
        let mut centers = vec![0f32; 3 * 4];
        for body in 0..3 {
            for axis in 0..3 {
                centers[body * 4 + axis] = rotate([10.0 * body as f32, 20.0, 30.0])[axis] + offset;
            }
        }
        for (slot, body, arm, velocity, weight) in [
            (0, 0, [0.0, -1.0, 0.0], [0.5, 4.0, 0.0], 2.0),
            (1, 0, [0.0, -2.0, 0.0], [0.5, 0.0, 1.0], 1.0),
            (64, 0, [1.0, 0.0, 0.0], [1.0, 2.0, 0.0], 3.0),
            (2, 1, [0.0, -1.0, 0.0], [1.0, 4.0, 0.0], 4.0),
        ] {
            let arm = rotate(arm);
            let velocity = rotate(velocity);
            for axis in 0..3 {
                queries[slot * 12 + axis] = (centers[body * 4 + axis] + arm[axis]).to_bits();
                velocities[slot * 4 + axis] = velocity[axis] + boost[axis];
            }
            queries[slot * 12 + 3] = 1f32.to_bits();
            queries[slot * 12 + 8] = body as u32;
            normals[slot * 4 + 1] = 1.0;
            impulses[slot * 4 + 3] = weight;
        }
        let mut jacobians = vec![0f32; 3 * 36];
        for body in 0..3 {
            for block in 0..2 {
                for column in 0..3 {
                    let mut unit = [0.0; 3];
                    unit[column] = 1.0;
                    let unit = rotate(unit);
                    for row in 0..3 {
                        jacobians[body * 36 + (block * 3 + row) * 6 + block * 3 + column] =
                            unit[row];
                    }
                }
            }
        }
        let rates = [3.0 + boost[0], 4.0 + boost[1], boost[2], 0.0, 0.0, -2.0];
        let queue = engine.queue().unwrap();
        queue.write_buffer(&inputs[0], 0, bytemuck::cast_slice(&queries));
        for (index, values) in [
            (1, normals.as_slice()),
            (2, velocities.as_slice()),
            (3, impulses.as_slice()),
            (4, jacobians.as_slice()),
            (5, rates.as_slice()),
            (6, centers.as_slice()),
        ] {
            queue.write_buffer(&inputs[index], 0, bytemuck::cast_slice(values));
        }
        let sample = run(&engine, &reader).unwrap();
        assert_eq!(sample[0].supporting_nodes, 3);
        assert_eq!(sample[0].normal_impulse, 6.0);
        let expected_slip = rotate([5.5 / 6.0, 0.0, -1.0 / 6.0]);
        let expected_soil = rotate([0.75, 14.0 / 6.0, 1.0 / 6.0]);
        let expected_speed = (7.0 + 3.25f32.sqrt()) / 6.0;
        assert!((sample[0].mean_tangential_slip_speed.unwrap() - expected_speed).abs() < 1e-6);
        assert!(sample[1].mean_tangential_slip_speed.unwrap() < 1e-6);
        for axis in 0..3 {
            assert!(
                (sample[0].tangential_slip_velocity.unwrap()[axis] - expected_slip[axis]).abs()
                    < 1e-6
            );
            assert!(
                (sample[0].soil_velocity.unwrap()[axis] - expected_soil[axis] - boost[axis]).abs()
                    < 1e-6
            );
            assert!(sample[1].tangential_slip_velocity.unwrap()[axis].abs() < 1e-6);
        }
        assert_eq!(sample[2].supporting_nodes, 0);
        assert!(sample[2].soil_velocity.is_none());
        assert!(sample[2].mean_tangential_slip_speed.is_none());
        queue.write_buffer(&inputs[3], 0, bytemuck::cast_slice(&vec![0.0f32; 65 * 4]));
        assert!(
            run(&engine, &reader)
                .unwrap()
                .iter()
                .all(|v| v.tangential_slip_velocity.is_none())
        );
        queue.write_buffer(&inputs[3], 0, bytemuck::cast_slice(&impulses));
        queue.write_buffer(&inputs[7], 0, bytemuck::cast_slice(&[2u32]));
        assert!(run(&engine, &reader).is_err());
        queue.write_buffer(&inputs[7], 0, bytemuck::cast_slice(&[0u32]));
        impulses[3] = f32::NAN;
        queue.write_buffer(&inputs[3], 0, bytemuck::cast_slice(&impulses));
        assert!(run(&engine, &reader).is_err());
    }
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(ContactMotionGpu::new(&engine, inputs.each_ref(), u32::MAX, 3, 65, 304).is_err());
    assert!(ContactMotionGpu::new(&engine, inputs.each_ref(), 6, 3, 65, 303).is_err());
    assert_eq!(allocations.finish().observations, 0);
}

#[test]
#[ignore = "requires GPU"]
fn opposed_slip_retains_speed_when_signed_mean_cancels() {
    let engine = Engine::new_gpu().unwrap();
    let inputs =
        [65 * 12, 65 * 4, 65 * 4, 65 * 4, 6, 1, 4, 4].map(|size| buffer(&engine, &vec![0; size]));
    let reader = ContactMotionGpu::new(&engine, inputs.each_ref(), 1, 1, 65, 112).unwrap();
    let mut queries = vec![0u32; 65 * 12];
    let mut geometry = vec![0f32; 65 * 4];
    let mut velocities = vec![0f32; 65 * 4];
    let mut impulses = vec![0f32; 65 * 4];
    for (slot, velocity) in [(0, 2.0), (64, -2.0)] {
        queries[slot * 12 + 3] = 1f32.to_bits();
        queries[slot * 12 + 7] = 1f32.to_bits();
        geometry[slot * 4 + 1] = 1.0;
        velocities[slot * 4] = velocity;
        impulses[slot * 4 + 3] = 2.0;
    }
    let queue = engine.queue().unwrap();
    queue.write_buffer(&inputs[0], 0, bytemuck::cast_slice(&queries));
    for (index, values) in [(1, &geometry), (2, &velocities), (3, &impulses)] {
        queue.write_buffer(&inputs[index], 0, bytemuck::cast_slice(values));
    }
    let sample = run(&engine, &reader).unwrap()[0];
    assert_eq!(sample.supporting_nodes, 2);
    assert_eq!(sample.normal_impulse, 4.0);
    assert_eq!(sample.tangential_slip_velocity, Some([0.0; 3]));
    assert_eq!(sample.mean_tangential_slip_speed, Some(2.0));
    queue.write_buffer(&inputs[3], 0, bytemuck::cast_slice(&vec![0.0f32; 65 * 4]));
    let empty = run(&engine, &reader).unwrap()[0];
    assert_eq!(empty.supporting_nodes, 0);
    assert!(empty.mean_tangential_slip_speed.is_none());
}
