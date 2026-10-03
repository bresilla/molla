#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_geometry::{ResidentShapeBounds, shape_world_aabb};
use molla_math::{GpuTransform, Quat, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder, ShapeGeometry};

fn storage(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shape bounds test storage"),
        size: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn staging(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("shape bounds test readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}
fn read(device: &wgpu::Device, buffer: &wgpu::Buffer) -> Vec<u32> {
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = buffer.slice(..).get_mapped_range();
    let result = bytemuck::cast_slice(&mapped).to_vec();
    drop(mapped);
    buffer.unmap();
    result
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn device_bounds_enclose_cpu_bounds_across_shapes_frames_and_scales() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let scales: Vec<_> = [1e-30, 1e-4, 1.0, 1e30].repeat(3);
    for &s in &scales {
        let body = builder.add_body(BodyParams::new());
        let local = Transform {
            position: Vec3::new(0.2, -0.3, 0.1) * s,
            rotation: Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7),
        };
        builder.add_shape_sphere(body, s, local).unwrap();
        builder
            .add_shape_box(body, Vec3::new(1.0, 2.0, 0.7) * s, local)
            .unwrap();
        builder
            .add_shape_capsule(body, 1.3 * s, 0.4 * s, local)
            .unwrap();
        builder
            .add_shape_cylinder(body, 1.3 * s, 0.4 * s, local)
            .unwrap();
        builder
            .add_shape_round_cylinder(body, 1.3 * s, 0.4 * s, 0.2 * s, local)
            .unwrap();
        builder
            .add_shape_cone(body, 1.3 * s, 0.4 * s, local)
            .unwrap();
        builder
            .add_shape_ellipsoid(body, Vec3::new(1.0, 2.0, 0.7) * s, local)
            .unwrap();
        builder
            .add_shape_triangle(
                body,
                Vec3::new(-1.0, 0.0, 0.5) * s,
                Vec3::new(1.0, 0.0, 0.0) * s,
                Vec3::new(0.0, 2.0, 0.0) * s,
                local,
            )
            .unwrap();
        builder
            .add_shape_plane(BodyId::NONE, Vec3::Y, local)
            .unwrap();
        builder
            .add_shape_sphere(BodyId::NONE, 0.5 * s, local)
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut state = model.state().unwrap();
    let pose_bytes = model.body_count as u64 * 32;
    let bounds_bytes = model.shape_count as u64 * 32;
    let pose = storage(device, pose_bytes);
    let output = storage(device, bounds_bytes);
    let status = storage(device, 4);
    let bounds = ResidentShapeBounds::new(device, &model, &[&pose], &output, &status, 0.0).unwrap();
    let mut data = Vec::new();
    let mut reference = Vec::new();
    for frame in 0..7 {
        for (i, &s) in scales.iter().enumerate() {
            let p = Transform {
                position: Vec3::new(frame as f64 - 2.0, 3.0, -1.0) * s,
                rotation: Quat::from_axis_angle(
                    Vec3::new(3.0, -1.0, 2.0).normalize(),
                    frame as f64 * 0.4,
                ),
            };
            state.body_q.host_mut().unwrap()[i] = p;
            data.push(GpuTransform::from(p));
        }
        for shape in 0..model.shape_count {
            reference.push(shape_world_aabb(&model, &state, shape, 0.0));
        }
    }
    let input = storage(device, data.len() as u64 * 32);
    queue.write_buffer(&input, 0, bytemuck::cast_slice(&data));
    let snapshots = staging(device, bounds_bytes * 7 + 4);
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(bounds.encode(&mut encoder, 1).is_err());
    for frame in 0..7 {
        encoder.copy_buffer_to_buffer(&input, frame * pose_bytes, &pose, 0, pose_bytes);
        bounds.encode(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(&output, 0, &snapshots, frame * bounds_bytes, bounds_bytes);
    }
    encoder.copy_buffer_to_buffer(&status, 0, &snapshots, bounds_bytes * 7, 4);
    queue.submit([encoder.finish()]);
    let words = read(device, &snapshots);
    assert_eq!(*words.last().unwrap(), 0);
    for (i, (gpu, cpu)) in words[..words.len() - 1]
        .chunks_exact(8)
        .zip(reference)
        .enumerate()
    {
        let shape = i % model.shape_count;
        let scale = scales[shape / 10];
        for axis in 0..3 {
            let lo = f32::from_bits(gpu[axis]) as f64;
            let hi = f32::from_bits(gpu[axis + 4]) as f64;
            assert!(
                lo <= cpu.min[axis] && hi >= cpu.max[axis],
                "shape {shape}, frame {}, axis {axis}: [{lo},{hi}] vs [{},{}]",
                i / model.shape_count,
                cpu.min[axis],
                cpu.max[axis]
            );
            if !matches!(
                model.shape_geometry[shape],
                ShapeGeometry::Triangle { .. } | ShapeGeometry::Plane { .. }
            ) {
                assert!((lo - cpu.min[axis]).abs() < 1e-4 * scale);
                assert!((hi - cpu.max[axis]).abs() < 1e-4 * scale);
            }
        }
    }
    assert_eq!(bounds.resource_counts(), [2, 2, 1]);
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn bounds_padding_handles_large_translation_cancellation_and_margin() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_position(Vec3::new(1e8 + 0.2, 0.0, 0.0)));
    builder
        .add_shape_box(
            body,
            Vec3::splat(0.1),
            Transform::from_translation(Vec3::new(-1e8, 0.0, 0.0)),
        )
        .unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let state = model.state().unwrap();
    let pose = storage(device, 32);
    queue.write_buffer(
        &pose,
        0,
        bytemuck::bytes_of(&GpuTransform::from(state.body_q.host().unwrap()[0])),
    );
    let output = storage(device, 32);
    let status = storage(device, 4);
    let bounds =
        ResidentShapeBounds::new(device, &model, &[&pose], &output, &status, 0.03).unwrap();
    let snapshots = staging(device, 36);
    let mut encoder = device.create_command_encoder(&Default::default());
    bounds.encode(&mut encoder, 0).unwrap();
    encoder.copy_buffer_to_buffer(&output, 0, &snapshots, 0, 32);
    encoder.copy_buffer_to_buffer(&status, 0, &snapshots, 32, 4);
    queue.submit([encoder.finish()]);
    let words = read(device, &snapshots);
    assert_eq!(words[8], 0);
    let cpu = shape_world_aabb(&model, &state, 0, 0.03);
    for axis in 0..3 {
        assert!(f32::from_bits(words[axis]) as f64 <= cpu.min[axis]);
        assert!(f32::from_bits(words[axis + 4]) as f64 >= cpu.max[axis]);
    }
    for margin in [-1.0, f64::NAN, f64::INFINITY, 1e-50, 1e40] {
        assert!(
            ResidentShapeBounds::new(device, &model, &[&pose], &output, &status, margin).is_err()
        );
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn bounds_faults_preserve_sticky_status_and_clear_invalid_output() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    builder
        .add_shape_sphere(body, 1.0, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut bad_pose = GpuTransform::from(Transform::IDENTITY);
    bad_pose.position[0] = f32::NAN;
    let pose = storage(device, 32);
    queue.write_buffer(&pose, 0, bytemuck::bytes_of(&bad_pose));
    let output = storage(device, 32);
    let status = storage(device, 4);
    let invalid_index = storage(device, 4);
    queue.write_buffer(&invalid_index, 0, bytemuck::bytes_of(&99_u32));
    let bounds = ResidentShapeBounds::new(device, &model, &[&pose], &output, &status, 0.0).unwrap();
    let snapshot = staging(device, 72);
    let mut encoder = device.create_command_encoder(&Default::default());
    bounds.encode(&mut encoder, 0).unwrap();
    encoder.copy_buffer_to_buffer(&status, 0, &snapshot, 0, 4);
    encoder.copy_buffer_to_buffer(&output, 0, &snapshot, 4, 32);
    encoder.copy_buffer_to_buffer(
        &invalid_index,
        0,
        model.shape_body.device_buffer().unwrap(),
        0,
        4,
    );
    bounds.encode(&mut encoder, 0).unwrap();
    encoder.copy_buffer_to_buffer(&status, 0, &snapshot, 36, 4);
    encoder.copy_buffer_to_buffer(&output, 0, &snapshot, 40, 32);
    queue.submit([encoder.finish()]);
    let words = read(device, &snapshot);
    assert_eq!(words[0], 16);
    assert_eq!(words[9], 18);
    assert!(words[1..9].iter().chain(&words[10..18]).all(|&v| v == 0));
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn unbounded_plane_reaches_distant_shapes_through_device_refit() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    for x in [1e9, 2e9] {
        let body = builder.add_body(BodyParams::new().with_position(Vec3::new(x, 0.0, 0.0)));
        builder
            .add_shape_sphere(body, 1.0, Transform::IDENTITY)
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let state = model.state().unwrap();
    let poses: Vec<_> = state
        .body_q
        .host()
        .unwrap()
        .iter()
        .copied()
        .map(GpuTransform::from)
        .collect();
    let pose = storage(device, poses.len() as u64 * 32);
    queue.write_buffer(&pose, 0, bytemuck::cast_slice(&poses));
    let status = storage(device, 4);
    let mut lbvh =
        molla_geometry::LbvhBroadPhase::new(device.clone(), queue.clone(), 3, 64).unwrap();
    lbvh.dispatch_no_readback(&molla_geometry::collect_shape_aabbs(&model, &state, 0.0))
        .unwrap();
    let bounds = ResidentShapeBounds::new(
        device,
        &model,
        &[&pose],
        lbvh.primitive_aabb_buffer(),
        &status,
        0.0,
    )
    .unwrap();
    let snapshot = staging(device, 24);
    let mut encoder = device.create_command_encoder(&Default::default());
    bounds.encode(&mut encoder, 0).unwrap();
    lbvh.encode_refit(&mut encoder, 3).unwrap();
    let (count, a, b) = lbvh.pair_buffers();
    encoder.copy_buffer_to_buffer(&status, 0, &snapshot, 0, 4);
    encoder.copy_buffer_to_buffer(count, 0, &snapshot, 4, 4);
    encoder.copy_buffer_to_buffer(a, 0, &snapshot, 8, 8);
    encoder.copy_buffer_to_buffer(b, 0, &snapshot, 16, 8);
    queue.submit([encoder.finish()]);
    let words = read(device, &snapshot);
    assert_eq!(&words[..2], &[0, 2]);
    let mut pairs = [(words[2], words[4]), (words[3], words[5])];
    pairs.sort();
    assert_eq!(pairs, [(0, 1), (0, 2)]);
    assert_eq!(lbvh.diagnostics().pair_readbacks, 0);
}
