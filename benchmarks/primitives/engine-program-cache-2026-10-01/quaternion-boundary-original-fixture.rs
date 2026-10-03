#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_kernels::{GpuQuaternionPair, QuaternionFkBindings, QuaternionFkKernel};
use molla_math::{GpuTransform, Quat, Transform, Vec3};
use molla_sim::{BodyFlags, BodyParams, Model, ModelBuilder};

fn fixture() -> (Model, Vec<f32>, Vec<f32>, Vec<GpuTransform>, Vec<f64>) {
    let mut builder = ModelBuilder::new();
    let mut angles = Vec::new();
    for angle in [1e-7, -0.013227509628473941, 0.4, -0.49, 2.8, 1e6 + 0.0215] {
        for boundary in [0, BodyFlags::FIXED, BodyFlags::KINEMATIC] {
            builder.begin_articulation();
            let mut parent = BodyId::NONE;
            if boundary != 0 {
                let body = builder.add_body(BodyParams {
                    flags: boundary,
                    ..BodyParams::new()
                });
                builder
                    .add_joint_fixed(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
                    .unwrap();
                parent = body;
                angles.push(angle);
            }
            let root = parent;
            for kind in 0..8 {
                let child = builder.add_body(BodyParams::new());
                let xp = Transform::new(
                    Vec3::new(0.2, -0.1, 0.3),
                    Quat::from_rotation_z(0.7) * Quat::from_rotation_y(-0.4),
                );
                let xc = Transform::new(
                    Vec3::new(-0.3, 0.1, 0.2),
                    Quat::from_rotation_y(0.3) * Quat::from_rotation_x(-0.2),
                );
                let axis = Vec3::new(1.0, 2.0, -3.0).normalize();
                let result = match kind {
                    0 => builder.add_joint_prismatic(parent, child, xp, xc, axis),
                    1 => builder.add_joint_revolute(parent, child, xp, xc, axis),
                    2 => builder.add_joint_ball(parent, child, xp, xc),
                    3 => builder.add_joint_fixed(parent, child, xp, xc),
                    4 => builder.add_joint_free(parent, child, xp, xc),
                    5 => builder.add_joint_distance(parent, child, xp, xc),
                    6 => builder.add_joint_d6(parent, child, xp, xc),
                    _ => builder.add_joint_cable(parent, child, xp, xc, 0.0, 0.0, 0.0, 0.0),
                };
                result.unwrap();
                angles.push(angle);
                parent = if kind == 4 { root } else { child };
            }
        }
    }
    let model = builder.build().unwrap();
    let offsets = model.joint_coord_offset.host().unwrap();
    let mut coordinates = vec![0.0; model.joint_coord_total];
    let mut residues = coordinates.clone();
    for (i, &kind) in model.joint_type.host().unwrap().iter().enumerate() {
        let offset = offsets[i] as usize;
        if matches!(kind, 0 | 1 | 7) {
            let offset = offset + usize::from(kind == 7);
            coordinates[offset] = angles[i] as f32;
            residues[offset] = (angles[i] - f64::from(coordinates[offset])) as f32;
        } else if kind == 2 || (4..=6).contains(&kind) {
            let offset = offset + if kind == 2 { 0 } else { 3 };
            let quaternion = Quat::from_scaled_axis(Vec3::new(0.1, -0.2, 0.3));
            for (lane, value) in quaternion.to_array().into_iter().enumerate() {
                coordinates[offset + lane] = (value * 1.01) as f32;
                residues[offset + lane] = 0.25;
            }
        }
    }
    let poses = (0..model.body_count)
        .map(|i| {
            GpuTransform::from(Transform::new(
                Vec3::new(i as f64, -0.4, 0.2),
                Quat::from_rotation_x(0.23) * Quat::from_rotation_y(-0.17),
            ))
        })
        .collect();
    (model, coordinates, residues, poses, angles)
}

fn oracle(
    model: &Model,
    coordinates: &[f32],
    residues: &[f32],
    poses: &[GpuTransform],
) -> Vec<Quat> {
    let mut rotations: Vec<_> = poses
        .iter()
        .map(|pose| Quat::from_array(pose.rotation.map(f64::from)))
        .collect();
    let offsets = model.joint_coord_offset.host().unwrap();
    let kinds = model.joint_type.host().unwrap();
    let parents = model.joint_parent.host().unwrap();
    let children = model.joint_child.host().unwrap();
    let axes = model.joint_axis.host().unwrap();
    let xp = model.joint_x_p.host().unwrap();
    let xc = model.joint_x_c.host().unwrap();
    let flags = model.body_flags.host().unwrap();
    for i in 0..model.joint_count {
        let child = children[i] as usize;
        if flags[child] & BodyFlags::IMMOVABLE != 0 {
            continue;
        }
        let kind = kinds[i];
        let offset = offsets[i] as usize;
        let local = if matches!(kind, 1 | 7) {
            let offset = offset + usize::from(kind == 7);
            let half = 0.5 * (f64::from(coordinates[offset]) + f64::from(residues[offset]));
            let axis = if kind == 7 { Vec3::X } else { axes[i] };
            Quat::from_xyzw(
                axis.x * half.sin(),
                axis.y * half.sin(),
                axis.z * half.sin(),
                half.cos(),
            )
        } else if kind == 2 || (4..=6).contains(&kind) {
            let offset = offset + if kind == 2 { 0 } else { 3 };
            Quat::from_array(std::array::from_fn(|lane| {
                f64::from(coordinates[offset + lane])
            }))
            .normalize()
        } else {
            Quat::IDENTITY
        };
        let parent = if parents[i] < 0 {
            Quat::IDENTITY
        } else {
            rotations[parents[i] as usize]
        };
        rotations[child] =
            (parent * xp[i].rotation * local * xc[i].rotation.conjugate()).normalize();
    }
    rotations
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn retained_fk_pairs_match_all_joint_kinds_and_prescribed_boundaries() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, coordinates, residues, poses, angles) = fixture();
    assert_eq!(model.articulation_count, 18);
    assert_eq!(model.joint_count, 156);
    let expected = oracle(&model, &coordinates, &residues, &poses);
    let storage = |bytes: &[u8], minimum: u64| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("quaternion FK test"),
            size: (bytes.len() as u64).max(minimum),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            queue.write_buffer(&buffer, 0, bytes);
        }
        buffer
    };
    let q = storage(bytemuck::cast_slice(&coordinates), 4);
    let low = storage(bytemuck::cast_slice(&residues), 4);
    let poses_buffer = storage(bytemuck::cast_slice(&poses), 32);
    let bytes = model.body_count as u64 * 32;
    let rotations = storage(&[], bytes);
    let rounding = storage(&[], model.body_count as u64 * 4);
    let status = storage(&[0; 4], 4);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("quaternion FK readback"),
        size: bytes + 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let kernel = QuaternionFkKernel::new(&engine, &model).unwrap();
    let bindings = |coordinates, coordinate_residues| QuaternionFkBindings {
        coordinates,
        coordinate_residues,
        poses: &poses_buffer,
        rotations: &rotations,
        rounding: &rounding,
        status: &status,
    };
    assert!(kernel.prepare(device, bindings(None, Some(&low))).is_err());
    let tiny = storage(&[0; 4], 4);
    assert!(
        kernel
            .prepare(device, bindings(Some(&tiny), Some(&low)))
            .is_err()
    );
    let dispatch = kernel
        .prepare(device, bindings(Some(&q), Some(&low)))
        .unwrap();
    let read = || {
        let mut encoder = device.create_command_encoder(&Default::default());
        dispatch.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&rotations, 0, &staging, 0, bytes);
        encoder.copy_buffer_to_buffer(&status, 0, &staging, bytes, 4);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let output = staging.slice(..).get_mapped_range().to_vec();
        staging.unmap();
        output
    };
    let mut previous = Vec::new();
    let mut maximum = [0.0_f64; 2];
    for _ in 0..3 {
        let output = read();
        assert_eq!(&output[bytes as usize..], &[0; 4]);
        if !previous.is_empty() {
            assert_eq!(output, previous);
        }
        let pairs: &[GpuQuaternionPair] = bytemuck::cast_slice(&output[..bytes as usize]);
        for (i, actual) in pairs.iter().enumerate() {
            if model.body_flags.host().unwrap()[i] & BodyFlags::IMMOVABLE == 0 {
                let extended = Quat::from_array(std::array::from_fn(|lane| {
                    f64::from(actual.high[lane]) + f64::from(actual.low[lane])
                }));
                assert!(
                    (extended.length_squared() - 1.0).abs() < 1e-12,
                    "body={i} actual={actual:?} norm={}",
                    extended.length_squared()
                );
            }
            let precise = angles[i].abs() <= 0.49;
            let tolerance = if precise { 2e-9 } else { 3e-7 };
            for (lane, reference) in expected[i].to_array().into_iter().enumerate() {
                let error =
                    (f64::from(actual.high[lane]) + f64::from(actual.low[lane]) - reference).abs();
                maximum[usize::from(!precise)] = maximum[usize::from(!precise)].max(error);
                assert!(
                    error <= tolerance,
                    "body={i} lane={lane} error={error} expected={reference} actual={actual:?}"
                );
            }
        }
        previous = output;
    }
    eprintln!(
        "quaternion FK {} joints / {} articulations max error: {maximum:?}",
        model.joint_count, model.articulation_count
    );
    let revolute = model
        .joint_type
        .host()
        .unwrap()
        .iter()
        .position(|&kind| kind == 1)
        .unwrap();
    let offset = model.joint_coord_offset.host().unwrap()[revolute] as u64 * 4;
    queue.write_buffer(&q, offset, &f32::NAN.to_ne_bytes());
    assert_eq!(&read()[bytes as usize..], &(1u32 << 15).to_ne_bytes());
    queue.write_buffer(&q, offset, &coordinates[offset as usize / 4].to_ne_bytes());
    assert_eq!(&read()[bytes as usize..], &(1u32 << 15).to_ne_bytes());
    queue.write_buffer(&status, 0, &[0; 4]);
    let ball = model
        .joint_type
        .host()
        .unwrap()
        .iter()
        .position(|&kind| kind == 2)
        .unwrap();
    let ball_offset = u64::from(model.joint_coord_offset.host().unwrap()[ball]) * 4;
    queue.write_buffer(&q, ball_offset, &[0; 16]);
    assert_eq!(&read()[bytes as usize..], &(1u32 << 15).to_ne_bytes());
    model.joint_type.host_mut().unwrap()[0] = 8;
    assert!(QuaternionFkKernel::new(&engine, &model).is_err());
    model.joint_type.host_mut().unwrap()[0] = 0;
    let child = model.joint_child.host().unwrap()[0];
    model.joint_child.host_mut().unwrap()[0] = -1;
    assert!(QuaternionFkKernel::new(&engine, &model).is_err());
    model.joint_child.host_mut().unwrap()[0] = child;
    model.joint_parent.host_mut().unwrap()[0] = child;
    assert!(QuaternionFkKernel::new(&engine, &model).is_err());
    model.joint_parent.host_mut().unwrap()[0] = model.joint_child.host().unwrap()[1];
    assert!(QuaternionFkKernel::new(&engine, &model).is_err());
    model.joint_parent.host_mut().unwrap()[0] = -1;
    let axis = model.joint_axis.host().unwrap()[0];
    model.joint_axis.host_mut().unwrap()[0] = Vec3::ZERO;
    assert!(QuaternionFkKernel::new(&engine, &model).is_err());
    model.joint_axis.host_mut().unwrap()[0] = axis;
    let rotation = model.joint_x_p.host().unwrap()[0].rotation;
    model.joint_x_p.host_mut().unwrap()[0].rotation = Quat::from_xyzw(0.0, 0.0, 0.0, 0.0);
    assert!(QuaternionFkKernel::new(&engine, &model).is_err());
    model.joint_x_p.host_mut().unwrap()[0].rotation = rotation;
    let offset = model.joint_coord_offset.host().unwrap()[1];
    model.joint_coord_offset.host_mut().unwrap()[1] = 0;
    assert!(QuaternionFkKernel::new(&engine, &model).is_err());
    model.joint_coord_offset.host_mut().unwrap()[1] = offset;
    model.articulation_start.host_mut().unwrap()[1] = u32::MAX;
    assert!(QuaternionFkKernel::new(&engine, &model).is_err());
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn unjointed_and_fixed_only_models_use_current_pose_without_coordinate_storage() {
    let engine = Engine::new_gpu().unwrap();
    for fixed_joint in [false, true] {
        let mut builder = ModelBuilder::new();
        let child = builder.add_body(BodyParams {
            flags: BodyFlags::KINEMATIC,
            ..BodyParams::new()
        });
        if fixed_joint {
            builder
                .add_joint_fixed(
                    BodyId::NONE,
                    child,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                )
                .unwrap();
        }
        let model = builder.build().unwrap();
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let buffer = |size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("fixed quaternion FK"),
                size,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let poses = buffer(32);
        let rotations = buffer(32);
        let rounding = buffer(4);
        let status = buffer(4);
        let pose = GpuTransform::from(Transform::new(Vec3::ZERO, Quat::from_rotation_z(0.4)));
        queue.write_buffer(&poses, 0, bytemuck::bytes_of(&pose));
        let dispatch = QuaternionFkKernel::new(&engine, &model)
            .unwrap()
            .prepare(
                device,
                QuaternionFkBindings {
                    coordinates: None,
                    coordinate_residues: None,
                    poses: &poses,
                    rotations: &rotations,
                    rounding: &rounding,
                    status: &status,
                },
            )
            .unwrap();
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 36,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        dispatch.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&rotations, 0, &staging, 0, 32);
        encoder.copy_buffer_to_buffer(&status, 0, &staging, 32, 4);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = staging.slice(..).get_mapped_range();
        assert_eq!(&mapped[..16], bytemuck::cast_slice(&pose.rotation));
        assert_eq!(&mapped[16..], &[0; 20]);
    }
}
