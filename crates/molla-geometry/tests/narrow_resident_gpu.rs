#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_geometry::{
    LbvhBroadPhase, NARROW_FAULT_CAPACITY, NARROW_FAULT_GJK, NARROW_FAULT_INDEX,
    NARROW_FAULT_NONFINITE, ResidentNarrowPhase,
};
use molla_math::{Transform, Vec3};
use molla_sim::{BodyParams, Model, ModelBuilder};

const MANIFOLD_BYTES: u64 = molla_geometry::RESIDENT_MANIFOLD_POINTS as u64 * 64;

fn fixture() -> Model {
    let mut builder = ModelBuilder::new();
    for x in [0.0, 1.5] {
        let body = builder.add_body(BodyParams::new().with_position(Vec3::new(x, 0.0, 0.0)));
        builder
            .add_shape_box(body, Vec3::ONE, Transform::IDENTITY)
            .unwrap();
    }
    builder.build().unwrap()
}

fn buffer(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u32]) -> wgpu::Buffer {
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resident narrow test input"),
        size: (data.len() * 4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buffer, 0, bytemuck::cast_slice(data));
    buffer
}

fn staging(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("resident narrow test readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn read(device: &wgpu::Device, output: &wgpu::Buffer) -> Vec<u32> {
    let (tx, rx) = std::sync::mpsc::channel();
    output
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let view = output.slice(..).get_mapped_range();
    let words = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    output.unmap();
    words
}

fn poses(x: f32) -> [f32; 16] {
    [
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, x, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    ]
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn capsule_manifolds_retain_support_span_under_scale_transform_and_exchange() {
    use molla_geometry::{ShapePair, pair_contact};
    use molla_math::{GpuTransform, Quat};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let mut pairs = Vec::new();
    let mut scales = Vec::new();
    let mut spans = Vec::new();
    for scale in [1e-3, 1.0, 1e3] {
        for transformed in [false, true] {
            let frame = if transformed {
                Transform::new(
                    Vec3::new(3.0, -2.0, 4.0) * scale,
                    Quat::from_rotation_x(0.4) * Quat::from_rotation_z(-0.7),
                )
            } else {
                Transform::IDENTITY
            };
            for (half, angle, shift, separation) in [
                (1.0, 0.0, 0.0, 0.15),
                (1.0, 0.0, 0.5, 0.15),
                (1.0, std::f64::consts::PI, 0.5, 0.15),
                (1.0, 0.01, 0.0, 0.15),
                (1.0, -0.01, 0.0, 0.15),
                (0.0, 0.0, 0.0, 0.15),
                (1.0, 0.0, 0.0, 0.21),
            ] {
                let local = Transform::new(
                    Vec3::new(0.13, -0.2, 0.17) * scale,
                    Quat::from_rotation_y(0.3),
                );
                let mut shapes = Vec::new();
                for pose in [
                    frame,
                    frame.compose(&Transform::new(
                        Vec3::new(separation, shift, 0.0) * scale,
                        Quat::from_rotation_z(angle),
                    )),
                ] {
                    let body_pose = pose.compose(&local.inverse());
                    let body = builder.add_body(BodyParams::new().with_pose(body_pose));
                    shapes.push(
                        builder
                            .add_shape_capsule(body, half * scale, 0.1 * scale, local)
                            .unwrap()
                            .0,
                    );
                }
                for exchange in [false, true] {
                    pairs.push(ShapePair {
                        a: shapes[usize::from(exchange)],
                        b: shapes[usize::from(!exchange)],
                    });
                    scales.push(scale);
                    spans.push(
                        (half == 1.0 && angle == 0.0 && separation == 0.15)
                            .then_some((frame.rotation * Vec3::Y, (2.0 - shift) * scale)),
                    );
                }
            }
        }
    }
    let mut model = builder.build().unwrap();
    let state = model.state().unwrap();
    let expected: Vec<_> = pairs
        .iter()
        .map(|&pair| {
            let mut contacts = Vec::new();
            pair_contact(&model, &state, pair, 0.0, 0.0, &mut contacts);
            contacts
        })
        .collect();
    let transforms: Vec<_> = state
        .body_q
        .host()
        .unwrap()
        .iter()
        .copied()
        .map(GpuTransform::from)
        .collect();
    model.upload_to_device(device, queue).unwrap();
    let pose = buffer(device, queue, bytemuck::cast_slice(&transforms));
    let a = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p.a).collect::<Vec<_>>(),
    );
    let b = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p.b).collect::<Vec<_>>(),
    );
    let count = buffer(device, queue, &[pairs.len() as u32]);
    let narrow =
        ResidentNarrowPhase::new(device, &model, &[&pose], &a, &b, &count, pairs.len() as u32)
            .unwrap();
    let row_bytes = 4 + MANIFOLD_BYTES;
    let frame_bytes = 4 + row_bytes * pairs.len() as u64;
    let output = staging(device, 2 * frame_bytes);
    let mut encoder = device.create_command_encoder(&Default::default());
    for frame in 0..2_u64 {
        if frame == 1 {
            encoder.clear_buffer(&count, 0, None);
        }
        narrow.encode(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, frame * frame_bytes, 4);
        for slot in 0..pairs.len() as u64 {
            let offset = frame * frame_bytes + 4 + slot * row_bytes;
            encoder.copy_buffer_to_buffer(
                narrow.manifold_count_buffer(),
                slot * 4,
                &output,
                offset,
                4,
            );
            encoder.copy_buffer_to_buffer(
                narrow.manifold_buffer(),
                slot * MANIFOLD_BYTES,
                &output,
                offset + 4,
                MANIFOLD_BYTES,
            );
        }
    }
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    for (frame, data) in words.chunks_exact(frame_bytes as usize / 4).enumerate() {
        assert_eq!(data[0], 0, "frame={frame}");
        for (slot, row) in data[1..].chunks_exact(row_bytes as usize / 4).enumerate() {
            if frame == 1 {
                assert!(row.iter().all(|&v| v == 0), "stale slot={slot}");
                continue;
            }
            let count = row[0] as usize;
            assert_eq!(
                count,
                expected[slot].len(),
                "slot={slot} scale={}",
                scales[slot]
            );
            let mut matched = vec![false; count];
            let mut projections = Vec::new();
            for (index, point) in row[1..].chunks_exact(16).enumerate() {
                if index >= count {
                    assert!(point.iter().all(|&v| v == 0));
                    continue;
                }
                assert_eq!(point[12], molla_geometry::EPA_STATUS_OK);
                let vector = |start: usize| {
                    Vec3::new(
                        f32::from_bits(point[start]) as f64,
                        f32::from_bits(point[start + 1]) as f64,
                        f32::from_bits(point[start + 2]) as f64,
                    )
                };
                let normal = vector(0);
                let depth = f32::from_bits(point[3]) as f64;
                let pa = vector(4);
                let pb = vector(8);
                assert!((normal.length() - 1.0).abs() < 2e-5);
                assert!(((pa - pb).dot(normal) - depth).abs() < 2e-5 * scales[slot]);
                let midpoint = (pa + pb) * 0.5;
                let found = expected[slot].iter().enumerate().position(|(j, contact)| {
                    !matched[j]
                        && (contact.point - midpoint).length() < 2e-5 * scales[slot]
                        && (contact.normal - normal).length() < 2e-5
                        && (contact.depth - depth).abs() < 2e-5 * scales[slot]
                });
                let found = found.unwrap_or_else(|| panic!("unmatched slot={slot} point={index}"));
                matched[found] = true;
                if let Some((axis, _)) = spans[slot] {
                    projections.push(midpoint.dot(axis));
                }
            }
            if let Some((_, span)) = spans[slot] {
                let min = projections.iter().copied().fold(f64::INFINITY, f64::min);
                let max = projections
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                assert!(
                    (max - min - span).abs() < 2e-5 * scales[slot],
                    "support span slot={slot}"
                );
            }
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn thin_near_parallel_capsules_preserve_interior_contact() {
    use molla_math::{GpuTransform, Quat};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let mut pairs = Vec::new();
    let mut references = Vec::new();
    for scale in [1e-3, 1.0, 1e3] {
        for angle in [-1e-3_f64, -1e-4, 1e-4, 1e-3] {
            let mut shapes = Vec::new();
            for pose in [
                Transform::IDENTITY,
                Transform::new(
                    Vec3::new(angle.sin() * 0.5, 0.3, 1e-5) * scale,
                    Quat::from_rotation_z(angle),
                ),
            ] {
                let body = builder.add_body(BodyParams::new().with_pose(pose));
                shapes.push(
                    builder
                        .add_shape_capsule(body, scale, 1e-5 * scale, Transform::IDENTITY)
                        .unwrap()
                        .0,
                );
            }
            for exchange in [false, true] {
                pairs.push([
                    shapes[usize::from(exchange)],
                    shapes[usize::from(!exchange)],
                ]);
                references.push((scale, angle, if exchange { -1.0 } else { 1.0 }));
            }
        }
    }
    let mut model = builder.build().unwrap();
    let state = model.state().unwrap();
    let poses: Vec<_> = state
        .body_q
        .host()
        .unwrap()
        .iter()
        .copied()
        .map(GpuTransform::from)
        .collect();
    model.upload_to_device(device, queue).unwrap();
    let pose = buffer(device, queue, bytemuck::cast_slice(&poses));
    let a = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p[0]).collect::<Vec<_>>(),
    );
    let b = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p[1]).collect::<Vec<_>>(),
    );
    let count = buffer(device, queue, &[pairs.len() as u32]);
    let narrow =
        ResidentNarrowPhase::new(device, &model, &[&pose], &a, &b, &count, pairs.len() as u32)
            .unwrap();
    let output = staging(device, 4 + pairs.len() as u64 * 68);
    let mut encoder = device.create_command_encoder(&Default::default());
    narrow.encode(&mut encoder, 0).unwrap();
    encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, 0, 4);
    for slot in 0..pairs.len() as u64 {
        encoder.copy_buffer_to_buffer(
            narrow.manifold_count_buffer(),
            slot * 4,
            &output,
            4 + slot * 68,
            4,
        );
        encoder.copy_buffer_to_buffer(
            narrow.contact_buffer(),
            slot * 64,
            &output,
            8 + slot * 68,
            64,
        );
    }
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    assert_eq!(words[0], 0);
    for (slot, (row, &(scale, angle, sign))) in
        words[1..].chunks_exact(17).zip(&references).enumerate()
    {
        assert_eq!(
            row[0], 1,
            "interior contact slot={slot} scale={scale} angle={angle}"
        );
        let point = &row[1..];
        assert_eq!(point[12], molla_geometry::EPA_STATUS_OK);
        let vector = |start: usize| {
            Vec3::new(
                f32::from_bits(point[start]) as f64,
                f32::from_bits(point[start + 1]) as f64,
                f32::from_bits(point[start + 2]) as f64,
            )
        };
        let normal = vector(0);
        let depth = f32::from_bits(point[3]) as f64;
        let pa = vector(4);
        let pb = vector(8);
        assert!(
            (normal - Vec3::Z * sign).length() < 1e-3,
            "slot={slot} normal={normal:?}"
        );
        assert!(
            (depth / (1e-5 * scale) - 1.0).abs() < 1e-3,
            "slot={slot} depth={depth}"
        );
        assert!(((pa - pb).dot(normal) - depth).abs() < 1e-8 * scale);
        let midpoint = Vec3::new(0.0, 0.3 + angle.cos() * 0.5, 0.5e-5) * scale;
        assert!(
            ((pa + pb) * 0.5 - midpoint).length() < 1e-3 * scale,
            "slot={slot} pa={pa:?} pb={pb:?}"
        );
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu P=molla-geometry T=narrow_resident_gpu"]
fn sphere_sphere_contacts_preserve_depth_symmetry_and_halted_status() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let mut pairs = Vec::new();
    let mut references = Vec::new();
    let mut lows = Vec::new();
    for scale in [1.0 / 1024.0, 1.0, 1024.0] {
        let offset = Vec3::new(1024.0, -2048.0, 4096.0) * scale;
        for direction in [Vec3::X, Vec3::new(0.3, -0.4, 0.5).normalize()] {
            for distance in [0.0, 0.1, 0.6, 0.75 - 1e-9, 0.75, 0.751] {
                if direction != Vec3::X && (distance == 0.75 || distance == 0.75 - 1e-9) {
                    continue;
                }
                let mut shapes = Vec::new();
                for (position, radius) in [
                    (offset, 0.5),
                    (offset + direction * (distance * scale), 0.25),
                ] {
                    lows.push(
                        [
                            position.x - position.x as f32 as f64,
                            position.y - position.y as f32 as f64,
                            position.z - position.z as f32 as f64,
                            0.0,
                        ]
                        .map(|v| v as f32),
                    );
                    let body = builder.add_body(BodyParams::new().with_position(position));
                    shapes.push(
                        builder
                            .add_shape_sphere(body, radius * scale, Transform::IDENTITY)
                            .unwrap()
                            .0,
                    );
                }
                for reversed in [false, true] {
                    pairs.push(if reversed {
                        [shapes[1], shapes[0]]
                    } else {
                        [shapes[0], shapes[1]]
                    });
                    let normal = if distance == 0.0 { Vec3::X } else { direction };
                    references.push((
                        normal * if reversed { -1.0 } else { 1.0 },
                        (0.75 - distance) * scale,
                        scale,
                    ));
                }
            }
        }
    }
    let mut model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let a = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p[0]).collect::<Vec<_>>(),
    );
    let b = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p[1]).collect::<Vec<_>>(),
    );
    let count = buffer(device, queue, &[pairs.len() as u32]);
    let low = buffer(device, queue, bytemuck::cast_slice(&lows));
    let narrow = ResidentNarrowPhase::with_collision_options(
        device,
        &model,
        &[state.body_q.device_buffer().unwrap()],
        &a,
        &b,
        &count,
        pairs.len() as u32,
        molla_geometry::ResidentCollisionOptions {
            plane_position_residues: &[&low],
            ..Default::default()
        },
    )
    .unwrap();
    let output = staging(device, 4 + pairs.len() as u64 * (4 + MANIFOLD_BYTES));
    let run = |status: u32| {
        queue.write_buffer(narrow.status_buffer(), 0, bytemuck::bytes_of(&status));
        let mut encoder = device.create_command_encoder(&Default::default());
        narrow.encode(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, 0, 4);
        encoder.copy_buffer_to_buffer(
            narrow.manifold_count_buffer(),
            0,
            &output,
            4,
            pairs.len() as u64 * 4,
        );
        encoder.copy_buffer_to_buffer(
            narrow.manifold_buffer(),
            0,
            &output,
            4 + pairs.len() as u64 * 4,
            pairs.len() as u64 * MANIFOLD_BYTES,
        );
        queue.submit([encoder.finish()]);
        read(device, &output)
    };
    let words = run(0);
    assert_eq!(words[0], 0);
    for (i, &(normal, depth, scale)) in references.iter().enumerate() {
        assert_eq!(
            words[1 + i],
            u32::from(depth > 0.0),
            "pair={i} depth={depth:e}"
        );
        if depth <= 0.0 {
            continue;
        }
        let base = 1 + pairs.len() + i * MANIFOLD_BYTES as usize / 4;
        let vector = |at| {
            Vec3::new(
                f32::from_bits(words[at]) as f64,
                f32::from_bits(words[at + 1]) as f64,
                f32::from_bits(words[at + 2]) as f64,
            )
        };
        let actual_depth = f32::from_bits(words[base + 3]) as f64;
        let tolerance = if depth < 1e-8 * scale {
            3e-13 * scale
        } else {
            2e-6 * scale
        };
        assert!((vector(base) - normal).length() < 2e-6, "pair={i}");
        assert!(
            (actual_depth - depth).abs() < tolerance,
            "pair={i} depth={actual_depth:e} expected={depth:e}"
        );
        if i % 2 == 1 {
            let previous = base - MANIFOLD_BYTES as usize / 4;
            assert_eq!(words[base + 3], words[previous + 3]);
            assert_eq!(vector(base), -vector(previous));
            assert_eq!(vector(base + 4), vector(previous + 8));
            assert_eq!(vector(base + 8), vector(previous + 4));
        }
    }
    for status in [1, 1 << 25, 1 << 30] {
        let words = run(status);
        assert_eq!(words[0], status);
        assert!(words[1..].iter().all(|&word| word == 0));
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu P=molla-geometry T=narrow_resident_gpu"]
fn sphere_box_analytic_contacts_cover_faces_edges_inside_and_low_translation_bits() {
    use molla_core::BodyId;
    use molla_math::Quat;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let mut pairs = Vec::new();
    let mut references = Vec::new();
    let mut lows = Vec::new();
    for scale in [1.0 / 1024.0, 1.0, 1024.0] {
        for rotation in [
            Quat::IDENTITY,
            Quat::from_rotation_z(0.37) * Quat::from_rotation_y(0.61),
        ] {
            let offset = Vec3::new(1024.0, -2048.0, 4096.0) * scale;
            let cube = builder
                .add_shape_box(
                    BodyId::NONE,
                    Vec3::splat(scale),
                    Transform {
                        position: offset,
                        rotation,
                    },
                )
                .unwrap();
            for local in [
                Vec3::new(0.0, 1.5 - 1e-9, 0.0),
                Vec3::new(0.0, 1.5 - 0.0008091666666666667, 0.0),
                Vec3::new(0.0, 1.5 - 0.0007841666666666667, 0.0),
                Vec3::new(0.0, 1.5, 0.0),
                Vec3::new(0.0, 1.51, 0.0),
                Vec3::new(0.0, 1.4, 0.0),
                Vec3::new(1.2, 1.2, 0.0),
                Vec3::splat(1.2),
                Vec3::new(0.2, 0.1, 0.3),
                Vec3::ZERO,
            ] {
                if (local.y > 1.499 && local.y <= 1.5) && rotation != Quat::IDENTITY {
                    continue;
                }
                let position = offset + rotation * (local * scale);
                lows.push(
                    [
                        position.x - position.x as f32 as f64,
                        position.y - position.y as f32 as f64,
                        position.z - position.z as f32 as f64,
                        0.0,
                    ]
                    .map(|v| v as f32),
                );
                let body = builder.add_body(BodyParams::new().with_position(position));
                let sphere = builder
                    .add_shape_sphere(body, 0.5 * scale, Transform::IDENTITY)
                    .unwrap();
                let closest = local.clamp(-Vec3::ONE, Vec3::ONE);
                let gap = local - closest;
                let (normal, depth) = if gap.length() > 0.0 {
                    (gap.normalize(), 0.5 - gap.length())
                } else {
                    let distances = Vec3::ONE - local.abs();
                    let axis = if distances.x <= distances.y && distances.x <= distances.z {
                        0
                    } else if distances.y <= distances.z {
                        1
                    } else {
                        2
                    };
                    let mut normal = Vec3::ZERO;
                    normal[axis] = if local[axis] < 0.0 { -1.0 } else { 1.0 };
                    (normal, 0.5 + distances[axis])
                };
                for reversed in [false, true] {
                    pairs.push(if reversed {
                        [cube.0, sphere.0]
                    } else {
                        [sphere.0, cube.0]
                    });
                    references.push((
                        rotation * normal * if reversed { 1.0 } else { -1.0 },
                        depth * scale,
                        scale,
                    ));
                }
            }
        }
    }
    let mut model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let a = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p[0]).collect::<Vec<_>>(),
    );
    let b = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p[1]).collect::<Vec<_>>(),
    );
    let count = buffer(device, queue, &[pairs.len() as u32]);
    let residue = buffer(device, queue, bytemuck::cast_slice(&lows));
    let narrow = ResidentNarrowPhase::with_collision_options(
        device,
        &model,
        &[state.body_q.device_buffer().unwrap()],
        &a,
        &b,
        &count,
        pairs.len() as u32,
        molla_geometry::ResidentCollisionOptions {
            plane_position_residues: &[&residue],
            ..Default::default()
        },
    )
    .unwrap();
    let size = 4 + pairs.len() as u64 * 4 + pairs.len() as u64 * MANIFOLD_BYTES;
    let output = staging(device, size);
    let mut encoder = device.create_command_encoder(&Default::default());
    narrow.encode_reset_status(&mut encoder);
    narrow.encode(&mut encoder, 0).unwrap();
    encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, 0, 4);
    encoder.copy_buffer_to_buffer(
        narrow.manifold_count_buffer(),
        0,
        &output,
        4,
        pairs.len() as u64 * 4,
    );
    encoder.copy_buffer_to_buffer(
        narrow.manifold_buffer(),
        0,
        &output,
        4 + pairs.len() as u64 * 4,
        pairs.len() as u64 * MANIFOLD_BYTES,
    );
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    assert_eq!(words[0], 0);
    for (i, &(normal, depth, scale)) in references.iter().enumerate() {
        assert_eq!(
            words[1 + i],
            u32::from(depth >= 0.0),
            "pair={i} depth={depth:e}"
        );
        if depth < 0.0 {
            continue;
        }
        let base = 1 + pairs.len() + i * MANIFOLD_BYTES as usize / 4;
        let got_normal = Vec3::new(
            f32::from_bits(words[base]) as f64,
            f32::from_bits(words[base + 1]) as f64,
            f32::from_bits(words[base + 2]) as f64,
        );
        let got_depth = f32::from_bits(words[base + 3]) as f64;
        assert!(
            (got_normal - normal).length() < 2e-6,
            "pair={i} normal={got_normal:?} expected={normal:?}"
        );
        let tolerance = if depth < 1e-8 * scale {
            3e-13 * scale
        } else if depth < 0.001 * scale {
            f32::EPSILON as f64 * depth.abs() + 3e-13 * scale
        } else {
            2e-6 * scale
        };
        assert!(
            (got_depth - depth).abs() < tolerance,
            "pair={i} depth={got_depth:e} expected={depth:e}"
        );
        if i % 2 == 1 {
            let previous = base - MANIFOLD_BYTES as usize / 4;
            assert_eq!(
                &words[base + 4..base + 7],
                &words[previous + 8..previous + 11]
            );
            assert_eq!(
                &words[base + 8..base + 11],
                &words[previous + 4..previous + 7]
            );
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_plane_depths_consume_sub_ulp_body_translation_residues() {
    use molla_geometry::ResidentCollisionOptions;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let mut pairs = Vec::new();
    let mut lows = Vec::new();
    let mut expected = Vec::new();
    for scale in [2.0_f64.powi(-20), 0.5, 1024.0] {
        let origin = 1024.0 * scale;
        let body = builder.add_body(BodyParams::new().with_position(Vec3::new(
            0.0,
            origin + 0.5 * scale,
            0.0,
        )));
        let plane_body =
            builder.add_body(BodyParams::new().with_position(Vec3::new(0.0, origin, 0.0)));
        let shape = builder
            .add_shape_box(body, Vec3::splat(0.5 * scale), Transform::IDENTITY)
            .unwrap();
        let plane = builder
            .add_shape_plane(plane_body, Vec3::Y, Transform::IDENTITY)
            .unwrap();
        let error = (scale * 1e-9) as f32;
        lows.extend([[0.0, -2.0 * error, 0.0, 0.0], [0.0, -error, 0.0, 0.0]]);
        pairs.extend([(shape.0, plane.0), (plane.0, shape.0)]);
        expected.extend([error as f64; 2]);
    }
    let mut model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let a = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p.0).collect::<Vec<_>>(),
    );
    let b = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p.1).collect::<Vec<_>>(),
    );
    let count = buffer(device, queue, &[pairs.len() as u32]);
    let low = buffer(device, queue, bytemuck::cast_slice(&lows));
    let narrow = ResidentNarrowPhase::with_collision_options(
        device,
        &model,
        &[state.body_q.device_buffer().unwrap()],
        &a,
        &b,
        &count,
        pairs.len() as u32,
        ResidentCollisionOptions {
            plane_position_residues: &[&low],
            ..Default::default()
        },
    )
    .unwrap();
    let stride = 4 + MANIFOLD_BYTES;
    let snapshot = staging(device, 4 + stride * pairs.len() as u64);
    let mut encoder = device.create_command_encoder(&Default::default());
    narrow.encode(&mut encoder, 0).unwrap();
    encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &snapshot, 0, 4);
    for i in 0..pairs.len() as u64 {
        encoder.copy_buffer_to_buffer(
            narrow.manifold_count_buffer(),
            4 * i,
            &snapshot,
            4 + stride * i,
            4,
        );
        encoder.copy_buffer_to_buffer(
            narrow.manifold_buffer(),
            MANIFOLD_BYTES * i,
            &snapshot,
            8 + stride * i,
            MANIFOLD_BYTES,
        );
    }
    queue.submit([encoder.finish()]);
    let result = read(device, &snapshot);
    assert_eq!(result[0], 0);
    for (i, expected) in expected.into_iter().enumerate() {
        let start = 1 + i * stride as usize / 4;
        assert_eq!(result[start], 4, "pair={i}");
        for point in 0..4 {
            let depth = f32::from_bits(result[start + 1 + 16 * point + 3]) as f64;
            assert!(
                (depth - expected).abs() <= expected * 2e-6,
                "pair={i} point={point} depth={depth:e} expected={expected:e}"
            );
        }
    }
    for residues in [vec![&low, &low], vec![&count]] {
        assert!(
            ResidentNarrowPhase::with_collision_options(
                device,
                &model,
                &[state.body_q.device_buffer().unwrap()],
                &a,
                &b,
                &count,
                pairs.len() as u32,
                ResidentCollisionOptions {
                    plane_position_residues: &residues,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
    queue.write_buffer(&low, 4, bytemuck::bytes_of(&f32::NAN));
    let mut encoder = device.create_command_encoder(&Default::default());
    narrow.encode(&mut encoder, 0).unwrap();
    encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &snapshot, 0, 4);
    encoder.copy_buffer_to_buffer(narrow.manifold_count_buffer(), 0, &snapshot, 4, 8);
    queue.submit([encoder.finish()]);
    let result = read(device, &snapshot);
    assert_eq!(result[0], NARROW_FAULT_NONFINITE);
    assert_eq!(&result[1..3], &[0, 0]);
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_plane_manifolds_match_cpu_with_transforms_and_endpoint_exchange() {
    use molla_core::BodyId;
    use molla_geometry::{RESIDENT_MANIFOLD_POINTS, ShapePair, pair_contact};
    use molla_math::{GpuTransform, Quat};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let mut pairs = Vec::new();
    let mut labels = Vec::new();
    for frame in 0..3 {
        let global = if frame == 0 {
            Transform::IDENTITY
        } else {
            Transform::new(
                Vec3::new(2.0, -3.0, 1.0),
                Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.63),
            )
        };
        let plane_body = if frame == 2 {
            builder.add_body(BodyParams::new().with_pose(global))
        } else {
            BodyId::NONE
        };
        let authored_normal = if frame == 2 { Vec3::X } else { Vec3::Y };
        let plane_local = if frame == 2 {
            Transform::new(Vec3::new(0.3, -0.4, 0.7), Quat::from_rotation_x(0.31))
        } else {
            global
        };
        let plane_world = if frame == 2 {
            global.compose(&plane_local)
        } else {
            global
        };
        let normal_frame = plane_world.compose(&Transform::from_rotation(Quat::from_rotation_arc(
            Vec3::Y,
            authored_normal,
        )));
        let plane = builder
            .add_shape_plane(plane_body, authored_normal, plane_local)
            .unwrap();
        for case in 0..6 {
            let height = [0.2, -2.0, 3.0, 0.2, -2.0, 0.2][case];
            let rotation = match case {
                3 | 4 => Quat::from_axis_angle(Vec3::new(2.0, 1.0, 3.0).normalize(), 0.81),
                5 => Quat::from_rotation_z(std::f64::consts::FRAC_PI_2),
                _ => Quat::IDENTITY,
            };
            for kind in 0..6 {
                let desired = normal_frame.compose(&Transform::new(Vec3::Y * height, rotation));
                let local = if frame == 0 {
                    Transform::IDENTITY
                } else {
                    Transform::new(Vec3::new(-0.1, 0.2, 0.3), Quat::from_rotation_z(0.21))
                };
                let body = builder
                    .add_body(BodyParams::new().with_pose(desired.compose(&local.inverse())));
                let shape = match kind {
                    0 => builder.add_shape_sphere(body, 0.5, local),
                    1 => builder.add_shape_box(body, Vec3::new(0.4, 0.5, 0.6), local),
                    2 => builder.add_shape_capsule(body, 0.7, 0.3, local),
                    3 => builder.add_shape_cylinder(body, 0.7, 0.3, local),
                    4 => builder.add_shape_cone(body, 0.7, 0.3, local),
                    _ => builder.add_shape_ellipsoid(body, Vec3::new(0.4, 0.5, 0.6), local),
                }
                .unwrap();
                for (a, b) in [(shape.0, plane.0), (plane.0, shape.0)] {
                    pairs.push(ShapePair { a, b });
                    labels.push(format!(
                        "frame={frame}, case={case}, kind={kind}, pair={a},{b}"
                    ));
                }
            }
        }
    }
    let mut model = builder.build().unwrap();
    let state = model.state().unwrap();
    let expected: Vec<_> = pairs
        .iter()
        .map(|&pair| {
            let mut contacts = Vec::new();
            pair_contact(&model, &state, pair, 0.0, 0.0, &mut contacts);
            contacts
        })
        .collect();
    assert!(expected.iter().any(|contacts| contacts.len() == 10));
    assert!(expected.iter().any(|contacts| contacts.len() == 5));
    assert!(expected.iter().any(|contacts| contacts.len() == 4));
    assert!(expected.iter().any(|contacts| contacts.len() == 12));
    assert!(expected.iter().any(|contacts| contacts.len() == 6));
    let transforms: Vec<_> = state
        .body_q
        .host()
        .unwrap()
        .iter()
        .copied()
        .map(GpuTransform::from)
        .collect();
    model.upload_to_device(device, queue).unwrap();
    let pose = buffer(device, queue, bytemuck::cast_slice(&transforms));
    let a = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p.a).collect::<Vec<_>>(),
    );
    let b = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p.b).collect::<Vec<_>>(),
    );
    let count = buffer(device, queue, &[pairs.len() as u32]);
    let narrow =
        ResidentNarrowPhase::new(device, &model, &[&pose], &a, &b, &count, pairs.len() as u32)
            .unwrap();
    let point_bytes = u64::from(RESIDENT_MANIFOLD_POINTS) * 64;
    let record_bytes = 4 + 64 + point_bytes;
    let frame_bytes = 4 + record_bytes * pairs.len() as u64;
    let output = staging(device, frame_bytes * 3);
    let counts = buffer(device, queue, &[pairs.len() as u32, 1, 0]);
    let mut encoder = device.create_command_encoder(&Default::default());
    narrow.encode_reset_status(&mut encoder);
    for step in 0..3_u64 {
        encoder.copy_buffer_to_buffer(&counts, step * 4, &count, 0, 4);
        narrow.encode(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, step * frame_bytes, 4);
        for i in 0..pairs.len() as u64 {
            let offset = step * frame_bytes + 4 + i * record_bytes;
            encoder.copy_buffer_to_buffer(
                narrow.manifold_count_buffer(),
                i * 4,
                &output,
                offset,
                4,
            );
            encoder.copy_buffer_to_buffer(narrow.contact_buffer(), i * 64, &output, offset + 4, 64);
            encoder.copy_buffer_to_buffer(
                narrow.manifold_buffer(),
                i * point_bytes,
                &output,
                offset + 68,
                point_bytes,
            );
        }
    }
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    for (step, frame) in words.chunks_exact(frame_bytes as usize / 4).enumerate() {
        assert_eq!(frame[0], 0, "step {step}");
        for (i, row) in frame[1..]
            .chunks_exact(record_bytes as usize / 4)
            .enumerate()
        {
            let contacts = if step == 0 || (step == 1 && i == 0) {
                &expected[i][..]
            } else {
                &[]
            };
            assert_eq!(row[0] as usize, contacts.len(), "{} step={step}", labels[i]);
            if contacts.is_empty() {
                assert!(row.iter().all(|&v| v == 0), "{}", labels[i]);
            } else {
                assert_eq!(&row[1..17], &row[17..33]);
            }
            let mut matched = vec![false; contacts.len()];
            for (j, point) in row[17..].chunks_exact(16).enumerate() {
                if j >= contacts.len() {
                    assert!(point.iter().all(|&v| v == 0));
                    continue;
                }
                assert_eq!(point[12], 1);
                let v = |lane| {
                    Vec3::new(
                        f32::from_bits(point[lane]) as f64,
                        f32::from_bits(point[lane + 1]) as f64,
                        f32::from_bits(point[lane + 2]) as f64,
                    )
                };
                let normal = v(0);
                let a = v(4);
                let b = v(8);
                let depth = f32::from_bits(point[3]) as f64;
                let midpoint = (a + b) * 0.5;
                let reference_index = contacts
                    .iter()
                    .enumerate()
                    .position(|(k, reference)| {
                        !matched[k] && (midpoint - reference.point).length() < 5e-6
                    })
                    .unwrap_or_else(|| {
                        panic!("{} point={j}: unmatched midpoint {midpoint:?}", labels[i])
                    });
                matched[reference_index] = true;
                let reference = &contacts[reference_index];
                assert!(
                    (normal - reference.normal).length() < 3e-6,
                    "{} point={j}",
                    labels[i]
                );
                assert!(
                    (depth - reference.depth).abs() < 5e-6,
                    "{} point={j}: {depth} vs {}",
                    labels[i],
                    reference.depth
                );
                assert!(
                    ((a + b) * 0.5 - reference.point).length() < 5e-6,
                    "{} point={j}: {:?} vs {:?}",
                    labels[i],
                    (a + b) * 0.5,
                    reference.point
                );
                assert!(
                    (a - b - normal * depth).length() < 5e-6,
                    "{} point={j}",
                    labels[i]
                );
                if step == 0 && i % 2 == 1 {
                    let previous_offset = 1 + (i - 1) * record_bytes as usize / 4 + 17 + j * 16;
                    let previous = &frame[previous_offset..previous_offset + 16];
                    assert_eq!(&point[4..7], &previous[8..11]);
                    assert_eq!(&point[8..11], &previous[4..7]);
                    assert_eq!(point[3], previous[3]);
                    for lane in 0..3 {
                        assert_eq!(f32::from_bits(point[lane]), -f32::from_bits(previous[lane]));
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_curved_plane_grazing_contacts_cover_unsampled_rim_arcs() {
    use molla_core::BodyId;
    use molla_geometry::{ShapePair, pair_contact};
    use molla_math::{GpuTransform, Quat};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let mut pairs = Vec::new();
    let mut expected = Vec::new();
    for case in 0..3 {
        let thin = case == 2;
        let half_height = if thin { 1e-6 } else { 0.7 };
        let radius = if thin { 1.0 } else { 0.3 };
        let normal = if thin {
            Vec3::new(1e-7, 1.0, 1e-7).normalize()
        } else {
            Vec3::ONE.normalize()
        };
        let penetration = if thin { 2e-8 } else { 0.01 };
        let extent = half_height * normal.y + radius * normal.x.hypot(normal.z);
        for scale in [1e-30, 1e-4, 1.0, 1e30] {
            let frame = if case == 1 {
                Transform::new(
                    Vec3::new(2.0, -3.0, 1.0) * scale,
                    Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.63),
                )
            } else {
                Transform::IDENTITY
            };
            for cone in [false, true] {
                let plane = builder
                    .add_shape_plane(BodyId::NONE, normal, frame)
                    .unwrap();
                let shape_pose = frame.compose(&Transform::from_translation(
                    normal * ((extent - penetration) * scale),
                ));
                let body = builder.add_body(BodyParams::new().with_pose(shape_pose));
                let shape = if cone {
                    builder.add_shape_cone(
                        body,
                        half_height * scale,
                        radius * scale,
                        Transform::IDENTITY,
                    )
                } else {
                    builder.add_shape_cylinder(
                        body,
                        half_height * scale,
                        radius * scale,
                        Transform::IDENTITY,
                    )
                }
                .unwrap();
                for reversed in [false, true] {
                    pairs.push(if reversed {
                        ShapePair {
                            a: plane.0,
                            b: shape.0,
                        }
                    } else {
                        ShapePair {
                            a: shape.0,
                            b: plane.0,
                        }
                    });
                    expected.push((
                        scale,
                        penetration,
                        frame.rotation * normal * if reversed { 1.0 } else { -1.0 },
                        if thin { 3e-12 } else { 3e-6 },
                    ));
                }
            }
        }
    }
    let mut model = builder.build().unwrap();
    let state = model.state().unwrap();
    for (i, &pair) in pairs.iter().enumerate() {
        let mut contacts = Vec::new();
        pair_contact(&model, &state, pair, 0.0, 0.0, &mut contacts);
        assert_eq!(contacts.len(), 1, "CPU pair={i}");
        assert!(
            (contacts[0].depth / expected[i].0 - expected[i].1).abs() < 1e-14,
            "CPU pair={i}"
        );
    }
    let transforms: Vec<_> = state
        .body_q
        .host()
        .unwrap()
        .iter()
        .copied()
        .map(GpuTransform::from)
        .collect();
    model.upload_to_device(device, queue).unwrap();
    let pose = buffer(device, queue, bytemuck::cast_slice(&transforms));
    let a = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p.a).collect::<Vec<_>>(),
    );
    let b = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p.b).collect::<Vec<_>>(),
    );
    let count = buffer(device, queue, &[pairs.len() as u32]);
    let narrow =
        ResidentNarrowPhase::new(device, &model, &[&pose], &a, &b, &count, pairs.len() as u32)
            .unwrap();
    let output = staging(device, 4 + pairs.len() as u64 * 68);
    let mut encoder = device.create_command_encoder(&Default::default());
    narrow.encode_reset_status(&mut encoder);
    narrow.encode(&mut encoder, 0).unwrap();
    encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, 0, 4);
    for i in 0..pairs.len() as u64 {
        encoder.copy_buffer_to_buffer(
            narrow.manifold_count_buffer(),
            i * 4,
            &output,
            4 + i * 68,
            4,
        );
        encoder.copy_buffer_to_buffer(
            narrow.manifold_buffer(),
            i * MANIFOLD_BYTES,
            &output,
            8 + i * 68,
            64,
        );
    }
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    assert_eq!(words[0], 0);
    for (i, row) in words[1..].chunks_exact(17).enumerate() {
        let (scale, depth, normal, tolerance) = expected[i];
        assert_eq!(row[0], 1, "GPU pair={i}");
        assert_eq!(row[13], 1);
        let measured = f32::from_bits(row[4]) as f64 / scale;
        assert!(
            (measured - depth).abs() < tolerance,
            "pair={i}: {measured} vs {depth}"
        );
        let n = Vec3::new(
            f32::from_bits(row[1]) as f64,
            f32::from_bits(row[2]) as f64,
            f32::from_bits(row[3]) as f64,
        );
        assert!((n - normal).length() < 1e-6, "pair={i}");
        if i % 2 == 1 {
            let previous = &words[1 + (i - 1) * 17..1 + i * 17];
            assert_eq!(&row[5..8], &previous[9..12]);
            assert_eq!(&row[9..12], &previous[5..8]);
            assert_eq!(row[4], previous[4]);
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_plane_manifolds_preserve_small_and_large_finite_scales() {
    use molla_core::BodyId;
    use molla_geometry::{ShapePair, pair_contact};
    use molla_math::{GpuTransform, Quat};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let mut unit_builder = ModelBuilder::new();
    let mut pairs = Vec::new();
    let mut scales = Vec::new();
    for scale in [1e-30, 1e-4, 1.0, 1e30] {
        for (target, s) in [(&mut builder, scale), (&mut unit_builder, 1.0)] {
            let global = Transform::new(
                Vec3::new(0.1, 0.2, -0.3) * s,
                Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.4),
            );
            target
                .add_shape_plane(BodyId::NONE, Vec3::Y, global)
                .unwrap();
            for kind in 0..6 {
                let pose = global.compose(&Transform::new(
                    Vec3::Y * (0.15 * s),
                    Quat::from_axis_angle(Vec3::new(3.0, 1.0, 2.0).normalize(), 0.7),
                ));
                let body = target.add_body(BodyParams::new().with_pose(pose));
                match kind {
                    0 => target.add_shape_sphere(body, 0.5 * s, Transform::IDENTITY),
                    1 => target.add_shape_box(
                        body,
                        Vec3::new(0.4, 0.5, 0.6) * s,
                        Transform::IDENTITY,
                    ),
                    2 => target.add_shape_capsule(body, 0.7 * s, 0.3 * s, Transform::IDENTITY),
                    3 => target.add_shape_cylinder(body, 0.7 * s, 0.3 * s, Transform::IDENTITY),
                    4 => target.add_shape_cone(body, 0.7 * s, 0.3 * s, Transform::IDENTITY),
                    _ => target.add_shape_ellipsoid(
                        body,
                        Vec3::new(0.4, 0.5, 0.6) * s,
                        Transform::IDENTITY,
                    ),
                }
                .unwrap();
            }
        }
        let plane = (scales.len() / 6 * 7) as u32;
        for offset in 1..7 {
            pairs.push(ShapePair {
                a: plane + offset,
                b: plane,
            });
            scales.push(scale);
        }
    }
    let mut model = builder.build().unwrap();
    let unit_model = unit_builder.build().unwrap();
    let unit_state = unit_model.state().unwrap();
    let expected: Vec<_> = pairs
        .iter()
        .map(|&pair| {
            let mut points = Vec::new();
            pair_contact(&unit_model, &unit_state, pair, 0.0, 0.0, &mut points);
            assert!(!points.is_empty());
            points
        })
        .collect();
    let transforms: Vec<_> = model
        .state()
        .unwrap()
        .body_q
        .host()
        .unwrap()
        .iter()
        .copied()
        .map(GpuTransform::from)
        .collect();
    model.upload_to_device(device, queue).unwrap();
    let pose = buffer(device, queue, bytemuck::cast_slice(&transforms));
    let a = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p.a).collect::<Vec<_>>(),
    );
    let b = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p.b).collect::<Vec<_>>(),
    );
    let count = buffer(device, queue, &[pairs.len() as u32]);
    let narrow =
        ResidentNarrowPhase::new(device, &model, &[&pose], &a, &b, &count, pairs.len() as u32)
            .unwrap();
    let stride = MANIFOLD_BYTES + 4;
    let output = staging(device, 4 + pairs.len() as u64 * stride);
    let mut encoder = device.create_command_encoder(&Default::default());
    narrow.encode_reset_status(&mut encoder);
    narrow.encode(&mut encoder, 0).unwrap();
    encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, 0, 4);
    for i in 0..pairs.len() as u64 {
        encoder.copy_buffer_to_buffer(
            narrow.manifold_count_buffer(),
            i * 4,
            &output,
            4 + i * stride,
            4,
        );
        encoder.copy_buffer_to_buffer(
            narrow.manifold_buffer(),
            i * MANIFOLD_BYTES,
            &output,
            8 + i * stride,
            MANIFOLD_BYTES,
        );
    }
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    assert_eq!(words[0], 0);
    for (i, row) in words[1..].chunks_exact(stride as usize / 4).enumerate() {
        assert_eq!(row[0] as usize, expected[i].len(), "pair {i}");
        for (j, point) in row[1..].chunks_exact(16).enumerate() {
            if j >= expected[i].len() {
                assert!(point.iter().all(|&v| v == 0));
                continue;
            }
            let values: Vec<_> = point
                .iter()
                .copied()
                .map(|v| f32::from_bits(v) as f64)
                .collect();
            let normal = Vec3::new(values[0], values[1], values[2]);
            let a = Vec3::new(values[4], values[5], values[6]) / scales[i];
            let b = Vec3::new(values[8], values[9], values[10]) / scales[i];
            let depth = values[3] / scales[i];
            let reference = &expected[i][j];
            assert_eq!(point[12], 1);
            assert!(
                (normal - reference.normal).length() < 3e-6,
                "pair={i} point={j}"
            );
            assert!(
                (depth - reference.depth).abs() < 3e-6,
                "pair={i} point={j}: {depth} vs {}",
                reference.depth
            );
            assert!(
                ((a + b) * 0.5 - reference.point).length() < 3e-6,
                "pair={i} point={j}"
            );
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_plane_depth_overflow_faults_without_publishing_points() {
    use molla_core::BodyId;
    use molla_math::GpuTransform;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::X, Transform::IDENTITY)
        .unwrap();
    let body = builder.add_body(BodyParams::new());
    builder
        .add_shape_sphere(body, 0.5, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let transforms = [
        Transform::IDENTITY,
        Transform::from_translation(Vec3::X * -2e38),
        Transform::from_translation(Vec3::X * 2e38),
    ]
    .map(GpuTransform::from);
    let source = buffer(device, queue, bytemuck::cast_slice(&transforms));
    let pose = buffer(device, queue, bytemuck::cast_slice(&transforms[..1]));
    let a = buffer(device, queue, &[1]);
    let b = buffer(device, queue, &[0]);
    let count = buffer(device, queue, &[1]);
    let narrow = ResidentNarrowPhase::new(device, &model, &[&pose], &a, &b, &count, 1).unwrap();
    let stride = MANIFOLD_BYTES + 72;
    let output = staging(device, 5 * stride);
    let mut encoder = device.create_command_encoder(&Default::default());
    narrow.encode_reset_status(&mut encoder);
    for step in 0..5_u64 {
        if step == 1 {
            encoder.copy_buffer_to_buffer(&source, 32, &pose, 0, 32);
            encoder.copy_buffer_to_buffer(
                &source,
                64,
                model.shape_transform.device_buffer().unwrap(),
                0,
                32,
            );
        }
        if step == 2 {
            encoder.clear_buffer(&count, 0, None);
        }
        if step == 3 {
            encoder.copy_buffer_to_buffer(&a, 0, &count, 0, 4);
            encoder.copy_buffer_to_buffer(&source, 0, &pose, 0, 32);
            encoder.copy_buffer_to_buffer(
                &source,
                0,
                model.shape_transform.device_buffer().unwrap(),
                0,
                32,
            );
        }
        if step == 4 {
            narrow.encode_reset_status(&mut encoder);
        }
        narrow.encode(&mut encoder, 0).unwrap();
        let base = step * stride;
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, base, 4);
        encoder.copy_buffer_to_buffer(narrow.manifold_count_buffer(), 0, &output, base + 4, 4);
        encoder.copy_buffer_to_buffer(narrow.contact_buffer(), 0, &output, base + 8, 64);
        encoder.copy_buffer_to_buffer(
            narrow.manifold_buffer(),
            0,
            &output,
            base + 72,
            MANIFOLD_BYTES,
        );
    }
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    for (step, frame) in words.chunks_exact(stride as usize / 4).enumerate() {
        assert_eq!(
            frame[0],
            if step == 0 || step == 4 {
                0
            } else {
                NARROW_FAULT_NONFINITE
            }
        );
        if (1..=3).contains(&step) {
            assert!(frame[1..].iter().all(|&v| v == 0));
        } else {
            assert_eq!(frame[1], 1);
            assert_eq!(&frame[2..18], &frame[18..34]);
            assert_eq!(f32::from_bits(frame[5]), 0.5);
            assert_eq!(frame[14], 1);
            assert!(frame[34..].iter().all(|&v| v == 0));
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_plane_manifold_chain_updates_and_retires_device_contacts() {
    use molla_core::BodyId;
    use molla_math::GpuTransform;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let body = builder.add_body(BodyParams::new().with_position(Vec3::Y * 0.25));
    builder
        .add_shape_box(body, Vec3::splat(0.5), Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let heights = [0.25, 0.5, 1.5, -2.0];
    let transforms: Vec<_> = heights
        .iter()
        .map(|&y| GpuTransform::from(Transform::from_translation(Vec3::Y * y)))
        .collect();
    let pose = buffer(device, queue, bytemuck::cast_slice(&transforms[..1]));
    let inputs = buffer(device, queue, bytemuck::cast_slice(&transforms));
    let mut lbvh = LbvhBroadPhase::new(device.clone(), queue.clone(), 2, 64).unwrap();
    lbvh.dispatch_no_readback(&[
        (Vec3::splat(f64::NEG_INFINITY), Vec3::splat(f64::INFINITY)),
        (-Vec3::ONE, Vec3::ONE),
    ])
    .unwrap();
    let (count, a, b) = lbvh.pair_buffers();
    let narrow = ResidentNarrowPhase::new(device, &model, &[&pose], a, b, count, 64).unwrap();
    let bounds = molla_geometry::ResidentShapeBounds::new(
        device,
        &model,
        &[&pose],
        lbvh.primitive_aabb_buffer(),
        narrow.status_buffer(),
        0.0,
    )
    .unwrap();
    let frame_bytes = 84 + MANIFOLD_BYTES;
    let output = staging(device, 100 * frame_bytes);
    let mut encoder = device.create_command_encoder(&Default::default());
    narrow.encode_reset_status(&mut encoder);
    for step in 0..100_u64 {
        encoder.copy_buffer_to_buffer(&inputs, (step % 4) * 32, &pose, 0, 32);
        bounds.encode(&mut encoder, 0).unwrap();
        lbvh.encode_refit(&mut encoder, 2).unwrap();
        narrow.encode(&mut encoder, 0).unwrap();
        let base = step * frame_bytes;
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, base, 4);
        encoder.copy_buffer_to_buffer(count, 0, &output, base + 4, 4);
        encoder.copy_buffer_to_buffer(a, 0, &output, base + 8, 4);
        encoder.copy_buffer_to_buffer(b, 0, &output, base + 12, 4);
        encoder.copy_buffer_to_buffer(narrow.manifold_count_buffer(), 0, &output, base + 16, 4);
        encoder.copy_buffer_to_buffer(narrow.contact_buffer(), 0, &output, base + 20, 64);
        encoder.copy_buffer_to_buffer(
            narrow.manifold_buffer(),
            0,
            &output,
            base + 84,
            MANIFOLD_BYTES,
        );
    }
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    for (step, frame) in words.chunks_exact(frame_bytes as usize / 4).enumerate() {
        assert_eq!(&frame[..2], &[0, 1], "step={step}");
        assert_eq!(frame[2] + frame[3], 1);
        let height = heights[step % 4];
        let expected_count = if height < 0.5 { 4 } else { 0 };
        assert_eq!(frame[4], expected_count, "step={step}");
        if expected_count == 0 {
            assert!(frame[5..].iter().all(|&v| v == 0));
            continue;
        }
        assert_eq!(&frame[5..21], &frame[21..37]);
        let mut corners = Vec::new();
        for (j, point) in frame[21..].chunks_exact(16).enumerate() {
            if j >= 4 {
                assert!(point.iter().all(|&v| v == 0));
                continue;
            }
            assert_eq!(point[12], 1);
            let values: Vec<_> = point.iter().copied().map(f32::from_bits).collect();
            let normal = if frame[2] == 0 { 1.0 } else { -1.0 };
            assert_eq!(values[1], normal);
            assert_eq!(values[3], (0.5 - height) as f32);
            assert_eq!((values[5] + values[9]) * 0.5, ((height - 0.5) * 0.5) as f32);
            assert_eq!(values[4].abs(), 0.5);
            assert_eq!(values[6].abs(), 0.5);
            corners.push((values[4].is_sign_positive(), values[6].is_sign_positive()));
        }
        corners.sort();
        corners.dedup();
        assert_eq!(corners.len(), 4);
    }
    assert_eq!(lbvh.diagnostics().pair_readbacks, 0);
    assert_eq!(lbvh.diagnostics().bind_groups_created, 3);
    assert_eq!(narrow.resource_counts(), [9, 6, 2]);
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_lbvh_gjk_epa_chain_has_no_substep_host_roundtrip() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = fixture();
    model.upload_to_device(device, queue).unwrap();
    let pose = buffer(device, queue, bytemuck::cast_slice(&poses(1.5)));
    let other_pose = buffer(device, queue, bytemuck::cast_slice(&poses(1.5)));
    let mut lbvh = LbvhBroadPhase::new(device.clone(), queue.clone(), 2, 64).unwrap();
    lbvh.dispatch_no_readback(&[
        (-Vec3::ONE, Vec3::ONE),
        (Vec3::new(0.5, -1.0, -1.0), Vec3::new(2.5, 1.0, 1.0)),
    ])
    .unwrap();
    let (count, a, b) = lbvh.pair_buffers();
    let narrow =
        ResidentNarrowPhase::new(device, &model, &[&pose, &other_pose], a, b, count, 64).unwrap();
    assert_eq!(narrow.resource_counts(), [9, 7, 2]);
    let bounds = molla_geometry::ResidentShapeBounds::new(
        device,
        &model,
        &[&pose, &other_pose],
        lbvh.primitive_aabb_buffer(),
        narrow.status_buffer(),
        0.0,
    )
    .unwrap();
    let mut data = Vec::new();
    let positions = [1.5_f32, 3.0, 1.0, -1.5];
    for x in positions {
        data.extend(poses(x));
    }
    let inputs = buffer(device, queue, bytemuck::cast_slice(&data));
    let output = staging(device, 100 * 76);
    let manifold_output = staging(device, 64 * (MANIFOLD_BYTES + 4));
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(narrow.encode(&mut encoder, 2).is_err());
    narrow.encode_reset_status(&mut encoder);
    for frame in 0..100_u64 {
        let source = (frame % 4) * 64;
        let selected = if frame % 2 == 0 { &pose } else { &other_pose };
        encoder.copy_buffer_to_buffer(&inputs, source, selected, 0, 64);
        bounds.encode(&mut encoder, (frame % 2) as usize).unwrap();
        lbvh.encode_refit(&mut encoder, 2).unwrap();
        narrow.encode(&mut encoder, (frame % 2) as usize).unwrap();
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, frame * 76, 4);
        encoder.copy_buffer_to_buffer(count, 0, &output, frame * 76 + 4, 4);
        encoder.copy_buffer_to_buffer(narrow.overlap_buffer(), 0, &output, frame * 76 + 8, 4);
        encoder.copy_buffer_to_buffer(narrow.contact_buffer(), 0, &output, frame * 76 + 12, 64);
    }
    encoder.copy_buffer_to_buffer(
        narrow.manifold_count_buffer(),
        0,
        &manifold_output,
        0,
        64 * 4,
    );
    encoder.copy_buffer_to_buffer(
        narrow.manifold_buffer(),
        0,
        &manifold_output,
        64 * 4,
        64 * MANIFOLD_BYTES,
    );
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    let manifold_words = read(device, &manifold_output);
    assert_eq!(manifold_words[0], 1);
    assert!(manifold_words[1..64].iter().all(|&v| v == 0));
    assert_eq!(&manifold_words[64..80], &words[99 * 19 + 3..100 * 19]);
    assert!(manifold_words[80..].iter().all(|&v| v == 0));
    for (frame, row) in words.chunks_exact(19).enumerate() {
        assert_eq!(row[0], 0, "fault at frame {frame}");
        let x = positions[frame % 4];
        if x == 3.0 {
            assert_eq!(&row[1..3], &[0, 0]);
            assert!(row[3..].iter().all(|&v| v == 0));
        } else {
            assert_eq!(&row[1..3], &[1, 1]);
            assert_eq!(row[15], 1);
            let depth = f32::from_bits(row[6]);
            assert!((depth - (2.0 - x.abs())).abs() < 1e-3, "{depth} at {x}");
            let normal = [row[3], row[4], row[5]].map(f32::from_bits);
            assert!((normal[0] - x.signum()).abs() < 1e-3, "{normal:?}");
        }
    }
    assert_eq!(lbvh.diagnostics().pair_readbacks, 0);
    assert_eq!(lbvh.diagnostics().bind_groups_created, 3);
    assert_eq!(narrow.resource_counts(), [9, 7, 2]);
    assert_eq!(bounds.resource_counts(), [2, 3, 1]);
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_faults_are_explicit_sticky_and_clearable() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = fixture();
    model.upload_to_device(device, queue).unwrap();
    let pose = buffer(device, queue, bytemuck::cast_slice(&poses(1.5)));
    let a = buffer(device, queue, &[0]);
    let b = buffer(device, queue, &[1]);
    let count = buffer(device, queue, &[1]);
    let narrow = ResidentNarrowPhase::new(device, &model, &[&pose], &a, &b, &count, 1).unwrap();
    let source = buffer(device, queue, &[0, 1, 2, 99, f32::NAN.to_bits()]);
    let output = staging(device, 6 * 68);
    let mut encoder = device.create_command_encoder(&Default::default());
    for frame in 0..6_u64 {
        if frame != 1 {
            narrow.encode_reset_status(&mut encoder);
        }
        let count_offset = if frame == 0 {
            8
        } else if frame <= 2 {
            0
        } else {
            4
        };
        encoder.copy_buffer_to_buffer(&source, count_offset, &count, 0, 4);
        if frame == 3 {
            encoder.copy_buffer_to_buffer(&source, 12, &b, 0, 4);
        }
        if frame == 4 {
            encoder.copy_buffer_to_buffer(&source, 4, &b, 0, 4);
            encoder.copy_buffer_to_buffer(
                &source,
                12,
                model.shape_geom_kind.device_buffer().unwrap(),
                0,
                4,
            );
        }
        if frame == 5 {
            encoder.copy_buffer_to_buffer(
                &source,
                8,
                model.shape_geom_kind.device_buffer().unwrap(),
                0,
                4,
            );
            encoder.copy_buffer_to_buffer(&source, 16, &pose, 0, 4);
        }
        narrow.encode(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, frame * 68, 4);
        encoder.copy_buffer_to_buffer(narrow.contact_buffer(), 0, &output, frame * 68 + 4, 64);
    }
    queue.submit([encoder.finish()]);
    let expected = [
        NARROW_FAULT_CAPACITY,
        NARROW_FAULT_CAPACITY,
        0,
        NARROW_FAULT_INDEX,
        NARROW_FAULT_GJK,
        NARROW_FAULT_NONFINITE,
    ];
    for (row, fault) in read(device, &output).chunks_exact(17).zip(expected) {
        assert_eq!(row[0], fault);
        assert_eq!(row[13], 0);
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_preparation_rejects_invalid_capacities_and_buffers() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = fixture();
    let pose = buffer(device, queue, bytemuck::cast_slice(&poses(1.5)));
    let small = buffer(device, queue, &[0]);
    assert!(ResidentNarrowPhase::new(device, &model, &[&pose], &small, &small, &small, 1).is_err());
    model.upload_to_device(device, queue).unwrap();
    for cap in [0, 2, u32::MAX] {
        assert!(
            ResidentNarrowPhase::new(device, &model, &[&pose], &small, &small, &small, cap)
                .is_err()
        );
    }
    assert!(
        ResidentNarrowPhase::new(device, &model, &[&small], &small, &small, &small, 1).is_err()
    );
    assert!(ResidentNarrowPhase::new(device, &model, &[], &small, &small, &small, 1).is_err());
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_batch_crosses_workgroups_and_clears_retired_slots() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = fixture();
    model.upload_to_device(device, queue).unwrap();
    let pose = buffer(device, queue, bytemuck::cast_slice(&poses(1.5)));
    let a = buffer(device, queue, &(0..130).map(|i| i % 2).collect::<Vec<_>>());
    let b = buffer(
        device,
        queue,
        &(0..130).map(|i| 1 - i % 2).collect::<Vec<_>>(),
    );
    let count = buffer(device, queue, &[130]);
    let source = buffer(device, queue, &[130, 1, 0]);
    let narrow = ResidentNarrowPhase::new(device, &model, &[&pose], &a, &b, &count, 130).unwrap();
    let stride = 4 + 130 * 64;
    let output = staging(device, 3 * stride);
    let mut encoder = device.create_command_encoder(&Default::default());
    for frame in 0..3 {
        encoder.copy_buffer_to_buffer(&source, frame * 4, &count, 0, 4);
        narrow.encode(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, frame * stride, 4);
        encoder.copy_buffer_to_buffer(
            narrow.contact_buffer(),
            0,
            &output,
            frame * stride + 4,
            130 * 64,
        );
    }
    queue.submit([encoder.finish()]);
    for (frame, row) in read(device, &output)
        .chunks_exact(stride as usize / 4)
        .enumerate()
    {
        assert_eq!(row[0], 0);
        let active = [130, 1, 0][frame];
        for (slot, contact) in row[1..].chunks_exact(16).enumerate() {
            if slot >= active {
                assert!(contact.iter().all(|&v| v == 0));
            } else {
                assert_eq!(contact[12], 1);
                assert!((f32::from_bits(contact[3]) - 0.5).abs() < 1e-3);
                let direction = if slot % 2 == 0 { 1.0 } else { -1.0 };
                assert!((f32::from_bits(contact[0]) - direction).abs() < 1e-3);
                let pa = contact[4..7]
                    .iter()
                    .copied()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>();
                let pb = contact[8..11]
                    .iter()
                    .copied()
                    .map(f32::from_bits)
                    .collect::<Vec<_>>();
                assert!(((pa[0] - pb[0]) * direction - 0.5).abs() < 1e-3);
            }
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_epa_nonconvergence_is_a_device_fault() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let first = builder.add_body(BodyParams::new());
    let second = builder.add_body(BodyParams::new());
    builder
        .add_shape_capsule(first, 0.4, 0.3, Transform::IDENTITY)
        .unwrap();
    builder
        .add_shape_box(second, Vec3::splat(0.5), Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let pose = buffer(device, queue, bytemuck::cast_slice(&poses(0.0)));
    let a = buffer(device, queue, &[0]);
    let b = buffer(device, queue, &[1]);
    let count = buffer(device, queue, &[1]);
    let narrow = ResidentNarrowPhase::new(device, &model, &[&pose], &a, &b, &count, 1).unwrap();
    let output = staging(device, 72);
    let mut encoder = device.create_command_encoder(&Default::default());
    narrow.encode(&mut encoder, 0).unwrap();
    encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, 0, 4);
    encoder.copy_buffer_to_buffer(narrow.overlap_buffer(), 0, &output, 4, 4);
    encoder.copy_buffer_to_buffer(narrow.contact_buffer(), 0, &output, 8, 64);
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    assert_eq!(words[0], molla_geometry::NARROW_FAULT_EPA);
    assert_eq!(words[1], 1);
    assert_eq!(words[14], molla_geometry::EPA_STATUS_FALLBACK);
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_collision_permissions_match_cpu_for_both_pair_orders() {
    use molla_core::{BodyId, WorldId};
    use molla_geometry::ResidentCollisionOptions;
    use molla_sim::ShapeFlags;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let mut bodies = Vec::new();
    let mut shapes = Vec::new();
    for i in 0..12 {
        let parent = if i == 8 || i == 9 {
            BodyId::NONE
        } else if i == 10 {
            bodies[0]
        } else {
            let body = builder.add_body(BodyParams::new().with_world(WorldId(u32::from(i == 1))));
            bodies.push(body);
            body
        };
        let shape = builder
            .add_shape_box(
                parent,
                Vec3::ONE,
                Transform::from_translation(Vec3::X * (i as f64 * 0.07)),
            )
            .unwrap();
        if i == 2 {
            builder.set_shape_collision_groups(shape, 1, 2).unwrap();
        }
        if i == 3 {
            builder.set_shape_collision_groups(shape, 2, 1).unwrap();
        }
        if i == 4 {
            builder.set_shape_collision_groups(shape, 1, 1).unwrap();
        }
        if i == 5 {
            builder
                .set_shape_flags(shape, ShapeFlags::COLLIDES | ShapeFlags::SENSOR)
                .unwrap();
        }
        if i == 6 {
            builder.set_shape_flags(shape, 0).unwrap();
        }
        shapes.push(shape);
    }
    builder.add_shape_collision_exclusion(shapes[0], shapes[11]);
    let mut model = builder.build().unwrap();
    model.shape_geom_kind.host_mut().unwrap()[6] = 0xff;
    model.body_world.host_mut().unwrap()[bodies[7].index()] = -1;
    model.upload_to_device(device, queue).unwrap();
    let state = model.state().unwrap();
    let gpu_poses: Vec<_> = state
        .body_q
        .host()
        .unwrap()
        .iter()
        .copied()
        .map(molla_math::GpuTransform::from)
        .collect();
    let pose = buffer(device, queue, bytemuck::cast_slice(&gpu_poses));
    let pairs: Vec<_> = (0..12_u32)
        .flat_map(|a| (0..12_u32).filter(move |&b| b != a).map(move |b| (a, b)))
        .collect();
    let a = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p.0).collect::<Vec<_>>(),
    );
    let b = buffer(
        device,
        queue,
        &pairs.iter().map(|p| p.1).collect::<Vec<_>>(),
    );
    let count = buffer(device, queue, &[pairs.len() as u32]);
    let extras = [(7, 0), (0, 7), (11, 0)];
    for include_sensor_overlaps in [false, true] {
        let narrow = ResidentNarrowPhase::with_collision_options(
            device,
            &model,
            &[&pose],
            &a,
            &b,
            &count,
            pairs.len() as u32,
            ResidentCollisionOptions {
                include_sensor_overlaps,
                extra_shape_exclusions: &extras,
                ..Default::default()
            },
        )
        .unwrap();
        let snapshot = staging(device, 4 + pairs.len() as u64 * 64);
        let mut encoder = device.create_command_encoder(&Default::default());
        narrow.encode(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &snapshot, 0, 4);
        encoder.copy_buffer_to_buffer(
            narrow.contact_buffer(),
            0,
            &snapshot,
            4,
            pairs.len() as u64 * 64,
        );
        queue.submit([encoder.finish()]);
        let words = read(device, &snapshot);
        assert_eq!(words[0], 0);
        let flags = model.shape_flags.host().unwrap();
        let mut admitted = 0;
        for ((a, b), contact) in pairs.iter().copied().zip(words[1..].chunks_exact(16)) {
            let allowed = molla_geometry::broad_phase::pair_passes_filters(
                &model,
                model.shape_body.host().unwrap(),
                model.body_world.host().unwrap(),
                a as usize,
                b as usize,
            ) && (include_sensor_overlaps
                || (flags[a as usize] | flags[b as usize]) & ShapeFlags::SENSOR == 0)
                && !extras
                    .iter()
                    .any(|&(x, y)| (a.min(b), a.max(b)) == (x.min(y), x.max(y)));
            assert_eq!(
                contact[12],
                u32::from(allowed),
                "pair {a},{b}, sensors={include_sensor_overlaps}"
            );
            if allowed {
                admitted += 1;
            } else {
                assert!(contact.iter().all(|&v| v == 0));
            }
        }
        assert!(admitted > 20 && admitted < pairs.len());
    }
    for extra in [[(0, 99)], [(1, 1)]] {
        assert!(
            ResidentNarrowPhase::with_collision_options(
                device,
                &model,
                &[&pose],
                &a,
                &b,
                &count,
                pairs.len() as u32,
                ResidentCollisionOptions {
                    extra_shape_exclusions: &extra,
                    ..Default::default()
                }
            )
            .is_err()
        );
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn rebuilt_resident_filters_follow_scene_breaks_repairs_and_compaction() {
    use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, JointDesc, RigidScene};
    use molla_sim::{JointType, MassProperties};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut scene = RigidScene::new().unwrap();
    let body = |x| BodyDesc {
        params: BodyParams::new().with_position(Vec3::X * x),
        additional_mass: Some(MassProperties {
            mass: 1.0,
            local_com: Vec3::ZERO,
            inertia: molla_math::Mat3::IDENTITY,
        }),
        ..Default::default()
    };
    let removed = scene.insert_body(body(20.0)).unwrap();
    let a = scene.insert_body(body(0.0)).unwrap();
    let b = scene.insert_body(body(1.5)).unwrap();
    let collider = |parent| ColliderDesc {
        parent: Some(parent),
        ..ColliderDesc::new(ColliderGeometry::Cuboid {
            half_extents: Vec3::ONE,
        })
    };
    scene.insert_collider(collider(removed)).unwrap();
    let ca = scene.insert_collider(collider(a)).unwrap();
    let cb = scene.insert_collider(collider(b)).unwrap();
    let check = |scene: &mut RigidScene, expected: u32, label: &str| {
        scene.upload_model_to_device(device, queue).unwrap();
        let data: Vec<_> = scene
            .state()
            .body_q
            .host()
            .unwrap()
            .iter()
            .copied()
            .map(molla_math::GpuTransform::from)
            .collect();
        let pose = buffer(device, queue, bytemuck::cast_slice(&data));
        let pairs: Vec<_> = scene
            .collider_indices(ca)
            .unwrap()
            .iter()
            .flat_map(|a| {
                scene
                    .collider_indices(cb)
                    .unwrap()
                    .iter()
                    .map(move |b| (a.0, b.0))
            })
            .collect();
        let first = buffer(
            device,
            queue,
            &pairs.iter().map(|p| p.0).collect::<Vec<_>>(),
        );
        let second = buffer(
            device,
            queue,
            &pairs.iter().map(|p| p.1).collect::<Vec<_>>(),
        );
        let count = buffer(device, queue, &[pairs.len() as u32]);
        let narrow = ResidentNarrowPhase::new(
            device,
            scene.model(),
            &[&pose],
            &first,
            &second,
            &count,
            pairs.len() as u32,
        )
        .unwrap();
        let output = staging(device, 4 + pairs.len() as u64 * 64);
        let mut encoder = device.create_command_encoder(&Default::default());
        narrow.encode(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, 0, 4);
        encoder.copy_buffer_to_buffer(
            narrow.contact_buffer(),
            0,
            &output,
            4,
            pairs.len() as u64 * 64,
        );
        queue.submit([encoder.finish()]);
        let words = read(device, &output);
        assert_eq!(words[0], 0, "{label}");
        for contact in words[1..].chunks_exact(16) {
            assert_eq!(contact[12], expected, "{label}");
        }
    };
    check(&mut scene, 1, "unconstrained");
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(a),
            child: b,
            kind: JointType::Fixed,
            axis: Vec3::X,
            frame_parent: Transform::from_translation(Vec3::X * 1.5),
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    check(&mut scene, 0, "joint-owned exclusion");
    scene.break_joints(&[joint]).unwrap();
    check(&mut scene, 1, "break releases joint policy");
    scene.repair_joint(joint).unwrap();
    check(&mut scene, 0, "repair restores joint policy");
    scene.set_body_pair_excluded(a, b, true).unwrap();
    scene.break_joints(&[joint]).unwrap();
    check(&mut scene, 0, "explicit body exclusion survives break");
    scene.set_collider_pair_excluded(cb, ca, true).unwrap();
    scene.set_body_pair_excluded(a, b, false).unwrap();
    check(&mut scene, 0, "independent collider exclusion");
    scene.remove_body(removed).unwrap();
    assert_eq!(scene.collider_index(ca).unwrap().0, 0);
    check(&mut scene, 0, "dense compaction remaps exclusions");
    let mut compound = collider(b);
    let tetrahedron = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
    compound.geometry = ColliderGeometry::ConvexParts(vec![tetrahedron.clone(), tetrahedron]);
    scene.replace_collider(cb, compound).unwrap();
    assert_eq!(scene.model().shape_collision_filter_pairs.len(), 2);
    check(
        &mut scene,
        0,
        "all compound leaves excluded before unsupported geometry",
    );
    scene.replace_collider(cb, collider(b)).unwrap();
    scene.set_collider_pair_excluded(ca, cb, false).unwrap();
    check(&mut scene, 1, "explicit release");
    let mut desc = scene.collider(cb).unwrap().clone();
    desc.enabled = false;
    scene.replace_collider(cb, desc.clone()).unwrap();
    check(&mut scene, 0, "disabled collider");
    desc.enabled = true;
    desc.sensor = true;
    scene.replace_collider(cb, desc.clone()).unwrap();
    check(&mut scene, 0, "sensor excluded from solid response");
    desc.sensor = false;
    scene.replace_collider(cb, desc).unwrap();
    check(&mut scene, 1, "re-enabled solid collider");
}
