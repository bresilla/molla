#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_geometry::{RESIDENT_MANIFOLD_POINTS, ResidentNarrowPhase};
use molla_math::{GpuSpatialVector, GpuTransform, Quat, SpatialVector, Transform, Vec3};
use molla_sim::{
    BodyFlags, BodyParams, CoefficientCombineRule, ContactMaterial, Contacts, FrictionCoefficients,
    ModelBuilder, ShapeFlags,
};
use molla_solvers::normal_contact_gpu::{
    ContactStateBuffers, NormalContactRow, ResidentNormalContacts,
};

fn buffer(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) -> wgpu::Buffer {
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("normal contact test input"),
        size: bytes.len().max(4) as u64,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    if !bytes.is_empty() {
        queue.write_buffer(&output, 0, bytes);
    }
    output
}

fn staging(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("normal contact test readback"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn read(device: &wgpu::Device, output: &wgpu::Buffer) -> Vec<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    output
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| tx.send(result).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let data = output.slice(..).get_mapped_range().to_vec();
    output.unmap();
    data
}

fn vector(v: [f32; 4]) -> Vec3 {
    Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
}
fn close(actual: Vec3, expected: Vec3) {
    assert!(
        (actual - expected).length() < 2e-5 * expected.length().max(1.0),
        "{actual:?} vs {expected:?}"
    );
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn normal_contact_faults_are_sticky_and_preparation_rejects_bad_inputs() {
    use molla_solvers::normal_contact_gpu::{
        CONTACT_FAULT_INPUT, CONTACT_FAULT_MATERIAL, CONTACT_FAULT_NONFINITE,
    };
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    let shape = builder
        .add_shape_box(body, Vec3::ONE, Transform::IDENTITY)
        .unwrap();
    let plane = builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    for shape in [shape, plane] {
        builder
            .set_shape_contact_material(
                shape,
                ContactMaterial {
                    stiffness: 1e30,
                    damping: 0.0,
                    friction_stiffness: 0.0,
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    let pose = buffer(
        device,
        queue,
        bytemuck::bytes_of(&GpuTransform::from(Transform::IDENTITY)),
    );
    let velocity = buffer(device, queue, &[0; 32]);
    let force = buffer(device, queue, &[0; 32]);
    let a = buffer(device, queue, bytemuck::bytes_of(&0u32));
    let b = buffer(device, queue, bytemuck::bytes_of(&1u32));
    let count = buffer(device, queue, bytemuck::bytes_of(&1u32));
    let counts = buffer(device, queue, bytemuck::bytes_of(&1u32));
    let mut geometry = vec![[0_f32; 16]; RESIDENT_MANIFOLD_POINTS as usize];
    geometry[0][1] = -1.0;
    geometry[0][3] = 1.0;
    geometry[0][5] = -1.0;
    geometry[0][12] = f32::from_bits(1);
    let points = buffer(device, queue, bytemuck::cast_slice(&geometry));
    let status = buffer(device, queue, &[0; 4]);
    let states = [ContactStateBuffers {
        poses: &pose,
        velocities: &velocity,
        external_forces: &force,
    }];
    let make = |model: &molla_sim::Model, capacity| {
        ResidentNormalContacts::new(
            device, model, &states, &a, &b, &count, &counts, &points, &status, capacity,
        )
    };
    assert!(make(&model, 0).is_err());
    assert!(make(&model, 2).is_err());
    assert!(make(&model, u32::MAX).is_err());
    let saved = model.shape_contact_material[0];
    for bad in [f64::NAN, -1.0, 1e-50, 1e50] {
        model.shape_contact_material[0].stiffness = bad;
        assert!(make(&model, 1).is_err());
    }
    model.shape_contact_material[0] = saved;
    let stage = make(&model, 1).unwrap();
    let constants = buffer(
        device,
        queue,
        bytemuck::cast_slice(&[
            0u32,
            1,
            99,
            RESIDENT_MANIFOLD_POINTS + 1,
            f32::NAN.to_bits(),
            (-1_f32).to_bits(),
            1e20_f32.to_bits(),
            1_f32.to_bits(),
        ]),
    );
    let output = staging(device, 8 * 36);
    let mut encoder = device.create_command_encoder(&Default::default());
    for step in 0..8_u64 {
        if step != 6 {
            encoder.clear_buffer(&status, 0, None);
        }
        if step == 1 {
            encoder.copy_buffer_to_buffer(&constants, 8, &a, 0, 4);
        }
        if step == 2 {
            encoder.copy_buffer_to_buffer(&constants, 0, &a, 0, 4);
            encoder.copy_buffer_to_buffer(&constants, 12, &counts, 0, 4);
        }
        if step == 3 {
            encoder.copy_buffer_to_buffer(&constants, 4, &counts, 0, 4);
            encoder.copy_buffer_to_buffer(&constants, 16, &points, 4, 4);
        }
        if step == 4 {
            encoder.copy_buffer_to_buffer(&constants, 20, &points, 4, 4);
            encoder.copy_buffer_to_buffer(&constants, 24, &points, 12, 4);
        }
        if step == 5 {
            encoder.copy_buffer_to_buffer(&constants, 28, &points, 12, 4);
            encoder.copy_buffer_to_buffer(&constants, 16, &force, 0, 4);
        }
        if step == 6 {
            encoder.clear_buffer(&force, 0, None);
            encoder.clear_buffer(&count, 0, None);
        }
        stage.encode(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(&status, 0, &output, step * 36, 4);
        encoder.copy_buffer_to_buffer(stage.force_buffer(), 0, &output, step * 36 + 4, 32);
    }
    queue.submit([encoder.finish()]);
    let words = read(device, &output);
    for (step, bytes) in words.chunks_exact(36).enumerate() {
        let expected = match step {
            0 | 7 => 0,
            1 | 2 => CONTACT_FAULT_INPUT,
            _ => CONTACT_FAULT_NONFINITE,
        };
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&bytes[..4]),
            expected,
            "step={step}"
        );
        if step != 0 {
            assert!(bytes[4..].iter().all(|&b| b == 0), "step={step}");
        }
    }
    for value in [1e20, 1e-30] {
        for material in &mut model.shape_contact_material {
            material.friction = Some(FrictionCoefficients::uniform(value));
            material.friction_combine = CoefficientCombineRule::Multiply;
        }
        let stage = make(&model, 1).unwrap();
        queue.write_buffer(&count, 0, bytemuck::bytes_of(&1u32));
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&status, 0, None);
        stage.encode(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(&status, 0, &output, 0, 4);
        queue.submit([encoder.finish()]);
        let bytes = read(device, &output);
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&bytes[..4]),
            CONTACT_FAULT_MATERIAL
        );
    }
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn normal_contact_rows_and_com_wrenches_match_cpu_materials_and_boundaries() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let rotation = Quat::from_axis_angle(Vec3::new(1.0, 2.0, 3.0).normalize(), 0.7);
    for i in 0..8 {
        let body = if i < 6 {
            builder.add_body(
                BodyParams::new()
                    .with_flags(match i {
                        2 => BodyFlags::KINEMATIC,
                        3 => BodyFlags::FIXED,
                        _ => 0,
                    })
                    .with_pose(Transform::new(Vec3::new(i as f64, -0.3, 0.7), rotation)),
            )
        } else if i == 6 {
            BodyId::from_index(0)
        } else {
            BodyId::NONE
        };
        let shape = builder
            .add_shape_box(body, Vec3::ONE, Transform::IDENTITY)
            .unwrap();
        builder
            .set_shape_material(shape, 0.3 + i as f64 * 0.05, i as f64 * 0.05)
            .unwrap();
        builder
            .set_shape_contact_material(
                shape,
                ContactMaterial {
                    friction: (i % 2 == 1).then_some(FrictionCoefficients {
                        mu_static: 0.8 + i as f64 * 0.1,
                        mu_dynamic: 0.2 + i as f64 * 0.01,
                    }),
                    stiffness: 100.0 + 10.0 * i as f64,
                    damping: 3.0 + i as f64,
                    friction_stiffness: 0.0,
                    friction_smoothing: 0.01,
                    friction_combine: [
                        CoefficientCombineRule::Average,
                        CoefficientCombineRule::Min,
                        CoefficientCombineRule::Multiply,
                        CoefficientCombineRule::Max,
                    ][i % 4],
                },
            )
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    model.shape_flags.host_mut().unwrap()[4] = ShapeFlags::COLLIDES | ShapeFlags::SENSOR;
    model.shape_flags.host_mut().unwrap()[5] = 0;
    for i in 0..8 {
        model.shape_surface_velocity[i] = Vec3::new(0.2 * i as f64, -0.1 * i as f64, 0.3);
    }
    for i in 0..6 {
        model.body_com.host_mut().unwrap()[i] = Vec3::new(0.1, -0.2, 0.05 * i as f64);
    }
    let mut state = model.state().unwrap();
    for i in 0..6 {
        state.body_qd.host_mut().unwrap()[i] = SpatialVector::new(
            Vec3::new(0.2 * i as f64, -0.4, 0.3),
            Vec3::new(0.1, -0.2, 0.03 * i as f64),
        );
        state.body_f.host_mut().unwrap()[i] =
            SpatialVector::new(Vec3::new(i as f64, 2.0, -1.0), Vec3::new(-2.0, 1.0, 0.3));
    }
    let pairs: Vec<_> = (0..8_u32)
        .flat_map(|a| (0..8_u32).filter(move |&b| b != a).map(move |b| (a, b)))
        .collect();
    let capacity = pairs.len() as u32 + 14;
    let width = RESIDENT_MANIFOLD_POINTS as usize;
    let mut a = vec![0; capacity as usize];
    let mut b = a.clone();
    let mut counts = a.clone();
    let mut points = vec![[0_f32; 16]; capacity as usize * width];
    let normal = Vec3::new(1.0, 2.0, -3.0).normalize();
    for (i, &(sa, sb)) in pairs.iter().enumerate() {
        a[i] = sa;
        b[i] = sb;
        counts[i] = 1 + i as u32 % 3;
        for j in 0..counts[i] as usize {
            let n = normal * if sa < sb { 1.0 } else { -1.0 };
            let p = Vec3::new(1.0, 0.2 * j as f64, -0.3);
            let row = &mut points[i * width + j];
            row[..3].copy_from_slice(&n.as_vec3().to_array());
            row[3] = 0.03;
            row[4..7].copy_from_slice(&(p + n * 0.015).as_vec3().to_array());
            row[8..11].copy_from_slice(&(p - n * 0.015).as_vec3().to_array());
            row[12] = f32::from_bits(1);
        }
    }
    let a = buffer(device, queue, bytemuck::cast_slice(&a));
    let b = buffer(device, queue, bytemuck::cast_slice(&b));
    let count = buffer(device, queue, bytemuck::bytes_of(&(pairs.len() as u32)));
    let counts_buffer = buffer(device, queue, bytemuck::cast_slice(&counts));
    let points_buffer = buffer(device, queue, bytemuck::cast_slice(&points));
    let status = buffer(device, queue, &[0; 4]);
    let poses: Vec<_> = state
        .body_q
        .host()
        .unwrap()
        .iter()
        .copied()
        .map(GpuTransform::from)
        .collect();
    let velocities: Vec<_> = state
        .body_qd
        .host()
        .unwrap()
        .iter()
        .copied()
        .map(GpuSpatialVector::from)
        .collect();
    let external: Vec<_> = state
        .body_f
        .host()
        .unwrap()
        .iter()
        .copied()
        .map(GpuSpatialVector::from)
        .collect();
    let pose = buffer(device, queue, bytemuck::cast_slice(&poses));
    let velocity = buffer(device, queue, bytemuck::cast_slice(&velocities));
    let force = buffer(device, queue, bytemuck::cast_slice(&external));
    let normal_stage = ResidentNormalContacts::new(
        device,
        &model,
        &[ContactStateBuffers {
            poses: &pose,
            velocities: &velocity,
            external_forces: &force,
        }],
        &a,
        &b,
        &count,
        &counts_buffer,
        &points_buffer,
        &status,
        capacity,
    )
    .unwrap();
    assert_eq!(normal_stage.resource_counts(), [7, 4, 3]);
    let row_bytes = capacity as u64 * width as u64 * 128;
    let frame_bytes = 4 + row_bytes + model.body_count as u64 * 32;
    let output = staging(device, frame_bytes * 2);
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(normal_stage.encode(&mut encoder, 1).is_err());
    for step in 0..2 {
        if step == 1 {
            encoder.clear_buffer(&count, 0, None);
        }
        normal_stage.encode(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(&status, 0, &output, step * frame_bytes, 4);
        encoder.copy_buffer_to_buffer(
            normal_stage.row_buffer(),
            0,
            &output,
            step * frame_bytes + 4,
            row_bytes,
        );
        encoder.copy_buffer_to_buffer(
            normal_stage.force_buffer(),
            0,
            &output,
            step * frame_bytes + 4 + row_bytes,
            model.body_count as u64 * 32,
        );
    }
    queue.submit([encoder.finish()]);
    let output = read(device, &output);
    for (step, bytes) in output.chunks_exact(frame_bytes as usize).enumerate() {
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&bytes[..4]), 0);
        let mut expected_forces = state.body_f.host().unwrap().to_vec();
        for (slot, bytes) in bytes[4..4 + row_bytes as usize]
            .chunks_exact(128)
            .enumerate()
        {
            let pair = slot / width;
            let j = slot % width;
            if step == 1 || pair >= pairs.len() || j >= counts[pair] as usize {
                assert!(bytes.iter().all(|&b| b == 0));
                continue;
            }
            let row = bytemuck::pod_read_unaligned::<NormalContactRow>(bytes);
            let (sa, sb) = pairs[pair];
            let mu = model
                .combined_shape_friction_coefficients(sa as usize, sb as usize)
                .unwrap();
            assert!((row.coefficients[0] as f64 - mu.mu_static).abs() < 1e-6);
            assert!((row.coefficients[1] as f64 - mu.mu_dynamic).abs() < 1e-6);
            let mut cpu = Contacts::with_capacity(1);
            cpu.count = 1;
            cpu.a_index.host_mut().unwrap()[0] =
                model.shape_body.host().unwrap()[sa as usize] as u32;
            cpu.b_index.host_mut().unwrap()[0] =
                model.shape_body.host().unwrap()[sb as usize] as u32;
            cpu.a_shape.host_mut().unwrap()[0] = sa;
            cpu.b_shape.host_mut().unwrap()[0] = sb;
            cpu.point.host_mut().unwrap()[0] = vector(row.point_depth);
            cpu.normal.host_mut().unwrap()[0] = vector(row.normal_force);
            cpu.depth.host_mut().unwrap()[0] = row.point_depth[3] as f64;
            cpu.static_friction.host_mut().unwrap()[0] = mu.mu_static;
            cpu.friction.host_mut().unwrap()[0] = mu.mu_dynamic;
            let reference = molla_solvers::penalty_contact::eval_penalty_contacts(
                &model,
                &state,
                &cpu,
                &mut expected_forces,
            )
            .unwrap()
            .remove(0);
            assert_eq!(row.coefficients[3], if reference.acted { 1.0 } else { 0.0 });
            let restitution = model.shape_restitution.host().unwrap();
            assert!(
                (row.coefficients[2] as f64
                    - restitution[sa as usize].max(restitution[sb as usize]))
                .abs()
                    < 1e-6
            );
            close(vector(row.force_on_b), reference.force_on_b);
            assert!(
                (row.normal_force[3] as f64 - reference.normal_force).abs()
                    < 2e-5 * reference.normal_force.max(1.0)
            );
            let mut speeds = [Vec3::ZERO; 2];
            let mut belts = [Vec3::ZERO; 2];
            for (side, shape) in [sa, sb].into_iter().enumerate() {
                let body = model.shape_body.host().unwrap()[shape as usize];
                belts[side] = model.shape_surface_velocity[shape as usize];
                if body >= 0 {
                    let body = body as usize;
                    let pose = state.body_q.host().unwrap()[body];
                    belts[side] = pose.rotation * belts[side];
                    if model.body_flags.host().unwrap()[body] & BodyFlags::FIXED == 0 {
                        let velocity = state.body_qd.host().unwrap()[body];
                        let arm = vector(row.point_depth)
                            - pose.transform_point(model.body_com.host().unwrap()[body]);
                        speeds[side] = velocity.linear + velocity.angular.cross(arm);
                    }
                }
            }
            close(vector(row.relative_velocity), speeds[1] - speeds[0]);
            close(vector(row.surface_velocity), belts[1] - belts[0]);
        }
        for (body, bytes) in bytes[4 + row_bytes as usize..].chunks_exact(32).enumerate() {
            let force = bytemuck::pod_read_unaligned::<GpuSpatialVector>(bytes);
            close(vector(force.linear), expected_forces[body].linear);
            close(vector(force.angular), expected_forces[body].angular);
        }
    }
    model.body_flags.host_mut().unwrap()[0] = BodyFlags::FIXED;
    model.shape_contact_material[0].friction = Some(FrictionCoefficients {
        mu_static: 0.05,
        mu_dynamic: 0.02,
    });
    model.shape_surface_velocity[0] = Vec3::new(0.9, -0.3, 0.4);
    let rebuilt = ResidentNormalContacts::new(
        device,
        &model,
        &[ContactStateBuffers {
            poses: &pose,
            velocities: &velocity,
            external_forces: &force,
        }],
        &a,
        &b,
        &count,
        &counts_buffer,
        &points_buffer,
        &status,
        capacity,
    )
    .unwrap();
    queue.write_buffer(&count, 0, bytemuck::bytes_of(&(pairs.len() as u32)));
    let snapshot = staging(device, 164);
    let mut encoder = device.create_command_encoder(&Default::default());
    rebuilt.encode(&mut encoder, 0).unwrap();
    encoder.copy_buffer_to_buffer(&status, 0, &snapshot, 0, 4);
    encoder.copy_buffer_to_buffer(rebuilt.row_buffer(), 0, &snapshot, 4, 128);
    encoder.copy_buffer_to_buffer(rebuilt.force_buffer(), 0, &snapshot, 132, 32);
    queue.submit([encoder.finish()]);
    let bytes = read(device, &snapshot);
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&bytes[..4]), 0);
    let row = bytemuck::pod_read_unaligned::<NormalContactRow>(&bytes[4..132]);
    assert!((row.coefficients[0] - 0.05).abs() < 1e-6);
    assert!((row.coefficients[1] - 0.02).abs() < 1e-6);
    close(
        vector(row.surface_velocity),
        rotation * (model.shape_surface_velocity[1] - model.shape_surface_velocity[0]),
    );
    let body_force = bytemuck::pod_read_unaligned::<GpuSpatialVector>(&bytes[132..164]);
    close(
        vector(body_force.linear),
        state.body_f.host().unwrap()[0].linear,
    );
    close(
        vector(body_force.angular),
        state.body_f.host().unwrap()[0].angular,
    );
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn device_normal_forces_drive_featherstone_drop_and_settle_like_cpu() {
    use molla_math::{Mat3, SpatialInertia};
    use molla_solvers::{FeatherstoneSolver, Solver};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_linear_damping(0.0);
    builder.set_angular_damping(0.0);
    let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 2.0,
        com: Vec3::ZERO,
        inertia: Mat3::IDENTITY / 3.0,
    }));
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let shape = builder
        .add_shape_box(body, Vec3::splat(0.5), Transform::IDENTITY)
        .unwrap();
    let plane = builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    for shape in [shape, plane] {
        builder.set_shape_material(shape, 0.0, 0.0).unwrap();
        builder
            .set_shape_contact_material(
                shape,
                ContactMaterial {
                    stiffness: 1e4,
                    damping: 100.0,
                    friction: Some(FrictionCoefficients::uniform(0.0)),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    initial.joint_q.host_mut().unwrap()[1] = 0.6;
    molla_sim::eval_fk(&model, &mut initial).unwrap();
    molla_sim::eval_body_velocities(&model, &mut initial).unwrap();
    let mut input = initial.clone_host().unwrap();
    let mut output_state = initial.clone_host().unwrap();
    let mut cpu = FeatherstoneSolver::new_cpu();
    let mut pipeline = molla_geometry::CollisionPipeline::new();
    let mut contacts = Contacts::with_capacity(64);
    let mut control = model.control();
    let dt = 0.001;
    let steps = 1000;
    let mut reference = Vec::new();
    for _ in 0..steps {
        pipeline.collide(&model, &input, &mut contacts).unwrap();
        cpu.step(&model, &input, &control, &contacts, dt, &mut output_state)
            .unwrap();
        reference.push((
            output_state.body_q.host().unwrap()[0],
            output_state.body_qd.host().unwrap()[0],
        ));
        std::mem::swap(&mut input, &mut output_state);
    }
    model.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    let mut states = [initial.clone_host().unwrap(), initial.clone_host().unwrap()];
    for state in &mut states {
        state.upload_to_device(device, queue).unwrap();
    }
    let mut gpu = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    let mut lbvh =
        molla_geometry::LbvhBroadPhase::new(device.clone(), queue.clone(), 2, 64).unwrap();
    lbvh.dispatch_no_readback(&[
        (-Vec3::ONE, Vec3::ONE),
        (Vec3::splat(f64::NEG_INFINITY), Vec3::splat(f64::INFINITY)),
    ])
    .unwrap();
    let poses = states.each_ref().map(|s| s.body_q.device_buffer().unwrap());
    let (count, a, b) = lbvh.pair_buffers();
    let narrow = ResidentNarrowPhase::new(device, &model, &poses, a, b, count, 64).unwrap();
    let bounds = molla_geometry::ResidentShapeBounds::new(
        device,
        &model,
        &poses,
        lbvh.primitive_aabb_buffer(),
        narrow.status_buffer(),
        0.0,
    )
    .unwrap();
    let external = buffer(device, queue, &[0; 32]);
    let state_buffers = states.each_ref().map(|s| ContactStateBuffers {
        poses: s.body_q.device_buffer().unwrap(),
        velocities: s.body_qd.device_buffer().unwrap(),
        external_forces: &external,
    });
    let normal = ResidentNormalContacts::new(
        device,
        &model,
        &state_buffers,
        a,
        b,
        count,
        narrow.manifold_count_buffer(),
        narrow.manifold_buffer(),
        narrow.status_buffer(),
        64,
    )
    .unwrap();
    let snapshot = staging(device, steps as u64 * 104);
    for step in 0..steps {
        let side = step % 2;
        let mut encoder = device.create_command_encoder(&Default::default());
        if step == 0 {
            narrow.encode_reset_status(&mut encoder);
        }
        bounds.encode(&mut encoder, side).unwrap();
        lbvh.encode_refit(&mut encoder, 2).unwrap();
        narrow.encode(&mut encoder, side).unwrap();
        normal.encode(&mut encoder, side).unwrap();
        encoder.copy_buffer_to_buffer(
            normal.force_buffer(),
            0,
            states[side].body_f.device_buffer().unwrap(),
            0,
            32,
        );
        queue.submit([encoder.finish()]);
        let [a, b] = &mut states;
        let (input, output_state) = if side == 0 { (a, b) } else { (b, a) };
        gpu.step(
            &model,
            input,
            &control,
            &Contacts::with_capacity(0),
            dt,
            output_state,
        )
        .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        let offset = step as u64 * 104;
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &snapshot, offset, 4);
        encoder.copy_buffer_to_buffer(
            gpu.gpu_status_buffer().unwrap(),
            0,
            &snapshot,
            offset + 4,
            4,
        );
        encoder.copy_buffer_to_buffer(
            output_state.body_q.device_buffer().unwrap(),
            0,
            &snapshot,
            offset + 8,
            32,
        );
        encoder.copy_buffer_to_buffer(
            output_state.body_qd.device_buffer().unwrap(),
            0,
            &snapshot,
            offset + 40,
            32,
        );
        encoder.copy_buffer_to_buffer(normal.force_buffer(), 0, &snapshot, offset + 72, 32);
        queue.submit([encoder.finish()]);
    }
    let data = read(device, &snapshot);
    let mut peak_position_error = 0.0_f64;
    let mut peak_velocity_error = 0.0_f64;
    let mut peak_force = 0.0_f64;
    let mut minimum_height = 1.0_f64;
    for (step, bytes) in data.chunks_exact(104).enumerate() {
        assert_eq!(
            bytemuck::pod_read_unaligned::<[u32; 2]>(&bytes[..8]),
            [0; 2],
            "step={step}"
        );
        let pose = bytemuck::pod_read_unaligned::<GpuTransform>(&bytes[8..40]);
        let velocity = bytemuck::pod_read_unaligned::<GpuSpatialVector>(&bytes[40..72]);
        let force = bytemuck::pod_read_unaligned::<GpuSpatialVector>(&bytes[72..104]);
        peak_position_error =
            peak_position_error.max((vector(pose.position) - reference[step].0.position).length());
        peak_velocity_error =
            peak_velocity_error.max((vector(velocity.linear) - reference[step].1.linear).length());
        minimum_height = minimum_height.min(pose.position[1] as f64);
        peak_force = peak_force.max(force.linear[1] as f64);
        assert!(vector(velocity.angular).length() < 1e-4);
    }
    eprintln!(
        "normal drop: position_error={peak_position_error:e}, velocity_error={peak_velocity_error:e}, min_height={minimum_height}, peak_force={peak_force}"
    );
    assert!(peak_position_error < 3e-5);
    assert!(peak_velocity_error < 2e-3);
    assert!(minimum_height > 0.485 && minimum_height < 0.5);
    assert!(peak_force > 100.0);
    let last =
        bytemuck::pod_read_unaligned::<GpuSpatialVector>(&data[data.len() - 64..data.len() - 32]);
    assert!(vector(last.linear).length() < 2e-4);
    assert_eq!(lbvh.diagnostics().pair_readbacks, 0);
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn resident_bounds_manifolds_and_normal_forces_chain_without_host_roundtrips() {
    use molla_solvers::contact_islands_gpu::{ContactIslandConfig, ResidentContactIslands};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let plane = builder
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let body = builder.add_body(BodyParams::new());
    let shape = builder
        .add_shape_box(body, Vec3::splat(0.5), Transform::IDENTITY)
        .unwrap();
    for shape in [shape, plane] {
        builder
            .set_shape_contact_material(
                shape,
                ContactMaterial {
                    stiffness: 100.0,
                    damping: 0.0,
                    friction: Some(FrictionCoefficients {
                        mu_static: 0.6,
                        mu_dynamic: 0.2,
                    }),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    model.body_com.host_mut().unwrap()[0] = Vec3::new(0.1, 0.0, 0.2);
    model.upload_to_device(device, queue).unwrap();
    let heights = [0.4, 0.6, 0.2, 1.0];
    let poses = heights.map(|y| GpuTransform::from(Transform::from_translation(Vec3::Y * y)));
    let source = buffer(device, queue, bytemuck::cast_slice(&poses));
    let pose_a = buffer(device, queue, bytemuck::bytes_of(&poses[0]));
    let pose_b = buffer(device, queue, bytemuck::bytes_of(&poses[0]));
    let velocity = buffer(device, queue, &[0; 32]);
    let external = GpuSpatialVector::from(SpatialVector::new(
        Vec3::new(1.0, 2.0, 3.0),
        Vec3::new(4.0, 5.0, 6.0),
    ));
    let force = buffer(device, queue, bytemuck::bytes_of(&external));
    let mut lbvh =
        molla_geometry::LbvhBroadPhase::new(device.clone(), queue.clone(), 2, 64).unwrap();
    lbvh.dispatch_no_readback(&[
        (Vec3::splat(f64::NEG_INFINITY), Vec3::splat(f64::INFINITY)),
        (-Vec3::ONE, Vec3::ONE),
    ])
    .unwrap();
    let (count, a, b) = lbvh.pair_buffers();
    let narrow =
        ResidentNarrowPhase::new(device, &model, &[&pose_a, &pose_b], a, b, count, 64).unwrap();
    let bounds = molla_geometry::ResidentShapeBounds::new(
        device,
        &model,
        &[&pose_a, &pose_b],
        lbvh.primitive_aabb_buffer(),
        narrow.status_buffer(),
        0.0,
    )
    .unwrap();
    let state = [&pose_a, &pose_b].map(|poses| ContactStateBuffers {
        poses,
        velocities: &velocity,
        external_forces: &force,
    });
    let normal = ResidentNormalContacts::new(
        device,
        &model,
        &state,
        a,
        b,
        count,
        narrow.manifold_count_buffer(),
        narrow.manifold_buffer(),
        narrow.status_buffer(),
        64,
    )
    .unwrap();
    let islands = ResidentContactIslands::new(
        &engine,
        &[0],
        &[],
        normal.row_buffer(),
        narrow.status_buffer(),
        ContactIslandConfig {
            articulation_count: 1,
            row_capacity: 64 * RESIDENT_MANIFOLD_POINTS,
            island_capacity: 1,
            contacts_per_island: 4,
        },
    )
    .unwrap();
    let output = staging(device, 100 * 192);
    let mut encoder = device.create_command_encoder(&Default::default());
    narrow.encode_reset_status(&mut encoder);
    for step in 0..100_u64 {
        let selected = (step % 2) as usize;
        encoder.copy_buffer_to_buffer(&source, (step % 4) * 32, state[selected].poses, 0, 32);
        bounds.encode(&mut encoder, selected).unwrap();
        lbvh.encode_refit(&mut encoder, 2).unwrap();
        narrow.encode(&mut encoder, selected).unwrap();
        normal.encode(&mut encoder, selected).unwrap();
        islands.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &output, step * 192, 4);
        encoder.copy_buffer_to_buffer(normal.force_buffer(), 0, &output, step * 192 + 4, 32);
        encoder.copy_buffer_to_buffer(normal.row_buffer(), 0, &output, step * 192 + 36, 128);
        encoder.copy_buffer_to_buffer(
            islands.island_count_buffer(),
            0,
            &output,
            step * 192 + 164,
            4,
        );
        encoder.copy_buffer_to_buffer(
            islands.contact_counts_buffer(),
            0,
            &output,
            step * 192 + 168,
            4,
        );
        encoder.copy_buffer_to_buffer(
            islands.articulation_islands_buffer(),
            0,
            &output,
            step * 192 + 172,
            4,
        );
        encoder.copy_buffer_to_buffer(
            islands.contact_indices_buffer(),
            0,
            &output,
            step * 192 + 176,
            16,
        );
    }
    queue.submit([encoder.finish()]);
    let output = read(device, &output);
    for (step, bytes) in output.chunks_exact(192).enumerate() {
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&bytes[..4]),
            0,
            "step={step}"
        );
        let magnitude = 400.0 * (0.5_f64 - heights[step % 4]).max(0.0);
        let force = bytemuck::pod_read_unaligned::<GpuSpatialVector>(&bytes[4..36]);
        close(
            vector(force.linear),
            vector(external.linear) + Vec3::Y * magnitude,
        );
        close(
            vector(force.angular),
            vector(external.angular) + Vec3::new(0.2 * magnitude, 0.0, -0.1 * magnitude),
        );
        if magnitude == 0.0 {
            assert!(bytes[36..164].iter().all(|&b| b == 0));
        }
        let actual = bytemuck::pod_read_unaligned::<[u32; 7]>(&bytes[164..]);
        let expected = if magnitude == 0.0 {
            [0, 0, u32::MAX, u32::MAX, u32::MAX, u32::MAX, u32::MAX]
        } else {
            [1, 4, 0, 0, 1, 2, 3]
        };
        assert_eq!(actual, expected, "step={step}");
    }
    assert_eq!(normal.resource_counts(), [7, 5, 3]);
    assert_eq!(lbvh.diagnostics().pair_readbacks, 0);
    assert_eq!(lbvh.diagnostics().bind_groups_created, 3);
}
