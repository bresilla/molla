use super::impact_tests::{buffer, fixture, fixture_with_boundary_joint, floats, projection, row};
use super::*;
use crate::constraint_rows_gpu::ScalarConstraintRow;
use crate::normal_contact_gpu::NormalContactRow;
use crate::rigid_impulse::{ImpactConstraint, ImpactContact, ImpactRow, apply_impact};
use molla_core::BodyId;
use molla_math::{SpatialVector, Vec3};

fn read_floats(engine: &Engine, source: &wgpu::Buffer) -> Vec<f32> {
    let device = engine.device().unwrap();
    let staging = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("position integration snapshot"),
            size: source.size(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    floats(&molla_core::gpu_readback!(staging.slice(..)))
}

fn displace_reference(model: &Model, state: &mut State, delta: &[f64]) {
    use molla_math::Quat;
    let coords = model.joint_coord_offset.host().unwrap();
    let dofs = model.joint_dof_offset.host().unwrap();
    let q = state.joint_q.host_mut().unwrap();
    for (j, &kind) in model.joint_type.host().unwrap().iter().enumerate() {
        let c = coords[j] as usize;
        let d = dofs[j] as usize;
        if model.body_flags.host().unwrap()[model.joint_child.host().unwrap()[j] as usize] & 3 != 0
        {
            continue;
        }
        match kind {
            0 | 1 => {
                let lo = model.joint_limit_lower.host().unwrap()[d];
                let hi = model.joint_limit_upper.host().unwrap()[d];
                if lo.is_finite() && hi.is_finite() && (hi - lo).abs() < 1e-12 {
                    q[c] = lo;
                } else {
                    q[c] += delta[d];
                }
            }
            2 | 4 | 5 | 6 => {
                let angular = if kind == 2 { 0 } else { 3 };
                for k in 0..angular {
                    q[c + k] += delta[d + k];
                }
                let old = Quat::from_xyzw(
                    q[c + angular],
                    q[c + angular + 1],
                    q[c + angular + 2],
                    q[c + angular + 3],
                );
                let spin = Quat::from_xyzw(
                    delta[d + angular] * 0.5,
                    delta[d + angular + 1] * 0.5,
                    delta[d + angular + 2] * 0.5,
                    0.,
                );
                let change = spin * old;
                let next = Quat::from_xyzw(
                    old.x + change.x,
                    old.y + change.y,
                    old.z + change.z,
                    old.w + change.w,
                )
                .normalize();
                q[c + angular..c + angular + 4].copy_from_slice(&next.to_array());
            }
            3 => (),
            _ => panic!("unsupported reference joint {kind}"),
        }
    }
    molla_sim::eval_fk(model, state).unwrap();
    molla_sim::eval_body_velocities(model, state).unwrap();
}

fn assert_state_close(engine: &Engine, actual: &mut State, expected: &State, tolerance: f64) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    actual.joint_q.download(device, queue).unwrap();
    actual.joint_qd.download(device, queue).unwrap();
    actual.download_from_device(device, queue).unwrap();
    for (a, b) in actual
        .joint_q
        .host()
        .unwrap()
        .iter()
        .zip(expected.joint_q.host().unwrap())
    {
        assert!((a - b).abs() < tolerance, "coordinate: {a} != {b}");
    }
    for (a, b) in actual
        .joint_qd
        .host()
        .unwrap()
        .iter()
        .zip(expected.joint_qd.host().unwrap())
    {
        assert_eq!(*a as f32, *b as f32, "physical joint rate changed");
    }
    for (a, b) in actual
        .body_q
        .host()
        .unwrap()
        .iter()
        .zip(expected.body_q.host().unwrap())
    {
        assert!(
            (a.position - b.position).length() < tolerance,
            "pose: {a:?} != {b:?}"
        );
        assert!((a.rotation.dot(b.rotation).abs() - 1.).abs() < tolerance);
    }
    for (a, b) in actual
        .body_qd
        .host()
        .unwrap()
        .iter()
        .zip(expected.body_qd.host().unwrap())
    {
        assert!(
            (a.linear - b.linear).length() < tolerance,
            "linear velocity: {a:?} != {b:?}"
        );
        assert!(
            (a.angular - b.angular).length() < tolerance,
            "angular velocity: {a:?} != {b:?}"
        );
    }
}

#[test]
#[ignore = "requires GPU"]
fn solved_displacement_updates_coordinates_then_redrifts_without_advancing_boundary_twice() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, mut input) = fixture_with_boundary_joint(true);
    let contact = ImpactContact {
        bodies: [Some(BodyId(3)), Some(BodyId(1))],
        point: Vec3::new(-0.8, 0.3, 0.2),
        normal: Vec3::Y,
        static_friction: 0.8,
        dynamic_friction: 0.4,
        restitution: 0.9,
        surface_velocity: [Vec3::ZERO; 2],
    };
    let mut reference = input.clone_host().unwrap();
    reference.joint_qd.host_mut().unwrap().fill(0.);
    reference
        .body_qd
        .host_mut()
        .unwrap()
        .fill(SpatialVector::ZERO);
    apply_impact(
        &model,
        &mut reference,
        &[ImpactConstraint::Nonnegative(normal_row(
            &model, &input, contact, 0.004,
        ))],
    )
    .unwrap();
    let delta = reference.joint_qd.host().unwrap().to_vec();
    assert!(delta.iter().any(|v| v.abs() > 1e-4));
    reference = input.clone_host().unwrap();
    displace_reference(&model, &mut reference, &delta);
    let mut projected = model.state().unwrap();
    let mut output = input.clone_host().unwrap();
    let mut expected_output = reference.clone_host().unwrap();
    for (i, rate) in output.joint_qd.host_mut().unwrap().iter_mut().enumerate() {
        *rate = i as f64 * 0.03125;
    }
    expected_output
        .joint_qd
        .host_mut()
        .unwrap()
        .copy_from_slice(output.joint_qd.host().unwrap());
    crate::FeatherstoneSolver::new_cpu()
        .drift_cpu_selected(&model, &reference, 0.01, &mut expected_output, None)
        .unwrap();
    let mut source_row = row(&model, &input, contact);
    source_row.point_depth[3] = 0.004;
    let rows = buffer(device, queue, bytemuck::bytes_of(&source_row));
    let onsets = buffer(device, queue, bytemuck::cast_slice(&[1u32]));
    let status = buffer(device, queue, &[0; 4]);
    model.upload_to_device(device, queue).unwrap();
    input.upload_to_device(device, queue).unwrap();
    projected.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(device, queue).unwrap();
    let response = ResidentImpactPositionResponse::new(
        &engine,
        &model,
        &[&input],
        &rows,
        &[],
        None,
        &onsets,
        &status,
        config(1, 1),
    )
    .unwrap();
    let mut kernel = molla_kernels::FeatherstoneKernel::new(device, queue, &model);
    kernel.set_status_buffer(&status).unwrap();
    kernel
        .prepare_coordinate_compensation(device, &[&input, &projected, &output])
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    response.encode(&mut encoder, 0).unwrap();
    queue.submit([encoder.finish()]);
    kernel
        .apply_position_displacement(
            device,
            queue,
            &model,
            &input,
            &control,
            response.displacement_buffer(),
            &mut projected,
        )
        .unwrap();
    assert_state_close(&engine, &mut projected, &reference, 2e-6);
    assert_eq!(
        projected.body_q.host().unwrap()[3],
        input.body_q.host().unwrap()[3]
    );
    kernel
        .drift(
            device,
            queue,
            &model,
            &projected,
            &control,
            0.01,
            false,
            &mut output,
        )
        .unwrap();
    assert_state_close(&engine, &mut output, &expected_output, 2e-6);
    assert_eq!(read_floats(&engine, &status)[0].to_bits(), 0);
    assert!(
        read_floats(&engine, kernel.velocity_delta_buffer())
            .iter()
            .all(|v| *v == 0.)
    );
}

#[test]
#[ignore = "requires GPU"]
fn displacement_retains_sub_ulp_motion_rates_and_cached_bindings_across_faults() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, mut a) = fixture_with_boundary_joint(true);
    a.joint_q.host_mut().unwrap()[0] = 1024.;
    molla_sim::eval_fk(&model, &mut a).unwrap();
    molla_sim::eval_body_velocities(&model, &mut a).unwrap();
    let mut b = a.clone_host().unwrap();
    let rates: Vec<_> = a
        .joint_qd
        .host()
        .unwrap()
        .iter()
        .map(|&v| v as f32)
        .collect();
    let boundary = a.body_q.host().unwrap()[3];
    model.upload_to_device(device, queue).unwrap();
    a.upload_to_device(device, queue).unwrap();
    b.upload_to_device(device, queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(device, queue).unwrap();
    let mut delta = vec![0f32; model.joint_dof_total];
    delta[0] = 1e-5;
    let displacement = buffer(device, queue, bytemuck::cast_slice(&delta));
    let status = buffer(device, queue, &[0; 4]);
    let mut kernel = molla_kernels::FeatherstoneKernel::new(device, queue, &model);
    kernel.set_status_buffer(&status).unwrap();
    kernel
        .prepare_coordinate_compensation(device, &[&a, &b])
        .unwrap();
    let mut groups = 0;
    for iteration in 0..64 {
        kernel
            .apply_position_displacement(device, queue, &model, &a, &control, &displacement, &mut b)
            .unwrap();
        std::mem::swap(&mut a, &mut b);
        if iteration == 1 {
            groups = kernel.bind_groups_created();
        }
    }
    assert_eq!(kernel.bind_groups_created(), groups);
    let q = read_floats(&engine, a.joint_q.device_buffer_current().unwrap());
    let residue = read_floats(&engine, kernel.coordinate_compensation_buffer(&a).unwrap());
    let exact = q[0] as f64 + residue[0] as f64;
    assert!(
        (exact - (1024. + 64. * delta[0] as f64)).abs() < 1e-10,
        "retained {exact}"
    );
    assert_eq!(
        read_floats(&engine, a.joint_qd.device_buffer_current().unwrap()),
        rates
    );
    a.download_from_device(device, queue).unwrap();
    assert_eq!(a.body_q.host().unwrap()[3], boundary);
    let saved = read_floats(&engine, a.body_q.device_buffer_current().unwrap());
    delta[0] = f32::NAN;
    molla_core::gpu_write_buffer!(queue, &displacement, 0, bytemuck::cast_slice(&delta));
    kernel
        .apply_position_displacement(device, queue, &model, &a, &control, &displacement, &mut b)
        .unwrap();
    assert_eq!(read_floats(&engine, &status)[0].to_bits(), 1 << 31);
    assert_eq!(
        read_floats(&engine, b.body_q.device_buffer_current().unwrap()),
        saved
    );
    assert_eq!(
        read_floats(&engine, b.joint_q.device_buffer_current().unwrap()),
        q
    );
    assert_eq!(
        read_floats(&engine, kernel.coordinate_compensation_buffer(&b).unwrap()),
        residue
    );
    delta[0] = 1e-5;
    molla_core::gpu_write_buffer!(queue, &displacement, 0, bytemuck::cast_slice(&delta));
    kernel
        .apply_position_displacement(device, queue, &model, &a, &control, &displacement, &mut b)
        .unwrap();
    assert_eq!(
        read_floats(&engine, b.joint_q.device_buffer_current().unwrap()),
        q
    );
    molla_core::gpu_write_buffer!(queue, &status, 0, &[0; 4]);
    kernel
        .apply_position_displacement(device, queue, &model, &a, &control, &displacement, &mut b)
        .unwrap();
    let q = read_floats(&engine, b.joint_q.device_buffer_current().unwrap());
    let residue = read_floats(&engine, kernel.coordinate_compensation_buffer(&b).unwrap());
    assert!((q[0] as f64 + residue[0] as f64 - (1024. + 65. * delta[0] as f64)).abs() < 1e-10);
    assert_eq!(read_floats(&engine, &status)[0].to_bits(), 0);
}

#[test]
#[ignore = "requires GPU"]
fn coordinate_displacement_uses_joint_frames_not_velocity_caps_or_prescribed_time() {
    use molla_math::{Quat, Transform};
    use molla_sim::{BodyParams, ModelBuilder, eval_body_velocities, eval_fk};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let parent = builder.add_body(
        BodyParams::new()
            .with_position(Vec3::new(0.5, -0.25, 0.125))
            .with_flags(BodyFlags::KINEMATIC),
    );
    builder
        .add_joint_fixed(
            BodyId::NONE,
            parent,
            Transform::IDENTITY,
            Transform::IDENTITY,
        )
        .unwrap();
    builder
        .set_kinematic_acceleration(parent, Some(SpatialVector::new(Vec3::Y * 3., Vec3::Z)))
        .unwrap();
    for kind in [0, 1, 2, 4, 5, 6, 0] {
        let body = builder.add_body(BodyParams::new().with_mass(1.));
        let xp = Transform {
            position: Vec3::new(0.2, 0.3, -0.1),
            rotation: Quat::from_rotation_y(0.3),
        };
        let xc = Transform {
            position: Vec3::new(-0.1, 0.2, 0.05),
            rotation: Quat::from_rotation_z(0.1),
        };
        let joint = match kind {
            0 => builder.add_joint_prismatic(parent, body, xp, xc, Vec3::X),
            1 => builder.add_joint_revolute(parent, body, xp, xc, Vec3::Y),
            2 => builder.add_joint_ball(parent, body, xp, xc),
            4 => builder.add_joint_free(parent, body, xp, xc),
            5 => builder.add_joint_distance(parent, body, xp, xc),
            6 => builder.add_joint_d6(parent, body, xp, xc),
            _ => unreachable!(),
        }
        .unwrap();
        let count = match kind {
            0 | 1 => 1,
            2 => 3,
            _ => 6,
        };
        builder
            .set_joint_velocity_limit(joint, &vec![1e-6; count])
            .unwrap();
        if body.index() == 7 {
            builder.set_joint_limits(joint, &[0.125], &[0.125]).unwrap();
        }
    }
    let mut model = builder.build().unwrap();
    let mut input = model.state().unwrap();
    for (d, rate) in input.joint_qd.host_mut().unwrap().iter_mut().enumerate() {
        *rate = (d as f64 - 5.) * 0.03125;
    }
    input.body_qd.host_mut().unwrap()[0] = SpatialVector::new(Vec3::X * 0.5, Vec3::Y * 0.25);
    eval_fk(&model, &mut input).unwrap();
    eval_body_velocities(&model, &mut input).unwrap();
    let delta: Vec<f32> = (0..model.joint_dof_total)
        .map(|d| (d as f32 + 1.) * 0.001)
        .collect();
    let mut expected = input.clone_host().unwrap();
    displace_reference(
        &model,
        &mut expected,
        &delta.iter().map(|&v| v as f64).collect::<Vec<_>>(),
    );
    let mut output = input.clone_host().unwrap();
    model.upload_to_device(device, queue).unwrap();
    input.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(device, queue).unwrap();
    let displacement = buffer(device, queue, bytemuck::cast_slice(&delta));
    let mut kernel = molla_kernels::FeatherstoneKernel::new(device, queue, &model);
    kernel
        .apply_position_displacement(
            device,
            queue,
            &model,
            &input,
            &control,
            &displacement,
            &mut output,
        )
        .unwrap();
    assert_state_close(&engine, &mut output, &expected, 2e-6);
    assert_eq!(
        output.body_q.host().unwrap()[0],
        input.body_q.host().unwrap()[0]
    );
    assert_eq!(
        output.body_qd.host().unwrap()[0],
        input.body_qd.host().unwrap()[0]
    );
    molla_core::gpu_write_buffer!(
        queue,
        &displacement,
        0,
        bytemuck::cast_slice(&vec![0f32; delta.len()]),
    );
    kernel
        .apply_position_displacement(
            device,
            queue,
            &model,
            &input,
            &control,
            &displacement,
            &mut output,
        )
        .unwrap();
    for (a, b) in [
        (
            input.joint_q.device_buffer_current().unwrap(),
            output.joint_q.device_buffer_current().unwrap(),
        ),
        (
            input.joint_qd.device_buffer_current().unwrap(),
            output.joint_qd.device_buffer_current().unwrap(),
        ),
        (
            input.body_q.device_buffer_current().unwrap(),
            output.body_q.device_buffer_current().unwrap(),
        ),
        (
            input.body_qd.device_buffer_current().unwrap(),
            output.body_qd.device_buffer_current().unwrap(),
        ),
    ] {
        assert_eq!(read_floats(&engine, a), read_floats(&engine, b));
    }
    let undersized = buffer(device, queue, &[0; 4]);
    assert!(
        kernel
            .apply_position_displacement(
                device,
                queue,
                &model,
                &input,
                &control,
                &undersized,
                &mut output
            )
            .is_err()
    );
}

#[test]
#[ignore = "requires GPU"]
fn predictor_projection_and_redrift_reuse_all_six_state_bindings() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, mut a) = fixture_with_boundary_joint(true);
    let mut b = a.clone_host().unwrap();
    let mut projected = a.clone_host().unwrap();
    model.upload_to_device(device, queue).unwrap();
    for state in [&mut a, &mut b, &mut projected] {
        state.upload_to_device(device, queue).unwrap();
    }
    let mut control = model.control();
    control.upload_to_device(device, queue).unwrap();
    let mut delta = vec![0f32; model.joint_dof_total];
    delta[1] = 1e-5;
    let displacement = buffer(device, queue, bytemuck::cast_slice(&delta));
    let mut kernel = molla_kernels::FeatherstoneKernel::new(device, queue, &model);
    kernel
        .prepare_coordinate_compensation(device, &[&a, &b, &projected])
        .unwrap();
    let mut groups = 0;
    for cycle in 0..8 {
        kernel
            .dispatch(
                device,
                queue,
                &model,
                &a,
                &control,
                &molla_sim::Contacts::with_capacity(0),
                0.,
                0.001,
                true,
                false,
                &mut b,
                None,
            )
            .unwrap();
        let predicted_delta = read_floats(&engine, kernel.velocity_delta_buffer());
        assert!(predicted_delta.iter().any(|v| v.abs() > 1e-4));
        kernel
            .apply_position_displacement(
                device,
                queue,
                &model,
                &a,
                &control,
                &displacement,
                &mut projected,
            )
            .unwrap();
        kernel
            .drift(
                device, queue, &model, &projected, &control, 0.001, false, &mut b,
            )
            .unwrap();
        assert_eq!(
            read_floats(&engine, kernel.velocity_delta_buffer()),
            predicted_delta
        );
        if cycle == 1 {
            groups = kernel.bind_groups_created();
        }
        std::mem::swap(&mut a, &mut b);
    }
    assert_eq!(groups, 31);
    assert_eq!(kernel.bind_groups_created(), groups);
    assert_eq!(read_floats(&engine, kernel.status_buffer())[0].to_bits(), 0);
}

fn capture(
    engine: &Engine,
    response: &ResidentImpactPositionResponse,
    status: &wgpu::Buffer,
    extra: &[&wgpu::Buffer],
) -> Vec<Vec<u8>> {
    let mut sources = vec![
        status,
        response.displacement_buffer(),
        response.gram_buffer(),
        response.position_bias_buffer(),
        response.solution_buffer(),
        response.islands().row_slots_buffer(),
        response.islands().island_count_buffer(),
    ];
    sources.extend_from_slice(extra);
    let device = engine.device().unwrap();
    let out = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("position response test"),
            size: sources.iter().map(|b| b.size()).sum(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    response.encode(&mut encoder, 0).unwrap();
    let mut offset = 0;
    for source in &sources {
        encoder.copy_buffer_to_buffer(source, 0, &out, offset, source.size());
        offset += source.size();
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    out.slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = molla_core::gpu_readback!(out.slice(..));
    let mut offset = 0;
    sources
        .iter()
        .map(|b| {
            let end = offset + b.size() as usize;
            let data = mapped[offset..end].to_vec();
            offset = end;
            data
        })
        .collect()
}

fn normal_row(model: &Model, state: &State, c: ImpactContact, depth: f64) -> ImpactRow {
    ImpactRow {
        ends: c
            .bodies
            .iter()
            .enumerate()
            .filter_map(|(side, &body)| {
                body.map(|b| {
                    let pose = state.body_q.host().unwrap()[b.index()];
                    let arm =
                        c.point - pose.transform_point(model.body_com.host().unwrap()[b.index()]);
                    let normal = c.normal * if side == 0 { -1. } else { 1. };
                    (b, SpatialVector::new(normal, arm.cross(normal)))
                })
            })
            .collect(),
        direct: None,
        velocity_bias: -depth,
    }
}

fn config(rows: u32, blocks: u32) -> ImpactPositionConfig {
    ImpactPositionConfig {
        row_capacity: rows,
        island_capacity: 3,
        blocks_per_island: blocks,
        max_iterations: 8192,
        position_tolerance: 1e-8,
    }
}

#[test]
#[ignore = "requires GPU"]
fn onset_position_displacements_match_cpu_articulated_scalar_oracle() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, mut state) = fixture();
    let scalar = ImpactRow {
        ends: vec![
            (BodyId(0), SpatialVector::new(-Vec3::X, Vec3::ZERO)),
            (BodyId(2), SpatialVector::new(Vec3::X, Vec3::ZERO)),
        ],
        direct: None,
        velocity_bias: -0.003,
    };
    let jacobian = projection(&model, &state, &scalar);
    let scalar_rows = buffer(device, queue, &[0; 32]);
    let jacobians = buffer(
        device,
        queue,
        bytemuck::cast_slice(&jacobian.iter().map(|&v| v as f32).collect::<Vec<_>>()),
    );
    let rows = buffer(device, queue, &[0; 256]);
    let onsets = buffer(device, queue, &[0; 8]);
    let status = buffer(device, queue, &[0; 4]);
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let response = ResidentImpactPositionResponse::new(
        &engine,
        &model,
        &[&state],
        &rows,
        &[],
        Some(ScalarPositionBuffers {
            rows: &scalar_rows,
            jacobians: &jacobians,
            capacity: 1,
        }),
        &onsets,
        &status,
        config(2, 3),
    )
    .unwrap();
    let mut peak = 0f64;
    for mask in [[1u32, 1], [1, 0], [0, 1], [0, 0]] {
        for kind in [1u32, 2] {
            for reverse in [false, true] {
                let mut contacts = [
                    ImpactContact {
                        bodies: [Some(BodyId(3)), Some(BodyId(1))],
                        point: Vec3::new(-0.8, 0.3, 0.2),
                        normal: Vec3::Y,
                        static_friction: 0.8,
                        dynamic_friction: 0.4,
                        restitution: 0.9,
                        surface_velocity: [Vec3::new(5., 8., 3.), Vec3::ZERO],
                    },
                    ImpactContact {
                        bodies: [None, Some(BodyId(2))],
                        point: Vec3::new(1.2, -0.3, -0.15),
                        normal: Vec3::Y,
                        static_friction: 0.6,
                        dynamic_friction: 0.2,
                        restitution: 0.5,
                        surface_velocity: [Vec3::ZERO; 2],
                    },
                ];
                if reverse {
                    for c in &mut contacts {
                        c.bodies.swap(0, 1);
                        c.normal = -c.normal;
                    }
                }
                let depths = [0.002_f32, 0.004];
                let input: Vec<_> = contacts
                    .iter()
                    .enumerate()
                    .map(|(i, &c)| {
                        let mut r = row(&model, &state, c);
                        r.point_depth[3] = depths[i];
                        r
                    })
                    .collect();
                molla_core::gpu_write_buffer!(queue, &rows, 0, bytemuck::cast_slice(&input));
                molla_core::gpu_write_buffer!(queue, &onsets, 0, bytemuck::cast_slice(&mask));
                molla_core::gpu_write_buffer!(
                    queue,
                    &scalar_rows,
                    0,
                    bytemuck::bytes_of(&ScalarConstraintRow {
                        endpoints: [0, 2, kind, 1],
                        velocity: [scalar.velocity_bias as f32, 0., 0., 0.],
                    }),
                );
                let mut reference = state.clone_host().unwrap();
                reference.joint_qd.host_mut().unwrap().fill(0.);
                reference
                    .body_qd
                    .host_mut()
                    .unwrap()
                    .fill(SpatialVector::ZERO);
                let mut constraints: Vec<_> = contacts
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| mask[*i] != 0)
                    .map(|(i, &c)| {
                        ImpactConstraint::Nonnegative(normal_row(
                            &model,
                            &state,
                            c,
                            depths[i] as f64,
                        ))
                    })
                    .collect();
                if mask != [0, 0] {
                    constraints.push(if kind == 1 {
                        ImpactConstraint::Bilateral(scalar.clone())
                    } else {
                        ImpactConstraint::Nonnegative(scalar.clone())
                    });
                    apply_impact(&model, &mut reference, &constraints).unwrap();
                }
                let snapshot = capture(
                    &engine,
                    &response,
                    &status,
                    &[state.joint_qd.device_buffer().unwrap()],
                );
                assert_eq!(
                    bytemuck::pod_read_unaligned::<u32>(&snapshot[0]),
                    0,
                    "mask={mask:?} kind={kind} reverse={reverse}"
                );
                for (i, &v) in floats(&snapshot[1]).iter().enumerate() {
                    let error = (v as f64 - reference.joint_qd.host().unwrap()[i]).abs();
                    peak = peak.max(error);
                    assert!(error < 2e-7, "dof={i}, error={error}");
                }
                for (actual, &original) in floats(&snapshot[7])
                    .iter()
                    .zip(state.joint_qd.host().unwrap())
                {
                    assert_eq!(*actual, original as f32);
                }
                for (i, s) in snapshot[5].chunks_exact(8).take(2).enumerate() {
                    let slot = bytemuck::pod_read_unaligned::<[u32; 2]>(s);
                    assert_eq!(slot[0] == u32::MAX, mask[i] == 0);
                }
            }
        }
    }
    eprintln!("articulated position displacement peak error={peak:e}");
}

#[test]
#[ignore = "requires GPU"]
fn exported_position_gaps_survive_large_rates_and_zero_depth_onsets() {
    use crate::distance_gpu::DistanceGpu;
    use molla_math::{Mat3, Transform};
    use molla_sim::runtime::{BodyDesc, DistanceLimits, JointDesc, RigidScene};
    use molla_sim::{BodyParams, JointType, MassProperties};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for speed in [0., 1e8] {
        let mut scene = RigidScene::new().unwrap();
        let body = scene
            .insert_body(BodyDesc {
                params: BodyParams {
                    pose: Transform::from_translation(Vec3::X * 1.0001),
                    ..Default::default()
                },
                additional_mass: Some(MassProperties {
                    mass: 1.,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY,
                }),
                velocity: SpatialVector::new(Vec3::X * speed, Vec3::ZERO),
                ..Default::default()
            })
            .unwrap();
        let joint = scene
            .insert_joint(JointDesc {
                parent: None,
                child: body,
                kind: JointType::Distance,
                axis: Vec3::X,
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        scene
            .set_distance_limits(
                joint,
                DistanceLimits {
                    min: None,
                    max: Some(1.),
                },
            )
            .unwrap();
        scene.upload_model_to_device(device, queue).unwrap();
        let mut states = [
            scene.state().clone_host().unwrap(),
            scene.state().clone_host().unwrap(),
        ];
        for state in &mut states {
            state.upload_to_device(device, queue).unwrap();
        }
        let status = buffer(device, queue, &[0; 4]);
        let delta = buffer(device, queue, &vec![0; scene.model().joint_dof_total * 4]);
        let distance =
            DistanceGpu::new_for_coupling(&engine, &scene, &states, &status, &delta, None)
                .unwrap()
                .unwrap();
        let mut contact = NormalContactRow {
            point_depth: [1.0001, 0., 0., 0.002],
            normal_force: [0., 1., 0., 1000.],
            coefficients: [0.8, 0.2, 0.9, 1.],
            endpoints: [0, 1, u32::MAX, scene.body_index(body).unwrap().0],
            relative_velocity: [speed as f32, 0., 0., 0.],
            ..Default::default()
        };
        let rows = buffer(device, queue, bytemuck::bytes_of(&contact));
        let onsets = buffer(device, queue, bytemuck::bytes_of(&1u32));
        let response = ResidentImpactPositionResponse::new(
            &engine,
            scene.model(),
            &[&states[0]],
            &rows,
            &[],
            distance.scalar_position_buffers(),
            &onsets,
            &status,
            config(1, 2),
        )
        .unwrap();
        for depth in [0.002_f32, 0.] {
            contact.point_depth[3] = depth;
            molla_core::gpu_write_buffer!(queue, &rows, 0, bytemuck::bytes_of(&contact));
            for dt in [0.001, 0.02] {
                distance.prepare_scalar_rows(&engine, 0, dt, 1, 0).unwrap();
                let snapshot = capture(
                    &engine,
                    &response,
                    &status,
                    &[
                        distance.scalar_position_buffers().unwrap().rows,
                        distance.scalar_buffers().unwrap().rows,
                        states[0].joint_qd.device_buffer().unwrap(),
                        &rows,
                    ],
                );
                assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[0]), 0);
                let bias = bytemuck::pod_read_unaligned::<ScalarConstraintRow>(&snapshot[7]);
                let expected_gap = 1.0001_f32 - 1.;
                assert_eq!(bias.endpoints[2..], [2, 1]);
                assert_eq!(bias.velocity, [-expected_gap, 0., 0., 0.]);
                let displacement = floats(&snapshot[1]);
                assert!((displacement[0] + expected_gap).abs() < 1e-8);
                assert!((displacement[1] - depth).abs() < 1e-8);
                assert!(displacement[2..].iter().all(|&v| v.abs() < 1e-8));
                if speed != 0. {
                    let velocity =
                        bytemuck::pod_read_unaligned::<ScalarConstraintRow>(&snapshot[8]);
                    assert_eq!(velocity.velocity[0], -speed as f32);
                }
                assert_eq!(floats(&snapshot[9])[0], speed as f32);
                assert_eq!(snapshot[10], bytemuck::bytes_of(&contact));
            }
        }
        molla_core::gpu_write_buffer!(queue, &onsets, 0, bytemuck::bytes_of(&0u32));
        let snapshot = capture(&engine, &response, &status, &[]);
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[0]), 0);
        assert!(floats(&snapshot[1]).iter().all(|&v| v == 0.));
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[6]), 0);
    }
}

#[test]
#[ignore = "requires GPU"]
fn generated_distance_cone_and_axis_position_rows_match_cpu_geometry() {
    use crate::distance_constraints::{self, DistanceBound};
    use crate::distance_gpu::DistanceGpu;
    use molla_math::{Mat3, Quat, Transform};
    use molla_sim::joint_control::JointAxis;
    use molla_sim::runtime::{
        BodyDesc, DistanceLimits, JointDesc, RigidScene, SphericalConeLimits,
    };
    use molla_sim::{JointType, MassProperties};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for case in 0..7 {
        let mut scene = RigidScene::new().unwrap();
        let body = scene
            .insert_body(BodyDesc {
                additional_mass: Some(MassProperties {
                    mass: 1.,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY,
                }),
                ..Default::default()
            })
            .unwrap();
        let kind = match case {
            4 => JointType::Ball,
            5 => JointType::Prismatic,
            6 => JointType::Revolute,
            _ => JointType::Distance,
        };
        let joint = scene
            .insert_joint(JointDesc {
                parent: None,
                child: body,
                kind,
                axis: if case == 6 { Vec3::Y } else { Vec3::X },
                frame_parent: Transform::IDENTITY,
                frame_child: Transform::IDENTITY,
            })
            .unwrap();
        if case < 4 {
            scene
                .set_distance_limits(
                    joint,
                    match case {
                        0 => DistanceLimits {
                            min: Some(1.),
                            max: Some(1.),
                        },
                        1 => DistanceLimits {
                            min: Some(1.002),
                            max: None,
                        },
                        2 => DistanceLimits {
                            min: None,
                            max: Some(1.),
                        },
                        _ => DistanceLimits {
                            min: None,
                            max: Some(0.),
                        },
                    },
                )
                .unwrap();
        } else if case == 4 {
            scene
                .set_cone_limits(
                    joint,
                    SphericalConeLimits {
                        angle0: Some(0.),
                        angle1: Some(0.),
                    },
                )
                .unwrap();
        } else {
            scene
                .set_joint_limits(
                    joint,
                    if case == 5 {
                        JointAxis::LinX
                    } else {
                        JointAxis::AngX
                    },
                    Some([-1., 0.]),
                )
                .unwrap();
        }
        let pose = match case {
            0..=2 => Transform::from_translation(Vec3::X * 1.001),
            3 => Transform::from_translation(Vec3::new(0.001, -0.002, 0.003)),
            5 => Transform::from_translation(Vec3::X * 0.001),
            _ => Transform {
                position: Vec3::ZERO,
                rotation: Quat::from_rotation_y(0.2),
            },
        };
        scene.set_body_pose(body, pose).unwrap();
        scene
            .set_body_velocity(body, SpatialVector::new(Vec3::X * 1e8, Vec3::Y * 1e8))
            .unwrap();
        scene.upload_model_to_device(device, queue).unwrap();
        let mut states = [
            scene.state().clone_host().unwrap(),
            scene.state().clone_host().unwrap(),
        ];
        let mut reference = states[0].clone_host().unwrap();
        reference.joint_qd.host_mut().unwrap().fill(0.);
        reference
            .body_qd
            .host_mut()
            .unwrap()
            .fill(SpatialVector::ZERO);
        let (groups, _) = distance_constraints::prepare(&scene, &reference, 1., None).unwrap();
        let mut cpu_rows: Vec<_> = groups.into_values().flatten().collect();
        distance_constraints::scalar_rows(
            scene.model(),
            &reference,
            &(0..scene.model().articulation_count).collect::<Vec<_>>(),
            1.,
            &mut cpu_rows,
        )
        .unwrap();
        let constraints: Vec<_> = cpu_rows
            .into_iter()
            .map(|r| {
                let row = ImpactRow {
                    ends: r
                        .ends
                        .into_iter()
                        .map(|(b, w)| (BodyId::from_index(b), w))
                        .collect(),
                    direct: r.direct,
                    velocity_bias: -r.target,
                };
                if r.bound == DistanceBound::Equality {
                    ImpactConstraint::Bilateral(row)
                } else {
                    ImpactConstraint::Nonnegative(row)
                }
            })
            .collect();
        apply_impact(scene.model(), &mut reference, &constraints).unwrap();
        assert!(
            reference
                .joint_qd
                .host()
                .unwrap()
                .iter()
                .any(|v| v.abs() > 1e-5),
            "case={case}"
        );
        for state in &mut states {
            state.upload_to_device(device, queue).unwrap();
        }
        let status = buffer(device, queue, &[0; 4]);
        let delta = buffer(device, queue, &vec![0; scene.model().joint_dof_total * 4]);
        let distance =
            DistanceGpu::new_for_coupling(&engine, &scene, &states, &status, &delta, None)
                .unwrap()
                .unwrap();
        let normal = buffer(
            device,
            queue,
            bytemuck::bytes_of(&NormalContactRow {
                point_depth: [
                    pose.position.x as f32,
                    pose.position.y as f32,
                    pose.position.z as f32,
                    0.,
                ],
                normal_force: [0., 1., 0., 0.],
                coefficients: [0., 0., 0., 1.],
                endpoints: [0, 1, u32::MAX, scene.body_index(body).unwrap().0],
                ..Default::default()
            }),
        );
        let onsets = buffer(device, queue, bytemuck::bytes_of(&1u32));
        let response = ResidentImpactPositionResponse::new(
            &engine,
            scene.model(),
            &[&states[0]],
            &normal,
            &[],
            distance.scalar_position_buffers(),
            &onsets,
            &status,
            config(1, 4),
        )
        .unwrap();
        distance
            .prepare_scalar_rows(&engine, 0, 0.013, 1, 0)
            .unwrap();
        let snapshot = capture(&engine, &response, &status, &[]);
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&snapshot[0]),
            0,
            "case={case}"
        );
        for (i, (&actual, &expected)) in floats(&snapshot[1])
            .iter()
            .zip(reference.joint_qd.host().unwrap())
            .enumerate()
        {
            assert!(
                (actual as f64 - expected).abs() < 2e-6,
                "case={case} dof={i} actual={actual} expected={expected}"
            );
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn redundant_position_rows_are_mass_scaled_and_faults_retire_displacements() {
    use crate::impact_constraints_gpu::ImpactGpuSolution;
    use molla_math::{Mat3, SpatialInertia, Transform};
    use molla_sim::{BodyParams, ModelBuilder};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for scale in [1e-6, 1., 1e6] {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: scale,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY * scale,
        }));
        builder
            .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let mut model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        state.joint_qd.host_mut().unwrap()[1] = 1000.;
        model.upload_to_device(device, queue).unwrap();
        state.upload_to_device(device, queue).unwrap();
        let mut contacts = [NormalContactRow {
            point_depth: [0., 0., 0., 0.002],
            normal_force: [0., 1., 0., 0.],
            coefficients: [0., 0., 1., 1.],
            endpoints: [0, 1, u32::MAX, 0],
            relative_velocity: [0., 1000., 0., 0.],
            ..Default::default()
        }; 2];
        contacts[1].point_depth[3] = 0.004;
        let rows = buffer(device, queue, bytemuck::cast_slice(&contacts));
        let onsets = buffer(device, queue, bytemuck::cast_slice(&[1u32, 1]));
        let status = buffer(device, queue, &[0; 4]);
        assert!(
            ResidentImpactPositionResponse::new(
                &engine,
                &model,
                &[&state],
                &rows,
                &[],
                None,
                &onsets,
                &status,
                ImpactPositionConfig {
                    position_tolerance: f32::NAN,
                    ..config(2, 2)
                }
            )
            .is_err()
        );
        let response = ResidentImpactPositionResponse::new(
            &engine,
            &model,
            &[&state],
            &rows,
            &[],
            None,
            &onsets,
            &status,
            config(2, 2),
        )
        .unwrap();
        let snapshot = capture(&engine, &response, &status, &[]);
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&snapshot[0]),
            0,
            "scale={scale}"
        );
        assert!((floats(&snapshot[1])[1] - 0.004).abs() < 1e-8);
        let solutions: Vec<ImpactGpuSolution> = snapshot[4]
            .chunks_exact(48)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        assert_eq!(solutions[0].impulse[0], 0.);
        assert!((solutions[1].impulse[0] as f64 / scale - 0.004).abs() < 1e-8);
        assert!((solutions[0].velocity[0] - 0.002).abs() < 1e-8);
        for bad_mask in [[2u32, 1], [1, 1]] {
            if bad_mask[0] == 1 {
                contacts[0].point_depth[3] = -0.1;
            }
            molla_core::gpu_write_buffer!(queue, &onsets, 0, bytemuck::cast_slice(&bad_mask));
            molla_core::gpu_write_buffer!(queue, &rows, 0, bytemuck::cast_slice(&contacts));
            molla_core::gpu_write_buffer!(queue, &status, 0, bytemuck::bytes_of(&0u32));
            let snapshot = capture(&engine, &response, &status, &[]);
            assert_eq!(
                bytemuck::pod_read_unaligned::<u32>(&snapshot[0]),
                CONTACT_RESPONSE_FAULT_INPUT
            );
            assert!(floats(&snapshot[1]).iter().all(|&v| v == 0.));
        }
        contacts[0].point_depth[3] = 0.002;
        molla_core::gpu_write_buffer!(queue, &rows, 0, bytemuck::cast_slice(&contacts));
        let halted = capture(&engine, &response, &status, &[]);
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&halted[0]),
            CONTACT_RESPONSE_FAULT_INPUT
        );
        molla_core::gpu_write_buffer!(queue, &status, 0, bytemuck::bytes_of(&0u32));
        molla_core::gpu_write_buffer!(queue, &onsets, 0, bytemuck::cast_slice(&[0u32, 0]));
        let empty = capture(&engine, &response, &status, &[]);
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&empty[0]), 0);
        assert!(floats(&empty[1]).iter().all(|&v| v == 0.));
    }
}
