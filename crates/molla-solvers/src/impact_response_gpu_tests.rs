use super::*;
use crate::articulated_response::ArticulatedResponse;
use crate::constraint_rows_gpu::ScalarConstraintRow;
use crate::coulomb_disk_gpu::CoulombDiskSolution;
use crate::normal_contact_gpu::NormalContactRow;
use crate::rigid_impulse::{ImpactConstraint, ImpactContact, ImpactRow, apply_impact};
use molla_core::BodyId;
use molla_math::{Mat3, Quat, SpatialInertia, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder, eval_body_velocities, eval_fk};

pub(super) fn buffer(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> wgpu::Buffer {
    let b = storage(device, "impact response input", data.len().max(4) as u64);
    if !data.is_empty() {
        molla_core::gpu_write_buffer!(queue, &b, 0, data);
    }
    b
}

fn capture(
    engine: &Engine,
    response: &ResidentImpactResponse,
    status: &wgpu::Buffer,
) -> Vec<Vec<u8>> {
    capture_after(engine, response, status, |_| {})
}

fn capture_after(
    engine: &Engine,
    response: &ResidentImpactResponse,
    status: &wgpu::Buffer,
    prepare: impl FnOnce(&mut wgpu::CommandEncoder),
) -> Vec<Vec<u8>> {
    let sources = [
        status,
        response.corrected_rates_buffer(),
        response.velocity_delta_buffer(),
        response.world_impulse_buffer(),
        response.scalar_solution_buffer(),
        response.gram_buffer(),
        response.islands().row_slots_buffer(),
        response.impacts().diagnostic_buffer(),
        response.islands().island_count_buffer(),
        response.free_velocity_buffer(),
        &response.core.predicted_contacts,
    ];
    let device = engine.device().unwrap();
    let snapshot = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("impact response snapshot"),
            size: sources.iter().map(|b| b.size()).sum(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    prepare(&mut encoder);
    response.encode(&mut encoder, 0).unwrap();
    let mut offset = 0;
    for source in &sources {
        encoder.copy_buffer_to_buffer(source, 0, &snapshot, offset, source.size());
        offset += source.size();
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    snapshot
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = molla_core::gpu_readback!(snapshot.slice(..));
    let mut offset = 0;
    sources
        .iter()
        .map(|source| {
            let size = source.size() as usize;
            let data = bytes[offset..offset + size].to_vec();
            offset += size;
            data
        })
        .collect()
}

pub(super) fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}
fn packed(v: Vec3, w: f32) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w]
}

pub(super) fn fixture() -> (Model, State) {
    fixture_with_boundary_joint(false)
}

pub(super) fn fixture_with_boundary_joint(articulated_boundary: bool) -> (Model, State) {
    let mut builder = ModelBuilder::new();
    let mut bodies = Vec::new();
    for i in 0..5 {
        bodies.push(
            builder.add_body(
                BodyParams::new()
                    .with_position(Vec3::X * (i as f64 - 1.))
                    .with_spatial_inertia(SpatialInertia {
                        mass: 1. + i as f64 * 0.25,
                        com: Vec3::new(0.1, -0.05, 0.08),
                        inertia: Mat3::from_cols(
                            Vec3::new(0.3, 0.02, 0.),
                            Vec3::new(0.02, 0.4, 0.01),
                            Vec3::new(0., 0.01, 0.5),
                        ),
                    })
                    .with_flags(if i == 3 {
                        BodyFlags::KINEMATIC
                    } else {
                        BodyFlags::DYNAMIC
                    }),
            ),
        );
    }
    builder
        .add_joint_free(
            BodyId::NONE,
            bodies[0],
            Transform::from_translation(-Vec3::X),
            Transform::IDENTITY,
        )
        .unwrap();
    builder
        .add_joint_revolute(
            bodies[0],
            bodies[1],
            Transform::from_translation(Vec3::Y * 0.6),
            Transform::IDENTITY,
            Vec3::Z,
        )
        .unwrap();
    builder.begin_articulation();
    builder
        .add_joint_free(
            BodyId::NONE,
            bodies[2],
            Transform::from_translation(Vec3::X),
            Transform::IDENTITY,
        )
        .unwrap();
    builder.begin_articulation();
    builder
        .add_joint_free(
            BodyId::NONE,
            bodies[4],
            Transform::from_translation(Vec3::X * 5.),
            Transform::IDENTITY,
        )
        .unwrap();
    builder
        .set_kinematic_acceleration(
            bodies[3],
            Some(SpatialVector::new(
                Vec3::new(0.2, 0.3, -0.1),
                Vec3::new(0.1, -0.05, 0.2),
            )),
        )
        .unwrap();
    if articulated_boundary {
        builder.begin_articulation();
        builder
            .add_joint_fixed(
                BodyId::NONE,
                bodies[3],
                Transform::from_translation(Vec3::X * 2.),
                Transform::IDENTITY,
            )
            .unwrap();
    }
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state.joint_q.host_mut().unwrap()[3..7].copy_from_slice(&Quat::from_rotation_y(0.2).to_array());
    state.joint_qd.host_mut().unwrap()[1] = -1.;
    state.joint_qd.host_mut().unwrap()[6] = 0.1;
    state.joint_qd.host_mut().unwrap()[8] = -0.7;
    state.joint_qd.host_mut().unwrap()[13] = 0.25;
    eval_fk(&model, &mut state).unwrap();
    state.body_qd.host_mut().unwrap()[3] =
        SpatialVector::new(Vec3::new(0.3, 0.1, 0.2), Vec3::new(0., 0.05, 0.));
    eval_body_velocities(&model, &mut state).unwrap();
    (model, state)
}

pub(super) fn row(model: &Model, state: &State, contact: ImpactContact) -> NormalContactRow {
    let mut relative = Vec3::ZERO;
    for side in 0..2 {
        if let Some(body) = contact.bodies[side] {
            let i = body.index();
            let pose = state.body_q.host().unwrap()[i];
            let arm = contact.point - pose.transform_point(model.body_com.host().unwrap()[i]);
            let v = state.body_qd.host().unwrap()[i];
            relative += (v.linear + v.angular.cross(arm)) * if side == 0 { -1. } else { 1. };
        }
    }
    NormalContactRow {
        point_depth: packed(contact.point, 0.001),
        normal_force: packed(contact.normal, 0.),
        coefficients: [
            contact.static_friction as f32,
            contact.dynamic_friction as f32,
            contact.restitution as f32,
            1.,
        ],
        endpoints: [
            1000,
            2000,
            contact.bodies[0].map_or(u32::MAX, |b| b.index() as u32),
            contact.bodies[1].map_or(u32::MAX, |b| b.index() as u32),
        ],
        relative_velocity: packed(relative, 0.),
        surface_velocity: packed(
            contact.surface_velocity[1] - contact.surface_velocity[0],
            0.,
        ),
        ..Default::default()
    }
}

pub(super) fn projection(model: &Model, state: &State, row: &ImpactRow) -> Vec<f64> {
    let mut response = ArticulatedResponse::new(model, state);
    let starts = model.articulation_start.host().unwrap();
    let offsets = model.joint_dof_offset.host().unwrap();
    let mut result = vec![0.; model.joint_dof_total];
    for &(body, wrench) in &row.ends {
        let art = match body.index() {
            0 | 1 => 0,
            2 => 1,
            4 => 2,
            _ => continue,
        };
        let first = offsets[starts[art] as usize] as usize;
        let n = offsets[starts[art + 1] as usize] as usize - first;
        let j = response.jacobian(art, body.index()).unwrap();
        for k in 0..n {
            result[first + k] += wrench.linear.dot(Vec3::new(j[k], j[n + k], j[2 * n + k]))
                + wrench
                    .angular
                    .dot(Vec3::new(j[3 * n + k], j[4 * n + k], j[5 * n + k]));
        }
    }
    result
}

fn gram_reference(
    model: &Model,
    state: &State,
    contacts: &[ImpactContact],
    scalar: Option<&ImpactRow>,
) -> Vec<f64> {
    let mut rows = Vec::new();
    for contact in contacts {
        let n = contact.normal;
        let t = n
            .cross(if n.x.abs() < 0.6 { Vec3::X } else { Vec3::Y })
            .normalize();
        for direction in [n, t, n.cross(t)] {
            let mut row = ImpactRow::default();
            for side in 0..2 {
                if let Some(body) = contact.bodies[side] {
                    let arm = contact.point
                        - state.body_q.host().unwrap()[body.index()]
                            .transform_point(model.body_com.host().unwrap()[body.index()]);
                    let force = direction * if side == 0 { -1. } else { 1. };
                    row.ends
                        .push((body, SpatialVector::new(force, arm.cross(force))));
                }
            }
            rows.push(projection(model, state, &row));
        }
    }
    if let Some(scalar) = scalar {
        rows.push(projection(model, state, scalar));
        rows.push(vec![0.; model.joint_dof_total]);
        rows.push(vec![0.; model.joint_dof_total]);
    }
    let mut gram = vec![0.; rows.len() * rows.len()];
    let mut response = ArticulatedResponse::new(model, state);
    let starts = model.articulation_start.host().unwrap();
    let offsets = model.joint_dof_offset.host().unwrap();
    for art in 0..model.articulation_count {
        let first = offsets[starts[art] as usize] as usize;
        let end = offsets[starts[art + 1] as usize] as usize;
        let projected = rows
            .iter()
            .map(|r| r[first..end].to_vec())
            .collect::<Vec<_>>();
        response
            .add_gram(art, end - first, &projected, &mut gram)
            .unwrap();
    }
    gram
}

#[test]
#[ignore = "requires GPU"]
fn articulated_impacts_match_cpu_with_joint_rows_boundaries_and_endpoint_exchange() {
    articulated_impact_cases(false);
}

#[test]
#[ignore = "requires GPU"]
fn predicted_selection_matches_cpu_with_articulation_and_accelerating_boundaries() {
    articulated_impact_cases(true);
}

fn articulated_impact_cases(select_predicted: bool) {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut model, mut state) = fixture();
    let predicted: Vec<_> = state
        .joint_qd
        .host()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(i, &v)| (v + ((i * 3 % 7) as f64 - 3.) * 0.001) as f32)
        .collect();
    let mut reference_input = state.clone_host().unwrap();
    reference_input
        .joint_qd
        .host_mut()
        .unwrap()
        .copy_from_slice(&predicted.iter().map(|&v| v as f64).collect::<Vec<_>>());
    let dt_value = 0.01_f32;
    reference_input.body_qd.host_mut().unwrap()[3] = state.body_qd.host().unwrap()[3]
        + model.body_kinematic_acceleration.host().unwrap()[3] * dt_value as f64;
    eval_body_velocities(&model, &mut reference_input).unwrap();
    let scalar = ImpactRow {
        ends: vec![
            (
                BodyId::from_index(0),
                SpatialVector::new(-Vec3::X, Vec3::ZERO),
            ),
            (
                BodyId::from_index(2),
                SpatialVector::new(Vec3::X, Vec3::ZERO),
            ),
        ],
        velocity_bias: -0.15,
        direct: None,
    };
    let jacobian = projection(&model, &state, &scalar);
    let scalar_velocity = scalar.velocity_bias
        + jacobian
            .iter()
            .zip(state.joint_qd.host().unwrap())
            .map(|(a, b)| a * b)
            .sum::<f64>();
    let scalar_row = buffer(device, queue, &[0; 32]);
    let jacobians = buffer(
        device,
        queue,
        bytemuck::cast_slice(&jacobian.iter().map(|&v| v as f32).collect::<Vec<_>>()),
    );
    let rows = buffer(device, queue, &[0; 512]);
    let predictions = buffer(device, queue, bytemuck::cast_slice(&predicted));
    let dt = buffer(device, queue, bytemuck::bytes_of(&dt_value));
    let status = buffer(device, queue, &[0; 4]);
    let onsets = buffer(device, queue, bytemuck::cast_slice(&[1u32, 1, 0, 0]));
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let response = ResidentImpactResponse::build(
        &engine,
        &model,
        &[ContactResponseState {
            input: &state,
            predicted_rates: &predictions,
        }],
        &rows,
        &[],
        Some(ScalarConstraintBuffers {
            rows: &scalar_row,
            jacobians: &jacobians,
            capacity: 1,
        }),
        None,
        select_predicted.then_some(&onsets),
        &dt,
        &status,
        ImpactResponseConfig {
            row_capacity: 4,
            impacts: ImpactGpuConfig {
                island_capacity: 3,
                blocks_per_island: 4,
                max_iterations: 8192,
                velocity_tolerance: 2e-5,
            },
        },
    )
    .unwrap();
    let mut peak: f64 = 0.;
    let mut peak_gram: f64 = 0.;
    for restitution in [0., 0.5, 1.] {
        for friction in [0., 0.4] {
            for reverse in [false, true] {
                for kind in [0, 1, 2, 3] {
                    let mut contacts = [
                        ImpactContact {
                            bodies: [Some(BodyId::from_index(3)), Some(BodyId::from_index(1))],
                            point: Vec3::new(-0.8, 0.3, 0.2),
                            normal: Vec3::Y,
                            static_friction: friction * 1.5,
                            dynamic_friction: friction,
                            restitution,
                            surface_velocity: [Vec3::new(0.2, 3., -0.1), Vec3::ZERO],
                        },
                        ImpactContact {
                            bodies: [None, Some(BodyId::from_index(2))],
                            point: Vec3::new(1.2, -0.3, -0.15),
                            normal: Vec3::Y,
                            static_friction: friction * 1.5,
                            dynamic_friction: friction,
                            restitution,
                            surface_velocity: [Vec3::ZERO; 2],
                        },
                    ];
                    if reverse {
                        for contact in &mut contacts {
                            contact.bodies.swap(0, 1);
                            contact.normal = -contact.normal;
                            contact.surface_velocity.swap(0, 1);
                        }
                    }
                    let mut input = contacts
                        .iter()
                        .map(|&c| row(&model, &state, c))
                        .collect::<Vec<_>>();
                    input.resize(4, NormalContactRow::default());
                    molla_core::gpu_write_buffer!(queue, &rows, 0, bytemuck::cast_slice(&input));
                    let gpu_scalar = if kind == 0 {
                        ScalarConstraintRow::default()
                    } else {
                        ScalarConstraintRow {
                            endpoints: [0, 2, kind, 1],
                            velocity: if kind == 3 {
                                [scalar_velocity as f32, 0.7, -0.2, 0.3]
                            } else {
                                [scalar_velocity as f32, 0., 0., 0.]
                            },
                        }
                    };
                    molla_core::gpu_write_buffer!(
                        queue,
                        &scalar_row,
                        0,
                        bytemuck::bytes_of(&gpu_scalar)
                    );
                    let mut constraints = contacts
                        .into_iter()
                        .map(ImpactConstraint::Contact)
                        .collect::<Vec<_>>();
                    if kind != 0 {
                        constraints.push(if kind == 1 {
                            ImpactConstraint::Bilateral(scalar.clone())
                        } else if kind == 2 {
                            ImpactConstraint::Nonnegative(scalar.clone())
                        } else {
                            ImpactConstraint::Bounded {
                                row: scalar.clone(),
                                impulses: [-0.2, 0.3],
                                response_diagonal: 0.7,
                            }
                        });
                    }
                    let mut reference = reference_input.clone_host().unwrap();
                    let expected = apply_impact(&model, &mut reference, &constraints).unwrap();
                    let snapshot = capture(&engine, &response, &status);
                    assert_eq!(
                        bytemuck::pod_read_unaligned::<u32>(&snapshot[0]),
                        0,
                        "e={restitution}, mu={friction}, reverse={reverse}, kind={kind}, diagnostics={:?}",
                        floats(&snapshot[7])
                    );
                    if select_predicted {
                        for (i, contact) in contacts.iter().enumerate() {
                            let data = bytemuck::pod_read_unaligned::<[u32; 2]>(
                                &snapshot[10][8 * i..8 * i + 8],
                            );
                            assert_eq!(data[0], 1);
                            let speed = row(&model, &reference_input, *contact).relative_velocity;
                            let normal = contact.normal;
                            let expected = speed[0] as f64 * normal.x
                                + speed[1] as f64 * normal.y
                                + speed[2] as f64 * normal.z;
                            assert!((f32::from_bits(data[1]) as f64 - expected).abs() < 3e-6);
                        }
                    }
                    for (i, &actual) in floats(&snapshot[1]).iter().enumerate() {
                        let error = (actual as f64 - reference.joint_qd.host().unwrap()[i]).abs();
                        peak = peak.max(error);
                        assert!(
                            error < 8e-5,
                            "dof={i}, error={error}, e={restitution}, mu={friction}, reverse={reverse}, kind={kind}"
                        );
                    }
                    for (i, raw) in snapshot[3].chunks_exact(16).take(2).enumerate() {
                        let actual = floats(raw);
                        let actual =
                            Vec3::new(actual[0] as f64, actual[1] as f64, actual[2] as f64);
                        assert!((actual - expected.contact_impulses[i].unwrap()).length() < 1e-4);
                    }
                    let actual =
                        bytemuck::pod_read_unaligned::<CoulombDiskSolution>(&snapshot[4][..32]);
                    if kind == 0 {
                        assert_eq!(actual.impulse, [0.; 2]);
                    } else {
                        assert!(
                            (actual.impulse[0] as f64 - expected.solution.impulses[6]).abs() < 1e-4
                        );
                        assert_eq!(actual.regime, 3 + kind);
                    }
                    let island_count = bytemuck::pod_read_unaligned::<u32>(&snapshot[8]);
                    assert_eq!(island_count, if kind == 0 { 2 } else { 1 });
                    if kind != 0 {
                        let expected = gram_reference(&model, &state, &contacts, Some(&scalar));
                        let actual = floats(&snapshot[5]);
                        for i in 0..9 {
                            for j in 0..9 {
                                let error = (actual[i * 12 + j] as f64 - expected[i * 9 + j]).abs();
                                peak_gram = peak_gram.max(error);
                                assert!(
                                    error < 6e-5 * expected[i * 9 + j].abs().max(1.),
                                    "gram[{i},{j}] error={error}"
                                );
                            }
                        }
                    }
                    for i in 13..19 {
                        assert_eq!(floats(&snapshot[1])[i], predicted[i]);
                    }
                }
            }
        }
    }
    eprintln!(
        "articulated impact response peak rate error={peak:e}, peak Gram error={peak_gram:e}"
    );
}

#[test]
#[ignore = "requires GPU"]
fn zero_friction_zero_penalty_predictor_change_and_fault_retirement() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 2.,
        com: Vec3::ZERO,
        inertia: Mat3::IDENTITY,
    }));
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state.joint_qd.host_mut().unwrap()[1] = -1.;
    eval_fk(&model, &mut state).unwrap();
    eval_body_velocities(&model, &mut state).unwrap();
    let contact = ImpactContact {
        bodies: [None, Some(body)],
        point: Vec3::ZERO,
        normal: Vec3::Y,
        static_friction: 0.,
        dynamic_friction: 0.,
        restitution: 0.5,
        surface_velocity: [Vec3::ZERO; 2],
    };
    let active = row(&model, &state, contact);
    let rows = buffer(device, queue, bytemuck::bytes_of(&active));
    let predicted = buffer(
        device,
        queue,
        bytemuck::cast_slice(&[0f32, -1., 0., 0., 0., 0.]),
    );
    let dt = buffer(device, queue, bytemuck::bytes_of(&0.01_f32));
    let status = buffer(device, queue, &[0; 4]);
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let response = ResidentImpactResponse::new(
        &engine,
        &model,
        &[ContactResponseState {
            input: &state,
            predicted_rates: &predicted,
        }],
        &rows,
        &[],
        None,
        &dt,
        &status,
        ImpactResponseConfig {
            row_capacity: 1,
            impacts: ImpactGpuConfig {
                island_capacity: 1,
                blocks_per_island: 1,
                max_iterations: 4096,
                velocity_tolerance: 2e-5,
            },
        },
    )
    .unwrap();
    for incoming in [-1f32, -2.] {
        molla_core::gpu_write_buffer!(
            queue,
            &predicted,
            0,
            bytemuck::cast_slice(&[0f32, incoming, 0., 0., 0., 0.]),
        );
        let snapshot = capture(&engine, &response, &status);
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[0]), 0);
        assert!((floats(&snapshot[1])[1] + 0.5 * incoming).abs() < 2e-5);
        assert!((floats(&snapshot[3])[1] + 3. * incoming).abs() < 4e-5);
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[8]), 1);
    }
    molla_core::gpu_write_buffer!(
        queue,
        &rows,
        0,
        bytemuck::bytes_of(&NormalContactRow::default())
    );
    let snapshot = capture(&engine, &response, &status);
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[0]), 0);
    assert_eq!(floats(&snapshot[1]), [0., -2., 0., 0., 0., 0.]);
    assert!(floats(&snapshot[3]).iter().all(|&v| v == 0.));
    molla_core::gpu_write_buffer!(queue, &rows, 0, bytemuck::bytes_of(&active));
    molla_core::gpu_write_buffer!(
        queue,
        &predicted,
        0,
        bytemuck::cast_slice(&[0f32, 1., 0., 0., 0., 0.]),
    );
    let snapshot = capture(&engine, &response, &status);
    assert_eq!(
        bytemuck::pod_read_unaligned::<u32>(&snapshot[0]),
        crate::impact_constraints_gpu::IMPACT_FAULT_INPUT
    );
    assert!(
        floats(&snapshot[1])
            .iter()
            .chain(floats(&snapshot[3]).iter())
            .all(|&v| v == 0.)
    );
    molla_core::gpu_write_buffer!(
        queue,
        &predicted,
        0,
        bytemuck::cast_slice(&[0f32, -1., 0., 0., 0., 0.]),
    );
    let snapshot = capture(&engine, &response, &status);
    assert_eq!(
        bytemuck::pod_read_unaligned::<u32>(&snapshot[0]),
        crate::impact_constraints_gpu::IMPACT_FAULT_INPUT
    );
    molla_core::gpu_write_buffer!(queue, &status, 0, bytemuck::bytes_of(&0u32));
    let snapshot = capture(&engine, &response, &status);
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[0]), 0);
    assert!((floats(&snapshot[1])[1] - 0.5).abs() < 2e-5);
    let mut immovable = active;
    immovable.endpoints[3] = u32::MAX;
    molla_core::gpu_write_buffer!(queue, &rows, 0, bytemuck::bytes_of(&immovable));
    let snapshot = capture(&engine, &response, &status);
    assert_eq!(
        bytemuck::pod_read_unaligned::<u32>(&snapshot[0]),
        crate::contact_islands_gpu::CONTACT_ISLAND_FAULT_INPUT
    );
}

#[test]
#[ignore = "requires GPU"]
fn retained_predictor_increment_survives_large_common_normal_velocity() {
    retained_predictor_cases(false);
}

#[test]
#[ignore = "requires GPU"]
fn predicted_selection_retains_sub_ulp_approach_and_retires_separating_rows() {
    retained_predictor_cases(true);
}

fn retained_predictor_cases(select_predicted: bool) {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let support = builder.add_body(BodyParams::new().with_flags(BodyFlags::KINEMATIC));
    let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.,
        com: Vec3::ZERO,
        inertia: Mat3::IDENTITY,
    }));
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    let base = 1048576_f32;
    let increment = -0.004_f32;
    assert_eq!(base + increment, base);
    state.joint_qd.host_mut().unwrap()[1] = base as f64;
    state.body_qd.host_mut().unwrap()[support.index()] =
        SpatialVector::new(Vec3::Y * (base as f64 + 0.01), Vec3::ZERO);
    eval_fk(&model, &mut state).unwrap();
    eval_body_velocities(&model, &mut state).unwrap();
    let contact = ImpactContact {
        bodies: [Some(support), Some(body)],
        point: Vec3::ZERO,
        normal: Vec3::Y,
        static_friction: 0.,
        dynamic_friction: 0.,
        restitution: 0.5,
        surface_velocity: [Vec3::ZERO; 2],
    };
    let active = row(&model, &state, contact);
    let rows = buffer(device, queue, bytemuck::bytes_of(&active));
    let predicted = buffer(
        device,
        queue,
        bytemuck::cast_slice(&[0f32, base, 0., 0., 0., 0.]),
    );
    let delta = buffer(
        device,
        queue,
        bytemuck::cast_slice(&[0f32, increment, 0., 0., 0., 0.]),
    );
    let dt = buffer(device, queue, bytemuck::bytes_of(&0.01_f32));
    let status = buffer(device, queue, &[0; 4]);
    let onsets = buffer(device, queue, bytemuck::bytes_of(&1u32));
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let response = ResidentImpactResponse::build(
        &engine,
        &model,
        &[ContactResponseState {
            input: &state,
            predicted_rates: &predicted,
        }],
        &rows,
        &[],
        None,
        Some(&delta),
        select_predicted.then_some(&onsets),
        &dt,
        &status,
        ImpactResponseConfig {
            row_capacity: 1,
            impacts: ImpactGpuConfig {
                island_capacity: 1,
                blocks_per_island: 1,
                max_iterations: 4096,
                velocity_tolerance: 1e-7,
            },
        },
    )
    .unwrap();
    let snapshot = capture(&engine, &response, &status);
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[0]), 0);
    let incoming = active.relative_velocity[1] + increment;
    assert!((floats(&snapshot[9])[0] - incoming).abs() < 1e-8);
    assert!((floats(&snapshot[2])[1] + 1.5 * incoming).abs() < 1e-7);
    assert!((floats(&snapshot[3])[1] + 1.5 * incoming).abs() < 1e-7);
    assert_eq!(floats(&snapshot[1])[1], base);
    if select_predicted {
        for increment in [0.02_f32, -0.004, 0.02] {
            molla_core::gpu_write_buffer!(queue, &delta, 4, bytemuck::bytes_of(&increment));
            let snapshot = capture(&engine, &response, &status);
            assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[0]), 0);
            let predicted_contact = bytemuck::pod_read_unaligned::<[u32; 2]>(&snapshot[10]);
            let incoming = active.relative_velocity[1] + increment;
            assert_eq!(predicted_contact[0], u32::from(incoming <= 0.));
            assert!((f32::from_bits(predicted_contact[1]) - incoming).abs() < 1e-8);
            if incoming > 0. {
                assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[8]), 0);
                assert!(floats(&snapshot[2]).iter().all(|&v| v == 0.));
                assert!(floats(&snapshot[3]).iter().all(|&v| v == 0.));
            } else {
                assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[8]), 1);
                assert!((floats(&snapshot[2])[1] + 1.5 * incoming).abs() < 1e-7);
            }
            assert_eq!(floats(&snapshot[1])[1], base);
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn episodes_select_coupled_dynamic_groups_without_rebounding_shared_ground_neighbors() {
    use crate::impact_episodes_gpu::{ImpactEpisodeConfig, ResidentImpactEpisodes};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    for i in 0..3 {
        if i != 0 {
            builder.begin_articulation();
        }
        let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }));
        builder
            .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    for i in 0..3 {
        state.joint_qd.host_mut().unwrap()[6 * i + 1] = -(i as f64 + 1.);
    }
    eval_fk(&model, &mut state).unwrap();
    eval_body_velocities(&model, &mut state).unwrap();
    let contacts: Vec<_> = (0..3)
        .map(|i| ImpactContact {
            bodies: [None, Some(BodyId(i))],
            point: Vec3::ZERO,
            normal: Vec3::Y,
            static_friction: 0.,
            dynamic_friction: 0.,
            restitution: 0.5 + i as f64 * 0.2,
            surface_velocity: [Vec3::ZERO; 2],
        })
        .collect();
    let prepared: Vec<_> = contacts
        .iter()
        .enumerate()
        .map(|(i, &contact)| {
            let mut r = row(&model, &state, contact);
            r.endpoints[..2].copy_from_slice(&[3, i as u32]);
            r
        })
        .collect();
    let rows = buffer(device, queue, bytemuck::cast_slice(&prepared));
    let full_rows = buffer(device, queue, bytemuck::cast_slice(&prepared));
    let mut old = prepared.clone();
    old[0].coefficients[3] = 0.;
    let old_rows = buffer(device, queue, bytemuck::cast_slice(&old));
    let identities = buffer(device, queue, bytemuck::cast_slice(&[11u32, 12, 13, 99]));
    let status = buffer(device, queue, &[0; 4]);
    let episodes = ResidentImpactEpisodes::new(
        device,
        &rows,
        &identities,
        &status,
        ImpactEpisodeConfig {
            row_capacity: 3,
            shape_capacity: 4,
            table_capacity: 8,
            approach_speed: 1e-4,
            normal_similarity: 0.95,
        },
    )
    .unwrap();
    let scalar_row = buffer(
        device,
        queue,
        bytemuck::bytes_of(&ScalarConstraintRow {
            endpoints: [0, 1, 1, 1],
            velocity: [-1., 0., 0., 0.],
        }),
    );
    let mut j = vec![0f32; model.joint_dof_total];
    j[1] = -1.;
    j[7] = 1.;
    let jacobian = buffer(device, queue, bytemuck::cast_slice(&j));
    let predicted: Vec<_> = state
        .joint_qd
        .host()
        .unwrap()
        .iter()
        .map(|&v| v as f32)
        .collect();
    let predicted_buffer = buffer(device, queue, bytemuck::cast_slice(&predicted));
    let dt = buffer(device, queue, bytemuck::bytes_of(&0.01_f32));
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let response = ResidentImpactResponse::with_episode_onsets(
        &engine,
        &model,
        &[ContactResponseState {
            input: &state,
            predicted_rates: &predicted_buffer,
        }],
        episodes.impact_row_buffer(),
        &[],
        Some(ScalarConstraintBuffers {
            rows: &scalar_row,
            jacobians: &jacobian,
            capacity: 1,
        }),
        None,
        episodes.onset_buffer(),
        &dt,
        &status,
        ImpactResponseConfig {
            row_capacity: 3,
            impacts: ImpactGpuConfig {
                island_capacity: 3,
                blocks_per_island: 4,
                max_iterations: 8192,
                velocity_tolerance: 1e-7,
            },
        },
    )
    .unwrap();
    let snapshot = capture_after(&engine, &response, &status, |encoder| {
        encoder.copy_buffer_to_buffer(&old_rows, 0, &rows, 0, rows.size());
        episodes.encode(encoder);
        encoder.copy_buffer_to_buffer(&full_rows, 0, &rows, 0, rows.size());
        episodes.encode(encoder);
    });
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[0]), 0);
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[8]), 1);
    let rates = floats(&snapshot[1]);
    assert!((rates[1] - 0.5).abs() < 2e-5, "{rates:?}");
    assert!((rates[7] - 0.5).abs() < 2e-5, "{rates:?}");
    assert_eq!(&rates[12..], &predicted[12..]);
    assert_eq!(&floats(&snapshot[3])[8..12], &[0.; 4]);
    let slots: Vec<u32> = snapshot[6]
        .chunks_exact(4)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    assert_eq!(slots, [0, 0, 0, 1, u32::MAX, u32::MAX, 0, 2]);
    let snapshot = capture_after(&engine, &response, &status, |encoder| {
        encoder.copy_buffer_to_buffer(&full_rows, 0, &rows, 0, rows.size());
        episodes.encode(encoder);
    });
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[0]), 0);
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[8]), 0);
    assert_eq!(floats(&snapshot[1]), predicted);
    assert!(floats(&snapshot[2]).iter().all(|&v| v == 0.));
    assert!(floats(&snapshot[4]).iter().all(|&v| v == 0.));
    for (first, second) in [(-1f32, 1f32), (-1., 0.), (-1., 1e-8), (0.1, 0.2)] {
        let mut predicted = predicted.clone();
        predicted[1] = first;
        predicted[7] = second;
        molla_core::gpu_write_buffer!(
            queue,
            &predicted_buffer,
            0,
            bytemuck::cast_slice(&predicted)
        );
        let snapshot = capture_after(&engine, &response, &status, |encoder| {
            encoder.copy_buffer_to_buffer(&old_rows, 0, &rows, 0, rows.size());
            episodes.encode(encoder);
            encoder.copy_buffer_to_buffer(&full_rows, 0, &rows, 0, rows.size());
            episodes.encode(encoder);
        });
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[0]), 0);
        let slots: Vec<u32> = snapshot[6]
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let rates = floats(&snapshot[1]);
        let masks: Vec<[u32; 2]> = snapshot[10]
            .chunks_exact(8)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        assert_eq!(masks[0][0], u32::from(first <= 0.));
        // The inclusion decision uses the same rounded predictor increment as assembly.
        let second_speed = -2f32 + (second + 2f32);
        assert_eq!(masks[1][0], u32::from(second_speed <= 0.));
        assert_eq!(f32::from_bits(masks[1][1]), second_speed);
        if first > 0. {
            assert_eq!(rates, predicted);
            assert!(slots.iter().all(|&s| s == u32::MAX));
            assert!(floats(&snapshot[4]).iter().all(|&v| v == 0.));
        } else {
            assert!((rates[1] - 0.5).abs() < 2e-5, "{rates:?}");
            assert!((rates[7] - 0.5).abs() < 2e-5, "{rates:?}");
            if second_speed > 0. {
                assert_eq!(&slots[2..4], &[u32::MAX; 2]);
                assert_eq!(&floats(&snapshot[3])[4..8], &[0.; 4]);
            } else {
                assert_eq!(&slots[2..4], &[0, 1]);
            }
            assert_eq!(&rates[12..], &predicted[12..]);
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn separating_contact_keeps_graph_connectivity_without_consuming_solve_capacity() {
    separating_chain(3);
    separating_chain(130);
}

fn separating_chain(bodies: usize) {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    for i in 0..bodies {
        if i != 0 {
            builder.begin_articulation();
        }
        let b = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }));
        builder
            .add_joint_free(BodyId::NONE, b, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    for i in 0..bodies {
        state.joint_qd.host_mut().unwrap()[6 * i + 1] = -(i as f64 + 1.);
    }
    eval_fk(&model, &mut state).unwrap();
    eval_body_velocities(&model, &mut state).unwrap();
    let contact = |a, b, normal, restitution| ImpactContact {
        bodies: [a, b],
        point: Vec3::ZERO,
        normal,
        static_friction: 0.,
        dynamic_friction: 0.,
        restitution,
        surface_velocity: [Vec3::ZERO; 2],
    };
    let mut contacts = vec![contact(None, Some(BodyId(0)), Vec3::Y, 0.5)];
    for i in 0..bodies - 2 {
        contacts.push(contact(
            Some(BodyId::from_index(i)),
            Some(BodyId::from_index(i + 1)),
            Vec3::X,
            0.,
        ));
    }
    for i in bodies - 2..bodies {
        contacts.push(contact(None, Some(BodyId::from_index(i)), Vec3::Y, 0.));
    }
    let rows = buffer(device, queue, &vec![0; contacts.len() * 128]);
    let mut mask = vec![0u32; contacts.len()];
    mask[0] = 1;
    let onsets = buffer(device, queue, bytemuck::cast_slice(&mask));
    let mut rates: Vec<_> = state
        .joint_qd
        .host()
        .unwrap()
        .iter()
        .map(|&v| v as f32)
        .collect();
    for i in 0..bodies {
        rates[6 * i] = i as f32;
    }
    let predicted = buffer(device, queue, bytemuck::cast_slice(&rates));
    let dt = buffer(device, queue, bytemuck::bytes_of(&0.01_f32));
    let status = buffer(device, queue, &[0; 4]);
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let response = ResidentImpactResponse::with_episode_onsets(
        &engine,
        &model,
        &[ContactResponseState {
            input: &state,
            predicted_rates: &predicted,
        }],
        &rows,
        &[],
        None,
        None,
        &onsets,
        &dt,
        &status,
        ImpactResponseConfig {
            row_capacity: contacts.len() as u32,
            impacts: ImpactGpuConfig {
                island_capacity: 1,
                blocks_per_island: 2,
                max_iterations: 4096,
                velocity_tolerance: 1e-7,
            },
        },
    )
    .unwrap();
    for reverse in [false, true] {
        if reverse {
            for c in &mut contacts {
                c.bodies.swap(0, 1);
                c.normal = -c.normal;
            }
        }
        let prepared: Vec<_> = contacts.iter().map(|&c| row(&model, &state, c)).collect();
        molla_core::gpu_write_buffer!(queue, &rows, 0, bytemuck::cast_slice(&prepared));
        let snapshot = capture(&engine, &response, &status);
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&snapshot[0]), 0);
        let actual = floats(&snapshot[1]);
        assert!((actual[1] - 0.5).abs() < 1e-6);
        assert!(actual[6 * (bodies - 2) + 1].abs() < 1e-6);
        for i in 0..rates.len() {
            if i != 1 && i != 6 * (bodies - 2) + 1 {
                assert_eq!(actual[i], rates[i]);
            }
        }
        let slots: Vec<u32> = snapshot[6]
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let mut expected = vec![u32::MAX; 2 * contacts.len()];
        expected[..2].copy_from_slice(&[0, 0]);
        expected[2 * (bodies - 1)..2 * bodies].copy_from_slice(&[0, 1]);
        assert_eq!(slots, expected);
        let mask: Vec<[u32; 2]> = snapshot[10]
            .chunks_exact(8)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        assert_eq!(mask[0][0], 1);
        assert!(mask[bodies - 1..].iter().all(|p| p[0] == 1));
        for (i, p) in mask.iter().enumerate().take(bodies - 1).skip(1) {
            assert_eq!(p[0], 0);
            assert_eq!(f32::from_bits(p[1]), 1.);
            assert_eq!(&floats(&snapshot[3])[4 * i..4 * i + 4], &[0.; 4]);
        }
    }
    let mut bad: Vec<_> = contacts.iter().map(|&c| row(&model, &state, c)).collect();
    bad[1].coefficients[2] = f32::NAN;
    molla_core::gpu_write_buffer!(queue, &rows, 0, bytemuck::cast_slice(&bad));
    let snapshot = capture(&engine, &response, &status);
    assert_eq!(
        bytemuck::pod_read_unaligned::<u32>(&snapshot[0]),
        CONTACT_RESPONSE_FAULT_NONFINITE
    );
    assert!(floats(&snapshot[3]).iter().all(|&v| v == 0.));
}
