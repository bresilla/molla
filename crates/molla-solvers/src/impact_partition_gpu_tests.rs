use super::impact_tests::{buffer, floats, row};
use super::*;
use crate::constraint_rows_gpu::ScalarConstraintRow;
use crate::impact_constraints_gpu::ImpactGpuConfig;
use crate::impact_merge_gpu::{ImpactMergeConfig, ResidentImpactMerge};
use crate::rigid_impulse::{ImpactConstraint, ImpactContact, ImpactRow, apply_impact};
use molla_core::BodyId;
use molla_math::{Mat3, SpatialInertia, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder, eval_body_velocities, eval_fk};

fn snapshot(
    engine: &Engine,
    sources: &[&wgpu::Buffer],
    encode: impl FnOnce(&mut wgpu::CommandEncoder),
) -> Vec<Vec<u8>> {
    let device = engine.device().unwrap();
    let staging = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("impact partition snapshot"),
            size: sources.iter().map(|s| s.size()).sum(),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encode(&mut encoder);
    let mut offset = 0;
    for source in sources {
        encoder.copy_buffer_to_buffer(source, 0, &staging, offset, source.size());
        offset += source.size();
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = molla_core::gpu_readback!(staging.slice(..));
    let mut offset = 0;
    sources
        .iter()
        .map(|s| {
            let end = offset + s.size() as usize;
            let result = mapped[offset..end].to_vec();
            offset = end;
            result
        })
        .collect()
}

#[test]
#[ignore = "requires GPU"]
fn chained_impact_and_friction_partition_contacts_scalars_and_physical_rates() {
    partitioned_response(false);
}

#[test]
#[ignore = "requires GPU"]
fn bounded_feedback_survives_impact_friction_partition_and_physical_merge() {
    partitioned_response(true);
}

fn partitioned_response(bounded: bool) {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    for i in 0..3 {
        builder.begin_articulation();
        let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY,
        }));
        builder
            .add_joint_free(
                BodyId::NONE,
                body,
                Transform::from_translation(Vec3::X * (i as f64 * 2.)),
                Transform::IDENTITY,
            )
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state.joint_qd.host_mut().unwrap()[0..3].copy_from_slice(&[0.5, -1., 0.25]);
    state.joint_qd.host_mut().unwrap()[6] = 1.;
    state.joint_qd.host_mut().unwrap()[12] = 0.3;
    eval_fk(&model, &mut state).unwrap();
    eval_body_velocities(&model, &mut state).unwrap();
    let contact = |body, restitution| ImpactContact {
        bodies: [None, Some(BodyId(body))],
        point: Vec3::X * (body as f64 * 2.)
            + if body == 0 {
                Vec3::Z * 0.15
            } else {
                Vec3::ZERO
            },
        normal: Vec3::Y,
        static_friction: 0.5,
        dynamic_friction: 0.5,
        restitution,
        surface_velocity: [Vec3::ZERO; 2],
    };
    let contacts = [contact(0, 0.5), contact(0, 0.), contact(1, 0.)];
    let mut rows: Vec<_> = contacts
        .iter()
        .map(|&c| {
            let mut r = row(&model, &state, c);
            r.normal_force[3] = 10.;
            r.force_on_b = [0., 10., 0., 0.];
            r
        })
        .collect();
    let mut increment = vec![0f32; model.joint_dof_total];
    increment[1] = -0.125;
    increment[6] = 0.125;
    increment[7] = 0.1;
    increment[12] = 0.125;
    let predicted: Vec<f32> = state
        .joint_qd
        .host()
        .unwrap()
        .iter()
        .zip(&increment)
        .map(|(&v, &d)| v as f32 + d)
        .collect();
    let scalar_cpu = ImpactRow {
        ends: vec![(BodyId(0), SpatialVector::new(Vec3::Z, Vec3::ZERO))],
        direct: None,
        velocity_bias: -0.15,
    };
    let mut expected = state.clone_host().unwrap();
    expected
        .joint_qd
        .host_mut()
        .unwrap()
        .copy_from_slice(&predicted.iter().map(|&v| v as f64).collect::<Vec<_>>());
    eval_body_velocities(&model, &mut expected).unwrap();
    apply_impact(
        &model,
        &mut expected,
        &[
            ImpactConstraint::Contact(contacts[0]),
            ImpactConstraint::Contact(contacts[1]),
            if bounded {
                ImpactConstraint::Bounded {
                    row: scalar_cpu,
                    impulses: [-0.04, 0.04],
                    response_diagonal: 0.7,
                }
            } else {
                ImpactConstraint::Bilateral(scalar_cpu)
            },
        ],
    )
    .unwrap();
    expected.joint_qd.host_mut().unwrap()[6] -= 0.05;
    expected.joint_qd.host_mut().unwrap()[12] = if bounded { 0.3 + 0.125 - 0.04 } else { 0. };
    let scalar_rows = [
        ScalarConstraintRow {
            endpoints: [u32::MAX, 0, if bounded { 3 } else { 1 }, 1],
            velocity: if bounded {
                [0.1, 0.7, -0.04, 0.04]
            } else {
                [0.1, 0., 0., 0.]
            },
        },
        ScalarConstraintRow {
            endpoints: [u32::MAX, 2, if bounded { 3 } else { 1 }, 1],
            velocity: if bounded {
                [0.3, 0.7, -0.04, 0.04]
            } else {
                [0.3, 0., 0., 0.]
            },
        },
    ];
    let mut jacobian = vec![0f32; model.joint_dof_total * 2];
    jacobian[2] = 1.;
    jacobian[model.joint_dof_total + 12] = 1.;
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let mut reaction_states = [state.clone_host().unwrap(), state.clone_host().unwrap()];
    for state in &mut reaction_states {
        state.upload_to_device(device, queue).unwrap();
    }
    let external_values = [
        [0f32, -32.5, 0., 0., 3., 0., 0., 0.],
        [12.5, 0., 0., 0., 0., 0., 0., 0.],
        [12.5, 0., 0., 0., 0., 0., 0., 0.],
    ];
    let external = buffer(device, queue, bytemuck::cast_slice(&external_values));
    let heads = buffer(device, queue, bytemuck::cast_slice(&[1u32, 5, u32::MAX]));
    let links = buffer(
        device,
        queue,
        bytemuck::cast_slice(&[u32::MAX, 3, u32::MAX, u32::MAX, u32::MAX, u32::MAX]),
    );
    let normal = buffer(device, queue, bytemuck::cast_slice(&rows));
    let scalar = buffer(device, queue, bytemuck::cast_slice(&scalar_rows));
    let jacobians = buffer(device, queue, bytemuck::cast_slice(&jacobian));
    let delta = buffer(device, queue, bytemuck::cast_slice(&increment));
    let predicted = buffer(device, queue, bytemuck::cast_slice(&predicted));
    let onsets = buffer(device, queue, bytemuck::cast_slice(&[1u32, 0, 0]));
    let status = buffer(device, queue, &[0; 4]);
    let dt = buffer(device, queue, bytemuck::cast_slice(&[0.01f32]));
    let scalar_input = Some(ScalarConstraintBuffers {
        rows: &scalar,
        jacobians: &jacobians,
        capacity: 2,
    });
    let impact = ResidentImpactResponse::with_episode_onsets(
        &engine,
        &model,
        &[ContactResponseState {
            input: &state,
            predicted_rates: &predicted,
        }],
        &normal,
        &[],
        scalar_input,
        Some(&delta),
        &onsets,
        &dt,
        &status,
        ImpactResponseConfig {
            row_capacity: 3,
            impacts: ImpactGpuConfig {
                island_capacity: 3,
                blocks_per_island: 3,
                max_iterations: 32768,
                velocity_tolerance: 2e-6,
            },
        },
    )
    .unwrap();
    for retained in [None, Some(&delta)] {
        let friction = ResidentContactResponse::with_excluded_impact_groups(
            &engine,
            &model,
            &[ContactResponseState {
                input: &state,
                predicted_rates: impact.corrected_rates_buffer(),
            }],
            &normal,
            &[],
            scalar_input,
            retained,
            impact.islands().articulation_islands_buffer(),
            1,
            &dt,
            &status,
            ContactResponseConfig {
                row_capacity: 3,
                disks: CoulombDiskConfig {
                    island_capacity: 3,
                    contacts_per_island: 2,
                    max_iterations: 8192,
                    velocity_tolerance: 1e-7,
                },
            },
        )
        .unwrap();
        let merge = ResidentImpactMerge::new(
            &engine,
            &impact,
            &friction,
            &status,
            ImpactMergeConfig {
                contact_rows: 3,
                scalar_rows: 2,
                dofs: model.joint_dof_total as u32,
            },
        )
        .unwrap();
        let reactions = crate::contact_reaction_gpu::ContactReactionGpu::new(
            &engine,
            &model,
            &reaction_states,
            crate::contact_reaction_gpu::ContactReactionInputs {
                rows: &normal,
                heads: &heads,
                links: &links,
                world_impulses: merge.world_impulse_buffer(),
                response_delta: merge.velocity_delta_buffer(),
            },
            &external,
            &dt,
            &delta,
            &status,
            3,
        );
        let capture = || {
            snapshot(
                &engine,
                &[
                    &status,
                    impact.corrected_rates_buffer(),
                    friction.corrected_rates_buffer(),
                    friction.velocity_delta_buffer(),
                    friction.world_impulse_buffer(),
                    impact.scalar_solution_buffer(),
                    friction.scalar_solution_buffer(),
                    friction.islands().row_slots_buffer(),
                    merge.world_impulse_buffer(),
                    merge.velocity_delta_buffer(),
                    merge.scalar_solution_buffer(),
                    impact.world_impulse_buffer(),
                    impact.velocity_delta_buffer(),
                    &reactions.loads,
                    &reactions.delta,
                ],
                |encoder| {
                    impact.encode(encoder, 0).unwrap();
                    friction.encode(encoder, 0).unwrap();
                    merge.encode(encoder);
                    reactions.encode(encoder, 0);
                },
            )
        };
        let data = capture();
        assert_eq!(u32::from_ne_bytes(data[0][..4].try_into().unwrap()), 0);
        let rates = floats(&data[2]);
        for (d, (&a, &b)) in rates
            .iter()
            .zip(expected.joint_qd.host().unwrap())
            .enumerate()
        {
            assert!(
                (a as f64 - b).abs() < 1e-5,
                "retained={}, dof={d}, {a} != {b}",
                retained.is_some()
            );
        }
        assert_eq!(&floats(&data[1])[..6], &rates[..6]);
        assert_eq!(&floats(&data[3])[..6], &[0.; 6]);
        assert_eq!(&floats(&data[4])[..8], &[0.; 8]);
        assert_eq!(&data[6][..32], &[0; 32]);
        assert_eq!(&data[5][32..64], &[0; 32]);
        assert_eq!(&data[10][..32], &data[5][..32]);
        assert_eq!(&data[10][32..64], &data[6][32..64]);
        for (merged, (a, b)) in floats(&data[8])
            .iter()
            .zip(floats(&data[4]).iter().zip(floats(&data[11])))
        {
            assert_eq!(*merged, a + b);
        }
        for (merged, (a, b)) in floats(&data[9])
            .iter()
            .zip(floats(&data[3]).iter().zip(floats(&data[12])))
        {
            assert_eq!(*merged, a + b);
        }
        let scalar_impulses = floats(&data[10]);
        let loads = floats(&data[13]);
        let total_delta = floats(&data[14]);
        let contact_impulses = floats(&data[8]);
        for body in 0..3 {
            let mut force = Vec3::new(
                external_values[body][0] as f64,
                external_values[body][1] as f64,
                external_values[body][2] as f64,
            );
            let mut torque = Vec3::new(
                external_values[body][4] as f64,
                external_values[body][5] as f64,
                external_values[body][6] as f64,
            );
            let center = state.body_q.host().unwrap()[body].position;
            for (r, source) in rows
                .iter()
                .enumerate()
                .filter(|(_, r)| r.endpoints[3] as usize == body)
            {
                let impulse = Vec3::new(
                    contact_impulses[r * 4] as f64,
                    contact_impulses[r * 4 + 1] as f64,
                    contact_impulses[r * 4 + 2] as f64,
                );
                let normal_force = Vec3::new(
                    source.force_on_b[0] as f64,
                    source.force_on_b[1] as f64,
                    source.force_on_b[2] as f64,
                );
                let applied = normal_force + impulse / 0.01f32 as f64;
                let point = Vec3::new(
                    source.point_depth[0] as f64,
                    source.point_depth[1] as f64,
                    source.point_depth[2] as f64,
                );
                force += applied;
                torque += (point - center).cross(applied);
            }
            for k in 0..3 {
                assert!((loads[body * 8 + k] as f64 - force[k]).abs() < 5e-5);
                assert!((loads[body * 8 + 4 + k] as f64 - torque[k]).abs() < 5e-5);
                let scalar_force = if body == 0 && k == 2 {
                    scalar_impulses[0]
                } else if body == 2 && k == 0 {
                    scalar_impulses[8]
                } else {
                    0.
                } / 0.01;
                assert!(
                    (total_delta[body * 6 + k] / 0.01 - loads[body * 8 + k] - scalar_force).abs()
                        < 1e-4
                );
                assert!(
                    (total_delta[body * 6 + 3 + k] / 0.01 - loads[body * 8 + 4 + k]).abs() < 1e-4
                );
            }
        }
        let slots: Vec<u32> = data[7]
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        assert_eq!(&slots[..4], &[u32::MAX; 4]);
        assert_ne!(slots[4], u32::MAX);
        assert_eq!(&slots[6..8], &[u32::MAX; 2]);
        assert_ne!(slots[8], u32::MAX);
        let repeat = capture();
        assert_eq!(data, repeat);
        molla_core::gpu_write_buffer!(
            queue,
            impact.scalar_solution_buffer(),
            0,
            bytemuck::bytes_of(&0f32),
        );
        let zero_active = snapshot(
            &engine,
            &[&status, merge.scalar_solution_buffer()],
            |encoder| merge.encode(encoder),
        );
        assert_eq!(
            u32::from_ne_bytes(zero_active[0][..4].try_into().unwrap()),
            0
        );
        assert_eq!(&zero_active[1][..4], &[0; 4]);
        assert_eq!(&zero_active[1][4..32], &data[5][4..32]);
        assert_eq!(
            u32::from_ne_bytes(zero_active[1][20..24].try_into().unwrap()),
            if bounded { 6 } else { 4 }
        );
        assert_eq!(capture(), data);
        let failed = || {
            snapshot(
                &engine,
                &[
                    &status,
                    merge.world_impulse_buffer(),
                    merge.velocity_delta_buffer(),
                    merge.scalar_solution_buffer(),
                ],
                |encoder| merge.encode(encoder),
            )
        };
        molla_core::gpu_write_buffer!(
            queue,
            impact.world_impulse_buffer(),
            0,
            bytemuck::cast_slice(&[f32::NAN]),
        );
        let failure = failed();
        assert_ne!(
            u32::from_ne_bytes(failure[0][..4].try_into().unwrap())
                & CONTACT_RESPONSE_FAULT_NONFINITE,
            0
        );
        assert!(failure[1..].iter().flatten().all(|&v| v == 0));
        assert!(failed()[1..].iter().flatten().all(|&v| v == 0));
        molla_core::gpu_write_buffer!(queue, &status, 0, &[0; 4]);
        assert_eq!(capture(), data);
        molla_core::gpu_write_buffer!(
            queue,
            friction.islands().row_slots_buffer(),
            0,
            bytemuck::cast_slice(&[0u32, 0]),
        );
        let failure = failed();
        assert_ne!(
            u32::from_ne_bytes(failure[0][..4].try_into().unwrap()) & CONTACT_RESPONSE_FAULT_INPUT,
            0
        );
        assert!(failure[1..].iter().flatten().all(|&v| v == 0));
        molla_core::gpu_write_buffer!(queue, &status, 0, &[0; 4]);
        assert_eq!(capture(), data);
        for (target, offset, value, fault) in [
            (
                friction.islands().row_slots_buffer(),
                24,
                0u32,
                CONTACT_RESPONSE_FAULT_INPUT,
            ),
            (
                friction.velocity_delta_buffer(),
                4,
                0.25f32.to_bits(),
                CONTACT_RESPONSE_FAULT_INPUT,
            ),
            (
                impact.scalar_solution_buffer(),
                20,
                0u32,
                CONTACT_RESPONSE_FAULT_INPUT,
            ),
            (
                impact.scalar_solution_buffer(),
                8,
                f32::NAN.to_bits(),
                CONTACT_RESPONSE_FAULT_NONFINITE,
            ),
            (
                friction.velocity_delta_buffer(),
                24,
                f32::NAN.to_bits(),
                CONTACT_RESPONSE_FAULT_NONFINITE,
            ),
        ] {
            molla_core::gpu_write_buffer!(queue, target, offset, bytemuck::bytes_of(&value));
            let failure = failed();
            assert_ne!(
                u32::from_ne_bytes(failure[0][..4].try_into().unwrap()) & fault,
                0
            );
            assert!(failure[1..].iter().flatten().all(|&v| v == 0));
            molla_core::gpu_write_buffer!(queue, &status, 0, &[0; 4]);
            assert_eq!(capture(), data);
        }
        assert!(
            ResidentImpactMerge::new(
                &engine,
                &impact,
                &friction,
                &status,
                ImpactMergeConfig {
                    contact_rows: 2,
                    scalar_rows: 2,
                    dofs: model.joint_dof_total as u32
                }
            )
            .is_err()
        );
    }
    rows[0].coefficients[3] = 0.;
    rows[1].coefficients[3] = 0.;
    molla_core::gpu_write_buffer!(queue, &normal, 0, bytemuck::cast_slice(&rows));
    molla_core::gpu_write_buffer!(queue, &onsets, 0, bytemuck::cast_slice(&[0u32; 3]));
    let retired = snapshot(
        &engine,
        &[
            &status,
            impact.islands().articulation_islands_buffer(),
            impact.scalar_solution_buffer(),
        ],
        |encoder| impact.encode(encoder, 0).unwrap(),
    );
    assert_eq!(u32::from_ne_bytes(retired[0][..4].try_into().unwrap()), 0);
    assert!(
        retired[1]
            .chunks_exact(4)
            .all(|v| u32::from_ne_bytes(v.try_into().unwrap()) == u32::MAX)
    );
    assert!(retired[2].iter().all(|&v| v == 0));
}

#[test]
#[ignore = "requires GPU"]
fn partition_masks_preserve_joint_only_islands_and_reject_split_graphs() {
    use crate::contact_islands_gpu::{CONTACT_ISLAND_FAULT_CAPACITY, CONTACT_ISLAND_FAULT_INPUT};
    use crate::normal_contact_gpu::NormalContactRow;
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let make_row = |a, b, force| NormalContactRow {
        endpoints: [0, 1, a, b],
        normal_force: [0., 1., 0., force],
        coefficients: [0.5, 0.5, 0., 1.],
        ..Default::default()
    };
    let mut rows = [
        make_row(3, 0, 10.),
        make_row(0, 1, 0.),
        make_row(3, 1, 10.),
        make_row(3, 2, 10.),
        make_row(3, 0, 10.),
    ];
    rows[1].coefficients[0] = 0.;
    rows[1].coefficients[1] = 0.;
    let mut scalars = [
        ScalarConstraintRow {
            endpoints: [1, u32::MAX, 1, 1],
            velocity: [0.; 4],
        },
        ScalarConstraintRow {
            endpoints: [2, u32::MAX, 1, 1],
            velocity: [0.; 4],
        },
        ScalarConstraintRow {
            endpoints: [u32::MAX, u32::MAX, 1, 1],
            velocity: [0.; 4],
        },
    ];
    let normal = buffer(device, queue, bytemuck::cast_slice(&rows));
    let scalar = buffer(device, queue, bytemuck::cast_slice(&scalars));
    let excluded = buffer(device, queue, bytemuck::cast_slice(&[0u32, 0, u32::MAX]));
    let status = buffer(device, queue, &[0; 4]);
    let owners = [0, 1, 2, u32::MAX];
    let config = ContactIslandConfig {
        articulation_count: 3,
        row_capacity: 5,
        island_capacity: 2,
        contacts_per_island: 2,
    };
    let islands = ResidentContactIslands::with_excluded_impacts(
        &engine,
        &owners,
        &[],
        &normal,
        Some((&scalar, 3)),
        &excluded,
        &status,
        config,
        1,
    )
    .unwrap();
    let capture = || {
        snapshot(
            &engine,
            &[
                &status,
                islands.island_count_buffer(),
                islands.contact_counts_buffer(),
                islands.contact_indices_buffer(),
                islands.articulation_islands_buffer(),
                islands.row_slots_buffer(),
            ],
            |encoder| islands.encode(encoder),
        )
    };
    let words = |bytes: &[u8]| {
        bytes
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned::<u32>)
            .collect::<Vec<_>>()
    };
    let output = capture();
    assert_eq!(words(&output[0]), [0]);
    assert_eq!(words(&output[1]), [2]);
    assert_eq!(words(&output[2]), [2, 1]);
    assert_eq!(words(&output[3]), [3, 6, 7, u32::MAX]);
    assert_eq!(&words(&output[4])[..3], &[u32::MAX, u32::MAX, 0]);
    for index in [0, 1, 2, 4, 5] {
        assert_eq!(&words(&output[5])[index * 2..index * 2 + 2], &[u32::MAX; 2]);
    }
    assert_eq!(output, capture());
    for (mask, expected_fault) in [
        ([0, u32::MAX, u32::MAX], CONTACT_ISLAND_FAULT_INPUT),
        ([u32::MAX; 3], CONTACT_ISLAND_FAULT_CAPACITY),
    ] {
        molla_core::gpu_write_buffer!(queue, &excluded, 0, bytemuck::cast_slice(&mask));
        molla_core::gpu_write_buffer!(queue, &status, 0, &[0; 4]);
        assert_ne!(words(&capture()[0])[0] & expected_fault, 0);
    }
    molla_core::gpu_write_buffer!(queue, &status, 0, &[0; 4]);
    molla_core::gpu_write_buffer!(
        queue,
        &excluded,
        0,
        bytemuck::cast_slice(&[0u32, u32::MAX, u32::MAX]),
    );
    rows[1].coefficients[3] = 0.;
    molla_core::gpu_write_buffer!(queue, &normal, 0, bytemuck::cast_slice(&rows));
    scalars[0].endpoints = [0, 1, 1, 1];
    molla_core::gpu_write_buffer!(queue, &scalar, 0, bytemuck::cast_slice(&scalars));
    assert_ne!(words(&capture()[0])[0] & CONTACT_ISLAND_FAULT_INPUT, 0);
    scalars[0].endpoints = [1, u32::MAX, 1, 1];
    molla_core::gpu_write_buffer!(queue, &scalar, 0, bytemuck::cast_slice(&scalars));
    let dependencies = ResidentContactIslands::with_excluded_impacts(
        &engine,
        &owners,
        &[[0, 1]],
        &normal,
        Some((&scalar, 3)),
        &excluded,
        &status,
        config,
        1,
    )
    .unwrap();
    molla_core::gpu_write_buffer!(queue, &status, 0, &[0; 4]);
    let result = snapshot(&engine, &[&status], |encoder| dependencies.encode(encoder));
    assert_ne!(words(&result[0])[0] & CONTACT_ISLAND_FAULT_INPUT, 0);
    molla_core::gpu_write_buffer!(
        queue,
        &excluded,
        0,
        bytemuck::cast_slice(&[0u32, 0, u32::MAX])
    );
    molla_core::gpu_write_buffer!(queue, &status, 0, &[0; 4]);
    rows[0].coefficients[0] = -1.;
    molla_core::gpu_write_buffer!(queue, &normal, 0, bytemuck::cast_slice(&rows));
    assert_ne!(words(&capture()[0])[0] & CONTACT_ISLAND_FAULT_INPUT, 0);
    let too_small = buffer(device, queue, &[0; 4]);
    assert!(
        ResidentContactIslands::with_excluded_impacts(
            &engine,
            &owners,
            &[],
            &normal,
            Some((&scalar, 3)),
            &too_small,
            &status,
            config,
            1
        )
        .is_err()
    );
}
