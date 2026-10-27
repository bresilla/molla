use super::*;
use crate::constraint_rows_gpu::ScalarConstraintRow;
use crate::coulomb_disk_gpu::{CoulombDiskSolution, FRICTION_FAULT_CONVERGENCE};
use crate::normal_contact_gpu::NormalContactRow;
use molla_core::BodyId;
use molla_math::{Mat3, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder};

fn input(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) -> wgpu::Buffer {
    let b = storage(device, "mixed response fixture", bytes.len().max(4) as u64);
    molla_core::gpu_write_buffer!(queue, &b, 0, bytes);
    b
}

fn capture(
    engine: &Engine,
    response: &ResidentContactResponse,
    status: &wgpu::Buffer,
) -> Vec<Vec<u8>> {
    capture_then(engine, response, status, &[], |_| {})
}

fn capture_then(
    engine: &Engine,
    response: &ResidentContactResponse,
    status: &wgpu::Buffer,
    extra: &[&wgpu::Buffer],
    after: impl FnOnce(&mut wgpu::CommandEncoder),
) -> Vec<Vec<u8>> {
    let device = engine.device().unwrap();
    let mut sources = vec![
        status,
        response.corrected_rates_buffer(),
        response.velocity_delta_buffer(),
        response.scalar_solution_buffer(),
        response.world_impulse_buffer(),
        response.gram_buffer(),
        response.islands().island_count_buffer(),
        response.islands().row_slots_buffer(),
    ];
    sources.extend_from_slice(extra);
    let size = sources.iter().map(|b| b.size()).sum();
    let snapshot = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("mixed response readback"),
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        }
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    response.encode(&mut encoder, 0).unwrap();
    after(&mut encoder);
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
        .map(|b| {
            let size = b.size() as usize;
            let data = bytes[offset..offset + size].to_vec();
            offset += size;
            data
        })
        .collect()
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-response-gpu"]
fn retained_predictor_increment_survives_rounded_rates_in_mixed_response() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
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
    let base = 1048576.0_f32;
    let increment = 0.004_f32;
    assert_eq!(base + increment, base);
    state.joint_qd.host_mut().unwrap()[0] = base as f64;
    molla_sim::eval_fk(&model, &mut state).unwrap();
    molla_sim::eval_body_velocities(&model, &mut state).unwrap();
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let predicted = input(
        device,
        queue,
        bytemuck::cast_slice(&[base, 0., 0., 0., 0., 0.]),
    );
    let delta = input(
        device,
        queue,
        bytemuck::cast_slice(&[increment, 0., 0., 0., 0., 0.]),
    );
    let normal_row = NormalContactRow {
        normal_force: [0., 0., 1., 2.],
        coefficients: [0.125, 0.1, 0., 1.],
        endpoints: [0, 1, u32::MAX, 0],
        relative_velocity: [0., 0.01, 0., 0.],
        ..Default::default()
    };
    let normal = input(device, queue, bytemuck::bytes_of(&normal_row));
    let row = input(
        device,
        queue,
        bytemuck::bytes_of(&ScalarConstraintRow {
            endpoints: [u32::MAX, 0, 1, 1],
            velocity: [0.; 4],
        }),
    );
    let jacobian = input(
        device,
        queue,
        bytemuck::cast_slice(&[1.0_f32, 0., 0., 0., 0., 0.]),
    );
    let dt = input(device, queue, bytemuck::bytes_of(&0.01_f32));
    let status = input(device, queue, &[0; 4]);
    let response = ResidentContactResponse::with_runtime_constraints(
        &engine,
        &model,
        &[ContactResponseState {
            input: &state,
            predicted_rates: &predicted,
        }],
        &normal,
        &[],
        Some(ScalarConstraintBuffers {
            rows: &row,
            jacobians: &jacobian,
            capacity: 1,
        }),
        Some(&delta),
        1,
        &dt,
        &status,
        ContactResponseConfig {
            row_capacity: 1,
            disks: CoulombDiskConfig {
                island_capacity: 1,
                contacts_per_island: 2,
                max_iterations: 8192,
                velocity_tolerance: 1e-7,
            },
        },
    )
    .unwrap();
    for contact in [true, false] {
        let mut normal_row = normal_row;
        normal_row.coefficients[3] = f32::from(contact);
        molla_core::gpu_write_buffer!(queue, &normal, 0, bytemuck::bytes_of(&normal_row));
        let data = capture(&engine, &response, &status);
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&data[0]), 0);
        let rates = floats(&data[1]);
        let correction = floats(&data[2]);
        let scalar = bytemuck::pod_read_unaligned::<CoulombDiskSolution>(&data[3]);
        assert!((scalar.impulse[0] + increment).abs() < 2e-7, "{scalar:?}");
        assert!((correction[0] + increment).abs() < 1e-7, "{correction:?}");
        assert_eq!(rates[0], base);
        assert!((rates[1] + if contact { 0.002 } else { 0. }).abs() < 1e-7);
    }
    molla_core::gpu_write_buffer!(queue, &delta, 0, bytemuck::bytes_of(&f32::NAN));
    let data = capture(&engine, &response, &status);
    assert_eq!(
        bytemuck::pod_read_unaligned::<u32>(&data[0]),
        CONTACT_RESPONSE_FAULT_NONFINITE
    );
    let halted = capture(&engine, &response, &status);
    assert!(halted[1..5].iter().flatten().all(|&v| v == 0));
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-response-gpu"]
fn scalar_impulse_residue_survives_mass_response_and_rate_cancellation() {
    let engine = Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software, "requires hardware GPU");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
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
    let base = -1537.173_f32;
    state.joint_qd.host_mut().unwrap()[0] = f64::from(base);
    molla_sim::eval_fk(&model, &mut state).unwrap();
    molla_sim::eval_body_velocities(&model, &mut state).unwrap();
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let predicted = input(
        device,
        queue,
        bytemuck::cast_slice(&[base, 0., 0., 0., 0., 0.]),
    );
    let normal = input(
        device,
        queue,
        bytemuck::bytes_of(&NormalContactRow::default()),
    );
    let rows = input(
        device,
        queue,
        bytemuck::bytes_of(&ScalarConstraintRow {
            endpoints: [u32::MAX, 0, 1, 1],
            velocity: [base, 0., 0., 0.],
        }),
    );
    let jacobian = input(
        device,
        queue,
        bytemuck::cast_slice(&[1_f32, 0.5, 0., 0., 0., 0.]),
    );
    let dt = input(device, queue, bytemuck::bytes_of(&0.01_f32));
    let status = input(device, queue, &[0; 4]);
    let response = ResidentContactResponse::with_constraints(
        &engine,
        &model,
        &[ContactResponseState {
            input: &state,
            predicted_rates: &predicted,
        }],
        &normal,
        &[],
        Some(ScalarConstraintBuffers {
            rows: &rows,
            jacobians: &jacobian,
            capacity: 1,
        }),
        &dt,
        &status,
        ContactResponseConfig {
            row_capacity: 1,
            disks: CoulombDiskConfig {
                island_capacity: 1,
                contacts_per_island: 2,
                max_iterations: 4096,
                velocity_tolerance: 1e-7,
            },
        },
    )
    .unwrap();
    let data = capture(&engine, &response, &status);
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&data[0]), 0);
    let solution = bytemuck::pod_read_unaligned::<CoulombDiskSolution>(&data[3]);
    let exact = -f64::from(base) / 1.25;
    let impulse = f64::from(solution.impulse[0]) + f64::from(solution.impulse[1]);
    assert!((impulse - exact).abs() < 1e-7, "{solution:?}");
    assert_ne!(solution.impulse[1], 0.);
    let rates = floats(&data[1]);
    println!(
        "mass_response_probe impulse={:?} delta={:?} rates={rates:?} exact={exact}",
        solution.impulse,
        floats(&data[2])
    );
    assert_eq!(rates[0], (f64::from(base) + exact) as f32);
    assert_eq!(rates[1], (0.5 * exact) as f32);
    assert!(rates[2..].iter().all(|&rate| rate == 0.));
    println!(
        "adapter={adapter:?} impulse={:?} rates={rates:?} absolute_kkt={}",
        solution.impulse, solution.residual
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-response-gpu"]
fn disk_impulse_residue_survives_mass_response_and_rate_cancellation() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
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
    let base = 1537.173_f32;
    state.joint_qd.host_mut().unwrap()[1] = -f64::from(base);
    molla_sim::eval_fk(&model, &mut state).unwrap();
    molla_sim::eval_body_velocities(&model, &mut state).unwrap();
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let predicted = input(
        device,
        queue,
        bytemuck::cast_slice(&[0., -base, 0., 0., 0., 0.]),
    );
    let normal = input(
        device,
        queue,
        bytemuck::bytes_of(&NormalContactRow {
            normal_force: [0., 0., 1., 2.],
            coefficients: [0.5, 0.3, 0., 1.],
            endpoints: [0, 1, u32::MAX, 0],
            relative_velocity: [0., -base, 0., 0.],
            ..Default::default()
        }),
    );
    let dt = input(device, queue, bytemuck::bytes_of(&0.01_f32));
    let status = input(device, queue, &[0; 4]);
    let response = ResidentContactResponse::new(
        &engine,
        &model,
        &[ContactResponseState {
            input: &state,
            predicted_rates: &predicted,
        }],
        &normal,
        &[],
        &dt,
        &status,
        ContactResponseConfig {
            row_capacity: 1,
            disks: CoulombDiskConfig {
                island_capacity: 1,
                contacts_per_island: 1,
                max_iterations: 4096,
                velocity_tolerance: 1e-7,
            },
        },
    )
    .unwrap();
    let primary = input(
        device,
        queue,
        bytemuck::bytes_of(&CoulombDiskSolution {
            impulse: [base, 0.],
            ..Default::default()
        }),
    );
    for low in [-2.4414063e-5_f32, 2.4414063e-5] {
        assert_eq!(base + low, base);
        let residues = input(device, queue, bytemuck::cast_slice(&[low, 0.]));
        let data = capture_then(&engine, &response, &status, &[], |encoder| {
            encoder.copy_buffer_to_buffer(&primary, 0, response.disks().solution_buffer(), 0, 32);
            encoder.copy_buffer_to_buffer(
                &residues,
                0,
                response.disks().solution_buffer(),
                response.disks().impulse_residue_offset(),
                8,
            );
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&response.core.pipelines[3]);
            pass.set_bind_group(0, &response.core.groups[0], &[]);
            pass.dispatch_workgroups(1, 1, 1);
        });
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&data[0]), 0);
        let rates = floats(&data[1]);
        assert_eq!(rates[1], low, "rates={rates:?} residue={low}");
        assert!(rates.iter().enumerate().all(|(i, &v)| i == 1 || v == 0.));
        println!("disk mass response primary={base} residue={low} rates={rates:?}");
    }
}

struct Fixture {
    engine: std::sync::Arc<Engine>,
    response: ResidentContactResponse,
    normal: wgpu::Buffer,
    rows: wgpu::Buffer,
    jacobian: wgpu::Buffer,
    status: wgpu::Buffer,
    source_rows: Vec<ScalarConstraintRow>,
    source_jacobian: Vec<f32>,
}

impl Fixture {
    fn new(scale: f32, kind: u32, sign: f32, cross_articulation: bool, contact: bool) -> Self {
        Self::with_engine(
            std::sync::Arc::new(Engine::new_gpu().unwrap()),
            scale,
            kind,
            sign,
            cross_articulation,
            contact,
        )
    }

    fn with_engine(
        engine: std::sync::Arc<Engine>,
        scale: f32,
        kind: u32,
        sign: f32,
        cross_articulation: bool,
        contact: bool,
    ) -> Self {
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut builder = ModelBuilder::new();
        for i in 0..2 {
            if i != 0 {
                builder.begin_articulation();
            }
            let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: scale as f64,
                com: Vec3::ZERO,
                inertia: Mat3::IDENTITY * scale as f64,
            }));
            builder
                .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
                .unwrap();
        }
        let mut model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        state.joint_qd.host_mut().unwrap()[1] = 0.01 * sign as f64;
        state.joint_qd.host_mut().unwrap()[6] = 0.02;
        molla_sim::eval_fk(&model, &mut state).unwrap();
        molla_sim::eval_body_velocities(&model, &mut state).unwrap();
        let prediction: Vec<f32> = state
            .joint_qd
            .host()
            .unwrap()
            .iter()
            .map(|&v| v as f32)
            .collect();
        model.upload_to_device(device, queue).unwrap();
        state.upload_to_device(device, queue).unwrap();
        let predicted = input(device, queue, bytemuck::cast_slice(&prediction));
        let normal = input(
            device,
            queue,
            bytemuck::bytes_of(&NormalContactRow {
                normal_force: [0., 0., 1., 2. * scale],
                coefficients: [0.125, 0.1, 0., f32::from(contact)],
                endpoints: [0, 1, u32::MAX, 0],
                relative_velocity: [0., 0.01 * sign, 0., 0.],
                ..Default::default()
            }),
        );
        let source_rows = vec![
            ScalarConstraintRow {
                endpoints: [if cross_articulation { 1 } else { u32::MAX }, 0, kind, 1],
                velocity: [0.01 * sign, 0., 0., 0.],
            },
            ScalarConstraintRow {
                endpoints: [u32::MAX, 1, 1, 1],
                velocity: [0.02, 0., 0., 0.],
            },
            ScalarConstraintRow {
                endpoints: [u32::MAX, u32::MAX, 1, 1],
                velocity: [0.; 4],
            },
        ];
        let mut source_jacobian = vec![0.0_f32; 36];
        source_jacobian[1] = 1.;
        if cross_articulation {
            source_jacobian[7] = -1.;
        } else {
            source_jacobian[0] = -1.;
        }
        source_jacobian[12 + 6] = 1.;
        let rows = input(device, queue, bytemuck::cast_slice(&source_rows));
        let jacobian = input(device, queue, bytemuck::cast_slice(&source_jacobian));
        let status = input(device, queue, &[0; 4]);
        let dt = input(device, queue, bytemuck::bytes_of(&0.01_f32));
        let response = ResidentContactResponse::with_constraints(
            &engine,
            &model,
            &[ContactResponseState {
                input: &state,
                predicted_rates: &predicted,
            }],
            &normal,
            &[],
            Some(ScalarConstraintBuffers {
                rows: &rows,
                jacobians: &jacobian,
                capacity: 3,
            }),
            &dt,
            &status,
            ContactResponseConfig {
                row_capacity: 1,
                disks: CoulombDiskConfig {
                    island_capacity: 3,
                    contacts_per_island: 4,
                    max_iterations: 8192,
                    velocity_tolerance: 1e-7,
                },
            },
        )
        .unwrap();
        Self {
            engine,
            response,
            normal,
            rows,
            jacobian,
            status,
            source_rows,
            source_jacobian,
        }
    }

    fn run(&self) -> Vec<Vec<u8>> {
        capture(&self.engine, &self.response, &self.status)
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-response-gpu"]
fn bounded_feedback_response_preserves_contact_capacity_and_physical_mass_response() {
    use crate::coupled_constraints::{Domain, solve};
    let engine = std::sync::Arc::new(Engine::new_gpu().unwrap());
    for scale in [0.01_f32, 1., 100.] {
        for cross in [false, true] {
            for contact in [false, true] {
                let f = Fixture::with_engine(engine.clone(), scale, 3, 1., cross, contact);
                let queue = f.engine.queue().unwrap();
                for free in [-0.01_f32, 0.01] {
                    for (lower, upper) in [(-0.001, 0.001), (0.002, 0.002), (-0.02, 0.02)] {
                        let mut rows = f.source_rows.clone();
                        rows[0].velocity = [free, 0.7 / scale, lower * scale, upper * scale];
                        molla_core::gpu_write_buffer!(
                            queue,
                            &f.rows,
                            0,
                            bytemuck::cast_slice(&rows)
                        );
                        let data = f.run();
                        assert_eq!(
                            bytemuck::pod_read_unaligned::<u32>(&data[0]),
                            0,
                            "scale={scale} cross={cross} contact={contact} free={free} bounds=[{lower},{upper}]"
                        );
                        let physical = if cross {
                            [1., 0., 1., 0., 1., 0., 1., 0., 2.]
                        } else {
                            [1., 0., 1., 0., 1., 1., 1., 1., 2.]
                        }
                        .map(|g| g / scale as f64);
                        let mut augmented = physical;
                        augmented[8] += rows[0].velocity[1] as f64;
                        let radius = if contact { 0.002 * scale as f64 } else { 0. };
                        let expected = solve(
                            &augmented,
                            &[0.01, 0., free as f64],
                            &[
                                Domain::Disk {
                                    static_cap: radius,
                                    dynamic_cap: radius,
                                    moving: true,
                                },
                                Domain::Interval {
                                    lower: rows[0].velocity[2] as f64,
                                    upper: rows[0].velocity[3] as f64,
                                },
                            ],
                        )
                        .unwrap()
                        .impulses;
                        let scalar =
                            bytemuck::pod_read_unaligned::<CoulombDiskSolution>(&data[3][..32]);
                        assert_eq!(scalar.regime, 6);
                        assert!(
                            (scalar.impulse[0] as f64 - expected[2]).abs() / (scale as f64) < 3e-7
                        );
                        let rates = floats(&data[1]);
                        let delta = floats(&data[2]);
                        let impulse = floats(&data[4]);
                        let expected_rates = [
                            (-expected[1] - if cross { 0. } else { expected[2] }) / scale as f64,
                            0.01 + (expected[0] + expected[2]) / scale as f64,
                            0.,
                        ];
                        for k in 0..3 {
                            assert!(
                                (rates[k] as f64 - expected_rates[k]).abs() < 3e-7,
                                "rates={rates:?} expected={expected_rates:?}"
                            );
                            assert!(
                                (rates[k] - delta[k] - if k == 1 { 0.01 } else { 0. }).abs() < 1e-8
                            );
                        }
                        assert!(
                            (rates[7] as f64
                                + if cross {
                                    expected[2] / scale as f64
                                } else {
                                    0.
                                })
                            .abs()
                                < 3e-7
                        );
                        assert!((impulse[0] as f64 + expected[1]).abs() / (scale as f64) < 3e-7);
                        assert!((impulse[1] as f64 - expected[0]).abs() / (scale as f64) < 3e-7);
                        let physical_slip = free as f64
                            + (0..3).map(|j| physical[6 + j] * expected[j]).sum::<f64>();
                        assert!((scalar.slip[0] as f64 - physical_slip).abs() < 3e-7);
                        let gram = floats(&data[5]);
                        let slots: Vec<[u32; 2]> = data[7]
                            .chunks_exact(8)
                            .map(bytemuck::pod_read_unaligned)
                            .collect();
                        let [island, block] = slots[1];
                        let diagonal =
                            island as usize * 64 + (2 * block) as usize * 8 + (2 * block) as usize;
                        assert!((gram[diagonal] as f64 * scale as f64 - 2.).abs() < 1e-6);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-response-gpu"]
fn common_articulated_response_matches_mixed_analytic_solution() {
    for scale in [1e-6_f32, 1., 1e6] {
        for case in 0..5 {
            let cross = case == 3;
            let contact = case != 4;
            let kind = if matches!(case, 1 | 2) { 2 } else { 1 };
            let sign = if case == 1 { -1. } else { 1. };
            let f = Fixture::new(scale, kind, sign, cross, contact);
            let data = f.run();
            assert_eq!(
                bytemuck::pod_read_unaligned::<u32>(&data[0]),
                0,
                "scale={scale} case={case}"
            );
            let rates = floats(&data[1]);
            let delta = floats(&data[2]);
            let scalar: Vec<CoulombDiskSolution> = data[3]
                .chunks_exact(32)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            let impulse = floats(&data[4]);
            let gram = floats(&data[5]);
            let d = 0.002_f32 / 2.0_f32.sqrt();
            let (expected, joint, tangent) = match case {
                0 => ([0.005 - d, 0.005 - d, 0.], -0.005, [-d, -d, 0.]),
                1 => ([-0.005 + d, -0.005 + d, 0.], 0.005, [d, d, 0.]),
                2 => ([0., 0.008, 0.], 0., [0., -0.002, 0.]),
                3 => ([0., 0.004, 0.], -0.004, [0., -0.002, 0.]),
                _ => ([0.005, 0.005, 0.], -0.005, [0.; 3]),
            };
            for k in 0..3 {
                assert!(
                    (rates[k] - expected[k]).abs() < 3e-7,
                    "case={case} rates={rates:?}"
                );
                assert!((impulse[k] / scale - tangent[k]).abs() < 3e-7);
                let initial = if k == 1 { 0.01 * sign } else { 0. };
                assert!((rates[k] - initial - delta[k]).abs() < 1e-8);
            }
            assert!((scalar[0].impulse[0] / scale - joint).abs() < 3e-7);
            assert!((scalar[1].impulse[0] / scale + 0.02).abs() < 3e-7);
            assert!(rates[6].abs() < 1e-7);
            assert!((rates[7] - if cross { 0.004 } else { 0. }).abs() < 3e-7);
            assert_eq!(scalar[2].regime, 4);
            assert_eq!(scalar[2].impulse, [0.; 2]);
            if contact {
                assert!((gram[2] * scale - 1.).abs() < 1e-6);
            }
            let islands = bytemuck::pod_read_unaligned::<u32>(&data[6]);
            assert_eq!(islands, if cross { 2 } else { 3 });
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-response-gpu"]
fn scalar_response_faults_recover_and_inactive_joint_rows_retire() {
    let f = Fixture::new(1., 1, 1., false, true);
    let queue = f.engine.queue().unwrap();
    for case in 0..5 {
        molla_core::gpu_write_buffer!(queue, &f.rows, 0, bytemuck::cast_slice(&f.source_rows));
        molla_core::gpu_write_buffer!(
            queue,
            &f.jacobian,
            0,
            bytemuck::cast_slice(&f.source_jacobian)
        );
        molla_core::gpu_write_buffer!(queue, &f.status, 0, bytemuck::bytes_of(&0u32));
        let mut rows = f.source_rows.clone();
        let mut jac = f.source_jacobian.clone();
        let expected = match case {
            0 => {
                jac[6] = 1.;
                CONTACT_RESPONSE_FAULT_INPUT
            }
            1 => {
                jac[0] = f32::NAN;
                CONTACT_RESPONSE_FAULT_NONFINITE
            }
            2 => {
                rows[0].velocity[1] = 1.;
                CONTACT_RESPONSE_FAULT_INPUT
            }
            3 => {
                rows[2].velocity[0] = 0.01;
                FRICTION_FAULT_CONVERGENCE
            }
            _ => {
                rows[0].endpoints[3] = 0;
                rows[1].endpoints[3] = 0;
                rows[2].endpoints[3] = 0;
                0
            }
        };
        molla_core::gpu_write_buffer!(queue, &f.rows, 0, bytemuck::cast_slice(&rows));
        molla_core::gpu_write_buffer!(queue, &f.jacobian, 0, bytemuck::cast_slice(&jac));
        let data = f.run();
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&data[0]),
            expected,
            "case={case}"
        );
        if expected != 0 {
            let halted = f.run();
            assert_eq!(bytemuck::pod_read_unaligned::<u32>(&halted[0]), expected);
            assert!(halted[1..5].iter().flatten().all(|&v| v == 0));
        } else {
            assert!(data[3].iter().all(|&v| v == 0));
            assert_eq!(bytemuck::pod_read_unaligned::<u32>(&data[6]), 1);
            assert!((floats(&data[1])[1] - 0.008).abs() < 1e-7);
        }
    }
    molla_core::gpu_write_buffer!(queue, &f.rows, 0, bytemuck::cast_slice(&f.source_rows));
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&f.run()[0]), 0);
    molla_core::gpu_write_buffer!(
        queue,
        &f.normal,
        0,
        bytemuck::bytes_of(&NormalContactRow::default()),
    );
    let data = f.run();
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&data[0]), 0);
    assert!(data[4].iter().all(|&v| v == 0));
    assert!((floats(&data[1])[1] - 0.005).abs() < 1e-7);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-response-gpu"]
fn device_generated_distance_rows_share_contact_response_without_double_prediction() {
    use crate::distance_gpu::DistanceGpu;
    use molla_math::SpatialVector;
    use molla_sim::runtime::{BodyDesc, DistanceLimits, JointDesc, RigidScene};
    use molla_sim::{JointType, MassProperties};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for case in 0..3 {
        eprintln!("generated_distance case={case} stage=scene");
        let mut scene = RigidScene::new().unwrap();
        let position = Vec3::new(-1., 1., 0.);
        let body = scene
            .insert_body(BodyDesc {
                params: BodyParams {
                    pose: Transform::from_translation(position),
                    ..Default::default()
                },
                additional_mass: Some(MassProperties {
                    mass: 1.,
                    local_com: Vec3::ZERO,
                    inertia: Mat3::IDENTITY,
                }),
                velocity: SpatialVector::new(Vec3::Y * 0.01, Vec3::ZERO),
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
                    min: if case != 2 {
                        Some(2.0_f64.sqrt())
                    } else {
                        None
                    },
                    max: if case != 1 {
                        Some(2.0_f64.sqrt())
                    } else {
                        None
                    },
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
        let status = input(device, queue, &[0; 4]);
        let delta = input(
            device,
            queue,
            bytemuck::cast_slice(&[0.002_f32, 0., 0., 0., 0., 0.]),
        );
        let prediction = input(
            device,
            queue,
            bytemuck::cast_slice(&[0.002_f32, 0.01, 0., 0., 0., 0.]),
        );
        let rows = DistanceGpu::new_for_coupling(&engine, &scene, &states, &status, &delta, None)
            .unwrap()
            .unwrap();
        assert!(
            rows.dispatch(&engine, 0, 0.01, 2, 0)
                .unwrap_err()
                .to_string()
                .contains("standalone distance solve is not prepared")
        );
        eprintln!("generated_distance case={case} stage=response");
        let normal = input(
            device,
            queue,
            bytemuck::bytes_of(&NormalContactRow {
                point_depth: [-1., 1., 0., 0.],
                normal_force: [0., 0., 1., 2.],
                coefficients: [0.125, 0.1, 0., 1.],
                endpoints: [0, 1, u32::MAX, 0],
                relative_velocity: [0., 0.01, 0., 0.],
                ..Default::default()
            }),
        );
        let dt = input(device, queue, bytemuck::bytes_of(&0.01_f32));
        let response = ResidentContactResponse::with_constraints(
            &engine,
            scene.model(),
            &[ContactResponseState {
                input: &states[0],
                predicted_rates: &prediction,
            }],
            &normal,
            &[],
            rows.scalar_buffers(),
            &dt,
            &status,
            ContactResponseConfig {
                row_capacity: 1,
                disks: CoulombDiskConfig {
                    island_capacity: 2,
                    contacts_per_island: 4,
                    max_iterations: 8192,
                    velocity_tolerance: 1e-7,
                },
            },
        )
        .unwrap();
        let publication = rows
            .scalar_publication(&engine, response.scalar_solution_buffer(), &status)
            .unwrap();
        eprintln!("generated_distance case={case} stage=dispatch");
        rows.prepare_scalar_rows(&engine, 0, 0.01, 2, 0).unwrap();
        let data = capture_then(
            &engine,
            &response,
            &status,
            &[&rows.row_forces, &rows.samples],
            |encoder| publication.encode(encoder),
        );
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&data[0]),
            0,
            "case={case}"
        );
        let rates = floats(&data[1]);
        let scalar = bytemuck::pod_read_unaligned::<CoulombDiskSolution>(&data[3][..32]);
        if case == 1 {
            let speed = 0.002_f32.hypot(0.01);
            assert!((rates[0] - (0.002 - 0.002 * 0.002 / speed)).abs() < 3e-7);
            assert!((rates[1] - (0.01 - 0.002 * 0.01 / speed)).abs() < 3e-7);
            assert_eq!(scalar.impulse[0], 0.);
        } else {
            let expected = 0.006 - 0.002 / 2.0_f32.sqrt();
            assert!(
                (rates[0] - expected).abs() < 3e-7,
                "case={case} rates={rates:?}"
            );
            assert!(
                (rates[1] - expected).abs() < 3e-7,
                "case={case} rates={rates:?}"
            );
            let impulse = if case == 0 { -1. } else { 1. } * 0.004 * 2.0_f32.sqrt();
            assert!((scalar.impulse[0] - impulse).abs() < 3e-7);
        }
        assert!(rates[2..].iter().all(|v| v.abs() < 1e-7));
        let force = if case == 1 { [0.; 3] } else { [0.4, -0.4, 0.] };
        let published = floats(&data[8]);
        let mean = floats(&data[9]);
        for k in 0..3 {
            assert!((published[k] - force[k]).abs() < 3e-5);
            assert!((mean[k] - 0.5 * force[k]).abs() < 3e-5);
        }
        rows.prepare_scalar_rows(&engine, 0, 0.01, 2, 1).unwrap();
        let second = capture_then(
            &engine,
            &response,
            &status,
            &[&rows.row_forces, &rows.samples],
            |encoder| publication.encode(encoder),
        );
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&second[0]), 0);
        for (got, expected) in floats(&second[9])[..3].iter().zip(force) {
            assert!((got - expected).abs() < 3e-5);
        }
        let tag = bytemuck::pod_read_unaligned::<u32>(&second[9][16..20]);
        assert_eq!(
            tag,
            if case == 0 {
                3
            } else if case == 1 {
                0
            } else {
                2
            }
        );
        molla_core::gpu_write_buffer!(queue, &status, 0, bytemuck::bytes_of(&512u32));
        let halted = capture_then(
            &engine,
            &response,
            &status,
            &[&rows.row_forces, &rows.samples],
            |encoder| publication.encode(encoder),
        );
        assert_eq!(halted[9], second[9]);
        assert!(halted[8].iter().all(|&v| v == 0));
        eprintln!("generated_distance case={case} stage=complete");
    }
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-response-gpu"]
fn device_generated_locked_cone_rows_share_off_center_contact_response() {
    use crate::distance_gpu::DistanceGpu;
    use molla_math::SpatialVector;
    use molla_sim::runtime::{BodyDesc, JointDesc, RigidScene, SphericalConeLimits};
    use molla_sim::{JointType, MassProperties};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
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
    let joint = scene
        .insert_joint(JointDesc {
            parent: None,
            child: body,
            kind: JointType::Ball,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_cone_limits(
            joint,
            SphericalConeLimits {
                angle0: Some(0.),
                angle1: Some(0.),
            },
        )
        .unwrap();
    scene
        .set_body_velocity(body, SpatialVector::new(Vec3::ZERO, Vec3::Y * 0.01))
        .unwrap();
    scene.upload_model_to_device(device, queue).unwrap();
    let mut states = [
        scene.state().clone_host().unwrap(),
        scene.state().clone_host().unwrap(),
    ];
    let predicted: Vec<f32> = states[0]
        .joint_qd
        .host()
        .unwrap()
        .iter()
        .map(|&v| v as f32)
        .collect();
    for state in &mut states {
        state.upload_to_device(device, queue).unwrap();
    }
    let status = input(device, queue, &[0; 4]);
    let delta = input(device, queue, &vec![0; predicted.len() * 4]);
    let prediction = input(device, queue, bytemuck::cast_slice(&predicted));
    let rows = DistanceGpu::new_for_coupling(&engine, &scene, &states, &status, &delta, None)
        .unwrap()
        .unwrap();
    let normal = input(
        device,
        queue,
        bytemuck::bytes_of(&NormalContactRow {
            point_depth: [1., 0., 0., 0.],
            normal_force: [1., 0., 0., 2.],
            coefficients: [0.125, 0.1, 0., 1.],
            endpoints: [0, 1, u32::MAX, 0],
            relative_velocity: [0., 0., -0.01, 0.],
            ..Default::default()
        }),
    );
    let dt = input(device, queue, bytemuck::bytes_of(&0.01_f32));
    let response = ResidentContactResponse::with_constraints(
        &engine,
        scene.model(),
        &[ContactResponseState {
            input: &states[0],
            predicted_rates: &prediction,
        }],
        &normal,
        &[],
        rows.scalar_buffers(),
        &dt,
        &status,
        ContactResponseConfig {
            row_capacity: 1,
            disks: CoulombDiskConfig {
                island_capacity: 2,
                contacts_per_island: 4,
                max_iterations: 8192,
                velocity_tolerance: 1e-7,
            },
        },
    )
    .unwrap();
    let publication = rows
        .scalar_publication(&engine, response.scalar_solution_buffer(), &status)
        .unwrap();
    rows.prepare_scalar_rows(&engine, 0, 0.01, 1, 0).unwrap();
    let data = capture_then(
        &engine,
        &response,
        &status,
        &[&rows.row_forces, &rows.samples],
        |encoder| publication.encode(encoder),
    );
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&data[0]), 0);
    assert!(
        floats(&data[1]).iter().all(|v| v.abs() < 1e-7),
        "rates={:?}",
        floats(&data[1])
    );
    let scalar: Vec<CoulombDiskSolution> = data[3]
        .chunks_exact(32)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    assert_eq!(scalar.len(), 2);
    assert!(
        scalar
            .iter()
            .all(|s| s.regime == 4 && s.slip[0].abs() < 1e-7)
    );
    let forces = floats(&data[8]);
    let contact = floats(&data[4]);
    let torque = Vec3::new(forces[0] as f64, forces[1] as f64, forces[2] as f64)
        + Vec3::new(forces[4] as f64, forces[5] as f64, forces[6] as f64)
        + Vec3::X.cross(Vec3::new(contact[0] as f64, contact[1] as f64, contact[2] as f64) / 0.01);
    assert!((torque + Vec3::Y).length() < 3e-5, "torque={torque:?}");
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-coupled-response-gpu"]
fn device_generated_scalar_stop_couples_distance_and_contact_rows() {
    use crate::distance_gpu::DistanceGpu;
    use molla_sim::joint_control::JointAxis;
    use molla_sim::runtime::{BodyDesc, DistanceLimits, JointDesc, RigidScene};
    use molla_sim::{JointType, MassProperties};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut scene = RigidScene::new().unwrap();
    let mut bodies = Vec::new();
    for i in 0..2 {
        bodies.push(
            scene
                .insert_body(BodyDesc {
                    params: BodyParams {
                        pose: Transform::from_translation(Vec3::X * i as f64),
                        ..Default::default()
                    },
                    additional_mass: Some(MassProperties {
                        mass: 1.,
                        local_com: Vec3::ZERO,
                        inertia: Mat3::IDENTITY,
                    }),
                    ..Default::default()
                })
                .unwrap(),
        );
    }
    let slider = scene
        .insert_joint(JointDesc {
            parent: None,
            child: bodies[0],
            kind: JointType::Prismatic,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_joint_limits(slider, JointAxis::LinX, Some([-1., 0.]))
        .unwrap();
    let distance = scene
        .insert_joint(JointDesc {
            parent: Some(bodies[0]),
            child: bodies[1],
            kind: JointType::Distance,
            axis: Vec3::X,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
        })
        .unwrap();
    scene
        .set_distance_limits(
            distance,
            DistanceLimits {
                min: Some(1.),
                max: Some(1.),
            },
        )
        .unwrap();
    let slider_index = scene.joint_index(slider).unwrap().index();
    let b = scene.body_index(bodies[1]).unwrap().index();
    let free_index = scene
        .model()
        .joint_child
        .host()
        .unwrap()
        .iter()
        .position(|&v| v == b as i32)
        .unwrap();
    let mut predicted = vec![0.0_f32; scene.model().joint_dof_total];
    predicted[scene.model().joint_dof_offset.host().unwrap()[slider_index] as usize] = 0.01;
    predicted[scene.model().joint_dof_offset.host().unwrap()[free_index] as usize] = 0.02;
    scene.upload_model_to_device(device, queue).unwrap();
    let mut states = [
        scene.state().clone_host().unwrap(),
        scene.state().clone_host().unwrap(),
    ];
    for state in &mut states {
        state.upload_to_device(device, queue).unwrap();
    }
    let status = input(device, queue, &[0; 4]);
    let prediction = input(device, queue, bytemuck::cast_slice(&predicted));
    let rows = DistanceGpu::new_for_coupling(&engine, &scene, &states, &status, &prediction, None)
        .unwrap()
        .unwrap();
    let normal = input(
        device,
        queue,
        bytemuck::bytes_of(&NormalContactRow {
            point_depth: [1., 0., 0., 0.],
            normal_force: [0., 1., 0., 2.],
            coefficients: [0.125, 0.1, 0., 1.],
            endpoints: [0, 1, u32::MAX, b as u32],
            ..Default::default()
        }),
    );
    let dt = input(device, queue, bytemuck::bytes_of(&0.01_f32));
    let response = ResidentContactResponse::with_constraints(
        &engine,
        scene.model(),
        &[ContactResponseState {
            input: &states[0],
            predicted_rates: &prediction,
        }],
        &normal,
        &[],
        rows.scalar_buffers(),
        &dt,
        &status,
        ContactResponseConfig {
            row_capacity: 1,
            disks: CoulombDiskConfig {
                island_capacity: 2,
                contacts_per_island: 4,
                max_iterations: 8192,
                velocity_tolerance: 1e-7,
            },
        },
    )
    .unwrap();
    let publication = rows
        .scalar_publication(&engine, response.scalar_solution_buffer(), &status)
        .unwrap();
    rows.prepare_scalar_rows(&engine, 0, 0.01, 1, 0).unwrap();
    let data = capture_then(
        &engine,
        &response,
        &status,
        &[&rows.row_forces, &rows.samples],
        |encoder| publication.encode(encoder),
    );
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&data[0]), 0);
    assert!(
        floats(&data[1]).iter().all(|v| v.abs() < 1e-7),
        "rates={:?}",
        floats(&data[1])
    );
    let scalar: Vec<CoulombDiskSolution> = data[3]
        .chunks_exact(32)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    assert_eq!(scalar.iter().filter(|s| s.regime == 4).count(), 1);
    assert_eq!(scalar.iter().filter(|s| s.regime == 5).count(), 2);
    assert_eq!(
        scalar
            .iter()
            .filter(|s| s.regime == 5 && s.impulse[0] > 0.)
            .count(),
        1
    );
    assert_eq!(bytemuck::pod_read_unaligned::<u32>(&data[6]), 1);
}
