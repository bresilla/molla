use super::*;
use crate::articulated_response::ArticulatedResponse;
use crate::contact_publication_gpu::{ContactPublicationBinding, ResidentContactPublication};
use crate::normal_contact_gpu::NormalContactRow;
use crate::penalty_contact::{ContactForce, FrictionRegime};
use molla_core::{BodyId, ShapeId};
use molla_math::{Mat3, Quat, SpatialInertia, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder};

fn input(device: &wgpu::Device, queue: &wgpu::Queue, bytes: &[u8]) -> wgpu::Buffer {
    let buffer = storage(device, "response test input", bytes.len().max(4) as u64);
    if !bytes.is_empty() {
        molla_core::gpu_write_buffer!(queue, &buffer, 0, bytes);
    }
    buffer
}
fn staging(device: &wgpu::Device, size: u64) -> wgpu::Buffer {
    molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("response snapshot"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    )
}
fn read(device: &wgpu::Device, b: &wgpu::Buffer) -> Vec<u8> {
    let (tx, rx) = std::sync::mpsc::channel();
    b.slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let bytes = molla_core::gpu_readback!(b.slice(..)).to_vec();
    b.unmap();
    bytes
}
fn vector(v: [f32; 4]) -> Vec3 {
    Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
}
fn packed(v: Vec3, w: f32) -> [f32; 4] {
    [v.x as f32, v.y as f32, v.z as f32, w]
}

fn model() -> Model {
    let mut builder = ModelBuilder::new();
    let mut bodies = Vec::new();
    for i in 0..5 {
        bodies.push(
            builder.add_body(
                BodyParams::new()
                    .with_spatial_inertia(SpatialInertia {
                        mass: 2.0 + i as f64 * 0.2,
                        com: Vec3::new(0.1, 0.03 * i as f64, -0.05),
                        inertia: Mat3::from_diagonal(Vec3::new(0.3, 0.4, 0.5)),
                    })
                    .with_flags(if i == 3 {
                        BodyFlags::KINEMATIC
                    } else if i == 4 {
                        BodyFlags::FIXED
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
            Transform::IDENTITY,
            Transform::IDENTITY,
        )
        .unwrap();
    builder
        .add_joint_revolute(
            bodies[0],
            bodies[1],
            Transform::from_translation(Vec3::X * 0.5),
            Transform::from_translation(Vec3::X * -0.2),
            Vec3::Z,
        )
        .unwrap();
    builder.begin_articulation();
    builder
        .add_joint_free(
            BodyId::NONE,
            bodies[2],
            Transform::from_translation(Vec3::X * 1.5),
            Transform::IDENTITY,
        )
        .unwrap();
    for body in bodies {
        builder
            .add_shape_sphere(body, 0.1, Transform::IDENTITY)
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    model.shape_surface_velocity[3] = Vec3::new(0.001, 0.0, -0.0005);
    model
}

fn rows(model: &Model, state: &State, reverse: bool) -> (Vec<NormalContactRow>, Vec<ContactForce>) {
    let mut rows = Vec::new();
    let mut readings = Vec::new();
    for (i, (a, b, normal)) in [
        (None, Some(0), Vec3::Y),
        (Some(3), Some(1), Vec3::new(0.2, 0.96, 0.2).normalize()),
        (Some(1), Some(2), Vec3::Z),
        (Some(0), Some(1), Vec3::X),
        (Some(4), Some(2), Vec3::Y),
    ]
    .into_iter()
    .enumerate()
    {
        let (a, b, normal) = if reverse {
            (b, a, -normal)
        } else {
            (a, b, normal)
        };
        let point = state.body_q.host().unwrap()[1].position + Vec3::new(0.1 * i as f64, -0.3, 0.2);
        let point = vector(packed(point, 0.0));
        let normal = vector(packed(normal, 0.0));
        let mut relative = Vec3::ZERO;
        let mut surface = Vec3::ZERO;
        for (body, sign) in [(a, -1.0), (b, 1.0)] {
            if let Some(body) = body {
                let pose = state.body_q.host().unwrap()[body];
                let arm = point - pose.transform_point(model.body_com.host().unwrap()[body]);
                let velocity = state.body_qd.host().unwrap()[body];
                if model.body_flags.host().unwrap()[body] & BodyFlags::FIXED == 0 {
                    relative += sign * (velocity.linear + velocity.angular.cross(arm));
                }
                surface += sign * (pose.rotation * model.shape_surface_velocity[body]);
            }
        }
        rows.push(NormalContactRow {
            point_depth: packed(point, 0.01),
            normal_force: packed(normal, 20.0),
            coefficients: [0.6, 0.2, 0.0, 1.0],
            endpoints: [
                a.map_or(u32::MAX, |v| v as u32),
                b.map_or(u32::MAX, |v| v as u32),
                a.map_or(u32::MAX, |v| v as u32),
                b.map_or(u32::MAX, |v| v as u32),
            ],
            relative_velocity: packed(relative, 0.0),
            surface_velocity: packed(surface, 0.0),
            ..Default::default()
        });
        readings.push(ContactForce {
            bodies: [a.map(BodyId::from_index), b.map(BodyId::from_index)],
            colliders: [a.map(ShapeId::from_index), b.map(ShapeId::from_index)],
            point,
            normal,
            penetration: 0.01,
            friction: 0.2_f32 as f64,
            static_friction: 0.6_f32 as f64,
            friction_regime: FrictionRegime::Inactive,
            applied_friction: 0.0,
            tangent_impulse: Vec3::ZERO,
            normal_force: 20.0,
            sample_dt: 0.01,
            normal_impulse: 0.2,
            force_on_b: normal * 20.0,
            acted: true,
        });
    }
    (rows, readings)
}

fn gram_reference(model: &Model, state: &State, readings: &[ContactForce]) -> Vec<f64> {
    let nrows = readings.len() * 2;
    let mut gram = vec![0.0; nrows * nrows];
    let mut response = ArticulatedResponse::new(model, state);
    let starts = model.articulation_start.host().unwrap();
    let offsets = model.joint_dof_offset.host().unwrap();
    for (art, range) in starts.windows(2).enumerate() {
        let n = (offsets[range[1] as usize] - offsets[range[0] as usize]) as usize;
        let children = &model.joint_child.host().unwrap()[range[0] as usize..range[1] as usize];
        let mut projections = vec![vec![0.0; n]; nrows];
        for (c, reading) in readings.iter().enumerate() {
            let tangent = reading
                .normal
                .cross(if reading.normal.x.abs() < 0.6 {
                    Vec3::X
                } else {
                    Vec3::Y
                })
                .normalize();
            for (axis, direction) in [tangent, reading.normal.cross(tangent)]
                .into_iter()
                .enumerate()
            {
                for (body, sign) in reading.bodies.into_iter().zip([-1.0, 1.0]) {
                    let Some(body) = body else {
                        continue;
                    };
                    let b = body.index();
                    if !children.contains(&(b as i32))
                        || model.body_flags.host().unwrap()[b] & BodyFlags::IMMOVABLE != 0
                    {
                        continue;
                    }
                    let arm = reading.point
                        - state.body_q.host().unwrap()[b]
                            .transform_point(model.body_com.host().unwrap()[b]);
                    let torque = arm.cross(direction);
                    let j = response.jacobian(art, b).unwrap();
                    for k in 0..n {
                        projections[2 * c + axis][k] += sign
                            * (direction.x * j[k]
                                + direction.y * j[n + k]
                                + direction.z * j[2 * n + k]
                                + torque.x * j[3 * n + k]
                                + torque.y * j[4 * n + k]
                                + torque.z * j[5 * n + k]);
                    }
                }
            }
        }
        response.add_gram(art, n, &projections, &mut gram).unwrap();
    }
    gram
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-response-gpu"]
fn articulated_response_matches_cpu_gram_and_generalized_friction() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = model();
    model.upload_to_device(device, queue).unwrap();
    assert_eq!(model.articulation_count, 2);
    let mut states = Vec::new();
    let mut predictions = Vec::new();
    let mut inputs = Vec::new();
    let mut references = Vec::new();
    for case in 0..12 {
        let mut state = model.state().unwrap();
        let offsets = model.joint_coord_offset.host().unwrap();
        state.joint_q.host_mut().unwrap()[offsets[1] as usize] = case as f64 * 0.05;
        let rotation = Quat::from_rotation_y(case as f64 * 0.03);
        state.joint_q.host_mut().unwrap()[3..7].copy_from_slice(&rotation.to_array());
        for (k, v) in state.joint_qd.host_mut().unwrap().iter_mut().enumerate() {
            *v = if case % 3 == 0 {
                0.0
            } else {
                ((k * 7 % 9) as f64 - 4.0) * 0.003
            };
        }
        molla_sim::eval_fk(&model, &mut state).unwrap();
        molla_sim::eval_body_velocities(&model, &mut state).unwrap();
        state.body_qd.host_mut().unwrap()[3] =
            SpatialVector::new(Vec3::X * 0.001, Vec3::Y * 0.0005);
        state.body_qd.host_mut().unwrap()[4] = SpatialVector::new(Vec3::ONE, Vec3::ONE);
        let predicted: Vec<f32> = state
            .joint_qd
            .host()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(k, v)| (*v + ((k * 3 % 7) as f64 - 3.0) * 0.0002) as f32)
            .collect();
        let (normal_rows, mut readings) = rows(&model, &state, case % 2 != 0);
        let gram = gram_reference(&model, &state, &readings);
        let mut rates: Vec<f64> = predicted.iter().map(|&v| v as f64).collect();
        crate::contact_friction::apply(&model, &state, &mut rates, 0.01_f32 as f64, &mut readings)
            .unwrap();
        references.push((gram, rates));
        inputs.push(normal_rows);
        state.upload_to_device(device, queue).unwrap();
        states.push(state);
        predictions.push(input(device, queue, bytemuck::cast_slice(&predicted)));
    }
    let state_bindings: Vec<_> = states
        .iter()
        .zip(&predictions)
        .map(|(input, predicted_rates)| ContactResponseState {
            input,
            predicted_rates,
        })
        .collect();
    let normal = input(device, queue, &vec![0; 8 * 128]);
    let status = input(device, queue, &[0; 4]);
    let dt = input(device, queue, bytemuck::bytes_of(&0.01_f32));
    let config = ContactResponseConfig {
        row_capacity: 8,
        disks: CoulombDiskConfig {
            island_capacity: 2,
            contacts_per_island: 8,
            max_iterations: 8192,
            velocity_tolerance: 1e-7,
        },
    };
    let component = ResidentContactResponse::new(
        &engine,
        &model,
        &state_bindings,
        &normal,
        &[],
        &dt,
        &status,
        config,
    )
    .unwrap();
    let rate_bytes = model.joint_dof_total as u64 * 4;
    let gram_bytes = component.gram_buffer().size();
    let stride = 4 + rate_bytes + gram_bytes + 32;
    let output = staging(device, stride * states.len() as u64);
    let mut encoder = device.create_command_encoder(&Default::default());
    for (case, data) in inputs.iter().enumerate() {
        let mut data = data.clone();
        data.resize(8, NormalContactRow::default());
        let source = input(device, queue, bytemuck::cast_slice(&data));
        encoder.copy_buffer_to_buffer(&source, 0, &normal, 0, normal.size());
        component.encode(&mut encoder, case).unwrap();
        let offset = case as u64 * stride;
        encoder.copy_buffer_to_buffer(&status, 0, &output, offset, 4);
        encoder.copy_buffer_to_buffer(
            component.corrected_rates_buffer(),
            0,
            &output,
            offset + 4,
            rate_bytes,
        );
        encoder.copy_buffer_to_buffer(
            component.gram_buffer(),
            0,
            &output,
            offset + 4 + rate_bytes,
            gram_bytes,
        );
        encoder.copy_buffer_to_buffer(
            component.disks().diagnostic_buffer(),
            0,
            &output,
            offset + 4 + rate_bytes + gram_bytes,
            32,
        );
    }
    queue.submit([encoder.finish()]);
    let bytes = read(device, &output);
    let mut peak_gram: f64 = 0.0;
    let mut peak_rate: f64 = 0.0;
    for (case, frame) in bytes.chunks_exact(stride as usize).enumerate() {
        let code = bytemuck::pod_read_unaligned::<u32>(&frame[..4]);
        let diagnostic: Vec<u32> = frame[frame.len() - 32..]
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        assert_eq!(code, 0, "case={case} diagnostic={diagnostic:?}");
        for (k, row) in frame[4..4 + rate_bytes as usize]
            .chunks_exact(4)
            .enumerate()
        {
            let actual = bytemuck::pod_read_unaligned::<f32>(row) as f64;
            let expected = references[case].1[k];
            peak_rate = peak_rate.max((actual - expected).abs());
            assert!(
                (actual - expected).abs() < 2e-5,
                "case={case} dof={k}: {actual:e} != {expected:e}"
            );
        }
        let actual: Vec<f32> = frame[4 + rate_bytes as usize..frame.len() - 32]
            .chunks_exact(4)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        for r in 0..10 {
            for c in 0..10 {
                let expected = references[case].0[r * 10 + c];
                let error = (actual[r * 16 + c] as f64 - expected).abs();
                peak_gram = peak_gram.max(error);
                assert!(
                    error < 5e-5 * expected.abs().max(1.0),
                    "case={case} [{r},{c}] {} != {expected}",
                    actual[r * 16 + c]
                );
            }
        }
    }
    eprintln!(
        "articulated contact response: peak gram error={peak_gram:e}, peak rate error={peak_rate:e}"
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-response-gpu"]
fn device_contact_friction_holds_loaded_box_and_matches_cpu_trajectory() {
    run_contact_trajectory(Trajectory::Hold);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-response-gpu"]
fn device_contact_friction_slide_matches_analytic_acceleration_and_cpu() {
    run_contact_trajectory(Trajectory::Slide);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-response-gpu"]
fn device_contact_friction_breaks_away_when_load_crosses_static_bound() {
    run_contact_trajectory(Trajectory::Breakaway);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-response-gpu"]
fn device_contact_friction_stops_resticks_and_reverses_under_load() {
    run_contact_trajectory(Trajectory::StopReverse);
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Trajectory {
    Hold,
    Slide,
    Breakaway,
    StopReverse,
}

impl Trajectory {
    fn force(self, step: usize) -> f64 {
        match self {
            Self::Breakaway if step < 500 => 8.0 * step as f64 / 500.0,
            Self::Breakaway if step < 1000 => 8.0,
            Self::Breakaway if step < 2000 => 8.0 + 5.0 * (step - 1000) as f64 / 1000.0,
            Self::Breakaway => 13.0,
            Self::StopReverse if step < 1000 => 1.0,
            Self::StopReverse if step < 2000 => -1.0,
            Self::StopReverse => -12.5,
            _ => 0.0,
        }
    }
}

fn run_contact_trajectory(case: Trajectory) {
    let [position, orientation, velocity] = measure_contact_trajectory(case, 1);
    assert!(position < 3e-5, "position={position:e}");
    assert!(orientation < 1e-3, "orientation={orientation:e}");
    assert!(velocity < 2e-3, "velocity={velocity:e}");
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-response-gpu"]
fn device_contact_friction_breakaway_time_refinement() {
    let mut errors = Vec::new();
    for substeps in [1, 2, 4, 8] {
        errors.push(measure_contact_trajectory(Trajectory::Breakaway, substeps));
    }
    eprintln!("GPU breakaway refinement [position, orientation, velocity]: {errors:?}");
    let [position, orientation, velocity] = errors[3];
    assert!(position < 3e-5, "position={position:e}");
    assert!(orientation < 1e-3, "orientation={orientation:e}");
    assert!(velocity < 2e-3, "velocity={velocity:e}");
}

fn measure_contact_trajectory(case: Trajectory, substeps: usize) -> [f64; 3] {
    use crate::normal_contact_gpu::{ContactStateBuffers, ResidentNormalContacts};
    use crate::{FeatherstoneSolver, Solver};
    use molla_geometry::{
        CollisionPipeline, LbvhBroadPhase, RESIDENT_MANIFOLD_POINTS, ResidentNarrowPhase,
        ResidentShapeBounds,
    };
    use molla_math::{GpuSpatialVector, GpuTransform};
    use molla_sim::{ContactMaterial, Contacts, FrictionCoefficients};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_linear_damping(0.0);
    builder.set_angular_damping(0.0);
    let slope: f64 = match case {
        Trajectory::Hold => 0.3,
        Trajectory::Slide => 0.5,
        _ => 0.0,
    };
    let gravity = Vec3::new(9.81 * slope.sin(), -9.81 * slope.cos(), 0.0);
    builder.set_world_gravity(molla_core::WorldId(0), gravity);
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
    for s in [shape, plane] {
        builder
            .set_shape_contact_material(
                s,
                ContactMaterial {
                    stiffness: 1e5,
                    damping: 300.0,
                    friction: Some(FrictionCoefficients {
                        mu_static: if case == Trajectory::Slide { 0.4 } else { 0.6 },
                        mu_dynamic: 0.2,
                    }),
                    ..Default::default()
                },
            )
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    initial.joint_q.host_mut().unwrap()[1] = 0.5 + 2.0 * gravity.y / 4e5;
    initial.joint_qd.host_mut().unwrap()[0] = match case {
        Trajectory::Slide => 0.1,
        Trajectory::StopReverse => 0.3,
        _ => 0.0,
    };
    molla_sim::eval_fk(&model, &mut initial).unwrap();
    molla_sim::eval_body_velocities(&model, &mut initial).unwrap();
    let dt = 0.001 / substeps as f64;
    let steps = match case {
        Trajectory::Hold => 10000,
        Trajectory::Slide => 1000,
        _ => 2500,
    } * substeps;
    let mut control = model.control();
    let mut cpu = FeatherstoneSolver::new_cpu();
    let mut collision = CollisionPipeline::new();
    let mut contacts = Contacts::with_capacity(64);
    let mut old = initial.clone_host().unwrap();
    let mut next = initial.clone_host().unwrap();
    let mut references = Vec::new();
    let mut reference_regimes = Vec::new();
    for step in 0..steps {
        old.body_f.host_mut().unwrap()[0] =
            SpatialVector::new(Vec3::X * case.force(step / substeps), Vec3::ZERO);
        collision.collide(&model, &old, &mut contacts).unwrap();
        cpu.step(&model, &old, &control, &contacts, dt, &mut next)
            .unwrap();
        references.push((
            next.body_q.host().unwrap()[0],
            next.body_qd.host().unwrap()[0],
        ));
        let mut regimes: Vec<_> = cpu
            .contact_forces()
            .iter()
            .map(|sample| match sample.friction_regime {
                FrictionRegime::Inactive => 0,
                FrictionRegime::Regularized => 1,
                FrictionRegime::Sticking => 2,
                FrictionRegime::Sliding => 3,
            })
            .collect();
        regimes.sort_unstable();
        if case == Trajectory::Breakaway && reference_regimes.last() != Some(&regimes) {
            eprintln!(
                "CPU breakaway step={step} regimes={regimes:?} vx={:e}",
                next.body_qd.host().unwrap()[0].linear.x
            );
        }
        reference_regimes.push(regimes);
        std::mem::swap(&mut old, &mut next);
    }
    model.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    let mut states = [initial.clone_host().unwrap(), initial.clone_host().unwrap()];
    for s in &mut states {
        s.upload_to_device(device, queue).unwrap();
    }
    let mut gpu = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    gpu.prepare_gpu_coordinate_compensation(&states.each_ref())
        .unwrap();
    let mut broad = LbvhBroadPhase::new(device.clone(), queue.clone(), 2, 64).unwrap();
    broad
        .dispatch_no_readback(&[
            (-Vec3::ONE, Vec3::ONE),
            (Vec3::splat(f64::NEG_INFINITY), Vec3::splat(f64::INFINITY)),
        ])
        .unwrap();
    let poses = states.each_ref().map(|s| s.body_q.device_buffer().unwrap());
    let (count, a, b) = broad.pair_buffers();
    let residues = states
        .each_ref()
        .map(|state| gpu.gpu_position_compensation_buffer(state).unwrap().clone());
    let narrow = ResidentNarrowPhase::with_collision_options(
        device,
        &model,
        &poses,
        a,
        b,
        count,
        64,
        molla_geometry::ResidentCollisionOptions {
            plane_position_residues: &residues.each_ref(),
            ..Default::default()
        },
    )
    .unwrap();
    let bounds = ResidentShapeBounds::new(
        device,
        &model,
        &poses,
        broad.primitive_aabb_buffer(),
        narrow.status_buffer(),
        0.0,
    )
    .unwrap();
    let external = input(device, queue, &[0; 32]);
    let forces: Vec<_> = (0..steps)
        .map(|step| GpuSpatialVector {
            linear: packed(Vec3::X * case.force(step / substeps), 0.0),
            angular: [0.0; 4],
        })
        .collect();
    let force_schedule = input(device, queue, bytemuck::cast_slice(&forces));
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
    let step_size = input(device, queue, bytemuck::bytes_of(&(dt as f32)));
    let response = ResidentContactResponse::new(
        &engine,
        &model,
        &[
            ContactResponseState {
                input: &states[0],
                predicted_rates: states[1].joint_qd.device_buffer().unwrap(),
            },
            ContactResponseState {
                input: &states[1],
                predicted_rates: states[0].joint_qd.device_buffer().unwrap(),
            },
        ],
        normal.row_buffer(),
        &[],
        &step_size,
        narrow.status_buffer(),
        ContactResponseConfig {
            row_capacity: 64 * RESIDENT_MANIFOLD_POINTS,
            disks: CoulombDiskConfig {
                island_capacity: 1,
                contacts_per_island: 4,
                max_iterations: 8192,
                velocity_tolerance: 1e-7,
            },
        },
    )
    .unwrap();
    const STRIDE: usize = 572;
    gpu.set_gpu_status_buffer(narrow.status_buffer()).unwrap();
    let publication_bindings: Vec<_> = (0..2)
        .flat_map(|side| {
            [
                ContactPublicationBinding {
                    source: normal.force_buffer(),
                    destination: states[side].body_f.device_buffer().unwrap(),
                    bytes: 32,
                },
                ContactPublicationBinding {
                    source: response.corrected_rates_buffer(),
                    destination: states[1 - side].joint_qd.device_buffer().unwrap(),
                    bytes: 24,
                },
            ]
        })
        .collect();
    let publication =
        ResidentContactPublication::new(device, narrow.status_buffer(), &publication_bindings)
            .unwrap();
    let snapshot = staging(device, steps as u64 * STRIDE as u64);
    let empty_contacts = Contacts::with_capacity(0);
    for step in 0..steps {
        let side = step % 2;
        let mut encoder = device.create_command_encoder(&Default::default());
        if step == 0 {
            narrow.encode_reset_status(&mut encoder);
        }
        encoder.copy_buffer_to_buffer(&force_schedule, step as u64 * 32, &external, 0, 32);
        bounds.encode(&mut encoder, side).unwrap();
        broad.encode_refit(&mut encoder, 2).unwrap();
        narrow.encode(&mut encoder, side).unwrap();
        normal.encode(&mut encoder, side).unwrap();
        publication.encode(&mut encoder, side * 2).unwrap();
        queue.submit([encoder.finish()]);
        let [a, b] = &mut states;
        let (old, next) = if side == 0 { (a, b) } else { (b, a) };
        gpu.step(&model, old, &control, &empty_contacts, dt, next)
            .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        response.encode(&mut encoder, side).unwrap();
        publication.encode(&mut encoder, side * 2 + 1).unwrap();
        queue.submit([encoder.finish()]);
        gpu.drift_gpu(&model, old, &control, dt, next).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        let offset = step as u64 * STRIDE as u64;
        encoder.copy_buffer_to_buffer(narrow.status_buffer(), 0, &snapshot, offset, 4);
        encoder.copy_buffer_to_buffer(
            gpu.gpu_status_buffer().unwrap(),
            0,
            &snapshot,
            offset + 4,
            4,
        );
        encoder.copy_buffer_to_buffer(
            next.body_q.device_buffer().unwrap(),
            0,
            &snapshot,
            offset + 8,
            32,
        );
        encoder.copy_buffer_to_buffer(
            next.body_qd.device_buffer().unwrap(),
            0,
            &snapshot,
            offset + 40,
            32,
        );
        encoder.copy_buffer_to_buffer(
            response.disks().diagnostic_buffer(),
            0,
            &snapshot,
            offset + 72,
            16,
        );
        encoder.copy_buffer_to_buffer(response.gram_buffer(), 0, &snapshot, offset + 88, 256);
        encoder.copy_buffer_to_buffer(
            response.free_velocity_buffer(),
            0,
            &snapshot,
            offset + 344,
            32,
        );
        encoder.copy_buffer_to_buffer(response.disk_input_buffer(), 0, &snapshot, offset + 376, 64);
        encoder.copy_buffer_to_buffer(
            response.disks().solution_buffer(),
            0,
            &snapshot,
            offset + 440,
            128,
        );
        encoder.copy_buffer_to_buffer(
            response.islands().contact_counts_buffer(),
            0,
            &snapshot,
            offset + 568,
            4,
        );
        queue.submit([encoder.finish()]);
    }
    let bytes = read(device, &snapshot);
    let mut peak_position: f64 = 0.0;
    let mut peak_orientation: f64 = 0.0;
    let mut peak_velocity: f64 = 0.0;
    let mut peak_drift: f64 = 0.0;
    let mut settled = Vec3::ZERO;
    let mut creep: f64 = 0.0;
    let mut slide_begin = 0.0;
    let mut slide_end = 0.0;
    let mut peak_kkt: f64 = 0.0;
    let mut samples = Vec::new();
    let mut first_sliding = None;
    for (step, frame) in bytes.chunks_exact(STRIDE).enumerate() {
        let code = bytemuck::pod_read_unaligned::<u32>(&frame[..4]);
        let diagnostic = bytemuck::pod_read_unaligned::<[u32; 4]>(&frame[72..88]);
        assert_eq!(code, 0, "step={step} diagnostic={diagnostic:?}");
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&frame[4..8]),
            0,
            "step={step}"
        );
        let pose = bytemuck::pod_read_unaligned::<GpuTransform>(&frame[8..40]);
        let velocity = bytemuck::pod_read_unaligned::<GpuSpatialVector>(&frame[40..72]);
        let point = vector(pose.position);
        let rotation = Quat::from_array(pose.rotation.map(f64::from)).normalize();
        let cosine = rotation
            .dot(references[step].0.rotation.normalize())
            .abs()
            .min(1.0);
        peak_orientation = peak_orientation.max(2.0 * cosine.acos());
        let rate = vector(velocity.linear);
        if step == 200 * substeps - 1 {
            slide_begin = rate.x;
        }
        if step == 1000 * substeps - 1 {
            slide_end = rate.x;
        }
        if step == 1000 * substeps - 1 {
            settled = point;
        }
        if step >= 1000 * substeps {
            creep = creep.max((point.x - settled.x).hypot(point.z - settled.z));
        }
        peak_position = peak_position.max((point - references[step].0.position).length());
        peak_velocity = peak_velocity.max((rate - references[step].1.linear).length());
        peak_drift = peak_drift.max(point.x.hypot(point.z));
        let count = bytemuck::pod_read_unaligned::<u32>(&frame[568..572]) as usize;
        assert_eq!(count, 4, "step={step}");
        let gram: Vec<f64> = frame[88..344]
            .chunks_exact(4)
            .map(|b| bytemuck::pod_read_unaligned::<f32>(b) as f64)
            .collect();
        let free: Vec<f64> = frame[344..376]
            .chunks_exact(4)
            .map(|b| bytemuck::pod_read_unaligned::<f32>(b) as f64)
            .collect();
        let solutions: Vec<crate::coulomb_disk_gpu::CoulombDiskSolution> = frame[440..568]
            .chunks_exact(32)
            .map(bytemuck::pod_read_unaligned)
            .collect();
        let mut regimes: Vec<_> = solutions.iter().map(|sample| sample.regime).collect();
        regimes.sort_unstable();
        if case == Trajectory::Breakaway
            && samples.last().map(|s: &(Vec3, Vec3, Vec<u32>)| &s.2) != Some(&regimes)
        {
            eprintln!(
                "GPU breakaway step={step} regimes={regimes:?} vx={:e}, CPU vx={:e}",
                rate.x, references[step].1.linear.x
            );
        }
        if regimes.contains(&3) && first_sliding.is_none() {
            first_sliding = Some(step);
        }
        samples.push((point, rate, regimes.clone()));
        if step % (100 * substeps) == 100 * substeps - 1 {
            assert_eq!(
                regimes, reference_regimes[step],
                "case={case:?} step={step}"
            );
        }
        let impulses: Vec<_> = solutions
            .iter()
            .flat_map(|r| r.impulse.map(f64::from))
            .collect();
        let slip: Vec<_> = (0..8)
            .map(|r| free[r] + (0..8).map(|c| gram[r * 8 + c] * impulses[c]).sum::<f64>())
            .collect();
        let energy: f64 = (0..8)
            .map(|r| 0.5 * impulses[r] * (free[r] + slip[r]))
            .sum();
        assert!(energy <= 1e-10, "step={step} energy={energy:e}");
        for c in 0..count {
            let authored = bytemuck::pod_read_unaligned::<crate::coulomb_disk_gpu::CoulombDiskInput>(
                &frame[376 + c * 16..392 + c * 16],
            );
            assert!(
                solutions[c].applied_cap == authored.static_cap
                    || solutions[c].applied_cap == authored.dynamic_cap
            );
            if solutions[c].regime == 3 {
                assert_eq!(
                    solutions[c].applied_cap, authored.dynamic_cap,
                    "case={case:?} step={step} contact={c}"
                );
            }
            let cap = solutions[c].applied_cap as f64;
            let length = impulses[2 * c].hypot(impulses[2 * c + 1]);
            assert!(length <= cap * (1.0 + 2e-6));
            let speed = slip[2 * c].hypot(slip[2 * c + 1]);
            let error = if cap == 0.0 {
                0.0
            } else if length == 0.0 {
                speed
            } else {
                let direction = [impulses[2 * c] / length, impulses[2 * c + 1] / length];
                let pressure =
                    (-(slip[2 * c] * direction[0] + slip[2 * c + 1] * direction[1])).max(0.0);
                let r = 2 * c;
                let curvature = (gram[r * 8 + r].abs() + gram[r * 8 + r + 1].abs())
                    .max(gram[(r + 1) * 8 + r + 1].abs() + gram[r * 8 + r + 1].abs());
                (slip[2 * c] + pressure * direction[0])
                    .hypot(slip[2 * c + 1] + pressure * direction[1])
                    .max(speed * (1.0 - length / cap).abs())
                    .max(speed.min(curvature * (cap - length).abs()))
            };
            peak_kkt = peak_kkt.max(error);
            assert!(
                error <= 1e-7,
                "step={step} contact={c} host KKT={error:e}, GPU={:?}, slip={:?}, free={free:?}, gram={gram:?}, impulses={impulses:?}",
                solutions[c],
                &slip[2 * c..2 * c + 2]
            );
        }
    }
    eprintln!(
        "GPU contact trajectory: case={case:?} substeps={substeps}, peak drift={peak_drift:e}, settled creep={creep:e}, CPU position error={peak_position:e}, orientation error={peak_orientation:e}, velocity error={peak_velocity:e}"
    );
    eprintln!("integrated published-impulse host KKT peak={peak_kkt:e}");
    if case == Trajectory::Hold {
        assert!(peak_drift < 2e-4);
        assert!(creep < 1e-6);
    } else if case == Trajectory::Slide {
        let acceleration = (slide_end - slide_begin) / 0.8;
        let expected = gravity.x + 0.2 * gravity.y;
        eprintln!("GPU kinetic slide: acceleration={acceleration:e}, analytic={expected:e}");
        assert!((acceleration - expected).abs() < 0.02 * expected.abs());
    } else if case == Trajectory::Breakaway {
        let start = first_sliding.expect("load must break static friction");
        eprintln!(
            "GPU breakaway: first sliding step={start}, force={} N",
            case.force(start / substeps)
        );
        assert!((1740..1780).contains(&(start / substeps)), "step={start}");
        for &(point, rate, ref regimes) in &samples[700 * substeps..1000 * substeps] {
            assert!(point.x.abs() < 2e-4);
            assert!(rate.x.abs() < 1e-5);
            assert_eq!(regimes, &[2; 4]);
        }
        let acceleration =
            (samples[2500 * substeps - 1].1.x - samples[2100 * substeps - 1].1.x) / 0.4;
        let expected = 13.0 / 2.0 - 0.2 * 9.81;
        assert!((acceleration - expected).abs() < 0.02 * expected);
        assert!(samples[2500 * substeps - 1].1.x > 1.0);
    } else {
        assert_eq!(first_sliding, Some(0));
        let resting = samples[500 * substeps].0;
        let mut rest_creep: f64 = 0.0;
        for (step, &(point, rate, ref regimes)) in samples
            .iter()
            .enumerate()
            .take(2000 * substeps)
            .skip(500 * substeps)
        {
            rest_creep = rest_creep.max((point.x - resting.x).abs());
            if !(1000 * substeps..1250 * substeps).contains(&step) {
                assert!(rate.x.abs() < 1e-5, "step={step} rest speed={:e}", rate.x);
                assert_eq!(regimes, &[2; 4]);
            }
        }
        eprintln!("GPU stop/re-stick: reversal-window displacement={rest_creep:e}");
        assert!(rest_creep < 2e-4);
        assert!(samples[2500 * substeps - 1].1.x < -1.0);
        assert_eq!(samples[2500 * substeps - 1].2, [3; 4]);
        let acceleration =
            (samples[2500 * substeps - 1].1.x - samples[2100 * substeps - 1].1.x) / 0.4;
        let expected = -12.5 / 2.0 + 0.2 * 9.81;
        assert!((acceleration - expected).abs() < 0.02 * expected.abs());
    }
    assert_eq!(broad.diagnostics().pair_readbacks, 0);
    [peak_position, peak_orientation, peak_velocity]
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-response-gpu"]
fn contact_response_faults_preserve_status_and_retire_rates_and_impulses() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = model();
    let mut state = model.state().unwrap();
    molla_sim::eval_fk(&model, &mut state).unwrap();
    molla_sim::eval_body_velocities(&model, &mut state).unwrap();
    let (mut valid, _) = rows(&model, &state, false);
    valid.resize(8, NormalContactRow::default());
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let normal = input(device, queue, bytemuck::cast_slice(&valid));
    let predicted_values = vec![0.0001_f32; model.joint_dof_total];
    let predicted = input(device, queue, bytemuck::cast_slice(&predicted_values));
    let dt = input(device, queue, bytemuck::bytes_of(&0.01_f32));
    let status = input(device, queue, &[0; 4]);
    let states = [ContactResponseState {
        input: &state,
        predicted_rates: &predicted,
    }];
    let config = ContactResponseConfig {
        row_capacity: 8,
        disks: CoulombDiskConfig {
            island_capacity: 2,
            contacts_per_island: 8,
            max_iterations: 8192,
            velocity_tolerance: 1e-7,
        },
    };
    let response =
        ResidentContactResponse::new(&engine, &model, &states, &normal, &[], &dt, &status, config)
            .unwrap();
    let rate_bytes = model.joint_dof_total as u64 * 4;
    let retired_bytes = 4 + rate_bytes * 2 + 8 * 16;
    let stride = retired_bytes + rate_bytes;
    let publication = ResidentContactPublication::new(
        device,
        &status,
        &[ContactPublicationBinding {
            source: response.corrected_rates_buffer(),
            destination: &predicted,
            bytes: rate_bytes,
        }],
    )
    .unwrap();
    let output = staging(device, stride * 8 * 4);
    let mut encoder = device.create_command_encoder(&Default::default());
    assert!(response.encode(&mut encoder, 1).is_err());
    for case in 0..8 {
        for phase in 0..4 {
            let mut data = if phase == 2 {
                vec![NormalContactRow::default(); 8]
            } else {
                valid.clone()
            };
            let mut time = 0.01_f32;
            let mut rates = predicted_values.clone();
            if phase == 1 {
                match case {
                    0 => time = 0.0,
                    1 => time = f32::NAN,
                    2 => data[0].normal_force[0..3].fill(0.0),
                    3 => data[0].point_depth[0] = f32::NAN,
                    4 => rates[0] = f32::NAN,
                    5 => {
                        time = 1e30;
                        data[0].normal_force[3] = 1e30;
                    }
                    6 => time = 1e-40,
                    _ => {
                        let stopped = input(device, queue, bytemuck::bytes_of(&(1_u32 << 30)));
                        encoder.copy_buffer_to_buffer(&stopped, 0, &status, 0, 4);
                    }
                }
            }
            if phase == 0 || phase == 3 {
                encoder.clear_buffer(&status, 0, None);
            }
            for (bytes, target) in [
                (bytemuck::cast_slice(&data), &normal),
                (bytemuck::bytes_of(&time), &dt),
                (bytemuck::cast_slice(&rates), &predicted),
            ] {
                let source = input(device, queue, bytes);
                encoder.copy_buffer_to_buffer(&source, 0, target, 0, bytes.len() as u64);
            }
            response.encode(&mut encoder, 0).unwrap();
            publication.encode(&mut encoder, 0).unwrap();
            let offset = (case * 4 + phase) as u64 * stride;
            encoder.copy_buffer_to_buffer(&status, 0, &output, offset, 4);
            encoder.copy_buffer_to_buffer(
                response.corrected_rates_buffer(),
                0,
                &output,
                offset + 4,
                rate_bytes,
            );
            encoder.copy_buffer_to_buffer(
                response.velocity_delta_buffer(),
                0,
                &output,
                offset + 4 + rate_bytes,
                rate_bytes,
            );
            encoder.copy_buffer_to_buffer(
                response.world_impulse_buffer(),
                0,
                &output,
                offset + 4 + 2 * rate_bytes,
                8 * 16,
            );
            encoder.copy_buffer_to_buffer(
                &predicted,
                0,
                &output,
                offset + retired_bytes,
                rate_bytes,
            );
        }
    }
    queue.submit([encoder.finish()]);
    let bytes = read(device, &output);
    for (case, frames) in bytes.chunks_exact(stride as usize * 4).enumerate() {
        let frames: Vec<_> = frames.chunks_exact(stride as usize).collect();
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&frames[0][..4]),
            0,
            "case={case}"
        );
        assert!(frames[0][4..].iter().any(|&v| v != 0));
        let code = bytemuck::pod_read_unaligned::<u32>(&frames[1][..4]);
        if case <= 2 {
            assert_eq!(code, CONTACT_RESPONSE_FAULT_INPUT);
        } else if case == 6 {
            assert!(
                code == CONTACT_RESPONSE_FAULT_INPUT || code == CONTACT_RESPONSE_FAULT_NONFINITE
            );
        } else if case == 7 {
            assert_eq!(code, 1 << 30);
        } else {
            assert_eq!(code, CONTACT_RESPONSE_FAULT_NONFINITE);
        }
        assert_eq!(bytemuck::pod_read_unaligned::<u32>(&frames[2][..4]), code);
        assert!(frames[2][4..retired_bytes as usize].iter().all(|&v| v == 0));
        for (phase, frame) in frames.iter().enumerate() {
            let published = &frame[retired_bytes as usize..];
            if phase == 0 || phase == 3 {
                assert_eq!(published, &frame[4..4 + rate_bytes as usize]);
            } else {
                let mut expected = predicted_values.clone();
                if phase == 1 && case == 4 {
                    expected[0] = f32::NAN;
                }
                assert_eq!(published, bytemuck::cast_slice::<f32, u8>(&expected));
            }
        }
        assert_eq!(frames[3], frames[0]);
    }
    assert!(
        ResidentContactResponse::new(&engine, &model, &states, &status, &[], &dt, &status, config)
            .is_err()
    );
    let huge = ContactResponseConfig {
        row_capacity: 8,
        disks: CoulombDiskConfig {
            contacts_per_island: u32::MAX,
            ..config.disks
        },
    };
    assert!(
        ResidentContactResponse::new(&engine, &model, &states, &normal, &[], &dt, &status, huge)
            .is_err()
    );
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-response-gpu"]
fn contact_publication_preserves_halted_predictor_and_redrift() {
    use crate::{FeatherstoneSolver, Solver};
    use molla_sim::Contacts;

    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    builder.set_linear_damping(0.0);
    builder.set_angular_damping(0.0);
    builder.set_world_gravity(molla_core::WorldId(0), Vec3::X);
    let body = builder.add_body(BodyParams::new());
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    let mut initial = model.state().unwrap();
    initial.joint_qd.host_mut().unwrap()[0] = 1.25;
    molla_sim::eval_fk(&model, &mut initial).unwrap();
    molla_sim::eval_body_velocities(&model, &mut initial).unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(device, queue).unwrap();
    let mut states = [initial.clone_host().unwrap(), initial.clone_host().unwrap()];
    for state in &mut states {
        state.upload_to_device(device, queue).unwrap();
    }
    let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();
    solver
        .prepare_gpu_coordinate_compensation(&states.each_ref())
        .unwrap();
    let status = input(device, queue, &[0; 4]);
    solver.set_gpu_status_buffer(&status).unwrap();
    let normal = input(device, queue, &[0; 128]);
    let dt = input(device, queue, bytemuck::bytes_of(&0.001_f32));
    let response = ResidentContactResponse::new(
        &engine,
        &model,
        &[
            ContactResponseState {
                input: &states[0],
                predicted_rates: states[1].joint_qd.device_buffer().unwrap(),
            },
            ContactResponseState {
                input: &states[1],
                predicted_rates: states[0].joint_qd.device_buffer().unwrap(),
            },
        ],
        &normal,
        &[],
        &dt,
        &status,
        ContactResponseConfig {
            row_capacity: 1,
            disks: CoulombDiskConfig {
                island_capacity: 1,
                contacts_per_island: 1,
                max_iterations: 8192,
                velocity_tolerance: 1e-7,
            },
        },
    )
    .unwrap();
    let publications: Vec<_> = (0..2)
        .map(|side| ContactPublicationBinding {
            source: response.corrected_rates_buffer(),
            destination: states[1 - side].joint_qd.device_buffer().unwrap(),
            bytes: 24,
        })
        .collect();
    let publication = ResidentContactPublication::new(device, &status, &publications).unwrap();
    let statuses = [
        0_u32,
        1 << 30,
        1 << 30,
        CONTACT_RESPONSE_FAULT_INPUT,
        CONTACT_RESPONSE_FAULT_INPUT,
        0,
    ];
    let schedule = input(device, queue, bytemuck::cast_slice(&statuses));
    let snapshot = staging(device, 164 * statuses.len() as u64);
    let contacts = Contacts::with_capacity(0);
    for step in 0..statuses.len() {
        let side = step % 2;
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(&schedule, step as u64 * 4, &status, 0, 4);
        queue.submit([encoder.finish()]);
        let [a, b] = &mut states;
        let (old, next) = if side == 0 { (a, b) } else { (b, a) };
        solver
            .step(&model, old, &control, &contacts, 0.001, next)
            .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        response.encode(&mut encoder, side).unwrap();
        publication.encode(&mut encoder, side).unwrap();
        queue.submit([encoder.finish()]);
        solver
            .drift_gpu(&model, old, &control, 0.001, next)
            .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        let mut offset = step as u64 * 164;
        for (buffer, bytes) in [
            (&status, 4),
            (next.body_q.device_buffer().unwrap(), 32),
            (next.body_qd.device_buffer().unwrap(), 32),
            (next.joint_q.device_buffer().unwrap(), 28),
            (next.joint_qd.device_buffer().unwrap(), 24),
            (solver.gpu_coordinate_compensation_buffer(next).unwrap(), 28),
            (solver.gpu_position_compensation_buffer(next).unwrap(), 16),
        ] {
            encoder.copy_buffer_to_buffer(buffer, 0, &snapshot, offset, bytes);
            offset += bytes;
        }
        queue.submit([encoder.finish()]);
    }
    let bytes = read(device, &snapshot);
    let frames: Vec<_> = bytes.chunks_exact(164).collect();
    for (step, frame) in frames.iter().enumerate() {
        assert_eq!(
            bytemuck::pod_read_unaligned::<u32>(&frame[..4]),
            statuses[step]
        );
        if (1..5).contains(&step) {
            assert_eq!(&frame[4..], &frames[0][4..], "halted step={step}");
        }
    }
    let rate = |frame: &[u8]| bytemuck::pod_read_unaligned::<f32>(&frame[96..100]);
    assert!((rate(frames[0]) - 1.251).abs() < 1e-6);
    assert!((rate(frames[5]) - 1.252).abs() < 1e-6);
    assert_ne!(&frames[0][4..36], &frames[5][4..36]);
}

#[test]
#[ignore = "requires GPU; run via make test-primitives-contact-response-gpu"]
fn coordinate_null_mass_is_supported_and_oblique_rank_deficiency_fails_closed() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for angle in [0.0_f64, 0.37] {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 2.0,
            com: Vec3::ZERO,
            inertia: Mat3::from_diagonal(Vec3::new(0.0, 1.0, 1.0)),
        }));
        builder
            .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let mut model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        state.joint_q.host_mut().unwrap()[3..7]
            .copy_from_slice(&Quat::from_rotation_y(angle).to_array());
        molla_sim::eval_fk(&model, &mut state).unwrap();
        molla_sim::eval_body_velocities(&model, &mut state).unwrap();
        model.upload_to_device(device, queue).unwrap();
        state.upload_to_device(device, queue).unwrap();
        let normal = input(
            device,
            queue,
            bytemuck::bytes_of(&NormalContactRow {
                point_depth: [0.0, -0.5, 0.0, 0.01],
                normal_force: [0.0, 1.0, 0.0, 10.0],
                coefficients: [0.6, 0.2, 0.0, 1.0],
                endpoints: [u32::MAX, 0, u32::MAX, 0],
                ..Default::default()
            }),
        );
        let predicted = input(device, queue, &[0; 24]);
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
                    max_iterations: 256,
                    velocity_tolerance: 1e-7,
                },
            },
        )
        .unwrap();
        let output = staging(device, 44);
        let mut encoder = device.create_command_encoder(&Default::default());
        response.encode(&mut encoder, 0).unwrap();
        encoder.copy_buffer_to_buffer(&status, 0, &output, 0, 4);
        encoder.copy_buffer_to_buffer(response.corrected_rates_buffer(), 0, &output, 4, 24);
        encoder.copy_buffer_to_buffer(response.gram_buffer(), 0, &output, 28, 16);
        queue.submit([encoder.finish()]);
        let result = read(device, &output);
        let code = bytemuck::pod_read_unaligned::<u32>(&result[..4]);
        if angle == 0.0 {
            assert_eq!(code, 0);
            let gram: Vec<f32> = result[28..]
                .chunks_exact(4)
                .map(bytemuck::pod_read_unaligned)
                .collect();
            assert!((gram[0] - 0.5).abs() < 1e-6);
            assert!((gram[3] - 0.75).abs() < 1e-6);
        } else {
            assert_eq!(code, CONTACT_RESPONSE_FAULT_MASS);
            assert!(result[4..28].iter().all(|&b| b == 0));
        }
    }
}
