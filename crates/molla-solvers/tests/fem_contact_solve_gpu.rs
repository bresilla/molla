#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::{BodyId, WorldId};
use molla_kernels::soft_rigid_contact::SoftRigidContactGpu;
use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use molla_kernels::{JacobianKernel, JacobianRequest, MassMatrixKernel};
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3};
use molla_sim::{BodyParams, Model, ModelBuilder, State, eval_fk};
use molla_solvers::soft_rigid_contact::{SoftRigidContact, SoftRigidContactConfig, solve_cpu};
use wgpu::util::DeviceExt;

fn read<T: bytemuck::Pod>(engine: &Engine, buffer: &wgpu::Buffer, count: usize) -> Vec<T> {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let size = (std::mem::size_of::<T>() * count) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("FEM test acceptance readback"),
        size,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec()
}

fn rigid_model() -> (Model, State) {
    let mut builder = ModelBuilder::new();
    builder.begin_articulation();
    let carrier = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 750.0,
        com: Vec3::new(0.02, 0.03, -0.04),
        inertia: Mat3::from_diagonal(Vec3::new(90.0, 120.0, 140.0)),
    }));
    let rotor = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 8.0,
        com: Vec3::ZERO,
        inertia: Mat3::from_diagonal(Vec3::splat(0.3)),
    }));
    builder
        .add_joint_free(
            BodyId::NONE,
            carrier,
            Transform::IDENTITY,
            Transform::IDENTITY,
        )
        .unwrap();
    builder
        .add_joint_revolute(
            carrier,
            rotor,
            Transform::from_translation(Vec3::new(0.0, 0.5, 0.0)),
            Transform::IDENTITY,
            Vec3::X,
        )
        .unwrap();
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    state.joint_q.host_mut().unwrap()[7] = 0.37;
    state.joint_qd.host_mut().unwrap()[6] = 0.7;
    eval_fk(&model, &mut state).unwrap();
    molla_sim::eval_body_velocities(&model, &mut state).unwrap();
    (model, state)
}

fn response_buffers(engine: &Engine, model: &Model, state: &State) -> (wgpu::Buffer, wgpu::Buffer) {
    let mass = MassMatrixKernel::new(engine, model).unwrap();
    let m = mass.create_output_buffer(engine.device().unwrap());
    mass.dispatch(engine, model, state, &m).unwrap();
    let jacobian = JacobianKernel::new(engine, model).unwrap();
    let n = model.joint_dof_total;
    let j = jacobian.create_output_buffer(engine.device().unwrap(), model.body_count * 6 * n);
    let requests: Vec<_> = model
        .body_com
        .host()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(body, p)| JacobianRequest {
            articulation: 0,
            target_body: body as u32,
            output_offset: (body * 6 * n) as u32,
            body_q_offset: 0,
            point_local: [p.x as f32, p.y as f32, p.z as f32, 0.0],
        })
        .collect();
    jacobian
        .dispatch(engine, model, state, &requests, &j)
        .unwrap();
    (m, j)
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_two_way_contact_matches_articulated_cpu_oracle() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut dense_reference = None;
    for (friction, fault, locked, sparse) in [
        (0.0, 0, false, 0),
        (0.4, 0, false, 0),
        (1.0, 0, false, 0),
        (0.4, 0, true, 0),
        (0.4, 1, false, 0),
        (0.4, 2, false, 0),
        (0.4, 0, false, 1),
        (0.4, 0, false, 2),
        (0.4, 0, false, 3),
    ] {
        let (mut rigid, mut gpu_rigid) = rigid_model();
        if locked {
            rigid.joint_limit_lower.host_mut().unwrap()[6] = 0.0;
            rigid.joint_limit_upper.host_mut().unwrap()[6] = 0.0;
            gpu_rigid.joint_q.host_mut().unwrap()[7] = 0.0;
            gpu_rigid.joint_qd.host_mut().unwrap()[6] = 0.0;
            eval_fk(&rigid, &mut gpu_rigid).unwrap();
            molla_sim::eval_body_velocities(&rigid, &mut gpu_rigid).unwrap();
        }
        let initial_rigid_velocity = gpu_rigid.joint_qd.host().unwrap().to_vec();
        let mut builder = ModelBuilder::new();
        for p in [
            Vec3::new(0.1, 0.8, -0.02),
            Vec3::new(0.1, 0.5, 0.3),
            Vec3::new(0.1, 0.0, 0.2),
        ] {
            builder.add_particle(p, 0.2, 0.0, WorldId(0));
        }
        let mut soft = builder.build().unwrap();
        let mut gpu_soft = soft.state().unwrap();
        for v in gpu_soft.particle_qd.host_mut().unwrap() {
            *v = Vec3::new(0.3, -0.8, -0.6);
        }
        let contacts = [
            SoftRigidContact {
                particle: 0,
                body: Some(1),
                normal: Vec3::Z,
                separation: 0.0,
                friction,
            },
            SoftRigidContact {
                particle: 1,
                body: Some(1),
                normal: Vec3::Y,
                separation: -0.0001,
                friction,
            },
            SoftRigidContact {
                particle: 2,
                body: None,
                normal: Vec3::Y,
                separation: 0.0,
                friction,
            },
        ];
        let mut cpu_soft = gpu_soft.clone_host().unwrap();
        let mut cpu_rigid = gpu_rigid.clone_host().unwrap();
        let cpu_report = solve_cpu(
            &soft,
            &mut cpu_soft,
            &rigid,
            &mut cpu_rigid,
            &contacts,
            1.0 / 2400.0,
            SoftRigidContactConfig {
                iterations: 60,
                ..Default::default()
            },
        )
        .unwrap();
        soft.upload_to_device(device, queue).unwrap();
        rigid.upload_to_device(device, queue).unwrap();
        gpu_soft.upload_to_device(device, queue).unwrap();
        gpu_rigid.upload_to_device(device, queue).unwrap();
        gpu_rigid.body_q.mark_device_dirty();
        let (mass, jacobians) = response_buffers(&engine, &rigid, &gpu_rigid);
        if fault == 1 {
            queue.write_buffer(&mass, 0, bytemuck::bytes_of(&-1.0_f32));
        }
        let mut gpu_contacts: Vec<_> = contacts
            .iter()
            .map(|c| SoftRigidContactGpu {
                normal_gap: [
                    c.normal.x as f32,
                    c.normal.y as f32,
                    c.normal.z as f32,
                    c.separation as f32,
                ],
                material: [c.friction as f32, 0.0, 0.0, 0.0],
                ids: [
                    c.particle as u32,
                    c.body.map_or(u32::MAX, |b| b as u32),
                    0,
                    1,
                ],
            })
            .collect();
        if fault == 2 {
            gpu_contacts[0].ids[0] = u32::MAX;
        }
        let (candidate_count, indices) = match sparse {
            0 => (3, [0, 1, 2]),
            1 => (131, [0, 64, 130]),
            2 => (129, [63, 64, 128]),
            _ => (65536, [0, 32768, 65535]),
        };
        let mut candidates = vec![
            SoftRigidContactGpu {
                normal_gap: [0.0; 4],
                material: [0.0; 4],
                ids: [0; 4],
            };
            candidate_count
        ];
        for (contact, index) in gpu_contacts.iter().zip(indices) {
            candidates[index] = *contact;
        }
        let contact_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("test contacts"),
            contents: bytemuck::cast_slice(&candidates),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let kernel =
            SoftRigidSolveKernel::new(device, rigid.joint_dof_total as u32, candidate_count as u32)
                .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        kernel
            .encode(
                device,
                &mut encoder,
                SoftRigidSolveBindings {
                    positions: gpu_soft.particle_q.device_buffer().unwrap(),
                    velocities: gpu_soft.particle_qd.device_buffer().unwrap(),
                    inverse_mass: soft.particle_inv_mass.device_buffer().unwrap(),
                    contacts: &contact_buffer,
                    body_poses: gpu_rigid.body_q.device_buffer().unwrap(),
                    body_com: rigid.body_com.device_buffer().unwrap(),
                    jacobians: &jacobians,
                    mass_matrix: &mass,
                    rigid_velocities: gpu_rigid.joint_qd.device_buffer().unwrap(),
                    body_velocities: gpu_rigid.body_qd.device_buffer().unwrap(),
                    particle_count: soft.particle_count as u32,
                    body_count: rigid.body_count as u32,
                },
                SoftRigidSolveSettings {
                    iterations: 60,
                    dt: 1.0 / 2400.0,
                    penetration_recovery: 0.2,
                    max_recovery_speed: 0.5,
                },
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        assert_eq!(read::<u32>(&engine, kernel.status(), 1), [fault]);
        assert_eq!(
            read::<u32>(&engine, kernel.active_rows(), 4),
            [3, indices[0] as u32, indices[1] as u32, indices[2] as u32]
        );
        if fault != 0 {
            let rigid_after = read::<f32>(
                &engine,
                gpu_rigid.joint_qd.device_buffer().unwrap(),
                rigid.joint_dof_total,
            );
            assert_eq!(
                rigid_after,
                initial_rigid_velocity
                    .iter()
                    .map(|v| *v as f32)
                    .collect::<Vec<_>>()
            );
            let soft_after = read::<[f32; 4]>(
                &engine,
                gpu_soft.particle_qd.device_buffer().unwrap(),
                soft.particle_count,
            );
            assert!(soft_after.iter().all(|v| v[0..3] == [0.3, -0.8, -0.6]));
            continue;
        }
        gpu_soft.particle_qd.mark_device_dirty();
        gpu_rigid.joint_qd.mark_device_dirty();
        gpu_rigid.body_qd.mark_device_dirty();
        gpu_soft.particle_qd.download(device, queue).unwrap();
        gpu_rigid.joint_qd.download(device, queue).unwrap();
        gpu_rigid.body_qd.download(device, queue).unwrap();
        if friction == 0.4 && !locked {
            let result = (
                gpu_soft.particle_qd.host().unwrap().to_vec(),
                gpu_rigid.joint_qd.host().unwrap().to_vec(),
            );
            if let Some(reference) = &dense_reference {
                assert_eq!(
                    &result, reference,
                    "sparse candidate padding changed contact order/results"
                );
            } else {
                dense_reference = Some(result);
            }
        }
        for (gpu, cpu) in gpu_rigid
            .body_qd
            .host()
            .unwrap()
            .iter()
            .zip(cpu_rigid.body_qd.host().unwrap())
        {
            assert!((gpu.linear - cpu.linear).length() < 3e-5);
            assert!((gpu.angular - cpu.angular).length() < 3e-5);
        }
        for (gpu, cpu) in gpu_soft
            .particle_qd
            .host()
            .unwrap()
            .iter()
            .zip(cpu_soft.particle_qd.host().unwrap())
        {
            assert!(
                (*gpu - *cpu).length() < 3e-5,
                "friction={friction}: GPU={gpu:?}, CPU={cpu:?}"
            );
        }
        for (gpu, cpu) in gpu_rigid
            .joint_qd
            .host()
            .unwrap()
            .iter()
            .zip(cpu_rigid.joint_qd.host().unwrap())
        {
            assert!((gpu - cpu).abs() < 3e-5, "GPU={gpu}, CPU={cpu}");
        }
        let rows = read::<f32>(&engine, kernel.row_state(), candidate_count * 8);
        for (r, contact) in contacts.iter().enumerate() {
            let index = indices[r];
            let normal = rows[index * 8 + 4] as Real;
            let tangent = (rows[index * 8 + 5] as Real).hypot(rows[index * 8 + 6] as Real);
            assert!(tangent <= friction * normal + 1e-6);
            assert!((cpu_report.impulses[r].dot(contact.normal) - normal).abs() < 3e-5);
        }
        if sparse == 2 {
            for contact in &mut candidates {
                contact.ids[3] = 0;
            }
            queue.write_buffer(&contact_buffer, 0, bytemuck::cast_slice(&candidates));
            let empty = SoftRigidSolveKernel::new(device, rigid.joint_dof_total as u32, 0).unwrap();
            let before = read::<f32>(
                &engine,
                gpu_rigid.joint_qd.device_buffer().unwrap(),
                rigid.joint_dof_total,
            );
            for kernel in [&kernel, &empty] {
                let mut encoder = device.create_command_encoder(&Default::default());
                kernel
                    .encode(
                        device,
                        &mut encoder,
                        SoftRigidSolveBindings {
                            positions: gpu_soft.particle_q.device_buffer().unwrap(),
                            velocities: gpu_soft.particle_qd.device_buffer().unwrap(),
                            inverse_mass: soft.particle_inv_mass.device_buffer().unwrap(),
                            contacts: &contact_buffer,
                            body_poses: gpu_rigid.body_q.device_buffer().unwrap(),
                            body_com: rigid.body_com.device_buffer().unwrap(),
                            jacobians: &jacobians,
                            mass_matrix: &mass,
                            rigid_velocities: gpu_rigid.joint_qd.device_buffer().unwrap(),
                            body_velocities: gpu_rigid.body_qd.device_buffer().unwrap(),
                            particle_count: soft.particle_count as u32,
                            body_count: rigid.body_count as u32,
                        },
                        SoftRigidSolveSettings {
                            iterations: 60,
                            dt: 1.0 / 2400.0,
                            penetration_recovery: 0.2,
                            max_recovery_speed: 0.5,
                        },
                    )
                    .unwrap();
                queue.submit([encoder.finish()]);
                assert_eq!(read::<u32>(&engine, kernel.active_rows(), 1), [0]);
                assert_eq!(read::<u32>(&engine, kernel.status(), 1), [0]);
                assert_eq!(
                    read::<f32>(
                        &engine,
                        gpu_rigid.joint_qd.device_buffer().unwrap(),
                        rigid.joint_dof_total
                    ),
                    before
                );
            }
        }
        eprintln!(
            "GPU articulated two-way contact matches CPU at friction={friction}, locked={locked}"
        );
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_motor_to_tetrahedral_rubber_runs_without_substep_readback() {
    motor_to_rubber(false);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_motor_drives_reinforced_rubber_with_airborne_reaction() {
    motor_to_rubber(true);
}

fn motor_to_rubber(reinforced: bool) {
    use molla_kernels::soft_rigid_contact::SoftRigidShapeGpu;
    use molla_sim::{
        JointTargetMode,
        soft_belt::{RubberBeltMesh, RubberBeltParams, RubberCordParams},
    };
    use molla_solvers::{
        fem_rigid::FemRigidConfig,
        fem_rigid_gpu::{FemRigidGpuScene, FemRigidGpuSystem},
    };
    use std::sync::Arc;
    let engine = Arc::new(Engine::new_gpu().unwrap());
    for (sign, floating) in [(-1.0, false), (1.0, false), (-1.0, true), (1.0, true)] {
        if reinforced && !floating {
            continue;
        }
        let mut speeds = Vec::new();
        for coupled in [false, true] {
            let mesh = RubberBeltMesh::circular(RubberBeltParams {
                segments: 32,
                width_cells: 2,
                thickness_cells: 2,
                ..Default::default()
            })
            .unwrap();
            let mut builder = ModelBuilder::new();
            builder.set_world_gravity(WorldId(0), Vec3::ZERO);
            builder.set_linear_damping(0.0);
            if reinforced {
                mesh.append_reinforced_to(
                    &mut builder,
                    WorldId(0),
                    Vec3::ZERO,
                    RubberCordParams::default(),
                )
                .unwrap();
            } else {
                mesh.append_to(&mut builder, WorldId(0), Vec3::ZERO)
                    .unwrap();
            }
            let soft_model = builder.build().unwrap();
            let soft_masses = soft_model.particle_mass.host().unwrap().to_vec();
            let soft_state = soft_model.state().unwrap();
            let mut builder = ModelBuilder::new();
            builder.set_world_gravity(WorldId(0), Vec3::ZERO);
            builder.begin_articulation();
            let carrier = if floating {
                let body =
                    builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
                        mass: 750.0,
                        com: Vec3::ZERO,
                        inertia: Mat3::from_diagonal(Vec3::splat(120.0)),
                    }));
                builder
                    .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
                    .unwrap();
                body
            } else {
                BodyId::NONE
            };
            let rotor = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 2.0,
                com: Vec3::ZERO,
                inertia: Mat3::from_diagonal(Vec3::new(0.2, 0.25, 0.3)),
            }));
            builder
                .add_joint_revolute(
                    carrier,
                    rotor,
                    Transform::IDENTITY,
                    Transform::IDENTITY,
                    Vec3::X,
                )
                .unwrap();
            let mut rigid_model = builder.build().unwrap();
            let drive_dof = if floating { 6 } else { 0 };
            let rigid_masses = rigid_model.body_mass.host().unwrap().to_vec();
            let rigid_inertias = rigid_model.body_inertia.host().unwrap().to_vec();
            rigid_model.joint_effort_limit.host_mut().unwrap()[drive_dof] = 5.0;
            let rigid_state = rigid_model.state().unwrap();
            let mut control = rigid_model.control();
            control.joint_target_mode.host_mut().unwrap()[drive_dof] =
                JointTargetMode::Velocity as u32;
            control.joint_target_vel.host_mut().unwrap()[drive_dof] = sign * 2.0;
            control.joint_target_kd.host_mut().unwrap()[drive_dof] = 5.0;
            let shapes = if coupled {
                vec![SoftRigidShapeGpu {
                    position: [0.0, 0.48, (-sign * 0.016) as f32, 0.0],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    data: [0.11, 0.04, 0.015, 0.0],
                    ids: [1, rotor.index() as u32, 0, 0],
                }]
            } else {
                vec![]
            };
            let mut system = FemRigidGpuSystem::new(
                engine.clone(),
                FemRigidGpuScene {
                    soft_model,
                    soft_state,
                    rigid_model,
                    rigid_state,
                    control,
                    shapes,
                },
                FemRigidConfig::default(),
            )
            .unwrap();
            system.step(0.1).unwrap();
            assert!(system.soft_state().particle_q.host().is_err());
            assert!(system.soft_state().particle_qd.host().is_err());
            assert!(system.rigid_state().joint_qd.host().is_err());
            assert_eq!(read::<u32>(&engine, system.status_buffer(), 1), [0]);
            let q: Vec<Vec3> = read::<[f32; 4]>(
                &engine,
                system.soft_state().particle_q.device_buffer().unwrap(),
                mesh.positions.len(),
            )
            .iter()
            .map(|p| Vec3::new(p[0] as Real, p[1] as Real, p[2] as Real))
            .collect();
            let v: Vec<Vec3> = read::<[f32; 4]>(
                &engine,
                system.soft_state().particle_qd.device_buffer().unwrap(),
                mesh.positions.len(),
            )
            .iter()
            .map(|p| Vec3::new(p[0] as Real, p[1] as Real, p[2] as Real))
            .collect();
            let omega = read::<f32>(
                &engine,
                system.rigid_state().joint_qd.device_buffer().unwrap(),
                drive_dof + 1,
            )[drive_dof] as Real;
            let angular: Vec3 = q
                .iter()
                .zip(&v)
                .zip(&soft_masses)
                .map(|((p, v), m)| p.cross(*v * *m))
                .sum();
            let displacement = q
                .iter()
                .zip(&mesh.positions)
                .map(|(a, b)| (*a - *b).length())
                .fold(0.0, Real::max);
            assert!(mesh.minimum_volume_ratio(&q).unwrap() > 0.75);
            eprintln!(
                "resident GPU tooth: reinforced={reinforced}, sign={sign}, floating={floating}, coupled={coupled}, omega={omega}, rubber Lx={}, displacement={displacement}",
                angular.x
            );
            if floating {
                let poses = read::<[f32; 8]>(
                    &engine,
                    system.rigid_state().body_q.device_buffer().unwrap(),
                    rigid_masses.len(),
                );
                let velocities = read::<[f32; 8]>(
                    &engine,
                    system.rigid_state().body_qd.device_buffer().unwrap(),
                    rigid_masses.len(),
                );
                let mut momentum: Vec3 = v.iter().zip(&soft_masses).map(|(v, m)| *v * *m).sum();
                eprintln!("GPU airborne soft P={momentum:?}, L={angular:?}");
                let mut total_angular = angular;
                for i in 0..rigid_masses.len() {
                    let p = Vec3::new(
                        poses[i][0] as Real,
                        poses[i][1] as Real,
                        poses[i][2] as Real,
                    );
                    let rotation = Mat3::from_quat(molla_math::Quat::from_xyzw(
                        poses[i][4] as Real,
                        poses[i][5] as Real,
                        poses[i][6] as Real,
                        poses[i][7] as Real,
                    ));
                    let v = Vec3::new(
                        velocities[i][0] as Real,
                        velocities[i][1] as Real,
                        velocities[i][2] as Real,
                    );
                    let w = Vec3::new(
                        velocities[i][4] as Real,
                        velocities[i][5] as Real,
                        velocities[i][6] as Real,
                    );
                    eprintln!(
                        "GPU airborne rigid {i}: P={:?}, w={w:?}",
                        v * rigid_masses[i]
                    );
                    momentum += v * rigid_masses[i];
                    total_angular += p.cross(v * rigid_masses[i])
                        + rotation * rigid_inertias[i] * rotation.transpose() * w;
                }
                eprintln!("GPU airborne totals P={momentum:?}, L={total_angular:?}");
                assert!(
                    momentum.length() < 1e-3,
                    "airborne linear momentum {momentum:?}"
                );
                assert!(
                    total_angular.length() < 1e-3,
                    "airborne angular momentum {total_angular:?}"
                );
            }
            if coupled {
                assert!(sign * angular.x > 0.1);
                assert!(displacement > 0.001);
                if !floating {
                    assert!(
                        (omega.abs() - 0.27505).abs() < 0.002,
                        "CPU/GPU coupled speed disagreement"
                    );
                }
            } else if reinforced {
                assert!(
                    displacement < 1e-6,
                    "unstressed cord displacement {displacement}"
                );
                assert!(
                    v.iter().all(|v| v.length() < 1e-3),
                    "unstressed cord velocity"
                );
            } else {
                assert!(v.iter().all(|v| *v == Vec3::ZERO));
                for (actual, rest) in q.iter().zip(&mesh.positions) {
                    assert_eq!(
                        *actual,
                        Vec3::new(
                            rest.x as f32 as Real,
                            rest.y as f32 as Real,
                            rest.z as f32 as Real
                        )
                    );
                }
            }
            speeds.push(sign * omega);
        }
        assert!(
            speeds[1] < 0.9 * speeds[0],
            "rubber must physically load GPU motor: {speeds:?}"
        );
    }
}
