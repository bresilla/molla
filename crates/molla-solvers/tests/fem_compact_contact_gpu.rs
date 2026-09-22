#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_contact::{
    SoftRigidBallJoint, SoftRigidContactBindings, SoftRigidContactGpu, SoftRigidContactKernel,
    SoftRigidSampleGpu, SoftRigidShapeGpu,
};
use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod gpu_readback;

#[path = "common/fem_rotor.rs"]
mod fem_rotor;

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn sampled_system_uses_bounded_contacts_and_reports_overflow() {
    use molla_math::Vec3;
    use molla_solvers::fem_rigid_gpu::FemRigidGpuSystem;
    use std::sync::Arc;

    let engine = Arc::new(Engine::new_gpu().unwrap());
    let scene = || {
        let mut scene = fem_rotor::scene();
        scene
            .soft_state
            .particle_qd
            .host_mut()
            .unwrap()
            .fill(-Vec3::Z * 0.1);
        scene.shapes = (0..13)
            .map(|i| SoftRigidShapeGpu {
                position: [i as f32 * 4.0, 0.0, -0.5, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                data: [1.0, 1.0, 0.5, 0.4],
                ids: [1, u32::MAX, 0, 0],
            })
            .collect();
        scene
    };
    let explicit = |capacity| {
        FemRigidGpuSystem::new_surface_sampled_with_contact_capacity_on_device(
            engine.device().unwrap().clone(),
            engine.queue().unwrap().clone(),
            scene(),
            fem_rotor::config(),
            &[],
            capacity,
        )
    };
    let mut bounded =
        FemRigidGpuSystem::new_surface_sampled(engine.clone(), scene(), fem_rotor::config())
            .unwrap();
    let mut full = explicit(14 * 13).unwrap();
    assert_eq!(bounded.contact_row_count(), 14 * 4);
    assert_eq!(full.contact_row_count(), 14 * 13);
    let positions = |system: &FemRigidGpuSystem| {
        gpu_readback::read::<[f32; 4]>(
            &engine,
            system.soft_state().particle_q.device_buffer().unwrap(),
            4,
        )
    };
    for _ in 0..8 {
        for system in [&mut bounded, &mut full] {
            system.step(fem_rotor::config().max_substep).unwrap();
            assert_eq!(
                gpu_readback::read::<u32>(&engine, system.status_buffer(), 1),
                [0]
            );
            assert!(system.soft_state().particle_q.host().is_err());
        }
        assert_eq!(positions(&bounded), positions(&full));
    }
    let velocities = gpu_readback::read::<[f32; 4]>(
        &engine,
        bounded.soft_state().particle_qd.device_buffer().unwrap(),
        4,
    );
    assert!(velocities.iter().take(3).all(|v| v[2] > -0.01));
    let mut overflow = explicit(1).unwrap();
    for _ in 0..2 {
        overflow.step(fem_rotor::config().max_substep).unwrap();
        assert_eq!(
            gpu_readback::read::<u32>(&engine, overflow.status_buffer(), 1),
            [16]
        );
    }
    assert!(explicit(0).is_err());
    let empty = FemRigidGpuSystem::new_surface_sampled(
        engine.clone(),
        fem_rotor::scene(),
        fem_rotor::config(),
    )
    .unwrap();
    assert_eq!(empty.contact_row_count(), 0);
}

fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("compact contact fixture"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

struct Fixture {
    positions: wgpu::Buffer,
    radii: wgpu::Buffer,
    surface: wgpu::Buffer,
    shapes: wgpu::Buffer,
    poses: wgpu::Buffer,
}

impl Fixture {
    fn new(engine: &Engine) -> Self {
        let device = engine.device().unwrap();
        let points = (0..130)
            .map(|i| {
                if [0, 63, 64, 65, 129].contains(&i) {
                    [if i == 64 { 8.0 } else { 0.0 }, 0.0, 0.095, 0.0]
                } else {
                    [100.0, 0.0, i as f32, 0.0]
                }
            })
            .collect::<Vec<_>>();
        let samples = (0..130)
            .map(|i| SoftRigidSampleGpu {
                nodes: [i, 0, 0, 1],
                weights: [1.0, 0.0, 0.0, 0.0],
            })
            .collect::<Vec<_>>();
        let shapes = (0..13)
            .map(|i| SoftRigidShapeGpu {
                position: [i as f32 * 4.0, 0.0, 0.0, 0.0],
                rotation: [0.0, 0.0, 0.0, 1.0],
                data: [0.1, 0.1, 0.1, 0.4],
                ids: [i % 3, if i % 2 == 0 { 0 } else { u32::MAX }, 0, 0],
            })
            .collect::<Vec<_>>();
        Self {
            positions: buffer(device, &points),
            radii: buffer(device, &[0.0_f32; 130]),
            surface: buffer(device, &samples),
            shapes: buffer(device, &shapes),
            poses: buffer(device, &[[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]; 2]),
        }
    }
    fn gather(&self, engine: &Engine, kernel: &SoftRigidContactKernel) -> Vec<SoftRigidContactGpu> {
        let device = engine.device().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        kernel
            .encode(
                device,
                &mut encoder,
                SoftRigidContactBindings {
                    positions: &self.positions,
                    radii: &self.radii,
                    surface: &self.surface,
                    shapes: &self.shapes,
                    body_poses: &self.poses,
                    particle_count: 130,
                    body_count: 2,
                },
                0.01,
            )
            .unwrap();
        engine.queue().unwrap().submit([encoder.finish()]);
        gpu_readback::read(engine, kernel.rows(), kernel.row_count() as usize)
    }
    fn solve(&self, engine: &Engine, gather: &SoftRigidContactKernel) -> (Vec<f32>, Vec<f32>, u32) {
        let device = engine.device().unwrap();
        let soft = buffer(device, &[[0.2_f32, 0.1, -0.5, 0.0]; 130]);
        let inv = buffer(device, &[1.0_f32; 130]);
        let com = buffer(device, &[[0.0_f32; 4]; 2]);
        let mut mass = [0.0_f32; 144];
        let mut jac = [0.0_f32; 144];
        for i in 0..12 {
            mass[i * 12 + i] = 2.0;
            jac[i * 12 + i] = 1.0;
        }
        let mass = buffer(device, &mass);
        let jac = buffer(device, &jac);
        let rigid = buffer(device, &[0.0_f32; 12]);
        let body_vel = buffer(device, &[[0.0_f32; 8]; 2]);
        let solver = SoftRigidSolveKernel::new(device, 12, gather.row_count()).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        solver
            .encode(
                device,
                &mut encoder,
                SoftRigidSolveBindings {
                    positions: &self.positions,
                    velocities: &soft,
                    inverse_mass: &inv,
                    contacts: gather.rows(),
                    body_poses: &self.poses,
                    body_com: &com,
                    jacobians: &jac,
                    mass_matrix: &mass,
                    rigid_velocities: &rigid,
                    body_velocities: &body_vel,
                    particle_count: 130,
                    body_count: 2,
                },
                SoftRigidSolveSettings {
                    iterations: 32,
                    dt: 0.001,
                    penetration_recovery: 0.2,
                    max_recovery_speed: 0.5,
                },
            )
            .unwrap();
        engine.queue().unwrap().submit([encoder.finish()]);
        (
            gpu_readback::read(engine, &soft, 130 * 4),
            gpu_readback::read(engine, &rigid, 12),
            gpu_readback::read::<u32>(engine, solver.status(), 1)[0],
        )
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn compact_contacts_match_dense_order_impulses_and_clear_stale_rows() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let fixture = Fixture::new(&engine);
    let joint = SoftRigidBallJoint {
        body_a: 0,
        body_b: 1,
        anchor_a: [0.0; 3],
        anchor_b: [0.0; 3],
    };
    let dense =
        SoftRigidContactKernel::new_sampled_with_ball_joints(device, queue, 130, 13, 2, &[joint])
            .unwrap();
    let rows = fixture.gather(&engine, &dense);
    let active = rows
        .iter()
        .copied()
        .filter(|r| r.ids[3] == 1)
        .collect::<Vec<_>>();
    assert_eq!(active.len(), 5);
    let capacity = active.len() as u32 + 67;
    let compact = SoftRigidContactKernel::new_compact_sampled_with_ball_joints(
        device,
        queue,
        130,
        13,
        2,
        capacity,
        &[joint],
    )
    .unwrap();
    let compact_rows = fixture.gather(&engine, &compact);
    assert_eq!(
        bytemuck::cast_slice::<_, u8>(&compact_rows[..active.len()]),
        bytemuck::cast_slice::<_, u8>(&active)
    );
    assert!(
        compact_rows[active.len()..capacity as usize]
            .iter()
            .all(|r| r.ids[3] == 0)
    );
    assert_eq!(compact_rows.last().unwrap().ids, [0, 1, 0, 3]);
    let a = fixture.solve(&engine, &dense);
    let b = fixture.solve(&engine, &compact);
    assert_eq!(a.2, 0);
    assert_eq!(b.2, 0);
    assert!(
        a.0.iter()
            .chain(&a.1)
            .zip(b.0.iter().chain(&b.1))
            .all(|(a, b)| (a - b).abs() < 1e-6)
    );
    assert!(b.1.iter().any(|v| v.abs() > 0.001));
    queue.write_buffer(
        &fixture.positions,
        0,
        bytemuck::cast_slice(&[[100.0_f32, 100.0, 100.0, 0.0]; 130]),
    );
    let empty = fixture.gather(&engine, &compact);
    assert!(empty[..capacity as usize].iter().all(|r| r.ids[3] == 0));
    assert_eq!(empty.last().unwrap().ids, [0, 1, 0, 3]);
    assert_eq!(compact.rows().size(), u64::from(capacity + 1) * 80);
    assert!(compact.rows().size() < dense.rows().size() / 10);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn compact_overflow_and_invalid_rows_stop_the_solver_without_partial_impulses() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let fixture = Fixture::new(&engine);
    let joint = SoftRigidBallJoint {
        body_a: 0,
        body_b: 1,
        anchor_a: [0.0; 3],
        anchor_b: [0.0; 3],
    };
    let overflow = SoftRigidContactKernel::new_compact_sampled_with_ball_joints(
        device,
        queue,
        130,
        13,
        2,
        4,
        &[joint],
    )
    .unwrap();
    let rows = fixture.gather(&engine, &overflow);
    assert_eq!(rows[0].ids[3], 4);
    assert!(rows[..4].iter().all(|r| r.ids[3] != 1));
    assert_eq!(rows[4].ids, [0, 1, 0, 3]);
    let (soft, rigid, status) = fixture.solve(&engine, &overflow);
    assert_eq!(status, 16);
    assert!(rigid.iter().all(|v| *v == 0.0));
    assert!(soft.chunks_exact(4).all(|v| v == [0.2, 0.1, -0.5, 0.0]));
    let invalid = SoftRigidContactKernel::new_compact_sampled_with_ball_joints(
        device,
        queue,
        130,
        13,
        2,
        64,
        &[joint],
    )
    .unwrap();
    queue.write_buffer(
        &fixture.surface,
        100 * 32,
        bytemuck::bytes_of(&SoftRigidSampleGpu {
            nodes: [u32::MAX, 0, 0, 1],
            weights: [1.0, 0.0, 0.0, 0.0],
        }),
    );
    let rows = fixture.gather(&engine, &invalid);
    assert_eq!(rows.iter().filter(|r| r.ids[3] == 2).count(), 13);
    let (soft, rigid, status) = fixture.solve(&engine, &invalid);
    assert_eq!(status, 2);
    assert!(rigid.iter().all(|v| *v == 0.0));
    assert!(soft.chunks_exact(4).all(|v| v == [0.2, 0.1, -0.5, 0.0]));
    assert!(
        SoftRigidContactKernel::new_compact_sampled_with_ball_joints(
            device,
            queue,
            130,
            13,
            2,
            0,
            &[]
        )
        .is_err()
    );
    assert!(
        SoftRigidContactKernel::new_compact_sampled_with_ball_joints(
            device,
            queue,
            u32::MAX,
            2,
            2,
            16,
            &[]
        )
        .is_err()
    );
}
