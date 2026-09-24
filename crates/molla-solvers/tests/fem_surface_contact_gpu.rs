#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_contact::{
    SoftRigidContactBindings, SoftRigidContactGpu, SoftRigidContactKernel, SoftRigidSampleGpu,
    SoftRigidShapeGpu,
};
use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use molla_math::Vec3;
use wgpu::util::DeviceExt;

fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, values: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("surface contact fixture"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn read<T: bytemuck::Pod>(engine: &Engine, source: &wgpu::Buffer, count: usize) -> Vec<T> {
    let device = engine.device().unwrap();
    let size = (std::mem::size_of::<T>() * count) as u64;
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("surface contact acceptance"),
        size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, size);
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec()
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn barycentric_contacts_transfer_equal_opposite_force_and_torque() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let points = [
        Vec3::new(-0.3, 0.0, 0.0),
        Vec3::new(0.7, 0.0, 0.0),
        Vec3::new(0.3, 0.8, 0.0),
    ];
    let masses = [1.0, 2.0, 3.0];
    let inertia = Vec3::new(2.0, 3.0, 4.0);
    let initial_velocity = Vec3::new(1.0, -2.0, -3.0);
    let positions = buffer(
        device,
        &points.map(|p| [p.x as f32, p.y as f32, p.z as f32, 0.0]),
    );
    let radii = buffer(device, &[0.0_f32; 3]);
    let vertex_indices = buffer(device, &[0_u32, 1, 2]);
    let inverse = buffer(device, &masses.map(|m| (1.0 / m) as f32));
    let pose = buffer(device, &[[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]);
    let com = buffer(device, &[[0.0_f32; 4]]);
    let mut jacobian = [0.0_f32; 36];
    let mut mass = [0.0_f32; 36];
    for i in 0..6 {
        jacobian[i * 6 + i] = 1.0;
        mass[i * 6 + i] = [5.0, 5.0, 5.0, 2.0, 3.0, 4.0][i];
    }
    let jacobian = buffer(device, &jacobian);
    let mass = buffer(device, &mass);
    for (count, weights) in [
        (1, [1.0_f32, 0.0, 0.0, 0.0]),
        (2, [0.5, 0.5, 0.0, 0.0]),
        (3, [0.2, 0.3, 0.5, 0.0]),
    ] {
        let point: Vec3 = points.iter().zip(weights).map(|(p, w)| *p * w as f64).sum();
        let surface = buffer(
            device,
            &[SoftRigidSampleGpu {
                nodes: [0, 1, 2, count],
                weights,
            }],
        );
        for kind in 0..4 {
            for friction in [0.0, 0.5] {
                let shape = SoftRigidShapeGpu {
                    position: [
                        point.x as f32,
                        point.y as f32,
                        -0.15,
                        if kind == 3 { 0.05 } else { 0.0 },
                    ],
                    rotation: [0.0, 0.0, 0.0, 1.0],
                    data: match kind {
                        0 => [0.15, 0.0, 0.0, friction],
                        1 => [0.1, 0.1, 0.15, friction],
                        2 => [0.15, 0.1, 0.0, friction],
                        _ => [0.1, 0.15, 0.15, friction],
                    },
                    ids: [kind, 0, 0, 0],
                };
                let shapes = buffer(device, &[shape]);
                let velocities = buffer(device, &[[1.0_f32, -2.0, -3.0, 0.0]; 3]);
                let rigid_velocity = buffer(device, &[0.0_f32; 6]);
                let body_velocity = buffer(device, &[[0.0_f32; 8]]);
                let gather = SoftRigidContactKernel::new_sampled(device, 1, 1).unwrap();
                let vertices = SoftRigidContactKernel::new(device, 3, 1).unwrap();
                let solve = SoftRigidSolveKernel::new(device, 6, 1).unwrap();
                let mut encoder = device.create_command_encoder(&Default::default());
                vertices
                    .encode(
                        device,
                        &mut encoder,
                        SoftRigidContactBindings {
                            positions: &positions,
                            radii: &radii,
                            surface: &vertex_indices,
                            shapes: &shapes,
                            body_poses: &pose,
                            particle_count: 3,
                            body_count: 1,
                        },
                        0.001,
                    )
                    .unwrap();
                gather
                    .encode(
                        device,
                        &mut encoder,
                        SoftRigidContactBindings {
                            positions: &positions,
                            radii: &radii,
                            surface: &surface,
                            shapes: &shapes,
                            body_poses: &pose,
                            particle_count: 3,
                            body_count: 1,
                        },
                        0.001,
                    )
                    .unwrap();
                solve
                    .encode(
                        device,
                        &mut encoder,
                        SoftRigidSolveBindings {
                            positions: &positions,
                            velocities: &velocities,
                            inverse_mass: &inverse,
                            contacts: gather.rows(),
                            body_poses: &pose,
                            body_com: &com,
                            jacobians: &jacobian,
                            mass_matrix: &mass,
                            rigid_velocities: &rigid_velocity,
                            body_velocities: &body_velocity,
                            particle_count: 3,
                            body_count: 1,
                        },
                        SoftRigidSolveSettings {
                            iterations: 60,
                            dt: 1.0 / 2400.0,
                            penetration_recovery: 0.2,
                            max_recovery_speed: 0.5,
                        },
                    )
                    .unwrap();
                engine.queue().unwrap().submit([encoder.finish()]);
                assert_eq!(read::<u32>(&engine, solve.status(), 1), [0]);
                if count > 1 {
                    assert!(
                        read::<SoftRigidContactGpu>(&engine, vertices.rows(), 3)
                            .iter()
                            .all(|c| c.ids[3] == 0)
                    );
                }
                let contact = read::<SoftRigidContactGpu>(&engine, gather.rows(), 1)[0];
                assert_eq!(contact.ids[3], 1);
                assert_eq!(contact.support[2], count);
                assert_eq!(contact.weights, weights);
                assert!(contact.normal_gap[0].abs() < 1e-5 && contact.normal_gap[1].abs() < 1e-5);
                assert!((contact.normal_gap[2] - 1.0).abs() < 1e-5);
                assert!(contact.normal_gap[3].abs() < 1e-6);
                let soft = read::<[f32; 4]>(&engine, &velocities, 3);
                let rigid = read::<f32>(&engine, &rigid_velocity, 6);
                let linear = Vec3::new(rigid[0] as f64, rigid[1] as f64, rigid[2] as f64);
                let angular = Vec3::new(rigid[3] as f64, rigid[4] as f64, rigid[5] as f64);
                let mut momentum_error = linear * 5.0;
                let mut angular_error = inertia * angular;
                let mut sample_velocity = Vec3::ZERO;
                let mut soft_delta = Vec3::ZERO;
                for i in 0..3 {
                    let v = Vec3::new(soft[i][0] as f64, soft[i][1] as f64, soft[i][2] as f64);
                    let delta = (v - initial_velocity) * masses[i];
                    momentum_error += delta;
                    angular_error += points[i].cross(delta);
                    soft_delta += delta;
                    sample_velocity += weights[i] as f64 * v;
                    if friction == 0.0 {
                        let effective: f64 = weights[..3]
                            .iter()
                            .zip(masses)
                            .map(|(w, m)| (*w as f64).powi(2) / m)
                            .sum();
                        let torque = point.cross(Vec3::Z);
                        let impulse = 3.0 / (effective + 1.0 / 5.0 + torque.dot(torque / inertia));
                        let expected =
                            initial_velocity + Vec3::Z * impulse * weights[i] as f64 / masses[i];
                        assert!(
                            (v - expected).length() < 2e-4,
                            "count={count}, kind={kind}, node={i}: {v:?} != {expected:?}"
                        );
                    }
                }
                assert!(momentum_error.length() < 1e-5, "{momentum_error:?}");
                assert!(angular_error.length() < 1e-5, "{angular_error:?}");
                let relative = sample_velocity - linear - angular.cross(point);
                assert!(relative.z.abs() < 2e-4, "{relative:?}");
                assert!(soft_delta.x.hypot(soft_delta.y) <= friction as f64 * soft_delta.z + 1e-5);
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn invalid_barycentric_samples_are_reported() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let positions = buffer(device, &[[0.0_f32; 4]; 3]);
    let radii = buffer(device, &[0.0_f32; 3]);
    let velocities = buffer(device, &[[1.0_f32, 2.0, 3.0, 0.0]; 3]);
    let inverse = buffer(device, &[1.0_f32; 3]);
    let com = buffer(device, &[[0.0_f32; 4]]);
    let mass = buffer(device, &[1.0_f32]);
    let jacobian = buffer(device, &[0.0_f32; 6]);
    let rigid_velocity = buffer(device, &[0.0_f32]);
    let body_velocity = buffer(device, &[[0.0_f32; 8]]);
    let pose = buffer(device, &[[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]);
    let shapes = buffer(
        device,
        &[SoftRigidShapeGpu {
            position: [0.0; 4],
            rotation: [0.0, 0.0, 0.0, 1.0],
            data: [1.0, 0.0, 0.0, 0.0],
            ids: [0, u32::MAX, 0, 0],
        }],
    );
    let valid = SoftRigidSampleGpu {
        nodes: [0, 1, 2, 3],
        weights: [0.2, 0.3, 0.5, 0.0],
    };
    for fault in 0..7 {
        let mut sample = valid;
        match fault {
            0 => sample.nodes[3] = 0,
            1 => sample.nodes[3] = 4,
            2 => sample.nodes[1] = 3,
            3 => sample.nodes[1] = 0,
            4 => sample.weights[1] = -0.1,
            5 => sample.weights[1] = 0.1,
            _ => sample.weights[1] = f32::NAN,
        }
        let surface = buffer(device, &[sample]);
        let gather = SoftRigidContactKernel::new_sampled(device, 1, 1).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        gather
            .encode(
                device,
                &mut encoder,
                SoftRigidContactBindings {
                    positions: &positions,
                    radii: &radii,
                    surface: &surface,
                    shapes: &shapes,
                    body_poses: &pose,
                    particle_count: 3,
                    body_count: 1,
                },
                0.001,
            )
            .unwrap();
        engine.queue().unwrap().submit([encoder.finish()]);
        let mut contact = read::<SoftRigidContactGpu>(&engine, gather.rows(), 1)[0];
        assert_eq!(contact.ids[3], 2, "fault={fault}");
        contact.ids[3] = 1;
        contact.normal_gap = [0.0, 0.0, 1.0, 0.0];
        let contacts = buffer(device, &[contact]);
        let solve = SoftRigidSolveKernel::new(device, 1, 1).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        solve
            .encode(
                device,
                &mut encoder,
                SoftRigidSolveBindings {
                    positions: &positions,
                    velocities: &velocities,
                    inverse_mass: &inverse,
                    contacts: &contacts,
                    body_poses: &pose,
                    body_com: &com,
                    jacobians: &jacobian,
                    mass_matrix: &mass,
                    rigid_velocities: &rigid_velocity,
                    body_velocities: &body_velocity,
                    particle_count: 3,
                    body_count: 1,
                },
                SoftRigidSolveSettings {
                    iterations: 12,
                    dt: 1.0 / 2400.0,
                    penetration_recovery: 0.2,
                    max_recovery_speed: 0.5,
                },
            )
            .unwrap();
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(
            read::<u32>(&engine, solve.status(), 1),
            [2],
            "fault={fault}"
        );
        assert_eq!(
            read::<[f32; 4]>(&engine, &velocities, 3),
            [[1.0, 2.0, 3.0, 0.0]; 3]
        );
    }
}
