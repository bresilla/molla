#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_contact::SoftRigidContactGpu;
use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod gpu_readback;

fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, values: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("static contact oracle fixture"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn fixed_ground_matches_barycentric_coulomb_oracle_without_rigid_reaction() {
    check_fixed_ground(false);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn cooperative_fixed_ground_matches_barycentric_coulomb_oracle_without_rigid_reaction() {
    check_fixed_ground(true);
}

fn check_fixed_ground(cooperative: bool) {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for n in [1, 29, 32, 33, 64] {
        for (friction, gap, invalid) in [
            (0.0_f32, 0.0_f32, false),
            (0.5, -0.0001, false),
            (10.0, 0.00001, false),
            (0.5, 0.0, true),
        ] {
            let position = buffer(device, &[[0.0_f32; 4]; 3]);
            let velocity = buffer(device, &[[3.0_f32, -2.0, 1.0, 0.0]; 3]);
            let inverse = buffer(device, &[2.0_f32; 3]);
            let poses = buffer(device, &[[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]);
            let com = buffer(device, &[[0.0_f32; 4]]);
            let mut matrix = vec![0.0_f32; n * n];
            for i in 0..n {
                matrix[i * n + i] = 1.0 + i as f32;
            }
            let mass = buffer(device, &matrix);
            let jacobian = buffer(device, &vec![0.125_f32; 6 * n]);
            let mut initial_rigid = (0..n).map(|i| i as f32 * 0.01 + 0.25).collect::<Vec<_>>();
            let rigid = buffer(device, &initial_rigid);
            let body_velocity = buffer(device, &[[0.0_f32; 8]]);
            let mut contact = SoftRigidContactGpu {
                normal_gap: [0.0, 1.0, 0.0, gap],
                material: [friction, 0.0, 0.0, 0.0],
                ids: [0, 0, 0, 1],
                support: [1, 2, 3, 0],
                weights: [0.25, 0.25, 0.5, 0.0],
            };
            let contacts = buffer(device, &[contact]);
            let constructor = if cooperative {
                SoftRigidSolveKernel::new_cooperative
            } else {
                SoftRigidSolveKernel::new
            };
            let kernel = constructor(device, n as u32, 1).unwrap();
            let bindings = || SoftRigidSolveBindings {
                positions: &position,
                velocities: &velocity,
                inverse_mass: &inverse,
                contacts: &contacts,
                body_poses: &poses,
                body_com: &com,
                jacobians: &jacobian,
                mass_matrix: &mass,
                rigid_velocities: &rigid,
                body_velocities: &body_velocity,
                particle_count: 3,
                body_count: 1,
            };
            let settings = SoftRigidSolveSettings {
                iterations: 12,
                dt: 1.0 / 19200.0,
                penetration_recovery: 0.2,
                max_recovery_speed: 0.5,
            };
            let mut encoder = device.create_command_encoder(&Default::default());
            kernel
                .encode_preparation(device, &mut encoder, bindings(), settings)
                .unwrap();
            queue.submit([encoder.finish()]);
            assert_eq!(gpu_readback::read::<u32>(&engine, kernel.status(), 1), [0]);
            let seeded = gpu_readback::read::<[f32; 8]>(&engine, kernel.row_state(), 1);
            assert!(seeded[0][0] > 0.75);
            if friction == 0.0 {
                let mut encoder = device.create_command_encoder(&Default::default());
                kernel
                    .encode(device, &mut encoder, bindings(), settings)
                    .unwrap();
                queue.submit([encoder.finish()]);
                assert_eq!(gpu_readback::read::<u32>(&engine, kernel.status(), 1), [0]);
                let effective = 0.75
                    + (0..n)
                        .map(|i| 0.125_f64.powi(2) / (i + 1) as f64)
                        .sum::<f64>();
                let relative = -2.0 - initial_rigid.iter().map(|v| 0.125 * *v as f64).sum::<f64>();
                let impulse = -relative / effective;
                let soft = gpu_readback::read::<[f32; 4]>(&engine, &velocity, 3);
                for (v, weight) in soft.iter().zip([0.25, 0.25, 0.5]) {
                    assert!(
                        (v[1] as f64 - (-2.0 + 2.0 * weight * impulse)).abs() < 3e-6,
                        "dynamic contact n={n}, cooperative={cooperative}, v={v:?}, impulse={impulse}"
                    );
                    assert_eq!([v[0], v[2], v[3]], [3.0, 1.0, 0.0]);
                }
                let rigid_after = gpu_readback::read::<f32>(&engine, &rigid, n);
                for (i, (&v, &before)) in rigid_after.iter().zip(&initial_rigid).enumerate() {
                    let expected = before as f64 - 0.125 * impulse / (i + 1) as f64;
                    assert!(
                        (v as f64 - expected).abs() < 3e-6,
                        "dynamic DOF{i}/{n}: {v} != {expected}"
                    );
                }
                let row = gpu_readback::read::<[f32; 8]>(&engine, kernel.row_state(), 1)[0];
                assert!((row[4] as f64 - impulse).abs() < 3e-6);
                queue.write_buffer(
                    &velocity,
                    0,
                    bytemuck::cast_slice(&[[3.0_f32, -2.0, 1.0, 0.0]; 3]),
                );
                queue.write_buffer(&rigid, 0, bytemuck::cast_slice(&initial_rigid));
            }
            contact.ids[1] = u32::MAX;
            queue.write_buffer(&contacts, 0, bytemuck::bytes_of(&contact));
            if invalid {
                initial_rigid[n - 1] = f32::NAN;
                queue.write_buffer(&rigid, 0, bytemuck::cast_slice(&initial_rigid));
            }
            let mut encoder = device.create_command_encoder(&Default::default());
            kernel
                .encode(device, &mut encoder, bindings(), settings)
                .unwrap();
            queue.submit([encoder.finish()]);
            assert_eq!(
                gpu_readback::read::<u32>(&engine, kernel.status(), 1),
                [if invalid { 4 } else { 0 }]
            );
            let after = gpu_readback::read::<f32>(&engine, &rigid, n);
            assert_eq!(
                after.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                initial_rigid
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            let soft = gpu_readback::read::<[f32; 4]>(&engine, &velocity, 3);
            if invalid {
                assert_eq!(soft, [[3.0, -2.0, 1.0, 0.0]; 3]);
                continue;
            }
            let w = 0.75_f64;
            let target = if gap < 0.0 {
                (-(gap as f64) / settings.dt as f64 * 0.2).min(0.5)
            } else {
                -(gap as f64) / settings.dt as f64
            };
            let normal = ((target + 2.0) / w).max(0.0);
            let tangent_scale = (friction as f64 * normal / 10.0_f64.sqrt()).min(1.0 / w);
            for (v, weight) in soft.iter().zip([0.25, 0.25, 0.5]) {
                let expected = [
                    3.0 - 6.0 * weight * tangent_scale,
                    -2.0 + 2.0 * weight * normal,
                    1.0 - 2.0 * weight * tangent_scale,
                ];
                for (actual, expected) in v[..3].iter().zip(expected) {
                    assert!(
                        (*actual as f64 - expected).abs() < 3e-6,
                        "dofs={n}, friction={friction}, actual={v:?}, expected={expected}"
                    );
                }
            }
            let row = gpu_readback::read::<[f32; 8]>(&engine, kernel.row_state(), 1)[0];
            assert_eq!(row[..2], [0.75, 0.75]);
            assert!((row[4] as f64 - normal).abs() < 3e-6);
        }
    }
}
