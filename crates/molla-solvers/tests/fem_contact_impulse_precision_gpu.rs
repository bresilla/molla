#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_contact::SoftRigidContactGpu;
use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("contact impulse precision fixture"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn many_small_contact_reactions_preserve_heavy_body_momentum() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let count = 2048u32;
    let mass_value = 10000.0_f32;
    for mode in ["serial", "cooperative", "separated", "blocked"] {
        for sign in [-1.0_f32, 1.0] {
            let mut uninterrupted = None;
            for split in [false, true] {
                let rigid_initial = sign * 0.03;
                let soft_initial = sign * 0.02999;
                let positions = buffer(device, &vec![[0.0_f32; 4]; count as usize]);
                let velocities =
                    buffer(device, &vec![[soft_initial, 0.0, 0.0, 0.0]; count as usize]);
                let inverse_mass = buffer(device, &vec![1.0_f32; count as usize]);
                let contacts: Vec<_> = (0..count)
                    .map(|node| SoftRigidContactGpu {
                        normal_gap: [sign, 0.0, 0.0, 0.0],
                        material: [0.0; 4],
                        ids: [node, 0, 0, 1],
                        support: [0, 0, 1, 0],
                        weights: [1.0, 0.0, 0.0, 0.0],
                    })
                    .collect();
                let contacts = buffer(device, &contacts);
                let poses = buffer(device, &[[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]);
                let com = buffer(device, &[[0.0_f32; 4]]);
                let jacobian = buffer(device, &[1.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0]);
                let mass = buffer(device, &[mass_value]);
                let rigid = buffer(device, &[rigid_initial]);
                let body_velocity = buffer(device, &[[0.0_f32; 8]]);
                let kernel = match mode {
                    "serial" => SoftRigidSolveKernel::new_serial(device, 1, count),
                    "cooperative" => SoftRigidSolveKernel::new_cooperative(device, 1, count),
                    "separated" => {
                        SoftRigidSolveKernel::new_separated_static(device, 1, count, count)
                    }
                    _ => SoftRigidSolveKernel::new_blocked_contacts(device, 1, count, count),
                }
                .unwrap();
                let bindings = SoftRigidSolveBindings {
                    positions: &positions,
                    velocities: &velocities,
                    inverse_mass: &inverse_mass,
                    contacts: &contacts,
                    body_poses: &poses,
                    body_com: &com,
                    jacobians: &jacobian,
                    mass_matrix: &mass,
                    rigid_velocities: &rigid,
                    body_velocities: &body_velocity,
                    particle_count: count,
                    body_count: 1,
                };
                let settings = SoftRigidSolveSettings {
                    iterations: 32,
                    dt: 0.001,
                    penetration_recovery: 0.2,
                    max_recovery_speed: 1.0,
                };
                let mut encoder = device.create_command_encoder(&Default::default());
                kernel
                    .encode(
                        device,
                        &mut encoder,
                        bindings,
                        SoftRigidSolveSettings {
                            iterations: if split { 8 } else { 32 },
                            ..settings
                        },
                    )
                    .unwrap();
                if split {
                    kernel
                        .encode_continuation(
                            device,
                            &mut encoder,
                            bindings,
                            SoftRigidSolveSettings {
                                iterations: 24,
                                ..settings
                            },
                        )
                        .unwrap();
                }
                kernel
                    .encode_residual_check(device, &mut encoder, bindings, settings, 1e-5)
                    .unwrap();
                queue.submit([encoder.finish()]);
                assert_eq!(readback::read::<u32>(&engine, kernel.status(), 1), [0]);
                let actual = f64::from(readback::read::<f32>(&engine, &rigid, 1)[0]);
                let soft = readback::read::<[f32; 4]>(&engine, &velocities, count as usize);
                let rows = readback::read::<[f32; 8]>(&engine, kernel.row_state(), count as usize);
                let state = (actual, soft.clone(), rows.clone());
                if let Some(expected) = &uninterrupted {
                    assert_eq!(&state, expected, "split sweep state mismatch in {mode}");
                } else {
                    uninterrupted = Some(state);
                }
                let impulse: f64 = rows.iter().map(|row| f64::from(sign * row[4])).sum();
                let rigid_delta = f64::from(mass_value) * (actual - f64::from(rigid_initial));
                let soft_delta: f64 = soft
                    .iter()
                    .map(|v| f64::from(v[0]) - f64::from(soft_initial))
                    .sum();
                let expected = (f64::from(mass_value) * f64::from(rigid_initial)
                    + f64::from(count) * f64::from(soft_initial))
                    / (f64::from(mass_value) + f64::from(count));
                eprintln!(
                    "mode={mode} sign={sign} split={split} rigid_error={} impulse_balance={} total_momentum={}",
                    actual - expected,
                    rigid_delta + impulse,
                    rigid_delta + soft_delta
                );
                assert!(
                    (rigid_delta + impulse).abs() <= 5e-4,
                    "lost generalized impulse"
                );
                assert!(
                    (rigid_delta + soft_delta).abs() <= 5e-4,
                    "lost total momentum"
                );
                assert!(
                    (actual - expected).abs() <= 5e-8,
                    "incorrect shared velocity"
                );
            }
        }
    }
}
