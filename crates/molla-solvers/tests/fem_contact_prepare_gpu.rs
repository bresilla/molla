#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_contact::SoftRigidContactGpu;
use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod gpu_readback;

fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, values: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("contact preparation fixture"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn quadratic(a: &[f32], b: &[f32], factor: &[f64], n: usize) -> f64 {
    let mut x = vec![0.0; n];
    for i in 0..n {
        x[i] = (b[i] as f64 - (0..i).map(|j| factor[i * n + j] * x[j]).sum::<f64>())
            / factor[i * n + i];
    }
    for i in (0..n).rev() {
        x[i] =
            (x[i] - (i + 1..n).map(|j| factor[j * n + i] * x[j]).sum::<f64>()) / factor[i * n + i];
    }
    a.iter().zip(x).map(|(a, x)| *a as f64 * x).sum()
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn native_scale_contact_preparation_has_bounded_completion_and_correct_response() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = molla_sim::ModelBuilder::new();
    builder.begin_articulation();
    let body = builder.add_body(molla_sim::BodyParams::new());
    builder
        .add_joint_revolute(
            molla_core::BodyId::NONE,
            body,
            molla_math::Transform::IDENTITY,
            molla_math::Transform::IDENTITY,
            molla_math::Vec3::X,
        )
        .unwrap();
    let _featherstone =
        molla_kernels::FeatherstoneKernel::new(device, queue, &builder.build().unwrap());
    let n = 29;
    let bodies = 22;
    let mut lower = vec![0.0; n * n];
    for i in 0..n {
        lower[i * n + i] = if i < 6 { 20.0 } else { 0.5 + i as f64 / 32.0 };
        for j in 0..i {
            lower[i * n + j] = ((i + j) % 5) as f64 / 64.0;
        }
    }
    let mut mass = vec![0.0_f32; n * n];
    for i in 0..n {
        for j in 0..n {
            mass[i * n + j] = (0..n)
                .map(|k| lower[i * n + k] * lower[j * n + k])
                .sum::<f64>() as f32;
        }
    }
    let mut jac = vec![0.0_f32; bodies * 6 * n];
    for body in 0..bodies {
        for axis in 0..3 {
            for d in 0..n {
                jac[(body * 6 + axis) * n + d] =
                    (((body + 1) * (axis + 2) + d * 3) % 13) as f32 / 32.0;
            }
        }
    }
    let position = buffer(device, &[[0.0_f32; 4]; 3]);
    let velocity = buffer(device, &[[0.0_f32; 4]; 3]);
    let inverse = buffer(device, &[2.0_f32; 3]);
    let poses = buffer(
        device,
        &vec![[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]; bodies],
    );
    let com = buffer(device, &vec![[0.0_f32; 4]; bodies]);
    let mass_buffer = buffer(device, &mass);
    let jac_buffer = buffer(device, &jac);
    let rigid_velocity = buffer(device, &vec![0.0_f32; n]);
    let body_velocity = buffer(device, &vec![[0.0_f32; 8]; bodies]);
    let contact = SoftRigidContactGpu {
        normal_gap: [0.0, 1.0, 0.0, -0.0001],
        material: [0.7, 0.0, 0.0, 0.0],
        ids: [0, 21, 0, 1],
        support: [1, 2, 3, 0],
        weights: [0.25, 0.25, 0.5, 0.0],
    };
    let ground = SoftRigidContactGpu {
        ids: [0, u32::MAX, 0, 1],
        ..contact
    };
    let ball = SoftRigidContactGpu {
        normal_gap: [0.0; 4],
        material: [0.0; 4],
        ids: [20, 21, 0, 3],
        support: [0; 4],
        weights: [0.0; 4],
    };
    let absent = SoftRigidContactGpu {
        ids: [0; 4],
        ..contact
    };
    let axis = |body: usize, axis: usize| &jac[(body * 6 + axis) * n..(body * 6 + axis + 1) * n];
    let x = axis(21, 0);
    let y = axis(21, 1);
    let z = axis(21, 2);
    let yy = quadratic(y, y, &lower, n);
    let xx = quadratic(x, x, &lower, n);
    let zz = quadratic(z, z, &lower, n);
    let xz = quadratic(x, z, &lower, n);
    let expected = [
        0.75 + yy,
        0.75 + 0.5 * (xx + zz + ((xx - zz).powi(2) + 4.0 * xz * xz).sqrt()),
        0.384,
    ];
    let ball_expected: Vec<_> = (0..3)
        .map(|a| {
            let j: Vec<_> = axis(21, a)
                .iter()
                .zip(axis(20, a))
                .map(|(b, a)| b - a)
                .collect();
            quadratic(&j, &j, &lower, n)
        })
        .collect();
    let native_rows = 128 * 1024 * 1024 / (3 * n * 4);
    for (count, dense) in [(65, true), (native_rows, false), (native_rows, true)] {
        let mut contacts = vec![absent; count];
        for r in 0..count {
            if dense || r < 3 || r >= count - 3 {
                contacts[r] = [contact, ground, ball][r % 3];
            }
        }
        let contacts_buffer = buffer(device, &contacts);
        let kernel = SoftRigidSolveKernel::new(device, n as u32, count as u32).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        kernel
            .encode_preparation(
                device,
                &mut encoder,
                SoftRigidSolveBindings {
                    positions: &position,
                    velocities: &velocity,
                    inverse_mass: &inverse,
                    contacts: &contacts_buffer,
                    body_poses: &poses,
                    body_com: &com,
                    jacobians: &jac_buffer,
                    mass_matrix: &mass_buffer,
                    rigid_velocities: &rigid_velocity,
                    body_velocities: &body_velocity,
                    particle_count: 3,
                    body_count: bodies as u32,
                },
                SoftRigidSolveSettings {
                    iterations: 1,
                    dt: 1.0 / 19200.0,
                    penetration_recovery: 0.2,
                    max_recovery_speed: 0.5,
                },
            )
            .unwrap();
        let start = Instant::now();
        eprintln!("contact prepare submit: dofs={n}, rows={count}, dense={dense}");
        let submission = queue.submit([encoder.finish()]);
        device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: Some(Duration::from_secs(30)),
            })
            .expect("contact preparation exceeded the bounded 30-second GPU wait");
        eprintln!("contact prepare complete: {:?}", start.elapsed());
        assert_eq!(gpu_readback::read::<u32>(&engine, kernel.status(), 1), [0]);
        let rows = gpu_readback::read::<[f32; 8]>(&engine, kernel.row_state(), count);
        for (r, row) in rows.iter().enumerate() {
            let target: &[f64] = match contacts[r].ids[3] {
                0 => &[0.0; 3],
                3 => &ball_expected,
                _ if contacts[r].ids[1] == u32::MAX => &[0.75, 0.75, 0.384],
                _ => &expected,
            };
            for (actual, expected) in row[..3].iter().zip(target) {
                assert!(
                    (*actual as f64 - expected).abs() < 2e-5,
                    "row={r}: {row:?} expected={target:?}"
                );
            }
            assert_eq!(&row[3..], &[0.0; 5]);
        }
        assert_eq!(
            gpu_readback::read::<f32>(&engine, &rigid_velocity, n),
            vec![0.0; n]
        );
        assert_eq!(
            gpu_readback::read::<[f32; 4]>(&engine, &velocity, 3),
            [[0.0; 4]; 3]
        );
    }
}
