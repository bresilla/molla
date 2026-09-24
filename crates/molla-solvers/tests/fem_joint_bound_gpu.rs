#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_contact::{
    SoftRigidBallJoint, SoftRigidContactGpu, SoftRigidDofBound,
};
use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

#[path = "common/fem_rotor.rs"]
mod fem_rotor;

fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("coupled bound fixture"),
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn linear(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Option<Vec<f64>> {
    for k in 0..b.len() {
        let pivot = (k..b.len())
            .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
            .unwrap();
        if a[pivot][k].abs() < 1e-12 {
            return None;
        }
        a.swap(k, pivot);
        b.swap(k, pivot);
        let scale = a[k][k];
        for j in k..b.len() {
            a[k][j] /= scale;
        }
        b[k] /= scale;
        for i in 0..b.len() {
            if i == k {
                continue;
            }
            let scale = a[i][k];
            for j in k..b.len() {
                a[i][j] -= scale * a[k][j];
            }
            b[i] -= scale * b[k];
        }
    }
    Some(b)
}

fn dot(a: &[f64; 3], b: &[f64; 3]) -> f64 {
    a.iter().zip(b).map(|(a, b)| a * b).sum()
}

fn oracle(initial: [f64; 3], sign: f64, lower: f64) -> ([f64; 3], [f64; 4]) {
    let inverse = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0 / 1.91, -0.3 / 1.91],
        [0.0, -0.3 / 1.91, 2.0 / 1.91],
    ];
    let j = [
        [sign, 0.0, -sign],
        [0.0, 1.0, -1.0],
        [0.0, sign, 0.0],
        [0.0, -sign, 0.0],
    ];
    let target = [0.0, 0.0, lower, -1.0];
    let response: [[f64; 3]; 4] =
        std::array::from_fn(|r| std::array::from_fn(|d| dot(&inverse[d], &j[r])));
    for mask in 0..8 {
        let active: Vec<usize> = (0..4)
            .filter(|&r| r == 1 || mask & (1 << if r == 0 { 0 } else { r - 1 }) != 0)
            .collect();
        let gram = active
            .iter()
            .map(|&r| active.iter().map(|&c| dot(&j[r], &response[c])).collect())
            .collect();
        let rhs = active
            .iter()
            .map(|&r| target[r] - dot(&j[r], &initial))
            .collect();
        let Some(solution) = linear(gram, rhs) else {
            continue;
        };
        let mut impulse = [0.0; 4];
        for (&r, &value) in active.iter().zip(&solution) {
            impulse[r] = value;
        }
        let velocity = std::array::from_fn(|d| {
            initial[d] + (0..4).map(|r| response[r][d] * impulse[r]).sum::<f64>()
        });
        if (0..4).all(|r| {
            if r == 1 {
                (dot(&j[r], &velocity) - target[r]).abs() < 1e-9
            } else {
                impulse[r] >= -1e-9 && dot(&j[r], &velocity) >= target[r] - 1e-9
            }
        }) {
            return (velocity, impulse);
        }
    }
    panic!("no feasible active set");
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn joint_bounds_contacts_and_loops_match_independent_active_sets() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..4 {
        for sign in [-1.0_f32, 1.0] {
            for (lower, initial) in [
                (0.0, [-2.0, 1.0, -1.0]),
                (0.2, [-2.0, 1.0, -1.0]),
                (-0.2, [-2.0, 1.0, -1.0]),
                (0.0, [0.4, 0.3, 0.3]),
                (0.0, [3.0, 2.0, 2.0]),
            ] {
                let initial = initial.map(|v| v * sign);
                let contacts = [
                    SoftRigidContactGpu {
                        normal_gap: [sign, 0.0, 0.0, 0.0],
                        material: [0.0; 4],
                        ids: [0, 1, 0, 1],
                        support: [0, 0, 1, 0],
                        weights: [1.0, 0.0, 0.0, 0.0],
                    },
                    SoftRigidBallJoint {
                        body_a: 0,
                        body_b: 1,
                        anchor_a: [0.0; 3],
                        anchor_b: [0.0; 3],
                    }
                    .row(2)
                    .unwrap(),
                    SoftRigidDofBound {
                        dof: 0,
                        sign,
                        speed: lower,
                    }
                    .row(2)
                    .unwrap(),
                    SoftRigidDofBound {
                        dof: 0,
                        sign: -sign,
                        speed: -1.0,
                    }
                    .row(2)
                    .unwrap(),
                ];
                let positions = buffer(device, &[[0.0_f32; 4]]);
                let velocities = buffer(device, &[[initial[0], 0.0, 0.0, 0.0]]);
                let inverse_mass = buffer(device, &[1.0_f32]);
                let contacts = buffer(device, &contacts);
                let poses = buffer(device, &[[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]; 2]);
                let com = buffer(device, &[[0.0_f32; 4]; 2]);
                let mut jacobian = [0.0_f32; 24];
                jacobian[0] = 1.0;
                jacobian[13] = 1.0;
                let jacobians = buffer(device, &jacobian);
                let mass = buffer(device, &[2.0_f32, 0.3, 0.3, 1.0]);
                let rigid = buffer(device, &initial[1..]);
                let bodies = buffer(device, &[[0.0_f32; 8]; 2]);
                let kernel = match mode {
                    0 => SoftRigidSolveKernel::new_serial(device, 2, 4),
                    1 => SoftRigidSolveKernel::new_cooperative(device, 2, 4),
                    2 => SoftRigidSolveKernel::new_separated_static(device, 2, 4, 1),
                    _ => SoftRigidSolveKernel::new_blocked_contacts(device, 2, 4, 1),
                }
                .unwrap();
                let bindings = SoftRigidSolveBindings {
                    positions: &positions,
                    velocities: &velocities,
                    inverse_mass: &inverse_mass,
                    contacts: &contacts,
                    body_poses: &poses,
                    body_com: &com,
                    jacobians: &jacobians,
                    mass_matrix: &mass,
                    rigid_velocities: &rigid,
                    body_velocities: &bodies,
                    particle_count: 1,
                    body_count: 2,
                };
                let settings = SoftRigidSolveSettings {
                    iterations: 128,
                    dt: 0.01,
                    penetration_recovery: 0.2,
                    max_recovery_speed: 2.0,
                };
                let mut encoder = device.create_command_encoder(&Default::default());
                kernel
                    .encode(device, &mut encoder, bindings, settings)
                    .unwrap();
                kernel
                    .encode_residual_check(device, &mut encoder, bindings, settings, 1e-5)
                    .unwrap();
                queue.submit([encoder.finish()]);
                assert_eq!(
                    readback::read::<u32>(&engine, kernel.status(), 1),
                    [0],
                    "mode={mode}, sign={sign}, lower={lower}"
                );
                let soft = readback::read::<f32>(&engine, &velocities, 4);
                let rigid = readback::read::<f32>(&engine, &rigid, 2);
                let (expected, impulses) =
                    oracle(initial.map(f64::from), f64::from(sign), f64::from(lower));
                for (a, b) in [soft[0], rigid[0], rigid[1]].into_iter().zip(expected) {
                    assert!((f64::from(a) - b).abs() < 1e-5, "mode={mode}: {a} != {b}");
                }
                let rows = readback::read::<f32>(&engine, kernel.row_state(), 32);
                for (r, expected) in impulses.into_iter().enumerate() {
                    assert!(
                        (f64::from(rows[r * 8 + 4]) - expected).abs() < 2e-5,
                        "row{r}: {} != {expected}",
                        rows[r * 8 + 4]
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn coupled_runtime_enforces_both_position_and_speed_bounds_and_releases() {
    use molla_solvers::fem_rigid_gpu::FemRigidGpuSystem;
    let engine = std::sync::Arc::new(Engine::new_gpu().unwrap());
    for sign in [-1.0, 1.0] {
        let mut scene = fem_rotor::scene();
        scene.rigid_model.joint_limit_lower.host_mut().unwrap()[0] =
            if sign > 0.0 { -1.0 } else { 0.0 };
        scene.rigid_model.joint_limit_upper.host_mut().unwrap()[0] =
            if sign > 0.0 { 0.0 } else { 1.0 };
        scene.rigid_model.joint_velocity_limit.host_mut().unwrap()[0] = 0.2;
        scene.rigid_state.joint_q.host_mut().unwrap()[0] = -sign * 0.0001;
        scene.rigid_state.joint_qd.host_mut().unwrap()[0] = sign;
        let mut system =
            FemRigidGpuSystem::new(engine.clone(), scene, fem_rotor::config()).unwrap();
        system.configure_contact_snapshot(true).unwrap();
        let mut previous = -sign * 0.0001;
        for step in 0..8 {
            if step == 4 {
                system.set_drive_effort(0, -sign * 0.25).unwrap();
            }
            system.step(fem_rotor::config().max_substep).unwrap();
            assert_eq!(
                readback::read::<u32>(&engine, system.status_buffer(), 1),
                [0]
            );
            let q = readback::read::<f32>(
                &engine,
                system.rigid_state().joint_q.device_buffer().unwrap(),
                1,
            )[0];
            let v = readback::read::<f32>(
                &engine,
                system.rigid_state().joint_qd.device_buffer().unwrap(),
                1,
            )[0];
            assert!(sign * f64::from(q) <= 1e-8);
            assert!(v.abs() <= 0.200001);
            assert!(
                (f64::from(q) - previous - f64::from(v) * fem_rotor::config().max_substep).abs()
                    < 1e-9
            );
            if step == 3 {
                assert!(q.abs() < 1e-8 && v.abs() < 1e-7);
            }
            if step >= 4 {
                assert!(sign * f64::from(v) < 0.0);
            }
            let solved = system
                .contact_snapshot()
                .unwrap()
                .solved_buffers()
                .nth(1)
                .unwrap()
                .1;
            assert_eq!(readback::read::<f32>(&engine, solved, 1), [v]);
            previous = f64::from(q);
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn bound_targets_follow_device_coordinates_and_reject_nonfinite_state() {
    use molla_kernels::soft_rigid_bounds::{SoftRigidBoundsKernel, SoftRigidScalarBound};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let descriptor = SoftRigidScalarBound {
        indices: [0, 0, 0, 0],
        limits: [-0.2, 0.3, 4.0, 0.0],
    };
    let (kernel, rows) = SoftRigidBoundsKernel::new(device, &[descriptor], 1, 1).unwrap();
    let coordinates = buffer(device, &[0.0_f32]);
    let contacts = buffer(device, &rows);
    let status = buffer(device, &[0u32]);
    for (q, lo, neg_hi) in [(0.0_f32, -2.0, -3.0), (0.2, -4.0, -1.0), (-0.2, 0.0, -4.0)] {
        queue.write_buffer(&coordinates, 0, bytemuck::bytes_of(&q));
        let mut encoder = device.create_command_encoder(&Default::default());
        kernel
            .encode(
                device,
                &mut encoder,
                &coordinates,
                &contacts,
                &status,
                0,
                0.1,
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        let result = readback::read::<SoftRigidContactGpu>(&engine, &contacts, 2);
        assert!((result[0].material[1] - lo).abs() < 1e-6);
        assert!((result[1].material[1] - neg_hi).abs() < 1e-6);
        assert_eq!(readback::read::<u32>(&engine, &status, 1), [0]);
    }
    queue.write_buffer(&coordinates, 0, bytemuck::bytes_of(&f32::NAN));
    let mut encoder = device.create_command_encoder(&Default::default());
    kernel
        .encode(
            device,
            &mut encoder,
            &coordinates,
            &contacts,
            &status,
            0,
            0.1,
        )
        .unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(readback::read::<u32>(&engine, &status, 1), [32]);
    assert!(
        SoftRigidDofBound {
            dof: 1,
            sign: 1.0,
            speed: 0.0
        }
        .row(1)
        .is_err()
    );
    assert!(
        SoftRigidDofBound {
            dof: 0,
            sign: 0.0,
            speed: 0.0
        }
        .row(1)
        .is_err()
    );
    assert!(
        SoftRigidDofBound {
            dof: 0,
            sign: 1.0,
            speed: f32::NAN
        }
        .row(1)
        .is_err()
    );
}
