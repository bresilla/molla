#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_contact::SoftRigidContactGpu;
use molla_kernels::soft_rigid_solve::{
    ContactLinearizationBindings, ContactLinearizationKernel, SoftRigidSolveBindings,
    SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use molla_math::{Mat3, Quat, Vec3};
use std::path::Path;
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

fn read<T: bytemuck::Pod>(root: &Path, name: &str) -> Vec<T> {
    let bytes = std::fs::read(root.join(format!("contact-{name}.bin"))).unwrap();
    assert_eq!(bytes.len() % std::mem::size_of::<T>(), 0);
    bytes
        .chunks_exact(std::mem::size_of::<T>())
        .map(bytemuck::pod_read_unaligned)
        .collect()
}
fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, values: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}
fn xyz(a: [f32; 4]) -> Vec3 {
    Vec3::new(a[0] as f64, a[1] as f64, a[2] as f64)
}
fn axes(normal: Vec3) -> Mat3 {
    let tangent = normal
        .cross(if normal.x.abs() < 0.8 {
            Vec3::X
        } else {
            Vec3::Y
        })
        .normalize();
    Mat3::from_cols(normal, tangent, normal.cross(tangent))
}
fn node(c: &SoftRigidContactGpu, i: usize) -> usize {
    if i == 0 {
        c.ids[0] as usize
    } else {
        c.support[i - 1] as usize
    }
}
fn rotate(pose: [f32; 8], v: Vec3) -> Vec3 {
    Quat::from_xyzw(
        pose[4] as f64,
        pose[5] as f64,
        pose[6] as f64,
        pose[7] as f64,
    ) * v
}
fn position(pose: [f32; 8]) -> Vec3 {
    Vec3::new(pose[0] as f64, pose[1] as f64, pose[2] as f64)
}
fn factor(matrix: &[f32], n: usize) -> Vec<f64> {
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut value = matrix[i * n + j] as f64;
            for k in 0..j {
                value -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = if i == j {
                assert!(value > 0.0);
                value.sqrt()
            } else {
                value / l[j * n + j]
            };
        }
    }
    l
}
fn inverse_action(l: &[f64], rhs: &[f64]) -> Vec<f64> {
    let n = rhs.len();
    let mut value = rhs.to_vec();
    for i in 0..n {
        for j in 0..i {
            value[i] -= l[i * n + j] * value[j];
        }
        value[i] /= l[i * n + i];
    }
    for i in (0..n).rev() {
        for j in i + 1..n {
            value[i] -= l[j * n + i] * value[j];
        }
        value[i] /= l[i * n + i];
    }
    value
}
struct Oracle {
    contacts: Vec<SoftRigidContactGpu>,
    inverse_mass: Vec<f32>,
    soft: Vec<[f32; 4]>,
    rigid: Vec<f32>,
    soft_direction: Vec<[f32; 4]>,
    rigid_direction: Vec<f32>,
    impulses: Vec<[f32; 4]>,
    impulse_direction: Vec<[f32; 4]>,
    projected: Vec<Vec<Vec3>>,
    diagonal: Vec<Vec3>,
    desired: Vec<Vec3>,
    l: Vec<f64>,
}
impl Oracle {
    fn soft_sum_scale(&self, impulses: &[[f32; 4]]) -> Vec<Vec3> {
        let mut scale = vec![Vec3::ZERO; self.soft.len()];
        for (r, c) in self
            .contacts
            .iter()
            .enumerate()
            .filter(|(_, c)| c.ids[3] == 1)
        {
            let b = axes(xyz(c.normal_gap));
            let impulse = xyz(impulses[r]).abs();
            let world = b.x_axis.abs() * impulse.x
                + b.y_axis.abs() * impulse.y
                + b.z_axis.abs() * impulse.z;
            for i in 0..c.support[2] as usize {
                scale[node(c, i)] += c.weights[i] as f64 * world;
            }
        }
        for (value, &inverse) in scale.iter_mut().zip(&self.inverse_mass) {
            *value *= inverse as f64;
        }
        scale
    }
    fn evaluate(&self, epsilon: f64) -> (Vec<Vec3>, Vec<f64>, Vec<Vec3>) {
        let mut soft = vec![Vec3::ZERO; self.soft.len()];
        let mut reaction = vec![0.0; self.rigid.len()];
        let mut contact = Vec::new();
        for (r, c) in self.contacts.iter().enumerate() {
            let impulse = xyz(self.impulses[r]) + epsilon * xyz(self.impulse_direction[r]);
            let mut relative = Vec3::ZERO;
            if c.ids[3] == 1 {
                let basis = axes(xyz(c.normal_gap));
                for i in 0..c.support[2] as usize {
                    let p = node(c, i);
                    soft[p] += c.weights[i] as f64 * (basis * impulse);
                    relative += c.weights[i] as f64
                        * (basis.transpose()
                            * (xyz(self.soft[p]) + epsilon * xyz(self.soft_direction[p])));
                }
            }
            for (d, projected) in self.projected[r].iter().enumerate() {
                reaction[d] += projected.dot(impulse);
                relative -=
                    projected * (self.rigid[d] as f64 + epsilon * self.rigid_direction[d] as f64);
            }
            let diagonal = self.diagonal[r];
            let mut value = impulse;
            if c.ids[3] == 3 {
                for k in 0..3 {
                    if diagonal[k] > 0.0 {
                        value[k] = relative[k] - self.desired[r][k];
                    } else {
                        assert_eq!(relative[k], self.desired[r][k]);
                    }
                }
            } else {
                let normal_trial = impulse.x + (self.desired[r].x - relative.x) / diagonal.x;
                value.x = if normal_trial > 0.0 {
                    relative.x - self.desired[r].x
                } else {
                    impulse.x * diagonal.x
                };
                if c.ids[3] == 1 {
                    let trial = Vec3::new(
                        0.0,
                        impulse.y - relative.y / diagonal.y,
                        impulse.z - relative.z / diagonal.y,
                    );
                    let radius = c.material[0] as f64 * impulse.x.max(0.0);
                    let projection = trial * (radius / trial.length().max(1e-300)).min(1.0);
                    value.y = (impulse.y - projection.y) * diagonal.y;
                    value.z = (impulse.z - projection.z) * diagonal.y;
                }
            }
            contact.push(value);
        }
        for (value, &inverse) in soft.iter_mut().zip(&self.inverse_mass) {
            *value *= inverse as f64;
        }
        let mut rigid = inverse_action(&self.l, &reaction);
        for (value, &direction) in rigid.iter_mut().zip(&self.rigid_direction) {
            *value += epsilon * direction as f64;
        }
        (soft, rigid, contact)
    }
}

fn check(actual: f32, expected: f64, label: &str, index: usize) {
    check_scaled(actual, expected, 0.0, label, index);
}
fn check_scaled(actual: f32, expected: f64, input_scale: f64, label: &str, index: usize) {
    assert!(
        within_tolerance(actual, expected, input_scale),
        "{label}[{index}] GPU={actual} CPU={expected}"
    );
}

fn within_tolerance(actual: f32, expected: f64, input_scale: f64) -> bool {
    actual.is_finite()
        && (actual as f64 - expected).abs()
            < 3e-5 * (1.0 + expected.abs()) + 8.0 * f32::EPSILON as f64 * input_scale
}

#[test]
fn captured_oracle_bounds_reject_sign_and_magnitude_corruption() {
    for (expected, scale) in [(0.2, 1.0), (-0.1, 10.0), (100.0, 200.0), (0.00001, 2300.0)] {
        assert!(within_tolerance(expected as f32, expected, scale));
        let corrupted = (expected + 0.01 * (1.0 + scale + expected.abs())) as f32;
        assert!(!within_tolerance(corrupted, expected, scale));
        if expected.abs() > 0.01 {
            assert!(!within_tolerance(-expected as f32, expected, scale));
        }
        assert!(!within_tolerance(f32::NAN, expected, scale));
    }
}

#[test]
#[ignore = "requires GPU and MOLLA_CONTACT_LINEARIZATION_SOURCE captured CEOL buffers"]
fn captured_ceol_contact_operator_matches_independent_geometry_and_derivatives() {
    let root =
        std::path::PathBuf::from(std::env::var_os("MOLLA_CONTACT_LINEARIZATION_SOURCE").unwrap());
    let metadata = std::fs::read_to_string(root.join("contact-dimensions.json")).unwrap();
    let number = |name: &str| -> f32 {
        let prefix = format!("\"{name}\":");
        let matches: Vec<_> = metadata
            .lines()
            .filter_map(|line| line.trim().strip_prefix(&prefix))
            .collect();
        assert_eq!(matches.len(), 1);
        matches[0].trim().trim_end_matches(',').parse().unwrap()
    };
    let settings = SoftRigidSolveSettings {
        iterations: 1,
        dt: number("dt"),
        penetration_recovery: number("penetration_recovery"),
        max_recovery_speed: number("max_recovery_speed"),
    };
    let positions: Vec<[f32; 4]> = read(&root, "positions");
    let poses: Vec<[f32; 8]> = read(&root, "body_poses");
    let com: Vec<[f32; 4]> = read(&root, "body_com");
    let jacobians: Vec<f32> = read(&root, "jacobians");
    let mass: Vec<f32> = read(&root, "mass_matrix");
    let rigid: Vec<f32> = read(&root, "rigid_velocities");
    let contacts: Vec<SoftRigidContactGpu> = read::<SoftRigidContactGpu>(&root, "contacts")
        .into_iter()
        .filter(|c| c.ids[3] != 0)
        .collect();
    let n = positions.len();
    let d = rigid.len();
    assert_eq!(n, number("particles") as usize);
    assert_eq!(d, number("dofs") as usize);
    assert_eq!(poses.len(), number("bodies") as usize);
    assert_eq!(mass.len(), d * d);
    assert!(n > 10000 && contacts.len() > 100);
    let l = factor(&mass, d);
    let inverse_mass: Vec<f32> = read(&root, "inverse_mass");
    let mut oracle = Oracle {
        contacts,
        inverse_mass,
        soft: read(&root, "velocities"),
        rigid,
        soft_direction: (0..n)
            .map(|i| {
                [
                    ((i * 17 % 31) as f32 - 15.0) * 0.003,
                    ((i * 11 % 23) as f32 - 11.0) * 0.004,
                    ((i * 7 % 19) as f32 - 9.0) * 0.005,
                    0.0,
                ]
            })
            .collect(),
        rigid_direction: (0..d)
            .map(|i| ((i * 7 % 11) as f32 - 5.0) * 0.002)
            .collect(),
        impulses: vec![],
        impulse_direction: vec![],
        projected: vec![],
        diagonal: vec![],
        desired: vec![],
        l,
    };
    let point_jacobian = |body: usize, offset: Vec3, dof: usize| {
        let base = body * 6 * d + dof;
        Vec3::new(
            jacobians[base] as f64,
            jacobians[base + d] as f64,
            jacobians[base + 2 * d] as f64,
        ) + Vec3::new(
            jacobians[base + 3 * d] as f64,
            jacobians[base + 4 * d] as f64,
            jacobians[base + 5 * d] as f64,
        )
        .cross(offset)
    };
    for (r, c) in oracle.contacts.iter().enumerate() {
        assert!([1, 3, 5].contains(&c.ids[3]));
        let basis = if c.ids[3] == 3 {
            Mat3::IDENTITY
        } else {
            axes(xyz(c.normal_gap))
        };
        let mut point = Vec3::ZERO;
        let mut w = 0.0;
        for i in 0..c.support[2] as usize {
            point += c.weights[i] as f64 * xyz(positions[node(c, i)]);
            w += (c.weights[i] as f64).powi(2) * oracle.inverse_mass[node(c, i)] as f64;
        }
        let mut desired = Vec3::ZERO;
        if c.ids[3] == 3 {
            point = position(poses[c.ids[1] as usize])
                + rotate(poses[c.ids[1] as usize], xyz(c.weights));
            let anchor_a = position(poses[c.ids[0] as usize])
                + rotate(poses[c.ids[0] as usize], xyz(c.material));
            desired = ((point - anchor_a)
                * (settings.penetration_recovery as f64 / settings.dt as f64))
                .clamp(
                    Vec3::splat(-settings.max_recovery_speed as f64),
                    Vec3::splat(settings.max_recovery_speed as f64),
                );
        } else if c.ids[3] == 5 {
            desired.x = c.material[1] as f64;
        } else {
            desired.x = -(c.normal_gap[3] as f64) / settings.dt as f64;
            if c.normal_gap[3] < 0.0 {
                desired.x = (desired.x * settings.penetration_recovery as f64)
                    .min(settings.max_recovery_speed as f64);
            }
        }
        let mut projected = vec![Vec3::ZERO; d];
        for (dof, value) in projected.iter_mut().enumerate() {
            if c.ids[3] == 5 {
                if dof == c.ids[0] as usize {
                    value.x = -(c.material[2] as f64);
                }
            } else if c.ids[1] != u32::MAX {
                let b = c.ids[1] as usize;
                *value = basis.transpose()
                    * point_jacobian(
                        b,
                        point - position(poses[b]) - rotate(poses[b], xyz(com[b])),
                        dof,
                    );
                if c.ids[3] == 3 {
                    let a = c.ids[0] as usize;
                    *value -=
                        point_jacobian(a, rotate(poses[a], xyz(c.material) - xyz(com[a])), dof);
                }
            }
        }
        let response: Vec<_> = (0..3)
            .map(|k| {
                inverse_action(
                    &oracle.l,
                    &projected.iter().map(|v| v[k]).collect::<Vec<_>>(),
                )
            })
            .collect();
        let mut diagonal = Vec3::splat(w);
        for k in 0..3 {
            for dof in 0..d {
                diagonal[k] += projected[dof][k] * response[k][dof];
            }
        }
        if c.ids[3] == 1 {
            let off: f64 = (0..d).map(|dof| projected[dof].z * response[1][dof]).sum();
            diagonal.y = 0.5
                * (diagonal.y
                    + diagonal.z
                    + ((diagonal.y - diagonal.z).powi(2) + 4.0 * off * off).sqrt());
        }
        oracle.projected.push(projected);
        oracle.diagonal.push(diagonal);
        oracle.desired.push(desired);
        oracle.impulses.push([
            0.1 + (r % 7) as f32 * 0.01,
            ((r * 3 % 11) as f32 - 5.0) * 0.03,
            ((r * 5 % 13) as f32 - 6.0) * 0.02,
            0.0,
        ]);
        oracle
            .impulse_direction
            .push([((r * 7 % 11) as f32 - 5.0) * 0.0002, 0.001, -0.002, 0.0]);
    }
    let expected = oracle.evaluate(0.0);
    let epsilon = 1e-5;
    let plus = oracle.evaluate(epsilon);
    let minus = oracle.evaluate(-epsilon);
    let half_plus = oracle.evaluate(epsilon * 0.5);
    let half_minus = oracle.evaluate(-epsilon * 0.5);
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let buffers = [
        buffer(device, &positions),
        buffer(device, &oracle.soft),
        buffer(device, &oracle.inverse_mass),
        buffer(device, &oracle.contacts),
        buffer(device, &poses),
        buffer(device, &com),
        buffer(device, &jacobians),
        buffer(device, &mass),
        buffer(device, &oracle.rigid),
        buffer(device, &vec![[0.0_f32; 4]; poses.len() * 2]),
        buffer(device, &oracle.soft_direction),
        buffer(device, &oracle.rigid_direction),
        buffer(device, &oracle.impulses),
        buffer(device, &oracle.impulse_direction),
    ];
    let before: Vec<_> = buffers
        .iter()
        .map(|b| readback::read::<u8>(&engine, b, b.size() as usize))
        .collect();
    let bindings = ContactLinearizationBindings {
        state: SoftRigidSolveBindings {
            positions: &buffers[0],
            velocities: &buffers[1],
            inverse_mass: &buffers[2],
            contacts: &buffers[3],
            body_poses: &buffers[4],
            body_com: &buffers[5],
            jacobians: &buffers[6],
            mass_matrix: &buffers[7],
            rigid_velocities: &buffers[8],
            body_velocities: &buffers[9],
            particle_count: n as u32,
            body_count: poses.len() as u32,
        },
        rigid_reference: &buffers[8],
        soft_direction: &buffers[10],
        rigid_direction: &buffers[11],
        impulses: &buffers[12],
        impulse_direction: &buffers[13],
    };
    let rows = oracle.contacts.len();
    let solve = SoftRigidSolveKernel::new(device, d as u32, rows as u32).unwrap();
    let mut operator = ContactLinearizationKernel::new(
        device,
        n as u32,
        rows as u32,
        d as u32,
        poses.len() as u32,
    )
    .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    solve
        .encode_preparation(device, &mut encoder, bindings.state, settings)
        .unwrap();
    operator
        .prepare(device, &mut encoder, &solve, bindings, settings)
        .unwrap();
    operator.encode(&mut encoder).unwrap();
    queue.submit([encoder.finish()]);
    assert_eq!(readback::read::<u32>(&engine, operator.status(), 2), [0, 0]);
    let soft = readback::read::<[f32; 4]>(&engine, operator.soft_terms(), n * 2);
    let rigid = readback::read::<[f32; 4]>(&engine, operator.rigid_terms(), d);
    let contact = readback::read::<[f32; 4]>(&engine, operator.contact_terms(), rows * 2);
    let soft_scale = oracle.soft_sum_scale(&oracle.impulses);
    let soft_direction_scale = oracle.soft_sum_scale(&oracle.impulse_direction);
    for i in 0..n {
        for k in 0..3 {
            check_scaled(
                soft[i][k],
                expected.0[i][k],
                soft_scale[i][k],
                "soft",
                i * 3 + k,
            );
            check_scaled(
                soft[n + i][k],
                (plus.0[i][k] - minus.0[i][k]) / (2.0 * epsilon),
                soft_direction_scale[i][k],
                "soft action",
                i * 3 + k,
            );
        }
    }
    for dof in 0..d {
        check(rigid[dof][0], expected.1[dof], "rigid", dof);
        check(
            rigid[dof][1],
            (plus.1[dof] - minus.1[dof]) / (2.0 * epsilon),
            "rigid action",
            dof,
        );
    }
    for i in 0..rows {
        for k in 0..3 {
            let full = (plus.2[i][k] - minus.2[i][k]) / (2.0 * epsilon);
            let half = (half_plus.2[i][k] - half_minus.2[i][k]) / epsilon;
            assert!(
                (full - half).abs() < 1e-5 * (1.0 + full.abs()),
                "oracle FD crossed nonsmooth branch row={i} axis={k}"
            );
            check(contact[i][k], expected.2[i][k], "contact", i * 3 + k);
            check(contact[rows + i][k], full, "contact action", i * 3 + k);
        }
    }
    for (b, original) in buffers.iter().zip(before) {
        assert_eq!(
            readback::read::<u8>(&engine, b, b.size() as usize),
            original
        );
    }
    println!(
        "captured CEOL contact operator: {n} nodes, {rows} active geometry rows, {d} DOFs; independent f64 geometry, mass inverse, residuals and FD actions PASS"
    );
}
