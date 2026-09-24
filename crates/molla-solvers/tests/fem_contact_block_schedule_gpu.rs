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
        label: None,
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn contacts(body: u32, shape: u32, weight: f32) -> Vec<SoftRigidContactGpu> {
    let point = SoftRigidContactGpu {
        normal_gap: [0.6, 0.8, 0.0, -1e-5],
        material: [0.6, 0.0, 0.0, 0.0],
        ids: [0, body, shape, 1],
        support: [0, 0, 1, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    vec![
        point,
        SoftRigidContactGpu {
            ids: [1, body, shape, 1],
            normal_gap: [0.6, 0.8, 0.0, 1e-5],
            ..point
        },
        SoftRigidContactGpu {
            normal_gap: [0.6, 0.8, 0.0, weight * -1e-5 + (1.0 - weight) * 1e-5],
            support: [1, 0, 2, 0],
            weights: [weight, 1.0 - weight, 0.0, 0.0],
            ..point
        },
    ]
}

fn expected_groups(contacts: &[SoftRigidContactGpu], rows: &[[f32; 8]]) -> Vec<[u32; 2]> {
    contacts
        .iter()
        .enumerate()
        .map(|(r, edge)| {
            if edge.ids[3] != 1
                || edge.support[2] != 2
                || edge.weights[..2].iter().any(|w| *w <= 0.0)
            {
                return [u32::MAX; 2];
            }
            let points: Vec<_> = [edge.ids[0], edge.support[0]]
                .into_iter()
                .map(|node| {
                    contacts.iter().position(|c| {
                        c.ids == [node, edge.ids[1], edge.ids[2], 1]
                            && c.support[2] == 1
                            && c.weights[0] == 1.0
                    })
                })
                .collect();
            let (Some(a), Some(b)) = (points[0], points[1]) else {
                return [u32::MAX; 2];
            };
            if a == b
                || contacts[a].normal_gap[..3] != edge.normal_gap[..3]
                || contacts[b].normal_gap[..3] != edge.normal_gap[..3]
                || rows[r][2] <= edge.weights[0] * rows[a][2] + edge.weights[1] * rows[b][2] + 1e-6
            {
                return [u32::MAX; 2];
            }
            [a as u32, b as u32]
        })
        .collect()
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn gpu_contact_blocks_cover_grouping_fallbacks_and_invalid_inputs() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in [
        "dynamic",
        "static",
        "static_reordered",
        "static_mirrored",
        "static_edge_only",
        "reordered",
        "unequal",
        "collision",
        "two_bodies",
        "overlap",
        "missing",
        "normal",
        "body",
        "shape",
        "unmixed",
        "zero_weight",
        "zero_mass",
        "empty",
        "inactive",
        "invalid_node",
        "invalid_friction",
    ] {
        let mut cs = contacts(
            if mode.starts_with("static") || mode == "zero_mass" {
                u32::MAX
            } else {
                0
            },
            0,
            if mode == "unequal" {
                0.2
            } else if mode == "zero_weight" {
                1.0
            } else {
                0.5
            },
        );
        match mode {
            "static_mirrored" | "static_edge_only" => cs.iter_mut().for_each(|c| {
                c.normal_gap[3] = -c.normal_gap[3];
                c.material[0] = 0.0;
            }),
            "reordered" | "static_reordered" => cs.rotate_right(1),
            "collision" => {
                cs.extend(contacts(0, 16, 0.5));
                cs.resize(16, bytemuck::Zeroable::zeroed());
            }
            "overlap" => cs.extend(contacts(u32::MAX, 0, 0.5)),
            "two_bodies" => cs.extend(contacts(1, 0, 0.5)),
            "missing" => {
                cs.remove(1);
            }
            "normal" => cs[1].normal_gap[..3].copy_from_slice(&[0.0, 1.0, 0.0]),
            "body" => cs[1].ids[1] = 1,
            "shape" => cs[1].ids[2] = 1,
            "unmixed" => cs.iter_mut().for_each(|c| c.normal_gap[3] = -1e-5),
            "empty" => cs.clear(),
            "inactive" => cs.fill(bytemuck::Zeroable::zeroed()),
            "invalid_node" => cs[0].ids[0] = u32::MAX,
            "invalid_friction" => cs[0].material[0] = -0.1,
            _ => (),
        }
        let count = cs.len() as u32;
        let raw_contacts = if cs.is_empty() {
            vec![bytemuck::Zeroable::zeroed()]
        } else {
            cs.clone()
        };
        let positions = buffer(device, &[[0.0_f32, 0.0, 0.0, 0.0], [0.01, 0.02, 0.0, 0.0]]);
        let mut initial = [[0.3_f32, -1.0, 0.2, 0.0]; 2];
        if mode == "static_edge_only" {
            for (node, sign) in [-1.0_f32, 1.0].iter().enumerate() {
                initial[node][0] += sign * 0.04 * 0.6;
                initial[node][1] += sign * 0.04 * 0.8;
            }
        }
        let velocities = buffer(device, &initial);
        let inv = if mode == "zero_mass" {
            [0.0_f32; 2]
        } else if mode == "static_mirrored" || mode == "static_edge_only" {
            [10.0; 2]
        } else {
            [10.0, 20.0]
        };
        let inverse = buffer(device, &inv);
        let contact_buffer = buffer(device, &raw_contacts);
        let poses = buffer(device, &[[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]; 2]);
        let com = buffer(device, &[[0.0_f32; 4]; 2]);
        let j = [
            0.2_f32, 0.1, -0.1, 0.3, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ]
        .repeat(2);
        let jacobian = buffer(device, &j);
        let matrix = [2.0_f32, 0.1, 0.1, 3.0];
        let mass = buffer(device, &matrix);
        let initial_rigid = [0.1_f32, -0.2];
        let rigid = buffer(device, &initial_rigid);
        let body_velocity = buffer(device, &[[0.0_f32; 8]; 2]);
        let bindings = || SoftRigidSolveBindings {
            positions: &positions,
            velocities: &velocities,
            inverse_mass: &inverse,
            contacts: &contact_buffer,
            body_poses: &poses,
            body_com: &com,
            jacobians: &jacobian,
            mass_matrix: &mass,
            rigid_velocities: &rigid,
            body_velocities: &body_velocity,
            particle_count: 2,
            body_count: 2,
        };
        let settings = SoftRigidSolveSettings {
            iterations: if mode == "static_mirrored" || mode == "static_edge_only" {
                1
            } else {
                64
            },
            dt: 1e-4,
            penetration_recovery: 0.2,
            max_recovery_speed: 0.5,
        };
        let mut reference = None;
        for blocked in [false, true] {
            queue.write_buffer(&velocities, 0, bytemuck::cast_slice(&initial));
            queue.write_buffer(&rigid, 0, bytemuck::cast_slice(&initial_rigid));
            let kernel = if blocked {
                SoftRigidSolveKernel::new_blocked_contacts(device, 2, count, 2)
            } else {
                SoftRigidSolveKernel::new_separated_static(device, 2, count, 2)
            }
            .unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            kernel
                .encode(device, &mut encoder, bindings(), settings)
                .unwrap();
            if mode == "zero_mass"
                || (blocked && (mode == "static_mirrored" || mode == "static_edge_only"))
            {
                kernel
                    .encode_residual_check(device, &mut encoder, bindings(), settings, 1e-4)
                    .unwrap();
            }
            queue.submit([encoder.finish()]);
            let status = gpu_readback::read::<u32>(&engine, kernel.status(), 1)[0];
            let rows =
                gpu_readback::read::<[f32; 8]>(&engine, kernel.row_state(), count.max(1) as usize);
            let after = gpu_readback::read::<[f32; 4]>(&engine, &velocities, 2);
            let rigid_after = gpu_readback::read::<f32>(&engine, &rigid, 2);
            let expected = if mode.starts_with("invalid") {
                vec![[u32::MAX; 2]; cs.len()]
            } else {
                expected_groups(&cs, &rows)
            };
            assert_eq!(
                status,
                if mode.starts_with("invalid") {
                    2
                } else if mode == "zero_mass" {
                    64
                } else {
                    0
                },
                "{mode}"
            );
            assert!(
                rows.iter()
                    .flatten()
                    .chain(after.iter().flatten())
                    .chain(&rigid_after)
                    .all(|v| v.is_finite())
            );
            if blocked {
                if mode == "static_mirrored" || mode == "static_edge_only" {
                    assert_eq!(rows[0][4], 0.0, "first point was not released: {mode}");
                    assert!(rows[2][4] > 0.0, "edge was not active: {mode}");
                    assert_eq!(rows[1][4] > 0.0, mode == "static_mirrored");
                }
                let words = gpu_readback::read::<u32>(
                    &engine,
                    kernel.active_rows(),
                    kernel.active_rows().size() as usize / 4,
                );
                let base =
                    count as usize + 1 + count.div_ceil(64) as usize + 132 + 2 + 3 * count as usize;
                let actual: Vec<[u32; 2]> = words[base..base + 2 * count as usize]
                    .chunks_exact(2)
                    .map(|v| [v[0], v[1]])
                    .collect();
                assert_eq!(actual, expected, "GPU group mismatch in {mode}");
                eprintln!(
                    "contact block mode={mode} groups={} status={status}",
                    actual.iter().filter(|g| g[0] != u32::MAX).count()
                );
                if expected.iter().all(|g| g[0] == u32::MAX) {
                    assert_eq!(
                        reference.as_ref().unwrap(),
                        &(rows.clone(), after.clone(), rigid_after.clone()),
                        "fallback changed {mode}"
                    );
                }
            } else {
                reference = Some((rows.clone(), after.clone(), rigid_after.clone()));
            }
            if status != 0 {
                continue;
            }
            let mut impulse = [[0.0_f64; 3]; 2];
            let mut generalized = [0.0_f64; 2];
            for (contact, row) in cs.iter().zip(&rows) {
                if contact.ids[3] != 1 {
                    continue;
                }
                let n = molla_math::Vec3::new(
                    contact.normal_gap[0] as f64,
                    contact.normal_gap[1] as f64,
                    contact.normal_gap[2] as f64,
                );
                let axis = if n.x.abs() < 0.8 {
                    molla_math::Vec3::X
                } else {
                    molla_math::Vec3::Y
                };
                let t = n.cross(axis).normalize();
                let p = n * row[4] as f64 + t * row[5] as f64 + n.cross(t) * row[6] as f64;
                assert!(
                    row[4] >= 0.0
                        && (row[5] as f64).hypot(row[6] as f64)
                            <= contact.material[0] as f64 * row[4] as f64 + 1e-7
                );
                for s in 0..contact.support[2] as usize {
                    let node = if s == 0 {
                        contact.ids[0]
                    } else {
                        contact.support[s - 1]
                    } as usize;
                    for k in 0..3 {
                        impulse[node][k] += contact.weights[s] as f64 * p[k];
                    }
                }
                if contact.ids[1] != u32::MAX {
                    for d in 0..2 {
                        for k in 0..3 {
                            generalized[d] -= j[k * 2 + d] as f64 * p[k];
                        }
                    }
                }
            }
            for node in 0..2 {
                for k in 0..3 {
                    if inv[node] > 0.0 {
                        assert!(
                            ((after[node][k] - initial[node][k]) as f64 / inv[node] as f64
                                - impulse[node][k])
                                .abs()
                                < 2e-6,
                            "soft impulse {mode}"
                        );
                    }
                }
            }
            for d in 0..2 {
                let change: f64 = (0..2)
                    .map(|k| matrix[2 * d + k] as f64 * (rigid_after[k] - initial_rigid[k]) as f64)
                    .sum();
                assert!(
                    (change - generalized[d]).abs() < 2e-6,
                    "rigid impulse {mode}"
                );
            }
        }
    }
}
