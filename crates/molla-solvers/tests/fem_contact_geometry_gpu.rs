#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::soft_rigid_contact::{
    SoftRigidContactBindings, SoftRigidContactGpu, SoftRigidContactKernel, SoftRigidSampleGpu,
    SoftRigidShapeGpu,
};
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod gpu_readback;

fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, values: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("contact geometry fixture"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    })
}

fn samples() -> [SoftRigidSampleGpu; 5] {
    [
        SoftRigidSampleGpu {
            nodes: [0, 0, 0, 1],
            weights: [1.0, 0.0, 0.0, 0.0],
        },
        SoftRigidSampleGpu {
            nodes: [1, 0, 0, 1],
            weights: [1.0, 0.0, 0.0, 0.0],
        },
        SoftRigidSampleGpu {
            nodes: [2, 0, 0, 1],
            weights: [1.0, 0.0, 0.0, 0.0],
        },
        SoftRigidSampleGpu {
            nodes: [0, 1, 0, 2],
            weights: [0.5, 0.5, 0.0, 0.0],
        },
        SoftRigidSampleGpu {
            nodes: [0, 1, 2, 3],
            weights: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0],
        },
    ]
}

fn gather(
    engine: &Engine,
    points: &[[f32; 4]; 3],
    radii: &[f32; 3],
    shape: SoftRigidShapeGpu,
    pose: [f32; 8],
    compact: bool,
) -> Vec<SoftRigidContactGpu> {
    let device = engine.device().unwrap();
    let positions = buffer(device, points);
    let radii = buffer(device, radii);
    let surface = buffer(device, &samples());
    let shapes = buffer(device, &[shape]);
    let poses = buffer(device, &[pose]);
    let kernel = if compact {
        SoftRigidContactKernel::new_compact_sampled_with_ball_joints(
            device,
            engine.queue().unwrap(),
            5,
            1,
            1,
            5,
            &[],
        )
        .unwrap()
    } else {
        SoftRigidContactKernel::new_sampled(device, 5, 1).unwrap()
    };
    let mut encoder = device.create_command_encoder(&Default::default());
    kernel
        .encode(
            device,
            &mut encoder,
            SoftRigidContactBindings {
                positions: &positions,
                radii: &radii,
                surface: &surface,
                shapes: &shapes,
                body_poses: &poses,
                particle_count: 3,
                body_count: 1,
            },
            2.0,
        )
        .unwrap();
    engine.queue().unwrap().submit([encoder.finish()]);
    gpu_readback::read(engine, kernel.rows(), 5)
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn box_face_nanometer_gaps_are_affine_consistent() {
    nanometer_face_gaps(1);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn prism_face_nanometer_gaps_are_affine_consistent() {
    nanometer_face_gaps(3);
}

fn nanometer_face_gaps(kind: u32) {
    let engine = Engine::new_gpu().unwrap();
    for axis in 0..3 {
        for sign in [-1.0_f32, 1.0] {
            for dynamic in [false, true] {
                for radii in [[0.0; 3], [0.0, 1.0e-8, 2.0e-8]] {
                    let mut center = [3.0, 7.0, -4.0, 0.0];
                    center[axis] = sign * -0.478_000_02_f32;
                    let face = center[axis] + sign * 0.5;
                    let mut points = [center; 3];
                    for (i, offset) in [0.0, 4.8428774e-8, -2.0489097e-8].into_iter().enumerate() {
                        points[i][axis] = face + sign * offset;
                    }
                    let mut extents = [5.0, 5.0, 5.0, 0.85];
                    extents[axis] = 0.5;
                    let mut shape = SoftRigidShapeGpu {
                        position: if dynamic { [0.0; 4] } else { center },
                        rotation: [0.0, 0.0, 0.0, 1.0],
                        data: extents,
                        ids: [kind, if dynamic { 0 } else { u32::MAX }, 0, 0],
                    };
                    if kind == 3 {
                        shape.position[3] = extents[1];
                    }
                    let pose = [center[0], center[1], center[2], 0.0, 0.0, 0.0, 0.0, 1.0];
                    let dense = gather(&engine, &points, &radii, shape, pose, false);
                    let compact = gather(&engine, &points, &radii, shape, pose, true);
                    assert_eq!(
                        bytemuck::cast_slice::<_, u8>(&dense),
                        bytemuck::cast_slice::<_, u8>(&compact)
                    );
                    for (row, sample) in dense.iter().zip(samples()) {
                        let expected: f64 = (0..sample.nodes[3] as usize)
                            .map(|i| {
                                let node = sample.nodes[i] as usize;
                                sample.weights[i] as f64
                                    * (sign as f64
                                        * (points[node][axis] as f64
                                            - (center[axis] as f64 + sign as f64 * 0.5))
                                        - radii[node] as f64)
                            })
                            .sum();
                        assert_eq!(row.ids[3], 1);
                        assert_eq!(row.normal_gap[axis], sign);
                        assert!(
                            (row.normal_gap[3] as f64 - expected).abs() < 1.0e-11,
                            "axis={axis} sign={sign} dynamic={dynamic} sample={sample:?}: gap={} expected={expected}",
                            row.normal_gap[3]
                        );
                        let vertex_gap: f64 = (0..sample.nodes[3] as usize)
                            .map(|i| {
                                sample.weights[i] as f64
                                    * dense[sample.nodes[i] as usize].normal_gap[3] as f64
                            })
                            .sum();
                        assert!((row.normal_gap[3] as f64 - vertex_gap).abs() < 1.0e-11);
                    }
                }
            }
        }
    }
}

fn rotate(q: [f64; 4], p: [f64; 3]) -> [f64; 3] {
    let cross = |a: [f64; 3], b: [f64; 3]| {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    };
    let xyz = [q[0], q[1], q[2]];
    let inner = cross(xyz, p);
    let outer = cross(xyz, std::array::from_fn(|i| inner[i] + q[3] * p[i]));
    std::array::from_fn(|i| p[i] + 2.0 * outer[i])
}

fn prism_gap(p: [f64; 3], size: [f64; 4]) -> f64 {
    let [axial, bottom, height, top] = size;
    let vertices = [
        [-bottom, -height],
        [bottom, -height],
        [top, height],
        [-top, height],
    ];
    let mut closest = f64::INFINITY;
    let mut inside = true;
    for i in 0..4 {
        let a = vertices[i];
        let b = vertices[(i + 1) % 4];
        let edge = [b[0] - a[0], b[1] - a[1]];
        let delta = [p[1] - a[0], p[2] - a[1]];
        inside &= edge[0] * delta[1] - edge[1] * delta[0] >= 0.0;
        let t = ((delta[0] * edge[0] + delta[1] * edge[1])
            / (edge[0] * edge[0] + edge[1] * edge[1]))
            .clamp(0.0, 1.0);
        closest = closest.min((delta[0] - t * edge[0]).hypot(delta[1] - t * edge[1]));
    }
    let planar = if inside { -closest } else { closest };
    let lateral = p[0].abs() - axial;
    planar.max(0.0).hypot(lateral.max(0.0)) + planar.max(lateral).min(0.0)
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn invalid_trapezoid_dimensions_emit_invalid_rows() {
    let engine = Engine::new_gpu().unwrap();
    for bad in [0.0, 1e-20, -0.001, f32::NAN, f32::INFINITY, 1e15] {
        for axis in 0..4 {
            let mut shape = SoftRigidShapeGpu {
                position: [0.0, 0.0, 0.0, 0.005],
                rotation: [0.0, 0.0, 0.0, 1.0],
                data: [0.025, 0.013, 0.013, 0.6],
                ids: [3, u32::MAX, 0, 0],
            };
            if axis == 3 {
                shape.position[3] = bad;
            } else {
                shape.data[axis] = bad;
            }
            assert!(shape.validate(0).is_err());
            for compact in [false, true] {
                let rows = gather(
                    &engine,
                    &[[0.0; 4]; 3],
                    &[0.0; 3],
                    shape,
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    compact,
                );
                assert!(rows.iter().all(|r| r.ids[3] == 2));
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn trapezoid_prism_flanks_caps_and_corners_match_f64() {
    let engine = Engine::new_gpu().unwrap();
    let size = [0.025_f32, 0.01262, 0.01290, 0.00545];
    assert!(prism_gap([0.0, 0.008, 0.0], size.map(f64::from)) < 0.0);
    for angle in [0.0_f32, 0.37, 1.2] {
        let q = [(angle / 2.0).sin(), 0.0, 0.0, (angle / 2.0).cos()];
        for dynamic in [false, true] {
            for point in [
                [0.0, 0.008, 0.0],
                [0.0, -0.008, 0.0],
                [0.024, 0.003, 0.001],
                [-0.027, -0.003, 0.001],
                [0.0, 0.002, 0.015],
                [0.0, 0.002, -0.015],
                [0.028, 0.015, 0.016],
                [-0.028, -0.015, -0.016],
            ] {
                let center = [0.125_f32, -0.25, 0.5];
                let points = std::array::from_fn(|i| {
                    let world = rotate(
                        q.map(f64::from),
                        [point[0] + i as f64 * 0.0001, point[1], point[2]],
                    );
                    [
                        world[0] as f32 + center[0],
                        world[1] as f32 + center[1],
                        world[2] as f32 + center[2],
                        0.0,
                    ]
                });
                let shape = SoftRigidShapeGpu {
                    position: if dynamic {
                        [0.0, 0.0, 0.0, size[3]]
                    } else {
                        [center[0], center[1], center[2], size[3]]
                    },
                    rotation: if dynamic { [0.0, 0.0, 0.0, 1.0] } else { q },
                    data: [size[0], size[1], size[2], 0.6],
                    ids: [3, if dynamic { 0 } else { u32::MAX }, 0, 0],
                };
                let pose = [center[0], center[1], center[2], 0.0, q[0], q[1], q[2], q[3]];
                let dense = gather(&engine, &points, &[0.0; 3], shape, pose, false);
                let compact = gather(&engine, &points, &[0.0; 3], shape, pose, true);
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&dense),
                    bytemuck::cast_slice::<_, u8>(&compact)
                );
                for (row, sample) in dense.iter().zip(samples()) {
                    assert_eq!(row.ids[3], 1, "trapezoid primitive rejected: {row:?}");
                    let world = std::array::from_fn(|axis| {
                        (0..sample.nodes[3] as usize)
                            .map(|i| {
                                sample.weights[i] as f64
                                    * points[sample.nodes[i] as usize][axis] as f64
                            })
                            .sum::<f64>()
                            - center[axis] as f64
                    });
                    let local = rotate(
                        [-q[0] as f64, -q[1] as f64, -q[2] as f64, q[3] as f64],
                        world,
                    );
                    let expected = prism_gap(local, size.map(f64::from));
                    assert!(
                        (row.normal_gap[3] as f64 - expected).abs() < 1e-7,
                        "gap {row:?} vs {expected}"
                    );
                    let gradient = std::array::from_fn(|axis| {
                        let mut a = local;
                        let mut b = local;
                        a[axis] += 1e-7;
                        b[axis] -= 1e-7;
                        (prism_gap(a, size.map(f64::from)) - prism_gap(b, size.map(f64::from)))
                            / 2e-7
                    });
                    let normal = rotate(q.map(f64::from), gradient);
                    for axis in 0..3 {
                        assert!(
                            (row.normal_gap[axis] as f64 - normal[axis]).abs() < 1e-4,
                            "normal {row:?} vs {normal:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn rotated_box_faces_edges_corners_match_f64_distance() {
    let engine = Engine::new_gpu().unwrap();
    for angle in [0.0_f32, 0.37, 1.2] {
        let rotation = [0.0, (angle / 2.0).sin(), 0.0, (angle / 2.0).cos()];
        let center = [0.125_f32, -0.25, 0.5, 0.0];
        for dynamic in [false, true] {
            for point in [
                [0.0, 0.0, 0.0],
                [0.51, 0.2, 0.1],
                [-0.51, 0.2, 0.1],
                [0.51, 0.72, 0.1],
                [-0.51, -0.72, -0.94],
                [0.1, -0.71, 0.0],
            ] {
                let points = std::array::from_fn(|i| {
                    let local = [point[0] + i as f64 * 0.001, point[1], point[2]];
                    let world = rotate(rotation.map(f64::from), local);
                    [
                        world[0] as f32 + center[0],
                        world[1] as f32 + center[1],
                        world[2] as f32 + center[2],
                        0.0,
                    ]
                });
                let shape = SoftRigidShapeGpu {
                    position: if dynamic { [0.0; 4] } else { center },
                    rotation: if dynamic {
                        [0.0, 0.0, 0.0, 1.0]
                    } else {
                        rotation
                    },
                    data: [0.5, 0.7, 0.9, 0.85],
                    ids: [1, if dynamic { 0 } else { u32::MAX }, 0, 0],
                };
                let pose = [
                    center[0],
                    center[1],
                    center[2],
                    0.0,
                    rotation[0],
                    rotation[1],
                    rotation[2],
                    rotation[3],
                ];
                let radii = [0.01, 0.02, 0.03];
                let rows = gather(&engine, &points, &radii, shape, pose, false);
                for (row, sample) in rows.iter().zip(samples()) {
                    let world: [f64; 3] = std::array::from_fn(|a| {
                        (0..sample.nodes[3] as usize)
                            .map(|i| {
                                sample.weights[i] as f64
                                    * (points[sample.nodes[i] as usize][a] as f64
                                        - center[a] as f64)
                            })
                            .sum()
                    });
                    let qrot = rotation.map(f64::from);
                    let local = rotate([-qrot[0], -qrot[1], -qrot[2], qrot[3]], world);
                    let q: [f64; 3] =
                        std::array::from_fn(|a| local[a].abs() - shape.data[a] as f64);
                    let r = q.iter().map(|v| v.max(0.0).powi(2)).sum::<f64>().sqrt();
                    let radius: f64 = (0..sample.nodes[3] as usize)
                        .map(|i| sample.weights[i] as f64 * radii[sample.nodes[i] as usize] as f64)
                        .sum();
                    let expected =
                        r + q.into_iter().fold(f64::NEG_INFINITY, f64::max).min(0.0) - radius;
                    let mut normal = [0.0; 3];
                    if r > 1e-12 {
                        normal = std::array::from_fn(|a| local[a].signum() * q[a].max(0.0) / r);
                    } else {
                        let axis = (0..3).max_by(|a, b| q[*a].total_cmp(&q[*b])).unwrap();
                        normal[axis] = if local[axis] >= 0.0 { 1.0 } else { -1.0 };
                    }
                    let normal = rotate(qrot, normal);
                    assert_eq!(row.ids[3], 1);
                    assert!(
                        (row.normal_gap[3] as f64 - expected).abs() < 4e-7,
                        "angle={angle} dynamic={dynamic} point={point:?}: {row:?} expected={expected}"
                    );
                    for a in 0..3 {
                        assert!((row.normal_gap[a] as f64 - normal[a]).abs() < 2e-5);
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn body_and_shape_transforms_preserve_box_distance() {
    let engine = Engine::new_gpu().unwrap();
    let body_q = [0.0_f32, 0.0, 0.31_f32.sin(), 0.31_f32.cos()];
    let local_q = [0.23_f32.sin(), 0.0, 0.0, 0.23_f32.cos()];
    let body_p = [1.0_f32, -2.0, 0.5];
    let shape = SoftRigidShapeGpu {
        position: [0.15, -0.2, 0.35, 0.0],
        rotation: local_q,
        data: [0.5, 0.7, 0.9, 0.85],
        ids: [1, 0, 0, 0],
    };
    let pose = [
        body_p[0], body_p[1], body_p[2], 0.0, body_q[0], body_q[1], body_q[2], body_q[3],
    ];
    let inverse = |q: [f32; 4]| [-q[0] as f64, -q[1] as f64, -q[2] as f64, q[3] as f64];
    for local in [
        [0.51, 0.2, 0.1],
        [-0.51, -0.72, 0.1],
        [0.51, 0.72, 0.94],
        [0.1, 0.69, 0.1],
    ] {
        let p = rotate(local_q.map(f64::from), local);
        let p = rotate(
            body_q.map(f64::from),
            std::array::from_fn(|a| p[a] + shape.position[a] as f64),
        );
        let p = [
            p[0] as f32 + body_p[0],
            p[1] as f32 + body_p[1],
            p[2] as f32 + body_p[2],
            0.0,
        ];
        let rows = gather(&engine, &[p; 3], &[0.0; 3], shape, pose, false);
        for (row, sample) in rows.iter().zip(samples()) {
            let body = rotate(
                inverse(body_q),
                std::array::from_fn(|a| p[a] as f64 - body_p[a] as f64),
            );
            let point = rotate(
                inverse(local_q),
                std::array::from_fn(|a| body[a] - shape.position[a] as f64),
            );
            let sum: f64 = sample.weights.iter().map(|v| *v as f64).sum();
            let q: [f64; 3] = std::array::from_fn(|a| point[a].abs() * sum - shape.data[a] as f64);
            let r = q.iter().map(|v| v.max(0.0).powi(2)).sum::<f64>().sqrt();
            let expected = r + q.into_iter().fold(f64::NEG_INFINITY, f64::max).min(0.0);
            assert!(
                (row.normal_gap[3] as f64 - expected).abs() < 5e-7,
                "local={local:?} row={row:?} expected={expected}"
            );
            let mut normal = [0.0; 3];
            if r > 1e-12 {
                normal = std::array::from_fn(|a| point[a].signum() * q[a].max(0.0) / r);
            } else {
                let a = (0..3).max_by(|a, b| q[*a].total_cmp(&q[*b])).unwrap();
                normal[a] = point[a].signum();
            }
            let normal = rotate(
                body_q.map(f64::from),
                rotate(local_q.map(f64::from), normal),
            );
            for a in 0..3 {
                assert!((row.normal_gap[a] as f64 - normal[a]).abs() < 2e-5);
            }
        }
    }
}
