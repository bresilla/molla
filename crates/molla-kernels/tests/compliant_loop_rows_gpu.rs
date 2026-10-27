#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    COMPLIANT_LOOP_ROW_FAULT, CompliantLoopBindings, CompliantLoopKernel, CompliantLoopRequest,
    CompliantLoopRow, GpuQuaternionPair,
};
use molla_math::{GpuSpatialVector, GpuTransform, Quat, SpatialVector, Transform, Vec3};
use molla_sim::joint_control::{JointAxis, JointMotor, JointSoftness, MotorModel};
use molla_sim::{JointTargetMode, JointType};

struct Case {
    poses: [Transform; 2],
    frames: [Transform; 2],
    centers: [Vec3; 2],
    velocities: [SpatialVector; 2],
    world: bool,
    axis: JointAxis,
    request: CompliantLoopRequest,
}

fn coordinate(case: &Case, poses: [Transform; 2]) -> f64 {
    let a = if case.world {
        case.frames[0]
    } else {
        poses[0].compose(&case.frames[0])
    };
    let b = poses[1].compose(&case.frames[1]);
    let axis = case.axis as usize;
    if axis < 3 {
        (a.rotation.conjugate() * (b.position - a.position))[axis]
    } else {
        let mut relative = a.rotation.conjugate() * b.rotation;
        if relative.w < 0.0 {
            relative = -relative;
        }
        relative.to_scaled_axis()[axis - 3]
    }
}

fn gradient(case: &Case, end: usize) -> [f64; 6] {
    if end == 0 && case.world {
        return [0.0; 6];
    }
    std::array::from_fn(|axis| {
        let epsilon = 1e-6;
        let mut sample = [0.0; 2];
        for (index, sign) in [-1.0, 1.0].into_iter().enumerate() {
            let mut poses = case.poses;
            if axis < 3 {
                poses[end].position[axis] += sign * epsilon;
            } else {
                let mut direction = Vec3::ZERO;
                direction[axis - 3] = sign * epsilon;
                let center = poses[end].position + poses[end].rotation * case.centers[end];
                poses[end].rotation = Quat::from_scaled_axis(direction) * poses[end].rotation;
                poses[end].position = center - poses[end].rotation * case.centers[end];
            }
            sample[index] = coordinate(case, poses);
        }
        (sample[1] - sample[0]) / (2.0 * epsilon)
    })
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for kind in [
        JointType::Fixed,
        JointType::Ball,
        JointType::Revolute,
        JointType::Prismatic,
        JointType::D6,
    ] {
        for world in [false, true] {
            for angle in [1e-7, 0.4, 2.8, std::f64::consts::PI - 0.002] {
                let alignment = if matches!(kind, JointType::Revolute | JointType::Prismatic) {
                    Quat::from_rotation_arc(Vec3::X, Vec3::new(2.0, -3.0, 4.0).normalize())
                } else {
                    Quat::IDENTITY
                };
                let parent_pose = Transform::new(
                    Vec3::new(100.00001, -3.0, 2.0),
                    Quat::from_rotation_z(0.7) * Quat::from_rotation_x(-0.4),
                );
                let mut parent_frame = Transform::new(
                    Vec3::new(0.2, -0.1, 0.3),
                    Quat::from_rotation_y(0.3) * alignment,
                );
                let child_frame = Transform::new(
                    Vec3::new(-0.3, 0.1, -0.2),
                    Quat::from_rotation_z(-0.2) * alignment,
                );
                let a = parent_pose.compose(&parent_frame);
                let child_pose = Transform::new(
                    Vec3::new(100.02001, -2.4, 1.7),
                    a.rotation
                        * Quat::from_axis_angle(Vec3::new(1.0, 2.0, -3.0).normalize(), angle)
                        * child_frame.rotation.conjugate(),
                );
                if world {
                    parent_frame = a;
                }
                for axis in JointAxis::ALL {
                    let frames = [parent_frame, child_frame];
                    let parent = if world { -1 } else { 0 };
                    let softness = JointSoftness {
                        natural_frequency: 7.0,
                        damping_ratio: 0.6,
                    };
                    let make_case = |request| Case {
                        poses: [parent_pose, child_pose],
                        frames,
                        centers: [Vec3::new(0.01, -0.03, 0.02), Vec3::new(-0.02, 0.04, 0.01)],
                        velocities: [
                            SpatialVector::new(
                                Vec3::new(0.2, -0.3, 0.4),
                                Vec3::new(-0.1, 0.2, 0.3),
                            ),
                            SpatialVector::new(
                                Vec3::new(-0.3, 0.4, 0.1),
                                Vec3::new(0.3, -0.2, 0.1),
                            ),
                        ],
                        world,
                        axis,
                        request,
                    };
                    let probe = make_case(
                        CompliantLoopRequest::limit(parent, 1, frames, axis, [0.0; 2], softness)
                            .unwrap(),
                    );
                    let position = coordinate(&probe, probe.poses);
                    for interval in [
                        [0.0, 0.0],
                        [-10.0, 10.0],
                        [position + 0.1, position + 0.2],
                        [position - 0.2, position - 0.1],
                    ] {
                        cases.push(make_case(
                            CompliantLoopRequest::limit(
                                parent, 1, frames, axis, interval, softness,
                            )
                            .unwrap(),
                        ));
                    }
                    for mode in [
                        JointTargetMode::None,
                        JointTargetMode::Position,
                        JointTargetMode::Velocity,
                        JointTargetMode::PositionVelocity,
                        JointTargetMode::Effort,
                    ] {
                        for model in [MotorModel::Force, MotorModel::Acceleration] {
                            cases.push(make_case(
                                CompliantLoopRequest::motor(
                                    parent,
                                    1,
                                    frames,
                                    axis,
                                    JointMotor {
                                        mode,
                                        model,
                                        target_position: -0.2,
                                        target_velocity: 0.3,
                                        stiffness: 12.0,
                                        damping: 3.0,
                                        max_force: 0.7,
                                    },
                                )
                                .unwrap(),
                            ));
                        }
                    }
                }
            }
        }
    }
    cases
}

#[test]
#[ignore = "requires GPU; run via make test-gpu"]
fn compliant_loop_rows_match_coordinate_derivatives_and_feedback_in_retained_batches() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let cases = cases();
    assert_eq!(cases.len(), 3360);
    let mut requests = Vec::new();
    let mut poses = Vec::<GpuTransform>::new();
    let mut velocities = Vec::<GpuSpatialVector>::new();
    let mut centers = Vec::<[f32; 4]>::new();
    let mut residues = Vec::<[f32; 4]>::new();
    for (index, case) in cases.iter().enumerate() {
        let mut request = case.request;
        request.metadata[0] = if case.world { -1 } else { 2 * index as i32 };
        request.metadata[1] = 2 * index as i32 + 1;
        requests.push(request);
        for end in 0..2 {
            let pose: GpuTransform = case.poses[end].into();
            let mut residue = [0.0; 4];
            for (axis, value) in residue.iter_mut().enumerate().take(3) {
                *value = (case.poses[end].position[axis] - pose.position[axis] as f64) as f32;
            }
            residues.push(residue);
            poses.push(pose);
            velocities.push(case.velocities[end].into());
            centers.push(case.centers[end].as_vec3().extend(0.0).to_array());
        }
    }
    let storage = |bytes: &[u8], size: u64| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("compliant loop row test"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if !bytes.is_empty() {
            queue.write_buffer(&buffer, 0, bytes);
        }
        buffer
    };
    let data = |bytes: &[u8]| storage(bytes, bytes.len() as u64);
    let requests_buffer = data(bytemuck::cast_slice(&requests));
    let poses_buffer = data(bytemuck::cast_slice(&poses));
    let rotations: Vec<_> = poses
        .iter()
        .map(|pose| GpuQuaternionPair {
            high: pose.rotation,
            low: [0.0; 4],
        })
        .collect();
    let rotations_buffer = data(bytemuck::cast_slice(&rotations));
    let velocities_buffer = data(bytemuck::cast_slice(&velocities));
    let centers_buffer = data(bytemuck::cast_slice(&centers));
    let residues_buffer = data(bytemuck::cast_slice(&residues));
    let row_bytes = cases.len() as u64 * 96;
    let rows_buffer = storage(&[], row_bytes);
    let status = storage(&0u32.to_ne_bytes(), 4);
    let rounding = storage(&[], cases.len() as u64 * 4);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("compliant loop row readback"),
        size: row_bytes + 4,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let kernel = CompliantLoopKernel::new(&engine).unwrap();
    let bind = || CompliantLoopBindings {
        requests: &requests_buffer,
        poses: &poses_buffer,
        velocities: &velocities_buffer,
        centers: &centers_buffer,
        position_residues: &residues_buffer,
        rows: &rows_buffer,
        status: &status,
        rounding: &rounding,
        rotations: &rotations_buffer,
    };
    assert!(kernel.prepare(device, bind(), 0).is_err());
    assert!(
        kernel
            .prepare(device, bind(), cases.len() as u32 + 1)
            .is_err()
    );
    let dispatch = kernel.prepare(device, bind(), cases.len() as u32).unwrap();
    let mut previous = Vec::new();
    for _ in 0..3 {
        let mut encoder = device.create_command_encoder(&Default::default());
        dispatch.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&rows_buffer, 0, &staging, 0, row_bytes);
        encoder.copy_buffer_to_buffer(&status, 0, &staging, row_bytes, 4);
        queue.submit([encoder.finish()]);
        let (tx, rx) = std::sync::mpsc::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                tx.send(result).unwrap();
            });
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        rx.recv().unwrap().unwrap();
        let mapped = staging.slice(..).get_mapped_range();
        assert_eq!(&mapped[row_bytes as usize..], &[0; 4]);
        if !previous.is_empty() {
            assert_eq!(previous, mapped[..row_bytes as usize]);
        }
        previous = mapped[..row_bytes as usize].to_vec();
        let rows = bytemuck::cast_slice::<u8, CompliantLoopRow>(&mapped[..row_bytes as usize]);
        for (index, (case, row)) in cases.iter().zip(rows).enumerate() {
            let position = coordinate(case, case.poses);
            let request = case.request;
            let limit = request.metadata[3] == 0;
            let lo = request.range[0] as f64;
            let hi = request.range[1] as f64;
            let effort = request.metadata[3] == 3;
            let active = if limit {
                lo == hi || position < lo || position > hi
            } else {
                effort || request.feedback[0] > 0.0 || request.feedback[1] > 0.0
            };
            assert_eq!(row.metadata[3] != 0, active, "case={index}");
            if effort {
                assert_eq!(row.metadata[3], 2);
            }
            if !active {
                assert_eq!(bytemuck::bytes_of(row), &[0; 96]);
                continue;
            }
            let error = if limit {
                position - position.clamp(lo, hi)
            } else if request.metadata[3] == 1 {
                position - request.range[2] as f64
            } else {
                0.0
            };
            assert!(
                (row.child_linear[3] as f64 - error).abs() < 2e-4,
                "case={index} error={error} row={row:?}"
            );
            if limit && lo == 0.0 && hi == 0.0 && case.axis as usize >= 3 && position.abs() < 0.001
            {
                let mut a: Transform = request.parent_frame.into();
                let mut b: Transform = request.child_frame.into();
                for (frame, residue) in [
                    (&mut a, request.parent_rotation_residue),
                    (&mut b, request.child_rotation_residue),
                ] {
                    frame.rotation = Quat::from_array(std::array::from_fn(|axis| {
                        frame.rotation.to_array()[axis] + residue[axis] as f64
                    }));
                }
                let parent: Transform = poses[2 * index].into();
                let child: Transform = poses[2 * index + 1].into();
                let a = if case.world { a } else { parent.compose(&a) };
                let b = child.compose(&b);
                let mut relative = (a.rotation.conjugate() * b.rotation).normalize();
                if relative.w < 0.0 {
                    relative = -relative;
                }
                let vector = Vec3::new(relative.x, relative.y, relative.z);
                let sine = vector.length();
                let log = if sine == 0.0 {
                    Vec3::ZERO
                } else {
                    vector * (2.0 * sine.atan2(relative.w) / sine)
                };
                let expected = log[case.axis as usize - 3];
                assert!(
                    (row.child_linear[3] as f64 - expected).abs() < 2e-8,
                    "packed small rotation case={index} expected={expected} row={row:?}"
                );
            }
            let child = gradient(case, 1);
            let parent = gradient(case, 0);
            let mut rate = 0.0;
            for (end, expected) in [parent, child].into_iter().enumerate() {
                let (linear, angular) = if end == 0 {
                    (row.parent_linear, row.parent_angular)
                } else {
                    (row.child_linear, row.child_angular)
                };
                for axis in 0..6 {
                    if end == 0 && case.world {
                        continue;
                    }
                    let actual = if axis < 3 {
                        linear[axis]
                    } else {
                        angular[axis - 3]
                    } as f64;
                    assert!(
                        (actual - expected[axis]).abs() < 3e-4 * (1.0 + expected[axis].abs()),
                        "case={index} end={end} axis={axis} actual={actual} expected={} row={row:?}",
                        expected[axis]
                    );
                }
                let velocity = case.velocities[end];
                for axis in 0..3 {
                    rate += expected[axis] * velocity.linear[axis]
                        + expected[axis + 3] * velocity.angular[axis];
                }
            }
            assert!(
                (row.child_angular[3] as f64 - rate).abs() < 3e-4 * (1.0 + rate.abs()),
                "case={index} rate={rate} row={row:?}"
            );
            assert_eq!(row.feedback, request.feedback);
            let expected_bounds = if effort {
                [request.range[2].clamp(-request.range[3], request.range[3]); 2]
            } else if limit && lo != hi {
                if position < lo {
                    [0.0, f32::INFINITY]
                } else {
                    [f32::NEG_INFINITY, 0.0]
                }
            } else {
                [-request.range[3], request.range[3]]
            };
            assert_eq!(
                [row.parent_linear[3], row.parent_angular[3]],
                expected_bounds
            );
        }
        drop(mapped);
        staging.unmap();
    }
    for request in &mut requests {
        request.metadata[2] = 6;
    }
    queue.write_buffer(&requests_buffer, 0, bytemuck::cast_slice(&requests));
    let mut encoder = device.create_command_encoder(&Default::default());
    dispatch.encode(&mut encoder);
    encoder.copy_buffer_to_buffer(&rows_buffer, 0, &staging, 0, row_bytes);
    encoder.copy_buffer_to_buffer(&status, 0, &staging, row_bytes, 4);
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let mapped = staging.slice(..).get_mapped_range();
    assert!(mapped[..row_bytes as usize].iter().all(|&byte| byte == 0));
    assert_eq!(
        &mapped[row_bytes as usize..],
        &COMPLIANT_LOOP_ROW_FAULT.to_ne_bytes()
    );
    drop(mapped);
    staging.unmap();
    engine.check_device().unwrap();
}
