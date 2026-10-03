#![cfg(feature = "gpu")]

use molla_compute::{Backend, Engine};
use molla_core::BodyId;
use molla_kernels::{ReactionWrenchBuffers, ReactionWrenchDispatch, ReactionWrenchKernel};
use molla_math::{GpuSpatialVector, Mat3, Quat, SpatialInertia, SpatialVector, Transform, Vec3};
use molla_sim::{BodyFlags, BodyParams, JointType, Model, ModelBuilder, State};
use molla_solvers::rnea::{
    inverse_dynamics_wrenches, inverse_dynamics_wrenches_with_boundary_accelerations,
};

fn fixture() -> (Model, State) {
    let mut builder = ModelBuilder::new();
    let params = || {
        BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 2.3,
            com: Vec3::new(0.2, -0.3, 0.1),
            inertia: Mat3::from_cols(
                Vec3::new(1.0, 0.1, 0.2),
                Vec3::new(0.1, 1.4, -0.1),
                Vec3::new(0.2, -0.1, 1.8),
            ),
        })
    };
    for root_kind in 0..3 {
        for reversed in [false, true] {
            for mode in 0..3 {
                for kind in [
                    JointType::Fixed,
                    JointType::Revolute,
                    JointType::Prismatic,
                    JointType::Ball,
                    JointType::Free,
                    JointType::D6,
                    JointType::Distance,
                    JointType::Cable,
                ] {
                    builder.begin_articulation();
                    let root_frame = Transform {
                        position: Vec3::new(16.0, -8.0, 4.0),
                        rotation: Quat::from_rotation_x(0.3),
                    };
                    let root = builder.add_body(BodyParams {
                        pose: root_frame,
                        flags: match root_kind {
                            1 => BodyFlags::KINEMATIC,
                            2 => BodyFlags::FIXED,
                            _ => 0,
                        },
                        ..params()
                    });
                    if root_kind == 0 {
                        builder
                            .add_joint_free(BodyId::NONE, root, root_frame, Transform::IDENTITY)
                            .unwrap();
                    } else {
                        builder
                            .add_joint_fixed(BodyId::NONE, root, root_frame, Transform::IDENTITY)
                            .unwrap();
                    }
                    let child = builder.add_body(params());
                    let xp = Transform {
                        position: Vec3::new(0.1, 0.4, -0.2),
                        rotation: Quat::from_rotation_y(-0.4),
                    };
                    let xc = Transform {
                        position: Vec3::new(-0.2, 0.1, 0.3),
                        rotation: Quat::from_rotation_z(0.5),
                    };
                    let axis = Vec3::new(1.0, 2.0, -1.0).normalize();
                    let joint = match kind {
                        JointType::Fixed => builder.add_joint_fixed(root, child, xp, xc),
                        JointType::Revolute => {
                            builder.add_joint_revolute(root, child, xp, xc, axis)
                        }
                        JointType::Prismatic => {
                            builder.add_joint_prismatic(root, child, xp, xc, axis)
                        }
                        JointType::Ball => builder.add_joint_ball(root, child, xp, xc),
                        JointType::Free => builder.add_joint_free(root, child, xp, xc),
                        JointType::D6 => builder.add_joint_d6(root, child, xp, xc),
                        JointType::Distance => builder.add_joint_distance(root, child, xp, xc),
                        JointType::Cable => {
                            builder.add_joint_cable(root, child, xp, xc, 0.0, 0.0, 0.0, 0.0)
                        }
                    }
                    .unwrap();
                    builder.set_joint_reversed(joint, reversed).unwrap();
                    builder.set_joint_enabled(joint, mode != 2).unwrap();
                    if mode == 1
                        && matches!(
                            kind,
                            JointType::Revolute
                                | JointType::Prismatic
                                | JointType::Ball
                                | JointType::D6
                        )
                    {
                        let width = match kind {
                            JointType::Ball => 3,
                            JointType::D6 => 6,
                            _ => 1,
                        };
                        let mut lo = vec![f64::NEG_INFINITY; width];
                        let mut hi = vec![f64::INFINITY; width];
                        lo[0] = 0.0;
                        hi[0] = 0.0;
                        builder.set_joint_limits(joint, &lo, &hi).unwrap();
                    }
                    let leaf = builder.add_body(params());
                    builder
                        .add_joint_fixed(
                            child,
                            leaf,
                            Transform::from_translation(Vec3::Y * 0.3),
                            Transform::IDENTITY,
                        )
                        .unwrap();
                }
            }
        }
    }
    let model = builder.build().unwrap();
    let mut state = model.state().unwrap();
    for j in 0..model.joint_count {
        let kind = model.joint_type.host().unwrap()[j];
        let c = model.joint_coord_offset.host().unwrap()[j] as usize;
        let d = model.joint_dof_offset.host().unwrap()[j] as usize;
        let end = model.joint_dof_offset.host().unwrap()[j + 1] as usize;
        if kind <= 1 {
            state.joint_q.host_mut().unwrap()[c] = 0.15;
        }
        let rotation = Quat::from_rotation_y(-0.2) * Quat::from_rotation_z(0.35);
        if kind == 2 {
            state.joint_q.host_mut().unwrap()[c..c + 4].copy_from_slice(&rotation.to_array());
        }
        if (4..=6).contains(&kind) {
            state.joint_q.host_mut().unwrap()[c..c + 3].copy_from_slice(&[0.2, -0.1, 0.3]);
            state.joint_q.host_mut().unwrap()[c + 3..c + 7].copy_from_slice(&rotation.to_array());
        }
        if kind == 7 {
            state.joint_q.host_mut().unwrap()[c..c + 2].copy_from_slice(&[0.1, -0.2]);
        }
        for dof in d..end {
            let locked = model.joint_limit_lower.host().unwrap()[dof]
                == model.joint_limit_upper.host().unwrap()[dof];
            state.joint_qd.host_mut().unwrap()[dof] = if locked {
                0.0
            } else {
                (dof % 7) as f64 * 0.13 - 0.3
            };
        }
    }
    molla_sim::eval_fk(&model, &mut state).unwrap();
    for body in 0..model.body_count {
        if model.body_flags.host().unwrap()[body] & BodyFlags::KINEMATIC != 0 {
            state.body_qd.host_mut().unwrap()[body] =
                SpatialVector::new(Vec3::new(0.4, -0.2, 0.3), Vec3::new(-0.2, 0.3, 0.5));
        }
    }
    molla_sim::eval_body_velocities(&model, &mut state).unwrap();
    (model, state)
}

fn read<T: bytemuck::Pod>(engine: &Engine, source: &wgpu::Buffer, count: usize) -> Vec<T> {
    let device = engine.device().unwrap();
    let size = (count * std::mem::size_of::<T>()) as u64;
    if count == 0 {
        return Vec::new();
    }
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("reaction test readback"),
        size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
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
    let data = staging.slice(..).get_mapped_range();
    let result = bytemuck::cast_slice(&data).to_vec();
    drop(data);
    staging.unmap();
    result
}

fn dispatch(
    engine: &Engine,
    prepared: &ReactionWrenchDispatch,
    buffers: &ReactionWrenchBuffers,
) -> u32 {
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    prepared.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    read::<u32>(engine, &buffers.status, 1)[0]
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn empty_and_maximum_fixed_trees_keep_wrenches_without_dofs() {
    let engine = Engine::new(Backend::AutoGpu).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for count in [0, 1, 64] {
        let mut builder = ModelBuilder::new();
        let mut bodies = Vec::new();
        for index in 0..count {
            let child = builder.add_body(BodyParams::new().with_mass(2.0));
            let parent = if index == 0 {
                BodyId::NONE
            } else {
                bodies[(index - 1) / 2]
            };
            builder
                .add_joint_fixed(
                    parent,
                    child,
                    Transform::from_translation(Vec3::X * 0.2 + Vec3::Y * 0.3),
                    Transform::IDENTITY,
                )
                .unwrap();
            bodies.push(child);
        }
        let mut model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        molla_sim::eval_fk(&model, &mut state).unwrap();
        let expected =
            inverse_dynamics_wrenches(&model, &state, 0, &[], &vec![SpatialVector::ZERO; count])
                .unwrap();
        model.upload_to_device(device, queue).unwrap();
        state.upload_to_device(device, queue).unwrap();
        let kernel = ReactionWrenchKernel::new(&engine, &model).unwrap();
        let mut buffers = kernel.create_buffers(&engine).unwrap();
        let prepared = kernel.prepare(&engine, &model, &state, &buffers).unwrap();
        assert_eq!(dispatch(&engine, &prepared, &buffers), 0);
        for (gpu, cpu) in read::<GpuSpatialVector>(&engine, &buffers.joints, count)
            .into_iter()
            .zip(expected.joints)
        {
            let gpu = SpatialVector::from(gpu);
            assert!((gpu.linear - cpu.linear).length() < 2e-4 * (1.0 + cpu.linear.length()));
            assert!((gpu.angular - cpu.angular).length() < 2e-4 * (1.0 + cpu.angular.length()));
        }
        buffers.status = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        assert!(
            kernel
                .prepare(&engine, &model, &state, &buffers)
                .err()
                .unwrap()
                .to_string()
                .contains("clear-buffer")
        );
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn general_gpu_wrenches_match_cpu_for_all_joint_kinds_and_boundaries() {
    check_general_gpu_wrenches(false);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn accelerating_boundaries_match_cpu_for_all_joint_kinds_and_reversals() {
    check_general_gpu_wrenches(true);
}

fn check_general_gpu_wrenches(accelerating: bool) {
    let engine = Engine::new(Backend::AutoGpu).unwrap();
    let (mut model, mut state) = fixture();
    let qdd: Vec<_> = (0..model.joint_dof_total)
        .map(|i| (i % 5) as f64 * 0.17 - 0.23)
        .collect();
    let external: Vec<_> = (0..model.body_count)
        .map(|i| {
            SpatialVector::new(
                Vec3::new((i % 3) as f64, -0.4, 0.7),
                Vec3::new(0.2, -0.3, 0.1),
            )
        })
        .collect();
    let boundary: Vec<_> = model
        .body_flags
        .host()
        .unwrap()
        .iter()
        .map(|&flags| {
            if accelerating && flags & BodyFlags::KINEMATIC != 0 {
                SpatialVector::new(Vec3::new(1.3, -0.7, 0.4), Vec3::new(-0.2, 0.8, 0.5))
            } else {
                SpatialVector::ZERO
            }
        })
        .collect();
    let mut expected = Vec::new();
    let mut efforts = Vec::new();
    for art in 0..model.articulation_count {
        let begin = model.articulation_start.host().unwrap()[art] as usize;
        let end = model.articulation_start.host().unwrap()[art + 1] as usize;
        let first = model.joint_dof_offset.host().unwrap()[begin] as usize;
        let last = model.joint_dof_offset.host().unwrap()[end] as usize;
        let result = inverse_dynamics_wrenches_with_boundary_accelerations(
            &model,
            &state,
            art,
            &qdd[first..last],
            &external,
            &boundary,
        )
        .unwrap();
        expected.extend(result.joints);
        efforts.extend(result.efforts);
    }
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let kernel = ReactionWrenchKernel::new(&engine, &model).unwrap();
    let buffers = kernel.create_buffers(&engine).unwrap();
    queue.write_buffer(
        &buffers.boundary_acceleration,
        0,
        bytemuck::cast_slice(
            &boundary
                .iter()
                .copied()
                .map(GpuSpatialVector::from)
                .collect::<Vec<_>>(),
        ),
    );
    queue.write_buffer(
        &buffers.acceleration,
        0,
        bytemuck::cast_slice(&qdd.iter().map(|&x| x as f32).collect::<Vec<_>>()),
    );
    queue.write_buffer(
        &buffers.external,
        0,
        bytemuck::cast_slice(
            &external
                .iter()
                .copied()
                .map(GpuSpatialVector::from)
                .collect::<Vec<_>>(),
        ),
    );
    let prepared = kernel.prepare(&engine, &model, &state, &buffers).unwrap();
    assert_eq!(dispatch(&engine, &prepared, &buffers), 0);
    let actual = read::<GpuSpatialVector>(&engine, &buffers.joints, model.joint_count);
    let mut peak = 0.0_f64;
    for (joint, (gpu, cpu)) in actual.into_iter().zip(expected).enumerate() {
        let gpu = SpatialVector::from(gpu);
        for (a, b) in [(gpu.linear, cpu.linear), (gpu.angular, cpu.angular)] {
            let relative = (a - b).length() / (1.0 + b.length());
            peak = peak.max(relative);
            assert!(
                relative < 3e-4,
                "joint={joint} gpu={gpu:?} cpu={cpu:?} scaled error={relative}"
            );
        }
    }
    for (dof, (gpu, cpu)) in read::<f32>(&engine, &buffers.efforts, model.joint_dof_total)
        .into_iter()
        .zip(efforts)
        .enumerate()
    {
        assert!(
            (gpu as f64 - cpu).abs() < 3e-4 * (1.0 + cpu.abs()),
            "dof={dof} gpu={gpu} cpu={cpu}"
        );
    }
    eprintln!(
        "general GPU reactions: accelerating={accelerating} arts={} joints={} dofs={} peak scaled wrench error={peak:e}",
        model.articulation_count, model.joint_count, model.joint_dof_total
    );
    queue.write_buffer(
        &buffers.active,
        0,
        bytemuck::cast_slice(&vec![0u32; model.articulation_count]),
    );
    assert_eq!(dispatch(&engine, &prepared, &buffers), 0);
    assert!(
        read::<f32>(&engine, &buffers.joints, model.joint_count * 8)
            .iter()
            .all(|&value| value == 0.0)
    );
    queue.write_buffer(
        &buffers.active,
        0,
        bytemuck::cast_slice(&vec![1u32; model.articulation_count]),
    );
    queue.write_buffer(&buffers.external, 0, bytemuck::cast_slice(&[f32::NAN; 8]));
    assert_eq!(dispatch(&engine, &prepared, &buffers), 1);
    queue.write_buffer(
        &buffers.external,
        0,
        bytemuck::bytes_of(&GpuSpatialVector::from(external[0])),
    );
    assert_eq!(dispatch(&engine, &prepared, &buffers), 0);
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "requires GPU resource counters through make test-gpu"]
fn prepared_reaction_dispatch_retains_resources() {
    let engine = Engine::new(Backend::AutoGpu).unwrap();
    let (mut model, mut state) = fixture();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let kernel = ReactionWrenchKernel::new(&engine, &model).unwrap();
    let buffers = kernel.create_buffers(&engine).unwrap();
    let prepared = kernel.prepare(&engine, &model, &state, &buffers).unwrap();
    let counters = || {
        device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        let c = device.get_internal_counters().hal;
        [
            c.buffers.read(),
            c.buffer_memory.read(),
            c.bind_groups.read(),
            c.compute_pipelines.read(),
            c.shader_modules.read(),
        ]
    };
    assert_eq!(dispatch(&engine, &prepared, &buffers), 0);
    let baseline = counters();
    for _ in 0..5 {
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        for _ in 0..8 {
            prepared.encode(&mut encoder);
        }
        queue.submit([encoder.finish()]);
        assert_eq!(counters(), baseline);
    }
    eprintln!("retained GPU reaction resources={baseline:?}");
}
