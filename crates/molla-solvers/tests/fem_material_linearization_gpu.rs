#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_math::{Mat3, Quat, Vec3};
use molla_sim::{MaterialNeoHookean, Model, ModelBuilder, State, TetViscosity};
use molla_solvers::VbdSolver;
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

fn scalar(x: f64) -> f64 {
    x as f32 as f64
}
fn projected(x: Vec3) -> Vec3 {
    Vec3::from_array(x.to_array().map(scalar))
}
fn matrix(x: Mat3) -> Mat3 {
    Mat3::from_cols(
        projected(x.x_axis),
        projected(x.y_axis),
        projected(x.z_axis),
    )
}
fn edges(x: &[Vec3; 4]) -> Mat3 {
    Mat3::from_cols(x[1] - x[0], x[2] - x[0], x[3] - x[0])
}
fn cofactor(x: Mat3) -> Mat3 {
    Mat3::from_cols(
        x.y_axis.cross(x.z_axis),
        x.z_axis.cross(x.x_axis),
        x.x_axis.cross(x.y_axis),
    )
}
fn trace(x: Mat3) -> f64 {
    x.x_axis.x + x.y_axis.y + x.z_axis.z
}

fn model(viscous: bool, pinned: bool) -> Model {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.set_linear_damping(0.0);
    for (i, p) in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z]
        .into_iter()
        .enumerate()
    {
        builder.add_particle(
            p * 0.02,
            if pinned && i == 0 { 0.0 } else { 1.0 / 3000.0 },
            0.0,
            WorldId(0),
        );
    }
    let mut material = MaterialNeoHookean::elastic(2e6, 0.45);
    if viscous {
        material.viscosity = TetViscosity {
            shear: 1379.3103448275863,
            bulk: 13333.333333333336,
        };
    }
    builder
        .add_tet_mesh_per_material(&[[0, 1, 2, 3]], &[material], &[0], None)
        .unwrap();
    builder.add_tension_element(1, 2, 0.025, 2e5).unwrap();
    builder.add_distance_constraint(0, 3, 0.021, 1e-5, 1.0);
    builder.build().unwrap()
}

fn residual(
    model: &Model,
    positions: [Vec3; 4],
    target: [Vec3; 4],
    velocity: [Vec3; 4],
    dt: f64,
) -> [Vec3; 4] {
    let inverse_mass: Vec<_> = model
        .particle_inv_mass
        .host()
        .unwrap()
        .iter()
        .copied()
        .map(scalar)
        .collect();
    let rest: [Vec3; 4] =
        std::array::from_fn(|i| projected(model.initial_particle_q.host().unwrap()[i]));
    let previous_displacement = std::array::from_fn(|i| positions[i] - rest[i]);
    let delta = std::array::from_fn(|i| {
        if inverse_mass[i] > 0.0 {
            dt * velocity[i]
        } else {
            Vec3::ZERO
        }
    });
    let inverse = matrix(model.tet_dm_inv.host().unwrap()[0]);
    let previous = Mat3::IDENTITY + edges(&previous_displacement) * inverse;
    let increment = edges(&delta) * inverse;
    let f = previous + increment;
    let cof = cofactor(f);
    let mu = scalar(model.tet_mu.host().unwrap()[0]);
    let lambda = scalar(model.tet_lambda.host().unwrap()[0]);
    let mut stress = mu * (f - cof) + (lambda + mu) * (f.determinant() - 1.0) * cof;
    let viscosity = model.tet_viscosity.host().unwrap()[0];
    let strain = (previous.transpose() * increment
        + increment.transpose() * previous
        + increment.transpose() * increment)
        * 0.5;
    let sv = 2.0 * scalar(viscosity.shear) * (strain - Mat3::IDENTITY * (trace(strain) / 3.0))
        + scalar(viscosity.bulk) * trace(strain) * Mat3::IDENTITY;
    stress += f * sv / dt;
    let volume = scalar(model.tet_rest_volume.host().unwrap()[0]);
    let mut force: [Vec3; 4] = std::array::from_fn(|i| {
        let gradient = if i == 0 {
            -inverse.transpose() * Vec3::ONE
        } else {
            inverse.transpose().col(i - 1)
        };
        -volume * stress * gradient
    });
    for spring in 0..model.spring_count {
        let a = model.spring_a.host().unwrap()[spring] as usize;
        let b = model.spring_b.host().unwrap()[spring] as usize;
        let diff = positions[a] - positions[b] + delta[a] - delta[b];
        let size = diff.length().max(1e-8);
        let extension = size - scalar(model.spring_rest_length.host().unwrap()[spring]);
        let tension = model.spring_tension_only.host().unwrap()[spring] != 0;
        if tension && extension <= 0.0 {
            continue;
        }
        let compliance = model.spring_compliance.host().unwrap()[spring];
        let stiffness = if tension {
            1.0 / compliance.max(1e-12)
        } else {
            (1.0 / compliance.max(1e-12)).clamp(1.0, 1e7)
        };
        let applied = -scalar(stiffness) * (extension / size) * diff;
        force[a] += applied;
        force[b] -= applied;
    }
    std::array::from_fn(|i| {
        if inverse_mass[i] == 0.0 {
            velocity[i]
        } else {
            velocity[i] - target[i] - dt * inverse_mass[i] * force[i]
        }
    })
}

fn snapshot(engine: &Engine, states: &[&State]) -> Vec<Vec<u8>> {
    states
        .iter()
        .flat_map(|state| {
            [
                state.particle_q.device_buffer(),
                state.particle_qd.device_buffer(),
                state.tet_f_plastic.device_buffer(),
                state.tet_plastic_strain.device_buffer(),
                state.tet_back_stress.device_buffer(),
            ]
        })
        .flatten()
        .map(|buffer| readback::read::<u8>(engine, buffer, buffer.size() as usize))
        .collect()
}

#[test]
fn independent_material_residual_is_translation_invariant() {
    let model = model(true, false);
    let q = [Vec3::ZERO, Vec3::X * 0.02, Vec3::Y * 0.02, Vec3::Z * 0.02].map(projected);
    let v = [Vec3::new(0.2, -0.1, 0.3); 4];
    let h = scalar(1.0 / 9600.0);
    let base = residual(&model, q, [Vec3::ZERO; 4], [Vec3::ZERO; 4], h);
    let moved = residual(&model, q, [Vec3::ZERO; 4], v, h);
    for i in 0..4 {
        assert!((moved[i] - base[i] - v[i]).length() < 1e-10);
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn material_linearization_handles_empty_topology_and_pinned_rows() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let gravity = Vec3::new(-2.0, -9.81, 0.1);
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), gravity);
    builder.set_linear_damping(0.0);
    builder.add_particle(Vec3::ZERO, 1.0, 0.0, WorldId(0));
    builder.add_particle(Vec3::X, 0.0, 0.0, WorldId(0));
    let mut model = builder.build().unwrap();
    let mut input = model.state().unwrap();
    let mut output = model.state().unwrap();
    input
        .particle_qd
        .host_mut()
        .unwrap()
        .fill(Vec3::new(0.1, 0.2, 0.3));
    output
        .particle_qd
        .host_mut()
        .unwrap()
        .fill(Vec3::new(0.4, 0.5, 0.6));
    model.upload_to_device(device, queue).unwrap();
    input.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    let mut solver = VbdSolver::new_gpu(&engine, &model).unwrap();
    let data = [[0.2_f32, -0.3, 0.1, 0.0]; 2];
    let direction = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&data),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let h = scalar(1.0 / 9600.0);
    let (residual, action) = solver
        .material_linearization_gpu(&model, &input, &mut output, h, &direction)
        .unwrap();
    assert_eq!(readback::read::<[f32; 4]>(&engine, action, 2), data);
    let residual = readback::read::<[f32; 4]>(&engine, residual, 2);
    let expected = projected(Vec3::new(0.4, 0.5, 0.6))
        - projected(Vec3::new(0.1, 0.2, 0.3))
        - h * projected(gravity);
    for k in 0..3 {
        assert!((residual[0][k] as f64 - expected[k]).abs() < 1e-6);
        assert_eq!(residual[1][k], [0.4, 0.5, 0.6][k]);
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn analytic_material_action_matches_independent_finite_differences() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (viscous, pinned) in [(false, false), (true, false), (true, true)] {
        let mut model = model(viscous, pinned);
        if pinned {
            model.spring_rest_length.host_mut().unwrap()[0] = 0.04;
        }
        let mut input = model.state().unwrap();
        let mut output = model.state().unwrap();
        let rotation = Quat::from_rotation_z(0.7) * Quat::from_rotation_x(0.4);
        let positions = std::array::from_fn(|i| {
            projected(
                rotation * (input.particle_q.host().unwrap()[i] * Vec3::new(1.01, 0.97, 1.04))
                    + Vec3::new(0.4, -0.2, 0.3),
            )
        });
        let target = [Vec3::new(0.02, -0.01, 0.03); 4].map(projected);
        input
            .particle_q
            .host_mut()
            .unwrap()
            .copy_from_slice(&positions);
        input
            .particle_qd
            .host_mut()
            .unwrap()
            .copy_from_slice(&target);
        model.upload_to_device(device, queue).unwrap();
        input.upload_to_device(device, queue).unwrap();
        output.upload_to_device(device, queue).unwrap();
        let mut solver = VbdSolver::new_gpu(&engine, &model).unwrap();
        for h in [1.0 / 4800.0, 1.0 / 9600.0] {
            let h = scalar(h);
            let velocity = std::array::from_fn(|i| {
                projected(Vec3::new(
                    0.07 * (i as f64),
                    -0.025,
                    0.08 * (i as f64 - 1.0),
                ))
            });
            let bytes: Vec<[f32; 4]> = velocity
                .iter()
                .map(|v| [v.x as f32, v.y as f32, v.z as f32, 0.0])
                .collect();
            queue.write_buffer(
                output.particle_qd.device_buffer().unwrap(),
                0,
                bytemuck::cast_slice(&bytes),
            );
            for mode in 0..3 {
                let direction: [Vec3; 4] = std::array::from_fn(|i| {
                    projected(if mode == 0 {
                        Vec3::new(0.2, -0.3, 0.1)
                    } else if mode == 1 {
                        Vec3::new(i as f64 * 0.25 - 0.4, 0.3 - i as f64 * 0.11, 0.2)
                    } else {
                        Vec3::new(0.1, i as f64 * 0.19 - 0.2, -0.15)
                    })
                });
                let data: Vec<[f32; 4]> = direction
                    .iter()
                    .map(|v| [v.x as f32, v.y as f32, v.z as f32, 0.0])
                    .collect();
                let vector = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&data),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });
                let before = snapshot(&engine, &[&input, &output]);
                let (r, a) = solver
                    .material_linearization_gpu(&model, &input, &mut output, h, &vector)
                    .unwrap();
                let actual_r = readback::read::<[f32; 4]>(&engine, r, 4);
                let actual_a = readback::read::<[f32; 4]>(&engine, a, 4);
                let (r, a) = (r.clone(), a.clone());
                let doubled: Vec<[f32; 4]> = data.iter().map(|v| v.map(|x| 2.0 * x)).collect();
                queue.write_buffer(&vector, 0, bytemuck::cast_slice(&doubled));
                let mut encoder = device.create_command_encoder(&Default::default());
                solver
                    .encode_material_linearization_gpu(&mut encoder)
                    .unwrap();
                queue.submit([encoder.finish()]);
                assert_eq!(readback::read::<[f32; 4]>(&engine, &r, 4), actual_r);
                let doubled_action = readback::read::<[f32; 4]>(&engine, &a, 4);
                for (actual, original) in doubled_action
                    .iter()
                    .flatten()
                    .zip(actual_a.iter().flatten())
                {
                    assert!((*actual - 2.0 * original).abs() < 2e-6 * (1.0 + original.abs()));
                }
                queue.write_buffer(&vector, 0, bytemuck::cast_slice(&data));
                assert_eq!(before, snapshot(&engine, &[&input, &output]));
                let expected_r = residual(&model, positions, target, velocity, h);
                let epsilon = 1e-4;
                let plus = residual(
                    &model,
                    positions,
                    target,
                    std::array::from_fn(|i| velocity[i] + epsilon * direction[i]),
                    h,
                );
                let minus = residual(
                    &model,
                    positions,
                    target,
                    std::array::from_fn(|i| velocity[i] - epsilon * direction[i]),
                    h,
                );
                for i in 0..4 {
                    let expected_a = (plus[i] - minus[i]) / (2.0 * epsilon);
                    for k in 0..3 {
                        let r_error = (actual_r[i][k] as f64 - expected_r[i][k]).abs();
                        let a_error = (actual_a[i][k] as f64 - expected_a[k]).abs();
                        assert!(
                            r_error <= 3e-5 + 2e-5 * expected_r[i][k].abs(),
                            "residual {viscous} {pinned} h={h} mode={mode} node={i} k={k} gpu={} cpu={}",
                            actual_r[i][k],
                            expected_r[i][k]
                        );
                        assert!(
                            a_error <= 3e-5 + 2e-5 * expected_a[k].abs(),
                            "action {viscous} {pinned} h={h} mode={mode} node={i} k={k} gpu={} cpu={}",
                            actual_a[i][k],
                            expected_a[k]
                        );
                        if mode == 0 && !pinned {
                            assert_eq!(actual_a[i][k], data[i][k]);
                        }
                    }
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn material_linearization_rejects_invalid_buffers_and_exposes_nonfinite_state() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = model(true, false);
    let mut input = model.state().unwrap();
    let mut output = model.state().unwrap();
    model.upload_to_device(device, queue).unwrap();
    input.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    let mut solver = VbdSolver::new_gpu(&engine, &model).unwrap();
    let buffer = |size, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage,
            mapped_at_creation: false,
        })
    };
    let direction = buffer(
        64,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    for dt in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-100] {
        assert!(
            solver
                .material_linearization_gpu(&model, &input, &mut output, dt, &direction)
                .is_err()
        );
        assert!(
            solver
                .encode_material_linearization_gpu(
                    &mut device.create_command_encoder(&Default::default())
                )
                .is_err()
        );
    }
    let h = 1.0 / 9600.0;
    for invalid in [
        buffer(16, wgpu::BufferUsages::STORAGE),
        buffer(64, wgpu::BufferUsages::COPY_DST),
    ] {
        assert!(
            solver
                .material_linearization_gpu(&model, &input, &mut output, h, &invalid)
                .is_err()
        );
    }
    let (residual, action) = solver
        .material_linearization_gpu(&model, &input, &mut output, h, &direction)
        .unwrap();
    let aliases = [residual.clone(), action.clone()];
    for invalid in aliases {
        assert!(
            solver
                .material_linearization_gpu(&model, &input, &mut output, h, &invalid)
                .is_err()
        );
        assert!(
            solver
                .encode_material_linearization_gpu(
                    &mut device.create_command_encoder(&Default::default())
                )
                .is_err()
        );
    }
    for value in [f32::NAN, f32::INFINITY] {
        queue.write_buffer(&direction, 32, bytemuck::bytes_of(&value));
        let (_, action) = solver
            .material_linearization_gpu(&model, &input, &mut output, h, &direction)
            .unwrap();
        assert!(
            readback::read::<[f32; 4]>(&engine, action, 4)
                .iter()
                .flatten()
                .any(|v| !v.is_finite())
        );
    }
    queue.write_buffer(&direction, 0, &[0; 64]);
    queue.write_buffer(
        output.tet_f_plastic.device_buffer().unwrap(),
        0,
        bytemuck::bytes_of(&2.0_f32),
    );
    let (_, action) = solver
        .material_linearization_gpu(&model, &input, &mut output, h, &direction)
        .unwrap();
    assert!(
        readback::read::<[f32; 4]>(&engine, action, 4)
            .iter()
            .all(|v| !v[0].is_finite())
    );
    assert!(solver.material_force_samples_gpu().is_none());
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn retained_material_operator_drives_device_resident_gmres() {
    use molla_kernels::{GpuGmres, GpuGmresDiagnostics, GpuGmresSettings};
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = model(true, true);
    let mut input = model.state().unwrap();
    let mut output = model.state().unwrap();
    model.upload_to_device(device, queue).unwrap();
    input.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    let known = [
        [0.0_f32; 4],
        [0.1, -0.2, 0.3, 0.0],
        [-0.4, 0.05, 0.2, 0.0],
        [0.07, -0.3, 0.15, 0.0],
    ];
    let vector = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&known),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let mut solver = VbdSolver::new_gpu(&engine, &model).unwrap();
    let h = scalar(1.0 / 9600.0);
    let (_, action) = solver
        .material_linearization_gpu(&model, &input, &mut output, h, &vector)
        .unwrap();
    let action = action.clone();
    let before = snapshot(&engine, &[&input, &output]);
    let rhs = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 48,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let inverse = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[1.0_f32; 12]),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &(0..4)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage {
                        read_only: binding % 2 == 0,
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("test material Krylov packing"), source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read> packed_input: array<f32>;
             @group(0) @binding(1) var<storage, read_write> direction: array<vec4<f32>>;
             @group(0) @binding(2) var<storage, read> action: array<vec4<f32>>;
             @group(0) @binding(3) var<storage, read_write> packed_output: array<f32>;
             @compute @workgroup_size(64)
             fn unpack(@builtin(global_invocation_id) id: vec3<u32>) {
                 if id.x < 4u { direction[id.x] = vec4<f32>(packed_input[3u*id.x],packed_input[3u*id.x+1u],packed_input[3u*id.x+2u],0.0); }
             }
             @compute @workgroup_size(64)
             fn flatten(@builtin(global_invocation_id) id: vec3<u32>) {
                 if id.x < 12u { packed_output[id.x] = action[id.x/3u][id.x%3u]; }
             }".into()),
    });
    let pipelines = ["unpack", "flatten"].map(|entry| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    });
    let group = |source: &wgpu::Buffer, destination: &wgpu::Buffer| {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &layout,
            entries: &[source, &vector, &action, destination]
                .into_iter()
                .enumerate()
                .map(|(binding, b)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: b.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
        })
    };
    let dispatch =
        |encoder: &mut wgpu::CommandEncoder, pipeline: usize, group: &wgpu::BindGroup| {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipelines[pipeline]);
            pass.set_bind_group(0, group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        };
    let mut encoder = device.create_command_encoder(&Default::default());
    dispatch(&mut encoder, 1, &group(&inverse, &rhs));
    let kernel = GpuGmres::new(device, 12, 12).unwrap();
    let mut cached = None;
    kernel
        .encode(
            device,
            &mut encoder,
            &rhs,
            &inverse,
            GpuGmresSettings {
                cycles: 2,
                absolute_tolerance: 1e-7,
                relative_tolerance: 2e-6,
                zero_initial_guess: true,
            },
            |encoder, source, destination| {
                let bindings = cached.get_or_insert_with(|| group(source, destination));
                dispatch(encoder, 0, bindings);
                solver.encode_material_linearization_gpu(encoder)?;
                dispatch(encoder, 1, bindings);
                Ok(())
            },
        )
        .unwrap();
    queue.submit([encoder.finish()]);
    let report = readback::read::<GpuGmresDiagnostics>(&engine, kernel.diagnostics(), 1)[0];
    assert_eq!((report.faults, report.converged), (0, 1), "{report:?}");
    let solution = readback::read::<f32>(&engine, kernel.solution(), 12);
    for i in 0..12 {
        assert!(
            (solution[i] - known[i / 3][i % 3]).abs() < 1e-5,
            "{i} {} {report:?}",
            solution[i]
        );
    }
    let positions = std::array::from_fn(|i| projected(model.initial_particle_q.host().unwrap()[i]));
    let direction = known.map(|v| Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64));
    let epsilon = 1e-4;
    let plus = residual(
        &model,
        positions,
        [Vec3::ZERO; 4],
        direction.map(|v| epsilon * v),
        h,
    );
    let minus = residual(
        &model,
        positions,
        [Vec3::ZERO; 4],
        direction.map(|v| -epsilon * v),
        h,
    );
    let rhs_values = readback::read::<f32>(&engine, &rhs, 12);
    for i in 0..12 {
        let expected = (plus[i / 3][i % 3] - minus[i / 3][i % 3]) / (2.0 * epsilon);
        assert!((rhs_values[i] as f64 - expected).abs() < 3e-5 * (1.0 + expected.abs()));
    }
    assert_eq!(before, snapshot(&engine, &[&input, &output]));
    solver
        .material_force_residual_gpu(&model, &input, &mut output, h)
        .unwrap();
    assert!(
        solver
            .encode_material_linearization_gpu(
                &mut device.create_command_encoder(&Default::default())
            )
            .is_err()
    );
    println!("actual pinned viscous FEM material GMRES: {report:?}; physical state preserved");
}
