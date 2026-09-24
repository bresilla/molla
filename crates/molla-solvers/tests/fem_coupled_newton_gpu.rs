#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_kernels::GpuGmresSettings;
use molla_kernels::soft_rigid_contact::{SoftRigidContactGpu, SoftRigidDofBound};
use molla_kernels::soft_rigid_solve::{
    ContactLinearizationBindings, ContactLinearizationKernel, CoupledNewtonBindings,
    CoupledNewtonDiagnostics, CoupledNewtonKernel, CoupledNewtonSettings, SoftRigidSolveBindings,
    SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use molla_math::{Mat3, Vec3};
use molla_sim::{MaterialNeoHookean, Model, ModelBuilder, TetViscosity};
use molla_solvers::VbdSolver;
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, data: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(data),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}
fn xyz(v: [f32; 4]) -> Vec3 {
    Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64)
}
fn f32_value(v: f64) -> f64 {
    v as f32 as f64
}
fn edges(v: [Vec3; 4]) -> Mat3 {
    Mat3::from_cols(v[1] - v[0], v[2] - v[0], v[3] - v[0])
}
fn cofactor(f: Mat3) -> Mat3 {
    Mat3::from_cols(
        f.y_axis.cross(f.z_axis),
        f.z_axis.cross(f.x_axis),
        f.x_axis.cross(f.y_axis),
    )
}
fn trace(f: Mat3) -> f64 {
    f.x_axis.x + f.y_axis.y + f.z_axis.z
}
fn material_residual(model: &Model, velocity: [Vec3; 4], target: Vec3, h: f64) -> [Vec3; 4] {
    let inverse = Mat3::from_cols_array(
        &model.tet_dm_inv.host().unwrap()[0]
            .to_cols_array()
            .map(f32_value),
    );
    let f = Mat3::IDENTITY + edges(velocity.map(|v| h * v)) * inverse;
    let cof = cofactor(f);
    let mu = f32_value(model.tet_mu.host().unwrap()[0]);
    let lambda = f32_value(model.tet_lambda.host().unwrap()[0]);
    let viscosity = model.tet_viscosity.host().unwrap()[0];
    let strain = 0.5 * (f.transpose() * f - Mat3::IDENTITY);
    let viscous =
        2.0 * f32_value(viscosity.shear) * (strain - Mat3::IDENTITY * (trace(strain) / 3.0))
            + f32_value(viscosity.bulk) * trace(strain) * Mat3::IDENTITY;
    let stress = mu * (f - cof) + (lambda + mu) * (f.determinant() - 1.0) * cof + f * viscous / h;
    let volume = f32_value(model.tet_rest_volume.host().unwrap()[0]);
    std::array::from_fn(|i| {
        let gradient = if i == 0 {
            -inverse.transpose() * Vec3::ONE
        } else {
            inverse.transpose().col(i - 1)
        };
        let force = -volume * stress * gradient;
        velocity[i] - target - h * f32_value(model.particle_inv_mass.host().unwrap()[i]) * force
    })
}
fn settings() -> CoupledNewtonSettings {
    CoupledNewtonSettings {
        block_maximum: false,
        iterations: 8,
        backtracks: 8,
        absolute_tolerance: 2e-5,
        relative_tolerance: 0.0,
        armijo: 1e-4,
        linear: GpuGmresSettings {
            cycles: 2,
            absolute_tolerance: 1e-7,
            relative_tolerance: 1e-5,
            zero_initial_guess: true,
        },
    }
}

#[test]
#[ignore = "requires real GPU through make test-gpu"]
fn coupled_fem_contact_newton_preserves_momentum_and_transfers_torque() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for friction in [0.0_f32, 0.25, 10.0] {
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        builder.set_linear_damping(0.0);
        for point in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
            builder.add_particle(Vec3::new(0.0, 0.2, 0.0) + point * 0.2, 1.0, 0.0, WorldId(0));
        }
        let mut material = MaterialNeoHookean::elastic(8000.0, 0.3);
        material.viscosity = TetViscosity {
            shear: 20.0,
            bulk: 40.0,
        };
        builder
            .add_tet_mesh_per_material(&[[0, 1, 2, 3]], &[material], &[0], None)
            .unwrap();
        let mut model = builder.build().unwrap();
        let mut input = model.state().unwrap();
        let mut candidate = model.state().unwrap();
        let target = xyz([-0.6, -0.3, 0.05, 0.0]);
        input.particle_qd.host_mut().unwrap().fill(target);
        candidate.particle_qd.host_mut().unwrap().fill(target);
        model.upload_to_device(device, queue).unwrap();
        input.upload_to_device(device, queue).unwrap();
        candidate.upload_to_device(device, queue).unwrap();
        let soft_velocity = candidate.particle_qd.device_buffer().unwrap().clone();
        let rigid_reference = [0.4_f32, 0.0, 0.0, 1.5];
        let mut jacobian = [0.0_f32; 24];
        for d in 0..3 {
            jacobian[d * 4 + d] = 1.0;
        }
        jacobian[23] = 1.0;
        let mass_diagonal = [3.0_f32, 5.0, 4.0, 0.4];
        let mut mass = [0.0_f32; 16];
        for d in 0..4 {
            mass[d * 4 + d] = mass_diagonal[d];
        }
        let contact = SoftRigidContactGpu {
            normal_gap: [0.0, 1.0, 0.0, 0.0],
            material: [friction, 0.0, 0.0, 0.0],
            ids: [0, 0, 0, 1],
            support: [0, 0, 1, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        let bound = SoftRigidDofBound {
            dof: 3,
            sign: -1.0,
            speed: -1.55,
        }
        .row(4)
        .unwrap();
        let contacts = buffer(device, &[contact, bound]);
        let poses = buffer(device, &[[0.0_f32; 4], [0.0, 0.0, 0.0, 1.0]]);
        let com = buffer(device, &[[0.0_f32; 4]]);
        let jacobians = buffer(device, &jacobian);
        let matrix = buffer(device, &mass);
        let reference = buffer(device, &rigid_reference);
        let rigid_velocity = buffer(device, &rigid_reference);
        let body_velocity = buffer(device, &[[0.0_f32; 4]; 2]);
        let impulses = buffer(device, &[[0.0_f32; 4]; 2]);
        let soft_direction = buffer(device, &[[0.0_f32; 4]; 4]);
        let rigid_direction = buffer(device, &[0.0_f32; 4]);
        let impulse_direction = buffer(device, &[[0.0_f32; 4]; 2]);
        let h = f32_value(0.01);
        let contact_settings = SoftRigidSolveSettings {
            iterations: 1,
            dt: h as f32,
            penetration_recovery: 0.2,
            max_recovery_speed: 1.0,
        };
        let state = SoftRigidSolveBindings {
            positions: input.particle_q.device_buffer().unwrap(),
            velocities: &soft_velocity,
            inverse_mass: model.particle_inv_mass.device_buffer().unwrap(),
            contacts: &contacts,
            body_poses: &poses,
            body_com: &com,
            jacobians: &jacobians,
            mass_matrix: &matrix,
            rigid_velocities: &rigid_velocity,
            body_velocities: &body_velocity,
            particle_count: 4,
            body_count: 1,
        };
        let solve = SoftRigidSolveKernel::new(device, 4, 2).unwrap();
        let mut contact_operator = ContactLinearizationKernel::new(device, 4, 2, 4, 1).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        solve
            .encode_preparation(device, &mut encoder, state, contact_settings)
            .unwrap();
        contact_operator
            .prepare(
                device,
                &mut encoder,
                &solve,
                ContactLinearizationBindings {
                    state,
                    rigid_reference: &reference,
                    soft_direction: &soft_direction,
                    rigid_direction: &rigid_direction,
                    impulses: &impulses,
                    impulse_direction: &impulse_direction,
                },
                contact_settings,
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        assert_eq!(
            readback::read::<u32>(&engine, contact_operator.status(), 2),
            [0, 0]
        );
        let mut material_operator = VbdSolver::new_gpu(&engine, &model).unwrap();
        let (material_residual_buffer, material_action) = material_operator
            .material_linearization_gpu(&model, &input, &mut candidate, h, &soft_direction)
            .unwrap();
        let material_residual_buffer = material_residual_buffer.clone();
        let material_action = material_action.clone();
        let newton = CoupledNewtonKernel::new(
            device,
            4,
            2,
            4,
            22,
            CoupledNewtonBindings {
                soft_velocity: &soft_velocity,
                rigid_velocity: &rigid_velocity,
                impulses: &impulses,
                soft_direction: &soft_direction,
                rigid_direction: &rigid_direction,
                impulse_direction: &impulse_direction,
                material_residual: &material_residual_buffer,
                material_action: &material_action,
                contact: &contact_operator,
            },
        )
        .unwrap();
        let immutable = [
            input.particle_q.device_buffer().unwrap(),
            input.particle_qd.device_buffer().unwrap(),
            candidate.particle_q.device_buffer().unwrap(),
            &poses,
            &com,
            &jacobians,
            &matrix,
            &reference,
            &contacts,
            solve.row_state(),
            solve.projected_rows(),
            solve.row_responses(),
        ];
        let before: Vec<_> = immutable
            .iter()
            .map(|b| readback::read::<u8>(&engine, b, b.size() as usize))
            .collect();
        let scaling: [f32; 22] = std::array::from_fn(|i| 0.25 * (1 + (i % 7)) as f32);
        let preconditioner = buffer(device, &scaling);
        let mut encoder = device.create_command_encoder(&Default::default());
        newton
            .encode(
                device,
                &mut encoder,
                &preconditioner,
                settings(),
                |encoder| {
                    material_operator.encode_material_linearization_gpu(encoder)?;
                    contact_operator.encode(encoder)
                },
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        let report =
            readback::read::<CoupledNewtonDiagnostics>(&engine, newton.diagnostics(), 1)[0];
        assert_eq!(
            (report.faults, report.converged),
            (0, 1),
            "mu={friction} {report:?}"
        );
        let soft: [Vec3; 4] = readback::read::<[f32; 4]>(&engine, &soft_velocity, 4)
            .into_iter()
            .map(xyz)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let rigid = readback::read::<f32>(&engine, &rigid_velocity, 4);
        let multipliers = readback::read::<[f32; 4]>(&engine, &impulses, 2);
        let lambda = xyz(multipliers[0]);
        let bound_impulse = multipliers[1][0] as f64;
        assert!(bound_impulse >= -2e-5 && rigid[3] <= 1.55 + 2e-5);
        assert!((bound_impulse * (1.55 - rigid[3] as f64)).abs() < 2e-5);
        assert!(multipliers[1][1].abs() < 1e-6 && multipliers[1][2].abs() < 1e-6);
        if friction == 10.0 {
            assert!(bound_impulse > 1e-3 && (rigid[3] - 1.55).abs() < 2e-5);
        }
        let world_impulse = Vec3::new(-lambda.z, lambda.x, -lambda.y);
        let material_r = material_residual(&model, soft, target, h);
        for (i, r) in material_r.into_iter().enumerate() {
            let balance = r - if i == 0 { world_impulse } else { Vec3::ZERO };
            assert!(
                balance.length() < 8e-5,
                "material momentum node={i} mu={friction} balance={balance:?}"
            );
        }
        let arm = f32_value(0.2);
        let reaction = [
            -lambda.z,
            lambda.x,
            -lambda.y,
            arm * lambda.z + bound_impulse,
        ];
        for d in 0..4 {
            let balance = mass_diagonal[d] as f64 * (rigid[d] as f64 - rigid_reference[d] as f64)
                + reaction[d];
            assert!(
                balance.abs() < 8e-5,
                "rigid impulse/torque dof={d} mu={friction} balance={balance}"
            );
        }
        let relative = soft[0]
            - Vec3::new(
                rigid[0] as f64 - arm * rigid[3] as f64,
                rigid[1] as f64,
                rigid[2] as f64,
            );
        assert!(
            lambda.x >= -2e-5 && relative.y >= -2e-5 && (lambda.x * relative.y).abs() < 2e-5,
            "normal mu={friction} lambda={lambda:?} relative={relative:?}"
        );
        let tangent = Vec3::new(0.0, lambda.y, lambda.z);
        let radius = friction as f64 * lambda.x.max(0.0);
        assert!(tangent.length() <= radius + 2e-5);
        let tangent_speed = Vec3::new(0.0, -relative.z, -relative.x);
        let lipschitz = 1.0 + 1.0 / 3.0 + arm * arm / (mass_diagonal[3] as f64);
        let trial = tangent - tangent_speed / lipschitz;
        let projected = trial * (radius / trial.length().max(1e-30)).min(1.0);
        assert!(
            (tangent - projected).length() < 2e-5,
            "Coulomb mu={friction} lambda={lambda:?} relative={relative:?}"
        );
        if friction > 0.0 {
            assert!(
                (rigid[3] - rigid_reference[3]).abs() > 1e-3,
                "no torque transfer"
            );
        }
        let rhs = readback::read::<f32>(&engine, newton.residual_rhs(), 22);
        let norm = rhs.iter().map(|v| (*v as f64).powi(2)).sum::<f64>().sqrt();
        assert!(norm < 2.1e-5 && (norm - report.residual_norm as f64).abs() < 1e-6);
        for (b, original) in immutable.iter().zip(before) {
            assert_eq!(
                readback::read::<u8>(&engine, b, b.size() as usize),
                original
            );
        }
        println!(
            "actual coupled FEM/contact mu={friction}: {report:?}, impulse={lambda:?}, angular speed={} -> {}",
            rigid_reference[3], rigid[3]
        );
    }
}

#[test]
#[ignore = "requires real GPU through make test-gpu"]
fn newton_backtracking_rejects_nonfinite_trials_and_restores_failed_updates() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let initial = [0.1_f32, 0.0, 0.0, 99.0];
    let contact = SoftRigidContactGpu {
        normal_gap: [0.0, 1.0, 0.0, 0.0],
        material: [0.0; 4],
        ids: [0, 0, 0, 0],
        support: [0, 0, 1, 0],
        weights: [1.0, 0.0, 0.0, 0.0],
    };
    let b = [
        buffer(device, &[[0.0_f32; 4]]),
        buffer(device, &[initial]),
        buffer(device, &[1.0_f32]),
        buffer(device, &[contact]),
        buffer(device, &[[0.0_f32; 4], [0.0, 0.0, 0.0, 1.0]]),
        buffer(device, &[[0.0_f32; 4]]),
        buffer(device, &[0.0_f32; 6]),
        buffer(device, &[1.0_f32]),
        buffer(device, &[0.0_f32]),
        buffer(device, &[[0.0_f32; 4]; 2]),
        buffer(device, &[0.0_f32]),
        buffer(device, &[[0.0_f32; 4]]),
        buffer(device, &[[0.0_f32; 4]]),
        buffer(device, &[0.0_f32]),
        buffer(device, &[[0.0_f32; 4]]),
        buffer(device, &[[0.0_f32; 4]]),
        buffer(device, &[[0.0_f32; 4]]),
    ];
    let state = SoftRigidSolveBindings {
        positions: &b[0],
        velocities: &b[1],
        inverse_mass: &b[2],
        contacts: &b[3],
        body_poses: &b[4],
        body_com: &b[5],
        jacobians: &b[6],
        mass_matrix: &b[7],
        rigid_velocities: &b[8],
        body_velocities: &b[9],
        particle_count: 1,
        body_count: 1,
    };
    let solve = SoftRigidSolveKernel::new(device, 1, 1).unwrap();
    let mut contact_operator = ContactLinearizationKernel::new(device, 1, 1, 1, 1).unwrap();
    let contact_settings = SoftRigidSolveSettings {
        iterations: 1,
        dt: 0.01,
        penetration_recovery: 0.2,
        max_recovery_speed: 1.0,
    };
    let mut encoder = device.create_command_encoder(&Default::default());
    solve
        .encode_preparation(device, &mut encoder, state, contact_settings)
        .unwrap();
    contact_operator
        .prepare(
            device,
            &mut encoder,
            &solve,
            ContactLinearizationBindings {
                state,
                rigid_reference: &b[10],
                soft_direction: &b[12],
                rigid_direction: &b[13],
                impulses: &b[11],
                impulse_direction: &b[14],
            },
            contact_settings,
        )
        .unwrap();
    queue.submit([encoder.finish()]);
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("nonlinear backtracking control, not a material model"),
        source: wgpu::ShaderSource::Wgsl(
            "@group(0) @binding(0) var<storage, read> velocity: array<vec4<f32>>;
             @group(0) @binding(1) var<storage, read> direction: array<vec4<f32>>;
             @group(0) @binding(2) var<storage, read_write> residual: array<vec4<f32>>;
             @group(0) @binding(3) var<storage, read_write> action: array<vec4<f32>>;
             @compute @workgroup_size(1)
             fn evaluate() {
                 let v = velocity[0]; let d = direction[0];
                 residual[0] = vec4<f32>(v.x*v.x*v.x-1.0,v.y,v.z,0.0);
                 action[0] = vec4<f32>(3.0*v.x*v.x*d.x,d.y,d.z,0.0);
                 if v.x > 2.0 { residual[0].x = bitcast<f32>(0x7fc00000u); }
             }"
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("evaluate"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[&b[1], &b[12], &b[15], &b[16]]
            .into_iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    let bindings = CoupledNewtonBindings {
        soft_velocity: &b[1],
        rigid_velocity: &b[8],
        impulses: &b[11],
        soft_direction: &b[12],
        rigid_direction: &b[13],
        impulse_direction: &b[14],
        material_residual: &b[15],
        material_action: &b[16],
        contact: &contact_operator,
    };
    let newton = CoupledNewtonKernel::new(device, 1, 1, 1, 7, bindings).unwrap();
    let inverse = buffer(device, &[1.0_f32; 7]);
    for (n, r, d) in [(0, 1, 1), (1, 0, 1), (1, 1, 0), (2, 1, 1), (u32::MAX, 1, 1)] {
        assert!(CoupledNewtonKernel::new(device, n, r, d, 7, bindings).is_err());
    }
    assert!(
        CoupledNewtonKernel::new(
            device,
            1,
            1,
            1,
            7,
            CoupledNewtonBindings {
                soft_direction: &b[1],
                ..bindings
            }
        )
        .is_err()
    );
    for bad in [
        CoupledNewtonSettings {
            iterations: 0,
            ..settings()
        },
        CoupledNewtonSettings {
            armijo: f32::NAN,
            ..settings()
        },
        CoupledNewtonSettings {
            linear: GpuGmresSettings {
                cycles: 0,
                ..settings().linear
            },
            ..settings()
        },
    ] {
        assert!(
            newton
                .encode(
                    device,
                    &mut device.create_command_encoder(&Default::default()),
                    &inverse,
                    bad,
                    |_| panic!("invalid settings callback")
                )
                .is_err()
        );
    }
    assert!(
        newton
            .encode(
                device,
                &mut device.create_command_encoder(&Default::default()),
                newton.residual_rhs(),
                settings(),
                |_| panic!("aliased input callback")
            )
            .is_err()
    );
    let evaluate = |encoder: &mut wgpu::CommandEncoder| {
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        contact_operator.encode(encoder)
    };
    for block_maximum in [true, false] {
        queue.write_buffer(&b[1], 0, bytemuck::bytes_of(&initial));
        queue.write_buffer(&b[8], 0, bytemuck::bytes_of(&0.8_f32));
        let mut encoder = device.create_command_encoder(&Default::default());
        newton
            .encode(
                device,
                &mut encoder,
                &inverse,
                CoupledNewtonSettings {
                    absolute_tolerance: 1.05,
                    relative_tolerance: 0.0,
                    block_maximum,
                    ..settings()
                },
                evaluate,
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        let report =
            readback::read::<CoupledNewtonDiagnostics>(&engine, newton.diagnostics(), 1)[0];
        assert_eq!((report.faults, report.converged), (0, 1));
        assert!((report.initial_maximum_norm - 0.999).abs() < 1e-6);
        assert!(report.initial_norm > 1.05);
        if block_maximum {
            assert_eq!(report.iterations, 0);
            assert!(
                report.residual_norm > report.threshold && report.maximum_norm < report.threshold
            );
        } else {
            assert!(report.iterations > 0 && report.residual_norm <= report.threshold);
        }
    }
    queue.write_buffer(&b[8], 0, bytemuck::bytes_of(&0.0_f32));
    for backtracks in [1, 12] {
        queue.write_buffer(&b[1], 0, bytemuck::bytes_of(&initial));
        let mut encoder = device.create_command_encoder(&Default::default());
        newton
            .encode(
                device,
                &mut encoder,
                &inverse,
                CoupledNewtonSettings {
                    backtracks,
                    ..settings()
                },
                evaluate,
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        let report =
            readback::read::<CoupledNewtonDiagnostics>(&engine, newton.diagnostics(), 1)[0];
        let v = readback::read::<[f32; 4]>(&engine, &b[1], 1)[0];
        assert_eq!(v[3], 99.0);
        if backtracks == 1 {
            assert_eq!((report.faults, report.converged), (4, 0), "{report:?}");
            assert_eq!(v, initial);
            assert!((report.residual_norm - 0.999).abs() < 1e-6);
        } else {
            assert_eq!((report.faults, report.converged), (0, 1), "{report:?}");
            assert!(((v[0] as f64).powi(3) - 1.0).abs() < 2e-5);
            assert!((v[0] - 1.0).abs() < 1e-5 && report.line_evaluations > report.iterations);
        }
        println!("synthetic backtracking control backtracks={backtracks}: {report:?}");
    }
    queue.write_buffer(&b[1], 0, bytemuck::bytes_of(&initial));
    let invalid_final = buffer(device, &[initial]);
    let mut encoder = device.create_command_encoder(&Default::default());
    newton
        .encode_split_with_finalize(
            device,
            &mut encoder,
            &inverse,
            settings(),
            |encoder, _| evaluate(encoder),
            |encoder| {
                encoder.copy_buffer_to_buffer(&invalid_final, 0, &b[1], 0, 16);
                Ok(())
            },
        )
        .unwrap();
    queue.submit([encoder.finish()]);
    let report = readback::read::<CoupledNewtonDiagnostics>(&engine, newton.diagnostics(), 1)[0];
    assert_eq!((report.faults, report.converged), (0, 0));
    assert!(report.iterations > 0 && (report.residual_norm - 0.999).abs() < 1e-6);
    assert_eq!(readback::read::<[f32; 4]>(&engine, &b[1], 1)[0], initial);
    for invalid in [0.0_f32, f32::NAN, f32::INFINITY] {
        queue.write_buffer(&b[1], 0, bytemuck::bytes_of(&initial));
        queue.write_buffer(&inverse, 0, bytemuck::bytes_of(&invalid));
        let mut encoder = device.create_command_encoder(&Default::default());
        newton
            .encode(device, &mut encoder, &inverse, settings(), evaluate)
            .unwrap();
        queue.submit([encoder.finish()]);
        let report =
            readback::read::<CoupledNewtonDiagnostics>(&engine, newton.diagnostics(), 1)[0];
        assert_eq!((report.faults, report.converged), (1, 0), "{report:?}");
        assert_eq!(readback::read::<[f32; 4]>(&engine, &b[1], 1)[0], initial);
    }
    queue.write_buffer(&inverse, 0, bytemuck::bytes_of(&1.0_f32));
    queue.write_buffer(
        contact_operator.status(),
        0,
        bytemuck::cast_slice(&[2_u32, 2]),
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    newton
        .encode(device, &mut encoder, &inverse, settings(), evaluate)
        .unwrap();
    queue.submit([encoder.finish()]);
    let report = readback::read::<CoupledNewtonDiagnostics>(&engine, newton.diagnostics(), 1)[0];
    assert_eq!((report.faults, report.converged), (1, 0), "{report:?}");
    assert_eq!(readback::read::<[f32; 4]>(&engine, &b[1], 1)[0], initial);
}
