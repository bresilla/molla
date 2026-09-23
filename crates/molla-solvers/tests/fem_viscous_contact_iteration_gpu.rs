#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_kernels::soft_rigid_contact::SoftRigidContactGpu;
use molla_kernels::soft_rigid_solve::{
    SoftRigidSolveBindings, SoftRigidSolveKernel, SoftRigidSolveSettings,
};
use molla_math::{Mat3, Vec3};
use molla_sim::{Contacts, MaterialNeoHookean, Model, ModelBuilder, TetViscosity};
use molla_solvers::{Solver, VbdSolver};
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

#[path = "common/fem_rotor.rs"]
#[allow(dead_code)]
mod fem_rotor;

fn model(pinned: bool, viscous: bool) -> Model {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.set_linear_damping(0.0);
    for (i, p) in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z]
        .into_iter()
        .enumerate()
    {
        let mass = if pinned && i != 2 {
            0.0
        } else {
            1000.0 * 0.02_f64.powi(3) / 24.0
        };
        builder.add_particle(p * 0.02, mass, 0.0, WorldId(0));
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
    builder.build().unwrap()
}

fn buffer<T: bytemuck::Pod>(device: &wgpu::Device, values: &[T]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("viscous contact iteration fixture"),
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn reference(h: f64, viscous: bool) -> Vec3 {
    let model = model(true, viscous);
    let mut input = model.state().unwrap();
    input.particle_qd.host_mut().unwrap().fill(Vec3::NEG_Y);
    let mut output = model.state().unwrap();
    VbdSolver::new_cpu()
        .with_iterations(256)
        .step(
            &model,
            &input,
            &model.control(),
            &Contacts::with_capacity(0),
            h,
            &mut output,
        )
        .unwrap();
    output.particle_qd.host().unwrap()[2]
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_material_contact_matches_reference_and_rejects_iteration_exhaustion() {
    use molla_solvers::fem_rigid_gpu::{
        FemRigidGpuSystem, MaterialContactConfig, SoftRigidShapeGpu,
    };
    let engine = std::sync::Arc::new(Engine::new_gpu().unwrap());
    let h = 1.0 / 19200.0;
    let expected = reference(h, true);
    let mut baseline = None;
    for (iterations, disable, reuse) in [
        (None, false, None),
        (Some(2), false, None),
        (Some(256), false, None),
        (Some(256), true, None),
        (Some(2), false, Some(1)),
        (Some(256), false, Some(1)),
        (Some(256), false, Some(8)),
        (Some(256), true, Some(1)),
    ] {
        let mut scene = fem_rotor::scene();
        scene.soft_model = model(false, true);
        scene.soft_state = scene.soft_model.state().unwrap();
        scene
            .soft_state
            .particle_qd
            .host_mut()
            .unwrap()
            .fill(Vec3::NEG_Y);
        scene.shapes.push(SoftRigidShapeGpu {
            position: [0.0, -0.1, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            data: [10.0, 0.1, 10.0, 1000.0],
            ids: [1, u32::MAX, 0, 0],
        });
        let mut system = FemRigidGpuSystem::new(
            engine.clone(),
            scene,
            molla_solvers::fem_rigid::FemRigidConfig {
                max_substep: h,
                elastic_iterations: 128,
                ..Default::default()
            },
        )
        .unwrap();
        if let Some(iterations) = iterations {
            system
                .configure_material_contact(Some(MaterialContactConfig {
                    iterations,
                    linear_tolerance: 1e-4,
                    angular_tolerance: 1e-4,
                }))
                .unwrap();
            assert!(
                system
                    .configure_material_contact(Some(MaterialContactConfig {
                        iterations: 0,
                        linear_tolerance: 1e-4,
                        angular_tolerance: 1e-4,
                    }))
                    .is_err()
            );
            assert!(system.material_contact_diagnostics().is_some());
            system.configure_material_contact_reuse(reuse).unwrap();
            for invalid in [0, 4097, u32::MAX] {
                assert!(
                    system
                        .configure_material_contact_reuse(Some(invalid))
                        .is_err()
                );
            }
        } else {
            assert!(system.configure_material_contact_reuse(Some(1)).is_err());
        }
        if disable {
            system.configure_material_contact(None).unwrap();
        }
        system.step(h).unwrap();
        assert!(system.configure_material_contact(None).is_err());
        assert!(system.configure_material_contact_reuse(None).is_err());
        let status = readback::read::<u32>(&engine, system.status_buffer(), 1)[0];
        let velocities = readback::read::<[f32; 4]>(
            &engine,
            system.soft_state().particle_qd.device_buffer().unwrap(),
            4,
        );
        let positions = readback::read::<[f32; 4]>(
            &engine,
            system.soft_state().particle_q.device_buffer().unwrap(),
            4,
        );
        if iterations == Some(2) {
            assert_eq!(status, 64);
            let residual =
                readback::read::<f32>(&engine, system.material_contact_diagnostics().unwrap(), 3);
            assert!(residual[0] > 1e-4);
        } else {
            assert_eq!(status, 0);
            if iterations.is_none() {
                baseline = Some((positions.clone(), velocities.clone()));
            }
            if disable {
                assert_eq!(baseline.as_ref().unwrap(), &(positions, velocities));
            } else if iterations.is_some() {
                let actual = Vec3::new(
                    velocities[2][0] as f64,
                    velocities[2][1] as f64,
                    velocities[2][2] as f64,
                );
                assert!(
                    (actual - expected).length() < 5e-4,
                    "runtime {actual:?}, oracle {expected:?}"
                );
                let residual = readback::read::<f32>(
                    &engine,
                    system.material_contact_diagnostics().unwrap(),
                    3,
                );
                assert!(residual.iter().all(|v| v.is_finite() && *v <= 1e-4));
                let force =
                    readback::read::<f32>(&engine, system.material_force_diagnostics().unwrap(), 1)
                        [0];
                assert!(force.is_finite() && force <= 1e-4, "force residual={force}");
                eprintln!(
                    "runtime material contact reuse={reuse:?} matches CPU, residuals={residual:?} force={force}"
                );
            }
        }
        assert!(system.soft_state().particle_q.host().is_err());
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_material_contact_transfers_impulse_to_a_dynamic_support() {
    use molla_core::BodyId;
    use molla_math::{SpatialInertia, Transform};
    use molla_sim::BodyParams;
    use molla_solvers::fem_rigid_gpu::{
        FemRigidGpuSystem, MaterialContactConfig, SoftRigidShapeGpu,
    };
    let engine = std::sync::Arc::new(Engine::new_gpu().unwrap());
    let m = 1000.0 * 0.02_f64.powi(3) / 24.0;
    let rigid_mass = 0.001;
    for (h, reuse) in [
        (1.0 / 19200.0, None),
        (1.0 / 38400.0, None),
        (1.0 / 19200.0, Some(1)),
        (1.0 / 38400.0, Some(1)),
    ] {
        let mut oracle = model(true, true);
        let reduced_mass = m * (rigid_mass + 3.0 * m) / (rigid_mass + 4.0 * m);
        oracle.particle_mass.host_mut().unwrap()[2] = reduced_mass;
        oracle.particle_inv_mass.host_mut().unwrap()[2] = 1.0 / reduced_mass;
        let mut input = oracle.state().unwrap();
        input.particle_qd.host_mut().unwrap()[2].y = -rigid_mass / (rigid_mass + 3.0 * m);
        let mut output = oracle.state().unwrap();
        VbdSolver::new_cpu()
            .with_iterations(256)
            .step(
                &oracle,
                &input,
                &oracle.control(),
                &Contacts::with_capacity(0),
                h,
                &mut output,
            )
            .unwrap();
        let relative = output.particle_qd.host().unwrap()[2].y;
        let expected_body = (-4.0 * m - m * relative) / (rigid_mass + 4.0 * m);
        let expected_top = expected_body + relative;
        let mut scene = fem_rotor::scene();
        scene.soft_model = model(false, true);
        scene.soft_state = scene.soft_model.state().unwrap();
        scene
            .soft_state
            .particle_qd
            .host_mut()
            .unwrap()
            .fill(Vec3::NEG_Y);
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        builder.set_linear_damping(0.0);
        builder.begin_articulation();
        let body = builder.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: rigid_mass,
            com: Vec3::ZERO,
            inertia: Mat3::IDENTITY * 0.01,
        }));
        builder
            .add_joint_prismatic(
                BodyId::NONE,
                body,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::Y,
            )
            .unwrap();
        scene.rigid_model = builder.build().unwrap();
        scene.rigid_state = scene.rigid_model.state().unwrap();
        scene.control = scene.rigid_model.control();
        scene.shapes.push(SoftRigidShapeGpu {
            position: [0.0, -0.1, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            data: [10.0, 0.1, 10.0, 1000.0],
            ids: [1, body.index() as u32, 0, 0],
        });
        let mut system = FemRigidGpuSystem::new(
            engine.clone(),
            scene,
            molla_solvers::fem_rigid::FemRigidConfig {
                max_substep: h,
                elastic_iterations: 128,
                ..Default::default()
            },
        )
        .unwrap();
        system
            .configure_material_contact(Some(MaterialContactConfig {
                iterations: 256,
                linear_tolerance: 1e-4,
                angular_tolerance: 1e-4,
            }))
            .unwrap();
        system.configure_material_contact_reuse(reuse).unwrap();
        system.step(h).unwrap();
        assert_eq!(
            readback::read::<u32>(&engine, system.status_buffer(), 1),
            [0]
        );
        let velocities = readback::read::<[f32; 4]>(
            &engine,
            system.soft_state().particle_qd.device_buffer().unwrap(),
            4,
        );
        let body_velocity = readback::read::<f32>(
            &engine,
            system.rigid_state().joint_qd.device_buffer().unwrap(),
            1,
        )[0] as f64;
        assert!((body_velocity - expected_body).abs() < 5e-4);
        assert!((velocities[2][1] as f64 - expected_top).abs() < 5e-4);
        for i in [0, 1, 3] {
            assert!((velocities[i][1] as f64 - body_velocity).abs() < 1e-4);
        }
        let momentum =
            m * velocities.iter().map(|v| v[1] as f64).sum::<f64>() + rigid_mass * body_velocity;
        assert!((momentum + 4.0 * m).abs() < 1e-7);
        let residuals =
            readback::read::<f32>(&engine, system.material_contact_diagnostics().unwrap(), 3);
        assert!(residuals.iter().all(|v| v.is_finite() && *v <= 1e-4));
        let force =
            readback::read::<f32>(&engine, system.material_force_diagnostics().unwrap(), 1)[0];
        assert!(
            force.is_finite() && force <= 1e-4,
            "dynamic force residual={force}"
        );
        eprintln!(
            "dynamic support h={h}: body={body_velocity}/{expected_body}, top={}/{expected_top}, residuals={residuals:?}",
            velocities[2][1]
        );
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn material_reuse_survives_substeps_and_idle_mode_changes() {
    material_reuse_trajectory(0.85, None);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn material_reuse_rejects_unconverged_high_friction_trajectory() {
    material_reuse_trajectory(1000.0, Some(2));
}

fn material_reuse_trajectory(friction: f32, rejected_step: Option<usize>) {
    use molla_solvers::fem_rigid_gpu::{
        FemRigidGpuSystem, MaterialContactConfig, SoftRigidShapeGpu,
    };
    let engine = std::sync::Arc::new(Engine::new_gpu().unwrap());
    let h = 1.0 / 19200.0;
    let build = |outer| {
        let mut scene = fem_rotor::scene();
        scene.soft_model = model(false, true);
        scene.soft_state = scene.soft_model.state().unwrap();
        scene
            .soft_state
            .particle_qd
            .host_mut()
            .unwrap()
            .fill(Vec3::NEG_Y);
        scene.shapes.push(SoftRigidShapeGpu {
            position: [0.0, -0.1, 0.0, 0.0],
            rotation: [0.0, 0.0, 0.0, 1.0],
            data: [10.0, 0.1, 10.0, friction],
            ids: [1, u32::MAX, 0, 0],
        });
        let mut system = FemRigidGpuSystem::new(
            engine.clone(),
            scene,
            molla_solvers::fem_rigid::FemRigidConfig {
                max_substep: h,
                elastic_iterations: 128,
                ..Default::default()
            },
        )
        .unwrap();
        system
            .configure_material_contact(Some(MaterialContactConfig {
                iterations: outer,
                linear_tolerance: 1e-4,
                angular_tolerance: 1e-4,
            }))
            .unwrap();
        system
    };
    let mut cold = build(256);
    let mut retained = build(256);
    let mass = 1000.0 * 0.02_f64.powi(3) / 24.0;
    let mut previous_energy = [2.0 * mass; 2];
    for (step, reuse) in [Some(8), None, Some(8)].into_iter().enumerate() {
        retained.configure_material_contact_reuse(reuse).unwrap();
        let mut results = Vec::new();
        for (index, system) in [&mut cold, &mut retained].into_iter().enumerate() {
            assert!(system.try_step_substep(h).unwrap());
            assert!(system.configure_material_contact_reuse(None).is_err());
            let status = readback::read::<u32>(&engine, system.status_buffer(), 1)[0];
            let force =
                readback::read::<f32>(&engine, system.material_force_diagnostics().unwrap(), 1)[0];
            let corrections =
                readback::read::<f32>(&engine, system.material_contact_diagnostics().unwrap(), 3);
            eprintln!(
                "step={step} index={index} reuse={reuse:?} force={force} corrections={corrections:?} status={status}"
            );
            if rejected_step == Some(step) && index == 1 {
                assert_eq!(status, 64);
                assert!(force.is_finite() && force > 1e-4);
                assert!(corrections.iter().all(|v| v.is_finite()));
                assert!(corrections[0] > 1e-4);
                return;
            }
            assert_eq!(
                status, 0,
                "step={step} index={index} reuse={reuse:?} force={force}"
            );
            assert!(force.is_finite() && force <= 1e-4);
            while !system.poll_completion().unwrap() {
                std::thread::yield_now();
            }
            let q = readback::read::<[f32; 4]>(
                &engine,
                system.soft_state().particle_q.device_buffer().unwrap(),
                4,
            );
            let v = readback::read::<[f32; 4]>(
                &engine,
                system.soft_state().particle_qd.device_buffer().unwrap(),
                4,
            );
            let points = q
                .iter()
                .map(|p| Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
                .collect::<Vec<_>>();
            let f = Mat3::from_cols(
                points[1] - points[0],
                points[2] - points[0],
                points[3] - points[0],
            ) / 0.02;
            let j = f.determinant();
            assert!(j > 0.5);
            let mu = 2e6 / 2.9;
            let lambda = 2e6 * 0.45 / (1.45 * 0.1) + mu;
            let alpha = 1.0 + mu / lambda;
            let norm =
                f.x_axis.length_squared() + f.y_axis.length_squared() + f.z_axis.length_squared();
            let elastic = 0.02_f64.powi(3) / 12.0
                * (mu * (norm - 3.0) + lambda * ((j - alpha).powi(2) - (1.0 - alpha).powi(2)));
            let kinetic = v
                .iter()
                .map(|v| 0.5 * mass * v[..3].iter().map(|x| (*x as f64).powi(2)).sum::<f64>())
                .sum::<f64>();
            assert!(
                elastic + kinetic <= previous_energy[index] + 1e-8,
                "reuse={reuse:?} energy={} previous={}",
                elastic + kinetic,
                previous_energy[index]
            );
            previous_energy[index] = elastic + kinetic;
            results.push((q, v));
        }
        for field in 0..2 {
            let a = if field == 0 {
                &results[0].0
            } else {
                &results[0].1
            };
            let b = if field == 0 {
                &results[1].0
            } else {
                &results[1].1
            };
            for (a, b) in a.iter().zip(b) {
                for i in 0..3 {
                    assert!((a[i] - b[i]).abs() < 2e-5);
                }
            }
        }
    }
    assert!(
        rejected_step.is_none(),
        "expected force guard did not reject"
    );
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn contact_residual_guard_rejects_cycles_and_accepts_coulomb_sliding() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for case in 0..5 {
        let contact = |normal: [f32; 3], gap, friction| SoftRigidContactGpu {
            normal_gap: [normal[0], normal[1], normal[2], gap],
            material: [friction, 0.0, 0.0, 0.0],
            ids: [0, u32::MAX, 0, 1],
            support: [0, 0, 1, 0],
            weights: [1.0, 0.0, 0.0, 0.0],
        };
        let rows = if case == 0 {
            vec![
                contact([1.0, 0.0, 0.0], -0.005, 0.0),
                contact([-1.0, 0.0, 0.0], -0.005, 0.0),
            ]
        } else {
            vec![contact(
                [0.0, 1.0, 0.0],
                0.0,
                if case == 3 { 0.5 } else { 1.0 },
            )]
        };
        let contacts = buffer(device, &rows);
        let positions = buffer(device, &[[0.0_f32; 4]]);
        let velocities = buffer(
            device,
            &[[
                if case == 3 { 2.0_f32 } else { 0.0 },
                if case == 0 { 0.0 } else { -1.0 },
                0.0,
                0.0,
            ]],
        );
        let inverse = buffer(device, &[1.0_f32]);
        let poses = buffer(device, &[[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]);
        let com = buffer(device, &[[0.0_f32; 4]]);
        let jacobian = buffer(device, &[0.0_f32; 6]);
        let mass = buffer(device, &[1.0_f32]);
        let rigid = buffer(device, &[0.0_f32]);
        let bodies = buffer(device, &[[0.0_f32; 8]]);
        let bindings = SoftRigidSolveBindings {
            positions: &positions,
            velocities: &velocities,
            inverse_mass: &inverse,
            contacts: &contacts,
            body_poses: &poses,
            body_com: &com,
            jacobians: &jacobian,
            mass_matrix: &mass,
            rigid_velocities: &rigid,
            body_velocities: &bodies,
            particle_count: 1,
            body_count: 1,
        };
        let settings = SoftRigidSolveSettings {
            iterations: 4,
            dt: 0.001,
            penetration_recovery: 0.2,
            max_recovery_speed: 2.0,
        };
        let kernel = SoftRigidSolveKernel::new(device, 1, rows.len() as u32).unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        kernel
            .encode(device, &mut encoder, bindings, settings)
            .unwrap();
        queue.submit([encoder.finish()]);
        let before = readback::read::<[f32; 4]>(&engine, &velocities, 1);
        if case == 1 {
            queue.write_buffer(
                &velocities,
                0,
                bytemuck::cast_slice(&[[0.0_f32, 1.0, 0.0, 0.0]]),
            );
        }
        if case == 2 || case == 4 {
            queue.write_buffer(
                &velocities,
                0,
                bytemuck::cast_slice(&[[if case == 4 { 1e30_f32 } else { 1.0 }, 0.0, 0.0, 0.0]]),
            );
        }
        let mut encoder = device.create_command_encoder(&Default::default());
        if case == 0 {
            kernel
                .encode_continuation(device, &mut encoder, bindings, settings)
                .unwrap();
        }
        kernel
            .encode_residual_check(device, &mut encoder, bindings, settings, 1e-4)
            .unwrap();
        queue.submit([encoder.finish()]);
        let actual = readback::read::<u32>(&engine, kernel.status(), 1)[0];
        assert_eq!(
            actual,
            match case {
                3 => 0,
                4 => 4,
                _ => 64,
            },
            "case {case}"
        );
        let after = readback::read::<[f32; 4]>(&engine, &velocities, 1);
        if case == 0 {
            assert_eq!(
                before, after,
                "cyclic constraint residual despite zero net correction"
            );
        }
        if case == 3 {
            assert!((after[0][0] - 1.5).abs() < 1e-6);
        }
        let residuals = readback::read::<[f32; 8]>(&engine, kernel.row_state(), rows.len());
        if case == 3 {
            assert!(residuals.iter().all(|r| r[3] <= 1e-4));
        } else if case < 3 {
            assert!(residuals.iter().any(|r| r[3] > 0.5));
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn material_contact_residual_kernel_checks_all_channels_and_nonfinite_values() {
    use molla_kernels::soft_rigid_solve::{
        SoftMaterialIterationBindings, SoftMaterialIterationKernel,
    };
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = SoftMaterialIterationKernel::new(device);
    for (channel, value, final_iteration, expected_status) in [
        (0, 0.1_f32, false, 0),
        (0, 0.1, true, 64),
        (1, 0.1, true, 64),
        (2, 0.1, true, 64),
        (0, 1e-6, true, 0),
        (0, f32::NAN, true, 4),
        (1, f32::INFINITY, true, 4),
        (2, f32::NAN, true, 4),
    ] {
        let input = buffer(device, &[[0.0_f32; 4]]);
        let predicted = buffer(device, &[[0.0_f32; 4]]);
        let corrected = buffer(
            device,
            &[[if channel == 0 { value } else { 0.0 }, 0.0, 0.0, 0.0]],
        );
        let before = buffer(device, &[[0.0_f32; 8]]);
        let mut body = [0.0_f32; 8];
        if channel > 0 {
            body[(channel - 1) * 4] = value;
        }
        let after = buffer(device, &[body]);
        let status = buffer(device, &[0_u32]);
        let mut encoder = device.create_command_encoder(&Default::default());
        kernel
            .encode(
                device,
                &mut encoder,
                SoftMaterialIterationBindings {
                    input_velocity: &input,
                    predicted_velocity: &predicted,
                    corrected_velocity: &corrected,
                    previous_body_velocity: &before,
                    corrected_body_velocity: &after,
                    status: &status,
                    particles: 1,
                    bodies: 1,
                },
                1e-4,
                1e-4,
                final_iteration,
            )
            .unwrap();
        queue.submit([encoder.finish()]);
        assert_eq!(
            readback::read::<u32>(&engine, &status, 1),
            [expected_status]
        );
        if value.is_finite() {
            let residual = readback::read::<f32>(&engine, kernel.diagnostics(), 3);
            assert!((residual[channel] - value).abs() < 1e-7);
            for i in 0..3 {
                if i != channel {
                    assert_eq!(residual[i], 0.0);
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn lagged_gpu_contact_impulses_match_constrained_viscous_tet() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (h, viscous) in [
        (1.0 / 19200.0, true),
        (1.0 / 38400.0, true),
        (1.0 / 19200.0, false),
    ] {
        let expected = reference(h, viscous);
        let mut model = model(false, viscous);
        let mut input = model.state().unwrap();
        input.particle_qd.host_mut().unwrap().fill(Vec3::NEG_Y);
        let mut output = model.state().unwrap();
        model.upload_to_device(device, queue).unwrap();
        input.upload_to_device(device, queue).unwrap();
        output.upload_to_device(device, queue).unwrap();
        let mut fem = VbdSolver::new_gpu(&engine, &model)
            .unwrap()
            .with_iterations(128)
            .with_contact_stiffness(0.0);
        let kernel = SoftRigidSolveKernel::new(device, 1, 3).unwrap();
        let contacts = buffer(
            device,
            &[0, 1, 3].map(|particle| SoftRigidContactGpu {
                normal_gap: [0.0, 1.0, 0.0, 0.0],
                material: [1000.0, 0.0, 0.0, 0.0],
                ids: [particle, u32::MAX, 0, 1],
                support: [0, 0, 1, 0],
                weights: [1.0, 0.0, 0.0, 0.0],
            }),
        );
        let pose = buffer(device, &[[0.0_f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]);
        let com = buffer(device, &[[0.0_f32; 4]]);
        let jacobian = buffer(device, &[0.0_f32; 6]);
        let mass = buffer(device, &[1.0_f32]);
        let rigid_velocity = buffer(device, &[0.0_f32]);
        let body_velocity = buffer(device, &[[0.0_f32; 8]]);
        let predictor = buffer(device, &[[0.0_f32; 4]; 4]);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test-only lagged impulse accumulation"),
            source: wgpu::ShaderSource::Wgsl(
                r#"
                @group(0) @binding(0) var<storage, read_write> input: array<vec4<f32>>;
                @group(0) @binding(1) var<storage, read> before: array<vec4<f32>>;
                @group(0) @binding(2) var<storage, read> after: array<vec4<f32>>;
                @compute @workgroup_size(4)
                fn accumulate(@builtin(local_invocation_index) i: u32) {
                    input[i] += after[i] - before[i];
                }
            "#
                .into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("test-only lagged impulse accumulation"),
            layout: None,
            module: &shader,
            entry_point: Some("accumulate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                input.particle_qd.device_buffer().unwrap(),
                &predictor,
                output.particle_qd.device_buffer().unwrap(),
            ]
            .into_iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
        });
        let settings = SoftRigidSolveSettings {
            iterations: 12,
            dt: h as f32,
            penetration_recovery: 0.2,
            max_recovery_speed: 0.5,
        };
        let mut first_error = 0.0;
        for iteration in 0..256 {
            fem.step(
                &model,
                &input,
                &model.control(),
                &Contacts::with_capacity(0),
                h,
                &mut output,
            )
            .unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.copy_buffer_to_buffer(
                output.particle_qd.device_buffer().unwrap(),
                0,
                &predictor,
                0,
                64,
            );
            let bindings = SoftRigidSolveBindings {
                positions: input.particle_q.device_buffer().unwrap(),
                velocities: output.particle_qd.device_buffer().unwrap(),
                inverse_mass: model.particle_inv_mass.device_buffer().unwrap(),
                contacts: &contacts,
                body_poses: &pose,
                body_com: &com,
                jacobians: &jacobian,
                mass_matrix: &mass,
                rigid_velocities: &rigid_velocity,
                body_velocities: &body_velocity,
                particle_count: 4,
                body_count: 1,
            };
            if iteration == 0 {
                kernel
                    .encode(device, &mut encoder, bindings, settings)
                    .unwrap();
            } else {
                kernel
                    .encode_continuation(device, &mut encoder, bindings, settings)
                    .unwrap();
            }
            {
                let mut pass = encoder.begin_compute_pass(&Default::default());
                pass.set_pipeline(&pipeline);
                pass.set_bind_group(0, &group, &[]);
                pass.dispatch_workgroups(1, 1, 1);
            }
            queue.submit([encoder.finish()]);
            input.particle_qd.mark_device_dirty();
            if [0, 7, 63, 255].contains(&iteration) {
                let v = readback::read::<[f32; 4]>(
                    &engine,
                    output.particle_qd.device_buffer().unwrap(),
                    4,
                );
                let actual = Vec3::new(v[2][0] as f64, v[2][1] as f64, v[2][2] as f64);
                let error = (actual - expected).length();
                eprintln!(
                    "coupled tet h={h} viscous={viscous} iteration={} error={error} actual={actual:?} expected={expected:?}",
                    iteration + 1
                );
                if iteration == 0 {
                    first_error = error;
                }
                assert!(v.iter().flatten().all(|x| x.is_finite()));
                for node in [0, 1, 3] {
                    assert!(v[node][..3].iter().all(|x| x.abs() < 1e-5));
                }
                if iteration == 255 {
                    assert!(error < 5e-4 && error < first_error * 0.01);
                }
            }
        }
        assert_eq!(readback::read::<u32>(&engine, kernel.status(), 1), vec![0]);
        let rows = readback::read::<[f32; 8]>(&engine, kernel.row_state(), 3);
        let velocities =
            readback::read::<[f32; 4]>(&engine, output.particle_qd.device_buffer().unwrap(), 4);
        let velocities = velocities
            .iter()
            .map(|v| Vec3::new(v[0] as f64, v[1] as f64, v[2] as f64))
            .collect::<Vec<_>>();
        let impulse = rows
            .iter()
            .map(|r| {
                assert!(r[4] > 0.0 && r[5].hypot(r[6]) < 1000.0 * r[4]);
                Vec3::new(-r[6] as f64, r[4] as f64, -r[5] as f64)
            })
            .sum::<Vec3>();
        let node_mass = 1000.0 * 0.02_f64.powi(3) / 24.0;
        let momentum_change =
            (velocities.iter().copied().sum::<Vec3>() + Vec3::Y * 4.0) * node_mass;
        assert!((momentum_change - impulse).length() < 1e-7);
        let positions = [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z].map(|p| p * 0.02);
        let q = positions
            .iter()
            .zip(&velocities)
            .map(|(q, v)| *q + h * *v)
            .collect::<Vec<_>>();
        let f = Mat3::from_cols(q[1] - q[0], q[2] - q[0], q[3] - q[0]) / 0.02;
        let j = f.determinant();
        assert!(j > 0.5);
        let mu = 2e6 / 2.9;
        let lambda = 2e6 * 0.45 / (1.45 * 0.1) + mu;
        let alpha = 1.0 + mu / lambda;
        let squared =
            f.x_axis.length_squared() + f.y_axis.length_squared() + f.z_axis.length_squared();
        let elastic = 0.02_f64.powi(3) / 12.0
            * (mu * (squared - 3.0) + lambda * ((j - alpha).powi(2) - (1.0 - alpha).powi(2)));
        let kinetic = velocities
            .iter()
            .map(|v| 0.5 * node_mass * v.length_squared())
            .sum::<f64>();
        assert!(elastic + kinetic <= 2.0 * node_mass + 1e-9);
    }
}
