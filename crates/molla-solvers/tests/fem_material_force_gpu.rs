#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_kernels::soft_rigid_solve::SoftMaterialIterationKernel;
use molla_math::Vec3;
use molla_sim::{Contacts, MaterialNeoHookean, Model, ModelBuilder, TetViscosity};
use molla_solvers::{Solver, VbdSolver};
use wgpu::util::DeviceExt;

#[path = "common/fem_rotor.rs"]
#[allow(dead_code)]
mod fem_rotor;
#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

fn model(pinned: bool, viscous: bool) -> Model {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), Vec3::ZERO);
    builder.set_linear_damping(0.0);
    for (i, p) in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z]
        .into_iter()
        .enumerate()
    {
        builder.add_particle(
            p * 0.02,
            if pinned && i != 2 {
                0.0
            } else {
                1000.0 * 0.02_f64.powi(3) / 24.0
            },
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
    builder.build().unwrap()
}

fn axial_residual(h: f64, previous: f64, velocity: f64, viscous: bool) -> f64 {
    let mu = 2e6 / (2.0 * 1.45);
    let lambda = 2e6 * 0.45 / (1.45 * 0.1);
    let increment = h * velocity / 0.02;
    let current = previous + increment;
    let elastic_stress = (lambda + 2.0 * mu) * (current - 1.0);
    let viscous_stress = if viscous {
        current
            * (13333.333333333336 + 4.0 * 1379.3103448275863 / 3.0)
            * (previous * increment + 0.5 * increment * increment)
            / h
    } else {
        0.0
    };
    let force = -0.02_f64.powi(3) / 6.0 / 0.02 * (elastic_stress + viscous_stress);
    (-1.0 - velocity + h * force / (1000.0 * 0.02_f64.powi(3) / 24.0)).abs()
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn material_force_matches_axial_oracle_without_mutating_state() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (h, previous_y, viscous) in [
        (1.0 / 19200.0, 0.02, true),
        (1.0 / 38400.0, 0.021, true),
        (1.0 / 19200.0, 0.02, false),
    ] {
        let mut model = model(true, viscous);
        let mut input = model.state().unwrap();
        input.particle_q.host_mut().unwrap()[2].y = previous_y;
        input.particle_qd.host_mut().unwrap()[2] = Vec3::NEG_Y;
        let mut output = model.state().unwrap();
        model.upload_to_device(device, queue).unwrap();
        input.upload_to_device(device, queue).unwrap();
        output.upload_to_device(device, queue).unwrap();
        let mut solver = VbdSolver::new_gpu(&engine, &model)
            .unwrap()
            .with_iterations(128)
            .with_contact_stiffness(0.0);
        solver
            .step(
                &model,
                &input,
                &model.control(),
                &Contacts::with_capacity(0),
                h,
                &mut output,
            )
            .unwrap();
        let velocities =
            readback::read::<[f32; 4]>(&engine, output.particle_qd.device_buffer().unwrap(), 4);
        for velocity in [-1.0_f32, 0.0, 0.2, velocities[2][1]] {
            let mut candidate = velocities.clone();
            candidate[2] = [0.0, velocity, 0.0, 0.0];
            queue.write_buffer(
                output.particle_qd.device_buffer().unwrap(),
                0,
                bytemuck::cast_slice(&candidate),
            );
            let before = snapshot(&engine, &input, &output);
            let residual = solver
                .material_force_residual_gpu(&model, &input, &mut output, h)
                .unwrap();
            let actual = readback::read::<[f32; 4]>(&engine, residual, 4);
            let after = snapshot(&engine, &input, &output);
            assert_eq!(before, after);
            for i in [0, 1, 3] {
                assert_eq!(actual[i][3], 0.0);
            }
            let gpu_previous = (previous_y as f32 as f64 - 0.02_f32 as f64) * 50.0 + 1.0;
            let expected = axial_residual(h as f32 as f64, gpu_previous, velocity as f64, viscous);
            assert!(
                (actual[2][3] as f64 - expected).abs() < 3e-4 + 3e-5 * expected,
                "h={h} previous={gpu_previous} v={velocity} viscous={viscous}: GPU={} oracle={expected}",
                actual[2][3]
            );
            if velocity == velocities[2][1] {
                assert!(
                    actual[2][3] < 1e-4,
                    "converged force residual={}",
                    actual[2][3]
                );
            }
        }
        for velocity in [f32::NAN, f32::INFINITY, f32::MAX] {
            queue.write_buffer(
                output.particle_qd.device_buffer().unwrap(),
                32,
                bytemuck::cast_slice(&[0.0, velocity, 0.0, 0.0]),
            );
            let residual = solver
                .material_force_residual_gpu(&model, &input, &mut output, h)
                .unwrap();
            assert!(!readback::read::<[f32; 4]>(&engine, residual, 4)[2][3].is_finite());
        }
        for dt in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                solver
                    .material_force_residual_gpu(&model, &input, &mut output, dt)
                    .is_err()
            );
        }
        solver
            .step(
                &model,
                &input,
                &model.control(),
                &Contacts::with_capacity(0),
                h,
                &mut output,
            )
            .unwrap();
        assert_eq!(
            velocities,
            readback::read::<[f32; 4]>(&engine, output.particle_qd.device_buffer().unwrap(), 4)
        );
        queue.write_buffer(
            output.tet_f_plastic.device_buffer().unwrap(),
            0,
            bytemuck::cast_slice(&[f32::NAN]),
        );
        let residual = solver
            .material_force_residual_gpu(&model, &input, &mut output, h)
            .unwrap();
        assert!(!readback::read::<[f32; 4]>(&engine, residual, 4)[2][3].is_finite());
        model.linear_damping = 1.0;
        assert!(
            solver
                .material_force_residual_gpu(&model, &input, &mut output, h)
                .is_err()
        );
    }
}

fn snapshot(engine: &Engine, input: &molla_sim::State, output: &molla_sim::State) -> Vec<Vec<u32>> {
    [input, output]
        .into_iter()
        .flat_map(|s| {
            [
                s.particle_q.device_buffer().unwrap(),
                s.particle_qd.device_buffer().unwrap(),
                s.tet_f_plastic.device_buffer().unwrap(),
                s.tet_plastic_strain.device_buffer().unwrap(),
                s.tet_back_stress.device_buffer().unwrap(),
            ]
        })
        .map(|b| readback::read::<u32>(engine, b, b.size() as usize / 4))
        .collect()
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn retained_material_iterates_match_uninterrupted_sweeps_and_enforce_lifecycle() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let h = 1.0 / 19200.0;
    for (pinned, viscous) in [(false, true), (true, true), (false, false)] {
        let mut model = model(pinned, viscous);
        let mut input = model.state().unwrap();
        for (i, v) in input.particle_qd.host_mut().unwrap().iter_mut().enumerate() {
            *v = Vec3::new(0.1 * i as f64, -0.5 * i as f64, 0.0);
        }
        let mut reference = model.state().unwrap();
        let mut output = model.state().unwrap();
        let mut other = model.state().unwrap();
        model.upload_to_device(device, queue).unwrap();
        for state in [&mut input, &mut reference, &mut output, &mut other] {
            state.upload_to_device(device, queue).unwrap();
        }
        let mut solver = VbdSolver::new_gpu(&engine, &model)
            .unwrap()
            .with_iterations(128)
            .with_contact_stiffness(0.0);
        solver
            .step(
                &model,
                &input,
                &model.control(),
                &Contacts::with_capacity(0),
                h,
                &mut reference,
            )
            .unwrap();
        assert!(
            solver
                .material_iteration_gpu(&model, &input, &mut output, h, 1, false)
                .is_err()
        );
        solver
            .material_iteration_gpu(&model, &input, &mut output, h, 17, true)
            .unwrap();
        solver
            .material_iteration_gpu(&model, &input, &mut output, h, 111, false)
            .unwrap();
        assert_eq!(
            snapshot(&engine, &input, &reference),
            snapshot(&engine, &input, &output)
        );
        for invalid in [0.0, h * 0.5, f64::NAN] {
            solver
                .material_iteration_gpu(&model, &input, &mut output, h, 1, true)
                .unwrap();
            assert!(
                solver
                    .material_iteration_gpu(&model, &input, &mut output, invalid, 1, false)
                    .is_err()
            );
            assert!(
                solver
                    .material_iteration_gpu(&model, &input, &mut output, h, 1, false)
                    .is_err()
            );
        }
        solver
            .material_iteration_gpu(&model, &input, &mut output, h, 1, true)
            .unwrap();
        let before = snapshot(&engine, &input, &other);
        assert!(
            solver
                .material_iteration_gpu(&model, &input, &mut other, h, 1, false)
                .is_err()
        );
        assert_eq!(before, snapshot(&engine, &input, &other));
        solver
            .material_iteration_gpu(&model, &input, &mut output, h, 1, true)
            .unwrap();
        assert!(
            solver
                .material_iteration_gpu(&model, &other, &mut output, h, 1, false)
                .is_err()
        );
        solver
            .material_iteration_gpu(&model, &input, &mut output, h, 1, true)
            .unwrap();
        solver
            .material_force_residual_gpu(&model, &input, &mut output, h)
            .unwrap();
        assert!(
            solver
                .material_iteration_gpu(&model, &input, &mut output, h, 1, false)
                .is_err()
        );
        solver
            .material_iteration_gpu(&model, &input, &mut output, h, 1, true)
            .unwrap();
        solver
            .step(
                &model,
                &input,
                &model.control(),
                &Contacts::with_capacity(0),
                h,
                &mut output,
            )
            .unwrap();
        assert!(
            solver
                .material_iteration_gpu(&model, &input, &mut output, h, 1, false)
                .is_err()
        );
        solver
            .material_iteration_gpu(&model, &input, &mut output, h, 128, true)
            .unwrap();
        let target = [[0.0_f32, -0.75, 0.0, 0.0]; 4];
        queue.write_buffer(
            input.particle_qd.device_buffer().unwrap(),
            0,
            bytemuck::cast_slice(&target),
        );
        solver
            .material_iteration_gpu(&model, &input, &mut output, h, 128, false)
            .unwrap();
        let residual = solver
            .material_force_residual_gpu(&model, &input, &mut output, h)
            .unwrap();
        let force = readback::read::<[f32; 4]>(&engine, residual, 4);
        assert!(
            force.iter().all(|v| v[3].is_finite() && v[3] <= 1e-4),
            "{force:?}"
        );
        solver
            .step(
                &model,
                &input,
                &model.control(),
                &Contacts::with_capacity(0),
                h,
                &mut reference,
            )
            .unwrap();
        let cold =
            readback::read::<[f32; 4]>(&engine, reference.particle_qd.device_buffer().unwrap(), 4);
        let warm =
            readback::read::<[f32; 4]>(&engine, output.particle_qd.device_buffer().unwrap(), 4);
        for (a, b) in cold.iter().zip(&warm) {
            for i in 0..3 {
                assert!((a[i] - b[i]).abs() < 2e-5);
            }
        }
        assert!(
            solver
                .material_iteration_gpu(&model, &input, &mut output, h, 0, true)
                .is_err()
        );
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn material_force_monitor_rejects_exhaustion_and_nonfinite_values() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = SoftMaterialIterationKernel::new(device);
    let buffer = |bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    };
    for (value, expected) in [
        (0.0_f32, 0),
        (1e-4, 0),
        (0.01, 64),
        (-1.0, 4),
        (f32::NAN, 4),
        (f32::INFINITY, 4),
    ] {
        let mut data = [[0.0_f32; 4]; 65];
        data[64][3] = value;
        let values = buffer(bytemuck::cast_slice(&data));
        let status = buffer(bytemuck::cast_slice(&[2_u32]));
        let mut encoder = device.create_command_encoder(&Default::default());
        kernel
            .encode_force_check(device, &mut encoder, &values, &status, 65, 1e-4)
            .unwrap();
        queue.submit([encoder.finish()]);
        assert_eq!(readback::read::<u32>(&engine, &status, 1)[0], expected | 2);
        let maximum = readback::read::<f32>(&engine, kernel.force_diagnostics(), 1)[0];
        assert_eq!(maximum, if expected == 4 { 0.0 } else { value });
        for (count, tolerance) in [
            (0, 1e-4),
            (66, 1e-4),
            (65, 0.0),
            (65, f32::NAN),
            (65, f32::from_bits(1)),
        ] {
            let mut encoder = device.create_command_encoder(&Default::default());
            assert!(
                kernel
                    .encode_force_check(device, &mut encoder, &values, &status, count, tolerance)
                    .is_err()
            );
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn runtime_rejects_internal_force_error_with_zero_contact_correction() {
    use molla_solvers::fem_rigid::FemRigidConfig;
    use molla_solvers::fem_rigid_gpu::{FemRigidGpuSystem, MaterialContactConfig};
    let engine = std::sync::Arc::new(Engine::new_gpu().unwrap());
    let h = 1.0 / 19200.0;
    for inner in [1, 256] {
        let mut scene = fem_rotor::scene();
        scene.soft_model = model(true, true);
        scene.soft_state = scene.soft_model.state().unwrap();
        scene.soft_state.particle_qd.host_mut().unwrap()[2] = Vec3::NEG_Y;
        scene.shapes.clear();
        let mut system = FemRigidGpuSystem::new(
            engine.clone(),
            scene,
            FemRigidConfig {
                max_substep: h,
                elastic_iterations: inner,
                ..Default::default()
            },
        )
        .unwrap();
        system
            .configure_material_contact(Some(MaterialContactConfig {
                iterations: 2,
                linear_tolerance: 1e-4,
                angular_tolerance: 1e-4,
            }))
            .unwrap();
        system.step(h).unwrap();
        let status = readback::read::<u32>(&engine, system.status_buffer(), 1)[0];
        let corrections =
            readback::read::<f32>(&engine, system.material_contact_diagnostics().unwrap(), 3);
        let force =
            readback::read::<f32>(&engine, system.material_force_diagnostics().unwrap(), 1)[0];
        assert_eq!(corrections, [0.0; 3]);
        eprintln!("inner={inner} force={force} status={status}");
        if inner == 1 {
            assert_eq!(status, 64);
            assert!(force > 1e-4);
        } else {
            assert_eq!(status, 0);
            assert!(force <= 1e-4);
        }
    }
}
