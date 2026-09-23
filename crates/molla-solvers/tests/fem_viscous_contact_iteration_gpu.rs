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
