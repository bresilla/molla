#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_kernels::soft_rigid_solve::SoftMaterialIterationKernel;
use molla_math::{Mat3, Quat, Vec3};
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
        assert!(solver.material_force_samples_gpu().is_none());
        assert_eq!(solver.spring_stiffness_gpu().unwrap().size(), 4);
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
        assert!(solver.material_force_samples_gpu().is_none());
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
            assert_eq!(
                actual,
                readback::read::<[f32; 4]>(
                    &engine,
                    solver.material_force_samples_gpu().unwrap(),
                    4
                )
            );
            assert_eq!(&actual[2][..3], &[0.0, -(h as f32), 0.0]);
            let after = snapshot(&engine, &input, &output);
            assert_eq!(before, after);
            for i in [0, 1, 3] {
                assert_eq!(actual[i], [0.0; 4]);
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
            assert!(solver.material_force_samples_gpu().is_none());
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
        assert!(solver.material_force_samples_gpu().is_none());
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
        assert!(solver.material_force_samples_gpu().is_none());
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
fn diagnostic_stiffness_matches_projected_spring_table() {
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    builder.add_particle(Vec3::ZERO, 1.0, 0.0, WorldId(0));
    builder.add_particle(Vec3::X, 1.0, 0.0, WorldId(0));
    builder.add_distance_constraint(0, 1, 1.0, 1e-9, 1.0);
    builder.add_tension_element(0, 1, 1.0, 2e8).unwrap();
    builder.add_distance_constraint(0, 1, 1.0, 2.0, 1.0);
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let solver = VbdSolver::new_gpu(&engine, &model).unwrap();
    assert_eq!(
        readback::read::<f32>(&engine, solver.spring_stiffness_gpu().unwrap(), 3),
        vec![1e7, 2e8, 1.0]
    );
    assert!(solver.material_force_samples_gpu().is_none());
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn rotated_cord_prestress_matches_balanced_f64_force() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let h = (1.0 / 19200.0) as f32 as f64;
    let decode = |bits: [u32; 3]| Vec3::from_array(bits.map(|b| f32::from_bits(b) as f64));
    let project = |v: Vec3| Vec3::from_array(v.to_array().map(|x| x as f32 as f64));
    let q = [
        decode([3200589562, 1027526611, 1009447054]),
        decode([3200589597, 1027518020, 3145421470]),
        decode([3200589391, 1027511982, 3163834364]),
    ];
    let velocity = [
        decode([3014370048, 919367983, 3036084566]),
        decode([3022099684, 915001823, 3035613440]),
        decode([881136076, 907789975, 3034190500]),
    ]
    .map(|d| project(d / h));
    let delta = velocity.map(|v| project(v * h));
    let inverse_mass = [1141663981, 1141343957, 1141346677].map(|bits| f32::from_bits(bits) as f64);
    let stiffness = 821926.125;
    for tension in [false, true] {
        for rest in [0.014194300398230553_f64, 0.0144_f32 as f64] {
            let mut builder = ModelBuilder::new();
            builder.set_world_gravity(WorldId(0), Vec3::ZERO);
            builder.set_linear_damping(0.0);
            for i in 0..3 {
                builder.add_particle(q[i], 1.0 / inverse_mass[i], 0.0, WorldId(0));
            }
            for (a, b) in [(0, 1), (1, 2)] {
                if tension {
                    builder.add_tension_element(a, b, rest, stiffness).unwrap();
                } else {
                    builder.add_distance_constraint(a, b, rest, 1.0 / stiffness, 1.0);
                }
            }
            let mut model = builder.build().unwrap();
            let mut input = model.state().unwrap();
            let mut output = model.state().unwrap();
            model.upload_to_device(device, queue).unwrap();
            let mut solver = VbdSolver::new_gpu(&engine, &model).unwrap();
            let stiffness =
                readback::read::<f32>(&engine, solver.spring_stiffness_gpu().unwrap(), 2);
            let mut forces = [Vec3::ZERO; 3];
            for (index, (a, b)) in [(0, 1), (1, 2)].into_iter().enumerate() {
                let edge = q[a] - q[b] + delta[a] - delta[b];
                let length = edge.length();
                if !tension || length > rest {
                    let force = -stiffness[index] as f64 * ((length - rest) / length) * edge;
                    forces[a] += force;
                    forces[b] -= force;
                }
            }
            let incoming: Vec<_> = (0..3)
                .map(|i| project(velocity[i] - forces[i] * (h * inverse_mass[i]) + Vec3::X * 7e-5))
                .collect();
            input
                .particle_qd
                .host_mut()
                .unwrap()
                .copy_from_slice(&incoming);
            output
                .particle_qd
                .host_mut()
                .unwrap()
                .copy_from_slice(&velocity);
            input.upload_to_device(device, queue).unwrap();
            output.upload_to_device(device, queue).unwrap();
            let residual = solver
                .material_force_residual_gpu(&model, &input, &mut output, h)
                .unwrap();
            let samples = readback::read::<[f32; 4]>(&engine, residual, 3);
            for i in 0..3 {
                let target = project(incoming[i] * h);
                let expected =
                    ((target - delta[i]) / h + forces[i] * (h * inverse_mass[i])).length();
                let error = (samples[i][3] as f64 - expected).abs();
                eprintln!(
                    "cord tension={tension} rest={rest} node={i} gpu={} oracle={expected} error={error}",
                    samples[i][3]
                );
                assert!(error < 3e-6, "cord force precision error {error}");
                assert_eq!(samples[i][3] > 1e-4, expected > 1e-4);
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn spring_force_resolves_sub_position_ulp_extension() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let h = (1.0 / 19200.0) as f32;
    for tension in [false, true] {
        for rest in [0.02_f32, 0.019, 0.021] {
            let mut builder = ModelBuilder::new();
            builder.set_world_gravity(WorldId(0), Vec3::ZERO);
            builder.set_linear_damping(0.0);
            builder.add_particle(Vec3::ZERO, 0.001, 0.0, WorldId(0));
            builder.add_particle(Vec3::X * 0.02_f32 as f64, 0.001, 0.0, WorldId(0));
            if tension {
                builder.add_tension_element(0, 1, rest as f64, 2e8).unwrap();
            } else {
                builder.add_distance_constraint(0, 1, rest as f64, 1e-7, 1.0);
            }
            let mut model = builder.build().unwrap();
            let mut input = model.state().unwrap();
            let mut output = model.state().unwrap();
            model.upload_to_device(device, queue).unwrap();
            input.upload_to_device(device, queue).unwrap();
            output.upload_to_device(device, queue).unwrap();
            let mut solver = VbdSolver::new_gpu(&engine, &model).unwrap();
            let stiffness =
                readback::read::<f32>(&engine, solver.spring_stiffness_gpu().unwrap(), 1)[0] as f64;
            for velocity in [-1e-6_f32, 1e-6, 0.0, -0.1, 0.1, -192.0, -384.0, -768.0] {
                queue.write_buffer(
                    output.particle_qd.device_buffer().unwrap(),
                    16,
                    bytemuck::cast_slice(&[velocity, 0.0, 0.0, 0.0]),
                );
                let residual = solver
                    .material_force_residual_gpu(&model, &input, &mut output, h as f64)
                    .unwrap();
                let actual = readback::read::<[f32; 4]>(&engine, residual, 2);
                let delta = (h * velocity) as f64;
                let edge = 0.02_f32 as f64 + delta;
                let length = edge.abs().max(1e-8);
                let force = if tension && length <= rest as f64 {
                    0.0
                } else {
                    -stiffness * ((length - rest as f64) / length) * edge
                };
                let inertial = -delta / (1000.0 * (h as f64).powi(2));
                let expected = (inertial + force).abs() * h as f64 * 1000.0;
                let other = force.abs() * h as f64 * 1000.0;
                for (node, reference) in [(0, other), (1, expected)] {
                    let error = (actual[node][3] as f64 - reference).abs();
                    assert!(
                        error < 1e-5 + 3e-6 * reference,
                        "tension={tension} rest={rest} v={velocity} node={node}: gpu={} expected={reference}",
                        actual[node][3]
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn translated_thin_tet_force_matches_projected_geometry() {
    let engine = Engine::new_gpu().unwrap();
    let h = 1.0 / 19200.0;
    let rest = [
        Vec3::ZERO,
        Vec3::X * 0.02,
        Vec3::Y * 0.0005,
        Vec3::Z * 0.005,
    ];
    for offset in [
        Vec3::ZERO,
        Vec3::new(1.03, 0.87, -0.63),
        Vec3::new(-0.83, -1.07, 0.79),
    ] {
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        builder.set_linear_damping(0.0);
        for p in rest {
            builder.add_particle(p, 1000.0 * 0.02 * 0.0005 * 0.005 / 24.0, 0.0, WorldId(0));
        }
        builder
            .add_tet_mesh_per_material(
                &[[0, 1, 2, 3]],
                &[MaterialNeoHookean::elastic(2e6, 0.45)],
                &[0],
                None,
            )
            .unwrap();
        let mut model = builder.build().unwrap();
        let mut input = model.state().unwrap();
        for p in input.particle_q.host_mut().unwrap() {
            *p += offset;
        }
        let mut output = model.state().unwrap();
        let project = |v: f64| v as f32 as f64;
        let q: Vec<_> = input
            .particle_q
            .host()
            .unwrap()
            .iter()
            .map(|v| Vec3::from_array(v.to_array().map(project)))
            .collect();
        let rest = rest.map(|v| Vec3::from_array(v.to_array().map(project)));
        let inverse = Mat3::from_cols_array(
            &model.tet_dm_inv.host().unwrap()[0]
                .to_cols_array()
                .map(project),
        );
        let edge = |i: usize| (q[i] - q[0]) - (rest[i] - rest[0]);
        let f = Mat3::IDENTITY + Mat3::from_cols(edge(1), edge(2), edge(3)) * inverse;
        let cof = Mat3::from_cols(
            f.y_axis.cross(f.z_axis),
            f.z_axis.cross(f.x_axis),
            f.x_axis.cross(f.y_axis),
        );
        let mu = project(model.tet_mu.host().unwrap()[0]);
        let lam = project(model.tet_lambda.host().unwrap()[0]);
        let stress = (f - cof) * mu + cof * ((lam + mu) * (f.determinant() - 1.0));
        let volume = project(model.tet_rest_volume.host().unwrap()[0]);
        let inverse_mass = project(model.particle_inv_mass.host().unwrap()[0]);
        let rows = inverse.transpose();
        let gradients = [
            -(rows.x_axis + rows.y_axis + rows.z_axis),
            rows.x_axis,
            rows.y_axis,
            rows.z_axis,
        ];
        let expected =
            gradients.map(|g| (stress * g * volume).length() * project(h) * inverse_mass);
        model
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        input
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        output
            .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        let mut solver = VbdSolver::new_gpu(&engine, &model)
            .unwrap()
            .with_contact_stiffness(0.0);
        let residual = solver
            .material_force_residual_gpu(&model, &input, &mut output, h)
            .unwrap();
        let actual = readback::read::<[f32; 4]>(&engine, residual, 4);
        for i in 0..4 {
            let error = (actual[i][3] as f64 - expected[i]).abs();
            eprintln!(
                "offset={offset:?} node={i} GPU={} oracle={} error={error}",
                actual[i][3], expected[i]
            );
            assert!(error < 5e-5 + expected[i] * 1e-5);
        }
    }
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn rotated_thin_tet_increment_matches_projected_force() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let project = |x: f64| x as f32 as f64;
    let vector = |v: Vec3| Vec3::from_array(v.to_array().map(project));
    let h = project(1.0 / 19200.0);
    for viscous in [false, true] {
        for (axis, angle, plastic_x) in [
            (Vec3::X, 1.4, 1.0),
            (Vec3::X, 2.1, 1.0),
            (Vec3::Y, 2.1, 1.0),
            (Vec3::Z, 2.1, 1.0),
            (Vec3::ONE.normalize(), 2.4, 1.0),
            (Vec3::ONE.normalize(), 2.4, 0.5),
            (Vec3::Y, 2.1, 2.0),
        ] {
            let mut builder = ModelBuilder::new();
            builder.set_world_gravity(WorldId(0), Vec3::ZERO);
            builder.set_linear_damping(0.0);
            let points = [
                Vec3::ZERO,
                Vec3::X * 0.02,
                Vec3::new(0.01, 0.0005, 0.0),
                Vec3::new(0.01, 0.0, 0.005),
            ];
            for p in points {
                builder.add_particle(p, 0.00005, 0.0, WorldId(0));
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
            let mut model = builder.build().unwrap();
            let mut input = model.state().unwrap();
            let mut output = model.state().unwrap();
            let rotation = Quat::from_axis_angle(axis, angle);
            let plastic = Mat3::from_diagonal(Vec3::new(plastic_x, 1.0, 1.0));
            output.tet_f_plastic.host_mut().unwrap()[0] = plastic;
            for (i, p) in input.particle_q.host_mut().unwrap().iter_mut().enumerate() {
                *p = rotation * (plastic * *p) + Vec3::new(0.5, 0.15, 0.52);
                output.particle_qd.host_mut().unwrap()[i] = Vec3::new(
                    0.08 + i as f64 * 0.0007,
                    -0.05 + i as f64 * 0.0003,
                    0.04 - i as f64 * 0.0004,
                );
            }
            let q: Vec<_> = input
                .particle_q
                .host()
                .unwrap()
                .iter()
                .copied()
                .map(vector)
                .collect();
            let rest = points.map(vector);
            let delta: Vec<_> = output
                .particle_qd
                .host()
                .unwrap()
                .iter()
                .map(|v| vector(vector(*v) * h))
                .collect();
            let inverse = Mat3::from_cols_array(
                &model.tet_dm_inv.host().unwrap()[0]
                    .to_cols_array()
                    .map(project),
            );
            let edge = |i: usize| (q[i] - q[0]) - (rest[i] - rest[0]);
            let previous = Mat3::IDENTITY + Mat3::from_cols(edge(1), edge(2), edge(3)) * inverse;
            let increment = Mat3::from_cols(
                delta[1] - delta[0],
                delta[2] - delta[0],
                delta[3] - delta[0],
            ) * inverse;
            let total_f = previous + increment;
            let f = total_f * plastic.inverse();
            let cof = Mat3::from_cols(
                f.y_axis.cross(f.z_axis),
                f.z_axis.cross(f.x_axis),
                f.x_axis.cross(f.y_axis),
            );
            let mu = project(model.tet_mu.host().unwrap()[0]);
            let lam = project(model.tet_lambda.host().unwrap()[0]);
            let mut stress = (f - cof) * mu + cof * ((lam + mu) * (f.determinant() - 1.0));
            if viscous {
                let strain = (previous.transpose() * increment
                    + increment.transpose() * previous
                    + increment.transpose() * increment)
                    * 0.5;
                let trace = strain.x_axis.x + strain.y_axis.y + strain.z_axis.z;
                let rate_stress = (strain - Mat3::IDENTITY * (trace / 3.0))
                    * (2.0 * project(material.viscosity.shear))
                    + Mat3::IDENTITY * (project(material.viscosity.bulk) * trace);
                stress += total_f * rate_stress * (1.0 / h);
            }
            let rows = inverse.transpose();
            let gradients = [
                -(rows.x_axis + rows.y_axis + rows.z_axis),
                rows.x_axis,
                rows.y_axis,
                rows.z_axis,
            ];
            let volume = project(model.tet_rest_volume.host().unwrap()[0]);
            let inv_mass = project(model.particle_inv_mass.host().unwrap()[0]);
            model.upload_to_device(device, queue).unwrap();
            input.upload_to_device(device, queue).unwrap();
            output.upload_to_device(device, queue).unwrap();
            let mut solver = VbdSolver::new_gpu(&engine, &model).unwrap();
            let residual = solver
                .material_force_residual_gpu(&model, &input, &mut output, h)
                .unwrap();
            let actual = readback::read::<[f32; 4]>(&engine, residual, 4);
            for i in 0..4 {
                let force = -delta[i] / (inv_mass * h * h) - stress * gradients[i] * volume;
                let expected = force.length() * h * inv_mass;
                let error = (actual[i][3] as f64 - expected).abs();
                eprintln!(
                    "rotated increment viscous={viscous} axis={axis:?} angle={angle} plastic_x={plastic_x} node={i} error={error}"
                );
                assert!(error < 3e-5, "GPU={} oracle={expected}", actual[i][3]);
            }
        }
    }
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
        assert!(solver.material_force_samples_gpu().is_some());
        assert!(
            solver
                .material_iteration_gpu(&model, &input, &mut output, h, 1, false)
                .is_err()
        );
        assert!(solver.material_force_samples_gpu().is_none());
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
        assert!(system.material_force_samples().is_none());
        assert!(system.material_spring_stiffness().is_none());
        system
            .configure_material_contact(Some(MaterialContactConfig {
                iterations: 2,
                linear_tolerance: 1e-4,
                angular_tolerance: 1e-4,
            }))
            .unwrap();
        assert!(system.material_force_samples().is_none());
        assert_eq!(system.material_spring_stiffness().unwrap().size(), 4);
        system.step(h).unwrap();
        let status = readback::read::<u32>(&engine, system.status_buffer(), 1)[0];
        let corrections =
            readback::read::<f32>(&engine, system.material_contact_diagnostics().unwrap(), 3);
        let force =
            readback::read::<f32>(&engine, system.material_force_diagnostics().unwrap(), 1)[0];
        let samples =
            readback::read::<[f32; 4]>(&engine, system.material_force_samples().unwrap(), 4);
        assert_eq!(force, samples.iter().map(|s| s[3]).fold(0.0_f32, f32::max));
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
