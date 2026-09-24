#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::WorldId;
use molla_math::{Mat3, Vec3};
use molla_sim::{ModelBuilder, TetViscosity};
use molla_solvers::VbdSolver;
use wgpu::util::DeviceExt;

#[path = "common/fem_gradient_fixtures.rs"]
mod fixtures;
#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

fn scalar(bits: u32) -> f64 {
    f32::from_bits(bits) as f64
}

fn vectors(bits: &[u32]) -> Vec<Vec3> {
    bits.chunks_exact(4)
        .map(|v| Vec3::new(scalar(v[0]), scalar(v[1]), scalar(v[2])))
        .collect()
}

fn projected(v: Vec3) -> Vec3 {
    Vec3::from_array(v.to_array().map(|x| x as f32 as f64))
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn tet_sum_corrections_match_exact_rounding() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let source = include_str!("../../molla-kernels/shaders/vbd_cloth.wgsl");
    let start = source.find("fn tet_sum_error(").unwrap();
    let end = source[start..].find("fn tet_gradient_product(").unwrap() + start;
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("tet sum correction regression"),
        source: wgpu::ShaderSource::Wgsl(
            format!(
                r#"
{}
@group(0) @binding(0) var<storage, read> inputs: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> outputs: array<vec4<f32>>;
@compute @workgroup_size(64)
fn check(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if gid.x >= arrayLength(&inputs) {{ return; }}
    let pair = inputs[gid.x];
    let a = vec3<f32>(pair.x, pair.y, pair.x);
    let b = vec3<f32>(pair.y, pair.x, -pair.y);
    let sum = a + b;
    outputs[gid.x] = vec4<f32>(tet_sum_error(a, b, sum), sum.x);
}}
"#,
                &source[start..end]
            )
            .into(),
        ),
    });
    let mut pairs = vec![[0.0f32, 0.0, 0.0, 0.0], [1.0, -1.0, 0.0, 0.0]];
    let mut rng = 0x1935a73du32;
    for exponent in [47u32, 80, 100, 127, 160, 207] {
        for gap in (0..=27).chain([32, 60, 100]) {
            if gap >= exponent - 24 {
                continue;
            }
            for sign in 0..4u32 {
                for _ in 0..32 {
                    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                    let a =
                        f32::from_bits((exponent << 23) | (rng & 0x7fffff) | ((sign & 1) << 31));
                    rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                    let b = f32::from_bits(
                        ((exponent - gap) << 23) | (rng & 0x7fffff) | ((sign >> 1) << 31),
                    );
                    pairs.push([a, b, 0.0, 0.0]);
                }
            }
        }
    }
    let inputs = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&pairs),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let outputs = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: inputs.size(),
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("check"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: inputs.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: outputs.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((pairs.len() as u32).div_ceil(64), 1, 1);
    }
    queue.submit([encoder.finish()]);
    let values = readback::read::<[f32; 4]>(&engine, &outputs, pairs.len());
    for (index, (pair, gpu)) in pairs.iter().zip(values).enumerate() {
        assert_eq!(gpu[3], pair[0] + pair[1], "sum at {index}");
        for (lane, (a, b)) in [(pair[0], pair[1]), (pair[1], pair[0]), (pair[0], -pair[1])]
            .into_iter()
            .enumerate()
        {
            let sum = (a + b) as f64;
            let recovered_b = sum - a as f64;
            let expected = ((a as f64 - (sum - recovered_b)) + (b as f64 - recovered_b)) as f32;
            assert_eq!(gpu[lane], expected, "correction at {index}/{lane}: {a}+{b}");
        }
    }
    eprintln!("exact sum corrections: {} lanes", pairs.len() * 3);
}

#[test]
#[ignore = "requires a real GPU through make test-gpu"]
fn cancelling_tet_gradients_match_f64_force() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (case, fixture) in fixtures::CASES
        .iter()
        .chain(fixtures::EDGE_CASES)
        .chain(fixtures::INCREMENT_CASES)
        .enumerate()
    {
        let rest = vectors(fixture.rest);
        let q = vectors(fixture.positions);
        let velocity = vectors(fixture.velocities);
        let target = vectors(fixture.targets);
        let inverse_mass: Vec<_> = fixture.inverse_mass.iter().map(|v| scalar(*v)).collect();
        let tets: Vec<[u32; 4]> = fixture
            .tets
            .chunks_exact(4)
            .map(|v| v.try_into().unwrap())
            .collect();
        let columns = vectors(fixture.inverse_rest);
        let inverse: Vec<_> = columns
            .chunks_exact(3)
            .map(|c| Mat3::from_cols(c[0], c[1], c[2]))
            .collect();
        let h = fixture.dt as f64;
        let mut builder = ModelBuilder::new();
        builder.set_world_gravity(WorldId(0), Vec3::ZERO);
        builder.set_linear_damping(0.0);
        for (&p, &w) in rest.iter().zip(&inverse_mass) {
            builder.add_particle(p, 1.0 / w, 0.0, WorldId(0));
        }
        builder.add_tet_mesh(&tets, 2e6, 0.45, None).unwrap();
        let mut model = builder.build().unwrap();
        model
            .tet_dm_inv
            .host_mut()
            .unwrap()
            .copy_from_slice(&inverse);
        for t in 0..tets.len() {
            model.tet_rest_volume.host_mut().unwrap()[t] = scalar(fixture.volume[t]);
            model.tet_mu.host_mut().unwrap()[t] = scalar(fixture.mu[t]);
            model.tet_lambda.host_mut().unwrap()[t] = scalar(fixture.lambda[t]);
            model.tet_viscosity.host_mut().unwrap()[t] = TetViscosity {
                shear: scalar(fixture.viscosity[t * 2]),
                bulk: scalar(fixture.viscosity[t * 2 + 1]),
            };
        }
        let mut input = model.state().unwrap();
        let mut output = model.state().unwrap();
        input.particle_q.host_mut().unwrap().copy_from_slice(&q);
        output
            .particle_qd
            .host_mut()
            .unwrap()
            .copy_from_slice(&velocity);
        let incoming: Vec<_> = target.iter().map(|y| projected(*y / h)).collect();
        input
            .particle_qd
            .host_mut()
            .unwrap()
            .copy_from_slice(&incoming);
        let delta: Vec<_> = velocity.iter().map(|v| projected(*v * h)).collect();
        let target: Vec<_> = incoming.iter().map(|v| projected(*v * h)).collect();
        let node = fixture.node;
        let mut force = (target[node] - delta[node]) / (inverse_mass[node] * h * h);
        for (t, tet) in tets.iter().enumerate() {
            let Some(role) = tet.iter().position(|i| *i as usize == node) else {
                continue;
            };
            let origin = tet[0] as usize;
            let edge = |i: usize| {
                (q[tet[i] as usize] - q[origin]) - (rest[tet[i] as usize] - rest[origin])
            };
            let previous = Mat3::IDENTITY + Mat3::from_cols(edge(1), edge(2), edge(3)) * inverse[t];
            let increment = Mat3::from_cols(
                delta[tet[1] as usize] - delta[origin],
                delta[tet[2] as usize] - delta[origin],
                delta[tet[3] as usize] - delta[origin],
            ) * inverse[t];
            let f = previous + increment;
            let cof = Mat3::from_cols(
                f.y_axis.cross(f.z_axis),
                f.z_axis.cross(f.x_axis),
                f.x_axis.cross(f.y_axis),
            );
            let mu = scalar(fixture.mu[t]);
            let lam = scalar(fixture.lambda[t]);
            let mut stress = (f - cof) * mu + cof * ((lam + mu) * (f.determinant() - 1.0));
            let strain = (previous.transpose() * increment
                + increment.transpose() * previous
                + increment.transpose() * increment)
                * 0.5;
            let trace = strain.x_axis.x + strain.y_axis.y + strain.z_axis.z;
            let viscous_stress = (strain - Mat3::IDENTITY * (trace / 3.0))
                * (2.0 * scalar(fixture.viscosity[2 * t]))
                + Mat3::IDENTITY * (trace * scalar(fixture.viscosity[2 * t + 1]));
            stress += f * viscous_stress / h;
            let rows = inverse[t].transpose();
            let gradients = [
                -(rows.x_axis + rows.y_axis + rows.z_axis),
                rows.x_axis,
                rows.y_axis,
                rows.z_axis,
            ];
            force -= stress * gradients[role] * scalar(fixture.volume[t]);
        }
        model.upload_to_device(device, queue).unwrap();
        input.upload_to_device(device, queue).unwrap();
        output.upload_to_device(device, queue).unwrap();
        let mut solver = VbdSolver::new_gpu(&engine, &model).unwrap();
        let buffer = solver
            .material_force_residual_gpu(&model, &input, &mut output, h)
            .unwrap();
        let samples = readback::read::<[f32; 4]>(&engine, buffer, q.len());
        assert_eq!(
            samples[node][..3],
            target[node].to_array().map(|v| v as f32)
        );
        let expected = force.length() * h * inverse_mass[node];
        let error = (samples[node][3] as f64 - expected).abs();
        eprintln!(
            "case={case} GPU={} oracle={expected} error={error}",
            samples[node][3]
        );
        assert!(
            error < 3e-6,
            "case={case}: GPU/oracle force mismatch {error}"
        );
        assert_eq!(
            samples[node][3] > 1e-4,
            expected > 1e-4,
            "case={case}: force acceptance mismatch"
        );
    }
}
