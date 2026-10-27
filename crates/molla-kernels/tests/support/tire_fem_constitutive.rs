use super::*;
use wgpu::util::DeviceExt;

type Matrix = [[f64; 3]; 3];

fn energy(f: Matrix, mu: f64, lambda: f64) -> f64 {
    let j = f[0][0] * (f[1][1] * f[2][2] - f[1][2] * f[2][1])
        - f[0][1] * (f[1][0] * f[2][2] - f[1][2] * f[2][0])
        + f[0][2] * (f[1][0] * f[2][1] - f[1][1] * f[2][0]);
    assert!(j > 0.2);
    0.5 * mu * (f.iter().flatten().map(|x| x * x).sum::<f64>() - 3.) - mu * j.ln()
        + 0.5 * lambda * j.ln().powi(2)
}
fn piola(f: Matrix, mu: f64, lambda: f64) -> Matrix {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let h = 1e-5;
            let sample = |x: f64| {
                let mut f = f;
                f[i][j] += x * h;
                energy(f, mu, lambda)
            };
            (sample(-2.) - 8. * sample(-1.) + 8. * sample(1.) - sample(2.)) / (12. * h)
        })
    })
}
fn deformations() -> Vec<Matrix> {
    vec![
        [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
        [[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]],
        [[0.6, -0.8, 0.], [0.8, 0.6, 0.], [0., 0., 1.]],
        [[1., 0.25, 0.], [0., 1., 0.], [0., 0., 1.]],
        [[1., 0., 0.5], [0., 1., 0.], [0., 0., 1.]],
        [[1., 0., 0.], [0., 1., -0.25], [0., 0., 1.]],
        [[1.1, 0.2, -0.1], [0.1, 0.9, 0.3], [0.05, -0.2, 1.2]],
    ]
}
fn deformation(input: &TireFemGpuInput) -> Matrix {
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (1..4)
                .map(|a| {
                    (input.positions[a][i] as f64 - input.positions[0][i] as f64)
                        * input.tet_grads[a][j] as f64
                })
                .sum()
        })
    })
}

fn translated_cases() -> Vec<TireFemGpuInput> {
    let mut inputs = Vec::new();
    for translation in [[0., 2., 0.], [64., 32., -64.], [1024., 512., -1024.]] {
        for mut input in cases() {
            let rest = [
                [0., 0., 0.],
                [0.7, 0.2, 0.1],
                [0.1, 0.6, -0.2],
                [0.2, 0.1, 0.9],
            ];
            let dm: Matrix = std::array::from_fn(|i| std::array::from_fn(|j| rest[j + 1][i]));
            let det = dm[0][0] * (dm[1][1] * dm[2][2] - dm[1][2] * dm[2][1])
                - dm[0][1] * (dm[1][0] * dm[2][2] - dm[1][2] * dm[2][0])
                + dm[0][2] * (dm[1][0] * dm[2][1] - dm[1][1] * dm[2][0]);
            let f = deformation(&input);
            for j in 0..3 {
                for i in 0..3 {
                    input.tet_grads[j + 1][i] = ((dm[(i + 1) % 3][(j + 1) % 3]
                        * dm[(i + 2) % 3][(j + 2) % 3]
                        - dm[(i + 1) % 3][(j + 2) % 3] * dm[(i + 2) % 3][(j + 1) % 3])
                        / det) as f32;
                }
            }
            for i in 0..3 {
                input.tet_grads[0][i] =
                    -(input.tet_grads[1][i] + input.tet_grads[2][i] + input.tet_grads[3][i]);
            }
            input.tet_volumes[0] = (det / 6.) as f32;
            for a in 0..4 {
                for i in 0..3 {
                    input.rest_positions[a][i] = (rest[a][i] + translation[i]) as f32;
                    input.positions[a][i] =
                        ((0..3).map(|j| f[i][j] * rest[a][j]).sum::<f64>() + translation[i]) as f32;
                }
            }
            input.params.hub = translation.map(|v| v as f32);
            inputs.push(input);
        }
    }
    inputs
}
fn cases() -> Vec<TireFemGpuInput> {
    let mut inputs = Vec::new();
    for (mu, lambda) in [(2., 5.), (0.5, 1.), (7., 2.)] {
        for f in deformations() {
            let mut input = simple_tire_fem_input(0.);
            input.positions = input
                .rest_positions
                .iter()
                .map(|p| {
                    let xyz: [f32; 3] = std::array::from_fn(|i| {
                        ((0..3).map(|j| f[i][j] * p[j] as f64).sum::<f64>()
                            + if i == 1 { 2. } else { 0. }) as f32
                    });
                    [xyz[0], xyz[1], xyz[2], 0.]
                })
                .collect();
            input.tet_mu[0] = mu;
            input.tet_lambda[0] = lambda;
            input.tread_ids.clear();
            input.params.gravity_y = 0.;
            input.params.pressure_pa = 0.;
            input.params.node_damping = 0.;
            input.params.xpbd_volume_compliance = 1e30;
            input.params.xpbd_bead_compliance = 1e30;
            input.params.xpbd_sidewall_ply_compliance = 1e30;
            inputs.push(input);
        }
    }
    inputs
}

#[test]
fn neo_hookean_energy_oracle_shear_rotation() {
    let cases = deformations();
    let p = piola(cases[3], 2., 5.);
    assert!((p[0][1] - 0.5).abs() < 1e-8 && (p[1][0] - 0.5).abs() < 1e-8);
    for f in &cases[..3] {
        assert!(piola(*f, 2., 5.).iter().flatten().all(|v| v.abs() < 1e-8));
    }
}

fn matrix_free_stress(engine: &Engine, inputs: &[TireFemGpuInput]) -> Vec<[f32; 4]> {
    let src = include_str!("../../shaders/tire_fem_matrix_free_newton.wgsl");
    let function = src
        .find("fn cofactor3(")
        .or_else(|| src.find("fn adjugate3("))
        .unwrap();
    let end = src[function..].find("\nfn residual_at_node").unwrap() + function;
    let start = src.find("        let p0 = (*positions)[i0];").unwrap();
    let stop = src[start..]
        .find("        let volume = tet_volume[t];")
        .unwrap()
        + start;
    let mut law = src[start..stop]
        .replace("tet_mu[t]", "input[9u*t+8u].x")
        .replace("tet_lambda[t]", "input[9u*t+8u].y");
    for a in 0..4 {
        law = law
            .replace(
                &format!("(*positions)[i{a}]"),
                &format!("input[9u*t+{a}u].xyz"),
            )
            .replace(
                &format!("tet_grads[base + {a}u].xyz"),
                &format!("input[9u*t+{}u].xyz", a + 4),
            );
    }
    let shader = format!(
        "{}\n@group(0) @binding(0) var<storage,read> input:array<vec4<f32>>;\n@group(0) @binding(1) var<storage,read_write> output:array<vec4<f32>>;\n@compute @workgroup_size(64) fn probe(@builtin(global_invocation_id) id:vec3<u32>) {{let t=id.x;if t*9u>=arrayLength(&input) {{return;}}{law}\nfor(var c=0u;c<3u;c++) {{output[3u*t+c]=vec4<f32>(p_stress[c],0.0);}}}}",
        &src[function..end]
    );
    let rows: Vec<[f32; 4]> = inputs
        .iter()
        .flat_map(|input| {
            [
                input.positions[0],
                input.positions[1],
                input.positions[2],
                input.positions[3],
                input.tet_grads[0],
                input.tet_grads[1],
                input.tet_grads[2],
                input.tet_grads[3],
                [input.tet_mu[0], input.tet_lambda[0], 0., 0.],
            ]
        })
        .collect();
    let device = engine.device().unwrap();
    let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&rows),
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 48 * inputs.len() as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: None,
        source: wgpu::ShaderSource::Wgsl(shader.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: source.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: output.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((inputs.len() as u32).div_ceil(64), 1, 1);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    let bytes = super::cavity_frame_tests::read(engine, &output);
    bytes
        .chunks_exact(16)
        .map(bytemuck::pod_read_unaligned)
        .collect()
}

#[test]
#[ignore = "native GPU; actual single/atlas forces and matrix-free law vs energy derivative"]
fn neo_hookean_energy_derivative_gpu() {
    check_energy_derivative(cases());
}

fn fiber_stress(f: Matrix, a: [f64; 3], k1: f64, k2: f64) -> Matrix {
    let fa: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| f[i][j] * a[j]).sum());
    let stretch = fa.iter().map(|v| v * v).sum::<f64>() - 1.;
    let coefficient = if k1 <= 0. || stretch <= 0. {
        0.
    } else {
        2. * k1 * stretch * (k2.max(0.) * stretch * stretch).min(30.).exp()
    };
    std::array::from_fn(|i| std::array::from_fn(|j| coefficient * fa[i] * a[j]))
}

#[test]
#[ignore = "native GPU; fiber force/integration against independent directional stress derivative"]
fn fiber_tangent_integration_gpu() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireFemGpuKernel::new(device);
    let mut inputs = Vec::new();
    for family in 0..4 {
        for stretch in [0.8, 1.001, 1.3, 2.1, 3.] {
            for k2 in [-1., 0., 2., 5.] {
                let mut input = cases()[0].clone();
                input.tet_mu[0] = 0.;
                input.tet_lambda[0] = 0.;
                input.params.dt = 0.2;
                input.params.semi_implicit_stiffness_scale = 1.;
                let f = [[stretch, 0.1, 0.], [0., stretch, 0.], [0., 0., 1.]];
                for a in 0..4 {
                    for i in 0..3 {
                        input.positions[a][i] = ((0..3)
                            .map(|j| f[i][j] * input.rest_positions[a][j] as f64)
                            .sum::<f64>()
                            + if i == 1 { 4. } else { 0. })
                            as f32;
                    }
                }
                let (dir, k1s, k2s) = match family {
                    0 => (
                        &mut input.tet_fiber_dir,
                        &mut input.tet_fiber_k1,
                        &mut input.tet_fiber_k2,
                    ),
                    1 => (
                        &mut input.tet_fiber_dir_b,
                        &mut input.tet_fiber_k1_b,
                        &mut input.tet_fiber_k2_b,
                    ),
                    2 => (
                        &mut input.tet_fiber_dir_c,
                        &mut input.tet_fiber_k1_c,
                        &mut input.tet_fiber_k2_c,
                    ),
                    _ => (
                        &mut input.tet_fiber_dir_d,
                        &mut input.tet_fiber_k1_d,
                        &mut input.tet_fiber_k2_d,
                    ),
                };
                dir[0] = [0.6, 0.8, 0., 0.];
                k1s[0] = 2.3;
                k2s[0] = k2;
                inputs.push((input, k2));
            }
        }
    }
    let inputs: Vec<_> = inputs
        .into_iter()
        .flat_map(|(input, k2)| {
            let mut explicit = input.clone();
            explicit.params.semi_implicit_stiffness_scale = 0.;
            [(explicit, k2), (input, k2)]
        })
        .collect();
    let payloads: Vec<_> = inputs.iter().map(|(p, _)| p.clone()).collect();
    let atlas = kernel
        .dispatch_batch_atlas_blocking(device, queue, &payloads)
        .unwrap();
    let mut maximum = [0f64; 3];
    for (case, (input, k2)) in inputs.iter().enumerate() {
        let f = deformation(input);
        let a = [0.6f32 as f64, 0.8f32 as f64, 0.];
        let fa: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| f[i][j] * a[j]).sum());
        let length = fa.iter().map(|v| v * v).sum::<f64>().sqrt();
        let q = fa.map(|v| v / length);
        let p = fiber_stress(f, a, 2.3f32 as f64, *k2 as f64);
        let single = kernel.dispatch_blocking(device, queue, input).unwrap();
        for (route, result) in [(0, &single), (1, &atlas[case])] {
            assert!(
                result
                    .positions
                    .iter()
                    .chain(&result.velocities)
                    .chain(&result.forces)
                    .flatten()
                    .all(|v| v.is_finite())
            );
            for node in 0..4 {
                let g: [f64; 3] = std::array::from_fn(|j| input.tet_grads[node][j] as f64);
                let h = 1e-5;
                let sample = |step: f64| {
                    let perturbed = std::array::from_fn(|i| {
                        std::array::from_fn(|j| f[i][j] + h * step * q[i] * g[j])
                    });
                    let stress = fiber_stress(perturbed, a, 2.3f32 as f64, *k2 as f64);
                    (0..3)
                        .map(|i| q[i] * (0..3).map(|j| stress[i][j] * g[j]).sum::<f64>())
                        .sum::<f64>()
                };
                let tangent =
                    (sample(-2.) - 8. * sample(-1.) + 8. * sample(1.) - sample(2.)) / (12. * h);
                let dt = input.params.dt as f64;
                let volume = input.tet_volumes[0] as f64;
                let denom = input.masses[node] as f64
                    + input.params.semi_implicit_stiffness_scale as f64
                        * volume
                        * tangent
                        * dt
                        * dt;
                for i in 0..3 {
                    let force = -volume * (0..3).map(|j| p[i][j] * g[j]).sum::<f64>();
                    let velocity = (force * dt / denom).clamp(-8., 8.);
                    for (slot, actual, expected) in [
                        (2, result.forces[node][i], force),
                        (route, result.velocities[node][i], velocity),
                        (
                            route,
                            result.positions[node][i],
                            input.positions[node][i] as f64 + velocity * dt,
                        ),
                    ] {
                        let error = (actual as f64 - expected).abs() / (1. + expected.abs());
                        maximum[slot] = maximum[slot].max(error);
                        assert!(
                            error <= 2e-5,
                            "case {case} route {route} node {node} axis {i} slot {slot} error {error:e}"
                        );
                    }
                }
            }
        }
    }
    println!(
        "160 fiber cases, four independent families, stiffness enabled/disabled: single/atlas integration and force scaled errors {maximum:?}"
    );
}

#[test]
#[ignore = "native GPU; translated skew tetrahedra vs quantized-coordinate energy derivative"]
fn neo_hookean_translated_energy_derivative_gpu() {
    check_energy_derivative(translated_cases());
}

fn check_energy_derivative(inputs: Vec<TireFemGpuInput>) {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireFemGpuKernel::new(device);
    let atlas = kernel
        .dispatch_batch_atlas_blocking(device, queue, &inputs)
        .unwrap();
    let matrix = matrix_free_stress(&engine, &inputs);
    let mut maximum = [0f64; 3];
    for (case, input) in inputs.iter().enumerate() {
        let f = deformation(input);
        let p = piola(f, input.tet_mu[0] as f64, input.tet_lambda[0] as f64);
        let single = kernel.dispatch_blocking(device, queue, input).unwrap();
        for (route, result) in [(0, &single), (1, &atlas[case])] {
            assert!(result.forces.iter().flatten().all(|x| x.is_finite()));
            for node in 0..4 {
                for axis in 0..3 {
                    let expected = -input.tet_volumes[0] as f64
                        * (0..3)
                            .map(|j| {
                                let g = if node == 0 {
                                    -(1..4).map(|a| input.tet_grads[a][j] as f64).sum::<f64>()
                                } else {
                                    input.tet_grads[node][j] as f64
                                };
                                p[axis][j] * g
                            })
                            .sum::<f64>();
                    let error =
                        (result.forces[node][axis] as f64 - expected).abs() / (1. + expected.abs());
                    maximum[route] = maximum[route].max(error);
                }
            }
        }
        for col in 0..3 {
            for row in 0..3 {
                let expected = p[row][col];
                let actual = matrix[3 * case + col][row] as f64;
                assert!(actual.is_finite());
                maximum[2] = maximum[2].max((actual - expected).abs() / (1. + expected.abs()));
            }
        }
    }
    println!(
        "{} identity/rotation/shear/general cases; single/atlas/matrix-free max scaled energy-derivative errors {maximum:?}",
        inputs.len()
    );
    assert!(
        maximum.iter().all(|e| *e <= 2e-5),
        "constitutive gate {maximum:?}"
    );
}
