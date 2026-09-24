#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{GpuGmres, GpuGmresDiagnostics, GpuGmresSettings};
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod readback;

fn buffer(device: &wgpu::Device, values: &[f32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(values),
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

struct Matrix {
    pipeline: wgpu::ComputePipeline,
    matrix: wgpu::Buffer,
    params: wgpu::Buffer,
    group: Option<wgpu::BindGroup>,
    n: u32,
}
impl Matrix {
    fn new(device: &wgpu::Device, matrix: &[f32], n: u32, banded: bool) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("test nonsymmetric operator"),
            source: wgpu::ShaderSource::Wgsl(
                "struct Params { n: u32, banded: u32, pad: vec2<u32> }
                @group(0) @binding(0) var<storage, read> matrix: array<f32>;
                @group(0) @binding(1) var<storage, read> input_vector: array<f32>;
                @group(0) @binding(2) var<storage, read_write> output_vector: array<f32>;
                @group(0) @binding(3) var<uniform> u: Params;
                @compute @workgroup_size(64)
                fn multiply(@builtin(global_invocation_id) id: vec3<u32>) {
                    let i = id.x;
                    if i >= u.n { return; }
                    var value = 0.0;
                    if u.banded != 0u {
                        value = matrix[i]*input_vector[i];
                        if i > 0u { value = fma(-0.7,input_vector[i-1u],value); }
                        if i+1u < u.n { value = fma(0.3,input_vector[i+1u],value); }
                    } else {
                        for (var j = 0u; j < u.n; j++) { value = fma(matrix[i*u.n+j],input_vector[j],value); }
                    }
                    output_vector[i] = value;
                }".into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: None,
            module: &shader,
            entry_point: Some("multiply"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            pipeline,
            matrix: buffer(device, matrix),
            n,
            group: None,
            params: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&[n, u32::from(banded), 0, 0]),
                usage: wgpu::BufferUsages::UNIFORM,
            }),
        }
    }
    fn encode(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &wgpu::Buffer,
        output: &wgpu::Buffer,
    ) -> molla_core::Result<()> {
        let group = self.group.get_or_insert_with(|| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &self.pipeline.get_bind_group_layout(0),
                entries: &[&self.matrix, input, output, &self.params]
                    .into_iter()
                    .enumerate()
                    .map(|(binding, b)| wgpu::BindGroupEntry {
                        binding: binding as u32,
                        resource: b.as_entire_binding(),
                    })
                    .collect::<Vec<_>>(),
            })
        });
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &*group, &[]);
        pass.dispatch_workgroups(self.n.div_ceil(64), 1, 1);
        Ok(())
    }
}

fn settings(cycles: u32) -> GpuGmresSettings {
    GpuGmresSettings {
        cycles,
        absolute_tolerance: 1e-6,
        relative_tolerance: 2e-6,
        zero_initial_guess: true,
    }
}
fn dense_product(matrix: &[f32], vector: &[f32]) -> Vec<f64> {
    matrix
        .chunks_exact(vector.len())
        .map(|row| {
            row.iter()
                .zip(vector)
                .map(|(&a, &b)| a as f64 * b as f64)
                .sum()
        })
        .collect()
}
fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.map(|v| v * v).sum::<f64>().sqrt()
}
fn diagnostics(engine: &Engine, kernel: &GpuGmres) -> GpuGmresDiagnostics {
    readback::read::<GpuGmresDiagnostics>(engine, kernel.diagnostics(), 1)[0]
}
fn solve(
    engine: &Engine,
    kernel: &GpuGmres,
    matrix: &mut Matrix,
    rhs: &wgpu::Buffer,
    inverse: &wgpu::Buffer,
    settings: GpuGmresSettings,
) -> GpuGmresDiagnostics {
    let device = engine.device().unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    kernel
        .encode(
            device,
            &mut encoder,
            rhs,
            inverse,
            settings,
            |encoder, input, output| matrix.encode(device, encoder, input, output),
        )
        .unwrap();
    engine.queue().unwrap().submit([encoder.finish()]);
    diagnostics(engine, kernel)
}

#[test]
#[ignore = "requires real GPU through make test-gpu"]
fn gmres_solves_nonsymmetric_indefinite_and_restarted_systems() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    for n in [1, 2, 4, 17, 65] {
        let mut a = vec![0.0_f32; n * n];
        for i in 0..n {
            a[i * n + i] = if i % 3 == 0 {
                -3.0
            } else {
                2.5 + (i % 5) as f32
            };
            if i > 0 {
                a[i * n + i - 1] = -0.7;
            }
            if i + 1 < n {
                a[i * n + i + 1] = 0.3;
            }
        }
        if n == 2 {
            a.copy_from_slice(&[0.0, -1.0, 1.0, 0.0]);
        }
        let known: Vec<_> = (0..n)
            .map(|i| ((i * 7 % 19) as f32 - 9.0) * 0.125)
            .collect();
        let rhs: Vec<_> = dense_product(&a, &known)
            .into_iter()
            .map(|v| v as f32)
            .collect();
        let inverse: Vec<_> = (0..n)
            .map(|i| {
                if a[i * n + i] == 0.0 {
                    1.0
                } else {
                    1.0 / a[i * n + i].abs()
                }
            })
            .collect();
        let rhs_buffer = buffer(device, &rhs);
        let inverse_buffer = buffer(device, &inverse);
        let kernel = GpuGmres::new(device, n as u32, (n as u32).min(4)).unwrap();
        let mut matrix = Matrix::new(device, &a, n as u32, false);
        let result = solve(
            &engine,
            &kernel,
            &mut matrix,
            &rhs_buffer,
            &inverse_buffer,
            settings(12),
        );
        let x = readback::read::<f32>(&engine, kernel.solution(), n);
        assert_eq!(result.faults, 0, "n={n} {result:?}");
        assert_eq!(result.converged, 1, "n={n} {result:?}");
        let actual = norm(
            dense_product(&a, &x)
                .into_iter()
                .zip(&rhs)
                .map(|(a, &b)| a - b as f64),
        );
        assert!(
            actual <= 1.3 * result.threshold as f64,
            "n={n} independent={actual} {result:?}"
        );
        for (actual, expected) in x.iter().zip(&known) {
            assert!(
                (actual - expected).abs() < 2e-5,
                "n={n} {actual} != {expected}"
            );
        }
        if n > 4 {
            assert!(result.iterations > 4, "restart untested: {result:?}");
        }
        assert_eq!(readback::read::<f32>(&engine, &rhs_buffer, n), rhs);
        assert_eq!(readback::read::<f32>(&engine, &inverse_buffer, n), inverse);
        assert_eq!(readback::read::<f32>(&engine, &matrix.matrix, n * n), a);
        println!("nonsymmetric n={n}: {result:?}, independent residual={actual}");
    }
}

#[test]
#[ignore = "requires real GPU through make test-gpu"]
fn gmres_handles_initial_guesses_zero_rhs_and_singular_breakdown() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = GpuGmres::new(device, 4, 4).unwrap();
    let rhs = buffer(device, &[1.0, 0.0, 2.0, 0.0]);
    let inverse = buffer(device, &[1.0; 4]);
    let mut a = [0.0_f32; 16];
    a[0] = 1.0;
    a[10] = 2.0;
    let mut matrix = Matrix::new(device, &a, 4, false);
    let result = solve(&engine, &kernel, &mut matrix, &rhs, &inverse, settings(2));
    assert_eq!((result.faults, result.converged), (0, 1), "{result:?}");
    let x = readback::read::<f32>(&engine, kernel.solution(), 4);
    assert!((x[0] - 1.0).abs() < 1e-5 && (x[2] - 1.0).abs() < 1e-5);
    queue.write_buffer(&matrix.matrix, 0, bytemuck::cast_slice(&[0.0_f32; 16]));
    let result = solve(&engine, &kernel, &mut matrix, &rhs, &inverse, settings(3));
    assert_eq!((result.faults, result.converged), (0, 0), "{result:?}");
    assert!(
        result.breakdowns > 0 && result.residual_norm > 2.0,
        "{result:?}"
    );
    assert_eq!(
        readback::read::<f32>(&engine, kernel.solution(), 4),
        vec![0.0; 4]
    );
    for i in 0..4 {
        a[i * 4 + i] = 1.0;
    }
    queue.write_buffer(&matrix.matrix, 0, bytemuck::cast_slice(&a));
    queue.write_buffer(&rhs, 0, bytemuck::cast_slice(&[0.0_f32; 4]));
    queue.write_buffer(
        kernel.solution(),
        0,
        bytemuck::cast_slice(&[0.25_f32, -0.5, 0.75, -1.0]),
    );
    let mut config = settings(2);
    config.zero_initial_guess = false;
    let result = solve(&engine, &kernel, &mut matrix, &rhs, &inverse, config);
    assert_eq!((result.faults, result.converged), (0, 1), "{result:?}");
    assert!(result.initial_norm > 1.0 && result.iterations > 0);
    assert!(
        readback::read::<f32>(&engine, kernel.solution(), 4)
            .iter()
            .all(|x| x.abs() < 1e-6)
    );
    let result = solve(&engine, &kernel, &mut matrix, &rhs, &inverse, settings(2));
    assert_eq!(
        (result.faults, result.converged, result.iterations),
        (0, 1, 0),
        "{result:?}"
    );
}

#[test]
#[ignore = "requires real GPU through make test-gpu"]
fn gmres_rejects_invalid_inputs_and_recovers_after_nonfinite_data() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (n, restart) in [
        (0, 1),
        (3, 0),
        (3, 4),
        (70, 65),
        (u32::MAX, 32),
        (u32::MAX, u32::MAX),
    ] {
        assert!(GpuGmres::new(device, n, restart).is_err());
    }
    let kernel = GpuGmres::new(device, 2, 2).unwrap();
    let rhs = buffer(device, &[1.0, 2.0]);
    let inverse = buffer(device, &[1.0, 1.0]);
    let mut matrix = Matrix::new(device, &[1.0, 0.0, 0.0, 1.0], 2, false);
    let small = buffer(device, &[1.0]);
    let mut encoder = device.create_command_encoder(&Default::default());
    for rhs in [&small, kernel.solution()] {
        assert!(
            kernel
                .encode(
                    device,
                    &mut encoder,
                    rhs,
                    &inverse,
                    settings(1),
                    |_, _, _| panic!("invalid input callback")
                )
                .is_err()
        );
    }
    for config in [
        GpuGmresSettings {
            cycles: 0,
            ..settings(1)
        },
        GpuGmresSettings {
            absolute_tolerance: f32::NAN,
            ..settings(1)
        },
        GpuGmresSettings {
            relative_tolerance: 1.0,
            ..settings(1)
        },
        GpuGmresSettings {
            absolute_tolerance: 0.0,
            relative_tolerance: 0.0,
            ..settings(1)
        },
    ] {
        assert!(
            kernel
                .encode(
                    device,
                    &mut encoder,
                    &rhs,
                    &inverse,
                    config,
                    |_, _, _| panic!("invalid settings callback")
                )
                .is_err()
        );
    }
    for (buffer, bad, bit) in [
        (&rhs, f32::NAN, 1),
        (&inverse, 0.0, 4),
        (&inverse, f32::INFINITY, 4),
        (&matrix.matrix.clone(), f32::NAN, 1),
    ] {
        queue.write_buffer(buffer, 0, bytemuck::bytes_of(&bad));
        let result = solve(&engine, &kernel, &mut matrix, &rhs, &inverse, settings(2));
        assert!(
            result.faults & bit != 0 && result.converged == 0,
            "{result:?}"
        );
        queue.write_buffer(buffer, 0, bytemuck::bytes_of(&1.0_f32));
        let result = solve(&engine, &kernel, &mut matrix, &rhs, &inverse, settings(2));
        assert_eq!((result.faults, result.converged), (0, 1), "{result:?}");
    }
}

#[test]
#[ignore = "requires real GPU through make test-gpu"]
fn gmres_runs_ceol_sized_vectors_without_host_recurrence() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let n = 3 * 29568 + 27 + 3 * 7576;
    let diagonal: Vec<_> = (0..n).map(|i| 2.0 + (i % 11) as f32).collect();
    let inverse: Vec<_> = diagonal.iter().map(|d| 1.0 / d).collect();
    let known: Vec<_> = (0..n)
        .map(|i| ((i * 13 % 37) as f32 - 18.0) * 0.125)
        .collect();
    let product = |vector: &[f32]| -> Vec<f64> {
        (0..n)
            .map(|i| {
                diagonal[i] as f64 * vector[i] as f64
                    + if i > 0 {
                        -0.7_f32 as f64 * vector[i - 1] as f64
                    } else {
                        0.0
                    }
                    + if i + 1 < n {
                        0.3_f32 as f64 * vector[i + 1] as f64
                    } else {
                        0.0
                    }
            })
            .collect()
    };
    let rhs: Vec<_> = product(&known).into_iter().map(|v| v as f32).collect();
    let mut matrix = Matrix::new(device, &diagonal, n as u32, true);
    let kernel = GpuGmres::new(device, n as u32, 12).unwrap();
    let result = solve(
        &engine,
        &kernel,
        &mut matrix,
        &buffer(device, &rhs),
        &buffer(device, &inverse),
        settings(4),
    );
    let x = readback::read::<f32>(&engine, kernel.solution(), n);
    let actual = norm(
        product(&x)
            .into_iter()
            .zip(&rhs)
            .map(|(a, &b)| a - b as f64),
    );
    assert_eq!((result.faults, result.converged), (0, 1), "{result:?}");
    assert!(
        actual <= 1.3 * result.threshold as f64,
        "independent={actual} {result:?}"
    );
    assert!(x.iter().zip(&known).all(|(a, b)| (a - b).abs() < 3e-5));
    println!(
        "CEOL-sized nonsymmetric fixture n={n}: {result:?}; independent residual={actual}; not a CEOL physics benchmark"
    );
}

#[test]
#[ignore = "requires real GPU through make test-gpu"]
fn gmres_norms_do_not_underflow_or_overflow_finite_vectors() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = GpuGmres::new(device, 2, 2).unwrap();
    let rhs = buffer(device, &[0.0; 2]);
    let inverse = buffer(device, &[1.0; 2]);
    let mut matrix = Matrix::new(device, &[1.0, 0.0, 0.0, 1.0], 2, false);
    for scale in [1e-25_f32, 1e25] {
        let values = [scale, -2.0 * scale];
        queue.write_buffer(&rhs, 0, bytemuck::cast_slice(&values));
        let result = solve(
            &engine,
            &kernel,
            &mut matrix,
            &rhs,
            &inverse,
            GpuGmresSettings {
                absolute_tolerance: 0.0,
                relative_tolerance: 1e-5,
                ..settings(2)
            },
        );
        assert_eq!(
            (result.faults, result.converged),
            (0, 1),
            "scale={scale} {result:?}"
        );
        assert!(
            result.initial_norm > 0.0 && result.initial_norm.is_finite() && result.iterations > 0,
            "{result:?}"
        );
        for (actual, expected) in readback::read::<f32>(&engine, kernel.solution(), 2)
            .iter()
            .zip(values)
        {
            assert!(
                ((*actual as f64 - expected as f64) / expected as f64).abs() < 1e-5,
                "{actual} vs {expected} {result:?}"
            );
        }
    }
}
