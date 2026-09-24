#![cfg(feature = "gpu")]

use molla_compute::Engine;
use wgpu::util::DeviceExt;

#[path = "common/gpu_readback.rs"]
#[allow(dead_code)]
mod gpu_readback;

fn velocity(input: &[f64], lambda: &[f64; 9], i: usize) -> f64 {
    input[81 + i]
        + (0..9)
            .map(|j| input[i * 9 + j] * (lambda[j] - input[90 + j]))
            .sum::<f64>()
}

fn reference(input: &[f64], iterations: usize) -> [f64; 9] {
    let mut lambda: [f64; 9] = input[90..99].try_into().unwrap();
    for _ in 0..iterations {
        for r in 0..3 {
            let n = 3 * r;
            lambda[n] = (lambda[n]
                + (input[99 + r] - velocity(input, &lambda, n)) / input[n * 9 + n])
                .max(0.0);
            let mut tangent = [
                lambda[n + 1] - velocity(input, &lambda, n + 1) / input[105 + r],
                lambda[n + 2] - velocity(input, &lambda, n + 2) / input[105 + r],
            ];
            let magnitude = tangent[0].hypot(tangent[1]);
            let limit = input[102 + r] * lambda[n];
            if magnitude > limit {
                tangent.iter_mut().for_each(|v| *v *= limit / magnitude);
            }
            lambda[n + 1..n + 3].copy_from_slice(&tangent);
        }
    }
    lambda
}

fn residual(input: &[f64], lambda: &[f64; 9]) -> f64 {
    let mut maximum = 0.0_f64;
    for r in 0..3 {
        let n = 3 * r;
        let inverse = input[n * 9 + n];
        let normal = ((lambda[n] + (input[99 + r] - velocity(input, lambda, n)) / inverse)
            .max(0.0)
            - lambda[n])
            .abs()
            * inverse;
        let mut trial = [
            lambda[n + 1] - velocity(input, lambda, n + 1) / input[105 + r],
            lambda[n + 2] - velocity(input, lambda, n + 2) / input[105 + r],
        ];
        let length = trial[0].hypot(trial[1]);
        let limit = input[102 + r] * lambda[n];
        if length > limit {
            trial.iter_mut().for_each(|v| *v *= limit / length);
        }
        let tangent = (trial[0] - lambda[n + 1]).hypot(trial[1] - lambda[n + 2]) * input[105 + r];
        maximum = maximum.max(normal).max(tangent);
        assert!(lambda[n] >= 0.0);
        assert!(lambda[n + 1].hypot(lambda[n + 2]) <= limit + 1e-9);
    }
    maximum
}

#[test]
#[ignore = "requires MOLLA_CONTACT_BLOCK_FIXTURE and a real GPU through make test-gpu"]
fn captured_coulomb_block_matches_converged_f64_reference() {
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_CONTACT_BLOCK_FIXTURE").unwrap());
    let bytes = std::fs::read(root.join("input.bin")).unwrap();
    let floats: &[f32] = bytemuck::cast_slice(&bytes);
    assert_eq!(floats.len(), 108);
    let exact_bytes = std::fs::read(root.join("input-f64.bin")).unwrap();
    let input: Vec<f64> = exact_bytes
        .chunks_exact(8)
        .map(|v| f64::from_le_bytes(v.try_into().unwrap()))
        .collect();
    assert_eq!(input.len(), 108);
    assert!(input.iter().all(|v| v.is_finite()));
    let expected = reference(&input, 131072);
    let expected_error = residual(&input, &expected);
    eprintln!("normal/tangent CPU residual={expected_error:e}");
    assert!(expected_error < 1e-7);
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("captured Coulomb block probe"),
        source: wgpu::ShaderSource::Wgsl(include_str!("common/contact_block_probe.wgsl").into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: None,
        module: &shader,
        entry_point: Some("solve"),
        compilation_options: Default::default(),
        cache: None,
    });
    let input_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: &bytes,
        usage: wgpu::BufferUsages::STORAGE,
    });
    let output = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 19 * 4,
        mapped_at_creation: false,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    });
    for (iterations, blocked) in [(1024_u32, 0_u32), (1, 1), (8, 1), (32, 1), (128, 1)] {
        let settings = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&[iterations, blocked, 0, 0]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: settings.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let values = gpu_readback::read::<f32>(&engine, &output, 19);
        assert!(values.iter().all(|v| v.is_finite()));
        let lambda: [f64; 9] = std::array::from_fn(|i| values[i] as f64);
        let error = residual(&input, &lambda);
        let velocity_error = (0..9)
            .map(|i| (velocity(&input, &lambda, i) - velocity(&input, &expected, i)).abs())
            .fold(0.0, f64::max);
        let reported_error = (0..9)
            .map(|i| (velocity(&input, &lambda, i) - values[9 + i] as f64).abs())
            .fold(0.0, f64::max);
        assert!(reported_error < 1e-4);
        eprintln!(
            "block={blocked} iterations={iterations} accepted_blocks={} residual={error:e} velocity_error={velocity_error:e}",
            values[18]
        );
        std::fs::write(
            root.join(format!("output-{blocked}-{iterations}.bin")),
            bytemuck::cast_slice(&values),
        )
        .unwrap();
        if blocked == 0 {
            assert!(error > 1e-4);
        }
        if blocked == 1 && iterations == 128 {
            assert!(error < 1e-4 && velocity_error < 1e-4);
        }
    }
}
