use super::*;
use crate::ExperimentalMassPredictor;
use crate::featherstone::rounding_tests::words;

pub(super) fn pairs(values: &[f64]) -> Vec<[f32; 2]> {
    values
        .iter()
        .map(|&v| {
            let high = v as f32;
            [high, (v - f64::from(high)) as f32]
        })
        .collect()
}

pub(super) fn input(device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> wgpu::Buffer {
    let buffer = molla_core::gpu_create_buffer!(
        device,
        &wgpu::BufferDescriptor {
            label: Some("mass predictor fixture"),
            size: data.len() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }
    );
    molla_core::gpu_write_buffer!(queue, &buffer, 0, data);
    buffer
}

fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    predictor: &ExperimentalMassPredictor,
    reset: bool,
) {
    let mut encoder = device.create_command_encoder(&Default::default());
    let before = device.get_internal_counters().hal;
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    if reset {
        predictor.reset(&mut encoder);
    }
    predictor.encode(&mut encoder);
    let after = device.get_internal_counters().hal;
    assert_eq!(before.buffers.read(), after.buffers.read());
    assert_eq!(before.bind_groups.read(), after.bind_groups.read());
    assert_eq!(
        before.compute_pipelines.read(),
        after.compute_pipelines.read()
    );
    assert_eq!(allocations.finish().observations, 0);
    assert_eq!(transfers.finish(), Default::default());
    let counts = dispatches.finish();
    assert!(!counts.overflowed);
    assert_eq!(counts.encoded_dispatches, 1);
    queue.submit([encoder.finish()]);
}

pub(super) fn values(raw: &[u32]) -> Vec<f64> {
    raw.chunks_exact(2)
        .map(|p| f64::from(f32::from_bits(p[0])) + f64::from(f32::from_bits(p[1])))
        .collect()
}

#[test]
#[ignore = "requires f64 GPU, resource counters and frozen biased acceleration fixture"]
fn retained_mass_predictor_gpu_contract() {
    assert!(molla_core::gpu_metrics::ENABLED);
    let directory =
        std::path::PathBuf::from(std::env::var_os("MOLLA_ACCELERATION_ORACLE_CAPTURE").unwrap());
    let capture =
        std::path::PathBuf::from(std::env::var_os("MOLLA_MASS_PREDICTOR_CAPTURE").unwrap());
    std::fs::create_dir(&capture).unwrap();
    let fixture = read(&directory.join("step-00-variant-0.bin"));
    validate_fixture_geometry(&fixture);
    let start = std::time::Instant::now();
    let engine = molla_compute::Engine::new_gpu_with_features(wgpu::Features::SHADER_F64).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for width in [1u32, 32] {
        let n = width as usize;
        let matrix: Vec<f32> = (0..n * n)
            .map(|k| {
                if k / n == k % n {
                    2.0 + (k / n) as f32
                } else {
                    0.125
                }
            })
            .collect();
        let expected: Vec<f64> = (0..n).map(|i| i as f64 / 8.0 - 0.125).collect();
        let b: Vec<f64> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| f64::from(matrix[i * n + j]) * expected[j])
                    .sum()
            })
            .collect();
        let bias: Vec<f64> = (0..n).map(|i| 10000.0 + i as f64).collect();
        let drive: Vec<f64> = b.iter().zip(&bias).map(|(b, c)| b + c).collect();
        let m = input(device, queue, bytemuck::cast_slice(&matrix));
        let f = input(device, queue, bytemuck::cast_slice(&pairs(&drive)));
        let c = input(device, queue, bytemuck::cast_slice(&pairs(&bias)));
        assert!(
            ExperimentalMassPredictor::new(device, width, &m, &f, &c, 8 * u64::from(width) + 3)
                .is_err()
        );
        let predictor =
            ExperimentalMassPredictor::new(device, width, &m, &f, &c, 8 * u64::from(width) + 4)
                .unwrap();
        dispatch(device, queue, &predictor, false);
        assert_eq!(words(device, queue, predictor.status()), [0]);
        let raw = words(device, queue, predictor.acceleration());
        let actual = values(&raw);
        assert!(
            actual
                .iter()
                .zip(&expected)
                .all(|(a, b)| (a - b).abs() < 1e-9)
        );
        std::fs::write(
            capture.join(format!("width-{width}.bin")),
            bytemuck::cast_slice(&raw),
        )
        .unwrap();
        let mut invalid = matrix.clone();
        invalid[0] = -1.0;
        queue.write_buffer(&m, 0, bytemuck::cast_slice(&invalid));
        dispatch(device, queue, &predictor, false);
        assert_eq!(words(device, queue, predictor.status()), [2]);
        assert_eq!(words(device, queue, predictor.acceleration()), raw);
        queue.write_buffer(&m, 0, bytemuck::cast_slice(&matrix));
        dispatch(device, queue, &predictor, false);
        assert_eq!(words(device, queue, predictor.status()), [2]);
        assert_eq!(words(device, queue, predictor.acceleration()), raw);
        dispatch(device, queue, &predictor, true);
        assert_eq!(words(device, queue, predictor.status()), [0]);
        assert_eq!(words(device, queue, predictor.acceleration()), raw);
        for (buffer, data) in [
            (&m, vec![f32::NAN]),
            (&f, vec![f32::INFINITY, 0.0]),
            (&c, vec![0.0, f32::NAN]),
        ] {
            queue.write_buffer(buffer, 0, bytemuck::cast_slice(&data));
            dispatch(device, queue, &predictor, true);
            assert_eq!(words(device, queue, predictor.status()), [1]);
            assert_eq!(words(device, queue, predictor.acceleration()), raw);
            queue.write_buffer(&m, 0, bytemuck::cast_slice(&matrix));
            queue.write_buffer(&f, 0, bytemuck::cast_slice(&pairs(&drive)));
            queue.write_buffer(&c, 0, bytemuck::cast_slice(&pairs(&bias)));
        }
        if width == 1 {
            queue.write_buffer(&m, 0, bytemuck::cast_slice(&[1e-30f32]));
            queue.write_buffer(&f, 0, bytemuck::cast_slice(&[1e30f32, 0.0]));
            dispatch(device, queue, &predictor, true);
            assert_eq!(words(device, queue, predictor.status()), [3]);
            assert_eq!(words(device, queue, predictor.acceleration()), raw);
        } else {
            invalid.clone_from(&matrix);
            invalid[1] += 1.0;
            queue.write_buffer(&m, 0, bytemuck::cast_slice(&invalid));
            dispatch(device, queue, &predictor, true);
            assert_eq!(words(device, queue, predictor.status()), [1]);
            assert_eq!(words(device, queue, predictor.acceleration()), raw);
        }
        eprintln!("mass_predictor_width={width} numerical/fault/sticky/reset=pass");
    }
    let mass = mass_matrix(&fixture);
    let packed: Vec<f32> = mass.iter().flatten().map(|&v| v as f32).collect();
    let offset = balance::force_offset(
        &fixture,
        Vec3::new(0.0, -(molla_math::GRAVITY as f32) as f64, 0.0),
    );
    let m = input(device, queue, bytemuck::cast_slice(&packed));
    let f = input(device, queue, &[0; 80]);
    let c = input(
        device,
        queue,
        bytemuck::cast_slice(&pairs(&offset.map(|v| -v))),
    );
    let predictor = ExperimentalMassPredictor::new(device, 10, &m, &f, &c, 84).unwrap();
    let mut base = [0.0; 10];
    let mut worst_absolute = 0.0f64;
    let mut worst_increment = 0.0f64;
    let mut worst_rate = 0.0f64;
    for step in 0..10 {
        let drive: [f64; 10] = std::array::from_fn(|i| {
            17.0 * (step + 1) as f64 * (i + 1) as f64 * if i % 2 == 0 { 1.0 } else { -1.0 }
        });
        queue.write_buffer(&f, 0, bytemuck::cast_slice(&pairs(&drive)));
        dispatch(device, queue, &predictor, false);
        assert_eq!(words(device, queue, predictor.status()), [0]);
        let raw = words(device, queue, predictor.acceleration());
        std::fs::write(
            capture.join(format!("biased-{step:02}.bin")),
            bytemuck::cast_slice(&raw),
        )
        .unwrap();
        let actual = values(&raw);
        let rhs = std::array::from_fn(|i| drive[i] + offset[i]);
        let expected = solve(mass, rhs);
        let scale = 1.0 + rhs.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        for i in 0..10 {
            let error =
                ((0..10).map(|j| mass[i][j] * actual[j]).sum::<f64>() - rhs[i]).abs() / scale;
            worst_absolute = worst_absolute.max(error);
            worst_rate =
                worst_rate.max((actual[i] - expected[i]).abs() * f64::from(1.0f32 / 2040.0));
        }
        if step == 0 {
            base.copy_from_slice(&actual);
        } else {
            for i in 0..10 {
                let delta =
                    17.0 * step as f64 * (i + 1) as f64 * if i % 2 == 0 { 1.0 } else { -1.0 };
                let applied: f64 = (0..10).map(|j| mass[i][j] * (actual[j] - base[j])).sum();
                worst_increment =
                    worst_increment.max((applied - delta).abs() / (1.0 + 170.0 * step as f64));
            }
        }
    }
    eprintln!(
        "mass_predictor_absolute={worst_absolute:e} incremental={worst_increment:e} rate={worst_rate:e} elapsed={:?}",
        start.elapsed()
    );
    assert!(worst_absolute <= 1e-6);
    assert!(worst_increment <= 1e-6);
    assert!(worst_rate <= 1e-5);
}
