//! FFT benches: Stockham radix-2 1D N=4096, 2D 64×64, 3D 32³ on the
//! GPU; CPU 1D N=4096 via `rustfft` for comparison.
//!
//! PERF NOTE (2026-05): `FftKernel::dispatch_*` blocks on a full
//! `map_async` + `device.poll(wait_indefinitely)` readback inside
//! the per-iter loop (see `kernels/fft.rs` ~line 553). On a 4096-
//! point 1D FFT the GPU compute is < 10 µs but the round-trip
//! latency is ~1.2 ms — i.e. CPU rustfft beats GPU FFT by ~300×
//! at N=4096 (smoke: 3.9 µs vs 1.2 ms). The 3D 32³ case at 54 ms
//! is closer to compute-bound but still pays one readback per
//! sample. To compare raw kernel throughput, batch many FFTs into
//! one submit and read back once — needs an engine API change.
//!
//! ```bash
//! cargo bench -p molla --bench bench_fft --features gpu
//! ```

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::compute::{Backend, Engine};
use molla::kernels::{Complex32, FftKernel};
use rustfft::{num_complex::Complex32 as RfComplex, FftPlanner};

fn rand_complex(n: usize, seed: u64) -> Vec<Complex32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let re = (s >> 32) as u32 as f32 / u32::MAX as f32 - 0.5;
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let im = (s >> 32) as u32 as f32 / u32::MAX as f32 - 0.5;
            Complex32::new(re, im)
        })
        .collect()
}

fn bench_fft_gpu(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench_fft GPU pass — no GPU: {e}");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = FftKernel::new(&device);

    let mut group = c.benchmark_group("fft_gpu");
    group.sample_size(15).measurement_time(Duration::from_secs(10));

    // 1D N=4096
    {
        let n = 4096;
        let input = rand_complex(n, 0xa1);
        let mut output = vec![Complex32::new(0.0, 0.0); n];
        group.bench_with_input(BenchmarkId::from_parameter("1d_4096"), &n, |bencher, _| {
            bencher.iter(|| {
                kernel
                    .dispatch_1d(&device, &queue, black_box(&input), &mut output, n, false)
                    .unwrap();
                black_box(output[0])
            });
        });
    }
    // 2D 64×64
    {
        let h = 64;
        let w = 64;
        let n = h * w;
        let input = rand_complex(n, 0xb2);
        let mut output = vec![Complex32::new(0.0, 0.0); n];
        group.bench_with_input(BenchmarkId::from_parameter("2d_64x64"), &n, |bencher, _| {
            bencher.iter(|| {
                kernel
                    .dispatch_2d(&device, &queue, black_box(&input), &mut output, h, w, false)
                    .unwrap();
                black_box(output[0])
            });
        });
    }
    // 3D 32×32×32
    {
        let d = 32;
        let h = 32;
        let w = 32;
        let n = d * h * w;
        let input = rand_complex(n, 0xc3);
        let mut output = vec![Complex32::new(0.0, 0.0); n];
        group.bench_with_input(BenchmarkId::from_parameter("3d_32x32x32"), &n, |bencher, _| {
            bencher.iter(|| {
                kernel
                    .dispatch_3d(
                        &device,
                        &queue,
                        black_box(&input),
                        &mut output,
                        d,
                        h,
                        w,
                        false,
                    )
                    .unwrap();
                black_box(output[0])
            });
        });
    }
    group.finish();
}

fn bench_fft_cpu(c: &mut Criterion) {
    // CPU oracle via rustfft for the 1D 4096 size only.
    let mut group = c.benchmark_group("fft_cpu");
    group.sample_size(20).measurement_time(Duration::from_secs(8));
    let n = 4096;
    let mut planner = FftPlanner::<f32>::new();
    let plan = planner.plan_fft_forward(n);
    let input = rand_complex(n, 0xa1);
    group.bench_with_input(BenchmarkId::from_parameter("1d_4096_rustfft"), &n, |bencher, _| {
        bencher.iter(|| {
            let mut buf: Vec<RfComplex> =
                input.iter().map(|c| RfComplex::new(c.re, c.im)).collect();
            plan.process(&mut buf);
            black_box(buf[0])
        });
    });
    group.finish();
}

criterion_group!(benches, bench_fft_gpu, bench_fft_cpu);
criterion_main!(benches);
