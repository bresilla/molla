//! Stage 39F — SpGEMM wall-time: host-symbolic v1 vs full-GPU
//! Stage 39F. Both paths produce the same CSR; we report the speedup
//! at sizes 64, 256, 1024, 4096.
//!
//! ```bash
//! nixVulkan cargo bench -p molla --features gpu --bench bench_spgemm_gpu
//! ```

use std::time::{Duration, Instant};

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_compute::Engine;
use molla_kernels::{BsrAlgebraKernel, BsrGpu};
use molla_math::Real;
use molla_math::sparse::BsrMatrix;

fn lcg(seed: u64) -> impl FnMut() -> Real {
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(1);
    move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 32) as u32) as Real) / (u32::MAX as Real) - 0.5
    }
}

fn random_banded(n: usize, seed: u64) -> BsrMatrix {
    let mut rng = lcg(seed);
    let mut triplets: Vec<(usize, usize, Real)> = Vec::new();
    for i in 0..n {
        triplets.push((i, i, 2.0 + rng().abs()));
    }
    let band = 3usize;
    for i in 0..n {
        for d in 1..=band {
            let j = (i + d) % n;
            if j == i {
                continue;
            }
            let v = (rng()) * 0.1;
            triplets.push((i, j, v));
            triplets.push((j, i, v));
        }
    }
    BsrMatrix::from_scalar_triplets(n, n, &triplets).expect("triplets")
}

fn bench_spgemm(c: &mut Criterion) {
    let engine = match Engine::new_gpu() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench_spgemm_gpu — no GPU adapter: {e:?}");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let kernel = BsrAlgebraKernel::new(device.clone(), queue.clone());

    let mut group = c.benchmark_group("spgemm_gpu_vs_host_symbolic");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(8));

    // Side-by-side timing table — criterion's own report is per-bench,
    // so we also print a plain-text speedup summary at the end.
    let mut summary: Vec<(usize, f64, f64)> = Vec::new();

    for &n in &[64usize, 256, 1024, 4096] {
        let a_cpu = random_banded(n, 0x1111 + n as u64);
        let b_cpu = random_banded(n, 0x2222 + n as u64);
        let a_gpu = BsrGpu::upload(&device, &queue, &a_cpu).expect("a upload");
        let b_gpu = BsrGpu::upload(&device, &queue, &b_cpu).expect("b upload");

        // Warm-up + capture single-shot wall-times for the summary
        // table. We measure here (not via Criterion's iterator) so the
        // summary captures the FULL pipeline including device.poll for
        // GPU readback — the user wants speedup, not throughput.
        let warmup = 2;
        for _ in 0..warmup {
            let _ = kernel.spgemm_host_symbolic(&a_gpu, &b_gpu).unwrap();
            let _ = kernel.spgemm_gpu_symbolic(&a_gpu, &b_gpu).unwrap();
        }

        let trials = 5;
        let mut host_total = 0.0_f64;
        let mut gpu_total = 0.0_f64;
        for _ in 0..trials {
            let t0 = Instant::now();
            let _c = kernel.spgemm_host_symbolic(&a_gpu, &b_gpu).unwrap();
            host_total += t0.elapsed().as_secs_f64();

            let t1 = Instant::now();
            let _c = kernel.spgemm_gpu_symbolic(&a_gpu, &b_gpu).unwrap();
            gpu_total += t1.elapsed().as_secs_f64();
        }
        let host_ms = (host_total / trials as f64) * 1000.0;
        let gpu_ms = (gpu_total / trials as f64) * 1000.0;
        summary.push((n, host_ms, gpu_ms));

        group.bench_with_input(BenchmarkId::new("host_symbolic", n), &n, |b, _| {
            b.iter(|| {
                let c = kernel.spgemm_host_symbolic(&a_gpu, &b_gpu).unwrap();
                black_box(c.nnz);
            });
        });
        group.bench_with_input(BenchmarkId::new("gpu_symbolic", n), &n, |b, _| {
            b.iter(|| {
                let c = kernel.spgemm_gpu_symbolic(&a_gpu, &b_gpu).unwrap();
                black_box(c.nnz);
            });
        });
    }

    group.finish();

    eprintln!("\n=== Stage 39F SpGEMM wall-time summary (n × n banded × banded) ===");
    eprintln!(
        "{:>6}  {:>14}  {:>14}  {:>10}",
        "n", "host_sym (ms)", "gpu_sym (ms)", "speedup"
    );
    for (n, host_ms, gpu_ms) in &summary {
        let speedup = host_ms / gpu_ms;
        eprintln!(
            "{:>6}  {:>14.3}  {:>14.3}  {:>10.2}x",
            n, host_ms, gpu_ms, speedup
        );
    }
    eprintln!();
}

criterion_group!(benches, bench_spgemm);
criterion_main!(benches);
