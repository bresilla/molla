//! Stage 19 — LBVH `refit_only` (cached topology, fresh AABBs).
//! GPU only.
//!
//! ```bash
//! cargo bench -p molla --bench bench_lbvh_refit --features gpu
//! ```

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::compute::{Backend, Engine};
use molla::kernels::LbvhBroadPhase;
use molla::math::Vec3;

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 32) as u32 as f64) / (u32::MAX as f64)
}

fn random_aabbs(n: usize, extent: f64, half: f64) -> Vec<(Vec3, Vec3)> {
    let mut s: u64 = 0xdead_beef_cafe;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let cx = (lcg(&mut s) - 0.5) * extent;
        let cy = (lcg(&mut s) - 0.5) * extent;
        let cz = (lcg(&mut s) - 0.5) * extent;
        let h = Vec3::splat(half);
        out.push((Vec3::new(cx, cy, cz) - h, Vec3::new(cx, cy, cz) + h));
    }
    out
}

fn jitter(aabbs: &mut [(Vec3, Vec3)], step: f64, seed: &mut u64) {
    for (lo, hi) in aabbs.iter_mut() {
        let dx = (lcg(seed) - 0.5) * step;
        let dy = (lcg(seed) - 0.5) * step;
        let dz = (lcg(seed) - 0.5) * step;
        let d = Vec3::new(dx, dy, dz);
        *lo += d;
        *hi += d;
    }
}

fn bench_refit(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench_lbvh_refit — no GPU: {e}");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut group = c.benchmark_group("lbvh_refit_gpu");
    group.sample_size(10).measurement_time(Duration::from_secs(15));

    for &n in &[1_000usize, 10_000, 100_000] {
        let mut aabbs = random_aabbs(n, 100.0, 0.5);
        let mut lbvh = LbvhBroadPhase::new(
            device.clone(),
            queue.clone(),
            n.next_power_of_two().max(1024),
            (n * 8).max(8192),
        )
        .expect("LbvhBroadPhase::new");
        // Seed the topology cache.
        lbvh.dispatch(&aabbs).expect("seed dispatch");
        let mut s: u64 = 0xfeed_face;
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                jitter(&mut aabbs, 0.05, &mut s);
                let pairs = lbvh.refit_only(black_box(&aabbs)).expect("refit");
                black_box(pairs.a.len())
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_refit);
criterion_main!(benches);
