//! Stage 19 — LBVH full-build over N synthetic AABBs (random + clustered).
//! GPU only.
//!
//! ```bash
//! cargo bench -p molla --bench bench_lbvh_build --features gpu
//! ```

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::compute::{Backend, Engine};
use molla::kernels::LbvhBroadPhase;
use molla::math::Vec3;

/// Deterministic LCG so we don't pull `rand` for benches.
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

fn clustered_aabbs(n: usize) -> Vec<(Vec3, Vec3)> {
    let side = ((n as f64).cbrt().ceil() as usize).max(1);
    let mut out = Vec::with_capacity(n);
    let half = Vec3::splat(0.4);
    'outer: for ix in 0..side {
        for iy in 0..side {
            for iz in 0..side {
                if out.len() >= n {
                    break 'outer;
                }
                let c = Vec3::new(
                    ix as f64 * 1.0,
                    iy as f64 * 1.0,
                    iz as f64 * 1.0,
                );
                out.push((c - half, c + half));
            }
        }
    }
    out
}

fn bench_build(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench_lbvh_build — no GPU: {e}");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut group = c.benchmark_group("lbvh_build_gpu");
    group.sample_size(10).measurement_time(Duration::from_secs(15));

    for &n in &[1_000usize, 10_000, 100_000] {
        for (kind, aabbs) in [
            ("random", random_aabbs(n, 100.0, 0.5)),
            ("clustered", clustered_aabbs(n)),
        ] {
            let mut lbvh = LbvhBroadPhase::new(
                device.clone(),
                queue.clone(),
                n.next_power_of_two().max(1024),
                (n * 8).max(8192),
            )
            .expect("LbvhBroadPhase::new");
            let label = format!("{kind}_n={n}");
            group.bench_with_input(BenchmarkId::from_parameter(label), &n, |bencher, _| {
                bencher.iter(|| {
                    let pairs = lbvh.dispatch(black_box(&aabbs)).expect("dispatch");
                    black_box(pairs.a.len())
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_build);
criterion_main!(benches);
