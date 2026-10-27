//! `reduce_contacts` + `reduce_with_history` on N synthetic contacts.
//!
//! ```bash
//! cargo bench -p molla --bench bench_contact_reduction
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_geometry::contact_reduction::{Contact, reduce_contacts as reduce, reduce_with_history};
use molla_math::{Real, Vec3};

fn lcg(seed: &mut u64) -> f64 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 32) as u32 as f64) / (u32::MAX as f64)
}

fn synth(n: usize, pairs: usize) -> Vec<Contact> {
    let mut s: u64 = 0xa11_face;
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let p = (i % pairs) as u32;
        out.push(Contact {
            a_shape: p,
            b_shape: p + pairs as u32,
            id: i as u64,
            point: Vec3::new(
                (lcg(&mut s) - 0.5) * 2.0,
                (lcg(&mut s) - 0.5) * 2.0,
                (lcg(&mut s) - 0.5) * 2.0,
            ),
            normal: Vec3::Y,
            depth: lcg(&mut s) as Real * 0.05,
        });
    }
    out
}

fn bench_reduce(c: &mut Criterion) {
    let mut group = c.benchmark_group("contact_reduction");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(8));
    for &n in &[64usize, 256, 1024] {
        let contacts = synth(n, (n / 16).max(1));
        group.bench_with_input(BenchmarkId::new("reduce_contacts", n), &n, |bencher, _| {
            bencher.iter(|| {
                let kept = reduce(black_box(&contacts), black_box(4));
                black_box(kept.len())
            });
        });
        let prev = synth(n, (n / 16).max(1));
        group.bench_with_input(
            BenchmarkId::new("reduce_with_history", n),
            &n,
            |bencher, _| {
                bencher.iter(|| {
                    let kept = reduce_with_history(
                        black_box(&prev),
                        black_box(&contacts),
                        black_box(4),
                        black_box(0.05),
                    );
                    black_box(kept.len())
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_reduce);
criterion_main!(benches);
