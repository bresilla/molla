//! Insert N voxels (random + clustered) into a NanoVDB-style
//! `SparseGrid`. CPU.
//!
//! ```bash
//! cargo bench -p molla --bench bench_sparse_grid_insert
//! ```

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::geometry::sparse_grid::SparseGrid;

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*seed >> 32) as u32
}

fn random_coords(n: usize, extent: i32) -> Vec<[i32; 3]> {
    let mut s: u64 = 0xa11_face;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        let x = (lcg(&mut s) as i32) % extent;
        let y = (lcg(&mut s) as i32) % extent;
        let z = (lcg(&mut s) as i32) % extent;
        out.push([x, y, z]);
    }
    out
}

fn clustered_coords(n: usize) -> Vec<[i32; 3]> {
    let side = ((n as f64).cbrt().ceil() as i32).max(1);
    let mut out = Vec::with_capacity(n);
    'outer: for ix in 0..side {
        for iy in 0..side {
            for iz in 0..side {
                if out.len() >= n {
                    break 'outer;
                }
                out.push([ix, iy, iz]);
            }
        }
    }
    out
}

fn bench_insert(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_grid_insert_cpu");
    group.sample_size(10).measurement_time(Duration::from_secs(15));
    for &n in &[1_000usize, 10_000, 100_000] {
        for (kind, coords) in [
            ("random", random_coords(n, 4096)),
            ("clustered", clustered_coords(n)),
        ] {
            let label = format!("{kind}_n={n}");
            group.bench_with_input(BenchmarkId::from_parameter(label), &n, |bencher, _| {
                bencher.iter(|| {
                    let mut g = SparseGrid::new(0.0);
                    for c in &coords {
                        g.insert(c[0], c[1], c[2], 1.0);
                    }
                    black_box(g.leaf_count())
                });
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_insert);
criterion_main!(benches);
