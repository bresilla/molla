//! N point lookups against a pre-built `SparseGrid`. CPU.
//!
//! ```bash
//! cargo bench -p molla --bench bench_sparse_grid_query
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_geometry::sparse_grid::SparseGrid;

fn lcg(seed: &mut u64) -> u32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*seed >> 32) as u32
}

fn build_grid(n: usize, extent: i32) -> (SparseGrid, Vec<[i32; 3]>) {
    let mut g = SparseGrid::new(0.0);
    let mut s: u64 = 0xfeed_face_cafe;
    let mut coords = Vec::with_capacity(n);
    for _ in 0..n {
        let x = (lcg(&mut s) as i32) % extent;
        let y = (lcg(&mut s) as i32) % extent;
        let z = (lcg(&mut s) as i32) % extent;
        g.insert(x, y, z, 1.0);
        coords.push([x, y, z]);
    }
    (g, coords)
}

fn bench_query(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_grid_query_cpu");
    group
        .sample_size(15)
        .measurement_time(Duration::from_secs(10));
    for &n in &[1_000usize, 10_000, 100_000] {
        let (grid, coords) = build_grid(n, 4096);
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                let mut acc = 0.0_f32;
                for c in &coords {
                    acc += grid.lookup(c[0], c[1], c[2]);
                }
                black_box(acc)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_query);
criterion_main!(benches);
