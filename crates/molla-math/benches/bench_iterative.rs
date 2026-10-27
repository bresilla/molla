//! `cg_solve`, `pcg_solve`, `bicgstab_solve` against a 1D Laplacian
//! `(2I - left/right shift)` matrix-free operator. Tests N×N systems
//! at N ∈ {100, 1k, 10k}.
//!
//! ```bash
//! cargo bench -p molla --bench bench_iterative
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_math::Real;
use molla_math::iterative::{JacobiPreconditioner, bicgstab_solve, cg_solve, pcg_solve};

fn laplacian_op(x: &[Real]) -> Vec<Real> {
    let n = x.len();
    let mut y = vec![0.0; n];
    for i in 0..n {
        let xi = x[i];
        let xl = if i > 0 { x[i - 1] } else { 0.0 };
        let xr = if i + 1 < n { x[i + 1] } else { 0.0 };
        y[i] = 2.0 * xi - xl - xr;
    }
    y
}

fn bench_iter(c: &mut Criterion) {
    let mut group = c.benchmark_group("iterative_solvers_cpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));

    for &n in &[100usize, 1_000, 10_000] {
        // RHS: simple linear ramp; x0 = 0.
        let b: Vec<Real> = (0..n).map(|i| (i as Real + 1.0) * 0.001).collect();
        let x0 = vec![0.0_f64; n];
        // Jacobi diag = 2.0 everywhere
        let diag = vec![2.0_f64; n];
        let pre = JacobiPreconditioner::new(&diag);
        let max_iters = (n * 2).min(2000);

        group.bench_with_input(BenchmarkId::new("cg", n), &n, |bencher, _| {
            bencher.iter(|| {
                let r = cg_solve(laplacian_op, black_box(&b), black_box(&x0), 1e-6, max_iters);
                black_box(r.iters)
            });
        });
        group.bench_with_input(BenchmarkId::new("pcg", n), &n, |bencher, _| {
            bencher.iter(|| {
                let r = pcg_solve(
                    laplacian_op,
                    |r| pre.apply(r),
                    black_box(&b),
                    black_box(&x0),
                    1e-6,
                    max_iters,
                );
                black_box(r.iters)
            });
        });
        group.bench_with_input(BenchmarkId::new("bicgstab", n), &n, |bencher, _| {
            bencher.iter(|| {
                let r =
                    bicgstab_solve(laplacian_op, black_box(&b), black_box(&x0), 1e-6, max_iters);
                black_box(r.iters)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_iter);
criterion_main!(benches);
