//! Math primitives at fixed sizes: SVD3, polar, eig3, dense Cholesky
//! solve at 6×6 and 12×12.
//!
//! ```bash
//! cargo bench -p molla --bench bench_math
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_math::dense_solve::dense_chol_solve;
use molla_math::dense3::{eig3_sym, polar_decompose_3, svd3};
use molla_math::{Mat3, Real, Vec3};

fn random_spd(n: usize, seed: u64) -> Vec<Real> {
    // Build A = LLᵀ where L is lower-triangular with positive diag and
    // small off-diagonal random entries. This gives an SPD matrix in
    // row-major layout that `dense_chol_solve` accepts.
    let mut s = seed;
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let r = (s >> 32) as u32 as f64 / u32::MAX as f64;
            l[i * n + j] = if i == j { 1.0 + r } else { r * 0.1 };
        }
    }
    let mut a = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += l[i * n + k] * l[j * n + k];
            }
            a[i * n + j] = sum;
        }
    }
    a
}

fn bench_3x3(c: &mut Criterion) {
    let m = Mat3::from_cols(
        Vec3::new(2.0, 0.5, 0.1),
        Vec3::new(0.5, 1.7, 0.3),
        Vec3::new(0.1, 0.3, 1.4),
    );
    let mut group = c.benchmark_group("math_3x3");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(8));
    group.bench_function("svd3", |bencher| {
        bencher.iter(|| {
            let (u, s, v) = svd3(black_box(m));
            black_box((u.x_axis.x, s.x, v.x_axis.x))
        });
    });
    group.bench_function("polar3", |bencher| {
        bencher.iter(|| {
            let (r, s) = polar_decompose_3(black_box(m));
            black_box((r.x_axis.x, s.x_axis.x))
        });
    });
    group.bench_function("eig3_sym", |bencher| {
        bencher.iter(|| {
            let (vals, vecs) = eig3_sym(black_box(m));
            black_box((vals.x, vecs.x_axis.x))
        });
    });
    group.finish();
}

fn bench_dense_solve(c: &mut Criterion) {
    let mut group = c.benchmark_group("math_dense_chol_solve");
    group
        .sample_size(20)
        .measurement_time(Duration::from_secs(8));
    for &n in &[3usize, 6, 12] {
        let a = random_spd(n, 0xdead_beef);
        let b = vec![1.0_f64; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                let mut a = a.clone();
                let x = dense_chol_solve(&mut a, black_box(&b), n).unwrap();
                black_box(x[0])
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_3x3, bench_dense_solve);
criterion_main!(benches);
