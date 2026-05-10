//! `dh_dq_analytic` reverse-mode for an N-link revolute chain. CPU.
//!
//! ```bash
//! cargo bench -p molla --bench bench_autograd_featherstone
//! ```

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::autograd::{FeatherstoneRevoluteChain, JointKind, RevoluteLink};
use molla::math::{Mat3, Real, Transform, Vec3};

fn build_n_chain(n: usize) -> FeatherstoneRevoluteChain {
    let mass = 1.0_f64;
    let l = 0.5;
    let i = mass * l * l / 12.0;
    let inertia = Mat3::from_diagonal(Vec3::splat(i));
    let mut links = Vec::with_capacity(n);
    for k in 0..n {
        let x_p = if k == 0 {
            Transform::IDENTITY
        } else {
            Transform::from_translation(Vec3::new(0.0, l, 0.0))
        };
        links.push(RevoluteLink {
            kind: JointKind::Revolute,
            axis_local: Vec3::Z,
            x_p,
            mass,
            com_local: Vec3::new(0.0, l * 0.5, 0.0),
            inertia_local: inertia,
        });
    }
    FeatherstoneRevoluteChain {
        links,
        gravity: Vec3::new(0.0, -9.81, 0.0),
        damping: 0.05,
        dt: 0.005,
        horizon: 1,
        ee_local: Vec3::new(0.0, l, 0.0),
        projections: Vec::new(),
    }
}

fn bench_dh_dq(c: &mut Criterion) {
    let mut group = c.benchmark_group("autograd_featherstone_dh_dq");
    group.sample_size(15).measurement_time(Duration::from_secs(10));
    for &n in &[3usize, 6, 10, 20] {
        let chain = build_n_chain(n);
        let q = vec![0.1_f64; n];
        let qd = vec![0.0_f64; n];
        let qdd = vec![0.0_f64; n];
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
            bencher.iter(|| {
                let dh = chain
                    .dh_dq_analytic(black_box(&q), black_box(&qd), black_box(&qdd))
                    .expect("dh_dq_analytic");
                black_box(dh.len() as Real)
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_dh_dq);
criterion_main!(benches);
