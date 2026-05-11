//! Stage 38.1 — wall-time benchmark for CPU diff-MPM forward + backward.
//!
//! ```bash
//! cargo bench -p molla --bench bench_mpm_diff
//! ```
//!
//! Measures three things:
//!   - `forward_only`      — forward pass without tape recording.
//!   - `forward_with_tape` — forward pass with full per-step tape.
//!   - `backward`          — reverse pass from a seeded `grad_pos_final`.
//!
//! The ratio `backward / forward_only` measures the differentiability
//! overhead. For Stage 38.1's 64-particle / 30-step / 16³ grid demo,
//! it lands at ~3× (the SVD adjoint dominates).

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::autograd::mpm_adjoint::{backward, centroid_loss_and_grad};
use molla::compute::mpm_cpu_diff::{forward, forward_with_tape, MpmCpuConfig, ParticleState};
use molla::math::{Real, Vec3};

fn make_cube(n_per_side: usize) -> ParticleState {
    let mut pos = Vec::with_capacity(n_per_side * n_per_side * n_per_side);
    let spacing = 0.08;
    let center = Vec3::new(0.0, 0.40, 0.0);
    let half = (n_per_side as Real - 1.0) * 0.5;
    for i in 0..n_per_side {
        for j in 0..n_per_side {
            for k in 0..n_per_side {
                let off = Vec3::new(
                    (i as Real - half) * spacing,
                    (j as Real - half) * spacing,
                    (k as Real - half) * spacing,
                );
                pos.push(center + off);
            }
        }
    }
    ParticleState::new(&pos, &[])
}

fn demo_cfg() -> MpmCpuConfig {
    MpmCpuConfig {
        grid_size: 16,
        cell_size: 0.08,
        grid_origin: [-0.64, -0.08, -0.64],
        gravity_y: -9.81,
        youngs: 3.0e4,
        poisson: 0.3,
        particle_volume: 1.0e-3,
        particle_mass: 1.0e-2,
        floor_y: 0.0,
        floor_friction: 0.4,
    }
}

fn bench_mpm_diff(c: &mut Criterion) {
    let cfg = demo_cfg();
    let dt: Real = 5.0e-4;
    let steps = 10;
    let init = make_cube(4); // 64 particles

    let mut group = c.benchmark_group("mpm_cpu_diff");
    group.measurement_time(Duration::from_secs(3));

    group.bench_function(BenchmarkId::new("forward_only", "64p_10steps"), |b| {
        b.iter(|| {
            let out = forward(&cfg, black_box(&init), dt, steps);
            black_box(out);
        });
    });

    group.bench_function(BenchmarkId::new("forward_with_tape", "64p_10steps"), |b| {
        b.iter(|| {
            let tape = forward_with_tape(&cfg, black_box(&init), dt, steps);
            black_box(tape);
        });
    });

    // Pre-build a tape outside the timed region so `backward` measures
    // only the reverse pass.
    let tape = forward_with_tape(&cfg, &init, dt, steps);
    let target = Vec3::ZERO;
    let interior: Vec<usize> = (0..init.n()).collect();
    let (_, grad_pos_final) =
        centroid_loss_and_grad(&tape.final_state.pos, target, &interior);

    group.bench_function(BenchmarkId::new("backward", "64p_10steps"), |b| {
        b.iter(|| {
            let adj = backward(&tape, black_box(&grad_pos_final));
            black_box(adj.grad_youngs(cfg.poisson));
        });
    });

    group.finish();
}

criterion_group!(benches, bench_mpm_diff);
criterion_main!(benches);
