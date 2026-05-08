//! Featherstone CPU vs GPU benchmark.
//!
//! Compares per-step cost of [`FeatherstoneSolver::step`] on the CPU
//! vs GPU backends across three workload shapes:
//!
//! 1. **chain length scaling** — one articulation with `M` revolute
//!    joints in series. CPU should win at small `M` because the GPU
//!    pays a constant launch + bind-group cost; the gap closes as
//!    `M` grows.
//! 2. **batch scaling** — `N` independent single-joint pendulums in
//!    one scene. Each is its own articulation, so the GPU kernel
//!    runs `N` workgroup invocations in parallel. This is the
//!    canonical RL "many environments" pattern; GPU should pull
//!    ahead clearly past `N ≈ 32`.
//! 3. **mixed** — `N` chains of `M` joints. The realistic robotics
//!    scenario.
//!
//! ```bash
//! cargo bench -p molla --bench featherstone_cpu_vs_gpu --features gpu
//! # or
//! make bench-featherstone-cpu-vs-gpu
//! ```

#![cfg(feature = "gpu")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::compute::{Backend, Engine};
use molla::core::BodyId;
use molla::math::{Mat3, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;

/// Build one chain of `m` revolute joints (parent at world, child at
/// 1m offset down the chain).
fn build_chain(b: &mut ModelBuilder, m: usize) {
    let mut prev = BodyId::NONE;
    for _ in 0..m {
        let link = b.add_body(
            BodyParams::new().with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: Mat3::IDENTITY * 0.083,
            }),
        );
        b.add_joint_revolute(
            prev,
            link,
            Transform::from_translation(if prev == BodyId::NONE {
                V3::ZERO
            } else {
                V3::new(0.0, -0.5, 0.0)
            }),
            Transform::from_translation(V3::new(0.0, 0.5, 0.0)),
            V3::Z,
        )
        .unwrap();
        prev = link;
    }
}

/// `n` independent chains of length `m` in one scene.
fn build_n_chains(n: usize, m: usize) -> Model {
    let mut b = ModelBuilder::new();
    for i in 0..n {
        if i > 0 {
            b.begin_articulation();
        }
        build_chain(&mut b, m);
    }
    b.build().unwrap()
}

fn run_cpu(group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>, model: &Model, label: String) {
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_cpu();
    group.bench_function(BenchmarkId::new("cpu", label), |bencher| {
        bencher.iter(|| {
            solver
                .step(
                    black_box(model),
                    black_box(&s_a),
                    black_box(&control),
                    black_box(&contacts),
                    black_box(0.005),
                    &mut s_b,
                )
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        });
    });
}

fn run_gpu(
    group: &mut criterion::BenchmarkGroup<criterion::measurement::WallTime>,
    engine: &Engine,
    model: &mut Model,
    label: String,
) {
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    model.upload_to_device(&device, &queue).unwrap();
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.upload_to_device(&device, &queue).unwrap();
    s_b.upload_to_device(&device, &queue).unwrap();
    let mut control = model.control();
    control.upload_to_device(&device, &queue).unwrap();
    let contacts = Contacts::with_capacity(0);
    let mut solver = FeatherstoneSolver::new_gpu(engine, model).unwrap();
    group.bench_function(BenchmarkId::new("gpu", label), |bencher| {
        bencher.iter(|| {
            solver
                .step(
                    black_box(model),
                    black_box(&s_a),
                    black_box(&control),
                    black_box(&contacts),
                    black_box(0.005),
                    &mut s_b,
                )
                .unwrap();
            std::mem::swap(&mut s_a, &mut s_b);
        });
    });
}

fn bench_chain_length(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench — no GPU: {e}");
            return;
        }
    };
    let mut group = c.benchmark_group("featherstone/chain_length");
    for &m in &[1usize, 4, 16, 32] {
        let model_cpu = build_n_chains(1, m);
        run_cpu(&mut group, &model_cpu, format!("M={m}"));
        let mut model_gpu = build_n_chains(1, m);
        run_gpu(&mut group, &engine, &mut model_gpu, format!("M={m}"));
    }
    group.finish();
}

fn bench_articulation_batch(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench — no GPU: {e}");
            return;
        }
    };
    let mut group = c.benchmark_group("featherstone/batch");
    for &n in &[1usize, 8, 64, 256, 1024] {
        let model_cpu = build_n_chains(n, 1);
        run_cpu(&mut group, &model_cpu, format!("N={n}"));
        let mut model_gpu = build_n_chains(n, 1);
        run_gpu(&mut group, &engine, &mut model_gpu, format!("N={n}"));
    }
    group.finish();
}

fn bench_mixed(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench — no GPU: {e}");
            return;
        }
    };
    // 64 robots × 7 joints (Franka-like).
    let mut group = c.benchmark_group("featherstone/mixed");
    for &(n, m) in &[(1usize, 7usize), (16, 7), (64, 7), (256, 7)] {
        let label = format!("N={n}xM={m}");
        let model_cpu = build_n_chains(n, m);
        run_cpu(&mut group, &model_cpu, label.clone());
        let mut model_gpu = build_n_chains(n, m);
        run_gpu(&mut group, &engine, &mut model_gpu, label);
    }
    group.finish();
}

criterion_group!(benches, bench_chain_length, bench_articulation_batch, bench_mixed);
criterion_main!(benches);
