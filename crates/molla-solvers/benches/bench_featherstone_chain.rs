//! Featherstone forward dynamics on an N-link revolute pendulum chain.
//! Compares CPU and GPU. Mirrors `featherstone_cpu_vs_gpu`'s
//! chain-length sweep but pushes N up further.
//!
//! PERF NOTE (2026-05): the GPU branch (`featherstone_chain_gpu/*`)
//! reports non-monotonic times (smoke: M=32 → 660 µs, M=128 → 84 µs)
//! because `FeatherstoneSolver::step` on the GPU returns once the
//! command buffer is submitted — no `map_async` / poll happens here
//! (see `kernels/featherstone.rs` ~line 280). Criterion is timing
//! host-side bind-group / encoder construction, NOT the actual
//! kernel work. CPU times in the same bench ARE reliable. The same
//! pattern affects every other GPU bench whose dispatch is
//! download-free: `bench_xpbd_pile_gpu`, `bench_vbd_cloth_gpu`,
//! `bench_vbd_tets_gpu`, `bench_soft_vt_contact_gpu`,
//! `bench_xpbd_pile_cpu_vs_gpu`, `bench_vbd_cloth_cpu_vs_gpu`,
//! `bench_vbd_tets_cpu_vs_gpu`, `bench_sensor_camera`. The
//! `bench_fft`, `bench_sensor_raycast`, `bench_lbvh_build`, and
//! `bench_lbvh_refit` benches DO sync (their dispatches read back),
//! so their numbers are real.
//!
//! ```bash
//! cargo bench -p molla --bench bench_featherstone_chain
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_core::BodyId;
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder};
use molla_solvers::{FeatherstoneSolver, Solver};

#[cfg(feature = "gpu")]
use molla_compute::{Backend, Engine};

const DT: Real = 0.005;

fn build_chain(m: usize) -> Model {
    let mut b = ModelBuilder::new();
    let mut prev = BodyId::NONE;
    for _ in 0..m {
        let link = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass: 1.0,
            com: V3::ZERO,
            inertia: Mat3::IDENTITY * 0.083,
        }));
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
    b.build().unwrap()
}

fn bench_chain_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("featherstone_chain_cpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));
    for &m in &[2usize, 8, 32, 128] {
        let model = build_chain(m);
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new_cpu();
        group.bench_with_input(BenchmarkId::from_parameter(m), &m, |bencher, _| {
            bencher.iter(|| {
                Solver::step(
                    &mut solver,
                    black_box(&model),
                    black_box(&s_a),
                    black_box(&control),
                    black_box(&contacts),
                    black_box(DT),
                    &mut s_b,
                )
                .unwrap();
                std::mem::swap(&mut s_a, &mut s_b);
            });
        });
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_chain_gpu(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench_featherstone_chain GPU pass — no GPU: {e}");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut group = c.benchmark_group("featherstone_chain_gpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));
    for &m in &[2usize, 8, 32, 128] {
        let mut model = build_chain(m);
        model.upload_to_device(&device, &queue).unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        s_a.upload_to_device(&device, &queue).unwrap();
        s_b.upload_to_device(&device, &queue).unwrap();
        let mut control = model.control();
        control.upload_to_device(&device, &queue).unwrap();
        let contacts = Contacts::with_capacity(0);
        let mut solver = FeatherstoneSolver::new_gpu(&engine, &model).unwrap();

        group.bench_with_input(BenchmarkId::from_parameter(m), &m, |bencher, _| {
            bencher.iter(|| {
                Solver::step(
                    &mut solver,
                    black_box(&model),
                    black_box(&s_a),
                    black_box(&control),
                    black_box(&contacts),
                    black_box(DT),
                    &mut s_b,
                )
                .unwrap();
                std::mem::swap(&mut s_a, &mut s_b);
            });
        });
    }
    group.finish();
}

#[cfg(feature = "gpu")]
criterion_group!(benches, bench_chain_cpu, bench_chain_gpu);
#[cfg(not(feature = "gpu"))]
criterion_group!(benches, bench_chain_cpu);
criterion_main!(benches);
