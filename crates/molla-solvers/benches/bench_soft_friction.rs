//! Stage 33.5 — full soft-contact pipeline (gather + project + warm-
//! start friction) per substep, exercised through `VbdSolver::step`
//! with VT contact margin / sweeps active. CPU and GPU.
//!
//! PERF NOTE (2026-05): the GPU side does NOT actually exercise the
//! Stage 33.5 friction model — the GPU VBD backend skips EE contacts
//! and friction in v1 (see `solvers/vbd.rs`: "EE contacts (Stage
//! 33.1) and friction (Stage 33.5) are CPU-only and only run on the
//! CPU VBD backend"). The GPU group here measures VBD substep +
//! Stage 33.2 on-device VT contact pipeline only. The wall-clock for
//! the GPU side is ALSO non-monotonic in problem size (smoke run:
//! cloth=64 < cloth=8) because the kernel dispatches do not block
//! and `Solver::step` returns once the command buffer is submitted.
//! Without per-iter `device.poll(wait_indefinitely)` criterion is
//! timing host-side dispatch + bind-group construction, NOT actual
//! GPU compute. Real GPU latency lands when something downstream
//! reads the buffers back; for true scaling numbers run the existing
//! `bench_soft_vt_contact_gpu` and consume `download_*` between
//! iterations.
//!
//! ```bash
//! cargo bench -p molla --bench bench_soft_friction
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
#[cfg(feature = "gpu")]
use molla_compute::{Backend, Engine};
use molla_core::WorldId;
use molla_math::{Real, Vec3};
use molla_sim::{Contacts, Model, ModelBuilder};
use molla_solvers::{Solver, VbdSolver};

const SUBSTEPS: usize = 10;
const DT: Real = 1.0 / 240.0;
const MARGIN: Real = 0.05;

fn cube_to_six_tets(c: [u32; 8]) -> [[u32; 4]; 6] {
    [
        [c[0], c[1], c[3], c[7]],
        [c[0], c[3], c[2], c[7]],
        [c[0], c[2], c[6], c[7]],
        [c[0], c[6], c[4], c[7]],
        [c[0], c[4], c[5], c[7]],
        [c[0], c[5], c[1], c[7]],
    ]
}

fn build_scene(cloth_n: u32, tet_n: u32) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.4);

    let cloth_w: Real = 1.0;
    let cloth_y: Real = 1.0;
    let (cloth_first, _, _) = b.add_cloth_grid(
        Vec3::new(-cloth_w * 0.5, cloth_y, -cloth_w * 0.5),
        Vec3::new(cloth_w, 0.0, 0.0),
        Vec3::new(0.0, 0.0, cloth_w),
        cloth_n,
        cloth_n,
        0.05,
        0.02,
        5.0e-6,
        0.05,
        WorldId(0),
    );
    for ix in 0..cloth_n {
        b.pin_particle(cloth_first + ix);
        b.pin_particle(cloth_first + (cloth_n - 1) * cloth_n + ix);
    }
    let dx: Real = 0.10;
    let cube_origin = Vec3::new(
        -dx * (tet_n as Real - 1.0) * 0.5,
        cloth_y + 0.02,
        -dx * (tet_n as Real - 1.0) * 0.5,
    );
    let first_tet_v = cloth_n * cloth_n;
    for kk in 0..tet_n {
        for jj in 0..tet_n {
            for ii in 0..tet_n {
                let p = cube_origin + Vec3::new(ii as Real * dx, jj as Real * dx, kk as Real * dx);
                b.add_particle(p, 0.05, dx * 0.5, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 { first_tet_v + (k * tet_n + j) * tet_n + i };
    let mut tets = Vec::new();
    for kk in 0..tet_n - 1 {
        for jj in 0..tet_n - 1 {
            for ii in 0..tet_n - 1 {
                let c = [
                    v_at(ii, jj, kk),
                    v_at(ii + 1, jj, kk),
                    v_at(ii, jj + 1, kk),
                    v_at(ii + 1, jj + 1, kk),
                    v_at(ii, jj, kk + 1),
                    v_at(ii + 1, jj, kk + 1),
                    v_at(ii, jj + 1, kk + 1),
                    v_at(ii + 1, jj + 1, kk + 1),
                ];
                tets.extend_from_slice(&cube_to_six_tets(c));
            }
        }
    }
    b.add_tet_mesh(&tets, 2.0e5, 0.3, None).unwrap();
    b.build().unwrap()
}

fn bench_friction_cpu(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_friction_cpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(20));
    for &(cloth_n, tet_n) in &[(16u32, 3u32), (32, 3), (32, 5), (64, 5)] {
        let model = build_scene(cloth_n, tet_n);
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = VbdSolver::new_cpu()
            .with_iterations(4)
            .with_vt_contact_margin(MARGIN)
            .with_vt_contact_sweeps(2);
        let label = format!("cloth={cloth_n}_tet={tet_n}");
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &cloth_n,
            |bencher, _| {
                bencher.iter(|| {
                    for _ in 0..SUBSTEPS {
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
                    }
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "gpu")]
fn bench_friction_gpu(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench_soft_friction GPU pass — no GPU: {e}");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut group = c.benchmark_group("soft_friction_gpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(25));
    for &(cloth_n, tet_n) in &[(16u32, 3u32), (32, 3), (32, 5), (64, 5)] {
        let mut model = build_scene(cloth_n, tet_n);
        model.upload_to_device(&device, &queue).unwrap();
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        s_a.upload_to_device(&device, &queue).unwrap();
        s_b.upload_to_device(&device, &queue).unwrap();
        let mut control = model.control();
        control.upload_to_device(&device, &queue).unwrap();
        let contacts = Contacts::with_capacity(0);
        let mut solver = VbdSolver::new_gpu(&engine, &model)
            .expect("VbdSolver::new_gpu")
            .with_iterations(4)
            .with_vt_contact_margin(MARGIN)
            .with_vt_contact_sweeps(2);
        let label = format!("cloth={cloth_n}_tet={tet_n}");
        group.bench_with_input(
            BenchmarkId::from_parameter(label),
            &cloth_n,
            |bencher, _| {
                bencher.iter(|| {
                    for _ in 0..SUBSTEPS {
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
                    }
                });
            },
        );
    }
    group.finish();
}

#[cfg(feature = "gpu")]
criterion_group!(benches, bench_friction_cpu, bench_friction_gpu);
#[cfg(not(feature = "gpu"))]
criterion_group!(benches, bench_friction_cpu);
criterion_main!(benches);
