//! Stage 33 — soft VT contact gather + project on GPU. Run via the
//! `VbdSolver::new_gpu` step path with VT contacts enabled (margin
//! and sweeps non-zero), which routes through the GPU
//! `SoftVtContactKernel`.
//!
//! ```bash
//! cargo bench -p molla --bench bench_soft_vt_contact_gpu --features gpu
//! ```

#![cfg(feature = "gpu")]

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::core::WorldId;
use molla::math::Real;
use molla::prelude::*;

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
    let first_tet_v = (cloth_n * cloth_n) as u32;
    for kk in 0..tet_n {
        for jj in 0..tet_n {
            for ii in 0..tet_n {
                let p = cube_origin
                    + Vec3::new(ii as Real * dx, jj as Real * dx, kk as Real * dx);
                b.add_particle(p, 0.05, dx * 0.5, WorldId(0));
            }
        }
    }
    let v_at = |i: u32, j: u32, k: u32| -> u32 {
        first_tet_v + (k * tet_n + j) * tet_n + i
    };
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

fn bench_soft_vt_gpu(c: &mut Criterion) {
    let engine = match Engine::new(Backend::AutoGpu) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("skipping bench_soft_vt_contact_gpu — no GPU: {e}");
            return;
        }
    };
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut group = c.benchmark_group("soft_vt_step_gpu");
    group.sample_size(10).measurement_time(Duration::from_secs(20));

    for &cloth_n in &[8u32, 16, 32, 64] {
        for &tet_n in &[3u32, 5] {
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
            group.bench_with_input(BenchmarkId::from_parameter(label), &cloth_n, |bencher, _| {
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
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_soft_vt_gpu);
criterion_main!(benches);
