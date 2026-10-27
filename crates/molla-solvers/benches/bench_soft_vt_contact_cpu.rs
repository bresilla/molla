//! Stage 33 — soft VT (vertex-triangle) contact gather + project on CPU.
//! Cloth-grid sandwich on top of a tet jelly cube; sizes scale both
//! `cloth_grid_n` and `tet_n`.
//!
//! ```bash
//! cargo bench -p molla --bench bench_soft_vt_contact_cpu
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_core::WorldId;
use molla_geometry::SoftBvh;
use molla_math::{Real, Vec3};
use molla_sim::{Model, ModelBuilder};
use molla_solvers::gather_vt_contacts;
use molla_solvers::vbd_soft_contact::project_vt_contacts;

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

    // Cloth
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

    // Tet jelly cube placed just above the cloth so contacts engage.
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

fn bench_gather(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_vt_gather_cpu");
    group
        .sample_size(15)
        .measurement_time(Duration::from_secs(10));
    for &cloth_n in &[8u32, 16, 32, 64] {
        for &tet_n in &[3u32, 5] {
            let model = build_scene(cloth_n, tet_n);
            let state = model.state().unwrap();
            let bvh = SoftBvh::build(&model, &state);
            let label = format!("cloth={cloth_n}_tet={tet_n}");
            group.bench_with_input(
                BenchmarkId::from_parameter(label),
                &cloth_n,
                |bencher, _| {
                    bencher.iter(|| {
                        let contacts = gather_vt_contacts(
                            black_box(&model),
                            black_box(&state),
                            black_box(&bvh),
                            black_box(MARGIN),
                        );
                        black_box(contacts.len())
                    });
                },
            );
        }
    }
    group.finish();
}

fn bench_project(c: &mut Criterion) {
    let mut group = c.benchmark_group("soft_vt_project_cpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));
    for &cloth_n in &[8u32, 16, 32, 64] {
        for &tet_n in &[3u32, 5] {
            let model = build_scene(cloth_n, tet_n);
            let label = format!("cloth={cloth_n}_tet={tet_n}");
            group.bench_with_input(
                BenchmarkId::from_parameter(label),
                &cloth_n,
                |bencher, _| {
                    bencher.iter(|| {
                        let mut state = model.state().unwrap();
                        let mut bvh = SoftBvh::build(&model, &state);
                        let q_pre = state.particle_q.host().unwrap().to_vec();
                        let qd_pre = state.particle_qd.host().unwrap().to_vec();
                        let prev: Vec<_> = Vec::new();
                        let res = project_vt_contacts(
                            black_box(&model),
                            &mut state,
                            &mut bvh,
                            MARGIN,
                            4,
                            1.0 / 240.0,
                            &prev,
                            &q_pre,
                            &qd_pre,
                        )
                        .unwrap();
                        black_box(res.len())
                    });
                },
            );
        }
    }
    group.finish();
}

criterion_group!(benches, bench_gather, bench_project);
criterion_main!(benches);
