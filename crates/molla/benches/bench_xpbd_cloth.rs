//! N×N XPBD cloth grid, run 60 substeps. CPU. Stresses distance-
//! constraint projection scaling on a structured 2D mesh.
//!
//! ```bash
//! cargo bench -p molla --bench bench_xpbd_cloth
//! ```

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use molla::core::WorldId;
use molla::prelude::*;

const SUBSTEPS: usize = 60;
const DT: Real = 1.0 / 240.0;

fn build_cloth(n: u32) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.05);
    let (first, _, _) = b.add_cloth_grid(
        Vec3::new(-0.5, 1.0, -0.5),
        Vec3::new(1.0, 0.0, 0.0),
        Vec3::new(0.0, 0.0, 1.0),
        n,
        n,
        0.05,   // particle mass
        0.02,   // radius
        1.0e-5, // compliance
        0.05,   // max stretch
        WorldId(0),
    );
    // Pin the four corners so the cloth hangs.
    b.pin_particle(first);
    b.pin_particle(first + n - 1);
    b.pin_particle(first + (n - 1) * n);
    b.pin_particle(first + n * n - 1);
    b.build().unwrap()
}

fn bench_cloth(c: &mut Criterion) {
    let mut group = c.benchmark_group("xpbd_cloth_cpu");
    group.sample_size(10).measurement_time(Duration::from_secs(20));
    for &n in &[16u32, 32, 64, 128] {
        let model = build_cloth(n);
        let mut s_a = model.state().unwrap();
        let mut s_b = model.state().unwrap();
        let control = model.control();
        let contacts = Contacts::with_capacity(0);
        let mut solver = XpbdSolver::new_cpu().with_iterations(4).with_substeps(1);

        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |bencher, _| {
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
    group.finish();
}

criterion_group!(benches, bench_cloth);
criterion_main!(benches);
