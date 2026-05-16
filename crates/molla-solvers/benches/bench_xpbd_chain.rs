//! Chain of N rope particles connected via XPBD distance constraints.
//! Stresses the particle solver path (predict + project + recover).
//!
//! ```bash
//! cargo bench -p molla --bench bench_xpbd_chain
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_core::WorldId;
use molla_math::{Real, Vec3};
use molla_sim::{Contacts, Model, ModelBuilder};
use molla_solvers::{Solver, XpbdSolver};

const SUBSTEPS: usize = 60;
const DT: Real = 1.0 / 240.0;

fn build_chain(n: usize) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.05);
    let n = n as u32;
    let (first, _, _) = b.add_cable_polyline(
        Vec3::new(0.0, 5.0, 0.0),
        Vec3::new((n as Real) * 0.05, 5.0, 0.0),
        n,
        0.05,  // particle mass
        0.02,  // radius
        1.0e5, // stiffness
        0.05,  // max stretch
        WorldId(0),
    );
    // Pin the first particle so the chain swings.
    b.pin_particle(first);
    b.build().unwrap()
}

fn bench_chain(c: &mut Criterion) {
    let mut group = c.benchmark_group("xpbd_chain_cpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));
    for &n in &[16usize, 64, 256, 1024] {
        let model = build_chain(n);
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

criterion_group!(benches, bench_chain);
criterion_main!(benches);
