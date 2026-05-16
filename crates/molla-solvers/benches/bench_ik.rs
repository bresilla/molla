//! IK on a 6-DoF revolute chain to N independent targets. CPU.
//!
//! ```bash
//! cargo bench -p molla --bench bench_ik
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};
use molla_core::BodyId;
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3 as V3};
use molla_sim::{BodyParams, Model, ModelBuilder, eval_fk};
use molla_solvers::{IkObjective, IkSolver};

fn build_chain_6dof() -> (Model, usize) {
    let mut b = ModelBuilder::new();
    let mass = 1.0_f64;
    let half = V3::splat(0.04);
    let h2 = half * half;
    let third = mass / 3.0;
    let inertia = Mat3::from_diagonal(V3::new(
        third * (h2.y + h2.z),
        third * (h2.x + h2.z),
        third * (h2.x + h2.y),
    ));
    let axes = [V3::Z, V3::Y, V3::Z, V3::Y, V3::Z, V3::Y];
    let link_lens = [0.3, 0.3, 0.3, 0.3, 0.3, 0.2];
    let mut prev = BodyId::NONE;
    let mut last = 0usize;
    for k in 0..6 {
        let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
            mass,
            com: V3::ZERO,
            inertia,
        }));
        last = body.index();
        let x_p = if k == 0 {
            Transform::IDENTITY
        } else {
            Transform::from_translation(V3::new(0.0, 0.0, link_lens[k - 1]))
        };
        b.add_joint_revolute(prev, body, x_p, Transform::IDENTITY, axes[k])
            .unwrap();
        prev = body;
    }
    (b.build().unwrap(), last)
}

fn bench_ik(c: &mut Criterion) {
    let (model, ee_body) = build_chain_6dof();
    let mut state = model.state().unwrap();
    eval_fk(&model, &mut state).unwrap();
    let ee_at_zero = state.body_q.host().unwrap()[ee_body].position;
    let dof = 6usize;
    let q0 = vec![0.0_f64; dof];

    let mut group = c.benchmark_group("ik_lm_cpu");
    group
        .sample_size(10)
        .measurement_time(Duration::from_secs(15));
    for &n_targets in &[1usize, 16, 256] {
        // Build a sequence of nearby targets the solver can reach.
        let targets: Vec<V3> = (0..n_targets)
            .map(|i| ee_at_zero + V3::new(0.05 * (i as Real * 0.1).sin(), 0.05, -0.05))
            .collect();
        group.bench_with_input(
            BenchmarkId::from_parameter(n_targets),
            &n_targets,
            |bencher, _| {
                bencher.iter(|| {
                    for t in &targets {
                        let solver = IkSolver::new(vec![IkObjective::Position {
                            body: ee_body,
                            point_local: V3::ZERO,
                            target_world: *t,
                            weight: 1.0,
                        }]);
                        let mut s = model.state().unwrap();
                        let res = solver.solve(black_box(&model), &mut s, &q0).unwrap();
                        black_box(res.iters);
                    }
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_ik);
criterion_main!(benches);
