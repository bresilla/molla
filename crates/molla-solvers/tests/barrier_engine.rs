//! C0 integration — `BarrierSolver` as an engine `Solver`: a cloth
//! grid built through the REAL `ModelBuilder` path (particles, XPBD
//! springs, `SoftSurface`, static plane + sphere shapes) dropped
//! through the `Solver::step` interface. The barrier invariant must
//! hold against obstacles AND self-pairs, and the drape must settle.

use molla_core::{BodyId, WorldId};
use molla_math::{Real, Transform, Vec3};
use molla_sim::{Contacts, ModelBuilder};
use molla_solvers::{BarrierSolver, Solver};

#[test]
fn cloth_grid_drapes_over_sphere_via_solver_trait() {
    let mut b = ModelBuilder::new();
    b.add_cloth_grid(
        Vec3::new(-0.105, 0.15, -0.105),
        Vec3::X * 0.03,
        Vec3::Z * 0.03,
        8,
        8,
        0.02,
        0.005,
        2.5e-3, // compliance → k = 400
        1.5,
        WorldId(0),
    );
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.add_shape_sphere(BodyId::NONE, 0.06, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    assert!(model.spring_count > 0, "cloth springs missing");
    assert!(
        !model.soft_surface_tris.is_empty(),
        "soft-surface triangles missing"
    );

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let contacts = Contacts::with_capacity(4);
    let mut solver = BarrierSolver::new();

    let h: Real = 1.0 / 240.0;
    for step in 0..240 {
        solver
            .step(&model, &s_a, &control, &contacts, h, &mut s_b)
            .unwrap();
        assert!(
            solver.last_min_gap > 0.0,
            "step {step}: INVARIANT violated through the engine path — gap {}",
            solver.last_min_gap
        );
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.particle_q.host().unwrap();
    let qd = s_a.particle_qd.host().unwrap();
    let v_max = qd.iter().map(|v| v.length()).fold(0.0, Real::max);
    let gap = q
        .iter()
        .map(|p| p.y.min(p.length() - 0.06))
        .fold(Real::INFINITY, Real::min);
    eprintln!("engine drape: final gap {gap:.2e}, v_max {v_max:.3e}");
    assert!(v_max < 0.08, "cloth did not settle: {v_max}");
    assert!(
        gap > 0.0 && gap < 2.0e-3,
        "resting gap must sit in the barrier band: {gap:.3e}"
    );
}
