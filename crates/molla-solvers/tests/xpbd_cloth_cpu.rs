//! Default CPU cloth correctness tests for Tier-2 soft/cloth gates.
//!
//! GPU parity lives in `xpbd_particles_gpu.rs`; these tests keep the
//! small CPU oracle lane honest for pinned particles and strain limits.

use molla_core::{BodyId, WorldId};
use molla_geometry::CollisionPipeline;
use molla_math::{Real, Transform, Vec3};
use molla_sim::{Contacts, Model, ModelBuilder};
use molla_solvers::{Solver, XpbdSolver};

const N: u32 = 6;

fn build_pinned_corner_cloth() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.2);
    let size = 1.0_f64;
    let origin = Vec3::new(-0.5, 1.5, -0.5);
    let u_axis = Vec3::new(size, 0.0, 0.0);
    let v_axis = Vec3::new(0.0, 0.0, size);
    let (first, _, _) = b.add_cloth_grid(
        origin,
        u_axis,
        v_axis,
        N,
        N,
        0.05,
        0.02,
        5.0e-5,
        0.05,
        WorldId(0),
    );
    b.pin_particle(first);
    b.pin_particle(first + N - 1);
    b.pin_particle(first + (N - 1) * N);
    b.pin_particle(first + (N - 1) * N + N - 1);
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().unwrap()
}

fn build_high_res_pinned_edge_cloth(n: u32) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.25);
    let size = 1.6_f64;
    let origin = Vec3::new(-0.8, 1.8, -0.8);
    let (first, _, _) = b.add_cloth_grid(
        origin,
        Vec3::new(size, 0.0, 0.0),
        Vec3::new(0.0, 0.0, size),
        n,
        n,
        0.03,
        0.015,
        8.0e-6,
        0.08,
        WorldId(0),
    );
    // Pin one edge so the rest of the cloth can sag and collide with the floor.
    for ix in 0..n {
        b.pin_particle(first + ix);
    }
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    b.build().unwrap()
}

fn build_zero_gravity_rest_cloth() -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    let size = 1.0_f64;
    let origin = Vec3::new(-0.5, 1.5, -0.5);
    b.add_cloth_grid(
        origin,
        Vec3::new(size, 0.0, 0.0),
        Vec3::new(0.0, 0.0, size),
        N,
        N,
        0.05,
        0.02,
        5.0e-5,
        0.05,
        WorldId(0),
    );
    b.build().unwrap()
}

fn build_zero_gravity_uniform_cloth(damping: Real) -> Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(damping);
    b.set_world_gravity(WorldId(0), Vec3::ZERO);
    let size = 1.0_f64;
    b.add_cloth_grid(
        Vec3::new(-0.5, 1.5, -0.5),
        Vec3::new(size, 0.0, 0.0),
        Vec3::new(0.0, 0.0, size),
        N,
        N,
        0.05,
        0.02,
        5.0e-5,
        0.05,
        WorldId(0),
    );
    b.build().unwrap()
}

fn run_cpu(model: &Model, steps: usize) -> (Vec<Vec3>, Vec<Vec3>) {
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let initial = s_a.particle_q.host().unwrap().to_vec();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(12);
    let dt: Real = 0.005;
    for _ in 0..steps {
        pipeline.collide(model, &s_a, &mut contacts).unwrap();
        solver
            .step(model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    (initial, s_a.particle_q.host().unwrap().to_vec())
}

fn run_cpu_with_initial_particle_velocity(
    model: &Model,
    initial_velocity: Vec3,
    steps: usize,
) -> (Vec<Vec3>, Vec<Vec3>, Vec<Vec3>) {
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    for v in s_a.particle_qd.host_mut().unwrap() {
        *v = initial_velocity;
    }
    let initial = s_a.particle_q.host().unwrap().to_vec();
    let control = model.control();
    let contacts = Contacts::with_capacity(0);
    let mut solver = XpbdSolver::new_cpu().with_iterations(12);
    let dt: Real = 0.005;
    for _ in 0..steps {
        solver
            .step(model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    (
        initial,
        s_a.particle_q.host().unwrap().to_vec(),
        s_a.particle_qd.host().unwrap().to_vec(),
    )
}

#[test]
fn cpu_cloth_rest_shape_stays_stationary_without_forces() {
    let model = build_zero_gravity_rest_cloth();
    let (initial, final_q) = run_cpu(&model, 120);

    let mut max_drift: Real = 0.0;
    for (i, (a, b)) in initial.iter().zip(final_q.iter()).enumerate() {
        assert!(
            b.is_finite(),
            "rest-shape cloth particle {i} became non-finite: {b:?}"
        );
        max_drift = max_drift.max((*b - *a).length());
    }

    eprintln!("XPBD cloth zero-force rest-shape max drift = {max_drift:.3e} m");
    assert!(
        max_drift < 1.0e-10,
        "zero-force cloth rest shape drifted by {max_drift:.3e} m"
    );
}

#[test]
fn cpu_cloth_linear_damping_reduces_uniform_motion_without_distortion() {
    let damping: Real = 8.0;
    let dt: Real = 0.005;
    let steps = 80usize;
    let v0 = Vec3::new(1.2, -0.3, 0.7);
    let model = build_zero_gravity_uniform_cloth(damping);

    let (initial, final_q, final_qd) = run_cpu_with_initial_particle_velocity(&model, v0, steps);

    let factor = (1.0 - damping * dt).max(0.0);
    let expected_velocity = v0 * factor.powi(steps as i32);
    let expected_displacement = if (1.0 - factor).abs() < 1.0e-12 {
        v0 * (dt * steps as Real)
    } else {
        v0 * (dt * factor * (1.0 - factor.powi(steps as i32)) / (1.0 - factor))
    };

    let mut max_velocity_error: Real = 0.0;
    let mut max_displacement_error: Real = 0.0;
    for (i, ((q0, q1), v1)) in initial
        .iter()
        .zip(final_q.iter())
        .zip(final_qd.iter())
        .enumerate()
    {
        assert!(
            q1.is_finite() && v1.is_finite(),
            "damped cloth particle {i} became non-finite: q={q1:?}, qd={v1:?}"
        );
        max_velocity_error = max_velocity_error.max((*v1 - expected_velocity).length());
        max_displacement_error =
            max_displacement_error.max((*q1 - *q0 - expected_displacement).length());
    }

    eprintln!(
        "XPBD cloth damping expected |v|={:.5}, max velocity error={max_velocity_error:.3e}, \
         max displacement error={max_displacement_error:.3e}",
        expected_velocity.length()
    );
    assert!(
        expected_velocity.length() < v0.length() * 0.05,
        "test setup did not damp the uniform velocity enough"
    );
    assert!(
        max_velocity_error < 1.0e-10,
        "linear damping velocity mismatch: {max_velocity_error:.3e}"
    );
    assert!(
        max_displacement_error < 1.0e-9,
        "uniform damped cloth distorted or drifted unexpectedly: {max_displacement_error:.3e}"
    );
}

#[test]
fn cpu_pinned_cloth_corners_stay_fixed_and_finite() {
    let model = build_pinned_corner_cloth();
    let (initial, final_q) = run_cpu(&model, 120);
    let corners = [0, N - 1, (N - 1) * N, (N - 1) * N + N - 1];

    for (i, p) in final_q.iter().enumerate() {
        assert!(
            p.x.is_finite() && p.y.is_finite() && p.z.is_finite(),
            "particle {i} became non-finite: {p:?}"
        );
    }

    for c in corners {
        let i = c as usize;
        let drift = (final_q[i] - initial[i]).length();
        assert!(
            drift < 1.0e-10,
            "pinned corner {i} drifted by {drift:.3e} m"
        );
    }
}

#[test]
fn cpu_cloth_respects_spring_strain_limit() {
    let model = build_pinned_corner_cloth();
    let (_initial, final_q) = run_cpu(&model, 180);
    let spring_a = model.spring_a.host().unwrap();
    let spring_b = model.spring_b.host().unwrap();
    let rest = model.spring_rest_length.host().unwrap();
    let max_stretch = model.spring_max_stretch.host().unwrap();

    let mut worst_ratio = 0.0;
    for i in 0..model.spring_count {
        let a = spring_a[i] as usize;
        let b = spring_b[i] as usize;
        let len = (final_q[a] - final_q[b]).length();
        let limit = rest[i] * (1.0 + max_stretch[i]);
        let ratio = if rest[i] > 0.0 { len / rest[i] } else { 1.0 };
        if ratio > worst_ratio {
            worst_ratio = ratio;
        }
        assert!(
            len <= limit + 2.0e-3,
            "spring {i} stretched too far: len={len:.6}, limit={limit:.6}, ratio={ratio:.4}"
        );
    }
    assert!(
        worst_ratio <= 1.06,
        "worst spring stretch ratio {worst_ratio:.4} exceeded strain gate"
    );
}

#[test]
fn high_resolution_cpu_cloth_stays_finite_and_strain_limited() {
    let n = 14u32;
    let model = build_high_res_pinned_edge_cloth(n);
    let (initial, final_q) = run_cpu(&model, 240);

    for (i, p) in final_q.iter().enumerate() {
        assert!(
            p.is_finite(),
            "high-res cloth particle {i} became non-finite: {p:?}"
        );
        assert!(
            p.y > -0.05 && p.y < 2.2,
            "high-res cloth particle {i} escaped expected bounds: {p:?}"
        );
    }

    for ix in 0..n {
        let i = ix as usize;
        let drift = (final_q[i] - initial[i]).length();
        assert!(
            drift < 1.0e-10,
            "high-res pinned edge particle {i} drifted by {drift:.3e}"
        );
    }

    let spring_a = model.spring_a.host().unwrap();
    let spring_b = model.spring_b.host().unwrap();
    let rest = model.spring_rest_length.host().unwrap();
    let max_stretch = model.spring_max_stretch.host().unwrap();
    let mut worst_ratio = 0.0;
    for i in 0..model.spring_count {
        let a = spring_a[i] as usize;
        let b = spring_b[i] as usize;
        let len = (final_q[a] - final_q[b]).length();
        let limit = rest[i] * (1.0 + max_stretch[i]);
        let ratio = if rest[i] > 0.0 { len / rest[i] } else { 1.0 };
        worst_ratio = ratio.max(worst_ratio);
        assert!(
            len <= limit + 3.0e-3,
            "high-res spring {i} stretched too far: len={len:.6}, limit={limit:.6}, ratio={ratio:.4}"
        );
    }
    assert!(
        worst_ratio <= 1.10,
        "high-res cloth worst stretch ratio {worst_ratio:.4} exceeded stability gate"
    );
}
