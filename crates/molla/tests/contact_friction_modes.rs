//! Rolling + torsional friction tests. Both are CPU-only for now —
//! GPU XPBD will pick them up when the WGSL kernel is updated.
//!
//! Coulomb tangential friction already damps any angular velocity
//! whose contact-point projection is non-zero (via the lever-arm
//! effect: tangential impulse at offset `r_contact` produces a
//! torque about the COM). The tests below isolate rolling /
//! torsional damping by **comparing two simulations** that differ
//! only in the friction-mode coefficient — anything that's down to
//! Coulomb appears in both runs and cancels out.

use molla::core::BodyId;
use molla::geometry::CollisionPipeline;
use molla::math::{Mat3, SpatialInertia, SpatialVector, Transform, Vec3 as V3};
use molla::prelude::*;

fn sphere_inertia(mass: f64, r: f64) -> Mat3 {
    let i = (2.0 / 5.0) * mass * r * r;
    Mat3::from_diagonal(V3::new(i, i, i))
}

fn build_sphere_on_plane(coulomb: f64, rolling: f64, torsional: f64) -> molla::sim::Model {
    let mut b = ModelBuilder::new();
    b.set_linear_damping(0.0);
    b.set_angular_damping(0.0);
    let radius = 0.1_f64;
    let mass = 1.0_f64;
    let inertia = sphere_inertia(mass, radius);
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia { mass, com: V3::ZERO, inertia })
            .with_position(V3::new(0.0, radius, 0.0)),
    );
    let sphere = b.add_shape_sphere(body, radius, Transform::IDENTITY).unwrap();
    let plane = b
        .add_shape_plane(BodyId::NONE, V3::Y, Transform::IDENTITY)
        .unwrap();
    b.set_shape_material(sphere, coulomb, 0.0).unwrap();
    b.set_shape_material(plane, coulomb, 0.0).unwrap();
    b.set_shape_friction_modes(sphere, rolling, torsional).unwrap();
    b.build().unwrap()
}

fn run_steady(model: &molla::sim::Model, w0: V3, n_steps: usize) -> V3 {
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    s_a.body_qd.host_mut().unwrap()[0] = SpatialVector::new(V3::ZERO, w0);
    let control = model.control();
    let mut contacts = molla::sim::Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(8);
    let dt = 0.005_f64;
    for _ in 0..n_steps {
        pipeline.collide(model, &s_a, &mut contacts).unwrap();
        solver
            .step(model, &s_a, &control, &contacts, dt, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    s_a.body_qd.host().unwrap()[0].angular
}

/// With Coulomb = 0, μ_rolling > 0 is the ONLY thing that can damp
/// horizontal-axis spin. Without rolling friction the sphere spins
/// forever (Coulomb is the only existing damping path; we've zeroed it).
#[test]
fn rolling_friction_damps_horizontal_axis_spin_when_coulomb_zero() {
    let m_zero = build_sphere_on_plane(0.0, 0.0, 0.0);
    let m_with = build_sphere_on_plane(0.0, 0.5, 0.0);
    let w_zero = run_steady(&m_zero, V3::new(10.0, 0.0, 0.0), 60);
    let w_with = run_steady(&m_with, V3::new(10.0, 0.0, 0.0), 60);
    eprintln!(
        "horizontal spin |ω| after 0.3 s w/ Coulomb=0: roll=0 → {:.4}, roll=0.5 → {:.4}",
        w_zero.length(),
        w_with.length()
    );
    assert!(
        w_zero.length() > 9.5,
        "no Coulomb + no rolling: should keep spin, got |ω|={}",
        w_zero.length()
    );
    assert!(
        w_with.length() < w_zero.length() - 1.0,
        "rolling friction did nothing: zero={}, with={}",
        w_zero.length(),
        w_with.length()
    );
}

/// μ_torsional > 0 damps normal-axis spin. Coulomb has no purchase
/// on this DOF (the contact-point projection of pure normal-axis
/// rotation is zero), so this test stays clean even with non-zero
/// Coulomb friction.
#[test]
fn torsional_friction_damps_spin_along_normal() {
    let m_zero = build_sphere_on_plane(0.7, 0.0, 0.0);
    let m_with = build_sphere_on_plane(0.7, 0.0, 0.5);
    let w_zero = run_steady(&m_zero, V3::new(0.0, 10.0, 0.0), 60);
    let w_with = run_steady(&m_with, V3::new(0.0, 10.0, 0.0), 60);
    eprintln!(
        "normal-axis spin |ω.y| after 0.3 s: μ_tors=0 → {:.4}, μ_tors=0.5 → {:.4}",
        w_zero.y.abs(),
        w_with.y.abs()
    );
    assert!(
        w_zero.y.abs() > 9.5,
        "Coulomb shouldn't damp normal-axis spin, but |ω.y|={}",
        w_zero.y.abs()
    );
    assert!(
        w_with.y.abs() < w_zero.y.abs() - 0.3,
        "torsional friction not stronger than μ=0 baseline"
    );
}
