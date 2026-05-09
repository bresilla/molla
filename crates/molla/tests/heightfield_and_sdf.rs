//! Heightfield + SDF primitives — drop tests + AABB sanity.

use molla::core::BodyId;
use molla::geometry::{shape_world_aabb, CollisionPipeline};
use molla::math::{Mat3, SpatialInertia, Transform, Vec3 as V3};
use molla::prelude::*;
use molla::sim::{Heightfield, SdfShape};
use molla::solvers::Solver as _;

fn sphere_inertia(mass: f64, r: f64) -> Mat3 {
    let i = (2.0 / 5.0) * mass * r * r;
    Mat3::from_diagonal(V3::new(i, i, i))
}

#[test]
fn heightfield_local_aabb_matches_extents() {
    let hf = Heightfield {
        nx: 4,
        nz: 4,
        cell_size: 1.0,
        origin: V3::new(-1.5, 0.0, -1.5),
        heights: vec![0.0; 16],
    };
    let (lo, hi) = hf.local_aabb();
    // origin (-1.5, 0, -1.5), 4 samples, spacing 1 → max = (-1.5 + 3, 0, -1.5 + 3) = (1.5, 0, 1.5).
    assert!((lo - V3::new(-1.5, 0.0, -1.5)).length() < 1e-9);
    assert!((hi - V3::new(1.5, 0.0, 1.5)).length() < 1e-9);
}

#[test]
fn sphere_settles_on_flat_heightfield() {
    let mut b = ModelBuilder::new();
    let r = 0.2_f64;
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: sphere_inertia(1.0, r),
            })
            .with_position(V3::new(0.0, 2.0, 0.0)),
    );
    b.add_shape_sphere(body, r, Transform::IDENTITY).unwrap();
    // 4×4 flat heightfield at y = 0, covering x,z ∈ [-2, 2].
    let hf = Heightfield {
        nx: 5,
        nz: 5,
        cell_size: 1.0,
        origin: V3::new(-2.0, 0.0, -2.0),
        heights: vec![0.0; 25],
    };
    let field_id = b.add_heightfield(hf);
    b.add_shape_heightfield(BodyId::NONE, field_id, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = molla::sim::Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(8);
    let dt = 0.005_f64;
    for _ in 0..400 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver.step(&model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let y = s_a.body_q.host().unwrap()[body.index()].position.y;
    eprintln!("sphere on heightfield: y = {y:.4} m");
    // Sphere centre should sit ~r above the surface with a bit of
    // contact-margin slack. Allow a generous band.
    assert!(
        (y - r).abs() < 0.05,
        "sphere didn't settle on heightfield: y={y}"
    );
}

#[test]
fn sphere_settles_outside_torus_sdf() {
    let mut b = ModelBuilder::new();
    let r = 0.15_f64;
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia {
                mass: 1.0,
                com: V3::ZERO,
                inertia: sphere_inertia(1.0, r),
            })
            // Drop the sphere onto the top of a torus ring at y=1.
            .with_position(V3::new(0.5, 2.0, 0.0)),
    );
    b.add_shape_sphere(body, r, Transform::IDENTITY).unwrap();
    // Torus: major=0.5, minor=0.1, sitting at world (0, 1, 0).
    let sdf_id = b.add_sdf(SdfShape::Torus { major: 0.5, minor: 0.1 });
    b.add_shape_sdf(
        BodyId::NONE,
        sdf_id,
        Transform::from_translation(V3::new(0.0, 1.0, 0.0)),
    )
    .unwrap();
    // Floor as a backstop.
    b.add_shape_plane(BodyId::NONE, V3::Y, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();

    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = molla::sim::Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new_cpu().with_iterations(8);
    let dt = 0.005_f64;
    for _ in 0..400 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver.step(&model, &s_a, &control, &contacts, dt, &mut s_b).unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }
    let p = s_a.body_q.host().unwrap()[body.index()].position;
    eprintln!("sphere near torus settled at {p:?}");
    // Distance from sphere centre to torus surface = signed distance.
    // Want it ≥ -tol (i.e. NOT deeply inside the torus tube).
    let p_local = p - V3::new(0.0, 1.0, 0.0);
    let r_xz = (p_local.x * p_local.x + p_local.z * p_local.z).sqrt();
    let q = V3::new(r_xz - 0.5, p_local.y, 0.0);
    let signed = (q.x * q.x + q.y * q.y).sqrt() - 0.1;
    let pen = signed - r;
    assert!(
        pen > -0.02,
        "sphere penetrating torus by {} m at p={:?}",
        -pen,
        p
    );
}

#[test]
fn sdf_world_aabb_grows_with_torus_dims() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_spatial_inertia(SpatialInertia {
        mass: 1.0,
        com: V3::ZERO,
        inertia: Mat3::IDENTITY,
    }));
    let sdf_id = b.add_sdf(SdfShape::Torus { major: 0.5, minor: 0.1 });
    b.add_shape_sdf(body, sdf_id, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();
    let aabb = shape_world_aabb(&model, &state, 0, 0.0);
    let extent = aabb.max - aabb.min;
    // Torus extent: 2·(major+minor) on x/z, 2·minor on y.
    assert!((extent.x - 2.0 * 0.6).abs() < 1e-9);
    assert!((extent.z - 2.0 * 0.6).abs() < 1e-9);
    assert!((extent.y - 2.0 * 0.1).abs() < 1e-9);
}
