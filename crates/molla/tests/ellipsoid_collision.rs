//! Stage-30 ellipsoid primitive tests.
//!
//! Closed-form pairs (Ellipsoid-vs-Sphere, Ellipsoid-vs-Plane) check
//! depth + normal against analytic ground truth; the GJK-fallback
//! pair (Ellipsoid-vs-Box) just asserts a sensible non-zero contact
//! is produced and lives within tolerance of the analytic answer.
//!
//! Builder smoke test asserts `add_shape_ellipsoid` produces a body
//! whose BVH AABB matches the rotated semi-axis envelope.

use molla::core::BodyId;
use molla::geometry::{shape_world_aabb, support_with_hulls, CollisionPipeline};
use molla::math::{Quat, Real, Transform, Vec3};
use molla::prelude::*;
use molla::sim::ShapeGeometry;

/// Helper: read out the first contact (depth, normal) from `contacts`.
fn first_contact(contacts: &molla::sim::Contacts) -> Option<(Real, Vec3)> {
    if contacts.count == 0 {
        return None;
    }
    let depth = contacts.depth.host().expect("depth")[0];
    let normal = contacts.normal.host().expect("normal")[0];
    Some((depth, normal))
}

#[test]
fn ellipsoid_support_axis_poles() {
    // Sanity: support along ±X/±Y/±Z lands on the matching pole.
    // Same coverage as the in-module unit test; a re-cast at the
    // public-API level here makes Stage-30's surface contract
    // visible in `cargo test --test ellipsoid_collision`.
    let geom = ShapeGeometry::Ellipsoid {
        half_axes: Vec3::new(1.5, 0.7, 0.3),
    };
    let xf = Transform::IDENTITY;
    for (dir, want) in &[
        (Vec3::X, Vec3::new(1.5, 0.0, 0.0)),
        (-Vec3::X, Vec3::new(-1.5, 0.0, 0.0)),
        (Vec3::Y, Vec3::new(0.0, 0.7, 0.0)),
        (-Vec3::Y, Vec3::new(0.0, -0.7, 0.0)),
        (Vec3::Z, Vec3::new(0.0, 0.0, 0.3)),
        (-Vec3::Z, Vec3::new(0.0, 0.0, -0.3)),
    ] {
        let p = support_with_hulls(geom, &xf, *dir, &[]);
        assert!(
            (p - *want).length() < 1e-9,
            "support({dir:?}) = {p:?} expected {want:?}"
        );
    }
}

#[test]
fn ellipsoid_sphere_overlap_along_x_axis() {
    // Ellipsoid half_axes=(1, 0.5, 0.5) at origin (so X-pole at
    // (1, 0, 0)). Sphere of radius 0.3 centred at (1.2, 0, 0):
    //   surface separation = 1.2 − 1.0 = 0.2
    //   penetration depth  = 0.3 − 0.2 = 0.1
    //   contact normal a→b = +X (gradient at (1, 0, 0) is +X).
    let mut b = ModelBuilder::new();
    let body_e = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_ellipsoid(body_e, Vec3::new(1.0, 0.5, 0.5), Transform::IDENTITY)
        .unwrap();
    let body_s = b.add_body(BodyParams::new().with_position(Vec3::new(1.2, 0.0, 0.0)));
    b.add_shape_sphere(body_s, 0.3, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();

    let mut pipeline = CollisionPipeline::new();
    let mut contacts = molla::sim::Contacts::with_capacity(8);
    pipeline.collide(&model, &state, &mut contacts).unwrap();
    let (depth, normal) = first_contact(&contacts).expect("expected contact");
    let depth_err: Real = (depth - 0.1).abs();
    let normal_err: Real = (normal.normalize() - Vec3::X).length();
    assert!(depth_err < 1e-6, "depth = {depth} (expected ~0.1)");
    assert!(normal_err < 1e-6, "normal = {normal:?} (expected +X)");
}

#[test]
fn ellipsoid_sphere_separated_no_contact() {
    let mut b = ModelBuilder::new();
    let body_e = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_ellipsoid(body_e, Vec3::new(1.0, 0.5, 0.5), Transform::IDENTITY)
        .unwrap();
    let body_s = b.add_body(BodyParams::new().with_position(Vec3::new(3.0, 0.0, 0.0)));
    b.add_shape_sphere(body_s, 0.3, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();

    let mut pipeline = CollisionPipeline::new();
    let mut contacts = molla::sim::Contacts::with_capacity(8);
    pipeline.collide(&model, &state, &mut contacts).unwrap();
    assert_eq!(contacts.count, 0, "expected no contacts");
}

#[test]
fn ellipsoid_sphere_oblate_y_face() {
    // Oblate ellipsoid (1, 0.3, 1) — Y-pole at (0, 0.3, 0). Sphere
    // r=0.2 sits at (0, 0.4, 0); separation = 0.1, depth = 0.1.
    let mut b = ModelBuilder::new();
    let body_e = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_ellipsoid(body_e, Vec3::new(1.0, 0.3, 1.0), Transform::IDENTITY)
        .unwrap();
    let body_s = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 0.4, 0.0)));
    b.add_shape_sphere(body_s, 0.2, Transform::IDENTITY).unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();

    let mut pipeline = CollisionPipeline::new();
    let mut contacts = molla::sim::Contacts::with_capacity(8);
    pipeline.collide(&model, &state, &mut contacts).unwrap();
    assert_eq!(contacts.count, 1);
    let (depth, normal) = first_contact(&contacts).unwrap();
    let depth_err: Real = (depth - 0.1).abs();
    assert!(depth_err < 1e-6, "depth = {depth}");
    let n_err: Real = (normal - Vec3::Y).length();
    assert!(n_err < 1e-6, "normal = {normal:?} (expected +Y)");
}

#[test]
fn ellipsoid_plane_touches_at_lowest_pole() {
    // Ellipsoid (0.6, 0.4, 0.5) centred at (0, 0.3, 0); plane at
    // y=0 with normal +Y. Lowest pole is at y = 0.3 − 0.4 = −0.1
    // → depth = 0.1, normal a→b (ellipsoid → plane) = −Y.
    let mut b = ModelBuilder::new();
    let body_e = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 0.3, 0.0)));
    b.add_shape_ellipsoid(body_e, Vec3::new(0.6, 0.4, 0.5), Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();

    let mut pipeline = CollisionPipeline::new();
    let mut contacts = molla::sim::Contacts::with_capacity(8);
    pipeline.collide(&model, &state, &mut contacts).unwrap();
    assert_eq!(contacts.count, 1);
    let (depth, normal) = first_contact(&contacts).unwrap();
    let depth_err: Real = (depth - 0.1).abs();
    assert!(depth_err < 1e-6, "depth = {depth}");
    // Normal points from ellipsoid into plane → −Y.
    let n_err: Real = (normal + Vec3::Y).length();
    assert!(n_err < 1e-6, "normal = {normal:?}");
}

#[test]
fn ellipsoid_plane_above_no_contact() {
    let mut b = ModelBuilder::new();
    let body_e = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 5.0, 0.0)));
    b.add_shape_ellipsoid(body_e, Vec3::new(0.6, 0.4, 0.5), Transform::IDENTITY)
        .unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();

    let mut pipeline = CollisionPipeline::new();
    let mut contacts = molla::sim::Contacts::with_capacity(8);
    pipeline.collide(&model, &state, &mut contacts).unwrap();
    assert_eq!(contacts.count, 0);
}

#[test]
fn ellipsoid_box_shallow_overlap() {
    // GJK + EPA fallback. Ellipsoid (0.5, 0.5, 0.5) (a sphere, in
    // disguise) at (0, 1.4, 0) versus a unit box at origin. Sphere
    // surface dips below the +Y face by 0.1 (1.0 + 0.5 − 1.4 = 0.1),
    // so we expect depth ≈ 0.1, normal ≈ ±Y. Tolerance is loose
    // (~5e-3) because EPA halts on a polytope-face threshold.
    let mut b = ModelBuilder::new();
    let body_e = b.add_body(BodyParams::new().with_position(Vec3::new(0.0, 1.4, 0.0)));
    b.add_shape_ellipsoid(body_e, Vec3::new(0.5, 0.5, 0.5), Transform::IDENTITY)
        .unwrap();
    let body_b = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_box(body_b, Vec3::splat(1.0), Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();

    let mut pipeline = CollisionPipeline::new();
    let mut contacts = molla::sim::Contacts::with_capacity(8);
    pipeline.collide(&model, &state, &mut contacts).unwrap();
    let (depth, normal) = first_contact(&contacts).expect("expected GJK contact");
    let depth_err: Real = (depth - 0.1).abs();
    assert!(
        depth_err < 5e-3,
        "depth = {depth} (expected ≈ 0.1, EPA tolerance ~5e-3)"
    );
    // Normal should be mostly along ±Y. Convention: a → b
    // (ellipsoid → box). With the ellipsoid above the box, that's
    // pointing DOWN (into the box) → −Y.
    let dot_y = normal.dot(Vec3::Y);
    assert!(
        dot_y.abs() > 0.95,
        "normal not mostly along ±Y: {normal:?}"
    );
}

#[test]
fn ellipsoid_box_separated_no_contact() {
    let mut b = ModelBuilder::new();
    let body_e = b.add_body(BodyParams::new().with_position(Vec3::new(5.0, 0.0, 0.0)));
    b.add_shape_ellipsoid(body_e, Vec3::new(0.4, 0.4, 0.4), Transform::IDENTITY)
        .unwrap();
    let body_b = b.add_body(BodyParams::new().with_position(Vec3::ZERO));
    b.add_shape_box(body_b, Vec3::splat(1.0), Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();

    let mut pipeline = CollisionPipeline::new();
    let mut contacts = molla::sim::Contacts::with_capacity(8);
    pipeline.collide(&model, &state, &mut contacts).unwrap();
    assert_eq!(contacts.count, 0);
}

#[test]
fn add_shape_ellipsoid_aabb_axis_aligned() {
    // Builder smoke test: identity rotation → world AABB equals
    // ±half_axes around the body's pose. No body-frame transform
    // funny business.
    let half_axes = Vec3::new(1.5, 0.7, 0.3);
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::new(2.0, 3.0, 4.0)));
    b.add_shape_ellipsoid(body, half_axes, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();

    let aabb = shape_world_aabb(&model, &state, 0, 0.0);
    let want_min = Vec3::new(2.0, 3.0, 4.0) - half_axes;
    let want_max = Vec3::new(2.0, 3.0, 4.0) + half_axes;
    assert!(
        (aabb.min - want_min).length() < 1e-9,
        "aabb.min = {:?}, expected {want_min:?}",
        aabb.min
    );
    assert!(
        (aabb.max - want_max).length() < 1e-9,
        "aabb.max = {:?}, expected {want_max:?}",
        aabb.max
    );
}

#[test]
fn add_shape_ellipsoid_aabb_rotated_90deg_about_z() {
    // Ellipsoid (a, b, c) = (1.5, 0.7, 0.3) rotated 90° about Z.
    // Local +X axis maps to world +Y, local +Y to world −X. So the
    // world AABB half-extents become (b, a, c) = (0.7, 1.5, 0.3).
    let half_axes = Vec3::new(1.5, 0.7, 0.3);
    let mut b = ModelBuilder::new();
    let body = b.add_body(
        BodyParams::new()
            .with_pose(Transform::new(Vec3::ZERO, Quat::from_rotation_z(std::f64::consts::FRAC_PI_2))),
    );
    b.add_shape_ellipsoid(body, half_axes, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();

    let aabb = shape_world_aabb(&model, &state, 0, 0.0);
    let want_h = Vec3::new(0.7, 1.5, 0.3);
    assert!(
        (aabb.min + want_h).length() < 1e-9,
        "aabb.min = {:?}, expected {:?}",
        aabb.min,
        -want_h
    );
    assert!(
        (aabb.max - want_h).length() < 1e-9,
        "aabb.max = {:?}, expected {want_h:?}",
        aabb.max
    );
}

#[test]
fn add_shape_ellipsoid_rejects_non_positive_axes() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new());
    let r = b.add_shape_ellipsoid(body, Vec3::new(1.0, 0.0, 1.0), Transform::IDENTITY);
    assert!(r.is_err(), "zero axis should be rejected");
    let r = b.add_shape_ellipsoid(body, Vec3::new(-1.0, 1.0, 1.0), Transform::IDENTITY);
    assert!(r.is_err(), "negative axis should be rejected");
}
