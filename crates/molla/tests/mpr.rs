//! Stage 28 — MPR narrow-phase integration tests.
//!
//! Cross-checks MPR against GJK + EPA on the cases where both
//! converge, and shows MPR survives a deep-penetration regime where
//! EPA's polytope can struggle.

use molla::geometry::{epa, gjk, mpr_collide, GjkResult, ShapeSupport};
use molla::math::{Quat, Transform, Vec3};
use molla::sim::ShapeGeometry;

/// Tolerance for "MPR vs GJK+EPA agree" — the two algorithms have
/// independent termination criteria so a few permil disagreement is
/// expected even on well-conditioned inputs.
const AGREEMENT_TOL: f64 = 1e-3;

#[test]
fn mpr_matches_gjk_epa_on_shallow_box_box() {
    // Two unit cubes overlapping by 0.2 along +X.
    let xf_a = Transform::IDENTITY;
    let xf_b = Transform::from_translation(Vec3::new(1.8, 0.0, 0.0));
    let geom = ShapeGeometry::Box { half_extents: Vec3::splat(1.0) };

    // MPR.
    let a = ShapeSupport::new(geom, &xf_a);
    let b = ShapeSupport::new(geom, &xf_b);
    let mpr = mpr_collide(&a, &b, 1e-8, 64).expect("MPR should report overlap");

    // GJK + EPA.
    let gjk_result = gjk(geom, &xf_a, geom, &xf_b);
    let simplex = match gjk_result {
        GjkResult::Overlap(s) => s,
        _ => panic!("GJK should report overlap"),
    };
    let epa_contact = epa(geom, &xf_a, geom, &xf_b, &simplex).expect("EPA");

    // Depth agreement.
    let dd = (mpr.depth - epa_contact.depth).abs();
    assert!(
        dd < AGREEMENT_TOL,
        "depth disagreement: MPR {} vs EPA {} (Δ={})",
        mpr.depth,
        epa_contact.depth,
        dd
    );

    // Normal agreement (or anti-aligned — both algorithms can pick
    // either orientation depending on tiny fp tie-breaks; the
    // canonical molla convention is A → B, but this test only
    // demands the axes line up).
    let dot = mpr.normal.dot(epa_contact.normal).abs();
    assert!(
        (dot - 1.0).abs() < 1e-2,
        "normal disagreement: MPR {:?} vs EPA {:?} (|·|={})",
        mpr.normal,
        epa_contact.normal,
        dot
    );

    // And both should report the +X axis (within a sign).
    assert!(
        (mpr.normal.x.abs() - 1.0).abs() < 1e-2,
        "MPR normal not on +X: {:?}",
        mpr.normal
    );
}

#[test]
fn mpr_matches_gjk_epa_on_shallow_sphere_sphere() {
    // Two unit spheres centres 1.5 apart (overlap 0.5 along +X).
    let xf_a = Transform::IDENTITY;
    let xf_b = Transform::from_translation(Vec3::new(1.5, 0.0, 0.0));
    let geom = ShapeGeometry::Sphere { radius: 1.0 };

    let a = ShapeSupport::new(geom, &xf_a);
    let b = ShapeSupport::new(geom, &xf_b);
    let mpr = mpr_collide(&a, &b, 1e-8, 64).expect("MPR overlap");

    let gjk_result = gjk(geom, &xf_a, geom, &xf_b);
    let simplex = match gjk_result {
        GjkResult::Overlap(s) => s,
        _ => panic!("GJK overlap"),
    };
    let epa_contact = epa(geom, &xf_a, geom, &xf_b, &simplex).expect("EPA");

    // Spheres are easy for both algorithms — agreement should be
    // tighter than the box-box case.
    let dd = (mpr.depth - epa_contact.depth).abs();
    assert!(
        dd < AGREEMENT_TOL,
        "depth: MPR {} vs EPA {} (Δ={})",
        mpr.depth,
        epa_contact.depth,
        dd
    );
    let dot = mpr.normal.dot(epa_contact.normal).abs();
    assert!((dot - 1.0).abs() < 1e-2, "normal alignment: |·|={dot}");
}

#[test]
fn mpr_handles_deep_penetration() {
    // Two unit cubes 50% interpenetrating along +X. Centres are
    // half_extent = 1.0 apart, so the overlap is 1.0 in each axis;
    // expected penetration along +X is 1.0.
    //
    // EPA on this configuration sometimes diverges depending on the
    // initial GJK simplex; MPR's portal refinement should remain
    // stable. The assertion: MPR returns a valid manifold with
    // sensible depth + normal — we don't compare against EPA here
    // because the "EPA may diverge" caveat is precisely why we're
    // shipping MPR as an alternative.
    let xf_a = Transform::IDENTITY;
    let xf_b = Transform::from_translation(Vec3::new(1.0, 0.0, 0.0));
    let geom = ShapeGeometry::Box { half_extents: Vec3::splat(1.0) };

    let a = ShapeSupport::new(geom, &xf_a);
    let b = ShapeSupport::new(geom, &xf_b);
    let mpr = mpr_collide(&a, &b, 1e-6, 64)
        .expect("MPR must return a manifold for deep penetration");

    // Depth is bounded above by the box's diagonal (≈ 2√3) and
    // below by 0; the analytic separation along any axis is 1.0.
    assert!(
        mpr.depth > 0.5 && mpr.depth < 2.5,
        "depth out of range: {}",
        mpr.depth
    );
    // Normal must be unit-length and finite.
    assert!(mpr.normal.is_finite(), "normal not finite: {:?}", mpr.normal);
    let nlen = mpr.normal.length();
    assert!(
        (nlen - 1.0).abs() < 1e-3,
        "normal not unit-length: |n|={}",
        nlen
    );
    // For two axis-aligned cubes overlapping equally on every axis
    // the contact normal should land on a cardinal direction (the
    // algorithm picks one — we accept any).
    let largest = mpr.normal.x.abs().max(mpr.normal.y.abs()).max(mpr.normal.z.abs());
    assert!(
        largest > 0.95,
        "normal should be near a cardinal axis, got {:?}",
        mpr.normal
    );
}

#[test]
fn mpr_handles_very_deep_sphere_penetration() {
    // Two unit spheres almost concentric (centres 0.1 apart) — depth
    // should be ~ 1.9. EPA's polytope can be touchy here because the
    // initial simplex degenerates; MPR shouldn't care.
    let xf_a = Transform::IDENTITY;
    let xf_b = Transform::from_translation(Vec3::new(0.1, 0.0, 0.0));
    let geom = ShapeGeometry::Sphere { radius: 1.0 };

    let a = ShapeSupport::new(geom, &xf_a);
    let b = ShapeSupport::new(geom, &xf_b);
    let mpr = mpr_collide(&a, &b, 1e-6, 128).expect("MPR must converge");

    assert!(
        (mpr.depth - 1.9).abs() < 5e-2,
        "expected depth ≈ 1.9, got {}",
        mpr.depth
    );
    // Normal must be finite + unit + roughly along +X.
    assert!(mpr.normal.is_finite());
    let nlen = mpr.normal.length();
    assert!((nlen - 1.0).abs() < 1e-3);
    assert!(
        mpr.normal.x.abs() > 0.95,
        "normal should align with +X, got {:?}",
        mpr.normal
    );
}

#[test]
fn mpr_returns_none_on_separated_shapes() {
    // Cubes separated by a gap of 0.5 along +X.
    let xf_a = Transform::IDENTITY;
    let xf_b = Transform::from_translation(Vec3::new(2.5, 0.0, 0.0));
    let geom = ShapeGeometry::Box { half_extents: Vec3::splat(1.0) };

    let a = ShapeSupport::new(geom, &xf_a);
    let b = ShapeSupport::new(geom, &xf_b);
    assert!(mpr_collide(&a, &b, 1e-6, 64).is_none());
}

#[test]
fn mpr_returns_none_on_grazing_contact() {
    // Cubes touching at the +X face but not overlapping.
    // Distance between centres = 2.0 = half_extent_a + half_extent_b.
    // Strictly speaking this is a "touching" configuration — depth 0.
    // The contract for `mpr_collide` is "None when shapes are
    // strictly disjoint"; on a knife-edge touch we accept either
    // None or Some(depth ≈ 0).
    let xf_a = Transform::IDENTITY;
    let xf_b = Transform::from_translation(Vec3::new(2.0 + 1e-3, 0.0, 0.0));
    let geom = ShapeGeometry::Box { half_extents: Vec3::splat(1.0) };

    let a = ShapeSupport::new(geom, &xf_a);
    let b = ShapeSupport::new(geom, &xf_b);
    let res = mpr_collide(&a, &b, 1e-6, 64);
    // Slightly separated → must be None.
    assert!(
        res.is_none(),
        "expected None for separated cubes, got {:?}",
        res
    );
}

#[test]
fn mpr_works_on_rotated_capsule_box() {
    // Capsule rotated 30° about Y, box at origin; centres 0.5 apart
    // along +X. They overlap; we just want a finite manifold.
    let xf_a = Transform::IDENTITY;
    let xf_b = Transform::new(
        Vec3::new(0.5, 0.0, 0.0),
        Quat::from_rotation_y(std::f64::consts::PI / 6.0),
    );
    let box_g = ShapeGeometry::Box { half_extents: Vec3::splat(0.5) };
    let cap_g = ShapeGeometry::Capsule { half_height: 0.5, radius: 0.3 };

    let a = ShapeSupport::new(box_g, &xf_a);
    let b = ShapeSupport::new(cap_g, &xf_b);
    let m = mpr_collide(&a, &b, 1e-6, 64).expect("overlap expected");

    assert!(m.depth > 0.0);
    assert!((m.normal.length() - 1.0).abs() < 1e-3);
    assert!(m.point_a.is_finite() && m.point_b.is_finite());
}
