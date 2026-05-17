//! Minkowski Portal Refinement (MPR) — alternative narrow-phase to
//! GJK + EPA.
//!
//! Reference: Newhouse, *Game Programming Gems 7* (2008) ch. 2.5
//! "XenoCollide: Complex Collision Made Simple"; original MPR paper
//! Snethen 2007. Newton's `geometry/mpr.py` is the structural model.
//!
//! MPR works directly on the Minkowski difference `A − B` like GJK,
//! but instead of building a simplex around the origin and then
//! growing it (the EPA step) it constructs a *portal* — a triangle
//! on the boundary of the Minkowski difference whose extension hits
//! the origin — and refines it until the portal face is tight against
//! the origin. Two big practical wins over GJK + EPA:
//!
//! 1. **No separate penetration-depth pass.** GJK only answers the
//!    yes/no overlap question and hands the simplex to EPA; MPR
//!    returns depth + normal in one algorithm.
//! 2. **Robust on deep penetration.** EPA's polytope expansion can
//!    diverge when the GJK simplex is highly distorted (e.g. shapes
//!    50 % interpenetrating, or concentric cylinders). MPR's portal
//!    refinement is a single smooth descent and tends to converge
//!    where EPA stalls.
//!
//! This is **not a replacement** for GJK + EPA — both paths stay
//! live. GJK + EPA is the default; callers who hit deep-penetration
//! pathologies (cloth, MPM, reset states) can opt in to MPR.
//!
//! All routines are pure CPU. A WGSL port is deferred to Stage 28.1.

use molla_math::{Real, Transform, Vec3};
use molla_sim::ShapeGeometry;

use crate::support::support_with_hulls;

/// Trait for any convex shape exposing a support map. The MPR
/// algorithm only ever talks to a shape via this trait; the same
/// interface works for analytic primitives, convex hulls, and ad-hoc
/// per-test fixtures (closures, mocks).
pub trait SupportFn {
    /// Point on the shape farthest along `dir` (world frame). `dir`
    /// need not be unit-length; if it is zero, any point is valid.
    fn support(&self, dir: Vec3) -> Vec3;
}

/// Adapter wrapping a `(ShapeGeometry, Transform, hulls)` triple as a
/// `SupportFn`. Construct with [`ShapeSupport::new`] / [`ShapeSupport::with_hulls`].
pub struct ShapeSupport<'a> {
    geom: ShapeGeometry,
    xf: &'a Transform,
    hulls: &'a [Vec<Vec3>],
}

impl<'a> ShapeSupport<'a> {
    /// Convenience constructor for shapes that don't reference the
    /// hull table (everything except `ShapeGeometry::ConvexHull`).
    pub fn new(geom: ShapeGeometry, xf: &'a Transform) -> Self {
        Self {
            geom,
            xf,
            hulls: &[],
        }
    }

    /// Constructor with explicit hull table — required for
    /// `ShapeGeometry::ConvexHull { hull_id }` to resolve.
    pub fn with_hulls(geom: ShapeGeometry, xf: &'a Transform, hulls: &'a [Vec<Vec3>]) -> Self {
        Self { geom, xf, hulls }
    }
}

impl<'a> SupportFn for ShapeSupport<'a> {
    #[inline]
    fn support(&self, dir: Vec3) -> Vec3 {
        support_with_hulls(self.geom, self.xf, dir, self.hulls)
    }
}

/// MPR contact manifold — single canonical contact point, normal,
/// and depth, plus closest points on each shape. Mirrors EPA's
/// output so callers can swap freely.
#[derive(Copy, Clone, Debug)]
pub struct Manifold {
    /// World-space closest point on shape A.
    pub point_a: Vec3,
    /// World-space closest point on shape B.
    pub point_b: Vec3,
    /// Unit normal pointing from A into B.
    pub normal: Vec3,
    /// Penetration depth (positive when shapes overlap).
    pub depth: Real,
}

/// One vertex of the MPR portal — Minkowski-difference point plus
/// the per-shape support points that produced it. Mirrors GJK's
/// `SupportPoint` so we can recover closest-points on the originals.
#[derive(Copy, Clone, Debug)]
struct Sp {
    /// `support_A(d) − support_B(−d)` in world frame.
    w: Vec3,
    /// `support_A(d)` in world frame.
    a: Vec3,
    /// `support_B(−d)` in world frame.
    b: Vec3,
}

#[inline]
fn pair_support<A: SupportFn + ?Sized, B: SupportFn + ?Sized>(a: &A, b: &B, dir: Vec3) -> Sp {
    let pa = a.support(dir);
    let pb = b.support(-dir);
    Sp {
        w: pa - pb,
        a: pa,
        b: pb,
    }
}

/// Compute a guaranteed-interior point of the Minkowski difference
/// `A − B`. Strategy: average two opposing supports per shape to get
/// each shape's centre estimate (always interior to a convex shape),
/// then return the difference. This is interior to `A − B` because
/// the Minkowski difference is the set `{a − b : a ∈ A, b ∈ B}` and
/// the centroid difference is a particular instance.
fn interior_point<A: SupportFn + ?Sized, B: SupportFn + ?Sized>(a: &A, b: &B) -> Sp {
    // Use orthogonal direction pairs to get a stable centre estimate
    // even on shapes whose +X/−X support pair degenerates (a flat
    // plate normal to X collapses both supports to the same point).
    let dirs = [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z];
    let mut a_sum = Vec3::ZERO;
    let mut b_sum = Vec3::ZERO;
    for &d in &dirs {
        a_sum += a.support(d);
        b_sum += b.support(-d);
    }
    let a_mid = a_sum / 6.0;
    let b_mid = b_sum / 6.0;
    Sp {
        w: a_mid - b_mid,
        a: a_mid,
        b: b_mid,
    }
}

/// Run MPR on two convex shapes given by their support functions.
///
/// * `tolerance` — portal-refinement convergence threshold (units of
///   distance in Minkowski space). Newton uses `1e-6`; values down to
///   `1e-9` improve accuracy at the cost of a few extra iters.
/// * `max_iters` — safety cap on portal refinement steps. 64 is the
///   library default and almost always over-provisioned.
///
/// Returns `Some(Manifold)` on overlap (or surface contact) and
/// `None` when the shapes are strictly disjoint. The shapes are
/// assumed convex; behaviour on non-convex inputs is undefined.
pub fn mpr_collide<A: SupportFn + ?Sized, B: SupportFn + ?Sized>(
    a: &A,
    b: &B,
    tolerance: Real,
    max_iters: u32,
) -> Option<Manifold> {
    // ── Phase 1: Interior point v0 of A − B (guaranteed inside).
    let v0 = interior_point(a, b);
    if v0.w.length_squared() < 1e-24 {
        // Interior point is the origin → shapes already overlap at
        // their centroids. Sample a probe direction to recover a
        // non-zero portal seed.
        let probe = pair_support(a, b, Vec3::X);
        return Some(Manifold {
            point_a: v0.a,
            point_b: v0.b,
            normal: if probe.w.length_squared() > 1e-24 {
                probe.w.normalize()
            } else {
                Vec3::Y
            },
            depth: 0.0,
        });
    }

    // ── Phase 2: First portal vertex v1 — support along (origin − v0) = −v0.
    let mut dir = -v0.w;
    let mut v1 = pair_support(a, b, dir);
    if v1.w.dot(dir) <= 0.0 {
        // Origin lies outside the supporting half-space → no overlap.
        return None;
    }

    // ── Phase 3: Second portal vertex v2 — search direction is
    // perpendicular to the segment v0-v1 and in the half-plane
    // containing the origin.
    dir = v0.w.cross(v1.w);
    if dir.length_squared() < 1e-24 {
        // v0, v1, and origin are colinear. Origin lies on segment
        // v0-v1 iff v1.w and v0.w have opposite signs along the line
        // — which by construction (v0 inside, v1 the support past
        // origin in direction −v0) is always true if there's overlap.
        // Pick a perpendicular direction and proceed.
        dir = perp_to(v1.w - v0.w);
        if dir.length_squared() < 1e-24 {
            return None;
        }
    }
    let mut v2 = pair_support(a, b, dir);
    if v2.w.dot(dir) <= 0.0 {
        return None;
    }

    // ── Phase 4: Find v3 such that triangle (v1, v2, v3) brackets
    // the ray from v0 through the origin.
    let mut iter = 0u32;
    loop {
        if iter > max_iters {
            return None;
        }
        iter += 1;

        // Candidate portal-face normal — outward from the polytope
        // (i.e. away from v0).
        dir = (v1.w - v0.w).cross(v2.w - v0.w);
        // Orient so it points away from v0 (toward the boundary).
        if dir.dot(v0.w) > 0.0 {
            dir = -dir;
        }
        if dir.length_squared() < 1e-24 {
            return None;
        }

        let v3 = pair_support(a, b, dir);
        if v3.w.dot(dir) <= 0.0 {
            return None;
        }

        // Origin-encloses test:
        //
        // The ray r(t) = t * (origin - v0) = -t * v0 should pierce
        // the triangle (v1, v2, v3). Equivalently: origin should lie
        // on the same side of every edge plane as v0's image inside
        // the tetra (v0, v1, v2, v3).
        //
        // Standard MPR test: evaluate (v0 × v_i) · v_j for adjacent
        // pairs of (v1, v2, v3) — if origin is "outside" an edge,
        // replace the opposite vertex with v3 and retry.

        // Test edge (v1, v3): is origin outside the v0-v1-v3 plane?
        let cross_v0_v3 = v0.w.cross(v3.w);
        if cross_v0_v3.dot(v1.w) < 0.0 {
            // Origin is on the v2 side of the v0-v1-v3 plane —
            // replace v2 with v3 and re-test.
            v2 = v3;
            continue;
        }
        // Test edge (v2, v3): is origin outside the v0-v2-v3 plane?
        if cross_v0_v3.dot(v2.w) > 0.0 {
            // Origin is on the v1 side of the v0-v2-v3 plane —
            // replace v1 with v3 and re-test.
            v1 = v3;
            continue;
        }
        // Origin is bracketed by (v1, v2, v3) → start refinement.
        return refine_portal(a, b, v0, v1, v2, v3, tolerance, max_iters);
    }
}

/// Phase 5: refine the portal until its outward face is within
/// `tolerance` of being touched by a fresh support point. Each
/// iteration computes the portal-face normal `n`, queries support
/// along `n`, and:
///
/// * If the new support point sits within `tolerance` of the portal
///   plane → converged. Return the manifold.
/// * Otherwise replace whichever of (v1, v2, v3) keeps the new
///   tetrahedron (v0, v1, v2, v3) bracketing the ray v0 → origin.
fn refine_portal<A: SupportFn + ?Sized, B: SupportFn + ?Sized>(
    a: &A,
    b: &B,
    v0: Sp,
    mut v1: Sp,
    mut v2: Sp,
    mut v3: Sp,
    tolerance: Real,
    max_iters: u32,
) -> Option<Manifold> {
    let mut last_n = Vec3::Y;
    for _ in 0..max_iters {
        // Outward portal-face normal from triangle (v1, v2, v3).
        let mut n_raw = (v2.w - v1.w).cross(v3.w - v1.w);
        // Orient n so it points OUTWARD (away from v0). Inside the
        // polytope we have v0 — the portal triangle separates v0
        // from the origin, so outward = away from v0.
        if n_raw.dot(v1.w - v0.w) < 0.0 {
            n_raw = -n_raw;
        }
        let n_len_sq = n_raw.length_squared();
        if n_len_sq < 1e-24 {
            // Degenerate face — emit current best with the last
            // valid normal.
            return Some(extract_contact(v1, v2, v3, last_n));
        }
        let n = n_raw / n_len_sq.sqrt();
        last_n = n;

        // Distance from origin to the portal-face plane (along +n).
        let face_dist = v1.w.dot(n);

        // Sample a support past the portal.
        let v4 = pair_support(a, b, n);
        let new_dist = v4.w.dot(n);

        // Convergence: new support is no further past the face than
        // the requested tolerance. (`face_dist` may be negative if
        // the portal triangle is on the wrong side of the origin —
        // this can happen during early iters; the test still works.)
        if new_dist - face_dist < tolerance {
            return Some(extract_contact(v1, v2, v3, n));
        }

        // Choose which of v1/v2/v3 to replace. Reference: Snethen
        // 2007 §3.3 / Newhouse 2008 §2.5.
        //
        // Form the triangle (v0, v4) × (v1, v2, v3). The origin
        // ray r(t) = -t * v0 should still pierce the chosen
        // sub-portal after replacement.
        //
        // The test compares signs of `(v4 × v0) · v_i` — this is
        // proportional to the signed volume of the tetra (origin,
        // v0, v4, v_i). Together with v_j they decide which sub-
        // portal contains the ray.
        let cross_v4_v0 = v4.w.cross(v0.w);
        let d1 = v1.w.dot(cross_v4_v0);
        let d2 = v2.w.dot(cross_v4_v0);
        let d3 = v3.w.dot(cross_v4_v0);

        if d1 >= 0.0 {
            if d2 >= 0.0 {
                // Origin in (v0, v4, v1, v2) sub-tetra → drop v3.
                v3 = v4;
            } else {
                // Origin in (v0, v4, v3, v1) sub-tetra → drop v2.
                v2 = v4;
            }
        } else {
            if d3 >= 0.0 {
                // Origin in (v0, v4, v2, v3) sub-tetra → drop v1.
                v1 = v4;
            } else {
                // Origin in (v0, v4, v1, v2) sub-tetra → drop v3.
                v3 = v4;
            }
        }
    }

    // Iter cap — return current best.
    Some(extract_contact(v1, v2, v3, last_n))
}

/// Build the final `Manifold` from the converged portal. Mixes the
/// per-shape support points by the origin's barycentric weights on
/// the portal triangle to recover closest-points on each shape.
fn extract_contact(v1: Sp, v2: Sp, v3: Sp, normal: Vec3) -> Manifold {
    // Project the origin onto the portal plane (face of the
    // Minkowski difference) and compute barycentric coords.
    let face_dist = v1.w.dot(normal);
    let proj = normal * face_dist;
    let (u, v, w) = barycentric(v1.w, v2.w, v3.w, proj);
    let pa = v1.a * u + v2.a * v + v3.a * w;
    let pb = v1.b * u + v2.b * v + v3.b * w;
    Manifold {
        point_a: pa,
        point_b: pb,
        // MPR's outward face normal in Minkowski space (A − B)
        // points from A's surface toward B's surface in world space —
        // matching molla's "normal A → B" convention.
        normal: if normal.length_squared() > 0.5 {
            normal
        } else {
            (pb - pa).normalize_or_zero()
        },
        depth: face_dist.max(0.0),
    }
}

/// Pick a vector roughly perpendicular to `v`. Stable for any non-
/// zero `v`.
#[inline]
fn perp_to(v: Vec3) -> Vec3 {
    let (ax, ay, az) = (v.x.abs(), v.y.abs(), v.z.abs());
    let other = if ax <= ay && ax <= az {
        Vec3::X
    } else if ay <= az {
        Vec3::Y
    } else {
        Vec3::Z
    };
    let p = v.cross(other);
    if p.length_squared() < 1e-24 {
        // v parallel to `other` — fall back to a different axis.
        v.cross(Vec3::Z)
    } else {
        p
    }
}

/// Barycentric coordinates of `p` on triangle `abc`. Same as
/// `epa::barycentric` — duplicated here to keep MPR self-contained.
fn barycentric(a: Vec3, b: Vec3, c: Vec3, p: Vec3) -> (Real, Real, Real) {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    if denom.abs() < 1e-24 {
        return (1.0, 0.0, 0.0);
    }
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    (u, v, w)
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_math::Quat;

    #[test]
    fn shape_support_adapter_matches_support_fn() {
        // Sanity check the ShapeSupport adapter delegates correctly.
        let xf = Transform::IDENTITY;
        let s = ShapeSupport::new(ShapeGeometry::Sphere { radius: 2.0 }, &xf);
        let p = s.support(Vec3::X);
        assert!((p - Vec3::new(2.0, 0.0, 0.0)).length() < 1e-12);
    }

    #[test]
    fn mpr_overlapping_spheres_returns_overlap() {
        // Two unit spheres centres 1.5 apart → 0.5 overlap along +X.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(1.5, 0.0, 0.0));
        let a = ShapeSupport::new(ShapeGeometry::Sphere { radius: 1.0 }, &xf_a);
        let b = ShapeSupport::new(ShapeGeometry::Sphere { radius: 1.0 }, &xf_b);
        let m = mpr_collide(&a, &b, 1e-6, 64).expect("expected overlap");
        assert!(
            (m.depth - 0.5).abs() < 5e-2,
            "depth got {}, expected 0.5",
            m.depth
        );
        assert!(
            (m.normal - Vec3::X).length() < 5e-2,
            "normal got {:?}",
            m.normal
        );
    }

    #[test]
    fn mpr_separated_spheres_returns_none() {
        // Centres 3 apart, radii 1 each → 1.0 gap.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(3.0, 0.0, 0.0));
        let a = ShapeSupport::new(ShapeGeometry::Sphere { radius: 1.0 }, &xf_a);
        let b = ShapeSupport::new(ShapeGeometry::Sphere { radius: 1.0 }, &xf_b);
        assert!(mpr_collide(&a, &b, 1e-6, 64).is_none());
    }

    #[test]
    fn mpr_box_box_shallow_overlap() {
        // Identical unit boxes, 1.8 apart along +X → 0.2 shallow overlap.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(1.8, 0.0, 0.0));
        let g = ShapeGeometry::Box {
            half_extents: Vec3::splat(1.0),
        };
        let a = ShapeSupport::new(g, &xf_a);
        let b = ShapeSupport::new(g, &xf_b);
        let m = mpr_collide(&a, &b, 1e-6, 64).expect("expected overlap");
        assert!((m.depth - 0.2).abs() < 1e-2, "depth: {}", m.depth);
        assert!(
            (m.normal - Vec3::X).length() < 5e-2,
            "normal: {:?}",
            m.normal
        );
    }

    #[test]
    fn mpr_rotated_box_overlap() {
        // Box A at origin, box B rotated 45° about Y, partly overlapping.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::new(
            Vec3::new(1.4, 0.0, 0.0),
            Quat::from_rotation_y(std::f64::consts::FRAC_PI_4),
        );
        let g = ShapeGeometry::Box {
            half_extents: Vec3::splat(1.0),
        };
        let a = ShapeSupport::new(g, &xf_a);
        let b = ShapeSupport::new(g, &xf_b);
        let m = mpr_collide(&a, &b, 1e-6, 64).expect("overlap");
        assert!(m.depth > 0.0);
    }
}
