//! Gilbert-Johnson-Keerthi (GJK) algorithm — finds whether two
//! convex shapes overlap, and if so, drives EPA. If they don't
//! overlap, GJK additionally returns the closest points on each.
//!
//! References:
//! - Ericson, *Real-Time Collision Detection* (2005), §9.5.
//! - van den Bergen, *A Fast and Robust GJK Implementation* (1999).
//!
//! The "boolean" GJK (overlap test) is the standard 4-simplex
//! algorithm: walk a simplex (point → segment → triangle → tetra)
//! through Minkowski-difference space, terminating when the simplex
//! contains the origin (overlap) or when no closer support point
//! exists (separated). The "distance" variant — implemented here —
//! preserves the simplex sub-coordinates so we can recover closest
//! points on the originals.
//!
//! All routines are pure CPU. A WGSL port lives in stage 6.x for the
//! GPU narrow phase.

use crate::math::{Real, Transform, Vec3};
use crate::sim::ShapeGeometry;

use super::support::{support, support_with_hulls};

/// Result of a successful GJK distance query on disjoint shapes.
#[derive(Copy, Clone, Debug)]
pub struct GjkDistance {
    /// World-space closest point on shape A.
    pub point_a: Vec3,
    /// World-space closest point on shape B.
    pub point_b: Vec3,
    /// Distance between `point_a` and `point_b` (positive).
    pub distance: Real,
}

/// What GJK found. `Overlap` carries the final simplex so EPA can
/// pick up where GJK left off.
#[derive(Clone, Debug)]
pub enum GjkResult {
    /// Shapes are disjoint or touching; closest points and distance
    /// are valid.
    Disjoint(GjkDistance),
    /// Shapes overlap; the simplex contains the origin.
    Overlap(GjkSimplex),
}

/// A GJK simplex of up to 4 points, with the per-point support data
/// needed to recover closest-points on the originals.
#[derive(Clone, Debug, Default)]
pub struct GjkSimplex {
    /// Up to 4 vertices in Minkowski-difference space.
    pub points: Vec<SupportPoint>,
}

/// One simplex vertex: the Minkowski-difference point plus the
/// individual support points on each shape that produced it.
#[derive(Copy, Clone, Debug)]
pub struct SupportPoint {
    /// `support_A(d) − support_B(−d)` in world frame.
    pub w: Vec3,
    /// `support_A(d)` in world frame (used by the "distance" path).
    pub a: Vec3,
    /// `support_B(−d)` in world frame.
    pub b: Vec3,
}

/// Run GJK on two convex shapes positioned by `xf_a` and `xf_b`.
/// Returns either `Disjoint` (with closest points) or `Overlap`
/// (with the simplex feeding EPA).
pub fn gjk(
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
) -> GjkResult {
    gjk_with_hulls(geom_a, xf_a, geom_b, xf_b, &[])
}

/// `gjk` variant that resolves `ShapeGeometry::ConvexHull` via the
/// caller's hull table. Pass `&[]` for shapes that don't reference
/// hulls. Used by the narrow-phase entry point which has access to
/// `model.hulls`.
pub fn gjk_with_hulls(
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    hulls: &[Vec<Vec3>],
) -> GjkResult {
    const MAX_ITERS: usize = 64;
    const EPS: Real = 1e-12;

    // Initial direction: from B's centre toward A's. Any non-zero
    // direction works.
    let mut dir = xf_a.position - xf_b.position;
    if dir.length_squared() < EPS {
        dir = Vec3::X;
    }

    // Track `v` = the simplex's current closest point to the origin
    // (in Minkowski space). The search direction is `-v`, i.e. from
    // v toward the origin. Termination follows Ericson §9.5:
    //   v·v − v·w ≤ ε · v·v  (where w = next support point)
    // means w doesn't lie deeper toward the origin than v already
    // does → simplex is converged.
    let mut simplex: Vec<SupportPoint> = Vec::with_capacity(4);
    simplex.push(compute_support(geom_a, xf_a, geom_b, xf_b, hulls, dir));
    let mut v = simplex[0].w;

    for _ in 0..MAX_ITERS {
        let v_dot_v = v.length_squared();
        if v_dot_v < EPS {
            // Origin coincides with the simplex's closest point —
            // treat as overlap.
            return GjkResult::Overlap(GjkSimplex { points: simplex });
        }
        let dir_search = -v;
        let new_pt = compute_support(geom_a, xf_a, geom_b, xf_b, hulls, dir_search);
        let v_dot_w = v.dot(new_pt.w);
        // No-progress termination: w is no closer to the origin
        // along `v`'s direction than v itself.
        if v_dot_v - v_dot_w <= 1e-10 * v_dot_v {
            let (pa, pb, dist) = closest_on_simplex_with_points(&simplex);
            return GjkResult::Disjoint(GjkDistance {
                point_a: pa,
                point_b: pb,
                distance: dist,
            });
        }

        simplex.push(new_pt);
        match do_simplex(&mut simplex) {
            SimplexStep::ContainsOrigin => {
                return GjkResult::Overlap(GjkSimplex { points: simplex });
            }
            SimplexStep::Continue(_) => {
                // Recompute v from the (reduced) simplex.
                let (_, _, _) = (&simplex, 0, 0);
                v = simplex_closest_point(&simplex);
            }
        }
        // Pre-empt a redundant simplex if it doesn't improve.
        if v.length_squared() >= v_dot_v {
            let (pa, pb, dist) = closest_on_simplex_with_points(&simplex);
            return GjkResult::Disjoint(GjkDistance {
                point_a: pa,
                point_b: pb,
                distance: dist,
            });
        }
    }

    // Iteration cap reached without convergence — treat as touching.
    let (pa, pb, dist) = closest_on_simplex_with_points(&simplex);
    GjkResult::Disjoint(GjkDistance {
        point_a: pa,
        point_b: pb,
        distance: dist,
    })
}

#[inline]
fn compute_support(
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    hulls: &[Vec<Vec3>],
    dir: Vec3,
) -> SupportPoint {
    let a = support_with_hulls(geom_a, xf_a, dir, hulls);
    let b = support_with_hulls(geom_b, xf_b, -dir, hulls);
    SupportPoint {
        w: a - b,
        a,
        b,
    }
}

#[allow(dead_code)]
fn _suppress_unused_support_warning(g: ShapeGeometry, xf: &Transform, d: Vec3) -> Vec3 {
    support(g, xf, d)
}

enum SimplexStep {
    ContainsOrigin,
    Continue(#[allow(dead_code)] Vec3),
}

/// Reduce `simplex` to the smallest sub-feature containing the
/// origin's projection (a vertex / edge / triangle), and return the
/// next search direction toward the origin from that sub-feature.
fn do_simplex(simplex: &mut Vec<SupportPoint>) -> SimplexStep {
    // The newest support point is always at the back.
    let n = simplex.len();
    match n {
        2 => do_simplex_line(simplex),
        3 => do_simplex_triangle(simplex),
        4 => do_simplex_tetra(simplex),
        _ => SimplexStep::Continue(Vec3::ZERO),
    }
}

fn do_simplex_line(s: &mut Vec<SupportPoint>) -> SimplexStep {
    // s = [a, b], with `b` the newest.
    let a = s[1].w;
    let b = s[0].w;
    let ab = b - a;
    let ao = -a;
    if ab.dot(ao) > 0.0 {
        // Origin lies in the AB strip; keep both, search perpendicular.
        let dir = triple_product(ab, ao, ab);
        SimplexStep::Continue(dir)
    } else {
        // Origin is "behind" A; drop B.
        s.remove(0);
        SimplexStep::Continue(ao)
    }
}

fn do_simplex_triangle(s: &mut Vec<SupportPoint>) -> SimplexStep {
    // s = [c, b, a], with `a` the newest.
    let a = s[2].w;
    let b = s[1].w;
    let c = s[0].w;
    let ab = b - a;
    let ac = c - a;
    let ao = -a;
    let abc = ab.cross(ac);
    if abc.cross(ac).dot(ao) > 0.0 {
        // Origin outside in the AC region.
        if ac.dot(ao) > 0.0 {
            // Keep [a, c]; drop b.
            s.remove(1);
            SimplexStep::Continue(triple_product(ac, ao, ac))
        } else {
            // Drop c, recurse to line.
            s.remove(0);
            do_simplex_line(s)
        }
    } else if ab.cross(abc).dot(ao) > 0.0 {
        // AB region. Drop c, recurse.
        s.remove(0);
        do_simplex_line(s)
    } else {
        // Origin is above or below the triangle.
        if abc.dot(ao) > 0.0 {
            SimplexStep::Continue(abc)
        } else {
            // Flip orientation: swap b and c.
            s.swap(0, 1);
            SimplexStep::Continue(-abc)
        }
    }
}

fn do_simplex_tetra(s: &mut Vec<SupportPoint>) -> SimplexStep {
    // s = [d, c, b, a], with `a` the newest. Test the three faces
    // ABC / ACD / ADB; if origin is on the inside of all three, the
    // tetra contains it.
    let a = s[3].w;
    let b = s[2].w;
    let c = s[1].w;
    let d = s[0].w;
    let ab = b - a;
    let ac = c - a;
    let ad = d - a;
    let ao = -a;

    let abc = ab.cross(ac);
    let acd = ac.cross(ad);
    let adb = ad.cross(ab);

    if abc.dot(ao) > 0.0 {
        // Drop d, recurse to triangle [c, b, a].
        s.remove(0);
        return do_simplex_triangle(s);
    }
    if acd.dot(ao) > 0.0 {
        // Drop b, recurse to triangle [d, c, a].
        s.remove(2);
        return do_simplex_triangle(s);
    }
    if adb.dot(ao) > 0.0 {
        // Drop c, reorder to triangle [b, d, a].
        s.remove(1);
        // Current order: [d, b, a]. We want [b, d, a] for the
        // triangle routine's [c, b, a] convention. Swap [0] and [1].
        s.swap(0, 1);
        return do_simplex_triangle(s);
    }
    SimplexStep::ContainsOrigin
}

/// `(a × b) × c` — the GJK "triple product".
#[inline]
fn triple_product(a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    a.cross(b).cross(c)
}

/// Closest point on the simplex's convex hull to the origin, in
/// Minkowski space. Used to update GJK's search vector.
fn simplex_closest_point(s: &[SupportPoint]) -> Vec3 {
    match s.len() {
        0 => Vec3::ZERO,
        1 => s[0].w,
        2 => {
            let p0 = s[0].w;
            let p1 = s[1].w;
            let d = p1 - p0;
            let len_sq = d.length_squared();
            let t = if len_sq > 1e-24 {
                (-p0).dot(d) / len_sq
            } else {
                0.0
            }
            .clamp(0.0, 1.0);
            p0 + d * t
        }
        3 => {
            let (u, v, w) = bary_triangle(s[0].w, s[1].w, s[2].w, Vec3::ZERO);
            s[0].w * u + s[1].w * v + s[2].w * w
        }
        _ => {
            // 4-vertex simplex with origin inside should have been
            // routed through `Overlap`. As a fallback, project onto
            // the closest face.
            let faces = [[0, 1, 2usize], [0, 1, 3], [0, 2, 3], [1, 2, 3]];
            let mut best_p = Vec3::ZERO;
            let mut best_d = Real::INFINITY;
            for f in faces {
                let (u, v, w) = bary_triangle(s[f[0]].w, s[f[1]].w, s[f[2]].w, Vec3::ZERO);
                let p = s[f[0]].w * u + s[f[1]].w * v + s[f[2]].w * w;
                let d = p.length();
                if d < best_d {
                    best_d = d;
                    best_p = p;
                }
            }
            best_p
        }
    }
}

/// Compute the world-space closest point on shape A and shape B
/// from the current simplex, plus the current Minkowski-distance.
/// The simplex must lie on the shape's surface boundary in
/// Minkowski space (i.e. GJK has terminated without containing
/// origin).
fn closest_on_simplex_with_points(s: &[SupportPoint]) -> (Vec3, Vec3, Real) {
    match s.len() {
        1 => {
            let dist = s[0].w.length();
            (s[0].a, s[0].b, dist)
        }
        2 => {
            // Project origin onto segment in Minkowski space.
            let p0 = s[0].w;
            let p1 = s[1].w;
            let d = p1 - p0;
            let len_sq = d.length_squared();
            let t = if len_sq > 1e-24 {
                (-p0).dot(d) / len_sq
            } else {
                0.0
            }
            .clamp(0.0, 1.0);
            let pa = s[0].a + (s[1].a - s[0].a) * t;
            let pb = s[0].b + (s[1].b - s[0].b) * t;
            let pm = p0 + d * t;
            (pa, pb, pm.length())
        }
        3 => {
            // Project origin onto triangle in Minkowski space.
            let (u, v, w) = bary_triangle(s[0].w, s[1].w, s[2].w, Vec3::ZERO);
            let pa = s[0].a * u + s[1].a * v + s[2].a * w;
            let pb = s[0].b * u + s[1].b * v + s[2].b * w;
            let pm = s[0].w * u + s[1].w * v + s[2].w * w;
            (pa, pb, pm.length())
        }
        _ => {
            // 4-vertex simplex: origin is inside, distance is 0
            // — caller should have routed through Overlap.
            // Fall back to the closest face.
            let (pa, pb) = closest_face_points(s);
            (pa, pb, 0.0)
        }
    }
}

/// Barycentric coordinates of `p` on triangle `abc`. Clamped into
/// the triangle so the result is the closest point on the triangle.
fn bary_triangle(a: Vec3, b: Vec3, c: Vec3, p: Vec3) -> (Real, Real, Real) {
    // Voronoi-region projection (Ericson §3.4 / §5.1.5).
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (1.0, 0.0, 0.0);
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (0.0, 1.0, 0.0);
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (1.0 - v, v, 0.0);
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (0.0, 0.0, 1.0);
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (1.0 - w, 0.0, w);
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (0.0, 1.0 - w, w);
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (1.0 - v - w, v, w)
}

/// Fallback: pick the face of a 4-simplex whose closest point to
/// the origin is nearest, then return the closest point on each
/// shape from that face's barycentrics.
fn closest_face_points(s: &[SupportPoint]) -> (Vec3, Vec3) {
    // 4-tetra faces: 0-1-2, 0-1-3, 0-2-3, 1-2-3.
    let faces = [
        [0usize, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
    ];
    let mut best_dist = Real::INFINITY;
    let mut best_pa = Vec3::ZERO;
    let mut best_pb = Vec3::ZERO;
    for f in faces {
        let (u, v, w) = bary_triangle(s[f[0]].w, s[f[1]].w, s[f[2]].w, Vec3::ZERO);
        let pm = s[f[0]].w * u + s[f[1]].w * v + s[f[2]].w * w;
        let d = pm.length();
        if d < best_dist {
            best_dist = d;
            best_pa = s[f[0]].a * u + s[f[1]].a * v + s[f[2]].a * w;
            best_pb = s[f[0]].b * u + s[f[1]].b * v + s[f[2]].b * w;
        }
    }
    (best_pa, best_pb)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Quat;

    #[test]
    fn gjk_two_separated_spheres_distance() {
        // A: r=1 at origin. B: r=0.5 at (3, 0, 0). Distance between
        // surfaces = 3 - 1 - 0.5 = 1.5.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(3.0, 0.0, 0.0));
        let res = gjk(
            ShapeGeometry::Sphere { radius: 1.0 },
            &xf_a,
            ShapeGeometry::Sphere { radius: 0.5 },
            &xf_b,
        );
        match res {
            GjkResult::Disjoint(d) => {
                assert!(
                    (d.distance - 1.5).abs() < 1e-3,
                    "distance: got {} expected 1.5",
                    d.distance
                );
                // Closest point on A: (1, 0, 0). On B: (2.5, 0, 0).
                assert!(
                    (d.point_a - Vec3::new(1.0, 0.0, 0.0)).length() < 1e-3,
                    "point_a: {:?}",
                    d.point_a
                );
                assert!(
                    (d.point_b - Vec3::new(2.5, 0.0, 0.0)).length() < 1e-3,
                    "point_b: {:?}",
                    d.point_b
                );
            }
            GjkResult::Overlap(_) => panic!("expected disjoint"),
        }
    }

    #[test]
    fn gjk_overlapping_spheres_returns_overlap() {
        // Penetrating: distance between centres = 1.5, sum r = 2.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(1.5, 0.0, 0.0));
        let res = gjk(
            ShapeGeometry::Sphere { radius: 1.0 },
            &xf_a,
            ShapeGeometry::Sphere { radius: 1.0 },
            &xf_b,
        );
        assert!(matches!(res, GjkResult::Overlap(_)));
    }

    #[test]
    fn gjk_box_box_separated() {
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(3.0, 0.0, 0.0));
        let res = gjk(
            ShapeGeometry::Box {
                half_extents: Vec3::splat(1.0),
            },
            &xf_a,
            ShapeGeometry::Box {
                half_extents: Vec3::splat(1.0),
            },
            &xf_b,
        );
        match res {
            GjkResult::Disjoint(d) => {
                // Faces touch at x=1 vs x=2 → distance 1.
                assert!(
                    (d.distance - 1.0).abs() < 1e-3,
                    "distance: {}",
                    d.distance
                );
            }
            GjkResult::Overlap(_) => panic!("expected disjoint"),
        }
    }

    #[test]
    fn gjk_box_box_overlapping() {
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(1.5, 0.0, 0.0));
        let res = gjk(
            ShapeGeometry::Box {
                half_extents: Vec3::splat(1.0),
            },
            &xf_a,
            ShapeGeometry::Box {
                half_extents: Vec3::splat(1.0),
            },
            &xf_b,
        );
        assert!(matches!(res, GjkResult::Overlap(_)));
    }

    #[test]
    fn gjk_capsule_box_separated_distance() {
        // Capsule along Y at origin, half=0.5 r=0.3. Box at (2, 0, 0),
        // half=(0.5, 0.5, 0.5). Closest distance between surfaces:
        // capsule cylinder radius 0.3 on +X side reaches x=0.3, box
        // -X face at x=1.5. Distance = 1.5 - 0.3 = 1.2.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(2.0, 0.0, 0.0));
        let res = gjk(
            ShapeGeometry::Capsule {
                half_height: 0.5,
                radius: 0.3,
            },
            &xf_a,
            ShapeGeometry::Box {
                half_extents: Vec3::splat(0.5),
            },
            &xf_b,
        );
        match res {
            GjkResult::Disjoint(d) => {
                assert!(
                    (d.distance - 1.2).abs() < 1e-2,
                    "distance: got {} expected ~1.2",
                    d.distance
                );
            }
            GjkResult::Overlap(_) => panic!("expected disjoint"),
        }
    }

    #[test]
    fn gjk_cylinder_cylinder_perpendicular_overlap() {
        // Cylinder A along +Y at origin, half=1 r=0.4.
        // Cylinder B along +X at (0, 0.5, 0) (rotated −π/2 about Z),
        // half=1 r=0.4. They cross.
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::new(
            Vec3::new(0.0, 0.5, 0.0),
            Quat::from_rotation_z(-std::f64::consts::FRAC_PI_2),
        );
        let res = gjk(
            ShapeGeometry::Cylinder {
                half_height: 1.0,
                radius: 0.4,
            },
            &xf_a,
            ShapeGeometry::Cylinder {
                half_height: 1.0,
                radius: 0.4,
            },
            &xf_b,
        );
        assert!(matches!(res, GjkResult::Overlap(_)));
    }
}
