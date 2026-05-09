//! Support functions for convex shapes — `support(direction) → point`
//! returns the point on the shape farthest along `direction`.
//!
//! This is the GJK / EPA foundation: those algorithms only ever
//! query a shape via its support map. Each routine below works in
//! **world frame** by transforming the query direction into the
//! shape's local frame, evaluating the analytic support, and
//! transforming the result back.

use crate::math::{Real, Transform, Vec3};
use crate::sim::ShapeGeometry;

/// Point on `geom` (positioned by `xf`) that is farthest along
/// `dir_world`. `dir_world` need not be unit-length; if it's zero,
/// any point is valid (we return the shape origin in that case).
pub fn support(geom: ShapeGeometry, xf: &Transform, dir_world: Vec3) -> Vec3 {
    support_with_hulls(geom, xf, dir_world, &[])
}

/// Like [`support`] but with access to a hull table for
/// `ShapeGeometry::ConvexHull`. Standalone callers (no Model in
/// scope) can pass `&[]` and get a sentinel.
pub fn support_with_hulls(
    geom: ShapeGeometry,
    xf: &Transform,
    dir_world: Vec3,
    hulls: &[Vec<Vec3>],
) -> Vec3 {
    let len_sq = dir_world.length_squared();
    if len_sq < 1e-24 {
        return xf.position;
    }
    let inv_rot = xf.rotation.conjugate();
    let dir_local = inv_rot * dir_world;
    let local_p = match geom {
        ShapeGeometry::Sphere { radius } => support_sphere(radius, dir_local),
        ShapeGeometry::Plane { .. } => Vec3::ZERO,
        ShapeGeometry::Box { half_extents } => support_box(half_extents, dir_local),
        ShapeGeometry::Capsule {
            half_height,
            radius,
        } => support_capsule(half_height, radius, dir_local),
        ShapeGeometry::Cylinder {
            half_height,
            radius,
        } => support_cylinder(half_height, radius, dir_local),
        ShapeGeometry::Cone {
            half_height,
            radius,
        } => support_cone(half_height, radius, dir_local),
        ShapeGeometry::Triangle { v0, v1, v2 } => support_triangle(v0, v1, v2, dir_local),
        ShapeGeometry::TriMesh { .. } => Vec3::ZERO,
        // Heightfield + Sdf use bespoke narrow-phase paths (per-cell
        // triangle dispatch + ball-marching, respectively); a generic
        // GJK support function isn't useful here.
        ShapeGeometry::Heightfield { .. } => Vec3::ZERO,
        ShapeGeometry::Sdf { .. } => Vec3::ZERO,
        ShapeGeometry::ConvexHull { hull_id } => {
            let id = hull_id as usize;
            if id < hulls.len() {
                support_convex_hull(&hulls[id], dir_local)
            } else {
                Vec3::ZERO
            }
        }
    };
    xf.position + xf.rotation * local_p
}

/// Sphere support: any unit vector pointing along `dir`, scaled by
/// the radius.
#[inline]
pub fn support_sphere(radius: Real, dir: Vec3) -> Vec3 {
    let n = dir.normalize_or_zero();
    n * radius
}

/// Box support: pick the corner whose components match the sign of
/// `dir`, componentwise.
#[inline]
pub fn support_box(half: Vec3, dir: Vec3) -> Vec3 {
    Vec3::new(
        if dir.x >= 0.0 { half.x } else { -half.x },
        if dir.y >= 0.0 { half.y } else { -half.y },
        if dir.z >= 0.0 { half.z } else { -half.z },
    )
}

/// Capsule support along the local +Y axis: end-cap centre selected
/// by the sign of `dir.y`, plus a sphere-support for the cap radius.
#[inline]
pub fn support_capsule(half_height: Real, radius: Real, dir: Vec3) -> Vec3 {
    let cap_y = if dir.y >= 0.0 { half_height } else { -half_height };
    let sphere = support_sphere(radius, dir);
    Vec3::new(sphere.x, cap_y + sphere.y, sphere.z)
}

/// Cylinder support along the local +Y axis: the +Y or −Y disk based
/// on `dir.y`, with the in-plane (XZ) component pointing along `dir`'s
/// XZ projection scaled to `radius`. A degenerate XZ projection (dir
/// purely along Y) picks an arbitrary disk point.
#[inline]
pub fn support_cylinder(half_height: Real, radius: Real, dir: Vec3) -> Vec3 {
    let cap_y = if dir.y >= 0.0 { half_height } else { -half_height };
    let xz = Vec3::new(dir.x, 0.0, dir.z);
    let xz_len = xz.length();
    let edge = if xz_len > 1e-12 {
        xz * (radius / xz_len)
    } else {
        // dir is along ±Y; any disk-edge point works. Pick +X.
        Vec3::new(radius, 0.0, 0.0)
    };
    Vec3::new(edge.x, cap_y, edge.z)
}

/// Cone support along the local +Y axis: apex at `(0, +half_h, 0)`,
/// base disc at `y = -half_h` with radius `r`. Pick whichever has
/// the larger dot with `dir`.
#[inline]
pub fn support_cone(half_height: Real, radius: Real, dir: Vec3) -> Vec3 {
    let apex_dot = half_height * dir.y;
    let xz = Vec3::new(dir.x, 0.0, dir.z);
    let xz_len = xz.length();
    let base_edge = if xz_len > 1e-12 {
        xz * (radius / xz_len)
    } else {
        Vec3::new(radius, 0.0, 0.0)
    };
    let base_dot = base_edge.dot(dir) - half_height * dir.y;
    if apex_dot >= base_dot {
        Vec3::new(0.0, half_height, 0.0)
    } else {
        Vec3::new(base_edge.x, -half_height, base_edge.z)
    }
}

/// Convex-hull support: linear scan over `vertices` for the one
/// with the largest dot product against `dir`. O(N) per query —
/// fine for hulls up to ~50 vertices; past that consider DOP /
/// hill-climbing-with-edge-graph instead.
#[inline]
pub fn support_convex_hull(vertices: &[Vec3], dir: Vec3) -> Vec3 {
    let mut best = vertices[0];
    let mut best_dot = best.dot(dir);
    for &v in &vertices[1..] {
        let d = v.dot(dir);
        if d > best_dot {
            best_dot = d;
            best = v;
        }
    }
    best
}

/// Triangle support: argmax of the three vertices along `dir`.
#[inline]
pub fn support_triangle(v0: Vec3, v1: Vec3, v2: Vec3, dir: Vec3) -> Vec3 {
    let d0 = v0.dot(dir);
    let d1 = v1.dot(dir);
    let d2 = v2.dot(dir);
    if d0 >= d1 && d0 >= d2 {
        v0
    } else if d1 >= d2 {
        v1
    } else {
        v2
    }
}

/// Minkowski-difference support: `support_A(d) − support_B(−d)`.
/// This is what GJK / EPA query.
#[inline]
pub fn support_minkowski(
    geom_a: ShapeGeometry,
    xf_a: &Transform,
    geom_b: ShapeGeometry,
    xf_b: &Transform,
    dir_world: Vec3,
) -> Vec3 {
    support(geom_a, xf_a, dir_world) - support(geom_b, xf_b, -dir_world)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Quat;

    #[test]
    fn sphere_support_unit_directions() {
        let xf = Transform::IDENTITY;
        let geom = ShapeGeometry::Sphere { radius: 2.0 };
        for &d in &[Vec3::X, Vec3::Y, Vec3::Z, -Vec3::X, -Vec3::Y, -Vec3::Z] {
            let p = support(geom, &xf, d);
            assert!(
                (p - d * 2.0).length() < 1e-12,
                "support({d:?}) = {p:?}, expected {:?}",
                d * 2.0
            );
        }
    }

    #[test]
    fn box_support_picks_corner() {
        let xf = Transform::IDENTITY;
        let geom = ShapeGeometry::Box {
            half_extents: Vec3::new(1.0, 2.0, 3.0),
        };
        let p = support(geom, &xf, Vec3::new(1.0, -1.0, 1.0));
        assert!(
            (p - Vec3::new(1.0, -2.0, 3.0)).length() < 1e-12,
            "got {p:?}"
        );
    }

    #[test]
    fn capsule_support_endpoint_then_sphere() {
        // Capsule along +Y at origin, half_h=1, radius=0.5.
        // Support along +Y → top end-cap centre (0, 1, 0) + 0.5·Y.
        let xf = Transform::IDENTITY;
        let geom = ShapeGeometry::Capsule {
            half_height: 1.0,
            radius: 0.5,
        };
        let p = support(geom, &xf, Vec3::Y);
        assert!((p - Vec3::new(0.0, 1.5, 0.0)).length() < 1e-9);
        // Support along +X: cap centre selected by sign(0)=+ → +Y;
        // sphere support along +X = (0.5, 0, 0); sum = (0.5, 1, 0).
        let p = support(geom, &xf, Vec3::X);
        assert!((p - Vec3::new(0.5, 1.0, 0.0)).length() < 1e-9);
    }

    #[test]
    fn cylinder_support_disk_edge() {
        let xf = Transform::IDENTITY;
        let geom = ShapeGeometry::Cylinder {
            half_height: 1.0,
            radius: 0.5,
        };
        // +X: top disk + (+0.5, 0, 0) edge → (0.5, 1, 0).
        let p = support(geom, &xf, Vec3::X);
        assert!((p - Vec3::new(0.5, 1.0, 0.0)).length() < 1e-9);
        // +Y: top disk, arbitrary edge — verify y is correct.
        let p = support(geom, &xf, Vec3::Y);
        assert!((p.y - 1.0).abs() < 1e-9);
        let r = (p.x * p.x + p.z * p.z).sqrt();
        assert!((r - 0.5).abs() < 1e-9);
    }

    #[test]
    fn support_respects_world_transform() {
        // Sphere at world (10, 0, 0), still radius 2. Support along
        // +X should land at (12, 0, 0).
        let xf = Transform::from_translation(Vec3::new(10.0, 0.0, 0.0));
        let geom = ShapeGeometry::Sphere { radius: 2.0 };
        let p = support(geom, &xf, Vec3::X);
        assert!((p - Vec3::new(12.0, 0.0, 0.0)).length() < 1e-12);
    }

    #[test]
    fn support_handles_rotation() {
        // Box rotated 90° about Z so local +X axis becomes world +Y.
        let xf = Transform::new(
            Vec3::ZERO,
            Quat::from_rotation_z(std::f64::consts::FRAC_PI_2),
        );
        let geom = ShapeGeometry::Box {
            half_extents: Vec3::new(1.0, 0.0, 0.0),
        };
        // dir +Y world → local +X (after inverse rotation), which
        // picks +X corner → in world that's (0, 1, 0).
        let p = support(geom, &xf, Vec3::Y);
        assert!((p - Vec3::new(0.0, 1.0, 0.0)).length() < 1e-9, "got {p:?}");
    }

    #[test]
    fn minkowski_support_two_spheres() {
        // A: sphere r=1 at origin. B: sphere r=0.5 at (3, 0, 0).
        // Minkowski A−B's support along +X should be:
        //   support_A(+X) − support_B(−X) = (1,0,0) − (3-0.5, 0, 0)
        //                                 = (1, 0, 0) − (2.5, 0, 0)
        //                                 = (-1.5, 0, 0).
        let xf_a = Transform::IDENTITY;
        let xf_b = Transform::from_translation(Vec3::new(3.0, 0.0, 0.0));
        let p = support_minkowski(
            ShapeGeometry::Sphere { radius: 1.0 },
            &xf_a,
            ShapeGeometry::Sphere { radius: 0.5 },
            &xf_b,
            Vec3::X,
        );
        assert!((p - Vec3::new(-1.5, 0.0, 0.0)).length() < 1e-12);
    }
}
