//! Narrow phase: produces real contacts from candidate shape pairs.
//!
//! Stage 4.0 ships sphere-sphere and sphere-plane. Box / capsule / GJK
//! follow in 4.x.

use crate::math::{Real, Vec3};
use crate::sim::{Model, ShapeGeometry, State};

use super::broad_phase::ShapePair;
use super::shape_world::shape_world_transform;

/// Output of the narrow phase. The contact normal points from `a` to
/// `b`; `depth > 0` means penetration. Caller is responsible for
/// pushing this into a `Contacts` buffer.
#[derive(Copy, Clone, Debug)]
pub struct ContactManifold {
    pub a_shape: u32,
    pub b_shape: u32,
    /// World-space contact point (mid-plane between the two surfaces).
    pub point: Vec3,
    /// Unit normal, points from `a` toward `b` (so we can correct
    /// `b += depth * normal` and `a -= depth * normal`).
    pub normal: Vec3,
    /// Penetration depth (m). Always non-negative on output.
    pub depth: Real,
    pub friction: Real,
    pub restitution: Real,
}

/// Generate contacts for one candidate pair. Returns `None` if there
/// is no contact (separation > 0). Stage-4.0 supports sphere-sphere
/// and sphere-plane (in either order).
pub fn pair_contact(
    model: &Model,
    state: &State,
    pair: ShapePair,
    default_friction: Real,
    default_restitution: Real,
) -> Option<ContactManifold> {
    let geom_a = model.shape_geometry[pair.a as usize];
    let geom_b = model.shape_geometry[pair.b as usize];
    let xf_a = shape_world_transform(model, state, pair.a as usize);
    let xf_b = shape_world_transform(model, state, pair.b as usize);

    let manifold = match (geom_a, geom_b) {
        (ShapeGeometry::Sphere { radius: ra }, ShapeGeometry::Sphere { radius: rb }) => {
            sphere_sphere(xf_a.position, ra, xf_b.position, rb, pair.a, pair.b)
        }
        (ShapeGeometry::Sphere { radius }, ShapeGeometry::Plane { normal }) => {
            sphere_plane(xf_a.position, radius, xf_b.position, xf_b.rotation * normal, pair.a, pair.b)
        }
        (ShapeGeometry::Plane { normal }, ShapeGeometry::Sphere { radius }) => {
            // Same as above but flip a/b so normal still points a→b.
            sphere_plane(
                xf_b.position,
                radius,
                xf_a.position,
                xf_a.rotation * normal,
                pair.b,
                pair.a,
            )
            .map(flip_manifold)
        }
        (ShapeGeometry::Plane { .. }, ShapeGeometry::Plane { .. }) => None,
    };

    manifold.map(|mut m| {
        m.friction = default_friction;
        m.restitution = default_restitution;
        m
    })
}

fn sphere_sphere(
    pa: Vec3,
    ra: Real,
    pb: Vec3,
    rb: Real,
    a_shape: u32,
    b_shape: u32,
) -> Option<ContactManifold> {
    let delta = pb - pa;
    let d_sq = delta.length_squared();
    let r_sum = ra + rb;
    if d_sq >= r_sum * r_sum {
        return None;
    }
    let dist = d_sq.sqrt();
    // Pick a deterministic fallback normal when the centres coincide.
    let normal = if dist > 1e-12 {
        delta / dist
    } else {
        Vec3::Y
    };
    let depth = r_sum - dist;
    let point = pa + normal * (ra - 0.5 * depth);
    Some(ContactManifold {
        a_shape,
        b_shape,
        point,
        normal,
        depth,
        friction: 0.0,
        restitution: 0.0,
    })
}

fn sphere_plane(
    sphere_p: Vec3,
    radius: Real,
    plane_p: Vec3,
    plane_normal: Vec3,
    sphere_shape: u32,
    plane_shape: u32,
) -> Option<ContactManifold> {
    let signed = (sphere_p - plane_p).dot(plane_normal);
    if signed >= radius {
        return None;
    }
    let depth = radius - signed;
    // Normal points from the sphere TOWARD the plane (so corrections
    // push the sphere out of the plane, away from `plane_normal`).
    let normal = -plane_normal;
    let point = sphere_p + normal * (radius - 0.5 * depth);
    Some(ContactManifold {
        a_shape: sphere_shape,
        b_shape: plane_shape,
        point,
        normal,
        depth,
        friction: 0.0,
        restitution: 0.0,
    })
}

fn flip_manifold(mut m: ContactManifold) -> ContactManifold {
    std::mem::swap(&mut m.a_shape, &mut m.b_shape);
    m.normal = -m.normal;
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sphere_sphere_overlap() {
        let m = sphere_sphere(Vec3::ZERO, 1.0, Vec3::new(1.5, 0.0, 0.0), 1.0, 0, 1).unwrap();
        // r_sum = 2, dist = 1.5 → depth = 0.5
        assert!((m.depth - 0.5).abs() < 1e-9);
        assert!((m.normal - Vec3::X).length() < 1e-9);
    }

    #[test]
    fn sphere_sphere_separated() {
        assert!(sphere_sphere(Vec3::ZERO, 1.0, Vec3::new(3.0, 0.0, 0.0), 1.0, 0, 1).is_none());
    }

    #[test]
    fn sphere_above_plane_no_contact() {
        // sphere at y=1, radius 0.5; plane at y=0 with +Y normal. signed = 1, radius = 0.5 → no contact.
        let m = sphere_plane(
            Vec3::new(0.0, 1.0, 0.0),
            0.5,
            Vec3::ZERO,
            Vec3::Y,
            0,
            1,
        );
        assert!(m.is_none());
    }

    #[test]
    fn sphere_through_plane_contacts() {
        // sphere at y=0.3, radius 0.5; plane at y=0 with +Y normal. signed=0.3, depth=0.2.
        let m = sphere_plane(
            Vec3::new(0.0, 0.3, 0.0),
            0.5,
            Vec3::ZERO,
            Vec3::Y,
            0,
            1,
        )
        .unwrap();
        assert!((m.depth - 0.2).abs() < 1e-9);
        // Normal points from sphere toward plane → -Y.
        assert!((m.normal + Vec3::Y).length() < 1e-9);
    }
}
