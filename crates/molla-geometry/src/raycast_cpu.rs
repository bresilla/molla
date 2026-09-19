//! CPU nearest-hit queries over live rigid-scene colliders.

use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_sim::runtime::{BodyHandle, ColliderGeometry, ColliderHandle, RigidScene};
use molla_sim::{Heightfield, TriMesh};

use crate::{Aabb, MeshBvh, collider_query::collider_local_aabb, convex_hull};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub collider: ColliderHandle,
    pub body: Option<BodyHandle>,
    /// Physical distance in metres, independent of the input direction's magnitude.
    pub distance: Real,
    pub point: Vec3,
    pub normal: Vec3,
}

/// Enabled sensors participate; origins inside solid shapes return distance zero.
pub fn cast_ray(
    scene: &RigidScene,
    origin: Vec3,
    direction: Vec3,
    max_distance: Real,
) -> Result<Option<RayHit>> {
    cast_ray_filtered(scene, origin, direction, max_distance, |_| true)
}

/// `include` selects eligible colliders before geometry work.
pub fn cast_ray_filtered(
    scene: &RigidScene,
    origin: Vec3,
    direction: Vec3,
    max_distance: Real,
    mut include: impl FnMut(ColliderHandle) -> bool,
) -> Result<Option<RayHit>> {
    let magnitude = direction.abs().max_element();
    if !origin.is_finite()
        || !direction.is_finite()
        || magnitude == 0.0
        || max_distance.is_nan()
        || max_distance < 0.0
    {
        return Err(Error::Build("invalid ray".into()));
    }
    let direction = (direction / magnitude).normalize();
    let mut best: Option<RayHit> = None;
    let mut limit = max_distance;
    for (id, desc) in scene.colliders() {
        if !desc.enabled || !include(id) {
            continue;
        }
        let pose = scene.collider_world_pose(id)?;
        let o = pose.inverse_transform_point(origin);
        let d = pose.rotation.conjugate() * direction;
        let bounds = match &desc.geometry {
            ColliderGeometry::Heightfield(field) => Aabb {
                min: Vec3::new(field.origin.x, Real::NEG_INFINITY, field.origin.z),
                max: Vec3::new(
                    field.origin.x + (field.nx - 1) as Real * field.cell_size,
                    Real::INFINITY,
                    field.origin.z + (field.nz - 1) as Real * field.cell_size_z,
                ),
            },
            _ => collider_local_aabb(scene, id)?,
        };
        let Some((entry, exit)) = ray_bounds(o, d, bounds, limit) else {
            continue;
        };
        let hit = match &desc.geometry {
            ColliderGeometry::TriMesh(mesh) => ray_mesh(mesh, o, d, limit),
            ColliderGeometry::Heightfield(field) => ray_heightfield(field, o, d, entry, exit),
            ColliderGeometry::ConvexHull(points) => ray_hull(points, o, d, limit),
            geometry => ray_primitive(geometry, o, d, entry, exit),
        };
        if let Some((distance, normal)) = hit
            && (best.is_none() || distance < limit)
        {
            limit = distance;
            best = Some(RayHit {
                collider: id,
                body: desc.parent,
                distance,
                point: origin + direction * distance,
                normal: (pose.rotation * normal).normalize_or_zero(),
            });
        }
    }
    Ok(best)
}

fn ray_bounds(o: Vec3, d: Vec3, bounds: Aabb, max: Real) -> Option<(Real, Real)> {
    let (mut lo, mut hi) = (0.0_f64, max);
    for i in 0..3 {
        if d[i] == 0.0 {
            if o[i] < bounds.min[i] || o[i] > bounds.max[i] {
                return None;
            }
        } else {
            let a = (bounds.min[i] - o[i]) / d[i];
            let b = (bounds.max[i] - o[i]) / d[i];
            lo = lo.max(a.min(b));
            hi = hi.min(a.max(b));
            if lo > hi {
                return None;
            }
        }
    }
    Some((lo, hi))
}

fn primitive_distance(geometry: &ColliderGeometry, p: Vec3) -> (Real, Vec3) {
    match geometry {
        ColliderGeometry::Ball { radius } => (p.length() - radius, p.normalize_or_zero()),
        ColliderGeometry::Cuboid { half_extents } => {
            let q = p.abs() - *half_extents;
            let outside = q.max(Vec3::ZERO);
            let distance = outside.length() + q.max_element().min(0.0);
            let n = if outside.length_squared() > 0.0 {
                outside.normalize()
            } else {
                let mut axis = Vec3::ZERO;
                let i = if q.x >= q.y && q.x >= q.z {
                    0
                } else if q.y >= q.z {
                    1
                } else {
                    2
                };
                axis[i] = 1.0;
                axis
            };
            (distance, n * p.signum())
        }
        ColliderGeometry::Capsule {
            half_height,
            radius,
        } => {
            let v = p - Vec3::Y * p.y.clamp(-half_height, *half_height);
            (v.length() - radius, v.normalize_or_zero())
        }
        ColliderGeometry::CapsuleEndpoints { a, b, radius } => {
            let ab = *b - *a;
            let t = if ab.length_squared() > 0.0 {
                ((p - *a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let v = p - (*a + ab * t);
            (v.length() - radius, v.normalize_or_zero())
        }
        ColliderGeometry::Cylinder {
            half_height,
            radius,
        }
        | ColliderGeometry::RoundCylinder {
            half_height,
            radius,
            ..
        } => {
            let border = match geometry {
                ColliderGeometry::RoundCylinder { border_radius, .. } => *border_radius,
                _ => 0.0,
            };
            let radial = Vec3::new(p.x, 0.0, p.z);
            let x = radial.length() - radius;
            let y = p.y.abs() - half_height;
            let normal = if x > 0.0 && y > 0.0 {
                (radial.normalize_or_zero() * x + Vec3::Y * p.y.signum() * y).normalize()
            } else if x > y {
                radial.normalize_or_zero()
            } else {
                Vec3::Y * p.y.signum()
            };
            (
                x.max(0.0).hypot(y.max(0.0)) + x.max(y).min(0.0) - border,
                normal,
            )
        }
        _ => unreachable!("non-primitive ray dispatch"),
    }
}

fn ray_primitive(
    geometry: &ColliderGeometry,
    o: Vec3,
    d: Vec3,
    entry: Real,
    exit: Real,
) -> Option<(Real, Vec3)> {
    if let ColliderGeometry::Ball { radius } = geometry {
        if o.length_squared() <= radius * radius {
            return Some((0.0, -d));
        }
        let center = -o.dot(d) / d.length_squared();
        let perpendicular = o + d * center;
        let disc = radius * radius - perpendicular.length_squared();
        if disc < 0.0 {
            return None;
        }
        let t = center - (disc / d.length_squared()).sqrt();
        return (t >= entry && t <= exit).then(|| (t, (o + d * t).normalize_or_zero()));
    }
    let sample = |t| primitive_distance(geometry, o + d * t);
    let (value, normal) = sample(entry);
    if value <= 0.0 {
        return Some((entry, if entry == 0.0 { -d } else { normal }));
    }
    let (mut a, mut b) = (entry, exit);
    let ratio = 0.618_033_988_749_894_9;
    let (mut c, mut e) = (b - ratio * (b - a), a + ratio * (b - a));
    let (mut fc, mut fe) = (sample(c).0, sample(e).0);
    for _ in 0..96 {
        if fc < fe {
            b = e;
            e = c;
            fe = fc;
            c = b - ratio * (b - a);
            fc = sample(c).0;
        } else {
            a = c;
            c = e;
            fc = fe;
            e = a + ratio * (b - a);
            fe = sample(e).0;
        }
    }
    let mut minimum = (a + b) * 0.5;
    if sample(exit).0 < sample(minimum).0 {
        minimum = exit;
    }
    let (depth, normal) = sample(minimum);
    let tolerance = 1e-12;
    if depth > tolerance {
        return None;
    }
    if depth > 0.0 {
        return Some((minimum, normal));
    }
    let (mut lo, mut hi) = (entry, minimum);
    for _ in 0..80 {
        let mid = (lo + hi) * 0.5;
        if sample(mid).0 > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Some((hi, sample(hi).1))
}

fn ray_hull(points: &[Vec3], o: Vec3, d: Vec3, max: Real) -> Option<(Real, Vec3)> {
    let hull = convex_hull(points);
    if hull.faces.is_empty() {
        return None;
    }
    let (mut enter, mut exit) = (0.0_f64, max);
    let mut normal = -d;
    for [a, b, c] in hull.faces {
        let p = hull.vertices[a as usize];
        let n = (hull.vertices[b as usize] - p)
            .cross(hull.vertices[c as usize] - p)
            .normalize_or_zero();
        let offset = n.dot(p - o);
        let speed = n.dot(d);
        if speed == 0.0 {
            if offset < 0.0 {
                return None;
            }
        } else if speed < 0.0 {
            let t = offset / speed;
            if t > enter {
                enter = t;
                normal = n;
            }
        } else {
            exit = exit.min(offset / speed);
        }
        if enter > exit {
            return None;
        }
    }
    Some((enter, normal))
}

fn ray_mesh(mesh: &TriMesh, o: Vec3, d: Vec3, max: Real) -> Option<(Real, Vec3)> {
    let bvh = MeshBvh::build(&mesh.vertices, &mesh.indices, 4);
    bvh.ray_cast(&mesh.vertices, &mesh.indices, o, d, max)
        .map(|(_, t, n)| {
            let n = n.normalize_or_zero();
            (t, if n.dot(d) > 0.0 { -n } else { n })
        })
}

fn ray_triangle(o: Vec3, d: Vec3, tri: [Vec3; 3], max: Real) -> Option<(Real, Vec3)> {
    let [a, b, c] = tri;
    let (t, normal) = crate::mesh_bvh::ray_triangle(o, d, a, b, c)?;
    if t < 0.0 || t > max {
        return None;
    }
    let normal = normal.normalize_or_zero();
    Some((t, if normal.dot(d) > 0.0 { -normal } else { normal }))
}

fn ray_heightfield(
    field: &Heightfield,
    o: Vec3,
    d: Vec3,
    entry: Real,
    exit: Real,
) -> Option<(Real, Vec3)> {
    let start = o + d * entry;
    let mut i = (((start.x - field.origin.x) / field.cell_size).floor() as i64)
        .clamp(0, field.nx as i64 - 2);
    let mut j = (((start.z - field.origin.z) / field.cell_size_z).floor() as i64)
        .clamp(0, field.nz as i64 - 2);
    let (sx, sz) = (d.x.signum() as i64, d.z.signum() as i64);
    for _ in 0..field.nx as u64 + field.nz as u64 + 2 {
        if i < 0 || j < 0 || i >= field.nx as i64 - 1 || j >= field.nz as i64 - 1 {
            break;
        }
        let mut closest: Option<(Real, Vec3)> = None;
        for triangle in field
            .cell_triangles(i as u32, j as u32)
            .into_iter()
            .flatten()
        {
            if let Some(hit) = ray_triangle(o, d, triangle, exit)
                && closest.is_none_or(|old| hit.0 < old.0)
            {
                closest = Some(hit);
            }
        }
        if closest.is_some() {
            return closest;
        }
        let tx = if d.x == 0.0 {
            Real::INFINITY
        } else {
            (field.origin.x + (i + i64::from(sx > 0)) as Real * field.cell_size - o.x) / d.x
        };
        let tz = if d.z == 0.0 {
            Real::INFINITY
        } else {
            (field.origin.z + (j + i64::from(sz > 0)) as Real * field.cell_size_z - o.z) / d.z
        };
        if tx.min(tz) > exit || !tx.min(tz).is_finite() {
            break;
        }
        if tx <= tz {
            i += sx;
        }
        if tz <= tx {
            j += sz;
        }
    }
    None
}
