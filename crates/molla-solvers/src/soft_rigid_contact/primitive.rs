use super::SoftRigidContact;
use molla_core::{Error, Result};
use molla_math::{Real, Vec3};
use molla_sim::runtime::{ColliderGeometry, ColliderHandle, RigidScene};
use molla_sim::{Model, State};

/// Gather surface-node contacts against explicit balls, boxes and Y-axis cylinders.
pub fn gather_primitive_contacts(
    model: &Model,
    state: &State,
    surface_nodes: &[u32],
    scene: &RigidScene,
    colliders: &[ColliderHandle],
    margin: Real,
) -> Result<Vec<SoftRigidContact>> {
    let positions = state.particle_q.host()?;
    let radii = model.particle_radius.host()?;
    if !margin.is_finite()
        || margin < 0.0
        || positions.len() != model.particle_count
        || radii.len() != positions.len()
        || surface_nodes.iter().any(|&n| n as usize >= positions.len())
        || positions.iter().any(|v| !v.is_finite())
        || radii.iter().any(|r| !r.is_finite() || *r < 0.0)
    {
        return Err(Error::Solver("invalid FEM surface contact inputs".into()));
    }
    let mut nodes = surface_nodes.to_vec();
    nodes.sort_unstable();
    nodes.dedup();
    let mut colliders = colliders.to_vec();
    colliders.sort_unstable();
    colliders.dedup();
    let mut contacts = Vec::new();
    for id in colliders {
        let collider = scene
            .collider(id)
            .ok_or_else(|| Error::Solver("missing FEM collider".into()))?;
        if !collider.enabled || collider.sensor {
            continue;
        }
        if collider
            .parent
            .is_some_and(|b| scene.body(b).is_none_or(|b| !b.enabled))
        {
            continue;
        }
        if !matches!(
            collider.geometry,
            ColliderGeometry::Ball { .. }
                | ColliderGeometry::Cuboid { .. }
                | ColliderGeometry::Cylinder { .. }
        ) {
            return Err(Error::Solver(
                "unsupported FEM contact collider geometry".into(),
            ));
        }
        let pose = scene.collider_world_pose(id)?;
        let body = collider
            .parent
            .and_then(|b| scene.body_index(b))
            .map(|b| b.index());
        for &node in &nodes {
            let particle = node as usize;
            let local = pose.rotation.conjugate() * (positions[particle] - pose.position);
            let (normal, distance) = distance(&collider.geometry, local);
            let separation = distance - radii[particle];
            if separation <= margin {
                contacts.push(SoftRigidContact {
                    particle,
                    body,
                    normal: pose.rotation * normal,
                    separation,
                    friction: collider.friction,
                });
            }
        }
    }
    Ok(contacts)
}

fn distance(geometry: &ColliderGeometry, p: Vec3) -> (Vec3, Real) {
    match *geometry {
        ColliderGeometry::Ball { radius } => {
            let length = p.length();
            (
                if length > 1e-12 { p / length } else { Vec3::Y },
                length - radius,
            )
        }
        ColliderGeometry::Cuboid { half_extents } => {
            let q = p.abs() - half_extents;
            let outside = q.max(Vec3::ZERO);
            let length = outside.length();
            let sign = Vec3::new(
                1.0_f64.copysign(p.x),
                1.0_f64.copysign(p.y),
                1.0_f64.copysign(p.z),
            );
            let normal = if length > 1e-12 {
                sign * outside / length
            } else {
                let axis = if q.x >= q.y && q.x >= q.z {
                    0
                } else if q.y >= q.z {
                    1
                } else {
                    2
                };
                let mut n = Vec3::ZERO;
                n[axis] = sign[axis];
                n
            };
            (normal, length + q.max_element().min(0.0))
        }
        ColliderGeometry::Cylinder {
            half_height,
            radius,
        } => {
            let radial = Vec3::new(p.x, 0.0, p.z);
            let length = radial.length();
            let side = if length > 1e-12 {
                radial / length
            } else {
                Vec3::X
            };
            let cap = Vec3::Y * 1.0_f64.copysign(p.y);
            let dr = length - radius;
            let dy = p.y.abs() - half_height;
            let outside = dr.max(0.0).hypot(dy.max(0.0));
            let normal = if outside > 1e-12 {
                (side * dr.max(0.0) + cap * dy.max(0.0)) / outside
            } else if dr >= dy {
                side
            } else {
                cap
            };
            (normal, outside + dr.max(dy).min(0.0))
        }
        _ => unreachable!("validated contact geometry"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitive_distances_have_unit_normals_and_correct_surfaces() {
        let shapes = [
            ColliderGeometry::Ball { radius: 0.5 },
            ColliderGeometry::Cuboid {
                half_extents: Vec3::splat(0.5),
            },
            ColliderGeometry::Cylinder {
                half_height: 0.4,
                radius: 0.5,
            },
        ];
        for shape in &shapes {
            for p in [
                Vec3::ZERO,
                Vec3::new(0.7, 0.8, -0.4),
                Vec3::new(-0.1, 0.2, 0.0),
            ] {
                let (normal, d) = distance(shape, p);
                assert!((normal.length() - 1.0).abs() < 1e-12);
                assert!(distance(shape, p - d * normal).1.abs() < 1e-12);
            }
        }
    }
}
