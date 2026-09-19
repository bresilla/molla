//! Collider volume integrals and explicit runtime body-mass recomputation.

use molla_core::{Error, Result};
use molla_math::{Mat3, Real, Vec3};
use molla_sim::runtime::{
    BodyHandle, ColliderDesc, ColliderGeometry, RigidScene, validate_convex_points,
};
use molla_sim::{ColliderMass, MassProperties};
use parry3d_f64::{math::Vector, shape::SharedShape};
use std::collections::BTreeMap;

/// Properties in the authored collider frame, before its local pose is applied.
/// Rounded cylinders use the inner cylinder; compound pieces contribute additively.
pub fn collider_mass_properties(desc: &ColliderDesc) -> Result<MassProperties> {
    desc.mass.validate()?;
    let unit = unit_properties(&desc.geometry)?;
    let (mass, inertia) = match desc.mass {
        ColliderMass::Density(value) => (unit.mass * value, unit.inertia * value),
        ColliderMass::Mass(value) if unit.mass > 0.0 => (value, (unit.inertia / unit.mass) * value),
        ColliderMass::Mass(value) => (value, Mat3::ZERO),
    };
    let result = MassProperties {
        mass,
        local_com: unit.local_com,
        inertia,
    };
    result.validate()?;
    Ok(result)
}

/// Sum enabled colliders, including sensors, and the supplied additional component.
/// The previous body total is not part of this sum.
pub fn body_mass_properties(
    scene: &RigidScene,
    body: BodyHandle,
    additional: MassProperties,
) -> Result<MassProperties> {
    if scene.body_index(body).is_none() {
        return Err(Error::Build("unknown runtime body handle".into()));
    }
    additional.validate()?;
    let mut total = MassProperties::ZERO;
    for (_, collider) in scene
        .colliders()
        .filter(|(_, c)| c.parent == Some(body) && c.enabled)
    {
        total = total.combined(collider_mass_properties(collider)?.transformed(collider.pose)?)?;
    }
    total.combined(additional)
}

/// Recompute explicitly; the caller owns the replaceable additional component.
pub fn recompute_body_mass(
    scene: &mut RigidScene,
    body: BodyHandle,
    additional: MassProperties,
) -> Result<()> {
    let total = body_mass_properties(scene, body, additional)?;
    scene.set_body_mass_properties(body, total)
}

fn unit_properties(geometry: &ColliderGeometry) -> Result<MassProperties> {
    let positive = |value: Real| -> Result<()> {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::Build("invalid collider mass geometry".into()));
        }
        Ok(())
    };
    let shape = match geometry {
        ColliderGeometry::Ball { radius } => {
            positive(*radius)?;
            SharedShape::ball(*radius)
        }
        ColliderGeometry::Cuboid { half_extents: h } => {
            for value in h.to_array() {
                positive(value)?;
            }
            SharedShape::cuboid(h.x, h.y, h.z)
        }
        ColliderGeometry::Capsule {
            half_height,
            radius,
        } => {
            positive(*radius)?;
            if !half_height.is_finite() || *half_height < 0.0 {
                return Err(Error::Build("invalid capsule height".into()));
            }
            SharedShape::capsule_y(*half_height, *radius)
        }
        ColliderGeometry::CapsuleEndpoints { a, b, radius } => {
            let canonical = ColliderGeometry::Capsule {
                half_height: (*b - *a).length() * 0.5,
                radius: *radius,
            };
            return unit_properties(&canonical)?.transformed(geometry.geometry_transform()?);
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
            positive(*half_height)?;
            positive(*radius)?;
            if let ColliderGeometry::RoundCylinder { border_radius, .. } = geometry
                && (!border_radius.is_finite() || *border_radius < 0.0)
            {
                return Err(Error::Build("invalid rounded cylinder border".into()));
            }
            SharedShape::cylinder(*half_height, *radius)
        }
        ColliderGeometry::ConvexHull(points) => return mesh_properties(points, None),
        ColliderGeometry::ConvexParts(parts) => {
            if parts.is_empty() {
                return Err(Error::Build("empty convex compound".into()));
            }
            let mut total = MassProperties::ZERO;
            for points in parts {
                total = total.combined(mesh_properties(points, None)?)?;
            }
            return Ok(total);
        }
        ColliderGeometry::TriMesh(mesh) => {
            return mesh_properties(&mesh.vertices, Some(&mesh.indices));
        }
        ColliderGeometry::Heightfield(field) => {
            field.validate()?;
            return Ok(MassProperties::ZERO);
        }
    };
    from_parry(shape.mass_properties(1.0))
}

fn mesh_properties(points: &[Vec3], indices: Option<&[[u32; 3]]>) -> Result<MassProperties> {
    if points.is_empty() || points.iter().any(|p| !p.is_finite()) {
        return Err(Error::Build("invalid mass-properties mesh vertices".into()));
    }
    let (points, indices) = if let Some(indices) = indices {
        if indices.is_empty()
            || indices
                .iter()
                .flatten()
                .any(|&i| i as usize >= points.len())
        {
            return Err(Error::Build("invalid mass-properties mesh indices".into()));
        }
        let mut used = BTreeMap::new();
        for &i in indices.iter().flatten() {
            used.insert(i, 0);
        }
        let mut compact = Vec::with_capacity(used.len());
        for (&i, mapped) in &mut used {
            *mapped = compact.len() as u32;
            compact.push(points[i as usize]);
        }
        let triangles = indices
            .iter()
            .map(|t| t.map(|i| used[&i]))
            .collect::<Vec<_>>();
        (compact, Some(triangles))
    } else {
        validate_convex_points(points)?;
        (points.to_vec(), None)
    };
    let lo = points
        .iter()
        .copied()
        .fold(Vec3::splat(Real::INFINITY), Vec3::min);
    let hi = points
        .iter()
        .copied()
        .fold(Vec3::splat(Real::NEG_INFINITY), Vec3::max);
    let center = lo * 0.5 + hi * 0.5;
    let scale = (hi - lo).max_element();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Build("degenerate mass-properties mesh".into()));
    }
    let points: Vec<Vector> = points
        .iter()
        .map(|p| {
            let p = (*p - center) / scale;
            Vector::new(p.x, p.y, p.z)
        })
        .collect();
    let properties = std::panic::catch_unwind(|| -> Result<_> {
        let shape = if let Some(indices) = indices {
            SharedShape::trimesh(points, indices)
                .map_err(|e| Error::Build(format!("mass-properties mesh: {e}")))?
        } else {
            SharedShape::convex_hull(&points)
                .ok_or_else(|| Error::Build("mass-properties hull failed".into()))?
        };
        from_parry(shape.mass_properties(1.0))
    })
    .map_err(|_| Error::Build("mass-properties mesh calculation failed".into()))??;
    let result = MassProperties {
        mass: properties.mass * scale.powi(3),
        local_com: properties.local_com * scale + center,
        inertia: properties.inertia * scale.powi(5),
    };
    result.validate()?;
    Ok(result)
}

fn from_parry(properties: parry3d_f64::mass_properties::MassProperties) -> Result<MassProperties> {
    let com = properties.local_com;
    let result = MassProperties {
        mass: properties.mass(),
        local_com: Vec3::new(com.x, com.y, com.z),
        inertia: Mat3::from_cols_array(&properties.reconstruct_inertia_matrix().to_cols_array()),
    };
    result.validate()?;
    Ok(result)
}
