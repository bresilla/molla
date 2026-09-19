//! Stable-handle collider geometry queries in collider and world frames.

use std::collections::BTreeMap;

use molla_core::{Error, Result};
use molla_math::{Real, Transform, Vec3};
use molla_sim::runtime::{ColliderGeometry, ColliderHandle, RigidScene};

use crate::{Aabb, convex_hull, shape_world::shape_aabb_at_pose};

#[derive(Clone, Debug, PartialEq)]
pub enum ColliderShapeView {
    Cuboid {
        half_extents: Vec3,
    },
    Ball {
        radius: Real,
    },
    Capsule {
        a: Vec3,
        b: Vec3,
        radius: Real,
    },
    Cylinder {
        half_height: Real,
        radius: Real,
    },
    RoundCylinder {
        half_height: Real,
        radius: Real,
        border_radius: Real,
    },
    ConvexPolyhedron {
        points: Vec<Vec3>,
        edges: Vec<[u32; 2]>,
    },
    Other,
}

/// Bounds in the collider's geometry frame, excluding its local pose.
pub fn collider_local_aabb(scene: &RigidScene, id: ColliderHandle) -> Result<Aabb> {
    collider_aabb_at_pose(scene, id, Transform::IDENTITY)
}

pub fn collider_world_aabb(scene: &RigidScene, id: ColliderHandle) -> Result<Aabb> {
    collider_aabb_at_pose(scene, id, scene.collider_world_pose(id)?)
}

/// Bounds with the collider's entire pose replaced by `pose`.
pub fn collider_aabb_at_pose(
    scene: &RigidScene,
    id: ColliderHandle,
    pose: Transform,
) -> Result<Aabb> {
    if !pose.position.is_finite()
        || !pose.rotation.is_finite()
        || (pose.rotation.length_squared() - 1.0).abs() > 1e-6
    {
        return Err(Error::Build("invalid collider query pose".into()));
    }
    let desc = scene
        .collider(id)
        .ok_or_else(|| Error::Build("unknown collider".into()))?;
    let indices = scene
        .collider_indices(id)
        .ok_or_else(|| Error::Build("unknown collider".into()))?;
    let pose = pose.compose(&desc.geometry.geometry_transform()?);
    let mut bounds = Aabb {
        min: Vec3::splat(Real::INFINITY),
        max: Vec3::splat(Real::NEG_INFINITY),
    };
    for index in indices {
        let part = shape_aabb_at_pose(scene.model(), index.index(), pose, 0.0);
        bounds.min = bounds.min.min(part.min);
        bounds.max = bounds.max.max(part.max);
    }
    Ok(bounds)
}

pub fn collider_shape_view(scene: &RigidScene, id: ColliderHandle) -> Result<ColliderShapeView> {
    let desc = scene
        .collider(id)
        .ok_or_else(|| Error::Build("unknown collider".into()))?;
    Ok(match &desc.geometry {
        ColliderGeometry::Ball { radius } => ColliderShapeView::Ball { radius: *radius },
        ColliderGeometry::Cuboid { half_extents } => ColliderShapeView::Cuboid {
            half_extents: *half_extents,
        },
        ColliderGeometry::Capsule {
            half_height,
            radius,
        } => ColliderShapeView::Capsule {
            a: -Vec3::Y * *half_height,
            b: Vec3::Y * *half_height,
            radius: *radius,
        },
        ColliderGeometry::CapsuleEndpoints { a, b, radius } => ColliderShapeView::Capsule {
            a: *a,
            b: *b,
            radius: *radius,
        },
        ColliderGeometry::Cylinder {
            half_height,
            radius,
        } => ColliderShapeView::Cylinder {
            half_height: *half_height,
            radius: *radius,
        },
        ColliderGeometry::RoundCylinder {
            half_height,
            radius,
            border_radius,
        } => ColliderShapeView::RoundCylinder {
            half_height: *half_height,
            radius: *radius,
            border_radius: *border_radius,
        },
        ColliderGeometry::ConvexHull(points) => {
            let hull = convex_hull(points);
            let mut edges: BTreeMap<[u32; 2], Vec<Vec3>> = BTreeMap::new();
            for [a, b, c] in hull.faces {
                let normal = (hull.vertices[b as usize] - hull.vertices[a as usize])
                    .cross(hull.vertices[c as usize] - hull.vertices[a as usize])
                    .normalize_or_zero();
                for [i, j] in [[a, b], [b, c], [c, a]] {
                    edges.entry([i.min(j), i.max(j)]).or_default().push(normal);
                }
            }
            let edges = edges
                .into_iter()
                .filter_map(|(edge, normals)| {
                    (normals.len() != 2 || normals[0].dot(normals[1]) < 1.0 - 1e-10).then_some(edge)
                })
                .collect();
            ColliderShapeView::ConvexPolyhedron {
                points: hull.vertices,
                edges,
            }
        }
        ColliderGeometry::TriMesh(_)
        | ColliderGeometry::Heightfield(_)
        | ColliderGeometry::ConvexParts(_) => ColliderShapeView::Other,
    })
}
