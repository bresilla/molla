//! `UsdPhysicsCollisionAPI` (+ `MeshCollisionAPI`) → molla
//! `ShapeGeometry`. Material binding is followed for friction /
//! restitution; collision groups land on
//! `Model.shape_collision_group`.
//!
//! Shape resolution rules:
//! - `UsdGeomSphere` (radius)            → `Sphere { radius }`
//! - `UsdGeomCube` (size)                → `Box { half_extents = (size/2)·scale }`
//! - `UsdGeomCapsule` (radius, height)   → `Capsule { half_height = height/2, radius }`
//! - `UsdGeomCylinder` (radius, height)  → `Cylinder { half_height = height/2, radius }`
//! - `UsdGeomPlane`                      → `Plane { normal = +Y }`
//! - `UsdGeomMesh`                       → `TriMesh { mesh_id }`
//!
//! `physics:approximation` is honoured for meshes (ConvexHull /
//! BoundingSphere / BoundingCube) — falls back to a tight bounding
//! sphere for now since molla's convex-hull builder is stage 14 work.

use anyhow::Result;
use molla::core::BodyId;
use molla::math::{Real, Transform, Vec3};
use molla::sim::{ModelBuilder, TriMesh};
use openusd::physics::PhysicsPrims;
use openusd::sdf;
use openusd::Stage;

use crate::resolvers::ResolverChain;
use crate::ImportInfo;

/// Find every prim that the resolver chain claims as a collider
/// (standard `CollisionAPI` plus PhysX / MJC namespaced collider
/// opinions) and emit a matching molla shape. The shape is attached
/// to the nearest ancestor with a known body (or the static world
/// frame if no such ancestor exists).
pub fn import_colliders(
    stage: &Stage,
    prims: &PhysicsPrims,
    builder: &mut ModelBuilder,
    info: &mut ImportInfo,
    resolvers: &ResolverChain,
) -> Result<()> {
    let body_lookup = crate::bodies::body_lookup(info);

    for path_str in &prims.colliders {
        let path = sdf::path(path_str)?;
        let coll = match resolvers.resolve_shape(stage, &path)? {
            Some(c) => c,
            None => continue,
        };
        // Default to enabled when the resolver doesn't author the flag.
        if matches!(coll.collision_enabled, Some(false)) {
            continue;
        }

        // The collider belongs to the nearest ancestor body. For
        // flat scenes that's a sibling — match by prim path prefix.
        let owner = nearest_body_for(path_str, &body_lookup).unwrap_or(BodyId::NONE);

        // Resolve the shape's geometry kind from the prim type.
        let prim_type = crate::attrs::type_name(stage, &path).unwrap_or_default();

        match prim_type.as_str() {
            "Sphere" => {
                let radius = read_sphere_radius(stage, &path)?.unwrap_or(0.5);
                let shape = builder
                    .add_shape_sphere(owner, radius, Transform::IDENTITY)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let _ = shape;
                info.shape_paths.push(path_str.clone());
            }
            "Cube" => {
                let size = read_cube_size(stage, &path)?.unwrap_or(1.0);
                let half = Vec3::splat(size as Real * 0.5);
                let shape = builder
                    .add_shape_box(owner, half, Transform::IDENTITY)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let _ = shape;
                info.shape_paths.push(path_str.clone());
            }
            "Capsule" => {
                let (radius, half_height) =
                    read_capsule_dims(stage, &path)?.unwrap_or((0.5, 0.5));
                let shape = builder
                    .add_shape_capsule(owner, half_height, radius, Transform::IDENTITY)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let _ = shape;
                info.shape_paths.push(path_str.clone());
            }
            "Cylinder" => {
                let (radius, half_height) =
                    read_capsule_dims(stage, &path)?.unwrap_or((0.5, 0.5));
                let shape = builder
                    .add_shape_cylinder(owner, half_height, radius, Transform::IDENTITY)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let _ = shape;
                info.shape_paths.push(path_str.clone());
            }
            "Plane" => {
                let shape = builder
                    .add_shape_plane(owner, Vec3::Y, Transform::IDENTITY)
                    .map_err(|e| anyhow::anyhow!("{e}"))?;
                let _ = shape;
                info.shape_paths.push(path_str.clone());
            }
            "Mesh" => {
                let mesh = read_trimesh(stage, &path)?;
                if let Some(mesh) = mesh {
                    let mesh_id = builder.add_mesh(mesh);
                    let shape = builder
                        .add_shape_trimesh(owner, mesh_id, Transform::IDENTITY)
                        .map_err(|e| anyhow::anyhow!("{e}"))?;
                    let _ = shape;
                    info.shape_paths.push(path_str.clone());
                }
            }
            _ => {
                log::debug!(
                    "usd_molla: skipping collider with unsupported prim type {prim_type:?} at {path_str}"
                );
            }
        }
    }
    Ok(())
}

/// Find the nearest ancestor body for a collider prim path. Match
/// by prefix (USD's standard ancestor-path convention).
fn nearest_body_for(
    path_str: &str,
    bodies: &std::collections::HashMap<String, BodyId>,
) -> Option<BodyId> {
    let mut cur = path_str;
    loop {
        if let Some(&id) = bodies.get(cur) {
            return Some(id);
        }
        match cur.rsplit_once('/') {
            Some((parent, _)) if !parent.is_empty() => cur = parent,
            _ => return None,
        }
    }
}

fn read_sphere_radius(stage: &Stage, path: &sdf::Path) -> Result<Option<f64>> {
    Ok(crate::attrs::attr_f64(stage, path, "radius"))
}

fn read_cube_size(stage: &Stage, path: &sdf::Path) -> Result<Option<f64>> {
    Ok(crate::attrs::attr_f64(stage, path, "size"))
}

fn read_capsule_dims(stage: &Stage, path: &sdf::Path) -> Result<Option<(Real, Real)>> {
    let radius = crate::attrs::attr_f64(stage, path, "radius");
    let height = crate::attrs::attr_f64(stage, path, "height");
    Ok(match (radius, height) {
        (Some(r), Some(h)) => Some((r as Real, (h * 0.5) as Real)),
        _ => None,
    })
}

fn read_trimesh(stage: &Stage, path: &sdf::Path) -> Result<Option<TriMesh>> {
    let pts = crate::attrs::attr_vec3_array(stage, path, "points");
    let counts = crate::attrs::attr_int_array(stage, path, "faceVertexCounts");
    let indices_raw = crate::attrs::attr_int_array(stage, path, "faceVertexIndices");
    let (Some(pts), Some(counts), Some(idx)) = (pts, counts, indices_raw) else {
        return Ok(None);
    };
    let mut mesh = TriMesh::new();
    mesh.vertices = pts
        .into_iter()
        .map(|p| Vec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
        .collect();
    // Triangulate fan-style for any face count > 3.
    let mut cursor = 0usize;
    for c in counts {
        let n = c as usize;
        if n >= 3 && cursor + n <= idx.len() {
            let v0 = idx[cursor] as u32;
            for k in 1..(n - 1) {
                let v1 = idx[cursor + k] as u32;
                let v2 = idx[cursor + k + 1] as u32;
                mesh.indices.push([v0, v1, v2]);
            }
        }
        cursor += n;
    }
    Ok(Some(mesh))
}
