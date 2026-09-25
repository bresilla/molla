//! Owned, revisioned geometry/color snapshots with shared local-space mesh BVHs.

use crate::readback::DeviceBuffer;
use crate::scene_geometry::{DEFAULT_QUERY_BYTES, SceneGeometry};
pub use crate::scene_identity::{
    SceneIdentitySnapshot, SensorBodyHandle, SensorSceneRemap, SensorShapeHandle,
};
use molla_compute::Engine;
use molla_core::{BodyId, Error, Result};
use molla_math::{Real, Transform, Vec3};
use molla_sim::{Model, ShapeGeometry, TriMesh};
use molla_storage::{Buffer, BufferUsage};
use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

static NEXT_SCENE: AtomicU64 = AtomicU64::new(1);

/// Owned-scene ray shape and LiDAR first/last index when any included geometry has an invalid pose.
pub const INVALID_SCENE_INDEX: i32 = -2;
/// Owned-scene ray distance and camera radial depth when the scene pose is invalid.
pub const INVALID_SCENE_DISTANCE: Real = -2.0;
/// Owned-scene camera shape ID when the scene pose is invalid; distinct from u32::MAX misses.
pub const INVALID_SCENE_SHAPE_ID: u32 = u32::MAX - 1;

/// Shape indices are scoped to this scene identity and geometry revision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SceneStamp {
    pub scene_id: u64,
    pub geometry_revision: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnsupportedGeometryPolicy {
    Reject,
    ReportAndSkip,
}

#[derive(Debug, Clone, Copy)]
pub struct SensorSceneOptions {
    pub unsupported: UnsupportedGeometryPolicy,
    pub max_vertices: usize,
    pub max_triangles: usize,
}

impl Default for SensorSceneOptions {
    fn default() -> Self {
        Self {
            unsupported: UnsupportedGeometryPolicy::Reject,
            max_vertices: 1_000_000,
            max_triangles: 1_000_000,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct UnsupportedShape {
    pub shape_index: u32,
    pub kind: &'static str,
}

/// Immutable geometry/colors; queries consume live body poses from the caller's State.
/// Replacing the snapshot invalidates all previous revision-scoped shape indices.
pub struct SensorScene {
    pub(crate) device: Arc<wgpu::Device>,
    pub(crate) model: Model,
    pub(crate) geometry: SceneGeometry,
    max_query_bytes: u64,
    stamp: SceneStamp,
    options: SensorSceneOptions,
    unsupported: Vec<UnsupportedShape>,
    identities: SceneIdentitySnapshot,
}

impl SensorScene {
    pub fn from_model(
        engine: &Engine,
        source: &Model,
        options: SensorSceneOptions,
    ) -> Result<Self> {
        Self::from_model_with_query_budget(engine, source, options, DEFAULT_QUERY_BYTES)
    }

    /// Limit packed shape/mesh GPU payload; excludes model copies, host trees and refit/ray storage.
    pub fn from_model_with_query_budget(
        engine: &Engine,
        source: &Model,
        options: SensorSceneOptions,
        max_query_bytes: u64,
    ) -> Result<Self> {
        let scene_id = NEXT_SCENE
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .map_err(|_| Error::Build("sensor scene identity exhausted".into()))?;
        let stamp = SceneStamp {
            scene_id,
            geometry_revision: 1,
        };
        Self::compile(engine, source, options, max_query_bytes, stamp, None, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn compile(
        engine: &Engine,
        source: &Model,
        options: SensorSceneOptions,
        max_query_bytes: u64,
        stamp: SceneStamp,
        previous: Option<&SceneIdentitySnapshot>,
        remap: Option<SensorSceneRemap<'_>>,
    ) -> Result<Self> {
        engine.check_device()?;
        let device = engine
            .device()
            .ok_or_else(|| Error::Gpu("sensor scene requires GPU".into()))?
            .clone();
        let n = source.shape_count;
        if [n, source.body_count].into_iter().any(|count| {
            (count as u64).checked_mul(32).is_none_or(|bytes| {
                bytes > device.limits().max_storage_buffer_binding_size as u64
                    || bytes > device.limits().max_buffer_size
            })
        }) {
            return Err(Error::Build(
                "sensor scene: shape/body capacity exceeds GPU limits".into(),
            ));
        }
        if n > i32::MAX as usize
            || source.shape_geometry.len() != n
            || source.shape_body.len() != n
            || source.shape_transform.len() != n
            || source.shape_color.len() != n
        {
            return Err(Error::Build(
                "sensor scene: inconsistent shape arrays".into(),
            ));
        }
        let bodies = source.shape_body.host()?;
        let transforms = source.shape_transform.host()?;
        let colors = source.shape_color.host()?;
        if colors
            .iter()
            .any(|color| !color.is_finite() || !color.as_vec3().is_finite())
        {
            return Err(Error::Build("sensor scene: non-finite shape color".into()));
        }
        for (&body, &pose) in bodies.iter().zip(transforms) {
            if body < -1 || body >= 0 && body as usize >= source.body_count {
                return Err(Error::Build(
                    "sensor scene: invalid shape body index".into(),
                ));
            }
            validate_pose(pose)?;
        }
        let identities =
            SceneIdentitySnapshot::build(stamp, source.body_count, bodies, previous, remap)?;
        let mut model = Model::empty();
        model.body_count = source.body_count;
        model.shape_count = n;
        model.shape_geometry = source.shape_geometry.clone();
        let mut kinds = Vec::with_capacity(n);
        let mut data = Vec::with_capacity(n);
        let mut remap = HashMap::new();
        let mut unsupported = Vec::new();
        let mut vertices = 0usize;
        let mut triangles = 0usize;
        for (shape, geometry) in source.shape_geometry.iter().copied().enumerate() {
            let mut mesh_source = None;
            let primitive = match geometry {
                ShapeGeometry::Sphere { radius } => Some((0, positive(Vec3::splat(radius))?)),
                ShapeGeometry::Plane { normal } => {
                    if !normal.is_finite() || normal.length_squared() < 1e-24 {
                        return Err(Error::Build("sensor scene: invalid plane".into()));
                    }
                    Some((1, (normal / normal.abs().max_element()).normalize()))
                }
                ShapeGeometry::Box { half_extents } => Some((2, positive(half_extents)?)),
                ShapeGeometry::Capsule {
                    half_height,
                    radius,
                } => Some((3, radial(half_height, radius, true)?)),
                ShapeGeometry::Cylinder {
                    half_height,
                    radius,
                } => Some((4, radial(half_height, radius, false)?)),
                ShapeGeometry::Cone {
                    half_height,
                    radius,
                } => Some((5, radial(half_height, radius, false)?)),
                ShapeGeometry::Ellipsoid { half_axes } => Some((6, positive(half_axes)?)),
                ShapeGeometry::TriMesh { mesh_id } => {
                    mesh_source = Some((0, mesh_id));
                    None
                }
                ShapeGeometry::Triangle { .. } => {
                    mesh_source = Some((1, shape as u32));
                    None
                }
                ShapeGeometry::Heightfield { field_id } => {
                    mesh_source = Some((2, field_id));
                    None
                }
                ShapeGeometry::ConvexHull { hull_id }
                    if source
                        .hulls
                        .get(hull_id as usize)
                        .is_some_and(|points| points.len() >= 4) =>
                {
                    mesh_source = Some((3, hull_id));
                    None
                }
                ShapeGeometry::RoundCylinder {
                    half_height,
                    radius,
                    border_radius,
                } => {
                    if !(border_radius as f32).is_finite() || border_radius < 0.0 {
                        return Err(Error::Build(
                            "sensor scene: invalid round cylinder border".into(),
                        ));
                    }
                    let dilated = ShapeGeometry::Cylinder {
                        half_height: half_height + border_radius,
                        radius: radius + border_radius,
                    };
                    model.shape_geometry[shape] = dilated;
                    Some((
                        4,
                        radial(half_height + border_radius, radius + border_radius, false)?,
                    ))
                }
                ShapeGeometry::ConvexHull { .. } | ShapeGeometry::Sdf { .. } => {
                    let kind = match geometry {
                        ShapeGeometry::ConvexHull { .. } => "ConvexHull",
                        _ => "Sdf",
                    };
                    if options.unsupported == UnsupportedGeometryPolicy::Reject {
                        return Err(Error::Build(format!(
                            "sensor scene: unsupported {kind} at shape {shape}"
                        )));
                    }
                    unsupported.push(UnsupportedShape {
                        shape_index: shape as u32,
                        kind,
                    });
                    None
                }
            };
            if let Some((kind, parameters)) = primitive {
                kinds.push(kind);
                data.push(parameters);
                continue;
            }
            kinds.push(0xff);
            data.push(Vec3::ZERO);
            if let Some(key) = mesh_source {
                let id = if let Some(&id) = remap.get(&key) {
                    id
                } else {
                    let mesh = match geometry {
                        ShapeGeometry::TriMesh { mesh_id } => {
                            let mesh = source.meshes.get(mesh_id as usize).ok_or_else(|| {
                                Error::Build("sensor scene: invalid mesh id".into())
                            })?;
                            check_budget(
                                vertices,
                                triangles,
                                mesh.vertices.len(),
                                mesh.indices.len(),
                                options,
                            )?;
                            mesh.clone()
                        }
                        ShapeGeometry::Triangle { v0, v1, v2 } => {
                            check_budget(vertices, triangles, 3, 1, options)?;
                            TriMesh {
                                vertices: vec![v0, v1, v2],
                                indices: vec![[0, 1, 2]],
                            }
                        }
                        ShapeGeometry::Heightfield { field_id } => {
                            let field =
                                source.heightfields.get(field_id as usize).ok_or_else(|| {
                                    Error::Build("sensor scene: invalid heightfield id".into())
                                })?;
                            field.validate()?;
                            let count = (field.nx as usize - 1)
                                .checked_mul(field.nz as usize - 1)
                                .and_then(|n| n.checked_mul(2))
                                .ok_or_else(|| {
                                    Error::Build("sensor scene: terrain triangle overflow".into())
                                })?;
                            check_budget(vertices, triangles, field.heights.len(), count, options)?;
                            let mut mesh = TriMesh {
                                vertices: Vec::with_capacity(field.heights.len()),
                                indices: Vec::with_capacity(count),
                            };
                            for z in 0..field.nz {
                                for x in 0..field.nx {
                                    mesh.vertices.push(
                                        field.origin
                                            + Vec3::new(
                                                x as Real * field.cell_size,
                                                field.sample(x, z),
                                                z as Real * field.cell_size_z,
                                            ),
                                    );
                                }
                            }
                            for z in 0..field.nz - 1 {
                                for x in 0..field.nx - 1 {
                                    let a = z * field.nx + x;
                                    mesh.indices.push([a, a + field.nx, a + 1]);
                                    mesh.indices.push([a + 1, a + field.nx, a + field.nx + 1]);
                                }
                            }
                            mesh
                        }
                        ShapeGeometry::ConvexHull { hull_id } => {
                            let points = &source.hulls[hull_id as usize];
                            if points.iter().any(|p| !p.is_finite()) {
                                return Err(Error::Build(
                                    "sensor scene: non-finite convex hull".into(),
                                ));
                            }
                            let hull = molla_geometry::convex_hull(points);
                            if hull.faces.is_empty() {
                                if options.unsupported == UnsupportedGeometryPolicy::Reject {
                                    return Err(Error::Build(format!(
                                        "sensor scene: degenerate ConvexHull at shape {shape}"
                                    )));
                                }
                                unsupported.push(UnsupportedShape {
                                    shape_index: shape as u32,
                                    kind: "ConvexHull",
                                });
                                continue;
                            }
                            check_budget(
                                vertices,
                                triangles,
                                hull.vertices.len(),
                                hull.faces.len(),
                                options,
                            )?;
                            TriMesh {
                                vertices: hull.vertices,
                                indices: hull.faces,
                            }
                        }
                        _ => unreachable!(),
                    };
                    vertices += mesh.vertices.len();
                    triangles += mesh.indices.len();
                    let id = u32::try_from(model.meshes.len())
                        .map_err(|_| Error::Build("sensor scene: too many meshes".into()))?;
                    model.meshes.push(mesh);
                    remap.insert(key, id);
                    id
                };
                model.shape_geometry[shape] = ShapeGeometry::TriMesh { mesh_id: id };
            }
        }
        model.shape_body = uploaded(engine, bodies.to_vec())?;
        model.shape_transform = uploaded(engine, transforms.to_vec())?;
        model.shape_geom_kind = uploaded(engine, kinds)?;
        model.shape_geom_data = uploaded(engine, data)?;
        model.shape_color = uploaded(engine, colors.to_vec())?;
        let geometry = SceneGeometry::new(&device, &model, max_query_bytes)?;
        Ok(Self {
            device,
            model,
            geometry,
            max_query_bytes,
            stamp,
            options,
            unsupported,
            identities,
        })
    }

    pub fn stamp(&self) -> SceneStamp {
        self.stamp
    }

    /// Retain a cheap immutable identity snapshot alongside asynchronous results.
    pub fn identities(&self) -> SceneIdentitySnapshot {
        self.identities.clone()
    }
    pub fn mesh_count(&self) -> usize {
        self.model.meshes.len()
    }
    /// Allocated immutable query-atlas payload, including empty buffer padding.
    pub fn query_device_bytes(&self) -> u64 {
        self.geometry.device_bytes
    }
    pub fn unsupported_shapes(&self) -> &[UnsupportedShape] {
        &self.unsupported
    }

    /// Compile a replacement with new handles; failures preserve geometry, revision and identities.
    pub fn replace_model(&mut self, engine: &Engine, source: &Model) -> Result<()> {
        self.replace(engine, source, None)
    }

    /// Preserve explicitly mapped handles; omitted old handles are retired, None entries are new.
    pub fn replace_model_with_remap(
        &mut self,
        engine: &Engine,
        source: &Model,
        remap: SensorSceneRemap<'_>,
    ) -> Result<()> {
        self.replace(engine, source, Some(remap))
    }

    fn replace(
        &mut self,
        engine: &Engine,
        source: &Model,
        remap: Option<SensorSceneRemap<'_>>,
    ) -> Result<()> {
        self.check_engine(engine)?;
        let revision = self
            .stamp
            .geometry_revision
            .checked_add(1)
            .ok_or_else(|| Error::Build("sensor scene revision exhausted".into()))?;
        let stamp = SceneStamp {
            scene_id: self.stamp.scene_id,
            geometry_revision: revision,
        };
        let replacement = Self::compile(
            engine,
            source,
            self.options,
            self.max_query_bytes,
            stamp,
            Some(&self.identities),
            remap,
        )?;
        *self = replacement;
        Ok(())
    }

    pub fn resolve_body(&self, stamp: SceneStamp, shape_index: u32) -> Result<BodyId> {
        if stamp != self.stamp {
            return Err(Error::Solver(
                "sensor result belongs to another scene revision".into(),
            ));
        }
        let body = *self
            .model
            .shape_body
            .host()?
            .get(shape_index as usize)
            .ok_or_else(|| Error::Solver("sensor shape index out of range".into()))?;
        Ok(if body < 0 {
            BodyId::NONE
        } else {
            BodyId::from_index(body as usize)
        })
    }

    pub(crate) fn check_engine(&self, engine: &Engine) -> Result<()> {
        engine.check_device()?;
        if engine
            .device()
            .is_none_or(|d| !Arc::ptr_eq(d, &self.device))
        {
            return Err(Error::Gpu("sensor scene: foreign engine".into()));
        }
        Ok(())
    }
}

fn uploaded<T: molla_storage::BufferElement>(engine: &Engine, values: Vec<T>) -> Result<Buffer<T>> {
    Ok(DeviceBuffer::from_host(engine, values, BufferUsage::Storage)?.buffer)
}

fn check_budget(
    v: usize,
    t: usize,
    more_v: usize,
    more_t: usize,
    options: SensorSceneOptions,
) -> Result<()> {
    if v.checked_add(more_v)
        .is_none_or(|n| n > options.max_vertices || n > u32::MAX as usize)
        || t.checked_add(more_t)
            .is_none_or(|n| n > options.max_triangles || n > i32::MAX as usize / 3)
    {
        return Err(Error::Build("sensor scene geometry budget exceeded".into()));
    }
    Ok(())
}

fn validate_pose(pose: Transform) -> Result<()> {
    if !pose.position.as_vec3().is_finite()
        || !pose.rotation.is_finite()
        || (pose.rotation.length_squared() - 1.0).abs() > 1e-6
    {
        return Err(Error::Build(
            "sensor scene requires finite rigid unit-quaternion transforms".into(),
        ));
    }
    Ok(())
}

fn positive(v: Vec3) -> Result<Vec3> {
    if !v.as_vec3().is_finite() || v.as_vec3().min_element() <= 0.0 {
        return Err(Error::Build(
            "sensor scene requires positive finite f32 dimensions".into(),
        ));
    }
    Ok(v)
}

fn radial(height: Real, radius: Real, zero_height: bool) -> Result<Vec3> {
    positive(Vec3::splat(radius))?;
    if !(height as f32).is_finite() || height < 0.0 || !zero_height && height as f32 <= 0.0 {
        return Err(Error::Build(
            "sensor scene: invalid radial shape height".into(),
        ));
    }
    Ok(Vec3::new(height, radius, 0.0))
}
