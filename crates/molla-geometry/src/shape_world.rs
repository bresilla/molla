//! World-frame view of a shape: pose + geometry, suitable for the
//! collision pipeline.

use molla_math::{Mat3, Real, Transform, Vec3};
use molla_sim::runtime::{RigidScene, SceneStamp};
use molla_sim::{Model, SdfShape, ShapeGeometry, State};
use std::borrow::Cow;

use crate::aabb::Aabb;

#[path = "heightfield_bounds.rs"]
mod heightfield_bounds;
use heightfield_bounds::HeightfieldTiles;

#[cfg(test)]
#[path = "scene_bounds_tests.rs"]
mod scene_bounds_tests;

/// Heightfield, mesh and hull bounds and tile maxima retained until the
/// compiled model changes.
#[derive(Default)]
pub struct SceneShapeBounds {
    stamp: Option<SceneStamp>,
    heightfields: Vec<(Vec3, Vec3)>,
    tiles: Vec<Option<HeightfieldTiles>>,
    meshes: Vec<(Vec3, Vec3)>,
    hulls: Vec<(Vec3, Vec3)>,
}

impl SceneShapeBounds {
    pub fn prepare<'a>(&'a mut self, scene: &'a RigidScene) -> PreparedShapeBounds<'a> {
        let stamp = scene.model_stamp();
        if self.stamp.as_ref() != Some(&stamp) {
            let model = scene.model();
            self.heightfields = model
                .heightfields
                .iter()
                .map(|field| field.local_aabb())
                .collect();
            self.stamp = Some(stamp);
            self.tiles = model
                .heightfields
                .iter()
                .map(HeightfieldTiles::new)
                .collect();
            self.meshes = model
                .meshes
                .iter()
                .map(|mesh| local_bounds(&mesh.vertices))
                .collect();
            self.hulls = model.hulls.iter().map(|hull| local_bounds(hull)).collect();
        }
        PreparedShapeBounds {
            model: scene.model(),
            heightfields: Cow::Borrowed(&self.heightfields),
            tiles: Cow::Borrowed(&self.tiles),
            meshes: Cow::Borrowed(&self.meshes),
            hulls: Cow::Borrowed(&self.hulls),
            stamp: self.stamp.clone(),
        }
    }
}

/// Heightfield, mesh and hull bounds reusable while the model remains
/// immutably borrowed.
pub struct PreparedShapeBounds<'a> {
    model: &'a Model,
    heightfields: Cow<'a, [(Vec3, Vec3)]>,
    tiles: Cow<'a, [Option<HeightfieldTiles>]>,
    meshes: Cow<'a, [(Vec3, Vec3)]>,
    hulls: Cow<'a, [(Vec3, Vec3)]>,
    /// The compiled model of the scene they were prepared for, when they came from one.
    stamp: Option<SceneStamp>,
}

impl<'a> PreparedShapeBounds<'a> {
    pub fn new(model: &'a Model) -> Self {
        Self {
            model,
            heightfields: Cow::Owned(
                model
                    .heightfields
                    .iter()
                    .map(|field| field.local_aabb())
                    .collect(),
            ),
            tiles: Cow::Owned(
                model
                    .heightfields
                    .iter()
                    .map(HeightfieldTiles::new)
                    .collect(),
            ),
            meshes: Cow::Owned(
                model
                    .meshes
                    .iter()
                    .map(|mesh| local_bounds(&mesh.vertices))
                    .collect(),
            ),
            hulls: Cow::Owned(model.hulls.iter().map(|hull| local_bounds(hull)).collect()),
            stamp: None,
        }
    }

    pub fn model(&self) -> &'a Model {
        self.model
    }

    pub fn stamp(&self) -> Option<&SceneStamp> {
        self.stamp.as_ref()
    }

    pub(crate) fn heightfield_may_contact(
        &self,
        state: &State,
        field_shape: usize,
        other: Aabb,
    ) -> bool {
        let ShapeGeometry::Heightfield { field_id } = self.model.shape_geometry[field_shape] else {
            return true;
        };
        let Some(Some(tiles)) = self.tiles.get(field_id as usize) else {
            return true;
        };
        let Some(field) = self.model.heightfields.get(field_id as usize) else {
            return true;
        };
        if !other.min.is_finite() || !other.max.is_finite() {
            return true;
        }
        let pose = shape_world_transform(self.model, state, field_shape).inverse();
        let padding = 1e-9
            * (1.0
                + other
                    .min
                    .abs()
                    .max_element()
                    .max(other.max.abs().max_element())
                    .max(pose.position.abs().max_element()));
        let local = transformed_bounds(other.min, other.max, pose, padding);
        tiles.may_contact(field, local)
    }

    /// A shape's world bounds. Heightfields, meshes and hulls take their
    /// cached local box through the pose, so a mesh's vertices are not
    /// walked on every query; the box is loose by the turn of the pose.
    pub fn world_aabb(&self, state: &State, shape: usize, margin: Real) -> Aabb {
        let pose = shape_world_transform(self.model, state, shape);
        let local = match self.model.shape_geometry[shape] {
            ShapeGeometry::Heightfield { field_id } => self.heightfields.get(field_id as usize),
            ShapeGeometry::TriMesh { mesh_id } => self.meshes.get(mesh_id as usize),
            ShapeGeometry::ConvexHull { hull_id } => self.hulls.get(hull_id as usize),
            _ => return shape_aabb_at_pose(self.model, shape, pose, margin),
        };
        match local {
            Some(&(lo, hi)) if lo.cmple(hi).all() => transformed_bounds(lo, hi, pose, margin),
            _ => Aabb::infinite(),
        }
    }
}

/// The local box round a set of points; inverted when there are none.
fn local_bounds(points: &[Vec3]) -> (Vec3, Vec3) {
    points.iter().fold(
        (Vec3::splat(Real::INFINITY), Vec3::splat(-Real::INFINITY)),
        |(lo, hi), &p| (lo.min(p), hi.max(p)),
    )
}

fn transformed_bounds(lo: Vec3, hi: Vec3, pose: Transform, margin: Real) -> Aabb {
    let corners = [
        Vec3::new(lo.x, lo.y, lo.z),
        Vec3::new(hi.x, lo.y, lo.z),
        Vec3::new(lo.x, hi.y, lo.z),
        Vec3::new(hi.x, hi.y, lo.z),
        Vec3::new(lo.x, lo.y, hi.z),
        Vec3::new(hi.x, lo.y, hi.z),
        Vec3::new(lo.x, hi.y, hi.z),
        Vec3::new(hi.x, hi.y, hi.z),
    ];
    let mut min = Vec3::splat(Real::INFINITY);
    let mut max = Vec3::splat(-Real::INFINITY);
    for corner in corners {
        let point = pose.transform_point(corner);
        min = min.min(point);
        max = max.max(point);
    }
    Aabb { min, max }.expanded(margin)
}

/// Compute the world-frame transform of `shape_idx`. If the shape is
/// attached to a body, this is `body_q * shape_local_transform`. If
/// it's static (`shape_body == -1`), it's just the local transform.
pub fn shape_world_transform(model: &Model, state: &State, shape_idx: usize) -> Transform {
    let shape_local = model.shape_transform.host().expect("shape_transform")[shape_idx];
    let body_idx = model.shape_body.host().expect("shape_body")[shape_idx];
    if body_idx < 0 {
        return shape_local;
    }
    let body_q = state.body_q.host().expect("body_q");
    let bi = body_idx as usize;
    body_q[bi].compose(&shape_local)
}

/// Compute a world-frame AABB for `shape_idx`. Used by broad-phase
/// pair generation. `margin` adds slack to handle CCD-ish overshoot.
pub fn shape_world_aabb(model: &Model, state: &State, shape_idx: usize, margin: Real) -> Aabb {
    let xf = shape_world_transform(model, state, shape_idx);
    shape_aabb_at_pose(model, shape_idx, xf, margin)
}

/// Bounds of a canonical shape at an arbitrary pose, without changing simulation state.
pub fn shape_aabb_at_pose(model: &Model, shape_idx: usize, xf: Transform, margin: Real) -> Aabb {
    let geom = model.shape_geometry[shape_idx];
    match geom {
        ShapeGeometry::Sphere { radius } => {
            // Sphere AABB is rotation-invariant.
            Aabb::centred(xf.position, Vec3::splat(radius)).expanded(margin)
        }
        ShapeGeometry::Plane { .. } => Aabb::infinite(),
        ShapeGeometry::Box { half_extents } => {
            // World-axis half-extents = |R| · h_local, where |R|
            // is the elementwise absolute of the rotation matrix.
            let rot = Mat3::from_quat(xf.rotation);
            let abs_x = Vec3::new(rot.x_axis.x.abs(), rot.x_axis.y.abs(), rot.x_axis.z.abs());
            let abs_y = Vec3::new(rot.y_axis.x.abs(), rot.y_axis.y.abs(), rot.y_axis.z.abs());
            let abs_z = Vec3::new(rot.z_axis.x.abs(), rot.z_axis.y.abs(), rot.z_axis.z.abs());
            let h: Vec3 = abs_x * half_extents.x + abs_y * half_extents.y + abs_z * half_extents.z;
            Aabb::centred(xf.position, h).expanded(margin)
        }
        ShapeGeometry::Capsule {
            half_height,
            radius,
        } => {
            // Capsule AABB = AABB of two end-cap spheres.
            let axis_world = xf.rotation * Vec3::Y;
            let p0 = xf.position - axis_world * half_height;
            let p1 = xf.position + axis_world * half_height;
            let lo = p0.min(p1) - Vec3::splat(radius);
            let hi = p0.max(p1) + Vec3::splat(radius);
            Aabb { min: lo, max: hi }.expanded(margin)
        }
        ShapeGeometry::Cylinder {
            half_height,
            radius,
        }
        | ShapeGeometry::RoundCylinder {
            half_height,
            radius,
            ..
        } => {
            // Conservative AABB: take the cylinder's bounding sphere
            // along the rotated axis, then add radius slack on every
            // axis. Tight bound: project the disk axis onto each
            // world axis and combine with `half_height`.
            let r = Mat3::from_quat(xf.rotation);
            let axis_world = r.y_axis; // local +Y
            // Half-extents along world X/Y/Z:
            //   along axis direction: half_height
            //   perpendicular: radius
            // For a generic rotation: extent along world axis e_i is
            //   |e_i · axis_world|·half_height + sqrt(1-(e_i·axis_world)²)·radius
            let absax = Vec3::new(axis_world.x.abs(), axis_world.y.abs(), axis_world.z.abs());
            let one = Vec3::splat(1.0);
            let perp_scale = (one - absax * absax).max(Vec3::ZERO);
            let perp = Vec3::new(
                perp_scale.x.sqrt(),
                perp_scale.y.sqrt(),
                perp_scale.z.sqrt(),
            );
            let border = match geom {
                ShapeGeometry::RoundCylinder { border_radius, .. } => border_radius,
                _ => 0.0,
            };
            let h = absax * half_height + perp * radius + Vec3::splat(border);
            Aabb::centred(xf.position, h).expanded(margin)
        }
        ShapeGeometry::Triangle { v0, v1, v2 } => {
            // Triangle vertices are in shape-local frame; transform
            // each to world.
            let w0 = xf.transform_point(v0);
            let w1 = xf.transform_point(v1);
            let w2 = xf.transform_point(v2);
            let lo = w0.min(w1).min(w2);
            let hi = w0.max(w1).max(w2);
            Aabb { min: lo, max: hi }.expanded(margin)
        }
        ShapeGeometry::TriMesh { mesh_id } => {
            // Loose bound: AABB of all mesh vertices in world frame.
            let mesh = match model.meshes.get(mesh_id as usize) {
                Some(m) => m,
                None => return Aabb::infinite(),
            };
            if mesh.vertices.is_empty() {
                return Aabb::infinite();
            }
            let mut lo = Vec3::splat(Real::INFINITY);
            let mut hi = Vec3::splat(-Real::INFINITY);
            for &v in &mesh.vertices {
                let w = xf.transform_point(v);
                lo = lo.min(w);
                hi = hi.max(w);
            }
            Aabb { min: lo, max: hi }.expanded(margin)
        }
        ShapeGeometry::Cone {
            half_height,
            radius,
        } => {
            // Same trick as Cylinder: project the local axis onto each
            // world axis and split between height and radius extents.
            let r = Mat3::from_quat(xf.rotation);
            let axis_world = r.y_axis;
            let absax = Vec3::new(axis_world.x.abs(), axis_world.y.abs(), axis_world.z.abs());
            let one = Vec3::splat(1.0);
            let perp_scale = (one - absax * absax).max(Vec3::ZERO);
            let perp = Vec3::new(
                perp_scale.x.sqrt(),
                perp_scale.y.sqrt(),
                perp_scale.z.sqrt(),
            );
            let h = absax * half_height + perp * radius;
            Aabb::centred(xf.position, h).expanded(margin)
        }
        ShapeGeometry::ConvexHull { hull_id } => {
            let hull = match model.hulls.get(hull_id as usize) {
                Some(h) => h,
                None => return Aabb::infinite(),
            };
            if hull.is_empty() {
                return Aabb::infinite();
            }
            let mut lo = Vec3::splat(Real::INFINITY);
            let mut hi = Vec3::splat(-Real::INFINITY);
            for &v in hull {
                let w = xf.transform_point(v);
                lo = lo.min(w);
                hi = hi.max(w);
            }
            Aabb { min: lo, max: hi }.expanded(margin)
        }
        ShapeGeometry::Heightfield { field_id } => {
            let field = match model.heightfields.get(field_id as usize) {
                Some(f) => f,
                None => return Aabb::infinite(),
            };
            let (lo_local, hi_local) = field.local_aabb();
            transformed_bounds(lo_local, hi_local, xf, margin)
        }
        ShapeGeometry::Ellipsoid { half_axes } => {
            // World-axis half-extents of an oriented ellipsoid:
            //   h_world.i = sqrt(Σ_j (R[i, j] · half_axes.j)²).
            // Equivalent to projecting the rotated semi-axes onto each
            // world axis and combining via Pythagoras (the surface's
            // support point along eᵢ has coordinate equal to that
            // sqrt). Reduces to the sphere case when all axes match.
            let r = Mat3::from_quat(xf.rotation);
            let row_x = Vec3::new(r.x_axis.x, r.y_axis.x, r.z_axis.x);
            let row_y = Vec3::new(r.x_axis.y, r.y_axis.y, r.z_axis.y);
            let row_z = Vec3::new(r.x_axis.z, r.y_axis.z, r.z_axis.z);
            let h_sq = Vec3::new(
                (row_x * half_axes).length_squared(),
                (row_y * half_axes).length_squared(),
                (row_z * half_axes).length_squared(),
            );
            let h = Vec3::new(h_sq.x.sqrt(), h_sq.y.sqrt(), h_sq.z.sqrt());
            Aabb::centred(xf.position, h).expanded(margin)
        }
        ShapeGeometry::Sdf { sdf_id } => {
            // Conservative AABB depending on the SDF type.
            let sdf = match model.sdfs.get(sdf_id as usize) {
                Some(s) => *s,
                None => return Aabb::infinite(),
            };
            let h_local = match sdf {
                SdfShape::Torus { major, minor } => Vec3::new(major + minor, minor, major + minor),
                SdfShape::HelicalThread {
                    major,
                    minor,
                    half_height,
                    ..
                } => Vec3::new(major + minor, half_height + minor, major + minor),
                SdfShape::Sphere { r } => Vec3::splat(r),
            };
            // Rotate the AABB into world (uses the box rotation trick).
            let r = Mat3::from_quat(xf.rotation);
            let abs_x = Vec3::new(r.x_axis.x.abs(), r.x_axis.y.abs(), r.x_axis.z.abs());
            let abs_y = Vec3::new(r.y_axis.x.abs(), r.y_axis.y.abs(), r.y_axis.z.abs());
            let abs_z = Vec3::new(r.z_axis.x.abs(), r.z_axis.y.abs(), r.z_axis.z.abs());
            let h = abs_x * h_local.x + abs_y * h_local.y + abs_z * h_local.z;
            Aabb::centred(xf.position, h).expanded(margin)
        }
    }
}
