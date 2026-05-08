//! World-frame view of a shape: pose + geometry, suitable for the
//! collision pipeline.

use crate::math::{Transform, Vec3};
use crate::sim::{Model, ShapeGeometry, State};

use super::aabb::Aabb;

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
    let body_com = model.body_com.host().expect("body_com");
    let bi = body_idx as usize;
    // body_q stores the body-frame origin in world (NOT the COM). The
    // shape transform is relative to the body's local frame, so we just
    // compose: world = body_q * shape_local.
    let _ = body_com[bi]; // currently unused; flagged for stage-3 articulation
    body_q[bi].compose(&shape_local)
}

/// Compute a world-frame AABB for `shape_idx`. Used by broad-phase
/// pair generation. `margin` adds slack to handle CCD-ish overshoot.
pub fn shape_world_aabb(
    model: &Model,
    state: &State,
    shape_idx: usize,
    margin: crate::math::Real,
) -> Aabb {
    let geom = model.shape_geometry[shape_idx];
    let xf = shape_world_transform(model, state, shape_idx);
    match geom {
        ShapeGeometry::Sphere { radius } => {
            // Sphere AABB is rotation-invariant.
            Aabb::centred(xf.position, Vec3::splat(radius)).expanded(margin)
        }
        ShapeGeometry::Plane { .. } => Aabb::infinite(),
        ShapeGeometry::Box { half_extents } => {
            // World-axis half-extents = |R| · h_local, where |R|
            // is the elementwise absolute of the rotation matrix.
            let rot = crate::math::Mat3::from_quat(xf.rotation);
            let abs_x = Vec3::new(rot.x_axis.x.abs(), rot.x_axis.y.abs(), rot.x_axis.z.abs());
            let abs_y = Vec3::new(rot.y_axis.x.abs(), rot.y_axis.y.abs(), rot.y_axis.z.abs());
            let abs_z = Vec3::new(rot.z_axis.x.abs(), rot.z_axis.y.abs(), rot.z_axis.z.abs());
            let h: Vec3 = abs_x * half_extents.x + abs_y * half_extents.y + abs_z * half_extents.z;
            Aabb::centred(xf.position, h).expanded(margin)
        }
        ShapeGeometry::Capsule { half_height, radius } => {
            // Capsule AABB = AABB of two end-cap spheres.
            let axis_world = xf.rotation * Vec3::Y;
            let p0 = xf.position - axis_world * half_height;
            let p1 = xf.position + axis_world * half_height;
            let lo = p0.min(p1) - Vec3::splat(radius);
            let hi = p0.max(p1) + Vec3::splat(radius);
            Aabb { min: lo, max: hi }.expanded(margin)
        }
        ShapeGeometry::Cylinder { half_height, radius } => {
            // Conservative AABB: take the cylinder's bounding sphere
            // along the rotated axis, then add radius slack on every
            // axis. Tight bound: project the disk axis onto each
            // world axis and combine with `half_height`.
            let r = crate::math::Mat3::from_quat(xf.rotation);
            let axis_world = r.y_axis; // local +Y
            // Half-extents along world X/Y/Z:
            //   along axis direction: half_height
            //   perpendicular: radius
            // For a generic rotation: extent along world axis e_i is
            //   |e_i · axis_world|·half_height + sqrt(1-(e_i·axis_world)²)·radius
            let absax = Vec3::new(
                axis_world.x.abs(),
                axis_world.y.abs(),
                axis_world.z.abs(),
            );
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
            let mut lo = Vec3::splat(crate::math::Real::INFINITY);
            let mut hi = Vec3::splat(-crate::math::Real::INFINITY);
            for &v in &mesh.vertices {
                let w = xf.transform_point(v);
                lo = lo.min(w);
                hi = hi.max(w);
            }
            Aabb { min: lo, max: hi }.expanded(margin)
        }
    }
}


