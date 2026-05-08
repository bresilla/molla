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
    }
}

