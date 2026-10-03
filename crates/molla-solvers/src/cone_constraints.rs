//! Runtime cone frames and angular-impulse readout.

use molla_core::Result;
use molla_math::{Quat, Real, Vec3};
use molla_sim::{
    State,
    runtime::{JointDesc, RigidScene, SphericalConeLimits},
    spherical_cone::{self, ConeGeometry},
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ConeReading {
    pub swing: [Real; 2],
    pub violation: Real,
    pub active: bool,
    /// Frame-average cone torque on the child in world coordinates.
    pub torque: Vec3,
    pub sample_dt: Real,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BallAnchorReading {
    pub separation: Real,
    /// Frame-average force at the child anchor, in world coordinates and N.
    pub force: Vec3,
    pub sample_dt: Real,
}

pub(crate) fn geometry(
    scene: &RigidScene,
    state: &State,
    joint: &JointDesc,
    limits: SphericalConeLimits,
) -> Result<(Quat, ConeGeometry)> {
    let poses = state.body_q.host()?;
    let parent = joint.parent.map_or(joint.frame_parent.rotation, |p| {
        poses[scene.body_index(p).unwrap().index()].rotation * joint.frame_parent.rotation
    });
    let child =
        poses[scene.body_index(joint.child).unwrap().index()].rotation * joint.frame_child.rotation;
    Ok((
        parent,
        spherical_cone::evaluate(limits, joint.axis, parent.conjugate() * child)?,
    ))
}
