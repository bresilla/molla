//! Transactional rigid-scene mutation with persistent handles and dense solver snapshots.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::joint_control::{JointAxes, JointAxis, JointAxisSettings, JointMotor};
use molla_core::{BodyId, Error, JointId, Result, ShapeId, WorldId};
use molla_math::{Quat, Real, SpatialVector, Transform, Vec3};

use crate::{
    BodyFlags, BodyParams, CoefficientCombineRule, ColliderMass, ContactMaterial, Control,
    Heightfield, JointType, MassProperties, Model, ModelBuilder, ShapeFlags, State, TriMesh,
};

macro_rules! handle {
    ($name:ident) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
        pub struct $name(u64);

        impl $name {
            pub fn to_bits(self) -> u64 {
                self.0
            }
        }
    };
}

handle!(BodyHandle);
handle!(ColliderHandle);
handle!(JointHandle);

/// Compiled scene identity; simulation steps and motor updates retain it.
#[derive(Clone, Debug)]
pub struct SceneStamp(Arc<()>);

impl PartialEq for SceneStamp {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for SceneStamp {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BodyKind {
    Dynamic,
    Kinematic,
    Fixed,
}

#[derive(Clone, Debug)]
pub struct BodyDesc {
    pub params: BodyParams,
    pub kind: BodyKind,
    pub enabled: bool,
    pub ccd_enabled: bool,
    pub velocity: SpatialVector,
    pub linear_damping: Real,
    pub angular_damping: Real,
    pub additional_mass: Option<MassProperties>,
    pub user_tag: u64,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            params: BodyParams::new(),
            kind: BodyKind::Dynamic,
            enabled: true,
            ccd_enabled: false,
            velocity: SpatialVector::ZERO,
            linear_damping: 0.0,
            angular_damping: 0.0,
            additional_mass: None,
            user_tag: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ColliderGeometry {
    Ball {
        radius: Real,
    },
    Cuboid {
        half_extents: Vec3,
    },
    Capsule {
        half_height: Real,
        radius: Real,
    },
    CapsuleEndpoints {
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
    ConvexHull(Vec<Vec3>),
    /// Convex pieces in one shared collider frame, prepared by a mesh decomposer.
    ConvexParts(Vec<Vec<Vec3>>),
    TriMesh(TriMesh),
    Heightfield(Heightfield),
}

impl ColliderGeometry {
    /// Transform from the canonical primitive frame into the collider frame.
    pub fn geometry_transform(&self) -> Result<Transform> {
        let Self::CapsuleEndpoints { a, b, .. } = self else {
            return Ok(Transform::IDENTITY);
        };
        let delta = *b - *a;
        let length = delta.length();
        if !a.is_finite() || !b.is_finite() || !length.is_finite() {
            return Err(Error::Build("invalid capsule endpoints".into()));
        }
        let rotation = if length > 0.0 {
            Quat::from_rotation_arc(Vec3::Y, delta / length)
        } else {
            Quat::IDENTITY
        };
        Ok(Transform::new(*a * 0.5 + *b * 0.5, rotation))
    }
}

/// Validate a finite three-dimensional support-point cloud.
/// Coplanarity tolerance is `1e-12` of the largest point-cloud extent.
pub fn validate_convex_points(points: &[Vec3]) -> Result<()> {
    let invalid = || Error::Build("convex collider requires a finite solid point cloud".into());
    if points.len() < 4 || points.iter().any(|p| !p.is_finite()) {
        return Err(invalid());
    }
    let lo = points
        .iter()
        .copied()
        .fold(Vec3::splat(Real::INFINITY), Vec3::min);
    let hi = points
        .iter()
        .copied()
        .fold(Vec3::splat(Real::NEG_INFINITY), Vec3::max);
    let scale = (hi - lo).max_element();
    if !scale.is_finite() || scale <= 0.0 {
        return Err(invalid());
    }
    let offset = |p: &Vec3| (*p - points[0]) / scale;
    let edge = points
        .iter()
        .map(offset)
        .max_by(|a, b| a.length_squared().total_cmp(&b.length_squared()))
        .ok_or_else(invalid)?;
    let normal = points
        .iter()
        .map(|p| edge.cross(offset(p)))
        .max_by(|a, b| a.length_squared().total_cmp(&b.length_squared()))
        .ok_or_else(invalid)?;
    if normal.length_squared() == 0.0
        || !points
            .iter()
            .any(|p| normal.dot(offset(p)).abs() > normal.length() * 1e-12)
    {
        return Err(invalid());
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct ColliderDesc {
    pub parent: Option<BodyHandle>,
    pub geometry: ColliderGeometry,
    /// Authored density or total collider mass; body totals change on explicit recomputation.
    pub mass: ColliderMass,
    pub pose: Transform,
    pub friction: Real,
    pub restitution: Real,
    pub contact_material: ContactMaterial,
    pub memberships: u32,
    pub filter: u32,
    pub enabled: bool,
    pub sensor: bool,
    pub user_tag: u64,
}

impl ColliderDesc {
    pub fn new(geometry: ColliderGeometry) -> Self {
        Self {
            parent: None,
            geometry,
            mass: ColliderMass::default(),
            pose: Transform::IDENTITY,
            friction: 0.5,
            restitution: 0.0,
            contact_material: ContactMaterial {
                friction_combine: CoefficientCombineRule::Average,
                ..ContactMaterial::default()
            },
            memberships: u32::MAX,
            filter: u32::MAX,
            enabled: true,
            sensor: false,
            user_tag: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct JointDesc {
    pub parent: Option<BodyHandle>,
    pub child: BodyHandle,
    pub kind: JointType,
    pub frame_parent: Transform,
    pub frame_child: Transform,
    pub axis: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct JointSettings {
    pub enabled: bool,
    pub contacts_enabled: bool,
}

impl Default for JointSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            contacts_enabled: false,
        }
    }
}

#[derive(Clone, Default)]
struct Description {
    next_id: u64,
    bodies: BTreeMap<BodyHandle, BodyDesc>,
    colliders: BTreeMap<ColliderHandle, ColliderDesc>,
    collider_stamps: BTreeMap<ColliderHandle, SceneStamp>,
    joints: BTreeMap<JointHandle, JointDesc>,
    joint_settings: BTreeMap<JointHandle, JointSettings>,
    joint_axes: BTreeMap<JointHandle, [JointAxisSettings; 6]>,
}

impl Description {
    fn allocate(&mut self) -> Result<u64> {
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| Error::Build("runtime handle space exhausted".into()))?;
        Ok(self.next_id)
    }
}

struct Snapshot {
    stamp: SceneStamp,
    model: Model,
    state: State,
    control: Control,
    bodies: BTreeMap<BodyHandle, BodyId>,
    colliders: BTreeMap<ColliderHandle, Vec<ShapeId>>,
    shape_colliders: Vec<ColliderHandle>,
    joints: BTreeMap<JointHandle, JointId>,
    roots: BTreeMap<BodyHandle, JointId>,
    drives: BTreeMap<DriveOwner, DriveState>,
    valid_poses: BTreeMap<BodyHandle, Transform>,
}

#[derive(Clone, Copy, Eq, PartialEq, Ord, PartialOrd)]
enum DriveOwner {
    Body(BodyHandle),
    Joint(JointHandle),
}

#[derive(Clone)]
struct DriveState {
    mode: Vec<u32>,
    position: Vec<Real>,
    velocity: Vec<Real>,
    kp: Vec<Real>,
    kd: Vec<Real>,
    force: Vec<Real>,
    motor_model: Vec<u32>,
    max_force: Vec<Real>,
}

impl DriveState {
    fn capture(snapshot: &Snapshot, joint: JointId) -> Result<Self> {
        let offsets = snapshot.model.joint_dof_offset.host()?;
        let range = offsets[joint.index()] as usize..offsets[joint.index() + 1] as usize;
        let clean = |values: &[Real]| {
            values[range.clone()]
                .iter()
                .map(|&v| if v.is_finite() { v } else { 0.0 })
                .collect()
        };
        Ok(Self {
            mode: snapshot.control.joint_target_mode.host()?[range.clone()].to_vec(),
            position: clean(snapshot.control.joint_target_pos.host()?),
            velocity: clean(snapshot.control.joint_target_vel.host()?),
            kp: clean(snapshot.control.joint_target_kp.host()?),
            kd: clean(snapshot.control.joint_target_kd.host()?),
            force: clean(snapshot.control.joint_force.host()?),
            motor_model: snapshot.control.joint_motor_model.host()?[range.clone()]
                .iter()
                .map(|&m| m.min(1))
                .collect(),
            max_force: snapshot.control.joint_motor_max_force.host()?[range]
                .iter()
                .map(|&v| if v.is_nan() || v < 0.0 { 0.0 } else { v })
                .collect(),
        })
    }

    fn restore(&self, model: &Model, control: &mut Control, joint: JointId) -> Result<()> {
        let offsets = model.joint_dof_offset.host()?;
        let range = offsets[joint.index()] as usize..offsets[joint.index() + 1] as usize;
        if range.len() != self.mode.len() {
            return Ok(());
        }
        control.joint_target_mode.host_mut()?[range.clone()].copy_from_slice(&self.mode);
        control.joint_motor_model.host_mut()?[range.clone()].copy_from_slice(&self.motor_model);
        for (destination, source) in [
            (&mut control.joint_target_pos, &self.position),
            (&mut control.joint_target_vel, &self.velocity),
            (&mut control.joint_target_kp, &self.kp),
            (&mut control.joint_target_kd, &self.kd),
            (&mut control.joint_force, &self.force),
            (&mut control.joint_motor_max_force, &self.max_force),
        ] {
            destination.host_mut()?[range.clone()].copy_from_slice(source);
        }
        Ok(())
    }
}

/// CPU-owned rigid scene. Successful edits invalidate dense indices, not handles.
pub struct RigidScene {
    description: Description,
    snapshot: Snapshot,
    gravity: Vec3,
    revision: u64,
}

impl RigidScene {
    pub fn new() -> Result<Self> {
        let description = Description::default();
        let gravity = Vec3::new(0.0, -9.81, 0.0);
        let snapshot = compile(&description, gravity, None)?;
        Ok(Self {
            description,
            snapshot,
            gravity,
            revision: 0,
        })
    }

    pub fn model(&self) -> &Model {
        &self.snapshot.model
    }
    pub fn state(&self) -> &State {
        &self.snapshot.state
    }
    pub fn state_mut(&mut self) -> &mut State {
        &mut self.snapshot.state
    }
    pub fn control(&self) -> &Control {
        &self.snapshot.control
    }
    pub fn control_mut(&mut self) -> &mut Control {
        &mut self.snapshot.control
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn stamp(&self) -> SceneStamp {
        self.snapshot.stamp.clone()
    }
    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    pub fn body_index(&self, id: BodyHandle) -> Option<BodyId> {
        self.snapshot.bodies.get(&id).copied()
    }

    pub fn body(&self, id: BodyHandle) -> Option<&BodyDesc> {
        self.description.bodies.get(&id)
    }

    pub fn set_body_enabled(&mut self, id: BodyHandle, enabled: bool) -> Result<()> {
        self.edit(|d| {
            d.bodies
                .get_mut(&id)
                .ok_or_else(|| missing("body"))?
                .enabled = enabled;
            Ok(())
        })
    }

    /// Store the CCD request; CPU penalty-contact stepping currently uses discrete collision.
    pub fn set_body_ccd(&mut self, id: BodyHandle, enabled: bool) -> Result<()> {
        self.description
            .bodies
            .get_mut(&id)
            .ok_or_else(|| missing("body"))?
            .ccd_enabled = enabled;
        Ok(())
    }

    /// Disable bodies with invalid kinematics, forces or incident generalized state/control.
    pub fn quarantine_non_finite(&mut self) -> Result<Vec<BodyHandle>> {
        let mut invalid = BTreeSet::new();
        let state = self.state();
        for (&id, index) in &self.snapshot.bodies {
            let i = index.index();
            if validate_pose(state.body_q.host()?[i]).is_err()
                || !finite_spatial(state.body_qd.host()?[i])
                || !finite_spatial(state.body_f.host()?[i])
                || !finite_spatial(self.control().body_external_f.host()?[i])
            {
                invalid.insert(id);
            }
        }
        let model = self.model();
        let coordinates = model.joint_coord_offset.host()?;
        let dofs = model.joint_dof_offset.host()?;
        for joint in 0..model.joint_count {
            let q = coordinates[joint] as usize..coordinates[joint + 1] as usize;
            let v = dofs[joint] as usize..dofs[joint + 1] as usize;
            let values = &state.joint_q.host()?[q];
            let rotation = match model.joint_type.host()?[joint] {
                x if x == JointType::Ball as u32 => Some(&values[..4]),
                x if x == JointType::Free as u32 || x == JointType::D6 as u32 => {
                    Some(&values[3..7])
                }
                _ => None,
            };
            let bad_coordinates = values.iter().any(|v| !v.is_finite())
                || rotation.is_some_and(|q| {
                    let norm: Real = q.iter().map(|v| v * v).sum();
                    !norm.is_finite() || norm <= 1e-24
                });
            let bad_rates = [
                state.joint_qd.host()?,
                self.control().joint_target_pos.host()?,
                self.control().joint_target_vel.host()?,
                self.control().joint_target_kp.host()?,
                self.control().joint_target_kd.host()?,
                self.control().joint_force.host()?,
            ]
            .iter()
            .any(|values| values[v.clone()].iter().any(|v| !v.is_finite()));
            let bad_motor = self.control().joint_motor_model.host()?[v.clone()]
                .iter()
                .any(|&m| m > 1)
                || self.control().joint_motor_max_force.host()?[v]
                    .iter()
                    .any(|v| v.is_nan() || *v < 0.0);
            if bad_coordinates || bad_rates || bad_motor {
                let body = BodyId::from_index(model.joint_child.host()?[joint] as usize);
                if let Some(id) = self.body_handle(body) {
                    invalid.insert(id);
                }
            }
        }
        if invalid.is_empty() {
            return Ok(Vec::new());
        }
        self.edit(|d| {
            for id in &invalid {
                d.bodies.get_mut(id).ok_or_else(|| missing("body"))?.enabled = false;
            }
            Ok(())
        })?;
        Ok(invalid.into_iter().collect())
    }

    pub fn body_pose(&self, id: BodyHandle) -> Result<Transform> {
        let i = self.body_index(id).ok_or_else(|| missing("body"))?.index();
        Ok(self.state().body_q.host()?[i])
    }

    pub fn body_velocity(&self, id: BodyHandle) -> Result<SpatialVector> {
        let i = self.body_index(id).ok_or_else(|| missing("body"))?.index();
        Ok(self.state().body_qd.host()?[i])
    }

    /// Set COM velocity projected onto the body's joint freedoms; descendants follow.
    pub fn set_body_velocity(&mut self, id: BodyHandle, velocity: SpatialVector) -> Result<()> {
        if !velocity.linear.is_finite() || !velocity.angular.is_finite() {
            return Err(Error::Build("non-finite body velocity".into()));
        }
        let kind = self.body(id).ok_or_else(|| missing("body"))?.kind;
        if kind == BodyKind::Fixed {
            return Ok(());
        }
        let i = self.body_index(id).ok_or_else(|| missing("body"))?.index();
        let mut next = self.state().clone_host()?;
        next.body_qd.host_mut()?[i] = velocity;
        if kind == BodyKind::Dynamic {
            let joint = self
                .model()
                .joint_child
                .host()?
                .iter()
                .position(|&child| child as usize == i)
                .ok_or_else(|| missing("body joint"))?;
            let coordinates = next.joint_q.host()?.to_vec();
            seed_joint(self.model(), &mut next, JointId::from_index(joint))?;
            next.joint_q.host_mut()?.copy_from_slice(&coordinates);
        }
        crate::eval_body_velocities(self.model(), &mut next)?;
        self.commit_body_state(next)
    }

    /// Teleport a body's subtree, projecting its pose onto any incoming joint freedoms.
    pub fn set_body_pose(&mut self, id: BodyHandle, pose: Transform) -> Result<()> {
        validate_pose(pose)?;
        let kind = self.body(id).ok_or_else(|| missing("body"))?.kind;
        let delta = pose.compose(&self.body_pose(id)?.inverse());
        let mut moved = BTreeSet::from([id]);
        let mut pending = vec![id];
        while let Some(parent) = pending.pop() {
            for (_, joint) in self
                .joints()
                .filter(|(id, j)| j.parent == Some(parent) && self.snapshot.joints.contains_key(id))
            {
                if moved.insert(joint.child) {
                    pending.push(joint.child);
                }
            }
        }
        let mut next = self.state().clone_host()?;
        for body in moved {
            let i = self
                .body_index(body)
                .ok_or_else(|| missing("body"))?
                .index();
            next.body_q.host_mut()?[i] = delta.compose(&next.body_q.host()?[i]);
        }
        if kind == BodyKind::Dynamic {
            let i = self.body_index(id).ok_or_else(|| missing("body"))?.index();
            let joint = self
                .model()
                .joint_child
                .host()?
                .iter()
                .position(|&child| child as usize == i)
                .ok_or_else(|| missing("body joint"))?;
            seed_joint(self.model(), &mut next, JointId::from_index(joint))?;
        }
        crate::eval_fk(self.model(), &mut next)?;
        crate::eval_body_velocities(self.model(), &mut next)?;
        self.commit_body_state(next)
    }

    fn commit_body_state(&mut self, next: State) -> Result<()> {
        self.commit_step(next)?;
        self.snapshot.stamp = SceneStamp(Arc::new(()));
        self.revision += 1;
        Ok(())
    }

    /// Commit finite rigid state while retaining the scene identity and last valid poses.
    pub fn commit_step(&mut self, next: State) -> Result<()> {
        if next.body_q.host()?.len() != self.model().body_count
            || next.body_qd.host()?.len() != self.model().body_count
            || next.body_f.host()?.len() != self.model().body_count
            || next.joint_q.host()?.len() != self.model().joint_coord_total
            || next.joint_qd.host()?.len() != self.model().joint_dof_total
        {
            return Err(Error::Solver(
                "rigid state dimensions do not match scene".into(),
            ));
        }
        for &pose in next.body_q.host()? {
            validate_pose(pose)?;
        }
        if next
            .body_qd
            .host()?
            .iter()
            .chain(next.body_f.host()?)
            .any(|v| !finite_spatial(*v))
            || next
                .joint_q
                .host()?
                .iter()
                .chain(next.joint_qd.host()?)
                .any(|v| !v.is_finite())
        {
            return Err(Error::Build("body edit produced non-finite state".into()));
        }
        let poses = next.body_q.host()?;
        for (&id, index) in &self.snapshot.bodies {
            self.snapshot.valid_poses.insert(id, poses[index.index()]);
        }
        self.snapshot.state = next;
        Ok(())
    }

    pub fn body_world_com(&self, id: BodyHandle) -> Result<Vec3> {
        let pose = self.body_pose(id)?;
        Ok(pose.position + pose.rotation * self.body_mass_properties(id)?.local_com)
    }

    pub fn body_velocity_at_point(&self, id: BodyHandle, point: Vec3) -> Result<Vec3> {
        let velocity = self.body_velocity(id)?;
        Ok(velocity.linear + velocity.angular.cross(point - self.body_world_com(id)?))
    }

    pub fn body_colliders(&self, id: BodyHandle) -> impl Iterator<Item = ColliderHandle> + '_ {
        self.colliders()
            .filter(move |(_, c)| c.parent == Some(id))
            .map(|(id, _)| id)
    }

    pub fn set_body_damping(&mut self, id: BodyHandle, linear: Real, angular: Real) -> Result<()> {
        if !linear.is_finite() || linear < 0.0 || !angular.is_finite() || angular < 0.0 {
            return Err(Error::Build("invalid body damping".into()));
        }
        let body = self
            .description
            .bodies
            .get_mut(&id)
            .ok_or_else(|| missing("body"))?;
        body.linear_damping = linear;
        body.angular_damping = angular;
        Ok(())
    }

    /// Add a world-space force and torque about the current COM.
    pub fn add_body_wrench(&mut self, id: BodyHandle, wrench: SpatialVector) -> Result<()> {
        if !wrench.linear.is_finite() || !wrench.angular.is_finite() {
            return Err(Error::Build("non-finite body wrench".into()));
        }
        let body = self.body(id).ok_or_else(|| missing("body"))?;
        if body.kind != BodyKind::Dynamic {
            return Ok(());
        }
        let i = self.body_index(id).ok_or_else(|| missing("body"))?.index();
        let current = self.state().body_f.host()?[i];
        let next = SpatialVector::new(
            current.linear + wrench.linear,
            current.angular + wrench.angular,
        );
        if !next.linear.is_finite() || !next.angular.is_finite() {
            return Err(Error::Build("body wrench overflow".into()));
        }
        self.state_mut().body_f.host_mut()?[i] = next;
        Ok(())
    }

    pub fn add_body_force_at_point(
        &mut self,
        id: BodyHandle,
        force: Vec3,
        point: Vec3,
    ) -> Result<()> {
        self.add_body_wrench(
            id,
            SpatialVector::new(force, (point - self.body_world_com(id)?).cross(force)),
        )
    }

    pub fn reset_body_forces(&mut self, id: BodyHandle) -> Result<()> {
        let i = self.body_index(id).ok_or_else(|| missing("body"))?.index();
        self.state_mut().body_f.host_mut()?[i] = SpatialVector::ZERO;
        Ok(())
    }

    /// The sole dense shape index; multi-piece colliders have no single index.
    pub fn collider_index(&self, id: ColliderHandle) -> Option<ShapeId> {
        self.collider_indices(id)
            .filter(|indices| indices.len() == 1)
            .map(|indices| indices[0])
    }

    pub fn collider_indices(&self, id: ColliderHandle) -> Option<&[ShapeId]> {
        self.snapshot.colliders.get(&id).map(Vec::as_slice)
    }

    pub fn collider(&self, id: ColliderHandle) -> Option<&ColliderDesc> {
        self.description.colliders.get(&id)
    }

    pub fn collider_is_enabled(&self, id: ColliderHandle) -> Result<bool> {
        let collider = self.collider(id).ok_or_else(|| missing("collider"))?;
        Ok(collider.enabled
            && collider
                .parent
                .is_none_or(|id| self.body(id).is_some_and(|b| b.enabled)))
    }

    /// Identity of collider geometry/material authoring, preserved by pose-only edits.
    pub fn collider_stamp(&self, id: ColliderHandle) -> Option<SceneStamp> {
        self.description.collider_stamps.get(&id).cloned()
    }

    pub fn collider_world_pose(&self, id: ColliderHandle) -> Result<Transform> {
        let desc = self.collider(id).ok_or_else(|| missing("collider"))?;
        match desc.parent {
            Some(parent) => {
                let i = self
                    .body_index(parent)
                    .ok_or_else(|| missing("body"))?
                    .index();
                Ok(self.state().body_q.host()?[i].compose(&desc.pose))
            }
            None => Ok(desc.pose),
        }
    }

    pub fn set_collider_pose(&mut self, id: ColliderHandle, pose: Transform) -> Result<()> {
        self.edit(|d| {
            d.colliders
                .get_mut(&id)
                .ok_or_else(|| missing("collider"))?
                .pose = pose;
            Ok(())
        })
    }

    pub fn set_collider_world_pose(&mut self, id: ColliderHandle, pose: Transform) -> Result<()> {
        validate_pose(pose)?;
        let desc = self.collider(id).ok_or_else(|| missing("collider"))?;
        let local = match desc.parent {
            Some(parent) => {
                let i = self
                    .body_index(parent)
                    .ok_or_else(|| missing("body"))?
                    .index();
                self.state().body_q.host()?[i].inverse().compose(&pose)
            }
            None => pose,
        };
        self.set_collider_pose(id, local)
    }

    pub fn body_handle(&self, index: BodyId) -> Option<BodyHandle> {
        self.snapshot
            .bodies
            .iter()
            .find_map(|(&id, &i)| (i == index).then_some(id))
    }

    pub fn collider_handle(&self, index: ShapeId) -> Option<ColliderHandle> {
        self.snapshot.shape_colliders.get(index.index()).copied()
    }

    /// Dense index of an active joint; suspended authored joints return `None`.
    pub fn joint_index(&self, id: JointHandle) -> Option<JointId> {
        self.snapshot.joints.get(&id).copied()
    }

    pub fn joint(&self, id: JointHandle) -> Option<&JointDesc> {
        self.description.joints.get(&id)
    }

    pub fn joint_settings(&self, id: JointHandle) -> Option<JointSettings> {
        self.description.joint_settings.get(&id).copied()
    }

    pub fn set_joint_settings(&mut self, id: JointHandle, settings: JointSettings) -> Result<()> {
        self.edit(|d| {
            *d.joint_settings
                .get_mut(&id)
                .ok_or_else(|| missing("joint"))? = settings;
            Ok(())
        })
    }

    pub fn joint_axis_settings(
        &self,
        id: JointHandle,
        axis: JointAxis,
    ) -> Result<JointAxisSettings> {
        axis.dof(self.joint(id).ok_or_else(|| missing("joint"))?.kind)?;
        Ok(self.description.joint_axes[&id][axis as usize])
    }

    pub fn joint_locked_axes(&self, id: JointHandle) -> Result<JointAxes> {
        let kind = self.joint(id).ok_or_else(|| missing("joint"))?.kind;
        let mut axes = JointAxes::NONE;
        for axis in JointAxis::ALL {
            if axis.dof(kind).is_err() || self.description.joint_axes[&id][axis as usize].locked {
                axes = axes.with(axis);
            }
        }
        Ok(axes)
    }

    pub fn set_joint_locked_axes(&mut self, id: JointHandle, locked: JointAxes) -> Result<()> {
        if self.joint(id).ok_or_else(|| missing("joint"))?.kind != JointType::D6
            || locked.0 & !63 != 0
        {
            return Err(Error::Build(
                "explicit axis locking requires a D6 joint and six-bit mask".into(),
            ));
        }
        self.edit(|d| {
            for axis in JointAxis::ALL {
                d.joint_axes.get_mut(&id).ok_or_else(|| missing("joint"))?[axis as usize].locked =
                    locked.contains(axis);
            }
            Ok(())
        })
    }

    pub fn set_joint_limits(
        &mut self,
        id: JointHandle,
        axis: JointAxis,
        limits: Option<[Real; 2]>,
    ) -> Result<()> {
        axis.dof(self.joint(id).ok_or_else(|| missing("joint"))?.kind)?;
        if limits.is_some_and(|[lo, hi]| !lo.is_finite() || !hi.is_finite() || lo > hi) {
            return Err(Error::Build("invalid joint axis limits".into()));
        }
        self.edit(|d| {
            d.joint_axes.get_mut(&id).ok_or_else(|| missing("joint"))?[axis as usize].limits =
                limits;
            Ok(())
        })
    }

    /// Update motor authoring and active controls without rebuilding the solver snapshot.
    pub fn set_joint_motor(
        &mut self,
        id: JointHandle,
        axis: JointAxis,
        motor: JointMotor,
    ) -> Result<()> {
        motor.validate()?;
        let dof = axis.dof(self.joint(id).ok_or_else(|| missing("joint"))?.kind)?;
        if let Some(joint) = self.joint_index(id) {
            let offset = self.model().joint_dof_offset.host()?[joint.index()] as usize;
            motor.apply(&mut self.snapshot.control, offset + dof)?;
        }
        self.description
            .joint_axes
            .get_mut(&id)
            .ok_or_else(|| missing("joint"))?[axis as usize]
            .motor = Some(motor);
        Ok(())
    }

    pub fn joints_between(
        &self,
        a: BodyHandle,
        b: BodyHandle,
    ) -> impl Iterator<Item = JointHandle> + '_ {
        self.joints()
            .filter(move |(_, j)| {
                (j.parent == Some(a) && j.child == b) || (j.parent == Some(b) && j.child == a)
            })
            .map(|(id, _)| id)
    }

    pub fn set_joint_frames(
        &mut self,
        id: JointHandle,
        parent: Transform,
        child: Transform,
    ) -> Result<()> {
        self.edit(|d| {
            let joint = d.joints.get_mut(&id).ok_or_else(|| missing("joint"))?;
            joint.frame_parent = parent;
            joint.frame_child = child;
            Ok(())
        })
    }

    pub fn bodies(&self) -> impl Iterator<Item = (BodyHandle, &BodyDesc)> {
        self.description.bodies.iter().map(|(&id, desc)| (id, desc))
    }

    pub fn colliders(&self) -> impl Iterator<Item = (ColliderHandle, &ColliderDesc)> {
        self.description
            .colliders
            .iter()
            .map(|(&id, desc)| (id, desc))
    }

    pub fn joints(&self) -> impl Iterator<Item = (JointHandle, &JointDesc)> {
        self.description.joints.iter().map(|(&id, desc)| (id, desc))
    }

    pub fn insert_body(&mut self, mut desc: BodyDesc) -> Result<BodyHandle> {
        if let Some(mass) = desc.additional_mass {
            mass.validate()?;
            desc.params.mass = mass.mass;
            desc.params.com = mass.local_com;
            desc.params.inertia = mass.inertia;
        }
        self.edit(|d| {
            let id = BodyHandle(d.allocate()?);
            d.bodies.insert(id, desc);
            Ok(id)
        })
    }

    /// Remove a body, its colliders and incident joints; surviving children become roots.
    pub fn remove_body(&mut self, id: BodyHandle) -> Result<()> {
        self.edit(|d| {
            d.bodies.remove(&id).ok_or_else(|| missing("body"))?;
            d.colliders.retain(|_, c| c.parent != Some(id));
            d.collider_stamps
                .retain(|id, _| d.colliders.contains_key(id));
            d.joints
                .retain(|_, j| j.parent != Some(id) && j.child != id);
            d.joint_settings.retain(|id, _| d.joints.contains_key(id));
            d.joint_axes.retain(|id, _| d.joints.contains_key(id));
            Ok(())
        })
    }

    pub fn set_body_kind(&mut self, id: BodyHandle, kind: BodyKind) -> Result<()> {
        self.edit(|d| {
            d.bodies.get_mut(&id).ok_or_else(|| missing("body"))?.kind = kind;
            Ok(())
        })
    }

    pub fn body_mass_properties(&self, id: BodyHandle) -> Result<MassProperties> {
        let i = self.body_index(id).ok_or_else(|| missing("body"))?.index();
        Ok(MassProperties {
            mass: self.model().body_mass.host()?[i],
            local_com: self.model().body_com.host()?[i],
            inertia: self.model().body_inertia.host()?[i],
        })
    }

    /// Replace total mass while preserving the velocity field and world force application points.
    pub fn set_body_mass_properties(&mut self, id: BodyHandle, mass: MassProperties) -> Result<()> {
        mass.validate()?;
        self.edit(|d| {
            let params = &mut d.bodies.get_mut(&id).ok_or_else(|| missing("body"))?.params;
            params.mass = mass.mass;
            params.com = mass.local_com;
            params.inertia = mass.inertia;
            Ok(())
        })
    }

    /// Commit a recomputed total and its independently replaceable additional component.
    pub fn set_body_mass_components(
        &mut self,
        id: BodyHandle,
        total: MassProperties,
        additional: MassProperties,
    ) -> Result<()> {
        total.validate()?;
        additional.validate()?;
        self.edit(|d| {
            let body = d.bodies.get_mut(&id).ok_or_else(|| missing("body"))?;
            body.params.mass = total.mass;
            body.params.com = total.local_com;
            body.params.inertia = total.inertia;
            body.additional_mass = Some(additional);
            Ok(())
        })
    }

    pub fn insert_collider(&mut self, desc: ColliderDesc) -> Result<ColliderHandle> {
        self.edit(|d| {
            let id = ColliderHandle(d.allocate()?);
            d.colliders.insert(id, desc);
            d.collider_stamps.insert(id, SceneStamp(Arc::new(())));
            Ok(id)
        })
    }

    pub fn replace_collider(&mut self, id: ColliderHandle, desc: ColliderDesc) -> Result<()> {
        self.edit(|d| {
            *d.colliders
                .get_mut(&id)
                .ok_or_else(|| missing("collider"))? = desc;
            d.collider_stamps.insert(id, SceneStamp(Arc::new(())));
            Ok(())
        })
    }

    pub fn set_collider_mass(&mut self, id: ColliderHandle, mass: ColliderMass) -> Result<()> {
        mass.validate()?;
        self.edit(|d| {
            d.colliders
                .get_mut(&id)
                .ok_or_else(|| missing("collider"))?
                .mass = mass;
            Ok(())
        })
    }

    pub fn remove_collider(&mut self, id: ColliderHandle) -> Result<()> {
        self.edit(|d| {
            d.colliders.remove(&id).ok_or_else(|| missing("collider"))?;
            d.collider_stamps.remove(&id);
            Ok(())
        })
    }

    pub fn insert_joint(&mut self, desc: JointDesc) -> Result<JointHandle> {
        self.edit(|d| {
            let id = JointHandle(d.allocate()?);
            d.joints.insert(id, desc);
            d.joint_settings.insert(id, JointSettings::default());
            d.joint_axes.insert(id, [JointAxisSettings::default(); 6]);
            Ok(id)
        })
    }

    pub fn remove_joint(&mut self, id: JointHandle) -> Result<()> {
        self.edit(|d| {
            d.joints.remove(&id).ok_or_else(|| missing("joint"))?;
            d.joint_settings.remove(&id);
            d.joint_axes.remove(&id);
            Ok(())
        })
    }

    pub fn set_gravity(&mut self, gravity: Vec3) -> Result<()> {
        if !gravity.is_finite() {
            return Err(Error::Build("non-finite gravity".into()));
        }
        self.snapshot.model.gravity.host_mut()?[0] = gravity;
        self.gravity = gravity;
        Ok(())
    }

    fn edit<T>(&mut self, change: impl FnOnce(&mut Description) -> Result<T>) -> Result<T> {
        let mut next = self.description.clone();
        let result = change(&mut next)?;
        let snapshot = compile(&next, self.gravity, Some(&self.snapshot))?;
        self.description = next;
        self.snapshot = snapshot;
        self.revision += 1;
        Ok(result)
    }
}

fn missing(kind: &str) -> Error {
    Error::Build(format!("unknown runtime {kind} handle"))
}

fn finite_spatial(value: SpatialVector) -> bool {
    value.linear.is_finite() && value.angular.is_finite()
}

fn compile(d: &Description, gravity: Vec3, old: Option<&Snapshot>) -> Result<Snapshot> {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), gravity);
    let mut bodies = BTreeMap::new();
    let mut poses = BTreeMap::new();
    for (&id, desc) in &d.bodies {
        validate_body(desc)?;
        let mut params = desc.params.clone();
        if let Some(previous) = old.and_then(|s| s.bodies.get(&id).map(|i| (s, i))) {
            params.pose = previous.0.state.body_q.host()?[previous.1.index()];
            if !desc.enabled && validate_pose(params.pose).is_err() {
                params.pose = previous.0.valid_poses[&id];
            }
        }
        validate_pose(params.pose)?;
        params.flags = if !desc.enabled {
            BodyFlags::KINEMATIC
        } else {
            match desc.kind {
                BodyKind::Dynamic => BodyFlags::DYNAMIC,
                BodyKind::Kinematic => BodyFlags::KINEMATIC,
                BodyKind::Fixed => BodyFlags::FIXED,
            }
        };
        poses.insert(id, params.pose);
        bodies.insert(id, builder.add_body(params));
    }
    let mut colliders = BTreeMap::new();
    let mut shape_colliders = Vec::new();
    for (&id, desc) in &d.colliders {
        validate_pose(desc.pose)?;
        desc.mass.validate()?;
        let parent = dense_body(&bodies, desc.parent)?;
        let shapes = add_shapes(&mut builder, parent, desc)?;
        if desc.friction < 0.0 || !(0.0..=1.0).contains(&desc.restitution) {
            return Err(Error::Build(
                "invalid runtime friction or restitution".into(),
            ));
        }
        let flags = if desc.enabled && desc.parent.is_none_or(|p| d.bodies[&p].enabled) {
            ShapeFlags::COLLIDES
        } else {
            0
        } | if desc.sensor { ShapeFlags::SENSOR } else { 0 };
        for &shape in &shapes {
            builder.set_shape_material(shape, desc.friction, desc.restitution)?;
            builder.set_shape_contact_material(shape, desc.contact_material)?;
            builder.set_shape_collision_groups(shape, desc.memberships, desc.filter)?;
            builder.set_shape_flags(shape, flags)?;
            shape_colliders.push(id);
        }
        colliders.insert(id, shapes);
    }
    let mut authored_incoming = BTreeMap::new();
    for (&id, joint) in &d.joints {
        dense_body(&bodies, joint.parent)?;
        dense_body(&bodies, Some(joint.child))?;
        validate_pose(joint.frame_parent)?;
        validate_pose(joint.frame_child)?;
        if matches!(joint.kind, JointType::Distance | JointType::Cable) {
            return Err(Error::Build("runtime joint kind is not supported".into()));
        }
        if matches!(joint.kind, JointType::Revolute | JointType::Prismatic)
            && (!joint.axis.length().is_finite() || joint.axis.length() < 1e-9)
        {
            return Err(Error::Build(
                "runtime joint axis must be finite and nonzero".into(),
            ));
        }
        if joint.parent == Some(joint.child) || authored_incoming.insert(joint.child, id).is_some()
        {
            return Err(Error::Build(
                "runtime joints require a forest with one parent per body".into(),
            ));
        }
    }
    for &body in d.bodies.keys() {
        let mut visited = BTreeSet::new();
        let mut cursor = Some(body);
        while let Some(body) = cursor {
            if !visited.insert(body) {
                return Err(Error::Build("runtime joint cycle".into()));
            }
            cursor = authored_incoming
                .get(&body)
                .and_then(|id| d.joints[id].parent);
        }
    }
    let mut incoming = BTreeMap::new();
    let mut children: BTreeMap<BodyHandle, Vec<JointHandle>> = BTreeMap::new();
    for (&id, joint) in &d.joints {
        if !d.joint_settings[&id].enabled
            || !d.bodies[&joint.child].enabled
            || joint.parent.is_some_and(|p| !d.bodies[&p].enabled)
        {
            continue;
        }
        incoming.insert(joint.child, id);
        if d.bodies[&joint.child].kind != BodyKind::Dynamic {
            return Err(Error::Build(
                "an immovable body must be an articulation root".into(),
            ));
        }
        if let Some(parent) = joint.parent {
            children.entry(parent).or_default().push(id);
        }
    }
    let mut joints = BTreeMap::new();
    let mut roots = BTreeMap::new();
    let mut visited = BTreeSet::new();
    for (&body, desc) in &d.bodies {
        if incoming
            .get(&body)
            .is_some_and(|j| d.joints[j].parent.is_some())
        {
            continue;
        }
        builder.begin_articulation();
        let mut stack = vec![body];
        while let Some(body) = stack.pop() {
            if !visited.insert(body) {
                return Err(Error::Build("runtime joint cycle".into()));
            }
            if let Some(&joint) = incoming.get(&body) {
                let j = add_joint(&mut builder, &bodies, &d.joints[&joint])?;
                let mut lower = vec![Real::NEG_INFINITY; d.joints[&joint].kind.dof_count()];
                let mut upper = vec![Real::INFINITY; lower.len()];
                for axis in JointAxis::ALL {
                    if let Ok(dof) = axis.dof(d.joints[&joint].kind) {
                        let setting = d.joint_axes[&joint][axis as usize];
                        let bounds = if setting.locked {
                            Some([0.0, 0.0])
                        } else {
                            setting.limits
                        };
                        if let Some([lo, hi]) = bounds {
                            lower[dof] = lo;
                            upper[dof] = hi;
                        }
                    }
                }
                builder.set_joint_limits(j, &lower, &upper)?;
                joints.insert(joint, j);
            } else {
                let pose = poses[&body];
                let j = if desc.enabled && desc.kind == BodyKind::Dynamic {
                    builder.add_joint_free(
                        BodyId::NONE,
                        bodies[&body],
                        Transform::IDENTITY,
                        Transform::IDENTITY,
                    )?
                } else {
                    builder.add_joint_fixed(
                        BodyId::NONE,
                        bodies[&body],
                        pose,
                        Transform::IDENTITY,
                    )?
                };
                roots.insert(body, j);
            }
            if let Some(next) = children.get(&body) {
                stack.extend(next.iter().rev().map(|j| d.joints[j].child));
            }
        }
    }
    if visited.len() != d.bodies.len() {
        return Err(Error::Build("runtime joint cycle".into()));
    }
    for (&id, joint) in &d.joints {
        if !d.joint_settings[&id].contacts_enabled
            && let Some(parent) = joint.parent
        {
            let shapes_for = |body| {
                d.colliders
                    .iter()
                    .filter(move |(_, c)| c.parent == Some(body))
                    .flat_map(|(id, _)| colliders[id].iter().copied())
            };
            for a in shapes_for(parent) {
                for b in shapes_for(joint.child) {
                    builder.add_shape_collision_exclusion(a, b);
                }
            }
        }
    }
    let mut model = builder.build()?;
    for (&id, index) in &bodies {
        if d.bodies[&id].enabled && d.bodies[&id].kind == BodyKind::Dynamic {
            let p = &d.bodies[&id].params;
            model.body_inv_inertia.host_mut()?[index.index()] = MassProperties {
                mass: p.mass,
                local_com: p.com,
                inertia: p.inertia,
            }
            .inverse_inertia()?;
        }
    }
    let mut state = model.state()?;
    let mut control = model.control();
    for (&id, index) in &bodies {
        let i = index.index();
        state.body_qd.host_mut()?[i] = d.bodies[&id].velocity;
        if let Some((s, index)) = old.and_then(|s| s.bodies.get(&id).map(|i| (s, i))) {
            state.body_qd.host_mut()?[i] = s.state.body_qd.host()?[index.index()];
            state.body_f.host_mut()?[i] = s.state.body_f.host()?[index.index()];
            control.body_external_f.host_mut()?[i] =
                s.control.body_external_f.host()?[index.index()];
            if !d.bodies[&id].enabled {
                for value in [
                    &mut state.body_qd.host_mut()?[i],
                    &mut state.body_f.host_mut()?[i],
                    &mut control.body_external_f.host_mut()?[i],
                ] {
                    if !finite_spatial(*value) {
                        *value = SpatialVector::ZERO;
                    }
                }
            }
            let shift = state.body_q.host()?[i].rotation
                * (model.body_com.host()?[i] - s.model.body_com.host()?[index.index()]);
            if shift != Vec3::ZERO {
                let velocity = &mut state.body_qd.host_mut()?[i];
                velocity.linear += velocity.angular.cross(shift);
                if !shift.is_finite() || !velocity.linear.is_finite() {
                    return Err(Error::Build("mass edit overflowed body velocity".into()));
                }
                for force in [
                    &mut state.body_f.host_mut()?[i],
                    &mut control.body_external_f.host_mut()?[i],
                ] {
                    force.angular -= shift.cross(force.linear);
                    if !force.angular.is_finite() {
                        return Err(Error::Build("mass edit overflowed body torque".into()));
                    }
                }
            }
        }
        if d.bodies[&id].kind == BodyKind::Fixed {
            state.body_qd.host_mut()?[i] = SpatialVector::ZERO;
        }
    }
    for (&body, &joint) in &roots {
        if d.bodies[&body].enabled && d.bodies[&body].kind == BodyKind::Dynamic {
            if let Some((s, &previous)) = old.and_then(|s| s.roots.get(&body).map(|j| (s, j)))
                && s.model.joint_type.host()?[previous.index()] == JointType::Free as u32
            {
                copy_joint(s, previous, &model, &mut state, &mut control, joint)?;
            } else {
                seed_joint(&model, &mut state, joint)?;
            }
        }
    }
    for (&id, &joint) in &joints {
        if let Some((s, &previous)) = old.and_then(|s| s.joints.get(&id).map(|j| (s, j))) {
            copy_joint(s, previous, &model, &mut state, &mut control, joint)?;
        } else {
            seed_joint(&model, &mut state, joint)?;
        }
    }
    let mut drives = old.map(|s| s.drives.clone()).unwrap_or_default();
    if let Some(old) = old {
        for (owner, joint) in old
            .roots
            .iter()
            .map(|(&id, &j)| (DriveOwner::Body(id), j))
            .chain(
                old.joints
                    .iter()
                    .map(|(&id, &j)| (DriveOwner::Joint(id), j)),
            )
        {
            let drive = DriveState::capture(old, joint)?;
            if !drive.mode.is_empty() {
                drives.insert(owner, drive);
            }
        }
    }
    drives.retain(|owner, _| match owner {
        DriveOwner::Body(id) => d.bodies.contains_key(id),
        DriveOwner::Joint(id) => d.joints.contains_key(id),
    });
    for (owner, joint) in roots
        .iter()
        .map(|(&id, &j)| (DriveOwner::Body(id), j))
        .chain(joints.iter().map(|(&id, &j)| (DriveOwner::Joint(id), j)))
    {
        if let Some(drive) = drives.get(&owner) {
            drive.restore(&model, &mut control, joint)?;
        }
    }
    for (&id, joint) in &joints {
        let offset = model.joint_dof_offset.host()?[joint.index()] as usize;
        for axis in JointAxis::ALL {
            if let Ok(dof) = axis.dof(d.joints[&id].kind)
                && let Some(motor) = d.joint_axes[&id][axis as usize].motor
            {
                motor.apply(&mut control, offset + dof)?;
            }
        }
    }
    Ok(Snapshot {
        stamp: SceneStamp(Arc::new(())),
        model,
        state,
        control,
        bodies,
        colliders,
        shape_colliders,
        joints,
        roots,
        drives,
        valid_poses: poses,
    })
}

fn dense_body(bodies: &BTreeMap<BodyHandle, BodyId>, id: Option<BodyHandle>) -> Result<BodyId> {
    id.map(|id| bodies.get(&id).copied().ok_or_else(|| missing("body")))
        .unwrap_or(Ok(BodyId::NONE))
}

fn validate_pose(pose: Transform) -> Result<()> {
    if !pose.position.is_finite()
        || !pose.rotation.is_finite()
        || (pose.rotation.length_squared() - 1.0).abs() > 1e-6
    {
        return Err(Error::Build(
            "runtime pose must be finite with a unit quaternion".into(),
        ));
    }
    Ok(())
}

fn validate_body(desc: &BodyDesc) -> Result<()> {
    validate_pose(desc.params.pose)?;
    MassProperties {
        mass: desc.params.mass,
        local_com: desc.params.com,
        inertia: desc.params.inertia,
    }
    .validate()?;
    if let Some(additional) = desc.additional_mass {
        additional.validate()?;
    }
    if !desc.linear_damping.is_finite()
        || desc.linear_damping < 0.0
        || !desc.angular_damping.is_finite()
        || desc.angular_damping < 0.0
    {
        return Err(Error::Build("invalid body damping".into()));
    }
    if desc.params.world != WorldId(0)
        || !desc.params.mass.is_finite()
        || desc.params.mass < 0.0
        || !desc.params.inertia.is_finite()
        || !desc.params.com.is_finite()
        || !desc.velocity.linear.is_finite()
        || !desc.velocity.angular.is_finite()
    {
        return Err(Error::Build(
            "invalid runtime body mass, inertia, velocity or world".into(),
        ));
    }
    Ok(())
}

fn add_shapes(
    builder: &mut ModelBuilder,
    body: BodyId,
    desc: &ColliderDesc,
) -> Result<Vec<ShapeId>> {
    let pose = desc.pose.compose(&desc.geometry.geometry_transform()?);
    validate_pose(pose)?;
    if let ColliderGeometry::ConvexParts(parts) = &desc.geometry {
        if parts.is_empty() {
            return Err(Error::Build("invalid convex decomposition pieces".into()));
        }
        return parts
            .iter()
            .map(|points| {
                validate_convex_points(points)?;
                let hull = builder.add_hull(points.clone());
                builder.add_shape_convex_hull(body, hull, pose)
            })
            .collect();
    }
    let shape = match &desc.geometry {
        ColliderGeometry::ConvexParts(_) => unreachable!(),
        ColliderGeometry::Ball { radius } => builder.add_shape_sphere(body, *radius, pose),
        ColliderGeometry::Cuboid { half_extents } => {
            builder.add_shape_box(body, *half_extents, pose)
        }
        ColliderGeometry::Capsule {
            half_height,
            radius,
        } => builder.add_shape_capsule(body, *half_height, *radius, pose),
        ColliderGeometry::CapsuleEndpoints { a, b, radius } => {
            builder.add_shape_capsule(body, (*b - *a).length() * 0.5, *radius, pose)
        }
        ColliderGeometry::Cylinder {
            half_height,
            radius,
        } => builder.add_shape_cylinder(body, *half_height, *radius, pose),
        ColliderGeometry::RoundCylinder {
            half_height,
            radius,
            border_radius,
        } => builder.add_shape_round_cylinder(body, *half_height, *radius, *border_radius, pose),
        ColliderGeometry::ConvexHull(points) => {
            validate_convex_points(points)?;
            let hull = builder.add_hull(points.clone());
            builder.add_shape_convex_hull(body, hull, pose)
        }
        ColliderGeometry::TriMesh(mesh) => {
            if mesh.vertices.iter().any(|p| !p.is_finite())
                || mesh
                    .indices
                    .iter()
                    .flatten()
                    .any(|&i| i as usize >= mesh.vertices.len())
            {
                return Err(Error::Build("invalid runtime mesh".into()));
            }
            let mesh = builder.add_mesh(mesh.clone());
            builder.add_shape_trimesh(body, mesh, pose)
        }
        ColliderGeometry::Heightfield(field) => {
            field.validate()?;
            let field = builder.add_heightfield(field.clone());
            builder.add_shape_heightfield(body, field, pose)
        }
    }?;
    Ok(vec![shape])
}

fn add_joint(
    builder: &mut ModelBuilder,
    bodies: &BTreeMap<BodyHandle, BodyId>,
    desc: &JointDesc,
) -> Result<JointId> {
    let p = dense_body(bodies, desc.parent)?;
    let c = dense_body(bodies, Some(desc.child))?;
    let (xp, xc) = (desc.frame_parent, desc.frame_child);
    match desc.kind {
        JointType::Revolute => builder.add_joint_revolute(p, c, xp, xc, desc.axis),
        JointType::Prismatic => builder.add_joint_prismatic(p, c, xp, xc, desc.axis),
        JointType::Fixed => builder.add_joint_fixed(p, c, xp, xc),
        JointType::Ball => builder.add_joint_ball(p, c, xp, xc),
        JointType::Free => builder.add_joint_free(p, c, xp, xc),
        JointType::D6 => builder.add_joint_d6(p, c, xp, xc),
        JointType::Distance | JointType::Cable => {
            Err(Error::Build("runtime joint kind is not supported".into()))
        }
    }
}

fn seed_joint(model: &Model, state: &mut State, joint: JointId) -> Result<()> {
    let j = joint.index();
    let parent = model.joint_parent.host()?[j];
    let child = model.joint_child.host()?[j] as usize;
    let poses = state.body_q.host()?;
    let velocities = state.body_qd.host()?;
    let child_frame = poses[child].compose(&model.joint_x_c.host()?[j]);
    let parent_frame = if parent < 0 {
        model.joint_x_p.host()?[j]
    } else {
        poses[parent as usize].compose(&model.joint_x_p.host()?[j])
    };
    let relative = parent_frame.inverse().compose(&child_frame);
    let body_com = model.body_com.host()?;
    let com = |i: usize| poses[i].position + poses[i].rotation * body_com[i];
    let child_v = velocities[child];
    let parent_v = if parent < 0 {
        SpatialVector::ZERO
    } else {
        velocities[parent as usize]
    };
    let v_child = child_v.linear + child_v.angular.cross(child_frame.position - com(child));
    let v_parent = if parent < 0 {
        Vec3::ZERO
    } else {
        parent_v.linear
            + parent_v
                .angular
                .cross(child_frame.position - com(parent as usize))
    };
    let linear = parent_frame.rotation.conjugate() * (v_child - v_parent);
    let angular = parent_frame.rotation.conjugate() * (child_v.angular - parent_v.angular);
    let c = model.joint_coord_offset.host()?[j] as usize;
    let v = model.joint_dof_offset.host()?[j] as usize;
    let q = state.joint_q.host_mut()?;
    let qd = state.joint_qd.host_mut()?;
    match model.joint_type.host()?[j] {
        x if x == JointType::Free as u32 || x == JointType::D6 as u32 => {
            q[c..c + 3].copy_from_slice(&relative.position.to_array());
            q[c + 3..c + 7].copy_from_slice(&relative.rotation.to_array());
            qd[v..v + 3].copy_from_slice(&linear.to_array());
            qd[v + 3..v + 6].copy_from_slice(&angular.to_array());
        }
        x if x == JointType::Ball as u32 => {
            q[c..c + 4].copy_from_slice(&relative.rotation.to_array());
            qd[v..v + 3].copy_from_slice(&angular.to_array());
        }
        x if x == JointType::Prismatic as u32 => {
            let axis = model.joint_axis.host()?[j];
            q[c] = relative.position.dot(axis);
            qd[v] = linear.dot(axis);
        }
        x if x == JointType::Revolute as u32 => {
            let axis = model.joint_axis.host()?[j];
            let r = relative.rotation;
            q[c] = 2.0 * Vec3::new(r.x, r.y, r.z).dot(axis).atan2(r.w);
            qd[v] = angular.dot(axis);
        }
        _ => {}
    }
    Ok(())
}

fn copy_joint(
    old: &Snapshot,
    previous: JointId,
    model: &Model,
    state: &mut State,
    control: &mut Control,
    joint: JointId,
) -> Result<()> {
    let range = |model: &Model, joint: JointId, coords: bool| -> Result<std::ops::Range<usize>> {
        let offsets = if coords {
            model.joint_coord_offset.host()?
        } else {
            model.joint_dof_offset.host()?
        };
        Ok(offsets[joint.index()] as usize..offsets[joint.index() + 1] as usize)
    };
    let cq = range(model, joint, true)?;
    let oq = range(&old.model, previous, true)?;
    let cv = range(model, joint, false)?;
    let ov = range(&old.model, previous, false)?;
    state.joint_q.host_mut()?[cq].copy_from_slice(&old.state.joint_q.host()?[oq]);
    state.joint_qd.host_mut()?[cv.clone()].copy_from_slice(&old.state.joint_qd.host()?[ov.clone()]);
    macro_rules! copy_control {
        ($($field:ident),+) => { $(control.$field.host_mut()?[cv.clone()].copy_from_slice(&old.control.$field.host()?[ov.clone()]);)+ };
    }
    copy_control!(
        joint_target_mode,
        joint_target_pos,
        joint_target_vel,
        joint_target_kp,
        joint_target_kd,
        joint_motor_model,
        joint_motor_max_force,
        joint_force
    );
    Ok(())
}
