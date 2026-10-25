//! Transactional rigid-scene mutation with persistent handles and dense solver snapshots.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

mod builder;
mod readout;
mod tree;

pub use builder::RigidSceneBuilder;

pub use crate::joint_control::{
    DistanceLimits, JointBreakLimits, JointBreakTrigger, SphericalConeLimits,
};
use crate::joint_control::{JointAxes, JointAxis, JointAxisSettings, JointMotor, JointSoftness};
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

/// State, force, control and description identity for solved dynamics readouts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DynamicsStamp(SceneStamp, u64);

impl PartialEq for SceneStamp {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl Eq for SceneStamp {}

/// Input-mutation identity for one body, including global scene edits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BodyActivationStamp(SceneStamp, u64, u64);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BodyKind {
    Dynamic,
    Kinematic,
    Fixed,
}

#[derive(Clone, Debug, PartialEq)]
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
    /// Velocity of the collider's surface in its own frame, as a belt
    /// running over it: contacts are dragged along its tangential part.
    pub surface_velocity: Vec3,
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
            surface_velocity: Vec3::ZERO,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
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
    body_identities: BTreeMap<BodyHandle, SceneStamp>,
    kinematic_acceleration: BTreeMap<BodyHandle, SpatialVector>,
    colliders: BTreeMap<ColliderHandle, ColliderDesc>,
    collider_stamps: BTreeMap<ColliderHandle, SceneStamp>,
    joints: BTreeMap<JointHandle, JointDesc>,
    joint_identities: BTreeMap<JointHandle, SceneStamp>,
    joint_settings: BTreeMap<JointHandle, JointSettings>,
    joint_axes: BTreeMap<JointHandle, [JointAxisSettings; 6]>,
    joint_softness: BTreeMap<JointHandle, JointSoftness>,
    loop_joints: BTreeSet<JointHandle>,
    distance_limits: BTreeMap<JointHandle, DistanceLimits>,
    cone_limits: BTreeMap<JointHandle, SphericalConeLimits>,
    break_limits: BTreeMap<JointHandle, JointBreakLimits>,
    broken_joints: BTreeSet<JointHandle>,
    body_pair_exclusions: BTreeSet<(BodyHandle, BodyHandle)>,
    collider_pair_exclusions: BTreeSet<(ColliderHandle, ColliderHandle)>,
}

impl Description {
    fn insert_body(&mut self, mut desc: BodyDesc) -> Result<BodyHandle> {
        if let Some(mass) = desc.additional_mass {
            mass.validate()?;
            desc.params.mass = mass.mass;
            desc.params.com = mass.local_com;
            desc.params.inertia = mass.inertia;
        }
        let id = BodyHandle(self.allocate()?);
        self.bodies.insert(id, desc);
        self.body_identities.insert(id, SceneStamp(Arc::new(())));
        Ok(id)
    }

    fn insert_collider(&mut self, desc: ColliderDesc) -> Result<ColliderHandle> {
        let id = ColliderHandle(self.allocate()?);
        self.colliders.insert(id, desc);
        self.collider_stamps.insert(id, SceneStamp(Arc::new(())));
        Ok(id)
    }

    fn insert_joint(&mut self, desc: JointDesc) -> Result<JointHandle> {
        let id = JointHandle(self.allocate()?);
        if desc.kind == JointType::Distance {
            self.distance_limits.insert(id, DistanceLimits::default());
            self.loop_joints.insert(id);
        }
        self.joints.insert(id, desc);
        self.joint_identities.insert(id, SceneStamp(Arc::new(())));
        self.joint_settings.insert(id, JointSettings::default());
        self.joint_axes
            .insert(id, [JointAxisSettings::default(); 6]);
        Ok(id)
    }

    fn allocate(&mut self) -> Result<u64> {
        self.next_id = self
            .next_id
            .checked_add(1)
            .ok_or_else(|| Error::Build("runtime handle space exhausted".into()))?;
        Ok(self.next_id)
    }
}

struct Snapshot {
    model_stamp: SceneStamp,
    stamp: SceneStamp,
    model: Model,
    state: State,
    control: Control,
    bodies: BTreeMap<BodyHandle, BodyId>,
    body_handles: Vec<BodyHandle>,
    colliders: BTreeMap<ColliderHandle, Vec<ShapeId>>,
    shape_colliders: Vec<ColliderHandle>,
    joints: BTreeMap<JointHandle, JointId>,
    joint_layouts: BTreeMap<JointHandle, [Option<usize>; 6]>,
    soft_joints: BTreeSet<JointHandle>,
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
    layout: [Option<usize>; 6],
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
    fn capture(snapshot: &Snapshot, joint: JointId, layout: [Option<usize>; 6]) -> Result<Self> {
        let offsets = snapshot.model.joint_dof_offset.host()?;
        let range = offsets[joint.index()] as usize..offsets[joint.index() + 1] as usize;
        let clean = |values: &[Real]| {
            values[range.clone()]
                .iter()
                .map(|&v| if v.is_finite() { v } else { 0.0 })
                .collect()
        };
        Ok(Self {
            layout,
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

    fn restore(
        &self,
        model: &Model,
        control: &mut Control,
        joint: JointId,
        layout: [Option<usize>; 6],
    ) -> Result<()> {
        let offsets = model.joint_dof_offset.host()?;
        let range = offsets[joint.index()] as usize..offsets[joint.index() + 1] as usize;
        for (source, destination) in self.layout.iter().zip(layout) {
            let (Some(source), Some(destination)) = (*source, destination) else {
                continue;
            };
            let destination = range.start + destination;
            control.joint_target_mode.host_mut()?[destination] = self.mode[source];
            control.joint_motor_model.host_mut()?[destination] = self.motor_model[source];
            for (buffer, values) in [
                (&mut control.joint_target_pos, &self.position),
                (&mut control.joint_target_vel, &self.velocity),
                (&mut control.joint_target_kp, &self.kp),
                (&mut control.joint_target_kd, &self.kd),
                (&mut control.joint_force, &self.force),
                (&mut control.joint_motor_max_force, &self.max_force),
            ] {
                buffer.host_mut()?[destination] = values[source];
            }
        }
        Ok(())
    }
}

/// CPU-owned rigid scene. Successful edits invalidate dense indices, not handles.
/// Implicit free roots use world COM translation and velocity; authored joints retain their frames.
pub struct RigidScene {
    description: Description,
    snapshot: Snapshot,
    gravity: Vec3,
    revision: u64,
    activation_global: u64,
    activation_identity: SceneStamp,
    activation_bodies: BTreeMap<BodyHandle, u64>,
    structure_stamps: BTreeMap<BodyHandle, SceneStamp>,
    contact_stamps: BTreeMap<BodyHandle, SceneStamp>,
    contact_episode_stamps: BTreeMap<BodyHandle, SceneStamp>,
    state_edit_stamp: SceneStamp,
    state_base_stamp: SceneStamp,
    body_state_stamps: BTreeMap<BodyHandle, SceneStamp>,
    dynamics_stamp: DynamicsStamp,
    solver_input_stamp: DynamicsStamp,
}

pub struct RigidStepCheckpoint {
    model_stamp: SceneStamp,
    state: State,
    valid_poses: BTreeMap<BodyHandle, Transform>,
    dynamics_stamp: DynamicsStamp,
    solver_input_stamp: DynamicsStamp,
    activation_global: u64,
    activation_identity: SceneStamp,
    activation_bodies: BTreeMap<BodyHandle, u64>,
}

impl RigidScene {
    /// Save host state and activation validity before solver-only commits.
    pub fn checkpoint_step(&self) -> Result<RigidStepCheckpoint> {
        Ok(RigidStepCheckpoint {
            model_stamp: self.model_stamp(),
            state: self.state().clone_host()?,
            valid_poses: self.snapshot.valid_poses.clone(),
            dynamics_stamp: self.dynamics_stamp.clone(),
            solver_input_stamp: self.solver_input_stamp.clone(),
            activation_global: self.activation_global,
            activation_identity: self.activation_identity.clone(),
            activation_bodies: self.activation_bodies.clone(),
        })
    }

    /// Restore solver-only commits while retaining the same compiled model and controls.
    pub fn restore_step(&mut self, checkpoint: RigidStepCheckpoint) -> Result<()> {
        if self.model_stamp() != checkpoint.model_stamp {
            return Err(Error::Build(
                "step checkpoint belongs to another compiled model".into(),
            ));
        }
        if self.solver_input_stamp != checkpoint.solver_input_stamp {
            return Err(Error::Build(
                "step checkpoint cannot restore across dynamics input edits".into(),
            ));
        }
        self.snapshot.state = checkpoint.state;
        self.snapshot.valid_poses = checkpoint.valid_poses;
        self.dynamics_stamp = checkpoint.dynamics_stamp;
        self.activation_global = checkpoint.activation_global;
        self.activation_identity = checkpoint.activation_identity;
        self.activation_bodies = checkpoint.activation_bodies;
        Ok(())
    }

    /// Fork current host dynamics and authoring into an independent compiled snapshot.
    pub fn fork(&self) -> Result<Self> {
        Ok(Self {
            description: self.description.clone(),
            snapshot: compile(&self.description, self.gravity, Some(&self.snapshot))?,
            gravity: self.gravity,
            revision: self.revision,
            activation_global: self.activation_global,
            activation_identity: self.activation_identity.clone(),
            activation_bodies: self.activation_bodies.clone(),
            structure_stamps: self.structure_stamps.clone(),
            contact_stamps: self.contact_stamps.clone(),
            contact_episode_stamps: self.contact_episode_stamps.clone(),
            state_edit_stamp: self.state_edit_stamp.clone(),
            state_base_stamp: self.state_base_stamp.clone(),
            body_state_stamps: self.body_state_stamps.clone(),
            dynamics_stamp: self.dynamics_stamp.clone(),
            solver_input_stamp: self.solver_input_stamp.clone(),
        })
    }
    pub fn new() -> Result<Self> {
        Self::from_description(Description::default(), Vec3::new(0.0, -9.81, 0.0))
    }

    fn from_description(description: Description, gravity: Vec3) -> Result<Self> {
        let snapshot = compile(&description, gravity, None)?;
        let empty = Description::default();
        let mut structure_stamps = BTreeMap::new();
        let mut contact_stamps = BTreeMap::new();
        let mut contact_episode_stamps = BTreeMap::new();
        readout::update_structure_stamps::<false, false>(
            &mut structure_stamps,
            &empty,
            &description,
        );
        readout::update_structure_stamps::<true, false>(&mut contact_stamps, &empty, &description);
        readout::update_structure_stamps::<true, true>(
            &mut contact_episode_stamps,
            &empty,
            &description,
        );
        Ok(Self {
            description,
            snapshot,
            gravity,
            revision: 0,
            activation_global: 0,
            activation_identity: SceneStamp(Arc::new(())),
            activation_bodies: BTreeMap::new(),
            structure_stamps,
            contact_stamps,
            contact_episode_stamps,
            state_edit_stamp: SceneStamp(Arc::new(())),
            state_base_stamp: SceneStamp(Arc::new(())),
            body_state_stamps: BTreeMap::new(),
            dynamics_stamp: DynamicsStamp(SceneStamp(Arc::new(())), 0),
            solver_input_stamp: DynamicsStamp(SceneStamp(Arc::new(())), 0),
        })
    }

    pub fn model(&self) -> &Model {
        &self.snapshot.model
    }

    /// Allocate and upload the current compiled model on the selected device.
    #[cfg(feature = "gpu")]
    pub fn upload_model_to_device(
        &mut self,
        device: &Arc<wgpu::Device>,
        queue: &wgpu::Queue,
    ) -> Result<()> {
        self.snapshot.model.upload_to_device(device, queue)
    }

    /// Synchronize gravity, tree compliance and controls without replacing state or scene identity.
    #[cfg(feature = "gpu")]
    pub fn upload_controls_to_device(
        &mut self,
        device: &Arc<wgpu::Device>,
        queue: &wgpu::Queue,
    ) -> Result<()> {
        self.snapshot.model.gravity.ensure_device(device)?;
        self.snapshot.model.gravity.upload(queue)?;
        self.snapshot.model.joint_compliance.ensure_device(device)?;
        self.snapshot.model.joint_compliance.upload(queue)?;
        self.snapshot.control.upload_to_device(device, queue)
    }
    pub fn state(&self) -> &State {
        &self.snapshot.state
    }
    pub fn state_mut(&mut self) -> &mut State {
        self.activate_all();
        self.state_edit_stamp = SceneStamp(Arc::new(()));
        self.state_base_stamp = self.state_edit_stamp.clone();
        self.body_state_stamps.clear();
        &mut self.snapshot.state
    }
    pub fn control(&self) -> &Control {
        &self.snapshot.control
    }
    pub fn control_mut(&mut self) -> &mut Control {
        self.activate_all();
        &mut self.snapshot.control
    }
    pub fn revision(&self) -> u64 {
        self.revision
    }
    pub fn stamp(&self) -> SceneStamp {
        self.snapshot.stamp.clone()
    }

    pub fn dynamics_stamp(&self) -> DynamicsStamp {
        self.dynamics_stamp.clone()
    }

    fn touch_dynamics(&mut self) {
        let (version, overflow) = self.dynamics_stamp.1.overflowing_add(1);
        if overflow {
            self.dynamics_stamp.0 = SceneStamp(Arc::new(()));
        }
        self.dynamics_stamp.1 = version;
        self.solver_input_stamp = self.dynamics_stamp.clone();
    }

    /// Compiled-model identity; pose, velocity, gravity and control edits retain it.
    pub fn model_stamp(&self) -> SceneStamp {
        self.snapshot.model_stamp.clone()
    }
    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    pub fn activation_stamp(&self, id: BodyHandle) -> Option<BodyActivationStamp> {
        self.body(id).map(|_| {
            BodyActivationStamp(
                self.activation_identity.clone(),
                self.activation_global,
                self.activation_bodies.get(&id).copied().unwrap_or(0),
            )
        })
    }

    fn activate_all(&mut self) {
        self.touch_dynamics();
        self.activation_global = self.activation_global.wrapping_add(1);
        self.activation_bodies.clear();
    }

    fn activate_body(&mut self, id: BodyHandle) {
        self.touch_dynamics();
        let version = self.activation_bodies.entry(id).or_default();
        *version = version.wrapping_add(1);
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
        self.set_body_velocity_with_wake(id, velocity, true)
    }

    /// Set projected COM velocity, optionally activating the body's dependency group.
    pub fn set_body_velocity_with_wake(
        &mut self,
        id: BodyHandle,
        velocity: SpatialVector,
        wake: bool,
    ) -> Result<()> {
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
        let changed = [id];
        self.commit_body_state(next, &changed, wake)
    }

    /// Atomically set dense rates in the current model layout and optionally activate changed joints.
    pub fn set_generalized_velocities_with_wake(
        &mut self,
        velocities: &[Real],
        wake: bool,
    ) -> Result<()> {
        if velocities.len() != self.model().joint_dof_total
            || velocities.iter().any(|v| !v.is_finite())
        {
            return Err(Error::Build("invalid generalized velocities".into()));
        }
        let previous = self.state().joint_qd.host()?;
        if previous.len() != velocities.len() {
            return Err(Error::Build("invalid generalized velocity state".into()));
        }
        let offsets = self.model().joint_dof_offset.host()?;
        let children = self.model().joint_child.host()?;
        let mut changed = Vec::new();
        for (joint, range) in offsets.windows(2).enumerate() {
            let range = range[0] as usize..range[1] as usize;
            if previous[range.clone()] != velocities[range] {
                changed.push(
                    self.body_handle(BodyId(children[joint] as u32))
                        .ok_or_else(|| missing("joint body"))?,
                );
            }
        }
        if changed.is_empty() {
            return Ok(());
        }
        let mut next = self.state().clone_host()?;
        next.joint_qd.host_mut()?.copy_from_slice(velocities);
        crate::eval_body_velocities(self.model(), &mut next)?;
        self.commit_body_state(next, &changed, wake)
    }

    /// Teleport a body's subtree, projecting its pose onto any incoming joint freedoms.
    pub fn set_body_pose(&mut self, id: BodyHandle, pose: Transform) -> Result<()> {
        self.set_body_pose_with_wake(id, pose, true)
    }

    /// Teleport a subtree with optional activation; geometry and query state always update.
    pub fn set_body_pose_with_wake(
        &mut self,
        id: BodyHandle,
        pose: Transform,
        wake: bool,
    ) -> Result<()> {
        validate_pose(pose)?;
        let kind = self.body(id).ok_or_else(|| missing("body"))?.kind;
        let previous = self.body_pose(id)?;
        if pose == previous {
            return Ok(());
        }
        let delta = pose.compose(&previous.inverse());
        let root = self.body_index(id).ok_or_else(|| missing("body"))?.index();
        let mut moved = BTreeSet::from([root]);
        let parents = self.model().joint_parent.host()?;
        for (j, &child) in self.model().joint_child.host()?.iter().enumerate() {
            if parents[j] >= 0 && moved.contains(&(parents[j] as usize)) {
                moved.insert(child as usize);
            }
        }
        let mut next = self.state().clone_host()?;
        for i in moved {
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
            seed_joint_near(
                self.model(),
                &mut next,
                JointId::from_index(joint),
                self.state(),
            )?;
        }
        crate::eval_fk(self.model(), &mut next)?;
        crate::eval_body_velocities(self.model(), &mut next)?;
        let changed = [id];
        self.commit_body_state(next, &changed, wake)
    }

    /// Atomically set world poses; unlisted descendants follow their parent.
    /// Joint freedoms project requested poses. Rates persist unless reset.
    pub fn set_body_poses(
        &mut self,
        poses: &[(BodyHandle, Transform)],
        reset_velocity: bool,
    ) -> Result<()> {
        let mut requested = BTreeMap::new();
        for &(id, pose) in poses {
            validate_pose(pose)?;
            let index = self.body_index(id).ok_or_else(|| missing("body"))?.index();
            if requested.insert(index, pose).is_some() {
                return Err(Error::Build("duplicate body in pose batch".into()));
            }
        }
        if requested.is_empty() {
            return Ok(());
        }
        let model = self.model();
        let original = self.state().body_q.host()?;
        let mut next = self.state().clone_host()?;
        let mut moved = BTreeSet::new();
        for (&index, &pose) in &requested {
            next.body_q.host_mut()?[index] = pose;
            moved.insert(index);
        }
        let parents = model.joint_parent.host()?;
        let children = model.joint_child.host()?;
        for (j, &child) in children.iter().enumerate() {
            let child = child as usize;
            let parent = parents[j];
            if parent >= 0 && moved.contains(&(parent as usize)) && !requested.contains_key(&child)
            {
                let parent = parent as usize;
                let delta = next.body_q.host()?[parent].compose(&original[parent].inverse());
                next.body_q.host_mut()?[child] = delta.compose(&original[child]);
                moved.insert(child);
            }
        }
        let rates = next.joint_qd.host()?.to_vec();
        for (j, &child) in children.iter().enumerate() {
            if requested.contains_key(&(child as usize)) {
                seed_joint_near(model, &mut next, JointId::from_index(j), self.state())?;
            }
        }
        next.joint_qd.host_mut()?.copy_from_slice(&rates);
        if reset_velocity {
            for (j, &child) in children.iter().enumerate() {
                if moved.contains(&(child as usize)) {
                    let v = model.joint_dof_offset.host()?;
                    next.joint_qd.host_mut()?[v[j] as usize..v[j + 1] as usize].fill(0.0);
                }
            }
            for i in moved {
                next.body_qd.host_mut()?[i] = SpatialVector::ZERO;
            }
        }
        crate::eval_fk(model, &mut next)?;
        crate::eval_body_velocities(model, &mut next)?;
        self.commit_body_state(
            next,
            &poses.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            true,
        )
    }

    fn commit_body_state(&mut self, next: State, changed: &[BodyHandle], wake: bool) -> Result<()> {
        self.commit_state(next)?;
        if wake {
            for &id in changed {
                self.activate_body(id);
            }
        }
        self.snapshot.stamp = SceneStamp(Arc::new(()));
        self.state_edit_stamp = self.snapshot.stamp.clone();
        self.mark_body_state_edits(changed);
        self.revision += 1;
        Ok(())
    }

    /// Commit finite rigid state while retaining the scene identity and last valid poses.
    pub fn commit_step(&mut self, next: State) -> Result<()> {
        let inputs = self.solver_input_stamp.clone();
        self.commit_state(next)?;
        self.activate_all();
        self.solver_input_stamp = inputs;
        Ok(())
    }

    fn commit_state(&mut self, next: State) -> Result<()> {
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
        self.touch_dynamics();
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
        let changed = body.linear_damping != linear || body.angular_damping != angular;
        body.linear_damping = linear;
        body.angular_damping = angular;
        if changed {
            self.activate_body(id);
        }
        Ok(())
    }

    /// Add a world-space force and torque about the current COM.
    pub fn add_body_wrench(&mut self, id: BodyHandle, wrench: SpatialVector) -> Result<()> {
        self.add_body_wrench_with_wake(id, wrench, true)
    }

    /// Accumulate a persistent COM wrench, optionally activating the body on change.
    pub fn add_body_wrench_with_wake(
        &mut self,
        id: BodyHandle,
        wrench: SpatialVector,
        wake: bool,
    ) -> Result<()> {
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
        self.snapshot.state.body_f.host_mut()?[i] = next;
        if current != next {
            self.touch_dynamics();
        }
        if wake && current != next {
            self.activate_body(id);
        }
        Ok(())
    }

    pub fn add_body_force_at_point(
        &mut self,
        id: BodyHandle,
        force: Vec3,
        point: Vec3,
    ) -> Result<()> {
        self.add_body_force_at_point_with_wake(id, force, point, true)
    }

    /// Accumulate a world-space point force with optional activation.
    pub fn add_body_force_at_point_with_wake(
        &mut self,
        id: BodyHandle,
        force: Vec3,
        point: Vec3,
        wake: bool,
    ) -> Result<()> {
        self.add_body_wrench_with_wake(
            id,
            SpatialVector::new(force, (point - self.body_world_com(id)?).cross(force)),
            wake,
        )
    }

    pub fn reset_body_forces(&mut self, id: BodyHandle) -> Result<()> {
        self.reset_body_forces_with_wake(id, true)
    }

    /// Clear persistent force and torque, optionally activating the body on change.
    pub fn reset_body_forces_with_wake(&mut self, id: BodyHandle, wake: bool) -> Result<()> {
        let i = self.body_index(id).ok_or_else(|| missing("body"))?.index();
        let changed = self.snapshot.state.body_f.host()?[i] != SpatialVector::ZERO;
        self.snapshot.state.body_f.host_mut()?[i] = SpatialVector::ZERO;
        if changed {
            self.touch_dynamics();
        }
        if wake && changed {
            self.activate_body(id);
        }
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

    /// Set a collider's belt surface velocity (its own frame) in place,
    /// without a rebuild; a change wakes the scene so resting bodies ride it.
    pub fn set_collider_surface_velocity(
        &mut self,
        id: ColliderHandle,
        velocity: Vec3,
    ) -> Result<()> {
        if !velocity.is_finite() {
            return Err(Error::Build(
                "collider surface velocity must be finite".into(),
            ));
        }
        let desc = self
            .description
            .colliders
            .get_mut(&id)
            .ok_or_else(|| missing("collider"))?;
        if desc.surface_velocity == velocity {
            return Ok(());
        }
        desc.surface_velocity = velocity;
        let in_body = desc.pose.rotation * velocity;
        for shape in self.snapshot.colliders.get(&id).into_iter().flatten() {
            self.snapshot.model.shape_surface_velocity[shape.index()] = in_body;
        }
        self.activate_all();
        Ok(())
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
        self.snapshot.body_handles.get(index.index()).copied()
    }

    pub fn collider_handle(&self, index: ShapeId) -> Option<ColliderHandle> {
        self.snapshot.shape_colliders.get(index.index()).copied()
    }

    /// Dense index of an active tree joint; suspended and force joints return `None`.
    pub fn joint_index(&self, id: JointHandle) -> Option<JointId> {
        self.snapshot.joints.get(&id).copied()
    }

    pub fn joint(&self, id: JointHandle) -> Option<&JointDesc> {
        self.description.joints.get(&id)
    }

    pub fn body_pair_excluded(&self, a: BodyHandle, b: BodyHandle) -> Result<bool> {
        self.body(a).ok_or_else(|| missing("body"))?;
        self.body(b).ok_or_else(|| missing("body"))?;
        Ok(self
            .description
            .body_pair_exclusions
            .contains(&(a.min(b), a.max(b))))
    }

    /// Exclude every collider pair across two bodies independently of joint policy.
    pub fn set_body_pair_excluded(
        &mut self,
        a: BodyHandle,
        b: BodyHandle,
        excluded: bool,
    ) -> Result<()> {
        if a == b {
            return Err(Error::Build("body pair endpoints must differ".into()));
        }
        if self.body_pair_excluded(a, b)? == excluded {
            return Ok(());
        }
        self.rebuild(|d| {
            let pair = (a.min(b), a.max(b));
            if excluded {
                d.body_pair_exclusions.insert(pair);
            } else {
                d.body_pair_exclusions.remove(&pair);
            }
            Ok(())
        })?;
        self.activate_body(a);
        self.activate_body(b);
        Ok(())
    }

    pub fn collider_pair_excluded(&self, a: ColliderHandle, b: ColliderHandle) -> Result<bool> {
        self.collider(a).ok_or_else(|| missing("collider"))?;
        self.collider(b).ok_or_else(|| missing("collider"))?;
        Ok(self
            .description
            .collider_pair_exclusions
            .contains(&(a.min(b), a.max(b))))
    }

    /// Exclude all compiled shape pairs for two stable collider handles.
    pub fn set_collider_pair_excluded(
        &mut self,
        a: ColliderHandle,
        b: ColliderHandle,
        excluded: bool,
    ) -> Result<()> {
        if a == b {
            return Err(Error::Build("collider pair endpoints must differ".into()));
        }
        if self.collider_pair_excluded(a, b)? == excluded {
            return Ok(());
        }
        let parents = [
            self.collider(a).unwrap().parent,
            self.collider(b).unwrap().parent,
        ];
        self.rebuild(|d| {
            let pair = (a.min(b), a.max(b));
            if excluded {
                d.collider_pair_exclusions.insert(pair);
            } else {
                d.collider_pair_exclusions.remove(&pair);
            }
            Ok(())
        })?;
        for body in parents.into_iter().flatten() {
            self.activate_body(body);
        }
        Ok(())
    }

    /// External force/impulse joints omitted from the reduced-coordinate tree.
    pub fn loop_joints(&self) -> impl Iterator<Item = (JointHandle, &JointDesc)> {
        self.description
            .loop_joints
            .iter()
            .map(|id| (*id, &self.description.joints[id]))
    }

    /// Rigid external ball constraints, including disabled joints.
    pub fn rigid_ball_joints(&self) -> impl Iterator<Item = (JointHandle, &JointDesc)> {
        self.rigid_loop_joints()
            .filter(|(_, joint)| joint.kind == JointType::Ball)
    }

    /// Rigid external nonradial constraints, including disabled joints.
    pub fn rigid_loop_joints(&self) -> impl Iterator<Item = (JointHandle, &JointDesc)> {
        self.loop_joints().filter(|(id, joint)| {
            joint.kind != JointType::Distance && !self.description.joint_softness.contains_key(id)
        })
    }

    /// Radial links, including disabled links, outside the reduced-coordinate tree.
    pub fn distance_joints(
        &self,
    ) -> impl Iterator<Item = (JointHandle, &JointDesc, DistanceLimits)> {
        self.description
            .distance_limits
            .iter()
            .map(|(&id, &limits)| (id, &self.description.joints[&id], limits))
    }

    pub fn distance_limits(&self, id: JointHandle) -> Result<DistanceLimits> {
        self.description
            .distance_limits
            .get(&id)
            .copied()
            .ok_or_else(|| Error::Build("distance limits require a distance joint".into()))
    }

    /// Replace radial bounds transactionally and activate their endpoint dependencies.
    pub fn set_distance_limits(&mut self, id: JointHandle, limits: DistanceLimits) -> Result<()> {
        limits.validate()?;
        if self.distance_limits(id)? == limits {
            return Ok(());
        }
        let joint = self.joint(id).ok_or_else(|| missing("joint"))?;
        let endpoints = [joint.parent, Some(joint.child)];
        self.rebuild(|d| {
            d.distance_limits.insert(id, limits);
            Ok(())
        })?;
        for body in endpoints.into_iter().flatten() {
            self.activate_body(body);
        }
        Ok(())
    }

    /// Explicit cone configurations on tree and loop ball joints, including disabled joints.
    pub fn cone_joints(
        &self,
    ) -> impl Iterator<Item = (JointHandle, &JointDesc, SphericalConeLimits)> {
        self.description
            .cone_limits
            .iter()
            .map(|(&id, &limits)| (id, &self.description.joints[&id], limits))
    }

    pub fn cone_limits(&self, id: JointHandle) -> Result<SphericalConeLimits> {
        if self.joint(id).ok_or_else(|| missing("joint"))?.kind != JointType::Ball {
            return Err(Error::Build("cone limits require a ball joint".into()));
        }
        Ok(self
            .description
            .cone_limits
            .get(&id)
            .copied()
            .unwrap_or_default())
    }

    /// Store cone bounds transactionally and activate their endpoint dependencies.
    pub fn set_cone_limits(&mut self, id: JointHandle, limits: SphericalConeLimits) -> Result<()> {
        limits.validate()?;
        if self.cone_limits(id)? == limits {
            return Ok(());
        }
        let joint = self.joint(id).ok_or_else(|| missing("joint"))?;
        if !limits.is_unrestricted() {
            crate::spherical_cone::basis(joint.axis)?;
        }
        let endpoints = [joint.parent, Some(joint.child)];
        self.rebuild(|d| {
            if limits == SphericalConeLimits::default() {
                d.cone_limits.remove(&id);
            } else {
                d.cone_limits.insert(id, limits);
            }
            Ok(())
        })?;
        for body in endpoints.into_iter().flatten() {
            self.activate_body(body);
        }
        Ok(())
    }

    pub fn joint_settings(&self, id: JointHandle) -> Option<JointSettings> {
        self.description.joint_settings.get(&id).copied()
    }

    pub fn joint_break_limits(&self, id: JointHandle) -> Result<JointBreakLimits> {
        self.joint(id).ok_or_else(|| missing("joint"))?;
        Ok(self
            .description
            .break_limits
            .get(&id)
            .copied()
            .unwrap_or_default())
    }

    pub fn breakable_joints(&self) -> impl Iterator<Item = (JointHandle, JointBreakLimits)> {
        self.description
            .break_limits
            .iter()
            .filter_map(|(&id, &limits)| {
                let joint = &self.description.joints[&id];
                (limits.is_enabled()
                    && self.description.joint_settings[&id].enabled
                    && self.description.bodies[&joint.child].enabled
                    && joint
                        .parent
                        .is_none_or(|p| self.description.bodies[&p].enabled))
                .then_some((id, limits))
            })
    }

    /// Replace thresholds without rebuilding topology; activate endpoint dependencies.
    pub fn set_joint_break_limits(
        &mut self,
        id: JointHandle,
        limits: JointBreakLimits,
    ) -> Result<()> {
        limits.validate()?;
        if self.joint_break_limits(id)? == limits {
            return Ok(());
        }
        let joint = self.joint(id).unwrap();
        let endpoints = [joint.parent, Some(joint.child)];
        if limits == JointBreakLimits::default() {
            self.description.break_limits.remove(&id);
        } else {
            self.description.break_limits.insert(id, limits);
        }
        self.touch_dynamics();
        for body in endpoints.into_iter().flatten() {
            self.activate_body(body);
        }
        Ok(())
    }

    pub fn joint_broken(&self, id: JointHandle) -> Result<bool> {
        self.joint(id).ok_or_else(|| missing("joint"))?;
        Ok(self.description.broken_joints.contains(&id))
    }

    /// Detach a batch atomically, preserving handles, body poses and velocities.
    pub fn break_joints(&mut self, joints: &[JointHandle]) -> Result<Vec<JointHandle>> {
        let mut changed = BTreeSet::new();
        for &id in joints {
            if !self.joint_broken(id)? {
                changed.insert(id);
            }
        }
        if changed.is_empty() {
            return Ok(Vec::new());
        }
        let endpoints: BTreeSet<_> = changed
            .iter()
            .flat_map(|id| {
                let joint = &self.description.joints[id];
                [joint.parent, Some(joint.child)].into_iter().flatten()
            })
            .collect();
        self.rebuild(|d| {
            for &id in &changed {
                d.joint_settings.get_mut(&id).unwrap().enabled = false;
                d.broken_joints.insert(id);
            }
            Ok(())
        })?;
        for body in endpoints {
            self.activate_body(body);
        }
        Ok(changed.into_iter().collect())
    }

    /// Explicitly clear broken state and re-enable the authored joint.
    pub fn repair_joint(&mut self, id: JointHandle) -> Result<()> {
        let mut settings = self.joint_settings(id).ok_or_else(|| missing("joint"))?;
        settings.enabled = true;
        self.set_joint_settings(id, settings)
    }

    pub fn set_joint_settings(&mut self, id: JointHandle, settings: JointSettings) -> Result<()> {
        self.edit(|d| {
            *d.joint_settings
                .get_mut(&id)
                .ok_or_else(|| missing("joint"))? = settings;
            if settings.enabled {
                d.broken_joints.remove(&id);
            }
            Ok(())
        })
    }

    pub fn joint_softness(&self, id: JointHandle) -> Result<Option<JointSoftness>> {
        self.joint(id).ok_or_else(|| missing("joint"))?;
        Ok(self.description.joint_softness.get(&id).copied())
    }

    /// Gain edits retain the compiled layout and activate endpoints; toggles rebuild it.
    pub fn set_joint_softness(
        &mut self,
        id: JointHandle,
        softness: Option<JointSoftness>,
    ) -> Result<()> {
        let previous = self.joint_softness(id)?;
        if self.joint(id).unwrap().kind == JointType::Distance {
            return if softness.is_none() {
                Ok(())
            } else {
                Err(Error::Build(
                    "distance joints do not support scalar compliance".into(),
                ))
            };
        }
        if softness.is_none()
            && self.description.loop_joints.contains(&id)
            && !matches!(
                self.joint(id).unwrap().kind,
                JointType::Ball
                    | JointType::Fixed
                    | JointType::Revolute
                    | JointType::Prismatic
                    | JointType::D6
            )
        {
            return Err(Error::Build("loop joints require compliance".into()));
        }
        if let Some(value) = softness {
            value.validate()?;
        }
        if previous == softness {
            return Ok(());
        }
        if let (Some(_), Some(value)) = (previous, softness) {
            let joint = self.joint(id).unwrap();
            let endpoints = [joint.parent, Some(joint.child)];
            if let Some(joint) = self.joint_index(id) {
                let offsets = self.model().joint_dof_offset.host()?;
                let range = offsets[joint.index()] as usize..offsets[joint.index() + 1] as usize;
                for record in &mut self.snapshot.model.joint_compliance.host_mut()?[range] {
                    record.natural_frequency = value.natural_frequency;
                    record.damping_ratio = value.damping_ratio;
                }
            }
            self.description.joint_softness.insert(id, value);
            self.touch_dynamics();
            for body in endpoints.into_iter().flatten() {
                self.activate_body(body);
            }
            return Ok(());
        }
        self.edit(|d| {
            if let Some(value) = softness {
                d.joint_softness.insert(id, value);
            } else {
                d.joint_softness.remove(&id);
            }
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
        if kind == JointType::Distance {
            return Err(Error::Build(
                "distance joints use radial limits, not axis locks".into(),
            ));
        }
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

    /// Cap a tree rate or external-loop coordinate rate in m/s or rad/s; None removes it.
    /// External angular rates use the rotation-log coordinates shared with drives.
    pub fn set_joint_velocity_limit(
        &mut self,
        id: JointHandle,
        axis: JointAxis,
        limit: Option<Real>,
    ) -> Result<()> {
        let kind = self.joint(id).ok_or_else(|| missing("joint"))?.kind;
        axis.dof(kind)?;
        if self.description.loop_joints.contains(&id)
            && !matches!(
                kind,
                JointType::Revolute | JointType::Prismatic | JointType::Ball | JointType::D6
            )
        {
            return Err(Error::Build(
                "this external-loop speed cap is not supported".into(),
            ));
        }
        if limit.is_some_and(|value| !value.is_finite() || value < 0.0) {
            return Err(Error::Build("invalid joint velocity limit".into()));
        }
        if self.description.joint_axes[&id][axis as usize].velocity_limit == limit {
            return Ok(());
        }
        self.edit(|d| {
            d.joint_axes.get_mut(&id).ok_or_else(|| missing("joint"))?[axis as usize]
                .velocity_limit = limit;
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
        self.set_joint_motor_with_wake(id, axis, motor, true)
    }

    /// Update motor controls, optionally activating endpoints when their input changes.
    pub fn set_joint_motor_with_wake(
        &mut self,
        id: JointHandle,
        axis: JointAxis,
        motor: JointMotor,
        wake: bool,
    ) -> Result<()> {
        motor.validate()?;
        axis.dof(self.joint(id).ok_or_else(|| missing("joint"))?.kind)?;
        if motor.mode != crate::JointTargetMode::None
            && self.rigid_loop_joints().any(|(joint, _)| joint == id)
            && (!matches!(
                self.joint(id).unwrap().kind,
                JointType::Revolute | JointType::Prismatic | JointType::D6 | JointType::Ball
            ))
        {
            return Err(Error::Build(
                "this rigid-loop motor kind is not supported".into(),
            ));
        }
        let dof = self.snapshot.joint_layouts[&id][axis as usize].expect("validated joint axis");
        let mut changed = self.description.joint_axes[&id][axis as usize].motor != Some(motor);
        let ends = self
            .joint(id)
            .map(|joint| [joint.parent, Some(joint.child)])
            .unwrap();
        if let Some(joint) = self.joint_index(id) {
            let offset = self.model().joint_dof_offset.host()?[joint.index()] as usize;
            let i = offset + dof;
            let values = |control: &Control| -> Result<_> {
                Ok((
                    control.joint_target_mode.host()?[i],
                    control.joint_motor_model.host()?[i],
                    control.joint_target_pos.host()?[i],
                    control.joint_target_vel.host()?[i],
                    control.joint_target_kp.host()?[i],
                    control.joint_target_kd.host()?[i],
                    control.joint_motor_max_force.host()?[i],
                ))
            };
            let before = values(&self.snapshot.control)?;
            motor.apply(&mut self.snapshot.control, i)?;
            changed |= before != values(&self.snapshot.control)?;
        }
        self.description
            .joint_axes
            .get_mut(&id)
            .ok_or_else(|| missing("joint"))?[axis as usize]
            .motor = Some(motor);
        if changed {
            self.touch_dynamics();
        }
        if changed && wake {
            for body in ends.into_iter().flatten() {
                self.activate_body(body);
            }
        }
        Ok(())
    }

    /// Set a joint axis's brake damping in place, without a rebuild.
    pub fn set_joint_brake(
        &mut self,
        id: JointHandle,
        axis: JointAxis,
        damping: Real,
    ) -> Result<()> {
        if !damping.is_finite() || damping < 0.0 {
            return Err(Error::Build(
                "joint brake damping must be finite and nonnegative".into(),
            ));
        }
        axis.dof(self.joint(id).ok_or_else(|| missing("joint"))?.kind)?;
        if damping != 0.0
            && self.rigid_loop_joints().any(|(joint, _)| joint == id)
            && !matches!(
                self.joint(id).unwrap().kind,
                JointType::Revolute | JointType::Prismatic | JointType::D6 | JointType::Ball
            )
        {
            return Err(Error::Build(
                "rigid loops do not yet support axis brakes".into(),
            ));
        }
        let settings = &mut self
            .description
            .joint_axes
            .get_mut(&id)
            .ok_or_else(|| missing("joint"))?[axis as usize];
        if settings.brake == damping {
            return Ok(());
        }
        settings.brake = damping;
        self.touch_dynamics();
        if let (Some(joint), Some(dof)) = (
            self.joint_index(id),
            self.snapshot.joint_layouts[&id][axis as usize],
        ) {
            let offset = self.model().joint_dof_offset.host()?[joint.index()] as usize;
            self.snapshot.control.joint_damping.host_mut()?[offset + dof] = damping;
        }
        let ends = self
            .joint(id)
            .map(|j| [j.parent, Some(j.child)])
            .unwrap_or_default();
        for body in ends.into_iter().flatten() {
            self.activate_body(body);
        }
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

    pub fn insert_body(&mut self, desc: BodyDesc) -> Result<BodyHandle> {
        self.edit(|d| d.insert_body(desc))
    }

    /// Remove a body, its colliders and incident joints; surviving children become roots.
    pub fn remove_body(&mut self, id: BodyHandle) -> Result<()> {
        self.edit(|d| {
            d.bodies.remove(&id).ok_or_else(|| missing("body"))?;
            d.body_identities.remove(&id);
            d.kinematic_acceleration.remove(&id);
            d.colliders.retain(|_, c| c.parent != Some(id));
            d.collider_stamps
                .retain(|id, _| d.colliders.contains_key(id));
            d.joints
                .retain(|_, j| j.parent != Some(id) && j.child != id);
            d.joint_identities.retain(|id, _| d.joints.contains_key(id));
            d.joint_settings.retain(|id, _| d.joints.contains_key(id));
            d.joint_axes.retain(|id, _| d.joints.contains_key(id));
            d.joint_softness.retain(|id, _| d.joints.contains_key(id));
            d.loop_joints.retain(|id| d.joints.contains_key(id));
            d.distance_limits.retain(|id, _| d.joints.contains_key(id));
            d.cone_limits.retain(|id, _| d.joints.contains_key(id));
            d.break_limits.retain(|id, _| d.joints.contains_key(id));
            d.broken_joints.retain(|id| d.joints.contains_key(id));
            d.body_pair_exclusions
                .retain(|(a, b)| d.bodies.contains_key(a) && d.bodies.contains_key(b));
            d.collider_pair_exclusions
                .retain(|(a, b)| d.colliders.contains_key(a) && d.colliders.contains_key(b));
            Ok(())
        })
    }

    pub fn set_body_kind(&mut self, id: BodyHandle, kind: BodyKind) -> Result<()> {
        self.set_body_kind_with_wake(id, kind, true)
    }

    /// Change mobility, optionally activating a non-fixed body's dependents.
    pub fn set_body_kind_with_wake(
        &mut self,
        id: BodyHandle,
        kind: BodyKind,
        wake: bool,
    ) -> Result<()> {
        if self.body(id).ok_or_else(|| missing("body"))?.kind == kind {
            return Ok(());
        }
        self.rebuild(|d| {
            d.bodies.get_mut(&id).ok_or_else(|| missing("body"))?.kind = kind;
            if kind != BodyKind::Kinematic {
                d.kinematic_acceleration.remove(&id);
            }
            Ok(())
        })?;
        if wake && kind != BodyKind::Fixed {
            self.activate_body(id);
        }
        Ok(())
    }

    /// Constant world COM/angular acceleration; None holds externally authored motion.
    pub fn kinematic_acceleration(&self, id: BodyHandle) -> Result<Option<SpatialVector>> {
        self.body(id).ok_or_else(|| missing("body"))?;
        Ok(self.description.kinematic_acceleration.get(&id).copied())
    }

    /// Enable acceleration-driven motion; Some(ZERO) advances at constant velocity.
    /// Disabled bodies retain this setting without advancing. Changing mobility clears it.
    /// Consumed by the CPU and GPU Featherstone runtime.
    pub fn set_kinematic_acceleration(
        &mut self,
        id: BodyHandle,
        acceleration: Option<SpatialVector>,
    ) -> Result<()> {
        let body = self.body(id).ok_or_else(|| missing("body"))?;
        if acceleration.is_some_and(|a| {
            body.kind != BodyKind::Kinematic || !a.linear.is_finite() || !a.angular.is_finite()
        }) {
            return Err(Error::Build(
                "invalid prescribed kinematic acceleration".into(),
            ));
        }
        if self.kinematic_acceleration(id)? == acceleration {
            return Ok(());
        }
        self.rebuild(|d| {
            if let Some(acceleration) = acceleration {
                d.kinematic_acceleration.insert(id, acceleration);
            } else {
                d.kinematic_acceleration.remove(&id);
            }
            Ok(())
        })?;
        self.activate_body(id);
        Ok(())
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
        self.set_body_mass_properties_with_wake(id, mass, true)
    }

    /// Replace total mass, optionally activating the edited body's dependents.
    pub fn set_body_mass_properties_with_wake(
        &mut self,
        id: BodyHandle,
        mass: MassProperties,
        wake: bool,
    ) -> Result<()> {
        mass.validate()?;
        let body = self.body(id).ok_or_else(|| missing("body"))?;
        if body.params.mass == mass.mass
            && body.params.com == mass.local_com
            && body.params.inertia == mass.inertia
        {
            return Ok(());
        }
        self.rebuild(|d| {
            let params = &mut d.bodies.get_mut(&id).ok_or_else(|| missing("body"))?.params;
            params.mass = mass.mass;
            params.com = mass.local_com;
            params.inertia = mass.inertia;
            Ok(())
        })?;
        if wake {
            self.activate_body(id);
        }
        Ok(())
    }

    /// Commit a recomputed total and its independently replaceable additional component.
    pub fn set_body_mass_components(
        &mut self,
        id: BodyHandle,
        total: MassProperties,
        additional: MassProperties,
    ) -> Result<()> {
        self.set_body_mass_components_with_wake(id, total, additional, true)
    }

    /// Replace total and additional mass atomically with optional body activation.
    pub fn set_body_mass_components_with_wake(
        &mut self,
        id: BodyHandle,
        total: MassProperties,
        additional: MassProperties,
        wake: bool,
    ) -> Result<()> {
        total.validate()?;
        additional.validate()?;
        let body = self.body(id).ok_or_else(|| missing("body"))?;
        if body.params.mass == total.mass
            && body.params.com == total.local_com
            && body.params.inertia == total.inertia
            && body.additional_mass == Some(additional)
        {
            return Ok(());
        }
        self.rebuild(|d| {
            let body = d.bodies.get_mut(&id).ok_or_else(|| missing("body"))?;
            body.params.mass = total.mass;
            body.params.com = total.local_com;
            body.params.inertia = total.inertia;
            body.additional_mass = Some(additional);
            Ok(())
        })?;
        if wake {
            self.activate_body(id);
        }
        Ok(())
    }

    pub fn insert_collider(&mut self, desc: ColliderDesc) -> Result<ColliderHandle> {
        self.edit(|d| d.insert_collider(desc))
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
            d.collider_pair_exclusions
                .retain(|(a, b)| *a != id && *b != id);
            Ok(())
        })
    }

    /// Removes `removes` and inserts `inserts` in one rebuild of the model;
    /// the new handles in the order given. Nothing changes when any fails.
    pub fn edit_colliders(
        &mut self,
        inserts: Vec<ColliderDesc>,
        removes: &[ColliderHandle],
    ) -> Result<Vec<ColliderHandle>> {
        let gone: BTreeSet<ColliderHandle> = removes.iter().copied().collect();
        self.edit(|d| {
            for id in &gone {
                d.colliders.remove(id).ok_or_else(|| missing("collider"))?;
                d.collider_stamps.remove(id);
            }
            d.collider_pair_exclusions
                .retain(|(a, b)| !gone.contains(a) && !gone.contains(b));
            inserts
                .into_iter()
                .map(|desc| d.insert_collider(desc))
                .collect()
        })
    }

    pub fn insert_joint(&mut self, desc: JointDesc) -> Result<JointHandle> {
        self.edit(|d| d.insert_joint(desc))
    }

    /// Insert a compliant force joint without merging or projecting either tree.
    pub fn insert_loop_joint(
        &mut self,
        desc: JointDesc,
        softness: JointSoftness,
    ) -> Result<JointHandle> {
        if desc.kind == JointType::Distance {
            return Err(Error::Build(
                "insert a rigid distance joint and set its radial limits".into(),
            ));
        }
        softness.validate()?;
        self.edit(|d| {
            let id = JointHandle(d.allocate()?);
            d.joints.insert(id, desc);
            d.joint_identities.insert(id, SceneStamp(Arc::new(())));
            d.joint_settings.insert(id, JointSettings::default());
            d.joint_axes.insert(id, [JointAxisSettings::default(); 6]);
            d.joint_softness.insert(id, softness);
            d.loop_joints.insert(id);
            Ok(id)
        })
    }

    /// Insert a rigid axis or ball constraint without merging the endpoint trees.
    pub fn insert_rigid_loop_joint(&mut self, desc: JointDesc) -> Result<JointHandle> {
        if !matches!(
            desc.kind,
            JointType::Ball
                | JointType::Fixed
                | JointType::Revolute
                | JointType::Prismatic
                | JointType::D6
        ) {
            return Err(Error::Build(
                "rigid loop insertion requires a fixed, revolute, prismatic, ball or D6 joint"
                    .into(),
            ));
        }
        self.edit(|d| {
            let id = JointHandle(d.allocate()?);
            d.joints.insert(id, desc);
            d.joint_identities.insert(id, SceneStamp(Arc::new(())));
            d.joint_settings.insert(id, JointSettings::default());
            d.joint_axes.insert(id, [JointAxisSettings::default(); 6]);
            d.loop_joints.insert(id);
            Ok(id)
        })
    }

    pub fn remove_joint(&mut self, id: JointHandle) -> Result<()> {
        self.edit(|d| {
            d.joints.remove(&id).ok_or_else(|| missing("joint"))?;
            d.joint_identities.remove(&id);
            d.joint_settings.remove(&id);
            d.joint_axes.remove(&id);
            d.joint_softness.remove(&id);
            d.loop_joints.remove(&id);
            d.distance_limits.remove(&id);
            d.cone_limits.remove(&id);
            d.break_limits.remove(&id);
            d.broken_joints.remove(&id);
            Ok(())
        })
    }

    pub fn set_gravity(&mut self, gravity: Vec3) -> Result<()> {
        if !gravity.is_finite() {
            return Err(Error::Build("non-finite gravity".into()));
        }
        if self.gravity != gravity {
            self.activate_all();
        }
        self.snapshot.model.gravity.host_mut()?[0] = gravity;
        self.gravity = gravity;
        Ok(())
    }

    fn edit<T>(&mut self, change: impl FnOnce(&mut Description) -> Result<T>) -> Result<T> {
        let result = self.rebuild(change)?;
        self.activate_all();
        Ok(result)
    }

    fn rebuild<T>(&mut self, change: impl FnOnce(&mut Description) -> Result<T>) -> Result<T> {
        let mut next = self.description.clone();
        let result = change(&mut next)?;
        let snapshot = compile(&next, self.gravity, Some(&self.snapshot))?;
        readout::update_structure_stamps::<false, false>(
            &mut self.structure_stamps,
            &self.description,
            &next,
        );
        readout::update_structure_stamps::<true, false>(
            &mut self.contact_stamps,
            &self.description,
            &next,
        );
        readout::update_structure_stamps::<true, true>(
            &mut self.contact_episode_stamps,
            &self.description,
            &next,
        );
        self.body_state_stamps
            .retain(|body, _| next.bodies.contains_key(body));
        self.description = next;
        self.snapshot = snapshot;
        self.touch_dynamics();
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

fn joint_layout(kind: JointType, soft: bool) -> [Option<usize>; 6] {
    JointAxis::ALL.map(|axis| {
        axis.dof(kind)
            .ok()
            .map(|dof| if soft { axis as usize } else { dof })
    })
}

fn drive_layout(
    owner: DriveOwner,
    layouts: &BTreeMap<JointHandle, [Option<usize>; 6]>,
    model: &Model,
    joint: JointId,
) -> Result<[Option<usize>; 6]> {
    Ok(match owner {
        DriveOwner::Joint(id) => layouts[&id],
        DriveOwner::Body(_) => {
            if model.joint_type.host()?[joint.index()] == JointType::Free as u32 {
                joint_layout(JointType::Free, false)
            } else {
                [None; 6]
            }
        }
    })
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
        let body = builder.add_body(params);
        if desc.enabled {
            builder.set_kinematic_acceleration(body, d.kinematic_acceleration.get(&id).copied())?;
        }
        bodies.insert(id, body);
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
            builder
                .set_shape_surface_velocity(shape, desc.pose.rotation * desc.surface_velocity)?;
            builder.set_shape_collision_groups(shape, desc.memberships, desc.filter)?;
            builder.set_shape_flags(shape, flags)?;
            shape_colliders.push(id);
        }
        colliders.insert(id, shapes);
    }
    let (joints, roots) = tree::compile_joints(d, &bodies, &poses, &mut builder)?;
    let joint_layouts = d
        .joints
        .iter()
        .map(|(&id, joint)| {
            (
                id,
                joint_layout(joint.kind, d.joint_softness.contains_key(&id)),
            )
        })
        .collect();
    for (&id, joint) in &d.joints {
        if !d.joint_settings[&id].contacts_enabled
            && !d.broken_joints.contains(&id)
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
    for &(a, b) in &d.body_pair_exclusions {
        let shapes_for = |body| {
            d.colliders
                .iter()
                .filter(move |(_, c)| c.parent == Some(body))
                .flat_map(|(id, _)| colliders[id].iter().copied())
        };
        for a in shapes_for(a) {
            for b in shapes_for(b) {
                builder.add_shape_collision_exclusion(a, b);
            }
        }
    }
    for &(a, b) in &d.collider_pair_exclusions {
        for &a in &colliders[&a] {
            for &b in &colliders[&b] {
                builder.add_shape_collision_exclusion(a, b);
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
                if s.model.joint_x_c.host()?[previous.index()]
                    != model.joint_x_c.host()?[joint.index()]
                {
                    seed_joint(&model, &mut state, joint)?;
                }
            } else {
                seed_joint(&model, &mut state, joint)?;
            }
        }
    }
    for (&id, &joint) in &joints {
        if let Some((s, &previous)) = old.and_then(|s| s.joints.get(&id).map(|j| (s, j)))
            && !d.joint_softness.contains_key(&id)
            && !s.soft_joints.contains(&id)
        {
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
            let drive = DriveState::capture(
                old,
                joint,
                drive_layout(owner, &old.joint_layouts, &old.model, joint)?,
            )?;
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
            drive.restore(
                &model,
                &mut control,
                joint,
                drive_layout(owner, &joint_layouts, &model, joint)?,
            )?;
        }
    }
    for (&id, joint) in &joints {
        let offset = model.joint_dof_offset.host()?[joint.index()] as usize;
        for axis in JointAxis::ALL {
            let Some(dof) = joint_layouts[&id][axis as usize] else {
                continue;
            };
            let settings = d.joint_axes[&id][axis as usize];
            if let Some(motor) = settings.motor {
                motor.apply(&mut control, offset + dof)?;
            }
            control.joint_damping.host_mut()?[offset + dof] = settings.brake;
        }
    }
    let mut body_handles = vec![BodyHandle(0); model.body_count];
    for (&handle, &index) in &bodies {
        body_handles[index.index()] = handle;
    }
    Ok(Snapshot {
        model_stamp: SceneStamp(Arc::new(())),
        stamp: SceneStamp(Arc::new(())),
        model,
        state,
        control,
        bodies,
        body_handles,
        colliders,
        shape_colliders,
        joints,
        joint_layouts,
        soft_joints: d.joint_softness.keys().copied().collect(),
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

fn seed_joint_near(
    model: &Model,
    state: &mut State,
    joint: JointId,
    previous: &State,
) -> Result<()> {
    seed_joint(model, state, joint)?;
    let j = joint.index();
    if model.joint_type.host()?[j] == JointType::Revolute as u32 {
        let c = model.joint_coord_offset.host()?[j] as usize;
        let old = previous.joint_q.host()?[c];
        let q = &mut state.joint_q.host_mut()?[c];
        *q += ((old - *q) / std::f64::consts::TAU).round() * std::f64::consts::TAU;
    }
    Ok(())
}

fn seed_joint(model: &Model, state: &mut State, joint: JointId) -> Result<()> {
    let j = joint.index();
    let mut parent = model.joint_parent.host()?[j];
    let mut child = model.joint_child.host()?[j] as usize;
    let mut xp = model.joint_x_p.host()?[j];
    let mut xc = model.joint_x_c.host()?[j];
    if model.joint_reversed.host()?[j] != 0 {
        let previous = child;
        child = parent as usize;
        parent = previous as i32;
        std::mem::swap(&mut xp, &mut xc);
    }
    let poses = state.body_q.host()?;
    let velocities = state.body_qd.host()?;
    let child_frame = poses[child].compose(&xc);
    let parent_frame = if parent < 0 {
        xp
    } else {
        poses[parent as usize].compose(&xp)
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
            // The angle reads from the quaternion's w >= 0 half, in [-pi, pi].
            let r = if relative.rotation.w < 0.0 { -relative.rotation } else { relative.rotation };
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
        joint_force,
        joint_damping
    );
    Ok(())
}
