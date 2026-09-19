//! Transactional rigid-scene mutation with persistent handles and dense solver snapshots.

use std::collections::{BTreeMap, BTreeSet};

use molla_core::{BodyId, Error, JointId, Result, ShapeId, WorldId};
use molla_math::{Real, SpatialVector, Transform, Vec3};

use crate::{
    BodyFlags, BodyParams, CoefficientCombineRule, ContactMaterial, Control, Heightfield,
    JointType, Model, ModelBuilder, ShapeFlags, State, TriMesh,
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
    pub velocity: SpatialVector,
    pub user_tag: u64,
}

impl Default for BodyDesc {
    fn default() -> Self {
        Self {
            params: BodyParams::new(),
            kind: BodyKind::Dynamic,
            velocity: SpatialVector::ZERO,
            user_tag: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub enum ColliderGeometry {
    Ball { radius: Real },
    Cuboid { half_extents: Vec3 },
    Capsule { half_height: Real, radius: Real },
    Cylinder { half_height: Real, radius: Real },
    ConvexHull(Vec<Vec3>),
    TriMesh(TriMesh),
    Heightfield(Heightfield),
}

#[derive(Clone, Debug)]
pub struct ColliderDesc {
    pub parent: Option<BodyHandle>,
    pub geometry: ColliderGeometry,
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

#[derive(Clone, Default)]
struct Description {
    next_id: u64,
    bodies: BTreeMap<BodyHandle, BodyDesc>,
    colliders: BTreeMap<ColliderHandle, ColliderDesc>,
    joints: BTreeMap<JointHandle, JointDesc>,
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
    model: Model,
    state: State,
    control: Control,
    bodies: BTreeMap<BodyHandle, BodyId>,
    colliders: BTreeMap<ColliderHandle, ShapeId>,
    joints: BTreeMap<JointHandle, JointId>,
    roots: BTreeMap<BodyHandle, JointId>,
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
    pub fn gravity(&self) -> Vec3 {
        self.gravity
    }

    pub fn body_index(&self, id: BodyHandle) -> Option<BodyId> {
        self.snapshot.bodies.get(&id).copied()
    }

    pub fn collider_index(&self, id: ColliderHandle) -> Option<ShapeId> {
        self.snapshot.colliders.get(&id).copied()
    }

    pub fn body_handle(&self, index: BodyId) -> Option<BodyHandle> {
        self.snapshot
            .bodies
            .iter()
            .find_map(|(&id, &i)| (i == index).then_some(id))
    }

    pub fn collider_handle(&self, index: ShapeId) -> Option<ColliderHandle> {
        self.snapshot
            .colliders
            .iter()
            .find_map(|(&id, &i)| (i == index).then_some(id))
    }

    pub fn joint_index(&self, id: JointHandle) -> Option<JointId> {
        self.snapshot.joints.get(&id).copied()
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
            d.joints
                .retain(|_, j| j.parent != Some(id) && j.child != id);
            Ok(())
        })
    }

    pub fn set_body_kind(&mut self, id: BodyHandle, kind: BodyKind) -> Result<()> {
        self.edit(|d| {
            d.bodies.get_mut(&id).ok_or_else(|| missing("body"))?.kind = kind;
            Ok(())
        })
    }

    pub fn insert_collider(&mut self, desc: ColliderDesc) -> Result<ColliderHandle> {
        self.edit(|d| {
            let id = ColliderHandle(d.allocate()?);
            d.colliders.insert(id, desc);
            Ok(id)
        })
    }

    pub fn replace_collider(&mut self, id: ColliderHandle, desc: ColliderDesc) -> Result<()> {
        self.edit(|d| {
            *d.colliders
                .get_mut(&id)
                .ok_or_else(|| missing("collider"))? = desc;
            Ok(())
        })
    }

    pub fn remove_collider(&mut self, id: ColliderHandle) -> Result<()> {
        self.edit(|d| {
            d.colliders.remove(&id).ok_or_else(|| missing("collider"))?;
            Ok(())
        })
    }

    pub fn insert_joint(&mut self, desc: JointDesc) -> Result<JointHandle> {
        self.edit(|d| {
            let id = JointHandle(d.allocate()?);
            d.joints.insert(id, desc);
            Ok(id)
        })
    }

    pub fn remove_joint(&mut self, id: JointHandle) -> Result<()> {
        self.edit(|d| {
            d.joints.remove(&id).ok_or_else(|| missing("joint"))?;
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

fn compile(d: &Description, gravity: Vec3, old: Option<&Snapshot>) -> Result<Snapshot> {
    let mut builder = ModelBuilder::new();
    builder.set_world_gravity(WorldId(0), gravity);
    let mut bodies = BTreeMap::new();
    for (&id, desc) in &d.bodies {
        validate_body(desc)?;
        let mut params = desc.params.clone();
        if let Some(previous) = old.and_then(|s| s.bodies.get(&id).map(|i| (s, i))) {
            params.pose = previous.0.state.body_q.host()?[previous.1.index()];
        }
        params.flags = match desc.kind {
            BodyKind::Dynamic => BodyFlags::DYNAMIC,
            BodyKind::Kinematic => BodyFlags::KINEMATIC,
            BodyKind::Fixed => BodyFlags::FIXED,
        };
        bodies.insert(id, builder.add_body(params));
    }
    let mut colliders = BTreeMap::new();
    for (&id, desc) in &d.colliders {
        validate_pose(desc.pose)?;
        let parent = dense_body(&bodies, desc.parent)?;
        let shape = add_shape(&mut builder, parent, desc)?;
        builder.set_shape_material(shape, desc.friction, desc.restitution)?;
        if desc.friction < 0.0 || !(0.0..=1.0).contains(&desc.restitution) {
            return Err(Error::Build(
                "invalid runtime friction or restitution".into(),
            ));
        }
        builder.set_shape_contact_material(shape, desc.contact_material)?;
        builder.set_shape_collision_groups(shape, desc.memberships, desc.filter)?;
        let flags = if desc.enabled {
            ShapeFlags::COLLIDES
        } else {
            0
        } | if desc.sensor { ShapeFlags::SENSOR } else { 0 };
        builder.set_shape_flags(shape, flags)?;
        colliders.insert(id, shape);
    }
    let mut incoming = BTreeMap::new();
    let mut children: BTreeMap<BodyHandle, Vec<JointHandle>> = BTreeMap::new();
    for (&id, joint) in &d.joints {
        dense_body(&bodies, joint.parent)?;
        dense_body(&bodies, Some(joint.child))?;
        validate_pose(joint.frame_parent)?;
        validate_pose(joint.frame_child)?;
        if joint.parent == Some(joint.child) || incoming.insert(joint.child, id).is_some() {
            return Err(Error::Build(
                "runtime joints require a forest with one parent per body".into(),
            ));
        }
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
                joints.insert(joint, j);
            } else {
                let pose = old
                    .and_then(|s| s.bodies.get(&body).map(|i| (s, i)))
                    .map(|(s, i)| s.state.body_q.host().map(|q| q[i.index()]))
                    .transpose()?
                    .unwrap_or(desc.params.pose);
                let j = if desc.kind == BodyKind::Dynamic {
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
    let model = builder.build()?;
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
        }
        if d.bodies[&id].kind == BodyKind::Fixed {
            state.body_qd.host_mut()?[i] = SpatialVector::ZERO;
        }
    }
    for (&body, &joint) in &roots {
        if d.bodies[&body].kind == BodyKind::Dynamic {
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
    Ok(Snapshot {
        model,
        state,
        control,
        bodies,
        colliders,
        joints,
        roots,
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

fn add_shape(builder: &mut ModelBuilder, body: BodyId, desc: &ColliderDesc) -> Result<ShapeId> {
    let pose = desc.pose;
    match &desc.geometry {
        ColliderGeometry::Ball { radius } => builder.add_shape_sphere(body, *radius, pose),
        ColliderGeometry::Cuboid { half_extents } => {
            builder.add_shape_box(body, *half_extents, pose)
        }
        ColliderGeometry::Capsule {
            half_height,
            radius,
        } => builder.add_shape_capsule(body, *half_height, *radius, pose),
        ColliderGeometry::Cylinder {
            half_height,
            radius,
        } => builder.add_shape_cylinder(body, *half_height, *radius, pose),
        ColliderGeometry::ConvexHull(points) => {
            if points.iter().any(|p| !p.is_finite()) {
                return Err(Error::Build("non-finite hull".into()));
            }
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
            if field.nx < 2
                || field.nz < 2
                || field.nx as u64 * field.nz as u64 != field.heights.len() as u64
                || !field.origin.is_finite()
                || !field.cell_size.is_finite()
                || field.cell_size <= 0.0
                || field.heights.iter().any(|h| !h.is_finite())
            {
                return Err(Error::Build("invalid runtime heightfield".into()));
            }
            let field = builder.add_heightfield(field.clone());
            builder.add_shape_heightfield(body, field, pose)
        }
    }
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
        joint_force
    );
    Ok(())
}
