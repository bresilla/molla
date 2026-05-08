//! `ModelBuilder` — ergonomic authoring surface for [`Model`].
//!
//! Stage 1 surface: rigid bodies, sphere shapes, per-world gravity. More
//! arrives in stages 3 (joints) and 4 (other shapes).

use crate::core::{BodyId, Error, JointId, Result, ShapeId, WorldId};
use crate::math::{Mat3, Real, SpatialInertia, Transform, Vec3};
use crate::storage::{Buffer, BufferUsage};

use super::flags::BodyFlags;
use super::Model;

/// Per-shape geometry tag. Stage 4 ships sphere, plane, and box;
/// capsule / cylinder land in stage 6.1 (closed-form narrow phase
/// for capsule + GJK fallback for cylinder).
#[derive(Copy, Clone, Debug)]
pub enum ShapeGeometry {
    /// Solid sphere centred on the shape's local origin.
    Sphere {
        /// Radius (m).
        radius: Real,
    },
    /// Infinite half-space plane in the shape's local frame.
    /// Points satisfying `normal · x <= 0` are *inside* the half-space
    /// (i.e. penetrating). The plane is anchored at the shape's local
    /// origin; use the shape's local `Transform` to position it.
    Plane {
        /// Outward-facing unit normal in shape-local frame.
        normal: Vec3,
    },
    /// Solid axis-aligned box in the shape's local frame, centred on
    /// the shape's local origin. World orientation comes from the
    /// shape's transform — i.e. it's an OBB once you compose with
    /// the body pose.
    Box {
        /// Half-extents along the shape-local x/y/z axes (m).
        half_extents: Vec3,
    },
    /// Capsule (cylinder + two hemispherical caps) along the
    /// shape-local **Y axis** — matches URDF/MJCF convention.
    /// Total length = `2·(half_height + radius)`.
    Capsule {
        /// Half-length of the cylindrical section (excluding caps).
        half_height: Real,
        /// Radius of the cylindrical section + caps.
        radius: Real,
    },
    /// Cylinder along the shape-local **Y axis**. Flat caps (no
    /// hemisphere). Mostly handled via GJK + EPA in stage 6.x.
    Cylinder {
        /// Half-length of the cylindrical section.
        half_height: Real,
        /// Radius.
        radius: Real,
    },
    /// Triangle in the shape's local frame. Three vertex positions.
    /// Used as a primitive for narrow-phase queries against
    /// individual mesh triangles; the user typically builds meshes
    /// via [`super::Model::meshes`] and references them with
    /// [`Self::TriMesh`]. Volume-degenerate — narrow phase routes
    /// triangle-vs-volume pairs through GJK+EPA.
    Triangle {
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
    },
    /// Triangle mesh — references entry `mesh_id` in
    /// [`super::Model::meshes`]. Used for static terrain and
    /// convex-hull approximations of arbitrary meshes.
    TriMesh {
        /// Index into [`super::Model::meshes`].
        mesh_id: u32,
    },
}

/// Parameters for [`ModelBuilder::add_body`].
#[derive(Clone, Debug)]
pub struct BodyParams {
    /// Mass in kg. Set to 0 for static / kinematic bodies.
    pub mass: Real,
    /// Inertia tensor in body frame at COM (kg·m²). Use [`Mat3::ZERO`]
    /// to skip angular dynamics for that body.
    pub inertia: Mat3,
    /// Centre of mass in body frame (m).
    pub com: Vec3,
    /// Initial pose in world frame.
    pub pose: Transform,
    /// Body flags (see [`super::BodyFlags`]).
    pub flags: u32,
    /// World grouping index.
    pub world: WorldId,
}

impl BodyParams {
    /// A dynamic, unit-mass body at the origin with identity rotation.
    #[must_use]
    pub fn new() -> Self {
        Self {
            mass: 1.0,
            inertia: Mat3::IDENTITY,
            com: Vec3::ZERO,
            pose: Transform::IDENTITY,
            flags: BodyFlags::DYNAMIC,
            world: WorldId(0),
        }
    }

    /// Set total mass, leaving other fields untouched.
    #[must_use]
    pub fn with_mass(mut self, mass: Real) -> Self {
        self.mass = mass;
        self
    }

    /// Set inertia tensor.
    #[must_use]
    pub fn with_inertia(mut self, inertia: Mat3) -> Self {
        self.inertia = inertia;
        self
    }

    /// Apply a [`SpatialInertia`] (mass + inertia + COM) in one go.
    #[must_use]
    pub fn with_spatial_inertia(mut self, si: SpatialInertia) -> Self {
        self.mass = si.mass;
        self.inertia = si.inertia;
        self.com = si.com;
        self
    }

    /// Set the COM offset (body frame).
    #[must_use]
    pub fn with_com(mut self, com: Vec3) -> Self {
        self.com = com;
        self
    }

    /// Set the initial world-frame pose.
    #[must_use]
    pub fn with_pose(mut self, pose: Transform) -> Self {
        self.pose = pose;
        self
    }

    /// Set the initial position, leaving rotation as identity.
    #[must_use]
    pub fn with_position(mut self, p: Vec3) -> Self {
        self.pose = Transform::from_translation(p);
        self
    }

    /// Set the body world index.
    #[must_use]
    pub fn with_world(mut self, world: WorldId) -> Self {
        self.world = world;
        self
    }

    /// Set body flags (bitfield from [`super::BodyFlags`]).
    #[must_use]
    pub fn with_flags(mut self, flags: u32) -> Self {
        self.flags = flags;
        self
    }
}

impl Default for BodyParams {
    fn default() -> Self {
        Self::new()
    }
}

/// Authoring surface for a [`Model`]. Stores everything in `Vec`s and
/// finalises into typed [`Buffer`]s in [`ModelBuilder::build`].
#[derive(Debug, Default)]
pub struct ModelBuilder {
    // Body arrays.
    body_mass: Vec<Real>,
    body_inv_mass: Vec<Real>,
    body_inertia: Vec<Mat3>,
    body_inv_inertia: Vec<Mat3>,
    body_com: Vec<Vec3>,
    body_flags: Vec<u32>,
    body_world: Vec<i32>,
    body_q0: Vec<Transform>,

    // Shape arrays.
    shape_body: Vec<i32>,
    shape_transform: Vec<Transform>,
    shape_geometry: Vec<ShapeGeometry>,
    shape_flags: Vec<u32>,
    shape_friction: Vec<Real>,
    shape_restitution: Vec<Real>,
    meshes: Vec<super::TriMesh>,

    // Joint arrays.
    joint_type: Vec<u32>,
    joint_parent: Vec<i32>,
    joint_child: Vec<i32>,
    joint_x_p: Vec<Transform>,
    joint_x_c: Vec<Transform>,
    joint_axis: Vec<Vec3>,
    joint_coord_offset: Vec<u32>, // CSR; ends at total
    joint_dof_offset: Vec<u32>,
    /// Initial coordinates accumulated as joints are added; length
    /// equals the last entry of `joint_coord_offset`.
    joint_q0: Vec<Real>,
    // Per-DOF limit / drive tables — grow with `joint_dof_offset`.
    joint_limit_lower: Vec<Real>,
    joint_limit_upper: Vec<Real>,
    joint_friction: Vec<Real>,
    joint_effort_limit: Vec<Real>,
    joint_velocity_limit: Vec<Real>,

    // Articulation CSR — joint `i` is in articulation `a` iff
    // `articulation_start[a] <= i < articulation_start[a+1]`.
    // The vec always starts at `[0]`; a new articulation is opened
    // by `begin_articulation()` (or implicitly by calling
    // `add_joint_*` after a closed one).
    articulation_start: Vec<u32>,
    /// `true` iff the most recent push was to a still-open articulation
    /// — used to coalesce consecutive joints into the same chain.
    articulation_open: bool,

    // Per-world arrays.
    world_gravity: Vec<Vec3>,

    // Solver knobs (defaults match Newton's `SemiImplicitSolver`).
    angular_damping: Real,
    particle_velocity_max: Real,
    contact_margin: Real,
    default_friction: Real,
    default_restitution: Real,
}

impl ModelBuilder {
    /// Construct a fresh builder with one world (`WorldId(0)`) at standard
    /// gravity pointing in -Y.
    #[must_use]
    pub fn new() -> Self {
        Self {
            world_gravity: vec![Vec3::new(0.0, -crate::math::GRAVITY, 0.0)],
            angular_damping: 0.05,
            contact_margin: 0.01,
            default_friction: 0.5,
            default_restitution: 0.0,
            particle_velocity_max: 1.0e6,
            // CSR offset arrays start with a single 0 sentinel; the
            // first joint pushes the next entry.
            joint_coord_offset: vec![0],
            joint_dof_offset: vec![0],
            articulation_start: vec![0],
            articulation_open: false,
            ..Default::default()
        }
    }

    /// Close the current articulation (if any joints are open) and start
    /// a fresh one. The next `add_joint_*` will be the first joint of
    /// the new chain. Calling this twice in a row is a no-op.
    pub fn begin_articulation(&mut self) {
        if self.articulation_open {
            self.articulation_start.push(self.joint_type.len() as u32);
            self.articulation_open = false;
        }
    }

    /// Add a rigid body. Returns a stable [`BodyId`] equal to the body's
    /// index in the resulting [`Model`].
    pub fn add_body(&mut self, params: BodyParams) -> BodyId {
        // Make sure the requested world exists.
        let w = params.world.0 as usize;
        while self.world_gravity.len() <= w {
            self.world_gravity
                .push(Vec3::new(0.0, -crate::math::GRAVITY, 0.0));
        }

        let inv_mass = if params.mass > 0.0 {
            1.0 / params.mass
        } else {
            0.0
        };
        let inv_inertia = if params.mass > 0.0 && params.inertia != Mat3::ZERO {
            params.inertia.inverse()
        } else {
            Mat3::ZERO
        };

        let id = BodyId::from_index(self.body_mass.len());
        self.body_mass.push(params.mass);
        self.body_inv_mass.push(inv_mass);
        self.body_inertia.push(params.inertia);
        self.body_inv_inertia.push(inv_inertia);
        self.body_com.push(params.com);
        self.body_flags.push(params.flags);
        self.body_world.push(params.world.0 as i32);
        self.body_q0.push(params.pose);
        id
    }

    /// Attach a sphere shape to `body`. Pass [`BodyId::NONE`] to make
    /// the shape part of the static world frame instead of any body.
    pub fn add_shape_sphere(
        &mut self,
        body: BodyId,
        radius: Real,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_sphere")?;
        if !radius.is_finite() || radius <= 0.0 {
            return Err(Error::Build(format!(
                "add_shape_sphere: radius must be a positive finite number, got {radius}"
            )));
        }
        Ok(self.push_shape(body, local_pose, ShapeGeometry::Sphere { radius }))
    }

    /// Attach a solid box shape to `body`. `half_extents` are the
    /// local half-widths along x/y/z (must be strictly positive). Pass
    /// [`BodyId::NONE`] for a static-world box.
    pub fn add_shape_box(
        &mut self,
        body: BodyId,
        half_extents: Vec3,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_box")?;
        if !half_extents.x.is_finite()
            || !half_extents.y.is_finite()
            || !half_extents.z.is_finite()
            || half_extents.x <= 0.0
            || half_extents.y <= 0.0
            || half_extents.z <= 0.0
        {
            return Err(Error::Build(format!(
                "add_shape_box: half_extents must be positive finite, got {half_extents:?}"
            )));
        }
        Ok(self.push_shape(body, local_pose, ShapeGeometry::Box { half_extents }))
    }

    /// Attach a capsule shape to `body`. Capsule axis is the
    /// shape-local +Y direction (URDF convention). `half_height` is
    /// the half-length of the cylindrical section (cap centres are
    /// at `±half_height·Y` in shape-local frame); `radius` is the
    /// cap + cylinder radius. Both must be strictly positive.
    pub fn add_shape_capsule(
        &mut self,
        body: BodyId,
        half_height: Real,
        radius: Real,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_capsule")?;
        if !half_height.is_finite() || half_height <= 0.0 {
            return Err(Error::Build(format!(
                "add_shape_capsule: half_height must be positive finite, got {half_height}"
            )));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(Error::Build(format!(
                "add_shape_capsule: radius must be positive finite, got {radius}"
            )));
        }
        Ok(self.push_shape(
            body,
            local_pose,
            ShapeGeometry::Capsule { half_height, radius },
        ))
    }

    /// Attach a cylinder shape to `body`. Cylinder axis is the
    /// shape-local +Y direction. Flat caps (no hemispheres). Use
    /// [`Self::add_shape_capsule`] if you need rounded caps.
    pub fn add_shape_cylinder(
        &mut self,
        body: BodyId,
        half_height: Real,
        radius: Real,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_cylinder")?;
        if !half_height.is_finite() || half_height <= 0.0 {
            return Err(Error::Build(format!(
                "add_shape_cylinder: half_height must be positive finite, got {half_height}"
            )));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(Error::Build(format!(
                "add_shape_cylinder: radius must be positive finite, got {radius}"
            )));
        }
        Ok(self.push_shape(
            body,
            local_pose,
            ShapeGeometry::Cylinder { half_height, radius },
        ))
    }

    /// Register a triangle mesh in the model and return its
    /// `mesh_id`. Pass that id into [`Self::add_shape_trimesh`].
    pub fn add_mesh(&mut self, mesh: super::TriMesh) -> u32 {
        let id = self.meshes.len() as u32;
        self.meshes.push(mesh);
        id
    }

    /// Attach a triangle-mesh shape to `body`. The `mesh_id` must be
    /// a value previously returned by [`Self::add_mesh`].
    pub fn add_shape_trimesh(
        &mut self,
        body: BodyId,
        mesh_id: u32,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_trimesh")?;
        if (mesh_id as usize) >= self.meshes.len() {
            return Err(Error::Build(format!(
                "add_shape_trimesh: mesh_id {mesh_id} not registered (have {} meshes)",
                self.meshes.len()
            )));
        }
        Ok(self.push_shape(body, local_pose, ShapeGeometry::TriMesh { mesh_id }))
    }

    /// Attach a triangle shape to `body`. Standalone — for direct
    /// per-triangle collision (terrain patches, etc.).
    pub fn add_shape_triangle(
        &mut self,
        body: BodyId,
        v0: Vec3,
        v1: Vec3,
        v2: Vec3,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_triangle")?;
        Ok(self.push_shape(
            body,
            local_pose,
            ShapeGeometry::Triangle { v0, v1, v2 },
        ))
    }

    /// Attach a static-plane half-space shape to `body`. Pass
    /// [`BodyId::NONE`] for a world-frame ground / wall. `normal` is
    /// the outward-facing direction in shape-local frame; the
    /// `local_pose`'s translation positions a point on the plane.
    pub fn add_shape_plane(
        &mut self,
        body: BodyId,
        normal: Vec3,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_plane")?;
        let len = normal.length();
        if !len.is_finite() || len < 1e-9 {
            return Err(Error::Build(format!(
                "add_shape_plane: normal must be a finite non-zero vector, got {normal:?}"
            )));
        }
        let unit = normal / len;
        Ok(self.push_shape(body, local_pose, ShapeGeometry::Plane { normal: unit }))
    }

    fn validate_body_for_shape(&self, body: BodyId, fn_name: &str) -> Result<()> {
        if body == BodyId::NONE {
            return Ok(());
        }
        if body.index() >= self.body_mass.len() {
            return Err(Error::Build(format!(
                "{fn_name}: body id {body:?} out of range (have {} bodies)",
                self.body_mass.len()
            )));
        }
        Ok(())
    }

    fn push_shape(&mut self, body: BodyId, local_pose: Transform, geom: ShapeGeometry) -> ShapeId {
        let id = ShapeId::from_index(self.shape_geometry.len());
        let body_idx = if body == BodyId::NONE { -1 } else { body.0 as i32 };
        self.shape_body.push(body_idx);
        self.shape_transform.push(local_pose);
        self.shape_geometry.push(geom);
        self.shape_flags.push(super::flags::ShapeFlags::COLLIDES);
        self.shape_friction.push(Real::NAN);
        self.shape_restitution.push(Real::NAN);
        id
    }

    /// Override the friction/restitution for an already-added shape.
    /// `NaN` falls back to `Model::default_*`.
    pub fn set_shape_material(
        &mut self,
        shape: ShapeId,
        friction: Real,
        restitution: Real,
    ) -> Result<()> {
        let i = shape.index();
        if i >= self.shape_geometry.len() {
            return Err(Error::Build(format!(
                "set_shape_material: shape id {shape:?} out of range"
            )));
        }
        self.shape_friction[i] = friction;
        self.shape_restitution[i] = restitution;
        Ok(())
    }

    // ── Joints ──────────────────────────────────────────────────────

    /// Attach a revolute joint between `parent` and `child` rotating
    /// about `axis` (joint-local). Pass [`BodyId::NONE`] for `parent`
    /// to ground the joint to the world frame. `x_p` / `x_c` place
    /// the joint frame in each body's local frame.
    pub fn add_joint_revolute(
        &mut self,
        parent: BodyId,
        child: BodyId,
        x_p: Transform,
        x_c: Transform,
        axis: Vec3,
    ) -> Result<JointId> {
        self.validate_joint(parent, child, "add_joint_revolute")?;
        let len = axis.length();
        if !len.is_finite() || len < 1e-9 {
            return Err(Error::Build(format!(
                "add_joint_revolute: axis must be a finite non-zero vector, got {axis:?}"
            )));
        }
        Ok(self.push_joint(
            super::JointType::Revolute,
            parent,
            child,
            x_p,
            x_c,
            axis / len,
        ))
    }

    /// Prismatic joint — translation along `axis` (joint-local).
    pub fn add_joint_prismatic(
        &mut self,
        parent: BodyId,
        child: BodyId,
        x_p: Transform,
        x_c: Transform,
        axis: Vec3,
    ) -> Result<JointId> {
        self.validate_joint(parent, child, "add_joint_prismatic")?;
        let len = axis.length();
        if !len.is_finite() || len < 1e-9 {
            return Err(Error::Build(format!(
                "add_joint_prismatic: axis must be finite non-zero, got {axis:?}"
            )));
        }
        Ok(self.push_joint(
            super::JointType::Prismatic,
            parent,
            child,
            x_p,
            x_c,
            axis / len,
        ))
    }

    /// Fixed joint — locks all 6 DoFs.
    pub fn add_joint_fixed(
        &mut self,
        parent: BodyId,
        child: BodyId,
        x_p: Transform,
        x_c: Transform,
    ) -> Result<JointId> {
        self.validate_joint(parent, child, "add_joint_fixed")?;
        Ok(self.push_joint(
            super::JointType::Fixed,
            parent,
            child,
            x_p,
            x_c,
            Vec3::ZERO,
        ))
    }

    /// Free 6-DoF joint between `child` and the world frame (or
    /// `parent`). Adds 7 coordinates (xyz + quat) and 6 dof.
    pub fn add_joint_free(
        &mut self,
        parent: BodyId,
        child: BodyId,
        x_p: Transform,
        x_c: Transform,
    ) -> Result<JointId> {
        self.validate_joint(parent, child, "add_joint_free")?;
        Ok(self.push_joint(
            super::JointType::Free,
            parent,
            child,
            x_p,
            x_c,
            Vec3::ZERO,
        ))
    }

    fn validate_joint(&self, parent: BodyId, child: BodyId, fn_name: &str) -> Result<()> {
        if child == BodyId::NONE {
            return Err(Error::Build(format!(
                "{fn_name}: child body required (BodyId::NONE not allowed)"
            )));
        }
        if child.index() >= self.body_mass.len() {
            return Err(Error::Build(format!(
                "{fn_name}: child {child:?} out of range (have {} bodies)",
                self.body_mass.len()
            )));
        }
        if parent != BodyId::NONE && parent.index() >= self.body_mass.len() {
            return Err(Error::Build(format!(
                "{fn_name}: parent {parent:?} out of range (have {} bodies)",
                self.body_mass.len()
            )));
        }
        if parent != BodyId::NONE && parent == child {
            return Err(Error::Build(format!(
                "{fn_name}: parent and child must differ ({parent:?})"
            )));
        }
        Ok(())
    }

    fn push_joint(
        &mut self,
        joint_type: super::JointType,
        parent: BodyId,
        child: BodyId,
        x_p: Transform,
        x_c: Transform,
        axis: Vec3,
    ) -> JointId {
        let id = JointId::from_index(self.joint_type.len());
        let parent_idx = if parent == BodyId::NONE { -1 } else { parent.0 as i32 };
        self.joint_type.push(joint_type as u32);
        self.joint_parent.push(parent_idx);
        self.joint_child.push(child.0 as i32);
        self.joint_x_p.push(x_p);
        self.joint_x_c.push(x_c);
        self.joint_axis.push(axis);

        let (cc, dc) = joint_type.coord_dof();
        let prev_coord = *self.joint_coord_offset.last().unwrap();
        let prev_dof = *self.joint_dof_offset.last().unwrap();
        self.joint_coord_offset.push(prev_coord + cc as u32);
        self.joint_dof_offset.push(prev_dof + dc as u32);
        joint_type.write_zero_coords(&mut self.joint_q0);
        // Default limits / drives: NaN = unlimited, 0 friction.
        for _ in 0..dc {
            self.joint_limit_lower.push(Real::NAN);
            self.joint_limit_upper.push(Real::NAN);
            self.joint_friction.push(0.0);
            self.joint_effort_limit.push(Real::NAN);
            self.joint_velocity_limit.push(Real::NAN);
        }
        // First joint of a (re-)opened articulation: nothing to push
        // into `articulation_start` — the trailing offset is set when
        // the articulation closes (either via `begin_articulation` or
        // at `build` time).
        self.articulation_open = true;
        id
    }

    /// Set the position limits for a joint's DOFs. Pass `NaN` to leave a
    /// bound unset. For multi-DOF joints (e.g. `Free`, future ball/D6),
    /// `lower` and `upper` must have length = `joint.dof_count()`.
    pub fn set_joint_limits(
        &mut self,
        joint: JointId,
        lower: &[Real],
        upper: &[Real],
    ) -> Result<()> {
        let (start, end) = self.joint_dof_range(joint, "set_joint_limits")?;
        let n = end - start;
        if lower.len() != n || upper.len() != n {
            return Err(Error::Build(format!(
                "set_joint_limits: joint {joint:?} has {n} DOFs, got lower={} upper={}",
                lower.len(),
                upper.len()
            )));
        }
        self.joint_limit_lower[start..end].copy_from_slice(lower);
        self.joint_limit_upper[start..end].copy_from_slice(upper);
        Ok(())
    }

    /// Set per-DOF damping (joint friction, N·m·s/rad or N·s/m).
    pub fn set_joint_friction(
        &mut self,
        joint: JointId,
        friction: &[Real],
    ) -> Result<()> {
        let (start, end) = self.joint_dof_range(joint, "set_joint_friction")?;
        let n = end - start;
        if friction.len() != n {
            return Err(Error::Build(format!(
                "set_joint_friction: joint {joint:?} has {n} DOFs, got {}",
                friction.len()
            )));
        }
        self.joint_friction[start..end].copy_from_slice(friction);
        Ok(())
    }

    /// Set per-DOF effort cap (N·m or N). `NaN` = unlimited.
    pub fn set_joint_effort_limit(
        &mut self,
        joint: JointId,
        effort: &[Real],
    ) -> Result<()> {
        let (start, end) = self.joint_dof_range(joint, "set_joint_effort_limit")?;
        let n = end - start;
        if effort.len() != n {
            return Err(Error::Build(format!(
                "set_joint_effort_limit: joint {joint:?} has {n} DOFs, got {}",
                effort.len()
            )));
        }
        self.joint_effort_limit[start..end].copy_from_slice(effort);
        Ok(())
    }

    /// Set per-DOF velocity cap (rad/s or m/s). `NaN` = unlimited.
    pub fn set_joint_velocity_limit(
        &mut self,
        joint: JointId,
        velocity: &[Real],
    ) -> Result<()> {
        let (start, end) = self.joint_dof_range(joint, "set_joint_velocity_limit")?;
        let n = end - start;
        if velocity.len() != n {
            return Err(Error::Build(format!(
                "set_joint_velocity_limit: joint {joint:?} has {n} DOFs, got {}",
                velocity.len()
            )));
        }
        self.joint_velocity_limit[start..end].copy_from_slice(velocity);
        Ok(())
    }

    fn joint_dof_range(
        &self,
        joint: JointId,
        fn_name: &'static str,
    ) -> Result<(usize, usize)> {
        let i = joint.index();
        if i + 1 >= self.joint_dof_offset.len() {
            return Err(Error::Build(format!(
                "{fn_name}: joint id {joint:?} out of range (have {} joints)",
                self.joint_type.len()
            )));
        }
        Ok((
            self.joint_dof_offset[i] as usize,
            self.joint_dof_offset[i + 1] as usize,
        ))
    }

    /// Set the gravity vector for `world`. Auto-extends the world list as
    /// needed; new worlds default to -9.81 m/s² on -Y.
    pub fn set_world_gravity(&mut self, world: WorldId, gravity: Vec3) {
        let w = world.0 as usize;
        while self.world_gravity.len() <= w {
            self.world_gravity
                .push(Vec3::new(0.0, -crate::math::GRAVITY, 0.0));
        }
        self.world_gravity[w] = gravity;
    }

    /// Set the global angular-damping coefficient (1/s) applied each step.
    pub fn set_angular_damping(&mut self, damping: Real) {
        self.angular_damping = damping;
    }

    /// Number of bodies registered so far.
    #[must_use]
    pub fn body_count(&self) -> usize {
        self.body_mass.len()
    }

    /// Number of shapes registered so far.
    #[must_use]
    pub fn shape_count(&self) -> usize {
        self.shape_geometry.len()
    }

    /// Finalise into a [`Model`].
    pub fn build(mut self) -> Result<Model> {
        let s = BufferUsage::Storage;
        let body_count = self.body_mass.len();
        let shape_count = self.shape_geometry.len();
        let world_count = self.world_gravity.len();

        // Close the trailing articulation if the user didn't.
        if self.articulation_open {
            self.articulation_start.push(self.joint_type.len() as u32);
            self.articulation_open = false;
        }
        let articulation_count = self.articulation_start.len().saturating_sub(1);

        // Validate. `shape_body == -1` is the static-world sentinel.
        for (i, &body) in self.shape_body.iter().enumerate() {
            if body != -1 && (body < 0 || (body as usize) >= body_count) {
                return Err(Error::Build(format!(
                    "shape {i} references invalid body id {body}"
                )));
            }
        }

        Ok(Model {
            particle_count: 0,
            particle_mass: Buffer::from_vec(Vec::new(), s),
            particle_inv_mass: Buffer::from_vec(Vec::new(), s),
            particle_radius: Buffer::from_vec(Vec::new(), s),
            particle_flags: Buffer::from_vec(Vec::new(), s),
            particle_world: Buffer::from_vec(Vec::new(), s),

            body_count,
            body_mass: Buffer::from_vec(self.body_mass, s),
            body_inv_mass: Buffer::from_vec(self.body_inv_mass, s),
            body_inertia: Buffer::from_vec(self.body_inertia, s),
            body_inv_inertia: Buffer::from_vec(self.body_inv_inertia, s),
            body_com: Buffer::from_vec(self.body_com, s),
            body_flags: Buffer::from_vec(self.body_flags, s),
            body_world: Buffer::from_vec(self.body_world, s),
            initial_body_q: Buffer::from_vec(self.body_q0, s),

            shape_count,
            shape_body: Buffer::from_vec(self.shape_body, s),
            shape_transform: Buffer::from_vec(self.shape_transform, s),
            shape_geometry: self.shape_geometry,
            shape_flags: Buffer::from_vec(self.shape_flags, s),
            shape_friction: Buffer::from_vec(self.shape_friction, s),
            shape_restitution: Buffer::from_vec(self.shape_restitution, s),
            meshes: self.meshes,

            joint_count: self.joint_type.len(),
            joint_type: Buffer::from_vec(self.joint_type, s),
            joint_parent: Buffer::from_vec(self.joint_parent, s),
            joint_child: Buffer::from_vec(self.joint_child, s),
            joint_x_p: Buffer::from_vec(self.joint_x_p, s),
            joint_x_c: Buffer::from_vec(self.joint_x_c, s),
            joint_axis: Buffer::from_vec(self.joint_axis, s),
            joint_coord_total: *self.joint_coord_offset.last().unwrap_or(&0) as usize,
            joint_dof_total: *self.joint_dof_offset.last().unwrap_or(&0) as usize,
            joint_coord_offset: Buffer::from_vec(self.joint_coord_offset, s),
            joint_dof_offset: Buffer::from_vec(self.joint_dof_offset, s),
            initial_joint_q: Buffer::from_vec(self.joint_q0, s),
            joint_limit_lower: Buffer::from_vec(self.joint_limit_lower, s),
            joint_limit_upper: Buffer::from_vec(self.joint_limit_upper, s),
            joint_friction: Buffer::from_vec(self.joint_friction, s),
            joint_effort_limit: Buffer::from_vec(self.joint_effort_limit, s),
            joint_velocity_limit: Buffer::from_vec(self.joint_velocity_limit, s),

            articulation_count,
            articulation_start: Buffer::from_vec(self.articulation_start, s),

            world_count,
            gravity: Buffer::from_vec(self.world_gravity, s),

            angular_damping: self.angular_damping,
            particle_velocity_max: self.particle_velocity_max,
            contact_margin: self.contact_margin,
            default_friction: self.default_friction,
            default_restitution: self.default_restitution,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_one_sphere_body() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
                .with_position(Vec3::new(0.0, 5.0, 0.0)),
        );
        b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
        let m = b.build().unwrap();
        assert_eq!(m.body_count, 1);
        assert_eq!(m.shape_count, 1);
        assert_eq!(m.world_count, 1);
        let q = m.initial_body_q.host().unwrap();
        assert!((q[0].position.y - 5.0).abs() < 1e-12);
    }

    #[test]
    fn invalid_radius_is_rejected() {
        let mut b = ModelBuilder::new();
        let body = b.add_body(BodyParams::new());
        assert!(b.add_shape_sphere(body, -1.0, Transform::IDENTITY).is_err());
        assert!(b.add_shape_sphere(body, 0.0, Transform::IDENTITY).is_err());
    }

    #[test]
    fn set_world_gravity_extends_world_list() {
        let mut b = ModelBuilder::new();
        b.set_world_gravity(WorldId(2), Vec3::new(0.0, 0.0, -1.0));
        let m = b.build().unwrap();
        assert_eq!(m.world_count, 3);
        let g = m.gravity.host().unwrap();
        assert!((g[2].z + 1.0).abs() < 1e-12);
    }
}
