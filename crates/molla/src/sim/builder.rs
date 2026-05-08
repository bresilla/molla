//! `ModelBuilder` — ergonomic authoring surface for [`Model`].
//!
//! Stage 1 surface: rigid bodies, sphere shapes, per-world gravity. More
//! arrives in stages 3 (joints) and 4 (other shapes).

use crate::core::{BodyId, Error, Result, ShapeId, WorldId};
use crate::math::{Mat3, Real, SpatialInertia, Transform, Vec3};
use crate::storage::{Buffer, BufferUsage};

use super::flags::BodyFlags;
use super::Model;

/// Per-shape geometry tag. Stage 1 only ships [`ShapeGeometry::Sphere`];
/// boxes/capsules/planes/meshes follow in stage 4.
#[derive(Copy, Clone, Debug)]
pub enum ShapeGeometry {
    /// Solid sphere centred on the shape's local origin.
    Sphere {
        /// Radius (m).
        radius: Real,
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

    // Per-world arrays.
    world_gravity: Vec<Vec3>,

    // Solver knobs (defaults match Newton's `SemiImplicitSolver`).
    angular_damping: Real,
    particle_velocity_max: Real,
}

impl ModelBuilder {
    /// Construct a fresh builder with one world (`WorldId(0)`) at standard
    /// gravity pointing in -Y.
    #[must_use]
    pub fn new() -> Self {
        Self {
            world_gravity: vec![Vec3::new(0.0, -crate::math::GRAVITY, 0.0)],
            angular_damping: 0.05,
            particle_velocity_max: 1.0e6,
            ..Default::default()
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

    /// Attach a sphere shape to `body`.
    pub fn add_shape_sphere(
        &mut self,
        body: BodyId,
        radius: Real,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        if body.index() >= self.body_mass.len() {
            return Err(Error::Build(format!(
                "add_shape_sphere: body id {body:?} out of range (have {} bodies)",
                self.body_mass.len()
            )));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(Error::Build(format!(
                "add_shape_sphere: radius must be a positive finite number, got {radius}"
            )));
        }
        let id = ShapeId::from_index(self.shape_geometry.len());
        self.shape_body.push(body.0 as i32);
        self.shape_transform.push(local_pose);
        self.shape_geometry.push(ShapeGeometry::Sphere { radius });
        self.shape_flags.push(super::flags::ShapeFlags::COLLIDES);
        Ok(id)
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
    pub fn build(self) -> Result<Model> {
        let s = BufferUsage::Storage;
        let body_count = self.body_mass.len();
        let shape_count = self.shape_geometry.len();
        let world_count = self.world_gravity.len();

        // Validate.
        for (i, &body) in self.shape_body.iter().enumerate() {
            if body < 0 || (body as usize) >= body_count {
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

            joint_count: 0,
            joint_parent: Buffer::from_vec(Vec::new(), s),
            joint_child: Buffer::from_vec(Vec::new(), s),
            joint_x_p: Buffer::from_vec(Vec::new(), s),
            joint_x_c: Buffer::from_vec(Vec::new(), s),

            world_count,
            gravity: Buffer::from_vec(self.world_gravity, s),

            angular_damping: self.angular_damping,
            particle_velocity_max: self.particle_velocity_max,
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
