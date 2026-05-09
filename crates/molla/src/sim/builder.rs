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
    /// Convex hull — references entry `hull_id` in
    /// [`super::Model::hulls`]. Support function = `argmax_v (v · dir)`
    /// over the hull's vertex list. Adequate for ~50 vertices; past
    /// that the linear scan dominates and a per-hull DOP / spherical
    /// acceleration becomes worth it.
    ConvexHull {
        /// Index into [`super::Model::hulls`].
        hull_id: u32,
    },
    /// Solid cone along the shape-local **+Y axis**: apex at
    /// `(0, +half_height, 0)`, base disc at `y = -half_height` with
    /// radius `radius`. Note this is the standard graphics/physics
    /// convention; some CAD tools use apex-down.
    Cone {
        /// Half the apex-to-base height (m).
        half_height: Real,
        /// Radius of the base disc (m).
        radius: Real,
    },
    /// Heightfield (regular grid of height samples) — references
    /// entry `field_id` in [`super::Model::heightfields`]. Surface
    /// is defined by `height[i, j]` at grid coordinates
    /// `(origin + i·cell_size·X, height, origin + j·cell_size·Z)`.
    /// Each quad cell decomposes into 2 triangles on the fly during
    /// narrow phase; collision queries route through the same
    /// triangle-based GJK/EPA path as `TriMesh`.
    Heightfield {
        /// Index into [`super::Model::heightfields`].
        field_id: u32,
    },
    /// Analytic signed-distance-field shape — referenced by
    /// `sdf_id` in [`super::Model::sdfs`]. The current v1 ships
    /// `SdfShape::Torus { major, minor }` and `SdfShape::Capped`
    /// (sentinel for arbitrary callbacks); the support function
    /// is approximated via finite-difference normals at the SDF's
    /// surface ("ball-marching" approach used by Newton's
    /// `geometry/sdf.py`).
    Sdf {
        /// Index into [`super::Model::sdfs`].
        sdf_id: u32,
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
    // Particle arrays.
    particle_mass: Vec<Real>,
    particle_inv_mass: Vec<Real>,
    particle_radius: Vec<Real>,
    particle_flags: Vec<u32>,
    particle_world: Vec<i32>,
    particle_q0: Vec<Vec3>,

    // Spring (distance constraint) arrays.
    spring_a: Vec<u32>,
    spring_b: Vec<u32>,
    spring_rest_length: Vec<Real>,
    spring_compliance: Vec<Real>,
    spring_max_stretch: Vec<Real>,

    // Triangle topology — flat `tri_count*3` u32 array.
    tri_indices: Vec<u32>,

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
    shape_rolling_friction: Vec<Real>,
    shape_torsional_friction: Vec<Real>,
    shape_restitution: Vec<Real>,
    meshes: Vec<super::TriMesh>,
    hulls: Vec<Vec<Vec3>>,
    heightfields: Vec<super::Heightfield>,
    sdfs: Vec<super::SdfShape>,
    shape_collision_group: Vec<u32>,
    shape_collision_filter_pairs: Vec<(u32, u32)>,
    shape_color: Vec<Vec3>,
    shape_metallic: Vec<Real>,
    shape_roughness: Vec<Real>,

    // Equality-constraint arrays.
    equality_type: Vec<u32>,
    equality_body_a: Vec<i32>,
    equality_body_b: Vec<i32>,
    equality_anchor_a: Vec<Vec3>,
    equality_anchor_b: Vec<Vec3>,
    equality_relative_rotation: Vec<crate::math::Quat>,
    equality_enabled: Vec<u32>,
    equality_constraint_world: Vec<i32>,

    // Mimic-constraint arrays.
    mimic_joint_a: Vec<i32>,
    mimic_joint_b: Vec<i32>,
    mimic_c0: Vec<Real>,
    mimic_c1: Vec<Real>,
    mimic_enabled: Vec<u32>,
    mimic_constraint_world: Vec<i32>,

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
    /// Per-joint world (defaults to child body's world).
    joint_world: Vec<i32>,
    /// Per-articulation world (set when articulation closes).
    articulation_world: Vec<i32>,
    /// Per-joint enable flag (1 = active by default).
    joint_enabled: Vec<u32>,
    /// Per-joint ancestor index in the articulation tree.
    joint_ancestor: Vec<i32>,
    /// Per-DOF static joint stiffness (default 0).
    joint_target_ke: Vec<Real>,

    // Per-world arrays.
    world_gravity: Vec<Vec3>,

    // Solver knobs (defaults match Newton's `SemiImplicitSolver`).
    angular_damping: Real,
    linear_damping: Real,
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
            linear_damping: 0.0,
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
            // Capture the world of the just-closed articulation: the
            // first joint in this chain dictates it (matches Newton).
            let first_joint_idx =
                *self.articulation_start.last().unwrap_or(&0) as usize;
            let world = self
                .joint_world
                .get(first_joint_idx)
                .copied()
                .unwrap_or(0);
            self.articulation_world.push(world);
            self.articulation_start.push(self.joint_type.len() as u32);
            self.articulation_open = false;
        }
    }

    /// Add a particle at `position`. Returns the particle index.
    /// Pass `mass = 0.0` for a kinematic anchor (its position will be
    /// passed through unchanged by the integrator).
    pub fn add_particle(
        &mut self,
        position: Vec3,
        mass: Real,
        radius: Real,
        world: WorldId,
    ) -> u32 {
        // Make sure the requested world exists.
        let w = world.0 as usize;
        while self.world_gravity.len() <= w {
            self.world_gravity
                .push(Vec3::new(0.0, -crate::math::GRAVITY, 0.0));
        }
        let id = self.particle_mass.len() as u32;
        let inv_mass = if mass > 0.0 { 1.0 / mass } else { 0.0 };
        self.particle_mass.push(mass);
        self.particle_inv_mass.push(inv_mass);
        self.particle_radius.push(radius);
        self.particle_flags.push(super::ParticleFlags::ACTIVE);
        self.particle_world.push(world.0 as i32);
        self.particle_q0.push(position);
        id
    }

    /// Add a distance constraint between two particles.
    ///
    /// * `compliance` — XPBD α (m/N). 0 = stiff, 1e-7 = stiff
    ///   stretch, 1e-2 = soft. Equivalent to `1 / spring_stiffness`.
    /// * `max_stretch` — hard upper bound on stretch as a ratio of
    ///   rest length. 0.0 = perfectly inextensible, 0.05 = 5 % linen,
    ///   0.5 = jersey, ≥ 1.0 effectively disables the cap. Enforced
    ///   as a non-linear projection after the soft XPBD pass so
    ///   pulling on cloth never exceeds the configured stretch
    ///   regardless of how the user yanks the cursor.
    pub fn add_distance_constraint(
        &mut self,
        a: u32,
        b: u32,
        rest_length: Real,
        compliance: Real,
        max_stretch: Real,
    ) -> usize {
        let id = self.spring_a.len();
        self.spring_a.push(a);
        self.spring_b.push(b);
        self.spring_rest_length.push(rest_length);
        self.spring_compliance.push(compliance);
        self.spring_max_stretch.push(max_stretch);
        id
    }

    /// Add a triangle (CCW) referencing three particles. Used for
    /// rendering / future bending constraints.
    pub fn add_triangle(&mut self, v0: u32, v1: u32, v2: u32) -> usize {
        let id = self.tri_indices.len() / 3;
        self.tri_indices.push(v0);
        self.tri_indices.push(v1);
        self.tri_indices.push(v2);
        id
    }

    /// Helper: build a `n_x × n_y` cloth grid spanned by `u_axis` and
    /// `v_axis` at `origin`. The grid corners sit at `origin`,
    /// `origin + u_axis`, `origin + v_axis`, `origin + u_axis + v_axis`.
    /// Wires structural (edge) + shear (diagonal) + bending ("skip-one")
    /// distance constraints and emits two CCW triangles per quad cell
    /// for rendering.
    ///
    /// For a horizontal cloth pass `u_axis = +X`, `v_axis = +Z`. For a
    /// vertical curtain hanging from the top edge pass `u_axis = +X`,
    /// `v_axis = -Y`. Returns `(first_particle_index, n_x, n_y)`. Pin
    /// specific particles afterwards via `pin_particle`.
    pub fn add_cloth_grid(
        &mut self,
        origin: Vec3,
        u_axis: Vec3,
        v_axis: Vec3,
        n_x: u32,
        n_y: u32,
        particle_mass: Real,
        particle_radius: Real,
        compliance: Real,
        max_stretch: Real,
        world: WorldId,
    ) -> (u32, u32, u32) {
        self.add_cloth_grid_with_bending(
            origin,
            u_axis,
            v_axis,
            n_x,
            n_y,
            particle_mass,
            particle_radius,
            compliance,
            max_stretch,
            // Bending compliance: 30× softer than the structural
            // compliance by default. Cloth folds easily but isn't
            // glued flat — same ratio Macklin (2014, §A.2) uses for
            // PBD cloth scenes. Pass `bending_compliance < 0` to
            // disable.
            compliance * 30.0,
            world,
        )
    }

    /// Like [`Self::add_cloth_grid`] but with explicit bending
    /// compliance. `bending_compliance < 0` skips the bending pass
    /// entirely (cheaper for stiff sheets where shear + structural
    /// springs already discourage folding).
    #[allow(clippy::too_many_arguments)]
    pub fn add_cloth_grid_with_bending(
        &mut self,
        origin: Vec3,
        u_axis: Vec3,
        v_axis: Vec3,
        n_x: u32,
        n_y: u32,
        particle_mass: Real,
        particle_radius: Real,
        compliance: Real,
        max_stretch: Real,
        bending_compliance: Real,
        world: WorldId,
    ) -> (u32, u32, u32) {
        let nx = n_x.max(2);
        let ny = n_y.max(2);
        let du = u_axis / (nx - 1) as Real;
        let dv = v_axis / (ny - 1) as Real;
        let dx = du.length();
        let dy = dv.length();
        let first = self.particle_mass.len() as u32;

        // Particles laid out row-major: linear(ix, iy) = first + iy*nx + ix.
        for iy in 0..ny {
            for ix in 0..nx {
                let p = origin + du * ix as Real + dv * iy as Real;
                self.add_particle(p, particle_mass, particle_radius, world);
            }
        }

        let idx = |ix: u32, iy: u32| first + iy * nx + ix;

        // Structural constraints (4-neighbour edges).
        for iy in 0..ny {
            for ix in 0..nx {
                if ix + 1 < nx {
                    self.add_distance_constraint(
                        idx(ix, iy),
                        idx(ix + 1, iy),
                        dx,
                        compliance,
                        max_stretch,
                    );
                }
                if iy + 1 < ny {
                    self.add_distance_constraint(
                        idx(ix, iy),
                        idx(ix, iy + 1),
                        dy,
                        compliance,
                        max_stretch,
                    );
                }
            }
        }

        // Shear constraints (both diagonals of every quad cell).
        let diag = (dx * dx + dy * dy).sqrt();
        for iy in 0..ny - 1 {
            for ix in 0..nx - 1 {
                self.add_distance_constraint(
                    idx(ix, iy),
                    idx(ix + 1, iy + 1),
                    diag,
                    compliance,
                    max_stretch,
                );
                self.add_distance_constraint(
                    idx(ix + 1, iy),
                    idx(ix, iy + 1),
                    diag,
                    compliance,
                    max_stretch,
                );
            }
        }

        // Bending springs ("skip-one"): connect every-other particle
        // along U and V. These are slack at rest but stretch when
        // the cloth folds out of plane, providing dihedral-angle
        // resistance via the existing distance constraint
        // infrastructure (the trick popularised by Bridson 2002 and
        // PBD; Newton's `solver_xpbd` cloth example uses the same
        // pattern). Disabled when `bending_compliance < 0`.
        if bending_compliance >= 0.0 {
            // Skip the strain-limit cap on bending springs — folding
            // cloth needs them to stretch significantly. Pass the
            // sentinel "disabled" value (>= 1.0) to `max_stretch`.
            const BEND_STRETCH_DISABLED: Real = 2.0;
            for iy in 0..ny {
                for ix in 0..nx {
                    if ix + 2 < nx {
                        self.add_distance_constraint(
                            idx(ix, iy),
                            idx(ix + 2, iy),
                            dx * 2.0,
                            bending_compliance,
                            BEND_STRETCH_DISABLED,
                        );
                    }
                    if iy + 2 < ny {
                        self.add_distance_constraint(
                            idx(ix, iy),
                            idx(ix, iy + 2),
                            dy * 2.0,
                            bending_compliance,
                            BEND_STRETCH_DISABLED,
                        );
                    }
                }
            }
        }

        // Triangle topology — two CCW triangles per quad.
        for iy in 0..ny - 1 {
            for ix in 0..nx - 1 {
                let v00 = idx(ix, iy);
                let v10 = idx(ix + 1, iy);
                let v01 = idx(ix, iy + 1);
                let v11 = idx(ix + 1, iy + 1);
                self.add_triangle(v00, v10, v11);
                self.add_triangle(v00, v11, v01);
            }
        }

        (first, nx, ny)
    }

    /// Pin particle `i` in place by setting its mass to zero.
    pub fn pin_particle(&mut self, i: u32) {
        let i = i as usize;
        if i >= self.particle_mass.len() {
            return;
        }
        self.particle_mass[i] = 0.0;
        self.particle_inv_mass[i] = 0.0;
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
    pub fn add_mesh(&mut self, mut mesh: super::TriMesh) -> u32 {
        // Build the BVH eagerly so the first narrow-phase query is
        // already accelerated. Cheap (~µs per 100 triangles); avoids
        // an interior-mut cache invariant inside `Model`.
        if mesh.bvh.nodes.is_empty() && !mesh.indices.is_empty() {
            mesh.build_bvh(4);
        }
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

    /// Register a convex hull (vertex list, in shape-local frame)
    /// and return its `hull_id`. Pass that id into
    /// [`Self::add_shape_convex_hull`]. The hull is assumed convex
    /// — caller is responsible for that; if you have a non-convex
    /// mesh, run a hull builder (`qhull` / `chull3d`) first.
    pub fn add_hull(&mut self, vertices: Vec<Vec3>) -> u32 {
        let id = self.hulls.len() as u32;
        self.hulls.push(vertices);
        id
    }

    /// Attach a convex-hull shape to `body`. The `hull_id` must be a
    /// value previously returned by [`Self::add_hull`].
    pub fn add_shape_convex_hull(
        &mut self,
        body: BodyId,
        hull_id: u32,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_convex_hull")?;
        if (hull_id as usize) >= self.hulls.len() {
            return Err(Error::Build(format!(
                "add_shape_convex_hull: hull_id {hull_id} not registered (have {} hulls)",
                self.hulls.len()
            )));
        }
        if self.hulls[hull_id as usize].is_empty() {
            return Err(Error::Build(format!(
                "add_shape_convex_hull: hull {hull_id} has no vertices"
            )));
        }
        Ok(self.push_shape(body, local_pose, ShapeGeometry::ConvexHull { hull_id }))
    }

    /// Attach a cone shape to `body`. Cone apex sits at
    /// `(0, +half_height, 0)` in shape-local frame; base disc at
    /// `y = -half_height` with radius `radius`.
    pub fn add_shape_cone(
        &mut self,
        body: BodyId,
        half_height: Real,
        radius: Real,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_cone")?;
        if !(half_height > 0.0 && radius > 0.0) {
            return Err(Error::Build(format!(
                "add_shape_cone: half_height ({half_height}) and radius ({radius}) must be > 0"
            )));
        }
        Ok(self.push_shape(body, local_pose, ShapeGeometry::Cone { half_height, radius }))
    }

    /// Register a heightfield (regular grid of height samples) and
    /// return its `field_id`. The grid sits at `origin` in shape-local
    /// frame with `nx · nz` samples spaced `cell_size` apart.
    pub fn add_heightfield(&mut self, hf: super::Heightfield) -> u32 {
        let id = self.heightfields.len() as u32;
        self.heightfields.push(hf);
        id
    }

    /// Attach a heightfield shape to `body`.
    pub fn add_shape_heightfield(
        &mut self,
        body: BodyId,
        field_id: u32,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_heightfield")?;
        if (field_id as usize) >= self.heightfields.len() {
            return Err(Error::Build(format!(
                "add_shape_heightfield: field_id {field_id} not registered"
            )));
        }
        Ok(self.push_shape(body, local_pose, ShapeGeometry::Heightfield { field_id }))
    }

    /// Register an analytic SDF shape and return its id. v1 ships
    /// torus + sphere; user-defined SDFs need a procedural callback
    /// path (TODO).
    pub fn add_sdf(&mut self, sdf: super::SdfShape) -> u32 {
        let id = self.sdfs.len() as u32;
        self.sdfs.push(sdf);
        id
    }

    /// Attach an SDF shape to `body`.
    pub fn add_shape_sdf(
        &mut self,
        body: BodyId,
        sdf_id: u32,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_sdf")?;
        if (sdf_id as usize) >= self.sdfs.len() {
            return Err(Error::Build(format!(
                "add_shape_sdf: sdf_id {sdf_id} not registered"
            )));
        }
        Ok(self.push_shape(body, local_pose, ShapeGeometry::Sdf { sdf_id }))
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
        self.shape_rolling_friction.push(Real::NAN);
        self.shape_torsional_friction.push(Real::NAN);
        self.shape_restitution.push(Real::NAN);
        self.shape_collision_group.push(0xffff_ffff);
        // Visual material defaults: white Lambertian.
        self.shape_color.push(Vec3::new(1.0, 1.0, 1.0));
        self.shape_metallic.push(0.0);
        self.shape_roughness.push(0.5);
        id
    }

    /// Override the per-shape visual material (PBR-style colour +
    /// metallic + roughness). Consumed by the GPU camera-trace kernel
    /// ([`crate::sensors::TiledCamera`]); does not affect collision /
    /// dynamics. Default is white, non-metallic, mid-roughness.
    pub fn set_shape_visual_material(
        &mut self,
        shape: ShapeId,
        color: Vec3,
        metallic: Real,
        roughness: Real,
    ) -> Result<()> {
        let i = shape.index();
        if i >= self.shape_geometry.len() {
            return Err(Error::Build(format!(
                "set_shape_visual_material: shape id {shape:?} out of range"
            )));
        }
        self.shape_color[i] = color;
        self.shape_metallic[i] = metallic.clamp(0.0, 1.0);
        self.shape_roughness[i] = roughness.clamp(0.0, 1.0);
        Ok(())
    }

    /// Set the collision-group bitmask for an already-added shape.
    /// Two shapes collide only if `group_a & group_b != 0`. Default
    /// is `0xffff_ffff` (all groups).
    pub fn set_shape_collision_group(
        &mut self,
        shape: ShapeId,
        group: u32,
    ) -> Result<()> {
        let i = shape.index();
        if i >= self.shape_collision_group.len() {
            return Err(Error::Build(format!(
                "set_shape_collision_group: shape {shape:?} out of range"
            )));
        }
        self.shape_collision_group[i] = group;
        Ok(())
    }

    /// Add an explicit shape-pair exclusion. The pair is always
    /// filtered out by the broad phase, even if group bitmasks would
    /// allow it.
    pub fn add_shape_collision_exclusion(&mut self, a: ShapeId, b: ShapeId) {
        let (lo, hi) = if a.0 < b.0 {
            (a.0, b.0)
        } else {
            (b.0, a.0)
        };
        self.shape_collision_filter_pairs.push((lo, hi));
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

    /// Override the rolling / torsional friction for an
    /// already-added shape. `NaN` (or zero) disables that mode.
    /// Rolling friction opposes spin perpendicular to the contact
    /// normal (wheel-on-ground); torsional opposes spin along the
    /// normal (top spinning on a surface).
    pub fn set_shape_friction_modes(
        &mut self,
        shape: ShapeId,
        rolling: Real,
        torsional: Real,
    ) -> Result<()> {
        let i = shape.index();
        if i >= self.shape_geometry.len() {
            return Err(Error::Build(format!(
                "set_shape_friction_modes: shape id {shape:?} out of range"
            )));
        }
        self.shape_rolling_friction[i] = rolling;
        self.shape_torsional_friction[i] = torsional;
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

    /// 3-DoF spherical / ball joint between `parent` and `child`.
    /// Adds 4 coordinates (quaternion qx, qy, qz, qw) and 3 dof.
    pub fn add_joint_ball(
        &mut self,
        parent: BodyId,
        child: BodyId,
        x_p: Transform,
        x_c: Transform,
    ) -> Result<JointId> {
        self.validate_joint(parent, child, "add_joint_ball")?;
        Ok(self.push_joint(
            super::JointType::Ball,
            parent,
            child,
            x_p,
            x_c,
            Vec3::ZERO,
        ))
    }

    /// 6-DoF distance joint — like `Free` but with a distance limit
    /// between the joint anchors. Adds 7 coordinates / 6 dof. Set
    /// `min_distance` / `max_distance` via the per-DOF
    /// `set_joint_limits` once the joint id is known.
    pub fn add_joint_distance(
        &mut self,
        parent: BodyId,
        child: BodyId,
        x_p: Transform,
        x_c: Transform,
    ) -> Result<JointId> {
        self.validate_joint(parent, child, "add_joint_distance")?;
        Ok(self.push_joint(
            super::JointType::Distance,
            parent,
            child,
            x_p,
            x_c,
            Vec3::ZERO,
        ))
    }

    /// 6-DoF D6 joint with per-axis lock/limit configuration. Adds
    /// 7 coordinates / 6 dof. Per-axis locks are applied via
    /// `set_joint_limits(joint, lower, upper)` — set `lower==upper==0`
    /// on the locked axes.
    pub fn add_joint_d6(
        &mut self,
        parent: BodyId,
        child: BodyId,
        x_p: Transform,
        x_c: Transform,
    ) -> Result<JointId> {
        self.validate_joint(parent, child, "add_joint_d6")?;
        Ok(self.push_joint(
            super::JointType::D6,
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
        // Joint world inherits from the child body's world.
        let child_world = self
            .body_world
            .get(child.0 as usize)
            .copied()
            .unwrap_or(0);
        self.joint_world.push(child_world);
        self.joint_enabled.push(1);
        // Ancestor: scan the current articulation's joints for one
        // whose child body matches this joint's parent body.
        let parent_body_idx = parent_idx;
        let articulation_first =
            *self.articulation_start.last().unwrap_or(&0) as usize;
        let mut ancestor: i32 = -1;
        if parent_body_idx >= 0 {
            for k in articulation_first..self.joint_child.len().saturating_sub(1) {
                if self.joint_child[k] == parent_body_idx {
                    ancestor = k as i32;
                    break;
                }
            }
        }
        self.joint_ancestor.push(ancestor);
        // Per-DOF static stiffness defaults to 0.
        for _ in 0..dc {
            self.joint_target_ke.push(0.0);
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

    /// Enable or disable a joint. Disabled joints don't contribute
    /// to the dynamics solve.
    pub fn set_joint_enabled(&mut self, joint: JointId, enabled: bool) -> Result<()> {
        let i = joint.index();
        if i >= self.joint_enabled.len() {
            return Err(Error::Build(format!(
                "set_joint_enabled: joint {joint:?} out of range"
            )));
        }
        self.joint_enabled[i] = if enabled { 1 } else { 0 };
        Ok(())
    }

    /// Per-DOF static joint stiffness (e.g. for spring-style drives).
    pub fn set_joint_target_ke(&mut self, joint: JointId, ke: &[Real]) -> Result<()> {
        let (start, end) = self.joint_dof_range(joint, "set_joint_target_ke")?;
        let n = end - start;
        if ke.len() != n {
            return Err(Error::Build(format!(
                "set_joint_target_ke: joint {joint:?} has {n} DOFs, got {}",
                ke.len()
            )));
        }
        self.joint_target_ke[start..end].copy_from_slice(ke);
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

    /// Add a CONNECT (ball-weld) equality constraint: pin
    /// `anchor_a` (in body A's local frame) to `anchor_b` (in body
    /// B's local frame). Bodies remain free to rotate.
    pub fn add_equality_connect(
        &mut self,
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vec3,
        anchor_b: Vec3,
    ) -> usize {
        let id = self.equality_type.len();
        self.equality_type.push(super::EqualityType::Connect as u32);
        self.equality_body_a.push(if body_a == BodyId::NONE { -1 } else { body_a.0 as i32 });
        self.equality_body_b.push(if body_b == BodyId::NONE { -1 } else { body_b.0 as i32 });
        self.equality_anchor_a.push(anchor_a);
        self.equality_anchor_b.push(anchor_b);
        self.equality_relative_rotation
            .push(crate::math::Quat::IDENTITY);
        self.equality_enabled.push(1);
        let world = if body_a != BodyId::NONE {
            self.body_world
                .get(body_a.0 as usize)
                .copied()
                .unwrap_or(0)
        } else if body_b != BodyId::NONE {
            self.body_world
                .get(body_b.0 as usize)
                .copied()
                .unwrap_or(0)
        } else {
            0
        };
        self.equality_constraint_world.push(world);
        id
    }

    /// Add a WELD equality constraint: pin BOTH position and
    /// orientation. `relative_rotation` is the rotation that
    /// transforms body A's frame into body B's frame at rest.
    pub fn add_equality_weld(
        &mut self,
        body_a: BodyId,
        body_b: BodyId,
        anchor_a: Vec3,
        anchor_b: Vec3,
        relative_rotation: crate::math::Quat,
    ) -> usize {
        let id = self.add_equality_connect(body_a, body_b, anchor_a, anchor_b);
        self.equality_type[id] = super::EqualityType::Weld as u32;
        self.equality_relative_rotation[id] = relative_rotation;
        id
    }

    /// Add a mimic constraint: forces `q_driven = c0 + c1 · q_driver`.
    /// Both joints must already exist. The Featherstone solver
    /// applies this in a pre-step pass — disabling the driven
    /// joint's dynamics for that DOF.
    pub fn add_mimic_constraint(
        &mut self,
        driven: JointId,
        driver: JointId,
        c0: Real,
        c1: Real,
    ) -> Result<usize> {
        if driven.index() >= self.joint_type.len()
            || driver.index() >= self.joint_type.len()
        {
            return Err(Error::Build(format!(
                "add_mimic_constraint: joint id out of range"
            )));
        }
        let id = self.mimic_joint_a.len();
        self.mimic_joint_a.push(driven.0 as i32);
        self.mimic_joint_b.push(driver.0 as i32);
        self.mimic_c0.push(c0);
        self.mimic_c1.push(c1);
        self.mimic_enabled.push(1);
        let world = self
            .joint_world
            .get(driver.index())
            .copied()
            .unwrap_or(0);
        self.mimic_constraint_world.push(world);
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

    /// Set the global linear-damping coefficient (1/s). Models air
    /// drag / rolling resistance crudely as `v *= exp(-damp · dt)`.
    /// 0.0 disables it; values around 0.05–0.5 give visible slowdown.
    pub fn set_linear_damping(&mut self, damping: Real) {
        self.linear_damping = damping;
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
            let first_joint_idx =
                *self.articulation_start.last().unwrap_or(&0) as usize;
            let world = self
                .joint_world
                .get(first_joint_idx)
                .copied()
                .unwrap_or(0);
            self.articulation_world.push(world);
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

        let particle_count = self.particle_mass.len();
        let spring_count = self.spring_a.len();
        let tri_count = self.tri_indices.len() / 3;

        // Greedy graph colouring of springs: edges are springs, nodes
        // are particles. Two springs share a colour iff they don't
        // share an endpoint. The GPU XPBD kernel then dispatches
        // `K` parallel passes (one per colour); within a colour no
        // two threads write to the same particle, so the per-spring
        // read-modify-write of `particle_q` is race-free without atomics.
        let (spring_color_count, spring_color_offsets, spring_indices_by_color) =
            color_springs(particle_count, &self.spring_a, &self.spring_b);

        Ok(Model {
            particle_count,
            particle_mass: Buffer::from_vec(self.particle_mass, s),
            particle_inv_mass: Buffer::from_vec(self.particle_inv_mass, s),
            particle_radius: Buffer::from_vec(self.particle_radius, s),
            particle_flags: Buffer::from_vec(self.particle_flags, s),
            particle_world: Buffer::from_vec(self.particle_world, s),
            initial_particle_q: Buffer::from_vec(self.particle_q0, s),

            spring_count,
            spring_a: Buffer::from_vec(self.spring_a, s),
            spring_b: Buffer::from_vec(self.spring_b, s),
            spring_rest_length: Buffer::from_vec(self.spring_rest_length, s),
            spring_compliance: Buffer::from_vec(self.spring_compliance, s),
            spring_max_stretch: Buffer::from_vec(self.spring_max_stretch, s),
            spring_color_count,
            spring_color_offsets: Buffer::from_vec(spring_color_offsets, s),
            spring_indices_by_color: Buffer::from_vec(spring_indices_by_color, s),

            tri_count,
            tri_indices: Buffer::from_vec(self.tri_indices, s),

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
            // Flatten ShapeGeometry into GPU-friendly parallel arrays.
            // Encoding matches the doc on `Model::shape_geom_kind` /
            // `_data`. Triangle / TriMesh stay host-only for now (mesh
            // BVH lives on host until stage 11), so their kind is the
            // unsupported sentinel 0xff and the kernel skips them.
            shape_geom_kind: Buffer::from_vec(
                self.shape_geometry
                    .iter()
                    .map(|g| match g {
                        ShapeGeometry::Sphere { .. } => 0u32,
                        ShapeGeometry::Plane { .. } => 1,
                        ShapeGeometry::Box { .. } => 2,
                        ShapeGeometry::Capsule { .. } => 3,
                        ShapeGeometry::Cylinder { .. } => 4,
                        // Cone (5) gets a real kind for GPU GJK;
                        // ConvexHull/Heightfield/Sdf are CPU-only
                        // for now — sentinel 0xff so the GPU
                        // kernels skip them.
                        ShapeGeometry::Cone { .. } => 5,
                        ShapeGeometry::ConvexHull { .. } => 0xff,
                        ShapeGeometry::Heightfield { .. } => 0xff,
                        ShapeGeometry::Sdf { .. } => 0xff,
                        ShapeGeometry::Triangle { .. } => 0xff,
                        ShapeGeometry::TriMesh { .. } => 0xff,
                    })
                    .collect(),
                s,
            ),
            shape_geom_data: Buffer::from_vec(
                self.shape_geometry
                    .iter()
                    .map(|g| match *g {
                        ShapeGeometry::Sphere { radius } => Vec3::new(radius, 0.0, 0.0),
                        ShapeGeometry::Plane { normal } => normal,
                        ShapeGeometry::Box { half_extents } => half_extents,
                        ShapeGeometry::Capsule { half_height, radius } => {
                            Vec3::new(half_height, radius, 0.0)
                        }
                        ShapeGeometry::Cylinder { half_height, radius } => {
                            Vec3::new(half_height, radius, 0.0)
                        }
                        ShapeGeometry::Triangle { .. } => Vec3::ZERO,
                        ShapeGeometry::TriMesh { .. } => Vec3::ZERO,
                        ShapeGeometry::Cone { half_height, radius } => {
                            Vec3::new(half_height, radius, 0.0)
                        }
                        ShapeGeometry::ConvexHull { .. } => Vec3::ZERO,
                        ShapeGeometry::Heightfield { .. } => Vec3::ZERO,
                        ShapeGeometry::Sdf { .. } => Vec3::ZERO,
                    })
                    .collect(),
                s,
            ),
            shape_transform: Buffer::from_vec(self.shape_transform, s),
            shape_geometry: self.shape_geometry,
            shape_flags: Buffer::from_vec(self.shape_flags, s),
            shape_friction: Buffer::from_vec(self.shape_friction, s),
            shape_rolling_friction: Buffer::from_vec(self.shape_rolling_friction, s),
            shape_torsional_friction: Buffer::from_vec(self.shape_torsional_friction, s),
            shape_restitution: Buffer::from_vec(self.shape_restitution, s),
            meshes: self.meshes,
            hulls: self.hulls,
            heightfields: self.heightfields,
            sdfs: self.sdfs,
            shape_collision_group: Buffer::from_vec(self.shape_collision_group, s),
            shape_collision_filter_pairs: {
                let mut p = self.shape_collision_filter_pairs;
                p.sort_unstable();
                p.dedup();
                p
            },
            shape_color: Buffer::from_vec(self.shape_color, s),
            shape_metallic: Buffer::from_vec(self.shape_metallic, s),
            shape_roughness: Buffer::from_vec(self.shape_roughness, s),

            equality_count: self.equality_type.len(),
            equality_type: Buffer::from_vec(self.equality_type, s),
            equality_body_a: Buffer::from_vec(self.equality_body_a, s),
            equality_body_b: Buffer::from_vec(self.equality_body_b, s),
            equality_anchor_a: Buffer::from_vec(self.equality_anchor_a, s),
            equality_anchor_b: Buffer::from_vec(self.equality_anchor_b, s),
            equality_relative_rotation: Buffer::from_vec(
                self.equality_relative_rotation,
                s,
            ),
            equality_enabled: Buffer::from_vec(self.equality_enabled, s),
            equality_constraint_world: Buffer::from_vec(
                self.equality_constraint_world,
                s,
            ),

            mimic_count: self.mimic_joint_a.len(),
            mimic_joint_a: Buffer::from_vec(self.mimic_joint_a, s),
            mimic_joint_b: Buffer::from_vec(self.mimic_joint_b, s),
            mimic_c0: Buffer::from_vec(self.mimic_c0, s),
            mimic_c1: Buffer::from_vec(self.mimic_c1, s),
            mimic_enabled: Buffer::from_vec(self.mimic_enabled, s),
            mimic_constraint_world: Buffer::from_vec(self.mimic_constraint_world, s),

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
            articulation_world: Buffer::from_vec(self.articulation_world, s),
            joint_world: Buffer::from_vec(self.joint_world, s),
            joint_enabled: Buffer::from_vec(self.joint_enabled, s),
            joint_ancestor: Buffer::from_vec(self.joint_ancestor, s),
            joint_target_ke: Buffer::from_vec(self.joint_target_ke, s),

            world_count,
            gravity: Buffer::from_vec(self.world_gravity, s),

            angular_damping: self.angular_damping,
            linear_damping: self.linear_damping,
            particle_velocity_max: self.particle_velocity_max,
            contact_margin: self.contact_margin,
            default_friction: self.default_friction,
            default_restitution: self.default_restitution,
        })
    }
}

/// Greedy graph colouring over the spring endpoint graph. Returns
/// `(color_count, csr_offsets, indices_sorted_by_color)`.
///
/// For each spring in declaration order, picks the smallest colour
/// that no neighbouring spring (sharing endpoint `a` or `b`) is
/// already using. The "neighbouring" lookup uses two scratch arrays
/// indexed by particle id, which makes the algorithm O(spring_count
/// · max_degree) rather than O(spring_count²).
///
/// For an N×N cloth grid (4-neighbour stretch springs + diagonal
/// shears), the natural colouring uses 6 colours and balances the
/// per-colour spring count well — Newton's `solver_xpbd.cu` ships
/// the same approach for the same reason.
fn color_springs(
    particle_count: usize,
    spring_a: &[u32],
    spring_b: &[u32],
) -> (usize, Vec<u32>, Vec<u32>) {
    let n = spring_a.len();
    if n == 0 {
        return (0, vec![0u32], Vec::new());
    }
    // Per-particle adjacency: list of (spring_idx, color) once we've
    // assigned a spring's colour, store the colour against both
    // endpoints. To pick the next spring's colour we scan its two
    // endpoints' colour sets and find the smallest colour absent
    // from the union.
    let mut endpoint_colors: Vec<Vec<u32>> = vec![Vec::new(); particle_count];
    let mut spring_color: Vec<u32> = vec![0u32; n];
    let mut max_color: u32 = 0;

    let mut used = Vec::<bool>::with_capacity(8);
    for i in 0..n {
        let a = spring_a[i] as usize;
        let b = spring_b[i] as usize;
        let in_bounds = a < particle_count && b < particle_count;
        let neighbour_max = if in_bounds {
            endpoint_colors[a].iter().chain(endpoint_colors[b].iter()).copied().max()
        } else {
            None
        };
        let need_size = neighbour_max.map_or(0, |c| c as usize + 2);
        if used.len() < need_size {
            used.resize(need_size, false);
        }
        for u in used.iter_mut() {
            *u = false;
        }
        if in_bounds {
            for &c in &endpoint_colors[a] {
                if (c as usize) < used.len() {
                    used[c as usize] = true;
                }
            }
            for &c in &endpoint_colors[b] {
                if (c as usize) < used.len() {
                    used[c as usize] = true;
                }
            }
        }
        let mut chosen: u32 = 0;
        while (chosen as usize) < used.len() && used[chosen as usize] {
            chosen += 1;
        }
        spring_color[i] = chosen;
        if chosen > max_color {
            max_color = chosen;
        }
        if in_bounds {
            endpoint_colors[a].push(chosen);
            endpoint_colors[b].push(chosen);
        }
    }

    let color_count = (max_color as usize) + 1;
    // Build CSR: count per colour, prefix-sum to offsets, then place.
    let mut counts = vec![0u32; color_count];
    for &c in &spring_color {
        counts[c as usize] += 1;
    }
    let mut offsets = vec![0u32; color_count + 1];
    for k in 0..color_count {
        offsets[k + 1] = offsets[k] + counts[k];
    }
    let mut cursors = offsets[..color_count].to_vec();
    let mut indices = vec![0u32; n];
    for (i, &c) in spring_color.iter().enumerate() {
        let slot = cursors[c as usize];
        indices[slot as usize] = i as u32;
        cursors[c as usize] += 1;
    }
    (color_count, offsets, indices)
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

    /// Joints inherit their child body's world; articulations
    /// inherit their first joint's world. With two pendulums in
    /// different worlds, both indices propagate correctly.
    #[test]
    fn joint_and_articulation_worlds_propagate() {
        let mut b = ModelBuilder::new();
        let l0 = b.add_body(BodyParams::new().with_world(WorldId(0)));
        let _ = b.add_joint_revolute(
            BodyId::NONE,
            l0,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::Z,
        )
        .unwrap();
        b.begin_articulation();
        let l1 = b.add_body(BodyParams::new().with_world(WorldId(2)));
        let _ = b.add_joint_revolute(
            BodyId::NONE,
            l1,
            Transform::IDENTITY,
            Transform::IDENTITY,
            Vec3::Z,
        )
        .unwrap();

        let m = b.build().unwrap();
        assert_eq!(m.articulation_count, 2);
        let aw = m.articulation_world.host().unwrap();
        assert_eq!(aw, &[0i32, 2]);
        let jw = m.joint_world.host().unwrap();
        assert_eq!(jw, &[0i32, 2]);
    }

    #[test]
    fn graph_color_no_endpoint_conflicts() {
        // Triangle springs (0-1, 1-2, 2-0) — every spring shares an
        // endpoint with both others, so the colouring needs 3 colours.
        let a = vec![0u32, 1, 2];
        let b = vec![1u32, 2, 0];
        let (k, offsets, indices) = color_springs(3, &a, &b);
        assert_eq!(k, 3);
        assert_eq!(indices.len(), 3);
        assert_eq!(offsets[k], 3);

        // Within each colour, no two springs share a particle.
        for c in 0..k {
            let lo = offsets[c] as usize;
            let hi = offsets[c + 1] as usize;
            let mut seen = std::collections::HashSet::new();
            for &si in &indices[lo..hi] {
                let pa = a[si as usize];
                let pb = b[si as usize];
                assert!(seen.insert(pa), "color {c} repeats particle {pa}");
                assert!(seen.insert(pb), "color {c} repeats particle {pb}");
            }
        }
    }

    #[test]
    fn graph_color_disjoint_springs_share_color() {
        // Three springs over disjoint particle pairs — all in colour 0.
        let a = vec![0u32, 2, 4];
        let b = vec![1u32, 3, 5];
        let (k, offsets, _indices) = color_springs(6, &a, &b);
        assert_eq!(k, 1);
        assert_eq!(offsets, vec![0u32, 3]);
    }

    #[test]
    fn cloth_grid_includes_bending_springs() {
        // 4×4 cloth: structural = 24, shear = 18, bending = 16.
        // Total = 58. Grid size N=4 has bending count = 2*(N-2)*N = 16.
        let mut b = ModelBuilder::new();
        b.add_cloth_grid(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Z,
            4,
            4,
            0.05,
            0.02,
            1.0e-5,
            0.1,
            crate::core::WorldId(0),
        );
        let m = b.build().unwrap();
        let n = 4_usize;
        let structural = 2 * n * (n - 1);          // 24
        let shear = 2 * (n - 1) * (n - 1);         // 18
        let bending = 2 * (n - 2) * n;             // 16
        assert_eq!(m.spring_count, structural + shear + bending);
    }

    #[test]
    fn cloth_grid_bending_disabled_when_negative() {
        let mut b = ModelBuilder::new();
        b.add_cloth_grid_with_bending(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Z,
            4,
            4,
            0.05,
            0.02,
            1.0e-5,
            0.1,
            -1.0, // disable bending
            crate::core::WorldId(0),
        );
        let m = b.build().unwrap();
        let n = 4_usize;
        let structural = 2 * n * (n - 1);
        let shear = 2 * (n - 1) * (n - 1);
        assert_eq!(m.spring_count, structural + shear);
    }

    #[test]
    fn graph_color_cloth_grid_few_colors() {
        // Tiny 4×4 grid with the standard cloth springs (4-neighbour
        // stretch + diagonal shears). The grid is regular so we should
        // get a single-digit colour count, not anywhere near
        // `spring_count`.
        let mut b = ModelBuilder::new();
        b.add_cloth_grid(
            Vec3::ZERO,
            Vec3::X,
            Vec3::Z,
            4,
            4,
            0.05,
            0.02,
            1.0e-5,
            0.1,
            crate::core::WorldId(0),
        );
        let m = b.build().unwrap();
        // 4×4 grid with structural + shear + bending springs needs
        // ~10 colours under the greedy algorithm — far better than
        // `spring_count` (58), and Newton's `solver_xpbd` reports
        // similar numbers for the same topology.
        assert!(
            m.spring_color_count <= 16,
            "cloth colouring blew up: {} colours",
            m.spring_color_count
        );
        assert_eq!(
            m.spring_indices_by_color.host().unwrap().len(),
            m.spring_count
        );
    }
}
