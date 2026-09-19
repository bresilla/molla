//! `ModelBuilder` — ergonomic authoring surface for [`Model`].
//!
//! Stage 1 surface: rigid bodies, sphere shapes, per-world gravity. More
//! arrives in stages 3 (joints) and 4 (other shapes).

use molla_core::{BodyId, Error, JointId, Result, ShapeId, WorldId};
use molla_math::{Mat3, Real, SpatialInertia, Transform, Vec3};
use molla_storage::{Buffer, BufferUsage};

use super::Model;
use super::flags::BodyFlags;

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
    /// shape-local **Y axis**.
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
    /// Y-axis cylinder dilated by a sphere of `border_radius`.
    RoundCylinder {
        half_height: Real,
        radius: Real,
        border_radius: Real,
    },
    /// Triangle in the shape's local frame. Three vertex positions.
    /// Used as a primitive for narrow-phase queries against
    /// individual mesh triangles; the user typically builds meshes
    /// via [`super::Model::meshes`] and references them with
    /// [`Self::TriMesh`]. Volume-degenerate — narrow phase routes
    /// triangle-vs-volume pairs through GJK+EPA.
    Triangle { v0: Vec3, v1: Vec3, v2: Vec3 },
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
    /// `(origin.x + i·cell_size, origin.y + height, origin.z + j·cell_size_z)`.
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
    /// Solid ellipsoid centred on the shape-local origin with
    /// per-axis half-extents (semi-axes) `half_axes.x/y/z`.
    /// Equation in the shape-local frame: `Σ (xᵢ/aᵢ)² ≤ 1`.
    /// Reduces to a sphere when all three components are equal.
    /// Stage 30: closed-form vs Sphere/Plane, GJK + EPA fallback
    /// against everything else (uses `support_ellipsoid`).
    Ellipsoid {
        /// Per-axis semi-axes `(a, b, c)` in shape-local frame (m).
        /// All three must be strictly positive.
        half_axes: Vec3,
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

/// Per-tet Neo-Hookean material parameters (Stage 20.3 multi-material
/// extension). One material applies to one or more tets via the
/// `material_idx` slice passed to
/// [`ModelBuilder::add_tet_mesh_per_material`]. Convert to the per-tet
/// Lamé `(μ, λ)` pair via the standard formulas at build time.
///
/// `yield_stress` + `hardening` are the v1 plasticity knobs:
/// - `yield_stress = Real::INFINITY` (or just `1e30`) → purely elastic
///   (the return-mapping pass is a no-op for that tet).
/// - `hardening = 0` → perfectly plastic (yield surface stays at the
///   initial radius).
/// - `hardening > 0` → linear isotropic hardening
///   (`σ_y(ε_p) = σ_y0 + H · ε_p`).
#[derive(Copy, Clone, Debug)]
pub struct MaterialNeoHookean {
    /// Young's modulus (Pa). Clamped to `[1, 1e9]` at build time.
    pub youngs: Real,
    /// Poisson ratio. Clamped to `[-0.49, 0.49]` to keep the Lamé
    /// `λ` finite (ν → 0.5 is the incompressible limit).
    pub poisson: Real,
    /// von Mises yield stress `σ_y0` (Pa). Set to `Real::INFINITY` to
    /// keep the material purely elastic.
    pub yield_stress: Real,
    /// Linear hardening modulus `H` (Pa). 0 = perfectly plastic.
    pub hardening: Real,
    /// Stage 20.4 — kinematic-hardening (back-stress) modulus `c`
    /// (Pa). When `c > 0`, the yield function shifts from
    /// `‖dev(τ)‖` to `‖dev(τ) - α‖` and `α` evolves per the Prager
    /// rule; produces the Bauschinger effect under cyclic loading
    /// (forward yielding lowers the reverse yield). `c = 0` reduces
    /// to pure isotropic hardening (Stage 20.3 back-compat).
    pub kinematic_hardening: Real,
}

impl MaterialNeoHookean {
    /// Convenience for the elastic-only case (no plasticity).
    #[must_use]
    pub fn elastic(youngs: Real, poisson: Real) -> Self {
        Self {
            youngs,
            poisson,
            yield_stress: Real::INFINITY,
            hardening: 0.0,
            kinematic_hardening: 0.0,
        }
    }

    /// Stage 20.4 — convenience for plastic with isotropic hardening
    /// only (back-compat with Stage 20.3 four-arg construction).
    #[must_use]
    pub fn plastic(youngs: Real, poisson: Real, yield_stress: Real, hardening: Real) -> Self {
        Self {
            youngs,
            poisson,
            yield_stress,
            hardening,
            kinematic_hardening: 0.0,
        }
    }

    /// Stage 20.4 — full plastic-with-kinematic-hardening recipe.
    #[must_use]
    pub fn plastic_kinematic(
        youngs: Real,
        poisson: Real,
        yield_stress: Real,
        hardening: Real,
        kinematic_hardening: Real,
    ) -> Self {
        Self {
            youngs,
            poisson,
            yield_stress,
            hardening,
            kinematic_hardening,
        }
    }
}

impl Default for MaterialNeoHookean {
    fn default() -> Self {
        Self::elastic(1.0e5, 0.4)
    }
}

/// Stable identifier for a tetrahedral soft body added via
/// [`ModelBuilder::add_tet_mesh`]. 0-based dense index into the
/// model's `tet_mesh_offsets` CSR. Wrapped for type safety so a
/// stray `usize` can't be passed where a tet mesh id is expected.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct TetMeshId(pub u32);

impl TetMeshId {
    /// Raw 0-based index.
    #[must_use]
    pub fn index(self) -> usize {
        self.0 as usize
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

    // Cable polylines (Stage 20.1) — book-keeping over groups of CABLE
    // particles + segment springs that the builder also pushes into the
    // generic spring arrays above. CSR-encoded so a model can hold many
    // independent cables without allocating per-cable Vecs.
    cable_particle_offsets: Vec<u32>,
    cable_particles: Vec<u32>,
    cable_rest_length: Vec<Real>,
    cable_stiffness: Vec<Real>,
    cable_bending_stiffness: Vec<Real>,

    // Tetrahedra (Stage 20.2 volumetric soft bodies). One TetMesh =
    // one contiguous slice of `tet_indices` (4 ids per tet) with
    // shared Lamé material parameters per element. Mirrors `cable_*`
    // shape — CSR-encoded so a model can hold many independent
    // tet meshes without allocating per-mesh Vecs.
    tet_indices: Vec<u32>,
    tet_dm_inv: Vec<Mat3>,
    tet_rest_volume: Vec<Real>,
    tet_mu: Vec<Real>,
    tet_lambda: Vec<Real>,
    // Stage 20.3 plasticity. F_p starts as identity for every tet;
    // accumulated plastic strain + per-tet yield/hardening stay zero
    // for the back-compat path (yield_stress = INFINITY ⇒ purely
    // elastic). The per-material API in `add_tet_mesh_per_material`
    // surfaces both knobs.
    tet_f_plastic: Vec<Mat3>,
    tet_plastic_strain: Vec<Real>,
    tet_yield_stress: Vec<Real>,
    tet_hardening: Vec<Real>,
    /// Stage 20.4 — per-tet kinematic-hardening modulus `c` (Pa).
    tet_kinematic_hardening: Vec<Real>,
    tet_mesh_offsets: Vec<u32>,
    /// Stage 33 — per-tet-mesh first-particle index, captured when the
    /// builder added the mesh (every tet mesh's particles live in a
    /// contiguous range, since `add_particle` is monotonic between
    /// `add_tet_mesh` calls). Length == `tet_mesh_offsets.len() - 1`.
    /// Used at build time to compute boundary triangles + the soft
    /// surface registry entries.
    tet_mesh_vertex_first: Vec<u32>,
    /// Stage 33 — per-tet-mesh vertex count (max(tet vertex idx) - first + 1
    /// for every tet in the mesh). Stored explicitly so the registry
    /// can encode the shared particle pool slice without re-deriving.
    tet_mesh_vertex_count: Vec<u32>,

    // Stage 33 — cloth mesh registry (mirrors `tet_mesh_offsets`):
    // `add_cloth_grid*` pushes one entry per cloth, recording its
    // particle pool slice and the first/last tri index it contributed
    // to `tri_indices`.
    cloth_mesh_first_particle: Vec<u32>,
    cloth_mesh_particle_count: Vec<u32>,
    /// Tri offset into the global flat tri list (`tri_indices`)
    /// for each cloth mesh. Length == cloth_mesh_first_particle.len() + 1.
    cloth_mesh_tri_offsets: Vec<u32>,

    // Stage 33.3 — per-mesh "self-collision enabled" toggle.
    tet_mesh_self_collision: Vec<bool>,
    cloth_mesh_self_collision: Vec<bool>,
    // Stage 33.5 — per-mesh friction/restitution overrides for the
    // soft-surface registry.
    tet_mesh_friction: Vec<Real>,
    tet_mesh_restitution: Vec<Real>,
    cloth_mesh_friction: Vec<Real>,
    cloth_mesh_restitution: Vec<Real>,

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
    shape_contact_material: Vec<super::ContactMaterial>,
    shape_rolling_friction: Vec<Real>,
    shape_torsional_friction: Vec<Real>,
    shape_restitution: Vec<Real>,
    meshes: Vec<super::TriMesh>,
    hulls: Vec<Vec<Vec3>>,
    heightfields: Vec<super::Heightfield>,
    sdfs: Vec<super::SdfShape>,
    shape_collision_group: Vec<u32>,
    shape_collision_mask: Vec<u32>,
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
    equality_relative_rotation: Vec<molla_math::Quat>,
    equality_enabled: Vec<u32>,
    equality_constraint_world: Vec<i32>,
    // Joint-mimic side-tables (Stage 32) — per-equality driver/driven
    // joint + 5-coeff polynomial. Length matches `equality_type`; the
    // entries are only consulted when the constraint's type is
    // `EqualityType::JointMimic`. Non-mimic slots store `(-1, -1, [0; 5])`.
    equality_mimic_driver_joint: Vec<i32>,
    equality_mimic_driven_joint: Vec<i32>,
    equality_mimic_poly: Vec<Real>, // 5 entries per equality, flat

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
    joint_compliance: Vec<crate::joint_control::JointCompliance>,

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
    joint_reversed: Vec<u32>,
    /// Per-joint ancestor index in the articulation tree.
    joint_ancestor: Vec<i32>,
    /// Per-DOF static joint stiffness (default 0).
    joint_target_ke: Vec<Real>,
    /// Per-joint linear (stretch) stiffness for CABLE joints (N/m).
    /// Length = joint_count; 0 for non-cable joints.
    joint_cable_stiffness_linear: Vec<Real>,
    /// Per-joint bend stiffness for CABLE joints (N·m/rad).
    joint_cable_stiffness_bend: Vec<Real>,
    /// Per-joint linear (stretch) damping for CABLE joints (N·s/m).
    joint_cable_damping_linear: Vec<Real>,
    /// Per-joint bend damping for CABLE joints (N·m·s/rad).
    joint_cable_damping_bend: Vec<Real>,

    // Stage 42 v2 — per-muscle Hill-type parameters. Parallel arrays
    // of length `muscle_count`. Each muscle drives one CABLE joint's
    // linear stretch DoF (see `crate::actuators::muscle`). v2 allows
    // multiple muscles per joint (co-contraction); the CSR
    // `joint_muscle_offsets` / `joint_muscle_indices` is constructed
    // at `build()` from `muscle_joint`.
    muscle_joint: Vec<i32>,
    muscle_f_max: Vec<Real>,
    muscle_l_opt: Vec<Real>,
    muscle_v_max: Vec<Real>,
    muscle_activation: Vec<Real>,
    muscle_excitation: Vec<Real>,
    muscle_tau_activation: Vec<Real>,
    muscle_l_tendon_slack: Vec<Real>,
    muscle_tendon_stiffness_ratio: Vec<Real>,
    muscle_pennation_angle: Vec<Real>,
    muscle_polarity: Vec<Real>,

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
    ///
    /// # Examples
    ///
    /// ```
    /// use molla_math::Transform;
    /// use molla_sim::{BodyParams, ModelBuilder};
    ///
    /// let mut b = ModelBuilder::new();
    /// let body = b.add_body(BodyParams::new());
    /// b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    /// let model = b.build().unwrap();
    /// assert_eq!(model.body_count, 1);
    /// ```
    #[must_use]
    pub fn new() -> Self {
        Self {
            world_gravity: vec![Vec3::new(0.0, -molla_math::GRAVITY, 0.0)],
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
            let first_joint_idx = *self.articulation_start.last().unwrap_or(&0) as usize;
            let world = self.joint_world.get(first_joint_idx).copied().unwrap_or(0);
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
                .push(Vec3::new(0.0, -molla_math::GRAVITY, 0.0));
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

    /// Build a CABLE polyline through the given particle indices.
    ///
    /// Wires a chain of [`Self::add_distance_constraint`] segments
    /// between consecutive particles, derives each segment's rest
    /// length from the current particle positions, and registers the
    /// polyline as cable `cable_id`. Cable bookkeeping (rest length,
    /// stiffness, polyline indices) is stored on the Model so the
    /// VBD solver / Dahl friction kernel / viewer can address "the
    /// i-th cable" as one logical unit.
    ///
    /// The CABLE primitive in Stage 20.1 is **stretch-only**: the
    /// individual segment springs handle inextensibility via XPBD
    /// compliance (`α = 1/stiffness`). Bending is deferred to
    /// Stage 20.2 (it needs an extra angle-pair adjacency that the
    /// per-vertex VBD block already supports but the builder doesn't
    /// emit yet).
    ///
    /// * `particles` — polyline indices, in order. Must have ≥ 2 entries.
    /// * `stiffness` — per-segment stretch stiffness (N/m). Mapped onto
    ///   XPBD compliance as `α = 1 / max(stiffness, 1.0)`.
    /// * `max_stretch` — per-segment hard stretch cap (ratio of rest
    ///   length). 0.0 = perfectly inextensible after the soft pass.
    ///
    /// Returns the new cable id (0-based index).
    pub fn add_cable(&mut self, particles: &[u32], stiffness: Real, max_stretch: Real) -> usize {
        assert!(
            particles.len() >= 2,
            "add_cable: polyline needs >= 2 particle indices"
        );
        let cable_id = self.cable_rest_length.len();
        let compliance = if stiffness > 1.0 {
            1.0 / stiffness
        } else {
            1.0
        };
        let mut total_rest = 0.0_f64;
        for win in particles.windows(2) {
            let a = win[0];
            let b = win[1];
            let pa = self.particle_q0[a as usize];
            let pb = self.particle_q0[b as usize];
            let rest = (pa - pb).length();
            total_rest += rest;
            self.add_distance_constraint(a, b, rest, compliance, max_stretch);
        }
        let off_lo = self.cable_particles.len() as u32;
        for &p in particles {
            self.cable_particles.push(p);
        }
        let off_hi = self.cable_particles.len() as u32;
        // Lazily seed the offsets array with the leading 0 sentinel.
        if self.cable_particle_offsets.is_empty() {
            self.cable_particle_offsets.push(0);
        }
        debug_assert_eq!(*self.cable_particle_offsets.last().unwrap(), off_lo);
        self.cable_particle_offsets.push(off_hi);
        self.cable_rest_length.push(total_rest);
        self.cable_stiffness.push(stiffness);
        // Stage 20.2 bending stiffness defaults to 0 (pure stretch
        // polyline, Stage 20.1 behaviour). Use
        // [`Self::set_cable_bending_stiffness`] to enable bending.
        self.cable_bending_stiffness.push(0.0);
        cable_id
    }

    /// Convenience: add a straight-line CABLE polyline of `n` particles
    /// from `start` to `end` with uniform spacing, masses, and radius.
    /// All particles are unpinned by default — call [`Self::pin_particle`]
    /// on the returned indices to anchor end points.
    ///
    /// Returns `(first_particle, n, cable_id)`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_cable_polyline(
        &mut self,
        start: Vec3,
        end: Vec3,
        n: u32,
        particle_mass: Real,
        particle_radius: Real,
        stiffness: Real,
        max_stretch: Real,
        world: WorldId,
    ) -> (u32, u32, usize) {
        let n = n.max(2);
        let first = self.particle_mass.len() as u32;
        for i in 0..n {
            let t = (i as Real) / ((n - 1) as Real);
            let p = start + (end - start) * t;
            self.add_particle(p, particle_mass, particle_radius, world);
        }
        let ids: Vec<u32> = (first..first + n).collect();
        let cable_id = self.add_cable(&ids, stiffness, max_stretch);
        (first, n, cable_id)
    }

    /// Set the bending stiffness `k_b` (N·m) on cable `cable_id`. The
    /// VBD step then folds a discrete bending energy
    /// `E_b = ½ · k_b · (2·sin(θ/2))²` for every interior triple
    /// `(v_{i-1}, v_i, v_{i+1})` along the polyline. `0` (the default)
    /// disables bending — the cable behaves as a pure stretch
    /// polyline (Stage 20.1).
    pub fn set_cable_bending_stiffness(&mut self, cable_id: usize, bending_stiffness: Real) {
        if cable_id < self.cable_bending_stiffness.len() {
            self.cable_bending_stiffness[cable_id] = bending_stiffness.max(0.0);
        }
    }

    /// Add a tetrahedral mesh sharing the model's particle pool.
    ///
    /// Each `tets[i]` is `[v0, v1, v2, v3]` — particle indices forming
    /// one tet. The builder pre-computes `Dm⁻¹` (inverse rest-state
    /// basis) and the rest volume per element; the VBD per-vertex
    /// block solve then folds in a Neo-Hookean stress + Hessian
    /// contribution from each incident tet.
    ///
    /// * `youngs_modulus` — Young's modulus `E` (Pa = N/m²). Stiff
    ///   rubber ≈ 5e6 Pa, jelly ≈ 1e4 Pa, gel ≈ 1e3 Pa. Capped at
    ///   1e9 to keep the per-tet Hessian condition number sane.
    /// * `poisson_ratio` — `ν ∈ [0, 0.5)`. 0.4 is a common default
    ///   for elastic rubber-like material; 0.49+ approaches
    ///   incompressibility but stiffens the volumetric Hessian
    ///   significantly. Clamped to `[-0.49, 0.49]` so `(1 + ν)(1 -
    ///   2ν)` doesn't blow up.
    /// * `rest_volume` — optional override for per-element rest volume
    ///   (m³). When `None`, the builder computes `det(Dm) / 6`. Pass
    ///   `Some(...)` for rest configurations that aren't represented
    ///   in `particle_q0` (rare; useful for plasticity Stage 20.3).
    ///
    /// Returns the new tet mesh id (0-based index).
    ///
    /// # Errors
    ///
    /// Returns `Error::Build` when:
    /// - any vertex index is out of range,
    /// - the rest configuration produces a negative or zero `det(Dm)`
    ///   (degenerate / inverted tet),
    /// - `tets.is_empty()`.
    pub fn add_tet_mesh(
        &mut self,
        tets: &[[u32; 4]],
        youngs_modulus: Real,
        poisson_ratio: Real,
        rest_volume: Option<&[Real]>,
    ) -> Result<TetMeshId> {
        // Single-material shim: every tet shares the same material slot.
        let materials = [MaterialNeoHookean::elastic(youngs_modulus, poisson_ratio)];
        let material_idx: Vec<u32> = vec![0; tets.len()];
        self.add_tet_mesh_per_material(tets, &materials, &material_idx, rest_volume)
    }

    /// Multi-material extension of [`Self::add_tet_mesh`] (Stage 20.3).
    ///
    /// Each tet picks its material by indexing into `materials` via the
    /// matching `material_idx[i]`. This lets a single `TetMeshId` mix
    /// soft + stiff regions (jelly cube with stiff corners) and turn
    /// plasticity on/off per element. Returns the new tet mesh id.
    ///
    /// # Errors
    ///
    /// Returns `Error::Build` when:
    /// - `tets.is_empty()`,
    /// - `material_idx.len() != tets.len()`,
    /// - any `material_idx[i] >= materials.len()`,
    /// - any vertex index is out of range,
    /// - the rest configuration produces a non-positive `det(Dm)`.
    pub fn add_tet_mesh_per_material(
        &mut self,
        tets: &[[u32; 4]],
        materials: &[MaterialNeoHookean],
        material_idx: &[u32],
        rest_volume: Option<&[Real]>,
    ) -> Result<TetMeshId> {
        if tets.is_empty() {
            return Err(Error::Build("add_tet_mesh: tets must be non-empty".into()));
        }
        if materials.is_empty() {
            return Err(Error::Build(
                "add_tet_mesh_per_material: materials must be non-empty".into(),
            ));
        }
        if material_idx.len() != tets.len() {
            return Err(Error::Build(format!(
                "add_tet_mesh_per_material: material_idx.len() = {} ≠ tets.len() = {}",
                material_idx.len(),
                tets.len()
            )));
        }
        if let Some(rv) = rest_volume
            && rv.len() != tets.len()
        {
            return Err(Error::Build(format!(
                "add_tet_mesh: rest_volume.len() = {} ≠ tets.len() = {}",
                rv.len(),
                tets.len()
            )));
        }
        let n_particles = self.particle_mass.len() as u32;

        // Pre-compute Lamé `(μ, λ)` for each material. Clamp ν to keep
        // `(1+ν)(1-2ν)` finite; clamp E to a sane range.
        let mut lame: Vec<(Real, Real, Real, Real, Real)> = Vec::with_capacity(materials.len());
        for m in materials {
            let nu = m.poisson.clamp(-0.49, 0.49);
            let e = m.youngs.max(1.0).min(1.0e9);
            let mu = e / (2.0 * (1.0 + nu));
            let lambda = e * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
            // Yield stress / hardening: keep INFINITY as the "purely
            // elastic" sentinel; downstream kernels short-circuit when
            // the trial deviator norm is below σ_y.
            let yield_stress = if m.yield_stress.is_finite() {
                m.yield_stress.max(0.0)
            } else {
                Real::INFINITY
            };
            let hardening = m.hardening.max(0.0);
            // Stage 20.4 — kinematic-hardening modulus `c`. Clamped to
            // ≥ 0 (negative `c` is unphysical and would destabilise the
            // back-stress evolution).
            let kin_hard = m.kinematic_hardening.max(0.0);
            lame.push((mu, lambda, yield_stress, hardening, kin_hard));
        }

        let mesh_id = self.tet_mesh_offsets.len().saturating_sub(1);
        if self.tet_mesh_offsets.is_empty() {
            self.tet_mesh_offsets.push(0);
        }
        // Stage 33 — track this mesh's vertex range so the
        // soft-surface registry can encode it without re-scanning.
        // We compute (min, max) over the tet vertex ids; for
        // contiguous allocations (the canonical pattern) this equals
        // (first_particle, first_particle + n_verts - 1).
        let mut vmin: u32 = u32::MAX;
        let mut vmax: u32 = 0;
        for t in tets {
            for &v in t {
                if v < vmin {
                    vmin = v;
                }
                if v > vmax {
                    vmax = v;
                }
            }
        }
        for (i, t) in tets.iter().enumerate() {
            for &v in t {
                if v >= n_particles {
                    return Err(Error::Build(format!(
                        "add_tet_mesh: tet {} references particle {} ≥ particle_count {}",
                        i, v, n_particles
                    )));
                }
            }
            let mi = material_idx[i] as usize;
            if mi >= materials.len() {
                return Err(Error::Build(format!(
                    "add_tet_mesh_per_material: tet {i} material_idx {mi} ≥ materials.len() = {}",
                    materials.len()
                )));
            }
            let v0 = self.particle_q0[t[0] as usize];
            let v1 = self.particle_q0[t[1] as usize];
            let v2 = self.particle_q0[t[2] as usize];
            let v3 = self.particle_q0[t[3] as usize];
            let dm = Mat3::from_cols(v1 - v0, v2 - v0, v3 - v0);
            let det = dm.determinant();
            if det.abs() < 1.0e-20 {
                return Err(Error::Build(format!(
                    "add_tet_mesh: tet {i} is degenerate (det(Dm) = {det:e})"
                )));
            }
            if det <= 0.0 {
                return Err(Error::Build(format!(
                    "add_tet_mesh: tet {i} has non-positive rest volume \
                     (det(Dm) = {det:e}); reorder its vertices CCW"
                )));
            }
            let dm_inv = dm.inverse();
            let rest_vol = match rest_volume {
                Some(rv) => rv[i].max(1.0e-12),
                None => det / 6.0,
            };
            for &v in t {
                self.tet_indices.push(v);
            }
            self.tet_dm_inv.push(dm_inv);
            self.tet_rest_volume.push(rest_vol);
            let (mu, lambda, yield_s, hard, kin_hard) = lame[mi];
            self.tet_mu.push(mu);
            self.tet_lambda.push(lambda);
            // Plasticity init: F_p = I, ε_p = 0, σ_y / H from material.
            self.tet_f_plastic.push(Mat3::IDENTITY);
            self.tet_plastic_strain.push(0.0);
            self.tet_yield_stress.push(yield_s);
            self.tet_hardening.push(hard);
            // Stage 20.4 — kinematic-hardening modulus c.
            self.tet_kinematic_hardening.push(kin_hard);
        }
        // Push the closing CSR offset (next mesh starts here).
        self.tet_mesh_offsets
            .push(self.tet_indices.len() as u32 / 4);
        // Stage 33 — record this mesh's vertex range. Vertex count =
        // `vmax - vmin + 1` (max(vid) - min(vid) + 1) under the
        // contiguous-allocation assumption that every builder caller
        // honours by allocating the mesh's particles in one block.
        let v_first = vmin;
        let v_count = if vmax >= vmin { vmax - vmin + 1 } else { 0 };
        self.tet_mesh_vertex_first.push(v_first);
        self.tet_mesh_vertex_count.push(v_count);
        // Stage 33.3 — default this mesh to "no self-collision".
        // Callers opt in via `set_tet_mesh_self_collision` or the
        // post-build `Model::set_soft_surface_self_collision` API.
        self.tet_mesh_self_collision.push(false);
        Ok(TetMeshId(mesh_id as u32))
    }

    /// Stage 33.3 — toggle self-collision for the named tet mesh.
    /// Defaults to `false` (no self-contact, matches Stage 33's
    /// cross-mesh-only behaviour). Out-of-range ids are silently
    /// ignored — the typical caller pattern is "build → maybe set →
    /// finalise" so the flag-set call only matters when a tet mesh
    /// was actually added.
    pub fn set_tet_mesh_self_collision(&mut self, id: TetMeshId, enable: bool) {
        let idx = id.index();
        if idx < self.tet_mesh_self_collision.len() {
            self.tet_mesh_self_collision[idx] = enable;
        }
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
        // Stage 33 — record this cloth's tri offset BEFORE the
        // builder pushes any new tris. We close the offset at the
        // bottom of the function so the registry can slice
        // `tri_indices[tri_off_start..end]`.
        let tri_off_start = (self.tri_indices.len() / 3) as u32;

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

        // Stage 33 — push this cloth into the cloth registry. Vertex
        // count = nx · ny (every grid sample is a particle); tri
        // range = [tri_off_start, current_tri_count).
        let v_count = nx * ny;
        let tri_off_end = (self.tri_indices.len() / 3) as u32;
        if self.cloth_mesh_tri_offsets.is_empty() {
            self.cloth_mesh_tri_offsets.push(tri_off_start);
        } else {
            // The previous cloth mesh's closing offset must equal this
            // mesh's opening offset, or the registry slicing
            // `tri_indices[offsets[m]*3 .. offsets[m+1]*3]` silently
            // reads the wrong triangle range. This breaks only if some
            // other path (e.g. `add_triangle`) pushed triangles
            // between two `add_cloth_grid` calls — fail loudly in
            // debug builds instead of corrupting the registry.
            debug_assert_eq!(
                tri_off_start,
                *self.cloth_mesh_tri_offsets.last().unwrap(),
                "add_cloth_grid: triangles were pushed between cloth meshes; \
                 the cloth-mesh tri-offset registry would be corrupted"
            );
        }
        self.cloth_mesh_tri_offsets.push(tri_off_end);
        self.cloth_mesh_first_particle.push(first);
        self.cloth_mesh_particle_count.push(v_count);
        // Stage 33.3 — default cloth meshes to "no self-collision".
        self.cloth_mesh_self_collision.push(false);

        (first, nx, ny)
    }

    /// Stage 33.3 — toggle self-collision for the named cloth mesh.
    pub fn set_cloth_mesh_self_collision(&mut self, cloth_mesh_id: usize, enable: bool) {
        if cloth_mesh_id < self.cloth_mesh_self_collision.len() {
            self.cloth_mesh_self_collision[cloth_mesh_id] = enable;
        }
    }

    /// Stage 33.5 — set the Coulomb friction coefficient for a tet
    /// soft surface. Two contacting surfaces' values combine via the
    /// geometric mean. Default is 0.4.
    pub fn set_tet_mesh_friction(&mut self, mesh: crate::TetMeshId, mu: Real) {
        let i = mesh.0 as usize;
        while self.tet_mesh_friction.len() <= i {
            self.tet_mesh_friction.push(0.4);
        }
        self.tet_mesh_friction[i] = mu.max(0.0);
    }

    /// Stage 33.5 — set the restitution coefficient (0..=1) for a tet
    /// soft surface. Default is 0.1.
    pub fn set_tet_mesh_restitution(&mut self, mesh: crate::TetMeshId, e: Real) {
        let i = mesh.0 as usize;
        while self.tet_mesh_restitution.len() <= i {
            self.tet_mesh_restitution.push(0.1);
        }
        self.tet_mesh_restitution[i] = e.clamp(0.0, 1.0);
    }

    /// Stage 33.5 — set the Coulomb friction coefficient for a cloth
    /// soft surface. Default is 0.4.
    pub fn set_cloth_friction(&mut self, cloth: crate::ClothMeshId, mu: Real) {
        let i = cloth.0 as usize;
        while self.cloth_mesh_friction.len() <= i {
            self.cloth_mesh_friction.push(0.4);
        }
        self.cloth_mesh_friction[i] = mu.max(0.0);
    }

    /// Stage 33.5 — set the restitution coefficient for a cloth soft
    /// surface. Default is 0.0.
    pub fn set_cloth_restitution(&mut self, cloth: crate::ClothMeshId, e: Real) {
        let i = cloth.0 as usize;
        while self.cloth_mesh_restitution.len() <= i {
            self.cloth_mesh_restitution.push(0.0);
        }
        self.cloth_mesh_restitution[i] = e.clamp(0.0, 1.0);
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
                .push(Vec3::new(0.0, -molla_math::GRAVITY, 0.0));
        }

        let is_kinematic = (params.flags & BodyFlags::IMMOVABLE) != 0;
        let inv_mass = if params.mass > 0.0 && !is_kinematic {
            1.0 / params.mass
        } else {
            0.0
        };
        let inv_inertia = if params.mass > 0.0 && params.inertia != Mat3::ZERO && !is_kinematic {
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
    /// shape-local +Y direction. `half_height` is
    /// the half-length of the cylindrical section (cap centres are
    /// at `±half_height·Y` in shape-local frame); `radius` is the
    /// cap + cylinder radius. Zero half-height gives a sphere.
    pub fn add_shape_capsule(
        &mut self,
        body: BodyId,
        half_height: Real,
        radius: Real,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_capsule")?;
        if !half_height.is_finite() || half_height < 0.0 {
            return Err(Error::Build(format!(
                "add_shape_capsule: half_height must be nonnegative finite, got {half_height}"
            )));
        }
        if !radius.is_finite() || radius <= 0.0 {
            return Err(Error::Build(format!(
                "add_shape_capsule: radius must be positive finite, got {radius}"
            )));
        }
        if half_height == 0.0 {
            return self.add_shape_sphere(body, radius, local_pose);
        }
        Ok(self.push_shape(
            body,
            local_pose,
            ShapeGeometry::Capsule {
                half_height,
                radius,
            },
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
            ShapeGeometry::Cylinder {
                half_height,
                radius,
            },
        ))
    }

    /// Attach a cylinder dilated outward by `border_radius` (CPU collision).
    pub fn add_shape_round_cylinder(
        &mut self,
        body: BodyId,
        half_height: Real,
        radius: Real,
        border_radius: Real,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_round_cylinder")?;
        if !half_height.is_finite()
            || half_height <= 0.0
            || !radius.is_finite()
            || radius <= 0.0
            || !border_radius.is_finite()
            || border_radius < 0.0
            || !(half_height + border_radius).is_finite()
            || !(radius + border_radius).is_finite()
        {
            return Err(Error::Build("invalid rounded cylinder dimensions".into()));
        }
        Ok(self.push_shape(
            body,
            local_pose,
            ShapeGeometry::RoundCylinder {
                half_height,
                radius,
                border_radius,
            },
        ))
    }

    /// Register a triangle mesh in the model and return its
    /// `mesh_id`. Pass that id into [`Self::add_shape_trimesh`].
    pub fn add_mesh(&mut self, mesh: super::TriMesh) -> u32 {
        // Mesh acceleration is owned by geometry, not simulation. Keep the
        // model raw so `molla-sim` can become independent of `molla-geometry`.
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
        Ok(self.push_shape(
            body,
            local_pose,
            ShapeGeometry::Cone {
                half_height,
                radius,
            },
        ))
    }

    /// Register a heightfield (regular grid of height samples) and
    /// return its `field_id`. The grid sits at `origin` in shape-local
    /// frame with independent X/Z sample spacing.
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
        self.heightfields[field_id as usize].validate()?;
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

    /// Attach an ellipsoid shape to `body`. `half_axes` are the
    /// per-axis semi-axes (must all be strictly positive). Reduces
    /// to a sphere when all three components are equal — but the
    /// dispatch always routes through the ellipsoid path (a sphere
    /// would normally be the more efficient call).
    pub fn add_shape_ellipsoid(
        &mut self,
        body: BodyId,
        half_axes: Vec3,
        local_pose: Transform,
    ) -> Result<ShapeId> {
        self.validate_body_for_shape(body, "add_shape_ellipsoid")?;
        if !half_axes.x.is_finite()
            || !half_axes.y.is_finite()
            || !half_axes.z.is_finite()
            || half_axes.x <= 0.0
            || half_axes.y <= 0.0
            || half_axes.z <= 0.0
        {
            return Err(Error::Build(format!(
                "add_shape_ellipsoid: half_axes must be positive finite, got {half_axes:?}"
            )));
        }
        Ok(self.push_shape(body, local_pose, ShapeGeometry::Ellipsoid { half_axes }))
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
        Ok(self.push_shape(body, local_pose, ShapeGeometry::Triangle { v0, v1, v2 }))
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
        let body_idx = if body == BodyId::NONE {
            -1
        } else {
            body.0 as i32
        };
        self.shape_body.push(body_idx);
        self.shape_transform.push(local_pose);
        self.shape_geometry.push(geom);
        self.shape_flags.push(super::flags::ShapeFlags::COLLIDES);
        self.shape_friction.push(Real::NAN);
        self.shape_contact_material
            .push(super::ContactMaterial::default());
        self.shape_rolling_friction.push(Real::NAN);
        self.shape_torsional_friction.push(Real::NAN);
        self.shape_restitution.push(Real::NAN);
        self.shape_collision_group.push(0xffff_ffff);
        self.shape_collision_mask.push(0xffff_ffff);
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
    pub fn set_shape_collision_group(&mut self, shape: ShapeId, group: u32) -> Result<()> {
        let i = shape.index();
        if i >= self.shape_collision_group.len() {
            return Err(Error::Build(format!(
                "set_shape_collision_group: shape {shape:?} out of range"
            )));
        }
        self.shape_collision_group[i] = group;
        self.shape_collision_mask[i] = group;
        Ok(())
    }

    pub fn set_shape_collision_groups(
        &mut self,
        shape: ShapeId,
        memberships: u32,
        filter: u32,
    ) -> Result<()> {
        self.validate_shape_id(shape)?;
        self.shape_collision_group[shape.index()] = memberships;
        self.shape_collision_mask[shape.index()] = filter;
        Ok(())
    }

    pub fn set_shape_contact_material(
        &mut self,
        shape: ShapeId,
        material: super::ContactMaterial,
    ) -> Result<()> {
        self.validate_shape_id(shape)?;
        material.validate()?;
        self.shape_contact_material[shape.index()] = material;
        Ok(())
    }

    pub fn set_shape_flags(&mut self, shape: ShapeId, flags: u32) -> Result<()> {
        self.validate_shape_id(shape)?;
        self.shape_flags[shape.index()] = flags;
        Ok(())
    }

    fn validate_shape_id(&self, shape: ShapeId) -> Result<()> {
        if shape.index() >= self.shape_geometry.len() {
            return Err(Error::Build("shape handle out of range".into()));
        }
        Ok(())
    }

    /// Add an explicit shape-pair exclusion. The pair is always
    /// filtered out by the broad phase, even if group bitmasks would
    /// allow it.
    pub fn add_shape_collision_exclusion(&mut self, a: ShapeId, b: ShapeId) {
        let (lo, hi) = if a.0 < b.0 { (a.0, b.0) } else { (b.0, a.0) };
        self.shape_collision_filter_pairs.push((lo, hi));
    }

    /// Override the friction/restitution for an already-added shape.
    /// Both values must be finite (no `NaN`, no `±Inf`); pass the
    /// builder's `default_friction` / `default_restitution` explicitly
    /// if you want the global default for one knob and an override for
    /// the other.
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
        if !friction.is_finite() {
            return Err(Error::Build(format!(
                "set_shape_material: friction must be finite, got {friction}"
            )));
        }
        if !restitution.is_finite() {
            return Err(Error::Build(format!(
                "set_shape_material: restitution must be finite, got {restitution}"
            )));
        }
        self.shape_friction[i] = friction;
        self.shape_restitution[i] = restitution;
        Ok(())
    }

    /// Override the rolling / torsional friction for an
    /// already-added shape. Both values must be finite — pass `0.0` to
    /// disable that mode. Rolling friction opposes spin perpendicular
    /// to the contact normal (wheel-on-ground); torsional opposes spin
    /// along the normal (top spinning on a surface).
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
        if !rolling.is_finite() {
            return Err(Error::Build(format!(
                "set_shape_friction_modes: rolling friction must be finite, got {rolling}"
            )));
        }
        if !torsional.is_finite() {
            return Err(Error::Build(format!(
                "set_shape_friction_modes: torsional friction must be finite, got {torsional}"
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
        Ok(self.push_joint(super::JointType::Fixed, parent, child, x_p, x_c, Vec3::ZERO))
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
        Ok(self.push_joint(super::JointType::Ball, parent, child, x_p, x_c, Vec3::ZERO))
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
        Ok(self.push_joint(super::JointType::D6, parent, child, x_p, x_c, Vec3::ZERO))
    }

    /// CABLE joint (Stage 39E) — 1 linear stretch DoF + 1 isotropic
    /// bend DoF. Adds 2 coordinates and 2 dofs. The cable axis is the
    /// joint-local **+z**, and the bend axis is the joint-local **+x**.
    ///
    /// Per-joint stiffness + damping coefficients are stored on the
    /// Model:
    ///
    /// ```text
    ///   τ_l = −k_l · q_l − c_l · q̇_l    (stretch)
    ///   τ_b = −k_b · q_b − c_b · q̇_b    (bend)
    /// ```
    ///
    /// Mirrors Newton's `JointType.CABLE` (`xtra/newton/_src/sim/enums.py`).
    pub fn add_joint_cable(
        &mut self,
        parent: BodyId,
        child: BodyId,
        x_p: Transform,
        x_c: Transform,
        stiffness_linear: Real,
        stiffness_bend: Real,
        damping_linear: Real,
        damping_bend: Real,
    ) -> Result<JointId> {
        self.validate_joint(parent, child, "add_joint_cable")?;
        for (label, v) in [
            ("stiffness_linear", stiffness_linear),
            ("stiffness_bend", stiffness_bend),
            ("damping_linear", damping_linear),
            ("damping_bend", damping_bend),
        ] {
            if !v.is_finite() || v < 0.0 {
                return Err(Error::Build(format!(
                    "add_joint_cable: {label} must be finite and ≥ 0, got {v}"
                )));
            }
        }
        let id = self.push_joint(super::JointType::Cable, parent, child, x_p, x_c, Vec3::ZERO);
        let idx = id.index();
        self.joint_cable_stiffness_linear[idx] = stiffness_linear;
        self.joint_cable_stiffness_bend[idx] = stiffness_bend;
        self.joint_cable_damping_linear[idx] = damping_linear;
        self.joint_cable_damping_bend[idx] = damping_bend;
        Ok(id)
    }

    /// Stage 42 — attach a Hill-type muscle to a CABLE joint. The
    /// muscle drives the cable's **linear stretch DoF** with an
    /// active tensile force (sign: pulls `q_l` toward negative,
    /// i.e. shorter cable). See [`crate::actuators::muscle`] for
    /// the curve definitions.
    ///
    /// v2 supports multiple muscles per joint — call `add_muscle`
    /// again to attach an antagonist for co-contraction. The solver
    /// sums torques from all muscles on the same joint.
    ///
    /// Returns the new muscle id (0-based index into the
    /// `model.muscle_*` parallel arrays).
    ///
    /// # Errors
    ///
    /// - the joint id is out of range
    /// - the joint is not `JointType::Cable`
    /// - any Hill parameter is non-finite or non-positive
    pub fn add_muscle(&mut self, joint: JointId, config: crate::MuscleConfig) -> Result<usize> {
        let idx = joint.index();
        if idx >= self.joint_type.len() {
            return Err(Error::Build(format!(
                "add_muscle: joint {joint:?} out of range (have {} joints)",
                self.joint_type.len()
            )));
        }
        if self.joint_type[idx] != super::JointType::Cable as u32 {
            return Err(Error::Build(format!(
                "add_muscle: joint {joint:?} is not a CABLE joint \
                 (type = {}); muscles only drive cable stretch DoFs",
                self.joint_type[idx]
            )));
        }
        for (label, v) in [
            ("f_max", config.f_max),
            ("l_opt", config.l_opt),
            ("v_max", config.v_max),
        ] {
            if !v.is_finite() || v <= 0.0 {
                return Err(Error::Build(format!(
                    "add_muscle: {label} must be finite and > 0, got {v}"
                )));
            }
        }
        if !config.activation.is_finite() {
            return Err(Error::Build(format!(
                "add_muscle: activation must be finite, got {}",
                config.activation
            )));
        }
        if !config.excitation.is_finite() {
            return Err(Error::Build(format!(
                "add_muscle: excitation must be finite, got {}",
                config.excitation
            )));
        }
        if !config.tau_activation.is_finite() || config.tau_activation < 0.0 {
            return Err(Error::Build(format!(
                "add_muscle: tau_activation must be finite and >= 0, got {}",
                config.tau_activation
            )));
        }
        if !config.l_tendon_slack.is_finite() || config.l_tendon_slack < 0.0 {
            return Err(Error::Build(format!(
                "add_muscle: l_tendon_slack must be finite and >= 0, got {}",
                config.l_tendon_slack
            )));
        }
        if !config.tendon_stiffness_ratio.is_finite() || config.tendon_stiffness_ratio <= 0.0 {
            return Err(Error::Build(format!(
                "add_muscle: tendon_stiffness_ratio must be finite and > 0, got {}",
                config.tendon_stiffness_ratio
            )));
        }
        if !config.pennation_angle.is_finite() {
            return Err(Error::Build(format!(
                "add_muscle: pennation_angle must be finite, got {}",
                config.pennation_angle
            )));
        }
        if !config.polarity.is_finite() || config.polarity == 0.0 {
            return Err(Error::Build(format!(
                "add_muscle: polarity must be finite and non-zero (use ±1), got {}",
                config.polarity
            )));
        }
        let polarity_clamped = if config.polarity >= 0.0 { 1.0 } else { -1.0 };
        let muscle_id = self.muscle_joint.len();
        self.muscle_joint.push(idx as i32);
        self.muscle_f_max.push(config.f_max);
        self.muscle_l_opt.push(config.l_opt);
        self.muscle_v_max.push(config.v_max);
        self.muscle_activation
            .push(config.activation.clamp(0.0, 1.0));
        self.muscle_excitation
            .push(config.excitation.clamp(0.0, 1.0));
        self.muscle_tau_activation.push(config.tau_activation);
        self.muscle_l_tendon_slack.push(config.l_tendon_slack);
        self.muscle_tendon_stiffness_ratio
            .push(config.tendon_stiffness_ratio);
        self.muscle_pennation_angle.push(config.pennation_angle);
        self.muscle_polarity.push(polarity_clamped);
        Ok(muscle_id)
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
        Ok(self.push_joint(super::JointType::Free, parent, child, x_p, x_c, Vec3::ZERO))
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
        let parent_idx = if parent == BodyId::NONE {
            -1
        } else {
            parent.0 as i32
        };
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
            self.joint_compliance
                .push(crate::joint_control::JointCompliance::DISABLED);
        }
        // Joint world inherits from the child body's world.
        let child_world = self.body_world.get(child.0 as usize).copied().unwrap_or(0);
        self.joint_world.push(child_world);
        self.joint_enabled.push(1);
        self.joint_reversed.push(0);
        // Ancestor: scan the current articulation's joints for one
        // whose child body matches this joint's parent body.
        let parent_body_idx = parent_idx;
        let articulation_first = *self.articulation_start.last().unwrap_or(&0) as usize;
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
        // Stage 39E — per-joint CABLE parameters. Pushed for *every*
        // joint (one row per joint, not per DoF) so the parallel
        // buffers stay length `joint_count`. Defaults to 0 for non-
        // cable joints; `add_joint_cable` overwrites them afterwards.
        self.joint_cable_stiffness_linear.push(0.0);
        self.joint_cable_stiffness_bend.push(0.0);
        self.joint_cable_damping_linear.push(0.0);
        self.joint_cable_damping_bend.push(0.0);
        // First joint of a (re-)opened articulation: nothing to push
        // into `articulation_start` — the trailing offset is set when
        // the articulation closes (either via `begin_articulation` or
        // at `build` time).
        self.articulation_open = true;
        id
    }

    /// Set the position limits for a joint's DOFs. Pass
    /// `Real::NEG_INFINITY` / `Real::INFINITY` (or `NaN` as a
    /// back-compat alias — substituted at finalize) to leave a bound
    /// unset. For multi-DOF joints (e.g. `Free`, future ball/D6),
    /// `lower` and `upper` must have length = `joint.dof_count()`.
    /// Replaces any compliant bounds on this joint.
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
        // Reject infinities-of-the-wrong-sign and reject ordering bugs
        // up front. `NaN` stays accepted as the "unset" sentinel for
        // back-compat; finalize substitutes ±∞.
        for k in 0..n {
            let lo = lower[k];
            let hi = upper[k];
            if !lo.is_nan() && lo == Real::INFINITY {
                return Err(Error::Build(format!(
                    "set_joint_limits: joint {joint:?} dof {k} lower bound is +∞"
                )));
            }
            if !hi.is_nan() && hi == Real::NEG_INFINITY {
                return Err(Error::Build(format!(
                    "set_joint_limits: joint {joint:?} dof {k} upper bound is -∞"
                )));
            }
            if !lo.is_nan() && !hi.is_nan() && lo > hi {
                return Err(Error::Build(format!(
                    "set_joint_limits: joint {joint:?} dof {k} lower {lo} > upper {hi}"
                )));
            }
        }
        self.joint_limit_lower[start..end].copy_from_slice(lower);
        self.joint_limit_upper[start..end].copy_from_slice(upper);
        self.joint_compliance[start..end].fill(crate::joint_control::JointCompliance::DISABLED);
        Ok(())
    }

    /// Replace D6 hard bounds with mass-scaled spring intervals (Hz, damping ratio).
    pub fn set_joint_compliance(
        &mut self,
        joint: JointId,
        softness: crate::joint_control::JointSoftness,
        lower: &[Real; 6],
        upper: &[Real; 6],
    ) -> Result<()> {
        softness.validate()?;
        let (start, end) = self.joint_dof_range(joint, "set_joint_compliance")?;
        if self.joint_type[joint.index()] != crate::JointType::D6 as u32 {
            return Err(Error::Build("joint compliance requires a D6 joint".into()));
        }
        let mut records = [crate::joint_control::JointCompliance::DISABLED; 6];
        for (i, record) in records.iter_mut().enumerate() {
            let (lo, hi) = (lower[i], upper[i]);
            if lo.is_nan()
                || hi.is_nan()
                || lo == Real::INFINITY
                || hi == Real::NEG_INFINITY
                || lo > hi
            {
                return Err(Error::Build("invalid compliant joint interval".into()));
            }
            *record = crate::joint_control::JointCompliance {
                enabled: u32::from(lo.is_finite() || hi.is_finite()),
                padding: 0,
                lower: lo,
                upper: hi,
                natural_frequency: softness.natural_frequency,
                damping_ratio: softness.damping_ratio,
            };
        }
        self.joint_compliance[start..end].copy_from_slice(&records);
        self.joint_limit_lower[start..end].fill(Real::NEG_INFINITY);
        self.joint_limit_upper[start..end].fill(Real::INFINITY);
        Ok(())
    }

    /// Set per-DOF damping (joint friction, N·m·s/rad or N·s/m).
    pub fn set_joint_friction(&mut self, joint: JointId, friction: &[Real]) -> Result<()> {
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

    /// Set per-DOF effort cap (N·m or N). Pass `Real::INFINITY` (or
    /// `NaN` as a back-compat alias — substituted at finalize) for
    /// unlimited; finite values must be ≥ 0.
    pub fn set_joint_effort_limit(&mut self, joint: JointId, effort: &[Real]) -> Result<()> {
        let (start, end) = self.joint_dof_range(joint, "set_joint_effort_limit")?;
        let n = end - start;
        if effort.len() != n {
            return Err(Error::Build(format!(
                "set_joint_effort_limit: joint {joint:?} has {n} DOFs, got {}",
                effort.len()
            )));
        }
        for (k, &e) in effort.iter().enumerate() {
            if !e.is_nan() && e.is_finite() && e < 0.0 {
                return Err(Error::Build(format!(
                    "set_joint_effort_limit: joint {joint:?} dof {k} effort cap {e} < 0"
                )));
            }
        }
        self.joint_effort_limit[start..end].copy_from_slice(effort);
        Ok(())
    }

    /// Invert the joint transform and motion subspace without changing authored coordinates.
    pub fn set_joint_reversed(&mut self, joint: JointId, reversed: bool) -> Result<()> {
        let value = self
            .joint_reversed
            .get_mut(joint.index())
            .ok_or_else(|| Error::Build("joint reversal index out of range".into()))?;
        *value = u32::from(reversed);
        Ok(())
    }

    /// Enable or disable a joint's dynamics contribution.
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

    /// Set per-DOF velocity cap (rad/s or m/s). Pass `Real::INFINITY`
    /// (or `NaN` as a back-compat alias — substituted at finalize)
    /// for unlimited; finite values must be ≥ 0.
    pub fn set_joint_velocity_limit(&mut self, joint: JointId, velocity: &[Real]) -> Result<()> {
        let (start, end) = self.joint_dof_range(joint, "set_joint_velocity_limit")?;
        let n = end - start;
        if velocity.len() != n {
            return Err(Error::Build(format!(
                "set_joint_velocity_limit: joint {joint:?} has {n} DOFs, got {}",
                velocity.len()
            )));
        }
        for (k, &v) in velocity.iter().enumerate() {
            if !v.is_nan() && v.is_finite() && v < 0.0 {
                return Err(Error::Build(format!(
                    "set_joint_velocity_limit: joint {joint:?} dof {k} velocity cap {v} < 0"
                )));
            }
        }
        self.joint_velocity_limit[start..end].copy_from_slice(velocity);
        Ok(())
    }

    fn joint_dof_range(&self, joint: JointId, fn_name: &'static str) -> Result<(usize, usize)> {
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
        self.equality_body_a.push(if body_a == BodyId::NONE {
            -1
        } else {
            body_a.0 as i32
        });
        self.equality_body_b.push(if body_b == BodyId::NONE {
            -1
        } else {
            body_b.0 as i32
        });
        self.equality_anchor_a.push(anchor_a);
        self.equality_anchor_b.push(anchor_b);
        self.equality_relative_rotation
            .push(molla_math::Quat::IDENTITY);
        self.equality_enabled.push(1);
        let world = if body_a != BodyId::NONE {
            self.body_world.get(body_a.0 as usize).copied().unwrap_or(0)
        } else if body_b != BodyId::NONE {
            self.body_world.get(body_b.0 as usize).copied().unwrap_or(0)
        } else {
            0
        };
        self.equality_constraint_world.push(world);
        // Stage-32 joint-mimic side-tables stay length-aligned with the
        // equality arrays. Non-mimic constraints get the sentinel
        // `(-1, -1, [0; 5])` — the kernel branches on `equality_type`
        // and never touches these for Connect / Weld.
        self.equality_mimic_driver_joint.push(-1);
        self.equality_mimic_driven_joint.push(-1);
        self.equality_mimic_poly.extend_from_slice(&[0.0; 5]);
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
        relative_rotation: molla_math::Quat,
    ) -> usize {
        let id = self.add_equality_connect(body_a, body_b, anchor_a, anchor_b);
        self.equality_type[id] = super::EqualityType::Weld as u32;
        self.equality_relative_rotation[id] = relative_rotation;
        id
    }

    /// Add a JOINT-mimic equality constraint (Newton's third equality
    /// kind, Stage 32): force the driven joint's coordinate to be a
    /// polynomial of the driver's coordinate,
    ///
    /// ```text
    /// q_driven = poly[0] + poly[1]·q_dr + poly[2]·q_dr²
    ///          + poly[3]·q_dr³ + poly[4]·q_dr⁴
    /// ```
    ///
    /// Both joints must be 1-DoF (revolute or prismatic) — only the
    /// joint's first coordinate is read / written. Use
    /// `[0, 1, 0, 0, 0]` for an identity mimic (`q_driven = q_driver`),
    /// `[0, -1, 0, 0, 0]` for an antisymmetric Robotiq-style coupling.
    ///
    /// The constraint is projected during XPBD's equality-projection
    /// step (one scalar Lagrange-multiplier solve per pair) — see
    /// `crate::solvers::xpbd::equality_projection_pass` (private). Joint limits
    /// on the *driven* joint are NOT enforced by this constraint:
    /// `q_driven` follows the polynomial verbatim, even past its own
    /// limit. The driver's limits / friction / effort caps are still
    /// applied by the joint subsystem.
    ///
    /// Returns the newly-allocated equality-constraint slot (same
    /// numbering as `add_equality_connect` / `add_equality_weld`).
    pub fn add_equality_joint_mimic(
        &mut self,
        driver_joint: JointId,
        driven_joint: JointId,
        polynomial: [Real; 5],
    ) -> Result<usize> {
        if driver_joint.index() >= self.joint_type.len()
            || driven_joint.index() >= self.joint_type.len()
        {
            return Err(Error::Build(
                "add_equality_joint_mimic: joint id out of range".to_string(),
            ));
        }
        // Allocate the equality slot. Bodies / anchors / relrot are
        // unused for JointMimic but the parallel arrays still need a
        // length-matched entry so per-equality indexing stays valid.
        let id = self.equality_type.len();
        self.equality_type
            .push(super::EqualityType::JointMimic as u32);
        self.equality_body_a.push(-1);
        self.equality_body_b.push(-1);
        self.equality_anchor_a.push(Vec3::ZERO);
        self.equality_anchor_b.push(Vec3::ZERO);
        self.equality_relative_rotation
            .push(molla_math::Quat::IDENTITY);
        self.equality_enabled.push(1);
        let world = self
            .joint_world
            .get(driver_joint.index())
            .copied()
            .unwrap_or(0);
        self.equality_constraint_world.push(world);
        // Mimic side-table.
        self.equality_mimic_driver_joint.push(driver_joint.0 as i32);
        self.equality_mimic_driven_joint.push(driven_joint.0 as i32);
        self.equality_mimic_poly.extend_from_slice(&polynomial);
        Ok(id)
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
        if driven.index() >= self.joint_type.len() || driver.index() >= self.joint_type.len() {
            return Err(Error::Build(
                "add_mimic_constraint: joint id out of range".to_string(),
            ));
        }
        let id = self.mimic_joint_a.len();
        self.mimic_joint_a.push(driven.0 as i32);
        self.mimic_joint_b.push(driver.0 as i32);
        self.mimic_c0.push(c0);
        self.mimic_c1.push(c1);
        self.mimic_enabled.push(1);
        let world = self.joint_world.get(driver.index()).copied().unwrap_or(0);
        self.mimic_constraint_world.push(world);
        Ok(id)
    }

    /// Set the gravity vector for `world`. Auto-extends the world list as
    /// needed; new worlds default to -9.81 m/s² on -Y.
    pub fn set_world_gravity(&mut self, world: WorldId, gravity: Vec3) {
        let w = world.0 as usize;
        while self.world_gravity.len() <= w {
            self.world_gravity
                .push(Vec3::new(0.0, -molla_math::GRAVITY, 0.0));
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

    /// Set the model-wide default friction coefficient — used when a
    /// shape is added without an explicit per-shape friction (set via
    /// [`Self::set_shape_material`]). Must be set
    /// **before** the shapes are added (or `build()` is called); the
    /// finalize substitutes NaN-sentinel per-shape friction with this
    /// value as a one-shot bake.
    pub fn set_default_friction(&mut self, friction: Real) -> Result<()> {
        if !friction.is_finite() {
            return Err(Error::Build(format!(
                "set_default_friction: must be finite, got {friction}"
            )));
        }
        self.default_friction = friction;
        Ok(())
    }

    /// Set the model-wide default coefficient of restitution — see
    /// [`Self::set_default_friction`] for the bake-at-finalize
    /// semantics.
    pub fn set_default_restitution(&mut self, restitution: Real) -> Result<()> {
        if !restitution.is_finite() {
            return Err(Error::Build(format!(
                "set_default_restitution: must be finite, got {restitution}"
            )));
        }
        self.default_restitution = restitution;
        Ok(())
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
            let first_joint_idx = *self.articulation_start.last().unwrap_or(&0) as usize;
            let world = self.joint_world.get(first_joint_idx).copied().unwrap_or(0);
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

        // Substitute per-shape material NaN sentinels with the
        // configured defaults so downstream consumers (narrow phase,
        // GPU contact resolver) see only finite values. The "NaN means
        // use default" convention is a builder-side authoring nicety;
        // the solver should never witness it.
        for v in &mut self.shape_friction {
            if v.is_nan() {
                *v = self.default_friction;
            }
        }
        for v in &mut self.shape_rolling_friction {
            if v.is_nan() {
                *v = 0.0;
            }
        }
        for v in &mut self.shape_torsional_friction {
            if v.is_nan() {
                *v = 0.0;
            }
        }
        for v in &mut self.shape_restitution {
            if v.is_nan() {
                *v = self.default_restitution;
            }
        }

        // Joint limits use NaN as the "unset" sentinel at the builder
        // surface. Replace with ±∞ so the solver's clamp / projection
        // can branch on `is_finite()` (or just use the value directly
        // — `x.clamp(-INF, INF)` is a no-op).
        for v in &mut self.joint_limit_lower {
            if v.is_nan() {
                *v = Real::NEG_INFINITY;
            }
        }
        for v in &mut self.joint_limit_upper {
            if v.is_nan() {
                *v = Real::INFINITY;
            }
        }
        // Effort / velocity limits feed `clamp_or_passthrough(x, limit)`
        // — `Real::INFINITY` makes the clamp a no-op, expressing
        // "unlimited" without a runtime NaN check.
        for v in &mut self.joint_effort_limit {
            if v.is_nan() {
                *v = Real::INFINITY;
            }
        }
        for v in &mut self.joint_velocity_limit {
            if v.is_nan() {
                *v = Real::INFINITY;
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

        // Stage 20.2: pre-expand per-vertex CABLE bending adjacency
        // into a flat CSR so the VBD per-vertex block can pick up the
        // bending term in one CSR walk (no per-cable rescan inside
        // the kernel). Each entry is `(v_prev, v_mid, v_next, role)`
        // + a parallel f32 `k_b`.
        let (vcb_offsets, vcb_entries, vcb_k) = build_vertex_cable_bending_adjacency(
            particle_count,
            &self.cable_particle_offsets,
            &self.cable_particles,
            &self.cable_bending_stiffness,
        );

        // VBD prep: per-vertex adjacency CSR + greedy particle
        // colouring. Used by `VbdSolver`; ignored by XPBD. Cheap to
        // compute (O(spring_count + particle_count · max_degree)) so
        // we always run it — picking a solver should not require
        // rebuilding the model.
        let (vertex_adjacent_springs_offsets, vertex_adjacent_springs) =
            build_vertex_spring_adjacency(particle_count, &self.spring_a, &self.spring_b);
        // Stage 20.2: per-vertex tet adjacency CSR + tet edges fed
        // into the colouring graph. Each tet wires 6 edges between
        // its 4 vertices; the greedy colouring honors them so the
        // per-vertex VBD block solve stays race-free even when
        // multiple tets share a vertex.
        let (vertex_adjacent_tets_offsets, vertex_adjacent_tets) =
            build_vertex_tet_adjacency(particle_count, &self.tet_indices);
        let (particle_color_count, particle_color_offsets, particle_indices_by_color) =
            color_particles_with_tets_and_bending(
                particle_count,
                &vertex_adjacent_springs_offsets,
                &vertex_adjacent_springs,
                &self.spring_a,
                &self.spring_b,
                &self.tet_indices,
                &self.cable_particle_offsets,
                &self.cable_particles,
                &self.cable_bending_stiffness,
            );

        // Stage 33 — build the soft-surface registry. Each tet mesh
        // contributes its boundary tris (computed via the standard
        // "tris that belong to exactly one tet" pass); each cloth
        // mesh contributes its existing tessellation tris straight
        // from `tri_indices`. Both append into the flat
        // `soft_surface_tris` array; per-surface entries record the
        // [tri_offset .. tri_offset + tri_count) slice + the vertex
        // pool slice the surface owns.
        let mut soft_surfaces: Vec<crate::SoftSurface> = Vec::new();
        let mut soft_surface_tris: Vec<[u32; 3]> = Vec::new();
        // Stage 33.1 — flat per-surface edge list, derived from each
        // surface's tris via `compute_surface_edges` (unique
        // undirected edges).
        let mut soft_surface_edges: Vec<[u32; 2]> = Vec::new();
        let n_tet_meshes = self.tet_mesh_offsets.len().saturating_sub(1);
        for m in 0..n_tet_meshes {
            let lo = self.tet_mesh_offsets[m] as usize;
            let hi = self.tet_mesh_offsets[m + 1] as usize;
            let mut mesh_tets: Vec<[u32; 4]> = Vec::with_capacity(hi - lo);
            for t in lo..hi {
                mesh_tets.push([
                    self.tet_indices[t * 4],
                    self.tet_indices[t * 4 + 1],
                    self.tet_indices[t * 4 + 2],
                    self.tet_indices[t * 4 + 3],
                ]);
            }
            let bnd = crate::compute_tet_surface_triangles(&mesh_tets);
            let tri_off = soft_surface_tris.len() as u32;
            for tri in &bnd {
                soft_surface_tris.push(*tri);
            }
            let edges = crate::compute_surface_edges(&bnd);
            let edge_off = soft_surface_edges.len() as u32;
            let edge_cnt = edges.len() as u32;
            for e in edges {
                soft_surface_edges.push(e);
            }
            let v_first = self.tet_mesh_vertex_first.get(m).copied().unwrap_or(0);
            let v_count = self.tet_mesh_vertex_count.get(m).copied().unwrap_or(0);
            let self_coll = self
                .tet_mesh_self_collision
                .get(m)
                .copied()
                .unwrap_or(false);
            // Stage 33.5: per-mesh friction/restitution overrides.
            // Default friction = 0.4, default tet restitution = 0.1.
            let friction = self.tet_mesh_friction.get(m).copied().unwrap_or(0.4);
            let restitution = self.tet_mesh_restitution.get(m).copied().unwrap_or(0.1);
            soft_surfaces.push(crate::SoftSurface {
                kind: crate::SoftSurfaceKind::TetMesh,
                vertex_offset: v_first,
                vertex_count: v_count,
                tri_offset: tri_off,
                tri_count: bnd.len() as u32,
                edge_offset: edge_off,
                edge_count: edge_cnt,
                tet_mesh_id: Some(crate::TetMeshId(m as u32)),
                cloth_mesh_id: None,
                self_collision: self_coll,
                friction,
                restitution,
            });
        }
        let n_cloth_meshes = self.cloth_mesh_first_particle.len();
        for m in 0..n_cloth_meshes {
            let tri_lo = self.cloth_mesh_tri_offsets[m] as usize;
            let tri_hi = self.cloth_mesh_tri_offsets[m + 1] as usize;
            let tri_off = soft_surface_tris.len() as u32;
            let mut mesh_tris: Vec<[u32; 3]> = Vec::with_capacity(tri_hi - tri_lo);
            for t in tri_lo..tri_hi {
                let tri = [
                    self.tri_indices[t * 3],
                    self.tri_indices[t * 3 + 1],
                    self.tri_indices[t * 3 + 2],
                ];
                soft_surface_tris.push(tri);
                mesh_tris.push(tri);
            }
            let edges = crate::compute_surface_edges(&mesh_tris);
            let edge_off = soft_surface_edges.len() as u32;
            let edge_cnt = edges.len() as u32;
            for e in edges {
                soft_surface_edges.push(e);
            }
            let self_coll = self
                .cloth_mesh_self_collision
                .get(m)
                .copied()
                .unwrap_or(false);
            // Stage 33.5: per-cloth friction/restitution overrides.
            // Default friction = 0.4, default cloth restitution = 0.0.
            let friction = self.cloth_mesh_friction.get(m).copied().unwrap_or(0.4);
            let restitution = self.cloth_mesh_restitution.get(m).copied().unwrap_or(0.0);
            soft_surfaces.push(crate::SoftSurface {
                kind: crate::SoftSurfaceKind::Cloth,
                vertex_offset: self.cloth_mesh_first_particle[m],
                vertex_count: self.cloth_mesh_particle_count[m],
                tri_offset: tri_off,
                tri_count: (tri_hi - tri_lo) as u32,
                edge_offset: edge_off,
                edge_count: edge_cnt,
                tet_mesh_id: None,
                cloth_mesh_id: Some(crate::ClothMeshId(m as u32)),
                self_collision: self_coll,
                friction,
                restitution,
            });
        }

        // Stage 33.3 — build the per-vertex incident-tri CSR over the
        // flat `soft_surface_tris` array. One pass: count incidence
        // per vertex, prefix-sum into offsets, then fill the payload.
        // The self-collision filter in `gather_vt_contacts` walks this
        // in O(incident-tris-per-vertex) (typically 6 for tets / 6 for
        // cloth) and skips candidate tris that are in the set.
        let mut svit_offsets: Vec<u32> = vec![0u32; particle_count + 1];
        for tri in &soft_surface_tris {
            for &v in tri {
                let v = v as usize;
                if v < particle_count {
                    svit_offsets[v + 1] += 1;
                }
            }
        }
        for i in 1..svit_offsets.len() {
            svit_offsets[i] += svit_offsets[i - 1];
        }
        let total = *svit_offsets.last().unwrap_or(&0) as usize;
        let mut svit: Vec<u32> = vec![0u32; total];
        // Cursor copy of offsets → bumped per insertion.
        let mut cursor: Vec<u32> = svit_offsets.clone();
        for (tri_global, tri) in soft_surface_tris.iter().enumerate() {
            for &v in tri {
                let v = v as usize;
                if v < particle_count {
                    let pos = cursor[v] as usize;
                    svit[pos] = tri_global as u32;
                    cursor[v] += 1;
                }
            }
        }

        // Stage 42 v2 — build joint → muscle CSR before the Model
        // literal moves `joint_type` and `muscle_joint`.
        let (joint_muscle_offsets_vec, joint_muscle_indices_vec) = {
            let n_joints = self.joint_type.len();
            let mut counts = vec![0u32; n_joints];
            for &j in &self.muscle_joint {
                if j >= 0 && (j as usize) < n_joints {
                    counts[j as usize] += 1;
                }
            }
            let mut offsets = Vec::with_capacity(n_joints + 1);
            offsets.push(0u32);
            let mut acc = 0u32;
            for c in &counts {
                acc += *c;
                offsets.push(acc);
            }
            let mut cursor = vec![0u32; n_joints];
            let mut running = 0u32;
            for (i, c) in counts.iter().enumerate() {
                cursor[i] = running;
                running += *c;
            }
            let mut indices = vec![0u32; self.muscle_joint.len()];
            for (m_id, &j) in self.muscle_joint.iter().enumerate() {
                if j < 0 || (j as usize) >= n_joints {
                    continue;
                }
                let slot = cursor[j as usize] as usize;
                indices[slot] = m_id as u32;
                cursor[j as usize] += 1;
            }
            (offsets, indices)
        };

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

            vertex_adjacent_springs_offsets: Buffer::from_vec(vertex_adjacent_springs_offsets, s),
            vertex_adjacent_springs: Buffer::from_vec(vertex_adjacent_springs, s),
            particle_color_count,
            particle_color_offsets: Buffer::from_vec(particle_color_offsets, s),
            particle_indices_by_color: Buffer::from_vec(particle_indices_by_color, s),

            tri_count,
            tri_indices: Buffer::from_vec(self.tri_indices, s),

            cable_count: self.cable_rest_length.len(),
            cable_particle_offsets: Buffer::from_vec(
                if self.cable_particle_offsets.is_empty() {
                    vec![0u32]
                } else {
                    self.cable_particle_offsets
                },
                s,
            ),
            cable_particles: Buffer::from_vec(self.cable_particles, s),
            cable_rest_length: Buffer::from_vec(self.cable_rest_length, s),
            cable_stiffness: Buffer::from_vec(self.cable_stiffness, s),
            cable_bending_stiffness: Buffer::from_vec(self.cable_bending_stiffness, s),
            vertex_cable_bending_offsets: Buffer::from_vec(vcb_offsets, s),
            vertex_cable_bending_entries: Buffer::from_vec(vcb_entries, s),
            vertex_cable_bending_k: Buffer::from_vec(vcb_k, s),

            tet_count: self.tet_indices.len() / 4,
            tet_indices: Buffer::from_vec(self.tet_indices, s),
            tet_dm_inv: Buffer::from_vec(self.tet_dm_inv, s),
            tet_rest_volume: Buffer::from_vec(self.tet_rest_volume, s),
            tet_mu: Buffer::from_vec(self.tet_mu, s),
            tet_lambda: Buffer::from_vec(self.tet_lambda, s),
            tet_f_plastic: Buffer::from_vec(self.tet_f_plastic, s),
            tet_plastic_strain: Buffer::from_vec(self.tet_plastic_strain, s),
            tet_yield_stress: Buffer::from_vec(self.tet_yield_stress, s),
            tet_hardening: Buffer::from_vec(self.tet_hardening, s),
            tet_kinematic_hardening: Buffer::from_vec(self.tet_kinematic_hardening, s),
            tet_mesh_count: self.tet_mesh_offsets.len().saturating_sub(1),
            tet_mesh_offsets: Buffer::from_vec(
                if self.tet_mesh_offsets.is_empty() {
                    vec![0u32]
                } else {
                    self.tet_mesh_offsets
                },
                s,
            ),
            soft_surfaces,
            soft_surface_tris,
            soft_vertex_incident_tris_offsets: svit_offsets,
            soft_vertex_incident_tris: svit,
            soft_surface_edges,
            #[cfg(feature = "gpu")]
            soft_surface_tris_gpu: None,
            #[cfg(feature = "gpu")]
            soft_surfaces_gpu: None,
            vertex_adjacent_tets_offsets: Buffer::from_vec(vertex_adjacent_tets_offsets, s),
            vertex_adjacent_tets: Buffer::from_vec(vertex_adjacent_tets, s),

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
            // `_data`. Triangle / TriMesh stay host-only — their
            // narrow-phase walks the mesh BVH on the CPU side. The
            // GPU sentinel `0xff` keeps them out of the kernel
            // dispatch (the kernel skips entries with kind == 0xff).
            shape_geom_kind: Buffer::from_vec(
                self.shape_geometry
                    .iter()
                    .map(|g| match g {
                        ShapeGeometry::Sphere { .. } => 0u32,
                        ShapeGeometry::Plane { .. } => 1,
                        ShapeGeometry::Box { .. } => 2,
                        ShapeGeometry::Capsule { .. } => 3,
                        ShapeGeometry::Cylinder { .. } => 4,
                        ShapeGeometry::RoundCylinder { .. } => 0xff,
                        // Cone (5) gets a real kind for GPU GJK;
                        // ConvexHull/Heightfield/Sdf stay CPU-only —
                        // sentinel 0xff so the GPU kernels skip them.
                        // Promotion to GPU kinds is owned by their
                        // respective stages (Heightfield in Stage 26,
                        // SDF in Stage 21).
                        ShapeGeometry::Cone { .. } => 5,
                        // Ellipsoid (6) gets a real kind — host packs the
                        // three semi-axes into shape_geom_data; GPU
                        // kernels that need it (gjk/epa/raycast/camera)
                        // implement the corresponding support /
                        // intersection. Stage 30.
                        ShapeGeometry::Ellipsoid { .. } => 6,
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
                        ShapeGeometry::Capsule {
                            half_height,
                            radius,
                        } => Vec3::new(half_height, radius, 0.0),
                        ShapeGeometry::Cylinder {
                            half_height,
                            radius,
                        } => Vec3::new(half_height, radius, 0.0),
                        ShapeGeometry::Triangle { .. } => Vec3::ZERO,
                        ShapeGeometry::RoundCylinder {
                            half_height,
                            radius,
                            border_radius,
                        } => Vec3::new(half_height, radius, border_radius),
                        ShapeGeometry::TriMesh { .. } => Vec3::ZERO,
                        ShapeGeometry::Cone {
                            half_height,
                            radius,
                        } => Vec3::new(half_height, radius, 0.0),
                        ShapeGeometry::Ellipsoid { half_axes } => half_axes,
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
            shape_contact_material: self.shape_contact_material,
            shape_rolling_friction: Buffer::from_vec(self.shape_rolling_friction, s),
            shape_torsional_friction: Buffer::from_vec(self.shape_torsional_friction, s),
            shape_restitution: Buffer::from_vec(self.shape_restitution, s),
            meshes: self.meshes,
            hulls: self.hulls,
            heightfields: self.heightfields,
            sdfs: self.sdfs,
            shape_collision_group: Buffer::from_vec(self.shape_collision_group, s),
            shape_collision_mask: Buffer::from_vec(self.shape_collision_mask, s),
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
            equality_relative_rotation: Buffer::from_vec(self.equality_relative_rotation, s),
            equality_enabled: Buffer::from_vec(self.equality_enabled, s),
            equality_constraint_world: Buffer::from_vec(self.equality_constraint_world, s),
            equality_mimic_driver_joint: Buffer::from_vec(self.equality_mimic_driver_joint, s),
            equality_mimic_driven_joint: Buffer::from_vec(self.equality_mimic_driven_joint, s),
            equality_mimic_poly: Buffer::from_vec(self.equality_mimic_poly, s),

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
            joint_compliance: Buffer::from_vec(self.joint_compliance, s),

            articulation_count,
            articulation_start: Buffer::from_vec(self.articulation_start, s),
            articulation_world: Buffer::from_vec(self.articulation_world, s),
            joint_world: Buffer::from_vec(self.joint_world, s),
            joint_enabled: Buffer::from_vec(self.joint_enabled, s),
            joint_reversed: Buffer::from_vec(self.joint_reversed, s),
            joint_ancestor: Buffer::from_vec(self.joint_ancestor, s),
            joint_target_ke: Buffer::from_vec(self.joint_target_ke, s),
            joint_cable_stiffness_linear: Buffer::from_vec(self.joint_cable_stiffness_linear, s),
            joint_cable_stiffness_bend: Buffer::from_vec(self.joint_cable_stiffness_bend, s),
            joint_cable_damping_linear: Buffer::from_vec(self.joint_cable_damping_linear, s),
            joint_cable_damping_bend: Buffer::from_vec(self.joint_cable_damping_bend, s),

            // Stage 42 v2 — Hill-type muscles. CSR maps joint →
            // muscle ids so co-contraction sums correctly in the
            // solver. `muscle_joint` (the builder-side per-muscle
            // array) is consumed here and discarded; downstream
            // consumers go through the CSR.
            muscle_count: self.muscle_joint.len(),
            joint_muscle_offsets: Buffer::from_vec(joint_muscle_offsets_vec, s),
            joint_muscle_indices: Buffer::from_vec(joint_muscle_indices_vec, s),
            muscle_f_max: Buffer::from_vec(self.muscle_f_max, s),
            muscle_l_opt: Buffer::from_vec(self.muscle_l_opt, s),
            muscle_v_max: Buffer::from_vec(self.muscle_v_max, s),
            muscle_activation: Buffer::from_vec(self.muscle_activation, s),
            muscle_excitation: Buffer::from_vec(self.muscle_excitation, s),
            muscle_tau_activation: Buffer::from_vec(self.muscle_tau_activation, s),
            muscle_l_tendon_slack: Buffer::from_vec(self.muscle_l_tendon_slack, s),
            muscle_tendon_stiffness_ratio: Buffer::from_vec(self.muscle_tendon_stiffness_ratio, s),
            muscle_pennation_angle: Buffer::from_vec(self.muscle_pennation_angle, s),
            muscle_polarity: Buffer::from_vec(self.muscle_polarity, s),

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
    // Stage 38d.1 — route through the shared
    // `math::coloring::color_constraints` utility. Each
    // spring is a constraint touching its two endpoint particles;
    // the utility (MCS + greedy on the line-graph) tends to use
    // fewer colours than the previous ad-hoc endpoint-scan greedy
    // on irregular topology.
    let _ = particle_count; // accepted for API stability
    let constraint_verts: Vec<Vec<u32>> = (0..n).map(|i| vec![spring_a[i], spring_b[i]]).collect();
    let colors = molla_math::color_constraints(&constraint_verts);
    let color_count = colors.iter().copied().max().map_or(0, |c| c as usize + 1);

    // Convert to CSR (offsets + indices ordered by colour). The
    // helper produces `groups[k]` = list of constraint indices in
    // colour k; we lay them out in colour-major order.
    let groups = molla_math::group_by_color(&colors);
    let mut offsets = vec![0u32; color_count + 1];
    for k in 0..color_count {
        offsets[k + 1] = offsets[k] + groups[k].len() as u32;
    }
    let mut indices = vec![0u32; n];
    let mut cursor = 0usize;
    for group in &groups {
        for &i in group {
            indices[cursor] = i as u32;
            cursor += 1;
        }
    }
    (color_count, offsets, indices)
}

/// Build a per-vertex CSR table of incident springs.
/// Returns `(offsets, flat_indices)` where vertex `v` owns
/// `flat_indices[offsets[v] .. offsets[v + 1]]`.
///
/// Used by [`crate::solvers::VbdSolver`]: each VBD vertex block
/// solve sums forces + Hessians over all incident constraints,
/// which the CSR makes O(degree) per vertex.
fn build_vertex_spring_adjacency(
    particle_count: usize,
    spring_a: &[u32],
    spring_b: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    if particle_count == 0 || spring_a.is_empty() {
        return (vec![0u32; particle_count + 1], Vec::new());
    }
    let mut counts = vec![0u32; particle_count];
    for i in 0..spring_a.len() {
        let a = spring_a[i] as usize;
        let b = spring_b[i] as usize;
        if a < particle_count {
            counts[a] += 1;
        }
        if b < particle_count {
            counts[b] += 1;
        }
    }
    let mut offsets = vec![0u32; particle_count + 1];
    for v in 0..particle_count {
        offsets[v + 1] = offsets[v] + counts[v];
    }
    let total = offsets[particle_count] as usize;
    let mut indices = vec![0u32; total];
    let mut cursors = offsets[..particle_count].to_vec();
    for i in 0..spring_a.len() {
        let a = spring_a[i] as usize;
        let b = spring_b[i] as usize;
        if a < particle_count {
            let slot = cursors[a] as usize;
            indices[slot] = i as u32;
            cursors[a] += 1;
        }
        if b < particle_count {
            let slot = cursors[b] as usize;
            indices[slot] = i as u32;
            cursors[b] += 1;
        }
    }
    (offsets, indices)
}

/// Build the per-vertex tet adjacency CSR (Stage 20.2). Each entry
/// pairs a tet id with the local vertex order `v_order ∈ {0..3}` so
/// the kernel can pick the right column of `Dm⁻¹`. Layout matches
/// Newton's `_fill_adjacent_tets`: for vertex `v`, the slice
/// `[offsets[v] .. offsets[v+1])` is `(tet0, order0, tet1, order1,
/// ...)` — i.e. each tet contributes 2 u32s.
///
/// Returns `(offsets, flat_pairs)`. `offsets` has length
/// `particle_count + 1`; `flat_pairs` has length `2 * Σ degree`.
pub(crate) fn build_vertex_tet_adjacency(
    particle_count: usize,
    tet_indices: &[u32],
) -> (Vec<u32>, Vec<u32>) {
    if particle_count == 0 || tet_indices.is_empty() {
        return (vec![0u32; particle_count + 1], Vec::new());
    }
    let n_tets = tet_indices.len() / 4;
    let mut counts = vec![0u32; particle_count];
    for t in 0..n_tets {
        for o in 0..4 {
            let v = tet_indices[t * 4 + o] as usize;
            if v < particle_count {
                counts[v] += 1;
            }
        }
    }
    // Each entry is *two* u32s (tet id + v_order), so the offsets
    // are scaled by 2.
    let mut offsets = vec![0u32; particle_count + 1];
    for v in 0..particle_count {
        offsets[v + 1] = offsets[v] + counts[v] * 2;
    }
    let total = offsets[particle_count] as usize;
    let mut data = vec![0u32; total];
    let mut cursors = offsets[..particle_count].to_vec();
    for t in 0..n_tets {
        for o in 0..4 {
            let v = tet_indices[t * 4 + o] as usize;
            if v < particle_count {
                let slot = cursors[v] as usize;
                data[slot] = t as u32;
                data[slot + 1] = o as u32;
                cursors[v] += 2;
            }
        }
    }
    (offsets, data)
}

/// Pre-expand the per-vertex CABLE bending adjacency from the
/// per-cable polyline lists. For every interior triple
/// `(v_prev, v_mid, v_next)` of a cable with `k_b > 0`, push a 4-u32
/// record `(v_prev, v_mid, v_next, role)` into the CSR for each of
/// the three particles, alongside a parallel f32 `k_b` per record.
///
/// Returns `(offsets, entries, ks)`. `offsets` has length
/// `particle_count + 1`, scaled by 4 (one entry = 4 u32s). `ks` has
/// length `Σ degree` (one f32 per record).
pub(crate) fn build_vertex_cable_bending_adjacency(
    particle_count: usize,
    cable_particle_offsets: &[u32],
    cable_particles: &[u32],
    cable_bending_stiffness: &[Real],
) -> (Vec<u32>, Vec<u32>, Vec<Real>) {
    if particle_count == 0 || cable_bending_stiffness.is_empty() {
        return (vec![0u32; particle_count + 1], Vec::new(), Vec::new());
    }
    // Pass 1 — count entries per vertex.
    let mut counts = vec![0u32; particle_count];
    for c in 0..cable_bending_stiffness.len() {
        let kb = cable_bending_stiffness[c];
        if kb <= 0.0 {
            continue;
        }
        let lo = cable_particle_offsets[c] as usize;
        let hi = cable_particle_offsets[c + 1] as usize;
        if hi - lo < 3 {
            continue;
        }
        for i in (lo + 1)..(hi - 1) {
            let v_prev = cable_particles[i - 1] as usize;
            let v_mid = cable_particles[i] as usize;
            let v_next = cable_particles[i + 1] as usize;
            for &v in &[v_prev, v_mid, v_next] {
                if v < particle_count {
                    counts[v] += 1;
                }
            }
        }
    }
    // Build CSR offsets (4 u32s per entry).
    let mut offsets = vec![0u32; particle_count + 1];
    for v in 0..particle_count {
        offsets[v + 1] = offsets[v] + counts[v] * 4;
    }
    let total_entries: u32 = counts.iter().sum();
    let mut entries = vec![0u32; (total_entries as usize) * 4];
    let mut ks = vec![0.0_f64; total_entries as usize];
    let mut cursors = vec![0u32; particle_count];
    let mut entry_cursors = vec![0u32; particle_count];
    let mut entry_idx_per_vertex = vec![0u32; particle_count];
    // We need the entry index (the index into `ks`) per vertex; compute
    // a parallel base from `offsets[v] / 4`.
    for v in 0..particle_count {
        cursors[v] = offsets[v];
        entry_cursors[v] = offsets[v] / 4;
        entry_idx_per_vertex[v] = offsets[v] / 4;
    }

    for c in 0..cable_bending_stiffness.len() {
        let kb = cable_bending_stiffness[c];
        if kb <= 0.0 {
            continue;
        }
        let lo = cable_particle_offsets[c] as usize;
        let hi = cable_particle_offsets[c + 1] as usize;
        if hi - lo < 3 {
            continue;
        }
        for i in (lo + 1)..(hi - 1) {
            let v_prev = cable_particles[i - 1];
            let v_mid = cable_particles[i];
            let v_next = cable_particles[i + 1];
            for (role, &v) in [v_prev, v_mid, v_next].iter().enumerate() {
                let v_us = v as usize;
                if v_us >= particle_count {
                    continue;
                }
                let slot = cursors[v_us] as usize;
                entries[slot] = v_prev;
                entries[slot + 1] = v_mid;
                entries[slot + 2] = v_next;
                entries[slot + 3] = role as u32;
                let entry_i = entry_cursors[v_us] as usize;
                ks[entry_i] = kb;
                cursors[v_us] += 4;
                entry_cursors[v_us] += 1;
            }
        }
    }
    let _ = entry_idx_per_vertex;
    (offsets, entries, ks)
}

/// Stage 33 — re-colour `particle_count` particles given a set of
/// dynamic VT contact pairs (vertex/tri-vertex). Combines the static
/// constraint adjacency (springs + tets + cable bending) with the
/// supplied per-substep VT pairs and runs the same greedy colouring.
///
/// The VBD GPU port (Stage 33.2) will call this once per substep
/// after `gather_vt_contacts` so the per-color sweep can include the
/// VT penalty terms without races. v1 ships the projection as a
/// separate post-substep pass and so doesn't need this — the function
/// is provided to keep the API stable for the GPU port.
#[allow(clippy::too_many_arguments, dead_code)]
pub(crate) fn color_particles_with_vt_contacts(
    particle_count: usize,
    vertex_adj_offsets: &[u32],
    vertex_adj_springs: &[u32],
    spring_a: &[u32],
    spring_b: &[u32],
    tet_indices: &[u32],
    cable_particle_offsets: &[u32],
    cable_particles: &[u32],
    cable_bending_stiffness: &[Real],
    vt_pairs: &[(u32, u32)],
) -> (usize, Vec<u32>, Vec<u32>) {
    color_particles_with_extra_pairs(
        particle_count,
        vertex_adj_offsets,
        vertex_adj_springs,
        spring_a,
        spring_b,
        tet_indices,
        cable_particle_offsets,
        cable_particles,
        cable_bending_stiffness,
        vt_pairs,
    )
}

/// Greedy graph colouring of particles (Stage 20.2). Within a colour,
/// no two particles share a constraint, so VBD's per-vertex block
/// solve can be dispatched in parallel without races. Returns
/// `(color_count, csr_offsets, indices_sorted_by_color)`.
///
/// Incorporates tet adjacency edges *and* CABLE bending hinges in
/// addition to spring-graph edges. Two particles `(u, v)` are coloured
/// non-adjacent iff *any* of the following pairs them:
///   - a spring connects them,
///   - they share a tet, or
///   - they appear in the same `(v_prev, v_mid, v_next)` bending
///     hinge.
///     Without this, the per-vertex VBD block solve would race within a
///     colour whenever a non-spring constraint touches both endpoints.
#[allow(clippy::too_many_arguments)]
fn color_particles_with_tets_and_bending(
    particle_count: usize,
    vertex_adj_offsets: &[u32],
    vertex_adj_springs: &[u32],
    spring_a: &[u32],
    spring_b: &[u32],
    tet_indices: &[u32],
    cable_particle_offsets: &[u32],
    cable_particles: &[u32],
    cable_bending_stiffness: &[Real],
) -> (usize, Vec<u32>, Vec<u32>) {
    color_particles_with_extra_pairs(
        particle_count,
        vertex_adj_offsets,
        vertex_adj_springs,
        spring_a,
        spring_b,
        tet_indices,
        cable_particle_offsets,
        cable_particles,
        cable_bending_stiffness,
        &[],
    )
}

/// Internal — runs the greedy colouring with optional extra
/// vertex-pair edges (e.g. VT contacts).
#[allow(clippy::too_many_arguments)]
fn color_particles_with_extra_pairs(
    particle_count: usize,
    vertex_adj_offsets: &[u32],
    vertex_adj_springs: &[u32],
    spring_a: &[u32],
    spring_b: &[u32],
    tet_indices: &[u32],
    cable_particle_offsets: &[u32],
    cable_particles: &[u32],
    cable_bending_stiffness: &[Real],
    extra_pairs: &[(u32, u32)],
) -> (usize, Vec<u32>, Vec<u32>) {
    if particle_count == 0 {
        return (0, vec![0u32], Vec::new());
    }
    // Build a per-vertex set of "must differ" colour neighbours:
    //   - spring endpoints (covered by the existing CSR), and
    //   - any other corner of any tet this vertex is part of.
    // We materialise it as a per-vertex Vec<u32> of unique neighbour
    // indices; greedy colouring then picks the smallest free colour.
    let mut neighbour: Vec<Vec<u32>> = vec![Vec::new(); particle_count];
    // Springs.
    for v in 0..particle_count {
        let lo = vertex_adj_offsets[v] as usize;
        let hi = vertex_adj_offsets[v + 1] as usize;
        for s in &vertex_adj_springs[lo..hi] {
            let s = *s as usize;
            let a = spring_a.get(s).copied().unwrap_or(u32::MAX) as usize;
            let b = spring_b.get(s).copied().unwrap_or(u32::MAX) as usize;
            let nbr = if a == v { b } else { a };
            if nbr < particle_count {
                neighbour[v].push(nbr as u32);
            }
        }
    }
    // Tets — every pair (i, j) of the four corners contributes a
    // mutual neighbour entry. Tiny constant per-tet cost (12 unique
    // pairs, deduped at the colouring step).
    let n_tets = tet_indices.len() / 4;
    for t in 0..n_tets {
        let v0 = tet_indices[t * 4] as usize;
        let v1 = tet_indices[t * 4 + 1] as usize;
        let v2 = tet_indices[t * 4 + 2] as usize;
        let v3 = tet_indices[t * 4 + 3] as usize;
        let corners = [v0, v1, v2, v3];
        for i in 0..4 {
            for j in 0..4 {
                if i == j {
                    continue;
                }
                if corners[i] < particle_count && corners[j] < particle_count {
                    neighbour[corners[i]].push(corners[j] as u32);
                }
            }
        }
    }
    // CABLE bending — for every interior triple `(v_prev, v_mid,
    // v_next)` of a cable with `k_b > 0`, declare each pair as
    // mutual neighbours so the colouring keeps them in distinct
    // colours.
    for c in 0..cable_bending_stiffness.len() {
        if cable_bending_stiffness[c] <= 0.0 {
            continue;
        }
        let lo = cable_particle_offsets[c] as usize;
        let hi = cable_particle_offsets[c + 1] as usize;
        if hi - lo < 3 {
            continue;
        }
        for i in (lo + 1)..(hi - 1) {
            let triple = [
                cable_particles[i - 1] as usize,
                cable_particles[i] as usize,
                cable_particles[i + 1] as usize,
            ];
            for a in 0..3 {
                for b in 0..3 {
                    if a == b {
                        continue;
                    }
                    if triple[a] < particle_count && triple[b] < particle_count {
                        neighbour[triple[a]].push(triple[b] as u32);
                    }
                }
            }
        }
    }
    // Stage 33 — extra dynamic pairs (e.g. VT contact endpoints).
    // Both directions are added for symmetry.
    for &(u, v) in extra_pairs {
        let ui = u as usize;
        let vi = v as usize;
        if ui >= particle_count || vi >= particle_count || ui == vi {
            continue;
        }
        neighbour[ui].push(v);
        neighbour[vi].push(u);
    }

    // Stage 38d.1 — route the actual colour assignment through the
    // shared `math::coloring::mcs_color` (Maximum Cardinality
    // Search ordering + greedy). The domain-specific bit above builds
    // the neighbour list; the algorithm is generic and lives in the
    // utility.
    let color_of = molla_math::mcs_color(&neighbour);
    let color_count = color_of.iter().copied().max().map_or(0, |c| c as usize + 1);
    let groups = molla_math::group_by_color(&color_of);

    // Convert groups to CSR — same offsets + indices layout the
    // downstream VBD kernel expects.
    let mut offsets = vec![0u32; color_count + 1];
    for k in 0..color_count {
        offsets[k + 1] = offsets[k] + groups[k].len() as u32;
    }
    let mut indices = vec![0u32; particle_count];
    let mut cursor = 0usize;
    for group in &groups {
        for &v in group {
            indices[cursor] = v as u32;
            cursor += 1;
        }
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
        let _ = b
            .add_joint_revolute(
                BodyId::NONE,
                l0,
                Transform::IDENTITY,
                Transform::IDENTITY,
                Vec3::Z,
            )
            .unwrap();
        b.begin_articulation();
        let l1 = b.add_body(BodyParams::new().with_world(WorldId(2)));
        let _ = b
            .add_joint_revolute(
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
            molla_core::WorldId(0),
        );
        let m = b.build().unwrap();
        let n = 4_usize;
        let structural = 2 * n * (n - 1); // 24
        let shear = 2 * (n - 1) * (n - 1); // 18
        let bending = 2 * (n - 2) * n; // 16
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
            molla_core::WorldId(0),
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
            molla_core::WorldId(0),
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
