//! `Model` — static, immutable simulation description.

use crate::core::Result;
use crate::math::{Mat3, Real, Transform, Vec3};
use crate::storage::{Buffer, BufferUsage};

/// A triangle mesh stored on the host. Vertices are in the mesh's
/// local frame; world positioning comes from the shape's transform.
/// Triangle indexing convention: `indices[t]` gives the three
/// vertex indices forming triangle `t`, wound CCW when looking
/// against the outward normal (right-hand rule).
#[derive(Clone, Debug, Default)]
pub struct TriMesh {
    pub vertices: Vec<Vec3>,
    pub indices: Vec<[u32; 3]>,
    /// Per-mesh AABB BVH for accelerating shape-vs-trimesh narrow
    /// phase and ray-cast queries. Empty until [`Self::build_bvh`]
    /// is called (the builder runs that automatically when a mesh
    /// is added). Without this every shape-mesh pair iterates ALL
    /// triangles — fine for ~50-tri shapes, ruinous past 1k.
    pub bvh: crate::geometry::mesh_bvh::MeshBvh,
}

impl TriMesh {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// (Re)build the BVH from the current `vertices` + `indices`.
    /// `max_leaf` is the per-leaf triangle cap (4 is a reasonable
    /// default — Embree's hint).
    pub fn build_bvh(&mut self, max_leaf: usize) {
        self.bvh = crate::geometry::mesh_bvh::MeshBvh::build(
            &self.vertices,
            &self.indices,
            max_leaf,
        );
    }
}

/// Static description of a simulation. Built once from a `ModelBuilder`
/// and shared across many [`State`](super::State) instances.
#[derive(Debug)]
pub struct Model {
    // ── Particles ───────────────────────────────────────────────────
    /// Number of particles.
    pub particle_count: usize,
    /// Per-particle mass (kg).
    pub particle_mass: Buffer<Real>,
    /// Per-particle inverse mass (1/kg), 0 for kinematic.
    pub particle_inv_mass: Buffer<Real>,
    /// Per-particle radius (m).
    pub particle_radius: Buffer<Real>,
    /// Per-particle bitflags (see [`super::ParticleFlags`]).
    pub particle_flags: Buffer<u32>,
    /// Per-particle world index (-1 = global).
    pub particle_world: Buffer<i32>,
    /// Initial particle positions snapshotted at build time. Reset via
    /// `State::for_model`.
    pub initial_particle_q: Buffer<Vec3>,

    // ── Springs / distance constraints ──────────────────────────────
    /// Number of distance constraints (cloth edges, ropes, springs).
    pub spring_count: usize,
    /// Endpoint-A particle index.
    pub spring_a: Buffer<u32>,
    /// Endpoint-B particle index.
    pub spring_b: Buffer<u32>,
    /// Rest length of each constraint (m).
    pub spring_rest_length: Buffer<Real>,
    /// XPBD compliance α (m/N). 0 = perfectly rigid, ~1e-7 = stiff
    /// stretch, ~1e-2 = soft. `(1 / stiffness)` if you'd think in
    /// terms of a spring constant.
    pub spring_compliance: Buffer<Real>,
    /// Per-spring hard stretch cap as a ratio of rest length:
    /// `length ≤ rest · (1 + max_stretch)` is enforced as a
    /// post-XPBD non-linear projection. 0.0 means perfectly
    /// inextensible (length capped at rest length). 0.05 = 5 %
    /// stretchy linen. >= ~1.0 effectively disables the cap.
    pub spring_max_stretch: Buffer<Real>,

    // ── Spring graph colouring ──────────────────────────────────────
    // Computed once at build time so the GPU XPBD kernel can solve
    // distance / strain-limit constraints in parallel: within a
    // colour, no two springs share a particle, so threads can
    // read-modify-write `particle_q` without races.
    /// Number of colours used by the greedy graph coloring.
    pub spring_color_count: usize,
    /// CSR offsets over `spring_indices_by_color` — colour `k` owns
    /// springs in range `[spring_color_offsets[k], spring_color_offsets[k+1])`.
    /// Length `spring_color_count + 1`. Empty when `spring_count == 0`.
    pub spring_color_offsets: Buffer<u32>,
    /// Spring indices sorted by colour. Length `spring_count`. The
    /// GPU kernel dispatches `spring_color_count` passes; pass `k`
    /// reads slot `i` and looks up `spring = spring_indices_by_color[offset_k + i]`.
    pub spring_indices_by_color: Buffer<u32>,

    // ── Triangle topology (cloth surfaces) ──────────────────────────
    /// Number of triangles. Used for rendering today; bending /
    /// area-preservation constraints will read this in stage 12.x.
    pub tri_count: usize,
    /// Flat `tri_count * 3` array of particle indices (CCW winding).
    pub tri_indices: Buffer<u32>,

    // ── Rigid bodies ────────────────────────────────────────────────
    /// Number of rigid bodies.
    pub body_count: usize,
    /// Per-body mass (kg).
    pub body_mass: Buffer<Real>,
    /// Per-body inverse mass (1/kg), 0 for static/kinematic.
    pub body_inv_mass: Buffer<Real>,
    /// Per-body inertia tensor at COM, body frame (kg·m²).
    pub body_inertia: Buffer<Mat3>,
    /// Per-body inverse inertia tensor.
    pub body_inv_inertia: Buffer<Mat3>,
    /// Per-body centre of mass in body frame (m).
    pub body_com: Buffer<Vec3>,
    /// Per-body bitflags (see [`super::BodyFlags`]).
    pub body_flags: Buffer<u32>,
    /// Per-body world index.
    pub body_world: Buffer<i32>,
    /// Initial body pose snapshotted at build time. [`super::State::for_model`]
    /// copies this into the working state.
    pub initial_body_q: Buffer<Transform>,

    // ── Shapes ──────────────────────────────────────────────────────
    /// Number of collision shapes.
    pub shape_count: usize,
    /// Owning body index, or -1 for static-world shapes.
    pub shape_body: Buffer<i32>,
    /// **GPU-side shape kind**, parallel to `shape_geometry`. The
    /// `ShapeGeometry` enum is a Rust-side convenience; for the GPU we
    /// flatten it into two parallel arrays — kind discriminant +
    /// packed scalar parameters — so kernels can branch on it.
    /// Values: 0=Sphere · 1=Plane · 2=Box · 3=Capsule · 4=Cylinder ·
    /// 5=Triangle · 6=TriMesh · 0xff=unsupported-on-gpu (skipped).
    pub shape_geom_kind: Buffer<u32>,
    /// **GPU-side shape parameters**, parallel to `shape_geometry`.
    /// Per-kind packing:
    /// - Sphere    → `(radius, 0, 0)`
    /// - Plane     → `(nx, ny, nz)` (local-frame normal)
    /// - Box       → `(hx, hy, hz)` (half-extents)
    /// - Capsule   → `(half_height, radius, 0)`
    /// - Cylinder  → `(half_height, radius, 0)`
    /// - Triangle  → `(0, 0, 0)` (vertices held host-side; see
    ///                `ShapeGeometry::Triangle`)
    /// - TriMesh   → `(mesh_id_as_f32, 0, 0)` (mesh BVH lives on host
    ///                until stage 11 ports it)
    pub shape_geom_data: Buffer<Vec3>,
    /// Shape transform in owning body frame.
    pub shape_transform: Buffer<Transform>,
    /// Per-shape geometry tags (sphere radius, etc).
    pub shape_geometry: Vec<super::ShapeGeometry>,
    /// Per-shape bitflags.
    pub shape_flags: Buffer<u32>,
    /// Per-shape Coulomb friction coefficient. `NaN` = use the
    /// `default_friction` model knob; otherwise the pair-rule is
    /// `max(μ_a, μ_b)` (matches Newton).
    pub shape_friction: Buffer<Real>,
    /// Per-shape **rolling** friction (m·μ) — opposes the angular
    /// velocity component perpendicular to the contact normal.
    /// Without this, balls/cylinders roll forever on a flat plane.
    /// Pair-rule: `max(μ_a, μ_b)`. `NaN` = `0.0` (no rolling damping).
    pub shape_rolling_friction: Buffer<Real>,
    /// Per-shape **torsional** friction — opposes the angular
    /// velocity component ALONG the contact normal (spinning).
    /// Stops a ball from spinning forever about the contact axis.
    /// Same convention as `shape_rolling_friction`.
    pub shape_torsional_friction: Buffer<Real>,
    /// Per-shape restitution. Same `NaN` sentinel + max-rule as friction.
    pub shape_restitution: Buffer<Real>,
    /// Triangle-mesh storage. `ShapeGeometry::TriMesh { mesh_id }`
    /// references entries here.
    pub meshes: Vec<TriMesh>,
    /// Convex-hull vertex storage. `ShapeGeometry::ConvexHull { hull_id }`
    /// references entries here. Each hull is a flat `Vec<Vec3>` of
    /// vertex positions in the shape's local frame; the GJK support
    /// function scans them linearly.
    pub hulls: Vec<Vec<Vec3>>,
    /// Per-shape collision group bitmask. Two shapes can collide
    /// only if `group_a & group_b != 0`. Default `0xffff_ffff` =
    /// "collides with every group". Length = `shape_count`.
    pub shape_collision_group: Buffer<u32>,
    /// Explicit shape-pair exclusions (always filtered out, even
    /// when group bitmasks would allow). Stored as `(min, max)`
    /// shape-index pairs; broad phase tests pair membership against
    /// this set.
    pub shape_collision_filter_pairs: Vec<(u32, u32)>,

    // ── Equality constraints (CONNECT / WELD) ──────────────────────
    /// Number of equality constraints.
    pub equality_count: usize,
    /// Per-constraint type — `EqualityType::{Connect, Weld}` as `u32`.
    pub equality_type: Buffer<u32>,
    /// Per-constraint body A index (-1 = world).
    pub equality_body_a: Buffer<i32>,
    /// Per-constraint body B index (-1 = world).
    pub equality_body_b: Buffer<i32>,
    /// Per-constraint anchor in body A's local frame.
    pub equality_anchor_a: Buffer<Vec3>,
    /// Per-constraint anchor in body B's local frame.
    pub equality_anchor_b: Buffer<Vec3>,
    /// Per-constraint relative orientation (used by `Weld`). For
    /// `Connect` it's ignored.
    pub equality_relative_rotation: Buffer<crate::math::Quat>,
    /// Per-constraint enabled flag (1 = active).
    pub equality_enabled: Buffer<u32>,
    /// Per-constraint world index (defaults to body A's world).
    pub equality_constraint_world: Buffer<i32>,

    // ── Mimic constraints (q_a = c0 + c1 · q_b) ────────────────────
    /// Number of mimic constraints.
    pub mimic_count: usize,
    /// Per-mimic driven joint index (the one whose q is overridden).
    pub mimic_joint_a: Buffer<i32>,
    /// Per-mimic driver joint index (the one whose q is read).
    pub mimic_joint_b: Buffer<i32>,
    /// Per-mimic constant offset.
    pub mimic_c0: Buffer<Real>,
    /// Per-mimic linear coefficient.
    pub mimic_c1: Buffer<Real>,
    /// Per-mimic enabled flag.
    pub mimic_enabled: Buffer<u32>,
    /// Per-mimic world index (defaults to driver joint's world).
    pub mimic_constraint_world: Buffer<i32>,

    // ── Joints ──────────────────────────────────────────────────────
    /// Number of joints.
    pub joint_count: usize,
    /// Joint kind, as a `u32` mirror of [`super::JointType`].
    pub joint_type: Buffer<u32>,
    /// Per-joint parent body index (-1 = world).
    pub joint_parent: Buffer<i32>,
    /// Per-joint child body index.
    pub joint_child: Buffer<i32>,
    /// Joint frame in parent body.
    pub joint_x_p: Buffer<Transform>,
    /// Joint frame in child body.
    pub joint_x_c: Buffer<Transform>,
    /// Per-joint axis in joint-local frame. Unused for fixed/free.
    pub joint_axis: Buffer<Vec3>,
    /// CSR-style offset into `state.joint_q` for joint `i` —
    /// length `joint_count + 1`, last entry = total coord count.
    pub joint_coord_offset: Buffer<u32>,
    /// CSR-style offset into `state.joint_qd` — length `joint_count + 1`.
    pub joint_dof_offset: Buffer<u32>,
    /// Total joint coordinates (== last entry of `joint_coord_offset`).
    pub joint_coord_total: usize,
    /// Total joint DoFs (== last entry of `joint_dof_offset`).
    pub joint_dof_total: usize,
    /// Initial joint coordinates snapshotted at build time —
    /// [`super::State::for_model`] copies these into `joint_q`.
    pub initial_joint_q: Buffer<Real>,
    /// Per-DOF lower position limit (rad or m). `NaN` = unlimited.
    /// Length = `joint_dof_total`.
    pub joint_limit_lower: Buffer<Real>,
    /// Per-DOF upper position limit. `NaN` = unlimited.
    pub joint_limit_upper: Buffer<Real>,
    /// Per-DOF Coulomb-style damping coefficient (N·m·s/rad or N·s/m).
    /// 0 = none. Damping torque each step is `−friction·qd`.
    pub joint_friction: Buffer<Real>,
    /// Per-DOF effort cap (N·m or N). `NaN` = unlimited. Applied
    /// torque (joint_force + drive + friction) is clamped to
    /// `[−effort_limit, +effort_limit]`.
    pub joint_effort_limit: Buffer<Real>,
    /// Per-DOF velocity cap (rad/s or m/s). `NaN` = unlimited.
    /// Post-step `qd` is clamped to `[−velocity_limit, +velocity_limit]`.
    pub joint_velocity_limit: Buffer<Real>,

    // ── Articulations ───────────────────────────────────────────────
    /// Number of independent articulations (kinematic chains). The
    /// Featherstone solver dispatches one workgroup invocation per
    /// articulation. Equals `articulation_start.len() − 1`.
    pub articulation_count: usize,
    /// CSR offsets into the joint arrays — length `articulation_count + 1`.
    /// Articulation `a` owns joints `articulation_start[a]..articulation_start[a+1]`.
    pub articulation_start: Buffer<u32>,
    /// Per-articulation world index — defaults to the world of the
    /// first joint's child body. Length = `articulation_count`.
    pub articulation_world: Buffer<i32>,
    /// Per-joint world index. Defaults to the child body's world.
    /// Length = `joint_count`.
    pub joint_world: Buffer<i32>,
    /// Per-joint enable flag (1 = active, 0 = disabled). Disabled
    /// joints don't contribute torque or constrain motion. Length =
    /// `joint_count`.
    pub joint_enabled: Buffer<u32>,
    /// Per-joint ancestor — index of the parent joint in the
    /// articulation tree, or -1 for a root. Length = `joint_count`.
    /// Currently informational; future tree traversals consume it.
    pub joint_ancestor: Buffer<i32>,
    /// Per-DOF stiffness (N·m/rad or N/m). Distinct from
    /// `Control.joint_target_kp` — `target_ke` is a static joint
    /// property used by Featherstone for spring-style joint drives;
    /// `target_kp` is the per-step PD gain. Length = `joint_dof_total`.
    pub joint_target_ke: Buffer<Real>,

    // ── Worlds ──────────────────────────────────────────────────────
    /// Number of independent worlds (RL/multi-env grouping).
    pub world_count: usize,
    /// Per-world gravity vector (m/s²).
    pub gravity: Buffer<Vec3>,

    // ── Global solver knobs ─────────────────────────────────────────
    /// Angular damping coefficient applied each step (1/s).
    pub angular_damping: Real,
    /// Linear damping coefficient applied to body velocities each step
    /// (1/s). Models air drag / rolling resistance crudely:
    /// `v *= exp(-linear_damping · dt)`. Default 0 — keeps existing
    /// tests bit-identical.
    pub linear_damping: Real,
    /// Velocity cap applied to particles (m/s) for stability.
    pub particle_velocity_max: Real,
    /// Slack added to AABBs in the broad phase (m). Catches contacts
    /// that would otherwise be missed at the boundary of a step.
    pub contact_margin: Real,
    /// Default Coulomb friction coefficient applied to fresh contacts
    /// when shapes don't carry a per-shape material.
    pub default_friction: Real,
    /// Default restitution applied to fresh contacts when shapes
    /// don't carry a per-shape material.
    pub default_restitution: Real,
}

impl Model {
    /// An empty model: zero particles, zero bodies, zero shapes, one
    /// world with standard gravity.
    #[must_use]
    pub fn empty() -> Self {
        let s = BufferUsage::Storage;
        Self {
            particle_count: 0,
            particle_mass: Buffer::from_vec(Vec::new(), s),
            particle_inv_mass: Buffer::from_vec(Vec::new(), s),
            particle_radius: Buffer::from_vec(Vec::new(), s),
            particle_flags: Buffer::from_vec(Vec::new(), s),
            particle_world: Buffer::from_vec(Vec::new(), s),
            initial_particle_q: Buffer::from_vec(Vec::new(), s),

            spring_count: 0,
            spring_a: Buffer::from_vec(Vec::new(), s),
            spring_b: Buffer::from_vec(Vec::new(), s),
            spring_rest_length: Buffer::from_vec(Vec::new(), s),
            spring_compliance: Buffer::from_vec(Vec::new(), s),
            spring_max_stretch: Buffer::from_vec(Vec::new(), s),
            spring_color_count: 0,
            // Single sentinel offset so empty models don't need a
            // separate device-side branch.
            spring_color_offsets: Buffer::from_vec(vec![0u32], s),
            spring_indices_by_color: Buffer::from_vec(Vec::new(), s),

            tri_count: 0,
            tri_indices: Buffer::from_vec(Vec::new(), s),

            body_count: 0,
            body_mass: Buffer::from_vec(Vec::new(), s),
            body_inv_mass: Buffer::from_vec(Vec::new(), s),
            body_inertia: Buffer::from_vec(Vec::new(), s),
            body_inv_inertia: Buffer::from_vec(Vec::new(), s),
            body_com: Buffer::from_vec(Vec::new(), s),
            body_flags: Buffer::from_vec(Vec::new(), s),
            body_world: Buffer::from_vec(Vec::new(), s),
            initial_body_q: Buffer::from_vec(Vec::new(), s),

            shape_count: 0,
            shape_body: Buffer::from_vec(Vec::new(), s),
            shape_geom_kind: Buffer::from_vec(Vec::new(), s),
            shape_geom_data: Buffer::from_vec(Vec::new(), s),
            shape_transform: Buffer::from_vec(Vec::new(), s),
            shape_geometry: Vec::new(),
            shape_flags: Buffer::from_vec(Vec::new(), s),
            shape_friction: Buffer::from_vec(Vec::new(), s),
            shape_rolling_friction: Buffer::from_vec(Vec::new(), s),
            shape_torsional_friction: Buffer::from_vec(Vec::new(), s),
            shape_restitution: Buffer::from_vec(Vec::new(), s),
            meshes: Vec::new(),
            hulls: Vec::new(),
            shape_collision_group: Buffer::from_vec(Vec::new(), s),
            shape_collision_filter_pairs: Vec::new(),

            equality_count: 0,
            equality_type: Buffer::from_vec(Vec::new(), s),
            equality_body_a: Buffer::from_vec(Vec::new(), s),
            equality_body_b: Buffer::from_vec(Vec::new(), s),
            equality_anchor_a: Buffer::from_vec(Vec::new(), s),
            equality_anchor_b: Buffer::from_vec(Vec::new(), s),
            equality_relative_rotation: Buffer::from_vec(Vec::new(), s),
            equality_enabled: Buffer::from_vec(Vec::new(), s),
            equality_constraint_world: Buffer::from_vec(Vec::new(), s),

            mimic_count: 0,
            mimic_joint_a: Buffer::from_vec(Vec::new(), s),
            mimic_joint_b: Buffer::from_vec(Vec::new(), s),
            mimic_c0: Buffer::from_vec(Vec::new(), s),
            mimic_c1: Buffer::from_vec(Vec::new(), s),
            mimic_enabled: Buffer::from_vec(Vec::new(), s),
            mimic_constraint_world: Buffer::from_vec(Vec::new(), s),

            joint_count: 0,
            joint_type: Buffer::from_vec(Vec::new(), s),
            joint_parent: Buffer::from_vec(Vec::new(), s),
            joint_child: Buffer::from_vec(Vec::new(), s),
            joint_x_p: Buffer::from_vec(Vec::new(), s),
            joint_x_c: Buffer::from_vec(Vec::new(), s),
            joint_axis: Buffer::from_vec(Vec::new(), s),
            joint_coord_offset: Buffer::from_vec(vec![0u32], s),
            joint_dof_offset: Buffer::from_vec(vec![0u32], s),
            joint_coord_total: 0,
            joint_dof_total: 0,
            initial_joint_q: Buffer::from_vec(Vec::new(), s),
            joint_limit_lower: Buffer::from_vec(Vec::new(), s),
            joint_limit_upper: Buffer::from_vec(Vec::new(), s),
            joint_friction: Buffer::from_vec(Vec::new(), s),
            joint_effort_limit: Buffer::from_vec(Vec::new(), s),
            joint_velocity_limit: Buffer::from_vec(Vec::new(), s),

            articulation_count: 0,
            articulation_start: Buffer::from_vec(vec![0u32], s),
            articulation_world: Buffer::from_vec(Vec::new(), s),
            joint_world: Buffer::from_vec(Vec::new(), s),
            joint_enabled: Buffer::from_vec(Vec::new(), s),
            joint_ancestor: Buffer::from_vec(Vec::new(), s),
            joint_target_ke: Buffer::from_vec(Vec::new(), s),

            world_count: 1,
            gravity: Buffer::from_vec(vec![Vec3::new(0.0, -crate::math::GRAVITY, 0.0)], s),

            angular_damping: 0.0,
            linear_damping: 0.0,
            particle_velocity_max: 1.0e6,
            contact_margin: 0.01,
            default_friction: 0.5,
            default_restitution: 0.0,
        }
    }

    /// Allocate a fresh [`super::State`] sized for this model.
    pub fn state(&self) -> Result<super::State> {
        super::State::for_model(self)
    }

    /// Allocate a fresh [`super::Control`] sized for this model.
    #[must_use]
    pub fn control(&self) -> super::Control {
        super::Control::for_model(self)
    }

    /// Allocate device buffers for every model array and upload host data.
    ///
    /// The model is read-only at simulation time, so this only needs to
    /// run once before the first GPU step. Re-running is harmless but
    /// re-uploads the same data.
    #[cfg(feature = "gpu")]
    pub fn upload_to_device(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) -> Result<()> {
        macro_rules! up {
            ($field:ident) => {{
                self.$field.ensure_device(device)?;
                self.$field.upload(queue)?;
            }};
        }
        up!(particle_mass);
        up!(particle_inv_mass);
        up!(particle_radius);
        up!(particle_flags);
        up!(particle_world);
        up!(initial_particle_q);
        up!(spring_a);
        up!(spring_b);
        up!(spring_rest_length);
        up!(spring_compliance);
        up!(spring_max_stretch);
        up!(spring_color_offsets);
        up!(spring_indices_by_color);
        up!(tri_indices);
        up!(body_mass);
        up!(body_inv_mass);
        up!(body_inertia);
        up!(body_inv_inertia);
        up!(body_com);
        up!(body_flags);
        up!(body_world);
        up!(initial_body_q);
        up!(shape_body);
        up!(shape_geom_kind);
        up!(shape_geom_data);
        up!(shape_transform);
        up!(shape_flags);
        up!(joint_type);
        up!(joint_parent);
        up!(joint_child);
        up!(joint_x_p);
        up!(joint_x_c);
        up!(joint_axis);
        up!(joint_coord_offset);
        up!(joint_dof_offset);
        up!(initial_joint_q);
        up!(joint_limit_lower);
        up!(joint_limit_upper);
        up!(joint_friction);
        up!(joint_effort_limit);
        up!(joint_velocity_limit);
        up!(articulation_start);
        up!(articulation_world);
        up!(joint_world);
        up!(joint_enabled);
        up!(joint_ancestor);
        up!(joint_target_ke);
        up!(shape_collision_group);
        // Equality + mimic constraint arrays — added when the GPU
        // `XpbdEqualityKernel` replaced the host-side shim.
        up!(equality_type);
        up!(equality_body_a);
        up!(equality_body_b);
        up!(equality_anchor_a);
        up!(equality_anchor_b);
        up!(equality_relative_rotation);
        up!(equality_enabled);
        up!(gravity);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_model_has_one_world_with_gravity() {
        let m = Model::empty();
        assert_eq!(m.body_count, 0);
        assert_eq!(m.particle_count, 0);
        assert_eq!(m.world_count, 1);
        let g = m.gravity.host().unwrap();
        assert_eq!(g.len(), 1);
        assert!((g[0].y + crate::math::GRAVITY).abs() < 1e-12);
    }
}
