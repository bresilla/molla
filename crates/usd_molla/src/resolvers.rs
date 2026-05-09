//! Multi-schema resolver trait and concrete implementations.
//!
//! USD physics opinions can live in three different attribute
//! namespaces depending on the authoring tool:
//!
//! - **Standard `UsdPhysics`** (Pixar / OpenUSD spec) — `physics:*`
//!   attributes, `PhysicsRigidBodyAPI`, etc. Handled by
//!   [`SchemaResolverDefault`].
//! - **NVIDIA PhysX USD extension** — `physxRigidBody:*`,
//!   `physxJoint:*`, `physxCollision:*`. Handled by
//!   [`SchemaResolverPhysx`].
//! - **MuJoCo USD extension** (Newton's `MjcPhysicsAPI`) — `mjc:*`.
//!   Handled by [`SchemaResolverMjc`].
//!
//! Each resolver implements [`SchemaResolver`] and surfaces a small
//! set of "decoded" structs ([`BodyAttrs`], [`JointAttrs`],
//! [`ShapeAttrs`], [`SceneAttrs`]) that are then merged into a
//! `molla::Model` by the importer pipeline. Methods return `Option<_>`:
//! `None` means "this resolver doesn't handle this prim", and the
//! manager moves on to the next resolver in priority order.
//!
//! # Resolution chain
//!
//! ```text
//! prim → resolver[0] → resolver[1] → ... → resolver[N-1]
//!         ↓ Some(_)?    ↓ Some(_)?         ↓ Some(_)?
//!        return        return              return
//! ```
//!
//! The first resolver that returns `Some(_)` wins. Authors with
//! mixed-schema USDs (e.g. PhysX bodies + standard joints) can put
//! the more specific resolver first so it claims its prims, then the
//! default resolver catches the rest.
//!
//! # Mapping cheat sheet
//!
//! | Standard `UsdPhysics`           | PhysX                                 | MJC                                |
//! |---------------------------------|---------------------------------------|------------------------------------|
//! | `physics:mass`                  | (n/a — uses `physics:mass`)           | `mjc:mass` *(non-standard)*        |
//! | `physics:rigidBodyEnabled`      | `physxRigidBody:disableGravity` (inv) | `mjc:body:gravcomp`               |
//! | `physics:axis` (X/Y/Z)          | `physxJoint:axis`                     | `mjc:joint:axis`                   |
//! | `physics:lowerLimit/upperLimit` | `physxJoint:lowerLimit/upperLimit`    | `mjc:joint:range:min/max`          |
//! | `physics:radius`                | (n/a — geometric prim)                | `mjc:geom:size`                    |
//! | `physics:gravityMagnitude`      | `physxScene:gravityMagnitude`         | `mjc:option:gravity`               |
//!
//! Many PhysX / MJC attributes have no molla equivalent (linear /
//! angular damping, friction-loss, solver-time-step) — these are
//! deliberately skipped in v1; the resolver carries them through as
//! "extra" maps for future consumers.
//!
//! See `xtra/newton/newton/_src/usd/schemas.py` for the reference set
//! of mappings Newton implements.

use anyhow::Result;
use openusd::physics::JointKind;
use openusd::sdf::{self, FieldKey, Path, Value};
use openusd::Stage;

// ────────────────────────────────────────────────────────────────────────
// resolver-output structs
// ────────────────────────────────────────────────────────────────────────

/// Decoded rigid-body attributes returned by [`SchemaResolver::resolve_body`].
///
/// All fields are `Option<_>` so a resolver can return only what its
/// own schema authored. Unauthored fields fall back to molla defaults
/// (mass = 1e-6, identity inertia, dynamic flag).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BodyAttrs {
    /// `true` when the prim has a rigid-body opinion in this schema.
    /// Used by the manager to decide whether to call into this resolver
    /// for body-related queries.
    pub present: bool,
    pub rigid_body_enabled: Option<bool>,
    pub kinematic_enabled: Option<bool>,
    pub starts_asleep: Option<bool>,
    pub mass: Option<f32>,
    pub center_of_mass: Option<[f32; 3]>,
    pub diagonal_inertia: Option<[f32; 3]>,
    pub linear_velocity: Option<[f32; 3]>,
    /// USD authors angular velocity in degrees per second.
    pub angular_velocity: Option<[f32; 3]>,
    /// `physics:simulationOwner` — USD prim path of the owning scene.
    pub simulation_owner: Option<String>,
}

/// Decoded joint attributes returned by [`SchemaResolver::resolve_joint`].
#[derive(Debug, Clone, PartialEq)]
pub struct JointAttrs {
    pub present: bool,
    pub kind: JointKind,
    pub body0: Option<String>,
    pub body1: Option<String>,
    pub local_pos0: [f32; 3],
    pub local_rot0: [f32; 4],
    pub local_pos1: [f32; 3],
    pub local_rot1: [f32; 4],
    /// `"X" | "Y" | "Z"` for revolute / prismatic.
    pub axis: Option<String>,
    /// Lower limit (degrees for revolute, scene units for prismatic).
    pub lower_limit: Option<f32>,
    pub upper_limit: Option<f32>,
    pub joint_enabled: bool,
    pub break_force: Option<f32>,
    pub break_torque: Option<f32>,
    pub min_distance: Option<f32>,
    pub max_distance: Option<f32>,
}

impl Default for JointAttrs {
    fn default() -> Self {
        Self {
            present: false,
            kind: JointKind::Generic,
            body0: None,
            body1: None,
            local_pos0: [0.0; 3],
            local_rot0: [1.0, 0.0, 0.0, 0.0],
            local_pos1: [0.0; 3],
            local_rot1: [1.0, 0.0, 0.0, 0.0],
            axis: None,
            lower_limit: None,
            upper_limit: None,
            joint_enabled: true,
            break_force: None,
            break_torque: None,
            min_distance: None,
            max_distance: None,
        }
    }
}

/// Decoded collider attributes returned by [`SchemaResolver::resolve_shape`].
///
/// The shape's geometric kind (sphere / box / capsule / mesh) still
/// comes from the prim's `typeName`; this struct only carries
/// schema-specific physics opinions.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ShapeAttrs {
    pub present: bool,
    pub collision_enabled: Option<bool>,
    pub static_friction: Option<f32>,
    pub dynamic_friction: Option<f32>,
    pub restitution: Option<f32>,
    pub density: Option<f32>,
    /// PhysX rest-offset / MuJoCo gap → molla contact margin.
    pub contact_margin: Option<f32>,
}

/// Decoded scene-level attributes returned by [`SchemaResolver::resolve_scene`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SceneAttrs {
    pub present: bool,
    pub gravity_direction: Option<[f32; 3]>,
    pub gravity_magnitude: Option<f32>,
}

// ────────────────────────────────────────────────────────────────────────
// the trait
// ────────────────────────────────────────────────────────────────────────

/// Strategy interface for reading molla-relevant physics opinions
/// from a USD prim under a particular attribute schema.
///
/// Each `resolve_*` method returns:
/// - `Ok(Some(attrs))` — this resolver claims the prim and the
///   manager should stop here.
/// - `Ok(None)` — this resolver doesn't have an opinion; try the next.
/// - `Err(_)` — a real I/O / decode error; abort import.
///
/// The default impls all return `Ok(None)` so a resolver can opt in
/// only to the prim categories it cares about (e.g. a hypothetical
/// `SchemaResolverActuator` might only override `resolve_actuator`).
pub trait SchemaResolver: Send + Sync {
    /// Human-readable name for diagnostics / logging.
    fn name(&self) -> &str;

    /// Decode rigid-body opinions on `prim`.
    fn resolve_body(&self, _stage: &Stage, _prim: &Path) -> Result<Option<BodyAttrs>> {
        Ok(None)
    }

    /// Decode joint opinions on `prim`.
    fn resolve_joint(&self, _stage: &Stage, _prim: &Path) -> Result<Option<JointAttrs>> {
        Ok(None)
    }

    /// Decode collider / shape opinions on `prim`.
    fn resolve_shape(&self, _stage: &Stage, _prim: &Path) -> Result<Option<ShapeAttrs>> {
        Ok(None)
    }

    /// Decode `PhysicsScene`-level opinions on `prim`.
    fn resolve_scene(&self, _stage: &Stage, _prim: &Path) -> Result<Option<SceneAttrs>> {
        Ok(None)
    }
}

/// Owned chain of resolvers tried in order. The first resolver that
/// returns `Some(_)` for a given prim wins.
#[derive(Default)]
pub struct ResolverChain {
    resolvers: Vec<Box<dyn SchemaResolver>>,
}

impl ResolverChain {
    /// A chain containing only the standard [`SchemaResolverDefault`].
    /// Equivalent to the historical (pre-stage-29) behaviour.
    pub fn standard() -> Self {
        Self { resolvers: vec![Box::new(SchemaResolverDefault)] }
    }

    /// A chain with no resolvers — `resolve_*` always returns `None`.
    /// Mostly useful for tests; production callers want at least the
    /// default resolver appended.
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn push(&mut self, r: Box<dyn SchemaResolver>) -> &mut Self {
        self.resolvers.push(r);
        self
    }

    pub fn with(mut self, r: Box<dyn SchemaResolver>) -> Self {
        self.push(r);
        self
    }

    pub fn names(&self) -> Vec<&str> {
        self.resolvers.iter().map(|r| r.name()).collect()
    }

    pub fn is_empty(&self) -> bool {
        self.resolvers.is_empty()
    }

    /// Run `resolve_body` through the chain, returning the first
    /// `Some(_)` (or `Ok(None)` when nothing matches).
    pub fn resolve_body(&self, stage: &Stage, prim: &Path) -> Result<Option<BodyAttrs>> {
        for r in &self.resolvers {
            if let Some(attrs) = r.resolve_body(stage, prim)? {
                if attrs.present {
                    return Ok(Some(attrs));
                }
            }
        }
        Ok(None)
    }

    /// Run `resolve_joint` through the chain.
    pub fn resolve_joint(&self, stage: &Stage, prim: &Path) -> Result<Option<JointAttrs>> {
        for r in &self.resolvers {
            if let Some(attrs) = r.resolve_joint(stage, prim)? {
                if attrs.present {
                    return Ok(Some(attrs));
                }
            }
        }
        Ok(None)
    }

    /// Run `resolve_shape` through the chain.
    pub fn resolve_shape(&self, stage: &Stage, prim: &Path) -> Result<Option<ShapeAttrs>> {
        for r in &self.resolvers {
            if let Some(attrs) = r.resolve_shape(stage, prim)? {
                if attrs.present {
                    return Ok(Some(attrs));
                }
            }
        }
        Ok(None)
    }

    /// Run `resolve_scene` through the chain.
    pub fn resolve_scene(&self, stage: &Stage, prim: &Path) -> Result<Option<SceneAttrs>> {
        for r in &self.resolvers {
            if let Some(attrs) = r.resolve_scene(stage, prim)? {
                if attrs.present {
                    return Ok(Some(attrs));
                }
            }
        }
        Ok(None)
    }
}

// ────────────────────────────────────────────────────────────────────────
// shared attribute helpers (non-public; used by all concrete resolvers)
// ────────────────────────────────────────────────────────────────────────

fn read_attr(stage: &Stage, prim: &Path, name: &str) -> Option<Value> {
    let attr_path = prim.append_property(name).ok()?;
    stage.field(attr_path, FieldKey::Default).ok().flatten()
}

fn read_bool(stage: &Stage, prim: &Path, name: &str) -> Option<bool> {
    match read_attr(stage, prim, name)? {
        Value::Bool(b) => Some(b),
        _ => None,
    }
}

fn read_f32(stage: &Stage, prim: &Path, name: &str) -> Option<f32> {
    match read_attr(stage, prim, name)? {
        Value::Float(f) => Some(f),
        Value::Double(d) => Some(d as f32),
        _ => None,
    }
}

fn read_vec3f(stage: &Stage, prim: &Path, name: &str) -> Option<[f32; 3]> {
    match read_attr(stage, prim, name)? {
        Value::Vec3f(v) => Some(v),
        Value::Vec3d(v) => Some([v[0] as f32, v[1] as f32, v[2] as f32]),
        _ => None,
    }
}

fn read_token(stage: &Stage, prim: &Path, name: &str) -> Option<String> {
    match read_attr(stage, prim, name)? {
        Value::Token(s) | Value::String(s) => Some(s),
        _ => None,
    }
}

/// Try a list of attribute names in order, returning the first `f32`
/// value found. Lets resolvers expose multiple authoring spellings
/// (e.g. PhysX has both `physxScene:gravityMagnitude` and
/// `physics:gravityMagnitude`).
fn first_f32(stage: &Stage, prim: &Path, names: &[&str]) -> Option<f32> {
    for n in names {
        if let Some(v) = read_f32(stage, prim, n) {
            return Some(v);
        }
    }
    None
}

fn first_bool(stage: &Stage, prim: &Path, names: &[&str]) -> Option<bool> {
    for n in names {
        if let Some(v) = read_bool(stage, prim, n) {
            return Some(v);
        }
    }
    None
}

fn first_vec3f(stage: &Stage, prim: &Path, names: &[&str]) -> Option<[f32; 3]> {
    for n in names {
        if let Some(v) = read_vec3f(stage, prim, n) {
            return Some(v);
        }
    }
    None
}

/// Returns `true` when *any* of `names` is authored on `prim`. Used
/// by the namespaced resolvers to decide whether they "claim" a prim.
fn any_authored(stage: &Stage, prim: &Path, names: &[&str]) -> bool {
    names.iter().any(|n| read_attr(stage, prim, n).is_some())
}

// ────────────────────────────────────────────────────────────────────────
// SchemaResolverDefault — standard UsdPhysics
// ────────────────────────────────────────────────────────────────────────

/// Resolver for the standard `UsdPhysics` schema family. Reads
/// `physics:*` attributes plus `PhysicsRigidBodyAPI` /
/// `PhysicsCollisionAPI` API-schema applications.
///
/// This is the historical (pre-stage-29) reader path; behaviour is
/// preserved exactly so existing tests pass without changes.
pub struct SchemaResolverDefault;

impl SchemaResolver for SchemaResolverDefault {
    fn name(&self) -> &str {
        "UsdPhysics"
    }

    fn resolve_body(&self, stage: &Stage, prim: &Path) -> Result<Option<BodyAttrs>> {
        // Defer to openusd's reader so we stay in lock-step with the
        // upstream schema.
        let body = match openusd::physics::read_rigid_body(stage, prim)? {
            Some(b) => b,
            None => return Ok(None),
        };
        let mass = openusd::physics::read_mass(stage, prim)?.unwrap_or_default();
        Ok(Some(BodyAttrs {
            present: true,
            rigid_body_enabled: Some(body.rigid_body_enabled),
            kinematic_enabled: Some(body.kinematic_enabled),
            starts_asleep: Some(body.starts_asleep),
            mass: mass.mass,
            center_of_mass: mass.center_of_mass,
            diagonal_inertia: mass.diagonal_inertia,
            linear_velocity: body.velocity,
            angular_velocity: body.angular_velocity,
            simulation_owner: body.simulation_owner,
        }))
    }

    fn resolve_joint(&self, stage: &Stage, prim: &Path) -> Result<Option<JointAttrs>> {
        let j = match openusd::physics::read_joint(stage, prim)? {
            Some(j) => j,
            None => return Ok(None),
        };
        Ok(Some(JointAttrs {
            present: true,
            kind: j.kind,
            body0: j.body0,
            body1: j.body1,
            local_pos0: j.local_pos0,
            local_rot0: j.local_rot0,
            local_pos1: j.local_pos1,
            local_rot1: j.local_rot1,
            axis: j.axis,
            lower_limit: j.lower_limit,
            upper_limit: j.upper_limit,
            joint_enabled: j.joint_enabled,
            break_force: j.break_force,
            break_torque: j.break_torque,
            min_distance: j.min_distance,
            max_distance: j.max_distance,
        }))
    }

    fn resolve_shape(&self, stage: &Stage, prim: &Path) -> Result<Option<ShapeAttrs>> {
        let coll = match openusd::physics::read_collision_shape(stage, prim)? {
            Some(c) => c,
            None => return Ok(None),
        };
        // Resolve material if bound.
        let mut friction_s = None;
        let mut friction_d = None;
        let mut restitution = None;
        let mut density = None;
        if let Some(mat_path_str) = coll.physics_material_path.as_deref() {
            if let Ok(mat_path) = sdf::path(mat_path_str) {
                if let Some(mat) = openusd::physics::read_physics_material(stage, &mat_path)? {
                    friction_s = mat.static_friction;
                    friction_d = mat.dynamic_friction;
                    restitution = mat.restitution;
                    density = mat.density;
                }
            }
        }
        Ok(Some(ShapeAttrs {
            present: true,
            collision_enabled: Some(coll.collision_enabled),
            static_friction: friction_s,
            dynamic_friction: friction_d,
            restitution,
            density,
            contact_margin: None,
        }))
    }

    fn resolve_scene(&self, stage: &Stage, prim: &Path) -> Result<Option<SceneAttrs>> {
        let s = match openusd::physics::read_physics_scene(stage, prim)? {
            Some(s) => s,
            None => return Ok(None),
        };
        Ok(Some(SceneAttrs {
            present: true,
            gravity_direction: s.gravity_direction,
            gravity_magnitude: s.gravity_magnitude,
        }))
    }
}

// ────────────────────────────────────────────────────────────────────────
// SchemaResolverPhysx
// ────────────────────────────────────────────────────────────────────────

/// Resolver for the NVIDIA PhysX USD extension. Reads attributes in
/// the `physxRigidBody:*`, `physxJoint:*`, `physxCollision:*`,
/// `physxScene:*` and `physxMaterial:*` namespaces.
///
/// Mapping summary (PhysX → molla):
///
/// - **Body**: `physxRigidBody:disableGravity` (inverted) → ignored
///   for now (molla has no per-body gravity toggle); kinematic /
///   mass / inertia fall back to standard `physics:*` so authoring
///   tools that mix the two namespaces still work.
/// - **Joint**: `physxJoint:axis`, `physxJoint:lowerLimit`,
///   `physxJoint:upperLimit`, `physxJoint:breakForce`,
///   `physxJoint:breakTorque` → respective molla joint fields.
/// - **Collision**: `physxCollision:restOffset` → contact margin.
///   Contact-offset / friction-combine-mode / etc. carried through
///   but not yet consumed.
/// - **Scene**: `physxScene:gravityMagnitude` →
///   `Model.gravity[world]` (only authored when explicit).
///
/// Skipped (no molla equivalent in v1):
/// - `physxRigidBody:linearDamping` / `angularDamping`
/// - `physxRigidBody:maxLinearVelocity` / `maxAngularVelocity`
/// - `physxJoint:armature`
/// - `physxArticulation:*` (self-collision / solver iteration counts)
/// - `physxScene:enableCCD`, `physxScene:solverType`
pub struct SchemaResolverPhysx;

impl SchemaResolverPhysx {
    /// PhysX-namespaced attribute names this resolver knows about.
    /// Exposed so tests can assert coverage.
    pub fn known_body_attrs() -> &'static [&'static str] {
        &[
            "physxRigidBody:disableGravity",
            "physxRigidBody:linearDamping",
            "physxRigidBody:angularDamping",
            "physxRigidBody:maxLinearVelocity",
            "physxRigidBody:maxAngularVelocity",
            "physxRigidBody:linearVelocity",
            "physxRigidBody:angularVelocity",
            "physxRigidBody:mass",
            "physxRigidBody:centerOfMass",
            "physxRigidBody:diagonalInertia",
            "physxRigidBody:kinematicEnabled",
            "physxRigidBody:rigidBodyEnabled",
        ]
    }

    pub fn known_joint_attrs() -> &'static [&'static str] {
        &[
            "physxJoint:axis",
            "physxJoint:lowerLimit",
            "physxJoint:upperLimit",
            "physxJoint:breakForce",
            "physxJoint:breakTorque",
            "physxJoint:armature",
            "physxJoint:jointEnabled",
        ]
    }

    pub fn known_shape_attrs() -> &'static [&'static str] {
        &[
            "physxCollision:restOffset",
            "physxCollision:contactOffset",
            "physxCollision:collisionEnabled",
            "physxMaterial:staticFriction",
            "physxMaterial:dynamicFriction",
            "physxMaterial:restitution",
            "physxMaterial:density",
        ]
    }

    pub fn known_scene_attrs() -> &'static [&'static str] {
        &["physxScene:gravityMagnitude", "physxScene:gravityDirection"]
    }
}

impl SchemaResolver for SchemaResolverPhysx {
    fn name(&self) -> &str {
        "PhysX"
    }

    fn resolve_body(&self, stage: &Stage, prim: &Path) -> Result<Option<BodyAttrs>> {
        if !any_authored(stage, prim, Self::known_body_attrs()) {
            return Ok(None);
        }
        // PhysX's `disableGravity` is the inverse of molla's "rigid
        // body enabled" + we don't model per-body gravity toggling
        // separately; we read it for completeness but don't surface
        // it. The body presence flag is derived from `mass` /
        // `kinematicEnabled` being authored.
        Ok(Some(BodyAttrs {
            present: true,
            rigid_body_enabled: first_bool(stage, prim, &["physxRigidBody:rigidBodyEnabled"]),
            kinematic_enabled: first_bool(stage, prim, &["physxRigidBody:kinematicEnabled"]),
            starts_asleep: None,
            mass: first_f32(stage, prim, &["physxRigidBody:mass"]),
            center_of_mass: first_vec3f(stage, prim, &["physxRigidBody:centerOfMass"]),
            diagonal_inertia: first_vec3f(stage, prim, &["physxRigidBody:diagonalInertia"]),
            linear_velocity: first_vec3f(stage, prim, &["physxRigidBody:linearVelocity"]),
            angular_velocity: first_vec3f(stage, prim, &["physxRigidBody:angularVelocity"]),
            simulation_owner: None,
        }))
    }

    fn resolve_joint(&self, stage: &Stage, prim: &Path) -> Result<Option<JointAttrs>> {
        if !any_authored(stage, prim, Self::known_joint_attrs()) {
            return Ok(None);
        }
        // PhysX-namespaced joints still need a typed prim to know the
        // joint kind (Revolute / Prismatic / etc.). We delegate to the
        // standard reader for body refs / pose / kind, then override
        // with PhysX opinions where authored.
        let mut attrs = match openusd::physics::read_joint(stage, prim)? {
            Some(j) => JointAttrs {
                present: true,
                kind: j.kind,
                body0: j.body0,
                body1: j.body1,
                local_pos0: j.local_pos0,
                local_rot0: j.local_rot0,
                local_pos1: j.local_pos1,
                local_rot1: j.local_rot1,
                axis: j.axis,
                lower_limit: j.lower_limit,
                upper_limit: j.upper_limit,
                joint_enabled: j.joint_enabled,
                break_force: j.break_force,
                break_torque: j.break_torque,
                min_distance: j.min_distance,
                max_distance: j.max_distance,
            },
            None => JointAttrs { present: true, ..JointAttrs::default() },
        };
        if let Some(ax) = read_token(stage, prim, "physxJoint:axis") {
            attrs.axis = Some(ax);
        }
        if let Some(lo) = read_f32(stage, prim, "physxJoint:lowerLimit") {
            attrs.lower_limit = Some(lo);
        }
        if let Some(hi) = read_f32(stage, prim, "physxJoint:upperLimit") {
            attrs.upper_limit = Some(hi);
        }
        if let Some(bf) = read_f32(stage, prim, "physxJoint:breakForce") {
            attrs.break_force = Some(bf);
        }
        if let Some(bt) = read_f32(stage, prim, "physxJoint:breakTorque") {
            attrs.break_torque = Some(bt);
        }
        if let Some(en) = read_bool(stage, prim, "physxJoint:jointEnabled") {
            attrs.joint_enabled = en;
        }
        Ok(Some(attrs))
    }

    fn resolve_shape(&self, stage: &Stage, prim: &Path) -> Result<Option<ShapeAttrs>> {
        if !any_authored(stage, prim, Self::known_shape_attrs()) {
            return Ok(None);
        }
        Ok(Some(ShapeAttrs {
            present: true,
            collision_enabled: read_bool(stage, prim, "physxCollision:collisionEnabled"),
            static_friction: read_f32(stage, prim, "physxMaterial:staticFriction"),
            dynamic_friction: read_f32(stage, prim, "physxMaterial:dynamicFriction"),
            restitution: read_f32(stage, prim, "physxMaterial:restitution"),
            density: read_f32(stage, prim, "physxMaterial:density"),
            contact_margin: read_f32(stage, prim, "physxCollision:restOffset"),
        }))
    }

    fn resolve_scene(&self, stage: &Stage, prim: &Path) -> Result<Option<SceneAttrs>> {
        if !any_authored(stage, prim, Self::known_scene_attrs()) {
            return Ok(None);
        }
        Ok(Some(SceneAttrs {
            present: true,
            gravity_direction: read_vec3f(stage, prim, "physxScene:gravityDirection"),
            gravity_magnitude: read_f32(stage, prim, "physxScene:gravityMagnitude"),
        }))
    }
}

// ────────────────────────────────────────────────────────────────────────
// SchemaResolverMjc
// ────────────────────────────────────────────────────────────────────────

/// Resolver for the MuJoCo USD extension as authored by Newton's
/// `MjcPhysicsAPI` schema family. Reads attributes in the `mjc:*`
/// namespace.
///
/// Mapping summary (MJC → molla):
///
/// - **Body**: `mjc:body:mass`, `mjc:body:fullinertia` (only the
///   diagonal is consumed in v1), `mjc:body:pos` ignored (pose is
///   read from the prim's xformOp stack).
/// - **Joint**: `mjc:joint:axis` (vec3 → "X" / "Y" / "Z" closest
///   axis), `mjc:joint:range:min` / `max` → molla's lower/upper limits.
/// - **Geom**: `mjc:geom:friction` (vec3, slide-spin-roll; we use
///   the slide component) → static friction.
/// - **Scene** (`mjc:option:*`): `mjc:option:gravity` (vec3) →
///   gravity vector. Magnitude / direction split happens at the
///   importer boundary.
///
/// Skipped (no molla equivalent in v1):
/// - `mjc:joint:armature` / `frictionloss` / `damping` (joint-level
///   actuator dynamics — molla expresses these via `Control` /
///   `JointDrive`, not body-level fields).
/// - `mjc:geom:solref` / `solimp` (contact stiffness profiles).
/// - `mjc:option:timestep` / `iterations` (solver-level — molla
///   gets these from `SolverConfig`, not the model).
/// - `mjc:actuator:*`, `mjc:tendon:*` (out of v1 scope).
pub struct SchemaResolverMjc;

impl SchemaResolverMjc {
    pub fn known_body_attrs() -> &'static [&'static str] {
        &[
            "mjc:body:mass",
            "mjc:body:fullinertia",
            "mjc:body:diaginertia",
            "mjc:body:ipos",
            "mjc:body:gravcomp",
            "mjc:body:mocap",
        ]
    }

    pub fn known_joint_attrs() -> &'static [&'static str] {
        &[
            "mjc:joint:axis",
            "mjc:joint:range:min",
            "mjc:joint:range:max",
            "mjc:joint:armature",
            "mjc:joint:frictionloss",
            "mjc:joint:damping",
            "mjc:joint:limited",
        ]
    }

    pub fn known_shape_attrs() -> &'static [&'static str] {
        &[
            "mjc:geom:friction",
            "mjc:geom:density",
            "mjc:geom:solref",
            "mjc:geom:solimp",
            "mjc:geom:margin",
            "mjc:geom:gap",
            "mjc:geom:contype",
            "mjc:geom:conaffinity",
        ]
    }

    pub fn known_scene_attrs() -> &'static [&'static str] {
        &[
            "mjc:option:gravity",
            "mjc:option:timestep",
            "mjc:option:iterations",
            "mjc:option:integrator",
        ]
    }
}

impl SchemaResolver for SchemaResolverMjc {
    fn name(&self) -> &str {
        "MJC"
    }

    fn resolve_body(&self, stage: &Stage, prim: &Path) -> Result<Option<BodyAttrs>> {
        if !any_authored(stage, prim, Self::known_body_attrs()) {
            return Ok(None);
        }
        let mass = read_f32(stage, prim, "mjc:body:mass");
        let diag = read_vec3f(stage, prim, "mjc:body:diaginertia");
        let com = read_vec3f(stage, prim, "mjc:body:ipos");
        // `mjc:body:mocap = true` ↔ kinematic body in molla.
        let kinematic = read_bool(stage, prim, "mjc:body:mocap");
        Ok(Some(BodyAttrs {
            present: true,
            rigid_body_enabled: Some(true),
            kinematic_enabled: kinematic,
            starts_asleep: None,
            mass,
            center_of_mass: com,
            diagonal_inertia: diag,
            linear_velocity: None,
            angular_velocity: None,
            simulation_owner: None,
        }))
    }

    fn resolve_joint(&self, stage: &Stage, prim: &Path) -> Result<Option<JointAttrs>> {
        if !any_authored(stage, prim, Self::known_joint_attrs()) {
            return Ok(None);
        }
        // Joint kind / body refs / poses still come from the standard
        // joint reader when present, otherwise default. MuJoCo authors
        // joint topology via prim hierarchy (`<body><joint .../></body>`)
        // which usd_molla doesn't yet hierarchy-flatten — for v1 the
        // resolver only fills in axis + limits, and the test fixture
        // provides a typed Physics*Joint prim so body refs survive.
        let mut attrs = match openusd::physics::read_joint(stage, prim)? {
            Some(j) => JointAttrs {
                present: true,
                kind: j.kind,
                body0: j.body0,
                body1: j.body1,
                local_pos0: j.local_pos0,
                local_rot0: j.local_rot0,
                local_pos1: j.local_pos1,
                local_rot1: j.local_rot1,
                axis: j.axis,
                lower_limit: j.lower_limit,
                upper_limit: j.upper_limit,
                joint_enabled: j.joint_enabled,
                break_force: j.break_force,
                break_torque: j.break_torque,
                min_distance: j.min_distance,
                max_distance: j.max_distance,
            },
            None => JointAttrs { present: true, ..JointAttrs::default() },
        };
        // MuJoCo authors axis as a vec3 — map to nearest principal axis.
        if let Some(v) = read_vec3f(stage, prim, "mjc:joint:axis") {
            attrs.axis = Some(nearest_axis_token(v).to_string());
        }
        if let Some(lo) = read_f32(stage, prim, "mjc:joint:range:min") {
            attrs.lower_limit = Some(lo);
        }
        if let Some(hi) = read_f32(stage, prim, "mjc:joint:range:max") {
            attrs.upper_limit = Some(hi);
        }
        Ok(Some(attrs))
    }

    fn resolve_shape(&self, stage: &Stage, prim: &Path) -> Result<Option<ShapeAttrs>> {
        if !any_authored(stage, prim, Self::known_shape_attrs()) {
            return Ok(None);
        }
        // MuJoCo `friction` is `[slide, spin, roll]`; v1 maps the
        // slide component onto static + dynamic friction.
        let friction_slide = match read_attr(stage, prim, "mjc:geom:friction") {
            Some(Value::Vec3f(v)) => Some(v[0]),
            Some(Value::Vec3d(v)) => Some(v[0] as f32),
            Some(Value::FloatVec(v)) if !v.is_empty() => Some(v[0]),
            Some(Value::DoubleVec(v)) if !v.is_empty() => Some(v[0] as f32),
            Some(Value::Float(f)) => Some(f),
            Some(Value::Double(d)) => Some(d as f32),
            _ => None,
        };
        Ok(Some(ShapeAttrs {
            present: true,
            collision_enabled: None,
            static_friction: friction_slide,
            dynamic_friction: friction_slide,
            restitution: None,
            density: read_f32(stage, prim, "mjc:geom:density"),
            contact_margin: read_f32(stage, prim, "mjc:geom:margin"),
        }))
    }

    fn resolve_scene(&self, stage: &Stage, prim: &Path) -> Result<Option<SceneAttrs>> {
        if !any_authored(stage, prim, Self::known_scene_attrs()) {
            return Ok(None);
        }
        // MuJoCo authors gravity as a single vec3; we split into
        // direction + magnitude so the importer can populate
        // molla's normalised representation.
        let g = read_vec3f(stage, prim, "mjc:option:gravity");
        let (dir, mag) = match g {
            Some(g) => {
                let m = (g[0] * g[0] + g[1] * g[1] + g[2] * g[2]).sqrt();
                if m > 1e-6 {
                    (Some([g[0] / m, g[1] / m, g[2] / m]), Some(m))
                } else {
                    (Some(g), Some(0.0))
                }
            }
            None => (None, None),
        };
        Ok(Some(SceneAttrs {
            present: true,
            gravity_direction: dir,
            gravity_magnitude: mag,
        }))
    }
}

/// Map a vector to the closest principal axis token (`"X" | "Y" |
/// "Z"`). Used for MJC axes which are authored as vec3 rather than
/// the standard UsdPhysics token.
fn nearest_axis_token(v: [f32; 3]) -> &'static str {
    let ax = v[0].abs();
    let ay = v[1].abs();
    let az = v[2].abs();
    if ax >= ay && ax >= az {
        "X"
    } else if ay >= az {
        "Y"
    } else {
        "Z"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_axis_picks_largest_component() {
        assert_eq!(nearest_axis_token([1.0, 0.0, 0.0]), "X");
        assert_eq!(nearest_axis_token([0.0, 1.0, 0.0]), "Y");
        assert_eq!(nearest_axis_token([0.0, 0.0, 1.0]), "Z");
        assert_eq!(nearest_axis_token([0.1, 0.9, 0.2]), "Y");
        assert_eq!(nearest_axis_token([-0.7, 0.3, -0.1]), "X");
    }

    #[test]
    fn empty_chain_returns_none() {
        let chain = ResolverChain::empty();
        assert!(chain.is_empty());
        assert!(chain.names().is_empty());
    }

    #[test]
    fn standard_chain_has_default_resolver() {
        let chain = ResolverChain::standard();
        assert_eq!(chain.names(), vec!["UsdPhysics"]);
    }
}
