//! Pure-Rust UsdPhysics ↔ molla bridge.
//!
//! Two top-level functions:
//! - [`build_model_from_stage`] walks a composed [`openusd::Stage`],
//!   extracts every UsdPhysics opinion (`RigidBodyAPI`, `MassAPI`,
//!   `CollisionAPI`, `Joint*`, `PhysicsScene`, `PhysicsMaterial`,
//!   `CollisionGroup`), and produces a [`molla::Model`] that
//!   matches the authored intent.
//! - [`project_state_to_stage`] writes simulation results back as
//!   time samples on the same prims (`xformOp:translate`,
//!   `xformOp:orient`, `physics:linearVelocity`, `physics:angularVelocity`).
//!
//! The crate has **no Bevy dependency** — it consumes / produces
//! only `openusd::Stage` and `molla::{Model, State}` values. A Bevy
//! host (or the molla viewer) layers on top by wiring scene loading
//! into ECS systems.
//!
//! # Mapping cheat sheet
//!
//! | UsdPhysics                       | `molla::Model`                                    |
//! |----------------------------------|---------------------------------------------------|
//! | `UsdPhysicsScene`                | per-world gravity entry                           |
//! | `RigidBodyAPI` (+ `MassAPI`)     | `BodyParams` (mass / inertia / com / kinematic)   |
//! | `CollisionAPI` on a Sphere/Cube/…| `ShapeGeometry::{Sphere, Box, Capsule, …}`        |
//! | `PhysicsMaterial`                | per-shape friction / restitution                  |
//! | `RevoluteJoint`                  | `JointType::Revolute`                             |
//! | `PrismaticJoint`                 | `JointType::Prismatic`                            |
//! | `FixedJoint`                     | `JointType::Fixed`                                |
//! | `SphericalJoint`                 | `JointType::Ball`                                 |
//! | `DistanceJoint`                  | `JointType::Distance`                             |
//! | `Joint` (generic 6-DOF)          | `JointType::D6`                                   |
//! | `CollisionGroup`                 | `Model.shape_collision_group` bitmask             |
//! | `PhysicsLimitAPI:<dof>`          | `Model.joint_limit_lower / upper`                 |
//! | `PhysicsDriveAPI:<dof>`          | `Control.joint_target_*` defaults                 |
//! | `JOINT-mimic` (custom relation)  | `Model.constraint_mimic_*`                        |

pub mod attrs;
pub mod bodies;
pub mod colliders;
pub mod joints;
pub mod scene;
pub mod write_back;

use anyhow::Result;
use molla::sim::Model;
use openusd::Stage;

/// Per-call options for [`build_model_from_stage`].
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// If `true`, every joint without an explicit `physics:simulationOwner`
    /// is folded into world 0 even when multiple `PhysicsScene`s exist.
    /// Default `false` — multi-scene USDs spawn multiple molla worlds.
    pub merge_scenes: bool,
    /// Scale every linear quantity by this factor at import time.
    /// `metersPerUnit` from the stage metadata is multiplied into this
    /// — the default `1.0` (combined with metersPerUnit) is the right
    /// behaviour for almost all USDs.
    pub linear_scale_override: Option<f64>,
}

/// Bookkeeping returned from [`build_model_from_stage`] so callers
/// can map back from `molla` ids to USD prim paths (for writeback /
/// debug tooling).
#[derive(Debug, Default, Clone)]
pub struct ImportInfo {
    /// `body_id.index() → usd path` (in declaration order).
    pub body_paths: Vec<String>,
    /// `joint_id.index() → usd path`.
    pub joint_paths: Vec<String>,
    /// `shape_id.index() → usd path`.
    pub shape_paths: Vec<String>,
    /// `world_index → physics-scene path`.
    pub scene_paths: Vec<String>,
}

/// Walk `stage`, collect UsdPhysics opinions, and build a fully
/// populated [`molla::Model`]. The returned [`ImportInfo`] holds the
/// prim-path ↔ id maps so [`project_state_to_stage`] can find the
/// right prims when writing time samples back.
pub fn build_model_from_stage(
    stage: &Stage,
    options: &ImportOptions,
) -> Result<(Model, ImportInfo)> {
    use openusd::physics;

    let prims = physics::find_physics_prims(stage)?;
    let mut info = ImportInfo::default();
    let mut builder = molla::sim::ModelBuilder::new();

    scene::import_scenes(stage, &prims, &mut builder, &mut info, options)?;
    bodies::import_bodies(stage, &prims, &mut builder, &mut info)?;
    colliders::import_colliders(stage, &prims, &mut builder, &mut info)?;
    joints::import_joints(stage, &prims, &mut builder, &mut info)?;

    let model = builder.build().map_err(|e| anyhow::anyhow!("{e}"))?;
    Ok((model, info))
}

/// Write `state` back to `stage` as time samples at `time` seconds
/// for every body and joint that [`build_model_from_stage`] imported.
/// Bodies write `xformOp:translate` + `xformOp:orient`; joints write
/// nothing for now (joint-q time samples land in a follow-up).
pub fn project_state_to_stage(
    stage: &mut Stage,
    info: &ImportInfo,
    state: &molla::sim::State,
    time: f64,
) -> Result<()> {
    write_back::write_body_poses(stage, info, state, time)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Empty USDA stage produces an empty model.
    #[test]
    fn empty_stage_builds_empty_model() {
        // Author a minimal empty stage in memory.
        let usda = "#usda 1.0\n";
        let tmp = std::env::temp_dir().join("usd_molla_empty.usda");
        std::fs::write(&tmp, usda).unwrap();
        let stage = Stage::open(tmp.to_str().unwrap()).unwrap();
        let opts = ImportOptions::default();
        let (model, info) = build_model_from_stage(&stage, &opts).unwrap();
        assert_eq!(model.body_count, 0);
        assert!(info.body_paths.is_empty());
        std::fs::remove_file(&tmp).ok();
    }
}
