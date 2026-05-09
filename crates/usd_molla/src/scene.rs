//! `UsdPhysicsScene` → molla `world_count + gravity[]`.
//!
//! Each authored `UsdPhysicsScene` becomes one molla world.
//! `physics:gravityDirection` × `physics:gravityMagnitude` fills
//! `Model.gravity[world]`. Scene paths are stored on
//! [`crate::ImportInfo::scene_paths`] for the body-import step to
//! resolve `physics:simulationOwner` rels.

use anyhow::Result;
use molla::math::Vec3;
use molla::sim::ModelBuilder;
use openusd::physics::PhysicsPrims;
use openusd::sdf;
use openusd::Stage;

use crate::resolvers::ResolverChain;
use crate::{ImportInfo, ImportOptions};

/// Process every `UsdPhysicsScene` (or PhysX / MJC scene-equivalent)
/// prim discovered by the resolver chain and populate per-world
/// gravity in the builder.
pub fn import_scenes(
    stage: &Stage,
    prims: &PhysicsPrims,
    builder: &mut ModelBuilder,
    info: &mut ImportInfo,
    options: &ImportOptions,
    resolvers: &ResolverChain,
) -> Result<()> {
    if prims.scenes.is_empty() {
        // No scene prim → caller gets the default world 0 with
        // standard gravity (-Y, 9.806 m/s²) that ModelBuilder::new
        // already populated.
        return Ok(());
    }

    let merge = options.merge_scenes;
    for (idx, scene_path_str) in prims.scenes.iter().enumerate() {
        let world_idx = if merge { 0 } else { idx };
        let path = sdf::path(scene_path_str)?;
        let scene = resolvers.resolve_scene(stage, &path)?.unwrap_or_default();
        let dir = scene
            .gravity_direction
            .map(|d| Vec3::new(d[0] as f64, d[1] as f64, d[2] as f64))
            .unwrap_or(Vec3::new(0.0, -1.0, 0.0));
        let mag = scene.gravity_magnitude.unwrap_or(9.806_65) as f64;
        builder.set_world_gravity(
            molla::core::WorldId(world_idx as u32),
            dir.normalize_or_zero() * mag,
        );
        info.scene_paths.push(scene_path_str.clone());
    }
    Ok(())
}
