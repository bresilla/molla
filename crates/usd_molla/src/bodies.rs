//! `UsdPhysicsRigidBodyAPI` (+ `MassAPI`, world Xform) → `BodyParams`.
//!
//! Body world pose comes from the prim's composed Xform stack. Mass
//! priority follows the UsdPhysics spec:
//! 1. Explicit `physics:mass`  on `MassAPI`.
//! 2. `physics:density` × volume (from collider geometry) — TODO.
//! 3. Tiny safety mass + inertia so the body can take a step.
//!
//! Inertia: when `physics:diagonalInertia` is authored, it goes into
//! the diagonal of the body-frame inertia tensor; otherwise we fall
//! back to a unit identity scaled by `mass / 6` (rough sphere-ish
//! default).

use anyhow::Result;
use glam::DVec3;
use molla::core::{BodyId, WorldId};
use molla::math::{Mat3, Quat, Real, SpatialInertia, Transform, Vec3};
use molla::sim::{BodyFlags, BodyParams, ModelBuilder};
use openusd::physics::PhysicsPrims;
use openusd::sdf;
use openusd::Stage;
use std::collections::HashMap;

use crate::resolvers::{BodyAttrs, ResolverChain};
use crate::ImportInfo;

/// Iterate every prim that the resolver chain claims as a rigid body
/// (the standard `RigidBodyAPI` plus any extras the PhysX / MJC
/// resolvers contributed via `extend_prims_via_resolvers`), decode
/// mass / inertia / pose, and call `builder.add_body`.
///
/// Returns a path → `BodyId` map via `info.body_paths` (caller can
/// resolve via `info.body_paths.iter().position(|p| p == &path)`).
pub fn import_bodies(
    stage: &Stage,
    prims: &PhysicsPrims,
    builder: &mut ModelBuilder,
    info: &mut ImportInfo,
    resolvers: &ResolverChain,
) -> Result<()> {
    for path_str in &prims.rigid_bodies {
        let path = sdf::path(path_str)?;
        // Walk the resolver chain — first one that returns Some wins.
        let attrs: BodyAttrs = resolvers
            .resolve_body(stage, &path)?
            .unwrap_or_default();
        // Re-extract body / mass via the legacy structure for fields
        // the resolver doesn't surface (kinematic fallback, etc).
        let body = openusd::physics::read_rigid_body(stage, &path)?
            .unwrap_or_default();
        let mass = openusd::physics::read_mass(stage, &path)?
            .unwrap_or_default();
        // Resolver's authored fields take precedence over legacy
        // reads. The resolver may itself defer to the standard reader
        // (SchemaResolverDefault) so values agree by construction.
        let mass = openusd::physics::ReadMass {
            mass: attrs.mass.or(mass.mass),
            center_of_mass: attrs.center_of_mass.or(mass.center_of_mass),
            diagonal_inertia: attrs.diagonal_inertia.or(mass.diagonal_inertia),
            principal_axes: mass.principal_axes,
            density: mass.density,
        };
        let body = openusd::physics::ReadRigidBody {
            rigid_body_enabled: attrs.rigid_body_enabled.unwrap_or(body.rigid_body_enabled),
            kinematic_enabled: attrs.kinematic_enabled.unwrap_or(body.kinematic_enabled),
            starts_asleep: attrs.starts_asleep.unwrap_or(body.starts_asleep),
            velocity: attrs.linear_velocity.or(body.velocity),
            angular_velocity: attrs.angular_velocity.or(body.angular_velocity),
            simulation_owner: attrs.simulation_owner.or(body.simulation_owner),
        };

        // World pose from Xform stack. usd_schema's xform helper
        // composes the prim's xformOp stack; for an isolated test
        // scene we fall back to identity if it can't resolve.
        let world_pose = world_pose_for_prim(stage, path_str).unwrap_or_default();

        let mass_val = mass.mass.unwrap_or(0.0) as Real;
        let com_val = mass
            .center_of_mass
            .map(|c| Vec3::new(c[0] as f64, c[1] as f64, c[2] as f64))
            .unwrap_or(Vec3::ZERO);
        let inertia = match mass.diagonal_inertia {
            Some(d) => Mat3::from_diagonal(Vec3::new(
                d[0] as f64,
                d[1] as f64,
                d[2] as f64,
            )),
            None if mass_val > 0.0 => {
                // Sphere-ish fallback: I = (2/5) m r², r = 0.1.
                let r = 0.1_f64;
                let i = 2.0 / 5.0 * mass_val * r * r;
                Mat3::from_diagonal(Vec3::splat(i))
            }
            None => Mat3::ZERO,
        };

        let mut flags = BodyFlags::DYNAMIC;
        if !body.rigid_body_enabled {
            flags |= BodyFlags::KINEMATIC;
        }
        if body.kinematic_enabled {
            flags |= BodyFlags::KINEMATIC;
        }

        let body_id = builder.add_body(
            BodyParams::new()
                .with_spatial_inertia(SpatialInertia {
                    mass: mass_val.max(1e-6),
                    com: com_val,
                    inertia,
                })
                .with_pose(world_pose)
                .with_flags(flags)
                .with_world(WorldId(0)),
        );
        let _ = body_id; // index == info.body_paths.len() so we just push
        info.body_paths.push(path_str.clone());
    }
    Ok(())
}

/// Compose a prim's local pose from its Xform stack. For flat
/// scenes (no nested xforms) this is the world pose. Hierarchical
/// composition (walk parents and multiply) is a follow-up.
fn world_pose_for_prim(stage: &Stage, path_str: &str) -> Option<Transform> {
    let path = sdf::path(path_str).ok()?;
    let xform = usd_schema::xform::read_transform(stage, &path).ok()??;
    Some(Transform::new(
        Vec3::new(
            xform.translate[0] as f64,
            xform.translate[1] as f64,
            xform.translate[2] as f64,
        ),
        Quat::from_xyzw(
            xform.rotate[0] as f64,
            xform.rotate[1] as f64,
            xform.rotate[2] as f64,
            xform.rotate[3] as f64,
        ),
    ))
}

/// Resolve a USD prim path to its `BodyId` (if a body was imported
/// for that path). Used by joint / collider import to wire references.
pub fn body_id_for_path(info: &ImportInfo, path: &str) -> Option<BodyId> {
    info.body_paths
        .iter()
        .position(|p| p == path)
        .map(|i| BodyId::from_index(i))
}

/// Build a path → `BodyId` lookup table for hot loops where
/// repeatedly scanning `body_paths` would be O(N²). The keys are
/// owned strings so the table outlives borrows of `info`.
pub fn body_lookup(info: &ImportInfo) -> HashMap<String, BodyId> {
    info.body_paths
        .iter()
        .enumerate()
        .map(|(i, p)| (p.clone(), BodyId::from_index(i)))
        .collect()
}

// Suppress unused-import warning until joints.rs starts using it.
#[allow(dead_code)]
fn _hint(_v: DVec3) {}
