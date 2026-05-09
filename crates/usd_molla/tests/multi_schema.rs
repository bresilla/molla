//! Stage 29 — multi-schema USD reader tests.
//!
//! Hand-authored USDA fixtures exercise the three resolver paths
//! (PhysX-namespaced, MJC-namespaced, default `UsdPhysics`) plus a
//! mixed scene where the chain dispatches per-prim.
//!
//! The fixtures are intentionally minimal — enough to exercise the
//! resolver mapping, not real-world physics scenes (PhysX / MJC USD
//! exports in OSS-friendly licences are rare; the tests piggyback on
//! the standard `Physics*Joint` typenames so joint topology survives
//! the import even when the schema-specific attribute is the only
//! authoritative source for axis / limits).

use openusd::Stage;
use usd_molla::{
    build_model_from_stage, ImportOptions, SchemaResolverDefault, SchemaResolverMjc,
    SchemaResolverPhysx,
};

fn write_scene(usda: &str, slug: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "usd_molla_multi_schema_{}_{}.usda",
        slug,
        std::process::id()
    ));
    std::fs::write(&path, usda).unwrap();
    path
}

// ────────────────────────────────────────────────────────────────────────
// 1. PhysX-namespaced rigid body
// ────────────────────────────────────────────────────────────────────────

const PHYSX_BODY_USDA: &str = r#"#usda 1.0
(
    upAxis = "Y"
    metersPerUnit = 1.0
)

def Sphere "PxBall"
{
    double radius = 0.25
    double3 xformOp:translate = (0, 3, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    # PhysX-namespaced rigid-body attributes — no PhysicsRigidBodyAPI
    # applied. Standard reader sees nothing; SchemaResolverPhysx
    # picks it up.
    float physxRigidBody:mass = 2.5
    bool  physxRigidBody:rigidBodyEnabled = 1
    bool  physxRigidBody:kinematicEnabled = 0
    float3 physxRigidBody:diagonalInertia = (0.05, 0.05, 0.05)
    float3 physxRigidBody:linearVelocity = (0.0, 0.0, 0.0)
}
"#;

#[test]
fn physx_rigid_body_round_trips_via_resolver() {
    let path = write_scene(PHYSX_BODY_USDA, "physx_body");
    let stage = Stage::open(path.to_str().unwrap()).unwrap();

    let opts = ImportOptions::default().with_resolvers(vec![
        Box::new(SchemaResolverPhysx),
        Box::new(SchemaResolverDefault),
    ]);
    let (model, info) = build_model_from_stage(&stage, &opts).unwrap();

    assert_eq!(model.body_count, 1, "expected the PhysX-only body");
    assert_eq!(info.body_paths, vec!["/PxBall".to_string()]);

    let mass = model.body_mass.host().unwrap();
    assert!(
        (mass[0] - 2.5).abs() < 1e-5,
        "PhysX mass not surfaced: got {}",
        mass[0]
    );

    let q = model.initial_body_q.host().unwrap();
    let p = q[0].position;
    assert!(
        (p.y - 3.0).abs() < 1e-6,
        "PhysX body world translate not preserved: {:?}",
        p
    );

    std::fs::remove_file(&path).ok();
}

// ────────────────────────────────────────────────────────────────────────
// 2. MJC-namespaced joint
// ────────────────────────────────────────────────────────────────────────

const MJC_JOINT_USDA: &str = r#"#usda 1.0
(
    upAxis = "Y"
    metersPerUnit = 1.0
)

def Sphere "MjBase" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    double radius = 0.1
    float  physics:mass = 0.0
    bool   physics:rigidBodyEnabled = 0
    bool   physics:kinematicEnabled = 1
    double3 xformOp:translate = (0, 1, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

def Sphere "MjLink" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    double radius = 0.1
    float  physics:mass = 1.0
    double3 xformOp:translate = (0, 0.5, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

def PhysicsRevoluteJoint "MjHinge"
{
    rel physics:body0 = </MjBase>
    rel physics:body1 = </MjLink>

    # MJC namespace authors axis as a vec3 + range as separate
    # min/max scalars — molla's resolver maps these onto axis="X"
    # and lower/upper limit floats.
    float3 mjc:joint:axis = (1.0, 0.0, 0.0)
    float  mjc:joint:range:min = -45.0
    float  mjc:joint:range:max = 60.0
}
"#;

#[test]
fn mjc_joint_round_trips_via_resolver() {
    let path = write_scene(MJC_JOINT_USDA, "mjc_joint");
    let stage = Stage::open(path.to_str().unwrap()).unwrap();

    let opts = ImportOptions::default().with_resolvers(vec![
        Box::new(SchemaResolverMjc),
        Box::new(SchemaResolverDefault),
    ]);
    let (model, info) = build_model_from_stage(&stage, &opts).unwrap();

    assert_eq!(model.body_count, 2, "expected MjBase + MjLink");
    assert_eq!(info.joint_paths, vec!["/MjHinge".to_string()]);

    // Verify the joint limits came from MJC range:min/max.
    let lo = model.joint_limit_lower.host().unwrap();
    let hi = model.joint_limit_upper.host().unwrap();
    assert!(
        !lo.is_empty() && !hi.is_empty(),
        "expected at least one joint limit slot; lower={lo:?} upper={hi:?}"
    );
    // Revolute limits are stored in radians; -45° → -0.7854, 60° → 1.0472.
    let lo_deg = lo[0].to_degrees();
    let hi_deg = hi[0].to_degrees();
    assert!(
        (lo_deg - -45.0).abs() < 1e-3,
        "MJC lower limit not surfaced: {lo_deg} vs -45"
    );
    assert!(
        (hi_deg - 60.0).abs() < 1e-3,
        "MJC upper limit not surfaced: {hi_deg} vs 60"
    );

    std::fs::remove_file(&path).ok();
}

// ────────────────────────────────────────────────────────────────────────
// 3. Default resolver still handles standard UsdPhysics (back-compat)
// ────────────────────────────────────────────────────────────────────────

const STANDARD_USDA: &str = r#"#usda 1.0
(
    upAxis = "Y"
    metersPerUnit = 1.0
)

def PhysicsScene "PhysicsScene"
{
    vector3f physics:gravityDirection = (0, -1, 0)
    float    physics:gravityMagnitude = 9.806649
}

def Sphere "Ball" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI", "PhysicsCollisionAPI"]
)
{
    double radius = 0.5
    float  physics:mass = 1.0
    bool   physics:rigidBodyEnabled = 1
    double3 xformOp:translate = (0, 5, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
"#;

#[test]
fn default_resolver_back_compat() {
    let path = write_scene(STANDARD_USDA, "standard");
    let stage = Stage::open(path.to_str().unwrap()).unwrap();

    // No explicit resolver chain — default chain (= [Default]) used.
    let opts = ImportOptions::default();
    let (model, info) = build_model_from_stage(&stage, &opts).unwrap();

    assert_eq!(model.body_count, 1);
    assert!(model.shape_count >= 1);
    assert_eq!(info.body_paths, vec!["/Ball".to_string()]);
    let mass = model.body_mass.host().unwrap();
    assert!((mass[0] - 1.0).abs() < 1e-6);

    // Now do the same with an explicit-default chain — same result.
    let opts2 = ImportOptions::default()
        .with_resolvers(vec![Box::new(SchemaResolverDefault)]);
    let (model2, _) = build_model_from_stage(&stage, &opts2).unwrap();
    assert_eq!(model2.body_count, 1);
    let mass2 = model2.body_mass.host().unwrap();
    assert!((mass2[0] - 1.0).abs() < 1e-6);

    std::fs::remove_file(&path).ok();
}

// ────────────────────────────────────────────────────────────────────────
// 4. Multi-resolver: mixed PhysX + standard prims in one stage
// ────────────────────────────────────────────────────────────────────────

const MIXED_USDA: &str = r#"#usda 1.0
(
    upAxis = "Y"
    metersPerUnit = 1.0
)

def PhysicsScene "PhysicsScene"
{
    vector3f physics:gravityDirection = (0, -1, 0)
    float    physics:gravityMagnitude = 9.806649
}

# Standard UsdPhysics body — handled by SchemaResolverDefault.
def Sphere "StdBall" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    double radius = 0.5
    float  physics:mass = 1.0
    bool   physics:rigidBodyEnabled = 1
    double3 xformOp:translate = (0, 5, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}

# PhysX-only body — no PhysicsRigidBodyAPI, only physxRigidBody:* attrs.
# Handled by SchemaResolverPhysx.
def Sphere "PxBall"
{
    double radius = 0.25
    double3 xformOp:translate = (2, 5, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
    float  physxRigidBody:mass = 3.0
    bool   physxRigidBody:rigidBodyEnabled = 1
    float3 physxRigidBody:diagonalInertia = (0.07, 0.07, 0.07)
}
"#;

#[test]
fn multi_resolver_dispatches_per_prim() {
    let path = write_scene(MIXED_USDA, "mixed");
    let stage = Stage::open(path.to_str().unwrap()).unwrap();

    let opts = ImportOptions::default().with_resolvers(vec![
        Box::new(SchemaResolverPhysx),
        Box::new(SchemaResolverDefault),
    ]);
    let (model, info) = build_model_from_stage(&stage, &opts).unwrap();

    assert_eq!(
        model.body_count, 2,
        "expected both StdBall and PxBall imported"
    );
    // Both prim paths must show up — order is StdBall first
    // (find_physics_prims sweep) then PxBall (extend_prims_via_resolvers).
    assert!(info.body_paths.iter().any(|p| p == "/StdBall"));
    assert!(info.body_paths.iter().any(|p| p == "/PxBall"));

    // Each body's mass must come from the correct schema source.
    let mass = model.body_mass.host().unwrap();
    let std_idx = info
        .body_paths
        .iter()
        .position(|p| p == "/StdBall")
        .unwrap();
    let px_idx = info
        .body_paths
        .iter()
        .position(|p| p == "/PxBall")
        .unwrap();
    assert!(
        (mass[std_idx] - 1.0).abs() < 1e-6,
        "StdBall (UsdPhysics) mass not surfaced: {}",
        mass[std_idx]
    );
    assert!(
        (mass[px_idx] - 3.0).abs() < 1e-6,
        "PxBall (PhysX) mass not surfaced: {}",
        mass[px_idx]
    );

    std::fs::remove_file(&path).ok();
}

// ────────────────────────────────────────────────────────────────────────
// 5. Resolver priority — first resolver to claim a prim wins
// ────────────────────────────────────────────────────────────────────────

const PRIORITY_USDA: &str = r#"#usda 1.0
(
    upAxis = "Y"
    metersPerUnit = 1.0
)

# Prim with both PhysX and standard opinions for mass. With
# [PhysX, Default], PhysX wins; with [Default, PhysX], standard wins.
def Sphere "DualBall" (
    apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]
)
{
    double radius = 0.3
    float  physics:mass = 1.0
    float  physxRigidBody:mass = 7.5
    bool   physics:rigidBodyEnabled = 1
    bool   physxRigidBody:rigidBodyEnabled = 1
    double3 xformOp:translate = (0, 4, 0)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
"#;

#[test]
fn resolver_priority_first_wins() {
    let path = write_scene(PRIORITY_USDA, "priority");
    let stage = Stage::open(path.to_str().unwrap()).unwrap();

    // PhysX first → mass = 7.5.
    let opts_px_first = ImportOptions::default().with_resolvers(vec![
        Box::new(SchemaResolverPhysx),
        Box::new(SchemaResolverDefault),
    ]);
    let (model_px, _) = build_model_from_stage(&stage, &opts_px_first).unwrap();
    assert_eq!(model_px.body_count, 1);
    let mass_px = model_px.body_mass.host().unwrap()[0];
    assert!(
        (mass_px - 7.5).abs() < 1e-5,
        "PhysX-first chain didn't pick PhysX mass: {mass_px}"
    );

    // Default first → mass = 1.0.
    let opts_default_first = ImportOptions::default().with_resolvers(vec![
        Box::new(SchemaResolverDefault),
        Box::new(SchemaResolverPhysx),
    ]);
    let (model_def, _) = build_model_from_stage(&stage, &opts_default_first).unwrap();
    let mass_def = model_def.body_mass.host().unwrap()[0];
    assert!(
        (mass_def - 1.0).abs() < 1e-5,
        "Default-first chain didn't pick standard mass: {mass_def}"
    );

    std::fs::remove_file(&path).ok();
}
