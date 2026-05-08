//! End-to-end USD → molla → simulate → write-back round trip.
//!
//! Authors a minimal UsdPhysics scene in memory (a sphere falling
//! onto a plane), imports it into a molla `Model`, runs the XPBD
//! solver for ~1 second of sim time, and checks that the sphere
//! settled near `y = radius` — the same physics check from
//! `xpbd_floor::sphere_settles_on_plane`, but driven by an
//! authored USD scene rather than a hand-built `ModelBuilder`.

use molla::geometry::CollisionPipeline;
use molla::prelude::*;
use openusd::Stage;
use usd_molla::{build_model_from_stage, ImportOptions};

const SCENE_USDA: &str = r#"#usda 1.0
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

def Plane "Ground" (
    apiSchemas = ["PhysicsCollisionAPI"]
)
{
    uniform token axis = "Y"
}
"#;

fn write_scene(usda: &str) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!(
        "usd_molla_round_trip_{}.usda",
        std::process::id()
    ));
    std::fs::write(&path, usda).unwrap();
    path
}

#[test]
fn sphere_on_plane_round_trip() {
    // Author the scene to disk, open it as a Stage.
    let scene_path = write_scene(SCENE_USDA);
    let stage = Stage::open(scene_path.to_str().unwrap()).unwrap();

    // Import.
    let opts = ImportOptions::default();
    let (model, info) = build_model_from_stage(&stage, &opts).unwrap();
    eprintln!(
        "imported: {} bodies, {} joints, {} shapes, {} scenes",
        info.body_paths.len(),
        info.joint_paths.len(),
        info.shape_paths.len(),
        info.scene_paths.len()
    );

    assert_eq!(model.body_count, 1, "expected the Ball body");
    assert!(
        model.shape_count >= 1,
        "expected at least the Ball collider (got {})",
        model.shape_count
    );

    // Verify the sphere imported with the right mass + position.
    let body_q = model.initial_body_q.host().unwrap();
    let p = body_q[0].position;
    assert!(
        (p - Vec3::new(0.0, 5.0, 0.0)).length() < 1e-6,
        "Ball world translate not preserved: got {:?}",
        p
    );
    assert!(
        (model.body_mass.host().unwrap()[0] - 1.0).abs() < 1e-6,
        "Ball mass not 1.0"
    );

    // Simulate ~1 second.
    let mut s_a = model.state().unwrap();
    let mut s_b = model.state().unwrap();
    let control = model.control();
    let mut contacts = Contacts::with_capacity(64);
    let mut pipeline = CollisionPipeline::new();
    let mut solver = XpbdSolver::new().with_iterations(8);
    for _ in 0..400 {
        pipeline.collide(&model, &s_a, &mut contacts).unwrap();
        solver
            .step(&model, &s_a, &control, &contacts, 0.005, &mut s_b)
            .unwrap();
        std::mem::swap(&mut s_a, &mut s_b);
    }

    let q = s_a.body_q.host().unwrap();
    let y = q[0].position.y;
    eprintln!("ball settled at y = {y:.3}");
    // Sphere of radius 0.5 should rest near y = 0.5 on the ground.
    assert!(
        (0.5 - 5e-3..0.6).contains(&y),
        "ball didn't settle near y=0.5: y={y}"
    );

    std::fs::remove_file(&scene_path).ok();
}
