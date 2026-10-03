#[path = "../benches/primitives/fixtures.rs"]
mod fixtures;
#[path = "../benches/primitives/validation.rs"]
mod validation;

use molla_math::Vec3;
use molla_sim::runtime::{BodyDesc, RigidScene};
use molla_solvers::rigid_world::RigidWorld;

#[test]
fn all_benchmark_fixtures_advance_exactly_without_quarantine() {
    for &kind in fixtures::CASES {
        let (scene, _) = fixtures::build(kind, 1).unwrap();
        let mut world = RigidWorld::from_scene(scene).unwrap();
        world.solver.substeps = 4;
        world.set_reaction_readout(true).unwrap();
        for expected in 1..=3 {
            world.step(1.0 / 240.0).unwrap();
            validation::check_frame(&world, expected).unwrap();
        }
        validation::check_final_state(&world).unwrap();
    }
}

#[test]
fn benchmark_guard_rejects_unadvanced_and_wrong_step_counts() {
    let mut world = RigidWorld::from_scene(RigidScene::new().unwrap()).unwrap();
    assert!(validation::check_frame(&world, 1).is_err());
    validation::check_frame(&world, 0).unwrap();
    world.step(1.0 / 240.0).unwrap();
    validation::check_frame(&world, 1).unwrap();
    assert!(validation::check_frame(&world, 0).is_err());
}

#[test]
fn benchmark_guard_rejects_a_successful_step_with_quarantine() {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    let index = world.scene.body_index(body).unwrap().index();
    world.scene.state_mut().body_q.host_mut().unwrap()[index]
        .position
        .x = f64::NAN;
    world.step(1.0 / 240.0).unwrap();
    assert_eq!(world.quarantined_bodies(), &[body]);
    let error = validation::check_frame(&world, 1).unwrap_err();
    assert!(error.to_string().contains("quarantined"));
}

#[test]
fn benchmark_final_guard_rejects_nonfinite_last_output_without_another_step() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.step(1.0 / 240.0).unwrap();
    let index = world.scene.body_index(body).unwrap().index();
    world.scene.state_mut().body_qd.host_mut().unwrap()[index]
        .angular
        .x = f64::NAN;
    validation::check_frame(&world, 1).unwrap();
    assert!(validation::check_final_state(&world).is_err());
}
