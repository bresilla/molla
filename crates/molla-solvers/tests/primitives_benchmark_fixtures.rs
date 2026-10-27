#[path = "../benches/primitives/fixtures.rs"]
mod fixtures;

use molla_math::{SpatialVector, Vec3};

#[test]
fn staged_fixtures_preserve_counts_masses_and_initial_velocities() {
    for &kind in fixtures::CASES {
        for count in [1, 100] {
            let (scene, breaks) = fixtures::build(kind, count).unwrap();
            let bodies = if kind == "mixed" { 6 * count } else { count };
            let shapes = match kind {
                "mixed" => 4 * count,
                "friction" => 2 * count,
                _ => 0,
            };
            let break_count = match kind {
                "break" => count,
                "mixed" => 1,
                _ => 0,
            };
            assert_eq!(scene.model().body_count, bodies, "{kind}/{count}");
            assert_eq!(scene.model().articulation_count, bodies, "{kind}/{count}");
            assert_eq!(scene.model().shape_count, shapes, "{kind}/{count}");
            assert_eq!(breaks.len(), break_count, "{kind}/{count}");
            assert_eq!(scene.revision(), 0, "{kind}/{count}");
            assert!(
                scene
                    .model()
                    .body_mass
                    .host()
                    .unwrap()
                    .iter()
                    .all(|&m| m == 1.)
            );
            for (index, &velocity) in scene.state().body_qd.host().unwrap().iter().enumerate() {
                let friction = kind == "friction" || (kind == "mixed" && index % 6 == 5);
                let expected = if friction {
                    SpatialVector::new(Vec3::X * 0.1, Vec3::ZERO)
                } else {
                    SpatialVector::ZERO
                };
                assert_eq!(velocity, expected, "{kind}/{count}/{index}");
            }
        }
    }
}

#[test]
fn ten_thousand_independent_articulations_compile_without_intermediate_snapshots() {
    let begin = std::time::Instant::now();
    let (scene, breaks) = fixtures::build("distance", 10_000).unwrap();
    assert_eq!(scene.model().body_count, 10_000);
    assert_eq!(scene.model().articulation_count, 10_000);
    assert_eq!(scene.distance_joints().count(), 10_000);
    assert!(breaks.is_empty());
    assert_eq!(scene.revision(), 0);
    for (body, _) in scene.bodies() {
        assert!(scene.body_structure_stamp(body).is_some());
        assert!(scene.body_contact_stamp(body).is_some());
        assert!(scene.body_contact_episode_stamp(body).is_some());
    }
    println!(
        "distance_initial_construction\tinstances=10000\tseconds={:.9}",
        begin.elapsed().as_secs_f64()
    );
}
