use std::collections::BTreeSet;
use std::sync::Arc;

use molla_math::{Transform, Vec3};
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, ColliderHandle, RigidScene};
use molla_solvers::rigid_world::RigidWorld;

fn compound_scene() -> (RigidScene, ColliderHandle, ColliderHandle) {
    let mut scene = RigidScene::new().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let mut body = BodyDesc::default();
    body.params.pose = Transform::from_translation(Vec3::Y * 0.4);
    let body = scene.insert_body(body).unwrap();
    let parts = [-1.0, 1.0]
        .into_iter()
        .map(|center| {
            let mut points = Vec::new();
            for x in [-0.3, 0.3] {
                for y in [-0.5, 0.5] {
                    for z in [-0.3, 0.3] {
                        points.push(Vec3::new(center + x, y, z));
                    }
                }
            }
            points
        })
        .collect();
    let mut desc = ColliderDesc::new(ColliderGeometry::ConvexParts(parts));
    desc.parent = Some(body);
    let compound = scene.insert_collider(desc).unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(4.0, 0.5, 4.0),
    });
    ground.pose = Transform::from_translation(-Vec3::Y * 0.5);
    let ground = scene.insert_collider(ground).unwrap();
    (scene, compound, ground)
}

#[test]
fn compound_contacts_aggregate_piece_forces_under_one_collider_pair() {
    let (scene, compound, ground) = compound_scene();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.step(1e-5).unwrap();
    let pairs = world.contact_pairs();
    assert_eq!(pairs.len(), 1, "{pairs:?}");
    assert_eq!(pairs[0].colliders, [compound, ground]);
    let pieces: BTreeSet<_> = pairs[0]
        .points
        .iter()
        .filter(|p| p.acted)
        .map(|p| p.colliders[0].unwrap().index())
        .collect();
    assert_eq!(
        pieces,
        world
            .scene
            .collider_indices(compound)
            .unwrap()
            .iter()
            .map(|s| s.index())
            .collect()
    );
    assert!(
        pairs[0]
            .points
            .iter()
            .all(|p| p.normal_impulse.is_finite() && p.normal_impulse >= 0.0)
    );
    let mut desc = world.scene.collider(compound).unwrap().clone();
    desc.sensor = true;
    world
        .scene
        .replace_collider(compound, desc.clone())
        .unwrap();
    assert!(world.contact_pairs().is_empty());
    world.step(1e-5).unwrap();
    let pairs = world.contact_pairs();
    assert_eq!(pairs.len(), 1);
    assert!(
        pairs[0]
            .points
            .iter()
            .all(|p| !p.acted && p.normal_force == 0.0)
    );
    desc.enabled = false;
    world.scene.replace_collider(compound, desc).unwrap();
    world.step(1e-5).unwrap();
    assert!(world.contact_pairs().is_empty());
}

#[test]
fn replacing_scene_with_equal_revision_does_not_expose_old_contacts() {
    let (scene, _, _) = compound_scene();
    let mut world = RigidWorld::from_scene(scene).unwrap();
    world.step(1e-5).unwrap();
    assert!(!world.contact_pairs().is_empty());
    let (replacement, _, _) = compound_scene();
    assert_eq!(replacement.revision(), world.scene.revision());
    world.scene = replacement;
    assert!(world.contact_pairs().is_empty());
}

#[test]
fn query_cache_preserves_send_sync_and_supports_shared_world_reads() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RigidWorld>();
    let (scene, compound, _) = compound_scene();
    let world = Arc::new(RigidWorld::from_scene(scene).unwrap());
    let workers: Vec<_> = (0..4)
        .map(|_| {
            let world = Arc::clone(&world);
            std::thread::spawn(move || {
                for _ in 0..32 {
                    let hit = world
                        .cast_ray(Vec3::new(1.0, 3.0, 0.0), -Vec3::Y, 5.0)
                        .unwrap()
                        .unwrap();
                    assert_eq!(hit.collider, compound);
                    assert!((hit.distance - 2.1).abs() < 1e-10);
                }
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
}

#[test]
fn panicking_query_predicate_does_not_disable_later_queries() {
    let (scene, compound, _) = compound_scene();
    let world = RigidWorld::from_scene(scene).unwrap();
    let origin = Vec3::new(1.0, 3.0, 0.0);
    let panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.cast_ray_filtered(origin, -Vec3::Y, 5.0, |_| panic!("predicate failed"))
    }));
    assert!(panic.is_err());
    assert_eq!(
        world
            .cast_ray(origin, -Vec3::Y, 5.0)
            .unwrap()
            .unwrap()
            .collider,
        compound
    );
}
