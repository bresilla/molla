use super::*;
use molla_sim::Heightfield;
use molla_sim::runtime::{ColliderDesc, ColliderGeometry};

#[test]
fn scene_steps_reuse_storage_and_distinct_scenes_refresh_bounds() {
    let mut scene = RigidScene::new().unwrap();
    scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Heightfield(
            Heightfield::from_scaled(2, 2, vec![0.0, 1.0, 2.0, 3.0], Vec3::ONE).unwrap(),
        )))
        .unwrap();
    let mut cache = SceneShapeBounds::default();
    cache.prepare(&scene);
    let allocation = cache.heightfields.as_ptr();
    let tile_allocation = cache.tiles.as_ptr();
    let stamp = scene.stamp();
    for _ in 0..20 {
        scene
            .commit_step(scene.state().clone_host().unwrap())
            .unwrap();
        let bounds = cache.prepare(&scene);
        assert!(matches!(bounds.heightfields, Cow::Borrowed(_)));
        assert_eq!(bounds.heightfields.as_ptr(), allocation);
        assert!(matches!(bounds.tiles, Cow::Borrowed(_)));
        assert_eq!(bounds.tiles.as_ptr(), tile_allocation);
        assert_eq!(scene.stamp(), stamp);
    }
    let mut other = RigidScene::new().unwrap();
    other
        .insert_collider(ColliderDesc::new(ColliderGeometry::Heightfield(
            Heightfield::from_scaled(2, 2, vec![-9.0; 4], Vec3::ONE).unwrap(),
        )))
        .unwrap();
    let bounds = cache.prepare(&other);
    assert_eq!(bounds.heightfields[0].0.y, -9.0);
}
