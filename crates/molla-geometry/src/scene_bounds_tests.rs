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

#[test]
fn mesh_and_hull_bounds_take_their_cached_box_through_the_pose() {
    use molla_math::Quat;
    use molla_sim::TriMesh;
    let points = vec![
        Vec3::ZERO,
        Vec3::new(4.0, 0.0, 0.0),
        Vec3::new(0.0, 2.0, 1.0),
        Vec3::new(0.5, 0.5, 3.0),
    ];
    let mut scene = RigidScene::new().unwrap();
    let mut turned = ColliderDesc::new(ColliderGeometry::TriMesh(TriMesh {
        vertices: points.clone(),
        indices: vec![[0, 1, 2]],
    }));
    turned.pose = Transform::new(Vec3::new(10.0, -3.0, 2.0), Quat::from_rotation_z(0.5));
    scene.insert_collider(turned).unwrap();
    let mut hull = ColliderDesc::new(ColliderGeometry::ConvexHull(points));
    hull.pose = Transform::new(Vec3::new(-5.0, 1.0, 0.0), Quat::IDENTITY);
    scene.insert_collider(hull).unwrap();
    let mut cache = SceneShapeBounds::default();
    let bounds = cache.prepare(&scene);
    let state = scene.state();
    for shape in 0..2 {
        let exact = shape_world_aabb(scene.model(), state, shape, 0.0);
        let cached = bounds.world_aabb(state, shape, 0.0);
        assert!(
            cached.min.cmple(exact.min + 1e-12).all() && cached.max.cmpge(exact.max - 1e-12).all(),
            "shape {shape}: {cached:?} misses {exact:?}"
        );
    }
    let exact = shape_world_aabb(scene.model(), state, 1, 0.0);
    let cached = bounds.world_aabb(state, 1, 0.0);
    assert!(
        (cached.min - exact.min).length() < 1e-12 && (cached.max - exact.max).length() < 1e-12,
        "an unturned hull's box is exact"
    );
}

#[test]
fn the_kept_static_sweep_pairs_what_brute_force_pairs_as_bodies_move() {
    use crate::broad_phase::{BroadPhase, brute_force_pairs};
    use molla_math::Quat;
    use molla_sim::runtime::BodyDesc;
    let mut scene = RigidScene::new().unwrap();
    let mut grid = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(0.5, 0.5, 0.5),
    });
    for k in 0..60 {
        grid.pose = Transform::new(
            Vec3::new(1.5 * (k % 10) as Real, 0.0, 1.5 * (k / 10) as Real),
            Quat::IDENTITY,
        );
        scene.insert_collider(grid.clone()).unwrap();
    }
    let mut ground = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(200.0, 0.1, 200.0),
    });
    ground.pose = Transform::new(Vec3::new(0.0, -0.6, 0.0), Quat::IDENTITY);
    scene.insert_collider(ground).unwrap();
    for k in 0..6 {
        let body = scene.insert_body(BodyDesc::default()).unwrap();
        let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.6 });
        ball.parent = Some(body);
        ball.pose = Transform::new(Vec3::new(2.0 * k as Real, 0.0, 3.0), Quat::IDENTITY);
        scene.insert_collider(ball).unwrap();
    }
    let mut cache = SceneShapeBounds::default();
    let mut broad = BroadPhase::new();
    let mut state = scene.state().clone_host().unwrap();
    for step in 0..4 {
        for q in state.body_q.host_mut().unwrap().iter_mut() {
            q.position += Vec3::new(0.7, 0.0, 0.9 * step as Real);
        }
        let bounds = cache.prepare(&scene);
        assert!(bounds.stamp().is_some());
        let (mut kept, mut every) = (Vec::new(), Vec::new());
        broad.sap_pairs_prepared(&bounds, &state, 0.01, &mut kept);
        brute_force_pairs(scene.model(), &state, 0.01, &mut every);
        let mut kept: Vec<_> = kept.iter().map(|p| (p.a, p.b)).collect();
        let mut every: Vec<_> = every.iter().map(|p| (p.a, p.b)).collect();
        kept.sort();
        every.sort();
        assert!(!every.is_empty(), "step {step}: the test scene has pairs");
        assert_eq!(kept, every, "step {step}");
    }
}
