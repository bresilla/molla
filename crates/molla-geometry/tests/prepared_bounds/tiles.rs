use super::*;

#[test]
fn local_height_bounds_preserve_contacts_across_shapes_frames_and_tile_edges() {
    for rotation in [
        Quat::IDENTITY,
        Quat::from_rotation_x(0.31) * Quat::from_rotation_y(-0.2) * Quat::from_rotation_z(0.17),
    ] {
        for reverse in [false, true] {
            for kind in 0..6 {
                let mut builder = ModelBuilder::new();
                let mut heights = vec![0.0; 19 * 25];
                heights[8 * 25 + 8] = 1.0;
                heights[16 * 25 + 16] = 2.0;
                heights[18 * 25 + 24] = 4.0;
                let field = builder.add_heightfield(
                    Heightfield::from_scaled(19, 25, heights, Vec3::new(12.0, 1.0, 27.0)).unwrap(),
                );
                let pose = Transform::new(Vec3::new(5.0, 2.0, -7.0), rotation);
                let body = builder.add_body(BodyParams::new());
                if !reverse {
                    builder
                        .add_shape_heightfield(BodyId::NONE, field, pose)
                        .unwrap();
                }
                match kind {
                    0 => {
                        builder
                            .add_shape_sphere(body, 0.25, Transform::IDENTITY)
                            .unwrap();
                    }
                    1 => {
                        builder
                            .add_shape_box(body, Vec3::splat(0.25), Transform::IDENTITY)
                            .unwrap();
                    }
                    2 => {
                        builder
                            .add_shape_capsule(body, 0.2, 0.15, Transform::IDENTITY)
                            .unwrap();
                    }
                    3 => {
                        builder
                            .add_shape_cylinder(body, 0.2, 0.25, Transform::IDENTITY)
                            .unwrap();
                    }
                    4 => {
                        builder
                            .add_shape_round_cylinder(body, 0.2, 0.25, 0.05, Transform::IDENTITY)
                            .unwrap();
                    }
                    _ => {
                        let vertices = (0..8)
                            .map(|i| {
                                Vec3::new(
                                    if i & 1 == 0 { -0.25 } else { 0.25 },
                                    if i & 2 == 0 { -0.25 } else { 0.25 },
                                    if i & 4 == 0 { -0.25 } else { 0.25 },
                                )
                            })
                            .collect();
                        let hull = builder.add_hull(vertices);
                        builder
                            .add_shape_convex_hull(body, hull, Transform::IDENTITY)
                            .unwrap();
                    }
                }
                if reverse {
                    builder
                        .add_shape_heightfield(BodyId::NONE, field, pose)
                        .unwrap();
                }
                let model = builder.build().unwrap();
                let bounds = PreparedShapeBounds::new(&model);
                let mut state = model.state().unwrap();
                let mut optimized = CollisionPipeline::new();
                let mut oracle = CollisionPipeline::new();
                let mut a = Contacts::with_capacity(64);
                let mut b = Contacts::with_capacity(64);
                let mut pruned = 0;
                let mut contacts = 0;
                for x in [-6.0, -2.001, -2.0, -1.999, 0.0, 2.0, 5.999, 6.0] {
                    for z in [-13.5, -1.501, -1.5, -1.499, 0.0, 10.5, 13.5] {
                        for y in [-0.2, 0.25, 1.5, 4.1] {
                            state.body_q.host_mut().unwrap()[body.index()] = Transform::new(
                                pose.transform_point(Vec3::new(x, y, z)),
                                rotation * Quat::from_rotation_z(0.1),
                            );
                            optimized
                                .collide_prepared(&bounds, &state, &mut a, |_, _| false)
                                .unwrap();
                            oracle.collide(&model, &state, &mut b).unwrap();
                            equal_contacts(&a, &b);
                            assert_eq!(
                                optimized.last_broad_phase_pair_count(),
                                oracle.last_broad_phase_pair_count()
                            );
                            pruned += oracle.last_cpu_narrow_phase_calls()
                                - optimized.last_cpu_narrow_phase_calls();
                            contacts += a.count;
                        }
                    }
                }
                assert!(pruned > 0 && contacts > 0, "kind={kind}, reverse={reverse}");
            }
        }
    }
}

#[test]
fn scene_tile_cache_refreshes_when_local_peaks_change() {
    use molla_geometry::shape_world::SceneShapeBounds;
    use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, RigidScene};

    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.25 });
    ball.parent = Some(body);
    ball.pose.position = Vec3::new(-6.0, 1.0, -6.0);
    scene.insert_collider(ball).unwrap();
    let terrain = |height| {
        let mut heights = vec![0.0; 17 * 17];
        heights[16 * 17 + 16] = 3.0;
        for z in 0..=4 {
            for x in 0..=4 {
                heights[z * 17 + x] = height;
            }
        }
        ColliderDesc::new(ColliderGeometry::Heightfield(
            Heightfield::from_scaled(17, 17, heights, Vec3::new(16.0, 1.0, 16.0)).unwrap(),
        ))
    };
    let ground = scene.insert_collider(terrain(0.0)).unwrap();
    let mut cache = SceneShapeBounds::default();
    let mut optimized = CollisionPipeline::new();
    let mut oracle = CollisionPipeline::new();
    let mut a = Contacts::with_capacity(64);
    let mut b = Contacts::with_capacity(64);
    for height in [0.0, 0.9, 0.0, 0.9] {
        scene.replace_collider(ground, terrain(height)).unwrap();
        let bounds = cache.prepare(&scene);
        optimized
            .collide_prepared(&bounds, scene.state(), &mut a, |_, _| false)
            .unwrap();
        oracle
            .collide(scene.model(), scene.state(), &mut b)
            .unwrap();
        equal_contacts(&a, &b);
        assert_eq!(a.count > 0, height > 0.0);
        assert_eq!(
            optimized.last_cpu_narrow_phase_calls(),
            usize::from(height > 0.0)
        );
        assert_eq!(oracle.last_cpu_narrow_phase_calls(), 1);
        assert_eq!(optimized.last_broad_phase_pair_count(), 1);
    }
}
