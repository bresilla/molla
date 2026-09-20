use molla_core::BodyId;
use molla_geometry::shape_world::{PreparedShapeBounds, shape_world_aabb};
use molla_geometry::{CollisionPipeline, NarrowPhaseBackend};
use molla_math::{Quat, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Heightfield, Model, ModelBuilder, State};

fn fixture(side: usize) -> (Model, State) {
    let mut builder = ModelBuilder::new();
    let terrain = builder.add_heightfield(
        Heightfield::from_scaled(
            side,
            side,
            vec![0.0; side * side],
            Vec3::new(800.0, 1.0, 800.0),
        )
        .unwrap(),
    );
    builder
        .add_shape_heightfield(BodyId::NONE, terrain, Transform::IDENTITY)
        .unwrap();
    let moving_ground =
        builder.add_body(BodyParams::new().with_position(Vec3::new(10.0, -1.0, 0.0)));
    builder
        .add_shape_heightfield(moving_ground, terrain, Transform::IDENTITY)
        .unwrap();
    for x in [-1.0, 1.0] {
        let body = builder.add_body(BodyParams::new().with_position(Vec3::new(x, 0.15, 0.0)));
        builder
            .add_shape_sphere(body, 0.25, Transform::IDENTITY)
            .unwrap();
    }
    let model = builder.build().unwrap();
    let state = model.state().unwrap();
    (model, state)
}

fn equal_contacts(a: &Contacts, b: &Contacts) {
    assert_eq!(a.count, b.count);
    let n = a.count as usize;
    macro_rules! equal {
        ($($field:ident),+) => { $(assert_eq!(a.$field.host().unwrap()[..n], b.$field.host().unwrap()[..n], stringify!($field));)+ };
    }
    equal!(
        a_index,
        b_index,
        a_shape,
        b_shape,
        point,
        normal,
        depth,
        friction,
        rolling_friction,
        torsional_friction,
        restitution
    );
    assert_eq!(
        a.body_contact_offsets.host().unwrap(),
        b.body_contact_offsets.host().unwrap()
    );
    assert_eq!(
        a.body_contact_indices.host().unwrap(),
        b.body_contact_indices.host().unwrap()
    );
}

#[test]
fn prepared_contacts_match_fresh_bounds_across_substeps_and_filters() {
    let (model, mut state) = fixture(17);
    let bounds = PreparedShapeBounds::new(&model);
    for backend in [NarrowPhaseBackend::Gjk, NarrowPhaseBackend::MprCpu] {
        let mut cached = CollisionPipeline::new().with_narrow_phase_backend(backend);
        let mut oracle = CollisionPipeline::new().with_narrow_phase_backend(backend);
        let mut a = Contacts::with_capacity(64);
        let mut b = Contacts::with_capacity(64);
        let mut contacts_seen = 0;
        for tick in 0..40 {
            state.body_q.host_mut().unwrap()[0] = Transform::new(
                Vec3::new(10.0, -1.0 + tick as f64 * 0.05, 0.0),
                Quat::from_rotation_z(tick as f64 * 0.003),
            );
            state.body_q.host_mut().unwrap()[1].position.y = 0.1 + (tick % 7) as f64 * 0.04;
            let excluded = |a: Option<BodyId>, b: Option<BodyId>| {
                tick % 3 == 0 && [a, b].contains(&Some(BodyId(1)))
            };
            cached
                .collide_prepared(&bounds, &state, &mut a, excluded)
                .unwrap();
            oracle
                .collide_filtered(&model, &state, &mut b, excluded)
                .unwrap();
            equal_contacts(&a, &b);
            assert_eq!(
                cached.last_broad_phase_pair_count(),
                oracle.last_broad_phase_pair_count()
            );
            contacts_seen += a.count;
            for shape in 0..model.shape_count {
                for margin in [0.0, 0.01, 0.5] {
                    let a = bounds.world_aabb(&state, shape, margin);
                    let b = shape_world_aabb(&model, &state, shape, margin);
                    assert_eq!(a.min, b.min);
                    assert_eq!(a.max, b.max);
                }
            }
        }
        assert!(contacts_seen > 0);
    }
}

#[test]
fn new_borrow_observes_raw_heightfield_edits_and_replacements() {
    let (mut model, state) = fixture(9);
    for edit in 0..5 {
        match edit {
            0 => model.heightfields[0].heights[17] = 25.0,
            1 => model.heightfields[0].heights[17] = -40.0,
            2 => model.heightfields[0].origin = Vec3::new(2.0, -3.0, 7.0),
            3 => {
                model.heightfields[0] = Heightfield::from_scaled(
                    2,
                    3,
                    vec![1.0, -2.0, 3.0, 0.0, 9.0, -4.0],
                    Vec3::new(3.0, -2.0, 8.0),
                )
                .unwrap()
            }
            _ => model.heightfields.clear(),
        }
        let prepared = PreparedShapeBounds::new(&model);
        for shape in 0..model.shape_count {
            let a = prepared.world_aabb(&state, shape, 0.1);
            let b = shape_world_aabb(&model, &state, shape, 0.1);
            assert_eq!(a.min, b.min);
            assert_eq!(a.max, b.max);
        }
    }
}

#[test]
fn scene_cache_tracks_edits_removals_failed_edits_and_scene_replacement() {
    use molla_geometry::shape_world::SceneShapeBounds;
    use molla_sim::runtime::{BodyDesc, BodyKind, ColliderDesc, ColliderGeometry, RigidScene};

    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let fixed = scene
        .insert_body(BodyDesc {
            kind: BodyKind::Fixed,
            ..Default::default()
        })
        .unwrap();
    let mut ground_desc = ColliderDesc::new(ColliderGeometry::Heightfield(
        Heightfield::from_scaled(3, 3, vec![0.0; 9], Vec3::new(4.0, 1.0, 6.0)).unwrap(),
    ));
    ground_desc.parent = Some(fixed);
    let ground = scene.insert_collider(ground_desc.clone()).unwrap();
    let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    ball.parent = Some(body);
    scene.insert_collider(ball).unwrap();
    let mut cache = SceneShapeBounds::default();
    let mut cached = CollisionPipeline::new().with_sensor_overlaps(true);
    let mut oracle = CollisionPipeline::new().with_sensor_overlaps(true);
    let mut contacts_seen = 0;
    for edit in 0..12 {
        match edit {
            0 => {}
            1 => {
                let index = scene.body_index(fixed).unwrap().index();
                scene.state_mut().body_q.host_mut().unwrap()[index]
                    .position
                    .y = -0.2;
            }
            2 => {
                scene
                    .set_collider_pose(
                        ground,
                        Transform::new(Vec3::Y * 0.1, Quat::from_rotation_x(0.05)),
                    )
                    .unwrap();
            }
            3 => {
                ground_desc.geometry = ColliderGeometry::Heightfield(
                    Heightfield::from_scaled(2, 3, vec![0.1; 6], Vec3::new(5.0, 1.0, 8.0)).unwrap(),
                );
                scene.replace_collider(ground, ground_desc.clone()).unwrap();
            }
            4 => {
                let stamp = scene.stamp();
                let mut invalid = ground_desc.clone();
                invalid.pose.position.x = f64::NAN;
                assert!(scene.replace_collider(ground, invalid).is_err());
                assert_eq!(scene.stamp(), stamp);
            }
            5 => {
                scene.set_body_enabled(fixed, false).unwrap();
            }
            6 => {
                scene.set_body_enabled(fixed, true).unwrap();
            }
            7 => {
                ground_desc.geometry = ColliderGeometry::Cuboid {
                    half_extents: Vec3::ONE,
                };
                scene.replace_collider(ground, ground_desc.clone()).unwrap();
            }
            8 => {
                ground_desc.geometry = ColliderGeometry::Heightfield(
                    Heightfield::from_scaled(2, 2, vec![-0.1; 4], Vec3::new(3.0, 1.0, 2.0))
                        .unwrap(),
                );
                scene.replace_collider(ground, ground_desc.clone()).unwrap();
            }
            9 => {
                scene.remove_body(fixed).unwrap();
            }
            10 => {
                scene = RigidScene::new().unwrap();
            }
            _ => {
                ground_desc.parent = None;
                scene.insert_collider(ground_desc.clone()).unwrap();
            }
        }
        let bounds = cache.prepare(&scene);
        for shape in 0..scene.model().shape_count {
            let a = bounds.world_aabb(scene.state(), shape, 0.1);
            let b = shape_world_aabb(scene.model(), scene.state(), shape, 0.1);
            assert_eq!(a.min, b.min, "edit={edit}");
            assert_eq!(a.max, b.max, "edit={edit}");
        }
        let mut a = Contacts::with_capacity(64);
        let mut b = Contacts::with_capacity(64);
        cached
            .collide_prepared(&bounds, scene.state(), &mut a, |_, _| false)
            .unwrap();
        oracle
            .collide(scene.model(), scene.state(), &mut b)
            .unwrap();
        equal_contacts(&a, &b);
        contacts_seen += a.count;
    }
    assert!(contacts_seen > 0);
}

#[test]
#[ignore = "release timing of full-resolution terrain bounds"]
fn full_resolution_terrain_collision_timing() {
    let (model, state) = fixture(801);
    let mut pipeline = CollisionPipeline::new();
    let mut contacts = Contacts::with_capacity(64);
    for prepared in [false, true, true, false] {
        let started = std::time::Instant::now();
        for _ in 0..100 {
            let bounds = prepared.then(|| PreparedShapeBounds::new(&model));
            for _ in 0..8 {
                if let Some(bounds) = &bounds {
                    pipeline
                        .collide_prepared(bounds, &state, &mut contacts, |_, _| false)
                        .unwrap();
                } else {
                    pipeline.collide(&model, &state, &mut contacts).unwrap();
                }
                assert!(contacts.count > 0);
            }
        }
        eprintln!(
            "prepared={prepared} mean_8_substeps_ms={:.6}",
            started.elapsed().as_secs_f64() * 10.0
        );
    }
}
