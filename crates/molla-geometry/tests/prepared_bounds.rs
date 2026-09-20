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
