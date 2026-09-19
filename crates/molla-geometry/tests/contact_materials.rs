use molla_geometry::{CollisionPipeline, ShapePair, pair_contact, reduce_contacts};
use molla_math::{Transform, Vec3};
use molla_sim::{
    BodyParams, CoefficientCombineRule as Rule, ContactMaterial, Contacts, ModelBuilder,
};

#[test]
fn friction_combine_priority_is_symmetric_and_matches_narrow_phase() {
    let rules = [Rule::Average, Rule::Min, Rule::Multiply, Rule::Max];
    for a in rules {
        for b in rules {
            let expected = match a.max(b) {
                Rule::Average => 0.5,
                Rule::Min => 0.2,
                Rule::Multiply => 0.16,
                Rule::Max => 0.8,
            };
            assert!((a.combine(b, 0.2, 0.8) - expected).abs() < 1e-12);
            let mut builder = ModelBuilder::new();
            let ba = builder.add_body(BodyParams::new());
            let bb = builder.add_body(BodyParams::new().with_position(Vec3::X));
            let sa = builder
                .add_shape_sphere(ba, 0.6, Transform::IDENTITY)
                .unwrap();
            let sb = builder
                .add_shape_sphere(bb, 0.6, Transform::IDENTITY)
                .unwrap();
            builder.set_shape_material(sa, 0.2, 0.0).unwrap();
            builder.set_shape_material(sb, 0.8, 0.0).unwrap();
            builder
                .set_shape_contact_material(
                    sa,
                    ContactMaterial {
                        friction_combine: a,
                        ..ContactMaterial::default()
                    },
                )
                .unwrap();
            builder
                .set_shape_contact_material(
                    sb,
                    ContactMaterial {
                        friction_combine: b,
                        ..ContactMaterial::default()
                    },
                )
                .unwrap();
            let model = builder.build().unwrap();
            let state = model.state().unwrap();
            let mut manifold = Vec::new();
            pair_contact(
                &model,
                &state,
                ShapePair { a: sa.0, b: sb.0 },
                0.5,
                0.0,
                &mut manifold,
            );
            assert!((manifold[0].friction - expected).abs() < 1e-12);
            let mut contacts = Contacts::with_capacity(0);
            CollisionPipeline::new()
                .collide(&model, &state, &mut contacts)
                .unwrap();
            assert!((contacts.friction.host().unwrap()[0] - expected).abs() < 1e-12);
            assert_eq!(contacts.a_shape.host().unwrap()[0], sa.0);
            assert_eq!(contacts.b_shape.host().unwrap()[0], sb.0);
        }
    }
}

#[test]
fn membership_and_filter_are_checked_in_both_directions_using_all_32_bits() {
    for filter_b in [0, 1] {
        let mut builder = ModelBuilder::new();
        let a = builder.add_body(BodyParams::new());
        let b = builder.add_body(BodyParams::new());
        let sa = builder
            .add_shape_sphere(a, 0.5, Transform::IDENTITY)
            .unwrap();
        let sb = builder
            .add_shape_sphere(b, 0.5, Transform::IDENTITY)
            .unwrap();
        builder.set_shape_collision_groups(sa, 1, 1 << 31).unwrap();
        builder
            .set_shape_collision_groups(sb, 1 << 31, filter_b)
            .unwrap();
        let model = builder.build().unwrap();
        let mut contacts = Contacts::with_capacity(0);
        let count = CollisionPipeline::new()
            .collide(&model, &model.state().unwrap(), &mut contacts)
            .unwrap();
        assert_eq!(count, filter_b as usize);
    }
}

#[test]
fn callback_removes_only_selected_body_pairs_before_narrow_phase() {
    let mut builder = ModelBuilder::new();
    let bodies: Vec<_> = (0..3)
        .map(|_| builder.add_body(BodyParams::new()))
        .collect();
    for &body in &bodies {
        builder
            .add_shape_sphere(body, 0.5, Transform::IDENTITY)
            .unwrap();
    }
    let model = builder.build().unwrap();
    let mut pipeline = CollisionPipeline::new();
    let mut contacts = Contacts::with_capacity(0);
    pipeline
        .collide_filtered(&model, &model.state().unwrap(), &mut contacts, |a, b| {
            a == Some(bodies[0]) && b == Some(bodies[1])
        })
        .unwrap();
    assert_eq!(contacts.count, 2);
    assert_eq!(pipeline.last_cpu_narrow_phase_calls(), 2);
    pipeline
        .collide_filtered(&model, &model.state().unwrap(), &mut contacts, |_, _| true)
        .unwrap();
    assert_eq!(contacts.count, 0);
    assert_eq!(pipeline.last_cpu_narrow_phase_calls(), 0);
}

#[test]
fn reduction_preserves_distinct_collider_pairs_and_remaps_body_csr() {
    use molla_storage::{Buffer, BufferUsage};
    let mut contacts = Contacts::with_capacity(3);
    contacts.count = 3;
    contacts.a_index.host_mut().unwrap().fill(0);
    contacts.b_index.host_mut().unwrap().fill(1);
    contacts
        .a_shape
        .host_mut()
        .unwrap()
        .copy_from_slice(&[2, 2, 7]);
    contacts
        .b_shape
        .host_mut()
        .unwrap()
        .copy_from_slice(&[3, 3, 8]);
    contacts
        .depth
        .host_mut()
        .unwrap()
        .copy_from_slice(&[0.1, 0.3, 0.2]);
    contacts.body_contact_offsets = Buffer::from_vec(vec![0, 3, 6], BufferUsage::StorageReadWrite);
    let b = Contacts::B_SIDE_BIT;
    contacts.body_contact_indices = Buffer::from_vec(
        vec![0, 1, 2, b, b | 1, b | 2],
        BufferUsage::StorageReadWrite,
    );
    assert_eq!(reduce_contacts(&mut contacts, 1.0).unwrap(), 2);
    assert_eq!(&contacts.a_shape.host().unwrap()[..2], &[2, 7]);
    assert_eq!(&contacts.b_shape.host().unwrap()[..2], &[3, 8]);
    assert_eq!(&contacts.depth.host().unwrap()[..2], &[0.3, 0.2]);
    assert_eq!(contacts.body_contact_offsets.host().unwrap(), &[0, 2, 4]);
    assert_eq!(
        &contacts.body_contact_indices.host().unwrap()[..4],
        &[0, 1, b, b | 1]
    );
    let ids = molla_geometry::build_contact_id_table(&contacts, 1.0).unwrap();
    assert_ne!(ids[0].0, ids[1].0);
    assert_eq!(
        molla_geometry::match_contacts(&contacts, &ids, 1.0).unwrap(),
        vec![Some(0), Some(1)]
    );
}

#[test]
fn sensors_are_opt_in_to_contact_generation() {
    let mut builder = ModelBuilder::new();
    let a = builder.add_body(BodyParams::new());
    let b = builder.add_body(BodyParams::new());
    let sa = builder
        .add_shape_sphere(a, 0.5, Transform::IDENTITY)
        .unwrap();
    builder
        .add_shape_sphere(b, 0.5, Transform::IDENTITY)
        .unwrap();
    builder
        .set_shape_flags(
            sa,
            molla_sim::ShapeFlags::COLLIDES | molla_sim::ShapeFlags::SENSOR,
        )
        .unwrap();
    let model = builder.build().unwrap();
    let state = model.state().unwrap();
    let mut contacts = Contacts::with_capacity(0);
    assert_eq!(
        CollisionPipeline::new()
            .collide(&model, &state, &mut contacts)
            .unwrap(),
        0
    );
    assert_eq!(
        molla_geometry::collide(&model, &state, &mut contacts, &mut Vec::new()).unwrap(),
        0
    );
    assert_eq!(
        CollisionPipeline::new()
            .with_sensor_overlaps(true)
            .collide(&model, &state, &mut contacts)
            .unwrap(),
        1
    );
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU adapter"]
fn gpu_narrow_phase_retains_material_rules_and_body_filtering() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let a = builder.add_body(BodyParams::new());
    let b = builder.add_body(BodyParams::new().with_position(Vec3::new(0.8, 0.17, 0.13)));
    let sa = builder
        .add_shape_box(a, Vec3::splat(0.5), Transform::IDENTITY)
        .unwrap();
    let sb = builder
        .add_shape_box(b, Vec3::splat(0.5), Transform::IDENTITY)
        .unwrap();
    builder.set_shape_collision_groups(sa, 1, 1 << 31).unwrap();
    builder.set_shape_collision_groups(sb, 1 << 31, 1).unwrap();
    builder.set_shape_material(sa, 0.2, 0.0).unwrap();
    builder.set_shape_material(sb, 0.8, 0.0).unwrap();
    builder
        .set_shape_contact_material(
            sa,
            ContactMaterial {
                friction_combine: Rule::Min,
                ..ContactMaterial::default()
            },
        )
        .unwrap();
    builder
        .set_shape_contact_material(
            sb,
            ContactMaterial {
                friction_combine: Rule::Average,
                ..ContactMaterial::default()
            },
        )
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut pipeline = CollisionPipeline::new();
    pipeline.enable_full_gpu_narrow_phase(&engine, 64).unwrap();
    let mut contacts = Contacts::with_capacity(0);
    pipeline
        .collide(&model, &model.state().unwrap(), &mut contacts)
        .unwrap();
    assert_eq!(contacts.count, 1);
    assert_eq!(contacts.friction.host().unwrap()[0], 0.2);
    assert_eq!(contacts.a_shape.host().unwrap()[0], sa.0);
    assert_eq!(contacts.b_shape.host().unwrap()[0], sb.0);
    assert_eq!(pipeline.last_cpu_narrow_phase_calls(), 0);
    pipeline
        .collide_filtered(&model, &model.state().unwrap(), &mut contacts, |_, _| true)
        .unwrap();
    assert_eq!(contacts.count, 0);
}
