//! GPU contact sensor — golden-value tests against hand-computed
//! expectations. `#[ignore]` because they need a real wgpu adapter;
//! run with `make test-contact-gpu`.

#![cfg(feature = "gpu")]

use molla::compute::Engine;
use molla::geometry::CollisionPipeline;
use molla::math::{Transform, Vec3};
use molla::prelude::*;
use molla::sensors::ContactSensor;

fn setup(model: Model) -> Option<(Engine, Model, State, Contacts)> {
    let engine = Engine::new_gpu().ok()?;
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();
    let mut model = model;
    model.upload_to_device(&device, &queue).ok()?;
    let mut state = model.state().ok()?;
    state.upload_to_device(&device, &queue).ok()?;
    let mut contacts = Contacts::with_capacity(64);
    contacts.upload_to_device(&device, &queue).ok()?;
    Some((engine, model, state, contacts))
}

/// Sphere of radius 0.5 sits exactly on the static plane at y = 0.5.
/// One contact, normal +Y on the sphere side, depth ≈ 0.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-contact-gpu"]
fn sensor_sphere_on_plane_one_contact() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(0.0, 0.4999, 0.0)), // tiny overlap
    );
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();

    let (engine, model, state, mut contacts) = setup(model).expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    // Run narrow phase to populate the contact arrays.
    let mut pipeline = CollisionPipeline::new();
    pipeline.collide(&model, &state, &mut contacts).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut sensor = ContactSensor::new(&engine, &model).unwrap();
    let readings = sensor.update(&engine, &contacts).unwrap();
    assert_eq!(readings.len(), 1);
    let r = readings[0];
    eprintln!("sphere/plane reading: {r:?}");
    assert_eq!(r.count, 1, "expected 1 contact, got {}", r.count);
    // Penetration: 0.5 - 0.4999 = 0.0001m.
    assert!(
        r.max_depth >= 0.0 && r.max_depth < 0.01,
        "depth = {}",
        r.max_depth
    );
    // Net push direction on the sphere is +Y (plane pushes up).
    assert!(
        (r.normal_sum.normalize().y - 1.0).abs() < 1.0e-3,
        "normal_sum {:?}",
        r.normal_sum
    );
    // Contact point ≈ at y = 0 (where the sphere touches the plane).
    assert!(r.centroid.y.abs() < 0.01, "centroid {:?}", r.centroid);
}

/// Sphere floating well above the plane → zero contacts.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-contact-gpu"]
fn sensor_floating_sphere_zero_contacts() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(0.0, 5.0, 0.0)),
    );
    b.add_shape_sphere(body, 0.5, Transform::IDENTITY).unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let (engine, model, state, mut contacts) = setup(model).expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut pipeline = CollisionPipeline::new();
    pipeline.collide(&model, &state, &mut contacts).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut sensor = ContactSensor::new(&engine, &model).unwrap();
    let readings = sensor.update(&engine, &contacts).unwrap();
    let r = readings[0];
    assert_eq!(r.count, 0, "should be no contacts when floating");
    assert!(r.max_depth < 1.0e-6);
    let _ = r.normal_sum;
    let _ = r.centroid;
}

/// Two stacked spheres on a plane: lower sphere has 2 contacts (with
/// plane + with upper), upper sphere has 1 contact (with lower). Their
/// normal_sum vectors point in opposite directions.
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-contact-gpu"]
fn sensor_stack_of_two_spheres() {
    let mut b = ModelBuilder::new();
    // Lower sphere just barely overlapping the plane.
    let lower = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(0.0, 0.499, 0.0)),
    );
    b.add_shape_sphere(lower, 0.5, Transform::IDENTITY).unwrap();
    // Upper sphere overlapping the lower one by ~0.01m so the
    // narrow phase reliably finds the contact.
    let upper = b.add_body(
        BodyParams::new()
            .with_spatial_inertia(SpatialInertia::sphere(1.0, 0.5))
            .with_position(Vec3::new(0.0, 1.49, 0.0)),
    );
    b.add_shape_sphere(upper, 0.5, Transform::IDENTITY).unwrap();
    b.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let (engine, model, state, mut contacts) = setup(model).expect("GPU adapter required for this test (gated by #[ignore])");
    let device = engine.device().unwrap().clone();
    let queue = engine.queue().unwrap().clone();

    let mut pipeline = CollisionPipeline::new();
    pipeline.collide(&model, &state, &mut contacts).unwrap();
    contacts.upload_to_device(&device, &queue).unwrap();

    let mut sensor = ContactSensor::new(&engine, &model).unwrap();
    let readings = sensor.update(&engine, &contacts).unwrap();
    eprintln!("lower: {:?}", readings[0]);
    eprintln!("upper: {:?}", readings[1]);
    assert_eq!(readings[0].count, 2, "lower should have 2 contacts");
    assert_eq!(readings[1].count, 1, "upper should have 1 contact");
    // Lower sphere is pushed UP by the plane (+Y) and pushed DOWN
    // by the upper sphere (-Y) — equal and opposite, net ≈ 0. This
    // is exactly the "I'm sandwiched" case the sensor should report.
    assert!(
        readings[0].normal_sum.length() < 0.1,
        "lower net push should be near zero (sandwiched), got {:?}",
        readings[0].normal_sum
    );
    // Upper sphere is pushed up by the lower one → +Y.
    assert!(
        readings[1].normal_sum.y > 0.5,
        "upper should be pushed +Y, got {:?}",
        readings[1].normal_sum
    );
}
