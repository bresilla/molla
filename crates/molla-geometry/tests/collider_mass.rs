use molla_geometry::mass_properties::{
    body_mass_properties, collider_mass_properties, recompute_body_mass,
};
use molla_math::{Mat3, Quat, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, RigidScene};
use molla_sim::{ColliderMass, Heightfield, MassProperties, TriMesh};

fn near(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9 * b.abs().max(1.0), "{a} != {b}");
}

fn matrix_near(a: Mat3, b: Mat3) {
    for (a, b) in a.to_cols_array().into_iter().zip(b.to_cols_array()) {
        near(a, b);
    }
}

fn cube(center: Vec3, half: f64) -> Vec<Vec3> {
    let mut points = Vec::new();
    for x in [-half, half] {
        for y in [-half, half] {
            for z in [-half, half] {
                points.push(center + Vec3::new(x, y, z));
            }
        }
    }
    points
}

#[test]
fn analytic_primitives_density_and_total_mass() {
    let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 2.0 });
    ball.mass = ColliderMass::Density(3.0);
    let p = collider_mass_properties(&ball).unwrap();
    near(p.mass, 32.0 * std::f64::consts::PI);
    matrix_near(p.inertia, Mat3::from_diagonal(Vec3::splat(p.mass * 1.6)));
    let mut cuboid = ColliderDesc::new(ColliderGeometry::Cuboid {
        half_extents: Vec3::new(1.0, 2.0, 3.0),
    });
    cuboid.mass = ColliderMass::Mass(6.0);
    let p = collider_mass_properties(&cuboid).unwrap();
    near(p.mass, 6.0);
    matrix_near(p.inertia, Mat3::from_diagonal(Vec3::new(26.0, 20.0, 10.0)));
    cuboid.mass = ColliderMass::Mass(0.0);
    assert_eq!(
        collider_mass_properties(&cuboid).unwrap(),
        MassProperties::ZERO
    );
}

#[test]
fn capsule_endpoint_frame_and_degenerate_segment() {
    let center = Vec3::new(3.0, 4.0, 5.0);
    let endpoint = ColliderDesc::new(ColliderGeometry::CapsuleEndpoints {
        a: center - Vec3::X,
        b: center + Vec3::X,
        radius: 0.5,
    });
    let canonical = ColliderDesc::new(ColliderGeometry::Capsule {
        half_height: 1.0,
        radius: 0.5,
    });
    let p = collider_mass_properties(&endpoint).unwrap();
    let expected = collider_mass_properties(&canonical)
        .unwrap()
        .transformed(endpoint.geometry.geometry_transform().unwrap())
        .unwrap();
    near((p.local_com - center).length(), 0.0);
    near(p.mass, expected.mass);
    matrix_near(p.inertia, expected.inertia);
    assert!(p.inertia.x_axis.x < p.inertia.y_axis.y);
    let point = ColliderDesc::new(ColliderGeometry::CapsuleEndpoints {
        a: center,
        b: center,
        radius: 0.5,
    });
    let sphere =
        collider_mass_properties(&ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 }))
            .unwrap();
    let p = collider_mass_properties(&point).unwrap();
    near(p.mass, sphere.mass);
    matrix_near(p.inertia, sphere.inertia);
    near((p.local_com - center).length(), 0.0);
}

#[test]
fn rounded_cylinder_matches_reference_inner_cylinder_mass() {
    let cylinder = ColliderDesc::new(ColliderGeometry::Cylinder {
        half_height: 2.0,
        radius: 0.5,
    });
    let rounded = ColliderDesc::new(ColliderGeometry::RoundCylinder {
        half_height: 2.0,
        radius: 0.5,
        border_radius: 0.4,
    });
    let p = collider_mass_properties(&cylinder).unwrap();
    near(p.mass, std::f64::consts::PI);
    matrix_near(
        p.inertia,
        Mat3::from_diagonal(Vec3::new(16.75 / 12.0, 0.125, 16.75 / 12.0) * p.mass),
    );
    assert_eq!(p, collider_mass_properties(&rounded).unwrap());
}

#[test]
fn hulls_compound_properties_and_single_mass_assignment() {
    let mut shape = ColliderDesc::new(ColliderGeometry::ConvexParts(vec![
        cube(-Vec3::X * 2.0, 0.5),
        cube(Vec3::X * 2.0, 0.5),
    ]));
    let p = collider_mass_properties(&shape).unwrap();
    near(p.mass, 2.0);
    near(p.local_com.length(), 0.0);
    matrix_near(
        p.inertia,
        Mat3::from_diagonal(Vec3::new(1.0 / 3.0, 25.0 / 3.0, 25.0 / 3.0)),
    );
    shape.mass = ColliderMass::Mass(10.0);
    let explicit = collider_mass_properties(&shape).unwrap();
    near(explicit.mass, 10.0);
    matrix_near(explicit.inertia, p.inertia * 5.0);
    let center = Vec3::new(1234.0, 5678.0, -3456.0);
    let small = ColliderDesc::new(ColliderGeometry::ConvexHull(cube(center, 0.01)));
    let p = collider_mass_properties(&small).unwrap();
    near((p.local_com - center).length(), 0.0);
    assert!((p.mass / 8e-6 - 1.0).abs() < 1e-9);
    for value in [p.inertia.x_axis.x, p.inertia.y_axis.y, p.inertia.z_axis.z] {
        assert!((value / (8e-6 * 2e-4 / 3.0) - 1.0).abs() < 1e-9);
    }
}

#[test]
fn indexed_mesh_ignores_unused_points_and_matches_hull() {
    let points = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
    let triangles = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let expected = collider_mass_properties(&ColliderDesc::new(ColliderGeometry::ConvexHull(
        points.clone(),
    )))
    .unwrap();
    let mut points = points;
    points.push(Vec3::splat(1e200));
    let p = collider_mass_properties(&ColliderDesc::new(ColliderGeometry::TriMesh(TriMesh {
        vertices: points,
        indices: triangles,
    })))
    .unwrap();
    near(p.mass, 1.0 / 6.0);
    near((p.local_com - Vec3::splat(0.25)).length(), 0.0);
    matrix_near(p.inertia, expected.inertia);
}

#[test]
fn recompute_filters_disabled_includes_sensors_and_replaces_additional() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let geometry = ColliderGeometry::Cuboid {
        half_extents: Vec3::new(0.5, 1.0, 1.5),
    };
    let mut sensor = ColliderDesc::new(geometry.clone());
    sensor.parent = Some(body);
    sensor.mass = ColliderMass::Mass(2.0);
    sensor.sensor = true;
    sensor.pose = Transform::new(Vec3::X, Quat::from_rotation_z(0.6));
    let sensor_props = collider_mass_properties(&sensor)
        .unwrap()
        .transformed(sensor.pose)
        .unwrap();
    scene.insert_collider(sensor.clone()).unwrap();
    let mut disabled = sensor.clone();
    disabled.enabled = false;
    disabled.mass = ColliderMass::Mass(1000.0);
    let disabled_id = scene.insert_collider(disabled).unwrap();
    scene.insert_collider(ColliderDesc::new(geometry)).unwrap();
    let additional = MassProperties {
        mass: 3.0,
        local_com: -Vec3::X,
        inertia: Mat3::IDENTITY,
    };
    let expected = sensor_props.combined(additional).unwrap();
    for _ in 0..2 {
        recompute_body_mass(&mut scene, body, additional).unwrap();
        let p = scene.body_mass_properties(body).unwrap();
        near(p.mass, 5.0);
        matrix_near(p.inertia, expected.inertia);
        near((p.local_com - expected.local_com).length(), 0.0);
    }
    recompute_body_mass(&mut scene, body, MassProperties::ZERO).unwrap();
    near(scene.body_mass_properties(body).unwrap().mass, 2.0);
    let mut enabled = scene.collider(disabled_id).unwrap().clone();
    enabled.enabled = true;
    scene.replace_collider(disabled_id, enabled).unwrap();
    near(
        body_mass_properties(&scene, body, MassProperties::ZERO)
            .unwrap()
            .mass,
        1002.0,
    );
}

#[test]
fn invalid_properties_are_atomic_and_heightfields_have_no_volume() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut collider = ColliderDesc::new(ColliderGeometry::Ball { radius: 1.0 });
    collider.parent = Some(body);
    collider.mass = ColliderMass::Density(-1.0);
    let stamp = scene.stamp();
    assert!(scene.insert_collider(collider.clone()).is_err());
    assert_eq!(scene.stamp(), stamp);
    collider.mass = ColliderMass::Mass(f64::NAN);
    assert!(collider_mass_properties(&collider).is_err());
    let field = Heightfield::from_scaled(2, 2, vec![0.0; 4], Vec3::ONE).unwrap();
    let mut terrain = ColliderDesc::new(ColliderGeometry::Heightfield(field));
    assert_eq!(
        collider_mass_properties(&terrain).unwrap(),
        MassProperties::ZERO
    );
    terrain.parent = Some(body);
    terrain.mass = ColliderMass::Mass(1.0);
    let p = collider_mass_properties(&terrain).unwrap();
    assert_eq!(p.mass, 1.0);
    assert_eq!(p.inertia, Mat3::ZERO);
    scene.insert_collider(terrain).unwrap();
    let stamp = scene.stamp();
    let before = scene.body_mass_properties(body).unwrap();
    let invalid = MassProperties {
        mass: -1.0,
        ..MassProperties::ZERO
    };
    assert!(recompute_body_mass(&mut scene, body, invalid).is_err());
    assert_eq!(scene.stamp(), stamp);
    assert_eq!(scene.body_mass_properties(body).unwrap(), before);
}
