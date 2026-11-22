use molla_geometry::{
    cell_contact::{cell_manifold, face_normal},
    convex_cut::{ConvexCell, CutPlane},
    gjk::{GjkResult, gjk_with_hulls},
};
use molla_math::{Quat, Transform, Vec3};
use molla_sim::ShapeGeometry;

fn pose(position: Vec3) -> Transform {
    Transform::new(position, Quat::IDENTITY)
}

fn surface(cell: &ConvexCell, pose: Transform, point: Vec3, tolerance: f64) {
    let local = pose.rotation.conjugate() * (point - pose.position);
    let mut nearest = f64::INFINITY;
    for (i, face) in cell.faces().iter().enumerate() {
        let distance =
            (local - cell.vertices()[face.vertices[0] as usize]).dot(face_normal(cell, i));
        assert!(distance < tolerance, "outside cell by {distance}");
        nearest = nearest.min(distance.abs());
    }
    assert!(nearest < tolerance, "witness not on surface: {nearest}");
}

#[test]
fn flat_caps_produce_spread_surface_witnesses_not_spherical_caps() {
    let cell = ConvexCell::prism(0.13, 0.7, 6).unwrap();
    let a = pose(Vec3::ZERO);
    let b = pose(Vec3::Z * 0.69);
    let contact = cell_manifold(&cell, a, &cell, b, 0.0).unwrap().unwrap();
    assert!(contact.normal.distance(Vec3::Z) < 1e-12);
    assert_eq!(contact.points.len(), 4);
    for p in &contact.points {
        assert!(((p.a - p.b).dot(contact.normal) - 0.01).abs() < 1e-12);
        surface(&cell, a, p.a, 1e-12);
        surface(&cell, b, p.b, 1e-12);
    }
    assert!(
        cell_manifold(&cell, a, &cell, pose(Vec3::Z * 0.7001), 0.0)
            .unwrap()
            .is_none()
    );
    let touching = cell_manifold(&cell, a, &cell, pose(Vec3::Z * 0.7), 1e-8)
        .unwrap()
        .unwrap();
    assert_eq!(touching.points.len(), 4);
}

#[test]
fn oblique_cut_caps_touch_without_overlap_or_artificial_gap() {
    let parent = ConvexCell::prism(0.13, 1.0, 6).unwrap();
    let plane = CutPlane::new(Vec3::Z * 0.163, Vec3::new(0.3, 0.2, 1.0)).unwrap();
    let [a, b] = parent.split(plane, 1).unwrap().unwrap();
    let xf = Transform::new(Vec3::new(4.0, -3.0, 2.0), Quat::from_rotation_x(0.6));
    let contact = cell_manifold(&a, xf, &b, xf, 1e-10).unwrap().unwrap();
    assert!(contact.normal.distance(xf.rotation * plane.normal()) < 1e-12);
    for p in contact.points {
        assert!(p.a.distance(p.b) < 1e-12);
        surface(&a, xf, p.a, 1e-11);
        surface(&b, xf, p.b, 1e-11);
    }
}

#[test]
fn sat_classification_matches_existing_hull_gjk_for_rotated_cut_cells() {
    let parent = ConvexCell::prism(0.2, 0.8, 6).unwrap();
    let a = parent
        .split(
            CutPlane::new(Vec3::Z * 0.111, Vec3::new(0.4, -0.2, 1.0)).unwrap(),
            1,
        )
        .unwrap()
        .unwrap()[0]
        .clone();
    let b = ConvexCell::prism(0.17, 0.65, 5).unwrap();
    let hulls = [a.vertices().to_vec(), b.vertices().to_vec()];
    let mut contacts = 0;
    let mut edges = 0;
    for i in 0..600 {
        let t = i as f64;
        let xa = Transform::new(Vec3::ZERO, Quat::from_rotation_z(t * 0.23));
        let xb = Transform::new(
            Vec3::new(
                (t * 0.37).sin() * 0.55,
                (t * 0.71).cos() * 0.4,
                (t * 0.53).sin() * 0.65,
            ),
            Quat::from_rotation_x(t * 0.33) * Quat::from_rotation_y(t * 0.19),
        );
        let contact = cell_manifold(&a, xa, &b, xb, 0.0).unwrap();
        let gjk = gjk_with_hulls(
            ShapeGeometry::ConvexHull { hull_id: 0 },
            &xa,
            ShapeGeometry::ConvexHull { hull_id: 1 },
            &xb,
            &hulls,
        );
        assert_eq!(
            contact.is_some(),
            matches!(gjk, GjkResult::Overlap(_)),
            "case {i}: {gjk:?}"
        );
        if let Some(contact) = contact {
            contacts += 1;
            edges += usize::from(contact.points.len() == 1);
            for p in contact.points {
                surface(&a, xa, p.a, 1e-6);
                surface(&b, xb, p.b, 1e-6);
                assert!((p.b - p.a).dot(contact.normal) <= 1e-8);
            }
        }
    }
    assert!(
        contacts > 50 && edges > 10,
        "contacts={contacts} edges={edges}"
    );
}

#[test]
fn containment_separates_through_a_real_surface_and_invalid_inputs_fail() {
    let outer = ConvexCell::prism(1.0, 2.0, 6).unwrap();
    let inner = ConvexCell::prism(0.1, 0.2, 6).unwrap();
    let xf = pose(Vec3::ZERO);
    let contact = cell_manifold(&outer, xf, &inner, xf, 0.0).unwrap().unwrap();
    for p in contact.points {
        surface(&outer, xf, p.a, 1e-10);
        surface(&inner, xf, p.b, 1e-10);
        assert!((p.a - p.b).dot(contact.normal) > 0.9);
    }
    assert!(cell_manifold(&outer, xf, &inner, xf, -1.0).is_err());
    assert!(cell_manifold(&outer, pose(Vec3::splat(f64::NAN)), &inner, xf, 0.0).is_err());
}
