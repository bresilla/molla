use molla_geometry::convex_cut::{ConvexCell, CutPlane, FaceOrigin, RigidCellMotion};
use molla_math::{Mat3, Quat, Transform, Vec3};

fn matrix_error(a: Mat3, b: Mat3) -> f64 {
    (a - b)
        .to_cols_array()
        .into_iter()
        .map(f64::abs)
        .fold(0.0, f64::max)
}

fn verify_split(cell: &ConvexCell, plane: CutPlane, id: u64) -> [ConvexCell; 2] {
    let before = cell.clone();
    let [negative, positive] = cell.split(plane, id).unwrap().unwrap();
    assert_eq!(*cell, before);
    let parent = cell.mass_properties(650.0).unwrap();
    let children = [
        negative.mass_properties(650.0).unwrap(),
        positive.mass_properties(650.0).unwrap(),
    ];
    let combined = children[0].combined(children[1]).unwrap();
    let extent = cell
        .vertices()
        .iter()
        .map(|p| p.length())
        .fold(0.0, f64::max);
    assert!((combined.mass - parent.mass).abs() < parent.mass * 2e-10);
    assert!(combined.local_com.distance(parent.local_com) < extent * 2e-10);
    assert!(
        matrix_error(combined.inertia, parent.inertia) < parent.mass * extent * extent * 2e-10,
        "plane={plane:?}, parent={parent:?}, combined={combined:?}, error={}",
        matrix_error(combined.inertia, parent.inertia)
    );
    let motion = RigidCellMotion {
        pose: Transform::new(
            Vec3::new(3.0, -2.0, 4.0),
            Quat::from_rotation_x(0.3) * Quat::from_rotation_y(-0.6) * Quat::from_rotation_z(0.2),
        ),
        linear_velocity: Vec3::new(1.2, -0.7, 0.3),
        angular_velocity: Vec3::new(0.4, 0.8, -0.3),
    };
    let (p, l, k) = motion.momentum(parent).unwrap();
    let mut total = (Vec3::ZERO, Vec3::ZERO, 0.0);
    for properties in children {
        let transferred = motion.restricted(parent, properties).unwrap();
        let (p, l, k) = transferred.momentum(properties).unwrap();
        total.0 += p;
        total.1 += l;
        total.2 += k;
    }
    assert!(total.0.distance(p) < p.length().max(parent.mass) * 3e-10);
    assert!(total.1.distance(l) < l.length().max(parent.mass) * 3e-10);
    assert!((total.2 - k).abs() < k * 3e-10);
    for (side, child) in [&negative, &positive].into_iter().enumerate() {
        child.validate().unwrap();
        let cap = child
            .faces()
            .iter()
            .filter(|f| {
                f.origin
                    == FaceOrigin::Cut {
                        id,
                        positive_side: side == 1,
                    }
            })
            .collect::<Vec<_>>();
        assert_eq!(cap.len(), 1);
        for &v in &cap[0].vertices {
            assert!(plane.distance(child.vertices()[v as usize]).abs() < extent * 2e-12);
        }
        for &p in child.vertices() {
            assert!(plane.distance(p) * if side == 0 { 1.0 } else { -1.0 } < extent * 2e-12);
        }
    }
    [negative, positive]
}

#[test]
fn off_node_cuts_conserve_volume_inertia_and_moving_rigid_fields() {
    for scale in [1e-4, 1.0, 1e4] {
        let cell = ConvexCell::prism(0.13 * scale, 1.7 * scale, 8).unwrap();
        let properties = cell.mass_properties(650.0).unwrap();
        let volume =
            4.0 * (std::f64::consts::PI / 4.0).sin() * (0.13 * scale).powi(2) * 1.7 * scale;
        assert!((properties.mass - 650.0 * volume).abs() < properties.mass * 1e-12);
        for fraction in [0.017, 0.137, 0.371, 0.509, 0.923, 0.999] {
            let point = Vec3::Z * ((fraction - 0.5) * 1.7 * scale);
            let children = verify_split(&cell, CutPlane::new(point, Vec3::Z).unwrap(), 17);
            assert!(
                (children[0].mass_properties(650.0).unwrap().mass / properties.mass - fraction)
                    .abs()
                    < 2e-11
            );
        }
    }
}

#[test]
fn oblique_and_longitudinal_cuts_have_actual_plane_aligned_caps() {
    let cell = ConvexCell::prism(0.18, 2.0, 12).unwrap();
    for (point, normal) in [
        (Vec3::new(0.0, 0.0, 0.157), Vec3::new(0.6, -0.3, 1.0)),
        (Vec3::new(0.037, 0.0, 0.0), Vec3::X),
        (Vec3::ZERO, Vec3::new(1.0, 0.37, 0.12)),
        (Vec3::ZERO, Vec3::Y),
    ] {
        verify_split(&cell, CutPlane::new(point, normal).unwrap(), 33);
    }
}

#[test]
fn repeated_cuts_keep_all_caps_and_do_not_mutate_the_parent() {
    let root = ConvexCell::prism(0.2, 2.0, 8).unwrap();
    let total = root.mass_properties(1.0).unwrap();
    let mut pieces = vec![root];
    for (id, (point, normal)) in [
        (Vec3::Z * 0.173, Vec3::new(0.2, 0.3, 1.0)),
        (Vec3::Z * -0.427, Vec3::new(-0.5, 0.1, 1.0)),
        (Vec3::X * 0.027, Vec3::new(1.0, 0.1, 0.0)),
    ]
    .into_iter()
    .enumerate()
    {
        let plane = CutPlane::new(point, normal).unwrap();
        let mut next = Vec::new();
        for cell in pieces {
            match cell.split(plane, id as u64).unwrap() {
                Some(children) => next.extend(children),
                None => next.push(cell),
            }
        }
        pieces = next;
    }
    assert!(pieces.len() >= 6);
    let mut combined = molla_sim::MassProperties::ZERO;
    for cell in pieces {
        cell.validate().unwrap();
        assert!(
            cell.faces()
                .iter()
                .any(|f| matches!(f.origin, FaceOrigin::Cut { .. }))
        );
        combined = combined
            .combined(cell.mass_properties(1.0).unwrap())
            .unwrap();
    }
    assert!((combined.mass - total.mass).abs() < 1e-11);
    assert!(combined.local_com.distance(total.local_com) < 1e-11);
    assert!(matrix_error(combined.inertia, total.inertia) < 1e-11);
}

#[test]
fn invalid_cells_planes_and_unresolved_slivers_are_rejected() {
    let cell = ConvexCell::prism(0.1, 1.0, 8).unwrap();
    assert!(CutPlane::new(Vec3::ZERO, Vec3::ZERO).is_err());
    assert!(CutPlane::new(Vec3::splat(f64::NAN), Vec3::Z).is_err());
    assert!(cell.mass_properties(0.0).is_err());
    assert!(
        cell.split(CutPlane::new(Vec3::Z * (0.5 - 1e-12), Vec3::Z).unwrap(), 0)
            .is_err()
    );
    for z in [0.5, 0.6, -0.5, -0.7] {
        assert!(
            cell.split(CutPlane::new(Vec3::Z * z, Vec3::Z).unwrap(), 0)
                .unwrap()
                .is_none()
        );
    }
    let mut open = cell.faces().to_vec();
    open.pop();
    assert!(ConvexCell::new(cell.vertices().to_vec(), open).is_err());
    let mut inward = cell.faces().to_vec();
    inward[0].vertices.reverse();
    assert!(ConvexCell::new(cell.vertices().to_vec(), inward).is_err());
    assert!(ConvexCell::prism(0.1, 0.0, 8).is_err());
}

#[test]
fn world_plane_conversion_preserves_the_selected_blade_location() {
    let pose = Transform::new(
        Vec3::new(2.0, 3.0, -1.0),
        Quat::from_axis_angle(Vec3::new(0.2, 0.6, 0.8).normalize(), 0.9),
    );
    let plane = CutPlane::new(Vec3::new(2.0, 3.0, -0.83), Vec3::new(0.7, 0.2, 1.0)).unwrap();
    let local = plane.to_local(pose).unwrap();
    for p in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z] {
        assert!(
            (local.distance(p) - plane.distance(pose.position + pose.rotation * p)).abs() < 1e-14
        );
    }
    let cell = ConvexCell::prism(0.2, 2.0, 8).unwrap();
    verify_split(&cell, local, 91);
}

#[test]
fn cuts_through_vertices_and_edges_are_watertight_across_scales() {
    for scale in [1e-4, 1.0, 1e4] {
        let cell = ConvexCell::prism(scale * 0.2, scale * 1.3, 12).unwrap();
        for (i, &vertex) in cell.vertices().iter().enumerate() {
            let normal = Vec3::new(0.3 + (i as f64).sin(), 0.7, 0.21);
            let plane = CutPlane::new(vertex, normal).unwrap();
            let distances: Vec<_> = cell.vertices().iter().map(|&p| plane.distance(p)).collect();
            if distances.iter().any(|d| *d < -scale * 1e-10)
                && distances.iter().any(|d| *d > scale * 1e-10)
            {
                verify_split(&cell, plane, i as u64);
            }
        }
        for i in 0..200 {
            let n = Vec3::new(
                (i as f64 * 2.1).sin(),
                (i as f64 * 0.7).cos(),
                (i as f64 * 0.31).cos(),
            );
            let plane = CutPlane::new(Vec3::new(0.03, -0.02, 0.07) * scale, n).unwrap();
            verify_split(&cell, plane, 100 + i);
        }
    }
}
