use molla_core::BodyId;
use molla_geometry::collider_query::{
    ColliderShapeView, collider_aabb_at_pose, collider_local_aabb, collider_shape_view,
    collider_world_aabb,
};
use molla_geometry::shape_world::{shape_world_aabb, shape_world_transform};
use molla_geometry::support::support;
use molla_geometry::{CollisionPipeline, NarrowPhaseBackend, ShapePair, pair_contact};
use molla_math::{Quat, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, RigidScene};
use molla_sim::{BodyParams, Contacts, Heightfield, ModelBuilder, TriMesh};

fn close(a: Vec3, b: Vec3) {
    assert!(a.distance(b) < 1e-11, "{a:?} != {b:?}");
}

#[test]
fn endpoint_capsule_preserves_collider_frame_and_support_under_pose_changes() {
    for (a, b) in [
        (Vec3::new(1.0, 2.0, -1.0), Vec3::new(3.0, 0.0, 2.0)),
        (Vec3::Y, -Vec3::Y),
        (Vec3::splat(2.0), Vec3::splat(2.0)),
    ] {
        let mut scene = RigidScene::new().unwrap();
        let mut body = BodyDesc::default();
        body.params.pose = Transform::new(Vec3::new(12.0, 3.0, -2.0), Quat::from_rotation_z(0.6));
        body.params.com = Vec3::new(9.0, 7.0, 6.0);
        let parent = scene.insert_body(body).unwrap();
        let mut desc = ColliderDesc::new(ColliderGeometry::CapsuleEndpoints { a, b, radius: 0.3 });
        desc.parent = Some(parent);
        desc.pose = Transform::new(Vec3::new(0.5, 1.0, 2.0), Quat::from_rotation_x(0.7));
        let id = scene.insert_collider(desc.clone()).unwrap();
        assert_eq!(scene.collider(id).unwrap().pose, desc.pose);
        assert_eq!(
            collider_shape_view(&scene, id).unwrap(),
            ColliderShapeView::Capsule { a, b, radius: 0.3 }
        );
        let local = collider_local_aabb(&scene, id).unwrap();
        close(local.min, a.min(b) - Vec3::splat(0.3));
        close(local.max, a.max(b) + Vec3::splat(0.3));
        let poses = [
            scene.collider_world_pose(id).unwrap(),
            Transform::new(Vec3::new(-8.0, 2.0, 9.0), Quat::from_rotation_y(0.9)),
        ];
        for pose in poses {
            scene.set_collider_world_pose(id, pose).unwrap();
            let i = scene.collider_index(id).unwrap().index();
            let xf = shape_world_transform(scene.model(), scene.state(), i);
            let wa = pose.transform_point(a);
            let wb = pose.transform_point(b);
            let bounds = collider_world_aabb(&scene, id).unwrap();
            close(bounds.min, wa.min(wb) - Vec3::splat(0.3));
            close(bounds.max, wa.max(wb) + Vec3::splat(0.3));
            let broad = shape_world_aabb(scene.model(), scene.state(), i, 0.0);
            close(bounds.min, broad.min);
            close(bounds.max, broad.max);
            for direction in [Vec3::X, Vec3::Y, Vec3::Z, Vec3::new(1.0, 2.0, -3.0)] {
                let p = support(scene.model().shape_geometry[i], &xf, direction);
                let expected = wa.dot(direction).max(wb.dot(direction)) + 0.3 * direction.length();
                assert!((p.dot(direction) - expected).abs() < 1e-10);
            }
        }
    }
}

#[test]
fn round_cylinder_support_and_rotated_bounds_include_border_exactly() {
    let mut scene = RigidScene::new().unwrap();
    let id = scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::RoundCylinder {
            half_height: 0.4,
            radius: 0.8,
            border_radius: 0.15,
        }))
        .unwrap();
    let local = collider_local_aabb(&scene, id).unwrap();
    close(local.max, Vec3::new(0.95, 0.55, 0.95));
    close(local.min, -local.max);
    let index = scene.collider_index(id).unwrap().index();
    assert_eq!(scene.model().shape_geom_kind.host().unwrap()[index], 0xff);
    let geom = scene.model().shape_geometry[index];
    for pose in [
        Transform::IDENTITY,
        Transform::new(
            Vec3::new(3.0, -4.0, 7.0),
            Quat::from_rotation_x(0.3) * Quat::from_rotation_y(0.7) * Quat::from_rotation_z(0.9),
        ),
    ] {
        let bounds = collider_aabb_at_pose(&scene, id, pose).unwrap();
        for (i, direction) in [Vec3::X, Vec3::Y, Vec3::Z].into_iter().enumerate() {
            assert!((bounds.max[i] - support(geom, &pose, direction)[i]).abs() < 1e-12);
            assert!((bounds.min[i] - support(geom, &pose, -direction)[i]).abs() < 1e-12);
        }
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    let d = Vec3::new(x as f64, y as f64, z as f64);
                    if d == Vec3::ZERO {
                        continue;
                    }
                    let p = support(geom, &pose, pose.rotation * d);
                    let expected = pose.transform_point(
                        Vec3::new(d.x, 0.0, d.z).normalize_or_zero() * 0.8
                            + Vec3::Y * if d.y >= 0.0 { 0.4 } else { -0.4 }
                            + d.normalize() * 0.15,
                    );
                    assert!(
                        (p.dot(pose.rotation * d) - expected.dot(pose.rotation * d)).abs() < 1e-11
                    );
                }
            }
        }
    }
    assert_eq!(scene.collider_world_pose(id).unwrap(), Transform::IDENTITY);
}

#[test]
fn invalid_dimensions_and_query_handles_leave_scene_unchanged() {
    let mut scene = RigidScene::new().unwrap();
    let id = scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Ball { radius: 1.0 }))
        .unwrap();
    let revision = scene.revision();
    for geometry in [
        ColliderGeometry::CapsuleEndpoints {
            a: Vec3::NAN,
            b: Vec3::ZERO,
            radius: 1.0,
        },
        ColliderGeometry::CapsuleEndpoints {
            a: Vec3::ZERO,
            b: Vec3::ZERO,
            radius: -1.0,
        },
        ColliderGeometry::RoundCylinder {
            half_height: 1.0,
            radius: 1.0,
            border_radius: -0.1,
        },
        ColliderGeometry::RoundCylinder {
            half_height: 1.0,
            radius: 1.0,
            border_radius: f64::NAN,
        },
    ] {
        assert!(
            scene
                .replace_collider(id, ColliderDesc::new(geometry))
                .is_err()
        );
        assert_eq!(scene.revision(), revision);
    }
    let bad = Transform::new(Vec3::ZERO, Quat::from_xyzw(0.0, 0.0, 0.0, 0.0));
    assert!(scene.set_collider_pose(id, bad).is_err());
    assert!(scene.set_collider_world_pose(id, bad).is_err());
    assert!(collider_aabb_at_pose(&scene, id, bad).is_err());
    assert_eq!(scene.revision(), revision);
    scene.remove_collider(id).unwrap();
    assert!(collider_local_aabb(&scene, id).is_err());
    assert!(collider_shape_view(&scene, id).is_err());
    assert!(scene.collider_world_pose(id).is_err());
}

#[test]
fn convex_shape_view_contains_true_edges_not_face_diagonals() {
    let mut points = Vec::new();
    for x in [-1.0, 1.0] {
        for y in [-1.0, 1.0] {
            for z in [-1.0, 1.0] {
                points.push(Vec3::new(x, y, z));
            }
        }
    }
    points.push(Vec3::ZERO);
    let mut scene = RigidScene::new().unwrap();
    let id = scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::ConvexHull(points)))
        .unwrap();
    let ColliderShapeView::ConvexPolyhedron { points, edges } =
        collider_shape_view(&scene, id).unwrap()
    else {
        panic!("missing hull view");
    };
    assert_eq!(points.len(), 8);
    assert_eq!(edges.len(), 12);
    assert!(edges.windows(2).all(|w| w[0] < w[1]));
    for [a, b] in edges {
        assert!((points[a as usize].distance(points[b as usize]) - 2.0).abs() < 1e-12);
    }
}

#[test]
fn heightfield_bounds_include_vertical_origin_and_mesh_bounds_use_vertices() {
    let mut scene = RigidScene::new().unwrap();
    let field = Heightfield {
        nx: 3,
        nz: 2,
        cell_size: 2.0,
        origin: Vec3::new(-2.0, 7.0, -1.0),
        heights: vec![1.0, -2.0, 0.0, 2.0, 1.0, 0.0],
    };
    let id = scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Heightfield(field)))
        .unwrap();
    let bounds = collider_local_aabb(&scene, id).unwrap();
    close(bounds.min, Vec3::new(-2.0, 5.0, -1.0));
    close(bounds.max, Vec3::new(2.0, 9.0, 1.0));
    assert_eq!(
        collider_shape_view(&scene, id).unwrap(),
        ColliderShapeView::Other
    );
    let pose = Transform::new(Vec3::new(4.0, 0.0, -5.0), Quat::from_rotation_y(0.4));
    scene.set_collider_pose(id, pose).unwrap();
    let world = collider_world_aabb(&scene, id).unwrap();
    for x in [bounds.min.x, bounds.max.x] {
        for y in [bounds.min.y, bounds.max.y] {
            for z in [bounds.min.z, bounds.max.z] {
                let p = pose.transform_point(Vec3::new(x, y, z));
                assert!(p.cmpge(world.min - Vec3::splat(1e-12)).all());
                assert!(p.cmple(world.max + Vec3::splat(1e-12)).all());
            }
        }
    }
    let vertices = vec![
        Vec3::new(-1.0, 0.0, 0.0),
        Vec3::new(2.0, 1.0, 0.0),
        Vec3::new(0.0, 0.0, 3.0),
    ];
    scene
        .replace_collider(
            id,
            ColliderDesc::new(ColliderGeometry::TriMesh(TriMesh {
                vertices: vertices.clone(),
                indices: vec![[0, 1, 2]],
            })),
        )
        .unwrap();
    let bounds = collider_aabb_at_pose(&scene, id, pose).unwrap();
    close(
        bounds.min,
        vertices
            .iter()
            .map(|&p| pose.transform_point(p))
            .fold(Vec3::splat(f64::INFINITY), Vec3::min),
    );
    close(
        bounds.max,
        vertices
            .iter()
            .map(|&p| pose.transform_point(p))
            .fold(Vec3::splat(f64::NEG_INFINITY), Vec3::max),
    );
}

#[test]
fn collapsed_capsule_collides_as_a_sphere() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut desc = ColliderDesc::new(ColliderGeometry::CapsuleEndpoints {
        a: Vec3::X,
        b: Vec3::X,
        radius: 0.5,
    });
    desc.parent = Some(body);
    scene.insert_collider(desc).unwrap();
    let mut other = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    other.pose = Transform::from_translation(Vec3::X * 1.8);
    scene.insert_collider(other).unwrap();
    let mut contacts = Contacts::with_capacity(0);
    CollisionPipeline::new()
        .collide(scene.model(), scene.state(), &mut contacts)
        .unwrap();
    assert_eq!(contacts.count, 1);
    assert!((contacts.depth.host().unwrap()[0] - 0.2).abs() < 1e-12);
    close(contacts.normal.host().unwrap()[0], Vec3::X);
}

#[test]
fn rounded_cylinder_plane_contacts_include_border_in_both_pair_orders() {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new().with_position(Vec3::Y * 0.55));
    let cylinder = b
        .add_shape_round_cylinder(body, 0.5, 0.8, 0.1, Transform::IDENTITY)
        .unwrap();
    let plane = b
        .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
        .unwrap();
    let model = b.build().unwrap();
    let state = model.state().unwrap();
    for pair in [
        ShapePair {
            a: cylinder.0,
            b: plane.0,
        },
        ShapePair {
            a: plane.0,
            b: cylinder.0,
        },
    ] {
        let mut contacts = Vec::new();
        pair_contact(&model, &state, pair, 0.5, 0.0, &mut contacts);
        assert_eq!(contacts.len(), 5);
        for contact in &contacts {
            assert!((contact.depth - 0.05).abs() < 1e-12);
            close(
                contact.normal,
                if pair.a == cylinder.0 {
                    -Vec3::Y
                } else {
                    Vec3::Y
                },
            );
            assert_eq!(contact.a_shape, pair.a);
            assert_eq!(contact.b_shape, pair.b);
        }
        close(
            contacts.iter().map(|c| c.point).sum::<Vec3>() / 5.0,
            -Vec3::Y * 0.025,
        );
    }
}

fn rounded_sphere_model() -> molla_sim::Model {
    let mut b = ModelBuilder::new();
    let body = b.add_body(BodyParams::new());
    b.add_shape_round_cylinder(body, 0.5, 0.8, 0.1, Transform::IDENTITY)
        .unwrap();
    b.add_shape_sphere(
        BodyId::NONE,
        0.1,
        Transform::from_translation(Vec3::new(0.9, 0.6, 0.0)),
    )
    .unwrap();
    b.build().unwrap()
}

#[test]
fn rounded_corner_contact_uses_gjk_fallback_with_either_backend_selection() {
    let model = rounded_sphere_model();
    for backend in [NarrowPhaseBackend::Gjk, NarrowPhaseBackend::MprCpu] {
        let mut contacts = Contacts::with_capacity(0);
        let mut pipeline = CollisionPipeline::new().with_narrow_phase_backend(backend);
        pipeline
            .collide(&model, &model.state().unwrap(), &mut contacts)
            .unwrap();
        assert_eq!(contacts.count, 1, "{backend:?}");
        let expected = 0.2 - 0.1_f64.hypot(0.1);
        assert!(
            (contacts.depth.host().unwrap()[0] - expected).abs() < 1e-5,
            "{backend:?}: {:?}",
            contacts.depth.host().unwrap()
        );
        assert!(
            contacts.normal.host().unwrap()[0]
                .normalize()
                .dot(Vec3::new(1.0, 1.0, 0.0).normalize())
                > 0.99999
        );
    }
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "requires a GPU adapter"]
fn gpu_pipeline_keeps_rounded_cylinder_on_cpu_without_dropping_contact() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let mut model = rounded_sphere_model();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut pipeline = CollisionPipeline::new();
    pipeline.enable_full_gpu_narrow_phase(&engine, 16).unwrap();
    let mut contacts = Contacts::with_capacity(0);
    pipeline
        .collide(&model, &model.state().unwrap(), &mut contacts)
        .unwrap();
    assert_eq!(contacts.count, 1);
    assert_eq!(pipeline.last_cpu_narrow_phase_calls(), 1);
    assert_eq!(contacts.a_shape.host().unwrap()[0], 0);
    assert_eq!(contacts.b_shape.host().unwrap()[0], 1);
}
