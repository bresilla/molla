use molla_geometry::{
    CollisionPipeline,
    collider_query::{ColliderShapeView, collider_local_aabb, collider_shape_view},
    decomposition::{DecompositionOptions, convex_decomposition},
    raycast_cpu::{CpuRaycaster, cast_ray},
};
use molla_math::{Transform, Vec3};
use molla_sim::JointType;
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, RigidScene};
use molla_sim::runtime::{BodyKind, JointDesc};
use molla_sim::{Contacts, TriMesh};

fn l_mesh() -> TriMesh {
    let polygon = [
        (0.0, 0.0),
        (3.0, 0.0),
        (3.0, 1.0),
        (1.0, 1.0),
        (1.0, 3.0),
        (0.0, 3.0),
    ];
    let vertices = [-0.5, 0.5]
        .into_iter()
        .flat_map(|y| polygon.iter().map(move |&(x, z)| Vec3::new(x, y, z)))
        .collect();
    let mut indices = Vec::new();
    for [a, b, c] in [[0, 1, 3], [1, 2, 3], [0, 3, 5], [3, 4, 5]] {
        indices.push([a, b, c]);
        indices.push([a + 6, c + 6, b + 6]);
    }
    for a in 0..6 {
        let b = (a + 1) % 6;
        indices.extend([[a, a + 6, b + 6], [a, b + 6, b]]);
    }
    TriMesh { vertices, indices }
}

fn cube(center: Vec3) -> Vec<Vec3> {
    let mut points = Vec::new();
    for x in [-0.5, 0.5] {
        for y in [-0.5, 0.5] {
            for z in [-0.5, 0.5] {
                points.push(center + Vec3::new(x, y, z));
            }
        }
    }
    points
}

#[test]
fn indexed_concave_mesh_keeps_cavity_and_replays_deterministically() {
    let mesh = l_mesh();
    let options = DecompositionOptions {
        resolution: 24,
        max_parts: 16,
        ..Default::default()
    };
    let shape = convex_decomposition(&mesh.vertices, &mesh.indices, options).unwrap();
    let ColliderGeometry::ConvexParts(parts) = &shape else {
        panic!("expected parts");
    };
    assert!(
        parts.len() >= 2 && parts.len() <= 16,
        "parts: {}",
        parts.len()
    );
    let mut unused = mesh.vertices.clone();
    unused.push(Vec3::splat(1e200));
    let replay = convex_decomposition(&unused, &mesh.indices, options).unwrap();
    let ColliderGeometry::ConvexParts(replay) = replay else {
        panic!("expected parts");
    };
    assert_eq!(*parts, replay);
    let mut scene = RigidScene::new().unwrap();
    let id = scene.insert_collider(ColliderDesc::new(shape)).unwrap();
    assert!(scene.collider_index(id).is_none());
    assert!(scene.collider_indices(id).unwrap().len() >= 2);
    for &piece in scene.collider_indices(id).unwrap() {
        assert_eq!(scene.collider_handle(piece), Some(id));
    }
    assert!(
        cast_ray(&scene, Vec3::new(1.75, 2.0, 1.75), -Vec3::Y, 5.0)
            .unwrap()
            .is_none()
    );
    let arm = cast_ray(&scene, Vec3::new(0.5, 2.0, 2.0), -Vec3::Y, 5.0)
        .unwrap()
        .unwrap();
    assert_eq!(arm.collider, id);
    assert!((arm.distance - 1.5).abs() < 1e-8, "{arm:?}");
    let bounds = collider_local_aabb(&scene, id).unwrap();
    assert!(bounds.min.distance(Vec3::new(0.0, -0.5, 0.0)) < 1e-8);
    assert!(bounds.max.distance(Vec3::new(3.0, 0.5, 3.0)) < 1e-8);
    assert_eq!(
        collider_shape_view(&scene, id).unwrap(),
        ColliderShapeView::Other
    );
    let mut body = BodyDesc::default();
    body.params.pose = Transform::from_translation(Vec3::new(1.75, 0.0, 1.75));
    let body = scene.insert_body(body).unwrap();
    let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.2 });
    ball.parent = Some(body);
    scene.insert_collider(ball).unwrap();
    let mut contacts = Contacts::with_capacity(0);
    CollisionPipeline::new()
        .collide(scene.model(), scene.state(), &mut contacts)
        .unwrap();
    assert_eq!(contacts.count, 0, "decomposition filled concavity");
}

#[test]
fn compound_removal_replacement_and_materials_preserve_stable_identity() {
    let mut scene = RigidScene::new().unwrap();
    let first = scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Ball { radius: 1.0 }))
        .unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut desc = ColliderDesc::new(ColliderGeometry::ConvexParts(vec![
        cube(-Vec3::X),
        cube(Vec3::X),
    ]));
    desc.parent = Some(body);
    desc.friction = 0.17;
    desc.memberships = 1 << 31;
    desc.filter = 3;
    desc.user_tag = 72;
    let compound = scene.insert_collider(desc.clone()).unwrap();
    let stamp = scene.stamp();
    assert_eq!(scene.collider_indices(compound).unwrap().len(), 2);
    scene.remove_collider(first).unwrap();
    assert_ne!(scene.stamp(), stamp);
    for (i, &piece) in scene.collider_indices(compound).unwrap().iter().enumerate() {
        assert_eq!(piece.index(), i);
        assert_eq!(scene.collider_handle(piece), Some(compound));
        assert_eq!(scene.model().shape_friction.host().unwrap()[i], 0.17);
        assert_eq!(
            scene.model().shape_collision_group.host().unwrap()[i],
            1 << 31
        );
        assert_eq!(scene.model().shape_collision_mask.host().unwrap()[i], 3);
    }
    assert_eq!(scene.collider(compound).unwrap().user_tag, 72);
    let stamp = scene.stamp();
    for parts in [
        vec![],
        vec![vec![Vec3::ZERO; 4]],
        vec![vec![Vec3::ZERO, Vec3::X, Vec3::Z, Vec3::X + Vec3::Z]],
        vec![vec![Vec3::NAN; 4]],
    ] {
        let mut invalid = desc.clone();
        invalid.geometry = ColliderGeometry::ConvexParts(parts);
        assert!(scene.replace_collider(compound, invalid).is_err());
        assert_eq!(scene.stamp(), stamp);
    }
    desc.geometry = ColliderGeometry::Ball { radius: 0.7 };
    scene.replace_collider(compound, desc).unwrap();
    assert_eq!(scene.collider_index(compound).unwrap().index(), 0);
    assert_eq!(scene.model().shape_count, 1);
    scene.remove_body(body).unwrap();
    assert!(scene.collider_indices(compound).is_none());
    assert!(scene.collider_handle(molla_core::ShapeId(0)).is_none());
}

#[test]
fn cached_queries_reuse_geometry_and_invalidate_across_edits_and_distinct_scenes() {
    let mut cache = CpuRaycaster::default();
    let mesh = l_mesh();
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut desc = ColliderDesc::new(ColliderGeometry::TriMesh(mesh));
    desc.parent = Some(body);
    let id = scene.insert_collider(desc).unwrap();
    let origin = Vec3::new(0.5, 2.0, 2.0);
    assert!(
        cache
            .cast_ray_filtered(&scene, origin, -Vec3::Y, 4.0, |_| false)
            .unwrap()
            .is_none()
    );
    assert_eq!(cache.preparation_count(), 0);
    for _ in 0..10 {
        assert!(
            cache
                .cast_ray(&scene, origin, -Vec3::Y, 4.0)
                .unwrap()
                .is_some()
        );
    }
    assert_eq!(cache.preparation_count(), 1);
    let stamp = scene.stamp();
    scene.state_mut().body_q.host_mut().unwrap()[0].position.y = -1.0;
    assert_eq!(scene.stamp(), stamp);
    assert!(
        (cache
            .cast_ray(&scene, origin, -Vec3::Y, 4.0)
            .unwrap()
            .unwrap()
            .distance
            - 2.5)
            .abs()
            < 1e-8
    );
    assert_eq!(cache.preparation_count(), 1);
    scene
        .replace_collider(
            id,
            ColliderDesc::new(ColliderGeometry::ConvexParts(vec![
                cube(Vec3::ZERO),
                cube(Vec3::Z * 2.0),
            ])),
        )
        .unwrap();
    assert!(
        cache
            .cast_ray(&scene, origin, -Vec3::Y, 4.0)
            .unwrap()
            .is_some()
    );
    assert_eq!(cache.preparation_count(), 2);
    let mut other = RigidScene::new().unwrap();
    other.insert_body(BodyDesc::default()).unwrap();
    let other_id = other
        .insert_collider(ColliderDesc::new(ColliderGeometry::Ball { radius: 0.1 }))
        .unwrap();
    other
        .set_collider_pose(other_id, Transform::IDENTITY)
        .unwrap();
    assert_eq!(other.revision(), scene.revision());
    assert_eq!(other_id, id);
    assert!(
        cache
            .cast_ray(&other, origin, -Vec3::Y, 4.0)
            .unwrap()
            .is_none()
    );
    assert_eq!(cache.preparation_count(), 2);
    assert!(
        cache
            .cast_ray(&other, Vec3::Y, -Vec3::Y, 4.0)
            .unwrap()
            .is_some()
    );
    assert_eq!(cache.preparation_count(), 3);
    other.remove_collider(other_id).unwrap();
    assert!(
        cache
            .cast_ray(&other, origin, -Vec3::Y, 4.0)
            .unwrap()
            .is_none()
    );
    assert_eq!(cache.preparation_count(), 3);
}

#[test]
fn invalid_indexed_meshes_and_authoring_limits_are_rejected() {
    let mesh = l_mesh();
    for indices in [
        vec![],
        vec![[0, 1, 99]; 4],
        vec![[0, 0, 1]; 4],
        vec![[0, 1, 2]; 4],
    ] {
        assert!(convex_decomposition(&mesh.vertices, &indices, Default::default()).is_err());
    }
    for options in [
        DecompositionOptions {
            resolution: 0,
            ..Default::default()
        },
        DecompositionOptions {
            max_parts: 0,
            ..Default::default()
        },
        DecompositionOptions {
            concavity: f64::NAN,
            ..Default::default()
        },
    ] {
        assert!(convex_decomposition(&mesh.vertices, &mesh.indices, options).is_err());
    }
}

#[test]
fn translated_small_mesh_decomposition_retains_units_and_authoring_frame() {
    let mut mesh = l_mesh();
    let shift = Vec3::new(1000.0, 100.0, -700.0);
    for point in &mut mesh.vertices {
        *point = *point * 0.01 + shift;
    }
    let geometry = convex_decomposition(
        &mesh.vertices,
        &mesh.indices,
        DecompositionOptions {
            resolution: 24,
            ..Default::default()
        },
    )
    .unwrap();
    let mut scene = RigidScene::new().unwrap();
    scene.insert_collider(ColliderDesc::new(geometry)).unwrap();
    assert!(
        cast_ray(
            &scene,
            shift + Vec3::new(0.0175, 0.02, 0.0175),
            -Vec3::Y,
            0.05
        )
        .unwrap()
        .is_none()
    );
    let hit = cast_ray(&scene, shift + Vec3::new(0.005, 0.02, 0.02), -Vec3::Y, 0.05)
        .unwrap()
        .unwrap();
    assert!((hit.distance - 0.015).abs() < 1e-8, "{hit:?}");
}

#[test]
fn mesh_preparation_survives_pose_body_and_joint_only_edits() {
    let mut scene = RigidScene::new().unwrap();
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut desc = ColliderDesc::new(ColliderGeometry::TriMesh(l_mesh()));
    desc.parent = Some(body);
    let id = scene.insert_collider(desc).unwrap();
    let geometry_stamp = scene.collider_stamp(id).unwrap();
    let mut cache = CpuRaycaster::default();
    let query = |cache: &mut CpuRaycaster, scene: &RigidScene| {
        assert!(
            cache
                .cast_ray(scene, Vec3::new(0.5, 2.0, 2.0), -Vec3::Y, 4.0)
                .unwrap()
                .is_some()
        );
        assert_eq!(cache.preparation_count(), 1);
        assert_eq!(scene.collider_stamp(id).unwrap(), geometry_stamp);
    };
    query(&mut cache, &scene);
    let child = scene.insert_body(BodyDesc::default()).unwrap();
    query(&mut cache, &scene);
    let joint = scene
        .insert_joint(JointDesc {
            parent: Some(body),
            child,
            kind: JointType::Fixed,
            frame_parent: Transform::IDENTITY,
            frame_child: Transform::IDENTITY,
            axis: Vec3::Y,
        })
        .unwrap();
    query(&mut cache, &scene);
    scene.remove_joint(joint).unwrap();
    scene.set_body_kind(body, BodyKind::Fixed).unwrap();
    query(&mut cache, &scene);
    scene
        .set_collider_pose(id, Transform::from_translation(-Vec3::Y * 0.2))
        .unwrap();
    query(&mut cache, &scene);
    scene.remove_body(child).unwrap();
    query(&mut cache, &scene);
    scene.remove_body(body).unwrap();
    assert!(
        cache
            .cast_ray(&scene, Vec3::new(0.5, 2.0, 2.0), -Vec3::Y, 4.0)
            .unwrap()
            .is_none()
    );
    assert!(scene.collider_stamp(id).is_none());
}
