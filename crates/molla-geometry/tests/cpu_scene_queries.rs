use molla_geometry::{
    CollisionPipeline,
    collider_query::collider_local_aabb,
    raycast_cpu::{cast_ray, cast_ray_filtered},
};
use molla_math::{Quat, Transform, Vec3};
use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, RigidScene};
use molla_sim::{Contacts, Heightfield, TriMesh};

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-8, "{a} != {b}");
}
fn vector_close(a: Vec3, b: Vec3) {
    assert!(a.distance(b) < 1e-8, "{a:?} != {b:?}");
}

#[test]
fn scaled_heightfield_is_centered_row_major_and_uses_independent_spacing() {
    let hf = Heightfield::from_scaled(
        2,
        3,
        vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        Vec3::new(8.0, 2.0, 6.0),
    )
    .unwrap();
    assert_eq!(
        (hf.nx, hf.nz, hf.cell_size, hf.cell_size_z),
        (3, 2, 4.0, 6.0)
    );
    vector_close(hf.vertex(0, 0), Vec3::new(-4.0, 0.0, -3.0));
    vector_close(hf.vertex(1, 1), Vec3::new(0.0, 8.0, 3.0));
    vector_close(hf.vertex(2, 1), Vec3::new(4.0, 10.0, 3.0));
    let (height, normal) = hf.height_and_normal(0.0, 0.0).unwrap();
    close(height, 5.0);
    vector_close(normal, Vec3::new(-0.5, 1.0, -1.0).normalize());
    let mut scene = RigidScene::new().unwrap();
    let id = scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Heightfield(hf)))
        .unwrap();
    let bounds = collider_local_aabb(&scene, id).unwrap();
    vector_close(bounds.min, Vec3::new(-4.0, 0.0, -3.0));
    vector_close(bounds.max, Vec3::new(4.0, 10.0, 3.0));
    let hit = cast_ray(&scene, Vec3::Y * 20.0, -Vec3::Y * 7.0, 15.0)
        .unwrap()
        .unwrap();
    close(hit.distance, 15.0);
    vector_close(hit.normal, normal);
    assert_eq!(hit.collider, id);
    assert!(
        cast_ray(&scene, Vec3::Y * 20.0, -Vec3::Y, 14.9)
            .unwrap()
            .is_none()
    );
}

#[test]
fn heightfield_sampling_rays_and_hull_contacts_share_triangle_diagonal() {
    let hf =
        Heightfield::from_scaled(2, 2, vec![0.0, 0.0, 0.0, 1.0], Vec3::new(2.0, 2.0, 2.0)).unwrap();
    close(hf.height_and_normal(-0.5, -0.5).unwrap().0, 0.0);
    close(hf.height_and_normal(0.5, 0.5).unwrap().0, 1.0);
    close(hf.height_and_normal(0.5, -0.5).unwrap().0, 0.0);
    let mut scene = RigidScene::new().unwrap();
    let terrain = scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Heightfield(hf)))
        .unwrap();
    let hit = cast_ray(&scene, Vec3::new(0.5, 3.0, 0.5), -Vec3::Y, 10.0)
        .unwrap()
        .unwrap();
    close(hit.distance, 2.0);
    vector_close(hit.normal, Vec3::new(-1.0, 1.0, -1.0).normalize());
    let body = scene.insert_body(BodyDesc::default()).unwrap();
    let mut hull = ColliderDesc::new(ColliderGeometry::ConvexHull(vec![
        Vec3::new(0.5, 0.9, 0.5),
        Vec3::new(0.4, 1.5, 0.5),
        Vec3::new(0.5, 1.5, 0.4),
        Vec3::new(0.6, 1.5, 0.6),
    ]));
    hull.parent = Some(body);
    scene.insert_collider(hull).unwrap();
    let mut contacts = Contacts::with_capacity(0);
    CollisionPipeline::new()
        .collide(scene.model(), scene.state(), &mut contacts)
        .unwrap();
    assert_eq!(contacts.count, 1);
    close(contacts.depth.host().unwrap()[0], 0.1 / 3.0_f64.sqrt());
    assert_eq!(
        scene.collider_handle(molla_core::ShapeId(contacts.a_shape.host().unwrap()[0])),
        Some(terrain)
    );
}

#[test]
fn ball_on_a_sloped_heightfield_meets_the_plane_under_its_centre_once() {
    let hf =
        Heightfield::from_scaled(2, 2, vec![0.0, 0.0, 0.0, 1.0], Vec3::new(2.0, 2.0, 2.0)).unwrap();
    let mut scene = RigidScene::new().unwrap();
    let terrain = scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Heightfield(hf)))
        .unwrap();
    let mut body = BodyDesc::default();
    body.params.pose = Transform::from_translation(Vec3::new(0.5, 1.5, 0.5));
    let body = scene.insert_body(body).unwrap();
    let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    ball.parent = Some(body);
    scene.insert_collider(ball).unwrap();
    let mut contacts = Contacts::with_capacity(0);
    CollisionPipeline::new()
        .collide(scene.model(), scene.state(), &mut contacts)
        .unwrap();
    assert_eq!(contacts.count, 1);
    close(
        contacts.depth.host().unwrap()[0],
        0.5 - 0.5 / 3.0_f64.sqrt(),
    );
    vector_close(
        contacts.normal.host().unwrap()[0],
        Vec3::new(-1.0, 1.0, -1.0).normalize(),
    );
    assert_eq!(
        scene.collider_handle(molla_core::ShapeId(contacts.a_shape.host().unwrap()[0])),
        Some(terrain)
    );
}

#[test]
fn heightfield_validation_is_atomic_and_accepts_flat_or_inverted_y_scale() {
    for (rows, cols, heights, scale) in [
        (1, 2, vec![0.0; 2], Vec3::ONE),
        (2, 3, vec![0.0; 5], Vec3::ONE),
        (2, 2, vec![0.0; 4], Vec3::new(1.0, 1.0, 0.0)),
        (2, 2, vec![f64::NAN; 4], Vec3::ONE),
        (2, 2, vec![f64::MAX; 4], Vec3::splat(2.0)),
    ] {
        assert!(Heightfield::from_scaled(rows, cols, heights, scale).is_err());
    }
    for y in [0.0, -2.0] {
        let hf = Heightfield::from_scaled(2, 2, vec![1.0; 4], Vec3::new(1.0, y, 2.0)).unwrap();
        close(hf.height_and_normal(0.0, 0.0).unwrap().0, y);
        let mut bad = hf.clone();
        bad.cell_size_z = -1.0;
        let mut scene = RigidScene::new().unwrap();
        let id = scene
            .insert_collider(ColliderDesc::new(ColliderGeometry::Heightfield(hf)))
            .unwrap();
        let revision = scene.revision();
        assert!(
            scene
                .replace_collider(id, ColliderDesc::new(ColliderGeometry::Heightfield(bad)))
                .is_err()
        );
        assert_eq!(scene.revision(), revision);
    }
}

#[test]
fn rotated_heightfield_collision_sweeps_all_query_bounds_corners() {
    let mut scene = RigidScene::new().unwrap();
    let mut ground = ColliderDesc::new(ColliderGeometry::Heightfield(
        Heightfield::from_scaled(5, 5, vec![0.0; 25], Vec3::new(4.0, 1.0, 4.0)).unwrap(),
    ));
    ground.pose = Transform::from_rotation(Quat::from_rotation_y(std::f64::consts::FRAC_PI_4));
    scene.insert_collider(ground).unwrap();
    let mut body = BodyDesc::default();
    body.params.pose = Transform::from_translation(Vec3::Y * 0.4);
    let body = scene.insert_body(body).unwrap();
    let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    ball.parent = Some(body);
    scene.insert_collider(ball).unwrap();
    let mut contacts = Contacts::with_capacity(0);
    CollisionPipeline::new()
        .collide(scene.model(), scene.state(), &mut contacts)
        .unwrap();
    assert!(contacts.count > 0);
    assert!(
        (contacts.depth.host().unwrap()[..contacts.count as usize]
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
            - 0.1)
            .abs()
            < 1e-5
    );
}

#[test]
fn tangent_rays_and_boundary_origins_are_finite_and_bounded() {
    let mut scene = RigidScene::new().unwrap();
    scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Ball { radius: 1.0 }))
        .unwrap();
    let tangent = cast_ray(&scene, Vec3::new(-3.0, 1.0, 0.0), Vec3::X, f64::INFINITY)
        .unwrap()
        .unwrap();
    assert!((tangent.distance - 3.0).abs() < 1e-6);
    vector_close(tangent.normal, Vec3::Y);
    assert!(
        cast_ray(&scene, Vec3::new(-3.0, 1.00001, 0.0), Vec3::X, 10.0)
            .unwrap()
            .is_none()
    );
    close(
        cast_ray(&scene, -Vec3::X, -Vec3::X, 0.0)
            .unwrap()
            .unwrap()
            .distance,
        0.0,
    );
}

#[test]
fn rays_along_flat_surface_return_start_of_tangent_interval() {
    for (geometry, origin, direction, expected) in [
        (
            ColliderGeometry::RoundCylinder {
                half_height: 0.5,
                radius: 1.0,
                border_radius: 0.2,
            },
            Vec3::new(-3.0, 0.7, 0.0),
            Vec3::X,
            2.0,
        ),
        (
            ColliderGeometry::Capsule {
                half_height: 1.0,
                radius: 0.5,
            },
            Vec3::new(0.5, -3.0, 0.0),
            Vec3::Y,
            2.0,
        ),
    ] {
        let mut scene = RigidScene::new().unwrap();
        scene.insert_collider(ColliderDesc::new(geometry)).unwrap();
        let hit = cast_ray(&scene, origin, direction, 10.0).unwrap().unwrap();
        assert!((hit.distance - expected).abs() < 1e-7, "{hit:?}");
        assert!(
            cast_ray(&scene, origin, direction, expected - 1e-4)
                .unwrap()
                .is_none()
        );
    }
}

#[test]
fn primitive_rays_return_nearest_metric_distance_and_inside_zero() {
    for (geometry, o, expected, normal) in [
        (
            ColliderGeometry::Ball { radius: 1.0 },
            Vec3::new(-3.0, 0.6, 0.0),
            2.2,
            Vec3::new(-0.8, 0.6, 0.0),
        ),
        (
            ColliderGeometry::Cuboid {
                half_extents: Vec3::ONE,
            },
            -Vec3::X * 3.0,
            2.0,
            -Vec3::X,
        ),
        (
            ColliderGeometry::Cylinder {
                half_height: 0.5,
                radius: 1.0,
            },
            -Vec3::X * 3.0,
            2.0,
            -Vec3::X,
        ),
        (
            ColliderGeometry::RoundCylinder {
                half_height: 0.5,
                radius: 1.0,
                border_radius: 0.2,
            },
            Vec3::new(-3.0, 0.62, 0.0),
            1.84,
            Vec3::new(-0.8, 0.6, 0.0),
        ),
        (
            ColliderGeometry::Capsule {
                half_height: 1.0,
                radius: 0.5,
            },
            Vec3::new(-3.0, 1.3, 0.0),
            2.6,
            Vec3::new(-0.8, 0.6, 0.0),
        ),
        (
            ColliderGeometry::CapsuleEndpoints {
                a: -Vec3::X,
                b: Vec3::X,
                radius: 0.5,
            },
            -Vec3::X * 3.0,
            1.5,
            -Vec3::X,
        ),
    ] {
        let mut scene = RigidScene::new().unwrap();
        let id = scene.insert_collider(ColliderDesc::new(geometry)).unwrap();
        for magnitude in [0.001, 1.0, 100.0, 1e308] {
            let hit = cast_ray(&scene, o, Vec3::X * magnitude, 10.0)
                .unwrap()
                .unwrap();
            close(hit.distance, expected);
            vector_close(hit.normal, normal);
            assert_eq!(hit.collider, id);
            assert!(
                cast_ray(&scene, o, Vec3::X, expected - 1e-5)
                    .unwrap()
                    .is_none()
            );
        }
        close(
            cast_ray(&scene, Vec3::ZERO, Vec3::X, 0.0)
                .unwrap()
                .unwrap()
                .distance,
            0.0,
        );
        assert!(cast_ray(&scene, o, -Vec3::X, 10.0).unwrap().is_none());
    }
}

#[test]
fn rays_respect_live_pose_edits_filters_sensors_and_stable_ids() {
    let mut scene = RigidScene::new().unwrap();
    let parent = scene.insert_body(BodyDesc::default()).unwrap();
    let mut near = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    near.parent = Some(parent);
    near.pose = Transform::from_translation(Vec3::X * 2.0);
    near.sensor = true;
    near.memberships = 0;
    near.filter = 0;
    let a = scene.insert_collider(near.clone()).unwrap();
    let mut far = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
    far.pose = Transform::from_translation(Vec3::X * 5.0);
    let b = scene.insert_collider(far).unwrap();
    let hit = cast_ray(&scene, Vec3::ZERO, Vec3::X, 10.0)
        .unwrap()
        .unwrap();
    assert_eq!((hit.collider, hit.body), (a, Some(parent)));
    close(hit.distance, 1.5);
    assert_eq!(
        cast_ray_filtered(&scene, Vec3::ZERO, Vec3::X, 10.0, |id| id != a)
            .unwrap()
            .unwrap()
            .collider,
        b
    );
    near.enabled = false;
    scene.replace_collider(a, near).unwrap();
    assert_eq!(
        cast_ray(&scene, Vec3::ZERO, Vec3::X, 10.0)
            .unwrap()
            .unwrap()
            .collider,
        b
    );
    scene.remove_collider(a).unwrap();
    scene
        .set_collider_pose(b, Transform::from_translation(Vec3::X * 7.0))
        .unwrap();
    close(
        cast_ray(&scene, Vec3::ZERO, Vec3::X, 10.0)
            .unwrap()
            .unwrap()
            .distance,
        6.5,
    );
    scene.remove_body(parent).unwrap();
    assert_eq!(
        cast_ray(&scene, Vec3::ZERO, Vec3::X, 10.0)
            .unwrap()
            .unwrap()
            .collider,
        b
    );
    for (direction, max) in [
        (Vec3::ZERO, 1.0),
        (Vec3::NAN, 1.0),
        (Vec3::X, -1.0),
        (Vec3::X, f64::NAN),
    ] {
        assert!(cast_ray(&scene, Vec3::ZERO, direction, max).is_err());
    }
}

#[test]
fn ray_poses_follow_body_motion_without_stepping_or_using_com_as_origin() {
    let mut scene = RigidScene::new().unwrap();
    let mut body = BodyDesc::default();
    body.params.pose = Transform::new(Vec3::new(10.0, 2.0, -7.0), Quat::from_rotation_x(0.7));
    body.params.com = Vec3::new(6.0, 7.0, 8.0);
    let parent = scene.insert_body(body).unwrap();
    let mut desc = ColliderDesc::new(ColliderGeometry::RoundCylinder {
        half_height: 0.5,
        radius: 1.0,
        border_radius: 0.2,
    });
    desc.parent = Some(parent);
    desc.pose = Transform::new(Vec3::new(0.5, 1.0, 2.0), Quat::from_rotation_z(0.4));
    let id = scene.insert_collider(desc).unwrap();
    for translation in [Vec3::new(10.0, 2.0, -7.0), Vec3::new(-6.0, 9.0, 3.0)] {
        scene.state_mut().body_q.host_mut().unwrap()[0].position = translation;
        let pose = scene.collider_world_pose(id).unwrap();
        let hit = cast_ray(
            &scene,
            pose.transform_point(Vec3::new(-3.0, 0.62, 0.0)),
            pose.rotation * Vec3::X,
            10.0,
        )
        .unwrap()
        .unwrap();
        close(hit.distance, 1.84);
        vector_close(hit.normal, pose.rotation * Vec3::new(-0.8, 0.6, 0.0));
    }
}

#[test]
fn large_heightfield_vertical_rays_stay_on_their_local_cells() {
    let (rows, cols) = (257, 513);
    let mut scene = RigidScene::new().unwrap();
    let hf = Heightfield::from_scaled(
        rows,
        cols,
        vec![0.3; rows * cols],
        Vec3::new(512.0, 2.0, 128.0),
    )
    .unwrap();
    scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::Heightfield(hf)))
        .unwrap();
    let start = std::time::Instant::now();
    for n in 0..512 {
        let origin = Vec3::new(
            (n % 511) as f64 - 255.5,
            10.0,
            (n % 255) as f64 * 0.5 - 63.5,
        );
        close(
            cast_ray(&scene, origin, -Vec3::Y, 10.0)
                .unwrap()
                .unwrap()
                .distance,
            9.4,
        );
    }
    eprintln!(
        "512 vertical rays over 131841 samples: {:?}",
        start.elapsed()
    );
}

fn grid_mesh(hf: &Heightfield) -> TriMesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();
    for j in 0..hf.nz - 1 {
        for i in 0..hf.nx - 1 {
            for triangle in hf.cell_triangles(i, j).unwrap() {
                let start = vertices.len() as u32;
                vertices.extend(triangle);
                indices.push([start, start + 1, start + 2]);
            }
        }
    }
    TriMesh { vertices, indices }
}

#[test]
fn transformed_heightfield_dda_matches_triangle_mesh_for_oblique_and_boundary_rays() {
    let heights = (0..35)
        .map(|i| ((i * 17 % 23) as f64 - 11.0) * 0.1)
        .collect();
    let hf = Heightfield::from_scaled(5, 7, heights, Vec3::new(12.0, 2.0, 5.0)).unwrap();
    let mesh = grid_mesh(&hf);
    let pose = Transform::new(
        Vec3::new(10.0, -3.0, 4.0),
        Quat::from_rotation_y(0.6) * Quat::from_rotation_x(0.2),
    );
    let mut scene = RigidScene::new().unwrap();
    let mut field = ColliderDesc::new(ColliderGeometry::Heightfield(hf));
    field.pose = pose;
    let field = scene.insert_collider(field).unwrap();
    let mut triangles = ColliderDesc::new(ColliderGeometry::TriMesh(mesh));
    triangles.pose = pose;
    let triangles = scene.insert_collider(triangles).unwrap();
    let mut hits = 0;
    for n in 0..200 {
        let x = (n * 29 % 101) as f64 / 100.0 * 16.0 - 8.0;
        let z = (n * 43 % 97) as f64 / 96.0 * 7.0 - 3.5;
        let o = pose.transform_point(Vec3::new(x, 5.0, z));
        let d =
            pose.rotation * Vec3::new((n % 5) as f64 * 0.3 - 0.6, -1.0, (n % 3) as f64 * 0.2 - 0.2);
        let a = cast_ray_filtered(&scene, o, d, 20.0, |id| id == field).unwrap();
        let b = cast_ray_filtered(&scene, o, d, 20.0, |id| id == triangles).unwrap();
        assert_eq!(a.is_some(), b.is_some(), "ray {n}: {a:?} {b:?}");
        if let (Some(a), Some(b)) = (a, b) {
            hits += 1;
            close(a.distance, b.distance);
            vector_close(a.normal, b.normal);
        }
    }
    assert!(hits > 50);
}

#[test]
fn convex_and_mesh_rays_include_endpoints_and_parallel_slabs() {
    let mut points = Vec::new();
    for x in [-1.0, 1.0] {
        for y in [-1.0, 1.0] {
            for z in [-1.0, 1.0] {
                points.push(Vec3::new(x, y, z));
            }
        }
    }
    let mut scene = RigidScene::new().unwrap();
    let hull = scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::ConvexHull(points)))
        .unwrap();
    close(
        cast_ray(&scene, -Vec3::X * 3.0, Vec3::X, 2.0)
            .unwrap()
            .unwrap()
            .distance,
        2.0,
    );
    close(
        cast_ray(&scene, Vec3::ZERO, Vec3::Z, 0.0)
            .unwrap()
            .unwrap()
            .distance,
        0.0,
    );
    scene.remove_collider(hull).unwrap();
    scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::TriMesh(TriMesh {
            vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Z],
            indices: vec![[0, 2, 1]],
        })))
        .unwrap();
    for p in [Vec3::ZERO, Vec3::X * 0.5, Vec3::new(0.2, 0.0, 0.2)] {
        close(
            cast_ray(&scene, p + Vec3::Y, -Vec3::Y, 1.0)
                .unwrap()
                .unwrap()
                .distance,
            1.0,
        );
        close(
            cast_ray(&scene, p, -Vec3::Y, 0.0)
                .unwrap()
                .unwrap()
                .distance,
            0.0,
        );
    }
}

#[test]
fn a_ray_among_given_colliders_passes_through_the_rest() {
    use molla_geometry::raycast_cpu::CpuRaycaster;
    let mut scene = RigidScene::new().unwrap();
    let mut ball = |x: f64| {
        let mut desc = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.5 });
        desc.pose = Transform::from_translation(Vec3::X * x);
        scene.insert_collider(desc).unwrap()
    };
    let (near, middle, far) = (ball(2.0), ball(5.0), ball(8.0));
    let mut queries = CpuRaycaster::default();
    let hit = |queries: &mut CpuRaycaster, ids: &[_]| {
        queries
            .cast_ray_among(&scene, ids.iter().copied(), Vec3::ZERO, Vec3::X, 20.0)
            .unwrap()
    };
    assert_eq!(hit(&mut queries, &[far, middle]).unwrap().collider, middle);
    close(hit(&mut queries, &[far]).unwrap().distance, 7.5);
    assert_eq!(hit(&mut queries, &[near, far]).unwrap().collider, near);
    assert!(hit(&mut queries, &[]).is_none());
}
