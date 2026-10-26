use molla_core::BodyId;
use molla_geometry::{CollisionPipeline, NarrowPhaseBackend};
use molla_math::{Quat, Transform, Vec3};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder, State, TriMesh};

fn fixture(mesh_first: bool) -> (Model, State, usize) {
    let mut builder = ModelBuilder::new();
    let bodies = [-0.5, 0.5]
        .map(|x| builder.add_body(BodyParams::new().with_position(Vec3::new(x, 0.15, 0.0))));
    let mesh = builder.add_mesh(TriMesh {
        vertices: [0.0, -4.0]
            .into_iter()
            .flat_map(|y| {
                [
                    Vec3::new(-3.0, y, -3.0),
                    Vec3::new(3.0, y, -3.0),
                    Vec3::new(3.0, y, 3.0),
                    Vec3::new(-3.0, y, 3.0),
                ]
            })
            .collect(),
        indices: vec![[0, 2, 1], [0, 3, 2]],
    });
    if mesh_first {
        builder
            .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
            .unwrap();
    }
    for body in bodies {
        builder
            .add_shape_sphere(body, 0.25, Transform::IDENTITY)
            .unwrap();
    }
    if !mesh_first {
        builder
            .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
            .unwrap();
    }
    let model = builder.build().unwrap();
    let state = model.state().unwrap();
    (model, state, if mesh_first { 0 } else { 2 })
}

fn compare(pipeline: &mut CollisionPipeline, model: &Model, state: &State) -> u32 {
    let mut cached = Contacts::with_capacity(16);
    let mut reference = Contacts::with_capacity(16);
    pipeline.collide(model, state, &mut cached).unwrap();
    molla_geometry::pipeline::collide(model, state, &mut reference, &mut Vec::new()).unwrap();
    assert_eq!(cached.count, reference.count);
    let n = cached.count as usize;
    macro_rules! same {
        ($($field:ident),+) => { $(assert_eq!(cached.$field.host().unwrap()[..n], reference.$field.host().unwrap()[..n], stringify!($field));)+ };
    }
    same!(
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
        cached.body_contact_offsets.host().unwrap(),
        reference.body_contact_offsets.host().unwrap()
    );
    assert_eq!(
        cached.body_contact_indices.host().unwrap(),
        reference.body_contact_indices.host().unwrap()
    );
    cached.count
}

#[test]
fn cached_mesh_contacts_follow_raw_geometry_edits_and_live_poses() {
    for mesh_first in [false, true] {
        for backend in [NarrowPhaseBackend::Gjk, NarrowPhaseBackend::MprCpu] {
            let (mut model, mut state, mesh_shape) = fixture(mesh_first);
            let mut pipeline = CollisionPipeline::new().with_narrow_phase_backend(backend);
            for _ in 0..3 {
                assert!(compare(&mut pipeline, &model, &state) > 0);
                assert_eq!(pipeline.mesh_bvh_build_count(), 1);
            }
            state.body_q.host_mut().unwrap()[0].position.x += 0.1;
            model.shape_transform.host_mut().unwrap()[mesh_shape].rotation =
                Quat::from_rotation_z(0.05);
            model.shape_friction.host_mut().unwrap()[mesh_shape] = 0.21;
            assert!(compare(&mut pipeline, &model, &state) > 0);
            assert_eq!(pipeline.mesh_bvh_build_count(), 1);
            model.shape_transform.host_mut().unwrap()[mesh_shape] = Transform::IDENTITY;

            for vertex in &mut model.meshes[0].vertices {
                vertex.x += 10.0;
            }
            for pose in state.body_q.host_mut().unwrap() {
                pose.position.x += 10.0;
            }
            assert!(compare(&mut pipeline, &model, &state) > 0);
            assert_eq!(pipeline.mesh_bvh_build_count(), 2);

            model.meshes[0]
                .indices
                .copy_from_slice(&[[4, 6, 5], [4, 7, 6]]);
            for pose in state.body_q.host_mut().unwrap() {
                pose.position.y -= 4.0;
            }
            assert!(compare(&mut pipeline, &model, &state) > 0);
            assert_eq!(pipeline.mesh_bvh_build_count(), 3);
            model.meshes[0].vertices.push(Vec3::new(12.0, -4.0, 0.0));
            assert!(compare(&mut pipeline, &model, &state) > 0);
            assert_eq!(pipeline.mesh_bvh_build_count(), 4);

            let (replacement, replacement_state, _) = fixture(mesh_first);
            assert!(compare(&mut pipeline, &replacement, &replacement_state) > 0);
            assert_eq!(pipeline.mesh_bvh_build_count(), 5);
            let empty = ModelBuilder::new().build().unwrap();
            assert_eq!(compare(&mut pipeline, &empty, &empty.state().unwrap()), 0);
            assert!(compare(&mut pipeline, &replacement, &replacement_state) > 0);
            assert_eq!(pipeline.mesh_bvh_build_count(), 6);
        }
    }
}

#[test]
fn excluded_pairs_do_not_prepare_meshes_and_reenabling_reuses_them() {
    let (model, state, _) = fixture(false);
    let mut pipeline = CollisionPipeline::new();
    let mut contacts = Contacts::with_capacity(16);
    pipeline
        .collide_filtered(&model, &state, &mut contacts, |_, _| true)
        .unwrap();
    assert_eq!((contacts.count, pipeline.mesh_bvh_build_count()), (0, 0));
    assert!(compare(&mut pipeline, &model, &state) > 0);
    pipeline
        .collide_filtered(&model, &state, &mut contacts, |_, _| true)
        .unwrap();
    assert_eq!((contacts.count, pipeline.mesh_bvh_build_count()), (0, 1));
    assert!(compare(&mut pipeline, &model, &state) > 0);
    assert_eq!(pipeline.mesh_bvh_build_count(), 1);
}

#[cfg(feature = "gpu")]
#[test]
#[ignore = "needs a real wgpu adapter — run via make test-gpu"]
fn gpu_selected_mesh_fallbacks_reuse_and_refresh_bvhs() {
    let engine = molla_compute::Engine::new_gpu().expect("gpu");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for mode in 0..3 {
        let (mut model, mut state, _) = fixture(false);
        model.upload_to_device(device, queue).unwrap();
        let mut pipeline = CollisionPipeline::new();
        match mode {
            0 => {
                pipeline.set_narrow_phase_backend(NarrowPhaseBackend::MprGpu);
                pipeline.enable_mpr_gpu_backend(&engine, 4, 1e-7).unwrap();
            }
            1 => pipeline.enable_full_gpu_narrow_phase(&engine, 4).unwrap(),
            _ => pipeline.enable_gpu_prefilter(&engine, 4).unwrap(),
        }
        for _ in 0..3 {
            assert!(compare(&mut pipeline, &model, &state) > 0);
            assert_eq!(pipeline.last_cpu_narrow_phase_calls(), 2);
            assert_eq!(pipeline.mesh_bvh_build_count(), 1);
        }
        for vertex in &mut model.meshes[0].vertices {
            vertex.x += 10.0;
        }
        for pose in state.body_q.host_mut().unwrap() {
            pose.position.x += 10.0;
        }
        assert!(compare(&mut pipeline, &model, &state) > 0);
        assert_eq!(pipeline.mesh_bvh_build_count(), 2);
    }
}

/// A 40 m square of two triangles at y = 0.
fn floor() -> TriMesh {
    TriMesh {
        vertices: vec![
            Vec3::new(-20.0, 0.0, -20.0),
            Vec3::new(20.0, 0.0, -20.0),
            Vec3::new(20.0, 0.0, 20.0),
            Vec3::new(-20.0, 0.0, 20.0),
        ],
        indices: vec![[0, 2, 1], [0, 3, 2]],
    }
}

/// Where 144 balls a metre apart sink 5 cm into the floor.
fn balls() -> impl Iterator<Item = Vec3> {
    (0..144).map(|n| Vec3::new((n / 12) as f64 - 6.0, 0.2, (n % 12) as f64 - 6.0))
}

/// The contacts by shape pair and point, whatever order they were found in.
fn rows(contacts: &Contacts) -> Vec<(u32, u32, [f64; 3], f64)> {
    let n = contacts.count as usize;
    let (a, b) = (contacts.a_shape.host().unwrap(), contacts.b_shape.host().unwrap());
    let (point, depth) = (contacts.point.host().unwrap(), contacts.depth.host().unwrap());
    let mut rows: Vec<_> = (0..n)
        .map(|i| (a[i], b[i], [point[i].x, point[i].y, point[i].z], depth[i]))
        .collect();
    rows.sort_by(|x, y| (x.0, x.1).cmp(&(y.0, y.1)).then(x.2[0].total_cmp(&y.2[0])));
    rows
}

fn reference(model: &Model, state: &State) -> Vec<(u32, u32, [f64; 3], f64)> {
    let mut contacts = Contacts::with_capacity(16);
    molla_geometry::pipeline::collide(model, state, &mut contacts, &mut Vec::new()).unwrap();
    rows(&contacts)
}

#[test]
fn many_pairs_across_the_pool_find_the_contacts_one_at_a_time_does() {
    let mut builder = ModelBuilder::new();
    let floor = builder.add_mesh(floor());
    builder
        .add_shape_trimesh(BodyId::NONE, floor, Transform::IDENTITY)
        .unwrap();
    for at in balls() {
        let ball = builder.add_body(BodyParams::new().with_position(at));
        builder
            .add_shape_sphere(ball, 0.25, Transform::IDENTITY)
            .unwrap();
    }
    let model = builder.build().unwrap();
    let state = model.state().unwrap();
    let mut pooled = Contacts::with_capacity(16);
    CollisionPipeline::new()
        .collide(&model, &state, &mut pooled)
        .unwrap();
    assert!(pooled.count >= 144, "{} contacts", pooled.count);
    assert_eq!(rows(&pooled), reference(&model, &state));
}

#[test]
fn a_scene_s_moving_shapes_sweep_across_the_pool_to_the_pairs_one_sweep_finds() {
    use molla_geometry::shape_world::SceneShapeBounds;
    use molla_sim::runtime::{BodyDesc, ColliderDesc, ColliderGeometry, RigidScene};
    let mut scene = RigidScene::new().unwrap();
    scene
        .insert_collider(ColliderDesc::new(ColliderGeometry::TriMesh(floor())))
        .unwrap();
    for at in balls() {
        let params = BodyParams::new().with_pose(Transform::from_translation(at));
        let body = scene
            .insert_body(BodyDesc { params, ..Default::default() })
            .unwrap();
        let mut ball = ColliderDesc::new(ColliderGeometry::Ball { radius: 0.25 });
        ball.parent = Some(body);
        scene.insert_collider(ball).unwrap();
    }
    let mut bounds = SceneShapeBounds::default();
    let mut pooled = Contacts::with_capacity(16);
    CollisionPipeline::new()
        .collide_prepared(&bounds.prepare(&scene), scene.state(), &mut pooled, |_, _| false)
        .unwrap();
    assert!(pooled.count >= 144, "{} contacts", pooled.count);
    assert_eq!(rows(&pooled), reference(scene.model(), scene.state()));
}
