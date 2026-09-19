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
