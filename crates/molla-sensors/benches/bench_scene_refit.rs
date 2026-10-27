//! Completed hierarchy construction and warmed GPU refitting, without ray traversal.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Transform, Vec3};
use molla_sensors::{SceneAcceleration, SensorScene};
use molla_sim::{ModelBuilder, TriMesh};
use std::time::Duration;

fn bench_refit(c: &mut Criterion) {
    let engine = Engine::new_gpu().expect("scene refit benchmark requires GPU");
    eprintln!("scene refit adapter: {:?}", engine.adapter_info());
    let device = engine.device().unwrap();
    let wait = || {
        device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(Duration::from_secs(30)),
            })
            .unwrap()
    };
    let mut group = c.benchmark_group("scene_refit");
    group
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    for n in [1_000, 10_000] {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(Default::default());
        let mesh = builder.add_mesh(TriMesh {
            vertices: vec![-Vec3::X, Vec3::X, Vec3::Z],
            indices: vec![[0, 1, 2]],
        });
        for i in 0..n {
            let pose = Transform::from_translation(Vec3::new(
                (i % 100) as f64 * 3.0,
                0.0,
                (i / 100) as f64 * 3.0,
            ));
            if i % 4 == 0 {
                builder.add_shape_sphere(body, 0.5, pose).unwrap();
            } else {
                builder.add_shape_trimesh(body, mesh, pose).unwrap();
            }
        }
        builder
            .add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY)
            .unwrap();
        let model = builder.build().unwrap();
        let mut state = model.state().unwrap();
        state
            .upload_to_device(device, engine.queue().unwrap())
            .unwrap();
        let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
        group.bench_function(BenchmarkId::new("construct_completed", n), |b| {
            b.iter(|| {
                let hierarchy =
                    SceneAcceleration::new(&engine, &scene, None, Default::default()).unwrap();
                engine.queue().unwrap().submit([]);
                wait();
                std::hint::black_box(hierarchy);
            })
        });
        let mut hierarchy =
            SceneAcceleration::new(&engine, &scene, None, Default::default()).unwrap();
        hierarchy.dispatch(&engine, &scene, &state).unwrap();
        wait();
        group.bench_function(BenchmarkId::new("warm_refit_completed", n), |b| {
            b.iter(|| {
                hierarchy.dispatch(&engine, &scene, &state).unwrap();
                wait();
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_refit);
criterion_main!(benches);
