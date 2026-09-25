#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Transform, Vec3};
use molla_sensors::{Raycaster, SensorScene};
use molla_sim::{ModelBuilder, TriMesh};
use std::sync::Mutex;

static GPU: Mutex<()> = Mutex::new(());

fn quantized(value: Vec3) -> Vec3 {
    value.as_vec3().as_dvec3()
}

fn reference(
    vertices: &[Vec3],
    indices: &[[u32; 3]],
    origin: Vec3,
    direction: Vec3,
) -> Option<(f64, Vec3)> {
    let direction = quantized(direction).normalize();
    let origin = quantized(origin);
    let mut nearest = None;
    for triangle in indices {
        let [a, b, c] = triangle.map(|i| quantized(vertices[i as usize]));
        let area = (b - a).cross(c - a);
        if area.length_squared() == 0.0 {
            continue;
        }
        let mut normal = area.normalize();
        let denominator = normal.dot(direction);
        if denominator == 0.0 {
            continue;
        }
        let distance = normal.dot(a - origin) / denominator;
        if !(distance > 0.0 && distance < 1_000_010.0) {
            continue;
        }
        let point = origin + direction * distance;
        if [(a, b), (b, c), (c, a)]
            .iter()
            .any(|&(start, end)| (end - start).cross(point - start).dot(normal) < 0.0)
        {
            continue;
        }
        if denominator > 0.0 {
            normal = -normal;
        }
        if nearest.is_none_or(|(old, _)| distance < old) {
            nearest = Some((distance, normal));
        }
    }
    nearest
}

fn compare(engine: &Engine, scale: f64, rays: &[(Vec3, Vec3)]) {
    let vertices = vec![
        Vec3::new(-scale, -scale, 0.0),
        Vec3::new(scale, -scale, 0.0),
        Vec3::new(0.0, scale, 0.0),
        Vec3::ZERO,
        Vec3::X * scale,
        -Vec3::X * scale,
    ];
    let indices = vec![[0, 0, 1], [3, 4, 5], [0, 1, 2]];
    let expected: Vec<_> = rays
        .iter()
        .map(|&(o, d)| reference(&vertices, &indices, o, d))
        .collect();
    assert!(expected.iter().any(Option::is_some));
    assert!(expected.iter().any(Option::is_none));
    let mut builder = ModelBuilder::new();
    builder.add_body(Default::default());
    let mesh = builder.add_mesh(TriMesh { vertices, indices });
    builder
        .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let scene = SensorScene::from_model(engine, &model, Default::default()).unwrap();
    let origins: Vec<_> = rays.iter().map(|r| r.0).collect();
    let directions: Vec<_> = rays.iter().map(|r| r.1).collect();
    let mut query = Raycaster::new(engine, rays.len() as u32).unwrap();
    for owned in [false, true] {
        let hits = if owned {
            query.cast_scene(engine, &scene, &state, &origins, &directions, 1_000_010.0)
        } else {
            query.cast(engine, &model, &state, &origins, &directions, 1_000_010.0)
        }
        .unwrap();
        for (i, (hit, expected)) in hits.iter().zip(&expected).enumerate() {
            let label = format!("scale={scale:e} owned={owned} ray={:?}", rays[i]);
            let Some((distance, normal)) = expected else {
                assert!(hit.shape_id.is_none(), "{label}: {hit:?}");
                continue;
            };
            assert_eq!(hit.shape_id, Some(0), "{label}: {hit:?}");
            let magnitude = distance.max(1.0) as f32;
            let tolerance = 2.0 * f64::from(magnitude.next_up() - magnitude) + 2e-4;
            assert!(
                (hit.t - distance).abs() < tolerance,
                "{label}: {hit:?} expected={distance}"
            );
            assert!(
                (hit.normal - normal).length() < 4e-4,
                "{label}: {hit:?} normal={normal:?}"
            );
            assert!((hit.normal.length() - 1.0).abs() < 2e-6, "{label}: {hit:?}");
            let point = quantized(rays[i].0) + quantized(rays[i].1).normalize() * distance;
            assert!(
                (hit.point - point).length() < 3.0 * tolerance,
                "{label}: {hit:?} point={point:?}"
            );
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn tiny_and_large_meshes_with_degenerate_neighbors_match_f64_reference() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    for scale in [1e-30, 1e-20, 1e-10, 1e-7, 1.0, 1e10, 1e20, 1e30] {
        let mut rays = Vec::new();
        for distance in [4.0, 1_000_000.0] {
            for sign in [-1.0, 1.0] {
                for magnitude in [1e-30, 1.0, 1e30] {
                    for x in [0.0, 0.25 * scale, 1.5 * scale] {
                        rays.push((
                            Vec3::new(x, 0.0, sign * distance),
                            Vec3::Z * (-sign * magnitude),
                        ));
                    }
                }
            }
        }
        compare(&engine, scale, &rays);
    }
}

#[test]
#[ignore = "requires GPU"]
fn finite_grazing_mesh_hits_and_parallel_misses_match_f64_reference() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut rays = Vec::new();
    for slope in [1e-6, 1e-10, 1e-13, 1e-15] {
        for magnitude in [1e-15, 1.0, 1e15] {
            rays.push((
                Vec3::new(-1.0, 0.0, slope),
                Vec3::new(1.0, 0.0, -slope) * magnitude,
            ));
            rays.push((Vec3::new(-1.0, 0.0, slope), Vec3::X * magnitude));
        }
    }
    rays.push((Vec3::ZERO, Vec3::X));
    rays.push((Vec3::ZERO, -Vec3::Z));
    compare(&engine, 2.0, &rays);
}

#[test]
#[ignore = "requires GPU"]
fn long_oblique_mesh_hits_and_misses_match_f64_reference() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut rays = Vec::new();
    for distance in [4.0, 1000.0, 10_000.0, 1_000_000.0] {
        for x in [-0.25, 0.0, 0.25, 1.5] {
            for magnitude in [1e-30, 1.0, 1e30] {
                rays.push((
                    Vec3::new(distance * 0.75 + x, distance * 0.25, distance),
                    Vec3::new(-0.75, -0.25, -1.0) * magnitude,
                ));
            }
        }
    }
    compare(&engine, 1.0, &rays);
}
