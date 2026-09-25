#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_geometry::{MeshBvh, MeshBvhGpu, MeshBvhRaycastKernel};
use molla_kernels::CameraTraceKernel;
use molla_math::{Quat, Transform, Vec3};
use molla_sensors::{CameraChannels, RayHit, Raycaster, SensorScene};
use molla_sim::{Model, ModelBuilder, State};
use molla_storage::{Buffer, BufferElement, BufferUsage};
use std::sync::Mutex;

static GPU: Mutex<()> = Mutex::new(());

fn distance_error(distance: f64) -> f64 {
    let magnitude = distance.abs().max(1.0) as f32;
    2.0 * f64::from(magnitude.next_up() - magnitude) + 2e-4
}

fn fixture(engine: &Engine, kind: usize) -> (Model, State) {
    let mut builder = ModelBuilder::new();
    builder.add_body(Default::default());
    match kind {
        0 => builder.add_shape_sphere(BodyId::NONE, 0.7, Transform::IDENTITY),
        1 => builder.add_shape_box(BodyId::NONE, Vec3::new(0.7, 1.1, 0.9), Transform::IDENTITY),
        2 => builder.add_shape_capsule(BodyId::NONE, 1.0, 0.7, Transform::IDENTITY),
        3 => builder.add_shape_cylinder(BodyId::NONE, 1.0, 0.7, Transform::IDENTITY),
        4 => builder.add_shape_cone(BodyId::NONE, 1.0, 1.4, Transform::IDENTITY),
        5 => {
            builder.add_shape_ellipsoid(BodyId::NONE, Vec3::new(0.7, 1.1, 0.9), Transform::IDENTITY)
        }
        _ => builder.add_shape_plane(BodyId::NONE, Vec3::Y, Transform::IDENTITY),
    }
    .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    (model, state)
}

fn buffer<T: BufferElement>(engine: &Engine, values: Vec<T>) -> Buffer<T> {
    let mut buffer = Buffer::from_vec(values, BufferUsage::StorageReadWrite);
    buffer.ensure_device(engine.device().unwrap()).unwrap();
    buffer.upload(engine.queue().unwrap()).unwrap();
    buffer
}

fn query(
    engine: &Engine,
    model: &Model,
    state: &State,
    origins: &[Vec3],
    directions: &[Vec3],
    range: f64,
    mode: u32,
) -> Vec<RayHit> {
    if mode < 2 {
        let mut raycaster = Raycaster::new(engine, origins.len() as u32).unwrap();
        return if mode == 0 {
            raycaster.cast(engine, model, state, origins, directions, range)
        } else {
            let scene = SensorScene::from_model(engine, model, Default::default()).unwrap();
            raycaster.cast_scene(engine, &scene, state, origins, directions, range)
        }
        .unwrap();
    }
    let (device, queue) = (engine.device().unwrap(), engine.queue().unwrap());
    let mut depth = buffer(engine, vec![0.0f64; origins.len()]);
    let mut color = buffer(engine, vec![0u32; origins.len()]);
    let mut normal = buffer(engine, vec![Vec3::ZERO; origins.len()]);
    let mut albedo = buffer(engine, vec![0u32; origins.len()]);
    let mut ids = buffer(engine, vec![u32::MAX; origins.len()]);
    CameraTraceKernel::new(device)
        .dispatch(
            device,
            queue,
            model,
            state.body_q.device_buffer_current().unwrap(),
            &buffer(engine, origins.to_vec()),
            &buffer(engine, directions.to_vec()),
            &mut depth,
            &mut color,
            &mut normal,
            &mut albedo,
            &mut ids,
            origins.len() as u32,
            1,
            1,
            range as f32,
            CameraChannels::ALL.bits(),
            Vec3::ONE,
            Vec3::NEG_Y,
            Vec3::ONE,
        )
        .unwrap();
    depth.download(device, queue).unwrap();
    normal.download(device, queue).unwrap();
    ids.download(device, queue).unwrap();
    (0..origins.len())
        .map(|i| {
            let id = ids.host().unwrap()[i];
            RayHit {
                shape_id: (id != u32::MAX).then_some(id),
                body_id: BodyId::NONE,
                t: if id == u32::MAX {
                    f64::INFINITY
                } else {
                    depth.host().unwrap()[i]
                },
                point: Vec3::ZERO,
                normal: normal.host().unwrap()[i],
            }
        })
        .collect()
}

fn check_long_surfaces(mode: u32) {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let radius = f64::from(0.7f32);
    let z_half = f64::from(0.9f32);
    for kind in 0..6 {
        let (model, state) = fixture(&engine, kind);
        let origins: Vec<_> = [4.0, 1000.0, 10_000.0, 1_000_000.0]
            .into_iter()
            .flat_map(|z| [0.0, 0.25, 0.6875, 2.0].map(|x| Vec3::new(x, 0.0, z)))
            .collect();
        let hits = query(
            &engine,
            &model,
            &state,
            &origins,
            &vec![Vec3::NEG_Z; origins.len()],
            2_000_000.0,
            mode,
        );
        for (i, (origin, hit)) in origins.iter().zip(hits).enumerate() {
            let label = format!("kind={kind}, mode={mode}, ray={i}, origin={origin:?}");
            if origin.x > radius {
                assert_eq!(hit.shape_id, None, "{label}: {hit:?}");
                assert_eq!(hit.t, f64::INFINITY, "{label}");
                continue;
            }
            let z = match kind {
                1 => z_half,
                5 => z_half * (1.0 - (origin.x / radius).powi(2)).sqrt(),
                _ => (radius * radius - origin.x * origin.x).sqrt(),
            };
            let point = Vec3::new(origin.x, 0.0, z);
            let normal = match kind {
                1 => Vec3::Z,
                4 => Vec3::new(origin.x, radius * radius, z).normalize(),
                5 => {
                    Vec3::new(origin.x / (radius * radius), 0.0, z / (z_half * z_half)).normalize()
                }
                _ => point.normalize(),
            };
            let distance = origin.z - z;
            assert_eq!(hit.shape_id, Some(0), "{label}: {hit:?}");
            assert!(
                (hit.t - distance).abs() < distance_error(distance),
                "{label}: {} vs {distance}",
                hit.t
            );
            assert!(
                hit.normal.is_finite() && (hit.normal - normal).length() < 4e-4,
                "{label}: {:?} vs {normal:?}",
                hit.normal
            );
            assert!((hit.normal.length() - 1.0).abs() < 2e-6, "{label}");
            if mode < 2 {
                assert!(
                    (hit.point - point).length() < 4e-4,
                    "{label}: {:?} vs {point:?}",
                    hit.point
                );
            }
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn raw_long_primitive_rays_match_analytic_surfaces() {
    check_long_surfaces(0);
}

#[test]
#[ignore = "requires GPU"]
fn owned_long_primitive_rays_match_analytic_surfaces() {
    check_long_surfaces(1);
}

#[test]
#[ignore = "requires GPU"]
fn low_level_camera_long_rays_match_analytic_surfaces() {
    check_long_surfaces(2);
}

#[test]
#[ignore = "requires GPU"]
fn ellipsoid_misses_remain_misses_when_range_exceeds_old_finite_sentinel() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, state) = fixture(&engine, 5);
    for mode in 0..3 {
        for max_distance in [100.0, 1e31] {
            let hits = query(
                &engine,
                &model,
                &state,
                &[Vec3::new(2.0, 0.0, 2.0)],
                &[Vec3::Y],
                max_distance,
                mode,
            );
            assert_eq!(
                hits[0].shape_id, None,
                "mode={mode}, range={max_distance}: {:?}",
                hits[0]
            );
            assert_eq!(hits[0].t, f64::INFINITY);
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn grazing_planes_keep_finite_intersections_and_parallel_misses() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, state) = fixture(&engine, 6);
    for mode in 0..3 {
        for slope in [1e-6f32, 1e-10, 1e-15] {
            let direction = Vec3::new(1.0, -f64::from(slope), 0.0).normalize();
            let expected = -1.0 / direction.y;
            let hits = query(
                &engine,
                &model,
                &state,
                &[Vec3::Y, Vec3::Y],
                &[direction, Vec3::X],
                expected * 2.0,
                mode,
            );
            assert_eq!(
                hits[0].shape_id,
                Some(0),
                "mode={mode}, slope={slope}: {:?}",
                hits[0]
            );
            assert!((hits[0].t - expected).abs() < distance_error(expected));
            assert!((hits[0].normal - Vec3::Y).length() < 2e-6);
            assert_eq!(hits[1].shape_id, None);
            if mode < 2 {
                assert!(hits[0].point.y.abs() < 2e-6);
            }
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn unit_sphere_silhouette_distinguishes_adjacent_f32_rays() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    builder.add_body(Default::default());
    builder
        .add_shape_sphere(BodyId::NONE, 1.0, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let origins: Vec<_> = [4.0, 1000.0, 10_000.0, 1_000_000.0]
        .into_iter()
        .flat_map(|z| {
            [1.0f32.next_down(), 1.0, 1.0f32.next_up()].map(|x| Vec3::new(f64::from(x), 0.0, z))
        })
        .collect();
    for mode in 0..3 {
        let hits = query(
            &engine,
            &model,
            &state,
            &origins,
            &vec![Vec3::NEG_Z; origins.len()],
            2_000_000.0,
            mode,
        );
        for (origin, hit) in origins.iter().zip(hits) {
            let label = format!("mode={mode}, origin={origin:?}");
            if origin.x > 1.0 {
                assert_eq!(hit.shape_id, None, "{label}: {hit:?}");
            } else {
                let z = (1.0 - origin.x * origin.x).sqrt();
                let point = Vec3::new(origin.x, 0.0, z);
                assert_eq!(hit.shape_id, Some(0), "{label}: {hit:?}");
                assert!(
                    (hit.t - (origin.z - z)).abs() < distance_error(origin.z),
                    "{label}"
                );
                assert!(
                    (hit.normal - point).length() < 1e-4,
                    "{label}: {:?} vs {point:?}",
                    hit.normal
                );
            }
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn oblique_long_sphere_rays_match_independent_chord_geometry() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (model, state) = fixture(&engine, 0);
    let radius = f64::from(0.7f32);
    let mut origins = Vec::new();
    let mut directions = Vec::new();
    for distance in [4.0, 1000.0, 10_000.0, 1_000_000.0] {
        for direction in [
            Vec3::new(0.6, 0.0, 0.8),
            Vec3::new(-0.6, 0.0, 0.8),
            Vec3::new(0.0, 0.8, -0.6),
            Vec3::new(0.8, -0.6, 0.0),
        ] {
            let axis = if direction.y == 0.0 { Vec3::Y } else { Vec3::Z };
            let perpendicular = direction.cross(axis).normalize();
            for offset in [0.0, 0.25, 1.0] {
                origins.push(
                    (-direction * distance + perpendicular * offset)
                        .as_vec3()
                        .as_dvec3(),
                );
                directions.push(direction);
            }
        }
    }
    for mode in 0..3 {
        let hits = query(
            &engine,
            &model,
            &state,
            &origins,
            &directions,
            2_000_000.0,
            mode,
        );
        for (i, hit) in hits.iter().enumerate() {
            let direction = directions[i].as_vec3().as_dvec3().normalize();
            let approach = -origins[i].dot(direction);
            let closest = origins[i] + direction * approach;
            let label = format!("mode={mode}, ray={i}, origin={:?}", origins[i]);
            if closest.length() > radius {
                assert_eq!(hit.shape_id, None, "{label}: {hit:?}");
            } else {
                let chord = (radius * radius - closest.length_squared()).sqrt();
                let distance = approach - chord;
                let point = closest - direction * chord;
                assert_eq!(hit.shape_id, Some(0), "{label}: {hit:?}");
                assert!(
                    (hit.t - distance).abs() < distance_error(distance),
                    "{label}: {} vs {distance}",
                    hit.t
                );
                assert!(
                    (hit.normal - point / radius).length() < 4e-4,
                    "{label}: {:?} vs {:?}",
                    hit.normal,
                    point / radius
                );
                if mode < 2 {
                    assert!(
                        (hit.point - point).length() < 4e-4,
                        "{label}: {:?} vs {point:?}",
                        hit.point
                    );
                }
            }
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn rotating_spheres_preserves_long_range_geometry() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    builder.add_body(Default::default());
    builder
        .add_shape_sphere(
            BodyId::NONE,
            0.7,
            Transform::new(
                Vec3::ZERO,
                Quat::from_rotation_y(0.61) * Quat::from_rotation_x(0.37),
            ),
        )
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let origins: Vec<_> = [4.0, 1000.0, 10_000.0, 1_000_000.0]
        .into_iter()
        .flat_map(|z| [0.0, 0.25, 0.6875, 2.0].map(|x| Vec3::new(x, 0.0, z)))
        .collect();
    let radius = f64::from(0.7f32);
    for mode in 0..3 {
        let hits = query(
            &engine,
            &model,
            &state,
            &origins,
            &vec![Vec3::NEG_Z; origins.len()],
            2_000_000.0,
            mode,
        );
        for (origin, hit) in origins.iter().zip(hits) {
            let label = format!("mode={mode}, origin={origin:?}");
            if origin.x > radius {
                assert_eq!(hit.shape_id, None, "{label}: {hit:?}");
                continue;
            }
            let z = (radius * radius - origin.x * origin.x).sqrt();
            let point = Vec3::new(origin.x, 0.0, z);
            assert_eq!(hit.shape_id, Some(0), "{label}: {hit:?}");
            assert!(
                (hit.t - (origin.z - z)).abs() < distance_error(origin.z),
                "{label}: {} vs {}",
                hit.t,
                origin.z - z
            );
            assert!(
                (hit.normal - point / radius).length() < 4e-4,
                "{label}: {:?} vs {:?}",
                hit.normal,
                point / radius
            );
            if mode < 2 {
                assert!((hit.point - point).length() < 4e-4, "{label}: {hit:?}");
            }
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn low_level_mesh_distances_and_clipping_are_metric_for_scaled_directions() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (device, queue) = (engine.device().unwrap(), engine.queue().unwrap());
    let vertices = [
        Vec3::new(-10.0, -10.0, 0.0),
        Vec3::new(10.0, -10.0, 0.0),
        Vec3::new(0.0, 10.0, 0.0),
    ];
    let triangles = [[0, 1, 2]];
    let bvh = MeshBvh::build(&vertices, &triangles, 4);
    let mesh = MeshBvhGpu::upload(device, queue, &bvh, &vertices, &triangles).unwrap();
    let kernel = MeshBvhRaycastKernel::new(device);
    let directions: Vec<_> = [1e-30, 1.0, 1e30, 0.0]
        .map(|scale| Vec3::new(0.6, 0.0, -0.8) * scale)
        .into();
    let origin = buffer(&engine, vec![Vec3::new(0.0, 0.0, 3.0); directions.len()]);
    let direction = buffer(&engine, directions.clone());
    let (mut depth, mut tri, mut normal) =
        MeshBvhRaycastKernel::alloc_hits(device, queue, directions.len()).unwrap();
    for range in [4.0, 3.5, 4.0] {
        kernel
            .dispatch(
                device,
                queue,
                &mesh,
                &origin,
                &direction,
                &mut depth,
                &mut tri,
                &mut normal,
                range,
                directions.len() as u32,
            )
            .unwrap();
        depth.download(device, queue).unwrap();
        tri.download(device, queue).unwrap();
        normal.download(device, queue).unwrap();
        for (i, direction) in directions.iter().enumerate() {
            let label = format!("range={range}, direction={direction:?}");
            if i == 3 || range == 3.5 {
                assert_eq!(tri.host().unwrap()[i], -1, "{label}");
                assert_eq!(depth.host().unwrap()[i], f64::from(range), "{label}");
                assert_eq!(normal.host().unwrap()[i], Vec3::ZERO, "{label}");
            } else {
                let represented = direction.as_vec3().as_dvec3().normalize();
                let distance = -3.0 / represented.z;
                assert_eq!(tri.host().unwrap()[i], 0, "{label}");
                assert!(
                    (depth.host().unwrap()[i] - distance).abs() < 2e-6,
                    "{label}: {} vs {distance}",
                    depth.host().unwrap()[i]
                );
                assert!(
                    (normal.host().unwrap()[i] - Vec3::Z).length() < 2e-6,
                    "{label}"
                );
            }
        }
    }
}
