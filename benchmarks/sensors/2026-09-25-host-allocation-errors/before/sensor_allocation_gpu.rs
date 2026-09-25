#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Quat, Transform, Vec3};
use molla_sensors::{
    AnnotationConfig, AnnotationProcessor, CameraBatch, CameraSpec, ImageProcessor,
    ImageProcessorConfig, SensorCommandBatch, SensorFramePoolConfig, SensorScene, ShapeLabel,
    TiledCameraConfig,
};
use molla_sim::ModelBuilder;
use molla_storage::{Buffer, BufferElement, BufferUsage};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
    time::Duration,
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Counts {
    allocations: u64,
    reallocations: u64,
    requested_bytes: u64,
}

thread_local! {
    static ACTIVE: Cell<Option<Counts>> = const { Cell::new(None) };
}

struct CountingAllocator;

fn record(bytes: usize, reallocation: bool) {
    let _ = ACTIVE.try_with(|cell| {
        if let Some(mut counts) = cell.get() {
            counts.allocations += u64::from(!reallocation);
            counts.reallocations += u64::from(reallocation);
            counts.requested_bytes += bytes as u64;
            cell.set(Some(counts));
        }
    });
}

unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() {
            record(layout.size(), false);
        }
        result
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() {
            record(layout.size(), false);
        }
        result
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            record(size, true);
        }
        result
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

struct Recording;
impl Drop for Recording {
    fn drop(&mut self) {
        ACTIVE.with(|cell| cell.set(None));
    }
}

fn measure<T>(f: impl FnOnce() -> T) -> (T, Counts) {
    ACTIVE.with(|cell| {
        assert!(cell.get().is_none());
        cell.set(Some(Counts::default()));
    });
    let guard = Recording;
    let result = f();
    let counts = ACTIVE.with(|cell| cell.get().unwrap());
    drop(guard);
    (result, counts)
}

#[test]
fn allocation_counter_tracks_alloc_zeroed_realloc_and_resets_after_unwind() {
    let (_, counts) = measure(|| unsafe {
        let layout = Layout::from_size_align(128, 8).unwrap();
        let pointer = std::alloc::alloc_zeroed(layout);
        assert!(!pointer.is_null());
        let pointer = std::alloc::realloc(pointer, layout, 256);
        assert!(!pointer.is_null());
        std::alloc::dealloc(pointer, Layout::from_size_align(256, 8).unwrap());
    });
    assert_eq!(
        counts,
        Counts {
            allocations: 1,
            reallocations: 1,
            requested_bytes: 384
        }
    );
    let (_, counts) = measure(|| std::hint::black_box(Vec::<u64>::with_capacity(16)));
    assert_eq!(
        counts,
        Counts {
            allocations: 1,
            reallocations: 0,
            requested_bytes: 128
        }
    );
    assert!(std::panic::catch_unwind(|| measure(|| panic!("probe unwind"))).is_err());
    assert_eq!(measure(|| ()).1, Counts::default());
}

#[test]
fn mesh_bvh_construction_reserves_only_occupied_nodes() {
    use molla_geometry::{MeshBvh, MeshBvhNode};
    let vertices = [Vec3::ZERO, Vec3::X, Vec3::Z];
    let triangles = vec![[0, 1, 2]; 65_537];
    let (tree, counts) = measure(|| MeshBvh::build(&vertices, &triangles, 8));
    eprintln!(
        "triangles={} nodes={} node_capacity={} node_stride={} construction={counts:?}",
        triangles.len(),
        tree.nodes.len(),
        tree.nodes.capacity(),
        size_of::<MeshBvhNode>(),
    );
    assert_eq!(tree.nodes.capacity(), tree.nodes.len());
    assert_eq!(tree.triangle_order.len(), triangles.len());
    assert_eq!(counts.reallocations, 0);
    let expected = triangles.len() * (3 * size_of::<Vec3>() + size_of::<u32>())
        + tree.nodes.len() * size_of::<MeshBvhNode>();
    assert_eq!(counts.requested_bytes, expected as u64);
}

#[test]
#[ignore = "requires GPU"]
fn processor_construction_preflight_rejects_before_large_host_allocations() {
    use molla_sensors::{
        CloudGrid, CloudGridConfig, CloudProcessor, CloudProcessorConfig, ProcessorStorageLimits,
        ProcessorStorageUsage,
    };
    fn rejected<T>(label: &str, create: impl FnOnce() -> molla_core::Result<T>) {
        let (result, counts) = measure(create);
        let message = match result {
            Err(error) => error.to_string(),
            Ok(_) => panic!("{label}: under-budget constructor accepted"),
        };
        assert!(
            message.contains("processor construction requires"),
            "{label}: {message}"
        );
        assert!(counts.requested_bytes < 4096, "{label}: {counts:?}");
        eprintln!("{label}: rejected before payload allocation; {counts:?}");
    }
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap().limits();
    let image = ImageProcessorConfig {
        max_pixels: 65_537,
        max_images: 3,
        max_operations_per_image: 2,
        max_buffer_bytes: u64::MAX,
    };
    let usage = ProcessorStorageUsage::for_image(image, &device).unwrap();
    rejected("image", || {
        ImageProcessor::new_with_storage_limits(
            &engine,
            image,
            ProcessorStorageLimits {
                max_construction_bytes: usage.device_bytes,
            },
        )
    });
    let annotation = AnnotationConfig {
        max_pixels: 65_537,
        max_images: 3,
        max_labels: 5,
        max_buffer_bytes: u64::MAX,
    };
    let usage = ProcessorStorageUsage::for_annotation(annotation, &device).unwrap();
    rejected("annotation", || {
        AnnotationProcessor::new_with_storage_limits(
            &engine,
            annotation,
            ProcessorStorageLimits {
                max_construction_bytes: usage.device_bytes,
            },
        )
    });
    let cloud = CloudProcessorConfig {
        max_points: 65_537,
        max_jobs: 3,
        max_buffer_bytes: u64::MAX,
    };
    let usage = ProcessorStorageUsage::for_cloud(cloud, &device).unwrap();
    rejected("cloud", || {
        CloudProcessor::new_with_storage_limits(
            &engine,
            cloud,
            ProcessorStorageLimits {
                max_construction_bytes: usage.device_bytes,
            },
        )
    });
    let grid = CloudGridConfig {
        max_points: 65_537,
        max_cells: 17,
        max_buffer_bytes: u64::MAX,
    };
    let usage = ProcessorStorageUsage::for_grid(grid, &device).unwrap();
    rejected("grid", || {
        CloudGrid::new_with_storage_limits(
            &engine,
            grid,
            ProcessorStorageLimits {
                max_construction_bytes: usage.device_bytes,
            },
        )
    });
    engine.check_device().unwrap();
}

#[test]
#[ignore = "requires GPU to compile the scene"]
fn scene_identity_snapshot_clones_and_successful_lookups_do_not_allocate() {
    let engine = Engine::new_gpu().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(Default::default());
    builder
        .add_shape_sphere(body, 1.0, Transform::IDENTITY)
        .unwrap();
    let model = builder.build().unwrap();
    let mut scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let previous = scene.identities();
    scene.replace_model(&engine, &model).unwrap();
    let (_, counts) = measure(|| {
        for _ in 0..1024 {
            for snapshot in [scene.identities(), previous.clone()] {
                let stamp = snapshot.stamp();
                let shape = snapshot.shape_handle(stamp, 0).unwrap();
                let owner = snapshot.shape_body(stamp, 0).unwrap().unwrap();
                assert_eq!(snapshot.body_handle(stamp, body).unwrap(), Some(owner));
                assert_eq!(snapshot.body_index(owner).unwrap(), body);
                assert_eq!(snapshot.shape_index(shape).unwrap(), 0);
                std::hint::black_box(snapshot);
            }
        }
    });
    assert_eq!(counts, Counts::default());
}

#[test]
#[ignore = "requires GPU"]
fn scene_query_budget_precedes_model_clone_conversion_and_upload() {
    use molla_sim::{Heightfield, TriMesh};
    let engine = Engine::new_gpu().unwrap();
    let mut measured = Vec::new();
    for kind in 0..3 {
        let mut builder = ModelBuilder::new();
        match kind {
            0 => {
                for _ in 0..65_537 {
                    builder
                        .add_shape_sphere(BodyId::NONE, 1.0, Transform::IDENTITY)
                        .unwrap();
                }
            }
            1 => {
                let mesh = builder.add_mesh(TriMesh {
                    vertices: vec![Vec3::ZERO, Vec3::X, Vec3::Z],
                    indices: vec![[0, 1, 2]; 65_537],
                });
                builder
                    .add_shape_trimesh(BodyId::NONE, mesh, Transform::IDENTITY)
                    .unwrap();
            }
            _ => {
                let field = builder.add_heightfield(Heightfield {
                    nx: 257,
                    nz: 257,
                    cell_size: 1.0,
                    cell_size_z: 1.0,
                    origin: Vec3::ZERO,
                    heights: vec![0.0; 257 * 257],
                });
                builder
                    .add_shape_heightfield(BodyId::NONE, field, Transform::IDENTITY)
                    .unwrap();
            }
        }
        let model = builder.build().unwrap();
        let (result, counts) = measure(|| {
            SensorScene::from_model_with_query_budget(&engine, &model, Default::default(), 65_536)
        });
        let message = match result {
            Err(error) => error.to_string(),
            Ok(_) => panic!("kind={kind}: oversized scene accepted"),
        };
        assert!(
            message.contains("scene query exceeds allocation budget"),
            "{message}"
        );
        eprintln!("scene kind={kind}: {counts:?}");
        measured.push(counts);
    }
    assert!(
        measured.iter().all(|counts| counts.requested_bytes < 8192),
        "{measured:?}"
    );
}

#[test]
#[ignore = "requires GPU; reports calling-thread Rust allocations, not a zero-allocation gate"]
fn warmed_camera_processing_allocation_audit() {
    let engine = Engine::new_gpu().unwrap();
    eprintln!("allocation audit adapter: {:?}", engine.adapter_info());
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_sphere(BodyId::NONE, 1.0, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let state = model.state().unwrap();
    let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let specs = std::array::from_fn::<_, 4, _>(|i| CameraSpec {
        sensor_id: i as u64,
        config: TiledCameraConfig {
            width: 9,
            height: 7,
            ..Default::default()
        },
        pose: Transform::new(Vec3::new(i as f64 * 0.1, 0.0, 3.0), Quat::IDENTITY),
    });
    let mut cameras = CameraBatch::new(&engine, &specs, Default::default()).unwrap();
    let mut images = ImageProcessor::new(
        &engine,
        ImageProcessorConfig {
            max_pixels: 252,
            max_images: 4,
            max_operations_per_image: 1,
            max_buffer_bytes: 1 << 20,
        },
    )
    .unwrap();
    let mut annotations = AnnotationProcessor::new(
        &engine,
        AnnotationConfig {
            max_pixels: 252,
            max_images: 4,
            max_labels: 1,
            max_buffer_bytes: 1 << 20,
        },
    )
    .unwrap();
    annotations
        .set_labels(&[ShapeLabel {
            shape_id: 0,
            class_id: 7,
        }])
        .unwrap();
    let config = SensorFramePoolConfig {
        slots: 1,
        max_total_bytes: 1 << 20,
    };
    let mut cp = cameras.output_pool(&engine, config).unwrap();
    let mut ip = images.output_pool(&engine, config).unwrap();
    let mut ap = annotations.output_pool(&engine, config).unwrap();
    let mut samples = [[Counts::default(); 7]; 16];
    for frame in 0..24 {
        let (mut batch, create) = measure(|| SensorCommandBatch::new(&engine).unwrap());
        let (camera, capture) = measure(|| {
            cameras
                .encode_scene(
                    &mut batch,
                    &engine,
                    &scene,
                    &state,
                    &mut cp,
                    Some(frame as f64),
                )
                .unwrap()
                .unwrap()
        });
        let jobs = std::array::from_fn::<_, 4, _>(|i| camera.image_job(i as u64, &[]).unwrap());
        let labels = std::array::from_fn::<_, 4, _>(|i| camera.annotation_job(i as u64).unwrap());
        let (processed, image) = measure(|| {
            images
                .encode_pending(&mut batch, &jobs, &mut ip, Default::default())
                .unwrap()
                .unwrap()
        });
        let (annotated, annotation) = measure(|| {
            annotations
                .encode_pending(&mut batch, &labels, &mut ap, Default::default())
                .unwrap()
                .unwrap()
        });
        let (submitted, submit) = measure(|| batch.submit().unwrap());
        assert_eq!(submitted.stats.encoded_frames, 3);
        assert_eq!(submitted.stats.submissions, 1);
        assert!(submitted.frame(&processed.frame).is_ok());
        assert!(submitted.frame(&annotated.frame).is_ok());
        let (_, completion) = measure(|| {
            engine
                .device()
                .unwrap()
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(Duration::from_secs(30)),
                })
                .unwrap()
        });
        let (_, release) = measure(|| drop((submitted, processed, annotated, camera)));
        if frame >= 8 {
            samples[frame - 8] = [
                create, capture, image, annotation, submit, completion, release,
            ];
        }
    }
    for (index, phase) in [
        "create",
        "camera",
        "image",
        "annotation",
        "submit",
        "completion",
        "release",
    ]
    .iter()
    .enumerate()
    {
        let min_allocs = samples
            .iter()
            .map(|row| row[index].allocations)
            .min()
            .unwrap();
        let max_allocs = samples
            .iter()
            .map(|row| row[index].allocations)
            .max()
            .unwrap();
        let min_reallocs = samples
            .iter()
            .map(|row| row[index].reallocations)
            .min()
            .unwrap();
        let max_reallocs = samples
            .iter()
            .map(|row| row[index].reallocations)
            .max()
            .unwrap();
        let min_bytes = samples
            .iter()
            .map(|row| row[index].requested_bytes)
            .min()
            .unwrap();
        let max_bytes = samples
            .iter()
            .map(|row| row[index].requested_bytes)
            .max()
            .unwrap();
        eprintln!(
            "allocation audit {phase}: alloc={min_allocs}..{max_allocs} realloc={min_reallocs}..{max_reallocs} bytes={min_bytes}..{max_bytes}"
        );
    }
    assert_eq!(cp.group_stats(0).unwrap().buffer_allocations, 5);
    assert_eq!(ip.stats().buffer_allocations, 2);
    assert_eq!(ap.stats().buffer_allocations, 4);
    assert_eq!(cameras.group_stats(0).unwrap().trace_preparations, 1);
    assert_eq!(cp.available_slots().unwrap(), 1);
    assert_eq!(ip.available_slots().unwrap(), 1);
    assert_eq!(ap.available_slots().unwrap(), 1);
}

#[test]
#[ignore = "requires GPU"]
fn full_output_and_input_capacity_preserves_completion_leases_and_inline_iteration() {
    let engine = Engine::new_gpu().unwrap();
    let model = ModelBuilder::new().build().unwrap();
    let state = model.state().unwrap();
    let scene = SensorScene::from_model(&engine, &model, Default::default()).unwrap();
    let mut camera = molla_sensors::TiledCamera::new(
        &engine,
        TiledCameraConfig {
            width: 1,
            height: 1,
            channels: molla_sensors::CameraChannels::NONE,
            ..Default::default()
        },
        1,
    )
    .unwrap();
    let mut seed_pool = camera
        .output_pool(
            &engine,
            SensorFramePoolConfig {
                slots: 2,
                max_total_bytes: 64,
            },
        )
        .unwrap();
    let specs = [molla_sensors::SensorChannelSpec::of::<u32>(0)];
    let mut pool_a = molla_sensors::SensorFramePool::new(
        &engine,
        SensorFramePoolConfig {
            slots: 64,
            max_total_bytes: 256,
        },
        &specs,
    )
    .unwrap();
    let mut pool_b = molla_sensors::SensorFramePool::new(
        &engine,
        SensorFramePoolConfig {
            slots: 64,
            max_total_bytes: 256,
        },
        &specs,
    )
    .unwrap();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    let seed = camera
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &mut seed_pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    for _ in 0..63 {
        batch
            .encode_from_pending(
                &[&seed.frame],
                &mut pool_a,
                &specs,
                &[0..0],
                Default::default(),
                |_, _, _| Ok(0),
            )
            .unwrap()
            .unwrap();
    }
    let old = batch.submit().unwrap().into_frames();
    assert_eq!(old.len(), 64);
    let references: Vec<_> = old.iter().collect();
    let mut batch = SensorCommandBatch::new(&engine).unwrap();
    batch.encode_with_frames(&references, |_| Ok(())).unwrap();
    drop(references);
    drop(old);
    let seed = camera
        .encode_scene(
            &mut batch,
            &engine,
            &scene,
            &state,
            &mut seed_pool,
            Default::default(),
        )
        .unwrap()
        .unwrap();
    for _ in 0..63 {
        batch
            .encode_from_pending(
                &[&seed.frame],
                &mut pool_b,
                &specs,
                &[0..0],
                Default::default(),
                |_, _, _| Ok(0),
            )
            .unwrap()
            .unwrap();
    }
    assert!(
        batch
            .encode_from_pending(
                &[&seed.frame],
                &mut pool_b,
                &specs,
                &[0..0],
                Default::default(),
                |_, _, _| Ok(0)
            )
            .is_err()
    );
    let submitted = batch.submit().unwrap();
    assert_eq!(submitted.stats.encoded_frames, 64);
    let (mut frames, counts) = measure(|| submitted.into_frames_iter());
    assert_eq!(counts, Counts::default());
    assert_eq!(frames.len(), 64);
    let first = frames.next().unwrap();
    assert_eq!(frames.len(), 63);
    assert_eq!(first.channel_count(), 5);
    drop(frames);
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
    assert_eq!(pool_a.available_slots().unwrap(), 64);
    assert_eq!(pool_b.available_slots().unwrap(), 64);
    assert_eq!(seed_pool.available_slots().unwrap(), 1);
    drop(first);
    assert_eq!(seed_pool.available_slots().unwrap(), 2);
}

fn check_upload<T: BufferElement + std::fmt::Debug + PartialEq>(engine: &Engine, values: &[T]) {
    let mut scratch = Vec::with_capacity(values.len());
    let storage = scratch.as_ptr();
    let capacity = scratch.capacity();
    for n in [values.len(), values.len() / 2, 0, values.len()] {
        let mut buffer = Buffer::from_vec(values[..n].to_vec(), BufferUsage::Storage);
        buffer.ensure_device(engine.device().unwrap()).unwrap();
        buffer
            .upload_with_scratch(engine.queue().unwrap(), &mut scratch)
            .unwrap();
        assert_eq!(scratch.as_ptr(), storage);
        assert_eq!(scratch.capacity(), capacity);
        if n > 0 {
            assert!(buffer.device_buffer_for(engine.device().unwrap()).is_ok());
            buffer.mark_device_dirty();
            assert!(buffer.host().is_err());
            #[cfg(debug_assertions)]
            {
                let packed = bytemuck::cast_slice::<_, u8>(&scratch).to_vec();
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        buffer
                            .upload_with_scratch(engine.queue().unwrap(), &mut scratch)
                            .unwrap();
                    }))
                    .is_err()
                );
                assert_eq!(bytemuck::cast_slice::<_, u8>(&scratch), packed);
            }
            buffer
                .download(engine.device().unwrap(), engine.queue().unwrap())
                .unwrap();
            let expected: Vec<_> = values[..n]
                .iter()
                .map(|v| T::from_gpu(&v.to_gpu()))
                .collect();
            assert_eq!(buffer.host().unwrap(), expected);
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn retained_upload_scratch_preserves_conversion_capacity_and_authority() {
    let engine = Engine::new_gpu().unwrap();
    check_upload(&engine, &[0.1f64, -4.5, 2.0 / 3.0, 10000.25]);
    check_upload(&engine, &[Vec3::X, Vec3::new(0.1, -2.0 / 3.0, 9.3)]);
    check_upload(
        &engine,
        &[
            Transform::IDENTITY,
            Transform::new(Vec3::new(0.1, -2.0, 7.1), Quat::from_rotation_x(0.3)),
        ],
    );
    let mut absent = Buffer::from_vec(vec![1.0f64], BufferUsage::Storage);
    let mut scratch = vec![42.0f32];
    absent
        .upload_with_scratch(engine.queue().unwrap(), &mut scratch)
        .unwrap();
    assert_eq!(scratch, [42.0]);
    assert!(absent.device_buffer().is_none());
    assert!(absent.device_buffer_current().is_err());
}
