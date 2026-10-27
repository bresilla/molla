#![cfg(feature = "resource-counters")]
//! Warm sensor frames keep every GPU resource they created (wgpu internal
//! counters are flat across steady-state frames), and the heap allocations of a
//! warm frame are attributed to the crate that performs them.

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Quat, SpatialVector, Transform, Vec3};
use molla_sensors::readback::ReadbackStamp;
use molla_sensors::{
    AnnotationConfig, AnnotationProcessor, CameraBatch, CameraBatchOutputPool, CameraSpec,
    CloudFilter, CloudGrid, CloudGridConfig, CloudGridSpec, CloudProcessor, CloudProcessorConfig,
    CloudRaster, ContactSensor, FrameTarget, FrameTransformSensor, ImageProcessor,
    ImageProcessorConfig, ImuMount, ImuSensor, LidarBatch, LidarPattern, LidarScan,
    PendingCloudGridInput, PendingCloudInput, PendingCloudJob, PendingCloudSource, Raycaster,
    ScanMetadata, SensorCommandBatch, SensorFramePool, SensorFramePoolConfig, SensorScene,
    ShapeLabel, TiledCameraConfig,
};
use molla_sim::{BodyParams, Contacts, Model, ModelBuilder, State};
use molla_storage::{Buffer, BufferElement, BufferUsage};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    backtrace::Backtrace,
    cell::{Cell, RefCell},
    collections::BTreeMap,
    sync::Mutex,
    time::Duration,
};

static GPU: Mutex<()> = Mutex::new(());

thread_local! {
    static COUNT: Cell<Option<u64>> = const { Cell::new(None) };
    static TRACE: Cell<bool> = const { Cell::new(false) };
    static IN_TRACE: Cell<bool> = const { Cell::new(false) };
    static TRACES: RefCell<Vec<Backtrace>> = const { RefCell::new(Vec::new()) };
}

struct TracingAllocator;

fn note() {
    if IN_TRACE.try_with(Cell::get).unwrap_or(true) {
        return;
    }
    let _ = COUNT.try_with(|cell| {
        if let Some(n) = cell.get() {
            cell.set(Some(n + 1));
        }
    });
    if TRACE.try_with(Cell::get).unwrap_or(false) {
        let _ = IN_TRACE.try_with(|cell| cell.set(true));
        let trace = Backtrace::force_capture();
        let _ = TRACES.try_with(|traces| traces.borrow_mut().push(trace));
        let _ = IN_TRACE.try_with(|cell| cell.set(false));
    }
}

unsafe impl GlobalAlloc for TracingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc(layout) };
        if !result.is_null() {
            note();
        }
        result
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        let result = unsafe { System.alloc_zeroed(layout) };
        if !result.is_null() {
            note();
        }
        result
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        unsafe { System.dealloc(pointer, layout) };
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        let result = unsafe { System.realloc(pointer, layout, size) };
        if !result.is_null() {
            note();
        }
        result
    }
}

#[global_allocator]
static ALLOCATOR: TracingAllocator = TracingAllocator;

fn counted<T>(f: impl FnOnce() -> T) -> (T, u64) {
    COUNT.with(|cell| cell.set(Some(0)));
    let result = f();
    let n = COUNT.with(|cell| cell.replace(None)).unwrap();
    (result, n)
}

fn traced<T>(f: impl FnOnce() -> T) -> (T, Vec<Backtrace>) {
    TRACES.with(|traces| traces.borrow_mut().clear());
    TRACE.with(|cell| cell.set(true));
    let result = f();
    TRACE.with(|cell| cell.set(false));
    let traces = TRACES.with(|traces| std::mem::take(&mut *traces.borrow_mut()));
    (result, traces)
}

/// Crates whose frames only forward an allocation for their caller.
const PLUMBING: &[&str] = &[
    "std",
    "core",
    "alloc",
    "hashbrown",
    "smallvec",
    "arrayvec",
    "parking_lot",
    "parking_lot_core",
    "lock_api",
    "indexmap",
    "thin_vec",
    "bytemuck",
    "bit_vec",
    "allocator_api2",
];
const WGPU: &[&str] = &[
    "wgpu",
    "wgpu_core",
    "wgpu_hal",
    "wgpu_types",
    "naga",
    "ash",
    "gpu_allocator",
    "gpu_alloc",
    "gpu_descriptor",
    "profiling",
];

/// The crate of a frame; `<Generic as trait_crate::Trait>` frames belong to the
/// trait's crate because the implementing type is only a type parameter.
fn crate_of(symbol: &str) -> &str {
    let symbol = symbol.trim_start_matches('<');
    let head = symbol.split("::").next().unwrap_or(symbol);
    match head.split_once(" as ") {
        Some((ty, trait_crate)) if !ty.contains("::") => trait_crate,
        _ => head,
    }
}

/// `(symbol, file:line)` per frame, innermost first.
fn symbols(trace: &Backtrace) -> Vec<(String, String)> {
    let mut frames: Vec<(String, String)> = Vec::new();
    for line in trace.to_string().lines().map(str::trim) {
        if let Some(location) = line.strip_prefix("at ") {
            if let Some(last) = frames.last_mut() {
                let short = location
                    .rsplit_once("/crates/")
                    .map_or(location, |(_, p)| p);
                last.1 = short.to_string();
            }
        } else if let Some((index, symbol)) = line.split_once(": ")
            && index.parse::<usize>().is_ok()
        {
            frames.push((symbol.to_string(), String::new()));
        }
    }
    frames
}

/// `(owner, site)`: the first non-plumbing crate and, for wgpu-owned
/// allocations, the Molla or test function that called into wgpu.
fn attribute(trace: &Backtrace) -> (String, String) {
    let frames = symbols(trace);
    let own = |symbol: &str| {
        !symbol.contains("TracingAllocator")
            && !symbol.contains("resource_retention_gpu::note")
            && !symbol.contains("__rust")
            && !symbol.contains("__rdl")
            && !PLUMBING.contains(&crate_of(symbol))
    };
    let Some(first) = frames.iter().position(|(s, _)| own(s)) else {
        return ("unresolved".into(), String::new());
    };
    let owner = crate_of(&frames[first].0).to_string();
    let site = |(symbol, location): &(String, String)| format!("{symbol} [{location}]");
    if WGPU.contains(&owner.as_str()) {
        let caller = frames[first..]
            .iter()
            .find(|(s, _)| {
                let c = crate_of(s);
                c.starts_with("molla_") || c == "sensor_resource_retention_gpu"
            })
            .map(site)
            .unwrap_or_default();
        return ("wgpu".into(), caller);
    }
    (owner, site(&frames[first]))
}

fn config(slots: usize) -> SensorFramePoolConfig {
    SensorFramePoolConfig {
        slots,
        max_total_bytes: 64 << 20,
    }
}

fn drain(engine: &Engine) {
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(Duration::from_secs(30)),
        })
        .unwrap();
}

fn stamp(sample: u64) -> ReadbackStamp {
    ReadbackStamp {
        sensor_id: 1,
        sample_id: sample,
        scene_revision: 1,
        simulation_time: Some(sample as f64 * 0.01),
    }
}

/// Live device objects by kind, after all submitted work completed.
fn resources(engine: &Engine) -> BTreeMap<&'static str, isize> {
    drain(engine);
    let hal = engine.device().unwrap().get_internal_counters().hal;
    BTreeMap::from([
        ("buffers", hal.buffers.read()),
        ("buffer_memory", hal.buffer_memory.read()),
        ("textures", hal.textures.read()),
        ("texture_views", hal.texture_views.read()),
        ("bind_groups", hal.bind_groups.read()),
        ("bind_group_layouts", hal.bind_group_layouts.read()),
        ("compute_pipelines", hal.compute_pipelines.read()),
        ("pipeline_layouts", hal.pipeline_layouts.read()),
        ("shader_modules", hal.shader_modules.read()),
        ("query_sets", hal.query_sets.read()),
        ("samplers", hal.samplers.read()),
        ("command_encoders", hal.command_encoders.read()),
    ])
}

fn uploaded(engine: &Engine, builder: ModelBuilder) -> (Model, State) {
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

struct CameraChain {
    cameras: CameraBatch,
    images: ImageProcessor,
    annotations: AnnotationProcessor,
    cp: CameraBatchOutputPool,
    ip: SensorFramePool,
    ap: SensorFramePool,
}

impl CameraChain {
    fn new(engine: &Engine) -> Self {
        let specs = std::array::from_fn::<_, 4, _>(|i| CameraSpec {
            sensor_id: i as u64,
            config: TiledCameraConfig {
                width: 9,
                height: 7,
                ..Default::default()
            },
            pose: Transform::new(Vec3::new(i as f64 * 0.1, 0.0, 3.0), Quat::IDENTITY),
        });
        let cameras = CameraBatch::new(engine, &specs, Default::default()).unwrap();
        let images = ImageProcessor::new(
            engine,
            ImageProcessorConfig {
                max_pixels: 252,
                max_images: 4,
                max_operations_per_image: 1,
                max_buffer_bytes: 1 << 20,
            },
        )
        .unwrap();
        let mut annotations = AnnotationProcessor::new(
            engine,
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
        let cp = cameras.output_pool(engine, config(1)).unwrap();
        let ip = images.output_pool(engine, config(1)).unwrap();
        let ap = annotations.output_pool(engine, config(1)).unwrap();
        Self {
            cameras,
            images,
            annotations,
            cp,
            ip,
            ap,
        }
    }

    fn frame(&mut self, engine: &Engine, scene: &SensorScene, state: &State, frame: usize) {
        let mut batch = SensorCommandBatch::new(engine).unwrap();
        let camera = self
            .cameras
            .encode_scene(
                &mut batch,
                engine,
                scene,
                state,
                &mut self.cp,
                Some(frame as f64),
            )
            .unwrap()
            .unwrap();
        let jobs = std::array::from_fn::<_, 4, _>(|i| camera.image_job(i as u64, &[]).unwrap());
        let labels = std::array::from_fn::<_, 4, _>(|i| camera.annotation_job(i as u64).unwrap());
        let processed = self
            .images
            .encode_pending(&mut batch, &jobs, &mut self.ip, Default::default())
            .unwrap()
            .unwrap();
        let annotated = self
            .annotations
            .encode_pending(&mut batch, &labels, &mut self.ap, Default::default())
            .unwrap()
            .unwrap();
        let submitted = batch.submit().unwrap();
        assert_eq!(submitted.stats.submissions, 1);
        drain(engine);
        drop((submitted, processed, annotated, camera));
    }
}

fn sphere_scene(engine: &Engine) -> (Model, State, SensorScene) {
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_sphere(BodyId::NONE, 1.0, Transform::IDENTITY)
        .unwrap();
    let (model, state) = uploaded(engine, builder);
    let scene = SensorScene::from_model(engine, &model, Default::default()).unwrap();
    (model, state, scene)
}

const RAYS: usize = 16;

struct RayChain {
    rays: Raycaster,
    lidar: LidarBatch,
    cloud: CloudProcessor,
    grid: CloudGrid,
    pr: SensorFramePool,
    pl: SensorFramePool,
    pc: SensorFramePool,
    pg: SensorFramePool,
    origins: Vec<Vec3>,
    directions: Vec<Vec3>,
}

impl RayChain {
    fn new(engine: &Engine) -> Self {
        let rays = Raycaster::new(engine, RAYS as u32).unwrap();
        let lidar = LidarBatch::new(engine, 2, 512).unwrap();
        let cloud = CloudProcessor::new(
            engine,
            CloudProcessorConfig {
                max_points: RAYS as u32,
                max_jobs: 1,
                max_buffer_bytes: 64 << 20,
            },
        )
        .unwrap();
        let grid = CloudGrid::new(
            engine,
            CloudGridConfig {
                max_points: RAYS as u32,
                max_cells: 64,
                max_buffer_bytes: 64 << 20,
            },
        )
        .unwrap();
        let pr = rays.output_pool(engine, config(1)).unwrap();
        let pl = lidar.output_pool(engine, config(1)).unwrap();
        let pc = cloud.output_pool(engine, config(1)).unwrap();
        let pg = grid.output_pool(engine, config(1)).unwrap();
        let origins = (0..RAYS)
            .map(|i| Vec3::new(i as f64 * 0.1 - 0.8, 5.0, 0.0))
            .collect();
        Self {
            rays,
            lidar,
            cloud,
            grid,
            pr,
            pl,
            pc,
            pg,
            origins,
            directions: vec![Vec3::NEG_Y; RAYS],
        }
    }

    fn frame(&mut self, engine: &Engine, scene: &SensorScene, state: &State, frame: usize) {
        let mut batch = SensorCommandBatch::new(engine).unwrap();
        let r = self
            .rays
            .encode_scene(
                &mut batch,
                engine,
                scene,
                state,
                &self.origins,
                &self.directions,
                20.0,
                &mut self.pr,
                stamp(frame as u64),
            )
            .unwrap()
            .unwrap();
        let mut scan = ScanMetadata::new(Vec3::new(0.0, 4.0, 0.0), 8, 32);
        scan.forward = Vec3::NEG_Y;
        scan.up = Vec3::Z;
        let scans = [
            LidarScan {
                sensor_id: 2,
                pattern: LidarPattern::Terrestrial(scan),
            },
            LidarScan {
                sensor_id: 3,
                pattern: LidarPattern::Terrestrial(scan),
            },
        ];
        let l = self
            .lidar
            .encode_scene(
                &mut batch,
                engine,
                scene,
                state,
                &scans,
                &mut self.pl,
                stamp(frame as u64),
            )
            .unwrap()
            .unwrap();
        let jobs = [PendingCloudJob {
            input: PendingCloudInput::Rays {
                channels: [4, 5, 1, 2, 3].map(|channel| PendingCloudSource {
                    frame: &r.frame,
                    channel,
                    elements: 0..RAYS,
                }),
                raster: CloudRaster {
                    sensor_id: 1,
                    rows: 1,
                    columns: RAYS as u32,
                    subrays: 1,
                    source_offset: 0,
                    angles: None,
                },
            },
            transform: Transform::IDENTITY,
            filter: CloudFilter::default(),
            stamp: stamp(frame as u64),
        }];
        let c = self
            .cloud
            .encode_pending(&mut batch, &jobs, &mut self.pc, stamp(frame as u64))
            .unwrap()
            .unwrap();
        let g = self
            .grid
            .encode_pending(
                &mut batch,
                PendingCloudGridInput {
                    frame: &c.frame,
                    channel: 0,
                    points: 0..RAYS,
                },
                CloudGridSpec {
                    minimum: [-2.0, -2.0, -2.0],
                    maximum: [2.0, 2.0, 2.0],
                    dimensions: [4, 4, 4],
                },
                &mut self.pg,
                stamp(frame as u64),
            )
            .unwrap()
            .unwrap();
        let submitted = batch.submit().unwrap();
        assert_eq!(submitted.stats.submissions, 1);
        drain(engine);
        drop((submitted, r, l, c, g));
    }
}

const MOUNTS: usize = 8;

struct AuxChain {
    imu: ImuSensor,
    contact: ContactSensor,
    frames: FrameTransformSensor,
    pi: SensorFramePool,
    pc: SensorFramePool,
    pf: SensorFramePool,
}

fn aux_model(engine: &Engine) -> (Model, State, Contacts, Vec<ImuMount>) {
    let mut builder = ModelBuilder::new();
    let mounts: Vec<_> = (0..MOUNTS)
        .map(|i| ImuMount {
            body: builder.add_body(
                BodyParams::new().with_pose(Transform::new(Vec3::X * i as f64, Quat::IDENTITY)),
            ),
            local_pose: Transform::new(Vec3::Z * 0.1, Quat::IDENTITY),
        })
        .collect();
    let (model, mut state) = uploaded(engine, builder);
    let velocities: Vec<_> = (0..MOUNTS)
        .map(|_| SpatialVector::new(Vec3::X, Vec3::Y * 0.5).to_gpu())
        .collect();
    engine.queue().unwrap().write_buffer(
        state.body_qd.device_buffer().unwrap(),
        0,
        bytemuck::cast_slice(&velocities),
    );
    state.body_qd.mark_device_dirty();
    let mut contacts = Contacts::with_capacity(MOUNTS - 1);
    contacts.count = (MOUNTS - 1) as u32;
    for i in 0..MOUNTS - 1 {
        contacts.a_index.host_mut().unwrap()[i] = i as u32;
        contacts.b_index.host_mut().unwrap()[i] = (i + 1) as u32;
        contacts.normal.host_mut().unwrap()[i] = Vec3::Y;
        contacts.depth.host_mut().unwrap()[i] = 0.001;
        contacts.point.host_mut().unwrap()[i] = Vec3::X * i as f64;
    }
    let mut offsets = vec![0];
    let mut indices = Vec::new();
    for i in 0..MOUNTS {
        if i > 0 {
            indices.push((i - 1) as u32 | Contacts::B_SIDE_BIT);
        }
        if i < MOUNTS - 1 {
            indices.push(i as u32);
        }
        offsets.push(indices.len() as u32);
    }
    contacts.body_contact_offsets = Buffer::from_vec(offsets, BufferUsage::StorageReadWrite);
    contacts.body_contact_indices = Buffer::from_vec(indices, BufferUsage::StorageReadWrite);
    contacts
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    (model, state, contacts, mounts)
}

impl AuxChain {
    fn new(engine: &Engine, model: &Model, mounts: &[ImuMount]) -> Self {
        let imu = ImuSensor::new(engine, model, mounts).unwrap();
        let contact = ContactSensor::new(engine, model).unwrap();
        let targets: Vec<_> = mounts
            .iter()
            .map(|mount| FrameTarget {
                body: mount.body,
                local_pose: mount.local_pose,
            })
            .collect();
        let frames = FrameTransformSensor::new(engine, model, None, &targets).unwrap();
        let pi = imu.output_pool(engine, config(1)).unwrap();
        let pc = contact.output_pool(engine, config(1)).unwrap();
        let pf = frames.output_pool(engine, config(1)).unwrap();
        Self {
            imu,
            contact,
            frames,
            pi,
            pc,
            pf,
        }
    }

    fn frame(
        &mut self,
        engine: &Engine,
        model: &Model,
        state: &State,
        contacts: &Contacts,
        frame: usize,
    ) {
        let mut batch = SensorCommandBatch::new(engine).unwrap();
        let i = self
            .imu
            .encode(
                &mut batch,
                engine,
                model,
                state,
                0.01,
                &mut self.pi,
                stamp(frame as u64),
            )
            .unwrap()
            .unwrap();
        let c = self
            .contact
            .encode(
                &mut batch,
                engine,
                contacts,
                &mut self.pc,
                stamp(frame as u64),
            )
            .unwrap()
            .unwrap();
        let f = self
            .frames
            .encode(&mut batch, engine, state, &mut self.pf, stamp(frame as u64))
            .unwrap()
            .unwrap();
        let submitted = batch.submit().unwrap();
        assert_eq!(submitted.stats.submissions, 1);
        drain(engine);
        drop((submitted, i, c, f));
    }
}

fn assert_flat(label: &str, warm: &BTreeMap<&str, isize>, later: &BTreeMap<&str, isize>) {
    eprintln!("{label} warm resources: {warm:?}");
    assert!(warm["buffers"] > 0, "{label}: wgpu counters are disabled");
    assert_eq!(
        warm, later,
        "{label}: steady-state frames changed live GPU objects"
    );
}

#[test]
#[ignore = "requires GPU"]
fn warm_sensor_frames_create_no_gpu_resources() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    eprintln!("adapter: {:?}", engine.adapter_info());

    let (_model, state, scene) = sphere_scene(&engine);
    let mut cameras = CameraChain::new(&engine);
    for frame in 0..4 {
        cameras.frame(&engine, &scene, &state, frame);
    }
    let warm = resources(&engine);
    for frame in 4..36 {
        cameras.frame(&engine, &scene, &state, frame);
    }
    assert_flat("camera/image/annotation", &warm, &resources(&engine));

    let mut rays = RayChain::new(&engine);
    for frame in 0..4 {
        rays.frame(&engine, &scene, &state, frame);
    }
    let warm = resources(&engine);
    for frame in 4..36 {
        rays.frame(&engine, &scene, &state, frame);
    }
    assert_flat("ray/lidar/cloud/grid", &warm, &resources(&engine));

    let (model, state, contacts, mounts) = aux_model(&engine);
    let mut aux = AuxChain::new(&engine, &model, &mounts);
    for frame in 0..4 {
        aux.frame(&engine, &model, &state, &contacts, frame);
    }
    let warm = resources(&engine);
    for frame in 4..36 {
        aux.frame(&engine, &model, &state, &contacts, frame);
    }
    assert_flat("imu/contact/frame", &warm, &resources(&engine));
}

fn report(label: &str, per_frame: &[u64], traces: &[Backtrace]) -> BTreeMap<String, usize> {
    let mut owners: BTreeMap<String, usize> = BTreeMap::new();
    let mut sites: BTreeMap<(String, String), usize> = BTreeMap::new();
    for trace in traces {
        let (owner, site) = attribute(trace);
        *owners.entry(owner.clone()).or_default() += 1;
        *sites.entry((owner, site)).or_default() += 1;
    }
    eprintln!(
        "{label}: per-frame allocations {}..{}, traced frame {} by owner {owners:?}",
        per_frame.iter().min().unwrap(),
        per_frame.iter().max().unwrap(),
        traces.len()
    );
    let mut ranked: Vec<_> = sites.into_iter().collect();
    ranked.sort_by(|a, b| b.1.cmp(&a.1));
    for ((owner, site), n) in ranked {
        eprintln!("  {n:4} {owner:<12} {site}");
    }
    owners
}

#[test]
#[ignore = "requires GPU; prints the attribution of warm-frame heap allocations"]
fn warm_sensor_frame_allocations_are_attributed() {
    let _guard = GPU.lock().unwrap_or_else(|e| e.into_inner());
    let engine = Engine::new_gpu().unwrap();
    let (_model, state, scene) = sphere_scene(&engine);
    let mut cameras = CameraChain::new(&engine);
    let mut rays = RayChain::new(&engine);
    let (model, aux_state, contacts, mounts) = aux_model(&engine);
    let mut aux = AuxChain::new(&engine, &model, &mounts);
    for frame in 0..8 {
        cameras.frame(&engine, &scene, &state, frame);
        rays.frame(&engine, &scene, &state, frame);
        aux.frame(&engine, &model, &aux_state, &contacts, frame);
    }
    let per = |f: &mut dyn FnMut(usize)| {
        (8..24)
            .map(|frame| counted(|| f(frame)).1)
            .collect::<Vec<_>>()
    };
    let camera_counts = per(&mut |frame| cameras.frame(&engine, &scene, &state, frame));
    let ray_counts = per(&mut |frame| rays.frame(&engine, &scene, &state, frame));
    let aux_counts = per(&mut |frame| aux.frame(&engine, &model, &aux_state, &contacts, frame));
    let ((), camera_traces) = traced(|| cameras.frame(&engine, &scene, &state, 24));
    let ((), ray_traces) = traced(|| rays.frame(&engine, &scene, &state, 24));
    let ((), aux_traces) = traced(|| aux.frame(&engine, &model, &aux_state, &contacts, 24));
    for (label, counts, traces) in [
        ("camera/image/annotation", &camera_counts, &camera_traces),
        ("ray/lidar/cloud/grid", &ray_counts, &ray_traces),
        ("imu/contact/frame", &aux_counts, &aux_traces),
    ] {
        let owners = report(label, counts, traces);
        assert!(
            !owners.contains_key("unresolved"),
            "{label}: unresolved frames"
        );
    }
}
