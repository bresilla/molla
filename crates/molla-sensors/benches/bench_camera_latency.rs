//! Completed camera and camera/image/annotation chains with per-frame latency records.

#[path = "support/camera_latency_smoke.rs"]
mod camera_latency_smoke;
#[path = "support/metrics.rs"]
mod metrics;
#[path = "support/run_mode.rs"]
mod run_mode;
#[path = "support/timing.rs"]
mod timing;

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{GpuTransform, Quat, Real, Transform, Vec3};
use molla_sensors::{
    AnnotationConfig, AnnotationProcessor, CameraBatch, CameraBatchLimits, CameraBatchOutputPool,
    CameraChannels, CameraSpec, DeviceBoundingBox, ImageOp, ImageProcessor, ImageProcessorConfig,
    PhysicalOptics, RaycasterLimits, SensorCommandBatch, SensorFrame, SensorFramePool,
    SensorFramePoolConfig, SensorScene, ShapeLabel, TiledCameraConfig, TiledCameraLimits,
    readback::{ReadbackConfig, ReadbackRing, ReadbackSource},
};
use molla_sim::{Model, ModelBuilder, State, TriMesh};
use molla_storage::BufferElement;
use std::{
    io::{BufWriter, Write},
    time::{Duration, Instant},
};

#[global_allocator]
static ALLOCATOR: metrics::CountingAllocator = metrics::CountingAllocator;

const OPS: [ImageOp; 4] = [
    ImageOp::Exposure(0.2),
    ImageOp::GrayWorld,
    ImageOp::Reinhard,
    ImageOp::Gamma(2.2),
];

#[derive(Clone, Copy)]
struct Case {
    width: u32,
    height: u32,
    count: u32,
    aperture: bool,
    heterogeneous: bool,
    moving: bool,
    chain: bool,
}

impl Case {
    fn id(self, host: bool) -> String {
        format!(
            "{}x{}_c{}_{}_{}_{}_{}_{}",
            self.width,
            self.height,
            self.count,
            if self.aperture {
                "aperture4"
            } else {
                "pinhole"
            },
            if self.heterogeneous {
                "heterogeneous"
            } else {
                "homogeneous"
            },
            if self.moving { "moving" } else { "warm" },
            if self.chain { "chain" } else { "camera" },
            if host { "host" } else { "device" }
        )
    }
    fn pixels(self) -> u32 {
        self.width * self.height * self.count
    }
}

fn cases(full: bool) -> Vec<Case> {
    let mut result = Vec::new();
    for (width, height) in [(640, 480), (1280, 720)] {
        for count in [1, 4] {
            for aperture in [false, true] {
                for chain in [false, true] {
                    let case = Case {
                        width,
                        height,
                        count,
                        aperture,
                        heterogeneous: false,
                        moving: false,
                        chain,
                    };
                    result.push(case);
                    if full && count == 4 {
                        result.push(Case {
                            heterogeneous: true,
                            ..case
                        });
                    }
                    if full && count == 4 && width == 640 {
                        result.push(Case {
                            moving: true,
                            ..case
                        });
                    }
                }
            }
        }
    }
    if !full {
        result.retain(|c| {
            c.width == 640 && c.count == 1 && !c.aperture
                || c.width == 1280 && c.count == 4 && c.aperture && c.chain
        });
    }
    result
}

fn fixture(engine: &Engine) -> (Model, State, SensorScene) {
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(
            BodyId::NONE,
            Vec3::Z,
            Transform::from_translation(Vec3::NEG_Z * 12.0),
        )
        .unwrap();
    let body = builder.add_body(Default::default());
    let mut mesh = TriMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for y in 0..=128 {
        for x in 0..=128 {
            let x = x as f64 / 16.0 - 4.0;
            let y = y as f64 / 16.0 - 4.0;
            mesh.vertices
                .push(Vec3::new(x, y, -5.0 + 0.2 * x.sin() * y.cos()));
        }
    }
    for y in 0..128 {
        for x in 0..128 {
            let i = y * 129 + x;
            mesh.indices
                .extend([[i, i + 1, i + 129], [i + 1, i + 130, i + 129]]);
        }
    }
    let mesh = builder.add_mesh(mesh);
    builder
        .add_shape_trimesh(body, mesh, Transform::IDENTITY)
        .unwrap();
    for i in 0..100 {
        builder
            .add_shape_sphere(
                BodyId::NONE,
                0.18,
                Transform::from_translation(Vec3::new(
                    (i % 10) as f64 * 0.7 - 3.2,
                    (i / 10) as f64 * 0.7 - 3.2,
                    -3.8,
                )),
            )
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    for (i, color) in model.shape_color.host_mut().unwrap().iter_mut().enumerate() {
        *color = Vec3::new(0.2 + (i % 5) as f64 * 0.15, 0.3 + (i % 3) as f64 * 0.2, 0.6);
    }
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let scene = SensorScene::from_model(engine, &model, Default::default()).unwrap();
    (model, state, scene)
}

struct Chain {
    images: ImageProcessor,
    annotations: AnnotationProcessor,
    image_pool: SensorFramePool,
    annotation_pool: SensorFramePool,
}

impl Chain {
    fn new(engine: &Engine, case: Case, shapes: usize) -> Self {
        let images = ImageProcessor::new(
            engine,
            ImageProcessorConfig {
                max_pixels: case.pixels(),
                max_images: case.count,
                max_operations_per_image: OPS.len() as u32,
                max_buffer_bytes: 128 << 20,
            },
        )
        .unwrap();
        let mut annotations = AnnotationProcessor::new(
            engine,
            AnnotationConfig {
                max_pixels: case.pixels(),
                max_images: case.count,
                max_labels: shapes as u32,
                max_buffer_bytes: 128 << 20,
            },
        )
        .unwrap();
        let labels: Vec<_> = (0..shapes)
            .map(|shape| ShapeLabel {
                shape_id: shape as u32,
                class_id: (shape % 8) as u32 + 1,
            })
            .collect();
        annotations.set_labels(&labels).unwrap();
        let config = SensorFramePoolConfig {
            slots: 1,
            max_total_bytes: 256 << 20,
        };
        let image_pool = images.output_pool(engine, config).unwrap();
        let annotation_pool = annotations.output_pool(engine, config).unwrap();
        Self {
            images,
            annotations,
            image_pool,
            annotation_pool,
        }
    }
}

fn source<T: BufferElement>(frame: &SensorFrame, channel: usize) -> ReadbackSource<'_> {
    let view = frame.channel::<T>(channel).unwrap();
    view.readback_source(0..view.len()).unwrap()
}

struct Sample {
    encode_us: f64,
    submit_us: f64,
    complete_us: f64,
    uploads: u64,
    downloads: u64,
    submissions: u64,
    waits: u64,
    dispatches: u64,
}

#[allow(clippy::too_many_arguments)]
fn frame(
    engine: &Engine,
    camera: &mut CameraBatch,
    pool: &mut CameraBatchOutputPool,
    scene: &SensorScene,
    state: &mut State,
    chain: &mut Option<Chain>,
    ring: &mut ReadbackRing,
    timer: Option<&timing::Timer>,
    case: Case,
    index: usize,
    host: bool,
) -> Sample {
    let start = Instant::now();
    let pose_bytes = if case.moving {
        let pose: GpuTransform = Transform::new(
            Vec3::X * (index as f64 * 0.13).sin() * 0.2,
            Quat::from_rotation_y((index as f64 * 0.03).sin() * 0.05),
        )
        .into();
        engine.queue().unwrap().write_buffer(
            state.body_q.device_buffer().unwrap(),
            0,
            bytemuck::bytes_of(&pose),
        );
        state.body_q.mark_device_dirty();
        32
    } else {
        0
    };
    let before_read = ring.stats();
    let mut batch = SensorCommandBatch::new(engine).unwrap();
    if let Some(timer) = timer {
        timer.begin(&mut batch);
    }
    let output = camera
        .encode_scene(&mut batch, engine, scene, state, pool, None)
        .unwrap()
        .expect("completed camera slots");
    let processed = if let Some(chain) = chain {
        let image_jobs: Vec<_> = (0..case.count)
            .map(|id| output.image_job(id as u64, &OPS).unwrap())
            .collect();
        let annotation_jobs: Vec<_> = (0..case.count)
            .map(|id| output.annotation_job(id as u64).unwrap())
            .collect();
        let images = chain
            .images
            .encode_pending(
                &mut batch,
                &image_jobs,
                &mut chain.image_pool,
                Default::default(),
            )
            .unwrap()
            .unwrap();
        let annotations = chain
            .annotations
            .encode_pending(
                &mut batch,
                &annotation_jobs,
                &mut chain.annotation_pool,
                Default::default(),
            )
            .unwrap()
            .unwrap();
        Some((images, annotations))
    } else {
        None
    };
    if let Some(timer) = timer {
        timer.end(&mut batch);
    }
    let encode_us = start.elapsed().as_secs_f64() * 1e6;
    let submitted = batch.submit().unwrap();
    let submit_us = start.elapsed().as_secs_f64() * 1e6;
    let mut waits = 0;
    if host {
        let mut sources = Vec::new();
        if let Some((images, annotations)) = &processed {
            let image = submitted.frame(&images.frame).unwrap();
            let labels = submitted.frame(&annotations.frame).unwrap();
            sources.extend([
                source::<u32>(image, 0),
                source::<u32>(labels, 0),
                source::<u32>(labels, 1),
                source::<DeviceBoundingBox>(labels, 2),
                source::<u32>(labels, 3),
            ]);
        } else {
            for group in &output.groups {
                let frame = submitted.frame(&group.frame).unwrap();
                sources.extend([
                    source::<Real>(frame, 0),
                    source::<u32>(frame, 1),
                    source::<Vec3>(frame, 2),
                    source::<u32>(frame, 3),
                    source::<u32>(frame, 4),
                ]);
            }
        }
        let ticket = ring
            .try_enqueue(&sources, Default::default())
            .unwrap()
            .unwrap();
        let collected = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
        if processed.is_some() {
            let counts = collected.channel::<u32>(4).unwrap();
            for count in counts.chunks_exact(4) {
                assert_eq!(count[0], case.width * case.height);
                assert_eq!(&count[1..3], &[0, 0]);
                assert!(count[3] > 0);
            }
            assert!(
                collected
                    .channel_iter::<u32>(2)
                    .unwrap()
                    .all(|class| (1..=8).contains(&class))
            );
        } else {
            for group in 0..output.groups.len() {
                assert!(
                    collected
                        .channel_iter::<Real>(group * 5)
                        .unwrap()
                        .all(|v| v.is_finite() && v > 0.0)
                );
                assert!(
                    collected
                        .channel_iter::<u32>(group * 5 + 4)
                        .unwrap()
                        .all(|v| v < 102)
                );
            }
        }
        std::hint::black_box(collected);
    } else {
        timing::wait(engine);
        waits = 1;
    }
    let after_read = ring.stats();
    let uploads = submitted.stats.input_upload_bytes + pose_bytes;
    let submissions =
        submitted.stats.submissions + after_read.submissions - before_read.submissions;
    let dispatches = submitted.stats.kernel_dispatches;
    drop(submitted);
    drop(processed);
    drop(output);
    Sample {
        encode_us,
        submit_us,
        complete_us: start.elapsed().as_secs_f64() * 1e6,
        uploads,
        downloads: after_read.copied_bytes - before_read.copied_bytes,
        submissions,
        waits: waits + after_read.blocking_waits - before_read.blocking_waits,
        dispatches,
    }
}

fn rss() -> String {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .filter(|line| line.starts_with("VmRSS:") || line.starts_with("VmHWM:"))
        .collect::<Vec<_>>()
        .join("; ")
}

fn run(
    engine: &Engine,
    case: Case,
    host: bool,
    samples: usize,
    warmup: usize,
    repeat: usize,
    output: &mut impl Write,
) {
    let id = case.id(host);
    let build = Instant::now();
    let (model, mut state, scene) = fixture(engine);
    let specs: Vec<_> = (0..case.count)
        .map(|i| CameraSpec {
            sensor_id: i as u64,
            pose: Transform::from_translation(
                Vec3::X * (i as f64 - (case.count - 1) as f64 * 0.5) * 0.3,
            ),
            config: TiledCameraConfig {
                width: case.width,
                height: case.height,
                channels: CameraChannels::ALL,
                max_distance: 30.0,
                fov_y_radians: if case.heterogeneous && i % 2 == 1 {
                    0.9
                } else {
                    1.05
                },
                optics: PhysicalOptics {
                    lens_diameter_m: if case.aperture { 0.04 } else { 0.0 },
                    aperture_samples: if case.aperture { 4 } else { 1 },
                    focal_distance_m: 5.0,
                    ..Default::default()
                },
                ..Default::default()
            },
        })
        .collect();
    let mut camera = CameraBatch::new_with_storage_limits(
        engine,
        &specs,
        CameraBatchLimits::default(),
        TiledCameraLimits {
            max_storage_bytes: 2 << 30,
            ray_limits: RaycasterLimits {
                max_ray_storage_bytes: 2 << 30,
            },
        },
    )
    .unwrap();
    let mut pool = camera
        .output_pool(
            engine,
            SensorFramePoolConfig {
                slots: 1,
                max_total_bytes: 256 << 20,
            },
        )
        .unwrap();
    let mut chain = case
        .chain
        .then(|| Chain::new(engine, case, model.shape_count));
    let bytes = if case.chain {
        u64::from(case.pixels()) * 12 + u64::from(case.count) * (model.shape_count as u64 * 32 + 16)
    } else {
        u64::from(case.pixels()) * 32
    };
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: bytes,
            max_total_bytes: bytes,
        },
    )
    .unwrap();
    let mut timer = timing::Timer::new(engine);
    engine.queue().unwrap().submit([]);
    timing::wait(engine);
    eprintln!(
        "case={id} repeat={repeat} pixels={} setup_ms={:.3} groups={} camera_arrays_bytes={} scene_atlas_bytes={} output_camera_pool_bytes={} staging_bytes={bytes} chain_buffers={:?} timer={} {}",
        case.pixels(),
        build.elapsed().as_secs_f64() * 1e3,
        camera.group_count(),
        camera.storage_bytes(),
        scene.query_device_bytes(),
        u64::from(case.pixels()) * 32,
        chain.as_ref().map(|c| (
            c.images.stats().buffer_bytes,
            c.annotations.stats().buffer_bytes
        )),
        timer.is_some(),
        rss()
    );
    let mut complete = Vec::with_capacity(samples);
    let mut encode = Vec::with_capacity(samples);
    let mut gpu = Vec::with_capacity(samples);
    for index in 0..samples + warmup + 1 {
        let phase = if index == 0 {
            "first"
        } else if index <= warmup {
            "warmup"
        } else {
            "sample"
        };
        let (sample, counts) = metrics::measure(|| {
            frame(
                engine,
                &mut camera,
                &mut pool,
                &scene,
                &mut state,
                &mut chain,
                &mut ring,
                timer.as_ref(),
                case,
                index,
                host,
            )
        });
        let gpu_us = timer.as_mut().map(timing::Timer::collect_us);
        if phase == "sample" {
            complete.push(sample.complete_us);
            encode.push(sample.encode_us);
            if let Some(us) = gpu_us {
                gpu.push(us);
            }
        }
        writeln!(output, "{id}\t{repeat}\t{phase}\t{index}\t{}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            case.pixels(), sample.encode_us, sample.submit_us, sample.complete_us, gpu_us.map_or_else(|| "NA".into(), |v| format!("{v:.3}")),
            sample.uploads, sample.downloads, sample.submissions, sample.waits, counts.allocations, counts.reallocations, counts.requested_bytes, sample.dispatches).unwrap();
    }
    let ci = metrics::bootstrap_p95(&complete);
    eprintln!(
        "summary case={id} repeat={repeat} n={samples} complete_p50_us={:.3} complete_p95_us={:.3} p95_bootstrap95_us=[{:.3},{:.3}] encode_p50_us={:.3} gpu_p50_us={} pixels_per_second={:.0} {}",
        metrics::quantile(&complete, 50),
        metrics::quantile(&complete, 95),
        ci.0,
        ci.1,
        metrics::quantile(&encode, 50),
        if gpu.is_empty() {
            "NA".into()
        } else {
            format!("{:.3}", metrics::quantile(&gpu, 50))
        },
        case.pixels() as f64 * samples as f64 * 1e6 / complete.iter().sum::<f64>(),
        rss()
    );
    assert_eq!(camera.stats().submissions, 0);
    for group in 0..camera.group_count() {
        let stats = camera.query_stats(group).unwrap();
        assert_eq!(stats.scene_builds, 1);
        assert_eq!(stats.refit_preparations, 1);
        eprintln!(
            "group={group} camera={:?} query={stats:?}",
            camera.group_stats(group).unwrap()
        );
    }
    if let Some(chain) = chain {
        assert_eq!(chain.images.stats().submissions, 0);
        assert_eq!(chain.annotations.stats().submissions, 0);
        eprintln!(
            "image={:?} annotation={:?}",
            chain.images.stats(),
            chain.annotations.stats()
        );
    }
    output.flush().unwrap();
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut full = false;
    let (mut samples, mut warmup, mut repeats) = run_mode::defaults(&args);
    let mut timestamps = true;
    let mut filter = String::new();
    let mut destination = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--camera-matrix" => full = true,
            "--no-timestamps" => timestamps = false,
            "--samples" | "--warmup" | "--repeats" | "--filter" | "--output" => {
                let key = &args[i];
                i += 1;
                let value = args.get(i).expect("missing option value");
                match key.as_str() {
                    "--samples" => samples = value.parse().unwrap(),
                    "--warmup" => warmup = value.parse().unwrap(),
                    "--repeats" => repeats = value.parse().unwrap(),
                    "--filter" => filter = value.clone(),
                    _ => destination = Some(value.clone()),
                }
            }
            "--bench" | "--smoke" | "--test" => {}
            "--help" => {
                println!(
                    "--bench selects original resolutions, 64 samples, 8 warmup and 3 repeats; otherwise smoke uses one-tenth image dimensions and 2/1/1 sampling, retaining camera counts, aperture and processing cases. --smoke | --test --camera-matrix --samples N --warmup N --repeats N --filter SUBSTRING --output NEW.tsv --no-timestamps"
                );
                return;
            }
            option => panic!("unknown benchmark option: {option}"),
        }
        i += 1;
    }
    assert!((1..=10000).contains(&samples) && warmup <= 10000 && (1..=20).contains(&repeats));
    let selected: Vec<_> = cases(full)
        .into_iter()
        .map(|mut case| {
            (case.width, case.height) = camera_latency_smoke::dimensions(
                case.width,
                case.height,
                run_mode::is_smoke(&args),
            );
            case
        })
        .flat_map(|case| [false, true].map(|host| (case, host)))
        .filter(|(case, host)| case.id(*host).contains(&filter))
        .collect();
    assert!(!selected.is_empty(), "filter matched no cases");
    let mut output: Box<dyn Write> = if let Some(path) = destination {
        Box::new(BufWriter::new(
            std::fs::OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(path)
                .unwrap(),
        ))
    } else {
        Box::new(BufWriter::new(std::io::stdout().lock()))
    };
    writeln!(output, "case\trepeat\tphase\tindex\tpixels\tencode_us\tsubmit_elapsed_us\tcomplete_us\tgpu_us\tpose_upload_bytes\tresult_download_bytes\tsubmissions\twaits\tallocations\treallocations\trequested_bytes\tdispatches").unwrap();
    let engine = timing::engine(timestamps);
    eprintln!(
        "adapter={:?} profile={} cases={} samples={samples} warmup={warmup} repeats={repeats} pid={} smoke={}",
        engine.adapter_info(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        selected.len(),
        std::process::id(),
        run_mode::is_smoke(&args)
    );
    for repeat in 0..repeats {
        for (case, host) in selected
            .iter()
            .cycle()
            .skip(repeat % selected.len())
            .take(selected.len())
        {
            run(&engine, *case, *host, samples, warmup, repeat, &mut output);
        }
    }
}
