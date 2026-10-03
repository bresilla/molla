//! Completed Featherstone stepping followed by device-resident sensor workloads.

#[path = "../../molla-sensors/benches/support/cloud_chain.rs"]
#[allow(dead_code)]
mod cloud_chain;
#[path = "../../molla-sensors/benches/support/metrics.rs"]
mod metrics;
#[path = "support/sensor_physics.rs"]
mod physics;
#[path = "../../molla-sensors/benches/support/run_mode.rs"]
mod run_mode;
#[path = "../../molla-sensors/benches/support/timing.rs"]
mod timing;

use molla_compute::Engine;
use molla_math::{Real, Vec3};
use molla_sensors::{
    CameraChannels, CloudGridCell, CloudGridDiagnostics, CloudPoint, LidarBatch, LidarPattern,
    LidarScan, PhysicalOptics, ScanMetadata, SensorCommandBatch, SensorFrame, SensorFramePool,
    SensorFramePoolConfig, SensorScene, TiledCamera, TiledCameraConfig, TiledCameraLimits,
    readback::{ReadbackConfig, ReadbackRing, ReadbackSource, ReadbackStamp},
};
use molla_storage::BufferElement;
use std::{
    io::{BufWriter, Write},
    time::{Duration, Instant},
};

#[global_allocator]
static ALLOCATOR: metrics::CountingAllocator = metrics::CountingAllocator;

#[derive(Clone, Copy)]
struct Case {
    gpu: bool,
    articulations: usize,
    steps: usize,
    load: u32,
    host: bool,
}

impl Case {
    fn id(self) -> String {
        format!(
            "{}_a{}_steps{}_{}_{}",
            if self.gpu { "gpu" } else { "cpu" },
            self.articulations,
            self.steps,
            [
                "lidar1k",
                "cloud100k_camera640",
                "cloud1m_camera4x720_aperture4"
            ][self.load as usize],
            if self.host { "host" } else { "device" }
        )
    }
    fn dimensions(self) -> (u32, u32, u32, u32, usize) {
        match self.load {
            0 => (1_000, 1, 0, 0, 0),
            1 => (100_000, 4, 640, 480, 1),
            _ => (1_000_000, 16, 1280, 720, 4),
        }
    }
}

fn source<T: BufferElement>(frame: &SensorFrame, channel: usize) -> ReadbackSource<'_> {
    let view = frame.channel::<T>(channel).unwrap();
    view.readback_source(0..view.len()).unwrap()
}

fn rss() -> String {
    std::fs::read_to_string("/proc/self/status")
        .unwrap_or_default()
        .lines()
        .filter(|s| s.starts_with("VmHWM:") || s.starts_with("VmRSS:"))
        .collect::<Vec<_>>()
        .join("; ")
}

struct Workload {
    physics: physics::Physics,
    scene: SensorScene,
    lidar: LidarBatch,
    lidar_pool: SensorFramePool,
    scans: Vec<LidarScan>,
    cloud: Option<cloud_chain::CloudChain>,
    camera: Option<(TiledCamera, SensorFramePool)>,
    ring: ReadbackRing,
    rays: u32,
    pixels: u32,
    host_bytes: u64,
}

impl Workload {
    fn new(engine: &Engine, case: Case) -> Self {
        let physics = physics::Physics::new(engine, case.articulations, case.gpu);
        let scene = SensorScene::from_model(engine, &physics.model, Default::default()).unwrap();
        let (nominal, mounts, width, height, cameras) = case.dimensions();
        let rows = 32;
        let columns = nominal.div_ceil(mounts * rows);
        let rays = rows * columns * mounts;
        let side = (case.articulations as f64).sqrt().ceil() as usize;
        let scans = (0..mounts)
            .map(|i| {
                let art = i as usize * case.articulations / mounts as usize;
                LidarScan {
                    sensor_id: i as u64,
                    pattern: LidarPattern::Terrestrial(ScanMetadata {
                        forward: Vec3::NEG_Y,
                        up: Vec3::Z,
                        theta_range: (1.2, 1.9),
                        phi_range: (-0.4, 0.4),
                        max_distance: 30.0,
                        ..ScanMetadata::new(
                            Vec3::new(
                                (art % side) as f64 * 4.0 + 0.5 + i as f64 * 0.02,
                                12.0,
                                (art / side) as f64 * 4.0,
                            ),
                            rows,
                            columns,
                        )
                    }),
                }
            })
            .collect();
        let lidar = LidarBatch::new(engine, mounts as usize, rays).unwrap();
        let pool_config = SensorFramePoolConfig {
            slots: 1,
            max_total_bytes: 256 << 20,
        };
        let lidar_pool = lidar.output_pool(engine, pool_config).unwrap();
        let cloud =
            (case.load != 0).then(|| cloud_chain::CloudChain::new(engine, rays, mounts, "keep"));
        let pixels = width * height * cameras as u32;
        let camera = (cameras != 0).then(|| {
            let config = TiledCameraConfig {
                width,
                height,
                channels: CameraChannels::ALL,
                max_distance: 30.0,
                optics: PhysicalOptics {
                    lens_diameter_m: if case.load == 2 { 0.04 } else { 0.0 },
                    aperture_samples: if case.load == 2 { 4 } else { 1 },
                    focal_distance_m: 8.0,
                    ..Default::default()
                },
                ..Default::default()
            };
            let mut camera = TiledCamera::new_with_limits(
                engine,
                config,
                cameras,
                TiledCameraLimits {
                    max_storage_bytes: 2 << 30,
                    ..Default::default()
                },
            )
            .unwrap();
            for i in 0..cameras {
                let art = i * case.articulations / cameras;
                let target = Vec3::new(
                    (art % side) as f64 * 4.0 + i as f64 * 0.1,
                    0.0,
                    (art / side) as f64 * 4.0,
                );
                camera
                    .set_camera_look_at(i, target + Vec3::Y * 13.0, target, Vec3::Z)
                    .unwrap();
            }
            eprintln!("camera_arrays={:?}", camera.storage_usage());
            let pool = camera.output_pool(engine, pool_config).unwrap();
            (camera, pool)
        });
        let host_bytes = u64::from(pixels) * 32
            + if cloud.is_some() {
                cloud_chain::CloudChain::host_bytes(rays, mounts)
            } else {
                u64::from(rays) * 52
            };
        let ring = ReadbackRing::new(
            engine,
            ReadbackConfig {
                slots: 1,
                bytes_per_slot: host_bytes,
                max_total_bytes: host_bytes.next_multiple_of(8),
            },
        )
        .unwrap();
        eprintln!(
            "bodies={} shapes={} actual_rays={rays} pixels={pixels} host_bytes={host_bytes} lidar_arrays={:?}",
            physics.model.body_count,
            physics.model.shape_count,
            lidar.storage_usage()
        );
        if let Some(cloud) = &cloud {
            cloud.report();
        }
        Self {
            physics,
            scene,
            lidar,
            lidar_pool,
            scans,
            cloud,
            camera,
            ring,
            rays,
            pixels,
            host_bytes,
        }
    }

    fn frame(
        &mut self,
        engine: &Engine,
        case: Case,
        index: usize,
        timer: Option<&timing::Timer>,
    ) -> Sample {
        let start = Instant::now();
        let mut marker_submissions = 0;
        if let Some(timer) = timer {
            let mut marker = SensorCommandBatch::new(engine).unwrap();
            timer.begin(&mut marker);
            marker_submissions = marker.submit().unwrap().stats.submissions;
            assert_eq!(marker_submissions, 1);
        }
        let physics_start = Instant::now();
        let pose_bytes = self.physics.step(engine, case.steps);
        let physics_cpu_us = physics_start.elapsed().as_secs_f64() * 1e6;
        let sensor_start = Instant::now();
        let before_read = self.ring.stats();
        let before_lidar = self.lidar.stats();
        let stamp = ReadbackStamp {
            sample_id: index as u64,
            simulation_time: Some((index + 1) as f64 * case.steps as f64 * physics::DT),
            ..Default::default()
        };
        let mut batch = SensorCommandBatch::new(engine).unwrap();
        let lidar = self
            .lidar
            .encode_scene(
                &mut batch,
                engine,
                &self.scene,
                &self.physics.state,
                &self.scans,
                &mut self.lidar_pool,
                stamp,
            )
            .unwrap()
            .unwrap();
        let cloud = self
            .cloud
            .as_mut()
            .map(|cloud| cloud.encode(&mut batch, &lidar, index));
        let camera = self.camera.as_mut().map(|(camera, pool)| {
            camera
                .encode_scene(
                    &mut batch,
                    engine,
                    &self.scene,
                    &self.physics.state,
                    pool,
                    stamp,
                )
                .unwrap()
                .unwrap()
        });
        if let Some(timer) = timer {
            timer.end(&mut batch);
        }
        let sensor_encode_us = sensor_start.elapsed().as_secs_f64() * 1e6;
        let submitted = batch.submit().unwrap();
        let submit_elapsed_us = start.elapsed().as_secs_f64() * 1e6;
        assert_eq!(submitted.stats.submissions, 1);
        assert_eq!(
            submitted.stats.encoded_frames,
            if case.load == 0 { 1 } else { 4 }
        );
        if case.host {
            let mut sources = Vec::new();
            if let Some((cloud, grid)) = &cloud {
                let cf = submitted.frame(&cloud.frame).unwrap();
                let gf = submitted.frame(&grid.frame).unwrap();
                sources.extend([
                    source::<CloudPoint>(cf, 0),
                    source::<u32>(cf, 1),
                    source::<u32>(gf, 1),
                    source::<CloudGridCell>(gf, 2),
                    source::<u32>(gf, 3),
                ]);
            } else {
                let f = submitted.frame(&lidar.frame).unwrap();
                sources.extend([
                    source::<Real>(f, 2),
                    source::<i32>(f, 3),
                    source::<Vec3>(f, 4),
                    source::<Vec3>(f, 5),
                    source::<i32>(f, 6),
                    source::<i32>(f, 7),
                    source::<u32>(f, 8),
                ]);
            }
            let camera_offset = sources.len();
            if let Some(camera) = &camera {
                let f = submitted.frame(&camera.frame).unwrap();
                sources.extend([
                    source::<Real>(f, 0),
                    source::<u32>(f, 1),
                    source::<Vec3>(f, 2),
                    source::<u32>(f, 3),
                    source::<u32>(f, 4),
                ]);
            }
            let ticket = self.ring.try_enqueue(&sources, stamp).unwrap().unwrap();
            let collected = self.ring.wait(&ticket, Duration::from_secs(30)).unwrap();
            if cloud.is_some() {
                assert_eq!(collected.channel::<u32>(1).unwrap()[0], self.rays);
                let diag = CloudGridDiagnostics::from_words(&collected.channel::<u32>(4).unwrap())
                    .unwrap();
                assert_eq!(
                    diag,
                    CloudGridDiagnostics {
                        accepted: self.rays,
                        ..Default::default()
                    }
                );
                assert_eq!(
                    collected
                        .channel_iter::<CloudGridCell>(3)
                        .unwrap()
                        .map(|c| {
                            assert_eq!(c.header[2], 0);
                            c.header[1] as u64
                        })
                        .sum::<u64>(),
                    self.rays as u64
                );
                for p in collected.channel_iter::<CloudPoint>(0).unwrap() {
                    assert_eq!(p.sample_id(), index as u64);
                    assert!(p.position[..3].iter().all(|x| x.is_finite()));
                    assert!((p.sensor_id() as usize) < self.scans.len());
                    assert!((p.extra[0] as usize) < self.physics.model.shape_count);
                }
            } else {
                assert!(
                    collected
                        .channel_iter::<Real>(0)
                        .unwrap()
                        .all(|t| t > 0.0 && t.is_finite())
                );
                assert!(collected.channel_iter::<i32>(1).unwrap().all(|id| id >= 0));
            }
            if camera.is_some() {
                assert!(
                    collected
                        .channel_iter::<Real>(camera_offset)
                        .unwrap()
                        .all(|t| t > 0.0 && t.is_finite())
                );
                assert!(
                    collected
                        .channel_iter::<u32>(camera_offset + 4)
                        .unwrap()
                        .all(|id| (id as usize) < self.physics.model.shape_count)
                );
            }
            std::hint::black_box(collected);
        } else {
            timing::wait(engine);
        }
        let after_read = self.ring.stats();
        let dispatches = submitted.stats.kernel_dispatches;
        let sensor_uploads = submitted.stats.input_upload_bytes;
        let read_submissions = after_read.submissions - before_read.submissions;
        let downloads = after_read.copied_bytes - before_read.copied_bytes;
        assert_eq!(read_submissions, u64::from(case.host));
        assert_eq!(downloads, if case.host { self.host_bytes } else { 0 });
        drop(submitted);
        drop((lidar, cloud, camera));
        let complete_us = start.elapsed().as_secs_f64() * 1e6;
        let lidar_uploads =
            self.lidar.stats().descriptor_upload_bytes - before_lidar.descriptor_upload_bytes;
        let camera_uploads = if self.pixels == 0 {
            0
        } else {
            case.dimensions().4 as u64 * 32
        };
        assert_eq!(sensor_uploads, lidar_uploads + camera_uploads);
        Sample {
            physics_cpu_us,
            sensor_encode_us,
            submit_elapsed_us,
            complete_us,
            pose_bytes,
            sensor_uploads,
            downloads,
            sensor_dispatches: dispatches,
            marker_submissions,
            read_submissions,
            waits: if case.host {
                after_read.blocking_waits - before_read.blocking_waits
            } else {
                1
            },
        }
    }
}

struct Sample {
    physics_cpu_us: f64,
    sensor_encode_us: f64,
    submit_elapsed_us: f64,
    complete_us: f64,
    pose_bytes: u64,
    sensor_uploads: u64,
    downloads: u64,
    sensor_dispatches: u64,
    marker_submissions: u64,
    read_submissions: u64,
    waits: u64,
}

fn run(
    engine: &Engine,
    case: Case,
    samples: usize,
    warmup: usize,
    repeat: usize,
    output: &mut impl Write,
) {
    let id = case.id();
    let setup = Instant::now();
    let mut workload = Workload::new(engine, case);
    let initial = workload.physics.state.body_q.host().unwrap()[0].position;
    let mut timer = timing::Timer::new(engine);
    engine.queue().unwrap().submit([]);
    timing::wait(engine);
    eprintln!(
        "case={id} repeat={repeat} setup_ms={:.3} timer={} {}",
        setup.elapsed().as_secs_f64() * 1e3,
        timer.is_some(),
        rss()
    );
    let mut latencies = Vec::with_capacity(samples);
    let mut spans = Vec::with_capacity(samples);
    let mut physics_times = Vec::with_capacity(samples);
    for index in 0..1 + warmup + samples {
        let phase = if index == 0 {
            "first"
        } else if index <= warmup {
            "warmup"
        } else {
            "sample"
        };
        let (sample, allocations) =
            metrics::measure(|| workload.frame(engine, case, index, timer.as_ref()));
        let span = timer.as_mut().map(timing::Timer::collect_us);
        if phase == "sample" {
            latencies.push(sample.complete_us);
            physics_times.push(sample.physics_cpu_us);
            if let Some(span) = span {
                spans.push(span);
            }
        }
        writeln!(output, "{id}\t{repeat}\t{phase}\t{index}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t1\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            workload.rays, workload.pixels, workload.physics.model.body_count, sample.physics_cpu_us,
            sample.sensor_encode_us, sample.submit_elapsed_us, sample.complete_us,
            span.map_or_else(|| "NA".into(), |s| format!("{s:.3}")), sample.pose_bytes,
            sample.sensor_uploads, if case.gpu { case.steps * 48 } else { 0 }, sample.downloads,
            if case.gpu { case.steps * 5 } else { 0 }, sample.read_submissions, sample.marker_submissions,
            sample.waits, sample.sensor_dispatches, allocations.allocations, allocations.reallocations,
            allocations.requested_bytes).unwrap();
    }
    let ci = metrics::bootstrap_p95(&latencies);
    eprintln!(
        "summary case={id} repeat={repeat} n={samples} complete_p50_us={:.3} complete_p95_us={:.3} p95_bootstrap95_us=[{:.3},{:.3}] physics_cpu_p50_us={:.3} gpu_queue_span_p50_us={} {} lidar={:?}",
        metrics::quantile(&latencies, 50),
        metrics::quantile(&latencies, 95),
        ci.0,
        ci.1,
        metrics::quantile(&physics_times, 50),
        if spans.is_empty() {
            "NA".into()
        } else {
            format!("{:.3}", metrics::quantile(&spans, 50))
        },
        rss(),
        workload.lidar.query_stats()
    );
    if let Some(cloud) = &workload.cloud {
        cloud.report();
    }
    if let Some((camera, _)) = &workload.camera {
        eprintln!(
            "camera={:?} query={:?}",
            camera.dispatch_stats(),
            camera.query_stats()
        );
    }
    assert_eq!(workload.lidar.readback_stats().submissions, 0);
    assert_eq!(workload.lidar.query_stats().scene_builds, 1);
    assert!(workload.lidar.query_stats().refit_preparations <= 2);
    if case.gpu {
        assert!(workload.physics.state.body_q.host().is_err());
        workload
            .physics
            .state
            .body_q
            .download(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
    }
    let final_poses = workload.physics.state.body_q.host().unwrap();
    assert!(
        final_poses
            .iter()
            .all(|p| p.position.is_finite() && p.rotation.is_finite())
    );
    assert!((final_poses[0].position - initial).length() > 1e-7);
    output.flush().unwrap();
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let (mut samples, mut warmup, mut repeats) = run_mode::defaults(&args);
    let (mut full, mut timestamps) = (false, true);
    let mut filter = String::new();
    let mut destination = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--physics-matrix" => full = true,
            "--no-timestamps" => timestamps = false,
            "--samples" | "--warmup" | "--repeats" | "--filter" | "--output" => {
                let key = &args[i];
                i += 1;
                let value = args.get(i).expect("missing argument");
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
                    "--bench selects 64 samples, 8 warmup and 3 repeats; otherwise smoke defaults are 2/1/1. --smoke | --test --physics-matrix --samples N --warmup N --repeats N --filter SUBSTRING --output NEW.tsv --no-timestamps"
                );
                return;
            }
            option => panic!("unknown option {option}"),
        }
        i += 1;
    }
    assert!((1..=10000).contains(&samples) && warmup <= 10000 && (1..=20).contains(&repeats));
    let mut cases = Vec::new();
    for gpu in [false, true] {
        for articulations in [1, 32] {
            for steps in [1, 10] {
                for load in 0..3 {
                    for host in [false, true] {
                        let case = Case {
                            gpu,
                            articulations,
                            steps,
                            load,
                            host,
                        };
                        if (full || articulations == 1 && steps == 1 && load != 2)
                            && case.id().contains(&filter)
                        {
                            cases.push(case);
                        }
                    }
                }
            }
        }
    }
    assert!(!cases.is_empty(), "no matching cases");
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
    writeln!(output, "case\trepeat\tphase\tindex\trays\tpixels\tbodies\tphysics_cpu_elapsed_us\tsensor_encode_us\tsubmit_elapsed_us\tcomplete_us\tgpu_queue_span_us\tcpu_pose_upload_bytes\tsensor_descriptor_pose_upload_bytes\tphysics_uniform_bytes_source_count\tresult_download_bytes\tphysics_submissions_source_count\tsensor_submissions\treadback_submissions\ttimer_marker_submissions\tblocking_waits\tsensor_dispatches\tallocations\treallocations\trequested_bytes").unwrap();
    let engine = timing::engine(timestamps);
    eprintln!(
        "adapter={:?} profile={} cases={} samples={samples} warmup={warmup} repeats={repeats} pid={} smoke={}",
        engine.adapter_info(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        cases.len(),
        std::process::id(),
        run_mode::is_smoke(&args)
    );
    for repeat in 0..repeats {
        for case in cases
            .iter()
            .cycle()
            .skip(repeat % cases.len())
            .take(cases.len())
        {
            run(&engine, *case, samples, warmup, repeat, &mut output);
        }
    }
}
