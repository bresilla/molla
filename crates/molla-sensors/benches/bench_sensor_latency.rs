//! Per-frame LiDAR latency samples, GPU timestamps and explicit transfer/allocation counters.

#[path = "support/cloud_chain.rs"]
mod cloud_chain;
#[path = "support/metrics.rs"]
mod metrics;
#[path = "support/timing.rs"]
mod timing;

use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{GpuTransform, Real, Transform, Vec3};
use molla_sensors::{
    LidarBatch, LidarPattern, LidarScan, ScanMetadata, SensorCommandBatch, SensorFramePool,
    SensorFramePoolConfig, SensorScene,
    readback::{ReadbackConfig, ReadbackRing},
};
use molla_sim::{Heightfield, Model, ModelBuilder, State, TriMesh};
use std::{
    io::{BufWriter, Write},
    time::{Duration, Instant},
};

#[global_allocator]
static ALLOCATOR: metrics::CountingAllocator = metrics::CountingAllocator;

#[derive(Clone, Copy)]
struct Case {
    geometry: &'static str,
    rays: u32,
    mounts: u32,
    subrays: u32,
    lifecycle: &'static str,
    cloud: Option<&'static str>,
}

impl Case {
    fn id(self, host: bool) -> String {
        format!(
            "{}_r{}_m{}_s{}_{}{}_{}",
            self.geometry,
            self.rays,
            self.mounts,
            self.subrays,
            self.lifecycle,
            self.cloud
                .map_or_else(String::new, |mode| format!("_cloud_{mode}")),
            if host { "host" } else { "device" }
        )
    }
}

fn cases(full: bool) -> Vec<Case> {
    if !full {
        return vec![
            Case {
                geometry: "prim100",
                rays: 1_000,
                mounts: 1,
                subrays: 1,
                lifecycle: "warm",
                cloud: None,
            },
            Case {
                geometry: "prim1000",
                rays: 10_000,
                mounts: 4,
                subrays: 1,
                lifecycle: "warm",
                cloud: None,
            },
            Case {
                geometry: "mesh10",
                rays: 100_000,
                mounts: 16,
                subrays: 4,
                lifecycle: "pose",
                cloud: None,
            },
            Case {
                geometry: "terrain",
                rays: 100_000,
                mounts: 4,
                subrays: 1,
                lifecycle: "warm",
                cloud: None,
            },
            Case {
                geometry: "mesh1",
                rays: 10_000,
                mounts: 1,
                subrays: 1,
                lifecycle: "replace",
                cloud: None,
            },
        ];
    }
    let mut cases = Vec::new();
    for rays in [1_000, 10_000, 100_000, 1_000_000] {
        for geometry in [
            "prim100",
            "prim1000",
            "prim10000",
            "mesh1",
            "mesh10",
            "mesh100",
            "terrain",
        ] {
            cases.push(Case {
                geometry,
                rays,
                mounts: 1,
                subrays: 1,
                lifecycle: "warm",
                cloud: None,
            });
        }
        for mounts in [1, 4, 16] {
            for subrays in [1, 4] {
                cases.push(Case {
                    geometry: "mesh10",
                    rays,
                    mounts,
                    subrays,
                    lifecycle: "pose",
                    cloud: None,
                });
            }
        }
        for geometry in ["prim1000", "mesh1"] {
            cases.push(Case {
                geometry,
                rays,
                mounts: 4,
                subrays: 1,
                lifecycle: "replace",
                cloud: None,
            });
        }
    }
    cases
}

fn cloud_cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for rays in [1_000, 10_000, 100_000, 1_000_000] {
        for (geometry, mounts, subrays, lifecycle) in [
            ("mesh1", 1, 1, "warm"),
            ("mesh10", 4, 4, "pose"),
            ("mesh100", 16, 1, "pose"),
            ("terrain", 4, 1, "warm"),
        ] {
            for mode in ["keep", "crop", "reject"] {
                cases.push(Case {
                    geometry,
                    rays,
                    mounts,
                    subrays,
                    lifecycle,
                    cloud: Some(mode),
                });
            }
        }
    }
    cases
}

fn detailed_mesh() -> TriMesh {
    let mut mesh = TriMesh {
        vertices: Vec::new(),
        indices: Vec::new(),
    };
    for z in 0..=128 {
        for x in 0..=128 {
            let x = x as f64 / 16.0 - 4.0;
            let z = z as f64 / 16.0 - 4.0;
            mesh.vertices.push(Vec3::new(x, 0.2 * x.sin() * z.cos(), z));
        }
    }
    for z in 0..128 {
        for x in 0..128 {
            let i = z * 129 + x;
            mesh.indices
                .extend([[i, i + 129, i + 1], [i + 1, i + 129, i + 130]]);
        }
    }
    mesh
}

fn fixture(geometry: &str, variant: bool) -> Model {
    let mut builder = ModelBuilder::new();
    builder
        .add_shape_plane(
            BodyId::NONE,
            Vec3::Y,
            Transform::from_translation(Vec3::NEG_Y * 2.0),
        )
        .unwrap();
    if let Some(n) = geometry.strip_prefix("prim") {
        let count: usize = n.parse().unwrap();
        let side = (count as f64).sqrt().ceil() as usize;
        for i in 0..count {
            let pose = Transform::from_translation(Vec3::new(
                (i % side) as f64 - side as f64 * 0.5,
                0.0,
                (i / side) as f64 - side as f64 * 0.5,
            ));
            builder.add_shape_sphere(BodyId::NONE, 0.4, pose).unwrap();
        }
    } else if let Some(n) = geometry.strip_prefix("mesh") {
        let count: usize = n.parse().unwrap();
        let mesh = builder.add_mesh(detailed_mesh());
        let side = (count as f64).sqrt().ceil() as usize;
        for i in 0..count {
            let body = builder.add_body(Default::default());
            builder
                .add_shape_trimesh(
                    body,
                    mesh,
                    Transform::from_translation(Vec3::new(
                        (i % side) as f64 * 10.0,
                        0.0,
                        (i / side) as f64 * 10.0,
                    )),
                )
                .unwrap();
        }
    } else {
        assert_eq!(geometry, "terrain");
        let field = builder.add_heightfield(Heightfield {
            nx: 129,
            nz: 129,
            cell_size: 0.25,
            cell_size_z: 0.25,
            origin: Vec3::new(-16.0, 0.0, -16.0),
            heights: (0..129 * 129)
                .map(|i| 0.2 * (i as f64 * 0.013).sin())
                .collect(),
        });
        builder
            .add_shape_heightfield(BodyId::NONE, field, Transform::IDENTITY)
            .unwrap();
    }
    if variant {
        builder
            .add_shape_sphere(
                BodyId::NONE,
                0.25,
                Transform::from_translation(Vec3::X * 1000.0),
            )
            .unwrap();
    }
    builder.build().unwrap()
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
    sensor: &mut LidarBatch,
    pool: &mut SensorFramePool,
    scene: &mut SensorScene,
    model: &Model,
    alternate: Option<&Model>,
    state: &mut State,
    poses: &mut [GpuTransform],
    scans: &[LidarScan],
    ring: &mut ReadbackRing,
    timer: Option<&timing::Timer>,
    mut cloud: Option<&mut cloud_chain::CloudChain>,
    case: Case,
    index: usize,
    host: bool,
) -> Sample {
    let start = Instant::now();
    let mut pose_bytes = 0;
    if case.lifecycle == "pose" {
        for (i, pose) in poses.iter_mut().enumerate() {
            *pose = Transform::from_translation(Vec3::Y * ((index + i) as f64 * 0.13).sin() * 0.2)
                .into();
        }
        engine.queue().unwrap().write_buffer(
            state.body_q.device_buffer().unwrap(),
            0,
            bytemuck::cast_slice(poses),
        );
        state.body_q.mark_device_dirty();
        pose_bytes = poses.len() as u64 * 32;
    }
    if case.lifecycle == "replace" {
        scene
            .replace_model(
                engine,
                if index % 2 == 0 {
                    alternate.unwrap()
                } else {
                    model
                },
            )
            .unwrap();
    }
    let before = sensor.stats();
    let before_read = ring.stats();
    let mut batch = SensorCommandBatch::new(engine).unwrap();
    if let Some(timer) = timer {
        timer.begin(&mut batch);
    }
    let receipt = sensor
        .encode_scene(
            &mut batch,
            engine,
            scene,
            state,
            scans,
            pool,
            Default::default(),
        )
        .unwrap()
        .expect("completed slot must be available");
    let processed = cloud
        .as_mut()
        .map(|pipeline| pipeline.encode(&mut batch, &receipt, index));
    if let Some(timer) = timer {
        timer.end(&mut batch);
    }
    let encode_us = start.elapsed().as_secs_f64() * 1e6;
    let submitted = batch.submit().unwrap();
    let submit_us = start.elapsed().as_secs_f64() * 1e6;
    let mut waits = 0;
    if host && let Some((cloud, grid)) = &processed {
        cloud_chain::CloudChain::collect(
            ring,
            submitted.frame(&cloud.frame).unwrap(),
            submitted.frame(&grid.frame).unwrap(),
            &receipt,
            case.cloud.unwrap(),
            index,
        );
    } else if host {
        let output = submitted.frame(&receipt.frame).unwrap();
        let t = output.channel::<Real>(2).unwrap();
        let id = output.channel::<i32>(3).unwrap();
        let point = output.channel::<Vec3>(4).unwrap();
        let normal = output.channel::<Vec3>(5).unwrap();
        let first = output.channel::<i32>(6).unwrap();
        let last = output.channel::<i32>(7).unwrap();
        let count = output.channel::<u32>(8).unwrap();
        let sources = [
            t.readback_source(0..t.len()).unwrap(),
            id.readback_source(0..id.len()).unwrap(),
            point.readback_source(0..point.len()).unwrap(),
            normal.readback_source(0..normal.len()).unwrap(),
            first.readback_source(0..first.len()).unwrap(),
            last.readback_source(0..last.len()).unwrap(),
            count.readback_source(0..count.len()).unwrap(),
        ];
        let ticket = ring
            .try_enqueue(&sources, Default::default())
            .unwrap()
            .unwrap();
        let collected = ring.wait(&ticket, Duration::from_secs(30)).unwrap();
        let sum: f64 = collected.channel_iter::<Real>(0).unwrap().sum();
        assert!(sum.is_finite() && sum > 0.0);
        assert!(collected.channel_iter::<i32>(1).unwrap().all(|id| id >= 0));
        std::hint::black_box(collected);
    } else {
        timing::wait(engine);
        waits = 1;
    }
    let after_read = ring.stats();
    let submissions =
        submitted.stats.submissions + after_read.submissions - before_read.submissions;
    let dispatches = submitted.stats.kernel_dispatches;
    assert_eq!(submissions, if host { 2 } else { 1 });
    assert_eq!(
        submitted.stats.encoded_frames,
        if cloud.is_some() { 3 } else { 1 }
    );
    drop(submitted);
    drop(receipt);
    drop(processed);
    let complete_us = start.elapsed().as_secs_f64() * 1e6;
    Sample {
        encode_us,
        submit_us,
        complete_us,
        uploads: sensor.stats().descriptor_upload_bytes - before.descriptor_upload_bytes
            + pose_bytes,
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
    let model = fixture(case.geometry, false);
    let alternate = (case.lifecycle == "replace").then(|| fixture(case.geometry, true));
    let mut state = model.state().unwrap();
    state
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let mut scene = SensorScene::from_model(engine, &model, Default::default()).unwrap();
    let requested_pulses = case.rays.div_ceil(case.mounts * case.subrays);
    let rows = requested_pulses.min(32);
    let columns = requested_pulses.div_ceil(rows);
    let pulses_per_mount = rows * columns;
    let rays = pulses_per_mount * case.mounts * case.subrays;
    let scans: Vec<_> = (0..case.mounts)
        .map(|i| {
            let count: u32 = case
                .geometry
                .strip_prefix("mesh")
                .map_or(1, |n| n.parse().unwrap());
            let tile = i * count / case.mounts;
            let side = (count as f64).sqrt().ceil() as u32;
            let origin = Vec3::new(
                (tile % side) as f64 * 10.0 + i as f64 * 0.01,
                10.0,
                (tile / side) as f64 * 10.0,
            );
            LidarScan {
                sensor_id: i as u64,
                pattern: LidarPattern::Terrestrial(ScanMetadata {
                    forward: Vec3::NEG_Y,
                    up: Vec3::Z,
                    theta_range: (1.2, 1.9),
                    phi_range: (-0.4, 0.4),
                    subray_count: case.subrays,
                    beam_divergence: 0.003,
                    max_distance: 30.0,
                    ..ScanMetadata::new(origin, rows, columns)
                }),
            }
        })
        .collect();
    let mut sensor = LidarBatch::new(engine, case.mounts as usize, rays).unwrap();
    let mut pool = sensor
        .output_pool(
            engine,
            SensorFramePoolConfig {
                slots: 1,
                max_total_bytes: 256 << 20,
            },
        )
        .unwrap();
    let mut cloud = case
        .cloud
        .map(|mode| cloud_chain::CloudChain::new(engine, rays, case.mounts, mode));
    let bytes = if cloud.is_some() {
        cloud_chain::CloudChain::host_bytes(rays, case.mounts)
    } else {
        u64::from(rays) * 40 + u64::from(pulses_per_mount * case.mounts) * 12
    };
    let mut ring = ReadbackRing::new(
        engine,
        ReadbackConfig {
            slots: 1,
            bytes_per_slot: bytes,
            max_total_bytes: (bytes + 7) / 8 * 8,
        },
    )
    .unwrap();
    let mut poses = vec![GpuTransform::from(Transform::IDENTITY); model.body_count];
    let mut timer = timing::Timer::new(engine);
    engine.queue().unwrap().submit([]);
    timing::wait(engine);
    let setup_ms = build.elapsed().as_secs_f64() * 1e3;
    eprintln!(
        "case={id} repeat={repeat} actual_rays={rays} rows={rows} columns={columns} setup_ms={setup_ms:.3} shape_count={} unique_meshes={} unique_triangles={} scene_atlas_bytes={} retained_sensor_arrays={:?} output_pool_bytes={} staging_bytes={} timer={} {}",
        model.shape_count,
        scene.mesh_count(),
        model
            .meshes
            .iter()
            .map(|mesh| mesh.indices.len())
            .sum::<usize>()
            + model
                .heightfields
                .iter()
                .map(|field| 2 * (field.nx as usize - 1) * (field.nz as usize - 1))
                .sum::<usize>(),
        scene.query_device_bytes(),
        sensor.storage_usage(),
        u64::from(rays) * 84,
        bytes,
        timer.is_some(),
        rss()
    );
    let mut latencies = Vec::with_capacity(samples);
    if let Some(cloud) = &cloud {
        cloud.report();
    }
    let mut gpu_times = Vec::with_capacity(samples);
    let mut encode_times = Vec::with_capacity(samples);
    for i in 0..samples + warmup + 1 {
        let phase = if i == 0 {
            "first"
        } else if i <= warmup {
            "warmup"
        } else {
            "sample"
        };
        let (sample, counts) = metrics::measure(|| {
            frame(
                engine,
                &mut sensor,
                &mut pool,
                &mut scene,
                &model,
                alternate.as_ref(),
                &mut state,
                &mut poses,
                &scans,
                &mut ring,
                timer.as_ref(),
                cloud.as_mut(),
                case,
                i,
                host,
            )
        });
        let gpu_us = timer.as_mut().map(timing::Timer::collect_us);
        if phase == "sample" {
            latencies.push(sample.complete_us);
            encode_times.push(sample.encode_us);
            if let Some(us) = gpu_us {
                gpu_times.push(us);
            }
        }
        writeln!(output, "{id}\t{repeat}\t{phase}\t{i}\t{rays}\t{:.3}\t{:.3}\t{:.3}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
            sample.encode_us, sample.submit_us, sample.complete_us, gpu_us.map_or_else(|| "NA".into(), |v| format!("{v:.3}")),
            sample.uploads, sample.downloads, sample.submissions, sample.waits, counts.allocations, counts.reallocations, counts.requested_bytes, sample.dispatches).unwrap();
    }
    let ci = metrics::bootstrap_p95(&latencies);
    assert_eq!(sensor.readback_stats().submissions, 0);
    if let Some(cloud) = &cloud {
        cloud.report();
    }
    if case.lifecycle != "replace" {
        assert_eq!(sensor.query_stats().scene_builds, 1);
        assert_eq!(sensor.query_stats().scene_preparations, 1);
        assert_eq!(sensor.query_stats().refit_preparations, 1);
    }
    eprintln!(
        "summary case={id} repeat={repeat} n={samples} complete_p50_us={:.3} complete_p95_us={:.3} p95_bootstrap95_us=[{:.3},{:.3}] encode_p50_us={:.3} gpu_p50_us={} rays_per_second={:.0} {} stats={:?}",
        metrics::quantile(&latencies, 50),
        metrics::quantile(&latencies, 95),
        ci.0,
        ci.1,
        metrics::quantile(&encode_times, 50),
        if gpu_times.is_empty() {
            "NA".into()
        } else {
            format!("{:.3}", metrics::quantile(&gpu_times, 50))
        },
        rays as f64 * samples as f64 * 1e6 / latencies.iter().sum::<f64>(),
        rss(),
        sensor.query_stats()
    );
    output.flush().unwrap();
}

fn main() {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let mut full = false;
    let mut cloud_matrix = false;
    let mut samples = 64usize;
    let mut warmup = 8usize;
    let mut repeats = 3usize;
    let mut timestamps = true;
    let mut filter = String::new();
    let mut destination = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--lidar-matrix" => full = true,
            "--cloud-matrix" => cloud_matrix = true,
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
            "--bench" => {}
            "--help" => {
                println!(
                    "--lidar-matrix | --cloud-matrix --samples N --warmup N --repeats N --filter SUBSTRING --output NEW.tsv --no-timestamps"
                );
                return;
            }
            option => panic!("unknown benchmark option: {option}"),
        }
        i += 1;
    }
    assert!((1..=10000).contains(&samples) && warmup <= 10000 && (1..=20).contains(&repeats));
    assert!(!(full && cloud_matrix), "select only one matrix");
    let selected: Vec<_> = if cloud_matrix {
        cloud_cases()
    } else {
        cases(full)
    }
    .into_iter()
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
    writeln!(output, "case\trepeat\tphase\tindex\trays\tencode_us\tsubmit_elapsed_us\tcomplete_us\tgpu_us\tdescriptor_pose_upload_bytes\tresult_download_bytes\tsubmissions\twaits\tallocations\treallocations\trequested_bytes\tdispatches").unwrap();
    let engine = timing::engine(timestamps);
    eprintln!(
        "adapter={:?} profile={} cases={} samples={samples} warmup={warmup} repeats={repeats} pid={}",
        engine.adapter_info(),
        if cfg!(debug_assertions) {
            "debug"
        } else {
            "release"
        },
        selected.len(),
        std::process::id()
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
