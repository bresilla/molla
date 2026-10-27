use super::*;
use crate::mpm::{MpmGridStorage, MpmUnilateralConfig};
use crate::soil_tractor::{SoilTractor, SoilTractorConfig, SoilTractorTransfer};
use std::path::Path;

const DT: f64 = 1.0 / 60.0;
const FRAMES: u64 = 3;
const PARTICLES: u32 = 50000;
const PUBLIC_BOUND: f64 = 2e-5;

fn coupling() -> MpmUnilateralConfig {
    MpmUnilateralConfig {
        node_capacity: 4096,
        search_distance: 0.05,
        max_iterations: 64,
    }
}

#[test]
fn admm_actual_tractor_contract() {
    let cfg = SoilTractorConfig::default();
    assert_eq!(cfg.particle_count().unwrap(), PARTICLES);
    assert_eq!(cfg.seed, [100, 10, 50]);
    assert_eq!(cfg.max_substep_s, 0.0005);
    assert_eq!((DT / cfg.max_substep_s).ceil() as usize, 34);
    assert_eq!(
        (cfg.grid_size, cfg.cell_size, cfg.youngs_pa),
        (64, 0.1, 3000.0)
    );
    assert_eq!(coupling().node_capacity, 4096);
    assert_eq!(coupling().max_iterations, 64);
    assert_eq!(PUBLIC_BOUND, 2e-5);
}

fn save_json(path: impl AsRef<Path>, value: &serde_json::Value) {
    std::fs::write(path, serde_json::to_vec_pretty(value).unwrap()).unwrap();
}

fn snapshot(tractor: &mut SoilTractor, engine: &Engine, root: &Path, frame: u64) -> Result<()> {
    let (ordinal, checkpoint) = if frame == 0 {
        assert!(tractor.read_soil_checkpoint(PARTICLES).is_err());
        let checkpoint = tractor
            .world_mut_for_test()
            .gpu
            .as_mut()
            .unwrap()
            .soil
            .as_mut()
            .unwrap()
            .soil
            .checkpoint_soil_with(engine.queue().unwrap())?;
        (0, checkpoint)
    } else {
        tractor.read_soil_checkpoint(PARTICLES)?
    };
    assert_eq!(ordinal, frame);
    assert_eq!(checkpoint.particle_count(), PARTICLES);
    std::fs::write(
        root.join(format!("soil-{frame}.bin")),
        checkpoint.to_bytes(),
    )
    .unwrap();
    let positions = tractor.read_soil_positions()?;
    let velocities = tractor
        .world_mut_for_test()
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .soil
        .read_velocities_with(engine.device().unwrap(), engine.queue().unwrap())?;
    assert_eq!(
        (positions.len(), velocities.len()),
        (PARTICLES as usize, PARTICLES as usize)
    );
    let mut values: Vec<f64> = positions
        .iter()
        .chain(&velocities)
        .flat_map(|p| p.iter().map(|&v| f64::from(v)))
        .collect();
    let handles = tractor.handles();
    for body in std::iter::once(handles.chassis)
        .chain(handles.wheels)
        .chain([handles.tine])
    {
        let pose = tractor.world().scene.body_pose(body)?;
        let velocity = tractor.world().scene.body_velocity(body)?;
        values.extend(pose.position.to_array());
        values.extend(pose.rotation.to_array());
        values.extend(velocity.linear.to_array());
        values.extend(velocity.angular.to_array());
    }
    assert_eq!(values.len(), PARTICLES as usize * 6 + 6 * 13);
    assert!(values.iter().all(|v| v.is_finite()));
    let bytes: Vec<_> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(root.join(format!("public-{frame}.bin")), bytes).unwrap();
    for (i, bytes) in super::super::motion_tests::retry_machine_words(tractor.world())
        .iter()
        .enumerate()
    {
        std::fs::write(root.join(format!("machine-{frame}-{i}.bin")), bytes).unwrap();
    }
    Ok(())
}

fn run_arm(root: &Path, native: bool) -> Result<()> {
    let start = std::time::Instant::now();
    let engine = Arc::new(Engine::new_gpu_with_features(
        wgpu::Features::SHADER_FLOAT32_ATOMIC
            | wgpu::Features::SHADER_F64
            | wgpu::Features::PASSTHROUGH_SHADERS
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
    )?);
    assert!(!engine.adapter_info().unwrap().is_software);
    let config = SoilTractorConfig::default();
    let transfer = SoilTractorTransfer::FixedOrder {
        max_gather_bytes: u64::from(PARTICLES) * 64,
    };
    let mut tractor = if native {
        SoilTractor::new_native_prediction_with_admm_transfer(
            engine.clone(),
            config,
            coupling(),
            0.3,
            MpmGridStorage::Dense,
            DT,
            transfer,
        )?
    } else {
        SoilTractor::new_native_explicit_with_paired_admm_transfer(
            engine.clone(),
            config,
            coupling(),
            0.3,
            molla_kernels::FeatherstoneExecution::Generic,
            MpmGridStorage::Dense,
            transfer,
        )?
    };
    assert_eq!(
        tractor.world().mpm_soil_admm_backend(),
        Some("experimental-admm-guarded")
    );
    assert_eq!(tractor.world().mpm_soil_newton_backend(), None);
    let preparation_backend = tractor.world().mpm_soil_admm_preparation_backend().unwrap();
    assert_eq!(
        preparation_backend,
        crate::soil_admm_geometry_gpu::ExperimentalSoilAdmmGeometryGpu::preparation_backend_label()
    );
    if let Ok(expected) = std::env::var("MOLLA_ADMM50K_PREPARATION") {
        assert_eq!(preparation_backend, expected);
    }
    assert_eq!(tractor.world().gpu_uses_native_prediction(), native);
    let info = tractor.world().mpm_soil_transfer_info().unwrap();
    assert!(info.fixed_order && info.particle_ordering && info.parallel_sort);
    assert_eq!(info.gather_bytes, 3_200_000);
    tractor.prepare_soil_frame_profiling_with_preparation(34)?;
    tractor.prepare_soil_frame_loads()?;
    if let Some(value) = std::env::var_os("MOLLA_ADMM50K_TRACE") {
        assert_eq!(value, "1");
        tractor
            .world_mut_for_test()
            .gpu
            .as_mut()
            .unwrap()
            .soil
            .as_mut()
            .unwrap()
            .projection
            .as_mut()
            .unwrap()
            .enable_admm_work_trace(engine.device().unwrap(), FRAMES as usize * 34)?;
    }
    let setup_ms = start.elapsed().as_secs_f64() * 1000.0;
    snapshot(&mut tractor, &engine, root, 0)?;
    let mut records = Vec::new();
    for frame in 1..=FRAMES {
        let allocations = molla_core::gpu_metrics::DeviceScope::start()?;
        let transfers = molla_core::gpu_metrics::TransferScope::start()?;
        let dispatches = molla_core::gpu_metrics::DispatchScope::start()?;
        let started = std::time::Instant::now();
        let result = tractor.step(DT, [300.0; 4]);
        let step_ms = started.elapsed().as_secs_f64() * 1000.0;
        let dispatches = dispatches.finish();
        let transfers = transfers.finish();
        let allocations = allocations.finish();
        if let Err(error) = result {
            save_json(
                root.join(format!("failure-{frame}.json")),
                &serde_json::json!({
                    "error":error.to_string(), "frame":frame, "step_ms":step_ms,
                    "allocations":allocations.observations, "dispatches":dispatches.encoded_dispatches,
                    "upload_bytes":transfers.upload_bytes, "readback_bytes":transfers.readback_bytes,
                    "committed_frames":tractor.world().mpm_soil_stats().unwrap().committed_frames,
                    "acceptance":false
                }),
            );
            return Err(error);
        }
        assert_eq!(allocations.observations, 0);
        assert!(!allocations.invalid && !transfers.overflowed && !dispatches.overflowed);
        let profile = tractor.read_soil_frame_profile()?.unwrap();
        assert_eq!(profile.committed_frame, frame);
        assert_eq!(profile.profile.particles, PARTICLES);
        assert_eq!(profile.profile.requested_substeps, 34);
        assert_eq!(profile.profile.contact_stages.len(), 34 * 3);
        assert_eq!(profile.profile.preparation_stages.len(), 34 * 8);
        assert!(profile.newton_work.is_empty());
        let soil_ms: f64 = profile.profile.stages.iter().map(|s| s.elapsed_ms).sum();
        let machine_ms: f64 = profile.machine_stages.iter().map(|s| s.elapsed_ms).sum();
        let stages = |v: &[crate::mpm::MpmStageTiming]| {
            v.iter().map(|s| serde_json::json!({
            "stage":s.stage.name(), "substep":s.substep, "start_ms":s.start_ms, "gpu_ms":s.elapsed_ms
        })).collect::<Vec<_>>()
        };
        let loads = tractor.read_soil_frame_loads()?;
        assert_eq!(loads.committed_frame, frame);
        assert_eq!(loads.bodies.len(), 5);
        let record = serde_json::json!({
            "frame":frame, "cold":frame==1, "step_wall_ms":step_ms, "soil_gpu_ms":soil_ms,
            "machine_gpu_ms":machine_ms, "allocations":allocations.observations,
            "upload_bytes":transfers.upload_bytes, "readback_bytes":transfers.readback_bytes,
            "dispatches":dispatches.encoded_dispatches, "soil_stages":stages(&profile.profile.stages),
            "contact_stages":stages(&profile.profile.contact_stages),
            "preparation_stages":stages(&profile.profile.preparation_stages),
            "machine_stages":profile.machine_stages.iter().map(|s| serde_json::json!({"stage":s.stage,
                "substep":s.substep,"start_ms":s.start_ms,"gpu_ms":s.elapsed_ms})).collect::<Vec<_>>(),
            "host_stages":profile.host_stage_ms,
            "loads":loads.bodies.iter().map(|b| serde_json::json!({"linear":b.linear,
                "angular":b.angular,"world_origin_angular":b.world_origin_angular})).collect::<Vec<_>>(),
            "retained_coupling_bytes":tractor.world().mpm_soil_stats().unwrap().retained_bytes
        });
        save_json(root.join(format!("frame-{frame}.json")), &record);
        records.push(record);
        snapshot(&mut tractor, &engine, root, frame)?;
        eprintln!(
            "ADMM50K native={native} frame={frame} soil_gpu_ms={soil_ms:.3} machine_gpu_ms={machine_ms:.3} wall_ms={step_ms:.3}"
        );
    }
    if let Some(trace) = &tractor
        .world_mut_for_test()
        .gpu
        .as_ref()
        .unwrap()
        .soil
        .as_ref()
        .unwrap()
        .projection
        .as_ref()
        .unwrap()
        .admm_work_trace
    {
        trace.save(&engine, root)?;
    }
    save_json(
        root.join("result.json"),
        &serde_json::json!({
            "native_prediction":native,"preparation_backend":preparation_backend,
            "particles":PARTICLES,"frames":FRAMES,"substeps":34,
            "dt":DT,"max_substep":config.max_substep_s,"setup_ms":setup_ms,"records":records,
            "public_bound":PUBLIC_BOUND,"soil_target_ms":4.0,"acceptance":false,
            "physically_calibrated":false,"scope":"three-frame actual workload; first frame cold; no percentile or long-horizon acceptance"
        }),
    );
    Ok(())
}

#[test]
#[ignore = "requires RTX GPU; actual 50k ADMM tractor, explicit arm and 180-second batch cap"]
fn admm_actual_tractor_50k_arm() {
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_ADMM50K_OUTPUT").unwrap());
    let native = match std::env::var("MOLLA_ADMM50K_ARM").unwrap().as_str() {
        "native" => true,
        "featherstone" => false,
        _ => panic!("explicit ADMM arm required"),
    };
    std::fs::create_dir(&root).unwrap();
    let result = run_arm(&root, native);
    if let Err(error) = &result {
        std::fs::write(root.join("error.txt"), error.to_string()).unwrap();
    }
    result.unwrap();
}

#[test]
#[ignore = "requires saved actual 50k trajectory artifacts; CPU-only comparison"]
fn admm_actual_tractor_compare() {
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_ADMM50K_OUTPUT").unwrap());
    let mut records = Vec::new();
    let mut passed = true;
    for frame in 0..=FRAMES {
        let read = |arm| {
            let bytes = std::fs::read(root.join(arm).join(format!("public-{frame}.bin"))).unwrap();
            assert_eq!(bytes.len(), (PARTICLES as usize * 6 + 78) * 8);
            bytes
                .chunks_exact(8)
                .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
                .collect::<Vec<_>>()
        };
        let a = read("native");
        let b = read("featherstone");
        let mut maximum = [0.0f64; 3];
        for (i, (a, b)) in a.into_iter().zip(b).enumerate() {
            let delta = (a - b).abs();
            assert!(delta.is_finite());
            let field = if i < PARTICLES as usize * 3 {
                0
            } else if i < PARTICLES as usize * 6 {
                1
            } else {
                2
            };
            maximum[field] = maximum[field].max(delta);
        }
        let valid = maximum.iter().all(|&v| v <= PUBLIC_BOUND);
        passed &= valid;
        records.push(
            serde_json::json!({"frame":frame,"maximum_absolute_delta":maximum,"passed":valid}),
        );
    }
    save_json(
        root.join("comparison.json"),
        &serde_json::json!({
            "fields":["soil_positions","soil_velocities","six_body_pose_velocity"],
            "records":records,"bound":PUBLIC_BOUND,"passed":passed,"acceptance":false
        }),
    );
    assert!(
        passed,
        "actual 50k native/Featherstone trajectory exceeds 2e-5; see comparison.json"
    );
}
