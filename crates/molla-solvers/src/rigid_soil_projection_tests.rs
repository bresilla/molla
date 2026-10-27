use super::*;

#[path = "rigid_soil_admm_tractor_tests.rs"]
mod admm_tractor;
#[path = "rigid_soil_admm_work_tests.rs"]
pub(crate) mod admm_work;
#[path = "rigid_soil_native_prediction_tests.rs"]
mod native_prediction;
#[path = "rigid_soil_newton_work_tests.rs"]
pub(crate) mod newton_work;
use crate::mpm::{MpmConfig, MpmSolidShape};
use molla_math::{GpuTransform, Mat3, Vec3};

#[cfg(all(feature = "publication-products-newton", not(target_arch = "wasm32")))]
#[path = "rigid_soil_ordered_tractor_tests.rs"]
mod ordered_tractor;

#[cfg(all(feature = "active-publication-newton", not(target_arch = "wasm32")))]
fn newton_shadow_frame_count(value: Option<&str>) -> Option<u32> {
    match value {
        None | Some("14") => Some(14),
        Some("33") => Some(33),
        _ => None,
    }
}

#[cfg(all(feature = "active-publication-newton", not(target_arch = "wasm32")))]
#[test]
fn newton_shadow_frame_count_contract() {
    assert_eq!(newton_shadow_frame_count(None), Some(14));
    assert_eq!(newton_shadow_frame_count(Some("14")), Some(14));
    assert_eq!(newton_shadow_frame_count(Some("33")), Some(33));
    for value in ["", "0", "1", "32", "34", "033", "33 ", "-1"] {
        assert_eq!(newton_shadow_frame_count(Some(value)), None);
    }
}

#[cfg(all(feature = "active-publication-newton", not(target_arch = "wasm32")))]
#[test]
#[ignore = "requires native/protected GPU, shadow=1 and fresh SOIL_NEWTON_SHADOW_DIR; 14 or 33 frame 50k diagnostic"]
fn newton_tractor_equal_input_publication_shadow() {
    use std::io::Write;
    assert_eq!(
        std::env::var("MOLLA_NEWTON_PUBLICATION_SHADOW").as_deref(),
        Ok("1")
    );
    assert!(!cfg!(feature = "warm-newton"));
    let frames =
        newton_shadow_frame_count(std::env::var("SOIL_NEWTON_SHADOW_FRAMES").ok().as_deref())
            .expect("SOIL_NEWTON_SHADOW_FRAMES must be 14 or 33");
    let root = std::path::PathBuf::from(std::env::var_os("SOIL_NEWTON_SHADOW_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC | wgpu::Features::PASSTHROUGH_SHADERS,
        )
        .unwrap(),
    );
    assert!(!engine.adapter_info().unwrap().is_software);
    let methods = if cfg!(feature = "publication-products-newton") {
        [21, 22]
    } else {
        [20, 21]
    };
    eprintln!(
        "constructing 50k tractor with equal-input method-{} shadow of method {}",
        methods[0], methods[1]
    );
    let mut tractor =
        crate::soil_tractor::SoilTractor::new_native_explicit_with_paired_newton_execution(
            engine.clone(),
            crate::soil_tractor::SoilTractorConfig::default(),
            crate::mpm::MpmUnilateralConfig {
                node_capacity: 4096,
                search_distance: 0.05,
                max_iterations: 64,
            },
            0.3,
            molla_kernels::FeatherstoneExecution::Generic,
        )
        .unwrap();
    let buffers = tractor
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
        .newton_shadow_buffers();
    let mut log = std::fs::File::create(root.join("frames.jsonl")).unwrap();
    let mut final_words = Vec::new();
    for frame in 0..frames {
        let step = tractor.step(1. / 60., [300.; 4]);
        let words: Vec<Vec<u32>> = buffers
            .iter()
            .map(|b| {
                let bytes = crate::soil_shared_wall_friction_gpu::capture::read(&engine, b);
                bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
            })
            .collect();
        let record = serde_json::json!({"frame":frame,"step_error":step.as_ref().err().map(ToString::to_string),"parity_headers":words});
        writeln!(log, "{record}").unwrap();
        log.flush().unwrap();
        step.unwrap();
        let stats = tractor.world().mpm_soil_stats().unwrap();
        assert_eq!(stats.committed_frames, u64::from(frame) + 1);
        assert!(!stats.faulted);
        for header in &words {
            assert_eq!(header[0], (frame + 1) * 17);
            assert!(header[8] > 0);
        }
        eprintln!(
            "shadow frame={frame} calls={:?} different={:?}",
            words.iter().map(|h| h[0]).collect::<Vec<_>>(),
            words.iter().map(|h| h[1]).collect::<Vec<_>>()
        );
        final_words = words;
        if final_words.iter().any(|h| h[1] != 0) {
            break;
        }
    }
    let exact =
        final_words.len() == 2 && final_words.iter().all(|h| h[0] == frames * 17 && h[1] == 0);
    std::fs::write(root.join("result.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "schema":1,"acceptance":false,"physical_calibration":"uncalibrated",
        "scope":"equal immutable input buffers per live substep; separate cold-start control shadow after candidate; shadow outputs never feed physics; not performance or full-trajectory acceptance",
        "particles":50000,"requested_frames":frames,"substeps_per_frame":34,"methods":methods,
        "buffer_order":["outputs","rates","publication","state","report"],
        "header_order":["calls","mismatched_calls","first_call","first_buffer","first_word","control_bits","candidate_bits","different_words","compared_words"],
        "parity_headers":final_words,"exact":exact,
    })).unwrap()).unwrap();
    assert!(
        exact,
        "equal-input solver outputs differ; retained first word and call in result.json"
    );
}

#[test]
#[ignore = "requires native GPU and a fresh SOIL_NEWTON_SUCCESS_DIR; actual 50k single-frame capture"]
fn newton_tractor_captures_successful_step() {
    assert_eq!(
        std::env::var("MOLLA_NEWTON_CAPTURE_SUCCESS").as_deref(),
        Ok("1")
    );
    let root = std::path::PathBuf::from(std::env::var_os("SOIL_NEWTON_SUCCESS_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    let mut features = wgpu::Features::SHADER_FLOAT32_ATOMIC;
    if cfg!(all(
        feature = "protected-newton",
        not(target_arch = "wasm32")
    )) {
        features |= wgpu::Features::PASSTHROUGH_SHADERS;
    }
    let engine = Arc::new(Engine::new_gpu_with_features(features).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    eprintln!("constructing actual paired native 50k tractor for successful Newton input capture");
    let mut tractor = crate::soil_tractor::SoilTractor::new_native_explicit_with_paired_newton(
        engine.clone(),
        crate::soil_tractor::SoilTractorConfig::default(),
        crate::mpm::MpmUnilateralConfig {
            node_capacity: 4096,
            search_distance: 0.05,
            max_iterations: 64,
        },
        0.3,
    )
    .unwrap();
    tractor.step(1. / 60., [300.; 4]).unwrap();
    let stats = tractor.world().mpm_soil_stats().unwrap();
    assert_eq!(stats.committed_frames, 1);
    assert!(!stats.faulted);
    let projection = tractor
        .world_mut_for_test()
        .gpu
        .as_ref()
        .unwrap()
        .soil
        .as_ref()
        .unwrap()
        .projection
        .as_ref()
        .unwrap();
    let seeds = projection.newton_success_seeds();
    assert!(
        seeds
            .iter()
            .all(|seed| seed.is_some() == cfg!(feature = "warm-newton"))
    );
    for (parity, buffer) in projection.newton_success_buffers().iter().enumerate() {
        crate::soil_newton_failure::capture_success_with_seed(
            &engine,
            buffer,
            seeds[parity].as_ref(),
            &root.join(format!("parity-{parity}")),
            |buffer| map_read_with_completion(engine.device().unwrap(), buffer, None),
        )
        .unwrap();
    }
    std::fs::write(root.join("scope.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema":1, "acceptance":false, "frame":0, "committed_frames":1,
        "scope":"last nontrivial published Newton solve per parity during first actual 50k frame; diagnostic capture work and readbacks exclude performance acceptance"
    })).unwrap()).unwrap();
}

#[test]
#[ignore = "requires native GPU and fresh SOIL_WALL_STAGE_TRACE; synchronized diagnostic only"]
fn wall_tractor_traces_shared_stages() {
    let root = std::path::PathBuf::from(
        std::env::var_os("SOIL_WALL_STAGE_TRACE").expect("fresh stage trace root required"),
    );
    std::fs::create_dir(&root).unwrap();
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    eprintln!("constructing actual native 50k tractor for synchronized shared-wall stage trace");
    let mut tractor = crate::soil_tractor::SoilTractor::new_native_explicit(
        engine.clone(),
        crate::soil_tractor::SoilTractorConfig::default(),
    )
    .unwrap();
    let mut coefficients: Vec<_> = tractor
        .handles()
        .wheels
        .into_iter()
        .map(|body| (body, 0.3))
        .collect();
    coefficients.push((tractor.handles().tine, 0.3));
    tractor
        .enable_wall_friction_coupling(
            crate::mpm::MpmUnilateralConfig {
                node_capacity: 4096,
                search_distance: 0.05,
                max_iterations: 64,
            },
            &coefficients,
        )
        .unwrap();
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
        .trace_wall_stages(&root);
    let result = tractor.step(1.0 / 60.0, [300.0; 4]);
    let stats = tractor.world().mpm_soil_stats().unwrap();
    let error = result.as_ref().err().map(ToString::to_string);
    std::fs::write(root.join("frame.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema": 1, "error": error, "committed_frames": stats.committed_frames,
        "rolled_back_frames": stats.rolled_back_frames, "faulted": stats.faulted,
        "acceptance": false, "scope": "Actual native 50k single-frame diagnostic with per-shared-stage synchronization and readback; no solver bypass; not production timing or acceptance"
    })).unwrap()).unwrap();
    eprintln!("synchronized shared-wall frame result={result:?}");
    if engine.check_device().is_err() {
        std::process::exit(76);
    }
    assert!(root.join("parity-0/solve-000/capture.json").exists());
    if result.is_ok() {
        assert_eq!(stats.committed_frames, 1);
    } else {
        assert_eq!(stats.committed_frames, 0);
    }
}

#[test]
#[ignore = "requires native GPU and SOIL_WALL_INPUT_CAPTURE; deliberate frame abort"]
fn wall_tractor_captures_first_shared_input() {
    let root = std::path::PathBuf::from(
        std::env::var_os("SOIL_WALL_INPUT_CAPTURE").expect("new capture directory required"),
    );
    std::fs::create_dir(&root).unwrap();
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    eprintln!("constructing native 50k tractor for first shared-wall input capture");
    let mut tractor = crate::soil_tractor::SoilTractor::new_native_explicit(
        engine.clone(),
        crate::soil_tractor::SoilTractorConfig::default(),
    )
    .unwrap();
    let mut coefficients: Vec<_> = tractor
        .handles()
        .wheels
        .into_iter()
        .map(|body| (body, 0.3))
        .collect();
    coefficients.push((tractor.handles().tine, 0.3));
    tractor
        .enable_wall_friction_coupling(
            crate::mpm::MpmUnilateralConfig {
                node_capacity: 4096,
                search_distance: 0.05,
                max_iterations: 64,
            },
            &coefficients,
        )
        .unwrap();
    let captures = tractor
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
        .capture_first_wall_input();
    let error = tractor
        .step(1.0 / 60.0, [300.0; 4])
        .expect_err("capture must deliberately abort publication");
    if engine.check_device().is_err() {
        std::fs::write(root.join("device-failure.txt"), error.to_string()).unwrap();
        eprintln!("terminal capture GPU failure: {error}");
        std::process::exit(76);
    }
    let mut saved = 0;
    for (parity, capture) in captures.iter().enumerate() {
        if capture.save(&engine, &root.join(format!("parity-{parity}"))) {
            saved += 1;
        }
    }
    let stats = tractor.world().mpm_soil_stats().unwrap();
    let simulated_seconds = tractor.telemetry().unwrap().simulated_seconds;
    std::fs::write(root.join("origin.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema": 1, "error": error.to_string(), "particles": 50000,
        "captured_parities": saved, "committed_frames": stats.committed_frames,
        "rolled_back_frames": stats.rolled_back_frames, "faulted": stats.faulted,
        "simulated_seconds": simulated_seconds,
        "native_atomics": true, "fixed_walls": true, "coefficient": 0.3,
        "scope": "Actual first shared-wall input; test-only solver bypass with deliberate rollback. Not successful physical trajectory or performance acceptance."
    })).unwrap()).unwrap();
    assert!(!stats.faulted);
    assert_eq!(stats.committed_frames, 0);
    assert_eq!(stats.rolled_back_frames, 1);
    assert_eq!(simulated_seconds, 0.0);
    assert!(saved > 0);
    eprintln!("captured first shared-wall input, parities={saved}, deliberate abort: {error}");
}

#[test]
#[ignore = "requires native GPU and a new SOIL_NATIVE_REPLAY_OUTPUT directory"]
fn unilateral_tractor_captures_evolved_native_p2g() {
    let output = std::path::PathBuf::from(
        std::env::var_os("SOIL_NATIVE_REPLAY_OUTPUT").expect("new output required"),
    );
    assert!(!output.exists());
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    eprintln!("constructing actual 50k tractor for evolved native P2G capture");
    let mut tractor = crate::soil_tractor::SoilTractor::new_native_explicit(
        engine.clone(),
        crate::soil_tractor::SoilTractorConfig::default(),
    )
    .unwrap();
    tractor
        .enable_unilateral_coupling(crate::mpm::MpmUnilateralConfig {
            node_capacity: 4096,
            search_distance: 0.05,
            max_iterations: 64,
        })
        .unwrap();
    for frame in 0..15 {
        tractor.step(1.0 / 60.0, [300.0; 4]).unwrap();
        eprintln!("native P2G capture committed frame {frame}");
    }
    let soil = &mut tractor
        .world_mut_for_test()
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .soil;
    assert_eq!(soil.particle_count(), 50_000);
    crate::mpm::native_replay::capture(soil, &engine, &output);
    std::fs::write(output.join("origin.json"), serde_json::to_vec_pretty(&serde_json::json!({"committed_frames":15,"simulated_seconds":0.25,"stage":"frame boundary before frame 15; not a captured first divergent substep","scope":"actual native unilateral 50k tractor; diagnostic only"})).unwrap()).unwrap();
}

#[test]
#[ignore = "requires native GPU, SOIL_NORMAL_REFERENCE_SHADER and full 50k diagnostic"]
fn unilateral_tractor_compares_every_normal_solve() {
    let source = std::fs::read_to_string(
        std::env::var_os("SOIL_NORMAL_REFERENCE_SHADER").expect("reference shader required"),
    )
    .unwrap();
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    eprintln!("constructing actual 50k tractor for same-input normal comparison");
    let mut tractor = crate::soil_tractor::SoilTractor::new_native_explicit(
        engine.clone(),
        crate::soil_tractor::SoilTractorConfig::default(),
    )
    .unwrap();
    tractor
        .enable_unilateral_coupling(crate::mpm::MpmUnilateralConfig {
            node_capacity: 4096,
            search_distance: 0.05,
            max_iterations: 64,
        })
        .unwrap();
    let counts = tractor
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
        .prepare_normal_reference_check(&source);
    for frame in 0..33 {
        tractor.step(1.0 / 60.0, [300.0; 4]).unwrap();
        let observed = counts.each_ref().map(|buffer| {
            crate::soil_normal_projection_gpu::replay::read_reference_counts(&engine, buffer)
        });
        eprintln!("normal shadow frame={frame} counts={observed:?}");
        assert_eq!(observed[0][0] + observed[1][0], (frame + 1) * 34);
        for (parity, counters) in observed.iter().enumerate() {
            assert_eq!(
                &counters[1..],
                &[0; 4],
                "normal output words differ at frame {frame}, parity {parity}"
            );
        }
    }
    eprintln!(
        "1122 same-input normal solves matched status/rate/impulse/velocity bits; not trajectory repeatability or performance acceptance"
    );
}

#[test]
#[ignore = "requires native GPU and SOIL_NORMAL_CAPTURE_DIR; captures expected workload failure"]
fn unilateral_tractor_captures_first_normal_failure() {
    let root = std::path::PathBuf::from(
        std::env::var_os("SOIL_NORMAL_CAPTURE_DIR").expect("new capture directory required"),
    );
    std::fs::create_dir(&root).unwrap();
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    eprintln!("constructing actual 50k tractor for first-local-failure capture");
    let mut tractor = crate::soil_tractor::SoilTractor::new_native_explicit(
        engine.clone(),
        crate::soil_tractor::SoilTractorConfig::default(),
    )
    .unwrap();
    tractor
        .enable_unilateral_coupling(crate::mpm::MpmUnilateralConfig {
            node_capacity: 4096,
            search_distance: 0.05,
            max_iterations: 64,
        })
        .unwrap();
    let captures = tractor
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
        .prepare_normal_failure_capture();
    for frame in 0..33 {
        if let Err(error) = tractor.step(1.0 / 60.0, [300.0; 4]) {
            let mut saved = 0;
            for (parity, capture) in captures.iter().enumerate() {
                if capture.save(&engine, &root.join(format!("parity-{parity}"))) {
                    saved += 1;
                }
            }
            std::fs::write(root.join("failure.json"),serde_json::to_vec_pretty(&serde_json::json!({"schema":1,"frame":frame,"error":error.to_string(),"captured_parities":saved,"scope":"test-only first-failure capture; not performance acceptance"})).unwrap()).unwrap();
            assert!(saved > 0, "no local normal fault captured: {error}");
            eprintln!("captured actual frame {frame} failure: {error}");
            return;
        }
        eprintln!("normal capture frame {frame} committed");
    }
    panic!("expected workload failure did not occur; no acceptance inferred");
}

#[test]
#[ignore = "requires real GPU native float atomics and a full 50k tractor diagnostic"]
fn tractor_boundary_projection_fixed_input_replay() {
    let engine =
        Arc::new(Engine::new_gpu_with_features(wgpu::Features::SHADER_FLOAT32_ATOMIC).unwrap());
    assert!(!engine.adapter_info().unwrap().is_software);
    if let Some(path) = std::env::var_os("SOIL_PROJECTION_CAPTURE_DIR") {
        std::fs::create_dir(path).unwrap();
    }
    let mut tractor = crate::soil_tractor::SoilTractor::new_native_explicit(
        engine.clone(),
        crate::soil_tractor::SoilTractorConfig::default(),
    )
    .unwrap();
    tractor.enable_mass_coupling(4096).unwrap();
    let mut captures = None;
    let mut maximum_ratio = 0.0f64;
    let mut boundary_nodes = 0;
    for frame in 0..32 {
        if frame == 24 {
            captures = Some(
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
                    .prepare_replay_capture(),
            );
        }
        tractor.step(1.0 / 60.0, [300.0; 4]).unwrap();
        if [24, 25, 31].contains(&frame) {
            for (state, capture) in captures.as_ref().unwrap().iter().enumerate() {
                let (ratio, boundaries) =
                    crate::soil_contact_projection_gpu::tests::replay::check_capture(
                        &engine,
                        capture,
                        &format!("frame-{frame}-state-{state}"),
                        true,
                    );
                maximum_ratio = maximum_ratio.max(ratio);
                boundary_nodes += boundaries;
            }
        }
    }
    println!("captured tractor projection worst tolerance ratio={maximum_ratio}");
    assert!(
        boundary_nodes > 0,
        "capture must include occupied boundaries"
    );
    assert!(
        maximum_ratio <= 1.0,
        "fixed-input node-order gate failed: {maximum_ratio}"
    );
}

fn machine_momentum_energy(scene: &RigidScene) -> (Vec3, f64) {
    let mut momentum = Vec3::ZERO;
    let mut energy = 0.0;
    for (body, _) in scene.bodies() {
        let mass = scene.body_mass_properties(body).unwrap();
        let velocity = scene.body_velocity(body).unwrap();
        let rotation = Mat3::from_quat(scene.body_pose(body).unwrap().rotation);
        momentum += velocity.linear * mass.mass;
        energy += 0.5 * mass.mass * velocity.linear.length_squared()
            + 0.5
                * velocity
                    .angular
                    .dot(rotation * mass.inertia * rotation.transpose() * velocity.angular);
    }
    (momentum, energy)
}

fn machine_angular_momentum(scene: &RigidScene) -> Vec3 {
    scene
        .bodies()
        .map(|(body, _)| {
            let mass = scene.body_mass_properties(body).unwrap();
            let velocity = scene.body_velocity(body).unwrap();
            let rotation = Mat3::from_quat(scene.body_pose(body).unwrap().rotation);
            rotation * mass.inertia * rotation.transpose() * velocity.angular
                + scene
                    .body_world_com(body)
                    .unwrap()
                    .cross(velocity.linear * mass.mass)
        })
        .sum()
}

fn checkpoint(world: &mut RigidWorld, engine: &Engine) -> Vec<u8> {
    world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .soil
        .checkpoint_soil_with(engine.queue().unwrap())
        .unwrap()
        .to_bytes()
}

#[test]
#[ignore = "requires real GPU native float atomics and timestamp support"]
fn projected_tractor_runtime_conserves_impulse_and_rolls_back() {
    projected_tractor_runtime(
        crate::kernels::FeatherstoneRounding::Storage,
        SoilBoundaryScan::Full,
    );
}

#[test]
#[ignore = "requires real GPU native float atomics and timestamp support"]
fn shared_rounding_projected_tractor_runtime_conserves_impulse_and_rolls_back() {
    projected_tractor_runtime(
        crate::kernels::FeatherstoneRounding::ExperimentalShared,
        SoilBoundaryScan::Full,
    );
}

#[test]
#[ignore = "requires real GPU native float atomics and timestamp support"]
fn cached_boundary_projected_tractor_runtime_conserves_impulse_and_rolls_back() {
    projected_tractor_runtime(
        crate::kernels::FeatherstoneRounding::Storage,
        SoilBoundaryScan::ExperimentalCached,
    );
}

fn projected_tractor_runtime(
    rounding: crate::kernels::FeatherstoneRounding,
    boundary_scan: SoilBoundaryScan,
) {
    projected_tractor_runtime_with_execution(
        rounding,
        boundary_scan,
        crate::kernels::FeatherstoneExecution::Generic,
    );
}

#[test]
#[ignore = "requires real GPU native float atomics and timestamp support"]
fn shared_scratch_projected_tractor_runtime_conserves_impulse_and_rolls_back() {
    projected_tractor_runtime_with_execution(
        crate::kernels::FeatherstoneRounding::Storage,
        SoilBoundaryScan::Full,
        crate::kernels::FeatherstoneExecution::ExperimentalWorkgroupScratch,
    );
}

#[test]
#[ignore = "requires real GPU native float atomics and timestamp support"]
fn constraint_scratch_projected_tractor_runtime_conserves_impulse_and_rolls_back() {
    projected_tractor_runtime_with_execution(
        crate::kernels::FeatherstoneRounding::Storage,
        SoilBoundaryScan::Full,
        crate::kernels::FeatherstoneExecution::ExperimentalConstraintScratch,
    );
}

fn projected_tractor_runtime_with_execution(
    rounding: crate::kernels::FeatherstoneRounding,
    boundary_scan: SoilBoundaryScan,
    execution: crate::kernels::FeatherstoneExecution,
) {
    projected_tractor_runtime_mode(rounding, boundary_scan, execution, None, false, false);
}

#[test]
#[ignore = "requires real GPU native float atomics and cold articulated pipelines"]
fn unilateral_tractor_runtime_conserves_impulse_and_rolls_back() {
    projected_tractor_runtime_mode(
        crate::kernels::FeatherstoneRounding::Storage,
        SoilBoundaryScan::Full,
        crate::kernels::FeatherstoneExecution::Generic,
        Some(crate::mpm::MpmUnilateralConfig {
            node_capacity: 512,
            search_distance: 0.05,
            max_iterations: 64,
        }),
        false,
        false,
    );
}

#[test]
#[ignore = "requires real GPU native float atomics and cold articulated pipelines"]
fn friction_tractor_runtime_conserves_impulse_and_rolls_back() {
    projected_tractor_runtime_mode(
        crate::kernels::FeatherstoneRounding::Storage,
        SoilBoundaryScan::Full,
        crate::kernels::FeatherstoneExecution::Generic,
        Some(crate::mpm::MpmUnilateralConfig {
            node_capacity: 512,
            search_distance: 0.05,
            max_iterations: 64,
        }),
        true,
        false,
    );
}

#[test]
#[ignore = "requires real GPU native float atomics and cold articulated pipelines"]
fn single_lane_friction_tractor_runtime_conserves_impulse_and_rolls_back() {
    projected_tractor_runtime_mode(
        crate::kernels::FeatherstoneRounding::Storage,
        SoilBoundaryScan::Full,
        crate::kernels::FeatherstoneExecution::ExperimentalSingleLane,
        Some(crate::mpm::MpmUnilateralConfig {
            node_capacity: 512,
            search_distance: 0.05,
            max_iterations: 64,
        }),
        true,
        false,
    );
}

#[test]
#[ignore = "requires real GPU native atomics; specialized joint-type runtime"]
fn joint_types_friction_tractor_runtime_conserves_impulse_and_rolls_back() {
    projected_tractor_runtime_mode(
        crate::kernels::FeatherstoneRounding::Storage,
        SoilBoundaryScan::Full,
        crate::kernels::FeatherstoneExecution::ExperimentalJointTypes,
        Some(crate::mpm::MpmUnilateralConfig {
            node_capacity: 512,
            search_distance: 0.05,
            max_iterations: 64,
        }),
        true,
        false,
    );
}

#[test]
#[ignore = "requires real native GPU; compact articulated friction runtime"]
fn compact_joint_types_friction_tractor_runtime() {
    projected_tractor_runtime_layout(
        crate::kernels::FeatherstoneRounding::Storage,
        SoilBoundaryScan::Full,
        crate::kernels::FeatherstoneExecution::ExperimentalJointTypes,
        Some(crate::mpm::MpmUnilateralConfig {
            node_capacity: 512,
            search_distance: 0.05,
            max_iterations: 64,
        }),
        true,
        false,
        true,
    );
}

#[test]
#[ignore = "requires real GPU native float atomics and cold articulated pipelines"]
fn wall_friction_tractor_runtime_conserves_impulse_and_rolls_back() {
    projected_tractor_runtime_mode(
        crate::kernels::FeatherstoneRounding::Storage,
        SoilBoundaryScan::Full,
        crate::kernels::FeatherstoneExecution::Generic,
        Some(crate::mpm::MpmUnilateralConfig {
            node_capacity: 512,
            search_distance: 0.05,
            max_iterations: 64,
        }),
        true,
        true,
    );
}

fn wall_runtime_read(engine: &Engine, buffer: &wgpu::Buffer) -> Vec<u8> {
    let device = engine.device().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("wall runtime trace"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, buffer.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(15)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(15))
        .unwrap()
        .unwrap();
    let bytes = staging.slice(..).get_mapped_range().to_vec();
    staging.unmap();
    bytes
}

fn check_body_render(world: &RigidWorld, engine: &Engine, frame: u64, side: usize) {
    let view = world.gpu_body_render_view().unwrap();
    assert_eq!(view.committed_frame(), frame);
    assert_eq!(view.active_side(), side);
    assert_eq!(view.body_count(), world.scene.model().body_count);
    let bytes = wall_runtime_read(engine, view.buffers()[side]);
    let expected: Vec<GpuTransform> = world
        .scene
        .state()
        .body_q
        .host()
        .unwrap()
        .iter()
        .copied()
        .map(GpuTransform::from)
        .collect();
    assert_eq!(
        &bytes[..expected.len() * 32],
        bytemuck::cast_slice::<GpuTransform, u8>(&expected)
    );
}

fn wall_runtime_step(
    world: &mut RigidWorld,
    engine: &Engine,
    dt: f64,
    fixed_walls: bool,
) -> Result<()> {
    let result = world.step(dt);
    if result.is_err() && fixed_walls {
        if let Some(path) = std::env::var_os("SOIL_WALL_RUNTIME_CAPTURE") {
            let path = std::path::PathBuf::from(path);
            std::fs::create_dir(&path).unwrap();
            let projection = world
                .gpu
                .as_ref()
                .unwrap()
                .soil
                .as_ref()
                .unwrap()
                .projection
                .as_ref()
                .unwrap();
            for (parity, trace) in projection.wall_trace().iter().enumerate() {
                for (name, buffer) in ["velocity", "contact", "wall", "conditional", "status"]
                    .iter()
                    .zip(trace)
                {
                    let bytes = wall_runtime_read(engine, buffer);
                    std::fs::write(path.join(format!("parity-{parity}-{name}.bin")), bytes)
                        .unwrap();
                }
            }
            std::fs::write(path.join("SCOPE.md"),"Post-error retained contact outputs and last conditional inputs, after frame rollback. Not a full pre-solve generalized-state replay.\n").unwrap();
            eprintln!("wall runtime failure capture: {}", path.display());
        }
    }
    result
}

fn enable_projection_test(
    world: &mut RigidWorld,
    nodes: u32,
    scan: SoilBoundaryScan,
    unilateral: Option<crate::mpm::MpmUnilateralConfig>,
    friction: bool,
    fixed_walls: bool,
) -> Result<()> {
    if let Some(mut config) = unilateral {
        config.node_capacity = nodes;
        if friction {
            let mut coefficients: Vec<_> = world
                .gpu
                .as_ref()
                .unwrap()
                .soil
                .as_ref()
                .unwrap()
                .bindings
                .iter()
                .enumerate()
                .map(|(i, binding)| (binding.body, 0.2 + 0.1 * i as f32))
                .collect();
            coefficients.reverse();
            if fixed_walls {
                world.enable_mpm_soil_wall_friction_coupling(config, &coefficients)
            } else {
                world.enable_mpm_soil_friction_coupling(config, &coefficients)
            }
        } else {
            world.enable_mpm_soil_unilateral_coupling(config)
        }
    } else {
        world.enable_mpm_soil_mass_coupling_with_boundary_scan(nodes, scan)
    }
}

fn projected_tractor_runtime_mode(
    rounding: crate::kernels::FeatherstoneRounding,
    boundary_scan: SoilBoundaryScan,
    execution: crate::kernels::FeatherstoneExecution,
    unilateral: Option<crate::mpm::MpmUnilateralConfig>,
    friction: bool,
    fixed_walls: bool,
) {
    projected_tractor_runtime_layout(
        rounding,
        boundary_scan,
        execution,
        unilateral,
        friction,
        fixed_walls,
        false,
    );
}

fn projected_tractor_runtime_layout(
    rounding: crate::kernels::FeatherstoneRounding,
    boundary_scan: SoilBoundaryScan,
    execution: crate::kernels::FeatherstoneExecution,
    unilateral: Option<crate::mpm::MpmUnilateralConfig>,
    friction: bool,
    fixed_walls: bool,
    compact: bool,
) {
    use crate::soil_tractor::{
        TINE_HALF_EXTENTS, WHEEL_HALF_WIDTH, WHEEL_RADIUS, build_soil_tractor_scene,
    };
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC
                | wgpu::Features::TIMESTAMP_QUERY
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
                | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        )
        .unwrap(),
    );
    assert!(!engine.adapter_info().unwrap().is_software);
    let create_soil = |cfg| {
        if compact {
            MpmSimulation::new_native_sparse(&engine, cfg, 1024)
        } else {
            MpmSimulation::new_native_explicit(&engine, cfg)
        }
        .unwrap()
    };
    let (mut scene, handles) = build_soil_tractor_scene().unwrap();
    scene.set_gravity(Vec3::ZERO).unwrap();
    let bindings: Vec<_> = handles
        .wheels
        .into_iter()
        .map(|body| MpmRigidDeviceBinding {
            body,
            shape: MpmSolidShape::Cylinder {
                radius: WHEEL_RADIUS as f32,
                half_width: WHEEL_HALF_WIDTH as f32,
            },
        })
        .chain([MpmRigidDeviceBinding {
            body: handles.tine,
            shape: MpmSolidShape::Box {
                half_extents: TINE_HALF_EXTENTS.as_vec3().to_array(),
            },
        }])
        .collect();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 48;
    cfg.grid_origin = [-3.0, -0.5, -3.0];
    cfg.cell_size = 0.125;
    cfg.max_particles = 65;
    cfg.max_rigid = 5;
    cfg.gravity_y = 0.0;
    cfg.youngs = 100.0;
    cfg.particle_mass = 0.5 / 65.0;
    cfg.particle_volume = cfg.particle_mass / 1600.0;
    let boundary_cfg = cfg.clone();
    let initial_velocity = if friction {
        Vec3::new(1.0, 0.0, 0.5)
    } else {
        Vec3::X
    };
    let particle_mass = f64::from(cfg.particle_mass);
    let mut soil = create_soil(cfg);
    assert_eq!(soil.active_grid_enabled(), compact);
    println!(
        "tractor soil compact={compact} grid_bytes={}",
        soil.grid_storage_bytes()
    );
    soil.set_particles_with_velocity(
        engine.queue().unwrap(),
        &vec![[-1.45, 0.0, 0.0]; 65],
        &vec![initial_velocity.as_vec3().to_array(); 65],
    )
    .unwrap();
    println!("projected runtime: constructing actual six-body tractor");
    let mut world = RigidWorld::from_scene_gpu_with_machine_options(
        scene,
        engine.clone(),
        GpuContactCapacity::default(),
        rounding,
        execution,
    )
    .unwrap();
    world.solver.substeps = 3;
    world
        .attach_mpm_soil(soil, bindings.clone(), 32 * 1024 * 1024)
        .unwrap();
    println!("projected runtime: machine ready; preparing mass coupling");
    let (render_frame, render_view) = world.mpm_soil_particle_render_view().unwrap();
    assert_eq!((render_frame, render_view.particle_count()), (0, 65));
    let render_source = render_view.source();
    check_body_render(&world, &engine, 0, 0);
    let render_buffers = world
        .gpu_body_render_view()
        .unwrap()
        .buffers()
        .map(Clone::clone);
    world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .stats
        .faulted = true;
    assert!(world.mpm_soil_particle_render_view().is_err());
    assert!(world.gpu_body_render_view().is_err());
    world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .stats
        .faulted = false;
    assert_eq!(world.mpm_soil_boundary_scan(), None);
    if friction {
        let valid: Vec<_> = bindings
            .iter()
            .map(|binding| (binding.body, 0.5f32))
            .collect();
        let mut duplicate = valid.clone();
        duplicate[0] = valid[1];
        let mut foreign = valid.clone();
        foreign[0].0 = handles.chassis;
        let mut invalid_values = vec![
            vec![],
            valid[..valid.len() - 1].to_vec(),
            duplicate,
            foreign,
        ];
        for value in [-1.0, f32::NAN, f32::INFINITY] {
            let mut invalid = valid.clone();
            invalid[0].1 = value;
            invalid_values.push(invalid);
        }
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        for coefficients in invalid_values {
            assert!(
                world
                    .enable_mpm_soil_friction_mode(unilateral.unwrap(), &coefficients, fixed_walls)
                    .is_err()
            );
        }
        assert_eq!(allocations.finish().observations, 0);
        assert_eq!(world.mpm_soil_friction_coefficients(), None);
    }
    enable_projection_test(
        &mut world,
        512,
        boundary_scan,
        unilateral,
        friction,
        fixed_walls,
    )
    .unwrap();
    assert_eq!(
        world.mpm_soil_boundary_scan(),
        unilateral.is_none().then_some(boundary_scan)
    );
    assert_eq!(
        world.mpm_soil_unilateral_config(),
        if friction { None } else { unilateral }
    );
    assert_eq!(
        world.mpm_soil_friction_config(),
        if friction { unilateral } else { None }
    );
    assert_eq!(
        world.mpm_soil_friction_fixed_walls(),
        friction.then_some(fixed_walls)
    );
    assert_eq!(
        world.mpm_soil_friction_coefficients(),
        friction.then(|| bindings
            .iter()
            .enumerate()
            .map(|(i, b)| (b.body, 0.2 + 0.1 * i as f32))
            .collect())
    );
    assert!(
        world
            .enable_mpm_soil_mass_coupling_with_boundary_scan(512, SoilBoundaryScan::Full)
            .is_err()
    );
    {
        let soil = world.gpu.as_mut().unwrap().soil.as_mut().unwrap();
        assert_eq!(
            soil.projection.as_ref().unwrap().boundary_scan(),
            boundary_scan
        );
        let bytes = soil.projection.take().unwrap().bytes();
        soil.stats.retained_bytes -= bytes;
    }
    world.prepare_mpm_soil_frame_loads().unwrap();
    world.prepare_mpm_soil_frame_profiling(3).unwrap();
    assert!(world.read_mpm_soil_contact_motion().is_err());
    let mut observed_contact_nodes = 0u64;
    let mut tangential_impulse = 0.0f64;
    let dt = 0.0003;
    for frame in 1..=3 {
        let before = machine_momentum_energy(&world.scene).0;
        let angular_before = machine_angular_momentum(&world.scene);
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        wall_runtime_step(
            &mut world,
            &engine,
            if unilateral.is_some() && frame == 2 {
                dt * 0.5
            } else {
                dt
            },
            fixed_walls,
        )
        .unwrap();
        assert_eq!(
            world
                .gpu
                .as_ref()
                .unwrap()
                .soil
                .as_ref()
                .unwrap()
                .projection
                .as_ref()
                .unwrap()
                .boundary_scan(),
            boundary_scan
        );
        let allocations = allocations.finish();
        check_body_render(&world, &engine, frame, 1);
        if frame > 1 {
            assert_eq!(allocations.observations, 0);
        }
        let (after, mut energy) = machine_momentum_energy(&world.scene);
        let angular_after = machine_angular_momentum(&world.scene);
        let loads = world.read_mpm_soil_frame_loads().unwrap();
        assert_eq!(loads.committed_frame, frame);
        if unilateral.is_some() {
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let motion = world.read_mpm_soil_contact_motion().unwrap();
            assert_eq!(allocations.finish().observations, 0);
            let transfers = transfers.finish();
            assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 240));
            assert_eq!(motion.committed_frame, frame);
            assert_eq!(motion.bodies.len(), bindings.len());
            let expected_dt = if frame == 2 { dt * 0.5 / 3.0 } else { dt / 3.0 };
            assert_eq!(motion.substep_dt, expected_dt as f32);
            for (sample, binding) in motion.bodies.iter().zip(&bindings) {
                assert_eq!(sample.body, binding.body);
                observed_contact_nodes += u64::from(sample.motion.supporting_nodes);
                assert_eq!(
                    sample.motion.soil_velocity.is_some(),
                    sample.motion.supporting_nodes > 0
                );
            }
            println!("contact motion frame={frame}: {:?}", motion.bodies);
        } else {
            assert!(world.read_mpm_soil_contact_motion().is_err());
        }
        let impulse: Vec3 = loads
            .bodies
            .iter()
            .map(|b| Vec3::from_array(b.linear.map(f64::from)))
            .sum();
        let angular_impulse: Vec3 = loads
            .bodies
            .iter()
            .map(|b| Vec3::from_array(b.world_origin_angular.map(f64::from)))
            .sum();
        let angular_residual = angular_after - angular_before - angular_impulse;
        println!("frame {frame} world-origin machine angular residual {angular_residual:?}");
        assert!(
            angular_residual.length() < 2e-5,
            "frame {frame} angular residual {angular_residual:?}"
        );
        if friction {
            tangential_impulse += impulse.z.abs();
        }
        assert!(
            (after - before - impulse).length() < 2e-5,
            "frame {frame} impulse residual {:?}",
            after - before - impulse
        );
        let velocities = world
            .gpu
            .as_mut()
            .unwrap()
            .soil
            .as_mut()
            .unwrap()
            .soil
            .read_velocities_with(engine.device().unwrap(), engine.queue().unwrap())
            .unwrap();
        let mut total = after;
        for v in velocities {
            let v = Vec3::from_array(v.map(f64::from));
            total += v * particle_mass;
            energy += 0.5 * particle_mass * v.length_squared();
        }
        assert!(
            (total - initial_velocity * (particle_mass * 65.0)).length() < 2e-4,
            "combined momentum {total:?}"
        );
        assert!(
            energy <= 0.5 * particle_mass * 65.0 * initial_velocity.length_squared() + 1e-4,
            "unforced energy {energy}"
        );
        let profile = world.read_mpm_soil_frame_profile().unwrap().unwrap();
        assert_eq!(
            profile
                .machine_stages
                .iter()
                .filter(|s| s.stage == "soil-corrected-drift")
                .count(),
            3
        );
        assert_eq!(
            profile
                .machine_stages
                .iter()
                .filter(|s| s.stage == "momentum")
                .count(),
            3
        );
        assert_eq!(world.gpu_diagnostics().unwrap().last_substeps, 3);
        println!(
            "projected runtime frame {frame}: total momentum={total:?}, kinetic energy={energy:.9}, allocations={}",
            allocations.observations
        );
    }
    if friction {
        assert!(
            tangential_impulse > 1e-6,
            "missing consumed tangential machine impulse"
        );
        println!("friction runtime: consumed transverse impulse={tangential_impulse}");
    }
    let saved = checkpoint(&mut world, &engine);
    let pose = world.scene.body_pose(handles.chassis).unwrap();
    let velocity = world.scene.body_velocity(handles.chassis).unwrap();
    if unilateral.is_some() {
        assert!(observed_contact_nodes > 0);
    }
    let committed_loads = world.read_mpm_soil_frame_loads().unwrap();
    world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .fail_after_steps = Some(2);
    let error = world.step(dt).unwrap_err();
    assert!(error.to_string().contains("injected failure"), "{error}");
    assert!(world.gpu_body_render_view().is_err());
    assert!(world.read_mpm_soil_contact_motion().is_err());
    assert_eq!(checkpoint(&mut world, &engine), saved);
    assert_eq!(world.scene.body_pose(handles.chassis).unwrap(), pose);
    assert_eq!(
        world.scene.body_velocity(handles.chassis).unwrap(),
        velocity
    );
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 3);
    assert_eq!(world.mpm_soil_stats().unwrap().rolled_back_frames, 1);
    let (render_frame, render_view) = world.mpm_soil_particle_render_view().unwrap();
    assert_eq!((render_frame, render_view.particle_count()), (3, 65));
    assert!(render_view.matches_source(&render_source));
    assert_eq!(
        world.read_mpm_soil_frame_loads().unwrap().committed_frame,
        committed_loads.committed_frame
    );
    world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .fail_after_steps = None;
    world.step(dt).unwrap();
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 4);
    check_body_render(&world, &engine, 4, 1);
    if unilateral.is_some() {
        assert_eq!(
            world
                .read_mpm_soil_contact_motion()
                .unwrap()
                .committed_frame,
            4
        );
    }
    assert_eq!(world.gpu.as_ref().unwrap().rounding, rounding);
    assert_eq!(world.gpu.as_ref().unwrap().execution, execution);
    assert_eq!(world.solver.gpu_execution(), Some(execution));
    let saved = checkpoint(&mut world, &engine);
    let pose = world.scene.body_pose(handles.chassis).unwrap();
    let velocity = world.scene.body_velocity(handles.chassis).unwrap();
    let original = world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .soil
        .inject_late_particle_fault();
    let error = world.step(dt).unwrap_err();
    assert!(error.to_string().contains("particle-grid-fault"), "{error}");
    assert!(world.gpu_body_render_view().is_err());
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 4);
    assert_eq!(world.mpm_soil_stats().unwrap().rolled_back_frames, 2);
    assert_eq!(checkpoint(&mut world, &engine), saved);
    assert_eq!(world.scene.body_pose(handles.chassis).unwrap(), pose);
    assert_eq!(
        world.scene.body_velocity(handles.chassis).unwrap(),
        velocity
    );
    world
        .gpu
        .as_mut()
        .unwrap()
        .soil
        .as_mut()
        .unwrap()
        .soil
        .restore_test_gather(original);
    world.step(dt).unwrap();
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 5);
    check_body_render(&world, &engine, 5, 1);
    assert_eq!(
        world.gpu_body_render_view().unwrap().buffers(),
        [&render_buffers[0], &render_buffers[1]]
    );
    println!(
        "actual tractor: late particle fault rejected, exact soil/chassis rollback, valid retry committed"
    );
    let soil = world.detach_mpm_soil().unwrap().unwrap();
    assert_eq!(world.mpm_soil_boundary_scan(), None);
    world
        .attach_mpm_soil(soil, bindings.clone(), 32 * 1024 * 1024)
        .unwrap();
    let retained = world.mpm_soil_stats().unwrap().retained_bytes;
    world.gpu.as_mut().unwrap().soil.as_mut().unwrap().max_bytes = retained;
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        enable_projection_test(
            &mut world,
            512,
            boundary_scan,
            unilateral,
            friction,
            fixed_walls
        )
        .is_err()
    );
    assert_eq!(world.mpm_soil_boundary_scan(), None);
    assert_eq!(allocations.finish().observations, 0);
    world.gpu.as_mut().unwrap().soil.as_mut().unwrap().max_bytes = 32 * 1024 * 1024;
    enable_projection_test(
        &mut world,
        1,
        boundary_scan,
        unilateral,
        friction,
        fixed_walls,
    )
    .unwrap();
    let saved = checkpoint(&mut world, &engine);
    let pose = world.scene.body_pose(handles.chassis).unwrap();
    let error = world.step(dt).unwrap_err();
    assert!(error.to_string().contains("solver fault 32"), "{error}");
    assert_eq!(checkpoint(&mut world, &engine), saved);
    assert_eq!(world.scene.body_pose(handles.chassis).unwrap(), pose);
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 0);
    assert!(!world.mpm_soil_stats().unwrap().faulted);
    let mut soil = world.detach_mpm_soil().unwrap().unwrap();
    if friction {
        let mut cfg = boundary_cfg;
        cfg.grid_origin[2] = if fixed_walls {
            -2.0 * cfg.cell_size
        } else {
            0.0
        };
        let mut boundary = create_soil(cfg);
        boundary
            .set_particles_with_velocity(
                engine.queue().unwrap(),
                &vec![[-1.45, 0.0, 0.0]; 65],
                &vec![initial_velocity.as_vec3().to_array(); 65],
            )
            .unwrap();
        world
            .attach_mpm_soil(boundary, bindings.clone(), 32 * 1024 * 1024)
            .unwrap();
        enable_projection_test(
            &mut world,
            512,
            boundary_scan,
            unilateral,
            true,
            fixed_walls,
        )
        .unwrap();
        if fixed_walls {
            world.prepare_mpm_soil_frame_loads().unwrap();
            for _ in 0..2 {
                let before = machine_momentum_energy(&world.scene).0;
                wall_runtime_step(&mut world, &engine, dt, fixed_walls).unwrap();
                let after = machine_momentum_energy(&world.scene).0;
                let impulse: Vec3 = world
                    .read_mpm_soil_frame_loads()
                    .unwrap()
                    .bodies
                    .iter()
                    .map(|body| Vec3::from_array(body.linear.map(f64::from)))
                    .sum();
                assert!((after - before - impulse).length() < 2e-5);
                assert!(
                    world
                        .read_mpm_soil_contact_motion()
                        .unwrap()
                        .bodies
                        .iter()
                        .any(|b| b.motion.supporting_nodes > 0)
                );
                let positions = world.read_mpm_soil_positions().unwrap();
                assert!(positions.iter().all(|p| p[2] >= -1e-6 && p[2] < 0.01));
            }
            let traces = world
                .gpu
                .as_ref()
                .unwrap()
                .soil
                .as_ref()
                .unwrap()
                .projection
                .as_ref()
                .unwrap()
                .wall_trace();
            let mut boundary_nodes = 0;
            for trace in &traces {
                let bytes = wall_runtime_read(&engine, &trace[3]);
                for chunk in bytes.chunks_exact(112) {
                    let input = bytemuck::pod_read_unaligned::<
                        crate::soil_wall_friction_gpu::SoilWallFrictionInput,
                    >(chunk);
                    if input.soil_velocity_mass[3] > 0.0 && input.boundary[0] & 16 != 0 {
                        boundary_nodes += 1;
                    }
                }
            }
            assert!(boundary_nodes > 0, "no lower-Z wall contacts exercised");
            println!(
                "wall runtime: {boundary_nodes} retained lower-Z conditional contacts across state parities"
            );
            let saved = checkpoint(&mut world, &engine);
            let pose = world.scene.body_pose(handles.chassis).unwrap();
            world
                .gpu
                .as_mut()
                .unwrap()
                .soil
                .as_mut()
                .unwrap()
                .fail_after_steps = Some(2);
            assert!(
                world
                    .step(dt)
                    .unwrap_err()
                    .to_string()
                    .contains("injected failure")
            );
            assert_eq!(checkpoint(&mut world, &engine), saved);
            assert_eq!(world.scene.body_pose(handles.chassis).unwrap(), pose);
            world
                .gpu
                .as_mut()
                .unwrap()
                .soil
                .as_mut()
                .unwrap()
                .fail_after_steps = None;
            world.step(dt).unwrap();
            assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 3);
            assert_eq!(world.mpm_soil_stats().unwrap().rolled_back_frames, 1);
            println!(
                "wall friction runtime: boundary frames, machine contact impulse and rollback/resume pass"
            );
        } else {
            let saved = checkpoint(&mut world, &engine);
            let pose = world.scene.body_pose(handles.chassis).unwrap();
            let velocity = world.scene.body_velocity(handles.chassis).unwrap();
            let error = world.step(dt).unwrap_err();
            assert!(
                error
                    .to_string()
                    .contains("fixed-wall-friction-unsupported"),
                "{error}"
            );
            assert_eq!(checkpoint(&mut world, &engine), saved);
            assert_eq!(world.scene.body_pose(handles.chassis).unwrap(), pose);
            assert_eq!(
                world.scene.body_velocity(handles.chassis).unwrap(),
                velocity
            );
            assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 0);
            assert_eq!(world.mpm_soil_stats().unwrap().rolled_back_frames, 1);
            assert!(!world.mpm_soil_stats().unwrap().faulted);
            assert!(world.read_mpm_soil_contact_motion().is_err());
        }
        world.detach_mpm_soil().unwrap().unwrap();
        if !fixed_walls {
            println!("friction runtime: fixed-wall rejection rolled back machine and particles");
        }
    }
    soil.set_particles(engine.queue().unwrap(), &[]).unwrap();
    world
        .attach_mpm_soil(soil, bindings, 32 * 1024 * 1024)
        .unwrap();
    enable_projection_test(
        &mut world,
        512,
        SoilBoundaryScan::Full,
        unilateral,
        friction,
        fixed_walls,
    )
    .unwrap();
    assert_eq!(
        world.mpm_soil_boundary_scan(),
        unilateral.is_none().then_some(SoilBoundaryScan::Full)
    );
    world.step(dt).unwrap();
    assert_eq!(world.read_mpm_soil_positions().unwrap().len(), 0);
    assert!(world.read_mpm_soil_contact_motion().is_err());
    assert_eq!(world.mpm_soil_stats().unwrap().committed_frames, 1);
    world.detach_mpm_soil().unwrap().unwrap();
    let compilations = world.gpu_diagnostics().unwrap().model_compilations;
    let before_rebuild = world
        .gpu_body_render_view()
        .unwrap()
        .buffers()
        .map(Clone::clone);
    let _ = world.scene.state_mut();
    assert!(world.gpu_body_render_view().is_err());
    let mut mass = world.scene.body_mass_properties(handles.chassis).unwrap();
    mass.mass += 1.0;
    world
        .scene
        .set_body_mass_properties(handles.chassis, mass)
        .unwrap();
    assert!(world.gpu_body_render_view().is_err());
    world.solver.substeps = 2;
    world.step(dt).unwrap();
    check_body_render(
        &world,
        &engine,
        world.gpu_diagnostics().unwrap().completed_frames,
        0,
    );
    assert_ne!(
        world.gpu_body_render_view().unwrap().buffers(),
        [&before_rebuild[0], &before_rebuild[1]]
    );
    assert!(world.gpu_diagnostics().unwrap().model_compilations > compilations);
    assert_eq!(world.solver.gpu_execution(), Some(execution));
    println!(
        "projected runtime: three-substep ping-pong, impulse/momentum/energy, rollback/resume, budget rejection, GPU overflow rollback and empty soil pass"
    );
}
