use super::retry_machine_words;
use crate::mpm::{MpmGridStorage, MpmUnilateralConfig};
use crate::soil_tractor::{SoilTractor, SoilTractorConfig, SoilTractorTransfer};
use molla_compute::Engine;
use std::{io::Write, path::Path, sync::Arc};

const DT: f64 = 1.0 / 60.0;
const BUDGET: u64 = 16 + 128 * 50_000;

#[derive(PartialEq)]
struct Snapshot {
    soil: Vec<u8>,
    machine: Vec<Vec<u8>>,
    cpu: String,
}

impl Snapshot {
    fn read(tractor: &mut SoilTractor, frame: u64) -> Self {
        let handles = tractor.handles();
        let cpu = format!(
            "{:?}\n{:?}\n",
            tractor.telemetry().unwrap(),
            std::iter::once(handles.chassis)
                .chain(handles.wheels)
                .chain(std::iter::once(handles.tine))
                .map(|body| (
                    tractor.world().scene.body_pose(body).unwrap(),
                    tractor.world().scene.body_velocity(body).unwrap(),
                ))
                .collect::<Vec<_>>()
        );
        let (ordinal, checkpoint) = tractor.read_soil_checkpoint(50_000).unwrap();
        assert_eq!(ordinal, frame);
        Self {
            soil: checkpoint.to_bytes(),
            machine: retry_machine_words(tractor.world()),
            cpu,
        }
    }

    fn save(&self, root: &Path, name: &str) {
        std::fs::write(root.join(format!("{name}-soil.bin")), &self.soil).unwrap();
        std::fs::write(root.join(format!("{name}-cpu.txt")), &self.cpu).unwrap();
        for (field, bytes) in self.machine.iter().enumerate() {
            std::fs::write(root.join(format!("{name}-machine-{field}.bin")), bytes).unwrap();
        }
    }

    fn record(&self, history: &mut std::fs::File, arm: &str, frame: u64) {
        let digest = |bytes: &[u8]| blake3::hash(bytes).to_hex().to_string();
        writeln!(
            history,
            "{}",
            serde_json::json!({"arm":arm,"frame":frame,"soil":digest(&self.soil),
                "machine":self.machine.iter().map(|b|digest(b)).collect::<Vec<_>>(),
                "cpu":digest(self.cpu.as_bytes())})
        )
        .unwrap();
    }
}

#[test]
#[ignore = "requires RTX GPU; actual 50k native Newton tractor, 33 frames per arm, motion-fault retry"]
fn tractor_50k_motion_fault_exact_retry() {
    actual_tractor_motion_retry(false, false);
}

#[cfg(feature = "soa-scratch-newton")]
#[test]
#[ignore = "requires RTX GPU; actual 50k SoA Newton tractor with fused proposals and motion-fault retry"]
fn tractor_50k_fused_motion_fault_exact_retry() {
    actual_tractor_motion_retry(true, false);
}

#[cfg(feature = "soa-scratch-newton")]
#[test]
#[ignore = "requires RTX GPU; actual 50k tractor fault after 17 committed internal substeps"]
fn tractor_50k_fused_later_motion_fault_exact_retry() {
    actual_tractor_motion_retry(true, true);
}

fn actual_tractor_motion_retry(fused_retry: bool, delayed_fault: bool) {
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_MOTION_RETRY_CAPTURE").expect("explicit retry capture"),
    );
    std::fs::create_dir(&root).unwrap();
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC
                | wgpu::Features::SHADER_F64
                | wgpu::Features::PASSTHROUGH_SHADERS
                | molla_kernels::FeatherstoneKernel::paired_drift_required_features(),
        )
        .unwrap(),
    );
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software);
    let cfg = SoilTractorConfig::default();
    assert_eq!(cfg.particle_count().unwrap(), 50_000);
    assert_eq!((DT / cfg.max_substep_s).ceil() as usize, 34);
    std::fs::write(
        root.join("manifest.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "adapter":adapter.name,"driver":adapter.driver_info,
            "particles":50000,"bodies":6,"bindings":5,"frames_per_arm":33,
            "substeps_per_frame":34,"frame_seconds":DT,"drive_torque_nm":([300.;4]),
            "transfer":"shared-chunk","transfer_bytes":3200000,"compact_tiles":1024,
            "node_capacity":4096,"outer_iterations":64,"friction":0.3,
            "motion_guard_bytes":BUDGET,"fault_before_frame":4,
            "fault_after_internal_substeps":if delayed_fault {17} else {0},
            "fault_injection":if delayed_fault {"synthetic-fused-motion-bit"} else {"zero-displacement-bound"},
            "requested_fusion":([false, false, fused_retry]),
            "soa_scratch_newton":cfg!(feature = "soa-scratch-newton"),
            "physical_calibration":"uncalibrated","performance_acceptance":false
        }))
        .unwrap(),
    )
    .unwrap();
    let build = |fusion| {
        let mut tractor = SoilTractor::new_native_prediction_with_newton_transfer(
            engine.clone(),
            cfg,
            MpmUnilateralConfig {
                node_capacity: 4096,
                search_distance: 0.05,
                max_iterations: 64,
            },
            0.3,
            MpmGridStorage::Compact {
                tile_capacity: 1024,
            },
            DT,
            SoilTractorTransfer::SharedChunk {
                max_gather_bytes: 3_200_000,
            },
        )
        .unwrap();
        tractor
            .set_soil_motion_bound(Some([1.; 3]), BUDGET)
            .unwrap();
        tractor.set_soil_motion_projection_fused(fusion).unwrap();
        assert_eq!(
            tractor.world().mpm_soil_motion_projection_fused(),
            Some(fusion)
        );
        tractor.prepare_soil_coupling().unwrap();
        tractor
    };
    let mut tractors = [build(false), build(false), build(fused_retry)];
    let names = ["control", "repeat", "retry"];
    let mut history = std::fs::File::create(root.join("history.jsonl")).unwrap();
    for frame in 1..=33 {
        if frame == 4 {
            let retry = &mut tractors[2];
            let prefix = Snapshot::read(retry, 3);
            let delayed = if delayed_fault {
                let soil = &mut retry
                    .world_mut_for_test()
                    .gpu
                    .as_mut()
                    .unwrap()
                    .soil
                    .as_mut()
                    .unwrap()
                    .soil;
                let start = soil
                    .read_active_grid_status_with(engine.device().unwrap(), engine.queue().unwrap())
                    .unwrap()
                    .completed_substeps;
                assert_eq!(start, 3 * 34);
                let target = start.wrapping_add(17);
                std::fs::write(root.join("delayed-fault.json"), serde_json::to_vec_pretty(&serde_json::json!({
                    "start_counter":start,"trigger_previous_counter":target,"expected_fault_counter":target.wrapping_add(1),
                    "successful_internal_substeps":17,"failed_internal_substep":18,"synthetic_motion_bit":true
                })).unwrap()).unwrap();
                Some((
                    soil.inject_fused_motion_fault_at(target),
                    target.wrapping_add(1),
                ))
            } else {
                retry.set_soil_motion_bound(Some([0.; 3]), BUDGET).unwrap();
                None
            };
            retry.prepare_soil_coupling().unwrap();
            let error = retry.step(DT, [300.; 4]).unwrap_err().to_string();
            assert_eq!(
                retry.world().mpm_soil_motion_projection_fused(),
                Some(fused_retry)
            );
            std::fs::write(root.join("fault.txt"), &error).unwrap();
            assert!(error.contains("motion-bound"), "{error}");
            if let Some((_, counter)) = &delayed {
                assert!(error.contains(&format!("g2p-counter={counter}")), "{error}");
            }
            let after = Snapshot::read(retry, 3);
            after.save(&root, "after-fault");
            assert!(after.soil == prefix.soil, "complete soil rollback");
            assert!(
                after.machine[5..] == prefix.machine[5..],
                "retained machine precision rollback"
            );
            assert!(
                after.cpu == prefix.cpu,
                "six-body CPU state and committed time rollback"
            );
            assert_eq!(retry.world().completed_steps(), 3);
            assert!(retry.world().gpu_body_render_view().is_err());
            let stats = retry.world().mpm_soil_stats().unwrap();
            assert_eq!(
                (
                    stats.committed_frames,
                    stats.rolled_back_frames,
                    stats.faulted
                ),
                (3, 1, false)
            );
            if let Some((pipeline, _)) = delayed {
                retry
                    .world_mut_for_test()
                    .gpu
                    .as_mut()
                    .unwrap()
                    .soil
                    .as_mut()
                    .unwrap()
                    .soil
                    .restore_fused_motion_pipeline(pipeline);
            } else {
                retry.set_soil_motion_bound(Some([1.; 3]), BUDGET).unwrap();
            }
            retry.prepare_soil_coupling().unwrap();
            println!("actual tractor motion fault: {error}; rollback exact");
        }
        let mut states = Vec::new();
        for (arm, (name, tractor)) in names.iter().zip(&mut tractors).enumerate() {
            tractor.step(DT, [300.; 4]).unwrap();
            assert_eq!(
                tractor.world().mpm_soil_motion_projection_fused(),
                Some(arm == 2 && fused_retry)
            );
            assert_eq!(tractor.world().solver.substeps, 34);
            let state = Snapshot::read(tractor, frame);
            state.record(&mut history, name, frame);
            if frame == 3 || frame == 33 {
                state.save(&root, &format!("{name}-{frame}"));
            }
            states.push(state);
        }
        for arm in 1..3 {
            if states[arm] != states[0] {
                states[0].save(&root, &format!("mismatch-control-{frame}"));
                states[arm].save(&root, &format!("mismatch-{}-{frame}", names[arm]));
                panic!("{} differs from exact control at frame {frame}", names[arm]);
            }
        }
        println!("frame {frame}/33: soil, nine machine streams and six-body CPU state exact");
    }
    for (arm, tractor) in tractors.iter().enumerate() {
        let stats = tractor.world().mpm_soil_stats().unwrap();
        assert_eq!(
            (
                stats.committed_frames,
                stats.rolled_back_frames,
                stats.faulted
            ),
            (33, u64::from(arm == 2), false)
        );
        assert_eq!(tractor.world().completed_steps(), 33);
        assert!((tractor.telemetry().unwrap().simulated_seconds - 0.55).abs() < 1e-14);
    }
    std::fs::write(
        root.join("passed.json"),
        "{\"frames_per_arm\":33,\"exact_frame_comparisons\":66,\"faults\":1,\"passed\":true}\n",
    )
    .unwrap();
}
