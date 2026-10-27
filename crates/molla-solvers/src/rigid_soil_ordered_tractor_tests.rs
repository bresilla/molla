use super::*;
use std::io::Write;

fn machine_values(scene: &RigidScene) -> Vec<f64> {
    let mut values = Vec::new();
    for (body, _) in scene.bodies() {
        let pose = scene.body_pose(body).unwrap();
        let velocity = scene.body_velocity(body).unwrap();
        values.extend(pose.position.to_array());
        values.extend(pose.rotation.to_array());
        values.extend(velocity.linear.to_array());
        values.extend(velocity.angular.to_array());
    }
    values.extend_from_slice(scene.state().joint_q.host().unwrap());
    values.extend_from_slice(scene.state().joint_qd.host().unwrap());
    assert!(values.iter().all(|v| v.is_finite()));
    values
}

#[test]
#[ignore = "requires native/protected GPU and fresh SOIL_ORDERED_TRACTOR_DIR; actual 50k, 33 frames"]
fn ordered_tractor_full_trajectory_capture() {
    assert!(!cfg!(feature = "warm-newton"));
    assert!(std::env::vars().all(|(key, _)| !key.starts_with("MOLLA_NEWTON_")));
    assert!(std::env::var_os("MOLLA_SOIL_DIAGNOSTIC_GLOBAL_P2G").is_none());
    let root = std::path::PathBuf::from(std::env::var_os("SOIL_ORDERED_TRACTOR_DIR").unwrap());
    std::fs::create_dir(&root).unwrap();
    let started = std::time::Instant::now();
    let engine = Arc::new(
        Engine::new_gpu_with_features(
            wgpu::Features::SHADER_FLOAT32_ATOMIC | wgpu::Features::PASSTHROUGH_SHADERS,
        )
        .unwrap(),
    );
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut tractor =
        crate::soil_tractor::SoilTractor::new_ordered_newton_test(engine.clone()).unwrap();
    let construction_ms = started.elapsed().as_secs_f64() * 1000.;
    let soil = &tractor
        .world_mut_for_test()
        .gpu
        .as_ref()
        .unwrap()
        .soil
        .as_ref()
        .unwrap()
        .soil;
    assert!(soil.uses_native_float_atomics());
    assert!(soil.particle_tile_ordering_enabled());
    assert!(soil.particle_tile_ordering_uses_parallel_sort());
    assert!(soil.deterministic_p2g_storage_bytes() > 0);
    assert_eq!(soil.particle_count(), 50_000);
    let ordering_bytes = soil.particle_tile_ordering_storage_bytes();
    let gather_bytes = soil.deterministic_p2g_storage_bytes();
    let initial_machine = machine_values(&tractor.world().scene);
    std::fs::write(
        root.join("initial-soil.bin"),
        checkpoint(tractor.world_mut_for_test(), &engine),
    )
    .unwrap();
    std::fs::write(root.join("manifest.json"), serde_json::to_vec_pretty(&serde_json::json!({
        "schema":1,"acceptance":false,"physical_calibration":"uncalibrated",
        "scope":"independent live fixed-order trajectory; diagnostic capture, not performance acceptance",
        "method":if cfg!(feature="retained-local-newton") {23} else {22},"particles":50000,"frames":33,"substeps_per_frame":34,
        "dt":1.0/60.0,"torques":[300,300,300,300],"grid_size":64,"cell_size":0.1,
        "youngs_pa":3000,"density_kg_m3":1600,"machine":"generic-paired-storage",
        "nodes":4096,"iterations":64,"search_distance":0.05,"friction":0.3,
        "transfer":"native-dense-fixed-order-gather","ordering":"parallel-stable-tile",
        "ordering_bytes":ordering_bytes,"gather_bytes":gather_bytes,
        "machine_values":initial_machine.len(),"initial_machine":initial_machine,
        "initial_machine_bits":initial_machine.iter().map(|v| format!("{:016x}",v.to_bits())).collect::<Vec<_>>(),
        "renderer":false,"construction_ms":construction_ms
    })).unwrap()).unwrap();
    let mut log = std::fs::File::create_new(root.join("frames.jsonl")).unwrap();
    for frame in 0..33u64 {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
        let begin = std::time::Instant::now();
        let result = tractor.step(1. / 60., [300.; 4]);
        let wall_ms = begin.elapsed().as_secs_f64() * 1000.;
        let allocations = allocations.finish();
        let transfers = transfers.finish();
        let dispatches = dispatches.finish();
        let stats = tractor.world().mpm_soil_stats().unwrap();
        let values = machine_values(&tractor.world().scene);
        let record = serde_json::json!({"frame":frame,"step_error":result.as_ref().err().map(ToString::to_string),
            "committed_frames":stats.committed_frames,"rolled_back_frames":stats.rolled_back_frames,
            "live_bindings":stats.live_bindings,"faulted":stats.faulted,"values":values,
            "bits":values.iter().map(|v| format!("{:016x}",v.to_bits())).collect::<Vec<_>>(),
            "simulated_seconds":tractor.telemetry().unwrap().simulated_seconds,
            "wall_ms":wall_ms,"allocations":allocations.observations,"upload_bytes":transfers.upload_bytes,
            "readback_bytes":transfers.readback_bytes,"dispatches":dispatches.encoded_dispatches});
        writeln!(log, "{record}").unwrap();
        log.flush().unwrap();
        result.unwrap();
        assert!(!allocations.invalid && !transfers.overflowed && !dispatches.overflowed);
        assert!(!stats.faulted);
        assert_eq!(stats.committed_frames, frame + 1);
        assert_eq!(stats.rolled_back_frames, 0);
        assert_eq!(stats.live_bindings, 5);
        assert_eq!(values.len(), initial_machine.len());
        if frame >= 3 {
            assert_eq!(allocations.observations, 0);
        }
        eprintln!(
            "ordered frame={frame} committed={} wall_ms={wall_ms:.3}",
            stats.committed_frames
        );
    }
    let final_soil = checkpoint(tractor.world_mut_for_test(), &engine);
    std::fs::write(root.join("final-soil.bin"), &final_soil).unwrap();
    let positions = tractor.read_soil_positions().unwrap();
    assert_eq!(positions.len(), 50_000);
    assert!(positions.iter().flatten().all(|v| v.is_finite()));
    let position_bytes: Vec<_> = positions
        .iter()
        .flatten()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    std::fs::write(root.join("final-positions.bin"), position_bytes).unwrap();
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "schema":1,"acceptance":false,"physical_calibration":"uncalibrated","frames":33,
            "particles":positions.len(),"checkpoint_bytes":final_soil.len(),
            "simulated_seconds":tractor.telemetry().unwrap().simulated_seconds,
            "elapsed_ms":started.elapsed().as_secs_f64()*1000.,"construction_ms":construction_ms,
            "status":"diagnostic_complete"
        }))
        .unwrap(),
    )
    .unwrap();
}
