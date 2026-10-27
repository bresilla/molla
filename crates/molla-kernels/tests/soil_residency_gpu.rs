#![cfg(feature = "gpu")]

use molla_compute::Engine;
use molla_kernels::{
    TireDemContactCell, TireDemGpuKernel, TireDemResidentFemTerrainOwner,
    TireDemResidentTerrainParams, TireDemStepParams,
};

fn grains(n: usize) -> (Vec<[f32; 4]>, Vec<[f32; 4]>) {
    let positions = (0..n)
        .map(|i| {
            [
                (i % 7) as f32 * 0.08 - 0.2,
                0.08 + (i / 21) as f32 * 0.07,
                (i / 7 % 3) as f32 * 0.08 - 0.1,
                0.0,
            ]
        })
        .collect();
    let velocities = (0..n)
        .map(|i| [(i % 3) as f32 * 0.01, -0.02, (i % 5) as f32 * -0.01, 0.0])
        .collect();
    (positions, velocities)
}

fn terrain() -> TireDemResidentTerrainParams {
    TireDemResidentTerrainParams {
        origin: [-1.0, -1.0],
        spacing: 0.125,
        nx: 16,
        nz: 16,
        floor_y: 0.0,
        grain_radius: 0.05,
    }
}

fn inputs(step: usize) -> (Vec<TireDemContactCell>, TireDemStepParams) {
    let contacts = (0..step % 4)
        .map(|i| TireDemContactCell {
            center_radius: [step as f32 * 0.01, 0.1, i as f32 * 0.1, 0.3],
            force: [step as f32 - 2.0, -15.0, i as f32, 0.0],
        })
        .collect();
    (
        contacts,
        TireDemStepParams {
            dt: 0.0001 + (step % 3) as f32 * 0.00002,
            gravity: [step as f32 * 0.03, -9.81, 0.0],
            ..Default::default()
        },
    )
}

fn assert_rows(a: &[[f32; 4]], b: &[[f32; 4]], tolerance: f32) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        for k in 0..4 {
            assert!(
                (a[k] - b[k]).abs() <= tolerance,
                "row {i} axis {k}: {} != {}",
                a[k],
                b[k]
            );
        }
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_dem_retained_pipeline_matches_legacy_with_changing_parameters() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    let (positions, velocities) = grains(65);
    let mut owner = TireDemResidentFemTerrainOwner::from_grain_state(
        device,
        queue,
        &kernel,
        terrain(),
        &positions,
        &velocities,
    )
    .unwrap();
    let (min, extent) = owner.lbvh_scene_bounds();
    let mut baseline = kernel
        .create_resident_state(device, queue, &positions, &velocities)
        .unwrap();
    for step in 0..12 {
        let (contacts, params) = inputs(step);
        let tree = kernel
            .build_resident_sparse_lbvh_tree_rows_gpu_radix(
                device, queue, &baseline, min, extent, 8,
            )
            .unwrap();
        baseline = kernel
            .dispatch_lbvh_step_resident_state_with_resident_lbvh(
                device, queue, &baseline, &contacts, params, &tree,
            )
            .unwrap()
            .into_state();
    }
    for step in 0..12 {
        let (contacts, params) = inputs(step);
        owner
            .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, 8)
            .unwrap();
    }
    assert_eq!(owner.state_revision(), 12);
    assert_eq!(owner.mirror_revision(), 0);
    assert_eq!(owner.positions(), positions);
    let expected = kernel
        .readback_resident_state(device, queue, &baseline)
        .unwrap();
    owner.refresh_mirror(device, queue, &kernel).unwrap();
    assert_eq!(owner.mirror_revision(), owner.state_revision());
    assert_rows(owner.positions(), &expected.positions, 2e-5);
    assert_rows(owner.velocities(), &expected.velocities, 2e-5);
    let baseline_terrain = kernel
        .dispatch_resident_terrain_raster_reference(
            device,
            queue,
            &baseline,
            owner.reference_heights(),
            terrain(),
        )
        .unwrap();
    let baseline_raster = kernel
        .readback_resident_terrain(device, queue, &baseline_terrain)
        .unwrap();
    let raster = kernel
        .readback_resident_terrain(device, queue, owner.resident_terrain())
        .unwrap();
    assert_eq!(raster.heights, baseline_raster.heights);
    assert_eq!(raster.sinkage, baseline_raster.sinkage);
    assert_rows(
        &raster.surface_velocity_map.rows,
        &baseline_raster.surface_velocity_map.rows,
        2e-5,
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_dem_retained_owner_reseed_growth_and_device_validation() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    let (positions, velocities) = grains(31);
    let mut owner = TireDemResidentFemTerrainOwner::from_grain_state(
        device,
        queue,
        &kernel,
        terrain(),
        &positions,
        &velocities,
    )
    .unwrap();
    let (contacts, params) = inputs(3);
    owner
        .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, 8)
        .unwrap();
    let old_revision = owner.state_revision();
    let invalid = TireDemStepParams {
        dt: f32::NAN,
        ..params
    };
    assert!(
        owner
            .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, invalid, 8)
            .is_err()
    );
    assert_eq!(owner.state_revision(), old_revision);
    let second = Engine::new_gpu().expect("second GPU device required");
    assert!(
        owner
            .refresh_mirror(device, second.queue().unwrap(), &kernel)
            .is_err()
    );
    assert_eq!(owner.state_revision(), old_revision);
    assert!(
        owner
            .step_sparse_lbvh_gpu_resident(
                second.device().unwrap(),
                second.queue().unwrap(),
                &kernel,
                &contacts,
                params,
                8
            )
            .is_err()
    );
    let wrong_kernel = TireDemGpuKernel::new(second.device().unwrap());
    assert!(
        owner
            .step_sparse_lbvh_gpu_resident(device, queue, &wrong_kernel, &contacts, params, 8)
            .is_err()
    );
    assert_eq!(owner.state_revision(), old_revision);
    for n in [31, 65, 31] {
        let (positions, velocities) = grains(n);
        owner
            .reseed_grain_state(device, queue, &kernel, &positions, &velocities)
            .unwrap();
        assert_eq!(owner.mirror_revision(), owner.state_revision());
        assert!(owner.resident_lbvh().is_none());
        let mut cold = TireDemResidentFemTerrainOwner::from_grain_state(
            device,
            queue,
            &kernel,
            terrain(),
            &positions,
            &velocities,
        )
        .unwrap();
        for (leaf_size, contact_count) in [(8, 3), (2, 17), (2, 0)] {
            let mut contacts = contacts.clone();
            contacts.resize(contact_count, contacts[0]);
            owner
                .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, leaf_size)
                .unwrap();
            cold.step_sparse_lbvh_gpu_resident(
                device, queue, &kernel, &contacts, params, leaf_size,
            )
            .unwrap();
        }
        owner.refresh_mirror(device, queue, &kernel).unwrap();
        cold.refresh_mirror(device, queue, &kernel).unwrap();
        assert_rows(owner.positions(), cold.positions(), 2e-5);
        assert_rows(owner.velocities(), cold.velocities(), 2e-5);
    }
    owner
        .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, 8)
        .unwrap();
    drop(owner);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "needs a real GPU adapter and resource counters"]
fn soil_dem_frame_mirror_preserves_substeps_with_one_snapshot() {
    use molla_core::gpu_metrics::TransferScope;
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    let (positions, velocities) = grains(65);
    let mut per_step = TireDemResidentFemTerrainOwner::from_grain_state(
        device,
        queue,
        &kernel,
        terrain(),
        &positions,
        &velocities,
    )
    .unwrap();
    let mut per_frame = TireDemResidentFemTerrainOwner::from_grain_state(
        device,
        queue,
        &kernel,
        terrain(),
        &positions,
        &velocities,
    )
    .unwrap();
    let (contacts, params) = inputs(3);
    let scope = TransferScope::start().unwrap();
    for _ in 0..16 {
        per_step
            .step_sparse_lbvh_gpu_mirrored(device, queue, &kernel, &contacts, params, 8)
            .unwrap();
    }
    let before = scope.finish();
    let scope = TransferScope::start().unwrap();
    per_frame
        .step_sparse_lbvh_gpu_mirrored_n(device, queue, &kernel, &contacts, params, 8, 16)
        .unwrap();
    let after = scope.finish();
    assert_eq!(per_step.state_revision(), 16);
    assert_eq!(per_frame.state_revision(), 16);
    assert_eq!(per_frame.mirror_revision(), 16);
    assert_rows(per_step.positions(), per_frame.positions(), 2e-5);
    assert_rows(per_step.velocities(), per_frame.velocities(), 2e-5);
    assert!(after.readback_bytes > 0);
    assert_eq!(before.readback_bytes, after.readback_bytes * 16);
    assert!(!before.overflowed && !after.overflowed);
    assert!(
        per_frame
            .step_sparse_lbvh_gpu_mirrored_n(device, queue, &kernel, &contacts, params, 8, 0)
            .is_err()
    );
    assert_eq!(per_frame.state_revision(), 16);
    eprintln!(
        "frame bridge logical readback bytes: per-step={} per-frame={}",
        before.readback_bytes, after.readback_bytes
    );
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "needs a real GPU adapter and resource counters"]
fn soil_dem_steady_steps_reuse_resources_without_logical_readbacks() {
    dem_steady_resources(1);
}

#[cfg(feature = "resource-counters")]
#[test]
#[ignore = "needs a real GPU adapter and resource counters"]
fn soil_dem_ordered_refit_reuses_resources_without_logical_readbacks() {
    dem_steady_resources(32);
}

#[cfg(feature = "resource-counters")]
fn dem_steady_resources(interval: u32) {
    use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    let (positions, velocities) = grains(129);
    let mut owner = TireDemResidentFemTerrainOwner::from_grain_state(
        device,
        queue,
        &kernel,
        terrain(),
        &positions,
        &velocities,
    )
    .unwrap();
    owner.set_lbvh_rebuild_interval(interval).unwrap();
    let (contacts, params) = inputs(3);
    let initialization = DeviceScope::start().unwrap();
    owner
        .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, 8)
        .unwrap();
    assert!(initialization.finish().observations > 0);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let before = device.get_internal_counters().hal;
    assert!(before.bind_groups.read() > 0 && before.compute_pipelines.read() > 0);
    let heights = owner.resident_terrain().heights.clone();
    let tree = owner.resident_lbvh().unwrap().node_rows.clone();
    let positions_a = owner.resident_state().positions.clone();
    let resources = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let dispatches = DispatchScope::start().unwrap();
    for step in 0..8 {
        let (contacts, params) = inputs(step);
        owner
            .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, 8)
            .unwrap();
        assert_eq!(owner.resident_terrain().heights, heights);
        assert_eq!(owner.resident_lbvh().unwrap().node_rows, tree);
    }
    let resources = resources.finish();
    let transfers = transfers.finish();
    let dispatches = dispatches.finish();
    assert_eq!(
        resources.observations, 0,
        "no explicit GPU buffer allocations after warmup"
    );
    assert!(!resources.invalid);
    assert_eq!(transfers.readback_bytes, 0);
    assert!(transfers.upload_bytes > 0);
    assert!(!transfers.overflowed);
    if interval == 1 {
        assert!(dispatches.encoded_dispatches >= 8 * 15);
        assert_eq!(owner.topology_stats().rebuilds, 9);
    } else {
        assert_eq!(owner.topology_stats().rebuilds, 1);
        assert_eq!(owner.topology_stats().refits, 8);
        assert!(dispatches.encoded_dispatches >= 8 * 7);
    }
    assert_eq!(owner.resident_state().positions, positions_a);
    let after = device.get_internal_counters().hal;
    assert_eq!(before.bind_groups.read(), after.bind_groups.read());
    assert_eq!(
        before.compute_pipelines.read(),
        after.compute_pipelines.read()
    );
    assert_eq!(owner.mirror_revision(), 0);
    eprintln!(
        "resident DEM: resources={resources:?}, transfers={transfers:?}, dispatches={dispatches:?}"
    );
    owner.refresh_mirror(device, queue, &kernel).unwrap();
    assert_eq!(owner.mirror_revision(), 9);
}

#[test]
#[ignore = "bounded GPU solver-plus-export performance measurement"]
fn soil_dem_resident_benchmark() {
    let refit_interval: u32 = std::env::var("SOIL_REFIT_INTERVAL")
        .unwrap_or_else(|_| "32".into())
        .parse()
        .expect("positive refit interval");
    assert!((1..=128).contains(&refit_interval));
    let engine = Engine::new_gpu().expect("GPU required");
    let adapter = engine.adapter_info().expect("adapter identity");
    assert!(!adapter.is_software);
    eprintln!("adapter={adapter:?}");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    let positions: Vec<_> = (0..50_000)
        .map(|i| {
            [
                (i % 40) as f32 * 0.045 - 0.9,
                (i / 1600) as f32 * 0.045 + 0.026,
                (i / 40 % 40) as f32 * 0.045 - 0.9,
                0.0,
            ]
        })
        .collect();
    let velocities = vec![[0.0; 4]; positions.len()];
    let terrain = TireDemResidentTerrainParams {
        origin: [-1.0, -1.0],
        spacing: 0.0625,
        nx: 32,
        nz: 32,
        floor_y: 0.0,
        grain_radius: 0.025,
    };
    let params = TireDemStepParams {
        grain_mass: 0.18,
        grain_radius: 0.025,
        normal_stiffness: 8000.0,
        damping: 40.0,
        friction: 0.6,
        floor_y: 0.0,
        gravity: [0.0, -9.81, 0.0],
        dt: 0.0005,
        tire_position_push_scale: 0.0,
    };
    let contacts = [TireDemContactCell {
        center_radius: [0.0, 1.4, 0.0, 0.3],
        force: [10.0, -80.0, 0.0, 0.0],
    }];
    let mut owner = TireDemResidentFemTerrainOwner::from_grain_state(
        device,
        queue,
        &kernel,
        terrain,
        &positions,
        &velocities,
    )
    .unwrap();
    let mut refitted = TireDemResidentFemTerrainOwner::from_grain_state(
        device,
        queue,
        &kernel,
        terrain,
        &positions,
        &velocities,
    )
    .unwrap();
    refitted.set_lbvh_rebuild_interval(refit_interval).unwrap();
    let mut baseline = kernel
        .create_resident_state(device, queue, &positions, &velocities)
        .unwrap();
    let mut baseline_terrain = None;
    let (min, extent) = owner.lbvh_scene_bounds();
    eprintln!(
        "{{\"schema\":1,\"case\":\"dem-resident\",\"scope\":\"solver+export\",\"particles\":50000,\"seed\":0,\"distribution\":\"40x40-layered-lattice\",\"particle_spacing_m\":0.045,\"grain_mass_kg\":0.18,\"grain_radius_m\":0.025,\"normal_stiffness\":8000,\"damping\":40,\"friction\":0.6,\"floor_y_m\":0,\"gravity_m_s2\":[0,-9.81,0],\"dt_s\":0.0005,\"substeps_per_sample\":4,\"simulated_seconds_per_sample\":0.002,\"raster\":[32,32],\"raster_spacing_m\":0.0625,\"contact_center_radius\":[0,1.4,0,0.3],\"contact_force_n\":[10,-80,0],\"push_scale\":0,\"leaf_size\":8,\"snapshot_mode\":\"final-parity-only\",\"paired_order\":\"rotating-three-way\",\"warmup_windows_per_mode\":1,\"samples_per_mode\":12}}"
    );
    let mut samples = [Vec::new(), Vec::new(), Vec::new()];
    eprintln!(
        "{{\"schema\":1,\"case\":\"dem-resident\",\"refit_interval_steps\":{refit_interval}}}"
    );
    for window in 0..13 {
        for turn in 0..3 {
            let mode = (window + turn) % 3;
            let start = std::time::Instant::now();
            for _ in 0..4 {
                if mode == 0 {
                    let tree = kernel
                        .build_resident_sparse_lbvh_tree_rows_gpu_radix(
                            device, queue, &baseline, min, extent, 8,
                        )
                        .unwrap();
                    baseline = kernel
                        .dispatch_lbvh_step_resident_state_with_resident_lbvh(
                            device, queue, &baseline, &contacts, params, &tree,
                        )
                        .unwrap()
                        .into_state();
                    baseline_terrain = Some(
                        kernel
                            .dispatch_resident_terrain_raster(
                                device,
                                queue,
                                &baseline,
                                owner.reference_heights(),
                                terrain,
                            )
                            .unwrap(),
                    );
                } else if mode == 1 {
                    owner
                        .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, 8)
                        .unwrap();
                } else {
                    refitted
                        .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, 8)
                        .unwrap();
                }
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            if window != 0 {
                samples[mode].push(start.elapsed().as_secs_f64() * 1000.0);
            }
        }
    }
    let refit_mode = format!("retained-refit{refit_interval}");
    for (mode, times) in ["reallocated", "retained-rebuild", refit_mode.as_str()]
        .into_iter()
        .zip(&mut samples)
    {
        assert!(times.iter().all(|t| t.is_finite() && *t > 0.0));
        eprintln!(
            "{{\"schema\":1,\"case\":\"dem-resident\",\"mode\":\"{mode}\",\"raw_ms\":{times:?}}}"
        );
        times.sort_by(f64::total_cmp);
        eprintln!(
            "{{\"schema\":1,\"case\":\"dem-resident\",\"mode\":\"{mode}\",\"samples\":12,\"p50_ms\":{},\"p95_ms\":{},\"simulated_seconds_per_sample\":0.002,\"rtf_wall_over_sim_at_p50\":{},\"sim_rate_at_p50\":{},\"scope\":\"solver+export\"}}",
            times[6],
            times[11],
            times[6] / 2.0,
            2.0 / times[6]
        );
    }
    let expected = kernel
        .readback_resident_state(device, queue, &baseline)
        .unwrap();
    owner.refresh_mirror(device, queue, &kernel).unwrap();
    assert_rows(owner.positions(), &expected.positions, 2e-5);
    assert_rows(owner.velocities(), &expected.velocities, 2e-5);
    let expected_terrain = kernel
        .readback_resident_terrain(device, queue, &baseline_terrain.unwrap())
        .unwrap();
    let got_terrain = kernel
        .readback_resident_terrain(device, queue, owner.resident_terrain())
        .unwrap();
    assert_rows(
        &got_terrain.surface_velocity_map.rows,
        &expected_terrain.surface_velocity_map.rows,
        2e-5,
    );
    assert_eq!(got_terrain.heights, expected_terrain.heights);
    assert_eq!(got_terrain.sinkage, expected_terrain.sinkage);
    refitted.refresh_mirror(device, queue, &kernel).unwrap();
    assert_rows(refitted.positions(), &expected.positions, 2e-5);
    assert_rows(refitted.velocities(), &expected.velocities, 2e-5);
    let refit_terrain = kernel
        .readback_resident_terrain(device, queue, refitted.resident_terrain())
        .unwrap();
    assert_rows(
        &refit_terrain.surface_velocity_map.rows,
        &expected_terrain.surface_velocity_map.rows,
        2e-5,
    );
    for (a, b) in refit_terrain.heights.iter().zip(&expected_terrain.heights) {
        assert!((a - b).abs() <= 1e-6);
    }
    for (a, b) in refit_terrain.sinkage.iter().zip(&expected_terrain.sinkage) {
        assert!((a - b).abs() <= 1e-6);
    }
    let rebuilds = 52u32.div_ceil(refit_interval);
    assert_eq!(refitted.topology_stats().rebuilds, u64::from(rebuilds));
    assert_eq!(refitted.topology_stats().refits, u64::from(52 - rebuilds));
    eprintln!("refit_topology_stats={:?}", refitted.topology_stats());
    eprintln!(
        "{{\"schema\":1,\"case\":\"dem-resident\",\"final_parity\":true,\"steps_per_mode_including_warmup\":52,\"simulated_seconds_per_mode_including_warmup\":0.026}}"
    );
}
