use super::refit_diagnostic::{difference, save};
use super::*;
use std::io::Write;

#[test]
fn dem_grouped_refit_source_and_tree_partition() {
    let module =
        naga::front::wgsl::parse_str(include_str!("../shaders/soil_dem_grouped_refit.wgsl"))
            .unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    for padded in [1usize, 2, 4, 8, 16, 32, 64, 128, 256, 8192] {
        let batch = padded.min(64);
        let mut owners = vec![0u32; 2 * padded - 1];
        for owner in &mut owners[..padded / batch - 1] {
            *owner += 1;
        }
        for group in 0..padded / batch {
            for lane in 0..batch {
                owners[padded - 1 + group * batch + lane] += 1;
            }
            let mut width = batch / 2;
            let mut base = padded / 2;
            while width > 0 {
                for lane in 0..width {
                    owners[base - 1 + group * width + lane] += 1;
                }
                width /= 2;
                base /= 2;
            }
        }
        assert!(
            owners.iter().all(|&n| n == 1),
            "each node has exactly one initializer, padded={padded}"
        );
    }
}

fn tree_rows(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    tree: &TireDemResidentLbvh,
) -> Vec<[f32; 4]> {
    let stage = stage_vec4(device, "grouped refit tree check", tree.node_row_count);
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(
        &tree.node_rows,
        0,
        &stage,
        0,
        (tree.node_row_count * 16) as u64,
    );
    queue.submit([encoder.finish()]);
    read_vec4(device, &stage, "grouped refit tree check").unwrap()
}

#[test]
#[ignore = "requires GPU; grouped refit tree and 50k trajectory equivalence"]
fn dem_grouped_refit_gpu_tree_and_trajectory() {
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_DEM_GROUPED_OUTPUT").expect("fresh output"),
    );
    std::fs::create_dir(&root).unwrap();
    let engine = molla_compute::Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
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
    let mut log = std::fs::File::create(root.join("cases.csv")).unwrap();
    writeln!(
        log,
        "particles,leaf_size,steps,tree_words_exact,state_words_exact"
    )
    .unwrap();
    for (n, leaf_size) in [
        (1, 8),
        (2, 1),
        (17, 1),
        (65, 1),
        (129, 1),
        (129, 8),
        (513, 8),
        (1031, 3),
        (50_000, 8),
    ] {
        let positions = (0..n)
            .map(|i| {
                [
                    (i % 40) as f32 * 0.045 - 0.9,
                    (i / 1600) as f32 * 0.045 + 0.026,
                    (i / 40 % 40) as f32 * 0.045 - 0.9,
                    0.0,
                ]
            })
            .collect::<Vec<_>>();
        let velocities = vec![[0.0; 4]; n];
        let mut owners: [_; 2] = std::array::from_fn(|_| {
            TireDemResidentFemTerrainOwner::from_grain_state(
                device,
                queue,
                &kernel,
                terrain,
                &positions,
                &velocities,
            )
            .unwrap()
        });
        for owner in &mut owners {
            owner.set_lbvh_rebuild_interval(32).unwrap();
        }
        let steps = if n == 50_000 { 52 } else { 5 };
        for step in 0..steps {
            for owner in &mut owners {
                owner
                    .step_sparse_lbvh_gpu_resident(
                        device, queue, &kernel, &contacts, params, leaf_size,
                    )
                    .unwrap();
            }
            let states = owners
                .iter()
                .map(|o| {
                    kernel
                        .readback_resident_state(device, queue, o.resident_state())
                        .unwrap()
                })
                .collect::<Vec<_>>();
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&states[0].positions),
                bytemuck::cast_slice::<_, u32>(&states[1].positions),
                "positions n={n} step={step}"
            );
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&states[0].velocities),
                bytemuck::cast_slice::<_, u32>(&states[1].velocities),
                "velocities n={n} step={step}"
            );
            let trees = owners
                .iter()
                .map(|o| tree_rows(device, queue, o.resident_lbvh().unwrap()))
                .collect::<Vec<_>>();
            assert_eq!(
                bytemuck::cast_slice::<_, u32>(&trees[0]),
                bytemuck::cast_slice::<_, u32>(&trees[1]),
                "tree n={n} step={step}"
            );
            if step == 0 {
                owners[0].stepper.as_mut().unwrap().legacy_tree_refit = true;
            }
        }
        writeln!(log, "{n},{leaf_size},{steps},true,true").unwrap();
        println!("grouped refit n={n} leaf={leaf_size}: {steps} tree/particle states byte-exact");
    }
}

#[test]
#[ignore = "requires GPU; paired advancing 50k solver and export timings"]
fn dem_grouped_refit_paired_timing() {
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_DEM_GROUPED_OUTPUT").expect("fresh output"),
    );
    std::fs::create_dir(&root).unwrap();
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software);
    std::fs::write(root.join("adapter.txt"), format!("{adapter:?}\n")).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    let positions = (0..50_000)
        .map(|i| {
            [
                (i % 40) as f32 * 0.045 - 0.9,
                (i / 1600) as f32 * 0.045 + 0.026,
                (i / 40 % 40) as f32 * 0.045 - 0.9,
                0.0,
            ]
        })
        .collect::<Vec<_>>();
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
    let mut owners: [_; 2] = std::array::from_fn(|_| {
        TireDemResidentFemTerrainOwner::from_grain_state(
            device,
            queue,
            &kernel,
            terrain,
            &positions,
            &velocities,
        )
        .unwrap()
    });
    for owner in &mut owners {
        owner.set_lbvh_rebuild_interval(32).unwrap();
    }
    for owner in &mut owners {
        owner
            .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, 8)
            .unwrap();
    }
    owners[0].stepper.as_mut().unwrap().legacy_tree_refit = true;
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let mut samples: [_; 2] = std::array::from_fn(|_| Vec::new());
    let mut output = std::fs::File::create(root.join("samples.csv")).unwrap();
    writeln!(output, "window,arm,wall_ms,simulated_seconds").unwrap();
    for window in 0..13 {
        for turn in 0..2 {
            let arm = (window + turn) % 2;
            let start = std::time::Instant::now();
            for _ in 0..4 {
                owners[arm]
                    .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, 8)
                    .unwrap();
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let milliseconds = start.elapsed().as_secs_f64() * 1000.0;
            assert!(milliseconds.is_finite() && milliseconds > 0.0);
            if window > 0 {
                samples[arm].push(milliseconds);
                writeln!(output, "{window},{arm},{milliseconds},0.002").unwrap();
            }
        }
    }
    let final_states = owners
        .iter()
        .map(|o| {
            kernel
                .readback_resident_state(device, queue, o.resident_state())
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert!(
        difference(&final_states[0], &final_states[1]).2.is_none(),
        "timed trajectories bit-exact"
    );
    save(&root, "final-reference", &final_states[0]);
    save(&root, "final-candidate", &final_states[1]);
    let surfaces = owners
        .iter()
        .map(|o| {
            kernel
                .readback_resident_terrain(device, queue, o.resident_terrain())
                .unwrap()
        })
        .collect::<Vec<_>>();
    assert_eq!(surfaces[0].heights, surfaces[1].heights);
    assert_eq!(surfaces[0].sinkage, surfaces[1].sinkage);
    let mut surface_error = 0.0f32;
    for (a, b) in surfaces[0]
        .surface_velocity_map
        .rows
        .iter()
        .zip(&surfaces[1].surface_velocity_map.rows)
    {
        for axis in 0..4 {
            assert!(a[axis].is_finite() && b[axis].is_finite());
            surface_error = surface_error.max((a[axis] - b[axis]).abs());
        }
    }
    assert!(
        surface_error <= 2e-5,
        "surface velocity error={surface_error}"
    );
    let mut ratios = samples[0]
        .iter()
        .zip(&samples[1])
        .map(|(a, b)| b / a)
        .collect::<Vec<_>>();
    ratios.sort_by(f64::total_cmp);
    for sample in &mut samples {
        sample.sort_by(f64::total_cmp);
    }
    let report = format!(
        "samples=12\nsubsteps_per_sample=4\nsubstep_seconds=0.0005\nsetup_steps=1\nwarmup_steps=4\nfinal_steps=53\nreference_p50_ms={}\ncandidate_p50_ms={}\nreference_p95_ms={}\ncandidate_p95_ms={}\npaired_ratio_upper_median={}\nreference_rebuilds={}\ncandidate_rebuilds={}\ncandidate_refits={}\nfinal_state_heights_sinkage_exact=true\nsurface_velocity_max_error={surface_error}\nsurface_velocity_tolerance=0.00002\nfull_acceptance=false\n",
        samples[0][6],
        samples[1][6],
        samples[0][11],
        samples[1][11],
        ratios[6],
        owners[0].topology_stats().rebuilds,
        owners[1].topology_stats().rebuilds,
        owners[1].topology_stats().refits
    );
    std::fs::write(root.join("timing.txt"), &report).unwrap();
    println!("{report}");
}
