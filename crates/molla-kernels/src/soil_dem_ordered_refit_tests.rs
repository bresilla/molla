use super::*;

fn source() -> String {
    super::super::ordered::source()
}
#[test]
fn dem_ordered_refit_source_contract() {
    let source = source();
    let module = naga::front::wgsl::parse_str(&source).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(source.matches("after = hits[used - 1u]").count(), 1);
    assert_eq!(source.matches("if used < 32u { break; }").count(), 1);
}

#[test]
#[ignore = "requires GPU; 50k ordered-refit trajectory plus overfull neighbor batches"]
fn dem_ordered_refit_exact_trajectory_and_overflow() {
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_DEM_ORDERED_OUTPUT").expect("fresh output"),
    );
    std::fs::create_dir(&root).unwrap();
    let engine = molla_compute::Engine::new_gpu().unwrap();
    let adapter = engine.adapter_info().unwrap();
    assert!(!adapter.is_software);
    std::fs::write(root.join("adapter.txt"), format!("{adapter:?}\n")).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireDemGpuKernel::new(device);
    for n in [50_000, 65] {
        let positions = (0..n)
            .map(|i| {
                if n == 50_000 {
                    [
                        (i % 40) as f32 * 0.045 - 0.9,
                        (i / 1600) as f32 * 0.045 + 0.026,
                        (i / 40 % 40) as f32 * 0.045 - 0.9,
                        0.0,
                    ]
                } else {
                    [
                        (i % 5) as f32 * 0.001,
                        0.1 + (i / 25) as f32 * 0.001,
                        (i / 5 % 5) as f32 * 0.001,
                        0.0,
                    ]
                }
            })
            .collect::<Vec<_>>();
        let velocities = vec![[0.0; 4]; n];
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
            dt: if n == 50_000 { 0.0005 } else { 0.000001 },
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
        owners[1].set_lbvh_rebuild_interval(32).unwrap();
        let mut history = std::fs::File::create(root.join(format!("history-{n}.csv"))).unwrap();
        writeln!(history, "step,position_max,velocity_max,bit_exact").unwrap();
        for step in 1..=52 {
            for owner in &mut owners {
                owner
                    .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, 8)
                    .unwrap();
            }
            let a = kernel
                .readback_resident_state(device, queue, owners[0].resident_state())
                .unwrap();
            let b = kernel
                .readback_resident_state(device, queue, owners[1].resident_state())
                .unwrap();
            let delta = difference(&a, &b);
            writeln!(
                history,
                "{step},{},{},{}",
                delta.0,
                delta.1,
                delta.2.is_none()
            )
            .unwrap();
            if delta.2.is_some() {
                save(&root, &format!("mismatch-{n}-{step}-reference"), &a);
                save(&root, &format!("mismatch-{n}-{step}-candidate"), &b);
            }
            assert!(
                delta.2.is_none(),
                "n={n} step={step} candidate changed original rebuild state: {delta:?}"
            );
            if step == 52 {
                save(&root, &format!("final-{n}"), &a);
            }
        }
        assert_eq!(owners[0].topology_stats().rebuilds, 52);
        assert_eq!(owners[1].topology_stats().rebuilds, 2);
        assert_eq!(owners[1].topology_stats().refits, 50);
        println!(
            "{n} particles: 52 rebuilt/ordered-refit states bit-exact; candidate 2 rebuilds + 50 refits"
        );
    }
}

#[test]
#[ignore = "requires GPU; paired advancing 50k solver and export timings"]
fn dem_ordered_refit_paired_timing() {
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_DEM_ORDERED_OUTPUT").expect("fresh output"),
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
    owners[1].set_lbvh_rebuild_interval(32).unwrap();
    for owner in &mut owners {
        owner
            .step_sparse_lbvh_gpu_resident(device, queue, &kernel, &contacts, params, 8)
            .unwrap();
    }
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
