use super::*;

fn output_words(engine: &Engine, projection: &SoilContactProjectionGpu) -> Vec<u32> {
    [
        projection.rates_buffer(),
        projection.motion_buffer(),
        projection.impulses_buffer(),
        projection.boundary_impulses_buffer(),
        projection.status_buffer(),
    ]
    .into_iter()
    .flat_map(|b| read(engine, b).into_iter().map(f32::to_bits))
    .collect()
}

#[test]
fn source_preserves_reference_and_reduction_width() {
    assert_eq!(
        projection_source(32),
        concat!(
            include_str!("../shaders/soil_contact_projection.wgsl"),
            "\n",
            include_str!("../shaders/soil_boundary_projection.wgsl")
        )
    );
    for width in 1..32 {
        let source = projection_source(width);
        assert!(!source.contains("array<f32,1024>"));
        assert!(!source.contains("array<f32,32>"));
        assert!(!source.contains("array<vec3<u32>,32>"));
        assert!(source.contains(&format!("array<f32,{}>", width * width)));
        assert!(source.contains("array<Moments,64>"));
        assert!(source.contains("var width=32u; width>0u; width/=2u"));
    }
    for width in 1..=32 {
        let source = shared_boundary_source(width);
        for name in ["lower", "q", "r"] {
            assert!(source.contains(&format!(
                "var<workgroup> {name}: array<f32,{}>;",
                width * width
            )));
        }
        assert!(source.contains(&format!("var lower: array<f32,{}>;", width * width)));
        assert!(source.contains("@workgroup_size(1)\nfn boundary()"));
        assert!(source.contains("array<Moments,64>"));
        assert!(source.contains(&format!("x: ptr<function,array<f32,{width}>>")));
        assert!(!source.contains("ptr<workgroup"));
        assert!(source.contains("fn independent_boundary(key: vec3<u32>, count: u32)"));
        let cached = cached_boundary_source(width);
        assert!(cached.contains("fn scan_boundaries_full("));
        assert!(cached.contains("if !boundary_cache_complete { return scan_boundaries_full("));
        assert!(cached.contains("if !present { return; }\n    cache_boundary_nodes();"));
    }
}

#[test]
#[ignore = "requires GPU"]
fn boundary_cache_overflow_and_reuse_preserve_full_scan_outputs() {
    let engine = Engine::new_gpu().unwrap();
    let nodes_buffer = buffer(&engine, &[0; 130 * 32]);
    let spans = buffer(&engine, bytemuck::cast_slice(&[[0_u32, 130]]));
    let mass = buffer(
        &engine,
        bytemuck::cast_slice(&[2.0_f32, 0., 0., 0., 2., 0., 0., 0., 2.]),
    );
    let jacobian = buffer(
        &engine,
        bytemuck::cast_slice(&[
            1.0_f32, 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ]),
    );
    let free = buffer(&engine, bytemuck::cast_slice(&[-0.1_f32, -0.1, 0.05]));
    let input = || SoilProjectionInputs {
        nodes: &nodes_buffer,
        spans: &spans,
        mass: &mass,
        jacobians: &jacobian,
        free_rates: &free,
    };
    let capacity = SoilProjectionCapacity {
        dofs: 3,
        bodies: 1,
        nodes: 130,
        max_owned_bytes: 240,
    };
    let baseline =
        SoilContactProjectionGpu::new_with_fixed_boundaries(&engine, input(), capacity).unwrap();
    assert!(
        SoilContactProjectionGpu::new_with_fixed_boundaries_cached(
            &engine,
            input(),
            SoilProjectionCapacity {
                max_owned_bytes: 239,
                ..capacity
            }
        )
        .is_err()
    );
    let candidate =
        SoilContactProjectionGpu::new_with_fixed_boundaries_cached(&engine, input(), capacity)
            .unwrap();
    assert_eq!(
        baseline.owned_buffer_bytes(),
        candidate.owned_buffer_bytes()
    );
    for count in [130, 128, 1, 0, 129, 2] {
        let nodes: Vec<_> = (0..130)
            .map(|i| SoilProjectionNode {
                arm_mass: [0.0, 0.0, 0.0, 0.01],
                momentum: [-0.001, -0.002, 0.0003, if i < count { 4.0 } else { 0.0 }],
            })
            .collect();
        engine
            .queue()
            .unwrap()
            .write_buffer(&nodes_buffer, 0, bytemuck::cast_slice(&nodes));
        submit(&engine, &baseline, true);
        submit(&engine, &candidate, true);
        assert_eq!(
            read(&engine, baseline.status_buffer())[0].to_bits(),
            0,
            "count={count}"
        );
        assert_eq!(
            output_words(&engine, &baseline),
            output_words(&engine, &candidate),
            "count={count}"
        );
    }
}

#[test]
#[ignore = "requires GPU"]
fn scratch_widths_match_reference_for_two_body_boundaries() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    for n in 1usize..=32 {
        let mut mass = vec![0.0f32; n * n];
        for i in 0..n {
            mass[i * n + i] = 1.0 + i as f32 * 0.125;
        }
        let mut jac = vec![0.0f32; 12 * n];
        for b in 0..2 {
            for c in 0..n {
                jac[(b * 6 + c % 6) * n + c] =
                    (1.0 + 0.1 * (c / 6) as f32) * (1.0 + b as f32 * 0.25);
            }
        }
        let mut nodes = Vec::new();
        for b in 0..2 {
            for k in 0..6 {
                nodes.push(SoilProjectionNode {
                    arm_mass: [
                        (k % 3) as f32 * 0.13 - 0.1,
                        (k / 3) as f32 * 0.17 - 0.07,
                        0.02 * k as f32,
                        0.25,
                    ],
                    momentum: [-0.03, -0.04, 0.01, if b == 0 { 4.0 } else { 1.0 }],
                });
            }
        }
        let buffers = [
            buffer(&engine, bytemuck::cast_slice(&nodes)),
            buffer(&engine, bytemuck::cast_slice(&[[0u32, 6], [6, 6]])),
            buffer(&engine, bytemuck::cast_slice(&mass)),
            buffer(&engine, bytemuck::cast_slice(&jac)),
            buffer(&engine, bytemuck::cast_slice(&vec![-0.1f32; n])),
        ];
        let input = || SoilProjectionInputs {
            nodes: &buffers[0],
            spans: &buffers[1],
            mass: &buffers[2],
            jacobians: &buffers[3],
            free_rates: &buffers[4],
        };
        let capacity = SoilProjectionCapacity {
            dofs: n as u32,
            bodies: 2,
            nodes: 12,
            max_owned_bytes: 65536,
        };
        assert!(
            SoilContactProjectionGpu::prepare(
                &engine,
                input(),
                capacity,
                true,
                n as u32 - 1,
                ProjectionShaderMode::Sized
            )
            .is_err()
        );
        let baseline = SoilContactProjectionGpu::prepare(
            &engine,
            input(),
            capacity,
            true,
            32,
            ProjectionShaderMode::Sized,
        )
        .unwrap();
        let candidate = SoilContactProjectionGpu::prepare(
            &engine,
            input(),
            capacity,
            true,
            n as u32,
            ProjectionShaderMode::Sized,
        )
        .unwrap();
        let specialized =
            (std::env::var("SOIL_PROJECTION_SPECIALIZE").as_deref() == Ok("1")).then(|| {
                SoilContactProjectionGpu::prepare(
                    &engine,
                    input(),
                    capacity,
                    true,
                    n as u32,
                    ProjectionShaderMode::ConstantWidth,
                )
                .unwrap()
            });
        let split = SoilContactProjectionGpu::prepare(
            &engine,
            input(),
            capacity,
            true,
            n as u32,
            ProjectionShaderMode::SplitBoundary,
        )
        .unwrap();
        let shared =
            (std::env::var("SOIL_PROJECTION_SHARED_BOUNDARY").as_deref() == Ok("1")).then(|| {
                SoilContactProjectionGpu::prepare(
                    &engine,
                    input(),
                    capacity,
                    true,
                    n as u32,
                    ProjectionShaderMode::SharedBoundary,
                )
                .unwrap()
            });
        let cached =
            (std::env::var("SOIL_PROJECTION_CACHED_BOUNDARY").as_deref() == Ok("1")).then(|| {
                SoilContactProjectionGpu::prepare(
                    &engine,
                    input(),
                    capacity,
                    true,
                    32,
                    ProjectionShaderMode::CachedBoundary,
                )
                .unwrap()
            });
        assert_eq!(
            baseline.owned_buffer_bytes(),
            candidate.owned_buffer_bytes()
        );
        for empty in [false, true] {
            if empty {
                engine
                    .queue()
                    .unwrap()
                    .write_buffer(&buffers[0], 0, &vec![0; 12 * 32]);
            }
            submit(&engine, &baseline, true);
            submit(&engine, &candidate, true);
            if let Some(specialized) = &specialized {
                submit(&engine, specialized, true);
            }
            submit(&engine, &split, true);
            if let Some(shared) = &shared {
                submit(&engine, shared, true);
            }
            if let Some(cached) = &cached {
                submit(&engine, cached, true);
            }
            assert_eq!(
                read(&engine, baseline.status_buffer())[0].to_bits(),
                0,
                "width={n} empty={empty}"
            );
            assert_eq!(
                read(&engine, candidate.status_buffer())[0].to_bits(),
                0,
                "width={n} empty={empty}"
            );
            let a = output_words(&engine, &baseline);
            let b = output_words(&engine, &candidate);
            assert_eq!(a, b, "width={n} empty={empty}");
            assert_eq!(
                a,
                output_words(&engine, &split),
                "split width={n} empty={empty}"
            );
            if let Some(specialized) = &specialized {
                assert_eq!(
                    a,
                    output_words(&engine, specialized),
                    "specialized width={n} empty={empty}"
                );
            }
            if let Some(shared) = &shared {
                assert_eq!(
                    a,
                    output_words(&engine, shared),
                    "shared boundary width={n} empty={empty}"
                );
            }
            if let Some(cached) = &cached {
                assert_eq!(
                    a,
                    output_words(&engine, cached),
                    "cached boundary width={n} empty={empty}"
                );
            }
            if empty {
                assert_eq!(read(&engine, candidate.rates_buffer()), vec![-0.1; n]);
            }
        }
        println!(
            "scratch width={n}: occupied two-body boundaries and empty contact outputs bit-exact"
        );
    }
}

#[test]
#[ignore = "requires GPU timestamps and the saved actual tractor capture"]
fn scratch_capacity_paired_captured_benchmark() {
    let path = std::path::PathBuf::from(
        std::env::var_os("SOIL_PROJECTION_CAPTURE_DIR").expect("capture directory required"),
    )
    .join("frame-24-state-0");
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path.join("manifest.json")).unwrap()).unwrap();
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    println!("adapter={:?}", engine.adapter_info().unwrap());
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let buffers: [wgpu::Buffer; 5] = std::array::from_fn(|i| {
        let bytes = std::fs::read(path.join(format!("input-{i}.bin"))).unwrap();
        let words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        buffer(&engine, bytemuck::cast_slice(&words))
    });
    let input = || SoilProjectionInputs {
        nodes: &buffers[0],
        spans: &buffers[1],
        mass: &buffers[2],
        jacobians: &buffers[3],
        free_rates: &buffers[4],
    };
    let capacity = SoilProjectionCapacity {
        dofs: manifest["dofs"].as_u64().unwrap() as u32,
        bodies: manifest["bodies"].as_u64().unwrap() as u32,
        nodes: manifest["nodes"].as_u64().unwrap() as u32,
        max_owned_bytes: u64::MAX,
    };
    let specialize = match std::env::var("SOIL_PROJECTION_SPECIALIZE") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "1" => true,
        _ => panic!("SOIL_PROJECTION_SPECIALIZE must be absent or 1"),
    };
    println!("candidate_constant_width={specialize}");
    let split = match std::env::var("SOIL_PROJECTION_SPLIT_BOUNDARY") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "1" => true,
        _ => panic!("SOIL_PROJECTION_SPLIT_BOUNDARY must be absent or 1"),
    };
    let shared = match std::env::var("SOIL_PROJECTION_SHARED_BOUNDARY") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "1" => true,
        _ => panic!("SOIL_PROJECTION_SHARED_BOUNDARY must be absent or 1"),
    };
    let cached = match std::env::var("SOIL_PROJECTION_CACHED_BOUNDARY") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(value) if value == "1" => true,
        _ => panic!("SOIL_PROJECTION_CACHED_BOUNDARY must be absent or 1"),
    };
    assert!(
        usize::from(specialize) + usize::from(split) + usize::from(shared) + usize::from(cached)
            <= 1,
        "select one candidate mode"
    );
    let mode = if cached {
        ProjectionShaderMode::CachedBoundary
    } else if shared {
        ProjectionShaderMode::SharedBoundary
    } else if split {
        ProjectionShaderMode::SplitBoundary
    } else if specialize {
        ProjectionShaderMode::ConstantWidth
    } else {
        ProjectionShaderMode::Sized
    };
    println!("candidate_mode={mode:?}");
    let mut programs = [
        SoilContactProjectionGpu::prepare(
            &engine,
            input(),
            capacity,
            true,
            32,
            ProjectionShaderMode::Sized,
        )
        .unwrap(),
        SoilContactProjectionGpu::prepare(
            &engine,
            input(),
            capacity,
            true,
            if cached { 32 } else { capacity.dofs },
            mode,
        )
        .unwrap(),
    ];
    submit(&engine, &programs[0], true);
    assert_eq!(read(&engine, programs[0].status_buffer())[0].to_bits(), 0);
    let expected = output_words(&engine, &programs[0]);
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("projection scratch timing"),
        ty: wgpu::QueryType::Timestamp,
        count: 40,
    });
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("projection scratch times"),
        size: 40 * 8,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    const PASS_QUERIES: u32 = 10 * 34 * 10;
    assert!(PASS_QUERIES <= wgpu::QUERY_SET_MAX_QUERIES);
    let pass_profile: Option<[(wgpu::QuerySet, wgpu::Buffer); 2]> =
        match std::env::var("SOIL_PROJECTION_PASS_PROFILE") {
            Err(std::env::VarError::NotPresent) => None,
            Ok(value) if value == "1" => Some(std::array::from_fn(|_| {
                (
                    device.create_query_set(&wgpu::QuerySetDescriptor {
                        label: Some("projection pass attribution"),
                        ty: wgpu::QueryType::Timestamp,
                        count: PASS_QUERIES,
                    }),
                    device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("projection pass attribution resolve"),
                        size: u64::from(PASS_QUERIES) * 8,
                        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                        mapped_at_creation: false,
                    }),
                )
            })),
            _ => panic!("SOIL_PROJECTION_PASS_PROFILE must be absent or 1"),
        };
    println!("pass_attribution={}", pass_profile.is_some());
    let mut wall = [[0.0f64; 2]; 10];
    for sample in 0..10 {
        for slot in 0..2 {
            let variant = (sample + slot) % 2;
            let projection = &mut programs[variant];
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
            let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let started = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.clear_buffer(projection.status_buffer(), 0, None);
            let index = (sample * 2 + variant) as u32 * 2;
            encoder.write_timestamp(&queries, index);
            for repeat in 0..34 {
                projection.pass_profile = pass_profile.as_ref().map(|profiles| {
                    (
                        profiles[variant].0.clone(),
                        (sample * 34 + repeat) as u32 * 10,
                    )
                });
                projection.encode(&mut encoder);
            }
            encoder.write_timestamp(&queries, index + 1);
            queue.submit([encoder.finish()]);
            device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: Some(std::time::Duration::from_secs(5)),
                })
                .unwrap();
            wall[sample][variant] = started.elapsed().as_secs_f64() * 1000.0;
            assert_eq!(allocations.finish().observations, 0);
            assert_eq!(dispatches.finish().encoded_dispatches, 170);
            let transfers = transfers.finish();
            assert_eq!(transfers.upload_bytes + transfers.readback_bytes, 0);
            assert_eq!(
                output_words(&engine, projection),
                expected,
                "sample={sample} variant={variant}"
            );
        }
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.resolve_query_set(&queries, 0..40, &resolve, 0);
    if let Some(profiles) = &pass_profile {
        for (queries, buffer) in profiles {
            encoder.resolve_query_set(queries, 0..PASS_QUERIES, buffer, 0);
        }
    }
    queue.submit([encoder.finish()]);
    let words = read(&engine, &resolve);
    let ticks: Vec<_> = words
        .chunks_exact(2)
        .map(|w| u64::from(w[0].to_bits()) | (u64::from(w[1].to_bits()) << 32))
        .collect();
    if let Some(profiles) = &pass_profile {
        for (variant, (_, buffer)) in profiles.iter().enumerate() {
            let words = read(&engine, buffer);
            let times: Vec<_> = words
                .chunks_exact(2)
                .map(|pair| u64::from(pair[0].to_bits()) | (u64::from(pair[1].to_bits()) << 32))
                .collect();
            for sample in 0..10 {
                for (stage, name) in ["reduce", "assemble", "solve", "boundary", "publish"]
                    .into_iter()
                    .enumerate()
                {
                    let ticks: u64 = (0..34)
                        .map(|repeat| {
                            let i = (sample * 34 + repeat) * 10 + stage * 2;
                            times[i + 1]
                                .checked_sub(times[i])
                                .expect("ordered pass timestamps")
                        })
                        .sum();
                    let ms = ticks as f64 * f64::from(queue.get_timestamp_period()) / 1e6;
                    assert!(ms.is_finite() && ms > 0.0);
                    println!(
                        "pass sample={sample} warmup={} variant={variant} stage={name} gpu_ms={ms}",
                        sample < 2
                    );
                }
            }
        }
    }
    for sample in 0..10 {
        for variant in 0..2 {
            let index = (sample * 2 + variant) * 2;
            let gpu_ms = (ticks[index + 1] - ticks[index]) as f64
                * f64::from(queue.get_timestamp_period())
                / 1e6;
            assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
            println!(
                "scratch sample={sample} warmup={} variant={variant} projections=34 completed_wall_ms={} gpu_span_ms={gpu_ms}",
                sample < 2,
                wall[sample][variant]
            );
        }
    }
    println!(
        "fixed-input component replay only: no MPM or simulated-time advancement; timestamps include between-pass gaps"
    );
}
