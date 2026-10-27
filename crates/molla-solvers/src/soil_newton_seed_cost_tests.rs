use super::*;

fn order(pair: usize) -> [usize; 2] {
    if pair % 2 == 0 { [0, 1] } else { [1, 0] }
}

fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty() && values.iter().all(|x| x.is_finite()));
    let mut values = values.to_vec();
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
fn newton_seed_cost_schedule_and_statistics() {
    assert_eq!(
        (0..4).flat_map(order).collect::<Vec<_>>(),
        [0, 1, 1, 0, 0, 1, 1, 0]
    );
    assert_eq!(median(&[3., 1., 4., 2.]), 3.);
    assert_eq!(median(&[2.]), 2.);
}

pub(super) fn compare(
    engine: &Engine,
    solver: &ExperimentalSoilNewtonGpu,
    fixture: &Fixture,
    reference: &Reference,
    seed: &[u8],
    packet: &std::path::Path,
    output: &std::path::Path,
) {
    let projection = std::env::var_os("MOLLA_NEWTON_PUBLICATION_PROJECTION").is_some();
    let clocks = std::env::var_os("MOLLA_NEWTON_CLOCK_COST").is_some();
    let unroll = std::env::var_os("MOLLA_NEWTON_PRECONDITION_UNROLL").is_some();
    let energy = std::env::var_os("MOLLA_NEWTON_PUBLICATION_ENERGY").is_some();
    let snapshot = std::env::var_os("MOLLA_NEWTON_CONTROL_SNAPSHOT").is_some();
    let input_cache = std::env::var_os("MOLLA_NEWTON_PUBLICATION_INPUT_CACHE").is_some();
    let scalar_product = std::env::var_os("MOLLA_NEWTON_SCALAR_PRODUCT").is_some();
    let operator_rows = std::env::var_os("MOLLA_NEWTON_OPERATOR_ROWS").is_some();
    let residual_fusion = std::env::var_os("MOLLA_NEWTON_RESIDUAL_FUSION").is_some();
    assert!(!input_cache || cfg!(feature = "soa-scratch-newton"));
    assert!(!scalar_product || cfg!(feature = "soa-scratch-newton"));
    assert!(!operator_rows || cfg!(feature = "soa-scratch-newton"));
    assert!(!residual_fusion || cfg!(feature = "soa-scratch-newton"));
    assert!(
        u8::from(projection)
            + u8::from(clocks)
            + u8::from(unroll)
            + u8::from(energy)
            + u8::from(snapshot)
            + u8::from(input_cache)
            + u8::from(scalar_product)
            + u8::from(operator_rows)
            + u8::from(residual_fusion)
            <= 1
    );
    let identical_seed = projection
        || clocks
        || unroll
        || energy
        || snapshot
        || input_cache
        || scalar_product
        || operator_rows
        || residual_fusion;
    let bytes = std::fs::read(packet.join("newton.bin")).unwrap();
    let hash = blake3::hash(&bytes).to_hex().to_string();
    let root = output.join(&hash);
    std::fs::create_dir_all(output).unwrap();
    std::fs::create_dir(&root).expect("fresh seed comparison output");
    let device = engine.device().unwrap();
    assert!(
        device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    );
    assert_eq!(seed.len() as u64, solver.state_buffer().size());
    let seeds = [
        storage(
            engine,
            &if identical_seed {
                seed.to_vec()
            } else {
                vec![0u8; seed.len()]
            },
        ),
        storage(engine, seed),
    ];
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("warm cold solve timing"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("warm cold solve timing"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let execute = |mode: usize| {
        if projection {
            solver.set_publication_projection_probe(engine, mode != 0);
        }
        if energy {
            solver.set_publication_energy_probe(engine, mode != 0);
        }
        if clocks {
            solver.set_phase_clock_probe(engine, mode != 0);
        }
        if unroll {
            solver.set_precondition_unroll_probe(engine, mode != 0);
        }
        if snapshot {
            solver.set_control_snapshot_probe(engine, mode != 0);
        }
        #[cfg(feature = "soa-scratch-newton")]
        if input_cache {
            solver.set_publication_input_cache_probe(engine, mode != 0);
        }
        #[cfg(feature = "soa-scratch-newton")]
        if scalar_product {
            solver.set_scalar_product_probe(engine, mode != 0);
        }
        #[cfg(feature = "soa-scratch-newton")]
        if operator_rows {
            solver.set_operator_rows_probe(engine, mode != 0);
        }
        #[cfg(feature = "soa-scratch-newton")]
        if residual_fusion {
            solver.set_residual_fusion_probe(engine, mode != 0);
        }
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_reset(&mut encoder);
        encoder.copy_buffer_to_buffer(&seeds[mode], 0, solver.state_buffer(), 0, seed.len() as u64);
        encoder.write_timestamp(&queries, 0);
        solver.encode(&mut encoder);
        encoder.write_timestamp(&queries, 1);
        encoder.resolve_query_set(&queries, 0..2, &resolved, 0);
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let traffic = traffic.finish();
        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
        assert_eq!(
            dispatch.finish().encoded_dispatches,
            solver.replay_dispatch_count()
        );
        let ticks: [u64; 2] = bytemuck::pod_read_unaligned(&capture::read(engine, &resolved));
        assert!(ticks[1] >= ticks[0]);
        let ms = (ticks[1] - ticks[0]) as f64
            * f64::from(engine.queue().unwrap().get_timestamp_period())
            / 1e6;
        (ms, mixed_tests::snapshot(engine, solver))
    };
    let saved = [execute(0).1, execute(1).1];
    if identical_seed {
        assert_eq!(
            saved[0], saved[1],
            "same-input mode comparison exact public buffers"
        );
    }
    for (mode, buffers) in saved.iter().enumerate() {
        for (name, data) in ["states", "rates", "outputs", "status", "publication"]
            .iter()
            .zip(buffers)
        {
            std::fs::write(root.join(format!("{mode}-{name}.bin")), data).unwrap();
        }
    }
    let reports: [[u32; 8]; 2] =
        std::array::from_fn(|i| bytemuck::pod_read_unaligned(&saved[i][3]));
    let publications: [[u32; 8]; 2] =
        std::array::from_fn(|i| bytemuck::pod_read_unaligned(&saved[i][4]));
    assert_eq!(saved[1][3], bytes[64..96]);
    assert_eq!(saved[1][4], bytes[96..128]);
    let n = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let c = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    let start = 128 + 4 * n * n + 8 * n;
    let archived: Vec<u8> = (0..c.max(1))
        .flat_map(|i| {
            let offset = start + i * (32 * n + 128) + 32 * n + 64;
            bytes[offset..offset + 64].iter().copied()
        })
        .collect();
    assert_eq!(saved[1][0], archived);
    let physical: [f64; 2] = std::array::from_fn(|i| {
        reference.physical_error(&mixed_tests::states(&saved[i][0], reference))
    });
    let delta = mixed_tests::public_delta(&saved[0], &saved[1], reference);
    let mut timings: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    let mut pairs = Vec::new();
    let mut clock_samples = Vec::new();
    let mut clock_counts = None;
    for pair in 0..8 {
        let mut sample = [0.; 2];
        for mode in order(pair) {
            let (ms, buffers) = execute(mode);
            assert_eq!(
                buffers, saved[mode],
                "mode {mode} pair {pair} exact repeats"
            );
            sample[mode] = ms;
            timings[mode].push(ms);
            if clocks {
                use crate::soil_newton_precise_tests::shader_clock::{TRACE_BYTES, phase_names};
                let bytes = capture::read(engine, solver.clock_buffer());
                assert_eq!(bytes.len() as u64, TRACE_BYTES);
                let words: &[u32] = bytemuck::cast_slice(&bytes);
                if mode == 0 {
                    assert!(words.iter().all(|&word| word == 0));
                }
                let counts: Vec<_> = words.chunks_exact(4).map(|v| v[2]).collect();
                if mode == 1 {
                    assert!(counts[0] > 0 && counts[1] > 0 && counts[2] > 0 && counts[3] > 0);
                    if let Some(expected) = &clock_counts {
                        assert_eq!(&counts, expected);
                    } else {
                        clock_counts = Some(counts);
                    }
                }
                let phases: Vec<_> = phase_names().iter().zip(words.chunks_exact(4)).map(|(name, v)| {
                    assert_eq!(v[3], 0);
                    let ticks = u64::from(v[0]) | (u64::from(v[1]) << 32);
                    assert!(v[2] == 0 || ticks > 0);
                    serde_json::json!({"phase": name, "calls": v[2], "inclusive_subgroup_ticks": ticks})
                }).collect();
                clock_samples
                    .push(serde_json::json!({"pair": pair, "mode": mode, "phases": phases}));
            }
        }
        pairs.push(100. * (sample[1] / sample[0] - 1.));
    }
    let valid = (0..2).all(|i| {
        reports[i][0] == 0
            && publications[i][0] == 0
            && physical[i].is_finite()
            && physical[i] <= 1e-6
    });
    let result = serde_json::json!({
        "acceptance":false, "scope":if identical_seed {"one compiled branch-controlled method, identical seed; excludes mode upload, restore and frame work; not an original-binary performance comparison"} else {"one compiled method, identical captured inputs, zero versus same-solve seed; excludes restore/copy and frame work"},
        "method":solver.failure_method().name(), "input_blake3":hash,
        "mode_order":if clocks {["clock sampling disabled", "clock sampling enabled"]} else if projection {["serial projection", "cooperative projection"]} else if unroll {["looped precondition", "unrolled precondition"]} else if snapshot {["scalar control loads", "control snapshots"]} else if input_cache {["repeated input validation", "cached immutable validation"]} else if scalar_product {["paired scalar product", "guarded scalar product"]} else if residual_fusion {["separate residual passes", "fused residual norm"]} else if operator_rows {["serial operator rows", "interleaved operator rows"]} else if energy {["serial energy", "cooperative energy"]} else {["cold", "warm"]}, "reports":reports, "publication":publications,
        "clock_samples":clock_samples,
        "physical":physical, "physical_bound":1e-6, "valid":valid,
        "public_delta":delta, "public_bound":2e-5, "public_passed":delta<=2e-5,
        "timestamps_ms":timings, "upper_medians_ms":timings.each_ref().map(|x| median(x)),
        "paired_change_percent":pairs, "paired_upper_median_percent":median(&pairs),
        "exact_repeats":16, "warm_matches_archived":true,
    });
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    eprintln!("warm/cold cost {result}");
    assert!(valid && delta <= 2e-5, "warm/cold numerical gates");
    for mode in 0..2 {
        execute(mode);
        super::publication::verify_outputs(
            engine,
            solver.outputs_buffer(),
            solver.rates_buffer(),
            fixture,
            reference,
        );
    }
}
