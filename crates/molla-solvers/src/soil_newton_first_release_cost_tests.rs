use super::*;

fn order(pair: usize) -> [usize; 2] {
    if pair % 2 == 0 { [0, 1] } else { [1, 0] }
}

#[test]
fn newton_first_release_pair_order() {
    assert_eq!(
        (0..4).flat_map(order).collect::<Vec<_>>(),
        [0, 1, 1, 0, 0, 1, 1, 0]
    );
}

pub(super) fn compare(
    engine: &Engine,
    solver: &ExperimentalSoilNewtonGpu,
    fixture: &Fixture,
    reference: &Reference,
    packet: &std::path::Path,
    output: &std::path::Path,
) {
    let bytes = std::fs::read(packet.join("newton.bin")).unwrap();
    let hash = blake3::hash(&bytes).to_hex().to_string();
    let root = output.join(&hash);
    std::fs::create_dir_all(output).unwrap();
    std::fs::create_dir(&root).expect("fresh first-release timing output");
    let prior =
        std::path::PathBuf::from(std::env::var_os("MOLLA_NEWTON_FIRST_RELEASE_REFERENCE").unwrap())
            .join(&hash);
    let prior_profile: serde_json::Value =
        serde_json::from_slice(&std::fs::read(prior.join("profile.json")).unwrap()).unwrap();
    assert_eq!(prior_profile["input_blake3"], hash);
    assert_eq!(
        prior_profile["experimental_policy"],
        "guarded-first-release"
    );
    let device = engine.device().unwrap();
    assert!(
        device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    );
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("first release timing"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("first release timing"),
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let execute = |mode: usize| {
        crate::soil_newton_gpu::first_release::set_policy(solver, engine, mode == 1);
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_reset(&mut encoder);
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
        assert!(ticks[1] > ticks[0]);
        let ms = (ticks[1] - ticks[0]) as f64
            * f64::from(engine.queue().unwrap().get_timestamp_period())
            / 1e6;
        (ms, mixed_tests::snapshot(engine, solver))
    };
    let saved = [execute(0).1, execute(1).1];
    for i in [0, 1, 2, 4] {
        assert_eq!(
            saved[0][i], saved[1][i],
            "control/candidate physical buffers"
        );
    }
    assert_eq!(saved[0][3], bytes[64..96]);
    assert_eq!(saved[0][4], bytes[96..128]);
    let word = |i: usize| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()) as usize;
    let (n, c) = (word(1), word(2));
    let start = 128 + 4 * n * n + 8 * n;
    let archived: Vec<u8> = (0..c.max(1))
        .flat_map(|i| {
            let offset = start + i * (32 * n + 128) + 32 * n + 64;
            bytes[offset..offset + 64].iter().copied()
        })
        .collect();
    assert_eq!(saved[0][0], archived);
    for (i, name) in ["states", "rates", "outputs", "status", "publication"]
        .iter()
        .enumerate()
    {
        assert_eq!(
            saved[1][i],
            std::fs::read(prior.join(format!("{name}.bin"))).unwrap(),
            "prior candidate {name}"
        );
        for mode in 0..2 {
            std::fs::write(root.join(format!("{mode}-{name}.bin")), &saved[mode][i]).unwrap();
        }
    }
    let physical: [f64; 2] = std::array::from_fn(|i| {
        reference.physical_error(&mixed_tests::states(&saved[i][0], reference))
    });
    assert!(physical.iter().all(|p| p.is_finite() && *p <= 1e-6));
    let reports: [[u32; 8]; 2] =
        std::array::from_fn(|i| bytemuck::pod_read_unaligned(&saved[i][3]));
    assert!(reports.iter().all(|r| r[0] == 0));
    for mode in 0..2 {
        assert_eq!(execute(mode).1, saved[mode]);
        super::publication::verify_outputs(
            engine,
            solver.outputs_buffer(),
            solver.rates_buffer(),
            fixture,
            reference,
        );
    }
    let mut samples = Vec::new();
    for pair in 0..8 {
        let mut ms = [0.; 2];
        for mode in order(pair) {
            let (elapsed, buffers) = execute(mode);
            assert_eq!(buffers, saved[mode], "pair={pair} mode={mode}");
            assert!(elapsed.is_finite() && elapsed > 0.);
            ms[mode] = elapsed;
        }
        samples.push(serde_json::json!({"pair":pair,"order":order(pair),"ms":ms}));
    }
    let result = serde_json::json!({"acceptance":false,"input_blake3":hash,
        "scope":"one compiled branch-controlled program; excludes reset, uniform upload and readbacks; not original-binary or whole-frame timing",
        "mode_order":["control","guarded-first-release"],"samples":samples,"reports":reports,
        "physical":physical,"physical_limit":1e-6,"exact_repeats":16,"policy_upload_bytes_per_sample":4,
        "archived_control_exact":true,"prior_candidate_exact":true,"physical_buffers_equal":true});
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    eprintln!("first release paired {result}");
}
