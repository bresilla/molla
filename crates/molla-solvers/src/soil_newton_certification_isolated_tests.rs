use super::*;

fn snapshot(engine: &Engine, solver: &ExperimentalSoilNewtonGpu) -> [Vec<u8>; 4] {
    [
        solver.state_buffer(),
        solver.rates_buffer(),
        solver.outputs_buffer(),
        solver.publication_status_buffer(),
    ]
    .map(|buffer| capture::read(engine, buffer))
}

fn measure(
    engine: &Engine,
    solver: &ExperimentalSoilNewtonGpu,
    queries: &wgpu::QuerySet,
    resolved: &wgpu::Buffer,
) -> f64 {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    encoder.write_timestamp(queries, 0);
    solver.encode(&mut encoder);
    encoder.write_timestamp(queries, 1);
    encoder.resolve_query_set(queries, 0..2, resolved, 0);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(dispatch.finish().encoded_dispatches, 1);
    let bytes = capture::read(engine, resolved);
    let ticks: [u64; 2] = bytemuck::pod_read_unaligned(&bytes[..16]);
    assert!(ticks[1] >= ticks[0]);
    (ticks[1] - ticks[0]) as f64 * f64::from(engine.queue().unwrap().get_timestamp_period()) / 1e6
}

pub(super) fn compare(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    f: &Fixture,
    p: &Reference,
    packet: &std::path::Path,
    output: &std::path::Path,
) {
    let bytes = std::fs::read(packet.join("newton.bin")).unwrap();
    let hash = blake3::hash(&bytes).to_hex().to_string();
    let reference =
        std::path::PathBuf::from(std::env::var_os("MOLLA_NEWTON_CERTIFICATION_REFERENCE").unwrap())
            .join(&hash);
    assert_eq!(
        std::fs::read(reference.join("original-newton.bin")).unwrap(),
        bytes
    );
    let expected: [Vec<u8>; 4] = ["states", "rates", "outputs", "publication"]
        .map(|name| std::fs::read(reference.join(format!("v1-{name}.bin"))).unwrap());
    let status: [u32; 8] = bytemuck::pod_read_unaligned(&expected[3]);
    assert_eq!(status[0], 0);
    let root = output.join(&hash);
    std::fs::create_dir_all(output).unwrap();
    std::fs::create_dir(&root).expect("fresh isolated certification output");
    let indices = std::env::var_os("MOLLA_NEWTON_CERTIFICATION_PARALLEL_INDICES").is_some();
    let solvers = [1, 2].map(|copies| {
        if indices {
            return crate::soil_newton_gpu::distributed::mixed::active_indices::probe_new(
                engine,
                SoilNewtonInputs {
                    mass: input.mass,
                    free_rates: input.free_rates,
                    point_rows: input.point_rows,
                    point_response: input.point_response,
                    contacts: input.contacts,
                    upstream_status: input.upstream_status,
                },
                capacity(p.free.len(), f.contacts.len()),
                copies == 2,
                true,
            )
            .unwrap();
        }
        crate::soil_newton_gpu::distributed::mixed::certification_cost::isolated_new(
            engine,
            SoilNewtonInputs {
                mass: input.mass,
                free_rates: input.free_rates,
                point_rows: input.point_rows,
                point_response: input.point_response,
                contacts: input.contacts,
                upstream_status: input.upstream_status,
            },
            capacity(p.free.len(), f.contacts.len()),
            copies,
        )
        .unwrap()
    });
    let queries = engine
        .device()
        .unwrap()
        .create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("isolated certification timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 2,
        });
    let resolved = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("isolated certification timestamps"),
            size: 16,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
    let cold = vec![0; expected[0].len()];
    let mut poison = expected[0].clone();
    let active = p.contacts.first().unwrap().index;
    poison[64 * active + 12..64 * active + 16].copy_from_slice(&f32::INFINITY.to_le_bytes());
    let mut records = Vec::new();
    let mut checks = 0;
    for (label, state, measured) in [
        ("accepted", &expected[0], true),
        ("cold", &cold, true),
        ("poisoned", &poison, false),
        ("recovered", &expected[0], false),
    ] {
        let mut previous = None;
        for solver in &solvers {
            assert_eq!(solver.state_buffer().size() as usize, state.len());
            engine
                .queue()
                .unwrap()
                .write_buffer(solver.state_buffer(), 0, state);
        }
        let pairs = if measured { 10 } else { 2 };
        for pair in 0..pairs {
            for order in 0..2 {
                let variant = (pair + order) % 2;
                let solver = &solvers[variant];
                let ms = measure(engine, solver, &queries, &resolved);
                let observed = snapshot(engine, solver);
                assert_eq!(&observed[0], state, "certification must not mutate states");
                let status: [u32; 8] = bytemuck::pod_read_unaligned(&observed[3]);
                if label == "accepted" || label == "recovered" {
                    assert_eq!(observed, expected, "archived method-29 publication parity");
                    super::super::publication::verify_outputs(
                        engine,
                        solver.outputs_buffer(),
                        solver.rates_buffer(),
                        f,
                        p,
                    );
                } else {
                    assert_ne!(status[0], 0, "rejected-state fixture");
                    if label == "poisoned" {
                        assert_ne!(status[0] & 1, 0);
                    }
                    assert!(observed[1].iter().chain(&observed[2]).all(|&b| b == 0));
                }
                if let Some(previous) = &previous {
                    assert_eq!(&observed, previous, "exact copies/repeats");
                }
                previous = Some(observed.clone());
                checks += 1;
                if pair == 0 {
                    for (name, data) in ["states", "rates", "outputs", "publication"]
                        .into_iter()
                        .zip(&observed)
                    {
                        std::fs::write(root.join(format!("{label}-v{variant}-{name}.bin")), data)
                            .unwrap();
                    }
                }
                records.push(serde_json::json!({"state":label,"pair":pair,"variant":variant,"copies":if indices {1}else{variant+1},"parallel_indices":indices && variant==1,"warmup":pair<2,"measured":measured,"gpu_ms":ms,"fault":status[0]}));
                std::fs::write(
                    root.join("samples.json"),
                    serde_json::to_vec_pretty(&records).unwrap(),
                )
                .unwrap();
            }
        }
    }
    assert_eq!(checks, 48);
    std::fs::write(root.join("complete.json"), serde_json::to_vec_pretty(&serde_json::json!({"packet":hash,"exact_checks":checks,"reference_parity":true,"physical_publication":true,"scope":"standalone certification including active-index setup, accepted/cold endpoints only; not additive full-solve attribution"})).unwrap()).unwrap();
    eprintln!("isolated certification case={hash} exact_checks={checks}");
}
