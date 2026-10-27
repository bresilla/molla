use super::*;

pub(super) fn compare(
    engine: &Engine,
    input: SoilNewtonInputs<'_>,
    f: &Fixture,
    p: &Reference,
    packet: &std::path::Path,
    output: &std::path::Path,
) {
    use crate::soil_newton_gpu::distributed::NonlinearProbe;
    let bytes = std::fs::read(packet.join("newton.bin")).unwrap();
    let hash = blake3::hash(&bytes).to_hex().to_string();
    std::fs::create_dir_all(output).unwrap();
    let root = output.join(&hash);
    std::fs::create_dir(&root).expect("fresh nonlinear replay");
    std::fs::write(root.join("original-newton.bin"), bytes).unwrap();
    let mut baseline = None;
    let mut records = Vec::new();
    let mut failures = 0;
    let phase_profile = std::env::var("MOLLA_NEWTON_PHASE_TIMESTAMPS").as_deref() == Ok("1");
    let modes: &[u32] = if phase_profile {
        &[0, 5]
    } else if std::env::var("MOLLA_NEWTON_HYBRID_BOUNDARY").as_deref() == Ok("1") {
        &[0, 1, 5]
    } else {
        &[0, 1, 2, 3, 4]
    };
    let probes: Vec<_> = modes
        .iter()
        .copied()
        .map(|mode| {
            println!("distributed-nonlinear constructing mode={mode}");
            NonlinearProbe::new(
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
                mode,
            )
            .unwrap()
        })
        .collect();
    for probe in &probes {
        let mode = probe.mode();
        for repeat in 0..2 {
            let mut encoder = engine
                .device()
                .unwrap()
                .create_command_encoder(&Default::default());
            #[cfg(feature = "resource-counters")]
            let scopes = (
                molla_core::gpu_metrics::DeviceScope::start().unwrap(),
                molla_core::gpu_metrics::TransferScope::start().unwrap(),
                molla_core::gpu_metrics::DispatchScope::start().unwrap(),
            );
            probe.encode(&mut encoder);
            engine.queue().unwrap().submit([encoder.finish()]);
            #[cfg(feature = "resource-counters")]
            {
                let a = scopes.0.finish();
                let t = scopes.1.finish();
                let d = scopes.2.finish();
                assert_eq!(a.observations, 0);
                assert!(!a.invalid && !t.overflowed && !d.overflowed);
                assert_eq!((t.upload_bytes, t.readback_bytes), (0, 0));
                assert_eq!(
                    d.encoded_dispatches,
                    match mode {
                        0 => 1,
                        1 | 5 => 65 * 3,
                        4 => 65 * (3 + 2 * 8),
                        _ => 65 * (3 + 4 * 8),
                    }
                );
            }
            let buffers = [
                ("states", probe.owner.state_buffer()),
                ("rates", probe.owner.rates_buffer()),
                ("outputs", probe.owner.outputs_buffer()),
                ("status", probe.owner.status_buffer()),
                ("publication", probe.owner.publication_status_buffer()),
            ];
            if let Some(buffer) = probe.indirect_buffer() {
                let args: [u32; 6] = bytemuck::pod_read_unaligned(&capture::read(engine, buffer));
                assert_eq!(
                    args,
                    [0, 1, 1, 0, 1, 1],
                    "completed solve disables indirect work"
                );
            }
            let captured: Vec<_> = buffers
                .iter()
                .map(|(name, buffer)| {
                    let bytes = capture::read(engine, buffer);
                    std::fs::write(root.join(format!("m{mode}-r{repeat}-{name}.bin")), &bytes)
                        .unwrap();
                    bytes
                })
                .collect();
            let exact = baseline.as_ref().is_none_or(|b| b == &captured);
            if baseline.is_none() {
                baseline = Some(captured.clone());
            }
            let report: [u32; 8] = bytemuck::pod_read_unaligned(&captured[3]);
            let publication: [u32; 8] = bytemuck::pod_read_unaligned(&captured[4]);
            let states: Vec<f32> = captured[0]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let x: Vec<_> = p
                .contacts
                .iter()
                .flat_map(|c| {
                    (0..6).map(|i| {
                        let k = 16 * c.index + if i < 3 { i } else { i + 1 };
                        f64::from(states[k]) + f64::from(states[k + 8])
                    })
                })
                .collect();
            let physical = p.physical_error(&x);
            let valid = report[0] == 0
                && publication[0] == 0
                && x.iter().all(|v| v.is_finite())
                && physical <= 1e-6;
            println!(
                "distributed-nonlinear case={hash} mode={mode} repeat={repeat} fault={} publication={} steps={} iterations={} physical={physical:e} exact={exact} valid={valid}",
                report[0], publication[0], report[1], report[2]
            );
            records.push(serde_json::json!({"mode":mode,"repeat":repeat,"report":report,"publication":publication,"physical":physical,"exact":exact,"valid":valid}));
            std::fs::write(
                root.join("results.json"),
                serde_json::to_vec_pretty(&records).unwrap(),
            )
            .unwrap();
            if valid {
                super::super::publication::verify_outputs(
                    engine,
                    probe.owner.outputs_buffer(),
                    probe.owner.rates_buffer(),
                    f,
                    p,
                );
                verify_solution(f, p, &x);
            }
            failures += usize::from(!exact || !valid);
        }
    }
    assert_eq!(failures, 0, "complete nonlinear comparison");
    if phase_profile {
        profile_phases(engine, &probes, baseline.as_ref().unwrap(), &root);
    }
    if std::env::var_os("MOLLA_NEWTON_DISTRIBUTED_TIMING").is_some() {
        time_probes(engine, &probes, &baseline.unwrap(), &root);
    }
}

fn profile_phases(
    engine: &Engine,
    probes: &[crate::soil_newton_gpu::distributed::NonlinearProbe],
    expected: &[Vec<u8>],
    root: &std::path::Path,
) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let probe = &probes[1];
    assert_eq!(probe.mode(), 5);
    let count = probe.phase_query_count();
    assert_eq!(count, 390);
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("Newton phase timestamps"),
        ty: wgpu::QueryType::Timestamp,
        count: count + 2,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: u64::from(count + 2) * 8,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut records = Vec::new();
    for round in 0..5 {
        for profiled in if round % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        } {
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.write_timestamp(&queries, count);
            #[cfg(feature = "resource-counters")]
            let scopes = (
                molla_core::gpu_metrics::DeviceScope::start().unwrap(),
                molla_core::gpu_metrics::TransferScope::start().unwrap(),
                molla_core::gpu_metrics::DispatchScope::start().unwrap(),
            );
            if profiled {
                probe.encode_profiled(&mut encoder, &queries);
            } else {
                probe.encode(&mut encoder);
            }
            encoder.write_timestamp(&queries, count + 1);
            #[cfg(feature = "resource-counters")]
            {
                let a = scopes.0.finish();
                let t = scopes.1.finish();
                let d = scopes.2.finish();
                assert!(!a.invalid && !t.overflowed && !d.overflowed);
                assert_eq!(
                    (
                        a.observations,
                        t.upload_bytes,
                        t.readback_bytes,
                        d.encoded_dispatches
                    ),
                    (0, 0, 0, 195)
                );
            }
            let first = if profiled { 0 } else { count };
            encoder.resolve_query_set(&queries, first..count + 2, &resolved, 0);
            queue.submit([encoder.finish()]);
            let bytes = capture::read(engine, &resolved);
            let ticks: Vec<_> = bytes
                .chunks_exact(8)
                .map(|b| u64::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let duration = |a: u64, b: u64| {
                b.checked_sub(a).unwrap() as f64 * f64::from(queue.get_timestamp_period()) / 1e6
            };
            let offset = if profiled { count as usize } else { 0 };
            let total_ms = duration(ticks[offset], ticks[offset + 1]);
            assert!(total_ms.is_finite() && total_ms > 0.0);
            let passes: Vec<_> = if profiled {
                ticks[..count as usize].chunks_exact(2).enumerate().map(|(i, pair)| {
                    assert!(pair[0] >= ticks[offset] && pair[1] <= ticks[offset+1]);
                    if i > 0 {assert!(pair[0] >= ticks[2*i-1]);}
                    serde_json::json!({"slot":i/3,"role":i%3,"gpu_ms":duration(pair[0],pair[1])})
                }).collect()
            } else {
                Vec::new()
            };
            let outputs = [
                probe.owner.state_buffer(),
                probe.owner.rates_buffer(),
                probe.owner.outputs_buffer(),
                probe.owner.status_buffer(),
                probe.owner.publication_status_buffer(),
            ];
            let exact = outputs
                .iter()
                .zip(expected)
                .all(|(b, e)| capture::read(engine, b) == *e);
            assert!(exact, "phase timestamps changed numerical outputs");
            records.push(serde_json::json!({"round":round,"profiled":profiled,"exact":exact,"total_gpu_ms":total_ms,"passes":passes}));
            std::fs::write(
                root.join("phases.json"),
                serde_json::to_vec_pretty(&records).unwrap(),
            )
            .unwrap();
            println!(
                "newton-phase-timestamps round={round} profiled={profiled} total_ms={total_ms} exact={exact}"
            );
        }
    }
}

fn time_probes(
    engine: &Engine,
    probes: &[crate::soil_newton_gpu::distributed::NonlinearProbe],
    expected: &[Vec<u8>],
    root: &std::path::Path,
) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    assert!(
        device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    );
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("distributed full solve"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut records = Vec::new();
    for round in 0..9 {
        let mut order: Vec<_> = (0..probes.len()).collect();
        if round % 2 != 0 {
            order.reverse();
        }
        for index in order {
            let probe = &probes[index];
            let mode = probe.mode();
            let start = std::time::Instant::now();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.write_timestamp(&queries, 0);
            probe.encode(&mut encoder);
            encoder.write_timestamp(&queries, 1);
            encoder.resolve_query_set(&queries, 0..2, &resolved, 0);
            queue.submit([encoder.finish()]);
            let bytes = capture::read(engine, &resolved);
            let wall_ms = start.elapsed().as_secs_f64() * 1e3;
            let ticks: [u64; 2] = bytemuck::pod_read_unaligned(&bytes);
            assert!(ticks[1] >= ticks[0]);
            let gpu_ms =
                (ticks[1] - ticks[0]) as f64 * f64::from(queue.get_timestamp_period()) / 1e6;
            assert!(gpu_ms.is_finite() && gpu_ms > 0.0);
            let outputs = [
                probe.owner.state_buffer(),
                probe.owner.rates_buffer(),
                probe.owner.outputs_buffer(),
                probe.owner.status_buffer(),
                probe.owner.publication_status_buffer(),
            ];
            let exact = outputs
                .iter()
                .zip(expected)
                .all(|(b, e)| capture::read(engine, b) == *e);
            records.push(serde_json::json!({"round":round,"mode":mode,"gpu_ms":gpu_ms,"wall_with_query_readback_ms":wall_ms,"exact":exact}));
            std::fs::write(
                root.join("timing.json"),
                serde_json::to_vec_pretty(&records).unwrap(),
            )
            .unwrap();
            println!(
                "distributed-nonlinear-timing round={round} mode={mode} gpu_ms={gpu_ms} wall_ms={wall_ms} exact={exact}"
            );
            assert!(exact, "timed complete solve changed outputs");
        }
    }
}
