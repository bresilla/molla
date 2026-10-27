use super::*;
use crate::soil_admm_gpu::{ExperimentalSoilAdmmGpu, SoilAdmmCapacity, SoilAdmmInputs};

fn run(
    engine: &Engine,
    solver: &ExperimentalSoilAdmmGpu,
    reset: bool,
    publication_only: bool,
) -> [Vec<u8>; 5] {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if reset {
        solver.encode_reset(&mut encoder);
    }
    if publication_only {
        solver.encode_publication_probe(&mut encoder);
    } else {
        solver.encode(&mut encoder);
    }
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(
        dispatch.finish().encoded_dispatches,
        if publication_only { 1 } else { 2 }
    );
    let [states, report] = solver.diagnostic_buffers();
    [
        states,
        solver.rates_buffer(),
        solver.outputs_buffer(),
        report,
        solver.status_buffer(),
    ]
    .map(|b| capture::read(engine, b))
}

fn rejection(root: &std::path::Path, label: &str, buffers: &[Vec<u8>; 5], mask: u32) -> [u32; 8] {
    let status: [u32; 8] = bytemuck::pod_read_unaligned(&buffers[4]);
    for (name, bytes) in ["states", "rates", "outputs", "report", "publication"]
        .iter()
        .zip(buffers)
    {
        std::fs::write(root.join(format!("api-{label}-{name}.bin")), bytes).unwrap();
    }
    assert_ne!(status[0] & mask, 0, "resident rejection {label}");
    assert!(
        buffers[1..3].iter().flatten().all(|&v| v == 0),
        "resident fail closed {label}"
    );
    status
}

fn staging_comparison(
    engine: &Engine,
    input: SoilAdmmInputs<'_>,
    capacity: SoilAdmmCapacity,
    _compact: &ExperimentalSoilAdmmGpu,
    expected: &[Vec<u8>; 5],
    root: &std::path::Path,
    variant: u32,
) -> serde_json::Value {
    let global =
        ExperimentalSoilAdmmGpu::new_global_staging_control(engine, input, capacity).unwrap();
    let compact = &global;
    let onchip = variant == 1;
    let padded = variant == 2;
    let shared_iterate = variant == 3;
    let response_cache = variant == 4;
    let response_slots = if response_cache {
        ExperimentalSoilAdmmGpu::response_cache_slots(capacity.dofs).unwrap()
    } else {
        0
    };
    let control = if response_cache {
        ExperimentalSoilAdmmGpu::new_response_cache_probe(engine, input, capacity).unwrap()
    } else if shared_iterate {
        ExperimentalSoilAdmmGpu::new_shared_iterate_probe(engine, input, capacity).unwrap()
    } else if onchip {
        ExperimentalSoilAdmmGpu::new_onchip_probe(engine, input, capacity).unwrap()
    } else if padded {
        ExperimentalSoilAdmmGpu::new_padded_probe(engine, input, capacity).unwrap()
    } else {
        ExperimentalSoilAdmmGpu::new_full_staging_control(engine, input, capacity).unwrap()
    };
    assert_eq!(
        control.shares_program(compact),
        !onchip && !shared_iterate && !response_cache && capacity.dofs == 32
    );
    let device = engine.device().unwrap();
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("ADMM staging capacity comparison"),
        ty: wgpu::QueryType::Timestamp,
        count: 3,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 24,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let execute = |solver: &ExperimentalSoilAdmmGpu| {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        solver.encode_reset(&mut encoder);
        encoder.write_timestamp(&queries, 0);
        solver.encode_solve(&mut encoder);
        encoder.write_timestamp(&queries, 1);
        solver.encode_publication(&mut encoder);
        encoder.write_timestamp(&queries, 2);
        encoder.resolve_query_set(&queries, 0..3, &resolved, 0);
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let traffic = traffic.finish();
        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
        assert_eq!(dispatch.finish().encoded_dispatches, 2);
        let ticks: [u64; 3] = bytemuck::pod_read_unaligned(&capture::read(engine, &resolved));
        assert!(ticks[0] < ticks[1] && ticks[1] <= ticks[2]);
        let period = f64::from(engine.queue().unwrap().get_timestamp_period()) / 1e6;
        let [states, report] = solver.diagnostic_buffers();
        let snapshot = [
            states,
            solver.rates_buffer(),
            solver.outputs_buffer(),
            report,
            solver.status_buffer(),
        ]
        .map(|b| capture::read(engine, b));
        if variant != 0 {
            let mode = usize::from(!std::ptr::eq(solver, compact));
            for (i, bytes) in snapshot.iter().enumerate() {
                let tag = if response_cache {
                    "response-cache"
                } else if shared_iterate {
                    "shared-iterate"
                } else if padded {
                    "padding"
                } else {
                    "onchip"
                };
                std::fs::write(root.join(format!("{tag}-{mode}-{i}.bin")), bytes).unwrap();
            }
        }
        assert_eq!(&snapshot, expected, "staging capacity exact parity");
        [
            (ticks[1] - ticks[0]) as f64 * period,
            (ticks[2] - ticks[1]) as f64 * period,
            (ticks[2] - ticks[0]) as f64 * period,
        ]
    };
    let solvers = if variant != 0 {
        [compact, &control]
    } else {
        [&control, compact]
    };
    let cold = solvers.map(execute);
    let mut samples: [Vec<[f64; 3]>; 2] = Default::default();
    let mut changes = Vec::new();
    for pair in 0..8 {
        let mut total = [0.; 2];
        for mode in order(pair) {
            let timing = execute(solvers[mode]);
            total[mode] = timing[2];
            samples[mode].push(timing);
        }
        changes.push(100. * (total[1] / total[0] - 1.));
    }
    let result = serde_json::json!({
        "acceptance":false,
        "mode_order":if response_cache { ["tiled response staging", "resident response cache"] } else if shared_iterate { ["global iteration state", "workgroup iteration state"] } else if onchip { ["width-sized uncached", "width-sized on-chip cache"] } else if padded { ["width-sized uncached", "padded operand staging"] } else { ["full 32-DOF staging", "width-sized staging"] },
        "shared_iterate":shared_iterate,"response_cache":response_cache,"response_slots":response_slots,
        "onchip":onchip,"onchip_capacity":if onchip {ExperimentalSoilAdmmGpu::onchip_slots(capacity.dofs).unwrap()} else {0},
        "padded":padded,
        "staging_bytes":[if variant==0 {33792} else {1024+1024*capacity.dofs.max(4)}, if response_cache {1024+12*(capacity.dofs*(response_slots+1)).max(64*capacity.dofs.max(4))} else {1024+16*capacity.dofs.max(4)*if padded && capacity.dofs<32 {65} else {64}}],
        "dofs":capacity.dofs,"iterations":capacity.iterations,
        "exact_repeats_per_mode":9,"exact_control":true,
        "encode_allocations":0,"encode_upload_bytes":0,"encode_readback_bytes":0,"dispatches":2,
        "timing_columns":["solve_ms","publication_ms","total_ms"],
        "timing_scope":"solve and guarded publication; excludes setup, reset, query resolution and diagnostic reads; not frame timing",
        "cold_ms":cold,"timings_ms":samples,
        "upper_medians_ms":samples.each_ref().map(|s| [median(s.iter().map(|v| v[0])), median(s.iter().map(|v| v[1])), median(s.iter().map(|v| v[2]))]),
        "paired_total_change_percent":changes,
        "paired_upper_median_percent":median(changes.iter().copied())
    });
    std::fs::write(
        root.join(if response_cache {
            "response-cache-comparison.json"
        } else if shared_iterate {
            "shared-iterate-comparison.json"
        } else if padded {
            "padding-comparison.json"
        } else if onchip {
            "onchip-comparison.json"
        } else {
            "staging-comparison.json"
        }),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    result
}

fn onchip_boundaries(engine: &Engine, root: &std::path::Path, response_only: bool) {
    let mut records = Vec::new();
    let mut cases = vec![
        (1u32, 0u32),
        (10, 255),
        (10, 256),
        (10, 257),
        (10, 513),
        (32, 17),
    ];
    if response_only {
        cases.extend([
            (10, 297),
            (10, 298),
            (10, 299),
            (32, 83),
            (32, 84),
            (32, 85),
            (1, 511),
            (1, 512),
            (1, 513),
        ]);
    }
    for (case, (n, count)) in cases.into_iter().enumerate() {
        let mut mass = vec![0f32; (n * n) as usize];
        for i in 0..n {
            mass[(i * n + i) as usize] = 1.;
        }
        let free: Vec<_> = (0..n).flat_map(|i| [i as f32 * 0.03125, 1e-10]).collect();
        let mut contacts = vec![[0f32; 16]; count.max(1) as usize];
        for (i, c) in contacts.iter_mut().enumerate().take(count as usize) {
            c[..4].copy_from_slice(&[
                0.03,
                -0.25 - i as f32 * 0.0001,
                0.02,
                1. + (i % 7) as f32 * 0.01,
            ]);
            c[5] = 1.;
            c[8] = 0.3;
        }
        let mass = storage(engine, &mass);
        let free = storage(engine, &free);
        let mut row_values = vec![0f32; (4 * n * count.max(1)) as usize];
        if response_only {
            for c in 0..count {
                for axis in 0..n {
                    for component in 0..3 {
                        row_values[(4 * (c * n + axis) + component) as usize] =
                            ((c * 7 + axis * 3 + component) % 11) as f32 * 0.00001;
                    }
                }
            }
        }
        let rows = storage(engine, &row_values);
        let contacts = storage(engine, &contacts);
        let upstream = storage(engine, &[0u32]);
        let input = SoilAdmmInputs {
            mass: &mass,
            free_rates: &free,
            point_rows: &rows,
            point_response: &rows,
            contacts: &contacts,
            upstream_status: &upstream,
        };
        let cap = SoilAdmmCapacity {
            dofs: n,
            contacts: count,
            iterations: 64,
            max_owned_bytes: u64::MAX,
        };
        let control =
            ExperimentalSoilAdmmGpu::new_global_staging_control(engine, input, cap).unwrap();
        let response =
            ExperimentalSoilAdmmGpu::new_response_cache_probe(engine, input, cap).unwrap();
        assert_eq!(control.owned_buffer_bytes(), response.owned_buffer_bytes());
        let expected = run(engine, &control, true, false);
        let owners = if response_only {
            vec![control, response]
        } else {
            let candidate = ExperimentalSoilAdmmGpu::new_onchip_probe(engine, input, cap).unwrap();
            let padded = ExperimentalSoilAdmmGpu::new_padded_probe(engine, input, cap).unwrap();
            let shared =
                ExperimentalSoilAdmmGpu::new_shared_iterate_probe(engine, input, cap).unwrap();
            for owner in [&candidate, &padded, &shared] {
                assert_eq!(control.owned_buffer_bytes(), owner.owned_buffer_bytes());
            }
            vec![control, candidate, padded, shared, response]
        };
        for (mode, owner) in owners.iter().enumerate() {
            let actual = run(engine, owner, true, false);
            for (i, bytes) in actual.iter().enumerate() {
                std::fs::write(
                    root.join(format!("onchip-boundary-{case}-{mode}-{i}.bin")),
                    bytes,
                )
                .unwrap();
            }
            assert_eq!(actual, expected, "on-chip boundary {case} mode {mode}");
            assert_eq!(
                &actual[4][..4],
                &0u32.to_le_bytes(),
                "boundary publication {case}"
            );
            for buffer in owner.scratch_buffers() {
                engine.queue().unwrap().write_buffer(
                    buffer,
                    0,
                    bytemuck::cast_slice(&vec![0x7fc00000u32; buffer.size() as usize / 4]),
                );
            }
            assert_eq!(
                run(engine, owner, true, false),
                expected,
                "boundary poison recovery {case} mode {mode}"
            );
        }
        records.push(serde_json::json!({"dofs":n,"contacts":count,"cache_slots":ExperimentalSoilAdmmGpu::onchip_slots(n).unwrap(),"response_slots":ExperimentalSoilAdmmGpu::response_cache_slots(n).unwrap(),"nonzero_rows":response_only && count>0,"exact":true,"padded_exact":!response_only,"shared_iterate_exact":!response_only,"response_cache_exact":true,"poison_recovery":true,"publication_fault":0}));
    }
    std::fs::write(
        root.join(if response_only {
            "response-boundaries.json"
        } else {
            "onchip-boundaries.json"
        }),
        serde_json::to_vec_pretty(&records).unwrap(),
    )
    .unwrap();
}

pub(super) fn check(
    engine: &Engine,
    input: SoilAdmmInputs<'_>,
    dofs: u32,
    contacts: u32,
    saved: &[Vec<u8>; 5],
    old: &[Vec<u8>; 5],
    root: &std::path::Path,
) -> serde_json::Value {
    let mut capacity = SoilAdmmCapacity {
        dofs,
        contacts,
        iterations: 64,
        max_owned_bytes: u64::MAX,
    };
    let bytes = ExperimentalSoilAdmmGpu::required_bytes(capacity).unwrap();
    capacity.max_owned_bytes = bytes;
    let short = storage(engine, &[0u32]);
    let wrong = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8 * u64::from(dofs),
            usage: wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        ExperimentalSoilAdmmGpu::new(
            engine,
            input,
            SoilAdmmCapacity {
                max_owned_bytes: bytes - 1,
                ..capacity
            }
        )
        .is_err()
    );
    assert!(
        ExperimentalSoilAdmmGpu::new(
            engine,
            SoilAdmmInputs {
                free_rates: &short,
                ..input
            },
            capacity
        )
        .is_err()
    );
    assert!(
        ExperimentalSoilAdmmGpu::new(
            engine,
            SoilAdmmInputs {
                free_rates: &wrong,
                ..input
            },
            capacity
        )
        .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let solver = ExperimentalSoilAdmmGpu::new(engine, input, capacity).unwrap();
    assert_eq!(allocations.finish().observations, 10);
    assert_eq!(solver.owned_buffer_bytes(), bytes);
    let first = run(engine, &solver, true, false);
    for (name, data) in ["states", "rates", "outputs", "report", "publication"]
        .iter()
        .zip(&first)
    {
        std::fs::write(root.join(format!("api-valid-{name}.bin")), data).unwrap();
    }
    assert_eq!(&first, saved, "resident API versus captured probe");
    let staging = staging_comparison(engine, input, capacity, &solver, &first, root, 0);
    let response_only = std::env::var_os("MOLLA_ADMM_CACHE_FOCUS").is_some();
    if !response_only {
        staging_comparison(engine, input, capacity, &solver, &first, root, 1);
        staging_comparison(engine, input, capacity, &solver, &first, root, 2);
        staging_comparison(engine, input, capacity, &solver, &first, root, 3);
    }
    staging_comparison(engine, input, capacity, &solver, &first, root, 4);
    onchip_boundaries(engine, root, response_only);
    crate::soil_newton_publish_gpu::check_sorted_sum_gpu(engine, root, |b| {
        capture::read(engine, b)
    });
    let publication = publication_comparison(engine, input, capacity, &solver, &first, root);
    assert_eq!(
        run(engine, &solver, false, false),
        first,
        "resident repeat without reset"
    );
    for buffer in solver.scratch_buffers() {
        engine.queue().unwrap().write_buffer(
            buffer,
            0,
            bytemuck::cast_slice(&vec![0x7fc00000u32; buffer.size() as usize / 4]),
        );
    }
    assert_eq!(
        run(engine, &solver, true, false),
        first,
        "resident poisoned scratch recovery"
    );
    engine
        .queue()
        .unwrap()
        .write_buffer(input.upstream_status, 0, &1u32.to_le_bytes());
    let upstream = rejection(root, "upstream", &run(engine, &solver, true, false), 1);
    engine
        .queue()
        .unwrap()
        .write_buffer(input.upstream_status, 0, &0u32.to_le_bytes());
    let latched = rejection(root, "latched", &run(engine, &solver, false, false), 1);
    assert_eq!(
        run(engine, &solver, true, false),
        first,
        "resident upstream reset recovery"
    );
    let original_contacts = capture::read(engine, input.contacts);
    engine
        .queue()
        .unwrap()
        .write_buffer(input.contacts, 12, &f32::NAN.to_le_bytes());
    let invalid_contact = rejection(
        root,
        "invalid-contact",
        &run(engine, &solver, true, false),
        1,
    );
    engine
        .queue()
        .unwrap()
        .write_buffer(input.contacts, 12, &original_contacts[12..16]);
    assert_eq!(
        run(engine, &solver, true, false),
        first,
        "resident contact recovery"
    );
    let [state, report] = solver.diagnostic_buffers();
    engine.queue().unwrap().write_buffer(state, 0, &old[0]);
    engine.queue().unwrap().write_buffer(report, 0, &old[3]);
    let numerical = rejection(
        root,
        "mandatory-numerical",
        &run(engine, &solver, true, true),
        8,
    );
    assert_eq!(
        run(engine, &solver, true, false),
        first,
        "resident numerical recovery"
    );
    let bounded = ExperimentalSoilAdmmGpu::new(
        engine,
        input,
        SoilAdmmCapacity {
            iterations: 1,
            ..capacity
        },
    )
    .unwrap();
    let limited = run(engine, &bounded, true, false);
    assert!(solver.shares_program(&bounded));
    let limited_report: [u32; 8] = bytemuck::pod_read_unaligned(&limited[3]);
    assert_eq!(&limited_report[..3], &[0, 1, 1]);
    let limited_status = rejection(root, "one-iteration", &limited, 8);
    assert_eq!(
        run(engine, &solver, false, false),
        first,
        "independent instance isolation"
    );
    let empty_mass = storage(engine, &[2f32]);
    let empty_free = storage(engine, &[0.125f32, 1e-9]);
    let empty_rows = storage(engine, &[0f32; 4]);
    let empty_contacts = storage(engine, &[0u32; 16]);
    let empty_upstream = storage(engine, &[0u32]);
    let empty_state = storage(engine, &[0u32; 16]);
    let empty_report = storage(engine, &[0u32; 8]);
    let empty_input = SoilAdmmInputs {
        mass: &empty_mass,
        free_rates: &empty_free,
        point_rows: &empty_rows,
        point_response: &empty_rows,
        contacts: &empty_contacts,
        upstream_status: &empty_upstream,
    };
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        ExperimentalSoilAdmmGpu::new(
            engine,
            empty_input,
            SoilAdmmCapacity {
                dofs: 1,
                contacts: 0,
                iterations: 64,
                max_owned_bytes: 415
            }
        )
        .is_err()
    );
    assert!(
        ExperimentalSoilNewtonPublishGpu::new_guarded_active(
            engine,
            SoilNewtonPublishInputs {
                mass: &empty_mass,
                free_rates: &empty_free,
                point_rows: &empty_rows,
                point_response: &empty_rows,
                contacts: &empty_contacts,
                state: &empty_state,
                upstream_status: &empty_upstream,
                solver_status: &empty_report
            },
            1,
            0,
            u64::MAX,
            &empty_upstream
        )
        .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    let empty = ExperimentalSoilAdmmGpu::new(
        engine,
        SoilAdmmInputs {
            mass: &empty_mass,
            free_rates: &empty_free,
            point_rows: &empty_rows,
            point_response: &empty_rows,
            contacts: &empty_contacts,
            upstream_status: &empty_upstream,
        },
        SoilAdmmCapacity {
            dofs: 1,
            contacts: 0,
            iterations: 64,
            max_owned_bytes: 416,
        },
    )
    .unwrap();
    let empty_result = run(engine, &empty, true, false);
    assert_eq!(solver.shares_program(&empty), dofs <= 4);
    assert_eq!(empty.owned_buffer_bytes(), 416);
    assert_eq!(empty_result[1], capture::read(engine, &empty_free));
    assert!(empty_result[2].iter().all(|&v| v == 0));
    let empty_status: [u32; 8] = bytemuck::pod_read_unaligned(&empty_result[4]);
    assert_eq!(empty_status[0], 0);
    assert_eq!(empty_status[2], 0);
    assert_eq!(run(engine, &empty, false, false), empty_result);
    serde_json::json!({"bytes":bytes,"allocations":10,"admission_rejections":3,"encode_allocations":0,"encode_upload_bytes":0,"encode_readback_bytes":0,"dispatches":2,"exact_probe_parity":true,"repeat_exact":true,"poison_recovery_exact":true,"upstream_status":upstream,"latched_status":latched,"invalid_contact_status":invalid_contact,"reset_recovery_exact":true,"mandatory_numerical_status":numerical,"one_iteration_report":limited_report,"one_iteration_status":limited_status,"instance_isolation_exact":true,"shared_program":true,"empty_shared_program":solver.shares_program(&empty),"staging_comparison":staging,"publication_comparison":publication,"empty_status":empty_status,"empty_bytes":416,"empty_admission_rejections":2,"empty_repeat_exact":true})
}

fn publication_comparison(
    engine: &Engine,
    input: SoilAdmmInputs<'_>,
    capacity: SoilAdmmCapacity,
    solver: &ExperimentalSoilAdmmGpu,
    expected: &[Vec<u8>; 5],
    root: &std::path::Path,
) -> serde_json::Value {
    let [state, solver_status] = solver.diagnostic_buffers();
    let control = ExperimentalSoilNewtonPublishGpu::new_active_probe(
        engine,
        SoilNewtonPublishInputs {
            mass: input.mass,
            free_rates: input.free_rates,
            point_rows: input.point_rows,
            point_response: input.point_response,
            contacts: input.contacts,
            state,
            upstream_status: input.upstream_status,
            solver_status,
        },
        capacity.dofs,
        capacity.contacts,
        u64::MAX,
        solver.active_indices_buffer(),
    )
    .unwrap();
    control.set_active_probe(engine, true);
    control.set_numerical_probe(engine, true);
    let device = engine.device().unwrap();
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("ADMM publication summation comparison"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let execute = |mode| {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        if mode == 0 {
            control.encode_reset(&mut encoder);
        } else {
            solver.encode_reset(&mut encoder);
        }
        encoder.write_timestamp(&queries, 0);
        if mode == 0 {
            control.encode(&mut encoder);
        } else {
            solver.encode_publication_probe(&mut encoder);
        }
        encoder.write_timestamp(&queries, 1);
        encoder.resolve_query_set(&queries, 0..2, &resolved, 0);
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let traffic = traffic.finish();
        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
        assert_eq!(dispatch.finish().encoded_dispatches, 1);
        let ticks: [u64; 2] = bytemuck::pod_read_unaligned(&capture::read(engine, &resolved));
        assert!(ticks[1] > ticks[0]);
        let buffers = if mode == 0 {
            [
                control.rates_buffer(),
                control.outputs_buffer(),
                control.status_buffer(),
            ]
        } else {
            [
                solver.rates_buffer(),
                solver.outputs_buffer(),
                solver.status_buffer(),
            ]
        };
        for ((name, buffer), original) in ["rates", "outputs", "status"]
            .iter()
            .zip(buffers)
            .zip([1, 2, 4])
        {
            let bytes = capture::read(engine, buffer);
            std::fs::write(root.join(format!("sum-{mode}-{name}.bin")), &bytes).unwrap();
            assert_eq!(
                bytes, expected[original],
                "publication summation exact parity: {name}"
            );
        }
        (ticks[1] - ticks[0]) as f64 * f64::from(engine.queue().unwrap().get_timestamp_period())
            / 1e6
    };
    let cold = [execute(0), execute(1)];
    let mut samples: [Vec<f64>; 2] = Default::default();
    let mut changes = Vec::new();
    for pair in 0..8 {
        let mut measured = [0.; 2];
        for mode in order(pair) {
            measured[mode] = execute(mode);
            samples[mode].push(measured[mode]);
        }
        changes.push(100. * (measured[1] / measured[0] - 1.));
    }
    let result = serde_json::json!({
        "acceptance":false,"mode_order":["TwoSum", "magnitude-sorted canonical-zero FastTwoSum"],
        "exact_repeats_per_mode":9,"exact_control":true,"roundtrips_per_sum":[6,3],
        "encode_allocations":0,"encode_upload_bytes":0,"encode_readback_bytes":0,"dispatches":1,
        "cold_ms":cold,"timings_ms":samples,
        "upper_medians_ms":samples.each_ref().map(|s| median(s.iter().copied())),
        "paired_change_percent":changes,"paired_upper_median_percent":median(changes.iter().copied()),
        "scope":"guarded active publication only; identical resident solved state; excludes reset, solve, preparation, query resolution and diagnostic reads"
    });
    std::fs::write(
        root.join("publication-comparison.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    result
}
