use super::*;
use crate::soil_newton_publish_gpu::{ExperimentalSoilNewtonPublishGpu, SoilNewtonPublishInputs};

const SOURCE: &str = include_str!("../shaders/soil_admm_probe.wgsl");

#[path = "soil_admm_resident_probe.rs"]
mod resident;

#[path = "soil_admm_prepared_cases.rs"]
mod prepared_cases;

#[path = "soil_admm_geometry_probe.rs"]
mod geometry_cases;

fn order(pair: usize) -> [usize; 2] {
    if pair % 2 == 0 { [0, 1] } else { [1, 0] }
}

fn median(values: impl Iterator<Item = f64>) -> f64 {
    let mut values: Vec<_> = values.collect();
    assert!(!values.is_empty() && values.iter().all(|v| v.is_finite()));
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

#[test]
fn admm_gpu_active_order_and_timing_contract() {
    assert_eq!(
        (0..4).flat_map(order).collect::<Vec<_>>(),
        [0, 1, 1, 0, 0, 1, 1, 0]
    );
    assert_eq!(median([3., 1., 2., 4.].into_iter()), 3.);
    for count in [0, 1, 63, 64, 65, 251, 256, 257, 4096, 16384] {
        for stride in [1, 2, 17, 63] {
            let ids: Vec<_> = (0..count).filter(|i| i % stride == 0).collect();
            assert!(ids.windows(2).all(|v| v[0] < v[1]));
            let mut owners = vec![0usize; count];
            for lane in 0..64 {
                for slot in (lane..ids.len()).step_by(64) {
                    owners[ids[slot]] += 1;
                }
            }
            for (id, owners) in owners.into_iter().enumerate() {
                assert_eq!(owners, usize::from(id % stride == 0));
            }
            let scan = (0..count)
                .filter(|i| i % stride == 0)
                .fold(0f32, |a, i| a + (i as f32 * 0.17).sin());
            let compact = ids.iter().fold(0f32, |a, &i| a + (i as f32 * 0.17).sin());
            assert_eq!(scan.to_bits(), compact.to_bits());
        }
    }
    assert!(SOURCE.contains("active_list.ids[active_count]=c;active_count++;"));
    assert!(!SOURCE.contains("atomicAdd"));
}

fn staging_order(block: usize) {
    for width in [1usize, 3, 4, 5, 10, 32] {
        for count in [0usize, 1, 63, 64, 65, 251, 256, 257, 4096, 16384] {
            let point = |i: usize| [(i as f32 * 0.17).sin(), (i as f32 * 0.31).cos(), 0.003];
            let response = |i: usize, j: usize| {
                [
                    (i as f32 * 0.07 + j as f32).cos(),
                    0.01,
                    (j as f32 * 0.13).sin(),
                ]
            };
            let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            for local in [false, true] {
                let mut original = vec![0f32; width];
                let mut staged = vec![0f32; width];
                for j in 0..width {
                    original[j] = if local { 0. } else { j as f32 * 0.1 };
                    for i in 0..count {
                        let term = dot(response(i, j), point(i));
                        if local {
                            original[j] += term;
                        } else {
                            original[j] -= term;
                        }
                    }
                }
                for base in (0..width).step_by(block) {
                    for lane in 0..block {
                        if base + lane < width {
                            staged[base + lane] = if local {
                                0.
                            } else {
                                (base + lane) as f32 * 0.1
                            };
                        }
                    }
                    for first in (0..count).step_by(64) {
                        let mut points = [[0f32; 3]; 64];
                        let mut responses = [[[0f32; 3]; 64]; 32];
                        for lane in 0..64 {
                            if first + lane < count {
                                points[lane] = point(first + lane);
                                for axis in 0..block {
                                    if base + axis < width {
                                        responses[axis][lane] = response(first + lane, base + axis);
                                    }
                                }
                            }
                        }
                        for lane in 0..block {
                            if base + lane < width {
                                for offset in 0..64.min(count - first) {
                                    let term = dot(responses[lane][offset], points[offset]);
                                    if local {
                                        staged[base + lane] += term;
                                    } else {
                                        staged[base + lane] -= term;
                                    }
                                }
                            }
                        }
                    }
                }
                assert_eq!(
                    original.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    staged.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
            }
        }
    }
    assert_eq!(SOURCE.matches("staged_rates(lane,selected,").count(), 2);
    assert!(SOURCE.contains("staged_responses:array<vec4<f32>,2048>"));
    assert!(SOURCE.contains("staged_points:array<vec4<f32>,64>"));
}

#[test]
fn admm_gpu_staged_operand_order_contract() {
    staging_order(4);
}

#[test]
fn admm_gpu_full_width_order_contract() {
    staging_order(32);
    assert_eq!(
        SOURCE.matches("staged_rates_wide(lane,selected,").count(),
        2
    );
}

#[test]
fn admm_gpu_fixed_contact_cache_contract() {
    let module = naga::front::wgsl::parse_str(SOURCE).unwrap();
    let fixed = module
        .types
        .iter()
        .find(|(_, t)| t.name.as_deref() == Some("FixedContact"))
        .unwrap()
        .1;
    let naga::TypeInner::Struct { members, span } = &fixed.inner else {
        panic!("fixed contact layout")
    };
    assert_eq!(*span, 64);
    assert_eq!(
        members.iter().map(|m| m.offset).collect::<Vec<_>>(),
        [0, 16, 32, 48]
    );
    assert!(
        SOURCE.contains("scratch[c]=Iterate();states[c]=State();fixed_contacts[c]=FixedContact();")
    );
    let initialize = SOURCE
        .find("fixed_contacts[c]=FixedContact(vec4<f32>")
        .unwrap();
    let factor = SOURCE.find("for(var index=lane;index<n*n;").unwrap();
    assert!(initialize < factor && SOURCE[initialize..factor].contains("storageBarrier();"));
    assert!(SOURCE.contains("if params.pad==4u {return fixed_contacts[c].free.xyz;}"));
    assert!(SOURCE.contains("return uncached_free(c,normal);"));
    assert_eq!(
        SOURCE
            .matches("if params.pad>=3u {staged_rates_wide")
            .count(),
        2
    );
}

#[test]
fn admm_gpu_shader_contract() {
    let module = naga::front::wgsl::parse_str(SOURCE).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    assert_eq!(module.entry_points.len(), 1);
    assert_eq!(module.entry_points[0].workgroup_size, [64, 1, 1]);
    assert!(SOURCE.contains("report.factors=1u"));
    for field in ["pad0", "pad1", "pad2", "pad3", "pad4"] {
        assert!(SOURCE.contains(&format!("report.{field}=0u;")));
    }
    assert!(SOURCE.contains("iteration<params.iterations"));
    if let Some(path) = std::env::var_os("MOLLA_ADMM_SHADER_ARTIFACT") {
        std::fs::write(path, SOURCE).unwrap();
    }
}

fn delta(a: impl Iterator<Item = f64>, b: impl Iterator<Item = f64>) -> Option<f64> {
    let mut largest = 0f64;
    for (a, b) in a.zip(b) {
        if !a.is_finite() || !b.is_finite() {
            return None;
        }
        largest = largest.max((a - b).abs());
    }
    Some(largest)
}

fn public_delta(a: &[Vec<u8>; 5], b: &[Vec<u8>; 5], p: &Reference) -> Option<f64> {
    let mut result = delta(
        mixed_tests::states(&a[0], p).into_iter(),
        mixed_tests::states(&b[0], p).into_iter(),
    )?;
    let word = |b: &[u8]| f64::from(f32::from_le_bytes(b.try_into().unwrap()));
    assert_eq!(a[1].len(), b[1].len());
    assert_eq!(a[2].len(), b[2].len());
    result = result.max(delta(
        a[1].chunks_exact(8).map(|v| word(&v[..4]) + word(&v[4..])),
        b[1].chunks_exact(8).map(|v| word(&v[..4]) + word(&v[4..])),
    )?);
    for (a, b) in a[2].chunks_exact(80).zip(b[2].chunks_exact(80)) {
        if a[64..] != b[64..] {
            return None;
        }
        result = result.max(delta(
            a[..64].chunks_exact(4).map(word),
            b[..64].chunks_exact(4).map(word),
        )?);
    }
    Some(result)
}

fn fixed_cache_error(bytes: &[u8], f: &Fixture, free: &[[f32; 2]]) -> f64 {
    assert_eq!(bytes.len(), 64 * f.contacts.len().max(1));
    let mut error = 0f64;
    for (c, words) in bytes.chunks_exact(64).enumerate() {
        let mut expected = [0f64; 16];
        if c < f.contacts.len() && f.contacts[c].soil_mass[3] > 0. {
            let node = &f.contacts[c];
            let m = f64::from(node.soil_mass[3]);
            let normal = node.normal_gap[..3]
                .iter()
                .map(|&v| f64::from(v).powi(2))
                .sum::<f64>()
                .sqrt();
            for axis in 0..3 {
                expected[axis] = f64::from(node.normal_gap[axis]) / normal;
                expected[4 + axis] = f64::from(node.soil_mass[axis])
                    + expected[axis] * f64::from(node.normal_gap[3]);
                for (j, rate) in free.iter().enumerate() {
                    expected[4 + axis] -= f64::from(f.rows[c * free.len() + j][axis])
                        * (f64::from(rate[0]) + f64::from(rate[1]));
                }
                let wall = (node.boundary[0] >> (2 * axis)) & 3 != 0;
                expected[8 + axis] = if wall { 2. * m / 3. } else { m / 2. };
                expected[12 + axis] = if wall { -m / 3. } else { 0. };
            }
            expected[3] = 1. / m;
        } else {
            assert!(
                words.iter().all(|&v| v == 0),
                "inactive fixed cache slot {c}"
            );
        }
        for (word, expected) in words.chunks_exact(4).zip(expected) {
            let actual = f64::from(f32::from_le_bytes(word.try_into().unwrap()));
            assert!(actual.is_finite());
            error = error.max((actual - expected).abs() / expected.abs().max(1.));
        }
    }
    error
}

fn publication_snapshot(
    engine: &Engine,
    publication: &ExperimentalSoilNewtonPublishGpu,
    reset: bool,
) -> [Vec<u8>; 3] {
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    let mut encoder = engine
        .device()
        .unwrap()
        .create_command_encoder(&Default::default());
    if reset {
        publication.encode_reset(&mut encoder);
    }
    publication.encode(&mut encoder);
    engine.queue().unwrap().submit([encoder.finish()]);
    assert_eq!(allocations.finish().observations, 0);
    let traffic = traffic.finish();
    assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    assert_eq!(dispatch.finish().encoded_dispatches, 1);
    [
        publication.rates_buffer(),
        publication.outputs_buffer(),
        publication.status_buffer(),
    ]
    .map(|b| capture::read(engine, b))
}

fn publication_rejection(root: &std::path::Path, label: &str, buffers: &[Vec<u8>; 3]) -> [u32; 8] {
    for (name, bytes) in ["rates", "outputs", "publication"].iter().zip(buffers) {
        std::fs::write(root.join(format!("{label}-{name}.bin")), bytes).unwrap();
    }
    let status: [u32; 8] = bytemuck::pod_read_unaligned(&buffers[2]);
    assert_ne!(status[0] & (8 | 16), 0, "numerical rejection: {label}");
    assert!(
        buffers[..2].iter().flatten().all(|&v| v == 0),
        "fail closed: {label}"
    );
    status
}

fn numerical_archive_check(
    engine: &Engine,
    publication: &ExperimentalSoilNewtonPublishGpu,
    states: &wgpu::Buffer,
    report: &wgpu::Buffer,
    old: &[Vec<u8>; 5],
    saved: &[Vec<u8>; 5],
    root: &std::path::Path,
) -> serde_json::Value {
    let queue = engine.queue().unwrap();
    queue.write_buffer(states, 0, &old[0]);
    queue.write_buffer(report, 0, &old[3]);
    publication.set_numerical_probe(engine, false);
    let accepted = publication_snapshot(engine, publication, true);
    for (name, bytes) in ["rates", "outputs", "publication"].iter().zip(&accepted) {
        std::fs::write(root.join(format!("old-physical-only-{name}.bin")), bytes).unwrap();
    }
    assert_eq!(&accepted[0], &old[1], "old physical-only rates");
    assert_eq!(&accepted[1], &old[2], "old physical-only outputs");
    let old_status: [u32; 8] = bytemuck::pod_read_unaligned(&accepted[2]);
    let archived_status: [u32; 8] = bytemuck::pod_read_unaligned(&old[4]);
    let status_exact = old_status == archived_status;
    assert_eq!(
        &old_status[..4],
        &archived_status[..4],
        "old physical-only fault fields"
    );
    assert_eq!(
        &old_status[6..],
        &archived_status[6..],
        "old physical-only reserved fields"
    );
    assert!(
        old_status[4..6]
            .iter()
            .all(|&v| f32::from_bits(v).is_finite())
    );
    assert_eq!(old_status[0], 0);
    assert_eq!(
        publication_snapshot(engine, publication, true),
        accepted,
        "old same-pipeline self repeat"
    );
    publication.set_numerical_probe(engine, true);
    let rejected = publication_snapshot(engine, publication, true);
    let status = publication_rejection(root, "old-numerical-rejected", &rejected);
    queue.write_buffer(states, 0, &saved[0]);
    queue.write_buffer(report, 0, &saved[3]);
    let latched = publication_snapshot(engine, publication, false);
    let latched_status = publication_rejection(root, "numerical-fault-latched", &latched);
    let recovered = publication_snapshot(engine, publication, true);
    assert_eq!(&recovered[0], &saved[1], "numerical recovery rates");
    assert_eq!(&recovered[1], &saved[2], "numerical recovery outputs");
    assert_eq!(&recovered[2], &saved[4], "numerical recovery status");
    serde_json::json!({"old_physical_only_exact":status_exact,"old_rates_outputs_exact":true,"old_fault_fields_exact":true,"old_same_pipeline_repeat_exact":true,"old_status":old_status,"archived_status":archived_status,"old_status_scale_ulp_distance":old_status[5].abs_diff(archived_status[5]),"guard_status":status,"latched_status":latched_status,"zero_on_reject":true,"reset_recovery_exact":true})
}

fn publication_index_faults(
    engine: &Engine,
    publication: &ExperimentalSoilNewtonPublishGpu,
    active: &wgpu::Buffer,
    valid: &[u32],
    slots: usize,
    saved: &[Vec<u8>; 5],
    root: &std::path::Path,
) -> Vec<serde_json::Value> {
    let count = valid[0] as usize;
    assert!(count > 1 && count < slots);
    let inactive = (0..slots as u32)
        .find(|i| valid[1..=count].binary_search(i).is_err())
        .unwrap();
    let publish = || publication_snapshot(engine, publication, true);
    publication.set_active_probe(engine, true);
    publication.set_numerical_probe(engine, true);
    let mut cases = Vec::new();
    for label in [
        "extra-count",
        "omission",
        "duplicate",
        "out-of-range",
        "inactive",
        "reversed",
        "overflow-count",
    ] {
        let mut bad = valid.to_vec();
        match label {
            "extra-count" => bad[0] += 1,
            "omission" => bad[0] -= 1,
            "duplicate" => bad[2] = bad[1],
            "out-of-range" => bad[1] = slots as u32,
            "inactive" => bad[1] = inactive,
            "reversed" => bad[1..=count].reverse(),
            "overflow-count" => bad[0] = u32::MAX,
            _ => unreachable!(),
        }
        engine
            .queue()
            .unwrap()
            .write_buffer(active, 0, bytemuck::cast_slice(&bad));
        let failed = publish();
        let status: [u32; 8] = bytemuck::pod_read_unaligned(&failed[2]);
        let zero = failed[..2].iter().flatten().all(|&v| v == 0);
        std::fs::write(root.join(format!("fault-{label}.bin")), &failed[2]).unwrap();
        assert_ne!(status[0] & 1, 0, "input rejection: {label}");
        assert!(zero, "fail-closed rates and outputs: {label}");
        engine
            .queue()
            .unwrap()
            .write_buffer(active, 0, bytemuck::cast_slice(valid));
        let recovered = publish();
        assert_eq!(&recovered[0], &saved[1], "recovered rates: {label}");
        assert_eq!(&recovered[1], &saved[2], "recovered outputs: {label}");
        assert_eq!(&recovered[2], &saved[4], "recovered status: {label}");
        cases.push(serde_json::json!({"case":label,"status":status,"zero_outputs_rates":zero,"recovery_exact":true}));
    }
    cases
}

pub(super) fn replay(
    engine: &Engine,
    f: &Fixture,
    p: &Reference,
    free_pairs: &[[f32; 2]],
    prepared: &[[f32; 4]],
    hash: &str,
    output: &std::path::Path,
) {
    let n = p.free.len();
    let count = f.contacts.len();
    assert!((1..=32).contains(&n) && count <= 16384);
    assert_eq!(f.rows.len(), count * n);
    assert_eq!(prepared.len(), count * n);
    let root = output.join(hash);
    assert!(!root.exists());
    std::fs::create_dir_all(&root).unwrap();
    let device = engine.device().unwrap();
    assert!(
        device.limits().max_compute_workgroup_storage_size >= 38400,
        "full-width ADMM staging requires 38400 shared bytes; device exposes {}",
        device.limits().max_compute_workgroup_storage_size
    );
    let free = storage(engine, free_pairs);
    let rows = storage(engine, &f.rows);
    let response = storage(engine, prepared);
    let contacts = storage(engine, &f.contacts);
    let mass = storage(engine, &f.mass);
    let upstream = storage(engine, &[0u32]);
    let scratch = storage(engine, &vec![0u8; 96 * count.max(1)]);
    let states = storage(engine, &vec![0u8; 64 * count.max(1)]);
    let report = storage(engine, &[0u32; 8]);
    let active = storage(engine, &vec![u32::MAX; count + 1]);
    let fixed = storage(engine, &vec![0u8; 64 * count.max(1)]);
    let params = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("ADMM probe parameters"),
        size: 16,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    engine.queue().unwrap().write_buffer(
        &params,
        0,
        bytemuck::cast_slice(&[n as u32, count as u32, 64u32, 0u32]),
    );
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("ADMM probe layout"),
        entries: &(0..10)
            .map(|binding| wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: if binding == 7 {
                        wgpu::BufferBindingType::Uniform
                    } else {
                        wgpu::BufferBindingType::Storage {
                            read_only: binding < 4,
                        }
                    },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            })
            .collect::<Vec<_>>(),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("mass-metric ADMM probe"),
        source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("admm"),
        compilation_options: Default::default(),
        cache: None,
    });
    let bindings = [
        &free, &rows, &response, &contacts, &scratch, &states, &report, &params, &active, &fixed,
    ];
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &bindings
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: b.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    let publication = ExperimentalSoilNewtonPublishGpu::new_active_probe(
        engine,
        SoilNewtonPublishInputs {
            mass: &mass,
            free_rates: &free,
            point_rows: &rows,
            point_response: &response,
            contacts: &contacts,
            state: &states,
            upstream_status: &upstream,
            solver_status: &report,
        },
        n as u32,
        count as u32,
        u64::MAX,
        &active,
    )
    .unwrap();
    assert!(
        device
            .features()
            .contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    );
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("ADMM solve and publication timing"),
        ty: wgpu::QueryType::Timestamp,
        count: 3,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 24,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let execute = |mode: usize| {
        engine
            .queue()
            .unwrap()
            .write_buffer(&params, 12, &4u32.to_le_bytes());
        publication.set_active_probe(engine, true);
        publication.set_numerical_probe(engine, mode != 0);
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        publication.encode_reset(&mut encoder);
        encoder.write_timestamp(&queries, 0);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        molla_core::gpu_metrics::record_dispatch();
        encoder.write_timestamp(&queries, 1);
        publication.encode(&mut encoder);
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
        let timing = [
            (ticks[1] - ticks[0]) as f64 * period,
            (ticks[2] - ticks[1]) as f64 * period,
            (ticks[2] - ticks[0]) as f64 * period,
        ];
        let snapshot = [
            &states,
            publication.rates_buffer(),
            publication.outputs_buffer(),
            &report,
            publication.status_buffer(),
        ]
        .map(|b| capture::read(engine, b));
        (timing, snapshot)
    };
    let control = execute(0).1;
    engine.queue().unwrap().write_buffer(
        &fixed,
        0,
        bytemuck::cast_slice(&vec![0x7fc00000u32; 16 * count.max(1)]),
    );
    let warm = [control, execute(1).1];
    let fixed_bytes = capture::read(engine, &fixed);
    std::fs::write(root.join("fixed-contacts.bin"), &fixed_bytes).unwrap();
    let cache_error = fixed_cache_error(&fixed_bytes, f, free_pairs);
    assert!(cache_error <= 2e-6, "fixed contact cache: {cache_error}");
    for (mode, buffers) in warm.iter().enumerate() {
        for (name, data) in ["states", "rates", "outputs", "status", "publication"]
            .iter()
            .zip(buffers)
        {
            std::fs::write(root.join(format!("{mode}-{name}.bin")), data).unwrap();
        }
    }
    let index_bytes = capture::read(engine, &active);
    let indices: Vec<u32> = index_bytes
        .chunks_exact(4)
        .map(bytemuck::pod_read_unaligned)
        .collect();
    assert_eq!(indices[0] as usize, p.contacts.len());
    assert_eq!(
        &indices[1..1 + p.contacts.len()],
        p.contacts
            .iter()
            .map(|c| c.index as u32)
            .collect::<Vec<_>>()
            .as_slice()
    );
    std::fs::write(root.join("active-indices.bin"), &index_bytes).unwrap();
    assert_eq!(
        warm[0], warm[1],
        "same-pipeline physical-only versus physical and numerical publication"
    );
    let previous = std::path::PathBuf::from(
        std::env::var_os("MOLLA_ADMM_FP32_REFERENCE").expect("frozen FP32 control"),
    );
    for (name, data) in ["states", "rates", "outputs", "status", "publication"]
        .iter()
        .zip(&warm[0])
    {
        assert_eq!(
            data,
            &std::fs::read(previous.join(hash).join(format!("{name}.bin"))).unwrap(),
            "original FP32 {name}"
        );
    }
    let mut samples: [Vec<[f64; 3]>; 2] = [Vec::new(), Vec::new()];
    let mut paired_change = Vec::new();
    for pair in 0..8 {
        let mut total = [0.; 2];
        for mode in order(pair) {
            let (timing, buffers) = execute(mode);
            assert_eq!(buffers, warm[mode], "exact five-buffer repeat mode={mode}");
            if mode == 1 {
                assert_eq!(
                    capture::read(engine, &fixed),
                    fixed_bytes,
                    "exact fixed-cache rebuild"
                );
            }
            total[mode] = timing[2];
            samples[mode].push(timing);
        }
        paired_change.push(100. * (total[1] / total[0] - 1.));
    }
    let saved = &warm[1];
    let index_fault_cases =
        publication_index_faults(engine, &publication, &active, &indices, count, saved, &root);
    for (name, data) in ["states", "rates", "outputs", "status", "publication"]
        .iter()
        .zip(saved)
    {
        std::fs::write(root.join(format!("{name}.bin")), data).unwrap();
    }
    let x = mixed_tests::states(&saved[0], p);
    let mut reference = p.clone();
    reference.use_prepared_response(prepared);
    let cpu = admm::trial(&reference).unwrap().0;
    let physical = p.physical_error(&x);
    let prepared_physical = reference.physical_error(&x);
    let numerical = reference
        .residual(&x)
        .into_iter()
        .map(|v| {
            if v.is_finite() {
                v.abs()
            } else {
                f64::INFINITY
            }
        })
        .fold(0., f64::max);
    let cpu_delta = delta(x.iter().copied(), cpu.iter().copied());
    let reports: [u32; 8] = bytemuck::pod_read_unaligned(&saved[3]);
    let published: [u32; 8] = bytemuck::pod_read_unaligned(&saved[4]);
    let controls = std::path::PathBuf::from(
        std::env::var_os("MOLLA_ADMM_GUARDED_REFERENCE").expect("frozen old and guarded buffers"),
    );
    let old: [Vec<u8>; 5] = ["states", "rates", "outputs", "status", "publication"]
        .map(|name| std::fs::read(controls.join(hash).join(format!("0-{name}.bin"))).unwrap());
    let guarded: [Vec<u8>; 5] = ["states", "rates", "outputs", "status", "publication"]
        .map(|name| std::fs::read(controls.join(hash).join(format!("1-{name}.bin"))).unwrap());
    let prepared_cases = prepared_cases::check(engine, &root);
    let geometry_cases = geometry_cases::check(engine, &root);
    let resident_api = resident::check(
        engine,
        crate::soil_admm_gpu::SoilAdmmInputs {
            mass: &mass,
            free_rates: &free,
            point_rows: &rows,
            point_response: &response,
            contacts: &contacts,
            upstream_status: &upstream,
        },
        n as u32,
        count as u32,
        saved,
        &old,
        &root,
    );
    let numerical_archive =
        numerical_archive_check(engine, &publication, &states, &report, &old, saved, &root);
    engine
        .queue()
        .unwrap()
        .write_buffer(&params, 8, &1u32.to_le_bytes());
    let incomplete = execute(1).1;
    let incomplete_report: [u32; 8] = bytemuck::pod_read_unaligned(&incomplete[3]);
    assert_eq!(&incomplete_report[..3], &[0, 1, 1]);
    let incomplete_status = publication_rejection(
        &root,
        "one-iteration-rejected",
        &[
            incomplete[1].clone(),
            incomplete[2].clone(),
            incomplete[4].clone(),
        ],
    );
    std::fs::write(root.join("one-iteration-states.bin"), &incomplete[0]).unwrap();
    let incomplete_residual = reference
        .residual(&mixed_tests::states(&incomplete[0], p))
        .into_iter()
        .map(|v| {
            if v.is_finite() {
                v.abs()
            } else {
                f64::INFINITY
            }
        })
        .fold(0., f64::max);
    assert!(incomplete_residual.is_finite() && incomplete_residual > 1e-6);
    engine
        .queue()
        .unwrap()
        .write_buffer(&params, 8, &64u32.to_le_bytes());
    engine.queue().unwrap().write_buffer(&states, 0, &saved[0]);
    engine.queue().unwrap().write_buffer(&report, 0, &saved[3]);
    let recovered = publication_snapshot(engine, &publication, true);
    assert_eq!(&recovered[0], &saved[1]);
    assert_eq!(&recovered[1], &saved[2]);
    assert_eq!(&recovered[2], &saved[4]);
    let old_delta = public_delta(saved, &old, p);
    let guarded_delta = public_delta(saved, &guarded, p);
    let candidate_valid = reports[0] == 0
        && published[0] == 0
        && physical <= 1e-6
        && prepared_physical <= 1e-6
        && numerical <= 1e-6
        && cpu_delta.is_some_and(|v| v <= 2e-5)
        && guarded_delta.is_some_and(|v| v <= 2e-5);
    let mut result = serde_json::json!({"acceptance":false,"scope":"cold GPU FP32 mass-metric ADMM probe; prepared captured inputs; no production integration or full-frame timing","packet":hash,"active_contacts":p.contacts.len(),"slots":count,"dofs":n,"iterations":64,"factorizations":1,"reports":reports,"publication":published,"exact_repeats_per_mode":9,"exact_control":true,"exact_modes":true,"active_indices_exact":true,"mode_order":["physical-only publication","physical and numerical publication"],"index_fault_cases":index_fault_cases,"index_recovery_exact":true,"solver_modes":[4,4],"active_publication":true,"staging_array_bytes":33792,"required_workgroup_limit":38400,"device_workgroup_limit":device.limits().max_compute_workgroup_storage_size,"timing_columns":["solve_ms","publication_ms","total_ms"],"timings_ms":samples,"upper_medians_ms":[[median(samples[0].iter().map(|s|s[0])),median(samples[0].iter().map(|s|s[1])),median(samples[0].iter().map(|s|s[2]))],[median(samples[1].iter().map(|s|s[0])),median(samples[1].iter().map(|s|s[1])),median(samples[1].iter().map(|s|s[2]))]],"paired_total_change_percent":paired_change,"paired_upper_median_percent":median(paired_change.iter().copied()),"timing_scope":"same compiled pipeline; includes input validation, state clear, active list construction, full-width ordered operand staging, factorization, fixed-contact setup, 64 iterations, validated active-index publication and optional numerical guard; excludes preparation/uploads/mode selection/publication-status reset/query resolution/readbacks and full frame","physical":physical,"prepared_physical":prepared_physical,"projected_residual":numerical,"physical_bound":1e-6,"numerical_bound":1e-6,"public_bound":2e-5,"cpu_delta":cpu_delta,"old_public_delta":old_delta,"old_public_passed":old_delta.is_some_and(|v|v<=2e-5),"guarded_public_delta":guarded_delta,"candidate_valid":candidate_valid});
    result.as_object_mut().unwrap().extend(serde_json::json!({"fixed_cache_bytes":64*count.max(1),"fixed_cache_error":cache_error,"fixed_cache_bound":2e-6,"fixed_cache_exact_rebuilds":9,"fixed_cache_poison_recovery":true}).as_object().unwrap().clone());
    result.as_object_mut().unwrap().extend(serde_json::json!({"numerical_archive":numerical_archive,"one_iteration_report":incomplete_report,"one_iteration_status":incomplete_status,"one_iteration_residual":incomplete_residual,"one_iteration_recovery_exact":true}).as_object().unwrap().clone());
    result
        .as_object_mut()
        .unwrap()
        .insert("resident_api".into(), resident_api);
    result
        .as_object_mut()
        .unwrap()
        .insert("prepared_cases".into(), prepared_cases);
    result
        .as_object_mut()
        .unwrap()
        .insert("geometry_cases".into(), geometry_cases);
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    println!("ADMM_GPU {result}");
}
