use super::*;

const NEWTON_STAGES: [MpmProfileStage; 9] = [
    MpmProfileStage::ParticleOrdering,
    MpmProfileStage::GridReset,
    MpmProfileStage::ActiveTiles,
    MpmProfileStage::RigidReset,
    MpmProfileStage::ParticleToGrid,
    MpmProfileStage::GridUpdate,
    MpmProfileStage::ContactProjection,
    MpmProfileStage::GuardedGatherProjection,
    MpmProfileStage::GuardedParticleCommit,
];
const CONTACT_STAGES: [MpmProfileStage; 3] = [
    MpmProfileStage::ContactPreparation,
    MpmProfileStage::ContactSolve,
    MpmProfileStage::ContactPublication,
];

#[test]
fn profile_query_budget_covers_ordered_newton_and_existing_capacities() {
    for details in [false, true] {
        let max = if details { 102 } else { 204 };
        let stages = NEWTON_STAGES.len()
            + CONTACT_STAGES.len()
            + if details {
                MpmProfileStage::PREPARATION_STAGES.len()
            } else {
                0
            };
        for n in 1..=max {
            assert_eq!(profile_query_count(n, details).unwrap(), 2 + n * stages * 2);
        }
        for n in [0, max + 1, usize::MAX] {
            assert!(profile_query_count(n, details).is_err());
        }
    }
    assert_eq!(profile_query_count(3, false).unwrap(), 74);
    assert_eq!(profile_query_count(204, false).unwrap(), 4898);
    assert_eq!(
        (u64::from(wgpu::QUERY_SET_MAX_QUERIES) * 8) % wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,
        0
    );
    assert_eq!(wgpu::QUERY_SET_MAX_QUERIES % 2, 0);
}

fn stamp(profiler: &StepProfiler, encoder: &mut wgpu::CommandEncoder, index: u32, pass: bool) {
    let (queries, local) = profiler.query(index);
    assert!(
        index
            < profile_query_count(profiler.capacity, profiler.preparation_details).unwrap() as u32
    );
    if pass {
        encoder
            .begin_compute_pass(&Default::default())
            .write_timestamp(queries, local);
    } else {
        encoder.write_timestamp(queries, local);
    }
}

fn begin(
    profiler: &mut StepProfiler,
    encoder: &mut wgpu::CommandEncoder,
    stage: MpmProfileStage,
    pass: bool,
) -> u32 {
    let index = profiler.start(
        stage,
        (stage != MpmProfileStage::Initialization).then_some(0),
    );
    stamp(profiler, encoder, index, pass);
    index + 1
}

#[test]
#[cfg(feature = "resource-counters")]
#[ignore = "requires GPU timestamp queries inside encoders and passes"]
fn profile_query_pages_cover_nested_windows_and_reuse() {
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (n, details) in [
        (1, false),
        (3, false),
        (34, false),
        (170, false),
        (171, false),
        (186, false),
        (204, false),
        (1, true),
        (3, true),
        (34, true),
        (102, true),
    ] {
        let mut profiler = StepProfiler::new_with_preparation(device, n, details).unwrap();
        let expected_queries = profile_query_count(n, details).unwrap();
        assert_eq!(
            profiler.queries.len(),
            expected_queries.div_ceil(wgpu::QUERY_SET_MAX_QUERIES as usize)
        );
        for repeat in 0..2 {
            profiler
                .begin(
                    n,
                    0.0005,
                    1,
                    true,
                    true,
                    true,
                    true,
                    true,
                    false,
                    queue.get_timestamp_period(),
                )
                .unwrap();
            profiler.window = Some(ProfileWindow {
                remaining: n,
                offset: 0,
                dt_bits: 0.0005_f32.to_bits(),
                epoch: Arc::new(()),
            });
            let resources = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
            for step in 0..n {
                let mut encoder = device.create_command_encoder(&Default::default());
                if step == 0 {
                    let end = begin(
                        &mut profiler,
                        &mut encoder,
                        MpmProfileStage::Initialization,
                        false,
                    );
                    stamp(&profiler, &mut encoder, end, false);
                }
                for stage in NEWTON_STAGES {
                    let pass = stage != MpmProfileStage::ContactProjection;
                    let end = begin(&mut profiler, &mut encoder, stage, pass);
                    if stage == MpmProfileStage::ContactProjection {
                        for contact in CONTACT_STAGES {
                            let nested = begin(&mut profiler, &mut encoder, contact, false);
                            if details && contact == MpmProfileStage::ContactPreparation {
                                for preparation in MpmProfileStage::PREPARATION_STAGES {
                                    let leaf =
                                        begin(&mut profiler, &mut encoder, preparation, false);
                                    stamp(&profiler, &mut encoder, leaf, false);
                                }
                            }
                            stamp(&profiler, &mut encoder, nested, false);
                        }
                    }
                    stamp(&profiler, &mut encoder, end, pass);
                }
                profiler.finish_batch(&mut encoder, 1);
                assert_eq!(profiler.pending, step + 1 == n);
                queue.submit([encoder.finish()]);
            }
            let resources = resources.finish();
            let transfers = transfers.finish();
            assert!(!resources.invalid && !transfers.overflowed);
            assert_eq!(resources.observations, 0);
            assert_eq!(transfers.readback_bytes, 0);
            assert_eq!(profiler.entries.len() * 2, expected_queries);
            assert!(
                profiler
                    .begin(
                        1,
                        0.0005,
                        1,
                        true,
                        true,
                        true,
                        true,
                        true,
                        false,
                        queue.get_timestamp_period()
                    )
                    .is_err()
            );
            let capture = profiler.read(device).unwrap().unwrap();
            assert_eq!(capture.requested_substeps, n);
            assert_eq!(capture.readback_bytes, expected_queries as u64 * 8);
            assert_eq!(capture.stages.len(), 1 + n * NEWTON_STAGES.len());
            assert_eq!(capture.contact_stages.len(), n * CONTACT_STAGES.len());
            assert_eq!(
                capture.preparation_stages.len(),
                if details { n * 8 } else { 0 }
            );
            assert_eq!(capture.stages[0].substep, None);
            for step in 0..n {
                let stages: Vec<_> = capture
                    .stages
                    .iter()
                    .filter(|s| s.substep == Some(step))
                    .map(|s| s.stage)
                    .collect();
                assert_eq!(stages, NEWTON_STAGES);
            }
            assert!(profiler.read(device).unwrap().is_none());
            engine.check_device().unwrap();
            println!(
                "query pages: steps={n} details={details} repeat={repeat} queries={expected_queries} nested scopes and retained reuse pass"
            );
        }
    }
}
