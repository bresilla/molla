use super::*;

fn read(engine: &Engine, buffer: &wgpu::Buffer) -> Vec<u32> {
    let (tx, rx) = std::sync::mpsc::channel();
    wgpu::util::DownloadBuffer::read_buffer(
        engine.device().unwrap(),
        engine.queue().unwrap(),
        &buffer.slice(..),
        move |r| {
            let _ = tx.send(r);
        },
    );
    engine
        .device()
        .unwrap()
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .unwrap()
        .unwrap()
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn expected(grid: u32, capacity: u32, tiles: &[u32], sparse: bool) -> Vec<[u32; 2]> {
    let width = grid.div_ceil(4);
    let mut pairs = Vec::with_capacity(capacity as usize * 64);
    for slot in 0..capacity as usize {
        for local in 0..64u32 {
            let mut pair = [u32::MAX; 2];
            if let Some(&tile) = tiles.get(slot) {
                let x = (tile % width) * 4 + local % 4;
                let y = (tile / width % width) * 4 + local / 4 % 4;
                let z = (tile / (width * width)) * 4 + local / 16;
                if x < grid && y < grid && z < grid {
                    let world = x + grid * (y + grid * z);
                    pair = [
                        world,
                        if sparse {
                            slot as u32 * 64 + local
                        } else {
                            world
                        },
                    ];
                }
            }
            pairs.push(pair);
        }
    }
    pairs.sort_by_key(|p| p[0]);
    pairs
}

#[test]
fn candidate_index_sort_budget_contract() {
    for cells in [64, 1728, 65536] {
        assert_eq!(
            GridIndex::bytes_for(cells, true) - GridIndex::bytes_for(cells, false),
            1024
        );
        assert_eq!(
            GridIndex::bytes(cells),
            GridIndex::bytes_for(cells, cfg!(feature = "parallel-candidate-index"))
        );
    }
    assert_eq!(GridIndex::bytes_for(65536, true), 1_311_808);
}

#[test]
#[ignore = "requires GPU timestamps; stable parallel candidate indexing and paired cost"]
fn parallel_candidate_index_exact_and_cost() {
    let start = std::time::Instant::now();
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_CANDIDATE_INDEX_CAPTURE").unwrap());
    std::fs::create_dir(&root).unwrap();
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let buffer = |size| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("candidate index comparison"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    };
    let mut records = Vec::new();
    for (case, (grid, capacity, sparse)) in
        [(9u32, 27u32, false), (9, 27, true), (1536, 1024, true)]
            .into_iter()
            .enumerate()
    {
        let metadata = buffer(4 * u64::from(capacity + 8));
        let statuses = [buffer(4), buffer(4)];
        let index = SoilCandidateGridIndex {
            metadata: &metadata,
            metadata_word: 3,
            tile_ids_word: 8,
            tile_capacity: capacity,
            sparse,
        };
        let config = SoilCandidateConfig {
            origin: [0.; 3],
            cell_size: 0.1,
            grid_size: grid,
            bodies: 1,
            dofs: 1,
            capacity: 4096,
            dt: 0.001,
            search_distance: 0.05,
            max_owned_bytes: GridIndex::bytes_for(capacity * 64, true),
        };
        let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        assert!(
            GridIndex::build(
                &engine,
                SoilCandidateConfig {
                    max_owned_bytes: config.max_owned_bytes - 1,
                    ..config
                },
                index,
                &statuses[1],
                true
            )
            .is_err()
        );
        assert!(
            GridIndex::build(
                &engine,
                config,
                SoilCandidateGridIndex {
                    tile_ids_word: 9,
                    ..index
                },
                &statuses[1],
                true
            )
            .is_err()
        );
        assert_eq!(scope.finish().observations, 0);
        let owners = [
            GridIndex::build(&engine, config, index, &statuses[0], false).unwrap(),
            GridIndex::build(&engine, config, index, &statuses[1], true).unwrap(),
        ];
        assert_eq!(owners[0].backend(), "serial-stable-radix");
        assert_eq!(owners[1].backend(), "parallel-stable-radix");
        let mut tiles: Vec<_> = (0..capacity)
            .rev()
            .map(|i| {
                if grid == 9 {
                    i
                } else {
                    (i * 104729 + 99991) % grid.div_ceil(4).pow(3)
                }
            })
            .collect();
        for pass in 0..6 {
            if pass == 1 {
                tiles.rotate_left(7);
            }
            if pass == 2 {
                tiles[1] = tiles[0];
            }
            if pass == 3 {
                tiles.truncate(128);
            }
            if pass == 4 {
                tiles.truncate(3);
            }
            if pass == 5 {
                tiles.clear();
            }
            let mut words = vec![0u32; (capacity + 8) as usize];
            words[3] = tiles.len() as u32;
            words[8..8 + tiles.len()].copy_from_slice(&tiles);
            queue.write_buffer(&metadata, 0, bytemuck::cast_slice(&words));
            let reference = expected(grid, capacity, &tiles, sparse);
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            for (owner, status) in owners.iter().zip(&statuses) {
                encoder.clear_buffer(status, 0, None);
                owner.encode(&mut encoder);
            }
            queue.submit([encoder.finish()]);
            assert_eq!(allocations.finish().observations, 0);
            assert_eq!(transfers.finish(), Default::default());
            assert_eq!(dispatches.finish().encoded_dispatches, 30);
            for (arm, owner) in owners.iter().enumerate() {
                assert_eq!(read(&engine, &statuses[arm]), [0]);
                let keys = read(&engine, &owner.worlds);
                let values = read(&engine, &owner.physical);
                std::fs::write(
                    root.join(format!("case-{case}-pass-{pass}-arm-{arm}-keys.bin")),
                    bytemuck::cast_slice(&keys),
                )
                .unwrap();
                std::fs::write(
                    root.join(format!("case-{case}-pass-{pass}-arm-{arm}-values.bin")),
                    bytemuck::cast_slice(&values),
                )
                .unwrap();
                assert_eq!(
                    keys.iter()
                        .copied()
                        .zip(values.iter().copied())
                        .map(|(k, v)| [k, v])
                        .collect::<Vec<_>>(),
                    reference
                );
            }
            if matches!(pass, 0 | 3 | 4) && capacity == 1024 {
                let q = device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("candidate index cost"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 36,
                });
                let ticks = device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: 288,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let mut encoder = device.create_command_encoder(&Default::default());
                for i in 0..18u32 {
                    let arm = ((i / 2 + i % 2) % 2) as usize;
                    encoder.write_timestamp(&q, 2 * i);
                    owners[arm].encode(&mut encoder);
                    encoder.write_timestamp(&q, 2 * i + 1);
                }
                encoder.resolve_query_set(&q, 0..36, &ticks, 0);
                queue.submit([encoder.finish()]);
                let raw = read(&engine, &ticks);
                std::fs::write(
                    root.join(format!("pass-{pass}-ticks.bin")),
                    bytemuck::cast_slice(&raw),
                )
                .unwrap();
                let mut samples = [Vec::new(), Vec::new()];
                for (i, t) in raw.chunks_exact(4).enumerate() {
                    let a = u64::from(t[0]) | (u64::from(t[1]) << 32);
                    let b = u64::from(t[2]) | (u64::from(t[3]) << 32);
                    assert!(b > a);
                    samples[(i / 2 + i % 2) % 2]
                        .push((b - a) as f64 * f64::from(queue.get_timestamp_period()) * 1e-6);
                }
                let mut sorted = samples.clone();
                for s in &mut sorted {
                    s.sort_by(f64::total_cmp);
                }
                records.push(serde_json::json!({"cells":capacity*64,"active_tiles":tiles.len(),"samples_ms":samples,"reference_median_ms":sorted[0][4],"parallel_median_ms":sorted[1][4],"timestamp_period_ns":queue.get_timestamp_period()}));
                for (owner, status) in owners.iter().zip(&statuses) {
                    assert_eq!(read(&engine, status), [0]);
                    assert_eq!(
                        read(&engine, &owner.worlds),
                        reference.iter().map(|p| p[0]).collect::<Vec<_>>()
                    );
                    assert_eq!(
                        read(&engine, &owner.physical),
                        reference.iter().map(|p| p[1]).collect::<Vec<_>>()
                    );
                }
            }
        }
        for upstream in [false, true] {
            queue.write_buffer(&metadata, 12, bytemuck::bytes_of(&(capacity + 1)));
            let mut encoder = device.create_command_encoder(&Default::default());
            for (owner, status) in owners.iter().zip(&statuses) {
                encoder.clear_buffer(status, 0, None);
                owner.encode(&mut encoder);
            }
            queue.submit([encoder.finish()]);
            for (owner, status) in owners.iter().zip(&statuses) {
                assert_eq!(read(&engine, status), [1 << 31]);
                assert!(read(&engine, &owner.worlds).iter().all(|&v| v == u32::MAX));
            }
            queue.write_buffer(&metadata, 12, bytemuck::bytes_of(&0u32));
            let mut encoder = device.create_command_encoder(&Default::default());
            for (owner, status) in owners.iter().zip(&statuses) {
                if !upstream {
                    encoder.clear_buffer(status, 0, None);
                }
                owner.encode(&mut encoder);
            }
            queue.submit([encoder.finish()]);
            for status in &statuses {
                assert_eq!(read(&engine, status), [if upstream { 1 << 31 } else { 0 }]);
            }
        }
    }
    std::fs::write(root.join("result.json"),serde_json::to_vec_pretty(&serde_json::json!({"acceptance":false,"exact_cpu_stable_order":true,"cases":3,"metadata_passes_per_case":6,"fault_sticky_reset":true,"extra_retained_bytes":1024,"timings":records})).unwrap()).unwrap();
    eprintln!(
        "parallel candidate index: {records:?} elapsed={:?}",
        start.elapsed()
    );
}
