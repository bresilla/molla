use super::*;
use crate::soil_contact_geometry_gpu::*;
use crate::soil_floor_projection_gpu::*;

#[path = "soil_lattice_candidates_tests.rs"]
mod lattice;

#[test]
#[ignore = "requires real GPU"]
fn paired_soil_consumer_candidate_coverage_uses_low_rates() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut grid = vec![[0f32; 4]; 729];
    grid[index(5, 4, 4)][3] = 1.;
    let arrays = [
        buffer(&engine, bytemuck::cast_slice(&grid), 16),
        buffer(&engine, &[], 16),
        buffer(
            &engine,
            bytemuck::cast_slice(&[
                0f32, 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.1, 0.1, 0.1, 1.,
            ]),
            64,
        ),
        buffer(
            &engine,
            bytemuck::cast_slice(&[1f32, 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
            48,
        ),
        buffer(
            &engine,
            bytemuck::cast_slice(&[1024f32, 1e-5, -1024., 0.]),
            16,
        ),
        buffer(&engine, &[], 16),
        buffer(&engine, &[], 48),
        buffer(&engine, &[], 4),
        buffer(&engine, &[], 16),
    ];
    let inputs = |rates| SoilCandidateInputs {
        grid: &arrays[0],
        centers: &arrays[1],
        shapes: &arrays[2],
        jacobians: &arrays[3],
        corrected_rates: rates,
        corrected_velocities: &arrays[5],
    };
    let outputs = || SoilCandidateOutputs {
        queries: &arrays[6],
        node_ids: &arrays[7],
        status: &arrays[8],
    };
    let config = SoilCandidateConfig {
        dofs: 2,
        ..config(1, 0.001, 1)
    };
    let owner = ExperimentalSoilCandidatesGpu::with_rate_encoding(
        &engine,
        inputs(&arrays[4]),
        outputs(),
        config,
        RateEncoding::Paired,
    )
    .unwrap();
    let scalar_rates = buffer(&engine, bytemuck::cast_slice(&[1024f32, -1024.]), 8);
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        ExperimentalSoilCandidatesGpu::with_rate_encoding(
            &engine,
            inputs(&scalar_rates),
            outputs(),
            config,
            RateEncoding::Paired
        )
        .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    let scalar =
        ExperimentalSoilCandidatesGpu::new(&engine, inputs(&scalar_rates), outputs(), config)
            .unwrap();
    let run = |owner: &ExperimentalSoilCandidatesGpu| {
        engine
            .queue()
            .unwrap()
            .write_buffer(&arrays[8], 0, bytemuck::cast_slice(&[0u32; 4]));
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        owner.encode_candidates(&mut encoder);
        owner.encode_certify_and_publish_grid(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let transfers = transfers.finish();
        assert_eq!((transfers.upload_bytes, transfers.readback_bytes), (0, 0));
        read(&engine, &arrays[8])[0].to_bits()
    };
    assert_eq!(run(&scalar), 0);
    assert_eq!(run(&owner), CANDIDATE_COVERAGE);
    let witness: Vec<_> = read(&engine, owner.count_buffer())
        .into_iter()
        .map(f32::to_bits)
        .collect();
    assert_eq!(witness[1], 1);
    assert_eq!(witness[2], 3);
    assert_eq!(witness[3], index(5, 4, 4) as u32);
    assert_eq!(witness[4], 0);
    assert_eq!(witness[5], 0);
    assert!(f32::from_bits(witness[12]) < -1e-6);
    engine
        .queue()
        .unwrap()
        .write_buffer(&arrays[4], 4, bytemuck::bytes_of(&-1e-5f32));
    assert_eq!(run(&owner), 0);
    assert_eq!(read(&engine, owner.count_buffer())[1].to_bits(), 0);
    engine
        .queue()
        .unwrap()
        .write_buffer(&arrays[4], 4, bytemuck::bytes_of(&f32::NAN));
    assert_eq!(run(&owner), 4);
}

fn buffer(engine: &Engine, bytes: &[u8], size: u64) -> wgpu::Buffer {
    let value = engine
        .device()
        .unwrap()
        .create_buffer(&wgpu::BufferDescriptor {
            label: Some("candidate fixture"),
            size: size.max(bytes.len() as u64),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
    if !bytes.is_empty() {
        engine.queue().unwrap().write_buffer(&value, 0, bytes);
    }
    value
}
fn read(engine: &Engine, source: &wgpu::Buffer) -> Vec<f32> {
    let device = engine.device().unwrap();
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("candidate observation"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |v| {
        let _ = tx.send(v);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(10)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(10))
        .unwrap()
        .unwrap();
    let output = bytemuck::cast_slice::<u8, f32>(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    output
}
fn index(x: usize, y: usize, z: usize) -> usize {
    x + 9 * y + 81 * z
}
fn config(capacity: u32, margin: f32, bodies: u32) -> SoilCandidateConfig {
    SoilCandidateConfig {
        origin: [-0.4; 3],
        cell_size: 0.1,
        grid_size: 9,
        bodies,
        dofs: 1,
        capacity,
        dt: 0.01,
        search_distance: margin,
        max_owned_bytes: u64::MAX,
    }
}
struct Fixture {
    buffers: [wgpu::Buffer; 6],
    candidates: ExperimentalSoilCandidatesGpu,
    geometry: ExperimentalSoilContactGeometryGpu,
    floor: ExperimentalSoilFloorProjectionGpu,
}
impl Fixture {
    fn new(engine: &Engine, grid: &[[f32; 4]], radius: f32, config: SoilCandidateConfig) -> Self {
        Self::new_indexed(engine, grid, radius, config, None)
    }

    fn new_indexed(
        engine: &Engine,
        grid: &[[f32; 4]],
        radius: f32,
        config: SoilCandidateConfig,
        index: Option<SoilCandidateGridIndex<'_>>,
    ) -> Self {
        Self::new_window(engine, grid, radius, config, index, [0; 3])
    }

    fn new_window(
        engine: &Engine,
        grid: &[[f32; 4]],
        radius: f32,
        config: SoilCandidateConfig,
        index: Option<SoilCandidateGridIndex<'_>>,
        cells: [i32; 3],
    ) -> Self {
        let centers = vec![[0.0f32; 4]; config.bodies as usize];
        let shapes: Vec<[f32; 4]> = (0..config.bodies)
            .flat_map(|_| {
                [
                    [0.0; 4],
                    [0.0; 4],
                    [0.0, 0.0, 0.0, 1.0],
                    [radius, radius, radius, 1.0],
                ]
            })
            .collect();
        let jacobians: Vec<f32> = (0..config.bodies)
            .flat_map(|_| [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            .collect();
        let buffers = [
            buffer(engine, bytemuck::cast_slice(grid), 16),
            buffer(engine, bytemuck::cast_slice(&centers), 16),
            buffer(engine, bytemuck::cast_slice(&shapes), 64),
            buffer(engine, bytemuck::cast_slice(&jacobians), 24),
            buffer(engine, &[], 48 * u64::from(config.capacity)),
            buffer(engine, &[], 4 * u64::from(config.capacity)),
        ];
        let nodes = buffer(engine, &[], 48 * u64::from(config.capacity));
        let rows = buffer(engine, &[], 4 * u64::from(config.capacity));
        let mass = buffer(engine, bytemuck::cast_slice(&[860.0f32]), 4);
        let free = buffer(engine, bytemuck::cast_slice(&[0.0f32]), 4);
        let floor = ExperimentalSoilFloorProjectionGpu::new(
            engine,
            SoilFloorInputs {
                mass: &mass,
                free_rates: &free,
                rows: &rows,
                nodes: &nodes,
            },
            SoilFloorCapacity {
                dofs: 1,
                nodes: config.capacity,
                max_iterations: 64,
                max_owned_bytes: u64::MAX,
            },
        )
        .unwrap();
        let geometry = ExperimentalSoilContactGeometryGpu::new(
            engine,
            SoilContactGeometryInputs {
                centers: &buffers[1],
                shapes: &buffers[2],
                jacobians: &buffers[3],
                queries: &buffers[4],
            },
            SoilContactGeometryOutputs {
                nodes: &nodes,
                rows: &rows,
                status: floor.status_buffer(),
            },
            SoilContactGeometryCapacity {
                dofs: 1,
                bodies: config.bodies,
                queries: config.capacity,
                max_owned_bytes: u64::MAX,
            },
        )
        .unwrap();
        let candidates = ExperimentalSoilCandidatesGpu::with_grid_window(
            engine,
            SoilCandidateInputs {
                grid: &buffers[0],
                centers: &buffers[1],
                shapes: &buffers[2],
                jacobians: &buffers[3],
                corrected_rates: floor.rates_buffer(),
                corrected_velocities: floor.velocities_buffer(),
            },
            SoilCandidateOutputs {
                queries: &buffers[4],
                node_ids: &buffers[5],
                status: floor.status_buffer(),
            },
            config,
            RateEncoding::Scalar,
            index,
            cells,
        )
        .unwrap();
        Self {
            buffers,
            candidates,
            geometry,
            floor,
        }
    }
    fn run(&self, engine: &Engine, reset: bool) -> u32 {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = engine
            .device()
            .unwrap()
            .create_command_encoder(&Default::default());
        if reset {
            encoder.clear_buffer(self.floor.status_buffer(), 0, None);
        }
        self.candidates.encode_candidates(&mut encoder);
        self.geometry.encode(&mut encoder);
        self.floor.encode(&mut encoder);
        self.candidates
            .encode_certify_and_publish_grid(&mut encoder);
        engine.queue().unwrap().submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let traffic = transfers.finish();
        assert_eq!(traffic.upload_bytes, 0);
        assert_eq!(traffic.readback_bytes, 0);
        read(engine, self.floor.status_buffer())[0].to_bits()
    }
}

#[test]
#[ignore = "requires real GPU; world-ordered active/compact candidate and publication parity"]
fn indexed_candidates_preserve_world_order_and_publication() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let queue = engine.queue().unwrap();
    let words = |b: &wgpu::Buffer| {
        read(&engine, b)
            .into_iter()
            .map(f32::to_bits)
            .collect::<Vec<_>>()
    };
    for closing in [false, true] {
        let mut grid = if closing {
            vec![[0f32; 4]; 729]
        } else {
            vec![[0.0, 0.0, 0.0, 1.0]; 729]
        };
        if closing {
            grid[index(5, 4, 4)] = [-1.0, 0.0, 0.0, 1.0];
        }
        let cfg = config(128, 0.057, 1);
        let dense = Fixture::new(&engine, &grid, 0.11, cfg);
        assert_eq!(dense.run(&engine, true), 0);
        let expected_queries = words(&dense.buffers[4]);
        let expected_ids = words(&dense.buffers[5]);
        let expected_grid = words(&dense.buffers[0]);
        if closing {
            assert_ne!(expected_grid[index(5, 4, 4) * 4], (-1.0f32).to_bits());
        }
        for sparse in [false, true] {
            let metadata = buffer(&engine, &[], 31 * 4);
            let mut tiles: Vec<u32> = (0..27).rev().collect();
            let pack = |tiles: &[u32]| {
                let mut physical = vec![[0f32; 4]; 27 * 64];
                let mut addresses = vec![None; 27 * 64];
                for (slot, &tile) in tiles.iter().enumerate() {
                    for local in 0..64usize {
                        let x = (tile % 3) as usize * 4 + local % 4;
                        let y = (tile / 3 % 3) as usize * 4 + local / 4 % 4;
                        let z = (tile / 9) as usize * 4 + local / 16;
                        if x < 9 && y < 9 && z < 9 {
                            let world = index(x, y, z);
                            physical[slot * 64 + local] = grid[world];
                            addresses[slot * 64 + local] = Some(world);
                        }
                    }
                }
                (physical, addresses)
            };
            let initial = if sparse { pack(&tiles).0 } else { grid.clone() };
            let fixture = Fixture::new_indexed(
                &engine,
                &initial,
                0.11,
                cfg,
                Some(SoilCandidateGridIndex {
                    metadata: &metadata,
                    metadata_word: 0,
                    tile_ids_word: 4,
                    tile_capacity: 27,
                    sparse,
                }),
            );
            let expected_bytes = 20 * 1728 + 4 * 7 + 132 + grid_index::GridIndex::bytes(1728);
            assert_eq!(
                fixture.candidates.indexed_sort_backend(),
                Some(if cfg!(feature = "parallel-candidate-index") {
                    "parallel-stable-radix"
                } else {
                    "serial-stable-radix"
                })
            );
            assert_eq!(fixture.candidates.owned_buffer_bytes(), expected_bytes);
            for _ in 0..3 {
                let mut meta = vec![27, 27, 0, 0];
                meta.extend(&tiles);
                queue.write_buffer(&metadata, 0, bytemuck::cast_slice(&meta));
                let (physical, addresses) = pack(&tiles);
                queue.write_buffer(
                    &fixture.buffers[0],
                    0,
                    bytemuck::cast_slice(if sparse { &physical } else { &grid }),
                );
                assert_eq!(fixture.run(&engine, true), 0);
                assert_eq!(words(&fixture.buffers[4]), expected_queries);
                assert_eq!(words(&fixture.buffers[5]), expected_ids);
                let actual = words(&fixture.buffers[0]);
                if sparse {
                    for (slot, world) in addresses.into_iter().enumerate() {
                        let expected = world.map_or([0; 4], |i| {
                            expected_grid[4 * i..4 * i + 4].try_into().unwrap()
                        });
                        assert_eq!(&actual[4 * slot..4 * slot + 4], &expected);
                    }
                } else {
                    assert_eq!(actual, expected_grid);
                }
                tiles.rotate_left(7);
            }
            let before = words(&fixture.buffers[0]);
            queue.write_buffer(&metadata, 8, bytemuck::bytes_of(&1u32));
            assert_eq!(fixture.run(&engine, true), 1 << 31);
            assert_eq!(words(&fixture.buffers[0]), before);
            queue.write_buffer(&metadata, 0, bytemuck::cast_slice(&[0u32; 4]));
            assert_eq!(fixture.run(&engine, true), 0);
            assert_eq!(words(fixture.candidates.count_buffer())[0], 0);
            assert!(words(&fixture.buffers[5]).iter().all(|&id| id == u32::MAX));
            eprintln!(
                "indexed candidates sparse={sparse} closing={closing}: three tile permutations, exact world queries/publication, fault freeze and empty reset pass; bytes={expected_bytes}"
            );
        }
    }
}

#[test]
#[ignore = "requires GPU"]
fn candidates_are_sorted_complete_and_repeatable_across_blocks() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut grid = vec![[0.0, 0.0, 0.0, 1.0]; 729];
    grid[index(4, 4, 4)][3] = 1e-12;
    let fixture = Fixture::new(&engine, &grid, 0.11, config(128, 0.057, 1));
    let expected: Vec<u32> = (0..729u32)
        .filter(|&id| {
            let xyz = [id % 9, (id / 9) % 9, id / 81];
            let distance = xyz
                .iter()
                .map(|&v| (-0.4f64 + 0.1 * v as f64).powi(2))
                .sum::<f64>()
                .sqrt();
            distance <= 0.167
        })
        .collect();
    let mut previous = None;
    for _ in 0..3 {
        assert_eq!(fixture.run(&engine, true), 0);
        let count = read(&engine, fixture.candidates.count_buffer())[0].to_bits() as usize;
        let ids: Vec<_> = read(&engine, &fixture.buffers[5])
            .into_iter()
            .map(f32::to_bits)
            .collect();
        assert_eq!(&ids[..count], &expected);
        assert!(ids[count..].iter().all(|&v| v == u32::MAX));
        let queries: Vec<_> = read(&engine, &fixture.buffers[4])
            .into_iter()
            .map(f32::to_bits)
            .collect();
        if let Some(last) = previous {
            assert_eq!(queries, last);
        }
        previous = Some(queries);
        assert_eq!(
            read(&engine, &fixture.buffers[0]),
            grid.iter().flatten().copied().collect::<Vec<_>>()
        );
    }
    assert!(expected.contains(&(index(4, 4, 4) as u32)));
    let empty = vec![[0.0f32; 4]; 729];
    engine
        .queue()
        .unwrap()
        .write_buffer(&fixture.buffers[0], 0, bytemuck::cast_slice(&empty));
    assert_eq!(fixture.run(&engine, true), 0);
    assert_eq!(
        read(&engine, fixture.candidates.count_buffer())[0].to_bits(),
        0
    );
    assert!(
        read(&engine, &fixture.buffers[4])
            .iter()
            .all(|v| v.to_bits() == 0)
    );
    println!(
        "{} sorted candidates, tiny positive mass, partial final block, exact repeats and stale-slot clearing pass",
        expected.len()
    );
}

#[test]
#[ignore = "requires GPU"]
fn omitted_approaching_contact_faults_before_grid_publication() {
    let engine = Engine::new_gpu().unwrap();
    let mut grid = vec![[0.0f32; 4]; 729];
    let id = index(6, 4, 4);
    grid[id] = [-20.0, 0.0, 0.0, 1.0];
    let narrow = Fixture::new(&engine, &grid, 0.11, config(4, 0.0, 1));
    assert_eq!(narrow.run(&engine, true), CANDIDATE_COVERAGE);
    let witness: Vec<_> = read(&engine, narrow.candidates.count_buffer())
        .into_iter()
        .map(f32::to_bits)
        .collect();
    assert_eq!(witness.len(), 17);
    assert_eq!(witness[1], 1);
    assert_eq!(witness[2], 3);
    assert_eq!(witness[3], id as u32);
    assert_eq!(witness[4], 0);
    assert_eq!(witness[5], u32::MAX);
    assert_eq!(f32::from_bits(witness[8]), 0.01);
    assert!(f32::from_bits(witness[12]) < -1e-6);
    assert_eq!(f32::from_bits(witness[16]), 1.);
    assert_eq!(
        read(&engine, &narrow.buffers[0]),
        grid.iter().flatten().copied().collect::<Vec<_>>()
    );
    let wide = Fixture::new(&engine, &grid, 0.11, config(4, 0.1, 1));
    assert_eq!(wide.run(&engine, true), 0);
    assert_eq!(
        read(&engine, wide.candidates.count_buffer())[1].to_bits(),
        0
    );
    let gap = read(&engine, wide.geometry.geometry_buffer())[3] as f64 / (0.01f32 as f64);
    let impulse = (20.0 - gap) * 860.0 / 861.0;
    let q = read(&engine, wide.floor.rates_buffer())[0] as f64;
    let v = read(&engine, &wide.buffers[0])[4 * id] as f64;
    assert!((q + impulse / 860.0).abs() < 2e-5);
    assert!((v - (-20.0 + impulse)).abs() < 2e-5);
    assert!((860.0 * q + v + 20.0).abs() < 2e-5);
    assert!((v - q + gap).abs() < 2e-5);
    println!(
        "excluded incoming contact is rejected; wider GPU candidate/geometry/floor/certificate/publication matches finite-mass impulse"
    );
}

#[test]
#[ignore = "requires GPU"]
fn corrected_machine_motion_can_activate_an_excluded_constraint() {
    let engine = Engine::new_gpu().unwrap();
    let mut grid = vec![[0.0f32; 4]; 729];
    let left = index(3, 4, 4);
    let right = index(6, 4, 4);
    grid[left] = [10.0, 0.0, 0.0, 1.0];
    grid[right] = [0.0, 0.0, 0.0, 1.0];
    let narrow = Fixture::new(&engine, &grid, 0.19995, config(4, 0.0, 1));
    assert_eq!(narrow.run(&engine, true), CANDIDATE_COVERAGE);
    assert!(read(&engine, narrow.floor.rates_buffer())[0] > 0.005);
    assert_eq!(
        read(&engine, &narrow.buffers[0]),
        grid.iter().flatten().copied().collect::<Vec<_>>()
    );
    let wide = Fixture::new(&engine, &grid, 0.19995, config(4, 0.001, 1));
    assert_eq!(wide.run(&engine, true), 0);
    let ids: Vec<_> = read(&engine, &wide.buffers[5])
        .into_iter()
        .map(f32::to_bits)
        .collect();
    assert_eq!(&ids[..2], &[left as u32, right as u32]);
    let samples = read(&engine, wide.geometry.geometry_buffer());
    let right_slot = ids.iter().position(|&v| v == right as u32).unwrap();
    let allowance = samples[4 * right_slot + 3] as f64 / (0.01f32 as f64);
    let q = read(&engine, wide.floor.rates_buffer())[0] as f64;
    assert!((q - (10.0 + allowance) / 862.0).abs() < 2e-5);
    println!("coverage uses corrected, not only free, machine motion");
}

#[test]
#[ignore = "requires GPU"]
fn candidate_capacity_overlap_and_invalid_input_fail_closed() {
    let engine = Engine::new_gpu().unwrap();
    let mut grid = vec![[0.0f32; 4]; 729];
    grid[index(4, 4, 4)] = [0.0, 0.0, 0.0, 1.0];
    grid[index(5, 4, 4)] = [0.0, 0.0, 0.0, 1.0];
    for (cfg, expected) in [
        (config(1, 0.0, 1), CANDIDATE_CAPACITY),
        (config(4, 0.0, 2), CANDIDATE_MULTIPLE_BODIES),
    ] {
        let fixture = Fixture::new(&engine, &grid, 0.11, cfg);
        assert_eq!(fixture.run(&engine, true), expected);
        assert_eq!(
            read(&engine, &fixture.buffers[0]),
            grid.iter().flatten().copied().collect::<Vec<_>>()
        );
        assert_eq!(fixture.run(&engine, false), expected);
    }
    let fixture = Fixture::new(&engine, &grid, 0.11, config(4, 0.0, 1));
    grid[index(4, 4, 4)][3] = -1.0;
    engine
        .queue()
        .unwrap()
        .write_buffer(&fixture.buffers[0], 0, bytemuck::cast_slice(&grid));
    assert_eq!(fixture.run(&engine, true), CANDIDATE_INPUT);
    grid[index(4, 4, 4)][3] = 1.0;
    engine
        .queue()
        .unwrap()
        .write_buffer(&fixture.buffers[0], 0, bytemuck::cast_slice(&grid));
    assert_eq!(fixture.run(&engine, true), 0);
    let mut small = config(4, 0.0, 1);
    small.max_owned_bytes = fixture.candidates.owned_buffer_bytes() - 1;
    let scope = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let rejected = ExperimentalSoilCandidatesGpu::new(
        &engine,
        SoilCandidateInputs {
            grid: &fixture.buffers[0],
            centers: &fixture.buffers[1],
            shapes: &fixture.buffers[2],
            jacobians: &fixture.buffers[3],
            corrected_rates: fixture.floor.rates_buffer(),
            corrected_velocities: fixture.floor.velocities_buffer(),
        },
        SoilCandidateOutputs {
            queries: &fixture.buffers[4],
            node_ids: &fixture.buffers[5],
            status: fixture.floor.status_buffer(),
        },
        small,
    );
    assert!(rejected.is_err());
    assert_eq!(scope.finish().observations, 0);
    assert_eq!(
        fixture.candidates.owned_buffer_bytes(),
        20 * 729 + 4 * 3 + 132
    );
}

#[test]
#[ignore = "requires GPU"]
fn dense_64_cubed_grid_uses_global_sorted_offsets() {
    let engine = Engine::new_gpu().unwrap();
    let grid = vec![[0.0f32, 0.0, 0.0, 1.0]; 64 * 64 * 64];
    let mut cfg = config(64, 0.057, 1);
    cfg.origin = [-3.0; 3];
    cfg.grid_size = 64;
    let fixture = Fixture::new(&engine, &grid, 0.11, cfg);
    assert_eq!(fixture.run(&engine, true), 0);
    let expected: Vec<_> = (0..64 * 64 * 64u32)
        .filter(|&id| {
            let xyz = [id % 64, (id / 64) % 64, id / 4096];
            xyz.iter()
                .map(|&v| (-3.0f64 + 0.1 * v as f64).powi(2))
                .sum::<f64>()
                .sqrt()
                <= 0.167
        })
        .collect();
    let count = read(&engine, fixture.candidates.count_buffer())[0].to_bits() as usize;
    let ids: Vec<_> = read(&engine, &fixture.buffers[5])
        .into_iter()
        .map(f32::to_bits)
        .collect();
    assert_eq!(count, 19);
    assert_eq!(&ids[..count], &expected);
    assert_eq!(fixture.candidates.owned_buffer_bytes(), 5_247_092);
    assert_eq!(
        read(&engine, &fixture.buffers[0]),
        grid.iter().flatten().copied().collect::<Vec<_>>()
    );
    println!(
        "64^3 occupied grid: 1,024 block offsets, 19 sorted candidates and exhaustive coverage pass"
    );
}
