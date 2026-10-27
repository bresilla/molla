use super::*;

#[test]
fn lattice_contact_node_limits_and_shader_layouts() {
    for cells in [[0; 3], [-1, 0, 1], [1, 0, -1], [-8_388_607, 0, 8_388_544]] {
        validate_node_window([-3.0, -0.5, -3.0], 0.1, 64, cells).unwrap();
    }
    for cells in [
        [i32::MIN, 0, 0],
        [i32::MAX, 0, 0],
        [-8_388_608, 0, 0],
        [0, 0, 8_388_545],
    ] {
        assert!(validate_node_window([-3.0, -0.5, -3.0], 0.1, 64, cells).is_err());
    }
    assert!(validate_node_window([0.0; 3], 0.1, u32::MAX, [0; 3]).is_err());
    assert!(validate_node_window([0.0; 3], 0.1, 0, [0; 3]).is_err());
    assert!(validate_node_window([f32::MAX; 3], f32::MAX, 64, [1; 3]).is_err());
    for h in [0.0, -1.0, f32::NAN, f32::INFINITY] {
        assert!(validate_node_window([0.0; 3], h, 64, [0; 3]).is_err());
    }
    let source = include_str!("../shaders/soil_candidates.wgsl").replace(
        "// ANALYTIC_GEOMETRY",
        include_str!("../shaders/soil_contact_geometry_common.wgsl"),
    );
    for source in [
        candidate_source(source.clone(), RateEncoding::Scalar),
        candidate_source(source, RateEncoding::Paired),
        include_str!("../shaders/mpm_mass_coupling.wgsl").to_owned(),
    ] {
        let module = naga::front::wgsl::parse_str(&source).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        let params = module
            .types
            .iter()
            .find(|(_, ty)| ty.name.as_deref() == Some("Params"))
            .unwrap()
            .1;
        let naga::TypeInner::Struct { members, span } = &params.inner else {
            panic!("params must be a struct")
        };
        assert_eq!(*span, 64);
        assert_eq!(members.last().unwrap().offset, 48);
    }
}

fn flat(xyz: [u32; 3]) -> usize {
    (xyz[0] + 64 * (xyz[1] + 64 * xyz[2])) as usize
}

#[test]
#[ignore = "requires GPU; anchored candidate query, geometry and publication parity"]
fn lattice_contact_gpu_candidates_decimal_window() {
    let engine = Engine::new_gpu().unwrap();
    let output =
        std::path::PathBuf::from(std::env::var_os("MOLLA_LATTICE_CONTACT_OUTPUT").unwrap());
    std::fs::create_dir_all(&output).unwrap();
    let cfg = SoilCandidateConfig {
        origin: [-3.0, -0.5, -3.0],
        grid_size: 64,
        ..config(16, 0.057, 1)
    };
    let global = [[29, 5, 30], [31, 5, 30], [30, 6, 30], [30, 5, 31]];
    let mut expected = None;
    let mut records = Vec::new();
    for cells in [[0; 3], [-1, 0, 1], [1, 0, -1]] {
        let local =
            global.map(|p| std::array::from_fn(|axis| (p[axis] as i32 - cells[axis]) as u32));
        let mut grid = vec![[0.0_f32; 4]; 64_usize.pow(3)];
        for p in local {
            grid[flat(p)] = [0.0, 0.0, 0.0, 1.0];
        }
        let mut tiles = local
            .map(|p| p[0] / 4 + 16 * (p[1] / 4 + 16 * (p[2] / 4)))
            .to_vec();
        tiles.sort_unstable();
        tiles.dedup();
        tiles.reverse();
        let capacity = tiles.len() as u32 + 1;
        let mut metadata_words = vec![tiles.len() as u32, tiles.len() as u32, 0, 0];
        metadata_words.extend(&tiles);
        metadata_words.push(0);
        let metadata = buffer(&engine, bytemuck::cast_slice(&metadata_words), 16);
        let mut packed = vec![[0.0_f32; 4]; capacity as usize * 64];
        let mut occupied_slots = [0; 4];
        for (slot, &tile) in tiles.iter().enumerate() {
            for node in 0..64 {
                let xyz = [
                    tile % 16 * 4 + node % 4,
                    tile / 16 % 16 * 4 + node / 4 % 4,
                    tile / 256 * 4 + node / 16,
                ];
                let address = slot * 64 + node as usize;
                packed[address] = grid[flat(xyz)];
                if let Some(which) = local.iter().position(|&p| p == xyz) {
                    occupied_slots[which] = address;
                }
            }
        }
        for mode in 0..3 {
            let fixture = Fixture::new_window(
                &engine,
                if mode == 2 { &packed } else { &grid },
                0.11,
                cfg,
                (mode != 0).then_some(SoilCandidateGridIndex {
                    metadata: &metadata,
                    metadata_word: 0,
                    tile_ids_word: 4,
                    tile_capacity: capacity,
                    sparse: mode == 2,
                }),
                cells,
            );
            assert_eq!(fixture.run(&engine, true), 0);
            let words = |buffer: &wgpu::Buffer| {
                read(&engine, buffer)
                    .into_iter()
                    .map(f32::to_bits)
                    .collect::<Vec<_>>()
            };
            let queries = words(&fixture.buffers[4]);
            let ids = words(&fixture.buffers[5]);
            assert_eq!(ids.iter().filter(|&&id| id != u32::MAX).count(), 4);
            let normalized_ids: Vec<_> = ids
                .iter()
                .filter(|&&id| id != u32::MAX)
                .map(|&id| {
                    let p = [id % 64, id / 64 % 64, id / 4096];
                    flat(std::array::from_fn(|axis| {
                        (p[axis] as i32 + cells[axis]) as u32
                    })) as u32
                })
                .collect();
            let final_grid = words(&fixture.buffers[0]);
            let occupied: Vec<_> = local
                .iter()
                .enumerate()
                .flat_map(|(i, &p)| {
                    let address = if mode == 2 {
                        occupied_slots[i]
                    } else {
                        flat(p)
                    };
                    final_grid[address * 4..address * 4 + 4].iter().copied()
                })
                .collect();
            let state = (
                queries.clone(),
                normalized_ids,
                occupied,
                words(fixture.floor.rates_buffer()),
                words(fixture.floor.velocities_buffer()),
            );
            if let Some(expected) = &expected {
                assert_eq!(&state, expected);
            } else {
                expected = Some(state);
            }
            let name = format!("c-{}-m-{mode}", cells[0]);
            std::fs::write(
                output.join(format!("{name}.queries")),
                bytemuck::cast_slice(&queries),
            )
            .unwrap();
            records.push(serde_json::json!({"cells":cells,"mode":mode,"exact_query_geometry_publication":true,"candidates":4,"owned_bytes":fixture.candidates.owned_buffer_bytes()}));
        }
    }
    let report = serde_json::json!({"records":records,"scope":"candidate/SDF/floor-publication component; not projected tractor trajectory"});
    std::fs::write(
        output.join("candidates.json"),
        serde_json::to_vec_pretty(&report).unwrap(),
    )
    .unwrap();
    println!("lattice_candidate_report={report}");
}
