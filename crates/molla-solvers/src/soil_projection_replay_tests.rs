use super::*;

#[test]
#[ignore = "requires GPU"]
fn capture_preserves_storage_only_inputs() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let words = [
        vec![0.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 4.0],
        vec![f32::from_bits(0), f32::from_bits(1)],
        vec![2.0],
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        vec![-1.0],
    ];
    let inputs = words.each_ref().map(|words| {
        let result = engine
            .device()
            .unwrap()
            .create_buffer(&wgpu::BufferDescriptor {
                label: Some("storage-only capture input"),
                size: words.len() as u64 * 4,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        engine
            .queue()
            .unwrap()
            .write_buffer(&result, 0, bytemuck::cast_slice(words));
        result
    });
    let mut projection = SoilContactProjectionGpu::new_with_fixed_boundaries(
        &engine,
        SoilProjectionInputs {
            nodes: &inputs[0],
            spans: &inputs[1],
            mass: &inputs[2],
            jacobians: &inputs[3],
            free_rates: &inputs[4],
        },
        SoilProjectionCapacity {
            dofs: 1,
            bodies: 1,
            nodes: 1,
            max_owned_bytes: 1024,
        },
    )
    .unwrap();
    let capture = projection.prepare_replay_capture(&engine);
    submit(&engine, &projection, true);
    for (snapshot, source) in capture.inputs.iter().zip(&words) {
        let actual = read(&engine, snapshot);
        assert_eq!(
            actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            source.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    assert_eq!(read(&engine, &capture.outputs[4])[0].to_bits(), 0);
    assert_eq!(read(&engine, &capture.outputs[0]), vec![0.0]);
    for (snapshot, source) in capture.intermediates.iter().zip(&projection.replay_working) {
        assert_eq!(read(&engine, snapshot), read(&engine, source));
    }
}

pub(crate) fn check_capture(
    engine: &Engine,
    capture: &SoilProjectionReplay,
    label: &str,
    archive: bool,
) -> (f64, usize) {
    let original = capture.inputs.each_ref().map(|source| read(engine, source));
    let expected = capture
        .outputs
        .each_ref()
        .map(|source| read(engine, source));
    let intermediate = capture
        .intermediates
        .each_ref()
        .map(|source| read(engine, source));
    assert_eq!(expected[4][0].to_bits(), 0, "{label}: captured fault");
    let spans: Vec<_> = original[1]
        .chunks_exact(2)
        .map(|s| (s[0].to_bits() as usize, s[1].to_bits() as usize))
        .collect();
    let nodes: Vec<[f32; 8]> = original[0]
        .chunks_exact(8)
        .map(|n| n.try_into().unwrap())
        .collect();
    let boundary_nodes = spans
        .iter()
        .flat_map(|&(start, count)| &nodes[start..start + count])
        .filter(|n| n[7] != 0.0)
        .count();
    println!(
        "{label}: live nodes={}, boundary nodes={boundary_nodes}",
        spans.iter().map(|s| s.1).sum::<usize>()
    );
    if let Some(root) = std::env::var_os("SOIL_PROJECTION_CAPTURE_DIR").filter(|_| archive) {
        let path = std::path::PathBuf::from(root).join(label);
        std::fs::create_dir(&path).unwrap();
        for (kind, data) in [
            ("input", original.as_slice()),
            ("output", expected.as_slice()),
            ("intermediate", intermediate.as_slice()),
        ] {
            for (index, words) in data.iter().enumerate() {
                let bytes: Vec<_> = words
                    .iter()
                    .flat_map(|v| v.to_bits().to_le_bytes())
                    .collect();
                std::fs::write(path.join(format!("{kind}-{index}.bin")), bytes).unwrap();
            }
        }
        std::fs::write(path.join("manifest.json"), serde_json::to_vec_pretty(&serde_json::json!({
            "schema":1, "dofs":capture.capacity.dofs, "bodies":capture.capacity.bodies,
            "nodes":capture.capacity.nodes, "boundary_nodes":boundary_nodes,
            "input_order":["nodes", "spans_u32", "mass", "jacobians", "free_rates"],
            "output_order":["rates", "motion", "impulse", "boundary_impulse", "status_u32"],
            "intermediate_order":["moments", "system", "rhs"], "encoding":"little-endian f32/u32",
        })).unwrap()).unwrap();
    }
    let buffers = original
        .each_ref()
        .map(|words| buffer(engine, bytemuck::cast_slice(words)));
    let projection = SoilContactProjectionGpu::new_with_fixed_boundaries(
        engine,
        SoilProjectionInputs {
            nodes: &buffers[0],
            spans: &buffers[1],
            mass: &buffers[2],
            jacobians: &buffers[3],
            free_rates: &buffers[4],
        },
        capture.capacity,
    )
    .unwrap();
    let mut maximum_ratio = 0.0f64;
    let mut seed = 0x63e409d87b12a55fu64;
    for permutation in 0..8 {
        let mut ordered = nodes.clone();
        for &(start, count) in &spans {
            let live = &mut ordered[start..start + count];
            match permutation {
                0 | 6 => {}
                1 => live.reverse(),
                2 if count > 0 => live.rotate_left(1),
                3 | 4 => {
                    live.sort_by(|a, b| {
                        a[0].total_cmp(&b[0])
                            .then(a[1].total_cmp(&b[1]))
                            .then(a[2].total_cmp(&b[2]))
                    });
                    if permutation == 4 {
                        live.reverse();
                    }
                }
                _ => {
                    for i in (1..count).rev() {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        live.swap(i, seed as usize % (i + 1));
                    }
                }
            }
        }
        engine
            .queue()
            .unwrap()
            .write_buffer(&buffers[0], 0, bytemuck::cast_slice(&ordered));
        for frozen_reduction in [false, true] {
            if frozen_reduction {
                for (target, words) in projection.replay_working.iter().zip(&intermediate) {
                    engine
                        .queue()
                        .unwrap()
                        .write_buffer(target, 0, bytemuck::cast_slice(words));
                }
                let mut flags = vec![0.0f32; capture.capacity.bodies as usize * 8];
                for (b, &(start, count)) in spans.iter().enumerate() {
                    flags[b * 8 + 3] = ordered[start..start + count]
                        .iter()
                        .fold(0u32, |mask, n| mask | n[7] as u32)
                        as f32;
                }
                engine.queue().unwrap().write_buffer(
                    projection.boundary_impulses_buffer(),
                    0,
                    bytemuck::cast_slice(&flags),
                );
                let mut encoder = engine
                    .device()
                    .unwrap()
                    .create_command_encoder(&Default::default());
                encoder.clear_buffer(projection.status_buffer(), 0, None);
                for (pipeline, groups) in [
                    (&projection.pipelines[2], 1),
                    (projection.boundary_pipeline.as_ref().unwrap(), 1),
                    (
                        &projection.pipelines[3],
                        capture.capacity.bodies.div_ceil(64),
                    ),
                ] {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &projection.group, &[]);
                    pass.dispatch_workgroups(groups, 1, 1);
                }
                engine.queue().unwrap().submit([encoder.finish()]);
            } else {
                submit(engine, &projection, true);
            }
            let status = read(engine, projection.status_buffer())[0].to_bits();
            assert_eq!(
                status, 0,
                "{label} permutation={permutation} frozen={frozen_reduction}"
            );
            let actual = [
                projection.rates_buffer(),
                projection.motion_buffer(),
                projection.impulses_buffer(),
                projection.boundary_impulses_buffer(),
            ]
            .map(|b| read(engine, b));
            let mut mismatches = 0;
            let mut ratios = [0.0f64; 4];
            let mut absolute = [0.0f64; 4];
            for (field, (a, b)) in actual.iter().zip(&expected).enumerate() {
                for (&a, &b) in a.iter().zip(b) {
                    assert!(a.is_finite() && b.is_finite());
                    mismatches += usize::from(a.to_bits() != b.to_bits());
                    let error = (f64::from(a) - f64::from(b)).abs();
                    absolute[field] = absolute[field].max(error);
                    let tolerance = 2e-5
                        * if field < 2 {
                            1.0
                        } else {
                            f64::from(b).abs().max(1.0)
                        };
                    ratios[field] = ratios[field].max(error / tolerance);
                }
            }
            println!(
                "{label} permutation={permutation} frozen={frozen_reduction} mismatches={mismatches} absolute={absolute:?} tolerance_ratios={ratios:?}"
            );
            if permutation == 0 || permutation == 6 {
                assert_eq!(mismatches, 0, "identical-input replay");
            }
            maximum_ratio = maximum_ratio.max(ratios.into_iter().fold(0.0, f64::max));
        }
    }
    (maximum_ratio, boundary_nodes)
}

#[test]
#[ignore = "requires GPU and SOIL_PROJECTION_CAPTURE_DIR from an actual tractor capture"]
fn projection_replays_saved_captures() {
    let root = std::path::PathBuf::from(
        std::env::var_os("SOIL_PROJECTION_CAPTURE_DIR").expect("capture directory required"),
    );
    let mut paths: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .filter(|path| path.is_dir() && path.join("manifest.json").is_file())
        .collect();
    paths.sort();
    assert!(!paths.is_empty(), "no captured fixtures");
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut maximum_ratio = 0.0f64;
    let mut boundary_nodes = 0;
    for path in paths {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(manifest["schema"], 1);
        assert_eq!(manifest["encoding"], "little-endian f32/u32");
        let load = |kind: &str, index| {
            let bytes = std::fs::read(path.join(format!("{kind}-{index}.bin"))).unwrap();
            assert_eq!(bytes.len() % 4, 0);
            assert!(!bytes.is_empty());
            let words: Vec<u32> = bytes
                .chunks_exact(4)
                .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            buffer(&engine, bytemuck::cast_slice(&words))
        };
        let capture = SoilProjectionReplay {
            inputs: std::array::from_fn(|i| load("input", i)),
            outputs: std::array::from_fn(|i| load("output", i)),
            intermediates: std::array::from_fn(|i| load("intermediate", i)),
            capacity: SoilProjectionCapacity {
                dofs: manifest["dofs"].as_u64().unwrap().try_into().unwrap(),
                bodies: manifest["bodies"].as_u64().unwrap().try_into().unwrap(),
                nodes: manifest["nodes"].as_u64().unwrap().try_into().unwrap(),
                max_owned_bytes: u64::MAX,
            },
        };
        let (ratio, boundaries) = check_capture(
            &engine,
            &capture,
            path.file_name().unwrap().to_str().unwrap(),
            false,
        );
        maximum_ratio = maximum_ratio.max(ratio);
        boundary_nodes += boundaries;
    }
    println!("saved tractor projection worst tolerance ratio={maximum_ratio}");
    assert!(
        boundary_nodes > 0,
        "no occupied boundaries in saved fixtures"
    );
    assert!(
        maximum_ratio <= 1.0,
        "fixed-input node-order gate failed: {maximum_ratio}"
    );
}

#[test]
#[ignore = "requires GPU and a saved single-boundary tractor capture"]
fn boundary_membership_sensitivity_matches_f64_oracle() {
    let root = std::path::PathBuf::from(
        std::env::var_os("SOIL_PROJECTION_CAPTURE_DIR").expect("capture directory required"),
    )
    .join(
        std::env::var_os("SOIL_PROJECTION_CAPTURE_CASE")
            .unwrap_or_else(|| "frame-24-state-0".into()),
    );
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    let n = manifest["dofs"].as_u64().unwrap() as usize;
    let bodies = manifest["bodies"].as_u64().unwrap() as usize;
    let original: [Vec<f32>; 5] = std::array::from_fn(|i| {
        std::fs::read(root.join(format!("input-{i}.bin")))
            .unwrap()
            .chunks_exact(4)
            .map(|b| f32::from_bits(u32::from_le_bytes(b.try_into().unwrap())))
            .collect()
    });
    let spans: Vec<_> = original[1]
        .chunks_exact(2)
        .map(|s| (s[0].to_bits() as usize, s[1].to_bits() as usize))
        .collect();
    let mut boundary = Vec::new();
    for (body, &(start, count)) in spans.iter().enumerate() {
        for index in start..start + count {
            let mask = original[0][8 * index + 7] as u32;
            for face in 0..6 {
                if mask & (1 << face) != 0 {
                    boundary.push((body, index, face));
                }
            }
        }
    }
    assert_eq!(
        boundary.len(),
        1,
        "single-constraint analytic fixture required"
    );
    let (boundary_body, boundary_index, face) = boundary[0];
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let buffers = original
        .each_ref()
        .map(|words| buffer(&engine, bytemuck::cast_slice(words)));
    let projection = SoilContactProjectionGpu::new_with_fixed_boundaries(
        &engine,
        SoilProjectionInputs {
            nodes: &buffers[0],
            spans: &buffers[1],
            mass: &buffers[2],
            jacobians: &buffers[3],
            free_rates: &buffers[4],
        },
        SoilProjectionCapacity {
            dofs: n as u32,
            bodies: bodies as u32,
            nodes: manifest["nodes"].as_u64().unwrap() as u32,
            max_owned_bytes: u64::MAX,
        },
    )
    .unwrap();
    let mut reference = None;
    for variant in ["occupied", "normal-disabled", "node-removed"] {
        let mut nodes = original[0].clone();
        let mut counts = spans.clone();
        if variant == "normal-disabled" {
            nodes[8 * boundary_index + 7] = 0.0;
        }
        if variant == "node-removed" {
            let (start, count) = counts[boundary_body];
            for lane in 0..8 {
                nodes[8 * boundary_index + lane] = nodes[8 * (start + count - 1) + lane];
            }
            counts[boundary_body].1 -= 1;
        }
        let span_words: Vec<u32> = counts
            .iter()
            .flat_map(|&(a, b)| [a as u32, b as u32])
            .collect();
        engine
            .queue()
            .unwrap()
            .write_buffer(&buffers[0], 0, bytemuck::cast_slice(&nodes));
        engine
            .queue()
            .unwrap()
            .write_buffer(&buffers[1], 0, bytemuck::cast_slice(&span_words));
        let mut h: Vec<f64> = original[2].iter().map(|&x| f64::from(x)).collect();
        let mut rhs: Vec<f64> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| h[i * n + j] * f64::from(original[4][j]))
                    .sum()
            })
            .collect();
        let mut normal_row = vec![0.0; n];
        for (body, &(start, count)) in counts.iter().enumerate() {
            let j: Vec<_> = original[3][body * 6 * n..(body + 1) * 6 * n]
                .iter()
                .map(|&x| f64::from(x))
                .collect();
            for index in start..start + count {
                let node = &nodes[index * 8..index * 8 + 8];
                let cols = point_columns(
                    &j,
                    n,
                    Vec3::new(node[0] as f64, node[1] as f64, node[2] as f64),
                );
                let momentum = Vec3::new(node[4] as f64, node[5] as f64, node[6] as f64);
                for i in 0..n {
                    rhs[i] += cols[i].dot(momentum);
                    for k in 0..n {
                        h[i * n + k] += f64::from(node[3]) * cols[i].dot(cols[k]);
                    }
                }
                if variant == "occupied" && index == boundary_index && body == boundary_body {
                    let mut normal = Vec3::ZERO;
                    normal[face / 2] = if face % 2 == 0 { 1.0 } else { -1.0 };
                    normal_row = cols.iter().map(|c| c.dot(normal)).collect();
                }
            }
        }
        let mut expected = solve_cpu(h.clone(), rhs);
        if variant == "occupied" {
            let violation = normal_row
                .iter()
                .zip(&expected)
                .map(|(a, x)| a * x)
                .sum::<f64>();
            if violation < 0.0 {
                let response = solve_cpu(h, normal_row.clone());
                let denominator = normal_row
                    .iter()
                    .zip(&response)
                    .map(|(a, x)| a * x)
                    .sum::<f64>();
                assert!(denominator > 0.0);
                for (x, r) in expected.iter_mut().zip(response) {
                    *x -= r * violation / denominator;
                }
            }
        }
        submit(&engine, &projection, true);
        assert_eq!(read(&engine, projection.status_buffer())[0].to_bits(), 0);
        let actual = read(&engine, projection.rates_buffer());
        let maximum_error = actual
            .iter()
            .zip(&expected)
            .map(|(&a, &b)| (f64::from(a) - b).abs())
            .fold(0.0, f64::max);
        let previous = reference.get_or_insert_with(|| expected.clone());
        let maximum_change = expected
            .iter()
            .zip(previous.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f64::max);
        println!(
            "membership variant={variant} boundary_body={boundary_body} node_mass={} maximum_gpu_oracle_error={maximum_error} maximum_rate_change={maximum_change} rates={actual:?}",
            original[0][8 * boundary_index + 3]
        );
        assert!(
            maximum_error <= 2e-5,
            "independent f64 rate oracle: {maximum_error}"
        );
    }
}
