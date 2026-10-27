use super::*;

fn source() -> String {
    let mut s = refined_source_from(
        source_from(super::super::linear_capture::method20_source(
            shader_source_for(NewtonVariant::Fischer),
        )),
        0,
    );
    let a = s
        .find("@compute @workgroup_size(256)\nfn cycle_seed(")
        .unwrap();
    let b = a + s[a..]
        .find("@compute @workgroup_size(256)\nfn cycle_resident(")
        .unwrap();
    s.replace_range(a..b,r#"
@compute @workgroup_size(256)
fn cycle_seed(@builtin(local_invocation_index) id:u32) {
    lane=id;rate_lane=id;
    if lane==0u {status.iterations=0u;status.relative=0.0;status.beta=0.0;status.residual=0.0;status.pad=0.0;}
    for(var c=lane;c<params.contacts;c+=64u) {
        direction[c]=Six();states[c]=State();
        direction[max(params.contacts,1u)+c]=Six();direction[2u*max(params.contacts,1u)+c]=Six();
        basis[params.iterations*max(params.contacts,1u)+c].z=Six();
    }
    for(var slot=lane;slot<status.active_count;slot+=64u) {
        let c=active_indices[slot];var rhs:array<f32,6>;
        for(var i=0u;i<6u;i++) {rhs[i]=-blocks[c].residual[i];}
        vector[c]=packed(rhs);scratch[c]=packed(rhs);
    }
    storageBarrier();let initial=scratch_norm();
    if lane==0u {condensed[2180u]=vec2<f32>(initial,0.0);}
}
"#);
    s
}

#[test]
fn newton_distributed_frozen_source_contract() {
    let s = source();
    let seed = s
        .split_once("fn cycle_seed(")
        .unwrap()
        .1
        .split_once("fn cycle_resident(")
        .unwrap()
        .0;
    assert!(!seed.contains("prepare_linear()") && !seed.contains("blocks[c]="));
    for entry in [
        "cycle_seed",
        "refined_initialize",
        "cycle_precondition",
        "cycle_advance",
        "cycle_forward",
        "cycle_resident",
    ] {
        crate::soil_newton_protected::compile(&marked(s.clone()), entry)
            .unwrap_or_else(|e| panic!("{entry}: {e}"));
    }
}

fn floats(bytes: &[u8]) -> Vec<f64> {
    assert_eq!(bytes.len() % 4, 0);
    bytes
        .chunks_exact(4)
        .map(|v| {
            let x = f32::from_le_bytes(v.try_into().unwrap());
            assert!(x.is_finite());
            f64::from(x)
        })
        .collect()
}

fn residual(
    n: usize,
    active: &[usize],
    blocks: &[f64],
    rows: &[f64],
    response: &[f64],
    state: &[f64],
) -> f64 {
    let x = |c: usize, i: usize| {
        let at = 16 * c + if i < 3 { i } else { i + 1 };
        state[at] + state[at + 8]
    };
    let q: Vec<f64> = (0..n)
        .map(|i| {
            active
                .iter()
                .map(|&c| {
                    (0..3)
                        .map(|a| -response[4 * (c * n + i) + a] * x(c, a))
                        .sum::<f64>()
                })
                .sum()
        })
        .collect();
    let mut rhs = 0.;
    let mut error = 0.;
    for &c in active {
        let point: [f64; 3] =
            std::array::from_fn(|a| (0..n).map(|i| rows[4 * (c * n + i) + a] * q[i]).sum());
        let b = &blocks[114 * c..114 * (c + 1)];
        for i in 0..6 {
            let direct: f64 = (0..6)
                .map(|j| (b[6 + 2 * (6 * i + j)] + b[7 + 2 * (6 * i + j)]) * x(c, j))
                .sum();
            let machine: f64 = (0..3)
                .map(|j| (b[78 + 2 * (3 * i + j)] + b[79 + 2 * (3 * i + j)]) * point[j])
                .sum();
            error += (b[i] + direct + machine).powi(2);
            rhs += b[i] * b[i];
        }
    }
    if rhs == 0. {
        assert_eq!(error, 0.);
        0.
    } else {
        (error / rhs).sqrt()
    }
}

#[test]
#[ignore = "requires protected GPU and exact captured tractor linearizations"]
fn newton_distributed_frozen_replay() {
    use crate::soil_shared_wall_friction_gpu::capture;
    use wgpu::util::DeviceExt;
    let input =
        std::path::PathBuf::from(std::env::var_os("MOLLA_DISTRIBUTED_LINEAR_INPUT").unwrap());
    let output =
        std::path::PathBuf::from(std::env::var_os("MOLLA_DISTRIBUTED_LINEAR_OUTPUT").unwrap());
    std::fs::create_dir(&output).expect("fresh replay directory");
    let mut paths: Vec<_> = std::fs::read_dir(&input)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.join("manifest.json").is_file())
        .collect();
    paths.sort();
    assert_eq!(paths.len(), 8);
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mixed = std::env::var_os("MOLLA_NEWTON_FROZEN_MIXED").is_some();
    let mut failures = 0;
    let mut maximum_iterations = 0;
    for path in paths {
        let meta: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("manifest.json")).unwrap()).unwrap();
        assert_eq!(meta["schema"], 1);
        let n = meta["width"].as_u64().unwrap() as usize;
        let count = meta["contacts"].as_u64().unwrap() as usize;
        let active_count = meta["active"].as_u64().unwrap() as usize;
        assert!((1..=32).contains(&n) && count <= 16384 && active_count <= count);
        assert_eq!(meta["iterations"], 64);
        let read = |name: &str| {
            let bytes = std::fs::read(path.join(format!("{name}.bin"))).unwrap();
            assert_eq!(
                bytes.len() as u64,
                meta["files"][name]["bytes"].as_u64().unwrap()
            );
            assert_eq!(
                blake3::hash(&bytes).to_hex().as_str(),
                meta["files"][name]["blake3"].as_str().unwrap()
            );
            bytes
        };
        let names = ["mass", "free", "rows", "response", "contacts", "upstream"];
        let borrowed: Vec<_> = names
            .map(|name| {
                let bytes = read(name);
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(name),
                    contents: &bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                })
            })
            .into_iter()
            .collect();
        let snapshots = [
            read("blocks"),
            read("inverse"),
            read("linear-status"),
            read("active"),
        ];
        let active: Vec<_> = snapshots[3]
            .chunks_exact(4)
            .take(active_count)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
            .collect();
        assert!(active.iter().all(|&c| c < count));
        assert_eq!(
            active
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            active.len()
        );
        let blocks = floats(&snapshots[0]);
        let rows = floats(&read("rows"));
        let response = floats(&read("response"));
        assert_eq!(blocks.len(), 114 * count.max(1));
        assert_eq!(rows.len(), 4 * n * count.max(1));
        assert_eq!(response.len(), rows.len());
        let case = output.join(path.file_name().unwrap());
        std::fs::create_dir(&case).unwrap();
        let mut baseline = None;
        let mut results = Vec::new();
        let mut control_fault = None;
        for variant in 0..2 {
            let mut variant_baseline = None;
            let probe = CycleProbe::new_source(
                &engine,
                SoilNewtonInputs {
                    mass: &borrowed[0],
                    free_rates: &borrowed[1],
                    point_rows: &borrowed[2],
                    point_response: &borrowed[3],
                    contacts: &borrowed[4],
                    upstream_status: &borrowed[5],
                },
                SoilNewtonCapacity {
                    dofs: n as u32,
                    contacts: count as u32,
                    newton_iterations: 64,
                    krylov_iterations: 64,
                    max_owned_bytes: u64::MAX,
                },
                variant == 1 && !mixed,
                true,
                if variant == 1 && mixed {
                    super::super::mixed::selected_source(source())
                } else {
                    source()
                },
            )
            .unwrap();
            for repeat in 0..2 {
                for (&index, bytes) in [0usize, 1, 6, 8].iter().zip(&snapshots) {
                    let buffer = &probe.owner.scratch[index];
                    assert_eq!(bytes.len() as u64, buffer.size());
                    queue.write_buffer(buffer, 0, bytes);
                }
                let mut encoder = device.create_command_encoder(&Default::default());
                probe.encode(&mut encoder);
                queue.submit([encoder.finish()]);
                let report = capture::read(&engine, probe.owner.status_buffer());
                let state = capture::read(&engine, probe.owner.state_buffer());
                std::fs::write(
                    case.join(format!("v{variant}-r{repeat}-report.bin")),
                    &report,
                )
                .unwrap();
                std::fs::write(case.join(format!("v{variant}-r{repeat}-state.bin")), &state)
                    .unwrap();
                let exact = baseline
                    .as_ref()
                    .is_none_or(|(r, s)| r == &report && s == &state);
                let repeat_exact = variant_baseline
                    .as_ref()
                    .is_none_or(|(r, s)| r == &report && s == &state);
                if variant_baseline.is_none() {
                    variant_baseline = Some((report.clone(), state.clone()));
                }
                if baseline.is_none() {
                    baseline = Some((report.clone(), state.clone()));
                }
                let word =
                    |i: usize| u32::from_le_bytes(report[4 * i..4 * i + 4].try_into().unwrap());
                let fault = word(0);
                if control_fault.is_none() {
                    control_fault = Some(fault);
                }
                let iterations = word(2);
                maximum_iterations = maximum_iterations.max(iterations);
                assert!(fault == 0 || fault == 8);
                assert!(iterations <= 64);
                let error = residual(n, &active, &blocks, &rows, &response, &floats(&state));
                assert!(error.is_finite());
                let valid = fault != 0 || error <= 1e-5;
                let comparison = if mixed {
                    repeat_exact && (control_fault != Some(0) || fault == 0)
                } else {
                    exact
                };
                failures += usize::from(!comparison || !valid);
                println!(
                    "distributed-frozen case={} variant={variant} repeat={repeat} active={active_count} iterations={iterations} fault={fault} f64_relative={error:e} exact={exact} valid={valid}",
                    path.file_name().unwrap().to_string_lossy()
                );
                results.push(serde_json::json!({"variant":variant,"repeat":repeat,"iterations":iterations,"fault":fault,"f64_relative":error,"exact":exact,"repeat_exact":repeat_exact,"comparison":comparison,"mixed":mixed,"mixed_kind":mixed.then(super::super::mixed::kind),"valid":valid}));
            }
        }
        std::fs::write(
            case.join("results.json"),
            serde_json::to_vec_pretty(&results).unwrap(),
        )
        .unwrap();
    }
    println!("distributed-frozen failures={failures} maximum_iterations={maximum_iterations}");
    assert_eq!(failures, 0);
}
