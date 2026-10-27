use super::*;
use crate::soil_shared_wall_friction_gpu::capture;
use std::path::Path;
use wgpu::util::DeviceExt;

fn source(input: String) -> String {
    let mut source = soa_scratch::program(input, true);
    let start = source.find("fn prepare_linear() {\n").unwrap() + "fn prepare_linear() {\n".len();
    let reset_end = start + source[start..].find("    storageBarrier();").unwrap();
    let reset = source[start..reset_end].to_owned();
    let loop_start = reset_end
        + source[reset_end..]
            .find("        let c=active_indices[slot];")
            .unwrap();
    let loop_end = loop_start
        + source[loop_start..]
            .find("\n    }\n    storageBarrier();")
            .unwrap();
    let body = source[loop_start..loop_end].to_owned();
    source.push_str(&format!(r#"
@compute @workgroup_size(256)
fn linear_resident(@builtin(local_invocation_index) id:u32) {{
    lane=id;rate_lane=id;prepare_linear();
}}
@compute @workgroup_size(256)
fn linear_reset(@builtin(local_invocation_index) id:u32) {{
    lane=id;rate_lane=id;
{reset}
}}
@compute @workgroup_size(64)
fn linear_parallel(@builtin(local_invocation_index) id:u32,@builtin(workgroup_id) group:vec3<u32>) {{
    lane=id;rate_lane=id;let slot=group.x*64u+id;
    if slot>=status.active_count {{return;}}
{body}
}}
"#));
    source
}

#[test]
fn newton_parallel_linear_source_and_ownership() {
    for count in [1usize, 3, 256, 4096] {
        for words in [36usize, 114] {
            let bytes: Vec<u8> = (0..count * words)
                .flat_map(|i| (0x7f800000u32.wrapping_add(i as u32)).to_le_bytes())
                .collect();
            let encoded = soa(&bytes, count, words);
            let values: Vec<f32> = encoded
                .chunks_exact(4)
                .map(|v| f32::from_bits(u32::from_le_bytes(v.try_into().unwrap())))
                .collect();
            assert_eq!(
                bytemuck::cast_slice::<f32, u8>(&soa_scratch::logical_values(&values, words == 36)),
                bytes
            );
        }
    }
    let source = source(shader_source_for(NewtonVariant::Fischer));
    for entry in ["linear_resident", "linear_parallel"] {
        crate::soil_newton_protected::compile(&source, entry).unwrap();
    }
    let reset = crate::soil_newton_width::specialize(source.clone()).unwrap();
    let module = naga::front::wgsl::parse_str(&reset).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
    for count in [0usize, 1, 63, 64, 65, 255, 256, 257, 601, 4096] {
        let mut visits = vec![0; count];
        for group in 0..count.max(1).div_ceil(64) {
            for lane in 0..64 {
                let slot = group * 64 + lane;
                if slot < count {
                    visits[slot] += 1;
                }
            }
        }
        assert!(visits.iter().all(|&n| n == 1));
    }
    let parallel = source.split("fn linear_parallel(").nth(1).unwrap();
    assert!(!parallel.contains("Barrier") && !parallel.contains("atomicStore"));
    assert!(parallel.contains("atomicAdd(&status.fallbacks,1u)"));
}

fn read_checked(root: &Path, manifest: &serde_json::Value, name: &str) -> Vec<u8> {
    let bytes = std::fs::read(root.join(format!("{name}.bin"))).unwrap();
    assert_eq!(
        manifest["files"][name]["bytes"].as_u64().unwrap(),
        bytes.len() as u64
    );
    assert_eq!(
        manifest["files"][name]["blake3"].as_str().unwrap(),
        blake3::hash(&bytes).to_hex().as_str()
    );
    bytes
}

fn soa(bytes: &[u8], count: usize, words: usize) -> Vec<u8> {
    assert_eq!(bytes.len(), count * words * 4);
    let mut result = vec![0; bytes.len()];
    for c in 0..count {
        for w in 0..words {
            let to = if words == 36 || w < 6 {
                w * count + c
            } else {
                let base = if w < 78 { 6 } else { 78 };
                base * count + 2 * ((w - base) / 2 * count + c) + (w - base) % 2
            };
            result[4 * to..4 * (to + 1)]
                .copy_from_slice(&bytes[4 * (c * words + w)..4 * (c * words + w + 1)]);
        }
    }
    result
}

#[test]
#[ignore = "requires protected GPU; captured linear preparation, lifecycle and paired component timing"]
fn newton_parallel_linear_captured_preparation() {
    captured_preparation(false);
}

#[test]
#[ignore = "requires protected GPU; refresh inverse references on unchanged captured coefficients"]
fn newton_parallel_linear_refresh_references() {
    captured_preparation(true);
}

fn refreshed_reference(
    path: &Path,
    manifest: &serde_json::Value,
    output: &Path,
    snapshot: &[Vec<u8>],
) {
    let relative = if manifest["step"] == 0 {
        "linearizations"
    } else {
        "step-1/linearizations"
    };
    let destination = output.join(relative).join(path.file_name().unwrap());
    std::fs::create_dir_all(&destination).unwrap();
    let mut updated = manifest.clone();
    updated["profile"] = "method20-blocks-current-soa-preparation".into();
    updated["reference_source_manifest_blake3"] =
        blake3::hash(&std::fs::read(path.join("manifest.json")).unwrap())
            .to_hex()
            .to_string()
            .into();
    updated["reference_source_profile"] = manifest["profile"].clone();
    for name in manifest["files"].as_object().unwrap().keys() {
        let bytes = if name == "inverse" {
            let words: Vec<f32> = snapshot[0]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            bytemuck::cast_slice::<f32, u8>(&soa_scratch::logical_values(&words, true)).to_vec()
        } else if name == "linear-status" {
            snapshot[4].clone()
        } else {
            read_checked(path, manifest, name)
        };
        std::fs::write(destination.join(format!("{name}.bin")), &bytes).unwrap();
        updated["files"][name] = serde_json::json!({"bytes":bytes.len(),"blake3":blake3::hash(&bytes).to_hex().to_string()});
    }
    std::fs::copy(
        path.join("original-newton.bin"),
        destination.join("original-newton.bin"),
    )
    .unwrap();
    std::fs::write(
        destination.join("manifest.json"),
        serde_json::to_vec_pretty(&updated).unwrap(),
    )
    .unwrap();
}

fn captured_preparation(refresh: bool) {
    let root =
        std::path::PathBuf::from(std::env::var_os("MOLLA_LINEAR_PREPARATION_INPUT").unwrap());
    let output =
        std::path::PathBuf::from(std::env::var_os("MOLLA_LINEAR_PREPARATION_OUTPUT").unwrap());
    std::fs::create_dir(&output).expect("fresh output directory");
    let engine = Engine::new_gpu_with_features(
        wgpu::Features::PASSTHROUGH_SHADERS
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )
    .unwrap();
    assert!(protected_enabled(&engine) && !engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let storage = |name: &str, bytes: &[u8]| {
        device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytes,
            usage: wgpu::BufferUsages::STORAGE,
        })
    };
    let mut cases: Vec<_> = ["linearizations", "step-1/linearizations"]
        .into_iter()
        .flat_map(|directory| {
            std::fs::read_dir(root.join(directory))
                .unwrap()
                .map(|p| p.unwrap().path())
        })
        .filter(|p| p.join("manifest.json").is_file())
        .collect();
    cases.sort();
    assert_eq!(cases.len(), 16);
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: None,
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolved = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut results = Vec::new();
    for path in cases {
        let manifest: serde_json::Value =
            serde_json::from_slice(&std::fs::read(path.join("manifest.json")).unwrap()).unwrap();
        assert!(matches!(
            manifest["profile"].as_str().unwrap(),
            "method20-pre-hybrid" | "method20-blocks-current-soa-preparation"
        ));
        let n = manifest["width"].as_u64().unwrap() as u32;
        let c = manifest["contacts"].as_u64().unwrap() as usize;
        let active = manifest["active"].as_u64().unwrap() as usize;
        let mass = storage("mass", &read_checked(&path, &manifest, "mass"));
        let free = storage("free", &read_checked(&path, &manifest, "free"));
        let rows = storage("rows", &read_checked(&path, &manifest, "rows"));
        let response = storage("response", &read_checked(&path, &manifest, "response"));
        let contacts = storage("contacts", &read_checked(&path, &manifest, "contacts"));
        let upstream = storage("upstream", &read_checked(&path, &manifest, "upstream"));
        let cap = SoilNewtonCapacity {
            dofs: n,
            contacts: c as u32,
            newton_iterations: 64,
            krylov_iterations: 64,
            max_owned_bytes: u64::MAX,
        };
        let solver = ExperimentalSoilNewtonGpu::new_with_storage(
            &engine,
            SoilNewtonInputs {
                mass: &mass,
                free_rates: &free,
                point_rows: &rows,
                point_response: &response,
                contacts: &contacts,
                upstream_status: &upstream,
            },
            cap,
            NewtonVariant::Fischer,
            source,
            "linear_resident",
            true,
        )
        .unwrap();
        let source = source(shader_source_for(NewtonVariant::Fischer));
        let bindings = solver.pipeline.get_bind_group_layout(0);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[Some(&bindings)],
            immediate_size: 0,
        });
        let pipelines: Vec<_> = ["linear_reset", "linear_parallel"]
            .into_iter()
            .map(|entry| {
                let module = if entry == "linear_reset" {
                    device.create_shader_module(wgpu::ShaderModuleDescriptor {
                        label: Some(entry),
                        source: wgpu::ShaderSource::Wgsl(
                            crate::soil_newton_width::specialize(source.clone())
                                .unwrap()
                                .into(),
                        ),
                    })
                } else {
                    crate::soil_newton_protected::shader(device, &source, entry).unwrap()
                };
                device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(entry),
                    layout: Some(&layout),
                    module: &module,
                    entry_point: Some(entry),
                    compilation_options: Default::default(),
                    cache: None,
                })
            })
            .collect();
        let blocks = soa(&read_checked(&path, &manifest, "blocks"), c, 114);
        let indices = read_checked(&path, &manifest, "active");
        let original_status = read_checked(&path, &manifest, "linear-status");
        queue.write_buffer(&solver.scratch[0], 0, &blocks);
        queue.write_buffer(&solver.scratch[8], 0, &indices);
        let archived_inverse = soa(&read_checked(&path, &manifest, "inverse"), c, 36);
        let observed = [
            &solver.scratch[1],
            &solver.scratch[3],
            &solver.scratch[4],
            &solver.scratch[5],
            &solver.scratch[6],
        ];
        let execute = |candidate: bool, count: usize| {
            let mut status = original_status.clone();
            status[8..12].copy_from_slice(&(count as u32).to_le_bytes());
            queue.write_buffer(&solver.scratch[6], 0, &status);
            for buffer in &observed[..4] {
                queue.write_buffer(buffer, 0, &vec![0x5a; buffer.size() as usize]);
            }
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let dispatch = molla_core::gpu_metrics::DispatchScope::start().unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.write_timestamp(&queries, 0);
            if candidate {
                for (i, pipeline) in pipelines.iter().enumerate() {
                    let mut pass = encoder.begin_compute_pass(&Default::default());
                    pass.set_pipeline(pipeline);
                    pass.set_bind_group(0, &solver.group, &[]);
                    pass.dispatch_workgroups(
                        if i == 0 {
                            1
                        } else {
                            (c as u32).max(1).div_ceil(64)
                        },
                        1,
                        1,
                    );
                    molla_core::gpu_metrics::record_dispatch();
                }
            } else {
                solver.encode(&mut encoder);
            }
            encoder.write_timestamp(&queries, 1);
            encoder.resolve_query_set(&queries, 0..2, &resolved, 0);
            queue.submit([encoder.finish()]);
            assert_eq!(allocations.finish().observations, 0);
            let traffic = traffic.finish();
            assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
            assert_eq!(
                dispatch.finish().encoded_dispatches,
                if candidate { 2 } else { 1 }
            );
            let mut buffers = observed.to_vec();
            buffers.push(&resolved);
            let mut snapshot = capture::read_many(&engine, &buffers);
            let ticks: [u64; 2] = bytemuck::pod_read_unaligned(&snapshot.pop().unwrap());
            let ms = (ticks[1].checked_sub(ticks[0]).unwrap()) as f64
                * f64::from(queue.get_timestamp_period())
                / 1e6;
            assert!(ms.is_finite() && ms > 0.0);
            (ms, snapshot)
        };
        if refresh {
            let snapshot = execute(false, active).1;
            assert_eq!(u32::from_le_bytes(snapshot[4][..4].try_into().unwrap()), 0);
            refreshed_reference(&path, &manifest, &output, &snapshot);
            results.push(serde_json::json!({"case":path.file_name().unwrap().to_str().unwrap(),"refreshed":true}));
            println!(
                "refreshed current inverse on unchanged captured coefficients: {}",
                path.display()
            );
            continue;
        }
        let mut lifecycle = 0;
        for count in [
            active,
            0,
            active.min(1),
            active.min(63),
            active.min(64),
            active.min(65),
            active,
        ] {
            let control = execute(false, count).1;
            let candidate = execute(true, count).1;
            assert_eq!(
                control, candidate,
                "preparation parity {path:?} count={count}"
            );
            for slot in 0..count {
                let index = u32::from_le_bytes(indices[4 * slot..4 * slot + 4].try_into().unwrap())
                    as usize;
                for word in 0..36 {
                    let range = 4 * (word * c + index)..4 * (word * c + index + 1);
                    assert_eq!(
                        control[0][range.clone()],
                        archived_inverse[range],
                        "archived inverse {path:?} contact={index} word={word}"
                    );
                }
            }
            lifecycle += 1;
        }
        let expected = execute(false, active).1;
        let name = path.file_name().unwrap().to_str().unwrap();
        let case_output = output.join(name);
        std::fs::create_dir(&case_output).unwrap();
        for (label, bytes) in ["inverse", "vector", "scratch", "direction", "status"]
            .into_iter()
            .zip(&expected)
        {
            std::fs::write(case_output.join(format!("{label}.bin")), bytes).unwrap();
        }
        for (label, bytes) in ["inverse", "vector", "scratch", "direction", "status"]
            .into_iter()
            .zip(execute(true, active).1)
        {
            std::fs::write(case_output.join(format!("parallel-{label}.bin")), bytes).unwrap();
        }
        let mut pairs = Vec::new();
        for repeat in 0..8 {
            let mut times = [0.0; 2];
            for candidate in if repeat % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            } {
                let (ms, snapshot) = execute(candidate, active);
                assert_eq!(snapshot, expected, "timed repeat {name} mode={candidate}");
                times[usize::from(candidate)] = ms;
            }
            pairs.push(serde_json::json!({"pair":repeat,"resident_ms":times[0],"parallel_ms":times[1],"ratio":times[1]/times[0]}));
        }
        results.push(serde_json::json!({"case":name,"active":active,"capacity":c,"lifecycle_pairs":lifecycle,"timing_pairs":pairs,"exact":true,"scope":"linear preparation only; two dispatches versus one; no nonlinear integration or full-plan acceptance"}));
        std::fs::write(
            output.join("results.json"),
            serde_json::to_vec_pretty(&results).unwrap(),
        )
        .unwrap();
        println!(
            "parallel linear preparation {name}: active={active} lifecycle={lifecycle} exact timing pairs=8"
        );
        engine.check_device().unwrap();
    }
    assert_eq!(results.len(), 16);
}
