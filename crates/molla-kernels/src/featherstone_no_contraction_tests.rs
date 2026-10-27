use super::*;
use molla_compute::Engine;
use std::collections::{BTreeMap, BTreeSet};

const HELPERS: [&str; 2] = ["position_sum", "coordinate_sum"];

#[derive(Clone, Copy, Debug)]
enum Protection {
    WholeSum,
    Intermediate,
    Residual,
}

fn buffer_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

pub(super) fn control_source(source: &str) -> String {
    let mut source = source.to_owned();
    let shared_branch = "    if local_position_rounding {\n        atomicStore(&position_rounding_scratch[position_rounding_lane], bitcast<u32>(value));\n        return bitcast<f32>(atomicAdd(&position_rounding_scratch[position_rounding_lane], 0u));\n    }\n";
    assert_eq!(source.matches(shared_branch).count(), 1);
    source = source.replace(shared_branch, "").replace(
        "var<workgroup> position_rounding_scratch: array<atomic<u32>,64>;",
        "",
    );
    source.replace(
        "override local_position_rounding: bool = false;",
        "const local_position_rounding: bool = false;",
    )
}

pub(super) fn candidate_source(source: &str) -> String {
    let mut source = control_source(source);
    for (old, new) in [
        (
            "let vb = position_round(child, sum - a);",
            "let vb = sum - a;",
        ),
        (
            "let va = position_round(child, sum - vb);",
            "let va = sum - vb;",
        ),
        (
            "let virtual_b = coordinate_round(coord, sum - a);",
            "let virtual_b = sum - a;",
        ),
        (
            "let virtual_a = coordinate_round(coord, sum - virtual_b);",
            "let virtual_a = sum - virtual_b;",
        ),
    ] {
        assert_eq!(
            source.matches(old).count(),
            1,
            "missing exact rounding anchor"
        );
        source = source.replace(old, new);
    }
    source
}

fn instructions(words: &[u32]) -> Vec<(usize, &[u32])> {
    assert_eq!(words[0], 0x07230203);
    let mut result = Vec::new();
    let mut cursor = 5;
    while cursor < words.len() {
        let count = (words[cursor] >> 16) as usize;
        assert!(count > 0 && count <= words.len() - cursor);
        result.push((cursor, &words[cursor..cursor + count]));
        cursor += count;
    }
    result
}

fn named_functions(words: &[u32], names: &[&str]) -> BTreeMap<u32, String> {
    let mut helpers = BTreeMap::new();
    for (_, instruction) in instructions(words) {
        if instruction[0] & 0xffff == 5 {
            let name: Vec<u8> = instruction[2..]
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .take_while(|b| *b != 0)
                .collect();
            let name = std::str::from_utf8(&name).unwrap();
            if names.contains(&name) {
                assert!(helpers.insert(instruction[1], name.to_owned()).is_none());
            }
        }
    }
    helpers
}

pub(super) fn check_rounding_barriers(words: &[u32]) -> std::result::Result<(), &'static str> {
    let helpers = named_functions(words, &["coordinate_round", "position_round"]);
    if helpers.len() != 2 {
        return Err("missing rounding helpers");
    }
    let mut function = None;
    let mut counts = BTreeMap::<u32, [u32; 2]>::new();
    for (_, instruction) in instructions(words) {
        let op = instruction[0] & 0xffff;
        if op == 54 {
            function = Some(instruction[2]);
        }
        if let Some(id) = function.filter(|id| helpers.contains_key(id)) {
            if matches!(op, 227 | 228) {
                counts.entry(id).or_default()[(op - 227) as usize] += 1;
            }
        }
        if op == 56 {
            function = None;
        }
    }
    if helpers.keys().any(|id| counts.get(id) != Some(&[1, 1])) {
        return Err("rounding helper lost atomic load/store");
    }
    Ok(())
}

fn decorate(words: &[u32]) -> (Vec<u32>, usize) {
    decorate_with(words, Protection::WholeSum)
}

fn decorate_with(words: &[u32], protection: Protection) -> (Vec<u32>, usize) {
    let code = instructions(words);
    let helpers = named_functions(words, &HELPERS);
    assert_eq!(helpers.len(), HELPERS.len(), "missing named sum helpers");
    let mut function = None;
    let mut results = BTreeSet::new();
    let mut per_helper = BTreeMap::<u32, Vec<&[u32]>>::new();
    for (_, instruction) in &code {
        let op = instruction[0] & 0xffff;
        if op == 54 {
            function = Some(instruction[2]);
        }
        if function.is_some_and(|id| helpers.contains_key(&id)) && matches!(op, 129 | 131) {
            assert_eq!(instruction.len(), 5);
            per_helper
                .entry(function.unwrap())
                .or_default()
                .push(instruction);
        }
        if op == 56 {
            function = None;
        }
    }
    assert_eq!(per_helper.len(), 2);
    for operations in per_helper.values() {
        let opcodes: Vec<_> = operations.iter().map(|i| i[0] & 0xffff).collect();
        assert_eq!(
            opcodes,
            [129, 131, 131, 131, 131, 129],
            "unexpected sum arithmetic"
        );
        let o = operations;
        assert_eq!(o[1][4], o[0][3]);
        assert_eq!(o[2][3], o[1][3]);
        assert_eq!(o[2][4], o[1][2]);
        assert_eq!(&o[3][3..], &[o[0][3], o[2][2]]);
        assert_eq!(&o[4][3..], &[o[0][4], o[1][2]]);
        assert_eq!(&o[5][3..], &[o[3][2], o[4][2]]);
        for (index, instruction) in operations.iter().enumerate() {
            let protect = match protection {
                Protection::WholeSum => true,
                Protection::Intermediate => matches!(index, 1 | 2),
                Protection::Residual => index != 0,
            };
            if protect {
                assert!(results.insert(instruction[2]));
            }
        }
    }
    let insert = code
        .iter()
        .find(|(_, i)| matches!(i[0] & 0xffff, 71 | 72 | 19))
        .unwrap()
        .0;
    let mut result = Vec::with_capacity(words.len() + 3 * results.len());
    result.extend_from_slice(&words[..insert]);
    for id in &results {
        result.extend_from_slice(&[(3 << 16) | 71, *id, 42]);
    }
    result.extend_from_slice(&words[insert..]);
    (result, results.len())
}

pub(super) fn compile(source: &str, entry: &str) -> Vec<u32> {
    compile_with_bounds(source, entry, naga::proc::BoundsCheckPolicy::Restrict)
}

fn compile_with_bounds(
    source: &str,
    entry: &str,
    buffer_policy: naga::proc::BoundsCheckPolicy,
) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let mut options = naga::back::spv::Options::default();
    options.bounds_check_policies.index = naga::proc::BoundsCheckPolicy::Restrict;
    options.bounds_check_policies.buffer = buffer_policy;
    options.flags |= naga::back::spv::WriterFlags::DEBUG;
    options.fake_missing_bindings = false;
    for (_, global) in module.global_variables.iter() {
        if let Some(binding) = global.binding {
            options.binding_map.insert(
                binding,
                naga::back::spv::BindingInfo {
                    descriptor_set: binding.group,
                    binding: binding.binding,
                    binding_array_size: None,
                },
            );
        }
    }
    naga::back::spv::write_vec(
        &module,
        &info,
        &options,
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: entry.into(),
        }),
    )
    .unwrap()
}

pub(super) fn shader(device: &wgpu::Device, source: &str, entry: &str) -> wgpu::ShaderModule {
    load_shader(device, source, entry, Some(Protection::WholeSum))
}

pub(super) fn control_shader(
    device: &wgpu::Device,
    source: &str,
    entry: &str,
) -> wgpu::ShaderModule {
    load_shader(device, source, entry, None)
}

pub(super) fn intermediate_shader(
    device: &wgpu::Device,
    source: &str,
    entry: &str,
) -> wgpu::ShaderModule {
    load_shader(device, source, entry, Some(Protection::Intermediate))
}

pub(super) fn residual_shader(
    device: &wgpu::Device,
    source: &str,
    entry: &str,
) -> wgpu::ShaderModule {
    load_shader(device, source, entry, Some(Protection::Residual))
}

fn load_shader(
    device: &wgpu::Device,
    source: &str,
    entry: &str,
    protection: Option<Protection>,
) -> wgpu::ShaderModule {
    assert!(
        device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS)
    );
    let compiled = compile(source, entry);
    let (words, count) = if let Some(protection) = protection {
        decorate_with(&compiled, protection)
    } else {
        (compiled, 0)
    };
    check_rounding_barriers(&words).expect("retained atomic rounding barriers");
    let directory = std::path::PathBuf::from(
        std::env::var_os("SOIL_NO_CONTRACTION_OUTPUT").expect("explicit artifact directory"),
    );
    std::fs::create_dir_all(&directory).unwrap();
    let binary = directory.join(format!("{entry}.spv"));
    std::fs::write(directory.join(format!("{entry}.wgsl")), source).unwrap();
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    std::fs::write(&binary, bytes).unwrap();
    let validator = std::env::var_os("SPIRV_VAL").expect("explicit SPIRV-Tools validator");
    let validation = std::process::Command::new(validator)
        .args(["--target-env", "vulkan1.1"])
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        validation.status.success(),
        "SPIR-V validation failed: {}",
        String::from_utf8_lossy(&validation.stderr)
    );
    eprintln!("validated {entry}: {count} selective NoContraction decorations");
    // Naga checks bounds; SPIRV-Tools validates the decoration-only binary change.
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("test selective no-contraction"),
            spirv: Some(words.into()),
            ..Default::default()
        })
    }
}

fn component_source() -> String {
    let source = include_str!("../shaders/featherstone.wgsl");
    let coordinate = source
        .split("fn coordinate_round")
        .nth(1)
        .unwrap()
        .split("fn coordinate_advance")
        .next()
        .unwrap();
    let position = source
        .split("override local_position_rounding")
        .nth(1)
        .unwrap()
        .split("fn position_add")
        .next()
        .unwrap();
    format!(
        "fn coordinate_round{coordinate}\noverride local_position_rounding{position}\n{}",
        r#"
@group(0) @binding(0) var<storage,read> inputs:array<vec2<f32>>;
@group(0) @binding(1) var<storage,read_write> coordinate_error:array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> position_error:array<atomic<u32>>;
@group(0) @binding(3) var<storage,read_write> outputs:array<vec4<f32>>;
@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) id:vec3<u32>) {
    if id.x>=arrayLength(&inputs) { return; }
    let input=inputs[id.x];
    outputs[id.x]=vec4<f32>(position_sum(id.x,input.x,input.y),coordinate_sum(id.x,input.x,input.y));
}
"#
    )
}

#[test]
fn selective_decoration_preserves_binary_and_helper_scope() {
    let source = component_source();
    let candidate = candidate_source(&source);
    assert_eq!(candidate.matches("let vb = sum - a;").count(), 1);
    let original = compile(&candidate, "evaluate");
    let (decorated, count) = decorate(&original);
    assert_eq!(count, 12);
    let mut stripped = decorated[..5].to_vec();
    for (_, instruction) in instructions(&decorated) {
        if instruction[0] & 0xffff != 71 || instruction[2] != 42 {
            stripped.extend_from_slice(instruction);
        }
    }
    assert_eq!(stripped, original);
}

#[test]
fn intermediate_protection_preserves_every_other_binary_word() {
    let compiled = compile(&candidate_source(&component_source()), "evaluate");
    for (protection, expected) in [(Protection::Intermediate, 4), (Protection::Residual, 10)] {
        let (decorated, count) = decorate_with(&compiled, protection);
        assert_eq!(count, expected);
        assert_eq!(check_rounding_barriers(&decorated), Ok(()));
        let mut stripped = decorated[..5].to_vec();
        for (_, instruction) in instructions(&decorated) {
            if instruction[0] & 0xffff != 71 || instruction[2] != 42 {
                stripped.extend_from_slice(instruction);
            }
        }
        assert_eq!(stripped, compiled);
    }
}

#[test]
fn passthrough_control_keeps_sum_bodies_and_has_no_decorations() {
    let original = component_source();
    let control = control_source(&original);
    for name in HELPERS {
        let body = |text: &str| {
            let start = text.find(&format!("fn {name}(")).unwrap();
            let end = start + text[start..].find("\n}").unwrap() + 2;
            text[start..end].to_owned()
        };
        assert_eq!(body(&original), body(&control));
    }
    let words = compile(&control, "evaluate");
    assert_eq!(check_rounding_barriers(&words), Ok(()));
    assert!(
        !instructions(&words)
            .iter()
            .any(|(_, i)| i[0] & 0xffff == 71 && i[2] == 42)
    );
}

#[test]
fn selective_barrier_guard_rejects_conditional_nonatomic_lowering() {
    let source = candidate_source(&component_source());
    let atomic = compile(&source, "evaluate");
    assert_eq!(check_rounding_barriers(&atomic), Ok(()));
    let conditional = compile_with_bounds(
        &source,
        "evaluate",
        naga::proc::BoundsCheckPolicy::ReadZeroSkipWrite,
    );
    assert_eq!(
        check_rounding_barriers(&conditional),
        Err("rounding helper lost atomic load/store")
    );
    let (decorated, _) = decorate(&conditional);
    assert!(check_rounding_barriers(&decorated).is_err());
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools"]
fn selective_sum_matches_storage_and_f64_oracle() {
    component_comparison(false, Some(Protection::WholeSum));
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools"]
fn selective_sum_preserves_composed_caller_products() {
    component_comparison(true, Some(Protection::WholeSum));
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools"]
fn passthrough_control_sum_matches_storage_and_f64_oracle() {
    component_comparison(false, None);
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools"]
fn intermediate_sum_matches_storage_and_f64_oracle() {
    component_comparison(false, Some(Protection::Intermediate));
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools"]
fn intermediate_sum_preserves_composed_caller_products() {
    component_comparison(true, Some(Protection::Intermediate));
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools"]
fn residual_sum_matches_storage_and_f64_oracle() {
    component_comparison(false, Some(Protection::Residual));
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools"]
fn residual_sum_preserves_composed_caller_products() {
    component_comparison(true, Some(Protection::Residual));
}

fn component_comparison(product: bool, protection: Option<Protection>) {
    let engine = Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    assert_eq!(engine.adapter_info().unwrap().backend, "Vulkan");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut cases = vec![
        [1.0f32, 2.0f32.powi(-25)],
        [1e8, 1.0],
        [65536.0, -0.000004],
        [0.8, -0.09553365],
        [-1.0, 2.0f32.powi(-26)],
        [1e-20, 1e20],
        [-1e20, -1e-20],
        [1.0, -1.0],
    ];
    let mut seed = 0x71468ac3u32;
    for _ in 0..4097 {
        cases.push(std::array::from_fn(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            f32::from_bits((seed & 0x807fffff) | ((67 + ((seed >> 10) % 120)) << 23))
        }));
    }
    if product {
        cases = (0..4105)
            .map(|i| {
                let a = f32::from_bits(0x3f800001 + 7919 * i);
                [a, -(a * a)]
            })
            .collect();
    }
    let buffer = |size| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let input = buffer(cases.len() as u64 * 8);
    let coordinate = buffer(cases.len() as u64 * 4);
    let position = buffer(cases.len() as u64 * 16);
    let output = buffer(cases.len() as u64 * 16);
    queue.write_buffer(&input, 0, bytemuck::cast_slice(&cases));
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[
            storage_entries(1, true)[0],
            storage_entries(2, false)[1],
            storage_entries(3, false)[2],
            storage_entries(4, false)[3],
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[
            buffer_entry(0, &input),
            buffer_entry(1, &coordinate),
            buffer_entry(2, &position),
            buffer_entry(3, &output),
        ],
    });
    let reference = component_source();
    let reference = if product {
        assert_eq!(reference.matches("let input=inputs[id.x];").count(), 1);
        reference.replace(
            "let input=inputs[id.x];",
            "let input=vec2<f32>(inputs[id.x].x*inputs[id.x].x,inputs[id.x].y);",
        )
    } else {
        reference
    };
    let candidate = if protection.is_none() {
        control_source(&reference)
    } else {
        candidate_source(&reference)
    };
    let mut results = Vec::new();
    for candidate_run in [false, true] {
        let module = if candidate_run {
            load_shader(device, &candidate, "evaluate", protection)
        } else {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(reference.clone().into()),
            })
        };
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("evaluate"),
            compilation_options: Default::default(),
            cache: None,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.clear_buffer(&coordinate, 0, None);
        encoder.clear_buffer(&position, 0, None);
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
        }
        queue.submit([encoder.finish()]);
        let result = rounding_tests::words(device, queue, &output);
        let coordinate = rounding_tests::words(device, queue, &coordinate);
        let position = rounding_tests::words(device, queue, &position);
        for (i, ([a, b], pair)) in cases.iter().zip(result.chunks_exact(4)).enumerate() {
            let (a, b) = (f64::from(*a), f64::from(*b));
            let sum = a + b;
            let vb = sum - a;
            let tail = (a - (sum - vb)) + (b - vb);
            let high = sum as f32;
            let low = ((sum - f64::from(high)) + tail) as f32;
            if !product {
                assert_eq!(
                    pair,
                    &[high.to_bits(), low.to_bits(), high.to_bits(), low.to_bits()],
                    "candidate={candidate_run} case={i}"
                );
            }
            assert_eq!(coordinate[i], pair[3]);
            assert_eq!(position[4 * i + 3], pair[1]);
            assert_eq!(&position[4 * i..4 * i + 3], &[0, 0, 0]);
        }
        results.push(result);
    }
    let differences: Vec<_> = results[0]
        .iter()
        .zip(&results[1])
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .collect();
    for (index, (a, b)) in differences.iter().take(8) {
        eprintln!(
            "selective sum product={product} word={index} reference={a:#010x} candidate={b:#010x}"
        );
    }
    assert_eq!(
        differences.len(),
        0,
        "product={product}: reference/candidate words"
    );
    eprintln!(
        "selective sum product={product} protection={protection:?}: {} cases, both helpers and scratch writebacks exact",
        cases.len()
    );
}
