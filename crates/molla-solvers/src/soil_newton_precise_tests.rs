use std::collections::{BTreeMap, BTreeSet};

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_float_controls_tests.rs"]
pub(crate) mod float_controls;

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[path = "soil_newton_shader_clock_tests.rs"]
pub(crate) mod shader_clock;

const HELPERS: [&str; 7] = [
    "rate_sum",
    "rate_add",
    "rate_mul",
    "value",
    "fischer_product",
    "fischer_div",
    "fischer_sqrt",
];

pub(crate) fn local_trace_source(mut source: String, contact: u32) -> String {
    let start = source.find("fn fischer_sqrt(").unwrap();
    let end = source.find("fn fischer_precise(").unwrap();
    let root = &source[start..end];
    let returned =
        "return rate_add(vec2<f32>(high,0.0),vec2<f32>(value(remainder)/(2.0*high),0.0));";
    assert_eq!(root.matches(returned).count(), 1);
    let root = root.replace("let remainder=rate_add(square,-rate_mul(high,vec2<f32>(high,0.0)));", "let precise_square=rate_mul(high,vec2<f32>(high,0.0));let remainder=rate_add(square,-precise_square);");
    let root = root.replace(returned, "let result=rate_add(vec2<f32>(high,0.0),vec2<f32>(value(remainder)/(2.0*high),0.0));if precision_sqrt_trace {condensed[67]=vec2<f32>(high,0.0);condensed[68]=remainder;condensed[69]=vec2<f32>(value(remainder),2.0*high);condensed[70]=vec2<f32>(value(remainder)/(2.0*high),0.0);condensed[71]=result;condensed[72]=precise_square;}return result;");
    source.replace_range(start..end, &root);
    let anchor = "block=fischer_block(a,block.residual);";
    assert_eq!(source.matches(anchor).count(), 1);
    source = source.replace(
        anchor,
        "precision_capture(a);block=fischer_block(a,block.residual);",
    );
    source.push_str(
        &include_str!("../shaders/soil_newton_precision_trace.wgsl")
            .replace("__CONTACT__", &contact.to_string()),
    );
    source
}

fn instructions(words: &[u32]) -> Vec<(usize, &[u32])> {
    assert!(words.len() >= 5 && words[0] == 0x07230203);
    let mut result = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        assert!(count > 0 && count <= words.len() - offset);
        result.push((offset, &words[offset..offset + count]));
        offset += count;
    }
    result
}

fn names(words: &[u32]) -> BTreeMap<u32, String> {
    instructions(words)
        .into_iter()
        .filter_map(|(_, code)| {
            if code[0] & 0xffff != 5 {
                return None;
            }
            let bytes: Vec<_> = code[2..]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .take_while(|v| *v != 0)
                .collect();
            Some((code[1], String::from_utf8(bytes).unwrap()))
        })
        .collect()
}

fn source(source: &str, candidate: bool) -> String {
    let mut source = source.to_owned();
    let choice = "struct ReleaseChoice { state:Six, score:f32, valid:u32 }";
    if source.contains(choice) {
        assert_eq!(source.matches(choice).count(), 1);
        source = source.replace(
            choice,
            "struct ReleaseChoice { k:vec4<f32>, w:vec4<f32>, score:f32, valid:u32 }",
        );
        source = source.replace("ReleaseChoice(", "release_choice(");
        source = source.replace("release_best.state", "Six(release_best.k,release_best.w)");
        source.push_str("\nfn release_choice(state:Six,score:f32,valid:u32)->ReleaseChoice {return ReleaseChoice(state.k,state.w,score,valid);}\n");
    }
    if !candidate {
        return source;
    }
    let old = "fn rate_round(v:f32)->f32 {\n    atomicStore(&rate_scratch[rate_lane],bitcast<u32>(v));\n    return bitcast<f32>(atomicAdd(&rate_scratch[rate_lane],0u));\n}";
    assert_eq!(source.matches(old).count(), 1);
    let scratch = "var<workgroup> rate_scratch:array<atomic<u32>,64>;\n";
    assert_eq!(source.matches(scratch).count(), 1);
    let mut source = source.replace(old, "fn rate_round(v:f32)->f32 {return v;}");
    let root = "fn fischer_root(x:f32)->f32 {return rate_round(sqrt(x));}";
    if source.contains(root) {
        assert_eq!(source.matches(root).count(), 1);
        source = source.replace(
            root,
            "fn fischer_root(x:f32)->f32 {return retained_round(sqrt(x));}",
        );
    }
    let normal = include_str!("../shaders/soil_normal_rate_contract.wgsl");
    if source.contains("fn normal_rate_add(") {
        assert_eq!(source.matches(normal).count(), 1);
        source = source.replace(normal, &normal.replace("rate_round(", "retained_round("));
    }
    let product = "return rate_sum(high,rate_round(fma(a,b.x,-high)+rate_round(a*b.y)));";
    assert_eq!(source.matches(product).count(), 1);
    source = source.replace(
        product,
        "return retained_pair(rate_sum(high,rate_round(fma(a,b.x,-high)+rate_round(a*b.y))));",
    );
    let fischer = "return rate_sum(high,rate_round(tail+cross));";
    if source.contains("fn fischer_product(") {
        assert_eq!(source.matches(fischer).count(), 1);
        source = source.replace(
            fischer,
            "return retained_pair(rate_sum(high,rate_round(tail+cross)));",
        );
    }
    source.push_str(&old.replace("fn rate_round(", "\nfn retained_round("));
    source.push_str("\nfn retained_pair(v:vec2<f32>)->vec2<f32> {return vec2<f32>(retained_round(v.x),retained_round(v.y));}\n");
    source
}

pub(crate) fn boundary_variant() -> &'static str {
    match std::env::var("MOLLA_NEWTON_PRECISE_BOUNDARY").as_deref() {
        Err(std::env::VarError::NotPresent) | Ok("products") => "products",
        Ok("quotient") => "quotient",
        Ok("low-product") => "low-product",
        _ => panic!("MOLLA_NEWTON_PRECISE_BOUNDARY must be products, quotient or low-product"),
    }
}

fn quotient_source(mut source: String) -> String {
    let product =
        "return retained_pair(rate_sum(high,rate_round(fma(a,b.x,-high)+rate_round(a*b.y))));";
    assert_eq!(source.matches(product).count(), 1);
    source = source.replace(
        product,
        "return rate_sum(high,rate_round(fma(a,b.x,-high)+rate_round(a*b.y)));",
    );
    source = source.replace(
        "return retained_pair(rate_sum(high,rate_round(tail+cross)));",
        "return rate_sum(high,rate_round(tail+cross));",
    );
    if source.contains("fn fischer_div(") {
        let high = "let high=rate_round(a.x/b.x);";
        assert_eq!(source.matches(high).count(), 1);
        source = source.replace(high, "let high=retained_round(a.x/b.x);");
    }
    source.replace(
        "let high=a.x/b.x;outputs",
        "let high=retained_round(a.x/b.x);outputs",
    )
}

fn low_product_source(source: String) -> String {
    let pair = "return vec2<f32>(retained_round(v.x),retained_round(v.y));";
    assert_eq!(source.matches(pair).count(), 1);
    source.replace(pair, "return vec2<f32>(v.x,retained_round(v.y));")
}

fn compile(source: &str, entry: &str) -> Vec<u32> {
    let module = naga::front::wgsl::parse_str(source).unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .unwrap();
    let mut options = naga::back::spv::Options::default();
    options.bounds_check_policies.index = naga::proc::BoundsCheckPolicy::Restrict;
    options.bounds_check_policies.buffer = naga::proc::BoundsCheckPolicy::Restrict;
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
    let words = naga::back::spv::write_vec(
        &module,
        &info,
        &options,
        Some(&naga::back::spv::PipelineOptions {
            shader_stage: naga::ShaderStage::Compute,
            entry_point: entry.into(),
        }),
    )
    .unwrap();
    implicit_workgroup_layout(&words)
}

fn implicit_workgroup_layout(words: &[u32]) -> Vec<u32> {
    let code = instructions(words);
    let mut children = BTreeMap::<u32, Vec<u32>>::new();
    let mut workgroup = Vec::new();
    let mut interface = Vec::new();
    for (_, op) in &code {
        match op[0] & 0xffff {
            28 | 29 => {
                children.insert(op[1], vec![op[2]]);
            }
            30 => {
                children.insert(op[1], op[2..].to_vec());
            }
            32 if op[2] == 4 => workgroup.push(op[3]),
            32 if matches!(op[2], 2 | 9 | 12) => interface.push(op[3]),
            _ => {}
        }
    }
    let reachable = |mut pending: Vec<u32>| {
        let mut found = BTreeSet::new();
        while let Some(id) = pending.pop() {
            if found.insert(id) {
                pending.extend(children.get(&id).into_iter().flatten().copied());
            }
        }
        found
    };
    let workgroup = reachable(workgroup);
    let interface = reachable(interface);
    let mut result = words[..5].to_vec();
    for (_, op) in code {
        let layout = (op[0] & 0xffff == 71 && op[2] == 6)
            || (op[0] & 0xffff == 72 && matches!(op[3], 4 | 5 | 7 | 35));
        if layout && workgroup.contains(&op[1]) {
            assert!(
                !interface.contains(&op[1]),
                "shared workgroup/interface array type requires distinct layout"
            );
        } else {
            result.extend_from_slice(op);
        }
    }
    result
}

fn decorate(words: &[u32]) -> (Vec<u32>, usize) {
    let names = names(words);
    let round = names
        .iter()
        .find(|(_, name)| *name == "rate_round")
        .unwrap()
        .0;
    let code = instructions(words);
    let glsl = code
        .iter()
        .find(|(_, op)| {
            op[0] & 0xffff == 11
                && op[2..]
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .take_while(|byte| *byte != 0)
                    .eq(b"GLSL.std.450".iter().copied())
        })
        .map(|(_, op)| op[1])
        .expect("GLSL extended instruction set");
    let mut function = 0;
    let mut protected = BTreeSet::new();
    let mut calls = 0;
    for (_, op) in &code {
        match op[0] & 0xffff {
            54 => function = op[2],
            56 => function = 0,
            57 if op[3] == *round => {
                let caller = names.get(&function).unwrap();
                assert!(
                    HELPERS.contains(&caller.as_str()),
                    "unprotected rounding caller {caller}"
                );
                calls += 1;
            }
            129 | 131 | 133 | 136
                if names
                    .get(&function)
                    .is_some_and(|name| HELPERS.contains(&name.as_str())) =>
            {
                assert!(protected.insert(op[2]));
            }
            12 if op[3] == glsl
                && op[4] == 50
                && names
                    .get(&function)
                    .is_some_and(|name| HELPERS.contains(&name.as_str())) =>
            {
                assert!(protected.insert(op[2]));
            }
            _ => {}
        }
    }
    assert!(calls > 0 && protected.len() >= 3);
    let insert = code
        .iter()
        .find(|(_, op)| matches!(op[0] & 0xffff, 71 | 72 | 19))
        .unwrap()
        .0;
    let mut result = words[..insert].to_vec();
    for id in &protected {
        result.extend_from_slice(&[(3 << 16) | 71, *id, 42]);
    }
    result.extend_from_slice(&words[insert..]);
    (result, protected.len())
}

pub(crate) fn shader(
    device: &wgpu::Device,
    original: &str,
    entry: &str,
    candidate: bool,
) -> wgpu::ShaderModule {
    assert!(
        device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS)
    );
    let source = source(original, candidate);
    let source = match (candidate, boundary_variant()) {
        (true, "quotient") => quotient_source(source),
        (true, "low-product") => low_product_source(source),
        _ => source,
    };
    let original_words = compile(&source, entry);
    let (words, count) = if candidate {
        decorate(&original_words)
    } else {
        (original_words, 0)
    };
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_NEWTON_PRECISE_OUTPUT").expect("explicit artifact directory"),
    );
    let key = format!(
        "{entry}-{}-{}",
        if candidate { "precise" } else { "control" },
        blake3::hash(source.as_bytes()).to_hex()
    );
    let directory = root.join(key);
    std::fs::create_dir_all(&directory).unwrap();
    std::fs::write(directory.join("shader.wgsl"), &source).unwrap();
    let binary = directory.join("shader.spv");
    let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&binary, &bytes).unwrap();
    let validation = std::process::Command::new(
        std::env::var_os("SPIRV_VAL").expect("explicit SPIRV-Tools validator"),
    )
    .args(["--target-env", "vulkan1.1"])
    .arg(&binary)
    .output()
    .unwrap();
    std::fs::write(directory.join("validation.log"), &validation.stderr).unwrap();
    assert!(
        validation.status.success(),
        "{}",
        String::from_utf8_lossy(&validation.stderr)
    );
    eprintln!(
        "Newton precise candidate={candidate} validated decorations={count} bytes={}",
        bytes.len()
    );
    // Naga validates source and bounds; SPIRV-Tools validates the decorated binary.
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("test Newton precise arithmetic"),
            spirv: Some(words.into()),
            ..Default::default()
        })
    }
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
pub(crate) fn rounding_shader(
    device: &wgpu::Device,
    original: &str,
    entry: &str,
    volatile: bool,
) -> wgpu::ShaderModule {
    use spirv_tools::val::Validator;
    let source = low_product_source(source(original, true));
    let (words, _) = decorate(&compile(&source, entry));
    let words = if volatile {
        crate::soil_newton_protected::volatile::transform(&words).unwrap()
    } else {
        words
    };
    spirv_tools::val::create(Some(spirv_tools::TargetEnv::Vulkan_1_1))
        .validate(&words, None)
        .unwrap();
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_NEWTON_ROUNDING_ARTIFACTS")
            .expect("explicit rounding artifact directory"),
    );
    std::fs::create_dir_all(&root).unwrap();
    let key = format!("{volatile}-{}", blake3::hash(source.as_bytes()).to_hex());
    std::fs::write(root.join(format!("{key}.wgsl")), &source).unwrap();
    std::fs::write(
        root.join(format!("{key}.spv")),
        words
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    // Source and protected binary pass Naga and SPIRV-Tools validation.
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("rounding access probe"),
            spirv: Some(words.into()),
            ..Default::default()
        })
    }
}

#[test]
fn newton_precise_scope_and_control() {
    let source = format!(
        "{}\n@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) id:u32) {{rate_lane=id;let result=rate_mul(2.0,rate_add(vec2<f32>(1.0),vec2<f32>(3.0)));}}",
        crate::soil_rate_encoding::sorted_arithmetic(64)
    );
    let control = compile(&source, "main");
    assert!(
        instructions(&control)
            .iter()
            .any(|(_, op)| op[0] & 0xffff == 228)
    );
    assert!(
        instructions(&control)
            .iter()
            .any(|(_, op)| op[0] & 0xffff == 234)
    );
    let candidate = compile(&self::source(&source, true), "main");
    let candidate_names = names(&candidate);
    let mut function = 0;
    let mut atomic_count = 0;
    for (_, op) in instructions(&candidate) {
        match op[0] & 0xffff {
            54 => function = op[2],
            56 => function = 0,
            227..=242 => {
                assert_eq!(candidate_names[&function], "retained_round");
                atomic_count += 1;
            }
            _ => {}
        }
    }
    assert_eq!(atomic_count, 2);
    let (decorated, count) = decorate(&candidate);
    assert!(count >= 3);
    let mut restored = decorated[..5].to_vec();
    for (_, op) in instructions(&decorated) {
        if !(op[0] & 0xffff == 71 && op.len() == 3 && op[2] == 42) {
            restored.extend_from_slice(op);
        }
    }
    assert_eq!(restored, candidate);
}

#[test]
#[should_panic(expected = "unprotected rounding caller")]
fn newton_precise_rejects_unprotected_rounding() {
    let source = format!(
        "{}\n@compute @workgroup_size(64) fn main() {{let result=rate_round(1.0);}}",
        crate::soil_rate_encoding::sorted_arithmetic(64)
    );
    decorate(&compile(&self::source(&source, true), "main"));
}

#[test]
fn newton_precise_full_shader_structure() {
    let source = crate::soil_newton_gpu::ExperimentalSoilNewtonGpu::conditioned_source(
        crate::soil_newton_gpu::shader_source_for(crate::soil_newton_gpu::NewtonVariant::Fischer),
    );
    for candidate in [false, true] {
        let words = compile(&self::source(&source, candidate), "newton");
        if candidate {
            let (_, count) = decorate(&words);
            assert!(count >= 12);
            eprintln!("full Newton protected arithmetic operations={count}");
        }
    }
}

#[test]
fn newton_precise_quotient_shader_structure() {
    let source = crate::soil_newton_gpu::ExperimentalSoilNewtonGpu::conditioned_source(
        crate::soil_newton_gpu::shader_source_for(crate::soil_newton_gpu::NewtonVariant::Fischer),
    );
    let candidate = quotient_source(self::source(&source, true));
    assert_eq!(candidate.matches("retained_pair(").count(), 1);
    assert!(candidate.contains("let high=retained_round(a.x/b.x);"));
    assert!(candidate.contains("return retained_round(sqrt(x));"));
    let (_, count) = decorate(&compile(&candidate, "newton"));
    assert!(count >= 12);
}

#[cfg(all(feature = "protected-newton", not(target_arch = "wasm32")))]
#[test]
fn newton_protected_runtime_matches_private_compiler() {
    let source = crate::soil_newton_gpu::ExperimentalSoilNewtonGpu::conditioned_source(
        crate::soil_newton_gpu::shader_source_for(crate::soil_newton_gpu::NewtonVariant::Fischer),
    );
    let private = decorate(&compile(
        &low_product_source(self::source(&source, true)),
        "newton",
    ))
    .0;
    let runtime = crate::soil_newton_protected::compile(&source, "newton").unwrap();
    assert_eq!(private, runtime);
}

#[test]
fn newton_precise_local_shader_structure() {
    let original = crate::soil_newton_gpu::ExperimentalSoilNewtonGpu::conditioned_source(
        crate::soil_newton_gpu::shader_source_for(crate::soil_newton_gpu::NewtonVariant::Fischer),
    );
    for candidate in [false, true] {
        let words = compile(
            &source(&local_trace_source(original.clone(), 116), candidate),
            "precision_local",
        );
        if candidate {
            decorate(&words);
        }
    }
}

#[test]
fn newton_precise_low_product_shader_structure() {
    let original = crate::soil_newton_gpu::ExperimentalSoilNewtonGpu::conditioned_source(
        crate::soil_newton_gpu::shader_source_for(crate::soil_newton_gpu::NewtonVariant::Fischer),
    );
    let candidate = low_product_source(source(&original, true));
    assert!(candidate.contains("return vec2<f32>(v.x,retained_round(v.y));"));
    assert_eq!(candidate.matches("retained_pair(").count(), 3);
    let (_, count) = decorate(&compile(&candidate, "newton"));
    assert!(count >= 12);
}

#[test]
#[should_panic(expected = "shared workgroup/interface array type")]
fn newton_precise_rejects_layout_alias() {
    let words = [
        0x07230203,
        0x10000,
        0,
        10,
        0,
        (4 << 16) | 71,
        3,
        6,
        4,
        (4 << 16) | 28,
        3,
        1,
        2,
        (4 << 16) | 32,
        4,
        4,
        3,
        (4 << 16) | 32,
        5,
        12,
        3,
    ];
    implicit_workgroup_layout(&words);
}

#[test]
fn newton_precise_preserves_interface_stride() {
    let words = [
        0x07230203,
        0x10000,
        0,
        10,
        0,
        (4 << 16) | 71,
        3,
        6,
        4,
        (4 << 16) | 28,
        3,
        1,
        2,
        (4 << 16) | 32,
        5,
        12,
        3,
    ];
    assert_eq!(implicit_workgroup_layout(&words), words);
}

#[test]
fn newton_precise_workgroup_member_layout() {
    let words = [
        0x07230203,
        0x10000,
        0,
        10,
        0,
        (5 << 16) | 72,
        3,
        0,
        35,
        0,
        (5 << 16) | 72,
        6,
        0,
        35,
        0,
        (3 << 16) | 30,
        3,
        1,
        (3 << 16) | 30,
        6,
        1,
        (4 << 16) | 32,
        4,
        4,
        3,
        (4 << 16) | 32,
        5,
        12,
        6,
    ];
    let mut expected = words[..5].to_vec();
    expected.extend_from_slice(&words[10..]);
    assert_eq!(implicit_workgroup_layout(&words), expected);
    let mut alias = words.to_vec();
    *alias.last_mut().unwrap() = 3;
    assert!(std::panic::catch_unwind(|| implicit_workgroup_layout(&alias)).is_err());
}
