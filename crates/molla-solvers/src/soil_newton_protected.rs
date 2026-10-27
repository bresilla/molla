use molla_core::{Error, Result};
use spirv_tools::val::Validator;
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
#[path = "soil_newton_barrier_fold.rs"]
pub(crate) mod barrier_fold;

pub(crate) const PREFIX: &str = "// NEWTON_PROTECTED_V2\n";
pub(crate) const VOLATILE_MARKER: &str = "// NEWTON_VOLATILE_ROUNDING\n";
pub(crate) const NATIVE_DIVISION_MARKER: &str = "// NEWTON_NATIVE_DIVISION\n";
pub(crate) const UNIT_DIVISION_MARKER: &str = "// NEWTON_UNIT_DIVISION\n";
pub(crate) const FIXED_WIDTH_MARKER: &str = "// NEWTON_FIXED_DOF_WIDTH:";
pub(crate) const FLOAT_CONTROLS_MARKER: &str = "// NEWTON_FLOAT_CONTROLS2_TEST\n";
pub(crate) const SHARED_FLOAT_CONTROLS_MARKER: &str = "// NEWTON_SHARED_FLOAT_CONTROLS2_TEST\n";
pub(crate) const SELECTIVE_FLOAT_CONTROLS_MARKER: &str =
    "// NEWTON_SELECTIVE_FLOAT_CONTROLS2_TEST\n";
#[cfg(test)]
#[path = "soil_newton_initialization.rs"]
pub(crate) mod initialization;
#[cfg(test)]
#[path = "soil_newton_initialization_tests.rs"]
mod initialization_tests;
#[cfg(test)]
#[path = "soil_newton_native_division.rs"]
pub(crate) mod native_division;
#[cfg(test)]
#[path = "soil_newton_noinline.rs"]
pub(crate) mod noinline;
#[cfg(any(test, feature = "strict-rounding-newton-preparation"))]
#[path = "soil_newton_prepare_rounding.rs"]
pub(crate) mod preparation_rounding;
#[cfg(test)]
#[path = "soil_newton_unit_division.rs"]
pub(crate) mod unit_division;
#[cfg(any(
    test,
    feature = "volatile-newton",
    feature = "strict-rounding-newton-preparation"
))]
#[path = "soil_newton_volatile.rs"]
pub(crate) mod volatile;
const HELPERS: [&str; 7] = [
    "rate_sum",
    "rate_add",
    "rate_mul",
    "value",
    "fischer_product",
    "fischer_div",
    "fischer_sqrt",
];

fn invalid(message: impl std::fmt::Display) -> Error {
    Error::Solver(format!("protected Newton compiler: {message}"))
}

pub(crate) fn uses_compiler(source: &str) -> Result<bool> {
    if source.contains(PREFIX) && !source.starts_with(PREFIX) {
        return Err(invalid("protected prefix must precede experiment markers"));
    }
    Ok(source.starts_with(PREFIX))
}

fn replace_once(source: &mut String, old: &str, new: &str) -> Result<()> {
    if source.matches(old).count() != 1 {
        return Err(invalid(format!("source contract mismatch: {old}")));
    }
    *source = source.replacen(old, new, 1);
    Ok(())
}

fn source(original: &str) -> Result<String> {
    let mut source = original.to_owned();
    replace_once(
        &mut source,
        "struct ReleaseChoice { state:Six, score:f32, valid:u32 }",
        "struct ReleaseChoice { k:vec4<f32>, w:vec4<f32>, score:f32, valid:u32 }",
    )?;
    if !source.contains("ReleaseChoice(") || !source.contains("release_best.state") {
        return Err(invalid("missing release-choice uses"));
    }
    source = source.replace("ReleaseChoice(", "release_choice(");
    source = source.replace("release_best.state", "Six(release_best.k,release_best.w)");
    source.push_str("\nfn release_choice(state:Six,score:f32,valid:u32)->ReleaseChoice {return ReleaseChoice(state.k,state.w,score,valid);}\n");
    let round = "fn rate_round(v:f32)->f32 {\n    atomicStore(&rate_scratch[rate_lane],bitcast<u32>(v));\n    return bitcast<f32>(atomicAdd(&rate_scratch[rate_lane],0u));\n}";
    if source
        .matches("var<workgroup> rate_scratch:array<atomic<u32>,64>;\n")
        .count()
        != 1
    {
        return Err(invalid("missing per-lane rounding storage"));
    }
    replace_once(&mut source, round, "fn rate_round(v:f32)->f32 {return v;}")?;
    replace_once(
        &mut source,
        "fn fischer_root(x:f32)->f32 {return rate_round(sqrt(x));}",
        "fn fischer_root(x:f32)->f32 {return retained_round(sqrt(x));}",
    )?;
    let normal = include_str!("../shaders/soil_normal_rate_contract.wgsl");
    replace_once(
        &mut source,
        normal,
        &normal.replace("rate_round(", "retained_round("),
    )?;
    replace_once(
        &mut source,
        "return rate_sum(high,rate_round(fma(a,b.x,-high)+rate_round(a*b.y)));",
        "return retained_pair(rate_sum(high,rate_round(fma(a,b.x,-high)+rate_round(a*b.y))));",
    )?;
    replace_once(
        &mut source,
        "return rate_sum(high,rate_round(tail+cross));",
        "return retained_pair(rate_sum(high,rate_round(tail+cross)));",
    )?;
    source.push_str(&round.replace("fn rate_round(", "\nfn retained_round("));
    source.push_str(
        "\nfn retained_pair(v:vec2<f32>)->vec2<f32> {return vec2<f32>(v.x,retained_round(v.y));}\n",
    );
    Ok(source)
}

fn instructions(words: &[u32]) -> Result<Vec<(usize, &[u32])>> {
    if words.len() < 5 || words[0] != 0x07230203 {
        return Err(invalid("invalid SPIR-V header"));
    }
    let mut result = Vec::new();
    let mut offset = 5;
    while offset < words.len() {
        let count = (words[offset] >> 16) as usize;
        let minimum = match words[offset] & 0xffff {
            5 | 11 | 29 | 54 | 129 | 131 | 133 | 136 => 3,
            12 => 5,
            28 | 32 | 57 => 4,
            30 => 2,
            71 => 3,
            72 => 4,
            _ => 1,
        };
        if count < minimum || count > words.len() - offset {
            return Err(invalid("truncated SPIR-V instruction"));
        }
        result.push((offset, &words[offset..offset + count]));
        offset += count;
    }
    Ok(result)
}

fn string(words: &[u32]) -> Result<String> {
    let bytes: Vec<_> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let end = bytes
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| invalid("unterminated SPIR-V name"))?;
    String::from_utf8(bytes[..end].to_vec()).map_err(invalid)
}

fn implicit_workgroup_layout(words: &[u32]) -> Result<Vec<u32>> {
    let code = instructions(words)?;
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
            if interface.contains(&op[1]) {
                return Err(invalid("shared workgroup/interface layout"));
            }
        } else {
            result.extend_from_slice(op);
        }
    }
    Ok(result)
}

fn decorate(words: &[u32]) -> Result<Vec<u32>> {
    let code = instructions(words)?;
    let mut names = BTreeMap::new();
    let mut glsl = None;
    for (_, op) in &code {
        match op[0] & 0xffff {
            5 => {
                names.insert(op[1], string(&op[2..])?);
            }
            11 if string(&op[2..])? == "GLSL.std.450" => glsl = Some(op[1]),
            _ => {}
        }
    }
    let glsl = glsl.ok_or_else(|| invalid("missing GLSL instructions"))?;
    let round: Vec<_> = names
        .iter()
        .filter(|(_, name)| name.as_str() == "rate_round")
        .map(|(&id, _)| id)
        .collect();
    if round.len() != 1 {
        return Err(invalid("ambiguous rounding helper"));
    }
    let mut function = 0;
    let mut protected = BTreeSet::new();
    let mut calls = 0;
    for (_, op) in &code {
        let helper = names
            .get(&function)
            .is_some_and(|name| HELPERS.contains(&name.as_str()));
        match op[0] & 0xffff {
            54 => function = op[2],
            56 => function = 0,
            57 if op[3] == round[0] => {
                if !helper {
                    return Err(invalid("unprotected rounding caller"));
                }
                calls += 1;
            }
            129 | 131 | 133 | 136 if helper => {
                protected.insert(op[2]);
            }
            12 if helper && op[3] == glsl && op[4] == 50 => {
                protected.insert(op[2]);
            }
            _ => {}
        }
    }
    if calls == 0 || protected.len() < 12 {
        return Err(invalid("missing protected arithmetic"));
    }
    let insert = code
        .iter()
        .find(|(_, op)| matches!(op[0] & 0xffff, 71 | 72 | 19))
        .ok_or_else(|| invalid("missing SPIR-V annotation section"))?
        .0;
    let mut result = words[..insert].to_vec();
    for id in protected {
        result.extend_from_slice(&[(3 << 16) | 71, id, 42]);
    }
    result.extend_from_slice(&words[insert..]);
    Ok(result)
}

pub(crate) fn compile(original: &str, entry: &str) -> Result<Vec<u32>> {
    if original.contains(SELECTIVE_FLOAT_CONTROLS_MARKER)
        && (!cfg!(test)
            || original.matches(SELECTIVE_FLOAT_CONTROLS_MARKER).count() != 1
            || !original.contains(FLOAT_CONTROLS_MARKER))
    {
        return Err(invalid(
            "selective float controls requires one complete test variant",
        ));
    }
    if original.contains(SHARED_FLOAT_CONTROLS_MARKER)
        && (!cfg!(test)
            || original.matches(SHARED_FLOAT_CONTROLS_MARKER).count() != 1
            || !original.contains(FLOAT_CONTROLS_MARKER)
            || !original.contains(VOLATILE_MARKER))
    {
        return Err(invalid(
            "shared float controls requires one complete test marker pair",
        ));
    }
    if original.contains(FLOAT_CONTROLS_MARKER)
        && (!cfg!(test)
            || original.matches(FLOAT_CONTROLS_MARKER).count() != 1
            || original.contains(NATIVE_DIVISION_MARKER)
            || original.contains(UNIT_DIVISION_MARKER))
    {
        return Err(invalid(
            "float controls is an isolated test-only compiler variant",
        ));
    }
    if original.contains(FIXED_WIDTH_MARKER) && !cfg!(test) {
        return Err(invalid("fixed machine width is test-only"));
    }
    if original.matches(UNIT_DIVISION_MARKER).count() > 1
        || (original.contains(UNIT_DIVISION_MARKER)
            && (!cfg!(test) || original.contains(NATIVE_DIVISION_MARKER)))
    {
        return Err(invalid(
            "unit division is an isolated test-only compiler variant",
        ));
    }
    if original.matches(NATIVE_DIVISION_MARKER).count() > 1
        || (original.contains(NATIVE_DIVISION_MARKER) && !cfg!(test))
    {
        return Err(invalid(
            "native division is an isolated test-only compiler variant",
        ));
    }
    #[cfg(test)]
    if original.contains(noinline::MARKER) && original.contains(noinline::DIVISION_MARKER) {
        return Err(invalid("function-control experiments must be isolated"));
    }
    if original.matches(VOLATILE_MARKER).count() > 1
        || (original.contains(VOLATILE_MARKER) && !cfg!(any(test, feature = "volatile-newton")))
    {
        return Err(invalid(
            "volatile rounding feature is not enabled or marker repeated",
        ));
    }
    let source = source(original)?;
    #[cfg(test)]
    let source = if original.contains(FLOAT_CONTROLS_MARKER) {
        let mut source = source;
        let replacement = if original.contains(SHARED_FLOAT_CONTROLS_MARKER) {
            "fn retained_round(v:f32)->f32 {\n    if params.pad0==1u {return v;}\n    atomicStore(&rate_scratch[rate_lane],bitcast<u32>(v));\n    return bitcast<f32>(atomicAdd(&rate_scratch[rate_lane],0u));\n}"
        } else {
            "fn retained_round(v:f32)->f32 {return v;}"
        };
        replace_once(
            &mut source,
            "fn retained_round(v:f32)->f32 {\n    atomicStore(&rate_scratch[rate_lane],bitcast<u32>(v));\n    return bitcast<f32>(atomicAdd(&rate_scratch[rate_lane],0u));\n}",
            replacement,
        )?;
        source
    } else {
        source
    };
    #[cfg(test)]
    let source = if original.contains(NATIVE_DIVISION_MARKER) {
        native_division::source(source)?
    } else {
        source
    };
    #[cfg(test)]
    let source = if original.contains(UNIT_DIVISION_MARKER) {
        unit_division::source(source)?
    } else {
        source
    };
    if !cfg!(any(test, feature = "wide-newton"))
        && source.contains(crate::soil_newton_width::MARKER)
    {
        return Err(invalid("wide Newton feature is not enabled"));
    }
    let source = crate::soil_newton_width::specialize(source)?;
    #[cfg(test)]
    let source = if original.contains(FIXED_WIDTH_MARKER) {
        crate::soil_newton_gpu::distributed::mixed::fixed_width::specialize(source)?
    } else {
        source
    };
    let module = naga::front::wgsl::parse_str(&source).map_err(invalid)?;
    let capabilities = naga::valid::Capabilities::empty();
    #[cfg(test)]
    let capabilities = if original.contains("// NEWTON_SUBGROUP_REDUCTION_TEST\n") {
        capabilities | naga::valid::Capabilities::SUBGROUP
    } else {
        capabilities
    };
    #[cfg(test)]
    let capabilities = if original.contains(NATIVE_DIVISION_MARKER) {
        capabilities | naga::valid::Capabilities::FLOAT64
    } else {
        capabilities
    };
    let info = naga::valid::Validator::new(naga::valid::ValidationFlags::all(), capabilities)
        .validate(&module)
        .map_err(invalid)?;
    let mut options = naga::back::spv::Options::default();
    #[cfg(test)]
    if original.contains("// NEWTON_SUBGROUP_REDUCTION_TEST\n") {
        options.lang_version = (1, 3);
    }
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
    .map_err(invalid)?;
    let words = decorate(&implicit_workgroup_layout(&words)?)?;
    #[cfg(any(test, feature = "volatile-newton"))]
    let words = if original.contains(VOLATILE_MARKER)
        && (!original.contains(FLOAT_CONTROLS_MARKER)
            || original.contains(SHARED_FLOAT_CONTROLS_MARKER))
    {
        volatile::transform(&words)?
    } else {
        words
    };
    #[cfg(test)]
    let words = if original.contains(FLOAT_CONTROLS_MARKER) {
        if original.contains(SELECTIVE_FLOAT_CONTROLS_MARKER) {
            crate::soil_newton_precise_tests::float_controls::selective(&words)
        } else {
            crate::soil_newton_precise_tests::float_controls::strict(&words)
        }
    } else {
        words
    };
    #[cfg(test)]
    let words = if original.contains(noinline::MARKER) {
        noinline::transform(&words)?
    } else if original.contains(noinline::DIVISION_MARKER) {
        noinline::transform_division(&words)?
    } else {
        words
    };
    #[cfg(test)]
    let words = if original.contains(barrier_fold::MARKER) {
        if original.matches(barrier_fold::MARKER).count() != 1 {
            return Err(invalid("barrier fold marker repeated"));
        }
        barrier_fold::transform(&words)?.0
    } else {
        words
    };
    spirv_tools::val::create(Some(spirv_tools::TargetEnv::Vulkan_1_1))
        .validate(&words, None)
        .map_err(invalid)?;
    Ok(words)
}

pub(crate) fn shader(
    device: &wgpu::Device,
    original: &str,
    entry: &str,
) -> Result<wgpu::ShaderModule> {
    if !device
        .features()
        .contains(wgpu::Features::PASSTHROUGH_SHADERS)
    {
        return Err(invalid("passthrough device feature is not enabled"));
    }
    let words = compile(original, entry)?;
    #[cfg(test)]
    if original.contains(FLOAT_CONTROLS_MARKER) {
        // The test factory enables this extension and shaderFloatControls2 together.
        let enabled = unsafe {
            device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .is_some_and(|hal| {
                    hal.enabled_device_extensions()
                        .contains(&ash::khr::shader_float_controls2::NAME)
                })
        };
        if !enabled {
            return Err(invalid(
                "float-controls2 logical-device feature was not enabled",
            ));
        }
    }
    if original.contains(NATIVE_DIVISION_MARKER)
        && !device.features().contains(wgpu::Features::SHADER_F64)
    {
        return Err(invalid("native division requires device f64"));
    }
    // Naga checks source/bounds; SPIRV-Tools validates the protected binary.
    Ok(unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("protected resident Newton"),
            spirv: Some(words.into()),
            ..Default::default()
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_protected_rejects_malformed_binary() {
        for words in [
            vec![],
            vec![0; 5],
            vec![0x07230203, 0, 0, 0, 0, 0],
            vec![0x07230203, 0, 0, 0, 0, (4 << 16) | 32],
        ] {
            assert!(instructions(&words).is_err());
        }
    }

    #[test]
    fn newton_protected_layout_preserves_interfaces_and_rejects_aliases() {
        let mut words = vec![
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
        assert_eq!(implicit_workgroup_layout(&words).unwrap(), words);
        words.extend_from_slice(&[(4 << 16) | 32, 4, 4, 3]);
        assert!(implicit_workgroup_layout(&words).is_err());
        words[15] = 4;
        let result = implicit_workgroup_layout(&words).unwrap();
        assert_eq!(&result[5..], &words[9..]);
    }

    #[test]
    fn newton_protected_rejects_source_drift() {
        let source = crate::soil_newton_gpu::ExperimentalSoilNewtonGpu::conditioned_source(
            crate::soil_newton_gpu::shader_source_for(
                crate::soil_newton_gpu::NewtonVariant::Fischer,
            ),
        );
        for anchor in [
            "fn rate_round(",
            "fn fischer_root(",
            "struct ReleaseChoice",
            "var<workgroup> rate_scratch:",
        ] {
            assert!(compile(&source.replace(anchor, "drift"), "newton").is_err());
        }
        assert!(compile(&source, "missing_entry").is_err());
        let bad = source.replace("fn fischer_root(x:f32)->f32 {return rate_round(sqrt(x));}",
            "fn fischer_root(x:f32)->f32 {return rate_round(sqrt(x));}\nfn unprotected(x:f32)->f32 {return rate_round(x);}");
        assert!(compile(&bad, "newton").is_err());
    }
}
