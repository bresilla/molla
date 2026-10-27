use super::*;

pub(crate) const ATOMIC: &str = "// PREPARATION_ATOMIC_ROUNDING\n";
pub(crate) const VOLATILE: &str = "// PREPARATION_VOLATILE_ROUNDING\n";
pub(crate) const STRICT: &str = "// PREPARATION_STRICT_VOLATILE_ROUNDING\n";

fn no_contraction(words: &[u32]) -> Result<Vec<u32>> {
    let code = instructions(words)?;
    let mut existing = BTreeSet::new();
    let mut targets = BTreeSet::new();
    let mut glsl = None;
    for (_, op) in &code {
        match op[0] & 0xffff {
            11 if string(&op[2..])? == "GLSL.std.450" => glsl = Some(op[1]),
            71 if op.len() == 3 && op[2] == 42 => {
                existing.insert(op[1]);
            }
            129 | 131 | 133 | 136 => {
                targets.insert(op[2]);
            }
            12 if Some(op[3]) == glsl && op[4] == 50 => {
                targets.insert(op[2]);
            }
            _ => {}
        }
    }
    if targets.is_empty() {
        return Err(invalid("preparation arithmetic missing"));
    }
    let insert = code
        .iter()
        .find(|(_, op)| matches!(op[0] & 0xffff, 71 | 72 | 19))
        .ok_or_else(|| invalid("preparation annotations missing"))?
        .0;
    let mut result = words[..insert].to_vec();
    for id in targets.difference(&existing) {
        result.extend_from_slice(&[(3 << 16) | 71, *id, 42]);
    }
    result.extend_from_slice(&words[insert..]);
    Ok(result)
}

pub(crate) fn compile(original: &str, entry: &str, volatile: bool) -> Result<Vec<u32>> {
    if (!cfg!(test) && (!volatile || !original.starts_with(STRICT)))
        || !matches!(entry, "factor" | "prepare")
        || original.matches("fn rate_round(").count() != 1
        || original.contains("retained_round")
    {
        return Err(invalid("preparation rounding source contract"));
    }
    let source = original.replace("rate_round", "retained_round");
    let module = naga::front::wgsl::parse_str(&source).map_err(invalid)?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::empty(),
    )
    .validate(&module)
    .map_err(invalid)?;
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
    .map_err(invalid)?;
    let words = implicit_workgroup_layout(&words)?;
    let words = if volatile {
        super::volatile::transform(&words)?
    } else {
        words
    };
    let words = if original.starts_with(STRICT) {
        no_contraction(&words)?
    } else {
        words
    };
    spirv_tools::val::create(Some(spirv_tools::TargetEnv::Vulkan_1_1))
        .validate(&words, None)
        .map_err(invalid)?;
    Ok(words)
}

pub(crate) fn shaders(
    engine: &molla_compute::Engine,
    source: &str,
    volatile: bool,
) -> Result<[wgpu::ShaderModule; 2]> {
    if !crate::soil_newton_gpu::protected_enabled(engine) {
        return Err(invalid("preparation rounding requires Vulkan passthrough"));
    }
    let device = engine
        .device()
        .ok_or_else(|| invalid("device unavailable"))?;
    let mut result = Vec::new();
    for entry in ["factor", "prepare"] {
        let words = compile(source, entry, volatile)?;
        // Naga checks source/bounds; SPIRV-Tools validates each transformed module.
        result.push(unsafe {
            device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                label: Some("experimental preparation rounding"),
                spirv: Some(words.into()),
                ..Default::default()
            })
        });
    }
    result
        .try_into()
        .map_err(|_| invalid("preparation module count"))
}

#[test]
fn newton_preparation_rounding_compiler_contract() {
    let source = crate::soil_newton_prepare_gpu::axis_parallel_source();
    for entry in ["factor", "prepare"] {
        let atomic = compile(&source, entry, false).unwrap();
        let volatile = compile(&source, entry, true).unwrap();
        assert_eq!(volatile, super::volatile::transform(&atomic).unwrap());
        assert_ne!(atomic, volatile);
        let strict = compile(&format!("{STRICT}{source}"), entry, true).unwrap();
        assert_eq!(strict, no_contraction(&volatile).unwrap());
        assert_eq!(strict, no_contraction(&strict).unwrap());
        if let Some(root) = std::env::var_os("MOLLA_PREPARATION_ROUNDING_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join("source.wgsl"), &source).unwrap();
            for (name, words) in [
                ("atomic", atomic),
                ("volatile", volatile),
                ("strict", strict),
            ] {
                std::fs::write(
                    root.join(format!("{entry}-{name}.spv")),
                    bytemuck::cast_slice(&words),
                )
                .unwrap();
            }
        }
    }
    assert!(compile(&source, "finish", false).is_err());
    assert!(compile("", "prepare", true).is_err());
    assert!(
        compile(
            &source.replace("rate_round", "retained_round"),
            "factor",
            false
        )
        .is_err()
    );
    assert!(
        compile(
            &source.replace(
                "atomicAdd(&rate_scratch[rate_lane],0u)",
                "atomicAdd(&rate_scratch[rate_lane],1u)"
            ),
            "prepare",
            true
        )
        .is_err()
    );
}
