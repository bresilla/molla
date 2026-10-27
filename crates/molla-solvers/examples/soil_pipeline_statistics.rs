use ash::{
    ext::robustness2,
    khr::{pipeline_executable_properties, shader_float_controls2},
    vk,
};
use serde_json::{Value, json};
use spirv_tools::val::Validator;
use std::{collections::BTreeMap, error::Error, ffi::CStr, path::Path};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

struct Context {
    _entry: ash::Entry,
    instance: ash::Instance,
    device: Option<ash::Device>,
}

impl Drop for Context {
    fn drop(&mut self) {
        unsafe {
            if let Some(device) = &self.device {
                device.destroy_device(None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

struct Pipeline<'a> {
    device: &'a ash::Device,
    bindings: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
}

impl Drop for Pipeline<'_> {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_shader_module(self.module, None);
            self.device.destroy_pipeline_layout(self.layout, None);
            self.device
                .destroy_descriptor_set_layout(self.bindings, None);
        }
    }
}

fn words(bytes: &[u8]) -> Result<Vec<u32>> {
    if bytes.len() < 20 || bytes.len() > 8 * 1024 * 1024 || bytes.len() % 4 != 0 {
        return Err("expected a bounded, word-aligned SPIR-V module".into());
    }
    let words: Vec<_> = bytes
        .chunks_exact(4)
        .map(|v| u32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    spirv_tools::val::create(Some(spirv_tools::TargetEnv::Vulkan_1_1))
        .validate(&words, None)
        .map_err(|error| format!("invalid shader: {error}"))?;
    layout_contract(&words)?;
    Ok(words)
}

fn layout_contract(words: &[u32]) -> Result<()> {
    let mut sets = BTreeMap::new();
    let mut bindings = BTreeMap::new();
    let mut variables = BTreeMap::new();
    let mut pointers = BTreeMap::new();
    let mut structures = Vec::new();
    let mut entries = Vec::new();
    let mut sizes = Vec::new();
    let mut at = 5;
    while at < words.len() {
        let length = (words[at] >> 16) as usize;
        if length == 0 || length > words.len() - at {
            return Err("truncated instruction".into());
        }
        let op = &words[at..at + length];
        match op[0] & 0xffff {
            17 if op.get(1) != Some(&1) => return Err("only Shader capability is supported".into()),
            15 => {
                if op.len() < 5 || op[1] != 5 || op[3..5] != [0x7477656e, 0x6e6f] {
                    return Err("expected only the newton compute entry".into());
                }
                entries.push(op[2]);
            }
            16 if op.len() == 6 && op[2] == 17 && op[3..] == [256, 1, 1] => sizes.push(op[1]),
            16 | 331 => return Err("expected literal 256x1x1 workgroup".into()),
            30 => structures.push(op[1]),
            32 => {
                pointers.insert(op[1], (op[2], op[3]));
            }
            59 if op[3] == 2 || op[3] == 12 => {
                variables.insert(op[2], (op[1], op[3]));
            }
            59 if !matches!(op[3], 1 | 4 | 6 | 7) => {
                return Err("unsupported resource storage class".into());
            }
            71 if op[2] == 33 => {
                bindings.insert(op[1], op[3]);
            }
            71 if op[2] == 34 => {
                sets.insert(op[1], op[3]);
            }
            _ => {}
        }
        at += length;
    }
    if entries.len() != 1
        || entries != sizes
        || bindings.len() != 22
        || sets.len() != 22
        || variables.len() != 22
    {
        return Err("expected the protected condensed Newton interface".into());
    }
    let mut used = std::collections::BTreeSet::new();
    for (id, (pointer, class)) in variables {
        let binding = *bindings.get(&id).ok_or("missing binding")?;
        let &(pointer_class, pointee) = pointers.get(&pointer).ok_or("missing pointer")?;
        if sets.get(&id) != Some(&0)
            || binding >= 22
            || !used.insert(binding)
            || class != if binding == 10 { 2 } else { 12 }
            || pointer_class != class
            || !structures.contains(&pointee)
        {
            return Err("mismatched Newton buffer interface".into());
        }
    }
    Ok(())
}

fn statistic(stat: &vk::PipelineExecutableStatisticKHR<'_>) -> Result<Value> {
    let name = stat.name_as_c_str()?.to_string_lossy();
    let value = unsafe {
        match stat.format {
            vk::PipelineExecutableStatisticFormatKHR::BOOL32 => json!(stat.value.b32 != 0),
            vk::PipelineExecutableStatisticFormatKHR::INT64 => json!(stat.value.i64),
            vk::PipelineExecutableStatisticFormatKHR::UINT64 => json!(stat.value.u64),
            vk::PipelineExecutableStatisticFormatKHR::FLOAT64 => {
                if !stat.value.f64.is_finite() {
                    return Err("non-finite pipeline statistic".into());
                }
                json!(stat.value.f64)
            }
            _ => return Err("unknown pipeline statistic format".into()),
        }
    };
    Ok(json!({
        "name": name,
        "description": stat.description_as_c_str()?.to_string_lossy(),
        "format": stat.format.as_raw(), "value": value,
        "warning": if name == "Local Memory Size" && value.as_u64().is_some_and(|v| v >= 1u64 << 32) {
            Some("suspicious per-thread size; retain raw value, do not infer spilling or truncate high bits")
        } else { None },
    }))
}

fn inspect(context: &Context, path: &Path) -> Result<Value> {
    eprintln!("pipeline-statistics begin {}", path.display());
    let bytes = std::fs::read(path)?;
    let code = words(&bytes)?;
    let device = context.device.as_ref().ok_or("device not initialized")?;
    let mut pipeline = Pipeline {
        device,
        bindings: vk::DescriptorSetLayout::null(),
        layout: vk::PipelineLayout::null(),
        module: vk::ShaderModule::null(),
        pipeline: vk::Pipeline::null(),
    };
    let bindings: Vec<_> = (0..22)
        .map(|binding| {
            vk::DescriptorSetLayoutBinding::default()
                .binding(binding)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .descriptor_type(if binding == 10 {
                    vk::DescriptorType::UNIFORM_BUFFER
                } else {
                    vk::DescriptorType::STORAGE_BUFFER
                })
        })
        .collect();
    unsafe {
        pipeline.bindings = device.create_descriptor_set_layout(
            &vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings),
            None,
        )?;
        pipeline.layout = device.create_pipeline_layout(
            &vk::PipelineLayoutCreateInfo::default().set_layouts(&[pipeline.bindings]),
            None,
        )?;
        pipeline.module = device
            .create_shader_module(&vk::ShaderModuleCreateInfo::default().code(&code), None)?;
        let create = vk::ComputePipelineCreateInfo::default()
            .flags(vk::PipelineCreateFlags::CAPTURE_STATISTICS_KHR)
            .layout(pipeline.layout)
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(pipeline.module)
                    .name(c"newton"),
            );
        let start = std::time::Instant::now();
        match device.create_compute_pipelines(vk::PipelineCache::null(), &[create], None) {
            Ok(created) => pipeline.pipeline = created[0],
            Err((created, error)) => {
                for handle in created {
                    device.destroy_pipeline(handle, None);
                }
                return Err(error.into());
            }
        }
        let creation_ms = start.elapsed().as_secs_f64() * 1000.0;
        eprintln!(
            "pipeline-statistics created {} ms={creation_ms}",
            path.display()
        );
        let extension = pipeline_executable_properties::Device::new(&context.instance, device);
        let properties = extension.get_pipeline_executable_properties(
            &vk::PipelineInfoKHR::default().pipeline(pipeline.pipeline),
        )?;
        if properties.is_empty() {
            return Err("driver returned no executables".into());
        }
        let mut executables = Vec::new();
        for (index, property) in properties.iter().enumerate() {
            let statistics = extension.get_pipeline_executable_statistics(
                &vk::PipelineExecutableInfoKHR::default()
                    .pipeline(pipeline.pipeline)
                    .executable_index(index as u32),
            )?;
            if statistics.is_empty() {
                return Err("driver returned no statistics".into());
            }
            executables.push(json!({
                "name": property.name_as_c_str()?.to_string_lossy(),
                "description": property.description_as_c_str()?.to_string_lossy(),
                "subgroup_size": property.subgroup_size,
                "statistics": statistics.iter().map(statistic).collect::<Result<Vec<_>>>()?,
            }));
        }
        let result = json!({ "path": path, "blake3": blake3::hash(&bytes).to_hex().to_string(),
            "pipeline_creation_ms": creation_ms, "cache_state": "uncontrolled; not a cold compilation benchmark",
            "executables": executables });
        eprintln!(
            "pipeline-statistics result {}",
            serde_json::to_string(&result)?
        );
        Ok(result)
    }
}

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.is_empty() {
        return Err("usage: soil_pipeline_statistics --list | shader.spv [...]".into());
    }
    let entry = unsafe { ash::Entry::load()? };
    let app = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_1);
    let instance = unsafe {
        entry.create_instance(
            &vk::InstanceCreateInfo::default().application_info(&app),
            None,
        )?
    };
    let mut context = Context {
        _entry: entry,
        instance,
        device: None,
    };
    let mut adapters = Vec::new();
    let mut selected = Vec::new();
    unsafe {
        for physical in context.instance.enumerate_physical_devices()? {
            let properties = context.instance.get_physical_device_properties(physical);
            let name = properties.device_name_as_c_str()?.to_string_lossy();
            let extensions = context
                .instance
                .enumerate_device_extension_properties(physical)?;
            let supported = extensions.iter().any(|ext| {
                CStr::from_ptr(ext.extension_name.as_ptr()) == pipeline_executable_properties::NAME
            });
            let mut statistics =
                vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default();
            let float_controls2_extension = extensions.iter().any(|ext| {
                CStr::from_ptr(ext.extension_name.as_ptr()) == shader_float_controls2::NAME
            });
            let mut controls2 = vk::PhysicalDeviceShaderFloatControls2FeaturesKHR::default();
            let mut features = vk::PhysicalDeviceFeatures2::default().push_next(&mut statistics);
            if float_controls2_extension {
                features = features.push_next(&mut controls2);
            }
            context
                .instance
                .get_physical_device_features2(physical, &mut features);
            let mut controls = vk::PhysicalDeviceFloatControlsProperties::default();
            let controls_supported = properties.api_version >= vk::API_VERSION_1_2
                || extensions.iter().any(|ext| {
                    CStr::from_ptr(ext.extension_name.as_ptr())
                        == ash::khr::shader_float_controls::NAME
                });
            if controls_supported {
                context.instance.get_physical_device_properties2(
                    physical,
                    &mut vk::PhysicalDeviceProperties2::default().push_next(&mut controls),
                );
            }
            adapters.push(json!({ "name": name, "vendor_id": properties.vendor_id,
                "device_id": properties.device_id, "driver_version": properties.driver_version,
                "api_version": properties.api_version, "pipeline_executable_extension": supported,
                "pipeline_executable_feature": statistics.pipeline_executable_info != 0,
                "float_controls2_extension": float_controls2_extension,
                "float_controls2_feature": controls2.shader_float_controls2 != 0,
                "float_controls_properties_available": controls_supported,
                "float32_signed_zero_inf_nan_preserve": controls.shader_signed_zero_inf_nan_preserve_float32 != 0,
                "float32_denorm_preserve": controls.shader_denorm_preserve_float32 != 0,
                "float32_rounding_rte": controls.shader_rounding_mode_rte_float32 != 0,
            }));
            if name.contains("RTX 4080") && properties.vendor_id == 0x10de {
                selected.push((
                    physical,
                    supported && statistics.pipeline_executable_info != 0,
                    properties,
                ));
            }
        }
        if args == ["--list"] {
            println!("{}", serde_json::to_string_pretty(&adapters)?);
            return Ok(());
        }
        if selected.len() != 1 {
            return Err("expected exactly one NVIDIA RTX 4080 adapter".into());
        }
        let (physical, supported, properties) = selected[0];
        if !supported {
            return Err("RTX 4080 does not expose pipeline executable statistics".into());
        }
        let family = context
            .instance
            .get_physical_device_queue_family_properties(physical)
            .iter()
            .position(|q| q.queue_count > 0 && q.queue_flags.contains(vk::QueueFlags::COMPUTE))
            .ok_or("no compute queue")? as u32;
        let queues = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(family)
            .queue_priorities(&[1.0])];
        let mut statistics = vk::PhysicalDevicePipelineExecutablePropertiesFeaturesKHR::default()
            .pipeline_executable_info(true);
        let extensions = [pipeline_executable_properties::NAME.as_ptr()];
        let available = context.instance.get_physical_device_features(physical);
        let extension_properties = context
            .instance
            .enumerate_device_extension_properties(physical)?;
        let has_robustness2 = extension_properties
            .iter()
            .any(|ext| CStr::from_ptr(ext.extension_name.as_ptr()) == robustness2::NAME);
        let mut robustness = vk::PhysicalDeviceRobustness2FeaturesEXT::default();
        context.instance.get_physical_device_features2(
            physical,
            &mut vk::PhysicalDeviceFeatures2::default().push_next(&mut robustness),
        );
        let robust_buffer_access2 = has_robustness2 && robustness.robust_buffer_access2 != 0;
        let mut robustness = vk::PhysicalDeviceRobustness2FeaturesEXT::default()
            .robust_buffer_access2(robust_buffer_access2);
        let mut extensions = extensions.to_vec();
        if robust_buffer_access2 {
            extensions.push(robustness2::NAME.as_ptr());
        }
        let features = vk::PhysicalDeviceFeatures::default()
            .robust_buffer_access(available.robust_buffer_access != 0);
        let mut create = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queues)
            .enabled_extension_names(&extensions)
            .enabled_features(&features)
            .push_next(&mut statistics);
        if robust_buffer_access2 {
            create = create.push_next(&mut robustness);
        }
        context.device = Some(context.instance.create_device(physical, &create, None)?);
        let shaders = args
            .iter()
            .map(|arg| inspect(&context, Path::new(arg)))
            .collect::<Result<Vec<_>>>()?;
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "scope": "offline pipeline compilation; no dispatch or runtime timing; capture flags may affect compilation",
                "device": properties.device_name_as_c_str()?.to_string_lossy(),
                "driver_version": properties.driver_version,
                "robust_buffer_access": available.robust_buffer_access != 0,
            "robust_buffer_access2": robust_buffer_access2,
                "adapters": adapters, "shaders": shaders,
            }))?
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_modules() {
        for bytes in [
            vec![],
            vec![0; 19],
            vec![0; 20],
            vec![0; 21],
            vec![0; 8 * 1024 * 1024 + 4],
        ] {
            assert!(words(&bytes).is_err());
        }
    }

    #[test]
    fn typed_statistics_preserve_values() {
        let mut stat = vk::PipelineExecutableStatisticKHR::default();
        stat.format = vk::PipelineExecutableStatisticFormatKHR::UINT64;
        stat.value.u64 = u64::MAX;
        assert_eq!(statistic(&stat).unwrap()["value"], json!(u64::MAX));
        stat.format = vk::PipelineExecutableStatisticFormatKHR::FLOAT64;
        stat.value.f64 = f64::INFINITY;
        assert!(statistic(&stat).is_err());
        stat.format = vk::PipelineExecutableStatisticFormatKHR::from_raw(-1);
        assert!(statistic(&stat).is_err());
        stat = stat.name(c"Local Memory Size").unwrap();
        stat.format = vk::PipelineExecutableStatisticFormatKHR::UINT64;
        stat.value.u64 = 0x1000001d00;
        let result = statistic(&stat).unwrap();
        assert_eq!(result["value"], json!(0x1000001d00_u64));
        assert!(result["warning"].is_string());
    }

    #[test]
    fn rejects_other_compute_interfaces() {
        let module =
            naga::front::wgsl::parse_str("@compute @workgroup_size(256) fn newton() {}").unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        let code = naga::back::spv::write_vec(&module, &info, &Default::default(), None).unwrap();
        let bytes: Vec<_> = code.iter().flat_map(|v| v.to_le_bytes()).collect();
        assert!(words(&bytes).unwrap_err().to_string().contains("interface"));
    }
}
