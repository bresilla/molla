use super::*;
use ash::{khr::shader_float_controls2, vk};
use molla_compute::{AdapterInfo, Engine};
use spirv_tools::val::Validator;
use std::{
    future::Future,
    sync::Arc,
    task::{Context, Poll, Wake, Waker},
};

fn wait<T>(future: impl Future<Output = T>) -> T {
    struct Signal(std::thread::Thread);
    impl Wake for Signal {
        fn wake(self: Arc<Self>) {
            self.0.unpark();
        }
    }
    let waker = Waker::from(Arc::new(Signal(std::thread::current())));
    let mut future = std::pin::pin!(future);
    loop {
        match future.as_mut().poll(&mut Context::from_waker(&waker)) {
            Poll::Ready(value) => return value,
            Poll::Pending => std::thread::park(),
        }
    }
}

pub(crate) fn engine(extra_features: wgpu::Features) -> Engine {
    extension_engine(extra_features, false)
}

pub(crate) fn clock_engine(extra_features: wgpu::Features) -> Engine {
    extension_engine(extra_features, true)
}

fn extension_engine(extra_features: wgpu::Features, clock: bool) -> Engine {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..wgpu::InstanceDescriptor::new_without_display_handle()
    });
    let mut adapters: Vec<_> = wait(instance.enumerate_adapters(wgpu::Backends::VULKAN))
        .into_iter()
        .filter(|a| {
            let info = a.get_info();
            info.vendor == 0x10de && info.name.contains("RTX 4080")
        })
        .collect();
    assert_eq!(adapters.len(), 1, "explicit RTX 4080 fixture");
    let adapter = adapters.pop().unwrap();
    let info = AdapterInfo::from_wgpu(adapter.get_info());
    let features = wgpu::Features::PASSTHROUGH_SHADERS | extra_features;
    assert!(adapter.features().contains(features));
    let limits = adapter.limits();
    let hints = wgpu::MemoryHints::Performance;
    let mut enable =
        vk::PhysicalDeviceShaderFloatControls2FeaturesKHR::default().shader_float_controls2(true);
    let mut enable_clock =
        vk::PhysicalDeviceShaderClockFeaturesKHR::default().shader_subgroup_clock(true);
    let extension = if clock {
        ash::khr::shader_clock::NAME
    } else {
        shader_float_controls2::NAME
    };
    // The queried adapter creates this logical device with the extra feature enabled.
    let opened = unsafe {
        let hal = adapter.as_hal::<wgpu::hal::api::Vulkan>().unwrap();
        let raw = hal.shared_instance().raw_instance();
        let physical = hal.raw_physical_device();
        let extensions = raw.enumerate_device_extension_properties(physical).unwrap();
        assert!(
            extensions
                .iter()
                .any(|e| e.extension_name_as_c_str().unwrap() == extension)
        );
        let mut controls = vk::PhysicalDeviceShaderFloatControls2FeaturesKHR::default();
        let mut clocks = vk::PhysicalDeviceShaderClockFeaturesKHR::default();
        raw.get_physical_device_features2(
            physical,
            &mut vk::PhysicalDeviceFeatures2::default()
                .push_next(&mut controls)
                .push_next(&mut clocks),
        );
        if clock {
            assert_eq!(clocks.shader_subgroup_clock, vk::TRUE);
        } else {
            assert_eq!(controls.shader_float_controls2, vk::TRUE);
        }
        let mut properties = vk::PhysicalDeviceFloatControlsProperties::default();
        raw.get_physical_device_properties2(
            physical,
            &mut vk::PhysicalDeviceProperties2::default().push_next(&mut properties),
        );
        assert_eq!(
            properties.shader_signed_zero_inf_nan_preserve_float32,
            vk::TRUE
        );
        hal.open_with_callback(
            features,
            &limits,
            &hints,
            Some(Box::new(|args| {
                args.extensions.push(extension);
                *args.create_info = if clock {
                    args.create_info.push_next(&mut enable_clock)
                } else {
                    args.create_info.push_next(&mut enable)
                };
            })),
        )
        .unwrap()
    };
    // The HAL device and descriptor use the same adapter, features, limits and ownership.
    let (device, queue) = unsafe {
        adapter
            .create_device_from_hal::<wgpu::hal::api::Vulkan>(
                opened,
                &wgpu::DeviceDescriptor {
                    label: Some("test-only Vulkan extension device"),
                    required_features: features,
                    required_limits: limits,
                    memory_hints: hints,
                    ..Default::default()
                },
            )
            .unwrap()
    };
    let engine = Engine::from_device(Arc::new(device), Arc::new(queue), Some(info));
    let health = engine.device_health().unwrap().clone();
    engine
        .device()
        .unwrap()
        .set_device_lost_callback(move |reason, message| health.record(reason, message));
    engine
}

pub(crate) fn strict(words: &[u32]) -> Vec<u32> {
    let code = instructions(words);
    let floats: Vec<_> = code
        .iter()
        .filter(|(_, op)| op[0] & 0xffff == 22 && op[2] == 32)
        .map(|(_, op)| op[1])
        .collect();
    let uints: Vec<_> = code
        .iter()
        .filter(|(_, op)| op[0] & 0xffff == 21 && op[2..] == [32, 0])
        .map(|(_, op)| op[1])
        .collect();
    let entries: Vec<_> = code
        .iter()
        .filter(|(_, op)| op[0] & 0xffff == 15)
        .map(|(_, op)| op[2])
        .collect();
    assert_eq!(floats.len(), 1);
    assert_eq!(uints.len(), 1);
    assert_eq!(entries.len(), 1);
    assert!(
        !code
            .iter()
            .any(|(_, op)| op[0] & 0xffff == 71 && op[2] == 40)
    );
    assert!(!code.iter().any(|(_, op)| op[0] & 0xffff == 331));
    let extension = b"SPV_KHR_float_controls2\0";
    assert_eq!(extension.len() % 4, 0);
    let zero = words[3];
    let mut out = words[..5].to_vec();
    out[1] = 0x0001_0200;
    out[3] = zero.checked_add(1).unwrap();
    out.extend_from_slice(&[(2 << 16) | 17, 6029]);
    let mut extension_written = false;
    let mut mode_written = false;
    let mut constant_written = false;
    for (_, op) in code {
        let opcode = op[0] & 0xffff;
        if opcode == 71 && op[2] == 42 {
            continue;
        }
        if !extension_written && opcode != 17 {
            out.push(((1 + extension.len() as u32 / 4) << 16) | 10);
            out.extend(
                extension
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap())),
            );
            extension_written = true;
        }
        if !mode_written && matches!(opcode, 3..=8 | 19..=39 | 71 | 72 | 330) {
            out.extend_from_slice(&[(5 << 16) | 331, entries[0], 6028, floats[0], zero]);
            mode_written = true;
        }
        if !constant_written && opcode == 54 {
            out.extend_from_slice(&[(4 << 16) | 43, uints[0], zero, 0]);
            constant_written = true;
        }
        out.extend_from_slice(op);
    }
    assert!(extension_written && mode_written && constant_written);
    spirv_tools::val::create(Some(spirv_tools::TargetEnv::Vulkan_1_1))
        .validate(&out, None)
        .unwrap();
    out
}

pub(crate) fn selective(words: &[u32]) -> Vec<u32> {
    let code = instructions(words);
    let ids = names(words);
    let mut targets = BTreeSet::new();
    let mut existing = BTreeSet::new();
    let mut covered = BTreeSet::new();
    let mut function = 0;
    for (_, op) in &code {
        match op[0] & 0xffff {
            54 => function = op[2],
            56 => function = 0,
            71 if op[2] == 42 => {
                existing.insert(op[1]);
            }
            71 if op[2] == 40 => panic!("pre-existing float controls"),
            331 => panic!("pre-existing default float controls"),
            _ => {}
        }
        if ids
            .get(&function)
            .is_some_and(|name| HELPERS.contains(&name.as_str()) || name == "fischer_root")
            && (matches!(op[0] & 0xffff, 129 | 131 | 133 | 136)
                || (op[0] & 0xffff == 12 && matches!(op[4], 31 | 50)))
        {
            targets.insert(op[2]);
            covered.insert(ids[&function].as_str());
        }
    }
    assert!(existing.is_subset(&targets));
    for helper in ["rate_sum", "rate_add", "rate_mul"] {
        assert!(
            covered.contains(helper),
            "missing protected arithmetic for {helper}"
        );
    }
    let mut out = words[..5].to_vec();
    out[1] = 0x0001_0200;
    out.extend_from_slice(&[(2 << 16) | 17, 6029]);
    let extension = b"SPV_KHR_float_controls2\0";
    let mut extension_written = false;
    let mut annotations_written = false;
    for (_, op) in code {
        let opcode = op[0] & 0xffff;
        if !extension_written && opcode != 17 {
            out.push(((1 + extension.len() as u32 / 4) << 16) | 10);
            out.extend(
                extension
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap())),
            );
            extension_written = true;
        }
        if !annotations_written && matches!(opcode, 71 | 72 | 19) {
            for id in &targets {
                out.extend_from_slice(&[(4 << 16) | 71, *id, 40, 7]);
            }
            annotations_written = true;
        }
        if opcode == 71 && op[2] == 42 {
            continue;
        }
        out.extend_from_slice(op);
    }
    assert!(extension_written && annotations_written);
    spirv_tools::val::create(Some(spirv_tools::TargetEnv::Vulkan_1_1))
        .validate(&out, None)
        .unwrap();
    out
}

fn compile_strict(original: &str, local: bool) -> (String, Vec<u32>) {
    let old = "fn rate_round(v:f32)->f32 {\n    atomicStore(&rate_scratch[rate_lane],bitcast<u32>(v));\n    return bitcast<f32>(atomicAdd(&rate_scratch[rate_lane],0u));\n}";
    assert_eq!(original.matches(old).count(), 1);
    assert!(!original.contains("retained_round"));
    let source = original.replace(old, "fn rate_round(v:f32)->f32 {return v;}");
    let compiled = compile(&source, "main");
    let words = if local {
        selective(&compiled)
    } else {
        strict(&compiled)
    };
    assert!(
        !instructions(&words)
            .iter()
            .any(|(_, op)| matches!(op[0] & 0xffff, 227..=242))
    );
    (source, words)
}

pub(crate) fn shader(device: &wgpu::Device, original: &str, local: bool) -> wgpu::ShaderModule {
    assert!(
        device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS)
    );
    // This fixture creates the device with both the extension and its feature enabled.
    unsafe {
        assert!(
            device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .unwrap()
                .enabled_device_extensions()
                .contains(&shader_float_controls2::NAME)
        );
    }
    let (source, words) = compile_strict(original, local);
    let root = std::path::PathBuf::from(
        std::env::var_os("MOLLA_FLOAT_CONTROLS_ARTIFACTS")
            .expect("explicit float controls artifacts"),
    );
    std::fs::create_dir_all(&root).unwrap();
    let key = blake3::hash(source.as_bytes()).to_hex();
    std::fs::write(root.join(format!("{key}.wgsl")), source).unwrap();
    std::fs::write(
        root.join(format!("{key}.spv")),
        bytemuck::cast_slice(&words),
    )
    .unwrap();
    // Naga checks the source and SPIRV-Tools checks the float-controls module.
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("test-only no-memory rounding"),
            spirv: Some(words.into()),
            ..Default::default()
        })
    }
}

#[test]
fn newton_float_controls_binary_contract() {
    let source = format!(
        "{}\n@group(0) @binding(0) var<storage,read_write> output:array<vec2<f32>>;\n@compute @workgroup_size(64) fn main(@builtin(global_invocation_id) id:vec3<u32>) {{ rate_lane=id.x%64u;output[id.x]=rate_mul(output[id.x].x,rate_add(output[id.x],vec2<f32>(0.25))); }}",
        crate::soil_rate_encoding::sorted_arithmetic(64)
    );
    let (_, words) = compile_strict(&source, false);
    let (_, local) = compile_strict(&source, true);
    assert!(
        !instructions(&local)
            .iter()
            .any(|(_, op)| op[0] & 0xffff == 331)
    );
    assert!(
        std::panic::catch_unwind(|| {
            compile_strict(&source.replace("rate_mul", "unprotected_product"), true);
        })
        .is_err()
    );
    let code = instructions(&words);
    assert_eq!(
        code.iter()
            .filter(|(_, op)| op[0] & 0xffff == 17 && op[1] == 6029)
            .count(),
        1
    );
    assert_eq!(
        code.iter()
            .filter(|(_, op)| op[0] & 0xffff == 331 && op[2] == 6028)
            .count(),
        1
    );
    assert!(
        !code
            .iter()
            .any(|(_, op)| op[0] & 0xffff == 71 && matches!(op[2], 40 | 42))
    );
}

#[test]
#[cfg(feature = "parallel-indices-newton")]
fn newton_float_controls_full_source_contract() {
    use crate::soil_newton_gpu::{NewtonVariant, distributed::mixed, shader_source_for};
    use crate::soil_newton_protected::{FLOAT_CONTROLS_MARKER, compile};
    let original = shader_source_for(NewtonVariant::Fischer);
    let control = mixed::float_controls_program(original.clone(), false);
    let candidate = mixed::float_controls_program(original, true);
    assert_eq!(candidate, format!("{control}\n{FLOAT_CONTROLS_MARKER}"));
    assert!(compile(&format!("{candidate}{FLOAT_CONTROLS_MARKER}"), "newton").is_err());
    for (name, source) in [("control", control), ("candidate", candidate)] {
        let words = compile(&source, "newton").unwrap();
        let code = instructions(&words);
        let ids = names(&words);
        let retained = *ids
            .iter()
            .find(|(_, name)| name.as_str() == "retained_round")
            .unwrap()
            .0;
        let mut function = 0;
        let mut accesses = 0;
        for (_, op) in &code {
            match op[0] & 0xffff {
                54 => function = op[2],
                56 => function = 0,
                61 if function == retained && op.get(4).is_some_and(|mask| mask & 1 != 0) => {
                    accesses += 1
                }
                62 if function == retained && op.get(3).is_some_and(|mask| mask & 1 != 0) => {
                    accesses += 1
                }
                227..=242 if function == retained => accesses += 1,
                _ => {}
            }
        }
        assert_eq!(accesses, if name == "candidate" { 0 } else { 2 });
        if name == "candidate" {
            assert_eq!(
                code.iter()
                    .filter(|(_, op)| op[0] & 0xffff == 331 && op[2] == 6028)
                    .count(),
                1
            );
            assert!(
                !code
                    .iter()
                    .any(|(_, op)| op[0] & 0xffff == 71 && op[2] == 42)
            );
        }
        if let Some(root) = std::env::var_os("MOLLA_FLOAT_CONTROLS_ARTIFACTS") {
            let root = std::path::PathBuf::from(root);
            std::fs::create_dir_all(&root).unwrap();
            std::fs::write(root.join(format!("{name}.wgsl")), source).unwrap();
            std::fs::write(
                root.join(format!("{name}.spv")),
                bytemuck::cast_slice(&words),
            )
            .unwrap();
        }
    }
}

#[test]
#[cfg(feature = "parallel-indices-newton")]
fn newton_float_controls_shared_source_contract() {
    use crate::soil_newton_gpu::{NewtonVariant, distributed::mixed, shader_source_for};
    use crate::soil_newton_protected::{
        FLOAT_CONTROLS_MARKER, SHARED_FLOAT_CONTROLS_MARKER, compile,
    };
    let original = shader_source_for(NewtonVariant::Fischer);
    let source = mixed::float_controls_shared_program(original.clone());
    assert_eq!(
        source,
        format!(
            "{}\n{SHARED_FLOAT_CONTROLS_MARKER}",
            mixed::float_controls_program(original, true)
        )
    );
    assert!(compile(&source.replace(FLOAT_CONTROLS_MARKER, ""), "newton").is_err());
    assert!(compile(&format!("{source}{SHARED_FLOAT_CONTROLS_MARKER}"), "newton").is_err());
    let words = compile(&source, "newton").unwrap();
    let ids = names(&words);
    let retained = *ids
        .iter()
        .find(|(_, name)| name.as_str() == "retained_round")
        .unwrap()
        .0;
    let mut function = 0;
    let mut accesses = Vec::new();
    let mut selections = 0;
    let mut returns = 0;
    for (_, op) in instructions(&words) {
        match op[0] & 0xffff {
            54 => function = op[2],
            56 => function = 0,
            61 if function == retained && op.get(4) == Some(&1) => accesses.push(61),
            62 if function == retained && op.get(3) == Some(&1) => accesses.push(62),
            247 if function == retained => selections += 1,
            254 if function == retained => returns += 1,
            227..=242 => assert_ne!(function, retained),
            _ => {}
        }
    }
    assert_eq!(accesses, [62, 61]);
    assert_eq!(selections, 1);
    assert_eq!(returns, 2);
    if let Some(root) = std::env::var_os("MOLLA_FLOAT_CONTROLS_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("shared.wgsl"), source).unwrap();
        std::fs::write(root.join("shared.spv"), bytemuck::cast_slice(&words)).unwrap();
    }
}

#[test]
#[cfg(feature = "parallel-indices-newton")]
fn newton_float_controls_selective_source_contract() {
    use crate::soil_newton_gpu::{NewtonVariant, distributed::mixed, shader_source_for};
    use crate::soil_newton_protected::{
        FLOAT_CONTROLS_MARKER, SELECTIVE_FLOAT_CONTROLS_MARKER, compile,
    };
    let original = shader_source_for(NewtonVariant::Fischer);
    let control = compile(
        &mixed::float_controls_program(original.clone(), false),
        "newton",
    )
    .unwrap();
    let local = selective(&control);
    let unchanged = |words: &[u32]| {
        instructions(words)
            .into_iter()
            .filter(|(_, op)| {
                !(op[0] & 0xffff == 71 && matches!(op[2], 40 | 42)
                    || op[0] & 0xffff == 17 && op[1] == 6029
                    || op[0] & 0xffff == 10)
            })
            .flat_map(|(_, op)| op.iter().copied())
            .collect::<Vec<_>>()
    };
    assert_eq!(unchanged(&control), unchanged(&local));
    assert_eq!(control[3], local[3]);
    let source = format!(
        "{}\n{SELECTIVE_FLOAT_CONTROLS_MARKER}",
        mixed::float_controls_shared_program(original)
    );
    assert!(compile(&source.replace(FLOAT_CONTROLS_MARKER, ""), "newton").is_err());
    assert!(
        compile(
            &format!("{source}{SELECTIVE_FLOAT_CONTROLS_MARKER}"),
            "newton"
        )
        .is_err()
    );
    let words = compile(&source, "newton").unwrap();
    let code = instructions(&words);
    assert!(!code.iter().any(|(_, op)| op[0] & 0xffff == 331));
    assert!(
        code.iter()
            .any(|(_, op)| op[0] & 0xffff == 71 && op[2..] == [40, 7])
    );
    assert!(
        !code
            .iter()
            .any(|(_, op)| op[0] & 0xffff == 71 && op[2] == 42)
    );
    if let Some(root) = std::env::var_os("MOLLA_FLOAT_CONTROLS_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("selective.wgsl"), source).unwrap();
        std::fs::write(root.join("selective.spv"), bytemuck::cast_slice(&words)).unwrap();
    }
}

#[test]
#[cfg(feature = "parallel-indices-newton")]
fn newton_float_controls_specialized_source_contract() {
    use crate::soil_newton_gpu::{NewtonVariant, distributed::mixed, shader_source_for};
    use crate::soil_newton_protected::{SHARED_FLOAT_CONTROLS_MARKER, compile};
    let original = shader_source_for(NewtonVariant::Fischer);
    let control = mixed::float_controls_specialized_program(original.clone(), false);
    assert_eq!(
        control,
        mixed::float_controls_program(original.clone(), false)
    );
    let candidate = mixed::float_controls_specialized_program(original, true);
    assert!(!candidate.contains(SHARED_FLOAT_CONTROLS_MARKER));
    let words = compile(&candidate, "newton").unwrap();
    let ids = names(&words);
    let retained = *ids
        .iter()
        .find(|(_, name)| name.as_str() == "retained_round")
        .unwrap()
        .0;
    let code = instructions(&words);
    let mut function = 0;
    let mut accesses = 0;
    let mut branches = 0;
    for (_, op) in &code {
        match op[0] & 0xffff {
            54 => function = op[2],
            56 => function = 0,
            61 | 62 | 227..=242 if function == retained => accesses += 1,
            250 if function == retained => branches += 1,
            _ => {}
        }
    }
    assert_eq!((accesses, branches), (0, 0));
    assert!(!code.iter().any(|(_, op)| op[0] & 0xffff == 331));
    assert!(
        !code
            .iter()
            .any(|(_, op)| op[0] & 0xffff == 71 && op[2] == 42)
    );
    let controls: Vec<_> = code
        .iter()
        .filter(|(_, op)| op[0] & 0xffff == 71 && op[2] == 40)
        .collect();
    assert!(!controls.is_empty());
    assert!(controls.iter().all(|(_, op)| op[3] == 7));
    if let Some(root) = std::env::var_os("MOLLA_FLOAT_CONTROLS_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("specialized.wgsl"), candidate).unwrap();
        std::fs::write(root.join("specialized.spv"), bytemuck::cast_slice(&words)).unwrap();
    }
}
