use super::*;
use spirv_tools::val::Validator;

pub(crate) const TRACE_BYTES: u64 = 128;
pub(crate) const PHASES: [&str; 8] = [
    "evaluate",
    "prepare_linear",
    "hybrid_direction",
    "certify_candidate",
    "conditioned_release",
    "apply_operator",
    "inner_product",
    "project_rates",
];

const LINEAR_PHASES: [&str; 8] = [
    "hybrid_direction",
    "krylov_cycle",
    "scratch_norm",
    "recover_direction",
    "true_direction_error",
    "precondition",
    "apply_operator",
    "project_rates",
];

pub(crate) fn phase_names() -> [&'static str; 8] {
    if std::env::var_os("MOLLA_NEWTON_LINEAR_CLOCK").is_some() {
        LINEAR_PHASES
    } else {
        PHASES
    }
}

pub(crate) fn phase_source(source: String) -> String {
    phase_source_scope(
        source,
        std::env::var_os("MOLLA_NEWTON_LINEAR_CLOCK").is_some(),
    )
}

fn phase_source_scope(mut source: String, linear: bool) -> String {
    assert!(source.starts_with(crate::soil_newton_protected::PREFIX));
    let phases = if linear {
        [
            ("hybrid_direction", "id:u32", "id", ""),
            (
                "krylov_cycle",
                "id:u32,cycle_limit:u32",
                "id,cycle_limit",
                "",
            ),
            ("scratch_norm", "", "", "f32"),
            ("recover_direction", "count:u32", "count", "f32"),
            ("true_direction_error", "", "", "f32"),
            ("precondition", "c:u32,x:Six", "c,x", "Six"),
            ("apply_operator", "", "", ""),
            ("project_rates", "kind:u32", "kind", ""),
        ]
    } else {
        [
            ("evaluate", "", "", "f32"),
            ("prepare_linear", "", "", ""),
            ("hybrid_direction", "id:u32", "id", ""),
            ("certify_candidate", "id:u32", "id", "bool"),
            ("conditioned_release", "", "", "bool"),
            ("apply_operator", "", "", ""),
            (
                "inner_product",
                "a:u32,ai:u32,b:u32,bi:u32",
                "a,ai,b,bi",
                "f32",
            ),
            ("project_rates", "kind:u32", "kind", ""),
        ]
    };
    for (index, (name, args, call, result)) in phases.into_iter().enumerate() {
        let signature = format!("fn {name}(");
        assert_eq!(source.matches(&signature).count(), 1);
        source = source.replacen(&signature, &format!("fn clock_body_{name}("), 1);
        let return_type = if result.is_empty() {
            String::new()
        } else {
            format!("->{result}")
        };
        let invocation = if result.is_empty() {
            format!("clock_body_{name}({call});")
        } else {
            format!("let result=clock_body_{name}({call});")
        };
        let returned = if result.is_empty() {
            ""
        } else {
            "return result;"
        };
        source.push_str(&format!("\nfn {name}({args}){return_type} {{var start=vec2<u32>(0u);if lane==0u && params.pad2!=1u {{start=soil_clock();}}{invocation}if lane==0u && params.pad2!=1u {{clock_record({index}u,start,soil_clock());}}{returned}}}\n"));
    }
    source.push_str(
        r#"
@group(0) @binding(22) var<storage,read_write> clock_trace:array<vec4<u32>>;
fn soil_clock()->vec2<u32> {return clock_trace[0].xy;}
fn clock_record(index:u32,start:vec2<u32>,end:vec2<u32>) {
    let low=end.x-start.x;
    let high=end.y-start.y-select(0u,1u,end.x<start.x);
    let old=clock_trace[index];let sum=old.x+low;
    clock_trace[index]=vec4<u32>(sum,old.y+high+select(0u,1u,sum<old.x),old.z+1u,0u);
}
"#,
    );
    source
}

#[test]
fn newton_phase_clock_wrappers_compile_and_fit() {
    use crate::soil_newton_gpu::{NewtonVariant, shader_source_for, soa_scratch};
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    assert!(!control.contains("params.pad2"));
    let candidate = phase_source_scope(control.clone(), false);
    assert_eq!(PHASES.len() as u64 * 16, TRACE_BYTES);
    for name in PHASES {
        assert_eq!(
            candidate.matches(&format!("fn clock_body_{name}(")).count(),
            1
        );
        assert_eq!(candidate.matches(&format!("fn {name}(")).count(), 1);
    }
    assert_eq!(
        candidate.matches("storageBarrier()").count(),
        control.matches("storageBarrier()").count()
    );
    assert_eq!(
        candidate.matches("workgroupBarrier()").count(),
        control.matches("workgroupBarrier()").count()
    );
    let words = instrument(&crate::soil_newton_protected::compile(&candidate, "newton").unwrap());
    assert_eq!(
        instructions(&words)
            .iter()
            .filter(|(_, op)| op[0] & 0xffff == 5056)
            .count(),
        1
    );
}

#[test]
fn newton_linear_clock_wrappers_preserve_bodies() {
    use crate::soil_newton_gpu::{NewtonVariant, shader_source_for, soa_scratch};
    let control = soa_scratch::program(shader_source_for(NewtonVariant::Fischer), true);
    let candidate = phase_source_scope(control.clone(), true);
    let mut restored = candidate
        .split_once("\nfn hybrid_direction(")
        .unwrap()
        .0
        .to_owned();
    for name in LINEAR_PHASES {
        assert_eq!(
            candidate.matches(&format!("fn clock_body_{name}(")).count(),
            1
        );
        assert_eq!(candidate.matches(&format!("fn {name}(")).count(), 1);
        restored = restored.replace(&format!("fn clock_body_{name}("), &format!("fn {name}("));
    }
    assert_eq!(restored, control);
    assert_eq!(LINEAR_PHASES.len() as u64 * 16, TRACE_BYTES);
    for marker in ["storageBarrier()", "workgroupBarrier()", "var<workgroup>"] {
        assert_eq!(
            candidate.matches(marker).count(),
            control.matches(marker).count()
        );
    }
    let words = instrument(&crate::soil_newton_protected::compile(&candidate, "newton").unwrap());
    assert_eq!(
        instructions(&words)
            .iter()
            .filter(|(_, op)| op[0] & 0xffff == 5056)
            .count(),
        1
    );
    if let Some(root) = std::env::var_os("MOLLA_NEWTON_LINEAR_CLOCK_ARTIFACTS") {
        let root = std::path::PathBuf::from(root);
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(root.join("control.wgsl"), control).unwrap();
        std::fs::write(root.join("linear-clock.wgsl"), candidate).unwrap();
        std::fs::write(
            root.join("linear-clock.spv"),
            words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    }
}
pub(crate) fn phase_shader(device: &wgpu::Device, source: &str, entry: &str) -> wgpu::ShaderModule {
    unsafe {
        assert!(
            device
                .as_hal::<wgpu::hal::api::Vulkan>()
                .unwrap()
                .enabled_device_extensions()
                .contains(&ash::khr::shader_clock::NAME)
        );
    }
    let words = instrument(&crate::soil_newton_protected::compile(source, entry).unwrap());
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("test-only Newton phase clocks"),
            spirv: Some(words.into()),
            ..Default::default()
        })
    }
}

pub(crate) fn instrument(words: &[u32]) -> Vec<u32> {
    let code = instructions(words);
    let names = names(words);
    let clock: Vec<_> = names
        .iter()
        .filter(|(_, name)| name.as_str() == "soil_clock")
        .collect();
    assert_eq!(clock.len(), 1, "exactly one clock marker function");
    let clock = *clock[0].0;
    let function = code
        .iter()
        .position(|(_, op)| op[0] & 0xffff == 54 && op[2] == clock)
        .unwrap();
    let end = function
        + code[function..]
            .iter()
            .position(|(_, op)| op[0] & 0xffff == 56)
            .unwrap();
    let result_type = code[function].1[1];
    let vector = code
        .iter()
        .find(|(_, op)| op[0] & 0xffff == 23 && op[1] == result_type)
        .unwrap()
        .1;
    assert_eq!(vector[3], 2);
    let uint = vector[2];
    assert!(
        code.iter()
            .any(|(_, op)| op[0] & 0xffff == 21 && op[1..] == [uint, 32, 0])
    );
    assert!(
        !code[function + 1..end]
            .iter()
            .any(|(_, op)| op[0] & 0xffff == 55)
    );
    let scope = words[3];
    let label = scope + 1;
    let result = scope + 2;
    let removed: BTreeSet<_> = code[function + 1..end]
        .iter()
        .filter_map(|(_, op)| match op[0] & 0xffff {
            248 => Some(op[1]),
            59 | 61 | 65 | 80 | 81 | 124 => Some(op[2]),
            _ => None,
        })
        .collect();
    let mut output = words[..5].to_vec();
    output[3] = result + 1;
    output.extend([(2 << 16) | 17, 5055]);
    let mut inserted_extension = false;
    let mut inserted_scope = false;
    let mut i = 0;
    while i < code.len() {
        let op = code[i].1;
        let opcode = op[0] & 0xffff;
        if !inserted_extension && opcode != 17 {
            let mut extension = b"SPV_KHR_shader_clock\0".to_vec();
            while extension.len() % 4 != 0 {
                extension.push(0);
            }
            output.push((((1 + extension.len() / 4) as u32) << 16) | 10);
            output.extend(
                extension
                    .chunks_exact(4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap())),
            );
            inserted_extension = true;
        }
        if !inserted_scope && opcode == 54 {
            output.extend([(4 << 16) | 43, uint, scope, 3]);
            inserted_scope = true;
        }
        if i == function {
            output.extend_from_slice(op);
            output.extend([
                (2 << 16) | 248,
                label,
                (4 << 16) | 5056,
                result_type,
                result,
                scope,
                (2 << 16) | 254,
                result,
                (1 << 16) | 56,
            ]);
            i = end + 1;
            continue;
        }
        if !matches!(opcode, 5 | 71) || !removed.contains(&op[1]) {
            output.extend_from_slice(op);
        }
        i += 1;
    }
    spirv_tools::val::create(Some(spirv_tools::TargetEnv::Vulkan_1_1))
        .validate(&output, None)
        .unwrap();
    output
}

const PROBE: &str = r#"
@group(0) @binding(0) var<storage,read_write> output:array<vec2<u32>>;
fn soil_clock()->vec2<u32> { return output[0]; }
@compute @workgroup_size(32)
fn probe(@builtin(local_invocation_index) lane:u32) {
    if lane==0u {
        output[0]=soil_clock();
        var value=1u;
        for(var i=0u;i<4096u;i++) {value=(value^i)*1664525u+1013904223u;}
        output[2]=vec2<u32>(value,4096u);
        output[1]=soil_clock();
    }
}
"#;

#[test]
fn newton_shader_clock_binary_contract() {
    let words = instrument(&compile(PROBE, "probe"));
    assert_eq!(
        instructions(&words)
            .iter()
            .filter(|(_, op)| op[0] & 0xffff == 5056)
            .count(),
        1
    );
    assert_eq!(
        instructions(&words)
            .iter()
            .filter(|(_, op)| op[0] & 0xffff == 17 && op[1] == 5055)
            .count(),
        1
    );
}

#[test]
#[ignore = "requires RTX 4080 VK_KHR_shader_clock"]
fn newton_shader_clock_gpu_probe() {
    let words = instrument(&compile(PROBE, "probe"));
    let engine = float_controls::clock_engine(wgpu::Features::empty());
    let device = engine.device().unwrap();
    let module = unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("test-only subgroup clock"),
            spirv: Some(words.into()),
            ..Default::default()
        })
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("clock probe"),
        layout: Some(&pipeline_layout),
        module: &module,
        entry_point: Some("probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("clock probe output"),
        size: 24,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: buffer.as_entire_binding(),
        }],
    });
    let mut samples = Vec::new();
    for _ in 0..3 {
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(1, 1, 1);
        }
        engine.queue().unwrap().submit([encoder.finish()]);
        let bytes = crate::soil_shared_wall_friction_gpu::capture::read(&engine, &buffer);
        let words: &[u32] = bytemuck::cast_slice(&bytes);
        let start = u64::from(words[0]) | (u64::from(words[1]) << 32);
        let end = u64::from(words[2]) | (u64::from(words[3]) << 32);
        let delta = end.wrapping_sub(start);
        let expected = (0u32..4096).fold(1u32, |v, i| {
            (v ^ i).wrapping_mul(1664525).wrapping_add(1013904223)
        });
        assert_eq!(&words[4..], &[expected, 4096]);
        assert!(delta > 0 && delta < (1u64 << 63));
        samples.push(serde_json::json!({"start": start, "end": end, "delta_ticks": delta}));
    }
    let output = std::path::PathBuf::from(std::env::var_os("MOLLA_SHADER_CLOCK_RESULT").unwrap());
    assert!(!output.exists());
    std::fs::write(output, serde_json::to_vec_pretty(&serde_json::json!({
        "acceptance": false, "scope": "subgroup ticks, not nanoseconds or Newton phase attribution", "samples": samples
    })).unwrap()).unwrap();
}
