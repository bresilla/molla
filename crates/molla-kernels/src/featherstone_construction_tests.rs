use super::*;
use std::time::Instant;

pub(super) fn joint_kind_source(source: &str, kinds: &[u32]) -> Result<String> {
    joint_types::source(source, kinds)
}

#[test]
fn model_joint_kind_source_preserves_other_inputs_and_validates() {
    let source = super::rates_tests::shader_source();
    for kinds in [&[4][..], &[4, 1, 1, 1, 1, 1], &[0, 1, 2, 3, 4, 5, 6, 7]] {
        let generated = joint_kind_source(&source, kinds).unwrap();
        assert!(!generated.contains("joint_type["));
        let mut restored = generated
            .split("\nfn model_joint_kind")
            .next()
            .unwrap()
            .to_owned();
        while let Some(start) = restored.find("model_joint_kind(") {
            let argument = start + "model_joint_kind(".len();
            let end = restored[argument..].find(')').unwrap() + argument;
            let replacement = format!("joint_type[{}]", &restored[argument..end]);
            restored.replace_range(start..=end, &replacement);
        }
        assert_eq!(restored, source);
        assert_eq!(
            generated.matches("model_joint_kind(").count(),
            source.matches("joint_type[").count() + 1
        );
        let module = naga::front::wgsl::parse_str(&generated).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        for field in [
            "joint_enabled[",
            "joint_limit_lower[",
            "joint_limit_upper[",
            "joint_compliance[",
            "joint_velocity_limit[",
        ] {
            assert_eq!(
                generated.matches(field).count(),
                source.matches(field).count()
            );
        }
    }
    for kinds in [vec![], vec![8], vec![4; 33]] {
        assert!(joint_kind_source(&source, &kinds).is_err());
    }
    assert!(joint_kind_source("joint_type[j+1]", &[4]).is_err());
    assert!(joint_kind_source("joint_type[j", &[4]).is_err());
    assert!(joint_kind_source("no load", &[4]).is_err());
}

#[test]
#[ignore = "requires real GPU; model-vocabulary construction diagnostic"]
fn six_joint_model_vocabulary_pipeline_construction() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let start = Instant::now();
    let _program = FeatherstoneProgram::with_precision(
        engine.device().unwrap(),
        6,
        1,
        false,
        false,
        false,
        FeatherstonePasses::TractorJointKindsOnly,
    );
    eprintln!(
        "six-joint model vocabulary pipeline ready {:.6}s; construction only",
        start.elapsed().as_secs_f64()
    );
}

fn dont_unroll(words: &[u32]) -> std::result::Result<(Vec<u32>, usize), &'static str> {
    if words.len() < 5 || words[0] != 0x07230203 {
        return Err("invalid SPIR-V header");
    }
    let mut output = words.to_vec();
    let mut cursor = 5;
    let mut loops = 0;
    while cursor < words.len() {
        let count = (words[cursor] >> 16) as usize;
        if count == 0 || count > words.len() - cursor {
            return Err("invalid instruction length");
        }
        if words[cursor] & 0xffff == 246 {
            if count != 4 || words[cursor + 3] != 0 {
                return Err("unexpected loop control operands");
            }
            output[cursor + 3] = 2;
            loops += 1;
        }
        cursor += count;
    }
    if loops == 0 {
        return Err("no loop to annotate");
    }
    Ok((output, loops))
}

pub(super) fn loop_hint_shader(
    device: &wgpu::Device,
    source: &str,
    hint: bool,
) -> wgpu::ShaderModule {
    assert!(
        device
            .features()
            .contains(wgpu::Features::PASSTHROUGH_SHADERS)
    );
    let source = no_contraction_tests::control_source(source);
    let original = no_contraction_tests::compile(&source, "featherstone_step");
    let (candidate, loops) = dont_unroll(&original).unwrap();
    no_contraction_tests::check_rounding_barriers(&original).unwrap();
    no_contraction_tests::check_rounding_barriers(&candidate).unwrap();
    assert_eq!(
        original
            .iter()
            .zip(&candidate)
            .filter(|(a, b)| a != b)
            .count(),
        loops
    );
    let directory = std::path::PathBuf::from(
        std::env::var_os("SOIL_LOOP_HINT_OUTPUT").expect("explicit new output directory"),
    );
    std::fs::create_dir(&directory).unwrap();
    std::fs::write(directory.join("shader.wgsl"), source).unwrap();
    for (name, words) in [("original.spv", &original), ("dont-unroll.spv", &candidate)] {
        let path = directory.join(name);
        std::fs::write(
            &path,
            words
                .iter()
                .flat_map(|w| w.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        let validation = std::process::Command::new(
            std::env::var_os("SPIRV_VAL").expect("SPIRV-Tools validator"),
        )
        .args(["--target-env", "vulkan1.1"])
        .arg(&path)
        .output()
        .unwrap();
        std::fs::write(
            directory.join(format!("{name}.validation.txt")),
            &validation.stderr,
        )
        .unwrap();
        assert!(
            validation.status.success(),
            "{}",
            String::from_utf8_lossy(&validation.stderr)
        );
    }
    eprintln!(
        "loop hint candidate: {loops} LoopMerge control words, all other SPIR-V words and atomic rounding barriers retained, hint={hint}"
    );
    // Bounds-checked Naga output and both binaries are SPIRV-Tools validated.
    unsafe {
        device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
            label: Some("test loop-preservation hint"),
            spirv: Some(if hint { candidate } else { original }.into()),
            ..Default::default()
        })
    }
}

#[test]
fn loop_hint_changes_only_empty_loop_control_words_and_rejects_conflicts() {
    let header = [0x07230203, 0x10000, 0, 8, 0];
    let mut input = header.to_vec();
    input.extend([(4 << 16) | 246, 1, 2, 0, (2 << 16) | 249, 3]);
    let (output, count) = dont_unroll(&input).unwrap();
    assert_eq!(count, 1);
    assert_eq!(output.len(), input.len());
    for i in 0..input.len() {
        assert_eq!(output[i], if i == 8 { 2 } else { input[i] });
    }
    for mask in [1, 2, 4, 0x80, 0x100] {
        let mut bad = input.clone();
        bad[8] = mask;
        assert!(dont_unroll(&bad).is_err());
    }
    for bad in [
        vec![],
        header.to_vec(),
        vec![0; 5],
        [header.as_slice(), &[0]].concat(),
        input[..8].to_vec(),
    ] {
        assert!(dont_unroll(&bad).is_err());
    }
}

#[test]
#[ignore = "requires Vulkan passthrough and SPIRV-Tools; construction diagnostic"]
fn six_joint_loop_hint_pipeline_construction() {
    let hint = match std::env::var("SOIL_LOOP_HINT_ARM").unwrap().as_str() {
        "control" => false,
        "hint" => true,
        _ => panic!("explicit control or hint arm required"),
    };
    let engine =
        molla_compute::Engine::new_gpu_with_features(wgpu::Features::PASSTHROUGH_SHADERS).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let started = Instant::now();
    let _program = FeatherstoneProgram::with_precision(
        engine.device().unwrap(),
        6,
        1,
        false,
        false,
        false,
        FeatherstonePasses::LoopHintOnly(hint),
    );
    eprintln!(
        "six-joint loop-hint={hint} program ready in {:.6}s; no physical acceptance",
        started.elapsed().as_secs_f64()
    );
}

pub(super) fn phase_source(source: &str, phase: u32) -> String {
    assert!(phase <= 3);
    assert!(source.contains("uniforms.drift_only"));
    source.replace("uniforms.drift_only", &format!("{phase}u"))
}

#[test]
fn phase_specialization_changes_only_the_runtime_phase_operand() {
    let source = include_str!("../shaders/featherstone.wgsl");
    let pieces: Vec<_> = source.split("uniforms.drift_only").collect();
    assert!(pieces.len() > 10);
    for phase in 0..=3 {
        let generated = phase_source(source, phase);
        assert_eq!(generated, pieces.join(&format!("{phase}u")));
        assert!(!generated.contains("uniforms.drift_only"));
        assert!(generated.contains("drift_only:         u32,"));
        assert!(generated.contains("uniforms.external_scalar_bounds"));
    }
}

#[test]
#[ignore = "requires real GPU; construction diagnostic, not physical acceptance"]
fn six_joint_phase_pipeline_construction() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let started = Instant::now();
    let mut programs = Vec::new();
    for phase in 0..=3 {
        eprintln!(
            "phase pipeline {phase} begin, cumulative {:.6}s",
            started.elapsed().as_secs_f64()
        );
        let phase_start = Instant::now();
        let program = FeatherstoneProgram::with_precision(
            engine.device().unwrap(),
            6,
            1,
            false,
            false,
            false,
            FeatherstonePasses::PhaseOnly(phase),
        );
        eprintln!(
            "phase pipeline {phase} ready, {:.6}s, cumulative {:.6}s",
            phase_start.elapsed().as_secs_f64(),
            started.elapsed().as_secs_f64()
        );
        programs.push(program);
    }
    assert_eq!(programs.len(), 4);
    eprintln!("four six-joint phase pipelines ready; no simulation or physical acceptance claimed");
}

#[test]
#[ignore = "requires real GPU; unrestricted construction diagnostic, not physical acceptance"]
fn six_joint_unrestricted_scalar_pipeline_construction() {
    let engine = molla_compute::Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let started = Instant::now();
    let _program = FeatherstoneProgram::with_precision(
        engine.device().unwrap(),
        6,
        1,
        false,
        false,
        false,
        FeatherstonePasses::UnrestrictedScalarOnly,
    );
    eprintln!(
        "unrestricted scalar six-joint program ready in {:.6}s; no physical acceptance or adoption",
        started.elapsed().as_secs_f64()
    );
}
