use super::*;
use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Mat3, Quat, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder};

#[test]
fn single_lane_dynamics_changes_only_dispatch_width() {
    let source = include_str!("../shaders/featherstone.wgsl");
    let passes = FeatherstonePasses::GenericAndSingleLaneDynamics;
    let candidate = passes.dynamics_source(source).unwrap();
    assert_eq!(
        candidate.replace("@workgroup_size(1)", "@workgroup_size(64)"),
        source
    );
    assert_eq!(passes.dynamics_width(), 1);
    assert_eq!(FeatherstonePasses::Both.dynamics_width(), 64);
    assert!(!passes.shared_dynamics());
    assert!(!passes.constraint_scratch());
}

#[test]
#[ignore = "requires GPU and two cold six-body programs"]
fn single_lane_private_dynamics_matches_single_tractor() {
    articulation_comparison_with_program(
        false,
        false,
        false,
        true,
        false,
        false,
        &[0.0],
        ConstraintScratch::Off,
        None,
        Some(FeatherstonePasses::GenericAndSingleLaneDynamics),
    );
}

#[test]
#[ignore = "requires Vulkan passthrough, SPIRV-Tools and cold machine programs"]
fn selective_no_contraction_preserves_mutating_constraints() {
    articulation_comparison_with_program(
        false,
        false,
        false,
        true,
        true,
        false,
        &[0.0, 1024.0, 65536.0],
        ConstraintScratch::Off,
        None,
        Some(FeatherstonePasses::GenericAndNoContractionDynamics),
    );
}

#[test]
#[ignore = "requires Vulkan passthrough, SPIRV-Tools and cold machine programs"]
fn intermediate_rounding_preserves_mutating_constraints() {
    articulation_comparison_with_program(
        false,
        false,
        false,
        true,
        true,
        false,
        &[0.0, 1024.0, 65536.0],
        ConstraintScratch::Off,
        None,
        Some(FeatherstonePasses::GenericAndIntermediateRounding),
    );
}

#[test]
#[ignore = "requires Vulkan passthrough, SPIRV-Tools and cold machine programs"]
fn residual_rounding_preserves_mutating_constraints() {
    articulation_comparison_with_program(
        false,
        false,
        false,
        true,
        true,
        false,
        &[0.0, 1024.0, 65536.0],
        ConstraintScratch::Off,
        None,
        Some(FeatherstonePasses::GenericAndResidualRounding),
    );
}

#[test]
#[ignore = "requires Vulkan passthrough, SPIRV-Tools and cold machine programs"]
fn passthrough_control_preserves_mutating_constraints() {
    articulation_comparison_with_program(
        false,
        false,
        false,
        true,
        true,
        false,
        &[0.0, 1024.0, 65536.0],
        ConstraintScratch::Off,
        None,
        Some(FeatherstonePasses::GenericAndPassthroughControl),
    );
}

#[test]
#[ignore = "requires GPU and two cold constrained machine programs"]
fn single_lane_private_dynamics_preserves_mutating_constraints() {
    articulation_comparison_with_program(
        false,
        false,
        false,
        true,
        true,
        false,
        &[0.0, 1024.0, 65536.0],
        ConstraintScratch::Off,
        None,
        Some(FeatherstonePasses::GenericAndSingleLaneDynamics),
    );
}

#[test]
fn single_lane_execution_preserves_bounds_and_source() {
    let execution = FeatherstoneExecution::ExperimentalSingleLane;
    let mut limits = wgpu::Limits::default();
    limits.max_compute_workgroup_storage_size = 256;
    limits.max_compute_workgroups_per_dimension = 3;
    assert!(validate_execution(execution, 32, 3, &limits).is_ok());
    assert!(validate_execution(execution, 33, 3, &limits).is_err());
    assert!(validate_execution(execution, 32, 4, &limits).is_err());
    limits.max_compute_workgroup_storage_size = 255;
    assert!(validate_execution(execution, 1, 1, &limits).is_err());
    let source = include_str!("../shaders/featherstone.wgsl");
    assert_eq!(
        FeatherstonePasses::BothSingleLaneDynamics.dynamics_source(source),
        FeatherstonePasses::GenericAndSingleLaneDynamics.dynamics_source(source)
    );
    assert_eq!(
        FeatherstonePasses::BothSingleLaneDynamics.dynamics_width(),
        1
    );
    assert_ne!(execution.label(), FeatherstoneExecution::Generic.label());
}

#[test]
fn workgroup_execution_preflight_checks_storage_and_dispatch_limits() {
    let mut limits = wgpu::Limits::default();
    let candidate = FeatherstoneExecution::ExperimentalWorkgroupScratch;
    assert_eq!(
        FeatherstoneExecution::default(),
        FeatherstoneExecution::Generic
    );
    limits.max_compute_workgroup_storage_size = 2352;
    limits.max_compute_workgroups_per_dimension = 3;
    assert!(validate_execution(candidate, 6, 3, &limits).is_ok());
    assert!(validate_execution(candidate, 6, 4, &limits).is_err());
    limits.max_compute_workgroup_storage_size -= 1;
    assert!(validate_execution(candidate, 6, 3, &limits).is_err());
    assert!(validate_execution(FeatherstoneExecution::Generic, 6, 3, &limits).is_ok());
    limits.max_compute_workgroup_storage_size = 11296;
    assert!(validate_execution(candidate, 32, 3, &limits).is_ok());
    assert!(validate_execution(candidate, u32::MAX, 3, &limits).is_err());
    limits.max_compute_workgroup_storage_size = 656;
    assert!(validate_execution(candidate, 1, 3, &limits).is_ok());
    limits.max_compute_workgroup_storage_size -= 1;
    assert!(validate_execution(candidate, 1, 3, &limits).is_err());
}

#[test]
fn constraint_execution_has_distinct_identity_and_aligned_storage_bound() {
    let profile = FeatherstoneExecution::ExperimentalConstraintScratch;
    assert_ne!(
        profile.label(),
        FeatherstoneExecution::ExperimentalWorkgroupScratch.label()
    );
    let mut limits = wgpu::Limits::default();
    for (joints, bytes) in [(1, 1200), (6, 5552), (32, 28320)] {
        limits.max_compute_workgroup_storage_size = bytes;
        assert!(validate_execution(profile, joints, 1, &limits).is_ok());
        limits.max_compute_workgroup_storage_size -= 1;
        assert!(validate_execution(profile, joints, 1, &limits).is_err());
    }
    limits.max_compute_workgroup_storage_size = 28320;
    assert!(validate_execution(profile, u32::MAX, 1, &limits).is_err());
    assert!(
        validate_execution(
            profile,
            6,
            limits.max_compute_workgroups_per_dimension as usize + 1,
            &limits
        )
        .is_err()
    );
}

#[test]
#[ignore = "requires real GPU and cold single-body program profiles"]
fn workgroup_execution_cache_isolates_profiles_and_mutable_state() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(
        BodyParams::new()
            .with_mass(1.0)
            .with_inertia(Mat3::IDENTITY),
    );
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let make = |profile| {
        FeatherstoneKernel::new_cached_with_options(
            &engine,
            &model,
            FeatherstoneRounding::Storage,
            profile,
        )
        .unwrap()
    };
    let generic = make(FeatherstoneExecution::Generic);
    let scratch = make(FeatherstoneExecution::ExperimentalWorkgroupScratch);
    let extended = make(FeatherstoneExecution::ExperimentalConstraintScratch);
    let single = make(FeatherstoneExecution::ExperimentalSingleLane);
    assert_eq!(single.dynamics_workgroup_size, 1);
    assert_ne!(single.dynamics_pipeline, generic.dynamics_pipeline);
    assert_ne!(single.dynamics_pipeline, scratch.dynamics_pipeline);
    assert_ne!(single.dynamics_pipeline, extended.dynamics_pipeline);
    assert_eq!(generic.dynamics_pipeline, generic.pipeline);
    assert_ne!(scratch.dynamics_pipeline, scratch.pipeline);
    assert_ne!(extended.dynamics_pipeline, extended.pipeline);
    assert_ne!(generic.dynamics_pipeline, scratch.dynamics_pipeline);
    assert_ne!(generic.dynamics_pipeline, extended.dynamics_pipeline);
    assert_ne!(scratch.dynamics_pipeline, extended.dynamics_pipeline);
    for (profile, first) in [
        (FeatherstoneExecution::Generic, generic),
        (FeatherstoneExecution::ExperimentalSingleLane, single),
        (FeatherstoneExecution::ExperimentalWorkgroupScratch, scratch),
        (
            FeatherstoneExecution::ExperimentalConstraintScratch,
            extended,
        ),
    ] {
        let next = make(profile);
        assert_eq!(first.execution(), profile);
        assert_eq!(next.execution(), profile);
        assert_eq!(first.pipeline, next.pipeline);
        assert_eq!(first.dynamics_pipeline, next.dynamics_pipeline);
        assert_eq!(
            first.unrestricted_drift_pipeline,
            next.unrestricted_drift_pipeline
        );
        assert_ne!(first.velocity_delta, next.velocity_delta);
        assert_ne!(first.stop_status, next.stop_status);
    }
    println!(
        "execution cache: three distinct profiles, pipelines reused, mutable buffers isolated"
    );
}

#[test]
#[ignore = "requires real GPU and two cold one-body programs"]
fn shared_rounding_cache_isolates_profiles_and_reuses_programs() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(
        BodyParams::new()
            .with_mass(1.0)
            .with_inertia(Mat3::IDENTITY),
    );
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model
        .upload_to_device(engine.device().unwrap(), engine.queue().unwrap())
        .unwrap();
    let storage = FeatherstoneKernel::new_cached(&engine, &model).unwrap();
    let shared = FeatherstoneKernel::new_cached_with_rounding(
        &engine,
        &model,
        FeatherstoneRounding::ExperimentalShared,
    )
    .unwrap();
    assert_ne!(storage.pipeline, shared.pipeline);
    assert_ne!(
        storage.unrestricted_drift_pipeline,
        shared.unrestricted_drift_pipeline
    );
    for (profile, first) in [
        (FeatherstoneRounding::Storage, storage),
        (FeatherstoneRounding::ExperimentalShared, shared),
    ] {
        let next = FeatherstoneKernel::new_cached_with_rounding(&engine, &model, profile).unwrap();
        assert_eq!(first.pipeline, next.pipeline);
        assert_eq!(
            first.unrestricted_drift_pipeline,
            next.unrestricted_drift_pipeline
        );
        assert_ne!(first.velocity_delta, next.velocity_delta);
        assert_ne!(first.stop_status, next.stop_status);
    }
    println!(
        "rounding cache: two distinct profiles, both pipelines reused, mutable buffers isolated"
    );
}

#[test]
#[ignore = "requires GPU"]
fn local_position_rounding_preserves_two_sum_residue() {
    sum_component(false, false);
}

#[test]
#[ignore = "requires native GPU f64 support"]
fn native_sum_preserves_two_sum_residue() {
    sum_component(true, false);
}

#[test]
#[ignore = "requires GPU"]
fn shared_coordinate_rounding_preserves_two_sum_and_writeback() {
    sum_component(false, true);
}

fn sum_component(native: bool, coordinates: bool) {
    let features = if native {
        wgpu::Features::SHADER_F64
    } else {
        wgpu::Features::empty()
    };
    let engine = Engine::new_gpu_with_features(features).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let cases: Vec<[f32; 2]> = (0..1029)
        .map(|i| match i % 9 {
            0 => [1.0, 2.0_f32.powi(-25)],
            1 => [1e8, 1.0],
            2 => [65536.0, -0.000004],
            3 => [0.8, -0.09553365],
            4 => [-1.0, 2.0_f32.powi(-26)],
            5 => [1e-20, 1e20],
            6 => [-1e20, -1e-20],
            _ => std::array::from_fn(|j| {
                let bits = (i as u32)
                    .wrapping_mul(1664525)
                    .wrapping_add(1013904223 + j as u32 * 37);
                f32::from_bits((bits & 0x807fffff) | ((67 + ((bits >> 10) % 120)) << 23))
            }),
        })
        .collect();
    let buffer = |bytes, usage| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("rounding component"),
            size: bytes,
            usage,
            mapped_at_creation: false,
        })
    };
    let input = buffer(
        cases.len() as u64 * 8,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    );
    let scratch = buffer(
        cases.len() as u64 * 16,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    let output = buffer(
        cases.len() as u64 * 8,
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
    );
    queue.write_buffer(&input, 0, bytemuck::cast_slice(&cases));
    let helpers = include_str!("../shaders/featherstone.wgsl")
        .split("override local_position_rounding")
        .nth(1)
        .unwrap()
        .split("fn position_add")
        .next()
        .unwrap();
    let helpers = if native {
        format!(
            "{}\n{}\n{}",
            helpers.replace("fn position_sum(", "fn storage_position_sum("),
            include_str!("../shaders/featherstone_native_sum.wgsl"),
            "override native_sum: bool = false;\nfn position_sum(child: u32, a: f32, b: f32) -> vec2<f32> { if native_sum { return native_two_sum(a,b); } return storage_position_sum(child,a,b); }"
        )
    } else {
        helpers.to_owned()
    };
    let source = format!(
        "override local_position_rounding{helpers}\n{}",
        r#"
@group(0) @binding(0) var<storage,read> inputs: array<vec2<f32>>;
@group(0) @binding(1) var<storage,read_write> position_error: array<atomic<u32>>;
@group(0) @binding(2) var<storage,read_write> outputs: array<vec2<f32>>;
@compute @workgroup_size(64)
fn evaluate(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    position_rounding_lane=lane;
    if gid.x >= arrayLength(&inputs) { return; }
    outputs[gid.x]=position_sum(gid.x,inputs[gid.x].x,inputs[gid.x].y);
}
"#
    );
    let source = if coordinates {
        let coordinate_helpers = include_str!("../shaders/featherstone.wgsl")
            .split("fn coordinate_round")
            .nth(1)
            .unwrap()
            .split("fn coordinate_advance")
            .next()
            .unwrap();
        format!(
            "fn storage_coordinate_round{coordinate_helpers}\n{}\n{}\n{}",
            source
                .replace("position_error", "coordinate_error")
                .replace(
                    "outputs[gid.x]=position_sum",
                    "outputs[gid.x]=coordinate_sum"
                ),
            include_str!("../shaders/featherstone_shared_coordinate.wgsl")
                .replace("fn coordinate_round(", "fn shared_coordinate_round("),
            "override local_coordinate_rounding: bool = false;\nfn coordinate_round(coord: u32, value: f32) -> f32 { if local_coordinate_rounding { return shared_coordinate_round(coord,value); } return storage_coordinate_round(coord,value); }"
        )
    } else {
        source
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("rounding component"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let mut results = Vec::new();
    for local in [false, true] {
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rounding component"),
            layout: None,
            module: &shader,
            entry_point: Some("evaluate"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &[(
                    if native {
                        "native_sum"
                    } else if coordinates {
                        "local_coordinate_rounding"
                    } else {
                        "local_position_rounding"
                    },
                    if local { 1.0 } else { 0.0 },
                )],
                ..Default::default()
            },
            cache: None,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rounding component"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: scratch.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
            ],
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups((cases.len() as u32).div_ceil(64), 1, 1);
        }
        queue.submit([encoder.finish()]);
        let result = words(device, queue, &output);
        for (i, (values, pair)) in cases.iter().zip(result.chunks_exact(2)).enumerate() {
            let exact = f64::from(values[0]) + f64::from(values[1]);
            let virtual_b = exact - f64::from(values[0]);
            let tail =
                (f64::from(values[0]) - (exact - virtual_b)) + (f64::from(values[1]) - virtual_b);
            let high = exact as f32;
            let low = ((exact - f64::from(high)) + tail) as f32;
            assert_eq!(
                pair,
                &[high.to_bits(), low.to_bits()],
                "local={local}, case={i}"
            );
        }
        if coordinates {
            let stored = words(device, queue, &scratch);
            for (i, pair) in result.chunks_exact(2).enumerate() {
                assert_eq!(
                    stored[i], pair[1],
                    "coordinate residue writeback, local={local}, case={i}"
                );
            }
        }
        results.push(result);
    }
    assert_eq!(results[0], results[1]);
    println!(
        "rounding native={native} coordinates={coordinates}: 1029 two-sum cases across 17 workgroups match the compensated f64 residual oracle bit-for-bit"
    );
}

pub(super) fn words(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<u32> {
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rounding parity readback"),
        size: source.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, source.size());
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    staging.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .unwrap()
        .unwrap();
    let result = bytemuck::cast_slice(&staging.slice(..).get_mapped_range()).to_vec();
    staging.unmap();
    result
}

#[test]
#[ignore = "requires GPU and cold compilation of two Featherstone programs"]
fn local_position_rounding_matches_storage_for_compensated_articulations() {
    articulation_comparison(false, false, false);
}

#[test]
#[ignore = "requires GPU f64 and two cold six-body programs"]
fn native_sum_matches_storage_for_compensated_tractors() {
    articulation_comparison(true, false, false);
}

#[test]
#[ignore = "requires real GPU and cold Featherstone compilation"]
fn unrestricted_drift_matches_generic_compensated_tractors() {
    articulation_comparison(false, true, false);
}

#[test]
#[ignore = "requires real GPU and two cold dynamics programs"]
fn specialized_dynamics_matches_generic_compensated_tractors() {
    articulation_comparison_impl(false, false, false, true, false, false);
}

#[test]
#[ignore = "requires real GPU and two cold dynamics programs"]
fn specialized_dynamics_preserves_mutating_constraints() {
    articulation_comparison_impl(false, false, false, true, true, false);
}

#[test]
#[ignore = "requires real GPU and two cold dynamics programs"]
fn shared_dynamics_scratch_preserves_mutating_constraints() {
    articulation_comparison_impl(false, false, false, false, true, true);
}

#[test]
#[ignore = "requires real GPU and two cold single-tractor programs"]
fn shared_dynamics_scratch_matches_single_free_tractor() {
    articulation_comparison_at(
        false,
        false,
        false,
        false,
        false,
        true,
        &[0.0],
        ConstraintScratch::Off,
        None,
    );
}

#[test]
#[ignore = "requires real GPU and two cold dynamics programs"]
fn constraint_scratch_preserves_translated_mutating_constraints() {
    articulation_comparison_at(
        false,
        false,
        false,
        false,
        true,
        true,
        &[0.0, 1024.0, 65536.0],
        ConstraintScratch::GenericControl,
        None,
    );
}

#[test]
#[ignore = "requires real GPU and two cold single-tractor programs"]
fn constraint_scratch_matches_partial_scratch_single_tractor() {
    articulation_comparison_at(
        false,
        false,
        false,
        false,
        false,
        true,
        &[0.0],
        ConstraintScratch::PartialControl,
        None,
    );
}

#[test]
fn constraint_scratch_preserves_arithmetic_and_scalar_pointer_spaces() {
    let source = concat!(
        include_str!("../shaders/featherstone.wgsl"),
        "\n",
        include_str!("../shaders/featherstone_stops.wgsl"),
        "\n",
        include_str!("../shaders/featherstone_feedback.wgsl"),
        "\n",
        include_str!("../shaders/featherstone_mixed_stops.wgsl")
    );
    let candidate = constraint_scratch_source(source);
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        source.matches("var<workgroup>").count() + 19
    );
    assert_eq!(
        candidate.matches("ptr<function, StopScalars>").count(),
        source.matches("ptr<function, StopScalars>").count()
    );
    let normalize = |source: &str| {
        let mut normalized = source.to_owned();
        for (parameter, kind, global) in [
            ("counts", "StopCounts", "n_dof_arr"),
            ("sl", "StopVectors", "s_lin"),
            ("sa", "StopVectors", "s_ang"),
            ("ul", "StopVectors", "u_lin"),
            ("ua", "StopVectors", "u_ang"),
            ("inverse", "StopBlocks", "d_dense_inv"),
        ] {
            for gap in ["", " "] {
                normalized = normalized.replace(
                    &format!("{parameter}: ptr<function,{gap}{kind}>"),
                    &format!("{parameter}: u32"),
                );
            }
            normalized = normalized
                .replace(&format!("(*{parameter})"), global)
                .replace(&format!("&{global}"), "0u");
        }
        let mut lines: Vec<_> = normalized
            .lines()
            .map(|line| {
                line.trim()
                    .replace("var<workgroup> ", "var ")
                    .replace("ptr<workgroup,", "ptr<function,")
                    .replace("@workgroup_size(1)", "@workgroup_size(64)")
            })
            .filter(|line| !line.is_empty())
            .collect();
        lines.sort();
        lines
    };
    assert_eq!(normalize(source), normalize(&candidate));
    assert!(!candidate.contains("ptr<workgroup,"));
    for kind in ["StopCounts", "StopVectors", "StopBlocks"] {
        assert!(!candidate.contains(&format!("ptr<function, {kind}>")));
    }
}

#[test]
fn shared_dynamics_scratch_changes_only_storage_and_group_size() {
    let source = include_str!("../shaders/featherstone.wgsl");
    let candidate = shared_dynamics_source(source);
    assert!(candidate.contains("@compute @workgroup_size(1)"));
    assert_eq!(
        candidate.matches("uniforms.drift_only").count(),
        source.matches("uniforms.drift_only").count()
    );
    assert_eq!(
        candidate.matches("var<workgroup>").count(),
        source.matches("var<workgroup>").count() + 13
    );
    let normalized = |source: &str| {
        let mut lines: Vec<_> = source
            .lines()
            .map(|line| {
                line.trim()
                    .replace("var<workgroup> ", "var ")
                    .replace("@workgroup_size(1)", "@workgroup_size(64)")
            })
            .filter(|line| !line.is_empty())
            .collect();
        lines.sort();
        lines
    };
    assert_eq!(normalized(source), normalized(&candidate));
}

#[test]
fn dynamics_source_specializes_only_the_dispatch_mode() {
    let source = include_str!("../shaders/featherstone.wgsl");
    let candidate = dynamics_source(source);
    assert!(!candidate.contains("uniforms.drift_only"));
    assert_eq!(candidate, source.replace("uniforms.drift_only", "0u"));
    for preserved in [
        "uniforms.rigorous_d_solve",
        "uniforms.compensated_coordinates",
        "react_mixed_stops(",
        "apply_joint_compliance(",
    ] {
        assert!(candidate.contains(preserved), "{preserved}");
    }
}

#[test]
fn compliance_preflight_rejects_unsupported_joint_and_invalid_offsets() {
    use molla_sim::joint_control::JointCompliance;
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(
        BodyParams::new()
            .with_mass(1.0)
            .with_inertia(Mat3::IDENTITY),
    );
    builder
        .add_joint_d6(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.joint_compliance.host_mut().unwrap()[0] = JointCompliance {
        enabled: 1,
        padding: 0,
        lower: -0.1,
        upper: -0.1,
        natural_frequency: 12.0,
        damping_ratio: 0.7,
    };
    validate_compliance(&model).unwrap();
    model.joint_type.host_mut().unwrap()[0] = molla_sim::JointType::Free as u32;
    assert!(
        validate_compliance(&model)
            .unwrap_err()
            .to_string()
            .contains("six-DOF D6")
    );
    model.joint_compliance.host_mut().unwrap()[0] = JointCompliance::DISABLED;
    validate_compliance(&model).unwrap();
    for (index, value) in [(0, 1), (0, 7), (1, u32::MAX), (1, 0)] {
        let before = model.joint_dof_offset.host().unwrap()[index];
        model.joint_dof_offset.host_mut().unwrap()[index] = value;
        assert!(
            validate_compliance(&model)
                .unwrap_err()
                .to_string()
                .contains("offsets")
        );
        model.joint_dof_offset.host_mut().unwrap()[index] = before;
    }
    validate_compliance(&model).unwrap();
}

#[test]
#[ignore = "requires real GPU and two cold six-body programs"]
fn shared_rounding_matches_compensated_tractors() {
    articulation_comparison(false, false, true);
}

#[test]
#[ignore = "requires real GPU and two corrected-drift programs"]
fn shared_rounding_matches_corrected_drift_and_preserves_delta() {
    articulation_comparison(false, true, true);
}

#[test]
fn unrestricted_drift_eligibility_rechecks_all_bounds() {
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(
        BodyParams::new()
            .with_mass(1.0)
            .with_inertia(Mat3::IDENTITY),
    );
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    assert!(unrestricted_drift(&model).unwrap());
    for field in 0..3 {
        for bad in [0.0, 1.0, f64::NAN] {
            let buffer = match field {
                0 => &mut model.joint_limit_lower,
                1 => &mut model.joint_limit_upper,
                _ => &mut model.joint_velocity_limit,
            };
            let previous = buffer.host().unwrap()[5];
            buffer.host_mut().unwrap()[5] = bad;
            assert!(!unrestricted_drift(&model).unwrap());
            let buffer = match field {
                0 => &mut model.joint_limit_lower,
                1 => &mut model.joint_limit_upper,
                _ => &mut model.joint_velocity_limit,
            };
            buffer.host_mut().unwrap()[5] = previous;
            assert!(unrestricted_drift(&model).unwrap());
        }
    }
}

fn seed_velocity_delta(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    input: &wgpu::Buffer,
    corrected: &wgpu::Buffer,
    delta: &wgpu::Buffer,
) {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("drift delta fixture"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
@group(0) @binding(0) var<storage,read> original: array<f32>;
@group(0) @binding(1) var<storage,read> corrected: array<f32>;
@group(0) @binding(2) var<storage,read_write> delta: array<f32>;
@compute @workgroup_size(64)
fn seed(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x<arrayLength(&delta) { delta[gid.x]=corrected[gid.x]-original[gid.x]; }
}
"#
            .into(),
        ),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("drift delta fixture"),
        layout: None,
        module: &shader,
        entry_point: Some("seed"),
        compilation_options: Default::default(),
        cache: None,
    });
    let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("drift delta fixture"),
        layout: &pipeline.get_bind_group_layout(0),
        entries: &[input, corrected, delta]
            .into_iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect::<Vec<_>>(),
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &group, &[]);
        pass.dispatch_workgroups((delta.size() / 4).div_ceil(64) as u32, 1, 1);
    }
    queue.submit([encoder.finish()]);
}

fn articulation_comparison(native: bool, drift_specialization: bool, shared_coordinates: bool) {
    articulation_comparison_impl(
        native,
        drift_specialization,
        shared_coordinates,
        false,
        false,
        false,
    );
}

fn articulation_comparison_impl(
    native: bool,
    drift_specialization: bool,
    shared_coordinates: bool,
    dynamics_specialization: bool,
    mutate_constraints: bool,
    shared_scratch: bool,
) {
    articulation_comparison_at(
        native,
        drift_specialization,
        shared_coordinates,
        dynamics_specialization,
        mutate_constraints,
        shared_scratch,
        &[0.0, 1024.0, 65536.0],
        ConstraintScratch::Off,
        None,
    );
}

#[derive(Clone, Copy)]
enum ConstraintScratch {
    Off,
    GenericControl,
    PartialControl,
}

pub(super) fn zero_rotation_articulation_comparison(drift: bool, constraints: bool) {
    articulation_comparison_at(
        false,
        drift,
        drift,
        false,
        constraints,
        false,
        &[0.0, 1024.0, 65536.0],
        ConstraintScratch::Off,
        Some(if drift {
            FeatherstonePasses::UnrestrictedDriftWithoutZeroRotation
        } else {
            FeatherstonePasses::GenericWithoutZeroRotation
        }),
    );
}

pub(super) fn xyz_rotation_articulation_comparison(drift: bool) {
    articulation_comparison_at(
        false,
        drift,
        drift,
        false,
        false,
        false,
        &[0.0, 1024.0, 65536.0],
        ConstraintScratch::Off,
        Some(if drift {
            FeatherstonePasses::UnrestrictedDriftWithPositionProduct
        } else {
            FeatherstonePasses::GenericWithPositionProduct
        }),
    );
}

fn articulation_comparison_at(
    native: bool,
    drift_specialization: bool,
    shared_coordinates: bool,
    dynamics_specialization: bool,
    mutate_constraints: bool,
    shared_scratch: bool,
    translations: &[f64],
    constraint_scratch: ConstraintScratch,
    rotation_reference: Option<FeatherstonePasses>,
) {
    articulation_comparison_with_program(
        native,
        drift_specialization,
        shared_coordinates,
        dynamics_specialization,
        mutate_constraints,
        shared_scratch,
        translations,
        constraint_scratch,
        rotation_reference,
        None,
    );
}

fn articulation_comparison_with_program(
    native: bool,
    drift_specialization: bool,
    shared_coordinates: bool,
    dynamics_specialization: bool,
    mutate_constraints: bool,
    shared_scratch: bool,
    translations: &[f64],
    constraint_scratch: ConstraintScratch,
    rotation_reference: Option<FeatherstonePasses>,
    dynamics_program: Option<FeatherstonePasses>,
) {
    let zero_rotation = rotation_reference.is_some();
    let xyz_product = matches!(
        rotation_reference,
        Some(
            FeatherstonePasses::GenericWithPositionProduct
                | FeatherstonePasses::UnrestrictedDriftWithPositionProduct
        )
    );
    let partial_reference = matches!(constraint_scratch, ConstraintScratch::PartialControl);
    let constraint_scratch = !matches!(constraint_scratch, ConstraintScratch::Off);
    assert!(!translations.is_empty());
    let drift_only = drift_specialization && shared_coordinates;
    let features = wgpu::Features::TIMESTAMP_QUERY
        | if dynamics_program.is_some_and(FeatherstonePasses::uses_passthrough) {
            wgpu::Features::PASSTHROUGH_SHADERS
        } else {
            wgpu::Features::empty()
        }
        | if native {
            wgpu::Features::SHADER_F64
        } else {
            wgpu::Features::empty()
        };
    let engine = Engine::new_gpu_with_features(features).unwrap();
    if dynamics_program.is_some_and(FeatherstonePasses::uses_passthrough) {
        assert_eq!(engine.adapter_info().unwrap().backend, "Vulkan");
    }
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    for &offset in translations {
        builder.begin_articulation();
        if native
            || drift_specialization
            || shared_coordinates
            || dynamics_specialization
            || shared_scratch
            || zero_rotation
        {
            let box_inertia = |mass: f64, h: Vec3| {
                Mat3::from_diagonal(
                    Vec3::new(
                        h.y * h.y + h.z * h.z,
                        h.x * h.x + h.z * h.z,
                        h.x * h.x + h.y * h.y,
                    ) * (mass / 3.0),
                )
            };
            let root = builder.add_body(
                BodyParams::new()
                    .with_mass(700.0)
                    .with_inertia(box_inertia(700.0, Vec3::new(0.95, 0.22, 0.42))),
            );
            let root_frame = Transform::from_translation(Vec3::new(offset, 0.95, 0.0));
            if mutate_constraints {
                builder
                    .add_joint_d6(BodyId::NONE, root, root_frame, Transform::IDENTITY)
                    .unwrap();
            } else {
                builder
                    .add_joint_free(BodyId::NONE, root, root_frame, Transform::IDENTITY)
                    .unwrap();
            }
            let rotation = Quat::from_rotation_x(std::f64::consts::FRAC_PI_2);
            let radial = 35.0 * (3.0 * 0.45_f64.powi(2) + 0.28_f64.powi(2)) / 12.0;
            for (x, z) in [(-0.75, -0.65), (-0.75, 0.65), (0.75, -0.65), (0.75, 0.65)] {
                let wheel = builder.add_body(BodyParams::new().with_mass(35.0).with_inertia(
                    Mat3::from_diagonal(Vec3::new(radial, 35.0 * 0.45_f64.powi(2) / 2.0, radial)),
                ));
                builder
                    .add_joint_revolute(
                        root,
                        wheel,
                        Transform::from_translation(Vec3::new(x, -0.55, z)),
                        Transform::new(Vec3::ZERO, rotation.inverse()),
                        Vec3::Z,
                    )
                    .unwrap();
            }
            let tine = builder.add_body(
                BodyParams::new()
                    .with_mass(20.0)
                    .with_inertia(box_inertia(20.0, Vec3::new(0.1, 0.35, 0.1))),
            );
            builder
                .add_joint_fixed(
                    root,
                    tine,
                    Transform::from_translation(Vec3::new(-1.45, -0.72, 0.0)),
                    Transform::IDENTITY,
                )
                .unwrap();
            continue;
        }
        let root = builder.add_body(
            BodyParams::new()
                .with_mass(10.0)
                .with_inertia(Mat3::IDENTITY),
        );
        let child = builder.add_body(
            BodyParams::new()
                .with_mass(2.0)
                .with_inertia(Mat3::IDENTITY * 0.2),
        );
        builder
            .add_joint_free(
                BodyId::NONE,
                root,
                Transform::from_translation(Vec3::new(offset, 0.5, 0.0)),
                Transform::IDENTITY,
            )
            .unwrap();
        builder
            .add_joint_revolute(
                root,
                child,
                Transform::new(Vec3::new(0.8, -0.2, 0.1), Quat::from_rotation_y(0.3)),
                Transform::from_translation(Vec3::new(0.1, 0.05, 0.0)),
                Vec3::Z,
            )
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    let mut control = model.control();
    for (d, force) in control
        .joint_force
        .host_mut()
        .unwrap()
        .iter_mut()
        .enumerate()
    {
        *force = 0.1 * (d + 1) as f64;
    }
    let mut initial = model.state().unwrap();
    for (d, velocity) in initial.joint_qd.host_mut().unwrap().iter_mut().enumerate() {
        *velocity = 1e-3 * (d + 1) as f64;
    }
    model.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    let (offsets, mixed) = kernel_configuration(&model);
    let shared_program = (dynamics_specialization || shared_scratch)
        .then(|| {
            FeatherstoneProgram::with_precision(
                device,
                articulation_scratch_capacity(&offsets),
                mixed,
                false,
                false,
                false,
                if let Some(program) = dynamics_program {
                    program
                } else if constraint_scratch {
                    FeatherstonePasses::GenericAndConstraintScratch
                } else if shared_scratch {
                    FeatherstonePasses::GenericAndSharedDynamics
                } else {
                    FeatherstonePasses::GenericAndDynamics
                },
            )
        })
        .or_else(|| {
            (drift_specialization && !shared_coordinates).then(|| {
                eprintln!("compiling shared generic and unrestricted drift programs");
                FeatherstoneProgram::new(device, articulation_scratch_capacity(&offsets), mixed)
            })
        });
    let control_program = partial_reference.then(|| {
        FeatherstoneProgram::with_precision(
            device,
            articulation_scratch_capacity(&offsets),
            mixed,
            false,
            false,
            false,
            FeatherstonePasses::GenericAndSharedDynamics,
        )
    });
    let mut runs = Vec::new();
    for local in [false, true] {
        eprintln!(
            "compiling rounding variant candidate={local}, native={native}, bodies={}",
            model.body_count
        );
        let owned_program = shared_program.is_none().then(|| {
            FeatherstoneProgram::with_precision(
                device,
                articulation_scratch_capacity(&offsets),
                mixed,
                local && !native && !zero_rotation,
                local && native,
                local && shared_coordinates && !zero_rotation,
                if xyz_product && local || zero_rotation && !xyz_product && !local {
                    rotation_reference.unwrap()
                } else if drift_only {
                    FeatherstonePasses::UnrestrictedDriftOnly
                } else {
                    FeatherstonePasses::GenericOnly
                },
            )
        });
        let program = if !local && partial_reference {
            control_program.as_ref().unwrap()
        } else {
            shared_program.as_ref().or(owned_program.as_ref()).unwrap()
        };
        let mut kernel = FeatherstoneKernel::from_program(
            device,
            queue,
            &model,
            offsets.clone(),
            mixed,
            program,
        );
        kernel.force_generic_drift = drift_specialization && !local;
        kernel.force_generic_dynamics =
            (dynamics_specialization || shared_scratch) && !local && !partial_reference;
        let mut states = [initial.clone_host().unwrap(), initial.clone_host().unwrap()];
        for state in &mut states {
            state.upload_to_device(device, queue).unwrap();
        }
        kernel
            .prepare_coordinate_compensation(device, &[&states[0], &states[1]])
            .unwrap();
        for state in &states {
            kernel.seed_coordinate_compensation(queue, state).unwrap();
        }
        runs.push((kernel, states));
    }
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("rounding variant times"),
        ty: wgpu::QueryType::Timestamp,
        count: 32,
    });
    let mut mismatches = 0;
    let capture = std::env::var_os("SOIL_MACHINE_ROUNDING_CAPTURE").map(std::path::PathBuf::from);
    if let Some(path) = &capture {
        std::fs::create_dir(path).unwrap();
    }
    for step in 0..8 {
        if mutate_constraints {
            use molla_sim::joint_control::JointCompliance;
            for art in 0..translations.len() {
                let joint = art * 6 + 1;
                let dof = model.joint_dof_offset.host().unwrap()[joint] as usize;
                model.joint_limit_lower.host_mut().unwrap()[dof] =
                    if step == 2 { -1e-6 } else { f64::NEG_INFINITY };
                model.joint_limit_upper.host_mut().unwrap()[dof] =
                    if step == 2 { 1e-6 } else { f64::INFINITY };
                model.joint_velocity_limit.host_mut().unwrap()[dof] =
                    if step == 3 { 0.001 } else { f64::INFINITY };
                let root_dof = model.joint_dof_offset.host().unwrap()[art * 6] as usize;
                model.joint_compliance.host_mut().unwrap()[root_dof] = if step == 4 {
                    JointCompliance {
                        enabled: 1,
                        padding: 0,
                        lower: -0.1,
                        upper: -0.1,
                        natural_frequency: 12.0,
                        damping_ratio: 0.7,
                    }
                } else {
                    JointCompliance::DISABLED
                };
                model.joint_enabled.host_mut().unwrap()[joint] = u32::from(step != 5);
            }
            model.upload_to_device(device, queue).unwrap();
            println!(
                "dynamics mutation step={step}: {}",
                [
                    "refresh",
                    "unrestricted",
                    "scalar stops",
                    "velocity cap",
                    "compliance",
                    "disabled joint",
                    "restored",
                    "restored continuation"
                ][step]
            );
        }
        let mut results = [Vec::new(), Vec::new()];
        let order = if dynamics_program.is_some() && step % 2 == 1 {
            [1, 0]
        } else {
            [0, 1]
        };
        for variant in order {
            let (kernel, states) = &mut runs[variant];
            let [a, b] = states;
            let (input, output) = if step % 2 == 0 { (a, b) } else { (b, a) };
            let slot = (step * 4 + variant * 2) as u32;
            if step == 0 || !drift_specialization {
                kernel.set_dispatch_timestamps(Some((queries.clone(), [slot, slot + 1])));
            }
            if !drift_only && step == 0 {
                kernel
                    .refresh(device, queue, &model, input, &control, output)
                    .unwrap();
            } else if !drift_only {
                kernel
                    .dispatch(
                        device,
                        queue,
                        &model,
                        input,
                        &control,
                        &Contacts::with_capacity(0),
                        0.0,
                        0.0005,
                        true,
                        true,
                        output,
                        None,
                    )
                    .unwrap();
            }
            if (drift_specialization && step != 0) || drift_only {
                let corrected: Vec<f32> = (0..model.joint_dof_total)
                    .map(|d| 0.003 * (d + 1) as f32 * if (d + step) % 2 == 0 { 1.0 } else { -1.0 })
                    .collect();
                queue.write_buffer(
                    output.joint_qd.device_buffer_current().unwrap(),
                    0,
                    bytemuck::cast_slice(&corrected),
                );
                if drift_only {
                    seed_velocity_delta(
                        device,
                        queue,
                        input.joint_qd.device_buffer_current().unwrap(),
                        output.joint_qd.device_buffer_current().unwrap(),
                        &kernel.velocity_delta,
                    );
                }
                let before_delta = words(device, queue, &kernel.velocity_delta);
                if drift_only {
                    assert!(before_delta.iter().any(|&v| f32::from_bits(v) != 0.0));
                }
                kernel.set_dispatch_timestamps(Some((queries.clone(), [slot, slot + 1])));
                kernel
                    .drift(device, queue, &model, input, &control, 0.0005, true, output)
                    .unwrap();
                assert_eq!(
                    words(
                        device,
                        queue,
                        output.joint_qd.device_buffer_current().unwrap()
                    ),
                    corrected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(words(device, queue, &kernel.velocity_delta), before_delta);
            }
            assert_eq!(words(device, queue, &kernel.stop_status), [0]);
            if mutate_constraints && (2..=5).contains(&step) {
                let q = words(
                    device,
                    queue,
                    output.joint_q.device_buffer_current().unwrap(),
                );
                let qd = words(
                    device,
                    queue,
                    output.joint_qd.device_buffer_current().unwrap(),
                );
                for art in 0..translations.len() {
                    let joint = art * 6 + usize::from(step != 4);
                    let coord = model.joint_coord_offset.host().unwrap()[joint] as usize;
                    let dof = model.joint_dof_offset.host().unwrap()[joint] as usize;
                    let position = f32::from_bits(q[coord]);
                    let rate = f32::from_bits(qd[dof]);
                    assert!(position.is_finite() && rate.is_finite());
                    match step {
                        2 => assert!(position.abs() <= 1e-6 + 2e-7, "scalar stop: {position}"),
                        3 => assert!(rate.abs() <= 0.001 + 2e-7, "velocity cap: {rate}"),
                        4 => assert!(
                            rate < -0.01,
                            "compliant spring must drive toward -0.1: {rate}"
                        ),
                        5 => assert_eq!(rate, 0.0, "disabled joint"),
                        _ => unreachable!(),
                    }
                }
            }
            let mut state_words = Vec::new();
            let mut fields = String::new();
            for (name, buffer) in [
                ("body_q", output.body_q.device_buffer().unwrap()),
                ("body_qd", output.body_qd.device_buffer().unwrap()),
                ("joint_q", output.joint_q.device_buffer().unwrap()),
                ("joint_qd", output.joint_qd.device_buffer().unwrap()),
                (
                    "coordinate_error",
                    kernel.coordinate_compensation_buffer(output).unwrap(),
                ),
                ("velocity_delta", &kernel.velocity_delta),
            ] {
                let values = words(device, queue, buffer);
                fields.push_str(&format!(
                    "{}\t{}\t{name}\n",
                    state_words.len(),
                    values.len()
                ));
                state_words.extend(values);
            }
            let position_start = state_words.len();
            state_words.extend(
                words(
                    device,
                    queue,
                    kernel.position_compensation_buffer(output).unwrap(),
                )
                .into_iter()
                .enumerate()
                .filter_map(|(i, word)| (i % 4 != 3).then_some(word)),
            );
            fields.push_str(&format!(
                "{position_start}\t{}\tposition_error_xyz\n",
                state_words.len() - position_start
            ));
            if let Some(path) = &capture {
                let prefix = format!("step-{step}-variant-{variant}");
                let bytes: Vec<_> = state_words
                    .iter()
                    .flat_map(|word| word.to_le_bytes())
                    .collect();
                std::fs::write(path.join(format!("{prefix}.bin")), bytes).unwrap();
                std::fs::write(path.join(format!("{prefix}.tsv")), &fields).unwrap();
            }
            results[variant] = state_words;
        }
        assert_eq!(results[0].len(), results[1].len());
        let differences: Vec<_> = results[0]
            .iter()
            .zip(&results[1])
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .collect();
        mismatches += differences.len();
        for (index, (a, b)) in differences.iter().take(6) {
            println!(
                "rounding mismatch step={step} word={index} legacy={a:#010x} candidate={b:#010x}"
            );
        }
    }
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("rounding times resolve"),
        size: 32 * 8,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.resolve_query_set(&queries, 0..32, &resolve, 0);
    queue.submit([encoder.finish()]);
    let raw = words(device, queue, &resolve);
    let ticks: Vec<u64> = raw
        .chunks_exact(2)
        .map(|w| u64::from(w[0]) | (u64::from(w[1]) << 32))
        .collect();
    let mut times = [0.0; 2];
    for (step, record) in ticks.chunks_exact(4).enumerate() {
        for variant in 0..2 {
            let ms = (record[variant * 2 + 1] - record[variant * 2]) as f64
                * f64::from(queue.get_timestamp_period())
                * 1e-6;
            if step != 0 {
                times[variant] += ms;
            }
            println!("rounding step={step} local={} gpu_ms={ms}", variant == 1);
        }
    }
    println!(
        "rounding comparison native={native} specialized_drift={drift_specialization} shared_coordinates={shared_coordinates} specialized_dynamics={dynamics_specialization} mutate_constraints={mutate_constraints} shared_scratch={shared_scratch} constraint_scratch={constraint_scratch} partial_reference={partial_reference}: {} articulations, mismatched words={mismatches}; measured GPU ms reference={} candidate={}",
        translations.len(),
        times[0],
        times[1]
    );
    assert_eq!(mismatches, 0, "compensated state must remain bit-identical");
}
