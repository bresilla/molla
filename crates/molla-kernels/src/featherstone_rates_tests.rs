use super::*;
use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Mat3, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder};

pub(super) fn shader_source() -> String {
    rates::shader_source(
        &[
            include_str!("../shaders/featherstone.wgsl"),
            include_str!("../shaders/featherstone_stops.wgsl"),
            include_str!("../shaders/featherstone_feedback.wgsl"),
            include_str!("../shaders/featherstone_mass_inverse.wgsl"),
            include_str!("../shaders/quaternion_precision.wgsl"),
            include_str!("../shaders/joint_euler_chart.wgsl"),
            include_str!("../shaders/euler_bound_precision.wgsl"),
            include_str!("../shaders/featherstone_mixed_stops.wgsl"),
            include_str!("../shaders/cone_geometry.wgsl"),
            include_str!("../shaders/featherstone_rates.wgsl"),
            include_str!("../shaders/featherstone_body_rates.wgsl"),
        ]
        .join("\n"),
    )
}

#[test]
fn paired_rate_shader_compositions_validate() {
    let source = shader_source();
    for candidate in [
        source.clone(),
        rates::unrestricted_source(&source),
        rates::split_acceleration_source(&source, true),
        rates::split_acceleration_source(&source, false),
        rates::external_prediction_source(&source),
        source.replace("uniforms.paired_rates", "0u"),
        unrestricted_drift_source(&source),
        shared_dynamics_source(&source),
        constraint_scratch_source(&source),
    ] {
        let candidate = rates::dispatch_source(&candidate, true);
        let module = naga::front::wgsl::parse_str(&candidate)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&candidate)));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
    }
}

#[test]
fn external_prediction_prunes_only_forbidden_acceleration() {
    let source = shader_source();
    let reference = rates::split_acceleration_source(&source, false);
    let candidate = rates::external_prediction_source(&source);
    assert_ne!(reference, candidate);
    assert!(candidate.contains("uniforms.drift_only == 2u"));
    assert!(candidate.contains("uniforms.drift_only == 3u"));
    let specialize = |source: &str, mode: u32| {
        let mut result = source.replace("uniforms.drift_only", &format!("{mode}u"));
        for spacing in ["", " "] {
            for (operator, value) in [("==", mode == 0), ("!=", mode != 0)] {
                result = result.replace(
                    &format!("{mode}u{spacing}{operator}{spacing}0u"),
                    if value { "true" } else { "false" },
                );
            }
        }
        result
    };
    for mode in 1..=3 {
        assert_eq!(specialize(&reference, mode), specialize(&candidate, mode));
    }
    assert_ne!(specialize(&reference, 0), specialize(&candidate, 0));
    let dispatcher = include_str!("featherstone.rs");
    let guard = dispatcher
        .find("if self.external_prediction_only && !drift_only")
        .unwrap();
    let dispatch = dispatcher[guard..]
        .find("pass.dispatch_workgroups")
        .unwrap()
        + guard;
    assert!(guard < dispatch);
    assert!(
        dispatcher[guard..dispatch].contains("external prediction program cannot dispatch ABA")
    );
}

#[test]
fn paired_unrestricted_model_checks_mutable_assumptions() {
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    assert!(rates::unrestricted_model(&model).unwrap());
    model.joint_enabled.host_mut().unwrap()[0] |= 1 << 4;
    assert!(!rates::unrestricted_model(&model).unwrap());
    model.joint_enabled.host_mut().unwrap()[0] = 1;
    for value in [0., f64::NAN, f64::INFINITY] {
        model.joint_limit_lower.host_mut().unwrap()[0] = value;
        assert!(!rates::unrestricted_model(&model).unwrap());
    }
    model.joint_limit_lower.host_mut().unwrap()[0] = f64::NEG_INFINITY;
    for value in [0., f64::NAN, f64::NEG_INFINITY] {
        model.joint_limit_upper.host_mut().unwrap()[0] = value;
        assert!(!rates::unrestricted_model(&model).unwrap());
    }
    model.joint_limit_upper.host_mut().unwrap()[0] = f64::INFINITY;
    model.joint_velocity_limit.host_mut().unwrap()[0] = 4.;
    assert!(!rates::unrestricted_model(&model).unwrap());
    model.joint_velocity_limit.host_mut().unwrap()[0] = f64::INFINITY;
    for kind in [molla_sim::JointType::Distance as u32, u32::MAX] {
        model.joint_type.host_mut().unwrap()[0] = kind;
        assert!(!rates::unrestricted_model(&model).unwrap());
    }
    model.joint_type.host_mut().unwrap()[0] = molla_sim::JointType::Free as u32;
    assert!(rates::unrestricted_model(&model).unwrap());
}

fn precise(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<f64> {
    rounding_tests::words(device, queue, buffer)
        .chunks_exact(2)
        .map(|p| f32::from_bits(p[0]) as f64 + f32::from_bits(p[1]) as f64)
        .collect()
}

#[test]
#[ignore = "requires real GPU"]
fn paired_prediction_and_corrected_drift_retain_low_rates() {
    exercise_rates(true);
}

#[test]
#[ignore = "requires real GPU"]
fn paired_corrected_drift_uses_low_rates() {
    exercise_rates(false);
}

#[test]
#[ignore = "requires real GPU"]
fn paired_single_lane_dispatch_covers_all_articulations() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    for _ in 0..65 {
        builder.begin_articulation();
        let body = builder.add_body(BodyParams::new().with_mass(1.).with_inertia(Mat3::IDENTITY));
        builder
            .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    assert_eq!(model.articulation_count, 65);
    model.upload_to_device(device, queue).unwrap();
    let mut input = State::for_model(&model).unwrap();
    for i in 0..65 {
        let velocity = (i + 1) as f64 + 2f64.powi(-30);
        input.joint_qd.host_mut().unwrap()[i * 6] = velocity;
        input.body_qd.host_mut().unwrap()[i] =
            molla_math::SpatialVector::linear(Vec3::X * velocity);
    }
    let mut output = input.clone_host().unwrap();
    input.upload_to_device(device, queue).unwrap();
    output.upload_to_device(device, queue).unwrap();
    let (offsets, rows) = kernel_configuration(&model);
    let program = FeatherstoneProgram::with_rate_precision(
        device,
        1,
        rows,
        false,
        false,
        false,
        FeatherstonePasses::UnrestrictedDriftOnly,
        true,
    );
    let mut kernel =
        FeatherstoneKernel::from_program(device, queue, &model, offsets, rows, &program);
    kernel
        .prepare_coordinate_compensation(device, &[&input, &output])
        .unwrap();
    kernel
        .prepare_paired_rates(device, queue, &[&input, &output], 208 * 65)
        .unwrap();
    let active: Vec<bool> = (0..65).map(|i| i % 5 != 1).collect();
    kernel.set_active_articulations(queue, &active).unwrap();
    let mut control = Control::for_model(&model);
    control.upload_to_device(device, queue).unwrap();
    let dt = 0.001f32;
    kernel
        .drift(
            device,
            queue,
            &model,
            &input,
            &control,
            dt as f64,
            true,
            &mut output,
        )
        .unwrap();
    assert_eq!(
        rounding_tests::words(device, queue, kernel.status_buffer()),
        [0]
    );
    let high = rounding_tests::words(
        device,
        queue,
        output.body_q.device_buffer_current().unwrap(),
    );
    let low = rounding_tests::words(
        device,
        queue,
        kernel.position_compensation_buffer(&output).unwrap(),
    );
    for i in 0..65 {
        let position = f32::from_bits(high[i * 8]) as f64 + f32::from_bits(low[i * 4]) as f64;
        let expected = if active[i] {
            ((i + 1) as f64 + 2f64.powi(-30)) * dt as f64
        } else {
            0.
        };
        assert!(
            (position - expected).abs() < 4e-12,
            "articulation {i}: {position} != {expected}"
        );
    }
}

fn exercise_rates(prediction: bool) {
    exercise_rate_program(prediction, None);
}

#[test]
#[ignore = "requires real GPU"]
fn paired_split_acceleration_prediction_and_corrected_drift() {
    exercise_rate_program(true, Some(FeatherstonePasses::PairedUnrestricted));
}

#[test]
#[cfg(feature = "lazy-paired-drift")]
#[ignore = "requires GPU; public paired lazy drift prediction and lifecycle"]
fn paired_lazy_drift_public_runtime_preserves_rates_and_guards() {
    let records = exercise_rate_program_profile(
        true,
        Some(FeatherstonePasses::PairedUnrestricted),
        Some(FeatherstoneExecution::Generic),
    );
    assert_eq!(records.len(), 576);
}

fn exercise_rate_program(prediction: bool, pass: Option<FeatherstonePasses>) {
    exercise_rate_program_profile(prediction, pass, None);
}

#[test]
#[cfg(feature = "native-geometry-drift")]
#[ignore = "requires native f64 GPU; public geometry-only drift prediction and lifecycle"]
fn paired_native_geometry_public_runtime_preserves_rates_and_guards() {
    if cfg!(feature = "native-lazy-geometry-drift") {
        assert_eq!(DriftBackend::compiled(), DriftBackend::NativeLazyGeometry);
    }
    let records = exercise_rate_program_profile(
        true,
        Some(FeatherstonePasses::PairedUnrestricted),
        Some(FeatherstoneExecution::Generic),
    );
    assert_eq!(records.len(), 576);
    if let Some(path) = std::env::var_os("MOLLA_NATIVE_LAZY_PUBLIC_CAPTURE") {
        use std::io::Write;
        std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(path)
            .unwrap()
            .write_all(bytemuck::cast_slice(&records))
            .unwrap();
    }
}

#[test]
#[cfg(feature = "native-geometry-drift")]
#[ignore = "requires GPU; reject missing f64 before program construction"]
fn paired_native_geometry_rejects_missing_device_feature() {
    let engine = Engine::new_gpu().unwrap();
    assert!(
        !engine
            .device()
            .unwrap()
            .features()
            .contains(wgpu::Features::SHADER_F64)
    );
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(BodyParams::new());
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let model = builder.build().unwrap();
    let result = FeatherstoneKernel::new_cached_with_paired_rates(
        &engine,
        &model,
        FeatherstoneRounding::Storage,
        FeatherstoneExecution::Generic,
    );
    assert!(
        matches!(result,Err(Error::Gpu(ref message)) if message.contains("requires SHADER_F64"))
    );
}

#[test]
#[ignore = "requires real GPU and fresh SOIL_PAIRED_TYPES_OUTPUT; explicit reference/candidate arm"]
fn paired_joint_type_profile_retains_low_rates_and_guards() {
    let execution = match std::env::var("SOIL_PAIRED_TYPES_ARM").unwrap().as_str() {
        "reference" => FeatherstoneExecution::Generic,
        "candidate" => FeatherstoneExecution::ExperimentalJointTypes,
        _ => panic!("explicit reference or candidate required"),
    };
    let records = exercise_rate_program_profile(
        true,
        Some(FeatherstonePasses::PairedUnrestricted),
        Some(execution),
    );
    let output = std::env::var_os("SOIL_PAIRED_TYPES_OUTPUT").expect("new output file");
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)
        .unwrap();
    file.write_all(bytemuck::cast_slice(&records)).unwrap();
    eprintln!(
        "paired type profile {}: {} state words retained, all physical gates unchanged",
        execution.label(),
        records.len()
    );
}

fn exercise_rate_program_profile(
    prediction: bool,
    pass: Option<FeatherstonePasses>,
    execution: Option<FeatherstoneExecution>,
) -> Vec<u32> {
    let engine =
        Engine::new_gpu_with_features(FeatherstoneKernel::paired_drift_required_features())
            .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    let body = builder.add_body(
        BodyParams::new()
            .with_mass(1e6)
            .with_inertia(Mat3::IDENTITY),
    );
    builder
        .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
        .unwrap();
    let mut model = builder.build().unwrap();
    model.gravity.host_mut().unwrap().fill(Vec3::ZERO);
    model.upload_to_device(device, queue).unwrap();
    let mut initial = State::for_model(&model).unwrap();
    initial.joint_qd.host_mut().unwrap()[0] = 1.0 + 2f64.powi(-30);
    if !prediction {
        initial.joint_qd.host_mut().unwrap()[3] = 1.0 + 2f64.powi(-27);
    }
    let mut control = Control::for_model(&model);
    control.joint_force.host_mut().unwrap()[0] = 1.0;
    control.upload_to_device(device, queue).unwrap();
    let mut states = [initial.clone_host().unwrap(), initial.clone_host().unwrap()];
    for state in &mut states {
        state.upload_to_device(device, queue).unwrap();
    }
    let (offsets, rows) = kernel_configuration(&model);
    if pass == Some(FeatherstonePasses::PairedUnrestricted) {
        assert!(rates::unrestricted_model(&model).unwrap());
    }
    let passes = pass.unwrap_or(if prediction {
        FeatherstonePasses::GenericOnly
    } else {
        FeatherstonePasses::UnrestrictedDriftOnly
    });
    eprintln!("paired rate program construction: prediction={prediction}");
    let mut kernel = if let Some(execution) = execution {
        let kernel = FeatherstoneKernel::new_cached_with_paired_rates(
            &engine,
            &model,
            FeatherstoneRounding::Storage,
            execution,
        )
        .unwrap();
        assert_eq!(kernel.execution(), execution);
        assert!(kernel.split_acceleration);
        assert_eq!(kernel.drift_backend, DriftBackend::compiled());
        if DriftBackend::compiled() != DriftBackend::Eager {
            assert_eq!(
                kernel.unrestricted_drift_backend(),
                DriftBackend::compiled().label()
            );
            assert_ne!(kernel.pipeline, kernel.unrestricted_drift_pipeline);
            assert_ne!(kernel.dynamics_pipeline, kernel.unrestricted_drift_pipeline);
        }
        kernel
    } else {
        let program = FeatherstoneProgram::with_rate_precision(
            device, 1, rows, false, false, false, passes, true,
        );
        FeatherstoneKernel::from_program(device, queue, &model, offsets, rows, &program)
    };
    eprintln!("paired rate program ready: prediction={prediction}");
    let mut records = Vec::new();
    kernel
        .prepare_coordinate_compensation(device, &[&states[0], &states[1]])
        .unwrap();
    assert!(
        kernel
            .prepare_paired_rates(device, queue, &[&states[0], &states[1]], 207)
            .is_err()
    );
    assert!(
        kernel
            .prepare_paired_rates(device, queue, &[&states[0], &states[0]], 208)
            .is_err()
    );
    kernel
        .prepare_paired_rates(device, queue, &[&states[0], &states[1]], 208)
        .unwrap();
    assert_eq!(kernel.paired_rate_owned_bytes(), 208);
    let mut expected = 1.0 + 2f64.powi(-30);
    let dt = 0.001f32;
    for step in 0..8 {
        let (a, b) = states.split_at_mut(1);
        let (input, output) = if step % 2 == 0 {
            (&a[0], &mut b[0])
        } else {
            (&b[0], &mut a[0])
        };
        if prediction {
            kernel
                .dispatch(
                    device,
                    queue,
                    &model,
                    input,
                    &control,
                    &Contacts::with_capacity(0),
                    0.,
                    dt as f64,
                    true,
                    true,
                    output,
                    None,
                )
                .unwrap();
        } else {
            let values =
                rounding_tests::words(device, queue, kernel.paired_rates_buffer(input).unwrap());
            queue.write_buffer(
                kernel.paired_rates_buffer(output).unwrap(),
                0,
                bytemuck::cast_slice(&values),
            );
            kernel
                .drift(
                    device, queue, &model, input, &control, dt as f64, true, output,
                )
                .unwrap();
        }
        assert_eq!(
            rounding_tests::words(device, queue, kernel.status_buffer()),
            [0]
        );
        let next = precise(device, queue, kernel.paired_rates_buffer(output).unwrap())[0];
        let delta = precise(
            device,
            queue,
            kernel.paired_velocity_delta_buffer().unwrap(),
        )[0];
        let target_delta = if prediction {
            1e-6f32 as f64 * dt as f64
        } else {
            0.0
        };
        assert!((delta - target_delta).abs() < 2e-15, "delta {delta}");
        expected += delta;
        assert!(
            (next - expected).abs() < 2e-14,
            "rate {next}, expected {expected}"
        );
        assert!(next > 1.0);
        assert_eq!(
            rounding_tests::words(
                device,
                queue,
                output.joint_qd.device_buffer_current().unwrap()
            )[0],
            1f32.to_bits()
        );
        let correction = [1f32, (next - 1.0 + 2f64.powi(-31)) as f32];
        queue.write_buffer(
            kernel.paired_rates_buffer(output).unwrap(),
            0,
            bytemuck::cast_slice(&correction),
        );
        let saved = rounding_tests::words(
            device,
            queue,
            kernel.paired_velocity_delta_buffer().unwrap(),
        );
        kernel
            .drift(
                device, queue, &model, input, &control, dt as f64, true, output,
            )
            .unwrap();
        assert_eq!(
            rounding_tests::words(device, queue, kernel.status_buffer()),
            [0]
        );
        assert_eq!(
            rounding_tests::words(
                device,
                queue,
                kernel.paired_velocity_delta_buffer().unwrap()
            ),
            saved
        );
        expected = correction[0] as f64 + correction[1] as f64;
        let body_high = rounding_tests::words(
            device,
            queue,
            output.body_qd.device_buffer_current().unwrap(),
        );
        let body_low = rounding_tests::words(
            device,
            queue,
            kernel.body_velocity_compensation_buffer(output).unwrap(),
        );
        for lane in 0..8 {
            let value =
                f32::from_bits(body_high[lane]) as f64 + f32::from_bits(body_low[lane]) as f64;
            let reference = if lane == 0 {
                expected
            } else if lane == 4 && !prediction {
                1.0 + 2f64.powi(-27)
            } else {
                0.0
            };
            assert!(
                (value - reference).abs() < 2e-14,
                "body rate lane {lane}: {value} != {reference}"
            );
        }
        assert_eq!(
            precise(device, queue, kernel.paired_rates_buffer(output).unwrap())[0],
            expected
        );
        let q = rounding_tests::words(
            device,
            queue,
            output.joint_q.device_buffer_current().unwrap(),
        );
        let low = rounding_tests::words(
            device,
            queue,
            kernel.coordinate_compensation_buffer(output).unwrap(),
        );
        let qi = rounding_tests::words(
            device,
            queue,
            input.joint_q.device_buffer_current().unwrap(),
        );
        let li = rounding_tests::words(
            device,
            queue,
            kernel.coordinate_compensation_buffer(input).unwrap(),
        );
        let actual = f32::from_bits(q[0]) as f64 + f32::from_bits(low[0]) as f64;
        let previous = f32::from_bits(qi[0]) as f64 + f32::from_bits(li[0]) as f64;
        assert!(
            (actual - previous - expected * dt as f64).abs() < 2e-15,
            "drift {actual}, previous {previous}"
        );
        if !prediction {
            let before: [f64; 4] = std::array::from_fn(|i| {
                f32::from_bits(qi[3 + i]) as f64 + f32::from_bits(li[3 + i]) as f64
            });
            let omega = 1.0 + 2f64.powi(-27);
            let h = 0.5 * dt as f64 * omega;
            let mut reference = [
                before[0] + h * before[3],
                before[1] - h * before[2],
                before[2] + h * before[1],
                before[3] - h * before[0],
            ];
            let norm = reference.iter().map(|x| x * x).sum::<f64>().sqrt();
            for i in 0..4 {
                reference[i] /= norm;
                let actual = f32::from_bits(q[3 + i]) as f64 + f32::from_bits(low[3 + i]) as f64;
                assert!(
                    (actual - reference[i]).abs() < 2e-14,
                    "quaternion axis {i}: {actual} != {}",
                    reference[i]
                );
            }
        }
        if execution.is_some() {
            for buffer in [
                output.body_q.device_buffer_current().unwrap(),
                output.body_qd.device_buffer_current().unwrap(),
                output.joint_q.device_buffer_current().unwrap(),
                output.joint_qd.device_buffer_current().unwrap(),
                kernel.paired_rates_buffer(output).unwrap(),
                kernel.paired_velocity_delta_buffer().unwrap(),
                kernel.body_velocity_compensation_buffer(output).unwrap(),
                kernel.coordinate_compensation_buffer(output).unwrap(),
                kernel.position_compensation_buffer(output).unwrap(),
            ] {
                records.extend(rounding_tests::words(device, queue, buffer));
            }
        }
    }
    let (output, input) = states.split_at_mut(1);
    let input = &input[0];
    let output = &mut output[0];
    let retained = kernel.paired_rates_buffer(output).unwrap().clone();
    let original = rounding_tests::words(device, queue, &retained);
    let mirror = rounding_tests::words(
        device,
        queue,
        output.joint_qd.device_buffer_current().unwrap(),
    );
    let checkpoint = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("paired rate restore checkpoint"),
        size: 48,
        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&retained, 0, &checkpoint, 0, 48);
    queue.submit([encoder.finish()]);
    queue.write_buffer(&retained, 44, bytemuck::bytes_of(&f32::NAN));
    kernel
        .drift(
            device, queue, &model, input, &control, dt as f64, true, output,
        )
        .unwrap();
    assert_eq!(
        rounding_tests::words(device, queue, kernel.status_buffer()),
        [32]
    );
    assert_eq!(
        rounding_tests::words(
            device,
            queue,
            output.joint_qd.device_buffer_current().unwrap()
        ),
        mirror
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(&checkpoint, 0, &retained, 0, 48);
    encoder.clear_buffer(kernel.status_buffer(), 0, None);
    queue.submit([encoder.finish()]);
    assert_eq!(rounding_tests::words(device, queue, &retained), original);
    assert!(
        kernel
            .prepare_paired_rates(device, queue, &[input, output], 207)
            .is_err()
    );
    assert_eq!(kernel.paired_rates_buffer(output).unwrap(), &retained);
    kernel
        .drift(
            device, queue, &model, input, &control, dt as f64, true, output,
        )
        .unwrap();
    let position = |kernel: &FeatherstoneKernel, state: &State| {
        let high = rounding_tests::words(
            device,
            queue,
            state.joint_q.device_buffer_current().unwrap(),
        );
        let low = rounding_tests::words(
            device,
            queue,
            kernel.coordinate_compensation_buffer(state).unwrap(),
        );
        f32::from_bits(high[0]) as f64 + f32::from_bits(low[0]) as f64
    };
    let paired_position = position(&kernel, output);
    if kernel.split_acceleration {
        let delta = rounding_tests::words(device, queue, &kernel.velocity_delta);
        let rates_before = rounding_tests::words(device, queue, &retained);
        model.joint_limit_lower.host_mut().unwrap()[0] = 0.;
        model.upload_to_device(device, queue).unwrap();
        assert!(
            kernel
                .dispatch(
                    device,
                    queue,
                    &model,
                    input,
                    &control,
                    &Contacts::with_capacity(0),
                    0.,
                    dt as f64,
                    true,
                    true,
                    output,
                    None
                )
                .is_err()
        );
        assert_eq!(
            rounding_tests::words(device, queue, &kernel.velocity_delta),
            delta
        );
        assert_eq!(
            rounding_tests::words(device, queue, &retained),
            rates_before
        );
        assert_eq!(position(&kernel, output), paired_position);
        model.joint_limit_lower.host_mut().unwrap()[0] = f64::NEG_INFINITY;
        model.upload_to_device(device, queue).unwrap();
    }
    kernel.prepare_paired_rates(device, queue, &[], 0).unwrap();
    assert_eq!(kernel.paired_rate_owned_bytes(), 0);
    assert!(kernel.paired_rates_buffer(output).is_err());
    kernel
        .drift(
            device, queue, &model, input, &control, dt as f64, true, output,
        )
        .unwrap();
    let scalar_position = position(&kernel, output);
    let expected_difference = (expected - 1.0) * dt as f64;
    assert!(expected_difference > 1e-12);
    assert!((paired_position - scalar_position - expected_difference).abs() < 2e-15);
    records
}

#[test]
#[ignore = "requires real GPU"]
fn paired_prediction_arithmetic_preserves_force_increment_and_damping() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let source = format!(
        "{}\n{}",
        shader_source(),
        r#"
@compute @workgroup_size(1)
fn rate_probe() {
    rate_rounding_child=0u;
    if uniforms.drift_only==4u {
        fk_pair_rounding=true;
        var a=PositionPair(vec3<f32>(0.0),vec3<f32>(0.0));
        var b=a;
        for (var i=0u;i<3u;i++) {a.high[i]=paired_rates[i].x; a.low[i]=paired_rates[i].y; b.high[i]=paired_rates[i+3u].x; b.low[i]=paired_rates[i+3u].y;}
        let cross=paired_cross(0u,a,b);
        let sum=position_add(0u,PositionPair(vec3<f32>(0.25,0.5,0.75),vec3<f32>(1.862645149230957e-9,3.725290298461914e-9,5.587935447692871e-9)),cross);
        for (var i=0u;i<3u;i++) {paired_rate_delta[i]=vec2<f32>(cross.high[i],cross.low[i]); paired_rate_delta[i+3u]=vec2<f32>(sum.high[i],sum.low[i]);}
        return;
    }
    for (var d=0u; d<6u; d++) {
        let previous=paired_rates[d];
        joint_qd[d]=rate_integrate(0u,0u,d,1e-6,uniforms.joint_damping);
        if d==1u { joint_qd[d]=rate_cap(d,joint_qd[d]); }
        if d==2u { joint_qd[d]=0.0; }
        rate_finish(d,previous,d==2u);
    }
}"#
    );
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("paired predictor arithmetic"),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("paired predictor arithmetic"),
        layout: None,
        module: &shader,
        entry_point: Some("rate_probe"),
        compilation_options: Default::default(),
        cache: None,
    });
    let allocate = |bytes: &[u8], uniform: bool| {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST
                | if uniform {
                    wgpu::BufferUsages::UNIFORM
                } else {
                    wgpu::BufferUsages::STORAGE
                },
            mapped_at_creation: false,
        });
        queue.write_buffer(&buffer, 0, bytes);
        buffer
    };
    let offsets = allocate(bytemuck::cast_slice(&[0u32, 6]), false);
    let caps = allocate(
        bytemuck::cast_slice(&[
            f32::INFINITY,
            1.,
            f32::INFINITY,
            f32::INFINITY,
            f32::INFINITY,
            f32::INFINITY,
        ]),
        false,
    );
    let high = allocate(&[0; 24], false);
    let delta = allocate(&[0; 24], false);
    let scratch = allocate(&[0; 16], false);
    let coordinate_scratch = allocate(&[0; 4], false);
    let initial = [[1f32, 2f32.powi(-30)]; 6];
    let pairs = allocate(bytemuck::cast_slice(&initial), false);
    let pair_delta = allocate(&[0; 48], false);
    let mut uniforms = Uniforms {
        dt: 0.001,
        joint_damping: 1.0,
        joint_count: 1,
        articulation_count: 1,
        rigorous_d_solve: 1,
        rigorous_distance: 1,
        drift_only: 0,
        stable_motor_feedback: 0,
        external_scalar_bounds: 0,
        compensated_coordinates: 1,
        control_body_wrenches: 0,
        paired_rates: 1,
    };
    let params = allocate(bytemuck::bytes_of(&uniforms), true);
    let sets: [Vec<(u32, &wgpu::Buffer)>; 5] = [
        vec![(7, &offsets), (12, &caps)],
        vec![(7, &high)],
        vec![],
        vec![
            (3, &delta),
            (4, &coordinate_scratch),
            (5, &scratch),
            (6, &pairs),
            (7, &pair_delta),
        ],
        vec![(0, &params)],
    ];
    let groups: Vec<_> = sets
        .iter()
        .enumerate()
        .map(|(i, buffers)| {
            let entries: Vec<_> = buffers
                .iter()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: *binding,
                    resource: buffer.as_entire_binding(),
                })
                .collect();
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &pipeline.get_bind_group_layout(i as u32),
                entries: &entries,
            })
        })
        .collect();
    let mut expected = [1.0 + 2f64.powi(-30); 6];
    for step in 0..8 {
        uniforms.joint_damping = if step < 4 { 1.0 } else { 0.999f32 };
        queue.write_buffer(&params, 0, bytemuck::bytes_of(&uniforms));
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&Default::default());
            pass.set_pipeline(&pipeline);
            for (i, group) in groups.iter().enumerate() {
                pass.set_bind_group(i as u32, group, &[]);
            }
            pass.dispatch_workgroups(1, 1, 1);
        }
        queue.submit([encoder.finish()]);
        let output = precise(device, queue, &pairs);
        let increments = precise(device, queue, &pair_delta);
        for d in 0..6 {
            let previous = expected[d];
            expected[d] =
                (previous + 1e-6f32 as f64 * uniforms.dt as f64) * uniforms.joint_damping as f64;
            if d == 1 {
                expected[d] = expected[d].min(1.0);
            }
            if d == 2 {
                expected[d] = 0.0;
            }
            assert!(
                (output[d] - expected[d]).abs() < 2e-14,
                "step {step}, dof {d}: {} != {}",
                output[d],
                expected[d]
            );
            assert!(
                (increments[d] - (output[d] - previous)).abs() < 3e-14,
                "paired delta step {step} dof {d}"
            );
        }
    }
    let omega = Vec3::new(1., 1.25, 1.5) * (1.0 + 2f64.powi(-27));
    let arm = molla_math::Quat::from_rotation_z(0.17) * Vec3::new(0.125, -0.25, 0.5);
    let values: Vec<[f32; 2]> = omega
        .to_array()
        .into_iter()
        .chain(arm.to_array())
        .map(|v| [v as f32, (v - v as f32 as f64) as f32])
        .collect();
    uniforms.drift_only = 4;
    queue.write_buffer(&params, 0, bytemuck::bytes_of(&uniforms));
    queue.write_buffer(&pairs, 0, bytemuck::cast_slice(&values));
    let mut encoder = device.create_command_encoder(&Default::default());
    {
        let mut pass = encoder.begin_compute_pass(&Default::default());
        pass.set_pipeline(&pipeline);
        for (i, group) in groups.iter().enumerate() {
            pass.set_bind_group(i as u32, group, &[]);
        }
        pass.dispatch_workgroups(1, 1, 1);
    }
    queue.submit([encoder.finish()]);
    let actual = precise(device, queue, &pair_delta);
    let cross = omega.cross(arm);
    let sum = cross + Vec3::new(0.25, 0.5, 0.75) * (1.0 + 2f64.powi(-27));
    eprintln!("body arithmetic: actual={actual:?}, cross={cross:?}, sum={sum:?}");
    for (i, reference) in cross
        .to_array()
        .into_iter()
        .chain(sum.to_array())
        .enumerate()
    {
        assert!(
            (actual[i] - reference).abs() < 4e-14,
            "body arithmetic axis {i}: {} != {reference}",
            actual[i]
        );
    }
}
