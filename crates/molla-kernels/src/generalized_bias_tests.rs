use super::retained_predictor::{input, pairs, values};
use super::*;
use crate::featherstone::rounding_tests::words;
use crate::{
    ExperimentalBiasBindings, ExperimentalBiasKernel, ExperimentalMassPredictor, MassMatrixKernel,
};
use molla_core::BodyId;
use molla_math::{Quat, Transform};
use molla_sim::{BodyParams, Model, ModelBuilder, State};

fn fixture_model() -> Model {
    let mut builder = ModelBuilder::new();
    let root = builder.add_body(
        BodyParams::new()
            .with_mass(1500.)
            .with_inertia(Mat3::IDENTITY * 600.),
    );
    builder
        .add_joint_free(
            BodyId::NONE,
            root,
            Transform::from_translation(Vec3::new(65536., 0.95, 0.)),
            Transform::IDENTITY,
        )
        .unwrap();
    for (x, z) in [(-0.75, -0.65), (-0.75, 0.65), (0.75, -0.65), (0.75, 0.65)] {
        let body = builder.add_body(
            BodyParams::new()
                .with_mass(35.)
                .with_inertia(Mat3::IDENTITY * 3.),
        );
        builder
            .add_joint_revolute(
                root,
                body,
                Transform::from_translation(Vec3::new(x, -0.55, z)),
                Transform::new(
                    Vec3::ZERO,
                    Quat::from_rotation_x(-std::f64::consts::FRAC_PI_2),
                ),
                Vec3::Z,
            )
            .unwrap();
    }
    let tine = builder.add_body(
        BodyParams::new()
            .with_mass(20.)
            .with_inertia(Mat3::IDENTITY),
    );
    builder
        .add_joint_fixed(
            root,
            tine,
            Transform::from_translation(Vec3::new(-1.45, -0.72, 0.)),
            Transform::IDENTITY,
        )
        .unwrap();
    builder.build().unwrap()
}

#[test]
#[ignore = "requires f64 GPU, resource counters and frozen six-body fixture"]
fn retained_bias_mass_chain_gpu_contract() {
    bias_chain(false);
}

#[test]
#[ignore = "requires f64 GPU, resource counters and frozen six-body fixture"]
fn retained_precise_operator_chain_gpu_contract() {
    bias_chain(true);
}

fn bias_chain(precise: bool) {
    assert!(molla_core::gpu_metrics::ENABLED);
    let start = std::time::Instant::now();
    let directory =
        std::path::PathBuf::from(std::env::var_os("MOLLA_ACCELERATION_ORACLE_CAPTURE").unwrap());
    let capture = std::path::PathBuf::from(std::env::var_os("MOLLA_BIAS_CAPTURE").unwrap());
    std::fs::create_dir(&capture).unwrap();
    let fixture = read(&directory.join("step-00-variant-0.bin"));
    validate_fixture_geometry(&fixture);
    let engine = molla_compute::Engine::new_gpu_with_features(wgpu::Features::SHADER_F64).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = fixture_model();
    let mut state = State::for_model(&model).unwrap();
    for body in 0..6 {
        state.body_q.host_mut().unwrap()[body] = Transform::new(
            position(&fixture, body),
            Quat::from_xyzw(
                f32::from_bits(fixture[8 * body + 4]) as f64,
                f32::from_bits(fixture[8 * body + 5]) as f64,
                f32::from_bits(fixture[8 * body + 6]) as f64,
                f32::from_bits(fixture[8 * body + 7]) as f64,
            ),
        );
    }
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let poses = state.body_q.device_buffer().unwrap();
    let loads = state.body_f.device_buffer().unwrap();
    let low = input(device, queue, &[0; 96]);
    let initial_rates: [f64; 10] = std::array::from_fn(|i| f32::from_bits(fixture[107 + i]) as f64);
    let rates = input(device, queue, bytemuck::cast_slice(&pairs(&initial_rates)));
    let bindings = || ExperimentalBiasBindings {
        poses,
        position_low: &low,
        paired_rates: &rates,
        external_com_wrenches: loads,
    };
    assert!(ExperimentalBiasKernel::new(device, queue, &model, bindings(), 1635).is_err());
    let bias = ExperimentalBiasKernel::new(device, queue, &model, bindings(), 1636).unwrap();
    let legacy_matrix = (!precise).then(|| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("live mass matrix"),
            size: 400,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    });
    let matrix = legacy_matrix.as_ref().unwrap_or_else(|| bias.mass_matrix());
    let mass_dispatch = (!precise).then(|| {
        let mass_kernel = MassMatrixKernel::new_on_device(device, queue, &model).unwrap();
        assert_eq!(mass_kernel.output_len(), 100);
        mass_kernel
            .prepare_on_device(device, queue, &model, &state, matrix)
            .unwrap()
    });
    let drive = input(device, queue, &[0; 80]);
    let predictor = ExperimentalMassPredictor::new_with_bias_status(
        device,
        10,
        matrix,
        &drive,
        bias.bias(),
        bias.status(),
        84,
    )
    .unwrap();
    let dispatch = |reset_bias: bool, reset_predictor: bool| {
        let mut encoder = device.create_command_encoder(&Default::default());
        let before = device.get_internal_counters().hal;
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
        if reset_bias {
            bias.reset(&mut encoder);
        }
        if reset_predictor {
            predictor.reset(&mut encoder);
        }
        if let Some(mass_dispatch) = &mass_dispatch {
            mass_dispatch.encode(&mut encoder);
        }
        bias.encode(&mut encoder);
        predictor.encode(&mut encoder);
        let after = device.get_internal_counters().hal;
        assert_eq!(before.buffers.read(), after.buffers.read());
        assert_eq!(before.bind_groups.read(), after.bind_groups.read());
        assert_eq!(
            before.compute_pipelines.read(),
            after.compute_pipelines.read()
        );
        assert_eq!(allocations.finish().observations, 0);
        assert_eq!(transfers.finish(), Default::default());
        let count = dispatches.finish();
        assert!(!count.overflowed);
        assert_eq!(count.encoded_dispatches, if precise { 2 } else { 3 });
        queue.submit([encoder.finish()]);
    };
    let mass = mass_matrix(&fixture);
    let gravity = Vec3::new(0.0, -(molla_math::GRAVITY as f32) as f64, 0.0);
    let offset = balance::force_offset(&fixture, gravity);
    let mut base = [0.0; 10];
    let mut max_bias = 0.0f64;
    let mut max_absolute = 0.0f64;
    let mut max_increment = 0.0f64;
    let mut max_rate = 0.0f64;
    let mut final_acceleration = Vec::new();
    let mut final_bias = Vec::new();
    for step in 0..10 {
        let force: [f64; 10] = std::array::from_fn(|i| {
            17.0 * (step + 1) as f64 * (i + 1) as f64 * if i % 2 == 0 { 1.0 } else { -1.0 }
        });
        queue.write_buffer(&drive, 0, bytemuck::cast_slice(&pairs(&force)));
        dispatch(false, false);
        assert_eq!(words(device, queue, bias.status()), [0]);
        assert_eq!(words(device, queue, predictor.status()), [0]);
        final_bias = words(device, queue, bias.bias());
        final_acceleration = words(device, queue, predictor.acceleration());
        std::fs::write(
            capture.join(format!("bias-{step:02}.bin")),
            bytemuck::cast_slice(&final_bias),
        )
        .unwrap();
        std::fs::write(
            capture.join(format!("acceleration-{step:02}.bin")),
            bytemuck::cast_slice(&final_acceleration),
        )
        .unwrap();
        let actual_bias = values(&final_bias);
        let actual = values(&final_acceleration);
        let rhs: [f64; 10] = std::array::from_fn(|i| force[i] + offset[i]);
        let scale = 1.0 + rhs.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let bias_scale = 1.0 + offset.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let expected = solve(mass, rhs);
        for i in 0..10 {
            max_bias = max_bias.max((actual_bias[i] + offset[i]).abs() / bias_scale);
            max_absolute = max_absolute
                .max(((0..10).map(|j| mass[i][j] * actual[j]).sum::<f64>() - rhs[i]).abs() / scale);
            max_rate = max_rate.max((actual[i] - expected[i]).abs() * f64::from(1.0f32 / 2040.0));
        }
        if step == 0 {
            base.copy_from_slice(&actual);
        } else {
            for i in 0..10 {
                let expected =
                    17.0 * step as f64 * (i + 1) as f64 * if i % 2 == 0 { 1.0 } else { -1.0 };
                max_increment = max_increment.max(
                    ((0..10)
                        .map(|j| mass[i][j] * (actual[j] - base[j]))
                        .sum::<f64>()
                        - expected)
                        .abs()
                        / (1.0 + 170.0 * step as f64),
                );
            }
        }
    }
    let raw_matrix = words(device, queue, matrix);
    std::fs::write(capture.join("mass.bin"), bytemuck::cast_slice(&raw_matrix)).unwrap();
    let mut mass_error = 0.0f64;
    for i in 0..10 {
        for j in 0..10 {
            mass_error = mass_error.max(
                (f32::from_bits(raw_matrix[i * 10 + j]) as f64 - mass[i][j]).abs()
                    / (1.0 + mass[i][i].abs()),
            );
        }
    }
    eprintln!(
        "bias_chain_matrix={mass_error:e} bias={max_bias:e} absolute={max_absolute:e} incremental={max_increment:e} rate={max_rate:e}"
    );
    assert!(mass_error <= 1e-6);
    assert!(max_bias <= 1e-6);
    assert!(max_absolute <= 1e-6);
    assert!(max_increment <= 1e-6);
    assert!(max_rate <= 1e-5);
    queue.write_buffer(&rates, 0, bytemuck::cast_slice(&[f32::NAN, 0.0]));
    dispatch(false, false);
    assert_eq!(words(device, queue, bias.status()), [1]);
    assert_eq!(words(device, queue, predictor.status()), [4]);
    assert_eq!(words(device, queue, bias.bias()), final_bias);
    assert_eq!(words(device, queue, matrix), raw_matrix);
    assert_eq!(
        words(device, queue, predictor.acceleration()),
        final_acceleration
    );
    queue.write_buffer(&rates, 0, bytemuck::cast_slice(&pairs(&initial_rates)));
    dispatch(false, true);
    assert_eq!(words(device, queue, predictor.status()), [4]);
    assert_eq!(
        words(device, queue, predictor.acceleration()),
        final_acceleration
    );
    dispatch(true, false);
    assert_eq!(words(device, queue, bias.status()), [0]);
    assert_eq!(words(device, queue, predictor.status()), [4]);
    assert_eq!(
        words(device, queue, predictor.acceleration()),
        final_acceleration
    );
    dispatch(true, true);
    assert_eq!(words(device, queue, predictor.status()), [0]);
    assert_eq!(
        words(device, queue, predictor.acceleration()),
        final_acceleration
    );
    assert_eq!(words(device, queue, bias.bias()), final_bias);
    assert_eq!(words(device, queue, matrix), raw_matrix);
    eprintln!(
        "bias_chain_sticky_upstream_reset=pass elapsed={:?}",
        start.elapsed()
    );
}
