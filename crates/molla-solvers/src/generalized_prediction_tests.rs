use super::*;

#[path = "generalized_prediction_replay_tests.rs"]
mod replay;

#[path = "generalized_prediction_input_tests.rs"]
mod inputs;

#[path = "generalized_prediction_refresh_tests.rs"]
mod refreshed;

#[test]
fn generalized_prediction_shader_contract() {
    for width in [1, 10, 32] {
        let text = source(width, 1.0 / 2040.0, 0.02);
        for text in [text.clone(), active_source(text)] {
            let module = naga::front::wgsl::parse_str(&text)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&text)));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
        }
    }
    assert_eq!(
        PreparedGeneralizedPrediction::required_bytes(6, 10).unwrap(),
        3960
    );
    assert!(PreparedGeneralizedPrediction::required_bytes(0, 10).is_err());
    assert!(PreparedGeneralizedPrediction::required_bytes(6, 33).is_err());
}

#[test]
fn generalized_prediction_model_array_contract() {
    let mut model = model();
    assert!(validate_rates(&model).is_ok());
    model.joint_velocity_limit =
        crate::storage::Buffer::from_vec(vec![], crate::storage::BufferUsage::Storage);
    assert!(validate_rates(&model).is_err());
    model.joint_velocity_limit = crate::storage::Buffer::from_vec(
        vec![f64::INFINITY; 8],
        crate::storage::BufferUsage::Storage,
    );
    assert!(validate_rates(&model).is_ok());
    model.joint_friction =
        crate::storage::Buffer::from_vec(vec![], crate::storage::BufferUsage::Storage);
    assert!(validate_rates(&model).is_err());
}

fn pair(v: f64) -> [f32; 2] {
    let high = v as f32;
    [high, (v - f64::from(high)) as f32]
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer) -> Vec<u32> {
    let (tx, rx) = std::sync::mpsc::channel();
    wgpu::util::DownloadBuffer::read_buffer(device, queue, &buffer.slice(..), move |r| {
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
        .unwrap()
        .chunks_exact(4)
        .map(|x| u32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}

fn decoded(raw: &[u32]) -> Vec<f64> {
    raw.chunks_exact(2)
        .map(|x| f64::from(f32::from_bits(x[0])) + f64::from(f32::from_bits(x[1])))
        .collect()
}

fn solve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    let mut a = a.to_vec();
    let mut b = b.to_vec();
    for k in 0..n {
        let pivot = (k..n)
            .max_by(|&i, &j| a[i * n + k].abs().total_cmp(&a[j * n + k].abs()))
            .unwrap();
        assert!(a[pivot * n + k].abs() > 1e-12);
        for j in 0..n {
            a.swap(k * n + j, pivot * n + j);
        }
        b.swap(k, pivot);
        for i in k + 1..n {
            let c = a[i * n + k] / a[k * n + k];
            for j in k..n {
                a[i * n + j] -= c * a[k * n + j];
            }
            b[i] -= c * b[k];
        }
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = (b[i] - (i + 1..n).map(|j| a[i * n + j] * x[j]).sum::<f64>()) / a[i * n + i];
    }
    x
}

fn model() -> Model {
    use crate::{
        core::BodyId,
        math::{Mat3, Quat, Transform, Vec3},
        sim::{BodyParams, ModelBuilder},
    };
    let mut b = ModelBuilder::new();
    let root = b.add_body(
        BodyParams::new()
            .with_mass(30.0)
            .with_inertia(Mat3::from_diagonal(Vec3::new(4.0, 3.0, 2.0))),
    );
    b.add_joint_free(
        BodyId::NONE,
        root,
        Transform::new(Vec3::new(65536.0, 1.0, -32.0), Quat::from_rotation_z(0.3)),
        Transform::new(Vec3::new(0.125, 0.0, -0.25), Quat::from_rotation_x(-0.2)),
    )
    .unwrap();
    let hinge = b.add_body(
        BodyParams::new()
            .with_mass(3.0)
            .with_inertia(Mat3::from_diagonal(Vec3::new(0.5, 0.25, 0.75))),
    );
    b.add_joint_revolute(
        root,
        hinge,
        Transform::new(Vec3::new(0.75, -0.25, 0.5), Quat::from_rotation_y(0.4)),
        Transform::new(Vec3::new(0.0, 0.125, 0.0), Quat::from_rotation_z(-0.3)),
        Vec3::new(1.0, 2.0, 3.0).normalize(),
    )
    .unwrap();
    let slider = b.add_body(
        BodyParams::new()
            .with_mass(5.0)
            .with_inertia(Mat3::from_diagonal(Vec3::new(1.0, 0.5, 0.75))),
    );
    b.add_joint_prismatic(
        hinge,
        slider,
        Transform::new(Vec3::new(0.25, 0.5, -0.25), Quat::from_rotation_x(0.2)),
        Transform::new(Vec3::new(-0.125, 0.0, 0.25), Quat::from_rotation_y(0.15)),
        Vec3::new(1.0, -1.0, 2.0).normalize(),
    )
    .unwrap();
    let implement = b.add_body(
        BodyParams::new()
            .with_mass(8.0)
            .with_inertia(Mat3::from_diagonal(Vec3::new(1.0, 2.0, 1.5))),
    );
    b.add_joint_fixed(
        slider,
        implement,
        Transform::new(Vec3::new(-0.5, -0.25, 0.125), Quat::from_rotation_z(0.6)),
        Transform::IDENTITY,
    )
    .unwrap();
    let mut model = b.build().unwrap();
    model.joint_axis.host_mut().unwrap()[1] *= 1.0000002;
    model.joint_axis.host_mut().unwrap()[2] *= 0.9999998;
    for i in 0..4 {
        model.body_com.host_mut().unwrap()[i] = Vec3::new(0.125 * (i + 1) as f64, -0.0625, 0.125);
    }
    model
}

#[test]
#[ignore = "requires f64 GPU, timestamps and resource counters"]
fn generalized_prediction_gpu_physical_contract() {
    use crate::{
        math::{Quat, SpatialVector, Vec3},
        sim::State,
    };
    assert!(molla_core::gpu_metrics::ENABLED);
    let capture =
        std::path::PathBuf::from(std::env::var_os("MOLLA_GENERALIZED_PREDICTION_CAPTURE").unwrap());
    std::fs::create_dir(&capture).unwrap();
    let engine = crate::compute::Engine::new_gpu_with_features(
        wgpu::Features::SHADER_F64
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS,
    )
    .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut model = model();
    let mut state = State::for_model(&model).unwrap();
    crate::sim::eval_fk(&model, &mut state).unwrap();
    model.upload_to_device(device, queue).unwrap();
    state.upload_to_device(device, queue).unwrap();
    let buffer = |size| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("prediction fixture"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let low = buffer(64);
    let rates = buffer(64);
    let loads = buffer(128);
    let drive = buffer(64);
    let out = [buffer(64), buffer(32), buffer(64), buffer(32), buffer(4)];
    let dt = 1.0f32 / 2040.0;
    let damping = 0.01f32;
    let inputs = || ExperimentalBiasBindings {
        poses: state.body_q.device_buffer().unwrap(),
        position_low: &low,
        paired_rates: &rates,
        external_com_wrenches: &loads,
    };
    let outputs = || GeneralizedPredictionOutputs {
        paired_rates: &out[0],
        scalar_rates: &out[1],
        paired_delta: &out[2],
        scalar_delta: &out[3],
        status: &out[4],
    };
    let required = PreparedGeneralizedPrediction::required_bytes(4, 8).unwrap();
    assert!(
        PreparedGeneralizedPrediction::new(
            device,
            queue,
            &model,
            inputs(),
            &drive,
            outputs(),
            dt as f64,
            damping,
            required - 1
        )
        .is_err()
    );
    let predictor = PreparedGeneralizedPrediction::new(
        device,
        queue,
        &model,
        inputs(),
        &drive,
        outputs(),
        dt as f64,
        damping,
        required,
    )
    .unwrap();
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("prediction timings"),
        ty: wgpu::QueryType::Timestamp,
        count: 16,
    });
    let timing = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("prediction timestamps"),
        size: 128,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut worst = [0.0f64; 4];
    let mut load_words = Vec::new();
    for step in 0..8 {
        let orientation = Quat::from_euler(
            glam::EulerRot::XYZ,
            0.04 * step as f64,
            -0.03 * step as f64,
            0.07 * step as f64,
        );
        let q = state.joint_q.host_mut().unwrap();
        q[0] = 0.03 * step as f64;
        q[1] = -0.015 * step as f64;
        q[2] = 0.01 * step as f64;
        q[3..7].copy_from_slice(&orientation.to_array());
        q[7] = 0.11 * step as f64;
        q[8] = 0.025 * step as f64;
        for (i, v) in state.joint_qd.host_mut().unwrap().iter_mut().enumerate() {
            *v = (i + 1) as f64 * 0.2 + step as f64 * 0.01 + 2f64.powi(-29);
        }
        crate::sim::eval_fk(&model, &mut state).unwrap();
        state.upload_to_device(device, queue).unwrap();
        let lows: Vec<[f32; 4]> = state
            .body_q
            .host()
            .unwrap()
            .iter()
            .map(|p| {
                [
                    pair(p.position.x)[1],
                    pair(p.position.y)[1],
                    pair(p.position.z)[1],
                    0.0,
                ]
            })
            .collect();
        queue.write_buffer(&low, 0, bytemuck::cast_slice(&lows));
        let rate_values = state.joint_qd.host().unwrap();
        queue.write_buffer(
            &rates,
            0,
            bytemuck::cast_slice(&rate_values.iter().copied().map(pair).collect::<Vec<_>>()),
        );
        let external: Vec<_> = (0..4)
            .map(|i| {
                SpatialVector::new(
                    Vec3::new((i + 1) as f64 * 1.5, -0.25 * step as f64, 0.5),
                    Vec3::new(0.125, -0.25, 0.5 * (i + 1) as f64),
                )
            })
            .collect();
        load_words = external
            .iter()
            .flat_map(|f| {
                [
                    f.linear.x as f32,
                    f.linear.y as f32,
                    f.linear.z as f32,
                    0.0,
                    f.angular.x as f32,
                    f.angular.y as f32,
                    f.angular.z as f32,
                    0.0,
                ]
            })
            .collect();
        queue.write_buffer(&loads, 0, bytemuck::cast_slice(&load_words));
        let force: Vec<f64> = (0..8)
            .map(|i| {
                (i + 1) as f64 * (step + 1) as f64 * 1.25 * if i % 2 == 0 { 1.0 } else { -1.0 }
            })
            .collect();
        queue.write_buffer(
            &drive,
            0,
            bytemuck::cast_slice(&force.iter().copied().map(pair).collect::<Vec<_>>()),
        );
        let mass = crate::crba::mass_matrix(&model, &state, 0).unwrap();
        let bias = crate::rnea::inverse_dynamics_wrenches(&model, &state, 0, &[0.0; 8], &external)
            .unwrap()
            .efforts;
        let rhs: Vec<_> = force.iter().zip(&bias).map(|(f, b)| f - b).collect();
        let expected = solve(&mass, &rhs);
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.write_timestamp(&queries, 2 * step);
        let resources = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
        let before = device.get_internal_counters().hal;
        predictor.encode(&mut encoder);
        let after = device.get_internal_counters().hal;
        assert_eq!(before.buffers.read(), after.buffers.read());
        assert_eq!(before.bind_groups.read(), after.bind_groups.read());
        assert_eq!(
            before.compute_pipelines.read(),
            after.compute_pipelines.read()
        );
        assert_eq!(resources.finish().observations, 0);
        assert_eq!(transfers.finish(), Default::default());
        let count = dispatches.finish();
        assert_eq!(count.encoded_dispatches, 3);
        assert!(!count.overflowed);
        encoder.write_timestamp(&queries, 2 * step + 1);
        queue.submit([encoder.finish()]);
        assert_eq!(read(device, queue, &out[4]), [0]);
        let acceleration = read(device, queue, predictor.acceleration());
        let actual = decoded(&acceleration);
        let actual_bias = decoded(&read(device, queue, predictor.bias()));
        let actual_mass = read(device, queue, predictor.mass_matrix());
        let predicted = read(device, queue, &out[0]);
        let actual_rates = decoded(&predicted);
        let delta = read(device, queue, &out[2]);
        let deltas = decoded(&delta);
        assert_eq!(
            read(device, queue, &out[1]),
            predicted.chunks_exact(2).map(|x| x[0]).collect::<Vec<_>>()
        );
        assert_eq!(
            read(device, queue, &out[3]),
            delta.chunks_exact(2).map(|x| x[0]).collect::<Vec<_>>()
        );
        let force_scale = 1.0 + rhs.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let bias_scale = 1.0 + bias.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let mass_scale = 1.0 + mass.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        for i in 0..8 {
            worst[0] = worst[0].max((actual_bias[i] - bias[i]).abs() / bias_scale);
            worst[1] = worst[1].max(
                ((0..8).map(|j| mass[i * 8 + j] * actual[j]).sum::<f64>() - rhs[i]).abs()
                    / force_scale,
            );
            let target = (rate_values[i] + expected[i] * f64::from(dt))
                * f64::from((1.0 - damping * dt).max(0.0));
            worst[2] = worst[2].max((actual_rates[i] - target).abs());
            assert!((actual_rates[i] - rate_values[i] - deltas[i]).abs() < 1e-12);
        }
        for i in 0..64 {
            worst[3] = worst[3]
                .max((f64::from(f32::from_bits(actual_mass[i])) - mass[i]).abs() / mass_scale);
        }
        let jacobian_words = read(device, queue, predictor.com_jacobians());
        for body in 0..model.body_count {
            let reference = crate::jacobian::jacobian(
                &model,
                &state,
                0,
                body,
                model.body_com.host().unwrap()[body],
            )
            .unwrap();
            for (&word, expected) in jacobian_words[body * 48..(body + 1) * 48]
                .iter()
                .zip(reference)
            {
                let actual = f64::from(f32::from_bits(word));
                assert!(
                    actual.is_finite()
                        && (actual - expected).abs() / (1.0 + expected.abs()) <= 1e-6
                );
            }
        }
        std::fs::write(
            capture.join(format!("com-jacobians-{step}.bin")),
            bytemuck::cast_slice(&jacobian_words),
        )
        .unwrap();
        std::fs::write(
            capture.join(format!("rates-{step}.bin")),
            bytemuck::cast_slice(&predicted),
        )
        .unwrap();
        std::fs::write(
            capture.join(format!("delta-{step}.bin")),
            bytemuck::cast_slice(&delta),
        )
        .unwrap();
        std::fs::write(
            capture.join(format!("acceleration-{step}.bin")),
            bytemuck::cast_slice(&acceleration),
        )
        .unwrap();
    }
    eprintln!(
        "generalized_prediction_bias={:e} force={:e} rate={:e} matrix={:e}",
        worst[0], worst[1], worst[2], worst[3]
    );
    assert!(worst[0] <= 1e-6 && worst[1] <= 1e-6 && worst[2] <= 1e-5 && worst[3] <= 1e-6);
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.resolve_query_set(&queries, 0..16, &timing, 0);
    queue.submit([encoder.finish()]);
    let raw = read(device, queue, &timing);
    std::fs::write(capture.join("timestamps.bin"), bytemuck::cast_slice(&raw)).unwrap();
    let timestamps: Vec<u64> = raw
        .chunks_exact(2)
        .map(|x| u64::from(x[0]) | (u64::from(x[1]) << 32))
        .collect();
    let ms: Vec<f64> = timestamps
        .chunks_exact(2)
        .map(|t| {
            assert!(t[1] > t[0]);
            (t[1] - t[0]) as f64 * f64::from(queue.get_timestamp_period()) / 1e6
        })
        .collect();
    let mut measured = ms[1..].to_vec();
    measured.sort_by(f64::total_cmp);
    eprintln!(
        "generalized_prediction_gpu_median_ms={} samples={ms:?}",
        measured[3]
    );
    std::fs::write(capture.join("timings.json"),serde_json::to_vec_pretty(&serde_json::json!({"period_ns":queue.get_timestamp_period(),"samples_ms":ms,"median_ms":measured[3],"warmup":1})).unwrap()).unwrap();
    let previous: Vec<_> = out[..4].iter().map(|b| read(device, queue, b)).collect();
    let prior_jacobians = read(device, queue, predictor.com_jacobians());
    queue.write_buffer(&loads, 96, bytemuck::cast_slice(&[f32::NAN]));
    let mut encoder = device.create_command_encoder(&Default::default());
    predictor.encode(&mut encoder);
    queue.submit([encoder.finish()]);
    assert_eq!(read(device, queue, &out[4]), [32]);
    assert_eq!(
        read(device, queue, predictor.com_jacobians()),
        prior_jacobians
    );
    for (buffer, old) in out[..4].iter().zip(&previous) {
        assert_eq!(read(device, queue, buffer), *old);
    }
    queue.write_buffer(&loads, 0, bytemuck::cast_slice(&load_words));
    let mut encoder = device.create_command_encoder(&Default::default());
    predictor.reset_internal(&mut encoder);
    predictor.encode(&mut encoder);
    queue.submit([encoder.finish()]);
    assert_eq!(read(device, queue, &out[4]), [32]);
    for (buffer, old) in out[..4].iter().zip(&previous) {
        assert_eq!(read(device, queue, buffer), *old);
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    predictor.reset_internal(&mut encoder);
    encoder.clear_buffer(&out[4], 0, None);
    predictor.encode(&mut encoder);
    queue.submit([encoder.finish()]);
    assert_eq!(read(device, queue, &out[4]), [0]);
    for (buffer, old) in out[..4].iter().zip(&previous) {
        assert_eq!(read(device, queue, buffer), *old);
    }
    eprintln!("generalized_prediction_late_fault_preservation_shared_reset=pass");
}
