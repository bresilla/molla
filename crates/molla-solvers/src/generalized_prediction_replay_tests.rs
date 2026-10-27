use super::*;

fn values(value: &serde_json::Value, n: usize) -> Vec<f64> {
    let a = value.as_array().unwrap();
    assert_eq!(a.len(), n);
    a.iter()
        .map(|v| v.as_f64().filter(|v| v.is_finite()).unwrap())
        .collect()
}

#[test]
#[ignore = "requires f64 GPU and frozen baseline/native 50k frame-state inputs"]
fn native_prediction_actual_trajectory_replay() {
    use crate::math::{Quat, SpatialVector, Transform, Vec3};
    let start = std::time::Instant::now();
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_NATIVE_REPLAY_CAPTURE").unwrap());
    std::fs::create_dir(&root).unwrap();
    let inputs = [
        "MOLLA_NATIVE_REPLAY_BASELINE",
        "MOLLA_NATIVE_REPLAY_CANDIDATE",
    ]
    .map(|key| std::fs::read_to_string(std::env::var_os(key).unwrap()).unwrap());
    let engine = crate::compute::Engine::new_gpu_with_features(wgpu::Features::SHADER_F64).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut scene, _) = crate::soil_tractor::build_soil_tractor_scene().unwrap();
    scene.upload_model_to_device(device, queue).unwrap();
    let model = scene.model();
    assert_eq!((model.body_count, model.joint_dof_total), (6, 10));
    let body_order: Vec<_> = scene
        .bodies()
        .map(|(handle, _)| {
            (
                format!("{handle:?}"),
                scene.body_index(handle).unwrap().index(),
            )
        })
        .collect();
    let mut state = scene.state().clone_host().unwrap();
    state.upload_to_device(device, queue).unwrap();
    let buffer = |size| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("actual trajectory replay"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let low = buffer(96);
    let rates = buffer(80);
    let loads = buffer(192);
    let drive = buffer(80);
    let legacy_mass = buffer(400);
    let out = [buffer(80), buffer(40), buffer(80), buffer(40), buffer(4)];
    let dt = (1.0f64 / 60.0 / 34.0) as f32;
    let predictor = PreparedGeneralizedPrediction::new(
        device,
        queue,
        model,
        ExperimentalBiasBindings {
            poses: state.body_q.device_buffer_current().unwrap(),
            position_low: &low,
            paired_rates: &rates,
            external_com_wrenches: &loads,
        },
        &drive,
        GeneralizedPredictionOutputs {
            paired_rates: &out[0],
            scalar_rates: &out[1],
            paired_delta: &out[2],
            scalar_delta: &out[3],
            status: &out[4],
        },
        f64::from(dt),
        0.0,
        PreparedGeneralizedPrediction::required_bytes(6, 10).unwrap(),
    )
    .unwrap();
    let crba = molla_kernels::MassMatrixKernel::new(&engine, model)
        .unwrap()
        .prepare_on_device(device, queue, model, &state, &legacy_mass)
        .unwrap();
    let reference_operator = molla_kernels::ExperimentalBiasKernel::new(
        device,
        queue,
        model,
        ExperimentalBiasBindings {
            poses: state.body_q.device_buffer_current().unwrap(),
            position_low: &low,
            paired_rates: &rates,
            external_com_wrenches: &loads,
        },
        1636,
    )
    .unwrap();
    queue.write_buffer(&loads, 0, bytemuck::cast_slice(&[0.0f32; 48]));
    let mut records = Vec::new();
    let mut worst = [0.0f64; 7];
    for (route, text) in inputs.iter().enumerate() {
        let snapshots: Vec<serde_json::Value> = text
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(snapshots.len(), 33);
        for (frame, snapshot) in snapshots.iter().enumerate() {
            assert_eq!(snapshot["frame"].as_u64(), Some(frame as u64));
            assert_eq!(snapshot["committed_frame"].as_u64(), Some(frame as u64 + 1));
            state
                .joint_q
                .host_mut()
                .unwrap()
                .copy_from_slice(&values(&snapshot["joint_q"], 11));
            state
                .joint_qd
                .host_mut()
                .unwrap()
                .copy_from_slice(&values(&snapshot["joint_qd"], 10));
            assert_eq!(snapshot["bodies"].as_array().unwrap().len(), 6);
            for (body, (expected, index)) in snapshot["bodies"]
                .as_array()
                .unwrap()
                .iter()
                .zip(&body_order)
            {
                assert_eq!(body["body"].as_str(), Some(expected.as_str()));
                let p = values(&body["position"], 3);
                let q = values(&body["rotation"], 4);
                state.body_q.host_mut().unwrap()[*index] =
                    Transform::new(Vec3::from_slice(&p), Quat::from_slice(&q));
            }
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
            let original_rates = state.joint_qd.host().unwrap();
            queue.write_buffer(
                &rates,
                0,
                bytemuck::cast_slice(&original_rates.iter().copied().map(pair).collect::<Vec<_>>()),
            );
            let cpu_mass = crate::crba::mass_matrix(model, &state, 0).unwrap();
            let cpu_jacobians: Vec<_> = (0..6)
                .flat_map(|body| {
                    crate::jacobian::jacobian(
                        model,
                        &state,
                        0,
                        body,
                        model.body_com.host().unwrap()[body],
                    )
                    .unwrap()
                })
                .collect();
            let cpu_bias = crate::rnea::inverse_dynamics_wrenches(
                model,
                &state,
                0,
                &[0.0; 10],
                &[SpatialVector::ZERO; 6],
            )
            .unwrap()
            .efforts;
            let mut unloaded = Vec::new();
            let mut metrics = [0.0f64; 7];
            for driven in [false, true] {
                let force: Vec<f64> = (0..10)
                    .map(|i| if driven && i >= 6 { -300.0 } else { 0.0 })
                    .collect();
                queue.write_buffer(
                    &drive,
                    0,
                    bytemuck::cast_slice(&force.iter().copied().map(pair).collect::<Vec<_>>()),
                );
                let mut encoder = device.create_command_encoder(&Default::default());
                predictor.encode(&mut encoder);
                crba.encode(&mut encoder);
                reference_operator.encode(&mut encoder);
                queue.submit([encoder.finish()]);
                assert_eq!(read(device, queue, &out[4]), [0]);
                let acceleration = read(device, queue, predictor.acceleration());
                let a = decoded(&acceleration);
                let bias_words = read(device, queue, predictor.bias());
                assert_eq!(read(device, queue, reference_operator.status()), [0]);
                assert_eq!(bias_words, read(device, queue, reference_operator.bias()));
                let bias = decoded(&bias_words);
                let mass_words = read(device, queue, predictor.mass_matrix());
                let jacobian_words = read(device, queue, predictor.com_jacobians());
                assert_eq!(jacobian_words.len(), cpu_jacobians.len());
                for (&word, &reference) in jacobian_words.iter().zip(&cpu_jacobians) {
                    let value = f64::from(f32::from_bits(word));
                    assert!(value.is_finite());
                    metrics[6] =
                        metrics[6].max((value - reference).abs() / (1.0 + reference.abs()));
                }
                assert_eq!(
                    mass_words,
                    read(device, queue, reference_operator.mass_matrix())
                );
                let paired_mass = read(device, queue, predictor.paired_mass_matrix());
                assert_eq!(
                    paired_mass
                        .chunks_exact(2)
                        .map(|v| v[0])
                        .collect::<Vec<_>>(),
                    mass_words
                );
                assert!(decoded(&paired_mass).iter().all(|v| v.is_finite()));
                let mass: Vec<_> = mass_words
                    .iter()
                    .map(|&v| f64::from(f32::from_bits(v)))
                    .collect();
                let old_words = read(device, queue, &legacy_mass);
                let old_mass: Vec<_> = old_words
                    .iter()
                    .map(|&v| f64::from(f32::from_bits(v)))
                    .collect();
                let predicted = read(device, queue, &out[0]);
                let qd = decoded(&predicted);
                assert!(
                    a.iter()
                        .chain(&bias)
                        .chain(&mass)
                        .chain(&old_mass)
                        .chain(&qd)
                        .all(|v| v.is_finite())
                );
                let rhs: Vec<_> = force.iter().zip(&cpu_bias).map(|(f, b)| f - b).collect();
                let expected = solve(&cpu_mass, &rhs);
                let old = solve(&old_mass, &rhs);
                let native_mass_only = solve(&mass, &rhs);
                let scale = 1.0 + rhs.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                let bscale = 1.0 + cpu_bias.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                let mscale = 1.0 + cpu_mass.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                for i in 0..10 {
                    metrics[0] = metrics[0].max((bias[i] - cpu_bias[i]).abs() / bscale);
                    metrics[1] = metrics[1].max(
                        ((0..10).map(|j| cpu_mass[10 * i + j] * a[j]).sum::<f64>() - rhs[i]).abs()
                            / scale,
                    );
                    metrics[2] = metrics[2]
                        .max((qd[i] - original_rates[i] - expected[i] * f64::from(dt)).abs());
                    metrics[5] =
                        metrics[5].max((old[i] - native_mass_only[i]).abs() * f64::from(dt));
                    if driven {
                        metrics[4] = metrics[4].max(
                            ((0..10)
                                .map(|j| cpu_mass[10 * i + j] * (a[j] - unloaded[j]))
                                .sum::<f64>()
                                - force[i])
                                .abs()
                                / 301.0,
                        );
                    }
                }
                for i in 0..100 {
                    metrics[3] = metrics[3].max((mass[i] - cpu_mass[i]).abs() / mscale);
                }
                let prefix = format!("route-{route}-frame-{frame:02}-drive-{}", u8::from(driven));
                for (name, words) in [
                    ("rates", predicted),
                    ("acceleration", acceleration),
                    ("native-mass", mass_words),
                    ("legacy-mass", old_words),
                    ("paired-mass", paired_mass),
                    ("bias", bias_words),
                    ("com-jacobians", jacobian_words),
                ] {
                    std::fs::write(
                        root.join(format!("{prefix}-{name}.bin")),
                        bytemuck::cast_slice(&words),
                    )
                    .unwrap();
                }
                if !driven {
                    unloaded = a;
                }
            }
            for i in 0..7 {
                worst[i] = worst[i].max(metrics[i]);
            }
            records.push(serde_json::json!({"route":route,"frame":frame,"bias":metrics[0],"force":metrics[1],"rate":metrics[2],"matrix":metrics[3],"increment":metrics[4],"mass_only_rate_difference":metrics[5],"com_jacobian":metrics[6]}));
        }
    }
    std::fs::write(root.join("result.json"),serde_json::to_vec_pretty(&serde_json::json!({
        "acceptance":false,"scope":"matched reconstructed published frame states, not raw substep/contact replay",
        "snapshots":66,"evaluations":132,"worst":worst,"records":records,
        "exact_f32_mirror_and_bias":true,"prediction_mass_encoding":"f32x2",
    })).unwrap()).unwrap();
    eprintln!(
        "native_trajectory_replay_worst={worst:?} elapsed={:?}",
        start.elapsed()
    );
    assert!(
        worst[0] <= 1e-6
            && worst[1] <= 1e-6
            && worst[2] <= 1e-5
            && worst[3] <= 1e-6
            && worst[4] <= 1e-6
            && worst[6] <= 1e-6
    );
}
