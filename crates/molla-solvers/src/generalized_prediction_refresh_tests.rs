use super::*;

fn normalized_error(words: &[u32], reference: &[f64]) -> f64 {
    assert_eq!(words.len(), reference.len());
    words
        .iter()
        .zip(reference)
        .map(|(&word, &expected)| {
            let value = f64::from(f32::from_bits(word));
            assert!(value.is_finite() && expected.is_finite());
            (value - expected).abs() / (1.0 + expected.abs())
        })
        .fold(0.0, f64::max)
}

#[test]
fn refreshed_machine_reference_contract() {
    assert_eq!(normalized_error(&[1.0f32.to_bits()], &[1.0]), 0.0);
    assert!(normalized_error(&[1.01f32.to_bits()], &[1.0]) > 1e-6);
    assert!(std::panic::catch_unwind(|| normalized_error(&[f32::NAN.to_bits()], &[0.0])).is_err());
}

#[test]
#[ignore = "requires f64 GPU and retained tractor first-substep packets; no soil solver"]
fn refreshed_machine_first_input_operators() {
    use crate::math::{Quat, Transform, Vec3};
    use molla_kernels::{JacobianKernel, JacobianRequest, MassMatrixKernel};
    let start = std::time::Instant::now();
    let root = std::path::PathBuf::from(std::env::var_os("MOLLA_REFRESH_CAPTURE").unwrap());
    std::fs::create_dir(&root).unwrap();
    let engine = crate::compute::Engine::new_gpu_with_features(wgpu::Features::SHADER_F64).unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let (mut scene, _) = crate::soil_tractor::build_soil_tractor_scene().unwrap();
    scene.upload_model_to_device(device, queue).unwrap();
    let model = scene.model();
    let mut control = crate::sim::Control::for_model(model);
    control.upload_to_device(device, queue).unwrap();
    let mut states = [
        scene.state().clone_host().unwrap(),
        scene.state().clone_host().unwrap(),
    ];
    for state in &mut states {
        state.upload_to_device(device, queue).unwrap();
    }
    let mut drift =
        crate::featherstone::FeatherstoneSolver::new_gpu_for_native_prediction(&engine, model)
            .unwrap();
    drift
        .prepare_gpu_coordinate_compensation(&states.each_ref())
        .unwrap();
    drift
        .prepare_gpu_paired_rates(&states.each_ref(), 1 << 20)
        .unwrap();
    for state in &states {
        drift.seed_gpu_coordinate_compensation(state).unwrap();
        drift.seed_gpu_paired_rates(state).unwrap();
    }
    let [input, refreshed] = &mut states;
    drift
        .refresh_gpu(model, input, &control, refreshed)
        .unwrap();
    assert_eq!(read(device, queue, drift.gpu_status_buffer().unwrap()), [0]);
    let poses = read(
        device,
        queue,
        refreshed.body_q.device_buffer_current().unwrap(),
    );
    let low = read(
        device,
        queue,
        drift.gpu_position_compensation_buffer(refreshed).unwrap(),
    );
    let coordinates = read(
        device,
        queue,
        input.joint_q.device_buffer_current().unwrap(),
    );
    let coordinate_low = read(
        device,
        queue,
        drift.gpu_coordinate_compensation_buffer(input).unwrap(),
    );
    let f = |v: u32| f64::from(f32::from_bits(v));
    let mut cpu = scene.state().clone_host().unwrap();
    for (body, pose) in cpu.body_q.host_mut().unwrap().iter_mut().enumerate() {
        let p = Vec3::from_array(std::array::from_fn(|axis| {
            f(poses[8 * body + axis]) + f(low[4 * body + axis])
        }));
        let q = Quat::from_array(std::array::from_fn(|axis| f(poses[8 * body + 4 + axis])));
        *pose = Transform::new(p, q);
    }
    for (i, value) in cpu.joint_q.host_mut().unwrap().iter_mut().enumerate() {
        *value = f(coordinates[i]) + f(coordinate_low[i]);
    }
    let cpu_mass = crate::crba::mass_matrix(model, &cpu, 0).unwrap();
    let cpu_jac: Vec<_> = (0..6)
        .flat_map(|body| {
            crate::jacobian::jacobian(model, &cpu, 0, body, model.body_com.host().unwrap()[body])
                .unwrap()
        })
        .collect();
    let buffer = |size| {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("refreshed machine reference"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    };
    let loads = buffer(192);
    let drive = buffer(80);
    let efforts: Vec<_> = (0..10)
        .map(|i| pair(if i >= 6 { -300.0 } else { 0.0 }))
        .collect();
    queue.write_buffer(&drive, 0, bytemuck::cast_slice(&efforts));
    let outputs = [buffer(80), buffer(40), buffer(80), buffer(40), buffer(4)];
    let prediction = PreparedGeneralizedPrediction::new(
        device,
        queue,
        model,
        ExperimentalBiasBindings {
            poses: refreshed.body_q.device_buffer_current().unwrap(),
            position_low: drift.gpu_position_compensation_buffer(refreshed).unwrap(),
            paired_rates: drift.gpu_paired_rates_buffer(input).unwrap(),
            external_com_wrenches: &loads,
        },
        &drive,
        GeneralizedPredictionOutputs {
            paired_rates: &outputs[0],
            scalar_rates: &outputs[1],
            paired_delta: &outputs[2],
            scalar_delta: &outputs[3],
            status: &outputs[4],
        },
        1.0 / 60.0 / 34.0,
        0.0,
        PreparedGeneralizedPrediction::required_bytes(6, 10).unwrap(),
    )
    .unwrap();
    let legacy_mass = buffer(400);
    let crba = MassMatrixKernel::new(&engine, model)
        .unwrap()
        .prepare_on_device(device, queue, model, refreshed, &legacy_mass)
        .unwrap();
    let legacy_jac = buffer(1440);
    let requests: Vec<_> = (0..6)
        .map(|body| JacobianRequest {
            articulation: 0,
            target_body: body,
            output_offset: body * 60,
            body_q_offset: 0,
            point_local: model.body_com.host().unwrap()[body as usize]
                .to_array()
                .map(|v| v as f32)
                .into_iter()
                .chain([0.0])
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
        })
        .collect();
    let jac = JacobianKernel::new(&engine, model)
        .unwrap()
        .prepare_with_body_q_on_device(
            device,
            queue,
            model,
            refreshed.body_q.device_buffer_current().unwrap(),
            &requests,
            &legacy_jac,
        )
        .unwrap()
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    prediction.encode(&mut encoder);
    crba.encode(&mut encoder);
    jac.encode(&mut encoder);
    queue.submit([encoder.finish()]);
    assert_eq!(read(device, queue, &outputs[4]), [0]);
    let mass = [
        read(device, queue, &legacy_mass),
        read(device, queue, prediction.mass_matrix()),
    ];
    let jacobians = [
        read(device, queue, &legacy_jac),
        read(device, queue, prediction.com_jacobians()),
    ];
    let rates = read(device, queue, &outputs[0]);
    let mut reports = Vec::new();
    for (arm, key) in ["MOLLA_REFRESH_FEATHERSTONE", "MOLLA_REFRESH_NATIVE"]
        .iter()
        .enumerate()
    {
        let packet = std::path::PathBuf::from(std::env::var_os(key).unwrap());
        let bytes = std::fs::read(packet.join("newton.bin")).unwrap();
        let metadata: serde_json::Value =
            serde_json::from_slice(&std::fs::read(packet.join("capture.json")).unwrap()).unwrap();
        assert_eq!(metadata["blake3"], blake3::hash(&bytes).to_hex().as_str());
        assert_eq!(bytes.len(), 128 + 400 + 80 + 448 * 4096);
        let packet_words: Vec<_> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        assert_eq!(
            (
                packet_words[0],
                packet_words[1],
                packet_words[2],
                packet_words[7]
            ),
            (1, 10, 4096, 31)
        );
        let different_mass = mass[arm]
            .iter()
            .zip(&packet_words[32..132])
            .filter(|(a, b)| a != b)
            .count();
        let scale = 1.0 + cpu_mass.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let mass_error = mass[arm]
            .iter()
            .zip(&cpu_mass)
            .map(|(&v, &r)| (f(v) - r).abs() / scale)
            .fold(0.0, f64::max);
        let jacobian_error = normalized_error(&jacobians[arm], &cpu_jac);
        reports.push(serde_json::json!({"route":key,"different_capture_mass_words":different_mass,
            "cpu_mass_global_error":mass_error,"cpu_com_jacobian_error":jacobian_error,
            "mass_9_0":f(mass[arm][90]), "cpu_mass_9_0":cpu_mass[90],
            "different_native_rate_words":if arm==1 {Some(rates.iter().zip(&packet_words[132..152]).filter(|(a,b)| a != b).count())} else {None}}));
        for (name, words) in [("mass", &mass[arm]), ("com-jacobians", &jacobians[arm])] {
            std::fs::write(
                root.join(format!("arm-{arm}-{name}.bin")),
                bytemuck::cast_slice(words),
            )
            .unwrap();
        }
    }
    for (name, words) in [
        ("poses", poses),
        ("position-low", low),
        ("coordinates", coordinates),
        ("coordinate-low", coordinate_low),
        ("native-free-rates", rates),
    ] {
        std::fs::write(
            root.join(format!("{name}.bin")),
            bytemuck::cast_slice(&words),
        )
        .unwrap();
    }
    let result = serde_json::json!({"reports":reports,"elapsed_seconds":start.elapsed().as_secs_f64(),
        "scope":"machine-only initial refresh reconstruction; exact packet mass and native free-rate checks; CPU reference uses refreshed poses and authored model parameters; no contact point geometry, later states, trajectory or performance acceptance"});
    std::fs::write(
        root.join("result.json"),
        serde_json::to_vec_pretty(&result).unwrap(),
    )
    .unwrap();
    eprintln!("{result}");
    for report in &reports {
        assert_eq!(report["different_capture_mass_words"], 0);
        assert!(report["cpu_mass_global_error"].as_f64().unwrap() <= 1e-6);
        assert!(report["cpu_com_jacobian_error"].as_f64().unwrap() <= 1e-6);
    }
    assert_eq!(reports[1]["different_native_rate_words"], 0);
}
