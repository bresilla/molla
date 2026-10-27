use super::*;
use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder};

#[test]
#[ignore = "requires real GPU; paired external refresh/displacement/drift comparison"]
fn external_prediction_modes_preserve_all_state_words() {
    let engine =
        Engine::new_gpu_with_features(FeatherstoneKernel::paired_drift_required_features())
            .unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut builder = ModelBuilder::new();
    for _ in 0..65 {
        builder.begin_articulation();
        let body = builder.add_body(
            BodyParams::new()
                .with_mass(2.0)
                .with_inertia(Mat3::IDENTITY),
        );
        builder
            .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
    }
    let mut model = builder.build().unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut initial = State::for_model(&model).unwrap();
    for i in 0..65 {
        let velocity = (i + 1) as f64 * 0.125 + 2f64.powi(-29);
        initial.joint_qd.host_mut().unwrap()[i * 6] = velocity;
        initial.joint_qd.host_mut().unwrap()[i * 6 + 4] = 0.125 + 2f64.powi(-28);
        initial.body_qd.host_mut().unwrap()[i] = SpatialVector {
            linear: Vec3::X * velocity,
            angular: Vec3::Y * (0.125 + 2f64.powi(-28)),
        };
    }
    let mut control = Control::for_model(&model);
    control.joint_force.host_mut().unwrap().fill(123.0);
    control.upload_to_device(device, queue).unwrap();
    let displacement = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("external prediction displacement"),
        size: 65 * 6 * 4,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let values: Vec<f32> = (0..65)
        .flat_map(|i| {
            if i % 3 == 0 {
                [0.0; 6]
            } else {
                [(i + 1) as f32 * 0.001, 0.0, 0.0, 0.002, 0.0, 0.0]
            }
        })
        .collect();
    queue.write_buffer(&displacement, 0, bytemuck::cast_slice(&values));
    let mut reference = None;
    for passes in [
        FeatherstonePasses::PairedUnrestricted,
        FeatherstonePasses::PairedExternalPrediction,
    ] {
        let mut states = [initial.clone_host().unwrap(), initial.clone_host().unwrap()];
        for state in &mut states {
            state.upload_to_device(device, queue).unwrap();
        }
        let (offsets, rows) = kernel_configuration(&model);
        let program = FeatherstoneProgram::with_rate_precision(
            device, 1, rows, false, false, false, passes, true,
        );
        let mut kernel =
            FeatherstoneKernel::from_program(device, queue, &model, offsets, rows, &program);
        kernel
            .prepare_coordinate_compensation(device, &[&states[0], &states[1]])
            .unwrap();
        kernel
            .prepare_paired_rates(device, queue, &[&states[0], &states[1]], 208 * 65)
            .unwrap();
        kernel
            .set_active_articulations(queue, &(0..65).map(|i| i % 5 != 1).collect::<Vec<_>>())
            .unwrap();
        if passes == FeatherstonePasses::PairedExternalPrediction {
            let (a, b) = states.split_at_mut(1);
            let before =
                rounding_tests::words(device, queue, b[0].joint_q.device_buffer_current().unwrap());
            let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
            let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
            let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
            assert!(
                kernel
                    .dispatch(
                        device,
                        queue,
                        &model,
                        &a[0],
                        &control,
                        &Contacts::with_capacity(0),
                        0.0,
                        0.001,
                        true,
                        true,
                        &mut b[0],
                        None
                    )
                    .is_err()
            );
            assert_eq!(allocations.finish().observations, 0);
            let traffic = transfers.finish();
            assert_eq!(traffic.upload_bytes + traffic.readback_bytes, 0);
            let dispatches = dispatches.finish();
            assert!(!dispatches.overflowed);
            assert_eq!(dispatches.encoded_dispatches, 0);
            assert_eq!(
                rounding_tests::words(device, queue, b[0].joint_q.device_buffer_current().unwrap()),
                before
            );
        }
        let mut records = Vec::new();
        for (step, mode) in [3, 2, 1, 3, 2, 1].into_iter().enumerate() {
            let (a, b) = states.split_at_mut(1);
            let (input, output) = if step % 2 == 0 {
                (&a[0], &mut b[0])
            } else {
                (&b[0], &mut a[0])
            };
            match mode {
                1 => kernel
                    .drift(device, queue, &model, input, &control, 0.001, true, output)
                    .unwrap(),
                2 => kernel
                    .apply_position_displacement(
                        device,
                        queue,
                        &model,
                        input,
                        &control,
                        &displacement,
                        output,
                    )
                    .unwrap(),
                3 => kernel
                    .refresh(device, queue, &model, input, &control, output)
                    .unwrap(),
                _ => unreachable!(),
            }
            assert_eq!(
                rounding_tests::words(device, queue, kernel.status_buffer()),
                [0]
            );
            for buffer in [
                output.body_q.device_buffer_current().unwrap(),
                output.body_qd.device_buffer_current().unwrap(),
                output.joint_q.device_buffer_current().unwrap(),
                output.joint_qd.device_buffer_current().unwrap(),
                kernel.coordinate_compensation_buffer(output).unwrap(),
                kernel.position_compensation_buffer(output).unwrap(),
                kernel.paired_rates_buffer(output).unwrap(),
                kernel.body_velocity_compensation_buffer(output).unwrap(),
                kernel.paired_velocity_delta_buffer().unwrap(),
                kernel.velocity_delta_buffer(),
            ] {
                records.push(rounding_tests::words(device, queue, buffer));
            }
        }
        if let Some(reference) = &reference {
            assert_eq!(&records, reference);
            eprintln!(
                "external prediction: {} buffers / {} exact words; 65 articulations, 6 mixed modes",
                records.len(),
                records.iter().map(Vec::len).sum::<usize>()
            );
        } else {
            reference = Some(records);
        }
    }
}
