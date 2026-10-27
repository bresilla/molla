use super::*;
use molla_compute::Engine;
use molla_core::BodyId;
use molla_math::{Mat3, Quat, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, JointType, ModelBuilder};

#[test]
#[ignore = "requires real GPU"]
fn paired_body_velocities_match_f64_articulated_kinematics() {
    let engine = Engine::new_gpu().unwrap();
    body_velocities(&engine, false, false);
}

#[test]
#[ignore = "requires real GPU; paired eager/lazy drift programs"]
fn lazy_body_anchors_match_exact_state_and_f64_kinematics() {
    let engine = Engine::new_gpu().unwrap();
    let reference = body_velocities(&engine, false, false);
    let candidate = body_velocities(&engine, true, false);
    assert_eq!(reference.len(), 48);
    assert_eq!(reference, candidate);
}

#[test]
#[ignore = "requires native f64 GPU; stronger paired reference and exact full-state oracle"]
fn paired_native_sum_matches_exact_state_and_f64_kinematics() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_F64).unwrap();
    let reference = body_velocities(&engine, false, false);
    let candidate = body_velocities(&engine, false, true);
    assert_eq!(reference.len(), 48);
    for (case, (a, b)) in reference.iter().zip(&candidate).enumerate() {
        let differences: Vec<_> = a
            .iter()
            .zip(b)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .collect();
        for (word, (a, b)) in differences.iter().take(8) {
            eprintln!(
                "paired native sum case={case} word={word} reference={a:#010x} candidate={b:#010x}"
            );
        }
        assert_eq!(a, b, "paired native sum exact state case={case}");
    }
}

fn body_velocities(engine: &Engine, lazy: bool, native: bool) -> Vec<Vec<u32>> {
    body_velocities_with_geometry(engine, lazy, native, false, 2)
}

#[test]
#[ignore = "requires native f64 GPU; geometry-only exact, physical and lifecycle coverage"]
fn paired_native_geometry_matches_exact_state_and_f64_kinematics() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_F64).unwrap();
    let reference = body_velocities_with_geometry(&engine, false, false, false, 6);
    let candidate = body_velocities_with_geometry(&engine, false, true, true, 6);
    assert_eq!(reference.len(), 48);
    assert_eq!(reference, candidate);
}

#[test]
#[ignore = "requires native f64 GPU; combined geometry/lazy exact state and f64 kinematics"]
fn native_geometry_lazy_matches_exact_state_and_f64_kinematics() {
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_F64).unwrap();
    let reference = body_velocities_with_geometry(&engine, false, true, true, 6);
    let candidate = body_velocities_with_geometry(&engine, true, true, true, 6);
    assert_eq!(reference.len(), 48);
    assert_eq!(candidate.len(), 48);
    let root =
        std::path::PathBuf::from(std::env::var_os("MOLLA_NATIVE_LAZY_ORACLE_CAPTURE").unwrap());
    std::fs::create_dir(&root).unwrap();
    for (variant, states) in [&reference, &candidate].into_iter().enumerate() {
        for (case, state) in states.iter().enumerate() {
            std::fs::write(
                root.join(format!("case-{case:02}-variant-{variant}.bin")),
                bytemuck::cast_slice(state),
            )
            .unwrap();
        }
    }
    for (case, (a, b)) in reference.iter().zip(&candidate).enumerate() {
        assert_eq!(a.len(), b.len());
        let mismatch = a.iter().zip(b).position(|(a, b)| a != b);
        assert!(
            mismatch.is_none(),
            "case={case} first differing word={mismatch:?}"
        );
    }
}

fn body_velocities_with_geometry(
    engine: &Engine,
    lazy: bool,
    native: bool,
    geometry: bool,
    capacity: u32,
) -> Vec<Vec<u32>> {
    assert!(!geometry || native);
    assert!(matches!(capacity, 2 | 6));
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let construction = std::time::Instant::now();
    eprintln!(
        "paired body velocity drift program construction lazy={lazy} native={native} geometry={geometry} capacity={capacity}"
    );
    let program = FeatherstoneProgram::with_rate_precision(
        device,
        capacity,
        1,
        false,
        native,
        false,
        if geometry && lazy {
            FeatherstonePasses::UnrestrictedDriftNativeGeometryLazyAnchors
        } else if geometry {
            FeatherstonePasses::UnrestrictedDriftNativeGeometry
        } else if lazy {
            FeatherstonePasses::UnrestrictedDriftLazyAnchors
        } else {
            FeatherstonePasses::UnrestrictedDriftOnly
        },
        true,
    );
    eprintln!(
        "paired body velocity drift program ready lazy={lazy} native={native} geometry={geometry} capacity={capacity} construction_ms={}",
        construction.elapsed().as_secs_f64() * 1000.0
    );
    let mut worst = 0.0f64;
    let mut cases = 0;
    let mut records = Vec::new();
    for kind in [
        JointType::Prismatic,
        JointType::Revolute,
        JointType::Ball,
        JointType::Fixed,
        JointType::Free,
        JointType::Distance,
        JointType::D6,
        JointType::Cable,
    ] {
        for reversed in [false, true] {
            for boundary in [0u32, 1, 2] {
                let mut builder = ModelBuilder::new();
                let root = builder.add_body(
                    BodyParams::new()
                        .with_mass(2.0)
                        .with_inertia(Mat3::IDENTITY)
                        .with_com(Vec3::new(0.125, -0.25, 0.5)),
                );
                let child = builder.add_body(
                    BodyParams::new()
                        .with_mass(1.0)
                        .with_inertia(Mat3::IDENTITY)
                        .with_com(Vec3::new(-0.25, 0.125, 0.5)),
                );
                builder
                    .add_joint_free(BodyId::NONE, root, Transform::IDENTITY, Transform::IDENTITY)
                    .unwrap();
                let xp = Transform::new(Vec3::new(0.5, 1.25, -0.75), Quat::from_rotation_y(0.31));
                let xc = Transform::new(Vec3::new(-0.125, 0.25, 0.5), Quat::from_rotation_x(-0.23));
                let axis = Vec3::new(1., 2., 3.).normalize();
                match kind {
                    JointType::Prismatic => builder.add_joint_prismatic(root, child, xp, xc, axis),
                    JointType::Revolute => builder.add_joint_revolute(root, child, xp, xc, axis),
                    JointType::Ball => builder.add_joint_ball(root, child, xp, xc),
                    JointType::Fixed => builder.add_joint_fixed(root, child, xp, xc),
                    JointType::Free => builder.add_joint_free(root, child, xp, xc),
                    JointType::Distance => builder.add_joint_distance(root, child, xp, xc),
                    JointType::D6 => builder.add_joint_d6(root, child, xp, xc),
                    JointType::Cable => {
                        builder.add_joint_cable(root, child, xp, xc, 0., 0., 0., 0.)
                    }
                }
                .unwrap();
                let mut model = builder.build().unwrap();
                model.joint_reversed.host_mut().unwrap()[1] = u32::from(reversed);
                model.body_flags.host_mut().unwrap()[0] = boundary;
                model.upload_to_device(device, queue).unwrap();
                let mut initial = State::for_model(&model).unwrap();
                for (i, value) in initial.joint_qd.host_mut().unwrap().iter_mut().enumerate() {
                    *value = 0.25 * (i + 1) as f64 + 2f64.powi(-29) * (i + 1) as f64;
                }
                let offset = model.joint_coord_offset.host().unwrap()[1] as usize;
                let q = initial.joint_q.host_mut().unwrap();
                q[0] = 0.25 + 2f64.powi(-28);
                q[1] = -0.125;
                q[2] = 0.5;
                q[3..7].copy_from_slice(&Quat::from_rotation_z(0.17).to_array());
                match kind {
                    JointType::Prismatic | JointType::Revolute => q[offset] = 0.19 + 2f64.powi(-29),
                    JointType::Cable => {
                        q[offset] = 0.27;
                        q[offset + 1] = -0.13;
                    }
                    JointType::Ball => q[offset..offset + 4]
                        .copy_from_slice(&Quat::from_rotation_y(0.21).to_array()),
                    JointType::Free | JointType::Distance | JointType::D6 => {
                        q[offset..offset + 3].copy_from_slice(&[0.125, -0.25, 0.375]);
                        q[offset + 3..offset + 7]
                            .copy_from_slice(&Quat::from_rotation_y(-0.21).to_array());
                    }
                    JointType::Fixed => {}
                }
                initial.body_qd.host_mut().unwrap()[0] = SpatialVector::new(
                    Vec3::new(1.0 + 2f64.powi(-29), 0.25, -0.5),
                    Vec3::new(0.125, 0.25 + 2f64.powi(-28), -0.125),
                );
                let mut reference = initial.clone_host().unwrap();
                molla_sim::eval_fk(&model, &mut reference).unwrap();
                molla_sim::eval_body_velocities(&model, &mut reference).unwrap();
                let mut states = [initial.clone_host().unwrap(), initial.clone_host().unwrap()];
                for state in &mut states {
                    state.upload_to_device(device, queue).unwrap();
                }
                let mut control = Control::for_model(&model);
                control.upload_to_device(device, queue).unwrap();
                let (offsets, rows) = kernel_configuration(&model);
                let mut kernel = FeatherstoneKernel::from_program(
                    device, queue, &model, offsets, rows, &program,
                );
                kernel
                    .prepare_coordinate_compensation(device, &[&states[0], &states[1]])
                    .unwrap();
                for state in &states {
                    kernel.seed_coordinate_compensation(queue, state).unwrap();
                }
                let budget = model.joint_dof_total as u64 * 24 + model.body_count as u64 * 64;
                kernel
                    .prepare_paired_rates(device, queue, &[&states[0], &states[1]], budget)
                    .unwrap();
                assert_eq!(kernel.paired_rate_owned_bytes(), budget);
                let (input, output) = states.split_at_mut(1);
                kernel
                    .drift(
                        device,
                        queue,
                        &model,
                        &input[0],
                        &control,
                        0.0,
                        true,
                        &mut output[0],
                    )
                    .unwrap();
                assert_eq!(
                    rounding_tests::words(device, queue, kernel.status_buffer()),
                    [0]
                );
                let high = rounding_tests::words(
                    device,
                    queue,
                    output[0].body_qd.device_buffer_current().unwrap(),
                );
                let low = rounding_tests::words(
                    device,
                    queue,
                    kernel
                        .body_velocity_compensation_buffer(&output[0])
                        .unwrap(),
                );
                for (body, velocity) in reference.body_qd.host().unwrap().iter().enumerate() {
                    let expected: Vec<_> = velocity
                        .linear
                        .to_array()
                        .into_iter()
                        .chain([0.0])
                        .chain(velocity.angular.to_array())
                        .chain([0.0])
                        .collect();
                    for lane in 0..8 {
                        let actual = f32::from_bits(high[body * 8 + lane]) as f64
                            + f32::from_bits(low[body * 8 + lane]) as f64;
                        let error = (actual - expected[lane]).abs();
                        worst = worst.max(error);
                        assert!(
                            error < 4e-12,
                            "{kind:?}, reversed={reversed}, boundary={boundary}, body={body}, lane={lane}: {actual} != {}, error={error}",
                            expected[lane]
                        );
                    }
                }
                let mut record = high.clone();
                record.extend_from_slice(&low);
                for buffer in [
                    output[0].body_q.device_buffer_current().unwrap(),
                    output[0].joint_q.device_buffer_current().unwrap(),
                    output[0].joint_qd.device_buffer_current().unwrap(),
                    kernel.position_compensation_buffer(&output[0]).unwrap(),
                    kernel.coordinate_compensation_buffer(&output[0]).unwrap(),
                    kernel.paired_rates_buffer(&output[0]).unwrap(),
                    kernel.status_buffer(),
                ] {
                    record.extend(rounding_tests::words(device, queue, buffer));
                }
                records.push(record);
                if cases == 47 {
                    let residue = kernel
                        .body_velocity_compensation_buffer(&input[0])
                        .unwrap()
                        .clone();
                    let original = rounding_tests::words(device, queue, &residue);
                    let input_high = rounding_tests::words(
                        device,
                        queue,
                        input[0].body_qd.device_buffer_current().unwrap(),
                    );
                    let checkpoint = device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("body rate checkpoint"),
                        size: residue.size(),
                        usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    let mut encoder = device.create_command_encoder(&Default::default());
                    encoder.copy_buffer_to_buffer(&residue, 0, &checkpoint, 0, residue.size());
                    queue.submit([encoder.finish()]);
                    queue.write_buffer(&residue, 56, bytemuck::bytes_of(&f32::NAN));
                    kernel
                        .drift(
                            device,
                            queue,
                            &model,
                            &input[0],
                            &control,
                            0.0,
                            true,
                            &mut output[0],
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
                            output[0].body_qd.device_buffer_current().unwrap()
                        ),
                        input_high
                    );
                    let mut encoder = device.create_command_encoder(&Default::default());
                    encoder.copy_buffer_to_buffer(&checkpoint, 0, &residue, 0, residue.size());
                    encoder.clear_buffer(kernel.status_buffer(), 0, None);
                    queue.submit([encoder.finish()]);
                    assert_eq!(rounding_tests::words(device, queue, &residue), original);
                    kernel
                        .drift(
                            device,
                            queue,
                            &model,
                            &input[0],
                            &control,
                            0.0,
                            true,
                            &mut output[0],
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
                            output[0].body_qd.device_buffer_current().unwrap()
                        ),
                        high
                    );
                    assert_eq!(
                        rounding_tests::words(
                            device,
                            queue,
                            kernel
                                .body_velocity_compensation_buffer(&output[0])
                                .unwrap()
                        ),
                        low
                    );
                }
                cases += 1;
            }
        }
    }
    eprintln!("paired body velocity cases={cases}, maximum absolute f64 error={worst}");
    records
}
