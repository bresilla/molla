pub(super) fn source(source: &str) -> String {
    assert_eq!(
        source.matches("fn publish_paired_body_velocity(").count(),
        1
    );
    let start = source
        .find("    let parent_anchor_rotation=multiply(child,parent_rotation,")
        .unwrap();
    let end = source[start..].find("    if kind==JOINT_CABLE {").unwrap() + start;
    let old = &source[start..end];
    assert!(old.contains("let child_anchor_rotation=multiply(child,child_rotation,"));
    assert!(old.contains("if reversed { origin=parent_anchor; }"));
    let new = r#"    var parent_position=PositionPair(vec3<f32>(0.0),vec3<f32>(0.0));
    if parent>=0 { parent_position=rate_position(u32(parent)); }
    let child_center=position_add(child,rate_position(child),
        position_pair_rotate(child,child_rotation,PositionPair(body_com[child].xyz,vec3<f32>(0.0))));
    let reversed=joint_reversed[j]!=0u;
    var frame_rotation:QuaternionPair;
    if reversed {
        frame_rotation=multiply(child,child_rotation,QuaternionPair(xc.rotation,reference.child_rotation));
    } else {
        frame_rotation=multiply(child,parent_rotation,QuaternionPair(xp.rotation,reference.parent_rotation));
    }
    let floating=kind==JOINT_FREE || kind==JOINT_DISTANCE || kind==JOINT_D6;
    var origin:PositionPair;
    if reversed!=floating {
        origin=position_add(child,rate_position(child),
            position_pair_rotate(child,child_rotation,PositionPair(xc.position.xyz,reference.child_position.xyz)));
    } else {
        origin=position_add(child,parent_position,
            position_pair_rotate(child,parent_rotation,PositionPair(xp.position.xyz,reference.parent_position.xyz)));
    }
"#;
    let mut result = source.to_owned();
    result.replace_range(start..end, new);
    result
}

#[test]
fn lazy_body_anchors_runtime_scope() {
    use super::*;
    let original = rates_tests::shader_source();
    let eager = unrestricted_drift_source(&original);
    for (paired, lazy) in [(false, false), (false, true), (true, false)] {
        assert_eq!(
            unrestricted_drift_program_source(&original, paired, lazy),
            eager
        );
    }
    assert_eq!(
        unrestricted_drift_program_source(&original, true, true),
        source(&eager)
    );
    assert_eq!(
        FeatherstoneKernel::paired_drift_build_variant(),
        if cfg!(feature = "native-lazy-geometry-drift") {
            "native-geometry-lazy-anchors-unrestricted-storage-only"
        } else if cfg!(feature = "native-geometry-drift") {
            "native-geometry-unrestricted-storage-only"
        } else if cfg!(feature = "lazy-paired-drift") {
            "lazy-anchors-unrestricted-storage-only"
        } else {
            "eager-anchors"
        }
    );
    for bits in 0..16u32 {
        let (paired, local, native, coordinates) =
            (bits & 1 != 0, bits & 2 != 0, bits & 4 != 0, bits & 8 != 0);
        assert_eq!(
            lazy_drift_enabled(paired, local, native, coordinates),
            cfg!(feature = "lazy-paired-drift")
                && !cfg!(feature = "native-geometry-drift")
                && bits == 1
        );
    }
}

#[test]
fn lazy_body_anchors_source_contract() {
    let original = super::rates_tests::shader_source();
    let candidate = source(&original);
    let begin = original.find("fn publish_paired_body_velocity(").unwrap();
    assert_eq!(&original[..begin], &candidate[..begin]);
    let tail = "    if kind==JOINT_CABLE {";
    assert_eq!(
        original[begin..].split_once(tail).unwrap().1,
        candidate[begin..].split_once(tail).unwrap().1
    );
    for reversed in [false, true] {
        for kind in 0..8 {
            let floating = matches!(kind, 4..=6);
            let mut old_child_origin = reversed;
            if floating {
                old_child_origin = !reversed;
            }
            assert_eq!(old_child_origin, reversed != floating);
        }
    }
    let module = naga::front::wgsl::parse_str(&candidate).unwrap();
    naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .unwrap();
}

#[test]
#[ignore = "requires GPU; alternating six-body paired drift comparison"]
fn lazy_body_anchors_paired_tractor_drift_cost() {
    paired_tractor_drift_cost(0);
}

#[test]
#[ignore = "requires native f64 GPU; alternating six-body exact drift comparison"]
fn paired_native_sum_tractor_drift_cost() {
    paired_tractor_drift_cost(1);
}

#[test]
#[ignore = "requires native f64 GPU; position-only sum isolation with exact moving state"]
fn paired_native_position_sum_tractor_drift_cost() {
    paired_tractor_drift_cost(2);
}

#[test]
#[ignore = "requires native f64 GPU; geometry-only sum isolation preserving paired integration"]
fn paired_native_geometry_sum_tractor_drift_cost() {
    paired_tractor_drift_cost(3);
}

#[test]
#[ignore = "requires native f64 GPU; exact native geometry with lazy anchor comparison"]
fn paired_native_geometry_lazy_tractor_drift_cost() {
    paired_tractor_drift_cost(8);
}

#[test]
#[ignore = "requires native f64 GPU; exact force-driven six-body acceleration comparison"]
fn paired_native_geometry_tractor_acceleration_cost() {
    paired_tractor_drift_cost(4);
}

#[test]
#[ignore = "requires native f64 GPU; fixed-rigorous six-body acceleration comparison"]
fn paired_rigorous_native_geometry_tractor_acceleration_cost() {
    paired_tractor_drift_cost(5);
}

#[test]
#[ignore = "requires native f64 GPU; shared-program storage/native acceleration comparison"]
fn paired_switched_native_geometry_tractor_acceleration_cost() {
    paired_tractor_drift_cost(6);
}

#[test]
#[ignore = "requires native f64 GPU; zero-bias full ten-DOF acceleration operator"]
fn paired_zero_bias_native_geometry_acceleration_operator() {
    paired_tractor_drift_cost(7);
}

#[cfg(test)]
fn paired_tractor_drift_cost(mode: u32) {
    use super::*;
    use molla_core::BodyId;
    use molla_math::{Mat3, Quat, Transform, Vec3};
    use molla_sim::{BodyParams, ModelBuilder};
    assert!(mode <= 8);
    let combined = mode == 8;
    let basis = mode == 7;
    let steps = if basis { 11usize } else { 10 };
    let query_count = steps as u32 * 4;
    let acceleration = (4..8).contains(&mode);
    let fixed_rigorous = (5..8).contains(&mode);
    let switched = (6..8).contains(&mode);
    let native = mode != 0;
    let features = wgpu::Features::TIMESTAMP_QUERY
        | if native {
            wgpu::Features::SHADER_F64
        } else {
            wgpu::Features::empty()
        };
    if acceleration {
        eprintln!("acceleration probe: engine begin");
    }
    let engine = molla_compute::Engine::new_gpu_with_features(features).unwrap();
    if acceleration {
        eprintln!("acceleration probe: engine ready");
    }
    let label = if combined {
        "native geometry lazy anchors"
    } else if basis {
        "zero bias native paired geometry acceleration"
    } else if switched {
        "switched rigorous native paired geometry acceleration"
    } else if fixed_rigorous {
        "fixed rigorous native paired geometry acceleration"
    } else if acceleration {
        "native paired geometry acceleration"
    } else if mode == 3 {
        "native paired geometry sums"
    } else if mode == 2 {
        "native paired position sums"
    } else if native {
        "native paired sums"
    } else {
        "lazy anchors"
    };
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
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
        let wheel = builder.add_body(
            BodyParams::new()
                .with_mass(35.)
                .with_inertia(Mat3::IDENTITY * 3.),
        );
        builder
            .add_joint_revolute(
                root,
                wheel,
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
    let mut model = builder.build().unwrap();
    if basis {
        model.gravity.host_mut().unwrap().fill(Vec3::ZERO);
    }
    let mut initial = State::for_model(&model).unwrap();
    for (i, v) in initial.joint_qd.host_mut().unwrap().iter_mut().enumerate() {
        *v = if basis {
            0.0
        } else {
            0.125 * (i + 1) as f64 + 2f64.powi(-29) * (i + 1) as f64
        };
    }
    molla_sim::eval_fk(&model, &mut initial).unwrap();
    molla_sim::eval_body_velocities(&model, &mut initial).unwrap();
    model.upload_to_device(device, queue).unwrap();
    let mut control = Control::for_model(&model);
    control.upload_to_device(device, queue).unwrap();
    let (offsets, rows) = kernel_configuration(&model);
    let mut runs = Vec::new();
    let mut shared_program = None;
    for candidate in [false, true] {
        if acceleration {
            eprintln!("{label}: fixture ready; program candidate={candidate} begin");
        }
        let start = std::time::Instant::now();
        if !switched || shared_program.is_none() {
            shared_program = Some(FeatherstoneProgram::with_rate_precision(
                device,
                6,
                rows,
                false,
                (candidate || switched || combined) && native,
                false,
                if combined {
                    if candidate {
                        FeatherstonePasses::UnrestrictedDriftNativeGeometryLazyAnchors
                    } else {
                        FeatherstonePasses::UnrestrictedDriftNativeGeometry
                    }
                } else if switched {
                    FeatherstonePasses::PairedRigorousAccelerationSwitchedGeometry
                } else if fixed_rigorous {
                    if candidate {
                        FeatherstonePasses::PairedRigorousAccelerationNativeGeometry
                    } else {
                        FeatherstonePasses::PairedRigorousAccelerationOnly
                    }
                } else if acceleration {
                    if candidate {
                        FeatherstonePasses::PairedAccelerationNativeGeometry
                    } else {
                        FeatherstonePasses::PairedAccelerationOnly
                    }
                } else if candidate && mode == 3 {
                    FeatherstonePasses::UnrestrictedDriftNativeGeometry
                } else if candidate && mode == 2 {
                    FeatherstonePasses::UnrestrictedDriftNativePosition
                } else if candidate && !native {
                    FeatherstonePasses::UnrestrictedDriftLazyAnchors
                } else {
                    FeatherstonePasses::UnrestrictedDriftOnly
                },
                true,
            ));
        }
        let program = shared_program.as_ref().unwrap();
        eprintln!(
            "{label} tractor program candidate={candidate} construction_ms={}",
            start.elapsed().as_secs_f64() * 1000.
        );
        let mut kernel =
            FeatherstoneKernel::from_program(device, queue, &model, offsets.clone(), rows, program);
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
        kernel
            .prepare_paired_rates(
                device,
                queue,
                &[&states[0], &states[1]],
                model.joint_dof_total as u64 * 24 + model.body_count as u64 * 64,
            )
            .unwrap();
        runs.push((kernel, states));
    }
    let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("lazy anchors paired times"),
        ty: wgpu::QueryType::Timestamp,
        count: query_count,
    });
    let capture =
        std::env::var_os("MOLLA_PAIRED_MACHINE_COST_CAPTURE").map(std::path::PathBuf::from);
    if let Some(path) = &capture {
        std::fs::create_dir(path).unwrap();
    }
    let mut previous = Vec::new();
    let contacts = Contacts::with_capacity(0);
    for step in 0..steps {
        if acceleration {
            for (dof, force) in control
                .joint_force
                .host_mut()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                *force = if basis {
                    if step > 0 && dof == step - 1 {
                        100.0
                    } else {
                        0.0
                    }
                } else {
                    (step + 1) as f64
                        * (dof + 1) as f64
                        * 17.0
                        * if dof % 2 == 0 { 1.0 } else { -1.0 }
                };
            }
            control.upload_to_device(device, queue).unwrap();
        }
        let mut results = [Vec::new(), Vec::new()];
        for variant in if step % 2 == 0 { [0, 1] } else { [1, 0] } {
            let (kernel, [a, b]) = &mut runs[variant];
            let (input, output) = if step % 2 == 0 { (a, b) } else { (b, a) };
            let slot = (step * 4 + variant * 2) as u32;
            kernel.set_dispatch_timestamps(Some((queries.clone(), [slot, slot + 1])));
            if acceleration {
                kernel
                    .dispatch(
                        device,
                        queue,
                        &model,
                        input,
                        &control,
                        &contacts,
                        if switched && variant == 1 { 1.0 } else { 0.0 },
                        1.0 / 2040.0,
                        true,
                        true,
                        output,
                        None,
                    )
                    .unwrap();
                let forces = rounding_tests::words(device, queue, kernel.velocity_delta_buffer());
                if basis && step == 0 {
                    assert!(forces.iter().all(|&word| f32::from_bits(word) == 0.0));
                } else {
                    assert!(forces.iter().any(|&word| f32::from_bits(word) != 0.0));
                }
                assert!(forces.iter().all(|&word| f32::from_bits(word).is_finite()));
            } else {
                kernel
                    .drift(
                        device,
                        queue,
                        &model,
                        input,
                        &control,
                        1.0 / 2040.0,
                        true,
                        output,
                    )
                    .unwrap();
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
                kernel.velocity_delta_buffer(),
                kernel.paired_velocity_delta_buffer().unwrap(),
            ] {
                results[variant].extend(rounding_tests::words(device, queue, buffer));
            }
        }
        if let Some(path) = &capture {
            for (variant, words) in results.iter().enumerate() {
                let bytes: Vec<_> = words.iter().flat_map(|v| v.to_le_bytes()).collect();
                std::fs::write(
                    path.join(format!("step-{step:02}-variant-{variant}.bin")),
                    bytes,
                )
                .unwrap();
            }
        }
        assert_eq!(results[0].len(), 250);
        assert_eq!(results[0].len(), results[1].len());
        let differences: Vec<_> = results[0]
            .iter()
            .zip(&results[1])
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .collect();
        for (word, (a, b)) in differences.iter().take(8) {
            eprintln!("{label} step={step} word={word} reference={a:#010x} candidate={b:#010x}");
        }
        assert_eq!(differences.len(), 0, "full compensated state step={step}");
        assert_ne!(results[0], previous, "fixture outputs must change");
        previous = results[0].clone();
    }
    let resolve = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("lazy anchors paired timestamp resolve"),
        size: u64::from(query_count) * 8,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.resolve_query_set(&queries, 0..query_count, &resolve, 0);
    queue.submit([encoder.finish()]);
    let words = rounding_tests::words(device, queue, &resolve);
    let ticks: Vec<u64> = words
        .chunks_exact(2)
        .map(|w| u64::from(w[0]) | (u64::from(w[1]) << 32))
        .collect();
    let mut times = [Vec::new(), Vec::new()];
    for (step, record) in ticks.chunks_exact(4).enumerate() {
        for variant in 0..2 {
            assert!(record[variant * 2 + 1] >= record[variant * 2]);
            let ms = (record[variant * 2 + 1] - record[variant * 2]) as f64
                * f64::from(queue.get_timestamp_period())
                * 1e-6;
            assert!(ms.is_finite() && ms > 0.);
            eprintln!("{label} paired step={step} variant={variant} gpu_ms={ms}");
            if step > 0 {
                times[variant].push(ms);
            }
        }
    }
    for samples in &mut times {
        samples.sort_by(f64::total_cmp);
    }
    let phase = if acceleration {
        "acceleration"
    } else {
        "drift"
    };
    eprintln!(
        "{label} paired tractor {phase}: samples={} exact_steps={steps} reference_median_ms={} candidate_median_ms={}",
        steps - 1,
        times[0][(steps - 2) / 2],
        times[1][(steps - 2) / 2]
    );
}
