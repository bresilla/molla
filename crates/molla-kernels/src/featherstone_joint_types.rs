use super::*;

pub(super) fn validate_kinds(kinds: &[u32]) -> Result<()> {
    if kinds.is_empty() || kinds.len() > 32 || kinds.iter().any(|&kind| kind > 7) {
        return Err(Error::Gpu(
            "joint-type specialization requires 1–32 valid joints".into(),
        ));
    }
    Ok(())
}

pub(super) fn validate_model(model: &Model, kinds: Option<&[u32]>) -> Result<()> {
    if let Some(kinds) = kinds {
        if model.joint_count != kinds.len() || model.joint_type.host()? != kinds {
            return Err(Error::Gpu(
                "specialized joint types changed; recreate the kernel".into(),
            ));
        }
    }
    Ok(())
}

pub(super) fn source(source: &str, kinds: &[u32]) -> Result<String> {
    validate_kinds(kinds)?;
    let mut result = String::new();
    let mut tail = source;
    let mut loads = 0;
    while let Some(start) = tail.find("joint_type[") {
        result.push_str(&tail[..start]);
        tail = &tail[start + "joint_type[".len()..];
        let end = tail
            .find(']')
            .ok_or_else(|| Error::Gpu("joint load bracket".into()))?;
        let index = &tail[..end];
        if index.is_empty()
            || !index
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'_')
        {
            return Err(Error::Gpu("unexpected joint load index".into()));
        }
        result.push_str(&format!("model_joint_kind({index})"));
        tail = &tail[end + 1..];
        loads += 1;
    }
    if loads == 0 {
        return Err(Error::Gpu("no joint type loads".into()));
    }
    result.push_str(tail);
    result.push_str(&format!(
        "\nfn model_joint_kind(j: u32) -> u32 {{\n var kind = {}u;\n",
        kinds[0]
    ));
    for kind in 0..=7 {
        if kind == kinds[0] {
            continue;
        }
        let indices: Vec<_> = kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == kind)
            .map(|(j, _)| format!("j == {j}u"))
            .collect();
        if !indices.is_empty() {
            result.push_str(&format!(
                " if {} {{ kind = {kind}u; }}\n",
                indices.join(" || ")
            ));
        }
    }
    result.push_str(" return kind;\n}\n");
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use molla_core::BodyId;
    use molla_math::Transform;
    use molla_sim::{BodyParams, ModelBuilder};

    #[test]
    fn specialization_rejects_type_and_count_changes_but_not_mutable_controls() {
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(BodyParams::new());
        builder
            .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let mut model = builder.build().unwrap();
        let kinds = model.joint_type.host().unwrap().to_vec();
        validate_model(&model, Some(&kinds)).unwrap();
        model.joint_enabled.host_mut().unwrap()[0] = 0;
        model.joint_velocity_limit.host_mut().unwrap()[0] = 0.5;
        validate_model(&model, Some(&kinds)).unwrap();
        model.joint_type.host_mut().unwrap()[0] = molla_sim::JointType::D6 as u32;
        assert!(validate_model(&model, Some(&kinds)).is_err());
        validate_model(&model, None).unwrap();
        model.joint_type.host_mut().unwrap()[0] = kinds[0];
        model.joint_count = 0;
        assert!(validate_model(&model, Some(&kinds)).is_err());
        assert!(validate_kinds(&[]).is_err());
        assert!(validate_kinds(&[4; 33]).is_err());
        assert!(validate_kinds(&[u32::MAX]).is_err());
    }

    #[test]
    #[ignore = "requires real GPU; specialized physical step and mutation guard"]
    fn specialized_free_motion_and_rejected_mutation_preserve_state() {
        use molla_compute::Engine;
        use molla_math::{Mat3, SpatialVector, Vec3};
        let engine = Engine::new_gpu().unwrap();
        assert!(!engine.adapter_info().unwrap().is_software);
        let device = engine.device().unwrap();
        let queue = engine.queue().unwrap();
        let mut builder = ModelBuilder::new();
        let body = builder.add_body(
            BodyParams::new()
                .with_mass(2.0)
                .with_inertia(Mat3::IDENTITY),
        );
        builder
            .add_joint_free(BodyId::NONE, body, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        let mut model = builder.build().unwrap();
        model.gravity.host_mut().unwrap()[0] = Vec3::ZERO;
        let mut input = State::for_model(&model).unwrap();
        input.joint_qd.host_mut().unwrap()[0] = 0.25;
        input.body_qd.host_mut().unwrap()[0] = SpatialVector::linear(Vec3::X * 0.25);
        let mut output = input.clone_host().unwrap();
        let mut control = Control::for_model(&model);
        model.upload_to_device(device, queue).unwrap();
        input.upload_to_device(device, queue).unwrap();
        output.upload_to_device(device, queue).unwrap();
        control.upload_to_device(device, queue).unwrap();
        let started = std::time::Instant::now();
        let mut kernel = FeatherstoneKernel::new_cached_with_options(
            &engine,
            &model,
            FeatherstoneRounding::Storage,
            FeatherstoneExecution::ExperimentalJointTypes,
        )
        .unwrap();
        eprintln!(
            "specialized free program ready {:.6}s",
            started.elapsed().as_secs_f64()
        );
        assert_eq!(
            kernel.execution(),
            FeatherstoneExecution::ExperimentalJointTypes
        );
        assert_eq!(kernel.specialized_joint_types.as_deref(), Some(&[4][..]));
        let reused = FeatherstoneKernel::new_cached_with_options(
            &engine,
            &model,
            FeatherstoneRounding::Storage,
            FeatherstoneExecution::ExperimentalJointTypes,
        )
        .unwrap();
        assert_eq!(reused.pipeline, kernel.pipeline);
        assert_ne!(reused.velocity_delta, kernel.velocity_delta);
        let words = |buffer: &wgpu::Buffer| rounding_tests::words(device, queue, buffer);
        for step in 1..=4 {
            kernel
                .dispatch(
                    device,
                    queue,
                    &model,
                    &input,
                    &control,
                    &Contacts::with_capacity(0),
                    0.0,
                    1.0 / 64.0,
                    true,
                    true,
                    &mut output,
                    None,
                )
                .unwrap();
            let pose = words(output.body_q.device_buffer_current().unwrap());
            assert_eq!(f32::from_bits(pose[0]), step as f32 / 256.0);
            assert_eq!(pose[1..3], [0, 0]);
            assert_eq!(words(kernel.status_buffer()), [0]);
            std::mem::swap(&mut input, &mut output);
        }
        let before = words(output.body_q.device_buffer_current().unwrap());
        model.joint_type.host_mut().unwrap()[0] = molla_sim::JointType::D6 as u32;
        let error = kernel
            .dispatch(
                device,
                queue,
                &model,
                &input,
                &control,
                &Contacts::with_capacity(0),
                0.0,
                1.0 / 64.0,
                true,
                true,
                &mut output,
                None,
            )
            .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("specialized joint types changed")
        );
        assert_eq!(
            words(output.body_q.device_buffer_current().unwrap()),
            before
        );
        model.joint_type.host_mut().unwrap()[0] = molla_sim::JointType::Free as u32;
        model.upload_to_device(device, queue).unwrap();
        kernel
            .dispatch(
                device,
                queue,
                &model,
                &input,
                &control,
                &Contacts::with_capacity(0),
                0.0,
                1.0 / 64.0,
                true,
                true,
                &mut output,
                None,
            )
            .unwrap();
        assert_eq!(
            f32::from_bits(words(output.body_q.device_buffer_current().unwrap())[0]),
            5.0 / 256.0
        );
        eprintln!("five exact free-motion steps, reuse, type mutation rejection and recovery pass");
    }
}
