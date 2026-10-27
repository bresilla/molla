use super::*;

#[test]
fn native_prediction_control_shader_contract() {
    for (b, d) in [(1, 6), (6, 10), (64, 32)] {
        let text = control_source(b, d);
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

fn read(device: &wgpu::Device, queue: &wgpu::Queue, b: &wgpu::Buffer) -> Vec<u32> {
    let (tx, rx) = std::sync::mpsc::channel();
    wgpu::util::DownloadBuffer::read_buffer(device, queue, &b.slice(..), move |r| {
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
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn drain(device: &wgpu::Device, queue: &wgpu::Queue) {
    let submission = queue.submit([]);
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
}

fn snapshot(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    s: &State,
    solver: &FeatherstoneSolver,
) -> Vec<u32> {
    let mut words = Vec::new();
    for buffer in [
        s.body_q.device_buffer_current().unwrap(),
        s.body_qd.device_buffer_current().unwrap(),
        s.joint_q.device_buffer_current().unwrap(),
        s.joint_qd.device_buffer_current().unwrap(),
        solver.gpu_coordinate_compensation_buffer(s).unwrap(),
        solver.gpu_position_compensation_buffer(s).unwrap(),
        solver.gpu_paired_rates_buffer(s).unwrap(),
        solver.gpu_body_velocity_compensation_buffer(s).unwrap(),
        solver.gpu_velocity_delta_buffer().unwrap(),
        solver.gpu_paired_velocity_delta_buffer().unwrap(),
    ] {
        words.extend(read(device, queue, buffer));
    }
    assert_eq!(words.len(), 250);
    words
}

fn difference(words: &[u32], cpu: &State) -> f64 {
    let f = |i| f64::from(f32::from_bits(words[i]));
    let mut error = 0.0f64;
    for (i, &expected) in cpu.joint_q.host().unwrap().iter().enumerate() {
        error = error.max((f(96 + i) + f(117 + i) - expected).abs());
    }
    for (i, &expected) in cpu.joint_qd.host().unwrap().iter().enumerate() {
        error = error.max((f(152 + 2 * i) + f(153 + 2 * i) - expected).abs());
    }
    for i in 0..6 {
        let pose = cpu.body_q.host().unwrap()[i];
        let velocity = cpu.body_qd.host().unwrap()[i];
        for k in 0..3 {
            error = error.max((f(8 * i + k) + f(128 + 4 * i + k) - pose.position[k]).abs());
            error = error.max((f(48 + 8 * i + k) + f(172 + 8 * i + k) - velocity.linear[k]).abs());
            error = error.max((f(52 + 8 * i + k) + f(176 + 8 * i + k) - velocity.angular[k]).abs());
        }
        for k in 0..4 {
            error = error.max((f(8 * i + 4 + k) - pose.rotation.to_array()[k]).abs());
        }
    }
    error
}

fn compare_deferred(
    engine: &Engine,
    prepared: &PreparedNativePrediction,
    solver: &mut FeatherstoneSolver,
    model: &Model,
    control: &Control,
    input: &State,
    output: &mut State,
    dt: f64,
    full: &[u32],
    root: &std::path::Path,
    step: usize,
) {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    std::fs::write(
        root.join(format!("full-{step:02}.bin")),
        bytemuck::cast_slice(full),
    )
    .unwrap();
    let poison_poses = |solver: &FeatherstoneSolver, output: &State| {
        let mut encoder = device.create_command_encoder(&Default::default());
        for buffer in [
            output.body_q.device_buffer_current().unwrap(),
            output.body_qd.device_buffer_current().unwrap(),
            output.joint_q.device_buffer_current().unwrap(),
            output.body_f.device_buffer_current().unwrap(),
            solver.gpu_coordinate_compensation_buffer(output).unwrap(),
            solver.gpu_position_compensation_buffer(output).unwrap(),
            solver
                .gpu_body_velocity_compensation_buffer(output)
                .unwrap(),
        ] {
            encoder.clear_buffer(buffer, 0, None);
        }
        queue.submit([encoder.finish()]);
        drain(device, queue);
    };
    poison_poses(solver, output);
    let poisoned = snapshot(device, queue, output, solver);
    assert_ne!(poisoned, full);
    std::fs::write(
        root.join(format!("poisoned-{step:02}.bin")),
        bytemuck::cast_slice(&poisoned),
    )
    .unwrap();
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
    let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
    prepared
        .rates_profiled(solver, model, control, input, output, dt, None)
        .unwrap();
    drain(device, queue);
    assert_eq!(allocations.finish().observations, 0);
    assert_eq!(transfers.finish(), Default::default());
    assert_eq!(dispatches.finish().encoded_dispatches, 4);
    let rates = snapshot(device, queue, output, solver);
    std::fs::write(
        root.join(format!("rates-{step:02}.bin")),
        bytemuck::cast_slice(&rates),
    )
    .unwrap();
    assert_eq!(
        rates, poisoned,
        "rates-only prediction changed retained poses or predicted rates"
    );
    solver.drift_gpu(model, input, control, dt, output).unwrap();
    let deferred = snapshot(device, queue, output, solver);
    std::fs::write(
        root.join(format!("deferred-{step:02}.bin")),
        bytemuck::cast_slice(&deferred),
    )
    .unwrap();
    assert_eq!(
        deferred, full,
        "single drift differs from completed prediction"
    );
    if matches!(step, 0 | 6 | 14) {
        let mut corrected = full[152..172].to_vec();
        let mut delta = full[230..250].to_vec();
        for d in 0..10 {
            let change = (d as f32 - 4.5) * 0.000125;
            corrected[2 * d] = (f32::from_bits(corrected[2 * d]) + change).to_bits();
            delta[2 * d] = (f32::from_bits(delta[2 * d]) + change).to_bits();
        }
        let inject = |solver: &FeatherstoneSolver, output: &State| {
            queue.write_buffer(
                solver.gpu_paired_rates_buffer(output).unwrap(),
                0,
                bytemuck::cast_slice(&corrected),
            );
            queue.write_buffer(
                solver.gpu_paired_velocity_delta_buffer().unwrap(),
                0,
                bytemuck::cast_slice(&delta),
            );
        };
        inject(solver, output);
        solver.drift_gpu(model, input, control, dt, output).unwrap();
        let reference = snapshot(device, queue, output, solver);
        poison_poses(solver, output);
        prepared
            .rates_profiled(solver, model, control, input, output, dt, None)
            .unwrap();
        inject(solver, output);
        solver.drift_gpu(model, input, control, dt, output).unwrap();
        let candidate = snapshot(device, queue, output, solver);
        for (label, words) in [
            ("corrected-reference", &reference),
            ("corrected-deferred", &candidate),
        ] {
            std::fs::write(
                root.join(format!("{label}-{step:02}.bin")),
                bytemuck::cast_slice(words),
            )
            .unwrap();
        }
        assert_eq!(
            candidate, reference,
            "corrected final state differs when predictor drift is omitted"
        );
        prepared
            .step(solver, model, control, input, output, dt)
            .unwrap();
        assert_eq!(snapshot(device, queue, output, solver), full);
    }
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
            .with_mass(1500.0)
            .with_inertia(Mat3::IDENTITY * 600.0),
    );
    b.add_joint_free(
        BodyId::NONE,
        root,
        Transform::from_translation(Vec3::new(65536.0, 0.95, 0.0)),
        Transform::IDENTITY,
    )
    .unwrap();
    for (x, z) in [(-0.75, -0.65), (-0.75, 0.65), (0.75, -0.65), (0.75, 0.65)] {
        let wheel = b.add_body(
            BodyParams::new()
                .with_mass(35.0)
                .with_inertia(Mat3::IDENTITY * 3.0),
        );
        b.add_joint_revolute(
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
    let tine = b.add_body(
        BodyParams::new()
            .with_mass(20.0)
            .with_inertia(Mat3::IDENTITY),
    );
    b.add_joint_fixed(
        root,
        tine,
        Transform::from_translation(Vec3::new(-1.45, -0.72, 0.0)),
        Transform::IDENTITY,
    )
    .unwrap();
    let mut m = b.build().unwrap();
    m.joint_effort_limit.host_mut().unwrap()[6] = 35.0;
    m
}

#[test]
fn native_prediction_budget_contract() {
    let model = model();
    assert_eq!(
        PreparedNativePrediction::required_bytes(&model).unwrap(),
        4232
    );
    assert!(
        crate::generalized_prediction_gpu::PreparedGeneralizedPrediction::required_bytes(0, 10)
            .is_err()
    );
}

#[test]
#[ignore = "requires f64 GPU and bounded cold paired drift construction"]
fn native_prediction_drift_gpu_contract() {
    use crate::{
        Solver,
        math::{SpatialVector, Vec3},
        sim::Contacts,
    };
    let start = std::time::Instant::now();
    let capture =
        std::path::PathBuf::from(std::env::var_os("MOLLA_NATIVE_PREDICTION_CAPTURE").unwrap());
    std::fs::create_dir(&capture).unwrap();
    eprintln!("native prediction: engine begin");
    let engine = Engine::new_gpu_with_features(wgpu::Features::SHADER_F64).unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut model = model();
    let mut initial = State::for_model(&model).unwrap();
    for (i, v) in initial.joint_qd.host_mut().unwrap().iter_mut().enumerate() {
        *v = 0.025 * (i + 1) as f64 + 2f64.powi(-29);
    }
    initial.body_f.host_mut().unwrap()[0] =
        SpatialVector::new(Vec3::new(20.0, 0.0, 3.0), Vec3::new(0.0, 0.0, 0.125));
    crate::sim::eval_fk(&model, &mut initial).unwrap();
    crate::sim::eval_body_velocities(&model, &mut initial).unwrap();
    let mut cpu = [initial.clone_host().unwrap(), initial.clone_host().unwrap()];
    let mut states = [initial.clone_host().unwrap(), initial.clone_host().unwrap()];
    states[1]
        .body_f
        .host_mut()
        .unwrap()
        .fill(SpatialVector::ZERO);
    let mut control = Control::for_model(&model);
    control.joint_target_mode.host_mut().unwrap()[6] = 4;
    control.joint_target_pos.host_mut().unwrap()[6] = 60.0;
    control.joint_motor_max_force.host_mut().unwrap()[6] = 20.0;
    control.body_external_f.host_mut().unwrap()[5] =
        SpatialVector::new(Vec3::new(1.5, 0.25, -0.5), Vec3::new(0.0, 0.0, 0.25));
    model.upload_to_device(device, queue).unwrap();
    control.upload_to_device(device, queue).unwrap();
    for s in &mut states {
        s.upload_to_device(device, queue).unwrap();
    }
    eprintln!("native prediction: drift program begin");
    let mut solver = FeatherstoneSolver::new_gpu_for_native_prediction(&engine, &model)
        .unwrap()
        .with_damping(0.01);
    solver
        .prepare_gpu_coordinate_compensation(&states.each_ref())
        .unwrap();
    solver
        .prepare_gpu_paired_rates(&states.each_ref(), 1 << 20)
        .unwrap();
    for s in &states {
        solver.seed_gpu_coordinate_compensation(s).unwrap();
        solver.seed_gpu_paired_rates(s).unwrap();
    }
    eprintln!("native prediction: drift ready {:?}", start.elapsed());
    for buffer in [
        solver.gpu_velocity_delta_buffer().unwrap(),
        solver.gpu_paired_velocity_delta_buffer().unwrap(),
    ] {
        assert!(buffer.usage().contains(wgpu::BufferUsages::COPY_DST));
    }
    let dt = f64::from(1.0f32 / 2040.0);
    let budget = PreparedNativePrediction::required_bytes(&model).unwrap();
    assert!(
        PreparedNativePrediction::new(
            &engine,
            &solver,
            &model,
            &control,
            &states[0],
            &states[1],
            dt,
            budget - 1
        )
        .is_err()
    );
    let prepared = [
        PreparedNativePrediction::new(
            &engine, &solver, &model, &control, &states[0], &states[1], dt, budget,
        )
        .unwrap(),
        PreparedNativePrediction::new(
            &engine, &solver, &model, &control, &states[1], &states[0], dt, budget,
        )
        .unwrap(),
    ];
    {
        let [a, b] = &mut states;
        assert!(
            solver
                .step(&model, a, &control, &Contacts::with_capacity(0), dt, b)
                .is_err()
        );
    }
    let mut reference = FeatherstoneSolver::new()
        .with_rigorous_d_solve(true)
        .with_damping(0.01);
    let mut worst = 0.0f64;
    for step in 0..20 {
        let index = step % 2;
        let asleep = step == 8;
        solver.set_gpu_active_articulations(&[!asleep]).unwrap();
        for (i, f) in control
            .joint_force
            .host_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            *f = (step + 1) as f64 * (i + 1) as f64 * 0.5;
        }
        control.upload_to_device(device, queue).unwrap();
        let [ca, cb] = &mut cpu;
        let (ci, co) = if index == 0 { (ca, cb) } else { (cb, ca) };
        if asleep {
            *co = ci.clone_host().unwrap();
        } else {
            reference
                .step(&model, ci, &control, &Contacts::with_capacity(0), dt, co)
                .unwrap();
        }
        let [a, b] = &mut states;
        let (input, output) = if index == 0 { (a, b) } else { (b, a) };
        let input_words = if asleep {
            Some(snapshot(device, queue, input, &solver))
        } else {
            None
        };
        drain(device, queue);
        let before = device.get_internal_counters().hal;
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let transfers = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let dispatches = molla_core::gpu_metrics::DispatchScope::start().unwrap();
        prepared[index]
            .step(&mut solver, &model, &control, input, output, dt)
            .unwrap();
        drain(device, queue);
        let after = device.get_internal_counters().hal;
        assert_eq!(allocations.finish().observations, 0);
        let transfers = transfers.finish();
        assert_eq!(transfers.upload_bytes, 48);
        assert_eq!(transfers.readback_bytes, 0);
        assert!(!transfers.overflowed);
        let dispatches = dispatches.finish();
        assert_eq!(dispatches.encoded_dispatches, 5);
        assert!(!dispatches.overflowed);
        assert_eq!(before.buffers.read(), after.buffers.read());
        if step >= 2 {
            assert_eq!(before.bind_groups.read(), after.bind_groups.read());
        }
        assert_eq!(
            before.compute_pipelines.read(),
            after.compute_pipelines.read()
        );
        assert_eq!(
            read(device, queue, solver.gpu_status_buffer().unwrap()),
            [0]
        );
        let words = snapshot(device, queue, output, &solver);
        compare_deferred(
            &engine,
            &prepared[index],
            &mut solver,
            &model,
            &control,
            input,
            output,
            dt,
            &words,
            &capture,
            step,
        );
        assert!(output.body_f.host().is_err());
        assert_eq!(
            read(
                device,
                queue,
                output.body_f.device_buffer_current().unwrap()
            ),
            read(device, queue, input.body_f.device_buffer_current().unwrap())
        );
        if let Some(input_words) = input_words {
            assert_eq!(&words[..220], &input_words[..220]);
            assert_eq!(&words[220..], &[0; 30]);
        }
        let error = difference(&words, co);
        worst = worst.max(error);
        eprintln!("native_prediction_step={step} error={error:e}");
        assert!(error <= 2e-5);
        std::fs::write(
            capture.join(format!("step-{step:02}.bin")),
            bytemuck::cast_slice(&words),
        )
        .unwrap();
        if !asleep {
            let efforts = read(device, queue, prepared[index].applied_efforts());
            let expected =
                (control.joint_force.host().unwrap()[6] as f32 + 20.0).clamp(-35.0, 35.0);
            assert_eq!(efforts[12], expected.to_bits());
            assert_eq!(efforts[13], 0);
        }
    }
    let before = snapshot(device, queue, &states[0], &solver);
    let old_mass = model.body_mass.host().unwrap()[0];
    model.body_mass.host_mut().unwrap()[0] = old_mass + 1e-10;
    {
        let [input, output] = &mut states;
        assert!(
            prepared[0]
                .step(&mut solver, &model, &control, input, output, dt)
                .is_err()
        );
    }
    model.body_mass.host_mut().unwrap()[0] = old_mass;
    model.upload_to_device(device, queue).unwrap();
    {
        let [input, output] = &mut states;
        assert!(
            prepared[0]
                .step(&mut solver, &model, &control, input, output, 2.0 * dt)
                .is_err()
        );
    }
    let replaced = std::mem::replace(
        &mut control.joint_force,
        crate::storage::Buffer::from_vec(vec![0.0; 10], crate::storage::BufferUsage::Storage),
    );
    control.upload_to_device(device, queue).unwrap();
    {
        let [input, output] = &mut states;
        assert!(
            prepared[0]
                .step(&mut solver, &model, &control, input, output, dt)
                .is_err()
        );
    }
    control.joint_force = replaced;
    control.joint_target_mode.host_mut().unwrap()[6] = 2;
    control.upload_to_device(device, queue).unwrap();
    {
        let [input, output] = &mut states;
        prepared[0]
            .step(&mut solver, &model, &control, input, output, dt)
            .unwrap();
        assert_eq!(
            read(device, queue, solver.gpu_status_buffer().unwrap()),
            [32]
        );
        let failed = snapshot(device, queue, output, &solver);
        compare_deferred(
            &engine,
            &prepared[0],
            &mut solver,
            &model,
            &control,
            input,
            output,
            dt,
            &failed,
            &capture,
            99,
        );
        assert_eq!(&failed[..220], &before[..220]);
        assert_eq!(&failed[220..], &[0; 30]);
    }
    control.joint_target_mode.host_mut().unwrap()[6] = 4;
    control.upload_to_device(device, queue).unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    for p in &prepared {
        p.reset_internal(&mut encoder);
    }
    queue.submit([encoder.finish()]);
    {
        let [input, output] = &mut states;
        prepared[0]
            .step(&mut solver, &model, &control, input, output, dt)
            .unwrap();
        assert_eq!(
            read(device, queue, solver.gpu_status_buffer().unwrap()),
            [32]
        );
    }
    let mut encoder = device.create_command_encoder(&Default::default());
    for p in &prepared {
        p.reset_internal(&mut encoder);
    }
    encoder.clear_buffer(solver.gpu_status_buffer().unwrap(), 0, None);
    queue.submit([encoder.finish()]);
    {
        let [ci, co] = &mut cpu;
        reference
            .step(&model, ci, &control, &Contacts::with_capacity(0), dt, co)
            .unwrap();
        let [input, output] = &mut states;
        prepared[0]
            .step(&mut solver, &model, &control, input, output, dt)
            .unwrap();
        assert_eq!(
            read(device, queue, solver.gpu_status_buffer().unwrap()),
            [0]
        );
        let words = snapshot(device, queue, output, &solver);
        let error = difference(&words, co);
        assert!(error <= 2e-5);
        worst = worst.max(error);
        std::fs::write(capture.join("recovered.bin"), bytemuck::cast_slice(&words)).unwrap();
    }
    eprintln!(
        "native_prediction_drift_worst={worst:e} inactive_control_mutation_reset=pass elapsed={:?}",
        start.elapsed()
    );
    std::fs::write(capture.join("result.json"),serde_json::to_vec_pretty(&serde_json::json!({"acceptance":false,"deferred_exact_state_and_deltas":true,"corrected_rate_cases":3,"inactive_and_fault_preservation":true,"rates_only_dispatches":4,"rates_only_upload_bytes":0,"max_cpu_error":worst,"elapsed_seconds":start.elapsed().as_secs_f64()})).unwrap()).unwrap();
}
