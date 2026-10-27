use super::*;
use molla_core::BodyId;
use molla_math::{Mat3, SpatialVector, Transform, Vec3};
use molla_sim::{BodyParams, ModelBuilder};

#[test]
fn paired_momentum_shader_contract() {
    for width in [1, 8, 32] {
        let source = paired_source(width);
        let module = wgpu::naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|e| panic!("{}", e.emit_to_string(&source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap();
        assert_eq!(module.entry_points.len(), 1);
    }
}

fn storage<T: bytemuck::Pod>(engine: &Engine, values: &[T]) -> wgpu::Buffer {
    use wgpu::util::DeviceExt;
    engine
        .device()
        .unwrap()
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("paired momentum fixture"),
            contents: bytemuck::cast_slice(values),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
}
fn words(engine: &Engine, source: &wgpu::Buffer) -> Vec<u32> {
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &buffer, 0, source.size());
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    buffer
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |r| tx.send(r).unwrap());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .unwrap()
        .unwrap();
    let data = bytemuck::cast_slice(&buffer.slice(..).get_mapped_range()).to_vec();
    buffer.unmap();
    data
}
fn pairs(engine: &Engine, buffer: &wgpu::Buffer) -> Vec<f64> {
    words(engine, buffer)
        .chunks_exact(2)
        .map(|p| f32::from_bits(p[0]) as f64 + f32::from_bits(p[1]) as f64)
        .collect()
}

fn split(values: &[f64]) -> Vec<[f32; 2]> {
    values
        .iter()
        .map(|v| [*v as f32, (*v - *v as f32 as f64) as f32])
        .collect()
}

#[test]
#[ignore = "requires real GPU"]
fn paired_momentum_articulated_response_matches_f64_transport() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let mut worst = 0.0f64;
    for projected_case in [false, true] {
        let mut builder = ModelBuilder::new();
        let root = builder.add_body(
            BodyParams::new()
                .with_mass(100.)
                .with_inertia(Mat3::from_diagonal(Vec3::new(3., 4., 5.)))
                .with_com(Vec3::new(0.125, -0.25, 0.5)),
        );
        let child = builder.add_body(
            BodyParams::new()
                .with_mass(2.)
                .with_inertia(Mat3::from_diagonal(Vec3::new(0.5, 0.25, 0.75)))
                .with_com(Vec3::new(-0.125, 0.25, 0.)),
        );
        builder
            .add_joint_free(BodyId::NONE, root, Transform::IDENTITY, Transform::IDENTITY)
            .unwrap();
        builder
            .add_joint_revolute(
                root,
                child,
                Transform::from_translation(Vec3::new(1., 0.5, 0.)),
                Transform::from_translation(Vec3::new(0., 0.25, 0.)),
                Vec3::Z,
            )
            .unwrap();
        let mut model = builder.build().unwrap();
        model.gravity.host_mut().unwrap().fill(Vec3::ZERO);
        let mut initial = State::for_model(&model).unwrap();
        initial.joint_q.host_mut().unwrap()[0..3].copy_from_slice(&[0.25, -0.5, 0.125]);
        initial.joint_q.host_mut().unwrap()[7] = 0.2;
        for (i, value) in initial.joint_qd.host_mut().unwrap().iter_mut().enumerate() {
            *value = [0.8, -0.2, 0.1, 0.3, 0.2, -0.4, 0.5][i] + 2f64.powi(-30) * (i + 1) as f64;
        }
        molla_sim::eval_fk(&model, &mut initial).unwrap();
        molla_sim::eval_body_velocities(&model, &mut initial).unwrap();
        let dt = 0.0005f32;
        let mut predicted = initial.clone_host().unwrap();
        let changes: Vec<f64> = (0..7).map(|i| (i as f64 + 1.) * 1e-5 + 1e-13).collect();
        for (value, increment) in predicted
            .joint_qd
            .host_mut()
            .unwrap()
            .iter_mut()
            .zip(&changes)
        {
            *value += increment;
        }
        let rates = predicted.joint_qd.host().unwrap().to_vec();
        let q = predicted.joint_q.host_mut().unwrap();
        for i in 0..3 {
            q[i] += rates[i] * dt as f64;
        }
        let rotation = molla_math::Quat::from_xyzw(
            0.5 * rates[3] * dt as f64,
            0.5 * rates[4] * dt as f64,
            0.5 * rates[5] * dt as f64,
            1.,
        )
        .normalize();
        q[3..7].copy_from_slice(&rotation.to_array());
        q[7] += rates[6] * dt as f64;
        molla_sim::eval_fk(&model, &mut predicted).unwrap();
        molla_sim::eval_body_velocities(&model, &mut predicted).unwrap();
        let mut projected = initial.clone_host().unwrap();
        projected.joint_q.host_mut().unwrap()[7] += 0.002;
        molla_sim::eval_fk(&model, &mut projected).unwrap();
        let mut reference = predicted.clone_host().unwrap();
        crate::rigid_momentum::transport(
            &model,
            &initial,
            &mut reference,
            dt as f64,
            projected_case.then_some(&projected),
            None,
        )
        .unwrap();
        let mut states = [initial, predicted];
        let rate_buffers = states
            .each_ref()
            .map(|s| storage(&engine, &split(s.joint_qd.host().unwrap())));
        let velocity_buffers = states.each_ref().map(|s| {
            let values: Vec<_> = s
                .body_qd
                .host()
                .unwrap()
                .iter()
                .flat_map(|v| {
                    v.linear
                        .to_array()
                        .into_iter()
                        .chain([0.])
                        .chain(v.angular.to_array())
                        .chain([0.])
                })
                .collect();
            storage(
                &engine,
                &split(&values).iter().map(|p| p[1]).collect::<Vec<_>>(),
            )
        });
        let position_values = |s: &State| -> Vec<f32> {
            s.body_q
                .host()
                .unwrap()
                .iter()
                .flat_map(|p| p.position.to_array().into_iter().chain([0.]))
                .map(|v| (v - v as f32 as f64) as f32)
                .collect()
        };
        let position_buffers = states
            .each_ref()
            .map(|s| storage(&engine, &position_values(s)));
        let coordinate_buffers = states.each_ref().map(|s| {
            storage(
                &engine,
                &split(s.joint_q.host().unwrap())
                    .iter()
                    .map(|p| p[1])
                    .collect::<Vec<_>>(),
            )
        });
        let projected_low = storage(&engine, &position_values(&projected));
        let deltas = split(&changes.iter().map(|x| x * 0.6).collect::<Vec<_>>());
        let corrections = split(&changes.iter().map(|x| x * 0.4).collect::<Vec<_>>());
        let delta = storage(&engine, &deltas);
        let correction = storage(&engine, &corrections);
        let active = storage(&engine, &[1u32]);
        let status = storage(&engine, &[0u32]);
        for s in &mut states {
            s.upload_to_device(device, queue).unwrap();
        }
        projected.upload_to_device(device, queue).unwrap();
        let buffers = PairedBuffers {
            rates: [&rate_buffers[0], &rate_buffers[1]],
            velocities: [&velocity_buffers[0], &velocity_buffers[1]],
            positions: [&position_buffers[0], &position_buffers[1]],
            coordinates: [&coordinate_buffers[0], &coordinate_buffers[1]],
            projected_low: projected_case.then_some(&projected_low),
            active: &active,
            status: &status,
        };
        eprintln!("paired momentum program begin: projected={projected_case}");
        let transport = MomentumGpu::from_buffers(
            &engine,
            &model,
            &states,
            projected_case.then_some(&projected),
            &delta,
            Some(&correction),
            None,
            Some(&buffers),
        )
        .unwrap()
        .unwrap();
        eprintln!("paired momentum program ready: projected={projected_case}");
        transport.prepare(&engine, dt as f64);
        let mut encoder = device.create_command_encoder(&Default::default());
        transport.encode_profiled(&mut encoder, 0, None);
        queue.submit([encoder.finish()]);
        assert_eq!(words(&engine, &status), [0]);
        let actual = pairs(&engine, &rate_buffers[1]);
        for (i, (&actual, &expected)) in actual
            .iter()
            .zip(reference.joint_qd.host().unwrap())
            .enumerate()
        {
            let error = (actual - expected).abs() / (1. + expected.abs());
            worst = worst.max(error);
            assert!(
                error < 1e-6,
                "projected={projected_case}, rate {i}: {actual} != {expected}, normalized error={error}"
            );
        }
        let high = words(&engine, states[1].body_qd.device_buffer_current().unwrap());
        let low = words(&engine, &velocity_buffers[1]);
        for (body, v) in reference.body_qd.host().unwrap().iter().enumerate() {
            for (lane, expected) in v
                .linear
                .to_array()
                .into_iter()
                .chain([0.])
                .chain(v.angular.to_array())
                .chain([0.])
                .enumerate()
            {
                let actual = f32::from_bits(high[8 * body + lane]) as f64
                    + f32::from_bits(low[8 * body + lane]) as f64;
                let error = (actual - expected).abs() / (1. + expected.abs());
                worst = worst.max(error);
                assert!(
                    error < 1e-6,
                    "projected={projected_case}, body {body}, lane {lane}: {actual} != {expected}"
                );
            }
        }
    }
    eprintln!(
        "paired articulated momentum and positional-response cases: maximum normalized f64 error={worst}"
    );
}

#[test]
#[ignore = "requires real GPU"]
fn paired_momentum_preserves_sub_ulp_changes_and_failure_atomicity() {
    let engine = Engine::new_gpu().unwrap();
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
    let model = builder.build().unwrap();
    let mut states = [
        State::for_model(&model).unwrap(),
        State::for_model(&model).unwrap(),
    ];
    let dt = 0.001f32;
    for (i, state) in states.iter_mut().enumerate() {
        state.joint_qd.host_mut().unwrap()[0] = 1.0;
        state.joint_qd.host_mut().unwrap()[5] = 1.0;
        state.body_qd.host_mut().unwrap()[0] = SpatialVector::new(Vec3::X, Vec3::Z);
        state.joint_q.host_mut().unwrap()[0] = i as f64 * dt as f64;
        state.body_q.host_mut().unwrap()[0].position.x = i as f64 * dt as f64;
        state.upload_to_device(device, queue).unwrap();
    }
    let linear_low = 2f32.powi(-30);
    let angular_low = 2f32.powi(-27);
    let input = [
        [1., linear_low],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [1., angular_low],
    ];
    let output = [[1f32, 0.], [0., 0.], [0., 0.], [0., 0.], [0., 0.], [1., 0.]];
    let rates = [storage(&engine, &input), storage(&engine, &output)];
    let low = [
        storage(&engine, &[linear_low, 0., 0., 0., 0., 0., angular_low, 0.]),
        storage(&engine, &[0f32; 8]),
    ];
    let positions = [storage(&engine, &[0f32; 4]), storage(&engine, &[0f32; 4])];
    let coordinates = [storage(&engine, &[0f32; 7]), storage(&engine, &[0f32; 7])];
    let increments = [
        [1e-8f32, 1e-16],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [2e-8, 2e-16],
    ];
    let delta = storage(&engine, &increments);
    let active = storage(&engine, &[1u32]);
    let status = storage(&engine, &[0u32]);
    let p = PairedBuffers {
        rates: [&rates[0], &rates[1]],
        velocities: [&low[0], &low[1]],
        positions: [&positions[0], &positions[1]],
        coordinates: [&coordinates[0], &coordinates[1]],
        projected_low: None,
        active: &active,
        status: &status,
    };
    let short = storage(&engine, &[0f32; 6]);
    assert!(
        MomentumGpu::from_buffers(&engine, &model, &states, None, &short, None, None, Some(&p))
            .is_err()
    );
    let aliased = PairedBuffers {
        velocities: [&low[0], &rates[1]],
        ..p
    };
    let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
    assert!(
        MomentumGpu::from_buffers(
            &engine,
            &model,
            &states,
            None,
            &delta,
            None,
            None,
            Some(&aliased)
        )
        .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    let momentum =
        MomentumGpu::from_buffers(&engine, &model, &states, None, &delta, None, None, Some(&p))
            .unwrap()
            .unwrap();
    let other =
        MomentumGpu::from_buffers(&engine, &model, &states, None, &delta, None, None, Some(&p))
            .unwrap()
            .unwrap();
    assert_eq!(momentum.pipeline, other.pipeline);
    assert_ne!(momentum.params, other.params);
    other.prepare(&engine, 0.25);
    momentum.prepare(&engine, dt as f64);
    let run = || {
        let allocations = molla_core::gpu_metrics::DeviceScope::start().unwrap();
        let traffic = molla_core::gpu_metrics::TransferScope::start().unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        momentum.encode_profiled(&mut encoder, 0, None);
        queue.submit([encoder.finish()]);
        assert_eq!(allocations.finish().observations, 0);
        let traffic = traffic.finish();
        assert_eq!((traffic.upload_bytes, traffic.readback_bytes), (0, 0));
    };
    run();
    assert_eq!(words(&engine, &status), [0]);
    let result = pairs(&engine, &rates[1]);
    let mut expected = [0f64; 6];
    expected[0] = 1.0 + linear_low as f64 + increments[0][0] as f64 + increments[0][1] as f64;
    expected[5] = 1.0 + angular_low as f64 + increments[5][0] as f64 + increments[5][1] as f64;
    for i in 0..6 {
        assert!(
            (result[i] - expected[i]).abs() < 2e-14,
            "rate {i}: {} != {}",
            result[i],
            expected[i]
        );
    }
    let high = words(&engine, states[1].body_qd.device_buffer_current().unwrap());
    let tail = words(&engine, &low[1]);
    for (lane, dof) in [(0, 0), (1, 1), (2, 2), (4, 3), (5, 4), (6, 5)] {
        let v = f32::from_bits(high[lane]) as f64 + f32::from_bits(tail[lane]) as f64;
        assert!(
            (v - expected[dof]).abs() < 2e-14,
            "body lane {lane}: {v} != {}",
            expected[dof]
        );
    }
    let high = words(&engine, states[1].body_q.device_buffer_current().unwrap());
    let tail = words(&engine, &positions[1]);
    let position = f32::from_bits(high[0]) as f64 + f32::from_bits(tail[0]) as f64;
    assert!(
        (position - dt as f64 * expected[0]).abs() < 2e-15,
        "position {position}"
    );
    let high = words(&engine, states[1].joint_qd.device_buffer_current().unwrap());
    assert_eq!(high[0], 1f32.to_bits());
    assert_eq!(high[5], 1f32.to_bits());
    let outputs = [
        &rates[1],
        &low[1],
        &positions[1],
        &coordinates[1],
        states[1].body_q.device_buffer_current().unwrap(),
        states[1].body_qd.device_buffer_current().unwrap(),
        states[1].joint_q.device_buffer_current().unwrap(),
        states[1].joint_qd.device_buffer_current().unwrap(),
    ];
    let before: Vec<_> = outputs.iter().map(|b| words(&engine, b)).collect();
    queue.write_buffer(&rates[0], 44, bytemuck::bytes_of(&f32::NAN));
    run();
    assert_eq!(words(&engine, &status), [1]);
    for (i, b) in outputs.iter().enumerate() {
        assert_eq!(words(&engine, b), before[i]);
    }
    queue.write_buffer(&rates[0], 0, bytemuck::cast_slice(&input));
    run();
    for (i, b) in outputs.iter().enumerate() {
        assert_eq!(words(&engine, b), before[i]);
    }
    queue.write_buffer(&rates[0], 44, bytemuck::bytes_of(&f32::NAN));
    queue.write_buffer(&active, 0, bytemuck::bytes_of(&0u32));
    queue.write_buffer(&status, 0, bytemuck::bytes_of(&0u32));
    run();
    assert_eq!(words(&engine, &status), [0]);
    for (i, b) in outputs.iter().enumerate() {
        assert_eq!(words(&engine, b), before[i]);
    }
    eprintln!(
        "paired momentum high-mass translation/angular corrections, sub-ULP drift, late-NaN and sticky failure pass"
    );
}
