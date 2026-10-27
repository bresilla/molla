use super::*;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
use molla_kernels::{BrushTerrain, BrushTerrainGpu, PreparedTireBrushTerrain};
use wgpu::util::DeviceExt;

fn buffer(device: &wgpu::Device, bytes: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("resident brush fixture"),
        contents: bytes,
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
    })
}

fn read(engine: &Engine, source: &wgpu::Buffer) -> Vec<u8> {
    let device = engine.device().unwrap();
    let stage = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: source.size(),
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &stage, 0, source.size());
    engine.queue().unwrap().submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    stage.slice(..).map_async(wgpu::MapMode::Read, move |r| {
        tx.send(r).unwrap();
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: Some(std::time::Duration::from_secs(5)),
        })
        .unwrap();
    rx.recv_timeout(std::time::Duration::from_secs(5))
        .unwrap()
        .unwrap();
    let bytes = stage.slice(..).get_mapped_range().to_vec();
    stage.unmap();
    bytes
}

#[test]
#[ignore = "requires GPU; complete prepared chassis loop and eight parity combinations"]
fn resident_terrain_prepared_chassis_full_loop() {
    use molla_kernels::PreparedTireVehicleChassisStep as Step;
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let vehicle = TireVehicleYawGpuKernel::new(device);
    let brush = TireBrushGpuKernel::new(device);
    let planar = sample_planar_from_brush_input();
    let vertical = sample_vertical_input();
    let mut spin = sample_wheel_spin_input();
    spin.dt = planar.dt;
    let suspension = sample_suspension_input();
    let mut rows = sample_brush_rows_from_spin_input(
        tire_vehicle_wheel_spin_state_reference(&spin).unwrap().rows,
    );
    for row in &mut rows.extra_rows {
        row[1] = 1.;
    }
    let build = || {
        vehicle
            .create_resident_chassis_brush_owner_with_vertical(
                device,
                queue,
                &spin,
                &suspension,
                rows.clone(),
                &planar,
                &vertical,
            )
            .unwrap()
    };
    let mut heights = vec![0.25f32; 64];
    let friction = vec![0.6f32; 64];
    let hb = buffer(device, bytemuck::cast_slice(&heights));
    let fb = buffer(device, bytemuck::cast_slice(&friction));
    let terrain = BrushTerrainGpu {
        origin: [-4.; 2],
        spacing: 1.,
        nx: 8,
        nz: 8,
        heights: &hb,
        friction: 0.8,
        friction_map: Some(&fb),
    };
    let zero = buffer(device, &vec![0; spin.wheel_state.len() * 112]);
    let mut positive_loads = 0;
    for parity in 0..8 {
        let mut control = build();
        let mut candidate = build();
        for owner in [&mut control, &mut candidate] {
            if parity & 4 != 0 {
                vehicle
                    .step_wheel_spin_resident(device, queue, &mut owner.brush.spin, planar.dt)
                    .unwrap();
            }
            if parity & 2 != 0 {
                vehicle
                    .step_planar_resident_from_brush_output(
                        device,
                        queue,
                        &mut owner.planar,
                        &zero,
                        spin.wheel_state.len(),
                        planar.dt,
                    )
                    .unwrap();
            }
            if parity & 1 != 0 {
                vehicle
                    .step_vertical_resident_from_suspension(
                        device,
                        queue,
                        &mut owner.vertical,
                        owner.brush.suspension.rows_buffer(),
                        spin.wheel_state.len(),
                        planar.dt,
                    )
                    .unwrap();
            }
        }
        let budget = Step::required_bytes(candidate.wheel_count() as u32);
        let allocations = DeviceScope::start().unwrap();
        assert!(
            Step::new(
                &engine,
                &vehicle,
                &brush,
                &candidate,
                Some(&terrain),
                planar.dt,
                budget - 1
            )
            .is_err()
        );
        assert!(
            Step::new(
                &engine,
                &vehicle,
                &brush,
                &candidate,
                Some(&terrain),
                f32::NAN,
                budget
            )
            .is_err()
        );
        assert_eq!(allocations.finish().observations, 0);
        let prepared = Step::new(
            &engine,
            &vehicle,
            &brush,
            &candidate,
            Some(&terrain),
            planar.dt,
            budget,
        )
        .unwrap();
        let before = read(&engine, candidate.brush.spin.current_state_buffer());
        let saved = candidate.brush.brush_input.extra_rows[0][0];
        candidate.brush.brush_input.extra_rows[0][0] = saved + 1.;
        assert!(prepared.step(&engine, &mut candidate).is_err());
        assert_eq!(
            before,
            read(&engine, candidate.brush.spin.current_state_buffer())
        );
        candidate.brush.brush_input.extra_rows[0][0] = saved;
        let mut foreign = build();
        assert!(prepared.step(&engine, &mut foreign).is_err());
        assert!(prepared.step(&Engine::new_cpu(), &mut candidate).is_err());
        for frame in 0..4 {
            for (i, h) in heights.iter_mut().enumerate() {
                *h = 0.25 + 0.005 * frame as f32 + 0.002 * (i % 8) as f32;
            }
            queue.write_buffer(&hb, 0, bytemuck::cast_slice(&heights));
            let mut commands = spin.wheel_commands.clone();
            commands[0][0] += frame as f32;
            for owner in [&control, &candidate] {
                vehicle
                    .upload_resident_brush_owner_wheel_commands(queue, &owner.brush, &commands)
                    .unwrap();
            }
            let cpu = BrushTerrain {
                origin: [-4.; 2],
                spacing: 1.,
                nx: 8,
                nz: 8,
                heights: &heights,
                friction: 0.8,
                friction_map: Some(&friction),
            };
            let expected = vehicle
                .step_resident_chassis_brush_owner_from_resident_state(
                    &brush,
                    device,
                    queue,
                    &mut control,
                    planar.dt,
                    Some(&cpu),
                )
                .unwrap();
            let resources = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            let dispatch = DispatchScope::start().unwrap();
            let output = prepared.step(&engine, &mut candidate).unwrap();
            assert_eq!(resources.finish().observations, 0);
            assert_eq!(transfers.finish(), Default::default());
            assert_eq!(dispatch.finish().encoded_dispatches, 6);
            let bytes = read(&engine, &output.buffer);
            assert_eq!(bytes, read(&engine, &expected.buffer));
            for sample in bytemuck::cast_slice::<u8, molla_kernels::BrushWheelOutput>(&bytes) {
                assert!(sample.forces.iter().all(|v| v.is_finite()));
                positive_loads += usize::from(sample.forces[1] > 0.);
            }
            for (a, b) in [
                (
                    candidate.planar.current_state_buffer(),
                    control.planar.current_state_buffer(),
                ),
                (
                    candidate.vertical.current_state_buffer(),
                    control.vertical.current_state_buffer(),
                ),
                (
                    candidate.brush.spin.current_state_buffer(),
                    control.brush.spin.current_state_buffer(),
                ),
                (
                    candidate.brush.suspension.rows_buffer(),
                    control.brush.suspension.rows_buffer(),
                ),
            ] {
                assert_eq!(read(&engine, a), read(&engine, b));
            }
        }
    }
    assert!(positive_loads > 0, "nonzero tire contact load required");
    let mut owner = build();
    let prepared = Step::new(
        &engine,
        &vehicle,
        &brush,
        &owner,
        Some(&terrain),
        planar.dt,
        u64::MAX,
    )
    .unwrap();
    engine
        .device_health()
        .unwrap()
        .record(wgpu::DeviceLostReason::Destroyed, "fixture".into());
    assert!(prepared.step(&engine, &mut owner).is_err());
    println!(
        "8 parity combinations, 32 frames, 160 raw comparisons; {positive_loads} positive-load outputs; 192 dispatches, zero hot allocations/transfers; stale/foreign/budget/dt/health rejection pass"
    );
}

#[test]
#[ignore = "requires GPU; exact retained brush/legacy parity and hot resource contract"]
fn resident_terrain_brush_exact_and_resources() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireBrushGpuKernel::new(device);
    let spin = sample_wheel_spin_input();
    let states = tire_vehicle_wheel_spin_state_reference(&spin).unwrap().rows;
    let mut input = sample_brush_rows_from_spin_input(states);
    for row in &mut input.extra_rows {
        row[1] = 1.;
    }
    let wheels = tire_vehicle_brush_rows_from_spin_reference(&input)
        .unwrap()
        .wheels;
    let wheel_buffer = buffer(device, bytemuck::cast_slice(&wheels));
    for flags in 0..3 {
        let heights: Vec<f32> = (0..64).map(|i| 0.005 * (i % 8) as f32).collect();
        let friction = vec![0.45f32; 64];
        let hb = buffer(device, bytemuck::cast_slice(&heights));
        let fb = buffer(device, bytemuck::cast_slice(&friction));
        let cpu = BrushTerrain {
            origin: [-4.; 2],
            spacing: 1.,
            nx: 8,
            nz: 8,
            heights: &heights,
            friction: 0.8,
            friction_map: (flags == 2).then_some(friction.as_slice()),
        };
        let gpu = BrushTerrainGpu {
            origin: cpu.origin,
            spacing: cpu.spacing,
            nx: 8,
            nz: 8,
            heights: &hb,
            friction: 0.8,
            friction_map: (flags == 2).then_some(&fb),
        };
        let count = wheels.len() as u32;
        let prepared = PreparedTireBrushTerrain::new(
            &engine,
            &kernel,
            count,
            (flags > 0).then_some(&gpu),
            PreparedTireBrushTerrain::required_bytes(count),
        )
        .unwrap();
        let expected = kernel
            .dispatch_input_buffer_resident(
                device,
                queue,
                &wheel_buffer,
                wheels.len(),
                (flags > 0).then_some(&cpu),
            )
            .unwrap();
        let mut encoder = device.create_command_encoder(&Default::default());
        let resources = DeviceScope::start().unwrap();
        let transfers = TransferScope::start().unwrap();
        let dispatch = DispatchScope::start().unwrap();
        for _ in 0..8 {
            prepared
                .encode(&engine, &mut encoder, &wheel_buffer)
                .unwrap();
        }
        assert_eq!(resources.finish().observations, 0);
        assert_eq!(transfers.finish(), Default::default());
        assert_eq!(dispatch.finish().encoded_dispatches, 8);
        let output = prepared.output();
        drop(prepared);
        drop(hb);
        drop(fb);
        queue.submit([encoder.finish()]);
        assert_eq!(
            read(&engine, &output.buffer),
            read(&engine, &expected.buffer)
        );
    }
    let resources = DeviceScope::start().unwrap();
    assert!(PreparedTireBrushTerrain::new(&engine, &kernel, 0, None, u64::MAX).is_err());
    assert!(PreparedTireBrushTerrain::new(&engine, &kernel, 1, None, 259).is_err());
    assert_eq!(resources.finish().observations, 0);
    let prepared =
        PreparedTireBrushTerrain::new(&engine, &kernel, wheels.len() as u32, None, u64::MAX)
            .unwrap();
    let short = buffer(device, &[0; 4]);
    let mut encoder = device.create_command_encoder(&Default::default());
    let dispatch = DispatchScope::start().unwrap();
    assert!(prepared.encode(&engine, &mut encoder, &short).is_err());
    assert!(
        prepared
            .encode(&Engine::new_cpu(), &mut encoder, &wheel_buffer)
            .is_err()
    );
    assert_eq!(dispatch.finish().encoded_dispatches, 0);
    engine
        .device_health()
        .unwrap()
        .record(wgpu::DeviceLostReason::Destroyed, "fixture".into());
    assert!(
        prepared
            .encode(&engine, &mut encoder, &wheel_buffer)
            .is_err()
    );
    println!(
        "flat/height/friction exact; 24 retained encodes zero hot allocations/transfers; lifetime/budget/source/health checks pass"
    );
}

#[test]
#[ignore = "requires GPU; actual resident chassis consumer with changing GPU terrain"]
fn resident_terrain_chassis_consumer_exact() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let vehicle = TireVehicleYawGpuKernel::new(device);
    let kernel = TireBrushGpuKernel::new(device);
    let planar = sample_planar_from_brush_input();
    let vertical = sample_vertical_input();
    let mut spin = sample_wheel_spin_input();
    spin.dt = planar.dt;
    let suspension = sample_suspension_input();
    let mut rows = sample_brush_rows_from_spin_input(
        tire_vehicle_wheel_spin_state_reference(&spin).unwrap().rows,
    );
    for row in &mut rows.extra_rows {
        row[1] = 1.;
    }
    let build = || {
        vehicle
            .create_resident_chassis_brush_owner_with_vertical(
                device,
                queue,
                &spin,
                &suspension,
                rows.clone(),
                &planar,
                &vertical,
            )
            .unwrap()
    };
    let mut control = build();
    let mut candidate = build();
    let mut heights = vec![0f32; 64];
    let friction = vec![0.6f32; 64];
    let hb = buffer(device, bytemuck::cast_slice(&heights));
    let fb = buffer(device, bytemuck::cast_slice(&friction));
    let terrain = BrushTerrainGpu {
        origin: [-4.; 2],
        spacing: 1.,
        nx: 8,
        nz: 8,
        heights: &hb,
        friction: 0.8,
        friction_map: Some(&fb),
    };
    let prepared = PreparedTireBrushTerrain::new(
        &engine,
        &kernel,
        candidate.wheel_count() as u32,
        Some(&terrain),
        u64::MAX,
    )
    .unwrap();
    let mismatched = PreparedTireBrushTerrain::new(
        &engine,
        &kernel,
        (candidate.wheel_count() + 1) as u32,
        None,
        u64::MAX,
    )
    .unwrap();
    let before = read(&engine, candidate.brush.spin.current_state_buffer());
    assert!(
        vehicle
            .step_resident_chassis_brush_owner_with_gpu_terrain(
                &engine,
                &mut candidate,
                planar.dt,
                &mismatched
            )
            .is_err()
    );
    assert_eq!(
        before,
        read(&engine, candidate.brush.spin.current_state_buffer())
    );
    for step in 0..4 {
        for (i, h) in heights.iter_mut().enumerate() {
            *h = 0.01 * step as f32 + 0.002 * (i % 8) as f32;
        }
        queue.write_buffer(&hb, 0, bytemuck::cast_slice(&heights));
        let cpu = BrushTerrain {
            origin: [-4.; 2],
            spacing: 1.,
            nx: 8,
            nz: 8,
            heights: &heights,
            friction: 0.8,
            friction_map: Some(&friction),
        };
        let expected = vehicle
            .step_resident_chassis_brush_owner_from_resident_state(
                &kernel,
                device,
                queue,
                &mut control,
                planar.dt,
                Some(&cpu),
            )
            .unwrap();
        let got = vehicle
            .step_resident_chassis_brush_owner_with_gpu_terrain(
                &engine,
                &mut candidate,
                planar.dt,
                &prepared,
            )
            .unwrap();
        for (a, b) in [
            (&got.buffer, &expected.buffer),
            (
                candidate.planar.current_state_buffer(),
                control.planar.current_state_buffer(),
            ),
            (
                candidate.vertical.current_state_buffer(),
                control.vertical.current_state_buffer(),
            ),
            (
                candidate.brush.spin.current_state_buffer(),
                control.brush.spin.current_state_buffer(),
            ),
            (
                candidate.brush.suspension.rows_buffer(),
                control.brush.suspension.rows_buffer(),
            ),
        ] {
            assert_eq!(read(&engine, a), read(&engine, b));
        }
    }
    println!(
        "4 changing-terrain chassis frames: brush, planar, vertical, spin and suspension bytes exact; count mismatch rejects before state advance"
    );
}
