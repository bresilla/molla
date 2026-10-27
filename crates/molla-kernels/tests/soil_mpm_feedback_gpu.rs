#![cfg(all(feature = "gpu", feature = "resource-counters"))]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
use molla_kernels::{
    TireMpmContactCell, TireMpmGpuInPlaceParams, TireMpmGpuInput, TireMpmGpuKernel,
    TireMpmGpuOutput, TireMpmTerrainRasterParams, tire_mpm_reference_feedback,
};

const USAGE: wgpu::BufferUsages = wgpu::BufferUsages::STORAGE
    .union(wgpu::BufferUsages::COPY_SRC)
    .union(wgpu::BufferUsages::COPY_DST);

fn buffer(device: &wgpu::Device, size: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MPM feedback test"),
        size,
        usage,
        mapped_at_creation: false,
    })
}

fn state() -> TireMpmGpuOutput {
    TireMpmGpuOutput {
        positions: (0..129)
            .map(|i| {
                [
                    (i % 7) as f32 * 0.04 - 0.1,
                    0.2 + (i / 7) as f32 * 0.002,
                    (i % 3) as f32 * 0.04,
                    0.0,
                ]
            })
            .collect(),
        velocities: vec![[0.04, -0.01, 0.02, 0.0]; 129],
        compaction: vec![[0.0; 4]; 129],
    }
}

fn particles(device: &wgpu::Device, queue: &wgpu::Queue) -> [wgpu::Buffer; 3] {
    let s = state();
    [&s.positions, &s.velocities, &s.compaction].map(|rows| {
        let b = buffer(device, rows.len() as u64 * 16, USAGE);
        queue.write_buffer(&b, 0, bytemuck::cast_slice(rows));
        b
    })
}

fn step(i: usize) -> (Vec<TireMpmContactCell>, TireMpmGpuInPlaceParams) {
    let contacts = (0..i % 4)
        .map(|j| TireMpmContactCell {
            center_radius: [j as f32 * 0.03, 0.2, 0.0, 0.3],
            force_pressure: [i as f32 - 2.0, -10.0, j as f32, 1200.0 + i as f32 * 50.0],
        })
        .collect();
    (
        contacts,
        TireMpmGpuInPlaceParams {
            particle_count: [129, 65, 1, 64, 0, 129][i % 6],
            particle_mass: 1.0,
            dt: 0.0002 + i as f32 * 0.00001,
            plastic_compaction_gain: 0.0002,
            max_compaction: 0.15,
            damping: i as f32 * 2.0,
        },
    )
}

fn reference(
    s: &mut TireMpmGpuOutput,
    contacts: &[TireMpmContactCell],
    p: &TireMpmGpuInPlaceParams,
) {
    let n = p.particle_count as usize;
    if n == 0 {
        return;
    }
    let r = tire_mpm_reference_feedback(&TireMpmGpuInput {
        positions: s.positions[..n].to_vec(),
        velocities: s.velocities[..n].to_vec(),
        compaction: s.compaction[..n].to_vec(),
        contacts: contacts.to_vec(),
        particle_mass: p.particle_mass,
        dt: p.dt,
        plastic_compaction_gain: p.plastic_compaction_gain,
        max_compaction: p.max_compaction,
        damping: p.damping,
    })
    .unwrap();
    s.positions[..n].copy_from_slice(&r.positions);
    s.velocities[..n].copy_from_slice(&r.velocities);
    s.compaction[..n].copy_from_slice(&r.compaction);
}

fn read(device: &wgpu::Device, queue: &wgpu::Queue, source: &wgpu::Buffer) -> Vec<f32> {
    let stage = buffer(
        device,
        source.size(),
        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    encoder.copy_buffer_to_buffer(source, 0, &stage, 0, source.size());
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    stage
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    rx.recv().unwrap().unwrap();
    let result = bytemuck::cast_slice(&stage.slice(..).get_mapped_range()).to_vec();
    stage.unmap();
    result
}

fn assert_near(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    for (i, (a, b)) in a.iter().zip(b).enumerate() {
        assert!((a - b).abs() <= 2e-5, "index {i}: {a} != {b}");
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn retained_feedback_queued_changes_match_legacy_and_cpu_without_hot_resources() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireMpmGpuKernel::new(device);
    let candidate = particles(device, queue);
    let legacy = particles(device, queue);
    let mut retained = kernel
        .prepare_resident_in_place(
            device,
            queue,
            &candidate[0],
            &candidate[1],
            &candidate[2],
            129,
            3,
        )
        .unwrap();
    let inputs: Vec<_> = (0..12).map(step).collect();
    let mut cpu = state();
    for (c, p) in &inputs {
        reference(&mut cpu, c, p);
        if p.particle_count > 0 {
            kernel
                .dispatch_in_place(device, queue, &legacy[0], &legacy[1], &legacy[2], c, p)
                .unwrap();
        }
    }
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let before = device.get_internal_counters().hal;
    let resources = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let dispatches = DispatchScope::start().unwrap();
    for (c, p) in &inputs {
        retained.dispatch(device, queue, &kernel, c, p).unwrap();
    }
    let resources = resources.finish();
    let transfers = transfers.finish();
    let dispatches = dispatches.finish();
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
    let after = device.get_internal_counters().hal;
    assert_eq!(resources.observations, 0);
    assert!(!resources.invalid && !transfers.overflowed);
    assert_eq!(before.bind_groups.read(), after.bind_groups.read());
    assert_eq!(
        before.compute_pipelines.read(),
        after.compute_pipelines.read()
    );
    let expected_uploads: u64 = inputs
        .iter()
        .filter(|(_, p)| p.particle_count > 0)
        .map(|(c, _)| 48 + c.len() as u64 * 32)
        .sum();
    assert_eq!(transfers.upload_bytes, expected_uploads);
    assert_eq!(transfers.readback_bytes, 0);
    assert_eq!(dispatches.encoded_dispatches, 10);
    for (i, expected) in [&cpu.positions, &cpu.velocities, &cpu.compaction]
        .iter()
        .enumerate()
    {
        let result = read(device, queue, &candidate[i]);
        assert_eq!(result, read(device, queue, &legacy[i]));
        assert_near(&result, bytemuck::cast_slice(expected));
    }
    println!(
        "10 queued steps: exact legacy parity, CPU <=2e-5, 0 hot buffer allocations/readbacks, {expected_uploads} upload bytes"
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn immutable_records_chain_gpu_contacts_feedback_and_raster_in_one_encoder() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireMpmGpuKernel::new(device);
    let data = particles(device, queue);
    let contacts = buffer(device, 96, USAGE);
    let inputs: Vec<_> = (0..6).map(step).collect();
    let sources: Vec<_> = inputs
        .iter()
        .map(|(c, _)| {
            let source = buffer(device, 96, USAGE);
            if !c.is_empty() {
                queue.write_buffer(&source, 0, bytemuck::cast_slice(c));
            }
            source
        })
        .collect();
    let records: Vec<_> = inputs
        .iter()
        .map(|(c, p)| {
            kernel
                .prepare_in_place_from_buffers(
                    device,
                    &data[0],
                    &data[1],
                    &data[2],
                    &contacts,
                    c.len() as u32,
                    p,
                )
                .unwrap()
        })
        .collect();
    let snapshots: Vec<_> = inputs
        .iter()
        .map(|_| buffer(device, 129 * 16, USAGE))
        .collect();
    let outputs = [
        buffer(device, 16 * 4, USAGE),
        buffer(device, 16 * 4, USAGE),
        buffer(device, 16 * 16, USAGE),
    ];
    let raster_params = TireMpmTerrainRasterParams {
        particle_count: 129,
        origin: [-0.3, -0.3],
        spacing: 0.2,
        nx: 4,
        nz: 4,
        rest_height: 0.0,
    };
    let raster = kernel
        .prepare_raster_to_buffers(
            device,
            &data[0],
            &data[1],
            &data[2],
            &outputs[0],
            &outputs[1],
            &outputs[2],
            &raster_params,
            [0.0; 3],
        )
        .unwrap();
    let mut encoder = device.create_command_encoder(&Default::default());
    let resources = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    for ((record, source), snapshot) in records.iter().zip(&sources).zip(&snapshots) {
        encoder.copy_buffer_to_buffer(source, 0, &contacts, 0, 96);
        record.encode(&mut encoder);
        encoder.copy_buffer_to_buffer(&data[0], 0, snapshot, 0, snapshot.size());
    }
    raster.encode(&mut encoder);
    assert_eq!(resources.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!(transfers.upload_bytes, 0);
    assert_eq!(transfers.readback_bytes, 0);
    drop((records, sources, contacts, data, raster));
    queue.submit([encoder.finish()]);
    let mut cpu = state();
    for ((c, p), snapshot) in inputs.iter().zip(&snapshots) {
        reference(&mut cpu, c, p);
        assert_near(
            &read(device, queue, snapshot),
            bytemuck::cast_slice(&cpu.positions),
        );
    }
    let expected = kernel
        .dispatch_raster_blocking(
            device,
            queue,
            &cpu.positions,
            &cpu.velocities,
            &cpu.compaction,
            &raster_params,
            [0.0; 3],
        )
        .unwrap();
    assert_near(&read(device, queue, &outputs[0]), &expected.heights);
    assert_near(&read(device, queue, &outputs[1]), &expected.sinkage);
    assert_near(
        &read(device, queue, &outputs[2]),
        bytemuck::cast_slice(&expected.surface_velocity_map),
    );
    println!(
        "6 immutable records including count0: ordered GPU contact copies, feedback and raster; owner drop before submit, 0 encoding allocations/transfers"
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn retained_feedback_rejects_invalid_inputs_context_capacity_and_aliases() {
    let engine = Engine::new_gpu().unwrap();
    let other = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let kernel = TireMpmGpuKernel::new(device);
    let wrong_kernel = TireMpmGpuKernel::new(other.device().unwrap());
    let data = particles(device, queue);
    let contact = buffer(device, 96, USAGE);
    let small = buffer(device, 4, USAGE);
    let wrong_usage = buffer(device, 129 * 16, wgpu::BufferUsages::COPY_DST);
    let mut retained = kernel
        .prepare_resident_in_place(device, queue, &data[0], &data[1], &data[2], 129, 3)
        .unwrap();
    let (c, p) = step(3);
    let resources = DeviceScope::start().unwrap();
    let transfers = TransferScope::start().unwrap();
    let dispatches = DispatchScope::start().unwrap();
    for bad in [
        TireMpmGpuInPlaceParams {
            dt: f32::NAN,
            ..p.clone()
        },
        TireMpmGpuInPlaceParams {
            damping: -1.0,
            ..p.clone()
        },
        TireMpmGpuInPlaceParams {
            particle_mass: 0.0,
            ..p.clone()
        },
        TireMpmGpuInPlaceParams {
            plastic_compaction_gain: f32::INFINITY,
            ..p.clone()
        },
        TireMpmGpuInPlaceParams {
            particle_count: 130,
            ..p.clone()
        },
    ] {
        assert!(retained.dispatch(device, queue, &kernel, &c, &bad).is_err());
    }
    assert!(
        retained
            .dispatch(device, queue, &kernel, &[c[0]; 4], &p)
            .is_err()
    );
    for bad in [
        TireMpmContactCell {
            center_radius: [0., 0., 0., -1.],
            ..c[0]
        },
        TireMpmContactCell {
            force_pressure: [0., 0., f32::NAN, 0.],
            ..c[0]
        },
    ] {
        assert!(
            retained
                .dispatch(device, queue, &kernel, &[bad], &p)
                .is_err()
        );
    }
    assert!(
        retained
            .dispatch(device, other.queue().unwrap(), &kernel, &c, &p)
            .is_err()
    );
    assert!(
        retained
            .dispatch(
                other.device().unwrap(),
                other.queue().unwrap(),
                &kernel,
                &c,
                &p
            )
            .is_err()
    );
    assert!(
        retained
            .dispatch(device, queue, &wrong_kernel, &c, &p)
            .is_err()
    );
    for (pos, vel, contacts, n) in [
        (&small, &data[1], &contact, 3),
        (&wrong_usage, &data[1], &contact, 3),
        (&data[0], &data[0], &contact, 3),
        (&data[0], &data[1], &data[2], 3),
        (&data[0], &data[1], &contact, 4),
    ] {
        assert!(
            kernel
                .prepare_in_place_from_buffers(device, pos, vel, &data[2], contacts, n, &p)
                .is_err()
        );
    }
    assert!(
        kernel
            .prepare_resident_in_place(device, queue, &data[0], &data[1], &data[2], 129, u32::MAX)
            .is_err()
    );
    assert!(
        wrong_kernel
            .prepare_in_place_from_buffers(device, &data[0], &data[1], &data[2], &contact, 3, &p)
            .is_err()
    );
    assert_eq!(resources.finish().observations, 0);
    let transfers = transfers.finish();
    assert_eq!(transfers.upload_bytes, 0);
    assert_eq!(transfers.readback_bytes, 0);
    assert_eq!(dispatches.finish().encoded_dispatches, 0);
    let initial = state();
    for (i, expected) in [&initial.positions, &initial.velocities, &initial.compaction]
        .iter()
        .enumerate()
    {
        assert_eq!(
            read(device, queue, &data[i]),
            bytemuck::cast_slice::<_, f32>(expected)
        );
    }
}
