#![cfg(all(feature = "gpu", feature = "resource-counters"))]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
use molla_kernels::{PreparedTireMpmTerrainReadback, TireMpmTerrainRasterParams};

fn buffer(device: &wgpu::Device, bytes: u64, usage: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("terrain snapshot fixture"),
        size: bytes,
        usage,
        mapped_at_creation: false,
    })
}

fn params(n: u32) -> TireMpmTerrainRasterParams {
    TireMpmTerrainRasterParams {
        particle_count: n,
        origin: [-0.5, -0.5],
        spacing: 0.5,
        nx: 3,
        nz: 3,
        rest_height: 0.0,
    }
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_mpm_snapshot_reuses_staging_and_downloads_only_terrain_and_compact_summary() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let usage =
        wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST;
    let heights = buffer(device, 36, usage);
    let sinkage = buffer(device, 36, usage);
    let velocities = buffer(device, 144, usage);
    let compaction = buffer(device, 129 * 16, usage);
    let mut reader = PreparedTireMpmTerrainReadback::new(
        device,
        queue,
        [&heights, &sinkage, &velocities],
        &compaction,
        params(129),
        1e-5,
    )
    .unwrap();
    for step in 0..6 {
        let h: Vec<f32> = (0..9).map(|i| i as f32 * 0.01 + step as f32).collect();
        let s: Vec<f32> = h.iter().map(|v| v * 0.5).collect();
        let v: Vec<[f32; 4]> = h.iter().map(|v| [*v, -v, 0.25, 0.0]).collect();
        let c: Vec<[f32; 4]> = (0..129)
            .map(|i| {
                [
                    if step == 5 {
                        0.0
                    } else {
                        (i % 17) as f32 * 0.01 / (step + 1) as f32
                    },
                    123.0,
                    456.0,
                    789.0,
                ]
            })
            .collect();
        queue.write_buffer(&heights, 0, bytemuck::cast_slice(&h));
        queue.write_buffer(&sinkage, 0, bytemuck::cast_slice(&s));
        queue.write_buffer(&velocities, 0, bytemuck::cast_slice(&v));
        queue.write_buffer(&compaction, 0, bytemuck::cast_slice(&c));
        for diagnostics in [true, false] {
            let resources = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            let dispatches = DispatchScope::start().unwrap();
            let snapshot = reader.read_blocking(diagnostics).unwrap();
            let resources = resources.finish();
            let transfers = transfers.finish();
            let dispatches = dispatches.finish();
            assert_eq!(resources.observations, 0);
            assert!(!resources.invalid && !transfers.overflowed);
            assert_eq!(transfers.upload_bytes, 0);
            assert_eq!(
                transfers.readback_bytes,
                216 + if diagnostics { 16 } else { 0 }
            );
            assert_eq!(dispatches.encoded_dispatches, u64::from(diagnostics));
            assert_eq!(snapshot.terrain.heights, h);
            assert_eq!(snapshot.terrain.sinkage, s);
            assert_eq!(snapshot.terrain.surface_velocity_map, v);
            if diagnostics {
                let summary = snapshot.compaction.unwrap();
                assert_eq!(
                    summary.maximum,
                    c.iter().map(|c| c[0]).fold(0.0_f32, f32::max)
                );
                assert_eq!(
                    summary.active_particles as usize,
                    c.iter().filter(|c| c[0] > 1e-5).count()
                );
            } else {
                assert!(snapshot.compaction.is_none());
            }
        }
    }
    let mut empty = PreparedTireMpmTerrainReadback::new(
        device,
        queue,
        [&heights, &sinkage, &velocities],
        &compaction,
        params(0),
        1e-5,
    )
    .unwrap();
    drop((heights, sinkage, velocities, compaction));
    let snapshot = empty.read_blocking(true).unwrap();
    assert_eq!(snapshot.compaction.unwrap().active_particles, 0);
    assert_eq!(snapshot.compaction.unwrap().maximum, 0.0);
    assert_eq!(snapshot.terrain.heights.len(), 9);
    println!(
        "9-cell bridge: 216 terrain bytes + optional 16 statistics bytes, 0 steady-state buffer allocations/uploads, 0 summary dispatches when disabled"
    );
}

#[test]
#[ignore = "needs a real GPU adapter"]
fn soil_mpm_snapshot_rejects_invalid_layout_before_preparation() {
    let engine = Engine::new_gpu().expect("GPU required");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let usage = wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC;
    let terrain = buffer(device, 144, usage);
    let small = buffer(device, 4, usage);
    let no_copy = buffer(device, 144, wgpu::BufferUsages::STORAGE);
    let compaction = buffer(device, 129 * 16, usage);
    for (outputs, source, p, threshold) in [
        ([&small, &terrain, &terrain], &compaction, params(129), 1e-5),
        (
            [&terrain, &terrain, &no_copy],
            &compaction,
            params(129),
            1e-5,
        ),
        ([&terrain, &terrain, &terrain], &small, params(129), 1e-5),
        (
            [&terrain, &terrain, &terrain],
            &compaction,
            TireMpmTerrainRasterParams {
                nx: 0,
                ..params(129)
            },
            1e-5,
        ),
        (
            [&terrain, &terrain, &terrain],
            &compaction,
            params(129),
            f32::NAN,
        ),
        (
            [&terrain, &terrain, &terrain],
            &compaction,
            params(129),
            -1.0,
        ),
    ] {
        let resources = DeviceScope::start().unwrap();
        assert!(
            PreparedTireMpmTerrainReadback::new(device, queue, outputs, source, p, threshold)
                .is_err()
        );
        assert_eq!(resources.finish().observations, 0);
    }
}
