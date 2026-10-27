#![cfg(all(feature = "gpu", feature = "resource-counters"))]
use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
use molla_kernels::{PreparedSoilSurface, SoilSurfaceConfig};
use std::time::Duration;
use wgpu::util::DeviceExt;

fn cfg(nx: u32, nz: u32) -> SoilSurfaceConfig {
    SoilSurfaceConfig {
        origin: [-1.0, -2.0],
        spacing: 0.125,
        nx,
        nz,
        floor_height: -0.5,
    }
}

fn oracle(c: SoilSurfaceConfig, points: &[[f32; 4]]) -> (Vec<f32>, Vec<u32>, u32) {
    let axis = |origin: f32, n: u32| -> Vec<f32> {
        (0..=n)
            .map(|i| (origin as f64 + i as f64 * c.spacing as f64) as f32)
            .collect()
    };
    let xs = axis(c.origin[0], c.nx);
    let zs = axis(c.origin[1], c.nz);
    let mut heights = vec![c.floor_height; (c.nx * c.nz) as usize];
    let mut counts = vec![0; heights.len()];
    let mut outside = 0;
    for p in points {
        let x = xs.windows(2).position(|w| p[0] >= w[0] && p[0] < w[1]);
        let z = zs.windows(2).position(|w| p[2] >= w[0] && p[2] < w[1]);
        let (Some(x), Some(z)) = (x, z) else {
            outside += 1;
            continue;
        };
        let at = z * c.nx as usize + x;
        heights[at] = heights[at].max(p[1]);
        counts[at] += 1;
    }
    (heights, counts, outside)
}

fn poll_surface(surface: &mut PreparedSoilSurface) -> molla_kernels::SoilSurfaceObservation {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        if let Some(snapshot) = surface.poll_snapshot().unwrap() {
            return snapshot;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "nonblocking surface deadline"
        );
        std::thread::yield_now();
    }
}

#[test]
#[ignore = "requires GPU; bounded asynchronous geometry, generations and reuse"]
fn geometric_surface_async_revision_backpressure_and_reset() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let points = [[-0.95, 0.75, -1.95, 0f32]];
    let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&points),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let config = cfg(2, 1);
    let mut surface = PreparedSoilSurface::new(device, queue, &source, 1, config, 4096).unwrap();
    let other = PreparedSoilSurface::new(device, queue, &source, 1, config, 4096).unwrap();
    let bytes = surface.retained_bytes();
    let resources = DeviceScope::start().unwrap();
    let traffic = TransferScope::start().unwrap();
    assert!(surface.try_request_snapshot(4).unwrap());
    assert_eq!(traffic.finish().readback_bytes, 0);
    let dispatches = DispatchScope::start().unwrap();
    assert!(!surface.try_request_snapshot(5).unwrap());
    assert!(surface.try_request_snapshot(4).is_err());
    assert!(surface.read_blocking(Duration::from_secs(1)).is_err());
    assert_eq!(dispatches.finish().encoded_dispatches, 0);
    queue.write_buffer(
        &source,
        0,
        bytemuck::cast_slice(&[[-0.95f32, 1.25, -1.95, 0.0]]),
    );
    let mut encoder = device.create_command_encoder(&Default::default());
    surface.encode(&mut encoder);
    queue.submit([encoder.finish()]);
    let traffic = TransferScope::start().unwrap();
    let captured = poll_surface(&mut surface);
    assert_eq!(traffic.finish().readback_bytes, surface.readback_bytes());
    assert_eq!((captured.generation(), captured.revision()), (0, 4));
    assert_eq!(captured.surface().heights, oracle(config, &points).0);
    assert!(surface.is_current_snapshot(&captured));
    assert!(!other.is_current_snapshot(&captured));
    assert!(surface.poll_snapshot().unwrap().is_none());
    assert!(surface.try_request_snapshot(5).unwrap());
    let updated = poll_surface(&mut surface);
    assert_eq!(updated.surface().heights[0], 1.25);
    assert!(surface.try_request_snapshot(6).unwrap());
    surface.reset_snapshots().unwrap();
    assert_eq!(surface.snapshot_generation(), 1);
    assert!(!surface.is_current_snapshot(&captured));
    assert!(!surface.try_request_snapshot(0).unwrap());
    let traffic = TransferScope::start().unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while surface.snapshot_pending() {
        assert!(surface.poll_snapshot().unwrap().is_none());
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert_eq!(traffic.finish().readback_bytes, 0);
    assert!(surface.try_request_snapshot(0).unwrap());
    let restarted = poll_surface(&mut surface);
    assert_eq!((restarted.generation(), restarted.revision()), (1, 0));
    assert_eq!(restarted.surface().heights[0], 1.25);
    assert_eq!(surface.retained_bytes(), bytes);
    assert_eq!(resources.finish().observations, 0);
    assert!(surface.try_request_snapshot(1).unwrap());
    drop(surface);
    device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
}

#[test]
#[ignore = "requires GPU; asynchronous decode failure and source repair"]
fn geometric_surface_async_invalid_payload_recovery() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&[[f32::NAN, 0.75, -1.95, 0.0]]),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let mut surface = PreparedSoilSurface::new(device, queue, &source, 1, cfg(2, 1), 4096).unwrap();
    assert!(surface.try_request_snapshot(0).unwrap());
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    loop {
        match surface.poll_snapshot() {
            Err(_) => break,
            Ok(None) => {}
            Ok(Some(_)) => panic!("invalid source published"),
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    }
    assert!(!surface.snapshot_pending());
    queue.write_buffer(
        &source,
        0,
        bytemuck::cast_slice(&[[-0.95f32, 0.75, -1.95, 0.0]]),
    );
    assert!(surface.try_request_snapshot(1).unwrap());
    assert_eq!(poll_surface(&mut surface).surface().heights[0], 0.75);
    assert_eq!(
        surface
            .read_blocking(Duration::from_secs(10))
            .unwrap()
            .heights[0],
        0.75
    );
}

#[test]
#[ignore = "requires GPU; independent geometric surface oracle and resident reuse"]
fn geometric_surface_gpu_oracle_and_reuse() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for (c, n, crowded) in [
        (cfg(2, 1), 0, false),
        (cfg(2, 1), 1, false),
        (cfg(7, 3), 137, false),
        (cfg(129, 65), 5000, false),
        (cfg(7, 3), 5000, true),
        (
            SoilSurfaceConfig {
                origin: [-3.2, -2.0],
                spacing: 0.1,
                nx: 64,
                nz: 40,
                floor_height: -0.5,
            },
            5000,
            false,
        ),
    ] {
        let mut points: Vec<_> = (0..n)
            .map(|i| {
                let cell = if crowded { 1 } else { (i * 17) % (c.nx * c.nz) };
                [
                    c.origin[0] + ((cell % c.nx) as f32 + 0.5) * c.spacing,
                    (i % 19) as f32 * 0.125 - 1.0,
                    c.origin[1] + ((cell / c.nx) as f32 + 0.5) * c.spacing,
                    0.0,
                ]
            })
            .collect();
        if n > 4 {
            points[0][0] = c.origin[0];
            points[1][0] = c.origin[0] - 0.125;
            points[2][0] = c.origin[0] + c.nx as f32 * c.spacing;
            points[3][2] = c.origin[1] + c.nz as f32 * c.spacing;
        }
        if c.spacing == 0.1 {
            let mut i = 0;
            for (axis, origin, cells) in [(0, c.origin[0], c.nx), (2, c.origin[1], c.nz)] {
                for edge in 0..=cells {
                    let v = (origin as f64 + edge as f64 * c.spacing as f64) as f32;
                    for adjacent in [v.next_down(), v, v.next_up()] {
                        points[i][axis] = adjacent;
                        i += 1;
                    }
                }
            }
        }
        let payload = if points.is_empty() {
            vec![[0f32; 4]]
        } else {
            points.clone()
        };
        let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&payload),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        let before = DeviceScope::start().unwrap();
        assert!(PreparedSoilSurface::new(device, queue, &source, n, c, 0).is_err());
        assert_eq!(before.finish().observations, 0);
        let mut surface =
            PreparedSoilSurface::new(device, queue, &source, n, c, c.required_bytes(n).unwrap())
                .unwrap();
        assert_eq!(surface.retained_bytes(), c.required_bytes(n).unwrap());
        assert!(surface.matches(device, queue, &source, n));
        assert!(!surface.matches(device, queue, &source, n + 1));
        let expected = oracle(c, &points);
        for reference in [false, true, false] {
            let allocation = DeviceScope::start().unwrap();
            let traffic = TransferScope::start().unwrap();
            let dispatch = DispatchScope::start().unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            if reference {
                surface.encode_reference(&mut encoder);
            } else {
                surface.encode(&mut encoder);
            }
            queue.submit([encoder.finish()]);
            assert_eq!(allocation.finish().observations, 0);
            let t = traffic.finish();
            assert_eq!((t.readback_bytes, t.upload_bytes), (0, 0));
            assert!(dispatch.finish().encoded_dispatches > 0);
            let traffic = TransferScope::start().unwrap();
            let snapshot = surface.read_blocking(Duration::from_secs(15)).unwrap();
            assert_eq!(traffic.finish().readback_bytes, surface.readback_bytes());
            assert_eq!(
                snapshot.outside_particles, expected.2,
                "reference={reference}"
            );
            for cell in 0..snapshot.heights.len() {
                assert_eq!(
                    snapshot.counts[cell], expected.1[cell],
                    "count cell={cell} reference={reference}"
                );
                assert_eq!(
                    snapshot.heights[cell], expected.0[cell],
                    "height cell={cell} reference={reference}"
                );
            }
        }
        eprintln!(
            "surface oracle particles={n} cells={} crowded={crowded} passed",
            c.nx * c.nz
        );
    }
}

#[test]
#[ignore = "requires GPU; invalid positions and readback recovery"]
fn geometric_surface_gpu_invalid_and_recovery() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let points = [[-0.95, 0.75, -1.95, 0f32]];
    let source = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: None,
        contents: bytemuck::cast_slice(&points),
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
    });
    let mut surface = PreparedSoilSurface::new(device, queue, &source, 1, cfg(2, 1), 4096).unwrap();
    assert!(surface.read_blocking(Duration::ZERO).is_err());
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        for axis in 0..3 {
            let mut point = points;
            point[0][axis] = bad;
            queue.write_buffer(&source, 0, bytemuck::cast_slice(&point));
            let mut encoder = device.create_command_encoder(&Default::default());
            surface.encode(&mut encoder);
            queue.submit([encoder.finish()]);
            assert!(surface.read_blocking(Duration::from_secs(15)).is_err());
        }
    }
    queue.write_buffer(&source, 0, bytemuck::cast_slice(&points));
    let mut encoder = device.create_command_encoder(&Default::default());
    surface.encode(&mut encoder);
    queue.submit([encoder.finish()]);
    assert_eq!(
        surface
            .read_blocking(Duration::from_secs(15))
            .unwrap()
            .heights,
        [0.75, -0.5]
    );
}
