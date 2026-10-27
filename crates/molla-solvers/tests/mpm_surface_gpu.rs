#![cfg(all(feature = "gpu", feature = "resource-counters"))]
use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
use molla_kernels::SoilSurfaceConfig;
use molla_solvers::{MpmConfig, MpmSimulation};
use std::time::Duration;

#[test]
#[ignore = "requires GPU; asynchronous MPM surface ordering and source invalidation"]
fn mpm_surface_async_source_ordering_and_invalidation() {
    let engine = Engine::new_gpu().unwrap();
    let queue = engine.queue().unwrap();
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 16;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 2;
    let mut sim = MpmSimulation::new(&engine, cfg).unwrap();
    sim.set_particles(queue, &[[-0.125, 0.25, -0.125]]).unwrap();
    let config = SoilSurfaceConfig {
        origin: [-1.0; 2],
        spacing: 0.125,
        nx: 16,
        nz: 16,
        floor_height: -1.0,
    };
    let mut surface = sim
        .prepare_geometric_surface(&engine, config, 1 << 20)
        .unwrap();
    let allocations = DeviceScope::start().unwrap();
    let traffic = TransferScope::start().unwrap();
    assert!(
        sim.try_request_geometric_surface(&engine, &mut surface, 12)
            .unwrap()
    );
    assert_eq!(traffic.finish().readback_bytes, 0);
    let dispatch = DispatchScope::start().unwrap();
    assert!(
        sim.read_geometric_surface(&engine, &mut surface, Duration::from_secs(1))
            .is_err()
    );
    assert!(
        !sim.try_request_geometric_surface(&engine, &mut surface, 13)
            .unwrap()
    );
    assert_eq!(dispatch.finish().encoded_dispatches, 0);
    assert_eq!(allocations.finish().observations, 0);
    sim.step(&engine, 0.0005).unwrap();
    let allocations = DeviceScope::start().unwrap();
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    let sample = loop {
        if let Some(sample) = sim.poll_geometric_surface(&engine, &mut surface).unwrap() {
            break sample;
        }
        assert!(std::time::Instant::now() < deadline);
        std::thread::yield_now();
    };
    assert_eq!(sample.revision(), 12);
    assert_eq!(sample.surface().heights[7 * 16 + 7], 0.25);
    assert_eq!(allocations.finish().observations, 0);
    sim.set_particles(queue, &[[0.0, 0.25, 0.0], [0.1, 0.5, 0.0]])
        .unwrap();
    let dispatch = DispatchScope::start().unwrap();
    assert!(
        sim.try_request_geometric_surface(&engine, &mut surface, 13)
            .is_err()
    );
    assert!(sim.poll_geometric_surface(&engine, &mut surface).is_err());
    assert_eq!(dispatch.finish().encoded_dispatches, 0);
    let other = Engine::new_gpu().unwrap();
    assert!(sim.poll_geometric_surface(&other, &mut surface).is_err());
}

#[test]
#[ignore = "requires GPU; retained MPM surface, invalidation and particle-readback control"]
fn mpm_surface_source_lifecycle_and_evolved_geometry() {
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    let mut cfg = MpmConfig::sand();
    cfg.grid_size = 16;
    cfg.grid_origin = [-1.0; 3];
    cfg.cell_size = 0.125;
    cfg.max_particles = 2;
    let mut sim = MpmSimulation::new(&engine, cfg).unwrap();
    let queue = engine.queue().unwrap();
    sim.set_particles(queue, &[[-0.125, 0.25, -0.125], [0.125, 0.375, 0.125]])
        .unwrap();
    let config = SoilSurfaceConfig {
        origin: [-1.0; 2],
        spacing: 0.125,
        nx: 16,
        nz: 16,
        floor_height: -1.0,
    };
    let mut surface = sim
        .prepare_geometric_surface(&engine, config, 1 << 20)
        .unwrap();
    let allocations = DeviceScope::start().unwrap();
    let traffic = TransferScope::start().unwrap();
    let first = sim
        .read_geometric_surface(&engine, &mut surface, Duration::from_secs(15))
        .unwrap();
    assert_eq!(allocations.finish().observations, 0);
    assert_eq!(traffic.finish().readback_bytes, 16 * 16 * 24);
    assert_eq!(first.counts.iter().sum::<u32>(), 2);
    assert_eq!(first.heights[7 * 16 + 7], 0.25);
    assert_eq!(first.heights[9 * 16 + 9], 0.375);
    sim.step(&engine, 0.0005).unwrap();
    let evolved = sim
        .read_geometric_surface(&engine, &mut surface, Duration::from_secs(15))
        .unwrap();
    let points = sim.read_positions(&engine).unwrap();
    for p in &points {
        let at = (((p[2] + 1.0) / 0.125).floor() as usize) * 16
            + ((p[0] + 1.0) / 0.125).floor() as usize;
        assert_eq!(evolved.heights[at], p[1]);
        assert_eq!(evolved.counts[at], 1);
    }
    assert_ne!(first.heights, evolved.heights);
    sim.set_particles(queue, &[[0.0, 0.75, 0.0]]).unwrap();
    let dispatch = DispatchScope::start().unwrap();
    assert!(
        sim.read_geometric_surface(&engine, &mut surface, Duration::from_secs(15))
            .is_err()
    );
    assert_eq!(dispatch.finish().encoded_dispatches, 0);
    let mut replacement = sim
        .prepare_geometric_surface(&engine, config, 1 << 20)
        .unwrap();
    let current = sim
        .read_geometric_surface(&engine, &mut replacement, Duration::from_secs(15))
        .unwrap();
    assert_eq!(current.counts.iter().sum::<u32>(), 1);
    assert_eq!(current.heights[8 * 16 + 8], 0.75);
    let dispatch = DispatchScope::start().unwrap();
    assert!(
        sim.read_geometric_surface(&engine, &mut replacement, Duration::ZERO)
            .is_err()
    );
    assert_eq!(dispatch.finish().encoded_dispatches, 0);
    sim.set_particles(queue, &[[0.0, 0.5, 0.0]]).unwrap();
    assert_eq!(
        sim.read_geometric_surface(&engine, &mut replacement, Duration::from_secs(15))
            .unwrap()
            .heights[8 * 16 + 8],
        0.5
    );
    let mut other = MpmSimulation::new(&engine, cfg).unwrap();
    other.set_particles(queue, &[[0.0, 0.5, 0.0]]).unwrap();
    assert!(
        other
            .read_geometric_surface(&engine, &mut replacement, Duration::from_secs(15))
            .is_err()
    );
}
