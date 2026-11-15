#![cfg(feature = "resource-counters")]

use molla_compute::Engine;
use molla_core::gpu_metrics::{DeviceScope, DispatchScope, TransferScope};
use molla_kernels::{BarrierGpuObstacle, BarrierGpuParams, BarrierGpuSpring, BarrierNewtonGpu};
use std::time::Instant;

#[test]
#[ignore = "requires hardware GPU; run via make bench-ppf-gpu"]
fn barrier_steady_state_resource_profile() {
    let engine = Engine::new_gpu().unwrap();
    let info = engine.adapter_info().unwrap();
    assert!(!info.is_software, "hardware measurement required: {info:?}");
    eprintln!("PPF adapter={info:?} profile=release warmup=48 samples=32 repeats=3 dt=1/240");
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for side in [8, 24] {
        let spacing = 0.21 / (side - 1) as f32;
        let positions: Vec<[f32; 3]> = (0..side * side)
            .map(|i| {
                [
                    (i % side) as f32 * spacing - 0.105,
                    0.15,
                    (i / side) as f32 * spacing - 0.105,
                ]
            })
            .collect();
        let mut triangles = Vec::new();
        for y in 0..side - 1 {
            for x in 0..side - 1 {
                let a = (y * side + x) as u32;
                let b = a + 1;
                let c = a + side as u32;
                triangles.extend([[a, b, c], [b, c + 1, c]]);
            }
        }
        let mut edges = std::collections::BTreeSet::new();
        for t in &triangles {
            for (a, b) in [(t[0], t[1]), (t[1], t[2]), (t[2], t[0])] {
                edges.insert((a.min(b), a.max(b)));
            }
        }
        let springs: Vec<_> = edges
            .iter()
            .map(|&(a, b)| {
                let pa = positions[a as usize];
                let pb = positions[b as usize];
                let rest = (0..3).map(|i| (pa[i] - pb[i]).powi(2)).sum::<f32>().sqrt();
                BarrierGpuSpring {
                    a,
                    b,
                    rest,
                    k: 400.0,
                }
            })
            .collect();
        for repeat in 0..3 {
            let setup = Instant::now();
            let allocation_scope = DeviceScope::start().unwrap();
            let mut gpu = BarrierNewtonGpu::new_with_mesh(
                device,
                queue,
                &positions,
                &vec![0.02; positions.len()],
                &springs,
                &[
                    BarrierGpuObstacle::HalfSpace {
                        n: [0.0, 1.0, 0.0],
                        offset: 0.0,
                    },
                    BarrierGpuObstacle::Sphere {
                        center: [0.0; 3],
                        radius: 0.06,
                    },
                ],
                &triangles,
                &[],
                BarrierGpuParams::default(),
            )
            .unwrap();
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let setup_ms = setup.elapsed().as_secs_f64() * 1000.0;
            let setup_allocations = allocation_scope.finish();
            assert!(!setup_allocations.invalid);
            for _ in 0..48 {
                assert!(gpu.step(1.0 / 240.0).unwrap().min_gap > 0.0);
            }
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let allocations = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            let dispatches = DispatchScope::start().unwrap();
            let mut ms = Vec::new();
            let mut iters = 0;
            let mut caps = 0;
            for _ in 0..32 {
                let start = Instant::now();
                let stats = gpu.step(1.0 / 240.0).unwrap();
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                ms.push(start.elapsed().as_secs_f64() * 1000.0);
                assert!(stats.min_gap > 0.0 && stats.energy.is_finite());
                iters += stats.newton_iters;
                caps += usize::from(stats.hit_iter_cap);
            }
            let allocations = allocations.finish();
            let transfers = transfers.finish();
            let dispatches = dispatches.finish();
            assert!(!allocations.invalid && !transfers.overflowed && !dispatches.overflowed);
            assert_eq!(
                allocations.observations, 0,
                "step must reuse scalar staging"
            );
            assert!(dispatches.encoded_dispatches > 0 && transfers.readback_bytes > 0);
            ms.sort_by(f64::total_cmp);
            let counters = device.get_internal_counters().hal;
            eprintln!(
                "PPF vertices={} triangles={} springs={} repeat={repeat} setup_ms={setup_ms:.3} setup_buffers={} setup_peak_bytes={} p50_ms={:.3} p95_ms={:.3} max_ms={:.3} buffer_allocations={} resident_buffers={} resident_bytes={} readback_bytes={} upload_bytes={} dispatches={} newton_iters={iters} capped_steps={caps}",
                positions.len(),
                triangles.len(),
                springs.len(),
                setup_allocations.observations,
                setup_allocations.peak_buffer_bytes,
                ms[16],
                ms[30],
                ms[31],
                allocations.observations,
                counters.buffers.read(),
                counters.buffer_memory.read(),
                transfers.readback_bytes,
                transfers.upload_bytes,
                dispatches.encoded_dispatches
            );
            for p in gpu.read_positions().unwrap() {
                assert!(p.iter().all(|v| v.is_finite()));
                assert!(p[1] > 0.0);
                assert!(p.iter().map(|v| v * v).sum::<f32>().sqrt() > 0.06);
            }
        }
    }
}
