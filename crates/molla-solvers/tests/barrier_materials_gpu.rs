#![cfg(feature = "resource-counters")]

use molla_compute::Engine;
use molla_core::gpu_metrics::DeviceScope;
use molla_kernels::{BarrierGpuObstacle, BarrierGpuParams, BarrierNewtonGpu};
use molla_math::{
    Vec3,
    constitutive::{ElasticMaterial, ElasticModel},
};
use molla_sim::ElasticElement;
use molla_solvers::barrier_contact::{BarrierObstacle, BarrierParams, BarrierShell};

fn packed(x: &[Vec3]) -> Vec<[f32; 3]> {
    x.iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect()
}

fn unpacked(x: Vec<[f32; 3]>) -> Vec<Vec3> {
    x.into_iter()
        .map(|p| Vec3::from_array(p.map(f64::from)))
        .collect()
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-material-solver-gpu"]
fn spring_free_membranes_and_solids_follow_cpu_material_trajectories() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for model in [
        ElasticModel::Corotated,
        ElasticModel::QuarticStretch,
        ElasticModel::StableNeoHookean,
        ElasticModel::BaraffWitkin,
    ] {
        for n in [2, 3] {
            if n == 3 && model == ElasticModel::BaraffWitkin {
                continue;
            }
            let material = ElasticMaterial {
                model,
                mu: 1000.0,
                lambda: 1000.0,
            };
            let (rest, pins, triangles, element) = if n == 2 {
                let rest = vec![Vec3::Y, Vec3::Y + Vec3::X, Vec3::Y + Vec3::Z];
                let e = ElasticElement::membrane(&rest, [0, 1, 2], 0.05, material).unwrap();
                (rest, vec![0, 2], vec![[0, 1, 2]], e)
            } else {
                let rest = vec![Vec3::Y, Vec3::Y + Vec3::X, Vec3::Y * 2.0, Vec3::Y + Vec3::Z];
                let e = ElasticElement::tetrahedron(&rest, [0, 1, 2, 3], material).unwrap();
                (rest, vec![0, 1, 3], vec![], e)
            };
            let mut x = rest.clone();
            x[if n == 2 { 1 } else { 2 }] += Vec3::new(0.3, 0.0, 0.2);
            let initial_energy = element.energy(&x).unwrap();
            let mut cpu = BarrierShell::new(
                x.clone(),
                vec![0.05; x.len()],
                vec![],
                vec![BarrierObstacle::HalfSpace {
                    n: Vec3::Y,
                    offset: 0.0,
                }],
                BarrierParams {
                    gravity: Vec3::ZERO,
                    max_newton_iters: 40,
                    ..Default::default()
                },
            )
            .unwrap()
            .with_triangles(triangles.clone())
            .with_elastic_elements(vec![element.clone()])
            .unwrap();
            cpu.pin_vertices(&pins.iter().map(|i| *i as usize).collect::<Vec<_>>());
            let mut gpu = BarrierNewtonGpu::new_with_elastic_elements(
                device,
                queue,
                &packed(&x),
                &vec![0.05; x.len()],
                &[],
                &[BarrierGpuObstacle::HalfSpace {
                    n: [0.0, 1.0, 0.0],
                    offset: 0.0,
                }],
                &triangles,
                &[],
                &pins,
                &[element.clone()],
                BarrierGpuParams {
                    gravity: [0.0; 3],
                    max_newton_iters: 40,
                    ..Default::default()
                },
            )
            .unwrap();
            let mut worst = 0.0f64;
            for _ in 0..60 {
                cpu.step(0.02).unwrap();
                let stats = gpu.step(0.02).unwrap();
                assert!(stats.min_gap > 0.0);
                let current = unpacked(gpu.read_positions().unwrap());
                element.validate(&current).unwrap();
                for (i, (a, b)) in current.iter().zip(&cpu.positions).enumerate() {
                    worst = worst.max(a.distance(*b));
                    if pins.contains(&(i as u32)) {
                        assert_eq!(*a, rest[i]);
                    }
                }
            }
            let current = unpacked(gpu.read_positions().unwrap());
            let final_energy = element.energy(&current).unwrap();
            eprintln!(
                "material trajectory {model:?} n={n} worst_error={worst:.6e} energy={initial_energy:.6e}->{final_energy:.6e}"
            );
            assert!(final_energy < 0.7 * initial_energy);
            assert!(worst < 0.001, "{model:?} n={n} trajectory error={worst}");
        }
    }
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-material-solver-gpu"]
fn solid_material_drop_keeps_contact_and_positive_orientation() {
    let engine = Engine::new_gpu().unwrap();
    let rest = vec![
        Vec3::new(0.0, 0.2, 0.0),
        Vec3::new(0.2, 0.2, 0.0),
        Vec3::new(0.0, 0.4, 0.0),
        Vec3::new(0.0, 0.2, 0.2),
    ];
    let material = ElasticMaterial {
        model: ElasticModel::StableNeoHookean,
        mu: 1.0e5,
        lambda: 1.0e5,
    };
    let element = ElasticElement::tetrahedron(&rest, [0, 1, 2, 3], material).unwrap();
    let triangles = [[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    let mut solver = BarrierNewtonGpu::new_with_elastic_elements(
        engine.device().unwrap(),
        engine.queue().unwrap(),
        &packed(&rest),
        &[0.05; 4],
        &[],
        &[BarrierGpuObstacle::HalfSpace {
            n: [0.0, 1.0, 0.0],
            offset: 0.0,
        }],
        &triangles,
        &[],
        &[],
        &[element.clone()],
        BarrierGpuParams::default(),
    )
    .unwrap();
    solver.set_velocities(&[[0.0, -1.0, 0.0]; 4]).unwrap();
    let mut contact = false;
    for _ in 0..160 {
        solver.step(0.005).unwrap();
        let x = unpacked(solver.read_positions().unwrap());
        assert!(x.iter().all(|p| p.y > 0.0));
        assert!(element.deformation(&x).unwrap().determinant() > 0.0);
        contact |= x.iter().any(|p| p.y < 0.002);
    }
    assert!(contact, "drop must actually reach the contact band");
}

#[test]
#[ignore = "requires GPU; run via make test-ppf-material-solver-gpu"]
fn material_overflow_is_transactional_and_invalid_setup_allocates_nothing() {
    let engine = Engine::new_gpu().unwrap();
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    let rest = vec![Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::Z];
    let material = ElasticMaterial {
        model: ElasticModel::StableNeoHookean,
        mu: 1.0e37,
        lambda: 1.0e37,
    };
    let element = ElasticElement::tetrahedron(&rest, [0, 1, 2, 3], material).unwrap();
    let allocations = DeviceScope::start().unwrap();
    assert!(
        BarrierNewtonGpu::new_with_elastic_elements(
            device,
            queue,
            &packed(&rest[..3]),
            &[1.0; 3],
            &[],
            &[],
            &[],
            &[],
            &[],
            &[element.clone()],
            BarrierGpuParams::default(),
        )
        .is_err()
    );
    assert_eq!(allocations.finish().observations, 0);
    let mut solver = BarrierNewtonGpu::new_with_elastic_elements(
        device,
        queue,
        &packed(&rest),
        &[1.0; 4],
        &[],
        &[],
        &[],
        &[],
        &[],
        &[element],
        BarrierGpuParams {
            gravity: [0.0; 3],
            ..Default::default()
        },
    )
    .unwrap();
    let before = solver.read_positions().unwrap();
    let velocity = solver.read_velocities().unwrap();
    assert!(solver.step(100.0).is_err());
    assert_eq!(solver.read_positions().unwrap(), before);
    assert_eq!(solver.read_velocities().unwrap(), velocity);
    solver.step(0.001).unwrap();
    assert_eq!(solver.read_positions().unwrap(), before);
    assert_eq!(solver.read_velocities().unwrap(), velocity);
}

#[test]
#[ignore = "requires release GPU benchmark; run via make bench-ppf-material-solver"]
fn resident_material_solver_resource_profile() {
    use molla_core::gpu_metrics::{DispatchScope, TransferScope};
    use std::time::Instant;
    let engine = Engine::new_gpu().unwrap();
    assert!(!engine.adapter_info().unwrap().is_software);
    eprintln!(
        "material solver adapter={:?} warmup=48 samples=16 repeats=3 dt=1/240",
        engine.adapter_info().unwrap()
    );
    let device = engine.device().unwrap();
    let queue = engine.queue().unwrap();
    for side in [8usize, 16] {
        let spacing = 0.21 / (side - 1) as f64;
        let rest: Vec<_> = (0..side * side)
            .map(|i| {
                Vec3::new(
                    (i % side) as f64 * spacing - 0.105,
                    0.15,
                    (i / side) as f64 * spacing - 0.105,
                )
            })
            .collect();
        let mut triangles = Vec::new();
        for row in 0..side - 1 {
            for col in 0..side - 1 {
                let a = (row * side + col) as u32;
                let b = a + 1;
                let c = a + side as u32;
                triangles.extend([[a, b, c], [b, c + 1, c]]);
            }
        }
        let material = ElasticMaterial {
            model: ElasticModel::StableNeoHookean,
            mu: 1.0e4,
            lambda: 1.0e4,
        };
        let elements: Vec<_> = triangles
            .iter()
            .map(|t| ElasticElement::membrane(&rest, *t, 0.01, material).unwrap())
            .collect();
        let positions: Vec<_> = rest
            .iter()
            .map(|p| Vec3::new(p.x * 1.05, p.y, p.z * 1.02))
            .collect();
        for repeat in 0..3 {
            let start = Instant::now();
            let mut solver = BarrierNewtonGpu::new_with_elastic_elements(
                device,
                queue,
                &packed(&positions),
                &vec![0.02; positions.len()],
                &[],
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
                &[],
                &elements,
                BarrierGpuParams::default(),
            )
            .unwrap();
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
            let setup_ms = start.elapsed().as_secs_f64() * 1000.0;
            for _ in 0..48 {
                solver.step(1.0 / 240.0).unwrap();
            }
            let allocations = DeviceScope::start().unwrap();
            let transfers = TransferScope::start().unwrap();
            let dispatches = DispatchScope::start().unwrap();
            let mut ms = Vec::new();
            let mut newton = 0;
            let mut capped = 0;
            for _ in 0..16 {
                let start = Instant::now();
                let stats = solver.step(1.0 / 240.0).unwrap();
                device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
                ms.push(start.elapsed().as_secs_f64() * 1000.0);
                assert!(stats.min_gap > 0.0 && stats.energy.is_finite());
                newton += stats.newton_iters;
                capped += usize::from(stats.hit_iter_cap);
            }
            let a = allocations.finish();
            let t = transfers.finish();
            let d = dispatches.finish();
            assert!(!a.invalid && !t.overflowed && !d.overflowed);
            assert_eq!(a.observations, 0);
            assert_eq!(t.readback_bytes % 64, 0);
            ms.sort_by(f64::total_cmp);
            eprintln!(
                "material solver vertices={} elements={} repeat={repeat} setup_ms={setup_ms:.3} p50_ms={:.3} p95_ms={:.3} resident_bytes={} allocations={} uploads={} readbacks={} dispatches={} newton={newton} capped={capped}",
                rest.len(),
                elements.len(),
                ms[8],
                ms[15],
                device.get_internal_counters().hal.buffer_memory.read(),
                a.observations,
                t.upload_bytes,
                t.readback_bytes,
                d.encoded_dispatches
            );
            let current = unpacked(solver.read_positions().unwrap());
            assert!(current.iter().all(|p| p.y > 0.0 && p.length() > 0.06));
            for e in &elements {
                e.validate(&current).unwrap();
            }
        }
    }
}
